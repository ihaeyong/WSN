import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import os
import os.path
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random

import argparse,time
import math
from copy import deepcopy
from itertools import combinations, permutations

from networks.subnet import SubnetLinear, SubnetConv2d
from networks.resnet18 import SubnetBasicBlock
from networks.utils import *
from utils import *

from networks.lenet import SubnetLeNet as LeNet
from utils_rle import compress_ndarray, decompress_ndarray, comp_decomp_mask
from utils_huffman import comp_decomp_mask_huffman

import importlib

import wandb

def train(args, model, device, x, y, optimizer, criterion, task_id_nominal, consolidated_masks):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()
        output = model(data, task_id_nominal, mask=None, mode="train")
        loss = criterion(output, target)
        loss.backward()

        # Continual Subnet no backprop
        curr_head_keys = ["last.{}.weight".format(task_id_nominal), "last.{}.bias".format(task_id_nominal)]
        if consolidated_masks is not None and consolidated_masks != {}: # Only do this for tasks 1 and beyond
            # if args.use_continual_masks:
            for key in consolidated_masks.keys():

                # Skip if not task head is not for curent task
                if 'last' in key:
                    if key not in curr_head_keys:
                        continue

                # Determine whether it's an output head or not
                if (len(key.split('.')) == 3):  # e.g. last.1.weight
                    module_name, task_num, module_attr = key.split('.')
                    # curr_module = getattr(model, module_name)[int(task_num)]
                else: # e.g. fc1.weight
                    module_name, module_attr = key.split('.')
                    # curr_module = getattr(model, module_name)

                # Zero-out gradients
                if (hasattr(getattr(model, module_name), module_attr)):
                    if (getattr(getattr(model, module_name), module_attr) is not None):
                        getattr(getattr(model, module_name), module_attr).grad[consolidated_masks[key] == 1] = 0

        optimizer.step()

def test(args, model, device, x, y, criterion, task_id_nominal, curr_task_masks=None, mode="test", comp_flag=False):
    model.eval()

    comp_ratio = None
    total_loss = 0
    total_num = 0
    correct = 0
    r=np.arange(x.size(0))
    r=torch.LongTensor(r).to(device)

    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if ((i + args.batch_size_test) <= len(r)):
                b=r[i:i+args.batch_size_test]
            else: b=r[i:]

            data = x[b]
            data, target = data.to(device), y[b].to(device)
            if comp_flag:
                if args.encoding == 'huffman':
                    dec_task_mask, comp_ratio = comp_decomp_mask_huffman(curr_task_masks, task_id_nominal, device)
                else:
                    dec_task_mask, comp_ratio = comp_decomp_mask(curr_task_masks, task_id_nominal, device)
                output = model(data, task_id_nominal, mask=dec_task_mask, mode=mode)
            else:
                output = model(data, task_id_nominal, mask=curr_task_masks, mode=mode)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)

            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc, comp_ratio

def main(args):
    tstart=time.time()
    ## Device Setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    exp_dir = "results_{}".format(args.dataset)

    # Choose any task order - ref {yoon et al. ICLR 2020}
    task_order = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                  np.array([15, 12, 5, 9, 7, 16, 18, 17, 1, 0, 3, 8, 11, 14, 10, 6, 2, 4, 13, 19]),
                  np.array([17, 1, 19, 18, 12, 7, 6, 0, 11, 15, 10, 5, 13, 3, 9, 16, 4, 14, 2, 8]),
                  np.array([11, 9, 6, 5, 12, 4, 0, 10, 13, 7, 14, 3, 15, 16, 8, 1, 2, 19, 18, 17]),
                  np.array([6, 14, 0, 11, 12, 17, 13, 4, 9, 1, 7, 19, 8, 10, 3, 15, 18, 5, 2, 16])]

    ## Load CIFAR100_SUPERCLASS DATASET
    if False:
        from dataloader import cifar100_superclass as data_loader
        data, taskcla = data_loader.cifar100_superclass_python(task_order[args.t_order], group=5, validation=True)
        test_data,_   = data_loader.cifar100_superclass_python(task_order[args.t_order], group=5)
        print (taskcla)
    else:
        dataloader = importlib.import_module('dataloader.' + args.dataset)
        ids = task_order[args.t_order]
        data, output_info, input_size, n_tasks, n_outputs = dataloader.get(data_path=args.data_path, task_order=ids,
                                                                           seed=args.seed, pc_valid=args.pc_valid)
        args.n_tasks = n_tasks
        args.samples_per_task = int(data[0]['train']['y'].shape[0] / (1.0 - args.pc_valid))
        print('Task info =', output_info)
        print('Input size =', input_size, '\nOutput number=', n_outputs, '\nTotal task=', n_tasks)
        print('Task order =', ids)
        print('-' * 100)

        tasks_ = [t for t  in ids]
        n_outputs_ = [n_outputs] * n_tasks
        taskcla = [(t,n) for t,n in zip(tasks_, n_outputs_)]

    acc_matrix=np.zeros((n_tasks,n_tasks))
    sparsity_matrix = []
    sparsity_per_task = {}
    criterion = torch.nn.CrossEntropyLoss()

    # Model Instantiation
    if args.model == 'alexnet':
        model = AlexNet(taskcla, args.sparsity).to(device)
    elif args.model == 'resnet18':
        model = ResNet18(taskcla, nf=20, sparsity=args.sparsity).to(device) # base filters: 20
    elif args.model == 'lenet':
        model = LeNet(taskcla, args.sparsity).to(device)

    print ('Model parameters ---')
    for k_t, (m, param) in enumerate(model.named_parameters()):
        print (k_t,m,param.shape)
    print ('-'*40)

    task_id = 0
    task_list = []
    per_task_masks, consolidated_masks, per_prime_masks, prime_masks = {}, {}, {}, {}
    for k,ncla in taskcla:
        print('*'*100)
        print('Task {:2d} ({:s})'.format(k,data[k]['name']))
        print('*'*100)
        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest =data[k]['test']['x']
        ytest =data[k]['test']['y']
        task_list.append(k)

        lr = args.lr
        best_loss=np.inf
        print ('-'*40)
        print ('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        print ('-'*40)

        best_model=get_model(model)
        if args.optim == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif args.optim == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            raise Exception("[ERROR] The optimizer " + str(args.optim) + " is not supported!")

        # reinitialized weight score
        #model.init_masks(task_id=k)

        for epoch in range(1, args.n_epochs+1):
            # Train
            clock0 = time.time()
            train(args, model, device, xtrain, ytrain, optimizer, criterion, task_id, consolidated_masks)
            clock1 = time.time()
            tr_loss,tr_acc, tr_comp_ratio = test(args, model, device, xtrain, ytrain,  criterion, task_id, curr_task_masks=None, mode="valid")
            print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss,tr_acc, 1000*(clock1-clock0)),end='')
            # Validate
            valid_loss,valid_acc,vr_comp_ratio = test(args, model, device, xvalid, yvalid,  criterion, task_id, curr_task_masks=None, mode="valid")
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=get_model(model)
                patience=args.lr_patience
                print(' *',end='')
            else:
                patience-=1
                if patience<=0:
                    lr/=args.lr_factor
                    print(' lr={:.1e}'.format(lr),end='')
                    if lr<args.lr_min:
                        print()
                        break
                    patience=args.lr_patience
                    adjust_learning_rate(optimizer, epoch, args)
            print()

        # Restore the best model
        set_model_(model,best_model)

        # Save the per-task-dependent masks
        per_task_masks[task_id] = model.get_masks(task_id)

        # Consolidate task masks to keep track of parameters to-update or not
        curr_head_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        if task_id == 0:
            consolidated_masks = deepcopy(per_task_masks[task_id])
        else:
            for key in per_task_masks[task_id].keys():
                # Skip output head from other tasks
                # Also don't consolidate output head mask after training on new tasks; continue
                if "last" in key:
                    if key in curr_head_keys:
                        consolidated_masks[key] = deepcopy(per_task_masks[task_id][key])
                    continue
                # Or operation on sparsity
                if consolidated_masks[key] is not None and per_task_masks[task_id][key] is not None:
                    consolidated_masks[key] = 1 - ((1 - consolidated_masks[key]) * (1 - per_task_masks[task_id][key]))

        # Print Sparsity
        sparsity_per_layer = print_sparsity(consolidated_masks)
        all_sparsity = global_sparsity(consolidated_masks)
        print("Global Sparsity: {}".format(all_sparsity))
        sparsity_matrix.append(all_sparsity)
        sparsity_per_task[task_id] = sparsity_per_layer

        # Test
        print ('-'*40)
        test_loss, test_acc, test_comp_ratio = test(args, model, device, xtest, ytest,  criterion, task_id, curr_task_masks=per_task_masks[task_id], mode="test")
        print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))

        log_dict = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'task': task_id,
        }
        wandb.log(log_dict)

        # save accuracy
        for jj in range(n_tasks):

            xtest =data[jj]['test']['x']
            ytest =data[jj]['test']['y']

            if jj <= task_id:
                _, acc_matrix[task_id,jj], comp_ratio = test(args, model, device, xtest, ytest,criterion, jj, curr_task_masks=per_task_masks[jj], mode="test")
            else:
                _, acc_matrix[task_id,jj], comp_ratio = test(args, model, device, xtest, ytest,criterion, jj, curr_task_masks=per_task_masks[task_id], mode="test")

        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(n_tasks):
                print('{:5.1f} '.format(acc_matrix[i_a, j_a]),end='')
            print()

        # update task id
        task_id +=1

    # Simulation Results
    safe_save(exp_dir + "/csnb_cifar100_100/" + args.name + ".acc", acc_matrix)
    safe_save(exp_dir + "/csnb_cifar100_100/" + args.name + ".cap", sparsity_matrix)
    safe_save(exp_dir + "/csnb_cifar100_100/" + args.name + ".spar", sparsity_per_task)
    safe_save(exp_dir + "/csnb_cifar100_100/" + args.name + ".pertask", per_task_masks)
    safe_save(exp_dir + "/csnb_cifar100_100/" + args.name + ".fullmask", consolidated_masks)
    torch.save(model.to("cpu"), exp_dir + "/csnb_cifar100_100/" + args.name + ".ptmodel")

    print ('Task Order : {}'.format(task_order[args.t_order]))
    print ('Diagonal Final Avg Accuracy: {:5.2f}%'.format( np.mean([acc_matrix[i,i] for i in range(len(taskcla))] )))
    print ('Final Avg accuracy: {:5.2f}%'.format( np.mean(acc_matrix[len(taskcla) - 1])))
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1])
    print ('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    print('-'*50)

    model = model.to(device)
    # Test one more time
    test_acc_matrix=np.zeros((args.nperm, args.nperm))
    sparsity_matrix = []
    mask_comp_ratio = []
    sparsity_per_task = {}
    criterion = torch.nn.CrossEntropyLoss()

    task_list = []
    task_id=0
    for k,ncla in taskcla:

        print('*'*100)
        print('Task {:2d} ({:s})'.format(k,data[k]['name']))
        print('*'*100)
        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest =data[k]['test']['x']
        ytest =data[k]['test']['y']
        task_list.append(k)
        # Test
        print ('-'*40)

        test_loss, test_acc, comp_ratio = test(args, model, device, xtest, ytest,  criterion, task_id, curr_task_masks=per_task_masks[task_id], mode="test")
        print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))

        # save accuracy
        for jj in range(n_tasks):
            if jj != task_id:
                test_acc_matrix[task_id, jj] = acc_matrix[task_id, jj]
            else:
                xtest = data[jj]['test']['x']
                ytest = data[jj]['test']['y']

                _, test_acc_matrix[task_id,jj],comp_ratio = test(args, model, device, xtest, ytest,criterion, jj, curr_task_masks=per_task_masks[jj], mode="test", comp_flag=False)

        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(n_tasks):
                print('{:5.1f} '.format(test_acc_matrix[i_a,j_a]),end='')
            print()

        # update task id
        task_id +=1

    print('-'*50)

    # Print Sparsity
    safe_save(exp_dir + "/csnb_cifar100_100/" + args.name + ".comp_ratio", np.array(mask_comp_ratio))
    safe_save(exp_dir + "/csnb_cifar100_100/" + args.name + ".test_acc", test_acc_matrix)
    sparsity_per_layer = print_sparsity(consolidated_masks)
    all_sparsity = global_sparsity(consolidated_masks)
    print("Global Sparsity: {}".format(all_sparsity))
    print("Bit Mask Capacity: {}%".format(np.sum(mask_comp_ratio)))

    # Simulation Result
    print ('Task Order : {}'.format(np.array(task_list)))
    print ('Diagonal Final Avg Accuracy: {:5.2f}%'.format( np.mean([test_acc_matrix[i,i] for i in range(len(taskcla))] )))
    test_avg_acc = np.mean(acc_matrix[len(taskcla) - 1])
    print ('Final Avg accuracy: {:5.2f}%'.format( np.mean(acc_matrix[len(taskcla) - 1])))

    bwt=np.mean((test_acc_matrix[-1]-np.diag(acc_matrix))[:-1])
    print ('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))

    log_dict = {
        'test_avg_acc': test_avg_acc,
        'test_avg_bwt': bwt
    }
    wandb.log(log_dict)

    print('-'*50)
    print(args)


if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=50, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid',default=0.1,type=float,
                        help='fraction of training data used for validation')
    parser.add_argument('--t_order', type=int, default=0, metavar='TOD',
                        help='random seed (default: 0)')

    # Optimizer parameters
    parser.add_argument('--optim', type=str, default="sgd", metavar='OPTIM',
                        help='optimizer choice')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')

    # Model parameters
    parser.add_argument('--model', type=str, default="resnet18", metavar='MODEL',
                        help="Models to be incorporated for the experiment")

    # CUDA parameters
    parser.add_argument('--gpu', type=str, default="0", metavar='GPU',
                        help="GPU ID for single GPU training")
    # CSNB parameters
    parser.add_argument('--sparsity', type=float, default=0.5, metavar='SPARSITY',
                        help="Target current sparsity for each layer")

    # miniImagenet parameters
    parser.add_argument('--nperm', type=int, default=20, metavar='NPERM',
                        help='number of permutations/tasks')

    # data parameters
    parser.add_argument('--loader', type=str,
                        default='task_incremental_loader',
                        help='data loader to use')
    # increment
    parser.add_argument('--increment', type=int, default=5, metavar='S',
                        help='(default: 5)')

    parser.add_argument('--data_path', default='./data/', help='path where data is located')
    parser.add_argument("--dataset",
                        default='mnist_permutations',
                        type=str,
                        required=True,
                        choices=['mnist_permutations', 'cifar100', 'cifar100_superclass100', 'tinyimagenet', 'pmnist'],
                        help="Dataset to train and test on.")


    parser.add_argument('--samples_per_task', type=int, default=-1,
                        help='training samples per task (all if negative)')

    parser.add_argument("--workers", default=4, type=int, help="Number of workers preprocessing the data.")

    parser.add_argument("--glances", default=1, type=int,
                        help="# of times the model is allowed to train over a set of samples in the single pass setting")
    parser.add_argument("--class_order", default="random", type=str, choices=["random", "chrono", "old", "super"],
                        help="define classes order of increment ")

    # For cifar100
    parser.add_argument('--n_tasks', type=int, default=20,
                        help='total number of tasks, invalid for cifar100_superclass')
    parser.add_argument('--shuffle_task', default=False, action='store_true',
                        help='Invalid for cifar100_superclass')
    parser.add_argument('--encoding', type=str, default="huffman", metavar='',
                        help="")

    parser.add_argument('--name', type=str, default='baseline')
    parser.add_argument('--soft', type=float, default=0.0)
    parser.add_argument('--soft_grad', type=float, default=1.0)

    args = parser.parse_args()
    args.sparsity = 1 - args.sparsity

    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)

    # update name
    name = "{}_SEED_{}_LR_{}_SPARSITY_{:0.2f}_{}_soft{}_grad{}_{}".format(
        args.dataset,
        args.seed,
        args.lr,
        1 - args.sparsity,
        args.name, args.soft, args.soft_grad, args.t_order)
    args.name = name

    wandb.init(project='wsn_{}'.format(args.dataset),
               entity='haeyong',
               name=name,
               config=args)

    main(args)



