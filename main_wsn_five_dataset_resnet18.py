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

from utils import safe_save

from networks.subnet import SubnetLinear, SubnetConv2d
from networks.resnet18 import SubnetBasicBlock
from networks.utils import *

from networks.alexnet import SubnetAlexNet_norm as AlexNet
from networks.resnet18 import SubnetResNet18 as ResNet18

import wandb

def train(args, model, device, x,y, optimizer, criterion, task_id_nominal, consolidated_masks):

    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)

    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if ((i + args.batch_size_train) <= len(r)):
            b=r[i:i+args.batch_size_train]
        else:
            b=r[i:]

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

                # Determine wheter it's an output head or not
                key_split = key.split('.')
                if 'last' in key_split or len(key_split) == 2:
                    if 'last' in key_split:
                        module_attr = key_split[-1]
                        task_num = int(key_split[-2])
                        module_name = '.'.join(key_split[:-2])

                    else:
                        module_attr = key_split[1]
                        module_name = key_split[0]

                    # Zero-out gradients
                    if (hasattr(getattr(model, module_name), module_attr)):
                        if (getattr(getattr(model, module_name), module_attr) is not None):
                            getattr(getattr(model, module_name), module_attr).grad[consolidated_masks[key] == 1] = 0

                else:
                    module_attr = key_split[-1]

                    # Zero-out gradients
                    curr_module = getattr(getattr(model, key_split[0])[int(key_split[1])], key_split[2])
                    if hasattr(curr_module, module_attr):
                        if getattr(curr_module, module_attr) is not None:
                            getattr(curr_module, module_attr).grad[consolidated_masks[key] == 1] = 0

        optimizer.step()

def test(args, model, device, x, y, criterion, task_id_nominal, curr_task_masks=None, mode="test", epoch=1):
    model.eval()
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
            output = model(data, task_id_nominal, mask=curr_task_masks, mode=mode, epoch=epoch)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)

            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc

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

    ## Load Five_datasets DATASET
    from dataloader import five_datasets
    data,taskcla,inputsize=five_datasets.get(seed=args.seed, pc_valid=args.pc_valid)

    acc_matrix=np.zeros((5,5))
    sparsity_matrix = []
    sparsity_per_task = {}
    criterion = torch.nn.CrossEntropyLoss()

    # Model Instantiation
    if args.model == 'alexnet':
        model = AlexNet(taskcla, args.sparsity).to(device)
    elif args.model == 'resnet18':
        model = ResNet18(taskcla, nf=20, sparsity=args.sparsity).to(device) # base filters: 20
    print ('Model parameters ---')
    for k_t, (m, param) in enumerate(model.named_parameters()):
        print (k_t,m,param.shape)
    print ('-'*40)

    task_id = 0
    task_list = []
    per_task_masks, consolidated_masks, prime_masks = {}, {}, {}
    for k, ncla in taskcla:

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

        for epoch in range(1, args.n_epochs+1):
            # Train
            clock0 = time.time()
            train(args, model, device, xtrain, ytrain, optimizer, criterion, task_id, consolidated_masks)
            clock1 = time.time()
            tr_loss,tr_acc = test(args, model, device, xtrain, ytrain,  criterion, task_id, curr_task_masks=None, mode="valid")
            print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss,tr_acc, 1000*(clock1-clock0)),end='')
            # Validate
            valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, task_id, curr_task_masks=None, mode="valid", epoch=epoch)
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

        # Restore best model
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
        test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, task_id, curr_task_masks=per_task_masks[task_id], mode="test")
        print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))

        log_dict = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'task': task_id,
        }
        wandb.log(log_dict)

        # save accuracy
        for jj in np.array(task_list):
            xtest = data[jj]['test']['x']
            ytest = data[jj]['test']['y']

            if jj <= task_id:
                _, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion, jj, curr_task_masks=per_task_masks[jj], mode="test")
            else:
                _, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion, jj, curr_task_masks=per_task_masks[task_id], mode="test")

        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(args.n_tasks):
                print('{:5.1f} '.format(acc_matrix[i_a,j_a]),end='')
            print()

        # update task id
        task_id +=1

    # Save
    safe_save(exp_dir + "/csnb_five_data/" + args.name + ".acc", acc_matrix)
    safe_save(exp_dir + "/csnb_five_data/" + args.name + ".cap", sparsity_matrix)
    safe_save(exp_dir + "/csnb_five_data/" + args.name + ".spar", sparsity_per_task)
    safe_save(exp_dir + "/csnb_five_data/" + args.name + ".pertask", per_task_masks)
    safe_save(exp_dir + "/csnb_five_data/" + args.name + ".fullmask", consolidated_masks)
    torch.save(model.to("cpu"), exp_dir + "/csnb_five_data/"  + args.name + ".ptmodel")

    model = model.to(device)
    # Test one more time
    test_acc_matrix=np.zeros((5,5))
    sparsity_matrix = []
    sparsity_per_task = {}
    criterion = torch.nn.CrossEntropyLoss()

    task_list = []
    # per_task_masks, consolidated_masks = {}, {}
    task_id=0
    for k, ncla in taskcla:
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
        test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, task_id, curr_task_masks=per_task_masks[task_id], mode="test")
        print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))

        # save accuracy
        for jj in np.array(task_list)[0:task_id+1]:
            if jj != task_id:
                test_acc_matrix[task_id, jj] = acc_matrix[task_id-1, jj]
            else:
                xtest = data[jj]['test']['x']
                ytest = data[jj]['test']['y']

                _, test_acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion, jj, curr_task_masks=per_task_masks[jj], mode="test")

        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(args.n_tasks):
                print('{:5.1f} '.format(acc_matrix[i_a,j_a]),end='')
            print()

        # update task id
        task_id +=1

    print('-'*50)
    safe_save(exp_dir + "/csnb_five_data/" + args.name + ".test_acc", test_acc_matrix)
    # Simulation Results
    print ('Task Order : {}'.format(np.array(task_list)))
    print ('Diagonal Final Avg Accuracy: {:5.2f}%'.format( np.mean([acc_matrix[i,i] for i in range(len(taskcla))] )))
    print ('Final Avg accuracy: {:5.2f}%'.format( np.mean(acc_matrix[len(taskcla) - 1])))
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1])
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
    parser.add_argument('--batch_size_train', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=200, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--n_tasks', type=int, default=5, metavar='S',
                        help='number of tasks (default: 5)')
    parser.add_argument('--pc_valid',default=0.10,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--optim', type=str, default="adam", metavar='OPTIM',
                        help='optimizer choice')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-3, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    # CUDA parameters
    parser.add_argument('--gpu', type=str, default="0", metavar='GPU',
                        help="GPU ID for single GPU training")
    # CSNB parameters
    parser.add_argument('--sparsity', type=float, default=0.5, metavar='SPARSITY',
                        help="Target current sparsity for each layer")

    # Model parameters
    parser.add_argument('--model', type=str, default="alexnet", metavar='MODEL',
                        help="Models to be incorporated for the experiment")

    parser.add_argument("--dataset",
                        default='five_data',
                        type=str,
                        help="Dataset to train and test on.")

    parser.add_argument('--name', type=str, default='hard')
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
    name = "{}_SEED_{}_LR_{}_SPARSITY_{:0.2f}_{}_soft{}_grad{}".format(
        args.dataset,
        args.seed,
        args.lr,
        1 - args.sparsity,
        args.name, args.soft, args.soft_grad)
    args.name = name

    wandb.init(project='wsn_{}'.format(args.dataset),
               entity='haeyong',
               name=name,
               config=args)


    main(args)



