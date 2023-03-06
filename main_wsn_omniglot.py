import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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

from utils import *
from networks.utils import *

from networks.subnet import SubnetLinear, SubnetConv2d

import wandb

def train(args, model, device, x,y, optimizer,criterion, task_id_nominal, consolidated_masks):
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

def test(args, model, device, x, y, criterion, task_id_nominal, curr_task_masks=None, mode="test"):
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
            output = model(data, task_id_nominal, mask=curr_task_masks, mode=mode)
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

    ## Load PermutedMNIST
    from dataloader import omniglot_rotation
    data, taskcla, inputsize = omniglot_rotation.get(seed=args.seed, pc_valid=args.pc_valid)

    acc_matrix=np.zeros((100,100))
    sparsity_matrix = []
    criterion = torch.nn.CrossEntropyLoss()

    # Primes and Mod table of product of primes
    print ('Task primes ---')
    num_tasks = len(taskcla)
    num_primes = num_tasks // 10

    # Model Definition
    if args.model == "large":
        from networks.lenet import SubnetLeNet_Large as MLPNet
    elif args.model == "default":
        from networks.lenet import SubnetLeNet_Default as MLPNet
    else:
        raise Exception("[ERROR] The model " + str(args.model) + " is not supported!")

    # Model Instantiation
    model = MLPNet(taskcla, args.sparsity).to(device)
    print ('Model parameters ---')
    for k_t, (m, param) in enumerate(model.named_parameters()):
        print (k_t,m,param.shape)
    print ('-'*40)

    task_id = 0
    task_list = []
    sparsity_per_task = {}
    per_task_masks, consolidated_masks = {}, {}
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
            clock2=time.time()
            print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms | test time={:5.1f}ms'.format(epoch,\
                                                        tr_loss,tr_acc, 1000*(clock1-clock0), (clock2 - clock1)*1000 ), end='')
            # Validate
            valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, task_id, curr_task_masks=None, mode="valid")
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
    safe_save(exp_dir + "/omniglot/" + args.name + ".test_acc", acc_matrix)
    safe_save(exp_dir + "/omniglot/" + args.name + ".cap", sparsity_matrix)

    safe_save(exp_dir + "/omniglot/" + args.name + ".spar", sparsity_per_task)
    safe_save(exp_dir + "/omniglot/" + args.name + ".pertask", per_task_masks)
    safe_save(exp_dir + "/omniglot/" + args.name + ".fullmask", consolidated_masks)
    torch.save(model.to("cpu"), exp_dir + "/omniglot/" + args.name + ".ptmodel")

    print('-'*50)
    # Simulation Results 
    print ('Task Order : {}'.format(np.array(task_list)))
    print ('Diagonal Final Avg Accuracy: {:5.2f}%'.format( np.mean([acc_matrix[i,i] for i in range(len(taskcla))] ))) 
    print ('Final Avg accuracy: {:5.2f}%'.format( np.mean(acc_matrix[len(taskcla) - 1])))
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    print ('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    print('-'*50)


if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--batch_size_test', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=5, metavar='N',
                        help='number of training epochs/task (default: 5)')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')
    parser.add_argument('--pc_valid',default=0.1,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--optim', type=str, default="adam", metavar='OPTIM',
                        help='optimizer choice')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    # Architecture
    parser.add_argument('--n_hidden', type=int, default=256, metavar='NH',
                        help='number of hidden units in MLP (default: 100)')
    parser.add_argument('--n_outputs', type=int, default=10, metavar='NO',
                        help='number of output units in MLP (default: 10)')
    parser.add_argument('--n_tasks', type=int, default=100, metavar='NT',
                        help='number of tasks (default: 10)')
    # CUDA parameters
    parser.add_argument('--gpu', type=str, default="0", metavar='GPU',
                        help="GPU ID for single GPU training")
    # CSNB parameters
    parser.add_argument('--sparsity', type=float, default=0.5, metavar='SPARSITY',
                        help="Target current sparsity for each layer")
    # OMNIGLOT parameters
    parser.add_argument('--model', type=str, default='large', metavar='MODEL',
                        help='Default or Large')

    parser.add_argument("--dataset",
                        default='omniglot',
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
