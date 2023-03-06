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

from utils import safe_save, str2bool

from networks.subnet import SubnetLinear, SubnetConv2d
from networks.resnet18 import SubnetBasicBlock
from networks.utils import *
from utils_rle import compress_ndarray, decompress_ndarray, comp_decomp_mask

from networks.alexnet import SubnetAlexNet_norm as AlexNet
from networks.resnet18 import SubnetResNet18 as ResNet18
from networks.tinynet import SubNet
from utils_huffman import comp_decomp_mask_huffman

import importlib

import wandb

def train(args, model, device, train_loader, optimizer,criterion, task_id_nominal, consolidated_masks):
    model.train()

    total_loss = 0
    total_num = 0
    correct = 0

    # Loop batches
    for (k, (v_x, v_y)) in enumerate(train_loader):
        data = v_x.to(device)
        target = v_y.to(device)

        perm = torch.randperm(v_x.size(0))
        data = data[perm]
        target = target[perm]

        optimizer.zero_grad()
        output = model(data, task_id_nominal, mask=None, mode="train")

        loss = criterion(output, target)
        loss.backward()

        pred = output.argmax(dim=1, keepdim=True)

        correct    += pred.eq(target.view_as(pred)).sum().item()
        total_loss += loss.data.cpu().numpy().item()*data.size(0)
        total_num  += data.size(0)

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

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc

def test(args, model, device, test_loader, criterion, task_id_nominal, curr_task_masks=None, mode="test"):
    model.eval()
    total_loss = 0
    total_num = 0
    correct = 0
    with torch.no_grad():
        # Loop batches
        for (k, (v_x, v_y)) in enumerate(test_loader):
            data = v_x.to(device)
            target = v_y.to(device)
            output = model(data, task_id_nominal, mask=curr_task_masks, mode=mode)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)

            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*data.size(0)
            total_num  += data.size(0)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc

def eval_class_tasks(model, tasks, args, criterion, curr_task_masks=None, mode='test', idx=-1, device=None, end_idx=-1, comp_flag=False, encoding='huffman'):
    model.eval()

    result_acc = []
    result_lss = []
    comp_ratio = None

    with torch.no_grad():
        # Loop batches
        for t, task_loader in enumerate(tasks):

            if idx == -1 or idx == t:
                lss = 0.0
                acc = 0.0

                for (i, (x, y)) in enumerate(task_loader):
                    data = x.to(device)
                    target = y.to(device)

                    if curr_task_masks is not None:
                        if comp_flag:
                            if encoding == 'huffman':
                                per_task_mask, comp_ratio = comp_decomp_mask_huffman(curr_task_masks, t, device)
                            else:
                                per_task_mask, comp_ratio = comp_decomp_mask(curr_task_masks, t, device)
                            output = model(data, t, mask=per_task_mask, mode=mode)
                        else:
                            output = model(data, t, mask=curr_task_masks[t], mode=mode)
                    else:
                        output = model(data, t, mask=None, mode=mode)
                    loss = criterion(output, target)

                    pred = output.argmax(dim=1, keepdim=True).detach()
                    acc += pred.eq(target.view_as(pred)).sum().item()
                    lss += loss.data.cpu().numpy().item()*data.size(0)

                    _, p = torch.max(output.data.cpu(), 1, keepdim=False)

                result_lss.append(lss / len(task_loader.dataset))
                result_acc.append(acc / len(task_loader.dataset))

    return result_lss[-1], result_acc[-1] * 100, comp_ratio



def dec2bin(x, bits=10):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def bin2dec(b_mask, bits=10):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b_mask.device, b_mask.dtype)
    return torch.sum(mask * b_mask, -1)


def dec2bin_mask(int_masks, bits=10):
    if bits > 1 and False:
        import ipdb; ipdb.set_trace()

    mask = 2 ** torch.arange(bits - 1, -1, -1).to(int_masks.device).long()
    dec = int_masks.unsqueeze(-1).bitwise_and(mask).ne(0).long()
    if len(int_masks.size()) > 3:
        dec = dec.permute(4, 0, 1, 2, 3)
    else:
        dec = dec.permute(2, 0, 1)
    return dec.long()


def bin2dec_mask(key, per_task_masks, int_masks, bits=10):
    if bits > 2 and False:
        import ipdb; ipdb.set_trace()
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(int_masks.device).long()
    int_masks = torch.zeros_like(int_masks).long()
    for task_id in range(bits):
        int_masks += deepcopy(per_task_masks[task_id][key]).long() * mask[-(task_id+1)]
    
    return int_masks.long()

# example
num_tasks = 10
int_d = torch.randint(0, 10, (3, 3))
bin_d = dec2bin(int_d, num_tasks)

d_rec = bin2dec(bin_d, num_tasks)
mask_error = abs(int_d - d_rec).max()  # should be 0.

print("bin2dec_error:{}".format(mask_error))

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
    save_flag = False
    int_flag = True # for single integer mask

    exp_dir = "results_{}".format(args.dataset)
    
    ## Load TinyImageNet DATASET
    Loader = importlib.import_module('dataloader.' + args.loader)
    loader = Loader.IncrementalLoader(args, seed=args.seed)
    n_inputs, n_outputs, n_tasks, input_size = loader.get_dataset_info()

    # input_size: ch * size * size = n_inputs
    print('Input size =', input_size, '\nOutput number=', n_outputs, '\nTotal task=', n_tasks)
    print('-' * 100)

    # Load test and val datasets 
    test_tasks = loader.get_tasks('test')
    val_tasks = loader.get_tasks('val')

    acc_matrix=np.zeros((loader.n_tasks,loader.n_tasks))
    sparsity_matrix = []
    sparsity_per_task = {}
    criterion = torch.nn.CrossEntropyLoss()

    # Primes and Mod table of product of primes
    print ('Task primes ---')
    num_tasks = loader.n_tasks

    # Model Instantiation
    model = SubNet(input_size, n_outputs, n_tasks, args.sparsity).to(device)

    print ('Model parameters ---')
    for k_t, (m, param) in enumerate(model.named_parameters()):
        print (k_t,m,param.shape)
    print ('-'*40)

    task_list = []
    per_task_masks, consolidated_masks, per_int_masks, int_masks = {}, {}, {}, {}
    for k in range(num_tasks):
        task_info, train_loader, _, _ = loader.new_task()
        task_id=task_info['task']
        print('*'*100)
        print('Task {:2d}'.format(task_id))
        print('*'*100)
        task_list.append(task_id)

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
            tr_loss, tr_acc = train(args, model, device, train_loader, optimizer, criterion, task_id, consolidated_masks)
            clock1 = time.time()
            print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss,tr_acc, 1000*(clock1-clock0)),end='')
            # Validate
            if False:
                valid_loss,valid_acc = test(args, model, device, val_tasks[task_id], criterion, task_id, curr_task_masks=None, mode="valid")
            else:
                valid_loss,valid_acc, comp_ratio = eval_class_tasks(model=model,
                                                        tasks=val_tasks,
                                                        args=args,
                                                        criterion=criterion,
                                                        curr_task_masks=None, mode='valid', idx=task_id, device=device)
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

        # Accumulate prime masks
        if task_id == 0:
            if int_flag:
                int_masks = deepcopy(per_task_masks[task_id])
                for key in per_task_masks[task_id].keys():
                    if "last" in key:
                        if key in curr_head_keys:
                            continue
                    if 'weight' in key:
                        int_masks[key] = bin2dec_mask(
                            key=key,
                            per_task_masks=per_task_masks,
                            int_masks=int_masks[key],
                            bits=task_id+1)
        else:
            if int_flag:
                for key in per_task_masks[task_id].keys():
                    if "last" in key:
                        if key in curr_head_keys:
                            continue
                    if 'weight' in key:
                        int_masks[key] = bin2dec_mask(
                            key=key,
                            per_task_masks=per_task_masks,
                            int_masks=int_masks[key],
                            bits=task_id+1)

        # Consolidate task masks to keep track of parameters to-update or not
        curr_head_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        if k == 0:
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
        if not int_flag:
            test_loss, test_acc = test(args, model, device, test_tasks[task_id], criterion, task_id, curr_task_masks=per_task_masks[task_id], mode="test")
        else:
            per_int_mask = deepcopy(int_masks)
            per_int_masks = deepcopy(per_task_masks)
            for key in int_masks.keys():
                if 'weight' in key:
                    dec_int_masks=dec2bin_mask(per_int_mask[key], task_id+1)

                    if (dec_int_masks[0] - per_task_masks[task_id][key]).abs().sum() == 0:
                        per_int_masks[task_id][key] = dec_int_masks[0]
                    else:
                        import ipdb; ipdb.set_trace()

            if False:
                test_loss, test_acc = test(args, model, device, test_tasks[task_id], criterion, task_id, curr_task_masks=per_int_mask, mode="test")
            else:
                test_loss, test_acc, comp_ratio = eval_class_tasks(model=model,
                                                       tasks=test_tasks,
                                                       args=args,
                                                       criterion=criterion,
                                                       curr_task_masks=per_int_masks,
                                                       mode='test',
                                                       idx=task_id,
                                                       device=device)

        print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))

        log_dict = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'task': task_id,
        }
        wandb.log(log_dict)

        # save accuracy
        #jj = 0
        for jj in np.array(task_list):
            if jj <= task_id:
                #acc_matrix[task_id, jj] = acc_matrix[task_id-1, jj]
                _, acc_matrix[task_id,jj] = test(args, model, device,test_tasks[jj], criterion, jj, curr_task_masks=per_task_masks[jj], mode="test")
            else:
                if not int_flag:
                    _, acc_matrix[task_id,jj] = test(args, model, device,test_tasks[jj], criterion, jj, curr_task_masks=per_task_masks[task_id], mode="test")
                else:
                    per_int_mask = deepcopy(int_masks)
                    per_int_masks[jj] = deepcopy(int_masks)
                    for key in int_masks.keys():
                        if 'weight' in key:
                            dec_int_masks = dec2bin_mask(per_int_mask[key], jj+1)
                            if (dec_int_masks[0] - per_task_masks[jj][key]).abs().sum() == 0:
                                per_int_masks[jj][key] = dec_int_masks[0]
                            else:
                                import ipdb; ipdb.set_trace()

                    if False:
                        _, acc_matrix[task_id,jj] = test(args,model,device,test_tasks[task_id],criterion, jj,curr_task_masks=per_int_mask, mode="test")
                    else:
                        _, acc_matrix[task_id,jj], comp_ratio = eval_class_tasks(model=model,
                                                                     tasks=test_tasks,
                                                                     args=args,
                                                                     criterion=criterion,
                                                                     curr_task_masks=per_int_masks,
                                                                     mode='test',
                                                                     idx=jj,
                                                                     device=device)
            #jj +=1
        print('Accuracies =')
        for i_a in np.array(task_list):
            print('\t',end='')
            for j_a in np.array(task_list):
                print('{:5.1f} '.format(acc_matrix[i_a,j_a]),end='')
            print()

    # Save
    save_name = "csnb_tiny_data_tinynet_{}_SEED_{}_LR_{}_SPARSITY_{}".format(args.encoding, args.seed, args.lr, 1 - args.sparsity)
    safe_save(exp_dir + "/csnb_tiny_data/" + args.name + ".acc", acc_matrix)
    safe_save(exp_dir + "/csnb_tiny_data/" + args.name + ".intmask", int_masks)
    safe_save(exp_dir + "/csnb_tiny_data/" + args.name + ".cap", sparsity_matrix)
    safe_save(exp_dir + "/csnb_tiny_data/" + args.name + ".spar", sparsity_per_task)
    safe_save(exp_dir + "/csnb_tiny_data/" + args.name + ".pertask", per_task_masks)
    safe_save(exp_dir + "/csnb_tiny_data/" + args.name + ".fullmask", consolidated_masks)
    torch.save(model.to("cpu"), exp_dir + "/csnb_tiny_data/"  + args.name + ".ptmodel")
    model = model.to(device)
    
    # Test one more time
    test_acc_matrix=np.zeros((loader.n_tasks,loader.n_tasks))
    sparsity_matrix = []
    mask_comp_ratio = []
    sparsity_per_task = {}
    criterion = torch.nn.CrossEntropyLoss()

    for k in range(num_tasks):
        print('*'*100)
        task_id = task_list[k]
        print('Task {:2d}'.format(task_id))
        print('*'*100)
        
        # Test
        print ('-'*40)
        if not int_flag:
            test_loss, test_acc = test(args, model, device, test_tasks[task_id], criterion, task_id, curr_task_masks=per_task_masks[task_id], mode="test")
        else:
            per_int_mask = deepcopy(int_masks)
            per_int_masks = deepcopy(per_task_masks)
            for key in int_masks.keys():
                if 'weight' in key:
                    dec_int_masks=dec2bin_mask(per_int_mask[key], task_id+1)

                    if (dec_int_masks[0] - per_task_masks[task_id][key]).abs().sum() == 0:
                        per_int_masks[task_id][key] = dec_int_masks[0]
                    else:
                        import ipdb; ipdb.set_trace()

            if False:
                test_loss, test_acc = test(args, model, device, test_tasks[k],  criterion, task_id, curr_task_masks=per_int_mask, mode="test")
            else:
                test_loss, test_acc, comp_ratio = eval_class_tasks(model=model,
                                                       tasks=test_tasks,
                                                       args=args,
                                                       criterion=criterion,
                                                       curr_task_masks=per_int_masks,
                                                       mode='test',
                                                       idx=task_id,
                                                       device=device, comp_flag=False)
        print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))

        # save accuracy
        jj = 0
        for ii in np.array(task_list)[0:task_id+1]:
            if jj < task_id:
                test_acc_matrix[task_id, jj] = test_acc_matrix[task_id-1, jj]
            else:
                if not int_flag:
                    _, test_acc_matrix[task_id,jj] = test(args, model, device, test_tasks[ii],criterion, jj, curr_task_masks=per_task_masks[jj], mode="test")
                else:
                    per_int_mask = deepcopy(int_masks)
                    per_int_masks = deepcopy(per_task_masks)
                    for key in int_masks.keys():
                        if 'weight' in key:
                            dec_int_masks = dec2bin_mask(per_int_mask[key], jj+1)
                            if (dec_int_masks[0] - per_task_masks[jj][key]).abs().sum() == 0:
                                per_int_masks[jj][key] = dec_int_masks[0]
                            else:
                                import ipdb; ipdb.set_trace()

                    if False:
                        _, test_acc_matrix[task_id,jj] = test(args, model, device, test_tasks[jj],criterion, jj, curr_task_masks=per_int_mask, mode="test")
                    else:
                        _, test_acc_matrix[task_id,jj], comp_ratio = eval_class_tasks(model=model,
                                                                                      tasks=test_tasks,
                                                                                      args=args,
                                                                                      criterion=criterion,
                                                                                      curr_task_masks=per_int_masks,
                                                                                      mode='test',
                                                                                      idx=jj,
                                                                                      device=device, comp_flag=True, encoding=args.encoding)
                mask_comp_ratio.append(comp_ratio/32)

            jj +=1
        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(i_a + 1):
                print('{:5.1f} '.format(test_acc_matrix[i_a,j_a]),end='')
            print()

    print('-'*50)
    # Print Sparsity

    safe_save(exp_dir + "/csnb_tiny_data/" + args.name + ".tasklist", np.array(task_list))
    safe_save(exp_dir + "/csnb_tiny_data/" + args.name + ".comp_ratio_{}", np.array(mask_comp_ratio))
    safe_save(exp_dir + "/csnb_tiny_data/" + args.name + ".test_acc", test_acc_matrix)
    sparsity_per_layer = print_sparsity(consolidated_masks)
    all_sparsity = global_sparsity(consolidated_masks)
    print("Global Sparsity: {}%".format(all_sparsity * 100))
    print("Bit Mask Capacity: {}%".format(np.sum(mask_comp_ratio)))

    # Simulation Results
    print ('Task Order : {}'.format(np.array(task_list)))
    print ('Diagonal Final Avg Accuracy: {:5.2f}%'.format( np.mean([test_acc_matrix[i,i] for i in range(num_tasks)] )))
    print ('Final Avg accuracy: {:5.2f}%'.format( np.mean(test_acc_matrix[num_tasks - 1])))
    bwt=np.mean((test_acc_matrix[-1]-np.diag(acc_matrix))[:-1])
    print ('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
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
    parser.add_argument('--pc_valid',default=0.1,type=float,
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
    parser.add_argument('--model', type=str, default="tinynet", metavar='MODEL',
                        help="Models to be incorporated for the experiment")
    # data loader
    parser.add_argument('--loader', type=str,
                        default="class_incremental_loader",
                        metavar='MODEL',
                        help="Models to be incorporated for the experiment")
    # increment
    parser.add_argument('--increment', type=int, default=5, metavar='S',
                        help='(default: 5)')
    # workers
    parser.add_argument('--workers', type=int, default=8, metavar='S',
                        help='(default: 8)')
    # class order
    parser.add_argument('--class_order', type=str, default="random", metavar='MODEL',
                        help="")
    # dataset
    parser.add_argument('--dataset', type=str, default="tinyimagenet", metavar='',
                        help="")
    # dataset
    parser.add_argument('--data_path', type=str, default="data/tiny-imagenet-200/", metavar='',
                        help="")
    parser.add_argument("--glances", default=1, type=int,
                        help="# of times the model is allowed to train over a set of samples in the single pass setting")
    parser.add_argument('--memories', type=int, default=1000,
                        help='number of total memories stored in a reservoir sampling based buffer')
    parser.add_argument('--mem_batch_size', type=int, default=300,
                        help='the amount of items selected to update feature spaces.')

    #parser.add_argument('--use_track', type=str2bool, default=False)
    parser.add_argument('--freeze_bn', default=False, action='store_true', help='')

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



