# Authorized by Haeyong Kang.

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import numpy as np
from copy import deepcopy

import math
from itertools import combinations, permutations

# ------------------prime generation and prime mod tables---------------------
# Find closest number in a list
def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def save_model_params(saver_dict, model, task_id):

    print ('saving model parameters ---')

    saver_dict[task_id]['model']={}
    for k_t, (m, param) in enumerate(model.named_parameters()):
        saver_dict[task_id]['model'][m] = param
        print (k_t,m,param.shape)
    print ('-'*30)

    return saver_dict

def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if (epoch ==1):
            param_group['lr']=args.lr
        else:
            param_group['lr'] /= args.lr_factor

def is_prime(n):
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def get_primes(num_primes):
    primes = []
    for num in range(2,np.inf):
        if is_prime(num):
            primes.append(num)
            print(primes)

        if len(primes) >= num_primes:
            return primes

def checker(per_task_masks, consolidated_masks, task_id):
    # === checker ===
    for key in per_task_masks[task_id].keys():
        # Skip output head from other tasks
        # Also don't consolidate output head mask after training on new tasks; continue
        if "last" in key:
            if key in curr_head_keys:
                consolidated_masks[key] = deepcopy(per_task_masks[task_id][key])
            continue

        # Or operation on sparsity
        if 'weight' in key:
            num_cons = consolidated_masks[key].sum()
            num_prime = (prime_masks[key] > 0).sum()

            if num_cons != num_prime:
                print('diff.')

def print_sparsity(consolidated_masks, percent=1.0, item=False):
    sparsity_dict = {}
    for key in consolidated_masks.keys():
        # Skip output heads
        if "last" in key:
            continue

        mask = consolidated_masks[key]
        if mask is not None:
            sparsity = torch.sum(mask == 1) / np.prod(mask.shape)
            print("{:>12} {:>2.4f}".format(key, sparsity ))

            if item :
                sparsity_dict[key] = sparsity.item() * percent
            else:
                sparsity_dict[key] = sparsity * percent

    return sparsity_dict

def global_sparsity(consolidated_masks):
    denum, num = 0, 0
    for key in consolidated_masks.keys():
        # Skip output heads
        if "last" in key:
            continue

        mask = consolidated_masks[key]
        if mask is not None:
            num += torch.sum(mask == 1).item()
            denum += np.prod(mask.shape)

    return num / denum

def get_representation_matrix (net,device,x,y=None,task_id=0,mask=None,mode='valid',random=True):
    # Collect activations by forward pass
    r=np.arange(x.size(0))

    if random:
        np.random.shuffle(r)
        r=torch.LongTensor(r).to(device)
        b=r[0:300] # Take random training samples
        batch_list=[300,300,300]
    else:
        r=torch.LongTensor(r).to(device)
        b=r # Take all valid samples
        batch_list=[r.size(0),r.size(0),r.size(0)]

    example_data = x[b].view(-1,28*28)
    example_data = example_data.to(device)
    example_out  = net(example_data, task_id, mask, mode)

    mat_list=[] # list contains representation matrix of each layer
    act_key=list(net.act.keys())

    for i in range(len(act_key)):
        bsz=batch_list[i]
        act = net.act[act_key[i]].detach().cpu().numpy()
        activation = act[0:bsz].transpose()
        mat_list.append(activation)

    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_list)):
        print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
    print('-'*30)
    return mat_list


def update_GPM (model, mat_list, threshold, feature_list=[], task_id=None):
    print ('Threshold: ', threshold)
    if not feature_list:
        # After First Task
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)

            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1
            feature_list.append(U[:,0:r])
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()

            # Projected Representation (Eq-8)
            act_hat = activation - np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
            U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)

            # criteria (Eq-9)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            accumulated_sval = (sval_total-sval_hat)/sval_total

            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                print ('Skip Updating GPM for layer: {}'.format(i+1))
                continue
            # update GPM
            Ui=np.hstack((feature_list[i],U[:,0:r]))
            if Ui.shape[1] > Ui.shape[0] :
                feature_list[i]=Ui[:,0:Ui.shape[0]]
            else:
                feature_list[i]=Ui

    print('-'*40)
    print('Gradient Constraints Summary')
    print('-'*40)
    for i in range(len(feature_list)):
        print ('Layer {} : {}/{}'.format(i+1,feature_list[i].shape[1], feature_list[i].shape[0]))
    print('-'*40)
    return feature_list

def mask_projected (mask, feature_mat):

    # mask Projections
    for i, key in enumerate(mask.keys()):
        #for j in range(len(feature_mat)):
        if 'weight' in key:
            mask[key] = mask[key] - torch.mm(mask[key].float(), feature_mat[i])
        else:
            None

    return mask

## Define LeNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))
