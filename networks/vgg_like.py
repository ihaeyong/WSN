import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from collections import OrderedDict

from .subnet import SubnetConv2d, SubnetLinear

## Define AlexNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

#########################################################################################
################################### CIFAR Experiments ###################################
#########################################################################################


class CifarConv(nn.Module):
    def __init__(self,taskcla, sparsity=0.5):
        super(CifarConv, self).__init__()
        self.conv1 = SubnetConv2d(3, 32, 3, padding=1, sparsity=sparsity, bias=False)
        self.conv2 = SubnetConv2d(32, 32, 3, padding=1, sparsity=sparsity, bias=False)
        s=compute_conv_output_size(32,4,stride=1,padding=1)
        s=s//2
        self.conv3 = SubnetConv2d(32, 64, 3, padding=1, sparsity=sparsity, bias=False)
        self.conv4 = SubnetConv2d(64, 64, 3, padding=1, sparsity=sparsity, bias=False)
        s=compute_conv_output_size(s,3,stride=1,padding=1)
        s=s//2
        self.conv5 = SubnetConv2d(64, 128, 3, padding=1, sparsity=sparsity, bias=False)
        self.conv6 = SubnetConv2d(128, 128, 3, padding=1, sparsity=sparsity, bias=False)
        s=compute_conv_output_size(s,2,stride=1,padding=1)
        s=s//2
        self.smid=s
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.25)
        self.drop2=torch.nn.Dropout(0.5)

        self.fc1 = SubnetLinear(128*self.smid*self.smid, 256, sparsity=sparsity, bias=False)
        
        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(nn.Linear(256, n, bias=False))
        
        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

    def forward(self, x, task_id, mask, mode="train"):
        if mask is None:
            mask = self.none_masks

        bsz = deepcopy(x.size(0))
        x = self.relu(self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode))
        x = self.relu(self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode))
        x = self.drop1(self.maxpool(x))

        x = self.relu(self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode))
        x = self.relu(self.conv4(x, weight_mask=mask['conv4.weight'], bias_mask=mask['conv4.bias'], mode=mode))
        x = self.drop1(self.maxpool(x))

        x = self.relu(self.conv5(x, weight_mask=mask['conv5.weight'], bias_mask=mask['conv5.bias'], mode=mode))
        x = self.relu(self.conv6(x, weight_mask=mask['conv6.weight'], bias_mask=mask['conv6.bias'], mode=mode))
        x = self.drop1(self.maxpool(x))

        x=x.view(bsz,-1)
        x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        h_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        y = self.last[task_id](x)
        return y

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                print(name)
                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None
                
        return task_mask

#########################################################################################
################################# Omniglot Experiments ##################################
#########################################################################################

class OmniglotConv(nn.Module):
    def __init__(self,taskcla, sparsity=0.5):
        super(OmniglotConv, self).__init__()

        self.conv1 = SubnetConv2d(1, 64, 3, sparsity=sparsity, bias=False)
        s=compute_conv_output_size(28,3,stride=1,padding=0) # 26
        self.conv2 = SubnetConv2d(64, 64, 3, sparsity=sparsity, bias=False)
        s=compute_conv_output_size(s,3,stride=1,padding=0) # 24
        s=s//2  # 12

        self.conv3 = SubnetConv2d(64, 64, 3, sparsity=sparsity, bias=False)
        s=compute_conv_output_size(s,3,stride=1,padding=0) # 10
        self.conv4 = SubnetConv2d(64, 64, 3, sparsity=sparsity, bias=False)
        s=compute_conv_output_size(s,3,stride=1,padding=0) # 8
        s = s//2

        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        
        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(nn.Linear(s*s*64, n, bias=False))
        
        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

    def forward(self, x, task_id, mask, mode="train"):
        if mask is None:
            mask = self.none_masks

        bsz = deepcopy(x.size(0))
        x = self.relu(self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode))
        x = self.relu(self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode))
        x = self.maxpool(x)

        x = self.relu(self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode))
        x = self.relu(self.conv4(x, weight_mask=mask['conv4.weight'], bias_mask=mask['conv4.bias'], mode=mode))
        x = self.maxpool(x)

        x=x.view(bsz,-1)
        h_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        y = self.last[task_id](x)
        return y

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                print(name)
                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None
                
        return task_mask

#########################################################################################
################################# 8 Dataset Experiments #################################
#########################################################################################

class MixtureConv(nn.Module):
    def __init__(self,taskcla, sparsity=0.5):
        super(MixtureConv, self).__init__()

        self.taskcla = taskcla
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout()
        self.conv1 = SubnetConv2d(3, 64, 3, padding=1, stride=2, sparsity=sparsity, bias=False)
        self.conv2 = SubnetConv2d(64, 192, 3, padding=1, sparsity=sparsity, bias=False)
        self.conv3 = SubnetConv2d(192, 384, 3, padding=1, sparsity=sparsity, bias=False)
        self.conv4 = SubnetConv2d(384, 256, 3, padding=1, sparsity=sparsity, bias=False)
        self.conv5 = SubnetConv2d(256, 256, 3, padding=1, sparsity=sparsity, bias=False)
        self.fc1 = SubnetLinear(256 *1 * 1, 4096, sparsity=sparsity, bias=False)
        self.fc2 = SubnetLinear(4096, 4096, sparsity=sparsity, bias=False)
        
        self.last = nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(nn.Linear(4096, n, bias=False))
        
        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

    def forward(self, x, task_id, mask, mode="train"):
        if mask is None:
            mask = self.none_masks

        bsz = deepcopy(x.size(0))
        x = self.relu(self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode))
        x = self.maxpool(x)

        x = self.relu(self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode))
        x = self.maxpool(x)

        x = self.relu(self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode))
        x = self.relu(self.conv4(x, weight_mask=mask['conv4.weight'], bias_mask=mask['conv4.bias'], mode=mode))
        x = self.relu(self.conv5(x, weight_mask=mask['conv5.weight'], bias_mask=mask['conv5.bias'], mode=mode))
        x = self.maxpool(x)

        x = x.view(bsz, -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode))
        x = self.dropout(x)
        x = self.relu(self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode))

        h_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        y = self.last[task_id](x)
        return y

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                print(name)
                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None
                
        return task_mask