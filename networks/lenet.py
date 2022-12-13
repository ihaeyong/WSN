# Authorized by Haeyong Kang.

import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from collections import OrderedDict

from .subnet import SubnetConv2d, SubnetLinear
from .subweight import SubweightConv2d, SubweightLinear

## Define LeNet model 
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class STLLeNet(nn.Module):
    def __init__(self, taskcla, task_id):
        super(STLLeNet, self).__init__()
        self.map =[]
        self.in_channel =[]
        
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 20, 5, bias=False, padding=2)

        s=compute_conv_output_size(32,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.in_channel.append(3)        
        self.map.append(s)
        self.conv2 = nn.Conv2d(20, 50, 5, bias=False, padding=2)
        
        s=compute_conv_output_size(s,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.in_channel.append(20)        
        self.smid=s
        self.map.append(50*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(3,2,padding=1)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0)
        self.drop2=torch.nn.Dropout(0)

        self.fc1 = nn.Linear(50*self.smid*self.smid,800, bias=False)
        self.fc2 = nn.Linear(800,500, bias=False)
        self.map.extend([800])
        
        self.taskcla = taskcla
        self.fc3 = nn.Linear(500, taskcla[task_id][1], bias=False)
        
    def forward(self, x):
        bsz = deepcopy(x.size(0))
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(x)))

        x=x.reshape(bsz,-1)
        x = self.fc1(x)
        x = self.drop2(self.relu(x))

        x = self.fc2(x)
        x = self.drop2(self.relu(x))

        y = self.fc3(x)

        return y

class GPMLeNet(nn.Module):
    def __init__(self,taskcla):
        super(GPMLeNet, self).__init__()
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 20, 5, bias=False, padding=2)

        s=compute_conv_output_size(32,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.ksize.append(5)
        self.in_channel.append(3)        
        self.map.append(s)
        self.conv2 = nn.Conv2d(20, 50, 5, bias=False, padding=2)
        
        s=compute_conv_output_size(s,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.ksize.append(5)
        self.in_channel.append(20)        
        self.smid=s
        self.map.append(50*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(3,2,padding=1)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0)
        self.drop2=torch.nn.Dropout(0)

        self.fc1 = nn.Linear(50*self.smid*self.smid,800, bias=False)
        self.fc2 = nn.Linear(800,500, bias=False)
        self.map.extend([800])
        
        self.taskcla = taskcla
        self.fc3=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.fc3.append(torch.nn.Linear(500,n,bias=False))
        
    def forward(self, x):
        bsz = deepcopy(x.size(0))
        self.act['conv1']=x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(x)))

        self.act['conv2']=x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(x)))

        x=x.reshape(bsz,-1)
        self.act['fc1']=x
        x = self.fc1(x)
        x = self.drop2(self.relu(x))

        self.act['fc2']=x        
        x = self.fc2(x)
        x = self.drop2(self.relu(x))

        y=[]
        for t,i in self.taskcla:
            y.append(self.fc3[t](x))
            
        return y

class SubnetLeNet(nn.Module):
    def __init__(self,taskcla, sparsity):
        super(SubnetLeNet, self).__init__()
        self.in_channel =[]
        self.conv1 = SubnetConv2d(3, 20, 5, sparsity=sparsity, bias=False, padding=2)

        s=compute_conv_output_size(32,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.in_channel.append(3)    
        self.conv2 = SubnetConv2d(20, 50, 5, sparsity=sparsity, bias=False, padding=2)
        
        s=compute_conv_output_size(s,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.in_channel.append(20)        
        self.smid=s
        self.maxpool=torch.nn.MaxPool2d(3,2, padding=1)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0)
        self.drop2=torch.nn.Dropout(0)

        self.fc1 = SubnetLinear(50*self.smid*self.smid, 800, sparsity=sparsity, bias=False)
        self.fc2 = SubnetLinear(800, 500, sparsity=sparsity, bias=False)
        
        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            # self.last.append(SubnetLinear(500, n, bias=False))
            self.last.append(nn.Linear(500, n, bias=False))
        
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
        x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x=x.reshape(bsz,-1)
        x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        h_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        # y = self.last[task_id](x, mask[h_keys[0]], mask[h_keys[1]], mode=mode)
        y = self.last[task_id](x)

        return y

    def init_masks(self, task_id):
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                print("{}:reinitialized weight score".format(name))
                module.init_mask_parameters()

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

class SubweightLenet(nn.Module):
    def __init__(self, taskcla, task_id):
        super(SubweightLenet, self).__init__()

        self.bn_flag = True 
        
        self.map =[]
        self.in_channel =[]
        
        self.map.append(32)
        self.conv1 = SubweightConv2d(3, 20, 5, bias=False, padding=2)

        if self.bn_flag: 
            self.bn1 = nn.BatchNorm2d(20, track_running_stats=False, affine=False)

        s=compute_conv_output_size(32,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.in_channel.append(3)        
        self.map.append(s)
        self.conv2 = SubweightConv2d(20, 50, 5, bias=False, padding=2)

        if self.bn_flag: 
            self.bn2 = nn.BatchNorm2d(50, track_running_stats=False, affine=False)
        
        s=compute_conv_output_size(s,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.in_channel.append(20)        
        self.smid=s
        self.map.append(50*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(3,2,padding=1)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0)
        self.drop2=torch.nn.Dropout(0)

        self.fc1 = SubweightLinear(50*self.smid*self.smid,800, bias=False)
        self.fc2 = SubweightLinear(800,500, bias=False)
        self.map.extend([800])
        
        self.taskcla = taskcla
        self.fc3 = SubweightLinear(500, taskcla[task_id][1], bias=False)
        
    def forward(self, x):
        bsz = deepcopy(x.size(0))
        x = self.conv1(x)
        if self.bn_flag:
            x = self.maxpool(self.drop1(self.relu(self.bn1(x))))
        else:
            x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv2(x)
        if self.bn_flag:
            x = self.maxpool(self.drop1(self.relu(self.bn2(x))))
        else:
            x = self.maxpool(self.drop1(self.relu(x)))
        x=x.reshape(bsz,-1)
        x = self.fc1(x)
        x = self.drop2(self.relu(x))

        x = self.fc2(x)
        x = self.drop2(self.relu(x))

        y = self.fc3(x)

        return y

class SubnetLeNet_Large(nn.Module):
    def __init__(self,taskcla, sparsity):
        super(SubnetLeNet_Large, self).__init__()
        self.in_channel =[]
        self.conv1 = SubnetConv2d(1, 64, 5, sparsity=sparsity, bias=False, padding=2)

        s=compute_conv_output_size(32,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.in_channel.append(3)    
        self.conv2 = SubnetConv2d(64, 128, 5, sparsity=sparsity, bias=False, padding=2)
        
        s=compute_conv_output_size(s,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.in_channel.append(20)        
        self.smid=s
        self.maxpool=torch.nn.MaxPool2d(3,2, padding=1)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0)
        self.drop2=torch.nn.Dropout(0)

        self.fc1 = SubnetLinear(128*self.smid*self.smid, 2500, sparsity=sparsity, bias=False)
        self.fc2 = SubnetLinear(2500, 1500, sparsity=sparsity, bias=False)
        
        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            # self.last.append(SubnetLinear(500, n, bias=False))
            self.last.append(nn.Linear(1500, n, bias=False))
        
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
        x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x=x.reshape(bsz,-1)
        x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        h_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        # y = self.last[task_id](x, mask[h_keys[0]], mask[h_keys[1]], mode=mode)
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

class SubnetLeNet_Default(nn.Module):
    def __init__(self,taskcla, sparsity):
        super(SubnetLeNet_Default, self).__init__()
        self.in_channel =[]
        self.conv1 = SubnetConv2d(1, 10, 5, sparsity=sparsity, bias=False, padding=2)

        s=compute_conv_output_size(32,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.in_channel.append(3)    
        self.conv2 = SubnetConv2d(10, 20, 5, sparsity=sparsity, bias=False, padding=2)
        
        s=compute_conv_output_size(s,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.in_channel.append(20)        
        self.smid=s
        self.maxpool=torch.nn.MaxPool2d(3,2, padding=1)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0)
        self.drop2=torch.nn.Dropout(0)

        self.fc1 = SubnetLinear(20*self.smid*self.smid, 500, sparsity=sparsity, bias=False)
        self.fc2 = SubnetLinear(500, 300, sparsity=sparsity, bias=False)
        
        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            # self.last.append(SubnetLinear(500, n, bias=False))
            self.last.append(nn.Linear(300, n, bias=False))
        
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
        x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x=x.reshape(bsz,-1)
        x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        h_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        # y = self.last[task_id](x, mask[h_keys[0]], mask[h_keys[1]], mode=mode)
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

class SubnetLeNet_AGS(nn.Module):
    def __init__(self,taskcla, sparsity):
        super(SubnetLeNet_AGS, self).__init__()
        self.in_channel =[]
        self.conv1 = SubnetConv2d(1, 64, 5, sparsity=sparsity, bias=False, padding=2)

        s=compute_conv_output_size(28,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.in_channel.append(3)    
        self.conv2 = SubnetConv2d(64, 128, 5, sparsity=sparsity, bias=False, padding=2)
        
        s=compute_conv_output_size(s,5,1,2)
        s=compute_conv_output_size(s,3,2,1)
        self.in_channel.append(20)        
        self.smid=s
        self.maxpool=torch.nn.MaxPool2d(3,2, padding=1)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0)
        self.drop2=torch.nn.Dropout(0)

        self.fc1 = SubnetLinear(128*self.smid*self.smid, 2500, sparsity=sparsity, bias=False)
        self.fc2 = SubnetLinear(2500, 1500, sparsity=sparsity, bias=False)
        
        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            # self.last.append(SubnetLinear(500, n, bias=False))
            self.last.append(nn.Linear(1500, n, bias=False))
        
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
        x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x=x.reshape(bsz,-1)
        x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        h_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        # y = self.last[task_id](x, mask[h_keys[0]], mask[h_keys[1]], mode=mode)
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
