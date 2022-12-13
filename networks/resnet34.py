# Authorized by Haeyong Kang.

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.nn.functional import relu, avg_pool2d

from .subnet import SubnetConv2d, SubnetLinear

import ipdb

#######################################################################################
#      GPM ResNet34
#######################################################################################

## Define ResNet34 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)

class GPMBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(GPMBasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x):
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = x
        self.count +=1
        out = relu(self.bn1(self.conv1(x)))
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = out
        self.count +=1
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class GPMResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla, nf):
        super(GPMResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, 1)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        
        self.taskcla = taskcla
        self.linear=torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.linear.append(nn.Linear(nf * 8 * block.expansion * 4, n, bias=False))
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        self.act['conv_in'] = x.view(bsz, 3, 32, 32)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32)))) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        y=[]
        for t,i in self.taskcla:
            y.append(self.linear[t](out))
        return y

def GPMResNet34(taskcla, nf=32):
    return GPMResNet(GPMBasicBlock, [3, 4, 6, 3], taskcla, nf)


#######################################################################################
#       CSNB ResNet34
#######################################################################################

# Multiple Input Sequential
class mySequential(nn.Sequential):
    def forward(self, *inputs):
        mask = inputs[1]
        mode = inputs[2]
        inputs = inputs[0]
        for module in self._modules.values():
            if isinstance(module, SubnetBasicBlock):
                inputs = module(inputs, mask, mode)
            else:
                inputs = module(inputs)
        
        return inputs

## Define ResNet34 model
def subnet_conv3x3(in_planes, out_planes, stride=1, sparsity=0.5):
    return SubnetConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, sparsity=sparsity)

def subnet_conv7x7(in_planes, out_planes, stride=1, sparsity=0.5):
    return SubnetConv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False, sparsity=sparsity)

class SubnetBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, sparsity=0.5, name=""):
        super(SubnetBasicBlock, self).__init__()
        self.name = name
        self.conv1 = subnet_conv3x3(in_planes, planes, stride, sparsity=sparsity)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False, affine=False)
        self.conv2 = subnet_conv3x3(planes, planes, sparsity=sparsity)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False, affine=False)

        # Shortcut
        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = 1
            self.conv3 = SubnetConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, sparsity=sparsity)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=False, affine=False)
        self.count = 0

    def forward(self, x, mask, mode='train'):
        name = self.name + ".conv1"
        out = relu(self.bn1(self.conv1(x, weight_mask=mask[name+'.weight'], bias_mask=mask[name+'.bias'], mode=mode)))
        name = self.name + ".conv2"
        out = self.bn2(self.conv2(out, weight_mask=mask[name+'.weight'], bias_mask=mask[name+'.bias'], mode=mode))
        if self.shortcut is not None:
            name = self.name + ".conv3"
            out += self.bn3(self.conv3(x, weight_mask=mask[name+'.weight'], bias_mask=mask[name+'.bias'], mode=mode))
        else:
            out += x
        out = relu(out)
        return out

class SubnetResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla, nf, sparsity):
        super(SubnetResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = subnet_conv3x3(3, nf * 1, 1, sparsity=sparsity)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False, affine=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, sparsity=sparsity, name="layer1")
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, sparsity=sparsity, name="layer2")
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, sparsity=sparsity, name="layer3")
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, sparsity=sparsity, name="layer4")
        
        self.taskcla = taskcla
        self.last=torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(nf * 8 * block.expansion * 4, n, bias=False))
        self.act = OrderedDict()

        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

    def _make_layer(self, block, planes, num_blocks, stride, sparsity, name):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        name_count = 0
        for stride in strides:
            new_name = name + "." + str(name_count)
            layers.append(block(self.in_planes, planes, stride, sparsity, new_name))
            self.in_planes = planes * block.expansion
            name_count += 1
        # return nn.Sequential(*layers)
        return mySequential(*layers)

    def forward(self, x, task_id, mask, mode="train"):
        if mask is None:
            mask = self.none_masks

        bsz = x.size(0)
        x = x.reshape(bsz, 3, 32, 32)
        out = relu(self.bn1(self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode))) 
        out = self.layer1(out, mask, mode)
        out = self.layer2(out, mask, mode)
        out = self.layer3(out, mask, mode)
        out = self.layer4(out, mask, mode)
        out = avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        y = self.last[task_id](out)
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

def SubnetResNet34(taskcla, nf=32, sparsity=0.5):
    return SubnetResNet(SubnetBasicBlock, [3, 4, 6, 3], taskcla, nf, sparsity=sparsity)


#######################################################################################
#      STL ResNet34
#######################################################################################

## Define ResNet34 model
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)

class STLBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(STLBasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )
        self.count = 0

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class STLResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla, nf, ncla):
        super(STLResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, 1)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        
        self.taskcla = taskcla
        self.last = nn.Linear(nf * 8 * block.expansion * 4, ncla, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32)))) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        y = self.last(out)

        return y

def STLResNet34(taskcla, ncla, nf=32):
    return STLResNet(STLBasicBlock, [3, 4, 6, 3], taskcla, nf, ncla)
