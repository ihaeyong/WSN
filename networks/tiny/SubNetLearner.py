### Reproduce of La-MAML from https://github.com/montrealrobotics/La-MAML

import math
import os
import sys
import traceback
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import relu, avg_pool2d

from networks.subnet import SubnetConv2d, SubnetLinear

## Define model
def subnet_conv3x3(in_planes, out_planes, stride=1, padding=1, sparsity=0.5):
    return SubnetConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, sparsity=sparsity)

def subnet_conv7x7(in_planes, out_planes, stride=1, padding=1, sparsity=0.5):
    return SubnetConv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=padding, bias=False, sparsity=sparsity)


class SubNetLearner(nn.Module):

    def __init__(self, config, taskcla, sparsity):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(SubNetLearner, self).__init__()

        self.config = config
        self.sparsity = sparcity

        # conv2d-nbias
        in_planes = config[0][1][1]
        planes = config[0][1][0]
        stride = config[0][1][4]
        padding = config[0][1][5]
        self.conv1 = subnet_conv3x3(in_planes, planes, stride, padding,
                                    sparsity=sparsity)


        # conv2d-nbias
        in_planes = config[2][1][1]
        planes = config[2][1][0]
        stride = config[2][1][4]
        padding = config[2][1][5]
        self.conv2 = subnet_conv3x3(in_planes, planes, stride, padding,
                                    sparsity=sparsity)


        # conv2d-nbias
        in_planes = config[4][1][1]
        planes = config[4][1][0]
        stride = config[4][1][4]
        padding = config[4][1][5]
        self.conv3 = subnet_conv3x3(in_planes, planes, stride, padding,
                                    sparsity=sparsity)

        # conv2d-nbias
        in_planes = config[6][1][1]
        planes = config[6][1][0]
        stride = config[6][1][4]
        padding = config[6][1][5]
        self.conv4 = subnet_conv3x3(in_planes, planes, stride, padding,
                                    sparsity=sparsity)


        # linear-nbias
        in_planes = config[10][1][1]
        planes = config[10][1][0]
        self.linear5 = SubneLinear(in_planes, planes, bias=False,
                                   sparsity=sparsity)

        # linear-nbias
        in_planes = config[12][1][1]
        planes = config[12][1][0]
        self.last=torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(in_planes, planes, bias=False))

        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None


        import ipdb; ipdb.set_trace()


    def forward(self, x, task_id, mask, mode="train"):

        if mask is None:
            mask = self.none_masks

        bsz = x.size(0)
        x = x.reshape(bsz, 3, 64, 64)

        # conv2d-nbias
        out = self.conv1(x, weight_mask=mask['conv1.weight'],
                         bias_mask=mask['conv1.bias'], mode=mode)
        out = relu(out)

        # conv2d-nbias
        out = self.conv2(out, weight_mask=mask['conv2.weight'],
                         bias_mask=mask['conv2.bias'], mode=mode)
        out = relu(out)


        # conv2d-nbias
        out = self.conv3(out, weight_mask=mask['conv3.weight'],
                         bias_mask=mask['conv3.bias'], mode=mode)
        out = relu(out)


        # conv2d-nbias
        out = self.conv4(out, weight_mask=mask['conv4.weight'],
                         bias_mask=mask['conv4.bias'], mode=mode)
        out = relu(out)

        # flatten
        out = out.view(out.size(0), -1)


        # linear-nbias
        out = self.linear5(out)
        out = relu(out)

        y = self.last[task_id](out)
        return y
