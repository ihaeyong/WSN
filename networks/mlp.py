# Authorized by Haeyong Kang. 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from copy import deepcopy

from .subnet import SubnetConv2d, SubnetLinear, get_none_masks
from .conditional_task import TaskLinear

class SubnetMLPNet(nn.Module):
    def __init__(self, taskcla, sparsity, n_hidden=100):
        super(SubnetMLPNet, self).__init__()

        self.act=OrderedDict()
        self.fc1 = SubnetLinear(784, n_hidden, sparsity=sparsity, bias=False)
        self.fc2 = SubnetLinear(n_hidden, n_hidden, sparsity=sparsity, bias=False)

        self.taskcla = taskcla

        self.n_rep = 3
        self.multi_head = True

        if self.multi_head:
            self.last = nn.ModuleList()
            for t, n in self.taskcla:
                self.last.append(nn.Linear(n_hidden, n, bias=False))
        else:
            self.last = nn.Linear(n_hidden, taskcla[0][1], bias=False)

        self.relu = nn.ReLU()

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
        x=x.reshape(bsz,-1)
        self.act['Lin1'] = x
        x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
        x = self.relu(x)
        self.act['Lin2'] = x
        x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
        x = self.relu(x)
        self.act['fc1'] = x

        if self.multi_head:
            h_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
            # y = self.last[task_id](x, mask[h_keys[0]], mask[h_keys[1]], mode=mode)
            y = self.last[task_id](x)
        else:
            #y = self.last(x)
            y = self.last(x, weight_mask=mask['last.weight'], bias_mask=mask['last.bias'], mode=mode)

        return y

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if self.multi_head:
                if 'last' in name:
                    if name != 'last.' + str(task_id):
                        continue
            else:
                None

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                print(name)
                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.long)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.long)
                else:
                    task_mask[name + '.bias'] = None

        return task_mask
