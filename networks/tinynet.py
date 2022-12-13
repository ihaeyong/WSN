# Authorized by Haeyong Kang.

import numpy as np
import torch
import math
from networks.base import *

from collections import OrderedDict
from torch.nn.functional import relu, avg_pool2d

import networks.tiny.subnet_learner as Learner
import networks.tiny.modelfactory as mf

from .subnet import SubnetConv2d, SubnetLinear

class Net(BaseNet):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__(n_inputs, n_outputs, n_tasks, args)

        # steps for sharpness
        #self.inner_steps = args.inner_batches

        # eta1: update step size of weight perturbation
        #self.eta1 = args.eta1

        # eta2: learning rate of lambda(soft weight for basis)
        #self.eta2 = args.eta2

    def forward(self, x, t):
        output = self.net.forward(x)

        if self.net.multi_head :
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)

        return output


## Define model
def subnet_conv3x3(in_planes, out_planes, stride=1, padding=1, sparsity=0.5):
    return SubnetConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, sparsity=sparsity)

def subnet_conv7x7(in_planes, out_planes, stride=1, padding=1, sparsity=0.5):
    return SubnetConv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=padding, bias=False, sparsity=sparsity)


class SubNet(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, n_tasks, sparsity):
        super(SubNet, self).__init__()

        # setup network
        tasks_ = [t for t  in range(n_tasks)]
        n_outputs_ = [n_outputs] * n_tasks
        taskcla = [(t,n) for t,n in zip(tasks_, n_outputs_)]

        # tinyimagenet
        config = mf.ModelFactory.get_model(sizes=[n_outputs],
                                           dataset='tinyimagenet')

        self.bn_flag = False   # default
        self.use_track = False # degault
        self.drop1 = torch.nn.Dropout(0.0)
        self.drop2 = torch.nn.Dropout(0.0)

        self.make_layer(config, taskcla, sparsity)
        self.act = OrderedDict()

        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None


    def make_layer(self, config, taskcla, sparsity):
        # conv2d-nbias
        in_planes = config[0][1][1]
        planes = config[0][1][0]
        stride = config[0][1][4]
        padding = config[0][1][5]
        self.conv1 = subnet_conv3x3(in_planes, planes, stride, padding,
                                    sparsity=sparsity)

        if self.bn_flag:
            if self.use_track :
                self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
            else:
                self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False, affine=False)

        # conv2d-nbias
        in_planes = config[2][1][1]
        planes = config[2][1][0]
        stride = config[2][1][4]
        padding = config[2][1][5]
        self.conv2 = subnet_conv3x3(in_planes, planes, stride, padding,
                                    sparsity=sparsity)
        if self.bn_flag:
            if self.use_track :
                self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
            else:
                self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False, affine=False)

        # conv2d-nbias
        in_planes = config[4][1][1]
        planes = config[4][1][0]
        stride = config[4][1][4]
        padding = config[4][1][5]
        self.conv3 = subnet_conv3x3(in_planes, planes, stride, padding,
                                    sparsity=sparsity)
        if self.bn_flag:
            if self.use_track :
                self.bn3 = nn.BatchNorm2d(planes, momentum=0.1)
            else:
                self.bn3 = nn.BatchNorm2d(planes, track_running_stats=False, affine=False)

        # conv2d-nbias
        in_planes = config[6][1][1]
        planes = config[6][1][0]
        stride = config[6][1][4]
        padding = config[6][1][5]
        self.conv4 = subnet_conv3x3(in_planes, planes, stride, padding,
                                    sparsity=sparsity)
        if self.bn_flag:
            if self.use_track :
                self.bn4 = nn.BatchNorm2d(planes, momentum=0.1)
            else:
                self.bn4 = nn.BatchNorm2d(planes, track_running_stats=False, affine=False)


        # linear-nbias
        in_planes = config[10][1][1]
        planes = config[10][1][0]
        self.linear1 = SubnetLinear(in_planes, planes, bias=False,
                                    sparsity=sparsity)

        if self.bn_flag:
            if self.use_track :
                self.bn_l1 = nn.BatchNorm1d(planes, momentum=0.1)
            else:
                self.bn_l1 = nn.BatchNorm1d(planes, track_running_stats=False, affine=False)

        in_planes = config[12][1][1]
        planes = config[12][1][0]
        self.linear2 = SubnetLinear(in_planes, planes, bias=False,
                                    sparsity=sparsity)

        if self.bn_flag:
            if self.use_track :
                self.bn_l2 = nn.BatchNorm1d(planes, momentum=0.1)
            else:
                self.bn_l2 = nn.BatchNorm1d(planes, track_running_stats=False, affine=False)

        # linear-nbias
        in_planes = config[-1][1][1]
        planes = config[-1][1][0]
        self.last=torch.nn.ModuleList()
        for t, n in taskcla:
            self.last.append(nn.Linear(in_planes, planes, bias=False))


    def forward(self, x, task_id, mask=None, mode="train"):

        if mask is None:
            mask = self.none_masks

        bsz = x.size(0)
        x = x.reshape(bsz, 3, 64, 64)

        # conv2d-nbias
        out = self.conv1(x, weight_mask=mask['conv1.weight'],
                         bias_mask=mask['conv1.bias'], mode=mode)
        if self.bn_flag and False:
            out = self.bn1(out)
        out = relu(out)

        # conv2d-nbias
        out = self.conv2(out, weight_mask=mask['conv2.weight'],
                         bias_mask=mask['conv2.bias'], mode=mode)
        if self.bn_flag and False:
            out = self.bn2(out)
        out = relu(out)


        # conv2d-nbias
        out = self.conv3(out, weight_mask=mask['conv3.weight'],
                         bias_mask=mask['conv3.bias'], mode=mode)
        if self.bn_flag:
            out = self.bn3(out)
        out = relu(out)


        # conv2d-nbias
        out = self.conv4(out, weight_mask=mask['conv4.weight'],
                         bias_mask=mask['conv4.bias'], mode=mode)
        if self.bn_flag:
            out = self.bn4(out)
        out = relu(out)
        out = self.drop1(out)

        # flatten
        out = out.view(out.size(0), -1)

        # linear-nbias
        out = self.linear1(out,
                           weight_mask=mask['linear1.weight'],
                           bias_mask=mask['linear1.bias'],
                           mode=mode)
        if self.bn_flag:
            out = self.bn_l1(out)

        out = relu(out)
        out = self.drop2(out)

        # linear-nbias
        out = self.linear2(out,
                           weight_mask=mask['linear2.weight'],
                           bias_mask=mask['linear2.bias'],
                           mode=mode)
        if self.bn_flag:
            out = self.bn_l2(out)

        out = relu(out)

        y = self.last[task_id](out)

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

