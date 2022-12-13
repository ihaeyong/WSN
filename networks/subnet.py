# Authorized by Haeyong Kang.

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict
import math

import numpy as np

def percentile(scores, sparsity):
    k = 1 + round(.01 * float(sparsity) * (scores.numel() - 1))
    return scores.view(-1).kthvalue(k).values.item()

class GetSubnetFaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):
        k_val = percentile(scores, sparsity*100)
        return torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


class STEMult(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, m):
        ctx.save_for_backward(w)
        return w * m

    @staticmethod
    def backward(ctx, g):
        return g, g*ctx.saved_tensors[0].clone()

def get_none_masks(model):
        none_masks = {}
        for name, module in model.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                none_masks[name + '.weight'] = None
                none_masks[name + '.bias'] = None

class SubnetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, sparsity=0.5, trainable=True):
        super(self.__class__, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.sparsity = sparsity
        self.trainable = trainable

        # Mask Parameters of Weights and Bias
        self.w_m = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)
        if bias:
            self.b_m = nn.Parameter(torch.empty(out_features))
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        self.init_mask_parameters()

        self.Uf = None

        if trainable == False:
            raise Exception("Non-trainable version is not yet implemented")

    def get_gpm(self,x, weights):
        with torch.no_grad():
            # -- GPM ---
            bsz = x.size(0)
            b_idx = range(bsz)
            activation = torch.mm(x[b_idx,], weights.t()).t().cpu().numpy()
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)

            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<0.999)
            feat=U[:,0:r]
            self.Uf=torch.Tensor(np.dot(feat,feat.transpose())).to(weights.device)

    def infer_mask(self,x, weights, scores, sparsity):
        with torch.no_grad():
            # -- GPM ---
            bsz = x.size(0)
            b_idx = range(bsz)
            activation = torch.mm(x[b_idx,], weights.t()).t().cpu().numpy()
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)

            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<0.999)
            feat=U[:,0:r]
            Uf=torch.Tensor(np.dot(feat,feat.transpose())).to(weights.device)

            scores=torch.mm(scores.view(bsz, -1), Uf).view(scores.size())

        k_val = percentile(scores, sparsity*100)
        return torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))

    def forward(self, x, weight_mask=None, bias_mask=None, mode="train"):
        w_pruned, b_pruned = None, None
        # If training, Get the subnet by sorting the scores
        if mode == "train":
            if weight_mask is None:
                self.weight_mask=GetSubnetFaster.apply(self.w_m.abs(),
                                                        self.zeros_weight,
                                                        self.ones_weight,
                                                        self.sparsity)
            else:
                self.weight_mask = weight_mask
            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = GetSubnetFaster.apply(self.b_m.abs(),
                                                       self.zeros_bias,
                                                       self.ones_bias,
                                                       self.sparsity)
                b_pruned = self.bias_mask * self.bias

        # If inference/valid, use the last compute masks/subnetworks
        if mode == "valid":
            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias_mask * self.bias

        # If inference, no need to compute the subnetwork
        elif mode == "test":
            w_pruned = weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                b_pruned = bias_mask * self.bias

        return F.linear(input=x, weight=w_pruned, bias=b_pruned)

    def init_mask_parameters(self):
        nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_m)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_m, -bound, bound)

class SubnetConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, sparsity=0.5, trainable=True):
        super(self.__class__, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.stride = stride
        # self.padding = padding
        self.sparsity = sparsity
        self.trainable = trainable

        # Mask Parameters of Weight and Bias
        self.w_m = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)

        if bias:
            self.b_m = nn.Parameter(torch.empty(out_channels))
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        self.init_mask_parameters()

        if trainable == False:
            raise Exception("Non-trainable version is not yet implemented")

        self.Uf = None

    def get_gpm(self, x, weights, stride, padding):
        with torch.no_grad():
            # -- GPM ---
            activation = F.conv2d(input=x, weight=weights, bias=None,stride=stride, padding=padding).cpu().numpy()
            # --------------------------
            out_ch, in_ch, ksz, ksz = weights.size()
            bsz, out_ch, sz, sz = activation.shape

            p1d = (1, 1, 1, 1)
            k = 0
            #sf = compute_conv_output_size(activation.shape, ksz, stride, padding)
            b_idx=range(bsz)
            mat = np.zeros((ksz*ksz*in_ch, sz*sz*len(b_idx)))
            act = F.pad(x, p1d, "constant", 0).detach().cpu().numpy()
            for kk in b_idx:
                for ii in range(sz):
                    for jj in range(sz):
                        mat[:,k]=act[kk,:,stride*ii:ksz+stride*ii,stride*jj:ksz+stride*jj].reshape(-1)
                        k +=1
            # activation
            U,S,Vh = np.linalg.svd(mat, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<0.945)
            feat=U[:,0:r]
            self.Uf=torch.Tensor(np.dot(feat,feat.transpose())).to(weights.device)

    def forward(self, x, weight_mask=None, bias_mask=None, mode="train", epoch=1):
        w_pruned, b_pruned = None, None

        # If training, Get the subnet by sorting the scores
        if mode == "train":
            self.weight_mask = GetSubnetFaster.apply(self.w_m.abs(),
                                                         self.zeros_weight,
                                                         self.ones_weight,
                                                         self.sparsity)

            # w_pruned = STEMult.apply(self.weight, self.weight_mask)
            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = GetSubnetFaster.apply(self.b_m.abs(), self.zeros_bias, self.ones_bias, self.sparsity)
                b_pruned = self.bias_mask * self.bias

        # If inference/valid, use the last compute masks/subnetworks
        elif mode == "valid":
            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias_mask * self.bias

            if epoch == 1 and False:
                self.get_gpm(x, self.weight, self.stride, self.padding)

        # If inference/test, no need to compute the subnetwork
        elif mode == "test":
            w_pruned = weight_mask * self.weight
            # print(torch.sum(w_pruned))
            b_pruned = None
            if self.bias is not None:
                b_pruned = bias_mask * self.bias

        else:
            raise Exception("[ERROR] The mode " + str(mode) + " is not supported!")

        return F.conv2d(input=x, weight=w_pruned, bias=b_pruned, stride=self.stride, padding=self.padding)

    def init_mask_parameters(self):
        nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_m)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_m, -bound, bound)
