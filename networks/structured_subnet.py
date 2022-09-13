import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import ipdb

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

class StructuredSubnetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, sparsity=0.5, trainable=True):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.sparsity = sparsity
        self.trainable = trainable

        # Mask Parameters of Weight and Bias
        self.w_m = nn.Parameter(torch.empty(in_features))
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)

        # If Bias
        if bias:
            self.b_m = nn.Parameter(torch.empty(out_features))
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        self.init_mask_parameters()

        # Not Trainable (?)
        if trainable == False:
            raise Exception("Non-trainable version is not yet implemented")

    def forward(self, x, weight_mask=None, bias_mask=None, mode="train"):
        w_pruned, b_pruned = None, None

        # If Training, Get the Subnet by sorting the scores
        if mode == 'train':
            self.weight_mask = GetSubnetFaster.apply(self.w_m.abs(), self.zeros_weight, self.ones_weight, self.sparsity)
            w_pruned = self.weight * self.weight_mask 
            
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = GetSubnetFaster.apply(self.b_m.abs(), self.zeros_bias, self.ones_bias, self.sparsity)
                b_pruned = self.bias_mask * self.bias
        
        # If inference/valid, use the last computed masks/subnetworks
        elif mode == 'valid':
            w_pruned = self.weight * self.weight_mask
            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias * self.bias_mask

        # If inference/test, use the given masks and no need to compute the current subnetwork
        elif mode == 'test':
            w_pruned = self.weight * weight_mask
            b_pruned = None
            if self.bias is not None:
                b_runed = self.bias * bias_mask

        else:
            raise Exception("[ERROR] The mode " + str(mode) + " is not supported!")

        return F.linear(input=x, weight=w_pruned, bias=b_pruned)

    def init_mask_parameters(self):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.w_m, -bound, bound)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_m)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_m, -bound, bound)

class StructuredSubnetConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, sparsity=0.5, trainable=True):
        super(self.__class__, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        self.stride = stride
        self.sparsity = sparsity
        self.trainable = trainable

        # Mask Parameters of Weights and Biases
        self.w_m = nn.Parameter(torch.empty(in_channels))
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)

        # If Bias
        if bias:
            self.b_m = nn.Parameter(torch.empty(out_channels))
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        self.init_mask_parameters()

        # Non-trainable is still not implemented
        if trainable == False:
            raise Exception("Non-trainable version is still not implemented")

    def forward(self, x, weight_mask=None, bias_mask=None, mode="train"):
        w_pruned, b_pruned = None, None

        # If training, Get the subnet by sorting the scores
        if mode == "train":           
            self.weight_mask = GetSubnetFaster.apply(self.w_m.abs(), self.zeros_weight, self.ones_weight, self.sparsity)
            # w_pruned = STEMult.apply(self.weight, self.weight_mask)
            w_pruned = self.weight * self.weight_mask.view(1, -1, 1, 1)
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = GetSubnetFaster.apply(self.b_m.abs(), self.zeros_bias, self.ones_bias, self.sparsity)
                b_pruned = self.bias * self.bias_mask

        # If inference/valid, use the last compute masks/subnetworks
        elif mode == "valid":           
            w_pruned = self.weight * self.weight_mask.view(1, -1, 1, 1)
            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias * self.bias_mask

        # If inference, no need to compute the subnetwork
        elif mode == "test":   
            w_pruned = self.weight * weight_mask.view(1, -1, 1, 1)
            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias * bias_mask

        else:
            raise Exception("[ERROR] The mode " + str(mode) + " is not supported!")

        return F.conv2d(input=x, weight=w_pruned, bias=b_pruned, stride=self.stride, padding=self.padding)

    def init_mask_parameters(self):
        nn.init.kaiming_uniform_(self.w_m.view(1,-1), a=math.sqrt(5))
        if self.bias is not None:
            nn.init.kaiming_uniform_(self.b_m.view(1,-1), a=math.sqrt(5))
