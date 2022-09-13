import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict
import math

import ipdb

import time

def percentile(scores, sparsity):
    k = 1 + round(.01 * float(sparsity) * (scores.numel() - 1))
    return scores.view(-1).kthvalue(k).values.item()

class GetSubweightFaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):
        k_val = percentile(scores, sparsity*100)
        return torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None

def get_none_masks(model):
        none_masks = {}
        for name, module in model.named_modules():
            if isinstance(module, SubweightLinear) or isinstance(module, SubweightConv2d):
                none_masks[name + '.weight'] = None
                none_masks[name + '.bias'] = None

class SubweightLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, sparsity=0.5, trainable=True):
        super(self.__class__, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.sparsity = sparsity
        self.trainable = trainable

        # Mask Parameters of Weights and Bias
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.weight.shape), torch.ones(self.weight.shape)
        if bias:
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.bias.shape), torch.ones(self.bias.shape)
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        # self.init_mask_parameters()        

        if trainable == False:
            raise Exception("Non-trainable version is not yet implemented")

    def forward(self, x, weight_mask=None, bias_mask=None, mode="train"):
        w_pruned, b_pruned = None, None

        # clock1 = time.time()
        # If training, Get the subweight by sorting the scores
        if mode == "train":           
            self.weight_mask = GetSubweightFaster.apply(self.weight.abs(), self.zeros_weight, self.ones_weight, self.sparsity)
            # w_pruned = self.weight_mask * self.weight
            # b_pruned = None
            # if self.bias is not None:
            #     self.bias_mask = GetSubweightFaster.apply(self.bias.abs(), self.zeros_bias, self.ones_bias, self.sparsity)
            #     b_pruned = self.bias_mask * self.bias

        # If inference/valid, use the last compute masks/subweights
        if mode == "valid":           
            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias_mask * self.bias

        # If inference, no need to compute the subweight
        elif mode == "test":   
            w_pruned = weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                b_pruned = bias_mask * self.bias

        # return F.linear(input=x, weight=w_pruned, bias=b_pruned)
        return F.linear(x, weight=self.weight, bias=self.bias)

    # def init_mask_parameters(self):
    #     nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #     if self.bias is not None:
    #         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in)
    #         nn.init.uniform_(self.b_m, -bound, bound)

class SubweightConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, sparsity=0.5, trainable=True):
        super(self.__class__, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.stride = stride
        # self.padding = padding
        self.sparsity = sparsity
        self.trainable = trainable
        
        # Mask Parameters of Weight and Bias
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.weight.shape), torch.ones(self.weight.shape)

        if bias:
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.bias.shape), torch.ones(self.bias.shape)
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        # self.init_mask_parameters()

        if trainable == False:
            raise Exception("Non-trainable version is not yet implemented")

    def forward(self, x, weight_mask=None, bias_mask=None, mode="train"):
        w_pruned, b_pruned = None, None

        # If training, Get the subweight by sorting the scores
        if mode == "train":           
            self.weight_mask = GetSubweightFaster.apply(self.weight.abs(), self.zeros_weight, self.ones_weight, self.sparsity)
            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = GetSubweightFaster.apply(self.bias.abs(), self.zeros_bias, self.ones_bias, self.sparsity)
                b_pruned = self.bias_mask * self.bias

        # If inference/valid, use the last compute masks/subweights
        elif mode == "valid":           
            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias_mask * self.bias

        # If inference/test, no need to compute the subweight
        elif mode == "test": 
            w_pruned = weight_mask * self.weight
            # print(torch.sum(w_pruned))
            b_pruned = None
            if self.bias is not None:
                b_pruned = bias_mask * self.bias

        else:
            raise Exception("[ERROR] The mode " + str(mode) + " is not supported!")

        return F.conv2d(input=x, weight=w_pruned, bias=b_pruned, stride=self.stride, padding=self.padding)

    # def init_mask_parameters(self):
    #     nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #     if self.bias is not None:
    #         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in)
    #         nn.init.uniform_(self.b_m, -bound, bound)