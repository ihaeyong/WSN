from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb

# class MixerLayer(nn.Module):
#     def __init__(self, self, n_tokens, n_channels):


# class MixerBlock
    

class DifferentiableBinaryGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones):
        return torch.where(scores < 0.5, zeros.to(scores.device), ones.to(scores.device))
    
    @staticmethod
    def backward(ctx, g):
        return g, None, None

class TaskLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(self.__class__, self).__init__(in_features=in_features, out_features=out_features, bias=bias)

        self.tasker = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, out_features),
            nn.Sigmoid()
        )

        self.zeros, self.ones = torch.zeros(out_features), torch.ones(out_features)
        # Might need to add some regularizer after the Differentiable Binary Gate

    def forward(self, x, weight_mask=None, mode="train"):
        w_pruned, b_pruned = None, None

        # If training, Get the subnet by infering the binary gate outputs
        if mode == "train":
            scores = self.tasker(x)
            self.weight_mask = DifferentiableBinaryGate.apply(scores.mean(dim=0), self.zeros, self.ones)
            w_pruned = self.weight * self.weight_mask.view(-1, 1)

        # If inference/valid, use the last computed masks/subnetworks
        elif mode == "valid":
            w_pruned = self.weight * self.weight_mask.view(-1, 1)

        # If inference/testm no need to copute the subnetwork
        elif mode =="test":
            w_pruned = self.weight * weight_mask.view(-1,1)

        else:
            raise Exception("[ERROR] The mode " + str(mode) + " is not supported")

        return F.linear(input=x, weight=w_pruned, bias=None)

class TaskConv2d(nn.Conv2d):
    def __init__(self, dim, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias 
        )

        self.tasker = nn.Sequential(
            nn.Linear(dim*dim*in_channels, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, out_channels),
            nn.Sigmoid()
        )

        self.zeros, self.ones = torch.zeros(out_channels), torch.ones(out_channels)

    def forward(self, x, weight_mask=None, mode="train"):
        batch_size = x.shape[0]
        # ipdb.set_trace()

        # If training, Get the subnet by sorting the scores
        if mode == "train":           
            scores = self.tasker(x.view(batch_size, -1))
            self.weight_mask = DifferentiableBinaryGate.apply(scores.mean(dim=0), self.zeros, self.ones)
            w_pruned = self.weight * self.weight_mask.view(-1,1,1,1)
        
        # If inference/valid, use the last computed masks/subnetworks
        elif mode == "valid":
            w_pruned = self.weight * self.weight_mask.view(-1,1,1,1)

        # If inference/testm no need to copute the subnetwork
        elif mode =="test":
            w_pruned = self.weight * weight_mask.view(-1,1,1,1)

        else:
            raise Exception("[ERROR] The mode " + str(mode) + " is not supported")

        return F.conv2d(input=x, weight=w_pruned, bias=None, stride=self.stride, padding=self.padding)