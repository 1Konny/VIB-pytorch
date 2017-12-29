import os
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import scipy.misc

class One_Hot(nn.Module):
    # got it from :
    # https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/
    def __init__(self, depth):
        super(One_Hot,self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth)
    def forward(self, X_in):
        X_in = X_in.long()
        return Variable(self.ones.index_select(0,X_in.data))
    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def proto_dist(code, ck):
    batch_size = ck.size(0)
    eps = 1e-9
    code = code.unsqueeze(1).repeat(1,10,1)
    neg_dist = -(code-ck).pow(2).sum(2)
    out = neg_dist.exp().div(eps+neg_dist.exp().sum(1).unsqueeze(1).repeat(1,10))
    return out
