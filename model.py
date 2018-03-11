import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from utils import cuda

import time
from numbers import Number

class ToyNet(nn.Module):

    def __init__(self, K=256):
        super(ToyNet, self).__init__()
        self.K = K

        self.encode = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2*self.K))

        self.decode = nn.Sequential(
                nn.Linear(self.K, 10))

    def forward(self, x, num_sample=1, softmax=False):
        if x.dim() > 2 : x = x.view(x.size(0),-1)
        temp = self.encode(x)

        mu = temp[:,:self.K]
        std = F.softplus(temp[:,self.K:]-5,beta=1)

        Y = 0
        softmax = True
        num_sample = 500



        a = time.time()
        encoding = self.reparametrize_n(mu,std,num_sample)
        logit = self.decode(encoding)
        if softmax : y = F.softmax(logit,dim=2).mean(0)
        else : y = logit.mean(0)
        Y=y

        b = time.time()
        ms = (b-a)*1000
        print('**{:.3f}'.format(ms))

        a = time.time()

        for i in range(num_sample):
            encoding = self.reparametrize_n(mu,std)
            logit = self.decode(encoding)
            if softmax : y = F.softmax(logit,dim=1)
            else : y = logit
            Y += y/num_sample

        b = time.time()
        ms = (b-a)*1000
        print('{:.3f}'.format(ms))
        import ipdb; ipdb.set_trace()


        return (mu, std), Y

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])


def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()
