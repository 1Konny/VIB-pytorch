import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from utils import cuda

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
        for i in range(num_sample):
            encoding = self.reparametrize(mu,std)
            logit = self.decode(encoding)
            if softmax : y = F.softmax(logit,dim=1)
            else : y = logit
            Y += y/num_sample

        return (mu, std), Y

    def reparametrize(self, mu, std):
        eps = cuda(Variable(torch.randn(std.size())), std.is_cuda)

        return mu + eps*std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])


def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()
