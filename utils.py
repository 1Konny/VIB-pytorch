import torch
from torch import nn
from torch.autograd import Variable


class One_Hot(nn.Module):

    """
    codes from : https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/
    """

    def __init__(self, depth):
        super(One_Hot,self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth)

    def forward(self, X_in):
        X_in = X_in.long()
        return Variable(self.ones.index_select(0, X_in.data))

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


def str2bool(v):
    """
    codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def cuda(tensor, is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor


class Weight_EMA_Update(object):

    def __init__(self, model, initial_state_dict, decay=0.999):
        self.model = model
        self.model.load_state_dict(initial_state_dict, strict=True)
        self.decay = decay

    def update(self, new_state_dict):
        state_dict = self.model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = (self.decay)*state_dict[key] + (1-self.decay)*new_state_dict[key]
            #state_dict[key] = (1-self.decay)*state_dict[key] + (self.decay)*new_state_dict[key]

        self.model.load_state_dict(state_dict)
