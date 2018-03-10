import numpy as np
import torch, argparse, os, math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from tensorboardX import SummaryWriter

from utils import cuda, Weight_EMA_Update
from datasets.datasets import return_data
from model import ToyNet

class Solver(object):

    def __init__(self, args):
        self.args = args

        self.cuda = (args.cuda and torch.cuda.is_available())
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.eps = 1e-9
        self.K = args.K
        self.beta = args.beta
        self.num_avg = args.num_avg

        self.toynet = cuda(ToyNet(self.K), self.cuda)
        self.toynet_ema = Weight_EMA_Update(cuda(ToyNet(self.K), self.cuda),\
                self.toynet.state_dict(), decay=0.999)

        self.optim = optim.Adam(self.toynet.parameters(),lr=self.lr,betas=(0.5,0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optim,gamma=0.97)

        self.history = dict()
        self.history['avg_acc']=0.
        self.history['info_loss']=0.
        self.history['class_loss']=0.
        self.history['total_loss']=0.
        self.history['epoch']=0
        self.history['iter']=0

        self.env_name = args.env_name

        # Tensorboard Visualization
        self.global_iter = 0
        self.summary_dir = os.path.join(args.summary_dir,args.env_name)
        if not os.path.exists(self.summary_dir) : os.makedirs(self.summary_dir)
        self.tf = SummaryWriter(log_dir=self.summary_dir)
        self.tf.add_text(tag='argument',text_string=str(args),global_step=self.global_epoch)

        # Dataset init
        self.data_loader = return_data(args)

    def set_mode(self,mode='train'):
        if mode == 'train' :
            self.toynet.train()
            self.toynet_ema.model.train()
        elif mode == 'eval' :
            self.toynet.eval()
            self.toynet_ema.model.eval()
        else : raise('mode error. It should be either train or eval')

    def train(self):
        self.set_mode('train')
        for e in range(self.epoch) :
            self.global_epoch += 1

            IZY = []
            IZX = []
            CLASS_LOSS = []
            INFO_LOSS = []
            TOTAL_LOSS = []
            ACCURACY = []
            AVG_ACCURACY = []
            for idx, (images,labels) in enumerate(self.data_loader['train']):
                self.global_iter += 1

                x = cuda(Variable(images), self.cuda)
                y = cuda(Variable(labels), self.cuda)
                (mu, std), logit = self.toynet(x)

                class_loss = F.cross_entropy(logit,y).div(math.log(2))
                info_loss = -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum(1).mean().div(math.log(2))
                total_loss = class_loss + self.beta*info_loss

                izy_bound = math.log(10,2) - class_loss
                izx_bound = info_loss

                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()
                self.toynet_ema.update(self.toynet.state_dict())

                prediction = F.softmax(logit,dim=1).max(1)[1]
                accuracy = torch.eq(prediction,y).float().mean()

                if self.num_avg != 0 :
                    _, avg_soft_logit = self.toynet(x,self.num_avg,softmax=True)
                    avg_prediction = F.softmax(avg_soft_logit,dim=1).max(1)[1]
                    avg_accuracy = torch.eq(avg_prediction,y).float().mean()
                else : avg_accuracy = Variable(torch.zeros(accuracy.size()))

                IZY.append(izy_bound.data)
                IZX.append(izx_bound.data)
                CLASS_LOSS.append(class_loss.data)
                INFO_LOSS.append(info_loss.data)
                TOTAL_LOSS.append(total_loss.data)
                ACCURACY.append(accuracy.data)
                AVG_ACCURACY.append(avg_accuracy.data)

                if self.global_iter % 100 == 0 :
                    print('i:{} IZY:{:.2f} IZX:{:.2f}'
                            .format(idx+1, izy_bound.data[0], izx_bound.data[0]), end=' ')
                    print('acc:{:.4f} avg_acc:{:.4f}'
                            .format(accuracy.data[0], avg_accuracy.data[0]), end=' ')
                    print('err:{:.4f} avg_err:{:.4f}'
                            .format(1-accuracy.data[0], 1-avg_accuracy.data[0]))



            IZY = torch.cat(IZY).mean()
            IZX = torch.cat(IZX).mean()
            CLASS_LOSS = torch.cat(CLASS_LOSS).mean()
            INFO_LOSS = torch.cat(INFO_LOSS).mean()
            TOTAL_LOSS = torch.cat(TOTAL_LOSS).mean()
            ACCURACY = torch.cat(ACCURACY).mean()
            AVG_ACCURACY = torch.cat(AVG_ACCURACY).mean()
            print('e:{} IZY:{:.2f} IZX:{:.2f}'
                    .format(e+1, IZY, IZX), end=' ')
            print('acc:{:.4f} avg_acc:{:.4f}'
                    .format(ACCURACY, AVG_ACCURACY), end=' ')
            print('err:{:.4f} avg_err:{:.4f}'
                    .format(1-ACCURACY, 1-AVG_ACCURACY))
            print()

            if (self.global_epoch % 2) == 0 : self.scheduler.step()
            self.test()
        print(" [*] Training Finished!")

    def test(self):
        self.set_mode('eval')

        class_loss = 0
        info_loss = 0
        total_loss = 0
        total_num = 0
        izy_bound = 0
        izx_bound = 0
        correct = 0
        avg_correct = 0
        for idx, (images,labels) in enumerate(self.data_loader['test']):

            x = cuda(Variable(images), self.cuda)
            y = cuda(Variable(labels), self.cuda)
            (mu, std), logit = self.toynet_ema.model(x)

            class_loss += F.cross_entropy(logit,y,size_average=False).div(math.log(2))
            info_loss += -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum().div(math.log(2))
            total_loss += class_loss + self.beta*info_loss
            total_num += y.size(0)

            izy_bound += math.log(10,2) - class_loss
            izx_bound += info_loss

            prediction = F.softmax(logit,dim=1).max(1)[1]
            correct += torch.eq(prediction,y).float().sum()

            if self.num_avg != 0 :
                _, avg_soft_logit = self.toynet(x,self.num_avg,softmax=True)
                avg_prediction = F.softmax(avg_soft_logit,dim=1).max(1)[1]
                avg_correct += torch.eq(avg_prediction,y).float().sum()
            else :
                avg_correct = Variable(torch.zeros(correct.size()))

        accuracy = correct/total_num
        avg_accuracy = avg_correct/total_num

        izy_bound /= total_num
        izx_bound /= total_num
        class_loss /= total_num
        info_loss /= total_num
        total_loss /= total_num

        print('[TEST RESULT]')
        print('e:{} IZY:{:.2f} IZX:{:.2f}'
                .format(self.global_epoch, izy_bound.data[0], izx_bound.data[0]), end=' ')
        print('acc:{:.4f} avg_acc:{:.4f}'
                .format(accuracy.data[0], avg_accuracy.data[0]), end=' ')
        print('err:{:.4f} avg_erra:{:.4f}'
                .format(1-accuracy.data[0], 1-avg_accuracy.data[0]))
        print()

        if self.history['avg_acc'] < avg_accuracy :
            self.history['avg_acc'] = avg_accuracy
            self.history['class_loss'] = class_loss
            self.history['info_loss'] = info_loss
            self.history['total_loss'] = total_loss
            self.history['epoch'] = self.global_epoch
            self.history['iter'] = self.global_iter

        self.set_mode('train')
