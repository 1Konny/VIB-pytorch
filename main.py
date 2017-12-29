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

from utils.utils import One_Hot
from utils.visdom_utils import VisFunc
from utils.model import encoder, decoder

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)

np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)

class VIB(object):
    def __init__(self, args):
        # Basic Hyperparameters init
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.D_lr = args.D_lr
        self.E_lr = args.E_lr
        self.eps = 1e-9
        self.K = args.K
        self.beta = args.beta
        self.many = args.many
        # Geneorator & Discriminator & Encoder init
        ''' encoder softplus? '''
        self.D = decoder(self.K).cuda()
        self.D.weight_init()
        self.E = encoder(self.K).cuda()
        self.E.weight_init()
        # Optimizers & Learning Rate Schedulers
        self.ED_optim = optim.Adam([{'params':self.E.parameters(),
                                     'lr':self.E_lr},
                                    {'params':self.D.parameters(),
                                     'lr':self.D_lr}], betas=(0.5, 0.999))
        self.ED_scheduler = lr_scheduler.ExponentialLR(self.ED_optim,gamma=0.97)

        # Binary Cross-Entropy Loss
        self.CE_Loss = nn.CrossEntropyLoss()
        self.KLDiv_Loss = nn.KLDivLoss()
        self.CE_Loss_test = nn.CrossEntropyLoss(size_average=False)
        self.KLDiv_Loss_test = nn.KLDivLoss(size_average=False)

        # History
        self.history = dict()
        self.history['top_acc']=0.
        self.history['epoch']=0
        self.history['info_loss']=0.
        self.history['class_loss']=0.
        self.history['total_loss']=0.

        # One Hot module(convert labels to onehot vectors)
        self.onehot_module = One_Hot(10)

        # Visdom Sample Visualization
        self.env_name = args.env_name
        self.vf = VisFunc(enval=self.env_name,port=55558)

        # Tensorboard Visualization
        self.global_epoch = 0
        self.summary_dir = os.path.join(args.summary_dir,args.env_name)
        self.output_dir = os.path.join(args.output_dir,args.env_name)
        if not os.path.exists(self.summary_dir) : os.makedirs(self.summary_dir)
        #if not os.path.exists(self.output_dir) : os.makedirs(self.output_dir)
        #self.tf = SummaryWriter(log_dir=self.summary_dir)
        #self.tf.add_text(tag='argument',text_string=str(args),global_step=self.global_epoch)

        # Dataset init
        self.train_data = MNIST(root='data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True,)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=5,
                                       drop_last=True)
        self.test_data = MNIST(root='data',
                                train=False,
                                transform=transforms.ToTensor(),
                                download=True)
        self.test_loader = DataLoader(self.test_data,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       num_workers=5,
                                       drop_last=False)

    def set_mode(self,mode='train'):
        if mode == 'train' :
            self.D.train()
            self.E.train()
        elif mode == 'eval' :
            self.D.eval()
            self.E.eval()
        else : raise('mode error. It should be either train or eval')

    def scale_image(self, image):
        return image.mul(2).add(-1)

    def unscale_image(self, image):
        return image.add(1).mul(0.5)

    def train(self):
        self.set_mode('train')
        for e in range(self.epoch) :
            Class_Losses = []
            Info_Losses = []
            Total_Losses = []
            Accs = []
            for batch_idx, (images,labels) in enumerate(self.train_loader):

                x = Variable(images.cuda())
                x = self.scale_image(x)
                #y = self.onehot_module(Variable(labels)).cuda()
                labels = Variable(labels).cuda()
                z = self.E(x)
                y_ = self.D(z)
                prior = Variable(torch.randn(z.size())).cuda()

                ''' KLD? IZY? IZX? '''
                class_loss = self.CE_Loss(y_,labels).div(math.log(2))
                info_loss = self.KLDiv_Loss(z,prior).div(math.log(2))
                total_loss = class_loss + self.beta*info_loss
                IZY = math.log(10,2) - class_loss
                IZX = info_loss

                self.ED_optim.zero_grad()
                total_loss.backward()
                self.ED_optim.step()

                acc = torch.eq(y_.max(1)[1],labels).float().mean()
                Accs.append(acc.data)
                Total_Losses.append(total_loss.data)
                Info_Losses.append(info_loss.data)
                Class_Losses.append(class_loss.data)
                if batch_idx % 100 == 0 :
                    print('[{:03d}:{:03d}] class_loss:{:.3f} info_loss:{:.3f} total_loss:{:.3f} accuracy:{:.3f}%'.format(self.global_epoch,batch_idx,class_loss.data[0],info_loss.data[0],total_loss.data[0],acc.data[0]*100))

            Accs = torch.cat(Accs).mean()
            Total_Losses = torch.cat(Total_Losses).mean()
            Info_Losses = torch.cat(Info_Losses).mean()
            Class_Losses = torch.cat(Class_Losses).mean()
            print('[{:03d}] avg_class_loss:{:.3f} avg_info_loss:{:.3f} avg_total_loss:{:.3f} avg_accuracy:{:.3f}%'.format(self.global_epoch,Class_Losses,Info_Losses,Total_Losses,Accs*100))

            self.global_epoch += 1
            if (self.global_epoch%2)==0 : self.ED_scheduler.step()
            self.test()
        print(" [*] Training Finished!")

    def test(self):
        self.set_mode('eval')
        class_loss = 0.
        info_loss = 0.
        total_loss = 0.
        total_samples = 0.
        ACC = []
        AVG_ACC = []
        for batch_idx, (images,labels) in enumerate(self.test_loader):
            x = Variable(images.cuda())
            x = self.scale_image(x)
            labels = Variable(labels).cuda()
            z = self.E(x)
            y_ = self.D(z)
            prior = Variable(torch.randn(z.size())).cuda()

            class_loss += self.CE_Loss_test(y_,labels).div(math.log(2)).data[0]
            info_loss += self.KLDiv_Loss_test(z,prior).div(math.log(2)).data[0]
            total_loss += class_loss + self.beta*info_loss

            acc = torch.eq(y_.max(1)[1],labels)
            ACC.append(acc.data)
            total_samples += x.size(0)

        ACC = torch.cat(ACC).float().mean()
        class_loss /= total_samples
        info_loss /= total_samples
        total_loss /= total_samples

        if self.history['top_acc'] < ACC :
            self.history['top_acc'] = ACC
            self.history['class_loss'] = class_loss
            self.history['info_loss'] = info_loss
            self.history['total_loss'] = total_loss
            self.history['epoch'] = self.global_epoch

        print()
        print('[TEST] accuracy:{:.3f}% top_acc:{:.3f}% at {:d}'.format(ACC*100,self.history['top_acc']*100,self.history['epoch']))
        print()
        self.set_mode('train')


def main():
    parser = argparse.ArgumentParser(description='TOY VIB')
    parser.add_argument('--epoch', default = 200, type=int, help='epoch size')
    parser.add_argument('--D_lr', default = 1e-4, type=float, help='learning rate for the Discriminator')
    parser.add_argument('--E_lr', default = 1e-4, type=float, help='learning rate for the Encoder')
    parser.add_argument('--beta', default = 1e-3, type=float, help='beta')
    parser.add_argument('--K', default = 256, type=int, help='dimension of K')
    parser.add_argument('--many', default = 12, type=int, help='many')
    parser.add_argument('--batch_size', default = 100, type=int, help='batch size')
    parser.add_argument('--env_name', default='main', type=str, help='visdom env name')
    parser.add_argument('--summary_dir', default='summary', type=str, help='summary directory name')
    parser.add_argument('--output_dir', default='output', type=str, help='output directory name')
    args = parser.parse_args()

    net = VIB(args)
    net.test()
    net.train()
    return 0

if __name__ == "__main__":
    main()
