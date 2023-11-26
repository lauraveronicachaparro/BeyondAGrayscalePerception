#! /usr/bin/env python
###################################################################################################
# Library Imports                                                                                 #
###################################################################################################
import copy
import time
import random
import argparse
import datetime
from matplotlib import pyplot as plt
import numpy as np
import os.path as osp
from PIL import Image
import pandas as pd
from Dataloader import COCODataset
#from utils import Evaluator
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.autograd import Variable
import torch.optim.lr_scheduler as sch
from models import UnetBlock, Unet
from tqdm import tqdm
import math
import os, re
from skimage.color import rgb2lab, lab2rgb
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from torchvision.models import vgg as vgg
import torchvision

###################################################################################################
# Arguments
###################################################################################################
parser = argparse.ArgumentParser(description='PyTorch Deeplab v3 Example')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--gamma', type=float, default=2, metavar='M',
                    help='learning rate decay factor (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before '
                         'logging training status')
parser.add_argument('--save', type=str, default='models/modeloTodo20.pt',
                    help='file on which to save model weights')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

###################################################################################################
# Data loaders
###################################################################################################

root_dir = 'Data_mini'  # Change this based on the server you are using


train_dataset = COCODataset(root_dir, dataset_type='train',
                transform = transforms.Compose([
                transforms.Resize((256, 256))
            ])
                   )
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True, **kwargs)

###############################################################################################################################
# Losses
###############################################################################################################################

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if isinstance(input,np.ndarray):
            input = torch.from_numpy(input).to(self.mean.device)
        if isinstance(target,np.ndarray):
            target = torch.from_numpy(target).to(self.mean.device)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

###############################################################################################################################
# Model
###############################################################################################################################

class BetaModel(nn.Module):
    def __init__(self):
        super(BetaModel, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.upsample = nn.Upsample(scale_factor=(2,2))
        self.conv2d_1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(3,3),stride=2,padding=(1,1))
        self.conv2d_2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_3 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),stride=2,padding=(1,1))
        self.conv2d_4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=2,padding=(1,1))
        self.conv2d_6 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_7 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_8 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_9 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_10 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_11 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_12 = nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_13 = nn.Conv2d(in_channels=16,out_channels=2,kernel_size=(3,3),stride=1,padding=(1,1))

    def encoder(self, encoder_input):
        encoder_output = self.relu(self.conv2d_1(encoder_input))
        encoder_output = self.relu(self.conv2d_2(encoder_output))
        encoder_output = self.relu(self.conv2d_3(encoder_output))
        encoder_output = self.relu(self.conv2d_4(encoder_output))
        encoder_output = self.relu(self.conv2d_5(encoder_output))
        encoder_output = self.relu(self.conv2d_6(encoder_output))
        encoder_output = self.relu(self.conv2d_7(encoder_output))
        encoder_output = self.relu(self.conv2d_8(encoder_output))
        return encoder_output

    def decoder(self, decoder_input):
        decoder_output = self.relu(self.conv2d_9(decoder_input))
        decoder_output = self.upsample(decoder_output)
        decoder_output = self.relu(self.conv2d_10(decoder_output))
        decoder_output = self.upsample(decoder_output)
        decoder_output = self.relu(self.conv2d_11(decoder_output))
        decoder_output = self.relu(self.conv2d_12(decoder_output))
        decoder_output = self.tanh(self.conv2d_13(decoder_output))
        decoder_output = self.upsample(decoder_output)
        return decoder_output
    
    def forward(self, x):
      return self.decoder(self.encoder(x))
    

class GammaModel(nn.Module):
    def __init__(self):
        super(GammaModel, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.upsample = nn.Upsample(scale_factor=(2,2))
        self.conv2d_1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(3,3),stride=2,padding=(1,1))
        self.conv2d_2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_3 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),stride=2,padding=(1,1))
        self.conv2d_4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=2,padding=(1,1))
        self.conv2d_6 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_7 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_8 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=(3,3),stride=1,padding=(1,1))
        
        self.conv2d_9 = nn.Conv2d(in_channels=1256,out_channels=256,kernel_size=(1,1),stride=1,padding=(0,0))
        
        self.conv2d_10 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_11 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_12 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_13 = nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(3,3),stride=1,padding=(1,1))
        self.conv2d_14 = nn.Conv2d(in_channels=16,out_channels=2,kernel_size=(3,3),stride=1,padding=(1,1))       

    def encoder(self, encoder_input):
        encoder_output = self.relu(self.conv2d_1(encoder_input))
        encoder_output = self.relu(self.conv2d_2(encoder_output))
        encoder_output = self.relu(self.conv2d_3(encoder_output))
        encoder_output = self.relu(self.conv2d_4(encoder_output))
        encoder_output = self.relu(self.conv2d_5(encoder_output))
        encoder_output = self.relu(self.conv2d_6(encoder_output))
        encoder_output = self.relu(self.conv2d_7(encoder_output))
        encoder_output = self.relu(self.conv2d_8(encoder_output))
        return encoder_output

    def decoder(self, decoder_input):
        decoder_output = self.relu(self.conv2d_10(decoder_input))
        decoder_output = self.upsample(decoder_output)
        decoder_output = self.relu(self.conv2d_11(decoder_output))
        decoder_output = self.upsample(decoder_output)
        decoder_output = self.relu(self.conv2d_12(decoder_output))
        decoder_output = self.relu(self.conv2d_13(decoder_output))
        decoder_output = self.tanh(self.conv2d_14(decoder_output))
        decoder_output = self.upsample(decoder_output)
        return decoder_output
    
    def fusion(self, embed_input, encoder_output):
        fusion_output = embed_input.reshape([-1,1000,1,1])
        fusion_output = fusion_output.repeat(1,1,32*32,1)
        fusion_output = torch.reshape(fusion_output, (-1,1000, 32,32))
        fusion_output = torch.cat((encoder_output, fusion_output), 1)
        fusion_output = self.relu(self.conv2d_9(fusion_output))
        return fusion_output

    def forward(self, x, embed_input):
      return self.decoder(self.fusion(embed_input, self.encoder(x)))

def load_model():
  """load the classifier, use eval as the classifier is not being trained during the model training"""
  inception = models.mobilenet_v2(pretrained=True)
  inception.eval()
  return inception

MODEL = 'beta' # TODO: CHANGE THIS TO 'gamma' TO USE GAMMA MODEL

###############################################################################################################################
# Variables
###############################################################################################################################
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


dir_summary = './BetaGamma/'
dir_model = dir_summary+ MODEL + '_' + current_time
log_path = dir_model+'/logs/'
checkpoint_dir = dir_model+'/chkpt_'+str(args.batch_size)
checkpoint_path = checkpoint_dir+'/cp-{epoch:04d}.ckpt'


if not os.path.exists(dir_summary):
  os.mkdir(dir_summary)

if not os.path.exists(dir_model):
  os.mkdir(dir_model)

if not os.path.exists(log_path):
  os.mkdir(log_path)
if not os.path.exists(checkpoint_dir):
  os.mkdir(checkpoint_dir)


if MODEL == 'beta':
  model = BetaModel()
elif MODEL == 'gamma':
  model = GammaModel()
  inception = load_model()
  inception.to(device)
model.to(device)

optimizer = optim.Adam(params = model.parameters())


model.train()
criterion = nn.MSELoss()
writer = SummaryWriter()

###############################################################################################################################
# Training
###############################################################################################################################

for epoch in range(args.epochs):
    running_loss = 0.0
    best_loss = None
    for i, data in tqdm(enumerate(train_loader)):
        L, ab, input, original = data[0]['L'].to(device), data[0]['ab'].to(device), data[1].to(device), data[2].to(device)

        optimizer.zero_grad() 
        if MODEL == 'beta':
          outputs = model(L) 
        elif MODEL == 'gamma':
          with torch.no_grad():
            embed = inception(input)
          outputs = model(L, embed) 

        original = torch.cat((L, ab), dim=1)
        predicted = torch.cat((L, outputs), dim=1)
        
        loss = criterion(outputs, ab)
        loss.backward() 
        optimizer.step() 
        running_loss += loss.item()

    running_loss = running_loss/(i+1)
    print('[%d, %5d] train_loss: %.3f' %
                (epoch, i + 1, running_loss))
    
    with open(f'{dir_model}/output_train.txt', 'a') as f:
        print("[Epoch: {}] [Train loss: {:.4f}]".format(epoch + 1, running_loss), file=f)


    writer.add_scalar('loss', running_loss, epoch)
    checkpoint_path = os.path.join(checkpoint_dir, 'cp-{}.pth'.format(epoch))
    if best_loss is None or running_loss < best_loss:
      best_loss = running_loss
      with open(checkpoint_path, 'wb') as fp:
        torch.save(model.state_dict(), fp)