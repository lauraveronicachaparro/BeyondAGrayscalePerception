#! /usr/bin/env python
###################################################################################################
# Library Imports                                                                                 #
###################################################################################################
import copy
import time
import argparse
from matplotlib import pyplot as plt
import numpy as np
import os.path as osp
from PIL import Image
import pandas as pd
from Data import COCODataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from tqdm import tqdm
import math
import os
import cv2
###################################################################################################
# Arguments
###################################################################################################
parser = argparse.ArgumentParser(description='Baseline UNet')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before '
                         'logging training status')
parser.add_argument('--save', type=str, default='models/Baseline/modeloBaseline.pt',
                    help='file on which to save model weights')
parser.add_argument('--mode', type=str, default='train',
                    help='mode that indicates if the model should be trained, tested or demostrated')
parser.add_argument('--img', type=str, default='000000000632.jpg',
                    help='image for demo')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

###################################################################################################
# Data loaders
###################################################################################################

root_dir = '../Data_mini'  # Change this based on the server you are using

dir_model = 'models/'
dir_images = 'images/'
if not os.path.exists(dir_model):
    os.mkdir(dir_model)
if not os.path.exists(dir_images):
    os.mkdir(dir_images)


train_dataset = COCODataset(root_dir, dataset_type='train',
                 transform=transforms.Compose([
                    transforms.ToTensor()])
                   )
val_dataset = COCODataset(root_dir, dataset_type='test',
                 transform=transforms.Compose([
                    transforms.ToTensor()])
                   )

train_loader = DataLoader(
    COCODataset(root_dir, dataset_type='train',
                 transform=transforms.Compose([
                    transforms.ToTensor()])
                   ),
    batch_size=args.batch_size,
    shuffle=True, **kwargs)

val_loader = DataLoader(
    COCODataset(root_dir, dataset_type='test',
                 transform=transforms.Compose([
                    transforms.ToTensor()])
                   ),
    batch_size=args.batch_size,
    shuffle=False, **kwargs)

data = next(iter(train_loader))
print(data.shape)
# breakpoint()


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = self.inner_block(1, 32)
        self.block2 = self.inner_block(32, 64)
        self.block3 = self.inner_block(64, 128)
        self.block4 = self.inner_block(128, 256)
        self.block5 = self.inner_block(256, 384)
        
    def inner_block(self, in_c, out_c, maxpool = 2):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(out_c, out_c, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # 3, 150, 150
        h1 = self.block1(x) # 32, 75, 75
        h2 = self.block2(h1) # 64, 37, 37
        h3 = self.block3(h2) # 128, 18, 18
        h4 = self.block4(h3) # 256, 9, 9
        h5 = self.block5(h4) # 384, 4, 4
        
        return [h1, h2, h3, h4, h5]
    

class Decoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.inner1 = self.inner_block(384, 256, 3, 0)
        self.inner2 = self.inner_block(256, 128, 4, 1)
        self.inner3 = self.inner_block(128, 64, 3, 0)
        self.inner4 = self.inner_block(64, 32, 3, 0)
        self.inner5 = self.inner_block(32, 3, 4, 1, out = True)
        
        self.cb1 = self.conv_block(512, 256)
        self.cb2 = self.conv_block(256, 128)
        self.cb3 = self.conv_block(128, 64)
        self.cb4 = self.conv_block(64, 32)
        
        
    def inner_block(self, in_c, out_c, kernel_size, padding, out = False,):
        return  nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size = kernel_size, stride = 2, padding = padding, bias = False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(out_c, out_c, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU() if not out else nn.Sigmoid(),
            nn.Dropout(0.2) if not out else nn.Identity(),
        )
    
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
            
    def forward(self, h):
        
        # 384, 5, 5
        x = h[-1]
        x = self.inner1(x)         # 256, 9, 9 
        
        x = torch.concat([x, h[-2]], dim = 1)
        x = self.cb1(x)
        x = self.inner2(x)         # 128, 20, 20
        
        x = torch.concat([x, h[-3]], dim = 1)
        x = self.cb2(x)
        x = self.inner3(x)         # 64, 40, 40
        
        x = torch.concat([x, h[-4]], dim = 1)
        x = self.cb3(x)
        x = self.inner4(x)         # 32, 80, 80
        
        x = torch.concat([x, h[-5]], dim = 1)
        x = self.cb4(x)
        x = self.inner5(x)         # 3, 160, 160 
    
        return x



BATCH_SIZE = args.batch_size

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lr = 5e-4
        
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.history = {'t_loss' : [], 'v_loss': [], 'PSNR': []}
        self.TRAIN_BATCHES = math.ceil(len(train_dataset)/BATCH_SIZE)
        self.VAL_BATCHES = math.ceil(len(val_dataset)/BATCH_SIZE)
        
        self.grayscale = transforms.Grayscale(1)

        self.loss_fxn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        
    def forward(self, x):
        x = self.grayscale(x)
        h = self.encoder(x)    
        h = self.decoder(h)
        return h
    
    def training_step(self, X):
        pred = self.forward(X)
        loss = self.loss_fxn(pred, X)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def validation_step(self, X):
        
        with torch.no_grad():
            pred = self.forward(X)
            loss = self.loss_fxn(pred, X)
        return loss , pred

    def train(self, epochs = args.epochs):
        best_loss= None
        for epoch in tqdm(range(epochs)):
            epoch_tl, epoch_vl, epoch_psnr = 0, 0, 0
            
            for X in tqdm(train_loader):
                X = X.to(device)
                epoch_tl += self.training_step(X)
                
            for X in tqdm(val_loader):
                X = X.to(device)
                miloss, mipred = self.validation_step(X)
                epoch_vl += miloss

                mse = torch.mean((X - mipred) ** 2)
                psnr = 20 * math.log10(1.0 / math.sqrt(mse.item()))
                epoch_psnr += psnr
            
            epoch_tl /= self.TRAIN_BATCHES
            epoch_vl /= self.VAL_BATCHES
            epoch_psnr /= self.VAL_BATCHES
            
            self.history['t_loss'].append(epoch_tl.item())
            self.history['v_loss'].append(epoch_vl.item())
            self.history['PSNR'].append(epoch_psnr)

            if best_loss is None or epoch_vl < best_loss:
                best_loss = epoch_vl
                with open(args.save, 'wb') as fp:
                    state = model.state_dict()
                    torch.save(state, fp)

            with open("models/Baseline/trainloss.txt", 'a', newline='') as file:
                file.write(str(epoch_tl) + '\n')
            with open("models//Baseline/valloss.txt", 'a', newline='') as file:
                file.write(str(epoch_vl) + '\n')
            with open("models//Baseline/psnr.txt", 'a', newline='') as file:
                file.write(str(epoch_psnr) + '\n')
            
            print("[Epoch: {}] [Train loss: {:.4f}] [Val loss: {:.4f}] [PSNR: {:.4f}]".format(epoch + 1, epoch_tl, epoch_vl, epoch_psnr))
    
    def test(self):
        epoch_vl, epoch_psnr = 0, 0
    
        for X in tqdm(val_loader):
            X = X.to(device)
            miloss, mipred = self.validation_step(X)
            epoch_vl += miloss

            mse = torch.mean((X - mipred) ** 2)
            psnr = 20 * math.log10(1.0 / math.sqrt(mse.item()))
            epoch_psnr += psnr
        
        epoch_vl /= self.VAL_BATCHES
        epoch_psnr /= self.VAL_BATCHES
        
        self.history['v_loss'].append(epoch_vl.item())
        self.history['PSNR'].append(epoch_psnr)
        
        print("[Val loss: {:.4f}] [PSNR: {:.4f}]".format(epoch_vl, epoch_psnr))



def plot_images(img, x_img, pred):

    img = cv2.resize(img, (160,160))
    plt.figure(figsize = (10, 5))
    plt.subplot(1,3,1)
    plt.title('Grayscale input')
    plt.imshow(x_img[0][0].detach().numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title('Original')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title('Predicted')
    plt.imshow(pred.detach().numpy())
    plt.axis('off')

    plt.savefig('images/Baseline/' +args.img)


if __name__ == '__main__':
    best_loss = None
    if (args.mode=='train'):
        model = Net().to(device)
        model.train(epochs =args.epochs)

    elif(args.mode=='test'):
        if osp.exists(args.save):
          model = Net()
          with open(args.save, 'rb') as fp:
            state = torch.load(fp)
            model.load_state_dict(state)
            model = model.to(device)
            model.test()
        else:
            model = Net().to(device)
            model.train(epochs =args.epochs)

    elif(args.mode=='demo'):
        model = Net()
        model.load_state_dict(torch.load(args.save))
        

        val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((150, 150)),
        ])

        img = plt.imread(os.path.join("Dataminis/images/mycoco_val2017",args.img))[:, :, :3]
        img = np.copy(img)

        x_img = model.grayscale(val_transform(img)).view(-1, 1, 150, 150)

        model = model.to('cpu')
        with torch.no_grad():
            pred = model(x_img)
            pred = pred[0]
            pred = torch.einsum('ijk->jki', pred)
        
        plot_images(img, x_img, pred)

        
