###################################################################################################
# Library Imports                                                                                 #
###################################################################################################
import os 
import random 
from PIL import Image
import numpy as np
from skimage.transform import resize
from torchvision import transforms
from skimage.color import rgb2lab
import torch
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
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
# Arguments                                                                                       #
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
parser.add_argument('--resume', type=str, default='BetaGamma/gamma_2023-11-26_03-05-37loss/chkpt_10/cp-2.pth',
                    help='file from which to resume training') #TODO: CHANGE THIS TO THE PATH OF THE MODEL YOU WANT TO USE
parser.add_argument('--i', type=str, default='000000002532.jpg',
                    help='image to be tested')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
###################################################################################################
# Model
###################################################################################################

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


MODEL = 'gamma' # TODO: CHANGE THIS TO 'gamma' TO USE GAMMA MODEL
###################################################################################################
# Variables
###################################################################################################
current_time = "2023-11-26_01-12-26" #TODO: CHANGE THIS TO THE CURRENT TIME
dir_summary = './BetaGamma/'
dir_model = "test_"+ dir_summary+ MODEL + '_' + current_time
log_path = dir_model+'/logs/'
checkpoint_dir = dir_model+'/chkpt_'+str(args.batch_size)
checkpoint_path = checkpoint_dir+'/cp-{epoch:04d}.ckpt'

if MODEL == 'beta':
  model = BetaModel()
elif MODEL == 'gamma':
  model = GammaModel()
  inception = load_model()
  inception.to(device)
model.load_state_dict(torch.load(args.resume))
model.to(device)

optimizer = optim.Adam(params = model.parameters())


model.eval()
criterion = nn.MSELoss()
writer = SummaryWriter()
###################################################################################################
# Image Processing                                                                                #
###################################################################################################

preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])

transform = transforms.Compose([
                transforms.Resize((256, 256)),
            ])

###################################################################################################
# Load Images                                                                                     #
###################################################################################################

path_color = "Data_mini/images/mycoco_val2017"
#lista_img_color = os.listdir(path_color)

lista_img_color = [args.i]

random.shuffle(lista_img_color)
lista_img_color = lista_img_color[:2]
images = [np.array(Image.open(os.path.join(path_color, im)).convert('RGB').resize((256, 256))) for im in lista_img_color]
images_bw = [np.array(Image.open(os.path.join(path_color, im)).convert('L').resize((256, 256))) for im in lista_img_color]
cubo_original = np.stack(images)
cubo_original_bw = np.stack(images_bw)
cubo_imagen_procesado = np.zeros((len(lista_img_color), 256, 256,3))

for i, im in enumerate(lista_img_color):
    imageRGB = Image.open(os.path.join(path_color,im)).convert('RGB')
    imageRGB_np = np.array(imageRGB)
    newimageRGB = resize(imageRGB_np, (256,256))
    input_tensor = preprocess(imageRGB)
    image = transform(imageRGB)
    image = rgb2lab(image).astype('float32')
    cubo_imagen_procesado[i] = image
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cubo_imagen_procesado = torch.from_numpy(cubo_imagen_procesado)

rgb_ima = np.zeros_like(cubo_imagen_procesado)

for i, image in enumerate(cubo_imagen_procesado):
    L = (image[...,0:1]/50.)-1. 
    ab = image[...,1:]/128.
    L = L.float().unsqueeze(0).to(device) 
    L = L.transpose(1,3)
    ab = ab.float().unsqueeze(0)
    ab = ab.transpose(1,3)
    print(L.shape)
    print(ab.shape)
    im_l = image.float().unsqueeze(0).permute(0, 3, 1, 2).to(device)
    print(im_l.shape)
    with torch.no_grad():
      if MODEL == 'beta':
        outputs = model(L) 
      elif MODEL == 'gamma':
        with torch.no_grad():
          embed = inception(im_l)
        outputs = model(L, embed)
    newL = (L+1)*50.0
    newAB = (outputs) * 128.0
    lb_im = torch.cat((newL,newAB), dim=1)

    lb_im = lb_im.to(device)
    lb_im = lb_im.cpu().detach().numpy()
    lb_im = lb_im.transpose(0,3,2,1)
    
    rgb_ima[i] = lab2rgb(lb_im)

fig, axs = plt.subplots(len(lista_img_color), 3, figsize=(10, 5))
if len(lista_img_color) == 1:
    axs = np.expand_dims(axs, 0)
axs[0, 0].set_title('Grayscale Image')
axs[0, 1].set_title('Original')
axs[0, 2].set_title('Predicted')
for i in range(len(lista_img_color)):
    axs[i,0].imshow(cubo_original_bw[i], cmap='gray')
    axs[i,1].imshow(cubo_original[i])
    axs[i,2].imshow(rgb_ima[i])
    axs[i,0].axis('off')
    axs[i,1].axis('off')
    axs[i,2].axis('off')
plt.tight_layout()
plt.savefig('color_pipeline.png')