#############################################################################
# Imports
#############################################################################
print("Loading imports...")
import argparse
import os
from tqdm import tqdm
from colorizers import *
from Data import COCOValDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import math
print("Imports loaded.")
#############################################################################
# Arguments
#############################################################################
print("Loading arguments...")
parser = argparse.ArgumentParser(description='PyTorch Deeplab v3 Example')
parser.add_argument('--input-path', type=str, help='Path to folder of training images', default='imgs/train/mycoco_train2017/')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--batch-size', type=int, help='batch size per GPU', default=16)  # Adjusted batch size
parser.add_argument('--epochs', type=int, help='number of epochs', default=10)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#############################################################################
# DataLoader
#############################################################################
print("Loading data...")
test_folder = "images/mycoco_val2017"
test_dataset = COCOValDataset(data_path=test_folder, 
                              transform=transforms.Compose([
                                  transforms.Resize((256, 256)),
                              ]))
test_dataloader = DataLoader(test_dataset, batch_size= args.batch_size, shuffle=False)

print('Data loaded.')

#############################################################################
# Models
#############################################################################
print("Loading models...")
colorizer_eccv16 = eccv16(local_pth="models/2023-11-26_03-54-24/190_colorizer_eccv16.pth").eval()
colorizer_siggraph17 = siggraph17(local_pth="models/2023-11-26_03-54-24/474_colorizer_siggraph17.pth").eval()

print("Models loaded.")
if args.cuda:
    colorizer_eccv16 = nn.DataParallel(colorizer_eccv16)
    colorizer_siggraph17 = nn.DataParallel(colorizer_siggraph17)

print('Continue to device...')
colorizer_eccv16.to(device)
colorizer_siggraph17.to(device)

loss_fn_eccv16 = nn.MSELoss()
loss_fn_siggraph17 = nn.MSELoss()

opt_eccv16 = optim.Adam(colorizer_eccv16.parameters(), lr=3.16e-5, betas=(0.9, 0.99), weight_decay=0.001)
opt_siggraph17 = optim.Adam(colorizer_siggraph17.parameters(), lr=3.16e-5, betas=(0.9, 0.99), weight_decay=0.001)

print('Done with device.')

#############################################################################
# Functions
#############################################################################

import math

def PSNR(pred, target):
    mse = torch.mean((pred - target) ** 2)
    psnr = abs(10 * math.log10(1 / math.sqrt(mse.item())))
    return psnr

print('Testing model, press Ctrl+C to abort...')
##################################################################################
# Testing
##################################################################################

# Initialize variables to accumulate loss and PSNR
total_loss_eccv16 = 0.0
total_loss_siggraph17 = 0.0
lista_psnr_eccv16 = []
lista_psnr_siggraph17 = []

with torch.no_grad():
    for i, batch in enumerate(tqdm(test_dataloader)):
        img_pt = batch['img_pt'].to(device) # (batch_size, 256, 256,3)
        
        tens_l_orig = batch['tens_l_orig'].to(device) # (batch_size, 1, 256, 256)
        tens_l_rs = batch['tens_l_rs'].to(device) # (batch_size, 1, 256, 256)

        out_eccv16 = colorizer_eccv16(tens_l_rs) # (batch_size, 2, 256, 256)

        out_siggraph17 = colorizer_siggraph17(tens_l_rs) # (batch_size, 2, 256, 256)
        
        out_eccv16 = postprocess_tens(tens_l_orig, out_eccv16)
        out_siggraph17 = postprocess_tens(tens_l_orig, out_siggraph17)

        out_eccv16 = torch.from_numpy(out_eccv16).float().to(device).requires_grad_()
        out_siggraph17 = torch.from_numpy(out_siggraph17).float().to(device).requires_grad_()

        loss_eccv16 = loss_fn_eccv16(out_eccv16, img_pt)
        loss_siggraph17 = loss_fn_siggraph17(out_siggraph17, img_pt)

        psnr_eccv16 = PSNR(out_eccv16, img_pt)
        psnr_siggraph17 = PSNR(out_siggraph17, img_pt)

        total_loss_eccv16 += loss_eccv16.item()
        total_loss_siggraph17 += loss_siggraph17.item()
        lista_psnr_eccv16.append(psnr_eccv16)
        lista_psnr_siggraph17.append(psnr_siggraph17)

average_psnr_eccv16 = round(sum(lista_psnr_eccv16) / len(lista_psnr_eccv16),2)
average_psnr_siggraph17 = round(sum(lista_psnr_siggraph17) / len(lista_psnr_siggraph17),2)

print(f"Average PSNR ECCV16: {average_psnr_eccv16}")
print(f"Average PSNR SIGGRAPH17: {average_psnr_siggraph17}")