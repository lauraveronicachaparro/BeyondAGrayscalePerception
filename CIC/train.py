#############################################################################
# Imports
#############################################################################

import argparse
import os
from tqdm import tqdm
from colorizers import *
from Data import COCOTrainDataset, COCOValDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import datetime
import warnings

#############################################################################
# Arguments
#############################################################################
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs(f'models/{current_time}', exist_ok=True)

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
warnings.filterwarnings("ignore")
#############################################################################
# DataLoader
#############################################################################

train_folder = "../Data_mini/images/mycoco_train2017"

train_dataset = COCOTrainDataset(data_path=train_folder, transform=transforms.Compose([
    transforms.Resize((256, 256)),
]))
train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size * torch.cuda.device_count(),  # Adjusted batch size
                              shuffle=True, **kwargs)
print('Data loaded.')
#############################################################################
# Models
#############################################################################

colorizer_eccv16 = eccv16(pretrained=True)
colorizer_siggraph17 = siggraph17(pretrained=True)

if args.cuda:
    colorizer_eccv16 = nn.DataParallel(colorizer_eccv16)
    colorizer_siggraph17 = nn.DataParallel(colorizer_siggraph17)

colorizer_eccv16.to(device)
colorizer_siggraph17.to(device)

loss_fn_eccv16 = nn.MSELoss()
loss_fn_siggraph17 = nn.MSELoss()

opt_eccv16 = optim.Adam(colorizer_eccv16.parameters(), lr=3.16e-5, betas=(0.9, 0.99), weight_decay=0.001)
opt_siggraph17 = optim.Adam(colorizer_siggraph17.parameters(), lr=3.16e-5, betas=(0.9, 0.99), weight_decay=0.001)

#############################################################################
# Functions
#############################################################################

print('Training model, press Ctrl+C to abort...')

##################################################################################
# Training
##################################################################################

best_loss_eccv = None
best_loss_siggraph = None
save_index = 0

for epoch in range(args.epochs):
    print(f"Epoch: {epoch}")

    for i, batch in enumerate(tqdm(train_dataloader)):
        img_pt = batch['img_pt'].to(device)
        tens_l_orig = batch['tens_l_orig'].to(device)
        tens_l_rs = batch['tens_l_rs'].to(device)

        # Forward pass
        out_eccv16 = colorizer_eccv16(tens_l_rs)
        out_siggraph17 = colorizer_siggraph17(tens_l_rs)

        out_eccv16 = postprocess_tens(tens_l_orig, out_eccv16)
        out_siggraph17 = postprocess_tens(tens_l_orig, out_siggraph17)

        out_eccv16 = torch.from_numpy(out_eccv16).float().to(device).requires_grad_()
        out_siggraph17 = torch.from_numpy(out_siggraph17).float().to(device).requires_grad_()

        # Loss
        loss_eccv16 = loss_fn_eccv16(out_eccv16, img_pt)
        loss_siggraph17 = loss_fn_siggraph17(out_siggraph17, img_pt)

        # Backward pass
        opt_eccv16.zero_grad()
        loss_eccv16.backward()
        opt_eccv16.step()

        opt_siggraph17.zero_grad()
        loss_siggraph17.backward()
        opt_siggraph17.step()

        print(
            f"Epoch: {epoch}")

        # SAVE ECCV16
        epoch_vl = loss_eccv16.item()
        if best_loss_eccv is None or epoch_vl < best_loss_eccv:
            best_loss_eccv = epoch_vl
            with open(f"models/{current_time}/{i}_colorizer_eccv16.pth", "wb") as fp:
                torch.save(colorizer_eccv16.state_dict(), fp)

        # SAVE SIGGRAPH17
        epoch_vls = loss_siggraph17.item()
        if best_loss_siggraph is None or epoch_vls < best_loss_siggraph:
            best_loss_siggraph = epoch_vls
            with open(f"models/{current_time}/{i}_colorizer_siggraph17.pth", "wb") as fp:
                torch.save(colorizer_siggraph17.state_dict(), fp)
        save_index += 1

#############################################################################
# Saving Models
#############################################################################

torch.save(colorizer_eccv16.state_dict(), f'models/{current_time}final_colorizer_eccv16.pth')
torch.save(colorizer_siggraph17.state_dict(), f'models/{current_time}/final_colorizer_siggraph17.pth')