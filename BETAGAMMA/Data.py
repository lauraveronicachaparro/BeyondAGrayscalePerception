import os
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize
from PIL import Image
from skimage.color import rgb2lab
from torchvision import transforms

     
# Dataset class
class COCODataset(Dataset):
    def __init__(self, data_path, transform=None,dataset_type='train'):
        self.data_path = data_path
        self.transform = transform
        self.image_files = []
        self.label_files = []
        if dataset_type == 'train':
            image_dir = 'images/mycoco_train2017'
            #label_dir = 'labels/mycoco_train2017'
        elif dataset_type == 'test':
            image_dir = 'images/mycoco_val2017'
            #label_dir = 'labels/mycoco_val2017'
        else:
            raise ValueError(f'Invalid dataset type: {dataset_type}')
        
        for root, dirs, files in os.walk(os.path.join(data_path, image_dir)):
            for file in sorted(files):
                if file.endswith('.jpg'):
                    self.image_files.append(os.path.join(root, file))
        '''
        for root, dirs, files in os.walk(os.path.join(label_dir)):
            for file in sorted(files):
                if file.endswith('.png'):
                    self.label_files.append(os.path.join(root, file))
        '''
        print(len(self.image_files))
        #print(len(self.label_files))

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.image_files)

    def __getitem__(self, idx):
        
        'Generates one sample of data'
        image_path = os.path.join(self.image_files[idx])
        imageRGB = Image.open(image_path).convert('RGB')
        imageRGB_np = np.array(imageRGB)
        newimageRGB = resize(imageRGB_np, (256,256))

        input_tensor = self.preprocess(imageRGB)


        if self.transform:
            image = self.transform(imageRGB)

        image = rgb2lab(image).astype('float32')
        
        
        L = (image[...,0:1]/50.)-1. #[0,100] -> [-1,1]
        ab = image[...,1:]/128 #[-128,128] -> [-1,1]

        #return L, ab and embed separately
        return {'L': transforms.ToTensor()(L), 'ab': transforms.ToTensor()(ab)} , input_tensor, newimageRGB