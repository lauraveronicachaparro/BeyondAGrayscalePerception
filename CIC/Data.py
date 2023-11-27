import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from skimage.transform import resize
from colorizers import preprocess_img, load_img

class COCOTrainDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_paths = []

        for img_name in tqdm(os.listdir(data_path), desc="Loading image paths"):
            img_path = os.path.join(data_path, img_name)
            self.image_paths.append(img_path)

        print(f'Loaded {len(self.image_paths)} training image paths.')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = load_img(img_path)
        img_pt = torch.from_numpy(img.copy()).float()
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        tens_l_orig = tens_l_orig.squeeze(0)
        tens_l_rs = tens_l_rs.squeeze(0)

        return {'img_pt': img_pt, 'tens_l_orig': tens_l_orig, 'tens_l_rs': tens_l_rs}


class COCOValDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_paths = []

        for img_name in tqdm(os.listdir(data_path), desc="Loading image paths"):
            img_path = os.path.join(data_path, img_name)
            self.image_paths.append(img_path)

        print(f'Loaded {len(self.image_paths)} validation image paths.')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = load_img(img_path)
        img_pt = torch.from_numpy(img.copy()).float()
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        tens_l_orig = tens_l_orig.squeeze(0)
        tens_l_rs = tens_l_rs.squeeze(0)

        return {'img_pt': img_pt, 'tens_l_orig': tens_l_orig, 'tens_l_rs': tens_l_rs}