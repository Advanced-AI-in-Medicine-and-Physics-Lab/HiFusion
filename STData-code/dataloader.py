import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import random

def create_dataloaders_for_each_file(npy_file_paths, batch_size=32, transform=None, shuffle=True):
    dataloaders = {}
    for npy_file in npy_file_paths:
        name = npy_file.split('/')[-1][:-4]

        dataset = ImageDataset(npy_file, transform=transform)

        # def collate_fn_with_params(batch, graph_datas=graph_data):
        #     return collate_fn(batch, graph_datas)

        dataloaders[npy_file] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=False, num_workers=4)
    return dataloaders

class DualTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img1, img2):
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img1 = self.transform(img1)

        random.seed(seed)
        torch.manual_seed(seed)
        img2 = self.transform(img2)
        return img1, img2

class ImageDataset(Dataset):
    def __init__(self, data_infor_path, root, transform=None):
        # Load data_list and graph information from .npy file
        self.data_list = np.load(data_infor_path, allow_pickle=True).tolist()
        self.root = root
        self.transform = DualTransform(transform) # ensure consistent transform
        self.name = data_infor_path.split('/')[-2] # e.g. A1

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        spot_image_path = os.path.join(self.root, self.data_list[idx]['img_path_224'])
        spot_image = Image.open(spot_image_path).convert('RGB')

        region_image_path = os.path.join(img_dir, self.data_list[idx]['img_path_1120'])
        region_image = Image.open(region_image_path).convert('RGB')

        width, height = region_image.size

        center_x = width // 2
        center_y = height // 2

        half_spot_size = 224 
        left = center_x - half_spot_size
        top = center_y - half_spot_size
        right = center_x + half_spot_size
        bottom = center_y + half_spot_size

        region_image = region_image.crop((left, top, right, bottom))

        if self.transform:
            spot_image,region_image = self.transform(spot_image,region_image)

        label = self.data_list[idx]['label']

        return spot_image,region_image, label.astype(np.float32)