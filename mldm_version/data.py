import os

import numpy as np
import pandas as pd

from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from sklearn.model_selection import train_test_split


class MRI_Dataset(Dataset):
    def __init__(self, path_df, data_dir, transform=None):
        self.path_df = path_df
        self.transform = transform
        self.data_dir = data_dir
        
    def __len__(self):
        return self.path_df.shape[0]
    
    def __getitem__(self, idx):
        
        base_path = os.path.join(self.data_dir, self.path_df.iloc[idx]['directory'])
        img_path = os.path.join(base_path, self.path_df.iloc[idx]['images'])
        mask_path = os.path.join(base_path, self.path_df.iloc[idx]['masks'])
        
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        
        sample = (image, mask)
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
class PairedToTensor():
    def __call__(self, sample):
        img, mask = sample
        img = np.array(img)
        mask = np.expand_dims(mask, -1)
        img = np.moveaxis(img, -1, 0)
        mask = np.moveaxis(mask, -1, 0)
        img, mask = torch.FloatTensor(img), torch.FloatTensor(mask)
        img = img/255
        mask = mask/255
        return img, mask
    
class PairedRandomHorizontalFlip():
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, sample):
        img, mask = sample
        if np.random.random() < self.p:
            img, mask = TF.hflip(img), TF.hflip(mask)
            
        return img, mask
    
class PairedRandomAffine():
    
    def __init__(self, degrees= None, translate=None, scale_ranges=None,
                shears=None):
        self.params = {
            'degree': degrees,
            'translate': translate,
            'scale_ranges':scale_ranges,
            'shears':shears
        }
    def __call__(self, sample):
        img, mask = sample
        w, h = img.size
        
        angle, translations, scale, shear = transforms.RandomAffine.get_params(
            self.params['degree'], self.params['translate'],
            self.params['scale_ranges'], self.params['shears'],
            (w,h)
        )
        
        img = TF.affine(img, angle, translations, scale, shear)
        mask = TF.affine(mask, angle, translations, scale, shear)
        
        return img, mask
    
def get_train_val_datasets(data_dir, seed, validation_ratio=0.2,
                           hflip_augmentation=True, affine_augmentation=True,
                           max_rotation=15, max_translation=0.1, min_scale=0.8, max_scale=1.2):
    
    dirs, images, masks = [], [], []

    for root, folders, files in  os.walk(data_dir):
        for file in files:
            if 'mask' in file:
                dirs.append(root.replace(data_dir, ''))
                masks.append(file)
                images.append(file.replace("_mask", ""))
                
    PathDF = pd.DataFrame({'directory': dirs,
                          'images': images,
                          'masks': masks})

    train_df, valid_df = train_test_split(PathDF, random_state=seed,
                                     test_size = validation_ratio)
    
    transform_list = []
    if hflip_augmentation:
        transform_list.append(PairedRandomHorizontalFlip())
    if affine_augmentation:
        transform_list.append(PairedRandomAffine(degrees=(-max_rotation, max_rotation),
                                                 translate=(max_translation, max_translation),
                                                 scale_ranges=(min_scale, max_scale)))
    
    transform_list.append(PairedToTensor())
    
    train_transforms = transforms.Compose(transform_list)
    
    train_data = MRI_Dataset(train_df, data_dir, transform=train_transforms)
    valid_data = MRI_Dataset(valid_df, data_dir, transform=PairedToTensor())
    
    return train_data, valid_data