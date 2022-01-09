from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from augmix import RandomAugMix
from utils import in_colab

import albumentations as A
import torchvision.io as io
import pytorch_lightning as pl
import torch
import cv2

def get_default_transforms(img_size):
    transform = {
        'train': A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            A.SmallestMaxSize(max_size=img_size[0], p=1),
            A.RandomCrop(height=img_size[0], width=img_size[1], p=1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2()
        ]),
        'inference': A.Compose([
            A.SmallestMaxSize(max_size=img_size[0], p=1.0),
            A.CenterCrop(height=img_size[0], width=img_size[1], p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2()
        ]),
        'tta': [
            A.Compose([
                A.SmallestMaxSize(max_size=img_size[0], p=1.0),
                A.CenterCrop(height=img_size[0], width=img_size[1], p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2()
            ]),
            A.Compose([
                A.HorizontalFlip(p=1.0),
                A.SmallestMaxSize(max_size=img_size[0], p=1.0),
                A.CenterCrop(height=img_size[0], width=img_size[1], p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2()
            ]),
            A.Compose([
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                A.SmallestMaxSize(max_size=img_size[0], p=1.0),
                A.CenterCrop(height=img_size[0], width=img_size[1], p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2()
            ]),
            A.Compose([
                A.HorizontalFlip(p=1.0),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                A.SmallestMaxSize(max_size=img_size[0], p=1.0),
                A.CenterCrop(height=img_size[0], width=img_size[1], p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2()
            ]),
        ]
    }
    return transform

class Dataset(Dataset):
    def __init__(self, img_ids, targets=None, img_size=(224, 224), inference=False, tta=False):
        self.img_ids = img_ids
        self.targets = targets
        self.tta = tta
        if tta:
            self.augs = get_default_transforms(img_size)['tta']
        elif inference:
            self.augs = get_default_transforms(img_size)['inference']
        else:
            self.augs = get_default_transforms(img_size)['train']

    def __len__(self):
        return self.img_ids.shape[0]

    def __getitem__(self, i):
        image = cv2.imread(self.img_ids[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if not self.tta:
            image = self.augs(image=image)['image']
            
        if self.targets is not None:
            target = torch.as_tensor(self.targets[i]).float()
            return {
                'images': image,
                'targets': target
            }
        else:
            if self.tta:
                return {f'images_{i}': self.augs[i](image=image)['image'] for i in range(len(self.augs))}
            else:
                return {'images': image}

class DataModule(pl.LightningDataModule):
    def __init__(
        self, data, img_size=(224, 224), 
        train_filter=None, val_filter=None, 
        batch_size=64, inference=False, tta=False
    ):
        super().__init__()
        self.data = data
        self.img_size = img_size
        self.train_filter = train_filter
        self.val_filter = val_filter
        self.batch_size = batch_size
        self.inference = inference
        
        if tta:
            self.augs = get_default_transforms(img_size)['tta']

    def setup(self, stage=None):
        if not self.inference:
            self.train_df = self.data.loc[self.train_filter, :]
            self.val_df = self.data.loc[self.val_filter, :]

    def train_dataloader(self):
        img_ids = self.train_df['file_path'].values
        targets = self.train_df['Pawpularity'].values
        train_dset = Dataset(img_ids, targets, img_size=self.img_size)
        return DataLoader(
            train_dset, shuffle=True, num_workers=2 if in_colab() else 4,
            pin_memory=True, batch_size=self.batch_size, drop_last=True
        )

    def val_dataloader(self):
        img_ids = self.val_df['file_path'].values
        targets = self.val_df['Pawpularity'].values
        val_dset = Dataset(img_ids, targets, img_size=self.img_size, inference=True)
        return DataLoader(
            val_dset, shuffle=False, num_workers=2 if in_colab() else 4,
            pin_memory=True, batch_size=self.batch_size,
        )

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        img_ids = self.data['file_path'].values
        pred_dset = Dataset(img_ids, img_size=self.img_size, inference=True, tta=False)
        return DataLoader(
            pred_dset, shuffle=False, num_workers=2 if in_colab() else 4,
            pin_memory=True, batch_size=self.batch_size,
        )