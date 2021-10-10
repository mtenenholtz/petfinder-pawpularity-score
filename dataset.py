from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.io as io
import pytorch_lightning as pl
import torch

class Dataset(Dataset):
    def __init__(self, img_ids, targets=None, img_size=(224, 224), interpolation=T.InterpolationMode.BILINEAR):
        self.img_ids = img_ids
        self.targets = targets
        self.resize = T.Resize(img_size, interpolation=interpolation, antialias=True)
        #self.resize = T.RandomResizedCrop(img_size)

    def __len__(self):
        return self.img_ids.shape[0]

    def __getitem__(self, i):
        image = io.read_image(self.img_ids[i]).float()
        image /= 255.
        image = self.resize(image)
        if self.targets is not None:
            target = torch.as_tensor(self.targets[i]).float()
            return {
                'images': image,
                'targets': target
            }
        else:
            return {'images': image}

class DataModule(pl.LightningDataModule):
    def __init__(
        self, data, img_size=(224, 224), 
        train_filter=None, val_filter=None, 
        batch_size=64, inference=False,
        interpolation=T.InterpolationMode.BILINEAR
    ):
        super().__init__()
        self.data = data
        self.img_size = img_size
        self.train_filter = train_filter
        self.val_filter = val_filter
        self.batch_size = batch_size
        self.inference = inference
        self.interpolation = interpolation

    def setup(self, stage=None):
        if not self.inference:
            self.train_df = self.data.loc[self.train_filter, :]
            self.val_df = self.data.loc[self.val_filter, :]

    def train_dataloader(self):
        img_ids = self.train_df['file_path'].values
        targets = self.train_df['Pawpularity'].values
        train_dset = Dataset(img_ids, targets, img_size=self.img_size, interpolation=self.interpolation)
        return DataLoader(
            train_dset, shuffle=True, num_workers=4,
            pin_memory=True, batch_size=self.batch_size,
        )

    def val_dataloader(self):
        img_ids = self.val_df['file_path'].values
        targets = self.val_df['Pawpularity'].values
        val_dset = Dataset(img_ids, targets, img_size=self.img_size, interpolation=self.interpolation)
        return DataLoader(
            val_dset, shuffle=False, num_workers=4,
            pin_memory=True, batch_size=self.batch_size,
        )

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        img_ids = self.data['file_path'].values
        pred_dset = Dataset(img_ids, img_size=self.img_size, interpolation=self.interpolation)
        return DataLoader(
            pred_dset, shuffle=False, num_workers=2,
            pin_memory=True, batch_size=self.batch_size,
        )