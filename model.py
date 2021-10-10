from model_utils import GeM, Backbone, mixup, cutmix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import pytorch_lightning as pl
import numpy as np
import wandb

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_default_transforms():
    transform = {
        'train': T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
        'val': T.Compose(
            [
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
    }
    return transform

class PetFinderModel(pl.LightningModule):
    def __init__(
        self, model_name, epochs, lr, wd, 
        accumulate_grad_batches, 
        drop_rate, drop_path_rate,
        mixup, mixup_p, mixup_alpha,
        cutmix, cutmix_p, cutmix_alpha,
        classification=True, 
        pretrained=False
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained']);
        
        self.backbone = Backbone(model_name, pretrained=pretrained, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(self.backbone.out_features, 1),
        )

        self.classification = classification
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        if self.classification:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()

        tfms = get_default_transforms()
        self.train_tfms = tfms['train']
        self.val_tfms = tfms['val']

        self.best_bce_loss = None
        self.best_rmse_loss = None

        transformer_models = ['swin', 'vit', 'xcit', 'cait', 'mixer', 'resmlp']
        if any([t in model_name for t in transformer_models]):
            self.transformer = True
        else:
            self.transformer = False

    def forward(self, x):
        x = self.backbone(x)
        if not self.transformer:
            x = self.global_pool(x).squeeze()
        return self.head(x)

    def setup(self, stage):
        if stage == 'fit':
            train_batches = len(self.train_dataloader())
            self.train_steps = (self.hparams.epochs * train_batches) // self.hparams.accumulate_grad_batches

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.train_steps, eta_min=1e-6),
                'monitor': 'val_loss',
                'interval': 'step'
            }
        }

    def training_step(self, batch, batch_idx):
        images, targets = batch['images'], batch['targets']
        images = self.train_tfms(images)

        if self.hparams.mixup and torch.rand(1)[0] < self.hparams.mixup_p:
            mix_images, target_a, target_b, lam = mixup(images, targets/100., alpha=self.hparams.mixup_alpha)
            logits = self(mix_images).squeeze(1)
            loss = self.bce_loss(logits, target_a) * lam + \
                (1 - lam) * self.bce_loss(logits, target_b)
        elif self.hparams.cutmix and torch.rand(1)[0] < self.hparams.cutmix_p:
            mix_images, target_a, target_b, lam = cutmix(images, targets/100., alpha=self.hparams.mixup_alpha)
            logits = self(mix_images).squeeze(1)
            loss = self.bce_loss(logits, target_a) * lam + \
                (1 - lam) * self.bce_loss(logits, target_b)
        else:
            logits = self(images)
            loss = self.bce_loss(logits.squeeze(1), targets/100.)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch['images'], batch['targets']
        images = self.val_tfms(images)

        logits = self(images)
        bce_logits, rmse_logits = (logits, torch.sigmoid(logits)*100) if self.classification else (0, logits)

        return {
            'preds': torch.sigmoid(logits) if self.classification else logits,
            'targets': targets,
            'bce_logits': bce_logits,
            'rmse_logits': rmse_logits,
        }

    def validation_epoch_end(self, val_step_outputs):
        flattened_preds = torch.cat([v['preds'] for v in val_step_outputs], dim=0)
        flattened_preds_np = flattened_preds.detach().cpu().numpy()
        self.logger.experiment.log({
            'val_logits': wandb.Histogram(flattened_preds_np[~np.isnan(flattened_preds_np)]),
            'global_step': self.global_step
        })

        bce_logits = []
        rmse_logits = []
        targets = []
        for out in val_step_outputs:
            bce_logits.append(out['bce_logits'])
            rmse_logits.append(out['rmse_logits'])
            targets.append(out['targets'])

        bce_logits, rmse_logits, targets = torch.cat(bce_logits), torch.cat(rmse_logits), torch.cat(targets)

        bce_loss = self.bce_loss(bce_logits.squeeze().detach(), targets/100.)
        rmse_loss = torch.sqrt(((rmse_logits.squeeze().detach() - targets) ** 2).mean())
        self.log('val_bce_loss', bce_loss, prog_bar=True)
        self.log('val_rmse_loss', rmse_loss, prog_bar=True)

        if self.best_bce_loss is not None:
            if bce_loss < self.best_bce_loss:
                self.best_bce_loss = bce_loss
        else:
            self.best_bce_loss = bce_loss
        if self.best_rmse_loss is not None:
            if rmse_loss < self.best_rmse_loss:
                self.best_rmse_loss = rmse_loss
        else:
            self.best_rmse_loss = rmse_loss

    def predict_step(self, batch, batch_idx, dataloader_idx):
        images = batch['images']
        images = self.val_tfms(images)

        logits = self(images)

        return torch.sigmoid(logits)*100. if self.classification else logits

class PetFinderEmbeddingsModel(PetFinderModel):
    def __init__(
        self, model_name, epochs, lr, wd, 
        accumulate_grad_batches, 
        drop_rate, drop_path_rate,
        mixup, mixup_p, mixup_alpha,
        cutmix, cutmix_p, cutmix_alpha,
        classification=True, 
        pretrained=False
    ):
        super().__init__(
            model_name, epochs, lr, wd, 
            accumulate_grad_batches, 
            drop_rate, drop_path_rate,
            mixup, mixup_p, mixup_alpha,
            cutmix, cutmix_p, cutmix_alpha,
            classification, 
            pretrained
        )

    def forward(self, x):
        return self.backbone(x)

    def predict_step(self, batch, batch_idx, dataloader_idx):
        images = batch['images']
        images = self.val_tfms(images)

        logits = self(images)

        return logits.view(logits.shape[0], -1)