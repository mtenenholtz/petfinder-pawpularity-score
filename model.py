from model_utils import GeM, Backbone, mixup, cutmix, divide_norm_bias
from dataset import get_default_transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import pytorch_lightning as pl
import numpy as np
import wandb

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
            nn.Linear(self.backbone.out_features, 1)
        )

        self.classification = classification
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        if self.classification:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()

        self.best_bce_loss = None
        self.best_rmse_loss = None
        
        transformer_models = ['swin', 'vit', 'xcit', 'cait', 'mixer', 'resmlp', 'crossvit', 'beit']
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
            train_batches = len(self.trainer.datamodule.train_dataloader())
            self.train_steps = (self.hparams.epochs * train_batches) // self.hparams.accumulate_grad_batches

    def configure_optimizers(self):
        norm_bias_params, non_norm_bias_params = divide_norm_bias(self)
        optimizer = torch.optim.AdamW([
            {'params': norm_bias_params, 'weight_decay': self.hparams.wd},
            {'params': non_norm_bias_params, 'weight_decay': 0.},
        ], lr=self.hparams.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.train_steps, eta_min=1e-7),
                'monitor': 'val_loss',
                'interval': 'step'
            }
        }
        
    def _take_training_step(self, batch, batch_idx):
        images, targets = batch['images'], batch['targets']

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
            
        return loss
        

    def training_step(self, batch, batch_idx):
        loss = self._take_training_step(batch, batch_idx)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch['images'], batch['targets']

        logits = self(images)
        bce_logits, rmse_logits = (logits, torch.sigmoid(logits)*100) if self.classification else (0, logits)

        return {
            'preds': torch.sigmoid(logits) if self.classification else logits,
            'targets': targets,
            'bce_logits': bce_logits,
            'rmse_logits': rmse_logits,
        }

    def validation_epoch_end(self, val_step_outputs):
        flattened_preds = torch.cat([torch.atleast_2d(v['preds']) for v in val_step_outputs], dim=0)
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

    def predict_step(self, batch, batch_idx):
        if 'images_1' in batch:
            logits = 0.
            for images in batch.values():
                logits += self(images) / len(batch.values())
        else:
            images = batch['images']

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

    def predict_step(self, batch, batch_idx):
        images = batch['images']

        logits = self(images)

        return logits.view(logits.shape[0], -1)
    
class PetFinderBoostedModel(PetFinderModel):
    def __init__(
        self, model_name, epochs, lr, wd, 
        accumulate_grad_batches, 
        drop_rate, drop_path_rate,
        mixup, mixup_p, mixup_alpha,
        cutmix, cutmix_p, cutmix_alpha,
        classification=True, 
        pretrained=False, booster=None
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
        
        self.booster = booster

    def forward(self, x):
        emb = self.backbone(x)
        return emb, self.head(emb)

    def predict_step(self, batch, batch_idx):
        images = batch['images']

        emb, logits = self(images)
        booster_logits = self.booster.predict(emb.detach())
        
        logits = (logits + booster_logits) / 2

        return logits.view(logits.shape[0], -1)