from dataset import DataModule
from model import PetFinderModel
from callbacks import LogPredictionsCallback
from tqdm import tqdm
from utils import in_colab

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
import pytorch_lightning as pl
import argparse
import albumentations as A
import wandb
import gc
import os

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--model_name', type=str, default='efficientnet_b0')
parser.add_argument('--fold', type=int, default=-1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--img_size_x', type=int, default=224)
parser.add_argument('--img_size_y', type=int, default=224)
parser.add_argument('--drop_rate', type=float, default=0.)
parser.add_argument('--drop_path_rate', type=float, default=0.)
parser.add_argument('--mixup_alpha', type=float, default=0.5)
parser.add_argument('--cropped_imgs', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--accumulate_grad_batches', type=int, default=1)
parser.add_argument('--grad_clip_val', type=float, default=1.0)
parser.add_argument('--interpolation', type=str, default='bilinear')
parser.add_argument('--max_epochs', type=int, default=5)
parser.add_argument('--seed', type=int, default=34)

args = parser.parse_args()

pl.seed_everything(args.seed);
wandb.login()

data_dir = 'data'
img_path = 'crop' if args.cropped_imgs else 'train'
train_df = pd.read_csv(f'{data_dir}/train_folds.csv')
train_df['file_path'] = f'{data_dir}/{img_path}/' + train_df['Id'] + '.jpg'

pseudo_image_names = [f for f in os.listdir('data/pseudo') if f.endswith('-1.jpg')]
pseudo_df = pd.DataFrame({
    'file_path': [f'data/pseudo/{f}' for f in pseudo_image_names],
    'Pawpularity': [0. for _ in pseudo_image_names]
})

hparams = {
    'model_name': args.model_name,
    'epochs': args.max_epochs,
    'lr': args.lr,
    'wd': args.wd,
    'accumulate_grad_batches': args.accumulate_grad_batches,
    'classification': True,
    'drop_rate': args.drop_rate,
    'drop_path_rate': args.drop_path_rate,
    'mixup': True,
    'mixup_p': 0.5,
    'mixup_alpha': args.mixup_alpha,
    'cutmix': False,
    'cutmix_p': 0.5,
    'cutmix_alpha': 0.5
}

n_folds = 5
for i in range(n_folds):
    if args.fold != -1 and i != args.fold:
        continue
    
    ckpt_path = '/media/mten/storage/kaggle/petfinder-pawpularity-score/ckpts/' if not in_colab() else '/content/drive/MyDrive/Kaggle/petfinder-pawpularity/ckpts/'
    ckpt_name = f'{args.model_name}-seed-{args.seed}-{args.name}-fold-{i}'

    model_paths = [p for p in os.listdir(f'{ckpt_path}/') if p.startswith(ckpt_name)]
    assert len(model_paths) == 1
    model_path = model_paths[0]
    ckpt_model = PetFinderModel.load_from_checkpoint(f'{ckpt_path}/{model_path}')
    ckpt_model.freeze();
    
    dm = DataModule(
        pseudo_df, img_size=(args.img_size_x, args.img_size_y), 
        batch_size=args.batch_size, inference=True
    )

    trainer = pl.Trainer(
        gpus=-1, benchmark=True,
        precision=16,
        deterministic=True,
    )
    preds = torch.cat(trainer.predict(ckpt_model, datamodule=dm), dim=0).detach().cpu().numpy()
    pseudo_df['Pawpularity'] = preds
    pseudo_df['fold'] = n_folds
    
    del trainer
    del ckpt_model
    del dm
    gc.collect()
    
    #pseudo_df = pseudo_df[(pseudo_df['Pawpularity'] < 10) | (pseudo_df['Pawpularity'] > 90)]
    
    train_df = pd.concat([train_df, pseudo_df], ignore_index=True)

    train_filter = train_df['fold'] != i
    val_filter = train_df['fold'] == i
    dm = DataModule(
        train_df, img_size=(args.img_size_x, args.img_size_y), 
        train_filter=train_filter, val_filter=val_filter, 
        batch_size=args.batch_size,
    )

    model = PetFinderModel(**hparams, pretrained=True)

    ckpt = ModelCheckpoint(
        dirpath=ckpt_path, 
        monitor='val_rmse_loss', mode='min', 
        filename=f'{args.model_name}-seed-{args.seed}-{args.name}_pseudo-fold-{i}-{{val_bce_loss:.4f}}-{{val_rmse_loss:.4f}}'
    )
    early_stop = EarlyStopping('val_rmse_loss', mode='min', patience=4)
    wandb_logger = WandbLogger(project='petfinder-pawpularity-score', log_model=False, name=f'{args.model_name}-seed-{args.seed}-{args.name}_pseudo-fold-{i}')
    wandb_logger.watch(model, log='all')

    trainer = pl.Trainer(
        gpus=-1, benchmark=True,
        callbacks=[LearningRateMonitor(), ckpt, early_stop],
        logger=wandb_logger,
        enable_checkpointing=True,
        accumulate_grad_batches=hparams['accumulate_grad_batches'],
        deterministic=True,
        gradient_clip_val=args.grad_clip_val,
        precision=16,
        val_check_interval=0.25,
        max_epochs=args.max_epochs,
    )

    trainer.fit(model, datamodule=dm)

    wandb.run.summary['best_bce_loss'] = model.best_bce_loss
    wandb.run.summary['best_rmse_loss'] = model.best_rmse_loss
    wandb.run.summary['batch_size'] = args.batch_size

    wandb.finish()

    del model
    del dm
    del trainer
    gc.collect()

    torch.cuda.empty_cache()