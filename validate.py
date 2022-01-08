from dataset import DataModule
from model import PetFinderModel
from callbacks import LogPredictionsCallback
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
import pytorch_lightning as pl
import argparse
import wandb
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='resnet34d-simple_baseline')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--img_size_x', type=int, default=224)
parser.add_argument('--img_size_y', type=int, default=224)
parser.add_argument('--interpolation', type=str, default='bilinear')
parser.add_argument('--seed', type=int, default=34)
parser.add_argument('--data_seed', type=int, default=34)

args = parser.parse_args()

pl.seed_everything(args.seed);

data_dir = 'data'
train_df = pd.read_csv(f'{data_dir}/train_folds_seed_{args.data_seed}.csv')
train_df['file_path'] = f'{data_dir}/train/' + train_df['Id'] + '.jpg'

ckpt_path = '/media/mten/storage/kaggle/petfinder-pawpularity-score/ckpts'

model_paths = [p for p in os.listdir(f'{ckpt_path}/') if p.startswith(args.model_name + '-fold')]
model_paths = sorted(model_paths)
[print(p) for p in model_paths]
assert len(model_paths) == 5
models = [PetFinderModel.load_from_checkpoint(f'{ckpt_path}/{p}') for p in model_paths]
for model in models:
    model.freeze();

train_df['preds'] = 0.
for i in range(5):
    pred_df = train_df.loc[train_df['fold'] == i, :]
    dm = DataModule(
        pred_df, img_size=(args.img_size_x, args.img_size_y), 
        batch_size=args.batch_size, inference=True
    )

    model = models[i]

    trainer = pl.Trainer(
        gpus=-1, benchmark=True,
        precision=16,
        deterministic=True,
    )
    preds = torch.cat(trainer.predict(model, datamodule=dm), dim=0).detach().cpu().numpy()
    train_df.loc[train_df['fold'] == i, 'preds'] = preds
    rmse = mean_squared_error(pred_df['Pawpularity'].values, preds, squared=False)
    print(f'Fold {i} MSE: {rmse}')

train_df[['Id', 'preds']].to_csv(f'data/oof_preds/{args.model_name}.csv', index=False)

rmse = mean_squared_error(train_df['Pawpularity'].values, train_df['preds'].values, squared=False)
print(f'MSE: {rmse}')