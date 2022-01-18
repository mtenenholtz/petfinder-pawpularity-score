from dataset import DataModule
from model import PetFinderEmbeddingsModel
from callbacks import LogPredictionsCallback
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
import argparse
import albumentations as A
import wandb
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='resnet34d-simple_baseline')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--img_size_x', type=int, default=224)
parser.add_argument('--img_size_y', type=int, default=224)
parser.add_argument('--seed', type=int, default=34)

args = parser.parse_args()

pl.seed_everything(args.seed);

data_dir = 'data'
train_df = pd.read_csv(f'{data_dir}/train_folds_10.csv')
train_df['file_path'] = f'{data_dir}/train/' + train_df['Id'] + '.jpg'

ckpt_path = '/media/mten/storage/kaggle/petfinder-pawpularity-score/ckpts'

model_paths = [p for p in os.listdir(f'{ckpt_path}/') if p.startswith(args.model_name + '-fold')]
model_paths = sorted(model_paths)
[print(p) for p in model_paths]
assert len(model_paths) == 10
models = [PetFinderEmbeddingsModel.load_from_checkpoint(f'{ckpt_path}/{p}') for p in model_paths]
for model in models:
    model.freeze();

for i in range(10):
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
    if 'emb_0' not in train_df:
        emb_cols = [f'emb_{n}' for n in range(preds.shape[1])]
        train_df[emb_cols] = 0.
    train_df.loc[train_df['fold'] == i, emb_cols] = preds

train_df[['Id', 'Pawpularity', 'fold'] + emb_cols].to_csv(f'data/embeddings/{args.model_name}.csv', index=False)
