from tqdm import tqdm
from cuml.svm import SVR
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='resnet34d-simple_baseline')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--img_size_x', type=int, default=224)
parser.add_argument('--img_size_y', type=int, default=224)
parser.add_argument('--param_search', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=34)

args = parser.parse_args()

data_dir = 'data'
data = pd.read_csv(f'{data_dir}/embeddings/{args.model_name}.csv')
oof_preds = pd.read_csv(f'{data_dir}/oof_preds/{args.model_name}.csv')
data['file_path'] = f'{data_dir}/train/' + data['Id'] + '.jpg'
emb_cols = [c for c in data.columns if c.startswith('emb')]

data['preds'] = 0.
for i in range(5):
    train_df = data.loc[data['fold'] != i, :]
    val_df = data.loc[data['fold'] == i, :]

    svm = SVR(C=5.0).fit(
        train_df[emb_cols].values.astype(np.float32), 
        train_df['Pawpularity'].values.astype(np.float32)
    )
    preds = svm.predict(val_df[emb_cols].values.astype(np.float32))
    
    data.loc[data['fold'] == i, 'preds'] = preds

rmse = mean_squared_error(data['Pawpularity'].values, data['preds'].values, squared=False)
print(f'SVM MSE: {rmse}')

# Get ensemble OOF score
data['preds'] += oof_preds['preds']
data['preds'] /= 2.

rmse = mean_squared_error(data['Pawpularity'].values, data['preds'].values, squared=False)
print(f'Ensembled MSE: {rmse}')

pickle.dump(svm, open(f'svm_models/{args.model_name}.pkl', 'wb'))
data[['Id', 'preds']].to_csv(f'data/svm_oofs/{args.model_name}_svm.csv', index=False)

if args.param_search:
    rmses = []
    for c in tqdm(range(1, 50)):
        data['preds'] = 0.
        for i in range(5):
            train_df = data.loc[data['fold'] != i, :]
            val_df = data.loc[data['fold'] == i, :]

            svm = SVR(C=c).fit(
                train_df[emb_cols].values.astype(np.float32), 
                train_df['Pawpularity'].values.astype(np.float32)
            )
            preds = svm.predict(val_df[emb_cols].values.astype(np.float32))
            
            data.loc[data['fold'] == i, 'preds'] = preds

        rmse = mean_squared_error(data['Pawpularity'].values, data['preds'].values, squared=False)
        rmses.append(rmse)

    plt.plot(rmses)
    plt.savefig('data/plots/svm_rmses.png')