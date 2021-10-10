from scipy.optimize import minimize
from time import time
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import datetime
import os

def rmse(y_pred):
    #return np.mean((y_pred - y_true)**2)**(-1/2.)
    return mean_squared_error(y_true, y_pred, squared=False)

def metric(weights):
    oof_blend = np.tensordot(weights, oof, axes = ((0), (0)))
    score = rmse(oof_blend)
    return score


oof_dict = {
    'model1': 'data/oof_preds/swin_large_patch4_window7_224_in22k-mixup=0.5.csv',
    'model2': 'data/oof_preds/swin_base_patch4_window7_224_in22k-mixup=0.5.csv',
    'model3': 'data/oof_preds/swin_tiny_patch4_window7_224-mixup=0.5.csv',
    'model4': 'data/oof_preds/eca_nfnet_l2-mixup=0.5.csv',
    'model5': 'data/oof_preds/eca_nfnet_l2-img_size=320x320.csv',
    # 'model6': 'data/oof_preds/swin_base_patch4_window12_384_in22k-seed-34-mixup=0.5.csv',
    # 'model7': 'data/oof_preds/xcit_medium_24_p8_224_dist-seed-34-mixup=0.5.csv',
    # 'model8': 'data/oof_preds/eca_nfnet_l2-seed-34-img_size=512x512.csv',
}

data_dir = 'data'
train_df = pd.read_csv(f'{data_dir}/train_folds.csv')

train_df['file_path'] = f'{data_dir}/train/' + train_df['Id'] + '.jpg'

y_true = train_df[['Id', 'Pawpularity']].sort_values(by='Id')

oof_dfs = [pd.read_csv(f) for f in oof_dict.values()]
for i, df in enumerate(oof_dfs):
    y_true = y_true.merge(df[['Id']], on='Id')
    oof_dfs[i] = df.merge(y_true[['Id']], on='Id').sort_values(by='Id')

smallest_shape = min([df.shape[0] for df in oof_dfs])
oof = np.zeros((len(oof_dict), smallest_shape, 1))
for i, df in enumerate(oof_dfs):
    df = df.merge(y_true[['Id']], on='Id')
    oof[i] = df['preds'].values.reshape(-1, 1)

y_true = y_true['Pawpularity'].values.reshape(-1, 1)
print(y_true.shape)

rmse_scores = {}
for n, key in enumerate(oof_dict.keys()):
    score_oof = rmse(oof[n])
    rmse_scores[key] = score_oof
    print(f'{key:40s} CV: {score_oof:.6f}')
    
print('-' * 60)

tol = 1e-10
init_guess = [1 / oof.shape[0]] * oof.shape[0]
bnds = [(0, 1) for _ in range(oof.shape[0])]
cons = {'type': 'eq', 
        'fun': lambda x: np.sum(x) - 1, 
        'jac': lambda x: [1] * len(x)}

print(f'Inital Blend OOF: {metric(init_guess):.6f}', )
start_time = time()

res_scipy = minimize(fun = metric, 
                     x0 = init_guess, 
                     method = 'Powell', 
                     #method='SLSQP',
                     bounds = bnds, 
                     options=dict(maxiter=1_000_000),
                     tol = tol)

print(f'[{str(datetime.timedelta(seconds = time() - start_time))[2:7]}] Optimised Blend OOF: {res_scipy.fun:.6f}')
print(f'Optimised Weights: {res_scipy.x}')
print('-' * 70)

for n, key in enumerate(oof_dict.keys()):
    print(f'{key} Optimised Weights: {res_scipy.x[n]:.6f}')

ws = [ res_scipy.x[i] for i in range(len(oof_dict.keys()))]
print(f'Normalized weights:')
weights = ws / np.sum(ws)
#print(pd.DataFrame({'model': [k for k in oof_dict.keys()], 'weight': [w for w in weights]}))

weight_dict = {}
for i, (k, v) in enumerate(oof_dict.items()):
    model_name = v.split('/')[-1].split('.csv')[0].replace('=', '')
    weight_dict[model_name] = weights[i]

print(weight_dict)