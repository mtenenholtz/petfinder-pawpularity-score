import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection

def create_folds(data, num_splits):
    # we create a new column called kfold and fill it with -1
    data['fold'] = -1
    
    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate number of bins by Sturge's rule
    # I take the floor of the value, you can also
    # just round it
    #num_bins = int(np.floor(1 + np.log2(len(data))))
    num_bins = int(np.floor(1+(3.3)*(np.log2(len(data)))))
    
    # bin targets
    data.loc[:, 'bins'] = pd.cut(
        data['Pawpularity'], bins=num_bins, labels=False
    )
    
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=num_splits)
    
    # fill the new kfold column
    # note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'fold'] = f
    
    # drop the bins column
    data = data.drop('bins', axis=1)

    # return dataframe with folds
    return data

df = pd.read_csv('data/train.csv')
# df = create_folds(df, num_splits=5)

seed = 26
skf = model_selection.StratifiedKFold(
    n_splits=5, shuffle=True, random_state=seed
)

df['fold'] = -1
for i, (train_idx, val_idx) in enumerate(skf.split(df['Id'], df['Pawpularity'])):
    df.loc[val_idx, 'fold'] = i

print(df.fold.value_counts())
df.to_csv(f'data/train_folds_seed_{seed}.csv', index=False)