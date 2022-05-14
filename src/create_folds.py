import pandas as pd
from sklearn.model_selection import KFold

TRAIN_PATH = None
N_FOLDS = None
TARGET = None
if __name__ == "__main__":
    df = pd.read_csv(TRAIN_PATH)
    df.loc[:, 'kfold'] = -1
    df.sample(frac=1).reset_index(drop=True)
    
    kf = KFold(n_splits=N_FOLDS)
    for f_, (trn_, val_) in kf.split(X=df, y=df[TARGET].values):
        df.loc[val_, 'kfold'] = f_
    
    df.to_csv('../input/train_folds.csv', index=False)