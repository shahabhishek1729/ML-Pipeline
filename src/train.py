import os
import pandas as pd
from sklearn import preprocessing
from sklearn import ensemble

TRAIN_PATH = os.environ.get('TRAIN_PATH')
N_FOLDS = os.environ.get('N_FOLDS')
FOLD = os.environ.get('FOLD')
ID = os.environ.get('ID')
TARGET = os.environ.get('TARGET')

folds_x = range(N_FOLDS)
folds_y = [[i for i in range(N_FOLDS) if i != x] for x in folds_x]
FOLD_MAPPING = dict(zip(folds_x, folds_y))

if __name__ == "__main__":
    df = pd.read_csv(TRAIN_PATH)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold == FOLD]

    y_train = train_df[TARGET].values
    y_valid = valid_df[TARGET].values

    train_df = train_df.drop([ID, TARGET, 'kfold'], inplace=True, axis=1)
    valid_df = valid_df.drop([ID, TARGET, 'kfold'], inplace=True, axis=1)

    valid_df = valid_df[train_df.columns] # Make sure both train_df and valid_df have the same columns

    ### TRAINING ###
    clf = ensemble.RandomForestClassifier(n_jobs=-1, verbose=2)
    clf.fit(train_df, y_train)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(preds)