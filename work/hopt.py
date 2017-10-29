# coding=utf-8

# write code...
import os
import pickle
import logging

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from hyperopt import hp, tpe, Trials, STATUS_OK, fmin

from sklearn.metrics import make_scorer
import daiquiri



APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
ORG_DATA_DIR = os.path.join(APP_ROOT, "data/")

ORG_TARGET_DATA = os.path.join(ORG_DATA_DIR, "target.csv")

PREPROCESSED_DATA_DIR = os.path.join(APP_ROOT, 'data/preprocessed/')
PREPROCESSED_TRAIN_DATA = os.path.join(PREPROCESSED_DATA_DIR, "2017-10-23_preprocessed_train_data.csv.gz")

X = pd.read_csv(PREPROCESSED_TRAIN_DATA).values
y = pd.read_csv(ORG_TARGET_DATA).values
y_c, y_r = y.shape
y = y.reshape(y_c, )

def score(params):
    params = {"max_depth": int(params["max_depth"])}
              #"random_state": int(params["random_state"])}
    clf = lgb.LGBMClassifier(**params)
    print(clf)
    logloss = cross_val_score(clf, X, y, cv=StratifiedKFold()).mean()
    print(logloss)
    return {'loss':logloss, 'status': STATUS_OK }

space = {
    "max_depth": hp.quniform('max_depth', 1, 10 ,1)
    "random_state": random_state
    "subsample": hp.loguniform('subsample', np.log(0.3), np.log(0.7)),
    "colsample_bytree": hp.loguniform('colsample_bytree', np.log(0.3), np.log(0.7)),
    "num_leaves": hp.quniform('num_leaves', 1, 5 ,1)
    }


trials=Trials()
best_params = fmin(fn=score, space=space, algo=tpe.suggest, trials=trials, max_evals=2)
print(best_params)

