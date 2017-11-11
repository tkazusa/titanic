# coding=utf-8

# write code...
import os
import pickle
import logging

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from lightgbm.sklearn import LGBMClassifier


from bases.model import ModelBase
from util import Util



APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
ORG_DATA_DIR = os.path.join(APP_ROOT, "data/")

ORG_TARGET_DATA = os.path.join(ORG_DATA_DIR, "target.csv")

PREPROCESSED_DATA_DIR = os.path.join(APP_ROOT, 'data/preprocessed/')
PREPROCESSED_TRAIN_DATA = os.path.join(PREPROCESSED_DATA_DIR, "2017-10-23_preprocessed_train_data.csv.gz")


class ModelLgbm_initial(ModelBase):
    """Model Base interface

    Methods:
        train:Trains and generates background model.
        save_model:Saves background model generated by train method to specified file path.
        load__model:Loads background model from specified file path.
    """

    def __init__(self):
        raise NotImplementedError

    def _set_algorithm(self, prms):
        model = LGBMClassifier(objective="binary",
                               n_estimators=500,
                               learning_rate=0.05
                               **prms)
        return model

    def train(self, prms, X_tr, y_tr, X_val, y_val,w_tr, w_val):

        self.model  = self._set_algorithm(prms)
        self.model.fit(X_tr, y_tr,
                  sample_weight=w_tr,
                  eval_sample_weight=[w_val],
                  eval_set=[X_val, y_val],
                  eval_metric="logloss",
                  verbose=False
                  )

        self._score_acc = accuracy_score(y_val, model.predict(X_val), sample_weight=w_val)
        self._score_logloss = log_loss(y_val, model.predict_proba(X_val), sample_weight=w_val)

    def train_all(self, prms, X_tr, y_tr, w_tr):

        self.model  = self._set_algorithm(prms)
        self.model.fit(X_tr, y_tr,
                  sample_weight=w_tr,
                  eval_metric="logloss",
                  verbose=False
                  )


    def save_model(self):
        Util.dump(self.model, "../model/model/lgb_initial.pkl")

    def load_model(self):
        model = Util.load("../model/model/lgb_initial.pkl")
        self.model = model

    def predict(self, X):
        return self.model.predict(X)





