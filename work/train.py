# coding=utf-8

# write code..
import os
import pickle
import logging

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from lightgbm.sklearn import LGBMClassifier
import daiquiri


APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
ORG_DATA_DIR = os.path.join(APP_ROOT, "data/")

ORG_TARGET_DATA = os.path.join(ORG_DATA_DIR, "target.csv")

PREPROCESSED_DATA_DIR = os.path.join(APP_ROOT, 'data/preprocessed/')
PREPROCESSED_TRAIN_DATA = os.path.join(PREPROCESSED_DATA_DIR, "2017-10-23_preprocessed_train_data.csv.gz")


log_fmt = '%(asctime)s %(filename)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
daiquiri.setup(level=logging.DEBUG,
               outputs=(
                   daiquiri.output.Stream(formatter=daiquiri.formatter.ColorFormatter(fmt=log_fmt)),
                   daiquiri.output.File("predict.log", level=logging.DEBUG)
               ))
logger = daiquiri.getLogger(__name__)


def train_lightgbm(verbose=True):
    """Train a LightGBM."""
    logger.info("Training a LightGBM model")
    logger.info('load start')
    X = pd.read_csv(PREPROCESSED_TRAIN_DATA).values
    y = pd.read_csv(ORG_TARGET_DATA).values
    y_c, y_r = y.shape
    y = y.reshape(y_c, )
    logger.info('load end')

    num_pos = y.sum()
    num_neg = y.shape[0] - num_pos
    w = num_neg/num_pos
    sample_weight = np.where(y == 1, w, 1)
    logger.info("calc samples for LightGBM's sample weight neg: %s pos: %s" % (num_neg, num_pos))

    all_params = {'max_depth': [1, 2],
                  'learning_rate': [0.1,0.3],
                  'n_estimators': [2,3],
                  'subsample': [0.7],
                  'colsample_bytree': [0.5],
                  'boosting_type': ['gbdt'],
                  'num_leaves': [3, 5],
                  'is_unbalance': [True, False],
                  'random_state': [3655]
                  }

    best_acc = (1, 1, 1)
    best_logloss = (100, 100, 100)
    best_f1 = (1, 1, 1)
    best_auc = (1, 1, 1)
    best_params = None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3655)
    use_score = 0

    for params in ParameterGrid(all_params):
        list_score_acc = []
        list_score_logloss = []
        list_score_f1 = []
        list_score_auc = []
        list_best_iter = []
        for train, val in skf.split(X, y):
            X_train, X_val = X[train], X[val]
            y_train, y_val = y[train], y[val]
            weight_train = sample_weight[train]
            weight_val = sample_weight[val]


            clf = LGBMClassifier(**params)
            clf.fit(X_train, y_train,
                    sample_weight=weight_train,
                    eval_sample_weight=[weight_val],
                    eval_set=[(X_val, y_val)],
                    eval_metric="logloss",
                    #early_stopping_rounds=0,
                    verbose=False
                    )

            _score_acc = accuracy_score(y_val, clf.predict(X_val), sample_weight=weight_val)
            _score_logloss = log_loss(y_val, clf.predict_proba(X_val), sample_weight=weight_val)
            _score_f1 = f1_score(y_val, clf.predict(X_val), sample_weight=weight_val)
            _score_auc = roc_auc_score(y_val, clf.predict(X_val), sample_weight=weight_val)


            list_score_acc.append(_score_acc)
            list_score_logloss.append(_score_logloss)
            list_score_f1.append(_score_f1)
            list_score_auc.append(_score_auc)
            list_best_iter.append(params["n_estimators"])
            """
            ##n_estimaters=0 causes error at .fit()
            if clf.best_iteration_ != -1:
                list_best_iter.append(clf.best_iteration_)
            else:
                list_best_iter.append(params['n_estimators'])
            break
            """
        logger.info("n_estimators: {}".format(list_best_iter))
        params["n_estimators"] = np.mean(list_best_iter, dtype=int)

        score_acc = (np.mean(list_score_acc), np.min(list_score_acc), np.max(list_score_acc))
        score_logloss = (np.mean(list_score_logloss), np.min(list_score_logloss), np.max(list_score_logloss))
        score_f1 = (np.mean(list_score_f1), np.min(list_score_f1), np.max(list_score_f1))
        score_auc = (np.mean(list_score_auc), np.min(list_score_auc), np.max(list_score_auc))

        logger.info("param: %s" % (params))
        logger.info("acc: {} (avg min max {})".format(score_acc[use_score], score_acc))
        logger.info("logloss: {} (avg min max {})".format(score_logloss[use_score], score_logloss))
        logger.info("f1: {} (avg min max {})".format(score_f1[use_score], score_f1))
        logger.info("acu: {} (avg min max {})".format(score_auc[use_score], score_auc))
        if best_logloss[use_score] > score_logloss[use_score]:
            best_acc = score_acc
            best_logloss = score_logloss
            best_f1 = score_f1
            best_auc = score_auc
            best_params = params
    logger.info('best acc: {} {}'.format(best_acc[use_score], best_acc))
    logger.info('best logloss: {} {}'.format(best_logloss[use_score], best_logloss))
    logger.info('best f1: {} {}'.format(best_f1[use_score], best_f1))
    logger.info('best auc: {} {}'.format(best_auc[use_score], best_auc))
    logger.info("best_param: {}".format(best_params))

    logger.info("start training with best params")

    clf = LGBMClassifier(**best_params)
    clf.fit(X,y)

    imp = pd.DataFrame(clf.feature_importances_, columns=['imp'])
    print(pd.read_csv(PREPROCESSED_TRAIN_DATA).columns)
    imp.index = pd.read_csv(PREPROCESSED_TRAIN_DATA).columns
    print(imp)
    with open('features.py', 'a') as f:
        f.write('FEATURE = ["' + '","'.join(map(str, imp[imp['imp'] > 0].index.values)) + '"]\n')

    return clf

if __name__ == "__main__":
    clf = train_lightgbm()
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
