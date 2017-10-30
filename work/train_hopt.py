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
from hyperopt import hp, tpe, Trials, STATUS_OK, fmin
import daiquiri

random_state = np.random.RandomState(10)

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


def train_lightgbm_hopt(verbose=True):
    """Train a LightGBM with hopt."""
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



    def score(params):
        params = {"max_depth": int(params["max_depth"]),
                  "subsample": round(params["subsample"],3),
                  "colsample_bytree": round(params['colsample_bytree'],3),
                  "num_leaves": int(params['num_leaves']),
                  "n_jobs": -2
                  #"is_unbalance": params['is_unbalance']
                  }

        clf = LGBMClassifier(n_estimators=500, learning_rate=0.05, **params)

        list_score_acc = []
        list_score_logloss = []

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=3655)
        for train, val in skf.split(X, y):
            X_train, X_val = X[train], X[val]
            y_train, y_val = y[train], y[val]
            weight_train = sample_weight[train]
            weight_val = sample_weight[val]

            clf.fit(X_train, y_train,
                    sample_weight=weight_train,
                    eval_sample_weight=[weight_val],
                    eval_set=[(X_val, y_val)],

                    eval_metric="logloss",
                    early_stopping_rounds=0,
                    verbose=False
                    )

            _score_acc = accuracy_score(y_val, clf.predict(X_val), sample_weight=weight_val)
            _score_logloss = log_loss(y_val, clf.predict_proba(X_val), sample_weight=weight_val)


            list_score_acc.append(_score_acc)
            list_score_logloss.append(_score_logloss)
            """
            ##n_estimaters=0 causes error at .fit()
            if clf.best_iteration_ != -1:
                list_best_iter.append(clf.best_iteration_)
            else:
                list_best_iter.append(params['n_estimators'])
            break
            """
        #logger.info("n_estimators: {}".format(list_best_iter))
        #params["n_estimators"] = np.mean(list_best_iter, dtype=int)

        score_acc = (np.mean(list_score_acc), np.min(list_score_acc), np.max(list_score_acc))
        #logger.info("score_acc %s" % np.mean(list_score_acc))

        #score_logloss = (np.mean(list_score_logloss), np.min(list_score_logloss), np.max(list_score_logloss))
        #score_f1 = (np.mean(list_score_f1), np.min(list_score_f1), np.max(list_score_f1))
        #score_auc = (np.mean(list_score_auc), np.min(list_score_auc), np.max(list_score_auc))

        logloss = np.mean(list_score_logloss)
        return {'loss' : logloss,  'status': STATUS_OK, 'localCV_acc': score_acc}

    space = {"max_depth": hp.quniform('max_depth', 1, 10 ,1),
             "subsample": hp.uniform('subsample', 0.3, 0.8),
             "colsample_bytree": hp.uniform('colsample_bytree', 0.3, 0.8),
             "num_leaves": hp.quniform('num_leaves', 5, 100, 1),
             #"is_unbalance": hp.choice('is_unbalance', [True, False])
             }

    trials = Trials()
    best_params = fmin(rstate=random_state,fn=score, space=space, algo=tpe.suggest, trials=trials, max_evals=100)
    logger.info("localCV_acc %s" % list(filter(lambda x : x["loss"] == min(trials.losses()), trials.results))[0]["localCV_acc"][0])



    best_params = {"max_depth": int(best_params["max_depth"]),
                   "subsample": best_params["subsample"],
                   "colsample_bytree": best_params['colsample_bytree'],
                   "num_leaves": int(best_params['num_leaves'])
                  #"is_unbalance": best_params['is_unbalance']
                  }
    logger.info("best params are %s" % best_params)

    clf = LGBMClassifier(n_estimators=500, learning_rate=0.05, **best_params)
    logger.info("start training with best params")
    clf.fit(X,y)

    imp = pd.DataFrame(clf.feature_importances_, columns=['imp'])
    imp.index = pd.read_csv(PREPROCESSED_TRAIN_DATA).columns
    logger.info("important features are %s" %imp)
    with open('features.py', 'a') as f:
        f.write('FEATURE = ["' + '","'.join(map(str, imp[imp['imp'] > 0].index.values)) + '"]\n')

    return clf

if __name__ == "__main__":
    clf = train_lightgbm_hopt()
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)



