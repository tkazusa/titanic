# coding=utf-8

# write code...
import os
from datetime import date
from logging import getLogger, basicConfig

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from util import Util

log_fmt = '%(asctime)s %(filename)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
basicConfig(format=log_fmt, datefmt='%Y-%m-%d/%H:%M:%S', level='DEBUG')

logger = getLogger(__name__)

def split_cv(X, y, num_of_folds, DIR_TO_SAVED):
    """
    :param X: array
    :param y: array
    :param num_of_folds:[int, int, ...]
    :X_train, X_val, y_train, y_val are saved on DIr "data/n_folds_xx"
    """
    for num_of_fold in num_of_folds:
        logger.info("splitting to %s division data" % num_of_fold)
        skf = StratifiedKFold(n_splits=num_of_fold)
        CV_DIR = os.path.join(DIR_TO_SAVED, "n_folds_%s/" % num_of_fold)
        for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            logger.info("writing %s th cv data")
            X_train, X_val = pd.DataFrame(X[train_idx]), pd.DataFrame(X[test_idx])
            logger.info("X_train data shape %s %s" % X_train.shape)
            logger.info("X_val data shape %s %s" % X_val.shape)
            y_train, y_val = pd.DataFrame(y[train_idx]), pd.DataFrame(y[test_idx])
            logger.info("y_train data shape %s %s" % y_train.shape)
            logger.info("y_val data shape %s %s" % y_val.shape)

            logger.info("saving on %s" % CV_DIR)
            Util.to_csv(X_train,os.path.join(CV_DIR, "X_train_%s.csv" % i))
            Util.to_csv(X_val,os.path.join(CV_DIR, "X_val_%s.csv" % i))
            Util.to_csv(y_train,os.path.join(CV_DIR, "y_train_%s.csv" % i))
            Util.to_csv(y_val,os.path.join(CV_DIR, "y_val_%s.csv" % i))
