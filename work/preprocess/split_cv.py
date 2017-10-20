# coding=utf-8

# write code...
import os
from datetime import date
from logging import getLogger, basicConfig

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..util import Util


APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
ORG_DATA_DIR = os.path.join(APP_ROOT, "data/")

ORG_TARGET_DATA = os.path.join(ORG_DATA_DIR, "target.csv")
ORG_TRAIN_DATA = os.path.join(ORG_DATA_DIR, "train.csv")
ORG_TEST_DATA = os.path.join(ORG_DATA_DIR, "test.csv")
ORG_CONCAT_DATA = os.path.join(ORG_DATA_DIR, "data.csv")

PREPROCESSED_DATA_DIR = os.path.join(APP_ROOT, 'data/preprocessed/')
PREPROCESSED_TRAIN_DATA = os.path.join(PREPROCESSED_DATA_DIR, "2017-10-19_preprocessed_train_data.csv.gz")
PREPROCESSED_TEST_DATA = os.path.join(PREPROCESSED_DATA_DIR,
                                      str(date.today().isoformat())+"_preprocessed_test_data.csv.gz")


log_fmt = '%(asctime)s %(filename)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
basicConfig(format=log_fmt, datefmt='%Y-%m-%d/%H:%M:%S', level='DEBUG')

logger = getLogger(__name__)


if __name__ == '__main__':
    logger.info('load start')
    X = pd.read_csv(PREPROCESSED_TRAIN_DATA).values
    y = pd.read_csv(ORG_TARGET_DATA).values
    y_c, y_r = y.shape
    y = y.reshape(y_c, )
    logger.info('target end')


    for num_of_fold in [3, 5, 10]:
        print(type(num_of_fold))
        skf = StratifiedKFold(n_splits=num_of_fold)
        #Util.mkdir(os.path.join(ORG_DATA_DIR, "n_folds_%s" % num_of_fold))
        CV_DIR = os.path.join(ORG_DATA_DIR, "n_folds_%s/" % num_of_fold)
        for train_idx, test_idx in skf.split(X, y):
            print("%s %s" % (len(train_idx), len(test_idx)))
            X_train, X_val = X[train_idx], X[test_idx]
            y_train, y_val = y[train_idx], y[test_idx]

            Util.to_csv(X_train, os.path.join(CV_DIR, "X_train"))
            Util.to_csv(X_val, os.path.join(CV_DIR, "X_val"))

            Util.to_csv(y_train, os.path.join(CV_DIR, "y_train"))
            Util.to_csv(y_val, os.path.join(CV_DIR, "y_val"))





