# coding=utf-8

# write code...
"""
10 fold is the best
2017-10-21/14:28:23 finder_best_cv.py __main__ 86 [INFO][<module>] mean acc scores are 3_folds:0.797539149888, 5_folds:0.822667405966, 10_folds:0.81041212221
2017-10-21/14:28:23 finder_best_cv.py __main__ 89 [INFO][<module>] variance of acc scores are 3_folds:0.000858319695309, 5_folds:0.00111858253956, 10_folds:0.00289941848532

"""



import os
import logging

import numpy as np
import pandas as pd
from  lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import accuracy_score
import daiquiri

from preprocess.split_cv import split_cv

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
ORG_DATA_DIR = os.path.join(APP_ROOT, "data/")

ORG_TARGET_DATA = os.path.join(ORG_DATA_DIR, "target.csv")
ORG_TRAIN_DATA = os.path.join(ORG_DATA_DIR, "train.csv")
ORG_TEST_DATA = os.path.join(ORG_DATA_DIR, "test.csv")
ORG_CONCAT_DATA = os.path.join(ORG_DATA_DIR, "data.csv")

PREPROCESSED_DATA_DIR = os.path.join(APP_ROOT, 'data/preprocessed/')
PREPROCESSED_TRAIN_DATA = os.path.join(PREPROCESSED_DATA_DIR, "2017-10-23_preprocessed_train_data.csv.gz")
PREPROCESSED_TEST_DATA = os.path.join(PREPROCESSED_DATA_DIR, "2017-10-23_preprocessed_test_data.csv.gz")

log_fmt = '%(asctime)s %(filename)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
daiquiri.setup(level=logging.DEBUG,
               outputs=(
                   daiquiri.output.Stream(formatter=daiquiri.formatter.ColorFormatter(fmt=log_fmt)),
                   daiquiri.output.File("predict.log", level=logging.DEBUG)
               ))
logger = daiquiri.getLogger(__name__)

if __name__ == "__main__":

    logger.info('load start')
    X = pd.read_csv(PREPROCESSED_TRAIN_DATA).values
    y = pd.read_csv(ORG_TARGET_DATA).values
    y_c, y_r = y.shape
    y = y.reshape(y_c, )
    logger.info('target end')

    num_of_folds = [3, 5, 10]
    split_cv(X, y, num_of_folds, ORG_DATA_DIR)

    acc_score_means = []
    acc_score_vars = []
    for num_of_fold in num_of_folds:
        logger.info("evaluating %s fold" % num_of_fold)
        CV_DIR = os.path.join(ORG_DATA_DIR, "n_folds_%s/" % num_of_fold)
        acc_score = []
        for i in range(num_of_fold):
            logger.info("loading %s th cv data in %s folds" % (i, num_of_fold))
            X_train = pd.read_csv(os.path.join(CV_DIR, "X_train_%s.csv") % i, header=None, sep="\t").values
            X_val = pd.read_csv(os.path.join(CV_DIR, "X_val_%s.csv") % i, header=None, sep="\t").values
            y_train = pd.read_csv(os.path.join(CV_DIR, "y_train_%s.csv") % i, header=None, sep="\t").values
            y_c, y_r = y_train.shape
            y_train = y_train.reshape(y_c, )
            y_val = pd.read_csv(os.path.join(CV_DIR, "y_val_%s.csv") % i, header=None, sep="\t").values
            y_c, y_r = y_val.shape
            y_val = y_val.reshape(y_c, )
            logger.info("end loading %s th cv data in %s folds" % (i, num_of_fold))
            logger.info("X_train.shape: %s %s" % X_train.shape)
            logger.info("X_val.shape: %s %s" % X_val.shape)
            logger.info("y_train.shape: %s" % y_train.shape)
            logger.info("y_val.shape: %s" % y_val.shape)

            num_pos = y_train.sum()
            num_neg = y_train.shape[0] - num_pos
            w = num_neg/num_pos
            sample_weight = np.where(y_train == 1, w, 1)
            logger.info("calc samples for LightGBM's sample weight neg: %s pos: %s" % (num_neg, num_pos))

            clf = LGBMClassifier(objective="binary",
                                 n_estimators=20)
            clf.fit(X_train, y_train,
                    sample_weight=sample_weight,
                    eval_set=[(X_val, y_val)],
                    verbose=True)
            y_pred = clf.predict(X_val)
            logger.info("acc socore: %s folds, %s iteration" % (num_of_fold, i))
            acc_score.append(accuracy_score(y_val, y_pred))
        logger.info("mean acc score of %s folds is %s" % (num_of_fold, np.mean(acc_score)))
        acc_score_means.append(np.mean(acc_score))
        logger.info("variance of acc score of %s folds is %s" % (num_of_fold, np.var(acc_score)))
        acc_score_vars.append(np.var(acc_score))
    logger.info("mean acc scores are 3_folds:%s, 5_folds:%s, 10_folds:%s " % (acc_score_means[0],
                                                                              acc_score_means[1],
                                                                              acc_score_means[2]))
    logger.info("variance of acc scores are 3_folds:%s, 5_folds:%s, 10_folds:%s" % (acc_score_vars[0],
                                                                                    acc_score_vars[1],
                                                                                    acc_score_vars[2]))
