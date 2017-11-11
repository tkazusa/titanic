# -*- coding: utf-8 -*-
#
#
#
# Copyright (c) 2017 BrainPad, Inc. All rights reserved.
#
# This software is the confidential and proprietary information of
# BrainPad, Inc. ("Confidential Information").
# You shall not disclose such Confidential Information and shall
# use it only in accordance with the terms of the license agreement
# you entered into with BrainPad, Inc.
#
#
# Author: taketoshi.kazusa
#
import os
import datetime

import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold

logfile_name = "logs/" + str(datetime.date.today().isoformat())+ ".log"


class Util:

    @classmethod
    def mkdir(cls, dr):
        if not os.path.exists(dr):
            os.makedirs(dr)

    @classmethod
    def mkdir_file(cls, path):
        dr = os.path.dirname(path)
        if not os.path.exists(dr):
            os.makedirs(dr)

    @classmethod
    def dump(cls, obj, filename, compress=0):
        cls.mkdir_file(filename)
        joblib.dump(obj, filename, compress=compress)

    @classmethod
    def dumpc(cls, obj, filename):
        cls.mkdir_file(filename)
        cls.dump(obj, filename, compress=3)

    @classmethod
    def load(cls, filename):
        return joblib.load(filename)

    @classmethod
    def read_csv(cls, filename, sep=",", header = None, compression=None, chunksize=None):
        return pd.read_csv(filename, header = header, compression=compression, chunksize=chunksize, sep=sep)

    @classmethod
    def to_csv(cls, _df, filename, index=False, sep=","):
        cls.mkdir_file(filename)
        _df.to_csv(filename, sep=sep, index=index)

    @classmethod
    def nowstr(cls):
        return str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))

    @classmethod
    def nowstrhms(cls):
        return str(datetime.datetime.now().strftime("%H-%M-%S"))

    @classmethod
    def Logger(cls, logfile_name):
        import logging
        import daiquiri

        log_fmt = '%(asctime)s %(filename)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
        daiquiri.setup(level=logging.DEBUG,
                       outputs=(
                           daiquiri.output.Stream(formatter=daiquiri.formatter.ColorFormatter(fmt=log_fmt)),
                           daiquiri.output.File(logfile_name, level=logging.DEBUG)
                       ))
        return daiquiri.getLogger(__name__)


    @classmethod
    def split_cv(cls, X, y, num_of_folds, DIR_TO_SAVED):
        """
        :param X: array
        :param y: array
        :param num_of_folds:[int, int, ...]
        :X_train, X_val, y_train, y_val are saved on DIr "data/n_folds_xx"
        """
        logger = cls.Logger(logfile_name=logfile_name)
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
                cls.to_csv(X_train, os.path.join(CV_DIR, "X_train_%s.csv" % i))
                cls.to_csv(X_val, os.path.join(CV_DIR, "X_val_%s.csv" % i))
                cls.to_csv(y_train, os.path.join(CV_DIR, "y_train_%s.csv" % i))
                cls.to_csv(y_val, os.path.join(CV_DIR, "y_val_%s.csv" % i))

