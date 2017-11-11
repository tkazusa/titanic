# coding=utf-8

# write code...
import os
import gc
from datetime import date
import logging

import pandas as pd
import daiquiri

from bases.preprocesser import PreprocesserBase

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
ORG_DATA_DIR = os.path.join(APP_ROOT, "data/")

ORG_TRAIN_DATA = os.path.join(ORG_DATA_DIR, "train.csv")
ORG_TEST_DATA = os.path.join(ORG_DATA_DIR, "test.csv")
ORG_CONCAT_DATA = os.path.join(ORG_DATA_DIR, "data.csv")

PREPROCESSED_DATA_DIR = os.path.join(APP_ROOT, 'data/preprocessed/')
PREPROCESSED_TRAIN_DATA = os.path.join(PREPROCESSED_DATA_DIR, str(date.today().isoformat())+"_preprocessed_train_data.csv.gz")
PREPROCESSED_TEST_DATA = os.path.join(PREPROCESSED_DATA_DIR, str(date.today().isoformat())+"_preprocessed_test_data.csv.gz")


log_fmt = '%(asctime)s %(filename)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
daiquiri.setup(level=logging.DEBUG,
               outputs=(
                   daiquiri.output.Stream(formatter=daiquiri.formatter.ColorFormatter(fmt=log_fmt)),
                   daiquiri.output.File("predict.log", level=logging.DEBUG)
               ))
logger = daiquiri.getLogger(__name__)


class PreprocesserInitial(PreprocesserBase):

    def __init__(self):
        self.data = None

    def fetch_origin_data(self, ORG_DATA_PATH):
        logger.info("loading original data")
        data = pd.read_csv(ORG_DATA_PATH)
        logger.info("original data size %s,%s train %s test %s"  % (data.shape[0], data.shape[1], data[data["data_set"]=="train"].shape[0], data[data["data_set"]=="test"].shape[0]))
        logger.info("load end")

        self.data = data

    def fillna_median(self, list_col_fillna):
        for col_name in list_col_fillna:
            logger.info("filling NA in the %s column with median" % col_name)
            self.data[col_name].fillna(self.data[col_name].median(), inplace=True)

    def dummy(self, list_col_dummy):
        for col_name in list_col_dummy:
            logger.info("dummying in the %s column, unique value size: %s \n %s"
                        % (col_name, len(self.data[col_name].unique()), self.data[col_name].unique()))
            dummy = pd.get_dummies(self.data[col_name])
            self.data = pd.concat((self.data, dummy), axis=1)
            del dummy
            gc.collect()

            self.data = self.data.drop("%s" % col_name, axis=1)
        logger.info("dummied data size %s,%s" % self.data.shape)

    def drop_column(self, list_col_drop):
        for col_name in list_col_drop:
            self.data = self.data.drop("%s" % col_name, axis=1)
            logger.info("drop the %s column, data size:%s,%s" % (col_name, self.data.shape[0], self.data.shape[1]))

    def drop_na_samples(self):
        self.data = self.data.dropna(axis=0)
        logger.info("drop sample with na feature data size %s, %s" % self.data.shape)

    def save_train_data(self):
        train = self.data[self.data["data_set"] == "train"]
        train = train.drop("data_set", axis=1)
        logger.info("train data \n %s" % train.head(2))
        logger.info("writing preprocessed train data size:%s,%s" % train.shape)
        train.to_csv(PREPROCESSED_TRAIN_DATA, index=False, compression="gzip")

    def save_test_data(self):
        test = self.data[self.data["data_set"] == "test"]
        test = test.drop("data_set", axis=1)
        logger.info("test data \n %s" % test.head(2))
        logger.info("writing preprocessed test data size:%s,%s" % test.shape)
        test.to_csv(PREPROCESSED_TEST_DATA, index=False, compression="gzip")
