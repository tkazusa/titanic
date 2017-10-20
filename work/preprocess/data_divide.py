# coding=utf-8

# write code...

import os
from logging import getLogger, basicConfig

import pandas as pd

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
ORG_DATA_DIR = os.path.join(APP_ROOT, "data/")

ORG_TRAIN_DATA = os.path.join(ORG_DATA_DIR, "train.csv")
ORG_TEST_DATA = os.path.join(ORG_DATA_DIR, "test.csv")
ORG_CONCAT_DATA = os.path.join(ORG_DATA_DIR, "data.csv")

log_fmt = '%(asctime)s %(filename)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
basicConfig(format=log_fmt,datefmt='%Y-%m-%d/%H:%M:%S',level='DEBUG')

logger = getLogger(__name__)

if __name__ == "__main__":
    logger.info("load start")

    target = pd.read_csv(ORG_TRAIN_DATA, usecols=["Survived"])
    logger.info("target size %s %s" % target.shape)

    train = pd.read_csv(ORG_TRAIN_DATA).drop(labels=["Survived"], axis=1)
    logger.info("train size %s %s" % train.shape)

    test = pd.read_csv(ORG_TEST_DATA)
    logger.info("test size %s %s" % test.shape)

    logger.info("load end")

    train["data_set"] = "train"
    test["data_set"] = "test"

    logger.info("concatinating train & test data")
    data = pd.concat([train, test], axis=0)
    logger.info("data size %s %s" % data.shape)

    logger.info("writing target")
    target.to_csv("~/data/target.csv", index=False)

    logger.info("writing data")
    data.to_csv(ORG_CONCAT_DATA, index=False)

