# coding=utf-8

# write code...

# coding=utf-8

# write code...

import os
import logging

import pandas as pd
import daiquiri

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
ORG_DATA_DIR = os.path.join(APP_ROOT, "data/")

ORG_TRAIN_DATA = os.path.join(ORG_DATA_DIR, "train.csv")
ORG_TEST_DATA = os.path.join(ORG_DATA_DIR, "test.csv")
ORG_CONCAT_DATA = os.path.join(ORG_DATA_DIR, "data.csv")

log_fmt = '%(asctime)s %(filename)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
daiquiri.setup(level=logging.DEBUG,
               outputs=(
                   daiquiri.output.Stream(formatter=daiquiri.formatter.ColorFormatter(fmt=log_fmt)),
                   daiquiri.output.File("predict.log", level=logging.DEBUG)
               ))
logger = daiquiri.getLogger(__name__)

if __name__ == "__main__":
    logger.info("load start")

    train_target = pd.read_csv(ORG_TRAIN_DATA, usecols=["Survived"])
    logger.info("target size %s %s" % train_target.shape)

    test_target_ID = pd.read_csv(ORG_TEST_DATA, usecols=["PassengerId"])

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
    train_target.to_csv("~/data/train_target.csv", index=False)

    logger.info("writing test target id")
    test_target_ID.to_csv("~/data/test_target.csv", index=False)

    logger.info("writing data")
    data.to_csv(ORG_CONCAT_DATA, index=False)