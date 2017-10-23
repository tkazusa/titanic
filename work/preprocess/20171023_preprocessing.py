# coding=utf-8

# write code...

import os
import gc
from datetime import date
from logging import getLogger, basicConfig
import pandas as pd
from multiprocessing import Pool


APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
ORG_DATA_DIR = os.path.join(APP_ROOT, "data/")

ORG_TRAIN_DATA = os.path.join(ORG_DATA_DIR, "train.csv")
ORG_TEST_DATA = os.path.join(ORG_DATA_DIR, "test.csv")
ORG_CONCAT_DATA = os.path.join(ORG_DATA_DIR, "data.csv")

PREPROCESSED_DATA_DIR = os.path.join(APP_ROOT, 'data/preprocessed/')
PREPROCESSED_TRAIN_DATA = os.path.join(PREPROCESSED_DATA_DIR, str(date.today().isoformat())+"_preprocessed_train_data.csv.gz")
PREPROCESSED_TEST_DATA = os.path.join(PREPROCESSED_DATA_DIR, str(date.today().isoformat())+"_preprocessed_test_data.csv.gz")


log_fmt = '%(asctime)s %(filename)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
basicConfig(format=log_fmt,datefmt='%Y-%m-%d/%H:%M:%S',level='DEBUG')

logger = getLogger(__name__)


if __name__ == "__main__":
    logger.info("load data 0")
    ORG_DATA_PATH =  "~/data/data.csv"
    data = pd.read_csv(ORG_DATA_PATH)
    logger.info("size %s %s" % data.shape)
    logger.info("load end")

    data["Age"].fillna(data.Age.median(), inplace = True)
    logger.info("filling NA in the Age column with median")

    data["Fare"].fillna(data.Fare.median(), inplace = True)
    logger.info("filling NA in the Fare column with median")



    sex_dummy = pd.get_dummies(data["Sex"])
    data = pd.concat((data,sex_dummy), axis=1)
    del sex_dummy
    gc.collect()

    emb_dummy = pd.get_dummies(data["Embarked"])
    data = pd.concat((data,emb_dummy), axis=1)
    del  emb_dummy
    gc.collect()

    data = data.drop("Sex",axis=1)
    data = data.drop("Embarked", axis=1)
    logger.info("size %s %s " % data.shape)

    data = data.drop("PassengerId", axis=1)
    data = data.drop("Ticket", axis=1)
    data = data.drop("Cabin", axis=1)
    data = data.drop("Name", axis=1)
    #data = data.drop("PassengerId", axis=1)

    train = data[data["data_set"] == "train"]
    train = train.drop("data_set", axis=1)
    print(train.head())
    logger.info("size %s %s " % train.shape)
    logger.info("writing preprocessed train")
    train.to_csv(PREPROCESSED_TRAIN_DATA, index=False, compression="gzip")

    test = data[data["data_set"] == "test"]
    test = test.drop("data_set", axis=1)
    logger.info("size %s %s " % test.shape)
    logger.info("writing preprocessed test")
    test.to_csv(PREPROCESSED_TEST_DATA, index=False, compression="gzip")




