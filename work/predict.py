# coding=utf-8

# write code...
import os
import pickle
import logging
import gc

import pandas as pd
import numpy
import daiquiri

from features import FEATURE

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
ORG_DATA_DIR = os.path.join(APP_ROOT, "data/")

ORG_TARGET_DATA = os.path.join(ORG_DATA_DIR, "target.csv")

TARGET_ID = os.path.join(ORG_DATA_DIR, "test_target.csv")

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

def main():
    logger.info("start loading data")
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)


    all_test_data = pd.read_csv(PREPROCESSED_TEST_DATA, compression="gzip", chunksize=1000)
    #logger.info("end load test data size: %s %s" % all_test_data.shape)
    df_submit = pd.DataFrame()
    for i, df in enumerate(all_test_data):
        print(df.columns)
        df_submit = pd.read_csv(TARGET_ID)
        df_submit["Survived"]= model.predict(df)
        df_submit["Proba"] = model.predict_proba(df)[:,1]
        logger.info("end predict")

        logger.info("chunk %s: %s" %(i, df_submit.shape[0]))
        del df
        gc.collect()

    df_submit.to_csv("submit.csv", index=False)

if __name__ == "__main__":
    main()






