# coding=utf-8

# write code...
import os
from datetime import date
import logging

import daiquiri

from preprocess.preprocess_20171023_initial import PreprocesserInitial


APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
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


def main():
    list_col_fillna = ["Age", "Fare"]
    list_col_dummy = ["Sex", "Embarked"]
    list_col_drop = ["PassengerId", "Ticket", "Cabin", "Name"]

    prep = PreprocesserInitial()
    prep.fetch_origin_data(ORG_DATA_PATH=ORG_CONCAT_DATA)
    prep.fillna_median(list_col_fillna=list_col_fillna)
    prep.dummy(list_col_dummy=list_col_dummy)
    prep.drop_column(list_col_drop=list_col_drop)
    prep.drop_na_samples()
    prep.save_train_data()
    prep.save_test_data()

if __name__ == "__main__":
    main()