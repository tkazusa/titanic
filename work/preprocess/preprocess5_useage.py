# coding=utf-8

# write code...
import os
import gc
from datetime import date
import logging

import pandas as pd
import numpy as np

from util import Util
logfile_name = "logs/" + str(date.today().isoformat())+ ".log"
logger = Util.Logger(logfile_name=logfile_name)
from bases.preprocesser import PreprocesserBase

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
ORG_DATA_DIR = os.path.join(APP_ROOT, "data/")

ORG_TRAIN_DATA = os.path.join(ORG_DATA_DIR, "train.csv")
ORG_TEST_DATA = os.path.join(ORG_DATA_DIR, "test.csv")
ORG_CONCAT_DATA = os.path.join(ORG_DATA_DIR, "data.csv")

PREPROCESSED_DATA_DIR = os.path.join(APP_ROOT, 'data/preprocessed/')
PREPROCESSED_TRAIN_DATA = os.path.join(PREPROCESSED_DATA_DIR, "preprocessed5_train_data.csv")
PREPROCESSED_TEST_DATA = os.path.join(PREPROCESSED_DATA_DIR, "preprocessed5_test_data.csv")


class PreprocesserUseAge(PreprocesserBase):

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


    def use_honorific(self):
        honorifics = ["Master", "Mrs", "Miss", "Mr"]
        for honorific in honorifics:
            self.data[honorific] = self.data["Name"].where(self.data["Name"].str.contains(honorific), 0)
            self.data[honorific] = self.data[honorific].where(self.data[honorific] == 0, 1)

        self.data["honorific_Unknown"] = self.data["Name"].where(self.data["Name"].str.contains("Master")
                                                    | self.data["Name"].str.contains("Mr")
                                                    | self.data["Name"].str.contains("Miss")
                                                    | self.data["Name"].str.contains("Mrs")
                                                    , 1)
        self.data["honorific_Unknown"] = self.data["honorific_Unknown"].where(self.data["honorific_Unknown"] == 1, 0)


    def use_cabin_information(self):
        '''add U as unknown and extract cabin class'''
        self.data["U"] = self.data["Cabin"].where(self.data["Cabin"].isnull(), 0)
        self.data["U"] = self.data["U"].where(~self.data["U"].isnull(), 1)
        cabin_class = ["A","B","C","D","E","F"]
        for cabin in cabin_class:
            self.data["Cabin_%s" % cabin] = self.data["Cabin"].where(self.data["Cabin"].str.contains(cabin), 0)
            self.data["Cabin_%s" % cabin] = self.data["Cabin_%s" % cabin].where(self.data["Cabin_%s" % cabin] == 0, 1)

        self.data = self.data.drop("Cabin", axis=1)

    def use_familysize(self):
        self.data["familysize"] = self.data["SibSp"].values + self.data["Parch"].values + 1

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

    def estimate_age(self):
        from sklearn.ensemble import RandomForestRegressor
        logger.info("estimating missing age values with RF")

        self.data["Age_Unknown"] = self.data["Age"].where(self.data["Age"].isnull(), 0)
        self.data["Age_Unknown"] = self.data["Age"].where(self.data["Age"] == 0, 1)

        self.data["Fillna_Fare"] = self.data["Fare"]
        self.data["Fillna_Fare"].fillna(self.data["Fillna_Fare"].median(), inplace=True)

        age_train = self.data[~self.data["Age"].isnull()]
        age_test = self.data[self.data["Age"].isnull()]

        age_train_idx = age_train["PassengerId"]
        age_X_train = age_train.drop(["Age", "Age_Unknown", "Fare", "PassengerId", "Ticket", "Name", "data_set"], axis=1)
        age_y_train = age_train["Age"].reset_index(drop=True)

        age_test_idx =  age_test["PassengerId"]
        age_X_test = age_test.drop(["Age", "Age_Unknown", "Fare", "PassengerId", "Ticket", "Name", "data_set"], axis=1)

        rfr = RandomForestRegressor(n_estimators=200, n_jobs=-2)
        rfr.fit(age_X_train, age_y_train)
        age_y_pred = np.trunc(rfr.predict(age_X_test))

        age_pred_test = pd.concat([age_test_idx.reset_index(drop=True), pd.DataFrame(age_y_pred, columns = ["Age_pred"])], axis=1)
        age_pred_train = pd.concat([age_train_idx.reset_index(drop=True), age_y_train.rename("Age_pred")], axis=1)
        age_pred = pd.concat([age_pred_train, age_pred_test], axis=0).sort_values(by=["PassengerId"], ascending=True).reset_index(drop=True)

        self.data = pd.concat([self.data, age_pred["Age_pred"]], axis=1)
        self.data = self.data.drop("Age", axis=1)
        self.data = self.data.drop("Fillna_Fare", axis=1)

    def estimate_fare(self):
        from sklearn.ensemble import RandomForestRegressor
        logger.info("estimating missing fare values with RF")

        fare_train = self.data[~self.data["Fare"].isnull()]
        fare_test = self.data[self.data["Fare"].isnull()]

        fare_train_idx = fare_train["PassengerId"]
        fare_X_train = fare_train.drop(["Fare", "PassengerId", "Ticket", "Name", "data_set"], axis=1)
        fare_y_train = fare_train["Fare"].reset_index(drop=True)

        fare_test_idx =  fare_test["PassengerId"]
        fare_X_test = fare_test.drop(["Fare", "PassengerId", "Ticket", "Name", "data_set"], axis=1)

        rfr = RandomForestRegressor(n_estimators=100, max_depth=3)
        rfr.fit(fare_X_train, fare_y_train)
        print(fare_X_test.head())
        fare_y_pred = np.trunc(rfr.predict(fare_X_test))

        fare_pred_test = pd.concat([fare_test_idx.reset_index(drop=True), pd.DataFrame(fare_y_pred, columns = ["Fare_pred"])], axis=1)
        fare_pred_train = pd.concat([fare_train_idx.reset_index(drop=True), fare_y_train.rename("Fare_pred")], axis=1)
        fare_pred = pd.concat([fare_pred_train, fare_pred_test], axis=0).sort_values(by=["PassengerId"], ascending=True).reset_index(drop=True)

        self.data = pd.concat([self.data, fare_pred["Fare_pred"].round(1)], axis=1)
        self.data = self.data.drop("Fare", axis=1)

    def drop_na_samples(self):
        self.data = self.data.dropna(axis=0)
        logger.info("drop sample with na feature data size %s, %s" % self.data.shape)

    def save_train_data(self):
        train = self.data[self.data["data_set"] == "train"]
        train = train.drop("data_set", axis=1)
        logger.info("train data \n %s" % train.head(2))
        logger.info("writing preprocessed train data size:%s,%s" % train.shape)
        Util.to_csv(train, PREPROCESSED_TRAIN_DATA)

    def save_test_data(self):
        test = self.data[self.data["data_set"] == "test"]
        test = test.drop("data_set", axis=1)
        logger.info("test data \n %s" % test.head(2))
        logger.info("writing preprocessed test data size:%s,%s" % test.shape)
        Util.to_csv(test, PREPROCESSED_TEST_DATA, index=False)
