# coding=utf-8

# write code...

import pandas as pd
import xgboost as xgb


ORG_TRAIN_DATA_PATH = "~/data/train.csv"
ORG_TEST_DATA_PATH = "~/data/test.csv"

train = pd.read_csv(ORG_TRAIN_DATA_PATH)
test = pd.read_csv(ORG_TEST_DATA_PATH)

train["data_set"] = 1
test["data_set"] = 0

data = train.drop(labels = "Survived", axis = 1).append(test, ignore_index = True)
data.to_csv("~/data/data.csv", index=False)

print(train["data_set"].head())

print("~~checking data type~~~")
print("======train=======")
print(train.dtypes)

print("")
print("======test========")
print(test.dtypes)

print("")
print("~~cheking data shape")
print('Having {} training rows and {} test rows.'.format(train.shape[0], test.shape[0]))
print('Having {} training columns and {} test columns.'.format(train.shape[1], test.shape[1]))

print("")
print("checking missing values")
print("======train=======")
print(train.isnull().sum())
print(train.info())

print("")
print("======test========")
print(test.isnull().sum())
print(test.info())
