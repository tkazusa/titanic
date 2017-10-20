# coding=utf-8

# write code...

import pandas as pd


ORG_TRAIN_DATA_PATH = "~/data/train.csv"
ORG_TEST_DATA_PATH = "~/data/test.csv"

train = pd.read_csv(ORG_TRAIN_DATA_PATH)
test = pd.read_csv(ORG_TEST_DATA_PATH)

train["data_set"] = "train"
test["data_set"] = "test"

print("~~export target data for tableau as csv file")
target = train["Survived"]
target.to_csv("~/data/target.csv", index=False)

print("~~export data for tableau as csv file")
data = train.drop(labels = "Survived", axis = 1).append(test, ignore_index = True)
data.to_csv("~/data/data.csv", index=False)

print("~~checking data type~~~")
print("======train=======")
print(train.dtypes)
print(type(train.dtypes))


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


print("")
print("~~cheking unique values for categorical features")
print("")
print("======train=======")
categorical_features_train = train.ix[:, train.dtypes == object].columns
for categorical_feature in categorical_features_train:
    print("{} has {} unique features".format(categorical_feature, len(train[categorical_feature].unique())))
    print("")
    print(train[categorical_feature].unique())
    print("")
    print("========")

print("======test=======")
categorical_features_test = test.ix[:, test.dtypes == object].columns
for categorical_feature in categorical_features_test:
    print("{} has {} unique features".format(categorical_feature, len(test[categorical_feature].unique())))
    print("")
    print(test[categorical_feature].unique())
    print("")
    print("========")


print("")
print("diff of unique value between train and test")

for categorical_feature in categorical_features_train:
    print("{}:{}".format(categorical_feature, len(set(train[categorical_feature].unique()).symmetric_difference(set(test[categorical_feature].unique())))))

for categorical_feature in categorical_features_train:
    num_diff = len(set(train[categorical_feature].unique()).symmetric_difference(test[categorical_feature].unique()))
    if num_diff > 0 :
        print("=========")
        print(categorical_feature)
        print("instersection between train and test")
        print(len(set(train[categorical_feature].unique()).intersection(set(test[categorical_feature].unique()))))
        print(set(train[categorical_feature].unique()).intersection(set(test[categorical_feature].unique())))
        print("")
        print("only train has")
        print(len(set(train[categorical_feature].unique()).difference(set(test[categorical_feature].unique()))))
        print(set(train[categorical_feature].unique()).difference(set(test[categorical_feature].unique())))
        print("")
        print("only test has")
        print(len(set(test[categorical_feature].unique()).difference(set(train[categorical_feature].unique()))))
        print(set(test[categorical_feature].unique()).difference(set(train[categorical_feature].unique())))
        print("")

numerical_features = test.ix[:, train.dtypes != object].columns

print("")
print("===train===")
print(train[numerical_features].describe())
print("")
print("===test===")
print(test[numerical_features].describe())

print(data[numerical_features].corr)
