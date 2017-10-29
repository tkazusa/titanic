# coding=utf-8

# write code...
# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import os
# load or create your dataset
print('Load data...')

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
ORG_DATA_DIR = os.path.join(APP_ROOT, "data/")

ORG_TARGET_DATA = os.path.join(ORG_DATA_DIR, "target.csv")

PREPROCESSED_DATA_DIR = os.path.join(APP_ROOT, 'data/preprocessed/')
PREPROCESSED_TRAIN_DATA = os.path.join(PREPROCESSED_DATA_DIR, "2017-10-23_preprocessed_train_data.csv.gz")

X_train = pd.read_csv(PREPROCESSED_TRAIN_DATA).values
y_train = pd.read_csv(ORG_TARGET_DATA).values
y_c, y_r = y_train.shape
y_train = y_train.reshape(y_c, )

print('Start training...')
# train
gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        learning_rate=0.05)
gbm.fit(X_train, y_train,
        #eval_set=[(X_test, y_test)],
        eval_metric='l1')

print('Start predicting...')
# predict
#y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# eval
#print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# feature importances
#print('Feature importances:', list(gbm.feature_importances_))

# other scikit-learn modules
estimator = lgb.LGBMRegressor(num_leaves=31)

param_grid = {
    'learning_rate': [0.01, 0.1, 1, 0.3,0.5],
    'n_estimators': [1,2,3,4,5,6,7,8,9,10,11,12,13]
}

gbm = GridSearchCV(estimator, param_grid)

gbm.fit(X_train, y_train)

print('Best parameters found by grid search are:', gbm.best_params_)