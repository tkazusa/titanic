# coding=utf-8

# write code...
import os
import gc

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from lightgbm.sklearn import LGBMClassifier
from hyperopt import hp, tpe, Trials, STATUS_OK, fmin
hopt_random_state = np.random.RandomState(3655)

from runner_base import RunnerBase
from model_lgbm_initial import ModelLgbm_initial
from util import Util
logger = Util.Logger()
from preprocess.split_cv import split_cv


APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
ORG_DATA_DIR = os.path.join(APP_ROOT, "data/")

ORG_TARGET_DATA = os.path.join(ORG_DATA_DIR, "target.csv")
TARGET_ID = os.path.join(ORG_DATA_DIR, "test_target.csv")

PREPROCESSED_DATA_DIR = os.path.join(APP_ROOT, 'data/preprocessed/')
PREPROCESSED_TRAIN_DATA = os.path.join(PREPROCESSED_DATA_DIR, "2017-10-23_preprocessed_train_data.csv.gz")
PREPROCESSED_TEST_DATA = os.path.join(PREPROCESSED_DATA_DIR, "2017-10-23_preprocessed_test_data.csv.gz")


class RunnerLgbm_initial(RunnerBase):
    """Runner base
    Provide interface for model runner class, which run model.
    """

    def __init__(self, n_folds_list):
        self.n_fold_list = n_folds_list

    def _set_model(self):
        return ModelLgbm_initial()

    def _fetch_preprocessed_data(self):
        raise NotImplementedError

    def load_X(self):
        logger.info("Preprocessed X load start")
        X = pd.read_csv(PREPROCESSED_TRAIN_DATA).values
        logger.info("X load end")
        self.X = X


    def load_y(self):
        logger.info("Preprocessed y load start")
        y = pd.read_csv(ORG_TARGET_DATA).values
        y_c, y_r = y.shape
        y = y.reshape(y_c, )
        logger.info("y load end")
        self.y = y

    def _calc_w(self, y):
        """calculate weight for imbalance data"""
        num_pos = y.sum()
        num_neg = y.shape[0] - num_pos
        w = num_neg/num_pos

        sample_weight = np.where(y == 1, w, 1)
        logger.info("calc sample weight neg: %s pos: %s" % (num_neg, num_pos))
        return sample_weight

    def find_best_cv(self):
        split_cv(self.X, self.y, self.n_folds_list, ORG_DATA_DIR)

        acc_score_means = []
        acc_score_vars = []
        for num_of_fold in self.n_folds_list:
            for num_of_fold in self.n_folds_list:
                logger.info("evaluating %s fold" % num_of_fold)
                CV_DIR = os.path.join(ORG_DATA_DIR, "n_folds_%s/" % num_of_fold)
                acc_score = []
                for i in range(num_of_fold):
                    logger.info("loading %s th cv data in %s folds" % (i, num_of_fold))
                    X_train = pd.read_csv(os.path.join(CV_DIR, "X_train_%s.csv") % i, header=None, sep="\t").values
                    X_val = pd.read_csv(os.path.join(CV_DIR, "X_val_%s.csv") % i, header=None, sep="\t").values
                    y_train = pd.read_csv(os.path.join(CV_DIR, "y_train_%s.csv") % i, header=None, sep="\t").values
                    y_c, y_r = y_train.shape
                    y_train = y_train.reshape(y_c, )
                    y_val = pd.read_csv(os.path.join(CV_DIR, "y_val_%s.csv") % i, header=None, sep="\t").values
                    y_c, y_r = y_val.shape
                    y_val = y_val.reshape(y_c, )
                    logger.info("end loading %s th cv data in %s folds" % (i, num_of_fold))
                    logger.info("X_train.shape: %s %s" % X_train.shape)
                    logger.info("X_val.shape: %s %s" % X_val.shape)
                    logger.info("y_train.shape: %s" % y_train.shape)
                    logger.info("y_val.shape: %s" % y_val.shape)

                    clf = LGBMClassifier(objective="binary",
                                         n_estimators=20)

                    weight_train = self._calc_w(y_train)

                    clf.fit(X_train, y_train,
                            sample_weight=weight_train,
                            eval_set=[(X_val, y_val)],
                            verbose=True)
                    y_pred = clf.predict(X_val)
                    logger.info("acc socore: %s folds, %s iteration" % (num_of_fold, i))
                    acc_score.append(accuracy_score(y_val, y_pred))
                logger.info("mean acc score of %s folds is %s" % (num_of_fold, np.mean(acc_score)))
                acc_score_means.append(np.mean(acc_score))
                logger.info("variance of acc score of %s folds is %s" % (num_of_fold, np.var(acc_score)))
                acc_score_vars.append(np.var(acc_score))
            for i in range(len(self.n_folds_list)):
                logger.info(
                    "===%s_folds=== mean acc:%s, var acc: %s " % (self.n_folds_list[i],
                                                                  acc_score_means[i],
                                                                  acc_score_vars[i])
                )

    def set_best_cv(self, n_fold):
        self._best_cv = n_fold

    def run_train(self):
        raise NotImplementedError

    def run_train_hopt(self):
        def score(params):
            params = {"max_depth": int(params["max_depth"]),
                      "subsample": round(params["subsample"], 3),
                      "colsample_bytree": round(params['colsample_bytree'], 3),
                      "num_leaves": int(params['num_leaves']),
                      "n_jobs": -2
                      # "is_unbalance": params['is_unbalance']
                      }

            clf = LGBMClassifier(n_estimators=500, learning_rate=0.05, **params)

            list_score_acc = []
            list_score_logloss = []

            skf = StratifiedKFold(n_splits=self._best_cv, shuffle=True, random_state=3655)
            sample_weight = self._calc_w(self.y)
            for train, val in skf.split(self.X, self.y):
                X_train, X_val = self.X[train], self.X[val]
                y_train, y_val = self.y[train], self.y[val]

                weight_train = sample_weight[train]
                weight_val = sample_weight[val]

                clf.fit(X_train, y_train,
                        sample_weight=weight_train,
                        eval_sample_weight=[weight_val],
                        eval_set=[(X_val, y_val)],

                        eval_metric="logloss",
                        early_stopping_rounds=0,
                        verbose=False
                        )

                _score_acc = accuracy_score(y_val, clf.predict(X_val), sample_weight=weight_val)
                _score_logloss = log_loss(y_val, clf.predict_proba(X_val), sample_weight=weight_val)

                list_score_acc.append(_score_acc)
                list_score_logloss.append(_score_logloss)
                """
                ##n_estimaters=0 causes error at .fit()
                if clf.best_iteration_ != -1:
                    list_best_iter.append(clf.best_iteration_)
                else:
                    list_best_iter.append(params['n_estimators'])
                break
                """
            # logger.info("n_estimators: {}".format(list_best_iter))
            # params["n_estimators"] = np.mean(list_best_iter, dtype=int)

            score_acc = (np.mean(list_score_acc), np.min(list_score_acc), np.max(list_score_acc))
            # logger.info("score_acc %s" % np.mean(list_score_acc))

            # score_logloss = (np.mean(list_score_logloss), np.min(list_score_logloss), np.max(list_score_logloss))
            # score_f1 = (np.mean(list_score_f1), np.min(list_score_f1), np.max(list_score_f1))
            # score_auc = (np.mean(list_score_auc), np.min(list_score_auc), np.max(list_score_auc))

            logloss = np.mean(list_score_logloss)
            return {'loss': logloss, 'status': STATUS_OK, 'localCV_acc': score_acc}

        space = {"max_depth": hp.quniform('max_depth', 1, 10, 1),
                 "subsample": hp.uniform('subsample', 0.3, 0.8),
                 "colsample_bytree": hp.uniform('colsample_bytree', 0.3, 0.8),
                 "num_leaves": hp.quniform('num_leaves', 5, 100, 1),
                 # "is_unbalance": hp.choice('is_unbalance', [True, False])
                 }

        trials = Trials()
        best_params = fmin(rstate=hopt_random_state, fn=score, space=space, algo=tpe.suggest, trials=trials, max_evals=10)
        logger.info("localCV_acc %s" %
                    list(filter(lambda x: x["loss"] == min(trials.losses()), trials.results))[0]["localCV_acc"][0])

        self.best_params = {"max_depth": int(best_params["max_depth"]),
                       "subsample": best_params["subsample"],
                       "colsample_bytree": best_params['colsample_bytree'],
                       "num_leaves": int(best_params['num_leaves'])
                       # "is_unbalance": best_params['is_unbalance']
                       }
        logger.info("best params are %s" % self.best_params)

        model = LGBMClassifier(n_estimators=500, learning_rate=0.05, **self.best_params)
        logger.info("start training with best params")
        model.fit(self.X, self.y)

        imp = pd.DataFrame(model.feature_importances_, columns=['imp'])
        imp.index = pd.read_csv(PREPROCESSED_TRAIN_DATA).columns
        logger.info("important features are %s" % imp)
        with open('features.py', 'a') as f:
            f.write('FEATURE = ["' + '","'.join(map(str, imp[imp['imp'] > 0].index.values)) + '"]\n')

        Util.dump(model, os.path.join("models","model_lgbm_initial.pkl"))
        logger.info("model with best params is saved")

    def run_predict(self):
        model = Util.load(os.path.join("models", "model_lgbm_initial.pkl"))

        all_test_data = pd.read_csv(PREPROCESSED_TEST_DATA, compression="gzip", chunksize=1000)
        logger.info("end load test data size: %s %s" % all_test_data.shape)
        df_submit = pd.DataFrame()
        for i, df in enumerate(all_test_data):
            print(df.columns)
            df_submit = pd.read_csv(TARGET_ID)
            df_submit["Survived"] = model.predict(df)
            df_submit["Proba"] = model.predict_proba(df)[:, 1]
            logger.info("end predict")

            logger.info("chunk %s: %s" % (i, df_submit.shape[0]))
            del df
            gc.collect()

        Util.to_csv(df_submit, "submit.csv", index=False)


if __name__ == "__main__":
    n_folds_list = [3, 5, 10]
    runner = RunnerLgbm_initial(n_folds_list=n_folds_list)
    runner.load_X()
    runner.load_y()
    #runner.find_best_cv(n_folds_list)
    runner.set_best_cv(5)
    runner.run_train_hopt()
