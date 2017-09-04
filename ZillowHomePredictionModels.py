import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
import random


import ZillowHighCardinalityDataHandler

np.random.seed(0)
random.seed(0)
random_state = 0

# xgboost fix
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb



# Global Parameters
XGB_WEIGHT = 0.6415
BASELINE_WEIGHT = 0.0056
OLS_WEIGHT = 0.0828

XGB1_WEIGHT = 0.80#0.8083#0.9013 XGB Correlation#  # Weight of first in combination of two XGB models

BASELINE_PRED = 0.0114572195563 # Baseline based on mean of training data, per https://www.kaggle.com/bbrandt/using-avg-for-each-landusetypeid-for-prediction
#BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg

xgb_params1 = {
            'eta': 0.0371,
            'max_depth': 5,
            'subsample': 0.81,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'lambda': 0.8,
            'alpha': 0.4,
            'silent': 1
        }

xgb_params2 = {
            'eta': 0.03328,
            'max_depth': 6,
            'subsample': 0.79,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'silent': 1
        }

class ZillowHomePredictionModels:
    # This section is (I think) originally derived from SIDHARTH's script:
    #   https://www.kaggle.com/sidharthkumar/trying-lightgbm
    # which was forked and tuned by Yuqing Xue:
    #   https://www.kaggle.com/yuqingxue/lightgbm-85-97
    # and updated by Andy Harless:
    #   https://www.kaggle.com/aharless/lightgbm-with-outliers-remaining
    def generate_lgb_model(self, X_train, y_train):
        drop_cols = [
            'propertyzoningdesc', 'propertycountylandusecode',
            'fireplacecnt', 'fireplaceflag'
        ]
        # X_train = X_train.drop(drop_cols, axis=1)
        d_train = lgb.Dataset(X_train, label=y_train)
        lgb_params = {
            'max_bin': 10,
            'learning_rate': 0.0021,  # shrinkage_rate
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mae',
            'sub_feature': 0.345,
            'bagging_fraction': 0.85,  # sub_row
            'bagging_freq': 40,
            'num_leaves': 512,  # num_leaf
            'min_data': 500,  # min_data_in_leaf
            'min_hessian': 0.05,  # min_sum_hessian_in_leaf
            'verbose': 0,
            'feature_fraction_seed': 2,
            'bagging_seed': 3,
        }

        clf = lgb.train(lgb_params, d_train, 430)
        return clf

    def MAE(self, y, ypred):
        # logerror=log(Zestimate)−log(SalePrice)
        return np.sum([abs(y[i] - ypred[i]) for i in range(len(y))]) / len(y)

    def get_ols_features(self, df:pd.DataFrame):
        ols_df = df.copy()
        ols_df["transactiondate"] = pd.to_datetime(ols_df["transactiondate"])
        ols_df["transactiondate_year"] = ols_df["transactiondate"].dt.year
        ols_df["transactiondate_month"] = ols_df["transactiondate"].dt.month
        ols_df['transactiondate'] = ols_df['transactiondate'].dt.quarter
        ols_df = ols_df.fillna(-1.0)

        to_remove = [df.columns[c]for c in range(len(df.columns)) if df.dtypes[c] == 'int32']
        to_remove = ['logerror', 'parcelid'] + to_remove
        valid_columns = [c for c in ols_df.columns if c not in to_remove]
        return ols_df[valid_columns]

    def get_ols_prediction(self, ols_model, properties_df, transaction_date):
        ols_x_test = properties_df.copy()
        ols_x_test["transactiondate"] = transaction_date
        ols_x_test = self.get_ols_features(ols_x_test)
        return self.__get_model_prediction__(ols_model, ols_x_test)

    # This section is derived from the1owl's notebook:
    #    https://www.kaggle.com/the1owl/primer-for-the-zillow-pred-approach
    # which Andy Harless updated and made into a script:
    #    https://www.kaggle.com/aharless/updated-script-version-of-the1owl-s-basic-ols
    def generate_ols_model(self, x_train, transaction_date, y_train):
        np.random.seed(17)
        random.seed(17)
        ols_x_train = x_train.copy()
        ols_x_train['transactiondate'] = transaction_date
        ols_x_train = self.get_ols_features(ols_x_train)
        reg = LinearRegression(n_jobs=-1)
        reg.fit(ols_x_train, y_train)
        return reg

    #https://www.kaggle.com/mubashir44/simple-ensemble-model-stacking
    def get_oof(self, clf, x_train, y_train, x_test):
        # Some useful parameters which will come in handy later on
        ntrain = x_train.shape[0]
        ntest = x_test.shape[0]
        NFOLDS = 5  # set folds for out-of-fold prediction
        kf = KFold(n_splits=NFOLDS, random_state=random_state)

        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf.split(x_train,  y_train)):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    def get_oof_for_xgboost(self, x_train, y_train, x_test, xgb_params, num_boost_rounds):
        # Some useful parameters which will come in handy later on
        ntrain = x_train.shape[0]
        ntest = x_test.shape[0]
        SEED = 42  # for reproducibility
        NFOLDS = 5  # set folds for out-of-fold prediction
        kf = KFold(n_splits=NFOLDS, random_state=random_state)
        xgb_params['base_score'] = y_train.mean()

        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))
        x_train = x_train.values
        x_test = x_test.values

        for i, (train_index, test_index) in enumerate(kf.split(x_train)):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            clf = xgb.train(xgb_params, xgb.DMatrix(x_tr, y_tr), num_boost_round=num_boost_rounds)

            oof_train[test_index] = clf.predict(xgb.DMatrix(x_te))
            oof_test_skf[i, :] = clf.predict(xgb.DMatrix(x_test))

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    # This section is originally derived from Infinite Wing's script:
    #   https://www.kaggle.com/infinitewing/xgboost-without-outliers-lb-0-06463
    # inspired by this thread:
    #   https://www.kaggle.com/c/zillow-prize-1/discussion/33710
    # but the code has gone through a lot of changes since then
    # High cardinality treatment was taken from:
    # https://www.kaggle.com/scirpus/genetic-programming-lb-0-0643904
    def generate_combined_xgb_pred(self, x_train, y_train, x_test):
        #highCardDataHandler = ZillowHighCardinalityDataHandler.ZillowHighCardinalityDataHandler(x_train, y_train, x_test)
        #x_train, x_test = highCardDataHandler.high_cardinality_to_mean()
        #x_train.drop('y', axis=1, inplace=True)
        #x_test.drop('y', axis=1, inplace=True)

        # dtest = xgb.DMatrix(x_test)
        # model1 = self.generate_xgb_model(x_train, y_train, xgb_params1, 250)
        # xgb_test_pred1 = self.__get_model_prediction__(model1, dtest)
        # model2 = self.generate_xgb_model(x_train, y_train, xgb_params2, 150)
        # xgb_test_pred2 = self.__get_model_prediction__(model2, dtest)
        xgb_train_pred1, xgb_test_pred1 = self.get_oof_for_xgboost(x_train, y_train, x_test, xgb_params1, 250)
        xgb_train_pred2, xgb_test_pred2 = self.get_oof_for_xgboost(x_train, y_train, x_test, xgb_params2, 150)

        # new_x_train = pd.DataFrame({"model1": xgb_train_pred1.flatten(),
        #                             "model2": xgb_train_pred2.flatten()})
        # new_x_test = pd.DataFrame({"model1": xgb_test_pred1.flatten(),
        #                            "model2": xgb_test_pred2.flatten()})
        #combined_xgb_model = self.generate_xgb_model(new_x_train, y_train, xgb_params1, 100)
        combined_xgb_pred = XGB1_WEIGHT*xgb_test_pred1.flatten() + (1-XGB1_WEIGHT)*xgb_test_pred2.flatten()
        #new_d_test = xgb.DMatrix(new_x_test)
        #combined_xgb_pred = self.__get_model_prediction__(combined_xgb_model, new_d_test)

        #sel = PLSRegression(n_components=1,)
        #pca_x_train = sel.fit_transform(new_x_train, y_train)
        #pca_x_test = sel.transform(new_x_test)

        #reg = GradientBoostingRegressor(learning_rate=0.3, n_estimators=200, subsample=0.8, max_depth=5, alpha=0.3)
        #reg.fit(new_x_train, y_train)
        #combined_xgb_pred = self.__get_model_prediction__(reg, new_x_test)

        return combined_xgb_pred

    def generate_xgb_lgb_combined_predictions(self, x_train, y_train, x_test):
        lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT) / (1 - OLS_WEIGHT)
        xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
        baseline_weight0 = BASELINE_WEIGHT / (1 - OLS_WEIGHT)
        xgb_pred = self.generate_combined_xgb_pred(x_train, y_train, x_test)

        lgb_drop_cols = [
            'propertyzoningdesc', 'propertycountylandusecode',
            #'fireplacecnt', 'fireplaceflag'
        ]
        lgb_x_train = x_train.drop(lgb_drop_cols, axis=1)
        lgb_x_test = x_test.drop(lgb_drop_cols, axis=1)
        lgb_pred = self.__get_model_prediction__(self.generate_lgb_model(lgb_x_train, y_train),
                                                 lgb_x_test)
        pred0 = xgb_weight0 * xgb_pred + baseline_weight0 * BASELINE_PRED + lgb_weight * lgb_pred
        return pred0

    def generate_all_combined_predictions(self, merged_df, x_test, x_test_index, x_train, y_train, scaler=None):
        dates = ['2016-10-01', '2016-11-01', '2016-12-01', '2017-10-01', '2017-11-01', '2017-12-01']
        columns = ['201610', '201611', '201612', '201710', '201711', '201712']
        output = pd.DataFrame({'ParcelId': x_test_index})
        pred0 = self.generate_xgb_lgb_combined_predictions(x_train, y_train, x_test)
        ols_model = self.generate_ols_model(x_train, merged_df["transactiondate"].values, y_train)
        for i in range(len(dates)):
            transaction_date = dates[i]
            ols_pred = self.get_ols_prediction(ols_model, x_test, transaction_date)
            pred = OLS_WEIGHT * ols_pred + (1 - OLS_WEIGHT) * pred0
            if not scaler is None:
                pred = scaler.inverse_transform(pred)
            output[columns[i]] = [float(format(x, '.4f')) for x in pred]

        return output

    def __get_model_prediction__(self, model, x_test):
        return model.predict(x_test)

    def generate_xgb_model(self, X_train, y_train, xgb_params, boost_rounds) -> xgb.Booster:
        y_mean = y_train.mean()
        dtrain = xgb.DMatrix(X_train, y_train)
        xgb_params['base_score'] = y_mean

        # cv_result = xgb.cv(xgb_params,
        #                    dtrain,
        #                    nfold=10,
        #                    num_boost_round=boost_rounds,
        #                    early_stopping_rounds=5,
        #                    verbose_eval=10,
        #                    show_stdv=False
        #                    )
        # num_boost_rounds = len(cv_result)
        num_boost_rounds = boost_rounds
        print("Number of boost rounds:")
        print(num_boost_rounds)
        # train model
        model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

        mweight = pd.DataFrame.from_dict(model.get_score(), orient="index")
        mweight.columns = ["weight"]
        mgain = pd.DataFrame.from_dict(model.get_score(importance_type='gain'), orient="index")
        mgain.columns = ["gain"]
        mcover = pd.DataFrame.from_dict(model.get_score(importance_type='cover'), orient="index")
        mcover.columns = ["cover"]
        mscore = mweight.merge(mgain, left_index=True, right_index=True)
        mscore = mscore.merge(mcover, left_index=True, right_index=True)
        print("Model Score")
        print(mscore.sort_values("gain", ascending=False))
        return model