import os

import gc
import lightgbm as lgb
import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.serialization import load_ssh_public_key
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import LinearSVR
import random


import ZillowHighCardinalityDataHandler

np.random.seed(42)
random.seed(42)
random_state = 42

# xgboost fix
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb



# Global Parameters
XGB_WEIGHT = 0.64155
BASELINE_WEIGHT = 0#0.0056#0.0056
OLS_WEIGHT = 0.0828

XGB1_WEIGHT = 0.9013#0.8#0.80#0.8083#0.9013 XGB Correlation#  # Weight of first in combination of two XGB models

BASELINE_PRED = 0.0115 # Baseline based on mean of training data, per https://www.kaggle.com/bbrandt/using-avg-for-each-landusetypeid-for-prediction
#BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg

xgb_ols_params = {
            'booster': 'dart',
            'eta': 0.02044365,
            'max_depth': 3,
            'subsample': 0.85623748,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'lambda': 0.63761733,
            'alpha': 0.26980652,
            'silent': 1,
            'seed': random_state
        }

xgb_cities_models_comb = {
            'booster': 'dart',
            'eta': 0.02044365,
            'max_depth': 3,
            'subsample': 0.85623748,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'lambda': 0.63761733,
            'alpha': 0.26980652,
            'silent': 1,
            'seed': random_state
        }

xgb_lgb_params = {
            'eta': 0.02531735,
            'max_depth': 3,
            'subsample': 0.73690883,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'lambda': 0.64511885,
            'alpha': 0.36120982,
            'silent': 1,
            'seed': random_state
        }

xgb_comb_params = {
            'eta': 0.03240067,
            'max_depth': 3,
            'subsample': 0.62399971,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'lambda': 0.83544795,
            'alpha': 0.04455517,
            'silent': 1,
            'seed': random_state
        }



xgb_params1 = {
            'eta': 0.015,
            'max_depth': 8,
            'subsample': 0.9,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'lambda': 0.96104357,
            'alpha': 0.31159466,
            'silent': 1,
            'seed': random_state
        }

xgb_params2 = {
            'eta': 0.015,
            'max_depth': 3,
            'subsample': 0.81699297,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'lambda': 0.98900364,
            'alpha': 0,
            'silent': 1,
            'seed': random_state
        }

lgb_params = {
            'max_bin': 10,
            'learning_rate': 0.0021,  # shrinkage_rate
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'l1',
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

class ZillowHomePredictionModels:
    # This section is originally derived from SIDHARTH's script:
    #   https://www.kaggle.com/sidharthkumar/trying-lightgbm
    # which was forked and tuned by Yuqing Xue:
    #   https://www.kaggle.com/yuqingxue/lightgbm-85-97
    # and updated by Andy Harless:
    #   https://www.kaggle.com/aharless/lightgbm-with-outliers-remaining
    def __init__(self):
        self.xgb_comb_params = xgb_comb_params
        self.xgb_ols_params = xgb_ols_params
        self.xgb_lgb_params = xgb_lgb_params
        self.xgb_params1 = xgb_params1
        self.xgb_params2 = xgb_params2

    def generate_lgb_model(self, X_train, y_train):
        drop_cols = [
            'propertyzoningdesc', 'propertycountylandusecode',
            'fireplacecnt', 'fireplaceflag'
        ]
        # X_train = X_train.drop(drop_cols, axis=1)
        d_train = lgb.Dataset(X_train, label=y_train)


        clf = lgb.train(lgb_params, d_train, 430)
        return clf

    def MAE(self, y, ypred):
        # logerror=log(Zestimate)−log(SalePrice)
        return np.sum([abs(y[i] - ypred[i]) for i in range(len(y))]) / len(y)

    def get_date_features(self, df:pd.DataFrame):
        df["transactiondate"] = pd.to_datetime(df["transactiondate"])
        df["transactiondate_year"] = df["transactiondate"].dt.year
        df["transactiondate_month"] = df["transactiondate"].dt.month
        df['transactiondate'] = df['transactiondate'].dt.quarter
        return df

    def get_ols_features(self, df:pd.DataFrame):
        ols_df = df.copy()
        ols_df = self.get_date_features(df)
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
        ols_x_test = self.get_ols_test(properties_df, transaction_date)
        return self.__get_model_prediction__(ols_model, ols_x_test)

    def get_ols_test(self, properties_df, transaction_date):
        ols_x_test = properties_df.copy()
        ols_x_test["transactiondate"] = transaction_date
        ols_x_test = self.get_ols_features(ols_x_test)
        return ols_x_test

    def create_ols_model(self):
        np.random.seed(17)
        random.seed(17)
        #reg = LinearRegression(n_jobs=-1)
        reg = LinearSVR(C=0.9)
        return reg

    def get_ols_train(self, x_train, transaction_date):
        ols_x_train = x_train.copy()
        ols_x_train['transactiondate'] = transaction_date
        ols_x_train = self.get_ols_features(ols_x_train)
        return ols_x_train

    # This section is derived from the1owl's notebook:
    #    https://www.kaggle.com/the1owl/primer-for-the-zillow-pred-approach
    # which Andy Harless updated and made into a script:
    #    https://www.kaggle.com/aharless/updated-script-version-of-the1owl-s-basic-ols
    def generate_ols_model(self, x_train, transaction_date, y_train):
        reg = self.create_ols_model()
        ols_x_train = self.get_ols_train(x_train, transaction_date)
        reg.fit(ols_x_train, y_train)
        return reg

    #https://www.kaggle.com/mubashir44/simple-ensemble-model-stacking
    def get_oof(self, clf, x_train, y_train, x_test):
        # Some useful parameters which will come in handy later on
        ntrain = x_train.shape[0]
        ntest = x_test.shape[0]
        NFOLDS = 3  # set folds for out-of-fold prediction
        kf = KFold(n_splits=NFOLDS, random_state=random_state)

        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf.split(x_train,  y_train)):
            x_tr = x_train.iloc[train_index, :]
            y_tr = y_train[train_index]
            x_te = x_train.iloc[test_index, :]

            clf.fit(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    # https://www.kaggle.com/mubashir44/simple-ensemble-model-stacking
    def get_oof_for_xgboost(self, x_train, y_train, x_test, xgb_params, num_boost_rounds):
        # Some useful parameters which will come in handy later on
        ntrain = x_train.shape[0]
        ntest = x_test.shape[0]
        SEED = 42  # for reproducibility
        NFOLDS = 3  # set folds for out-of-fold prediction
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

    def get_oof_for_lgboost(self, x_train, y_train, x_test, num_boost_rounds):
        # Some useful parameters which will come in handy later on
        ntrain = x_train.shape[0]
        ntest = x_test.shape[0]
        SEED = 42  # for reproducibility
        NFOLDS = 3  # set folds for out-of-fold prediction
        kf = KFold(n_splits=NFOLDS, random_state=random_state)

        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))
        x_train = x_train.values
        x_test = x_test.values

        for i, (train_index, test_index) in enumerate(kf.split(x_train)):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            clf = lgb.train(lgb_params, lgb.Dataset(x_tr, label=y_tr), num_boost_rounds)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)

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

        print("Train XGB Model 1:")
        xgb_train_pred1, xgb_test_pred1 = self.get_oof_for_xgboost(x_train, y_train, x_test, self.xgb_params1, 250)
        # model1 = self.generate_xgb_model(x_train, y_train, xgb_params1, 250)
        # xgb_train_pred1 = self.__get_model_prediction__(model1, xgb.DMatrix(x_train))
        # xgb_test_pred1 = self.__get_model_prediction__(model1, dtest)

        print("Train XGB Model 2:")
        xgb_train_pred2, xgb_test_pred2 = self.get_oof_for_xgboost(x_train, y_train, x_test, self.xgb_params2, 150)
        # model2 = self.generate_xgb_model(x_train, y_train, xgb_params2, 150)
        # xgb_train_pred2 = self.__get_model_prediction__(model2, xgb.DMatrix(x_train))
        # xgb_test_pred2 = self.__get_model_prediction__(model2, dtest)


        new_x_train = pd.DataFrame({"model1": xgb_train_pred1.flatten(),
                                    "model2": xgb_train_pred2.flatten()})
        new_x_test = pd.DataFrame({"model1": xgb_test_pred1.flatten(),
                                   "model2": xgb_test_pred2.flatten()})



        #combined_xgb_pred = XGB1_WEIGHT*xgb_test_pred1.flatten() + (1-XGB1_WEIGHT)*xgb_test_pred2.flatten()
        # new_d_test = xgb.DMatrix(new_x_test)
        print("Train Combined XGB Model:")
        # combined_xgb_model = self.generate_xgb_model(new_x_train, y_train, xgb_comb_params, 150)
        # combined_xgb_train = self.__get_model_prediction__(combined_xgb_model, xgb.DMatrix(new_x_train))
        # combined_xgb_pred = self.__get_model_prediction__(combined_xgb_model, xgb.DMatrix(new_x_test))
        combined_xgb_train, combined_xgb_pred = self.get_oof_for_xgboost(new_x_train, y_train,
                                                                         new_x_test, self.xgb_comb_params, 200)

        #sel = PLSRegression(n_components=1,)
        #pca_x_train = sel.fit_transform(new_x_train, y_train)
        #pca_x_test = sel.transform(new_x_test)

        #reg = GradientBoostingRegressor(learning_rate=0.3, n_estimators=200, subsample=0.8, max_depth=5, alpha=0.3)
        #reg.fit(new_x_train, y_train)
        #combined_xgb_pred = self.__get_model_prediction__(reg, new_x_test)

        return combined_xgb_train, combined_xgb_pred

    def generate_xgb_lgb_combined_predictions(self, x_train, y_train, x_test):
        # lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT) / (1 - OLS_WEIGHT)
        # xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
        # baseline_weight0 = BASELINE_WEIGHT / (1 - OLS_WEIGHT)
        xgb_train, xgb_pred = self.generate_combined_xgb_pred(x_train, y_train, x_test)

        lgb_drop_cols = [
            'propertyzoningdesc', 'propertycountylandusecode',
            #'fireplacecnt', 'fireplaceflag'
        ]
        lgb_x_train = x_train.drop(lgb_drop_cols, axis=1)
        lgb_x_test = x_test.drop(lgb_drop_cols, axis=1)

        print("Train LGB Model:")
        lgb_train, lgb_test = self.get_oof_for_lgboost(lgb_x_train, y_train, lgb_x_test, 430)

        # lgb_model = self.generate_lgb_model(lgb_x_train, y_train)
        # lgb_train = self.__get_model_prediction__(lgb_model, lgb_x_train)
        # lgb_test = self.__get_model_prediction__(lgb_model, lgb_x_test)
        #pred0 = xgb_weight0 * xgb_pred + baseline_weight0 * BASELINE_PRED + lgb_weight * lgb_pred
        new_x_train = pd.DataFrame({"model1": xgb_train.flatten(),
                                    "model2": lgb_train.flatten()})
        new_x_test = pd.DataFrame({"model1": xgb_pred.flatten(),
                                   "model2": lgb_test.flatten()})

        print("Train XGB-LGB Model:")
        lgb_xgb_train_pred, lgb_xgb_test_pred = self.get_oof_for_xgboost(new_x_train, y_train,
                                                                         new_x_test, self.xgb_lgb_params, 180)

        # lgb_xgb_model = self.generate_xgb_model(new_x_train, y_train, xgb_lgb_params, 150)
        # lgb_xgb_train_pred = self.__get_model_prediction__(lgb_xgb_model, xgb.DMatrix(new_x_train))
        # lgb_xgb_test_pred = self.__get_model_prediction__(lgb_xgb_model, xgb.DMatrix(new_x_test))

        return lgb_xgb_train_pred.flatten(), lgb_xgb_test_pred.flatten()

    def generate_all_combined_predictions_with_date(self, merged_df, x_test,
                                                    x_test_index, x_train, y_train, scaler=None):
        dates = ['2016-10-01', '2016-11-01', '2016-12-01', '2017-10-01', '2017-11-01', '2017-12-01']
        columns = ['201610', '201611', '201612', '201710', '201711', '201712']
        output = pd.DataFrame({'ParcelId': x_test_index})
        x_train["transactiondate"] = merged_df["transactiondate"].values
        x_train = self.get_date_features(x_train)
        x_test_with_date = pd.DataFrame()

        for i in range(len(dates)):
            transaction_date = dates[i]
            x_test["transactiondate"] = transaction_date
            x_test = self.get_date_features(x_test)
            x_test_with_date = x_test_with_date.append(x_test)

        x_test = None
        gc.collect()

        train_xgb_lgb, pred_xgb_lgb = self.generate_xgb_lgb_combined_predictions(x_train, y_train, x_test_with_date)
        ols_model = self.generate_ols_model(x_train, merged_df["transactiondate"].values, y_train)

        for i in range(len(dates)):
            transaction_date = dates[i]
            ols_pred = self.get_ols_prediction(ols_model,
                                               x_test_with_date[x_test_with_date["transactiondate"] == transaction_date],
                                               transaction_date)
            pred = OLS_WEIGHT * ols_pred + (1 - OLS_WEIGHT) * pred_xgb_lgb

            if not scaler is None:
                pred = scaler.inverse_transform(pred)
            output[columns[i]] = [float(format(x, '.4f')) for x in pred]

        return output

    def get_top_npercent(self, df, feature, percent=0.8) -> np.array:
        groupedFeatures = df.groupby(feature)[feature].count()
        quantile = groupedFeatures.quantile(percent)
        filteredValues = groupedFeatures[groupedFeatures >= quantile].index
        return filteredValues

    def get_above_n(self, df, feature, n=5000) -> np.array:
        groupedFeatures = df.groupby(feature)[feature].count()
        filteredValues = groupedFeatures[groupedFeatures >= n].index
        return filteredValues

    def generate_all_combined_predictions(self, merged_df, x_test, x_test_index, x_train, y_train, scaler=None):
        dates = ['2016-10-01', '2016-11-01', '2016-12-01', '2017-10-01', '2017-11-01', '2017-12-01']
        columns = ['201610', '201611', '201612', '201710', '201711', '201712']
        output = pd.DataFrame({'ParcelId': x_test_index})

        topcities = self.get_above_n(x_train, 'regionidcity', 2000)
        train_by_city = []
        test_by_city = []
        y_train_by_city = []
        train_indexes = []
        test_indexes = []
        excludedCities = np.unique(x_train[~x_train["regionidcity"].isin(topcities)]["regionidcity"].values)

        for city in topcities:
            city_x_train = x_train[x_train["regionidcity"] == city]
            city_x_test = x_test[x_test["regionidcity"] == city]

            if city_x_train.shape[0] == 0 or city_x_test.shape[0] == 0:
                np.append(excludedCities, city)
                continue

            city_y_train = np.array(y_train[x_train["regionidcity"] == city])
            new_train, new_test = self.generate_xgb_lgb_combined_predictions(city_x_train,
                                                       city_y_train, city_x_test)
            train_by_city.extend(new_train)
            test_by_city.extend(new_test)
            y_train_by_city.extend(city_y_train)
            train_indexes.extend(city_x_train.index.values)
            test_indexes.extend(city_x_test.index.values)

        city_x_train = x_train[x_train["regionidcity"].isin(excludedCities)]
        city_x_test = x_test[x_test["regionidcity"].isin(excludedCities)]
        city_y_train = y_train[x_train["regionidcity"].isin(excludedCities)]
        new_train, new_test = self.generate_xgb_lgb_combined_predictions(city_x_train,city_y_train, city_x_test)

        train_by_city.extend(new_train)
        test_by_city.extend(new_test)
        y_train_by_city.extend(city_y_train)
        train_indexes.extend(city_x_train.index.values)
        test_indexes.extend(city_x_test.index.values)

        print("Train OLS Model:")
        ols_model = self.generate_ols_model(x_train.loc[train_indexes,:], merged_df.loc[train_indexes,:]["transactiondate"].values, y_train_by_city)
        ols_x_train = self.get_ols_train(x_train.loc[train_indexes, :], merged_df.loc[train_indexes, :]["transactiondate"].values)
        ols_train = self.__get_model_prediction__(ols_model, ols_x_train)
        #ols_clf = self.create_ols_model()
        for i in range(len(dates)):
            transaction_date = dates[i]
            ols_x_test = self.get_ols_test(x_test.loc[test_indexes,:], transaction_date)
            #ols_train, ols_pred = self.get_oof(ols_clf, ols_x_train, y_train, ols_x_test)
            ols_pred = self.__get_model_prediction__(ols_model, ols_x_test)
            # pred = OLS_WEIGHT * ols_pred + (1 - OLS_WEIGHT) * pred0
            new_x_train = pd.DataFrame({"model1": train_by_city,
                                        "model2": ols_train.flatten()})
            new_x_test = pd.DataFrame({"model1": test_by_city,
                                       "model2": ols_pred.flatten()})

            print("Train Combined Models For {}:".format(transaction_date))
            xgb_ols_train_pred, xgb_ols_test_pred = self.get_oof_for_xgboost(new_x_train, np.array(y_train_by_city),
                                                                             new_x_test, self.xgb_ols_params, 65)

            # xgb_ols_model = self.generate_xgb_model(new_x_train, y_train, xgb_lgb_params, 150)
            # xgb_ols_train_pred = self.__get_model_prediction__(xgb_ols_model, xgb.DMatrix(new_x_train))
            # xgb_ols_test_pred = self.__get_model_prediction__(xgb_ols_model, xgb.DMatrix(new_x_test))

            xgb_ols_test_pred = xgb_ols_test_pred.flatten()

            if not scaler is None:
                xgb_ols_test_pred = scaler.inverse_transform(xgb_ols_test_pred)

            xgb_ols_test_pred = ((1 - BASELINE_WEIGHT) * xgb_ols_test_pred) + (BASELINE_WEIGHT * BASELINE_PRED)
            output[columns[i]] = [float(format(x, '.4f')) for x in xgb_ols_test_pred]

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