#based on https://www.kaggle.com/danieleewww/xgboost-without-outliers-lb-0-06463/code
#inspired on:
#https://www.kaggle.com/c/zillow-prize-1/discussion/33710
#https://www.kaggle.com/philippsp/exploratory-analysis-zillow
#https://www.kaggle.com/davidfumo/boosted-trees-lb-0-0643707/code

import ZillowDataDecomposition
import ZillowDataRepository
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from datetime import datetime
import gc

#xgboost fix
import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb

# Global Parameters
XGB_WEIGHT = 0.6415
BASELINE_WEIGHT = 0.0056
OLS_WEIGHT = 0.0828

XGB1_WEIGHT = 0.8083  # Weight of first in combination of two XGB models

BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg

#'base_score': y_mean,
xgb_params = {
    'eta': 0.037,
    'max_depth': 5,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,
    'alpha': 0.4,
    'silent': 1
}
xgb_params1 = {
            'eta': 0.0372,
            'max_depth': 5,
            'subsample': 0.8,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'lambda': 0.8,
            'alpha': 0.4,
            'silent': 1
        }

xgb_params2 = {
            'eta': 0.03327,
            'max_depth': 6,
            'subsample': 0.8,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'silent': 1
        }

class ZillowHomePredictionModels:
    def generate_lgb_model(self, X_train, y_train):
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
        df["transactiondate"] = pd.to_datetime(df["transactiondate"])
        df["transactiondate_year"] = df["transactiondate"].dt.year
        df["transactiondate_month"] = df["transactiondate"].dt.month
        df['transactiondate'] = df['transactiondate'].dt.quarter
        df = df.fillna(-1.0)

        #to_remove = [df.columns[c]for c in range(len(df.columns)) if df.dtypes[c] == 'int64']
        to_remove = ['logerror', 'parcelid'] #+ to_remove
        valid_columns = [c for c in df.columns if c not in to_remove]
        return df[valid_columns]

    def get_ols_prediction(self, ols_model, properties_df, transaction_date):
        ols_x_test = properties_df
        ols_x_test["transactiondate"] = transaction_date
        ols_x_test = self.get_ols_features(ols_x_test)
        return self.__get_model_prediction__(ols_model, ols_x_test)

    def generate_ols_model(self, x_train, transaction_date, y_train):
        ols_x_train = x_train
        ols_x_train['transactiondate'] = transaction_date
        ols_x_train = self.get_ols_features(ols_x_train)
        reg = LinearRegression(n_jobs=-1)
        reg.fit(ols_x_train, y_train)
        return reg

    def generate_combined_xgb_pred(self, x_train, y_train, x_test):
        dtest = xgb.DMatrix(x_test)
        xgb_model1 = self.generate_xgb_model(x_train, y_train, xgb_params1, 250)
        xgb_pred1 = self.__get_model_prediction__(xgb_model1, dtest)

        xgb_model2 = self.generate_xgb_model(x_train, y_train, xgb_params2, 250)
        xgb_pred2 = self.__get_model_prediction__(xgb_model2, dtest)

        combined_xgb_pred = XGB1_WEIGHT*xgb_pred1 + (1-XGB1_WEIGHT)*xgb_pred2
        return combined_xgb_pred

    def generate_xgb_lgb_combined_predictions(self, x_train, y_train, x_test):
        lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT) / (1 - OLS_WEIGHT)
        xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
        baseline_weight0 = BASELINE_WEIGHT / (1 - OLS_WEIGHT)
        xgb_pred = self.generate_combined_xgb_pred(x_train, y_train, x_test)

        lgb_pred = self.__get_model_prediction__(self.generate_lgb_model(x_train, y_train), x_test)
        pred0 = xgb_weight0 * xgb_pred + baseline_weight0 * BASELINE_PRED + lgb_weight * lgb_pred
        return pred0

    def generate_all_combined_predictions(self, merged_df, x_test, x_test_index, x_train, y_train):
        dates = ['2016-10-01', '2016-11-01', '2016-12-01', '2017-10-01', '2017-11-01', '2017-12-01']
        columns = ['201610', '201611', '201612', '201710', '201711', '201712']
        output = pd.DataFrame({'ParcelId': x_test_index})
        pred0 = self.generate_xgb_lgb_combined_predictions(x_train, y_train, x_test)
        ols_model = self.generate_ols_model(x_train, merged_df["transactiondate"].values, y_train)
        for i in range(len(dates)):
            transaction_date = dates[i]
            ols_pred = self.get_ols_prediction(ols_model, x_test, transaction_date)
            pred = OLS_WEIGHT * ols_pred + (1 - OLS_WEIGHT) * pred0
            output[columns[i]] = [float(format(x, '.4f')) for x in pred]

        return output

    def __get_model_prediction__(self, model, x_test):
        return model.predict(x_test)

    def generate_xgb_model(self, X_train, y_train, xgb_params, boost_rounds):
        y_mean = y_train.mean()
        dtrain = xgb.DMatrix(X_train, y_train)
        xgb_params['base_score'] = y_mean

        cv_result = xgb.cv(xgb_params,
                           dtrain,
                           nfold=10,
                           num_boost_round=boost_rounds,
                           early_stopping_rounds=5,
                           verbose_eval=10,
                           show_stdv=False
                           )
        num_boost_rounds = len(cv_result)
        print("Number of boost rounds:")
        print(num_boost_rounds)
        # train model
        model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

        # mweight = pd.DataFrame.from_dict(model.get_score(), orient="index")
        # mweight.columns = ["weight"]
        # mgain = pd.DataFrame.from_dict(model.get_score(importance_type='gain'), orient="index")
        # mgain.columns = ["gain"]
        # mcover = pd.DataFrame.from_dict(model.get_score(importance_type='cover'), orient="index")
        # mcover.columns = ["cover"]
        # mscore = mweight.merge(mgain, left_index=True, right_index=True)
        # mscore = mscore.merge(mcover, left_index=True, right_index=True)
        # print("Model Score")
        # print(mscore.sort_values("gain", ascending=False))
        return model


class ZillowHomePrediction():
    def __init__(self):
        self.data_repo = ZillowDataRepository.ZillowDataRepository()
        self.zillow_models = ZillowHomePredictionModels()

    def __get_train_test_data__(self, data):
        x_cols = len(data.columns) - 2
        X = data.iloc[:, :x_cols]
        y = data.logerror.values
        return train_test_split(X, y, test_size=0.9, random_state=42)

    def __get_train_data_for_submission__(self, data):
        x_train = data.drop(['logerror', 'transactiondate'], axis=1)
        y_train = data.logerror.values
        return x_train, y_train

    def generate_xgboost_submission(self):
        data = self.data_repo.get_merged_data()
        y_mean = data.logerror.mean()
        X_train, y_train = self.__get_train_data_for_submission__(data)
        model = self.zillow_models.generate_xgb_model(X_train, y_train, y_mean)

        properties = self.data_repo.get_properties_data()
        dtest = xgb.DMatrix(properties)
        pred = model.predict(dtest)
        y_pred = []
        for i, predict in enumerate(pred):
            y_pred.append(str(round(predict, 4)))
        y_pred = np.array(y_pred)

        output = pd.DataFrame({'ParcelId': properties.index.astype(np.int32),
                               '201610': y_pred, '201611': y_pred, '201612': y_pred,
                               '201710': y_pred, '201711': y_pred, '201712': y_pred})

        cols = output.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        output = output[cols]
        output.to_csv('Submissions\\sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

    def generate_combined_model_submission(self):
        data = self.data_repo.get_merged_data()
        X_train, y_train = self.__get_train_data_for_submission__(data)

        x_test = self.data_repo.get_properties_data()
        output = self.zillow_models.generate_all_combined_predictions(data, x_test, x_test.index.values, X_train, y_train)
        output.to_csv('Submissions\\sub{}.csv.gz'.format(datetime.now().strftime('%Y%m%d_%H%M%S')),
                      index=False, compression='gzip')

    def generate_combined_model_with_decomp_submission(self):
        data = self.data_repo.get_merged_data()
        X_train, y_train = self.__get_train_data_for_submission__(data)
        data_decomp = ZillowDataDecomposition.ZillowDataDecomposition(X_train, y_train, 50)
        new_x_train = pd.DataFrame(data_decomp.get_pca_transformed_data(X_train))

        x_test = self.data_repo.get_properties_data()
        new_x_test = pd.DataFrame(data_decomp.get_pca_transformed_data(x_test))
        output = self.zillow_models.generate_all_combined_predictions(data, new_x_test,
                                                                      x_test.index.values, new_x_train,
                                                                      y_train)
        output.to_csv('Submissions\\sub{}.csv.gz'.format(datetime.now().strftime('%Y%m%d_%H%M%S')),
                      index=False, compression='gzip')

    def test_combined_models(self):
        data = self.data_repo.get_merged_data()
        X_train, X_test, y_train, y_test = self.__get_train_test_data__(data)

        pred = self.zillow_models.generate_xgb_lgb_combined_predictions(X_train,y_train, X_test)
        print("Model score:")
        print(r2_score(y_test, pred))
        print(mean_squared_error(y_test, pred))


    def test_with_xgboost(self):
        data = self.data_repo.get_merged_data()
        X_train, X_test, y_train, y_test = self.__get_train_test_data__(data)
        dtest = xgb.DMatrix(X_test)

        model = self.zillow_models.generate_xgb_model(X_train, y_train, xgb_params2, 500)
        pred = model.predict(dtest)
        print("Model score:")
        print(r2_score(y_test, pred))
        print(mean_squared_error(y_test, pred))


home_pred = ZillowHomePrediction()
#home_pred.test_combined_models()
home_pred.generate_combined_model_with_decomp_submission()