#based on https://www.kaggle.com/danieleewww/xgboost-without-outliers-lb-0-06463/code
#inspired on:
#https://www.kaggle.com/c/zillow-prize-1/discussion/33710
#https://www.kaggle.com/philippsp/exploratory-analysis-zillow
#https://www.kaggle.com/davidfumo/boosted-trees-lb-0-0643707/code
#https://www.kaggle.com/aharless/xgboost-lightgbm-and-ols/output
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from skopt import forest_minimize, dump
from pyswarm import pso

import ZillowDataDecomposition
import ZillowDataRepository
from ZillowHomePredictionModels import ZillowHomePredictionModels

# xgboost fix
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb

# xgb_params = {
#     'learning_rate': 0.037,
#     'max_depth': 5,
#     'subsample': 0.80,
#     'booster': 'gblinear',
#     'reg_lambda': 0.8,
#     'reg_alpha': 0.4,
#     'silent': True
# }
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

    def generate_combined_model_submission_with_date(self):
        data = self.data_repo.get_merged_data()
        X_train, y_train = self.__get_train_data_for_submission__(data)

        x_test = self.data_repo.get_properties_data()
        output = self.zillow_models.generate_all_combined_predictions_with_date(data, x_test,
                                                                      x_test.index.values, X_train, y_train,
                                                                      self.data_repo.train_data_scaler)
        output.to_csv('Submissions\\sub{}.csv.gz'.format(datetime.now().strftime('%Y%m%d_%H%M%S')),
                      index=False, compression='gzip')

    def generate_combined_model_submission(self):
        data = self.data_repo.get_merged_data()
        X_train, y_train = self.__get_train_data_for_submission__(data)

        x_test = self.data_repo.get_properties_data()
        output = self.zillow_models.generate_all_combined_predictions(data, x_test,
                                                                      x_test.index.values, X_train, y_train,
                                                                      self.data_repo.train_data_scaler)
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
        print(mean_absolute_error(y_test, pred))

    def cv_test_combined_models(self):
        data = self.data_repo.get_merged_data()
        x_train, y_train = self.__get_train_data_for_submission__(data)

        NFOLDS=5
        random_state = 42
        kf = KFold(n_splits=NFOLDS, random_state=random_state)

        columns = ['201610', '201611', '201612', '201710', '201711', '201712']
        maes = []

        for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
            df = data.iloc[train_index, :]
            x_tr = x_train.iloc[train_index, :]
            y_tr = y_train[train_index]
            x_te = x_train.iloc[test_index, :]
            y_te = y_train[test_index]

            pred = self.zillow_models.generate_all_combined_predictions(df, x_te, x_te.index.values, x_tr,
                                                                          y_tr)
            for col in columns:
                mae = mean_absolute_error(y_te, pred[col])
                maes.append(mae)

        avg_mae = np.average(maes)
        print('CV AVG MAE:{}'.format(avg_mae))
        return avg_mae

    def cv_test_combined_models_with_xgb_comb_params(self, params):
        eta, max_depth, subsample, l2lambda, l1alpha = params

        xgb_comb_params = {
            'eta': eta,
            'max_depth': max_depth,
            'subsample': subsample,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'lambda': l2lambda,
            'alpha': l1alpha,
            'silent': 1,
            'seed': 42
        }

        self.zillow_models.xgb_comb_params = xgb_comb_params

        return self.cv_test_combined_models()

    def cv_test_xgb_params(self, params):
        xgb1_eta, xgb1_max_depth, xgb1_subsample, xgb1_lambda, xgb1_alpha, \
        xgb2_eta, xgb2_max_depth, xgb2_subsample, xgb2_lambda, xgb2_alpha = params

        self.zillow_models.xgb_params1["eta"] = xgb1_eta
        self.zillow_models.xgb_params1["max_depth"] = int(xgb1_max_depth)
        self.zillow_models.xgb_params1["subsample"] = xgb1_subsample
        self.zillow_models.xgb_params1["lambda"] = xgb1_lambda
        self.zillow_models.xgb_params1["alpha"] = xgb1_alpha

        self.zillow_models.xgb_params2["eta"] = xgb2_eta
        self.zillow_models.xgb_params2["max_depth"] = int(xgb2_max_depth)
        self.zillow_models.xgb_params2["subsample"] = xgb2_subsample
        self.zillow_models.xgb_params2["lambda"] = xgb2_lambda
        self.zillow_models.xgb_params2["alpha"] = xgb2_alpha

        return self.cv_test_combined_models()

    def cv_test_comb_xgb_params(self, params):
        ols_eta, ols_max_depth, ols_subsample, ols_lambda, ols_alpha,\
            xgb_comb_eta, xgb_comb_max_depth, xgb_comb_subsample, xgb_comb_lambda, xgb_comb_alpha, \
            xgb_lgb_eta, xgb_lgb_max_depth, xgb_lgb_subsample, xgb_lgb_lambda, xgb_lgb_alpha = params
            # xgb1_eta, xgb1_max_depth, xgb1_subsample, xgb1_lambda, xgb1_alpha, \
            # xgb2_eta, xgb2_max_depth, xgb2_subsample, xgb2_lambda, xgb2_alpha\


        self.zillow_models.xgb_ols_params["eta"] = ols_eta
        self.zillow_models.xgb_ols_params["max_depth"] = int(ols_max_depth)
        self.zillow_models.xgb_ols_params["subsample"] = ols_subsample
        self.zillow_models.xgb_ols_params["lambda"] = ols_lambda
        self.zillow_models.xgb_ols_params["alpha"] = ols_alpha

        self.zillow_models.xgb_comb_params["eta"] = xgb_comb_eta
        self.zillow_models.xgb_comb_params["max_depth"] = int(xgb_comb_max_depth)
        self.zillow_models.xgb_comb_params["subsample"] = xgb_comb_subsample
        self.zillow_models.xgb_comb_params["lambda"] = xgb_comb_lambda
        self.zillow_models.xgb_comb_params["alpha"] = xgb_comb_alpha

        self.zillow_models.xgb_lgb_params["eta"] = xgb_lgb_eta
        self.zillow_models.xgb_lgb_params["max_depth"] = int(xgb_lgb_max_depth)
        self.zillow_models.xgb_lgb_params["subsample"] = xgb_lgb_subsample
        self.zillow_models.xgb_lgb_params["lambda"] = xgb_lgb_lambda
        self.zillow_models.xgb_lgb_params["alpha"] = xgb_lgb_alpha

        # self.zillow_models.xgb_params1["eta"] = xgb1_eta
        # self.zillow_models.xgb_params1["max_depth"] = xgb1_max_depth
        # self.zillow_models.xgb_params1["subsample"] = xgb1_subsample
        # self.zillow_models.xgb_params1["lambda"] = xgb1_lambda
        # self.zillow_models.xgb_params1["alpha"] = xgb1_alpha
        #
        # self.zillow_models.xgb_params2["eta"] = xgb2_eta
        # self.zillow_models.xgb_params2["max_depth"] = xgb2_max_depth
        # self.zillow_models.xgb_params2["subsample"] = xgb2_subsample
        # self.zillow_models.xgb_params2["lambda"] = xgb2_lambda
        # self.zillow_models.xgb_params2["alpha"] = xgb2_alpha

        return self.cv_test_combined_models()
    
    def optimize_with_swarm(self):
        lb = np.array([0.015, 3, 0.6, 0.5, 0, 0.015, 3, 0.6, 0.5, 0, 0.015, 3, 0.6, 0.5, 0.0])
        ub = np.array([0.035, 8, 0.9, 1, 0.5, 0.035, 8, 0.9, 1, 0.5, 0.035, 8, 0.9, 1.0, 0.5])
        xopt, fopt = pso(self.cv_test_comb_xgb_params, lb, ub, swarmsize=20, maxiter=7)
        #dump(res, "xgb_all_opt.gz")
        print(xopt)
        print(fopt)
        print("")

    def optimize_xgb_with_swarm(self):
        lb = np.array([0.015, 3, 0.6, 0.5, 0, 0.015, 3, 0.6, 0.5, 0])
        ub = np.array([0.035, 8, 0.9, 1, 0.5, 0.035, 8, 0.9, 1, 0.5])
        xopt, fopt = pso(self.cv_test_xgb_params, lb, ub, swarmsize=10, maxiter=5)
        #dump(res, "xgb_all_opt.gz")
        print(xopt)
        print(fopt)
        print("")

    def optimize_all_xgb_params(self):
        res = forest_minimize(self.cv_test_comb_xgb_params,
                          [(0.015, 0.035), (3, 8), (0.6, 0.9), (0.5, 1), (0, 0.5),
                           (0.015, 0.035), (3, 8), (0.6, 0.9), (0.5, 1), (0, 0.5),
                           (0.015, 0.035), (3, 8), (0.6, 0.9), (0.5, 1), (0, 0.5),
                           #(0.015, 0.035), (3, 8), (0.6, 0.9), (0.5, 1), (0, 0.5),
                           #(0.015, 0.035), (3, 8), (0.6, 0.9), (0.5, 1), (0, 0.5)
                           ],
                          x0=[0.0245, 5, 0.8, 1, 0,
                              0.03, 5, 0.8, 0.8, 0.4,
                              0.03, 5, 0.8, 1, 0,
                              0.037, 5, 0.8, 0.8, 0.4,
                              0.033, 6, 0.8, 1, 0], y0=0.05323593162735551,
                          n_calls=75)
        dump(res, "xgb_all_opt.gz")
        print(res.x)
        print(res.fun)
        print("")

    def optimize_xgb_comb_params(self):
        res = forest_minimize(self.cv_test_combined_models_with_xgb_comb_params,
                          [(0.015, 0.035), (3,6), (0.6, 0.9), (0.5, 1), (0, 0.5)],
                          x0=[0.03, 5, 0.8, 0.8, 0.4], y0=0.053262479044449806,
                          n_calls=10)
        dump(res, "xgb_com_opt.gz")
        print(res.x)
        print(res.fun)
        print("")

    def test_with_xgboost(self, xgb_params=xgb_params, boost_rounds=150):
        data = self.data_repo.get_merged_data()
        X_train, X_test, y_train, y_test = self.__get_train_test_data__(data)
        dtest = xgb.DMatrix(X_test)

        model = self.zillow_models.generate_xgb_model(X_train, y_train, xgb_params, boost_rounds)
        pred = model.predict(dtest)
        print("Model score:")
        print(r2_score(y_test, pred))
        print(mean_squared_error(y_test, pred))
        print(mean_absolute_error(y_test, pred))

start = time.time()
home_pred = ZillowHomePrediction()
#home_pred.cv_test_combined_models()
#home_pred.test_with_xgboost(xgb_params1, 250)
#home_pred.test_with_xgboost(xgb_params2, 150)
#home_pred.generate_combined_model_with_decomp_submission()
#home_pred.optimize_xgb_comb_params()
#home_pred.optimize_all_xgb_params()
#home_pred.optimize_with_swarm()
#home_pred.optimize_xgb_with_swarm()
home_pred.generate_combined_model_submission()
end = time.time()
print("Spent time")
print(end - start)