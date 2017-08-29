import pandas as pd
import numpy as np
import ZillowDataDecomposition
#xgboost fix
import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb

class ZillowEmptyDataHandler:
    def __init__(self, data):
        self.data = data

    def __get_train_data__(self, field):
        data_with_values = self.data[pd.notnull(self.data[field])]
        x = data_with_values.drop([field], axis=1)
        y = data_with_values[field].values
        return x, y

    def predict_field(self, field):
        x, y = self.__get_train_data__(field)

        y_mean = y.mean()
        dtrain = xgb.DMatrix(x, y)

        xgb_params = {
            'eta': 0.033,
            'max_depth': 3,
            'subsample': 0.60,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'base_score': y_mean,
            'silent': 1
        }
        num_boost_rounds = self.__run_crossval__(dtrain, xgb_params)

        model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

        mscore = self.__get_model_score__(model)
        print("Model Score")
        print(mscore.sort_values("gain", ascending=False))

        data = self.data[pd.isnull(self.data[field])]
        data = data.drop([field], axis=1)
        dtest = xgb.DMatrix(data)
        pred = model.predict(dtest)
        complete_data = np.concatenate((pred, y))
        return complete_data

    def __run_crossval__(self, dtrain, xgb_params):
        cv_result = xgb.cv(xgb_params,
                           dtrain,
                           nfold=3,
                           num_boost_round=10,
                           early_stopping_rounds=5,
                           verbose_eval=10,
                           show_stdv=False
                           )
        num_boost_rounds = len(cv_result)
        print("Number of boost rounds:")
        print(num_boost_rounds)
        return num_boost_rounds

    def __get_model_score__(self, model):
        mweight = pd.DataFrame.from_dict(model.get_score(), orient="index")
        mweight.columns = ["weight"]
        mgain = pd.DataFrame.from_dict(model.get_score(importance_type='gain'), orient="index")
        mgain.columns = ["gain"]
        mcover = pd.DataFrame.from_dict(model.get_score(importance_type='cover'), orient="index")
        mcover.columns = ["cover"]
        mscore = mweight.merge(mgain, left_index=True, right_index=True)
        mscore = mscore.merge(mcover, left_index=True, right_index=True)
        return mscore
