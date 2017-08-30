import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import ZillowEmptyDataHandler

#based on https://www.kaggle.com/danieleewww/xgboost-without-outliers-lb-0-06463/code
#inspired on:
#https://www.kaggle.com/c/zillow-prize-1/discussion/33710
#https://www.kaggle.com/philippsp/exploratory-analysis-zillow

class ZillowDataRepository:
    def __init__(self):
        self.dir = "Data\\"
        self.train_data_file = "train_2016_v2.csv"
        self.properties_file = "properties_2016.csv"
        self.__train_data__ = None
        self.__properties_data__ = None
        self.__merged_data__ = None

    def reduce_mem_usage(self, props):
        start_mem_usg = props.memory_usage().sum() / 1024 ** 2
        print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
        NAlist = []  # Keeps track of columns that have missing values filled in.
        for col in props.columns:
            if props[col].dtype != object:  # Exclude strings
                # make variables for Int, max and min
                IsInt = False
                mx = props[col].max()
                mn = props[col].min()

                # Integer does not support NA, therefore, NA needs to be filled
                if not np.isfinite(props[col]).all():
                    NAlist.append(col)
                    props[col].fillna(mn - 1, inplace=True)

                    # test if column can be converted to an integer
                asint = props[col].fillna(0).astype(np.int64)
                result = (props[col] - asint)
                result = result.sum()
                if result > -0.01 and result < 0.01:
                    IsInt = True

                # Make Integer/unsigned Integer datatypes
                if IsInt:
                    if mn >= 0:
                        if mx < 255:
                            props[col] = props[col].astype(np.uint8)
                        elif mx < 65535:
                            props[col] = props[col].astype(np.uint16)
                        elif mx < 4294967295:
                            props[col] = props[col].astype(np.uint32)
                        else:
                            props[col] = props[col].astype(np.uint64)
                    else:
                        if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                            props[col] = props[col].astype(np.int8)
                        elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                            props[col] = props[col].astype(np.int16)
                        elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                            props[col] = props[col].astype(np.int32)
                        elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                            props[col] = props[col].astype(np.int64)

                            # Make float datatypes 32 bit
                else:
                    props[col] = props[col].astype(np.float32)

    def get_train_data(self):
        if self.__train_data__ is None:
            self.__train_data__ = self.__read_train_data__()
        return self.__train_data__

    def get_properties_data(self):
        if self.__properties_data__ is None:
            self.__properties_data__ = self.__read_properties_data__()
        return self.__properties_data__

    def get_merged_data(self):
        if self.__merged_data__ is None:
            self.__merged_data__ = self.__read_and_merge_files__()
        return self.__merged_data__

    def y_and_n_to_bool_converter(self, value: str) -> bool:
        return value == "Y"

    def __read_train_data__(self) -> pd.DataFrame:
        train_data = pd.read_csv(self.dir + self.train_data_file, index_col=0)
        #train_data = self.__treat_train_data__(train_data)
        return train_data

    def remove_outliers(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        data_frame = data_frame[data_frame.logerror > -0.4]
        data_frame = data_frame[data_frame.logerror < 0.42]
        return data_frame

    def __remove_data__(self, data):
        low_gain_and_weight_columns = [
            'fireplacecnt',
            'fireplaceflag'
        ]

        data = data.drop(low_gain_and_weight_columns, axis=1)
        return data

    def __transform_to_dummy__(self, data, field, prefix):
        treated_data = data
        treated_data = treated_data.merge(pd.get_dummies(treated_data[field], prefix=prefix),
                                          left_index=True, right_index=True, how='left')
        treated_data.drop(field, axis=1, inplace=True)
        return treated_data

    def __treat_properties_data__(self, data: pd.DataFrame):
        treated_data = data
        treated_data = self.__transform_to_dummy__(treated_data, 'propertylandusetypeid', 'landuse')
        # treated_data = self.__transform_to_dummy__(treated_data, 'architecturalstyletypeid', 'archstyle')
        # treated_data = self.__transform_to_dummy__(treated_data, 'buildingqualitytypeid', 'qualid')
        # treated_data = self.__transform_to_dummy__(treated_data, 'buildingclasstypeid', 'classid')
        # treated_data = self.__transform_to_dummy__(treated_data, 'heatingorsystemtypeid', 'heatid')
        # treated_data = self.__transform_to_dummy__(treated_data, 'storytypeid', 'storyid')
        # treated_data = self.__transform_to_dummy__(treated_data, 'typeconstructiontypeid', 'typeid')
        #treated_data = self.__transform_to_dummy__(treated_data, 'decktypeid', 'deckid')
        #treated_data = self.__transform_to_dummy__(treated_data, 'airconditioningtypeid', 'aircond')
        #treated_data = self.__transform_to_dummy__(treated_data, 'hashottuborspa', 'tuborspa')
        treated_data['taxdelinquencyflag'] = treated_data['taxdelinquencyflag'].fillna(False)

        for col in treated_data.columns:
            if treated_data[col].dtype == 'object':
                treated_data[col] = treated_data[col].fillna(-1)
                lbl = LabelEncoder()
                lbl.fit(list(treated_data[col].values))
                treated_data[col] = lbl.transform(list(treated_data[col].values))
                print('{}.'.format(col))
                print('unique count: {}'.format(len(np.unique(treated_data[col].values))))
            if treated_data[col].dtype == np.int64:
                treated_data[col] = treated_data[col].astype(np.int32)
            if treated_data[col].dtype == np.float64:
                treated_data[col] = treated_data[col].astype(np.float32)
                treated_data[col].fillna(treated_data[col].median(),inplace = True)
        return treated_data

    def __read_properties_data__(self) -> pd.DataFrame:
        prop_data = pd.read_csv(self.dir + self.properties_file, index_col=0,
                                dtype={22: np.bool, 32: np.object, 34: np.object, 49: np.bool},
                                converters={55: self.y_and_n_to_bool_converter})

        prop_data = self.__remove_data__(prop_data)
        prop_data = self.__treat_properties_data__(prop_data)
        #prop_data = self.__generate_new_features__(prop_data)
        return prop_data

    def __read_and_merge_files__(self) -> pd.DataFrame:
        merged_data = pd.merge(self.get_properties_data(), self.get_train_data(), left_index=True, right_index=True)
        merged_data = self.remove_outliers(merged_data)
        return merged_data

    def __generate_new_features__(self, prop_data):
        #prop_data['taxpersquarefeet'] = prop_data.taxamount / prop_data.finishedsquarefeet12
        #prop_data['constructedareaproportion'] = prop_data.finishedsquarefeet12 / prop_data.lotsizesquarefeet
        #prop_data['numberofparcels'] = prop_data.taxvaluedollarcnt / prop_data.taxamount
        prop_data['taxwhenoverdue'] = prop_data.taxdelinquencyflag * prop_data.taxamount

        # Not worth removing
        # prop_data['pooltype7reason'] = prop_data.pooltypeid7 / prop_data.poolcnt
        #prop_data['yardproportion'] = prop_data.yardbuildingsqft17 / prop_data.finishedsquarefeet12
        #prop_data['garageproportion'] = prop_data.garagetotalsqft / prop_data.finishedsquarefeet12
        #prop_data['threequarterbathproportion'] = prop_data.threequarterbathnbr / prop_data.bathroomcnt

        #Drop the data that is now redundant
        prop_data = prop_data.drop(["taxdelinquencyflag"], axis=1)

        return prop_data