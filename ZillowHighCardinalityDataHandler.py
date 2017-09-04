from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

#From Scirpus: https://www.kaggle.com/scirpus/genetic-programming-lb-0-0643904
HIGH_CARDINALITY = ['airconditioningtypeid',
                   'architecturalstyletypeid',
                   'buildingclasstypeid',
                   'buildingqualitytypeid',
                   'decktypeid',
                   'fips',
                   'hashottuborspa',
                   'heatingorsystemtypeid',
                   'pooltypeid10',
                   'pooltypeid2',
                   'pooltypeid7',
                   'propertycountylandusecode',
                   'propertylandusetypeid',
                    'regionidcity',
                    'regionidcounty',
                    'regionidneighborhood',
                    'regionidzip',
                    'storytypeid',
                    'typeconstructiontypeid',
                    'fireplaceflag',
                    'taxdelinquencyflag']

class ZillowHighCardinalityDataHandler:
    def __init__(self, x_train, y_train, x_test):
        self.train = x_train.copy()
        self.train['y'] = y_train.copy()
        self.test = x_test.copy()
        self.test['y'] = np.nan

    @staticmethod
    def __project_on_median__(data1, data2, columnName):
        grpOutcomes = data1.groupby(list([columnName]))['y'].median().reset_index()
        grpCount = data1.groupby(list([columnName]))['y'].count().reset_index()
        grpOutcomes['cnt'] = grpCount.y
        grpOutcomes.drop('cnt', inplace=True, axis=1)
        #outcomes = data2['y'].values
        x = pd.merge(data2[[columnName, 'y']], grpOutcomes,
                     suffixes=('x_', ''),
                     how='left',
                     on=list([columnName]),
                     left_index=True)['y']

        return x.values

    def high_cardinality_to_mean(self):
        blindloodata = None
        folds = 20
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        for i, (train_index, test_index) in enumerate(kf.split(range(self.train.shape[0]))):
            print('Fold:', i)

            blindtrain = self.train.iloc[test_index].copy()
            vistrain = self.train.iloc[train_index].copy()

            for c in HIGH_CARDINALITY:
                blindtrain.insert(1, 'loo' + c, self.__project_on_median__(vistrain,
                                                                blindtrain, c))
            if (blindloodata is None):
                blindloodata = blindtrain.copy()
            else:
                blindloodata = pd.concat([blindloodata, blindtrain])

        test = self.test

        for c in HIGH_CARDINALITY:
            test.insert(1, 'loo' + c, self.__project_on_median__(self.train, test, c))

        test.drop(HIGH_CARDINALITY, inplace=True, axis=1)

        train = blindloodata
        train.drop(HIGH_CARDINALITY, inplace=True, axis=1)
        return train, test