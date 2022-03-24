import os
import pandas as pd
import numpy as np
import rampwf as rw
from data_cleaning import fix_categorical_features_dtype
from itertools import combinations_with_replacement
from sklearn.model_selection import ShuffleSplit
from rampwf.score_types.base import BaseScoreType
from sklearn.metrics import balanced_accuracy_score

split_factor = 0.2

problem_title = 'Covid Vaccince Prediction'

_target_names = ['Vaccine', 'Business']

_prediction_label_ = combinations_with_replacement(
                                ['Completely disagree', 
                                'Somewhat disagree', 
                                'No opinion', 
                                'Completely agree', 
                                'Somewhat agree'],2)

Predictions = rw.prediction_types.make_regression(label_names=_target_names)
workflow = rw.workflows.Regressor()


class BAS(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="BAS", precision=4):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        BAS_1 = np.mean(y_true[0] == y_pred[0])
        BAS_2 = np.mean(y_true[1] == y_pred[1])
        return (BAS_1+BAS_2)/2



score_types = [
    BAS(name="BAS"),
]

def get_cv(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=57)
    return cv.split(X, y)

def _get_data(path, split='train'):
    data_path_1 = os.path.join(path, "data", "data_1.csv")
    data_path_2 = os.path.join(path, "data", "data_2.csv")

    data_1 = pd.read_csv(data_path_1, encoding='iso-8859-1', index_col=0, na_values=' ')  # XXX rmv index_col=0 if ever needed
    data_1 = data_1.reset_index(drop=True)  # if index_col=0 in read_csv above
    dic_idx_country = dict(enumerate(['Brazil', 'Canada', 'China', 'Ecuador', 
                                  'France', 'Germany', 'India', 'Italy',
                                  'Mexico', 'Nigeria', 'Poland', 'Russia',
                                  'South_Africa', 'Singapore', 'South_Korea',
                                  'Spain', 'Sweden', 'UK', 'US'], start=1))
    data_1["Country"].replace(dic_idx_country, inplace=True)


    
    data_2 = pd.read_csv(data_path_2, encoding='iso-8859-1', na_values=' ')
    data_2=data_2.loc[(data_2['Day'] == '2020-09-01') & (data_2['Entity'].isin(['Brazil', 'Canada', 'China', 'Ecuador', 'France', 'Germany', 'India', 'Italy',
                                  'Mexico', 'Nigeria', 'Poland', 'Russia',
                                  'South Africa', 'Singapore', 'South Korea',
                                  'Spain', 'Sweden', 'United Kingdom','United States']))].copy()
    data_2 = data_2.drop(columns=['Code', 'Day'])
    data_2['population']=[212559417,37742154,1439323776,17643054,65273511,83783942,1380004385,
                  60461826,128932753,206139589,37846611,145934462,59308690,5850342,51269185,
                  46754778,10099265,67886011,331002651]
    data_2 = data_2['Entity'].replace(['United Kingdom','United States'],['UK', 'US'])
    data_merged=pd.merge(data_1, data_2, left_on='Country', right_on='Entity').drop(columns='Entity')

    if split == 'train':
        idx = np.random.permutation(data_merged.shape[0])
        train_split = int(data_merged.shape[0]*(1-split_factor))
        idx = idx[:train_split]
        data = data_merged.iloc[idx]

    if split == 'test':
        idx = np.random.permutation(data_merged.shape[0])
        train_split = int(data_merged.shape[0]*(1-split_factor))
        idx = idx[train_split:]
        data = data_merged.iloc[idx]


    X = data.drop(['Vaccine','Business2'], axis=1)
    Y = data[['Vaccine','Business2']]
    return X, Y

def get_train_data(path):
    return _get_data(path, 'train')

def get_test_data(path):
    return _get_data(path, 'test')




