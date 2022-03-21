import os
import pandas as pd
import rampwf as rw
from itertools import combinations_with_replacement
from sklearn.model_selection import ShuffleSplit
from data_cleaning import data_cleaning 

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

score_types = rw.score_types.Accuracy(name= 'acc')

def get_cv(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=57)
    return cv.split(X, y)

def _get_data(path, split):
    split_path = os.path.join(path, "data", split, "*.csv")
    data = pd.read_csv(split_path, encoding='iso-8859-1')
    data = data_cleaning(data)

    X = data.drop(['Vaccine','Business2'], axis=1)
    Y = data[['Vaccine','Business2']].values
    return X, Y

def get_train_data(path):
    return _get_data(path, 'test')

def get_test_data(path):
    return _get_data(path, 'test')

def clean_data(df):
    return data_cleaning(df)




