import os
import sys
from pathlib import Path

import pandas as pd
import rampwf as rw
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder
from prepare_data import prepare_data

problem_title = "Fire prediction"

_target_column_name = "fire"
_prediction_label_names = [0, 1]

Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)
workflow = rw.workflows.Estimator()

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

score_types = [
    rw.score_types.BalancedAccuracy(name="bacc"),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
    return cv.split(X, y)


def _get_data_utils(path, split="train"):

    assert split in ["train", "test"], "split must be either 'train' or 'test'"

    dataframe_path = Path(path, "data", "data.csv")

    data = pd.read_csv(dataframe_path, sep=";")

    data_preprocessed = data.drop(columns=["Date"])
    data_preprocessed = data_preprocessed.dropna()

    X = data_preprocessed.drop(columns=[_target_column_name])
    
    el=LabelEncoder()
    X.loc[:,"vegetation_class"]=el.fit_transform(X.loc[:,"vegetation_class"])

    y = data_preprocessed[_target_column_name]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, stratify=y
    )
    
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    if split == "train":
        return X_train, y_train.values

    return X_test, y_test.values

def _get_data(path):
    dataframe_path = Path(path, "data", "data.csv")

    X = pd.read_csv(dataframe_path, sep=";")
    return prepare_data(X)


def get_train_data(path):
    return _get_data_utils(path, "train")


def get_test_data(path):
    return _get_data_utils(path, "test")
