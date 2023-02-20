"""TODO"""
import os
import sys
from pathlib import Path

import pandas as pd
import rampwf as rw
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder

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
    """TODO"""
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
    return cv.split(X, y)


def _get_data_utils(path, split="train"):
    """TODO"""

    assert split in ["train", "test"], "split must be either 'train' or 'test'"

    dataframe_path = Path(path, "data", "data.csv")

    data = pd.read_csv(dataframe_path, sep=";")
    categorical_features = ["vegetation_class"]

    data_preprocessed = data.drop(columns=["Date"])
    data_preprocessed = data_preprocessed.dropna()

    X = data_preprocessed.drop(columns=[_target_column_name])
    # Preprocessing
    preprocessor = ColumnTransformer(
        [("OHE", OneHotEncoder(), categorical_features)], remainder="passthrough"
    )
    X = preprocessor.fit_transform(X)

    y = data_preprocessed[_target_column_name]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, stratify=y
    )

    if split == "train":
        return X_train, y_train.values

    return X_test, y_test.values


def get_train_data(path):
    """TODO"""
    return _get_data_utils(path, "train")


def get_test_data(path):
    """TODO"""
    return _get_data_utils(path, "test")
