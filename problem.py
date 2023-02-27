import os
import sys
from pathlib import Path

import pandas as pd
import rampwf as rw
from rampwf.score_types import ClassifierBaseScoreType
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder

problem_title = "Fire prediction"

_target_column_name = "fire"
_prediction_label_names = [0, 1]

Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names
    )
workflow = rw.workflows.Estimator()

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


class FBetaScore(ClassifierBaseScoreType):
    """
    Wrapper around scikit-learn's F-β score.

    Default is β = 2, weighting sensitivity higher than precision.
    """

    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0
    worse = 0.0

    def __init__(self, name=None, precision=2, beta=2):
        self.precision = precision
        self.beta = beta
        if name is None:
            self.name = f"F_{beta}_score"
        else:
            self.name = name

    def __call__(self, y_true, y_pred):
        score = fbeta_score(y_true, y_pred, beta=self.beta, average='binary')
        return score


class FBetaMacroScore(ClassifierBaseScoreType):
    """
    Wrapper around scikit-learn's F-β score.

    Default is β = 2, weighting sensitivity higher than precision.
    """

    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0
    worse = 0.0

    def __init__(self, name=None, precision=2, beta=2):
        self.precision = precision
        self.beta = beta
        if name is None:
            self.name = f"F_{beta}_score"
        else:
            self.name = name

    def __call__(self, y_true, y_pred):
        score = fbeta_score(y_true, y_pred, beta=self.beta, average='macro')
        return score


score_types = [rw.score_types.BalancedAccuracy(name="BAS"),
               FBetaMacroScore(name="Macro_F_beta=2"),
               FBetaScore(name="F_beta=2_1"),
               FBetaScore(name="F1_class_1", beta=1)]


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

    el = LabelEncoder()
    X.loc[:, "vegetation_class"] = el.fit_transform(
                                X.loc[:, "vegetation_class"]
                                )

    y = data_preprocessed[_target_column_name]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, stratify=y
    )

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    if split == "train":
        return X_train, y_train.values

    return X_test, y_test.values


def get_train_data(path):
    return _get_data_utils(path, "train")


def get_test_data(path):
    return _get_data_utils(path, "test")
