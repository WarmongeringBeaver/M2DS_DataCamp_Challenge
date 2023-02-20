"""TODO"""

from sklearn.ensemble import RandomForestClassifier


def get_estimator():
    """TODO"""
    # TODO: add pipeline for data cleaning like in https://github.com/ramp-kits/titanic/blob/master/submissions/random_forest_20_5/estimator.py
    model = RandomForestClassifier(
        n_jobs=-1, random_state=42, class_weight="balanced_subsample"
    )

    return model
