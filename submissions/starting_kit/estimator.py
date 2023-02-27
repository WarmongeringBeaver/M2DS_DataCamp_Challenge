from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder


def get_estimator():
    categorical_cols = ["vegetation_class"]
    categorical_pipeline = make_pipeline(OneHotEncoder(handle_unknown="ignore"))
    numerical_cols = [
        "mean_temp",
        "urban",
        "max_v_wind",
        "water",
        "forest_cover",
        "pop_dens",
        "max_temp",
        "sum_prec",
        "mean_soil",
        "mean_rel_hum",
        "wetland",
        "mean_wind_angle",
    ]

    # No need to scale numerical variables for this classifier
    numerical_pipeline = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=-1)
    )

    preprocessor = make_column_transformer(
        (categorical_pipeline, categorical_cols),
        (numerical_pipeline, numerical_cols),
    )

    pipeline = Pipeline(
        [
            ("transformer", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced_subsample",
                    criterion="gini",
                    max_depth=6,
                    min_samples_leaf=1,
                    min_samples_split=3,
                    n_estimators=50,
                ),
            ),
        ]
    )

    return pipeline
