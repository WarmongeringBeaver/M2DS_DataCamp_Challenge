from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import numpy as np 


class Regressor(BaseEstimator):
    def __init__(self):
        self.label_enc_bus = LabelEncoder()
        self.label_enc_vac = LabelEncoder()
        self.n_estimator   = 100
        self.objective     = 'multi:softmax'
        self.use_label     = False

        self.numeric_features=['trusthealth', 'sickwithCOVID', 'Age', 
        'Language', 'mortalityperm', 'trustngov', 'poptrusthealth', 
        'Region', 'stringency_index', 'case_rate', 'death_rate']

        self.categorical_features = ['Country', 'Universal_edu', 
        'Age_group', 'Gender','within_country', 'world_wide' ]


        self.preprocessor_ = ColumnTransformer([
            ("hot_encoder", OneHotEncoder(), self.categorical_features),
            ("scaler", MinMaxScaler(), self.numeric_features),
        ])

        self.model_1   = Pipeline([
            ("preprocessor", self.preprocessor_),
            ("classifier", xgb.XGBClassifier(n_estimator = self.n_estimator,
            objective= self.objective,
            use_label_encoder=self.use_label))
        ])

        self.model_2   = Pipeline([
            ("preprocessor", self.preprocessor_),
            ("classifier", xgb.XGBClassifier(n_estimator = self.n_estimator,
            objective= self.objective,
            use_label_encoder=self.use_label))
        ])
        
    def fit(self, X, Y):
        self.label_enc_vac.fit(Y['Vaccine'])
        self.label_enc_bus.fit(Y['Business2'])

        self.model_1.fit(X, self.label_enc_vac.transform(Y['Vaccine']))
        self.model_2.fit(X, self.label_enc_bus.transform(Y['Business2']))

    def predict(self, X):
        res_vac = self.model_1.predict(X)
        res_bus = self.model_2.predict(X)
        res = np.concatenate((res_vac, res_bus)).reshape(2, res_vac.shape[0])
        return res
