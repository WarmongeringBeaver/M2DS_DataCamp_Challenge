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

        self.numeric_features=[1, 2, 3, 9, 12, 13, 14, 15, 16, 18, 19]

        self.categorical_features = [0, 5, 4, 6, 7, 8]


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
        self.label_enc_vac.fit(Y[:,0])
        self.label_enc_bus.fit(Y[:,1])

        self.model_1.fit(X, self.label_enc_vac.transform(Y[:,0]))
        self.model_2.fit(X, self.label_enc_bus.transform(Y[:,1]))

    def predict(self, X):
        res_vac = self.label_enc_vac.inverse_transform(self.model_1.predict(X))
        res_bus = self.label_enc_bus.inverse_transform(self.model_2.predict(X))
        res = np.concatenate((np.expand_dims(res_vac, axis=0),np.expand_dims(res_bus, axis=0))).T
        return res
