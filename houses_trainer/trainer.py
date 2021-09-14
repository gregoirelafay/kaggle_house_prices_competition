"""
COMPUTE AND PRINT CROSS VALIDATED RMSLE OF A PIPELINED MODEL
FOR THE KAGGLE HOUSES ADVANCED REGRESSION CHALLENGE
"""

import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

from houses_trainer.pipeline import create_pipeline

class Trainer:
    def __init__(self, model="stacking"):
        self.competition_ref = "house-prices-advanced-regression-techniques"
        self.data_train = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_train_log = None
        self.cachedir = None
        self.pipe = None
        self.model = model
        self.train_path = "raw_data/train.csv"
        self.test_path = "raw_data/test.csv"
        self.predictions = None
        self.results = None
        self.submission_file_name = None

    def load_data(self):
        print("---- loading data... ----")
        self.data_train = pd.read_csv(self.train_path)
        self.data_test = pd.read_csv(self.test_path)
        self.X_train = self.data_train.drop(columns=['SalePrice','Id'])
        self.y_train = self.data_train.SalePrice
        self.y_train_log = np.log(self.y_train)

        self.X_test = self.data_test.drop(columns='Id')
        self.X_test_ids = self.data_test['Id']

        print(f"---- X_train shape: {self.X_train.shape} ----")
        print(f"---- X_test shape: {self.X_test.shape} ----")

    def build_pipeline(self, feature_cutoff_percentage=75):
        assert isinstance(
            self.X_train,
            pd.DataFrame), "Load data first using Trainer.load_data()"
        assert self.model in {
            "ridge", "knn", "svr", "decisionTree", "randomForest",
            "boostedTrees", "stacking", "xgboost"
        }, "Choose one of the following models: ridge, knn, svr, decisionTree, randomForest, boostedTrees, stacking, xgboost"
        print("---- preprocessing... ----")
        self.pipe = create_pipeline(self.X_train, self.model, feature_cutoff_percentage)
        # print shape post feature preprocessing
        X_preproc = self.pipe.named_steps["pipeline"].fit_transform(
            self.X_train, self.y_train_log)
        print(f"---- X preproc shape: {X_preproc.shape} ----")

    def cross_validate(self, cv=5):
        assert self.pipe, "Build pipeline first using Trainer.build_pipeline()"
        print(f"---- cross validate {cv} folds----")
        rmse = make_scorer(
            lambda y, y_true: mean_squared_error(y, y_true)**0.5)
        cvs = cross_val_score(self.pipe,
                              self.X_train,
                              self.y_train_log,
                              cv=5,
                              scoring=rmse,
                              verbose=0,
                              n_jobs=-1)
        print(f'Mean RMSLE: {cvs.mean()} \nStandard Dev : {cvs.std()}')

    def predict(self):
        assert self.pipe, "Build pipeline first using Trainer.build_pipeline()"
        print(f"---- Predict ----")
        self.pipe.fit(self.X_train, self.y_train_log)
        self.predictions = np.exp(self.pipe.predict(self.X_test))
        self.results = pd.concat(
            [self.X_test_ids,
             pd.Series(self.predictions, name="SalePrice")],
            axis=1)
        timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
        self.submission_file_name = f"submission/submission_{self.model}_{timestamp}.csv"
        self.results.to_csv(self.submission_file_name,
                            header=True,
                            index=False)
        print(f"---- Results saved in {self.submission_file_name} ----")


    def submit(self):
        assert self.submission_file_name, "Make a prediction first using Trainer.predict()"

        os.system(
            f'kaggle competitions submit {self.competition_ref} -f {self.submission_file_name} -m "Model: {self.model}"'
        )
