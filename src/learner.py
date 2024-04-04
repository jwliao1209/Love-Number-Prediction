import warnings

import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error

from src.constants import (
    CATBOOST,
    LGBM,
    XGBOOST,
    ONE_STEP_CATBOOST,
    ONE_STEP_LGBM,
    ONE_STEP_XGBOOST,
    TWO_STEP_CATBOOST,
    TWO_STEP_LGBM,
    TWO_STEP_XGBOOST,
    MAPE,
)
from src.logger import logger
from src.params import get_params_grids, PAEAMS_GRIDS
from src.objective import squared_log_error_objective, SquaredLogErrorObjective


warnings.simplefilter(action="ignore", category=UserWarning)


class Learner:
    def __init__(self, model_name):
        self.model_name = model_name
        self.best_param = None

    def train(self, X_train, X_valid, y_train, y_valid):
        # grid serach
        best_score = np.Inf
        for param in get_params_grids(PAEAMS_GRIDS[self.model_name]):
            model = init_estimator(self.model_name, param)

            if self.model_name in [TWO_STEP_CATBOOST, TWO_STEP_LGBM, TWO_STEP_XGBOOST]:
                model.fit(X_train, y_train, X_valid)
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_train)
            train_results = self.evaluate(y_train, y_pred, "train")

            y_pred = model.predict(X_valid)
            valid_results = self.evaluate(y_valid, y_pred, "valid")

            results = train_results | valid_results
            if best_score > results["valid_mape"]:
                best_score = results["valid_mape"]
                self.best_param = param
                evaluation_results = train_results | valid_results

            logger.info(param | results)

        model = init_estimator(self.model_name, self.best_param)
        if self.model_name in [TWO_STEP_CATBOOST, TWO_STEP_LGBM, TWO_STEP_XGBOOST]:
            model.fit(X_train, y_train, X_valid)
        else:
            model.fit(X_train, y_train)

        return evaluation_results

    def evaluate(self, y_true, y_pred, prefix):
        eval_fun = {
            MAPE: mean_absolute_percentage_error,
        }

        results = {}
        for name, f in eval_fun.items():
            results[f"{prefix}_{name}"] = f(y_true, y_pred)
        return results


class BaseModel:
    def label_transform(self, X, y):
        return y - X[:, 5]

    def inverse_label_transform(self, X, y):
        return y + X[:, 5]
    
    def fit(self):
        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError


class OneStepModel(BaseModel):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y = self.label_transform(X, y)
        self.estimator.fit(X, y)

    def predict(self, X):
        pred = self.estimator.predict(X)
        return self.inverse_label_transform(X, pred)
        

class TwoStepModel(BaseModel):
    def __init__(self, estimator1, estimator2):
        self.estimator1 = estimator1
        self.estimator2 = estimator2

    def fit(self, X_train, y_train, X_valid):
        y_train = self.label_transform(X_train, y_train)

        # step 1: train the model to predict the pseudo labels
        self.estimator1.fit(X_train, y_train)
        y_pseudo = self.estimator1.predict(X_valid)
        y_pseudo[y_pseudo < 0] = 0  # convert the nagative prediction to positive

        # step 2: combine the train and valid and train the model
        X = np.vstack([X_train, X_valid])
        y = np.hstack([y_train, y_pseudo])
        self.estimator2.fit(X, y)

    def predict(self, X):
        pred = self.estimator2.predict(X)
        return self.inverse_label_transform(X, pred)


def init_estimator(estimator_name: str, param: dict):
    if estimator_name == XGBOOST:
        estimator = XGBRegressor(
            **param,
            objective="reg:squaredlogerror",
            random_state=0,
            eval_metric=mean_absolute_percentage_error,
        )

    elif estimator_name == LGBM:
        estimator = LGBMRegressor(
            **param,
            objective=squared_log_error_objective,
            force_row_wise=True,
            random_state=0,
            verbose=-1,
        )
    
    elif estimator_name == CATBOOST:
        estimator = CatBoostRegressor(
            **param,
            objective=SquaredLogErrorObjective(),
            eval_metric="MAPE",
            allow_writing_files=False,
            random_state=250,
            verbose=0,
        )

    elif estimator_name == ONE_STEP_XGBOOST:
        estimator = OneStepModel(
            init_estimator(XGBOOST, param),
        )
    
    elif estimator_name == ONE_STEP_LGBM:
        estimator = OneStepModel(
            init_estimator(LGBM, param),
        )
    
    elif estimator_name == ONE_STEP_CATBOOST:
        estimator = OneStepModel(
            init_estimator(CATBOOST, param),
        )

    elif estimator_name == TWO_STEP_XGBOOST:
        estimator = TwoStepModel(
            init_estimator(XGBOOST, param),
            init_estimator(XGBOOST, param),
        )
    
    elif estimator_name == TWO_STEP_LGBM:
        estimator = TwoStepModel(
            init_estimator(LGBM, param),
            init_estimator(LGBM, param),
        )
    
    elif estimator_name == TWO_STEP_CATBOOST:
        estimator = TwoStepModel(
            init_estimator(CATBOOST, param),
            init_estimator(CATBOOST, param),
        )

    return estimator
