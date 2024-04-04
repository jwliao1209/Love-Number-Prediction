from typing import Tuple

import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error

from src.constants import CATBOOST, LGBM, XGBOOST, TWO_STEP_MODEL
from src.params import get_params_grids, PAEAMS_GRIDS


def init_estimator(estimator_name: str, param: dict):
    if estimator_name == XGBOOST:
        estimator = XGBRegressor(
            **param,
            objective="reg:squaredlogerror",
            random_state=250,
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
            random_state=0,
            verbose=-1,
        )
    
    elif estimator_name == TWO_STEP_MODEL:
        estimator = TwoStepModel(
            init_estimator(XGBOOST, param),
            init_estimator(XGBOOST, param),
        )
    return estimator


def compute_squared_log_error_grad(
        y_true: np.array,
        y_pred: np.array,
        epsilon: float = 1e-6,
    ) -> np.array:
    """
    grad = (log(y_pred+1) - log(y_true+1)) / (y_pred + 1)
    """
    y_pred[y_pred < -1] = -1 + epsilon
    return  (np.log1p(y_pred) - np.log1p(y_true)) / (y_pred + 1)


def compute_squared_log_error_hess(
        y_true: np.array,
        y_pred: np.array,
    ) -> np.array:
    """
    hess = (-log(y_pred+1) + log(y_true+1) + 1) / (y_pred + 1)^2
    """
    epsilon = 1e-6
    y_pred[y_pred < -1] = -1 + epsilon
    return  (-np.log1p(y_pred) + np.log1p(y_true) + 1) / (y_pred + 1) ** 2


def squared_log_error_objective(
        y_true: np.array,
        y_pred: np.array,
    ) -> Tuple[np.array, np.array]:
    """
    squared_log_error = 0.5 * (log(y_pred+1) - log(y_true+1)) ** 2
    """
    epsilon = 1e-6
    grad = compute_squared_log_error_grad(y_true, y_pred, epsilon)
    hess = compute_squared_log_error_hess(y_true, y_pred, epsilon)
    return grad, hess


class SquaredLogErrorObjective:
    def calc_ders_range(
            self,
            approxes: np.array,
            targets: np.array,
            weights: np.array
        ):
        weights = weights if weights is not None else np.ones(len(targets))
        epsilon = 1e-6
        grad = compute_squared_log_error_grad(targets, approxes, epsilon)
        hess = compute_squared_log_error_hess(targets, approxes, epsilon)
        return list(zip(grad * weights, hess * weights))


class TwoStepModel:
    def __init__(self, estimator1, estimator2):
        self.estimator1 = estimator1
        self.estimator2 = estimator2
    
    def label_transform(self, X, y):
        return y - X[:, 5]
    
    def inverse_label_transform(self, X, y):
        return y + X[:, 5]

    def fit(self, X_train, y_train, X_valid):
        y_train = self.label_transform(X_train, y_train)

        # step 1: train the model to predict the pseudo labels
        self.estimator1.fit(X_train, y_train)
        y_pseudo = self.estimator1.predict(X_valid)

        # step 2: combine the train and valid and train the model
        X = np.vstack([X_train, X_valid])
        y = np.hstack([y_train, y_pseudo])
        self.estimator2.fit(X, y)

    def predict(self, X):
        pred = self.estimator2.predict(X)
        return self.inverse_label_transform(X, pred)


class Learner:
    def __init__(self, model_name):
        self.model_name = model_name

    def train(self, X_train, X_valid, y_train, y_valid):
        evaluation_results = {}
        for param in get_params_grids(PAEAMS_GRIDS[self.model_name]):
            model = init_estimator(self.model_name, param)

            if self.model_name == TWO_STEP_MODEL:
                model.fit(X_train, y_train, X_valid)
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_train)
            train_results = self.evaluate(y_train, y_pred, "train")

            y_pred = model.predict(X_valid)
            valid_results = self.evaluate(y_valid, y_pred, "valid")

            results = train_results | valid_results

        return results

    def evaluate(self, y_true, y_pred, prefix):
        eval_fun = {
            "mean_absolute_percentage_error": mean_absolute_percentage_error,
        }

        results = {}
        for name, f in eval_fun.items():
            results[f"{prefix}_{name}"] = f(y_true, y_pred)
        return results
