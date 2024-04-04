from typing import List, Tuple
import numpy as np


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
        epsilon: float = 1e-6,
    ) -> np.array:
    """
    hess = (-log(y_pred+1) + log(y_true+1) + 1) / (y_pred + 1)^2
    """
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
    ) -> List[Tuple[np.array, np.array]]:
        weights = weights if weights is not None else np.ones(len(targets))
        epsilon = 1e-6
        grad = compute_squared_log_error_grad(targets, approxes, epsilon)
        hess = compute_squared_log_error_hess(targets, approxes, epsilon)
        return list(zip(grad * weights, hess * weights))
