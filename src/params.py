from itertools import product
from typing import List

from .constants import CATBOOST, LGBM, TWO_STEP_MODEL, XGBOOST


XGBOOST_PARAMS_GRIDS = {
    "n_estimators": [15],
    "max_depth": [10],
    "max_leaves": [100],
    "max_bin": [40],
}

LGBM_PARAMS_GRIDS = {
    "n_estimators": [150],
    "num_leaves": [5],
    "max_depth": [50],
    "max_leaves": [100],
    "max_bin": [30],
}

CATBOOST_PARAMS_GRIDS = {
    "n_estimators": [120],
    "max_depth": [4],
}

PAEAMS_GRIDS = {
    XGBOOST: XGBOOST_PARAMS_GRIDS,
    LGBM: LGBM_PARAMS_GRIDS,
    CATBOOST: CATBOOST_PARAMS_GRIDS,
    TWO_STEP_MODEL: XGBOOST_PARAMS_GRIDS,
}


def get_params_grids(params: dict) -> List[dict]:
    keys = params.keys()
    params_grids = product(*params.values())
    return [dict(zip(keys, items)) for items in params_grids]
