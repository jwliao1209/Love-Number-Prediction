from datetime import date
from typing import Tuple, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import train_test_split

from .constants import CATEGORY_FEATURES, TARGET_NAME


class DataPipeline:
    def __init__(
        self,
        processor,
        category_features: List[str],
        category_encoder,
        target_name: str,
    ) -> None:

        self.processor = processor
        self.category_features = category_features
        self.category_encoder = category_encoder
        self.target_name = target_name

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for p in self.processor:
            df = p(df)

        df = encode_category_features(
            df,
            features=self.category_features,
            target_name=self.target_name,
            encoder=self.category_encoder,
            is_train=True,
        )
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for p in self.processor:
            df = p(df)
        
        df = encode_category_features(
            df,
            features=self.category_features,
            target_name=self.target_name,
            encoder=self.category_encoder,
            is_train=False,
        )
        return df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "like_count_1h",
        "like_count_2h",
        "like_count_3h",
        "like_count_4h",
        "like_count_5h",
        "like_count_6h",
        "comment_count_1h",
        "comment_count_2h",
        "comment_count_3h",
        "comment_count_4h",
        "comment_count_5h",
        "comment_count_6h",
        "weekday",
        # "is_weekend",
        "time",
        "author_id_te",
        "forum_id_te",
        "forum_stats",
        "is_weekend_te",
        "weekday_te",
        # "title_len",
        "day",
        "like_count_2h-1h/1",
        "like_count_3h-2h/1",
        "like_count_4h-3h/1",
        "like_count_5h-4h/1",
        "like_count_6h-5h/1",
        "like_count_6h-1h/5",
        "like_count_6h-2h/4",
        "like_count_6h-3h/3",
        "like_count_6h-4h/2",
        "like_count_mean",
        "like_count_std",
        "like_count_6h-1h/std",
        "like_count_mean/std",
        "comment_count_2h-1h/1",
        "comment_count_3h-2h/1",
        "comment_count_4h-3h/1",
        "comment_count_5h-4h/1",
        "comment_count_6h-5h/1",
        "comment_count_6h-1h/5",
        "comment_count_6h-2h/4",
        "comment_count_6h-3h/3",
        "comment_count_6h-4h/2",
        "comment_count_6h-5h/1",
    ]

    data_processor = [
        add_like_count_first_order_difference,
        add_comment_count_first_order_difference,
        add_like_count_mean,
        add_like_count_std,
        add_like_count_mean_divide_std,
        add_like_count_range_divide_std,
        add_weekday,
        add_is_weekend,
        add_day,
        add_time,
        add_title_len,
    ]

    target_encoder = TargetEncoder(
        random_state=0,
        target_type="continuous",
    )
    data_pipeline = DataPipeline(
        processor=data_processor,
        category_features=CATEGORY_FEATURES,
        category_encoder=target_encoder,
        target_name=TARGET_NAME,
    )

    df = data_pipeline.fit_transform(df)
    X, y = convert_df_to_X_and_y(df, features)
    return X, y


def prepare_train_valid_dataset(df: pd.DataFrame) -> pd.DataFrame:
    X, y = prepare_dataset(df)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=0.2,
        random_state=0,
    )
    return X_train, X_valid, y_train, y_valid


def get_year(date: str) -> int:
    return int(date[0:3+1])


def get_month(date: str) -> int:
    return int(date[5:6+1])


def get_day(date: str) -> int:
    return int(date[8:9+1])


def get_hour(date: str) -> int:
    return int(date[11:12+1])


def get_minute(date: str) -> int:
    return int(date[14:15+1])


def get_second(date: str) -> int:
    return int(date[17:18+1])


def get_year_and_month_and_day(date: str) -> Tuple[int]:
    return get_year(date), get_month(date), get_day(date)


def get_time_second(date: str) -> int:
    return get_hour(date) * 3600 + get_minute(date) * 60 + get_second(date)


def add_like_count_first_order_difference(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["like_count_2h-1h/1"] = (df["like_count_2h"] - df["like_count_1h"]) / 1
    df["like_count_3h-2h/1"] = (df["like_count_3h"] - df["like_count_2h"]) / 1
    df["like_count_4h-3h/1"] = (df["like_count_4h"] - df["like_count_3h"]) / 1
    df["like_count_5h-4h/1"] = (df["like_count_5h"] - df["like_count_4h"]) / 1
    df["like_count_6h-5h/1"] = (df["like_count_6h"] - df["like_count_5h"]) / 1
    df["like_count_6h-1h/5"] = (df["like_count_6h"] - df["like_count_1h"]) / 5
    df["like_count_6h-2h/4"] = (df["like_count_6h"] - df["like_count_2h"]) / 4
    df["like_count_6h-3h/3"] = (df["like_count_6h"] - df["like_count_3h"]) / 3
    df["like_count_6h-4h/2"] = (df["like_count_6h"] - df["like_count_4h"]) / 2
    return df


def add_comment_count_first_order_difference(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["comment_count_2h-1h/1"] = (df["comment_count_2h"] - df["comment_count_1h"]) / 1
    df["comment_count_3h-2h/1"] = (df["comment_count_3h"] - df["comment_count_2h"]) / 1
    df["comment_count_4h-3h/1"] = (df["comment_count_4h"] - df["comment_count_3h"]) / 1
    df["comment_count_5h-4h/1"] = (df["comment_count_5h"] - df["comment_count_4h"]) / 1
    df["comment_count_6h-5h/1"] = (df["comment_count_6h"] - df["comment_count_5h"]) / 1
    df["comment_count_6h-1h/5"] = (df["comment_count_6h"] - df["comment_count_1h"]) / 5
    df["comment_count_6h-2h/4"] = (df["comment_count_6h"] - df["comment_count_2h"]) / 4
    df["comment_count_6h-3h/3"] = (df["comment_count_6h"] - df["comment_count_3h"]) / 3
    df["comment_count_6h-4h/2"] = (df["comment_count_6h"] - df["comment_count_4h"]) / 2
    df["comment_count_6h-5h/1"] = (df["comment_count_6h"] - df["comment_count_5h"]) / 1
    return df


def add_like_count_mean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hours = [1, 2, 3, 4, 5, 6]
    df["like_count_mean"] = sum([df[f"like_count_{i}h"] for i in hours]) / len(hours)
    return df


def add_like_count_std(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "like_count_mean" not in df.columns:
        df = add_like_count_mean(df)

    hours = [1, 2, 3, 4, 5, 6]
    like_count_var = sum([df[f"like_count_{i}h"] ** 2 for i in hours]) / len(hours) - df["like_count_mean"] ** 2
    df["like_count_std"] = like_count_var ** 0.5
    return df


def add_like_count_mean_divide_std(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["like_count_mean/std"] = df["like_count_mean"] / df["like_count_std"]
    df.loc[df["like_count_std"] == 0, "like_count_mean/std"] = 0
    return df


def add_like_count_range_divide_std(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["like_count_6h-1h/std"] = df["like_count_6h-1h/5"] * 5 / df["like_count_std"]
    df.loc[df["like_count_std"] == 0, "like_count_6h-1h/std"] = 0
    return df


def add_weekday(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weekday"] = df["created_at"].apply(
        lambda x: date(*get_year_and_month_and_day(x)).weekday() + 1
    )
    return df


def add_is_weekend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "weekday" not in df.columns:
        df = add_weekday(df)
    df["is_weekend"] = df["weekday"].apply(lambda x: True if x in [6, 7] else False)
    return df


def add_day(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day"] = df["created_at"].apply(lambda x: int(x[8:9+1]))
    return df


def add_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = df["created_at"].apply(lambda x: get_time_second(x) + 1)
    return df


def add_title_len(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title_len"] = df["title"].apply(lambda x: len(x))
    return df


def encode_category_features(
    df: pd.DataFrame,
    features: List[str],
    target_name: str,
    encoder,
    is_train: bool,
    ) -> pd.DataFrame:

    df = df.copy()
    encode_features = [f"{feature}_te" for feature in features]

    for feature in features:
        df[feature] = df[feature].astype("str")

    if is_train:
        df[encode_features] = encoder.fit_transform(df[features], df[target_name])
    else:
        df[encode_features] = encoder.transform(df[features])
    return df


def convert_df_to_X_and_y(
        df: pd.DataFrame,
        features: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:

    X = df[features].to_numpy()
    y = df[TARGET_NAME].to_numpy()
    return X, y
