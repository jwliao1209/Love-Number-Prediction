import json
from argparse import ArgumentParser, Namespace

import pandas as pd

from src.process import prepare_train_valid_dataset
from src.learner import Learner
from src.utils import save_report


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/train.csv"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=["2step-xgboost"],
        nargs="+",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    df = pd.read_csv(args.data_path)
    X_train, X_valid, y_train, y_valid = prepare_train_valid_dataset(df)

    evaluation_report = []
    for model in args.models:
        learner = Learner(model)
        results = learner.train(X_train, X_valid, y_train, y_valid)
        evaluation_report.append(
            {
                "model": model,
                "param": json.dumps(learner.best_param),
            } | results
        )

    evaluation_report = {
        k: [result[k] for result in evaluation_report]
        for k in evaluation_report[0].keys()
    }
    save_report(evaluation_report, "result/evaluation_report.csv")
