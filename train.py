from argparse import ArgumentParser, Namespace

import pandas as pd

from src.data_process import prepare_train_valid_dataset
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
        "--model",
        type=str,
        default="xgboost"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    df = pd.read_csv(args.data_path)
    X_train, X_valid, y_train, y_valid = prepare_train_valid_dataset(df)
    learner = Learner(args.model)
    results = learner.train(X_train, X_valid, y_train, y_valid)
    evaluation_report = {
            "model": args.model,
            **{k: [v] for k, v in results.items()}
    }
    print(evaluation_report)
    save_report(evaluation_report)
