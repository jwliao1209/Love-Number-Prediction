import pandas as pd


def save_report(results: dict) -> None:
    result_df = pd.DataFrame(results)
    result_df.to_csv("result.csv", index=False)
    return
