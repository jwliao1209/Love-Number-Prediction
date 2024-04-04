import os
import pandas as pd

from src.logger import logger


def save_report(
        results: dict,
        save_path: str,
    ) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result_df = pd.DataFrame(results)
    result_df.to_csv(save_path, index=False)
    logger.info(f"Save evaluation report to {save_path}")
    return
