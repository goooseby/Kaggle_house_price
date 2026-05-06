from pathlib import Path

import pandas as pd


def load_raw_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, sample_submission


def remove_known_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """Remove the two classic high-area, low-price outliers for this competition."""
    mask = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)
    return train.loc[~mask].copy()
