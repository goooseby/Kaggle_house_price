from pathlib import Path

import numpy as np
import pandas as pd


def _missing_table(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return pd.DataFrame(
        {
            "missing": missing,
            "pct": (missing / len(df) * 100).round(2),
        }
    ).head(top_n)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    table = df.copy()
    table.index = table.index.astype(str)
    table = table.reset_index()
    headers = [str(column) for column in table.columns]
    rows = [[str(value) for value in row] for row in table.to_numpy()]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_eda_report(train: pd.DataFrame, test: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target = train["SalePrice"]
    log_target = np.log1p(target)

    numeric_corr = (
        train.select_dtypes(include="number")
        .corr(numeric_only=True)["SalePrice"]
        .drop("SalePrice")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(20)
        .rename("corr")
        .to_frame()
    )

    outliers = train.loc[
        (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000),
        ["Id", "GrLivArea", "SalePrice", "OverallQual", "Neighborhood"],
    ]

    categorical_cardinality = (
        train.select_dtypes(include="object")
        .nunique()
        .sort_values(ascending=False)
        .head(20)
        .rename("nunique")
        .to_frame()
    )

    neighborhood_summary = (
        train.groupby("Neighborhood")["SalePrice"]
        .agg(["count", "median", "mean"])
        .sort_values("median", ascending=False)
        .round(1)
        .head(12)
    )

    lines = [
        "# Kaggle House Prices EDA Report",
        "",
        "## Dataset Shape",
        "",
        f"- train: {train.shape[0]} rows x {train.shape[1]} columns",
        f"- test: {test.shape[0]} rows x {test.shape[1]} columns",
        f"- train-only columns: {sorted(set(train.columns) - set(test.columns))}",
        f"- test-only columns: {sorted(set(test.columns) - set(train.columns))}",
        "",
        "## Target Summary",
        "",
        _markdown_table(target.describe().rename("SalePrice").to_frame()),
        "",
        f"- SalePrice skew: {target.skew():.4f}",
        f"- log1p(SalePrice) skew: {log_target.skew():.4f}",
        "",
        "## Top Missing Values - Train",
        "",
        _markdown_table(_missing_table(train)),
        "",
        "## Top Missing Values - Test",
        "",
        _markdown_table(_missing_table(test, top_n=30)),
        "",
        "## Numeric Correlation With SalePrice",
        "",
        _markdown_table(numeric_corr),
        "",
        "## Categorical Cardinality",
        "",
        _markdown_table(categorical_cardinality),
        "",
        "## High Median Price Neighborhoods",
        "",
        _markdown_table(neighborhood_summary),
        "",
        "## Known Outlier Candidates",
        "",
        _markdown_table(outliers),
        "",
        "## Notes",
        "",
        "- The target is strongly right-skewed, so the modeling pipeline uses log1p(SalePrice).",
        "- Many missing categorical values mean the house lacks that feature, not bad data.",
        "- The two GrLivArea outliers are removed before training.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
