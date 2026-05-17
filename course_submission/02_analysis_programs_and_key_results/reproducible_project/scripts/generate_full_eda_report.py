from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, skew

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from house_price.config import DEFAULT_DATA_DIR, PROJECT_ROOT
from house_price.data import load_raw_data


REPORT_DIR = PROJECT_ROOT / "reports" / "eda"
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
REPORT_PATH = REPORT_DIR / "20260506_full_eda_report.md"


def main() -> None:
    train, test, sample_submission = load_raw_data(DEFAULT_DATA_DIR)
    _prepare_dirs()
    _setup_plot_style()

    tables = _build_tables(train, test, sample_submission)
    _write_tables(tables)
    figure_paths = _build_figures(train, test, tables)
    _write_report(train, test, sample_submission, tables, figure_paths)

    print(f"EDA report written to: {REPORT_PATH}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Tables: {TABLE_DIR}")


def _prepare_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 160
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]


def _build_tables(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    numeric_columns = train.drop(columns=["SalePrice"]).select_dtypes(include="number").columns
    categorical_columns = train.select_dtypes(include=["object", "str"]).columns

    tables = {
        "dataset_overview": pd.DataFrame(
            [
                {
                    "dataset": "train",
                    "rows": train.shape[0],
                    "columns": train.shape[1],
                    "missing_cells": int(train.isna().sum().sum()),
                    "duplicate_rows": int(train.duplicated().sum()),
                },
                {
                    "dataset": "test",
                    "rows": test.shape[0],
                    "columns": test.shape[1],
                    "missing_cells": int(test.isna().sum().sum()),
                    "duplicate_rows": int(test.duplicated().sum()),
                },
                {
                    "dataset": "sample_submission",
                    "rows": sample_submission.shape[0],
                    "columns": sample_submission.shape[1],
                    "missing_cells": int(sample_submission.isna().sum().sum()),
                    "duplicate_rows": int(sample_submission.duplicated().sum()),
                },
            ]
        ),
        "dtype_summary": train.dtypes.value_counts().rename_axis("dtype").reset_index(name="count"),
        "target_summary": _target_summary(train),
        "missing_train": _missing_table(train),
        "missing_test": _missing_table(test),
        "numeric_correlations": _numeric_correlations(train),
        "numeric_skewness": _numeric_skewness(train[numeric_columns]),
        "categorical_cardinality": _categorical_cardinality(train[categorical_columns]),
        "neighborhood_price": _neighborhood_price(train),
        "overallqual_price": _overallqual_price(train),
        "outlier_candidates": _outlier_candidates(train),
        "train_test_numeric_drift": _numeric_drift(train, test, numeric_columns),
        "train_test_categorical_drift": _categorical_drift(train, test, categorical_columns),
        "low_variance_features": _low_variance_features(train),
    }
    return tables


def _target_summary(train: pd.DataFrame) -> pd.DataFrame:
    sale_price = train["SalePrice"]
    log_price = np.log1p(sale_price)
    return pd.DataFrame(
        [
            {
                "metric": "count",
                "SalePrice": sale_price.count(),
                "log1p_SalePrice": log_price.count(),
            },
            {
                "metric": "mean",
                "SalePrice": sale_price.mean(),
                "log1p_SalePrice": log_price.mean(),
            },
            {
                "metric": "std",
                "SalePrice": sale_price.std(),
                "log1p_SalePrice": log_price.std(),
            },
            {
                "metric": "min",
                "SalePrice": sale_price.min(),
                "log1p_SalePrice": log_price.min(),
            },
            {
                "metric": "25%",
                "SalePrice": sale_price.quantile(0.25),
                "log1p_SalePrice": log_price.quantile(0.25),
            },
            {
                "metric": "50%",
                "SalePrice": sale_price.median(),
                "log1p_SalePrice": log_price.median(),
            },
            {
                "metric": "75%",
                "SalePrice": sale_price.quantile(0.75),
                "log1p_SalePrice": log_price.quantile(0.75),
            },
            {
                "metric": "max",
                "SalePrice": sale_price.max(),
                "log1p_SalePrice": log_price.max(),
            },
            {
                "metric": "skew",
                "SalePrice": sale_price.skew(),
                "log1p_SalePrice": log_price.skew(),
            },
        ]
    ).round(4)


def _missing_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return pd.DataFrame(
        {
            "feature": missing.index,
            "missing": missing.values,
            "missing_pct": (missing.values / len(df) * 100).round(2),
        }
    )


def _numeric_correlations(train: pd.DataFrame) -> pd.DataFrame:
    corr = (
        train.select_dtypes(include="number")
        .corr(numeric_only=True)["SalePrice"]
        .drop("SalePrice")
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    table = corr.rename("corr_with_saleprice").reset_index()
    table.columns = ["feature", "corr_with_saleprice"]
    return table.round(4)


def _numeric_skewness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        values = df[column].dropna()
        if values.nunique() <= 2:
            continue
        rows.append({"feature": column, "skewness": skew(values)})
    return pd.DataFrame(rows).sort_values("skewness", ascending=False).round(4)


def _categorical_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        top_value = df[column].value_counts(dropna=False).index[0]
        top_pct = df[column].value_counts(dropna=False, normalize=True).iloc[0] * 100
        rows.append(
            {
                "feature": column,
                "unique_values": df[column].nunique(dropna=False),
                "top_value": str(top_value),
                "top_pct": round(top_pct, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("unique_values", ascending=False)


def _neighborhood_price(train: pd.DataFrame) -> pd.DataFrame:
    return (
        train.groupby("Neighborhood")["SalePrice"]
        .agg(["count", "mean", "median", "min", "max"])
        .sort_values("median", ascending=False)
        .round(2)
        .reset_index()
    )


def _overallqual_price(train: pd.DataFrame) -> pd.DataFrame:
    return (
        train.groupby("OverallQual")["SalePrice"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
        .reset_index()
    )


def _outlier_candidates(train: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000),
        train["SalePrice"] > train["SalePrice"].quantile(0.995),
        train["LotArea"] > train["LotArea"].quantile(0.995),
    ]
    mask = np.logical_or.reduce(conditions)
    columns = [
        "Id",
        "SalePrice",
        "GrLivArea",
        "LotArea",
        "OverallQual",
        "TotalBsmtSF",
        "GarageCars",
        "Neighborhood",
        "YearBuilt",
    ]
    return train.loc[mask, columns].sort_values("SalePrice", ascending=False)


def _numeric_drift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_columns: pd.Index,
) -> pd.DataFrame:
    rows = []
    for column in numeric_columns:
        train_values = train[column].dropna()
        test_values = test[column].dropna()
        if train_values.empty or test_values.empty:
            continue
        ks_stat, p_value = ks_2samp(train_values, test_values)
        rows.append(
            {
                "feature": column,
                "train_mean": train_values.mean(),
                "test_mean": test_values.mean(),
                "mean_diff": test_values.mean() - train_values.mean(),
                "ks_stat": ks_stat,
                "ks_pvalue": p_value,
            }
        )
    return pd.DataFrame(rows).sort_values("ks_stat", ascending=False).round(5)


def _categorical_drift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical_columns: pd.Index,
) -> pd.DataFrame:
    rows = []
    for column in categorical_columns:
        train_levels = set(train[column].dropna().unique())
        test_levels = set(test[column].dropna().unique())
        rows.append(
            {
                "feature": column,
                "train_unique": len(train_levels),
                "test_unique": len(test_levels),
                "test_only_levels": ", ".join(sorted(test_levels - train_levels)),
                "train_only_levels": ", ".join(sorted(train_levels - test_levels)),
            }
        )
    table = pd.DataFrame(rows)
    table["has_level_difference"] = (
        (table["test_only_levels"] != "") | (table["train_only_levels"] != "")
    )
    return table.sort_values(["has_level_difference", "feature"], ascending=[False, True])


def _low_variance_features(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in train.drop(columns=["Id", "SalePrice"]).columns:
        proportions = train[column].value_counts(dropna=False, normalize=True)
        if proportions.empty:
            continue
        top_pct = proportions.iloc[0] * 100
        if top_pct >= 95:
            rows.append(
                {
                    "feature": column,
                    "top_value": str(proportions.index[0]),
                    "top_pct": round(top_pct, 2),
                    "unique_values": train[column].nunique(dropna=False),
                }
            )
    return pd.DataFrame(rows).sort_values("top_pct", ascending=False)


def _write_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, table in tables.items():
        table.to_csv(TABLE_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")


def _build_figures(
    train: pd.DataFrame,
    test: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    figures = {}
    figures["target_distribution"] = _plot_target_distribution(train)
    figures["target_log_distribution"] = _plot_target_log_distribution(train)
    figures["missing_train"] = _plot_missing_values(tables["missing_train"], "Train Missing Values", "missing_train.png")
    figures["missing_test"] = _plot_missing_values(tables["missing_test"], "Test Missing Values", "missing_test.png")
    figures["correlation_bar"] = _plot_correlation_bar(tables["numeric_correlations"])
    figures["correlation_heatmap"] = _plot_correlation_heatmap(train, tables["numeric_correlations"])
    figures["grlivarea_saleprice"] = _plot_scatter(train, "GrLivArea", "SalePrice", "GrLivArea vs SalePrice", "grlivarea_saleprice.png")
    figures["totalbsmtsf_saleprice"] = _plot_scatter(train, "TotalBsmtSF", "SalePrice", "TotalBsmtSF vs SalePrice", "totalbsmtsf_saleprice.png")
    figures["overallqual_boxplot"] = _plot_overallqual_boxplot(train)
    figures["neighborhood_median_price"] = _plot_neighborhood_price(tables["neighborhood_price"])
    figures["yearbuilt_saleprice"] = _plot_year_scatter(train, "YearBuilt", "yearbuilt_saleprice.png")
    figures["yearremod_saleprice"] = _plot_year_scatter(train, "YearRemodAdd", "yearremod_saleprice.png")
    figures["train_test_lotarea"] = _plot_train_test_numeric(train, test, "LotArea", "train_test_lotarea.png")
    figures["train_test_grlivarea"] = _plot_train_test_numeric(train, test, "GrLivArea", "train_test_grlivarea.png")
    figures["train_test_overallqual"] = _plot_train_test_numeric(train, test, "OverallQual", "train_test_overallqual.png")
    figures["saleprice_by_house_style"] = _plot_category_boxplot(train, "HouseStyle", "saleprice_by_house_style.png")
    figures["saleprice_by_exterqual"] = _plot_category_boxplot(train, "ExterQual", "saleprice_by_exterqual.png")
    return figures


def _plot_target_distribution(train: pd.DataFrame) -> Path:
    path = FIGURE_DIR / "target_distribution.png"
    plt.figure(figsize=(9, 5))
    sns.histplot(train["SalePrice"], kde=True, bins=40)
    plt.title("SalePrice Distribution")
    plt.xlabel("SalePrice")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_target_log_distribution(train: pd.DataFrame) -> Path:
    path = FIGURE_DIR / "target_log_distribution.png"
    plt.figure(figsize=(9, 5))
    sns.histplot(np.log1p(train["SalePrice"]), kde=True, bins=40, color="#2a9d8f")
    plt.title("log1p(SalePrice) Distribution")
    plt.xlabel("log1p(SalePrice)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_missing_values(table: pd.DataFrame, title: str, filename: str) -> Path:
    path = FIGURE_DIR / filename
    plot_data = table.head(25).sort_values("missing_pct", ascending=True)
    plt.figure(figsize=(9, 7))
    sns.barplot(data=plot_data, x="missing_pct", y="feature", color="#457b9d")
    plt.title(title)
    plt.xlabel("Missing %")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_correlation_bar(table: pd.DataFrame) -> Path:
    path = FIGURE_DIR / "correlation_bar.png"
    plot_data = table.head(20).sort_values("corr_with_saleprice", ascending=True)
    plt.figure(figsize=(9, 7))
    sns.barplot(data=plot_data, x="corr_with_saleprice", y="feature", color="#1d3557")
    plt.title("Top Numeric Correlations With SalePrice")
    plt.xlabel("Correlation")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_correlation_heatmap(train: pd.DataFrame, corr_table: pd.DataFrame) -> Path:
    path = FIGURE_DIR / "correlation_heatmap.png"
    columns = corr_table.head(12)["feature"].tolist() + ["SalePrice"]
    corr = train[columns].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3, annot=False)
    plt.title("Correlation Heatmap of Key Numeric Features")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_scatter(
    train: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    filename: str,
) -> Path:
    path = FIGURE_DIR / filename
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=train, x=x, y=y, hue="OverallQual", palette="viridis", alpha=0.75)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_overallqual_boxplot(train: pd.DataFrame) -> Path:
    path = FIGURE_DIR / "overallqual_boxplot.png"
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=train, x="OverallQual", y="SalePrice", color="#a8dadc")
    plt.title("SalePrice by OverallQual")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_neighborhood_price(table: pd.DataFrame) -> Path:
    path = FIGURE_DIR / "neighborhood_median_price.png"
    plot_data = table.sort_values("median", ascending=True)
    plt.figure(figsize=(9, 9))
    sns.barplot(data=plot_data, x="median", y="Neighborhood", color="#e76f51")
    plt.title("Median SalePrice by Neighborhood")
    plt.xlabel("Median SalePrice")
    plt.ylabel("Neighborhood")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_year_scatter(train: pd.DataFrame, year_column: str, filename: str) -> Path:
    path = FIGURE_DIR / filename
    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=train, x=year_column, y="SalePrice", alpha=0.65, color="#264653")
    sns.regplot(data=train, x=year_column, y="SalePrice", scatter=False, color="#e76f51")
    plt.title(f"{year_column} vs SalePrice")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_train_test_numeric(
    train: pd.DataFrame,
    test: pd.DataFrame,
    column: str,
    filename: str,
) -> Path:
    path = FIGURE_DIR / filename
    plot_data = pd.concat(
        [
            pd.DataFrame({"dataset": "train", column: train[column]}),
            pd.DataFrame({"dataset": "test", column: test[column]}),
        ],
        axis=0,
        ignore_index=True,
    )
    plt.figure(figsize=(9, 5))
    if plot_data[column].min() >= 0 and plot_data[column].skew() > 1:
        plot_data[column] = np.log1p(plot_data[column])
        xlabel = f"log1p({column})"
    else:
        xlabel = column
    sns.kdeplot(data=plot_data, x=column, hue="dataset", common_norm=False, fill=True, alpha=0.25)
    plt.title(f"Train/Test Distribution: {xlabel}")
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_category_boxplot(train: pd.DataFrame, column: str, filename: str) -> Path:
    path = FIGURE_DIR / filename
    order = train.groupby(column)["SalePrice"].median().sort_values().index
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=train, x=column, y="SalePrice", order=order, color="#f4a261")
    plt.title(f"SalePrice by {column}")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _write_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    figures: dict[str, Path],
) -> None:
    target = train["SalePrice"]
    log_target = np.log1p(target)
    outliers = tables["outlier_candidates"]
    best_corr = tables["numeric_correlations"].head(10)
    top_missing_train = tables["missing_train"].head(10)
    top_missing_test = tables["missing_test"].head(10)
    drift_numeric = tables["train_test_numeric_drift"].head(10)
    drift_categorical = tables["train_test_categorical_drift"]
    drift_categorical = drift_categorical[drift_categorical["has_level_difference"]].head(15)
    neighborhood_top = tables["neighborhood_price"].head(10)
    neighborhood_bottom = tables["neighborhood_price"].tail(10).sort_values("median")

    lines = [
        "# House Prices 完整探索性分析报告",
        "",
        "## 1. 报告目的",
        "",
        "这份报告单独补充 Kaggle House Prices 题目的探索性分析，目标是为后续建模、复盘和正式报告提供图表与数据依据。",
        "",
        "本报告覆盖：",
        "",
        "- 数据规模和字段类型",
        "- 目标值 `SalePrice` 分布",
        "- 缺失值结构",
        "- 数值特征相关性",
        "- 类别特征与价格关系",
        "- 异常点识别",
        "- 训练集和测试集分布差异",
        "- 对建模策略的启发",
        "",
        "## 2. 数据概览",
        "",
        _markdown(tables["dataset_overview"]),
        "",
        "训练集比测试集多出的字段是 `SalePrice`，也就是本题的预测目标。",
        "",
        "字段类型统计：",
        "",
        _markdown(tables["dtype_summary"]),
        "",
        "## 3. 目标值 SalePrice 分析",
        "",
        _markdown(tables["target_summary"]),
        "",
        f"`SalePrice` 原始偏度为 `{target.skew():.4f}`，右偏明显；`log1p(SalePrice)` 后偏度为 `{log_target.skew():.4f}`，分布更接近对称。",
        "",
        f"![SalePrice Distribution]({ _relative(figures['target_distribution']) })",
        "",
        f"![log1p SalePrice Distribution]({ _relative(figures['target_log_distribution']) })",
        "",
        "建模启发：Kaggle 该题的 RMSLE 指标等价于在 log 价格上计算 RMSE，因此目标值使用 `log1p(SalePrice)` 是合理且必要的。",
        "",
        "## 4. 缺失值分析",
        "",
        "训练集缺失值 Top 10：",
        "",
        _markdown(top_missing_train),
        "",
        "测试集缺失值 Top 10：",
        "",
        _markdown(top_missing_test),
        "",
        f"![Train Missing Values]({ _relative(figures['missing_train']) })",
        "",
        f"![Test Missing Values]({ _relative(figures['missing_test']) })",
        "",
        "关键观察：",
        "",
        "- `PoolQC`、`MiscFeature`、`Alley`、`Fence` 缺失比例极高，多数情况下不是脏数据，而是表示房子没有对应设施。",
        "- `Garage*` 和 `Bsmt*` 系列字段的缺失往往成组出现，通常表示没有车库或地下室。",
        "- `LotFrontage` 缺失较多，适合按 `Neighborhood` 分组中位数填充。",
        "- 测试集存在少量训练集中没有缺失、测试集中有缺失的字段，预测前必须统一处理。",
        "",
        "## 5. 数值特征与 SalePrice 的关系",
        "",
        "相关性最高的数值特征：",
        "",
        _markdown(best_corr),
        "",
        f"![Correlation Bar]({ _relative(figures['correlation_bar']) })",
        "",
        f"![Correlation Heatmap]({ _relative(figures['correlation_heatmap']) })",
        "",
        "关键观察：",
        "",
        "- `OverallQual` 与价格关系最强，是最重要的单一数值特征。",
        "- `GrLivArea`、`GarageCars`、`GarageArea`、`TotalBsmtSF`、`1stFlrSF` 都和面积或容量相关，说明房屋规模是核心信号。",
        "- `YearBuilt`、`YearRemodAdd` 有明显正相关，新房和较新翻修的房子通常更贵。",
        "",
        f"![GrLivArea vs SalePrice]({ _relative(figures['grlivarea_saleprice']) })",
        "",
        f"![TotalBsmtSF vs SalePrice]({ _relative(figures['totalbsmtsf_saleprice']) })",
        "",
        "## 6. 质量、街区和类别特征",
        "",
        "`OverallQual` 分组价格统计：",
        "",
        _markdown(tables["overallqual_price"]),
        "",
        f"![OverallQual Boxplot]({ _relative(figures['overallqual_boxplot']) })",
        "",
        "房屋整体质量越高，价格中位数越高，这个关系非常稳定。",
        "",
        "房价中位数最高的街区：",
        "",
        _markdown(neighborhood_top),
        "",
        "房价中位数最低的街区：",
        "",
        _markdown(neighborhood_bottom),
        "",
        f"![Neighborhood Median Price]({ _relative(figures['neighborhood_median_price']) })",
        "",
        f"![SalePrice by HouseStyle]({ _relative(figures['saleprice_by_house_style']) })",
        "",
        f"![SalePrice by ExterQual]({ _relative(figures['saleprice_by_exterqual']) })",
        "",
        "建模启发：`Neighborhood` 不只是普通类别变量，它携带了强烈的位置溢价信息；质量类字段适合做序数编码，而不是全部简单 one-hot。",
        "",
        "## 7. 年份变量",
        "",
        f"![YearBuilt vs SalePrice]({ _relative(figures['yearbuilt_saleprice']) })",
        "",
        f"![YearRemodAdd vs SalePrice]({ _relative(figures['yearremod_saleprice']) })",
        "",
        "关键观察：新建年份和翻新年份越新，价格整体越高。后续特征工程中构造 `HouseAge`、`RemodAge`、`IsNewHouse` 是有依据的。",
        "",
        "## 8. 异常点分析",
        "",
        "候选异常点：",
        "",
        _markdown(outliers),
        "",
        "最重要的两个异常点是：",
        "",
        "- `Id=524`：`GrLivArea=4676`，但 `SalePrice=184750`",
        "- `Id=1299`：`GrLivArea=5642`，但 `SalePrice=160000`",
        "",
        "这两个点面积极大但价格偏低，会破坏面积和价格之间的主要趋势。因此 baseline 和后续模型中删除它们是合理的。",
        "",
        "## 9. 训练集与测试集分布差异",
        "",
        "数值特征分布差异 Top 10，按 KS statistic 排序：",
        "",
        _markdown(drift_numeric),
        "",
        f"![Train/Test LotArea]({ _relative(figures['train_test_lotarea']) })",
        "",
        f"![Train/Test GrLivArea]({ _relative(figures['train_test_grlivarea']) })",
        "",
        f"![Train/Test OverallQual]({ _relative(figures['train_test_overallqual']) })",
        "",
        "类别特征中，训练集和测试集存在取值差异的字段：",
        "",
        _markdown(drift_categorical),
        "",
        "建模启发：训练集和测试集整体结构接近，但某些稀有类别只在训练集出现。预处理时应该合并 train/test 后统一编码，避免 one-hot 列不一致。",
        "",
        "## 10. 低方差特征",
        "",
        _markdown(tables["low_variance_features"]),
        "",
        "这些字段绝大多数样本取同一个值，单独信息量有限。不过在树模型或特定交互中仍可能有少量价值，暂时不必全部删除。",
        "",
        "## 11. 对建模的总结启发",
        "",
        "这次 EDA 支持以下建模策略：",
        "",
        "1. 目标值使用 `log1p(SalePrice)`。",
        "2. 删除 `GrLivArea` 极大但价格异常低的两个训练样本。",
        "3. 缺失值需要按业务含义处理，不能简单全部均值填充。",
        "4. `OverallQual`、面积类、车库、地下室、年份和街区是主信号。",
        "5. 质量类字段适合序数编码。",
        "6. 面积类和部分数值特征右偏明显，适合 `log1p` 变换。",
        "7. train/test 编码必须统一处理，避免类别列不一致。",
        "8. 高价预测容易外推，后续模型需要关注高价区间和预测裁剪。",
        "",
        "## 12. 输出文件",
        "",
        "图表目录：",
        "",
        f"`{FIGURE_DIR}`",
        "",
        "表格目录：",
        "",
        f"`{TABLE_DIR}`",
        "",
        "本报告路径：",
        "",
        f"`{REPORT_PATH}`",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无记录_"
    table = df.copy()
    table = table.head(30)
    headers = [str(column) for column in table.columns]
    rows = [[_format_value(value) for value in row] for row in table.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _relative(path: Path) -> str:
    return path.relative_to(REPORT_PATH.parent).as_posix()


if __name__ == "__main__":
    main()
