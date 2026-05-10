from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from house_price.config import DEFAULT_DATA_DIR, PROJECT_ROOT
from house_price.data import load_raw_data, remove_known_outliers
from house_price.advanced_preprocessing import build_advanced_feature_matrix


OUTPUT_DIR = PROJECT_ROOT / "reports" / "final_report" / "tables"


FEATURE_GROUPS = {
    "房屋类型与 zoning": [
        "MSSubClass",
        "MSZoning",
        "BldgType",
        "HouseStyle",
    ],
    "地块与位置": [
        "LotFrontage",
        "LotArea",
        "Street",
        "Alley",
        "LotShape",
        "LandContour",
        "Utilities",
        "LotConfig",
        "LandSlope",
        "Neighborhood",
        "Condition1",
        "Condition2",
    ],
    "建筑质量与材料": [
        "OverallQual",
        "OverallCond",
        "RoofStyle",
        "RoofMatl",
        "Exterior1st",
        "Exterior2nd",
        "MasVnrType",
        "MasVnrArea",
        "ExterQual",
        "ExterCond",
        "Foundation",
    ],
    "地下室": [
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinSF1",
        "BsmtFinType2",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
    ],
    "室内面积与功能": [
        "Heating",
        "HeatingQC",
        "CentralAir",
        "Electrical",
        "1stFlrSF",
        "2ndFlrSF",
        "LowQualFinSF",
        "GrLivArea",
        "BsmtFullBath",
        "BsmtHalfBath",
        "FullBath",
        "HalfBath",
        "BedroomAbvGr",
        "KitchenAbvGr",
        "KitchenQual",
        "TotRmsAbvGrd",
        "Functional",
        "Fireplaces",
        "FireplaceQu",
    ],
    "车库": [
        "GarageType",
        "GarageYrBlt",
        "GarageFinish",
        "GarageCars",
        "GarageArea",
        "GarageQual",
        "GarageCond",
    ],
    "外部设施": [
        "PavedDrive",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "PoolArea",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "MiscVal",
    ],
    "时间与交易": [
        "YearBuilt",
        "YearRemodAdd",
        "MoSold",
        "YrSold",
        "SaleType",
        "SaleCondition",
    ],
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, sample_submission = load_raw_data(DEFAULT_DATA_DIR)

    _dataset_profile(train, test, sample_submission)
    _feature_type_summary(train)
    _feature_domain_groups(train)
    _missing_overview(train, test)
    _target_summary(train)
    _preprocessing_feature_summary(train, test)


def _dataset_profile(train: pd.DataFrame, test: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    rows = [
        _profile_row("train", train, has_target=True),
        _profile_row("test", test, has_target=False),
        {
            "dataset": "sample_submission",
            "rows": len(sample_submission),
            "columns": sample_submission.shape[1],
            "feature_columns": 0,
            "has_saleprice": "yes",
            "missing_cells": int(sample_submission.isna().sum().sum()),
            "duplicate_rows": int(sample_submission.duplicated().sum()),
        },
    ]
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "dataset_profile.csv", index=False, encoding="utf-8-sig")


def _profile_row(name: str, df: pd.DataFrame, has_target: bool) -> dict[str, object]:
    return {
        "dataset": name,
        "rows": len(df),
        "columns": df.shape[1],
        "feature_columns": df.shape[1] - int(has_target),
        "has_saleprice": "yes" if has_target else "no",
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def _feature_type_summary(train: pd.DataFrame) -> None:
    features = train.drop(columns=["SalePrice"])
    rows = []
    for dtype, count in features.dtypes.astype(str).value_counts().sort_index().items():
        rows.append({"dtype": dtype, "feature_count": int(count)})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "feature_type_summary.csv", index=False, encoding="utf-8-sig")


def _feature_domain_groups(train: pd.DataFrame) -> None:
    feature_set = set(train.columns) - {"Id", "SalePrice"}
    rows = []
    covered: set[str] = set()
    for group, columns in FEATURE_GROUPS.items():
        existing = [column for column in columns if column in feature_set]
        covered.update(existing)
        rows.append(
            {
                "feature_group": group,
                "feature_count": len(existing),
                "example_features": ", ".join(existing[:8]),
            }
        )
    uncovered = sorted(feature_set - covered)
    if uncovered:
        rows.append(
            {
                "feature_group": "其他",
                "feature_count": len(uncovered),
                "example_features": ", ".join(uncovered[:8]),
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "feature_domain_groups.csv", index=False, encoding="utf-8-sig")


def _missing_overview(train: pd.DataFrame, test: pd.DataFrame) -> None:
    rows = []
    for name, df in [("train", train), ("test", test)]:
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        rows.append(
            {
                "dataset": name,
                "features_with_missing": int((df.isna().sum() > 0).sum()),
                "missing_cells": int(df.isna().sum().sum()),
                "missing_cell_pct": round(float(df.isna().sum().sum() / df.size * 100), 4),
                "top_missing_features": "; ".join(
                    f"{feature}:{int(count)}" for feature, count in missing.head(8).items()
                ),
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "missing_overview.csv", index=False, encoding="utf-8-sig")


def _target_summary(train: pd.DataFrame) -> None:
    target = train["SalePrice"]
    log_target = np.log1p(target)
    rows = [
        {
            "variable": "SalePrice",
            "mean": target.mean(),
            "std": target.std(),
            "min": target.min(),
            "p25": target.quantile(0.25),
            "median": target.median(),
            "p75": target.quantile(0.75),
            "p95": target.quantile(0.95),
            "max": target.max(),
            "skew": target.skew(),
        },
        {
            "variable": "log1p(SalePrice)",
            "mean": log_target.mean(),
            "std": log_target.std(),
            "min": log_target.min(),
            "p25": log_target.quantile(0.25),
            "median": log_target.median(),
            "p75": log_target.quantile(0.75),
            "p95": log_target.quantile(0.95),
            "max": log_target.max(),
            "skew": pd.Series(log_target).skew(),
        },
    ]
    pd.DataFrame(rows).round(4).to_csv(OUTPUT_DIR / "target_summary_for_report.csv", index=False, encoding="utf-8-sig")


def _preprocessing_feature_summary(train: pd.DataFrame, test: pd.DataFrame) -> None:
    features, _, _ = build_advanced_feature_matrix(train, test)
    train_clean = remove_known_outliers(train)
    clean_features, _, _ = build_advanced_feature_matrix(train_clean, test)
    rows = [
        {"item": "raw_train_rows", "value": len(train), "note": "原始训练样本数"},
        {"item": "raw_test_rows", "value": len(test), "note": "原始测试样本数"},
        {"item": "raw_feature_columns", "value": train.shape[1] - 1, "note": "不含 SalePrice"},
        {"item": "outlier_removed_train_rows", "value": len(train_clean), "note": "删除两个经典异常点后的训练样本数"},
        {"item": "advanced_raw_feature_columns", "value": features.train.shape[1], "note": "不删异常点时高级预处理后的特征数"},
        {"item": "advanced_clean_train_rows", "value": clean_features.train.shape[0], "note": "正式建模使用的训练样本数"},
        {"item": "advanced_clean_test_rows", "value": clean_features.test.shape[0], "note": "正式建模使用的测试样本数"},
        {"item": "advanced_clean_feature_columns", "value": clean_features.train.shape[1], "note": "正式建模使用的高级预处理特征数"},
        {
            "item": "skew_transformed_columns",
            "value": len(clean_features.transformed_columns),
            "note": "执行 log1p 偏态修正的数值列数",
        },
        {
            "item": "missing_indicator_columns",
            "value": sum(name.endswith("_WasMissing") for name in clean_features.feature_names),
            "note": "缺失指示器特征数",
        },
    ]
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "preprocessing_feature_summary.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"transformed_column": clean_features.transformed_columns}).to_csv(
        OUTPUT_DIR / "skew_transformed_columns.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
