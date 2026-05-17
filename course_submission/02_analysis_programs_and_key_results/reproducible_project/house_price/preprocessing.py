from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


@dataclass(frozen=True)
class FeatureMatrices:
    train: pd.DataFrame
    test: pd.DataFrame


NONE_COLUMNS = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType",
]

ZERO_COLUMNS = [
    "GarageYrBlt",
    "GarageArea",
    "GarageCars",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "MasVnrArea",
]

MODE_COLUMNS = [
    "MSZoning",
    "Electrical",
    "KitchenQual",
    "Exterior1st",
    "Exterior2nd",
    "SaleType",
    "Utilities",
]

ORDINAL_MAP = {
    "Ex": 5,
    "Gd": 4,
    "TA": 3,
    "Fa": 2,
    "Po": 1,
    "None": 0,
}

ORDINAL_COLUMNS = [
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",
    "GarageQual",
    "GarageCond",
    "PoolQC",
]


def build_feature_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[FeatureMatrices, pd.Series, pd.Series]:
    target = np.log1p(train["SalePrice"])
    test_ids = test["Id"].copy()

    train_features = train.drop(columns=["SalePrice"])
    all_features = pd.concat([train_features, test], axis=0, ignore_index=True)

    all_features = _fill_missing_values(all_features)
    all_features = _add_features(all_features)
    all_features = _fix_categorical_types(all_features)
    all_features = _encode_ordered_quality(all_features)

    all_features = all_features.drop(columns=["Id"])
    all_features = pd.get_dummies(all_features, drop_first=False)

    all_features = all_features.replace([np.inf, -np.inf], np.nan)
    all_features = all_features.fillna(all_features.median(numeric_only=True))

    numeric_columns = all_features.select_dtypes(include="number").columns
    all_features = all_features.astype({column: "float64" for column in numeric_columns})
    scaler = RobustScaler()
    scaled_values = scaler.fit_transform(all_features[numeric_columns])
    all_features[numeric_columns] = pd.DataFrame(
        scaled_values,
        columns=numeric_columns,
        index=all_features.index,
    )

    train_matrix = all_features.iloc[: len(train)].copy()
    test_matrix = all_features.iloc[len(train) :].copy()
    return FeatureMatrices(train=train_matrix, test=test_matrix), target, test_ids


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda s: s.fillna(s.median())
        )

    for column in NONE_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("None")

    for column in ZERO_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna(0)

    for column in MODE_COLUMNS:
        if column in df.columns:
            mode = df[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "None"
            df[column] = df[column].fillna(fill_value)

    if "Functional" in df.columns:
        df["Functional"] = df["Functional"].fillna("Typ")

    for column in df.select_dtypes(include="number").columns:
        df[column] = df[column].fillna(df[column].median())

    for column in df.select_dtypes(include="object").columns:
        mode = df[column].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "None"
        df[column] = df[column].fillna(fill_value)

    return df


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBathrooms"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )
    df["TotalPorchSF"] = (
        df["OpenPorchSF"]
        + df["EnclosedPorch"]
        + df["3SsnPorch"]
        + df["ScreenPorch"]
        + df["WoodDeckSF"]
    )
    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).clip(lower=0)
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)

    return df


def _fix_categorical_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["MSSubClass", "MoSold", "YrSold"]:
        if column in df.columns:
            df[column] = df[column].astype(str)
    return df


def _encode_ordered_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ORDINAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(ORDINAL_MAP).fillna(0)
    return df
