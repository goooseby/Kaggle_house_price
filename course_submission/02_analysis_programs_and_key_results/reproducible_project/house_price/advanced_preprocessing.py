from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import RobustScaler

from house_price.preprocessing import MODE_COLUMNS, NONE_COLUMNS, ZERO_COLUMNS


@dataclass(frozen=True)
class AdvancedFeatureMatrices:
    train: pd.DataFrame
    test: pd.DataFrame
    transformed_columns: list[str]
    feature_names: list[str]


QUALITY_MAP = {
    "Ex": 5,
    "Gd": 4,
    "TA": 3,
    "Fa": 2,
    "Po": 1,
    "None": 0,
}

ORDINAL_MAPS = {
    "ExterQual": QUALITY_MAP,
    "ExterCond": QUALITY_MAP,
    "BsmtQual": QUALITY_MAP,
    "BsmtCond": QUALITY_MAP,
    "HeatingQC": QUALITY_MAP,
    "KitchenQual": QUALITY_MAP,
    "FireplaceQu": QUALITY_MAP,
    "GarageQual": QUALITY_MAP,
    "GarageCond": QUALITY_MAP,
    "PoolQC": QUALITY_MAP,
    "LotShape": {"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1},
    "LandSlope": {"Gtl": 3, "Mod": 2, "Sev": 1},
    "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0},
    "BsmtFinType1": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0},
    "BsmtFinType2": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0},
    "Functional": {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0},
    "GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0},
    "PavedDrive": {"Y": 3, "P": 2, "N": 1},
}

CATEGORICAL_NUMERIC_COLUMNS = ["MSSubClass", "MoSold", "YrSold"]
YEAR_COLUMNS = {"YearBuilt", "YearRemodAdd", "GarageYrBlt"}


def build_advanced_feature_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[AdvancedFeatureMatrices, pd.Series, pd.Series]:
    target = np.log1p(train["SalePrice"])
    test_ids = test["Id"].copy()

    train_features = train.drop(columns=["SalePrice"])
    all_features = pd.concat([train_features, test], axis=0, ignore_index=True)

    all_features = _add_missing_indicators(all_features)
    all_features = _fill_missing_values(all_features)
    all_features = _add_features(all_features)
    all_features = _fix_categorical_types(all_features)
    all_features = _encode_ordinals(all_features)
    all_features, transformed_columns = _transform_skewed_numeric_features(all_features)

    all_features = all_features.drop(columns=["Id"])
    all_features = pd.get_dummies(all_features, drop_first=False)
    all_features = all_features.replace([np.inf, -np.inf], np.nan)
    all_features = all_features.fillna(all_features.median(numeric_only=True))

    numeric_columns = all_features.select_dtypes(include="number").columns
    all_features = all_features.astype({column: "float64" for column in numeric_columns})
    scaler = RobustScaler()
    all_features[numeric_columns] = pd.DataFrame(
        scaler.fit_transform(all_features[numeric_columns]),
        columns=numeric_columns,
        index=all_features.index,
    )

    train_matrix = all_features.iloc[: len(train)].copy()
    test_matrix = all_features.iloc[len(train) :].copy()

    return (
        AdvancedFeatureMatrices(
            train=train_matrix,
            test=test_matrix,
            transformed_columns=transformed_columns,
            feature_names=all_features.columns.tolist(),
        ),
        target,
        test_ids,
    )


def _add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.columns:
        missing_count = df[column].isna().sum()
        if missing_count > 0:
            df[f"{column}_WasMissing"] = df[column].isna().astype(int)
    return df


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
    df["TotalLivingSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["TotalFinishedSF"] = df["BsmtFinSF1"] + df["BsmtFinSF2"] + df["1stFlrSF"] + df["2ndFlrSF"]
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
    df["GarageAge"] = np.where(df["GarageYrBlt"] > 0, df["YrSold"] - df["GarageYrBlt"], 0)
    df["GarageAge"] = pd.Series(df["GarageAge"]).clip(lower=0)

    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["IsNewHouse"] = (df["YrSold"] == df["YearBuilt"]).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasPorch"] = (df["TotalPorchSF"] > 0).astype(int)

    df["FinishedBasementRatio"] = _safe_divide(df["BsmtFinSF1"] + df["BsmtFinSF2"], df["TotalBsmtSF"])
    df["GarageAreaPerCar"] = _safe_divide(df["GarageArea"], df["GarageCars"])
    df["SecondFloorRatio"] = _safe_divide(df["2ndFlrSF"], df["TotalSF"])
    df["PorchRatio"] = _safe_divide(df["TotalPorchSF"], df["TotalSF"])
    df["LotAreaPerLivingSF"] = _safe_divide(df["LotArea"], df["GrLivArea"])

    df["OverallQual_TotalSF"] = df["OverallQual"] * df["TotalSF"]
    df["OverallQual_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]
    df["OverallQual_GarageCars"] = df["OverallQual"] * df["GarageCars"]
    df["OverallQual_TotalBathrooms"] = df["OverallQual"] * df["TotalBathrooms"]
    df["OverallQual_YearBuilt"] = df["OverallQual"] * df["YearBuilt"]
    df["Age_OverallQual"] = df["HouseAge"] * df["OverallQual"]
    df["Neighborhood_OverallQual"] = df["Neighborhood"].astype(str) + "_" + df["OverallQual"].astype(str)

    return df


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def _fix_categorical_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in CATEGORICAL_NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype(str)
    return df


def _encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column, mapping in ORDINAL_MAPS.items():
        if column in df.columns:
            df[column] = df[column].map(mapping).fillna(0)
    return df


def _transform_skewed_numeric_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    transformed_columns: list[str] = []

    numeric_columns = df.select_dtypes(include="number").columns
    for column in numeric_columns:
        if column in YEAR_COLUMNS or column.endswith("_WasMissing"):
            continue
        if df[column].min() < 0 or df[column].nunique() <= 10:
            continue
        column_skew = skew(df[column].dropna())
        if column_skew > 0.75:
            df[column] = np.log1p(df[column])
            transformed_columns.append(column)

    return df, transformed_columns
