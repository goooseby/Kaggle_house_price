from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetEncodingResult:
    train: pd.DataFrame
    test: pd.DataFrame
    feature_specs: list[tuple[str, ...]]


DEFAULT_TARGET_ENCODING_SPECS = [
    ("Neighborhood",),
    ("MSSubClass",),
    ("HouseStyle",),
    ("Exterior1st",),
    ("Exterior2nd",),
    ("SaleType",),
    ("SaleCondition",),
    ("Condition1",),
    ("Foundation",),
    ("GarageType",),
    ("RoofStyle",),
    ("BldgType",),
    ("Neighborhood", "OverallQual"),
    ("Neighborhood", "MSSubClass"),
    ("Neighborhood", "HouseStyle"),
    ("OverallQual", "MSSubClass"),
    ("OverallQual", "GarageCars"),
    ("OverallQual", "ExterQual"),
    ("OverallQual", "KitchenQual"),
    ("SaleCondition", "SaleType"),
]


def build_oof_target_encoding_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_log: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_specs: list[tuple[str, ...]] | None = None,
    smoothing: float = 12.0,
    min_samples_leaf: int = 1,
    scale: bool = True,
) -> TargetEncodingResult:
    """Create leakage-safe OOF target encoding features.

    Target is expected to be log1p(SalePrice). Train encodings are computed fold by
    fold from the fold training split only; test encodings use full train statistics.
    """
    feature_specs = feature_specs or DEFAULT_TARGET_ENCODING_SPECS
    target_log = pd.Series(target_log).reset_index(drop=True)
    train_source = train.reset_index(drop=True).copy()
    test_source = test.reset_index(drop=True).copy()
    global_mean = float(target_log.mean())

    train_encoded = pd.DataFrame(index=train_source.index)
    test_encoded = pd.DataFrame(index=test_source.index)

    for spec in feature_specs:
        existing_spec = tuple(column for column in spec if column in train_source.columns)
        if len(existing_spec) != len(spec):
            continue

        feature_name = _encoded_feature_name(existing_spec)
        train_key = _make_key(train_source, existing_spec)
        test_key = _make_key(test_source, existing_spec)

        encoded_train = pd.Series(global_mean, index=train_source.index, dtype=float)
        for fold_train_idx, fold_valid_idx in folds:
            stats = _smoothed_target_stats(
                train_key.iloc[fold_train_idx],
                target_log.iloc[fold_train_idx],
                global_mean,
                smoothing=smoothing,
                min_samples_leaf=min_samples_leaf,
            )
            encoded_train.iloc[fold_valid_idx] = (
                train_key.iloc[fold_valid_idx].map(stats).fillna(global_mean).astype(float)
            )

        full_stats = _smoothed_target_stats(
            train_key,
            target_log,
            global_mean,
            smoothing=smoothing,
            min_samples_leaf=min_samples_leaf,
        )
        encoded_test = test_key.map(full_stats).fillna(global_mean).astype(float)

        train_encoded[feature_name] = encoded_train
        test_encoded[feature_name] = encoded_test

    if scale and not train_encoded.empty:
        all_values = pd.concat([train_encoded, test_encoded], axis=0, ignore_index=True)
        scaled = _safe_robust_scale(all_values)
        train_encoded = scaled.iloc[: len(train_encoded)].reset_index(drop=True)
        test_encoded = scaled.iloc[len(train_encoded) :].reset_index(drop=True)

    return TargetEncodingResult(
        train=train_encoded,
        test=test_encoded,
        feature_specs=feature_specs,
    )


def _smoothed_target_stats(
    key: pd.Series,
    target_log: pd.Series,
    global_mean: float,
    smoothing: float,
    min_samples_leaf: int,
) -> pd.Series:
    data = pd.DataFrame({"key": key, "target": target_log})
    stats = data.groupby("key")["target"].agg(["mean", "count"])
    smoothing_factor = 1 / (1 + np.exp(-(stats["count"] - min_samples_leaf) / smoothing))
    encoded = global_mean * (1 - smoothing_factor) + stats["mean"] * smoothing_factor
    return encoded


def _make_key(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    values = []
    for column in columns:
        values.append(df[column].fillna("None").astype(str))
    key = values[0]
    for value in values[1:]:
        key = key + "__" + value
    return key


def _encoded_feature_name(columns: tuple[str, ...]) -> str:
    return "TE_" + "__".join(columns)


def _safe_robust_scale(values: pd.DataFrame) -> pd.DataFrame:
    median = values.median(axis=0)
    iqr = values.quantile(0.75, axis=0) - values.quantile(0.25, axis=0)
    safe_iqr = iqr.where(iqr.abs() > 0.05, 0.05)
    return (values - median) / safe_iqr
