from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import clone

from house_price.config import RANDOM_STATE
from house_price.preprocessing import FeatureMatrices


@dataclass(frozen=True)
class TrainingResults:
    best_model_name: str
    best_score: float
    blend_score: float
    submission_path: Path
    cv_results_path: Path


def train_and_predict(
    features: FeatureMatrices,
    target: pd.Series,
    test_ids: pd.Series,
    sample_submission: pd.DataFrame,
    output_dir: Path,
) -> TrainingResults:
    output_dir.mkdir(parents=True, exist_ok=True)

    models = _build_models()
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scores = []
    for name, model in models.items():
        rmse = _cross_val_rmse(model, features.train, target, cv)
        scores.append({"model": name, "cv_rmse": rmse})

    cv_results = pd.DataFrame(scores).sort_values("cv_rmse")
    cv_results_path = output_dir / "cv_results.csv"
    cv_results.to_csv(cv_results_path, index=False)

    blend_model_names = [name for name in ["ridge", "lasso", "elastic_net", "gbr"] if name in models]
    blend_oof = _out_of_fold_blend(models, blend_model_names, features.train, target, cv)
    blend_score = mean_squared_error(target, blend_oof) ** 0.5

    predictions = []
    for name in blend_model_names:
        model = clone(models[name])
        model.fit(features.train, target)
        predictions.append(model.predict(features.test))

    blended_log_predictions = np.mean(predictions, axis=0)
    sale_price_predictions = np.expm1(blended_log_predictions).clip(min=0)

    submission = sample_submission.copy()
    submission["Id"] = test_ids.values
    submission["SalePrice"] = sale_price_predictions

    submission_path = output_dir / "submission_baseline.csv"
    submission.to_csv(submission_path, index=False)

    best = cv_results.iloc[0]
    return TrainingResults(
        best_model_name=str(best["model"]),
        best_score=float(best["cv_rmse"]),
        blend_score=float(blend_score),
        submission_path=submission_path,
        cv_results_path=cv_results_path,
    )


def _build_models() -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=15.0),
        "lasso": Lasso(alpha=0.0005, max_iter=20000, random_state=RANDOM_STATE),
        "elastic_net": ElasticNet(
            alpha=0.0005,
            l1_ratio=0.9,
            max_iter=20000,
            random_state=RANDOM_STATE,
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=12,
            min_samples_split=20,
            loss="squared_error",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
    }


def _cross_val_rmse(model: object, x: pd.DataFrame, y: pd.Series, cv: KFold) -> float:
    scores = cross_val_score(
        clone(model),
        x,
        y,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=1,
    )
    return float(-scores.mean())


def _out_of_fold_blend(
    models: dict[str, object],
    model_names: list[str],
    x: pd.DataFrame,
    y: pd.Series,
    cv: KFold,
) -> np.ndarray:
    fold_predictions = np.zeros((len(x), len(model_names)))

    for fold_train_idx, fold_valid_idx in cv.split(x):
        x_train = x.iloc[fold_train_idx]
        x_valid = x.iloc[fold_valid_idx]
        y_train = y.iloc[fold_train_idx]

        for model_index, name in enumerate(model_names):
            model = clone(models[name])
            model.fit(x_train, y_train)
            fold_predictions[fold_valid_idx, model_index] = model.predict(x_valid)

    return fold_predictions.mean(axis=1)
