from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.svm import SVR

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from house_price.config import RANDOM_STATE
from house_price.advanced_preprocessing import AdvancedFeatureMatrices


@dataclass(frozen=True)
class OptimizationResults:
    cv_results: pd.DataFrame
    blend_results: pd.DataFrame
    submission_paths: dict[str, Path]
    report_path: Path
    best_submission_name: str
    best_blend_score: float


def train_optimized_models(
    features: AdvancedFeatureMatrices,
    target: pd.Series,
    test_ids: pd.Series,
    sample_submission: pd.DataFrame,
    output_dir: Path,
    submissions_dir: Path,
    reports_dir: Path,
) -> OptimizationResults:
    output_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    models = _build_models()

    oof_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    rows = []

    for name, model in models.items():
        oof, test_pred = _fit_oof_and_predict(model, features.train, target, features.test, cv)
        score = _rmse(target, oof)
        oof_predictions[name] = oof
        test_predictions[name] = test_pred
        rows.append({"model": name, "cv_rmse": score})
        print(f"{name}: {score:.5f}")

    cv_results = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
    cv_results_path = output_dir / "optimized_cv_results.csv"
    cv_results.to_csv(cv_results_path, index=False)

    selected_model_names = _select_models_for_blending(cv_results)
    blend_results, blend_predictions = _build_blends(
        selected_model_names,
        oof_predictions,
        test_predictions,
        target,
    )
    blend_results_path = output_dir / "optimized_blend_results.csv"
    blend_results.to_csv(blend_results_path, index=False)

    submission_paths = _write_submissions(
        sample_submission=sample_submission,
        test_ids=test_ids,
        blend_predictions=blend_predictions,
        target=target,
        submissions_dir=submissions_dir,
    )

    _write_prediction_diagnostics(
        oof_predictions=oof_predictions,
        test_predictions=test_predictions,
        target=target,
        output_dir=output_dir,
    )

    recommended_blend = blend_results.loc[
        blend_results["blend"].isin(["optimized_weight_blend", "inverse_cv_blend", "simple_blend"])
    ].iloc[0]

    report_path = _write_report(
        cv_results=cv_results,
        blend_results=blend_results,
        selected_model_names=selected_model_names,
        transformed_columns=features.transformed_columns,
        submission_paths=submission_paths,
        recommended_blend_name=str(recommended_blend["blend"]),
        report_path=reports_dir / "20260506_optimization_round_report.md",
    )

    return OptimizationResults(
        cv_results=cv_results,
        blend_results=blend_results,
        submission_paths=submission_paths,
        report_path=report_path,
        best_submission_name=str(recommended_blend["blend"]),
        best_blend_score=float(recommended_blend["cv_rmse"]),
    )


def _build_models() -> dict[str, object]:
    return {
        "lasso": Lasso(alpha=0.00035, max_iter=50000, random_state=RANDOM_STATE),
        "elastic_net": ElasticNet(
            alpha=0.00045,
            l1_ratio=0.88,
            max_iter=50000,
            random_state=RANDOM_STATE,
        ),
        "ridge": Ridge(alpha=12.0),
        "kernel_ridge": KernelRidge(alpha=0.38, kernel="polynomial", degree=2, coef0=2.5),
        "svr": SVR(C=18.0, epsilon=0.008, gamma=0.00035),
        "gbr": GradientBoostingRegressor(
            n_estimators=1800,
            learning_rate=0.018,
            max_depth=3,
            min_samples_leaf=12,
            min_samples_split=18,
            subsample=0.82,
            loss="squared_error",
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBRegressor(
            n_estimators=2200,
            learning_rate=0.018,
            max_depth=3,
            min_child_weight=2,
            subsample=0.82,
            colsample_bytree=0.72,
            reg_alpha=0.0004,
            reg_lambda=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=1800,
            learning_rate=0.018,
            num_leaves=12,
            max_depth=3,
            min_child_samples=12,
            subsample=0.82,
            subsample_freq=1,
            colsample_bytree=0.72,
            reg_alpha=0.0004,
            reg_lambda=0.15,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
        ),
        "catboost": CatBoostRegressor(
            iterations=1800,
            learning_rate=0.018,
            depth=4,
            l2_leaf_reg=4.0,
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        ),
    }


def _fit_oof_and_predict(
    model: object,
    x: pd.DataFrame,
    y: pd.Series,
    x_test: pd.DataFrame,
    cv: KFold,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(x))
    test_fold_predictions = np.zeros((len(x_test), cv.get_n_splits()))

    for fold_index, (train_idx, valid_idx) in enumerate(cv.split(x)):
        fold_model = clone(model)
        x_train = x.iloc[train_idx]
        x_valid = x.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        fold_model.fit(x_train, y_train)
        oof[valid_idx] = fold_model.predict(x_valid)
        test_fold_predictions[:, fold_index] = fold_model.predict(x_test)

    return oof, test_fold_predictions.mean(axis=1)


def _select_models_for_blending(cv_results: pd.DataFrame) -> list[str]:
    cutoff = cv_results["cv_rmse"].min() + 0.006
    selected = cv_results.loc[cv_results["cv_rmse"] <= cutoff, "model"].tolist()
    if len(selected) < 4:
        selected = cv_results["model"].head(4).tolist()
    return selected[:7]


def _build_blends(
    model_names: list[str],
    oof_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    target: pd.Series,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    oof_matrix = np.column_stack([oof_predictions[name] for name in model_names])
    test_matrix = np.column_stack([test_predictions[name] for name in model_names])

    simple_oof = oof_matrix.mean(axis=1)
    simple_test = test_matrix.mean(axis=1)

    inverse_weights = _inverse_score_weights(model_names, oof_predictions, target)
    inverse_oof = oof_matrix @ inverse_weights
    inverse_test = test_matrix @ inverse_weights

    optimized_weights = _optimize_weights(oof_matrix, target.to_numpy())
    optimized_oof = oof_matrix @ optimized_weights
    optimized_test = test_matrix @ optimized_weights

    stacking_model = RidgeCV(alphas=np.logspace(-4, 2, 40), fit_intercept=True)
    stacking_model.fit(oof_matrix, target)
    stacking_oof = stacking_model.predict(oof_matrix)
    stacking_test = stacking_model.predict(test_matrix)

    blend_predictions = {
        "simple_blend": simple_test,
        "inverse_cv_blend": inverse_test,
        "optimized_weight_blend": optimized_test,
        "ridge_stack": stacking_test,
    }

    rows = [
        {
            "blend": "simple_blend",
            "cv_rmse": _rmse(target, simple_oof),
            "models": ",".join(model_names),
            "weights": _format_weights(model_names, np.repeat(1 / len(model_names), len(model_names))),
        },
        {
            "blend": "inverse_cv_blend",
            "cv_rmse": _rmse(target, inverse_oof),
            "models": ",".join(model_names),
            "weights": _format_weights(model_names, inverse_weights),
        },
        {
            "blend": "optimized_weight_blend",
            "cv_rmse": _rmse(target, optimized_oof),
            "models": ",".join(model_names),
            "weights": _format_weights(model_names, optimized_weights),
        },
        {
            "blend": "ridge_stack",
            "cv_rmse": _rmse(target, stacking_oof),
            "models": ",".join(model_names),
            "weights": _format_weights(model_names, stacking_model.coef_),
        },
    ]

    return pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True), blend_predictions


def _inverse_score_weights(
    model_names: list[str],
    oof_predictions: dict[str, np.ndarray],
    target: pd.Series,
) -> np.ndarray:
    scores = np.array([_rmse(target, oof_predictions[name]) for name in model_names])
    weights = 1 / np.power(scores, 4)
    return weights / weights.sum()


def _optimize_weights(oof_matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    n_models = oof_matrix.shape[1]
    initial_weights = np.repeat(1 / n_models, n_models)
    bounds = [(0.0, 1.0)] * n_models
    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}]

    result = minimize(
        lambda weights: mean_squared_error(target, oof_matrix @ weights) ** 0.5,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        return initial_weights
    return result.x / result.x.sum()


def _write_submissions(
    sample_submission: pd.DataFrame,
    test_ids: pd.Series,
    blend_predictions: dict[str, np.ndarray],
    target: pd.Series,
    submissions_dir: Path,
) -> dict[str, Path]:
    submission_paths: dict[str, Path] = {}
    low_clip = float(np.expm1(target).quantile(0.001))
    high_clip = float(np.expm1(target).quantile(0.997))

    for name, log_predictions in blend_predictions.items():
        predictions = np.expm1(log_predictions).clip(min=0)
        submission_paths[name] = _write_submission(
            sample_submission,
            test_ids,
            predictions,
            submissions_dir / f"20260506_{name}.csv",
        )

        clipped_predictions = predictions.clip(low_clip, high_clip)
        submission_paths[f"{name}_clipped"] = _write_submission(
            sample_submission,
            test_ids,
            clipped_predictions,
            submissions_dir / f"20260506_{name}_clipped.csv",
        )

    return submission_paths


def _write_submission(
    sample_submission: pd.DataFrame,
    test_ids: pd.Series,
    predictions: np.ndarray,
    path: Path,
) -> Path:
    submission = sample_submission.copy()
    submission["Id"] = test_ids.values
    submission["SalePrice"] = predictions
    submission.to_csv(path, index=False)
    return path


def _write_prediction_diagnostics(
    oof_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    target: pd.Series,
    output_dir: Path,
) -> None:
    diagnostics = []
    for name, oof in oof_predictions.items():
        residuals = target.to_numpy() - oof
        diagnostics.append(
            {
                "model": name,
                "cv_rmse": _rmse(target, oof),
                "mean_abs_error_log": float(np.mean(np.abs(residuals))),
                "max_abs_error_log": float(np.max(np.abs(residuals))),
                "test_min_price": float(np.expm1(test_predictions[name]).min()),
                "test_max_price": float(np.expm1(test_predictions[name]).max()),
            }
        )
    pd.DataFrame(diagnostics).sort_values("cv_rmse").to_csv(
        output_dir / "optimized_prediction_diagnostics.csv",
        index=False,
    )


def _write_report(
    cv_results: pd.DataFrame,
    blend_results: pd.DataFrame,
    selected_model_names: list[str],
    transformed_columns: list[str],
    submission_paths: dict[str, Path],
    recommended_blend_name: str,
    report_path: Path,
) -> Path:
    lines = [
        "# 20260506 完整优化回合报告",
        "",
        "## 摘要",
        "",
        f"- 进入核心融合池的模型：{', '.join(selected_model_names)}",
        f"- 本地 CV 最低的融合方案：{blend_results.iloc[0]['blend']} ({blend_results.iloc[0]['cv_rmse']:.5f})",
        f"- 推荐第一提交：{recommended_blend_name}",
        f"- 做了 log1p 变换的数值特征数：{len(transformed_columns)}",
        "- 注意：ridge_stack 适合作为诊断参考，但它的显示 CV 偏乐观，因为二层模型是在同一份 OOF 预测矩阵上训练并评分。",
        "",
        "## 单模型 CV",
        "",
        _to_markdown(cv_results),
        "",
        "## 融合方案 CV",
        "",
        _to_markdown(blend_results),
        "",
        "## 做了 log1p 变换的特征",
        "",
        ", ".join(transformed_columns) if transformed_columns else "_None_",
        "",
        "## 候选提交文件",
        "",
        _to_markdown(
            pd.DataFrame(
                [{"candidate": name, "path": str(path)} for name, path in submission_paths.items()]
            )
        ),
        "",
        "## 提交建议",
        "",
        "1. `submissions/20260506_optimized_weight_blend.csv`",
        "2. `submissions/20260506_simple_blend.csv`",
        "3. `submissions/20260506_optimized_weight_blend_clipped.csv`",
        "4. `submissions/20260506_ridge_stack.csv`",
        "",
        "理由：",
        "",
        "- `optimized_weight_blend` 是最好的非负权重 OOF 融合，优先提交。",
        "- `simple_blend` 本地略弱，但结构简单，通常比较稳。",
        "- `optimized_weight_blend_clipped` 用于测试压住极端高价预测是否有帮助。",
        "- `ridge_stack` 本地分数最低，但偏乐观风险更高，建议放在稳健候选之后。",
        "",
        "## Kaggle 反馈",
        "",
        "提交后在这里补充 Public Score 和排名变化。",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    table = df.copy()
    headers = [str(column) for column in table.columns]
    rows = [[str(value) for value in row] for row in table.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _format_weights(model_names: list[str], weights: np.ndarray) -> str:
    return "; ".join(f"{name}:{weight:.4f}" for name, weight in zip(model_names, weights))


def _rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)
