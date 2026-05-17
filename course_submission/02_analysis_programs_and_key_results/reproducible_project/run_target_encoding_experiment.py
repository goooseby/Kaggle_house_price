from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from house_price.advanced_preprocessing import build_advanced_feature_matrix
from house_price.config import DEFAULT_DATA_DIR, PROJECT_ROOT, RANDOM_STATE
from house_price.data import load_raw_data, remove_known_outliers
from house_price.target_encoding import build_oof_target_encoding_features


EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "model_redesign_20260507" / "target_encoding"
SUBMISSION_DIR = PROJECT_ROOT / "submissions" / "model_redesign_20260507" / "target_encoding"
REPORT_PATH = PROJECT_ROOT / "reports" / "modeling" / "20260507_target_encoding_experiment_report.md"
EXPERIMENT_README_PATH = EXPERIMENT_DIR / "README.md"
SUBMISSION_README_PATH = SUBMISSION_DIR / "README.md"


def main() -> None:
    train, test, sample_submission = load_raw_data(DEFAULT_DATA_DIR)
    train_clean = remove_known_outliers(train)
    train_sale_price = train_clean["SalePrice"].reset_index(drop=True)

    features, target_log, test_ids = build_advanced_feature_matrix(train_clean, test)
    folds = _make_stratified_folds(target_log)
    te_features = build_oof_target_encoding_features(train_clean.drop(columns=["SalePrice"]), test, target_log, folds)

    x_train = pd.concat([features.train.reset_index(drop=True), te_features.train], axis=1)
    x_test = pd.concat([features.test.reset_index(drop=True), te_features.test], axis=1)

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    models = _build_models()
    model_rows = []
    oof_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}

    for name, model in models.items():
        print(f"Training {name}...")
        oof, test_pred = _fit_oof(model, x_train, target_log, x_test, folds)
        oof_predictions[name] = oof
        test_predictions[name] = test_pred
        row = {"candidate": name, "kind": "single_model", "cv_rmse": _rmse(target_log, oof)}
        model_rows.append(row)
        print(f"{name}: cv={row['cv_rmse']:.5f}")

    blend_rows, blend_predictions, blend_oof = _build_blends(
        oof_predictions,
        test_predictions,
        target_log,
        train_sale_price,
    )
    result_table = pd.DataFrame(model_rows + blend_rows).sort_values("cv_rmse")
    result_table.to_csv(EXPERIMENT_DIR / "model_cv_results.csv", index=False, encoding="utf-8-sig")

    _write_prediction_files(oof_predictions, test_predictions, blend_oof, blend_predictions)
    submission_paths = _write_submissions(
        blend_predictions,
        sample_submission,
        test_ids,
        train_sale_price,
    )
    _write_report(result_table, submission_paths, te_features.train.columns.tolist())
    _write_navigation_readmes(submission_paths)

    print(f"CV results: {EXPERIMENT_DIR / 'model_cv_results.csv'}")
    print(f"Report: {REPORT_PATH}")
    print("Candidate submissions:")
    for name, path in submission_paths.items():
        print(f"- {name}: {path}")


def _make_stratified_folds(target_log: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    bins = pd.qcut(target_log, q=10, labels=False, duplicates="drop")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(np.zeros(len(target_log)), bins))


def _build_models() -> dict[str, object]:
    return {
        "te_ridge": Ridge(alpha=11.0),
        "te_elastic_net": ElasticNet(alpha=0.00038, l1_ratio=0.86, max_iter=60000, random_state=RANDOM_STATE),
        "te_kernel_ridge": KernelRidge(alpha=0.42, kernel="polynomial", degree=2, coef0=2.3),
        "te_svr": SVR(C=16.0, epsilon=0.008, gamma=0.0003),
        "te_xgboost": XGBRegressor(
            n_estimators=1800,
            learning_rate=0.018,
            max_depth=3,
            min_child_weight=2,
            subsample=0.82,
            colsample_bytree=0.72,
            reg_alpha=0.0005,
            reg_lambda=0.9,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "te_lightgbm": LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.018,
            num_leaves=12,
            max_depth=3,
            min_child_samples=12,
            subsample=0.82,
            subsample_freq=1,
            colsample_bytree=0.72,
            reg_alpha=0.0004,
            reg_lambda=0.18,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
        ),
        "te_catboost": CatBoostRegressor(
            iterations=1600,
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


def _fit_oof(
    model: object,
    x_train: pd.DataFrame,
    y: pd.Series,
    x_test: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(x_train))
    test_fold_predictions = np.zeros((len(x_test), len(folds)))
    for fold_index, (train_idx, valid_idx) in enumerate(folds):
        fold_model = clone(model)
        fold_model.fit(x_train.iloc[train_idx], y.iloc[train_idx])
        oof[valid_idx] = fold_model.predict(x_train.iloc[valid_idx])
        test_fold_predictions[:, fold_index] = fold_model.predict(x_test)
    return oof, test_fold_predictions.mean(axis=1)


def _build_blends(
    oof_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    target_log: pd.Series,
    train_sale_price: pd.Series,
) -> tuple[list[dict[str, float | str]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    selected = ["te_elastic_net", "te_ridge", "te_svr", "te_xgboost", "te_catboost"]
    oof_matrix = np.column_stack([oof_predictions[name] for name in selected])
    test_matrix = np.column_stack([test_predictions[name] for name in selected])

    simple_weights = np.repeat(1 / len(selected), len(selected))
    optimized_weights = _optimize_weights(oof_matrix, target_log.to_numpy(), max_weight=0.35)
    conservative_weights = 0.5 * simple_weights + 0.5 * optimized_weights

    blend_specs = {
        "te_simple_blend": simple_weights,
        "te_weighted_blend": optimized_weights,
        "te_conservative_blend": conservative_weights,
    }

    rows = []
    blend_predictions = {}
    blend_oof = {}
    for name, weights in blend_specs.items():
        oof = oof_matrix @ weights
        test_pred = test_matrix @ weights
        blend_oof[name] = oof
        blend_predictions[name] = test_pred
        rows.append(
            {
                "candidate": name,
                "kind": "blend",
                "weights": _format_weights(selected, weights),
                "cv_rmse": _rmse(target_log, oof),
            }
        )
    return rows, blend_predictions, blend_oof


def _optimize_weights(oof_matrix: np.ndarray, target: np.ndarray, max_weight: float) -> np.ndarray:
    n_models = oof_matrix.shape[1]
    initial = np.repeat(1 / n_models, n_models)
    bounds = [(0.0, max_weight)] * n_models
    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}]
    result = minimize(
        lambda weights: float(np.mean((target - oof_matrix @ weights) ** 2) ** 0.5),
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        return initial
    return result.x / result.x.sum()


def _write_prediction_files(
    oof_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    blend_oof: dict[str, np.ndarray],
    blend_predictions: dict[str, np.ndarray],
) -> None:
    pd.DataFrame(oof_predictions | blend_oof).to_csv(EXPERIMENT_DIR / "oof_predictions.csv", index=False)
    pd.DataFrame(test_predictions | blend_predictions).to_csv(EXPERIMENT_DIR / "test_log_predictions.csv", index=False)


def _write_submissions(
    blend_predictions: dict[str, np.ndarray],
    sample_submission: pd.DataFrame,
    test_ids: pd.Series,
    train_sale_price: pd.Series,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    q993 = float(train_sale_price.quantile(0.993))
    previous_best_path = PROJECT_ROOT / "submissions" / "calibration_20260507" / "20260507_opt_clip_q993.csv"
    previous_best = pd.read_csv(previous_best_path)["SalePrice"].astype(float) if previous_best_path.exists() else None

    for name, log_predictions in blend_predictions.items():
        price_predictions = pd.Series(np.expm1(log_predictions))
        paths[name] = _write_submission(sample_submission, test_ids, price_predictions, SUBMISSION_DIR / f"20260507_{name}.csv")
        clipped = price_predictions.clip(upper=q993)
        paths[f"{name}_clip_q993"] = _write_submission(
            sample_submission,
            test_ids,
            clipped,
            SUBMISSION_DIR / f"20260507_{name}_clip_q993.csv",
        )
        if previous_best is not None:
            mixed = np.expm1(0.5 * np.log1p(clipped) + 0.5 * np.log1p(previous_best))
            paths[f"{name}_mix_current_best_clip_q993"] = _write_submission(
                sample_submission,
                test_ids,
                pd.Series(mixed).clip(upper=q993),
                SUBMISSION_DIR / f"20260507_{name}_mix_current_best_clip_q993.csv",
            )
    return paths


def _write_submission(sample_submission: pd.DataFrame, test_ids: pd.Series, predictions: pd.Series, path: Path) -> Path:
    submission = sample_submission.copy()
    submission["Id"] = test_ids.values
    submission["SalePrice"] = predictions.values
    submission.to_csv(path, index=False)
    return path


def _write_report(result_table: pd.DataFrame, submission_paths: dict[str, Path], te_columns: list[str]) -> None:
    recommended = [
        "te_conservative_blend_clip_q993",
        "te_weighted_blend_clip_q993",
        "te_conservative_blend_mix_current_best_clip_q993",
    ]
    lines = [
        "# 20260507 Target Encoding 实验报告",
        "",
        "## 1. 实验目的",
        "",
        "本轮正式执行 Target Encoding 模型改造主题。",
        "",
        "目标不是继续微调上一轮 q993 裁剪比例，而是验证：类别变量的目标均值信息是否能给现有高级特征体系带来结构性增量。",
        "",
        "## 2. 本轮完整方案",
        "",
        "### 2.1 基础特征",
        "",
        "沿用 `house_price.advanced_preprocessing.build_advanced_feature_matrix` 生成的高级特征矩阵，包括缺失值处理、偏态数值变换、人工组合特征和独热编码等。",
        "",
        "### 2.2 新增 Target Encoding 特征",
        "",
        f"本轮生成 TE 特征数：`{len(te_columns)}`",
        "",
        "TE 特征包括：",
        "",
        ", ".join(te_columns),
        "",
        "训练集 TE 使用 OOF 方式生成，测试集 TE 使用全量训练统计并做平滑，避免目标泄漏。",
        "",
        "编码后做安全 Robust 缩放：按中位数居中、按 IQR 缩放，但缩放分母下限固定为 0.05 个 log 点，避免低方差类别列被异常放大。",
        "",
        "### 2.3 单模型",
        "",
        "本轮训练 7 个单模型：`te_ridge`、`te_elastic_net`、`te_kernel_ridge`、`te_svr`、`te_xgboost`、`te_lightgbm`、`te_catboost`。",
        "",
        "### 2.4 融合方案",
        "",
        "融合只使用上一轮经验中更稳的模型族：ElasticNet、Ridge、SVR、XGBoost、CatBoost。",
        "",
        "- `te_simple_blend`：五个模型等权融合。",
        "- `te_weighted_blend`：基于 OOF RMSE 优化权重，单模型最高权重限制为 0.35。",
        "- `te_conservative_blend`：50% 等权融合 + 50% 优化权重融合，用来降低权重优化过拟合风险。",
        "",
        "### 2.5 提交文件变体",
        "",
        "每个融合方案会输出三类提交文件：",
        "",
        "- 原始版本：只使用本轮 TE 融合预测。",
        "- `_clip_q993`：按训练集 SalePrice 的 99.3% 分位数做硬裁剪，沿用上一轮已验证更稳的高价控制策略。",
        "- `_mix_current_best_clip_q993`：先做 q993 裁剪，再与当前公开最好文件 `20260507_opt_clip_q993.csv` 做 50/50 log 融合，用于测试 TE 是否提供增量信息。",
        "",
        "## 3. 本地训练记录",
        "",
        "下表记录本轮单模型与融合模型的 OOF CV RMSE。它只用于训练阶段参考，最终优劣仍以 Kaggle Public Score 为准。",
        "",
        "CV 记录：",
        "",
        _markdown(result_table),
        "",
        "## 4. 候选提交文件",
        "",
        _markdown(pd.DataFrame([{"candidate": name, "path": str(path)} for name, path in submission_paths.items()])),
        "",
        "## 5. 推荐优先提交顺序",
        "",
        _markdown(
            pd.DataFrame(
                [
                    {"candidate": name, "path": str(submission_paths[name])}
                    for name in recommended
                    if name in submission_paths
                ]
            )
        ),
        "",
        "推荐逻辑：",
        "",
        "- 优先选择 `clip_q993` 版本，因为已有验证表明 q993 高价硬裁剪有效。",
        "- 优先选择 conservative/weighted blend，而不是单模型。",
        "- 与当前最好提交做 50/50 log 融合的文件用于测试 TE 模型是否提供增量信息。",
        "",
        "## 6. 文件追溯",
        "",
        "- 训练入口：`run_target_encoding_experiment.py`",
        "- Target Encoding 特征：`house_price/target_encoding.py`",
        f"- 实验中间产物：`{EXPERIMENT_DIR.relative_to(PROJECT_ROOT)}`",
        f"- 提交候选文件：`{SUBMISSION_DIR.relative_to(PROJECT_ROOT)}`",
        "",
        "## 7. 初步结论",
        "",
        "本轮 TE 方案需要用 Kaggle Public Score 和上一轮 optimized 融合一起判断。若纯 TE 提交不能超过上一轮最好，则不能把 TE 视为直接替代方案。",
        "",
        "本轮最关键的验证不是原始 TE 文件，而是：TE q993 裁剪版是否接近当前最好成绩，以及 TE 与当前最好文件的 50/50 log 融合是否能提供增量。",
        "",
        "如果这些文件的 Public Score 不能接近或超过 `0.11805`，则说明当前 TE 组合不是高价值方向，下一轮应转向原生类别模型或更差异化的模型族融合。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _write_navigation_readmes(submission_paths: dict[str, Path]) -> None:
    EXPERIMENT_README_PATH.write_text(
        "\n".join(
            [
                "# Target Encoding 实验产物导航",
                "",
                "本目录记录 20260507 新模型重设计中的第一轮正式主题：OOF Target Encoding。",
                "",
                "## 文件说明",
                "",
                "- `model_cv_results.csv`：本轮所有单模型与融合模型的 OOF CV 记录。",
                "- `oof_predictions.csv`：训练集 OOF log 预测，用于后续融合、诊断和评分校准。",
                "- `test_log_predictions.csv`：测试集 log 预测，用于生成提交文件。",
                "",
                "## 对应报告",
                "",
                "- `reports/modeling/20260507_target_encoding_experiment_report.md`",
                "",
                "## 复现实验",
                "",
                "```powershell",
                "conda run -n kaggle_house python run_target_encoding_experiment.py",
                "```",
            ]
        ),
        encoding="utf-8",
    )
    SUBMISSION_README_PATH.write_text(
        "\n".join(
            [
                "# Target Encoding 提交候选导航",
                "",
                "本目录保存由 `run_target_encoding_experiment.py` 生成的提交候选文件。",
                "",
                "## 命名规则",
                "",
                "- `20260507_te_simple_blend.csv`：本轮 TE 单纯等权融合。",
                "- `20260507_te_weighted_blend.csv`：本轮 TE OOF 优化权重融合。",
                "- `20260507_te_conservative_blend.csv`：本轮 TE 保守融合。",
                "- `*_clip_q993.csv`：在对应融合基础上做 q993 高价硬裁剪。",
                "- `*_mix_current_best_clip_q993.csv`：对应 q993 裁剪版本与当前最好提交做 50/50 log 融合。",
                "",
                "## 本次生成文件",
                "",
                _markdown(pd.DataFrame([{"candidate": name, "path": path.name} for name, path in submission_paths.items()])),
                "",
                "## 选择原则",
                "",
                "最终以 Kaggle Public Score 判断好坏；提交时优先考虑 `conservative`、`weighted`、`clip_q993` 和 `mix_current_best` 这些更稳的版本。",
            ]
        ),
        encoding="utf-8",
    )


def _format_weights(model_names: list[str], weights: np.ndarray) -> str:
    return "; ".join(f"{name}:{weight:.4f}" for name, weight in zip(model_names, weights))


def _rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - y_pred) ** 2) ** 0.5)


def _markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无记录_"
    table = df.copy()
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
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    main()
