from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from house_price.config import DEFAULT_DATA_DIR, PROJECT_ROOT
from house_price.data import load_raw_data


SUBMISSION_DIR = PROJECT_ROOT / "submissions"
CALIBRATION_DIR = SUBMISSION_DIR / "calibration_20260507"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "20260507_calibration_round_report.md"

BASE_FILES = {
    "opt": SUBMISSION_DIR / "20260506_optimized_weight_blend.csv",
    "simple": SUBMISSION_DIR / "20260506_simple_blend.csv",
    "inverse": SUBMISSION_DIR / "20260506_inverse_cv_blend.csv",
    "stack": SUBMISSION_DIR / "20260506_ridge_stack.csv",
}

KNOWN_PUBLIC_SCORES = {
    "20260506_optimized_weight_blend_clipped.csv": 0.11865,
    "20260506_ridge_stack_clipped.csv": 0.11982,
    "20260506_simple_blend.csv": 0.12153,
    "20260506_optimized_weight_blend.csv": 0.12173,
    "20260506_ridge_stack.csv": 0.12258,
}


def main() -> None:
    train, test, sample_submission = load_raw_data(DEFAULT_DATA_DIR)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    base_predictions = _load_base_predictions(sample_submission)
    thresholds = _price_thresholds(train)
    candidate_predictions = _build_candidates(base_predictions, thresholds)
    summary = _write_candidates(candidate_predictions, sample_submission, test, train, thresholds)
    high_price_impact = _high_price_impact_table(base_predictions, candidate_predictions, test)
    summary.to_csv(CALIBRATION_DIR / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    high_price_impact.to_csv(CALIBRATION_DIR / "high_price_impact.csv", index=False, encoding="utf-8-sig")
    _write_report(summary, high_price_impact, thresholds)

    print(f"Calibration candidates written to: {CALIBRATION_DIR}")
    print(f"Report written to: {REPORT_PATH}")


def _load_base_predictions(sample_submission: pd.DataFrame) -> dict[str, pd.Series]:
    predictions = {}
    for name, path in BASE_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing base submission: {path}")
        submission = pd.read_csv(path)
        if not submission["Id"].equals(sample_submission["Id"]):
            raise ValueError(f"Id order mismatch in {path}")
        predictions[name] = submission["SalePrice"].astype(float)
    return predictions


def _price_thresholds(train: pd.DataFrame) -> dict[str, float]:
    sale_price = train["SalePrice"]
    quantiles = {
        "q990": 0.990,
        "q993": 0.993,
        "q995": 0.995,
        "q997": 0.997,
        "q999": 0.999,
    }
    thresholds = {name: float(sale_price.quantile(q)) for name, q in quantiles.items()}
    thresholds["low_q001"] = float(sale_price.quantile(0.001))
    thresholds["train_max"] = float(sale_price.max())
    return thresholds


def _build_candidates(
    base_predictions: dict[str, pd.Series],
    thresholds: dict[str, float],
) -> dict[str, pd.Series]:
    opt = base_predictions["opt"]
    simple = base_predictions["simple"]
    inverse = base_predictions["inverse"]
    stack = base_predictions["stack"]

    candidates: dict[str, pd.Series] = {}

    for threshold_name in ["q990", "q993", "q995", "q997", "q999"]:
        candidates[f"opt_clip_{threshold_name}"] = _clip(opt, thresholds[threshold_name], thresholds["low_q001"])

    candidates["mix50_opt_simple_clip_q997"] = _clip(
        _log_blend({"opt": opt, "simple": simple}, {"opt": 0.5, "simple": 0.5}),
        thresholds["q997"],
        thresholds["low_q001"],
    )
    candidates["mix70_opt30_simple_clip_q997"] = _clip(
        _log_blend({"opt": opt, "simple": simple}, {"opt": 0.7, "simple": 0.3}),
        thresholds["q997"],
        thresholds["low_q001"],
    )
    candidates["mix30_opt70_simple_clip_q997"] = _clip(
        _log_blend({"opt": opt, "simple": simple}, {"opt": 0.3, "simple": 0.7}),
        thresholds["q997"],
        thresholds["low_q001"],
    )
    candidates["mean_opt_simple_inverse_clip_q997"] = _clip(
        _log_blend(
            {"opt": opt, "simple": simple, "inverse": inverse},
            {"opt": 1 / 3, "simple": 1 / 3, "inverse": 1 / 3},
        ),
        thresholds["q997"],
        thresholds["low_q001"],
    )
    candidates["mean_opt_simple_stack_clip_q997"] = _clip(
        _log_blend(
            {"opt": opt, "simple": simple, "stack": stack},
            {"opt": 1 / 3, "simple": 1 / 3, "stack": 1 / 3},
        ),
        thresholds["q997"],
        thresholds["low_q001"],
    )

    candidates["opt_soft_q995_s035"] = _soft_shrink(opt, thresholds["q995"], 0.35, thresholds["low_q001"])
    candidates["opt_soft_q997_s050"] = _soft_shrink(opt, thresholds["q997"], 0.50, thresholds["low_q001"])
    candidates["mix50_soft_q995_s035"] = _soft_shrink(
        _log_blend({"opt": opt, "simple": simple}, {"opt": 0.5, "simple": 0.5}),
        thresholds["q995"],
        0.35,
        thresholds["low_q001"],
    )

    return candidates


def _clip(predictions: pd.Series, high: float, low: float) -> pd.Series:
    return predictions.clip(lower=low, upper=high)


def _soft_shrink(predictions: pd.Series, threshold: float, tail_strength: float, low: float) -> pd.Series:
    calibrated = predictions.copy()
    above_threshold = calibrated > threshold
    calibrated.loc[above_threshold] = threshold + (
        calibrated.loc[above_threshold] - threshold
    ) * tail_strength
    return calibrated.clip(lower=low)


def _log_blend(predictions: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    total_weight = sum(weights.values())
    log_prediction = None
    for name, prediction in predictions.items():
        weighted_log = np.log1p(prediction) * (weights[name] / total_weight)
        log_prediction = weighted_log if log_prediction is None else log_prediction + weighted_log
    return pd.Series(np.expm1(log_prediction), index=next(iter(predictions.values())).index)


def _write_candidates(
    candidate_predictions: dict[str, pd.Series],
    sample_submission: pd.DataFrame,
    test: pd.DataFrame,
    train: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows = []
    previous_best = pd.read_csv(SUBMISSION_DIR / "20260506_optimized_weight_blend_clipped.csv")[
        "SalePrice"
    ].astype(float)

    for name, predictions in candidate_predictions.items():
        path = CALIBRATION_DIR / f"20260507_{name}.csv"
        submission = sample_submission.copy()
        submission["SalePrice"] = predictions
        submission.to_csv(path, index=False)
        changed_from_best = np.abs(predictions - previous_best)
        rows.append(
            {
                "candidate": name,
                "file": str(path.relative_to(PROJECT_ROOT)),
                "min": predictions.min(),
                "median": predictions.median(),
                "p95": predictions.quantile(0.95),
                "p99": predictions.quantile(0.99),
                "max": predictions.max(),
                "n_above_q990": int((predictions > thresholds["q990"]).sum()),
                "n_above_q997": int((predictions > thresholds["q997"]).sum()),
                "mean_abs_diff_from_best": changed_from_best.mean(),
                "max_abs_diff_from_best": changed_from_best.max(),
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values(["n_above_q997", "p99", "max"]).round(4)


def _high_price_impact_table(
    base_predictions: dict[str, pd.Series],
    candidate_predictions: dict[str, pd.Series],
    test: pd.DataFrame,
) -> pd.DataFrame:
    selected_candidates = {
        "opt": base_predictions["opt"],
        "simple": base_predictions["simple"],
        "previous_best_clipped": pd.read_csv(
            SUBMISSION_DIR / "20260506_optimized_weight_blend_clipped.csv"
        )["SalePrice"].astype(float),
        "opt_clip_q995": candidate_predictions["opt_clip_q995"],
        "opt_clip_q997": candidate_predictions["opt_clip_q997"],
        "mix50_opt_simple_clip_q997": candidate_predictions["mix50_opt_simple_clip_q997"],
        "opt_soft_q995_s035": candidate_predictions["opt_soft_q995_s035"],
    }
    table = test[
        [
            "Id",
            "OverallQual",
            "GrLivArea",
            "TotalBsmtSF",
            "GarageCars",
            "GarageArea",
            "YearBuilt",
            "Neighborhood",
            "LotArea",
        ]
    ].copy()
    for name, predictions in selected_candidates.items():
        table[name] = predictions.values
    table["opt_minus_previous_best"] = table["opt"] - table["previous_best_clipped"]
    return table.sort_values("opt", ascending=False).head(30).round(2)


def _write_report(
    summary: pd.DataFrame,
    high_price_impact: pd.DataFrame,
    thresholds: dict[str, float],
) -> None:
    recommended = [
        "opt_clip_q995",
        "mix50_opt_simple_clip_q997",
        "opt_soft_q995_s035",
        "mean_opt_simple_inverse_clip_q997",
    ]
    recommended_table = (
        summary.set_index("candidate")
        .loc[recommended, ["file", "max", "p99", "mean_abs_diff_from_best"]]
        .reset_index()
    )
    lines = [
        "# 20260507 高价校准与保守融合实验报告",
        "",
        "## 实验目标",
        "",
        "上一轮提交显示，`clipped` 版本显著优于未裁剪版本。因此本轮不重新训练底层模型，而是围绕两个方向做高价值迭代：",
        "",
        "1. 调整高价预测裁剪阈值。",
        "2. 用更保守的 log 空间融合降低权重搜索过拟合。",
        "",
        "## 已知 Kaggle 反馈",
        "",
        _markdown(
            pd.DataFrame(
                [{"file": file, "public_score": score} for file, score in KNOWN_PUBLIC_SCORES.items()]
            ).sort_values("public_score")
        ),
        "",
        "## 裁剪阈值",
        "",
        _markdown(
            pd.DataFrame(
                [
                    {"threshold": name, "price": value}
                    for name, value in thresholds.items()
                    if name.startswith("q")
                ]
            )
        ),
        "",
        "上一轮最优 `optimized_weight_blend_clipped` 使用的是接近 `q997` 的上限。本轮会测试更低和更高的上限。",
        "",
        "## 本轮候选文件",
        "",
        _markdown(summary),
        "",
        "## 推荐提交顺序",
        "",
        _markdown(recommended_table),
        "",
        "推荐理由：",
        "",
        "- `opt_clip_q995`：在上一轮最优的基础上进一步降低高价上限，直接验证更强裁剪是否继续有效。",
        "- `mix50_opt_simple_clip_q997`：结合 optimized 和 simple 两种上一轮有效信号，保持 q997 裁剪。",
        "- `opt_soft_q995_s035`：不是硬裁剪，而是对高价尾部做软压缩，验证是否比硬裁剪更自然。",
        "- `mean_opt_simple_inverse_clip_q997`：三种非 stacking 融合的保守平均，测试更稳健的融合。",
        "",
        "## 高价样本影响分析",
        "",
        "下表展示上一轮 optimized 预测最高的 30 个测试样本，以及本轮几个候选对它们的校准效果。",
        "",
        _markdown(high_price_impact),
        "",
        "## 提交后如何判断方向",
        "",
        "如果 `opt_clip_q995` 继续优于 `0.11865`，说明高价上限还可以继续下调。",
        "",
        "如果 `mix50_opt_simple_clip_q997` 更好，说明保守融合比单纯权重优化更可靠。",
        "",
        "如果 `opt_soft_q995_s035` 更好，说明软校准比硬裁剪更合适，下一轮可以继续调软压缩强度。",
        "",
        "如果这些都没有超过 `0.11865`，说明当前裁剪收益可能已经接近上限，下一轮应转向 OOF target encoding 或更严格的 stacking。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无记录_"
    headers = [str(column) for column in df.columns]
    rows = [[_format_value(value) for value in row] for row in df.to_numpy()]
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


if __name__ == "__main__":
    main()
