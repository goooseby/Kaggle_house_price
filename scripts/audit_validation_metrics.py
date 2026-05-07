from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from house_price.config import DEFAULT_DATA_DIR, PROJECT_ROOT
from house_price.data import load_raw_data


EXPERIMENT_LOG = PROJECT_ROOT / "experiments" / "experiment_log.csv"
AUDIT_TABLE_PATH = PROJECT_ROOT / "experiments" / "submission_score_audit.csv"
CORRELATION_PATH = PROJECT_ROOT / "experiments" / "validation_metric_correlations.csv"
PREDICTION_COMPARISON_PATH = PROJECT_ROOT / "experiments" / "score_prediction_comparison.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "20260507_validation_audit_report.md"


SOURCE_CV_PROXY = {
    "baseline": 0.11122,
    "optimized": 0.10609,
    "simple": 0.10689,
    "ridge_stack": 0.10516,
}


def main() -> None:
    train, _test, _sample_submission = load_raw_data(DEFAULT_DATA_DIR)
    experiment_log = pd.read_csv(EXPERIMENT_LOG)
    audit_table = _build_submission_audit(experiment_log, train)
    correlations = _metric_correlations(audit_table)
    prediction_comparison = _score_prediction_comparison(audit_table)
    audit_table.to_csv(AUDIT_TABLE_PATH, index=False, encoding="utf-8-sig")
    correlations.to_csv(CORRELATION_PATH, index=False, encoding="utf-8-sig")
    prediction_comparison.to_csv(PREDICTION_COMPARISON_PATH, index=False, encoding="utf-8-sig")
    _write_report(audit_table, correlations, prediction_comparison, train)
    print(f"Audit table written to: {AUDIT_TABLE_PATH}")
    print(f"Correlation table written to: {CORRELATION_PATH}")
    print(f"Prediction comparison written to: {PREDICTION_COMPARISON_PATH}")
    print(f"Report written to: {REPORT_PATH}")


def _build_submission_audit(experiment_log: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    submitted = experiment_log.dropna(subset=["kaggle_public_score"]).copy()
    train_price = train["SalePrice"]
    log_train_price = np.log1p(train_price)
    thresholds = {
        "q990": train_price.quantile(0.990),
        "q993": train_price.quantile(0.993),
        "q995": train_price.quantile(0.995),
        "q997": train_price.quantile(0.997),
        "q999": train_price.quantile(0.999),
    }

    for _, row in submitted.iterrows():
        path = PROJECT_ROOT / str(row["submission_file"]).replace("\\", "/")
        if not path.exists():
            continue
        submission = pd.read_csv(path)
        predictions = submission["SalePrice"].astype(float)
        log_predictions = np.log1p(predictions)
        local_cv = _coerce_float(row.get("local_cv_rmse"))
        source_family = _source_family(str(row["experiment"]), str(row["submission_file"]))
        source_cv_proxy = SOURCE_CV_PROXY.get(source_family, np.nan)
        rows.append(
            {
                "experiment": row["experiment"],
                "file": str(row["submission_file"]),
                "public_score": float(row["kaggle_public_score"]),
                "local_cv_rmse": local_cv,
                "source_family": source_family,
                "source_cv_proxy": source_cv_proxy,
                "is_clipped": _is_clipped(str(row["submission_file"])),
                "min": predictions.min(),
                "mean": predictions.mean(),
                "median": predictions.median(),
                "std": predictions.std(),
                "p90": predictions.quantile(0.90),
                "p95": predictions.quantile(0.95),
                "p99": predictions.quantile(0.99),
                "max": predictions.max(),
                "log_mean": log_predictions.mean(),
                "log_std": log_predictions.std(),
                "log_p99": log_predictions.quantile(0.99),
                "max_over_train_q993": predictions.max() / thresholds["q993"],
                "max_over_train_q997": predictions.max() / thresholds["q997"],
                "p99_minus_train_p99": predictions.quantile(0.99) - train_price.quantile(0.99),
                "max_minus_train_q993": predictions.max() - thresholds["q993"],
                "max_minus_train_q997": predictions.max() - thresholds["q997"],
                "n_above_q990": int((predictions > thresholds["q990"]).sum()),
                "n_above_q993": int((predictions > thresholds["q993"]).sum()),
                "n_above_q995": int((predictions > thresholds["q995"]).sum()),
                "n_above_q997": int((predictions > thresholds["q997"]).sum()),
                "n_above_q999": int((predictions > thresholds["q999"]).sum()),
                "tail_mean_above_q993": _tail_mean(predictions, thresholds["q993"]),
                "tail_excess_sum_q993": _tail_excess_sum(predictions, thresholds["q993"]),
                "tail_excess_sum_q997": _tail_excess_sum(predictions, thresholds["q997"]),
                "log_ks_vs_train": _ks_stat(log_predictions, log_train_price),
                "abs_log_mean_diff_vs_train": abs(log_predictions.mean() - log_train_price.mean()),
                "abs_log_std_diff_vs_train": abs(log_predictions.std() - log_train_price.std()),
            }
        )

    audit_table = pd.DataFrame(rows)
    audit_table = audit_table.sort_values("public_score").reset_index(drop=True)
    return audit_table.round(6)


def _metric_correlations(audit_table: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "local_cv_rmse",
        "source_cv_proxy",
        "max",
        "p99",
        "std",
        "log_std",
        "max_over_train_q993",
        "max_over_train_q997",
        "p99_minus_train_p99",
        "max_minus_train_q993",
        "max_minus_train_q997",
        "n_above_q990",
        "n_above_q993",
        "n_above_q995",
        "n_above_q997",
        "n_above_q999",
        "tail_mean_above_q993",
        "tail_excess_sum_q993",
        "tail_excess_sum_q997",
        "log_ks_vs_train",
        "abs_log_mean_diff_vs_train",
        "abs_log_std_diff_vs_train",
    ]
    rows = []
    for metric in metric_columns:
        data = audit_table[["public_score", metric]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 4 or data[metric].nunique() < 2:
            continue
        pearson = pearsonr(data[metric], data["public_score"]).statistic
        spearman = spearmanr(data[metric], data["public_score"]).statistic
        rows.append(
            {
                "metric": metric,
                "n": len(data),
                "pearson_with_public": pearson,
                "spearman_with_public": spearman,
                "abs_spearman": abs(spearman),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False).round(6)


def _score_prediction_comparison(audit_table: pd.DataFrame) -> pd.DataFrame:
    data = audit_table.copy()
    old_feature = data[["source_cv_proxy"]].fillna(data["source_cv_proxy"].mean())
    improved_features = data[
        [
            "source_cv_proxy",
            "tail_excess_sum_q993",
            "tail_excess_sum_q997",
            "max_over_train_q993",
            "std",
            "n_above_q995",
        ]
    ].copy()
    improved_features["log_tail_excess_q993"] = np.log1p(improved_features["tail_excess_sum_q993"])
    improved_features["log_tail_excess_q997"] = np.log1p(improved_features["tail_excess_sum_q997"])
    improved_features = improved_features.drop(columns=["tail_excess_sum_q993", "tail_excess_sum_q997"])
    improved_features = improved_features.replace([np.inf, -np.inf], np.nan).fillna(
        improved_features.median(numeric_only=True)
    )
    y = data["public_score"]

    loo = LeaveOneOut()
    old_model = LinearRegression()
    improved_model = make_pipeline(StandardScaler(), LinearRegression())

    data["old_cv_calibrated_prediction"] = cross_val_predict(old_model, old_feature, y, cv=loo)
    data["improved_tail_calibrated_prediction"] = cross_val_predict(
        improved_model,
        improved_features,
        y,
        cv=loo,
    )
    data["old_abs_error"] = (data["old_cv_calibrated_prediction"] - data["public_score"]).abs()
    data["improved_abs_error"] = (
        data["improved_tail_calibrated_prediction"] - data["public_score"]
    ).abs()
    data["old_rank"] = data["old_cv_calibrated_prediction"].rank(method="min")
    data["improved_rank"] = data["improved_tail_calibrated_prediction"].rank(method="min")
    data["actual_rank"] = data["public_score"].rank(method="min")

    columns = [
        "experiment",
        "public_score",
        "source_cv_proxy",
        "old_cv_calibrated_prediction",
        "improved_tail_calibrated_prediction",
        "old_abs_error",
        "improved_abs_error",
        "actual_rank",
        "old_rank",
        "improved_rank",
        "max",
        "tail_excess_sum_q993",
        "n_above_q995",
    ]
    return data[columns].sort_values("public_score").round(6)


def _write_report(
    audit_table: pd.DataFrame,
    correlations: pd.DataFrame,
    prediction_comparison: pd.DataFrame,
    train: pd.DataFrame,
) -> None:
    submitted = audit_table.copy()
    best = submitted.iloc[0]
    local_available = submitted.dropna(subset=["local_cv_rmse"])
    source_available = submitted.dropna(subset=["source_cv_proxy"])
    train_price = train["SalePrice"]
    lines = [
        "# 20260507 本地验证与 Public Score 审计报告",
        "",
        "## 1. 目的",
        "",
        "这份报告不继续优化提交文件，而是审计已有实验：本地 CV、预测分布指标和 Kaggle Public Score 之间到底是什么关系。",
        "",
        "目标是为下一轮更大模型改造建立更可靠的本地评分策略，避免继续被漂亮但失真的 CV 误导。",
        "",
        "## 2. 已审计提交",
        "",
        _markdown(
            submitted[
                [
                    "experiment",
                    "public_score",
                    "local_cv_rmse",
                    "source_family",
                    "source_cv_proxy",
                    "is_clipped",
                    "max",
                    "p99",
                    "n_above_q993",
                ]
            ]
        ),
        "",
        "当前最佳 Public Score：",
        "",
        f"```text\n{best['experiment']}: {best['public_score']:.5f}\n```",
        "",
        "## 3. 本地 CV 的问题",
        "",
        "有直接本地 CV 的提交如下：",
        "",
        _markdown(
            local_available[
                ["experiment", "local_cv_rmse", "public_score", "source_family", "is_clipped"]
            ]
        ),
        "",
        "关键问题：",
        "",
        "- `ridge_stack` 本地 CV 最好，但 Public Score 很差，说明当前 stacking 评分方式明显偏乐观。",
        "- `optimized_weight_blend` 本地 CV 好于 `simple_blend`，但未裁剪 Public Score 反而略差。",
        "- 只看普通 KFold CV，会低估高价尾部过估带来的损失。",
        "",
        "## 4. Public Score 相关性较高的指标",
        "",
        "下表展示不同本地/预测分布指标与 Public Score 的相关性。由于样本数量有限，这不是严格统计结论，但足以帮助判断方向。",
        "",
        _markdown(correlations.head(15)),
        "",
        "解释方式：",
        "",
        "- `spearman_with_public` 越接近 1，说明该指标越大，Public Score 越差。",
        "- `spearman_with_public` 越接近 -1，说明该指标越大，Public Score 越好。",
        "- 当前我们更关注排序关系，所以 Spearman 比 Pearson 更有参考价值。",
        "",
        "## 5. 高价尾部指标为什么重要",
        "",
        "训练集高价分位数：",
        "",
        _markdown(
            pd.DataFrame(
                [
                    {"quantile": "q990", "SalePrice": train_price.quantile(0.990)},
                    {"quantile": "q993", "SalePrice": train_price.quantile(0.993)},
                    {"quantile": "q995", "SalePrice": train_price.quantile(0.995)},
                    {"quantile": "q997", "SalePrice": train_price.quantile(0.997)},
                    {"quantile": "q999", "SalePrice": train_price.quantile(0.999)},
                    {"quantile": "max", "SalePrice": train_price.max()},
                ]
            )
        ),
        "",
        "从提交结果看，高价尾部控制比原始 CV 更能解释 Public Score 的变化：",
        "",
        "- q997 裁剪优于不裁剪。",
        "- q995 优于 q997。",
        "- q993 优于 q995。",
        "- q990 又差于 q993。",
        "",
        "这说明模型主体对大多数样本已经不错，但高价尾部需要被纳入验证指标。",
        "",
        "## 6. 建议的新评分面板",
        "",
        "后续新模型不应该只报告一个 CV RMSE，而应报告一个评分面板：",
        "",
        "| 指标 | 作用 | 目标 |",
        "| --- | --- | --- |",
        "| `cv_rmse_mean` | 普通总体误差 | 越低越好 |",
        "| `cv_rmse_std` | CV 稳定性 | 越低越好 |",
        "| `price_stratified_cv` | 按价格分层后的 CV | 防止高价样本分布不均 |",
        "| `tail_rmse_top_5pct` | 高价样本 RMSE | 检查高价区间是否崩 |",
        "| `tail_bias_top_5pct` | 高价样本平均偏差 | 判断系统性高估/低估 |",
        "| `prediction_p99_gap` | 预测 p99 与训练 p99 差距 | 控制尾部外推 |",
        "| `prediction_max_over_q993` | 预测最大值相对 q993 | 标记高价风险 |",
        "| `neighborhood_group_rmse` | 街区分组误差 | 检查位置类泛化 |",
        "",
        "其中最该新增的是：",
        "",
        "```text",
        "tail_rmse_top_5pct",
        "tail_bias_top_5pct",
        "prediction_p99_gap",
        "prediction_max_over_q993",
        "```",
        "",
        "## 7. 旧评分方法 vs 改进评分方法",
        "",
        "### 旧评分方法是什么",
        "",
        "旧方法主要看普通 CV RMSE，也就是训练阶段输出的 `local_cv_rmse` 或同一模型族的 `source_cv_proxy`。",
        "",
        "它的问题是：",
        "",
        "- 不能区分裁剪和未裁剪版本。",
        "- 不能感知高价尾部预测是否外推过度。",
        "- 对 stacking 这类容易偏乐观的方案过于宽容。",
        "",
        "### 改进评分方法是什么",
        "",
        "改进方法不是一个最终定型的新 CV，而是一个审计代理分数，用于验证哪些本地指标更接近 Public Score。",
        "",
        "它使用：",
        "",
        "- `source_cv_proxy`：原模型族普通 CV",
        "- `tail_excess_sum_q993`：预测超过训练集 q993 的尾部超额总量",
        "- `tail_excess_sum_q997`：预测超过训练集 q997 的尾部超额总量",
        "- `max_over_train_q993`：最大预测值相对 q993 的比例",
        "- `std`：预测分布标准差",
        "- `n_above_q995`：超过 q995 的预测数量",
        "",
        "下面表格里的两个预测分数都使用 leave-one-out 方式估计，避免直接在同一批样本上拟合又评分。",
        "",
        "注意：样本只有 12 个，所以这不是可直接替代 Kaggle 的最终评分器。它只能说明：加入高价尾部指标后，排序和误差明显比旧 CV 更接近 Public Score。",
        "",
        "### 预测分数与真实 Public Score 对比",
        "",
        _markdown(
            prediction_comparison[
                [
                    "experiment",
                    "public_score",
                    "source_cv_proxy",
                    "old_cv_calibrated_prediction",
                    "improved_tail_calibrated_prediction",
                    "old_abs_error",
                    "improved_abs_error",
                    "actual_rank",
                    "old_rank",
                    "improved_rank",
                ]
            ]
        ),
        "",
        "误差汇总：",
        "",
        _markdown(
            pd.DataFrame(
                [
                    {
                        "method": "old_cv_calibrated_prediction",
                        "mean_abs_error": prediction_comparison["old_abs_error"].mean(),
                        "median_abs_error": prediction_comparison["old_abs_error"].median(),
                        "rank_spearman": spearmanr(
                            prediction_comparison["old_cv_calibrated_prediction"],
                            prediction_comparison["public_score"],
                        ).statistic,
                    },
                    {
                        "method": "improved_tail_calibrated_prediction",
                        "mean_abs_error": prediction_comparison["improved_abs_error"].mean(),
                        "median_abs_error": prediction_comparison["improved_abs_error"].median(),
                        "rank_spearman": spearmanr(
                            prediction_comparison["improved_tail_calibrated_prediction"],
                            prediction_comparison["public_score"],
                        ).statistic,
                    },
                ]
            )
        ),
        "",
        "## 8. 对下一轮建模的直接要求",
        "",
        "下一轮如果训练新模型，必须输出：",
        "",
        "1. 普通 CV。",
        "2. 按 `SalePrice` 分箱的分层 CV。",
        "3. 高价 top 5% / top 10% 的 OOF RMSE。",
        "4. 高价 top 5% / top 10% 的 OOF bias。",
        "5. OOF 预测分布和 test 预测分布摘要。",
        "6. 是否需要后处理的风险标记。",
        "",
        "这样我们才知道一个新模型是整体变强，还是只是本地 CV 变漂亮。",
        "",
        "## 9. 结论",
        "",
        "当前最重要的验证改进结论是：",
        "",
        "> 普通 CV 只能作为基础指标，不能单独决定提交。下一轮必须把高价尾部误差和预测分布风险纳入评分。",
        "",
        "这不会直接让当前分数大幅提升，但会让后续新模型的筛选更可靠。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _source_family(experiment: str, file: str) -> str:
    text = f"{experiment} {file}".lower()
    if "baseline" in text:
        return "baseline"
    if "simple" in text and "optimized" not in text:
        return "simple"
    if "ridge_stack" in text or "stack" in text:
        return "ridge_stack"
    if "optimized" in text or "opt_" in text or "optclip" in text:
        return "optimized"
    return "mixed_or_unknown"


def _is_clipped(file: str) -> bool:
    text = file.lower()
    return "clip" in text or "q99" in text


def _coerce_float(value: object) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _tail_mean(predictions: pd.Series, threshold: float) -> float:
    tail = predictions[predictions > threshold]
    if tail.empty:
        return 0.0
    return float(tail.mean())


def _tail_excess_sum(predictions: pd.Series, threshold: float) -> float:
    return float((predictions - threshold).clip(lower=0).sum())


def _ks_stat(left: pd.Series, right: pd.Series) -> float:
    left_values = np.sort(left.dropna().to_numpy())
    right_values = np.sort(right.dropna().to_numpy())
    combined = np.sort(np.concatenate([left_values, right_values]))
    left_cdf = np.searchsorted(left_values, combined, side="right") / len(left_values)
    right_cdf = np.searchsorted(right_values, combined, side="right") / len(right_values)
    return float(np.max(np.abs(left_cdf - right_cdf)))


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
