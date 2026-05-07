from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def evaluate_log_oof_predictions(
    y_true_log: pd.Series,
    oof_pred_log: pd.Series | np.ndarray,
    train_sale_price: pd.Series,
) -> dict[str, float]:
    """Score OOF log-price predictions with both overall and high-price metrics."""
    y_true_log = pd.Series(y_true_log).reset_index(drop=True)
    oof_pred_log = pd.Series(oof_pred_log).reset_index(drop=True)
    y_true_price = np.expm1(y_true_log)
    pred_price = np.expm1(oof_pred_log)

    top_10_cutoff = train_sale_price.quantile(0.90)
    top_5_cutoff = train_sale_price.quantile(0.95)
    top_1_cutoff = train_sale_price.quantile(0.99)

    return {
        "cv_rmse": rmse(y_true_log, oof_pred_log),
        "tail_rmse_top_10pct": _tail_rmse(y_true_log, oof_pred_log, y_true_price, top_10_cutoff),
        "tail_rmse_top_5pct": _tail_rmse(y_true_log, oof_pred_log, y_true_price, top_5_cutoff),
        "tail_rmse_top_1pct": _tail_rmse(y_true_log, oof_pred_log, y_true_price, top_1_cutoff),
        "tail_bias_top_10pct": _tail_bias(y_true_log, oof_pred_log, y_true_price, top_10_cutoff),
        "tail_bias_top_5pct": _tail_bias(y_true_log, oof_pred_log, y_true_price, top_5_cutoff),
        "tail_bias_top_1pct": _tail_bias(y_true_log, oof_pred_log, y_true_price, top_1_cutoff),
        "pred_p99_minus_train_p99": float(pred_price.quantile(0.99) - train_sale_price.quantile(0.99)),
        "pred_max_over_train_q993": float(pred_price.max() / train_sale_price.quantile(0.993)),
        "pred_max_over_train_q997": float(pred_price.max() / train_sale_price.quantile(0.997)),
        "n_pred_above_train_q993": int((pred_price > train_sale_price.quantile(0.993)).sum()),
        "n_pred_above_train_q997": int((pred_price > train_sale_price.quantile(0.997)).sum()),
    }


def summarize_test_predictions(
    test_pred_price: pd.Series | np.ndarray,
    train_sale_price: pd.Series,
) -> dict[str, float]:
    """Summarize test prediction distribution and high-price tail risk."""
    test_pred_price = pd.Series(test_pred_price)
    q990 = train_sale_price.quantile(0.990)
    q993 = train_sale_price.quantile(0.993)
    q995 = train_sale_price.quantile(0.995)
    q997 = train_sale_price.quantile(0.997)
    return {
        "test_pred_min": float(test_pred_price.min()),
        "test_pred_mean": float(test_pred_price.mean()),
        "test_pred_median": float(test_pred_price.median()),
        "test_pred_std": float(test_pred_price.std()),
        "test_pred_p95": float(test_pred_price.quantile(0.95)),
        "test_pred_p99": float(test_pred_price.quantile(0.99)),
        "test_pred_max": float(test_pred_price.max()),
        "test_pred_max_over_train_q993": float(test_pred_price.max() / q993),
        "test_pred_max_over_train_q997": float(test_pred_price.max() / q997),
        "test_pred_tail_excess_sum_q993": float((test_pred_price - q993).clip(lower=0).sum()),
        "test_pred_tail_excess_sum_q997": float((test_pred_price - q997).clip(lower=0).sum()),
        "test_pred_n_above_q990": int((test_pred_price > q990).sum()),
        "test_pred_n_above_q993": int((test_pred_price > q993).sum()),
        "test_pred_n_above_q995": int((test_pred_price > q995).sum()),
        "test_pred_n_above_q997": int((test_pred_price > q997).sum()),
    }


def _tail_rmse(
    y_true_log: pd.Series,
    pred_log: pd.Series,
    y_true_price: pd.Series,
    price_cutoff: float,
) -> float:
    mask = y_true_price >= price_cutoff
    if mask.sum() == 0:
        return np.nan
    return rmse(y_true_log[mask], pred_log[mask])


def _tail_bias(
    y_true_log: pd.Series,
    pred_log: pd.Series,
    y_true_price: pd.Series,
    price_cutoff: float,
) -> float:
    mask = y_true_price >= price_cutoff
    if mask.sum() == 0:
        return np.nan
    return float((pred_log[mask] - y_true_log[mask]).mean())
