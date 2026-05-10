from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "reports" / "final_report" / "tables"
FIGURE_DIR = PROJECT_ROOT / "reports" / "final_report" / "figures"
EXPERIMENT_LOG = PROJECT_ROOT / "experiments" / "experiment_log.csv"
SUMMARY_TABLE = TABLE_DIR / "experiment_results_summary.csv"


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY_TABLE)
    log = pd.read_csv(EXPERIMENT_LOG)

    _plot_public_score_progress(summary)
    _plot_stage_improvements(summary)
    _plot_submitted_candidate_ranking(log)


def _plot_public_score_progress(summary: pd.DataFrame) -> None:
    data = summary.copy()
    labels = [
        "Baseline",
        "Advanced\nblend",
        "Clipped\nblend",
        "q993\ncalibration",
        "TE\nalone",
        "Final\nmix",
    ]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(labels, data["kaggle_public_score"], marker="o", linewidth=2.4, color="#2563eb")
    ax.scatter(labels[-1], data["kaggle_public_score"].iloc[-1], s=90, color="#dc2626", zorder=3)

    for index, score in enumerate(data["kaggle_public_score"]):
        ax.annotate(
            f"{score:.5f}",
            (index, score),
            textcoords="offset points",
            xytext=(0, -18 if index in [0, 1] else 10),
            ha="center",
            fontsize=9,
        )

    ax.set_title("Kaggle Public Score Progress")
    ax.set_ylabel("Public Score (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "public_score_progress.png", dpi=180)
    plt.close(fig)


def _plot_stage_improvements(summary: pd.DataFrame) -> None:
    data = summary.dropna(subset=["score_change_vs_previous_key_stage"]).copy()
    data["improvement"] = -data["score_change_vs_previous_key_stage"].astype(float)
    labels = [
        "Advanced\nfeatures",
        "High-price\nclipping",
        "q993\ncalibration",
        "TE\nalone",
        "Final\nlog mix",
    ]
    colors = ["#0f766e", "#0f766e", "#64748b", "#64748b", "#dc2626"]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.bar(labels, data["improvement"], color=colors)
    ax.set_title("Score Improvement by Key Operation")
    ax.set_ylabel("Public Score improvement")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, data["improvement"]):
        ax.annotate(
            f"{value:.5f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "score_improvement_by_stage.png", dpi=180)
    plt.close(fig)


def _plot_submitted_candidate_ranking(log: pd.DataFrame) -> None:
    submitted = log.dropna(subset=["kaggle_public_score"]).copy()
    submitted["kaggle_public_score"] = submitted["kaggle_public_score"].astype(float)
    submitted = submitted.sort_values("kaggle_public_score", ascending=True)
    labels = submitted["experiment"].str.replace("20260507_", "", regex=False)
    labels = labels.str.replace("20260506_", "", regex=False)
    colors = ["#dc2626" if i == 0 else "#2563eb" for i in range(len(submitted))]

    fig_height = max(5.0, 0.36 * len(submitted))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    bars = ax.barh(labels, submitted["kaggle_public_score"], color=colors)
    ax.set_title("Submitted Candidate Ranking by Kaggle Public Score")
    ax.set_xlabel("Public Score (lower is better)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.22)
    ax.set_xlim(submitted["kaggle_public_score"].min() - 0.001, submitted["kaggle_public_score"].max() + 0.001)

    for bar, value in zip(bars, submitted["kaggle_public_score"]):
        ax.annotate(
            f"{value:.5f}",
            (value, bar.get_y() + bar.get_height() / 2),
            textcoords="offset points",
            xytext=(4, 0),
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "submitted_candidate_ranking.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
