from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def plot_confusion_matrix(tp: int, fp: int, fn: int, tn: int, output_path: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    mat = [[tn, fp], [fn, tp]]
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred Normal", "Pred Abnormal"])
    ax.set_yticks([0, 1], labels=["True Normal", "True Abnormal"])
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i][j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_f1_by_dataset(by_dataset_df: pd.DataFrame, output_path: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    if by_dataset_df.empty:
        ax.text(0.5, 0.5, "No dataset split available", ha="center", va="center")
        ax.set_axis_off()
    else:
        data = by_dataset_df.sort_values("f1", ascending=False)
        ax.bar(data["dataset"], data["f1"], color="#4C78A8")
        ax.set_ylim(0, 1)
        ax.set_title("F1 by Dataset")
        ax.set_ylabel("F1")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_error_type_counts(enriched_df: pd.DataFrame, output_path: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = (
        enriched_df["error_type"].value_counts()
        .reindex(["TP", "FP", "FN", "TN"], fill_value=0)
    )
    ax.bar(counts.index, counts.values, color=["#54A24B", "#E45756", "#F58518", "#72B7B2"])
    ax.set_title("Prediction Outcome Counts")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def generate_all_plots(
    out_dir: str,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    by_dataset_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
) -> Dict[str, str]:
    base = Path(out_dir)
    paths = {
        "plot_confusion_matrix": str(base / "plot_confusion_matrix.png"),
        "plot_f1_by_dataset": str(base / "plot_f1_by_dataset.png"),
        "plot_error_counts": str(base / "plot_error_counts.png"),
    }
    plot_confusion_matrix(tp, fp, fn, tn, paths["plot_confusion_matrix"])
    plot_f1_by_dataset(by_dataset_df, paths["plot_f1_by_dataset"])
    plot_error_type_counts(enriched_df, paths["plot_error_counts"])
    return paths

