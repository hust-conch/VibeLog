from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

from .error_analysis import add_error_flags, confusion_counts
from .metrics import compute_metrics
from .plots import generate_all_plots


def create_run_dir(base_dir: str, run_name: str = "exp") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base_dir) / f"{run_name}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _collect_error_profile(df: pd.DataFrame, top_k: int = 12) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["error_type", "keyword", "count"])
    token_pattern = re.compile(r"[a-zA-Z_]{4,}")
    rows = []
    for err in ["FP", "FN"]:
        sub = df[df["error_type"] == err]
        if sub.empty:
            continue
        tokens = []
        for txt in sub.get("normalized_log", sub.get("raw_log", pd.Series(dtype=str))).astype(str):
            tokens.extend(token_pattern.findall(txt.lower()))
        if not tokens:
            continue
        s = pd.Series(tokens)
        vc = s.value_counts().head(top_k)
        for k, c in vc.items():
            rows.append({"error_type": err, "keyword": k, "count": int(c)})
    return pd.DataFrame(rows)


def _build_summary_md(overall: Dict, by_dataset_df: pd.DataFrame, out_paths: Dict[str, str]) -> str:
    lines = []
    lines.append("# Experiment Summary")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Accuracy | {_fmt(overall['accuracy'])} |")
    lines.append(f"| Precision | {_fmt(overall['precision'])} |")
    lines.append(f"| Recall | {_fmt(overall['recall'])} |")
    lines.append(f"| F1 | {_fmt(overall['f1'])} |")
    lines.append(f"| Samples | {overall['size']} |")
    lines.append(f"| Abnormal(True) | {overall['abnormal_true']} |")
    lines.append(f"| Abnormal(Pred) | {overall['abnormal_pred']} |")
    lines.append("")
    lines.append("## Confusion Matrix")
    lines.append("")
    lines.append("| TP | FP | FN | TN |")
    lines.append("|---:|---:|---:|---:|")
    lines.append(f"| {overall['TP']} | {overall['FP']} | {overall['FN']} | {overall['TN']} |")
    lines.append("")

    if not by_dataset_df.empty:
        lines.append("## By Dataset")
        lines.append("")
        lines.append("| Dataset | Accuracy | Precision | Recall | F1 | Size |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, r in by_dataset_df.iterrows():
            lines.append(
                f"| {r['dataset']} | {_fmt(r['accuracy'])} | {_fmt(r['precision'])} | {_fmt(r['recall'])} | {_fmt(r['f1'])} | {int(r['size'])} |"
            )
        lines.append("")

    lines.append("## Files")
    lines.append("")
    for k, v in out_paths.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    return "\n".join(lines)


def save_report(
    predictions: pd.DataFrame,
    config_dict: Dict,
    output_dir: str,
) -> Dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_file = out_dir / "predictions.csv"
    predictions.to_csv(pred_file, index=False)

    enriched = add_error_flags(predictions)
    enriched_file = out_dir / "predictions_with_errors.csv"
    enriched.to_csv(enriched_file, index=False)

    test_df = enriched[enriched["split"] == "test"] if "split" in enriched.columns else enriched
    metrics = compute_metrics(test_df, pred_col="final_pred", gt_col="label_str")
    tp, fp, fn, tn = confusion_counts(test_df)
    metrics.update({"TP": tp, "FP": fp, "FN": fn, "TN": tn})

    by_dataset = []
    if "dataset" in test_df.columns:
        for name, g in test_df.groupby("dataset", sort=True):
            m = compute_metrics(g, pred_col="final_pred", gt_col="label_str")
            m["dataset"] = name
            by_dataset.append(m)
    by_dataset_df = pd.DataFrame(by_dataset)

    summary_file = out_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"overall": metrics, "by_dataset": by_dataset}, f, ensure_ascii=False, indent=2)
    metrics_by_dataset_file = out_dir / "metrics_by_dataset.csv"
    if not by_dataset_df.empty:
        by_dataset_df.to_csv(metrics_by_dataset_file, index=False)

    cfg_file = out_dir / "config.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

    fp_file = out_dir / "false_positives.csv"
    fn_file = out_dir / "false_negatives.csv"
    enriched[enriched["error_type"] == "FP"].to_csv(fp_file, index=False)
    enriched[enriched["error_type"] == "FN"].to_csv(fn_file, index=False)
    error_profile = _collect_error_profile(enriched)
    error_profile_file = out_dir / "error_profile.csv"
    error_profile.to_csv(error_profile_file, index=False)

    out_paths = {
        "summary_json": str(summary_file),
        "config": str(cfg_file),
        "predictions": str(pred_file),
        "predictions_with_errors": str(enriched_file),
        "false_positives": str(fp_file),
        "false_negatives": str(fn_file),
        "metrics_by_dataset": str(metrics_by_dataset_file),
        "error_profile": str(error_profile_file),
    }
    plot_paths = generate_all_plots(
        out_dir=str(out_dir),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        by_dataset_df=by_dataset_df,
        enriched_df=enriched,
    )
    out_paths.update(plot_paths)
    summary_md = _build_summary_md(metrics, by_dataset_df, out_paths)
    summary_md_file = out_dir / "summary.md"
    summary_md_file.write_text(summary_md, encoding="utf-8")

    return {
        "predictions": str(pred_file),
        "predictions_with_errors": str(enriched_file),
        "summary": str(summary_file),
        "summary_md": str(summary_md_file),
        "config": str(cfg_file),
        "fp": str(fp_file),
        "fn": str(fn_file),
        "metrics_by_dataset": str(metrics_by_dataset_file),
        "error_profile": str(error_profile_file),
        "overall_metrics": metrics,
    }
