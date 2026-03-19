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
    for err in ["FP", "FN", "FP_UNKNOWN", "FN_UNKNOWN"]:
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
    lines.append(f"| Unknown Pred | {overall.get('unknown_pred_count', 0)} |")
    lines.append(f"| Parse Failed | {overall.get('parse_failed_count', 0)} |")
    lines.append(f"| Fallback Order Mapping | {overall.get('fallback_order_mapping_count', 0)} |")
    lines.append(f"| First-Pass Fallback | {overall.get('first_pass_fallback_count', 0)} |")
    lines.append(f"| Keyword Fallback | {overall.get('keyword_fallback_count', 0)} |")
    lines.append(f"| Retry Used | {overall.get('retry_used_count', 0)} |")
    lines.append(f"| Retry Changed Label | {overall.get('retry_changed_label_count', 0)} |")
    lines.append(f"| Key Evidence Count | {overall.get('key_evidence_count', 0)} |")
    lines.append(f"| Key Evidence Rate | {_fmt(overall.get('key_evidence_rate', 0.0))} |")
    lines.append(f"| System-Level Evidence Count | {overall.get('system_level_evidence_count', 0)} |")
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

    if "evidence_type_distribution" in overall:
        lines.append("## Evidence Type Distribution")
        lines.append("")
        lines.append("| Evidence Type | Count |")
        lines.append("|---|---:|")
        for k, v in overall["evidence_type_distribution"].items():
            lines.append(f"| {k} | {int(v)} |")
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


def _classify_format_drift(raw_response: str) -> str:
    txt = str(raw_response or "")
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    low = txt.lower()
    if len(lines) > 1:
        return "multi_line_output"
    if any(k in low for k in ["prediction:", "answer:", "classification:"]):
        return "extra_prefix"
    label_hits = low.count("normal") + low.count("abnormal")
    if label_hits > 1:
        return "repeated_output"
    if not re.search(r"^\s*1[\.\)]\s*", txt):
        if re.search(r"\b(normal|abnormal)\b", low):
            return "missing_id_prefix"
        return "no_label_detected"
    if not re.search(r"^\s*1[\.\)]\s*(normal|abnormal)\b", low):
        return "label_not_at_start"
    return "other"


def _export_fallback_buckets(predictions: pd.DataFrame, out_dir: Path) -> str:
    if "first_pass_fallback" in predictions.columns:
        out = predictions[predictions["first_pass_fallback"].fillna(False)].copy()
    elif "initial_parse_method" in predictions.columns:
        out = predictions[predictions["initial_parse_method"] == "fallback_order_mapping"].copy()
    elif "parse_method" not in predictions.columns:
        out = predictions.head(0).copy()
    else:
        out = predictions[predictions["parse_method"] == "fallback_order_mapping"].copy()
    if not out.empty:
        out["format_bucket"] = out["raw_response"].map(_classify_format_drift)
    file_path = out_dir / "fallback_order_mapping_cases.csv"
    out.to_csv(file_path, index=False)
    return str(file_path)


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
    if "parse_method" in predictions.columns:
        metrics["parse_failed_count"] = int((predictions["parse_method"] == "parse_failed").sum())
        metrics["fallback_order_mapping_count"] = int(
            (predictions["parse_method"] == "fallback_order_mapping").sum()
        )
    else:
        metrics["parse_failed_count"] = 0
        metrics["fallback_order_mapping_count"] = 0
    if "used_keyword_fallback" in predictions.columns:
        metrics["keyword_fallback_count"] = int(predictions["used_keyword_fallback"].fillna(False).sum())
    else:
        metrics["keyword_fallback_count"] = 0
    if "used_retry" in predictions.columns:
        metrics["retry_used_count"] = int(predictions["used_retry"].fillna(False).sum())
    else:
        metrics["retry_used_count"] = 0
    if "first_pass_fallback" in predictions.columns:
        metrics["first_pass_fallback_count"] = int(predictions["first_pass_fallback"].fillna(False).sum())
    elif "initial_parse_method" in predictions.columns:
        metrics["first_pass_fallback_count"] = int(
            (predictions["initial_parse_method"] == "fallback_order_mapping").sum()
        )
    else:
        metrics["first_pass_fallback_count"] = 0
    if "retry_changed_label" in predictions.columns:
        metrics["retry_changed_label_count"] = int(predictions["retry_changed_label"].fillna(False).sum())
    else:
        metrics["retry_changed_label_count"] = 0
    if "is_key_evidence" in predictions.columns:
        metrics["key_evidence_count"] = int(predictions["is_key_evidence"].fillna(False).sum())
        metrics["key_evidence_rate"] = (
            float(metrics["key_evidence_count"] / len(predictions)) if len(predictions) else 0.0
        )
    else:
        metrics["key_evidence_count"] = 0
        metrics["key_evidence_rate"] = 0.0
    if "evidence_type" in predictions.columns:
        vc = predictions["evidence_type"].fillna("unknown").astype(str).value_counts()
        metrics["evidence_type_distribution"] = {str(k): int(v) for k, v in vc.items()}
        metrics["system_level_evidence_count"] = int(
            (predictions["evidence_type"].fillna("").astype(str) == "system_level_evidence").sum()
        )
    else:
        metrics["evidence_type_distribution"] = {}
        metrics["system_level_evidence_count"] = 0

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
    enriched[enriched["error_type"].isin(["FP", "FP_UNKNOWN"])].to_csv(fp_file, index=False)
    enriched[enriched["error_type"].isin(["FN", "FN_UNKNOWN"])].to_csv(fn_file, index=False)
    error_profile = _collect_error_profile(enriched)
    error_profile_file = out_dir / "error_profile.csv"
    error_profile.to_csv(error_profile_file, index=False)
    evidence_profile_file = out_dir / "evidence_profile.csv"
    if "evidence_type" in predictions.columns:
        ep = (
            predictions["evidence_type"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .rename_axis("evidence_type")
            .reset_index(name="count")
        )
        ep.to_csv(evidence_profile_file, index=False)
    else:
        pd.DataFrame(columns=["evidence_type", "count"]).to_csv(evidence_profile_file, index=False)
    fallback_cases_file = _export_fallback_buckets(predictions, out_dir)

    out_paths = {
        "summary_json": str(summary_file),
        "config": str(cfg_file),
        "predictions": str(pred_file),
        "predictions_with_errors": str(enriched_file),
        "false_positives": str(fp_file),
        "false_negatives": str(fn_file),
        "metrics_by_dataset": str(metrics_by_dataset_file),
        "error_profile": str(error_profile_file),
        "evidence_profile": str(evidence_profile_file),
        "fallback_order_mapping_cases": str(fallback_cases_file),
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
        "evidence_profile": str(evidence_profile_file),
        "fallback_order_mapping_cases": str(fallback_cases_file),
        "overall_metrics": metrics,
    }
