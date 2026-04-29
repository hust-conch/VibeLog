from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class DiagnosisConfig:
    window_size: int = 8
    window_stride: int = 4
    top_k_evidence: int = 3
    key_evidence_threshold: float = 65.0
    enable_stage3_llm: bool = True
    stage3_max_windows: int = 120


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: object, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _clip(text: object, n: int = 160) -> str:
    s = str(text) if text is not None else ""
    return s.strip().replace("\n", " ")[:n]


def _evidence_text(window_df: pd.DataFrame) -> str:
    # Use key evidence first to avoid dataset-level prefix bias.
    w = window_df.copy()
    if "is_key_evidence" in w.columns:
        key = w[w["is_key_evidence"].fillna(False).astype(bool)]
    else:
        key = w.head(0)
    if key.empty and "relevance_score" in w.columns:
        key = w.sort_values("relevance_score", ascending=False).head(3)
    pool = key if not key.empty else w
    parts = []
    for c in ["raw_log", "normalized_log", "reason"]:
        if c in pool.columns:
            parts.extend(pool[c].astype(str).tolist())
    return " ".join(parts).lower()


def _count_keywords(text: str, keywords: List[str]) -> int:
    return sum(1 for k in keywords if k in text)


def _infer_failure_type(window_df: pd.DataFrame) -> Tuple[str, str]:
    text = _evidence_text(window_df)
    # Fix: "kernel" token alone is too common in some datasets (e.g., BGL prefix).
    # Require stronger companion cues for kernel_failure.
    kernel_hits = _count_keywords(text, ["panic", "oops", "segfault", "trap", "unrecoverable"])
    storage_hits = _count_keywords(text, ["filesystem", "storage", "disk", "i/o error", "corrupt", "ioerror"])
    network_hits = _count_keywords(text, ["network", "connection", "timeout", "unavailable", "service outage"])
    critical_hits = _count_keywords(text, ["deadlock", "catastrophic", "unrecoverable"])
    app_hits = _count_keywords(text, ["app", "application", "process", "worker", "client", "task", "job"])
    recovery_hits = _count_keywords(text, ["recovered", "corrected", "resumed", "retry succeeded", "restored", "back online"])

    if storage_hits >= 1 and storage_hits >= max(network_hits, app_hits):
        return "storage_failure", "storage/filesystem evidence dominates key lines"
    if network_hits >= 1 and network_hits >= app_hits:
        return "service_or_network_failure", "service/network impact cues dominate key lines"
    if critical_hits >= 1:
        return "critical_system_failure", "critical unrecoverable cues appear in key lines"
    if kernel_hits >= 1:
        return "kernel_failure", "kernel failure cues (panic/oops/segfault) appear in key lines"
    if app_hits >= 1 and recovery_hits == 0:
        return "application_failure", "application-level failure cues dominate key lines"
    if recovery_hits >= 1:
        return "recovery_event", "recovery cues dominate key lines"
    return "benign_or_context", "no strong failure-type signature in key evidence"


def _window_rows(df_one_dataset: pd.DataFrame, cfg: DiagnosisConfig) -> List[pd.DataFrame]:
    rows: List[pd.DataFrame] = []
    n = len(df_one_dataset)
    if n == 0:
        return rows
    if n <= cfg.window_size:
        rows.append(df_one_dataset.copy())
        return rows
    start = 0
    while start < n:
        end = min(start + cfg.window_size, n)
        chunk = df_one_dataset.iloc[start:end].copy()
        if len(chunk) > 0:
            rows.append(chunk)
        if end == n:
            break
        start += max(1, cfg.window_stride)
    return rows


def build_diagnosis_windows(predictions: pd.DataFrame, cfg: DiagnosisConfig | None = None) -> pd.DataFrame:
    config = cfg or DiagnosisConfig()
    if predictions.empty:
        return pd.DataFrame()

    out_rows: List[Dict] = []
    for dataset, g_dataset in predictions.groupby("dataset", sort=True):
        g = g_dataset.sort_values("sample_id").reset_index(drop=True)
        windows = _window_rows(g, config)
        for widx, wdf in enumerate(windows, start=1):
            w = wdf.copy()
            if "relevance_score" not in w.columns:
                w["relevance_score"] = 0.0
            if "evidence_type" not in w.columns:
                w["evidence_type"] = "unknown"
            if "is_key_evidence" not in w.columns:
                w["is_key_evidence"] = False

            w["relevance_score"] = w["relevance_score"].map(lambda x: _safe_float(x, 0.0))
            w["is_key_evidence"] = w["is_key_evidence"].fillna(False).astype(bool)
            key = w[w["is_key_evidence"]].copy()
            if key.empty:
                key = w.sort_values("relevance_score", ascending=False).head(max(1, config.top_k_evidence))
            else:
                key = key.sort_values("relevance_score", ascending=False).head(max(1, config.top_k_evidence))

            evidence_counts = w["evidence_type"].astype(str).value_counts().to_dict()
            system_count = int((w["evidence_type"] == "system_level_evidence").sum())
            app_count = int((w["evidence_type"] == "application_level_evidence").sum())
            recovery_count = int((w["evidence_type"] == "recovery_evidence").sum())
            failure_count = int((w["evidence_type"] == "failure_evidence").sum())

            system_score = float(w.loc[w["evidence_type"] == "system_level_evidence", "relevance_score"].sum())
            app_score = float(w.loc[w["evidence_type"] == "application_level_evidence", "relevance_score"].sum())
            recovery_score = float(w.loc[w["evidence_type"] == "recovery_evidence", "relevance_score"].sum())
            failure_score = float(w.loc[w["evidence_type"] == "failure_evidence", "relevance_score"].sum())
            abnormal_line_count = int((w.get("final_pred", "").astype(str) == "abnormal").sum())

            window_pred = "normal"
            if system_count >= 2 or system_score >= 140 or (system_score >= 70 and abnormal_line_count > 0):
                window_pred = "abnormal"
            elif failure_score >= 160 and recovery_score < 40:
                window_pred = "abnormal"

            failure_type, ft_reason = _infer_failure_type(w)
            rationale = (
                f"window_pred={window_pred}; system_score={system_score:.1f}; "
                f"failure_score={failure_score:.1f}; app_score={app_score:.1f}; recovery_score={recovery_score:.1f}; "
                f"dominant_type={max(evidence_counts, key=evidence_counts.get) if evidence_counts else 'unknown'}; {ft_reason}"
            )

            top_lines = []
            for _, r in key.iterrows():
                top_lines.append(
                    {
                        "sample_id": _safe_int(r.get("sample_id", 0)),
                        "evidence_type": str(r.get("evidence_type", "unknown")),
                        "relevance_score": _safe_float(r.get("relevance_score", 0.0)),
                        "line": _clip(r.get("raw_log", r.get("normalized_log", ""))),
                        "reason": _clip(r.get("reason", ""), 120),
                    }
                )

            gt_window = "abnormal" if (w.get("label_str", "").astype(str) == "abnormal").any() else "normal"
            out_rows.append(
                {
                    "dataset": dataset,
                    "window_id": f"{dataset}_w{widx:04d}",
                    "window_index": widx,
                    "start_sample_id": _safe_int(w["sample_id"].min()),
                    "end_sample_id": _safe_int(w["sample_id"].max()),
                    "window_size": int(len(w)),
                    "window_pred": window_pred,
                    "window_true_proxy": gt_window,
                    "failure_type": failure_type,
                    "top_evidence_count": int(len(top_lines)),
                    "top_evidence_lines": json.dumps(top_lines, ensure_ascii=False),
                    "concise_rationale": rationale,
                    "system_evidence_count": system_count,
                    "application_evidence_count": app_count,
                    "recovery_evidence_count": recovery_count,
                    "failure_evidence_count": failure_count,
                }
            )
    return pd.DataFrame(out_rows)


def summarize_diagnosis_windows(diagnosis_df: pd.DataFrame) -> Dict:
    if diagnosis_df.empty:
        return {
            "windows": 0,
            "proxy_accuracy": 0.0,
            "proxy_precision": 0.0,
            "proxy_recall": 0.0,
            "proxy_f1": 0.0,
            "failure_type_distribution": {},
        }

    y_true = diagnosis_df["window_true_proxy"].astype(str).str.lower()
    y_pred = diagnosis_df["window_pred"].astype(str).str.lower()
    tp = int(((y_true == "abnormal") & (y_pred == "abnormal")).sum())
    fp = int(((y_true == "normal") & (y_pred == "abnormal")).sum())
    fn = int(((y_true == "abnormal") & (y_pred == "normal")).sum())
    tn = int(((y_true == "normal") & (y_pred == "normal")).sum())
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "windows": int(total),
        "proxy_accuracy": float(accuracy),
        "proxy_precision": float(precision),
        "proxy_recall": float(recall),
        "proxy_f1": float(f1),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "failure_type_distribution": diagnosis_df["failure_type"].value_counts().to_dict(),
    }


_STAGE3_LINE = re.compile(
    r"^\s*(normal|abnormal)\s*\|\s*([a-z_]+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$",
    re.IGNORECASE,
)


def _build_stage3_prompt(row: pd.Series) -> str:
    evidence_lines = []
    try:
        parsed = json.loads(str(row.get("top_evidence_lines", "[]")))
        if isinstance(parsed, list):
            for i, e in enumerate(parsed[:5], start=1):
                evidence_lines.append(
                    f"{i}) type={e.get('evidence_type','unknown')}, score={e.get('relevance_score',0)}, log={e.get('line','')}, why={e.get('reason','')}"
                )
    except Exception:
        pass
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "1) no parsed evidence"
    return (
        "You are a system log diagnosis assistant.\n"
        "Given a short window summary and top evidence lines, return one strict line:\n"
        "window_pred|failure_type|impact_component|concise_rationale\n"
        "Rules:\n"
        "- window_pred must be normal or abnormal.\n"
        "- failure_type must be one of: kernel_failure, storage_failure, service_or_network_failure, critical_system_failure, application_failure, recovery_event, benign_or_context.\n"
        "- impact_component should be short (kernel/storage/network/service/application/mixed/unknown).\n"
        "- rationale must cite evidence, not generic wording.\n\n"
        f"dataset={row.get('dataset','unknown')}\n"
        f"window_id={row.get('window_id','')}\n"
        f"rule_window_pred={row.get('window_pred','normal')}\n"
        f"rule_failure_type={row.get('failure_type','benign_or_context')}\n"
        f"evidence_counts: system={row.get('system_evidence_count',0)}, app={row.get('application_evidence_count',0)}, recovery={row.get('recovery_evidence_count',0)}, failure={row.get('failure_evidence_count',0)}\n"
        f"top_evidence_lines:\n{evidence_block}\n"
    )


def _parse_stage3_response(resp: str) -> Dict[str, str]:
    txt = str(resp or "").strip()
    m = _STAGE3_LINE.search(txt.splitlines()[0] if txt else "")
    if not m:
        return {}
    pred = m.group(1).lower().strip()
    ftype = m.group(2).lower().strip()
    comp = m.group(3).strip()
    rationale = m.group(4).strip()
    allowed = {
        "kernel_failure",
        "storage_failure",
        "service_or_network_failure",
        "critical_system_failure",
        "application_failure",
        "recovery_event",
        "benign_or_context",
    }
    if pred not in {"normal", "abnormal"}:
        return {}
    if ftype not in allowed:
        return {}
    return {
        "stage3_window_pred": pred,
        "stage3_failure_type": ftype,
        "stage3_impact_component": comp or "unknown",
        "stage3_rationale": rationale,
    }


async def enrich_with_stage3_llm(diagnosis_df: pd.DataFrame, llm, max_windows: int = 120) -> pd.DataFrame:
    if diagnosis_df.empty:
        return diagnosis_df.copy()
    out = diagnosis_df.copy()
    out["stage3_window_pred"] = out["window_pred"].astype(str)
    out["stage3_llm_window_pred_raw"] = out["window_pred"].astype(str)
    out["stage3_failure_type"] = out["failure_type"].astype(str)
    out["stage3_impact_component"] = "unknown"
    out["stage3_rationale"] = out["concise_rationale"].astype(str)
    out["stage3_parse_ok"] = False

    n = min(len(out), max(1, int(max_windows)))
    for i in range(n):
        row = out.iloc[i]
        prompt = _build_stage3_prompt(row)
        resp = await llm.get_response(prompt, temperature=0.0)
        parsed = _parse_stage3_response(resp)
        if parsed:
            out.at[out.index[i], "stage3_llm_window_pred_raw"] = parsed.get("stage3_window_pred", row.get("window_pred", "normal"))
            for k, v in parsed.items():
                out.at[out.index[i], k] = v
            out.at[out.index[i], "stage3_parse_ok"] = True
            # Keep stage-3 robust for publishable stability:
            # LLM refines failure type + rationale, while window abnormal/normal
            # remains anchored to stage-2 evidence aggregation.
            out.at[out.index[i], "stage3_window_pred"] = str(row.get("window_pred", "normal"))
    return out


def summarize_stage3(diagnosis_stage3_df: pd.DataFrame) -> Dict:
    if diagnosis_stage3_df.empty:
        return {
            "windows": 0,
            "stage3_parse_rate": 0.0,
            "stage3_proxy_accuracy": 0.0,
            "stage3_proxy_precision": 0.0,
            "stage3_proxy_recall": 0.0,
            "stage3_proxy_f1": 0.0,
            "stage3_failure_type_distribution": {},
        }
    y_true = diagnosis_stage3_df["window_true_proxy"].astype(str).str.lower()
    y_pred = diagnosis_stage3_df["stage3_window_pred"].astype(str).str.lower()
    tp = int(((y_true == "abnormal") & (y_pred == "abnormal")).sum())
    fp = int(((y_true == "normal") & (y_pred == "abnormal")).sum())
    fn = int(((y_true == "abnormal") & (y_pred == "normal")).sum())
    tn = int(((y_true == "normal") & (y_pred == "normal")).sum())
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    parse_rate = float(diagnosis_stage3_df["stage3_parse_ok"].fillna(False).mean())
    return {
        "windows": int(total),
        "stage3_parse_rate": parse_rate,
        "stage3_proxy_accuracy": float(accuracy),
        "stage3_proxy_precision": float(precision),
        "stage3_proxy_recall": float(recall),
        "stage3_proxy_f1": float(f1),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "stage3_failure_type_distribution": diagnosis_stage3_df["stage3_failure_type"].value_counts().to_dict(),
    }


def export_stage2_outputs(predictions: pd.DataFrame, out_dir: str, cfg: DiagnosisConfig | None = None) -> Tuple[str, str]:
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    diag_df = build_diagnosis_windows(predictions=predictions, cfg=cfg)
    diag_file = folder / "diagnosis_windows.csv"
    diag_df.to_csv(diag_file, index=False)

    summary = summarize_diagnosis_windows(diag_df)
    summary_file = folder / "diagnosis_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return str(diag_file), str(summary_file)


async def export_stage3_outputs(
    predictions: pd.DataFrame,
    out_dir: str,
    llm,
    cfg: DiagnosisConfig | None = None,
) -> Tuple[str, str]:
    config = cfg or DiagnosisConfig()
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    base = build_diagnosis_windows(predictions=predictions, cfg=config)
    stage3_df = await enrich_with_stage3_llm(base, llm=llm, max_windows=config.stage3_max_windows)
    stage3_file = folder / "diagnosis_stage3_windows.csv"
    stage3_df.to_csv(stage3_file, index=False)
    summary = summarize_stage3(stage3_df)
    summary_file = folder / "diagnosis_stage3_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return str(stage3_file), str(summary_file)
