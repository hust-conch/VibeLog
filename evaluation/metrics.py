from __future__ import annotations

from typing import Dict

import pandas as pd


def _to_binary_or_none(label: object):
    s = str(label).lower().strip()
    if s in {"abnormal", "1"}:
        return 1
    if s in {"normal", "0"}:
        return 0
    return None


def compute_metrics(df: pd.DataFrame, pred_col: str = "final_pred", gt_col: str = "label_str") -> Dict[str, float]:
    tp = fp = fn = tn = 0
    unknown_pred = 0

    for _, row in df.iterrows():
        y_true = _to_binary_or_none(row[gt_col])
        y_pred = _to_binary_or_none(row[pred_col])
        if pd.isna(y_true):
            continue

        if pd.isna(y_pred):
            unknown_pred += 1
            if y_true == 1:
                fn += 1
            else:
                fp += 1
            continue

        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "size": int(total),
        "abnormal_true": int(tp + fn),
        "abnormal_pred": int(tp + fp),
        "unknown_pred_count": int(unknown_pred),
        "unknown_pred_rate": float(unknown_pred / total) if total else 0.0,
    }
