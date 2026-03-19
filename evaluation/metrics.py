from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _to_binary(label: object) -> int:
    s = str(label).lower().strip()
    return 1 if s == "abnormal" or s == "1" else 0


def compute_metrics(df: pd.DataFrame, pred_col: str = "final_pred", gt_col: str = "label_str") -> Dict[str, float]:
    y_true = df[gt_col].map(_to_binary)
    y_pred = df[pred_col].map(_to_binary)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "size": int(len(df)),
        "abnormal_true": int(y_true.sum()),
        "abnormal_pred": int(y_pred.sum()),
    }
