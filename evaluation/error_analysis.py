from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.metrics import confusion_matrix


def _to_binary(label: object) -> int:
    s = str(label).lower().strip()
    return 1 if s == "abnormal" or s == "1" else 0


def add_error_flags(df: pd.DataFrame, pred_col: str = "final_pred", gt_col: str = "label_str") -> pd.DataFrame:
    out = df.copy()
    out["y_true"] = out[gt_col].map(_to_binary)
    out["y_pred"] = out[pred_col].map(_to_binary)
    out["is_error"] = out["y_true"] != out["y_pred"]
    out["error_type"] = "TN"
    out.loc[(out["y_true"] == 1) & (out["y_pred"] == 1), "error_type"] = "TP"
    out.loc[(out["y_true"] == 0) & (out["y_pred"] == 1), "error_type"] = "FP"
    out.loc[(out["y_true"] == 1) & (out["y_pred"] == 0), "error_type"] = "FN"
    return out


def confusion_counts(df: pd.DataFrame) -> Tuple[int, int, int, int]:
    y_true = df["y_true"]
    y_pred = df["y_pred"]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tp), int(fp), int(fn), int(tn)
