from __future__ import annotations

from typing import Tuple

import pandas as pd


def _to_binary_or_none(label: object):
    s = str(label).lower().strip()
    if s in {"abnormal", "1"}:
        return 1
    if s in {"normal", "0"}:
        return 0
    return None


def add_error_flags(df: pd.DataFrame, pred_col: str = "final_pred", gt_col: str = "label_str") -> pd.DataFrame:
    out = df.copy()
    out["y_true"] = out[gt_col].map(_to_binary_or_none)
    out["y_pred"] = out[pred_col].map(_to_binary_or_none)

    def _judge(row):
        yt, yp = row["y_true"], row["y_pred"]
        yt_missing = pd.isna(yt)
        yp_missing = pd.isna(yp)
        if yt_missing:
            return "INVALID"
        if yt == 1 and yp == 1:
            return "TP"
        if yt == 0 and yp == 0:
            return "TN"
        if yt == 0 and yp == 1:
            return "FP"
        if yt == 1 and yp == 0:
            return "FN"
        if yt == 0 and yp_missing:
            return "FP_UNKNOWN"
        if yt == 1 and yp_missing:
            return "FN_UNKNOWN"
        return "INVALID"

    out["error_type"] = out.apply(_judge, axis=1)
    out["is_error"] = ~out["error_type"].isin(["TP", "TN"])
    return out


def confusion_counts(df: pd.DataFrame) -> Tuple[int, int, int, int]:
    counts = df["error_type"].value_counts()
    tp = int(counts.get("TP", 0))
    tn = int(counts.get("TN", 0))
    fp = int(counts.get("FP", 0) + counts.get("FP_UNKNOWN", 0))
    fn = int(counts.get("FN", 0) + counts.get("FN_UNKNOWN", 0))
    return tp, fp, fn, tn
