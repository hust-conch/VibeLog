from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class ExampleBankConfig:
    max_per_dataset_label: int = 80
    seed: int = 42
    allowed_splits: Tuple[str, ...] = ("train",)
    drop_duplicate_keys: bool = True


ABNORMAL_HINTS = ["panic", "deadlock", "corrupt", "unrecoverable", "catastrophic", "i/o error"]
NORMAL_HINTS = ["info", "warn", "corrected", "recovered", "resumed", "retry", "synchronized"]


def _reason_from_log(log_text: str, label: str) -> str:
    s = str(log_text).lower()
    if label == "abnormal":
        for kw in ABNORMAL_HINTS:
            if kw in s:
                return f"System-level failure signal: {kw}."
        return "System-level failure evidence in log semantics."
    for kw in NORMAL_HINTS:
        if kw in s:
            return f"Non-critical/recoverable signal: {kw}."
    return "No clear system-level fatal evidence."


def build_example_bank(df: pd.DataFrame, config: ExampleBankConfig | None = None) -> pd.DataFrame:
    cfg = config or ExampleBankConfig()
    required = {"dataset", "raw_log", "normalized_log", "label_str"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for example bank: {sorted(missing)}")

    frame = df.copy()
    frame = frame[frame["label_str"].isin(["normal", "abnormal"])].copy()
    if "split" in frame.columns:
        frame = frame[frame["split"].isin(cfg.allowed_splits)].copy()
    if cfg.drop_duplicate_keys and not frame.empty:
        frame["_dedup_key"] = (
            frame["dataset"].astype(str)
            + "||"
            + frame["normalized_log"].astype(str)
            + "||"
            + frame["label_str"].astype(str)
        )
        frame = frame.drop_duplicates("_dedup_key").copy()
    if "template" not in frame.columns:
        frame["template"] = ""

    frame["reason"] = frame.apply(
        lambda r: _reason_from_log(r.get("normalized_log", r["raw_log"]), r["label_str"]),
        axis=1,
    )

    sampled_parts: List[pd.DataFrame] = []
    for dataset_name, g_dataset in frame.groupby("dataset", sort=True):
        for label_name, g_label in g_dataset.groupby("label_str", sort=True):
            n = min(cfg.max_per_dataset_label, len(g_label))
            sampled_parts.append(g_label.sample(n=n, random_state=cfg.seed) if len(g_label) > n else g_label)

    out = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else frame.head(0).copy()
    out = out[["dataset", "raw_log", "normalized_log", "template", "label_str", "reason"]]
    out = out.rename(columns={"label_str": "label"})
    return out
