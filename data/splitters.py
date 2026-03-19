from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def random_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train/dev/test ratios must sum to 1.")
    if len(df) == 0:
        out = df.copy()
        out["split"] = []
        return out

    tmp_ratio = dev_ratio + test_ratio
    stratify = df["label"] if df["label"].nunique() > 1 else None
    train_df, tmp_df = train_test_split(
        df, test_size=tmp_ratio, random_state=seed, stratify=stratify
    )

    if len(tmp_df) == 0:
        train_df = train_df.copy()
        train_df["split"] = "train"
        return train_df

    dev_part = dev_ratio / tmp_ratio if tmp_ratio > 0 else 0.5
    stratify_tmp = tmp_df["label"] if tmp_df["label"].nunique() > 1 else None
    dev_df, test_df = train_test_split(
        tmp_df, test_size=1 - dev_part, random_state=seed, stratify=stratify_tmp
    )

    train_df = train_df.copy()
    dev_df = dev_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    dev_df["split"] = "dev"
    test_df["split"] = "test"
    return pd.concat([train_df, dev_df, test_df], ignore_index=True)


def loso_split(df: pd.DataFrame, holdout_dataset: str) -> pd.DataFrame:
    holdout = holdout_dataset.lower()
    out = df.copy()
    out["split"] = out["dataset"].apply(lambda d: "test" if d == holdout else "train")
    return out


def get_loso_views(df: pd.DataFrame, holdout_dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_df = loso_split(df, holdout_dataset=holdout_dataset)
    train_df = split_df[split_df["split"] == "train"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()
    return train_df, test_df
