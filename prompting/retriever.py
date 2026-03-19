from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List

import pandas as pd


TOKEN_PATTERN = re.compile(r"[a-zA-Z_<>]+")


@dataclass
class RetrieverConfig:
    k_total: int = 6
    seed: int = 42
    min_per_label: int = 2


def _tokenize(text: str) -> set:
    return set(TOKEN_PATTERN.findall(str(text).lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(union, 1)


def _score_example(query_norm: str, query_template: str, cand_norm: str, cand_template: str) -> float:
    q1 = _tokenize(query_norm)
    q2 = _tokenize(query_template)
    c1 = _tokenize(cand_norm)
    c2 = _tokenize(cand_template)
    return 0.7 * _jaccard(q1, c1) + 0.3 * _jaccard(q2, c2)


class CrossSystemRetriever:
    def __init__(self, config: RetrieverConfig | None = None):
        self.config = config or RetrieverConfig()

    def retrieve(self, query_row: pd.Series, example_bank: pd.DataFrame, target_dataset: str) -> pd.DataFrame:
        if example_bank.empty:
            return example_bank.copy()

        pool = example_bank[example_bank["dataset"].str.lower() != target_dataset.lower()].copy()
        if pool.empty:
            return pool
        pool["bank_index"] = pool.index

        query_norm = query_row.get("normalized_log", query_row.get("raw_log", ""))
        query_template = query_row.get("template", "")
        pool["sim_score"] = pool.apply(
            lambda r: _score_example(
                query_norm=query_norm,
                query_template=query_template,
                cand_norm=r.get("normalized_log", ""),
                cand_template=r.get("template", ""),
            ),
            axis=1,
        )

        # 轻量平衡抽取：normal/abnormal 各取一半，至少 min_per_label
        k = self.config.k_total
        per_label = max(self.config.min_per_label, math.floor(k / 2))
        normal_top = pool[pool["label"] == "normal"].sort_values("sim_score", ascending=False).head(per_label)
        abnormal_top = pool[pool["label"] == "abnormal"].sort_values("sim_score", ascending=False).head(per_label)

        picked = pd.concat([normal_top, abnormal_top], ignore_index=True)
        if len(picked) < k:
            used = set(picked["bank_index"].tolist()) if "bank_index" in picked.columns else set()
            remain = pool[~pool["bank_index"].isin(used)].sort_values("sim_score", ascending=False)
            picked = pd.concat([picked, remain.head(k - len(picked))], ignore_index=True)

        return picked.head(k).drop(columns=["bank_index"], errors="ignore").reset_index(drop=True)
