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
    normal_ratio: float = 0.67
    max_per_source_dataset: int = 2


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

    def _take_with_system_cap(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        if k <= 0 or df.empty:
            return df.head(0).copy()
        picked_rows = []
        system_counter = {}
        for _, row in df.sort_values("sim_score", ascending=False).iterrows():
            ds = str(row.get("dataset", "unknown")).lower()
            cnt = system_counter.get(ds, 0)
            if cnt >= self.config.max_per_source_dataset:
                continue
            picked_rows.append(row.to_dict())
            system_counter[ds] = cnt + 1
            if len(picked_rows) >= k:
                break
        return pd.DataFrame(picked_rows)

    def _apply_global_system_cap(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        if df.empty:
            return df
        picked_rows = []
        system_counter = {}
        for _, row in df.sort_values("sim_score", ascending=False).iterrows():
            ds = str(row.get("dataset", "unknown")).lower()
            cnt = system_counter.get(ds, 0)
            if cnt >= self.config.max_per_source_dataset:
                continue
            picked_rows.append(row.to_dict())
            system_counter[ds] = cnt + 1
            if len(picked_rows) >= k:
                break
        return pd.DataFrame(picked_rows)

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

        # 轻量偏置控制：normal 主导 + 少量 abnormal 对照
        k = self.config.k_total
        normal_k = math.ceil(k * self.config.normal_ratio)
        abnormal_k = max(0, k - normal_k)

        normal_pool = pool[pool["label"] == "normal"].copy()
        abnormal_pool = pool[pool["label"] == "abnormal"].copy()

        normal_top = self._take_with_system_cap(normal_pool, normal_k)
        abnormal_top = self._take_with_system_cap(abnormal_pool, abnormal_k)
        picked = pd.concat([normal_top, abnormal_top], ignore_index=True)
        if len(picked) < k:
            used = set(picked["bank_index"].tolist()) if "bank_index" in picked.columns else set()
            remain = pool[~pool["bank_index"].isin(used)].copy()
            remain_top = self._take_with_system_cap(remain, k - len(picked))
            picked = pd.concat([picked, remain_top], ignore_index=True)

        picked = self._apply_global_system_cap(picked, k)
        if len(picked) < k:
            used = set(picked["bank_index"].tolist()) if "bank_index" in picked.columns else set()
            remain2 = pool[~pool["bank_index"].isin(used)].copy()
            fill = self._take_with_system_cap(remain2, k - len(picked))
            picked = pd.concat([picked, fill], ignore_index=True)
            picked = self._apply_global_system_cap(picked, k)

        return picked.head(k).drop(columns=["bank_index"], errors="ignore").reset_index(drop=True)
