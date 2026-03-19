from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .schemas import DatasetConfig, UNIFIED_COLUMNS, normalize_label


DEFAULT_DATASET_PATHS: Dict[str, str] = {
    "bgl": "LogPrompt/处理后数据集/BGL_5k.xlsx",
    "spirit": "LogPrompt/处理后数据集/Spirit_5k.xlsx",
    "thunderbird": "LogPrompt/处理后数据集/Thunderbird_5k.xlsx",
    "hdfs": "LogPrompt/处理后数据集/HDFS_5k.xlsx",
}


def _resolve_id_series(df: pd.DataFrame, id_col: Optional[str]) -> pd.Series:
    if id_col and id_col in df.columns:
        return df[id_col]
    if "log_id" in df.columns:
        return df["log_id"]
    if "original_id" in df.columns:
        return df["original_id"]
    return pd.Series(range(1, len(df) + 1), index=df.index)


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        repo_root = Path(__file__).resolve().parents[2]
        p2 = repo_root / path
        if p2.exists():
            p = p2
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if p.suffix.lower() in {".csv"}:
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file type: {p.suffix}")


def adapt_dataset(config: DatasetConfig) -> pd.DataFrame:
    df = _read_table(config.path)
    if config.log_col not in df.columns:
        raise ValueError(f"[{config.name}] missing log column: {config.log_col}")
    if config.label_col not in df.columns:
        raise ValueError(f"[{config.name}] missing label column: {config.label_col}")

    id_series = _resolve_id_series(df, config.id_col)
    logs = df[config.log_col].fillna("").astype(str)
    labels = df[config.label_col]

    unified_rows: List[dict] = []
    for i in df.index:
        norm = normalize_label(labels.loc[i])
        src_idx = id_series.loc[i]
        unified_rows.append(
            {
                "sample_id": f"{config.name}_{src_idx}",
                "dataset": config.name.lower(),
                "source_index": src_idx,
                "raw_log": logs.loc[i].strip(),
                "label_str": norm["label_str"],
                "label": norm["label"],
                "split": config.default_split,
            }
        )

    out = pd.DataFrame(unified_rows)
    out = out[UNIFIED_COLUMNS]
    return out


def build_dataset_configs(
    dataset_names: Iterable[str],
    dataset_paths: Optional[Dict[str, str]] = None,
) -> List[DatasetConfig]:
    paths = DEFAULT_DATASET_PATHS.copy()
    if dataset_paths:
        for k, v in dataset_paths.items():
            paths[k.lower()] = v

    configs: List[DatasetConfig] = []
    for name in dataset_names:
        key = name.lower()
        if key not in paths:
            raise ValueError(f"Unsupported dataset: {name}")
        configs.append(DatasetConfig(name=key, path=paths[key]))
    return configs


def load_datasets(
    dataset_names: Iterable[str],
    dataset_paths: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    configs = build_dataset_configs(dataset_names, dataset_paths=dataset_paths)
    frames = [adapt_dataset(cfg) for cfg in configs]
    if not frames:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    return merged
