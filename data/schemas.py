from dataclasses import dataclass
from typing import Dict, Optional


UNIFIED_COLUMNS = [
    "sample_id",
    "dataset",
    "source_index",
    "raw_log",
    "label_str",
    "label",
    "split",
]


@dataclass
class DatasetConfig:
    name: str
    path: str
    log_col: str = "log"
    label_col: str = "label"
    id_col: Optional[str] = None
    default_split: str = "unspecified"


def normalize_label(raw_label: object) -> Dict[str, object]:
    s = str(raw_label).strip().lower()
    if s in {"abnormal", "anomaly", "1", "1.0", "true", "yes"}:
        return {"label_str": "abnormal", "label": 1}
    if s in {"normal", "0", "0.0", "false", "no"}:
        return {"label_str": "normal", "label": 0}
    return {"label_str": "unknown", "label": -1}
