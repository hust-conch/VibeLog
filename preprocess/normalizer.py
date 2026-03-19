from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .cleaners import clean_text
from .template_builder import build_lightweight_template


@dataclass
class CanonicalizeConfig:
    add_template: bool = False
    keep_raw: bool = True


class LogCanonicalizer:
    IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    HEX_CANDIDATE = re.compile(r"\b(?:0x[0-9a-fA-F]+|[0-9a-fA-F]{8,})\b")
    UUID = re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    )
    PID = re.compile(r"\b(?:pid|tid|thread|proc|process)[=:]?\d+\b", re.IGNORECASE)
    PATH = re.compile(r"(?:/[A-Za-z0-9._-]+){2,}")
    DATETIME = re.compile(
        r"\b\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?\b"
    )
    TIME_ONLY = re.compile(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b")
    NUM = re.compile(r"\b\d+\b")
    SPACES = re.compile(r"\s+")

    def normalize_log(self, raw_log: str) -> str:
        s = clean_text(raw_log)
        s = self.DATETIME.sub("<DATETIME>", s)
        s = self.TIME_ONLY.sub("<TIME>", s)
        s = self.UUID.sub("<UUID>", s)
        s = self.IPV4.sub("<IP>", s)
        s = self.PATH.sub("<PATH>", s)
        s = self.PID.sub("<PID>", s)
        s = self.HEX_CANDIDATE.sub(self._replace_hex_token, s)
        s = self.NUM.sub("<NUM>", s)
        s = self.SPACES.sub(" ", s).strip()
        return s

    @staticmethod
    def _replace_hex_token(match: re.Match) -> str:
        token = match.group(0)
        if token.lower().startswith("0x"):
            return "<HEX>"
        if any(ch in "abcdefABCDEF" for ch in token):
            return "<HEX>"
        return token

    def canonicalize(
        self,
        df: pd.DataFrame,
        raw_col: str = "raw_log",
        template_col: Optional[str] = None,
        config: Optional[CanonicalizeConfig] = None,
    ) -> pd.DataFrame:
        cfg = config or CanonicalizeConfig()
        if raw_col not in df.columns:
            raise ValueError(f"Missing raw log column: {raw_col}")

        out = df.copy()
        out["normalized_log"] = out[raw_col].astype(str).map(self.normalize_log)

        # 模板增强策略：
        # 1) 输入里已有 template 列 -> 直接保留
        # 2) 否则在 add_template=True 时用轻量模板器补齐
        if template_col and template_col in out.columns:
            out["template"] = out[template_col].astype(str)
        elif cfg.add_template:
            out["template"] = out["normalized_log"].map(build_lightweight_template)

        if not cfg.keep_raw and raw_col == "raw_log":
            out = out.drop(columns=["raw_log"])
        return out
