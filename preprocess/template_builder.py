from __future__ import annotations

import re


_NUM_TOKEN = re.compile(r"\b\d+\b")
_HEX_TOKEN = re.compile(r"\b(?:0x[0-9a-fA-F]+|[0-9a-fA-F]{6,})\b")
_UUID_TOKEN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


def build_lightweight_template(normalized_log: str) -> str:
    s = normalized_log
    s = _UUID_TOKEN.sub("<UUID>", s)
    s = _HEX_TOKEN.sub(_replace_hex_token, s)
    s = _NUM_TOKEN.sub("<NUM>", s)
    return s


def _replace_hex_token(match: re.Match) -> str:
    token = match.group(0)
    if token.lower().startswith("0x"):
        return "<HEX>"
    if any(ch in "abcdefABCDEF" for ch in token):
        return "<HEX>"
    return token
