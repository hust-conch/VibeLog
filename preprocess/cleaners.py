import re


CONTROL_CHARS = re.compile(r"[\x00-\x1F\x7F]")
MULTI_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    s = CONTROL_CHARS.sub(" ", s)
    s = MULTI_SPACES.sub(" ", s).strip()
    return s
