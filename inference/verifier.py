from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class VerifierConfig:
    enable: bool = True


class RuleVerifier:
    STRONG_NORMAL = ["corrected", "recovered", "resumed", "synchronized", "retry succeeded"]
    STRONG_ABNORMAL = [
        "panic",
        "corrupt",
        "deadlock",
        "unrecoverable",
        "catastrophic",
        "kernel oops",
        "segfault",
        "i/o error",
    ]
    AMBIGUOUS = ["error", "fail", "fatal", "warning", "timeout"]
    SYSTEM_ENTITIES = ["kernel", "filesystem", "disk", "storage", "network", "node", "service", "daemon", "cluster"]
    SYSTEM_IMPACTS = ["unavailable", "panic", "corrupt", "deadlock", "unrecoverable", "i/o error", "service outage"]

    @classmethod
    def calibrate(
        cls,
        pred: str,
        reason: str,
        raw_log: str,
        normalized_log: str,
        parse_method: str = "",
        used_keyword_fallback: bool = False,
    ) -> Dict[str, str]:
        text = f"{raw_log} {normalized_log} {reason}".lower()
        hit_normal = [k for k in cls.STRONG_NORMAL if k in text]
        hit_abnormal = [k for k in cls.STRONG_ABNORMAL if k in text]
        hit_ambiguous = [k for k in cls.AMBIGUOUS if k in text]
        hit_entities = [k for k in cls.SYSTEM_ENTITIES if k in text]
        hit_impacts = [k for k in cls.SYSTEM_IMPACTS if k in text]
        final = pred
        decision = "keep"

        # 只做误报抑制：abnormal -> normal；不做 normal -> abnormal 的激进翻转
        if hit_normal and not hit_abnormal:
            if pred == "abnormal":
                final = "normal"
                decision = "rule_flip_to_normal"
        elif hit_ambiguous:
            # 歧义词仅在系统级证据充分时维持异常；否则作为误报抑制信号
            has_system_evidence = bool(hit_entities or hit_impacts)
            if pred == "abnormal" and not has_system_evidence:
                final = "normal"
                decision = "ambiguous_without_system_evidence_flip_to_normal"
            if (reason or "").strip() == "" or parse_method in {"parse_failed", "fallback_order_mapping"} or used_keyword_fallback:
                decision = "ambiguous_low_confidence"

        return {"final_pred": final, "verifier_action": decision}
