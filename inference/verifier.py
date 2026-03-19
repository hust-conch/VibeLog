from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class VerifierConfig:
    enable: bool = True


class RuleVerifier:
    NORMAL_BIAS = ["corrected", "recovered", "resumed", "synchronized", "retry succeeded"]
    ABNORMAL_BIAS = ["panic", "corrupt", "deadlock", "unrecoverable", "catastrophic", "kernel oops"]
    AMBIGUOUS = ["error", "fail", "fatal", "warning", "timeout"]

    @classmethod
    def calibrate(cls, pred: str, reason: str, raw_log: str, normalized_log: str) -> Dict[str, str]:
        text = f"{raw_log} {normalized_log} {reason}".lower()
        final = pred
        decision = "keep"

        if any(k in text for k in cls.NORMAL_BIAS):
            if pred == "abnormal":
                final = "normal"
                decision = "rule_flip_to_normal"

        if any(k in text for k in cls.ABNORMAL_BIAS):
            if pred == "normal":
                final = "abnormal"
                decision = "rule_flip_to_abnormal"

        ambiguous_hit = [k for k in cls.AMBIGUOUS if k in text]
        if ambiguous_hit and reason.strip() == "":
            decision = "ambiguous_low_reason"

        return {"final_pred": final, "verifier_action": decision}
