from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from inference.llm_service import AsyncVLLMService
from inference.parser import ResponseParser
from inference.verifier import RuleVerifier
from prompting.prompt_builder import PromptBuilder
from prompting.instructions import get_universal_instruction
from prompting.retriever import CrossSystemRetriever, RetrieverConfig


@dataclass
class InferenceConfig:
    concurrency_limit: int = 12
    parse_retry_limit: int = 1
    use_rule_verifier: bool = True
    few_shot_k: int = 6
    retriever_normal_ratio: float = 0.67
    retriever_max_per_source_dataset: int = 2
    instruction_profile: str = "strict"
    task_mode: str = "evidence_diag"
    context_window: int = 3
    key_evidence_threshold: float = 60.0


def _needs_parse_retry(parsed: dict) -> bool:
    if len(parsed) < 1:
        return True
    item = parsed.get(1, {})
    method = item.get("parse_method", "")
    return method in {"fallback_order_mapping", "parse_failed"}


def _get_out(parsed: dict) -> dict:
    return parsed.get(
        1,
        {
            "pred": "unknown",
            "reason": "",
            "raw_parsed_line": "",
            "parse_method": "parse_failed",
        },
    )


EVIDENCE_TYPES = {
    "system_level_evidence",
    "application_level_evidence",
    "recovery_evidence",
    "context_evidence",
    "failure_evidence",
    "noise_benign_evidence",
}


def _extract_reason_metadata(reason_text: str) -> tuple[str, float, str]:
    reason = str(reason_text or "").strip()
    low = reason.lower()
    evidence_type = ""
    score = None

    et = re.search(r"evidence_type\s*=\s*([a-z_]+)", low)
    if et:
        cand = et.group(1).strip()
        if cand in EVIDENCE_TYPES:
            evidence_type = cand
    sc = re.search(r"score\s*=\s*(\d{1,3})", low)
    if sc:
        try:
            score = float(max(0, min(100, int(sc.group(1)))))
        except Exception:
            score = None
    rs = re.search(r"reason\s*=\s*(.*)$", reason, flags=re.IGNORECASE)
    clean_reason = rs.group(1).strip() if rs else reason
    clean_reason = re.sub(r"\bevidence_type\s*=\s*[a-z_]+\s*;?", "", clean_reason, flags=re.IGNORECASE).strip()
    clean_reason = re.sub(r"\bscore\s*=\s*\d{1,3}\s*;?", "", clean_reason, flags=re.IGNORECASE).strip()
    clean_reason = re.sub(r"^[;,\-\s]+", "", clean_reason).strip()
    return evidence_type, (score if score is not None else -1.0), clean_reason


def _infer_evidence_type(pred: str, reason: str, raw_log: str, normalized_log: str) -> str:
    reason_text = str(reason or "").lower()
    log_text = f"{raw_log} {normalized_log}".lower()
    if any(k in log_text for k in ["recovered", "corrected", "resumed", "retry succeeded", "back online", "restored"]):
        return "recovery_evidence"
    if "no clear system-level" in reason_text and pred == "normal":
        if any(k in log_text for k in ["app", "application", "user job", "process", "task", "worker", "client"]):
            return "application_level_evidence"
        if any(k in log_text for k in ["error", "failed", "fail", "fatal", "timeout", "exception"]):
            return "failure_evidence"
        return "noise_benign_evidence"
    if any(
        k in log_text
        for k in [
            "kernel",
            "deadlock",
            "corrupt",
            "unrecoverable",
            "panic",
            "i/o error",
            "service unavailable",
            "node down",
            "filesystem",
            "storage",
            "network outage",
        ]
    ):
        return "system_level_evidence"
    if any(k in log_text for k in ["app", "application", "user job", "process", "task", "worker", "client"]):
        return "application_level_evidence"
    if any(k in log_text for k in ["error", "failed", "fail", "fatal", "timeout", "exception"]):
        return "failure_evidence"
    if pred == "abnormal":
        return "context_evidence"
    if any(k in log_text for k in ["info", "debug", "heartbeat", "routine", "initialized", "started"]):
        return "noise_benign_evidence"
    return "context_evidence"


def _infer_relevance_score(pred: str, evidence_type: str, clean_reason: str, parsed_score: float) -> float:
    if parsed_score >= 0:
        return parsed_score
    base = {
        "system_level_evidence": 88,
        "failure_evidence": 72,
        "application_level_evidence": 58,
        "recovery_evidence": 44,
        "context_evidence": 42,
        "noise_benign_evidence": 18,
    }.get(evidence_type, 40)
    if pred == "abnormal":
        base += 6
    if len(str(clean_reason)) > 80:
        base += 2
    return float(max(0, min(100, base)))


def _build_context_map(samples: pd.DataFrame, context_window: int) -> dict:
    if context_window <= 0 or "sample_id" not in samples.columns:
        return {}
    out = {}
    for _, grp in samples.groupby("dataset", sort=False):
        g = grp.sort_values("sample_id")
        recs = g[["sample_id", "raw_log", "normalized_log"]].to_dict("records")
        n = len(recs)
        for i, row in enumerate(recs):
            sid = row["sample_id"]
            left = max(0, i - context_window)
            right = min(n, i + context_window + 1)
            ctx = []
            for j in range(left, right):
                if j == i:
                    continue
                one = recs[j]
                text = str(one.get("raw_log") or one.get("normalized_log") or "")
                if text.strip():
                    ctx.append(text)
            out[sid] = ctx
    return out


async def _infer_single(
    sem: asyncio.Semaphore,
    llm: AsyncVLLMService,
    sample: pd.Series,
    example_bank: pd.DataFrame,
    prompt_builder: PromptBuilder,
    retriever: CrossSystemRetriever,
    cfg: InferenceConfig,
    micro_context: list[str] | None = None,
) -> dict:
    examples = retriever.retrieve(
        query_row=sample,
        example_bank=example_bank,
        target_dataset=sample["dataset"],
    )
    prompt = prompt_builder.build_single_prompt(sample, examples, request_id=1, micro_context=micro_context)

    async with sem:
        raw_response = await llm.get_response(prompt, temperature=0.0)

    parsed = ResponseParser.parse(raw_response, [1])
    initial_out = _get_out(parsed)
    initial_parse_method = initial_out.get("parse_method", "parse_failed")
    initial_pred = initial_out.get("pred", "unknown")
    initial_raw_response = raw_response
    first_pass_fallback = initial_parse_method == "fallback_order_mapping"
    retry_changed_label = False

    retry_count = 0
    if initial_parse_method == "fallback_order_mapping":
        # 格式修复重试：禁止重判语义
        while _needs_parse_retry(parsed) and retry_count < cfg.parse_retry_limit:
            retry_prompt = (
                "Reformat the following answer into exactly one line.\n\n"
                "Keep the original decision unchanged.\n"
                "Do not re-evaluate the log.\n"
                "Do not change normal to abnormal or abnormal to normal.\n\n"
                "Required format:\n"
                "1. normal - reason\n"
                "or\n"
                "1. abnormal - reason\n\n"
                "Original answer:\n"
                f"{raw_response}"
            )
            async with sem:
                retry_response = await llm.get_response(retry_prompt, temperature=0.0)
            retry_parsed = ResponseParser.parse(retry_response, [1])
            if retry_response:
                raw_response = retry_response
            if retry_parsed:
                cand = retry_parsed.get(1, {})
                cand_pred = cand.get("pred", "")
                if initial_pred in {"normal", "abnormal"} and cand_pred in {"normal", "abnormal"} and cand_pred != initial_pred:
                    cand["pred"] = initial_pred
                    retry_changed_label = True
                retry_parsed[1] = cand
                parsed.update(retry_parsed)
            retry_count += 1
    else:
        # 解析失败重试：允许重新判别
        while _needs_parse_retry(parsed) and retry_count < cfg.parse_retry_limit:
            retry_prompt = (
                "Return exactly one line only:\n"
                "1. normal - reason\n"
                "or\n"
                "1. abnormal - reason\n"
                f"raw_log: {sample.get('raw_log', '')}\n"
                f"normalized_log: {sample.get('normalized_log', '')}\n"
                f"template: {sample.get('template', '')}\n"
            )
            async with sem:
                retry_response = await llm.get_response(retry_prompt, temperature=0.0)
            retry_parsed = ResponseParser.parse(retry_response, [1])
            if retry_parsed:
                parsed.update(retry_parsed)
            if retry_response:
                raw_response = retry_response
            retry_count += 1

    out = _get_out(parsed)
    pred = out["pred"]
    used_keyword_fallback = False
    fallback_source = ""
    if pred not in ("normal", "abnormal"):
        pred = ResponseParser.keyword_fallback(sample.get("normalized_log", sample.get("raw_log", "")))
        used_keyword_fallback = True
        fallback_source = "keyword_fallback"

    verifier_action = "disabled"
    final_pred = pred
    if cfg.use_rule_verifier:
        ver_res = RuleVerifier.calibrate(
            pred=pred,
            reason=out.get("reason", ""),
            raw_log=sample.get("raw_log", ""),
            normalized_log=sample.get("normalized_log", ""),
            parse_method=out.get("parse_method", ""),
            used_keyword_fallback=used_keyword_fallback,
        )
        final_pred = ver_res["final_pred"]
        verifier_action = ver_res["verifier_action"]

    parsed_evidence_type, parsed_score, clean_reason = _extract_reason_metadata(out.get("reason", ""))
    evidence_type = parsed_evidence_type or _infer_evidence_type(
        pred=final_pred,
        reason=clean_reason,
        raw_log=sample.get("raw_log", ""),
        normalized_log=sample.get("normalized_log", ""),
    )
    relevance_score = _infer_relevance_score(
        pred=final_pred,
        evidence_type=evidence_type,
        clean_reason=clean_reason,
        parsed_score=parsed_score,
    )
    is_key_evidence = bool(relevance_score >= float(cfg.key_evidence_threshold))

    return {
        "sample_id": sample["sample_id"],
        "dataset": sample["dataset"],
        "split": sample.get("split", "unspecified"),
        "label_str": sample.get("label_str", "unknown"),
        "raw_log": sample.get("raw_log", ""),
        "normalized_log": sample.get("normalized_log", ""),
        "template": sample.get("template", ""),
        "llm_pred": pred,
        "final_pred": final_pred,
        "reason": clean_reason or out.get("reason", ""),
        "raw_reason": out.get("reason", ""),
        "evidence_type": evidence_type,
        "relevance_score": relevance_score,
        "is_key_evidence": is_key_evidence,
        "diagnosis_label": "system_fault" if final_pred == "abnormal" else "non_system_fault",
        "parse_method": out.get("parse_method", ""),
        "raw_parsed_line": out.get("raw_parsed_line", ""),
        "initial_parse_method": initial_parse_method,
        "initial_pred": initial_pred,
        "initial_raw_response": initial_raw_response,
        "first_pass_fallback": first_pass_fallback,
        "retry_changed_label": retry_changed_label,
        "used_retry": retry_count > 0,
        "used_keyword_fallback": used_keyword_fallback,
        "fallback_source": fallback_source,
        "verifier_action": verifier_action,
        "raw_response": raw_response,
        "few_shot_count": len(examples),
        "context_size": len(micro_context or []),
    }


async def run_inference_async(
    samples: pd.DataFrame,
    example_bank: pd.DataFrame,
    llm: AsyncVLLMService,
    cfg: Optional[InferenceConfig] = None,
) -> pd.DataFrame:
    config = cfg or InferenceConfig()
    sem = asyncio.Semaphore(config.concurrency_limit)
    prompt_builder = PromptBuilder(
        instruction=get_universal_instruction(config.instruction_profile, config.task_mode),
    )
    retriever = CrossSystemRetriever(
        RetrieverConfig(
            k_total=config.few_shot_k,
            normal_ratio=config.retriever_normal_ratio,
            max_per_source_dataset=config.retriever_max_per_source_dataset,
        )
    )

    context_map = _build_context_map(samples, config.context_window)
    tasks = [
        _infer_single(
            sem=sem,
            llm=llm,
            sample=row,
            example_bank=example_bank,
            prompt_builder=prompt_builder,
            retriever=retriever,
            cfg=config,
            micro_context=context_map.get(row.get("sample_id"), []),
        )
        for _, row in samples.iterrows()
    ]
    results = await tqdm_asyncio.gather(*tasks)
    return pd.DataFrame(results)
