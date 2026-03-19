from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from inference.llm_service import AsyncVLLMService
from inference.parser import ResponseParser
from inference.verifier import RuleVerifier, VerifierConfig
from prompting.prompt_builder import PromptBuilder
from prompting.retriever import CrossSystemRetriever, RetrieverConfig


@dataclass
class InferenceConfig:
    concurrency_limit: int = 12
    parse_retry_limit: int = 1
    use_rule_verifier: bool = True
    few_shot_k: int = 6


async def _infer_single(
    sem: asyncio.Semaphore,
    llm: AsyncVLLMService,
    sample: pd.Series,
    example_bank: pd.DataFrame,
    prompt_builder: PromptBuilder,
    retriever: CrossSystemRetriever,
    cfg: InferenceConfig,
) -> dict:
    examples = retriever.retrieve(
        query_row=sample,
        example_bank=example_bank,
        target_dataset=sample["dataset"],
    )
    prompt = prompt_builder.build_single_prompt(sample, examples, request_id=1)

    async with sem:
        raw_response = await llm.get_response(prompt, temperature=0.0)

    parsed = ResponseParser.parse(raw_response, [1])
    retry_count = 0
    while len(parsed) < 1 and retry_count < cfg.parse_retry_limit:
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
        parsed.update(retry_parsed)
        if retry_response:
            raw_response = retry_response
        retry_count += 1

    out = parsed.get(1, {"pred": "unknown", "explanation": "Parsing Failed"})
    pred = out["pred"]
    if pred not in ("normal", "abnormal"):
        pred = ResponseParser.keyword_fallback(sample.get("normalized_log", sample.get("raw_log", "")))

    verifier_action = "disabled"
    final_pred = pred
    if cfg.use_rule_verifier:
        ver_res = RuleVerifier.calibrate(
            pred=pred,
            reason=out.get("explanation", ""),
            raw_log=sample.get("raw_log", ""),
            normalized_log=sample.get("normalized_log", ""),
        )
        final_pred = ver_res["final_pred"]
        verifier_action = ver_res["verifier_action"]

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
        "reason": out.get("explanation", ""),
        "verifier_action": verifier_action,
        "raw_response": raw_response,
        "few_shot_count": len(examples),
    }


async def run_inference_async(
    samples: pd.DataFrame,
    example_bank: pd.DataFrame,
    llm: AsyncVLLMService,
    cfg: Optional[InferenceConfig] = None,
) -> pd.DataFrame:
    config = cfg or InferenceConfig()
    sem = asyncio.Semaphore(config.concurrency_limit)
    prompt_builder = PromptBuilder()
    retriever = CrossSystemRetriever(RetrieverConfig(k_total=config.few_shot_k))

    tasks = [
        _infer_single(
            sem=sem,
            llm=llm,
            sample=row,
            example_bank=example_bank,
            prompt_builder=prompt_builder,
            retriever=retriever,
            cfg=config,
        )
        for _, row in samples.iterrows()
    ]
    results = await tqdm_asyncio.gather(*tasks)
    return pd.DataFrame(results)
