from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List

from data.adapters import load_datasets
from data.splitters import loso_split, random_split
from evaluation.reporter import create_run_dir, save_report
from preprocess.normalizer import CanonicalizeConfig, LogCanonicalizer
from prompting.example_bank import ExampleBankConfig, build_example_bank


def _parse_dataset_path_overrides(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected name=path")
        k, v = item.split("=", 1)
        out[k.strip().lower()] = v.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V1 pipeline: Adapter + Canonicalizer + Prompt Builder + Inference + Evaluator"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["bgl", "spirit", "thunderbird", "hdfs"],
        help="Datasets to load (supported: bgl/spirit/thunderbird/hdfs).",
    )
    parser.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        help="Optional path override in name=path format. Example: bgl=/abs/path/BGL_5k.xlsx",
    )
    parser.add_argument(
        "--split-mode",
        choices=["random", "loso", "none"],
        default="random",
        help="Data split mode.",
    )
    parser.add_argument(
        "--holdout-dataset",
        default="hdfs",
        help="Used when split-mode=loso.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-template", action="store_true")
    parser.add_argument("--run-mode", choices=["module12", "full"], default="full")
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--concurrency-limit", type=int, default=12)
    parser.add_argument("--few-shot-k", type=int, default=6)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--api-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--tokenizer-path", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument(
        "--output",
        default="project/artifacts",
        help="Output base dir for reports.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    overrides = _parse_dataset_path_overrides(args.dataset_path)

    df = load_datasets(args.datasets, dataset_paths=overrides)
    if args.split_mode == "random":
        df = random_split(df, seed=args.seed)
    elif args.split_mode == "loso":
        df = loso_split(df, holdout_dataset=args.holdout_dataset)

    canonicalizer = LogCanonicalizer()
    df = canonicalizer.canonicalize(
        df,
        raw_col="raw_log",
        config=CanonicalizeConfig(add_template=args.with_template, keep_raw=True),
    )

    if args.run_mode == "module12":
        out_file = Path(args.output) / "unified_canonicalized_samples.csv"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)
        print("=" * 60)
        print("Unified Pipeline (Module1+2) Completed")
        print("=" * 60)
        print(f"Rows: {len(df)}")
        print(f"Datasets: {sorted(df['dataset'].unique().tolist())}")
        print(f"Label distribution: {df['label_str'].value_counts().to_dict()}")
        print(f"Split distribution: {df['split'].value_counts().to_dict()}")
        print(f"Output saved to: {out_file}")
        return

    # Full V1 pipeline
    example_source = df[df["split"] != "test"] if "split" in df.columns else df
    example_bank = build_example_bank(
        example_source,
        config=ExampleBankConfig(seed=args.seed),
    )

    test_df = df[df["split"] == "test"].copy() if "split" in df.columns else df.copy()
    if args.max_test_samples is not None and args.max_test_samples < len(test_df):
        test_df = test_df.sample(n=args.max_test_samples, random_state=args.seed).reset_index(drop=True)

    from inference.llm_service import AsyncVLLMService, LLMConfig
    from inference.pipeline import InferenceConfig, run_inference_async

    llm_cfg = LLMConfig(
        model_name=args.model_name,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        tokenizer_path=args.tokenizer_path,
    )
    infer_cfg = InferenceConfig(
        concurrency_limit=args.concurrency_limit,
        few_shot_k=args.few_shot_k,
    )

    async def _run():
        llm = AsyncVLLMService(llm_cfg)
        try:
            return await run_inference_async(test_df, example_bank=example_bank, llm=llm, cfg=infer_cfg)
        finally:
            await llm.aclose()

    predictions = asyncio.run(_run())

    run_dir = create_run_dir(args.output, run_name="v1")
    cfg_dump = {
        "datasets": args.datasets,
        "dataset_path_overrides": overrides,
        "split_mode": args.split_mode,
        "holdout_dataset": args.holdout_dataset,
        "seed": args.seed,
        "with_template": args.with_template,
        "max_test_samples": args.max_test_samples,
        "concurrency_limit": args.concurrency_limit,
        "few_shot_k": args.few_shot_k,
        "model_name": args.model_name,
        "api_base_url": args.api_base_url,
        "run_mode": args.run_mode,
        "rows_input": len(df),
        "rows_test": len(test_df),
    }
    report = save_report(predictions=predictions, config_dict=cfg_dump, output_dir=str(run_dir))

    print("=" * 60)
    print("Unified Pipeline V1 Completed")
    print("=" * 60)
    print(f"Run dir: {run_dir}")
    print(f"Overall metrics: {json.dumps(report['overall_metrics'], ensure_ascii=False)}")
    print(f"Summary file: {report['summary']}")


if __name__ == "__main__":
    main()
