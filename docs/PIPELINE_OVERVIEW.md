# Pipeline Overview (V1)

## Main Entry
- Preferred entry: `project/experiments/run_experiment.py`
- Backward-compatible entry: `project/run_experiment.py` (wrapper)

## Stages
1. Data Adapter (`project/data`)
   - Load BGL/Spirit/Thunderbird/HDFS into unified schema.
2. Canonicalizer (`project/preprocess`)
   - Produce `normalized_log` (and optional template).
3. Prompting (`project/prompting`)
   - Build cross-system few-shot prompts.
4. Inference Line Layer (`project/inference/pipeline.py`)
   - LLM prediction + robust parser + verifier + evidence fields.
5. Diagnosis Stage-2 (`project/inference/diagnosis.py`)
   - Window aggregation + top-k evidence + failure type.
6. Diagnosis Stage-3 (`project/inference/diagnosis.py`)
   - Window-level LLM diagnosis enrichment (type/component/rationale).
7. Evaluation (`project/evaluation`)
   - Metrics, error analysis, plots, reports.

## Output
- Default output root: `project/artifacts/`
- Per run folder: `v1_YYYYMMDD_HHMMSS/system_mode/`

