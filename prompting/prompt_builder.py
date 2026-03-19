from __future__ import annotations

import pandas as pd

from .instructions import get_output_constraint, get_universal_instruction


class PromptBuilder:
    def __init__(
        self,
        instruction: str | None = None,
        output_constraint: str | None = None,
        max_raw_chars: int = 260,
        max_norm_chars: int = 260,
        max_template_chars: int = 220,
    ):
        self.instruction = instruction or get_universal_instruction()
        self.output_constraint = output_constraint or get_output_constraint()
        self.max_raw_chars = max_raw_chars
        self.max_norm_chars = max_norm_chars
        self.max_template_chars = max_template_chars

    @staticmethod
    def _clip(text: object, n: int) -> str:
        s = str(text) if text is not None else ""
        return s[:n]

    def _format_example(self, i: int, row: pd.Series) -> str:
        template = self._clip(row.get("template", ""), self.max_template_chars)
        return (
            f"Example {i} ({row['dataset']}, label={row['label']}):\n"
            f"- raw_log: {self._clip(row['raw_log'], self.max_raw_chars)}\n"
            f"- normalized_log: {self._clip(row['normalized_log'], self.max_norm_chars)}\n"
            f"- template: {template}\n"
            f"- reason: {self._clip(row['reason'], 120)}\n"
        )

    def build_single_prompt(
        self,
        sample: pd.Series,
        retrieved_examples: pd.DataFrame,
        request_id: int = 1,
        micro_context: list[str] | None = None,
    ) -> str:
        few_shot_block = ""
        if not retrieved_examples.empty:
            chunks = [self._format_example(i + 1, row) for i, (_, row) in enumerate(retrieved_examples.iterrows())]
            few_shot_block = "Cross-system few-shot examples:\n" + "\n".join(chunks) + "\n"

        template = self._clip(sample.get("template", ""), self.max_template_chars)
        context_block = ""
        if micro_context:
            clipped = [self._clip(x, self.max_raw_chars) for x in micro_context if str(x).strip()]
            if clipped:
                context_lines = "\n".join([f"- {x}" for x in clipped])
                context_block = f"micro_context_logs:\n{context_lines}\n"
        query_block = (
            "Now classify the target log.\n"
            f"ID: {request_id}\n"
            f"target_dataset: {sample.get('dataset', 'unknown')}\n"
            f"raw_log: {self._clip(sample.get('raw_log', ''), self.max_raw_chars)}\n"
            f"normalized_log: {self._clip(sample.get('normalized_log', ''), self.max_norm_chars)}\n"
            f"template: {template}\n"
            f"{context_block}"
        )

        prompt = (
            f"{self.instruction}\n\n"
            f"{self.output_constraint}\n\n"
            f"{few_shot_block}"
            f"{query_block}"
        )
        return prompt
