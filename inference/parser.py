from __future__ import annotations

import json
import re
from typing import Dict, List


class ResponseParser:
    LINE_PATTERN = re.compile(
        r"^\s*[\*\-\[#\(\{<\s]*(\d+)[\*\-\]#\)\}>:\.\s]*(?:label|prediction|class)?\s*[:：-]?\s*\**\s*(normal|abnormal)\b\s*(?:-|:)?\s*(.*)$",
        re.IGNORECASE,
    )
    LABEL_PATTERN = re.compile(r"\b(normal|abnormal)\b", re.IGNORECASE)

    @staticmethod
    def _safe_json_loads(text):
        try:
            return json.loads(text)
        except Exception:
            return None

    @staticmethod
    def _parse_with_lines(response: str) -> List[tuple]:
        parsed_items = []
        for raw_line in response.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            match = ResponseParser.LINE_PATTERN.search(line)
            if not match:
                continue
            try:
                local_id = int(match.group(1))
                pred_label = match.group(2).lower()
                reason = match.group(3).strip()
                parsed_items.append((local_id, pred_label, reason, line, "line"))
            except ValueError:
                continue
        return parsed_items

    @staticmethod
    def _parse_with_json(response: str) -> List[tuple]:
        parsed_items = []
        candidates = [response.strip()]
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", response, flags=re.IGNORECASE)
        candidates.extend(fenced)
        for cand in candidates:
            data = ResponseParser._safe_json_loads(cand.strip())
            if data is None:
                continue
            items = data if isinstance(data, list) else [data]
            for obj in items:
                if not isinstance(obj, dict):
                    continue
                idx = obj.get("id", obj.get("index", obj.get("no")))
                pred = obj.get("prediction", obj.get("label", obj.get("class")))
                if idx is None or pred is None:
                    continue
                try:
                    idx = int(idx)
                except Exception:
                    continue
                pred_label = str(pred).lower().strip()
                if pred_label in ("normal", "abnormal"):
                    reason = obj.get("reason", obj.get("explanation", obj.get("rationale", "")))
                    parsed_items.append((idx, pred_label, str(reason).strip(), str(obj), "json"))
            if parsed_items:
                break
        return parsed_items

    @staticmethod
    def _map_items(parsed_items: List[tuple], expected_indices: List[int]) -> Dict[int, dict]:
        results = {}
        if not parsed_items:
            return results
        use_positional_mapping = False
        first_model_idx = parsed_items[0][0]
        if first_model_idx == 1 and expected_indices and expected_indices[0] != 1:
            use_positional_mapping = True

        for i, (model_idx, pred_label, reason, raw_line, parse_method) in enumerate(parsed_items):
            if use_positional_mapping:
                if i >= len(expected_indices):
                    continue
                global_idx = expected_indices[i]
            else:
                if model_idx in expected_indices:
                    global_idx = model_idx
                elif i < len(expected_indices):
                    global_idx = expected_indices[i]
                else:
                    continue
            if global_idx not in results:
                results[global_idx] = {
                    "pred": pred_label,
                    "reason": reason,
                    "raw_parsed_line": raw_line,
                    "parse_method": parse_method,
                }
        return results

    @staticmethod
    def keyword_fallback(log_text: str) -> str:
        text = str(log_text).lower()
        abnormal_keys = [
            "fatal",
            "panic",
            "catastrophic",
            "corrupt",
            "deadlock",
            "segfault",
            "kernel oops",
            "unrecoverable",
            "service timeout",
            "i/o error",
        ]
        return "abnormal" if any(k in text for k in abnormal_keys) else "normal"

    @staticmethod
    def parse(response: str, expected_indices: List[int]) -> Dict[int, dict]:
        if not response:
            return {}
        parsed_items = ResponseParser._parse_with_lines(response)
        results = ResponseParser._map_items(parsed_items, expected_indices)
        if len(results) >= len(expected_indices):
            return results

        json_items = ResponseParser._parse_with_json(response)
        json_mapped = ResponseParser._map_items(json_items, expected_indices)
        for k, v in json_mapped.items():
            if k not in results:
                results[k] = v
        if len(results) >= len(expected_indices):
            return results

        order_labels = [m.group(1).lower() for m in ResponseParser.LABEL_PATTERN.finditer(response)]
        for i, idx in enumerate(expected_indices):
            if idx in results:
                continue
            if i < len(order_labels):
                results[idx] = {
                    "pred": order_labels[i],
                    "reason": "",
                    "raw_parsed_line": "",
                    "parse_method": "fallback_order_mapping",
                }
        return results
