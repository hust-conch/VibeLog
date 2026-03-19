from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Optional

import httpx
from openai import AsyncOpenAI
from transformers import AutoTokenizer


@dataclass
class LLMConfig:
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    api_base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    tokenizer_path: str = "Qwen/Qwen2.5-14B-Instruct"
    timeout_sec: int = 90
    retry_times: int = 3
    retry_base_delay: float = 1.0
    max_tokens: int = 512
    disable_env_proxy: bool = True


class AsyncVLLMService:
    def __init__(self, config: LLMConfig):
        self.config = config
        http_client: Optional[httpx.AsyncClient] = None
        if config.disable_env_proxy:
            http_client = httpx.AsyncClient(trust_env=False)
            self.client = AsyncOpenAI(
                base_url=config.api_base_url,
                api_key=config.api_key,
                http_client=http_client,
            )
        else:
            self.client = AsyncOpenAI(base_url=config.api_base_url, api_key=config.api_key)
        self._http_client = http_client
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)
        except Exception:
            self.tokenizer = None

    async def aclose(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()

    async def get_response(self, prompt: str, temperature: float = 0.0) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant for cross-system log anomaly analysis."},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(self.config.retry_times + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout_sec,
                )
                if not response or not response.choices:
                    return ""
                return response.choices[0].message.content or ""
            except Exception as e:
                status_code = getattr(e, "status_code", None)
                err_text = str(e).lower()
                retryable = (
                    status_code in {429, 500, 502, 503, 504}
                    or "502" in err_text
                    or "timeout" in err_text
                    or "connection" in err_text
                )
                if retryable and attempt < self.config.retry_times:
                    delay = self.config.retry_base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    await asyncio.sleep(delay)
                    continue
                return ""
