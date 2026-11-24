from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class VLLMGeneration:
    text: str
    token_ids: list[int]
    logprobs: list[float]
    latency_ms: float


class VLLMRolloutClient:
    """Minimal HTTP client for vLLM's OpenAI-compatible server with exact token-ID return.

    Uses `/v1/completions` with `prompt` as token IDs and `return_tokens_as_token_ids=True`
    so the trainer receives the exact token IDs and rollout logprobs sampled by vLLM.
    """

    def __init__(self, base_url: str, model: str, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout_s)

    def generate_from_token_ids(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.8,
        stop: list[str] | None = None,
    ) -> VLLMGeneration:
        t0 = time.perf_counter()
        payload = {
            "model": self.model,
            "prompt": prompt_token_ids,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": 1,
            "return_tokens_as_token_ids": True,
            "stop": stop or ["</tool>"],
            "include_stop_str_in_output": True,
        }
        resp = self._client.post(f"{self.base_url}/v1/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        lp_block = choice["logprobs"]
        raw_tokens: list[str] = lp_block["tokens"]
        raw_lps: list[float] = [float(x) for x in lp_block["token_logprobs"]]

        token_ids: list[int] = []
        for tok in raw_tokens:
            # vLLM encodes token IDs as "token_id:12345" when return_tokens_as_token_ids=True.
            if tok.startswith("token_id:"):
                token_ids.append(int(tok.split(":", 1)[1]))
            else:
                raise ValueError(
                    f"Expected vLLM token_id format, got {tok!r}. "
                    "Ensure vLLM server supports return_tokens_as_token_ids."
                )

        return VLLMGeneration(
            text=str(choice["text"]),
            token_ids=token_ids,
            logprobs=raw_lps,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def load_lora_adapter(self, lora_name: str, lora_path: str) -> None:
        """Hot-reload an updated LoRA checkpoint into a running vLLM server (~200ms)."""
        resp = self._client.post(
            f"{self.base_url}/v1/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path},
        )
        resp.raise_for_status()

    def close(self) -> None:
        self._client.close()
