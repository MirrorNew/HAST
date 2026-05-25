# -*- coding: utf-8 -*-
"""LLM provider abstraction.

The framework keeps LLM calls behind this interface so experiments can be run
with real API calls, cached logs, or deterministic mock candidates.
"""

from __future__ import annotations

import json
import os
import random
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Any
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMProvider(Protocol):
    def generate(self, prompt: str, *, n: int = 1) -> list[str]:
        ...


@dataclass
class LoggedLLMConfig:
    model: str = "gpt-5.5"
    reasoning_effort: str = "none"
    temperature: float = 0.2
    base_url: str = "https://api.ritelt.com/v1"
    timeout_s: float = 90.0
    max_retries: int = 2
    retry_base_delay_s: float = 1.0


class OpenAICompatibleLLMProvider:
    """OpenAI-compatible chat-completions provider.

    Secrets are read from environment variables and never stored in repo files.
    Required env var: ``HAST_LLM_API_KEY`` or ``OPENAI_API_KEY``.
    Optional env vars: ``HAST_LLM_BASE_URL``, ``HAST_LLM_MODEL``.
    """

    def __init__(self, api_key: str, config: LoggedLLMConfig | None = None):
        if not api_key:
            raise ValueError("missing API key")
        self.api_key = api_key
        self.config = config or LoggedLLMConfig()

    @classmethod
    def from_env(cls) -> "OpenAICompatibleLLMProvider":
        config = LoggedLLMConfig(
            model=os.environ.get("HAST_LLM_MODEL", LoggedLLMConfig.model),
            reasoning_effort=os.environ.get("HAST_LLM_REASONING_EFFORT", LoggedLLMConfig.reasoning_effort),
            temperature=float(os.environ.get("HAST_LLM_TEMPERATURE", str(LoggedLLMConfig.temperature))),
            base_url=os.environ.get("HAST_LLM_BASE_URL", LoggedLLMConfig.base_url).rstrip("/"),
            timeout_s=float(os.environ.get("HAST_LLM_TIMEOUT_S", str(LoggedLLMConfig.timeout_s))),
            max_retries=int(os.environ.get("HAST_LLM_MAX_RETRIES", str(LoggedLLMConfig.max_retries))),
            retry_base_delay_s=float(
                os.environ.get("HAST_LLM_RETRY_BASE_DELAY_S", str(LoggedLLMConfig.retry_base_delay_s))
            ),
        )
        api_key = os.environ.get("HAST_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        return cls(api_key=api_key, config=config)

    def _chat_completions_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    def _request_once(self, prompt: str) -> str:
        body: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "reasoning_effort": self.config.reasoning_effort,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate Python network dismantling heuristics. "
                        "Return only one Python code block defining def degree_order(G)."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(
            self._chat_completions_url(),
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.config.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail[:500]}") from exc
        except URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        if "choices" in payload and payload["choices"]:
            message = payload["choices"][0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, list):
                parts = [item.get("text", "") for item in content if isinstance(item, dict)]
                return "\n".join(part for part in parts if part).strip()
            return str(content).strip()
        if "output_text" in payload:
            return str(payload["output_text"]).strip()
        raise RuntimeError(f"LLM response did not contain usable text: {str(payload)[:500]}")

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, socket.timeout, ssl.SSLError, URLError)):
            return True
        text = str(exc).lower()
        retry_markers = [
            "http 408",
            "http 409",
            "http 425",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "timed out",
            "timeout",
            "urlopen error",
            "unexpected_eof",
            "eof occurred",
            "connection reset",
            "temporarily unavailable",
            "overloaded",
            "ssl",
        ]
        return any(marker in text for marker in retry_markers)

    def _request_with_retries(self, prompt: str) -> str:
        attempts = max(1, int(self.config.max_retries) + 1)
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._request_once(prompt)
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts - 1 or not self._is_retryable_error(exc):
                    raise
                delay = max(0.0, self.config.retry_base_delay_s) * (2**attempt)
                delay += random.uniform(0.0, min(0.25, delay * 0.25))
                time.sleep(delay)
        raise RuntimeError(f"LLM request failed after {attempts} attempts: {last_exc}")

    def generate(self, prompt: str, *, n: int = 1) -> list[str]:
        return [self._request_with_retries(prompt) for _ in range(n)]


class NullLLMProvider:
    """Deterministic provider for smoke tests; never calls an external API."""

    def generate(self, prompt: str, *, n: int = 1) -> list[str]:
        del prompt
        return [
            """
def degree_order(G):
    H = G.copy()
    order = []
    while H.number_of_nodes() > 0:
        node = max(H.nodes(), key=lambda u: (H.degree[u], str(u)))
        order.append(node)
        H.remove_node(node)
    return order
""".strip()
            for _ in range(n)
        ]
