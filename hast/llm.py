# -*- coding: utf-8 -*-
"""LLM provider abstraction.

The framework keeps LLM calls behind this interface so experiments can be run
with real API calls, cached logs, or deterministic mock candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMProvider(Protocol):
    def generate(self, prompt: str, *, n: int = 1) -> list[str]:
        ...


@dataclass
class LoggedLLMConfig:
    model: str = "GPT-5.5"
    reasoning_effort: str = "none"
    temperature: float = 0.2


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
