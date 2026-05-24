"""AlphaEvolve-like generic search framework entrypoint."""

from __future__ import annotations

from .generic_llm_search_ablation import AlphaEvolvePolicy as Policy


METHOD_NAME = "AlphaEvolve-like"

__all__ = ["METHOD_NAME", "Policy"]
