"""FunSearch-like generic search framework entrypoint."""

from __future__ import annotations

from .generic_llm_search_ablation import FunSearchPolicy as Policy


METHOD_NAME = "FunSearch-like"

__all__ = ["METHOD_NAME", "Policy"]
