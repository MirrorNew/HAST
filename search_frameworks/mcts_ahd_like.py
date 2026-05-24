"""MCTS-AHD-like generic search framework entrypoint."""

from __future__ import annotations

from .generic_llm_search_ablation import MCTSAHDPolicy as Policy


METHOD_NAME = "MCTS-AHD-like"

__all__ = ["METHOD_NAME", "Policy"]
