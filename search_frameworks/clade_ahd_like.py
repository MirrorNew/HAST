"""Clade-AHD-like generic search framework entrypoint."""

from __future__ import annotations

from .generic_llm_search_ablation import CladeAHDPolicy as Policy


METHOD_NAME = "Clade-AHD-like"

__all__ = ["METHOD_NAME", "Policy"]
