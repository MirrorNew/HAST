"""ERA-like generic search framework entrypoint.

The historical raw run key was `ERA-like`; the paper-facing framework name is
`ERA-like`.
"""

from __future__ import annotations

from .generic_llm_search_ablation import EraLikePolicy as Policy


METHOD_NAME = "ERA-like"
RAW_COMPAT_NAME = "ERA-like"

__all__ = ["METHOD_NAME", "RAW_COMPAT_NAME", "Policy"]
