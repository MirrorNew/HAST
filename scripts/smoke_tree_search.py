# -*- coding: utf-8 -*-
"""Smoke-test the HAST tree-search execution path without external LLM calls."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hast.config import RUNS_ROOT
from hast.e4_e6 import E4E6Config, run_e4_e6
from hast.llm import NullLLMProvider


def main() -> None:
    config = E4E6Config(
        run_dir=RUNS_ROOT / "HAST" / "e4_e6" / "smoke_tree_semantics",
        proxy_datasets=["Powerlaw_500"],
        full_datasets=["Powerlaw_500"],
        stage1_budget=2,
        stage2_budget=1,
        stage3_budget=2,
        llm_workers=2,
        candidate_timeout_s=90.0,
    )
    result = run_e4_e6(config, NullLLMProvider())
    print(
        {
            "stage1_rows": result["stage1_rows"],
            "stage1_tree_nodes": result["stage1_tree_nodes"],
            "stage3_rows": result["stage3_rows"],
            "stage3_tree_nodes": result["stage3_tree_nodes"],
            "final_labels": list(result["final_code_manifest"].keys()),
        }
    )


if __name__ == "__main__":
    main()
