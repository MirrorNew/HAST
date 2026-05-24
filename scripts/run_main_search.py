# -*- coding: utf-8 -*-
"""Entry point for HAST-Lite-Full search configuration and dry runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hast.candidate import make_program
from hast.config import LLM_DEFAULTS, MAIN_BUDGETS
from hast.data import generate_powerlaw_network
from hast.llm import NullLLMProvider
from hast.search import run_three_stage_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without external LLM calls.")
    args = parser.parse_args()
    if not args.dry_run:
        raise SystemExit("Real LLM-backed execution is intentionally explicit; wire an LLMProvider before running.")
    provider = NullLLMProvider()
    stage1 = [make_program(text, family="degree-local", source_stage="stage1") for text in provider.generate("stage1", n=2)]
    stage3 = [make_program(text, family="bounded-degree-local", source_stage="stage3") for text in provider.generate("stage3", n=2)]
    result = run_three_stage_smoke(stage1, stage3, [generate_powerlaw_network(80, seed=42)], rate=0.30)
    print(
        json.dumps(
            {
                "budgets": MAIN_BUDGETS,
                "llm": LLM_DEFAULTS,
                "dry_run": args.dry_run,
                "stage1_rows": int(len(result["stage1"])),
                "stage2_policy_budget": result["policy"].llm_call_budget,
                "stage3_rows": int(len(result["stage3"])),
                "final": sorted(result["final"].keys()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
