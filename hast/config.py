# -*- coding: utf-8 -*-
"""Shared constants for the HAST project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT
BENCHMARK_ROOT = PROJECT_ROOT / "network"
RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

REAL_DATASETS = [
    "CEnew",
    "Collaboration",
    "condmat",
    "crime",
    "email",
    "Grid",
    "GrQC",
    "hamster",
    "HepPh",
    "PH",
    "Yeast",
]
DATASETS = REAL_DATASETS + ["Powerlaw_500"]
DATASET_RATES = {name: 0.30 for name in DATASETS}
DATASET_RATES["email"] = 0.40

MAIN_BUDGETS = {
    "stage1_candidates": 300,
    "stage2_llm_calls": 10,
    "stage3_candidates": 200,
    "candidates_per_llm_call": 1,
    "stage3_parent_limit": 24,
    "candidate_timeout_s": 90.0,
}

DELTA_CREDIT_MODES = ["parent", "root"]

LLM_DEFAULTS = {
    "model": "gpt-5.5",
    "reasoning_effort": "none",
    "temperature": 0.2,
    "base_url": "https://api.ritelt.com/v1",
}


@dataclass(frozen=True)
class SearchWeights:
    relative_credit: float
    fragmentation: float
    time: float
    absolute_quality: float


STAGE1_WEIGHTS = SearchWeights(relative_credit=0.45, fragmentation=0.25, time=0.20, absolute_quality=0.10)
STAGE3_WEIGHTS = SearchWeights(relative_credit=0.40, fragmentation=0.25, time=0.25, absolute_quality=0.10)
