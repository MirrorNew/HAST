# -*- coding: utf-8 -*-
"""Reference checks for E6-frozen HAST final candidates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ReferenceTarget:
    label: str
    legacy_code: str
    auc_cNBI: float | None = None
    R: float | None = None
    time_s: float | None = None
    source: str = "legacy_record"


LEGACY_REFERENCES = [
    ReferenceTarget(
        label="legacy_q",
        legacy_code="BT-n16-t8-u24",
        auc_cNBI=313.448,
        R=0.4288,
        time_s=1.4531,
        source="analyze_past_results/HAST_target_family_consolidated_conclusions_cn.md",
    ),
    ReferenceTarget(
        label="legacy_s",
        legacy_code="FAST21-cap24",
        auc_cNBI=356.253,
        R=None,
        time_s=0.556,
        source="historical HAST-Final-S 12-graph note in docs/scripts",
    ),
]


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def compare_candidate_to_reference(candidate: dict[str, Any], reference: ReferenceTarget) -> dict[str, Any]:
    auc = _finite_or_none(candidate.get("mean_auc_cNBI"))
    r_value = _finite_or_none(candidate.get("mean_R"))
    time_value = _finite_or_none(candidate.get("mean_time_s"))
    checks: dict[str, bool | None] = {
        "auc_cNBI_at_least_reference": None if reference.auc_cNBI is None or auc is None else auc >= reference.auc_cNBI,
        "R_at_most_reference": None if reference.R is None or r_value is None else r_value <= reference.R,
        "time_s_at_most_reference": None if reference.time_s is None or time_value is None else time_value <= reference.time_s,
    }
    available = [value for value in checks.values() if value is not None]
    return {
        "candidate_method": candidate.get("method"),
        "candidate_id": candidate.get("candidate_id"),
        "candidate": {
            "mean_auc_cNBI": auc,
            "mean_R": r_value,
            "mean_time_s": time_value,
        },
        "reference": asdict(reference),
        "checks": checks,
        "passes_available_checks": bool(available) and all(bool(value) for value in available),
    }


def write_reference_check(e7_output_dir: Path, references: list[ReferenceTarget] | None = None) -> dict[str, Any]:
    references = references or LEGACY_REFERENCES
    method_mean_path = e7_output_dir / "method_mean_metrics.csv"
    if not method_mean_path.exists():
        raise FileNotFoundError(f"Missing E7 method_mean_metrics.csv: {method_mean_path}")
    method_mean = pd.read_csv(method_mean_path, encoding="utf-8-sig")
    records = method_mean.to_dict("records")
    comparisons = [
        compare_candidate_to_reference(candidate, reference)
        for candidate in records
        for reference in references
    ]
    q_rows = [row for row in records if str(row.get("method")) == "HAST-Final-Q"]
    s_rows = [row for row in records if str(row.get("method")) == "HAST-Final-S"]
    q_passes = any(
        item["candidate_method"] == "HAST-Final-Q" and item["reference"]["label"] == "legacy_q" and item["passes_available_checks"]
        for item in comparisons
    )
    s_passes = any(
        item["candidate_method"] == "HAST-Final-S" and item["reference"]["label"] == "legacy_s" and item["passes_available_checks"]
        for item in comparisons
    )
    payload = {
        "gate": {
            "paper_refresh_allowed": bool(q_passes and s_passes),
            "requires_e6_frozen_methods": ["HAST-Final-Q", "HAST-Final-S"],
            "has_hast_final_q": bool(q_rows),
            "has_hast_final_s": bool(s_rows),
            "legacy_q_passed_by_hast_final_q": q_passes,
            "legacy_s_passed_by_hast_final_s": s_passes,
        },
        "comparisons": comparisons,
    }
    (e7_output_dir / "legacy_reference_check.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload
