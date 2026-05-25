# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hast.reference_check import ReferenceTarget, write_reference_check
from experiments.full_validation import load_e6_final_programs


def test_load_e6_final_programs_uses_fixed_labels(tmp_path: Path) -> None:
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    code = "def degree_order(G):\n    return list(G.nodes())\n"
    (final_dir / "HAST-Final-Q.py").write_text(code, encoding="utf-8")
    (final_dir / "HAST-Final-S.py").write_text(code, encoding="utf-8")

    programs, method_names = load_e6_final_programs(final_dir)

    assert method_names == ["HAST-Final-Q", "HAST-Final-S"]
    assert [program.source_stage for program in programs] == ["HAST-Final-Q", "HAST-Final-S"]


def test_reference_check_gate_passes_only_e6_frozen_labels(tmp_path: Path) -> None:
    out_dir = tmp_path / "full_validation"
    out_dir.mkdir()
    pd.DataFrame(
        [
            {
                "method": "HAST-Final-Q",
                "candidate_id": "q",
                "mean_auc_cNBI": 10.0,
                "mean_R": 0.2,
                "mean_time_s": 1.0,
            },
            {
                "method": "HAST-Final-S",
                "candidate_id": "s",
                "mean_auc_cNBI": 8.0,
                "mean_R": 0.3,
                "mean_time_s": 0.5,
            },
        ]
    ).to_csv(out_dir / "method_mean_metrics.csv", index=False, encoding="utf-8-sig")

    payload = write_reference_check(
        out_dir,
        references=[
            ReferenceTarget("legacy_q", "q-old", auc_cNBI=9.0, R=0.25, time_s=1.2),
            ReferenceTarget("legacy_s", "s-old", auc_cNBI=7.0, R=None, time_s=0.6),
        ],
    )

    assert payload["gate"]["paper_refresh_allowed"] is True
    written = json.loads((out_dir / "legacy_reference_check.json").read_text(encoding="utf-8"))
    assert written["gate"]["legacy_q_passed_by_hast_final_q"] is True
    assert written["gate"]["legacy_s_passed_by_hast_final_s"] is True
