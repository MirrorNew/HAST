# -*- coding: utf-8 -*-
"""Log-induced bound induction for Stage 2."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class BoundPolicy:
    preferred_families: list[str]
    pruned_families: list[str]
    allowed_signals: list[str]
    cap_bounds: dict[str, Any]
    update_bounds: dict[str, Any]
    forbidden_patterns: list[str]
    stage3_prompt_contract: str
    llm_call_budget: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "preferred_families": self.preferred_families,
            "pruned_families": self.pruned_families,
            "allowed_signals": self.allowed_signals,
            "cap_bounds": self.cap_bounds,
            "update_bounds": self.update_bounds,
            "forbidden_patterns": self.forbidden_patterns,
            "stage3_prompt_contract": self.stage3_prompt_contract,
            "llm_call_budget": self.llm_call_budget,
        }


def induce_bounds_from_log(records: pd.DataFrame, llm_policies: list[dict[str, Any]] | None = None) -> BoundPolicy:
    """Combine programmatic statistics with up to 10 LLM policy proposals."""
    valid = records[records["valid"].astype(bool)].copy() if not records.empty else records.copy()
    if valid.empty:
        return BoundPolicy(
            preferred_families=["degree-local"],
            pruned_families=[],
            allowed_signals=["residual_degree", "neighbor_degree", "two_hop_degree", "core_number"],
            cap_bounds={"max_neighbor_scan": 64, "max_two_hop_scan": 128},
            update_bounds={"mode": "local_residual_update", "avoid_global_rescan": True},
            forbidden_patterns=["all_pairs_shortest_path", "betweenness_recompute_each_step", "full_graph_sort_each_step"],
            stage3_prompt_contract="Generate bounded local-update dismantling candidates based on the best Stage-1 families.",
        )

    top = valid.sort_values("rank_score", ascending=False).head(max(5, min(30, len(valid))))
    family_counts = Counter(str(x) for x in top.get("family", []))
    preferred = [name for name, _ in family_counts.most_common(5) if name and name != "unknown"]
    if not preferred:
        preferred = ["degree-local"]

    allowed = {"residual_degree", "neighbor_degree", "two_hop_degree", "core_number", "component_size"}
    forbidden = {
        "all_pairs_shortest_path",
        "betweenness_recompute_each_step",
        "full_graph_sort_each_step",
        "unbounded_two_hop_scan",
    }
    cap_bounds: dict[str, Any] = {
        "max_neighbor_scan": 64,
        "max_two_hop_scan": 128,
        "max_candidate_runtime_s_on_proxy": float(max(0.05, top["time_s"].median() * 4.0)),
    }
    update_bounds: dict[str, Any] = {
        "mode": "local_residual_update",
        "avoid_global_rescan": True,
        "heap_or_bucket_updates_allowed": True,
    }

    for policy in (llm_policies or [])[:10]:
        allowed.update(policy.get("allowed_signals", []))
        forbidden.update(policy.get("forbidden_patterns", []))
        cap_bounds.update(policy.get("cap_bounds", {}))
        update_bounds.update(policy.get("update_bounds", {}))
        for fam in policy.get("preferred_families", []):
            if fam not in preferred:
                preferred.append(fam)

    return BoundPolicy(
        preferred_families=preferred[:8],
        pruned_families=["global-centrality-recompute", "unbounded-path-search"],
        allowed_signals=sorted(allowed),
        cap_bounds=cap_bounds,
        update_bounds=update_bounds,
        forbidden_patterns=sorted(forbidden),
        stage3_prompt_contract=(
            "Use the best Stage-1 family/top candidates as seeds; generate bounded, local-update "
            "algorithms that use only allowed signals and respect all cap/update bounds."
        ),
    )


def write_policy(path, policy: BoundPolicy) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
