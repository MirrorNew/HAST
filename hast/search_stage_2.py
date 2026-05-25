# -*- coding: utf-8 -*-
"""Log-induced bound induction for Stage 2."""

from __future__ import annotations

import json
import re
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


SIGNAL_ALIASES = {
    "degree": "residual_degree",
    "degree_backbone": "residual_degree",
    "residual_degree": "residual_degree",
    "neighbor_degree": "neighbor_degree",
    "two_hop": "bounded_two_hop",
    "two_hop_degree": "bounded_two_hop",
    "bounded_two_hop": "bounded_two_hop",
    "frontier": "frontier",
    "boundary": "boundary",
    "weak_tie": "weak_tie",
    "redundancy": "redundancy",
    "phase": "phase",
    "heap_update": "lazy_heap",
    "lazy_heap": "lazy_heap",
    "core_number": "core_number",
}

ALLOWED_SIGNALS = {
    "residual_degree",
    "neighbor_degree",
    "bounded_two_hop",
    "frontier",
    "boundary",
    "weak_tie",
    "redundancy",
    "phase",
    "lazy_heap",
    "core_number",
}

SIGNAL_DENY_RE = re.compile(
    r"(disallow|forbid|must_exclude|global|unbounded|component_refresh|rescan|notes?|required|optional|may_include)",
    re.IGNORECASE,
)

CANONICAL_FORBIDDEN = {
    "all_pairs_shortest_path",
    "betweenness_recompute_each_step",
    "pagerank_or_eigenvector_recompute_each_step",
    "connected_components_each_step",
    "full_graph_sort_each_step",
    "full_graph_rescan_each_step",
    "unbounded_two_hop_scan",
    "unbounded_bfs_or_dfs",
    "nondeterministic_random_ordering",
}


def _as_string_items(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                for key in ("family", "signal", "name", "pattern"):
                    if isinstance(item.get(key), str):
                        out.append(item[key])
                        break
        return out
    return []


def _sanitize_allowed_signals(items: list[Any]) -> list[str]:
    out: set[str] = set()
    for raw in items:
        for text in _as_string_items(raw):
            key = text.strip().lower().replace("-", "_").replace(" ", "_")
            if not key or SIGNAL_DENY_RE.search(key):
                continue
            mapped = SIGNAL_ALIASES.get(key)
            if mapped in ALLOWED_SIGNALS:
                out.add(mapped)
    return sorted(out)


def _sanitize_families(items: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        for text in _as_string_items(raw):
            family = re.sub(r"[^0-9A-Za-z_\-]+", "-", text.strip()).strip("-")
            if not family or SIGNAL_DENY_RE.search(family):
                continue
            if family not in seen:
                seen.add(family)
                out.append(family)
    return out


def _sanitize_forbidden(items: list[Any]) -> list[str]:
    out = set(CANONICAL_FORBIDDEN)
    for raw in items:
        for text in _as_string_items(raw):
            lowered = text.lower()
            if "connected_components" in lowered or "component" in lowered:
                out.add("connected_components_each_step")
            if "all_pairs" in lowered or "shortest" in lowered:
                out.add("all_pairs_shortest_path")
            if "betweenness" in lowered:
                out.add("betweenness_recompute_each_step")
            if "pagerank" in lowered or "eigenvector" in lowered:
                out.add("pagerank_or_eigenvector_recompute_each_step")
            if "full" in lowered and ("sort" in lowered or "rescan" in lowered or "recompute" in lowered):
                out.add("full_graph_rescan_each_step")
            if "unbounded" in lowered or "bfs" in lowered or "dfs" in lowered:
                out.add("unbounded_bfs_or_dfs")
            if "random" in lowered or "nondetermin" in lowered:
                out.add("nondeterministic_random_ordering")
    return sorted(out)


def _sanitize_cap_bounds(base: dict[str, Any], proposed: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in proposed.items():
        if not isinstance(key, str):
            continue
        safe_key = re.sub(r"[^0-9A-Za-z_\-]+", "_", key.strip())
        if not safe_key:
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[safe_key] = value
        elif isinstance(value, list):
            cleaned = [x for x in value if isinstance(x, (int, float, str, bool))]
            if cleaned:
                out[safe_key] = cleaned[:8]
        elif isinstance(value, dict):
            cleaned_dict = {
                re.sub(r"[^0-9A-Za-z_\-]+", "_", str(k)): v
                for k, v in value.items()
                if isinstance(v, (int, float, str, bool, list))
            }
            if cleaned_dict:
                out[safe_key] = cleaned_dict
    return out


def enforce_target_family_contract(policy: BoundPolicy) -> BoundPolicy:
    allowed = sorted(set(policy.allowed_signals) | {"bounded_two_hop", "frontier", "boundary", "weak_tie", "redundancy", "phase", "lazy_heap"})
    cap_bounds = dict(policy.cap_bounds)
    cap_bounds["target_family_cap_contract"] = {
        "Q_like": {"neighbor_cap": [24, 32, 48, 64], "two_hop_cap": [16, 24, 32], "affected_cap": [18, 24, 32]},
        "S_like": {"neighbor_cap": [12, 16, 24], "two_hop_cap": [6, 8, 12], "affected_cap": [16, 20, 24]},
        "bridge": {"neighbor_cap": [16, 24, 32], "two_hop_cap": [8, 12, 16, 24], "affected_cap": [18, 24, 28]},
        "repair": {"neighbor_cap": [12, 16, 24, 32], "two_hop_cap": [6, 8, 12, 16], "affected_cap": [12, 16, 20, 24]},
    }
    cap_bounds["bounded_two_hop_required"] = True
    cap_bounds["phase_schedule_required"] = True
    cap_bounds["forbid_policy_disabling_target_family_signals"] = True
    update_bounds = dict(policy.update_bounds)
    update_bounds["affected_scope"] = "neighbors plus capped neighbors-of-neighbors; never full graph refresh"
    return BoundPolicy(
        preferred_families=policy.preferred_families,
        pruned_families=policy.pruned_families,
        allowed_signals=allowed,
        cap_bounds=cap_bounds,
        update_bounds=update_bounds,
        forbidden_patterns=policy.forbidden_patterns,
        stage3_prompt_contract=(
            "Generate target-family bounded local-update candidates with residual degree, capped frontier/weak-tie/"
            "boundary/redundancy, phase weights, and capped two-hop affected refresh. LLM proposals may tune caps "
            "and weights but must not disable bounded two-hop or phase-aware local scoring."
        ),
        llm_call_budget=policy.llm_call_budget,
    )


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

    allowed = {"residual_degree", "neighbor_degree", "bounded_two_hop", "frontier", "boundary", "lazy_heap"}
    forbidden = set(CANONICAL_FORBIDDEN)
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
        allowed.update(_sanitize_allowed_signals(policy.get("allowed_signals", [])))
        forbidden.update(_sanitize_forbidden(policy.get("forbidden_patterns", [])))
        cap_bounds = _sanitize_cap_bounds(cap_bounds, policy.get("cap_bounds", {}) if isinstance(policy.get("cap_bounds"), dict) else {})
        if isinstance(policy.get("update_bounds"), dict):
            for key in ("mode", "strategy", "heap_policy", "refresh_scope", "affected_scope"):
                if key in policy["update_bounds"] and isinstance(policy["update_bounds"][key], (str, int, float, bool)):
                    update_bounds[key] = policy["update_bounds"][key]
        for fam in _sanitize_families(policy.get("preferred_families", [])):
            if fam not in preferred:
                preferred.append(fam)

    return enforce_target_family_contract(BoundPolicy(
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
    ))


def write_policy(path, policy: BoundPolicy) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
