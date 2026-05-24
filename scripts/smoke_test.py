# -*- coding: utf-8 -*-
"""Run a local smoke test for the independent HAST main project."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.algorithms import (
    bpd_minsum_fallback_order,
    ci_order,
    cluc_order,
    corehd_fast_order,
    dc_order,
    gnd_fallback_order,
    hda_fast_order,
    hda_original_order,
    kcore_order,
    ncdc_order,
    ndjc_order,
)
from hast.candidate import make_program
from hast.search1_3 import run_three_stage_smoke
from metrics.fragmentation import compute_metrics, summarize_metrics

import networkx as nx


STAGE1_CODE = """
def degree_order(G):
    H = G.copy()
    order = []
    while H.number_of_nodes() > 0:
        node = max(H.nodes(), key=lambda u: (H.degree[u], str(u)))
        order.append(node)
        H.remove_node(node)
    return order
"""

STAGE3_CODE = """
def degree_order(G):
    H = G.copy()
    order = []
    while H.number_of_nodes() > 0:
        node = max(H.nodes(), key=lambda u: (H.degree[u] + 0.01 * sum(H.degree[v] for v in H.neighbors(u)), str(u)))
        order.append(node)
        H.remove_node(node)
    return order
"""


def main() -> None:
    graph = nx.read_edgelist(ROOT / "network" / "smoke.edgelist", nodetype=int)
    graph = nx.Graph(graph)
    rate = 0.30
    baseline_rows = {}
    for name, fn in {
        "HDA-original": hda_original_order,
        "HDA-fast": hda_fast_order,
        "CoreHD-fast": corehd_fast_order,
        "DC": dc_order,
        "KCore": kcore_order,
        "CLUC": cluc_order,
        "CI": ci_order,
        "NDJC": ndjc_order,
        "NCDC": ncdc_order,
        "BPD/MinSum-fallback": bpd_minsum_fallback_order,
        "GND-py": gnd_fallback_order,
    }.items():
        order = fn(graph, rate)
        metrics = compute_metrics(graph, order, rate=rate, method_time=0.0)
        baseline_rows[name] = summarize_metrics(metrics)

    result = run_three_stage_smoke(
        programs_stage1=[make_program(STAGE1_CODE, family="degree-local", source_stage="stage1")],
        programs_stage3=[make_program(STAGE3_CODE, family="neighbor-degree", source_stage="stage3")],
        graphs=[graph],
        rate=rate,
    )
    out = {
        "baseline_methods": sorted(baseline_rows),
        "stage1_rows": int(len(result["stage1"])),
        "stage2_llm_call_budget": result["policy"].llm_call_budget,
        "stage3_rows": int(len(result["stage3"])),
        "final_labels": sorted(result["final"].keys()),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
