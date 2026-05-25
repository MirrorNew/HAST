# -*- coding: utf-8 -*-
"""Execute HAST motivation experiments E1-E3.

E1 reuses the logged Observation 1 cases. E2/E3 can either use deterministic
proxy candidates for smoke tests or real OpenAI-compatible LLM generations
cached as ``degree_order(G)`` programs under local artifacts.
"""

from __future__ import annotations

import json
import math
import sys
import time
import argparse
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.algorithms import hda_original_order
from hast.candidate import CandidateProgram, extract_code
from hast.data import generate_powerlaw_network, read_graph
from hast.llm import OpenAICompatibleLLMProvider
from hast.search_stage_1_and_3 import evaluate_candidate, evaluate_order_fn

OUT_TABLE = ROOT / "artifacts" / "source_tables" / "motivation_observation" / "e1_e3"
OUT_FIG = ROOT / "artifacts" / "figures"
OBS1_CASES = (
    ROOT
    / "artifacts"
    / "source_tables"
    / "motivation_observation"
    / "obs1_basic_baseline"
    / "motivation_obs1_basic_baseline_horizontal_cases.csv"
)

GROUP_SIZE = 100
RATE = 0.30
PROXY_DATASETS = ["smoke", "Powerlaw_120"]
DEFAULT_LLM_CACHE_TAG = "default"


def evaluate_candidate_worker(code: str, candidate_id: str, family: str, source_stage: str, graphs: list[nx.Graph], rate: float, queue: Any) -> None:
    program = CandidateProgram(candidate_id=candidate_id, code=code, family=family, source_stage=source_stage)
    record = evaluate_candidate(program, graphs, rate)
    queue.put(asdict(record))


@dataclass(frozen=True)
class CandidateSpec:
    group: str
    index: int
    family: str
    mode: str
    code: str
    feature_degree_backbone: bool
    feature_frontier: bool
    feature_weak_tie: bool
    feature_boundary: bool
    feature_redundancy: bool
    feature_phase: bool
    feature_heap_update: bool
    feature_global_rescan: bool
    feature_unbounded_two_hop: bool
    feature_connected_components: bool
    synthetic_invalid: bool = False
    llm_error: str = ""


def setup() -> None:
    OUT_TABLE.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#1F2937",
            "axes.edgecolor": "#374151",
            "xtick.color": "#1F2937",
            "ytick.color": "#1F2937",
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def run_e1() -> dict[str, Any]:
    if not OBS1_CASES.exists():
        raise FileNotFoundError(f"Missing E1 source cases: {OBS1_CASES}")
    cases = pd.read_csv(OBS1_CASES, encoding="utf-8-sig")
    rows: list[dict[str, Any]] = []
    for i, row in cases.iterrows():
        case_id = f"{row['dataset']} case {i + 1}"
        pair = f"{row['method_a']} vs {row['method_b']}"
        rows.append(
            {
                "case_id": case_id,
                "dataset": row["dataset"],
                "method": row["method_a"],
                "step": int(row["step_a"]),
                "GCC": float(row["gcc_a"]),
                "cNBI": float(row["hidden_score_a"]),
                "pair": pair,
            }
        )
        rows.append(
            {
                "case_id": case_id,
                "dataset": row["dataset"],
                "method": row["method_b"],
                "step": int(row["step_b"]),
                "GCC": float(row["gcc_b"]),
                "cNBI": float(row["hidden_score_b"]),
                "pair": pair,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLE / "e1_obs1_same_gcc_cnbi_cases.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    labels = [f"{r.method}\nstep {int(r.step)}" for r in out.itertuples()]
    values = [float(v) for v in out["cNBI"]]
    colors = ["#4C78A8" if i % 2 == 0 else "#F58518" for i in range(len(values))]
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="#1F2937", linewidth=0.7)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("cNBI at the same GCC/R")
    ax.set_title("Observation 1: cNBI separates same-GCC residual states")
    ymax = max(values) * 1.22
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.018,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#111827",
        )
    for case_idx, (_, group) in enumerate(out.groupby("case_id", sort=False)):
        center = case_idx * 2 + 0.5
        gcc = float(group["GCC"].iloc[0])
        gap = float(group["cNBI"].max() - group["cNBI"].min())
        ax.axvspan(case_idx * 2 - 0.45, case_idx * 2 + 1.45, color="#F8FAFC" if case_idx % 2 == 0 else "#FFF7ED", zorder=-1)
        ax.text(
            center,
            -0.23,
            f"{group['case_id'].iloc[0]}\nGCC/R={gcc:.6f}\nΔcNBI={gap:.1f}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="#374151",
        )
    ax.margins(x=0.03)
    fig.subplots_adjust(bottom=0.32, top=0.90, left=0.09, right=0.99)
    fig.savefig(OUT_FIG / "fig_obs1_same_gcc_cnbi_bar.png")
    plt.close(fig)

    summary = (
        out.groupby(["case_id", "pair"], as_index=False)
        .agg(GCC=("GCC", "mean"), cNBI_min=("cNBI", "min"), cNBI_max=("cNBI", "max"))
        .assign(cNBI_gap=lambda df: df["cNBI_max"] - df["cNBI_min"])
    )
    summary.to_csv(OUT_TABLE / "e1_obs1_same_gcc_cnbi_summary.csv", index=False, encoding="utf-8-sig")
    return {
        "rows": len(out),
        "cases": len(summary),
        "max_cnbi_gap": float(summary["cNBI_gap"].max()),
        "figure": str(OUT_FIG / "fig_obs1_same_gcc_cnbi_bar.png"),
    }


def candidate_code(
    *,
    marker: str,
    neighbor_weight: float,
    weak_weight: float,
    boundary_weight: float,
    redundancy_weight: float,
    phase_weight: float,
    cap_neighbors: int,
    cap_two_hop: int,
    global_rescan: bool,
    invalid: bool = False,
) -> str:
    if invalid:
        return f"# {marker}\ndef invalid_candidate(G):\n    return []\n"
    return f"""
# {marker}
def degree_order(G):
    H = G.copy()
    order = []
    n0 = max(1, H.number_of_nodes())
    while H.number_of_nodes() > 0:
        progress = len(order) / n0
        best = None
        best_score = None
        component_count = 0.0
        if {str(global_rescan)}:
            component_count = float(nx.number_connected_components(H))
        for u in list(H.nodes()):
            neighbors = list(H.neighbors(u))
            scan_neighbors = neighbors
            if {cap_neighbors} >= 0:
                scan_neighbors = neighbors[:{cap_neighbors}]
            degree_score = float(H.degree[u])
            neighbor_score = 0.0
            weak_score = 0.0
            boundary_score = 0.0
            redundancy_score = 0.0
            for v in scan_neighbors:
                dv = float(H.degree[v])
                neighbor_score += dv
                if dv <= 2.0:
                    weak_score += 1.0
                seen = 0
                for w in H.neighbors(v):
                    if w == u:
                        continue
                    boundary_score += float(H.degree[w])
                    seen += 1
                    if {cap_two_hop} >= 0 and seen >= {cap_two_hop}:
                        break
            for i in range(len(scan_neighbors)):
                a = scan_neighbors[i]
                for b in scan_neighbors[i + 1:]:
                    if H.has_edge(a, b):
                        redundancy_score += 1.0
            phase = 1.0 + {phase_weight:.6f} * progress
            score = (
                phase * degree_score
                + {neighbor_weight:.6f} * neighbor_score
                + {weak_weight:.6f} * weak_score
                + {boundary_weight:.6f} * boundary_score
                - {redundancy_weight:.6f} * redundancy_score
                + 0.01 * component_count
            )
            if best is None or score > best_score or (score == best_score and str(u) > str(best)):
                best = u
                best_score = score
        order.append(best)
        H.remove_node(best)
    return order
""".strip()


def make_specs(group: str, n: int) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for i in range(n):
        r = i % 25
        invalid = group == "Relative-Free" and i in {7, 29, 53, 71, 89}
        invalid = invalid or (group == "CostAware-Free" and i in {31, 77})
        if group == "R/GCC-only":
            params = (0.005 * (r % 5), 0.0, 0.0, 0.0, 0.05 * (r % 3), 8 + (r % 5) * 4, 0, False)
            family = "degree-backbone"
        elif group in {"Absolute-cNBI", "Relative-Free"}:
            params = (0.02 + 0.01 * (r % 5), 0.10 + 0.05 * (r % 4), 0.0015 + 0.0005 * (r % 6), 0.00 if r % 3 else 0.02, 0.15 * (r % 4), -1 if r % 2 == 0 else 64, -1 if r % 4 == 0 else 32, r % 10 == 0)
            family = "free-boundary-scan"
        elif group in {"Relative-Delta-cNBI", "CostAware-Free"}:
            params = (0.015 + 0.006 * (r % 5), 0.08 + 0.04 * (r % 4), 0.0008 + 0.0004 * (r % 5), 0.03 + 0.01 * (r % 4), 0.10 * (r % 4), 16 + (r % 4) * 8, 12 + (r % 4) * 8, r % 18 == 0)
            family = "relative-local-mix"
        elif group == "Bounded-Guided":
            params = (0.010 + 0.004 * (r % 5), 0.06 + 0.03 * (r % 4), 0.0005 + 0.0002 * (r % 5), 0.04 + 0.01 * (r % 5), 0.06 * (r % 4), 8 + (r % 4) * 4, 6 + (r % 4) * 4, False)
            family = "bounded-local-policy"
        else:
            raise ValueError(f"Unknown group: {group}")
        nw, ww, bw, rw, pw, cap_n, cap_2, rescan = params
        code = candidate_code(
            marker=f"{group}-{i:03d}",
            neighbor_weight=nw,
            weak_weight=ww,
            boundary_weight=bw,
            redundancy_weight=rw,
            phase_weight=pw,
            cap_neighbors=cap_n,
            cap_two_hop=cap_2,
            global_rescan=rescan,
            invalid=invalid,
        )
        specs.append(
            CandidateSpec(
                group=group,
                index=i,
                family=family,
                mode="deterministic-proxy",
                code=code,
                feature_degree_backbone=True,
                feature_frontier=nw > 0,
                feature_weak_tie=ww > 0,
                feature_boundary=bw > 0,
                feature_redundancy=rw > 0,
                feature_phase=pw > 0,
                feature_heap_update=False,
                feature_global_rescan=rescan,
                feature_unbounded_two_hop=cap_n < 0 or cap_2 < 0,
                feature_connected_components=rescan,
                synthetic_invalid=invalid,
            )
        )
    return specs


def llm_prompt(group: str, index: int) -> str:
    base = f"""
You are generating candidate #{index} for HAST motivation experiments on network dismantling.
Return only Python code defining:

def degree_order(G):
    return removal_order

Rules:
- Use only networkx/numpy/math/heapq/collections/itertools if imports are needed.
- Do not read or write files, use subprocess/network calls, or access external data.
- Return a complete node removal ordering for all nodes in G.
- Keep the implementation deterministic.
""".strip()
    if group == "R/GCC-only":
        objective = """
Objective for Observation 2 group R/GCC-only:
Optimize standard dismantling feedback, especially lowering GCC/R.
Prefer an HDA-like residual degree backbone. Do not explicitly optimize cNBI.
""".strip()
    elif group == "Absolute-cNBI":
        objective = """
Objective for Observation 2 group Absolute-cNBI:
Optimize absolute process fragmentation measured by AUC-cNBI.
You may add local frontier, weak-tie, boundary, two-hop, or redundancy signals.
""".strip()
    elif group == "Relative-Delta-cNBI":
        objective = """
Objective for Observation 2 group Relative-Delta-cNBI:
Improve over an HDA-original root by adding local mechanisms that create new residual fragmentation.
Prioritize the strongest relative Delta AUC-cNBI signal, even if the candidate becomes slower.
Prefer richer frontier, weak-tie, boundary, two-hop, component-sensitivity, redundancy, or phase signals.
This group is intentionally not cost-aware; runtime risk will be measured after generation.
""".strip()
    elif group == "Relative-Free":
        objective = """
Objective for Observation 3 group Relative-Free:
Freely generate a heuristic that improves relative AUC-cNBI over HDA.
No bounded-language restriction is imposed, and quality is more important than runtime.
Slow scans may appear here; runtime risk will be measured after generation.
""".strip()
    elif group == "CostAware-Free":
        objective = """
Objective for Observation 3 group CostAware-Free:
Improve relative AUC-cNBI over HDA while also keeping runtime light.
Avoid expensive global recomputation unless clearly justified.
""".strip()
    elif group == "Bounded-Guided":
        objective = """
Objective for Observation 3 group Bounded-Guided:
Generate a bounded local-update heuristic using the following prior Stage-2-style bounds.
These bounds are injected as prior knowledge for this small experiment and do not count as search.

Backbone:
- Must keep residual degree / dynamic degree as the main backbone.

Allowed local signals:
- frontier, weak-tie, two-hop boundary, neighbor degree, redundancy, phase weight, local bridge proxy.

Cap bounds:
- cap_n must be in [8, 48].
- cap_2 must be in [4, 16].
- update_cap must be in [16, 96].

Scan bounds:
- Forbid unbounded nested two-hop scans.
- High-degree nodes must use sampling or explicit caps.

Update rules:
- Use lazy heap, local affected set, and capped two-hop update when useful.

Forbidden patterns:
- Do not use betweenness, PageRank, all-pairs shortest path, spectral methods,
  frequent connected_components, or per-step full-graph recomputation.

Family pruning prior:
- Preserve families with high Delta AUC-cNBI, low time, and low invalid rate.
- Prune long-term low-gain or high-slowdown families.
""".strip()
    else:
        raise ValueError(f"Unknown group: {group}")
    return base + "\n\n" + objective


def infer_features_from_code(code: str) -> dict[str, bool]:
    text = code.lower()
    has_two_hop = "two" in text or "2-hop" in text or "single_source_shortest_path" in text
    has_cap = "cap" in text or "limit" in text or "[:" in text or "break" in text
    return {
        "feature_degree_backbone": "degree" in text or ".degree" in text,
        "feature_frontier": "frontier" in text or "neighbor" in text or "boundary" in text,
        "feature_weak_tie": "weak" in text or "<= 2" in text or "<=2" in text or "low" in text,
        "feature_boundary": "boundary" in text or has_two_hop,
        "feature_redundancy": "redundan" in text or "cluster" in text or "triangle" in text or "overlap" in text,
        "feature_phase": "phase" in text or "progress" in text or "len(order)" in text,
        "feature_heap_update": "heapq" in text or "heap" in text or "bucket" in text,
        "feature_global_rescan": "connected_components" in text or "core_number" in text or "k_core" in text,
        "feature_unbounded_two_hop": has_two_hop and not has_cap,
        "feature_connected_components": "connected_components" in text,
    }


def make_one_llm_spec(group: str, i: int, provider: OpenAICompatibleLLMProvider, code_dir: Path) -> CandidateSpec:
    safe_group = group.replace("/", "_")
    code_path = code_dir / f"{safe_group}_{i:03d}.py"
    error_path = code_dir / f"{safe_group}_{i:03d}.error.txt"
    error = ""
    if code_path.exists():
        code = code_path.read_text(encoding="utf-8")
        if error_path.exists():
            error = error_path.read_text(encoding="utf-8")
    else:
        try:
            response = provider.generate(llm_prompt(group, i), n=1)[0]
            code = extract_code(response)
        except Exception as exc:
            error = str(exc)
            code = f"# LLM request failed for {group} #{i}: {error}\ndef invalid_candidate(G):\n    return []\n"
        code_path.write_text(code, encoding="utf-8")
        if error:
            error_path.write_text(error, encoding="utf-8")
    features = infer_features_from_code(code)
    return CandidateSpec(
        group=group,
        index=i,
        family="llm-generated",
        mode="real-llm",
        code=code,
        synthetic_invalid=False,
        llm_error=error,
        **features,
    )


def safe_cache_tag(tag: str) -> str:
    keep = []
    for char in tag.strip() or DEFAULT_LLM_CACHE_TAG:
        keep.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(keep)


def make_llm_specs(
    group: str,
    n: int,
    provider: OpenAICompatibleLLMProvider,
    workers: int,
    cache_tag: str,
) -> list[CandidateSpec]:
    suffix = safe_cache_tag(cache_tag)
    code_dir = OUT_TABLE / ("llm_candidate_code" if suffix == DEFAULT_LLM_CACHE_TAG else f"llm_candidate_code_{suffix}")
    code_dir.mkdir(parents=True, exist_ok=True)
    specs: list[CandidateSpec] = []
    max_workers = max(1, workers)
    print(
        json.dumps(
            {
                "event": "candidate_generation_semantics",
                "group": group,
                "candidate_count": n,
                "max_parallel_llm_requests": max_workers,
                "unit": "one independent degree_order(G) program per candidate index",
                "not_a_search_tree": True,
                "tree_expansion": "none; no parent-child candidate links and no sequential feedback between indices",
                "cache_tag": suffix,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(make_one_llm_spec, group, i, provider, code_dir): i for i in range(n)}
        for future in as_completed(futures):
            spec = future.result()
            specs.append(spec)
            print(json.dumps({"group": group, "index": spec.index, "llm_error": bool(spec.llm_error)}, ensure_ascii=False), flush=True)
    return sorted(specs, key=lambda item: item.index)


def rank01(values: pd.Series, higher: bool) -> pd.Series:
    valid = values.notna()
    out = pd.Series([0.0] * len(values), index=values.index, dtype=float)
    if valid.sum() == 0:
        return out
    ranks = values[valid].rank(method="average", ascending=not higher)
    out.loc[valid] = 1.0 if len(ranks) == 1 else 1.0 - (ranks - 1.0) / (len(ranks) - 1.0)
    return out


def add_group_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["selection_score"] = -1.0
    for group, idx in out.groupby("group").groups.items():
        sub = out.loc[idx]
        valid_idx = sub.index[sub["valid"].astype(bool)]
        if len(valid_idx) == 0:
            continue
        r_delta = rank01(sub.loc[valid_idx, "delta_auc_cNBI"], True)
        r_auc = rank01(sub.loc[valid_idx, "auc_cNBI"], True)
        r_r = rank01(sub.loc[valid_idx, "R"], False)
        r_t = rank01(sub.loc[valid_idx, "time_s"], False)
        if group == "R/GCC-only":
            score = r_r
        elif group == "Absolute-cNBI":
            score = r_auc
        elif group in {"Relative-Delta-cNBI", "Relative-Free"}:
            score = r_delta
        elif group == "CostAware-Free":
            score = 0.60 * r_delta + 0.40 * r_t
        elif group == "Bounded-Guided":
            score = 0.40 * r_delta + 0.25 * r_r + 0.25 * r_t + 0.10 * r_auc
        else:
            score = r_delta
        out.loc[valid_idx, "selection_score"] = score
    return out


def load_proxy_graphs(dataset_names: list[str]) -> list[tuple[str, nx.Graph]]:
    graphs: list[tuple[str, nx.Graph]] = []
    for dataset in dataset_names:
        if dataset == "smoke":
            graph = nx.read_edgelist(ROOT / "network" / "smoke.edgelist", nodetype=int)
            graphs.append((dataset, nx.Graph(graph)))
        elif dataset.startswith("Powerlaw_"):
            graphs.append((dataset, generate_powerlaw_network(int(dataset.split("_", 1)[1]), seed=42)))
        else:
            graphs.append((dataset, read_graph(dataset)))
    return graphs


def evaluate_one_spec(spec: CandidateSpec, graph_values: list[nx.Graph], timeout_s: float) -> dict[str, Any]:
    candidate_id = f"{spec.group.replace('/', '_')}_{spec.index:03d}"
    if timeout_s <= 0:
        program = CandidateProgram(candidate_id=candidate_id, code=spec.code, family=spec.family, source_stage=spec.group)
        return asdict(evaluate_candidate(program, graph_values, RATE))

    queue: Any = mp.Queue()
    proc = mp.Process(
        target=evaluate_candidate_worker,
        args=(spec.code, candidate_id, spec.family, spec.group, graph_values, RATE, queue),
    )
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(2)
        return {
            "candidate_id": candidate_id,
            "family": spec.family,
            "source_stage": spec.group,
            "valid": False,
            "error": f"candidate evaluation timeout after {timeout_s:.1f}s",
            "R": math.nan,
            "auc_cNBI": math.nan,
            "auc_ACC": math.nan,
            "auc_NCC": math.nan,
            "final_ACC": math.nan,
            "final_NCC": math.nan,
            "final_cNBI": math.nan,
            "time_s": math.nan,
            "graph_count": len(graph_values),
        }
    if not queue.empty():
        return queue.get()
    return {
        "candidate_id": candidate_id,
        "family": spec.family,
        "source_stage": spec.group,
        "valid": False,
        "error": "candidate worker exited without a result",
        "R": math.nan,
        "auc_cNBI": math.nan,
        "auc_ACC": math.nan,
        "auc_NCC": math.nan,
        "final_ACC": math.nan,
        "final_NCC": math.nan,
        "final_cNBI": math.nan,
        "time_s": math.nan,
        "graph_count": len(graph_values),
    }


def evaluate_specs(specs: Iterable[CandidateSpec], graphs: list[tuple[str, nx.Graph]], root_auc: float, timeout_s: float) -> pd.DataFrame:
    graph_values = [g for _, g in graphs]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        started = time.perf_counter()
        item = evaluate_one_spec(spec, graph_values, timeout_s)
        item.update(asdict(spec))
        item.pop("code", None)
        item["wall_time_s"] = time.perf_counter() - started
        item["timed_out"] = "timeout" in str(item.get("error", "")).lower()
        if item["timed_out"]:
            item["time_s"] = timeout_s
        item["delta_auc_cNBI"] = item["auc_cNBI"] - root_auc if item["valid"] else math.nan
        rows.append(item)
    return add_group_scores(pd.DataFrame(rows))


def pareto_count(df: pd.DataFrame) -> int:
    rows = df.to_dict("records")
    count = 0
    for row in rows:
        dominated = False
        for other in rows:
            if row is other:
                continue
            better_or_equal = (
                float(other["auc_cNBI"]) >= float(row["auc_cNBI"])
                and float(other["R"]) <= float(row["R"])
                and float(other["time_s"]) <= float(row["time_s"])
            )
            strictly_better = (
                float(other["auc_cNBI"]) > float(row["auc_cNBI"])
                or float(other["R"]) < float(row["R"])
                or float(other["time_s"]) < float(row["time_s"])
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            count += 1
    return count


def summarize_groups(records: pd.DataFrame, slow_threshold_s: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, sub in records.groupby("group", sort=False):
        valid = sub[sub["valid"].astype(bool)].copy()
        top = valid.sort_values("selection_score", ascending=False).head(5)
        top_wall = top["wall_time_s"] if not top.empty else pd.Series(dtype=float)
        rows.append(
            {
                "group": group,
                "candidates": int(len(sub)),
                "valid_rate": float(len(valid) / max(1, len(sub))),
                "slow_threshold_s": slow_threshold_s,
                "slow_rate": float((sub["wall_time_s"] > slow_threshold_s).mean()),
                "timeout_rate": float(sub["timed_out"].astype(bool).mean()) if "timed_out" in sub else 0.0,
                "mean_auc_cNBI": float(valid["auc_cNBI"].mean()) if not valid.empty else math.nan,
                "best_auc_cNBI": float(valid["auc_cNBI"].max()) if not valid.empty else math.nan,
                "top5_mean_auc_cNBI": float(top["auc_cNBI"].mean()) if not top.empty else math.nan,
                "mean_delta_auc_cNBI": float(valid["delta_auc_cNBI"].mean()) if not valid.empty else math.nan,
                "best_delta_auc_cNBI": float(valid["delta_auc_cNBI"].max()) if not valid.empty else math.nan,
                "top5_mean_delta_auc_cNBI": float(top["delta_auc_cNBI"].mean()) if not top.empty else math.nan,
                "mean_R": float(valid["R"].mean()) if not valid.empty else math.nan,
                "best_R": float(valid["R"].min()) if not valid.empty else math.nan,
                "mean_time_s": float(valid["time_s"].mean()) if not valid.empty else math.nan,
                "top5_mean_time_s": float(top["time_s"].mean()) if not top.empty else math.nan,
                "mean_wall_time_s": float(sub["wall_time_s"].mean()) if not sub.empty else math.nan,
                "top5_mean_wall_time_s": float(top_wall.mean()) if not top_wall.empty else math.nan,
                "pareto_count": int(pareto_count(valid)),
                "degree_backbone_rate": float(valid["feature_degree_backbone"].mean()) if not valid.empty else math.nan,
                "frontier_rate": float(valid["feature_frontier"].mean()) if not valid.empty else math.nan,
                "weak_tie_rate": float(valid["feature_weak_tie"].mean()) if not valid.empty else math.nan,
                "boundary_rate": float(valid["feature_boundary"].mean()) if not valid.empty else math.nan,
                "redundancy_rate": float(valid["feature_redundancy"].mean()) if not valid.empty else math.nan,
                "unbounded_two_hop_rate": float(valid["feature_unbounded_two_hop"].mean()) if not valid.empty else math.nan,
                "global_rescan_rate": float(valid["feature_global_rescan"].mean()) if not valid.empty else math.nan,
            }
        )
    return pd.DataFrame(rows)


def add_bar_labels(ax: plt.Axes, bars: Any, *, fmt: str, ymin: float, ymax: float) -> None:
    offset = (ymax - ymin) * 0.025
    for bar in bars:
        value = float(bar.get_height())
        label_y = min(value + offset, ymax - offset * 0.35)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#111827",
        )


def plot_summary(summary: pd.DataFrame, name: str) -> None:
    if name == "e2_observation2":
        cols = ["top5_mean_delta_auc_cNBI", "top5_mean_auc_cNBI", "top5_mean_wall_time_s"]
        titles = ["Top-5 Delta AUC-cNBI", "Top-5 AUC-cNBI", "Top-5 wall runtime"]
        colors = ["#0072B2", "#009E73", "#D55E00"]
        path = OUT_FIG / "fig22_relative_credit_allocation_effect.png"
        title = "Observation 2: relative credit changes selected candidates"
        ymins = [6.0, 30.0, 0.6]
        ymax_overrides = [None, None, None]
        formats = ["{:.2f}", "{:.2f}", "{:.2f}s"]
    else:
        summary = summary.copy()
        summary["mean_R_percent"] = summary["mean_R"] * 100.0
        cols = ["top5_mean_auc_cNBI", "mean_R_percent", "top5_mean_wall_time_s"]
        titles = ["Top-5 AUC-cNBI", "Mean R/GCC (%)", "Top-5 wall runtime"]
        colors = ["#009E73", "#D55E00", "#0072B2"]
        path = OUT_FIG / "fig23_bounded_generation_controls_scan_cost.png"
        title = "Observation 3: bounded generation controls scan cost"
        ymins = [30.0, 2.0, 0.6]
        ymax_overrides = [None, None, None]
        formats = ["{:.2f}", "{:.2f}", "{:.2f}s"]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
    groups = summary["group"].tolist()
    for ax, col, subtitle, ymin, ymax_override, fmt in zip(axes, cols, titles, ymins, ymax_overrides, formats):
        values = [float(v) for v in summary[col]]
        ymax = ymax_override if ymax_override is not None else max(values) + (max(values) - ymin) * 0.22
        if ymax <= ymin:
            ymax = ymin + 1.0
        bars = ax.bar(groups, values, color=colors, edgecolor="#1F2937", linewidth=0.7)
        ax.set_ylim(ymin, ymax)
        ax.set_title(subtitle)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        add_bar_labels(ax, bars, fmt=fmt, ymin=ymin, ymax=ymax)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_candidate_experiment(
    groups: list[str],
    name: str,
    root_auc: float,
    graphs: list[tuple[str, nx.Graph]],
    *,
    group_size: int,
    provider: OpenAICompatibleLLMProvider | None,
    llm_workers: int,
    timeout_s: float,
    slow_threshold_s: float,
    cache_tag: str,
) -> dict[str, Any]:
    print(
        json.dumps(
            {
                "event": "candidate_experiment_plan",
                "experiment": name,
                "groups": groups,
                "candidates_per_group": group_size,
                "candidate_generation_semantics": (
                    "Each group samples/evaluates independent degree_order(G) programs. "
                    "The 100-candidate budget is a flat batch, not a sequential 100-node search tree."
                ),
                "real_llm": provider is not None,
                "max_parallel_llm_requests": llm_workers if provider is not None else 0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if provider is None:
        specs = [spec for group in groups for spec in make_specs(group, group_size)]
        mode = "local deterministic proxy; no external LLM calls"
    else:
        specs = [
            spec
            for group in groups
            for spec in make_llm_specs(group, group_size, provider, llm_workers, cache_tag)
        ]
        mode = "real LLM"
    records = evaluate_specs(specs, graphs, root_auc, timeout_s)
    summary = summarize_groups(records, slow_threshold_s)
    records_path = OUT_TABLE / f"{name}_candidate_records.csv"
    summary_path = OUT_TABLE / f"{name}_group_summary.csv"
    records.to_csv(records_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    plot_summary(summary, name)
    return {"mode": mode, "records": str(records_path), "summary": str(summary_path), "groups": summary.to_dict("records")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-llm", action="store_true", help="Use the configured real LLM provider for E2/E3 candidates.")
    parser.add_argument("--candidates-per-group", type=int, default=GROUP_SIZE)
    parser.add_argument("--llm-workers", type=int, default=6)
    parser.add_argument("--proxy-datasets", nargs="+", default=PROXY_DATASETS)
    parser.add_argument("--candidate-timeout-s", type=float, default=30.0)
    parser.add_argument("--slow-threshold-s", type=float, default=2.0)
    parser.add_argument("--llm-cache-tag", default=DEFAULT_LLM_CACHE_TAG)
    args = parser.parse_args()

    setup()
    started = time.perf_counter()
    graphs = load_proxy_graphs(args.proxy_datasets)
    root = evaluate_order_fn(lambda g: hda_original_order(g, rate=None), [graph for _, graph in graphs], RATE)
    provider = OpenAICompatibleLLMProvider.from_env() if args.real_llm else None
    e1 = run_e1()
    e2 = run_candidate_experiment(
        ["R/GCC-only", "Absolute-cNBI", "Relative-Delta-cNBI"],
        "e2_observation2",
        root["auc_cNBI"],
        graphs,
        group_size=args.candidates_per_group,
        provider=provider,
        llm_workers=args.llm_workers,
        timeout_s=args.candidate_timeout_s,
        slow_threshold_s=args.slow_threshold_s,
        cache_tag=args.llm_cache_tag,
    )
    e3 = run_candidate_experiment(
        ["Relative-Free", "CostAware-Free", "Bounded-Guided"],
        "e3_observation3",
        root["auc_cNBI"],
        graphs,
        group_size=args.candidates_per_group,
        provider=provider,
        llm_workers=args.llm_workers,
        timeout_s=args.candidate_timeout_s,
        slow_threshold_s=args.slow_threshold_s,
        cache_tag=args.llm_cache_tag,
    )
    manifest = {
        "mode": "real LLM" if args.real_llm else "local deterministic proxy; no external LLM calls",
        "candidate_contract": "degree_order(G) -> removal_order",
        "candidate_budget_per_group": args.candidates_per_group,
        "llm_workers": args.llm_workers if args.real_llm else 0,
        "candidate_timeout_s": args.candidate_timeout_s,
        "slow_threshold_s": args.slow_threshold_s,
        "llm_cache_tag": args.llm_cache_tag if args.real_llm else None,
        "llm": {
            "model": provider.config.model if provider is not None else None,
            "base_url": provider.config.base_url if provider is not None else None,
            "reasoning_effort": provider.config.reasoning_effort if provider is not None else None,
            "temperature": provider.config.temperature if provider is not None else None,
        },
        "rate": RATE,
        "proxy_datasets": [name for name, _ in graphs],
        "root": root,
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "elapsed_s": time.perf_counter() - started,
    }
    manifest_path = OUT_TABLE / "e1_e3_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "elapsed_s": manifest["elapsed_s"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
