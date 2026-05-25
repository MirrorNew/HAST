# -*- coding: utf-8 -*-
"""E4-E6 HAST search orchestration.

This module wires Stage 1 free search, Stage 2 bound induction, and Stage 3
bounded guided search. E7 full validation lives in ``experiments.full_validation``.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty
from typing import Any

import pandas as pd

from .candidate import CandidateProgram, extract_code, make_program
from .config import LLM_DEFAULTS, MAIN_BUDGETS, PROJECT_ROOT, RUNS_ROOT, STAGE1_WEIGHTS, STAGE3_WEIGHTS
from .data import read_graph
from .llm import LLMProvider
from .ranking import add_rank_scores, pareto_frontier, select_final_q_s
from .search_stage_1_and_3 import evaluate_candidate, evaluate_root
from .search_stage_2 import BoundPolicy, induce_bounds_from_log, write_policy


BENCHMARK_TABLE_ROOT = PROJECT_ROOT / "artifacts" / "source_tables" / "benchmark_12graph"
DEFAULT_RUN_DIR = RUNS_ROOT / "HAST" / "e4_e6"

HDA_ROOT_CODE = """
def degree_order(G):
    H = G.copy()
    order = []
    while H.number_of_nodes() > 0:
        node = max(H.nodes(), key=lambda u: (H.degree[u], str(u)))
        order.append(node)
        H.remove_node(node)
    return order
""".strip()


@dataclass(frozen=True)
class E4E6Config:
    run_dir: Path
    proxy_datasets: list[str]
    full_datasets: list[str]
    stage1_budget: int = MAIN_BUDGETS["stage1_candidates"]
    stage2_budget: int = MAIN_BUDGETS["stage2_llm_calls"]
    stage3_budget: int = MAIN_BUDGETS["stage3_candidates"]
    candidates_per_llm_call: int = MAIN_BUDGETS["candidates_per_llm_call"]
    stage3_parent_limit: int = MAIN_BUDGETS["stage3_parent_limit"]
    candidate_timeout_s: float = MAIN_BUDGETS["candidate_timeout_s"]
    delta_credit_mode: str = "parent"
    llm_workers: int = 4


def safe_run_token(text: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", text.strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "experiment"


def dated_run_dir(experiment_name: str, delta_credit_mode: str, date_text: str | None = None) -> Path:
    if delta_credit_mode not in {"parent", "root"}:
        raise ValueError("delta_credit_mode must be 'parent' or 'root'")
    date_value = date_text or datetime.now().strftime("%Y%m%d")
    return RUNS_ROOT / f"runs_HAST_{delta_credit_mode}_{safe_run_token(experiment_name)}_{date_value}"


def default_config(run_name: str = "main", delta_credit_mode: str = "parent", run_date: str | None = None) -> E4E6Config:
    return E4E6Config(
        run_dir=dated_run_dir(run_name, delta_credit_mode, run_date),
        proxy_datasets=["Powerlaw_500"],
        full_datasets=[
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
            "Powerlaw_500",
        ],
        delta_credit_mode=delta_credit_mode,
    )


def ensure_dirs(config: E4E6Config) -> None:
    for name in [
        "prompts/stage1",
        "prompts/stage2",
        "prompts/stage3",
        "raw_llm/stage1",
        "raw_llm/stage2",
        "raw_llm/stage3",
        "candidates/stage1",
        "candidates/stage3",
        "stage2",
    ]:
        (config.run_dir / name).mkdir(parents=True, exist_ok=True)


def read_benchmark_context(root: Path = BENCHMARK_TABLE_ROOT) -> dict[str, Any]:
    method_mean = pd.read_csv(root / "method_mean_metrics.csv", encoding="utf-8-sig")
    per_graph = pd.read_csv(root / "per_graph_metrics.csv", encoding="utf-8-sig")
    schema = pd.read_csv(root / "point_evaluation_schema.csv", encoding="utf-8-sig")
    hda = method_mean[method_mean["method"].eq("HDA")]
    era = method_mean[method_mean["method"].isin(["PUCT", "ERA-like"])]
    top = method_mean.sort_values("mean_auc_cNBI", ascending=False).head(8)
    return {
        "root": str(root),
        "method_mean_rows": int(len(method_mean)),
        "per_graph_rows": int(len(per_graph)),
        "point_schema": schema.to_dict("records"),
        "hda": hda.to_dict("records"),
        "era_like": era.to_dict("records"),
        "top_methods": top[
            ["method", "group", "datasets", "mean_R", "mean_auc_cNBI", "mean_time_s"]
        ].to_dict("records"),
    }


def benchmark_context_text(context: dict[str, Any]) -> str:
    top_rows = context.get("top_methods", [])
    quality_values = [
        float(row["mean_auc_cNBI"])
        for row in top_rows
        if math.isfinite(float(row.get("mean_auc_cNBI", float("nan"))))
    ]
    time_values = [
        float(row["mean_time_s"])
        for row in top_rows
        if math.isfinite(float(row.get("mean_time_s", float("nan"))))
    ]
    lines = [
        "Anonymous benchmark reference from artifacts/source_tables/benchmark_12graph:",
        f"- method_mean rows: {context['method_mean_rows']}",
        f"- per_graph rows: {context['per_graph_rows']}",
        "- Search is blind to historical method names and final candidate ids.",
    ]
    if quality_values:
        lines.append(
            f"- Strong external quality band: AUC-cNBI in "
            f"[{min(quality_values):.3f}, {max(quality_values):.3f}]"
        )
    if time_values:
        lines.append(f"- Strong external runtime band: time_s in [{min(time_values):.3f}, {max(time_values):.3f}]")
    if context["hda"]:
        row = context["hda"][0]
        lines.append(
            f"- HDA root reference: R={row['mean_R']:.4f}, "
            f"AUC-cNBI={row['mean_auc_cNBI']:.3f}, time_s={row['mean_time_s']:.3f}"
        )
    return "\n".join(lines)


def truncate_text(text: str, limit: int = 5000) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head].rstrip() + "\n# ... truncated ...\n" + text[-tail:].lstrip()


def metric_brief(record: dict[str, Any]) -> str:
    return (
        f"node_id={record.get('node_id', '')}, "
        f"candidate_id={record.get('candidate_id', '')}, "
        f"R={float(record.get('R', float('nan'))):.6f}, "
        f"AUC-cNBI={float(record.get('auc_cNBI', float('nan'))):.6f}, "
        f"early_cNBI={float(record.get('early_cNBI', float('nan'))):.6f}, "
        f"early_NCC={float(record.get('early_NCC', float('nan'))):.6f}, "
        f"early_GCC={float(record.get('early_GCC', float('nan'))):.6f}, "
        f"time_s={float(record.get('time_s', float('nan'))):.6f}, "
        f"rank_score={float(record.get('rank_score', 0.0)):.6f}"
    )


def active_delta_description(delta_credit_mode: str) -> str:
    if delta_credit_mode == "root":
        return "Delta AUC-cNBI for this node is AUC-cNBI(child) - AUC-cNBI(root HDA node)."
    return "Delta AUC-cNBI for this node is AUC-cNBI(child) - AUC-cNBI(parent)."


CANDIDATE_IMPLEMENTATION_GUARDS = """
Implementation guards:
- If you use tuple/list score keys, every return path of the scoring helper must have the same length and field types.
- Keep all negated heap-key fields numeric. Put str(u) or other string tie-breakers only at the final non-negated position.
- For d == 0 or isolated-node branches, return the full numeric score shape followed by the same final tie-breaker.
- Do not return explanations, markdown prose, or JSON; return executable Python only.
""".strip()


ANONYMOUS_TARGET_FAMILY_HINT = """
Anonymous target-family mechanism hints:
- Use an iterative lazy heap or bucket queue with deterministic numeric score keys.
- Score should combine residual degree with capped local frontier, weak-tie, boundary, redundancy, leaf-pressure, bounded two-hop, and bridge-pressure signals.
- Use small explicit caps for neighbor and second-hop scans; prefer fixed caps or sqrt-degree caps over full neighborhood cascades.
- Let phase/progress change weights: early keep hub pressure, middle emphasize frontier/weak ties, late emphasize boundary/bridge and avoid redundant dense neighborhoods.
- After removing a node, refresh only its neighbors and a capped set of second-hop affected nodes.
""".strip()


def build_stage1_prompt(
    index: int,
    parent: CandidateProgram,
    parent_record: dict[str, Any],
    context: dict[str, Any],
    delta_credit_mode: str = "parent",
) -> str:
    return f"""
You are expanding HAST E4 Stage 1 search-tree node #{index}.

Goal:
- Mutate the parent heuristic below into one child node of the Stage 1 search tree.
- Improve relative Delta AUC-cNBI without abandoning R/GCC quality.
- Keep runtime lightweight enough to avoid slow global scans.

Scoring:
S1 = 0.45 rank(Delta AUC-cNBI) + 0.25 rank(-R) + 0.20 rank(-T) + 0.10 rank(AUC-cNBI)
{active_delta_description(delta_credit_mode)}
The active rank also rewards early fragmentation at about 20% removal: higher cNBI, higher NCC, and lower GCC versus the root behavior.

Parent metrics:
{metric_brief(parent_record)}

Parent code:
```python
{truncate_text(parent.code, 4500)}
```

Candidate contract:
def degree_order(G):
    return removal_order

Rules:
- Return only Python code, preferably in one code block.
- Use deterministic NetworkX-compatible code.
- Allowed imports: math, heapq, random, itertools, collections, networkx, numpy.
- Do not read/write files, use subprocess/network calls, eval/exec, or external data.
- Avoid all-pairs shortest paths, per-step betweenness/PageRank, and unbounded two-hop scans.

{CANDIDATE_IMPLEMENTATION_GUARDS}

{ANONYMOUS_TARGET_FAMILY_HINT}

{benchmark_context_text(context)}
""".strip()


def policy_prompt(call_index: int, stage1: pd.DataFrame, features: pd.DataFrame, family_summary: pd.DataFrame, failures: dict[str, Any]) -> str:
    top = stage1[stage1["valid"].astype(bool)].sort_values("rank_score", ascending=False).head(20)
    evidence = {
        "stage1_top_records": top[
            [
                "candidate_id",
                "family",
                "R",
                "auc_cNBI",
                "early_cNBI",
                "early_NCC",
                "early_GCC",
                "delta_auc_cNBI",
                "time_s",
                "rank_score",
            ]
        ].to_dict("records"),
        "family_summary": family_summary.to_dict("records"),
        "feature_rates": features.drop(columns=["code_path"], errors="ignore").head(80).to_dict("records"),
        "failure_patterns": failures,
    }
    return f"""
You are HAST E5 Stage 2 induction call #{call_index}.

Use only the evidence below to induce a bounded candidate language for Stage 3.
Do not generate a degree_order candidate. Return one JSON object only, with keys:
allowed_signals, preferred_families, pruned_families, cap_bounds, update_bounds,
forbidden_patterns, stage3_prompt_contract.

Evidence JSON:
{json.dumps(evidence, ensure_ascii=False, indent=2)}
""".strip()


def build_stage3_prompt(
    index: int,
    parent: CandidateProgram,
    parent_record: dict[str, Any],
    policy: BoundPolicy,
    context: dict[str, Any],
    delta_credit_mode: str = "parent",
    branch_role: str = "bridge",
) -> str:
    branch_text = {
        "Q": (
            "Quality branch: preserve high AUC-cNBI and low R first. "
            "Use a degree backbone plus bounded frontier/boundary/two-hop and redundancy terms. "
            "A larger cap is allowed only when the update remains local and heap/lazy-refresh based."
        ),
        "S": (
            "Speed branch: keep the candidate fast and valid while staying close to the best local-fragmentation quality. "
            "Prefer smaller neighbor/two-hop caps, stable numeric heap keys, and cheap affected-node refresh."
        ),
        "B": (
            "Bridge branch: search the Pareto middle between quality and speed. "
            "Balance capped two-hop, frontier, weak-tie, and redundancy terms without adding global scans."
        ),
        "R": (
            "Repair branch: fix validity, timeout, mixed-type heap-key, and large-graph risk. "
            "Reduce caps or simplify refresh only if AUC/R degradation is bounded."
        ),
    }.get(branch_role, "Bridge branch: balance quality, R, and runtime under the bounded local contract.")
    return f"""
You are expanding HAST E6 Stage 3 bounded search-tree node #{index}.

Use this parent node as the mechanism seed, but compress it into the bounded language.
Stage 3 branch: {branch_role}
{branch_text}

Parent family: {parent.family}
Parent metrics:
{metric_brief(parent_record)}

Parent code:
```python
{truncate_text(parent.code, 4500)}
```

Bound policy JSON:
{json.dumps(policy.to_dict(), ensure_ascii=False, indent=2)}

Goal:
- Preserve effective local mechanisms from the parent.
- Only modify local signal combinations, weights, caps, phase schedule, and update rule.
- Do not add global slow algorithms.
- Emit one complete deterministic implementation, not a wrapper around the parent.

Scoring:
S3 = 0.40 rank(Delta AUC-cNBI) + 0.25 rank(-R) + 0.25 rank(-T) + 0.10 rank(AUC-cNBI)
{active_delta_description(delta_credit_mode)}
The active rank also rewards early fragmentation at about 20% removal: higher cNBI, higher NCC, and lower GCC versus the root behavior.

Candidate contract:
def degree_order(G):
    return removal_order

Rules:
- Return only Python code, preferably in one code block.
- Use deterministic NetworkX-compatible code.
- Allowed imports: math, heapq, random, itertools, collections, networkx, numpy.
- Do not read/write files, use subprocess/network calls, eval/exec, or external data.
- Keep all bounded local scans capped by the bound policy.

{CANDIDATE_IMPLEMENTATION_GUARDS}

{ANONYMOUS_TARGET_FAMILY_HINT}

{benchmark_context_text(context)}
""".strip()


def load_proxy_graphs(config: E4E6Config):
    return [read_graph(dataset) for dataset in config.proxy_datasets]


def invalid_row(candidate_id: str, family: str, source_stage: str, error: str, graph_count: int) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "family": family,
        "source_stage": source_stage,
        "valid": False,
        "error": error,
        "R": float("nan"),
        "auc_cNBI": float("nan"),
        "auc_ACC": float("nan"),
        "auc_NCC": float("nan"),
        "final_ACC": float("nan"),
        "final_NCC": float("nan"),
        "final_cNBI": float("nan"),
        "early_GCC": float("nan"),
        "early_NCC": float("nan"),
        "early_cNBI": float("nan"),
        "time_s": float("nan"),
        "graph_count": graph_count,
    }


def evaluate_candidate_worker(program: CandidateProgram, graphs, rate: float, queue) -> None:
    try:
        queue.put(asdict(evaluate_candidate(program, graphs, rate)))
    except Exception as exc:
        queue.put(
            invalid_row(
                program.candidate_id,
                program.family,
                program.source_stage,
                f"candidate worker failed: {exc}",
                len(graphs),
            )
        )


def evaluate_candidate_with_timeout(
    program: CandidateProgram,
    graphs,
    rate: float,
    timeout_s: float,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=evaluate_candidate_worker, args=(program, graphs, rate, queue))
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        return invalid_row(
            program.candidate_id,
            program.family,
            program.source_stage,
            f"candidate evaluation timeout after {timeout_s:.1f}s",
            len(graphs),
        )
    try:
        return queue.get_nowait()
    except Empty:
        return invalid_row(
            program.candidate_id,
            program.family,
            program.source_stage,
            f"candidate worker exited with code {process.exitcode} without a result",
            len(graphs),
        )


def request_llm_batch(
    provider: LLMProvider,
    prompts: list[str],
    workers: int,
    candidates_per_llm_call: int = 1,
) -> list[tuple[str, float]]:
    """Request one response per prompt, preserving prompt order."""
    workers = max(1, int(workers))
    if candidates_per_llm_call != 1:
        raise ValueError("HAST main search currently uses exactly one algorithm candidate per LLM call.")

    def one(prompt: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        try:
            response = provider.generate(prompt, n=candidates_per_llm_call)[0]
        except Exception as exc:
            response = f"# LLM request failed: {exc}"
        return response, time.perf_counter() - t0

    if workers == 1 or len(prompts) <= 1:
        return [one(prompt) for prompt in prompts]
    with ThreadPoolExecutor(max_workers=min(workers, len(prompts))) as executor:
        return list(executor.map(one, prompts))


def request_llm_one(provider: LLMProvider, prompt: str, candidates_per_llm_call: int = 1) -> tuple[str, float]:
    if candidates_per_llm_call != 1:
        raise ValueError("HAST tree expansion uses exactly one child candidate per LLM call.")
    t0 = time.perf_counter()
    try:
        response = provider.generate(prompt, n=candidates_per_llm_call)[0]
    except Exception as exc:
        response = f"# LLM request failed: {exc}"
    return response, time.perf_counter() - t0


def infer_family(code: str, default: str) -> str:
    lowered = code.lower()
    if "connected_components" in lowered or "number_connected_components" in lowered:
        return "component-refresh"
    if "two" in lowered or lowered.count("neighbors(") >= 2:
        return "two-hop-boundary"
    if "clustering" in lowered or "redund" in lowered:
        return "redundancy-aware"
    if "heapq" in lowered:
        return "heap-local-update"
    if "degree" in lowered:
        return "degree-local"
    return default


def code_features(candidate_id: str, code: str, code_path: str) -> dict[str, Any]:
    lowered = code.lower()
    neighbor_count = lowered.count("neighbors(")
    return {
        "candidate_id": candidate_id,
        "code_path": code_path,
        "degree_backbone": "degree" in lowered,
        "frontier": "frontier" in lowered or "boundary" in lowered or neighbor_count >= 1,
        "weak_tie": "weak" in lowered or "<= 2" in lowered or "< 3" in lowered,
        "two_hop": "two_hop" in lowered or "2-hop" in lowered or neighbor_count >= 2,
        "boundary": "boundary" in lowered or neighbor_count >= 2,
        "redundancy": "redund" in lowered or "clustering" in lowered or "has_edge" in lowered,
        "phase": "progress" in lowered or "phase" in lowered,
        "heap_update": "heapq" in lowered or "heappush" in lowered,
        "component_refresh": "connected_components" in lowered or "number_connected_components" in lowered,
        "global_rescan": "betweenness" in lowered or "pagerank" in lowered or "all_pairs" in lowered,
        "unbounded_two_hop": neighbor_count >= 2 and all(token not in lowered for token in ["cap", "limit", "break"]),
        "neighbor_calls": neighbor_count,
    }


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def make_tree_seed_record(
    *,
    node_id: str,
    program: CandidateProgram,
    metrics: dict[str, Any],
    code_path: Path,
    stage: str,
    depth: int,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "parent_node_id": "",
        "parent_candidate_id": "",
        "parent_auc_cNBI": float("nan"),
        "depth": depth,
        "stage_index": 0,
        "candidate_id": program.candidate_id,
        "family": program.family,
        "source_stage": program.source_stage,
        "valid": True,
        "error": "",
        "R": metrics["R"],
        "auc_cNBI": metrics["auc_cNBI"],
        "auc_ACC": metrics["auc_ACC"],
        "auc_NCC": metrics["auc_NCC"],
        "final_ACC": metrics["final_ACC"],
        "final_NCC": metrics["final_NCC"],
        "final_cNBI": metrics["final_cNBI"],
        "early_GCC": metrics.get("early_GCC", float("nan")),
        "early_NCC": metrics.get("early_NCC", float("nan")),
        "early_cNBI": metrics.get("early_cNBI", float("nan")),
        "time_s": metrics["time_s"],
        "graph_count": int(metrics.get("graph_count", 0)),
        "rank_score": 0.0,
        "delta_auc_cNBI": 0.0,
        "delta_root_auc_cNBI": 0.0,
        "prompt_path": "",
        "raw_response_path": "",
        "code_path": str(code_path),
        "llm_elapsed_s": 0.0,
        "tree_role": "seed",
    }


def refresh_generated_scores(
    records: list[dict[str, Any]],
    weights,
    root_auc_cNBI: float,
    delta_credit_mode: str,
) -> list[dict[str, Any]]:
    if not records:
        return []
    df = add_rank_scores(pd.DataFrame(records), weights, root_auc_cNBI)
    if delta_credit_mode == "root":
        df["delta_auc_cNBI"] = df["delta_root_auc_cNBI"]
        valid = df["valid"].astype(bool)
        df["rank_relative_credit"] = 0.0
        if valid.sum() > 0:
            idx = df.index[valid]
            df.loc[idx, "rank_relative_credit"] = df.loc[idx, "delta_auc_cNBI"].rank(
                method="average",
                ascending=False,
            )
            if len(idx) > 1:
                df.loc[idx, "rank_relative_credit"] = 1.0 - (df.loc[idx, "rank_relative_credit"] - 1.0) / (len(idx) - 1.0)
            else:
                df.loc[idx, "rank_relative_credit"] = 1.0
            df.loc[idx, "rank_score"] = (
                weights.relative_credit * df.loc[idx, "rank_relative_credit"]
                + weights.fragmentation * df.loc[idx, "rank_fragmentation"]
                + weights.time * df.loc[idx, "rank_time"]
                + weights.absolute_quality * df.loc[idx, "rank_absolute_quality"]
            )
    return df.to_dict("records")


STAGE1_HAST_PARENT_C = 0.10


def select_tree_parent(
    tree_records: list[dict[str, Any]],
    program_by_node: dict[str, CandidateProgram],
    *,
    parent_priority_mode: str = "legacy",
    c_hast: float = STAGE1_HAST_PARENT_C,
) -> dict[str, Any]:
    child_counts = Counter(str(row.get("parent_node_id", "")) for row in tree_records if row.get("parent_node_id"))
    candidates = [
        row
        for row in tree_records
        if bool(row.get("valid")) and str(row.get("node_id", "")) in program_by_node
    ]
    if not candidates:
        raise ValueError("search tree has no valid expandable parent")

    def key(row: dict[str, Any]) -> tuple[float, float, int, str]:
        node_id = str(row.get("node_id", ""))
        children = child_counts[node_id]
        base = finite_float(row.get("rank_score"), 0.0)
        if parent_priority_mode == "hast_stage1":
            exploration = c_hast / (1.0 + children)
            return (base + exploration, base, -children, node_id)
        exploration = 0.10 / math.sqrt(1.0 + children)
        depth_penalty = 0.005 * int(row.get("depth", 0) or 0)
        return (base + exploration - depth_penalty, -children, -int(row.get("depth", 0) or 0), node_id)

    return max(candidates, key=key)


def select_tree_parent_for_branch(
    tree_records: list[dict[str, Any]],
    program_by_node: dict[str, CandidateProgram],
    branch_role: str | None = None,
    *,
    parent_priority_mode: str = "legacy",
    c_hast: float = STAGE1_HAST_PARENT_C,
) -> dict[str, Any]:
    if not branch_role:
        return select_tree_parent(
            tree_records,
            program_by_node,
            parent_priority_mode=parent_priority_mode,
            c_hast=c_hast,
        )
    branch_records = [
        row
        for row in tree_records
        if str(row.get("tree_branch", "")) == branch_role
        and bool(row.get("valid"))
        and str(row.get("node_id", "")) in program_by_node
    ]
    if branch_records:
        return select_tree_parent(
            branch_records,
            program_by_node,
            parent_priority_mode=parent_priority_mode,
            c_hast=c_hast,
        )
    return select_tree_parent(
        tree_records,
        program_by_node,
        parent_priority_mode=parent_priority_mode,
        c_hast=c_hast,
    )


def stage3_branch_for_index(index: int, budget: int) -> str:
    base = [("Q", 60), ("S", 60), ("B", 50), ("R", 30)]
    if budget == 200:
        counts = dict(base)
    elif budget < 4:
        return ["Q", "S", "B"][max(0, index - 1) % 3]
    else:
        raw = [(name, max(1, round(budget * count / 200))) for name, count in base]
        diff = budget - sum(count for _, count in raw)
        while diff != 0:
            pos = 0 if diff > 0 else max(range(len(raw)), key=lambda i: raw[i][1])
            name, count = raw[pos]
            raw[pos] = (name, max(1, count + (1 if diff > 0 else -1)))
            diff = budget - sum(count for _, count in raw)
        counts = dict(raw)
    cursor = 0
    for name in ["Q", "S", "B", "R"]:
        cursor += counts[name]
        if index <= cursor:
            return name
    return "B"


def static_contract_violation(code: str) -> str:
    features = code_features("candidate", code, "")
    lowered = code.lower()
    if features["global_rescan"]:
        return "static contract violation: global centrality/rescan token"
    if features["component_refresh"]:
        return "static contract violation: connected components refresh"
    if features["unbounded_two_hop"]:
        return "static contract violation: unbounded two-hop scan"
    if "shortest_path" in lowered or "all_pairs" in lowered:
        return "static contract violation: path search"
    return ""


def generate_and_evaluate_tree_stage(
    *,
    config: E4E6Config,
    provider: LLMProvider,
    stage: str,
    budget: int,
    prompt_builder,
    family_default: str,
    graphs,
    root_auc_cNBI: float,
    initial_records: list[dict[str, Any]],
    initial_programs: dict[str, CandidateProgram],
    weights,
    branch_for_index=None,
    static_guard=None,
    parent_priority_mode: str = "legacy",
    c_hast: float = STAGE1_HAST_PARENT_C,
) -> tuple[pd.DataFrame, list[CandidateProgram], pd.DataFrame]:
    generated_records: list[dict[str, Any]] = []
    tree_records: list[dict[str, Any]] = list(initial_records)
    program_by_node = dict(initial_programs)
    stage_dir = config.run_dir / f"candidates/{stage}"

    for idx in range(1, budget + 1):
        branch_role = branch_for_index(idx, budget) if branch_for_index else ""
        parent_record = select_tree_parent_for_branch(
            tree_records,
            program_by_node,
            branch_role,
            parent_priority_mode=parent_priority_mode,
            c_hast=c_hast,
        )
        parent_node_id = str(parent_record["node_id"])
        parent_program = program_by_node[parent_node_id]
        try:
            prompt = prompt_builder(idx, parent_program, parent_record, branch_role)
        except TypeError:
            prompt = prompt_builder(idx, parent_program, parent_record)
        prompt_path = config.run_dir / f"prompts/{stage}/{idx:04d}.txt"
        raw_path = config.run_dir / f"raw_llm/{stage}/{idx:04d}.txt"
        code_path = stage_dir / f"{idx:04d}.py"
        prompt_path.write_text(prompt, encoding="utf-8")
        response, llm_elapsed_s = request_llm_one(provider, prompt, config.candidates_per_llm_call)
        raw_path.write_text(response, encoding="utf-8")

        code = extract_code(response)
        code_path.write_text(code, encoding="utf-8")
        family = infer_family(code, family_default)
        node_id = f"{stage}-{idx:04d}"
        try:
            if static_guard:
                violation = static_guard(code)
                if violation:
                    raise ValueError(violation)
            program = make_program(code, family=family, source_stage=stage)
            record = evaluate_candidate_with_timeout(program, graphs, rate=0.30, timeout_s=config.candidate_timeout_s)
            if bool(record.get("valid")):
                program_by_node[node_id] = program
        except Exception as exc:
            program = None
            record = invalid_row(node_id, family, stage, str(exc), len(graphs))

        record.update(
            {
                "node_id": node_id,
                "parent_node_id": parent_node_id,
                "parent_candidate_id": parent_record.get("candidate_id", ""),
                "parent_auc_cNBI": parent_record.get("auc_cNBI", float("nan")),
                "depth": int(parent_record.get("depth", 0) or 0) + 1,
                "stage_index": idx,
                "llm_elapsed_s": llm_elapsed_s,
                "prompt_path": str(prompt_path),
                "raw_response_path": str(raw_path),
                "code_path": str(code_path),
                "tree_role": "generated",
                "tree_branch": branch_role,
            }
        )
        generated_records.append(record)
        generated_records = refresh_generated_scores(generated_records, weights, root_auc_cNBI, config.delta_credit_mode)
        tree_records = tree_records[: len(initial_records)] + generated_records
        pd.DataFrame(generated_records).to_csv(config.run_dir / f"{stage}_candidate_log.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(tree_records).to_csv(config.run_dir / f"{stage}_tree_nodes.csv", index=False, encoding="utf-8-sig")

    df = pd.DataFrame(generated_records)
    tree_df = pd.DataFrame(tree_records)
    programs = [
        program_by_node[str(row["node_id"])]
        for row in generated_records
        if bool(row.get("valid")) and str(row.get("node_id", "")) in program_by_node
    ]
    return df, programs, tree_df


def feature_table_from_log(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.itertuples():
        path = Path(str(row.code_path))
        code = path.read_text(encoding="utf-8") if path.exists() else ""
        item = code_features(str(row.candidate_id), code, str(path))
        item["family"] = str(row.family)
        item["valid"] = bool(row.valid)
        item["rank_score"] = float(row.rank_score) if math.isfinite(float(row.rank_score)) else -1.0
        item["auc_cNBI"] = float(row.auc_cNBI) if math.isfinite(float(row.auc_cNBI)) else float("nan")
        item["time_s"] = float(row.time_s) if math.isfinite(float(row.time_s)) else float("nan")
        item["early_cNBI"] = float(row.early_cNBI) if hasattr(row, "early_cNBI") and math.isfinite(float(row.early_cNBI)) else float("nan")
        item["early_NCC"] = float(row.early_NCC) if hasattr(row, "early_NCC") and math.isfinite(float(row.early_NCC)) else float("nan")
        item["early_GCC"] = float(row.early_GCC) if hasattr(row, "early_GCC") and math.isfinite(float(row.early_GCC)) else float("nan")
        rows.append(item)
    return pd.DataFrame(rows)


def summarize_families(stage1: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    merged = stage1.merge(features[["candidate_id", "component_refresh", "global_rescan", "unbounded_two_hop"]], on="candidate_id", how="left")
    rows = []
    for family, group in merged.groupby("family"):
        valid = group[group["valid"].astype(bool)]
        rows.append(
            {
                "family": family,
                "candidates": int(len(group)),
                "valid_rate": float(group["valid"].astype(bool).mean()) if len(group) else 0.0,
                "mean_delta_auc_cNBI": float(valid["delta_auc_cNBI"].mean()) if len(valid) else float("nan"),
                "top_delta_auc_cNBI": float(valid["delta_auc_cNBI"].max()) if len(valid) else float("nan"),
                "mean_time_s": float(valid["time_s"].mean()) if len(valid) else float("nan"),
                "top_rank_score": float(valid["rank_score"].max()) if len(valid) else float("nan"),
                "slow_pattern_rate": float(
                    group[["component_refresh", "global_rescan", "unbounded_two_hop"]].fillna(False).any(axis=1).mean()
                )
                if len(group)
                else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["top_rank_score", "valid_rate"], ascending=False)


def failure_patterns(features: pd.DataFrame) -> dict[str, Any]:
    total = max(1, len(features))
    names = ["component_refresh", "global_rescan", "unbounded_two_hop", "heap_update"]
    return {
        name: {
            "count": int(features[name].fillna(False).sum()) if name in features else 0,
            "rate": float(features[name].fillna(False).sum() / total) if name in features else 0.0,
        }
        for name in names
    }


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    return json.loads(stripped[start : end + 1])


def run_stage2_induction(config: E4E6Config, provider: LLMProvider, stage1: pd.DataFrame) -> BoundPolicy:
    features = feature_table_from_log(stage1)
    family_summary = summarize_families(stage1, features)
    failures = failure_patterns(features)
    features.to_csv(config.run_dir / "stage2/code_feature_table.csv", index=False, encoding="utf-8-sig")
    family_summary.to_csv(config.run_dir / "stage2/family_summary.csv", index=False, encoding="utf-8-sig")
    (config.run_dir / "stage2/failure_patterns.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    proposals: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    prompts: list[str] = []
    for idx in range(1, config.stage2_budget + 1):
        prompt = policy_prompt(idx, stage1, features, family_summary, failures)
        prompts.append(prompt)
        prompt_path = config.run_dir / f"prompts/stage2/{idx:04d}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
    responses = request_llm_batch(provider, prompts, config.llm_workers, 1)
    for idx, (response, _llm_elapsed_s) in enumerate(responses, start=1):
        raw_path = config.run_dir / f"raw_llm/stage2/{idx:04d}.txt"
        raw_path.write_text(response, encoding="utf-8")
        try:
            proposals.append(parse_json_object(response))
        except Exception as exc:
            errors.append({"call": idx, "error": str(exc), "raw_response_path": str(raw_path)})

    policy = induce_bounds_from_log(stage1, llm_policies=proposals)
    write_policy(config.run_dir / "stage2/family_policy.json", policy)
    replay = {
        "llm_policy_calls": config.stage2_budget,
        "valid_policy_json": len(proposals),
        "policy_parse_errors": errors,
        "preferred_families": policy.preferred_families,
        "allowed_signals": policy.allowed_signals,
        "forbidden_patterns": policy.forbidden_patterns,
    }
    (config.run_dir / "stage2/policy_replay_report.md").write_text(
        "# HAST Stage 2 Policy Replay Report\n\n"
        + json.dumps(replay, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return policy


def select_stage3_parents(stage1: pd.DataFrame, programs: list[CandidateProgram], limit: int = 24) -> list[CandidateProgram]:
    by_id = {program.candidate_id: program for program in programs}
    valid = stage1[stage1["valid"].astype(bool)].sort_values("rank_score", ascending=False)
    picked: list[CandidateProgram] = []
    for row in valid.itertuples():
        program = by_id.get(str(row.candidate_id))
        if program and program.candidate_id not in {p.candidate_id for p in picked}:
            picked.append(program)
        if len(picked) >= limit:
            break
    if not picked:
        picked = programs[:limit]
    return picked


def rank01(series: pd.Series, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series([1.0] * len(series), index=series.index)
    ranked = numeric.rank(method="average", ascending=not higher_is_better)
    return (1.0 - (ranked - 1.0) / (len(series) - 1.0)).fillna(0.0)


def feature_signature_from_row(row: pd.Series) -> str:
    code_path = Path(str(row.get("code_path", "")))
    code = code_path.read_text(encoding="utf-8") if code_path.exists() else ""
    features = code_features(str(row.get("candidate_id", "")), code, str(code_path))
    keys = [
        "degree_backbone",
        "frontier",
        "weak_tie",
        "two_hop",
        "boundary",
        "redundancy",
        "phase",
        "heap_update",
    ]
    return "|".join(key for key in keys if features.get(key)) or "plain"


def first_unique_rows(parts: list[pd.DataFrame], limit: int) -> pd.DataFrame:
    usable = [part for part in parts if not part.empty]
    if not usable:
        return pd.DataFrame()
    out = pd.concat(usable, ignore_index=False)
    out = out.drop_duplicates(subset=["candidate_id"], keep="first")
    return out.head(limit)


def branch_seed_rows(parts: list[pd.DataFrame], limit: int) -> pd.DataFrame:
    """Preserve branch quotas; the same code may seed multiple branch prompts."""
    usable = [part.drop_duplicates(subset=["candidate_id"], keep="first") for part in parts if not part.empty]
    if not usable:
        return pd.DataFrame()
    out = pd.concat(usable, ignore_index=False)
    return out.head(limit)


def round_robin_by_signature(df: pd.DataFrame, score_col: str, n: int) -> pd.DataFrame:
    if df.empty or n <= 0:
        return df.head(0).copy()
    ranked = df.sort_values(score_col, ascending=False).copy()
    buckets = {name: sub for name, sub in ranked.groupby("feature_signature", sort=False)}
    rows: list[pd.Series] = []
    while len(rows) < n and any(len(bucket) for bucket in buckets.values()):
        for name in list(buckets):
            if len(rows) >= n:
                break
            bucket = buckets[name]
            if len(bucket):
                rows.append(bucket.iloc[0])
                buckets[name] = bucket.iloc[1:]
    return pd.DataFrame(rows)


def select_stage3_seed_nodes(
    stage1: pd.DataFrame,
    limit: int,
    fallback_record: dict[str, Any],
    fallback_program: CandidateProgram,
) -> tuple[list[dict[str, Any]], dict[str, CandidateProgram]]:
    valid = stage1[stage1["valid"].astype(bool)].copy()
    if not valid.empty:
        valid["feature_signature"] = valid.apply(feature_signature_from_row, axis=1)
        valid["q_seed_score"] = (
            0.45 * rank01(valid["delta_root_auc_cNBI"], True)
            + 0.25 * rank01(valid["auc_cNBI"], True)
            + 0.20 * rank01(valid["R"], False)
            + 0.10 * rank01(valid["time_s"], False)
        )
        auc_floor = valid["auc_cNBI"].quantile(0.55)
        speed_pool = valid[valid["auc_cNBI"] >= auc_floor].copy()
        speed_pool["s_seed_score"] = (
            0.40 * rank01(speed_pool["time_s"], False)
            + 0.25 * rank01(speed_pool["delta_root_auc_cNBI"], True)
            + 0.20 * rank01(speed_pool["R"], False)
            + 0.15 * rank01(speed_pool["auc_cNBI"], True)
        )
        valid["b_seed_score"] = (
            0.30 * rank01(valid["delta_root_auc_cNBI"], True)
            + 0.25 * rank01(valid["auc_cNBI"], True)
            + 0.25 * rank01(valid["R"], False)
            + 0.20 * rank01(valid["time_s"], False)
        )
    else:
        speed_pool = valid

    q_n = min(limit, max(1, round(limit * 8 / 24)))
    s_n = min(max(0, limit - q_n), max(1, round(limit * 6 / 24)))
    b_n = min(max(0, limit - q_n - s_n), max(1, round(limit * 6 / 24)))
    r_n = max(0, limit - q_n - s_n - b_n)

    picks: list[tuple[str, pd.DataFrame]] = []
    if not valid.empty:
        q_pick = valid.sort_values("q_seed_score", ascending=False).head(q_n).copy()
        q_pick["tree_branch"] = "Q"
        picks.append(("Q", q_pick))
        s_pick = speed_pool.sort_values("s_seed_score", ascending=False).head(s_n).copy() if not speed_pool.empty else valid.head(0).copy()
        s_pick["tree_branch"] = "S"
        picks.append(("S", s_pick))
        b_pick = round_robin_by_signature(valid, "b_seed_score", b_n).copy()
        b_pick["tree_branch"] = "B"
        picks.append(("B", b_pick))
        repair_pool = valid[
            valid["candidate_id"].isin(set(q_pick["candidate_id"]))
            | (valid["time_s"] >= valid["time_s"].quantile(0.75))
        ].copy()
        r_pick = repair_pool.sort_values("q_seed_score", ascending=False).head(r_n).copy()
        r_pick["tree_branch"] = "R"
        picks.append(("R", r_pick))
    selected = branch_seed_rows([part for _, part in picks], limit)
    if not selected.empty and len(selected) < limit:
        filler = valid[~valid["candidate_id"].isin(set(selected["candidate_id"]))].sort_values("rank_score", ascending=False)
        filler = filler.head(limit - len(selected)).copy()
        filler["tree_branch"] = "B"
        selected = branch_seed_rows([selected, filler], limit)
    seed_records: list[dict[str, Any]] = []
    seed_programs: dict[str, CandidateProgram] = {}
    for seed_index, (_, row) in enumerate(selected.iterrows(), start=1):
        code_path = Path(str(row.get("code_path", "")))
        if not code_path.exists():
            continue
        node_id = f"stage3-seed-{seed_index:04d}"
        family = str(row.get("family", "stage1-parent"))
        program = make_program(code_path.read_text(encoding="utf-8"), family=family, source_stage="stage3-seed")
        record = row.to_dict()
        record.update(
            {
                "node_id": node_id,
                "parent_node_id": "",
                "parent_candidate_id": "",
                "parent_auc_cNBI": float("nan"),
                "depth": 0,
                "stage_index": 0,
                "tree_role": "seed",
                "tree_branch": str(row.get("tree_branch", "B")),
            }
        )
        seed_records.append(record)
        seed_programs[node_id] = program
    if not seed_records:
        record = dict(fallback_record)
        record.update({"node_id": "stage3-seed-root", "depth": 0, "stage_index": 0, "tree_role": "seed", "tree_branch": "B"})
        seed_records.append(record)
        seed_programs["stage3-seed-root"] = fallback_program
    return seed_records, seed_programs


def write_stage3_final_selection(config: E4E6Config, stage3: pd.DataFrame) -> dict[str, Any]:
    final_dir = config.run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    records = stage3[stage3["valid"].astype(bool)].to_dict("records")
    frontier = pareto_frontier(records)
    final = select_final_q_s(frontier)
    (config.run_dir / "stage3_pareto_frontier.json").write_text(
        json.dumps(frontier, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (config.run_dir / "stage3_final_selection.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    code_manifest: dict[str, Any] = {}
    for label, row in final.items():
        if not row:
            code_manifest[label] = None
            continue
        source_path = Path(str(row.get("code_path", "")))
        out_path = final_dir / f"{label}.py"
        if not source_path.exists():
            code_manifest[label] = {"candidate_id": row.get("candidate_id"), "error": "code_path not found"}
            continue
        out_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
        code_manifest[label] = {
            "candidate_id": row.get("candidate_id"),
            "family": row.get("family"),
            "source_stage": row.get("source_stage"),
            "R": row.get("R"),
            "auc_cNBI": row.get("auc_cNBI"),
            "time_s": row.get("time_s"),
            "rank_score": row.get("rank_score"),
            "source_code_path": str(source_path),
            "code_path": str(out_path),
        }
    (final_dir / "final_code_manifest.json").write_text(
        json.dumps(code_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"frontier": frontier, "final": final, "final_code_manifest": code_manifest, "final_dir": str(final_dir)}


def write_prepare_manifest(config: E4E6Config, context: dict[str, Any]) -> dict[str, Any]:
    ensure_dirs(config)
    manifest = {
        "run_dir": str(config.run_dir),
        "input_parameters_path": str(config.run_dir / "input_parameters.json"),
        "llm": LLM_DEFAULTS,
        "budgets": {
            "stage1_candidates": config.stage1_budget,
            "stage2_llm_calls": config.stage2_budget,
            "stage3_candidates": config.stage3_budget,
            "candidates_per_llm_call": config.candidates_per_llm_call,
            "stage3_parent_limit": config.stage3_parent_limit,
            "candidate_timeout_s": config.candidate_timeout_s,
            "delta_credit_mode": config.delta_credit_mode,
        },
        "proxy_datasets": config.proxy_datasets,
        "full_datasets": config.full_datasets,
        "llm_workers": config.llm_workers,
        "llm_request_counts": {
            "stage1_tree_expansion_requests": config.stage1_budget,
            "stage2_policy_requests": config.stage2_budget,
            "stage3_tree_expansion_requests": config.stage3_budget,
        },
        "search_semantics": {
            "stage1": "sequential search-tree expansion from the HDA-original root",
            "stage2": "parallel policy/bounds induction with schema sanitization; does not generate candidate algorithms",
            "stage3": "Q/S/bridge/repair branch-constrained bounded expansion from archive-diverse Stage 1 parent nodes",
            "target_family_contract": "Stage 2 may tune but cannot disable bounded two-hop, phase-aware local scoring, or capped affected-node refresh.",
            "delta_auc_cNBI": "AUC-cNBI(child) - AUC-cNBI(root HDA node)"
            if config.delta_credit_mode == "root"
            else "AUC-cNBI(child) - AUC-cNBI(parent)",
            "delta_root_auc_cNBI": "AUC-cNBI(child) - AUC-cNBI(root HDA node)",
            "active_delta_credit": config.delta_credit_mode,
        },
        "e6_final_selection": "HAST-Final-Q/S are frozen by E6 from the Stage 3 proxy Pareto frontier.",
        "e7_evaluation_target": "E7 evaluates E6-frozen HAST-Final-Q/S only and must not reselect algorithms.",
        "paper_refresh_gate": "Refresh paper-facing artifacts only when E7 legacy_reference_check passes.",
        "benchmark_source": context["root"],
        "will_execute_llm": False,
        "will_run_e7": False,
        "api_key_source": "HAST_LLM_API_KEY or OPENAI_API_KEY environment variable",
    }
    (config.run_dir / "prepare_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_e4_e6(config: E4E6Config, provider: LLMProvider) -> dict[str, Any]:
    ensure_dirs(config)
    context = read_benchmark_context()
    graphs = load_proxy_graphs(config)
    root = evaluate_root(graphs, rate=0.30)
    root["graph_count"] = len(graphs)
    (config.run_dir / "root_proxy_metrics.json").write_text(json.dumps(root, ensure_ascii=False, indent=2), encoding="utf-8")
    root_code_path = config.run_dir / "candidates/stage1/root_hda_original.py"
    root_code_path.write_text(HDA_ROOT_CODE, encoding="utf-8")
    root_program = make_program(HDA_ROOT_CODE, family="HDA-original", source_stage="stage1-root")
    stage1_root_record = make_tree_seed_record(
        node_id="stage1-root",
        program=root_program,
        metrics=root,
        code_path=root_code_path,
        stage="stage1",
        depth=0,
    )

    stage1, stage1_programs, stage1_tree = generate_and_evaluate_tree_stage(
        config=config,
        provider=provider,
        stage="stage1",
        budget=config.stage1_budget,
        prompt_builder=lambda idx, parent, parent_record: build_stage1_prompt(
            idx, parent, parent_record, context, config.delta_credit_mode
        ),
        family_default="stage1-local",
        graphs=graphs,
        root_auc_cNBI=root["auc_cNBI"],
        initial_records=[stage1_root_record],
        initial_programs={"stage1-root": root_program},
        weights=STAGE1_WEIGHTS,
        parent_priority_mode="hast_stage1",
        c_hast=STAGE1_HAST_PARENT_C,
    )
    policy = run_stage2_induction(config, provider, stage1)
    stage3_seed_records, stage3_seed_programs = select_stage3_seed_nodes(
        stage1,
        limit=config.stage3_parent_limit,
        fallback_record=stage1_root_record,
        fallback_program=root_program,
    )
    (config.run_dir / "stage3_parent_ids.json").write_text(
        json.dumps(
            [
                {
                    "node_id": record["node_id"],
                    "candidate_id": record["candidate_id"],
                    "tree_branch": record.get("tree_branch", ""),
                    "rank_score": record.get("rank_score"),
                    "auc_cNBI": record.get("auc_cNBI"),
                }
                for record in stage3_seed_records
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    stage3, stage3_programs, stage3_tree = generate_and_evaluate_tree_stage(
        config=config,
        provider=provider,
        stage="stage3",
        budget=config.stage3_budget,
        prompt_builder=lambda idx, parent, parent_record, branch_role="B": build_stage3_prompt(
            idx, parent, parent_record, policy, context, config.delta_credit_mode, branch_role or "B"
        ),
        family_default="stage3-bounded-local",
        graphs=graphs,
        root_auc_cNBI=root["auc_cNBI"],
        initial_records=stage3_seed_records,
        initial_programs=stage3_seed_programs,
        weights=STAGE3_WEIGHTS,
        branch_for_index=stage3_branch_for_index,
        static_guard=static_contract_violation,
    )
    valid_ids = set(stage3[stage3["valid"].astype(bool)]["candidate_id"].astype(str))
    stage3_valid = stage3[stage3["candidate_id"].astype(str).isin(valid_ids)].copy()
    manifest_cols = ["candidate_id", "family", "source_stage", "R", "auc_cNBI", "time_s", "rank_score", "code_path"]
    valid_manifest = stage3_valid[[col for col in manifest_cols if col in stage3_valid.columns]].to_dict("records")
    (config.run_dir / "stage3_valid_candidate_manifest.json").write_text(
        json.dumps(valid_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    final_selection = write_stage3_final_selection(config, stage3)
    return {
        "run_dir": str(config.run_dir),
        "stage1_rows": len(stage1),
        "stage1_tree_nodes": len(stage1_tree),
        "stage2_policy": policy.to_dict(),
        "stage3_rows": len(stage3),
        "stage3_tree_nodes": len(stage3_tree),
        "stage3_valid_rows": len(stage3_valid),
        "stage3_valid_candidate_manifest": str(config.run_dir / "stage3_valid_candidate_manifest.json"),
        "stage3_pareto_frontier": str(config.run_dir / "stage3_pareto_frontier.json"),
        "stage3_final_selection": str(config.run_dir / "stage3_final_selection.json"),
        "final_dir": final_selection["final_dir"],
        "final_code_manifest": final_selection["final_code_manifest"],
    }
