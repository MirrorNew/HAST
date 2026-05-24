# -*- coding: utf-8 -*-
"""Search/evaluation orchestration for HAST-Lite-Full."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import networkx as nx
import pandas as pd

from baselines.algorithms import hda_original_order
from metrics.fragmentation import compute_metrics, summarize_metrics

from .candidate import CandidateProgram, compile_candidate
from .config import STAGE1_WEIGHTS, STAGE3_WEIGHTS, SearchWeights
from .ranking import add_rank_scores, pareto_frontier, select_final_q_s
from .stage2 import BoundPolicy, induce_bounds_from_log


@dataclass
class EvaluationRecord:
    candidate_id: str
    family: str
    source_stage: str
    valid: bool
    error: str
    R: float
    auc_cNBI: float
    auc_ACC: float
    auc_NCC: float
    final_ACC: float
    final_NCC: float
    final_cNBI: float
    time_s: float
    graph_count: int


def evaluate_order_fn(
    order_fn: Callable[[nx.Graph], list],
    graphs: Iterable[nx.Graph],
    rate: float,
) -> dict[str, float]:
    summaries = []
    total_time = 0.0
    for graph in graphs:
        t0 = time.perf_counter()
        order = order_fn(graph.copy())
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        df = compute_metrics(graph, order, rate=rate, method_time=elapsed)
        summaries.append(summarize_metrics(df))
    out: dict[str, float] = {}
    for key in ["R", "auc_cNBI", "auc_ACC", "auc_NCC", "final_ACC", "final_NCC", "final_cNBI"]:
        out[key] = float(sum(row[key] for row in summaries) / max(1, len(summaries)))
    out["time_s"] = total_time / max(1, len(summaries))
    return out


def evaluate_candidate(program: CandidateProgram, graphs: list[nx.Graph], rate: float) -> EvaluationRecord:
    try:
        runner = compile_candidate(program)
        summary = evaluate_order_fn(runner, graphs, rate)
        return EvaluationRecord(
            candidate_id=program.candidate_id,
            family=program.family,
            source_stage=program.source_stage,
            valid=True,
            error="",
            graph_count=len(graphs),
            **summary,
        )
    except Exception as exc:
        return EvaluationRecord(
            candidate_id=program.candidate_id,
            family=program.family,
            source_stage=program.source_stage,
            valid=False,
            error=str(exc),
            R=float("nan"),
            auc_cNBI=float("nan"),
            auc_ACC=float("nan"),
            auc_NCC=float("nan"),
            final_ACC=float("nan"),
            final_NCC=float("nan"),
            final_cNBI=float("nan"),
            time_s=float("nan"),
            graph_count=len(graphs),
        )


def evaluate_programs(
    programs: list[CandidateProgram],
    graphs: list[nx.Graph],
    rate: float,
    weights: SearchWeights,
    root_auc_cNBI: float | None = None,
) -> pd.DataFrame:
    records = [asdict(evaluate_candidate(program, graphs, rate)) for program in programs]
    df = pd.DataFrame(records)
    return add_rank_scores(df, weights=weights, root_auc_cNBI=root_auc_cNBI)


def evaluate_root(graphs: list[nx.Graph], rate: float) -> dict[str, float]:
    return evaluate_order_fn(lambda g: hda_original_order(g, rate=None), graphs, rate)


def run_three_stage_smoke(programs_stage1: list[CandidateProgram], programs_stage3: list[CandidateProgram], graphs: list[nx.Graph], rate: float):
    """Small deterministic execution path used by smoke tests and dry runs."""
    root = evaluate_root(graphs, rate)
    stage1 = evaluate_programs(programs_stage1, graphs, rate, STAGE1_WEIGHTS, root_auc_cNBI=root["auc_cNBI"])
    policy: BoundPolicy = induce_bounds_from_log(stage1)
    stage3 = evaluate_programs(programs_stage3, graphs, rate, STAGE3_WEIGHTS, root_auc_cNBI=root["auc_cNBI"])
    frontier = pareto_frontier(stage3.to_dict("records"))
    final = select_final_q_s(frontier)
    return {"root": root, "stage1": stage1, "policy": policy, "stage3": stage3, "frontier": frontier, "final": final}
