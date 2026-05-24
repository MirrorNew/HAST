# -*- coding: utf-8 -*-
"""Evaluate lightweight distillations of the HAST-FAC frontier/weak-tie idea."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
TABLE_DIR = ROOT / "tables"
EVAL12_SRC = WORKSPACE / "research" / "tree_search_ablation_20260520" / "src" / "evaluate_final_12graphs.py"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


E12 = load_module(EVAL12_SRC, "hast_fac_distill_eval12")


def _heap_order(G: Any, score_fn: Callable[[Any, Any, Dict[Any, int], int], tuple]) -> List[Any]:
    import heapq

    H = G.copy()
    n0 = max(1, H.number_of_nodes())
    deg = {u: H.degree[u] for u in H.nodes()}
    heap = []
    stamp: Dict[Any, int] = {}
    order: List[Any] = []

    def push(u: Any) -> None:
        if u not in H:
            return
        deg[u] = H.degree[u]
        stamp[u] = stamp.get(u, 0) + 1
        score = score_fn(H, u, deg, n0)
        heapq.heappush(heap, (*score, stamp[u], u))

    for u in list(H.nodes()):
        push(u)

    while H.number_of_nodes() > 0:
        while heap:
            *_, st, u = heapq.heappop(heap)
            if u in H and stamp.get(u) == st:
                break
        else:
            for u in list(H.nodes()):
                push(u)
            continue

        nbrs = list(H.neighbors(u))
        affected = set(nbrs)
        for v in nbrs:
            if v in H:
                affected.update(H.neighbors(v))

        order.append(u)
        H.remove_node(u)
        affected.discard(u)

        for v in nbrs:
            deg[v] = deg.get(v, 0) - 1
        for x in affected:
            if x in H:
                push(x)

    return order


def fw_lite_1hop(G: Any) -> List[Any]:
    """Very cheap frontier/weak-tie distillation: only one-hop degree profile."""

    def score(H: Any, u: Any, deg: Dict[Any, int], n0: int) -> tuple:
        du = deg.get(u, H.degree[u])
        if du <= 0:
            return (0.0, str(u))
        frontier = 0.0
        weak = 0.0
        mid = 0.0
        spill = 0.0
        for v in H.neighbors(u):
            dv = deg.get(v, H.degree[v])
            frontier += 1.0 / (dv + 1.0)
            spill += max(0.0, dv - 1.0) / (dv + 1.0)
            if dv <= 2:
                weak += 1.0
            elif dv <= 4:
                mid += 1.0
        val = du + 1.8 * frontier + 1.45 * weak + 0.55 * mid + 0.28 * spill
        return (-val, str(u))

    return _heap_order(G, score)


def fw_lite_bounded2(G: Any) -> List[Any]:
    """Bounded two-hop version: keep the core boundary signal, cap scans per neighbor."""

    cap = 12

    def score(H: Any, u: Any, deg: Dict[Any, int], n0: int) -> tuple:
        du = deg.get(u, H.degree[u])
        if du <= 0:
            return (0.0, str(u))
        ns = set(H.neighbors(u))
        frontier = 0.0
        weak = 0.0
        exposure = 0.0
        bridge = 0.0
        redundancy = 0.0
        for v in ns:
            dv = deg.get(v, H.degree[v])
            frontier += 1.0 / (dv + 1.0)
            if dv <= 2:
                weak += 1.8
            elif dv <= 4:
                weak += 0.9

            shared = 0
            outside = 0
            low_out = 0
            checked = 0
            for w in H.neighbors(v):
                if w == u:
                    continue
                checked += 1
                if w in ns:
                    shared += 1
                else:
                    outside += 1
                    if deg.get(w, H.degree[w]) <= 2:
                        low_out += 1
                if checked >= cap:
                    break
            exposure += outside / (dv + 1.0)
            bridge += (outside + 0.7 * low_out) / (1.0 + shared + dv)
            redundancy += shared / (dv + 1.0)

        val = 0.82 * du + 1.15 * frontier + 1.15 * weak + 0.95 * exposure + 1.25 * bridge - 0.65 * redundancy
        return (-val, -exposure, str(u))

    return _heap_order(G, score)


def fw_lite_phase_bounded(G: Any) -> List[Any]:
    """Bounded two-hop with only one phase knob: early fracture, later degree cleanup."""

    cap = 8

    def score(H: Any, u: Any, deg: Dict[Any, int], n0: int) -> tuple:
        du = deg.get(u, H.degree[u])
        if du <= 0:
            return (0.0, str(u))
        progress = 1.0 - (H.number_of_nodes() / float(n0))
        early = 1.0 - progress
        ns = set(H.neighbors(u))
        weak = 0.0
        exposure = 0.0
        boundary = 0.0
        redundancy = 0.0
        for v in ns:
            dv = deg.get(v, H.degree[v])
            if dv <= 2:
                weak += 2.0
            elif dv <= 5:
                weak += 0.75
            outside = 0
            shared = 0
            checked = 0
            for w in H.neighbors(v):
                if w == u:
                    continue
                checked += 1
                if w in ns:
                    shared += 1
                else:
                    outside += 1
                if checked >= cap:
                    break
            exposure += outside / (dv + 1.0)
            boundary += outside / (1.0 + shared + dv)
            redundancy += shared / (dv + 1.0)
        val = (
            (0.62 + 0.55 * progress) * du
            + (1.00 + 0.45 * early) * weak
            + (1.15 + 0.55 * early) * exposure
            + (1.25 + 0.50 * early) * boundary
            - (0.50 + 0.25 * progress) * redundancy
        )
        return (-val, -exposure, str(u))

    return _heap_order(G, score)


METHODS: Dict[str, Callable[[Any], List[Any]]] = {
    "FW-Lite-1hop": fw_lite_1hop,
    "FW-Lite-Bounded2": fw_lite_bounded2,
    "FW-Lite-PhaseBounded": fw_lite_phase_bounded,
}


def evaluate() -> pd.DataFrame:
    rows = []
    for dataset in E12.EVAL.DATASETS:
        graph = E12.EVAL.read_graph(dataset)
        rate = E12.EVAL.DATASET_RATES[dataset]
        for method, fn in METHODS.items():
            t0 = time.perf_counter()
            order = list(fn(graph.copy()))
            elapsed = time.perf_counter() - t0
            metrics = E12.EVAL.compute_metrics(graph, order, rate=rate, method_time=elapsed)
            x = metrics["removal_ratio"].to_numpy(dtype=float)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "R": float(metrics["GCC"].mean()),
                    "auc_cNBI": E12.EVAL.auc_mean(x, metrics["cNBI"].to_numpy(dtype=float)),
                    "time_s": elapsed,
                }
            )
            print(f"[distill] {dataset}/{method}: time={elapsed:.3f}s", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    detail = evaluate()
    detail_path = TABLE_DIR / "hast_fac_distilled12_detail.csv"
    mean_path = TABLE_DIR / "hast_fac_distilled12_mean.csv"
    cmp_path = TABLE_DIR / "hast_fac_distilled12_comparison.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    mean = detail.groupby("method")[["R", "auc_cNBI", "time_s"]].mean().reset_index().sort_values("auc_cNBI", ascending=False)
    mean.to_csv(mean_path, index=False, encoding="utf-8-sig")

    existing = pd.read_csv(ROOT / "final_12graph_eval" / "hast_12graph_summary.csv")
    existing_mean = existing.groupby("method")[["R", "auc_cNBI", "time_s"]].mean().reset_index()
    fac = pd.read_csv(TABLE_DIR / "hast_fac_online_full12_mean.csv")
    fac = fac.assign(method=fac["candidate_idx"].map(lambda x: f"HAST-FAC #{int(x)}"))[["method", "R", "auc_cNBI", "time_s"]]
    comparison = pd.concat([mean, fac, existing_mean], ignore_index=True, sort=False).sort_values("auc_cNBI", ascending=False)
    comparison.to_csv(cmp_path, index=False, encoding="utf-8-sig")
    print(mean.to_string(index=False), flush=True)
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {mean_path}", flush=True)
    print(f"Wrote {cmp_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
