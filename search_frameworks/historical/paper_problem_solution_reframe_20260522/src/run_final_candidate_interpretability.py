# -*- coding: utf-8 -*-
"""Deep interpretability package for the final HAST candidate group.

The analysis is intentionally tied to the found final algorithms:
- HAST-Final-Q: FAST21-cap24 trajectory from the existing full-12 detail file.
- HAST-Final-S: BT-n16-t8-u24 bounded-template candidate.

It produces candidate definitions, node/step behavior statistics, and bounded
template component ablations. It does not call any LLM API.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
TABLE_OUT = ROOT / "paper_tables"
FIG_OUT = ROOT / "figures"
REPORT_OUT = ROOT / "reports"
TREE_ROOT = WORKSPACE / "research" / "tree_search_ablation_20260520"
HAST_ROOT = WORKSPACE / "research" / "hast_experiment_20260521"
EVAL12_SRC = TREE_ROOT / "src" / "evaluate_final_12graphs.py"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


E12 = load_module(EVAL12_SRC, "interpret_eval12")


@dataclass(frozen=True)
class Weights:
    degree: float
    frontier: float
    weak_tie: float
    redundancy: float
    boundary: float
    leaf: float
    twohop: float
    bridge: float


def setup() -> None:
    for path in [TABLE_OUT, FIG_OUT, REPORT_OUT]:
        path.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )


def phase_weights(progress: float) -> Weights:
    if progress < 0.18:
        return Weights(1.20, 0.52, 0.22, 0.42, 0.20, 0.16, 0.10, 0.08)
    if progress < 0.62:
        return Weights(0.72, 1.18, 0.98, 0.60, 0.82, 0.34, 0.22, 0.26)
    return Weights(0.42, 1.42, 1.05, 0.78, 1.16, 0.76, 0.30, 0.40)


def local_terms(H: nx.Graph, u: Any, n0: int, cap_n: int = 16, cap_2: int = 8) -> dict[str, float]:
    if not H.has_node(u):
        return {}
    d = float(H.degree[u])
    nbrs = list(H.neighbors(u))
    if len(nbrs) > cap_n:
        nbrs = nbrs[:cap_n]
    ns = set(nbrs)
    frontier = weak_tie = redundancy = boundary = leaf_pressure = twohop = bridge = 0.0
    clustering = nx.clustering(H, u) if d > 1 else 0.0
    for v in nbrs:
        dv = H.degree[v]
        if dv <= 1:
            leaf_pressure += 2.0
        elif dv <= 3:
            frontier += 0.65
        frontier += 1.0 / (dv + 1.0)
        shared = external = scanned = 0
        for w in H.neighbors(v):
            if scanned >= cap_2:
                break
            if w == u:
                continue
            scanned += 1
            if w in ns:
                shared += 1
            else:
                external += 1
        redundancy += shared / (dv + 1.0)
        if external:
            weak_tie += external / (shared + 1.0)
            boundary += external / (dv + 1.0)
            bridge += external / (dv + shared + 1.0)
            twohop += external / (dv + 1.0)
    progress = 1.0 - H.number_of_nodes() / float(max(1, n0))
    w = phase_weights(progress)
    score = (
        w.degree * d
        + w.frontier * frontier
        + w.weak_tie * weak_tie
        + w.boundary * boundary
        + w.leaf * leaf_pressure
        + w.twohop * twohop
        + w.bridge * bridge
        - w.redundancy * redundancy
    )
    return {
        "degree": d,
        "frontier": frontier,
        "weak_tie": weak_tie,
        "redundancy": redundancy,
        "boundary": boundary,
        "leaf_pressure": leaf_pressure,
        "twohop": twohop,
        "bridge": bridge,
        "clustering": float(clustering),
        "score": float(score),
    }


def bt_order(graph: nx.Graph, variant: str = "full") -> list[Any]:
    import heapq

    H = graph.copy()
    n0 = max(1, H.number_of_nodes())
    order: list[Any] = []
    heap: list[tuple[float, str, int, Any]] = []
    stamp: dict[Any, int] = {}

    def score(u: Any) -> float | None:
        terms = local_terms(H, u, n0)
        if not terms:
            return None
        if variant == "no_degree":
            terms["degree"] = 0.0
        elif variant == "no_frontier_weak":
            terms["frontier"] = 0.0
            terms["weak_tie"] = 0.0
        elif variant == "no_twohop_boundary":
            terms["twohop"] = 0.0
            terms["boundary"] = 0.0
            terms["bridge"] = 0.0
        elif variant == "no_redundancy":
            terms["redundancy"] = 0.0
        progress = 1.0 - H.number_of_nodes() / float(n0)
        if variant == "no_phase":
            w = Weights(0.72, 1.18, 0.98, 0.60, 0.82, 0.34, 0.22, 0.26)
        else:
            w = phase_weights(progress)
        return (
            w.degree * terms["degree"]
            + w.frontier * terms["frontier"]
            + w.weak_tie * terms["weak_tie"]
            + w.boundary * terms["boundary"]
            + w.leaf * terms["leaf_pressure"]
            + w.twohop * terms["twohop"]
            + w.bridge * terms["bridge"]
            - w.redundancy * terms["redundancy"]
        )

    def push(u: Any) -> None:
        if H.has_node(u):
            s = score(u)
            if s is not None:
                stamp[u] = stamp.get(u, 0) + 1
                heapq.heappush(heap, (-s, str(u), stamp[u], u))

    for node in H.nodes():
        push(node)
    while H.number_of_nodes() > 0:
        while heap:
            _neg, _name, st, u = heapq.heappop(heap)
            if H.has_node(u) and stamp.get(u, 0) == st:
                break
        else:
            for node in list(H.nodes()):
                push(node)
            continue
        order.append(u)
        nbrs = list(H.neighbors(u))
        affected = set(nbrs)
        for v in nbrs[:24]:
            if H.has_node(v):
                cnt = 0
                for w in H.neighbors(v):
                    affected.add(w)
                    cnt += 1
                    if cnt >= 8:
                        break
        affected.discard(u)
        H.remove_node(u)
        for v in affected:
            push(v)
    return order


def complete_order(graph: nx.Graph, order: list[Any]) -> list[Any]:
    nodes = set(graph.nodes())
    seen = set()
    out = []
    for node in order:
        if node in nodes and node not in seen:
            seen.add(node)
            out.append(node)
    out.extend([node for node in graph.nodes if node not in seen])
    return out


def evaluate_order(dataset: str, method: str, order: list[Any], elapsed: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    graph = E12.EVAL.read_graph(dataset)
    rate = E12.EVAL.DATASET_RATES[dataset]
    metrics = E12.EVAL.compute_metrics(graph, order, rate=rate, method_time=elapsed)
    x = metrics["removal_ratio"].to_numpy(dtype=float)
    summary = {
        "dataset": dataset,
        "method": method,
        "R": float(metrics["GCC"].mean()),
        "auc_cNBI": E12.EVAL.auc_mean(x, metrics["cNBI"].to_numpy(dtype=float)),
        "time_s": elapsed,
    }
    return metrics, summary


def component_ablation() -> pd.DataFrame:
    variants = [
        ("full", "HAST-Final-S full"),
        ("no_degree", "remove residual degree"),
        ("no_frontier_weak", "remove frontier/weak-tie"),
        ("no_twohop_boundary", "remove two-hop/boundary"),
        ("no_redundancy", "remove redundancy penalty"),
        ("no_phase", "remove phase weights"),
    ]
    rows = []
    total = len(variants) * len(E12.EVAL.DATASETS)
    pbar = tqdm(total=total, desc="component knockout", unit="run", dynamic_ncols=True)
    for variant, label in variants:
        for dataset in E12.EVAL.DATASETS:
            pbar.set_postfix(variant=variant, dataset=dataset)
            try:
                graph = E12.EVAL.read_graph(dataset)
                t0 = time.perf_counter()
                order = bt_order(graph, variant=variant)
                elapsed = time.perf_counter() - t0
                _metrics, summary = evaluate_order(dataset, label, order, elapsed)
                summary["variant"] = variant
                rows.append(summary)
            finally:
                pbar.update(1)
    pbar.close()
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_OUT / "table_final_candidate_component_ablation_detail.csv", index=False, encoding="utf-8-sig")
    mean = out.groupby(["variant", "method"], as_index=False).agg(R=("R", "mean"), auc_cNBI=("auc_cNBI", "mean"), time_s=("time_s", "mean"))
    mean.to_csv(TABLE_OUT / "table_final_candidate_component_ablation.csv", index=False, encoding="utf-8-sig")
    return mean


def hda_order(graph: nx.Graph) -> list[Any]:
    H = graph.copy()
    order = []
    while H.number_of_nodes() > 0:
        node = sorted(H.degree(), key=lambda item: (-item[1], str(item[0])))[0][0]
        order.append(node)
        H.remove_node(node)
    return order


def selected_node_features() -> pd.DataFrame:
    methods: dict[str, Callable[[nx.Graph], list[Any]]] = {
        "HAST-Final-S": lambda g: bt_order(g, "full"),
        "HDA": hda_order,
    }
    rows = []
    total = len(E12.EVAL.DATASETS) * len(methods)
    pbar = tqdm(total=total, desc="selected-node features", unit="run", dynamic_ncols=True)
    for dataset in E12.EVAL.DATASETS:
        graph = E12.EVAL.read_graph(dataset)
        budget = max(1, int(round(graph.number_of_nodes() * min(0.10, E12.EVAL.DATASET_RATES[dataset]))))
        for method, fn in methods.items():
            pbar.set_postfix(dataset=dataset, method=method)
            H = graph.copy()
            order = complete_order(graph, fn(graph))[:budget]
            n0 = graph.number_of_nodes()
            for step, node in enumerate(order, start=1):
                terms = local_terms(H, node, n0)
                if not terms:
                    continue
                gcc_nodes = max(nx.connected_components(H), key=len) if H.number_of_nodes() else set()
                terms.update(
                    {
                        "dataset": dataset,
                        "method": method,
                        "step": step,
                        "removal_ratio": step / n0,
                        "in_gcc": bool(node in gcc_nodes),
                    }
                )
                rows.append(terms)
                if H.has_node(node):
                    H.remove_node(node)
            pbar.update(1)
    pbar.close()
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_OUT / "table_final_candidate_selected_node_features.csv", index=False, encoding="utf-8-sig")
    return out


def step_delta_summary() -> pd.DataFrame:
    curve_path = TABLE_OUT / "table_12graph_extended_curve_records.csv"
    if curve_path.exists():
        curves = pd.read_csv(curve_path, engine="python")
        curves = curves[curves["method"].isin(["HAST-Final-Q", "HAST-Final-S", "PUCT", "HDA", "CoreHD", "CI", "NDJC"])].copy()
    else:
        curves = pd.read_csv(TABLE_OUT / "unified_curve_records.csv", engine="python")
        curves = curves.rename(columns={"paper_label": "method"})
    rows = []
    for (dataset, method), sub in curves.groupby(["dataset", "method"]):
        sub = sub.sort_values("step").copy()
        if sub.empty:
            continue
        sub["delta_GCC"] = -sub["GCC"].diff().fillna(1.0 - sub["GCC"])
        sub["delta_cNBI"] = sub["cNBI"].diff().fillna(sub["cNBI"])
        early = sub[sub["removal_ratio"] <= 0.10]
        mid = sub[(sub["removal_ratio"] > 0.10) & (sub["removal_ratio"] <= 0.20)]
        for phase, part in [("early_0_10", early), ("mid_10_20", mid)]:
            if part.empty:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "phase": phase,
                    "mean_delta_GCC": float(part["delta_GCC"].mean()),
                    "mean_delta_cNBI": float(part["delta_cNBI"].mean()),
                    "mean_GCC": float(part["GCC"].mean()),
                    "mean_cNBI": float(part["cNBI"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_OUT / "table_final_candidate_step_delta_summary.csv", index=False, encoding="utf-8-sig")
    return out


def draw_component_ablation(mean: pd.DataFrame) -> None:
    order = ["full", "no_degree", "no_frontier_weak", "no_twohop_boundary", "no_redundancy", "no_phase"]
    plot = mean.set_index("variant").loc[order].reset_index()
    labels = ["Full", "-degree", "-frontier", "-twohop", "-redund.", "-phase"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    axes[0].bar(labels, plot["R"], color="#0072B2")
    axes[0].set_ylabel("Mean R/GCC (lower is better)")
    axes[0].set_title("GCC robustness")
    axes[1].bar(labels, plot["auc_cNBI"], color="#009E73")
    axes[1].set_ylabel("Mean auc-cNBI (higher is better)")
    axes[1].set_title("Residual fragmentation")
    axes[2].bar(labels, plot["time_s"], color="#D55E00")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Runtime (s, log)")
    axes[2].set_title("Runtime")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Component knockout: why the final bounded HAST candidate works", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "fig14_component_knockout_ablation.png", bbox_inches="tight")
    fig.savefig(FIG_OUT / "fig14_component_knockout_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_interpretability(features: pd.DataFrame, deltas: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    metrics = ["degree", "frontier", "weak_tie", "boundary", "redundancy", "clustering"]
    for ax, metric in zip(axes.ravel(), metrics):
        data = [features[features["method"].eq(m)][metric].dropna().to_numpy(dtype=float) for m in ["HDA", "HAST-Final-S"]]
        ax.boxplot(data, labels=["HDA", "HAST-S"], showfliers=False)
        ax.set_title(metric)
    fig.suptitle("Selected-node signatures in the first 10% removals", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "fig13_final_candidate_interpretability.png", bbox_inches="tight")
    fig.savefig(FIG_OUT / "fig13_final_candidate_interpretability.pdf", bbox_inches="tight")
    plt.close(fig)

    mean_delta = deltas.groupby(["method", "phase"], as_index=False).agg(mean_delta_GCC=("mean_delta_GCC", "mean"), mean_delta_cNBI=("mean_delta_cNBI", "mean"))
    keep = [m for m in ["HDA", "CoreHD", "CI", "NDJC", "PUCT", "HAST-Final-Q", "HAST-Final-S"] if m in set(mean_delta["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    for ax, col, title in [(axes[0], "mean_delta_GCC", "Per-step GCC drop"), (axes[1], "mean_delta_cNBI", "Per-step cNBI gain")]:
        pivot = mean_delta[mean_delta["method"].isin(keep)].pivot(index="method", columns="phase", values=col).reindex(keep)
        x = np.arange(len(pivot.index))
        width = 0.35
        for i, phase in enumerate(pivot.columns):
            ax.bar(x + (i - 0.5) * width, pivot[phase], width=width, label=phase)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=35, ha="right")
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "fig15_step_delta_interpretability.png", bbox_inches="tight")
    fig.savefig(FIG_OUT / "fig15_step_delta_interpretability.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(mean: pd.DataFrame, features: pd.DataFrame, deltas: pd.DataFrame) -> None:
    feature_mean = features.groupby("method")[["degree", "frontier", "weak_tie", "boundary", "redundancy", "clustering"]].mean().reset_index()
    lines = [
        "# 最终算法组深度可解释性报告",
        "",
        "## 最好算法组定义",
        "",
        "- `HAST-Final-Q`: quality-selected final candidate，对应内部 `FAST21-cap24`。",
        "- `HAST-Final-S`: speed-selected final candidate，对应内部 `BT-n16-t8-u24`。",
        "- 二者都是 HAST 自动搜索流程的输出，不是两个新框架。",
        "",
        "## Component knockout",
        "",
        mean.to_markdown(index=False),
        "",
        "## 早期删除节点特征均值",
        "",
        feature_mean.to_markdown(index=False),
        "",
        "## Step-level delta 摘要",
        "",
        deltas.groupby(["method", "phase"], as_index=False)[["mean_delta_GCC", "mean_delta_cNBI", "mean_GCC", "mean_cNBI"]].mean().head(40).to_markdown(index=False),
        "",
        "## 图表",
        "",
        "- `fig13_final_candidate_interpretability.png`",
        "- `fig14_component_knockout_ablation.png`",
        "- `fig15_step_delta_interpretability.png`",
    ]
    (REPORT_OUT / "final_candidate_interpretability_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup()
    mean = component_ablation()
    features = selected_node_features()
    deltas = step_delta_summary()
    draw_component_ablation(mean)
    draw_interpretability(features, deltas)
    write_report(mean, features, deltas)
    print(f"[done] wrote interpretability tables to {TABLE_OUT}")
    print(f"[done] wrote interpretability figures to {FIG_OUT}")


if __name__ == "__main__":
    main()
