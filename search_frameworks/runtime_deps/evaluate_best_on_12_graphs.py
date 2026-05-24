# -*- coding: utf-8 -*-
"""Evaluate the discovered HDA-root candidate on 11 real graphs + 1 generated graph.

This script intentionally lives under the same new experiment folder. It reads
the best candidate from outputs/summary.json after dacts_search.py finishes,
then recomputes MY, E26F, CoreHD, and the original unoptimized HDA online.
Recorded point-evaluation tables are used for older baselines when available.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List

import heapq

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "data" / "search_framework_records" / "raw" / "tree_search_ablation_20260520" / "runs" / "DACTS-rerun"
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
GRAPH_DIR = DATA_DIR / "network"
POINT_RECORD_DIR = PROJECT_ROOT / "artifacts" / "source_tables" / "benchmark_12graph"
OUT_DIR = PROJECT_ROOT / "runs" / "DACTS-style" / "final_12graph_eval"
FIG_DIR = OUT_DIR / "figures"
RECORD_DIR = OUT_DIR / "records"

REAL_DATASETS = [
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
]
DATASETS = REAL_DATASETS + ["Powerlaw_500"]
DATASET_RATES = {name: 0.30 for name in DATASETS}
DATASET_RATES["email"] = 0.40

EXISTING_METHODS = ["DC", "KCORE", "CLUC", "CI"]
ONLINE_METHODS = ["MY", "E26F", "CoreHD", "HDA"]
METHODS = ONLINE_METHODS + EXISTING_METHODS

COLOR_MAP = {
    "MY": "#d62728",
    "CoreHD": "#8C564B",
    "HDA": "#E377C2",
    "DC": "#1F77B4",
    "KCORE": "#FF7F0E",
    "CC": "#9467BD",
    "EC": "#7F7F7F",
    "CLUC": "#BCBD22",
    "CI": "#17BECF",
    "BC": "#8c6d31",
    "NDJC": "#2CA02C",
    "CR": "#393b79",
    "FINDER": "#e377c2",
    "E26F": "#111111",
}

LINESTYLE = {
    "MY": "-",
    "E26F": "--",
    "CoreHD": "-.",
    "HDA": ":",
    "DC": "-",
    "KCORE": "-",
    "CC": "--",
    "EC": "--",
    "CLUC": ":",
    "CI": "-.",
}


def load_dacts_module():
    path = PROJECT_ROOT / "search_frameworks" / "runtime_deps" / "dacts_search.py"
    spec = importlib.util.spec_from_file_location("dacts_search_eval", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dacts_search_eval"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


DACTS = load_dacts_module()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
        }
    )


def ensure_dirs() -> None:
    for path in [OUT_DIR, FIG_DIR, RECORD_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_graph(dataset: str) -> nx.Graph:
    if dataset == "Powerlaw_500":
        return generate_powerlaw_network(500, 2.5, seed=42)
    candidates = [DATA_DIR / f"{dataset}.txt", GRAPH_DIR / f"{dataset}.edgelist"]
    for path in candidates:
        if path.exists():
            graph = nx.read_edgelist(path, nodetype=int)
            graph = nx.Graph(graph)
            graph.remove_edges_from(nx.selfloop_edges(graph))
            return graph
    raise FileNotFoundError(f"No graph file found for {dataset}")


def generate_powerlaw_network(n: int, gamma: float, seed: int = 42, k_min: int = 2) -> nx.Graph:
    rng = np.random.default_rng(seed)
    k_max = max(k_min + 1, int(math.sqrt(n) * 4))
    degree_values = np.arange(k_min, k_max + 1)
    weights = np.array([k ** (-gamma) for k in degree_values], dtype=float)
    weights /= weights.sum()
    degrees = rng.choice(degree_values, size=n, replace=True, p=weights).astype(int).tolist()
    if sum(degrees) % 2 == 1:
        degrees[0] += 1
    graph = nx.configuration_model(degrees, seed=seed)
    graph = nx.Graph(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph = nx.convert_node_labels_to_integers(graph)
    if graph.number_of_nodes() > 0 and not nx.is_connected(graph):
        comps = [list(c) for c in nx.connected_components(graph)]
        for a, b in zip(comps[:-1], comps[1:]):
            graph.add_edge(a[0], b[0])
    return graph


def normalize_node(value: Any) -> Any:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        f = float(text)
        if f.is_integer():
            return int(f)
    except ValueError:
        pass
    return text


def complete_order(graph: nx.Graph, order: Iterable[Any]) -> List[Any]:
    nodes = set(graph.nodes())
    out = []
    seen = set()
    for node in order:
        if node in nodes and node not in seen:
            seen.add(node)
            out.append(node)
    out.extend([node for node in graph.nodes if node not in seen])
    return out


def load_existing_order_and_time(dataset: str, method: str) -> tuple[List[Any], float] | None:
    method_name = {"KCORE": "KCore"}.get(method, method)
    path = POINT_RECORD_DIR / dataset / "point_evaluations" / f"{method_name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "removed_node" not in df.columns:
        return None
    method_time = float("nan")
    if "total_time_s" in df.columns and len(df) > 0:
        try:
            method_time = float(df["total_time_s"].dropna().iloc[0])
        except Exception:
            method_time = float("nan")
    order = [node for node in (normalize_node(x) for x in df["removed_node"].tolist()) if node is not None]
    return order, method_time


def best_config_from_summary() -> Any:
    summary_path = EXPERIMENT_ROOT / "outputs" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Search summary is not ready: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best = summary["best"]
    fields = DACTS.AlgoConfig.__dataclass_fields__
    kwargs = {key: best[key] for key in fields if key in best}
    cfg = DACTS.AlgoConfig(**kwargs)
    cfg.assign_id()
    return cfg


def corehd_order(graph: nx.Graph, rate: float) -> List[Any]:
    """Online CoreHD prefix: maintain the 2-core and remove highest residual degree inside it."""
    h = graph.to_undirected() if graph.is_directed() else graph
    nodes = list(h.nodes())
    budget = max(1, min(len(nodes), int(round(len(nodes) * rate))))
    alive = set(nodes)
    nbrs = {u: set(h.neighbors(u)) for u in nodes}
    deg = {u: len(nbrs[u]) for u in nodes}

    core = set(nodes)
    core_deg = {u: deg[u] for u in nodes}
    queue = [u for u in nodes if core_deg[u] < 2]
    while queue:
        u = queue.pop()
        if u not in core:
            continue
        core.remove(u)
        for v in nbrs[u]:
            if v in core:
                core_deg[v] -= 1
                if core_deg[v] < 2:
                    queue.append(v)

    version = {u: 0 for u in nodes}
    core_heap = [(-deg[u], str(u), u, version[u]) for u in core]
    all_heap = [(-deg[u], str(u), u, version[u]) for u in nodes]
    heapq.heapify(core_heap)
    heapq.heapify(all_heap)

    def peel_from_core(start_nodes: Iterable[Any]) -> None:
        q = [u for u in start_nodes if u in core and core_deg.get(u, 0) < 2]
        while q:
            x = q.pop()
            if x not in core:
                continue
            core.remove(x)
            for y in nbrs[x]:
                if y in core:
                    core_deg[y] -= 1
                    if core_deg[y] < 2:
                        q.append(y)

    order: List[Any] = []
    while alive and len(order) < budget:
        if core:
            while core_heap:
                _, _, u, vu = heapq.heappop(core_heap)
                if u in alive and u in core and vu == version[u]:
                    break
            else:
                u = max(core, key=lambda x: (deg.get(x, 0), str(x)))
        else:
            while all_heap:
                _, _, u, vu = heapq.heappop(all_heap)
                if u in alive and vu == version[u]:
                    break
            else:
                u = max(alive, key=lambda x: (deg.get(x, 0), str(x)))

        if u not in alive:
            continue
        order.append(u)
        alive.remove(u)
        was_core = u in core
        if was_core:
            core.remove(u)
        touched = set()
        for v in list(nbrs[u]):
            if v not in alive:
                continue
            nbrs[v].discard(u)
            deg[v] = len(nbrs[v] & alive)
            version[v] += 1
            touched.add(v)
            heapq.heappush(all_heap, (-deg[v], str(v), v, version[v]))
            if v in core:
                if was_core:
                    core_deg[v] -= 1
                heapq.heappush(core_heap, (-deg[v], str(v), v, version[v]))
        nbrs[u].clear()
        deg[u] = 0
        version[u] += 1
        if touched:
            peel_from_core(touched)
    used = set(order)
    order.extend([node for node in graph.nodes if node not in used])
    return order


def hda_simple_order(graph: nx.Graph, rate: float) -> List[Any]:
    """Original unoptimized HDA root: scan current graph for max degree each step."""
    h = graph.copy()
    order: List[Any] = []
    budget = max(1, min(h.number_of_nodes(), int(round(h.number_of_nodes() * rate))))
    for _ in range(budget):
        if h.number_of_nodes() == 0:
            break
        node = max(h.nodes, key=lambda x: (h.degree[x], str(x)))
        order.append(node)
        h.remove_node(node)
    used = set(order)
    order.extend([node for node in graph.nodes if node not in used])
    return order


class DSUWithSizes:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n
        self.active = [False] * n
        self.size_counts: Counter[int] = Counter()
        self.active_count = 0
        self.components = 0
        self.sum_sq = 0.0
        self.sum_pair = 0.0

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def _add_size(self, s: int) -> None:
        self.size_counts[s] += 1
        self.sum_sq += s * s
        self.sum_pair += s * (s - 1)

    def _remove_size(self, s: int) -> None:
        self.size_counts[s] -= 1
        if self.size_counts[s] <= 0:
            del self.size_counts[s]
        self.sum_sq -= s * s
        self.sum_pair -= s * (s - 1)

    def activate(self, i: int) -> None:
        if self.active[i]:
            return
        self.active[i] = True
        self.parent[i] = i
        self.size[i] = 1
        self.active_count += 1
        self.components += 1
        self._add_size(1)

    def union(self, a: int, b: int) -> None:
        if not self.active[a] or not self.active[b]:
            return
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        sa, sb = self.size[ra], self.size[rb]
        self._remove_size(sa)
        self._remove_size(sb)
        self.parent[rb] = ra
        self.size[ra] = sa + sb
        self.components -= 1
        self._add_size(sa + sb)

    def snapshot(self, original_n: int) -> dict[str, float]:
        sizes: list[int] = []
        for size, count in self.size_counts.items():
            sizes.extend([size] * count)
        sizes.sort(reverse=True)
        largest = sizes[0] if sizes else 0
        top5 = sum(sizes[:5]) if sizes else 0
        rem = max(1, self.active_count)
        denom_pairs = max(1, original_n * (original_n - 1))
        hhi_remaining = self.sum_sq / (rem * rem)
        effective_components = 1.0 / hhi_remaining if hhi_remaining > 0 else 0.0
        pairwise_disconnected = 1.0 - self.sum_pair / denom_pairs
        top5_mass = top5 / original_n
        cnbi = pairwise_disconnected * effective_components / (1.0 + top5_mass)
        return {
            "ACC": largest / original_n,
            "GCC": largest / original_n,
            "NCC": float(self.components),
            "top5_component_mass": top5_mass,
            "hhi_remaining": hhi_remaining,
            "effective_components": effective_components,
            "pairwise_disconnected": pairwise_disconnected,
            "cNBI": cnbi,
        }


def compute_metrics(graph: nx.Graph, order: Iterable[Any], rate: float, method_time: float) -> pd.DataFrame:
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return pd.DataFrame()
    full_order = complete_order(graph, order)
    steps = max(1, min(len(full_order), int(round(n * rate))))
    prefix = full_order[:steps]
    removed = set(prefix)
    idx = {u: i for i, u in enumerate(nodes)}
    dsu = DSUWithSizes(n)

    for u in nodes:
        if u not in removed:
            dsu.activate(idx[u])
    for u, v in graph.edges():
        iu, iv = idx[u], idx[v]
        if dsu.active[iu] and dsu.active[iv]:
            dsu.union(iu, iv)

    rows: list[dict[str, float]] = []
    for k in range(steps, 0, -1):
        snap = dsu.snapshot(n)
        snap["step"] = k
        snap["removal_ratio"] = k / n
        rows.append(snap)
        u = prefix[k - 1]
        iu = idx[u]
        dsu.activate(iu)
        for v in graph.neighbors(u):
            dsu.union(iu, idx[v])
    rows.reverse()
    df = pd.DataFrame(rows)
    df["running_R"] = df["GCC"].expanding().mean()
    df["running_auc_cNBI"] = df["cNBI"].expanding().mean()
    df["total_time_s"] = method_time
    return df


def auc_mean(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float(y[0]) if len(y) else float("nan")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    span = x[-1] - x[0]
    if span <= 0:
        return float(np.mean(y))
    return float(np.trapezoid(y, x) / span)


def evaluate_dataset(dataset: str, best_cfg: Any) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    graph = read_graph(dataset)
    rate = DATASET_RATES[dataset]
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    online_orders: dict[str, tuple[list[Any], float]] = {}
    t0 = time.perf_counter()
    online_orders["MY"] = (DACTS.degree_order_by_config(graph, best_cfg, budget_ratio=rate), time.perf_counter() - t0)
    t0 = time.perf_counter()
    online_orders["E26F"] = (DACTS.degree_order_by_config(graph, DACTS.e26f_config(), budget_ratio=rate), time.perf_counter() - t0)
    t0 = time.perf_counter()
    online_orders["CoreHD"] = (corehd_order(graph, rate), time.perf_counter() - t0)
    t0 = time.perf_counter()
    online_orders["HDA"] = (hda_simple_order(graph, rate), time.perf_counter() - t0)

    for method in METHODS:
        if method in online_orders:
            order, method_time = online_orders[method]
            source = "online"
        else:
            existing = load_existing_order_and_time(dataset, method)
            if existing is None:
                continue
            order, method_time = existing
            source = "existing_record"

        metrics = compute_metrics(graph, order, rate=rate, method_time=method_time)
        if metrics.empty:
            continue
        metrics.insert(0, "method", method)
        metrics.insert(0, "dataset", dataset)
        metrics.insert(2, "source", source)
        rows.append(metrics)
        x = metrics["removal_ratio"].to_numpy(dtype=float)
        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                "source": source,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "rate": rate,
                "steps": int(metrics["step"].max()),
                "R": float(metrics["GCC"].mean()),
                "auc_ACC": auc_mean(x, metrics["ACC"].to_numpy(dtype=float)),
                "auc_NCC": auc_mean(x, metrics["NCC"].to_numpy(dtype=float)),
                "auc_cNBI": auc_mean(x, metrics["cNBI"].to_numpy(dtype=float)),
                "final_ACC": float(metrics["ACC"].iloc[-1]),
                "final_NCC": float(metrics["NCC"].iloc[-1]),
                "final_cNBI": float(metrics["cNBI"].iloc[-1]),
                "time_s": method_time,
            }
        )
        metrics.to_csv(RECORD_DIR / f"{dataset}_{method}_metrics.csv", index=False, encoding="utf-8-sig")
    return rows, summaries


def plot_metric_grid(all_df: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(15.5, 9.5), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, dataset in zip(axes, DATASETS):
        sub = all_df[all_df["dataset"] == dataset]
        for method in METHODS:
            ms = sub[sub["method"] == method]
            if ms.empty:
                continue
            lw = 2.5 if method == "MY" else 1.2
            alpha = 1.0 if method in {"MY", "E26F", "CoreHD"} else 0.82
            ax.plot(
                ms["removal_ratio"],
                ms[metric],
                label=method,
                color=COLOR_MAP.get(method, "#333333"),
                linestyle=LINESTYLE.get(method, "-"),
                linewidth=lw,
                alpha=alpha,
            )
        ax.set_title(dataset)
        ax.set_xlabel("Removal ratio")
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FIG_DIR / filename)
    plt.close(fig)


def plot_auc_bar(summary: pd.DataFrame) -> None:
    pivot = summary.pivot_table(index="dataset", columns="method", values="auc_cNBI", aggfunc="mean").reindex(DATASETS)
    methods = [m for m in METHODS if m in pivot.columns]
    x = np.arange(len(pivot.index))
    width = min(0.08, 0.78 / max(1, len(methods)))
    fig, ax = plt.subplots(figsize=(16.0, 5.8))
    for i, method in enumerate(methods):
        ax.bar(
            x + (i - (len(methods) - 1) / 2) * width,
            pivot[method].to_numpy(dtype=float),
            width=width,
            label=method,
            color=COLOR_MAP.get(method, "#333333"),
            edgecolor="black" if method == "MY" else "none",
            linewidth=0.7 if method == "MY" else 0.0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.set_ylabel("AUC-cNBI (higher is better)")
    ax.set_title("Concentration-aware fragmentation AUC across 12 graphs")
    ax.legend(ncol=min(6, len(methods)), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "auc_cNBI_comparison_bar.png")
    plt.close(fig)


def plot_time(summary: pd.DataFrame) -> None:
    timed = summary.dropna(subset=["time_s"]).copy()
    if timed.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    means = timed.groupby("method")["time_s"].mean().sort_values()
    ax.barh(
        means.index,
        means.values,
        color=[COLOR_MAP.get(m, "#333333") for m in means.index],
    )
    ax.set_xlabel("Mean ordering time per graph (s)")
    ax.set_title("Online runtime comparison")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "time_mean_by_method.png")
    plt.close(fig)

    pivot = timed.pivot_table(index="dataset", columns="method", values="time_s", aggfunc="mean").reindex(DATASETS)
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    for method in [m for m in METHODS if m in pivot.columns]:
        ax.plot(
            pivot.index,
            pivot[method],
            marker="o",
            label=method,
            color=COLOR_MAP.get(method, "#333333"),
            linewidth=2.2 if method == "MY" else 1.4,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Ordering time (s, log scale)")
    ax.set_title("Online runtime by dataset")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "time_by_dataset_method.png")
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    mean_rank = []
    for metric, higher in [("R", False), ("auc_cNBI", True), ("time_s", False)]:
        sub = summary.copy()
        if metric == "time_s":
            sub = sub.dropna(subset=["time_s"]).copy()
        ranks = []
        for dataset, group in sub.groupby("dataset"):
            group = group.dropna(subset=[metric])
            if group.empty:
                continue
            rank = group[metric].rank(ascending=not higher, method="average")
            for idx, val in rank.items():
                ranks.append({"metric": metric, "dataset": dataset, "method": group.loc[idx, "method"], "rank": float(val)})
        mean_rank.extend(ranks)
    rank_df = pd.DataFrame(mean_rank)
    rank_df.to_csv(OUT_DIR / "metric_ranks.csv", index=False, encoding="utf-8-sig")
    mean_metrics = summary.groupby("method")[["R", "auc_cNBI", "time_s"]].mean(numeric_only=True).reset_index()
    mean_ranks = rank_df.groupby(["metric", "method"])["rank"].mean().reset_index()

    def markdown_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                vals.append(f"{val:.6g}" if isinstance(val, (float, np.floating)) else str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# 12-Graph Evaluation Report",
        "",
        "- MY is the best candidate discovered by the HDA-root DACTS search.",
        "- E26F, CoreHD, and original unoptimized HDA are recomputed online in this script.",
        "- DC/KCORE/CC/EC/CLUC/CI use existing trusted removal records when available.",
        "",
        "## Mean Metrics",
        "",
        markdown_table(mean_metrics),
        "",
        "## Mean Ranks",
        "",
        markdown_table(mean_ranks),
    ]
    (OUT_DIR / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-datasets", type=int, default=None)
    args = parser.parse_args()

    setup_style()
    ensure_dirs()
    best_cfg = best_config_from_summary()
    selected = DATASETS[: args.max_datasets] if args.max_datasets else DATASETS

    all_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in selected:
        print(f"[eval] {dataset}", flush=True)
        rows, summaries = evaluate_dataset(dataset, best_cfg)
        all_rows.extend(rows)
        summary_rows.extend(summaries)

    all_df = pd.concat(all_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    all_df.to_csv(OUT_DIR / "all_trajectory_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False, encoding="utf-8-sig")

    plot_metric_grid(all_df, "ACC", "ACC / GCC", "acc_curves_12graphs.png")
    plot_metric_grid(all_df, "running_R", "Running R", "running_R_curves_12graphs.png")
    plot_metric_grid(all_df, "NCC", "NCC", "ncc_curves_12graphs.png")
    plot_metric_grid(all_df, "cNBI", "cNBI", "cNBI_curves_12graphs.png")
    plot_auc_bar(summary)
    plot_time(summary)
    write_report(summary)
    print(f"[done] outputs: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
