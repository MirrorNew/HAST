# -*- coding: utf-8 -*-
"""Regenerate paper figures from record-derived experiment tables.

This is the canonical paper-facing plotting entrypoint for the standalone
``HAST2026/main`` project. It reads only consolidated CSV files under
``main/artifacts/source_tables`` and exports figures to
``main/artifacts/figures``.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

MAIN_ARTIFACTS = ROOT / "artifacts"
LOCAL_FIG_DIR = MAIN_ARTIFACTS / "figures"
SOURCE_TABLE_DIR = MAIN_ARTIFACTS / "source_tables"
BENCHMARK_TABLE_DIR = SOURCE_TABLE_DIR / "benchmark_12graph"
SEARCH_RUNTIME_TABLE_DIR = SOURCE_TABLE_DIR / "search_runtime"
SCALING_TABLE_DIR = SOURCE_TABLE_DIR / "scaling"
GRAPH_DIR = ROOT / "network"
OBS1_TABLE_DIR = SOURCE_TABLE_DIR / "motivation_observation" / "obs1_basic_baseline"

ERA_LABEL = "ERA-like"
METHOD_DISPLAY = {
    "PUCT": ERA_LABEL,
    "HAST-Final-Q": "HAST-Final-Q",
    "HAST-Final-S": "HAST-Final-S",
}

COL = {
    "hast_q": "#0072B2",
    "hast_s": "#009E73",
    "era": "#E69F00",
    "llm": "#CC79A7",
    "classic": "#7B8794",
    "strong": "#6B7280",
    "warning": "#D55E00",
    "light": "#56B4E9",
    "grid": "#D1D5DB",
    "text": "#1F2937",
}

OBS1_BLUE = "#2C7BB6"
OBS1_RED = "#D7191C"
OBS1_ORANGE = "#FDAE61"
OBS1_GRAY = "#BDBDBD"
OBS1_EDGE = "#9E9E9E"


def setup() -> None:
    LOCAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.2,
            "legend.fontsize": 7.6,
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


def save(fig: plt.Figure, stem: str) -> None:
    LOCAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(LOCAL_FIG_DIR / f"{stem}.png", facecolor="white")
    fig.savefig(LOCAL_FIG_DIR / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def compressed_log_time(seconds: float, low_log: float = -2.0, fast_compress: float = 0.35) -> float:
    x = np.log10(max(float(seconds), 10**low_log))
    return x * fast_compress if x < 0 else x


def display(name: str) -> str:
    return METHOD_DISPLAY.get(str(name), str(name))


def as_bool(series: pd.Series) -> pd.Series:
    return series.map(lambda x: str(x).strip().lower() in {"true", "1", "yes", "ok"})


def safe_method_filename(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(name)).strip("_")
    return text or "method"


def read_point_evaluations(dataset_order: list[str], methods: list[str], usecols: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset in dataset_order:
        point_dir = BENCHMARK_TABLE_DIR / dataset / "point_evaluations"
        for method in methods:
            path = point_dir / f"{safe_method_filename(method)}.csv"
            if not path.exists():
                continue
            cols = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.tolist()
            selected = [c for c in usecols if c in cols]
            frame = pd.read_csv(path, encoding="utf-8-sig", usecols=selected)
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=usecols)
    return pd.concat(frames, ignore_index=True)


def draw_quality_runtime() -> None:
    data = pd.read_csv(BENCHMARK_TABLE_DIR / "method_mean_metrics.csv", encoding="utf-8-sig")
    data = data[data["method"].ne("E26F")].copy()
    data["x_plot"] = data["mean_time_s"].map(compressed_log_time)
    hast_methods = {"HAST-Final-Q", "HAST-Final-S"}
    search_methods = {"PUCT", "FunSearch-like", "Clade-AHD-like", "MCTS-AHD-like", "AlphaEvolve-like"}
    strong = {"NCDC", "NDC", "NDJC", "BPD/MinSum-fallback", "GND-py", "VE-py", "LGD-RA2-py", "LGD-RA2num-py", "LGD-CND-py"}
    traditional = {"CoreHD", "HDA", "DC", "CI", "KCore", "CLUC"}
    colors = {
        "HAST-Final-Q": COL["hast_q"],
        "HAST-Final-S": COL["hast_s"],
        "PUCT": COL["era"],
        "FunSearch-like": "#CC79A7",
        "Clade-AHD-like": "#D55E00",
        "MCTS-AHD-like": "#009E73",
        "AlphaEvolve-like": "#7B8794",
        "NCDC": "#80B1D3",
        "NDC": "#8DD3C7",
        "NDJC": "#2CA02C",
        "BPD/MinSum-fallback": "#B15928",
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.9))
    for _, r in data.iterrows():
        method = r["method"]
        color = colors.get(method, "#6B7280")
        if method in hast_methods:
            ax.scatter(r["x_plot"], r["mean_auc_cNBI"], marker="*", s=280, color=color, edgecolor="#111827", linewidth=1.35, zorder=5)
        elif method in search_methods:
            ax.scatter(r["x_plot"], r["mean_auc_cNBI"], s=92, facecolor=color, edgecolor="none", alpha=0.88, zorder=3)
        elif method in strong:
            ax.scatter(r["x_plot"], r["mean_auc_cNBI"], s=62, color=color, edgecolor="none", alpha=0.82, zorder=2)
        elif method in traditional:
            ax.scatter(r["x_plot"], r["mean_auc_cNBI"], s=58, color="#9CA3AF", edgecolor="none", alpha=0.86, zorder=2)
        else:
            ax.scatter(r["x_plot"], r["mean_auc_cNBI"], s=50, color="#6B7280", alpha=0.70)
    offsets = {
        "HAST-Final-Q": (-22, 10),
        "HAST-Final-S": (-22, -14),
        "PUCT": (-4, 10),
        "FunSearch-like": (6, 5),
        "Clade-AHD-like": (6, -10),
        "NCDC": (-24, -14),
        "CoreHD": (-16, -14),
        "DC": (-18, 6),
    }
    for _, r in data.iterrows():
        method = r["method"]
        label = display(method)
        dx, dy = offsets.get(method, (5, 3))
        emphasis = method in hast_methods | search_methods | {"NCDC"}
        ax.annotate(
            label,
            (r["x_plot"], r["mean_auc_cNBI"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8 if emphasis else 6.8,
            fontweight="bold" if method in hast_methods else "normal",
            color="#111827" if emphasis else "#374151",
        )
    tick_powers = [3, 2, 1, 0, -1, -2]
    ax.set_xticks([compressed_log_time(10**p) for p in tick_powers])
    ax.set_xticklabels([rf"$10^{{{p}}}$" for p in tick_powers])
    ax.set_xlim(compressed_log_time(10**3) + 0.12, compressed_log_time(10**-2) - 0.08)
    ax.set_ylim(-18, max(data["mean_auc_cNBI"]) + 38)
    ax.axvspan(compressed_log_time(10**0), compressed_log_time(10**-2), color="#F3F4F6", alpha=0.7, zorder=0)
    ax.set_xlabel("Mean runtime per graph (s, reversed log; sub-second region compressed)")
    ax.set_ylabel("Mean auc-cNBI (higher is better)")
    ax.set_title("Quality-runtime summary on 12 benchmark graphs")
    save(fig, "fig13_12graph_quality_runtime_all_methods")


def draw_high_quality_panel() -> None:
    rows = pd.read_csv(BENCHMARK_TABLE_DIR / "method_mean_metrics.csv", encoding="utf-8-sig")
    by = rows.set_index("method")
    q_auc = float(by.loc["HAST-Final-Q", "mean_auc_cNBI"])
    q_time = float(by.loc["HAST-Final-Q", "mean_time_s"])
    selected = ["Clade-AHD-like", "FunSearch-like", "PUCT", "HAST-Final-Q", "HAST-Final-S", "AlphaEvolve-like", "HDA", "CoreHD"]
    labels = {"PUCT": ERA_LABEL}
    items = []
    for method in selected:
        items.append((method, float(by.loc[method, "mean_auc_cNBI"]) / q_auc * 100, q_time / float(by.loc[method, "mean_time_s"])))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(len(items))
    colors = [COL["hast_q"] if n == "HAST-Final-Q" else COL["hast_s"] if n == "HAST-Final-S" else COL["era"] if n == "PUCT" else "#B8C2CC" for n, _, _ in items]
    vals = [x[1] for x in items]
    axes[0].barh(y, vals, color=colors, height=0.62, edgecolor="white")
    axes[0].axvline(100, color="#777", ls="--", lw=1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([labels.get(n, n) for n, _, _ in items])
    axes[0].invert_yaxis()
    axes[0].set_xlim(50, 106)
    axes[0].set_xlabel("Quality vs. HAST-Final-Q (%)")
    axes[0].set_title("Quality normalized to HAST-Final-Q")
    for yi, v, (n, _, _) in zip(y, vals, items):
        axes[0].text(v + 0.8, yi, f"{v:.1f}%", va="center", fontsize=8, weight="bold" if n.startswith("HAST") else "normal")
    items2 = sorted(items, key=lambda z: z[2])
    y2 = np.arange(len(items2))
    vals2 = [x[2] for x in items2]
    colors2 = [COL["hast_q"] if n == "HAST-Final-Q" else COL["hast_s"] if n == "HAST-Final-S" else COL["era"] if n == "PUCT" else "#B8C2CC" for n, _, _ in items2]
    axes[1].barh(y2, vals2, color=colors2, height=0.62, edgecolor="white")
    axes[1].axvline(1, color="#777", ls="--", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels([labels.get(n, n) for n, _, _ in items2])
    axes[1].set_xlabel("Speed vs. HAST-Final-Q (log)")
    axes[1].set_title("Runtime normalized to\nHAST-Final-Q")
    for yi, v, (n, _, _) in zip(y2, vals2, items2):
        axes[1].text(v * 1.08, yi, f"{v:.2f}x" if v < 0.1 else f"{v:.1f}x", va="center", fontsize=8, weight="bold" if n.startswith("HAST") else "normal")
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig17_hast_quality_speed_panel")


def draw_framework_search_time() -> None:
    df = pd.read_csv(SEARCH_RUNTIME_TABLE_DIR / "framework_search_time_summary.csv", encoding="utf-8-sig")
    df["paper_label"] = df["paper_label"].replace({"PUCT": ERA_LABEL})
    order = ["ERA-like", "FunSearch-like", "Clade-AHD-like", "MCTS-AHD-like", "AlphaEvolve-like", "HAST free search", "HAST bounded search", "HAST online check"]
    df["paper_label"] = pd.Categorical(df["paper_label"], categories=order, ordered=True)
    df = df.sort_values("paper_label")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(len(df))
    colors = [COL["hast_q"] if g == "HAST stage" else "#B8C2CC" for g in df["group"]]
    labels = [str(x).replace(" ", "\n", 1) if len(str(x)) > 16 else str(x) for x in df["paper_label"]]
    axes[0].barh(y, df["mean_logged_search_s_per_candidate"], color=colors, height=0.62, edgecolor="white")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean logged search time / candidate (s)")
    axes[0].set_title("Candidate-level search cost")
    for yi, v in zip(y, df["mean_logged_search_s_per_candidate"]):
        axes[0].text(v + 1.2, yi, f"{v:.1f}s", va="center", fontsize=7.5)
    vals_h = df["total_logged_search_s"] / 3600
    axes[1].barh(y, vals_h, color=colors, height=0.62, edgecolor="white")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Total logged search time (hours)")
    axes[1].set_title("Total logged search cost")
    for yi, v in zip(y, vals_h):
        axes[1].text(v + 0.12, yi, f"{v:.2f}h", va="center", fontsize=7.5)
    fig.text(0.55, -0.02, "Logged search time = prompt_elapsed_s + candidate validation time; root excluded.", ha="center", fontsize=8, color="#555")
    save(fig, "fig20_framework_search_time")


def draw_curves() -> None:
    use_methods = ["HAST-Final-Q", "HAST-Final-S", "PUCT", "HDA", "CoreHD", "DC", "KCore", "CLUC", "CI", "NCDC", "NDC", "BPD/MinSum-fallback"]
    dataset_order = ["CEnew", "Collaboration", "condmat", "crime", "email", "Grid", "GrQC", "hamster", "HepPh", "PH", "Powerlaw_500", "Yeast"]
    curves = read_point_evaluations(dataset_order, use_methods, ["dataset", "method", "removal_ratio", "GCC", "cNBI"])
    curves = curves[curves["method"].isin(use_methods)].copy()
    method_colors = {
        "HAST-Final-Q": COL["hast_q"],
        "HAST-Final-S": COL["hast_s"],
        "PUCT": COL["era"],
        "HDA": COL["classic"],
        "CoreHD": "#6B7280",
        "NCDC": "#80B1D3",
        "NDC": "#8DD3C7",
        "BPD/MinSum-fallback": "#B15928",
        "DC": "#BBBBBB",
        "KCore": "#999999",
        "CLUC": "#999999",
        "CI": "#17BECF",
    }
    for metric, ylabel, stem in [("GCC", "GCC (lower is better)", "fig10_gcc_curves_12graphs"), ("cNBI", "cNBI (higher is better)", "fig11_cnbi_curves_12graphs")]:
        fig, axes = plt.subplots(3, 4, figsize=(16.4, 10.0), sharex=False, sharey=False)
        handles = {}
        for ax, dataset in zip(axes.ravel(), dataset_order):
            sub = curves[curves["dataset"].eq(dataset)]
            for method in use_methods:
                ms = sub[sub["method"].eq(method)].sort_values("removal_ratio")
                if ms.empty:
                    continue
                hi = method.startswith("HAST")
                label = display(method)
                (line,) = ax.plot(
                    ms["removal_ratio"],
                    ms[metric],
                    label=label,
                    color=method_colors.get(method, "#555"),
                    linewidth=2.4 if hi else 1.1,
                    linestyle="-" if hi or method == "PUCT" else ":",
                    alpha=1.0 if hi or method == "PUCT" else 0.72,
                )
                handles.setdefault(label, line)
            ax.set_title(dataset)
            ax.set_xlabel("Removal ratio")
            ax.set_ylabel(ylabel)
        order = [display(m) for m in use_methods if display(m) in handles]
        fig.legend([handles[m] for m in order], order, loc="lower center", ncol=5, frameon=False)
        fig.tight_layout(rect=[0, 0.09, 1, 1])
        save(fig, stem)


def _obs1_read_graph(dataset: str) -> nx.Graph:
    path = GRAPH_DIR / f"{dataset}.edgelist"
    if not path.exists():
        raise FileNotFoundError(f"Missing graph for Observation 1: {path}")
    graph = nx.read_edgelist(path, nodetype=int)
    graph = nx.Graph(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def _obs1_load_order(dataset: str, method: str) -> list[int]:
    path = OBS1_TABLE_DIR / "node_orders.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing node-order source for Observation 1: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["dataset"].eq(dataset) & df["method"].eq(method)].copy()
    if df.empty:
        raise FileNotFoundError(f"Missing node order for Observation 1: dataset={dataset}, method={method}")
    df = df[df["step"].astype(int) > 0].sort_values("step")
    return [int(x) for x in df["removed_node"].tolist()]


def _obs1_residual_graph(graph: nx.Graph, order: list[int], step: int) -> nx.Graph:
    removed = set(order[:step])
    return graph.subgraph([u for u in graph.nodes if u not in removed]).copy()


def _obs1_component_metrics(graph: nx.Graph, original_n: int) -> dict[str, float]:
    comps = sorted(nx.connected_components(graph), key=len, reverse=True)
    sizes = [len(c) for c in comps]
    largest = sizes[0] if sizes else 0
    top5 = sum(sizes[:5]) if sizes else 0
    rem = max(1, graph.number_of_nodes())
    hhi = sum(s * s for s in sizes) / (rem * rem)
    effective_components = 1.0 / hhi if hhi > 0 else 0.0
    return {
        "gcc": largest / original_n,
        "top5": top5 / original_n,
        "effective_components": effective_components,
        "components": float(len(comps)),
    }


def _obs1_packed_component_layout(graph: nx.Graph, seed: int, spread_top5: bool) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    comps = sorted([sorted(c) for c in nx.connected_components(graph)], key=len, reverse=True)
    if not comps:
        return {}
    max_s = max(len(c) for c in comps)
    centers: list[tuple[float, float]] = []
    for rank in range(len(comps)):
        if rank == 0:
            centers.append((-0.30 if spread_top5 else -0.12, 0.0))
        elif rank < 5 and spread_top5:
            angle = 2 * math.pi * (rank - 1) / 4 + math.pi / 7
            centers.append((0.85 * math.cos(angle), 0.62 * math.sin(angle)))
        else:
            idx = rank - (5 if spread_top5 else 1)
            ring = 1 + int(math.sqrt(max(0, idx) / 8))
            count = max(8, ring * 10)
            angle = 2 * math.pi * (idx % count) / count
            radius = (1.18 if spread_top5 else 0.72) + 0.25 * ring
            centers.append((radius * math.cos(angle), radius * math.sin(angle)))
    pos: dict[int, np.ndarray] = {}
    golden = math.pi * (3 - math.sqrt(5))
    for rank, comp in enumerate(comps):
        size = len(comp)
        cx, cy = centers[rank]
        comp_radius = 0.045 + 0.22 * math.sqrt(size / max_s)
        if rank > 0:
            comp_radius *= 0.70
        for i, node in enumerate(comp):
            if size == 1:
                x, y = cx, cy
            else:
                angle = i * golden + rng.normal(0, 0.012)
                r = comp_radius * math.sqrt((i + 0.5) / size)
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
            pos[int(node)] = np.array([x, y], dtype=float)
    return pos


def _obs1_node_style(graph: nx.Graph) -> tuple[list[int], list[str], list[float]]:
    comps = sorted(nx.connected_components(graph), key=len, reverse=True)
    color_by_node = {}
    size_by_node = {}
    for rank, comp in enumerate(comps):
        if rank == 0:
            color, size = OBS1_BLUE, 7
        elif rank < 5:
            color, size = (OBS1_RED if rank == 1 else OBS1_ORANGE), 6
        else:
            color, size = OBS1_GRAY, 4
        for node in comp:
            color_by_node[node] = color
            size_by_node[node] = size
    nodes = list(graph.nodes())
    return nodes, [color_by_node[u] for u in nodes], [size_by_node[u] for u in nodes]


def _obs1_draw_panel(
    ax: plt.Axes,
    graph: nx.Graph,
    title: str,
    seed: int,
    spread_top5: bool,
    metrics: dict[str, float],
) -> None:
    pos = _obs1_packed_component_layout(graph, seed, spread_top5)
    nodes, colors, sizes = _obs1_node_style(graph)
    edge_graph = graph
    max_edges = 1600
    if graph.number_of_edges() > max_edges:
        edges = list(graph.edges())
        rng = np.random.default_rng(graph.number_of_nodes() + graph.number_of_edges())
        keep = rng.choice(len(edges), size=max_edges, replace=False)
        edge_graph = nx.Graph()
        edge_graph.add_nodes_from(graph.nodes())
        edge_graph.add_edges_from(edges[int(i)] for i in keep)
    nx.draw_networkx_edges(edge_graph, pos, ax=ax, edge_color=OBS1_EDGE, alpha=0.10, width=0.22)
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=nodes,
        node_color=colors,
        node_size=sizes,
        linewidths=0.0,
        alpha=0.92,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_axis_off()
    ax.text(
        0.02,
        0.02,
        f"R={metrics['gcc']:.6f}\nTop-5 mass={metrics['top5']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#D0D0D0", alpha=0.92),
    )


def draw_obs1_horizontal() -> None:
    script = ROOT / "experiments" / "obs1_basic_baseline_horizontal.py"
    subprocess.run([sys.executable, str(script)], check=True)


def draw_scaling() -> None:
    full = pd.read_csv(SCALING_TABLE_DIR / "full_eval_500_to_10k_unified.csv", encoding="utf-8-sig")
    ok = full[as_bool(full["ok"])].copy()
    summary = ok.groupby(["method", "n"], as_index=False).agg(R=("R", "mean"), auc_cNBI=("auc_cNBI", "mean"), time_s=("time_s", "mean"))
    methods = ["HDA-original", "HDA-fast", "CoreHD-fast", "HAST-Final-S", "HAST-Final-Q"]
    colors = {"HDA-original": "#4C78A8", "HDA-fast": "#7EA6D8", "CoreHD-fast": "#6B7280", "HAST-Final-S": COL["hast_s"], "HAST-Final-Q": "#C44E52"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    for ax, (metric, ylabel, logy) in zip(axes, [("R", "mean R (lower)", False), ("auc_cNBI", "mean auc-cNBI (higher)", False), ("time_s", "mean runtime (s)", True)]):
        for method in methods:
            sub = summary[summary["method"].eq(method)].sort_values("n")
            if not sub.empty:
                ax.plot(sub["n"], sub[metric], marker="o", linewidth=2.0, label=method, color=colors[method])
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("nodes")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, "scaling_full_eval_500_to_10k_unified")
    runtime = pd.read_csv(SCALING_TABLE_DIR / "runtime_only_500_to_1000k_unified.csv", encoding="utf-8-sig")
    grouped = runtime.groupby(["method", "n"], as_index=False).agg(total=("ok", "size"), ok_count=("ok", lambda s: int(as_bool(s).sum())), time_s=("time_s", "mean"))
    complete = grouped[grouped["ok_count"].eq(grouped["total"])].copy()
    incomplete = grouped[(grouped["ok_count"] < grouped["total"]) & grouped["method"].eq("HDA-original")].copy()
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    for method in methods:
        sub = complete[complete["method"].eq(method)].sort_values("n")
        if not sub.empty:
            ax.plot(sub["n"], sub["time_s"], marker="o", linewidth=2.1, label=method, color=colors[method])
    if not incomplete.empty:
        ax.scatter(incomplete["n"], np.maximum(incomplete["time_s"].to_numpy(dtype=float), 300.0), marker="x", s=72, linewidths=2.2, color=colors["HDA-original"], label="HDA-original timeout/incomplete")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nodes")
    ax.set_ylabel("ordering runtime (s)")
    ax.set_title("Runtime-only Scaling")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "runtime_only_scaling_500_to_1000k_unified")


def main() -> None:
    setup()
    draw_obs1_horizontal()
    draw_quality_runtime()
    draw_high_quality_panel()
    draw_framework_search_time()
    draw_curves()
    draw_scaling()
    print(f"record-based figures written to {LOCAL_FIG_DIR}")


if __name__ == "__main__":
    main()
