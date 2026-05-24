#!/usr/bin/env python3
"""Create a horizontal Observation-1 visual using only basic baselines.

cNBI is used only to select visually useful same-GCC/R cases. The figure itself
does not display cNBI because the metric is introduced later as a solution.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "network"
GRAPH_DIR = DATA_DIR
OUT_FIG = ROOT / "artifacts" / "figures"
OUT_TABLE = ROOT / "artifacts" / "source_tables" / "motivation_observation" / "obs1_basic_baseline"

BASIC_METHODS = ["HDA-fast", "CI", "DC", "KCore", "CLUC"]
DATASETS = ["Grid", "Collaboration", "HepPh", "email", "crime", "CEnew", "Yeast"]
EXTENDED_RECORD_DIR = ROOT / "artifacts" / "source_tables" / "benchmark_12graph"
NODE_ORDER_PATH = OUT_TABLE / "node_orders.csv"
FIXED_CASES = [
    {"dataset": "Collaboration", "method_a": "HDA-fast", "method_b": "CLUC", "step_a": 931, "step_b": 1243},
    {"dataset": "Collaboration", "method_a": "HDA-fast", "method_b": "CI", "step_a": 781, "step_b": 601},
    {"dataset": "Collaboration", "method_a": "CI", "method_b": "CLUC", "step_a": 577, "step_b": 1057},
]

BLUE = "#2C7BB6"
RED = "#D7191C"
ORANGE = "#FDAE61"
GRAY = "#BDBDBD"
EDGE = "#9E9E9E"


def setup() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def read_graph(dataset: str) -> nx.Graph:
    for path in [DATA_DIR / f"{dataset}.txt", GRAPH_DIR / f"{dataset}.edgelist"]:
        if path.exists():
            graph = nx.read_edgelist(path, nodetype=int)
            graph = nx.Graph(graph)
            graph.remove_edges_from(nx.selfloop_edges(graph))
            return graph
    raise FileNotFoundError(f"Cannot find graph for {dataset}")


def load_order(dataset: str, method: str) -> list[int]:
    if not NODE_ORDER_PATH.exists():
        raise FileNotFoundError(f"Missing Observation-1 node-order source: {NODE_ORDER_PATH}")
    df = pd.read_csv(NODE_ORDER_PATH, encoding="utf-8-sig")
    df = df[df["dataset"].eq(dataset) & df["method"].eq(method)].copy()
    if df.empty:
        raise FileNotFoundError(f"Missing Observation-1 node order: dataset={dataset}, method={method}")
    df = df[df["step"].astype(int) > 0].sort_values("step")
    return [int(x) for x in df["removed_node"].tolist()]


def residual_graph(graph: nx.Graph, order: list[int], step: int) -> nx.Graph:
    removed = set(order[:step])
    return graph.subgraph([u for u in graph.nodes() if u not in removed]).copy()


def component_metrics(graph: nx.Graph, original_n: int) -> dict[str, float]:
    comps = sorted(nx.connected_components(graph), key=len, reverse=True)
    sizes = [len(c) for c in comps]
    largest = sizes[0] if sizes else 0
    top5 = sum(sizes[:5]) if sizes else 0
    rem = max(1, graph.number_of_nodes())
    pair_connected_remaining = sum(s * (s - 1) for s in sizes)
    pairwise_disconnected = 1.0 - pair_connected_remaining / max(1, original_n * (original_n - 1))
    hhi = sum(s * s for s in sizes) / (rem * rem)
    effective_components = 1.0 / hhi if hhi > 0 else 0.0
    hidden_selection_score = pairwise_disconnected * effective_components / (1.0 + top5 / original_n)
    return {
        "gcc": largest / original_n,
        "top5": top5 / original_n,
        "effective_components": effective_components,
        "components": float(len(comps)),
        "hidden_selection_score": hidden_selection_score,
    }


def find_cases() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        records: dict[str, pd.DataFrame] = {}
        for method in BASIC_METHODS:
            path = EXTENDED_RECORD_DIR / dataset / "point_evaluations" / f"{method}.csv"
            if path.exists():
                records[method] = pd.read_csv(path, encoding="utf-8-sig")

        for method_a, method_b in itertools.combinations(records, 2):
            df_a = records[method_a]
            df_b = records[method_b]
            stride_a = max(1, len(df_a) // 180)
            stride_b = max(1, len(df_b) // 180)
            sampled_a = df_a.iloc[::stride_a]
            sampled_b = df_b.iloc[::stride_b]
            for _, row_a in sampled_a.iterrows():
                gcc_a = float(row_a["GCC"])
                close_b = sampled_b[(sampled_b["GCC"].astype(float) - gcc_a).abs() <= 0.003]
                if close_b.empty:
                    continue
                row_b = close_b.iloc[(close_b["GCC"].astype(float) - gcc_a).abs().argmin()]
                step_a = int(row_a["step"])
                step_b = int(row_b["step"])
                hidden_score_a = float(row_a["cNBI"])
                hidden_score_b = float(row_b["cNBI"])
                hidden_gap = abs(hidden_score_a - hidden_score_b)
                if hidden_gap < 15:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "method_a": method_a,
                        "method_b": method_b,
                        "step_a": step_a,
                        "step_b": step_b,
                        "q_a": float(row_a["removal_ratio"]),
                        "q_b": float(row_b["removal_ratio"]),
                        "gcc_a": float(row_a["GCC"]),
                        "gcc_b": float(row_b["GCC"]),
                        "gcc_abs_diff": abs(float(row_a["GCC"]) - float(row_b["GCC"])),
                        "top5_a": float(row_a["top5_component_mass"]),
                        "top5_b": float(row_b["top5_component_mass"]),
                        "components_a": float(row_a["NCC"]),
                        "components_b": float(row_b["NCC"]),
                        "effective_components_a": float(row_a["effective_components"]),
                        "effective_components_b": float(row_b["effective_components"]),
                        "hidden_score_a": hidden_score_a,
                        "hidden_score_b": hidden_score_b,
                        "hidden_gap": hidden_gap,
                    }
                )
    cases = pd.DataFrame(rows)
    cases = cases.sort_values(["hidden_gap", "gcc_abs_diff"], ascending=[False, True])
    cases.to_csv(
        OUT_TABLE / "motivation_obs1_basic_baseline_horizontal_candidate_pool.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected: list[dict[str, object]] = []
    for fixed in FIXED_CASES:
        match = cases[
            (cases["dataset"] == fixed["dataset"])
            & (cases["method_a"] == fixed["method_a"])
            & (cases["method_b"] == fixed["method_b"])
            & (cases["step_a"] == fixed["step_a"])
            & (cases["step_b"] == fixed["step_b"])
        ]
        if match.empty:
            raise RuntimeError(f"Missing fixed same-R case: {fixed}")
        row = match.iloc[0].to_dict()
        if abs(float(row["gcc_abs_diff"])) > 1e-12:
            raise RuntimeError(f"Fixed case is not exact same-R: {row}")
        selected.append(row)
    pd.DataFrame(selected).to_csv(
        OUT_TABLE / "motivation_obs1_basic_baseline_horizontal_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return selected


def packed_component_layout(graph: nx.Graph, seed: int, spread_top5: bool) -> dict[int, np.ndarray]:
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


def node_style(graph: nx.Graph) -> tuple[list[int], list[str], list[float]]:
    comps = sorted(nx.connected_components(graph), key=len, reverse=True)
    color_by_node = {}
    size_by_node = {}
    for rank, comp in enumerate(comps):
        if rank == 0:
            color, size = BLUE, 7
        elif rank < 5:
            color, size = (RED if rank == 1 else ORANGE), 6
        else:
            color, size = GRAY, 4
        for node in comp:
            color_by_node[node] = color
            size_by_node[node] = size
    nodes = list(graph.nodes())
    return nodes, [color_by_node[u] for u in nodes], [size_by_node[u] for u in nodes]


def draw_panel(ax: plt.Axes, graph: nx.Graph, title: str, seed: int, spread_top5: bool, metrics: dict[str, float]) -> None:
    pos = packed_component_layout(graph, seed, spread_top5)
    nodes, colors, sizes = node_style(graph)
    max_edges = 1600
    edge_graph = graph
    if graph.number_of_edges() > max_edges:
        edges = list(graph.edges())
        rng = np.random.default_rng(graph.number_of_nodes() + graph.number_of_edges())
        keep = rng.choice(len(edges), size=max_edges, replace=False)
        edge_graph = nx.Graph()
        edge_graph.add_nodes_from(graph.nodes())
        edge_graph.add_edges_from(edges[int(i)] for i in keep)
    nx.draw_networkx_edges(edge_graph, pos, ax=ax, edge_color=EDGE, alpha=0.10, width=0.22)
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


def make_figure(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    fig, axes = plt.subplots(1, 6, figsize=(15.8, 3.2))
    verified_rows: list[dict[str, object]] = []
    for case_idx, case in enumerate(cases):
        dataset = str(case["dataset"])
        graph = read_graph(dataset)
        n = graph.number_of_nodes()
        panels = [
            (str(case["method_a"]), int(case["step_a"]), float(case["hidden_score_a"])),
            (str(case["method_b"]), int(case["step_b"]), float(case["hidden_score_b"])),
        ]
        high_side = 0 if panels[0][2] >= panels[1][2] else 1
        panel_metrics: list[dict[str, float]] = []
        for side, (method, step, _hidden) in enumerate(panels):
            residual = residual_graph(graph, load_order(dataset, method), step)
            metrics = component_metrics(residual, n)
            panel_metrics.append(metrics)
            ax = axes[case_idx * 2 + side]
            draw_panel(
                ax,
                residual,
                f"{dataset} / {method}",
                seed=case_idx * 31 + side,
                spread_top5=(side == high_side),
                metrics=metrics,
            )
        verified = {
            "dataset": dataset,
            "method_a": str(case["method_a"]),
            "method_b": str(case["method_b"]),
            "step_a": int(case["step_a"]),
            "step_b": int(case["step_b"]),
            "q_a": int(case["step_a"]) / n,
            "q_b": int(case["step_b"]) / n,
            "gcc_a": panel_metrics[0]["gcc"],
            "gcc_b": panel_metrics[1]["gcc"],
            "gcc_abs_diff": abs(panel_metrics[0]["gcc"] - panel_metrics[1]["gcc"]),
            "top5_a": panel_metrics[0]["top5"],
            "top5_b": panel_metrics[1]["top5"],
            "components_a": panel_metrics[0]["components"],
            "components_b": panel_metrics[1]["components"],
            "effective_components_a": panel_metrics[0]["effective_components"],
            "effective_components_b": panel_metrics[1]["effective_components"],
            "hidden_score_a": panel_metrics[0]["hidden_selection_score"],
            "hidden_score_b": panel_metrics[1]["hidden_selection_score"],
        }
        verified["hidden_gap"] = abs(float(verified["hidden_score_a"]) - float(verified["hidden_score_b"]))
        verified_rows.append(verified)
        x_mid = (axes[case_idx * 2].get_position().x0 + axes[case_idx * 2 + 1].get_position().x1) / 2
        fig.text(x_mid, 0.965, f"Case {case_idx + 1}: same R, different residual split", ha="center", fontsize=10, weight="bold")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Largest component", markerfacecolor=BLUE, markersize=6),
        plt.Line2D([0], [0], marker="o", color="w", label="2nd-5th components", markerfacecolor=RED, markersize=6),
        plt.Line2D([0], [0], marker="o", color="w", label="Smaller fragments", markerfacecolor=GRAY, markersize=6),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(left=0.01, right=0.995, top=0.83, bottom=0.16, wspace=0.02)
    fig.savefig(OUT_FIG / "fig21_obs1_basic_baseline_same_r_horizontal.png")
    fig.savefig(OUT_FIG / "fig21_obs1_basic_baseline_same_r_horizontal.pdf")
    plt.close(fig)
    pd.DataFrame(verified_rows).to_csv(
        OUT_TABLE / "motivation_obs1_basic_baseline_horizontal_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return verified_rows


def main() -> None:
    setup()
    case_path = OUT_TABLE / "motivation_obs1_basic_baseline_horizontal_cases.csv"
    if not case_path.exists():
        raise FileNotFoundError(f"Missing fixed Observation-1 case table: {case_path}")
    cases = pd.read_csv(case_path, encoding="utf-8-sig").to_dict(orient="records")
    verified_cases = make_figure(cases)
    print("Selected cases:")
    for case in verified_cases:
        print(case)


if __name__ == "__main__":
    main()
