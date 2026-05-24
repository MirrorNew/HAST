#!/usr/bin/env python3
"""Generate GCC/R-matched residual-network case visuals for Observation 1."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


WORKSPACE = Path(r"D:\LLM-CodexProject\llm-LLM_Agent_for_Disintegration_of_Networks")
ROOT = WORKSPACE / "research" / "paper_problem_solution_reframe_20260522"
FURTHERWORK = Path(r"D:\LLMProject\LLM_Agent_for_Disintegration_of_Networks\furtherwork")
DATA_DIR = FURTHERWORK / "Data"
GRAPH_DIR = DATA_DIR / "network"
RECORD_DIR = FURTHERWORK / "metric_e26f_result_codex" / "records"
OUT_FIG = ROOT / "figures"
OUT_TABLE = ROOT / "tables"
OUT_REPORT = ROOT / "reports"


CASES = [
    {
        "dataset": "Collaboration",
        "method_a": "MY",
        "method_b": "CI",
        "step_a": 607,
        "step_b": 1053,
        "why": "exactly near-matched GCC with large cNBI gap",
    },
    {
        "dataset": "Grid",
        "method_a": "MY",
        "method_b": "HDA",
        "step_a": 519,
        "step_b": 571,
        "why": "grid-like case where same GCC hides more dispersed fragments",
    },
    {
        "dataset": "CEnew",
        "method_a": "MY",
        "method_b": "CI",
        "step_a": 101,
        "step_b": 127,
        "why": "small clean case with identical GCC",
    },
    {
        "dataset": "crime",
        "method_a": "MY",
        "method_b": "CI",
        "step_a": 127,
        "step_b": 184,
        "why": "small real graph case with identical GCC",
    },
]

LABELS = {
    "MY": "metric_e26f",
    "HDA": "HDA",
    "DC": "DC",
    "CI": "CI",
    "EC": "EC",
    "KCORE": "KCore",
    "CC": "CC",
}

BLUE = "#2166AC"
RED = "#B2182B"
LIGHT_RED = "#EF8A62"
GRAY = "#C7C7C7"
EDGE = "#9A9A9A"


def setup() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def read_graph(dataset: str) -> nx.Graph:
    candidates = [DATA_DIR / f"{dataset}.txt", GRAPH_DIR / f"{dataset}.edgelist"]
    for path in candidates:
        if path.exists():
            graph = nx.read_edgelist(path, nodetype=int)
            graph = nx.Graph(graph)
            graph.remove_edges_from(nx.selfloop_edges(graph))
            return graph
    raise FileNotFoundError(f"Cannot find graph for {dataset}: {candidates}")


def load_order(dataset: str, method: str) -> list[int]:
    path = RECORD_DIR / f"{dataset}_{method}_steps.csv"
    df = pd.read_csv(path)
    return [int(x) for x in df.sort_values("step")["removed_node"].tolist()]


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
    cnbi = pairwise_disconnected * effective_components / (1.0 + top5 / original_n)
    return {
        "gcc": largest / original_n,
        "top5": top5 / original_n,
        "pairdisc": pairwise_disconnected,
        "hhi": hhi,
        "effective_components": effective_components,
        "cnbi": cnbi,
        "components": len(comps),
        "remaining_nodes": graph.number_of_nodes(),
    }


def packed_component_layout(graph: nx.Graph, seed: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    comps = sorted([sorted(c) for c in nx.connected_components(graph)], key=len, reverse=True)
    if not comps:
        return {}
    max_s = max(len(c) for c in comps)
    centers: list[tuple[float, float]] = [(-0.15, 0.0)]
    for rank in range(1, len(comps)):
        ring = 1 + int(math.sqrt(rank / 8))
        idx = rank - (ring - 1) * (ring - 1) * 8
        count = max(8, ring * 10)
        angle = 2 * math.pi * (idx % count) / count
        radius = 0.55 + 0.32 * ring
        centers.append((radius * math.cos(angle), radius * math.sin(angle)))

    pos: dict[int, np.ndarray] = {}
    golden = math.pi * (3 - math.sqrt(5))
    for rank, comp in enumerate(comps):
        size = len(comp)
        cx, cy = centers[rank]
        comp_radius = 0.06 + 0.34 * math.sqrt(size / max_s)
        if rank > 0:
            comp_radius *= 0.75
        for i, node in enumerate(comp):
            if size == 1:
                x, y = cx, cy
            else:
                angle = i * golden + rng.normal(0, 0.015)
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
            color, size = BLUE, 12
        elif rank < 5:
            color, size = (RED if rank == 1 else LIGHT_RED), 10
        else:
            color, size = GRAY, 7
        for node in comp:
            color_by_node[node] = color
            size_by_node[node] = size
    nodes = list(graph.nodes())
    return nodes, [color_by_node[u] for u in nodes], [size_by_node[u] for u in nodes]


def draw_panel(ax: plt.Axes, graph: nx.Graph, pos: dict[int, np.ndarray], title: str, metrics: dict[str, float]) -> None:
    nodes, colors, sizes = node_style(graph)
    sub_pos = {u: pos[u] for u in graph.nodes() if u in pos}
    edge_graph = graph
    max_edges = 2600
    if graph.number_of_edges() > max_edges:
        edges = list(graph.edges())
        rng = np.random.default_rng(graph.number_of_nodes() + graph.number_of_edges())
        keep = rng.choice(len(edges), size=max_edges, replace=False)
        edge_graph = nx.Graph()
        edge_graph.add_nodes_from(graph.nodes())
        edge_graph.add_edges_from(edges[int(i)] for i in keep)
    nx.draw_networkx_edges(edge_graph, sub_pos, ax=ax, edge_color=EDGE, alpha=0.12, width=0.25)
    nx.draw_networkx_nodes(
        graph,
        sub_pos,
        nodelist=nodes,
        node_color=colors,
        node_size=sizes,
        linewidths=0.0,
        alpha=0.9,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_axis_off()
    text = (
        f"GCC/R={metrics['gcc']:.3f}\n"
        f"Top5={metrics['top5']:.3f}\n"
        f"EffComp={metrics['effective_components']:.1f}\n"
        f"cNBI={metrics['cnbi']:.1f}"
    )
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#DDDDDD", alpha=0.9),
    )


def plot_case(case: dict[str, object]) -> dict[str, object]:
    dataset = str(case["dataset"])
    method_a = str(case["method_a"])
    method_b = str(case["method_b"])
    step_a = int(case["step_a"])
    step_b = int(case["step_b"])

    graph = read_graph(dataset)
    n = graph.number_of_nodes()
    residual_a = residual_graph(graph, load_order(dataset, method_a), step_a)
    residual_b = residual_graph(graph, load_order(dataset, method_b), step_b)
    metrics_a = component_metrics(residual_a, n)
    metrics_b = component_metrics(residual_b, n)
    pos_a = packed_component_layout(residual_a, seed=abs(hash((dataset, method_a, step_a))) % (2**32))
    pos_b = packed_component_layout(residual_b, seed=abs(hash((dataset, method_b, step_b))) % (2**32))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9))
    title_a = f"{dataset}: {LABELS.get(method_a, method_a)} at q={step_a / n:.3f}"
    title_b = f"{dataset}: {LABELS.get(method_b, method_b)} at q={step_b / n:.3f}"
    draw_panel(axes[0], residual_a, pos_a, title_a, metrics_a)
    draw_panel(axes[1], residual_b, pos_b, title_b, metrics_b)
    fig.suptitle(
        "GCC/R-matched residual networks: blue = GCC, red = 2nd-5th components, gray = small fragments",
        fontsize=10,
        y=1.03,
    )
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Largest component", markerfacecolor=BLUE, markersize=7),
        plt.Line2D([0], [0], marker="o", color="w", label="2nd-5th components", markerfacecolor=RED, markersize=7),
        plt.Line2D([0], [0], marker="o", color="w", label="Small fragments", markerfacecolor=GRAY, markersize=7),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()

    stem = f"fig21_obs1_same_r_case_{dataset}_{method_a}_vs_{method_b}"
    fig.savefig(OUT_FIG / f"{stem}.png")
    fig.savefig(OUT_FIG / f"{stem}.pdf")
    plt.close(fig)

    return {
        "dataset": dataset,
        "method_a": method_a,
        "method_b": method_b,
        "step_a": step_a,
        "step_b": step_b,
        "q_a": step_a / n,
        "q_b": step_b / n,
        "gcc_a": metrics_a["gcc"],
        "gcc_b": metrics_b["gcc"],
        "gcc_abs_diff": abs(metrics_a["gcc"] - metrics_b["gcc"]),
        "cnbi_a": metrics_a["cnbi"],
        "cnbi_b": metrics_b["cnbi"],
        "cnbi_gap_a_minus_b": metrics_a["cnbi"] - metrics_b["cnbi"],
        "top5_a": metrics_a["top5"],
        "top5_b": metrics_b["top5"],
        "effective_components_a": metrics_a["effective_components"],
        "effective_components_b": metrics_b["effective_components"],
        "components_a": metrics_a["components"],
        "components_b": metrics_b["components"],
        "figure_png": str(OUT_FIG / f"{stem}.png"),
        "figure_pdf": str(OUT_FIG / f"{stem}.pdf"),
        "why": str(case["why"]),
    }


def write_report(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Observation 1 Same-R/GCC Visual Cases",
        "",
        "这些图用于第 3.1 小节：同一数据集、不同策略，在 GCC/R 基本相同的 residual 状态下，残余碎裂结构仍然明显不同。",
        "",
        "| case | GCC/R A | GCC/R B | cNBI A | cNBI B | 结论 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        case = f"{row['dataset']}: {row['method_a']} vs {row['method_b']}"
        conclusion = (
            f"GCC/R 差 {row['gcc_abs_diff']:.4f}，但 cNBI 差 {row['cnbi_gap_a_minus_b']:.1f}；"
            f"Top5 mass {row['top5_a']:.3f} vs {row['top5_b']:.3f}"
        )
        lines.append(
            f"| {case} | {row['gcc_a']:.3f} | {row['gcc_b']:.3f} | {row['cnbi_a']:.1f} | {row['cnbi_b']:.1f} | {conclusion} |"
        )
    lines.extend(
        [
            "",
            "建议正文只放 3-4 张图中的 2-3 张，剩余放附录。主文 caption 不要写 cNBI 替代 GCC/R，而应写：GCC/R-matched residuals can have different fragmentation profiles.",
            "",
        ]
    )
    (OUT_REPORT / "obs1_same_r_visual_cases_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup()
    rows = []
    for case in CASES:
        print(f"[obs1-case] {case['dataset']} {case['method_a']} vs {case['method_b']}", flush=True)
        rows.append(plot_case(case))
    pd.DataFrame(rows).to_csv(OUT_TABLE / "motivation_obs1_same_r_visual_case_summary.csv", index=False, encoding="utf-8-sig")
    write_report(rows)
    print(f"[obs1-case] wrote {len(rows)} visual cases", flush=True)


if __name__ == "__main__":
    main()
