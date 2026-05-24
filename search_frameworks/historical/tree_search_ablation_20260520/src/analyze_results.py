# -*- coding: utf-8 -*-
"""Analyze and plot generic tree-search ablation results."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = EXPERIMENT_ROOT.parents[1]
RUNS_DIR = EXPERIMENT_ROOT / "runs"
TABLE_DIR = EXPERIMENT_ROOT / "tables"
FIG_DIR = EXPERIMENT_ROOT / "figures"
REPORT_DIR = EXPERIMENT_ROOT / "reports"
CANONICAL_DACTS = WORKSPACE / "research" / "dacts_e26f_reverse_search_20260519"

METHOD_ORDER = [
    "DACTS",
    "PUCT",
    "MCTS-AHD-like",
    "Clade-AHD-like",
    "FunSearch-like",
    "AlphaEvolve-like",
]
COLORS = {
    "DACTS": "#D62728",
    "PUCT": "#4C78A8",
    "MCTS-AHD-like": "#59A14F",
    "Clade-AHD-like": "#F28E2B",
    "FunSearch-like": "#B07AA1",
    "AlphaEvolve-like": "#9C755F",
}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_records() -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for method in METHOD_ORDER:
        path = RUNS_DIR / method / "search_records.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "method" not in df.columns:
            df["method"] = method
        df["method"] = method
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No search_records.csv found under {RUNS_DIR}")
    all_df = pd.concat(rows, ignore_index=True)
    all_df["valid"] = all_df["valid"].astype(bool)
    for col in ["R", "cNBI", "Time", "idx", "rank_score"]:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors="coerce")
    return all_df


def add_global_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    valid = out["valid"] & out["R"].notna() & out["cNBI"].notna() & out["Time"].notna()
    out["global_rank_R"] = -1.0
    out["global_rank_cNBI"] = -1.0
    out["global_rank_Time"] = -1.0
    out["global_rank_score"] = -1.0
    sub = out[valid].copy()
    if sub.empty:
        return out
    n = len(sub)
    denom = max(1, n - 1)
    for metric, higher, target in [
        ("R", False, "global_rank_R"),
        ("cNBI", True, "global_rank_cNBI"),
        ("Time", False, "global_rank_Time"),
    ]:
        ordered = sub.sort_values(metric, ascending=not higher)
        for pos, idx in enumerate(ordered.index):
            out.loc[idx, target] = (denom - pos) / denom
    out.loc[valid, "global_rank_score"] = (
        0.4 * out.loc[valid, "global_rank_R"]
        + 0.3 * out.loc[valid, "global_rank_cNBI"]
        + 0.3 * out.loc[valid, "global_rank_Time"]
    )
    return out


def load_e26f_reference() -> Dict[str, float]:
    path = CANONICAL_DACTS / "outputs" / "reference_comparison.csv"
    ref = pd.read_csv(path)
    row = ref[ref["name"].eq("e26f_reference")].iloc[0]
    return {"R": float(row["R"]), "cNBI": float(row["cNBI"]), "Time": float(row["Time"])}


def add_family_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ref = load_e26f_reference()
    out["near_e26f_strict"] = (
        out["valid"]
        & (out["R"] <= ref["R"] + 0.0006)
        & (out["cNBI"] >= ref["cNBI"] - 0.12)
        & (out["Time"] <= ref["Time"] * 2.0)
    )
    out["near_e26f_loose"] = (
        out["valid"]
        & (out["R"] <= ref["R"] + 0.0015)
        & (out["cNBI"] >= ref["cNBI"] - 0.45)
        & (out["Time"] <= ref["Time"] * 3.0)
    )
    return out


def first_hit(group: pd.DataFrame, flag: str) -> int:
    hit = group[group[flag]]
    if hit.empty:
        return -1
    return int(hit["idx"].min())


def make_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if "prompt_elapsed_s" not in df.columns:
        df = df.copy()
        df["prompt_elapsed_s"] = np.nan
    valid = df[df["valid"]].copy()
    best_rows = []
    for method in METHOD_ORDER:
        sub = valid[valid["method"].eq(method)]
        if sub.empty:
            continue
        row = sub.sort_values("global_rank_score", ascending=False).iloc[0].to_dict()
        best_rows.append(row)
    best = pd.DataFrame(best_rows)

    hit_rows = []
    for method in METHOD_ORDER:
        sub = df[df["method"].eq(method)].sort_values("idx")
        if sub.empty:
            continue
        hit_rows.append(
            {
                "method": method,
                "nodes": len(sub),
                "valid_nodes": int(sub["valid"].sum()),
                "invalid_nodes": int((~sub["valid"]).sum()),
                "first_strict_near_e26f": first_hit(sub, "near_e26f_strict"),
                "first_loose_near_e26f": first_hit(sub, "near_e26f_loose"),
                "strict_count": int(sub["near_e26f_strict"].sum()),
                "loose_count": int(sub["near_e26f_loose"].sum()),
                "strict_fraction": float(sub["near_e26f_strict"].mean()),
                "loose_fraction": float(sub["near_e26f_loose"].mean()),
            }
        )
    hitting = pd.DataFrame(hit_rows)

    invalid = (
        df.groupby("method")
        .agg(
            total_nodes=("node_id", "count"),
            valid_nodes=("valid", "sum"),
            mean_prompt_s=("prompt_elapsed_s", "mean"),
        )
        .reset_index()
    )
    invalid["invalid_nodes"] = invalid["total_nodes"] - invalid["valid_nodes"]
    invalid["invalid_rate"] = invalid["invalid_nodes"] / invalid["total_nodes"]

    family = df[["method", "idx", "node_id", "near_e26f_strict", "near_e26f_loose", "global_rank_score"]].copy()
    for name, table in [
        ("global_search_records.csv", df),
        ("best_by_method.csv", best),
        ("hitting_time.csv", hitting),
        ("e26f_like_family_density.csv", family),
        ("invalid_rate_and_runtime.csv", invalid),
    ]:
        table.to_csv(TABLE_DIR / name, index=False, encoding="utf-8-sig")
    return {"best": best, "hitting": hitting, "invalid": invalid, "family": family}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "figure.dpi": 260,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_search_curves(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for method in METHOD_ORDER:
        sub = df[df["method"].eq(method)].sort_values("idx")
        if sub.empty:
            continue
        y = sub["global_rank_score"].fillna(-1).cummax()
        ax.plot(sub["idx"], y, label=method, color=COLORS.get(method), linewidth=2.2 if method == "DACTS" else 1.5)
    ax.set_xlabel("Evaluated nodes")
    ax.set_ylabel("Best-so-far global rank score")
    ax.set_title("Search efficiency under a shared evaluator")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "best_so_far_global_score.png")
    fig.savefig(FIG_DIR / "best_so_far_global_score.pdf")
    plt.close(fig)


def plot_hitting_and_density(df: pd.DataFrame, hitting: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    ordered = hitting.set_index("method").reindex([m for m in METHOD_ORDER if m in hitting["method"].values]).reset_index()
    vals = ordered["first_strict_near_e26f"].replace(-1, np.nan)
    axes[0].bar(ordered["method"], vals, color=[COLORS.get(m, "#777") for m in ordered["method"]])
    axes[0].set_ylabel("First strict near-e26f hit")
    axes[0].set_title("Hitting time")
    axes[0].tick_params(axis="x", rotation=30)
    for i, v in enumerate(vals):
        label = "miss" if np.isnan(v) else str(int(v))
        axes[0].text(i, 0 if np.isnan(v) else v, label, ha="center", va="bottom", fontsize=8)

    for method in METHOD_ORDER:
        sub = df[df["method"].eq(method)].sort_values("idx")
        if sub.empty:
            continue
        axes[1].plot(
            sub["idx"],
            sub["near_e26f_strict"].cumsum(),
            label=method,
            color=COLORS.get(method),
            linewidth=2.2 if method == "DACTS" else 1.4,
        )
    axes[1].set_xlabel("Evaluated nodes")
    axes[1].set_ylabel("Cumulative strict near-e26f nodes")
    axes[1].set_title("Family density")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hitting_time_and_family_density.png")
    fig.savefig(FIG_DIR / "hitting_time_and_family_density.pdf")
    plt.close(fig)


def hierarchical_positions(nodes: List[str], edges: List[tuple[str, str]]) -> Dict[str, tuple[float, float]]:
    children: Dict[str, List[str]] = defaultdict(list)
    has_parent = set()
    for p, c in edges:
        children[p].append(c)
        has_parent.add(c)
    roots = [n for n in nodes if n not in has_parent]
    if not roots and nodes:
        roots = [nodes[0]]
    root = roots[0] if roots else ""
    for ch in children.values():
        ch.sort()
    x_pos: Dict[str, float] = {}
    y_pos: Dict[str, float] = {}
    leaf = 0

    def visit(node: str, depth: int) -> float:
        nonlocal leaf
        y_pos[node] = -depth
        if not children.get(node):
            x_pos[node] = float(leaf)
            leaf += 1
        else:
            xs = [visit(c, depth + 1) for c in children[node]]
            x_pos[node] = float(np.mean(xs))
        return x_pos[node]

    if root:
        visit(root, 0)
    for n in nodes:
        if n not in x_pos:
            x_pos[n] = float(leaf)
            y_pos[n] = 0.0
            leaf += 1
    if leaf > 1:
        for n in x_pos:
            x_pos[n] = x_pos[n] / (leaf - 1)
    return {n: (x_pos[n], y_pos[n]) for n in nodes}


def plot_tree_panel(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.4))
    axes = axes.ravel()
    for ax, method in zip(axes, METHOD_ORDER):
        sub = df[df["method"].eq(method)].sort_values("idx")
        if sub.empty:
            ax.axis("off")
            continue
        nodes = sub["node_id"].astype(str).tolist()
        edges = [(str(r["parent_id"]), str(r["node_id"])) for _, r in sub.iterrows() if str(r.get("parent_id")) not in {"", "nan", "None"}]
        pos = hierarchical_positions(nodes, edges)
        for p, c in edges:
            if p in pos and c in pos:
                ax.plot([pos[p][0], pos[c][0]], [pos[p][1], pos[c][1]], color="#BBBBBB", lw=0.35, alpha=0.55)
        colors = ["#D62728" if bool(row["near_e26f_strict"]) else COLORS.get(method, "#555") for _, row in sub.iterrows()]
        sizes = [18 if bool(row["valid"]) else 8 for _, row in sub.iterrows()]
        ax.scatter([pos[n][0] for n in nodes], [pos[n][1] for n in nodes], s=sizes, c=colors, edgecolors="white", linewidths=0.2)
        best = sub[sub["valid"]].sort_values("global_rank_score", ascending=False).head(1)
        if not best.empty:
            n = str(best.iloc[0]["node_id"])
            ax.scatter([pos[n][0]], [pos[n][1]], s=70, c="#111111", marker="*", zorder=5)
        ax.set_title(method)
        ax.axis("off")
    fig.suptitle("500-node search trees (red = strict near-e26f, star = best)", y=0.99)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "search_tree_comparison_panel.png")
    fig.savefig(FIG_DIR / "search_tree_comparison_panel.pdf")
    plt.close(fig)


def plot_invalid(invalid: pd.DataFrame) -> None:
    if invalid.empty:
        return
    ordered = invalid.set_index("method").reindex([m for m in METHOD_ORDER if m in invalid["method"].values]).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(ordered["method"], ordered["invalid_rate"], color=[COLORS.get(m, "#777") for m in ordered["method"]])
    ax.set_ylabel("Invalid / duplicate / timeout rate")
    ax.set_title("Search robustness")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "invalid_rate_by_method.png")
    fig.savefig(FIG_DIR / "invalid_rate_by_method.pdf")
    plt.close(fig)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_empty_"
    show = df.head(max_rows).copy()
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(tables: Dict[str, pd.DataFrame]) -> None:
    best = tables["best"].copy()
    hitting = tables["hitting"].copy()
    invalid = tables["invalid"].copy()
    best_cols = [c for c in ["method", "idx", "node_id", "R", "cNBI", "Time", "global_rank_score", "rank_score"] if c in best.columns]
    lines = [
        "# 通用 LLM 树搜索框架消融实验分析",
        "",
        "## 主要结论读取方式",
        "",
        "- `first_strict_near_e26f` 越小，说明越早命中接近 e26f 参考质量的算法。",
        "- `strict_count / nodes` 越高，说明不是单点命中，而是形成了更密集的高质量算法族。",
        "- invalid rate 越低，说明搜索框架对 LLM 代码生成更稳健。",
        "- 若 DACTS 的 family density 高于通用框架，可以支撑“领域 typed search 更适合机制族发现”的贡献点。",
        "",
        "## Best By Method",
        "",
        markdown_table(best[best_cols] if best_cols else best),
        "",
        "## Hitting Time / Family Density",
        "",
        markdown_table(hitting),
        "",
        "## Invalid Rate",
        "",
        markdown_table(invalid),
        "",
        "## 可写进论文的创新性解释",
        "",
        "1. 通用框架比较的是搜索控制策略，DACTS 比较的是搜索表示和反馈机制：typed dismantling program、HDA-root protocol、R/cNBI/Time diagnostic feedback、complexity guard。",
        "2. 如果 DACTS 更早或更密集地产生 near-e26f family，说明它不是简单组合 PUCT/FunSearch/AlphaEvolve，而是把搜索空间压到网络瓦解可解释机制上。",
        "3. 如果某个通用框架最终也搜到强算法，则把贡献转为：DACTS 用更可审计、更少无效、更机制化的方式复现并解释强算法族。",
        "",
        "## 建议的下一步改进",
        "",
        "- 实现 DACTS + Clade-AHD Thompson Sampling：保留 typed clade，但用 Bayesian clade allocation 替换当前温和采样。",
        "- 补三个内部消融：no-diagnostics、no-clade、generic-code-only。",
        "- 增加 3 seeds × 300 nodes 稳定性实验，用 mean/std 支撑 AAAI 审稿的统计可信度。",
    ]
    (REPORT_DIR / "ablation_analysis_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tree", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    setup_style()
    df = load_records()
    df = add_global_scores(df)
    df = add_family_flags(df)
    tables = make_tables(df)
    plot_search_curves(df)
    plot_hitting_and_density(df, tables["hitting"])
    plot_invalid(tables["invalid"])
    if not args.skip_tree:
        plot_tree_panel(df)
    write_report(tables)
    print(f"Analyzed {len(df)} records from {df['method'].nunique()} methods.")
    print(tables["hitting"].to_string(index=False))


if __name__ == "__main__":
    main()
