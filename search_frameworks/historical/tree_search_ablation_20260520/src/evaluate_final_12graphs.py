# -*- coding: utf-8 -*-
"""Evaluate best candidates from search ablations on 11 real graphs + 1 generated graph."""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = EXPERIMENT_ROOT.parents[1]
RUNS_DIR = EXPERIMENT_ROOT / "runs"
OUT_DIR = EXPERIMENT_ROOT / "final_12graph_eval"
FIG_DIR = OUT_DIR / "figures"
RECORD_DIR = OUT_DIR / "records"

CANONICAL_DACTS = WORKSPACE / "research" / "dacts_e26f_reverse_search_20260519"
OLD_EVAL_SRC = CANONICAL_DACTS / "src" / "evaluate_best_on_12_graphs.py"
ABLATION_SEARCH_SRC = EXPERIMENT_ROOT / "src" / "ablation_search.py"

SEARCH_METHODS = [
    "DACTS",
    "PUCT",
    "MCTS-AHD-like",
    "Clade-AHD-like",
    "FunSearch-like",
    "AlphaEvolve-like",
]
REFERENCE_METHODS = ["E26F", "CoreHD", "HDA"]
EXISTING_METHODS = ["DC", "KCORE", "CC", "EC", "CLUC", "CI"]

COLOR_MAP = {
    "DACTS": "#d62728",
    "PUCT": "#4C78A8",
    "MCTS-AHD-like": "#59A14F",
    "Clade-AHD-like": "#F28E2B",
    "FunSearch-like": "#B07AA1",
    "AlphaEvolve-like": "#9C755F",
    "E26F": "#111111",
    "CoreHD": "#8C564B",
    "HDA": "#E377C2",
    "DC": "#1F77B4",
    "KCORE": "#FF7F0E",
    "CC": "#9467BD",
    "EC": "#7F7F7F",
    "CLUC": "#BCBD22",
    "CI": "#17BECF",
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EVAL = load_module(OLD_EVAL_SRC, "canonical_dacts_eval")
SEARCH = load_module(ABLATION_SEARCH_SRC, "ablation_search_runtime")


def ensure_dirs() -> None:
    for path in [OUT_DIR, FIG_DIR, RECORD_DIR]:
        path.mkdir(parents=True, exist_ok=True)


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
            "grid.alpha": 0.15,
        }
    )


def safe_name(method: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", method)


def available_search_methods() -> List[str]:
    methods = []
    for method in SEARCH_METHODS:
        if (RUNS_DIR / method / "best_candidate.py").exists():
            methods.append(method)
    return methods


def order_from_candidate(method: str, graph: Any, rate: float) -> tuple[List[Any], float]:
    if method == "DACTS":
        cfg = EVAL.best_config_from_summary()
        t0 = time.perf_counter()
        order = EVAL.DACTS.degree_order_by_config(graph, cfg, budget_ratio=rate)
        return order, time.perf_counter() - t0
    code = (RUNS_DIR / method / "best_candidate.py").read_text(encoding="utf-8")
    fn = SEARCH.compile_degree_order(code)
    t0 = time.perf_counter()
    order = fn(graph.copy())
    elapsed = time.perf_counter() - t0
    if not isinstance(order, (list, tuple)):
        raise ValueError(f"{method} degree_order did not return a list")
    return list(order), elapsed


def evaluate_dataset(dataset: str, methods: List[str]) -> tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
    graph = EVAL.read_graph(dataset)
    rate = EVAL.DATASET_RATES[dataset]
    rows: List[pd.DataFrame] = []
    summaries: List[Dict[str, Any]] = []
    orders: Dict[str, tuple[List[Any], float, str]] = {}

    for method in methods:
        try:
            order, elapsed = order_from_candidate(method, graph, rate)
            orders[method] = (order, elapsed, "search_best")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {dataset}/{method}: {type(exc).__name__}: {exc}", flush=True)

    t0 = time.perf_counter()
    orders["E26F"] = (EVAL.DACTS.degree_order_by_config(graph, EVAL.DACTS.e26f_config(), budget_ratio=rate), time.perf_counter() - t0, "online")
    t0 = time.perf_counter()
    orders["CoreHD"] = (EVAL.corehd_order(graph, rate), time.perf_counter() - t0, "online")
    t0 = time.perf_counter()
    orders["HDA"] = (EVAL.hda_simple_order(graph, rate), time.perf_counter() - t0, "online")

    for method in EXISTING_METHODS:
        existing = EVAL.load_existing_order_and_time(dataset, method)
        if existing is not None:
            order, elapsed = existing
            orders[method] = (order, elapsed, "existing_record")

    for method, (order, elapsed, source) in orders.items():
        metrics = EVAL.compute_metrics(graph, order, rate=rate, method_time=elapsed)
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
                "auc_ACC": EVAL.auc_mean(x, metrics["ACC"].to_numpy(dtype=float)),
                "auc_NCC": EVAL.auc_mean(x, metrics["NCC"].to_numpy(dtype=float)),
                "auc_cNBI": EVAL.auc_mean(x, metrics["cNBI"].to_numpy(dtype=float)),
                "final_ACC": float(metrics["ACC"].iloc[-1]),
                "final_NCC": float(metrics["NCC"].iloc[-1]),
                "final_cNBI": float(metrics["cNBI"].iloc[-1]),
                "time_s": elapsed,
            }
        )
        metrics.to_csv(RECORD_DIR / f"{dataset}_{safe_name(method)}_metrics.csv", index=False, encoding="utf-8-sig")
    return rows, summaries


def plot_metric_grid(all_df: pd.DataFrame, methods: List[str], metric: str, ylabel: str, filename: str) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(15.8, 9.6), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, dataset in zip(axes, EVAL.DATASETS):
        sub = all_df[all_df["dataset"].eq(dataset)]
        for method in methods:
            ms = sub[sub["method"].eq(method)]
            if ms.empty:
                continue
            lw = 2.5 if method == "DACTS" else 1.2
            alpha = 1.0 if method in {"DACTS", "E26F", "CoreHD"} else 0.82
            ax.plot(
                ms["removal_ratio"],
                ms[metric],
                label=method,
                color=COLOR_MAP.get(method, "#333333"),
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


def plot_auc_bar(summary: pd.DataFrame, methods: List[str]) -> None:
    pivot = summary.pivot_table(index="dataset", columns="method", values="auc_cNBI", aggfunc="mean").reindex(EVAL.DATASETS)
    methods = [m for m in methods if m in pivot.columns]
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
            edgecolor="black" if method == "DACTS" else "none",
            linewidth=0.7 if method == "DACTS" else 0.0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.set_ylabel("AUC-cNBI (higher is better)")
    ax.set_title("Final 12-graph concentration-aware fragmentation")
    ax.legend(ncol=min(6, len(methods)), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "auc_cNBI_comparison_bar.png")
    plt.close(fig)


def plot_time(summary: pd.DataFrame, methods: List[str]) -> None:
    timed = summary[summary["method"].isin(methods)].dropna(subset=["time_s"]).copy()
    if timed.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    means = timed.groupby("method")["time_s"].mean().sort_values()
    ax.barh(means.index, means.values, color=[COLOR_MAP.get(m, "#333333") for m in means.index])
    ax.set_xlabel("Mean ordering time per graph (s)")
    ax.set_title("Online runtime comparison")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "time_mean_by_method.png")
    plt.close(fig)

    pivot = timed.pivot_table(index="dataset", columns="method", values="time_s", aggfunc="mean").reindex(EVAL.DATASETS)
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    for method in [m for m in methods if m in pivot.columns]:
        ax.plot(
            pivot.index,
            pivot[method],
            marker="o",
            label=method,
            color=COLOR_MAP.get(method, "#333333"),
            linewidth=2.2 if method == "DACTS" else 1.4,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Ordering time (s, log scale)")
    ax.set_title("Online runtime by dataset")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "time_by_dataset_method.png")
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    mean_metrics = summary.groupby("method")[["R", "auc_cNBI", "time_s"]].mean(numeric_only=True).reset_index()
    mean_metrics.to_csv(OUT_DIR / "mean_metrics_by_method.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Final 12-Graph Evaluation",
        "",
        "- Search methods are the best candidates found by the generic framework ablation.",
        "- DACTS is the canonical HDA-root typed search result.",
        "- E26F/CoreHD/HDA are recomputed online; DC/KCORE/CC/EC/CLUC/CI use existing trusted records when available.",
        "",
        "## Mean Metrics",
        "",
        mean_metrics.to_markdown(index=False),
    ]
    (OUT_DIR / "final_12graph_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-datasets", type=int, default=None)
    args = parser.parse_args()
    ensure_dirs()
    setup_style()
    search_methods = available_search_methods()
    methods = search_methods + REFERENCE_METHODS + EXISTING_METHODS
    selected = EVAL.DATASETS[: args.max_datasets] if args.max_datasets else EVAL.DATASETS
    all_rows: List[pd.DataFrame] = []
    summaries: List[Dict[str, Any]] = []
    for dataset in selected:
        print(f"[eval] {dataset}", flush=True)
        rows, rows_summary = evaluate_dataset(dataset, search_methods)
        all_rows.extend(rows)
        summaries.extend(rows_summary)
    all_df = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    all_df.to_csv(OUT_DIR / "all_trajectory_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_DIR / "final_12graph_summary.csv", index=False, encoding="utf-8-sig")
    plot_metric_grid(all_df, methods, "ACC", "ACC", "acc_curves_grid.png")
    plot_metric_grid(all_df, methods, "GCC", "R / GCC", "r_gcc_curves_grid.png")
    plot_metric_grid(all_df, methods, "NCC", "NCC", "ncc_curves_grid.png")
    plot_metric_grid(all_df, methods, "cNBI", "cNBI", "cnbi_curves_grid.png")
    plot_auc_bar(summary_df, methods)
    plot_time(summary_df, methods)
    write_report(summary_df)
    print(summary_df.groupby("method")[["R", "auc_cNBI", "time_s"]].mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
