# -*- coding: utf-8 -*-
"""Build 12-graph curves with algorithm-found methods and Python baselines.

This script follows the paper-facing evidence policy:
- DACTS is excluded from this package.
- E26F is included only as an algorithm-found/reference candidate, not as a
  traditional baseline.
- Methods implemented in related_work_papers/code/python_baselines are reported
  with evidence_tier=python_baseline or python_fallback, because several are
  explicitly fallback implementations rather than official reproductions.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
TABLE_OUT = ROOT / "paper_tables"
FIG_OUT = ROOT / "figures"
REPORT_OUT = ROOT / "reports"
RECORD_OUT = ROOT / "paper_tables" / "extended_12graph_records"
EDGE_OUT = RECORD_OUT / "edgelists"
SEQ_OUT = RECORD_OUT / "python_baseline_sequences"

TREE_ROOT = WORKSPACE / "research" / "tree_search_ablation_20260520"
HAST_ROOT = WORKSPACE / "research" / "hast_experiment_20260521"
EVAL12_SRC = TREE_ROOT / "src" / "evaluate_final_12graphs.py"
PY_BASELINE = WORKSPACE / "research" / "related_work_papers" / "code" / "python_baselines" / "network_dismantling_baselines.py"


ALGORITHM_FOUND = {
    "PUCT": "PUCT",
    "FunSearch-like": "FunSearch-like",
    "Clade-AHD-like": "Clade-AHD-like",
    "MCTS-AHD-like": "MCTS-AHD-like",
    "AlphaEvolve-like": "AlphaEvolve-like",
    "E26F": "E26F",
}

PY_BASELINE_METHODS = [
    ("ndc", "NDC", "strong_python", "python_baseline"),
    ("ncdc", "NCDC", "strong_python", "python_baseline"),
    ("ndjc", "NDJC", "strong_python", "python_baseline"),
    ("bpd_minsum_python_baseline", "BPD/MinSum-fallback", "strong_python", "python_fallback"),
    ("gnd_python_baseline", "GND-py", "strong_python", "python_fallback"),
    ("ve_python_baseline", "VE-py", "strong_python", "python_fallback"),
    ("lgd_na_ra2_python_baseline", "LGD-RA2-py", "strong_python", "python_fallback"),
    ("lgd_na_ra2num_python_baseline", "LGD-RA2num-py", "strong_python", "python_fallback"),
    ("lgd_na_cnd_python_baseline", "LGD-CND-py", "strong_python", "python_fallback"),
]

CLASSIC_EXISTING = {
    "DC": "DC",
    "HDA": "HDA",
    "CoreHD": "CoreHD",
    "KCORE": "KCore",
    "CLUC": "CLUC",
    "CI": "CI",
}

PAPER_LABELS = {
    "FAST21-cap24": "HAST-Final-Q",
    "BT-n16-t8-u24": "HAST-Final-S",
}

METHOD_COLORS = {
    "PUCT": "#E69F00",
    "FunSearch-like": "#CC79A7",
    "Clade-AHD-like": "#D55E00",
    "MCTS-AHD-like": "#009E73",
    "AlphaEvolve-like": "#7B8794",
    "E26F": "#222222",
    "HAST-Final-Q": "#0072B2",
    "HAST-Final-S": "#56B4E9",
    "HDA": "#999999",
    "DC": "#8C8C8C",
    "CoreHD": "#6B7280",
    "CI": "#17BECF",
    "NDJC": "#2CA02C",
    "NDC": "#8DD3C7",
    "NCDC": "#80B1D3",
    "BPD/MinSum-fallback": "#B15928",
    "GND-py": "#A65628",
    "VE-py": "#984EA3",
    "LGD-RA2-py": "#4DAF4A",
    "LGD-RA2num-py": "#377EB8",
    "LGD-CND-py": "#FF7F00",
}

FIG10_FIG11_INCLUDED_METHODS = {
    "HAST-Final-Q",
    "HAST-Final-S",
    "DC",
    "HDA",
    "CoreHD",
    "KCore",
    "CLUC",
    "CI",
    "NDC",
    "NCDC",
    "NDJC",
    "BPD/MinSum-fallback",
    "GND-py",
    "VE-py",
    "LGD-RA2-py",
    "LGD-RA2num-py",
    "LGD-CND-py",
}
HIGHLIGHT_METHODS = {"HAST-Final-Q", "HAST-Final-S"}
SCATTER_HAST_METHODS = {"HAST-Final-Q", "HAST-Final-S"}
SCATTER_SEARCH_METHODS = {"PUCT", "FunSearch-like", "Clade-AHD-like", "MCTS-AHD-like", "AlphaEvolve-like"}
SCATTER_STRONG_BASELINES = {"NCDC", "NDC", "NDJC", "BPD/MinSum-fallback", "GND-py", "VE-py", "LGD-RA2-py", "LGD-RA2num-py", "LGD-CND-py"}
SCATTER_TRADITIONAL_METHODS = {"CoreHD", "HDA", "DC", "CI", "KCore", "CLUC"}
SCATTER_DISPLAY_LABELS = {"PUCT": "ERA-like"}

CACHE_LABEL_ALIASES = {
    "BPD/MinSum-fallback": ["BPD-MinSum-py"],
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


E12 = load_module(EVAL12_SRC, "extended_12graph_eval")


def setup() -> None:
    for path in [TABLE_OUT, FIG_OUT, REPORT_OUT, RECORD_OUT, EDGE_OUT, SEQ_OUT]:
        path.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
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


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def normalize_node(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        val = float(text)
        if val.is_integer():
            return int(val)
    except ValueError:
        pass
    return text


def summarize_metrics(df: pd.DataFrame, dataset: str, method: str, group: str, source: str, evidence_tier: str) -> dict[str, Any]:
    x = df["removal_ratio"].to_numpy(dtype=float)
    return {
        "dataset": dataset,
        "method": method,
        "group": group,
        "source": source,
        "evidence_tier": evidence_tier,
        "nodes": int(round(float(df["step"].iloc[-1]) / float(df["removal_ratio"].iloc[-1]))) if float(df["removal_ratio"].iloc[-1]) > 0 else np.nan,
        "steps": int(df["step"].max()),
        "R": float(df["GCC"].mean()),
        "auc_ACC": E12.EVAL.auc_mean(x, df["ACC"].to_numpy(dtype=float)),
        "auc_NCC": E12.EVAL.auc_mean(x, df["NCC"].to_numpy(dtype=float)),
        "auc_cNBI": E12.EVAL.auc_mean(x, df["cNBI"].to_numpy(dtype=float)),
        "final_GCC": float(df["GCC"].iloc[-1]),
        "final_cNBI": float(df["cNBI"].iloc[-1]),
        "time_s": float(df["total_time_s"].max()),
    }


def read_record(path: Path, dataset: str, method: str, group: str, source: str, evidence_tier: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path)
    df["dataset"] = dataset
    df["method"] = method
    df["group"] = group
    df["source"] = source
    df["evidence_tier"] = evidence_tier
    return df, summarize_metrics(df, dataset, method, group, source, evidence_tier)


def algorithm_found_records() -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    record_dir = TREE_ROOT / "final_12graph_eval" / "records"
    for dataset in E12.EVAL.DATASETS:
        for raw_method, label in ALGORITHM_FOUND.items():
            path = record_dir / f"{dataset}_{raw_method}_metrics.csv"
            if not path.exists():
                continue
            df, summary = read_record(path, dataset, label, "algorithm_found", "existing_unified_record", "recomputed_existing")
            rows.append(df)
            summaries.append(summary)

    fast = pd.read_csv(HAST_ROOT / "tables" / "hast_fact_fast_probe_full12_detail.csv")
    fast = fast[fast["method"].eq("FAST21-cap24")].copy()
    for dataset, sub in fast.groupby("dataset"):
        sub = sub.copy()
        sub["dataset"] = dataset
        sub["method"] = "HAST-Final-Q"
        sub["group"] = "algorithm_found"
        sub["source"] = "hast_fact_fast_probe_full12_detail"
        sub["evidence_tier"] = "recomputed_existing"
        rows.append(sub)
        summaries.append(summarize_metrics(sub, dataset, "HAST-Final-Q", "algorithm_found", "hast_fact_fast_probe_full12_detail", "recomputed_existing"))

    # Reuse the already computed HAST-Final-S curve if available; otherwise
    # recompute from the bounded-template generator.
    speed_cache = TABLE_OUT / "unified_curve_records.csv"
    if speed_cache.exists():
        cached = pd.read_csv(speed_cache, engine="python")
        cached = cached[cached["paper_label"].eq("HAST-Bounded speed")].copy()
        if not cached.empty:
            for dataset, sub in cached.groupby("dataset"):
                sub = sub.copy()
                sub["method"] = "HAST-Final-S"
                sub["group"] = "algorithm_found"
                sub["source"] = "bounded_template_curve_cache"
                sub["evidence_tier"] = "recomputed_existing"
                rows.append(sub)
                summaries.append(summarize_metrics(sub, dataset, "HAST-Final-S", "algorithm_found", "bounded_template_curve_cache", "recomputed_existing"))

    if not any(s["method"] == "HAST-Final-S" for s in summaries):
        bt_mod = load_module(HAST_ROOT / "src" / "hast_bounded_template_probe.py", "extended_bt_code")
        search_mod = E12.SEARCH
        fn = search_mod.compile_degree_order(bt_mod.make_code(16, 8, 24))
        for dataset in E12.EVAL.DATASETS:
            graph = E12.EVAL.read_graph(dataset)
            rate = E12.EVAL.DATASET_RATES[dataset]
            t0 = time.perf_counter()
            order = list(fn(graph.copy()))
            elapsed = time.perf_counter() - t0
            metrics = E12.EVAL.compute_metrics(graph, order, rate=rate, method_time=elapsed)
            metrics["dataset"] = dataset
            metrics["method"] = "HAST-Final-S"
            metrics["group"] = "algorithm_found"
            metrics["source"] = "bounded_template_recomputed"
            metrics["evidence_tier"] = "recomputed_existing"
            rows.append(metrics)
            summaries.append(summarize_metrics(metrics, dataset, "HAST-Final-S", "algorithm_found", "bounded_template_recomputed", "recomputed_existing"))
    return rows, summaries


def classic_existing_records() -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    record_dir = TREE_ROOT / "final_12graph_eval" / "records"
    for dataset in E12.EVAL.DATASETS:
        for raw_method, label in CLASSIC_EXISTING.items():
            path = record_dir / f"{dataset}_{raw_method}_metrics.csv"
            if not path.exists():
                continue
            df, summary = read_record(
                path,
                dataset,
                label,
                "static_or_classic",
                "existing_unified_record",
                "recomputed_existing",
            )
            rows.append(df)
            summaries.append(summary)
    return rows, summaries


def write_edge_list(dataset: str, graph: Any) -> Path:
    path = EDGE_OUT / f"{dataset}.edgelist"
    if path.exists():
        return path
    with path.open("w", encoding="utf-8", newline="") as handle:
        for u, v in graph.edges():
            handle.write(f"{u} {v}\n")
    return path


def should_skip_python_method(
    dataset: str,
    graph: Any,
    method: str,
    force_large_python: bool = False,
    allow_expensive_large: bool = False,
) -> str | None:
    n = graph.number_of_nodes()
    if force_large_python and (allow_expensive_large or method not in {"gnd_python_baseline", "ve_python_baseline"}):
        return None
    if n > 6000:
        return "skipped_large_graph_python_baseline"
    if method in {"gnd_python_baseline", "ve_python_baseline"} and n > 2500:
        return "skipped_large_graph_expensive_fallback"
    if method in {"ndc", "ncdc", "ndjc", "ci"} and n > 6000:
        return "skipped_large_graph_dynamic_python"
    return None


def run_python_baseline(
    dataset: str,
    graph: Any,
    method: str,
    label: str,
    timeout_s: int,
    force_large_python: bool = False,
    allow_expensive_large: bool = False,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    rate = E12.EVAL.DATASET_RATES[dataset]
    budget = max(1, int(round(graph.number_of_nodes() * rate)))
    skip_reason = should_skip_python_method(
        dataset,
        graph,
        method,
        force_large_python=force_large_python,
        allow_expensive_large=allow_expensive_large,
    )
    status = {
        "dataset": dataset,
        "method": label,
        "raw_method": method,
        "status": "pending",
        "note": "",
        "time_s": np.nan,
        "sequence_path": "",
    }

    metric_path = RECORD_OUT / f"{dataset}__{safe_name(label)}.csv"
    seq_path = SEQ_OUT / f"{dataset}__{safe_name(method)}.csv"
    if metric_path.exists():
        df = pd.read_csv(metric_path)
        status.update({"status": "cached", "time_s": float(df["total_time_s"].max()), "sequence_path": str(seq_path)})
        return df, status
    for alias in CACHE_LABEL_ALIASES.get(label, []):
        alias_path = RECORD_OUT / f"{dataset}__{safe_name(alias)}.csv"
        if alias_path.exists():
            df = pd.read_csv(alias_path)
            df["method"] = label
            df["source"] = f"python_baselines:{method}"
            df["evidence_tier"] = "python_fallback"
            status.update(
                {
                    "status": "cached_alias",
                    "note": f"loaded equivalent cached curve from label={alias}",
                    "time_s": float(df["total_time_s"].max()),
                    "sequence_path": str(seq_path),
                }
            )
            return df, status

    if skip_reason:
        status["status"] = skip_reason
        status["note"] = "large-graph Python implementation skipped after cache lookup; full curve uses existing unified records when available."
        return None, status

    edge_path = write_edge_list(dataset, graph)
    cmd = [
        sys.executable,
        str(PY_BASELINE),
        "--edge-list",
        str(edge_path),
        "--method",
        method,
        "--threshold",
        "0",
        "--max-steps",
        str(budget),
        "--output",
        str(seq_path),
    ]
    t0 = time.perf_counter()
    try:
        completed = subprocess.run(cmd, cwd=str(WORKSPACE), capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        status["status"] = "timeout"
        status["note"] = f"timeout after {timeout_s}s"
        return None, status
    elapsed = time.perf_counter() - t0
    if completed.returncode != 0 or not seq_path.exists():
        status["status"] = "failed"
        status["note"] = (completed.stderr or completed.stdout or "").strip()[:500]
        status["time_s"] = elapsed
        return None, status

    seq = pd.read_csv(seq_path)
    order = [node for node in (normalize_node(x) for x in seq["node"].tolist()) if node is not None]
    metrics = E12.EVAL.compute_metrics(graph, order, rate=rate, method_time=elapsed)
    metrics["dataset"] = dataset
    metrics["method"] = label
    metrics["group"] = "python_baseline"
    metrics["source"] = f"python_baselines:{method}"
    metrics["evidence_tier"] = "python_fallback" if method.endswith("_python_baseline") else "python_baseline"
    metrics.to_csv(metric_path, index=False, encoding="utf-8-sig")
    status.update({"status": "ok", "time_s": elapsed, "sequence_path": str(seq_path)})
    return metrics, status


def load_cached_python_baseline(dataset: str, method: str, label: str) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    metric_path = RECORD_OUT / f"{dataset}__{safe_name(label)}.csv"
    seq_path = SEQ_OUT / f"{dataset}__{safe_name(method)}.csv"
    if metric_path.exists():
        df = pd.read_csv(metric_path)
        status = {
            "dataset": dataset,
            "method": label,
            "raw_method": method,
            "status": "cached",
            "note": "loaded from cached curve while running a filtered supplement",
            "time_s": float(df["total_time_s"].max()),
            "sequence_path": str(seq_path),
        }
        return df, status
    for alias in CACHE_LABEL_ALIASES.get(label, []):
        alias_path = RECORD_OUT / f"{dataset}__{safe_name(alias)}.csv"
        if alias_path.exists():
            df = pd.read_csv(alias_path)
            df["method"] = label
            df["source"] = f"python_baselines:{method}"
            df["evidence_tier"] = "python_fallback"
            status = {
                "dataset": dataset,
                "method": label,
                "raw_method": method,
                "status": "cached_alias",
                "note": f"loaded equivalent cached curve from label={alias} while running a filtered supplement",
                "time_s": float(df["total_time_s"].max()),
                "sequence_path": str(seq_path),
            }
            return df, status
    return None, None


def python_baseline_records(
    timeout_s: int,
    selected_datasets: set[str] | None = None,
    selected_methods: set[str] | None = None,
    force_large_python: bool = False,
    allow_expensive_large: bool = False,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    datasets = list(E12.EVAL.DATASETS)
    methods = list(PY_BASELINE_METHODS)
    total = len(datasets) * len(methods)
    pbar = tqdm(total=total, desc="12graph python baselines", unit="run", dynamic_ncols=True)
    for dataset in datasets:
        graph = E12.EVAL.read_graph(dataset)
        pbar.write(f"[dataset] {dataset}: n={graph.number_of_nodes()} m={graph.number_of_edges()}")
        for raw_method, label, group, evidence in methods:
            pbar.set_postfix(dataset=dataset, method=label[:18])
            try:
                should_run = (
                    (selected_datasets is None or dataset in selected_datasets)
                    and (selected_methods is None or raw_method in selected_methods or label in selected_methods)
                )
                if should_run:
                    df, status = run_python_baseline(
                        dataset,
                        graph,
                        raw_method,
                        label,
                        timeout_s=timeout_s,
                        force_large_python=force_large_python,
                        allow_expensive_large=allow_expensive_large,
                    )
                else:
                    df, status = load_cached_python_baseline(dataset, raw_method, label)
                    if status is None:
                        status = {
                            "dataset": dataset,
                            "method": label,
                            "raw_method": raw_method,
                            "status": "not_selected_no_cache",
                            "note": "not requested in this filtered run and no cached curve exists",
                            "time_s": np.nan,
                            "sequence_path": "",
                        }
                status["group"] = group
                status["evidence_tier"] = evidence
                statuses.append(status)
                if df is not None:
                    df["group"] = group
                    df["evidence_tier"] = evidence
                    rows.append(df)
                    summaries.append(summarize_metrics(df, dataset, label, group, f"python_baselines:{raw_method}", evidence))
            finally:
                pbar.update(1)
    pbar.close()
    status_df = pd.DataFrame(statuses)
    status_df.to_csv(TABLE_OUT / "table_python_baseline_reproduction_status.csv", index=False, encoding="utf-8-sig")
    return rows, summaries, status_df


def save_all(rows: list[pd.DataFrame], summaries: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(TABLE_OUT / "table_12graph_extended_curve_records.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summaries)
    summary["rank_R"] = summary.groupby("dataset")["R"].rank(method="min", ascending=True)
    summary["rank_auc_cNBI"] = summary.groupby("dataset")["auc_cNBI"].rank(method="min", ascending=False)
    summary.to_csv(TABLE_OUT / "table_12graph_unified_metrics.csv", index=False, encoding="utf-8-sig")
    mean = (
        summary.groupby(["method", "group", "evidence_tier"], as_index=False)
        .agg(
            datasets=("dataset", "nunique"),
            mean_R=("R", "mean"),
            mean_auc_cNBI=("auc_cNBI", "mean"),
            mean_time_s=("time_s", "mean"),
            top1_auc=("rank_auc_cNBI", lambda s: int((s == 1).sum())),
            top3_auc=("rank_auc_cNBI", lambda s: int((s <= 3).sum())),
            mean_rank_auc=("rank_auc_cNBI", "mean"),
        )
        .sort_values(["mean_auc_cNBI", "mean_R"], ascending=[False, True])
    )
    mean.to_csv(TABLE_OUT / "table_12graph_method_mean_metrics.csv", index=False, encoding="utf-8-sig")
    return all_df, summary


def method_order(summary: pd.DataFrame, included_methods: set[str] | None = None) -> list[str]:
    if included_methods:
        summary = summary[summary["method"].isin(included_methods)].copy()
    mean = summary.groupby("method")["auc_cNBI"].mean().sort_values(ascending=False)
    preferred = [
        "HAST-Final-Q",
        "HAST-Final-S",
        "HDA",
        "CoreHD",
        "DC",
        "KCore",
        "CLUC",
        "CI",
        "NDJC",
        "NCDC",
        "NDC",
        "BPD/MinSum-fallback",
        "GND-py",
        "VE-py",
        "LGD-RA2-py",
        "LGD-RA2num-py",
        "LGD-CND-py",
    ]
    ordered = [m for m in preferred if m in mean.index]
    ordered.extend([m for m in mean.index if m not in ordered])
    return ordered


def plot_12grid(all_df: pd.DataFrame, summary: pd.DataFrame, metric: str, ylabel: str, stem: str, included_methods: set[str] | None = None) -> None:
    methods = method_order(summary, included_methods=included_methods)
    fig, axes = plt.subplots(3, 4, figsize=(16.4, 10.0), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, dataset in zip(axes, E12.EVAL.DATASETS):
        sub = all_df[all_df["dataset"].eq(dataset)]
        for method in methods:
            if included_methods and method not in included_methods:
                continue
            ms = sub[sub["method"].eq(method)].sort_values("removal_ratio")
            if ms.empty:
                continue
            lw = 2.4 if method in HIGHLIGHT_METHODS else 1.1
            alpha = 1.0 if method in HIGHLIGHT_METHODS else 0.72
            linestyle = "-" if method in HIGHLIGHT_METHODS else ("--" if method in ALGORITHM_FOUND.values() else ":")
            ax.plot(ms["removal_ratio"], ms[metric], label=method, color=METHOD_COLORS.get(method, "#555555"), lw=lw, alpha=alpha, ls=linestyle)
        ax.set_title(dataset)
        ax.set_xlabel("Removal ratio")
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(FIG_OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIG_OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(summary: pd.DataFrame) -> None:
    methods = method_order(summary)
    pivot = summary.pivot_table(index="dataset", columns="method", values="rank_auc_cNBI", aggfunc="mean").reindex(E12.EVAL.DATASETS)
    pivot = pivot[[m for m in methods if m in pivot.columns]]
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(pivot.columns)), 5.8))
    mat = pivot.to_numpy(dtype=float)
    im = ax.imshow(mat, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if math.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=6, color="white" if mat[i, j] <= 4 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Rank by auc-cNBI (1 is best)")
    ax.set_title("Per-dataset auc-cNBI ranks across 12 benchmark graphs")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "fig12_12graph_auc_rank_heatmap.png", bbox_inches="tight")
    fig.savefig(FIG_OUT / "fig12_12graph_auc_rank_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def compressed_log_time(seconds: float, low_log: float = -2.0, fast_compress: float = 0.35) -> float:
    x = math.log10(max(float(seconds), 10**low_log))
    if x < 0:
        return x * fast_compress
    return x


def plot_mean_scatter(summary: pd.DataFrame) -> None:
    mean = pd.read_csv(TABLE_OUT / "table_12graph_method_mean_metrics.csv")
    mean = mean[mean["method"] != "E26F"].copy()
    mean["x_plot"] = mean["mean_time_s"].map(compressed_log_time)

    fig, ax = plt.subplots(figsize=(8.4, 5.9))
    for _, row in mean.iterrows():
        method = row["method"]
        color = METHOD_COLORS.get(method, "#555555")
        if method in SCATTER_HAST_METHODS:
            ax.scatter(row["x_plot"], row["mean_auc_cNBI"], marker="*", s=280, color=color, edgecolor="#111827", linewidth=1.35, alpha=0.98, zorder=5)
        elif method in SCATTER_SEARCH_METHODS:
            ax.scatter(row["x_plot"], row["mean_auc_cNBI"], s=92, facecolor=color, edgecolor="none", alpha=0.88, zorder=3)
            circ = Ellipse(
                (row["x_plot"], row["mean_auc_cNBI"]),
                width=0.085,
                height=13.0,
                facecolor="none",
                edgecolor=color,
                linewidth=1.35,
                linestyle=(0, (3, 2)),
                zorder=4,
            )
            ax.add_patch(circ)
        elif method in SCATTER_STRONG_BASELINES:
            ax.scatter(row["x_plot"], row["mean_auc_cNBI"], s=60, color=color, edgecolor="none", alpha=0.82, zorder=2)
        elif method in SCATTER_TRADITIONAL_METHODS:
            ax.scatter(row["x_plot"], row["mean_auc_cNBI"], s=58, color="#9CA3AF", edgecolor="none", alpha=0.86, zorder=2)
        else:
            ax.scatter(row["x_plot"], row["mean_auc_cNBI"], s=50, color="#6B7280", edgecolor="none", alpha=0.70, zorder=2)

    label_offsets = {
        "HAST-Final-Q": (-22, 10),
        "HAST-Final-S": (-22, -14),
        "PUCT": (-4, 10),
        "FunSearch-like": (6, 5),
        "Clade-AHD-like": (6, -10),
        "NCDC": (-24, -14),
        "NDC": (6, 6),
        "BPD/MinSum-fallback": (6, 6),
        "LGD-RA2-py": (8, 10),
        "LGD-RA2num-py": (8, -2),
        "LGD-CND-py": (8, -14),
        "CoreHD": (-16, -14),
        "DC": (-18, 6),
        "CI": (5, 4),
        "KCore": (5, 3),
        "CLUC": (5, -10),
    }
    for _, row in mean.iterrows():
        method = row["method"]
        label = SCATTER_DISPLAY_LABELS.get(method, method)
        dx, dy = label_offsets.get(method, (5, 3))
        is_search_or_hast = method in (SCATTER_HAST_METHODS | SCATTER_SEARCH_METHODS | {"NCDC"})
        ax.annotate(
            label,
            (row["x_plot"], row["mean_auc_cNBI"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8 if is_search_or_hast else 6.8,
            fontweight="bold" if method in SCATTER_HAST_METHODS else "normal",
            color="#111827" if is_search_or_hast else "#374151",
        )

    tick_powers = [3, 2, 1, 0, -1, -2]
    tick_positions = [compressed_log_time(10**p) for p in tick_powers]
    tick_labels = [rf"$10^{{{p}}}$" for p in tick_powers]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(compressed_log_time(10**3) + 0.12, compressed_log_time(10**-2) - 0.08)
    ax.set_ylim(-18, max(mean["mean_auc_cNBI"]) + 38)
    ax.axvspan(compressed_log_time(10**0), compressed_log_time(10**-2), color="#F3F4F6", alpha=0.7, zorder=0)
    ax.grid(True, axis="y", color="#E5E7EB", lw=0.8)
    ax.grid(True, axis="x", color="#E5E7EB", lw=0.6, alpha=0.7)
    ax.set_xlabel("Mean runtime per graph (s, reversed log; sub-second region compressed)")
    ax.set_ylabel("Mean auc-cNBI (higher is better)")
    ax.set_title("Quality-runtime summary on 12 benchmark graphs")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "fig13_12graph_quality_runtime_all_methods.png", bbox_inches="tight")
    fig.savefig(FIG_OUT / "fig13_12graph_quality_runtime_all_methods.pdf", bbox_inches="tight")
    plt.close(fig)


def write_candidate_definition_table() -> None:
    rows = [
        {
            "paper_label": "HAST-Final-Q",
            "internal_id": "FAST21-cap24",
            "role": "quality-selected final HAST candidate",
            "selection_rule": "highest final quality among bounded/cost-aware HAST outputs used in the current evidence package",
            "is_new_framework": False,
        },
        {
            "paper_label": "HAST-Final-S",
            "internal_id": "BT-n16-t8-u24",
            "role": "speed-selected final HAST candidate",
            "selection_rule": "bounded-template candidate with cap_n=16, cap_2=8, update_cap=24; selected for quality-runtime tradeoff",
            "is_new_framework": False,
        },
        {
            "paper_label": "E26F",
            "internal_id": "E26F",
            "role": "algorithm-found reference from separate PUCT run",
            "selection_rule": "not a static/traditional baseline; reported only in the algorithm-found group",
            "is_new_framework": False,
        },
    ]
    pd.DataFrame(rows).to_csv(TABLE_OUT / "table_final_candidate_definitions.csv", index=False, encoding="utf-8-sig")


def write_python_baseline_alias_policy() -> None:
    rows = [
        {
            "paper_label": "BPD/MinSum-fallback",
            "raw_methods_in_script": "bpd_minsum_python_baseline; bpd_python_baseline; minsum_python_baseline",
            "implementation_status": "all three names map to choose_bpd_minsum_python_baseline in the local python_baselines script",
            "reporting_policy": "report as one fallback evidence item; do not claim independent BPD and MinSum reproductions",
            "split_in_main_experiment": False,
        }
    ]
    pd.DataFrame(rows).to_csv(TABLE_OUT / "table_python_baseline_alias_policy.csv", index=False, encoding="utf-8-sig")


def write_report(summary: pd.DataFrame, status: pd.DataFrame, timeout_s: int) -> None:
    mean = pd.read_csv(TABLE_OUT / "table_12graph_method_mean_metrics.csv")
    ok_statuses = ["ok", "cached", "cached_alias"]
    ok = status[status["status"].isin(ok_statuses)]
    failed = status[~status["status"].isin(ok_statuses)]
    lines = [
        "# 12 图统一曲线与 Python baseline 执行报告",
        "",
        "## 口径",
        "",
        "- DACTS 已排除。",
        "- E26F 只作为 algorithm-found/reference candidate，不作为传统 baseline。",
        "- `python_baselines` 中带 `_python_baseline` 的方法按 fallback 证据处理，不当作官方复现。",
        f"- `python_baselines` 外部复现实验采用单方法 {timeout_s}s 上限；已完成或缓存的结果保留，超时或用户决定不再继续等待的大图 fallback 只在状态表留档，不生成伪曲线。",
        "- `BPD/MinSum-fallback` 不拆成两个主实验项：本地脚本里的 `bpd_python_baseline` 与 `minsum_python_baseline` 都映射到同一个 fallback 函数，因此当前只能作为一个近似证据项。拆分需要独立官方/可信实现。",
        "- DC/HDA/CoreHD/KCore/CLUC/CI 的主表与主图优先使用已有 12 图统一评测记录；例如 condmat-DC 的算法计时为既有记录中的约 0.03s，而不是外部复现脚本的整段子进程耗时。",
        "- 现有 12 图集合按 benchmark graphs 表述，其中 `Powerlaw_500` 是 synthetic benchmark。",
        "",
        "## 平均指标 Top 15",
        "",
        mean.head(15).to_markdown(index=False),
        "",
        "## Python baseline 状态",
        "",
        f"- 成功或缓存：{len(ok)} rows",
        f"- timeout/skip/fail：{len(failed)} rows",
        "",
        failed.head(40).to_markdown(index=False) if not failed.empty else "无失败项。",
        "",
        "## 输出",
        "",
        "- `paper_tables/table_12graph_unified_metrics.csv`",
        "- `paper_tables/table_12graph_method_mean_metrics.csv`",
        "- `paper_tables/table_python_baseline_reproduction_status.csv`",
        "- `paper_tables/table_python_baseline_alias_policy.csv`",
        "- `figures/fig10_gcc_curves_12graphs.png`",
        "- `figures/fig11_cnbi_curves_12graphs.png`",
        "- `figures/fig12_12graph_auc_rank_heatmap.png`",
        "- `figures/fig13_12graph_quality_runtime_all_methods.png`",
    ]
    (REPORT_OUT / "12graph_curves_and_python_baselines_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names for Python-baseline reruns.")
    parser.add_argument("--methods", nargs="*", default=None, help="Optional raw or paper method labels for Python-baseline reruns.")
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument(
        "--force-large-python",
        action="store_true",
        help="Run large-graph Python baselines except the explicitly expensive GND/VE fallbacks.",
    )
    parser.add_argument(
        "--allow-expensive-large",
        action="store_true",
        help="Also attempt expensive GND/VE fallback baselines on large graphs.",
    )
    args = parser.parse_args()
    setup()
    write_candidate_definition_table()
    write_python_baseline_alias_policy()
    rows, summaries = algorithm_found_records()
    classic_rows, classic_summaries = classic_existing_records()
    rows.extend(classic_rows)
    summaries.extend(classic_summaries)
    py_rows, py_summaries, status = python_baseline_records(
        timeout_s=args.timeout_s,
        selected_datasets=set(args.datasets) if args.datasets else None,
        selected_methods=set(args.methods) if args.methods else None,
        force_large_python=args.force_large_python,
        allow_expensive_large=args.allow_expensive_large,
    )
    rows.extend(py_rows)
    summaries.extend(py_summaries)
    all_df, summary = save_all(rows, summaries)
    plot_12grid(all_df, summary, "GCC", "GCC (lower is better)", "fig10_gcc_curves_12graphs", included_methods=FIG10_FIG11_INCLUDED_METHODS)
    plot_12grid(all_df, summary, "cNBI", "cNBI (higher is better)", "fig11_cnbi_curves_12graphs", included_methods=FIG10_FIG11_INCLUDED_METHODS)
    plot_heatmap(summary)
    plot_mean_scatter(summary)
    write_report(summary, status, timeout_s=args.timeout_s)
    print(f"[done] wrote tables to {TABLE_OUT}")
    print(f"[done] wrote figures to {FIG_OUT}")
    print(f"[done] wrote report to {REPORT_OUT / '12graph_curves_and_python_baselines_cn.md'}")


if __name__ == "__main__":
    main()
