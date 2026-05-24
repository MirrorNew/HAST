# -*- coding: utf-8 -*-
"""Redraw paper-story figures with one-framework HAST naming.

The old exploratory figures used internal candidate names such as
FAST21-cap24, BT-n16-t8-u24, and HAST-FAC-T #24. For the paper story these are
stage outputs of one automatic framework, not separate methods. This script
regenerates figures with paper-facing labels.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "tables"
BOOST_TABLE = ROOT.parents[0] / "iclr_minimal_boost_20260522" / "tables"
HAST_ROOT = ROOT.parents[0] / "hast_experiment_20260521"
TREE_ROOT = ROOT.parents[0] / "tree_search_ablation_20260520"
FIG_DIR = ROOT / "figures"
OUT_TABLE_DIR = ROOT / "paper_tables"
CURVE_CACHE = OUT_TABLE_DIR / "unified_curve_records.csv"


COLORS = {
    "hast": "#0072B2",
    "hast_light": "#56B4E9",
    "baseline": "#6B7280",
    "warning": "#D55E00",
    "good": "#009E73",
    "puct": "#E69F00",
    "llm": "#CC79A7",
    "root": "#7B8794",
}


def setup() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
        }
    )


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_main_rows() -> pd.DataFrame:
    expanded = pd.read_csv(BOOST_TABLE / "expanded_existing_baseline_summary.csv")
    means = pd.read_csv(TABLE_DIR / "aaai_followup_method_mean_summary.csv")

    def mean_value(method: str, col: str) -> float:
        return float(means.loc[means["method"].eq(method), col].iloc[0])

    rows = [
        {
            "paper_label": "HDA root",
            "internal_name": "HDA",
            "role": "root heuristic",
            "auc_cNBI": mean_value("HDA", "mean_auc_cNBI"),
            "R": float(means.loc[means["method"].eq("HDA"), "mean_R"].iloc[0]),
            "time_s": mean_value("HDA", "mean_time_s"),
            "group": "Root",
            "main_table": True,
        },
        {
            "paper_label": "CoreHD",
            "internal_name": "CoreHD",
            "role": "traditional baseline",
            "auc_cNBI": float(expanded.loc[expanded["method"].eq("CoreHD"), "auc_cNBI"].iloc[0]),
            "R": float(expanded.loc[expanded["method"].eq("CoreHD"), "R"].iloc[0]),
            "time_s": float(expanded.loc[expanded["method"].eq("CoreHD"), "time_s"].iloc[0]),
            "group": "Baseline",
            "main_table": True,
        },
        {
            "paper_label": "PUCT search",
            "internal_name": "PUCT",
            "role": "search baseline",
            "auc_cNBI": mean_value("PUCT", "mean_auc_cNBI"),
            "R": float(means.loc[means["method"].eq("PUCT"), "mean_R"].iloc[0]),
            "time_s": mean_value("PUCT", "mean_time_s"),
            "group": "PUCT",
            "main_table": True,
        },
        {
            "paper_label": "FunSearch-like",
            "internal_name": "FunSearch-like",
            "role": "LLM-search baseline",
            "auc_cNBI": mean_value("FunSearch-like", "mean_auc_cNBI"),
            "R": float(means.loc[means["method"].eq("FunSearch-like"), "mean_R"].iloc[0]),
            "time_s": mean_value("FunSearch-like", "mean_time_s"),
            "group": "LLM baseline",
            "main_table": True,
        },
        {
            "paper_label": "Clade-AHD-like",
            "internal_name": "Clade-AHD-like",
            "role": "LLM-search baseline",
            "auc_cNBI": mean_value("Clade-AHD-like", "mean_auc_cNBI"),
            "R": float(means.loc[means["method"].eq("Clade-AHD-like"), "mean_R"].iloc[0]),
            "time_s": mean_value("Clade-AHD-like", "mean_time_s"),
            "group": "LLM baseline",
            "main_table": True,
        },
        {
            "paper_label": "HAST no cost control",
            "internal_name": "HAST-FAC C44",
            "role": "automatic ablation",
            "auc_cNBI": 380.4990332750692,
            "R": 0.3624424054861097,
            "time_s": 29.350590266667496,
            "group": "HAST ablation",
            "main_table": True,
        },
        {
            "paper_label": "HAST cost-aware",
            "internal_name": "HAST-FAC-T C24",
            "role": "automatic ablation",
            "auc_cNBI": mean_value("HAST-FAC-T online #24", "mean_auc_cNBI"),
            "R": float(means.loc[means["method"].eq("HAST-FAC-T online #24"), "mean_R"].iloc[0]),
            "time_s": mean_value("HAST-FAC-T online #24", "mean_time_s"),
            "group": "HAST ablation",
            "main_table": True,
        },
        {
            "paper_label": "HAST-Bounded quality",
            "internal_name": "FAST21-cap24",
            "role": "final automatic candidate",
            "auc_cNBI": mean_value("FAST21-cap24", "mean_auc_cNBI"),
            "R": float(means.loc[means["method"].eq("FAST21-cap24"), "mean_R"].iloc[0]),
            "time_s": mean_value("FAST21-cap24", "mean_time_s"),
            "group": "HAST final",
            "main_table": True,
        },
        {
            "paper_label": "HAST-Bounded speed",
            "internal_name": "BT-n16-t8-u24",
            "role": "final automatic candidate",
            "auc_cNBI": mean_value("BT-n16-t8-u24", "mean_auc_cNBI"),
            "R": float(means.loc[means["method"].eq("BT-n16-t8-u24"), "mean_R"].iloc[0]),
            "time_s": mean_value("BT-n16-t8-u24", "mean_time_s"),
            "group": "HAST final",
            "main_table": True,
        },
        {
            "paper_label": "PUCT reference E26F",
            "internal_name": "E26F",
            "role": "appendix reference from separate PUCT run",
            "auc_cNBI": mean_value("E26F", "mean_auc_cNBI"),
            "R": float(means.loc[means["method"].eq("E26F"), "mean_R"].iloc[0]),
            "time_s": mean_value("E26F", "mean_time_s"),
            "group": "Reference",
            "main_table": False,
        },
    ]
    out = pd.DataFrame(rows)
    puct_auc = float(out.loc[out["paper_label"].eq("PUCT search"), "auc_cNBI"].iloc[0])
    puct_time = float(out.loc[out["paper_label"].eq("PUCT search"), "time_s"].iloc[0])
    out["retention_vs_PUCT"] = out["auc_cNBI"] / puct_auc
    out["speedup_vs_PUCT"] = puct_time / out["time_s"]
    out.to_csv(OUT_TABLE_DIR / "table_main_results_unified_hast.csv", index=False, encoding="utf-8-sig")
    return out


def savefig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_framework() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    stages = [
        ("Root\nHDA/degree", "start from a simple\nremoval heuristic", COLORS["root"]),
        ("Automatic\nLLM mutation", "generate degree_order(G)\nunder constraints", COLORS["hast_light"]),
        ("Sandbox\nrollout", "execute on search graphs;\nrecord GCC, cNBI, time", COLORS["baseline"]),
        ("Relative\ncredit", "reward cNBI gain\nover the root", COLORS["hast"]),
        ("Cost-aware\nselection", "penalize slow or\nunstable candidates", COLORS["warning"]),
        ("Bounded\ncandidate", "select final candidate\nby frozen validation", COLORS["good"]),
    ]
    xs = np.linspace(0.08, 0.92, len(stages))
    y = 0.58
    box_w = 0.135
    box_h = 0.26
    for i, (title, body, color) in enumerate(stages):
        x = xs[i]
        box = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            linewidth=1.2,
            edgecolor=color,
            facecolor="#FFFFFF",
        )
        ax.add_patch(box)
        ax.text(x, y + 0.045, title, ha="center", va="center", fontsize=10, fontweight="bold", color=color)
        ax.text(x, y - 0.065, body, ha="center", va="center", fontsize=7.2, color="#333333")
        if i < len(stages) - 1:
            arr = FancyArrowPatch(
                (x + box_w / 2 + 0.008, y),
                (xs[i + 1] - box_w / 2 - 0.008, y),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.4,
                color="#555555",
            )
            ax.add_patch(arr)

    feedback = FancyArrowPatch(
        (xs[4], y - box_h / 2 - 0.035),
        (xs[1], y - box_h / 2 - 0.035),
        connectionstyle="arc3,rad=-0.22",
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        linestyle="--",
        color="#555555",
    )
    ax.add_patch(feedback)
    ax.text(0.47, 0.19, "search feedback: choose next parent and mutate again", ha="center", fontsize=8, color="#444444")
    ax.text(0.5, 0.93, "HAST: one automatic framework, progressively corrected credit and bounded generation", ha="center", fontsize=13, fontweight="bold")
    savefig(fig, "fig1_hast_framework_unified")


def draw_quality_runtime(rows: pd.DataFrame) -> None:
    data = rows[rows["main_table"]].copy()
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    marker = {
        "Root": "o",
        "Baseline": "s",
        "PUCT": "D",
        "LLM baseline": "^",
        "HAST ablation": "X",
        "HAST final": "*",
    }
    color = {
        "Root": COLORS["root"],
        "Baseline": COLORS["baseline"],
        "PUCT": COLORS["puct"],
        "LLM baseline": COLORS["llm"],
        "HAST ablation": COLORS["warning"],
        "HAST final": COLORS["good"],
    }
    for group, sub in data.groupby("group", sort=False):
        size = 210 if group == "HAST final" else 90
        ax.scatter(
            sub["time_s"],
            sub["auc_cNBI"],
            s=size,
            marker=marker[group],
            color=color[group],
            edgecolor="black",
            linewidth=0.7,
            alpha=0.95,
            label=group,
            zorder=3,
        )
    for _, r in data.iterrows():
        dx = 1.08 if r["time_s"] < 5 else 1.03
        dy = 3.5
        if "FunSearch" in r["paper_label"]:
            dy = -12
        if "Clade" in r["paper_label"]:
            dy = 5
        ax.text(r["time_s"] * dx, r["auc_cNBI"] + dy, r["paper_label"], fontsize=7.4)
    ax.set_xscale("log")
    ax.set_xlabel("Mean runtime per graph (s, log scale)")
    ax.set_ylabel("Mean auc-cNBI (higher is better)")
    ax.set_title("Quality-runtime frontier with unified HAST naming")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.axvline(1.0, color="#999999", linewidth=0.9, linestyle=":")
    ax.text(1.05, 205, "1 second", rotation=90, va="bottom", fontsize=8, color="#777777")
    savefig(fig, "fig2_quality_runtime_unified_hast")


def draw_progressive_path(rows: pd.DataFrame) -> None:
    order = [
        "HDA root",
        "HAST no cost control",
        "HAST cost-aware",
        "HAST-Bounded quality",
        "HAST-Bounded speed",
    ]
    data = rows[rows["paper_label"].isin(order)].copy()
    data["order"] = data["paper_label"].map({name: i for i, name in enumerate(order)})
    data = data.sort_values("order")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    ax.plot(data["order"], data["auc_cNBI"], color=COLORS["hast"], marker="o", linewidth=2.2, label="auc-cNBI")
    ax2 = ax.twinx()
    ax2.plot(data["order"], data["time_s"], color=COLORS["warning"], marker="s", linewidth=2.0, label="runtime")
    ax.set_xticks(data["order"])
    ax.set_xticklabels(["Root", "Relative\ncredit", "Cost-aware\ncredit", "Bounded\nquality", "Bounded\nspeed"], rotation=0)
    ax.set_ylabel("Mean auc-cNBI")
    ax2.set_ylabel("Mean runtime (s)")
    ax.set_title("One HAST framework: correction stages")
    ax.grid(True, axis="y", alpha=0.18)
    ax2.grid(False)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper left")

    ax = axes[1]
    path = rows[rows["paper_label"].isin(["PUCT search", "HAST-Bounded quality", "HAST-Bounded speed"])].copy()
    path = path.set_index("paper_label").loc[["PUCT search", "HAST-Bounded quality", "HAST-Bounded speed"]].reset_index()
    ax.barh(path["paper_label"], path["retention_vs_PUCT"] * 100, color=[COLORS["puct"], COLORS["good"], COLORS["good"]])
    ax.set_xlim(94, 101)
    ax.set_xlabel("auc-cNBI retained vs PUCT (%)")
    ax.set_title("Bounded candidates retain PUCT-level fragmentation")
    for i, r in path.iterrows():
        ax.text(r["retention_vs_PUCT"] * 100 + 0.08, i, f"{r['retention_vs_PUCT']*100:.1f}%", va="center", fontsize=8)
    savefig(fig, "fig3_hast_progressive_path")


def draw_appendix_map() -> None:
    items = [
        ("cNBI non-\nredundancy", "Table A1\nsame-GCC cases", "supports\nmetric signal"),
        ("Credit proxy\ncorrelation", "Table A2\nrho up to .90", "supports\nrelative credit"),
        ("Runtime drift\noutliers", "Table A3\nC21/HepPh", "supports\ncost control"),
        ("Frozen\nvalidation", "Table A4\nk=1/2/3", "supports\nselection protocol"),
        ("Search\nbudget", "Fig. A1\nfirst-hit curves", "supports\nautomation"),
        ("Per-dataset\nresults", "Fig. A2\n12 graphs", "supports\nrobustness"),
        ("Scaling", "Fig. A3\n10k nodes", "supports\npracticality"),
        ("Hard baseline\nsmoke test", "Table A5\nGND caveat", "supports\nbaseline policy"),
    ]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_axis_off()
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2)
    for idx, (title, artifact, claim) in enumerate(items):
        col = idx % 4
        row = 1 - idx // 4
        x = col + 0.08
        y = row + 0.12
        box = FancyBboxPatch(
            (x, y),
            0.84,
            0.72,
            boxstyle="round,pad=0.018,rounding_size=0.03",
            linewidth=1.0,
            edgecolor="#D1D5DB",
            facecolor="#FFFFFF",
        )
        ax.add_patch(box)
        ax.text(x + 0.42, y + 0.53, title, ha="center", va="center", fontweight="bold", fontsize=9, color=COLORS["hast"])
        ax.text(x + 0.42, y + 0.32, artifact, ha="center", va="center", fontsize=8, color="#333333")
        ax.text(x + 0.42, y + 0.12, claim, ha="center", va="center", fontsize=7.5, color="#666666")
    ax.text(2.0, 1.92, "Appendix evidence bank: broad experiments support the narrow main story", ha="center", fontsize=13, fontweight="bold")
    savefig(fig, "figA0_appendix_evidence_map")


def read_existing_record(dataset: str, method: str, source_dir: Path, paper_label: str) -> pd.DataFrame:
    path = source_dir / f"{dataset}_{method}_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["paper_label"] = paper_label
    return df


def candidate_file(run: str, idx: int) -> Path:
    files = sorted((HAST_ROOT / "runs" / run / "candidates").glob(f"candidate_{idx:04d}_*.py"))
    if not files:
        raise FileNotFoundError(f"missing candidate {run}/{idx}")
    return files[0]


def evaluate_candidate_curves(label: str, candidate_path: Path) -> pd.DataFrame:
    eval_mod = load_module(TREE_ROOT / "src" / "evaluate_final_12graphs.py", f"curve_eval_{label.replace(' ', '_')}")
    search_mod = eval_mod.SEARCH
    code = candidate_path.read_text(encoding="utf-8")
    fn = search_mod.compile_degree_order(code)
    rows = []
    for dataset in eval_mod.EVAL.DATASETS:
        graph = eval_mod.EVAL.read_graph(dataset)
        rate = eval_mod.EVAL.DATASET_RATES[dataset]
        import time

        t0 = time.perf_counter()
        order = list(fn(graph.copy()))
        elapsed = time.perf_counter() - t0
        metrics = eval_mod.EVAL.compute_metrics(graph, order, rate=rate, method_time=elapsed)
        metrics.insert(0, "paper_label", label)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def bt_code() -> str:
    mod = load_module(HAST_ROOT / "src" / "hast_bounded_template_probe.py", "curve_bt_code")
    return mod.make_code(16, 8, 24)


def evaluate_code_curves(label: str, code: str) -> pd.DataFrame:
    eval_mod = load_module(TREE_ROOT / "src" / "evaluate_final_12graphs.py", f"curve_eval_{label.replace(' ', '_')}")
    search_mod = eval_mod.SEARCH
    fn = search_mod.compile_degree_order(code)
    rows = []
    for dataset in eval_mod.EVAL.DATASETS:
        graph = eval_mod.EVAL.read_graph(dataset)
        rate = eval_mod.EVAL.DATASET_RATES[dataset]
        import time

        t0 = time.perf_counter()
        order = list(fn(graph.copy()))
        elapsed = time.perf_counter() - t0
        metrics = eval_mod.EVAL.compute_metrics(graph, order, rate=rate, method_time=elapsed)
        metrics.insert(0, "paper_label", label)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def build_curve_records() -> pd.DataFrame:
    if CURVE_CACHE.exists():
        return pd.read_csv(CURVE_CACHE)

    rows = []
    tree_records = TREE_ROOT / "final_12graph_eval" / "records"
    hast_records = HAST_ROOT / "final_12graph_eval" / "records"
    fast_detail = pd.read_csv(HAST_ROOT / "tables" / "hast_fact_fast_probe_full12_detail.csv")
    fast_detail = fast_detail[fast_detail["method"].eq("FAST21-cap24")].copy()
    fast_detail["paper_label"] = "HAST-Bounded quality"
    for dataset in sorted(fast_detail["dataset"].unique()):
        rows.append(fast_detail[fast_detail["dataset"].eq(dataset)])

    eval_mod = load_module(TREE_ROOT / "src" / "evaluate_final_12graphs.py", "curve_eval_datasets")
    for dataset in eval_mod.EVAL.DATASETS:
        rows.append(read_existing_record(dataset, "HDA", tree_records, "HDA root"))
        rows.append(read_existing_record(dataset, "PUCT", tree_records, "PUCT search"))
        # CoreHD from tree records is useful for appendix/runtime, but the main curve is kept narrow.

    rows.append(evaluate_candidate_curves("HAST no cost control", candidate_file("HAST-FAC", 44)))
    rows.append(evaluate_candidate_curves("HAST cost-aware", candidate_file("HAST-FACT-ONLINE60", 24)))
    rows.append(evaluate_code_curves("HAST-Bounded speed", bt_code()))

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(CURVE_CACHE, index=False, encoding="utf-8-sig")
    return out


def interpolate_mean_curve(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    grid = np.linspace(0.0, 0.30, 121)
    rows = []
    for (label, dataset), sub in df.groupby(["paper_label", "dataset"]):
        sub = sub.sort_values("removal_ratio")
        x = np.r_[0.0, sub["removal_ratio"].to_numpy(dtype=float)]
        if metric == "GCC":
            y0 = 1.0
        else:
            y0 = 0.0
        y = np.r_[y0, sub[metric].to_numpy(dtype=float)]
        keep = x <= 0.3000001
        x = x[keep]
        y = y[keep]
        if len(x) < 2:
            continue
        rows.append(pd.DataFrame({"paper_label": label, "dataset": dataset, "removal_ratio": grid, metric: np.interp(grid, x, y)}))
    all_interp = pd.concat(rows, ignore_index=True)
    return all_interp.groupby(["paper_label", "removal_ratio"], as_index=False)[metric].mean()


def draw_metric_curves(curves: pd.DataFrame) -> None:
    labels = [
        "HDA root",
        "PUCT search",
        "HAST no cost control",
        "HAST cost-aware",
        "HAST-Bounded quality",
        "HAST-Bounded speed",
    ]
    style = {
        "HDA root": ("#7B8794", "--"),
        "PUCT search": (COLORS["puct"], "-"),
        "HAST no cost control": (COLORS["warning"], "-"),
        "HAST cost-aware": (COLORS["hast_light"], "-"),
        "HAST-Bounded quality": (COLORS["good"], "-"),
        "HAST-Bounded speed": (COLORS["hast"], "-"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    for ax, metric, title, ylabel in [
        (axes[0], "GCC", "Mean GCC curve", "GCC (lower is better)"),
        (axes[1], "cNBI", "Mean cNBI curve", "cNBI (higher is better)"),
    ]:
        mean_curve = interpolate_mean_curve(curves, metric)
        for label in labels:
            sub = mean_curve[mean_curve["paper_label"].eq(label)]
            if sub.empty:
                continue
            color, ls = style[label]
            lw = 2.6 if label.startswith("HAST-Bounded") else 1.8
            ax.plot(sub["removal_ratio"], sub[metric], label=label, color=color, linestyle=ls, linewidth=lw)
        ax.set_title(title)
        ax.set_xlabel("Removal ratio")
        ax.set_ylabel(ylabel)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Deletion curves: same HAST stage names used throughout the paper", y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "fig4_gcc_cnbi_curves_unified_hast")


def draw_runtime_curve(rows: pd.DataFrame, curves: pd.DataFrame) -> None:
    # Per-dataset runtime from curve records. Node counts come from the same records.
    rt = (
        curves.groupby(["paper_label", "dataset"], as_index=False)
        .agg(nodes=("step", "max"), time_s=("total_time_s", "max"))
    )
    # Replace node count proxy with real node counts from per-dataset final step / removal_ratio.
    tmp = curves.groupby(["paper_label", "dataset"], as_index=False).tail(1)
    rt = tmp[["paper_label", "dataset", "step", "removal_ratio", "total_time_s"]].copy()
    rt["nodes"] = (rt["step"] / rt["removal_ratio"]).round().astype(int)
    labels = ["HDA root", "PUCT search", "HAST no cost control", "HAST cost-aware", "HAST-Bounded quality", "HAST-Bounded speed"]
    style = {
        "HDA root": ("#7B8794", "o"),
        "PUCT search": (COLORS["puct"], "D"),
        "HAST no cost control": (COLORS["warning"], "X"),
        "HAST cost-aware": (COLORS["hast_light"], "s"),
        "HAST-Bounded quality": (COLORS["good"], "*"),
        "HAST-Bounded speed": (COLORS["hast"], "*"),
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    for label in labels:
        sub = rt[rt["paper_label"].eq(label)].sort_values("nodes")
        if sub.empty:
            continue
        color, marker = style[label]
        ax.plot(sub["nodes"], sub["total_time_s"], label=label, color=color, marker=marker, linewidth=1.5, markersize=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Graph size |V|")
    ax.set_ylabel("Runtime per graph (s)")
    ax.set_title("Runtime curve across the 12 evaluation graphs")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    savefig(fig, "fig5_runtime_curve_unified_hast")


def write_markdown_table(rows: pd.DataFrame) -> None:
    main = rows[rows["main_table"]].copy()
    main["auc_cNBI"] = main["auc_cNBI"].map(lambda x: f"{x:.3f}")
    main["time_s"] = main["time_s"].map(lambda x: f"{x:.3f}")
    main["retention_vs_PUCT"] = main["retention_vs_PUCT"].map(lambda x: f"{100*x:.1f}%")
    main["speedup_vs_PUCT"] = main["speedup_vs_PUCT"].map(lambda x: f"{x:.2f}x")
    cols = ["paper_label", "role", "auc_cNBI", "time_s", "retention_vs_PUCT", "speedup_vs_PUCT", "internal_name"]
    (OUT_TABLE_DIR / "table_main_results_unified_hast.md").write_text(main[cols].to_markdown(index=False), encoding="utf-8")


def main() -> None:
    setup()
    rows = read_main_rows()
    write_markdown_table(rows)
    draw_framework()
    draw_quality_runtime(rows)
    draw_progressive_path(rows)
    draw_appendix_map()
    curves = build_curve_records()
    draw_metric_curves(curves)
    draw_runtime_curve(rows, curves)
    print(f"[done] wrote figures to {FIG_DIR}")
    print(f"[done] wrote tables to {OUT_TABLE_DIR}")


if __name__ == "__main__":
    main()
