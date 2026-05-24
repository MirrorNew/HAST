# -*- coding: utf-8 -*-
"""Regenerate local paper figures from record-derived experiment tables.

This script intentionally avoids the old plotting entrypoints whose relative
paths point to historical experiment folders. It reads only the consolidated
CSV files under HAST2026/03_main_tables and writes paper-facing figures under
HAST2026/02_main_figures.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAST_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = HAST_ROOT / "02_main_figures"
MAIN_ARTIFACTS = ROOT / "artifacts"
LOCAL_FIG_DIR = MAIN_ARTIFACTS / "figures"
PAPER_TABLE_DIR = MAIN_ARTIFACTS / "source_tables" / "paper_tables"
TABLE_DIR = MAIN_ARTIFACTS / "source_tables" / "tables"

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


def setup() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
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
    fig.savefig(FIG_DIR / f"{stem}.png", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def compressed_log_time(seconds: float, low_log: float = -2.0, fast_compress: float = 0.35) -> float:
    x = np.log10(max(float(seconds), 10**low_log))
    return x * fast_compress if x < 0 else x


def display(name: str) -> str:
    return METHOD_DISPLAY.get(str(name), str(name))


def draw_quality_runtime() -> None:
    data = pd.read_csv(PAPER_TABLE_DIR / "table_12graph_method_mean_metrics.csv", encoding="utf-8-sig")
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
    rows = pd.read_csv(PAPER_TABLE_DIR / "table_12graph_method_mean_metrics.csv", encoding="utf-8-sig")
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
    df = pd.read_csv(PAPER_TABLE_DIR / "table_framework_search_time_summary.csv", encoding="utf-8-sig")
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
    curves = pd.read_csv(
        PAPER_TABLE_DIR / "table_12graph_extended_curve_records.csv",
        encoding="utf-8-sig",
        usecols=["dataset", "method", "removal_ratio", "GCC", "cNBI"],
    )
    curves = curves[curves["method"].isin(use_methods)].copy()
    dataset_order = ["CEnew", "Collaboration", "condmat", "crime", "email", "Grid", "GrQC", "hamster", "HepPh", "PH", "Powerlaw_500", "Yeast"]
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


def draw_relative_credit() -> None:
    corr = pd.read_csv(TABLE_DIR / "hast_fac_credit_signal_correlations.csv", encoding="utf-8-sig")
    stages = pd.read_csv(TABLE_DIR / "motivation_obs2_obs3_stage_evidence.csv", encoding="utf-8-sig")
    wanted = [("fac_cNBI20_adv", "relative cNBI@20 credit"), ("fac_auc_adv", "relative process credit"), ("fac_GCC20_adv", "relative GCC@20 credit")]
    corr_rows = []
    for feature, label in wanted:
        row = corr[(corr["feature"].eq(feature)) & (corr["target"].eq("auc_cNBI"))].iloc[0]
        corr_rows.append({"label": label, "spearman": float(row["spearman"])})
    corr_df = pd.DataFrame(corr_rows)
    stage_keep = stages[stages["stage"].isin(["Initial automatic search", "Relative credit only"])].copy()
    stage_keep["label"] = stage_keep["stage"].map({"Initial automatic search": "Initial\nsearch", "Relative credit only": "Relative\ncredit"})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.45), gridspec_kw={"width_ratios": [1.08, 1.0]})
    bars = axes[0].barh(corr_df["label"], corr_df["spearman"], color=[COL["hast_s"], COL["hast_q"], COL["classic"]], alpha=0.88)
    axes[0].set_xlim(0, max(0.82, corr_df["spearman"].max() + 0.08))
    axes[0].set_xlabel("Spearman correlation with final auc-cNBI")
    axes[0].set_title("Credit signal predicts fragmentation")
    for bar, value in zip(bars, corr_df["spearman"]):
        axes[0].text(value + 0.018, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    bars = axes[1].bar(stage_keep["label"], stage_keep["mean_auc_cNBI"], color=[COL["classic"], COL["hast_q"]], alpha=0.88)
    axes[1].set_ylabel("mean auc-cNBI")
    axes[1].set_title("Relative credit improves discovery")
    axes[1].set_ylim(0, max(stage_keep["mean_auc_cNBI"]) * 1.28)
    for bar, (_, row) in zip(bars, stage_keep.iterrows()):
        x = bar.get_x() + bar.get_width() / 2
        axes[1].text(x, bar.get_height() + 8, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9, weight="bold")
        axes[1].text(x, bar.get_height() + 35, f"time={row['mean_time_s']:.3f}s", ha="center", va="bottom", fontsize=8, color=COL["warning"] if "Relative" in row["stage"] else COL["classic"])
    fig.suptitle("Observation 2: relative credit allocation strengthens fragmentation discovery", fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "fig22_relative_credit_allocation_effect")


def draw_bounded_generation() -> None:
    stage = pd.read_csv(TABLE_DIR / "motivation_obs2_obs3_stage_evidence.csv", encoding="utf-8-sig")
    bucket = pd.read_csv(TABLE_DIR / "motivation_obs2_fac_code_feature_by_time_bucket.csv", encoding="utf-8-sig")
    bucket_order = ["fast<=0.8", "ok<=1.2", "slow<=1.8", "too_slow"]
    bucket = bucket.set_index("time_bucket").loc[bucket_order].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.7), gridspec_kw={"width_ratios": [1.08, 1.0]})
    x = np.arange(len(bucket))
    width = 0.25
    axes[0].bar(x - width, bucket["mean_neighbor_scan_count"], width, label="neighbor scans", color=COL["hast_q"], alpha=0.85)
    axes[0].bar(x, bucket["rate_twohop_or_nested_neighbor_scan"], width, label="two-hop/nested rate", color=COL["era"], alpha=0.88)
    axes[0].bar(x + width, bucket["rate_global_node_sweep"], width, label="global sweep rate", color=COL["warning"], alpha=0.82)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["fast", "ok", "slow", "too slow"])
    axes[0].set_ylabel("count or rate")
    axes[0].set_title("High-credit free search drifts into broad scans")
    axes[0].legend(frameon=False, loc="upper left")
    for idx, row in bucket.iterrows():
        axes[0].text(idx, max(row["mean_neighbor_scan_count"], row["rate_twohop_or_nested_neighbor_scan"], row["rate_global_node_sweep"]) + 0.18, f"{row['mean_proxy_time_s']:.3f}s", ha="center", va="bottom", fontsize=8, color=COL["classic"])
    keep = ["Relative credit only", "Relative + time credit", "Full HAST-Q", "Full HAST-S"]
    s = stage[stage["stage"].isin(keep)].copy()
    s["stage"] = pd.Categorical(s["stage"], categories=keep, ordered=True)
    s = s.sort_values("stage")
    axes[1].plot(s["mean_time_s"], s["mean_auc_cNBI"], color="#9CA3AF", lw=1.2, zorder=1)
    axes[1].scatter(s["mean_time_s"], s["mean_auc_cNBI"], s=90 + 18 * np.arange(len(s)), c=[COL["warning"], COL["era"], COL["hast_q"], COL["hast_s"]], edgecolor="white", linewidth=1.0, zorder=2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("mean time (s, log scale)")
    axes[1].set_ylabel("mean auc-cNBI")
    axes[1].set_title("Bounded generation compresses cost")
    labels = {"Relative credit only": "relative\ncredit", "Relative + time credit": "+ time\ncredit", "Full HAST-Q": "HAST-Q", "Full HAST-S": "HAST-S"}
    for _, row in s.iterrows():
        axes[1].annotate(labels[row["stage"]], (row["mean_time_s"], row["mean_auc_cNBI"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    fig.suptitle("Observation 3: credit needs bounded, trust-guided generation", fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "fig23_bounded_generation_controls_scan_cost")


def draw_mechanism_compression() -> None:
    df = pd.read_csv(PAPER_TABLE_DIR / "table_module_ablation_three_mechanisms.csv", encoding="utf-8-sig")
    order = ["Initial automatic search", "Relative credit only", "Relative + time credit", "Full HAST, quality point", "Full HAST, speed point"]
    labels = ["Initial\nsearch", "Relative\ncredit", "Cost-aware\ncredit", "HAST-Q", "HAST-S"]
    df = df.set_index("ablation").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    x = np.arange(len(df))
    bars = ax.bar(x, df["auc_cNBI"], color=["#CBD5E1", "#E69F00", "#56B4E9", COL["hast_q"], COL["hast_s"]], edgecolor="white", linewidth=1.2, width=0.62)
    ax.set_ylabel("Mean auc-cNBI ↑")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(df["auc_cNBI"]) * 1.18)
    ax.set_title("HAST mechanism ablation: credit improves quality, bounds recover speed")
    for bar, auc in zip(bars, df["auc_cNBI"]):
        ax.text(bar.get_x() + bar.get_width() / 2, auc + 8, f"{auc:.0f}", ha="center", va="bottom", fontsize=10)
    ax2 = ax.twinx()
    ax2.plot(x, df["time_s"], color="#2F3437", marker="o", markersize=7, linewidth=2.6)
    ax2.set_yscale("log")
    ax2.set_ylabel("Mean runtime (s, log) ↓")
    for i, t in enumerate(df["time_s"]):
        ax2.text(i, t * 1.18, f"{t:.3f}s" if t < 1 else f"{t:.2f}s", ha="center", va="bottom", fontsize=9, color="#2F3437")
    fig.tight_layout()
    save(fig, "fig18_hast_mechanism_compression")


def draw_component_knockout() -> None:
    df = pd.read_csv(PAPER_TABLE_DIR / "table_final_candidate_component_ablation.csv", encoding="utf-8-sig")
    order = ["full", "no_degree", "no_frontier_weak", "no_twohop_boundary", "no_redundancy", "no_phase"]
    labels = {"full": "full", "no_degree": "no degree", "no_frontier_weak": "no frontier/\nweak-tie", "no_twohop_boundary": "no two-hop/\nboundary", "no_redundancy": "no redundancy", "no_phase": "no phase"}
    df["variant"] = pd.Categorical(df["variant"], categories=order, ordered=True)
    df = df.sort_values("variant")
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar(np.arange(len(df)), df["auc_cNBI"], color=[COL["hast_s"]] + [COL["classic"]] * (len(df) - 1), edgecolor="white", width=0.62)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels([labels[x] for x in df["variant"].astype(str)])
    ax.set_ylabel("Mean auc-cNBI")
    ax.set_title("Component knockout: degree backbone plus local fracture terms")
    for i, v in enumerate(df["auc_cNBI"]):
        ax.text(i, v + 6, f"{v:.0f}", ha="center", fontsize=8)
    save(fig, "fig14_component_knockout_ablation")


def draw_interpretability() -> None:
    features = pd.read_csv(PAPER_TABLE_DIR / "table_final_candidate_selected_node_features.csv", encoding="utf-8-sig")
    means = features.groupby("method")[["degree", "frontier", "weak_tie", "boundary", "redundancy", "clustering"]].mean()
    labels = ["degree", "frontier", "weak_tie", "boundary", "redundancy", "clustering"]
    ratio = means.loc["HAST-Final-S", labels].to_numpy(dtype=float) / means.loc["HDA", labels].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    x = np.arange(len(labels))
    ax.axhline(1.0, color="#777", linestyle="--", linewidth=1)
    ax.bar(x, ratio, color=[COL["hast_q"] if r >= 1 else COL["warning"] for r in ratio], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", "\n") for l in labels])
    ax.set_ylabel("HAST-Final-S / HDA feature mean")
    ax.set_title("Selected-node features: HAST keeps degree and increases fracture-oriented signals")
    for i, r in enumerate(ratio):
        ax.text(i, r + (0.025 if r >= 1 else -0.055), f"{r:.2f}x", ha="center", fontsize=8)
    save(fig, "fig13_final_candidate_interpretability")


def draw_scaling() -> None:
    full = pd.read_csv(TABLE_DIR / "scaling_full_eval_500_to_10k_unified.csv", encoding="utf-8-sig")
    ok = full[full["ok"].astype(bool)].copy()
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
    runtime = pd.read_csv(TABLE_DIR / "runtime_only_scaling_500_to_1000k_unified.csv", encoding="utf-8-sig")
    grouped = runtime.groupby(["method", "n"], as_index=False).agg(total=("ok", "size"), ok_count=("ok", lambda s: int(s.astype(bool).sum())), time_s=("time_s", "mean"))
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
    draw_quality_runtime()
    draw_high_quality_panel()
    draw_framework_search_time()
    draw_curves()
    draw_relative_credit()
    draw_bounded_generation()
    draw_mechanism_compression()
    draw_component_knockout()
    draw_interpretability()
    draw_scaling()
    print(f"record-based figures written to {LOCAL_FIG_DIR} and exported to {FIG_DIR}")


if __name__ == "__main__":
    main()
