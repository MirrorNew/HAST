#!/usr/bin/env python3
"""Plot Motivation Observation figures for credit allocation and bounded generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"

COLOR_BLUE = "#2C7BB6"
COLOR_GREEN = "#1B9E77"
COLOR_ORANGE = "#F59E0B"
COLOR_RED = "#D7191C"
COLOR_PURPLE = "#6A3D9A"
COLOR_GRAY = "#6B7280"


def setup() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 160,
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
    fig.savefig(FIG_DIR / f"{stem}.png")
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    plt.close(fig)


def plot_relative_credit_effect() -> None:
    corr = pd.read_csv(TABLE_DIR / "hast_fac_credit_signal_correlations.csv", encoding="utf-8-sig")
    stages = pd.read_csv(TABLE_DIR / "motivation_obs2_obs3_stage_evidence.csv", encoding="utf-8-sig")

    wanted = [
        ("fac_cNBI20_adv", "relative cNBI@20 credit"),
        ("fac_auc_adv", "relative process credit"),
        ("fac_GCC20_adv", "relative GCC@20 credit"),
    ]
    corr_rows = []
    for feature, label in wanted:
        row = corr[(corr["feature"] == feature) & (corr["target"] == "auc_cNBI")].iloc[0]
        corr_rows.append({"label": label, "spearman": float(row["spearman"])})
    corr_df = pd.DataFrame(corr_rows)

    stage_keep = stages[stages["stage"].isin(["Initial automatic search", "Relative credit only"])].copy()
    stage_keep["label"] = stage_keep["stage"].map(
        {
            "Initial automatic search": "Initial\nsearch",
            "Relative credit only": "Relative\ncredit",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.45), gridspec_kw={"width_ratios": [1.08, 1.0]})

    ax = axes[0]
    colors = [COLOR_GREEN, COLOR_BLUE, COLOR_GRAY]
    bars = ax.barh(corr_df["label"], corr_df["spearman"], color=colors, alpha=0.88)
    ax.set_xlim(0, 0.82)
    ax.set_xlabel("Spearman correlation with final auc-cNBI")
    ax.set_title("Credit signal predicts fragmentation")
    for bar, value in zip(bars, corr_df["spearman"]):
        ax.text(value + 0.018, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)

    ax = axes[1]
    bars = ax.bar(stage_keep["label"], stage_keep["mean_auc_cNBI"], color=[COLOR_GRAY, COLOR_BLUE], alpha=0.88)
    ax.set_ylabel("mean auc-cNBI")
    ax.set_title("Relative credit improves discovery")
    ax.set_ylim(0, max(stage_keep["mean_auc_cNBI"]) * 1.28)
    for bar, (_, row) in zip(bars, stage_keep.iterrows()):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.text(x, y + 8, f"{y:.1f}", ha="center", va="bottom", fontsize=9, weight="bold")
        ax.text(
            x,
            y + 35,
            f"time={row['mean_time_s']:.3f}s",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLOR_RED if row["stage"] == "Relative credit only" else COLOR_GRAY,
        )
    initial = float(stage_keep[stage_keep["stage"] == "Initial automatic search"]["mean_auc_cNBI"].iloc[0])
    relative = float(stage_keep[stage_keep["stage"] == "Relative credit only"]["mean_auc_cNBI"].iloc[0])
    gain = 100.0 * (relative / initial - 1.0)
    ax.annotate(
        f"+{gain:.1f}% fragmentation",
        xy=(1, relative),
        xytext=(0.42, relative + 78),
        arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=1.2),
        fontsize=9,
        color=COLOR_BLUE,
        ha="center",
    )

    fig.suptitle("Observation 2: relative credit allocation strengthens fragmentation discovery", fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "fig22_relative_credit_allocation_effect")


def plot_bounded_generation_need() -> None:
    stage = pd.read_csv(TABLE_DIR / "motivation_obs2_obs3_stage_evidence.csv", encoding="utf-8-sig")
    bucket = pd.read_csv(TABLE_DIR / "motivation_obs2_fac_code_feature_by_time_bucket.csv", encoding="utf-8-sig")

    bucket_order = ["fast<=0.8", "ok<=1.2", "slow<=1.8", "too_slow"]
    bucket = bucket.set_index("time_bucket").loc[bucket_order].reset_index()
    bucket_labels = ["fast", "ok", "slow", "too slow"]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.7), gridspec_kw={"width_ratios": [1.08, 1.0]})

    ax = axes[0]
    x = np.arange(len(bucket))
    width = 0.25
    ax.bar(x - width, bucket["mean_neighbor_scan_count"], width, label="neighbor scans", color=COLOR_BLUE, alpha=0.85)
    ax.bar(x, bucket["rate_twohop_or_nested_neighbor_scan"], width, label="two-hop/nested rate", color=COLOR_ORANGE, alpha=0.88)
    ax.bar(x + width, bucket["rate_global_node_sweep"], width, label="global sweep rate", color=COLOR_RED, alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylabel("count or rate")
    ax.set_title("High-credit free search drifts into broad scans")
    ax.legend(frameon=False, loc="upper left")
    for idx, row in bucket.iterrows():
        ax.text(
            idx,
            max(row["mean_neighbor_scan_count"], row["rate_twohop_or_nested_neighbor_scan"], row["rate_global_node_sweep"]) + 0.18,
            f"{row['mean_proxy_time_s']:.3f}s",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLOR_GRAY,
        )
    ax.text(0.02, 0.94, "text = proxy time", transform=ax.transAxes, fontsize=8, color=COLOR_GRAY, va="top")

    ax = axes[1]
    keep = ["Relative credit only", "Relative + time credit", "Full HAST-Q", "Full HAST-S"]
    s = stage[stage["stage"].isin(keep)].copy()
    s["stage"] = pd.Categorical(s["stage"], categories=keep, ordered=True)
    s = s.sort_values("stage")
    sizes = 80 + 18 * np.arange(len(s))
    colors = [COLOR_RED, COLOR_ORANGE, COLOR_GREEN, COLOR_BLUE]
    ax.plot(s["mean_time_s"], s["mean_auc_cNBI"], color="#9CA3AF", lw=1.2, zorder=1)
    ax.scatter(s["mean_time_s"], s["mean_auc_cNBI"], s=sizes, c=colors, edgecolor="white", linewidth=1.0, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("mean time (s, log scale)")
    ax.set_ylabel("mean auc-cNBI")
    ax.set_title("Bounded generation compresses cost")
    for _, row in s.iterrows():
        label = {
            "Relative credit only": "relative\ncredit",
            "Relative + time credit": "+ time\ncredit",
            "Full HAST-Q": "HAST-Q",
            "Full HAST-S": "HAST-S",
        }[row["stage"]]
        ax.annotate(label, (row["mean_time_s"], row["mean_auc_cNBI"]), xytext=(5, 4), textcoords="offset points", fontsize=8)

    fig.suptitle("Observation 3: credit needs bounded, trust-guided generation", fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "fig23_bounded_generation_controls_scan_cost")


def main() -> None:
    setup()
    plot_relative_credit_effect()
    plot_bounded_generation_need()


if __name__ == "__main__":
    main()
