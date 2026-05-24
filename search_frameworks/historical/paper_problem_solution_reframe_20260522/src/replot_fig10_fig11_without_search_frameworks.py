# -*- coding: utf-8 -*-
"""Replot Fig. 10 and Fig. 11 with baselines plus the selected HAST methods.

The script uses the cached 12-graph curve table. It does not rerun any
algorithm, so the only intended change is the visual method filter.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUT = ROOT / "paper_tables"
FIG_OUT = ROOT / "figures"

CURVE_TABLE = TABLE_OUT / "table_12graph_extended_curve_records.csv"
SUMMARY_TABLE = TABLE_OUT / "table_12graph_unified_metrics.csv"

DATASET_ORDER = [
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
    "Powerlaw_500",
    "Yeast",
]

INCLUDED_METHODS = [
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
HIGHLIGHT_METHODS = {"HAST-Final-Q", "HAST-Final-S"}

METHOD_COLORS = {
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


def setup_style() -> None:
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


def method_order(summary: pd.DataFrame) -> list[str]:
    filtered = summary[summary["method"].isin(INCLUDED_METHODS)].copy()
    mean = filtered.groupby("method")["auc_cNBI"].mean().sort_values(ascending=False)
    ordered = [method for method in INCLUDED_METHODS if method in mean.index]
    ordered.extend([method for method in mean.index if method not in ordered])
    return ordered


def plot_12grid(all_df: pd.DataFrame, summary: pd.DataFrame, metric: str, ylabel: str, stem: str) -> None:
    methods = method_order(summary)
    fig, axes = plt.subplots(3, 4, figsize=(16.4, 10.0), sharex=False, sharey=False)
    axes = axes.ravel()

    legend_handles: dict[str, object] = {}
    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = all_df[all_df["dataset"].eq(dataset)]
        for method in methods:
            ms = sub[sub["method"].eq(method)].sort_values("removal_ratio")
            if ms.empty:
                continue
            is_highlight = method in HIGHLIGHT_METHODS
            (line,) = ax.plot(
                ms["removal_ratio"],
                ms[metric],
                label=method,
                color=METHOD_COLORS.get(method, "#555555"),
                lw=2.4 if is_highlight else 1.1,
                alpha=1.0 if is_highlight else 0.72,
                ls="-" if is_highlight else ":",
            )
            legend_handles.setdefault(method, line)
        ax.set_title(dataset)
        ax.set_xlabel("Removal ratio")
        ax.set_ylabel(ylabel)

    handles = [legend_handles[method] for method in methods if method in legend_handles]
    labels = [method for method in methods if method in legend_handles]
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(FIG_OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIG_OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    all_df = pd.read_csv(CURVE_TABLE, encoding="utf-8-sig")
    summary = pd.read_csv(SUMMARY_TABLE, encoding="utf-8-sig")

    all_df = all_df[all_df["method"].isin(INCLUDED_METHODS)].copy()
    summary = summary[summary["method"].isin(INCLUDED_METHODS)].copy()

    missing = sorted(set(INCLUDED_METHODS).difference(all_df["method"]))
    if missing:
        print(f"[warn] requested methods are missing from cached curves and will be omitted: {missing}")

    plot_12grid(all_df, summary, "GCC", "GCC (lower is better)", "fig10_gcc_curves_12graphs")
    plot_12grid(all_df, summary, "cNBI", "cNBI (higher is better)", "fig11_cnbi_curves_12graphs")


if __name__ == "__main__":
    main()
