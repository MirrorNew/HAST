#!/usr/bin/env python3
"""Generate advisor-facing figures that make HAST's advantage visually explicit."""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "03_main_tables" / "paper_tables").exists()), None)
if ARCHIVE_ROOT is not None:
    TABLE_DIR = ARCHIVE_ROOT / "03_main_tables" / "paper_tables"
    FIG_DIR = ARCHIVE_ROOT / "02_main_figures"
else:
    TABLE_DIR = ROOT / "paper_tables"
    FIG_DIR = ROOT / "figures"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    }
)

COL = {
    "hast_q": "#D55E00",
    "hast_s": "#E69F00",
    "puct": "#0072B2",
    "llm": "#56B4E9",
    "classic": "#9AA6B2",
    "strong": "#6B7280",
    "bad": "#C7CDD3",
    "green": "#009E73",
    "text": "#1F2937",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (TABLE_DIR / name).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png")
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    plt.close(fig)


def method_style(method: str) -> tuple[str, str, int, float]:
    if method == "HAST-Final-Q":
        return COL["hast_q"], "*", 260, 1.0
    if method == "HAST-Final-S":
        return COL["hast_s"], "*", 260, 1.0
    if method == "PUCT":
        return COL["puct"], "o", 130, 0.95
    if method in {"FunSearch-like", "Clade-AHD-like", "MCTS-AHD-like", "AlphaEvolve-like", "E26F"}:
        return COL["llm"], "o", 90, 0.72
    if method in {"NCDC", "NDC", "NDJC", "BPD/MinSum-fallback", "GND-py", "VE-py"}:
        return COL["strong"], "s", 65, 0.55
    return COL["classic"], "o", 55, 0.42


def advantage_map(rows: list[dict[str, str]]) -> None:
    puct = next(r for r in rows if r["method"] == "PUCT")
    puct_auc = f(puct, "mean_auc_cNBI")
    puct_time = f(puct, "mean_time_s")

    fig, ax = plt.subplots(figsize=(6.7, 4.35))
    ax.axvspan(95, 110, color="#FDEBD3", alpha=0.55, zorder=0)
    ax.axhspan(1, 220, color="#E8F5EF", alpha=0.42, zorder=0)
    ax.axvline(95, color="#8C8C8C", lw=1.0, ls="--", alpha=0.7)
    ax.axhline(1, color="#8C8C8C", lw=1.0, ls="--", alpha=0.7)
    ax.text(95.4, 150, "near-PUCT quality", color="#8B5E00", fontsize=8.5)
    ax.text(46, 1.16, "faster than PUCT", color="#006B4F", fontsize=8.5)

    plotted = []
    for r in rows:
        method = r["method"]
        if method in {"CLUC", "KCore"}:
            continue
        x = f(r, "mean_auc_cNBI") / puct_auc * 100
        y = puct_time / max(f(r, "mean_time_s"), 1e-9)
        color, marker, size, alpha = method_style(method)
        edge = "#2B2B2B" if method.startswith("HAST") else "white"
        ax.scatter(x, y, s=size, marker=marker, color=color, alpha=alpha, edgecolor=edge, linewidth=0.8, zorder=3)
        plotted.append((method, x, y))

    label_offsets = {
        "HAST-Final-Q": (1.5, 1.1),
        "HAST-Final-S": (1.5, 1.1),
        "PUCT": (1.3, 1.08),
        "FunSearch-like": (-22, 0.82),
        "Clade-AHD-like": (-24, 0.74),
        "E26F": (1.4, 1.13),
        "HDA": (1.5, 1.18),
        "CoreHD": (-14, 0.72),
        "NCDC": (1.4, 1.08),
        "BPD/MinSum-fallback": (-36, 1.12),
    }
    for method, x, y in plotted:
        if method not in label_offsets:
            continue
        dx, ymul = label_offsets[method]
        weight = "bold" if method.startswith("HAST") else "normal"
        color = COL["text"] if method.startswith("HAST") else "#4B5563"
        ax.text(x + dx, y * ymul, method, fontsize=8.3, weight=weight, color=color)

    ax.annotate(
        "HAST-Final-Q: 99.5% cNBI,\n9.7x faster than PUCT",
        xy=(99.5, 9.7),
        xytext=(72, 32),
        arrowprops=dict(arrowstyle="->", color=COL["hast_q"], lw=1.4),
        fontsize=9,
        color=COL["hast_q"],
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#F1C7A8", alpha=0.92),
    )
    ax.annotate(
        "HAST-Final-S: 99.0% cNBI,\n17.4x faster",
        xy=(99.0, 17.4),
        xytext=(103, 60),
        arrowprops=dict(arrowstyle="->", color=COL["hast_s"], lw=1.4),
        fontsize=9,
        color="#A85F00",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#F3D590", alpha=0.92),
    )

    ax.set_yscale("log")
    ax.set_xlim(0, 112)
    ax.set_ylim(0.08, 220)
    ax.set_xlabel("auc-cNBI retained vs. PUCT (%)")
    ax.set_ylabel("Runtime speedup vs. PUCT (log scale)")
    ax.set_title("HAST occupies the high-quality and faster-than-PUCT region")
    ax.grid(True, which="major", axis="both")
    ax.grid(True, which="minor", axis="y", alpha=0.08)
    save(fig, "fig16_hast_advantage_map")


def high_quality_speed_panel(rows: list[dict[str, str]]) -> None:
    hast_q = next(r for r in rows if r["method"] == "HAST-Final-Q")
    hast_q_auc = f(hast_q, "mean_auc_cNBI")
    hast_q_time = f(hast_q, "mean_time_s")
    selected = [
        "Clade-AHD-like",
        "FunSearch-like",
        "PUCT",
        "HAST-Final-Q",
        "HAST-Final-S",
        "AlphaEvolve-like",
        "HDA",
        "CoreHD",
    ]
    label = {
        "PUCT": "ERA-like",
    }
    by_method = {r["method"]: r for r in rows}
    items = []
    for m in selected:
        r = by_method[m]
        retention = f(r, "mean_auc_cNBI") / hast_q_auc * 100
        speedup = hast_q_time / f(r, "mean_time_s")
        items.append((m, retention, speedup))

    fig, axes = plt.subplots(1, 2, figsize=(6.95, 3.05), gridspec_kw={"width_ratios": [1.1, 1]})

    ax = axes[0]
    names = [x[0] for x in items]
    y = np.arange(len(items))
    colors = [
        COL["hast_q"] if n == "HAST-Final-Q" else COL["hast_s"] if n == "HAST-Final-S" else COL["puct"] if n == "PUCT" else "#B8C2CC"
        for n in names
    ]
    vals = [x[1] for x in items]
    ax.barh(y, vals, color=colors, height=0.62, edgecolor="white", linewidth=0.8)
    ax.axvline(100, color="#777", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([label.get(n, n) for n in names])
    ax.invert_yaxis()
    ax.set_xlim(50, 106)
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.set_xlabel("Quality vs. HAST-Final-Q (%)")
    ax.set_title("Quality normalized to HAST-Final-Q")
    for yi, v, n in zip(y, vals, names):
        fw = "bold" if n.startswith("HAST") else "normal"
        ax.text(v + 0.8, yi, f"{v:.1f}%", va="center", fontsize=8, weight=fw, color=COL["text"])

    ax = axes[1]
    hq_items = sorted(items, key=lambda z: z[2])
    names2 = [x[0] for x in hq_items]
    y2 = np.arange(len(hq_items))
    vals2 = [x[2] for x in hq_items]
    colors2 = [
        COL["hast_q"] if n == "HAST-Final-Q" else COL["hast_s"] if n == "HAST-Final-S" else COL["puct"] if n == "PUCT" else "#B8C2CC"
        for n in names2
    ]
    ax.barh(y2, vals2, color=colors2, height=0.62, edgecolor="white", linewidth=0.8)
    ax.axvline(1, color="#777", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(y2)
    ax.set_yticklabels([label.get(n, n) for n in names2])
    ax.set_xlabel("Speed vs. HAST-Final-Q (log)")
    ax.set_title("Runtime normalized to\nHAST-Final-Q")
    for yi, v, n in zip(y2, vals2, names2):
        fw = "bold" if n.startswith("HAST") else "normal"
        value_label = f"{v:.2f}x" if v < 0.1 else f"{v:.1f}x"
        ax.text(v * 1.08, yi, value_label, va="center", fontsize=8, weight=fw, color=COL["text"])
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig17_hast_quality_speed_panel")


def mechanism_compression(ablation_rows: list[dict[str, str]]) -> None:
    stage_rows = [
        ("Initial\nsearch", "Initial automatic search"),
        ("Relative\ncredit", "Relative credit only"),
        ("Cost-aware\ncredit", "Relative + time credit"),
        ("Bounded\nHAST-Q", "Full HAST, quality point"),
        ("Bounded\nHAST-S", "Full HAST, speed point"),
    ]
    by_name = {r["ablation"]: r for r in ablation_rows}
    labels = [x[0] for x in stage_rows]
    auc = [f(by_name[x[1]], "auc_cNBI") for x in stage_rows]
    times = [f(by_name[x[1]], "time_s") for x in stage_rows]

    fig, ax1 = plt.subplots(figsize=(6.75, 3.25))
    x = np.arange(len(labels))
    colors = ["#C7CDD3", "#56B4E9", "#009E73", COL["hast_q"], COL["hast_s"]]
    bars = ax1.bar(x, auc, color=colors, width=0.62, edgecolor="white", linewidth=0.8, zorder=3)
    ax1.set_ylabel("auc-cNBI ↑")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, max(auc) * 1.18)
    ax1.set_title("HAST compresses slow high-fragmentation search into fast bounded candidates")
    for bar, v in zip(bars, auc):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 8, f"{v:.0f}", ha="center", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, times, color="#333333", marker="o", lw=2, zorder=4)
    ax2.set_yscale("log")
    ax2.set_ylabel("Runtime (s, log) ↓")
    for xi, t in zip(x, times):
        ax2.text(xi + 0.05, t * 1.25, f"{t:.2f}s", fontsize=8, color="#333")

    ax1.annotate(
        "free search finds quality\nbut drifts to 29.35s",
        xy=(1, auc[1]),
        xytext=(0.25, 440),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#555"),
        fontsize=8.5,
        color="#333",
    )
    ax1.annotate(
        "bounded generation keeps\nnear-quality at 1.01s / 0.67s",
        xy=(3.5, (auc[3] + auc[4]) / 2),
        xytext=(2.55, 445),
        arrowprops=dict(arrowstyle="->", lw=1.3, color=COL["hast_q"]),
        fontsize=8.5,
        color=COL["hast_q"],
        weight="bold",
    )
    ax1.grid(axis="y", alpha=0.16)
    ax2.grid(False)
    fig.tight_layout()
    save(fig, "fig18_hast_mechanism_compression")


def win_profile(metrics_rows: list[dict[str, str]]) -> None:
    baselines = ["HDA", "CoreHD", "NCDC", "BPD/MinSum-fallback", "NDJC", "PUCT", "FunSearch-like", "Clade-AHD-like"]
    hast_methods = ["HAST-Final-Q", "HAST-Final-S"]
    by_dataset: dict[str, dict[str, float]] = {}
    for r in metrics_rows:
        by_dataset.setdefault(r["dataset"], {})[r["method"]] = float(r["auc_cNBI"])

    records = []
    for h in hast_methods:
        for b in baselines:
            wins = 0
            total = 0
            for vals in by_dataset.values():
                if h in vals and b in vals:
                    total += 1
                    wins += int(vals[h] >= vals[b])
            if total:
                records.append((h, b, wins, total, 100 * wins / total))

    fig, ax = plt.subplots(figsize=(6.75, 3.15))
    labels = baselines
    y = np.arange(len(labels))
    width = 0.36
    q = [next((r[4] for r in records if r[0] == "HAST-Final-Q" and r[1] == b), np.nan) for b in baselines]
    s = [next((r[4] for r in records if r[0] == "HAST-Final-S" and r[1] == b), np.nan) for b in baselines]
    ax.barh(y - width / 2, q, height=width, color=COL["hast_q"], label="HAST-Final-Q", edgecolor="white")
    ax.barh(y + width / 2, s, height=width, color=COL["hast_s"], label="HAST-Final-S", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Datasets where HAST has higher auc-cNBI (%)")
    ax.set_title("Win profile: HAST consistently beats classic and strong reproduced baselines")
    ax.axvline(50, color="#777", ls="--", lw=1)
    for yi, v in enumerate(q):
        if not math.isnan(v):
            ax.text(v + 1, yi - width / 2, f"{v:.0f}%", va="center", fontsize=8)
    for yi, v in enumerate(s):
        if not math.isnan(v):
            ax.text(v + 1, yi + width / 2, f"{v:.0f}%", va="center", fontsize=8)
    ax.legend(loc="lower right")
    save(fig, "fig19_hast_win_profile")


def load_search_records(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def framework_search_time_summary() -> None:
    base = ROOT.parent
    specs = [
        (
            "PUCT",
            "PUCT",
            "Prior LLM-search",
            base / "tree_search_ablation_20260520" / "runs" / "PUCT" / "search_records.csv",
        ),
        (
            "FunSearch-like",
            "FunSearch-like",
            "Prior LLM-search",
            base / "tree_search_ablation_20260520" / "runs" / "FunSearch-like" / "search_records.csv",
        ),
        (
            "Clade-AHD-like",
            "Clade-AHD-like",
            "Prior LLM-search",
            base / "tree_search_ablation_20260520" / "runs" / "Clade-AHD-like" / "search_records.csv",
        ),
        (
            "MCTS-AHD-like",
            "MCTS-AHD-like",
            "Prior LLM-search",
            base / "tree_search_ablation_20260520" / "runs" / "MCTS-AHD-like" / "search_records.csv",
        ),
        (
            "AlphaEvolve-like",
            "AlphaEvolve-like",
            "Prior LLM-search",
            base / "tree_search_ablation_20260520" / "runs" / "AlphaEvolve-like" / "search_records.csv",
        ),
        (
            "HAST-free",
            "HAST free search",
            "HAST stage",
            base / "hast_experiment_20260521" / "runs" / "HAST" / "search_records.csv",
        ),
        (
            "HAST-FAC",
            "HAST bounded search",
            "HAST stage",
            base / "hast_experiment_20260521" / "runs" / "HAST-FAC" / "search_records.csv",
        ),
        (
            "HAST-FACT-ONLINE60",
            "HAST online check",
            "HAST stage",
            base / "hast_experiment_20260521" / "runs" / "HAST-FACT-ONLINE60" / "search_records.csv",
        ),
    ]

    summary = []
    for method, paper_label, group, path in specs:
        rows = [r for r in load_search_records(path) if r.get("stage") == "search"]
        if not rows:
            continue
        eval_times = [x for x in (as_float(r.get("Time")) for r in rows) if x is not None]
        prompt_times = [x for x in (as_float(r.get("prompt_elapsed_s")) for r in rows) if x is not None]
        valid = sum(1 for r in rows if str(r.get("valid")).lower() == "true")
        total_eval = sum(eval_times)
        total_prompt = sum(prompt_times)
        n = len(rows)
        summary.append(
            {
                "method": method,
                "paper_label": paper_label,
                "group": group,
                "candidates": n,
                "valid_rate": valid / n if n else 0.0,
                "mean_eval_s": statistics.mean(eval_times) if eval_times else 0.0,
                "median_eval_s": statistics.median(eval_times) if eval_times else 0.0,
                "total_eval_s": total_eval,
                "mean_prompt_s": statistics.mean(prompt_times) if prompt_times else 0.0,
                "median_prompt_s": statistics.median(prompt_times) if prompt_times else 0.0,
                "total_prompt_s": total_prompt,
                "mean_logged_search_s_per_candidate": (total_prompt + total_eval) / n if n else 0.0,
                "total_logged_search_s": total_prompt + total_eval,
            }
        )

    out = TABLE_DIR / "table_framework_search_time_summary.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    order = [
        "PUCT",
        "FunSearch-like",
        "Clade-AHD-like",
        "MCTS-AHD-like",
        "AlphaEvolve-like",
        "HAST-free",
        "HAST-FAC",
        "HAST-FACT-ONLINE60",
    ]
    by_method = {r["method"]: r for r in summary}
    rows = [by_method[m] for m in order if m in by_method]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), gridspec_kw={"width_ratios": [1.05, 1]})

    ax = axes[0]
    labels = [r["paper_label"] for r in rows]
    y = np.arange(len(rows))
    vals = [r["mean_logged_search_s_per_candidate"] for r in rows]
    colors = [COL["hast_q"] if r["group"] == "HAST stage" else "#B8C2CC" for r in rows]
    ax.barh(y, vals, height=0.62, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean logged search time / candidate (s)")
    ax.set_title("HAST proposes candidates faster")
    for yi, v, r in zip(y, vals, rows):
        weight = "bold" if r["group"] == "HAST stage" else "normal"
        ax.text(v + 1.4, yi, f"{v:.1f}s", va="center", fontsize=8, weight=weight, color=COL["text"])
    ax.set_xlim(0, max(vals) * 1.25)

    ax = axes[1]
    vals_total = [r["total_logged_search_s"] / 3600 for r in rows]
    colors_total = [COL["hast_q"] if r["group"] == "HAST stage" else "#B8C2CC" for r in rows]
    ax.barh(y, vals_total, height=0.62, color=colors_total, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("Total logged search time (hours)")
    ax.set_title("Logged search cost is also lower")
    for yi, v, r in zip(y, vals_total, rows):
        weight = "bold" if r["group"] == "HAST stage" else "normal"
        ax.text(v + 0.15, yi, f"{v:.2f}h", va="center", fontsize=8, weight=weight, color=COL["text"])
    ax.set_xlim(0, max(vals_total) * 1.25)

    fig.text(
        0.52,
        -0.04,
        "Search time = LLM/proposal wall time (prompt_elapsed_s) + candidate validation time (Time), root excluded.",
        ha="center",
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig20_framework_search_time")


def main() -> None:
    rows = read_csv("table_12graph_method_mean_metrics.csv")
    ablation_rows = read_csv("table_module_ablation_three_mechanisms.csv")
    metrics_rows = read_csv("table_12graph_unified_metrics.csv")
    advantage_map(rows)
    high_quality_speed_panel(rows)
    mechanism_compression(ablation_rows)
    win_profile(metrics_rows)
    framework_search_time_summary()
    print("Generated fig16_hast_advantage_map, fig17_hast_quality_speed_panel, fig18_hast_mechanism_compression, fig19_hast_win_profile, fig20_framework_search_time")


if __name__ == "__main__":
    main()
