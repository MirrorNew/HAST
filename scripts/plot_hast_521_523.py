# -*- coding: utf-8 -*-
"""Generate section 5.2.1-5.2.3 figures for one completed HAST run."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "artifacts" / "source_tables" / "benchmark_12graph"

COL = {
    "q": "#0072B2",
    "s": "#009E73",
    "era": "#E69F00",
    "llm": "#CC79A7",
    "classic": "#9CA3AF",
    "strong": "#6B7280",
    "warn": "#D55E00",
    "grid": "#D1D5DB",
    "text": "#111827",
}


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.2,
            "legend.fontsize": 7.4,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )


def safe_method_filename(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(name)).strip("_")
    return text or "method"


def compressed_log_time(seconds: float, low_log: float = -2.0, fast_compress: float = 0.35) -> float:
    x = np.log10(max(float(seconds), 10**low_log))
    return x * fast_compress if x < 0 else x


def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def run_method_map(run_dir: Path) -> dict[str, str]:
    manifest = json.loads((run_dir / "final" / "final_code_manifest.json").read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for label, item in manifest.items():
        if not item:
            continue
        cid = str(item.get("candidate_id", ""))
        method_mean = pd.read_csv(run_dir / "full_validation" / "method_mean_metrics.csv", encoding="utf-8-sig")
        matched = method_mean[method_mean["candidate_id"].astype(str).eq(cid)]
        if not matched.empty:
            out[str(matched.iloc[0]["method"])] = label
    return out


def load_quality_table(run_dir: Path) -> pd.DataFrame:
    base = pd.read_csv(BENCHMARK / "method_mean_metrics.csv", encoding="utf-8-sig")
    base = base[~base["method"].isin(["HAST-Final-Q", "HAST-Final-S", "E26F"])].copy()
    run = pd.read_csv(run_dir / "full_validation" / "method_mean_metrics.csv", encoding="utf-8-sig")
    mapping = run_method_map(run_dir)
    run["method"] = run["method"].map(lambda m: mapping.get(str(m), str(m)))
    run["group"] = "HAST-current"
    run["coverage"] = run["datasets"].astype(int).astype(str) + "/12"
    base["coverage"] = base["datasets"].astype(int).astype(str) + "/12"
    combined = pd.concat([base, run], ignore_index=True, sort=False)
    for col in ["mean_R", "mean_auc_cNBI", "mean_time_s", "datasets"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined


def draw_521_quality_runtime(run_dir: Path, out_dir: Path) -> Path:
    data = load_quality_table(run_dir)
    data["x_plot"] = data["mean_time_s"].map(compressed_log_time)
    hast = {"HAST-Final-Q", "HAST-Final-S"}
    search = {"PUCT", "FunSearch-like", "Clade-AHD-like", "MCTS-AHD-like", "AlphaEvolve-like"}
    strong = {"NCDC", "NDC", "NDJC", "BPD/MinSum-fallback", "GND-py", "VE-py", "LGD-RA2-py", "LGD-RA2num-py", "LGD-CND-py"}
    classic = {"CoreHD", "HDA", "DC", "CI", "KCore", "CLUC"}
    colors = {
        "HAST-Final-Q": COL["q"],
        "HAST-Final-S": COL["s"],
        "PUCT": COL["era"],
        "FunSearch-like": "#CC79A7",
        "Clade-AHD-like": "#D55E00",
        "MCTS-AHD-like": "#009E73",
        "AlphaEvolve-like": "#7B8794",
        "NCDC": "#80B1D3",
        "BPD/MinSum-fallback": "#B15928",
    }
    fig, ax = plt.subplots(figsize=(8.7, 5.8))
    for _, row in data.iterrows():
        method = str(row["method"])
        color = colors.get(method, "#6B7280")
        if method in hast:
            ax.scatter(row["x_plot"], row["mean_auc_cNBI"], marker="*", s=310, color=color, edgecolor="#111827", linewidth=1.4, zorder=6)
        elif method in search:
            ax.scatter(row["x_plot"], row["mean_auc_cNBI"], s=92, color=color, edgecolor="white", linewidth=0.5, alpha=0.9, zorder=4)
        elif method in strong:
            ax.scatter(row["x_plot"], row["mean_auc_cNBI"], s=64, color=color, alpha=0.78, zorder=3)
        elif method in classic:
            ax.scatter(row["x_plot"], row["mean_auc_cNBI"], s=58, color=COL["classic"], alpha=0.85, zorder=2)
        else:
            ax.scatter(row["x_plot"], row["mean_auc_cNBI"], s=45, color="#6B7280", alpha=0.6)
    labels = {"PUCT": "ERA-like"}
    important = hast | search | {"NCDC", "BPD/MinSum-fallback", "CoreHD", "HDA"}
    offsets = {
        "HAST-Final-Q": (8, 9),
        "HAST-Final-S": (8, -14),
        "PUCT": (5, 7),
        "FunSearch-like": (5, 5),
        "Clade-AHD-like": (5, -11),
        "NCDC": (5, 6),
        "CoreHD": (5, -12),
        "HDA": (5, 5),
    }
    for _, row in data.iterrows():
        method = str(row["method"])
        if method not in important:
            continue
        dx, dy = offsets.get(method, (5, 3))
        suffix = f" ({row['coverage']})" if method in hast else ""
        ax.annotate(labels.get(method, method) + suffix, (row["x_plot"], row["mean_auc_cNBI"]), xytext=(dx, dy), textcoords="offset points", fontsize=8, fontweight="bold" if method in hast else "normal")
    ax.axvspan(compressed_log_time(1), compressed_log_time(0.01), color="#F3F4F6", alpha=0.75, zorder=0)
    tick_powers = [3, 2, 1, 0, -1, -2]
    ax.set_xticks([compressed_log_time(10**p) for p in tick_powers])
    ax.set_xticklabels([rf"$10^{{{p}}}$" for p in tick_powers])
    ax.set_xlim(compressed_log_time(10**3) + 0.12, compressed_log_time(10**-2) - 0.08)
    ax.set_ylim(-18, max(data["mean_auc_cNBI"].max(), 380) + 35)
    ax.set_xlabel("Mean runtime per graph (s, reversed log; sub-second compressed)")
    ax.set_ylabel("Mean auc-cNBI (higher is better)")
    ax.set_title("5.2.1 Quality-runtime position, current HAST run")
    ax.text(0.01, 0.02, "Current HAST points use completed datasets only (7/12); timeout datasets are excluded from mean quality.", transform=ax.transAxes, fontsize=7.5, color="#4B5563")
    save(fig, out_dir, "fig_5_2_1_quality_runtime_current_run")
    return out_dir / "fig_5_2_1_quality_runtime_current_run.png"


def draw_522_high_quality_panel(run_dir: Path, out_dir: Path) -> Path:
    data = load_quality_table(run_dir).set_index("method")
    selected = [m for m in ["FunSearch-like", "Clade-AHD-like", "PUCT", "NCDC", "BPD/MinSum-fallback", "HAST-Final-Q", "HAST-Final-S", "CoreHD"] if m in data.index]
    q_auc = float(data.loc["HAST-Final-Q", "mean_auc_cNBI"])
    q_time = float(data.loc["HAST-Final-Q", "mean_time_s"])
    rows = []
    for method in selected:
        rows.append(
            {
                "method": "ERA-like" if method == "PUCT" else method,
                "quality_vs_q": float(data.loc[method, "mean_auc_cNBI"]) / q_auc * 100.0,
                "speed_vs_q": q_time / float(data.loc[method, "mean_time_s"]),
                "coverage": data.loc[method, "coverage"],
                "raw": method,
            }
        )
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), gridspec_kw={"width_ratios": [1.2, 1.0]})
    y = np.arange(len(frame))
    colors = [COL["q"] if m == "HAST-Final-Q" else COL["s"] if m == "HAST-Final-S" else COL["era"] if m == "PUCT" else "#B8C2CC" for m in frame["raw"]]
    axes[0].barh(y, frame["quality_vs_q"], color=colors, height=0.62, edgecolor="white")
    axes[0].axvline(100, color="#777", ls="--", lw=1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(frame["method"])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(110, frame["quality_vs_q"].max() + 8))
    axes[0].set_xlabel("auc-cNBI vs current HAST-Final-Q (%)")
    axes[0].set_title("High-quality region")
    speed_vals = frame["speed_vs_q"].clip(upper=25)
    axes[1].barh(y, speed_vals, color=colors, height=0.62, edgecolor="white")
    axes[1].axvline(1, color="#777", ls="--", lw=1)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Speed relative to HAST-Final-Q")
    axes[1].set_title("Runtime advantage")
    for ax in axes:
        ax.grid(axis="x", alpha=0.18)
    fig.suptitle("5.2.2 Among high-quality candidates, current HAST trades quality for speed", y=1.03, fontsize=10.5, fontweight="bold")
    fig.text(0.02, -0.03, "Current HAST-Final-Q/S coverage: 7/12 completed under 90s guard; timeout datasets excluded from mean quality.", fontsize=7.5, color="#4B5563")
    save(fig, out_dir, "fig_5_2_2_high_quality_speed_current_run")
    return out_dir / "fig_5_2_2_high_quality_speed_current_run.png"


def read_point(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def baseline_point(dataset: str, method: str) -> pd.DataFrame:
    return read_point(BENCHMARK / dataset / "point_evaluations" / f"{safe_method_filename(method)}.csv")


def run_point(run_dir: Path, dataset: str, method: str) -> pd.DataFrame:
    matches = list((run_dir / "full_validation" / "point_evaluations" / dataset).glob(f"{method}.csv"))
    return read_point(matches[0]) if matches else pd.DataFrame()


def draw_curve_grid(run_dir: Path, out_dir: Path, metric: str, stem: str, title: str) -> Path:
    per_graph = pd.read_csv(run_dir / "full_validation" / "per_graph_metrics.csv", encoding="utf-8-sig")
    method_map = run_method_map(run_dir)
    label_to_method = {label: method for method, label in method_map.items()}
    datasets = ["CEnew", "Collaboration", "condmat", "crime", "email", "Grid", "GrQC", "hamster", "HepPh", "PH", "Powerlaw_500", "Yeast"]
    compare = ["PUCT", "FunSearch-like", "CoreHD", "HDA"]
    fig, axes = plt.subplots(3, 4, figsize=(12.4, 8.0), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, dataset in zip(axes, datasets):
        for method in compare:
            df = baseline_point(dataset, method)
            if df.empty or metric not in df:
                continue
            ax.plot(pd.to_numeric(df["removal_ratio"], errors="coerce"), pd.to_numeric(df[metric], errors="coerce"), lw=1.0, alpha=0.72, label="ERA-like" if method == "PUCT" else method)
        timeout_labels = []
        for label, method in label_to_method.items():
            row = per_graph[(per_graph["dataset"].eq(dataset)) & (per_graph["method"].eq(method))]
            valid = (not row.empty) and str(row.iloc[0].get("valid", "")).lower() == "true"
            if not valid:
                timeout_labels.append(label)
                continue
            df = run_point(run_dir, dataset, method)
            if df.empty or metric not in df:
                continue
            ax.plot(pd.to_numeric(df["removal_ratio"], errors="coerce"), pd.to_numeric(df[metric], errors="coerce"), lw=1.8, label=label, color=COL["q"] if label.endswith("Q") else COL["s"])
        ax.set_title(dataset, fontsize=9)
        if timeout_labels:
            ax.text(0.5, 0.5, "timeout under 90s:\n" + ", ".join(timeout_labels), transform=ax.transAxes, ha="center", va="center", fontsize=7, color=COL["warn"], bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": COL["warn"], "alpha": 0.85})
        ax.set_xlabel("Removal ratio", fontsize=7.5)
        ax.tick_params(labelsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        uniq.setdefault(l, h)
    fig.legend(uniq.values(), uniq.keys(), loc="lower center", ncol=6, frameon=False)
    fig.suptitle(title, y=0.995, fontsize=11, fontweight="bold")
    fig.text(0.02, 0.015, "Current HAST curves are shown only where the candidate completed under the 90s guard; timeout panels are explicitly marked.", fontsize=7.5, color="#4B5563")
    save(fig, out_dir, stem)
    return out_dir / f"{stem}.png"


def draw_523_curves(run_dir: Path, out_dir: Path) -> list[Path]:
    return [
        draw_curve_grid(run_dir, out_dir, "GCC", "fig_5_2_3_gcc_curves_current_run", "5.2.3 GCC curves across benchmark graphs"),
        draw_curve_grid(run_dir, out_dir, "cNBI", "fig_5_2_3_cnbi_curves_current_run", "5.2.3 cNBI curves across benchmark graphs"),
    ]


def write_summary(run_dir: Path, out_dir: Path, paths: list[Path]) -> None:
    mean = pd.read_csv(run_dir / "full_validation" / "method_mean_metrics.csv", encoding="utf-8-sig")
    per = pd.read_csv(run_dir / "full_validation" / "per_graph_metrics.csv", encoding="utf-8-sig")
    method_map = run_method_map(run_dir)
    mean["label"] = mean["method"].map(lambda m: method_map.get(str(m), str(m)))
    rows = []
    for _, row in mean.iterrows():
        method = row["method"]
        sub = per[per["method"].eq(method)]
        rows.append(
            {
                "label": row["label"],
                "candidate_id": row["candidate_id"],
                "completed": int(sub["valid"].astype(str).str.lower().eq("true").sum()),
                "total": int(len(sub)),
                "mean_R_completed": float(row["mean_R"]),
                "mean_auc_cNBI_completed": float(row["mean_auc_cNBI"]),
                "mean_time_s_completed": float(row["mean_time_s"]),
                "timeouts": int(sub["error"].astype(str).str.contains("timeout", case=False, na=False).sum()),
            }
        )
    summary = {"figures": [str(p) for p in paths], "current_run": rows}
    (out_dir / "figures_5_2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    setup()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "figures_5_2"
    paths = [
        draw_521_quality_runtime(run_dir, out_dir),
        draw_522_high_quality_panel(run_dir, out_dir),
        *draw_523_curves(run_dir, out_dir),
    ]
    write_summary(run_dir, out_dir, paths)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
