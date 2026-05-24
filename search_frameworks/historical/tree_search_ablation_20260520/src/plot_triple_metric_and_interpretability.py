# -*- coding: utf-8 -*-
"""Generate triple-metric search-efficiency and interpretability figures.

This script is intentionally compact: it only reads existing ablation CSVs and
produces a few paper-facing tables/figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"

METHODS: Dict[str, Path] = {
    "DACTS(GPT-5.5-none)": ROOT / "runs" / "DACTS-rerun" / "outputs" / "search_records.csv",
    "DACTS(GPT-5.5-high)": ROOT / "runs" / "DACTS" / "search_records.csv",
    "PUCT": ROOT / "runs" / "PUCT" / "search_records.csv",
    "MCTS-AHD-like": ROOT / "runs" / "MCTS-AHD-like" / "search_records.csv",
    "Clade-AHD-like": ROOT / "runs" / "Clade-AHD-like" / "search_records.csv",
    "FunSearch-like": ROOT / "runs" / "FunSearch-like" / "search_records.csv",
    "AlphaEvolve-like": ROOT / "runs" / "AlphaEvolve-like" / "search_records.csv",
}

COLORS = {
    "DACTS(GPT-5.5-none)": "#D62728",
    "DACTS(GPT-5.5-high)": "#FF9896",
    "PUCT": "#4C78A8",
    "MCTS-AHD-like": "#59A14F",
    "Clade-AHD-like": "#F28E2B",
    "FunSearch-like": "#B07AA1",
    "AlphaEvolve-like": "#9C755F",
}


def setup() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
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
            "grid.alpha": 0.18,
        }
    )


def load_records() -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for method, path in METHODS.items():
        df = pd.read_csv(path)
        df["method"] = method
        df["valid"] = df["valid"].astype(str).str.lower().isin(["true", "1", "yes"])
        for col in ["idx", "R", "cNBI", "Time"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def add_global_triple_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["triple_score"] = np.nan
    valid = out["valid"] & out["R"].notna() & out["cNBI"].notna() & out["Time"].notna()
    sub = out[valid].copy()
    denom = max(1, len(sub) - 1)
    parts = {}
    for metric, higher, name in [
        ("R", False, "score_R"),
        ("cNBI", True, "score_cNBI"),
        ("Time", False, "score_Time"),
    ]:
        ordered = sub.sort_values(metric, ascending=not higher)
        vals = {}
        for pos, idx in enumerate(ordered.index):
            vals[idx] = (denom - pos) / denom
        parts[name] = vals
        out[name] = out.index.map(vals)
    out.loc[valid, "triple_score"] = (
        0.4 * out.loc[valid, "score_R"]
        + 0.3 * out.loc[valid, "score_cNBI"]
        + 0.3 * out.loc[valid, "score_Time"]
    )
    return out


def add_e26f_flags(df: pd.DataFrame) -> pd.DataFrame:
    ref_path = WORKSPACE / "research" / "dacts_e26f_reverse_search_20260519" / "outputs" / "reference_comparison.csv"
    ref = pd.read_csv(ref_path)
    e26f = ref[ref["name"].eq("e26f_reference")].iloc[0]
    r_ref = float(e26f["R"])
    c_ref = float(e26f["cNBI"])
    t_ref = float(e26f["Time"])
    out = df.copy()
    out["strict_e26f_like"] = (
        out["valid"]
        & (out["R"] <= r_ref + 0.0006)
        & (out["cNBI"] >= c_ref - 0.12)
        & (out["Time"] <= t_ref * 2.0)
    )
    out["loose_e26f_like"] = (
        out["valid"]
        & (out["R"] <= r_ref + 0.0015)
        & (out["cNBI"] >= c_ref - 0.45)
        & (out["Time"] <= t_ref * 3.0)
    )
    return out


def first_idx_at_k_hits(sub: pd.DataFrame, k: int) -> int:
    ordered = sub.sort_values("idx")
    cum = ordered["strict_e26f_like"].cumsum()
    hit = ordered.loc[cum >= k, "idx"]
    return int(hit.iloc[0]) if len(hit) else -1


def make_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        sub = df[df["method"].eq(method)].sort_values("idx")
        valid = sub[sub["valid"]]
        best = valid.sort_values("triple_score", ascending=False).iloc[0]
        rows.append(
            {
                "method": method,
                "nodes": len(sub),
                "valid": int(sub["valid"].sum()),
                "invalid": int((~sub["valid"]).sum()),
                "best_idx_by_triple_score": int(best["idx"]),
                "best_R": best["R"],
                "best_cNBI": best["cNBI"],
                "best_Time": best["Time"],
                "best_triple_score": best["triple_score"],
                "first_strict": first_idx_at_k_hits(sub, 1),
                "first_5_strict": first_idx_at_k_hits(sub, 5),
                "first_20_strict": first_idx_at_k_hits(sub, 20),
                "first_50_strict": first_idx_at_k_hits(sub, 50),
                "strict_count": int(sub["strict_e26f_like"].sum()),
                "loose_count": int(sub["loose_e26f_like"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "triple_metric_search_quality.csv", index=False, encoding="utf-8-sig")
    return out


def plot_best_so_far(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for method in METHODS:
        sub = df[df["method"].eq(method)].sort_values("idx")
        y = sub["triple_score"].fillna(-1).cummax()
        lw = 2.6 if method == "DACTS(GPT-5.5-none)" else 1.5
        ax.plot(sub["idx"], y, label=method, color=COLORS[method], lw=lw)
    ax.set_xlabel("Evaluated candidate index")
    ax.set_ylabel("Best-so-far triple-metric score")
    ax.set_title("Three-metric search progress")
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "triple_metric_best_so_far_with_dacts_rerun.png")
    fig.savefig(FIG_DIR / "triple_metric_best_so_far_with_dacts_rerun.pdf")
    plt.close(fig)


def plot_internal_rank_best_so_far() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    summary_rows = []
    for method, path in METHODS.items():
        sub = pd.read_csv(path)
        sub["valid"] = sub["valid"].astype(str).str.lower().isin(["true", "1", "yes"])
        sub["idx"] = pd.to_numeric(sub["idx"], errors="coerce")
        sub["rank_score"] = pd.to_numeric(sub["rank_score"], errors="coerce")
        sub = sub.sort_values("idx")
        y = sub["rank_score"].where(sub["valid"]).fillna(-1).cummax()
        lw = 2.6 if method == "DACTS(GPT-5.5-none)" else 1.5
        ax.plot(sub["idx"], y, label=method, color=COLORS[method], lw=lw)
        valid = sub[sub["valid"] & sub["rank_score"].notna()]
        best = valid.sort_values("rank_score", ascending=False).iloc[0]
        summary_rows.append(
            {
                "method": method,
                "best_idx_by_internal_rank_score": int(best["idx"]),
                "best_internal_rank_score": float(best["rank_score"]),
                "R": float(best["R"]),
                "cNBI": float(best["cNBI"]),
                "Time": float(best["Time"]),
            }
        )
    ax.set_xlabel("Evaluated candidate index")
    ax.set_ylabel("Best-so-far internal rank_score")
    ax.set_title("Best-so-far using each method's recorded score")
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "internal_rank_score_best_so_far.png")
    fig.savefig(FIG_DIR / "internal_rank_score_best_so_far.pdf")
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(
        TABLE_DIR / "internal_rank_score_best_by_method.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_sustained_hits(table: pd.DataFrame) -> None:
    ordered = table.set_index("method").loc[list(METHODS)].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    x = np.arange(len(ordered))
    axes[0].bar(x, ordered["strict_count"], color=[COLORS[m] for m in ordered["method"]])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ordered["method"], rotation=30, ha="right")
    axes[0].set_ylabel("# strict e26f-like candidates")
    axes[0].set_title("Family density, not just first hit")

    vals = ordered["first_20_strict"].replace(-1, np.nan)
    axes[1].bar(x, vals, color=[COLORS[m] for m in ordered["method"]])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ordered["method"], rotation=30, ha="right")
    axes[1].set_ylabel("Index of 20th strict hit")
    axes[1].set_title("Sustained-hit speed")
    for i, v in enumerate(ordered["first_20_strict"]):
        if int(v) < 0:
            axes[1].text(i, 5, "N/A", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sustained_e26f_like_hits.png")
    fig.savefig(FIG_DIR / "sustained_e26f_like_hits.pdf")
    plt.close(fig)


def candidate_code_size(method: str, df: pd.DataFrame) -> float:
    if method.startswith("DACTS"):
        return np.nan
    files = df.loc[df["method"].eq(method), "candidate_file"].dropna().astype(str).unique()
    sizes = []
    for item in files:
        path = Path(item)
        if path.exists():
            sizes.append(path.stat().st_size)
    return float(np.mean(sizes)) if sizes else np.nan


def make_interpretability_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    explicit_cols = [
        "clade",
        "mutation",
        "w_split",
        "w_bridge_mult",
        "w_degree",
        "w_nds",
        "w_core",
        "w_comp",
        "w_bridge_edges",
        "split_power",
        "degree_power",
        "nds_power",
        "core_power",
        "update_radius",
        "use_component",
        "comp_refresh",
    ]
    for method in METHODS:
        sub = df[df["method"].eq(method)].copy()
        typed_cols = [c for c in explicit_cols if c in sub.columns and sub[c].notna().any()]
        typed_coverage = 0.0
        if {"clade", "mutation"}.issubset(sub.columns):
            typed_coverage = float((sub["clade"].notna() & sub["mutation"].notna()).mean())
        non_llm_invalid = 0
        if "error" in sub.columns:
            err = sub.loc[~sub["valid"], "error"].fillna("").astype(str)
            non_llm_invalid = int((~err.str.contains("llm:", case=False, regex=False)).sum())
        strict = sub[sub["strict_e26f_like"]]
        typed_strict_families = 0
        if "clade" in strict.columns:
            typed_strict_families = int(strict["clade"].dropna().astype(str).nunique())
        rows.append(
            {
                "method": method,
                "typed_mechanism_coverage": typed_coverage,
                "explicit_mechanism_fields": len(typed_cols),
                "strict_e26f_like_count": int(sub["strict_e26f_like"].sum()),
                "typed_strict_family_count": typed_strict_families,
                "non_llm_invalid_rate": non_llm_invalid / max(1, len(sub)),
                "avg_generated_code_bytes": candidate_code_size(method, df),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "interpretability_metrics.csv", index=False, encoding="utf-8-sig")
    return out


def plot_interpretability(table: pd.DataFrame, df: pd.DataFrame) -> None:
    ordered = table.set_index("method").loc[list(METHODS)].reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    axes = axes.ravel()
    x = np.arange(len(ordered))
    labels = ordered["method"].tolist()

    axes[0].bar(x, ordered["typed_mechanism_coverage"], color=[COLORS[m] for m in labels])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Fraction of nodes")
    axes[0].set_title("Typed mechanism labels")

    axes[1].bar(x, ordered["explicit_mechanism_fields"], color=[COLORS[m] for m in labels])
    axes[1].set_ylabel("# fields")
    axes[1].set_title("Auditable mechanism state")

    axes[2].bar(x, ordered["typed_strict_family_count"], color=[COLORS[m] for m in labels])
    axes[2].set_ylabel("# typed families")
    axes[2].set_title("Families among strict hits")

    code = ordered["avg_generated_code_bytes"] / 1000.0
    axes[3].bar(x, code, color=[COLORS[m] for m in labels])
    axes[3].set_ylabel("KB")
    axes[3].set_title("Average generated code size")
    axes[3].text(0, 0.1, "typed config", ha="center", va="bottom", fontsize=7, rotation=90)
    axes[3].text(1, 0.1, "typed config", ha="center", va="bottom", fontsize=7, rotation=90)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.suptitle("Interpretability evidence: typed state vs. free-form code", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "interpretability_evidence_panel.png")
    fig.savefig(FIG_DIR / "interpretability_evidence_panel.pdf")
    plt.close(fig)

    strict = df[df["method"].eq("DACTS(GPT-5.5-none)") & df["strict_e26f_like"]].copy()
    if "clade" in strict.columns and not strict.empty:
        counts = strict["clade"].fillna("unknown").astype(str).value_counts()
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.bar(counts.index, counts.values, color="#D62728", alpha=0.85)
        ax.set_ylabel("# strict e26f-like candidates")
        ax.set_title("DACTS(GPT-5.5-none) mechanism families among strong candidates")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "dacts_strict_hit_mechanism_families.png")
        fig.savefig(FIG_DIR / "dacts_strict_hit_mechanism_families.pdf")
        plt.close(fig)


def main() -> None:
    setup()
    df = add_e26f_flags(add_global_triple_score(load_records()))
    df.to_csv(TABLE_DIR / "records_with_triple_score_and_flags.csv", index=False, encoding="utf-8-sig")
    quality = make_quality_table(df)
    interp = make_interpretability_table(df)
    plot_best_so_far(df)
    plot_internal_rank_best_so_far()
    plot_sustained_hits(quality)
    plot_interpretability(interp, df)
    print(quality.to_string(index=False))
    print(interp.to_string(index=False))


if __name__ == "__main__":
    main()
