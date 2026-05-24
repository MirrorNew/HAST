# -*- coding: utf-8 -*-
"""Time-aware analysis for HAST-FAC.

Uses existing 100-node HAST-FAC records and already completed full-12
evaluations. No new slow evaluation is performed here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "HAST-FAC"
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"


def boolish(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 170,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )


def fac_t_score(row: pd.Series) -> float:
    fac = float(row.get("fac_auc_adv") or 0.0)
    early = float(row.get("early_fac") or 0.0)
    worst = float(row.get("fac_worst_auc_adv") or 0.0)
    search_rank = float(row.get("rank_score") or 0.0)
    proxy_time = float(row.get("proxy_time_s") or 0.0)
    search_time = float(row.get("Time") or 0.0)
    benefit = 0.48 * fac + 0.20 * early + 0.10 * worst + 6.0 * search_rank
    penalty = (
        12.0 * math.log1p(max(0.0, proxy_time) / 0.70)
        + 180.0 * max(0.0, search_time - 0.022)
        + 8.0 * max(0.0, proxy_time - 1.20) ** 2
    )
    if proxy_time > 1.80:
        penalty += 16.0 + 10.0 * (proxy_time - 1.80)
    if search_time > 0.032:
        penalty += 10.0 + 500.0 * (search_time - 0.032)
    return benefit - penalty


def load_records() -> pd.DataFrame:
    df = pd.read_csv(RUN_DIR / "search_records.csv")
    df = df[(pd.to_numeric(df["idx"], errors="coerce") > 0) & boolish(df["valid"])].copy()
    for col in [
        "idx",
        "fac_score",
        "fac_auc_adv",
        "early_fac",
        "fac_worst_auc_adv",
        "proxy_auc_cNBI",
        "proxy_time_s",
        "Time",
        "R",
        "cNBI",
        "rank_score",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fac_t_score"] = df.apply(fac_t_score, axis=1)
    df["time_bucket"] = pd.cut(
        df["proxy_time_s"],
        bins=[-np.inf, 0.8, 1.2, 1.8, np.inf],
        labels=["fast<=0.8", "ok<=1.2", "slow<=1.8", "too_slow"],
    )
    return df


def make_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    cols = [
        "idx",
        "target_family",
        "fac_t_score",
        "fac_score",
        "fac_auc_adv",
        "early_fac",
        "fac_worst_auc_adv",
        "proxy_auc_cNBI",
        "proxy_time_s",
        "Time",
        "R",
        "cNBI",
        "candidate_file",
    ]
    old_top = df.sort_values("fac_score", ascending=False).head(15)[cols]
    new_top = df.sort_values("fac_t_score", ascending=False).head(15)[cols]
    fast_top = df[df["proxy_time_s"].le(1.2)].sort_values("fac_t_score", ascending=False).head(15)[cols]
    bucket = (
        df.groupby("time_bucket", observed=False)
        .agg(
            n=("idx", "count"),
            mean_fac_adv=("fac_auc_adv", "mean"),
            max_fac_adv=("fac_auc_adv", "max"),
            mean_proxy_auc=("proxy_auc_cNBI", "mean"),
            mean_proxy_time=("proxy_time_s", "mean"),
            best_fac_t=("fac_t_score", "max"),
        )
        .reset_index()
    )
    corr = df[["fac_score", "fac_t_score", "fac_auc_adv", "early_fac", "proxy_auc_cNBI", "proxy_time_s", "Time"]].corr(
        method="spearman"
    )
    old_top.to_csv(TABLE_DIR / "hast_fac_time_old_top.csv", index=False, encoding="utf-8-sig")
    new_top.to_csv(TABLE_DIR / "hast_fac_time_aware_top.csv", index=False, encoding="utf-8-sig")
    fast_top.to_csv(TABLE_DIR / "hast_fac_time_fast_candidates.csv", index=False, encoding="utf-8-sig")
    bucket.to_csv(TABLE_DIR / "hast_fac_time_bucket_summary.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(TABLE_DIR / "hast_fac_time_correlation.csv", encoding="utf-8-sig")
    return {"old_top": old_top, "new_top": new_top, "fast_top": fast_top, "bucket": bucket, "corr": corr}


def plot(df: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> None:
    setup_style()
    full_path = TABLE_DIR / "hast_fac_online_full12_mean.csv"
    full = pd.read_csv(full_path) if full_path.exists() else pd.DataFrame()
    base = pd.read_csv(ROOT / "final_12graph_eval" / "hast_12graph_summary.csv")
    base_mean = base.groupby("method")[["R", "auc_cNBI", "time_s"]].mean().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    sc = ax.scatter(df["proxy_time_s"], df["fac_auc_adv"], c=df["fac_t_score"], cmap="viridis", s=35)
    ax.axvline(1.2, color="#D62728", ls="--", lw=1)
    ax.axvline(1.8, color="#D62728", ls=":", lw=1)
    ax.set_xlabel("proxy_time_s")
    ax.set_ylabel("FAC AUC advantage")
    ax.set_title("A. FAC vs time: old score drifts slow")
    fig.colorbar(sc, ax=ax, label="FAC-T score")

    ax = axes[0, 1]
    for label, col in [("old FAC", "fac_score"), ("FAC-T", "fac_t_score")]:
        y = df.sort_values("idx")[col].cummax()
        ax.plot(df.sort_values("idx")["idx"], y, label=label, lw=2)
    ax.set_xlabel("candidate idx")
    ax.set_title("B. Best-so-far old FAC vs FAC-T")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    if not full.empty:
        ax.scatter(full["time_s"], full["auc_cNBI"], color="#D62728", label="HAST-FAC evaluated")
        for _, row in full.iterrows():
            ax.text(row["time_s"], row["auc_cNBI"], str(int(row["candidate_idx"])), fontsize=7)
    for method in ["HAST", "E26F", "PUCT", "Clade-AHD-like", "FunSearch-like"]:
        row = base_mean[base_mean["method"].eq(method)]
        if not row.empty:
            ax.scatter(row["time_s"], row["auc_cNBI"], label=method, s=30)
    ax.set_xlabel("full 12-graph mean time_s")
    ax.set_ylabel("full 12-graph AUC-cNBI")
    ax.set_title("C. Full 12-graph accuracy-time tradeoff")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[1, 1]
    b = tables["bucket"]
    ax.bar(b["time_bucket"].astype(str), b["max_fac_adv"], color="#4C78A8", label="max FAC advantage")
    ax2 = ax.twinx()
    ax2.plot(b["time_bucket"].astype(str), b["n"], color="#F28E2B", marker="o", label="count")
    ax.set_title("D. Time buckets")
    ax.set_ylabel("max FAC advantage")
    ax2.set_ylabel("candidate count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hast_fac_time_aware_analysis.png")
    fig.savefig(FIG_DIR / "hast_fac_time_aware_analysis.pdf")
    plt.close(fig)


def write_report(tables: Dict[str, pd.DataFrame]) -> None:
    full_path = TABLE_DIR / "hast_fac_online_full12_mean.csv"
    full = pd.read_csv(full_path) if full_path.exists() else pd.DataFrame()
    base = pd.read_csv(ROOT / "final_12graph_eval" / "hast_12graph_summary.csv")
    base_mean = base.groupby("method")[["R", "auc_cNBI", "time_s"]].mean().reset_index().sort_values("auc_cNBI", ascending=False)
    lines: List[str] = [
        "# HAST-FAC 时间漏洞与 FAC-T 修正",
        "",
        "## 结论",
        "",
        "你指出的问题是对的：旧 FAC 对时间惩罚太轻，导致搜索偏向复杂 frontier/two-hop 局部扫描。它能提高 cNBI，但在大图上会变慢。",
        "",
        "## 证据：旧 FAC top 候选",
        "",
        tables["old_top"].head(10).to_markdown(index=False),
        "",
        "## 时间敏感 FAC-T 重排",
        "",
        tables["new_top"].head(10).to_markdown(index=False),
        "",
        "## 快速候选池 proxy_time<=1.2",
        "",
        tables["fast_top"].head(10).to_markdown(index=False),
        "",
        "## 时间桶统计",
        "",
        tables["bucket"].to_markdown(index=False),
        "",
        "## 已完成 full 12 图结果",
        "",
        full.to_markdown(index=False) if not full.empty else "No full-12 HAST-FAC table found.",
        "",
        "## 现有方法 full 12 图均值",
        "",
        base_mean.to_markdown(index=False),
        "",
        "## 修正后的搜索规则",
        "",
        "1. 候选先过时间门槛：`proxy_time_s <= 1.2` 优先，`>1.8` 大幅扣分。",
        "2. search graph 平均 `Time > 0.032` 直接强惩罚，因为它和 proxy_time 高相关。",
        "3. FAC-T 分数不再只看碎裂优势，而是 `fracture benefit - log/time/gate penalty`。",
        "4. Prompt 里强制 capped neighborhood scan，例如 `nbrs[:48]`，避免无限二跳集合膨胀。",
        "5. 下一轮如果继续在线搜索，应该从候选 21/1/8/7 等快速路线附近继续，而不是从 44/56 的慢路线继续。",
    ]
    (REPORT_DIR / "hast_fac_time_aware_analysis_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    df = load_records()
    tables = make_tables(df)
    plot(df, tables)
    write_report(tables)
    print(REPORT_DIR / "hast_fac_time_aware_analysis_cn.md")


if __name__ == "__main__":
    main()
