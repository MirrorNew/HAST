# -*- coding: utf-8 -*-
"""Analyze the small HAST-FAC-T online run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "HAST-FACT-ONLINE60"
TABLE = ROOT / "tables"
FIG = ROOT / "figures"
REPORT = ROOT / "reports"
ABLATION = ROOT.parents[1] / "research" / "tree_search_ablation_20260520"


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    rec = pd.read_csv(RUN / "search_records.csv")
    rec = rec[rec["idx"].astype(int) > 0].copy()
    for col in ["idx", "R", "cNBI", "Time", "proxy_auc_cNBI", "proxy_time_s", "fac_auc_adv", "early_fac", "fac_score"]:
        rec[col] = num(rec[col])
    rec["valid_bool"] = rec["valid"].astype(str).str.lower().isin(["true", "1"])
    valid = rec[rec["valid_bool"]].copy()
    valid["best_fac"] = valid["fac_score"].cummax()
    valid["best_proxy_auc"] = valid["proxy_auc_cNBI"].cummax()
    valid["fast_gate"] = (valid["proxy_time_s"] <= 0.45) & (valid["Time"] <= 0.032)
    valid["strong"] = (valid["fac_auc_adv"] >= 50) & valid["fast_gate"]
    rec["strong"] = False
    rec.loc[valid.index, "strong"] = valid["strong"]

    fam = (
        rec.groupby("target_family")
        .agg(
            n=("idx", "count"),
            valid=("valid_bool", "sum"),
            best_fac=("fac_score", "max"),
            mean_fac=("fac_score", "mean"),
            mean_proxy_time=("proxy_time_s", "mean"),
            strong=("strong", "sum"),
        )
        .reset_index()
        .sort_values("best_fac", ascending=False)
    )
    fam.to_csv(TABLE / "HAST-FACT-ONLINE60_family_summary.csv", index=False, encoding="utf-8-sig")

    full = pd.read_csv(TABLE / "HAST-FACT-ONLINE60_full12_mean.csv")
    full["method"] = "HAST-FAC-T online #" + full["candidate_idx"].astype(str)
    fast = pd.read_csv(TABLE / "hast_fact_fast_probe_full12_mean_compare.csv")
    keep = ["FAST21-cap24", "FAST7-cap32-approx", "E26F", "PUCT", "FunSearch-like", "Clade-AHD-like", "HDA", "CoreHD"]
    comp = pd.concat(
        [
            fast[fast["method"].isin(keep)][["method", "R", "auc_cNBI", "time_s"]],
            full[["method", "R", "auc_cNBI", "time_s"]],
        ],
        ignore_index=True,
    ).drop_duplicates("method")
    comp = comp.sort_values("auc_cNBI", ascending=False)
    comp.to_csv(TABLE / "HAST-FACT-ONLINE60_full12_compare.csv", index=False, encoding="utf-8-sig")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(valid["idx"], valid["best_fac"], color="#D62728", lw=2.0, label="best FAC-T")
    ax.plot(valid["idx"], valid["best_proxy_auc"], color="#4C78A8", lw=1.6, label="best proxy AUC")
    ax.set_xlabel("candidate idx")
    ax.set_title("Online progress")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    colors = np.where(valid["strong"], "#D62728", np.where(valid["fast_gate"], "#59A14F", "#9AA0A6"))
    ax.scatter(valid["proxy_time_s"], valid["fac_auc_adv"], c=colors, s=34, alpha=0.9)
    ax.axvline(0.45, color="#D62728", ls="--", lw=1)
    ax.axhline(50, color="#D62728", ls="--", lw=1)
    ax.set_xlabel("proxy time (s)")
    ax.set_ylabel("FAC AUC advantage")
    ax.set_title("Fast strong region")

    ax = axes[1, 0]
    fam_plot = fam.sort_values("best_fac")
    ax.barh(fam_plot["target_family"], fam_plot["best_fac"], color="#59A14F")
    ax.set_xlabel("best FAC-T")
    ax.set_title("Which family received usable credit")

    ax = axes[1, 1]
    plot_comp = comp.copy()
    colors = ["#D62728" if ("HAST-FAC-T" in m or "FAST" in m) else "#9AA0A6" for m in plot_comp["method"]]
    ax.barh(plot_comp["method"], plot_comp["auc_cNBI"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("mean auc_cNBI")
    ax.set_title("12-graph check")
    fig.tight_layout()
    fig.savefig(FIG / "HAST-FACT-ONLINE60_diagnostics.png", dpi=240)
    plt.close(fig)

    # Simple runtime-quality frontier figure.
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(comp["time_s"], comp["auc_cNBI"], c=["#D62728" if ("HAST-FAC-T" in m or "FAST" in m) else "#9AA0A6" for m in comp["method"]], s=52)
    for _, r in comp.iterrows():
        label = str(r["method"]).replace("HAST-FAC-T online ", "online ")
        ax.annotate(label, (r["time_s"], r["auc_cNBI"]), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("mean time on 12 graphs (log s)")
    ax.set_ylabel("mean auc_cNBI")
    ax.set_title("Quality-runtime frontier")
    fig.tight_layout()
    fig.savefig(FIG / "HAST-FACT-ONLINE60_runtime_frontier.png", dpi=240)
    plt.close(fig)

    best_online = full.sort_values("auc_cNBI", ascending=False).iloc[0]
    fast21 = comp[comp["method"] == "FAST21-cap24"].iloc[0]
    e26f = comp[comp["method"] == "E26F"].iloc[0]
    puct = comp[comp["method"] == "PUCT"].iloc[0]
    strong_n = int(valid["strong"].sum())
    report = f"""# HAST-FAC-T ONLINE60 诊断

## 这轮说明了什么

- 在线搜索实际完成 {len(rec)} 个候选，其中有效 {int(valid.shape[0])} 个。
- FAC-T 确实改变了搜索方向：`twohop_advantage` family 的 best FAC-T 最高，并且产生了 {strong_n} 个满足 `fac_auc_adv>=50` 且速度门内的候选。
- 但在线 LLM 仍然偏向“略复杂的 capped two-hop”，没有自动找到我们手写 `FAST21-cap24` 那种更干净的版本。

## 12 图复核

- 在线最佳候选：idx {int(best_online['candidate_idx'])}，auc_cNBI={best_online['auc_cNBI']:.3f}，time={best_online['time_s']:.3f}s。
- 手写快速近似 `FAST21-cap24`：auc_cNBI={fast21['auc_cNBI']:.3f}，time={fast21['time_s']:.3f}s。
- E26F：auc_cNBI={e26f['auc_cNBI']:.3f}，time={e26f['time_s']:.3f}s。
- PUCT：auc_cNBI={puct['auc_cNBI']:.3f}，time={puct['time_s']:.3f}s。

## 对下一步 HAST 的真实启发

1. FAC-T 有用，但还不够“硬”。它会惩罚慢算法，但 prompt 仍允许 LLM 在二跳上做微小加法，导致候选比 `FAST21-cap24` 慢。
2. 下一步不要增加新搜索框架，而是把候选语言收窄：只允许 bounded two-hop template 的少量参数变体，例如 `CAP_N/CAP_2/update_cap/phase weights`。
3. 最应该搜索的是“低成本近似信号的参数和组合”，不是任意 Python 启发式。
4. 当前通用 LLM 树搜索的问题在这里很明显：整体分数会奖励复杂局部扫描；没有时间信用时会走向慢算法，有时间信用但没有结构约束时会走向“勉强不慢但仍冗余”的算法。

## 图和表

- 诊断图：`{(FIG / 'HAST-FACT-ONLINE60_diagnostics.png').as_posix()}`
- 运行时-质量前沿：`{(FIG / 'HAST-FACT-ONLINE60_runtime_frontier.png').as_posix()}`
- family 表：`{(TABLE / 'HAST-FACT-ONLINE60_family_summary.csv').as_posix()}`
- 12 图对比表：`{(TABLE / 'HAST-FACT-ONLINE60_full12_compare.csv').as_posix()}`
"""
    (REPORT / "HAST-FACT-ONLINE60_diagnostics_cn.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
