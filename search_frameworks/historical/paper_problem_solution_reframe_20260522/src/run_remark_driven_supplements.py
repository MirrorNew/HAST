# -*- coding: utf-8 -*-
"""Supplement experiments requested in remarks 1-7.

This script does not call any LLM API. It consolidates existing logs and curve
records into paper-facing ablations, comparisons, and interpretability figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HAST = ROOT.parents[0] / "hast_experiment_20260521"
BOOST = ROOT.parents[0] / "iclr_minimal_boost_20260522"
TABLE = ROOT / "tables"
PAPER_TABLE = ROOT / "paper_tables"
FIG = ROOT / "figures"
REPORT = ROOT / "reports"


COLORS = {
    "hast": "#0072B2",
    "bounded": "#009E73",
    "warn": "#D55E00",
    "puct": "#E69F00",
    "llm": "#CC79A7",
    "base": "#7B8794",
}


def setup() -> None:
    for p in [PAPER_TABLE, FIG, REPORT]:
        p.mkdir(parents=True, exist_ok=True)
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
        }
    )


def savefig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main_result_rows() -> pd.DataFrame:
    return pd.read_csv(PAPER_TABLE / "table_main_results_unified_hast.csv")


def search_cost_summary() -> pd.DataFrame:
    rows = []
    specs = [
        ("Stage 0: initial free search", HAST / "tables" / "hast_search_records.csv", "prompt+eval"),
        ("Stage 1: relative-credit free search", HAST / "tables" / "hast_fac_online_search_records.csv", "prompt+eval+proxy"),
        ("Stage 2: cost-aware free search", HAST / "tables" / "HAST-FACT-ONLINE60_search_records.csv", "prompt+eval+proxy"),
        ("Stage 3: constrained candidate search", HAST / "tables" / "hast_bounded_template_probe_summary.csv", "eval+proxy"),
    ]
    for label, path, cost_kind in specs:
        df = pd.read_csv(path)
        valid = df["valid"].astype(str).str.lower().isin(["true", "1", "yes"]) if "valid" in df.columns else pd.Series([True] * len(df))
        row = {
            "stage": label,
            "candidates": len(df),
            "valid": int(valid.sum()),
            "valid_rate": float(valid.mean()),
            "cost_kind": cost_kind,
            "logged_prompt_s": float(pd.to_numeric(df.get("prompt_elapsed_s", pd.Series(dtype=float)), errors="coerce").sum()),
            "logged_eval_s": float(pd.to_numeric(df.get("Time", pd.Series(dtype=float)), errors="coerce").sum()),
            "logged_proxy_eval_s": float(pd.to_numeric(df.get("proxy_time_s", pd.Series(dtype=float)), errors="coerce").sum()),
        }
        row["logged_total_s"] = row["logged_prompt_s"] + row["logged_eval_s"] + row["logged_proxy_eval_s"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(PAPER_TABLE / "table_search_cost_summary.csv", index=False, encoding="utf-8-sig")
    return out


def module_ablation_table(main: pd.DataFrame, search_cost: pd.DataFrame) -> pd.DataFrame:
    # Main ablation rows are selected to answer remark 7.1 directly.
    map_rows = [
        {
            "ablation": "Initial automatic search",
            "removed_module": "relative cNBI credit, time credit, generation constraints",
            "source": "HAST initial family-credit run",
            "paper_label": None,
            "R": 0.43944709765877016,
            "auc_cNBI": 242.5261230687997,
            "time_s": 3.383756616666991,
            "search_stage": "Stage 0: initial free search",
        },
        {
            "ablation": "Relative credit only",
            "removed_module": "time credit, generation constraints",
            "source": "HAST free search candidate C44",
            "paper_label": "HAST no cost control",
            "search_stage": "Stage 1: relative-credit free search",
        },
        {
            "ablation": "Relative + time credit",
            "removed_module": "generation constraints",
            "source": "HAST cost-aware candidate C24",
            "paper_label": "HAST cost-aware",
            "search_stage": "Stage 2: cost-aware free search",
        },
        {
            "ablation": "Full HAST, quality point",
            "removed_module": "none",
            "source": "constrained automatic search",
            "paper_label": "HAST-Bounded quality",
            "search_stage": "Stage 3: constrained candidate search",
        },
        {
            "ablation": "Full HAST, speed point",
            "removed_module": "none",
            "source": "constrained automatic search",
            "paper_label": "HAST-Bounded speed",
            "search_stage": "Stage 3: constrained candidate search",
        },
    ]
    rows = []
    cost = search_cost.set_index("stage")
    for spec in map_rows:
        row = spec.copy()
        if row.get("paper_label"):
            m = main[main["paper_label"].eq(row["paper_label"])].iloc[0]
            row["R"] = float(m["R"])
            row["auc_cNBI"] = float(m["auc_cNBI"])
            row["time_s"] = float(m["time_s"])
        stage = row["search_stage"]
        row["logged_search_total_s"] = float(cost.loc[stage, "logged_total_s"]) if stage in cost.index else np.nan
        row["valid_rate"] = float(cost.loc[stage, "valid_rate"]) if stage in cost.index else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(PAPER_TABLE / "table_module_ablation_three_mechanisms.csv", index=False, encoding="utf-8-sig")
    return out


def draw_module_ablation(df: pd.DataFrame) -> None:
    labels = ["Initial", "Relative\ncredit", "Time\ncredit", "Full\nquality", "Full\nspeed"]
    colors = [COLORS["base"], COLORS["warn"], COLORS["hast"], COLORS["bounded"], COLORS["bounded"]]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].bar(labels, df["R"], color=colors)
    axes[0].set_title("GCC/R after removal")
    axes[0].set_ylabel("Mean R / GCC (lower is better)")
    axes[1].bar(labels, df["auc_cNBI"], color=colors)
    axes[1].set_title("Residual fragmentation")
    axes[1].set_ylabel("Mean auc-cNBI (higher is better)")
    axes[2].bar(labels, df["time_s"], color=colors)
    axes[2].set_yscale("log")
    axes[2].set_title("Final algorithm runtime")
    axes[2].set_ylabel("Mean runtime per graph (s, log)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Module ablation: each mechanism fixes the failure exposed by the previous stage", y=1.03, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "fig6_module_ablation_three_mechanisms")


def top3_and_framework_table(main: pd.DataFrame) -> pd.DataFrame:
    method_summary = pd.read_csv(TABLE / "aaai_followup_method_mean_summary.csv")
    bounded_compare = pd.read_csv(HAST / "tables" / "hast_bounded_template_probe_full12_compare.csv")
    rows = []

    def add(label: str, role: str, auc: float, r: float, time_s: float, internal: str = "") -> None:
        rows.append({"paper_label": label, "role": role, "R": r, "auc_cNBI": auc, "time_s": time_s, "internal_name": internal})

    for method, role in [
        ("HDA", "root heuristic"),
        ("PUCT", "search baseline"),
        ("FunSearch-like", "LLM-search baseline"),
        ("Clade-AHD-like", "LLM-search baseline"),
        ("MCTS-AHD-like", "LLM-search baseline"),
        ("AlphaEvolve-like", "LLM-search baseline"),
    ]:
        m = method_summary[method_summary["method"].eq(method)].iloc[0]
        add(method if method != "HDA" else "HDA root", role, float(m["mean_auc_cNBI"]), float(m["mean_R"]), float(m["mean_time_s"]), method)

    core = main[main["paper_label"].eq("CoreHD")].iloc[0]
    add("CoreHD", "traditional baseline", float(core["auc_cNBI"]), float(core["R"]), float(core["time_s"]), "CoreHD")

    for method, label in [
        ("FAST21-cap24", "HAST final top-1 quality"),
        ("BT-n16-t8-u24", "HAST final top-1 speed"),
        ("BT-n16-t8-u18", "HAST final top-3 candidate"),
        ("BT-n32-t8-u24", "HAST final top-3 candidate"),
    ]:
        m = bounded_compare[bounded_compare["method"].eq(method)].iloc[0]
        add(label, "HAST constrained-search output", float(m["auc_cNBI"]), float(m["R"]), float(m["time_s"]), method)
    out = pd.DataFrame(rows)
    puct_auc = float(out[out["paper_label"].eq("PUCT")]["auc_cNBI"].iloc[0])
    puct_time = float(out[out["paper_label"].eq("PUCT")]["time_s"].iloc[0])
    out["retention_vs_PUCT"] = out["auc_cNBI"] / puct_auc
    out["speedup_vs_PUCT"] = puct_time / out["time_s"]
    out.to_csv(PAPER_TABLE / "table_top3_final_vs_frameworks.csv", index=False, encoding="utf-8-sig")
    return out


def draw_top3_comparison(df: pd.DataFrame) -> None:
    order = [
        "HDA root",
        "CoreHD",
        "PUCT",
        "FunSearch-like",
        "Clade-AHD-like",
        "MCTS-AHD-like",
        "AlphaEvolve-like",
        "HAST final top-1 quality",
        "HAST final top-1 speed",
        "HAST final top-3 candidate",
    ]
    # Keep the two top-3 candidate rows distinguishable by appending internal name for display.
    plot = df.copy()
    plot["display"] = plot["paper_label"]
    dup = plot["paper_label"].eq("HAST final top-3 candidate")
    plot.loc[dup, "display"] = plot.loc[dup, "paper_label"] + "\n(" + plot.loc[dup, "internal_name"] + ")"
    plot["sort_key"] = plot["paper_label"].map({k: i for i, k in enumerate(order)}).fillna(99)
    plot = plot.sort_values(["sort_key", "auc_cNBI"], ascending=[True, False])
    colors = [
        COLORS["bounded"] if x.startswith("HAST final") else (COLORS["llm"] if "like" in x or "Alpha" in x or "MCTS" in x else COLORS["base"])
        for x in plot["paper_label"]
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.1))
    axes[0].barh(plot["display"], plot["R"], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean R / GCC (lower is better)")
    axes[0].set_title("GCC/R")
    axes[1].barh(plot["display"], plot["auc_cNBI"], color=colors)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean auc-cNBI (higher is better)")
    axes[1].set_title("cNBI")
    axes[2].barh(plot["display"], plot["time_s"], color=colors)
    axes[2].invert_yaxis()
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Mean runtime (s, log)")
    axes[2].set_title("Runtime")
    fig.suptitle("Final HAST candidates compared with other search frameworks and heuristics", y=1.02, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "fig7_top3_final_vs_frameworks")


def draw_interpretability() -> pd.DataFrame:
    families = pd.read_csv(HAST / "tables" / "hast_learned_families.csv")
    family_rows = families[
        families["family"].isin(["local_twohop_neighbor", "local_twohop_bridge", "simple_degree", "component_heavy"])
    ].copy()
    family_rows.to_csv(PAPER_TABLE / "table_interpretability_family_summary.csv", index=False, encoding="utf-8-sig")

    term_rows = pd.DataFrame(
        [
            {
                "term": "residual degree",
                "what_it_rewards": "removing locally influential nodes",
                "why_it_helps": "keeps the root heuristic's strong first-order signal",
                "bounded_by": "local score only",
            },
            {
                "term": "frontier / weak tie",
                "what_it_rewards": "nodes adjacent to low-degree or fragile neighbors",
                "why_it_helps": "breaks bridges between residual components earlier",
                "bounded_by": "CAP_N neighbors",
            },
            {
                "term": "two-hop boundary",
                "what_it_rewards": "neighbors that spill into many outside nodes",
                "why_it_helps": "targets nodes that split the residual graph beyond the largest component",
                "bounded_by": "CAP_2 scanned two-hop nodes",
            },
            {
                "term": "redundancy penalty",
                "what_it_rewards": "avoiding densely overlapping neighborhoods",
                "why_it_helps": "discourages wasting removals inside already redundant local clusters",
                "bounded_by": "same local scan as boundary term",
            },
            {
                "term": "phase weights",
                "what_it_rewards": "different behavior early/mid/late in removal",
                "why_it_helps": "uses degree early, fragmentation pressure later",
                "bounded_by": "fixed removal-ratio phases",
            },
        ]
    )
    term_rows.to_csv(PAPER_TABLE / "table_interpretability_score_terms.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), gridspec_kw={"width_ratios": [1.05, 1.2]})
    ax = axes[0]
    ax.scatter(family_rows["mean_Time"], family_rows["mean_cNBI"], s=120, color=COLORS["hast"], edgecolor="black")
    for _, r in family_rows.iterrows():
        ax.text(float(r["mean_Time"]) * 1.05, float(r["mean_cNBI"]) + 0.7, str(r["family"]), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Mean proxy runtime by family (s, log)")
    ax.set_ylabel("Mean proxy cNBI")
    ax.set_title("Free search reveals useful but costly families")

    ax = axes[1]
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        ("degree", 0.08, 0.64, "#E8EDF2"),
        ("frontier / weak tie", 0.38, 0.64, "#E8F2EE"),
        ("two-hop boundary", 0.68, 0.64, "#FFF6E5"),
        ("redundancy penalty", 0.23, 0.25, "#F5E8E8"),
        ("phase weights", 0.55, 0.25, "#EEF2FF"),
    ]
    for text, x, y, fc in boxes:
        patch = FancyBboxPatch((x, y), 0.24, 0.18, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor=fc, edgecolor="#9CA3AF")
        ax.add_patch(patch)
        ax.text(x + 0.12, y + 0.09, text, ha="center", va="center", fontweight="bold", fontsize=9)
    ax.text(0.5, 0.09, "HAST-Bounded score = local fracture signals, with capped neighbor and two-hop access", ha="center", fontsize=9)
    ax.set_title("Why the final candidate works", fontsize=11)
    fig.tight_layout()
    savefig(fig, "fig8_interpretability_final_candidate")
    return term_rows


def write_analysis_report() -> None:
    text = """# 对备注 1-7 的处理计划

## 需要改故事的点

1. cNBI 不能突然出现，必须先说明传统网络瓦解通常以 LCC/GCC/R 为核心，而我们的补充问题是残余碎裂结构。
2. cNBI 公式必须给出：`cNBI = pairwise_disconnected * effective_components / (1 + top5_component_mass)`。
3. Spearman/Pearson 必须解释成“秩相关”和“线性相关”，用于证明 cNBI 与 R 有关但不重复。
4. HAST no cost control 的候选必须编号为 C44/C56/C41/C37，不能重复写 relative credit only。
5. 不使用 Pareto 作为主文术语，改成“quality point”和“speed point”。
6. 主文曲线要补 GCC、cNBI、runtime；其他框架对比进入补充图。
7. 新增三类补充实验：模块消融、top3 final candidates vs frameworks、最终候选可解释性。

## 已生成的新表/图

- `paper_tables/table_module_ablation_three_mechanisms.csv`
- `figures/fig6_module_ablation_three_mechanisms.png`
- `paper_tables/table_top3_final_vs_frameworks.csv`
- `figures/fig7_top3_final_vs_frameworks.png`
- `paper_tables/table_interpretability_family_summary.csv`
- `paper_tables/table_interpretability_score_terms.csv`
- `figures/fig8_interpretability_final_candidate.png`

## 文献判断

网络瓦解文献通常以 giant/largest connected component 和 robustness R 为核心评价。相关工作也会讨论 dismantled graph 应被分解为小连通分量，或报告 LCC/SLCC 等曲线。因此，本文不能说“别人完全没看到残余碎裂”，更准确的写法是：主流目标以 LCC/GCC/R 为主，而 HAST 的搜索信用需要一个更细的残余碎裂辅助信号。
"""
    (REPORT / "remark_response_analysis_cn.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup()
    main = main_result_rows()
    cost = search_cost_summary()
    ablation = module_ablation_table(main, cost)
    draw_module_ablation(ablation)
    top3 = top3_and_framework_table(main)
    draw_top3_comparison(top3)
    draw_interpretability()
    write_analysis_report()
    print(f"[done] wrote supplement tables to {PAPER_TABLE}")
    print(f"[done] wrote supplement figures to {FIG}")
    print(f"[done] wrote report to {REPORT / 'remark_response_analysis_cn.md'}")


if __name__ == "__main__":
    main()
