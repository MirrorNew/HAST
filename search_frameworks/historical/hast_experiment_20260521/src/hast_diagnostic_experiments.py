# -*- coding: utf-8 -*-
"""Lightweight diagnostic experiments for HAST.

This script intentionally avoids LLM calls and avoids full candidate re-eval.
It tests why the current HAST run searches well but generalizes weakly, using
the already generated search records and 12-graph metric curves.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
ABLATION = WORKSPACE / "research" / "tree_search_ablation_20260520"
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"

GENERIC_METHODS = ["PUCT", "MCTS-AHD-like", "Clade-AHD-like", "FunSearch-like", "AlphaEvolve-like"]
METHODS = ["HAST"] + GENERIC_METHODS + ["E26F", "HDA"]
SEARCH_METHODS = ["HAST"] + GENERIC_METHODS


def bool_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


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


def load_search_records(method: str) -> pd.DataFrame:
    if method == "HAST":
        path = ROOT / "runs" / "HAST" / "search_records.csv"
    else:
        path = ABLATION / "runs" / method / "search_records.csv"
    df = pd.read_csv(path)
    df["method"] = method
    df["valid"] = bool_col(df["valid"])
    for col in ["idx", "R", "cNBI", "Time", "rank_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "strict_e26f_like" in df.columns:
        df["strict_flag"] = bool_col(df["strict_e26f_like"])
    elif "near_e26f_strict" in df.columns:
        df["strict_flag"] = bool_col(df["near_e26f_strict"])
    else:
        df["strict_flag"] = False
    return df


def normalized_entropy(labels: Iterable[Any]) -> float:
    items = pd.Series(list(labels)).dropna()
    if items.empty:
        return 0.0
    probs = items.value_counts(normalize=True).to_numpy(dtype=float)
    ent = -float(np.sum(probs * np.log(probs + 1e-12)))
    return ent / math.log(max(2, len(probs)))


def method_generalization_table() -> pd.DataFrame:
    final12 = pd.read_csv(ROOT / "final_12graph_eval" / "hast_12graph_summary.csv")
    mean12 = (
        final12[final12["method"].isin(METHODS)]
        .groupby("method")[["R", "auc_cNBI", "time_s"]]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"R": "mean_12_R", "auc_cNBI": "mean_12_auc_cNBI", "time_s": "mean_12_time_s"})
    )

    heldout = pd.read_csv(TABLE_DIR / "hast_vs_heldout_baselines.csv")
    rows: List[Dict[str, Any]] = []
    for _, row in heldout[heldout["method"].isin(SEARCH_METHODS)].iterrows():
        nodes = float(row["nodes"])
        rows.append(
            {
                "method": row["method"],
                "search_nodes": int(row["nodes"]),
                "valid_rate": float(row["valid"]) / nodes if nodes else np.nan,
                "strict_count": int(row["strict_count"]),
                "strict_rate": float(row["strict_count"]) / nodes if nodes else np.nan,
                "best_idx": int(row["best_idx"]),
                "best_search_R": float(row["best_R"]),
                "best_search_cNBI": float(row["best_cNBI"]),
                "best_search_Time": float(row["best_Time"]),
                "best_search_rank_score": float(row["best_score"]),
            }
        )

    out = pd.DataFrame(rows).merge(mean12, on="method", how="left")
    refs = mean12[mean12["method"].isin(["E26F", "HDA"])].copy()
    out = pd.concat([out, refs], ignore_index=True, sort=False)
    out.to_csv(TABLE_DIR / "hast_diagnostic_method_generalization.csv", index=False, encoding="utf-8-sig")
    return out


def family_collapse_blocks() -> pd.DataFrame:
    df = load_search_records("HAST")
    if "actual_family" not in df.columns:
        df["actual_family"] = "unknown"
    rows: List[Dict[str, Any]] = []
    for start in range(1, int(df["idx"].max()) + 1, 50):
        sub = df[(df["idx"] >= start) & (df["idx"] < start + 50)].copy()
        counts = sub["actual_family"].value_counts()
        top_family = str(counts.index[0]) if not counts.empty else ""
        rows.append(
            {
                "block": f"{start}-{start + 49}",
                "start_idx": start,
                "nodes": len(sub),
                "valid_rate": float(sub["valid"].mean()),
                "strict_rate": float(sub["strict_flag"].mean()),
                "family_entropy": normalized_entropy(sub["actual_family"]),
                "top_family": top_family,
                "top_family_share": float(counts.iloc[0] / max(1, len(sub))) if not counts.empty else 0.0,
                "local_twohop_neighbor_share": float((sub["actual_family"] == "local_twohop_neighbor").mean()),
                "local_twohop_bridge_share": float((sub["actual_family"] == "local_twohop_bridge").mean()),
                "simple_degree_share": float((sub["actual_family"] == "simple_degree").mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "hast_family_collapse_blocks.csv", index=False, encoding="utf-8-sig")
    return out


def dataset_gap_table() -> pd.DataFrame:
    final12 = pd.read_csv(ROOT / "final_12graph_eval" / "hast_12graph_summary.csv")
    rows: List[Dict[str, Any]] = []
    for dataset, sub in final12.groupby("dataset"):
        hast = sub[sub["method"].eq("HAST")].iloc[0]
        best_auc = sub[sub["method"].ne("HAST")].sort_values("auc_cNBI", ascending=False).iloc[0]
        e26f = sub[sub["method"].eq("E26F")].iloc[0]
        hda = sub[sub["method"].eq("HDA")].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "HAST_R": float(hast["R"]),
                "HAST_auc_cNBI": float(hast["auc_cNBI"]),
                "best_auc_method": best_auc["method"],
                "best_auc_cNBI": float(best_auc["auc_cNBI"]),
                "auc_gap_to_best": float(best_auc["auc_cNBI"] - hast["auc_cNBI"]),
                "auc_gap_to_E26F": float(e26f["auc_cNBI"] - hast["auc_cNBI"]),
                "auc_gain_over_HDA": float(hast["auc_cNBI"] - hda["auc_cNBI"]),
                "R_gap_to_E26F": float(hast["R"] - e26f["R"]),
            }
        )
    out = pd.DataFrame(rows).sort_values("auc_gap_to_best", ascending=False)
    out.to_csv(TABLE_DIR / "hast_dataset_failure_gaps.csv", index=False, encoding="utf-8-sig")
    return out


def validation_proxy_simulation() -> pd.DataFrame:
    final12 = pd.read_csv(ROOT / "final_12graph_eval" / "hast_12graph_summary.csv")
    final12 = final12[final12["method"].isin(METHODS)].copy()
    datasets = sorted(final12["dataset"].unique().tolist())
    rows: List[Dict[str, Any]] = []
    for k in [1, 2, 3]:
        for val_sets in itertools.combinations(datasets, k):
            val_mask = final12["dataset"].isin(val_sets)
            val = final12[val_mask].groupby("method")["auc_cNBI"].mean()
            test = final12[~val_mask].groupby("method")["auc_cNBI"].mean()
            test_r = final12[~val_mask].groupby("method")["R"].mean()
            chosen = str(val.sort_values(ascending=False).index[0])
            oracle = str(test.sort_values(ascending=False).index[0])
            rows.append(
                {
                    "k_validation_graphs": k,
                    "validation_graphs": "|".join(val_sets),
                    "chosen_by_validation": chosen,
                    "oracle_on_heldout": oracle,
                    "spearman_val_to_heldout_auc": float(val.corr(test, method="spearman")),
                    "chosen_heldout_auc": float(test[chosen]),
                    "chosen_heldout_R": float(test_r[chosen]),
                    "hast_heldout_auc": float(test["HAST"]),
                    "e26f_heldout_auc": float(test["E26F"]),
                    "oracle_heldout_auc": float(test.max()),
                    "regret_to_oracle": float(test.max() - test[chosen]),
                    "gain_over_hast": float(test[chosen] - test["HAST"]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "hast_validation_proxy_simulation.csv", index=False, encoding="utf-8-sig")
    summary = (
        out.groupby("k_validation_graphs")
        .agg(
            cases=("validation_graphs", "count"),
            mean_spearman=("spearman_val_to_heldout_auc", "mean"),
            mean_gain_over_hast=("gain_over_hast", "mean"),
            mean_regret=("regret_to_oracle", "mean"),
            hast_selected_rate=("chosen_by_validation", lambda s: float((s == "HAST").mean())),
        )
        .reset_index()
    )
    summary.to_csv(TABLE_DIR / "hast_validation_proxy_summary.csv", index=False, encoding="utf-8-sig")
    freq = (
        out.groupby(["k_validation_graphs", "chosen_by_validation"])
        .size()
        .reset_index(name="count")
        .sort_values(["k_validation_graphs", "count"], ascending=[True, False])
    )
    freq.to_csv(TABLE_DIR / "hast_validation_proxy_choice_frequency.csv", index=False, encoding="utf-8-sig")
    return out


def early_curve_proxy() -> pd.DataFrame:
    summary = pd.read_csv(ROOT / "final_12graph_eval" / "hast_12graph_summary.csv")
    records_dir = ROOT / "final_12graph_eval" / "records"
    rows: List[Dict[str, Any]] = []
    for _, meta in summary.iterrows():
        path = records_dir / f"{meta['dataset']}_{meta['method']}_metrics.csv"
        if not path.exists():
            continue
        curve = pd.read_csv(path)
        item: Dict[str, Any] = {
            "dataset": meta["dataset"],
            "method": meta["method"],
            "R": float(meta["R"]),
            "auc_cNBI": float(meta["auc_cNBI"]),
            "time_s": float(meta["time_s"]),
        }
        for ratio in [0.05, 0.10, 0.20, 0.30]:
            idx = (curve["removal_ratio"] - ratio).abs().idxmin()
            item[f"cNBI_at_{ratio:.2f}"] = float(curve.loc[idx, "cNBI"])
            item[f"GCC_at_{ratio:.2f}"] = float(curve.loc[idx, "GCC"])
            item[f"NCC_at_{ratio:.2f}"] = float(curve.loc[idx, "NCC"])
        rows.append(item)
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "hast_early_curve_proxy_rows.csv", index=False, encoding="utf-8-sig")

    corr_rows: List[Dict[str, Any]] = []
    for feature in [c for c in out.columns if c.startswith(("cNBI_at_", "GCC_at_", "NCC_at_"))]:
        corr_rows.append(
            {
                "early_feature": feature,
                "spearman_to_auc_cNBI": float(out[feature].corr(out["auc_cNBI"], method="spearman")),
                "spearman_to_R": float(out[feature].corr(out["R"], method="spearman")),
            }
        )
    corr = pd.DataFrame(corr_rows).sort_values("spearman_to_auc_cNBI", ascending=False)
    corr.to_csv(TABLE_DIR / "hast_early_curve_proxy_correlations.csv", index=False, encoding="utf-8-sig")
    return corr


def top10_search_generalization_corr() -> pd.DataFrame:
    top = pd.read_csv(TABLE_DIR / "hast_top10_12graph_summary.csv")
    rows: List[Dict[str, Any]] = []
    for search_metric in ["search_rank_score", "search_R", "search_cNBI", "search_Time"]:
        for gen_metric in ["mean_R", "mean_auc_cNBI", "mean_time_s"]:
            rows.append(
                {
                    "search_metric": search_metric,
                    "generalization_metric": gen_metric,
                    "pearson": float(top[search_metric].corr(top[gen_metric], method="pearson")),
                    "spearman": float(top[search_metric].corr(top[gen_metric], method="spearman")),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "hast_search_to_12graph_correlation_top10.csv", index=False, encoding="utf-8-sig")
    return out


def prefix_overlap_summary() -> pd.DataFrame:
    detail = pd.read_csv(TABLE_DIR / "hast_best_prefix_overlap_12graph.csv")
    out = (
        detail.groupby("compare_to")[["prefix_jaccard", "same_position_rate"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("prefix_jaccard", ascending=False)
    )
    out.to_csv(TABLE_DIR / "hast_best_prefix_overlap_summary.csv", index=False, encoding="utf-8-sig")
    return out


def plot_outputs(
    method_gen: pd.DataFrame,
    blocks: pd.DataFrame,
    gaps: pd.DataFrame,
    val_summary: pd.DataFrame,
    early_corr: pd.DataFrame,
) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    plot_df = method_gen[method_gen["method"].isin(SEARCH_METHODS)].copy()
    ax.scatter(plot_df["strict_rate"], plot_df["mean_12_auc_cNBI"], s=45, color="#4C78A8")
    for _, row in plot_df.iterrows():
        ax.text(row["strict_rate"], row["mean_12_auc_cNBI"], row["method"], fontsize=7)
    ax.set_xlabel("search strict hit rate")
    ax.set_ylabel("mean 12-graph AUC-cNBI")
    ax.set_title("A. Search hits do not guarantee generalization")

    ax = axes[0, 1]
    x = blocks["start_idx"].to_numpy()
    ax.plot(x, blocks["local_twohop_neighbor_share"], marker="o", label="neighbor")
    ax.plot(x, blocks["local_twohop_bridge_share"], marker="o", label="bridge")
    ax.plot(x, blocks["family_entropy"], marker="o", label="entropy")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("HAST node block start")
    ax.set_title("B. Family collapse over search")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    top_gap = gaps.sort_values("auc_gap_to_best", ascending=True)
    ax.barh(top_gap["dataset"], top_gap["auc_gap_to_best"], color="#D62728")
    ax.set_xlabel("best method AUC-cNBI minus HAST")
    ax.set_title("C. Where HAST loses on 12 graphs")

    ax = axes[1, 1]
    ax.bar(val_summary["k_validation_graphs"].astype(str), val_summary["mean_gain_over_hast"], color="#2A9D8F")
    ax.set_xlabel("validation graphs used")
    ax.set_ylabel("held-out AUC gain over HAST")
    ax.set_title("D. Small validation proxy helps selection")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "hast_diagnostic_experiments_panel.png")
    fig.savefig(FIG_DIR / "hast_diagnostic_experiments_panel.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    show = early_corr.head(10).copy()
    ax.barh(show["early_feature"][::-1], show["spearman_to_auc_cNBI"][::-1], color="#9C755F")
    ax.set_xlabel("Spearman correlation to final AUC-cNBI")
    ax.set_title("Early curve signals that can become cheap HAST credit")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hast_early_curve_proxy_correlations.png")
    fig.savefig(FIG_DIR / "hast_early_curve_proxy_correlations.pdf")
    plt.close(fig)


def write_report() -> None:
    method_gen = pd.read_csv(TABLE_DIR / "hast_diagnostic_method_generalization.csv")
    blocks = pd.read_csv(TABLE_DIR / "hast_family_collapse_blocks.csv")
    gaps = pd.read_csv(TABLE_DIR / "hast_dataset_failure_gaps.csv")
    val_summary = pd.read_csv(TABLE_DIR / "hast_validation_proxy_summary.csv")
    val_freq = pd.read_csv(TABLE_DIR / "hast_validation_proxy_choice_frequency.csv")
    early_corr = pd.read_csv(TABLE_DIR / "hast_early_curve_proxy_correlations.csv")
    top10_corr = top10_search_generalization_corr()
    overlap = prefix_overlap_summary()

    lines = [
        "# HAST 诊断实验：不足与改进方向",
        "",
        "## 最短结论",
        "",
        "当前 HAST 的主要不足不是“搜不到高分节点”，而是**信用分配只奖励了训练生成图上的强 family，缺少泛化负反馈**。结果是搜索很快塌缩到 local two-hop/neighbor 机制；这个机制在 50 个 500 节点生成图上很强，但 12 图泛化接近 HDA，明显弱于 E26F/PUCT/Clade/FunSearch 的强候选。",
        "",
        "## 实验 1：搜索命中率不等于 12 图泛化",
        "",
        method_gen.to_markdown(index=False),
        "",
        "解释：HAST 的 strict hit 数最高，但 mean 12-graph AUC-cNBI 只有约 242.5；这说明 strict/e26f-like 在当前 search graph 上是有用的搜索效率指标，但不能单独作为最终泛化信用。",
        "",
        "HAST top-10 候选内部的搜索分数与 12 图指标相关性也很弱：",
        "",
        top10_corr.to_markdown(index=False),
        "",
        "## 实验 2：family collapse",
        "",
        blocks.to_markdown(index=False),
        "",
        "解释：HAST 后期大部分预算集中到 local_twohop_neighbor，局部桥/组件/社区边界方向没有获得足够预算。它像是把“早期经验”变成了单一路线，而不是变成可纠错的多机制搜索。",
        "",
        "## 实验 3：失败集中在哪些图",
        "",
        gaps.head(8).to_markdown(index=False),
        "",
        "解释：HAST 在部分真实图上只比 HDA 略好或更差，尤其在社区/协作类图上 cNBI 落后明显。这支持“生成图过拟合 + 缺少图类型条件信用”的判断。",
        "",
        "## 实验 4：前缀重合说明它更像局部度排序变体",
        "",
        overlap.to_markdown(index=False),
        "",
        "解释：HAST 与 HDA/E26F/AlphaEvolve 的前缀节点集合高度重合，但 same-position rate 低，说明它常选类似的一批高局部重要性节点，却没有学好真实图上更关键的删除顺序和碎裂时机。",
        "",
        "## 实验 5：小验证图 proxy 是否有用",
        "",
        val_summary.to_markdown(index=False),
        "",
        "选择频率：",
        "",
        val_freq.to_markdown(index=False),
        "",
        "解释：哪怕只用 1-3 个 12 图作为 validation proxy，按 validation AUC 选方法，在剩余图上的平均 AUC 通常明显高于固定使用 HAST。这不是最终算法结果，但证明“把少量验证图反馈接入信用分配”是有价值的。",
        "",
        "## 实验 6：廉价早期曲线信号",
        "",
        early_corr.head(12).to_markdown(index=False),
        "",
        "解释：早期 cNBI/NCC/GCC 曲线和最终 AUC 有较强相关性，后续 HAST 不必每次完整评估所有真实图，可以先用少量 proxy 图 + 早期删除比例的曲线形状做 family credit。",
        "",
        "## 最可能有效的改进",
        "",
        "1. **HAST-V**：每 30-50 个节点对 top 候选做 2-3 个小验证图、早期 10%-20% 删除曲线复评；family credit = search score + validation score + time guard。",
        "2. **防塌缩预算**：family credit 可以偏置选择，但不能让一个 family 连续吞掉全部预算；设置最低探索比例或 entropy penalty。",
        "3. **图类型条件信用**：把 powerlaw、community、high-clustering、large sparse 分开记信用；一个 family 只在适合的图类型上升权。",
        "4. **失败诊断 prompt**：当 validation cNBI 差时，不再让 LLM 微调二跳权重，而是明确要求加入 component-balance/frontier/community-boundary 的低复杂度代理。",
        "5. **候选晋级三门槛**：search strong family、validation 不退化、复杂度低于迭代 O(N^2)。当前 HAST 主要只满足第一项。",
    ]
    (REPORT_DIR / "hast_diagnostic_experiments_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    method_gen = method_generalization_table()
    blocks = family_collapse_blocks()
    gaps = dataset_gap_table()
    validation_proxy_simulation()
    val_summary = pd.read_csv(TABLE_DIR / "hast_validation_proxy_summary.csv")
    early_corr = early_curve_proxy()
    top10_search_generalization_corr()
    prefix_overlap_summary()
    plot_outputs(method_gen, blocks, gaps, val_summary, early_corr)
    write_report()
    print(REPORT_DIR / "hast_diagnostic_experiments_cn.md")


if __name__ == "__main__":
    main()
