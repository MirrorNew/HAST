# -*- coding: utf-8 -*-
"""Failure analysis for HAST generalization.

The goal is not to defend HAST, but to identify why the current search becomes
excellent on the 50 generated search graphs while generalizing poorly to the
12-graph benchmark.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
SOURCE_ROOT = WORKSPACE / "research" / "tree_search_ablation_20260520"
SOURCE_12 = SOURCE_ROOT / "src" / "evaluate_final_12graphs.py"
SEARCH_SRC = SOURCE_ROOT / "src" / "ablation_search.py"

TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"
RUN_DIR = ROOT / "runs" / "HAST"

GENERIC_METHODS = ["PUCT", "MCTS-AHD-like", "Clade-AHD-like", "FunSearch-like", "AlphaEvolve-like"]
REFERENCE_METHODS = ["E26F", "HDA"]
COMPARE_METHODS = ["HAST"] + GENERIC_METHODS + REFERENCE_METHODS


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


E12 = load_module(SOURCE_12, "hast_failure_eval12")
SEARCH = load_module(SEARCH_SRC, "hast_failure_search")


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


def bool_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_hast_records() -> pd.DataFrame:
    df = pd.read_csv(RUN_DIR / "search_records.csv")
    df["valid"] = bool_col(df["valid"])
    for col in ["idx", "R", "cNBI", "Time", "rank_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["strict_e26f_like"] = bool_col(df["strict_e26f_like"])
    df["loose_e26f_like"] = bool_col(df["loose_e26f_like"])
    return df


def select_representatives(records: pd.DataFrame) -> pd.DataFrame:
    valid = records[records["valid"] & records["candidate_file"].notna()].copy()
    reps: List[Tuple[str, pd.Series]] = []

    def add(label: str, sub: pd.DataFrame, sort_col: str, ascending: bool) -> None:
        if sub.empty:
            return
        row = sub.sort_values(sort_col, ascending=ascending).iloc[0]
        reps.append((label, row))

    add("best_rank", valid, "rank_score", False)
    add("best_search_R", valid, "R", True)
    add("best_search_cNBI", valid, "cNBI", False)
    strict = valid[valid["strict_e26f_like"]]
    add("fastest_strict", strict, "Time", True)
    add("early_strict", strict, "idx", True)
    for family in ["local_twohop_neighbor", "local_twohop_bridge", "simple_degree"]:
        add(f"best_{family}", valid[valid["actual_family"].eq(family)], "rank_score", False)

    rows = []
    seen = set()
    for label, row in reps:
        idx = int(row["idx"])
        if idx in seen:
            continue
        seen.add(idx)
        item = row.to_dict()
        item["representative"] = label
        rows.append(item)
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "hast_representative_candidates.csv", index=False, encoding="utf-8-sig")
    return out


def evaluate_representatives(reps: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, rec in reps.iterrows():
        code = Path(str(rec["candidate_file"])).read_text(encoding="utf-8")
        fn = SEARCH.compile_degree_order(code)
        for dataset in E12.EVAL.DATASETS:
            graph = E12.EVAL.read_graph(dataset)
            rate = E12.EVAL.DATASET_RATES[dataset]
            t0 = time.perf_counter()
            order = list(fn(graph.copy()))
            elapsed = time.perf_counter() - t0
            metrics = E12.EVAL.compute_metrics(graph, order, rate=rate, method_time=elapsed)
            if metrics.empty:
                continue
            x = metrics["removal_ratio"].to_numpy(dtype=float)
            rows.append(
                {
                    "representative": rec["representative"],
                    "idx": int(rec["idx"]),
                    "actual_family": rec["actual_family"],
                    "dataset": dataset,
                    "search_rank_score": float(rec["rank_score"]),
                    "search_R": float(rec["R"]),
                    "search_cNBI": float(rec["cNBI"]),
                    "search_Time": float(rec["Time"]),
                    "R": E12.EVAL.auc_mean(x, metrics["GCC"].to_numpy(dtype=float)),
                    "auc_cNBI": E12.EVAL.auc_mean(x, metrics["cNBI"].to_numpy(dtype=float)),
                    "time_s": elapsed,
                }
            )
        print(f"[rep] {rec['representative']} idx={int(rec['idx'])}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "hast_representative_12graph_detail.csv", index=False, encoding="utf-8-sig")
    mean = (
        out.groupby(["representative", "idx", "actual_family", "search_rank_score", "search_R", "search_cNBI", "search_Time"])[
            ["R", "auc_cNBI", "time_s"]
        ]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["R", "time_s"])
    )
    mean.to_csv(TABLE_DIR / "hast_representative_12graph_mean.csv", index=False, encoding="utf-8-sig")
    return mean


def dataset_gap_analysis() -> pd.DataFrame:
    summary = pd.read_csv(ROOT / "final_12graph_eval" / "hast_12graph_summary.csv")
    rows: List[Dict[str, Any]] = []
    for dataset, sub in summary.groupby("dataset"):
        hast = sub[sub["method"].eq("HAST")].iloc[0]
        best_auc = sub[sub["method"].ne("HAST")].sort_values("auc_cNBI", ascending=False).iloc[0]
        best_r = sub[sub["method"].ne("HAST")].sort_values("R", ascending=True).iloc[0]
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
                "best_R_method": best_r["method"],
                "best_R": float(best_r["R"]),
                "R_gap_to_best": float(hast["R"] - best_r["R"]),
                "auc_gap_to_E26F": float(e26f["auc_cNBI"] - hast["auc_cNBI"]),
                "R_gap_to_E26F": float(hast["R"] - e26f["R"]),
                "auc_gain_over_HDA": float(hast["auc_cNBI"] - hda["auc_cNBI"]),
                "R_gain_over_HDA": float(hda["R"] - hast["R"]),
            }
        )
    out = pd.DataFrame(rows).sort_values("auc_gap_to_best", ascending=False)
    out.to_csv(TABLE_DIR / "hast_dataset_failure_gaps.csv", index=False, encoding="utf-8-sig")
    return out


def graph_stats_and_correlations(gaps: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset in E12.EVAL.DATASETS:
        graph = E12.EVAL.read_graph(dataset)
        deg = np.array([d for _, d in graph.degree()], dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "avg_degree": float(deg.mean()) if len(deg) else 0.0,
                "degree_cv": float(deg.std() / max(1e-9, deg.mean())) if len(deg) else 0.0,
                "max_degree": float(deg.max()) if len(deg) else 0.0,
                "density": float(nx.density(graph)),
                "transitivity": float(nx.transitivity(graph)),
                "components": nx.number_connected_components(graph),
                "gcc_ratio": len(max(nx.connected_components(graph), key=len)) / max(1, graph.number_of_nodes()),
            }
        )
    stats = pd.DataFrame(rows)
    merged = stats.merge(gaps, on="dataset", how="left")
    merged.to_csv(TABLE_DIR / "hast_graph_stats_failure_correlation.csv", index=False, encoding="utf-8-sig")
    corr_cols = ["nodes", "edges", "avg_degree", "degree_cv", "max_degree", "density", "transitivity", "components", "gcc_ratio"]
    corr_rows = []
    for col in corr_cols:
        corr_rows.append(
            {
                "graph_stat": col,
                "corr_auc_gap_to_best": merged[col].corr(merged["auc_gap_to_best"], method="spearman"),
                "corr_R_gap_to_best": merged[col].corr(merged["R_gap_to_best"], method="spearman"),
            }
        )
    pd.DataFrame(corr_rows).to_csv(TABLE_DIR / "hast_graph_stat_gap_correlations.csv", index=False, encoding="utf-8-sig")
    return merged


def family_collapse_analysis(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, sub in records.groupby("actual_family"):
        rows.append(
            {
                "actual_family": family,
                "count": len(sub),
                "valid": int(sub["valid"].sum()),
                "strict": int(sub["strict_e26f_like"].sum()),
                "loose": int(sub["loose_e26f_like"].sum()),
                "strict_rate": float(sub["strict_e26f_like"].mean()),
                "mean_rank_score": float(sub["rank_score"].replace(-1, np.nan).mean()),
                "mean_Time": float(sub["Time"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values("count", ascending=False)
    total = out["count"].sum()
    probs = out["count"] / max(1, total)
    entropy = -float((probs * np.log(probs + 1e-12)).sum())
    out["family_entropy"] = entropy
    out["family_entropy_normalized"] = entropy / math.log(max(2, len(out)))
    out.to_csv(TABLE_DIR / "hast_family_collapse.csv", index=False, encoding="utf-8-sig")
    return out


def top10_correlation() -> pd.DataFrame:
    top = pd.read_csv(TABLE_DIR / "hast_top10_12graph_summary.csv")
    pairs = []
    for x in ["search_rank_score", "search_R", "search_cNBI", "search_Time"]:
        for y in ["mean_R", "mean_auc_cNBI", "mean_time_s"]:
            pairs.append(
                {
                    "search_metric": x,
                    "generalization_metric": y,
                    "pearson": top[x].corr(top[y], method="pearson"),
                    "spearman": top[x].corr(top[y], method="spearman"),
                }
            )
    out = pd.DataFrame(pairs)
    out.to_csv(TABLE_DIR / "hast_search_to_12graph_correlation_top10.csv", index=False, encoding="utf-8-sig")
    return out


def curve_probe() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset in E12.EVAL.DATASETS:
        for method in ["HAST", "HDA", "E26F", "PUCT", "Clade-AHD-like", "FunSearch-like"]:
            path = ROOT / "final_12graph_eval" / "records" / f"{dataset}_{method}_metrics.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            row: Dict[str, Any] = {"dataset": dataset, "method": method}
            for ratio in [0.05, 0.10, 0.20, 0.30]:
                idx = (df["removal_ratio"] - ratio).abs().idxmin()
                row[f"cNBI_at_{ratio:.2f}"] = float(df.loc[idx, "cNBI"])
                row[f"GCC_at_{ratio:.2f}"] = float(df.loc[idx, "GCC"])
                row[f"NCC_at_{ratio:.2f}"] = float(df.loc[idx, "NCC"])
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "hast_curve_probe.csv", index=False, encoding="utf-8-sig")
    return out


def prefix_overlap_summary() -> pd.DataFrame:
    overlap = pd.read_csv(TABLE_DIR / "hast_best_prefix_overlap_12graph.csv")
    mean = (
        overlap.groupby("compare_to")[["prefix_jaccard", "same_position_rate"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("prefix_jaccard", ascending=False)
    )
    mean.to_csv(TABLE_DIR / "hast_best_prefix_overlap_summary.csv", index=False, encoding="utf-8-sig")
    return mean


def plot_panel(gaps: pd.DataFrame, fam: pd.DataFrame, overlap: pd.DataFrame) -> None:
    setup_style()
    top = pd.read_csv(TABLE_DIR / "hast_top10_12graph_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    g = gaps.sort_values("auc_gap_to_best", ascending=True)
    ax.barh(g["dataset"], g["auc_gap_to_best"], color="#D62728")
    ax.set_title("A. HAST AUC-cNBI gap to best method")
    ax.set_xlabel("Best method AUC-cNBI minus HAST")

    ax = axes[0, 1]
    ax.bar(fam["actual_family"], fam["count"], color="#4C78A8", label="count")
    ax2 = ax.twinx()
    ax2.plot(fam["actual_family"], fam["strict_rate"], color="#D62728", marker="o", label="strict rate")
    ax.set_title("B. Family collapse in HAST")
    ax.set_ylabel("count")
    ax2.set_ylabel("strict rate")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[1, 0]
    ax.scatter(top["search_rank_score"], top["mean_auc_cNBI"], color="#2A9D8F")
    for _, row in top.iterrows():
        ax.text(row["search_rank_score"], row["mean_auc_cNBI"], str(int(row["idx"])), fontsize=7)
    ax.set_title("C. Search score does not predict 12-graph AUC")
    ax.set_xlabel("search rank_score")
    ax.set_ylabel("12-graph AUC-cNBI")

    ax = axes[1, 1]
    ax.bar(overlap["compare_to"], overlap["prefix_jaccard"], color="#9C755F")
    ax.set_title("D. HAST prefix node-set overlap")
    ax.set_ylabel("Jaccard with HAST removal prefix")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hast_failure_analysis_panel.png")
    fig.savefig(FIG_DIR / "hast_failure_analysis_panel.pdf")
    plt.close(fig)


def plot_failure_cases() -> None:
    setup_style()
    gaps = pd.read_csv(TABLE_DIR / "hast_dataset_failure_gaps.csv")
    cases = gaps.sort_values("auc_gap_to_best", ascending=False).head(4)["dataset"].tolist()
    methods = ["HAST", "HDA", "E26F", "PUCT", "Clade-AHD-like", "FunSearch-like"]
    colors = {
        "HAST": "#D62728",
        "HDA": "#E377C2",
        "E26F": "#111111",
        "PUCT": "#4C78A8",
        "Clade-AHD-like": "#F28E2B",
        "FunSearch-like": "#B07AA1",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, dataset in zip(axes.ravel(), cases):
        for method in methods:
            path = ROOT / "final_12graph_eval" / "records" / f"{dataset}_{method}_metrics.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            ax.plot(df["removal_ratio"], df["cNBI"], label=method, color=colors.get(method), lw=2.2 if method == "HAST" else 1.2)
        ax.set_title(dataset)
        ax.set_xlabel("removal ratio")
        ax.set_ylabel("cNBI")
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle("Worst HAST generalization cases: cNBI curves", y=0.99)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hast_failure_case_cnbi_curves.png")
    fig.savefig(FIG_DIR / "hast_failure_case_cnbi_curves.pdf")
    plt.close(fig)


def write_report(
    gaps: pd.DataFrame,
    stats: pd.DataFrame,
    fam: pd.DataFrame,
    corr: pd.DataFrame,
    overlap: pd.DataFrame,
    reps_mean: pd.DataFrame | None,
) -> None:
    worst = gaps.sort_values("auc_gap_to_best", ascending=False).head(5)
    stat_corr = pd.read_csv(TABLE_DIR / "hast_graph_stat_gap_correlations.csv").sort_values(
        "corr_auc_gap_to_best", ascending=False
    )
    lines = [
        "# HAST 泛化失败分析",
        "",
        "## 结论先行",
        "",
        "当前 HAST 的问题不是不能搜索，而是**搜索目标太窄**：它非常快地把搜索集中到 search graphs 上有效的 local two-hop/neighbor family，但这个 family 在真实图上没有稳定泛化。",
        "",
        "## 证据 1：失败不是单个 best 选错",
        "",
        "HAST 搜索 top-10 候选的 12 图结果都很接近，mean R 约 0.437-0.440，AUC-cNBI 约 242-243。因此不是只选错了一个 best，而是高分 family 整体泛化弱。",
        "",
        corr.to_markdown(index=False),
        "",
        "## 证据 2：搜索塌缩到单一 family",
        "",
        fam.to_markdown(index=False),
        "",
        "HAST 300 个候选里绝大多数落在 local_twohop_neighbor / local_twohop_bridge。这个 family 的 strict hit 很多，但多样性不足，后期 credit 没有引入泛化负反馈。",
        "",
        "## 证据 3：哪些真实图失败最明显",
        "",
        worst.to_markdown(index=False),
        "",
        "## 证据 4：HAST 在真实图上更像局部度排序，而不是新碎裂机制",
        "",
        overlap.to_markdown(index=False),
        "",
        "HAST 与 HDA/E26F/AlphaEvolve 的前缀节点集合重合度很高，但 same-position rate 很低。这说明它往往选的是类似的一批高度/局部核心节点，但排序和碎裂时机不够好；在 community/高聚类真实图上，cNBI 很容易掉下去。",
        "",
        "## 图结构相关线索",
        "",
        stat_corr.head(8).to_markdown(index=False),
        "",
        "这些相关性只来自 12 个图，不能当最终结论，但可以提示下一步要把图类型/泛化验证纳入 credit。",
        "",
        "## 可能改进方向",
        "",
        "1. **HAST-V：加入小型验证图 credit**。每 30-50 个节点，把 top candidates 在 3-4 个 proxy validation graphs 上快速复评，credit 同时看 search score 和 validation score。",
        "2. **防 family collapse**。给 family 选择加入熵/覆盖率约束，限制 local_twohop_neighbor 连续占用预算；保留 bridge、component、core、diverse-code family 的最低探索比例。",
        "3. **把泛化失败变成诊断反馈**。如果 validation 图上 cNBI 低，prompt 不再继续调二跳权重，而要求加入 component-balance/frontier/community-boundary 的廉价代理。",
        "4. **promotion gate 改成三门槛**：搜索图 strong density、验证图不退化、复杂度不超标。当前 HAST 只满足第一项。",
        "5. **候选选择不要只按 search rank_score**。本轮 top-10 表明 search score 对 12 图泛化预测很弱，需要加入行为多样性、图类型覆盖和早期 cNBI 曲线特征。",
    ]
    if reps_mean is not None:
        lines += ["", "## Representative candidates", "", reps_mean.to_markdown(index=False)]
    (REPORT_DIR / "hast_failure_analysis_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    records = load_hast_records()
    reps = select_representatives(records)
    reps_mean = evaluate_representatives(reps)
    gaps = dataset_gap_analysis()
    stats = graph_stats_and_correlations(gaps)
    fam = family_collapse_analysis(records)
    corr = top10_correlation()
    curve_probe()
    overlap = prefix_overlap_summary()
    plot_panel(gaps, fam, overlap)
    plot_failure_cases()
    write_report(gaps, stats, fam, corr, overlap, reps_mean)
    print(REPORT_DIR / "hast_failure_analysis_cn.md")


if __name__ == "__main__":
    main()
