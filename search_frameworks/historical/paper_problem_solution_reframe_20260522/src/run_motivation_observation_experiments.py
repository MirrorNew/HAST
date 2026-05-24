#!/usr/bin/env python3
"""Build motivation-observation evidence tables and figures for HAST.

The script reuses cached evaluation/search logs and adds lightweight static
diagnostics over generated candidate code. It does not rerun graph evaluation.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT.parents[0]
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"
HAST_TABLE_DIR = RESEARCH / "hast_experiment_20260521" / "tables"


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


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.png")
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    plt.close(fig)


def safe_float(value: object) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return math.nan
    return v


def feature_flags(code: str) -> dict[str, int]:
    patterns = {
        "twohop_or_nested_neighbor_scan": [
            r"for\s+\w+\s+in\s+H\.neighbors\(\w+\)",
            r"two[_-]?hop",
            r"uniq2",
            r"unique2",
            r"seen2",
        ],
        "frontier_weak_boundary_terms": [
            r"frontier",
            r"weak[_-]?tie",
            r"boundary",
            r"bridge",
            r"exposure",
            r"support_gap",
        ],
        "redundancy_terms": [r"redundancy", r"shared"],
        "phase_weights": [r"progress", r"if\s+progress", r"early"],
        "component_recompute": [
            r"connected_components",
            r"number_connected_components",
            r"\.subgraph\(",
            r"component",
        ],
        "local_heap_update": [r"heapq", r"affected\.update", r"stamp"],
        "global_node_sweep": [r"for\s+\w+\s+in\s+(?:list\()?H\.nodes\("],
    }
    out: dict[str, int] = {}
    for name, pats in patterns.items():
        out[name] = int(any(re.search(pat, code, flags=re.IGNORECASE) for pat in pats))
    out["neighbor_scan_count"] = len(re.findall(r"H\.neighbors\(", code))
    out["node_sweep_count"] = len(re.findall(r"H\.nodes\(", code))
    out["connected_component_count"] = len(re.findall(r"connected_components|number_connected_components", code))
    return out


def add_code_features(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in records.iterrows():
        path_text = str(row.get("candidate_file", ""))
        code_path = Path(path_text) if path_text and path_text != "nan" else None
        code = ""
        exists = False
        if code_path is not None and code_path.exists():
            exists = True
            code = code_path.read_text(encoding="utf-8", errors="replace")
        flags = feature_flags(code)
        record = row.to_dict()
        record["code_file_exists"] = exists
        record.update(flags)
        rows.append(record)
    return pd.DataFrame(rows)


def time_bucket(value: float) -> str:
    if value <= 0.8:
        return "fast<=0.8"
    if value <= 1.2:
        return "ok<=1.2"
    if value <= 1.8:
        return "slow<=1.8"
    return "too_slow"


def observation_1() -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = read_csv(TABLE_DIR / "aaai_followup_cnbi_nonredundancy_correlations.csv")
    same = read_csv(TABLE_DIR / "aaai_followup_same_gcc_cnbi_cases.csv")

    same = same.sort_values("delta_cNBI_a_minus_b", ascending=False)
    top_same = same.head(8).copy()
    top_same["case"] = (
        top_same["dataset"].astype(str)
        + ": "
        + top_same["method_a"].astype(str)
        + " vs "
        + top_same["method_b"].astype(str)
    )
    out_same = top_same[
        [
            "dataset",
            "method_a",
            "method_b",
            "gcc_abs_diff",
            "cNBI_a",
            "cNBI_b",
            "delta_cNBI_a_minus_b",
            "top5_mass_a",
            "top5_mass_b",
            "pairdisc_a",
            "pairdisc_b",
        ]
    ].copy()
    write_csv(corr, "motivation_obs1_cnbi_r_correlation.csv")
    write_csv(out_same, "motivation_obs1_same_gcc_top_cases.csv")

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    labels = top_same["case"].tolist()
    y = list(range(len(labels)))
    ax.barh(y, top_same["delta_cNBI_a_minus_b"], color="#0072B2", alpha=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("cNBI gap under nearly identical GCC/R")
    ax.set_title("Observation 1: similar GCC/R can hide different residual fragmentation")
    save_fig(fig, "fig21_motivation_obs1_same_gcc_cnbi_gap")
    return corr, out_same


def observation_2_and_3() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fac_records = read_csv(HAST_TABLE_DIR / "hast_fac_online_search_records.csv")
    fac_records = fac_records[fac_records["stage"].astype(str) != "root"].copy()
    fac_records["proxy_time_s"] = fac_records["proxy_time_s"].map(safe_float)
    fac_records["fac_score"] = fac_records["fac_score"].map(safe_float)
    fac_records["fac_auc_adv"] = fac_records["fac_auc_adv"].map(safe_float)
    fac_records["Time"] = fac_records["Time"].map(safe_float)
    fac_records["time_bucket"] = fac_records["proxy_time_s"].map(time_bucket)
    fac_features = add_code_features(fac_records)

    feature_cols = [
        "twohop_or_nested_neighbor_scan",
        "frontier_weak_boundary_terms",
        "redundancy_terms",
        "phase_weights",
        "component_recompute",
        "local_heap_update",
        "global_node_sweep",
    ]
    bucket_rows = []
    for bucket, group in fac_features.groupby("time_bucket", sort=False):
        row: dict[str, object] = {
            "time_bucket": bucket,
            "n": len(group),
            "mean_fac_score": group["fac_score"].mean(),
            "mean_fac_auc_adv": group["fac_auc_adv"].mean(),
            "mean_proxy_time_s": group["proxy_time_s"].mean(),
            "mean_runtime_s": group["Time"].mean(),
            "mean_neighbor_scan_count": group["neighbor_scan_count"].mean(),
        }
        for col in feature_cols:
            row[f"rate_{col}"] = group[col].mean()
        bucket_rows.append(row)
    bucket_summary = pd.DataFrame(bucket_rows)
    bucket_summary["_order"] = bucket_summary["time_bucket"].map(
        {"fast<=0.8": 0, "ok<=1.2": 1, "slow<=1.8": 2, "too_slow": 3}
    )
    bucket_summary = bucket_summary.sort_values("_order").drop(columns=["_order"])
    write_csv(bucket_summary, "motivation_obs2_fac_code_feature_by_time_bucket.csv")

    old_top = fac_features.sort_values("fac_score", ascending=False).head(12).copy()
    old_top = old_top[
        [
            "idx",
            "target_family",
            "fac_score",
            "fac_auc_adv",
            "proxy_auc_cNBI",
            "proxy_time_s",
            "Time",
            "neighbor_scan_count",
            *feature_cols,
            "candidate_file",
        ]
    ]
    write_csv(old_top, "motivation_obs2_top_fac_candidates_code_features.csv")

    fact_top = read_csv(HAST_TABLE_DIR / "hast_fac_time_aware_top.csv").head(12).copy()
    fact_top["proxy_time_s"] = fact_top["proxy_time_s"].map(safe_float)
    fact_top["Time"] = fact_top["Time"].map(safe_float)
    fact_top_features = add_code_features(fact_top)
    fact_top_features = fact_top_features[
        [
            "idx",
            "target_family",
            "fac_t_score",
            "fac_score",
            "fac_auc_adv",
            "proxy_auc_cNBI",
            "proxy_time_s",
            "Time",
            "neighbor_scan_count",
            *feature_cols,
            "candidate_file",
        ]
    ]
    write_csv(fact_top_features, "motivation_obs3_top_fact_candidates_code_features.csv")

    ablation = read_csv(TABLE_DIR / "aaai_followup_fac_ablation_summary.csv")
    stage_map = {
        "HAST": "Initial automatic search",
        "HAST-FAC-T online #24": "Relative + time credit",
        "FAST21-cap24": "Full HAST-Q",
        "BT-n16-t8-u24": "Full HAST-S",
    }
    fac_mean = read_csv(HAST_TABLE_DIR / "hast_fac_online_full12_mean.csv")
    rel_only = fac_mean.sort_values("auc_cNBI", ascending=False).head(1).copy()
    rel_only["stage"] = "Relative credit only"
    rel_only = rel_only.rename(
        columns={"auc_cNBI": "mean_auc_cNBI", "R": "mean_R", "time_s": "mean_time_s"}
    )
    rel_only = rel_only[["stage", "mean_auc_cNBI", "mean_R", "mean_time_s"]]
    selected = ablation[ablation["method"].isin(stage_map)].copy()
    selected["stage"] = selected["method"].map(stage_map)
    selected = selected[["stage", "mean_auc_cNBI", "mean_R", "mean_time_s"]]
    stage_order = [
        "Initial automatic search",
        "Relative credit only",
        "Relative + time credit",
        "Full HAST-Q",
        "Full HAST-S",
    ]
    stage_evidence = pd.concat([selected, rel_only], ignore_index=True)
    stage_evidence["_order"] = stage_evidence["stage"].map({s: i for i, s in enumerate(stage_order)})
    stage_evidence = stage_evidence.sort_values("_order").drop(columns=["_order"])
    write_csv(stage_evidence, "motivation_obs2_obs3_stage_evidence.csv")

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    scatter = ax.scatter(
        fac_records["proxy_time_s"],
        fac_records["fac_score"],
        c=fac_records["fac_auc_adv"],
        cmap="viridis",
        s=42,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.axvline(1.8, color="#D55E00", linestyle="--", linewidth=1.0)
    ax.text(1.86, fac_records["fac_score"].quantile(0.12), "too-slow region", color="#D55E00")
    ax.set_xlabel("proxy runtime during search (s)")
    ax.set_ylabel("FAC score")
    ax.set_title("Observation 2: fragmentation credit drifts toward slower scans")
    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("FAC auc advantage")
    save_fig(fig, "fig22_motivation_obs2_fac_runtime_drift")

    fig, ax1 = plt.subplots(figsize=(6.9, 3.8))
    x = list(range(len(stage_evidence)))
    ax1.plot(x, stage_evidence["mean_auc_cNBI"], marker="o", color="#0072B2", label="auc-cNBI")
    ax1.set_ylabel("mean auc-cNBI", color="#0072B2")
    ax1.tick_params(axis="y", labelcolor="#0072B2")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stage_evidence["stage"], rotation=18, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, stage_evidence["mean_time_s"], marker="s", color="#D55E00", label="runtime")
    ax2.set_ylabel("mean runtime (s)", color="#D55E00")
    ax2.tick_params(axis="y", labelcolor="#D55E00")
    ax1.set_title("Observation 3: time-aware credit helps, bounded generation compresses further")
    save_fig(fig, "fig23_motivation_obs3_stage_compression")

    return bucket_summary, old_top, fact_top_features, stage_evidence


def write_report(
    corr: pd.DataFrame,
    same: pd.DataFrame,
    bucket: pd.DataFrame,
    old_top: pd.DataFrame,
    fact_top: pd.DataFrame,
    stage: pd.DataFrame,
) -> Path:
    all_corr = corr[corr["scope"] == "all"].iloc[0]
    no_hda = corr[corr["scope"] == "without_HDA_CoreHD"].iloc[0]
    slow = bucket[bucket["time_bucket"] == "too_slow"].iloc[0]
    fast = bucket[bucket["time_bucket"] == "fast<=0.8"].iloc[0]
    rel = stage[stage["stage"] == "Relative credit only"].iloc[0]
    q = stage[stage["stage"] == "Full HAST-Q"].iloc[0]
    s = stage[stage["stage"] == "Full HAST-S"].iloc[0]

    lines = [
        "# Motivation Observation Experiments",
        "",
        "本报告基于已有缓存结果和候选代码静态诊断生成，没有重新运行昂贵的 12 图评测。",
        "",
        "## Observation 1：GCC/R 信号对搜索期信用太粗",
        "",
        f"- 全部 192 个 dataset-method 点中，auc-cNBI 与 R 的 Spearman 相关为 {all_corr['spearman_auc_cNBI_vs_R']:.3f}，Pearson 相关为 {all_corr['pearson_auc_cNBI_vs_R']:.3f}。",
        f"- 去掉 HDA/CoreHD 后，Spearman 仍为 {no_hda['spearman_auc_cNBI_vs_R']:.3f}，说明 cNBI 与 R/GCC 相关但不是重复指标。",
        f"- same-GCC top case 的 cNBI 差距达到 {same['delta_cNBI_a_minus_b'].max():.1f}，可作为 3.1 的小表证据。",
        "",
        "输出：`motivation_obs1_cnbi_r_correlation.csv`、`motivation_obs1_same_gcc_top_cases.csv`、`fig21_motivation_obs1_same_gcc_cnbi_gap.*`。",
        "",
        "## Observation 2：相对碎裂信用有效，但诱导慢扫描",
        "",
        f"- FAC-only 最强候选达到 mean auc-cNBI {rel['mean_auc_cNBI']:.3f}，但 mean runtime 为 {rel['mean_time_s']:.3f}s。",
        f"- too_slow bucket 的平均 FAC advantage 为 {slow['mean_fac_auc_adv']:.3f}，高于 fast bucket 的 {fast['mean_fac_auc_adv']:.3f}。",
        f"- too_slow bucket 的平均 proxy runtime 为 {slow['mean_proxy_time_s']:.3f}s，fast bucket 为 {fast['mean_proxy_time_s']:.3f}s。",
        f"- too_slow bucket 中 two-hop/nested-neighbor 扫描率为 {slow['rate_twohop_or_nested_neighbor_scan']:.2f}，frontier/weak/boundary 特征率为 {slow['rate_frontier_weak_boundary_terms']:.2f}。",
        "",
        "输出：`motivation_obs2_fac_code_feature_by_time_bucket.csv`、`motivation_obs2_top_fac_candidates_code_features.csv`、`fig22_motivation_obs2_fac_runtime_drift.*`。",
        "",
        "## Observation 3：时间惩罚还不够，需要日志归纳的有界生成",
        "",
        f"- Full HAST-Q 达到 mean auc-cNBI {q['mean_auc_cNBI']:.3f}，runtime {q['mean_time_s']:.3f}s。",
        f"- Full HAST-S 达到 mean auc-cNBI {s['mean_auc_cNBI']:.3f}，runtime {s['mean_time_s']:.3f}s。",
        f"- 相比 FAC-only，Full HAST-Q 保留 {q['mean_auc_cNBI'] / rel['mean_auc_cNBI'] * 100:.1f}% 的 auc-cNBI，同时 runtime 降为 {q['mean_time_s'] / rel['mean_time_s'] * 100:.1f}%。",
        f"- 相比 FAC-only，Full HAST-S 保留 {s['mean_auc_cNBI'] / rel['mean_auc_cNBI'] * 100:.1f}% 的 auc-cNBI，同时 runtime 降为 {s['mean_time_s'] / rel['mean_time_s'] * 100:.1f}%。",
        "",
        "输出：`motivation_obs3_top_fact_candidates_code_features.csv`、`motivation_obs2_obs3_stage_evidence.csv`、`fig23_motivation_obs3_stage_compression.*`。",
        "",
        "## 漏洞核查",
        "",
        "- 这些实验足以支撑 motivation 章节，但不能替代正式方法和主结果实验。",
        "- 候选代码诊断是静态关键词/结构统计，不是精确复杂度证明；正式论文中应表述为 search-log diagnostic evidence。",
        "- Observation 1 不能写成 cNBI 替代 GCC/R，只能写成 cNBI 为搜索期信用提供补充过程信号。",
        "- Observation 3 的 bounded generation 仍需在方法章节明确约束来自日志归纳，而不是人工事后挑选。",
        "",
        "## 建议放入第 13 版的位置",
        "",
        "- 3.1：放 `motivation_obs1_cnbi_r_correlation.csv` 的两行摘要 + same-GCC top cases 小表。",
        "- 3.2：放 `fig22` 或 `motivation_obs2_fac_code_feature_by_time_bucket.csv`，主文强调 FAC-only 强但慢。",
        "- 3.3：合并原 3.3/3.4，放 `fig23` 和失败模式到边界规则表。",
        "",
    ]
    path = REPORT_DIR / "motivation_observation_experiments_cn.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    ensure_dirs()
    corr, same = observation_1()
    bucket, old_top, fact_top, stage = observation_2_and_3()
    report = write_report(corr, same, bucket, old_top, fact_top, stage)
    print(f"Wrote report: {report}")
    print("Wrote motivation observation tables and figures.")


if __name__ == "__main__":
    main()
