# -*- coding: utf-8 -*-
"""Small evidence-driven probes for DACTS credit-assignment ideas.

The goal is not to introduce a new full experiment harness.  This script reads
the completed 500-node ablation run and runs compact analyses that test whether
typed operator credit assignment has measurable signal.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"

DACTS_SCRIPT = ROOT / "src" / "dacts_rerun_search.py"
DACTS_RECORDS = ROOT / "runs" / "DACTS-rerun" / "outputs" / "search_records.csv"
GRAPH_JSON = ROOT / "runs" / "DACTS-rerun" / "outputs" / "graphs_50_powerlaw500.json"


def load_dacts_module() -> Any:
    spec = importlib.util.spec_from_file_location("dacts_rerun_for_credit", DACTS_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dacts_rerun_for_credit"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_graphs() -> List[nx.Graph]:
    payload = json.loads(GRAPH_JSON.read_text(encoding="utf-8"))
    return [
        json_graph.node_link_graph(item, edges="edges" if "edges" in item else "links")
        for item in payload
    ]


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def cfg_from_row(dacts: Any, row: pd.Series) -> Any:
    cfg = dacts.AlgoConfig(
        clade=str(row["clade"]),
        w_split=float(row["w_split"]),
        w_bridge_mult=float(row["w_bridge_mult"]),
        w_degree=float(row["w_degree"]),
        w_nds=float(row["w_nds"]),
        w_core=float(row["w_core"]),
        w_comp=float(row["w_comp"]),
        w_bridge_edges=float(row["w_bridge_edges"]),
        split_power=float(row["split_power"]),
        degree_power=float(row["degree_power"]),
        nds_power=float(row["nds_power"]),
        core_power=float(row["core_power"]),
        update_radius=int(row["update_radius"]),
        use_component=as_bool(row["use_component"]),
        comp_refresh=int(row["comp_refresh"]),
        parent_id=str(row.get("parent_id", "")),
        mutation=str(row.get("mutation", "")),
        node_id=str(row.get("node_id", "")),
    )
    return cfg


def add_global_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["global_score"] = np.nan
    valid = out["valid"] & out["R"].notna() & out["cNBI"].notna() & out["Time"].notna()
    sub = out[valid].copy()
    denom = max(1, len(sub) - 1)
    for metric, higher, col in [
        ("R", False, "g_R"),
        ("cNBI", True, "g_cNBI"),
        ("Time", False, "g_Time"),
    ]:
        ordered = sub.sort_values(metric, ascending=not higher)
        score = {idx: (denom - pos) / denom for pos, idx in enumerate(ordered.index)}
        out[col] = out.index.map(score)
    out.loc[valid, "global_score"] = (
        0.4 * out.loc[valid, "g_R"]
        + 0.3 * out.loc[valid, "g_cNBI"]
        + 0.3 * out.loc[valid, "g_Time"]
    )
    return out


def load_records() -> pd.DataFrame:
    df = pd.read_csv(DACTS_RECORDS)
    df["valid"] = df["valid"].astype(str).str.lower().isin(["true", "1", "yes"])
    for col in [
        "idx",
        "R",
        "cNBI",
        "Time",
        "rank_score",
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
        "comp_refresh",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return add_global_score(df)


def parent_child_credit(df: pd.DataFrame) -> pd.DataFrame:
    by_id = {str(r.node_id): r for _, r in df.iterrows()}
    fields = [
        "w_split",
        "w_bridge_mult",
        "w_degree",
        "w_nds",
        "w_core",
        "w_comp",
        "w_bridge_edges",
        "update_radius",
        "use_component",
    ]
    rows: List[Dict[str, Any]] = []
    for _, child in df.iterrows():
        parent = by_id.get(str(child.get("parent_id", "")))
        if parent is None or not bool(child["valid"]) or not bool(parent["valid"]):
            continue
        delta_r = float(parent["R"]) - float(child["R"])
        delta_c = float(child["cNBI"]) - float(parent["cNBI"])
        delta_t = float(parent["Time"]) - float(child["Time"])
        delta_s = float(child["global_score"]) - float(parent["global_score"])
        for field in fields:
            cv = child[field]
            pv = parent[field]
            if field == "use_component":
                cv = 1.0 if as_bool(cv) else 0.0
                pv = 1.0 if as_bool(pv) else 0.0
            diff = float(cv) - float(pv)
            if abs(diff) < 1e-12:
                continue
            rows.append(
                {
                    "field": field,
                    "direction": "increase" if diff > 0 else "decrease",
                    "n": 1,
                    "delta_value": diff,
                    "delta_R_good": delta_r,
                    "delta_cNBI_good": delta_c,
                    "delta_Time_good": delta_t,
                    "delta_global_score": delta_s,
                }
            )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    summary = (
        raw.groupby(["field", "direction"], as_index=False)
        .agg(
            n=("n", "sum"),
            mean_delta_value=("delta_value", "mean"),
            mean_delta_R_good=("delta_R_good", "mean"),
            mean_delta_cNBI_good=("delta_cNBI_good", "mean"),
            mean_delta_Time_good=("delta_Time_good", "mean"),
            mean_delta_global_score=("delta_global_score", "mean"),
            median_delta_global_score=("delta_global_score", "median"),
        )
        .sort_values("mean_delta_global_score", ascending=False)
    )
    return summary


def clade_branch_credit(df: pd.DataFrame) -> pd.DataFrame:
    ref_path = WORKSPACE / "research" / "dacts_e26f_reverse_search_20260519" / "outputs" / "reference_comparison.csv"
    ref = pd.read_csv(ref_path)
    e26f = ref[ref["name"].eq("e26f_reference")].iloc[0]
    r_ref, c_ref, t_ref = float(e26f["R"]), float(e26f["cNBI"]), float(e26f["Time"])
    out = df.copy()
    out["strict_e26f_like"] = (
        out["valid"]
        & (out["R"] <= r_ref + 0.0006)
        & (out["cNBI"] >= c_ref - 0.12)
        & (out["Time"] <= t_ref * 2.0)
    )
    rows = []
    for clade, sub in out.groupby("clade"):
        valid = sub[sub["valid"]]
        if valid.empty:
            continue
        best = valid.sort_values("global_score", ascending=False).iloc[0]
        rows.append(
            {
                "clade": clade,
                "nodes": len(sub),
                "valid": int(valid.shape[0]),
                "strict_hits": int(valid["strict_e26f_like"].sum()),
                "strict_density": float(valid["strict_e26f_like"].mean()),
                "mean_global_score": float(valid["global_score"].mean()),
                "top10_mean_global_score": float(valid.sort_values("global_score", ascending=False).head(10)["global_score"].mean()),
                "best_idx": int(best["idx"]),
                "best_R": float(best["R"]),
                "best_cNBI": float(best["cNBI"]),
                "best_Time": float(best["Time"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["strict_hits", "top10_mean_global_score"], ascending=False)


def counterfactual_ablation(dacts: Any, df: pd.DataFrame) -> pd.DataFrame:
    graphs = load_graphs()
    best = df[df["valid"]].sort_values("global_score", ascending=False).iloc[0]
    base_cfg = cfg_from_row(dacts, best)
    variants: List[tuple[str, Any]] = [("original", base_cfg)]

    def add_variant(name: str, **updates: Any) -> None:
        cfg = copy.deepcopy(base_cfg)
        for key, val in updates.items():
            setattr(cfg, key, val)
        variants.append((name, cfg))

    add_variant("no_split", w_split=0.0)
    add_variant("no_bridge_mult", w_bridge_mult=0.0)
    add_variant("no_degree", w_degree=0.0)
    add_variant("no_neighbor_degree", w_nds=0.0)
    add_variant("no_core", w_core=0.0)
    add_variant("no_component", use_component=False, w_comp=0.0)
    add_variant("no_bridge_edges", w_bridge_edges=0.0)
    add_variant("radius_1", update_radius=1)
    add_variant("radius_2", update_radius=2)
    add_variant("split_power_0.75", split_power=0.75)
    add_variant("split_power_1.30", split_power=1.30)

    rows = []
    for name, cfg in variants:
        result = dacts.evaluate_config(cfg, graphs, budget_ratio=0.30)["avg"]
        rows.append(
            {
                "variant": name,
                "R": result["R"],
                "cNBI": result["cNBI"],
                "Time": result["Time"],
            }
        )
    out = pd.DataFrame(rows)
    base = out[out["variant"].eq("original")].iloc[0]
    out["delta_R_bad"] = out["R"] - float(base["R"])
    out["delta_cNBI_bad"] = float(base["cNBI"]) - out["cNBI"]
    out["delta_Time_bad"] = out["Time"] - float(base["Time"])
    out["evidence"] = (
        "ablation hurts" 
        + " R=" + out["delta_R_bad"].round(6).astype(str)
        + ", cNBI=" + out["delta_cNBI_bad"].round(3).astype(str)
        + ", Time=" + out["delta_Time_bad"].round(5).astype(str)
    )
    return out


def plot_counterfactual(counter: pd.DataFrame) -> None:
    data = counter[counter["variant"].ne("original")].copy()
    data = data.sort_values("delta_cNBI_bad", ascending=False)
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    x = np.arange(len(data))
    ax.bar(x, data["delta_cNBI_bad"], color="#4C78A8", label="cNBI loss when ablated")
    ax2 = ax.twinx()
    ax2.plot(x, data["delta_R_bad"], color="#D62728", marker="o", lw=1.5, label="R increase")
    ax.set_xticks(x)
    ax.set_xticklabels(data["variant"], rotation=35, ha="right")
    ax.set_ylabel("cNBI loss")
    ax2.set_ylabel("R increase")
    ax.set_title("Counterfactual operator ablation on DACTS-rerun best config")
    ax.grid(alpha=0.18, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dacts_counterfactual_operator_ablation.png", dpi=300)
    fig.savefig(FIG_DIR / "dacts_counterfactual_operator_ablation.pdf")
    plt.close(fig)


def write_report(
    df: pd.DataFrame,
    pc: pd.DataFrame,
    clade: pd.DataFrame,
    counter: pd.DataFrame,
) -> None:
    best = df[df["valid"]].sort_values("global_score", ascending=False).iloc[0]
    top_pc = pc.head(8).copy()
    report = [
        "# LLMTree 网络瓦解边界与新方向探索",
        "",
        "生成日期：2026-05-21",
        "",
        "## 1. 这次小实验想回答什么",
        "",
        "我们不再只问“哪个完整候选算法分数最高”，而是问：整体 R/cNBI/Time 分数能否被分解成 typed operator 的信用信号？如果可以，DACTS 的下一步就不只是网络瓦解应用，而是 outcome-only LLM algorithm search 的信用分配改进。",
        "",
        "## 2. 已完成搜索的边界证据",
        "",
        "- 通用框架能在 500 节点内找到搜索集上非常强的自由代码候选，说明“只靠搜索策略”并不是 DACTS 独有优势。",
        "- 但 DACTS-rerun 500/500 有效，通用框架无效率更高；DACTS 的 typed fields 让每个候选都有可审计机制槽位。",
        "- 现有 12 图探索显示，搜索集上很强的自由代码候选未必泛化到真实图；这暴露了 outcome-only/free-code search 的边界。",
        "",
        "## 3. DACTS-rerun best config",
        "",
        f"- best_idx: `{int(best['idx'])}`",
        f"- clade: `{best['clade']}`",
        f"- R: `{float(best['R']):.6f}`",
        f"- cNBI: `{float(best['cNBI']):.3f}`",
        f"- Time: `{float(best['Time']):.5f}`",
        f"- global_score: `{float(best['global_score']):.4f}`",
        "",
        "## 4. Parent-child operator credit 证据",
        "",
        "下表统计当某个 typed field 在 parent→child mutation 中增加/减少时，child 相对 parent 的平均指标变化。`delta_R_good` 越大表示 R 降得越多，`delta_cNBI_good` 越大表示 cNBI 升得越多，`delta_Time_good` 越大表示更快。",
        "",
        top_pc.to_markdown(index=False),
        "",
        "观察：如果某些 field 的 direction 在多次 mutation 中稳定带来正 delta，就说明 typed operator 不是装饰，而是可以被估计 credit 的结构变量。",
        "",
        "## 5. Clade branch-level Monte Carlo credit",
        "",
        clade.to_markdown(index=False),
        "",
        "观察：不同 clade 的 strict hit density、top-k 后代质量不同，说明搜索树的分支/机制族本身有可学习信用，而不是只有叶子候选有分数。",
        "",
        "## 6. 反事实算子消融",
        "",
        counter.to_markdown(index=False),
        "",
        "观察：反事实消融能回答“强候选到底靠哪个机制”。如果去掉某个算子显著损失 cNBI 或提高 R，该算子就获得正信用；如果去掉后更快但 R/cNBI 损失很小，说明它可能是冗余或复杂度负担。",
        "",
        "## 7. 新方向推荐",
        "",
        "最值得推进的是 DACTS-CA：Diagnostic Algorithmic Credit Tree Search。它把 scalar outcome reward 分解为三类 credit：",
        "",
        "1. parent-child progress credit：每次 mutation 记录相对父节点的 R/cNBI/Time 改善。",
        "2. counterfactual operator credit：对强候选去掉或替换 typed operator，估计边际贡献。",
        "3. branch-level Monte Carlo credit：对 clade/operator 子树统计 strict-hit density、top-k quality 和 invalid rate。",
        "",
        "这条线的边界也很清楚：如果反事实 credit 与后续 mutation 成功率不相关，它就只是解释工具；如果相关，它就是新的搜索控制机制。",
    ]
    (REPORT_DIR / "dacts_credit_boundary_exploration_20260521_cn.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dacts = load_dacts_module()
    df = load_records()
    pc = parent_child_credit(df)
    clade = clade_branch_credit(df)
    counter = counterfactual_ablation(dacts, df)
    pc.to_csv(TABLE_DIR / "dacts_parent_child_operator_credit.csv", index=False, encoding="utf-8-sig")
    clade.to_csv(TABLE_DIR / "dacts_clade_branch_credit.csv", index=False, encoding="utf-8-sig")
    counter.to_csv(TABLE_DIR / "dacts_counterfactual_operator_ablation.csv", index=False, encoding="utf-8-sig")
    plot_counterfactual(counter)
    write_report(df, pc, clade, counter)
    print("Wrote credit-boundary exploration tables and report.")


if __name__ == "__main__":
    main()
