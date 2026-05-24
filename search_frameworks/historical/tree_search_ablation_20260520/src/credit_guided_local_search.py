# -*- coding: utf-8 -*-
"""Credit-guided local search around the best DACTS-rerun typed config.

This is a small, non-LLM experiment: use the operator-credit evidence from
counterfactual ablation to constrain a typed local search, then test whether it
can find a better candidate quickly.
"""

from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
TABLE_DIR = ROOT / "tables"
REPORT_DIR = ROOT / "reports"
DACTS_SCRIPT = ROOT / "src" / "dacts_rerun_search.py"
GRAPH_JSON = ROOT / "runs" / "DACTS-rerun" / "outputs" / "graphs_50_powerlaw500.json"
DACTS_RECORDS = ROOT / "runs" / "DACTS-rerun" / "outputs" / "search_records.csv"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_graphs() -> List[nx.Graph]:
    payload = json.loads(GRAPH_JSON.read_text(encoding="utf-8"))
    return [
        json_graph.node_link_graph(item, edges="edges" if "edges" in item else "links")
        for item in payload
    ]


def load_eval12() -> Any:
    path = WORKSPACE / "research" / "dacts_e26f_reverse_search_20260519" / "src" / "evaluate_best_on_12_graphs.py"
    return load_module(path, "eval12_for_credit_local")


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def cfg_from_row(dacts: Any, row: pd.Series) -> Any:
    return dacts.AlgoConfig(
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


def add_global_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["local_score"] = np.nan
    valid = out["R"].notna() & out["cNBI"].notna() & out["Time"].notna()
    sub = out[valid]
    denom = max(1, len(sub) - 1)
    for metric, higher, col in [("R", False, "s_R"), ("cNBI", True, "s_cNBI"), ("Time", False, "s_Time")]:
        ordered = sub.sort_values(metric, ascending=not higher)
        vals = {idx: (denom - pos) / denom for pos, idx in enumerate(ordered.index)}
        out[col] = out.index.map(vals)
    out.loc[valid, "local_score"] = 0.4 * out.loc[valid, "s_R"] + 0.3 * out.loc[valid, "s_cNBI"] + 0.3 * out.loc[valid, "s_Time"]
    return out


def evaluate_cfg(dacts: Any, graphs: List[nx.Graph], cfg: Any, name: str, source: str) -> Dict[str, Any]:
    result = dacts.evaluate_config(cfg, graphs, budget_ratio=0.30)["avg"]
    return {
        "name": name,
        "source": source,
        "clade": cfg.clade,
        "w_split": cfg.w_split,
        "w_bridge_mult": cfg.w_bridge_mult,
        "w_degree": cfg.w_degree,
        "w_nds": cfg.w_nds,
        "w_core": cfg.w_core,
        "w_comp": cfg.w_comp,
        "w_bridge_edges": cfg.w_bridge_edges,
        "split_power": cfg.split_power,
        "degree_power": cfg.degree_power,
        "nds_power": cfg.nds_power,
        "update_radius": cfg.update_radius,
        "use_component": cfg.use_component,
        "R": result["R"],
        "cNBI": result["cNBI"],
        "Time": result["Time"],
    }


def make_credit_guided_variants(base: Any) -> List[tuple[str, Any]]:
    """Generate a compact grid focused on operators with measured credit."""
    variants: List[tuple[str, Any]] = []
    grid = {
        "w_split": [3.5, 4.0],
        "w_bridge_mult": [3.0, 4.5],
        "w_degree": [0.0, 0.055, 0.12, 0.20],
        "w_nds": [0.045, 0.085, 0.16, 0.28],
        "split_power": [1.15, 1.30],
        "nds_power": [0.75, 1.0],
        "update_radius": [1, 2],
    }
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        cfg = copy.deepcopy(base)
        for key, val in zip(keys, vals):
            setattr(cfg, key, val)
        cfg.clade = "bridge_aware"
        cfg.w_core = 0.0
        cfg.w_comp = 0.0
        cfg.w_bridge_edges = 0.0
        cfg.use_component = False
        variants.append(("credit_grid", cfg))
    return variants


def make_control_variants(base: Any) -> List[tuple[str, Any]]:
    """A small less-informed grid that ignores the ablation signal."""
    variants: List[tuple[str, Any]] = []
    grid = {
        "clade": ["bridge_aware", "component_aware", "local_bridge_edges"],
        "w_split": [0.0, 2.0, 4.0],
        "w_bridge_mult": [0.0, 2.0, 4.5],
        "w_degree": [0.0, 0.5, 1.0],
        "w_nds": [0.0, 0.5, 1.0],
        "update_radius": [1, 2],
    }
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        cfg = copy.deepcopy(base)
        for key, val in zip(keys, vals):
            setattr(cfg, key, val)
        cfg.split_power = 1.0
        cfg.degree_power = 1.0
        cfg.nds_power = 1.0
        cfg.w_core = 0.0
        cfg.w_comp = 0.2 if cfg.clade == "component_aware" else 0.0
        cfg.w_bridge_edges = 0.4 if cfg.clade == "local_bridge_edges" else 0.0
        cfg.use_component = cfg.clade == "component_aware"
        variants.append(("control_grid", cfg))
    return variants


def eval_12_graphs(dacts: Any, cfgs: pd.DataFrame) -> pd.DataFrame:
    eval12 = load_eval12()
    rows = []
    for _, row in cfgs.iterrows():
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
            update_radius=int(row["update_radius"]),
            use_component=as_bool(row["use_component"]),
            comp_refresh=25,
        )
        per = []
        for dataset in eval12.DATASETS:
            graph = eval12.read_graph(dataset)
            rate = eval12.DATASET_RATES[dataset]
            t0 = time.perf_counter()
            order = dacts.degree_order_by_config(graph, cfg, budget_ratio=rate)
            elapsed = time.perf_counter() - t0
            metrics = eval12.compute_metrics(graph, order, rate=rate, method_time=elapsed)
            x = metrics["removal_ratio"].to_numpy(dtype=float)
            per.append(
                {
                    "R": float(metrics["GCC"].mean()),
                    "auc_cNBI": float(np.trapz(metrics["cNBI"].to_numpy(dtype=float), x)),
                    "time_s": elapsed,
                }
            )
        rows.append(
            {
                "name": row["name"],
                "source": row["source"],
                "search_R": row["R"],
                "search_cNBI": row["cNBI"],
                "search_Time": row["Time"],
                "search_local_score": row["local_score"],
                "R": float(np.mean([x["R"] for x in per])),
                "auc_cNBI": float(np.mean([x["auc_cNBI"] for x in per])),
                "time_s": float(np.mean([x["time_s"] for x in per])),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dacts = load_module(DACTS_SCRIPT, "dacts_for_credit_local")
    graphs = load_graphs()
    records = pd.read_csv(DACTS_RECORDS)
    records["valid"] = records["valid"].astype(str).str.lower().isin(["true", "1", "yes"])
    for col in ["R", "cNBI", "Time", "rank_score"]:
        records[col] = pd.to_numeric(records[col], errors="coerce")
    base_row = records[records["valid"]].sort_values("rank_score", ascending=False).iloc[0]
    base_cfg = cfg_from_row(dacts, base_row)

    rows = [evaluate_cfg(dacts, graphs, base_cfg, "baseline_idx31", "baseline")]
    for i, (source, cfg) in enumerate(make_credit_guided_variants(base_cfg)):
        rows.append(evaluate_cfg(dacts, graphs, cfg, f"credit_{i:03d}", source))
    # Match the number of control evaluations to the credit grid for fairness.
    control = make_control_variants(base_cfg)[: len(rows) - 1]
    for i, (source, cfg) in enumerate(control):
        rows.append(evaluate_cfg(dacts, graphs, cfg, f"control_{i:03d}", source))

    result = add_global_score(pd.DataFrame(rows))
    result.to_csv(TABLE_DIR / "credit_guided_local_search.csv", index=False, encoding="utf-8-sig")

    selected = pd.concat(
        [
            result[result["source"].eq("baseline")],
            result[result["source"].eq("credit_grid")].sort_values("local_score", ascending=False).head(5),
            result[result["source"].eq("control_grid")].sort_values("local_score", ascending=False).head(5),
        ],
        ignore_index=True,
    )
    eval12 = eval_12_graphs(dacts, selected)
    eval12.to_csv(TABLE_DIR / "credit_guided_local_search_12graph.csv", index=False, encoding="utf-8-sig")

    best_credit = result[result["source"].eq("credit_grid")].sort_values("local_score", ascending=False).iloc[0]
    best_control = result[result["source"].eq("control_grid")].sort_values("local_score", ascending=False).iloc[0]
    lines = [
        "# Credit-guided Local Search 小实验",
        "",
        "目标：检验算子信用分配是否能在不调用 LLM 的情况下，快速约束 typed search 并找到更好的候选。",
        "",
        "## 搜索图结果",
        "",
        "| source | best name | R | cNBI | Time | local_score |",
        "|---|---:|---:|---:|---:|---:|",
        f"| baseline | baseline_idx31 | {float(base_row['R']):.6f} | {float(base_row['cNBI']):.3f} | {float(base_row['Time']):.5f} | {float(base_row['rank_score']):.4f} |",
        f"| credit_grid | {best_credit['name']} | {best_credit['R']:.6f} | {best_credit['cNBI']:.3f} | {best_credit['Time']:.5f} | {best_credit['local_score']:.4f} |",
        f"| control_grid | {best_control['name']} | {best_control['R']:.6f} | {best_control['cNBI']:.3f} | {best_control['Time']:.5f} | {best_control['local_score']:.4f} |",
        "",
        "## 12 图 top 候选复核",
        "",
        eval12.sort_values(["source", "search_local_score"], ascending=[True, False]).to_markdown(index=False),
        "",
        "## 初步判断",
        "",
        "- 如果 credit_grid 在同等评估数量下优于 control_grid，说明反事实信用可以缩小搜索空间。",
        "- 如果 12 图复核仍不提升，说明当前 credit 主要解释搜索图机制，还需要把泛化图或稳定性纳入 credit。",
    ]
    (REPORT_DIR / "credit_guided_local_search_20260521_cn.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote credit-guided local search results.")


if __name__ == "__main__":
    main()
