# -*- coding: utf-8 -*-
"""Check whether DACTS-discovered mechanism signals transfer to generic searches."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "tables"
REPORT_DIR = ROOT / "reports"
METHODS = ["PUCT", "MCTS-AHD-like", "Clade-AHD-like", "FunSearch-like", "AlphaEvolve-like"]


def read_records_with_code(method: str) -> pd.DataFrame:
    path = ROOT / "runs" / method / "records_with_code.jsonl"
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["method"] = method
    df["valid"] = df["valid"].astype(str).str.lower().isin(["true", "1", "yes"])
    for col in ["idx", "R", "cNBI", "Time", "rank_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["code"] = df["code"].fillna("")
    return df


def add_global_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["generic_global_score"] = np.nan
    valid = out["valid"] & out["R"].notna() & out["cNBI"].notna() & out["Time"].notna()
    sub = out[valid].copy()
    denom = max(1, len(sub) - 1)
    for metric, higher, col in [("R", False, "g_R"), ("cNBI", True, "g_cNBI"), ("Time", False, "g_Time")]:
        ordered = sub.sort_values(metric, ascending=not higher)
        vals = {idx: (denom - pos) / denom for pos, idx in enumerate(ordered.index)}
        out[col] = out.index.map(vals)
    out.loc[valid, "generic_global_score"] = (
        0.4 * out.loc[valid, "g_R"] + 0.3 * out.loc[valid, "g_cNBI"] + 0.3 * out.loc[valid, "g_Time"]
    )
    return out


def has_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def extract_features(code: str) -> Dict[str, bool]:
    c = code.lower()
    return {
        "heap_lazy_update": has_any(c, [r"heapq", r"version", r"stamp", r"push\("]),
        "neighbor_degree": has_any(c, [r"deg\.get\(v", r"h\.degree\(v", r"degree\(v", r"sum_nd", r"neigh_sum", r"s1", r"s2", r"avg_nd"]),
        "two_hop_exposure": has_any(c, [r"two[_-]?hop", r"second_unique", r"affected\.update", r"neighbors\(w\)", r"for z in .*neighbors\(w\)"]),
        "bridge_cut_signal": has_any(c, [r"bridge", r"cut", r"frontier", r"outside", r"external", r"escape", r"broker", r"articulation"]),
        "redundancy_penalty": has_any(c, [r"tri", r"cluster", r"redundan", r"internal_edges", r"has_edge"]),
        "component_refresh": has_any(c, [r"connected_components", r"comp_size", r"component"]),
        "core_signal": has_any(c, [r"core_number", r"kcore", r"core"]),
        "global_expensive": has_any(c, [r"betweenness", r"shortest_path", r"eigen", r"pagerank", r"community", r"spectral"]),
        "full_recompute_loop": has_any(c, [r"while .*number_of_nodes", r"score\(u\).*for u in h\.nodes", r"max\(h\.nodes"]),
    }


def summarize_feature(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = df[df["valid"] & df["generic_global_score"].notna()].copy()
    top_threshold = valid["generic_global_score"].quantile(0.8)
    valid["top20"] = valid["generic_global_score"] >= top_threshold
    for feat in [c for c in valid.columns if c.startswith("feat_")]:
        yes = valid[valid[feat]]
        no = valid[~valid[feat]]
        if yes.empty:
            continue
        rows.append(
            {
                "feature": feat.replace("feat_", ""),
                "n_yes": len(yes),
                "n_no": len(no),
                "top20_rate_yes": float(yes["top20"].mean()),
                "top20_rate_no": float(no["top20"].mean()) if len(no) else np.nan,
                "mean_score_yes": float(yes["generic_global_score"].mean()),
                "mean_score_no": float(no["generic_global_score"].mean()) if len(no) else np.nan,
                "mean_R_yes": float(yes["R"].mean()),
                "mean_R_no": float(no["R"].mean()) if len(no) else np.nan,
                "mean_cNBI_yes": float(yes["cNBI"].mean()),
                "mean_cNBI_no": float(no["cNBI"].mean()) if len(no) else np.nan,
                "mean_Time_yes": float(yes["Time"].mean()),
                "mean_Time_no": float(no["Time"].mean()) if len(no) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["score_lift"] = out["mean_score_yes"] - out["mean_score_no"]
    out["top20_lift"] = out["top20_rate_yes"] - out["top20_rate_no"]
    return out.sort_values(["top20_lift", "score_lift"], ascending=False)


def summarize_by_method(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in df.groupby("method"):
        valid = sub[sub["valid"] & sub["generic_global_score"].notna()]
        if valid.empty:
            continue
        top = valid.sort_values("generic_global_score", ascending=False).head(50)
        rows.append(
            {
                "method": method,
                "valid": len(valid),
                "top50_neighbor_degree": float(top["feat_neighbor_degree"].mean()),
                "top50_two_hop": float(top["feat_two_hop_exposure"].mean()),
                "top50_bridge_cut": float(top["feat_bridge_cut_signal"].mean()),
                "top50_redundancy": float(top["feat_redundancy_penalty"].mean()),
                "top50_component": float(top["feat_component_refresh"].mean()),
                "top50_core": float(top["feat_core_signal"].mean()),
                "top50_mean_Time": float(top["Time"].mean()),
                "top50_mean_score": float(top["generic_global_score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat([read_records_with_code(m) for m in METHODS], ignore_index=True)
    df = add_global_score(df)
    feats = df["code"].apply(extract_features)
    feat_df = pd.DataFrame(list(feats))
    for col in feat_df.columns:
        df[f"feat_{col}"] = feat_df[col].astype(bool)
    df.to_csv(TABLE_DIR / "cross_framework_candidate_mechanism_features.csv", index=False, encoding="utf-8-sig")
    feature_summary = summarize_feature(df)
    method_summary = summarize_by_method(df)
    feature_summary.to_csv(TABLE_DIR / "cross_framework_feature_quality_summary.csv", index=False, encoding="utf-8-sig")
    method_summary.to_csv(TABLE_DIR / "cross_framework_top50_mechanism_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 跨框架机制迁移检验",
        "",
        "问题：如果 DACTS 的机制信用是真实的，它是否也能解释 PUCT / MCTS-AHD-like / Clade-AHD-like / FunSearch-like / AlphaEvolve-like 的自由代码候选？",
        "",
        "## Feature-level summary",
        "",
        feature_summary.to_markdown(index=False),
        "",
        "## Top-50 by method",
        "",
        method_summary.to_markdown(index=False),
        "",
        "## 解释",
        "",
        "- 如果 neighbor_degree、two_hop_exposure、bridge_cut_signal 在通用框架 top20/top50 中也更常见，说明 DACTS 的 typed operator 不是只对 DACTS 有效。",
        "- 如果 component/core/global signals 提升分数但显著增加 Time，说明通用搜索也会沿着同一机制-复杂度张力移动。",
        "- 这可以支撑一个更强版本：DACTS 的创新不是发明某个网络瓦解启发式，而是把跨框架都会自发发现的机制显式 typed 化，并进行信用分配和复杂度约束。",
    ]
    (REPORT_DIR / "cross_framework_mechanism_transfer_20260521_cn.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote cross-framework mechanism transfer results.")


if __name__ == "__main__":
    main()
