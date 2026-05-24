# -*- coding: utf-8 -*-
"""HAST: Harnessed Adaptive Search Tree experiment.

This script keeps the HAST experiment intentionally compact:

1. Learn mechanism-level experience only from the first N nodes of the five
   generic free-code search frameworks.
2. Run a new HDA-root HAST specialization search.
3. Compare HAST against the held-out suffix of the same generic searches.

No DACTS-rerun or DACTS typed records are used for HAST training.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import re
import shutil
import sys
import textwrap
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DEPS = ROOT / "search_frameworks" / "runtime_deps"
SOURCE_ROOT = ROOT / "data" / "search_framework_records" / "raw" / "tree_search_ablation_20260520"
SOURCE_SEARCH = RUNTIME_DEPS / "ablation_search.py"
SOURCE_12GRAPH = RUNTIME_DEPS / "evaluate_final_12graphs.py"
DACTS_ROOT = SOURCE_ROOT / "runs" / "DACTS-rerun"

RUNS_DIR = ROOT / "runs"
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"

GENERIC_METHODS = ["ERA-like", "MCTS-AHD-like", "Clade-AHD-like", "FunSearch-like", "AlphaEvolve-like"]
HAST_METHOD = "HAST"
ROOT_ID = "hda_root"
DEFAULT_MODEL = "gpt-5.5"

FEATURES = [
    "heap_lazy_update",
    "neighbor_degree",
    "two_hop_exposure",
    "bridge_cut_signal",
    "redundancy_penalty",
    "component_refresh",
    "core_signal",
    "global_expensive",
    "full_recompute_loop",
]

FAMILY_ORDER = [
    "local_twohop_bridge",
    "local_twohop_neighbor",
    "bridge_core",
    "core_neighbor",
    "component_heavy",
    "global_expensive",
    "simple_degree",
]

FAMILY_GUIDANCE = {
    "local_twohop_bridge": (
        "Target a local two-hop / bridge-pressure variant. Keep heap or lazy local updates. "
        "Prefer local neighbor-degree, second-hop exposure, frontier or split-pressure proxies."
    ),
    "local_twohop_neighbor": (
        "Target a local two-hop neighbor-degree variant. Keep the mutation small: adjust how a node's "
        "degree, neighbor degrees, and second-hop exposure are mixed."
    ),
    "bridge_core": (
        "Target a bridge/core hybrid. Use only cheap local bridge or frontier proxies; do not call "
        "expensive articulation or all-pairs routines."
    ),
    "core_neighbor": (
        "Target a core/neighbor-degree hybrid. Keep the update local and avoid recomputing core numbers "
        "inside every deletion step."
    ),
    "component_heavy": (
        "Repair a component-aware idea by making component refresh sparse and cheap. Avoid per-step full "
        "connected-component recomputation."
    ),
    "global_expensive": (
        "Convert a globally expensive idea into a cheap local proxy. Remove shortest paths, betweenness, "
        "spectral methods, PageRank, and community detection."
    ),
    "simple_degree": (
        "Start from HDA and add only one cheap local signal. Keep the code short, iterative, and robust."
    ),
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SEARCH = load_module(SOURCE_SEARCH, "ablation_search_for_hast")


def ensure_dirs() -> None:
    for path in [RUNS_DIR, TABLE_DIR, FIG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    df = pd.read_csv(path)
    return df.where(pd.notna(df), None).to_dict("records")


def normalize_bool(s: Any) -> bool:
    return str(s).strip().lower() in {"true", "1", "yes"}


def has_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def extract_features(code: str) -> Dict[str, bool]:
    c = (code or "").lower()
    return {
        "heap_lazy_update": has_any(c, [r"heapq", r"heappush", r"heappop", r"version", r"stamp", r"push\("]),
        "neighbor_degree": has_any(
            c,
            [
                r"deg\.get\(v",
                r"h\.degree\(v",
                r"degree\(v",
                r"neighbor.*degree",
                r"neigh[_a-z]*sum",
                r"nn_sum",
                r"avg_nb",
                r"nbr[_a-z]*deg",
                r"avg_nd",
                r"sum_nd",
            ],
        ),
        "two_hop_exposure": has_any(
            c,
            [
                r"two[_ -]?hop",
                r"second[_a-z]*",
                r"neighbors\(w\)",
                r"for z in .*neighbors\(w\)",
                r"affected\.update",
                r"ex_sum",
                r"exposure",
                r"low2",
                r"s2",
            ],
        ),
        "bridge_cut_signal": has_any(
            c,
            [
                r"bridge",
                r"bridge_pressure",
                r"cut",
                r"split",
                r"frontier",
                r"outside",
                r"external",
                r"escape",
                r"broker",
                r"articulation",
                r"leaf_cnt",
                r"low2_cnt",
            ],
        ),
        "redundancy_penalty": has_any(c, [r"tri", r"cluster", r"redundan", r"internal_edges", r"has_edge"]),
        "component_refresh": has_any(c, [r"connected_components", r"component", r"comp_size", r"gcc", r"largest"]),
        "core_signal": has_any(c, [r"core_number", r"kcore", r"k_core", r"\bcore\b"]),
        "global_expensive": has_any(
            c,
            [r"betweenness", r"shortest_path", r"all_pairs", r"eigen", r"pagerank", r"community", r"spectral"],
        ),
        "full_recompute_loop": has_any(c, [r"while .*number_of_nodes", r"max\(h\.nodes", r"for u in h\.nodes\(\)"]),
    }


def classify_family(features: Dict[str, bool]) -> str:
    if features.get("global_expensive"):
        return "global_expensive"
    if features.get("component_refresh") and not (
        features.get("two_hop_exposure") and features.get("neighbor_degree")
    ):
        return "component_heavy"
    if features.get("two_hop_exposure") and features.get("bridge_cut_signal"):
        return "local_twohop_bridge"
    if features.get("two_hop_exposure") and features.get("neighbor_degree"):
        return "local_twohop_neighbor"
    if features.get("bridge_cut_signal") and features.get("core_signal"):
        return "bridge_core"
    if features.get("core_signal") and features.get("neighbor_degree"):
        return "core_neighbor"
    return "simple_degree"


def add_global_score(df: pd.DataFrame, out_col: str = "global_score") -> pd.DataFrame:
    out = df.copy()
    out[out_col] = np.nan
    valid = out["valid"] & out["R"].notna() & out["cNBI"].notna() & out["Time"].notna()
    sub = out[valid].copy()
    if sub.empty:
        return out
    denom = max(1, len(sub) - 1)
    for metric, higher, col in [("R", False, "g_R"), ("cNBI", True, "g_cNBI"), ("Time", False, "g_Time")]:
        ordered = sub.sort_values(metric, ascending=not higher)
        vals = {idx: (denom - pos) / denom for pos, idx in enumerate(ordered.index)}
        out[col] = out.index.map(vals)
    out.loc[valid, out_col] = 0.4 * out.loc[valid, "g_R"] + 0.3 * out.loc[valid, "g_cNBI"] + 0.3 * out.loc[valid, "g_Time"]
    return out


def e26f_reference() -> Tuple[float, float, float]:
    ref_path = DACTS_ROOT / "outputs" / "reference_comparison.csv"
    ref = pd.read_csv(ref_path)
    e26f = ref[ref["name"].eq("e26f_reference")].iloc[0]
    return float(e26f["R"]), float(e26f["cNBI"]), float(e26f["Time"])


def add_e26f_flags(df: pd.DataFrame) -> pd.DataFrame:
    r_ref, c_ref, t_ref = e26f_reference()
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


def read_records_with_code(method: str) -> pd.DataFrame:
    path = SOURCE_ROOT / "runs" / method / "records_with_code.jsonl"
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["method"] = method
    df["valid"] = df["valid"].map(normalize_bool)
    for col in ["idx", "R", "cNBI", "Time", "rank_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["code"] = df["code"].fillna("")
    return df


def load_generic_records() -> pd.DataFrame:
    rows = [read_records_with_code(method) for method in GENERIC_METHODS]
    df = pd.concat(rows, ignore_index=True)
    feats = df["code"].apply(extract_features)
    feat_df = pd.DataFrame(list(feats))
    for col in feat_df.columns:
        df[f"feat_{col}"] = feat_df[col].astype(bool)
    df["family"] = feat_df.apply(lambda row: classify_family(row.to_dict()), axis=1)
    return add_e26f_flags(df)


def learn_families(df: pd.DataFrame, train_cutoff: int) -> Tuple[pd.DataFrame, Dict[str, float], str]:
    train = df[df["idx"].le(train_cutoff)].copy()
    train = add_global_score(train, "train_score")
    valid = train[train["valid"] & train["train_score"].notna()].copy()
    if valid.empty:
        raise RuntimeError("No valid training records found")
    top_threshold = valid["train_score"].quantile(0.80)
    valid["top20"] = valid["train_score"] >= top_threshold
    t_min, t_max = valid["Time"].min(), valid["Time"].max()
    t_span = max(1e-9, float(t_max - t_min))

    rows: List[Dict[str, Any]] = []
    for family in FAMILY_ORDER:
        sub_all = train[train["family"].eq(family)]
        sub = valid[valid["family"].eq(family)]
        if sub_all.empty:
            rows.append(
                {
                    "family": family,
                    "n_total": 0,
                    "n_valid": 0,
                    "valid_rate": 0.0,
                    "top20_rate": 0.0,
                    "strict_rate": 0.0,
                    "loose_rate": 0.0,
                    "mean_score": 0.0,
                    "mean_R": np.nan,
                    "mean_cNBI": np.nan,
                    "mean_Time": np.nan,
                    "complexity_risk": 0.0,
                    "initial_credit": 0.05,
                }
            )
            continue
        valid_rate = len(sub) / max(1, len(sub_all))
        mean_time = float(sub["Time"].mean()) if len(sub) else float(t_max)
        time_penalty = (mean_time - float(t_min)) / t_span
        complexity_risk = float(
            sub_all[["feat_global_expensive", "feat_full_recompute_loop", "feat_component_refresh"]].mean(numeric_only=True).mean()
        )
        mean_score = float(sub["train_score"].mean()) if len(sub) else 0.0
        top20_rate = float(sub["top20"].mean()) if len(sub) else 0.0
        strict_rate = float(sub["strict_e26f_like"].mean()) if len(sub) else 0.0
        loose_rate = float(sub["loose_e26f_like"].mean()) if len(sub) else 0.0
        credit = (
            0.42 * mean_score
            + 0.22 * top20_rate
            + 0.18 * strict_rate
            + 0.08 * loose_rate
            + 0.10 * valid_rate
            - 0.12 * time_penalty
            - 0.10 * complexity_risk
        )
        rows.append(
            {
                "family": family,
                "n_total": int(len(sub_all)),
                "n_valid": int(len(sub)),
                "valid_rate": valid_rate,
                "top20_rate": top20_rate,
                "strict_rate": strict_rate,
                "loose_rate": loose_rate,
                "mean_score": mean_score,
                "mean_R": float(sub["R"].mean()) if len(sub) else np.nan,
                "mean_cNBI": float(sub["cNBI"].mean()) if len(sub) else np.nan,
                "mean_Time": mean_time if len(sub) else np.nan,
                "complexity_risk": complexity_risk,
                "initial_credit": float(max(0.01, credit)),
            }
        )
    fam = pd.DataFrame(rows).sort_values("initial_credit", ascending=False)
    credit = {row["family"]: float(row["initial_credit"]) for _, row in fam.iterrows()}
    return fam, credit, build_experience_block(fam, train_cutoff)


def build_experience_block(fam: pd.DataFrame, train_cutoff: int) -> str:
    top = fam.sort_values("initial_credit", ascending=False).head(4)
    bad = fam.sort_values("initial_credit", ascending=True).head(2)
    lines = [
        "HAST evidence block learned only from the first "
        f"{train_cutoff} nodes of ERA-like/MCTS-AHD-like/Clade-AHD-like/FunSearch-like/AlphaEvolve-like.",
        "Do not copy prior candidate code. Use these as search experience, not as fixed recipes.",
        "",
        "Promising families:",
    ]
    for _, row in top.iterrows():
        lines.append(
            "- "
            f"{row['family']}: credit={row['initial_credit']:.3f}, "
            f"top20={row['top20_rate']:.2f}, strict={row['strict_rate']:.2f}, "
            f"valid={row['valid_rate']:.2f}, mean_time={row['mean_Time']:.4f}. "
            f"{FAMILY_GUIDANCE.get(str(row['family']), '')}"
        )
    lines += ["", "Families to avoid or repair:"]
    for _, row in bad.iterrows():
        lines.append(
            "- "
            f"{row['family']}: credit={row['initial_credit']:.3f}, "
            f"complexity_risk={row['complexity_risk']:.2f}. "
            "Only use it if you convert the idea into cheap local updates."
        )
    lines += [
        "",
        "General rule learned from evidence: prefer cheap local neighbor/two-hop/frontier signals with heap/lazy updates; "
        "avoid per-step global recomputation and expensive centrality.",
    ]
    return "\n".join(lines)


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


def method_dir(run_name: str) -> Path:
    return RUNS_DIR / run_name


def load_hast_records(run_name: str) -> List[Dict[str, Any]]:
    return read_csv_records(method_dir(run_name) / "search_records.csv")


def save_hast_state(run_name: str, records: List[Dict[str, Any]], credit_trace: List[Dict[str, Any]]) -> None:
    out = method_dir(run_name)
    fieldnames = [
        "idx",
        "method",
        "stage",
        "node_id",
        "parent_id",
        "chosen_family",
        "actual_family",
        "valid",
        "error",
        "R",
        "cNBI",
        "Time",
        "rank_R",
        "rank_cNBI",
        "rank_Time",
        "rank_score",
        "strict_e26f_like",
        "loose_e26f_like",
        "code_hash",
        "candidate_file",
        "prompt_elapsed_s",
        "avg_hhi",
        "avg_pairdisc",
        "avg_top5_mass",
    ]
    write_csv(out / "search_records.csv", records, fieldnames=fieldnames)
    edges = [
        {
            "source": r.get("parent_id"),
            "target": r.get("node_id"),
            "idx": r.get("idx"),
            "valid": r.get("valid"),
            "chosen_family": r.get("chosen_family"),
            "actual_family": r.get("actual_family"),
        }
        for r in records
        if r.get("parent_id")
    ]
    write_csv(out / "tree_edges.csv", edges)
    write_csv(out / "family_credit_trace.csv", credit_trace)
    write_csv(TABLE_DIR / "hast_search_records.csv", records, fieldnames=fieldnames)


def stable_hash(text: str, n: int = 12) -> str:
    return SEARCH.stable_hash(text, n)


def rank_records(records: List[Dict[str, Any]]) -> None:
    SEARCH.rank_records(records)
    flags = add_e26f_flags(pd.DataFrame(records))
    for row, (_, flag_row) in zip(records, flags.iterrows()):
        row["strict_e26f_like"] = bool(flag_row.get("strict_e26f_like", False))
        row["loose_e26f_like"] = bool(flag_row.get("loose_e26f_like", False))


def best_valid(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    pool = [
        r
        for r in records
        if bool(r.get("valid")) and isinstance(r.get("rank_score"), (int, float)) and float(r.get("rank_score", -1)) >= 0
    ]
    if not pool:
        return None
    return max(pool, key=lambda r: float(r.get("rank_score", 0.0)))


def select_family(credits: Dict[str, float], counts: Counter, t: int, rng: random.Random) -> str:
    if rng.random() < 0.08:
        return rng.choice(FAMILY_ORDER)
    total = max(2, t)
    def ucb(family: str) -> float:
        return float(credits.get(family, 0.01)) + 0.12 * math.sqrt(math.log(total) / (counts[family] + 1))
    return max(FAMILY_ORDER, key=ucb)


def select_parent(records: List[Dict[str, Any]], family: str) -> Dict[str, Any]:
    rank_records(records)
    family_pool = [
        r
        for r in records
        if bool(r.get("valid"))
        and str(r.get("actual_family") or r.get("chosen_family") or "") == family
        and isinstance(r.get("rank_score"), (int, float))
    ]
    if family_pool:
        return max(family_pool, key=lambda r: float(r.get("rank_score", 0.0)))
    best = best_valid(records)
    if best is None:
        return records[0]
    return best


def update_credit(credits: Dict[str, float], family: str, record: Dict[str, Any]) -> float:
    old = float(credits.get(family, 0.05))
    if not record.get("valid"):
        reward = -0.20
    else:
        reward = float(record.get("rank_score") or 0.0)
        if record.get("strict_e26f_like"):
            reward += 0.25
        elif record.get("loose_e26f_like"):
            reward += 0.10
        time_val = float(record.get("Time") or 0.0)
        if time_val > 0.04:
            reward -= 0.08
        if str(record.get("actual_family")) in {"global_expensive", "component_heavy"}:
            reward -= 0.06
    new = max(0.005, 0.86 * old + 0.14 * reward)
    credits[family] = new
    return new


def prompt_top_summary(records: List[Dict[str, Any]], k: int = 4) -> List[Dict[str, Any]]:
    rank_records(records)
    pool = [r for r in records if bool(r.get("valid")) and isinstance(r.get("rank_score"), (int, float))]
    top = sorted(pool, key=lambda r: float(r.get("rank_score", 0.0)), reverse=True)[:k]
    return [
        {
            "node_id": r.get("node_id"),
            "family": r.get("actual_family") or r.get("chosen_family"),
            "score": round(float(r.get("rank_score", 0.0)), 4),
            "R": round(float(r.get("R", 0.0)), 6),
            "cNBI": round(float(r.get("cNBI", 0.0)), 4),
            "Time": round(float(r.get("Time", 0.0)), 5),
        }
        for r in top
    ]


def build_hast_prompt(
    parent: Dict[str, Any],
    records: List[Dict[str, Any]],
    chosen_family: str,
    credits: Dict[str, float],
    experience_block: str,
    max_parent_code_chars: int,
) -> List[Dict[str, str]]:
    top_summary = prompt_top_summary(records, k=4)
    parent_code = str(parent.get("code") or SEARCH.HDA_CODE)
    if max_parent_code_chars and len(parent_code) > max_parent_code_chars:
        keep_head = max(2000, int(max_parent_code_chars * 0.62))
        keep_tail = max(1000, max_parent_code_chars - keep_head)
        omitted = len(parent_code) - keep_head - keep_tail
        parent_code = (
            parent_code[:keep_head].rstrip()
            + f"\n\n# ... [parent code truncated: {omitted} chars omitted; keep the same degree_order(G) interface] ...\n\n"
            + parent_code[-keep_tail:].lstrip()
        )
    credit_snapshot = {f: round(float(credits.get(f, 0.0)), 4) for f in FAMILY_ORDER}
    user = f"""
We are running HAST (Harnessed Adaptive Search Tree), a harnessed LLM program search for iterative network dismantling.

Search framework: HAST
Framework instruction: free-code search with evidence-derived family credit. First learn from early generic search, then specialize.

Task:
- Write exactly one Python function: def degree_order(G):
- Input G is a NetworkX undirected graph.
- Return a full node-removal order as a Python list.
- Start from the parent program below and propose one new executable heuristic.
- Optimize the black-box evaluator: lower R, higher cNBI, lower runtime.
- The algorithm is iterative: after each selected node is removed, the graph state should influence later selections.
- Complexity guard: avoid all-pairs shortest paths, per-step betweenness, spectral methods, community detection, or full expensive global recomputation.
- Use only Python standard library, networkx, numpy, math, heapq, collections, itertools, and random.
- Do not read/write files, call the network, use subprocesses, or use hidden state.
- For deterministic tie-breaking, use str(u), not repr(u).
- Do not mention or copy any named previous discovered algorithm; return code only.

HAST learned experience:
{experience_block}

Current family credit snapshot:
{json.dumps(credit_snapshot, ensure_ascii=False)}

This round's target family:
- {chosen_family}: {FAMILY_GUIDANCE.get(chosen_family, '')}
- Make a focused mutation for this target family. Keep changes smaller than a rewrite unless the parent is clearly unsuitable.

Recent top HAST candidates:
{json.dumps(top_summary, ensure_ascii=False)}

Parent candidate:
```python
{parent_code}
```

Return only a Python code block or raw Python code containing degree_order(G).
"""
    system = (
        "You are an expert algorithm designer. Return only safe Python code. "
        "The code must be self-contained and define degree_order(G)."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def init_root(graph_payloads: List[Dict[str, Any]], budget_ratio: float, timeout_s: float) -> Dict[str, Any]:
    eval_result = SEARCH.evaluate_candidate_code(SEARCH.HDA_CODE, graph_payloads, budget_ratio=budget_ratio, timeout_s=timeout_s)
    if not eval_result.get("ok"):
        raise RuntimeError(f"HDA root failed: {eval_result.get('error')}")
    avg = eval_result["avg"]
    return {
        "idx": 0,
        "method": HAST_METHOD,
        "stage": "root",
        "node_id": ROOT_ID,
        "parent_id": "",
        "chosen_family": "simple_degree",
        "actual_family": "simple_degree",
        "valid": True,
        "error": "",
        "R": avg["R"],
        "cNBI": avg["cNBI"],
        "Time": avg["Time"],
        "rank_R": -1.0,
        "rank_cNBI": -1.0,
        "rank_Time": -1.0,
        "rank_score": -1.0,
        "strict_e26f_like": False,
        "loose_e26f_like": False,
        "code_hash": stable_hash(SEARCH.HDA_CODE, 16),
        "candidate_file": "",
        "prompt_elapsed_s": 0.0,
        "avg_hhi": avg.get("avg_hhi"),
        "avg_pairdisc": avg.get("avg_pairdisc"),
        "avg_top5_mass": avg.get("avg_top5_mass"),
        "code": SEARCH.HDA_CODE,
    }


def run_hast_search(args: argparse.Namespace, family_credit: Dict[str, float], experience_block: str) -> None:
    run_name = args.run_name
    out = method_dir(run_name)
    candidates = out / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    candidates.mkdir(parents=True, exist_ok=True)
    graph_payloads = SEARCH.load_search_graph_payloads()

    records = load_hast_records(run_name)
    if not records:
        records = [init_root(graph_payloads, args.budget_ratio, args.candidate_timeout_s)]
    else:
        for r in records:
            if r.get("idx") == 0:
                r["code"] = SEARCH.HDA_CODE
            elif r.get("candidate_file") and Path(str(r["candidate_file"])).exists():
                r["code"] = Path(str(r["candidate_file"])).read_text(encoding="utf-8")
            else:
                r["code"] = ""
    rank_records(records)

    credit_trace = read_csv_records(out / "family_credit_trace.csv")
    counts = Counter(str(r.get("chosen_family") or "") for r in records if int(r.get("idx") or 0) > 0)
    rng = random.Random(args.seed)
    llm_log_path = out / "llm_calls.jsonl"

    while len(records) <= args.nodes:
        rank_records(records)
        idx = len(records)
        chosen_family = select_family(family_credit, counts, idx, rng)
        counts[chosen_family] += 1
        parent = select_parent(records, chosen_family)
        prompt_id = f"HAST_{idx:04d}_{stable_hash(str(time.time()), 6)}"
        messages = build_hast_prompt(
            parent=parent,
            records=records,
            chosen_family=chosen_family,
            credits=family_credit,
            experience_block=experience_block,
            max_parent_code_chars=args.max_parent_code_chars,
        )

        raw_response = ""
        code = ""
        error = ""
        t_prompt = time.perf_counter()
        try:
            raw_response = SEARCH.call_llm_hard_timeout(
                messages,
                args.model,
                max_retries=args.llm_retries,
                timeout_s=args.llm_timeout_s,
                max_completion_tokens=args.max_completion_tokens,
                reasoning_effort=args.reasoning_effort,
            )
            code = SEARCH.extract_code(raw_response)
        except Exception as exc:  # noqa: BLE001
            error = f"llm: {type(exc).__name__}: {exc}"
        prompt_elapsed = time.perf_counter() - t_prompt

        candidate_file = candidates / f"candidate_{idx:04d}_{stable_hash(code or error, 16)}.py"
        candidate_file.write_text(code or f"# invalid: {error}\n", encoding="utf-8")

        valid = False
        avg: Dict[str, Any] = {}
        if not error:
            eval_result = SEARCH.evaluate_candidate_code(code, graph_payloads, budget_ratio=args.budget_ratio, timeout_s=args.candidate_timeout_s)
            if eval_result.get("ok"):
                valid = True
                avg = eval_result["avg"]
            else:
                error = str(eval_result.get("error", "unknown evaluation error"))

        features = extract_features(code)
        actual_family = classify_family(features)
        record = {
            "idx": idx,
            "method": HAST_METHOD,
            "stage": "search",
            "node_id": stable_hash(f"{idx}:{code}:{time.time()}", 12),
            "parent_id": parent.get("node_id", ROOT_ID),
            "chosen_family": chosen_family,
            "actual_family": actual_family,
            "valid": valid,
            "error": error,
            "R": avg.get("R"),
            "cNBI": avg.get("cNBI"),
            "Time": avg.get("Time"),
            "rank_R": -1.0,
            "rank_cNBI": -1.0,
            "rank_Time": -1.0,
            "rank_score": -1.0,
            "strict_e26f_like": False,
            "loose_e26f_like": False,
            "code_hash": stable_hash(code, 16),
            "candidate_file": str(candidate_file),
            "prompt_elapsed_s": prompt_elapsed,
            "avg_hhi": avg.get("avg_hhi"),
            "avg_pairdisc": avg.get("avg_pairdisc"),
            "avg_top5_mass": avg.get("avg_top5_mass"),
            "code": code,
        }
        records.append(record)
        rank_records(records)
        new_credit = update_credit(family_credit, chosen_family, records[-1])
        credit_trace.append(
            {
                "idx": idx,
                "chosen_family": chosen_family,
                "actual_family": actual_family,
                "valid": valid,
                "rank_score": records[-1].get("rank_score"),
                "strict_e26f_like": records[-1].get("strict_e26f_like"),
                "credit_after": new_credit,
                **{f"credit_{f}": family_credit.get(f, 0.0) for f in FAMILY_ORDER},
            }
        )

        with llm_log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "idx": idx,
                        "prompt_id": prompt_id,
                        "parent_id": parent.get("node_id"),
                        "chosen_family": chosen_family,
                        "actual_family": actual_family,
                        "messages": messages,
                        "response": raw_response,
                        "error": error,
                        "elapsed_s": prompt_elapsed,
                        "reasoning_effort": args.reasoning_effort,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        save_hast_state(run_name, records, credit_trace)
        if idx % 10 == 0 or idx == args.nodes:
            best = best_valid(records)
            print(
                f"[HAST] {idx}/{args.nodes} valid={sum(bool(r.get('valid')) for r in records)} "
                f"strict={sum(bool(r.get('strict_e26f_like')) for r in records)} "
                f"best={float(best.get('rank_score', 0.0)) if best else -1:.4f} family={chosen_family}",
                flush=True,
            )

    rank_records(records)
    best = best_valid(records)
    if best and best.get("candidate_file"):
        shutil.copyfile(str(best["candidate_file"]), out / "best_candidate.py")
    save_hast_state(run_name, records, credit_trace)


def first_step_at_k_hits(sub: pd.DataFrame, k: int, step_col: str) -> int:
    ordered = sub.sort_values(step_col)
    cum = ordered["strict_e26f_like"].fillna(False).astype(bool).cumsum()
    hit = ordered.loc[cum >= k, step_col]
    return int(hit.iloc[0]) if len(hit) else -1


def compare_with_heldout(run_name: str, generic: pd.DataFrame, train_cutoff: int) -> pd.DataFrame:
    hast = pd.read_csv(method_dir(run_name) / "search_records.csv")
    hast["method"] = HAST_METHOD
    hast["heldout_step"] = pd.to_numeric(hast["idx"], errors="coerce")
    hast = hast[hast["idx"].gt(0)].copy()
    held = generic[generic["idx"].gt(train_cutoff)].copy()
    held["heldout_step"] = held["idx"] - train_cutoff
    combo = pd.concat([hast, held], ignore_index=True, sort=False)
    combo["valid"] = combo["valid"].map(normalize_bool)
    for col in ["idx", "heldout_step", "R", "cNBI", "Time"]:
        combo[col] = pd.to_numeric(combo[col], errors="coerce")
    combo = add_e26f_flags(add_global_score(combo, "heldout_global_score"))

    rows: List[Dict[str, Any]] = []
    for method in [HAST_METHOD] + GENERIC_METHODS:
        sub = combo[combo["method"].eq(method)].sort_values("heldout_step")
        valid = sub[sub["valid"] & sub["heldout_global_score"].notna()]
        best = valid.sort_values("heldout_global_score", ascending=False).iloc[0] if len(valid) else None
        rows.append(
            {
                "method": method,
                "nodes": int(len(sub)),
                "valid": int(sub["valid"].sum()),
                "invalid": int((~sub["valid"]).sum()),
                "best_step": int(best["heldout_step"]) if best is not None else -1,
                "best_idx": int(best["idx"]) if best is not None else -1,
                "best_score": float(best["heldout_global_score"]) if best is not None else np.nan,
                "best_R": float(best["R"]) if best is not None else np.nan,
                "best_cNBI": float(best["cNBI"]) if best is not None else np.nan,
                "best_Time": float(best["Time"]) if best is not None else np.nan,
                "first_strict": first_step_at_k_hits(sub, 1, "heldout_step"),
                "first_5_strict": first_step_at_k_hits(sub, 5, "heldout_step"),
                "first_20_strict": first_step_at_k_hits(sub, 20, "heldout_step"),
                "strict_count": int(sub["strict_e26f_like"].sum()),
                "loose_count": int(sub["loose_e26f_like"].sum()),
                "invalid_rate": float((~sub["valid"]).mean()) if len(sub) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    combo.to_csv(TABLE_DIR / "hast_heldout_combined_records.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "hast_vs_heldout_baselines.csv", index=False, encoding="utf-8-sig")
    return summary


def plot_best_so_far() -> None:
    df = pd.read_csv(TABLE_DIR / "hast_heldout_combined_records.csv")
    setup_style()
    colors = {
        HAST_METHOD: "#D62728",
        "ERA-like": "#4C78A8",
        "MCTS-AHD-like": "#59A14F",
        "Clade-AHD-like": "#F28E2B",
        "FunSearch-like": "#B07AA1",
        "AlphaEvolve-like": "#9C755F",
    }
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for method in [HAST_METHOD] + GENERIC_METHODS:
        sub = df[df["method"].eq(method)].sort_values("heldout_step")
        y = sub["heldout_global_score"].fillna(-1).cummax()
        ax.plot(
            sub["heldout_step"],
            y,
            label=method,
            color=colors.get(method, "#333333"),
            lw=2.5 if method == HAST_METHOD else 1.4,
        )
    ax.set_xlabel("Specialization / held-out candidate step")
    ax.set_ylabel("Best-so-far triple score")
    ax.set_title("HAST specialization vs generic held-out search")
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hast_vs_heldout_best_so_far.png")
    fig.savefig(FIG_DIR / "hast_vs_heldout_best_so_far.pdf")
    plt.close(fig)


def plot_strict_density() -> None:
    df = pd.read_csv(TABLE_DIR / "hast_heldout_combined_records.csv")
    setup_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for method in [HAST_METHOD] + GENERIC_METHODS:
        sub = df[df["method"].eq(method)].sort_values("heldout_step")
        y = sub["strict_e26f_like"].fillna(False).astype(bool).cumsum()
        ax.plot(sub["heldout_step"], y, label=method, lw=2.5 if method == HAST_METHOD else 1.4)
    ax.set_xlabel("Specialization / held-out candidate step")
    ax.set_ylabel("Cumulative strict e26f-like hits")
    ax.set_title("Strong-family density during specialization")
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hast_strict_family_density.png")
    fig.savefig(FIG_DIR / "hast_strict_family_density.pdf")
    plt.close(fig)


def plot_family_credit(run_name: str) -> None:
    path = method_dir(run_name) / "family_credit_trace.csv"
    if not path.exists() or path.stat().st_size == 0:
        return
    df = pd.read_csv(path)
    setup_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for family in FAMILY_ORDER:
        col = f"credit_{family}"
        if col in df.columns:
            ax.plot(df["idx"], df[col], label=family, lw=1.4)
    ax.set_xlabel("HAST candidate index")
    ax.set_ylabel("Family credit")
    ax.set_title("Online credit assignment over HAST search")
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hast_family_credit_trace.png")
    fig.savefig(FIG_DIR / "hast_family_credit_trace.pdf")
    plt.close(fig)


def plot_learned_family_heatmap(fam: pd.DataFrame) -> None:
    setup_style()
    metrics = ["valid_rate", "top20_rate", "strict_rate", "loose_rate", "mean_score", "complexity_risk", "initial_credit"]
    mat = fam.set_index("family").reindex(FAMILY_ORDER)[metrics].fillna(0.0)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    im = ax.imshow(mat.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=35, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_title("Families learned from first 200 generic-search nodes")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hast_learned_family_heatmap.png")
    fig.savefig(FIG_DIR / "hast_learned_family_heatmap.pdf")
    plt.close(fig)


def plot_tree(run_name: str) -> None:
    path = method_dir(run_name) / "search_records.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if len(df) <= 1:
        return
    setup_style()
    graph = nx.DiGraph()
    by_id = {}
    for _, row in df.iterrows():
        node = str(row["node_id"])
        by_id[node] = row
        graph.add_node(node, idx=int(row["idx"]), family=str(row.get("actual_family") or row.get("chosen_family") or ""))
        parent = str(row.get("parent_id") or "")
        if parent:
            graph.add_edge(parent, node)
    depths = {ROOT_ID: 0}
    for node in nx.topological_sort(graph):
        for child in graph.successors(node):
            depths[child] = depths.get(node, 0) + 1
    layers: Dict[int, List[str]] = defaultdict(list)
    for node in graph.nodes:
        layers[depths.get(node, 0)].append(node)
    pos = {}
    for depth, nodes in layers.items():
        for i, node in enumerate(nodes):
            pos[node] = (depth, i - (len(nodes) - 1) / 2)
    palette = dict(zip(FAMILY_ORDER, plt.cm.tab10.colors[: len(FAMILY_ORDER)]))
    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.18, arrows=False, width=0.6)
    colors = [palette.get(graph.nodes[n].get("family"), (0.4, 0.4, 0.4)) for n in graph.nodes]
    sizes = [70 if n == ROOT_ID else 18 for n in graph.nodes]
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colors, node_size=sizes, linewidths=0.0)
    ax.set_title("HAST search tree colored by inferred family")
    ax.set_axis_off()
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[f], markersize=6, label=f)
        for f in FAMILY_ORDER
        if f in set(df["actual_family"].dropna().astype(str))
    ]
    ax.legend(handles=handles, ncol=2, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hast_search_tree_by_family.png")
    fig.savefig(FIG_DIR / "hast_search_tree_by_family.pdf")
    plt.close(fig)


def evaluate_12graph(run_name: str) -> None:
    eval_mod = load_module(SOURCE_12GRAPH, "source_12graph_for_hast")
    out_dir = ROOT / "final_12graph_eval"
    record_dir = out_dir / "records"
    fig_dir = out_dir / "figures"
    for p in [out_dir, record_dir, fig_dir]:
        p.mkdir(parents=True, exist_ok=True)

    hast_code = (method_dir(run_name) / "best_candidate.py").read_text(encoding="utf-8")
    hast_fn = SEARCH.compile_degree_order(hast_code)
    methods = [HAST_METHOD] + GENERIC_METHODS + ["E26F", "HDA"]
    summaries: List[Dict[str, Any]] = []

    for dataset in eval_mod.EVAL.DATASETS:
        graph = eval_mod.EVAL.read_graph(dataset)
        rate = eval_mod.EVAL.DATASET_RATES[dataset]
        orders: Dict[str, Tuple[List[Any], float, str]] = {}
        t0 = time.perf_counter()
        orders[HAST_METHOD] = (list(hast_fn(graph.copy())), time.perf_counter() - t0, "hast_best")
        for method in GENERIC_METHODS:
            try:
                code = (SOURCE_ROOT / "runs" / method / "best_candidate.py").read_text(encoding="utf-8")
                fn = SEARCH.compile_degree_order(code)
                t0 = time.perf_counter()
                orders[method] = (list(fn(graph.copy())), time.perf_counter() - t0, "generic_best")
            except Exception as exc:  # noqa: BLE001
                print(f"[12graph warn] {dataset}/{method}: {type(exc).__name__}: {exc}", flush=True)
        t0 = time.perf_counter()
        orders["E26F"] = (
            eval_mod.EVAL.DACTS.degree_order_by_config(graph, eval_mod.EVAL.DACTS.e26f_config(), budget_ratio=rate),
            time.perf_counter() - t0,
            "online",
        )
        t0 = time.perf_counter()
        orders["HDA"] = (eval_mod.EVAL.hda_simple_order(graph, rate), time.perf_counter() - t0, "online")

        for method, (order, elapsed, source) in orders.items():
            metrics = eval_mod.EVAL.compute_metrics(graph, order, rate=rate, method_time=elapsed)
            if metrics.empty:
                continue
            x = metrics["removal_ratio"].to_numpy(dtype=float)
            summaries.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "source": source,
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "rate": rate,
                    "R": eval_mod.EVAL.auc_mean(x, metrics["GCC"].to_numpy(dtype=float)),
                    "auc_cNBI": eval_mod.EVAL.auc_mean(x, metrics["cNBI"].to_numpy(dtype=float)),
                    "final_ACC": float(metrics["ACC"].iloc[-1]),
                    "final_NCC": float(metrics["NCC"].iloc[-1]),
                    "final_cNBI": float(metrics["cNBI"].iloc[-1]),
                    "time_s": elapsed,
                }
            )
            metrics.insert(0, "method", method)
            metrics.insert(0, "dataset", dataset)
            metrics.to_csv(record_dir / f"{dataset}_{method}_metrics.csv", index=False, encoding="utf-8-sig")
        print(f"[12graph] {dataset} done", flush=True)

    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "hast_12graph_summary.csv", index=False, encoding="utf-8-sig")
    mean = summary.groupby("method")[["R", "auc_cNBI", "time_s"]].mean(numeric_only=True).reset_index()
    mean.to_csv(TABLE_DIR / "hast_12graph_summary.csv", index=False, encoding="utf-8-sig")

    setup_style()
    colors = {HAST_METHOD: "#D62728", "E26F": "#111111", "HDA": "#E377C2"}
    colors.update({"ERA-like": "#4C78A8", "MCTS-AHD-like": "#59A14F", "Clade-AHD-like": "#F28E2B", "FunSearch-like": "#B07AA1", "AlphaEvolve-like": "#9C755F"})
    for metric, ylabel, filename in [
        ("R", "Mean R (lower is better)", "hast_12graph_R_bar.png"),
        ("auc_cNBI", "Mean AUC-cNBI (higher is better)", "hast_12graph_auc_cNBI_bar.png"),
        ("time_s", "Mean ordering time (s)", "hast_12graph_time_bar.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        data = mean.sort_values(metric, ascending=(metric != "auc_cNBI"))
        ax.bar(data["method"], data[metric], color=[colors.get(m, "#777777") for m in data["method"]])
        ax.set_ylabel(ylabel)
        ax.set_title(f"12-graph comparison: {metric}")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(fig_dir / filename)
        fig.savefig(FIG_DIR / filename)
        plt.close(fig)


def write_report(run_name: str, fam: pd.DataFrame, summary: pd.DataFrame, run_12graph: bool) -> None:
    hast_row = summary[summary["method"].eq(HAST_METHOD)].iloc[0] if HAST_METHOD in set(summary["method"]) else None
    held_mean_invalid = summary[summary["method"].ne(HAST_METHOD)]["invalid_rate"].mean()
    lines = [
        "# HAST 200-Train / 300-Specialize Experiment",
        "",
        "## 设置",
        "",
        "- HAST = Harnessed Adaptive Search Tree.",
        "- 训练只使用五个通用框架前 200 节点，不使用 DACTS/DACTS-rerun。",
        "- 对照为五个通用框架的 201-500 held-out 区间。",
        "- HAST 仍生成自由 `degree_order(G)` 程序，经验块只提供归纳出的机制方向，不贴已有强候选代码。",
        "",
        "## Learned families",
        "",
        fam.to_markdown(index=False),
        "",
        "## Held-out comparison",
        "",
        summary.to_markdown(index=False),
        "",
        "## 初步判断",
        "",
    ]
    if hast_row is not None:
        lines.append(
            f"- HAST strict_count={int(hast_row['strict_count'])}, "
            f"first_20_strict={int(hast_row['first_20_strict'])}, "
            f"invalid_rate={float(hast_row['invalid_rate']):.3f}；"
            f"generic held-out 平均 invalid_rate={held_mean_invalid:.3f}。"
        )
    lines += [
        "- 如果 HAST 曲线更早上升或 strict family 更密集，说明前 200 节点经验可以指导后续专精。",
        "- 如果 HAST 最终 best 没超过所有通用框架，也不能直接判失败；本实验主张的是更有目标、更可解释的搜索过程。",
    ]
    if run_12graph and (TABLE_DIR / "hast_12graph_summary.csv").exists():
        mean12 = pd.read_csv(TABLE_DIR / "hast_12graph_summary.csv")
        lines += ["", "## 12 图复核", "", mean12.to_markdown(index=False)]
    (REPORT_DIR / "hast_experiment_summary_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=300)
    parser.add_argument("--train-cutoff", type=int, default=200)
    parser.add_argument("--run-name", default="HAST")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--llm-timeout-s", type=float, default=90.0)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--candidate-timeout-s", type=float, default=35.0)
    parser.add_argument("--budget-ratio", type=float, default=0.30)
    parser.add_argument("--max-parent-code-chars", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--only-learn", action="store_true")
    parser.add_argument("--eval-12graph", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    if args.reset and method_dir(args.run_name).exists():
        shutil.rmtree(method_dir(args.run_name))
    generic = load_generic_records()
    fam, credit, experience_block = learn_families(generic, args.train_cutoff)
    fam.to_csv(TABLE_DIR / "hast_learned_families.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "hast_experience_block.txt").write_text(experience_block, encoding="utf-8")
    plot_learned_family_heatmap(fam)
    if args.only_learn:
        print((TABLE_DIR / "hast_learned_families.csv").resolve(), flush=True)
        return

    if not args.skip_search:
        run_hast_search(args, credit, experience_block)
    summary = compare_with_heldout(args.run_name, generic, args.train_cutoff)
    plot_best_so_far()
    plot_strict_density()
    plot_family_credit(args.run_name)
    plot_tree(args.run_name)
    if args.eval_12graph:
        evaluate_12graph(args.run_name)
    write_report(args.run_name, fam, summary, args.eval_12graph)
    print((REPORT_DIR / "hast_experiment_summary_cn.md").resolve(), flush=True)


if __name__ == "__main__":
    main()
