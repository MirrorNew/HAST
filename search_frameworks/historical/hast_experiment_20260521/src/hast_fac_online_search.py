# -*- coding: utf-8 -*-
"""HAST-FAC online search.

HAST-FAC tests a stronger credit-assignment hypothesis:
reward a candidate only for the *fracture advantage* it creates over HDA,
instead of rewarding absolute cNBI/R directly.

This script is intentionally compact and reuses the existing evaluator/sandbox.
It does not seed the search with manually written FAC prototypes; the root is
plain HDA, and all search candidates are generated online by the LLM.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import shutil
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
ABLATION = WORKSPACE / "research" / "tree_search_ablation_20260520"
SOURCE_SEARCH = ABLATION / "src" / "ablation_search.py"
SOURCE_12GRAPH = ABLATION / "src" / "evaluate_final_12graphs.py"

RUN_DIR = ROOT / "runs" / "HAST-FAC"
CAND_DIR = RUN_DIR / "candidates"
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"

DEFAULT_PROXY_DATASETS = ["CEnew", "crime", "Yeast", "Grid", "hamster", "Powerlaw_500"]

FAC_FAMILIES = [
    "frontier_weak_tie",
    "anti_redundancy",
    "phase_adaptive",
    "component_boundary_proxy",
    "twohop_advantage",
    "core_frontier",
]

FAMILY_GUIDANCE = {
    "frontier_weak_tie": (
        "Add cheap local frontier/weak-tie signals: nodes whose neighbors expose many outside second-hop nodes "
        "or whose adjacent nodes are low-degree bridges. Avoid connected_components inside the loop."
    ),
    "anti_redundancy": (
        "Penalize redundant clique-like neighborhoods using cheap sampled neighbor-neighbor edges. Reward exposure "
        "to different outside neighborhoods, not just high degree."
    ),
    "phase_adaptive": (
        "Use different local scoring weights in early/middle/late removal phases. Early may target hubs, middle "
        "should target boundary/frontier nodes, late should avoid wasting steps inside already fragmented parts."
    ),
    "component_boundary_proxy": (
        "Approximate component-boundary pressure locally without recomputing components each step. Use frontier, "
        "outside-neighbor count, weak ties, and local exposure as proxies."
    ),
    "twohop_advantage": (
        "Use two-hop exposure, but only when it creates advantage over HDA. Avoid merely re-ranking high-degree "
        "nodes with neighbor degree."
    ),
    "core_frontier": (
        "Use cheap core-like or shell-like proxies combined with frontier exposure. Do not recompute k-core every step."
    ),
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SEARCH = load_module(SOURCE_SEARCH, "hast_fac_search_runtime")
E12 = load_module(SOURCE_12GRAPH, "hast_fac_eval12_runtime")


def ensure_dirs() -> None:
    for path in [RUN_DIR, CAND_DIR, TABLE_DIR, FIG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def configure_run(run_name: str) -> None:
    global RUN_DIR, CAND_DIR
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in run_name).strip("._")
    if not safe:
        safe = "HAST-FAC"
    RUN_DIR = ROOT / "runs" / safe
    CAND_DIR = RUN_DIR / "candidates"


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


def stable_hash(text: str, n: int = 12) -> str:
    return SEARCH.stable_hash(text, n)


def auc_mean(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float(np.mean(y)) if len(y) else 0.0
    return float(np.trapezoid(y, x) / max(1e-9, x[-1] - x[0]))


def value_at(curve: pd.DataFrame, col: str, ratio: float) -> float:
    idx = (curve["removal_ratio"] - ratio).abs().idxmin()
    return float(curve.loc[idx, col])


def boolish(x: Any) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}


def load_existing_records() -> List[Dict[str, Any]]:
    path = RUN_DIR / "search_records.csv"
    if not path.exists() or path.stat().st_size == 0:
        return []
    df = pd.read_csv(path)
    rows = df.where(pd.notna(df), None).to_dict("records")
    for r in rows:
        if r.get("idx") == 0:
            r["code"] = SEARCH.HDA_CODE
        elif r.get("candidate_file") and Path(str(r["candidate_file"])).exists():
            r["code"] = Path(str(r["candidate_file"])).read_text(encoding="utf-8")
        else:
            r["code"] = ""
    return rows


def search_rank_fields(records: List[Dict[str, Any]]) -> None:
    SEARCH.rank_records(records)


def compute_hda_proxy_baselines(proxy_datasets: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for dataset in proxy_datasets:
        graph = E12.EVAL.read_graph(dataset)
        rate = E12.EVAL.DATASET_RATES[dataset]
        t0 = time.perf_counter()
        order = E12.EVAL.hda_simple_order(graph.copy(), rate)
        elapsed = time.perf_counter() - t0
        metrics = E12.EVAL.compute_metrics(graph, order, rate=rate, method_time=elapsed)
        x = metrics["removal_ratio"].to_numpy(dtype=float)
        out[dataset] = {
            "auc_cNBI": auc_mean(x, metrics["cNBI"].to_numpy(dtype=float)),
            "R": float(metrics["GCC"].mean()),
            "cNBI20": value_at(metrics, "cNBI", 0.20),
            "NCC20": value_at(metrics, "NCC", 0.20),
            "GCC20": value_at(metrics, "GCC", 0.20),
        }
    return out


def evaluate_proxy_fac(code: str, proxy_datasets: List[str], hda_base: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    fn = SEARCH.compile_degree_order(code)
    rows: List[Dict[str, Any]] = []
    for dataset in proxy_datasets:
        graph = E12.EVAL.read_graph(dataset)
        rate = E12.EVAL.DATASET_RATES[dataset]
        t0 = time.perf_counter()
        order = list(fn(graph.copy()))
        elapsed = time.perf_counter() - t0
        metrics = E12.EVAL.compute_metrics(graph, order, rate=rate, method_time=elapsed)
        x = metrics["removal_ratio"].to_numpy(dtype=float)
        auc = auc_mean(x, metrics["cNBI"].to_numpy(dtype=float))
        R = float(metrics["GCC"].mean())
        c20 = value_at(metrics, "cNBI", 0.20)
        n20 = value_at(metrics, "NCC", 0.20)
        g20 = value_at(metrics, "GCC", 0.20)
        base = hda_base[dataset]
        rows.append(
            {
                "dataset": dataset,
                "auc_cNBI": auc,
                "R": R,
                "time_s": elapsed,
                "fac_auc_adv": auc - base["auc_cNBI"],
                "fac_R_adv": base["R"] - R,
                "fac_cNBI20_adv": c20 - base["cNBI20"],
                "fac_NCC20_adv": n20 - base["NCC20"],
                "fac_GCC20_adv": base["GCC20"] - g20,
            }
        )
    df = pd.DataFrame(rows)
    early = 0.45 * df["fac_cNBI20_adv"] + 0.35 * df["fac_NCC20_adv"] + 0.20 * (100.0 * df["fac_GCC20_adv"])
    return {
        "proxy_R": float(df["R"].mean()),
        "proxy_auc_cNBI": float(df["auc_cNBI"].mean()),
        "proxy_time_s": float(df["time_s"].mean()),
        "fac_auc_adv": float(df["fac_auc_adv"].mean()),
        "fac_R_adv": float(df["fac_R_adv"].mean()),
        "early_fac": float(early.mean()),
        "fac_worst_auc_adv": float(df["fac_auc_adv"].min()),
        "proxy_detail": rows,
    }


def fac_total_score(record: Dict[str, Any]) -> float:
    if not boolish(record.get("valid")):
        return -1e9
    fac = float(record.get("fac_auc_adv") or 0.0)
    early = float(record.get("early_fac") or 0.0)
    worst = float(record.get("fac_worst_auc_adv") or 0.0)
    search_rank = float(record.get("rank_score") or 0.0)
    proxy_time = float(record.get("proxy_time_s") or 0.0)
    search_time = float(record.get("Time") or 0.0)
    # FAC-T: fracture advantage must pass an efficiency gate. The previous FAC
    # score under-penalized time and drifted toward slow local-neighborhood scans.
    benefit = 0.48 * fac + 0.20 * early + 0.10 * worst + 6.0 * search_rank
    time_penalty = (
        12.0 * math.log1p(max(0.0, proxy_time) / 0.70)
        + 180.0 * max(0.0, search_time - 0.022)
        + 8.0 * max(0.0, proxy_time - 1.20) ** 2
    )
    if proxy_time > 1.80:
        time_penalty += 16.0 + 10.0 * (proxy_time - 1.80)
    if search_time > 0.032:
        time_penalty += 10.0 + 500.0 * (search_time - 0.032)
    return benefit - time_penalty


def init_root(graph_payloads: List[Dict[str, Any]], proxy_datasets: List[str], hda_base: Dict[str, Dict[str, float]], args: argparse.Namespace) -> Dict[str, Any]:
    search_eval = SEARCH.evaluate_candidate_code(SEARCH.HDA_CODE, graph_payloads, budget_ratio=args.budget_ratio, timeout_s=args.candidate_timeout_s)
    if not search_eval.get("ok"):
        raise RuntimeError(f"HDA root failed: {search_eval.get('error')}")
    proxy_eval = evaluate_proxy_fac(SEARCH.HDA_CODE, proxy_datasets, hda_base)
    avg = search_eval["avg"]
    row = {
        "idx": 0,
        "stage": "root",
        "node_id": "hda_root",
        "parent_id": "",
        "target_family": "HDA",
        "valid": True,
        "error": "",
        "R": avg["R"],
        "cNBI": avg["cNBI"],
        "Time": avg["Time"],
        "rank_R": -1.0,
        "rank_cNBI": -1.0,
        "rank_Time": -1.0,
        "rank_score": -1.0,
        "fac_score": 0.0,
        "code_hash": stable_hash(SEARCH.HDA_CODE, 16),
        "candidate_file": "",
        "prompt_elapsed_s": 0.0,
        "code": SEARCH.HDA_CODE,
    }
    row.update({k: v for k, v in proxy_eval.items() if k != "proxy_detail"})
    return row


def select_family(credits: Dict[str, float], counts: Counter, idx: int, rng: random.Random) -> str:
    if idx <= 6:
        return FAC_FAMILIES[(idx - 1) % len(FAC_FAMILIES)]
    if rng.random() < 0.12:
        return rng.choice(FAC_FAMILIES)
    total = max(2, idx)
    recent_penalty = {f: 0.0 for f in FAC_FAMILIES}
    for family in FAC_FAMILIES:
        recent_penalty[family] = 0.10 * max(0, counts[family] - idx / len(FAC_FAMILIES))
    return max(
        FAC_FAMILIES,
        key=lambda f: float(credits.get(f, 0.0)) + 0.28 * math.sqrt(math.log(total) / (counts[f] + 1.0)) - recent_penalty[f],
    )


def select_parent(records: List[Dict[str, Any]], family: str, rng: random.Random) -> Dict[str, Any]:
    valid = [r for r in records if boolish(r.get("valid"))]
    if len(valid) <= 1:
        return records[0]
    fam_pool = [r for r in valid if r.get("target_family") == family and int(r.get("idx") or 0) > 0]
    pool = fam_pool if fam_pool else valid
    ranked = sorted(pool, key=fac_total_score, reverse=True)
    if len(ranked) > 3 and rng.random() < 0.18:
        return rng.choice(ranked[: min(6, len(ranked))])
    return ranked[0]


def update_credit(credits: Dict[str, float], family: str, record: Dict[str, Any]) -> float:
    old = float(credits.get(family, 0.0))
    if not boolish(record.get("valid")):
        reward = -8.0
    else:
        reward = fac_total_score(record) / 60.0
        if float(record.get("fac_worst_auc_adv") or 0.0) < -5.0:
            reward -= 0.25
        if float(record.get("proxy_time_s") or 0.0) > 2.0:
            reward -= 0.10
    new = 0.82 * old + 0.18 * reward
    credits[family] = new
    return new


def summarize_top(records: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    valid = [r for r in records if boolish(r.get("valid")) and int(r.get("idx") or 0) > 0]
    top = sorted(valid, key=fac_total_score, reverse=True)[:k]
    return [
        {
            "idx": int(r.get("idx") or 0),
            "family": r.get("target_family"),
            "fac_score": round(float(r.get("fac_score") or 0.0), 3),
            "fac_auc_adv": round(float(r.get("fac_auc_adv") or 0.0), 3),
            "early_fac": round(float(r.get("early_fac") or 0.0), 3),
            "worst_adv": round(float(r.get("fac_worst_auc_adv") or 0.0), 3),
            "proxy_auc": round(float(r.get("proxy_auc_cNBI") or 0.0), 3),
            "search_R": round(float(r.get("R") or 0.0), 6),
            "search_cNBI": round(float(r.get("cNBI") or 0.0), 3),
            "proxy_time": round(float(r.get("proxy_time_s") or 0.0), 3),
        }
        for r in top
    ]


def truncate_code(code: str, limit: int) -> str:
    if len(code) <= limit:
        return code
    head = int(limit * 0.65)
    tail = limit - head
    return code[:head].rstrip() + "\n# ... parent truncated ...\n" + code[-tail:].lstrip()


def build_prompt(parent: Dict[str, Any], records: List[Dict[str, Any]], family: str, credits: Dict[str, float], args: argparse.Namespace) -> List[Dict[str, str]]:
    parent_code = truncate_code(str(parent.get("code") or SEARCH.HDA_CODE), args.max_parent_code_chars)
    top = summarize_top(records, k=5)
    credit_snapshot = {k: round(v, 4) for k, v in credits.items()}
    user = f"""
We are running HAST-FAC: Harnessed Adaptive Search Tree with Fracture Advantage Credit.

Goal:
- Generate one executable iterative network dismantling heuristic.
- Define exactly one function: degree_order(G)
- Input: NetworkX undirected graph G.
- Output: a full node removal order list.

Important: root is plain HDA. Do not copy named discovered algorithms.

FAC credit principle:
- HDA already removes high-degree nodes well. Do not merely imitate HDA.
- A mutation receives credit only if it creates fracture advantage over HDA:
  FAC = AUC_cNBI(candidate) - AUC_cNBI(HDA)
  EarlyFAC = 0.45*Delta_cNBI_at_20pct + 0.35*Delta_NCC_at_20pct + 0.20*Delta_minus_GCC_at_20pct
- Prefer changes that improve early cNBI/NCC versus HDA, not just absolute degree score.
- FAC-T efficiency gate: a slow candidate loses credit even if its curve is good. Prefer proxy_time below 1.2s
  and search-graph Time below 0.022s. Avoid large nested second-hop scans and repeated set construction.

Complexity guard:
- The algorithm is iterative: after removing each node, update the graph and later choices.
- Runtime must stay below O(N^2) in spirit for sparse large graphs.
- Avoid all-pairs shortest paths, betweenness, eigen/spectral, PageRank, community detection, or per-step full component recomputation.
- Use cheap local signals, heaps/lazy updates, sampled neighborhood redundancy, frontier exposure, weak ties, phase-aware weights.
- Keep the scoring formula compact. Use capped neighbor samples such as nbrs[:48] or early exits for high-degree nodes.
- Use str(u), not repr(u), for deterministic tie-breaking.

Current target FAC family:
- {family}: {FAMILY_GUIDANCE[family]}

Current FAC family credits:
{json.dumps(credit_snapshot, ensure_ascii=False)}

Top HAST-FAC candidates so far:
{json.dumps(top, ensure_ascii=False)}

Parent candidate:
```python
{parent_code}
```

Make one focused mutation. Return only Python code defining degree_order(G).
"""
    return [
        {"role": "system", "content": "You are an expert algorithm designer. Return only safe Python code containing degree_order(G)."},
        {"role": "user", "content": user},
    ]


FIELDNAMES = [
    "idx", "stage", "node_id", "parent_id", "target_family", "valid", "error",
    "R", "cNBI", "Time", "rank_R", "rank_cNBI", "rank_Time", "rank_score",
    "proxy_R", "proxy_auc_cNBI", "proxy_time_s", "fac_auc_adv", "fac_R_adv",
    "early_fac", "fac_worst_auc_adv", "fac_score", "code_hash",
    "candidate_file", "prompt_elapsed_s",
]


def save_state(records: List[Dict[str, Any]], credit_trace: List[Dict[str, Any]]) -> None:
    write_csv(RUN_DIR / "search_records.csv", records, FIELDNAMES)
    write_csv(TABLE_DIR / f"{RUN_DIR.name}_search_records.csv", records, FIELDNAMES)
    edges = [
        {"source": r.get("parent_id"), "target": r.get("node_id"), "idx": r.get("idx"), "target_family": r.get("target_family"), "valid": r.get("valid")}
        for r in records
        if r.get("parent_id")
    ]
    write_csv(RUN_DIR / "tree_edges.csv", edges)
    write_csv(RUN_DIR / "family_credit_trace.csv", credit_trace)


def plot_progress(records: List[Dict[str, Any]], credit_trace: List[Dict[str, Any]]) -> None:
    setup = {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
    }
    plt.rcParams.update(setup)
    df = pd.DataFrame(records)
    df = df[df["idx"].astype(int).gt(0)].copy()
    if df.empty:
        return
    df["fac_score"] = pd.to_numeric(df["fac_score"], errors="coerce")
    df["fac_auc_adv"] = pd.to_numeric(df["fac_auc_adv"], errors="coerce")
    df["proxy_auc_cNBI"] = pd.to_numeric(df["proxy_auc_cNBI"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(df["idx"], df["fac_score"].cummax(), color="#D62728", lw=2.2, label="best FAC score")
    ax.plot(df["idx"], df["proxy_auc_cNBI"].cummax(), color="#4C78A8", lw=1.5, label="best proxy AUC-cNBI")
    ax.set_xlabel("candidate idx")
    ax.set_title("HAST-FAC online progress")
    ax.legend(frameon=False)
    ax = axes[1]
    ct = pd.DataFrame(credit_trace)
    for family in FAC_FAMILIES:
        col = f"credit_{family}"
        if col in ct.columns:
            ax.plot(ct["idx"], ct[col], label=family, lw=1.3)
    ax.set_xlabel("candidate idx")
    ax.set_title("Family credit")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{RUN_DIR.name}_online_progress.png")
    fig.savefig(FIG_DIR / f"{RUN_DIR.name}_online_progress.pdf")
    plt.close(fig)


def run_search(args: argparse.Namespace) -> None:
    ensure_dirs()
    rng = random.Random(args.seed)
    proxy_datasets = [x.strip() for x in args.proxy_datasets.split(",") if x.strip()]
    graph_payloads = SEARCH.load_search_graph_payloads()
    hda_base = compute_hda_proxy_baselines(proxy_datasets)
    records = load_existing_records()
    if not records:
        records = [init_root(graph_payloads, proxy_datasets, hda_base, args)]
    search_rank_fields(records)

    credit_trace: List[Dict[str, Any]] = []
    ct_path = RUN_DIR / "family_credit_trace.csv"
    if ct_path.exists() and ct_path.stat().st_size:
        credit_trace = pd.read_csv(ct_path).where(pd.notna(pd.read_csv(ct_path)), None).to_dict("records")

    credits = {family: 0.0 for family in FAC_FAMILIES}
    for r in records:
        fam = str(r.get("target_family") or "")
        if fam in credits and int(r.get("idx") or 0) > 0:
            update_credit(credits, fam, r)
    counts = Counter(str(r.get("target_family") or "") for r in records if int(r.get("idx") or 0) > 0)

    llm_log = RUN_DIR / "llm_calls.jsonl"
    while max(int(r.get("idx") or 0) for r in records) < args.nodes:
        search_rank_fields(records)
        idx = max(int(r.get("idx") or 0) for r in records) + 1
        family = select_family(credits, counts, idx, rng)
        counts[family] += 1
        parent = select_parent(records, family, rng)
        messages = build_prompt(parent, records, family, credits, args)

        raw_response = ""
        code = ""
        error = ""
        t0 = time.perf_counter()
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
        prompt_elapsed = time.perf_counter() - t0

        code_hash = stable_hash(code or error, 16)
        candidate_file = CAND_DIR / f"candidate_{idx:04d}_{code_hash}.py"
        candidate_file.write_text(code or f"# invalid: {error}\n", encoding="utf-8")

        valid = False
        avg: Dict[str, Any] = {}
        proxy: Dict[str, Any] = {}
        if not error:
            search_eval = SEARCH.evaluate_candidate_code(code, graph_payloads, budget_ratio=args.budget_ratio, timeout_s=args.candidate_timeout_s)
            if search_eval.get("ok"):
                avg = search_eval["avg"]
                try:
                    proxy = evaluate_proxy_fac(code, proxy_datasets, hda_base)
                    valid = True
                except Exception as exc:  # noqa: BLE001
                    error = f"proxy: {type(exc).__name__}: {exc}"
            else:
                error = str(search_eval.get("error", "unknown search evaluation error"))

        record = {
            "idx": idx,
            "stage": "search",
            "node_id": stable_hash(f"{idx}:{code}:{time.time()}", 12),
            "parent_id": parent.get("node_id", "hda_root"),
            "target_family": family,
            "valid": valid,
            "error": error,
            "R": avg.get("R"),
            "cNBI": avg.get("cNBI"),
            "Time": avg.get("Time"),
            "rank_R": -1.0,
            "rank_cNBI": -1.0,
            "rank_Time": -1.0,
            "rank_score": -1.0,
            "code_hash": code_hash,
            "candidate_file": str(candidate_file),
            "prompt_elapsed_s": prompt_elapsed,
            "code": code,
        }
        record.update({k: v for k, v in proxy.items() if k != "proxy_detail"})
        records.append(record)
        search_rank_fields(records)
        record["fac_score"] = fac_total_score(record) if valid else -1e9
        new_credit = update_credit(credits, family, record)
        credit_trace.append(
            {
                "idx": idx,
                "target_family": family,
                "valid": valid,
                "fac_score": record.get("fac_score"),
                "fac_auc_adv": record.get("fac_auc_adv"),
                "early_fac": record.get("early_fac"),
                "proxy_auc_cNBI": record.get("proxy_auc_cNBI"),
                "credit_after": new_credit,
                **{f"credit_{f}": credits[f] for f in FAC_FAMILIES},
            }
        )

        with llm_log.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "idx": idx,
                        "parent_id": parent.get("node_id"),
                        "target_family": family,
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
        save_state(records, credit_trace)
        if idx % 5 == 0 or idx == args.nodes:
            top = summarize_top(records, k=1)
            print(
                f"[HAST-FAC] {idx}/{args.nodes} valid={sum(boolish(r.get('valid')) for r in records)-1} "
                f"family={family} best={top[0] if top else None}",
                flush=True,
            )

    valid_records = [r for r in records if boolish(r.get("valid")) and int(r.get("idx") or 0) > 0]
    if valid_records:
        best = max(valid_records, key=fac_total_score)
        if best.get("candidate_file"):
            shutil.copyfile(str(best["candidate_file"]), RUN_DIR / "best_candidate.py")
    save_state(records, credit_trace)
    plot_progress(records, credit_trace)


def evaluate_full12_top(args: argparse.Namespace) -> pd.DataFrame:
    records = load_existing_records()
    valid = [r for r in records if boolish(r.get("valid")) and int(r.get("idx") or 0) > 0]
    top = sorted(valid, key=fac_total_score, reverse=True)[: args.eval_top_k]
    rows: List[Dict[str, Any]] = []
    for rec in top:
        code = Path(str(rec["candidate_file"])).read_text(encoding="utf-8")
        fn = SEARCH.compile_degree_order(code)
        for dataset in E12.EVAL.DATASETS:
            graph = E12.EVAL.read_graph(dataset)
            rate = E12.EVAL.DATASET_RATES[dataset]
            t0 = time.perf_counter()
            order = list(fn(graph.copy()))
            elapsed = time.perf_counter() - t0
            metrics = E12.EVAL.compute_metrics(graph, order, rate=rate, method_time=elapsed)
            x = metrics["removal_ratio"].to_numpy(dtype=float)
            rows.append(
                {
                    "candidate_idx": int(rec["idx"]),
                    "candidate_file": rec["candidate_file"],
                    "dataset": dataset,
                    "R": float(metrics["GCC"].mean()),
                    "auc_cNBI": auc_mean(x, metrics["cNBI"].to_numpy(dtype=float)),
                    "time_s": elapsed,
                }
            )
        print(f"[full12] candidate {rec['idx']}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / f"{RUN_DIR.name}_full12_detail.csv", index=False, encoding="utf-8-sig")
    mean = out.groupby(["candidate_idx", "candidate_file"])[["R", "auc_cNBI", "time_s"]].mean().reset_index().sort_values("auc_cNBI", ascending=False)
    mean.to_csv(TABLE_DIR / f"{RUN_DIR.name}_full12_mean.csv", index=False, encoding="utf-8-sig")
    return mean


def write_report() -> None:
    records = pd.read_csv(RUN_DIR / "search_records.csv")
    valid = records[(records["idx"] > 0) & records["valid"].astype(str).str.lower().isin(["true", "1"])]
    top = valid.sort_values("fac_score", ascending=False).head(10)
    full_path = TABLE_DIR / f"{RUN_DIR.name}_full12_mean.csv"
    full = pd.read_csv(full_path) if full_path.exists() else pd.DataFrame()
    base12 = pd.read_csv(ROOT / "final_12graph_eval" / "hast_12graph_summary.csv")
    base_mean = base12.groupby("method")[["R", "auc_cNBI", "time_s"]].mean().reset_index().sort_values("auc_cNBI", ascending=False)
    lines = [
        f"# {RUN_DIR.name} 在线搜索实验结果",
        "",
        "## 设置",
        "",
        "- root: 原始 HDA。",
        "- 在线 LLM 生成候选。",
        "- 选择与 credit 不使用普通总分，而使用 Fracture Advantage Credit: candidate 相对 HDA 的 cNBI/NCC/GCC 曲线优势。",
        "- 每个候选仍在 50 个 search graphs 上跑 R/cNBI/time，同时在 proxy graphs 上计算 FAC。",
        "",
        "## Top HAST-FAC candidates by FAC score",
        "",
        top[["idx", "target_family", "fac_score", "fac_auc_adv", "early_fac", "fac_worst_auc_adv", "proxy_auc_cNBI", "proxy_time_s", "R", "cNBI", "Time", "candidate_file"]].to_markdown(index=False),
    ]
    if not full.empty:
        lines += [
            "",
            "## Full 12-graph evaluation of top candidates",
            "",
            full.to_markdown(index=False),
            "",
            "## Existing method 12-graph means",
            "",
            base_mean.to_markdown(index=False),
        ]
    (REPORT_DIR / f"{RUN_DIR.name}_search_summary_cn.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="HAST-FAC")
    p.add_argument("--nodes", type=int, default=80)
    p.add_argument("--model", default="gpt-5.5")
    p.add_argument("--reasoning-effort", default="none")
    p.add_argument("--max-completion-tokens", type=int, default=1800)
    p.add_argument("--llm-timeout-s", type=float, default=180.0)
    p.add_argument("--llm-retries", type=int, default=2)
    p.add_argument("--candidate-timeout-s", type=float, default=180.0)
    p.add_argument("--budget-ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--proxy-datasets", default=",".join(DEFAULT_PROXY_DATASETS))
    p.add_argument("--max-parent-code-chars", type=int, default=9000)
    p.add_argument("--eval-top-k", type=int, default=3)
    p.add_argument("--skip-search", action="store_true")
    p.add_argument("--eval-full12", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_run(args.run_name)
    ensure_dirs()
    if not args.skip_search:
        run_search(args)
    if args.eval_full12:
        evaluate_full12_top(args)
    write_report()
    print(REPORT_DIR / f"{RUN_DIR.name}_search_summary_cn.md")


if __name__ == "__main__":
    main()
