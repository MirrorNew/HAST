# -*- coding: utf-8 -*-
"""Run HDA-root ablations for generic LLM search frameworks.

This script creates a common harness for generic LLM program-search policies:
PUCT, MCTS-AHD-like, Clade-AHD-like, FunSearch-like, and
AlphaEvolve/OpenEvolve-like.  DACTS is imported from the canonical completed
run and is not regenerated here.

All generated candidates must expose:

    def degree_order(G):
        return [...]

The prompt intentionally starts from plain HDA and does not inject e26f, DACTS
typed clades, or the bridge/open-wedge recipe.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
import textwrap
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import heapq
import itertools

import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = EXPERIMENT_ROOT.parents[1]
CANONICAL_DACTS = WORKSPACE / "research" / "dacts_e26f_reverse_search_20260519"
DACTS_SRC = CANONICAL_DACTS / "src" / "dacts_search.py"
DACTS_OUTPUTS = CANONICAL_DACTS / "outputs"

RUNS_DIR = EXPERIMENT_ROOT / "runs"
TABLE_DIR = EXPERIMENT_ROOT / "tables"
REPORT_DIR = EXPERIMENT_ROOT / "reports"
GLOBAL_LLM_LOG = EXPERIMENT_ROOT / "llm_calls.jsonl"

GENERIC_METHODS = [
    "PUCT",
    "MCTS-AHD-like",
    "Clade-AHD-like",
    "FunSearch-like",
    "AlphaEvolve-like",
]
ALL_METHODS = ["DACTS"] + GENERIC_METHODS

ROOT_ID = "hda_root"
DEFAULT_MODEL = "gpt-5.5"

HDA_CODE = r'''
def degree_order(G):
    """Plain unoptimized HDA root: repeatedly remove the current highest-degree node."""
    H = G.copy()
    order = []
    while H.number_of_nodes() > 0:
        node = max(H.nodes(), key=lambda u: (H.degree[u], str(u)))
        order.append(node)
        H.remove_node(node)
    return order
'''.strip()

ALLOWED_IMPORT_ROOTS = {
    "math",
    "heapq",
    "random",
    "itertools",
    "collections",
    "networkx",
    "numpy",
}
FORBIDDEN_TOKENS = [
    "__import__",
    "open(",
    "exec(",
    "eval(",
    "compile(",
    "input(",
    "globals(",
    "locals(",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "shutil",
    "pathlib",
    "pickle",
    "marshal",
    "ctypes",
    "multiprocessing",
    "threading",
    "os.",
    "sys.",
    "write(",
    "rmdir(",
    "unlink(",
]


def stable_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def ensure_dirs() -> None:
    for path in [RUNS_DIR, TABLE_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_dacts_module() -> Any:
    return load_module(DACTS_SRC, "dacts_search_for_ablation")


def load_search_graph_payloads() -> List[Dict[str, Any]]:
    path = DACTS_OUTPUTS / "graphs_50_powerlaw500.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical graph suite: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def rank_records(records: List[Dict[str, Any]]) -> None:
    valid = [
        r
        for r in records
        if bool(r.get("valid"))
        and isinstance(r.get("R"), (int, float))
        and isinstance(r.get("cNBI"), (int, float))
        and isinstance(r.get("Time"), (int, float))
        and math.isfinite(float(r["R"]))
        and math.isfinite(float(r["cNBI"]))
        and math.isfinite(float(r["Time"]))
    ]
    for r in records:
        r["rank_R"] = -1.0
        r["rank_cNBI"] = -1.0
        r["rank_Time"] = -1.0
        r["rank_score"] = -1.0
    if not valid:
        return

    def assign(metric: str, higher: bool, out_key: str) -> None:
        ordered = sorted(valid, key=lambda row: float(row[metric]), reverse=higher)
        denom = max(1, len(ordered) - 1)
        for pos, row in enumerate(ordered):
            row[out_key] = (denom - pos) / denom

    assign("R", False, "rank_R")
    assign("cNBI", True, "rank_cNBI")
    assign("Time", False, "rank_Time")
    for r in valid:
        r["rank_score"] = 0.4 * r["rank_R"] + 0.3 * r["rank_cNBI"] + 0.3 * r["rank_Time"]


def extract_code(response_text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", response_text, flags=re.DOTALL | re.IGNORECASE)
    for block in blocks:
        if "def degree_order" in block:
            return block.strip()
    idx = response_text.find("def degree_order")
    if idx >= 0:
        return response_text[idx:].strip()
    return response_text.strip()


def validate_code(code: str) -> str:
    code = textwrap.dedent(code).strip()
    if "def degree_order" not in code:
        raise ValueError("missing degree_order(G)")
    lowered = code.lower()
    for token in FORBIDDEN_TOKENS:
        if token.lower() in lowered:
            raise ValueError(f"forbidden token: {token}")
    tree = ast.parse(code)
    has_degree_order = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots: List[str] = []
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif node.module:
                roots = [node.module.split(".")[0]]
            if any(root not in ALLOWED_IMPORT_ROOTS for root in roots):
                raise ValueError(f"forbidden import: {roots}")
        elif isinstance(node, ast.FunctionDef):
            if node.name == "degree_order":
                has_degree_order = True
        elif isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            continue
        elif isinstance(node, ast.Assign):
            continue
        else:
            raise ValueError(f"forbidden top-level statement: {type(node).__name__}")
    if not has_degree_order:
        raise ValueError("missing degree_order function")
    return code


def guarded_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"blocked import: {name}")
    return __import__(name, globals_, locals_, fromlist, level)


def compile_degree_order(code: str) -> Any:
    code = validate_code(code)
    safe_builtins = {
        "__import__": guarded_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "Exception": Exception,
        "float": float,
        "getattr": getattr,
        "hasattr": hasattr,
        "int": int,
        "iter": iter,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "next": next,
        "pow": pow,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "ValueError": ValueError,
        "zip": zip,
    }
    env: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "nx": nx,
        "np": np,
        "math": math,
        "heapq": heapq,
        "random": random,
        "itertools": itertools,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "deque": deque,
    }
    exec(compile(code, "<candidate>", "exec"), env, env)
    fn = env.get("degree_order")
    if not callable(fn):
        raise ValueError("degree_order is not callable")
    return fn


def _candidate_worker(
    code: str,
    graph_payloads: List[Dict[str, Any]],
    budget_ratio: float,
    queue: mp.Queue,
) -> None:
    try:
        dacts = load_dacts_module()
        fn = compile_degree_order(code)
        rows = []
        for payload in graph_payloads:
            edge_key = "edges" if "edges" in payload else "links"
            graph = json_graph.node_link_graph(payload, edges=edge_key)
            t0 = time.perf_counter()
            order = fn(graph.copy())
            elapsed = time.perf_counter() - t0
            if not isinstance(order, (list, tuple)):
                raise ValueError("degree_order must return a list/tuple")
            metrics = dacts.evaluate_order(graph, list(order), budget_ratio=budget_ratio)
            metrics["Time"] = elapsed
            rows.append(metrics)
        avg = {
            "R": float(np.mean([r["R"] for r in rows])),
            "cNBI": float(np.mean([r["cNBI"] for r in rows])),
            "Time": float(np.mean([r["Time"] for r in rows])),
            "avg_top5_mass": float(np.mean([r["avg_top5_mass"] for r in rows])),
            "avg_hhi": float(np.mean([r["avg_hhi"] for r in rows])),
            "avg_pairdisc": float(np.mean([r["avg_pairdisc"] for r in rows])),
        }
        queue.put({"ok": True, "avg": avg})
    except Exception as exc:  # noqa: BLE001 - recorded as candidate failure
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def evaluate_candidate_code(
    code: str,
    graph_payloads: List[Dict[str, Any]],
    budget_ratio: float,
    timeout_s: float,
) -> Dict[str, Any]:
    try:
        code = validate_code(code)
    except Exception as exc:
        return {"ok": False, "error": f"validation: {type(exc).__name__}: {exc}"}
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_candidate_worker, args=(code, graph_payloads, budget_ratio, queue))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(3)
        return {"ok": False, "error": f"timeout>{timeout_s:.1f}s"}
    if queue.empty():
        return {"ok": False, "error": "worker produced no result"}
    return queue.get()


def child_count(records: List[Dict[str, Any]]) -> Counter:
    return Counter(str(r.get("parent_id") or "") for r in records)


def record_by_id(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r["node_id"]): r for r in records}


def depth_of(record: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> int:
    depth = 0
    cur = str(record.get("parent_id") or "")
    seen = set()
    while cur and cur in by_id and cur not in seen:
        seen.add(cur)
        depth += 1
        cur = str(by_id[cur].get("parent_id") or "")
    return depth


def valid_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in records if bool(r.get("valid"))]


class SearchPolicy:
    method = "base"

    def __init__(self, rng: random.Random):
        self.rng = rng

    def select_parent(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        pool = valid_records(records)
        return max(pool, key=lambda r: float(r.get("rank_score", 0.0)))

    def examples(self, records: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
        return sorted(valid_records(records), key=lambda r: float(r.get("rank_score", 0.0)), reverse=True)[:k]

    def hint(self, records: List[Dict[str, Any]], parent: Dict[str, Any]) -> str:
        return "Generate one concise mutation of the parent program."

    def update(self, record: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
        return None


class PUCTPolicy(SearchPolicy):
    method = "PUCT"

    def select_parent(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        counts = child_count(records)
        total = max(2, len(records))
        pool = valid_records(records)
        c = 1.35

        def value(r: Dict[str, Any]) -> float:
            q = max(0.0, float(r.get("rank_score", 0.0)))
            n = counts[str(r["node_id"])]
            return q + c * math.sqrt(math.log(total) / (1.0 + n))

        return max(pool, key=value)

    def hint(self, records: List[Dict[str, Any]], parent: Dict[str, Any]) -> str:
        return (
            "PUCT policy: exploit the parent if it is strong, but make one clear "
            "structural mutation so sibling branches remain diverse."
        )


class MCTSAHDPolicy(SearchPolicy):
    method = "MCTS-AHD-like"

    def select_parent(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_id = record_by_id(records)
        counts = child_count(records)
        pool = valid_records(records)
        total = max(2, len(records))

        def value(r: Dict[str, Any]) -> float:
            q = max(0.0, float(r.get("rank_score", 0.0)))
            n = counts[str(r["node_id"])]
            depth_bonus = 1.0 / (1.0 + 0.12 * depth_of(r, by_id))
            return q * depth_bonus + 1.1 * math.sqrt(math.log(total) / (1.0 + n))

        return max(pool, key=value)

    def hint(self, records: List[Dict[str, Any]], parent: Dict[str, Any]) -> str:
        return (
            "MCTS-AHD-like policy: treat the parent as a heuristic-design state. "
            "Expand it with a coherent operator-level change, not a parameter tweak only."
        )


def code_feature_signature(code: str) -> str:
    probes = {
        "heap": "heap" in code,
        "core": "core" in code.lower(),
        "tri": "tri" in code.lower() or "cluster" in code.lower(),
        "component": "component" in code.lower() or "connected" in code.lower(),
        "neighbor": "neighbor" in code.lower() or "nbr" in code.lower(),
        "twohop": "2" in code and ("hop" in code.lower() or "two" in code.lower()),
    }
    active = [k for k, v in probes.items() if v]
    return "+".join(active[:3]) if active else "simple"


class CladeAHDPolicy(SearchPolicy):
    method = "Clade-AHD-like"

    def select_parent(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        pool = valid_records(records)
        if len(pool) <= 2:
            return pool[-1]
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in pool:
            groups[str(r.get("generic_clade") or "simple")].append(r)
        global_median = float(np.median([float(r.get("rank_score", 0.0)) for r in pool]))
        samples = []
        for name, group in groups.items():
            wins = sum(float(r.get("rank_score", 0.0)) >= global_median for r in group)
            losses = max(0, len(group) - wins)
            theta = self.rng.betavariate(1 + wins, 1 + losses)
            theta += 0.08 / math.sqrt(len(group))
            samples.append((theta, name))
        _, selected = max(samples)
        group = sorted(groups[selected], key=lambda r: float(r.get("rank_score", 0.0)), reverse=True)
        return self.rng.choice(group[: min(5, len(group))])

    def hint(self, records: List[Dict[str, Any]], parent: Dict[str, Any]) -> str:
        return (
            "Clade-AHD-like policy: preserve the parent lineage's useful idea, "
            "but mutate it enough to form a distinguishable generic program family."
        )

    def update(self, record: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
        record["generic_clade"] = code_feature_signature(str(record.get("code", "")))


class FunSearchPolicy(SearchPolicy):
    method = "FunSearch-like"

    def __init__(self, rng: random.Random, islands: int = 6):
        super().__init__(rng)
        self.islands = islands
        self.next_island = 0

    def select_parent(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        pool = valid_records(records)
        if len(pool) <= 1:
            return pool[0]
        island = self.next_island % self.islands
        self.next_island += 1
        island_pool = [r for r in pool if int(r.get("island", 0)) == island]
        if not island_pool:
            island_pool = pool
        ranked = sorted(island_pool, key=lambda r: float(r.get("rank_score", 0.0)), reverse=True)
        return self.rng.choice(ranked[: min(4, len(ranked))])

    def hint(self, records: List[Dict[str, Any]], parent: Dict[str, Any]) -> str:
        island = int(parent.get("island", 0))
        return (
            f"FunSearch-like policy: produce a short standalone program for island {island}. "
            "Prefer a compact recombination of elite ideas over a long rewrite."
        )

    def update(self, record: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
        if record["node_id"] == ROOT_ID:
            record["island"] = 0
        else:
            record["island"] = int(stable_hash(record["node_id"], 4), 16) % self.islands


class AlphaEvolvePolicy(SearchPolicy):
    method = "AlphaEvolve-like"

    def select_parent(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        pool = valid_records(records)
        if len(pool) <= 1:
            return pool[0]
        bins: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in pool:
            bins[str(r.get("map_bin") or "root")].append(r)
        # MAP-Elites flavor: sample a niche inversely to occupancy, then use its elite.
        names = list(bins)
        weights = np.array([1.0 / math.sqrt(len(bins[n])) for n in names], dtype=float)
        weights = weights / weights.sum()
        selected = self.rng.choices(names, weights=weights.tolist(), k=1)[0]
        ranked = sorted(bins[selected], key=lambda r: float(r.get("rank_score", 0.0)), reverse=True)
        if self.rng.random() < 0.2:
            return self.rng.choice(pool)
        return ranked[0]

    def hint(self, records: List[Dict[str, Any]], parent: Dict[str, Any]) -> str:
        return (
            "AlphaEvolve/OpenEvolve-like policy: search for an archive niche. "
            "Keep the program executable, expose a genuinely different algorithmic structure, "
            "and avoid bloated code."
        )

    def update(self, record: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
        code = str(record.get("code", ""))
        length_bin = "short" if len(code) < 1800 else "long"
        loop_bin = "fewloops" if code.count("for ") + code.count("while ") < 7 else "manyloops"
        feature = code_feature_signature(code).split("+")[0]
        record["map_bin"] = f"{length_bin}:{loop_bin}:{feature}"


POLICIES = {
    "PUCT": PUCTPolicy,
    "MCTS-AHD-like": MCTSAHDPolicy,
    "Clade-AHD-like": CladeAHDPolicy,
    "FunSearch-like": FunSearchPolicy,
    "AlphaEvolve-like": AlphaEvolvePolicy,
}


def build_prompt(
    method: str,
    policy: SearchPolicy,
    parent: Dict[str, Any],
    records: List[Dict[str, Any]],
    max_parent_code_chars: int = 12000,
) -> List[Dict[str, str]]:
    top = policy.examples(records, k=4)
    top_summary = [
        {
            "node_id": r["node_id"],
            "score": round(float(r.get("rank_score", 0.0)), 4),
            "R": round(float(r.get("R", 0.0)), 6),
            "cNBI": round(float(r.get("cNBI", 0.0)), 4),
            "Time": round(float(r.get("Time", 0.0)), 5),
        }
        for r in top
    ]
    parent_code = str(parent.get("code") or HDA_CODE)
    if max_parent_code_chars and len(parent_code) > max_parent_code_chars:
        keep_head = max(2000, int(max_parent_code_chars * 0.62))
        keep_tail = max(1000, max_parent_code_chars - keep_head)
        omitted = len(parent_code) - keep_head - keep_tail
        parent_code = (
            parent_code[:keep_head].rstrip()
            + f"\n\n# ... [parent code truncated: {omitted} chars omitted; keep the same degree_order(G) interface] ...\n\n"
            + parent_code[-keep_tail:].lstrip()
        )
    user = f"""
We are running a generic LLM program-search ablation for iterative network dismantling.

Search framework: {method}
Framework instruction: {policy.hint(records, parent)}

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
- Do not mention or copy any named previous discovered algorithm; return code only.

Recent top candidates:
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


def call_llm(
    messages: List[Dict[str, str]],
    model: str,
    max_retries: int = 3,
    timeout_s: float = 90.0,
    max_completion_tokens: int = 2200,
    reasoning_effort: str = "",
) -> str:
    api_key = os.environ.get("SMAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("SMAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("SMAI_API_KEY/OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is required") from exc
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client_kwargs["timeout"] = timeout_s
    client = OpenAI(**client_kwargs)
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            request_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": False,
                "max_completion_tokens": max_completion_tokens,
                "timeout": timeout_s,
            }
            effort = (reasoning_effort or "").strip().lower()
            if effort and effort not in {"default", "medium"}:
                request_kwargs["reasoning_effort"] = reasoning_effort
            if effort and effort not in {"none", "off", "default"}:
                request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            response = client.chat.completions.create(**request_kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(10, 1.5 * attempt))
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")


def _llm_worker(
    messages: List[Dict[str, str]],
    model: str,
    max_retries: int,
    timeout_s: float,
    max_completion_tokens: int,
    reasoning_effort: str,
    queue: mp.Queue,
) -> None:
    try:
        content = call_llm(
            messages,
            model,
            max_retries=max_retries,
            timeout_s=timeout_s,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )
        queue.put({"ok": True, "content": content})
    except Exception as exc:  # noqa: BLE001
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def call_llm_hard_timeout(
    messages: List[Dict[str, str]],
    model: str,
    max_retries: int,
    timeout_s: float,
    max_completion_tokens: int,
    reasoning_effort: str,
) -> str:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_llm_worker,
        args=(messages, model, max_retries, timeout_s, max_completion_tokens, reasoning_effort, queue),
    )
    proc.start()
    proc.join(timeout_s + 5.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(3.0)
        raise RuntimeError(f"hard LLM timeout>{timeout_s:.1f}s")
    if queue.empty():
        raise RuntimeError("LLM worker produced no result")
    result = queue.get()
    if not result.get("ok"):
        raise RuntimeError(str(result.get("error", "unknown LLM error")))
    return str(result.get("content", ""))


def method_dir(method: str) -> Path:
    return RUNS_DIR / method.replace("/", "_")


def save_method_records(method: str, records: List[Dict[str, Any]]) -> None:
    out = method_dir(method)
    out.mkdir(parents=True, exist_ok=True)
    scrubbed = []
    for r in records:
        row = {k: v for k, v in r.items() if k != "code"}
        scrubbed.append(row)
    fieldnames = sorted({key for row in scrubbed for key in row.keys()})
    preferred = [
        "idx",
        "method",
        "stage",
        "node_id",
        "parent_id",
        "valid",
        "error",
        "R",
        "cNBI",
        "Time",
        "rank_R",
        "rank_cNBI",
        "rank_Time",
        "rank_score",
        "prompt_id",
        "code_hash",
        "generic_clade",
        "island",
        "map_bin",
    ]
    ordered = [c for c in preferred if c in fieldnames] + [c for c in fieldnames if c not in preferred]
    write_csv(out / "search_records.csv", scrubbed, ordered)
    edges = [
        {"source": r.get("parent_id"), "target": r.get("node_id"), "idx": r.get("idx"), "valid": r.get("valid")}
        for r in records
        if r.get("parent_id")
    ]
    write_csv(out / "tree_edges.csv", edges)
    best = max(valid_records(records), key=lambda r: float(r.get("rank_score", 0.0)), default=None)
    if best and best.get("code"):
        (out / "best_candidate.py").write_text(str(best["code"]), encoding="utf-8")
    (out / "records_with_code.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )


def load_method_records(method: str) -> List[Dict[str, Any]]:
    path = method_dir(method) / "records_with_code.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rank_records(rows)
    return rows


def root_record(method: str, graph_payloads: List[Dict[str, Any]], budget_ratio: float, timeout_s: float) -> Dict[str, Any]:
    result = evaluate_candidate_code(HDA_CODE, graph_payloads, budget_ratio, timeout_s)
    if not result.get("ok"):
        raise RuntimeError(f"HDA root failed: {result.get('error')}")
    avg = result["avg"]
    return {
        "idx": 0,
        "method": method,
        "stage": "root",
        "node_id": ROOT_ID,
        "parent_id": "",
        "valid": True,
        "error": "",
        "R": avg["R"],
        "cNBI": avg["cNBI"],
        "Time": avg["Time"],
        "avg_top5_mass": avg["avg_top5_mass"],
        "avg_hhi": avg["avg_hhi"],
        "avg_pairdisc": avg["avg_pairdisc"],
        "prompt_id": "",
        "code_hash": stable_hash(HDA_CODE, 16),
        "code": HDA_CODE,
    }


def run_method(args: argparse.Namespace, method: str, graph_payloads: List[Dict[str, Any]]) -> None:
    out = method_dir(method)
    out.mkdir(parents=True, exist_ok=True)
    candidate_dir = out / "candidates"
    if args.write_candidates:
        candidate_dir.mkdir(exist_ok=True)

    rng = random.Random(args.seed + int(stable_hash(method, 6), 16) % 10_000)
    policy = POLICIES[method](rng)

    records = load_method_records(method) if args.resume else []
    if not records:
        print(f"[{method}] evaluating HDA root", flush=True)
        root = root_record(method, graph_payloads, args.budget_ratio, args.timeout_s)
        policy.update(root, records)
        records = [root]
        rank_records(records)
        save_method_records(method, records)

    seen_hashes = {str(r.get("code_hash")) for r in records if r.get("code_hash")}
    while len(records) < args.nodes:
        rank_records(records)
        parent = policy.select_parent(records)
        idx = len(records)
        prompt_id = f"{method}_{idx:04d}_{stable_hash(str(time.time()), 6)}"
        messages = build_prompt(method, policy, parent, records, max_parent_code_chars=args.max_parent_code_chars)
        raw_response = ""
        code = ""
        llm_error = ""
        t_prompt = time.perf_counter()
        try:
            raw_response = call_llm_hard_timeout(
                messages,
                args.model,
                max_retries=args.llm_retries,
                timeout_s=args.llm_timeout_s,
                max_completion_tokens=args.max_completion_tokens,
                reasoning_effort=args.reasoning_effort,
            )
            code = extract_code(raw_response)
        except Exception as exc:  # noqa: BLE001
            llm_error = f"llm: {type(exc).__name__}: {exc}"
        prompt_elapsed = time.perf_counter() - t_prompt
        with GLOBAL_LLM_LOG.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "prompt_id": prompt_id,
                        "idx": idx,
                        "method": method,
                        "parent_id": parent["node_id"],
                        "messages": messages,
                        "response": raw_response,
                        "llm_error": llm_error,
                        "elapsed_s": prompt_elapsed,
                        "reasoning_effort": args.reasoning_effort,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if llm_error and args.skip_llm_failures:
            print(
                f"[{method} {idx:04d}/{args.nodes}] skip LLM failure after "
                f"{prompt_elapsed:.1f}s: {llm_error[:120]}",
                flush=True,
            )
            continue

        code_hash = stable_hash(code, 16) if code else stable_hash(llm_error + str(idx), 16)
        candidate_path = candidate_dir / f"candidate_{idx:04d}_{code_hash}.py"
        if code and args.write_candidates:
            candidate_path.write_text(code, encoding="utf-8")

        rec: Dict[str, Any] = {
            "idx": idx,
            "method": method,
            "stage": "search",
            "node_id": stable_hash(f"{method}:{idx}:{parent['node_id']}:{code_hash}", 12),
            "parent_id": parent["node_id"],
            "valid": False,
            "error": "",
            "R": math.nan,
            "cNBI": math.nan,
            "Time": math.nan,
            "avg_top5_mass": math.nan,
            "avg_hhi": math.nan,
            "avg_pairdisc": math.nan,
            "prompt_id": prompt_id,
            "code_hash": code_hash,
            "candidate_file": str(candidate_path) if args.write_candidates else "",
            "prompt_elapsed_s": prompt_elapsed,
            "code": code,
        }
        if llm_error:
            rec["error"] = llm_error
        elif code_hash in seen_hashes:
            rec["error"] = "duplicate_code"
        else:
            seen_hashes.add(code_hash)
            result = evaluate_candidate_code(code, graph_payloads, args.budget_ratio, args.timeout_s)
            if result.get("ok"):
                avg = result["avg"]
                rec.update(
                    {
                        "valid": True,
                        "error": "",
                        "R": avg["R"],
                        "cNBI": avg["cNBI"],
                        "Time": avg["Time"],
                        "avg_top5_mass": avg["avg_top5_mass"],
                        "avg_hhi": avg["avg_hhi"],
                        "avg_pairdisc": avg["avg_pairdisc"],
                    }
                )
            else:
                rec["error"] = str(result.get("error", "unknown_error"))
        policy.update(rec, records)
        records.append(rec)
        rank_records(records)
        if len(records) % args.checkpoint_every == 0 or len(records) == args.nodes:
            save_method_records(method, records)
        if len(records) % 10 == 0 or len(records) == args.nodes:
            best = max(valid_records(records), key=lambda r: float(r.get("rank_score", 0.0)))
            print(
                f"[{method} {len(records):04d}/{args.nodes}] "
                f"valid={sum(bool(r.get('valid')) for r in records)} "
                f"best={best['node_id']} score={float(best.get('rank_score', 0.0)):.4f} "
                f"R={float(best.get('R')):.6f} cNBI={float(best.get('cNBI')):.3f} "
                f"T={float(best.get('Time')):.4f}",
                flush=True,
            )

    save_method_records(method, records)


def import_dacts_run() -> None:
    out = method_dir("DACTS")
    out.mkdir(parents=True, exist_ok=True)
    src_csv = DACTS_OUTPUTS / "search_records.csv"
    if not src_csv.exists():
        raise FileNotFoundError(src_csv)
    df = pd.read_csv(src_csv)
    df.insert(1, "method", "DACTS")
    df["prompt_id"] = "canonical_dacts"
    df["code_hash"] = ""
    df["error"] = ""
    df["valid"] = df["valid"].astype(bool)
    df.to_csv(out / "search_records.csv", index=False, encoding="utf-8-sig")
    edges = df[df["parent_id"].notna()][["parent_id", "node_id", "idx", "valid"]].rename(
        columns={"parent_id": "source", "node_id": "target"}
    )
    edges.to_csv(out / "tree_edges.csv", index=False, encoding="utf-8-sig")
    candidate = CANONICAL_DACTS / "candidates" / "candidate_01_d116927734d7.py"
    if candidate.exists():
        shutil.copy2(candidate, out / "best_candidate.py")
    # Do not copy LLM logs here: the canonical run already owns them.


def write_manifest(args: argparse.Namespace) -> None:
    manifest = {
        "experiment": "tree_search_ablation_20260520",
        "created_from": str(Path(__file__).resolve()),
        "canonical_dacts": str(CANONICAL_DACTS),
        "graph_suite": str(DACTS_OUTPUTS / "graphs_50_powerlaw500.json"),
        "methods": ALL_METHODS,
        "model": args.model,
        "nodes_per_method": args.nodes,
        "max_completion_tokens": args.max_completion_tokens,
        "budget_ratio": args.budget_ratio,
        "seed": args.seed,
        "reasoning_effort": args.reasoning_effort,
        "api_key_written_to_disk": False,
    }
    (EXPERIMENT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_methods(value: str) -> List[str]:
    if value.lower() == "all":
        return GENERIC_METHODS
    methods = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [m for m in methods if m not in GENERIC_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    return methods


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--budget-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--llm-timeout-s", type=float, default=90.0)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--max-parent-code-chars", type=int, default=12000)
    parser.add_argument(
        "--skip-llm-failures",
        action="store_true",
        help="Record LLM transport failures in llm_calls.jsonl but do not consume candidate-node budget.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="",
        help="Use none to disable reasoning; omit for provider default, usually medium.",
    )
    parser.add_argument("--no-write-candidates", dest="write_candidates", action="store_false")
    parser.set_defaults(write_candidates=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--import-dacts", action="store_true", default=True)
    parser.add_argument("--skip-import-dacts", dest="import_dacts", action="store_false")
    args = parser.parse_args()

    ensure_dirs()
    write_manifest(args)
    if args.import_dacts:
        import_dacts_run()
    graph_payloads = load_search_graph_payloads()
    for method in parse_methods(args.methods):
        run_method(args, method, graph_payloads)


if __name__ == "__main__":
    main()
