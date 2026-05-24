# -*- coding: utf-8 -*-
"""
DACTS-style HDA-root search for network dismantling heuristics.

All artifacts produced by this script are written under local output folders
inside ``HAST2026/main``.

The runner intentionally keeps the old ERA-like experiment spirit:
- 50 generated 500-node power-law graphs
- rank-composite objective over R, fragmentation, and runtime
- executable `degree_order(G)` candidate export

New search ingredients:
- typed dismantling program configurations
- diagnostic-guided mutation
- clade-level budget allocation

Protocol constraints:
- the only search root is the original unoptimized HDA algorithm
- e26f is evaluated only after search as a reference
- LLM prompts may contain general dismantling clues, but not the e26f recipe
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

try:
    import openai
except Exception:  # pragma: no cover - optional dependency
    openai = None


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
RERUN_ROOT = EXPERIMENT_ROOT / "runs" / "DACTS-rerun"
OUTPUT_DIR = RERUN_ROOT / "outputs"
CANDIDATE_DIR = RERUN_ROOT / "candidates"
REPORT_DIR = RERUN_ROOT / "reports"


def stable_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def generate_powerlaw_network(
    n: int,
    gamma: float,
    k_min: int = 2,
    k_max: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> nx.Graph:
    """Generate a power-law graph using the same configuration-model idea as old ERA-like."""
    rng = rng or np.random.default_rng()
    if k_max is None:
        k_max = int(n ** (1 / (gamma - 1)))
    degrees: List[int] = []
    while len(degrees) < n:
        k = int(rng.zipf(gamma))
        if k_min <= k <= k_max:
            degrees.append(k)
    if sum(degrees) % 2 != 0:
        degrees[int(rng.integers(0, n))] += 1
    graph = nx.configuration_model(degrees, seed=int(rng.integers(0, 2**31 - 1)))
    graph = nx.Graph(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    if graph.number_of_nodes() == 0:
        return graph
    if nx.number_connected_components(graph) > 1:
        largest_cc = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest_cc).copy()
    return nx.convert_node_labels_to_integers(graph, first_label=0)


def make_graph_suite(
    n_graphs: int = 50,
    n_nodes: int = 500,
    gamma: float = 2.5,
    seed: int = 20260519,
) -> List[nx.Graph]:
    rng = np.random.default_rng(seed)
    graphs = []
    attempts = 0
    while len(graphs) < n_graphs and attempts < n_graphs * 10:
        attempts += 1
        graph = generate_powerlaw_network(n_nodes, gamma, rng=rng)
        if graph.number_of_nodes() >= int(0.85 * n_nodes) and graph.number_of_edges() > 0:
            graphs.append(graph)
    if len(graphs) != n_graphs:
        raise RuntimeError(f"Only generated {len(graphs)} usable graphs out of {n_graphs}.")
    return graphs


class DSU:
    def __init__(self, nodes: Iterable[int]):
        self.parent = {node: node for node in nodes}
        self.size = {node: 1 for node in nodes}
        self.ncc = len(self.parent)
        self.sum_sq = len(self.parent)
        self.size_counts = Counter({1: len(self.parent)})
        self.max_size = 1 if self.parent else 0

    def find(self, node: int) -> int:
        parent = self.parent[node]
        if parent != node:
            parent = self.find(parent)
            self.parent[node] = parent
        return parent

    def _remove_size(self, size: int) -> None:
        self.size_counts[size] -= 1
        if self.size_counts[size] <= 0:
            del self.size_counts[size]

    def _add_size(self, size: int) -> None:
        self.size_counts[size] += 1
        if size > self.max_size:
            self.max_size = size

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        sa, sb = self.size[ra], self.size[rb]
        self._remove_size(sa)
        self._remove_size(sb)
        self.sum_sq -= sa * sa + sb * sb
        self.parent[rb] = ra
        self.size[ra] = sa + sb
        self.size[rb] = 0
        self.sum_sq += (sa + sb) * (sa + sb)
        self._add_size(sa + sb)
        self.ncc -= 1
        return True

    def topk_mass(self, remaining: int, k: int = 5) -> float:
        if remaining <= 0:
            return 0.0
        taken = 0
        mass = 0
        for size in sorted(self.size_counts.keys(), reverse=True):
            count = self.size_counts[size]
            take = min(count, k - taken)
            mass += take * size
            taken += take
            if taken >= k:
                break
        return mass / remaining


def evaluate_order(
    graph: nx.Graph,
    order: Sequence[int],
    budget_ratio: float = 0.30,
) -> Dict[str, float]:
    """Evaluate prefix R and concentration-aware NBI using reverse union-find."""
    n = graph.number_of_nodes()
    if n <= 1:
        return {
            "R": 1.0,
            "cNBI": 0.0,
            "avg_gcc": 1.0,
            "avg_top5_mass": 1.0,
            "avg_hhi": 1.0,
            "avg_pairdisc": 0.0,
        }
    budget = max(1, min(n, int(round(n * budget_ratio))))
    prefix = [node for node in order[:budget] if node in graph]
    if len(prefix) < budget:
        seen = set(prefix)
        prefix.extend([node for node in graph.nodes if node not in seen][: budget - len(prefix)])

    removed_set = set(prefix)
    alive = [node for node in graph.nodes if node not in removed_set]
    dsu = DSU(alive)
    alive_set = set(alive)
    for u, v in graph.edges:
        if u in alive_set and v in alive_set:
            dsu.union(u, v)

    reverse_states: List[Tuple[float, float, float, float, float]] = []

    def snapshot(remaining: int) -> Tuple[float, float, float, float, float]:
        if remaining <= 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        gcc = dsu.max_size / n
        hhi = dsu.sum_sq / (remaining * remaining)
        eff_components = (remaining * remaining / dsu.sum_sq) if dsu.sum_sq > 0 else 0.0
        if remaining > 1:
            connected_pairs = (dsu.sum_sq - remaining) / 2.0
            total_pairs = remaining * (remaining - 1) / 2.0
            pairdisc = max(0.0, 1.0 - connected_pairs / total_pairs)
        else:
            pairdisc = 1.0
        top5 = dsu.topk_mass(remaining)
        cnbi = pairdisc * eff_components / (1.0 + top5)
        return gcc, cnbi, top5, hhi, pairdisc

    remaining = n - budget
    reverse_states.append(snapshot(remaining))
    added = set(alive)
    for node in reversed(prefix):
        dsu.parent[node] = node
        dsu.size[node] = 1
        dsu.ncc += 1
        dsu.sum_sq += 1
        dsu._add_size(1)
        added.add(node)
        remaining += 1
        for nbr in graph.neighbors(node):
            if nbr in added:
                dsu.union(node, nbr)
        reverse_states.append(snapshot(remaining))

    states = list(reversed(reverse_states))
    # Exclude q=0 state for pressure toward early removal; include all prefix states otherwise.
    states = states[1:] if len(states) > 1 else states
    gcc_vals = [s[0] for s in states]
    cnbi_vals = [s[1] for s in states]
    top5_vals = [s[2] for s in states]
    hhi_vals = [s[3] for s in states]
    pairdisc_vals = [s[4] for s in states]
    return {
        "R": float(sum(gcc_vals) / len(gcc_vals)),
        "cNBI": float(sum(cnbi_vals) / len(cnbi_vals)),
        "avg_gcc": float(sum(gcc_vals) / len(gcc_vals)),
        "avg_top5_mass": float(sum(top5_vals) / len(top5_vals)),
        "avg_hhi": float(sum(hhi_vals) / len(hhi_vals)),
        "avg_pairdisc": float(sum(pairdisc_vals) / len(pairdisc_vals)),
    }


@dataclass
class AlgoConfig:
    clade: str
    w_split: float = 1.0
    w_bridge_mult: float = 1.0
    w_degree: float = 1.0
    w_nds: float = 0.0
    w_core: float = 0.0
    w_comp: float = 0.0
    w_bridge_edges: float = 0.0
    split_power: float = 1.0
    degree_power: float = 1.0
    nds_power: float = 1.0
    core_power: float = 0.5
    update_radius: int = 1
    use_component: bool = False
    comp_refresh: int = 25
    tie_degree: bool = True
    name: str = ""
    parent_id: str = ""
    mutation: str = ""
    node_id: str = field(default_factory=str)

    def canonical(self) -> str:
        payload = asdict(self).copy()
        payload.pop("node_id", None)
        payload.pop("name", None)
        payload.pop("parent_id", None)
        payload.pop("mutation", None)
        rounded = {
            key: (round(val, 6) if isinstance(val, float) else val)
            for key, val in payload.items()
        }
        return json.dumps(rounded, sort_keys=True)

    def assign_id(self) -> None:
        self.node_id = stable_hash(self.canonical())
        if not self.name:
            self.name = f"{self.clade}_{self.node_id}"


def transform(value: float, power: float) -> float:
    value = max(0.0, value)
    if power == 1.0:
        return math.log1p(value)
    if power == 0.5:
        return math.sqrt(value + 1.0)
    return math.log1p(value ** power)


def degree_order_by_config(graph: nx.Graph, cfg: AlgoConfig, budget_ratio: float = 0.30) -> List[int]:
    """Generic lazy-heap adaptive dismantling solver induced by a typed config."""
    if graph.number_of_nodes() == 0:
        return []
    h = nx.convert_node_labels_to_integers(graph, first_label=0, label_attribute="old_label")
    old_labels = nx.get_node_attributes(h, "old_label")
    nodes = list(h.nodes)
    alive = set(nodes)
    nbrs = {u: set(h.neighbors(u)) for u in nodes}
    deg = {u: len(nbrs[u]) for u in nodes}
    try:
        core = nx.core_number(h) if h.number_of_edges() > 0 else {u: 0 for u in nodes}
    except Exception:
        core = {u: 0 for u in nodes}

    triangles: Dict[int, int] = {}
    for u in nodes:
        nu = nbrs[u]
        if len(nu) < 2:
            triangles[u] = 0
        else:
            triangles[u] = sum(len(nu & nbrs[v]) for v in nu) // 2

    neigh_deg_sum = {u: sum(deg[v] for v in nbrs[u]) for u in nodes}
    bridge_edges = {u: 0 for u in nodes}
    if cfg.w_bridge_edges:
        for u in nodes:
            cnt = 0
            for v in nbrs[u]:
                if len((nbrs[u] - {v}) & (nbrs[v] - {u})) == 0:
                    cnt += 1
            bridge_edges[u] = cnt

    comp_size = {u: len(nodes) for u in nodes}

    def refresh_components() -> None:
        if not cfg.use_component:
            return
        sub = h.subgraph(alive)
        for comp in nx.connected_components(sub):
            size = len(comp)
            for x in comp:
                comp_size[x] = size

    refresh_components()

    def score(u: int) -> float:
        if u not in alive:
            return -1e30
        d = deg.get(u, 0)
        if d <= 0:
            return -1e30
        total_pairs = d * (d - 1) / 2.0
        tri = min(float(triangles.get(u, 0)), total_pairs)
        split_pairs = max(0.0, total_pairs - tri)
        bridge_factor = 1.0 - (tri / total_pairs if total_pairs > 0 else 1.0)
        split = cfg.w_split * math.log1p((split_pairs ** cfg.split_power) * (1.0 + cfg.w_bridge_mult * bridge_factor))
        degree_term = cfg.w_degree * transform(d, cfg.degree_power)
        nds_term = cfg.w_nds * transform(neigh_deg_sum.get(u, 0), cfg.nds_power)
        core_term = cfg.w_core * ((core.get(u, 0) + 1.0) ** cfg.core_power)
        bridge_edge_term = cfg.w_bridge_edges * math.log1p(bridge_edges.get(u, 0))
        comp_term = cfg.w_comp * math.log1p(comp_size.get(u, 1)) if cfg.use_component else 0.0
        return split + degree_term + nds_term + core_term + bridge_edge_term + comp_term

    import heapq

    version = {u: 0 for u in nodes}
    heap: List[Tuple[float, int, int, int]] = []
    for u in nodes:
        heapq.heappush(heap, (-score(u), -deg[u] if cfg.tie_degree else 0, u, version[u]))

    budget = max(1, min(len(nodes), int(round(len(nodes) * budget_ratio))))
    order: List[int] = []
    for step in range(budget):
        if cfg.use_component and step > 0 and step % max(1, cfg.comp_refresh) == 0:
            refresh_components()
            touched_for_comp = set(alive)
            for x in touched_for_comp:
                version[x] += 1
                heapq.heappush(heap, (-score(x), -deg.get(x, 0), x, version[x]))

        while heap:
            _, _, u, vu = heapq.heappop(heap)
            if u in alive and vu == version[u]:
                break
        else:
            u = max(alive, key=lambda x: (score(x), deg.get(x, 0)))

        if u not in alive:
            continue
        order.append(old_labels.get(u, u))
        nu = set(nbrs[u] & alive)

        for v in nu:
            common = len((nbrs[v] & alive) & nu)
            if common > 0:
                triangles[v] = max(0, triangles.get(v, 0) - common)

        alive.remove(u)
        for v in nu:
            nbrs[v].discard(u)
            deg[v] = len(nbrs[v] & alive)
        deg[u] = 0
        nbrs[u].clear()
        triangles[u] = 0
        neigh_deg_sum[u] = 0
        bridge_edges[u] = 0

        touched = set(nu)
        if cfg.update_radius >= 2:
            for v in list(nu):
                touched.update(nbrs[v] & alive)
        for x in touched:
            if x not in alive:
                continue
            neigh_deg_sum[x] = sum(deg[y] for y in nbrs[x] if y in alive)
            if cfg.w_bridge_edges:
                bcnt = 0
                nx_alive = nbrs[x] & alive
                for y in nx_alive:
                    if len((nx_alive - {y}) & (nbrs[y] & alive - {x})) == 0:
                        bcnt += 1
                bridge_edges[x] = bcnt
            version[x] += 1
            heapq.heappush(heap, (-score(x), -deg.get(x, 0) if cfg.tie_degree else 0, x, version[x]))

    if len(order) < graph.number_of_nodes():
        used = set(order)
        order.extend([node for node in graph.nodes if node not in used])
    return order


def hda_order(graph: nx.Graph, budget_ratio: float = 0.30) -> List[int]:
    h = graph.copy()
    order: List[int] = []
    budget = max(1, min(h.number_of_nodes(), int(round(h.number_of_nodes() * budget_ratio))))
    for _ in range(budget):
        if h.number_of_nodes() == 0:
            break
        node = max(h.nodes, key=lambda x: (h.degree[x], x))
        order.append(node)
        h.remove_node(node)
    used = set(order)
    order.extend([node for node in graph.nodes if node not in used])
    return order


def e26f_config() -> AlgoConfig:
    cfg = AlgoConfig(
        clade="reference_e26f",
        w_split=2.10,
        w_bridge_mult=2.20,
        w_degree=0.75,
        w_nds=0.45,
        w_core=0.18,
        w_comp=0.0,
        w_bridge_edges=0.0,
        split_power=1.0,
        degree_power=1.0,
        nds_power=1.0,
        core_power=0.5,
        update_radius=2,
        use_component=False,
        comp_refresh=25,
        name="e26f_reference",
    )
    cfg.assign_id()
    return cfg


def initial_configs(rng: random.Random) -> List[AlgoConfig]:
    """Return the single original HDA root.

    No typed structural seeds are inserted. This keeps the discovery protocol
    honest: every non-HDA algorithm must be generated as a descendant of HDA.
    """
    root = AlgoConfig(
        "hda_root",
        w_split=0.0,
        w_bridge_mult=0.0,
        w_degree=1.0,
        w_nds=0.0,
        w_core=0.0,
        update_radius=0,
        name="root_original_hda",
        mutation="root_original_hda",
    )
    root.assign_id()
    return [root]


CLADES = [
    "hda_root",
    "degree_core",
    "open_wedge",
    "bridge_aware",
    "core_scale",
    "component_aware",
    "local_bridge_edges",
]

SEARCH_CLADES = [c for c in CLADES if c != "hda_root"]


def random_config(rng: random.Random, clade: Optional[str] = None) -> AlgoConfig:
    clade = clade or rng.choice(SEARCH_CLADES)
    if clade == "hda_root":
        clade = rng.choice(SEARCH_CLADES)
    cfg = AlgoConfig(clade=clade)
    cfg.w_split = rng.uniform(0.0, 2.8)
    cfg.w_bridge_mult = rng.uniform(0.0, 3.2)
    cfg.w_degree = rng.uniform(0.05, 1.3)
    cfg.w_nds = rng.uniform(0.0, 0.8)
    cfg.w_core = rng.uniform(0.0, 0.45)
    cfg.w_comp = rng.uniform(0.0, 0.35) if clade == "component_aware" else 0.0
    cfg.w_bridge_edges = rng.uniform(0.0, 0.6) if clade == "local_bridge_edges" else 0.0
    cfg.split_power = rng.choice([0.75, 1.0, 1.15, 1.3])
    cfg.degree_power = rng.choice([0.75, 1.0, 1.15])
    cfg.nds_power = rng.choice([0.75, 1.0])
    cfg.core_power = rng.choice([0.5, 0.75, 1.0])
    cfg.update_radius = rng.choice([1, 1, 2])
    cfg.use_component = clade == "component_aware"
    cfg.comp_refresh = rng.choice([10, 20, 25, 40])
    return cfg


def mutate_config(parent: AlgoConfig, rng: random.Random, diagnostics: Dict[str, float]) -> AlgoConfig:
    if parent.clade == "hda_root":
        cfg = random_config(rng)
    else:
        cfg = AlgoConfig(**{k: v for k, v in asdict(parent).items() if k in AlgoConfig.__dataclass_fields__})
    cfg.parent_id = parent.node_id
    cfg.node_id = ""
    cfg.name = ""

    mutation_notes: List[str] = []
    if rng.random() < 0.12:
        cfg.clade = rng.choice(SEARCH_CLADES)
        mutation_notes.append("clade_shift")

    # Diagnostic nudges: high R needs scale pressure; low cNBI needs split/bridge pressure.
    if diagnostics.get("needs_scale", 0.0) > rng.random():
        cfg.w_degree *= rng.uniform(1.03, 1.25)
        cfg.w_nds *= rng.uniform(1.02, 1.20)
        cfg.w_core *= rng.uniform(1.02, 1.18)
        mutation_notes.append("scale_nudge")
    if diagnostics.get("needs_fragmentation", 0.0) > rng.random():
        cfg.w_split *= rng.uniform(1.05, 1.28)
        cfg.w_bridge_mult *= rng.uniform(1.05, 1.30)
        mutation_notes.append("fragmentation_nudge")
    if diagnostics.get("needs_speed", 0.0) > rng.random():
        cfg.update_radius = 1
        cfg.use_component = False if rng.random() < 0.7 else cfg.use_component
        cfg.w_bridge_edges *= rng.uniform(0.5, 0.9)
        mutation_notes.append("speed_nudge")

    fields = ["w_split", "w_bridge_mult", "w_degree", "w_nds", "w_core", "w_comp", "w_bridge_edges"]
    for _ in range(rng.randint(1, 3)):
        field_name = rng.choice(fields)
        val = getattr(cfg, field_name)
        if field_name == "w_comp" and cfg.clade != "component_aware":
            continue
        if field_name == "w_bridge_edges" and cfg.clade != "local_bridge_edges":
            continue
        setattr(cfg, field_name, max(0.0, val * rng.lognormvariate(0.0, 0.18) + rng.uniform(-0.05, 0.05)))
        mutation_notes.append(f"jitter_{field_name}")

    if rng.random() < 0.15:
        cfg.split_power = rng.choice([0.75, 1.0, 1.15, 1.3])
        mutation_notes.append("split_power")
    if rng.random() < 0.10:
        cfg.update_radius = 2 if cfg.update_radius == 1 else 1
        mutation_notes.append("radius_flip")
    if rng.random() < 0.08:
        cfg.use_component = not cfg.use_component
        cfg.w_comp = max(cfg.w_comp, rng.uniform(0.05, 0.25)) if cfg.use_component else 0.0
        mutation_notes.append("component_toggle")

    cfg.w_split = min(cfg.w_split, 4.0)
    cfg.w_bridge_mult = min(cfg.w_bridge_mult, 4.5)
    cfg.w_degree = min(cfg.w_degree, 2.0)
    cfg.w_nds = min(cfg.w_nds, 1.5)
    cfg.w_core = min(cfg.w_core, 1.0)
    cfg.w_comp = min(cfg.w_comp, 0.8)
    cfg.w_bridge_edges = min(cfg.w_bridge_edges, 1.2)
    cfg.mutation = "+".join(mutation_notes) or "neutral"
    cfg.assign_id()
    return cfg


def _coerce_float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        val = float(value)
    except Exception:
        val = default
    if not math.isfinite(val):
        val = default
    return min(hi, max(lo, val))


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return default


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = text.strip()
    if "```json" in raw:
        start = raw.find("```json") + len("```json")
        end = raw.find("```", start)
        if end != -1:
            raw = raw[start:end].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except Exception:
        return None


async def llm_mutate_config_async(
    parent: AlgoConfig,
    diagnostics: Dict[str, float],
    records: List[Dict[str, Any]],
    rng: random.Random,
    model: str,
    reasoning_effort: str = "",
    max_completion_tokens: int = 1024,
    timeout_s: float = 240.0,
) -> Optional[AlgoConfig]:
    """Ask an online LLM to mutate a typed config. Secrets are read only from env."""
    api_key = os.environ.get("SMAI_API_KEY")
    base_url = os.environ.get("SMAI_BASE_URL", "https://api.smai.ai/v1")
    if not api_key:
        print("[llm-warning] SMAI_API_KEY is not set; falling back to local mutation.", flush=True)
        return None
    top = sorted(records, key=lambda r: r.get("rank_score", 0.0), reverse=True)[:8]
    prompt = {
        "task": "Mutate one typed network dismantling heuristic configuration.",
        "objective": "Improve rank-composite performance over R lower, cNBI higher, and Time lower on 50 generated 500-node power-law graphs.",
        "constraints": {
            "return_json_only": True,
            "allowed_clades": SEARCH_CLADES,
            "must_keep_typed_config": True,
            "do_not_generate_python_code": True,
            "ranges": {
                "w_split": [0.0, 4.0],
                "w_bridge_mult": [0.0, 4.5],
                "w_degree": [0.0, 2.0],
                "w_nds": [0.0, 1.5],
                "w_core": [0.0, 1.0],
                "w_comp": [0.0, 0.8],
                "w_bridge_edges": [0.0, 1.2],
                "split_power": [0.75, 1.0, 1.15, 1.3],
                "degree_power": [0.75, 1.0, 1.15],
                "nds_power": [0.75, 1.0],
                "core_power": [0.5, 0.75, 1.0],
                "update_radius": [1, 2],
                "comp_refresh": [10, 20, 25, 40],
            },
        },
        "parent_config": asdict(parent),
        "diagnostics": diagnostics,
        "top_records": [
            {
                "node_id": r["node_id"],
                "clade": r["clade"],
                "R": r["R"],
                "cNBI": r["cNBI"],
                "Time": r["Time"],
                "rank_score": r.get("rank_score", 0.0),
                "weights": {k: r[k] for k in r if k.startswith("w_")},
            }
            for r in top
        ],
        "knowledge_injection": (
            "You start from the original HDA algorithm: repeatedly remove the current "
            "highest-degree node. HDA is simple and scalable, but high-degree nodes "
            "are not always the best removal targets. You may consider general "
            "network-dismantling clues such as local redundancy, clustering, bridge-like "
            "roles, k-core or core-like priors, neighbor-degree mass, residual component "
            "concentration, and efficient adaptive updates. Do not assume any existing "
            "algorithm is the target. The goal is to discover a strong HDA descendant "
            "from first principles rather than reproduce a known formula."
        ),
        "complexity_requirement": (
            "This is an iterative network dismantling algorithm: after removing one node, "
            "the graph state is updated and the next node is selected. The total time "
            "complexity must be below O(N^2), where N is the number of nodes. It should "
            "scale to million-node networks. Prefer local updates, lazy heaps, local "
            "neighborhood statistics, and incrementally maintainable quantities. Avoid "
            "per-step betweenness centrality, all-pairs shortest paths, global community "
            "detection, spectral decomposition, or training-heavy message passing. If a "
            "new structural signal is introduced, explain how it can be approximated or "
            "updated locally."
        ),
        "output_schema": {
            "clade": "string",
            "w_split": "float",
            "w_bridge_mult": "float",
            "w_degree": "float",
            "w_nds": "float",
            "w_core": "float",
            "w_comp": "float",
            "w_bridge_edges": "float",
            "split_power": "float",
            "degree_power": "float",
            "nds_power": "float",
            "core_power": "float",
            "update_radius": "int",
            "use_component": "bool",
            "comp_refresh": "int",
            "mutation": "short explanation",
        },
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful AI algorithm designer. Return only valid JSON. "
                "You are optimizing typed network dismantling heuristic configs."
            ),
        },
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]
    if openai is not None:
        try:
            client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)
            request_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": 0.55,
                "max_completion_tokens": max_completion_tokens,
            }
            effort = (reasoning_effort or "").strip().lower()
            if effort and effort not in {"default", "medium"}:
                request_kwargs["reasoning_effort"] = reasoning_effort
            if effort and effort not in {"none", "off", "default"}:
                request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            resp = await client.chat.completions.create(**request_kwargs)
            content = resp.choices[0].message.content or ""
        except Exception as exc:
            print(f"[llm-warning] sdk mutation call failed: {type(exc).__name__}: {exc}", flush=True)
            return None
    else:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": 0.55,
            "max_completion_tokens": max_completion_tokens,
        }
        effort = (reasoning_effort or "").strip().lower()
        if effort and effort not in {"default", "medium"}:
            payload["reasoning_effort"] = reasoning_effort
        if effort and effort not in {"none", "off", "default"}:
            payload["thinking"] = {"type": "enabled"}
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = base_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8")
            obj_resp = json.loads(body)
            content = obj_resp["choices"][0]["message"].get("content") or ""
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"[llm-warning] http mutation call failed: HTTP {exc.code}: {body[:300]}", flush=True)
            return None
        except Exception as exc:
            print(f"[llm-warning] http mutation call failed: {type(exc).__name__}: {exc}", flush=True)
            return None
    with (OUTPUT_DIR / "llm_prompts_responses.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(
            {
                "parent_id": parent.node_id,
                "parent_clade": parent.clade,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "max_completion_tokens": max_completion_tokens,
                "prompt": prompt,
                "response": content,
            },
            ensure_ascii=False,
        ) + "\n")
    obj = _extract_json_object(content)
    if not obj:
        return None
    cfg = AlgoConfig(clade=str(obj.get("clade", parent.clade)))
    if cfg.clade not in SEARCH_CLADES:
        cfg.clade = parent.clade if parent.clade in SEARCH_CLADES else rng.choice(SEARCH_CLADES)
    cfg.w_split = _coerce_float(obj.get("w_split"), parent.w_split, 0.0, 4.0)
    cfg.w_bridge_mult = _coerce_float(obj.get("w_bridge_mult"), parent.w_bridge_mult, 0.0, 4.5)
    cfg.w_degree = _coerce_float(obj.get("w_degree"), parent.w_degree, 0.0, 2.0)
    cfg.w_nds = _coerce_float(obj.get("w_nds"), parent.w_nds, 0.0, 1.5)
    cfg.w_core = _coerce_float(obj.get("w_core"), parent.w_core, 0.0, 1.0)
    cfg.w_comp = _coerce_float(obj.get("w_comp"), parent.w_comp, 0.0, 0.8)
    cfg.w_bridge_edges = _coerce_float(obj.get("w_bridge_edges"), parent.w_bridge_edges, 0.0, 1.2)
    cfg.split_power = min([0.75, 1.0, 1.15, 1.3], key=lambda x: abs(x - _coerce_float(obj.get("split_power"), parent.split_power, 0.75, 1.3)))
    cfg.degree_power = min([0.75, 1.0, 1.15], key=lambda x: abs(x - _coerce_float(obj.get("degree_power"), parent.degree_power, 0.75, 1.15)))
    cfg.nds_power = min([0.75, 1.0], key=lambda x: abs(x - _coerce_float(obj.get("nds_power"), parent.nds_power, 0.75, 1.0)))
    cfg.core_power = min([0.5, 0.75, 1.0], key=lambda x: abs(x - _coerce_float(obj.get("core_power"), parent.core_power, 0.5, 1.0)))
    cfg.update_radius = 2 if int(_coerce_float(obj.get("update_radius"), parent.update_radius, 1, 2)) >= 2 else 1
    cfg.use_component = _coerce_bool(obj.get("use_component"), parent.use_component)
    cfg.comp_refresh = min([10, 20, 25, 40], key=lambda x: abs(x - int(_coerce_float(obj.get("comp_refresh"), parent.comp_refresh, 10, 40))))
    if cfg.clade != "component_aware" and not cfg.use_component:
        cfg.w_comp = 0.0
    if cfg.clade != "local_bridge_edges":
        cfg.w_bridge_edges = 0.0
    cfg.parent_id = parent.node_id
    cfg.mutation = "llm:" + str(obj.get("mutation", "typed_config_mutation"))[:120].replace("\n", " ")
    cfg.assign_id()
    return cfg


def llm_mutate_config(
    parent: AlgoConfig,
    diagnostics: Dict[str, float],
    records: List[Dict[str, Any]],
    rng: random.Random,
    model: str,
    reasoning_effort: str = "",
    max_completion_tokens: int = 1024,
    timeout_s: float = 240.0,
) -> Optional[AlgoConfig]:
    return asyncio.run(
        llm_mutate_config_async(
            parent,
            diagnostics,
            records,
            rng,
            model,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            timeout_s=timeout_s,
        )
    )


def evaluate_config(
    cfg: AlgoConfig,
    graphs: Sequence[nx.Graph],
    budget_ratio: float,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    rows = []
    algo_time = 0.0
    for idx, graph in enumerate(graphs):
        a0 = time.perf_counter()
        if cfg.clade == "hda_root":
            order = hda_order(graph, budget_ratio=budget_ratio)
        else:
            order = degree_order_by_config(graph, cfg, budget_ratio=budget_ratio)
        algo_time += time.perf_counter() - a0
        metrics = evaluate_order(graph, order, budget_ratio=budget_ratio)
        metrics["graph_idx"] = idx
        rows.append(metrics)
    avg = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ["R", "cNBI", "avg_gcc", "avg_top5_mass", "avg_hhi", "avg_pairdisc"]
    }
    avg["Time"] = algo_time / len(graphs)
    avg["wall_time"] = time.perf_counter() - t0
    return {"avg": avg, "per_graph": rows}


def evaluate_hda(graphs: Sequence[nx.Graph], budget_ratio: float) -> Dict[str, Any]:
    rows = []
    algo_time = 0.0
    for idx, graph in enumerate(graphs):
        t0 = time.perf_counter()
        order = hda_order(graph, budget_ratio=budget_ratio)
        algo_time += time.perf_counter() - t0
        metrics = evaluate_order(graph, order, budget_ratio=budget_ratio)
        metrics["graph_idx"] = idx
        rows.append(metrics)
    avg = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ["R", "cNBI", "avg_gcc", "avg_top5_mass", "avg_hhi", "avg_pairdisc"]
    }
    avg["Time"] = algo_time / len(graphs)
    return {"avg": avg, "per_graph": rows}


def add_rank_scores(records: List[Dict[str, Any]]) -> None:
    valid = [r for r in records if r.get("valid", True)]
    if not valid:
        return

    def rank_values(key: str, reverse: bool = False) -> Dict[str, float]:
        ordered = sorted(valid, key=lambda r: r[key], reverse=reverse)
        denom = max(1, len(ordered) - 1)
        return {r["node_id"]: 1.0 - i / denom for i, r in enumerate(ordered)}

    rank_r = rank_values("R", reverse=False)
    rank_c = rank_values("cNBI", reverse=True)
    rank_t = rank_values("Time", reverse=False)
    for r in records:
        if not r.get("valid", True):
            r["rank_score"] = -1.0
            continue
        r["rank_R"] = rank_r[r["node_id"]]
        r["rank_cNBI"] = rank_c[r["node_id"]]
        r["rank_Time"] = rank_t[r["node_id"]]
        r["rank_score"] = 0.4 * r["rank_R"] + 0.3 * r["rank_cNBI"] + 0.3 * r["rank_Time"]


def clade_diagnostics(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    if not records:
        return {clade: {"needs_scale": 0.5, "needs_fragmentation": 0.5, "needs_speed": 0.2} for clade in CLADES}
    r_med = float(np.median([r["R"] for r in records]))
    c_med = float(np.median([r["cNBI"] for r in records]))
    t_med = float(np.median([r["Time"] for r in records]))
    out = {}
    for clade in CLADES:
        rs = [r for r in records if r["clade"] == clade]
        if not rs:
            out[clade] = {"needs_scale": 0.6, "needs_fragmentation": 0.6, "needs_speed": 0.2}
            continue
        best = max(rs, key=lambda r: r.get("rank_score", 0.0))
        out[clade] = {
            "needs_scale": min(0.95, max(0.05, (best["R"] - r_med) / max(r_med, 1e-9) + 0.45)),
            "needs_fragmentation": min(0.95, max(0.05, (c_med - best["cNBI"]) / max(c_med, 1e-9) + 0.45)),
            "needs_speed": min(0.95, max(0.05, (best["Time"] - t_med) / max(t_med, 1e-9) + 0.20)),
        }
    return out


def select_clade(records: List[Dict[str, Any]], rng: random.Random, temperature: float = 0.10) -> str:
    if records and all(r["clade"] == "hda_root" for r in records):
        return "hda_root"
    by_clade = defaultdict(list)
    for r in records:
        by_clade[r["clade"]].append(r)
    samples = []
    for clade in SEARCH_CLADES:
        rs = by_clade.get(clade, [])
        if not rs:
            samples.append((10.0, clade))
            continue
        scores = [r.get("rank_score", 0.0) for r in rs]
        mean = float(np.mean(scores))
        uncertainty = 1.0 / math.sqrt(len(scores))
        sample = rng.gauss(mean, temperature + uncertainty * 0.15)
        # mild exploration bonus for under-sampled clades
        sample += 0.08 / math.sqrt(len(scores))
        samples.append((sample, clade))
    return max(samples, key=lambda x: x[0])[1]


def select_parent(records: List[Dict[str, Any]], clade: str, rng: random.Random) -> Dict[str, Any]:
    pool = [r for r in records if r["clade"] == clade]
    if not pool:
        return max(records, key=lambda r: r.get("rank_score", 0.0))
    pool = sorted(pool, key=lambda r: r.get("rank_score", 0.0), reverse=True)
    top_k = pool[: max(2, min(8, len(pool)))]
    weights = np.array([max(1e-6, r.get("rank_score", 0.0)) ** 2 for r in top_k], dtype=float)
    weights = weights / weights.sum()
    idx = int(rng.choices(range(len(top_k)), weights=weights, k=1)[0])
    return top_k[idx]


def config_to_code(cfg: AlgoConfig) -> str:
    payload = json.dumps(asdict(cfg), ensure_ascii=False, sort_keys=True, indent=2)
    if cfg.clade == "hda_root":
        return f'''# -*- coding: utf-8 -*-
"""
Exported original HDA root from DACTS search.
Config:
{payload}
"""

def degree_order(G):
    H = G.copy()
    order = []
    while H.number_of_nodes() > 0:
        node = max(H.nodes, key=lambda x: (H.degree[x], x))
        order.append(node)
        H.remove_node(node)
    return order
'''
    return f'''# -*- coding: utf-8 -*-
"""
Exported typed dismantling heuristic from DACTS search.
Config:
{payload}
"""

def degree_order(G):
    import math
    import heapq
    import networkx as nx

    cfg = {json.dumps(asdict(cfg), ensure_ascii=False, sort_keys=True)}
    if G.number_of_nodes() == 0:
        return []
    H = nx.convert_node_labels_to_integers(G, first_label=0, label_attribute="old_label")
    old_labels = nx.get_node_attributes(H, "old_label")
    nodes = list(H.nodes)
    alive = set(nodes)
    nbrs = {{u: set(H.neighbors(u)) for u in nodes}}
    deg = {{u: len(nbrs[u]) for u in nodes}}
    try:
        core = nx.core_number(H) if H.number_of_edges() > 0 else {{u: 0 for u in nodes}}
    except Exception:
        core = {{u: 0 for u in nodes}}
    triangles = {{}}
    for u in nodes:
        Nu = nbrs[u]
        triangles[u] = 0 if len(Nu) < 2 else sum(len(Nu & nbrs[v]) for v in Nu) // 2
    neigh_deg_sum = {{u: sum(deg[v] for v in nbrs[u]) for u in nodes}}
    version = {{u: 0 for u in nodes}}

    def score(u):
        if u not in alive:
            return -1e30
        d = deg.get(u, 0)
        if d <= 0:
            return -1e30
        total_pairs = d * (d - 1) / 2.0
        tri = min(float(triangles.get(u, 0)), total_pairs)
        split_pairs = max(0.0, total_pairs - tri)
        bridge_factor = 1.0 - (tri / total_pairs if total_pairs > 0 else 1.0)
        split = cfg["w_split"] * math.log1p((split_pairs ** cfg["split_power"]) * (1.0 + cfg["w_bridge_mult"] * bridge_factor))
        degree_term = cfg["w_degree"] * math.log1p(d ** cfg["degree_power"])
        nds_term = cfg["w_nds"] * math.log1p(max(0.0, neigh_deg_sum.get(u, 0)) ** cfg["nds_power"])
        core_term = cfg["w_core"] * ((core.get(u, 0) + 1.0) ** cfg["core_power"])
        return split + degree_term + nds_term + core_term

    heap = []
    for u in nodes:
        heapq.heappush(heap, (-score(u), -deg[u], u, version[u]))
    budget = len(nodes)
    order = []
    for _ in range(budget):
        while heap:
            _, _, u, vu = heapq.heappop(heap)
            if u in alive and vu == version[u]:
                break
        else:
            if not alive:
                break
            u = max(alive, key=lambda x: (score(x), deg.get(x, 0)))
        order.append(old_labels.get(u, u))
        Nu = set(nbrs[u] & alive)
        for v in Nu:
            common = len((nbrs[v] & alive) & Nu)
            if common > 0:
                triangles[v] = max(0, triangles.get(v, 0) - common)
        alive.remove(u)
        for v in Nu:
            nbrs[v].discard(u)
            deg[v] = len(nbrs[v] & alive)
        deg[u] = 0
        nbrs[u].clear()
        triangles[u] = 0
        neigh_deg_sum[u] = 0
        touched = set(Nu)
        if cfg["update_radius"] >= 2:
            for v in list(Nu):
                touched.update(nbrs[v] & alive)
        for x in touched:
            if x not in alive:
                continue
            neigh_deg_sum[x] = sum(deg[y] for y in nbrs[x] if y in alive)
            version[x] += 1
            heapq.heappush(heap, (-score(x), -deg.get(x, 0), x, version[x]))
    return order
'''


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_search(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    graphs_path = OUTPUT_DIR / f"graphs_{args.graphs}_powerlaw{args.nodes}.json"
    if args.reuse_graphs and graphs_path.exists():
        graphs_payload = json.loads(graphs_path.read_text(encoding="utf-8"))
        graphs = [
            json_graph.node_link_graph(item, edges="edges" if "edges" in item else "links")
            for item in graphs_payload
        ]
    else:
        graphs = make_graph_suite(args.graphs, args.nodes, args.gamma, args.seed)
        graphs_payload = [json_graph.node_link_data(g) for g in graphs]
        graphs_path.write_text(json.dumps(graphs_payload), encoding="utf-8")

    graph_stats = [
        {
            "idx": idx,
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "avg_degree": 2.0 * g.number_of_edges() / g.number_of_nodes(),
            "clustering": nx.average_clustering(g),
        }
        for idx, g in enumerate(graphs)
    ]
    write_csv(OUTPUT_DIR / "graph_stats.csv", graph_stats)

    records: List[Dict[str, Any]] = []
    configs: Dict[str, AlgoConfig] = {}
    seen = set()

    def eval_and_record(cfg: AlgoConfig, stage: str) -> None:
        if not cfg.node_id:
            cfg.assign_id()
        if cfg.canonical() in seen:
            return
        seen.add(cfg.canonical())
        configs[cfg.node_id] = cfg
        result = evaluate_config(cfg, graphs, args.budget_ratio)
        avg = result["avg"]
        rec = {
            "idx": len(records),
            "stage": stage,
            "node_id": cfg.node_id,
            "parent_id": cfg.parent_id,
            "clade": cfg.clade,
            "mutation": cfg.mutation,
            "R": avg["R"],
            "cNBI": avg["cNBI"],
            "Time": avg["Time"],
            "avg_top5_mass": avg["avg_top5_mass"],
            "avg_hhi": avg["avg_hhi"],
            "avg_pairdisc": avg["avg_pairdisc"],
            "wall_time": avg["wall_time"],
            "valid": True,
            **{k: v for k, v in asdict(cfg).items() if k.startswith("w_") or k.endswith("_power") or k in ["update_radius", "use_component", "comp_refresh"]},
        }
        records.append(rec)
        add_rank_scores(records)
        if len(records) % args.checkpoint_every == 0:
            write_csv(OUTPUT_DIR / "search_records.csv", records)

    for cfg in initial_configs(rng):
        eval_and_record(cfg, "seed")

    while len(records) < args.nodes_to_search:
        add_rank_scores(records)
        diags = clade_diagnostics(records)
        clade = select_clade(records, rng)
        parent_rec = select_parent(records, clade, rng)
        parent_cfg = configs[parent_rec["node_id"]]
        child = None
        if args.use_llm and (len(records) % max(1, args.llm_every) == 0):
            child = llm_mutate_config(
                parent_cfg,
                diags.get(clade, {}),
                records,
                rng,
                args.model,
                reasoning_effort=args.reasoning_effort,
                max_completion_tokens=args.max_completion_tokens,
                timeout_s=args.llm_timeout_s,
            )
        if child is None:
            child = mutate_config(parent_cfg, rng, diags.get(clade, {}))
        eval_and_record(child, "search")
        if len(records) % 25 == 0:
            best = max(records, key=lambda r: r.get("rank_score", 0.0))
            print(
                f"[{len(records):04d}] best={best['node_id']} clade={best['clade']} "
                f"score={best.get('rank_score', 0):.4f} R={best['R']:.6f} "
                f"cNBI={best['cNBI']:.6f} Time={best['Time']:.6f}",
                flush=True,
            )

    add_rank_scores(records)
    records = sorted(records, key=lambda r: r.get("rank_score", 0.0), reverse=True)
    for idx, rec in enumerate(records):
        rec["final_rank"] = idx + 1
    write_csv(OUTPUT_DIR / "search_records.csv", records)

    ref_e26f = e26f_config()
    e26f_result = evaluate_config(ref_e26f, graphs, args.budget_ratio)
    hda_result = evaluate_hda(graphs, args.budget_ratio)
    references = []
    for name, result in [("e26f_reference", e26f_result), ("HDA", hda_result)]:
        avg = result["avg"]
        references.append({
            "name": name,
            "R": avg["R"],
            "cNBI": avg["cNBI"],
            "Time": avg["Time"],
            "avg_top5_mass": avg["avg_top5_mass"],
            "avg_hhi": avg["avg_hhi"],
            "avg_pairdisc": avg["avg_pairdisc"],
        })
    write_csv(OUTPUT_DIR / "reference_comparison.csv", references)

    top = records[: min(20, len(records))]
    for rec in top[:10]:
        cfg = configs[rec["node_id"]]
        (CANDIDATE_DIR / f"candidate_{rec['final_rank']:02d}_{rec['node_id']}.json").write_text(
            json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (CANDIDATE_DIR / f"candidate_{rec['final_rank']:02d}_{rec['node_id']}.py").write_text(
            config_to_code(cfg),
            encoding="utf-8",
        )

    summary = {
        "args": vars(args),
        "graphs_path": str(graphs_path),
        "graph_count": len(graphs),
        "target_graph_nodes": args.nodes,
        "n_records": len(records),
        "best": records[0],
        "top10": top[:10],
        "references": references,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary)
    write_algorithm_flow()


def write_report(summary: Dict[str, Any]) -> None:
    best = summary["best"]
    refs = {r["name"]: r for r in summary["references"]}
    e26f = refs["e26f_reference"]
    hda = refs["HDA"]
    lines = [
        "# HDA-Root DACTS Search Report",
        "",
        f"- Evaluated typed algorithm nodes: {summary['n_records']}",
        f"- Graph suite: {summary['graph_count']} generated power-law graphs, target n={summary['target_graph_nodes']}",
        f"- Objective: rank composite = 0.4 * rank_R + 0.3 * rank_cNBI + 0.3 * rank_Time",
        "",
        "## Best discovered candidate",
        "",
        f"- node_id: `{best['node_id']}`",
        f"- clade: `{best['clade']}`",
        f"- parent_id: `{best['parent_id']}`",
        f"- mutation: `{best['mutation']}`",
        f"- R: {best['R']:.6f}",
        f"- cNBI: {best['cNBI']:.6f}",
        f"- Time: {best['Time']:.6f} s/graph",
        f"- rank_score: {best['rank_score']:.6f}",
        "",
        "## Reference comparison",
        "",
        "| method | R lower | cNBI higher | Time lower |",
        "| --- | ---: | ---: | ---: |",
        f"| best discovered | {best['R']:.6f} | {best['cNBI']:.6f} | {best['Time']:.6f} |",
        f"| e26f reference | {e26f['R']:.6f} | {e26f['cNBI']:.6f} | {e26f['Time']:.6f} |",
        f"| HDA | {hda['R']:.6f} | {hda['cNBI']:.6f} | {hda['Time']:.6f} |",
        "",
        "## Top candidates",
        "",
        "| rank | node_id | clade | R | cNBI | Time | score |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for rec in summary["top10"]:
        lines.append(
            f"| {rec['final_rank']} | `{rec['node_id']}` | {rec['clade']} | "
            f"{rec['R']:.6f} | {rec['cNBI']:.6f} | {rec['Time']:.6f} | {rec['rank_score']:.6f} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- The searched tree has exactly one root: original unoptimized HDA.",
        "- `e26f_reference` was evaluated as a fixed reference, not inserted into the searched node pool.",
        "- LLM prompts/responses are logged in `outputs/llm_prompts_responses.jsonl` when online mutation is enabled.",
        "- Top candidate JSON/Python exports are in `candidates/` under this experiment folder.",
    ])
    (REPORT_DIR / "search_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_algorithm_flow() -> None:
    lines = [
        "# Modified HDA-Root Tree Search Flow",
        "",
        "## Core Flow",
        "",
        "1. Build the fixed graph suite using the old configuration-model power-law generator.",
        "2. Evaluate the single root algorithm: original HDA, i.e. repeatedly remove the current highest-degree node after each graph update.",
        "3. Diagnose the current record pool with three signals: high R means insufficient dismantling pressure, low cNBI means fragmentation is too late or too concentrated, high runtime means the candidate violates the scalability direction.",
        "4. Select a non-root structural clade by clade-level sampling. If only HDA exists, expand HDA once to create the first descendant.",
        "5. Ask the LLM, or the local fallback mutator, to generate a typed descendant configuration under the hard complexity constraint: iterative update, total time below O(N^2), no global per-step centralities or spectral/community routines.",
        "6. Execute the candidate as a lazy-heap local-update dismantling algorithm and evaluate R, cNBI, and runtime on all generated graphs.",
        "7. Re-rank all nodes by rank_score = 0.4 * rank_R + 0.3 * rank_cNBI + 0.3 * rank_Time, checkpoint records, and repeat until the node budget is reached.",
        "8. After the search ends, evaluate e26f as an external reference only, then export the top candidate configs and executable Python code.",
        "",
        "## Changes From Old ERA-like",
        "",
        "- Root: old ERA-like could start from broad seeds or prior candidates; this version starts only from original HDA.",
        "- Search unit: old ERA-like mainly searched candidate implementations/variants; this version searches typed dismantling-program configurations that compile into executable algorithms.",
        "- Selection: old ERA-like uses node-level UCB/ERA-like pressure; this version uses clade-level budget allocation so structurally different ideas receive exploration budget.",
        "- Feedback: old ERA-like mostly uses scalar reward; this version exposes diagnostic feedback to mutation: scale pressure, fragmentation pressure, and speed pressure.",
        "- Objective: this version explicitly optimizes R lower, cNBI higher, and Time lower, with cNBI rewarding early and distributed fragmentation.",
        "- Complexity: this version makes sub-O(N^2) iterative scalability a first-class constraint in the mutation prompt and implementation template.",
        "- e26f handling: e26f is not injected as a seed and not named as a recipe; it is evaluated only after the search for comparison.",
    ]
    (REPORT_DIR / "algorithm_flow.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes-to-search", type=int, default=500)
    parser.add_argument("--graphs", type=int, default=50)
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=2.5)
    parser.add_argument("--budget-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--reuse-graphs", action="store_true")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-every", type=int, default=1)
    parser.add_argument("--model", type=str, default="gpt-5.5")
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="",
        help="Use none to disable reasoning; omit for provider default, usually medium.",
    )
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--llm-timeout-s", type=float, default=240.0)
    args = parser.parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
