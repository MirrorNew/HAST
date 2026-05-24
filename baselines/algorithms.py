# -*- coding: utf-8 -*-
"""Self-contained baseline algorithms for network dismantling experiments.

All functions return a full node order. The evaluated prefix is controlled by
the caller through the removal rate, which keeps baseline code independent from
metric code.
"""

from __future__ import annotations

import heapq
from typing import Any, Iterable, List, Mapping

import networkx as nx


def _node_key(node: Any) -> tuple[int, Any]:
    try:
        return (0, int(node))
    except (TypeError, ValueError):
        return (1, str(node))


def complete_order(graph: nx.Graph, order: Iterable[Any]) -> List[Any]:
    """Keep valid unique nodes from ``order`` and append all missing nodes."""
    nodes = set(graph.nodes())
    out: List[Any] = []
    seen = set()
    for node in order:
        if node in nodes and node not in seen:
            seen.add(node)
            out.append(node)
    out.extend([node for node in graph.nodes if node not in seen])
    return out


def score_order(graph: nx.Graph, scores: Mapping[Any, float]) -> List[Any]:
    """Return a deterministic descending-score order and append missing nodes."""
    order = [node for node, _ in sorted(scores.items(), key=lambda item: (-float(item[1]), _node_key(item[0])))]
    return complete_order(graph, order)


def _budget(graph: nx.Graph, rate: float | None) -> int:
    n = graph.number_of_nodes()
    if rate is None:
        return n
    return max(1, min(n, int(round(n * rate))))


def hda_original_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    """Original HDA: repeatedly rescan the residual graph for max degree."""
    h = graph.copy()
    order: List[Any] = []
    for _ in range(_budget(graph, rate)):
        if h.number_of_nodes() == 0:
            break
        node = max(h.nodes, key=lambda u: (h.degree[u], str(u)))
        order.append(node)
        h.remove_node(node)
    return complete_order(graph, order)


def hda_fast_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    """Lazy-heap HDA used only when an explicitly fast baseline is needed."""
    h = graph.to_undirected() if graph.is_directed() else graph
    alive = set(h.nodes())
    nbrs = {u: set(h.neighbors(u)) for u in h.nodes()}
    degree = {u: len(nbrs[u]) for u in h.nodes()}
    version = {u: 0 for u in h.nodes()}
    heap = [(-degree[u], str(u), u, version[u]) for u in h.nodes()]
    heapq.heapify(heap)
    order: List[Any] = []
    target = _budget(graph, rate)
    while alive and len(order) < target:
        while heap:
            _, _, u, vu = heapq.heappop(heap)
            if u in alive and vu == version[u]:
                break
        else:
            u = max(alive, key=lambda x: (degree.get(x, 0), str(x)))
        if u not in alive:
            continue
        order.append(u)
        alive.remove(u)
        for v in list(nbrs[u]):
            if v not in alive:
                continue
            nbrs[v].discard(u)
            degree[v] = len(nbrs[v] & alive)
            version[v] += 1
            heapq.heappush(heap, (-degree[v], str(v), v, version[v]))
        nbrs[u].clear()
    return complete_order(graph, order)


def dc_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    """Static degree centrality baseline."""
    order = sorted(graph.nodes(), key=lambda u: (graph.degree[u], str(u)), reverse=True)
    return complete_order(graph, order)


def kcore_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    """Static k-core baseline, tie-broken by degree."""
    if graph.number_of_edges() == 0:
        return dc_order(graph, rate)
    core = nx.core_number(nx.Graph(graph))
    order = sorted(graph.nodes(), key=lambda u: (core.get(u, 0), graph.degree[u], _node_key(u)), reverse=True)
    return complete_order(graph, order)


def cluc_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    """CLUC/ClusterRank-style static local-clustering baseline."""
    if graph.number_of_nodes() == 0:
        return []
    h = nx.Graph(graph)
    clustering = nx.clustering(h)
    degree = dict(h.degree())
    scores = {}
    for node in h.nodes:
        neighbor_pressure = sum(degree[neighbor] + 1 for neighbor in h.neighbors(node))
        scores[node] = (10.0 ** (-clustering[node])) * neighbor_pressure
    return score_order(graph, scores)


def ci_order(graph: nx.Graph, rate: float | None = None, radius: int = 2) -> List[Any]:
    """Collective Influence baseline with radius 2 by default."""
    h = nx.Graph(graph)
    degree = dict(h.degree())
    scores: dict[Any, float] = {}
    for node in h.nodes:
        if degree[node] <= 1:
            scores[node] = 0.0
            continue
        lengths = nx.single_source_shortest_path_length(h, node, cutoff=radius)
        boundary = [other for other, distance in lengths.items() if distance == radius]
        scores[node] = float((degree[node] - 1) * sum(max(0, degree[other] - 1) for other in boundary))
    return score_order(graph, scores)


def _communities_by_label_propagation(graph: nx.Graph) -> dict[Any, str]:
    if graph.number_of_nodes() == 0:
        return {}
    communities: dict[Any, str] = {}
    try:
        iterator = nx.community.label_propagation_communities(graph)
        for idx, community in enumerate(iterator):
            for node in community:
                communities[node] = str(idx)
    except Exception:
        communities = {}
    for node in graph.nodes:
        communities.setdefault(node, str(len(communities)))
    return communities


def nd_order(graph: nx.Graph, method: str = "ndjc", beta: float = 0.5) -> List[Any]:
    """NDC/NCDC/NDJC static fallback formulas.

    These are transparent NetworkX fallbacks for local redundancy baselines.
    They are not claimed as official package reproductions.
    """
    method = method.lower()
    if method not in {"ndc", "ncdc", "ndjc"}:
        raise ValueError("method must be one of: ndc, ncdc, ndjc")
    h = nx.Graph(graph)
    adjacency = {node: set(h.neighbors(node)) for node in h.nodes}
    communities = _communities_by_label_propagation(h)
    scores: dict[Any, float] = {}
    for node, neighbors in adjacency.items():
        degree = len(neighbors)
        if degree == 0:
            scores[node] = 0.0
            continue
        total = 0.0
        for neighbor in neighbors:
            value = float(len(neighbors - adjacency[neighbor]))
            if method in {"ncdc", "ndjc"} and communities[node] == communities[neighbor]:
                value *= beta
            if method == "ndjc":
                union_size = len(neighbors | adjacency[neighbor])
                value = 0.0 if union_size == 0 else value / union_size
            total += value
        scores[node] = total / degree
    return score_order(graph, scores)


def ndc_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    return nd_order(graph, method="ndc")


def ncdc_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    return nd_order(graph, method="ncdc")


def ndjc_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    return nd_order(graph, method="ndjc")


def betweenness_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    """Static betweenness centrality baseline for small and medium graphs."""
    h = nx.Graph(graph)
    if h.number_of_nodes() <= 1:
        return complete_order(graph, h.nodes())
    return score_order(graph, nx.betweenness_centrality(h, normalized=True))


def _largest_component_subgraph(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph.copy()
    component = max(nx.connected_components(graph), key=lambda c: (len(c), tuple(sorted(c, key=_node_key))))
    return graph.subgraph(component).copy()


def corehd_original_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    """Original-style CoreHD: recompute the 2-core after every removal."""
    h = nx.Graph(graph)
    order: List[Any] = []
    for _ in range(_budget(graph, rate)):
        if h.number_of_nodes() == 0:
            break
        if h.number_of_edges() > 0:
            core = nx.k_core(h, k=2)
        else:
            core = nx.Graph()
        pool = core if core.number_of_nodes() else h
        node = max(pool.nodes, key=lambda u: (pool.degree[u], _node_key(u)))
        order.append(node)
        h.remove_node(node)
    return complete_order(graph, order)


def bpd_minsum_fallback_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    """BPD/MinSum-style fallback: CoreHD-like decycling on the largest component.

    This is a practical Python fallback for smoke tests and comparison plumbing,
    not an official BPD or Min-Sum reproduction.
    """
    h = nx.Graph(graph)
    order: List[Any] = []
    for _ in range(_budget(graph, rate)):
        if h.number_of_nodes() == 0:
            break
        component = _largest_component_subgraph(h)
        if component.number_of_edges() > 0:
            core = nx.k_core(component, k=2)
        else:
            core = nx.Graph()
        pool = core if core.number_of_nodes() else component
        node = max(pool.nodes, key=lambda u: (pool.degree[u], _node_key(u)))
        order.append(node)
        h.remove_node(node)
    return complete_order(graph, order)


def gnd_fallback_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    """GND-style fallback: articulation-first greedy on the largest component."""
    h = nx.Graph(graph)
    n0 = max(1, h.number_of_nodes())
    order: List[Any] = []
    for _ in range(_budget(graph, rate)):
        if h.number_of_nodes() == 0:
            break
        component = _largest_component_subgraph(h)
        arts = list(nx.articulation_points(component))
        if arts:
            base = component.number_of_nodes()
            best = None
            best_score = -1.0
            for node in arts:
                trial = component.copy()
                trial.remove_node(node)
                largest = max((len(c) for c in nx.connected_components(trial)), default=0)
                score = (base - largest) / n0
                if score > best_score or (score == best_score and _node_key(node) < _node_key(best)):
                    best = node
                    best_score = score
            node = best
        else:
            scores = nx.betweenness_centrality(component, normalized=True) if component.number_of_nodes() > 1 else {}
            node = score_order(component, scores)[0] if scores else max(component.nodes, key=lambda u: (component.degree[u], _node_key(u)))
        order.append(node)
        h.remove_node(node)
    return complete_order(graph, order)


def corehd_fast_order(graph: nx.Graph, rate: float | None = None) -> List[Any]:
    """Online CoreHD-fast: remove max residual degree in the current 2-core."""
    h = graph.to_undirected() if graph.is_directed() else graph
    nodes = list(h.nodes())
    alive = set(nodes)
    nbrs = {u: set(h.neighbors(u)) for u in nodes}
    deg = {u: len(nbrs[u]) for u in nodes}

    core = set(nodes)
    core_deg = {u: deg[u] for u in nodes}
    queue = [u for u in nodes if core_deg[u] < 2]
    while queue:
        u = queue.pop()
        if u not in core:
            continue
        core.remove(u)
        for v in nbrs[u]:
            if v in core:
                core_deg[v] -= 1
                if core_deg[v] < 2:
                    queue.append(v)

    version = {u: 0 for u in nodes}
    core_heap = [(-deg[u], str(u), u, version[u]) for u in core]
    all_heap = [(-deg[u], str(u), u, version[u]) for u in nodes]
    heapq.heapify(core_heap)
    heapq.heapify(all_heap)

    def peel_from_core(start_nodes: Iterable[Any]) -> None:
        q = [u for u in start_nodes if u in core and core_deg.get(u, 0) < 2]
        while q:
            x = q.pop()
            if x not in core:
                continue
            core.remove(x)
            for y in nbrs[x]:
                if y in core:
                    core_deg[y] -= 1
                    if core_deg[y] < 2:
                        q.append(y)

    order: List[Any] = []
    target = _budget(graph, rate)
    while alive and len(order) < target:
        if core:
            while core_heap:
                _, _, u, vu = heapq.heappop(core_heap)
                if u in alive and u in core and vu == version[u]:
                    break
            else:
                u = max(core, key=lambda x: (deg.get(x, 0), str(x)))
        else:
            while all_heap:
                _, _, u, vu = heapq.heappop(all_heap)
                if u in alive and vu == version[u]:
                    break
            else:
                u = max(alive, key=lambda x: (deg.get(x, 0), str(x)))

        if u not in alive:
            continue
        order.append(u)
        alive.remove(u)
        was_core = u in core
        if was_core:
            core.remove(u)
        touched = set()
        for v in list(nbrs[u]):
            if v not in alive:
                continue
            nbrs[v].discard(u)
            deg[v] = len(nbrs[v] & alive)
            version[v] += 1
            touched.add(v)
            heapq.heappush(all_heap, (-deg[v], str(v), v, version[v]))
            if v in core:
                if was_core:
                    core_deg[v] -= 1
                heapq.heappush(core_heap, (-deg[v], str(v), v, version[v]))
        nbrs[u].clear()
        deg[u] = 0
        version[u] += 1
        if touched:
            peel_from_core(touched)
    return complete_order(graph, order)
