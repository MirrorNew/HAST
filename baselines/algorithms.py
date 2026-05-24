# -*- coding: utf-8 -*-
"""Self-contained baseline algorithms for network dismantling experiments.

All functions return a full node order. The evaluated prefix is controlled by
the caller through the removal rate, which keeps baseline code independent from
metric code.
"""

from __future__ import annotations

import heapq
from typing import Any, Iterable, List

import networkx as nx


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
