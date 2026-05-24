# -*- coding: utf-8 -*-
"""Graph loading and synthetic graph generation."""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import numpy as np

from .config import BENCHMARK_ROOT


def generate_powerlaw_network(n: int, gamma: float = 2.5, seed: int = 42, k_min: int = 2) -> nx.Graph:
    rng = np.random.default_rng(seed)
    k_max = max(k_min + 1, int(math.sqrt(n) * 4))
    degree_values = np.arange(k_min, k_max + 1)
    weights = np.array([k ** (-gamma) for k in degree_values], dtype=float)
    weights /= weights.sum()
    degrees = rng.choice(degree_values, size=n, replace=True, p=weights).astype(int).tolist()
    if sum(degrees) % 2 == 1:
        degrees[0] += 1
    graph = nx.configuration_model(degrees, seed=seed)
    graph = nx.Graph(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph = nx.convert_node_labels_to_integers(graph)
    if graph.number_of_nodes() > 0 and not nx.is_connected(graph):
        comps = [list(c) for c in nx.connected_components(graph)]
        for a, b in zip(comps[:-1], comps[1:]):
            graph.add_edge(a[0], b[0])
    return graph


def read_graph(dataset: str, benchmark_root: Path = BENCHMARK_ROOT) -> nx.Graph:
    if dataset == "Powerlaw_500":
        return generate_powerlaw_network(500, 2.5, seed=42)
    candidates = [
        benchmark_root / f"{dataset}.txt",
        benchmark_root / "network" / f"{dataset}.edgelist",
        benchmark_root / f"{dataset}.edgelist",
    ]
    for path in candidates:
        if path.exists():
            graph = nx.read_edgelist(path, nodetype=int)
            graph = nx.Graph(graph)
            graph.remove_edges_from(nx.selfloop_edges(graph))
            return graph
    raise FileNotFoundError(f"No graph file found for {dataset} under {benchmark_root}")
