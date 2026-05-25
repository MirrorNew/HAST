# -*- coding: utf-8 -*-
"""Fragmentation trajectory metrics for dismantling orders."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

from baselines.algorithms import complete_order


class DSUWithSizes:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n
        self.active = [False] * n
        self.size_counts: Counter[int] = Counter()
        self.active_count = 0
        self.components = 0
        self.sum_sq = 0.0
        self.sum_pair = 0.0

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def _add_size(self, s: int) -> None:
        self.size_counts[s] += 1
        self.sum_sq += s * s
        self.sum_pair += s * (s - 1)

    def _remove_size(self, s: int) -> None:
        self.size_counts[s] -= 1
        if self.size_counts[s] <= 0:
            del self.size_counts[s]
        self.sum_sq -= s * s
        self.sum_pair -= s * (s - 1)

    def activate(self, i: int) -> None:
        if self.active[i]:
            return
        self.active[i] = True
        self.parent[i] = i
        self.size[i] = 1
        self.active_count += 1
        self.components += 1
        self._add_size(1)

    def union(self, a: int, b: int) -> None:
        if not self.active[a] or not self.active[b]:
            return
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        sa, sb = self.size[ra], self.size[rb]
        self._remove_size(sa)
        self._remove_size(sb)
        self.parent[rb] = ra
        self.size[ra] = sa + sb
        self.components -= 1
        self._add_size(sa + sb)

    def snapshot(self, original_n: int) -> dict[str, float]:
        sizes: list[int] = []
        for size, count in self.size_counts.items():
            sizes.extend([size] * count)
        sizes.sort(reverse=True)
        largest = sizes[0] if sizes else 0
        top5 = sum(sizes[:5]) if sizes else 0
        rem = max(1, self.active_count)
        denom_pairs = max(1, original_n * (original_n - 1))
        hhi_remaining = self.sum_sq / (rem * rem)
        effective_components = 1.0 / hhi_remaining if hhi_remaining > 0 else 0.0
        pairwise_disconnected = 1.0 - self.sum_pair / denom_pairs
        top5_mass = top5 / original_n
        cnbi = pairwise_disconnected * effective_components / (1.0 + top5_mass)
        return {
            "ACC": largest / original_n,
            "GCC": largest / original_n,
            "NCC": float(self.components),
            "top5_component_mass": top5_mass,
            "hhi_remaining": hhi_remaining,
            "effective_components": effective_components,
            "pairwise_disconnected": pairwise_disconnected,
            "cNBI": cnbi,
        }


def compute_metrics(graph: nx.Graph, order: Iterable[Any], rate: float, method_time: float = 0.0) -> pd.DataFrame:
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return pd.DataFrame()
    full_order = complete_order(graph, order)
    steps = max(1, min(len(full_order), int(round(n * rate))))
    prefix = full_order[:steps]
    removed = set(prefix)
    idx = {u: i for i, u in enumerate(nodes)}
    dsu = DSUWithSizes(n)

    for u in nodes:
        if u not in removed:
            dsu.activate(idx[u])
    for u, v in graph.edges():
        iu, iv = idx[u], idx[v]
        if dsu.active[iu] and dsu.active[iv]:
            dsu.union(iu, iv)

    rows: list[dict[str, float]] = []
    for k in range(steps, 0, -1):
        snap = dsu.snapshot(n)
        snap["step"] = k
        snap["removal_ratio"] = k / n
        rows.append(snap)
        u = prefix[k - 1]
        iu = idx[u]
        dsu.activate(iu)
        for v in graph.neighbors(u):
            dsu.union(iu, idx[v])
    rows.reverse()
    df = pd.DataFrame(rows)
    df["running_R"] = df["GCC"].expanding().mean()
    df["running_auc_cNBI"] = df["cNBI"].expanding().mean()
    df["total_time_s"] = method_time
    return df


def auc_mean(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float(y[0]) if len(y) else float("nan")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    span = x[-1] - x[0]
    if span <= 0:
        return float(np.mean(y))
    return float(np.trapezoid(y, x) / span)


def summarize_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "R": float("nan"),
            "auc_ACC": float("nan"),
            "auc_NCC": float("nan"),
            "auc_cNBI": float("nan"),
            "final_ACC": float("nan"),
            "final_NCC": float("nan"),
            "final_cNBI": float("nan"),
            "early_GCC": float("nan"),
            "early_NCC": float("nan"),
            "early_cNBI": float("nan"),
            "time_s": float("nan"),
        }
    x = df["removal_ratio"].to_numpy(dtype=float)
    early_idx = int(np.abs(df["removal_ratio"].to_numpy(dtype=float) - 0.20).argmin())
    early = df.iloc[early_idx]
    return {
        "R": float(df["GCC"].mean()),
        "auc_ACC": auc_mean(x, df["ACC"].to_numpy(dtype=float)),
        "auc_NCC": auc_mean(x, df["NCC"].to_numpy(dtype=float)),
        "auc_cNBI": auc_mean(x, df["cNBI"].to_numpy(dtype=float)),
        "final_ACC": float(df["ACC"].iloc[-1]),
        "final_NCC": float(df["NCC"].iloc[-1]),
        "final_cNBI": float(df["cNBI"].iloc[-1]),
        "early_GCC": float(early["GCC"]),
        "early_NCC": float(early["NCC"]),
        "early_cNBI": float(early["cNBI"]),
        "time_s": float(df["total_time_s"].iloc[-1]),
    }
