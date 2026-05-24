# -*- coding: utf-8 -*-
"""Ranking and Pareto selection for HAST candidates."""

from __future__ import annotations

import math
from typing import Any, Iterable

import pandas as pd

from .config import SearchWeights


def _rank_series(values: pd.Series, higher_is_better: bool) -> pd.Series:
    ordered = values.rank(method="average", ascending=not higher_is_better)
    if len(values) <= 1:
        return pd.Series([1.0] * len(values), index=values.index)
    return 1.0 - (ordered - 1.0) / (len(values) - 1.0)


def add_rank_scores(df: pd.DataFrame, weights: SearchWeights, root_auc_cNBI: float | None = None) -> pd.DataFrame:
    out = df.copy()
    valid = out["valid"].astype(bool)
    out["delta_auc_cNBI"] = out["auc_cNBI"] - (root_auc_cNBI if root_auc_cNBI is not None else out["auc_cNBI"].min())
    out["rank_relative_credit"] = 0.0
    out["rank_fragmentation"] = 0.0
    out["rank_time"] = 0.0
    out["rank_absolute_quality"] = 0.0
    out["rank_score"] = -1.0
    if valid.sum() == 0:
        return out
    idx = out.index[valid]
    out.loc[idx, "rank_relative_credit"] = _rank_series(out.loc[idx, "delta_auc_cNBI"], True)
    out.loc[idx, "rank_fragmentation"] = _rank_series(out.loc[idx, "R"], False)
    out.loc[idx, "rank_time"] = _rank_series(out.loc[idx, "time_s"], False)
    out.loc[idx, "rank_absolute_quality"] = _rank_series(out.loc[idx, "auc_cNBI"], True)
    out.loc[idx, "rank_score"] = (
        weights.relative_credit * out.loc[idx, "rank_relative_credit"]
        + weights.fragmentation * out.loc[idx, "rank_fragmentation"]
        + weights.time * out.loc[idx, "rank_time"]
        + weights.absolute_quality * out.loc[idx, "rank_absolute_quality"]
    )
    return out


def pareto_frontier(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [r for r in records if r.get("valid") and math.isfinite(float(r.get("auc_cNBI", float("nan"))))]
    frontier: list[dict[str, Any]] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            better_or_equal = (
                float(other["auc_cNBI"]) >= float(row["auc_cNBI"])
                and float(other["R"]) <= float(row["R"])
                and float(other["time_s"]) <= float(row["time_s"])
            )
            strictly_better = (
                float(other["auc_cNBI"]) > float(row["auc_cNBI"])
                or float(other["R"]) < float(row["R"])
                or float(other["time_s"]) < float(row["time_s"])
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda r: (float(r.get("rank_score", 0.0)), float(r["auc_cNBI"])), reverse=True)


def select_final_q_s(frontier: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    if not frontier:
        return {"HAST-Final-Q": None, "HAST-Final-S": None}
    quality = max(frontier, key=lambda r: (float(r["auc_cNBI"]), -float(r["R"])))
    speed = min(frontier, key=lambda r: (float(r["time_s"]), -float(r["auc_cNBI"])))
    return {"HAST-Final-Q": quality, "HAST-Final-S": speed}
