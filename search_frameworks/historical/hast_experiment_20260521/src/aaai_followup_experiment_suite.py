# -*- coding: utf-8 -*-
"""AAAI follow-up experiment suite for HAST/FAC-T.

This script consolidates the small, low-risk experiments requested after the
AAAI readiness review. It deliberately reuses completed runs whenever possible
and only runs fresh computation for scaling/runtime.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import random
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
SRC = ROOT / "src"
TABLE = ROOT / "tables"
FIG = ROOT / "figures" / "aaai_followup"
REPORT = ROOT / "reports"
FINAL12 = ROOT / "final_12graph_eval"
FINAL12_RECORDS = FINAL12 / "records"
ABLATION = WORKSPACE / "research" / "tree_search_ablation_20260520"
EVAL12_SRC = ABLATION / "src" / "evaluate_final_12graphs.py"
SEARCH_SRC = ABLATION / "src" / "ablation_search.py"

FOLLOWUP_TABLE_PREFIX = "aaai_followup_"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


E12 = load_module(EVAL12_SRC, "aaai_followup_eval12")
SEARCH = load_module(SEARCH_SRC, "aaai_followup_search")


def ensure_dirs() -> None:
    for path in [TABLE, FIG, REPORT]:
        path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 170,
            "savefig.dpi": 260,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )


def auc_mean(x: Iterable[float], y: Iterable[float]) -> float:
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    if len(xa) < 2:
        return float(np.mean(ya)) if len(ya) else float("nan")
    order = np.argsort(xa)
    xa, ya = xa[order], ya[order]
    span = xa[-1] - xa[0]
    if span <= 0:
        return float(np.mean(ya))
    return float(np.trapezoid(ya, xa) / span)


def load_method_dataset_table() -> pd.DataFrame:
    """Build a method x dataset table with generic baselines plus FAST/BT."""
    base = pd.read_csv(TABLE / "hast_early_curve_proxy_rows.csv")
    base = base[["method", "dataset", "R", "auc_cNBI", "time_s"]].copy()

    extra_frames = []
    fast_path = TABLE / "hast_fact_fast_probe_full12_summary.csv"
    if fast_path.exists():
        fast = pd.read_csv(fast_path)
        extra_frames.append(fast[fast["method"].isin(["FAST21-cap24", "FAST7-cap32-approx"])][["method", "dataset", "R", "auc_cNBI", "time_s"]])

    bt_path = TABLE / "hast_bounded_template_probe_full12_detail.csv"
    if bt_path.exists():
        bt = pd.read_csv(bt_path)
        extra_frames.append(bt[bt["method"].isin(["BT-n16-t8-u24", "BT-n16-t8-u18", "BT-n32-t8-u24"])][["method", "dataset", "R", "auc_cNBI", "time_s"]])

    online_path = TABLE / "HAST-FACT-ONLINE60_full12_detail.csv"
    if online_path.exists():
        online = pd.read_csv(online_path)
        online["method"] = "HAST-FAC-T online #" + online["candidate_idx"].astype(str)
        extra_frames.append(online[["method", "dataset", "R", "auc_cNBI", "time_s"]])

    if extra_frames:
        base = pd.concat([base, *extra_frames], ignore_index=True)
    base = base.drop_duplicates(["method", "dataset"], keep="last")
    return base


def summarize_methods(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("method", as_index=False)
        .agg(
            mean_R=("R", "mean"),
            mean_auc_cNBI=("auc_cNBI", "mean"),
            mean_time_s=("time_s", "mean"),
            median_auc_cNBI=("auc_cNBI", "median"),
            datasets=("dataset", "nunique"),
        )
        .sort_values("mean_auc_cNBI", ascending=False)
    )
    out.to_csv(TABLE / f"{FOLLOWUP_TABLE_PREFIX}method_mean_summary.csv", index=False, encoding="utf-8-sig")
    return out


def dataset_rank_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, sub in df.groupby("dataset"):
        ranked = sub.sort_values("auc_cNBI", ascending=False).reset_index(drop=True)
        for pos, row in ranked.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "rank_auc_cNBI": pos + 1,
                    "method": row["method"],
                    "R": row["R"],
                    "auc_cNBI": row["auc_cNBI"],
                    "time_s": row["time_s"],
                    "best_method": ranked.iloc[0]["method"],
                    "gap_to_best_auc": float(ranked.iloc[0]["auc_cNBI"] - row["auc_cNBI"]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE / f"{FOLLOWUP_TABLE_PREFIX}dataset_method_ranks.csv", index=False, encoding="utf-8-sig")
    return out


def heldout_selection(df: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    """Simulate small validation graph selection and held-out performance."""
    data = df[df["method"].isin(methods)].copy()
    datasets = sorted(data["dataset"].unique())
    rows = []
    for k in [1, 2, 3]:
        for vals in combinations(datasets, k):
            val_mask = data["dataset"].isin(vals)
            val_auc = data[val_mask].groupby("method")["auc_cNBI"].mean()
            test_auc = data[~val_mask].groupby("method")["auc_cNBI"].mean()
            test_time = data[~val_mask].groupby("method")["time_s"].mean()
            common = sorted(set(val_auc.index) & set(test_auc.index))
            if not common:
                continue
            val_auc = val_auc.loc[common]
            test_auc = test_auc.loc[common]
            chosen = str(val_auc.sort_values(ascending=False).index[0])
            oracle = str(test_auc.sort_values(ascending=False).index[0])
            rows.append(
                {
                    "k_validation_graphs": k,
                    "validation_graphs": "|".join(vals),
                    "chosen_method": chosen,
                    "oracle_method": oracle,
                    "spearman_val_to_test_auc": float(val_auc.corr(test_auc, method="spearman")),
                    "chosen_test_auc": float(test_auc[chosen]),
                    "chosen_test_time_s": float(test_time[chosen]),
                    "oracle_test_auc": float(test_auc[oracle]),
                    "regret_to_oracle": float(test_auc[oracle] - test_auc[chosen]),
                    "chosen_beats_E26F": bool("E26F" in test_auc.index and test_auc[chosen] > test_auc["E26F"]),
                    "chosen_beats_PUCT": bool("PUCT" in test_auc.index and test_auc[chosen] > test_auc["PUCT"]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE / f"{FOLLOWUP_TABLE_PREFIX}heldout_selection_trials.csv", index=False, encoding="utf-8-sig")
    summary = (
        out.groupby("k_validation_graphs", as_index=False)
        .agg(
            cases=("validation_graphs", "count"),
            mean_spearman=("spearman_val_to_test_auc", "mean"),
            mean_regret=("regret_to_oracle", "mean"),
            beat_E26F_rate=("chosen_beats_E26F", "mean"),
            beat_PUCT_rate=("chosen_beats_PUCT", "mean"),
        )
    )
    summary.to_csv(TABLE / f"{FOLLOWUP_TABLE_PREFIX}heldout_selection_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def fac_ablation_summary(method_mean: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add(stage: str, method: str, note: str = "") -> None:
        hit = method_mean[method_mean["method"].eq(method)]
        if hit.empty:
            return
        r = hit.iloc[0]
        rows.append(
            {
                "stage": stage,
                "method": method,
                "mean_auc_cNBI": float(r["mean_auc_cNBI"]),
                "mean_R": float(r["mean_R"]),
                "mean_time_s": float(r["mean_time_s"]),
                "note": note,
            }
        )

    add("generic best", "FunSearch-like", "strong quality, slow")
    add("generic PUCT", "PUCT", "strong quality, slower than bounded templates")
    add("representative discovered heuristic", "E26F", "fast, interpretable baseline")
    add("HAST initial family credit", "HAST", "family collapse / weak generalization")
    add("FAC-T online", "HAST-FAC-T online #24", "time-aware online search")
    add("manual FAC-T compression", "FAST21-cap24", "capped two-hop compression")
    add("bounded language", "BT-n16-t8-u24", "template-constrained candidate")
    add("baseline", "HDA", "root heuristic")
    out = pd.DataFrame(rows)
    out.to_csv(TABLE / f"{FOLLOWUP_TABLE_PREFIX}fac_ablation_summary.csv", index=False, encoding="utf-8-sig")
    return out


def searcher_budget_curves() -> pd.DataFrame:
    """Compare search traces and a random best-of-N baseline from each pool."""
    runs = {
        "HAST": ROOT / "runs" / "HAST" / "search_records.csv",
        "HAST-FAC": ROOT / "runs" / "HAST-FAC" / "search_records.csv",
        "HAST-FAC-T": ROOT / "runs" / "HAST-FACT-ONLINE60" / "search_records.csv",
        "PUCT": ABLATION / "runs" / "PUCT" / "search_records.csv",
        "MCTS-AHD-like": ABLATION / "runs" / "MCTS-AHD-like" / "search_records.csv",
        "Clade-AHD-like": ABLATION / "runs" / "Clade-AHD-like" / "search_records.csv",
        "FunSearch-like": ABLATION / "runs" / "FunSearch-like" / "search_records.csv",
        "AlphaEvolve-like": ABLATION / "runs" / "AlphaEvolve-like" / "search_records.csv",
    }
    rows = []
    rng = random.Random(20260522)
    budgets = [10, 25, 50, 60, 100, 200, 300]
    for method, path in runs.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[pd.to_numeric(df["idx"], errors="coerce").fillna(0).astype(int) > 0].copy()
        df["valid_bool"] = df["valid"].astype(str).str.lower().isin(["true", "1", "yes"])
        for col in ["idx", "cNBI", "R", "Time", "rank_score", "fac_score", "proxy_time_s", "fac_auc_adv"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        valid = df[df["valid_bool"]].copy()
        if valid.empty:
            continue
        score_col = "fac_score" if "fac_score" in valid.columns and valid["fac_score"].notna().any() else "rank_score"
        valid = valid.sort_values("idx")
        for b in budgets:
            prefix = valid[valid["idx"] <= b]
            if prefix.empty:
                continue
            best = prefix.sort_values(score_col, ascending=False).iloc[0]
            pool_scores = valid[score_col].dropna().tolist()
            random_best = float("nan")
            if pool_scores:
                draws = []
                sample_n = min(b, len(pool_scores))
                for _ in range(200):
                    draws.append(max(rng.sample(pool_scores, sample_n)))
                random_best = float(np.mean(draws))
            rows.append(
                {
                    "method": method,
                    "budget": b,
                    "valid_seen": int(prefix.shape[0]),
                    "valid_rate_up_to_budget": float(df[df["idx"] <= b]["valid_bool"].mean()),
                    "score_col": score_col,
                    "best_score": float(best[score_col]),
                    "best_search_cNBI": float(best["cNBI"]) if "cNBI" in best else float("nan"),
                    "best_search_R": float(best["R"]) if "R" in best else float("nan"),
                    "best_search_Time": float(best["Time"]) if "Time" in best else float("nan"),
                    "random_best_of_N_expected_score": random_best,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE / f"{FOLLOWUP_TABLE_PREFIX}searcher_budget_curves.csv", index=False, encoding="utf-8-sig")
    return out


def cnbi_validity_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_rows = []
    for scope, sub in [("all", df), ("without_HDA_CoreHD", df[~df["method"].isin(["HDA", "CoreHD"])])]:
        corr_rows.append(
            {
                "scope": scope,
                "n": int(sub.shape[0]),
                "spearman_auc_cNBI_vs_R": float(sub["auc_cNBI"].corr(sub["R"], method="spearman")),
                "pearson_auc_cNBI_vs_R": float(sub["auc_cNBI"].corr(sub["R"], method="pearson")),
                "spearman_auc_cNBI_vs_time": float(sub["auc_cNBI"].corr(sub["time_s"], method="spearman")),
            }
        )
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(TABLE / f"{FOLLOWUP_TABLE_PREFIX}cnbi_nonredundancy_correlations.csv", index=False, encoding="utf-8-sig")

    rows = []
    for dataset in ["Collaboration", "Grid", "HepPh", "Yeast", "condmat", "email"]:
        for a, b in [("E26F", "HDA"), ("PUCT", "E26F"), ("FunSearch-like", "E26F")]:
            pa = FINAL12_RECORDS / f"{dataset}_{a}_metrics.csv"
            pb = FINAL12_RECORDS / f"{dataset}_{b}_metrics.csv"
            if not pa.exists() or not pb.exists():
                continue
            da = pd.read_csv(pa)
            db = pd.read_csv(pb)
            if da.empty or db.empty:
                continue
            # Find a non-trivial near-matched GCC point. Very early removals
            # often produce identical snapshots and are not useful evidence.
            da2 = da[(da["removal_ratio"] >= 0.05) & (da["GCC"] <= 0.98)].copy()
            db2 = db[(db["removal_ratio"] >= 0.05) & (db["GCC"] <= 0.98)].copy()
            if da2.empty or db2.empty:
                continue
            best = None
            fallback = None
            for _, ra in da2.iterrows():
                idx = (db2["GCC"] - float(ra["GCC"])).abs().idxmin()
                rb = db2.loc[idx]
                diff = abs(float(ra["GCC"]) - float(rb["GCC"]))
                delta = float(ra["cNBI"] - rb["cNBI"])
                candidate = (diff, abs(delta), delta, ra, rb)
                if fallback is None or diff < fallback[0]:
                    fallback = candidate
                if diff <= 0.005 and (best is None or abs(delta) > best[1]):
                    best = candidate
            if best is None:
                best = fallback
            if best is None:
                continue
            diff, _abs_delta, _delta, ra, rb = best
            rows.append(
                {
                    "dataset": dataset,
                    "method_a": a,
                    "method_b": b,
                    "gcc_abs_diff": float(diff),
                    "gcc_a": float(ra["GCC"]),
                    "gcc_b": float(rb["GCC"]),
                    "cNBI_a": float(ra["cNBI"]),
                    "cNBI_b": float(rb["cNBI"]),
                    "delta_cNBI_a_minus_b": float(ra["cNBI"] - rb["cNBI"]),
                    "top5_mass_a": float(ra["top5_component_mass"]),
                    "top5_mass_b": float(rb["top5_component_mass"]),
                    "pairdisc_a": float(ra["pairwise_disconnected"]),
                    "pairdisc_b": float(rb["pairwise_disconnected"]),
                    "ratio_a": float(ra["removal_ratio"]),
                    "ratio_b": float(rb["removal_ratio"]),
                }
            )
    same = pd.DataFrame(rows).sort_values("delta_cNBI_a_minus_b", ascending=False)
    same.to_csv(TABLE / f"{FOLLOWUP_TABLE_PREFIX}same_gcc_cnbi_cases.csv", index=False, encoding="utf-8-sig")
    return corr, same


def load_candidate_fn_from_file(path: Path) -> Callable[[nx.Graph], List[Any]]:
    code = path.read_text(encoding="utf-8")
    return SEARCH.compile_degree_order(code)


def get_scaling_methods() -> Dict[str, Callable[[nx.Graph], List[Any]]]:
    methods: Dict[str, Callable[[nx.Graph], List[Any]]] = {}

    def hda(g: nx.Graph) -> List[Any]:
        return E12.EVAL.hda_simple_order(g, 0.30)

    def corehd(g: nx.Graph) -> List[Any]:
        return E12.EVAL.corehd_order(g, 0.30)

    def e26f(g: nx.Graph) -> List[Any]:
        return E12.EVAL.DACTS.degree_order_by_config(g, E12.EVAL.DACTS.e26f_config(), budget_ratio=0.30)

    methods["HDA"] = hda
    methods["CoreHD"] = corehd
    methods["E26F"] = e26f
    fast_file = next((ROOT / "runs" / "HAST-FACT-FAST" / "candidates").glob("candidate_*FAST21*"), None)
    if fast_file is None:
        # The generated filenames are hash-based; fall back to the summary path.
        fast_summary = pd.read_csv(TABLE / "hast_fact_fast_probe_summary.csv")
        hit = fast_summary[fast_summary["method"].eq("FAST21-cap24")]
        if not hit.empty:
            fast_file = Path(str(hit.iloc[0]["candidate_file"]))
    if fast_file and fast_file.exists():
        methods["FAST21-cap24"] = load_candidate_fn_from_file(fast_file)

    bt_summary = pd.read_csv(TABLE / "hast_bounded_template_probe_summary.csv")
    hit = bt_summary[bt_summary["method"].eq("BT-n16-t8-u24")]
    if not hit.empty:
        bt_file = Path(str(hit.iloc[0]["candidate_file"]))
        if bt_file.exists():
            methods["BT-n16-t8-u24"] = load_candidate_fn_from_file(bt_file)

    puct_file = ABLATION / "runs" / "PUCT" / "best_candidate.py"
    if puct_file.exists():
        methods["PUCT"] = load_candidate_fn_from_file(puct_file)
    return methods


def make_scaling_graph(kind: str, n: int, seed: int) -> nx.Graph:
    if kind == "powerlaw":
        return E12.EVAL.generate_powerlaw_network(n, 2.5, seed=seed)
    if kind == "er":
        p = min(0.05, 6.0 / max(2, n - 1))
        g = nx.fast_gnp_random_graph(n, p, seed=seed)
    elif kind == "ws":
        k = min(8, max(2, n // 50 * 2))
        if k % 2 == 1:
            k += 1
        g = nx.watts_strogatz_graph(n, k, 0.08, seed=seed)
    elif kind == "sbm":
        blocks = 4
        sizes = [n // blocks] * blocks
        sizes[-1] += n - sum(sizes)
        p_in = min(0.08, 10.0 / max(2, n // blocks))
        p_out = p_in * 0.05
        probs = [[p_in if i == j else p_out for j in range(blocks)] for i in range(blocks)]
        g = nx.stochastic_block_model(sizes, probs, seed=seed)
    else:
        raise ValueError(kind)
    g = nx.Graph(g)
    g.remove_edges_from(nx.selfloop_edges(g))
    if g.number_of_nodes() and not nx.is_connected(g):
        comps = [list(c) for c in nx.connected_components(g)]
        for a, b in zip(comps[:-1], comps[1:]):
            g.add_edge(a[0], b[0])
    return nx.convert_node_labels_to_integers(g)


def run_scaling(sizes: List[int], seeds: List[int], kinds: List[str], include_puct_10k: bool = False) -> pd.DataFrame:
    methods = get_scaling_methods()
    rows = []
    for n in sizes:
        for kind in kinds:
            for seed in seeds:
                graph = make_scaling_graph(kind, n, seed)
                for method, fn in methods.items():
                    if n >= 10000 and method == "PUCT" and not include_puct_10k:
                        continue
                    t0 = time.perf_counter()
                    try:
                        order = list(fn(graph.copy()))
                        elapsed = time.perf_counter() - t0
                        metrics = E12.EVAL.compute_metrics(graph, order, rate=0.30, method_time=elapsed)
                        x = metrics["removal_ratio"].to_numpy(dtype=float)
                        rows.append(
                            {
                                "kind": kind,
                                "n": n,
                                "seed": seed,
                                "edges": graph.number_of_edges(),
                                "method": method,
                                "ok": True,
                                "time_s": elapsed,
                                "R": float(metrics["GCC"].mean()),
                                "auc_cNBI": auc_mean(x, metrics["cNBI"].to_numpy(dtype=float)),
                                "error": "",
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        rows.append(
                            {
                                "kind": kind,
                                "n": n,
                                "seed": seed,
                                "edges": graph.number_of_edges(),
                                "method": method,
                                "ok": False,
                                "time_s": time.perf_counter() - t0,
                                "R": float("nan"),
                                "auc_cNBI": float("nan"),
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    print(f"[scaling] n={n} kind={kind} seed={seed} method={method}", flush=True)
    out = pd.DataFrame(rows)
    suffix = "10k" if any(n >= 10000 for n in sizes) else "upto5k"
    out.to_csv(TABLE / f"{FOLLOWUP_TABLE_PREFIX}scaling_runtime_{suffix}.csv", index=False, encoding="utf-8-sig")
    return out




def plot_core_outputs(method_mean: pd.DataFrame, ranks: pd.DataFrame, heldout: pd.DataFrame, fac: pd.DataFrame, search: pd.DataFrame, scaling: pd.DataFrame | None) -> None:
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    top_methods = method_mean.head(12).copy()
    axes[0, 0].barh(top_methods["method"][::-1], top_methods["mean_auc_cNBI"][::-1], color="#4C78A8")
    axes[0, 0].set_xlabel("mean auc_cNBI")
    axes[0, 0].set_title("12-graph mean quality")

    focus = ranks[ranks["method"].isin(["BT-n16-t8-u24", "FAST21-cap24", "E26F", "PUCT", "FunSearch-like", "Clade-AHD-like"])]
    pivot = focus.pivot_table(index="dataset", columns="method", values="rank_auc_cNBI", aggfunc="mean")
    im = axes[0, 1].imshow(pivot.fillna(np.nan).to_numpy(dtype=float), aspect="auto", cmap="viridis_r")
    axes[0, 1].set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    axes[0, 1].set_yticks(range(len(pivot.index)), pivot.index)
    axes[0, 1].set_title("Dataset-level ranks")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, label="rank")

    axes[1, 0].bar(heldout["k_validation_graphs"].astype(str), heldout["mean_regret"], color="#F28E2B")
    axes[1, 0].set_xlabel("validation graphs")
    axes[1, 0].set_ylabel("mean regret to oracle")
    axes[1, 0].set_title("Held-out selector regret")

    axes[1, 1].scatter(fac["mean_time_s"], fac["mean_auc_cNBI"], color="#D62728", s=48)
    for _, row in fac.iterrows():
        axes[1, 1].annotate(row["stage"], (row["mean_time_s"], row["mean_auc_cNBI"]), fontsize=7)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("mean time_s (log)")
    axes[1, 1].set_ylabel("mean auc_cNBI")
    axes[1, 1].set_title("FAC/FAC-T/Template path")

    fig.tight_layout()
    fig.savefig(FIG / "aaai_followup_core_panel.png")
    plt.close(fig)

    if not search.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for method, sub in search.groupby("method"):
            ax.plot(sub["budget"], sub["best_score"], marker="o", label=method)
        ax.set_xlabel("candidate budget")
        ax.set_ylabel("best score in trace")
        ax.set_title("Search trace budget comparison")
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(FIG / "aaai_followup_search_budget_curves.png")
        plt.close(fig)

    if not fac.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
        order = fac.copy()
        colors = [
            "#9AA0A6",
            "#9AA0A6",
            "#4C78A8",
            "#C44E52",
            "#F28E2B",
            "#59A14F",
            "#2A9D8F",
            "#BDBDBD",
        ][: len(order)]
        x = np.arange(len(order))
        axes[0].bar(x, order["mean_auc_cNBI"], color=colors)
        axes[0].set_xticks(x, order["stage"], rotation=35, ha="right")
        axes[0].set_ylabel("mean auc_cNBI")
        axes[0].set_title("Ablation Path: Quality")
        for i, row in enumerate(order.itertuples(index=False)):
            axes[0].text(i, float(row.mean_auc_cNBI), f"{float(row.mean_auc_cNBI):.1f}", ha="center", va="bottom", fontsize=7)

        axes[1].bar(x, order["mean_time_s"], color=colors)
        axes[1].set_yscale("log")
        axes[1].set_xticks(x, order["stage"], rotation=35, ha="right")
        axes[1].set_ylabel("mean time_s (log)")
        axes[1].set_title("Ablation Path: Runtime")
        for i, row in enumerate(order.itertuples(index=False)):
            axes[1].text(i, float(row.mean_time_s), f"{float(row.mean_time_s):.2f}s", ha="center", va="bottom", fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "aaai_followup_fac_ablation_path.png")
        plt.close(fig)

    if scaling is not None and not scaling.empty:
        ok = scaling[scaling["ok"].astype(bool)].copy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for method, sub in ok.groupby("method"):
            agg = sub.groupby("n")["time_s"].mean().reset_index()
            axes[0].plot(agg["n"], agg["time_s"], marker="o", label=method)
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("nodes")
        axes[0].set_ylabel("runtime_s")
        axes[0].set_title("Scaling runtime")
        axes[0].legend(frameon=False, fontsize=6)

        for method, sub in ok.groupby("method"):
            agg = sub.groupby("n")["auc_cNBI"].mean().reset_index()
            axes[1].plot(agg["n"], agg["auc_cNBI"], marker="o", label=method)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("nodes")
        axes[1].set_ylabel("auc_cNBI")
        axes[1].set_title("Scaling quality")
        fig.tight_layout()
        fig.savefig(FIG / "aaai_followup_scaling_runtime.png")
        plt.close(fig)


def write_report(method_mean: pd.DataFrame, heldout: pd.DataFrame, fac: pd.DataFrame, ranks: pd.DataFrame, cnbi_corr: pd.DataFrame, same_gcc: pd.DataFrame, scaling: pd.DataFrame | None) -> None:
    bt = method_mean[method_mean["method"].eq("BT-n16-t8-u24")]
    fast = method_mean[method_mean["method"].eq("FAST21-cap24")]
    e26f = method_mean[method_mean["method"].eq("E26F")]
    puct = method_mean[method_mean["method"].eq("PUCT")]

    def fmt(row: pd.DataFrame, col: str) -> str:
        return "NA" if row.empty else f"{float(row.iloc[0][col]):.3f}"

    rank_focus = ranks[ranks["method"].isin(["BT-n16-t8-u24", "FAST21-cap24", "E26F", "PUCT"])]
    rank_summary = (
        rank_focus.groupby("method", as_index=False)
        .agg(top1=("rank_auc_cNBI", lambda s: int((s == 1).sum())), top3=("rank_auc_cNBI", lambda s: int((s <= 3).sum())), mean_rank=("rank_auc_cNBI", "mean"))
        .sort_values("mean_rank")
    )

    scaling_note = "未运行 scaling/runtime。"
    if scaling is not None and not scaling.empty:
        ok = scaling[scaling["ok"].astype(bool)]
        scaling_note = ok.groupby(["method", "n"])["time_s"].mean().reset_index().to_markdown(index=False)

    lines = [
        "# AAAI Follow-up Experiments for HAST/FAC-T",
        "",
        "## 结论先行",
        "",
        "这批实验支持一个更稳的 AAAI 叙事：HAST 的关键贡献不是自由 LLM mutation 本身，而是把搜索信用和候选语言约束到低复杂度、可解释的 bounded local fracture proxy。旧 FAC 能找到强碎裂候选，但会漂向慢二跳扫描；FAC-T 和 bounded template 能把质量-时间前沿拉回可用区域。",
        "",
        "## 12 图核心结果",
        "",
        f"- `BT-n16-t8-u24`: mean auc_cNBI={fmt(bt, 'mean_auc_cNBI')}，mean time={fmt(bt, 'mean_time_s')}s。",
        f"- `FAST21-cap24`: mean auc_cNBI={fmt(fast, 'mean_auc_cNBI')}，mean time={fmt(fast, 'mean_time_s')}s。",
        f"- `E26F`: mean auc_cNBI={fmt(e26f, 'mean_auc_cNBI')}，mean time={fmt(e26f, 'mean_time_s')}s。",
        f"- `PUCT`: mean auc_cNBI={fmt(puct, 'mean_auc_cNBI')}，mean time={fmt(puct, 'mean_time_s')}s。",
        "",
        "Dataset-level ranking 仍然提醒我们不能写 universal SOTA：",
        "",
        rank_summary.to_markdown(index=False),
        "",
        "## Held-out 小验证图选择",
        "",
        heldout.to_markdown(index=False),
        "",
        "解释：少量 validation graph 可用于模型/模板选择，但当前不是严格训练-测试协议。正式论文里要把 search/proxy/validation/test 固定下来，避免 test leakage。",
        "",
        "## FAC/FAC-T/Template 消融链",
        "",
        fac.to_markdown(index=False),
        "",
        "解释：这张表应作为论文主线的骨架。旧自由搜索给出上限，FAC-T 防止慢候选，bounded template 则证明候选语言约束比继续复杂化树策略更有效。",
        "",
        "## cNBI 非冗余与 same-GCC 证据",
        "",
        cnbi_corr.to_markdown(index=False),
        "",
        "Same-GCC 例子：",
        "",
        same_gcc.head(12).to_markdown(index=False),
        "",
        "解释：cNBI 与 R/GCC 相关但不等价；same-GCC 表能支撑“相近最大连通分量下，残余组件分布仍不同”的指标动机。",
        "",
        "## Scaling/runtime",
        "",
        scaling_note,
        "",
        "## 论文写法建议",
        "",
        "1. 不要主张 HAST/BT 在所有图上超过 FunSearch/Clade/PUCT；改成主张 quality-runtime frontier 更好。",
        "2. 主表同时报告 mean、dataset rank、time，避免均值被少数大图主导。",
        "3. 将旧 FAC 慢候选作为 failure-to-fix 证据，而不是藏起来。",
        "4. 最终提交前还需要一次冻结协议的 rerun：固定 proxy/validation/test split，模板参数只在 validation 上选。",
        "",
        "## 输出位置",
        "",
        f"- 表格目录：`{TABLE.as_posix()}`，文件前缀 `{FOLLOWUP_TABLE_PREFIX}`。",
        f"- 图目录：`{FIG.as_posix()}`。",
        f"- 消融实验图：`{(FIG / 'aaai_followup_fac_ablation_path.png').as_posix()}`。",
    ]
    (REPORT / "aaai_followup_experiments_20260522_cn.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-scaling", action="store_true")
    p.add_argument("--sizes", default="500,1000,5000")
    p.add_argument("--kinds", default="powerlaw,er,ws,sbm")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--include-puct-10k", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    setup_style()

    df = load_method_dataset_table()
    method_mean = summarize_methods(df)
    ranks = dataset_rank_table(df)
    selection_methods = [
        "BT-n16-t8-u24",
        "FAST21-cap24",
        "E26F",
        "PUCT",
        "FunSearch-like",
        "Clade-AHD-like",
        "HAST-FAC-T online #24",
        "HDA",
    ]
    heldout = heldout_selection(df, [m for m in selection_methods if m in set(df["method"])])
    fac = fac_ablation_summary(method_mean)
    search = searcher_budget_curves()
    cnbi_corr, same_gcc = cnbi_validity_tables(df)

    scaling = None
    if args.run_scaling:
        sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
        kinds = [x.strip() for x in args.kinds.split(",") if x.strip()]
        scaling = run_scaling(sizes=sizes, seeds=seeds, kinds=kinds, include_puct_10k=args.include_puct_10k)
    else:
        candidates = sorted(TABLE.glob(f"{FOLLOWUP_TABLE_PREFIX}scaling_runtime_*.csv"))
        if candidates:
            scaling = pd.concat([pd.read_csv(p) for p in candidates], ignore_index=True).drop_duplicates(
                ["kind", "n", "seed", "method"], keep="last"
            )

    plot_core_outputs(method_mean, ranks, heldout, fac, search, scaling)
    write_report(method_mean, heldout, fac, ranks, cnbi_corr, same_gcc, scaling)
    print(REPORT / "aaai_followup_experiments_20260522_cn.md")


if __name__ == "__main__":
    main()
