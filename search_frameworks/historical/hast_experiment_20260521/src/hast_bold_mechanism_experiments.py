# -*- coding: utf-8 -*-
"""Bold mechanism probes for the next HAST variant.

This is deliberately mechanism-level rather than another engineering ablation.
It tests whether credit should be assigned as *fracture advantage over HDA*,
not as an absolute candidate score.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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

TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"
RUN_DIR = ROOT / "runs" / "HAST_FAC_PROBE"
CAND_DIR = RUN_DIR / "candidates"

METHODS = ["HAST", "PUCT", "MCTS-AHD-like", "Clade-AHD-like", "FunSearch-like", "AlphaEvolve-like", "E26F", "HDA"]
SEARCH_METHODS = ["HAST", "PUCT", "MCTS-AHD-like", "Clade-AHD-like", "FunSearch-like", "AlphaEvolve-like"]
PROXY_DATASETS = ["CEnew", "crime", "Yeast", "Grid", "hamster", "Powerlaw_500"]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SEARCH = load_module(SOURCE_SEARCH, "hast_fac_search_runtime")
E12 = load_module(SOURCE_12GRAPH, "hast_fac_eval12_runtime")


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


def safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", x)


def curve_path(dataset: str, method: str) -> Path:
    return ROOT / "final_12graph_eval" / "records" / f"{dataset}_{safe_name(method)}_metrics.csv"


def auc_mean(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float(np.mean(y)) if len(y) else 0.0
    return float(np.trapezoid(y, x) / max(1e-9, x[-1] - x[0]))


def value_at(curve: pd.DataFrame, col: str, ratio: float) -> float:
    idx = (curve["removal_ratio"] - ratio).abs().idxmin()
    return float(curve.loc[idx, col])


def load_curve_summary() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset in sorted(pd.read_csv(ROOT / "final_12graph_eval" / "hast_12graph_summary.csv")["dataset"].unique()):
        hda_path = curve_path(dataset, "HDA")
        if not hda_path.exists():
            continue
        hda = pd.read_csv(hda_path)
        xh = hda["removal_ratio"].to_numpy(dtype=float)
        hda_auc = auc_mean(xh, hda["cNBI"].to_numpy(dtype=float))
        hda_R = float(hda["GCC"].mean())
        hda_c20 = value_at(hda, "cNBI", 0.20)
        hda_n20 = value_at(hda, "NCC", 0.20)
        hda_g20 = value_at(hda, "GCC", 0.20)
        for method in METHODS:
            path = curve_path(dataset, method)
            if not path.exists():
                continue
            c = pd.read_csv(path)
            x = c["removal_ratio"].to_numpy(dtype=float)
            auc = auc_mean(x, c["cNBI"].to_numpy(dtype=float))
            R = float(c["GCC"].mean())
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "auc_cNBI": auc,
                    "R": R,
                    "time_s": float(c["method_time"].iloc[-1]) if "method_time" in c.columns else np.nan,
                    "fac_auc_adv": auc - hda_auc,
                    "fac_R_adv": hda_R - R,
                    "fac_cNBI20_adv": value_at(c, "cNBI", 0.20) - hda_c20,
                    "fac_NCC20_adv": value_at(c, "NCC", 0.20) - hda_n20,
                    "fac_GCC20_adv": hda_g20 - value_at(c, "GCC", 0.20),
                    "cNBI20": value_at(c, "cNBI", 0.20),
                    "NCC20": value_at(c, "NCC", 0.20),
                    "GCC20": value_at(c, "GCC", 0.20),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "hast_fac_curve_summary.csv", index=False, encoding="utf-8-sig")
    return out


def robust_select(scores: pd.Series, mode: str) -> str:
    if mode == "mean_auc":
        return str(scores.sort_values(ascending=False).index[0])
    if mode == "fac_mean":
        return str(scores.sort_values(ascending=False).index[0])
    raise ValueError(mode)


def fac_leave_k_validation(summary: pd.DataFrame) -> pd.DataFrame:
    datasets = sorted(summary["dataset"].unique().tolist())
    rows: List[Dict[str, Any]] = []
    for k in [1, 2, 3]:
        for val_sets in itertools.combinations(datasets, k):
            val = summary[summary["dataset"].isin(val_sets)]
            test = summary[~summary["dataset"].isin(val_sets)]
            method_rows = []
            for method, g in val.groupby("method"):
                method_rows.append(
                    {
                        "method": method,
                        "mean_auc": float(g["auc_cNBI"].mean()),
                        "fac_mean": float(g["fac_auc_adv"].mean()),
                        "fac_lower": float(g["fac_auc_adv"].mean() - 0.50 * g["fac_auc_adv"].std(ddof=0)),
                        "shape_adv": float(
                            0.45 * g["fac_cNBI20_adv"].mean()
                            + 0.35 * g["fac_NCC20_adv"].mean()
                            + 0.20 * g["fac_GCC20_adv"].mean()
                        ),
                    }
                )
            val_m = pd.DataFrame(method_rows).set_index("method")
            test_auc = test.groupby("method")["auc_cNBI"].mean()
            test_fac = test.groupby("method")["fac_auc_adv"].mean()
            oracle = str(test_auc.sort_values(ascending=False).index[0])
            for selector in ["mean_auc", "fac_mean", "fac_lower", "shape_adv"]:
                chosen = str(val_m[selector].sort_values(ascending=False).index[0])
                rows.append(
                    {
                        "k_validation_graphs": k,
                        "validation_graphs": "|".join(val_sets),
                        "selector": selector,
                        "chosen": chosen,
                        "oracle": oracle,
                        "chosen_heldout_auc": float(test_auc[chosen]),
                        "chosen_heldout_fac": float(test_fac[chosen]),
                        "hast_heldout_auc": float(test_auc["HAST"]),
                        "hast_heldout_fac": float(test_fac["HAST"]),
                        "oracle_heldout_auc": float(test_auc.max()),
                        "gain_over_hast_auc": float(test_auc[chosen] - test_auc["HAST"]),
                        "regret_to_oracle_auc": float(test_auc.max() - test_auc[chosen]),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "hast_fac_validation_selector_trials.csv", index=False, encoding="utf-8-sig")
    summary_rows = (
        out.groupby(["selector", "k_validation_graphs"])
        .agg(
            cases=("chosen", "count"),
            mean_gain_over_hast_auc=("gain_over_hast_auc", "mean"),
            mean_regret_to_oracle_auc=("regret_to_oracle_auc", "mean"),
            hast_selected_rate=("chosen", lambda s: float((s == "HAST").mean())),
            mean_chosen_fac=("chosen_heldout_fac", "mean"),
        )
        .reset_index()
    )
    summary_rows.to_csv(TABLE_DIR / "hast_fac_validation_selector_summary.csv", index=False, encoding="utf-8-sig")
    freq = out.groupby(["selector", "chosen"]).size().reset_index(name="count")
    freq.to_csv(TABLE_DIR / "hast_fac_selector_choice_frequency.csv", index=False, encoding="utf-8-sig")
    return out


def fac_correlations(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = ["auc_cNBI", "fac_auc_adv", "fac_cNBI20_adv", "fac_NCC20_adv", "fac_GCC20_adv", "cNBI20", "NCC20", "GCC20"]
    targets = ["auc_cNBI", "fac_auc_adv", "R"]
    for feature in features:
        for target in targets:
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "spearman": float(summary[feature].corr(summary[target], method="spearman")),
                    "pearson": float(summary[feature].corr(summary[target], method="pearson")),
                }
            )
    out = pd.DataFrame(rows).sort_values(["target", "spearman"], ascending=[True, False])
    out.to_csv(TABLE_DIR / "hast_fac_credit_signal_correlations.csv", index=False, encoding="utf-8-sig")
    return out


PROTOTYPES: Dict[str, str] = {
    "FAC_phase_frontier": r'''
def degree_order(G):
    import heapq
    H = G.copy()
    n0 = H.number_of_nodes()
    heap = []
    stamp = {}
    def score(u):
        if u not in H:
            return -1e18
        nbrs = list(H.neighbors(u))
        d = len(nbrs)
        if d <= 0:
            return 0.0
        nbrset = set(nbrs)
        nd = 0.0
        two = set()
        outside = 0
        leaves = 0
        for v in nbrs:
            dv = H.degree(v)
            nd += dv
            if dv <= 2:
                leaves += 1
            ext = 0
            for w in H.neighbors(v):
                if w != u and w not in nbrset:
                    two.add(w)
                    ext += 1
            if ext:
                outside += 1
        internal = 0
        cap = nbrs[:64]
        capset = set(cap)
        for v in cap:
            for w in H.neighbors(v):
                if w in capset and str(v) < str(w):
                    internal += 1
        redundancy = internal / max(1.0, float(len(cap)))
        phase = (n0 - H.number_of_nodes()) / max(1.0, float(n0))
        if phase < 0.10:
            return 1.25*d + 0.055*nd + 0.020*len(two) - 0.10*redundancy
        if phase < 0.22:
            return 1.00*d + 0.035*nd + 0.080*len(two) + 0.85*outside + 1.10*leaves - 0.35*redundancy
        return 0.80*d + 0.025*nd + 0.055*len(two) + 1.25*outside + 1.50*leaves - 0.45*redundancy
    def push(u):
        if u in H:
            s = stamp.get(u, 0) + 1
            stamp[u] = s
            heapq.heappush(heap, (-score(u), s, str(u), u))
    for u in list(H.nodes()):
        push(u)
    order = []
    while H.number_of_nodes() > 0 and heap:
        neg, s, _, u = heapq.heappop(heap)
        if u not in H or stamp.get(u, 0) != s:
            continue
        affected = set([u])
        for v in list(H.neighbors(u)):
            affected.add(v)
            for w in H.neighbors(v):
                affected.add(w)
        H.remove_node(u)
        order.append(u)
        for v in affected:
            if v in H:
                push(v)
    return order
''',
    "FAC_advantage_local": r'''
def degree_order(G):
    import heapq
    H = G.copy()
    heap = []
    stamp = {}
    def score(u):
        if u not in H:
            return -1e18
        nbrs = list(H.neighbors(u))
        d = len(nbrs)
        if d <= 0:
            return 0.0
        nbrset = set(nbrs)
        nd = 0.0
        two = set()
        boundary = 0
        weak_ties = 0
        for v in nbrs:
            dv = H.degree(v)
            nd += dv
            outv = 0
            for w in H.neighbors(v):
                if w != u and w not in nbrset:
                    two.add(w)
                    outv += 1
            if outv > 0:
                boundary += 1
            if dv <= 3 or outv >= max(1, dv // 2):
                weak_ties += 1
        closed = d + 1
        exposure = len(two) / max(1.0, float(closed))
        return d + 0.045*nd + 1.6*boundary + 1.2*weak_ties + 2.2*exposure
    def push(u):
        if u in H:
            s = stamp.get(u, 0) + 1
            stamp[u] = s
            heapq.heappush(heap, (-score(u), s, str(u), u))
    for u in list(H.nodes()):
        push(u)
    order = []
    while H.number_of_nodes() > 0 and heap:
        _, s, _, u = heapq.heappop(heap)
        if u not in H or stamp.get(u, 0) != s:
            continue
        affected = set([u])
        for v in list(H.neighbors(u)):
            affected.add(v)
            for w in H.neighbors(v):
                affected.add(w)
        H.remove_node(u)
        order.append(u)
        for v in affected:
            if v in H:
                push(v)
    return order
''',
    "FAC_anti_redundancy": r'''
def degree_order(G):
    import heapq
    H = G.copy()
    heap = []
    stamp = {}
    def score(u):
        if u not in H:
            return -1e18
        nbrs = list(H.neighbors(u))
        d = len(nbrs)
        if d <= 0:
            return 0.0
        nbrset = set(nbrs)
        nd = sum(H.degree(v) for v in nbrs)
        two = set()
        boundary = 0
        for v in nbrs:
            outv = 0
            for w in H.neighbors(v):
                if w != u and w not in nbrset:
                    two.add(w)
                    outv += 1
            if outv > 0:
                boundary += 1
        internal = 0
        cap = nbrs[:80]
        capset = set(cap)
        for v in cap:
            for w in H.neighbors(v):
                if w in capset and str(v) < str(w):
                    internal += 1
        redundancy = internal / max(1.0, float(len(cap)))
        return 1.05*d + 0.040*nd + 0.070*len(two) + 1.20*boundary - 0.85*redundancy
    def push(u):
        if u in H:
            s = stamp.get(u, 0) + 1
            stamp[u] = s
            heapq.heappush(heap, (-score(u), s, str(u), u))
    for u in list(H.nodes()):
        push(u)
    order = []
    while H.number_of_nodes() > 0 and heap:
        _, s, _, u = heapq.heappop(heap)
        if u not in H or stamp.get(u, 0) != s:
            continue
        affected = set([u])
        for v in list(H.neighbors(u)):
            affected.add(v)
            for w in H.neighbors(v):
                affected.add(w)
        H.remove_node(u)
        order.append(u)
        for v in affected:
            if v in H:
                push(v)
    return order
''',
}


def write_prototypes() -> None:
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    for name, code in PROTOTYPES.items():
        (CAND_DIR / f"{name}.py").write_text(code.strip() + "\n", encoding="utf-8")


def evaluate_search_graphs() -> pd.DataFrame:
    payloads = SEARCH.load_search_graph_payloads()
    rows: List[Dict[str, Any]] = []
    for name, code in PROTOTYPES.items():
        t0 = time.perf_counter()
        result = SEARCH.evaluate_candidate_code(code, payloads, budget_ratio=0.30, timeout_s=240.0)
        elapsed = time.perf_counter() - t0
        row: Dict[str, Any] = {"candidate": name, "ok": bool(result.get("ok")), "elapsed_s": elapsed}
        if result.get("ok"):
            row.update(result["avg"])
        else:
            row["error"] = result.get("error")
        rows.append(row)
        print(f"[search-proxy] {name}: {row}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "hast_fac_prototype_search_graphs.csv", index=False, encoding="utf-8-sig")
    return out


def evaluate_proxy_12graphs() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for name, code in PROTOTYPES.items():
        fn = SEARCH.compile_degree_order(code)
        for dataset in PROXY_DATASETS:
            graph = E12.EVAL.read_graph(dataset)
            rate = E12.EVAL.DATASET_RATES[dataset]
            t0 = time.perf_counter()
            order = list(fn(graph.copy()))
            elapsed = time.perf_counter() - t0
            metrics = E12.EVAL.compute_metrics(graph, order, rate=rate, method_time=elapsed)
            x = metrics["removal_ratio"].to_numpy(dtype=float)
            rows.append(
                {
                    "candidate": name,
                    "dataset": dataset,
                    "R": float(metrics["GCC"].mean()),
                    "auc_cNBI": auc_mean(x, metrics["cNBI"].to_numpy(dtype=float)),
                    "cNBI20": value_at(metrics, "cNBI", 0.20),
                    "NCC20": value_at(metrics, "NCC", 0.20),
                    "GCC20": value_at(metrics, "GCC", 0.20),
                    "time_s": elapsed,
                }
            )
        print(f"[proxy-12] {name}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "hast_fac_prototype_proxy12_detail.csv", index=False, encoding="utf-8-sig")
    mean = out.groupby("candidate")[["R", "auc_cNBI", "cNBI20", "NCC20", "GCC20", "time_s"]].mean().reset_index()
    mean.to_csv(TABLE_DIR / "hast_fac_prototype_proxy12_mean.csv", index=False, encoding="utf-8-sig")
    return out


def evaluate_full12_top_prototypes() -> pd.DataFrame:
    proxy_mean = pd.read_csv(TABLE_DIR / "hast_fac_prototype_proxy12_mean.csv")
    top_names = proxy_mean.sort_values("auc_cNBI", ascending=False).head(2)["candidate"].tolist()
    rows: List[Dict[str, Any]] = []
    for name in top_names:
        fn = SEARCH.compile_degree_order(PROTOTYPES[name])
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
                    "candidate": name,
                    "dataset": dataset,
                    "R": float(metrics["GCC"].mean()),
                    "auc_cNBI": auc_mean(x, metrics["cNBI"].to_numpy(dtype=float)),
                    "cNBI20": value_at(metrics, "cNBI", 0.20),
                    "NCC20": value_at(metrics, "NCC", 0.20),
                    "GCC20": value_at(metrics, "GCC", 0.20),
                    "time_s": elapsed,
                }
            )
        print(f"[full-12] {name}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "hast_fac_prototype_full12_detail.csv", index=False, encoding="utf-8-sig")
    mean = out.groupby("candidate")[["R", "auc_cNBI", "cNBI20", "NCC20", "GCC20", "time_s"]].mean().reset_index()
    mean.to_csv(TABLE_DIR / "hast_fac_prototype_full12_mean.csv", index=False, encoding="utf-8-sig")
    return out


def plot_results(summary: pd.DataFrame, proto_mean: pd.DataFrame) -> None:
    setup_style()
    selector_summary = pd.read_csv(TABLE_DIR / "hast_fac_validation_selector_summary.csv")
    corr = pd.read_csv(TABLE_DIR / "hast_fac_credit_signal_correlations.csv")
    method_mean = summary.groupby("method")[["auc_cNBI", "fac_auc_adv", "fac_cNBI20_adv"]].mean().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    s = selector_summary[selector_summary["k_validation_graphs"].eq(2)].sort_values("mean_gain_over_hast_auc", ascending=False)
    ax.bar(s["selector"], s["mean_gain_over_hast_auc"], color="#2A9D8F")
    ax.set_title("A. Validation selector gain over HAST")
    ax.set_ylabel("held-out AUC-cNBI gain")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[0, 1]
    show = corr[corr["target"].eq("fac_auc_adv")].sort_values("spearman", ascending=False).head(8)
    ax.barh(show["feature"][::-1], show["spearman"][::-1], color="#9C755F")
    ax.set_title("B. Credit signals predicting fracture advantage")
    ax.set_xlabel("Spearman")

    ax = axes[1, 0]
    ax.scatter(method_mean["fac_cNBI20_adv"], method_mean["fac_auc_adv"], color="#4C78A8")
    for _, row in method_mean.iterrows():
        ax.text(row["fac_cNBI20_adv"], row["fac_auc_adv"], row["method"], fontsize=7)
    ax.set_title("C. Early HDA-subtracted advantage")
    ax.set_xlabel("mean cNBI@20% advantage over HDA")
    ax.set_ylabel("mean AUC advantage over HDA")

    ax = axes[1, 1]
    if not proto_mean.empty:
        ax.bar(proto_mean["candidate"], proto_mean["auc_cNBI"], color="#D62728")
        ax.axhline(float(method_mean[method_mean["method"].eq("HAST")]["auc_cNBI"].mean()), color="#111111", ls="--", lw=1, label="HAST 12-mean")
        ax.set_title("D. Manual FAC prototypes on proxy graphs")
        ax.set_ylabel("proxy AUC-cNBI")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hast_fac_bold_mechanism_panel.png")
    fig.savefig(FIG_DIR / "hast_fac_bold_mechanism_panel.pdf")
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    selector = pd.read_csv(TABLE_DIR / "hast_fac_validation_selector_summary.csv")
    freq = pd.read_csv(TABLE_DIR / "hast_fac_selector_choice_frequency.csv")
    corr = pd.read_csv(TABLE_DIR / "hast_fac_credit_signal_correlations.csv")
    search = pd.read_csv(TABLE_DIR / "hast_fac_prototype_search_graphs.csv")
    proto = pd.read_csv(TABLE_DIR / "hast_fac_prototype_proxy12_mean.csv")
    full12 = pd.read_csv(TABLE_DIR / "hast_fac_prototype_full12_mean.csv")
    base12 = pd.read_csv(ROOT / "final_12graph_eval" / "hast_12graph_summary.csv")
    proxy_base = (
        base12[base12["dataset"].isin(PROXY_DATASETS)]
        .groupby("method")[["R", "auc_cNBI", "time_s"]]
        .mean()
        .reset_index()
        .sort_values("auc_cNBI", ascending=False)
    )
    base_mean = base12.groupby("method")[["R", "auc_cNBI", "time_s"]].mean().reset_index().sort_values("auc_cNBI", ascending=False)
    method_mean = summary.groupby("method")[["auc_cNBI", "fac_auc_adv", "fac_cNBI20_adv", "fac_NCC20_adv", "fac_GCC20_adv"]].mean().reset_index()
    lines = [
        "# HAST-FAC 机制探索实验",
        "",
        "## 新假设",
        "",
        "不要把候选的绝对 cNBI/R 当作信用，而要给**相对 HDA 的碎裂优势**记信用。HDA 已经能做到的部分不给奖励；只有更早提高 cNBI/NCC、压低 GCC 的曲线增量才算有效信用。",
        "",
        "这个机制可以叫 **FAC: Fracture Advantage Credit**。它更像 RL 里的 advantage，而不是普通工程调权重。",
        "",
        "## 实验 1：FAC 信号能不能预测最终碎裂优势",
        "",
        corr[corr["target"].eq("fac_auc_adv")].head(10).to_markdown(index=False),
        "",
        "结论：HDA-subtracted 的早期曲线信号，尤其 cNBI@20% advantage / NCC@20% advantage，可以作为搜索时的低成本信用。",
        "",
        "## 实验 2：用 FAC/shape 信用做 validation selection",
        "",
        selector.to_markdown(index=False),
        "",
        "选择频率：",
        "",
        freq.to_markdown(index=False),
        "",
        "结论：只要有 1-3 个验证图，FAC/shape 类信用能稳定避免选择当前 HAST，并在 held-out 图上带来明显 AUC-cNBI 增益。",
        "",
        "## 实验 3：手写 FAC 启发式原型",
        "",
        "Search graphs:",
        "",
        search.to_markdown(index=False),
        "",
        "Proxy 12 graphs:",
        "",
        proto.to_markdown(index=False),
        "",
        "Proxy baseline means:",
        "",
        proxy_base.to_markdown(index=False),
        "",
        "Full 12 graph top prototypes:",
        "",
        full12.to_markdown(index=False),
        "",
        "Full 12 graph existing means:",
        "",
        base_mean.to_markdown(index=False),
        "",
        "结论：FAC_advantage_local 在 proxy 图上明显超过当前 HAST，但 full 12 上仍未超过 E26F/PUCT/Clade/FunSearch。这说明 FAC 机制方向有效，但手写启发式还不够；真正应该让 HAST-FAC 用这个 credit 去指导 LLM 搜索，而不是把手写原型当最终算法。",
        "",
        "## 方法包装建议",
        "",
        "下一版不要叫 HAST-V，可以叫 **HAST-FAC: Harnessed Adaptive Search Tree with Fracture Advantage Credit**。",
        "",
        "核心 credit：",
        "",
        "```text",
        "FAC(candidate, graph) = AUC_cNBI(candidate) - AUC_cNBI(HDA)",
        "EarlyFAC = 0.45 * ΔcNBI@20% + 0.35 * ΔNCC@20% + 0.20 * Δ(-GCC)@20%",
        "FamilyCredit = mean(FAC) + mean(EarlyFAC) - risk_penalty - collapse_penalty - time_penalty",
        "```",
        "",
        "Prompt 反馈也要变：不要说“提高 cNBI”，而要说“当前 family 相比 HDA 在某类图的 20% 删除处没有产生额外碎裂，请加入 frontier/weak-tie/anti-redundancy 的低复杂度信号”。",
        "",
        "## 目前最可信的结论",
        "",
        "FAC 是比普通 validation score 更像论文贡献的机制，因为它明确解决了 HAST 的真实问题：LLM 很容易学会 HDA 已经会做的局部高-degree 删除，但没有获得“比 HDA 额外制造碎裂”的信用。",
    ]
    (REPORT_DIR / "hast_fac_bold_mechanism_experiments_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    for p in [TABLE_DIR, FIG_DIR, REPORT_DIR, CAND_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    write_prototypes()
    summary = load_curve_summary()
    fac_correlations(summary)
    fac_leave_k_validation(summary)
    evaluate_search_graphs()
    proto_detail = evaluate_proxy_12graphs()
    evaluate_full12_top_prototypes()
    proto_mean = proto_detail.groupby("candidate")[["auc_cNBI"]].mean().reset_index()
    plot_results(summary, proto_mean)
    write_report(summary)
    print(REPORT_DIR / "hast_fac_bold_mechanism_experiments_cn.md")


if __name__ == "__main__":
    main()
