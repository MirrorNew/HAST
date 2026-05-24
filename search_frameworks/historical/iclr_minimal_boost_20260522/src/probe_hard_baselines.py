# -*- coding: utf-8 -*-
"""Probe hard-to-reproduce dismantling baselines with a low-risk smoke test.

This script does not promote the probed implementations to paper-grade
baselines. It records whether a candidate public package can be imported,
whether its algorithms can be connected to our cNBI evaluator, and what obvious
interface or validity issues appear on tiny/full-evaluator smoke datasets.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


THIS = Path(__file__).resolve()
OUT_ROOT = THIS.parents[1]
WORKSPACE = OUT_ROOT.parents[1]
TABLE_DIR = OUT_ROOT / "tables"
REPORT_DIR = OUT_ROOT / "reports"

EVAL12_SRC = WORKSPACE / "research" / "dacts_e26f_reverse_search_20260519" / "src" / "evaluate_best_on_12_graphs.py"
TREE_SUMMARY = WORKSPACE / "research" / "tree_search_ablation_20260520" / "final_12graph_eval" / "final_12graph_summary.csv"
PKG_ROOT = OUT_ROOT / "external_probe" / "network_dismantling_pkg"
PKG_CODE = PKG_ROOT / "network_dismantling"

SMOKE_DATASETS = ["Powerlaw_500", "PH"]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EVAL = load_module(EVAL12_SRC, "hard_probe_eval12")


def clean_network_dismantling_modules() -> None:
    for name in list(sys.modules):
        if name == "network_dismantling" or name.startswith("network_dismantling."):
            del sys.modules[name]


def try_top_level_import() -> dict[str, str]:
    clean_network_dismantling_modules()
    sys.path.insert(0, str(PKG_ROOT))
    try:
        importlib.import_module("network_dismantling")
        return {
            "package": "network-dismantling==0.0.1",
            "check": "top_level_import",
            "status": "ok",
            "detail": "Top-level import succeeded.",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "package": "network-dismantling==0.0.1",
            "check": "top_level_import",
            "status": "failed",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if str(PKG_ROOT) in sys.path:
            sys.path.remove(str(PKG_ROOT))
        clean_network_dismantling_modules()


def load_probe_classes() -> dict[str, Callable[[], Any]]:
    """Load selected package files while bypassing package __init__.py.

    The package top-level imports optional torch modules unconditionally. For a
    smoke test we load only the simple networkx algorithms and inject the
    abstract base module they depend on.
    """

    clean_network_dismantling_modules()
    sys.modules["network_dismantling"] = types.ModuleType("network_dismantling")
    sys.modules["network_dismantling.dismanlter"] = types.ModuleType("network_dismantling.dismanlter")

    def load(name: str, rel: str) -> Any:
        return load_module(PKG_CODE / rel, name)

    load("network_dismantling.dismanlter.dismantler", "dismanlter/dismantler.py")
    core = load("hard_probe_pkg_core_hd", "dismanlter/optimization/core_hd.py")
    ci = load("hard_probe_pkg_ci", "dismanlter/influence/collective_influence.py")
    gnd = load("hard_probe_pkg_gnd", "dismanlter/optimization/gnd.py")

    # Keep smoke-test output clean; the package uses tqdm for two methods.
    core.tqdm = lambda iterable, **_: iterable
    gnd.tqdm = lambda iterable, **_: iterable

    return {
        "pkg_CoreHD": core.CoreHDDismantling,
        "pkg_CI_l2": ci.CollectiveInfluenceDismantling,
        "pkg_GND_demo": gnd.GNDDismantling,
    }


def auc_mean(x: pd.Series, y: pd.Series) -> float:
    xa = x.to_numpy(dtype=float)
    ya = y.to_numpy(dtype=float)
    if len(xa) < 2:
        return float(np.nanmean(ya)) if len(ya) else float("nan")
    order = np.argsort(xa)
    xa = xa[order]
    ya = ya[order]
    span = xa[-1] - xa[0]
    if span <= 0:
        return float(np.nanmean(ya))
    return float(np.trapezoid(ya, xa) / span)


def evaluate_smoke() -> pd.DataFrame:
    classes = load_probe_classes()
    rows: list[dict[str, Any]] = []
    for dataset in SMOKE_DATASETS:
        graph = EVAL.read_graph(dataset)
        rate = EVAL.DATASET_RATES[dataset]
        budget = max(1, int(round(graph.number_of_nodes() * rate)))
        nodes = set(graph.nodes())
        for method, cls in classes.items():
            t0 = time.perf_counter()
            try:
                np.random.seed(20260522)
                order = cls().dismantle(graph.copy(), budget)
                elapsed = time.perf_counter() - t0
                missing = sum(1 for node in order if node not in nodes)
                duplicates = len(order) - len(set(order))
                metrics = EVAL.compute_metrics(graph, order, rate=rate, method_time=elapsed)
                x = metrics["removal_ratio"]
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "status": "ok",
                        "nodes": graph.number_of_nodes(),
                        "edges": graph.number_of_edges(),
                        "budget": budget,
                        "returned": len(order),
                        "missing_node_ids": missing,
                        "duplicate_node_ids": duplicates,
                        "R": float(metrics["GCC"].mean()),
                        "auc_ACC": auc_mean(x, metrics["ACC"]),
                        "auc_NCC": auc_mean(x, metrics["NCC"]),
                        "auc_cNBI": auc_mean(x, metrics["cNBI"]),
                        "time_s": elapsed,
                        "paper_grade": False,
                        "note": note_for(method, missing),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "status": "failed",
                        "nodes": graph.number_of_nodes(),
                        "edges": graph.number_of_edges(),
                        "budget": budget,
                        "returned": 0,
                        "missing_node_ids": np.nan,
                        "duplicate_node_ids": np.nan,
                        "R": np.nan,
                        "auc_ACC": np.nan,
                        "auc_NCC": np.nan,
                        "auc_cNBI": np.nan,
                        "time_s": time.perf_counter() - t0,
                        "paper_grade": False,
                        "note": f"{type(exc).__name__}: {exc}",
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "hard_baseline_smoke_results.csv", index=False, encoding="utf-8-sig")
    return out


def note_for(method: str, missing: int) -> str:
    if method == "pkg_GND_demo":
        if missing:
            return "一次性 Laplacian demo 返回数组索引而非稳健图节点标签；不是忠实 GND 复现。"
        return "一次性 Laplacian demo；只适合作为实现 smoke test，不能当成忠实 GND。"
    if method == "pkg_CI_l2":
        return "小图可运行，但每删一个点都朴素重算 CI；没有优化前不适合直接扩到完整 12 图主表。"
    if method == "pkg_CoreHD":
        return "与仓库已有 CoreHD-style baseline 重复；论文应使用本地优化实现。"
    return ""


def reference_rows(smoke: pd.DataFrame) -> pd.DataFrame:
    if not TREE_SUMMARY.exists():
        return pd.DataFrame()
    ref = pd.read_csv(TREE_SUMMARY)
    keep_methods = ["E26F", "CoreHD", "HDA", "CI", "PUCT", "FunSearch-like", "Clade-AHD-like"]
    ref = ref[ref["dataset"].isin(SMOKE_DATASETS) & ref["method"].isin(keep_methods)].copy()
    cols = ["dataset", "method", "R", "auc_cNBI", "time_s", "source"]
    ref[cols].to_csv(TABLE_DIR / "hard_baseline_smoke_reference_rows.csv", index=False, encoding="utf-8-sig")
    return ref


def write_report(import_check: dict[str, str], smoke: pd.DataFrame, ref: pd.DataFrame) -> None:
    lines: list[str] = [
        "# 难复现 Baseline Smoke Test",
        "",
        "## 范围",
        "",
        "这个 probe 检查轻量公开包 `network-dismantling==0.0.1` 能否作为快速 hard-baseline 桥接。它只用于可行性判断，不作为论文级 baseline 证据。",
        "",
        "## 导入检查",
        "",
        f"- 顶层导入：**{import_check['status']}**。",
        f"- 细节：`{import_check['detail']}`。",
        "",
        "## Smoke 结果",
        "",
        smoke[
            [
                "dataset",
                "method",
                "status",
                "R",
                "auc_cNBI",
                "time_s",
                "missing_node_ids",
                "note",
            ]
        ].to_markdown(index=False),
        "",
        "## 邻近参考",
        "",
    ]
    if ref.empty:
        lines.append("没有找到参考 summary。")
    else:
        small = ref[["dataset", "method", "R", "auc_cNBI", "time_s", "source"]].copy()
        lines.append(small.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## 判断",
            "",
            "- `pkg_CoreHD` 是冗余的；当前仓库已经能用论文评估器在线重算 CoreHD。",
            "- `pkg_CI_l2` 只能作为小图 smoke check；它每删一个点都朴素重算 CI，若没有专门优化，不适合直接扩到完整 12 图主表。",
            "- `pkg_GND_demo` 不能写成 GND baseline。它只是一次性 Laplacian 排序，并且存在节点标签脆弱性。",
            "- 论文短期应使用已有 CoreHD/CI 表，并注明 objective/source caveat；GND/BPD/MIND/LGD-NA 只在时间允许时安排忠实复现。",
            "",
        ]
    )
    (REPORT_DIR / "hard_baseline_feasibility_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    import_check = try_top_level_import()
    pd.DataFrame([import_check]).to_csv(TABLE_DIR / "hard_baseline_import_check.csv", index=False, encoding="utf-8-sig")
    smoke = evaluate_smoke()
    ref = reference_rows(smoke)
    write_report(import_check, smoke, ref)
    print(f"[done] wrote {TABLE_DIR / 'hard_baseline_smoke_results.csv'}")
    print(f"[done] wrote {REPORT_DIR / 'hard_baseline_feasibility_cn.md'}")


if __name__ == "__main__":
    main()
