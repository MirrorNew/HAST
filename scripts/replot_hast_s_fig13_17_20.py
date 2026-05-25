# -*- coding: utf-8 -*-
"""Replot paper Fig.13/Fig.17/Fig.20 with the current HAST-S rerun data.

The canonical figure style lives in plotting/paper_figures.py.  This script
only adapts the HAST rows in temporary source tables, then calls the original
drawing functions so baseline data and visual grammar stay unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import plotting.paper_figures as paper_figures


DATASETS_12 = [
    "CEnew",
    "Collaboration",
    "condmat",
    "crime",
    "email",
    "Grid",
    "GrQC",
    "hamster",
    "HepPh",
    "PH",
    "Yeast",
    "Powerlaw_500",
]

HAST_S_METHOD = "HAST-S3-0002-0367e94c"


@contextmanager
def patched_paper_paths(source_dir: Path, figure_dir: Path):
    old_local = paper_figures.LOCAL_FIG_DIR
    old_source = paper_figures.SOURCE_TABLE_DIR
    old_benchmark = paper_figures.BENCHMARK_TABLE_DIR
    old_search = paper_figures.SEARCH_RUNTIME_TABLE_DIR
    try:
        paper_figures.LOCAL_FIG_DIR = figure_dir
        paper_figures.SOURCE_TABLE_DIR = source_dir
        paper_figures.BENCHMARK_TABLE_DIR = source_dir / "benchmark_12graph"
        paper_figures.SEARCH_RUNTIME_TABLE_DIR = source_dir / "search_runtime"
        yield
    finally:
        paper_figures.LOCAL_FIG_DIR = old_local
        paper_figures.SOURCE_TABLE_DIR = old_source
        paper_figures.BENCHMARK_TABLE_DIR = old_benchmark
        paper_figures.SEARCH_RUNTIME_TABLE_DIR = old_search


def boolish(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "ok"}


def load_hast_s_per_graph(run_dir: Path, missing_dir: Path) -> pd.DataFrame:
    original = pd.read_csv(run_dir / "full_validation" / "per_graph_metrics.csv", encoding="utf-8-sig")
    original_s = original[original["method"].eq(HAST_S_METHOD)].copy()
    original_s = original_s[original_s["valid"].map(boolish)].copy()

    missing = pd.read_csv(missing_dir / "per_graph_metrics.csv", encoding="utf-8-sig")
    missing_s = missing[missing["method"].eq(HAST_S_METHOD)].copy()
    missing_s = missing_s[missing_s["valid"].map(boolish)].copy()

    combined = pd.concat([original_s, missing_s], ignore_index=True, sort=False)
    combined = combined.sort_values("dataset").drop_duplicates("dataset", keep="last")
    missing_datasets = sorted(set(DATASETS_12) - set(combined["dataset"].astype(str)))
    if missing_datasets:
        raise SystemExit(f"HAST-Final-S still lacks valid datasets: {missing_datasets}")
    return combined[combined["dataset"].isin(DATASETS_12)].copy()


def summarize_hast_s(per_graph: pd.DataFrame, benchmark_dir: Path) -> dict[str, object]:
    baseline_per = pd.read_csv(benchmark_dir / "per_graph_metrics.csv", encoding="utf-8-sig")
    without_old_s = baseline_per[baseline_per["method"].ne("HAST-Final-S")].copy()
    s_rows = per_graph.copy()
    s_rows["method"] = "HAST-Final-S"
    ranked_input = pd.concat([without_old_s, s_rows], ignore_index=True, sort=False)
    ranked_input["rank_auc_cNBI"] = ranked_input.groupby("dataset")["auc_cNBI"].rank(method="average", ascending=False)

    s_ranked = ranked_input[ranked_input["method"].eq("HAST-Final-S")].copy()
    for col in ["R", "auc_cNBI", "time_s", "rank_auc_cNBI"]:
        s_ranked[col] = pd.to_numeric(s_ranked[col], errors="coerce")
    return {
        "method": "HAST-Final-S",
        "group": "algorithm_found",
        "evidence_tier": "current_hast_s_rerun",
        "datasets": int(s_ranked["dataset"].nunique()),
        "mean_R": float(s_ranked["R"].mean()),
        "mean_auc_cNBI": float(s_ranked["auc_cNBI"].mean()),
        "mean_time_s": float(s_ranked["time_s"].mean()),
        "top1_auc": int((s_ranked["rank_auc_cNBI"] <= 1).sum()),
        "top3_auc": int((s_ranked["rank_auc_cNBI"] <= 3).sum()),
        "mean_rank_auc": float(s_ranked["rank_auc_cNBI"].mean()),
    }


def build_adapted_tables(run_dir: Path, missing_dir: Path, adapted_source: Path) -> dict[str, object]:
    source_root = ROOT / "artifacts" / "source_tables"
    if adapted_source.exists():
        shutil.rmtree(adapted_source)
    shutil.copytree(source_root, adapted_source)

    benchmark_dir = adapted_source / "benchmark_12graph"
    hast_s_per_graph = load_hast_s_per_graph(run_dir, missing_dir)
    hast_s_summary = summarize_hast_s(hast_s_per_graph, benchmark_dir)

    mean_path = benchmark_dir / "method_mean_metrics.csv"
    mean = pd.read_csv(mean_path, encoding="utf-8-sig")
    mean = mean[mean["method"].ne("HAST-Final-S")].copy()
    mean = pd.concat([mean, pd.DataFrame([hast_s_summary])], ignore_index=True, sort=False)
    mean.to_csv(mean_path, index=False, encoding="utf-8-sig")

    search_path = adapted_source / "search_runtime" / "framework_search_time_summary.csv"
    search = pd.read_csv(search_path, encoding="utf-8-sig")
    stage_logs = [
        pd.read_csv(run_dir / "stage1_candidate_log.csv", encoding="utf-8-sig"),
        pd.read_csv(run_dir / "stage3_candidate_log.csv", encoding="utf-8-sig"),
    ]
    stage = pd.concat(stage_logs, ignore_index=True, sort=False)
    valid_rate = float(stage["valid"].map(boolish).mean())
    eval_s = pd.to_numeric(stage["time_s"], errors="coerce").fillna(0.0)
    prompt_s = pd.to_numeric(stage["llm_elapsed_s"], errors="coerce").fillna(0.0)
    row = {
        "method": "HAST-tree",
        "paper_label": "HAST bounded search",
        "group": "HAST stage",
        "candidates": int(len(stage)),
        "valid_rate": valid_rate,
        "mean_eval_s": float(eval_s.mean()),
        "median_eval_s": float(eval_s.median()),
        "total_eval_s": float(eval_s.sum()),
        "mean_prompt_s": float(prompt_s.mean()),
        "median_prompt_s": float(prompt_s.median()),
        "total_prompt_s": float(prompt_s.sum()),
        "mean_logged_search_s_per_candidate": float((eval_s + prompt_s).mean()),
        "total_logged_search_s": float((eval_s + prompt_s).sum()),
    }
    search = search[search["paper_label"].ne("HAST bounded search")].copy()
    search = pd.concat([search, pd.DataFrame([row])], ignore_index=True, sort=False)
    search.to_csv(search_path, index=False, encoding="utf-8-sig")

    return {"hast_s_summary": hast_s_summary, "search_row": row}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--missing-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    missing_dir = args.missing_dir.resolve()
    out_dir = (args.out_dir or (run_dir / "figures_replotted_hast_s")).resolve()
    adapted_source = out_dir / "source_tables_adapted"
    figure_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_adapted_tables(run_dir, missing_dir, adapted_source)
    with patched_paper_paths(adapted_source, figure_dir):
        paper_figures.setup()
        paper_figures.draw_quality_runtime()
        paper_figures.draw_high_quality_panel()
        paper_figures.draw_framework_search_time()

    summary["figures"] = [
        str(figure_dir / "fig13_12graph_quality_runtime_all_methods.png"),
        str(figure_dir / "fig17_hast_quality_speed_panel.png"),
        str(figure_dir / "fig20_framework_search_time.png"),
    ]
    (out_dir / "replot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
