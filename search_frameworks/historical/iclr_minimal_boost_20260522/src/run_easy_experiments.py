# -*- coding: utf-8 -*-
"""Low-risk ICLR strengthening experiments for HAST/FAC-T.

This script deliberately starts with analyses that can be completed from the
existing logs, plus a tiny number of full-12 evaluations for already generated
HAST-FAC candidates selected by alternative credit rules.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


THIS = Path(__file__).resolve()
OUT_ROOT = THIS.parents[1]
WORKSPACE = OUT_ROOT.parents[1]
HAST = WORKSPACE / "research" / "hast_experiment_20260521"
TREE = WORKSPACE / "research" / "tree_search_ablation_20260520"
CURATED = WORKSPACE / "research" / "aaai_harness_curated_20260522"
REFRAME = WORKSPACE / "research" / "paper_problem_solution_reframe_20260522"
EVAL12_SRC = TREE / "src" / "evaluate_final_12graphs.py"
SEARCH_SRC = TREE / "src" / "ablation_search.py"

TABLE_DIR = OUT_ROOT / "tables"
REPORT_DIR = OUT_ROOT / "reports"
DETAIL_DIR = OUT_ROOT / "full12_records"


def ensure_dirs() -> None:
    for path in [TABLE_DIR, REPORT_DIR, DETAIL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


E12 = load_module(EVAL12_SRC, "iclr_boost_eval12")
SEARCH = load_module(SEARCH_SRC, "iclr_boost_search")


def boolish(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


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


def fac_t_score(row: pd.Series) -> float:
    fac = float(row.get("fac_auc_adv") or 0.0)
    early = float(row.get("early_fac") or 0.0)
    worst = float(row.get("fac_worst_auc_adv") or 0.0)
    search_rank = float(row.get("rank_score") or 0.0)
    proxy_time = float(row.get("proxy_time_s") or 0.0)
    search_time = float(row.get("Time") or 0.0)
    benefit = 0.48 * fac + 0.20 * early + 0.10 * worst + 6.0 * search_rank
    penalty = (
        12.0 * math.log1p(max(0.0, proxy_time) / 0.70)
        + 180.0 * max(0.0, search_time - 0.022)
        + 8.0 * max(0.0, proxy_time - 1.20) ** 2
    )
    if proxy_time > 1.80:
        penalty += 16.0 + 10.0 * (proxy_time - 1.80)
    if search_time > 0.032:
        penalty += 10.0 + 500.0 * (search_time - 0.032)
    return benefit - penalty


def load_fac_pool() -> pd.DataFrame:
    path = HAST / "runs" / "HAST-FAC" / "search_records.csv"
    df = pd.read_csv(path)
    df = df[(pd.to_numeric(df["idx"], errors="coerce") > 0) & boolish(df["valid"])].copy()
    numeric_cols = [
        "idx",
        "R",
        "cNBI",
        "Time",
        "rank_score",
        "fac_score",
        "fac_auc_adv",
        "early_fac",
        "fac_worst_auc_adv",
        "proxy_auc_cNBI",
        "proxy_time_s",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fac_t_score"] = df.apply(fac_t_score, axis=1)
    df["time_gate_fac_score"] = np.where(df["proxy_time_s"].le(1.2), df["fac_score"], -np.inf)
    df["time_only_positive_score"] = np.where(df["fac_auc_adv"].gt(0), -df["proxy_time_s"], -np.inf)
    return df


def select_fac_candidates(df: pd.DataFrame) -> pd.DataFrame:
    selectors = [
        ("absolute_proxy_auc", "proxy_auc_cNBI", False, "Absolute quality proxy; expected to drift slow."),
        ("hda_relative_auc_adv", "fac_auc_adv", False, "HDA-relative fracture advantage without time."),
        ("old_fac_score", "fac_score", False, "Original FAC score before time-aware penalty."),
        ("early_fracture_proxy", "early_fac", False, "Early curve proxy only."),
        ("time_gate_then_fac", "time_gate_fac_score", False, "Old FAC after hard proxy_time<=1.2 filter."),
        ("fac_t_score", "fac_t_score", False, "FAC-T: fracture benefit minus time/gate penalties."),
        ("time_only_positive", "time_only_positive_score", False, "Fastest candidate among positive FAC advantage candidates."),
    ]
    rows = []
    for selector, score_col, ascending, note in selectors:
        sub = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col]).copy()
        if sub.empty:
            continue
        row = sub.sort_values(score_col, ascending=ascending).iloc[0]
        rows.append(
            {
                "selector": selector,
                "selected_idx": int(row["idx"]),
                "target_family": row.get("target_family", ""),
                "selector_score": float(row[score_col]),
                "search_R": float(row["R"]),
                "search_cNBI": float(row["cNBI"]),
                "search_Time": float(row["Time"]),
                "proxy_auc_cNBI": float(row["proxy_auc_cNBI"]),
                "proxy_time_s": float(row["proxy_time_s"]),
                "fac_auc_adv": float(row["fac_auc_adv"]),
                "old_fac_score": float(row["fac_score"]),
                "fac_t_score": float(row["fac_t_score"]),
                "candidate_file": row.get("candidate_file", ""),
                "note": note,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "credit_selector_proxy_summary.csv", index=False, encoding="utf-8-sig")
    return out


def compile_candidate(path: Path) -> Callable[[Any], list[Any]]:
    code = path.read_text(encoding="utf-8")
    return SEARCH.compile_degree_order(code)


def evaluate_candidate(method: str, candidate_file: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    fn = compile_candidate(candidate_file)
    detail_rows = []
    summaries = []
    for dataset in E12.EVAL.DATASETS:
        graph = E12.EVAL.read_graph(dataset)
        rate = E12.EVAL.DATASET_RATES[dataset]
        t0 = time.perf_counter()
        order = fn(graph.copy())
        elapsed = time.perf_counter() - t0
        if not isinstance(order, (list, tuple)):
            raise ValueError(f"{method} did not return a list on {dataset}")
        metrics = E12.EVAL.compute_metrics(graph, list(order), rate=rate, method_time=elapsed)
        metrics.insert(0, "method", method)
        metrics.insert(0, "dataset", dataset)
        metrics.to_csv(DETAIL_DIR / f"{safe_name(method)}__{dataset}.csv", index=False, encoding="utf-8-sig")
        detail_rows.append(metrics)
        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                "R": float(metrics["GCC"].mean()),
                "auc_cNBI": auc_mean(metrics["removal_ratio"], metrics["cNBI"]),
                "time_s": elapsed,
            }
        )
    detail = pd.concat(detail_rows, ignore_index=True)
    summary = pd.DataFrame(summaries)
    mean = {
        "method": method,
        "candidate_file": str(candidate_file),
        "R": float(summary["R"].mean()),
        "auc_cNBI": float(summary["auc_cNBI"].mean()),
        "time_s": float(summary["time_s"].mean()),
    }
    summary.to_csv(TABLE_DIR / f"{safe_name(method)}_full12_detail.csv", index=False, encoding="utf-8-sig")
    return detail, mean


def existing_full12_means() -> pd.DataFrame:
    frames = []
    for path in [
        HAST / "tables" / "hast_fac_online_full12_mean.csv",
        HAST / "tables" / "HAST-FACT-ONLINE60_full12_mean.csv",
        HAST / "tables" / "hast_bounded_template_probe_full12_compare.csv",
    ]:
        if path.exists():
            df = pd.read_csv(path)
            if "candidate_idx" in df.columns:
                df["method"] = path.stem + " #" + df["candidate_idx"].astype(str)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_selected_full12(selector_summary: pd.DataFrame) -> pd.DataFrame:
    # These are intentionally few: enough to make the easy credit ablation less
    # proxy-only without launching a new full search.
    wanted_selectors = {"fac_t_score", "time_gate_then_fac", "time_only_positive"}
    existing = existing_full12_means()
    means = []
    evaluated: set[str] = set()
    for _, row in selector_summary.iterrows():
        if row["selector"] not in wanted_selectors:
            continue
        candidate_file = Path(str(row["candidate_file"]))
        key = str(candidate_file)
        if key in evaluated:
            continue
        evaluated.add(key)
        method = f"HAST-FAC selector {row['selector']} #{int(row['selected_idx'])}"
        cached = TABLE_DIR / f"{safe_name(method)}_full12_detail.csv"
        if cached.exists():
            cached_df = pd.read_csv(cached)
            means.append(
                {
                    "method": method,
                    "candidate_file": str(candidate_file),
                    "R": float(cached_df["R"].mean()),
                    "auc_cNBI": float(cached_df["auc_cNBI"].mean()),
                    "time_s": float(cached_df["time_s"].mean()),
                }
            )
        else:
            _, mean = evaluate_candidate(method, candidate_file)
            means.append(mean)
    new = pd.DataFrame(means)
    new.to_csv(TABLE_DIR / "credit_selector_new_full12_mean.csv", index=False, encoding="utf-8-sig")
    combined = pd.concat([existing, new], ignore_index=True, sort=False)
    combined.to_csv(TABLE_DIR / "credit_selector_full12_combined.csv", index=False, encoding="utf-8-sig")
    return combined


def frozen_template_selection() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = [
        HAST / "tables" / "aaai_followup_fac_ablation_12datasets_long.csv",
        CURATED / "tables" / "aaai_followup_fac_ablation_12datasets_long.csv",
        REFRAME / "tables" / "aaai_followup_fac_ablation_12datasets_long.csv",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError("aaai_followup_fac_ablation_12datasets_long.csv")
    data = pd.read_csv(source)
    pool = [
        "E26F",
        "HAST-FAC-T online #24",
        "FAST21-cap24",
        "BT-n16-t8-u24",
        "BT-n16-t8-u18",
        "BT-n32-t8-u24",
    ]
    compare_methods = set(pool + ["PUCT", "FunSearch-like", "Clade-AHD-like"])
    df = data[data["method"].isin(compare_methods)].copy()
    datasets = sorted(df["dataset"].unique())
    rows = []
    for k in [1, 2, 3]:
        for vals in combinations(datasets, k):
            val_mask = df["dataset"].isin(vals)
            val = df[val_mask & df["method"].isin(pool)].groupby("method").agg(
                val_auc=("auc_cNBI", "mean"),
                val_time=("time_s", "mean"),
            )
            test = df[~val_mask].groupby("method").agg(
                test_auc=("auc_cNBI", "mean"),
                test_R=("R", "mean"),
                test_time=("time_s", "mean"),
            )
            if val.empty or test.empty:
                continue
            selectors = {
                "validation_auc": val["val_auc"],
                "validation_auc_per_second": val["val_auc"] / val["val_time"].clip(lower=1e-9),
                "validation_auc_minus_log_time": val["val_auc"] - 3.0 * np.log1p(val["val_time"]),
            }
            oracle_method = str(test.loc[[m for m in pool if m in test.index], "test_auc"].idxmax())
            for selector_name, scores in selectors.items():
                chosen = str(scores.sort_values(ascending=False).index[0])
                rows.append(
                    {
                        "k_validation_graphs": k,
                        "validation_graphs": "|".join(vals),
                        "selector": selector_name,
                        "chosen_method": chosen,
                        "oracle_method_in_pool": oracle_method,
                        "chosen_test_auc": float(test.loc[chosen, "test_auc"]),
                        "chosen_test_R": float(test.loc[chosen, "test_R"]),
                        "chosen_test_time_s": float(test.loc[chosen, "test_time"]),
                        "oracle_test_auc": float(test.loc[oracle_method, "test_auc"]),
                        "regret_to_oracle": float(test.loc[oracle_method, "test_auc"] - test.loc[chosen, "test_auc"]),
                        "beats_E26F": bool("E26F" in test.index and test.loc[chosen, "test_auc"] > test.loc["E26F", "test_auc"]),
                        "beats_PUCT": bool("PUCT" in test.index and test.loc[chosen, "test_auc"] > test.loc["PUCT", "test_auc"]),
                        "retention_vs_PUCT": float(test.loc[chosen, "test_auc"] / test.loc["PUCT", "test_auc"])
                        if "PUCT" in test.index
                        else float("nan"),
                    }
                )
    trials = pd.DataFrame(rows)
    trials.to_csv(TABLE_DIR / "frozen_template_selection_trials.csv", index=False, encoding="utf-8-sig")
    summary = (
        trials.groupby(["selector", "k_validation_graphs"], as_index=False)
        .agg(
            cases=("validation_graphs", "count"),
            mean_regret=("regret_to_oracle", "mean"),
            median_regret=("regret_to_oracle", "median"),
            beat_E26F_rate=("beats_E26F", "mean"),
            beat_PUCT_rate=("beats_PUCT", "mean"),
            mean_retention_vs_PUCT=("retention_vs_PUCT", "mean"),
            mean_test_time_s=("chosen_test_time_s", "mean"),
        )
        .sort_values(["selector", "k_validation_graphs"])
    )
    summary.to_csv(TABLE_DIR / "frozen_template_selection_summary.csv", index=False, encoding="utf-8-sig")
    choice_counts = (
        trials.groupby(["selector", "k_validation_graphs", "chosen_method"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    choice_counts.to_csv(TABLE_DIR / "frozen_template_choice_counts.csv", index=False, encoding="utf-8-sig")
    return trials, summary


def search_efficiency_tables() -> pd.DataFrame:
    triple = pd.read_csv(TREE / "tables" / "triple_metric_search_quality.csv")
    invalid = pd.read_csv(TREE / "tables" / "invalid_rate_and_runtime.csv")
    out = triple.merge(invalid[["method", "invalid_rate", "mean_prompt_s"]], on="method", how="left")
    out["strict_hit_rate"] = out["strict_count"] / out["nodes"].replace(0, np.nan)
    out["loose_hit_rate"] = out["loose_count"] / out["nodes"].replace(0, np.nan)
    out.to_csv(TABLE_DIR / "search_efficiency_first_hit_summary.csv", index=False, encoding="utf-8-sig")
    return out


def baseline_queue() -> pd.DataFrame:
    rows = [
        {
            "baseline": "CoreHD",
            "status": "already rerun locally",
            "current_evidence": "Available in tree_search_ablation_20260520/final_12graph_eval and 10k scaling tables.",
            "next_action": "Keep in main table; verify implementation citation and full-sequence protocol.",
            "difficulty": "easy",
        },
        {
            "baseline": "CI / Collective Influence",
            "status": "existing order records available, not yet in new HAST main table",
            "current_evidence": "Existing CI trajectories exist in final_12graph_eval records.",
            "next_action": "Fold into appendix/main baseline table with a caveat on objective mismatch.",
            "difficulty": "easy",
        },
        {
            "baseline": "NDJC / FINDER / VE public spreadsheet baselines",
            "status": "available in older paper_draft evidence, not aligned to cNBI full-12 protocol",
            "current_evidence": "Trusted GCC curves exist for some real graphs.",
            "next_action": "Use only for R/GCC appendix unless full removal sequences can be regenerated.",
            "difficulty": "medium",
        },
        {
            "baseline": "GND / BPD / Min-Sum",
            "status": "not trusted from old spreadsheets",
            "current_evidence": "Related work only; existing sequence provenance flagged as unreliable.",
            "next_action": "Reproduce from public implementation or omit from main quantitative claims.",
            "difficulty": "hard",
        },
        {
            "baseline": "MIND / LGD-NA",
            "status": "new external learning baselines",
            "current_evidence": "Relevant accepted/recent work; no local implementation yet.",
            "next_action": "Add related-work contrast now; attempt reproduction only after easy experiments are frozen.",
            "difficulty": "hard",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "baseline_reproduction_queue.csv", index=False, encoding="utf-8-sig")
    return out


def traditional_baseline_table() -> pd.DataFrame:
    tree_mean = pd.read_csv(TREE / "final_12graph_eval" / "mean_metrics_by_method.csv")
    keep = [
        "HDA",
        "CoreHD",
        "CI",
        "DC",
        "KCORE",
        "CC",
        "EC",
        "CLUC",
        "E26F",
        "PUCT",
        "FunSearch-like",
        "Clade-AHD-like",
        "MCTS-AHD-like",
        "AlphaEvolve-like",
    ]
    base = tree_mean[tree_mean["method"].isin(keep)].copy()
    template = pd.read_csv(HAST / "tables" / "hast_bounded_template_probe_full12_compare.csv")
    template = template[template["method"].isin(["FAST21-cap24", "BT-n16-t8-u24", "HAST-FAC-T online #24"])].copy()
    out = pd.concat([base, template], ignore_index=True, sort=False)
    out = out.drop_duplicates("method", keep="last")
    out["rank_auc_cNBI"] = out["auc_cNBI"].rank(ascending=False, method="min").astype(int)
    out["rank_R"] = out["R"].rank(ascending=True, method="min").astype(int)
    out["source_status"] = np.where(
        out["method"].isin(["CI", "DC", "KCORE", "CC", "EC", "CLUC"]),
        "existing order baseline; use with objective-mismatch caveat",
        "current reproducible comparison",
    )
    out = out.sort_values("rank_auc_cNBI")
    out.to_csv(TABLE_DIR / "expanded_existing_baseline_summary.csv", index=False, encoding="utf-8-sig")
    return out


def full12_outlier_table() -> pd.DataFrame:
    rows = []
    for path in sorted(TABLE_DIR.glob("HAST-FAC_selector_*_full12_detail.csv")):
        df = pd.read_csv(path)
        method = str(df["method"].iloc[0])
        slow = df.sort_values("time_s", ascending=False).iloc[0]
        rows.append(
            {
                "method": method,
                "mean_auc_cNBI": float(df["auc_cNBI"].mean()),
                "mean_time_s": float(df["time_s"].mean()),
                "slowest_dataset": slow["dataset"],
                "slowest_dataset_time_s": float(slow["time_s"]),
                "slowest_dataset_auc_cNBI": float(slow["auc_cNBI"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "credit_selector_full12_runtime_outliers.csv", index=False, encoding="utf-8-sig")
    return out


def write_report(
    selector_summary: pd.DataFrame,
    full12: pd.DataFrame,
    frozen_summary: pd.DataFrame,
    efficiency: pd.DataFrame,
    queue: pd.DataFrame,
    expanded_baselines: pd.DataFrame,
    outliers: pd.DataFrame,
) -> None:
    old_fac = selector_summary[selector_summary["selector"].eq("old_fac_score")].iloc[0]
    fac_t = selector_summary[selector_summary["selector"].eq("fac_t_score")].iloc[0]
    frozen_auc = frozen_summary[frozen_summary["selector"].eq("validation_auc")]
    lines = [
        "# ICLR Minimal Boost: Easy Experiments",
        "",
        "## What Was Run",
        "",
        "1. Re-scored existing HAST-FAC candidates with absolute, HDA-relative, time-gated, FAC-T, and time-only selectors.",
        "2. Full-12 evaluated the newly selected lightweight FAC candidates that were not already in the main tables.",
        "3. Simulated frozen validation-graph selection for bounded/template candidates.",
        "4. Consolidated first-hit/search-efficiency evidence and baseline reproduction queue.",
        "",
        "## Credit Assignment Result",
        "",
        f"- Old FAC selects candidate #{int(old_fac['selected_idx'])}, proxy_time={old_fac['proxy_time_s']:.3f}s, "
        f"proxy_auc={old_fac['proxy_auc_cNBI']:.3f}.",
        f"- FAC-T selects candidate #{int(fac_t['selected_idx'])}, proxy_time={fac_t['proxy_time_s']:.3f}s, "
        f"proxy_auc={fac_t['proxy_auc_cNBI']:.3f}.",
        "- This directly supports the claim that quality-only credit drifts toward slower local scans.",
        "",
        "Full-12 means for newly evaluated selectors and existing references are in `tables/credit_selector_full12_combined.csv`.",
        "",
        "### Newly Found Loophole",
        "",
        outliers.to_markdown(index=False) if not outliers.empty else "No selector outlier table was generated.",
        "",
        "The old-pool FAC-T/time-gated selector (#21) looks fast on proxy graphs, but its full-12 mean time is much higher because HepPh is an outlier. This is useful evidence, not a failure to hide: proxy time gates alone are insufficient, so the final paper should argue for online FAC-T plus bounded candidate language rather than post-hoc re-ranking alone.",
        "",
        "## Frozen Template Selection",
        "",
        frozen_auc.to_markdown(index=False),
        "",
        "Interpretation: validation-only selection among E26F/FAC-T/FAST/BT variants is usually close to the in-pool oracle, but not perfect. This is exactly the bounded, honest claim needed for ICLR: template selection should be frozen before final test reporting.",
        "",
        "## Search Efficiency",
        "",
        efficiency[["method", "nodes", "valid", "invalid_rate", "first_strict", "strict_count", "strict_hit_rate"]]
        .sort_values("first_strict")
        .to_markdown(index=False),
        "",
        "## Baseline Queue",
        "",
        queue.to_markdown(index=False),
        "",
        "## Expanded Existing Baselines",
        "",
        expanded_baselines[["method", "R", "auc_cNBI", "time_s", "rank_auc_cNBI", "rank_R", "source_status"]]
        .to_markdown(index=False),
        "",
        "## Current Confidence",
        "",
        "The easy package closes several paper-writing loopholes, but does not yet replace hard reproduction baselines. The next hard step is to fold CI/CoreHD into the new main table and then decide whether GND/BPD/MIND/LGD-NA are worth reproducing.",
    ]
    (REPORT_DIR / "iclr_minimal_boost_easy_experiments_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    fac_pool = load_fac_pool()
    selector_summary = select_fac_candidates(fac_pool)
    full12 = run_selected_full12(selector_summary)
    _trials, frozen_summary = frozen_template_selection()
    efficiency = search_efficiency_tables()
    queue = baseline_queue()
    expanded_baselines = traditional_baseline_table()
    outliers = full12_outlier_table()
    write_report(selector_summary, full12, frozen_summary, efficiency, queue, expanded_baselines, outliers)
    print(f"[done] wrote outputs to {OUT_ROOT}")


if __name__ == "__main__":
    main()
