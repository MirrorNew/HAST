# Historical Search Frameworks

This directory preserves the non-main search framework code that informed the HAST2026 paper narrative. The canonical HAST-Lite-Full implementation remains in `hast/`; these files are kept for auditability, re-analysis, and controlled follow-up experiments.

## Contents

- `historical/tree_search_ablation_20260520/src/`: generic LLM search-framework ablations, including ERA-like, MCTS-AHD-like, Clade-AHD-like, FunSearch-like, AlphaEvolve-like, DACTS rerun analysis, credit-boundary probes, and cross-framework mechanism transfer analysis.
- `historical/hast_experiment_20260521/src/`: earlier HAST / HAST-FAC / HAST-FAC-T experiments, family learning, time-aware credit analysis, bounded-template probes, and follow-up experiment suites.
- `historical/paper_problem_solution_reframe_20260522/src/`: motivation observation experiments, final candidate interpretability scripts, paper figure refresh scripts, and problem-solution reframing utilities.
- `historical/iclr_minimal_boost_20260522/src/`: compact hard-baseline probes, selector checks, and minimal boost experiments.

## Naming Note

The public paper label is `ERA-like`. Some historical raw records and old scripts still use the internal raw key `PUCT` because that was the original run directory and CSV key. Paper-facing scripts map this raw key to `ERA-like` before exporting figures and tables.

## Data Locations

- Summary/source tables are local under `artifacts/source_tables/historical_search_frameworks/`.
- Raw historical search records are local under `data/search_framework_records/raw/`.
- Both locations are intentionally ignored by Git. Public GitHub uploads should include this code and documentation, not the large run records or generated artifacts.

## Status

These scripts are preserved as historical experiment entrypoints. New experiments should prefer the canonical project modules in `hast/`, `baselines/`, `metrics/`, and `scripts/`, then write outputs under `runs/`, `outputs/`, or `artifacts/`.
