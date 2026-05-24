# Search Framework Code and Data Index

This index records the historical search-framework assets now mirrored inside `HAST2026/main`.

## Code Mirrors

| Local path | Purpose |
| --- | --- |
| `search_frameworks/historical/tree_search_ablation_20260520/src/` | Generic framework ablation code for ERA-like, MCTS-AHD-like, Clade-AHD-like, FunSearch-like, AlphaEvolve-like, DACTS-style reruns, credit probes, and cross-framework mechanism transfer. |
| `search_frameworks/historical/hast_experiment_20260521/src/` | Earlier HAST, HAST-FAC, HAST-FAC-T, bounded-template, time-aware credit, and follow-up experiment code. |
| `search_frameworks/historical/paper_problem_solution_reframe_20260522/src/` | Motivation observation experiments, candidate interpretability, and paper figure/table refresh utilities. |
| `search_frameworks/historical/iclr_minimal_boost_20260522/src/` | Minimal boost, selector, and hard-baseline probe scripts. |

## Local Data Mirrors

| Local path | Contents | Git policy |
| --- | --- | --- |
| `artifacts/source_tables/historical_search_frameworks/tree_search_ablation_20260520/` | Derived CSV tables for generic search framework ablations and mechanism transfer. | Ignored |
| `artifacts/source_tables/historical_search_frameworks/hast_experiment_20260521/` | Derived CSV tables for early HAST / HAST-FAC / HAST-FAC-T experiments. | Ignored |
| `artifacts/source_tables/historical_search_frameworks/iclr_minimal_boost_20260522/` | Derived CSV tables for compact selector and baseline probes. | Ignored |
| `artifacts/source_tables/historical_search_frameworks/paper_problem_solution_reframe_20260522_tables/` | Motivation and paper-reframing CSV tables used by earlier figure scripts. | Ignored |
| `data/search_framework_records/raw/tree_search_ablation_20260520/` | Raw historical generic-framework run records, generated candidates, and logs. | Ignored |
| `data/search_framework_records/raw/hast_experiment_20260521/` | Raw historical HAST-family run records, generated candidates, and logs. | Ignored |

## Public Naming

Use `ERA-like` in paper text, figures, and documentation. Some historical CSVs and scripts use `PUCT` as an internal raw key because that was the original run-directory label. Paper-facing scripts should map `PUCT` to `ERA-like` at export time.

## Reuse Rule

New experiments should not import files from the pre-migration source tree. If a historical routine is needed, use the mirrored copy under `search_frameworks/historical/` or port the relevant logic into the canonical modules under `hast/`, `baselines/`, `metrics/`, and `scripts/`.
