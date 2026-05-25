# Experiments

This folder keeps only current, paper-facing experiment entrypoints without date suffixes.

| File | Role | Writes tables? | Draws figures? |
| --- | --- | --- | --- |
| `obs1_basic_baseline_horizontal.py` | Regenerate the verified Observation 1 same-R residual-network panel from the fixed Collaboration case table. | yes | yes |
| `hast_main_search.py` | Prepare or execute HAST E4-E6 search and optionally call E7 after Stage 3. | yes | no |
| `full_validation.py` | Run E7 full validation for E6-selected HAST-Final-Q/S candidates and write benchmark-compatible tables. | yes | no |
| `motivation_observation_experiments.py` | Execute E1-E3 motivation experiments, including the cNBI same-GCC bar chart and deterministic proxy 3 x 100 candidate runs for Observation 2/3. | yes | yes |
| `paper_source_tables.py` | Synchronize and normalize recorded experiment CSVs into `artifacts/source_tables/`. | yes | no |
| `motivation_observation_contract.py` | Documents the current Observation 2/3 rerun groups and budgets. | no | no |
| `scaling_contract.py` | Documents the current scaling sizes, seeds, and methods. | no | no |

Main HAST E4-E6 search orchestration lives in `experiments/hast_main_search.py` and `hast/e4_e6.py`; E7 full validation lives in `experiments/full_validation.py`, with `scripts/run_e7_full_validation.py` kept as a thin CLI wrapper.

Exploratory scripts were removed from this public `main` layout because they mixed experiments, plotting, local paths, and dated provenance. Their derived records are preserved under `artifacts/source_tables/` and local raw records under `data/`, both excluded from GitHub when large.
