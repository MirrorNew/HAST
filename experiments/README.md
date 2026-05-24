# Experiments

This folder keeps only current, paper-facing experiment entrypoints without date suffixes.

| File | Role | Writes tables? | Draws figures? |
| --- | --- | --- | --- |
| `obs1_basic_baseline_horizontal.py` | Regenerate the verified Observation 1 same-R residual-network panel from the fixed Collaboration case table. | yes | yes |
| `paper_source_tables.py` | Synchronize and normalize recorded experiment CSVs into `artifacts/source_tables/`. | yes | no |
| `motivation_observation_contract.py` | Documents the current Observation 2/3 rerun groups and budgets. | no | no |
| `scaling_contract.py` | Documents the current scaling sizes, seeds, and methods. | no | no |

Exploratory scripts were removed from this public `main` layout because they mixed experiments, plotting, local paths, and dated provenance. Their derived records are preserved under `artifacts/source_tables/` and local raw records under `data/`, both excluded from GitHub when large.
