# Plotting

This folder contains the current paper-facing plotting code.

| File | Role | Source data | Output |
| --- | --- | --- | --- |
| `paper_figures.py` | Regenerates main-paper figures from recorded CSVs. | `artifacts/source_tables/benchmark_12graph/`, `artifacts/source_tables/motivation_observation/`, `artifacts/source_tables/search_runtime/`, and `artifacts/source_tables/scaling/` | `artifacts/figures/` |

The plotting layer should not run searches or compute new algorithm outputs. If a figure needs new data, add or rerun an experiment entrypoint first, then consume the resulting CSV here.
