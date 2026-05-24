# Search Framework Records

This directory is the local-only holding area for historical search-framework run records.

## Local Layout

- `raw/tree_search_ablation_20260520/`: raw generic framework search traces and candidate records for ERA-like, MCTS-AHD-like, Clade-AHD-like, FunSearch-like, AlphaEvolve-like, and related analyses.
- `raw/hast_experiment_20260521/`: raw early HAST / HAST-FAC / HAST-FAC-T traces and candidate records.

## Git Policy

`raw/` is ignored by Git because it contains large logs, generated candidates, and raw LLM traces. Keep this README tracked so the local data contract remains visible after cloning.

For paper-facing figures and tables, use the derived source tables under `artifacts/source_tables/` and the regeneration scripts under `scripts/`.
