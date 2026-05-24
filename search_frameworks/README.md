# Search Frameworks

This directory contains search framework implementations only. Historical experiment drivers and plotting utilities are kept in `experiments/`, `plotting/`, `data/`, or `artifacts/` so the project structure stays clean.

## Contents

- `era_like.py`: paper-facing ERA-like policy entrypoint. Raw records may still use the internal `PUCT` key.
- `mcts_ahd_like.py`: MCTS-AHD-like policy entrypoint.
- `clade_ahd_like.py`: Clade-AHD-like policy entrypoint.
- `funsearch_like.py`: FunSearch-like policy entrypoint.
- `alphaevolve_like.py`: AlphaEvolve-like policy entrypoint.
- `generic_llm_search_ablation.py`: shared HDA-root LLM program-search harness used by the `xxx-like` policy entrypoints.
- `dacts_style_search.py`: DACTS-style typed HDA-root search runner preserved for mechanism-credit provenance.
- `hast_legacy_search.py`: earlier HAST search implementation that learns mechanism-level experience from generic search logs. It is kept because it is part of the paper's search-history provenance, not because it is the current main HAST implementation.
- `hast_fac_online_search.py`: HAST-FAC online search implementation using HDA-relative fracture advantage and time-aware credit.
- `runtime_deps/`: local copied runtime modules required by the framework harnesses. These files replace imports from external research folders.

## Nearby Directories

- `experiments/`: current paper-facing table/experiment entrypoints.
- `plotting/`: current paper-facing figure generation.
- `runs/` or other local-only run folders: raw search traces when reproduced locally.
- `artifacts/source_tables/hast_search_evidence/`: local-only HAST/search candidate evidence tables retained for rerun provenance.

## Naming Note

The public paper label is `ERA-like`. Some raw records and earlier scripts still use the internal raw key `PUCT` because that was the original run directory and CSV key. Paper-facing scripts map this raw key to `ERA-like` before exporting figures and tables.

## Data Locations

- Summary/source tables are local under the semantic subfolders of `artifacts/source_tables/`.
- Raw search records are local-only and should remain outside the public commit, usually under ignored `runs/` or `outputs/` folders.
- Both locations are intentionally ignored by Git. Public GitHub uploads should include this code and documentation, not the large run records or generated artifacts.

## Status

New experiments should prefer the canonical project modules in `hast/`, `baselines/`, `metrics/`, and `scripts/`, then write outputs under `runs/`, `outputs/`, or `artifacts/`. Use the framework files here when reproducing or extending ERA-like, MCTS-AHD-like, Clade-AHD-like, FunSearch-like, AlphaEvolve-like, DACTS-style, legacy HAST, or HAST-FAC search policies.
