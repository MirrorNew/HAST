# HAST-Lite-Full Main Project

This directory is the independent HAST2026 implementation workspace. It is designed to run without importing the old research experiment tree.

## What Is Included

- `hast/`: three-stage HAST-Lite-Full search code, candidate validation, Stage 2 bound induction, ranking, and Pareto selection.
- `baselines/`: HDA-original, HDA-fast, CoreHD-original, CoreHD-fast, DC, KCore, CLUC/ClusterRank-style, CI, NDC/NCDC/NDJC, and documented Python fallbacks for BPD/MinSum and GND-style baselines.
- `metrics/`: GCC/R, NCC, cNBI, AUC-cNBI, final metrics, and runtime summaries.
- `scripts/`: smoke test, dry-run main search, motivation experiment contract, full validation, scaling contract, and paper artifact export list.
- `configs/`: fixed experiment parameters.
- `search_frameworks/`: search framework implementations only, including ERA-like generic search, DACTS-style search, prior HAST, and HAST-FAC.
- `experiments/`: current paper-facing experiment/table entrypoints; old dated exploratory scripts are not kept here.
- `plotting/`: current paper-facing figure generation from recorded CSV tables.
- `docs/`: full experiment plan and reproduction notes.
- `network/`: local graph inputs needed by smoke tests and figure reconstruction. Paper-facing baseline tables live under `artifacts/source_tables/`.

## Project Architecture

```text
HAST2026/main/
├── hast/                     # Canonical three-stage HAST-Lite-Full implementation
│   ├── candidate.py          # Candidate contract, validation, and execution helpers
│   ├── search1_3.py             # Stage 1 free search and Stage 3 bounded search orchestration
│   ├── stage2.py             # Log-induced bound induction with the fixed 10-call budget
│   ├── ranking.py            # Unified proxy ranking and Pareto final selection
│   └── data.py               # Benchmark loading helpers and fixture access
├── baselines/                # Traditional/strong baseline orderings and documented fallbacks
├── metrics/                  # Fragmentation metrics and curve summaries
├── configs/                  # Fixed seeds, graph sets, LLM settings, and budgets
├── scripts/                  # Reproducible entrypoints for search, validation, scaling, figures, audit
├── search_frameworks/        # Search framework implementations only
│   ├── era_like.py
│   ├── mcts_ahd_like.py
│   ├── clade_ahd_like.py
│   ├── funsearch_like.py
│   ├── alphaevolve_like.py
│   ├── generic_llm_search_ablation.py
│   ├── dacts_style_search.py
│   ├── hast_legacy_search.py
│   ├── hast_fac_online_search.py
│   └── runtime_deps/          # Local copied runtime modules; no external research-tree imports
├── experiments/              # Current table/experiment contracts, no dated historical folders
│   ├── paper_source_tables.py
│   ├── motivation_observation_contract.py
│   └── scaling_contract.py
├── plotting/                 # Current paper figure regeneration
│   └── paper_figures.py
├── network/                  # Edgelists used by smoke tests and figure reconstruction
├── artifacts/
│   ├── source_tables/        # Local-only canonical CSV sources for paper figures/tables
│   │   ├── benchmark_12graph/
│   │   │   ├── CEnew/point_evaluations/
│   │   │   ├── ...
│   │   │   └── Yeast/point_evaluations/
│   │   ├── motivation_observation/
│   │   ├── search_runtime/
│   │   ├── scaling/
│   │   └── hast_search_evidence/
│   ├── figures/              # Local regenerated figures
│   └── reports/              # Local audit reports
└── docs/                     # Paper draft and experiment plan
```

The canonical experiment path is `configs -> hast/baselines/metrics -> experiments/scripts -> artifacts -> plotting`. Search framework code lives in `search_frameworks/`; current table-generation and experiment contracts live in `experiments/`; current paper plotting lives in `plotting/`. New paper-facing HAST runs should be launched from `scripts/` or `experiments/` and write outputs into `runs/`, `outputs/`, or `artifacts/`.

## Fixed HAST Budget

- Stage 1 `cost-aware free search`: 300 candidate calls.
- Stage 2 `log-induced bound induction`: 10 LLM calls. This stage only induces bounds/policy and does not generate candidate algorithms.
- Stage 3 `bounded guided search`: 200 candidate calls.
- LLM setting: `GPT-5.5`, reasoning effort `none`, temperature `0.2`.

## Local Data Policy

The following paths are intentionally local-only:

- large graph inputs beyond `network/`
- `artifacts/source_tables/`
- `artifacts/figures/`
- `artifacts/reports/`
- `runs/`
- `outputs/`
- raw LLM logs and large CSV records

They are excluded in `.gitignore`. GitHub should contain source code, configs, docs, and lightweight fixtures only.

## Quick Checks

```powershell
python scripts/smoke_test.py
python scripts/run_main_search.py --dry-run
python scripts/run_full_validation.py --datasets Powerlaw_500
```

The smoke test validates that baseline algorithms, metrics, candidate execution, Stage 2 policy induction, and final Pareto labels are wired correctly.

## Full Experiment Shape

The full paper run should execute:

1. Observation 2: `GCC/R-only` vs `Absolute-cNBI` vs `Relative-Delta-cNBI`, 100 candidates per group.
2. Observation 3: `Relative-Free` vs `CostAware-Free` vs `Bounded-Guided`, 100 candidates per group.
3. Main HAST: 300 + 10 + 200 budget, followed by full 12-graph validation and Pareto selection of `HAST-Final-Q/S`.
4. Scaling: full evaluation for 500 to 10k; runtime-only for 500 to 1000k.
5. Paper artifact refresh for all HAST-dependent figures and tables listed in `docs/hast_main_experiment_plan.md`.

## Local Provenance Assets

Prior run records needed for reproducibility are kept inside `main/data/` and `main/artifacts/`. The cross-folder source-map Markdown files were removed so that `main` is the only source boundary. Use the public paper label `ERA-like`; raw records may still use `PUCT` as the internal CSV/run key, and paper-facing scripts map that key to `ERA-like` during export.
