# HAST Main Project

This directory is the independent HAST2026 implementation workspace. It is designed to run without importing the old research experiment tree.

## What Is Included

- `hast/`: three-stage HAST search code, candidate validation, Stage 2 bound induction, ranking, and Pareto selection.
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
├── hast/                     # Canonical three-stage HAST implementation
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

- Stage 1 `cost-aware free search`: 300 sequential search-tree expansion calls from the HDA-original root.
- Stage 2 `log-induced bound induction`: 10 LLM calls. This stage only induces bounds/policy and does not generate candidate algorithms.
- Stage 3 `bounded guided search`: 200 sequential bounded search-tree expansion calls from selected Stage 1 parent nodes.
- Current paper-facing root run uses `Delta AUC-cNBI = AUC-cNBI(child) - AUC-cNBI(root HDA node)` for active search credit. Parent-relative and root-relative columns are both logged for audit, but `runs_HAST_root_e4_e7_main_tree_20260525` and `artifacts/source_tables/` now use the root-relative HAST data.
- LLM setting: `gpt-5.5`, reasoning effort `none`, temperature `0.2`, OpenAI-compatible base URL `https://api.ritelt.com/v1`.

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
python scripts/run_main_search.py
python scripts/run_e7_full_validation.py --prepare-only --datasets Powerlaw_500
```

The smoke test validates that baseline algorithms, metrics, candidate execution, Stage 2 policy induction, and final Pareto labels are wired correctly.

## Real LLM Provider

Real LLM calls are intentionally explicit and read secrets from environment variables:

```powershell
$env:HAST_LLM_API_KEY = "<your-api-key>"
$env:HAST_LLM_BASE_URL = "https://api.ritelt.com/v1"
$env:HAST_LLM_MODEL = "gpt-5.5"
$env:HAST_LLM_REASONING_EFFORT = "none"
$env:HAST_LLM_TEMPERATURE = "0.2"
python scripts/run_main_search.py --execute
```

Do not commit API keys. `.env` and `*.key` are ignored for local secret storage.

The E4-E7 runner is also explicit. Without `--execute`, it only prepares and
validates the benchmark/source-table wiring:

```powershell
python scripts/run_main_search.py
python scripts/run_main_search.py --execute
python scripts/run_main_search.py --execute --run-e7
python scripts/run_e7_full_validation.py --candidate-dir runs/runs_HAST_parent_main_YYYYMMDD/final
```

Each E4-E7 run creates a fresh directory directly under `runs/` using:

```text
runs/runs_HAST_parent_<experiment>_<YYYYMMDD>
runs/runs_HAST_root_<experiment>_<YYYYMMDD>
```

`parent` is the main parent-relative run; `root` is the root-relative
`e7_additional` ablation. Existing non-empty run directories are rejected unless
`--allow-existing-run-dir` is passed intentionally. Each run directory contains
`input_parameters.json`, which records the CLI inputs and resolved experiment
configuration without storing API keys.

The main run uses parent-relative tree credit by default. The matching
`e7_additional` ablation keeps the same budgets, model, timeout, datasets, and
candidate interface, but switches tree-node scoring to root-relative credit:

```powershell
python scripts/run_main_search.py --run-name e7_additional --execute --run-e7 --delta-credit-mode root
```

## Full Experiment Shape

The full paper run should execute:

1. Observation 2: `R/GCC-only` vs `Absolute-cNBI` vs `Relative-Delta-cNBI`, 100 candidates per group.
2. Observation 3: `Relative-Free` vs `CostAware-Free` vs `Bounded-Guided`, 100 candidates per group.
3. Main HAST: 300 + 10 + 200 tree-node budget, with E6 freezing `HAST-Final-Q/S` and E7 only evaluating the frozen pair on the full 12-graph benchmark.
4. Ablation/knockout: relative credit, time penalty, Stage 2 bounds, Stage 1-only parent choice, and bounded-family width.
5. Scaling: full evaluation for 500 to 10k; runtime-only for 500 to 1000k.
6. Paper artifact refresh for all HAST-dependent figures and tables listed in `docs/hast_main_experiment_plan.md`; Observation 1's `fig21_obs1_basic_baseline_same_r_horizontal.png` is retained rather than treated as a HAST rerun target.

## Local Provenance Assets

Prior run records needed for reproducibility are kept inside `main/data/` and `main/artifacts/`. The cross-folder source-map Markdown files were removed so that `main` is the only source boundary. Use the public paper label `ERA-like`; raw records may still use `PUCT` as the internal CSV/run key, and paper-facing scripts map that key to `ERA-like` during export.
