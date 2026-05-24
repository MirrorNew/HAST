# HAST-Lite-Full Main Project

This directory is the independent HAST2026 implementation workspace. It is designed to run without importing the old research experiment tree.

## What Is Included

- `hast/`: three-stage HAST-Lite-Full search code, candidate validation, Stage 2 bound induction, ranking, and Pareto selection.
- `baselines/`: HDA-original, HDA-fast, CoreHD-fast, and DC baseline implementations.
- `metrics/`: GCC/R, NCC, cNBI, AUC-cNBI, final metrics, and runtime summaries.
- `scripts/`: smoke test, dry-run main search, motivation experiment contract, full validation, scaling contract, and paper artifact export list.
- `configs/`: fixed experiment parameters.
- `search_frameworks/`: historical search-framework code mirrors for ERA-like, MCTS-AHD-like, Clade-AHD-like, FunSearch-like, AlphaEvolve-like, early HAST/HAST-FAC, and motivation experiments.
- `docs/`: full experiment plan and reproduction notes.
- `data/`: local data area. Raw benchmark data and baseline records are present locally but excluded from GitHub.

## Project Architecture

```text
HAST2026/main/
├── hast/                     # Canonical three-stage HAST-Lite-Full implementation
│   ├── candidate.py          # Candidate contract, validation, and execution helpers
│   ├── search.py             # Stage 1 free search and Stage 3 bounded search orchestration
│   ├── stage2.py             # Log-induced bound induction with the fixed 10-call budget
│   ├── ranking.py            # Unified proxy ranking and Pareto final selection
│   └── data.py               # Benchmark loading helpers and fixture access
├── baselines/                # HDA-original, HDA-fast, CoreHD-fast, DC
├── metrics/                  # Fragmentation metrics and curve summaries
├── configs/                  # Fixed seeds, graph sets, LLM settings, and budgets
├── scripts/                  # Reproducible entrypoints for search, validation, scaling, figures, audit
├── search_frameworks/        # Historical framework mirrors and audit-only code
│   └── historical/
│       ├── tree_search_ablation_20260520/
│       ├── hast_experiment_20260521/
│       ├── paper_problem_solution_reframe_20260522/
│       └── iclr_minimal_boost_20260522/
├── data/
│   ├── fixtures/             # Tiny tracked smoke-test graph
│   ├── benchmarks/raw/       # Local-only benchmark edgelists
│   ├── baseline_records/     # Local-only baseline record cache
│   └── search_framework_records/raw/
│       └── ...               # Local-only raw historical search traces
├── artifacts/
│   ├── source_tables/        # Local-only canonical CSV sources for paper figures/tables
│   ├── figures/              # Local regenerated figures
│   └── reports/              # Local audit reports
└── docs/                     # Data indexes, figure mapping, experiment plan, reproducibility notes
```

The canonical experiment path is `configs -> hast/baselines/metrics -> scripts -> artifacts`. Historical framework code is available for provenance and follow-up analyses, but new paper-facing HAST runs should be launched from `scripts/` and write outputs into `runs/`, `outputs/`, or `artifacts/`.

## Fixed HAST Budget

- Stage 1 `cost-aware free search`: 300 candidate calls.
- Stage 2 `log-induced bound induction`: 10 LLM calls. This stage only induces bounds/policy and does not generate candidate algorithms.
- Stage 3 `bounded guided search`: 200 candidate calls.
- LLM setting: `GPT-5.5`, reasoning effort `none`, temperature `0.2`.

## Local Data Policy

The following paths are intentionally local-only:

- `data/benchmarks/raw/`
- `data/baseline_records/`
- `data/search_framework_records/raw/`
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

## Historical Frameworks

The historical framework mirror is indexed in `docs/search_framework_data_index.md`. Use the public paper label `ERA-like`; historical raw records may still use `PUCT` as the internal CSV/run key, and paper-facing scripts map that key to `ERA-like` during export.
