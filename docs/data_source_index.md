# Data Source Index

`HAST2026/main` is the intended independent workspace for future HAST experiments.

## Canonical Local Sources

The current paper-facing figures are generated from:

- `artifacts/source_tables/paper_tables/table_12graph_method_mean_metrics.csv`
- `artifacts/source_tables/paper_tables/table_12graph_extended_curve_records.csv`
- `artifacts/source_tables/paper_tables/table_framework_search_time_summary.csv`
- `artifacts/source_tables/paper_tables/table_module_ablation_three_mechanisms.csv`
- `artifacts/source_tables/paper_tables/table_final_candidate_component_ablation.csv`
- `artifacts/source_tables/paper_tables/table_final_candidate_selected_node_features.csv`
- `artifacts/source_tables/tables/motivation_obs2_obs3_stage_evidence.csv`
- `artifacts/source_tables/tables/hast_fac_credit_signal_correlations.csv`
- `artifacts/source_tables/tables/motivation_obs2_fac_code_feature_by_time_bucket.csv`
- `artifacts/source_tables/tables/scaling_full_eval_500_to_10k_unified.csv`
- `artifacts/source_tables/tables/runtime_only_scaling_500_to_1000k_unified.csv`

Historical search-framework data mirrors are indexed in `docs/search_framework_data_index.md` and stored locally under:

- `artifacts/source_tables/historical_search_frameworks/`
- `data/search_framework_records/raw/`

## Canonical Local Scripts

- `scripts/sync_record_based_paper_tables.py`: synchronizes derived summary tables to the record-derived 12-graph metrics.
- `scripts/regenerate_record_based_figures.py`: regenerates local figures under `artifacts/figures/` and exports copies to `HAST2026/02_main_figures/`.
- `scripts/audit_paper_data_alignment.py`: checks paper tables against the `main/artifacts` CSVs and writes local audit outputs.
- `scripts/export_paper_artifacts.py`: lists which paper figures are refreshed or kept.

## Policy

After this migration, new experiments should write outputs under `main/runs/`, `main/outputs/`, or `main/artifacts/`. Paper-facing figures should be regenerated from `main` scripts, not from historical plotting entrypoints.
