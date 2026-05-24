# Paper Figure Mapping

This file maps figures referenced by `HAST2026/01_paper_story/14_chinese_paper_full_cn.md` to the `main/` project refresh policy.

| Paper figure file | Exists in `02_main_figures` | `main/` policy | Reason |
|---|---:|---|---|
| `fig21_obs1_basic_baseline_same_r_horizontal.png` | yes | keep | Observation 1 basic-baseline evidence; independent of HAST three-stage rerun |
| `fig22_relative_credit_allocation_effect.png` | yes | refresh | Observation 2 depends on new relative-credit experiment |
| `fig23_bounded_generation_controls_scan_cost.png` | yes | refresh | Observation 3 depends on new bounded-guided experiment |
| `Gemini-Framework.png` | yes | refresh | Framework figure must reflect 300 + 10 + 200 budget |
| `fig13_12graph_quality_runtime_all_methods.png` | yes | refresh | Main 12-graph HAST result |
| `fig17_hast_quality_speed_panel.png` | yes | refresh | HAST quality-speed focused panel in Section 5.2.2 |
| `fig10_gcc_curves_12graphs.png` | yes | refresh | 12-graph GCC curves for final candidates |
| `fig11_cnbi_curves_12graphs.png` | yes | refresh | 12-graph cNBI curves for final candidates |
| `fig20_framework_search_time.png` | yes | refresh | Search/runtime cost comparison |
| `fig18_hast_mechanism_compression.png` | yes | refresh | Mechanism compression after new Stage 2/3 design |
| `fig14_component_knockout_ablation.png` | yes | refresh | Ablation/knockout rerun |
| `fig13_final_candidate_interpretability.png` | yes | refresh | Final candidate interpretation after new search |
| `scaling_full_eval_500_to_10k_unified.png` | yes | refresh | Full-eval scaling rerun |
| `runtime_only_scaling_500_to_1000k_unified.png` | yes | refresh | Runtime-only extreme scaling rerun |

Current status: all paper-referenced figure files exist in `02_main_figures`; the `main/` refresh list now covers every HAST-dependent figure, while Observation 1 is explicitly marked as kept.
