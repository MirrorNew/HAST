# -*- coding: utf-8 -*-
"""Export paper figures/tables from completed HAST main outputs."""

PAPER_ARTIFACTS_TO_REFRESH = [
    "Gemini-Framework.png",
    "fig13_12graph_quality_runtime_all_methods.png",
    "fig17_hast_quality_speed_panel.png",
    "fig10_gcc_curves_12graphs.png",
    "fig11_cnbi_curves_12graphs.png",
    "fig20_framework_search_time.png",
    "scaling_full_eval_500_to_10k_unified.png",
    "runtime_only_scaling_500_to_1000k_unified.png",
]

PAPER_ARTIFACTS_TO_KEEP = [
    "fig21_obs1_basic_baseline_same_r_horizontal.png",
]

PAPER_ARTIFACTS_PENDING_REBUILD = [
    "fig22_relative_credit_allocation_effect.png",
    "fig23_bounded_generation_controls_scan_cost.png",
    "fig18_hast_mechanism_compression.png",
    "fig14_component_knockout_ablation.png",
    "fig13_final_candidate_interpretability.png",
    "fig15_step_delta_interpretability.png",
]


def main() -> None:
    print(
        {
            "paper_artifacts_to_refresh": PAPER_ARTIFACTS_TO_REFRESH,
            "paper_artifacts_to_keep": PAPER_ARTIFACTS_TO_KEEP,
            "paper_artifacts_pending_rebuild": PAPER_ARTIFACTS_PENDING_REBUILD,
        }
    )


if __name__ == "__main__":
    main()
