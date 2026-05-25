# -*- coding: utf-8 -*-
"""Run HAST E4-E6 search, optionally followed by E7."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hast.e4_e6 import default_config, read_benchmark_context, run_e4_e6, write_prepare_manifest
from experiments.full_validation import E7ValidationConfig, evaluate_full_validation, load_e6_final_programs
from hast.reference_check import write_reference_check
from hast.llm import OpenAICompatibleLLMProvider


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def write_input_parameters(config, args: argparse.Namespace, run_date: str) -> Path:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "argv": sys.argv,
        "cli_args": vars(args),
        "resolved": {
            "run_dir": str(config.run_dir),
            "run_date": run_date,
            "delta_credit_mode": config.delta_credit_mode,
            "proxy_datasets": config.proxy_datasets,
            "full_datasets": config.full_datasets,
            "stage1_budget": config.stage1_budget,
            "stage2_budget": config.stage2_budget,
            "stage3_budget": config.stage3_budget,
            "candidates_per_llm_call": config.candidates_per_llm_call,
            "stage3_parent_limit": config.stage3_parent_limit,
            "candidate_timeout_s": config.candidate_timeout_s,
            "llm_workers": config.llm_workers,
            "execute": bool(args.execute),
            "run_e7": bool(args.run_e7),
            "preset": args.preset,
            "proxy_profile": args.proxy_profile,
            "e7_role": "evaluation_only_no_reselection",
            "e6_final_selection_source": "stage3_final_selection.json",
        },
        "llm_env": {
            "api_key_source": "HAST_LLM_API_KEY or OPENAI_API_KEY environment variable",
            "model_env": "HAST_LLM_MODEL",
            "reasoning_effort_env": "HAST_LLM_REASONING_EFFORT",
            "temperature_env": "HAST_LLM_TEMPERATURE",
            "base_url_env": "HAST_LLM_BASE_URL",
            "timeout_env": "HAST_LLM_TIMEOUT_S",
            "api_key_saved": False,
        },
    }
    path = config.run_dir / "input_parameters.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="main", help="Experiment slug used in runs/HAST_<parent|root>_<run-name>_<YYYYMMDD>.")
    parser.add_argument("--run-date", default="", help="Override YYYYMMDD suffix for reproducible run directory naming.")
    parser.add_argument("--preset", choices=["default", "target-family"], default="default")
    parser.add_argument(
        "--proxy-profile",
        choices=["default", "xxx-like-powerlaw500"],
        default="default",
        help="Named online proxy profile. xxx-like-powerlaw500 keeps search comparable with other xxx-like runs.",
    )
    parser.add_argument("--proxy-datasets", default="", help="Comma-separated proxy datasets for Stage 1/3 search.")
    parser.add_argument("--full-datasets", default="", help="Comma-separated datasets for optional E7 full validation.")
    parser.add_argument("--stage1-budget", type=int, default=None)
    parser.add_argument("--stage2-budget", type=int, default=None)
    parser.add_argument("--stage3-budget", type=int, default=None)
    parser.add_argument("--candidate-timeout-s", type=float, default=None)
    parser.add_argument(
        "--delta-credit-mode",
        choices=["parent", "root"],
        default=None,
        help="Use parent-relative credit for main HAST, or root-relative credit for e7_additional.",
    )
    parser.add_argument("--llm-workers", type=int, default=None, help="Concurrent LLM requests.")
    parser.add_argument("--allow-existing-run-dir", action="store_true", help="Allow writing into an existing run directory.")
    parser.add_argument("--execute", action="store_true", help="Actually call the LLM provider and run E4-E6.")
    parser.add_argument("--dry-run", action="store_true", help="Alias for the default prepare-only behavior.")
    parser.add_argument("--run-e7", action="store_true", help="After E4-E6, run E7 on the E6-selected HAST-Final-Q/S.")
    args = parser.parse_args()

    delta_credit_mode = args.delta_credit_mode or ("root" if args.preset == "target-family" else "parent")
    run_date = args.run_date or datetime.now().strftime("%Y%m%d")
    config = default_config(args.run_name, delta_credit_mode=delta_credit_mode, run_date=run_date)
    if args.preset == "target-family":
        config = config.__class__(
            **{
                **config.__dict__,
                "stage1_budget": 300,
                "stage2_budget": 10,
                "stage3_budget": 200,
                "proxy_datasets": ["Powerlaw_500"],
            }
        )
    if args.proxy_profile == "xxx-like-powerlaw500":
        config = config.__class__(**{**config.__dict__, "proxy_datasets": ["Powerlaw_500"]})
    if args.proxy_datasets:
        config = config.__class__(**{**config.__dict__, "proxy_datasets": parse_csv_list(args.proxy_datasets)})
    if args.full_datasets:
        config = config.__class__(**{**config.__dict__, "full_datasets": parse_csv_list(args.full_datasets)})
    if (
        args.stage1_budget is not None
        or args.stage2_budget is not None
        or args.stage3_budget is not None
        or args.candidate_timeout_s is not None
        or args.llm_workers is not None
    ):
        config = config.__class__(
            run_dir=config.run_dir,
            proxy_datasets=config.proxy_datasets,
            full_datasets=config.full_datasets,
            stage1_budget=args.stage1_budget if args.stage1_budget is not None else config.stage1_budget,
            stage2_budget=args.stage2_budget if args.stage2_budget is not None else config.stage2_budget,
            stage3_budget=args.stage3_budget if args.stage3_budget is not None else config.stage3_budget,
            candidates_per_llm_call=config.candidates_per_llm_call,
            stage3_parent_limit=config.stage3_parent_limit,
            candidate_timeout_s=args.candidate_timeout_s if args.candidate_timeout_s is not None else config.candidate_timeout_s,
            delta_credit_mode=config.delta_credit_mode,
            llm_workers=args.llm_workers if args.llm_workers is not None else config.llm_workers,
        )

    context = read_benchmark_context()
    if args.dry_run and args.execute:
        raise SystemExit("Choose either --dry-run or --execute, not both.")
    if config.run_dir.exists() and any(config.run_dir.iterdir()) and not args.allow_existing_run_dir:
        raise SystemExit(
            f"Run directory already exists and is not empty: {config.run_dir}. "
            "Use a new --run-name/--run-date or pass --allow-existing-run-dir intentionally."
        )
    input_parameters_path = write_input_parameters(config, args, run_date)

    if not args.execute:
        manifest = write_prepare_manifest(config, context)
        manifest["input_parameters_path"] = str(input_parameters_path)
        manifest["preset"] = args.preset
        manifest["proxy_profile"] = args.proxy_profile
        manifest["can_run_e7_with"] = "python scripts/run_e7_full_validation.py --e6-final-dir <run_dir>/final"
        print(json.dumps({"prepared": True, **manifest}, ensure_ascii=False, indent=2))
        return

    provider = OpenAICompatibleLLMProvider.from_env()
    result = run_e4_e6(config, provider)
    result["input_parameters_path"] = str(input_parameters_path)
    if args.run_e7:
        programs, method_names = load_e6_final_programs(config.run_dir / "final", family="HAST-final")
        e7_config = E7ValidationConfig(
            output_dir=config.run_dir / "full_validation",
            datasets=config.full_datasets,
            source="hast_e6_final_then_e7",
            method_names=method_names,
            candidate_timeout_s=config.candidate_timeout_s,
        )
        result["e7_full_validation"] = evaluate_full_validation(e7_config, programs)
        result["legacy_reference_check"] = write_reference_check(config.run_dir / "full_validation")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
