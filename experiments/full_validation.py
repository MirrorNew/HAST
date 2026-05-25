# -*- coding: utf-8 -*-
"""Run E7 full validation for E6-selected HAST-Final-Q/S candidates.

E7 is an evaluator only: it must not choose new HAST-Final-Q/S algorithms or
overwrite the E6 final directory.  The Q/S mapping comes from E6.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics.fragmentation import compute_metrics, summarize_metrics

from hast.candidate import CandidateProgram, compile_candidate, make_program
from hast.config import DATASET_RATES, PROJECT_ROOT, RUNS_ROOT
from hast.data import read_graph
from hast.reference_check import write_reference_check

SOURCE_COLUMNS = [
    "dataset",
    "method",
    "group",
    "source",
    "evidence_tier",
    "nodes",
    "steps",
    "R",
    "auc_ACC",
    "auc_NCC",
    "auc_cNBI",
    "final_GCC",
    "final_cNBI",
    "time_s",
    "rank_R",
    "rank_auc_cNBI",
    "early_GCC",
    "early_NCC",
    "early_cNBI",
]


@dataclass(frozen=True)
class E7ValidationConfig:
    output_dir: Path
    datasets: list[str]
    method_prefix: str = "HAST-S3"
    method_names: list[str] | None = None
    source: str = "hast_e7"
    group: str = "HAST"
    evidence_tier: str = "new_run"
    final_dir: Path | None = None
    candidate_timeout_s: float = 90.0
    selection_source: str = "E6_stage3_proxy_pareto"


def default_e7_config(run_name: str = "manual") -> E7ValidationConfig:
    return E7ValidationConfig(
        output_dir=RUNS_ROOT / "HAST" / "e7" / run_name,
        datasets=[
            "CEnew",
            "Collaboration",
            "condmat",
            "crime",
            "email",
            "Grid",
            "GrQC",
            "hamster",
            "HepPh",
            "PH",
            "Yeast",
            "Powerlaw_500",
        ],
    )


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def load_candidate_file(path: Path, family: str = "HAST-final", source_stage: str | None = None) -> CandidateProgram:
    return make_program(path.read_text(encoding="utf-8"), family=family, source_stage=source_stage or path.stem)


def load_candidate_dir(path: Path, family: str = "HAST-final") -> list[CandidateProgram]:
    return [load_candidate_file(item, family=family, source_stage=item.stem) for item in sorted(path.glob("*.py"))]


def load_e6_final_programs(final_dir: Path, family: str = "HAST-final") -> tuple[list[CandidateProgram], list[str]]:
    labels = ["HAST-Final-Q", "HAST-Final-S"]
    programs: list[CandidateProgram] = []
    method_names: list[str] = []
    missing: list[str] = []
    for label in labels:
        path = final_dir / f"{label}.py"
        if not path.exists():
            missing.append(str(path))
            continue
        programs.append(load_candidate_file(path, family=family, source_stage=label))
        method_names.append(label)
    if missing:
        raise FileNotFoundError("Missing E6 final candidate file(s): " + ", ".join(missing))
    return programs, method_names


def boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def evaluate_program_on_graph_worker(program: CandidateProgram, graph, rate: float, queue) -> None:
    del rate
    try:
        runner = compile_candidate(program)
        t0 = time.perf_counter()
        order = runner(graph.copy())
        elapsed = time.perf_counter() - t0
        queue.put({"valid": True, "error": "", "order": order, "elapsed": elapsed})
    except Exception as exc:
        queue.put({"valid": False, "error": str(exc), "order": None, "elapsed": None})


def evaluate_program_on_graph_with_timeout(
    program: CandidateProgram,
    graph,
    rate: float,
    timeout_s: float,
) -> dict[str, object]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=evaluate_program_on_graph_worker, args=(program, graph, rate, queue))
    process.start()
    deadline = time.perf_counter() + timeout_s
    while process.is_alive() and time.perf_counter() < deadline:
        try:
            result = queue.get_nowait()
        except Empty:
            process.join(0.1)
            continue
        process.join(2)
        if process.is_alive():
            process.terminate()
            process.join(5)
        return result
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        return {
            "valid": False,
            "error": f"candidate evaluation timeout after {timeout_s:.1f}s",
            "order": None,
            "elapsed": timeout_s,
        }
    try:
        result = queue.get_nowait()
        process.join(2)
        return result
    except Empty:
        return {
            "valid": False,
            "error": f"candidate worker exited with code {process.exitcode} without a result",
            "order": None,
            "elapsed": None,
        }


def load_candidate_programs_from_stage_log(stage_log_path: Path) -> list[CandidateProgram]:
    df = pd.read_csv(stage_log_path, encoding="utf-8-sig")
    if "valid" in df:
        df = df[df["valid"].map(boolish)]
    programs: list[CandidateProgram] = []
    for row in df.itertuples():
        code_path = Path(str(row.code_path))
        if not code_path.exists():
            continue
        family = str(getattr(row, "family", "stage3-bounded-local"))
        source_stage = str(getattr(row, "source_stage", "stage3"))
        programs.append(load_candidate_file(code_path, family=family, source_stage=source_stage))
    return programs


def load_programs(args) -> list[CandidateProgram]:
    programs: list[CandidateProgram] = []
    for item in args.candidate:
        programs.append(load_candidate_file(Path(item), family=args.family))
    for item in args.candidate_dir:
        programs.extend(load_candidate_dir(Path(item), family=args.family))
    if args.stage3_log:
        programs.extend(load_candidate_programs_from_stage_log(Path(args.stage3_log)))
    return programs


def write_e7_prepare_manifest(config: E7ValidationConfig, candidate_count: int) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "output_dir": str(config.output_dir),
        "datasets": config.datasets,
        "candidate_count": candidate_count,
        "candidate_timeout_s": config.candidate_timeout_s,
        "selection_source": config.selection_source,
        "e7_role": "evaluation_only_no_reselection",
        "benchmark_source_shape": "benchmark_12graph-compatible point_evaluations/method_summary/per_graph/method_mean outputs",
        "will_call_llm": False,
        "project_root": str(PROJECT_ROOT),
    }
    (config.output_dir / "prepare_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def evaluate_full_validation(config: E7ValidationConfig, programs: Iterable[CandidateProgram]) -> dict[str, object]:
    """Validate E6-frozen HAST candidates and write benchmark-compatible E7 outputs."""
    output_dir = config.output_dir
    point_root = output_dir / "point_evaluations"
    point_root.mkdir(parents=True, exist_ok=True)

    per_graph_rows: list[dict[str, object]] = []
    point_manifest_rows: list[dict[str, object]] = []
    programs = list(programs)
    if config.method_names is not None and len(config.method_names) != len(programs):
        raise ValueError(
            f"method_names length ({len(config.method_names)}) must match program count ({len(programs)})"
        )
    evaluation_manifest = {
        "e7_role": "evaluation_only_no_reselection",
        "selection_source": config.selection_source,
        "candidate_count": len(programs),
        "datasets": config.datasets,
        "method_names": config.method_names,
        "will_write_final_code": False,
        "will_overwrite_e6_final": False,
    }
    (output_dir / "e7_evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for dataset in config.datasets:
        graph = read_graph(dataset)
        rate = DATASET_RATES.get(dataset, 0.30)
        (point_root / dataset).mkdir(parents=True, exist_ok=True)
        for program_index, program in enumerate(programs, start=1):
            if config.method_names is not None:
                method = config.method_names[program_index - 1]
            else:
                method = f"{config.method_prefix}-{program_index:04d}-{program.candidate_id[:8]}"
            evaluation = evaluate_program_on_graph_with_timeout(program, graph, rate, config.candidate_timeout_s)
            order = evaluation["order"]
            elapsed = evaluation["elapsed"]
            if not evaluation["valid"] or order is None or elapsed is None:
                per_graph_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "group": config.group,
                        "source": config.source,
                        "evidence_tier": config.evidence_tier,
                        "nodes": graph.number_of_nodes(),
                        "steps": 0,
                        "R": float("nan"),
                        "auc_ACC": float("nan"),
                        "auc_NCC": float("nan"),
                        "auc_cNBI": float("nan"),
                        "final_GCC": float("nan"),
                        "final_cNBI": float("nan"),
                        "early_GCC": float("nan"),
                        "early_NCC": float("nan"),
                        "early_cNBI": float("nan"),
                        "time_s": config.candidate_timeout_s,
                        "candidate_id": program.candidate_id,
                        "selection_source": config.selection_source,
                        "code_path": "",
                        "valid": False,
                        "error": evaluation["error"],
                    }
                )
                continue
            curve = compute_metrics(graph, order, rate=rate, method_time=float(elapsed))
            summary = summarize_metrics(curve)

            points = curve.copy()
            points.insert(0, "source", config.source)
            points.insert(0, "method", method)
            points.insert(0, "dataset", dataset)
            points["group"] = config.group
            points["evidence_tier"] = config.evidence_tier
            point_path = point_root / dataset / f"{method}.csv"
            points.to_csv(point_path, index=False, encoding="utf-8-sig")
            point_manifest_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "rows": int(len(points)),
                    "path": str(Path("point_evaluations") / dataset / f"{method}.csv"),
                }
            )
            per_graph_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "group": config.group,
                    "source": config.source,
                    "evidence_tier": config.evidence_tier,
                    "nodes": graph.number_of_nodes(),
                    "steps": int(len(curve)),
                    "R": summary["R"],
                    "auc_ACC": summary["auc_ACC"],
                    "auc_NCC": summary["auc_NCC"],
                    "auc_cNBI": summary["auc_cNBI"],
                    "final_GCC": summary["final_ACC"],
                    "final_cNBI": summary["final_cNBI"],
                    "early_GCC": summary["early_GCC"],
                    "early_NCC": summary["early_NCC"],
                    "early_cNBI": summary["early_cNBI"],
                    "time_s": summary["time_s"],
                    "candidate_id": program.candidate_id,
                    "selection_source": config.selection_source,
                    "code_path": "",
                    "valid": True,
                    "error": "",
                }
            )

    per_graph = pd.DataFrame(per_graph_rows)
    if per_graph.empty:
        per_graph.to_csv(output_dir / "per_graph_metrics.csv", index=False, encoding="utf-8-sig")
        return {"per_graph_rows": 0, "e7_role": "evaluation_only_no_reselection"}

    per_graph["rank_R"] = per_graph.groupby("dataset")["R"].rank(method="average", ascending=True)
    per_graph["rank_auc_cNBI"] = per_graph.groupby("dataset")["auc_cNBI"].rank(method="average", ascending=False)
    for dataset, sub in per_graph.groupby("dataset"):
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        sub[SOURCE_COLUMNS].sort_values("rank_auc_cNBI").to_csv(
            dataset_dir / "method_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(point_manifest_rows).to_csv(
        output_dir / "point_evaluation_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    per_graph[SOURCE_COLUMNS + ["candidate_id", "selection_source", "code_path", "valid", "error"]].to_csv(
        output_dir / "per_graph_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ranked = per_graph[per_graph["valid"].astype(bool)].copy()
    if ranked.empty:
        return {"per_graph_rows": len(per_graph), "e7_role": "evaluation_only_no_reselection"}

    method_mean = (
        ranked.groupby(["method", "group", "evidence_tier", "candidate_id"], as_index=False)
        .agg(
            datasets=("dataset", "nunique"),
            mean_R=("R", "mean"),
            mean_auc_cNBI=("auc_cNBI", "mean"),
            mean_time_s=("time_s", "mean"),
            top1_auc=("rank_auc_cNBI", lambda s: int((s <= 1).sum())),
            top3_auc=("rank_auc_cNBI", lambda s: int((s <= 3).sum())),
            mean_rank_auc=("rank_auc_cNBI", "mean"),
        )
        .sort_values("mean_auc_cNBI", ascending=False)
    )
    method_mean.to_csv(output_dir / "method_mean_metrics.csv", index=False, encoding="utf-8-sig")

    return {
        "output_dir": str(output_dir),
        "per_graph_rows": len(per_graph),
        "method_rows": len(method_mean),
        "e7_role": "evaluation_only_no_reselection",
        "selection_source": config.selection_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", default=[], help="Python file containing def degree_order(G).")
    parser.add_argument("--candidate-dir", action="append", default=[], help="Directory of Stage 3 candidate .py files.")
    parser.add_argument("--stage3-log", default="", help="Stage 3 candidate log CSV from E4-E6.")
    parser.add_argument("--e6-final-dir", default="", help="Directory containing E6-frozen HAST-Final-Q/S.py.")
    parser.add_argument("--datasets", default="", help="Comma-separated full-validation datasets.")
    parser.add_argument("--run-name", default="manual")
    parser.add_argument("--out-dir", default="", help="Output directory for benchmark-compatible E7 tables.")
    parser.add_argument("--final-dir", default="", help="Deprecated: E7 no longer exports or overwrites final code.")
    parser.add_argument("--family", default="HAST-final")
    parser.add_argument("--candidate-timeout-s", type=float, default=None)
    parser.add_argument(
        "--method-name",
        action="append",
        default=[],
        help="Explicit method name for each candidate, in candidate order. Useful for missing-only reruns.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Write manifest without evaluating candidates.")
    args = parser.parse_args()

    base = default_e7_config(args.run_name)
    datasets = parse_csv_list(args.datasets) if args.datasets else base.datasets
    output_dir = Path(args.out_dir) if args.out_dir else base.output_dir
    final_dir = Path(args.final_dir) if args.final_dir else None
    candidate_timeout_s = args.candidate_timeout_s if args.candidate_timeout_s is not None else base.candidate_timeout_s
    config = E7ValidationConfig(
        output_dir=output_dir,
        datasets=datasets,
        method_names=args.method_name or None,
        final_dir=final_dir,
        candidate_timeout_s=candidate_timeout_s,
    )
    if args.e6_final_dir:
        programs, method_names = load_e6_final_programs(Path(args.e6_final_dir), family=args.family)
        config = E7ValidationConfig(**{**config.__dict__, "method_names": method_names})
    else:
        programs = load_programs(args)

    if args.prepare_only:
        manifest = write_e7_prepare_manifest(config, len(programs))
        print(json.dumps({"prepared": True, **manifest}, ensure_ascii=False, indent=2))
        return

    if not programs:
        raise SystemExit("No candidates provided. Use --candidate, --candidate-dir, or --stage3-log.")

    result = evaluate_full_validation(config, programs)
    if config.method_names == ["HAST-Final-Q", "HAST-Final-S"]:
        result["legacy_reference_check"] = write_reference_check(config.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
