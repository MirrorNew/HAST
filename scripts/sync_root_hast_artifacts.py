# -*- coding: utf-8 -*-
"""Sync a completed root-relative HAST run into paper source tables.

This updates only HAST-derived rows and keeps all baseline rows intact.
The Q/S labels come from E6 ``stage3_final_selection.json``; E7 is used only
as the 12-graph evaluation of those frozen candidates.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "artifacts" / "source_tables"
BENCHMARK_DIR = SOURCE_ROOT / "benchmark_12graph"
SEARCH_DIR = SOURCE_ROOT / "search_runtime"
FIGURE_DIR = ROOT / "artifacts" / "figures"
PAPER_DOC = ROOT / "docs" / "14_chinese_paper_full_cn.md"

DATASET_ORDER = [
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
]

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
]


def boolish(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "ok"}


def safe_method_filename(name: str) -> str:
    import re

    text = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(name)).strip("_")
    return text or "method"


def backup_targets(tag: str) -> Path:
    backup_dir = ROOT / "artifacts" / "backups" / f"before_root_hast_sync_{tag}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in [BENCHMARK_DIR, SEARCH_DIR]:
        target = backup_dir / path.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(path, target)
    fig_target = backup_dir / "figures"
    fig_target.mkdir(parents=True, exist_ok=True)
    for stem in [
        "fig13_12graph_quality_runtime_all_methods",
        "fig17_hast_quality_speed_panel",
        "fig20_framework_search_time",
        "fig10_gcc_curves_12graphs",
        "fig11_cnbi_curves_12graphs",
    ]:
        for ext in [".png", ".pdf"]:
            src = FIGURE_DIR / f"{stem}{ext}"
            if src.exists():
                shutil.copy2(src, fig_target / src.name)
    return backup_dir


def load_e6_label_map(run_dir: Path) -> dict[str, str]:
    selection = json.loads((run_dir / "stage3_final_selection.json").read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for label, row in selection.items():
        if row:
            out[str(row["candidate_id"])] = label
    if set(out.values()) != {"HAST-Final-Q", "HAST-Final-S"}:
        raise SystemExit(f"Could not recover E6 HAST-Final-Q/S labels from {run_dir / 'stage3_final_selection.json'}")
    return out


def load_e6_label_rows(run_dir: Path) -> dict[str, dict[str, object]]:
    selection = json.loads((run_dir / "stage3_final_selection.json").read_text(encoding="utf-8"))
    out = {label: row for label, row in selection.items() if row}
    if set(out) != {"HAST-Final-Q", "HAST-Final-S"}:
        raise SystemExit(f"Could not recover E6 HAST-Final-Q/S rows from {run_dir / 'stage3_final_selection.json'}")
    return out


def restore_e6_final_dir(run_dir: Path) -> dict[str, object]:
    selection = json.loads((run_dir / "stage3_final_selection.json").read_text(encoding="utf-8"))
    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {}
    for label, row in selection.items():
        if not row:
            manifest[label] = None
            continue
        src = Path(str(row.get("source_code_path") or row["code_path"]))
        dst = final_dir / f"{label}.py"
        shutil.copy2(src, dst)
        manifest[label] = {
            "method": f"{row['source_stage'].replace('stage', 'HAST-S')}-{int(row['stage_index']):04d}-{str(row['candidate_id'])[:8]}",
            "candidate_id": row["candidate_id"],
            "code_path": str(dst),
            "source_code_path": str(src),
            "selection_source": "E6_stage3_final_selection",
        }
    (final_dir / "final_code_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def assert_reference_gate(run_dir: Path) -> dict[str, object]:
    path = run_dir / "full_validation" / "legacy_reference_check.json"
    if not path.exists():
        raise SystemExit(
            f"Missing reference gate report: {path}. "
            "Run E7 through experiments.hast_main_search or hast.reference_check before syncing paper artifacts."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = payload.get("gate", {})
    if not bool(gate.get("paper_refresh_allowed")):
        raise SystemExit(f"Reference gate did not pass; refusing to refresh paper-facing HAST artifacts: {path}")
    return payload


def root_hast_rows(run_dir: Path) -> pd.DataFrame:
    label_map = load_e6_label_map(run_dir)
    e7 = pd.read_csv(run_dir / "full_validation" / "per_graph_metrics.csv", encoding="utf-8-sig")
    if set(["HAST-Final-Q", "HAST-Final-S"]).issubset(set(e7["method"].astype(str))):
        e7 = e7[e7["method"].astype(str).isin(["HAST-Final-Q", "HAST-Final-S"])].copy()
    else:
        e7 = e7[e7["candidate_id"].astype(str).isin(label_map)].copy()
        e7["method"] = e7["candidate_id"].astype(str).map(label_map)
    e7["group"] = "algorithm_found"
    e7["source"] = "hast_root_e6_final_then_e7"
    e7["evidence_tier"] = "root_relative_new_run"
    e7 = e7[e7["valid"].map(boolish)].copy()
    missing = sorted(set(DATASET_ORDER) - set(e7["dataset"].astype(str)))
    labels = sorted(e7["method"].unique())
    if missing or labels != ["HAST-Final-Q", "HAST-Final-S"]:
        raise SystemExit(f"Root HAST E7 rows incomplete: missing={missing}, labels={labels}")
    return e7


def write_csv_with_lock_fallback(frame: pd.DataFrame, path: Path, locked_outputs: list[str]) -> None:
    try:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    except PermissionError:
        fallback = path.with_name(path.stem + ".root_20260525" + path.suffix)
        frame.to_csv(fallback, index=False, encoding="utf-8-sig")
        locked_outputs.append(str(path))


def update_benchmark_tables(run_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    locked_outputs: list[str] = []
    old = pd.read_csv(BENCHMARK_DIR / "per_graph_metrics.csv", encoding="utf-8-sig")
    hast = root_hast_rows(run_dir)
    combined = pd.concat(
        [old[~old["method"].isin(["HAST-Final-Q", "HAST-Final-S"])], hast],
        ignore_index=True,
        sort=False,
    )
    for col in ["R", "auc_cNBI", "time_s"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined["rank_R"] = combined.groupby("dataset")["R"].rank(method="average", ascending=True)
    combined["rank_auc_cNBI"] = combined.groupby("dataset")["auc_cNBI"].rank(method="average", ascending=False)
    combined = combined[SOURCE_COLUMNS].copy()
    write_csv_with_lock_fallback(combined, BENCHMARK_DIR / "per_graph_metrics.csv", locked_outputs)

    for dataset, sub in combined.groupby("dataset", sort=False):
        dataset_dir = BENCHMARK_DIR / str(dataset)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        write_csv_with_lock_fallback(
            sub.sort_values("rank_auc_cNBI"),
            dataset_dir / "method_summary.csv",
            locked_outputs,
        )

    valid = combined.dropna(subset=["R", "auc_cNBI", "time_s"]).copy()
    mean = (
        valid.groupby(["method", "group", "evidence_tier"], as_index=False)
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
    write_csv_with_lock_fallback(mean, BENCHMARK_DIR / "method_mean_metrics.csv", locked_outputs)
    return mean, locked_outputs


def copy_point_evaluations(run_dir: Path) -> None:
    label_rows = load_e6_label_rows(run_dir)
    manifest_rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        dst_dir = BENCHMARK_DIR / dataset / "point_evaluations"
        dst_dir.mkdir(parents=True, exist_ok=True)
        src_dir = run_dir / "full_validation" / "point_evaluations" / dataset
        for label, row in label_rows.items():
            candidate_id = str(row.get("candidate_id", ""))
            label_path = src_dir / f"{safe_method_filename(label)}.csv"
            matches = [label_path] if label_path.exists() else list(src_dir.glob(f"*{candidate_id[:8]}*.csv"))
            if not matches:
                raise SystemExit(f"Missing point evaluation for {label} on {dataset}")
            frame = pd.read_csv(matches[0], encoding="utf-8-sig")
            frame["method"] = label
            frame["group"] = "algorithm_found"
            frame["source"] = "hast_root_e6_final_then_e7"
            frame["evidence_tier"] = "root_relative_new_run"
            out_path = dst_dir / f"{safe_method_filename(label)}.csv"
            frame.to_csv(out_path, index=False, encoding="utf-8-sig")
            manifest_rows.append(
                {
                    "dataset": dataset,
                    "method": label,
                    "rows": int(len(frame)),
                    "path": str(Path(dataset) / "point_evaluations" / out_path.name),
                }
            )
    manifest_path = BENCHMARK_DIR / "point_evaluation_manifest.csv"
    if manifest_path.exists():
        old_manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
        old_manifest = old_manifest[~old_manifest["method"].isin(["HAST-Final-Q", "HAST-Final-S"])].copy()
        manifest = pd.concat([old_manifest, pd.DataFrame(manifest_rows)], ignore_index=True, sort=False)
    else:
        manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")


def update_result_tables(mean: pd.DataFrame) -> None:
    by = mean.set_index("method")
    era_name = "PUCT" if "PUCT" in by.index else "ERA-like"
    era_auc = float(by.loc[era_name, "mean_auc_cNBI"])
    era_time = float(by.loc[era_name, "mean_time_s"])

    main_path = BENCHMARK_DIR / "hast_main_results.csv"
    main = pd.read_csv(main_path, encoding="utf-8-sig")
    replacements = {
        "HAST-Bounded quality": ("HAST-Final-Q", "Root-HAST-Q", "root-relative final quality candidate"),
        "HAST-Bounded speed": ("HAST-Final-S", "Root-HAST-S", "root-relative final speed candidate"),
    }
    for paper_label, (method, internal, role) in replacements.items():
        idx = main["paper_label"].eq(paper_label)
        if not idx.any():
            continue
        row = by.loc[method]
        main.loc[idx, "internal_name"] = internal
        main.loc[idx, "role"] = role
        main.loc[idx, "auc_cNBI"] = float(row["mean_auc_cNBI"])
        main.loc[idx, "R"] = float(row["mean_R"])
        main.loc[idx, "time_s"] = float(row["mean_time_s"])
        main.loc[idx, "group"] = "HAST final"
        main.loc[idx, "retention_vs_era_like"] = float(row["mean_auc_cNBI"]) / era_auc
        main.loc[idx, "speedup_vs_era_like"] = era_time / float(row["mean_time_s"])
        main.loc[idx, "retention_label"] = main.loc[idx, "retention_vs_era_like"].map(lambda x: f"{float(x) * 100:.1f}%")
        main.loc[idx, "speedup_label"] = main.loc[idx, "speedup_vs_era_like"].map(lambda x: f"{float(x):.2f}x")
    main.to_csv(main_path, index=False, encoding="utf-8-sig")

    top_path = BENCHMARK_DIR / "top3_final_vs_frameworks.csv"
    top = pd.read_csv(top_path, encoding="utf-8-sig")
    top = top[~top["paper_label"].astype(str).str.startswith("HAST final")].copy()
    hast_rows = []
    for label, method, internal in [
        ("HAST final top-1 quality", "HAST-Final-Q", "Root-HAST-Q"),
        ("HAST final top-1 speed", "HAST-Final-S", "Root-HAST-S"),
    ]:
        row = by.loc[method]
        hast_rows.append(
            {
                "paper_label": label,
                "role": "HAST root-relative constrained-search output",
                "R": float(row["mean_R"]),
                "auc_cNBI": float(row["mean_auc_cNBI"]),
                "time_s": float(row["mean_time_s"]),
                "internal_name": internal,
                "retention_vs_PUCT": float(row["mean_auc_cNBI"]) / era_auc,
                "speedup_vs_PUCT": era_time / float(row["mean_time_s"]),
            }
        )
    top = pd.concat([top, pd.DataFrame(hast_rows)], ignore_index=True, sort=False)
    top.to_csv(top_path, index=False, encoding="utf-8-sig")


def update_search_runtime(run_dir: Path) -> None:
    search_path = SEARCH_DIR / "framework_search_time_summary.csv"
    search = pd.read_csv(search_path, encoding="utf-8-sig")
    logs = [
        pd.read_csv(run_dir / "stage1_candidate_log.csv", encoding="utf-8-sig"),
        pd.read_csv(run_dir / "stage3_candidate_log.csv", encoding="utf-8-sig"),
    ]
    stage = pd.concat(logs, ignore_index=True, sort=False)
    eval_s = pd.to_numeric(stage["time_s"], errors="coerce").fillna(0.0)
    prompt_s = pd.to_numeric(stage["llm_elapsed_s"], errors="coerce").fillna(0.0)
    row = {
        "method": "HAST-root-tree",
        "paper_label": "HAST bounded search",
        "group": "HAST stage",
        "candidates": int(len(stage)),
        "valid_rate": float(stage["valid"].map(boolish).mean()),
        "mean_eval_s": float(eval_s.mean()),
        "median_eval_s": float(eval_s.median()),
        "total_eval_s": float(eval_s.sum()),
        "mean_prompt_s": float(prompt_s.mean()),
        "median_prompt_s": float(prompt_s.median()),
        "total_prompt_s": float(prompt_s.sum()),
        "mean_logged_search_s_per_candidate": float((eval_s + prompt_s).mean()),
        "total_logged_search_s": float((eval_s + prompt_s).sum()),
    }
    search = search[search["paper_label"].ne("HAST bounded search")].copy()
    search = pd.concat([search, pd.DataFrame([row])], ignore_index=True, sort=False)
    search.to_csv(search_path, index=False, encoding="utf-8-sig")


def refresh_recorded_docs_and_figures(run_dir: Path, python_exe: str) -> dict[str, object]:
    commands = [
        [python_exe, str(ROOT / "scripts" / "sync_recorded_source_tables.py")],
        [python_exe, str(ROOT / "plotting" / "paper_figures.py")],
        [python_exe, str(ROOT / "scripts" / "audit_paper_data_alignment.py")],
    ]
    results: list[dict[str, object]] = []
    for command in commands:
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        results.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-1200:],
                "stderr_tail": completed.stderr[-1200:],
            }
        )
        if completed.returncode != 0:
            raise SystemExit(f"Post-sync command failed: {command}\n{completed.stderr}")
    summary_path = run_dir / "paper_refresh_command_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"commands": results, "summary_path": str(summary_path)}


def write_hast_framework_update_doc(run_dir: Path, mean: pd.DataFrame, reference_gate: dict[str, object] | None) -> Path:
    out_path = ROOT / "docs" / "hast_e4_e6_framework_update.md"
    rows = mean[mean["method"].isin(["HAST-Final-Q", "HAST-Final-S"])].copy()
    lines = [
        "# HAST E4-E6 Framework Update",
        "",
        "This note records the current E4-E6 contract: E6 freezes HAST-Final-Q/S; E7 only evaluates them.",
        "",
        "## Run",
        "",
        f"- run_dir: `{run_dir}`",
        "- search budget: Stage1 300, Stage2 10, Stage3 200",
        "- online proxy: Powerlaw_500 generated proxy profile",
        "- full validation: 12-graph benchmark",
        "",
        "## E6-Frozen Final Candidates",
        "",
        "| method | datasets | mean R | mean auc-cNBI | mean time s |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows.itertuples():
        lines.append(
            f"| {row.method} | {int(row.datasets)} | {float(row.mean_R):.6f} | "
            f"{float(row.mean_auc_cNBI):.6f} | {float(row.mean_time_s):.6f} |"
        )
    if reference_gate is not None:
        gate = reference_gate.get("gate", {})
        lines.extend(
            [
                "",
                "## Reference Gate",
                "",
                f"- paper_refresh_allowed: `{bool(gate.get('paper_refresh_allowed'))}`",
                f"- legacy_q_passed_by_hast_final_q: `{bool(gate.get('legacy_q_passed_by_hast_final_q'))}`",
                f"- legacy_s_passed_by_hast_final_s: `{bool(gate.get('legacy_s_passed_by_hast_final_s'))}`",
            ]
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--skip-reference-gate", action="store_true", help="Bypass the E7 legacy reference gate.")
    parser.add_argument("--skip-post-refresh", action="store_true", help="Skip docs/figure refresh commands after table sync.")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable for post-refresh scripts.")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    reference_gate = None if args.skip_reference_gate else assert_reference_gate(run_dir)
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_targets(tag)
    restored_manifest = restore_e6_final_dir(run_dir)
    mean, locked_outputs = update_benchmark_tables(run_dir)
    copy_point_evaluations(run_dir)
    update_result_tables(mean)
    update_search_runtime(run_dir)
    framework_doc = write_hast_framework_update_doc(run_dir, mean, reference_gate)
    post_refresh = None if args.skip_post_refresh else refresh_recorded_docs_and_figures(run_dir, args.python_exe)

    summary = {
        "run_dir": str(run_dir),
        "backup_dir": str(backup_dir),
        "restored_final_manifest": restored_manifest,
        "updated_hast_method_mean": mean[mean["method"].isin(["HAST-Final-Q", "HAST-Final-S"])].to_dict("records"),
        "reference_gate": reference_gate,
        "framework_update_doc": str(framework_doc),
        "post_refresh": post_refresh,
        "locked_outputs": locked_outputs,
    }
    out_path = run_dir / "artifact_sync_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
