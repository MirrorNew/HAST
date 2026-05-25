# -*- coding: utf-8 -*-
"""Live tqdm monitor for one HAST E4-E7 run directory."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal runtimes
    tqdm = None


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def process_running(pid: int | None) -> bool:
    if not pid:
        return True
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object -First 1"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return bool(result.stdout.strip())


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def stage_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    valid = sum(1 for row in rows if truthy(row.get("valid", "")))
    timeout = sum(1 for row in rows if "timeout" in str(row.get("error", "")).lower())
    latest = rows[-1] if rows else {}
    return {
        "rows": len(rows),
        "valid": valid,
        "timeout": timeout,
        "latest": latest.get("node_id") or latest.get("stage_index") or "-",
        "depth": latest.get("depth", "-"),
        "score": latest.get("rank_score", "-"),
    }


def set_bar(bar: Any, value: int, postfix: dict[str, Any]) -> None:
    if tqdm is None:
        return
    bar.n = min(value, bar.total or value)
    bar.set_postfix(postfix, refresh=False)
    bar.refresh()


def print_fallback(run_dir: Path, values: dict[str, Any]) -> None:
    print(
        f"{run_dir.name} | "
        f"stage1 {values['stage1']}/{values['stage1_total']} | "
        f"stage2 {values['stage2']}/{values['stage2_total']} | "
        f"stage3 {values['stage3']}/{values['stage3_total']} | "
        f"eval_all {values['eval_all']}/{values['eval_total']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--refresh-s", type=float, default=2.0)
    parser.add_argument("--hold", action="store_true", help="Keep the window open after completion.")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    params = load_json(run_dir / "input_parameters.json")
    resolved = params.get("resolved", {})
    stage1_total = int(resolved.get("stage1_budget", 300))
    stage2_total = int(resolved.get("stage2_budget", 10))
    stage3_total = int(resolved.get("stage3_budget", 200))
    full_datasets = resolved.get("full_datasets") or []
    eval_total = max(1, len(full_datasets) * 2)

    print(f"HAST monitor: {run_dir}")
    print(f"PID: {args.pid or 'unknown'}")
    print(f"Credit mode: {resolved.get('delta_credit_mode', '-')}, timeout: {resolved.get('candidate_timeout_s', '-')}s")
    print("Press Ctrl+C to close this monitor; it does not affect the experiment.\n")

    if tqdm is None:
        while True:
            stage1 = len(read_csv_rows(run_dir / "stage1_candidate_log.csv"))
            stage2 = len(list((run_dir / "raw_llm" / "stage2").glob("*.txt"))) if (run_dir / "raw_llm" / "stage2").exists() else 0
            stage3 = len(read_csv_rows(run_dir / "stage3_candidate_log.csv"))
            eval_rows = len(read_csv_rows(run_dir / "full_validation" / "per_graph_metrics.csv"))
            print_fallback(
                run_dir,
                {
                    "stage1": stage1,
                    "stage1_total": stage1_total,
                    "stage2": stage2,
                    "stage2_total": stage2_total,
                    "stage3": stage3,
                    "stage3_total": stage3_total,
                    "eval_all": eval_rows,
                    "eval_total": eval_total,
                },
            )
            if not process_running(args.pid) and eval_rows >= eval_total:
                break
            time.sleep(args.refresh_s)
        return

    bars = [
        tqdm(total=stage1_total, desc="stage1 tree", position=0, dynamic_ncols=True),
        tqdm(total=stage2_total, desc="stage2 bounds", position=1, dynamic_ncols=True),
        tqdm(total=stage3_total, desc="stage3 tree", position=2, dynamic_ncols=True),
        tqdm(total=eval_total, desc="eval_all E7", position=3, dynamic_ncols=True),
    ]
    last_phase = ""
    try:
        while True:
            stage1_rows = read_csv_rows(run_dir / "stage1_candidate_log.csv")
            stage3_rows = read_csv_rows(run_dir / "stage3_candidate_log.csv")
            e7_rows = read_csv_rows(run_dir / "full_validation" / "per_graph_metrics.csv")
            final_manifest = load_json(run_dir / "final" / "final_code_manifest.json")
            if final_manifest:
                final_count = sum(1 for value in final_manifest.values() if value)
                eval_total = max(1, len(full_datasets) * max(1, final_count))
                bars[3].total = eval_total

            stage2_raw_dir = run_dir / "raw_llm" / "stage2"
            stage2_count = len(list(stage2_raw_dir.glob("*.txt"))) if stage2_raw_dir.exists() else 0
            if (run_dir / "stage2" / "family_policy.json").exists():
                stage2_count = max(stage2_count, stage2_total)

            s1 = stage_stats(stage1_rows)
            s3 = stage_stats(stage3_rows)
            set_bar(bars[0], s1["rows"], {"valid": s1["valid"], "to": s1["timeout"], "last": s1["latest"], "d": s1["depth"]})
            set_bar(bars[1], stage2_count, {"policy": (run_dir / "stage2" / "family_policy.json").exists()})
            set_bar(bars[2], s3["rows"], {"valid": s3["valid"], "to": s3["timeout"], "last": s3["latest"], "d": s3["depth"]})
            set_bar(bars[3], len(e7_rows), {"final": bool(final_manifest), "mean": (run_dir / "full_validation" / "method_mean_metrics.csv").exists()})

            if s1["rows"] < stage1_total:
                phase = "stage1"
            elif stage2_count < stage2_total:
                phase = "stage2"
            elif s3["rows"] < stage3_total:
                phase = "stage3"
            elif len(e7_rows) < eval_total:
                phase = "eval_all"
            else:
                phase = "done"
            if phase != last_phase:
                tqdm.write(f"[phase] {phase}")
                last_phase = phase

            stderr_path = run_dir / "logs" / "stderr.log"
            if stderr_path.exists() and stderr_path.stat().st_size:
                tail = stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[-3:]
                if tail:
                    bars[3].set_postfix_str("stderr: " + " | ".join(line[:80] for line in tail), refresh=False)

            if phase == "done" or (not process_running(args.pid) and phase != "done"):
                if not process_running(args.pid) and phase != "done":
                    tqdm.write("[monitor] experiment process is no longer running before all bars completed; check logs/stderr.log")
                break
            time.sleep(args.refresh_s)
    except KeyboardInterrupt:
        tqdm.write("[monitor] closed by user")
    finally:
        for bar in bars:
            bar.close()
        if args.hold:
            input("Monitor ended. Press Enter to close...")


if __name__ == "__main__":
    main()
