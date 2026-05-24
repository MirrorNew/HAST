# -*- coding: utf-8 -*-
"""Live tqdm monitor for a running HAST experiment."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "runs" / "HAST"


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_records(path: Path) -> list[dict[str, str]] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, UnicodeDecodeError, csv.Error):
        return None


def is_pid_alive(pid: int | None) -> bool | None:
    if not pid:
        return None
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return str(pid) in out and "INFO:" not in out


def best_valid(records: list[dict[str, str]]) -> dict[str, str] | None:
    valid = [r for r in records if parse_bool(r.get("valid"))]
    if not valid:
        return None
    return max(valid, key=lambda r: parse_float(r.get("rank_score"), -1.0))


def summarize(records: list[dict[str, str]], total: int) -> dict[str, Any]:
    search_records = [r for r in records if parse_int(r.get("idx")) > 0]
    done = max([parse_int(r.get("idx")) for r in search_records], default=0)
    last = max(search_records, key=lambda r: parse_int(r.get("idx")), default={})
    best = best_valid(records)
    valid_count = sum(parse_bool(r.get("valid")) for r in records)
    strict_count = sum(parse_bool(r.get("strict_e26f_like")) for r in records)
    loose_count = sum(parse_bool(r.get("loose_e26f_like")) for r in records)

    summary = {
        "done": min(done, total),
        "raw_done": done,
        "records": len(records),
        "valid": valid_count,
        "strict": strict_count,
        "loose": loose_count,
        "last_idx": parse_int(last.get("idx")),
        "last_family": last.get("chosen_family") or "-",
        "last_actual": last.get("actual_family") or "-",
        "last_valid": parse_bool(last.get("valid")) if last else False,
        "last_error": (last.get("error") or "").strip(),
        "best_score": parse_float(best.get("rank_score"), -1.0) if best else -1.0,
        "best_r": parse_float(best.get("R")) if best else float("nan"),
        "best_cnbi": parse_float(best.get("cNBI")) if best else float("nan"),
        "best_time": parse_float(best.get("Time")) if best else float("nan"),
        "best_idx": parse_int(best.get("idx")) if best else 0,
        "best_family": best.get("actual_family") if best else "-",
    }
    return summary


def format_age(path: Path) -> str:
    if not path.exists():
        return "no file"
    age_s = max(0.0, time.time() - path.stat().st_mtime)
    if age_s < 60:
        return f"{age_s:.0f}s"
    return f"{age_s / 60:.1f}m"


def build_postfix(summary: dict[str, Any], csv_path: Path, alive: bool | None) -> str:
    alive_text = "alive" if alive is True else "done?" if alive is False else "pid?"
    best = (
        f"best#{summary['best_idx']} score={summary['best_score']:.4f} "
        f"R={summary['best_r']:.6f} cNBI={summary['best_cnbi']:.3f} "
        f"T={summary['best_time']:.4f}"
    )
    return (
        f"{alive_text} | valid={summary['valid']} strict={summary['strict']} "
        f"loose={summary['loose']} | last#{summary['last_idx']} "
        f"{summary['last_family']}->{summary['last_actual']} | {best} | "
        f"csv_age={format_age(csv_path)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor HAST progress with tqdm.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--total", type=int, default=300)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--stay-open", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    csv_path = run_dir / "search_records.csv"
    print("HAST 实时监控")
    print(f"run_dir: {run_dir}")
    print(f"records: {csv_path}")
    print(f"total nodes: {args.total}")
    if args.pid:
        print(f"process pid: {args.pid}")
    print("按 Ctrl+C 可退出监控；不会影响正在跑的实验。")
    print()

    last_done = -1
    last_records: list[dict[str, str]] = []
    last_alive_check = 0.0
    alive: bool | None = None
    with tqdm(total=args.total, dynamic_ncols=True, unit="node", initial=0) as bar:
        while True:
            now = time.time()
            if now - last_alive_check >= 5.0:
                alive = is_pid_alive(args.pid)
                last_alive_check = now

            records = read_records(csv_path)
            if records is None:
                records = last_records
            else:
                last_records = records
            summary = summarize(records, args.total)
            done = summary["done"]
            if done != bar.n:
                bar.n = done
                bar.refresh()

            bar.set_postfix_str(build_postfix(summary, csv_path, alive), refresh=True)

            if done != last_done and summary["last_idx"]:
                status = "valid" if summary["last_valid"] else "invalid"
                msg = (
                    f"{datetime.now().strftime('%H:%M:%S')} idx={summary['last_idx']}/{args.total} "
                    f"{status} family={summary['last_family']} actual={summary['last_actual']} "
                    f"best#{summary['best_idx']} score={summary['best_score']:.4f}"
                )
                if summary["last_error"]:
                    msg += f" error={summary['last_error'][:100]}"
                tqdm.write(msg)
                last_done = done

            finished_by_progress = done >= args.total
            finished_by_process = alive is False and args.pid is not None
            if finished_by_progress or finished_by_process:
                records = read_records(csv_path) or last_records
                summary = summarize(records, args.total)
                bar.n = summary["done"]
                bar.set_postfix_str(build_postfix(summary, csv_path, alive), refresh=True)
                bar.refresh()
                break

            time.sleep(max(0.2, args.interval))

    print()
    print("监控结束。最终快照：")
    final_records = read_records(csv_path) or last_records
    final = summarize(final_records, args.total)
    print(
        f"idx={final['raw_done']}/{args.total}, records={final['records']}, "
        f"valid={final['valid']}, strict={final['strict']}, loose={final['loose']}"
    )
    print(
        f"best#{final['best_idx']} score={final['best_score']:.4f}, "
        f"R={final['best_r']:.6f}, cNBI={final['best_cnbi']:.3f}, "
        f"Time={final['best_time']:.4f}, family={final['best_family']}"
    )
    if args.stay_open:
        input("按 Enter 关闭窗口...")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已退出监控，实验进程不受影响。")
        raise SystemExit(130)
