# -*- coding: utf-8 -*-
"""Live tqdm monitor for HAST-FAC full 12-graph evaluation."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "tables"
DEFAULT_DETAIL = TABLE_DIR / "hast_fac_online_full12_detail.csv"
DEFAULT_MEAN = TABLE_DIR / "hast_fac_online_full12_mean.csv"


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv_rows(path: Path) -> list[dict[str, str]] | None:
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


def format_age(path: Path) -> str:
    if not path.exists():
        return "not yet"
    age_s = max(0.0, time.time() - path.stat().st_mtime)
    if age_s < 60:
        return f"{age_s:.0f}s"
    return f"{age_s / 60:.1f}m"


def summarize(detail_path: Path, mean_path: Path, total: int) -> dict[str, Any]:
    detail = read_csv_rows(detail_path)
    mean = read_csv_rows(mean_path)
    if mean:
        done = min(total, len({row.get("candidate_idx") for row in mean if row.get("candidate_idx")}))
        return {"done": done, "detail_rows": len(detail or []), "mean_rows": len(mean), "finished": True}
    if detail:
        done = min(total, len({row.get("candidate_idx") for row in detail if row.get("candidate_idx")}))
        return {"done": done, "detail_rows": len(detail), "mean_rows": 0, "finished": done >= total}
    return {"done": 0, "detail_rows": 0, "mean_rows": 0, "finished": False}


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor HAST-FAC full-12 evaluation with tqdm.")
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--mean", type=Path, default=DEFAULT_MEAN)
    parser.add_argument("--total", type=int, default=5)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--after-epoch", type=float, default=0.0, help="Ignore result files older than this Unix timestamp.")
    parser.add_argument("--stay-open", action="store_true")
    args = parser.parse_args()

    detail_path = args.detail.resolve()
    mean_path = args.mean.resolve()
    print("HAST-FAC full-12 实时监控")
    print(f"detail: {detail_path}")
    print(f"mean:   {mean_path}")
    print(f"top-k candidates: {args.total}")
    if args.pid:
        print(f"process pid: {args.pid}")
    print("说明：当前评估脚本会在 full-12 全部跑完后统一写 CSV；因此中途主要显示进程存活和输出文件状态。")
    print("按 Ctrl+C 可退出监控；不会影响正在跑的评估。")
    print()

    started = time.time()
    alive: bool | None = None
    last_alive_check = 0.0
    last_done = -1
    with tqdm(total=args.total, dynamic_ncols=True, unit="cand", initial=0) as bar:
        while True:
            now = time.time()
            if now - last_alive_check >= 5.0:
                alive = is_pid_alive(args.pid)
                last_alive_check = now
            active_detail = detail_path if detail_path.exists() and detail_path.stat().st_mtime >= args.after_epoch else Path("__not_ready_detail__")
            active_mean = mean_path if mean_path.exists() and mean_path.stat().st_mtime >= args.after_epoch else Path("__not_ready_mean__")
            summary = summarize(active_detail, active_mean, args.total)
            if summary["done"] != bar.n:
                bar.n = summary["done"]
                bar.refresh()
            alive_text = "alive" if alive is True else "done?" if alive is False else "pid?"
            elapsed_m = (now - started) / 60.0
            bar.set_postfix_str(
                f"{alive_text} | elapsed={elapsed_m:.1f}m | "
                f"detail_rows={summary['detail_rows']} mean_rows={summary['mean_rows']} | "
                f"detail_age={format_age(detail_path)} mean_age={format_age(mean_path)}",
                refresh=True,
            )
            if summary["done"] != last_done:
                tqdm.write(
                    f"{datetime.now().strftime('%H:%M:%S')} full12 progress "
                    f"{summary['done']}/{args.total}; detail_rows={summary['detail_rows']}; "
                    f"mean_rows={summary['mean_rows']}"
                )
                last_done = summary["done"]
            if summary["finished"] or (alive is False and args.pid is not None):
                break
            time.sleep(max(0.5, args.interval))

    active_detail = detail_path if detail_path.exists() and detail_path.stat().st_mtime >= args.after_epoch else Path("__not_ready_detail__")
    active_mean = mean_path if mean_path.exists() and mean_path.stat().st_mtime >= args.after_epoch else Path("__not_ready_mean__")
    final = summarize(active_detail, active_mean, args.total)
    print()
    print("监控结束。最终快照：")
    print(f"progress={final['done']}/{args.total}, detail_rows={final['detail_rows']}, mean_rows={final['mean_rows']}")
    if args.stay_open:
        input("按 Enter 关闭窗口...")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已退出监控，评估进程不受影响。")
        raise SystemExit(130)
