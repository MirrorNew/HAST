# -*- coding: utf-8 -*-
"""Run the full paper-strengthening package with a terminal progress monitor."""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports"
LOG = REPORT / "full_boost_package_monitor.log"


STAGES = [
    ("compile scripts", [sys.executable, "-m", "py_compile",
                         str(ROOT / "src" / "replot_focused_story_figures.py"),
                         str(ROOT / "src" / "run_remark_driven_supplements.py"),
                         str(ROOT / "src" / "run_12graph_curves_and_python_baselines.py"),
                         str(ROOT / "src" / "run_final_candidate_interpretability.py")]),
    ("redraw unified HAST figures", [sys.executable, str(ROOT / "src" / "replot_focused_story_figures.py")]),
    ("remarks supplement package", [sys.executable, str(ROOT / "src" / "run_remark_driven_supplements.py")]),
    ("12-graph curves + python baselines", [sys.executable, str(ROOT / "src" / "run_12graph_curves_and_python_baselines.py")]),
    ("final-candidate interpretability", [sys.executable, str(ROOT / "src" / "run_final_candidate_interpretability.py")]),
]


def run_stage(name: str, cmd: list[str], log_handle) -> None:
    started = time.perf_counter()
    log_handle.write(f"\n\n===== {datetime.now().isoformat(timespec='seconds')} START {name} =====\n")
    log_handle.write("CMD: " + " ".join(cmd) + "\n")
    log_handle.flush()
    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT.parents[1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    with tqdm(desc=name, unit="line", dynamic_ncols=True, leave=False) as pbar:
        for line in process.stdout:
            line = line.rstrip("\n")
            log_handle.write(line + "\n")
            log_handle.flush()
            pbar.write(line)
            pbar.update(1)
    code = process.wait()
    elapsed = time.perf_counter() - started
    log_handle.write(f"===== {datetime.now().isoformat(timespec='seconds')} END {name} code={code} elapsed={elapsed:.2f}s =====\n")
    log_handle.flush()
    if code != 0:
        raise RuntimeError(f"stage failed: {name} code={code}")


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8", newline="") as log_handle:
        log_handle.write(f"\n\n######## full boost package run {datetime.now().isoformat(timespec='seconds')} ########\n")
        with tqdm(total=len(STAGES), desc="full boost package", unit="stage", dynamic_ncols=True) as stage_bar:
            for name, cmd in STAGES:
                stage_bar.set_postfix(stage=name[:28])
                run_stage(name, cmd, log_handle)
                stage_bar.update(1)
        log_handle.write(f"######## completed {datetime.now().isoformat(timespec='seconds')} ########\n")
    print(f"[done] monitor log: {LOG}")


if __name__ == "__main__":
    main()
