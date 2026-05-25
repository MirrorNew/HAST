"""Audit that the standalone main workspace has the expected code and data mirrors."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_DIRS = [
    "hast",
    "baselines",
    "metrics",
    "configs",
    "scripts",
    "docs",
    "search_frameworks",
    "experiments",
    "plotting",
    "network",
]

REQUIRED_FILES = [
    "README.md",
    ".gitignore",
    "search_frameworks/README.md",
    "search_frameworks/__init__.py",
    "search_frameworks/era_like.py",
    "search_frameworks/mcts_ahd_like.py",
    "search_frameworks/clade_ahd_like.py",
    "search_frameworks/funsearch_like.py",
    "search_frameworks/alphaevolve_like.py",
    "search_frameworks/generic_llm_search_ablation.py",
    "search_frameworks/dacts_style_search.py",
    "search_frameworks/hast_legacy_search.py",
    "search_frameworks/hast_fac_online_search.py",
    "experiments/README.md",
    "experiments/hast_main_search.py",
    "experiments/full_validation.py",
    "experiments/paper_source_tables.py",
    "experiments/motivation_observation_contract.py",
    "experiments/scaling_contract.py",
    "hast/e4_e6.py",
    "plotting/README.md",
    "plotting/paper_figures.py",
    "configs/hast.yaml",
    "scripts/run_e7_full_validation.py",
]


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def main() -> None:
    missing_dirs = [path for path in REQUIRED_DIRS if not (ROOT / path).is_dir()]
    missing_files = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]

    result = {
        "root": str(ROOT),
        "missing_dirs": missing_dirs,
        "missing_files": missing_files,
        "search_framework_file_count": count_files(ROOT / "search_frameworks"),
        "experiment_file_count": count_files(ROOT / "experiments"),
        "plotting_file_count": count_files(ROOT / "plotting"),
        "network_file_count": count_files(ROOT / "network"),
        "source_table_file_count": count_files(ROOT / "artifacts" / "source_tables"),
        "ok": not missing_dirs and not missing_files,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
