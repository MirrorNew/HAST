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
    "search_frameworks/historical/tree_search_ablation_20260520/src",
    "search_frameworks/historical/hast_experiment_20260521/src",
    "search_frameworks/historical/paper_problem_solution_reframe_20260522/src",
    "search_frameworks/historical/iclr_minimal_boost_20260522/src",
    "data/fixtures",
    "data/search_framework_records/raw/tree_search_ablation_20260520",
    "data/search_framework_records/raw/hast_experiment_20260521",
    "artifacts/source_tables/historical_search_frameworks/tree_search_ablation_20260520",
    "artifacts/source_tables/historical_search_frameworks/hast_experiment_20260521",
    "artifacts/source_tables/historical_search_frameworks/iclr_minimal_boost_20260522",
    "artifacts/source_tables/historical_search_frameworks/paper_problem_solution_reframe_20260522_tables",
]

REQUIRED_FILES = [
    "README.md",
    ".gitignore",
    "docs/data_source_index.md",
    "docs/search_framework_data_index.md",
    "search_frameworks/README.md",
    "data/README.md",
    "data/search_framework_records/README.md",
    "configs/hast_lite_full.yaml",
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
        "historical_code_file_count": count_files(ROOT / "search_frameworks" / "historical"),
        "historical_raw_file_count": count_files(ROOT / "data" / "search_framework_records" / "raw"),
        "historical_source_table_file_count": count_files(
            ROOT / "artifacts" / "source_tables" / "historical_search_frameworks"
        ),
        "ok": not missing_dirs and not missing_files,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
