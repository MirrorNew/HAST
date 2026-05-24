# -*- coding: utf-8 -*-
"""Run full validation for baselines and optional HAST candidate files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from baselines.algorithms import corehd_fast_order, dc_order, hda_fast_order, hda_original_order
from hast.candidate import make_program
from hast.config import DATASET_RATES, DATASETS
from hast.data import read_graph
from hast.search1_3 import evaluate_candidate, evaluate_order_fn


BASELINES = {
    "HDA-original": hda_original_order,
    "HDA-fast": hda_fast_order,
    "CoreHD-fast": corehd_fast_order,
    "DC": dc_order,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", default=[], help="Python file containing def degree_order(G).")
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "full_validation_summary.csv"))
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        graph = read_graph(dataset)
        rate = DATASET_RATES.get(dataset, 0.30)
        for method, fn in BASELINES.items():
            summary = evaluate_order_fn(lambda g, f=fn, r=rate: f(g, r), [graph], rate)
            rows.append({"dataset": dataset, "method": method, "source": "baseline", **summary})
        for path_text in args.candidate:
            path = Path(path_text)
            program = make_program(path.read_text(encoding="utf-8"), family="HAST-final", source_stage=path.stem)
            record = evaluate_candidate(program, [graph], rate)
            item = record.__dict__.copy()
            item["dataset"] = dataset
            item["method"] = path.stem
            item["source"] = "candidate_file"
            rows.append(item)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(json.dumps({"out": str(out), "rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
