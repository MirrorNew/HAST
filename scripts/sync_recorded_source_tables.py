# -*- coding: utf-8 -*-
"""Synchronize paper-facing summary tables to record-derived experiment CSVs."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

PAPER = ROOT / "docs" / "14_chinese_paper_full_cn.md"
MAIN_ARTIFACTS = ROOT / "artifacts"
SOURCE_TABLE_DIR = MAIN_ARTIFACTS / "source_tables"
BENCHMARK_TABLE_DIR = SOURCE_TABLE_DIR / "benchmark_12graph"


METHOD_MAP = {
    "HDA root": "HDA",
    "HDA": "HDA",
    "CoreHD": "CoreHD",
    "ERA-like": "ERA-like",
    "FunSearch-like": "FunSearch-like",
    "Clade-AHD-like": "Clade-AHD-like",
    "MCTS-AHD-like": "MCTS-AHD-like",
    "AlphaEvolve-like": "AlphaEvolve-like",
    "HAST-Bounded quality": "HAST-Final-Q",
    "HAST-Bounded speed": "HAST-Final-S",
    "HAST-Final-Q": "HAST-Final-Q",
    "HAST-Final-S": "HAST-Final-S",
    "NCDC": "NCDC",
    "BPD/MinSum-fallback": "BPD/MinSum-fallback",
    "NDC": "NDC",
    "NDJC": "NDJC",
    "DC": "DC",
    "CI": "CI",
    "KCore": "KCore",
    "CLUC": "CLUC",
}


def load_method_mean() -> pd.DataFrame:
    df = pd.read_csv(BENCHMARK_TABLE_DIR / "method_mean_metrics.csv", encoding="utf-8-sig")
    df["method"] = df["method"].replace({"PUCT": "ERA-like"})
    return df.set_index("method")


def sync_main_results(method_mean: pd.DataFrame) -> None:
    path = BENCHMARK_TABLE_DIR / "hast_main_results.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    keep_cols = ["paper_label", "internal_name", "role", "auc_cNBI", "R", "time_s", "group", "main_table", "paper_label_display"]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[keep_cols].copy()
    for idx, row in df.iterrows():
        source = METHOD_MAP.get(str(row["paper_label_display"])) or METHOD_MAP.get(str(row["paper_label"])) or METHOD_MAP.get(str(row["internal_name"]))
        if source in method_mean.index:
            src = method_mean.loc[source]
            df.loc[idx, "auc_cNBI"] = float(src["mean_auc_cNBI"])
            df.loc[idx, "R"] = float(src["mean_R"])
            df.loc[idx, "time_s"] = float(src["mean_time_s"])
            if source == "ERA-like":
                df.loc[idx, "paper_label"] = "ERA-like"
                df.loc[idx, "paper_label_display"] = "ERA-like"
                df.loc[idx, "internal_name"] = "ERA-like"
    era = df[df["paper_label_display"].eq("ERA-like")].iloc[0]
    df["retention_vs_era_like"] = df["auc_cNBI"].astype(float) / float(era["auc_cNBI"])
    df["speedup_vs_era_like"] = float(era["time_s"]) / df["time_s"].astype(float)
    df["retention_label"] = df["retention_vs_era_like"].map(lambda x: f"{100*x:.1f}%")
    df["speedup_label"] = df["speedup_vs_era_like"].map(lambda x: f"{x:.2f}x")
    df.to_csv(path, index=False, encoding="utf-8-sig")


def sync_top3_table(method_mean: pd.DataFrame) -> None:
    path = BENCHMARK_TABLE_DIR / "top3_final_vs_frameworks.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, encoding="utf-8-sig")
    for idx, row in df.iterrows():
        source = METHOD_MAP.get(str(row["paper_label"])) or METHOD_MAP.get(str(row["internal_name"]))
        if source in method_mean.index:
            src = method_mean.loc[source]
            df.loc[idx, "auc_cNBI"] = float(src["mean_auc_cNBI"])
            df.loc[idx, "R"] = float(src["mean_R"])
            df.loc[idx, "time_s"] = float(src["mean_time_s"])
    ref = df[df["paper_label"].isin(["ERA-like", "ERA-like"])]
    if not ref.empty:
        puct = ref.iloc[0]
        df["retention_vs_PUCT"] = df["auc_cNBI"].astype(float) / float(puct["auc_cNBI"])
        df["speedup_vs_PUCT"] = float(puct["time_s"]) / df["time_s"].astype(float)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def render_main_table(method_mean: pd.DataFrame) -> str:
    order = [
        "FunSearch-like",
        "Clade-AHD-like",
        "ERA-like",
        "HAST-Final-Q",
        "HAST-Final-S",
        "HDA",
        "CoreHD",
        "NCDC",
        "BPD/MinSum-fallback",
    ]
    display = {"ERA-like": "ERA-like"}
    lines = [
        "| method | datasets | mean R ↓ | mean auc-cNBI ↑ | mean time ↓ | top1 auc | top3 auc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in order:
        row = method_mean.loc[method]
        label = display.get(method, method)
        if label in {"HAST-Final-Q", "HAST-Final-S"}:
            label = f"**{label}**"
            time = f"**{float(row['mean_time_s']):.3f}s**"
        else:
            time = f"{float(row['mean_time_s']):.3f}s"
        lines.append(
            f"| {label} | {int(row['datasets'])} | {float(row['mean_R']):.3f} | "
            f"{float(row['mean_auc_cNBI']):.3f} | {time} | {int(row['top1_auc'])} | {int(row['top3_auc'])} |"
        )
    return "\n".join(lines)


def replace_table(text: str, header_regex: str, new_table: str) -> str:
    pattern = re.compile(rf"({header_regex}\n\|[-:| ]+\|\n(?:\|.*\n)+)", re.MULTILINE)
    return pattern.sub(new_table + "\n", text, count=1)


def sync_paper_markdown(method_mean: pd.DataFrame) -> None:
    text = PAPER.read_text(encoding="utf-8")
    text = replace_table(
        text,
        r"\| method \| datasets \| mean R ↓ \| mean auc-cNBI ↑ \| mean time ↓ \| top1 auc \| top3 auc \|",
        render_main_table(method_mean),
    )
    # Keep the most visible HAST-Final-S runtime statements in sync with the record-derived table.
    text = text.replace("HAST-Final-S 的 mean auc-cNBI 为 356.253、mean time 为 0.556s", "HAST-Final-S 的 mean auc-cNBI 为 356.253、mean time 为 0.556s")
    PAPER.write_text(text, encoding="utf-8")


def main() -> None:
    method_mean = load_method_mean()
    sync_main_results(method_mean)
    sync_top3_table(method_mean)
    sync_paper_markdown(method_mean)
    print("synced record-derived paper tables in main/artifacts")


if __name__ == "__main__":
    main()
