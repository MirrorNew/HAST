# -*- coding: utf-8 -*-
"""Audit whether paper-facing figures/tables align with recorded experiment data.

The script does not overwrite paper figures. It writes a compact Markdown report
and diagnostic plots under outputs/figure_audit/.



"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MAIN_ARTIFACTS = ROOT / "artifacts"
PAPER = ROOT / "docs" / "14_chinese_paper_full_cn.md"
FIG_DIR = MAIN_ARTIFACTS / "figures"
SOURCE_TABLE_DIR = MAIN_ARTIFACTS / "source_tables"
BENCHMARK_TABLE_DIR = SOURCE_TABLE_DIR / "benchmark_12graph"
SEARCH_RUNTIME_TABLE_DIR = SOURCE_TABLE_DIR / "search_runtime"
SCALING_TABLE_DIR = SOURCE_TABLE_DIR / "scaling"
OUT_DIR = ROOT / "outputs" / "figure_audit"


@dataclass
class Check:
    item: str
    status: str
    detail: str


def normalize_method(name: str) -> str:
    name = re.sub(r"\*", "", str(name)).strip()
    return {"ERA-like": "PUCT"}.get(name, name)


def number(value: str) -> float | None:
    if value is None:
        return None
    text = re.sub(r"\*", "", str(value)).strip()
    text = text.replace("%", "").replace("s", "").replace("h", "")
    if text in {"", "-", "incomplete", "timeout guard"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def markdown_tables(text: str) -> list[pd.DataFrame]:
    lines = text.splitlines()
    tables: list[pd.DataFrame] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:|\-]+\|?\s*$", lines[i + 1]):
            block = [lines[i], lines[i + 1]]
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                block.append(lines[j])
                j += 1
            header = [c.strip() for c in block[0].strip().strip("|").split("|")]
            rows = []
            for row in block[2:]:
                cells = [c.strip() for c in row.strip().strip("|").split("|")]
                if len(cells) == len(header):
                    rows.append(cells)
            if rows:
                tables.append(pd.DataFrame(rows, columns=header))
            i = j
        else:
            i += 1
    return tables


def find_table(tables: Iterable[pd.DataFrame], required_cols: list[str]) -> pd.DataFrame | None:
    for table in tables:
        cols = list(table.columns)
        if all(any(req in col for col in cols) for req in required_cols):
            return table
    return None


def approx_equal(a: float | None, b: float | None, tol: float = 5e-3) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


def audit_main_table(paper_main: pd.DataFrame | None, checks: list[Check]) -> pd.DataFrame:
    csv = pd.read_csv(BENCHMARK_TABLE_DIR / "method_mean_metrics.csv", encoding="utf-8-sig")
    if paper_main is None:
        checks.append(Check("5.2 main table", "FAIL", "Could not locate the Markdown main-results table."))
        return csv
    rows = []
    csv_by_method = {str(r["method"]): r for _, r in csv.iterrows()}
    for _, r in paper_main.iterrows():
        method = normalize_method(r["method"])
        if method not in csv_by_method:
            rows.append((r["method"], "missing_in_csv", "", ""))
            continue
        c = csv_by_method[method]
        comparisons = [
            ("datasets", number(r["datasets"]), float(c["datasets"]), 0.01),
            ("mean_R", number(r["mean R ↓"]), float(c["mean_R"]), 0.001),
            ("mean_auc_cNBI", number(r["mean auc-cNBI ↑"]), float(c["mean_auc_cNBI"]), 0.001),
            ("mean_time_s", number(r["mean time ↓"]), float(c["mean_time_s"]), 0.001),
            ("top1_auc", number(r["top1 auc"]), float(c["top1_auc"]), 0.01),
            ("top3_auc", number(r["top3 auc"]), float(c["top3_auc"]), 0.01),
        ]
        for metric, paper_v, csv_v, tol in comparisons:
            if not approx_equal(paper_v, csv_v, tol):
                rows.append((r["method"], metric, paper_v, csv_v))
    if rows:
        checks.append(Check("5.2 main table vs benchmark_12graph/method_mean_metrics.csv", "FAIL", f"{len(rows)} mismatched cells."))
    else:
        checks.append(Check("5.2 main table vs benchmark_12graph/method_mean_metrics.csv", "PASS", "All reported cells match the CSV source."))
    return pd.DataFrame(rows, columns=["method", "metric", "paper_value", "csv_value"])


def audit_hast_time_conflicts(checks: list[Check]) -> pd.DataFrame:
    method_mean = pd.read_csv(BENCHMARK_TABLE_DIR / "method_mean_metrics.csv", encoding="utf-8-sig")
    main_results = pd.read_csv(BENCHMARK_TABLE_DIR / "hast_main_results.csv", encoding="utf-8-sig")

    records = []
    for label, method, main_label in [
        ("HAST-Final-Q", "HAST-Final-Q", "HAST-Bounded quality"),
        ("HAST-Final-S", "HAST-Final-S", "HAST-Bounded speed"),
    ]:
        values = {
            "method_mean_metrics": float(method_mean.loc[method_mean["method"].eq(method), "mean_time_s"].iloc[0]),
            "main_results_unified": float(main_results.loc[main_results["paper_label_display"].eq(main_label), "time_s"].iloc[0]),
        }
        for source, value in values.items():
            records.append({"method": label, "source": source, "time_s": value})
        if max(values.values()) - min(values.values()) > 1e-3:
            checks.append(Check(f"{label} time consistency", "FAIL", str(values)))
        else:
            checks.append(Check(f"{label} time consistency", "PASS", str(values)))
    return pd.DataFrame(records)


def audit_scaling_tables(tables: list[pd.DataFrame], checks: list[Check]) -> pd.DataFrame:
    full = pd.read_csv(SCALING_TABLE_DIR / "full_eval_500_to_10k_unified.csv", encoding="utf-8-sig")
    summary = full[full["ok"].astype(bool)].groupby(["method", "n"], as_index=False).agg(
        R=("R", "mean"),
        auc_cNBI=("auc_cNBI", "mean"),
        time_s=("time_s", "mean"),
    )
    paper_scaling = find_table(tables, ["10k mean R", "10k mean auc-cNBI", "10k mean time"])
    rows = []
    if paper_scaling is None:
        checks.append(Check("5.5 full scaling table", "FAIL", "Could not locate paper scaling table."))
        return summary
    by_method = {r["method"]: r for _, r in summary[summary["n"].eq(10000)].iterrows()}
    for _, r in paper_scaling.iterrows():
        method = str(r["method"]).strip()
        if method not in by_method:
            rows.append((method, "missing_in_csv", "", ""))
            continue
        c = by_method[method]
        for metric, paper_col, csv_col in [
            ("R", "10k mean R", "R"),
            ("auc_cNBI", "10k mean auc-cNBI", "auc_cNBI"),
            ("time_s", "10k mean time", "time_s"),
        ]:
            paper_v = number(r[paper_col])
            csv_v = float(c[csv_col])
            if not approx_equal(paper_v, csv_v, 0.001):
                rows.append((method, metric, paper_v, csv_v))
    if rows:
        checks.append(Check("5.5 full scaling table", "FAIL", f"{len(rows)} mismatched cells."))
    else:
        checks.append(Check("5.5 full scaling table", "PASS", "10k values match scaling CSV."))
    return pd.DataFrame(rows, columns=["method", "metric", "paper_value", "csv_value"])


def plot_time_conflicts(df: pd.DataFrame) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    sources = list(df["source"].unique())
    methods = list(df["method"].unique())
    width = 0.22
    x = range(len(methods))
    colors = {"method_mean_metrics": "#0072B2", "main_results_unified": "#D55E00", "stage_evidence": "#009E73"}
    for i, source in enumerate(sources):
        sub = df[df["source"].eq(source)].set_index("method").reindex(methods)
        ax.bar([v + (i - 1) * width for v in x], sub["time_s"], width=width, label=source, color=colors.get(source))
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods)
    ax.set_ylabel("runtime (s)")
    ax.set_title("Audit: HAST final runtime differs across data sources")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = OUT_DIR / "audit_hast_final_runtime_sources.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_main_table_mismatches(mismatch: pd.DataFrame) -> Path | None:
    if mismatch.empty:
        return None
    numeric = mismatch.dropna(subset=["paper_value", "csv_value"]).copy()
    if numeric.empty:
        return None
    numeric["label"] = numeric["method"].astype(str) + "\n" + numeric["metric"].astype(str)
    fig, ax = plt.subplots(figsize=(max(7, len(numeric) * 0.55), 4.0))
    idx = range(len(numeric))
    ax.scatter(idx, numeric["paper_value"], label="paper", color="#D55E00", s=36)
    ax.scatter(idx, numeric["csv_value"], label="CSV", color="#0072B2", s=28)
    for i, (_, row) in enumerate(numeric.iterrows()):
        ax.plot([i, i], [row["paper_value"], row["csv_value"]], color="#9CA3AF", lw=1)
    ax.set_xticks(list(idx))
    ax.set_xticklabels(numeric["label"], rotation=55, ha="right", fontsize=7)
    ax.set_title("Audit: paper table cells that differ from CSV")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = OUT_DIR / "audit_main_table_mismatches.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_report(checks: list[Check], artifacts: list[Path], tables: dict[str, pd.DataFrame]) -> Path:
    lines = [
        "# HAST Figure/Data Alignment Audit",
        "",
        f"Paper: `{PAPER}`",
        f"Figure dir: `{FIG_DIR}`",
        "",
        "## Summary",
        "",
    ]
    for check in checks:
        marker = "OK" if check.status == "PASS" else "ISSUE"
        lines.append(f"- **{marker}** `{check.item}`: {check.detail}")
    lines.extend(["", "## Diagnostic Figures", ""])
    for path in artifacts:
        lines.append(f"- `{path}`")
    for name, df in tables.items():
        lines.extend(["", f"## {name}", ""])
        if df is None or df.empty:
            lines.append("No mismatches.")
        else:
            lines.append(df.to_markdown(index=False))
    path = OUT_DIR / "figure_data_alignment_audit.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = PAPER.read_text(encoding="utf-8")
    tables = markdown_tables(text)
    checks: list[Check] = []

    paper_main = find_table(tables, ["method", "mean R", "mean auc-cNBI", "mean time"])
    main_mismatch = audit_main_table(paper_main, checks)
    time_sources = audit_hast_time_conflicts(checks)
    scaling_mismatch = audit_scaling_tables(tables, checks)

    artifacts = [plot_time_conflicts(time_sources)]
    p = plot_main_table_mismatches(main_mismatch)
    if p:
        artifacts.append(p)

    report = write_report(
        checks,
        artifacts,
        {
            "Main Table Mismatches": main_mismatch,
            "Scaling Table Mismatches": scaling_mismatch,
            "HAST Runtime Sources": time_sources,
        },
    )
    print(report)


if __name__ == "__main__":
    main()
