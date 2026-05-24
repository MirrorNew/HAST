# -*- coding: utf-8 -*-
"""Bounded-template probe for HAST-FAC-T.

Instead of making the tree search policy more complicated, this experiment
tests whether narrowing the candidate language to a simple capped two-hop
template improves the quality/runtime frontier.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLE = ROOT / "tables"
FIG = ROOT / "figures"
REPORT = ROOT / "reports"
RUN = ROOT / "runs" / "HAST-BOUNDED-TEMPLATE"
CAND = RUN / "candidates"
ABLATION = ROOT.parents[1] / "research" / "tree_search_ablation_20260520"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FAC = load_module(SRC / "hast_fac_online_search.py", "hast_fac_runtime_bounded_template")
SEARCH = FAC.SEARCH
E12 = FAC.E12


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_code(cap_n: int, cap_2: int, affect_n: int) -> str:
    return f'''
def degree_order(G):
    import heapq
    H = G.copy()
    n0 = max(1, H.number_of_nodes())
    order = []
    heap = []
    stamp = {{}}
    CAP_N = {cap_n}
    CAP_2 = {cap_2}
    AFFECT_N = {affect_n}

    def local_score(u):
        if not H.has_node(u):
            return None
        d = H.degree[u]
        if d <= 0:
            return -1e18
        nbrs = list(H.neighbors(u))
        if len(nbrs) > CAP_N:
            nbrs = nbrs[:CAP_N]
        ns = set(nbrs)
        frontier = weak_tie = redundancy = boundary = leaf_pressure = twohop = bridge = 0.0
        for v in nbrs:
            dv = H.degree[v]
            if dv <= 1:
                leaf_pressure += 2.0
            elif dv <= 3:
                frontier += 0.65
            frontier += 1.0 / (dv + 1.0)
            shared = external = scanned = 0
            for w in H.neighbors(v):
                if scanned >= CAP_2:
                    break
                if w == u:
                    continue
                scanned += 1
                if w in ns:
                    shared += 1
                else:
                    external += 1
            redundancy += shared / (dv + 1.0)
            if external:
                weak_tie += external / (shared + 1.0)
                boundary += external / (dv + 1.0)
                bridge += external / (dv + shared + 1.0)
                twohop += external / (dv + 1.0)
        progress = 1.0 - H.number_of_nodes() / float(n0)
        if progress < 0.18:
            wd, wf, ww, wr, wb, wl, w2, wp = 1.20, 0.52, 0.22, 0.42, 0.20, 0.16, 0.10, 0.08
        elif progress < 0.62:
            wd, wf, ww, wr, wb, wl, w2, wp = 0.72, 1.18, 0.98, 0.60, 0.82, 0.34, 0.22, 0.26
        else:
            wd, wf, ww, wr, wb, wl, w2, wp = 0.42, 1.42, 1.05, 0.78, 1.16, 0.76, 0.30, 0.40
        return wd*d + wf*frontier + ww*weak_tie + wb*boundary + wl*leaf_pressure + w2*twohop + wp*bridge - wr*redundancy

    def push(u):
        if H.has_node(u):
            s = local_score(u)
            if s is not None:
                stamp[u] = stamp.get(u, 0) + 1
                heapq.heappush(heap, (-s, str(u), stamp[u], u))

    for u in H.nodes():
        push(u)
    while H.number_of_nodes() > 0:
        while heap:
            _, _, st, u = heapq.heappop(heap)
            if H.has_node(u) and stamp.get(u, 0) == st:
                break
        else:
            for u in list(H.nodes()):
                push(u)
            continue
        order.append(u)
        nbrs = list(H.neighbors(u))
        affected = set(nbrs)
        for v in nbrs[:AFFECT_N]:
            if H.has_node(v):
                cnt = 0
                for w in H.neighbors(v):
                    affected.add(w)
                    cnt += 1
                    if cnt >= CAP_2:
                        break
        affected.discard(u)
        H.remove_node(u)
        for v in affected:
            push(v)
    return order
'''


def auc_mean(x: Any, y: Any) -> float:
    return FAC.auc_mean(x, y)


def eval_full12(code: str, method: str) -> List[Dict[str, Any]]:
    fn = SEARCH.compile_degree_order(code)
    rows = []
    for dataset in E12.EVAL.DATASETS:
        graph = E12.EVAL.read_graph(dataset)
        rate = E12.EVAL.DATASET_RATES[dataset]
        t0 = time.perf_counter()
        order = list(fn(graph.copy()))
        elapsed = time.perf_counter() - t0
        metrics = E12.EVAL.compute_metrics(graph, order, rate=rate, method_time=elapsed)
        x = metrics["removal_ratio"].to_numpy(dtype=float)
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "R": float(metrics["GCC"].mean()),
                "auc_cNBI": auc_mean(x, metrics["cNBI"].to_numpy(dtype=float)),
                "time_s": elapsed,
            }
        )
    return rows


def main() -> None:
    for p in [TABLE, FIG, REPORT, CAND]:
        p.mkdir(parents=True, exist_ok=True)
    graph_payloads = SEARCH.load_search_graph_payloads()
    hda_base = FAC.compute_hda_proxy_baselines(FAC.DEFAULT_PROXY_DATASETS)
    rows: List[Dict[str, Any]] = []
    idx = 0
    for cap_n in [16, 24, 32]:
        for cap_2 in [8, 12, 16]:
            for affect_n in [12, 18, 24]:
                code = make_code(cap_n, cap_2, affect_n)
                method = f"BT-n{cap_n}-t{cap_2}-u{affect_n}"
                path = CAND / f"candidate_{idx:04d}_{SEARCH.stable_hash(method + code, 12)}.py"
                path.write_text(code, encoding="utf-8")
                row: Dict[str, Any] = {
                    "idx": idx,
                    "method": method,
                    "cap_n": cap_n,
                    "cap_2": cap_2,
                    "affect_n": affect_n,
                    "candidate_file": str(path),
                    "valid": False,
                    "code_hash": SEARCH.stable_hash(code, 16),
                }
                search_eval = SEARCH.evaluate_candidate_code(code, graph_payloads, budget_ratio=0.30, timeout_s=180.0)
                if search_eval.get("ok"):
                    avg = search_eval["avg"]
                    row.update({"valid": True, "R": avg["R"], "cNBI": avg["cNBI"], "Time": avg["Time"], "error": ""})
                    proxy = FAC.evaluate_proxy_fac(code, FAC.DEFAULT_PROXY_DATASETS, hda_base)
                    row.update({k: v for k, v in proxy.items() if k != "proxy_detail"})
                else:
                    row["error"] = search_eval.get("error", "search_eval_failed")
                rows.append(row)
                idx += 1
    SEARCH.rank_records(rows)
    for row in rows:
        row["fac_t_score"] = FAC.fac_total_score(row)
        row["adv_per_proxy_second"] = float(row.get("fac_auc_adv") or 0.0) / max(0.05, float(row.get("proxy_time_s") or 0.0))
    rows = sorted(rows, key=lambda r: float(r.get("fac_t_score") or -1e9), reverse=True)
    write_csv(TABLE / "hast_bounded_template_probe_summary.csv", rows)

    top = rows[:3]
    full_rows: List[Dict[str, Any]] = []
    for row in top:
        code = Path(str(row["candidate_file"])).read_text(encoding="utf-8")
        full_rows.extend(eval_full12(code, str(row["method"])))
    pd.DataFrame(full_rows).to_csv(TABLE / "hast_bounded_template_probe_full12_detail.csv", index=False, encoding="utf-8-sig")
    full_mean = pd.DataFrame(full_rows).groupby("method", as_index=False).agg(R=("R", "mean"), auc_cNBI=("auc_cNBI", "mean"), time_s=("time_s", "mean"))
    base = pd.read_csv(TABLE / "HAST-FACT-ONLINE60_full12_compare.csv")
    keep = ["FAST21-cap24", "HAST-FAC-T online #24", "E26F", "PUCT", "FunSearch-like", "Clade-AHD-like", "HDA", "CoreHD"]
    compare = pd.concat([base[base["method"].isin(keep)], full_mean], ignore_index=True).sort_values("auc_cNBI", ascending=False)
    compare.to_csv(TABLE / "hast_bounded_template_probe_full12_compare.csv", index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.18})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].scatter(df["proxy_time_s"], df["fac_auc_adv"], c=df["fac_t_score"], cmap="viridis", s=55)
    axes[0].axvline(0.45, color="#D62728", ls="--", lw=1)
    axes[0].set_xlabel("proxy time (s)")
    axes[0].set_ylabel("FAC AUC advantage")
    axes[0].set_title("Bounded template candidates")
    pivot = df.pivot_table(index="cap_n", columns="cap_2", values="fac_t_score", aggfunc="max")
    im = axes[1].imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    axes[1].set_xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    axes[1].set_yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    axes[1].set_xlabel("CAP_2")
    axes[1].set_ylabel("CAP_N")
    axes[1].set_title("Best FAC-T by caps")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    colors = ["#D62728" if m.startswith("BT-") or m.startswith("FAST") or "online" in m else "#9AA0A6" for m in compare["method"]]
    axes[2].barh(compare["method"], compare["auc_cNBI"], color=colors)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("mean auc_cNBI")
    axes[2].set_title("12-graph comparison")
    fig.tight_layout()
    fig.savefig(FIG / "hast_bounded_template_probe.png", dpi=240)
    plt.close(fig)

    best = rows[0]
    best_full = full_mean.sort_values("auc_cNBI", ascending=False).iloc[0]
    report = f"""# HAST bounded-template probe

## 实验目的

不增加搜索器复杂度，只把候选空间收窄到 capped two-hop 模板，搜索 `CAP_N/CAP_2/update_cap` 三个简单参数。

## 主要结果

- Proxy 最佳：`{best['method']}`，FAC-T={float(best['fac_t_score']):.3f}，fac_adv={float(best['fac_auc_adv']):.3f}，proxy_time={float(best['proxy_time_s']):.3f}s。
- 12 图最佳模板：`{best_full['method']}`，auc_cNBI={best_full['auc_cNBI']:.3f}，time={best_full['time_s']:.3f}s。

## 结论

如果 bounded template 接近或超过在线搜索，说明下一步 HAST 的关键不是更复杂的树策略，而是更合理的候选语言和信用函数：让 LLM 归纳机制，但把实现约束在低复杂度模板内。

图：`{(FIG / 'hast_bounded_template_probe.png').as_posix()}`
表：`{(TABLE / 'hast_bounded_template_probe_summary.csv').as_posix()}`
12 图表：`{(TABLE / 'hast_bounded_template_probe_full12_compare.csv').as_posix()}`
"""
    (REPORT / "hast_bounded_template_probe_cn.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
