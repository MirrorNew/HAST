# -*- coding: utf-8 -*-
"""Small HAST-FAC-T speed probe.

This probe tests whether the useful fracture-advantage signal can be kept after
replacing exact two-hop scans with capped or one-hop approximations. It is a
small incremental experiment by design: no online LLM calls and no full 12-graph
evaluation unless a fast proxy winner appears.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"
RUN_DIR = ROOT / "runs" / "HAST-FACT-FAST"
CAND_DIR = RUN_DIR / "candidates"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FAC = load_module(SRC / "hast_fac_online_search.py", "hast_fac_online_runtime_fast_probe")
SEARCH = FAC.SEARCH


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_candidate(idx: int) -> str:
    files = sorted((ROOT / "runs" / "HAST-FAC" / "candidates").glob(f"candidate_{idx:04d}_*.py"))
    if not files:
        raise FileNotFoundError(f"missing HAST-FAC candidate {idx}")
    return files[0].read_text(encoding="utf-8")


FAST21_CAP24 = r'''
def degree_order(G):
    import heapq
    H = G.copy()
    n0 = max(1, H.number_of_nodes())
    order = []
    heap = []
    stamp = {}
    CAP_N = 24
    CAP_2 = 12

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
        for v in nbrs[:CAP_N]:
            if H.has_node(v):
                for w in list(H.neighbors(v))[:CAP_2]:
                    affected.add(w)
        affected.discard(u)
        H.remove_node(u)
        for v in affected:
            push(v)
    return order
'''


FAST21_ONEHOP = r'''
def degree_order(G):
    import heapq
    H = G.copy()
    n0 = max(1, H.number_of_nodes())
    order = []
    heap = []
    stamp = {}

    def local_score(u):
        if not H.has_node(u):
            return None
        d = H.degree[u]
        if d <= 0:
            return -1e18
        frontier = weak = leaf = 0.0
        nbrs = list(H.neighbors(u))
        for v in nbrs:
            dv = H.degree[v]
            frontier += 1.0 / (dv + 1.0)
            if dv <= 2:
                weak += 1.6
            elif dv <= 5:
                weak += 0.55
            if dv <= 1:
                leaf += 1.8
        p = 1.0 - H.number_of_nodes() / float(n0)
        if p < 0.2:
            return 1.15*d + 0.45*frontier + 0.25*weak + 0.08*leaf
        if p < 0.65:
            return 0.78*d + 1.45*frontier + 0.95*weak + 0.20*leaf
        return 0.55*d + 1.70*frontier + 1.10*weak + 0.55*leaf

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
        affected = list(H.neighbors(u))
        H.remove_node(u)
        for v in affected:
            push(v)
    return order
'''


FAST7_CAP32_APPROX = r'''
def degree_order(G):
    import heapq
    H = G.copy()
    n0 = max(1, H.number_of_nodes())
    order = []
    heap = []
    stamp = {}
    CAP_N = 32
    CAP_2 = 10

    def local_score(u):
        if not H.has_node(u):
            return None
        du = H.degree[u]
        if du <= 0:
            return -1e18
        neigh = list(H.neighbors(u))
        if len(neigh) > CAP_N:
            neigh = neigh[:CAP_N]
        ns = set(neigh)
        frontier = weak_tie = redundancy = ext_approx = 0.0
        low_deg = 0
        for v in neigh:
            dv = H.degree[v]
            if dv <= 2:
                low_deg += 1
            frontier += 1.0 / (dv + 1.0)
            weak_tie += 1.5 if dv <= 3 else (0.7 if dv <= 5 else 0.2)
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
            ext_approx += external
            redundancy += shared / (dv + 1.0)
        p = 1.0 - H.number_of_nodes() / float(n0)
        return (
            (0.75 + 0.55*p)*du
            + (0.38 + 0.35*(1.0-p))*ext_approx
            + (1.05 + 0.45*(1.0-p))*frontier
            + (0.85 + 0.35*(1.0-p))*weak_tie
            + (0.40 + 0.25*(1.0-p))*low_deg
            - (0.75 + 0.15*p)*redundancy
        )

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
        for v in nbrs[:CAP_N]:
            if H.has_node(v):
                for w in list(H.neighbors(v))[:CAP_2]:
                    affected.add(w)
        affected.discard(u)
        H.remove_node(u)
        for v in affected:
            push(v)
    return order
'''


FAST7_ONEHOP_APPROX = r'''
def degree_order(G):
    import heapq
    H = G.copy()
    n0 = max(1, H.number_of_nodes())
    order = []
    heap = []
    stamp = {}

    def local_score(u):
        if not H.has_node(u):
            return None
        du = H.degree[u]
        if du <= 0:
            return -1e18
        frontier = weak = low = anti_dense = 0.0
        for v in H.neighbors(u):
            dv = H.degree[v]
            frontier += 1.0 / (dv + 1.0)
            if dv <= 2:
                low += 1.0
                weak += 1.35
            elif dv <= 5:
                weak += 0.55
            else:
                anti_dense += 0.04 * min(dv, 20)
        p = 1.0 - H.number_of_nodes() / float(n0)
        return (0.85 + 0.35*p)*du + (1.20 + 0.40*(1-p))*frontier + (0.85 + 0.20*(1-p))*weak + 0.35*low - anti_dense

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
        affected = list(H.neighbors(u))
        H.remove_node(u)
        for v in affected:
            push(v)
    return order
'''


FAST_BUCKET_FRONTIER = r'''
def degree_order(G):
    import heapq
    H = G.copy()
    n0 = max(1, H.number_of_nodes())
    order = []
    heap = []
    stamp = {}

    def local_score(u):
        if not H.has_node(u):
            return None
        d = H.degree[u]
        if d <= 0:
            return -1e18
        low = mid = leaf = 0
        inv_sum = 0.0
        for v in H.neighbors(u):
            dv = H.degree[v]
            inv_sum += 1.0 / (dv + 1.0)
            if dv <= 1:
                leaf += 1
            elif dv <= 3:
                low += 1
            elif dv <= 8:
                mid += 1
        p = 1.0 - H.number_of_nodes() / float(n0)
        return (1.05 - 0.35*p)*d + (1.10 + 0.55*p)*inv_sum + (0.55 + 0.25*p)*low + 0.18*mid + (0.70 + 0.35*p)*leaf

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
        affected = list(H.neighbors(u))
        H.remove_node(u)
        for v in affected:
            push(v)
    return order
'''


def main() -> None:
    for path in [TABLE_DIR, FIG_DIR, REPORT_DIR, CAND_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    variants = [
        ("HDA", SEARCH.HDA_CODE, "root"),
        ("FAC-T-old-idx21", read_candidate(21), "old_fast_parent"),
        ("FAC-T-old-idx7", read_candidate(7), "old_fast_parent"),
        ("FAST21-cap24", FAST21_CAP24, "capped_twohop"),
        ("FAST21-onehop", FAST21_ONEHOP, "onehop"),
        ("FAST7-cap32-approx", FAST7_CAP32_APPROX, "capped_twohop_no_seen2"),
        ("FAST7-onehop-approx", FAST7_ONEHOP_APPROX, "onehop_no_seen2"),
        ("FAST-bucket-frontier", FAST_BUCKET_FRONTIER, "degree_bucket"),
    ]

    graph_payloads = SEARCH.load_search_graph_payloads()
    proxy_datasets = FAC.DEFAULT_PROXY_DATASETS
    hda_base = FAC.compute_hda_proxy_baselines(proxy_datasets)

    records: List[Dict[str, Any]] = []
    proxy_details: List[Dict[str, Any]] = []
    for idx, (name, code, family) in enumerate(variants):
        cand_file = CAND_DIR / f"candidate_{idx:04d}_{SEARCH.stable_hash(name + code, 12)}.py"
        cand_file.write_text(code, encoding="utf-8")
        row: Dict[str, Any] = {
            "idx": idx,
            "method": name,
            "family": family,
            "valid": False,
            "candidate_file": str(cand_file),
            "code_hash": SEARCH.stable_hash(code, 16),
        }
        search_eval = SEARCH.evaluate_candidate_code(code, graph_payloads, budget_ratio=0.30, timeout_s=180.0)
        if not search_eval.get("ok"):
            row.update({"error": search_eval.get("error", "search_eval_failed")})
            records.append(row)
            continue
        avg = search_eval["avg"]
        row.update({"valid": True, "error": "", "R": avg["R"], "cNBI": avg["cNBI"], "Time": avg["Time"]})
        try:
            proxy = FAC.evaluate_proxy_fac(code, proxy_datasets, hda_base)
            row.update({k: v for k, v in proxy.items() if k != "proxy_detail"})
            for d in proxy["proxy_detail"]:
                d = dict(d)
                d["method"] = name
                proxy_details.append(d)
        except Exception as exc:
            row.update({"valid": False, "error": f"proxy_failed: {exc}"})
        records.append(row)

    SEARCH.rank_records(records)
    for row in records:
        row["fac_t_score"] = FAC.fac_total_score(row)
        denom = max(0.05, float(row.get("proxy_time_s") or 0.0))
        row["adv_per_proxy_second"] = float(row.get("fac_auc_adv") or 0.0) / denom

    records = sorted(records, key=lambda r: float(r.get("fac_t_score") or -1e9), reverse=True)
    write_csv(TABLE_DIR / "hast_fact_fast_probe_summary.csv", records)
    write_csv(TABLE_DIR / "hast_fact_fast_probe_proxy_detail.csv", proxy_details)

    df = pd.DataFrame(records)
    plot_df = df[df["valid"].astype(str).str.lower().isin(["true", "1"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].scatter(plot_df["proxy_time_s"], plot_df["fac_auc_adv"], c="#4C78A8", s=60)
    for _, r in plot_df.iterrows():
        axes[0].annotate(str(r["method"]).replace("FAC-T-old-", ""), (r["proxy_time_s"], r["fac_auc_adv"]), fontsize=7)
    axes[0].axvline(1.2, color="#D62728", ls="--", lw=1)
    axes[0].set_xlabel("Proxy time (s)")
    axes[0].set_ylabel("FAC AUC advantage")
    axes[0].set_title("Fracture gain vs time")

    axes[1].barh(plot_df["method"], plot_df["fac_t_score"], color="#59A14F")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("FAC-T score")
    axes[1].set_title("Time-aware credit")

    axes[2].scatter(plot_df["Time"], plot_df["cNBI"], c="#E15759", s=60)
    for _, r in plot_df.iterrows():
        axes[2].annotate(str(r["method"]).replace("FAC-T-old-", ""), (r["Time"], r["cNBI"]), fontsize=7)
    axes[2].axvline(0.022, color="#D62728", ls="--", lw=1)
    axes[2].set_xlabel("Search graph Time")
    axes[2].set_ylabel("Search cNBI")
    axes[2].set_title("Search-suite behavior")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hast_fact_fast_probe.png", dpi=220)
    plt.close(fig)

    best = records[0]
    fast = [r for r in records if str(r.get("valid")).lower() == "true" and float(r.get("proxy_time_s") or 999) <= 1.2]
    best_fast = fast[0] if fast else best
    report = f"""# HAST-FAC-T 快速近似小实验

## 结论

这次只做小增量 proxy 实验，没有跑完整 12 图。目标是检查 FAC 信用机制能否在控制时间后继续有效。

- FAC-T 当前最佳：`{best.get('method')}`，FAC-T={float(best.get('fac_t_score') or 0):.3f}，proxy_auc={float(best.get('proxy_auc_cNBI') or 0):.3f}，proxy_time={float(best.get('proxy_time_s') or 0):.3f}s，search Time={float(best.get('Time') or 0):.5f}。
- 1.2s proxy 时间门内最佳：`{best_fast.get('method')}`，FAC-T={float(best_fast.get('fac_t_score') or 0):.3f}，fac_adv={float(best_fast.get('fac_auc_adv') or 0):.3f}，proxy_time={float(best_fast.get('proxy_time_s') or 0):.3f}s。

## 对 HAST 的启发

1. 信用分配不能只看碎裂收益，必须把时间作为一等公民；否则搜索会自然偏向更大的二跳/集合扫描。
2. 下一版 HAST 不应该鼓励“更精确的 two-hop”，而应该鼓励“更便宜的 fracture proxy”：capped scan、one-hop frontier、近似外部暴露。
3. 如果 capped 版本分数接近旧候选，就说明可以把搜索 prompt/selector 改成 FAC-T；如果掉分明显，则说明 LLM 需要学会设计新的低成本 surrogate，而不是手工压缩旧公式。

图：`{(FIG_DIR / 'hast_fact_fast_probe.png').as_posix()}`
表：`{(TABLE_DIR / 'hast_fact_fast_probe_summary.csv').as_posix()}`
"""
    (REPORT_DIR / "hast_fact_fast_probe_cn.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
