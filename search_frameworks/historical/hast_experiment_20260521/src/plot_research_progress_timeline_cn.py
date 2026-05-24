# -*- coding: utf-8 -*-
"""Plot a Chinese academic-style research progress map with matplotlib."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FONT_REGULAR_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_BOLD_PATH = Path(r"C:\Windows\Fonts\msyhbd.ttc")
REG_FONT = None
BOLD_FONT = None


def add_box(ax, xy, width, height, title, body, fc="#F7F7F7", ec="#333333", lw=1.2):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.025,
        y + height - 0.055,
        title,
        ha="left",
        va="top",
        fontproperties=BOLD_FONT,
        color="#111111",
    )
    ax.text(
        x + 0.025,
        y + height - 0.135,
        body,
        ha="left",
        va="top",
        fontproperties=REG_FONT,
        color="#333333",
        linespacing=1.34,
    )


def add_arrow(ax, start, end, rad=0.0, color="#555555"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.2,
            color=color,
            shrinkA=4,
            shrinkB=4,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def main() -> None:
    global REG_FONT, BOLD_FONT
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    font_regular = FONT_REGULAR_PATH if FONT_REGULAR_PATH.exists() else Path(r"C:\Windows\Fonts\msyh.ttc")
    font_bold = FONT_BOLD_PATH if FONT_BOLD_PATH.exists() else font_regular
    font_manager.fontManager.addfont(str(font_regular))
    font_manager.fontManager.addfont(str(font_bold))
    REG_FONT = font_manager.FontProperties(fname=str(font_regular), size=9.6)
    BOLD_FONT = font_manager.FontProperties(fname=str(font_bold), size=13)
    TITLE_FONT = font_manager.FontProperties(fname=str(font_bold), size=20)
    SUBTITLE_FONT = font_manager.FontProperties(fname=str(font_regular), size=11.5)
    FOOT_FONT = font_manager.FontProperties(fname=str(font_regular), size=9.5)
    SMALL_FONT = font_manager.FontProperties(fname=str(font_regular), size=8.5)
    font_name = font_manager.FontProperties(fname=str(font_regular)).get_name()

    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(15.8, 8.8), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        "网络瓦解算法发现研究进展脉络",
        ha="center",
        va="center",
        fontproperties=TITLE_FONT,
        color="#111111",
    )
    ax.text(
        0.5,
        0.915,
        "从 PUCT 自由探索到 HAST-FAC 机制发现，再到当前轻量化蒸馏",
        ha="center",
        va="center",
        fontproperties=SUBTITLE_FONT,
        color="#555555",
    )

    w, h = 0.205, 0.18
    xs = [0.055, 0.285, 0.515, 0.745]
    y1, y2, y3 = 0.66, 0.395, 0.13

    boxes = [
        (
            (xs[0], y1),
            "1. PUCT 起点",
            "自由代码树搜索\n证明 LLM 能生成可执行\nnetwork dismantling 启发式\n问题：好节点像 lucky hit，解释弱",
            "#F8F8F8",
        ),
        (
            (xs[1], y1),
            "2. e26f / BPSD",
            "从搜索轨迹中出现 open-wedge、\nbridge、neighbor-degree 信号\n形成可解释候选机制\n但不能只讲 PUCT 原创",
            "#F8F8F8",
        ),
        (
            (xs[2], y1),
            "3. cNBI 指标重构",
            "GCC/R 只看最大连通分量\ncNBI 关注残余组件分布\n主线变为：评价目标引导发现",
            "#F8F8F8",
        ),
        (
            (xs[3], y1),
            "4. DACTS 机制化",
            "把 bridge / two-hop / update\n等机制写成 typed slots\n优点：稳定、可审计\n风险：像人工特征工程",
            "#F8F8F8",
        ),
        (
            (xs[0], y2),
            "5. 多框架对比",
            "PUCT / MCTS / Clade / FunSearch\n/ AlphaEvolve 统一比较\n结论：强代码常共享 two-hop、\nfrontier、bridge 等机制",
            "#F8F8F8",
        ),
        (
            (xs[1], y2),
            "6. HAST 专精",
            "从通用框架前 200 节点学习经验\n再跑 300 节点专精搜索\n搜索命中高：strict=184\n但 12 图泛化弱",
            "#FFF9ED",
        ),
        (
            (xs[2], y2),
            "7. HAST 诊断",
            "问题不是搜不到高分节点\n而是 family credit 过拟合生成图\n塌缩到 local two-hop/neighbor\n缺少真实图负反馈",
            "#FFF9ED",
        ),
        (
            (xs[3], y2),
            "8. HAST-FAC",
            "FAC 奖励相对 HDA 的碎裂优势\n不奖励 HDA 的花哨变体\n60 节点搜索后主线转向\nfrontier_weak_tie",
            "#EEF4FF",
        ),
        (
            (xs[0], y3),
            "9. Full-12 验证",
            "#44 frontier_weak_tie\nR=0.362, AUC-cNBI=380.5\n超过 FunSearch / Clade / PUCT\n但平均时间约 29.4s",
            "#EEF4FF",
        ),
        (
            (xs[1], y3),
            "10. 轻量化蒸馏",
            "FW-Lite 速度降到 3.3s 量级\n但 AUC-cNBI 降到 324.5\n说明机制有效，但实现还未压好",
            "#EEF4FF",
        ),
        (
            (xs[2], y3),
            "11. 当前状态",
            "HAST-FAC 100 节点搜索：94/100\nTop 仍为 frontier_weak_tie\n当前核心问题：\n强机制如何变成简单算法",
            "#EAF6F0",
        ),
        (
            (xs[3], y3),
            "12. 下一步",
            "停止盲目加复杂度\n抽取短公式：degree + outside-twohop\n+ weak-tie - redundancy\n做消融、速度约束、12 图复核",
            "#EAF6F0",
        ),
    ]

    for xy, title, body, fc in boxes:
        add_box(ax, xy, w, h, title, body, fc=fc)

    # Row arrows.
    for i in range(3):
        add_arrow(ax, (xs[i] + w, y1 + h * 0.52), (xs[i + 1], y1 + h * 0.52))
        add_arrow(ax, (xs[i] + w, y2 + h * 0.52), (xs[i + 1], y2 + h * 0.52))
        add_arrow(ax, (xs[i] + w, y3 + h * 0.52), (xs[i + 1], y3 + h * 0.52))

    # Down arrows linking rows.
    add_arrow(ax, (xs[3] + w * 0.5, y1), (xs[0] + w * 0.5, y2 + h), rad=0.22)
    add_arrow(ax, (xs[3] + w * 0.5, y2), (xs[0] + w * 0.5, y3 + h), rad=0.22)

    ax.text(
        0.055,
        0.055,
        "读图方式：上排是“发现问题与机制”，中排是“框架对比与诊断”，下排是“当前验证与压缩”。"
        " 颜色仅区分阶段：灰=基础探索，浅黄=诊断，浅蓝=FAC验证，浅绿=当前/下一步。",
        ha="left",
        va="center",
        fontproperties=FOOT_FONT,
        color="#555555",
    )
    ax.text(
        0.945,
        0.025,
        "生成：research/hast_experiment_20260521",
        ha="right",
        va="center",
        fontproperties=SMALL_FONT,
        color="#777777",
    )

    out_png = FIG_DIR / "research_progress_timeline_cn_fixed.png"
    out_pdf = FIG_DIR / "research_progress_timeline_cn_fixed.pdf"
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out_png.resolve())
    print(out_pdf.resolve())


if __name__ == "__main__":
    main()
