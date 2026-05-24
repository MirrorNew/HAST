# HAST：面向网络瓦解启发式的信用感知与有界 LLM 自动搜索

# 摘要

网络瓦解启发式通常用最大连通分量随删除比例的变化来评价，但这种反馈对于 LLM 驱动的程序搜索并不充分：一个候选程序可能因为继承了强 root heuristic 而取得较好分数，却没有真正贡献新的碎裂机制。本文提出 HAST，一个面向网络瓦解启发式自动发现的信用感知与有界 LLM 搜索框架。HAST 首先用过程碎裂指标 cNBI 描述残余图的非冗余碎裂形态，并将候选得分改写为相对 root heuristic 的新增碎裂信用；随后把时间成本纳入搜索期信用，抑制自由代码搜索向慢二跳扫描和频繁全图重算漂移；最后从自由搜索日志中归纳有界候选语言，将有效局部机制压缩为可执行、可解释、可限界的启发式模板。在 12 个 benchmark graphs 上，HAST-Final-Q 达到 mean auc-cNBI 358.066，保留 ERA-like 99.5% 的 auc-cNBI，同时运行时间从 9.785s 降至 1.008s；HAST-Final-S 达到 mean auc-cNBI 356.253，保留 99.0% 的 auc-cNBI，同时运行时间降至 0.556s。实验结果表明，HAST 的贡献不是宣称在所有图和所有指标上超过所有方法，而是在网络瓦解这个特定任务中，把 LLM 自动启发式搜索从“追逐绝对分数”校准为“给真实增量碎裂记功，并在有界局部语言中实现质量-时间折中”。

**关键词**：网络瓦解；LLM 程序搜索；启发式发现；信用分配；有界生成；复杂网络

# 1 引言

网络瓦解问题研究如何通过删除少量关键节点快速破坏网络连通性。给定图 $G=(V,E)$，典型目标是构造节点删除序列 $\pi=(v_1,\ldots,v_n)$，使删除前缀后最大连通分量尽快缩小。该问题在复杂系统鲁棒性、基础设施风险分析和信息传播控制中都有重要意义。已有工作已经提出了从中心性、2-core、消息传递、优化近似到学习式策略的一系列方法，它们回答了一个核心问题：给定网络，如何设计高质量删除序列。

LLM 程序搜索提供了另一条路径。与人工设计单个启发式不同，LLM 可以在可执行程序空间中生成、评估和迭代候选算法。FunSearch、ReEvo、HSEvo、AlphaEvolve 等工作表明，LLM 与自动评估器、进化或树搜索机制结合后，可以发现具有竞争力的程序或启发式。对于网络瓦解而言，这种范式很自然：把一个启发式写成函数，让 LLM 修改代码，用评价器检查删除序列质量，再把高分候选保留下来。

然而，网络瓦解给 LLM 程序搜索带来两个更细的问题。第一，标准 GCC/R 曲线是任务最终评价所必需的，但它对搜索期信用分配过于粗粒度。若候选程序从 HDA 或 degree root 出发，绝对得分中包含大量 root heuristic 已经具备的能力；LLM 修改代码带来的真实增量贡献可能被淹没。第二，一旦评价函数更重视碎裂质量，自由代码搜索容易通过更宽的二跳扫描、重复邻域枚举或频繁连通分量计算来换取短期指标提升，从而产生复杂度漂移。换言之，仅有“更强评价函数”并不够；搜索还需要知道哪些程序修改应被记功，以及哪些候选语言应被限制。

本文提出 HAST，围绕“信用分配”和“有界生成”组织 LLM 启发式搜索。HAST 将候选程序统一为接口 `HAST_order(G) -> removal order`，并在同一 evaluation harness 下评价 $R$、auc-cNBI 和运行时间。cNBI 用于补充 GCC/R 对残余碎裂形态的刻画；HDA-relative fracture credit 用于把 root 已有能力从候选信用中扣除；time-aware credit 用于把计算代价提前纳入搜索；bounded candidate language 则把自由搜索日志中反复出现的有效局部机制压缩为有上界的模板。

本文的贡献可以概括为三点。

1. 我们指出网络瓦解中的 LLM 启发式发现不仅是“生成更多候选程序”的问题，更是搜索期信用分配问题：候选绝对分数会混合 root 能力和新增机制，因而需要相对碎裂信用。
2. 我们提出 cNBI 与 HDA-relative fracture credit，将过程碎裂作为搜索信号，同时保持 GCC/R 作为标准瓦解目标的评价地位。
3. 我们给出一条从自由 LLM 搜索到有界候选语言的压缩路径，并用 12 图实验、消融、搜索成本统计和 10k 规模运行时间证据说明：HAST 的最终候选位于更实用的质量-时间折中区域，而不是依赖无界慢扫描获得高分。

# 2 相关工作

## 2.1 网络瓦解启发式

网络瓦解的经典研究首先关注如何构造有效删除序列。Morone 与 Makse 的最优渗流和 Collective Influence 方法从渗流角度识别对全局连通性有重要影响的节点，强调弱连接位置在系统断裂中的作用 [1]。Braunstein 等将 network dismantling 表述为具有集体性的优化问题，并提出消息传递和 Min-Sum 相关框架 [2]。Zdeborová 等提出 CoreHD，将 decycling 和 dismantling 限制在 2-core 上，通过高阶度贪心以较低计算代价取得较强表现 [3]。Ren 等进一步研究 generalized network dismantling，将异质节点成本纳入目标，使低代价瓦解成为正式优化问题 [4]。

这些方法建立了网络瓦解启发式的主要谱系：中心性和渗流方法提供可解释的结构信号，消息传递和优化方法提供更强理论动机，CoreHD/HDA 类方法在质量和效率之间取得实用折中。它们的共同目标是直接生成节点删除序列，而不是分析“当一个程序搜索器修改启发式代码时，哪一段修改真正带来了增量碎裂”。因此，经典网络瓦解工作为 HAST 提供了目标、指标和基线，但并不直接解决 LLM 程序搜索中的信用归因。

## 2.2 学习式网络瓦解

学习式方法将网络瓦解进一步转化为可从数据中学习的策略。FINDER 将关键节点识别表述为深度强化学习序列决策问题，学习攻击策略以寻找关键节点 [5]。Grassia 等的 GDM 展示了机器学习模型可以从小系统迁移到更大网络，并输出系统解体的早期预警信号 [6]。Zhang 与 Wang 的 NIRM 尝试从 tiny networks 学习可泛化的节点排序模式 [7]。

这些工作说明瓦解策略不必完全由人工规则设计，数据驱动策略可以学习到跨图的拆解模式。HAST 与其问题对象不同：HAST 不直接训练一个节点排序模型，而是让 LLM 在启发式程序空间中搜索候选算法。学习式瓦解方法通常把信用给到节点选择、策略轨迹或最终瓦解效果；HAST 的信用对象则是相对 root heuristic 的代码结构增量。这个区别决定了 HAST 需要显式处理 root 继承带来的信用混淆。

## 2.3 LLM 程序搜索与自动启发式设计

LLM 程序搜索和自动启发式设计近年迅速发展。FunSearch 证明了“LLM 生成 + 自动评估器 + 演化筛选”能够发现新的数学与算法程序 [8]。ReEvo 将反思文本与进化搜索结合，把 LLM 用作 language hyper-heuristics [9]。PPSN 相关基准表明，自动启发式设计不能仅依靠单次生成，演化式搜索在多个任务与模型上发挥重要作用 [10]。HSEvo 把多样性与收敛性的权衡显式放入自动启发式设计框架 [11]。AlphaEvolve 将编码代理用于更一般的科学与算法发现 [12]。AutoRNet 则将 LLM 与 evolutionary algorithm 用于鲁棒 scale-free network design [13]。

Google 的 ERA 系统进一步展示了面向实证软件研究的自动研究辅助范式：系统围绕可执行候选、自动评估、搜索日志和迭代改写组织研究过程，使 LLM 不只生成文本，而是参与可运行实验对象的构造与筛选 [14]。本文实验中的 `ERA-like` 基线即指这种“LLM 生成候选程序 + 自动 evaluator + 树式/迭代搜索”的近邻框架实现，而不是另一个独立命名的原论文框架。

这些工作回答了“给定评价器，如何搜索更好程序”的通用问题。HAST 的定位更窄：在网络瓦解这个任务中，评价曲线粗、root heuristic 强、候选程序容易通过慢扫描换取质量，因此 LLM 搜索需要任务结构化信用和生成边界。HAST 不试图取代通用 LLM 程序搜索框架，而是补足它们在网络瓦解场景中的两个具体缺口：相对碎裂信用和日志诱导的有界候选语言。

## 2.4 本文定位

已有网络瓦解工作主要关心“如何拆网络”，已有 LLM 程序搜索主要关心“如何搜索程序”。HAST 关注二者交叉处的搜索期问题：

- 当 root heuristic 很强且 GCC/R 反馈较粗时，如何给候选代码的真实增量贡献记功？
- 当碎裂信用鼓励复杂局部扫描时，如何约束 LLM 生成空间而不退回纯人工设计？

这一定位使 HAST 的主张保持在可证据支持的范围内：本文不宣称提出了所有网络上的最优瓦解算法，也不宣称 HAST 是通用最优 LLM tree search；本文只主张，在当前 12 图 benchmark 和统一 evaluation harness 下，信用感知与有界生成能够把 LLM 发现的启发式推向更实用的质量-时间折中。

# 3 动机观察

本章只负责把问题和解决方向逐层引出来，而不提前展开完整算法。网络瓦解的最终目标当然仍由 GCC/R 约束；本文要指出的是，当 LLM 在启发式程序空间中搜索候选时，标准 GCC/R 反馈不足以回答“哪一段候选代码真正带来了新增碎裂”。因此，本章按三个 observation 组织证据链：第一，GCC/R 对搜索期信用过粗；第二，搜索信用需要从绝对表现转向相对 root 的新增碎裂；第三，相对碎裂信用虽然有效，却会诱导自由代码搜索变慢，因而需要从搜索日志中归纳可信引导和生成边界。

## 3.1 Motivation Observation 1：GCC/R 无法全面评估网络的碎裂性。

网络瓦解通常以最大连通分量随删除比例的变化来评价，这一点在最终比较中是必要的。给定删除序列 $\pi=(v_1,\ldots,v_n)$，删除前 $t$ 个节点后得到残余图 $G_t$，GCC/R 直接描述最大连通分量是否被压低。然而，对 LLM 自动程序搜索而言，GCC/R 是粗粒度反馈：两个候选可以达到完全相同或非常接近的 $R$，却留下不同的 residual fragmentation。换言之，GCC/R 告诉我们“最大块有多大”，但不充分告诉我们“剩余节点被切成了什么形态”。

图 3.1 只使用 HDA、CI、CLUC 等 basic baselines。我们在离线分析中筛选 $R$/GCC 完全相同但 residual fragmentation 差异明显的 cases；图中蓝色表示最大连通分量，红色和橙色表示第 2 到第 5 大连通分量，灰色表示更小碎片。每个 case 左右两侧的 $R$/GCC 相同，但残余图的分散程度不同。

![Basic baseline same-R residual fragmentation](../artifacts/figures/fig21_obs1_basic_baseline_same_r_horizontal.png)

| Case | Baseline A | R/GCC A | Baseline B | R/GCC B | 观察 |
|---|---|---:|---|---:|---|
| Collaboration | HDA | 0.004810 | CLUC | 0.004810 | R/GCC 完全相同，但 HDA 留下更多分散小块，CLUC 的前几个残余块更集中 |
| Collaboration | HDA | 0.011063 | CI | 0.011063 | R/GCC 完全相同，但 HDA 的剩余碎片更分散，CI 留下更重的前五个残余块 |
| Collaboration | CI | 0.012987 | CLUC | 0.012987 | R/GCC 完全相同，但两种 basic baseline 的残余碎裂方向不同 |

我们得出，不能把 GCC/R 当做唯一的评价指标。对于搜索器来说，还需要一个能区分 same-GCC residual structure 的辅助信号。否则，当 LLM 从强 root heuristic 出发进行局部修改时，搜索很容易把“root 本来就能压低 GCC”的能力误记为候选修改的贡献。

针对 Observation 1，一个自然修正是让搜索不只看最大连通分量，还看残余图是否真的被切成更多、更均衡的碎片。本文拟提出 cNBI 作为过程碎裂信号。给定原图 $G=(V,E)$，$n=|V|$，候选启发式 $h$ 产生删除序列 $\pi_h=(v_1,\ldots,v_n)$。删除前 $t$ 个节点后得到残余图：
$$
G_t^h = G[V \setminus \{v_1,\ldots,v_t\}].
$$

设 $G_t^h$ 的连通分量大小为 $s_{t,1}\ge s_{t,2}\ge \cdots \ge s_{t,k_t}$，其中 $n_t=\sum_i s_{t,i}=n-t$。pairwise disconnectedness 按残余节点对归一化：

$$
\mathrm{PD}_t
= 1 - \frac{\sum_i s_{t,i}(s_{t,i}-1)}{n_t(n_t-1)}.
$$

当 $n_t\le 1$ 时，$\mathrm{PD}_t$ 定义为 1。它等价于两个随机残余节点落在不同连通分量的概率。有效碎片数使用 Hill number 或 inverse-Herfindahl 形式：

$$
p_{t,i} = \frac{s_{t,i}}{n_t},
\quad
\mathrm{EC}_t = \frac{1}{\sum_i p_{t,i}^2}.
$$

前五大残余块质量集中度定义为：

$$
\mathrm{Top5}_t
= \frac{\sum_{i=1}^{\min(5,k_t)} s_{t,i}}{n}.
$$

于是 cNBI 定义为：

$$
\mathrm{cNBI}_t
= \frac{\mathrm{PD}_t \cdot \mathrm{EC}_t}{1+\mathrm{Top5}_t}.
$$

完整过程曲线采用删除比例网格 $q_t=t/n$ 上的离散积分：

$$
\mathrm{AUC}_{\mathrm{cNBI}}(h;G)
= \sum_{t\in T} \mathrm{cNBI}_t(h;G)\Delta q_t.
$$

cNBI 的作用是补充而非替代 GCC/R。$\mathrm{PD}_t$ 奖励更多残余节点对被分到不同连通分量，$\mathrm{EC}_t$ 奖励有效碎片数量增加，$\mathrm{Top5}_t$ 惩罚前五个残余块仍占据过多原图质量。这样做避免单纯组件数被大量微小孤点虚高，也避免只看最大连通分量时忽略剩余质量是否仍集中在少数大块中。

## 3.2 Motivation Observation 2: 碎裂性评估真的改变 LLM 生成更有贡献的代码结构了吗？

直接奖励候选的绝对碎裂水平仍然不够。若候选从 HDA/degree root 出发，它的绝对 auc-cNBI 中包含大量 root heuristic 已经具备的 residual degree backbone；高分并不自动意味着 LLM 新增代码结构有效。因此，搜索信用应该从“候选绝对分数”转向“候选相对 root 的新增碎裂”：

$$
\Delta_{\mathrm{frag}}(h\mid h_0)
= \mathrm{Frag}(h)-\mathrm{Frag}(h_0),
$$

其中 $h_0$ 是 root heuristic，$\mathrm{Frag}(h)$ 可以由搜索图上的 cNBI 曲线或 early fracture proxy 计算。这个定义相当于把 root 已经能完成的高阶度删除能力视作 baseline value，只把新增的 frontier、weak-tie、boundary、redundancy-aware 等机制记入候选修改的信用。

图 3.2 对应的新实验需要重新运行。新版实验将直接比较 `GCC/R-only`、`Absolute-cNBI` 和 `Relative-ΔcNBI` 三种搜索信用，每组固定 100 个候选，并在同一候选接口和同一验证器下记录有效率、搜索时间、$R$、auc-cNBI 与最终 Pareto 位置。该图暂不沿用旧的 HAST 阶段统计，避免把上一版搜索日志误作为新三阶段框架的证据。

| Search credit | candidates | 状态 | 目标 |
|---|---:|---|---|
| GCC/R-only | 100 | 待重跑 | 检验只看 GCC/R 是否会漏掉过程碎裂差异 |
| Absolute-cNBI | 100 | 待重跑 | 检验绝对碎裂分数是否混入 root heuristic 贡献 |
| Relative-ΔcNBI | 100 | 待重跑 | 检验相对 root 的新增碎裂信用是否更适合搜索归因 |

这一 observation 的结论是：搜索器需要把评价从纯 GCC/R 扩展为 $R$、过程碎裂和时间三元证据，并且把碎裂信用相对化。它支撑 HAST 的第一部分设计，但还不能直接推出最终方法，因为相对碎裂信用会改变 LLM 的激励方向。

## 3.3 Motivation Observation 3：LLM自由搜索导致的启发式算法失控问题

Observation 2 让搜索更重视候选相对 root 新增的碎裂，确实能把候选推向更强的 residual fragmentation，但这也带来新的问题。如果让 LLM 在自由代码空间里追逐该指标，它可能通过无界 two-hop 扫描、全图重复扫描、频繁重算 connected components 等方式获得短期收益。这些代码在小图或搜索图上可能看起来有效，但在大图上会变慢，并且不一定对应简洁稳定的网络瓦解机制。

因此，HAST 不能只是“换一个奖励函数”。它还需要把自由探索日志转化为有约束、有针对性的候选语言。我们在观察一些代码后，使用人工进行了一次总结，让这些信息注入后，发现依然有提升。

| 自由探索日志观察 | 可信引导 | LLM 生成边界 |
|---|---|---|
| 高信用候选保留 residual degree backbone | degree/root 仍是可信骨架 | 不删除基础度信号，只在其上加局部碎裂项 |
| 高信用候选反复使用 frontier、weak-tie、boundary | 这些局部结构对碎裂有针对性 | 允许局部边界、弱连接、冗余抑制特征 |
| 慢候选常做无界 two-hop 或 nested-neighbor scan | 复杂扫描会制造虚假收益 | 限制 two-hop 范围和邻域枚举预算 |
| 慢候选频繁重算 connected components | 全图重算不适合作为默认生成模式 | 限制局部更新和刷新频率 |
| 低潜力 family 长期不改进 | family 级搜索方向不可信 | 剪掉低潜力 family，把预算给高潜力模板 |

实验结果需要使用新版三阶段框架重新生成。图 3.3 将比较 `Relative-Free`、`CostAware-Free` 和 `Bounded-Guided` 三组候选，每组 100 个候选，并记录候选有效率、搜索时间和完整碎裂指标。该实验的目标不是证明有界引导一定带来更高质量，而是检验它是否能在保留可接受 auc-cNBI 的同时降低无效生成和慢扫描成本。

| Generation mode | candidates | 状态 | 目标 |
|---|---:|---|---|
| Relative-Free | 100 | 待重跑 | 观察相对碎裂信用下的自由生成成本 |
| CostAware-Free | 100 | 待重跑 | 观察加入时间信用后是否抑制慢扫描 |
| Bounded-Guided | 100 | 待重跑 | 观察日志归纳边界是否提高有效率并降低搜索成本 |

我们认为这个生成约束是有潜力的，但是不能在实际搜索中使用人工任意限制，而是来自搜索日志中的总结并生成。哪些局部结构反复出现在高信用候选中，哪些代码模式反复导致慢扫描，哪些 family 长期没有改进潜力。换言之，HAST 使用“自由探索日志观察 -> 可信引导/生成边界”的机制，把自由代码搜索压缩成更可信的候选空间。

# 4 方法：HAST

## 4.1 HAST 总览

HAST 的目标不是提出一个更复杂的通用搜索器，而是在网络瓦解启发式发现中解决三个具体问题：GCC/R 对搜索期信用过粗，候选绝对分数混入 root heuristic 贡献，以及自由代码搜索容易通过慢扫描追逐碎裂指标。图 4.1 给出 HAST 的主要框架：LLM 生成候选程序，沙箱和统一接口检查候选是否可运行，图评估器计算 $R$、auc-cNBI 和 runtime，信用模块将绝对指标转化为相对 root 的碎裂贡献并加入时间惩罚，搜索日志再被压缩为 bounded candidate language，最终输出两个 Pareto 点。

![HAST framework](../artifacts/figures/Gemini-Framework.png)

从读图角度看，HAST 的关键闭环有三层。第一层是 evaluation loop：候选程序必须输出节点删除序列，统一评估器负责计算曲线指标，避免候选在内部自行定义评价。第二层是 credit loop：树搜索不只看候选绝对得分，而是根据相对碎裂信用和时间风险选择后续扩展方向。第三层是 distillation loop：自由搜索日志不只是临时记录，而是被用来归纳哪些局部信号可信、哪些代码模式应被限制。这个结构使 HAST 与一般“LLM 反复生成高分代码”的范式区分开来。

## 4.2 候选接口与统一评估器

所有候选启发式都被约束为同一接口：

```python
def HAST_order(G):
    return ordered_nodes
```

输入是图 $G$，输出是完整节点删除序列。传统启发式、ERA-like、FunSearch-like、Clade-AHD-like、MCTS-AHD-like、AlphaEvolve-like 和 HAST 最终候选都通过同一接口进入 evaluation harness。这样做有两个作用：其一，避免不同方法输出中间状态或局部评分函数导致比较口径不一致；其二，把 cNBI、GCC/R 和时间统计都放到外部评估器中，避免候选程序在内循环中直接优化评估器实现细节。

对每个候选 $h$ 和图 $G$，评估器记录：

$$
\mathcal{E}(h,G)
= \left(R(h,G), \mathrm{AUC}_{\mathrm{cNBI}}(h,G), \mathrm{Time}(h,G)\right).
$$

$R$ 越低越好，用于标准瓦解质量；$\mathrm{AUC}_{\mathrm{cNBI}}$ 越高越好，用于过程碎裂质量；$\mathrm{Time}$ 越低越好，用于候选算法运行成本。多图结果先在每个图上计算，再按图平均。本文明确区分最终启发式 runtime 和搜索框架成本：前者只统计候选算法在图上输出删除序列的时间，不包含 LLM 生成成本；后者统计 prompt elapsed time 和 candidate validation time，用来回答搜索过程是否更耗预算。

## 4.3 HDA-relative fracture credit

HAST 的树节点是一个候选启发式函数，树边是 LLM 对父候选的一次代码修改。设 root heuristic 为 $h_0$，候选为 $h$。最基本的相对碎裂信用为：

$$
\mathrm{FAC}(h)
= \mathrm{AUC}_{\mathrm{cNBI}}(h)-\mathrm{AUC}_{\mathrm{cNBI}}(h_0).
$$

在搜索阶段，为节省评估成本并更早发现有潜力的候选，HAST 可以使用 early fracture proxy：

$$
\mathrm{EarlyFAC}(h)
= 0.45\Delta \mathrm{cNBI}@20\%
+ 0.35\Delta \mathrm{NCC}@20\%
+ 0.20\Delta(-\mathrm{GCC})@20\%.
$$

这里的 $\Delta$ 都表示候选相对 root 的差值。该设计并不把 cNBI 写成唯一目标，而是把 root 已经能做到的高阶度删除能力从候选信用中扣除。对于从 degree/HDA root 出发的 LLM 修改，这一点尤其重要：如果不做相对化，搜索器会倾向于保留任何绝对分数高的候选，即使其新增代码只是搭在 root backbone 上而没有真实贡献。

## 4.4 Time-aware credit

FAC 能发现更强的碎裂候选，但 Observation 3 表明它会把自由代码搜索推向慢扫描。HAST 因此引入 time-aware fracture credit，把计算代价放进搜索期信用，而不是等最终表格再做后验筛选：

$$
\mathrm{FAC\mbox{-}T}(h)
= \mathrm{FAC}(h) - \lambda \cdot \phi(\mathrm{Time}(h)) - \gamma \cdot \psi(h).
$$

其中 $\phi(\cdot)$ 是时间惩罚，$\psi(h)$ 是风险惩罚，用于惩罚候选中的无界 two-hop 扫描、频繁全图重算、过宽邻域枚举和不稳定结构。已有实现中，proxy time 小于等于 1.2s 的候选优先，proxy time 大于 1.8s 的候选受到强惩罚；搜索图平均时间过高的候选也会被降权。这个门控不等价于简单删掉慢候选，而是改变树搜索的父节点选择和后续 mutation 方向。

实验中，HAST-FAC-T online #24 达到 mean auc-cNBI 350.480、mean time 2.109s，相比 FAC-only 的强慢候选更实用。但 FAC-T 仍不是最终答案，因为自然语言代码空间太宽，LLM 仍可能写出刚好通过时间门、但结构冗余的局部扫描。因此 HAST 还需要把自由搜索日志进一步压缩成有界候选语言。

## 4.5 日志归纳的有界候选语言

HAST 的 bounded candidate language 来自自由搜索日志，而不是预先手写一个完全固定的人工启发式。日志显示，高信用候选反复保留 residual degree backbone，并加入 frontier、weak-tie、boundary 和 redundancy-aware 信号；低效候选则反复出现无界二跳、嵌套邻域枚举、频繁 connected components 重算和过宽局部刷新。HAST 将这些观察转化为四类约束：

1. 保留 residual degree backbone，不删除基础度信号。
2. 允许 frontier、weak-tie、boundary、redundancy-aware 等局部碎裂特征。
3. 对邻居枚举、二跳扩展和局部更新设置上界，例如 `CAP_N`、`CAP_2` 和 `update_cap`。
4. 禁止候选在内循环中频繁执行全图 connected components 计算。

这种语言把 LLM 发现的慢机制压缩为可运行的局部 proxy。它不是后处理小技巧，而是 HAST 解决 complexity drift 的关键机制：FAC 负责找到“什么样的局部信号可能有效”，FAC-T 负责让搜索意识到成本，有界语言负责把这些信号落成可执行、可检查、可限界的候选。

## 4.6 算法流程与最终候选命名

```text
Input:
  benchmark graphs D
  root heuristic h0
  LLM generator M
  search budget B

Initialize search tree T with h0

Stage I: free credit-aware search
  for t = 1 ... B:
      select parent from T using FAC/FAC-T and exploration score
      ask M to mutate parent code
      if interface or sandbox check fails:
          continue
      evaluate candidate on search graphs
      compute root-relative fracture credit
      log code features, quality, time, and failure mode
      add candidate to T

Stage II: induce bounded candidate language
  summarize search logs:
      keep high-credit local signals
      prune low-potential or unstable families
      bound two-hop scans and local updates
      forbid frequent global component recomputation

Stage III: bounded search and selection
  instantiate or search bounded candidates
  evaluate candidates on full benchmark graphs
  return Pareto frontier over auc-cNBI and runtime
```

正文只使用两个最终候选名：HAST-Final-Q 和 HAST-Final-S。二者不是两个新框架，而是同一个 HAST 搜索过程输出的两个最终点：HAST-Final-Q 偏质量，HAST-Final-S 偏速度。内部编号只用于复现映射。

| 论文名 | 内部编号 | 含义 | 是否新框架 |
|---|---|---|---|
| HAST-Final-Q | FAST21-cap24 | 质量优先最终点 | 否 |
| HAST-Final-S | BT-n16-t8-u24 | 速度优先最终点 | 否 |

# 5 实验

## 5.1 实验设计

本节实验回答四个问题：HAST 的最终候选是否位于更实用的质量-时间折中区域；HAST 的机制链是否必要；HAST 是否只是花更多搜索预算堆出来；最终候选为什么有效。为避免把未做实验写成过强结论，本文统一采用“同一 evaluation harness 下的经验比较”口径，不宣称严格 equal-budget SOTA，也不宣称在所有图和所有指标上最优。

**数据集。** 实验在 12 个 benchmark graphs 上进行，覆盖真实网络和一个 synthetic benchmark。每个方法在每个图上输出完整节点删除序列，评估器沿相同删除比例网格计算 GCC/R 曲线和 cNBI 曲线。多图均值先对每个图计算对应指标，再按图平均；主表中的 `datasets` 列用于说明某些 Python fallback 是否覆盖完整 12 图。

**指标。** 主指标包括 $R$、auc-cNBI 和 runtime。$R$ 越低越好，表示达到瓦解阈值所需删除比例或对应的最终拆解质量；auc-cNBI 越高越好，表示删除过程中产生了更充分、更均衡的 residual fragmentation；runtime 越低越好，表示候选启发式本身更轻量。GCC 曲线和 cNBI 曲线同时报告，因为二者回答不同问题：GCC/R 保证标准瓦解目标，cNBI 揭示 same-GCC 下的过程碎裂差异。

**时间口径。** 本文区分 final heuristic comparison 和 search framework comparison。前者比较候选算法在同一图上输出删除序列的运行时间，不包含 LLM prompt、代码生成或搜索日志成本；后者比较不同自动搜索框架的候选数量、有效率、prompt elapsed time、candidate validation time 和 total logged search time。因此，表 5.2 和图 5.2 回答“最终算法是否好用”，表 5.5 和图 5.5 回答“搜索过程是否烧预算”。

**基线组。** 本文比较三类方法。第一类是 classic/static baselines，包括 DC、HDA、CoreHD、KCore、CLUC、CI 等。第二类是 algorithm-found references，包括 ERA-like、FunSearch-like、Clade-AHD-like、MCTS-AHD-like、AlphaEvolve-like。第三类是 Python strong baselines 或 fallback evidence items，包括 NDC、NCDC、NDJC、BPD/MinSum-fallback、GND-py、VE-py 和 LGD variants。HAST 输出为 HAST-Final-Q 和 HAST-Final-S。

**强基线复现限定。** BPD 和 MinSum 在相关工作中应作为不同方法讨论；但当前本地 `BPD/MinSum-fallback` 由同一个 fallback 函数映射，不能写成官方 BPD 和官方 MinSum 的两个独立复现。GND-py、VE-py、NDJC 等 Python fallback 若在大图超过 3600s 或无缓存，则只报告 timeout 或 unavailable 状态，不生成伪曲线，也不把缺失结果解释为这些方法质量差。这个限定不会削弱主结论，反而避免把实现语言、缓存状态和算法能力混在一起。

**搜索预算公平性。** 所有 LLM 生成候选均使用 GPT-5.5，reasoning effort 设为 `none`，temperature 设为 0.2；每次调用生成一个候选程序，不使用 self-consistency、majority voting 或人工二次改写。候选程序必须暴露同一接口：输入图 $G$，输出完整节点删除序列；搜索过程中不允许读取测试标签、外部数据或已有测试曲线。所有候选通过同一验证器，验证内容包括 Python 语法、函数可调用性、返回类型、重复节点、缺失节点、运行异常和超时；失败候选计入 invalid rate，不从分母中删除。搜索阶段使用固定的候选预算上限：通用搜索参照最多约 500 个候选，HAST free search 为 300 个候选，HAST bounded search 为 100 个候选，HAST online check 为 60 个候选。搜索期只使用 proxy/validation 图上的反馈信号；最终候选冻结后再进入 12 图统一 evaluation harness。搜索成本统计包括 candidates、valid rate、mean search time per candidate 和 total logged search time，具体数值在 5.2.4 报告；最终算法 runtime 与离线搜索成本分开报告。

**随机性与统计边界。** 大多数传统启发式和 HAST-Final-Q/S 的执行是确定性的；主要随机性来自 LLM 搜索过程、候选生成和可能的 tie-breaking。当前主表报告的是固定最终候选在 12 图上的均值，不是多随机种子显著性检验。因此，正文避免使用 “statistically significant” 或“显著优于”这类尚未由多 seed 支撑的措辞。后续可在附录补充 bootstrap over graphs 的置信区间或多次 LLM 搜索种子。

## 5.2 结果与分析

### 5.2.1 HAST 在质量-时间坐标里达到了最佳均衡

主结果首先直接展示所有方法在质量-时间坐标中的位置。横轴按运行时间反向显示，右侧更快；$10^0$ 到 $10^{-2}$ 的视觉距离被压缩，避免把亚秒级差异夸大。HAST-Final-Q/S 用带实线框的星形标出，其他搜索生成方法用彩色实心虚线圆，强基线用彩色圆点，传统方法用灰色圆点。这个图的核心不是证明 HAST 在单一指标上第一，而是说明它位于高 auc-cNBI 与低 runtime 同时成立的区域。

![All method quality runtime](../artifacts/figures/fig13_12graph_quality_runtime_all_methods.png)

| method | datasets | mean R ↓ | mean auc-cNBI ↑ | mean time ↓ | top1 auc | top3 auc |
|---|---:|---:|---:|---:|---:|---:|
| FunSearch-like | 12 | 0.429 | 374.870 | 27.330s | 5 | 8 |
| Clade-AHD-like | 12 | 0.373 | 373.101 | 51.004s | 2 | 6 |
| ERA-like | 12 | 0.369 | 359.868 | 9.785s | 0 | 6 |
| **HAST-Final-Q** | 12 | 0.380 | 358.066 | **1.008s** | 0 | 1 |
| **HAST-Final-S** | 12 | 0.382 | 356.253 | **0.556s** | 0 | 0 |
| HDA | 12 | 0.438 | 219.220 | 3.381s | 0 | 0 |
| CoreHD | 12 | 0.436 | 214.539 | 0.096s | 0 | 0 |
| NCDC | 12 | 0.362 | 372.402 | 240.258s | 1 | 5 |
| BPD/MinSum-fallback | 12 | 0.398 | 338.511 | 152.064s | 0 | 0 |

该结果支持一个边界清楚的结论：HAST 的优势是综合效益，而不是所有单项第一。FunSearch-like 和 Clade-AHD-like 的 mean auc-cNBI 更高，但运行时间分别达到 27.330s 和 51.004s；CoreHD 的 mean time 极低，但 mean auc-cNBI 只有 214.539；ERA-like 的 auc-cNBI 与 HAST-Final-Q/S 接近，但运行时间更高。HAST-Final-Q 保留 ERA-like 约 99.5% 的 auc-cNBI，同时快约 9.7 倍；HAST-Final-S 保留约 99.0% 的 auc-cNBI，同时快约 17.4 倍。这个位置最符合本文要找的“有用的强方法”：不是最慢的最高分点，也不是最快的弱方法，而是在高质量和轻量运行之间取得稳定折中。

### 5.2.2  只看高质量候选时，HAST 是快的那个

为了避免主图被大量弱 baseline 稀释，图 5.3 只聚焦高质量候选区域。这个视角回答一个更尖锐的问题：如果只比较已经达到较高 auc-cNBI 的候选，HAST 是否仍有优势？结果显示，HAST-Final-Q/S 的 auc-cNBI 接近 ERA-like，并明显快于 FunSearch-like 和 Clade-AHD-like；CoreHD 和 DC 虽快，却不属于高过程碎裂区域。

<img src="../artifacts/figures/fig17_hast_quality_speed_panel.png" alt="HAST quality speed panel" style="zoom: 33%;" />

这张图的叙事重点是“near-strong quality with lightweight runtime”。HAST-Final-Q/S 没有宣称超过所有 search-found candidates 的 auc-cNBI；它们展示的是 HAST 的有界候选语言能够保留通用 LLM 搜索发现的主要碎裂收益，同时把候选算法运行时间降到接近轻量启发式的范围。对于实际网络瓦解应用，这种质量-速度折中比单纯追求最高 auc-cNBI 更有用。

### 5.2.3  GCC 和 cNBI 曲线共同说明 HAST 没有偏离任务目标

图 5.4 和图 5.5 展示 12 图上的 GCC 曲线和 cNBI 曲线。GCC 曲线证明 HAST 没有偏离标准网络瓦解评价；cNBI 曲线展示 HAST 的优势主要体现在过程碎裂。两组曲线应与表 5.2 一起读，而不应把任何单张曲线图当成全部结论。

![12 graph GCC curves](../artifacts/figures/fig10_gcc_curves_12graphs.png)

![12 graph cNBI curves](../artifacts/figures/fig11_cnbi_curves_12graphs.png)

HAST 不是每个图、每个指标都第一。NCDC 在可完成的 10 个图上可以获得更高局部指标，但运行时间与覆盖状态限制了直接比较；CoreHD 在运行时间上更快，但过程碎裂质量明显不足。HAST-Final-Q 的 mean auc-cNBI 为 358.066、mean time 为 1.008s；HAST-Final-S 的 mean auc-cNBI 为 356.253、mean time 为 0.556s。它们的优势来自质量-速度均衡，而不是用一个指标遮蔽另一个指标。

### 5.2.4 框架搜索本身也更轻

一个自然质疑是：HAST 的最终候选是否只是花更多搜索成本堆出来的？为回答这个问题，本文统计自动搜索框架的候选数量、有效率、平均候选搜索时间和 total logged search time。logged search time 定义为 prompt elapsed time 加 candidate validation time，根节点不计入统计。

![Framework search time](../artifacts/figures/fig20_framework_search_time.png)

| method | candidates | valid rate | mean search time / candidate | total logged search time |
|---|---:|---:|---:|---:|
| ERA-like | 499 | 79.6% | 76.9s | 10.66h |
| FunSearch-like | 499 | 98.4% | 33.1s | 4.58h |
| Clade-AHD-like | 499 | 72.1% | 81.6s | 11.31h |
| MCTS-AHD-like | 499 | 75.6% | 80.0s | 11.09h |
| AlphaEvolve-like | 499 | 89.4% | 27.7s | 3.84h |
| **HAST free search** | 300 | 99.3% | **27.7s** | **2.31h** |
| **HAST bounded search** | 100 | 100.0% | **31.2s** | **0.87h** |
| **HAST online check** | 58 | 96.6% | **27.8s** | **0.45h** |

该结果说明，HAST 不仅最终候选运行快，搜索过程也没有比 ERA-like、Clade-AHD-like 或 MCTS-AHD-like 更重。需要强调的是，这不是严格 equal-budget 实验，因为各框架候选数量和 wall-clock 不完全一致；但在已记录预算下，HAST 的搜索日志成本低于多个通用 LLM 搜索参照，同时最终候选落在更好的质量-时间折中区域。这个证据足以支撑本文的谨慎主张：HAST 不是单靠更大搜索预算换取最终结果。

## 5.3 消融实验

本节需要在新版 HAST 三阶段实现完成后重跑。保留的实验问题是：相对碎裂信用、时间信用、日志归纳边界和最终 Pareto 选择分别贡献了什么。旧版消融表和旧图已经从 `main` 的 paper-facing artifacts 中移除，因为它们对应上一版阶段划分，不能直接支撑当前的 `Stage 1: cost-aware free search -> Stage 2: log-induced bound induction -> Stage 3: bounded guided search` 设计。

新版消融将至少包含以下组别，并使用同一模型、同一候选接口、同一验证器和同一 12 图 full validation：

| Ablation group | 预算/候选 | 状态 | 目标 |
|---|---:|---|---|
| Free search without relative credit | 100 candidates | 待重跑 | 检验没有相对碎裂信用时的搜索质量 |
| Relative credit without time penalty | 100 candidates | 待重跑 | 检验相对信用是否带来慢扫描风险 |
| Cost-aware free search | 100 candidates | 待重跑 | 检验 time penalty 是否改善质量-时间折中 |
| Bounded guided search | 200 candidates | 待重跑 | 检验日志归纳 bounds 是否提升有效率和可复现性 |
| Full HAST-Final-Q/S | Pareto frontier | 待重跑 | 给出最终质量点和速度点 |

## 5.4 Case Study：HAST-Final-S 的机制与搜索来源

本节 case study 将在新版 HAST 主实验完成后填入。为了避免把旧版候选误写成新版最终算法，当前版本只保留 case study 的分析模板：最终应说明具体是哪一个 `HAST-Final-S` 或 `HAST-Final-Q` 候选、它来自 Stage 3 的哪个 bounded family、其打分项和约束参数是什么，以及本次搜索一共生成多少候选、多少通过验证、各阶段消耗多少 LLM 调用和评估时间。

从论文层面看，HAST-Final-S 的打分结构可抽象为如下局部模板：

$$
s_t(v)=w_d(\rho_t)d_t(v)+w_f(\rho_t)f_t(v)+w_w(\rho_t)q_t(v)+w_b(\rho_t)b_t(v)-w_r(\rho_t)r_t(v),
$$

其中 $\rho_t=t/|V|$ 表示删除进度，$d_t(v)$ 是残余度信号，$f_t(v)$ 与 $q_t(v)$ 分别刻画 frontier 和 weak-tie 倾向，$b_t(v)$ 表示受限二跳边界信号，$r_t(v)$ 表示邻域冗余惩罚。这里的公式用于解释算法构成，不声称它是唯一实现形式。对应到机制含义，residual degree 保留 HDA 类方法已经验证有效的一阶局部强信号；frontier/weak-tie 偏向连接脆弱邻居或低度邻居的节点；two-hop boundary 在 `cap_2` 范围内估计节点删除后可能影响到的外部边界；redundancy penalty 降低对高度重叠局部团簇的重复攻击；phase weights 则允许算法在早期更依赖 degree，在中后期增加碎裂压力。这样的设计解释了为什么 HAST-Final-S 既不像纯 degree 规则，也不像自由搜索中出现的慢速全局扫描。

组件 knockout、早期删除节点特征和搜索总数据表均需要重跑后再填。新版表格将从 `main/runs/` 或 `main/artifacts/source_tables/` 中导出，不再复用旧版 `BT-n16-t8-u24` 记录。

## 5.5 扩展性和复现状态

### 5.5.1 扩展性实验

扩展性实验用于检查 HAST-Final-Q/S 在比 12 图 benchmark 更大的合成图上是否仍保持轻量。实验不重新运行 LLM 搜索，而是固定已经选出的 final heuristic，在新生成的合成图上直接输出删除序列。为避免把质量评估和极限排序压力测试混在一起，本节分成两个口径：$500$ 到 $10k$ 节点运行完整 evaluation harness，记录 $R$、auc-cNBI 和 runtime；$50k$ 到 $1000k$ 节点只记录 ordering runtime，不计算曲线指标。这里的 runtime 是 final heuristic runtime，不包含 prompt、候选生成、相对信用计算或离线搜索成本。需要注意的是，CoreHD 使用当前评估器里的快速增量实现，因此标为 CoreHD-fast；HDA 同时报告原始重扫版本 HDA-original 和快速维护版本 HDA-fast。

合成图设置覆盖四类网络：powerlaw、Erdos-Renyi、stochastic block model 和 Watts-Strogatz small-world。完整评估规模为 $n\in\{500,1000,5000,10000\}$，每个图族使用 seed 42、43、44；因此每个方法、每个规模对应 12 次运行。对比方法在本节标注为 HDA-original、HDA-fast、CoreHD-fast、HAST-Final-Q 和 HAST-Final-S。最终结果按 $n$ 聚合：先在每张生成图上记录指标，再对同一规模下的不同图族和 seed 求平均。因此表中的数字不是单张图的 best case，而是当前合成扩展设置下的平均结果。

| family | n | seeds | edge range |
|---|---:|---|---:|
| ER | 500 / 1000 / 5000 / 10000 | 42,43,44 | 1485-1545 / 2949-3074 / 14869-15117 / 29855-30137 |
| powerlaw | 500 / 1000 / 5000 / 10000 | 42,43,44 | 943-948 / 1889-2067 / 11030-11178 / 21783-22324 |
| SBM | 500 / 1000 / 5000 / 10000 | 42,43,44 | 2780-2855 / 5590-5743 / 28568-28677 / 56977-57525 |
| WS | 500 / 1000 / 5000 / 10000 | 42,43,44 | 2000 / 4000 / 20000 / 40000 |

图 5.5 首先展示 $500$ 到 $10k$ 的完整评估结果。HAST-Final-S/Q 在这组合成扩展图上取得更低的 mean $R$ 和更高的 mean auc-cNBI，同时 runtime 仍保持秒级：在 10k 节点图上，HAST-Final-S 为 1.705s，HAST-Final-Q 为 1.998s。作为参照，HDA-original 为 5.785s，HDA-fast 为 0.151s，CoreHD-fast 为 0.166s。这个结果支持一个有限但重要的结论：有界局部模板在当前 10k 合成图设置下仍能保持可用运行时间，并且没有退化成自由搜索中常见的慢扫描。它不能外推为 million-scale 质量证明，也不能说明所有图族上都保持同样速度；质量结论仍以 12 图 benchmark 和主结果表为主。

![Scaling full evaluation](../artifacts/figures/scaling_full_eval_500_to_10k_unified.png)

| method | 10k mean R | 10k mean auc-cNBI | 10k mean time |
|---|---:|---:|---:|
| HDA-original | 0.702 | 402.744 | 5.785s |
| HDA-fast | 0.713 | 384.571 | 0.151s |
| CoreHD-fast | 0.701 | 402.855 | 0.166s |
| HAST-Final-S | 0.669 | 450.594 | 1.705s |
| HAST-Final-Q | 0.669 | 450.556 | 1.998s |

为了进一步记录更大合成图上的排序时间，本文补充一个独立的 runtime-only stress test。该实验仍使用 powerlaw、ER、SBM 和 WS 四类合成图，规模覆盖 $n\in\{500,1000,5000,10000,50000,100000,1000000\}$，每个图族使用 seed 42、43、44。这里只有 ordering runtime，不调用完整 evaluation harness，不计算 $R$、auc-cNBI、GCC 或 cNBI 曲线；图生成时间也不计入算法时间。HDA-original 在 50k 的 powerlaw/ER/WS 上可完成，但 50k SBM 在 300s 上限内未完成；100k 和 1000k 因原始重扫复杂度过高，只记录 timeout guard，不参与均值曲线。图中用叉号标出 HDA-original 的 timeout/incomplete 规模。

![Runtime-only extreme scaling](../artifacts/figures/runtime_only_scaling_500_to_1000k_unified.png)

| method | 50k mean time | 100k mean time | 1000k mean time |
|---|---:|---:|---:|
| HDA-original | incomplete | timeout guard | timeout guard |
| HDA-fast | 0.974s | 2.479s | 39.952s |
| CoreHD-fast | 1.084s | 2.574s | 43.631s |
| HAST-Final-S | 10.371s | 24.415s | 387.187s |
| HAST-Final-Q | 12.132s | 28.499s | 458.151s |

这组极限记录的结论只限于时间可控性：在 50k、100k 和 1000k 合成图上，HDA-fast、CoreHD-fast、HAST-Final-S 和 HAST-Final-Q 均能在 12/12 个设置中输出完整删除序列；HAST-Final-S/Q 在百万节点图上的平均 ordering runtime 分别约为 387.2s 和 458.2s。与此同时，HDA-original 已经在 50k SBM 上触发 300s 上限，因此本文不把原始重扫 HDA 写成可扩展基线，也不把 runtime-only 结果替代质量评估。

### 5.5.2 强基线复现

强基线复现状态也应透明报告。复现实验覆盖 12 图 benchmark 上的传统启发式、Python strong baselines 和 fallback evidence items。复现协议是：优先读取已经完成的删除序列缓存；若某个数据集-方法组合缺少缓存，则在本轮 filtered supplement 中只补跑被选中的项目；对于大图或较慢 Python fallback，单图/单方法设置最多 3600s 的上限。已经完成或已有缓存的 baseline 写入曲线和主表；超过上限或没有缓存且未被本轮选中的项目只在复现状态中报告，不生成伪曲线，也不把它们当作负结果强行比较。

状态含义如下。`cached` 表示使用已完成的曲线缓存；`ok` 表示本轮补跑成功；`cached_alias` 表示方法名通过别名映射到已有缓存；`timeout` 表示达到 3600s 上限仍未完成；`not_selected_no_cache` 表示该方法在本轮过滤补跑中未被选中且没有可用缓存。特别地，`BPD/MinSum-fallback` 是本地 `python_baselines` 脚本中的 fallback evidence item，相关 raw method 名称会映射到同一个本地函数，因此本文只把它报告为一个 fallback 项，而不声称完成了彼此独立的官方 BPD 与 MinSum 复现。这样的写法保留强基线证据，也明确标出复现边界。

| status | count |
|---|---:|
| cached | 88 |
| ok | 3 |
| cached_alias | 3 |
| timeout | 3 |
| not_selected_no_cache | 11 |

Held-out 小验证图选择实验显示，少量 validation graphs 对候选选择有帮助，但相关性不完美。当使用 1、2、3 个 validation graphs 时，mean Spearman 分别为 0.549、0.481 和 0.418；beat ERA-like rate 分别为 0.833、0.788 和 0.773。这说明 HAST 候选具有一定跨图选择稳定性，但仍应在正式提交版本中固定 proxy/validation/test split，避免把该结果写成严格泛化证明。

## 6 结论

本文提出 HAST，一个面向网络瓦解启发式发现的信用感知与有界 LLM 搜索框架。HAST 的核心观点是：在网络瓦解场景中，LLM 程序搜索的关键不只是扩大候选空间，而是把评价信号校准到候选代码的真实增量贡献，并限制候选通过无界慢扫描获得表面收益。实验表明，在当前 12 图 benchmark 和统一 evaluation harness 下，HAST-Final-Q 与 HAST-Final-S 分别保留 ERA-like 约 99.5% 和 99.0% 的 auc-cNBI，同时实现约 9.7 倍和 17.4 倍的运行加速。结合 motivation observations、消融、搜索成本统计和机制解释，本文的证据支持一个边界清晰的结论：信用感知与有界生成可以把 LLM 发现的网络瓦解启发式推向更实用的质量-时间折中。

## 参考文献

[1] Flaviano Morone, Hernán A. Makse. Influence maximization in complex networks through optimal percolation. Nature, 2015. DOI: 10.1038/nature14604.

[2] Alfredo Braunstein, Luca Dall'Asta, Guilhem Semerjian, Lenka Zdeborová. Network dismantling. Proceedings of the National Academy of Sciences, 2016. DOI: 10.1073/pnas.1605083113.

[3] Lenka Zdeborová, Pan Zhang, Hai-Jun Zhou. Fast and simple decycling and dismantling of networks. Scientific Reports, 2016. DOI: 10.1038/srep37954.

[4] Xiao-Long Ren, Niels Gleinig, Dirk Helbing, Nino Antulov-Fantulin. Generalized network dismantling. Proceedings of the National Academy of Sciences, 2019. DOI: 10.1073/pnas.1806108116.

[5] Changjun Fan, Li Zeng, Yizhou Sun, Yang-Yu Liu. Finding key players in complex networks through deep reinforcement learning. Nature Machine Intelligence, 2020. DOI: 10.1038/s42256-020-0177-2.

[6] Marco Grassia, Manlio De Domenico, Giuseppe Mangioni. Machine learning dismantling and early-warning signals of disintegration in complex systems. Nature Communications, 2021. DOI: 10.1038/s41467-021-25485-8.

[7] Jiazheng Zhang, Bang Wang. Dismantling Complex Networks by a Neural Model Trained from Tiny Networks. Proceedings of the 31st ACM International Conference on Information and Knowledge Management, 2022. DOI: 10.1145/3511808.3557290.

[8] Bernardino Romera-Paredes et al. Mathematical discoveries from program search with large language models. Nature, 2024. DOI: 10.1038/s41586-023-06924-6.

[9] Haoran Ye, Jiarui Wang, Zhiguang Cao, Federico Berto, Chuanbo Hua, Haeyeon Kim, Jinkyoo Park, Guojie Song. ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution. Advances in Neural Information Processing Systems, 2024. DOI: 10.48550/arXiv.2402.01145.

[10] Rui Zhang, Fei Liu, Xi Lin, Zhenkun Wang, Zhichao Lu, Qingfu Zhang. Understanding the Importance of Evolutionary Search in Automated Heuristic Design with Large Language Models. Parallel Problem Solving from Nature, 2024. DOI: 10.1007/978-3-031-70068-2_12.

[11] Pham Vu Tuan Dat, Long Doan, Huynh Thi Thanh Binh. HSEvo: Elevating Automatic Heuristic Design with Diversity-Driven Harmony Search and Genetic Algorithm Using LLMs. AAAI Conference on Artificial Intelligence, 2025. DOI: 10.1609/aaai.v39i25.34898.

[12] Alexander Novikov et al. AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv, 2025. DOI: 10.48550/arXiv.2506.13131.

[13] He Yu, Jing Liu. Automatically optimizing heuristics for robust scale-free network design via large language models. Scientific Reports, 2025. DOI: 10.1038/s41598-025-25031-2.

[14] Google Research. An AI system to help scientists write expert-level empirical software. Nature, 2026. DOI: 10.1038/s41586-026-10658-6. arXiv:2509.06503.


