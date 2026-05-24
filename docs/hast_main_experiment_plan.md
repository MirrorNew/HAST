# HAST-Lite-Full 主实验计划

## 1. 目标

本文件记录 `HAST2026/07_source_code/main/` 的完整实施计划、实验预算、数据策略和论文图表更新清单。该目录应成为后续 HAST 实验的独立主项目：代码、baseline、metric、配置、文档和本地数据都在该目录内闭环，不再依赖旧的 `research/e26f` 路径。

## 2. 固定框架

主方法命名为 `HAST-Lite-Full`。三阶段为：

1. **Stage 1: cost-aware free search**
   - root: `HDA-original`
   - LLM candidate budget: 300
   - credit: `GCC/R + HDA-relative FAC + FAC-T time penalty`
   - selection: unified proxy + full validation 后选 Pareto frontier

2. **Stage 2: log-induced bound induction**
   - LLM induction budget: 10 calls
   - 输入：Stage 1 search log、候选有效率、运行时间、relative credit、family 分布、top candidates 代码特征
   - 输出：结构化 bound policy JSON
   - 注意：Stage 2 不生成候选算法，只归纳 Stage 3 的搜索边界

3. **Stage 3: bounded guided search**
   - LLM candidate budget: 200
   - 起点：Stage 1 输出中最优 family 或 top candidates
   - 约束：Stage 2 归纳出的 allowed signals、cap bounds、update bounds、forbidden patterns
   - 目标：保留 motivation 核心，同时减少自由生成带来的慢扫描和无效候选

最终选择：

- 对 Stage 3 有效候选做 full validation。
- 根据 `auc_cNBI`、`R`、`time_s` 的 Pareto frontier 选择：
  - `HAST-Final-Q`: 偏质量
  - `HAST-Final-S`: 偏速度

## 3. LLM 设置

- model: `GPT-5.5`
- reasoning effort: `none`
- temperature: `0.2`
- Stage 1: 300 candidate calls
- Stage 2: 10 induction calls
- Stage 3: 200 candidate calls

本文不声称这是严格 equal-budget SOTA comparison，而是 transparent logged-budget comparison：同一模型、同一候选接口、同一验证器，并完整记录候选数、有效率、搜索时间和最终验证结果。

## 4. Baseline 与指标

主项目内置 baseline：

- `HDA-original`: 原始残余图重扫 HDA。
- `HDA-fast`: lazy-heap HDA，仅在明确标注 fast baseline 的 runtime/scaling 场景使用。
- `CoreHD-fast`: lazy-heap CoreHD-fast，仅在明确标注 fast baseline 的 runtime/scaling 场景使用。
- `DC`: static degree centrality baseline。

指标：

- `GCC / R`
- `NCC`
- `cNBI`
- `AUC-cNBI`
- final ACC/NCC/cNBI
- algorithm runtime
- candidate valid rate
- search time

## 5. 需要重做的实验

### Observation 2: relative fragmentation credit is important

实验组：

- `GCC/R-only`
- `Absolute-cNBI`
- `Relative-Delta-cNBI`

每组 100 candidates，使用同一 root、同一候选接口、同一验证器和同一 proxy graph 设置。目标是证明相对碎裂性评估比只看绝对碎裂性或 GCC/R 更适合搜索 credit。结论只能落在“更适合引导候选搜索”这一层，不扩展为绝对最优算法声明。

### Observation 3: bounded guidance improves efficiency

实验组：

- `Relative-Free`
- `CostAware-Free`
- `Bounded-Guided`

每组 100 candidates。主要报告有效率、候选运行时间、搜索时间和 AUC-cNBI。预期结论是有界引导生成更快、更可控；若 cNBI 有小幅下降，只能表述为质量-成本折中，不说全面质量优势。

### Main HAST

执行 `300 + 10 + 200` 预算：

- Stage 1 生成 300 个 cost-aware free search candidates。
- Stage 2 调用 10 次 LLM，从日志中归纳完整 bounds。
- Stage 3 在 bounds 内生成 200 个 bounded guided candidates。
- 对 Stage 3 有效候选做 full 12-graph validation。
- 选择 `HAST-Final-Q/S`。

### Ablation / Knockout

至少包含：

- 去掉 relative credit
- 去掉 time penalty
- 去掉 Stage 2 bounds
- 只用 Stage 1 top candidate
- bounded family 不同宽度设置

### Scaling

两个实验分开：

- 500 / 1k / 5k / 10k：完整指标。
- 500 / 1k / 5k / 10k / 50k / 100k / 1000k：runtime-only。

横坐标使用 log scale，seed 统一为 42/43/44。图例必须明确 `HDA-original`、`HDA-fast`、`CoreHD-fast` 的版本含义。

## 6. 论文图表更新清单

需要更新的 HAST 相关图表：

- `fig22_relative_credit_allocation_effect.png`
- `fig23_bounded_generation_controls_scan_cost.png`
- `Gemini-Framework.png`
- `fig13_12graph_quality_runtime_all_methods.png`
- `fig10_gcc_curves_12graphs.png`
- `fig11_cnbi_curves_12graphs.png`
- `fig20_framework_search_time.png`
- `fig18_hast_mechanism_compression.png`
- `fig14_component_knockout_ablation.png`
- `fig13_final_candidate_interpretability.png`
- `scaling_full_eval_500_to_10k_unified.png`
- `runtime_only_scaling_500_to_1000k_unified.png`

需要更新的文字：

- 第三章 Observation 2 和 Observation 3 的实验解释。
- 4.1 HAST 总览图和预算描述。
- 5.1 实验设计。
- 5.2 主结果。
- 5.5 扩展性。
- 5.7 case study：必须说明具体算法和搜索总数据。
- 5.8 扩展性和复现状态：必须说明实验设置、数据转移和复现边界。

## 7. 数据与 GitHub 策略

本地 `main/` 已保留：

- 12 图 benchmark edgelist。
- baseline record cache。
- tiny smoke-test fixture。

GitHub 上传：

- 代码
- 配置
- 文档
- 轻量 fixtures
- `.gitignore`
- `README.md`

GitHub 不上传：

- baseline 原始数据
- 大 benchmark raw data
- raw LLM logs
- full runs cache
- 大型 CSV 输出
- API key 或本地私密路径配置

## 8. 验收命令

```powershell
python scripts/smoke_test.py
python scripts/run_main_search.py --dry-run
python scripts/run_full_validation.py --datasets Powerlaw_500
python scripts/run_motivation_experiments.py
python scripts/run_scaling.py
python scripts/export_paper_artifacts.py
```

验收标准：

- smoke test 能运行。
- `main/` 不 import 旧 `research/e26f` 路径。
- baseline/metric/evaluator 代码在 `main/` 内自洽。
- 本地数据已转移，但 `.gitignore` 阻止上传。
- 本文件完整记录计划、预算、实验和论文更新清单。
