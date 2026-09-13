# SearchAgent 回归测试清单

## 1. QueryUnderstanding

- [x] 单元测试：LLM 输出能按 `QUERY_UNDERSTANDING_SCHEMA` 解析。
- [x] 单元测试：第一次问题理解不输出 `plan_rationale`、`steps`、`first_execution` 或 `expansion_hints`。
- [x] 单元测试：启用 LLM 时不使用规则结果覆盖 QueryUnderstanding。
- [x] 单元测试：LLM/Function Call 失败时显式失败，不静默回退关键词抽取。
- [ ] 真实 LLM 回归：`case2node4` 实体分类错误修复（待可用 API key 后执行）。

### QU-REAL-001: case2node4 实体分类

**目的**

确认 GPT-5 只阅读原始问题时，不再把实验室指标当作基因改变，且不丢失 FGFR 改变。

**输入来源**

`predictive_clinical_benchmark/benchmark_multinode.json` 中的 `case_2 / node_4` 原始问题。禁止输入 RoundMemory、AnswerMemory、Planner Context 或任何规则预抽取结果。

**只运行**

```text
submit_query_understanding
```

不运行 Planner、Execution Agent、Retriever、Rerank 和 Generate Agent。

**必须正确抽取**

- `cancer_type`：肺腺癌。
- `gene_alterations`：包含 `EGFR L858R` 和 `FGFR突变`。
- `drugs`：包含伏美替尼 120 mg 方案。
- `responses`：包含已发生的 `SD`。
- `toxicities`：包含肝损伤，并保留 `ALT 90` / `AST 118` 信息。
- `metastatic_sites`：至少表达脑膜/脑部转移；骨转移可标准化为骨或左侧髂骨。

**禁止结果**

- `gene_alterations` 中出现 `ALT 90`、`AST 118`、`ECOG` 或任何药物剂量。
- 丢失 `FGFR突变`。
- `toxicities` 为空。
- 输出 `plan_rationale`、`steps`、`first_execution`、`search_query` 或 `selected_tools`。
- Function Call 失败后产生规则 fallback 结果。

**通过标准**

1. 上述必须字段全部满足。
2. 禁止结果全部未出现。
3. Workflow Trace 中该测试只有一次 `function:submit_query_understanding`。
4. 保存原始 Function arguments、解析后 JSON、耗时和 token 用量。

## 2. Initial Planner

- [x] QueryUnderstanding 与 Initial Planner 使用两个独立 Function Call。
- [x] Planner 输出仅包含最小多步计划及预算，不重新抽取患者事实。
- [x] 初始计划限制为 2–3 个原子证据目标，不预先穷举所有分支。
- [x] 在 hard total budget 允许时，初始预算至少为 Replanner 预留 1 轮动态空间。
- [x] 第一步优先 direct-case 证据。
- [ ] 真实 LLM 回归：检查 case2node4 计划是否将直接伏美替尼脑膜转移疗效放在第一步。

## 3. Replanner 与 Expansion

- [x] 每个已执行步骤的边界允许 Replanner 检查是否需要追加新步骤。
- [x] `revise_plan + plan_changes` 可追加 1–2 个不重复的原子步骤。
- [x] 新增步骤会产生 `replanner_steps_added` 审计事件并至少获得一次可执行预算。
- [x] 已完成步骤追加新步骤后，直接推进到新步骤，不重复执行旧步骤。
- [x] 不再为“覆盖所有初始步骤”而在 Replanner 之前自动跳过当前步骤。
- [x] Replanner Prompt 明确知道主检索库是 Step3→Step2 的本地 case-report 语料，不是完整 PubMed/试验库。
- [x] 零命中后要求删除至少两个非必要精确限定，禁止通过追加未验证试验名来改写 Query。
- [x] 部分命中后保留已命中语义，每轮只修改一个 critical-gap 轴。
- [x] 精确剂量、时间、ORR/DCR、CSF转阴等优先放入 `rerank_goal`，不全部塞入召回 Query。
- [x] `REPLAN_SCHEMA` 支持 `case_ddi` 和 `mutation_drug`。
- [x] Schema 不再暴露未实现的 `therapeutic_class`。
- [x] `mutation_drug` 可从 `mutation_drug_map_min.json` 返回映射药物。
- [ ] 真实 LLM 回归：Replanner 能通过严格 Function Schema 选择 `case_ddi`。
- [ ] 真实 LLM 回归：Replanner 能通过严格 Function Schema 选择 `mutation_drug`。

## 4. Execution / Retrieval / Rerank

- [ ] 每个计划步骤的 Execution Agent 自行选择最多两个互补检索工具。
- [ ] Dense Query 使用自然语义病例描述，BM25 Query 使用紧凑精确词。
- [ ] Step3 先筛文档，每文档最多选取 1–2 条 Step2 细节。
- [ ] Rerank 前候选不超过 16 条，最多两个并发请求。
- [ ] Fetch 后证据保留可审计的 chunk ID、PMID、时间和来源层。

## 5. Evidence Review / Context / Generation

- [ ] 其他药物的毒性不得被总结为目标药物毒性。
- [ ] 弱类比证据必须标记 analog/risk，不得升级为直接支持。
- [ ] Evidence Review 拒绝后的证据不得进入 Generate Context。
- [ ] 时间过滤模式下，cutoff 之后证据不得进入 Rerank/Generate。
- [ ] 最终输出明确标记 `run_status`、`final_prediction_source`、`failed_stages` 和 `fallback_stages`。

## 6. Benchmark Smoke

- [ ] 10 题 smoke 全部能产生最终预测。
- [ ] 每题记录总耗时、各阶段耗时、LLM 调用数、Function Call 数和 Tool Call 数。
- [ ] 单题 LLM 调用不超过正式预算。
- [ ] 中断后能通过 checkpoint 续跑，不重复已完成病例。
- [ ] 汇总检索命中、后续轮救回率、最终分类指标和 LLM-as-Judge 指标。
