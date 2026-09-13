# SearchAgent 当前工作流

本文描述当前代码中的真实运行逻辑。正文尽量简洁；所有 LLM 组件的输入/输出 Schema 和 Prompt 保留完整。

## 1. 总体流程

```text
用户问题 + 历史 Session
  → Initial Planner（一次拆解多步原子检索计划）
  → Replanner（选择当前步骤的重试 / 推进 / 扩展 / 修订 / 停止）
  → RetrievalExecutionAgent（当前步骤内的受限 Function Tool Loop）
      ↔ Structured / Dense / BM25 / Hybrid / Rerank / Fetch 真实工具结果即时回传
  → submit_step_execution_report
  → LLMReranker
  → 自动反证检索
  → EvidenceReviewAgent
  → RetrievalLedger（每个真实搜索工具调用记为 Micro Retrieval Attempt）
  → StepMemoryAgent（一次总结当前 Step 内的微检索批次）
  → ReplannerMemoryAgent（轮次/token 双触发，LLM 重新提炼长短期记忆）
  → AnswerMemoryAgent
  → Replanner 读取长短期记忆和最新 StepMemory 后决定下一动作
  → ClinicalSafetyGate
  → 安全结果反写 AnswerMemory
  → CitationAgent
  → AnswerContextAgent（最终回答专用上下文提炼）
  → AnswerGenerator
  → 保存 Session
```

所有 LLM Agent 共用 `LLMClient`，默认模型为 `gpt-5`。除最终自然语言回答外，结构化组件均使用强制 Tool Call：`strict: true`、指定 `tool_choice`、`parallel_tool_calls: false`。代码仍对工具参数执行白名单和业务校验；接口异常时才进入规则回退。

## 2. 组件列表

| 组件 | 类型 | 功能 |
|---|---|---|
| Initial Planner | LLM Tool Call + 规则回退 | 将主问题拆成有序的多步原子检索计划 |
| Replanner | LLM Tool Call + 规则回退 | 根据本轮客观Memory决定重试、推进、扩展、修订或停止 |
| RetrievalRouter | LLM Tool Call + 规则回退 | 识别 query type/基础约束并执行 Replan 给定的单步计划 |
| RetrievalExecutionAgent | 真实 Function Tool Loop + Router回退 | 在当前原子步骤内动态调用检索、Rerank、Fetch 和本地知识表工具，提交步骤报告 |
| RetrievalTools | 算法/规则 | Structured、Dense、BM25、Hybrid 检索 |
| Local Knowledge Tools | 确定性规则 | 基于 `data/` 的药名映射、PK、DDI、mutation-drug 和 case-DDI 结构化查询与 query expansion |
| LLMReranker | LLM Tool Call + 原排序回退 | 对 Hybrid 候选重新排序 |
| Counter-evidence Search | 规则 | 自动补充反证、进展、耐药和风险检索 |
| EvidenceReviewAgent | LLM Tool Call + 规则回退 | 审核证据相关性、支持、反证和风险 |
| RetrievalLedger | 确定性规则 | 逐个保存 Micro Retrieval Attempt 的 query、工具、约束、命中数和错误 |
| StepMemoryAgent | LLM Tool Call + 规则回退 | 聚合当前 Planning Step 的微检索 Ledger，提取信息增益、完成度、关键/可选缺口 |
| ReplannerMemoryAgent | LLM Tool Call | 重写 Replanner 长短期记忆；提炼时遗忘重复、已解决和低价值内容 |
| AnswerMemoryAgent | LLM Tool Call + 规则回退 | 跨轮维护具体 claim |
| MainAgentLoop | 规则 | 编排多轮调用和停止条件 |
| ClinicalSafetyGate | 硬规则 + LLM Tool Call | 审核 claim 并生成安全改写 |
| CitationAgent | 确定性规则 | claim 到数据库源文件的追溯 |
| AnswerContextAgent | LLM Tool Call + 精简回退 | 将长短期记忆、claims和安全约束提炼为最终回答上下文，不读取原始Tool Trace |
| AnswerGenerator | LLM + 模板回退 | 生成最终回答 |
| SessionMemoryStore | 规则 | 保存跨会话记忆 |

### Tool Call 协议

| 组件 | 强制调用的函数 |
|---|---|
| Initial Planner | `submit_multistep_plan` |
| Replanner | `submit_replan_decision` |
| RetrievalExecutionAgent | `structured_search` / `dense_search` / `bm25_search` / `hybrid_search` / `rerank_candidates` / `fetch_evidence` / `expand_query_with_knowledge` / `normalize_drug_names` / `lookup_pk_relations` / `lookup_ddi_rules` / `lookup_mutation_drug_relations` / `lookup_case_ddi_relations` / `submit_step_execution_report` |
| PlannerAgent | `submit_retrieval_plan` |
| LLMReranker | `submit_reranking` |
| EvidenceReviewAgent | `submit_evidence_review` |
| StepMemoryAgent | `commit_step_memory` |
| AnswerMemoryAgent | `commit_answer_memory` |
| ClinicalSafetyGate | `submit_safety_reflection` |

每个函数都有完整 JSON Schema，并设置 `additionalProperties: false`。Tool Call 解决 JSON 文本缺逗号、Markdown 包裹和额外说明等传输问题；确定性代码仍负责检查真实 `chunk_id`、工具白名单、证据引用、数值范围和临床安全底线。AnswerGenerator 输出 Markdown 正文，因此不使用 Tool Call。

---

## 2.1 多步计划与预算

Initial Planner 的输出新增：

```json
{
  "plan_rationale": "拆解理由",
  "initial_total_budget": 6,
  "steps": [
    {
      "step_id": "S1",
      "goal": "单一证据目标",
      "rerank_goal": "该步骤独立评分标准",
      "evidence_lane": "direct_case | analog_case | mechanism | ddi_safety",
      "success_criteria": ["完成标准"],
      "attempt_budget": 2
    }
  ]
}
```

`attempt_budget` 由 Planner 选择，只保留 1–16 的高位安全上限；初始计划只允许 2–5 个高层证据目标；`initial_total_budget` 由 Planner 选择，最高 128 且不得超过运行时硬上限。如果接口忽略 Schema 并返回超过 5 步，整份过度规划会被拒绝。具体药物或耐药机制分支只能在 StepMemory 中有真实新证据后，由 Replanner 动态插入。

Initial Planner 原始 System Prompt：

```text
你是临床检索 Initial Planner。初始计划只能是 2-5 个高层证据目标组成的最小骨架，不是研究大纲或最终回答目录。每一步只解决一个证据目标，并给出专属 rerank_goal 和 1-4 轮 attempt_budget。优先用“直接匹配证据→根据已发现机制/方案追加证据→反证→安全”的骨架。除非用户问题或已有证据明确提到，不得预先为每个具体药物、候选耐药机制或毒性各建一步；这些分支应由 Replanner 在真实检索证据触发后动态插入。“总结、综合、撰写回答”不是检索步骤。设置足以覆盖骨架的 initial_total_budget，避免无限重试。必须通过 submit_multistep_plan 提交。
```

Replanner 每轮输入中新增 `budget_state`：

```json
{
  "initial_total_budget": 4,
  "current_total_budget": 6,
  "hard_total_budget": 64,
  "rounds_used": 4,
  "rounds_remaining": 2,
  "active_step_attempt_budget": 2,
  "active_step_attempts_used": 1,
  "remaining_plan_steps": 3,
  "max_extension_per_request": 8
}
```

Replanner 输出新增两个字段：

```json
{
  "requested_budget_extension": 2,
  "budget_extension_reason": "剩余高价值安全步骤尚未执行"
}
```

扩容量由 Replanner 选择，Schema 只保留 0–32 的高位安全上限；实际批准量由编排器按“运行时单次扩容上限”和“全局硬上限”再次裁剪。

Replanner 原始 System Prompt：

```text
你是临床检索 Replanner。你读取既定计划、当前步骤、预算和 StepMemoryAgent 对一批 Micro Retrieval Attempts 的精炼总结。只由你决定重试、推进、扩展、修订或停止。completion_status=sufficiently_met 或 recommended_stop=true 时必须推进；minimally_met 时，只有 critical_gaps 非空且 marginal_value_of_more_search 较高才能继续；optional_gaps 不能单独成为继续理由。只有 StepMemory 中的真实新证据揭示会改变答案的具体药物、耐药机制或高风险问题时，才能 revise_plan。只有剩余预算不足以完成高价值步骤时才申请扩容。每次只能有一个原子目标。必须通过 submit_replan_decision 提交。
```

预算由三层组成：

- `--max-steps`：仅作为无 LLM/回退时的初始预算提示，不是 Planner 的最低预算。
- `attempt_budget`：Planner 给每个原子步骤的尝试预算；耗尽后该步骤记为 `budget_exhausted_partial` 并自动推进，不能独占总预算。
- `--max-total-steps`：不可突破的高位总硬上限，默认 128；`--max-budget-extension` 是单次扩容硬上限，默认 32。上限以内的具体值均由 Agent 决定。

当当前总预算恰好耗尽时，编排器仍允许 Replanner 做一次不执行检索的边界决策；只有扩容获批后才继续检索，否则以 `total_budget_exhausted` 停止。所有扩容和单步耗尽事件都写入 `budget_events`，最终状态写入 `budget_state` 和 `step_budget_status`，并显示在 `--print-workflow` 输出中。

完成度不再只用布尔值表示：`not_met` 继续补证；`minimally_met` 在存在 `critical_gaps` 且继续检索边际价值较高时可继续；`sufficiently_met` 或 `recommended_stop=true` 时推进。`optional_gaps` 不能单独作为继续理由。

LLM 上下文采用最小必要原则：Replanner 只读当前步骤最近 2 个 StepMemory 和最多 12 个高价值 claim；StepMemory 只读规则 Ledger 和精简证据审核；AnswerMemory 不接收完整 tool trace；AnswerGenerator 最多读 12 个 claim，每个 claim 最多 3 个精简 citation。

---

## 2.2 本地知识表工具与多策略扩展

`data/` 下的结构化表现在不再只作为 BM25 的被动文本源，还可以被 `RetrievalExecutionAgent` 直接调用，用于 query normalization、query expansion 和直接关系查询。

### 当前接入的本地知识表

| 文件 | 当前用途 |
|---|---|
| `data/all_drug_name_map.json` | 药物通用名 / 商品名 / alias 标准化与扩展 |
| `data/all_drug_pk_relation.json` | 药物 PK 关系、酶/转运体、related drug 扩展 |
| `data/all_ddi_rule.json` | DDI 机制规则查询与机制词扩展 |
| `data/mutation_drug_map_min.json` | mutation → 候选药物扩展 |
| `data/case_ddi.json` | 直接病例级 DDI 关系查询；也可把 interaction/outcome 词带回 query expansion |

### Execution Agent 可见的知识工具

- `expand_query_with_knowledge`
  - 输入：原始 query、drug_names、mutation_names、strategies
  - 输出：`expanded_query`、`added_terms`、命中的知识来源摘要
  - 行为：同一轮内可组合 `drug_alias`、`mutation_drug`、`case_ddi`、`pk_relation`、`ddi_rule`

- `normalize_drug_names`
  - 直接查药名映射表，返回同药物身份的标准名、商品名和 alias

- `lookup_mutation_drug_relations`
  - 直接查 mutation-drug 表，返回突变相关药物关系

- `lookup_pk_relations`
  - 直接查 PK 关系表，返回 enzyme/transporter、role、related drug、interaction type

- `lookup_ddi_rules`
  - 直接查 DDI 规则表，返回 mechanism rule

- `lookup_case_ddi_relations`
  - 直接查 `case_ddi.json`
  - 支持按 `drug_names`、`mutation_names`、`interaction_types` 过滤
  - 返回病例级 DDI 关系，而不是普通 rule 文本

### 自动 query override

当某一轮 agent 先调用 `expand_query_with_knowledge` 后，execution agent 会记住 `expanded_query`。之后如果同一轮继续调用：

- `bm25_search`
- `dense_search`
- `hybrid_search`

且查询仍是当前步骤的原始 `route_plan.search_query`，执行器会自动用 `expanded_query` 代替原 query。也就是说，模型不需要手工把扩展后的 query 再重复传一遍，扩展结果会真实参与后续检索。

### 多策略 expansion policy

`expand_current_step` 不再限制为“一轮只开放一个 expansion strategy”。当前策略如下：

- `ddi_safety` lane
  - 默认一次开放：`drug_alias` → `case_ddi` → `pk_relation` → `ddi_rule`
  - 如果当前决策显式请求 `mutation_drug`，也会一起开放

- `mutation_drug` query
  - 默认一次开放：`drug_alias` → `mutation_drug`

- 其他 lane/query
  - 默认只开放当前请求的单个策略

因此，DDI 问题现在允许 agent 在同一轮内先做药名标准化，再直接查病例级 DDI，再补 PK / 机制规则，最后带着扩展 query 回病例库检索。

---

## 3. PlannerAgent（Query识别与执行路由）

> 注：本节的 `submit_retrieval_plan` 现用于初始 query type/基础约束识别和 Router 兼容接口；多步检索控制由 `MultiStepPlanningAgent` 的 `submit_multistep_plan` 与 `submit_replan_decision` 负责。Replan 每轮只输出一个原子目标的 `search_query` 和 `rerank_goal`。

功能：读取主问题、历史轮次、已有 claim、失败方向和证据缺口，动态选择检索 query、工具和约束。

### 输入 Schema

```json
{
  "main_question": "用户主问题",
  "query": "当前候选查询",
  "planner_context": {
    "round_memories": [],
    "answer_memory": {
      "claims": [],
      "evidence_by_id": {},
      "informative_rounds": []
    },
    "current_state": {
      "confirmed_constraints": {},
      "missing_information": [],
      "next_retrieval_targets": []
    }
  }
}
```

### 输出 Schema

```json
{
  "query_type": "ddi | mutation_drug | mechanism | similar_case | treatment_advice",
  "search_query": "本轮实际执行的检索表达式",
  "selected_tools": ["工具名"],
  "constraints": {
    "cancer_type": "可选",
    "gene_alterations": [],
    "drugs": [],
    "responses": [],
    "toxicities": [],
    "metastatic_sites": [],
    "ddi_terms": []
  },
  "plan_rationale": "为何本轮这样检索",
  "targeted_information_gain": ["本轮要解决的具体缺口"],
  "avoid_repeating": ["不应重复的既往失败方向"],
  "stop_if": ["本轮结束后可停止的条件"]
}
```

### System Prompt

```text
你是肿瘤临床检索 PlannerAgent。你必须根据主问题、既往检索轮次记忆、已获得证据、失败方向和剩余缺口，动态制定本轮检索计划。你的目标是最大化对主问题的信息增益，同时主动寻找反证、安全风险和不可违背的临床约束。

必须通过 submit_retrieval_plan 函数提交计划，不要生成函数调用之外的回答。不得虚构数据库字段或工具。可用工具仅有：structured_search、dense_search、bm25_search、hybrid_search、fetch_evidence。
```

### User Prompt

```text
主问题：
{main_question}

当前候选查询：
{query}

Planner 上下文（包含既往逐轮记忆和累计回答记忆）：
{planner_context}

请输出：
{
  "query_type": "ddi | mutation_drug | mechanism | similar_case | treatment_advice",
  "search_query": "本轮实际执行的检索表达式",
  "selected_tools": ["工具名"],
  "constraints": {
    "cancer_type": "可选",
    "gene_alterations": [],
    "drugs": [],
    "responses": [],
    "toxicities": [],
    "metastatic_sites": [],
    "ddi_terms": []
  },
  "plan_rationale": "为何本轮这样检索",
  "targeted_information_gain": ["本轮要解决的具体缺口"],
  "avoid_repeating": ["不应重复的既往失败方向"],
  "stop_if": ["本轮结束后可停止的条件"]
}

要求：计划必须参考 Planner 上下文；优先补齐尚未解决且会改变最终回答的缺口；至少考虑一种反证或安全风险检索；search_query 不得混入回答内容。
```

规则：工具和 query type 必须在白名单内；非法字段被过滤；只要计划包含 Dense、BM25 或 Hybrid 检索，就强制追加 `fetch_evidence`，保证候选能回到结构化数据库形成可审核、可引用证据；LLM 失败时按关键词分类并使用固定工具组合。

### 3.1 RetrievalExecutionAgent

Replanner 确定当前原子步骤后，Execution Agent 获得该 Evidence Lane 允许的真实函数工具。模型每次只能调用一个工具；执行结果以 `role=tool` 和原 `tool_call_id` 立即回传，模型再决定下一个工具。相同函数+参数不允许重复，并受 `max_turns` / 检索次数 / Fetch次数限制。

```text
LLM tool_call
  → 参数校验
  → Python真实函数分发
  → role=tool结果回传
  → LLM继续调用
  → submit_step_execution_report结束当前Step
```

Direct Case Lane 只暴露病例检索、Rerank和Fetch。DDI Safety Lane 只在 Replanner 显式授权扩展时，才动态暴露 `normalize_drug_names` / `lookup_pk_relations` / `lookup_ddi_rules`。Execution Agent失败时回退到确定性 `RetrievalRouter.run`。

---

## 4. RetrievalTools 与检索规则

这些组件不使用生成式 LLM。

- Structured Search：按癌种、突变、药物、疗效和毒性等字段查询 SQLite。
- Dense Search：在 `case_semantic`、`event_semantic` 中做向量相似度检索。
- BM25 Search：`k1=1.5`、`b=0.75`。
- Hybrid Search：默认权重 `structured=0.35`、`dense=0.40`、`bm25=0.25`。
- `fetch_evidence`：根据 chunk ID 回结构化数据库读取完整证据和 citation metadata。

自动反证 query：

```text
ddi: 禁忌 相互作用 DDI CYP P-gp 严重不良反应
mutation_drug: 耐药 进展 无效 毒性 不良反应
mechanism: 不支持 冲突 反证 无效 进展
similar_case: 预后差 进展 毒性 不良反应
treatment_advice: 禁忌 进展 耐药 毒性 不良反应
```

---

## 5. LLMReranker

功能：对 Hybrid Search 的前 20 条候选做临床相关性重排。

### 输入 Schema

```json
{
  "query": "医生问题",
  "candidates": [
    {
      "id": "证据ID",
      "source": "hybrid",
      "hybrid_score": 0.0,
      "text": "最多500字"
    }
  ]
}
```

### 输出 Schema

```json
{
  "rankings": [
    {
      "id": "证据ID",
      "relevance_score": 0.0,
      "reason": "一句话理由"
    }
  ]
}
```

### System Prompt

```text
你是肿瘤临床检索助手。根据医生问题，评估每条候选证据与问题的临床相关性。必须通过 submit_reranking 函数提交排序结果。
```

### User Prompt

```text
医生问题：
{query}

候选证据（id 与摘要）：
{candidates}

请评估每条证据与问题的相关性，返回 JSON：
{
  "rankings": [
    {"id": "<证据id>", "relevance_score": 0.0-1.0, "reason": "<一句话理由>"}
  ]
}

要求：
1. relevance_score 越高表示越相关，范围 0-1。
2. rankings 必须覆盖所有候选 id，且每个 id 只出现一次。
3. 按 relevance_score 从高到低排序。
```

规则：分数裁剪到 0–1；遗漏候选补到末尾；失败时保持 Hybrid 原排序。

---

## 6. EvidenceReviewAgent

功能：逐条判断证据相关性、可信度、支持、反证和安全风险。

### 输入 Schema

```json
{
  "query": "本轮查询",
  "query_type": "string",
  "evidence": [
    {
      "chunk_id": "string",
      "evidence_level": "string",
      "pmid": "string|null",
      "title": "string|null",
      "text": "最多500字",
      "rule_signals": {
        "is_supporting": true,
        "is_contradicting": false,
        "is_safety_risk": false
      }
    }
  ]
}
```

### 输出 Schema

```json
{
  "verdict": "strongly_supports | partially_supports | contradicts | insufficient_evidence",
  "evidence_assessments": [
    {
      "chunk_id": "证据ID",
      "relevance_score": 0.0,
      "credibility_score": 0.0,
      "is_supporting": true,
      "is_contradicting": false,
      "is_safety_risk": false,
      "summary": "审核说明"
    }
  ]
}
```

### System Prompt

```text
你是肿瘤学检索证据审核助手。根据医生问题和候选证据片段，判断每条证据与问题的相关性、是否支持/反对结论、是否存在安全风险，并给出可信度评分。

必须通过 submit_evidence_review 函数提交审核结果。参数格式如下：
{
  "verdict": "strongly_supports | partially_supports | contradicts | insufficient_evidence",
  "evidence_assessments": [
    {
      "chunk_id": "证据ID",
      "relevance_score": 0.0,
      "credibility_score": 0.0,
      "is_supporting": true,
      "is_contradicting": false,
      "is_safety_risk": false,
      "summary": "一句话说明该证据为何支持/反对/无关"
    }
  ]
}

verdict 判定标准：
- strongly_supports：多条高质量证据一致支持回答方向
- partially_supports：有部分支持证据，但证据不足或质量一般
- contradicts：存在明确反证、无效、进展或重要安全风险
- insufficient_evidence：几乎没有可用证据

评分范围均为 0-1。每条输入证据都必须在 evidence_assessments 中出现且 chunk_id 与输入一致。
```

User Prompt 为输入 Schema 对应的 JSON。

规则回退：通过缓解/获益/进展/耐药/毒性等关键词分类；没有支持证据时不能给正向 verdict。

---

## 7. RetrievalLedger 与 StepMemoryAgent

功能：ExecutionAgent 的每个真实搜索工具调用先由规则 Ledger 完整记录为 Micro Retrieval Attempt；StepMemoryAgent 对当前 Planning Step 的这批 Ledger 和证据审核只总结一次。

### 输入 Schema

```json
{
  "round": 1,
  "main_question": "主问题",
  "executed_query": "本轮query",
  "micro_retrieval_ledger": [
    {"attempt": 1, "tool": "hybrid_search", "query": "...", "hit_count": 8},
    {"attempt": 2, "tool": "bm25_search", "query": "...", "hit_count": 6}
  ],
  "active_plan_step": {
    "step_id": "S1",
    "goal": "本轮原子目标",
    "rerank_goal": "本轮评分标准"
  },
  "retrieval_plan": {
    "query_type": "string",
    "selected_tools": [],
    "constraints": {},
    "planner_metadata": {}
  },
  "evidence_review": {},
  "previously_seen_chunk_ids": []
}
```

### 输出 Schema

```json
{
  "accepted_evidence": [
    {
      "chunk_id": "string",
      "evidence_role": "support | counter | risk",
      "reason": "准入理由"
    }
  ],
  "rejected_evidence": [
    {
      "chunk_id": "string",
      "reason": "淘汰理由"
    }
  ],
  "question_information_gain": {
    "score": 0.0,
    "new_facts": [],
    "resolved_questions": [],
    "new_conflicts": [],
    "new_chunk_ids": []
  },
  "goal_evaluation": {
    "matched_goal_count": 0,
    "best_goal_relevance": 0.0,
    "success_criteria_met": false,
    "completion_status": "not_met | minimally_met | sufficiently_met",
    "critical_gaps": [],
    "optional_gaps": [],
    "marginal_value_of_more_search": 0.0,
    "recommended_stop": false,
    "observed_gaps": [],
    "observed_failures": []
  }
}
```

### System Prompt

```text
你是 StepMemoryAgent。你一次总结当前 Planning Step 内多个 Micro Retrieval Attempts 的规则 Ledger 和证据审核结果。你只提取客观信息增益、冲突、完成度和缺口，不决定下一步动作。必须通过 commit_step_memory 提交。
```

### User Prompt

```text
输入：
{输入JSON}

输出证据准入/淘汰、question_information_gain 和 goal_evaluation。只允许使用输入中存在的 chunk_id；只记录观察，不得建议下一轮动作。
```

规则：虚构 chunk ID 会被删除；分数裁剪到 0–1；LLM 失败时按相关性阈值 `0.35` 准入。

---

### 7.1 ReplannerMemoryAgent

维护方式：上一版长短期记忆与尚未合并的 StepMemory 一起交给 LLM，LLM 返回完整的新版本并完全替代旧版本。默认累计 3 个 StepMemory，或估算上下文达到 2800 tokens 时触发。未被新版本保留的低价值内容即被工作记忆遗忘；原始 StepMemory、Tool Ledger 和证据不会删除。

输入 Schema：

```json
{
  "main_question": "string",
  "plan_steps": [],
  "active_step_id": "string",
  "previous_short_memory": {},
  "previous_long_memory": {},
  "new_step_memories": []
}
```

输出 Schema：

```json
{
  "short_memory": {
    "active_step_id": "string",
    "recent_information_gain": [],
    "recent_effective_strategies": [],
    "recent_failed_strategies": [],
    "current_critical_gaps": [],
    "recent_conflicts": [],
    "avoid_repeating": []
  },
  "long_memory": {
    "step_progress": [],
    "global_key_findings": [],
    "global_critical_gaps": [],
    "cross_step_constraints": [],
    "unresolved_conflicts": [],
    "search_strategy_lessons": [],
    "protected_safety_information": [],
    "evidence_ids": []
  }
}
```

原始 System Prompt：

```text
你维护供临床检索 Replanner 使用的长短期记忆。把旧记忆与尚未合并的 StepMemory 重新提炼成自包含的新版本；新版本完全替代旧版本，因此不机械追加。省略重复、已解决、低价值和被新信息替代的内容，这种提炼就是遗忘。短期记忆只保留近期会影响下一次检索的变化；长期记忆保留全局步骤进展、关键结论、关键缺口、跨步骤约束、未解决冲突、检索策略经验和临床安全信息。关键结论、冲突与安全信息必须保留真实 evidence/chunk id。不要虚构证据。必须通过 maintain_replanner_memory 提交。
```

如果 LLM 调用失败，不执行伪造的规则摘要，而是保留上一版本和待处理 StepMemory，之后再次尝试提炼。Replanner 每次还会单独接收当前步骤最新的一条 StepMemory。

## 8. AnswerMemoryAgent

功能：跨轮生成、合并和修订最终回答所需的具体 claim。

### 输入 Schema

```json
{
  "existing_answer_memory": {
    "claims": [],
    "evidence_by_id": {},
    "informative_rounds": []
  },
  "new_round_memory": {}
}
```

### 输出 Schema

```json
{
  "claims": [
    {
      "claim_id": "claim_01",
      "claim": "具体临床论断",
      "status": "provisional | supported | contested",
      "supporting_chunk_ids": [],
      "contradicting_chunk_ids": [],
      "source_rounds": [],
      "confidence": 0.0,
      "safety_status": "pending"
    }
  ],
  "informative_rounds": []
}
```

### System Prompt

```text
你是 AnswerMemoryAgent。你维护最终回答所需的跨轮 claim 记忆。你必须从有信息增益的证据中提取具体、可核验、粒度适中的临床论断；合并重复论断；保留支持与反证；新证据出现时修订旧论断。每个论断必须绑定真实 chunk_id。必须通过 commit_answer_memory 函数提交结果，不生成面向用户的最终回答。
```

### User Prompt

```text
输入：
{existing_answer_memory 和 new_round_memory 的 JSON}

输出 JSON：{"claims":[{"claim_id":"claim_01","claim":"具体临床论断","status":"provisional|supported|contested","supporting_chunk_ids":[],"contradicting_chunk_ids":[],"source_rounds":[],"confidence":0.0,"safety_status":"pending"}],"informative_rounds":[]}。必须保留仍有效的旧 claim；不得引用 evidence_by_id 中不存在的 chunk_id；confidence 范围0-1。
```

规则：claim 必须非空；引用必须存在于 `evidence_by_id`；confidence 裁剪到 0–1；失败时生成保守的通用 claim。

---

## 9. MainAgentLoop 规则

MainAgentLoop 负责持有多步 Plan、当前步骤和轮次状态；决策由 Replanner Tool Call 或其规则回退产生。

每轮控制动作：

```text
retry_current_step → 保持当前原子目标，更换检索词
advance_to_next_step → 当前步骤完成，进入Plan下一步
expand_current_step → 保持目标，使用显式关系扩展Query
revise_plan → 插入或修改原子步骤
stop → Replanner判定计划已完成或不应继续
单步达到 attempt_budget → 标记部分完成并自动推进
达到当前总预算 → Replanner可申请扩容；未获批则停止
达到 max_total_steps → 无条件硬停止
```

主要缺口：

```text
无支持证据
无反证
缺少安全/DDI证据
缺少突变约束
缺少药物/方案约束
缺少癌种约束
```

---

## 10. ClinicalSafetyGate

功能：先执行不可绕过的安全硬规则，再让 LLM 逐条反思 claim。

### LLM 输入 Schema

```json
{
  "query": "主问题",
  "query_type": "string",
  "constraints": {},
  "rule_safety_result": {},
  "claims": [],
  "evidence_by_id": {}
}
```

### LLM 输出 Schema

```json
{
  "claim_reviews": [
    {
      "claim_id": "string",
      "decision": "approve | revise | reject",
      "violations": [],
      "required_revision": "string",
      "safe_claim": "安全改写后的论断",
      "evidence_refs": []
    }
  ],
  "cross_claim_conflicts": [],
  "additional_issues": [
    {
      "code": "string",
      "severity": "medium | high | critical",
      "title": "string",
      "message": "string",
      "recommendation": "string",
      "evidence_refs": []
    }
  ],
  "reflection_summary": "string"
}
```

### System Prompt

```text
你是 ClinicalSafetyGate 的临床安全自反思模块。你必须逐条审核候选 claim 是否与主问题、患者约束、支持证据、反证、毒性、禁忌和证据等级冲突。规则安全问题是不可删除的底线。你可以批准、降级、改写或否决 claim，但不得扩大证据结论，不得给出处方决定。必须通过 submit_safety_reflection 函数提交结果。
```

### User Prompt

```text
输入：
{输入JSON}

输出 JSON：{"claim_reviews":[{"claim_id":"","decision":"approve|revise|reject","violations":[],"required_revision":"","safe_claim":"","evidence_refs":[]}],"cross_claim_conflicts":[],"additional_issues":[{"code":"","severity":"medium|high|critical","title":"","message":"","recommendation":"","evidence_refs":[]}],"reflection_summary":""}。每个输入 claim 必须有一条 review；规则安全结果不可撤销。
```

硬规则：

```text
可溯源证据少于3条 → high
问题包含剂量/频率 → high
有DDI意图但无DDI证据 → high
出现Grade 3/4、fatal、death、严重器官毒性 → critical
只有病例/抽取证据且无指南或标签 → medium
```

LLM 无权删除规则 issue，只能追加。`revise/reject` 会强制人工复核，并反写 `safe_claim` 或拒绝状态。

---

## 11. CitationAgent

CitationAgent 不使用 LLM，避免虚构引用。

功能：

```text
claim
→ supporting/contradicting chunk_id
→ evidence_by_id
→ doc_id/case_id/event_id
→ source_file/field_path/start_char/end_char
→ PMID/标题/原文片段
```

### 输入 Schema

```json
{
  "claims": [
    {
      "claim_id": "claim_01",
      "claim": "string",
      "supporting_chunk_ids": [],
      "contradicting_chunk_ids": []
    }
  ],
  "evidence_by_id": {}
}
```

### 输出 Schema

```json
{
  "claim_citations": [
    {
      "claim_id": "claim_01",
      "claim": "string",
      "citations": [
        {
          "chunk_id": "string",
          "doc_id": "string|null",
          "case_id": "string|null",
          "event_id": "string|null",
          "pmid": "string|null",
          "title": "string|null",
          "source_file": "string|null",
          "field_path": "string|null",
          "start_char": 0,
          "end_char": 0,
          "chunk_order": 0,
          "evidence_level": "string|null",
          "quoted_span": "最多500字"
        }
      ],
      "missing_refs": []
    }
  ],
  "unresolved_claim_ids": []
}
```

---

### 11.1 AnswerContextAgent

功能：为最终回答创建有界上下文。代码先从当前/历史 Session 中白名单提取 confirmed constraints、Replanner 长短期记忆、最多10条当前 claim、最多6条历史 claim和安全结果；LLM 再合并重复内容并遗忘检索过程。原始 StepMemory、Replan 历史和 Tool Trace 不进入该 Agent，也不进入 AnswerGenerator。

输出：

```json
{
  "case_context": [],
  "key_findings": [],
  "unresolved_gaps": [],
  "conflicts_and_limitations": [],
  "safety_boundaries": [],
  "evidence_ids": []
}
```

System Prompt：

```text
你是 AnswerContextAgent。你只为最终回答提炼上下文。输入已经过字段白名单压缩；请进一步合并重复内容，遗忘检索过程和已解决的低价值细节，只保留病例约束、关键发现、未解决缺口、证据冲突、临床安全边界和真实 evidence/chunk id。不得补充输入中不存在的医学事实或引用。必须通过 summarize_answer_context 提交。
```

## 12. AnswerGenerator

功能：基于安全审核后的 claim、支持证据、反证和 Citation 生成最终 Markdown 回答。

### 输入 Schema

```json
{
  "query": "用户问题",
  "query_type": "string",
  "prior_context": {},
  "answer_claims": [],
  "citation_map": [],
  "layer_distribution": [],
  "risk_level": "string",
  "safety_issues": [],
  "recommended_actions": [],
  "supporting_evidence": [],
  "counter_evidence": []
}
```

### 输出 Schema

输出为 Markdown 字符串，包含：直接结论、关键病例、支持证据、反证/风险、可信度、安全边界和引用。

### System Prompt

```text
你是肿瘤临床检索助手。基于检索到的病例证据生成带引用、分层和安全边界的回答。必须区分病例库证据与模型背景知识，不能给出超出证据范围的处方建议。引用只能来自输入的 CitationAgent 引用映射；禁止补充、猜测或凭模型记忆生成 PMID、DOI、标题、URL 或原文引句。
```

### User Prompt

```text
医生问题：
{query}

问题类型：{query_type}

历史会话上下文：
{prior_context}

经安全门修正的候选结论：
{answer_claims}

论断到源文件的引用映射：
{citation_map}

证据分层：
{layer_distribution}

安全门结果：
- 风险等级: {risk_level}
- 关键问题: {safety_issues}
- 建议动作: {recommended_actions}

支持证据（节选）：
{supporting_evidence}

反对/风险证据（节选）：
{counter_evidence}

请按以下固定结构输出中文回答（使用 Markdown 小标题）：
1. 直接结论
2. 关键病例证据
3. 支持证据
4. 反证/风险/毒性
5. 证据可信度说明
6. 临床安全边界
7. 引用（PMID/标题/片段）

要求：每条关键论断必须对应 citation_map 中的 chunk_id；引用章节只能逐项复制 citation_map 中存在的来源；禁止引用模型背景知识；证据不足处明确说明；结尾重申不能替代指南与 MDT。
```

引用硬保护：

```text
citation_map 为空 → 不调用 AnswerGenerator LLM，直接输出证据不足模板
被 SafetyGate reject 的 claim → 不进入最终 Prompt
存在 safe_claim → 用 safe_claim 替换原 claim
证据节选 → 只保留 CitationAgent 已验证的 chunk
输出出现 CitationAgent 未提供的 PMID/chunk/DOI/URL → 拒绝该输出并回退模板
```

LLM 失败或引用校验失败时输出固定 Markdown 模板。

---

## 13. SessionMemoryStore

不使用 LLM。保存：

```json
{
  "session_id": "string",
  "confirmed_constraints": {},
  "identified_entities": {},
  "round_memories": [],
  "answer_memory": {},
  "final_safety_review": {},
  "citations": {},
  "original_query": "string",
  "updated_at": "ISO-8601"
}
```

下一次会话会把这些内容直接交给 PlannerAgent。

## 14. LLM 与规则的边界

LLM 负责：规划、语义相关性、证据审核、信息增益、claim 维护、安全反思和回答生成。

规则负责：检索算法、工具白名单、真实 chunk 校验、分数范围、反证 query、停止条件、安全硬底线、引用追溯和 Session 存储。

原则：需要语义推理的交给 LLM；必须可验证且不能被模型绕过的交给规则。

---

## 15. 端到端硬约束与验收规则

以下规则是为了避免“LLM 看似完成，但证据链实际断开”。

### 15.1 检索必须回到结构化证据

只要 Planner 选择以下任一检索工具：

```text
structured_search
dense_search
bm25_search
hybrid_search
```

Router 会自动补充：

```text
fetch_evidence
```

原因是 EvidenceReviewAgent、AnswerMemoryAgent 和 CitationAgent 只能使用结构化数据库返回的权威证据，不能直接把向量/BM25 的摘要当作最终依据。

### 15.2 Replanner 独占检索控制决策

每批 StepMemoryAgent 只产生客观 `goal_evaluation`。Replanner 根据当前 Plan Step、长期记忆、短期记忆、最新 StepMemory 和精简 AnswerMemory 选择：

```text
retry_current_step
advance_to_next_step
expand_current_step
revise_plan
stop
```

连续两轮无新信息时才进入扩展；药名、PK关系和DDI规则的扩展必须保留来源和关系类型。

### 15.3 最终回答禁止产生未验证引用

AnswerGenerator 只能使用 CitationAgent 提供的引用映射。

如果没有可验证 citation：

- 不调用 LLM 生成带文献的回答；
- 使用模板明确说明数据库证据不足；
- 禁止输出 PMID、DOI、URL、期刊标题或模型记忆中的文献引句。

如果有 citation：

- LLM Prompt 明确禁止补充外部引用；
- 生成后校验 PMID 和 chunk ID 是否属于 CitationAgent 的允许集合；
- 出现未验证 PMID、URL、DOI 或 chunk ID 时，丢弃 LLM 输出并使用安全模板。

### 15.4 端到端验收指标

真实 GPT-5 测试至少应满足：

```text
PlannerAgent 的输出不是规则默认值
RoutePlan 包含 fetch_evidence
LLMReranker 输出 llm_relevance_score
EvidenceReviewAgent review_mode=llm
StepMemoryAgent memory_mode=llm
AnswerMemoryAgent memory_mode=llm（存在准入证据时）
ClinicalSafetyGate review_mode=rules+llm
CitationAgent 输出 claim_citations
最终回答没有 CitationAgent 之外的 PMID/DOI/URL
```

推荐测试命令：

```bash
set -a
source .env
set +a
python run_agent_loop.py \
  --query "EGFR突变NSCLC患者奥希替尼治疗后进展，检索相似病例和耐药机制，并列出安全风险" \
  --max-steps 2 \
  --max-total-steps 128 \
  --max-budget-extension 32 \
  --top-k 3 \
  --session-id e2e_case_fixed \
  --print-workflow
```

如果某个 LLM 调用失败，工作流会局部回退；验收时必须在输出中明确看到回退日志，不能把规则回退误认为 LLM 成功。

`--print-workflow` 现在只输出便于阅读的摘要和最终回答状态，不再随后打印可能达到数十万字符的完整 JSON。需要审计全量对象时显式增加 `--print-full-json`；Session JSON仍会正常保存。

### 15.5 LLM 延迟与超时控制

网络健康检查显示模型列表接口可在约 1 秒内返回，最小 GPT-5 请求通常数秒返回。Agent 超时主要来自大 Prompt、完整证据在多个 Agent 间重复传递，以及未限制生成长度。

当前控制措施：

```text
Replanner 只读取长期记忆、短期记忆、最新StepMemory、claims和证据ID
ReplannerMemory默认每3个StepMemory或估算输入超过2800 tokens时重新提炼
StepMemory 每条证据文本最多500字符
AnswerMemory 不重复发送完整 evidence_by_id
SafetyGate 最多读取30条证据，每条500字符
各 Agent 设置独立 max_completion_tokens
```

默认输出上限：

```text
Planner: 1200
Reranker: 1200
EvidenceReview: 2400
StepMemory: 1800
AnswerMemory: 1800
SafetyGate: 1800
AnswerGenerator: 3000
```

环境变量：

```bash
export LLM_TIMEOUT=120
export LLM_MAX_OUTPUT_TOKENS=2048
```

不建议把 `LLM_TIMEOUT` 降到 20 秒作为正常配置；20 秒适合快速失败测试，会导致复杂 Agent 请求频繁回退。
### Query 数据库穷尽权

RetrievalExecutionAgent 可在 `submit_step_execution_report` 中返回：

```json
{
  "query_database_status": "more_available | exhausted | uncertain",
  "exhaustion_reason": "互补检索为什么已无新增相关信息",
  "recommended_query_change": "建议的不同检索表达式",
  "queries_attempted": []
}
```

`exhausted` 只表示当前具体 query 在现有数据库中已穷尽，不表示整个 Planning Step 或主问题完成。StepMemory 保留该判断；Replanner 不得原样重复 `queries_attempted`，只能改写、扩展、推进或停止。编排器还会拦截模型意外生成的相同 query。
