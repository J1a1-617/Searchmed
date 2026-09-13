from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)

PLAN_ACTIONS = frozenset({"retry_current_step", "advance_to_next_step", "expand_current_step", "revise_plan", "stop"})
EVIDENCE_LANES = frozenset({"direct_case", "analog_case", "mechanism", "ddi_safety"})
EXPANSION_STRATEGIES = frozenset({
    "none", "drug_alias", "case_ddi", "pk_relation", "ddi_rule", "mutation_drug",
})

RETRIEVAL_DATABASE_PROFILE: Dict[str, Any] = {
    "primary_content": "local oncology case-report corpus, not a comprehensive PubMed/trial registry",
    "retrieval_layers": [
        "Step3 structured case cards and treatment events for document-level recall",
        "Step2 cleaned case-report text chunks for within-document clinical details",
        "curated drug alias, mutation-drug, PK, case-DDI and DDI-rule JSON tables",
    ],
    "high_recall_language": "patient-like disease, driver, treatment sequence, metastatic site and broad outcome wording",
    "poor_recall_patterns": [
        "unverified trial names used as mandatory anchors",
        "many exact endpoints, doses and assessment times packed into one recall query",
        "PubMed Boolean syntax or field-qualified expressions",
    ],
}

_STRING_ARRAY = {"type": "array", "items": {"type": "string"}}
_CONSTRAINTS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "Confirmed patient facts usable as hard retrieval filters; never include inferred candidates.",
    "properties": {
        "cancer_type": {"type": "string"},
        "gene_alterations": _STRING_ARRAY,
        "drugs": _STRING_ARRAY,
        "responses": _STRING_ARRAY,
        "toxicities": _STRING_ARRAY,
        "metastatic_sites": _STRING_ARRAY,
        "ddi_terms": _STRING_ARRAY,
    },
    "required": ["cancer_type", "gene_alterations", "drugs", "responses", "toxicities", "metastatic_sites", "ddi_terms"],
    "additionalProperties": False,
}
_STEP_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "step_id": {"type": "string", "description": "Stable unique identifier for this retrieval step."},
        "goal": {"type": "string", "description": "One atomic evidence question, not a broad research topic."},
        "rerank_goal": {"type": "string", "description": "Fine-grained criterion used to compare candidates retrieved for this step."},
        "evidence_lane": {"type": "string", "enum": sorted(EVIDENCE_LANES)},
        "success_criteria": {"type": "array", "description": "Observable evidence conditions that make this step complete.", "items": {"type": "string"}},
        "attempt_budget": {"type": "integer", "description": "Maximum micro retrieval attempts allocated to this step.", "minimum": 1, "maximum": 128},
    },
    "required": ["step_id", "goal", "rerank_goal", "evidence_lane", "success_criteria", "attempt_budget"],
    "additionalProperties": False,
}

INITIAL_PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "plan_rationale": {"type": "string"},
        "initial_total_budget": {"type": "integer", "minimum": 1, "maximum": 128},
        "steps": {"type": "array", "description": "Initial 2-3 step skeleton only. Leave newly discovered branches to Replanner.", "minItems": 2, "maxItems": 3, "items": _STEP_SCHEMA},
    },
    "required": ["plan_rationale", "initial_total_budget", "steps"],
    "additionalProperties": False,
}

REPLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "description": "Control decision for the current plan step after reading its latest memory.", "enum": sorted(PLAN_ACTIONS)},
        "active_step_id": {"type": "string"},
        "search_query": {"type": "string", "description": "Recall-oriented case-corpus intent: preserve 3-6 discriminative clinical concepts and one broad outcome; leave exact timing/endpoints to rerank_goal. It must materially differ after a failed attempt. Execution Agent adapts it per tool."},
        "rerank_goal": {"type": "string", "description": "Current atomic relevance criterion for LLM reranking."},
        "selected_tools": {"type": "array", "items": {"type": "string", "enum": ["structured_search", "dense_search", "bm25_search", "hybrid_search", "fetch_evidence"]}},
        "constraints": _CONSTRAINTS_SCHEMA,
        "expansion": {"type": "object", "properties": {
            "strategy": {
                "type": "string",
                "description": "Knowledge source used to expand the active retrieval step.",
                "enum": sorted(EXPANSION_STRATEGIES),
            },
            "source": {"type": "string"},
            "added_terms": _STRING_ARRAY,
            "relation_type": {"type": "string"},
        }, "required": ["strategy", "source", "added_terms", "relation_type"], "additionalProperties": False},
        "plan_changes": {"type": "array", "description": "At most two new atomic steps to insert immediately after the active step when action=revise_plan. Use only for a high-value gap not already covered by the remaining plan.", "maxItems": 2, "items": _STEP_SCHEMA},
        "avoid_repeating": _STRING_ARRAY,
        "decision_rationale": {"type": "string"},
        "requested_budget_extension": {"type": "integer", "minimum": 0, "maximum": 32},
        "budget_extension_reason": {"type": "string"},
    },
    "required": ["action", "active_step_id", "search_query", "rerank_goal", "selected_tools", "constraints", "expansion", "plan_changes", "avoid_repeating", "decision_rationale", "requested_budget_extension", "budget_extension_reason"],
    "additionalProperties": False,
}


class MultiStepPlanningAgent:
    """Create a task plan once, then control one atomic retrieval step per round."""

    PLAN_SYSTEM = """你是临床检索 Initial Planner。初始计划只能是 2-3 个高层证据目标组成的最小骨架，不是研究大纲或最终回答目录。不要预先穷举机制、PK、亚组、毒性和每个候选分支；后续 Replanner 可以根据真实检索结果增加新步骤。每一步只解决一个证据目标，并给出专属 rerank_goal 和必要的 attempt_budget（安全上限16）。必须按“预期信息增益×数据库可得性×对结论的影响÷检索成本”排序。默认第一步必须是 direct_case；明确毒性/禁忌会改变结论时，ddi_safety 次优先。机制、PK、稀有分子亚组和高度特异定量指标通常只作补充。“总结、综合、撰写回答”不是检索步骤。initial_total_budget 除覆盖初始步骤外，应尽可能预留至少 1 轮给 Replanner 的动态改写或新增步骤。必须通过 submit_multistep_plan 提交。"""
    REPLAN_SYSTEM = """你是面向本地肿瘤病例库的 Replanner，不是 PubMed 检索式生成器。你读取既定计划、active_step、预算、长短期记忆和最新 StepMemory，每次只对当前 active_step 决策。

状态语义：search_attempt_status只描述本次工具链；query_direction_status=locally_exhausted只表示当前query表达已穷尽；database_coverage_status=unknown时绝对不得推断数据库没有。只有程序聚合多个不同query和互补工具后给出absence_supported，才可将“本库可能无直接证据”作为推进依据。

数据库约束：
1. 主库是肿瘤 case report/病例语料，不是完整的临床试验或指南库。
2. Step3 是文档级病例卡片和治疗事件，适合先找患者/文档；Step2 是对应原文 chunk，适合再找剂量、时间、疗效和毒性细节。
3. 本地还有药物别名、突变-药物、PK、case-DDI 和 DDI-rule 表，只能在直接检索不足时通过 expansion 使用。

search_query 构造规则：
1. 先写一条面向病例召回的简短临床意图，通常只保留 3–6 个有区分度的概念：癌种/驱动、核心药物或治疗序列、关键转移部位、一个宽泛结局。
2. 精确剂量、6–12周、ORR/DCR、CSF转阴、首次 MRI 等是排序条件，优先放入 rerank_goal；除非它是当前步骤唯一的核心区分点，不要全部塞进 search_query。
3. 不得仅凭模型记忆强制加入 AURA/AURA2/AURA3/FLAURA/BLOOM/FURLONG 等试验名。只有 active_step 明确要求该试验，或已召回证据/标题证明库中存在它时，才可作为检索锚点。
4. 禁止 PubMed 布尔语法、字段标签、AND/OR/NOT、括号查询、注释和多行备选 query。Execution Agent 会先选工具，再改写成 dense/BM25/structured 专用表达。

根据上轮结果改写：
1. matched_goal_count=0 或 not_met：下一轮必须放宽，删除至少两个不必要的精确限定，改用病例叙事语言；不得通过增加新试验名、更窄时间或更多终点来“改写”。
2. minimally_met：保留上轮已命中的核心语义，一次只改一个 critical gap（例如药物别名、转移部位或治疗线次），不得同时追加所有缺口。
3. query_direction_status=locally_exhausted：不得原样重试；应改写query或换互补工具。它不等于数据库无信息。
4. sufficiently_met 或 recommended_stop=true：推进下一步或停止。

对照示例：
- 错误：上轮脑转移零命中后，改成“AURA3 intracranial ORR MRI 6 weeks T790M osimertinib 80 mg”。
- 正确：“EGFR突变肺腺癌 一代TKI后脑转移 换用奥希替尼 早期颅内反应”；把 6–12周、ORR/DCR 和 MRI 放入 rerank_goal。
- 错误：一条 query 同时要求伏美替尼120/160 mg、8/12周、LM-ORR、CSF转阴、神经症状和单药。
- 正确：“EGFR突变肺腺癌脑膜转移 伏美替尼治疗 临床反应”；精确剂量、时间和 CSF 结局交给 rerank_goal。

动态步骤规则：
1. 每个已执行步骤的边界都要检查：最新证据或 critical_gaps 是否暴露了一个当前剩余 plan 没有覆盖、且可能改变最终答案的高价值证据问题。
2. 如果有，使用 action=revise_plan，在 plan_changes 中追加 1 个（最多2个）原子步骤；不得复制现有 goal，不得为可选细节增步骤。
3. 新步骤的 success_criteria 必须可被证据观测，attempt_budget 默认1–2。requested_budget_extension 至少覆盖新步骤的一次执行；如果当前预留预算已足够，可为0。
4. 如果已有剩余步骤覆盖该问题，只 advance_to_next_step，不得重复添加。

必须通过 submit_replan_decision 提交。"""
    def __init__(
        self,
        llm_client: Optional["LLMClient"] = None,
        use_llm: bool = True,
        external_knowledge_enabled: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.use_llm = use_llm
        self.external_knowledge_enabled = bool(external_knowledge_enabled)

    @staticmethod
    def _complete_constraints(constraints: Dict[str, Any]) -> Dict[str, Any]:
        result = {"cancer_type": "", "gene_alterations": [], "drugs": [], "responses": [], "toxicities": [], "metastatic_sites": [], "ddi_terms": []}
        result.update({key: value for key, value in constraints.items() if key in result})
        return result

    @staticmethod
    def _drug_alias_terms(drugs: List[str]) -> List[str]:
        path = Path(__file__).resolve().parent.parent / "data" / "all_drug_name_map.json"
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        wanted = {str(value).strip().lower() for value in drugs if str(value).strip()}
        expanded: List[str] = []
        for row in rows if isinstance(rows, list) else []:
            names = [row.get("drug_name"), row.get("generic_name"), row.get("brand_name"), *(row.get("alias") or [])]
            normalized = {str(value).strip().lower() for value in names if str(value).strip()}
            if wanted & normalized:
                expanded.extend(str(value).strip() for value in names if str(value).strip() and str(value).strip().lower() not in wanted)
        return list(dict.fromkeys(expanded))[:8]

    @classmethod
    def _drug_knowledge_terms(
        cls,
        strategy: str,
        drugs: List[str],
        mutations: Optional[List[str]] = None,
    ) -> List[str]:
        if strategy == "drug_alias":
            return cls._drug_alias_terms(drugs)
        data_root = Path(__file__).resolve().parent.parent / "data"
        if strategy == "mutation_drug":
            wanted_mutations = {
                "".join(str(value).lower().split())
                for value in (mutations or [])
                if str(value).strip()
            }
            try:
                rows = json.loads((data_root / "mutation_drug_map_min.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return []
            terms: List[str] = []
            for row in rows if isinstance(rows, list) else []:
                mutation = str(row.get("mutation") or "").strip()
                normalized = "".join(mutation.lower().split())
                if wanted_mutations and not any(
                    wanted == normalized or wanted in normalized or normalized in wanted
                    for wanted in wanted_mutations
                ):
                    continue
                if mutation:
                    terms.append(mutation)
                terms.extend(str(value).strip() for value in row.get("drugs") or [] if str(value).strip())
            return list(dict.fromkeys(terms))[:10]
        wanted = {str(value).strip().lower() for value in [*drugs, *cls._drug_alias_terms(drugs)] if str(value).strip()}
        try:
            pk_rows = json.loads((data_root / "all_drug_pk_relation.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        related_pk = [row for row in pk_rows if str(row.get("drug_name") or "").strip().lower() in wanted]
        if strategy == "pk_relation":
            terms: List[str] = []
            for row in related_pk:
                terms.extend(str(row.get(key) or "").strip() for key in ("enzyme_transporter", "role", "related_pk_drug", "pk_interaction_type"))
            return list(dict.fromkeys(term for term in terms if term))[:10]
        if strategy == "ddi_rule":
            enzymes = {str(row.get("enzyme_transporter") or "").strip().lower() for row in related_pk}
            try:
                ddi_rows = json.loads((data_root / "all_ddi_rule.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return []
            terms = []
            for row in ddi_rows if isinstance(ddi_rows, list) else []:
                enzyme = str(row.get("enzyme_transporter") or "").strip()
                if enzyme.lower() not in enzymes:
                    continue
                terms.extend([enzyme, str(row.get("role1") or "").strip(), str(row.get("role2") or "").strip()])
            return list(dict.fromkeys(term for term in terms if term))[:10]
        if strategy == "case_ddi":
            try:
                payload = json.loads((data_root / "case_ddi.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return []
            rows = payload.get("case_ddis") if isinstance(payload, dict) else []
            terms: List[str] = []
            for row in rows if isinstance(rows, list) else []:
                row_drugs = {str(value).strip().lower() for value in row.get("drugs") or [] if str(value).strip()}
                if wanted and not (wanted & row_drugs):
                    continue
                terms.extend(str(value).strip() for value in row.get("drugs") or [] if str(value).strip())
                terms.append(str(row.get("interaction_type") or "").strip())
                terms.append(str((row.get("outcome") or {}).get("response") or "").strip())
                terms.extend(str(value).strip() for value in ((row.get("outcome") or {}).get("adverse_events") or []) if str(value).strip())
            return list(dict.fromkeys(term for term in terms if term))[:10]
        return []

    @staticmethod
    def _compact_round_memories(memories: List[Dict[str, Any]], active_step_id: str) -> List[Dict[str, Any]]:
        compact: List[Dict[str, Any]] = []
        active_memories = [memory for memory in memories if str(memory.get("plan_step_id") or "") == active_step_id]
        for memory in active_memories[-2:]:
            trace_summary = []
            for call in (memory.get("tool_trace") or [])[-6:]:
                result = call.get("result") or {}
                trace_summary.append({
                    "tool_name": call.get("tool_name"),
                    "hit_count": result.get("hit_count"),
                    "fetched_count": result.get("fetched_count"),
                    "error": result.get("error"),
                })
            action = (memory.get("retrieval_actions") or [{}])[0]
            diagnostics = []
            for entry in (memory.get("micro_retrieval_ledger") or [])[-4:]:
                diagnostics.append({key: entry.get(key) for key in (
                    "tool", "query", "constraints", "spaces", "requested_top_k",
                    "hit_count", "known_candidate_count", "top_hit_ids", "error",
                )})
            compact.append({
                "round": memory.get("round"),
                "plan_step_id": memory.get("plan_step_id"),
                "executed_query": action.get("query"),
                "selected_tools": action.get("selected_tools") or [],
                "rerank_goal": action.get("rerank_goal"),
                "expansion": action.get("expansion") or {},
                "question_information_gain": memory.get("question_information_gain") or {},
                "goal_evaluation": memory.get("goal_evaluation") or {},
                "retrieval_state": memory.get("retrieval_state") or {},
                "tool_trace_summary": trace_summary,
                "retrieval_diagnostics": diagnostics,
                "retrieval_funnel": memory.get("retrieval_funnel") or {},
                "rejection_summary": memory.get("rejection_summary") or {},
            })
        return compact

    @staticmethod
    def _compact_answer_claims(claims: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
        ranked = sorted(
            [claim for claim in claims if isinstance(claim, dict)],
            key=lambda claim: (
                str(claim.get("status") or "") in {"contested", "safety_limited"},
                float(claim.get("confidence") or 0.0),
            ),
            reverse=True,
        )
        return [{
            "claim_id": claim.get("claim_id"),
            "claim": claim.get("safe_claim") or claim.get("claim"),
            "status": claim.get("status"),
            "confidence": claim.get("confidence"),
            "supporting_chunk_ids": (claim.get("supporting_chunk_ids") or [])[:4],
            "contradicting_chunk_ids": (claim.get("contradicting_chunk_ids") or [])[:4],
        } for claim in ranked[:limit]]

    @staticmethod
    def _normalize_steps(steps: List[Dict[str, Any]], max_steps: int = 12) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for index, source in enumerate(steps[:max_steps], start=1):
            step = dict(source)
            step["step_id"] = str(step.get("step_id") or f"S{index}")
            try:
                budget = int(step.get("attempt_budget") or 2)
            except (TypeError, ValueError):
                budget = 2
            step["attempt_budget"] = max(1, min(128, budget))
            normalized.append(step)
        return normalized

    def create_plan(self, main_question: str, query_type: Optional[str], constraints: Dict[str, Any], context: Dict[str, Any], initial_total_budget: int = 4, hard_total_budget: int = 128, problem_representation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.use_llm or self.llm_client is None:
            raise RuntimeError("Initial Planner requires an LLM; category/rule fallback is disabled")
        prompt = json.dumps({"main_question": main_question, "problem_representation": problem_representation or {}, "confirmed_constraints": constraints, "prior_answer_claims": (context.get("answer_memory") or {}).get("claims") or [], "initial_budget_hint": initial_total_budget, "hard_total_budget_limit": hard_total_budget}, ensure_ascii=False, indent=2)
        try:
            payload = self.llm_client.call_function(system=self.PLAN_SYSTEM, user=prompt, function_name="submit_multistep_plan", description="Submit an ordered atomic retrieval plan.", parameters=INITIAL_PLAN_SCHEMA, temperature=0.2, max_output_tokens=8000)
            raw_steps = [dict(step) for step in payload.get("steps") or [] if isinstance(step, dict)]
            if len(raw_steps) > 3:
                raise ValueError(f"Initial planner returned too many steps: {len(raw_steps)}")
            steps = self._normalize_steps(raw_steps, max_steps=3)
            if not steps:
                raise ValueError("Initial planner returned no executable steps")
            try:
                requested = int(payload.get("initial_total_budget") or initial_total_budget)
            except (TypeError, ValueError):
                requested = initial_total_budget
            return {
                "plan_rationale": str(payload.get("plan_rationale") or ""),
                # Every planned step must have at least one executable round.
                # This is enforced locally because upstream schema support does
                # not guarantee that the model chose a semantically useful sum.
                "initial_total_budget": min(
                    hard_total_budget,
                    max(requested, len(steps) + (1 if hard_total_budget > len(steps) else 0)),
                ),
                "steps": steps,
            }
        except Exception:
            raise

    def replan(self, *, main_question: str, plan: Dict[str, Any], active_step_index: int, recent_memories: List[Dict[str, Any]], answer_memory: Dict[str, Any], base_query_type: Optional[str], base_constraints: Dict[str, Any], budget_state: Optional[Dict[str, Any]] = None, replanner_short_memory: Optional[Dict[str, Any]] = None, replanner_long_memory: Optional[Dict[str, Any]] = None, memory_timeline: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        steps = plan.get("steps") or []
        active = dict(steps[min(active_step_index, len(steps) - 1)])
        attempts = [m for m in recent_memories if m.get("plan_step_id") == active.get("step_id")]
        no_gain = sum(
            not bool((m.get("question_information_gain") or {}).get("new_chunk_ids"))
            for m in attempts[-3:]
        )
        matched_goal_count = max([int((m.get("goal_evaluation") or {}).get("matched_goal_count") or 0) for m in attempts] or [0])
        used_expansions = {
            str(((m.get("retrieval_actions") or [{}])[0].get("expansion") or {}).get("strategy") or "none")
            for m in attempts
        }
        latest_goal = (attempts[-1].get("goal_evaluation") or {}) if attempts else {}
        attempted_tools = {
            str(row.get("tool") or "")
            for memory in attempts
            for row in memory.get("micro_retrieval_ledger") or []
            if row.get("tool")
        }
        attempted_queries = {
            " ".join(str(row.get("query") or "").lower().split())
            for memory in attempts
            for row in memory.get("micro_retrieval_ledger") or []
            if str(row.get("query") or "").strip()
        }
        complementary_tool_coverage = bool(
            attempted_tools & {"dense_search", "hybrid_search"}
            and attempted_tools & {"bm25_search", "structured_search"}
        )
        absence_supported = bool(
            len(attempted_queries) >= 2
            and complementary_tool_coverage
            and no_gain >= 2
            and matched_goal_count == 0
        )
        coverage_status = str(latest_goal.get("database_coverage_status") or "unknown")
        if absence_supported:
            coverage_status = "absence_supported"
            latest_goal["database_coverage_status"] = coverage_status
        if (
            latest_goal.get("completion_status") == "sufficiently_met"
            or latest_goal.get("recommended_stop") is True
            or ("completion_status" not in latest_goal and latest_goal.get("success_criteria_met") is True)
        ):
            action = "advance_to_next_step" if active_step_index + 1 < len(steps) else "stop"
        elif coverage_status == "absence_supported" or no_gain >= 3:
            if matched_goal_count == 0:
                action = "advance_to_next_step" if active_step_index + 1 < len(steps) else "stop"
            else:
                action = "advance_to_next_step" if active_step_index + 1 < len(steps) else "stop"
        else:
            action = "retry_current_step"
        expansion_strategy = "none"
        if action == "expand_current_step":
            lane = str(active.get("evidence_lane") or "direct_case")
            if lane == "ddi_safety":
                expansion_order = ["drug_alias", "case_ddi", "pk_relation", "ddi_rule"]
            elif base_constraints.get("gene_alterations"):
                expansion_order = ["drug_alias", "mutation_drug"]
            else:
                expansion_order = ["drug_alias"]
            expansion_strategy = next((name for name in expansion_order if name not in used_expansions), expansion_order[-1])
        expansion_source = {
            "drug_alias": "all_drug_name_map.json",
            "pk_relation": "all_drug_pk_relation.json",
            "ddi_rule": "all_ddi_rule.json",
            "mutation_drug": "mutation_drug_map_min.json",
            "case_ddi": "case_ddi.json",
        }.get(expansion_strategy, "")
        fallback = {
            "action": action,
            "active_step_id": str(active.get("step_id") or "S1"),
            "search_query": f"{main_question} {active.get('goal', '')}".strip(),
            "rerank_goal": str(active.get("rerank_goal") or active.get("goal") or main_question),
            "selected_tools": ["structured_search", "dense_search", "bm25_search", "hybrid_search", "fetch_evidence"],
            "constraints": self._complete_constraints(base_constraints),
            "expansion": {"strategy": expansion_strategy, "source": expansion_source, "added_terms": [], "relation_type": {"drug_alias": "same_drug_identity", "pk_relation": "pharmacokinetic_relation", "ddi_rule": "mechanism_rule", "mutation_drug": "mutation_to_candidate_drug", "case_ddi": "direct_case_ddi_relation"}.get(expansion_strategy, "")},
            "plan_changes": [], "avoid_repeating": [str((m.get("retrieval_actions") or [{}])[0].get("query") or "") for m in attempts[-2:]],
            "decision_rationale": "根据当前步骤最近轮次的客观信息增益选择动作",
            "requested_budget_extension": 0,
            "budget_extension_reason": "",
        }
        if latest_goal.get("query_direction_status") == "locally_exhausted" and latest_goal.get("recommended_query_change"):
            fallback["search_query"] = str(latest_goal["recommended_query_change"])
            fallback["avoid_repeating"] = list(dict.fromkeys([*fallback["avoid_repeating"], *[str(x) for x in latest_goal.get("queries_attempted") or []]]))
        if not self.use_llm or self.llm_client is None:
            if action == "expand_current_step":
                terms = self._drug_knowledge_terms(
                    expansion_strategy,
                    list(base_constraints.get("drugs") or []),
                    list(base_constraints.get("gene_alterations") or []),
                )
                fallback["expansion"]["added_terms"] = terms
                fallback["search_query"] = " ".join([fallback["search_query"], *terms]).strip()
            return fallback
        # Empty/exhausted/sufficient and repeated-no-gain states are objective
        # workflow guards.  Do not spend an LLM call asking the Replanner to
        # rediscover a transition already fixed by those rules.
        deterministic_transition = coverage_status == "absence_supported" or no_gain >= 3
        if deterministic_transition:
            fallback["decision_mode"] = "rules"
            return fallback
        database_profile = dict(RETRIEVAL_DATABASE_PROFILE)
        database_profile["retrieval_layers"] = list(RETRIEVAL_DATABASE_PROFILE["retrieval_layers"])
        if not self.external_knowledge_enabled:
            database_profile["retrieval_layers"] = [
                layer for layer in database_profile["retrieval_layers"]
                if not layer.startswith("curated drug alias")
            ]
            database_profile["external_knowledge"] = "disabled for this ablation; use only the case-report corpus"
        prompt = json.dumps({
            "main_question": main_question,
            "database_profile": database_profile,
            "query_contract": {
                "search_query": "case-corpus recall intent with 3-6 discriminative concepts and one broad outcome",
                "rerank_goal": "exact dose, timing, endpoints, exclusions and evidence preference",
                "zero_match_change": "remove at least two unnecessary exact constraints; do not add an unverified trial name",
                "partial_match_change": "preserve the matched nucleus and change only one critical-gap axis",
            },
            "plan": plan,
            "active_step_index": active_step_index,
            "active_step": active,
            "memory_timeline": memory_timeline or {},
            "retrieval_state": (attempts[-1].get("retrieval_state") or {}) if attempts else {},
            "replanner_short_memory": replanner_short_memory or {},
            "replanner_long_memory": replanner_long_memory or {},
            "latest_step_memory": self._compact_round_memories(recent_memories, str(active.get("step_id") or ""))[-2:],
            "answer_claims": self._compact_answer_claims(answer_memory.get("claims") or []),
            "budget_state": budget_state or {},
            "allowed_actions": sorted(PLAN_ACTIONS),
        }, ensure_ascii=False, separators=(",", ":"))
        try:
            replan_system = self.REPLAN_SYSTEM
            if not self.external_knowledge_enabled:
                replan_system += "\n\n本次消融已禁用所有药物别名、突变-药物、PK、case-DDI 和 DDI-rule 扩展表。不得选择 expand_current_step；只能改写原始病例库的检索词。"
            payload = self.llm_client.call_function(system=replan_system, user=prompt, function_name="submit_replan_decision", description="Submit the next single-step retrieval decision.", parameters=REPLAN_SCHEMA, temperature=0.2, max_output_tokens=8000)
            result = dict(fallback)
            result.update(payload)
            # A replan decision is scoped to the orchestrator's current step;
            # the model may not silently retarget a later step.
            result["active_step_id"] = str(active.get("step_id") or "S1")
            if result["action"] not in PLAN_ACTIONS:
                result["action"] = fallback["action"]
            if not self.external_knowledge_enabled and result["action"] == "expand_current_step":
                result["action"] = "retry_current_step"
                result["expansion"] = {
                    "strategy": "none",
                    "source": "",
                    "added_terms": [],
                    "relation_type": "",
                }
                result["decision_rationale"] = (
                    f"{str(result.get('decision_rationale') or '').strip()} "
                    "External knowledge is disabled; continue with a case-corpus query reformulation."
                ).strip()
            # Synonym/outcome broadening with no knowledge expansion strategy
            # is a normal query retry. Do not switch Execution Agent into the
            # heavier drug-knowledge expansion loop for strategy=none.
            if (
                result["action"] == "expand_current_step"
                and str((result.get("expansion") or {}).get("strategy") or "none") == "none"
            ):
                result["action"] = "retry_current_step"
            if result["action"] == "expand_current_step" and not (result.get("expansion") or {}).get("added_terms"):
                strategy = str((result.get("expansion") or {}).get("strategy") or expansion_strategy)
                terms = self._drug_knowledge_terms(
                    strategy,
                    list(base_constraints.get("drugs") or []),
                    list(base_constraints.get("gene_alterations") or []),
                )
                result["expansion"] = {"strategy": strategy, "source": {"drug_alias": "all_drug_name_map.json", "pk_relation": "all_drug_pk_relation.json", "ddi_rule": "all_ddi_rule.json", "mutation_drug": "mutation_drug_map_min.json", "case_ddi": "case_ddi.json"}.get(strategy, ""), "added_terms": terms, "relation_type": {"drug_alias": "same_drug_identity", "pk_relation": "pharmacokinetic_relation", "ddi_rule": "mechanism_rule", "mutation_drug": "mutation_to_candidate_drug", "case_ddi": "direct_case_ddi_relation"}.get(strategy, "")}
                result["search_query"] = " ".join([str(result.get("search_query") or ""), *terms]).strip()
            return result
        except Exception as exc:
            logger.warning("Replanner failed, falling back to rules: %s", exc)
            return fallback
