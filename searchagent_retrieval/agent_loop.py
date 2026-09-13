from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from .router import RetrievalRouter
from .router import RoutePlan
from .planning import MultiStepPlanningAgent
from .execution_agent import RetrievalExecutionAgent
from .structured_output_recovery import configured_batch_size, recovery_trace, run_adaptive

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_timeline(
    *,
    current_round: int,
    round_memories: List[Dict[str, Any]],
    compaction_events: List[Dict[str, Any]],
    recent_limit: int = 2,
) -> Dict[str, Any]:
    compacted_rounds = sorted({
        int(value)
        for event in compaction_events
        for value in event.get("covered_rounds") or []
        if str(value).isdigit()
    })
    completed_rounds = sorted({
        int(memory.get("round") or 0) for memory in round_memories
        if int(memory.get("round") or 0) > 0
    })
    recent_full_rounds = completed_rounds[-max(1, recent_limit):]
    return {
        "current_agent_round": int(current_round),
        "latest_completed_agent_round": max(completed_rounds or [0]),
        "compacted_rounds": compacted_rounds,
        "compacted_round_range": {
            "start": min(compacted_rounds) if compacted_rounds else None,
            "end": max(compacted_rounds) if compacted_rounds else None,
        },
        "recent_full_rounds": recent_full_rounds,
        "instruction": (
            f"当前正在执行 Agent Round {current_round}；"
            f"历史摘要覆盖 Round {compacted_rounds or '无'}；"
            f"最近保留原始记忆的是 Round {recent_full_rounds or '无'}。"
            "Agent Round 是检索顺序，不是患者治疗周期；患者临床时间以证据 clinical_time/event_time 为准。"
        ),
    }

LLM_CITATION_VERDICTS = frozenset(
    {"strongly_supports", "partially_supports", "contradicts", "insufficient_evidence"}
)
RULE_CITATION_VERDICTS = frozenset(
    {
        "insufficient_evidence",
        "needs_caution_due_to_contradiction_or_risk",
        "not_supported",
        "supported_with_monitoring",
    }
)
POSITIVE_CITATION_VERDICTS = frozenset(
    {"supported_with_monitoring", "strongly_supports", "partially_supports"}
)
CAUTION_CITATION_VERDICTS = frozenset(
    {"needs_caution_due_to_contradiction_or_risk", "contradicts", "not_supported"}
)

_CITATION_SYSTEM_PROMPT = """你是 EvidenceReviewAgent，负责对本轮入选证据做一次性的医学解释和支持边界判断。你的结果是 StepMemory 和 AnswerMemory 的证据底稿；后续模块不应重新发明证据含义。

输入同时提供原始主问题、问题理解、当前原子Planning Step、实际检索query和rerank目标。原始主问题和已确认患者事实用于判断实体与临床边界；当前Step和rerank目标用于判断本轮信息增益；检索query只是召回表达，不得将其省略的条件当成患者事实。

对每条证据分别完成：
1. 忠实提取证据实际报告的最窄临床事实，不把模型知识、医生问题中的目标或其他证据内容写进该事实。
2. 对照医生问题中的患者、当前药物/方案、疾病与分子背景、结局、严重程度、时间窗和处置，判断哪些维度匹配、明确不匹配或未报告。
3. 给出 overall_role：direct_support、partial_support、analog_support、counter、risk、mixed 或 insufficient。如同一证据对不同结局方向不同，必须用 dimension_findings 分开记录，不得用单一总标签覆盖混合结果。
4. 生成 claim_scope：一条简短、可独立理解的最窄事实；保留原证据的主体和结局。不得把其他药物改成目标药物，不得把ALT/AST升高改成严重肝损伤，不得补时间、剂量、停药、减量、因果或总体疗效。
5. 区分“未报告”和“明确不匹配”：未报告不等于否定，也不能用于补全结论。
6. 如一条证据包含多个随访时点，dimension_findings必须按时间窗拆开。对“8–12周”问题，12周PR与11个月后进展是两个不同事实；后者只限定持久性，不得标为早期疗效counter。只有目标时间窗内明确无效/进展才是早期反证。

通用临床归因原则：局部/CNS疗效不等于全身疗效；影像缩小不自动等于RECIST PR；实验室异常不自动等于已确诊毒性；晚期耐药不自动反证早期疗效；联合方案结局不能全部归因于单药；不同药物通常只能提供类比支持。

必须通过 submit_evidence_review 函数提交审核结果。输出采用软Schema：chunk_id和claim_scope是稳定核心；其余医学维度可按证据内容增减，不要为填满字段而丢掉原文信息。推荐格式如下：
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
      "overall_role": "direct_support | partial_support | analog_support | counter | risk | mixed | insufficient",
      "target_entity_match": "exact | partial | class_analog | different | unknown",
      "supports_dimensions": [],
      "mismatched_dimensions": [],
      "unreported_dimensions": [],
      "claim_scope": "证据实际报告且能够支持的最窄事实",
      "dimension_findings": [{"dimension":"结局维度","finding":"原文事实","relation_to_target":"supports | contradicts | analog | unreported"}],
      "additional_findings": [],
      "summary": "简短说明支持边界和主要限制"
    }
  ]
}

verdict 判定标准：
- strongly_supports：多条高质量证据一致支持回答方向
- partially_supports：有部分支持证据，但证据不足或质量一般
- contradicts：存在明确反证、无效、进展或重要安全风险
- insufficient_evidence：几乎没有可用证据

评分范围均为 0-1。只输出能够作为 direct_support、partial_support、analog_support、counter、risk 或 mixed 进入后续推理的证据；判断为 irrelevant 或 insufficient 的证据直接省略，不要为了说明拒绝原因而罗列。保留条目的 chunk_id 必须与输入一致。不要跨证据拼接事实。"""


def _unique_list(values: List[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for value in values:
        item = value.strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _normalize_round_numbers(values: Any) -> List[int]:
    """Tolerate common LLM variants for a list of retrieval round numbers."""
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    normalized: List[int] = []
    for value in values:
        if isinstance(value, dict):
            value = next(
                (value[key] for key in ("round", "round_number", "step", "value") if key in value),
                None,
            )
        if isinstance(value, bool) or value is None:
            continue
        try:
            round_number = int(value)
        except (TypeError, ValueError):
            continue
        if round_number >= 0:
            normalized.append(round_number)
    return sorted(set(normalized))


@dataclass
class RetrievalEvidence:
    chunk_id: str
    evidence_level: str
    text: str
    pmid: Optional[str]
    title: Optional[str]
    relevance_signals: Dict[str, bool] = field(default_factory=dict)


@dataclass
class QueryState:
    original_query: str
    identified_entities: Dict[str, List[str]] = field(default_factory=dict)
    confirmed_constraints: Dict[str, Any] = field(default_factory=dict)
    search_hints: Dict[str, Any] = field(default_factory=dict)
    retrieved_evidence: List[RetrievalEvidence] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    next_retrieval_targets: List[str] = field(default_factory=list)
    stop_condition: str = "running"
    query_type: Optional[str] = None  # Deprecated trace field; planning is no longer category-driven.
    loop_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["retrieved_evidence"] = [asdict(item) for item in self.retrieved_evidence]
        return payload


@dataclass
class AgentLoopStep:
    step: int
    query: str
    query_type: Optional[str]
    selected_tools: List[str]
    citation_verdict: str
    contradiction_check: Dict[str, Any]
    missing_information: List[str]
    next_retrieval_targets: List[str]
    new_chunk_count: int
    plan_step_id: str = ""
    step_goal: str = ""
    replan_action: str = ""
    rerank_goal: str = ""


def _call_structured_llm(
    llm_client: "LLMClient",
    *,
    system: str,
    user: str,
    function_name: str,
    description: str,
    parameters: Dict[str, Any],
    temperature: float,
    max_output_tokens: int,
    strict: bool = True,
) -> Dict[str, Any]:
    return llm_client.call_function(
        system=system,
        user=user,
        function_name=function_name,
        description=description,
        parameters=parameters,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        strict=strict,
    )


_STRING_ARRAY = {"type": "array", "items": {"type": "string"}}

EVIDENCE_REVIEW_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": sorted(LLM_CITATION_VERDICTS)},
        "evidence_assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "relevance_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "credibility_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "is_supporting": {"type": "boolean"},
                    "is_contradicting": {"type": "boolean"},
                    "is_safety_risk": {"type": "boolean"},
                    "overall_role": {"type": "string"},
                    "evidence_role": {"type": "string"},
                    "target_entity_match": {"type": "string"},
                    "supports_dimensions": _STRING_ARRAY,
                    "mismatched_dimensions": _STRING_ARRAY,
                    "unreported_dimensions": _STRING_ARRAY,
                    "claim_scope": {"type": "string"},
                    "dimension_findings": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                    "additional_findings": _STRING_ARRAY,
                    "summary": {"type": "string"},
                },
                "required": ["chunk_id", "claim_scope"],
                "additionalProperties": True,
            },
        },
    },
    "required": ["verdict", "evidence_assessments"],
    "additionalProperties": True,
}

STEP_MEMORY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "accepted_evidence": {"type": "array", "items": {"type": "object", "properties": {
            "chunk_id": {"type": "string"}, "evidence_role": {"type": "string", "enum": ["direct_support", "partial_support", "analog_support", "counter", "risk"]},
            "target_entity_match": {"type": "string", "enum": ["exact", "partial", "class_analog", "different", "unknown"]},
            "supports_dimensions": _STRING_ARRAY, "unreported_dimensions": _STRING_ARRAY, "mismatched_dimensions": _STRING_ARRAY,
            "claim_scope": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["chunk_id", "evidence_role", "target_entity_match", "supports_dimensions", "unreported_dimensions", "mismatched_dimensions", "claim_scope", "reason"], "additionalProperties": False}},
        "rejected_evidence": {"type": "array", "items": {"type": "object", "properties": {
            "chunk_id": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["chunk_id", "reason"], "additionalProperties": False}},
        "question_information_gain": {"type": "object", "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 1}, "new_facts": _STRING_ARRAY,
            "resolved_questions": _STRING_ARRAY, "new_conflicts": _STRING_ARRAY, "new_chunk_ids": _STRING_ARRAY},
            "required": ["score", "new_facts", "resolved_questions", "new_conflicts", "new_chunk_ids"], "additionalProperties": False},
        "goal_evaluation": {"type": "object", "properties": {
            "matched_goal_count": {"type": "integer", "minimum": 0},
            "best_goal_relevance": {"type": "number", "minimum": 0, "maximum": 1},
            "success_criteria_met": {"type": "boolean"},
            "completion_status": {"type": "string", "enum": ["not_met", "minimally_met", "sufficiently_met"]},
            "critical_gaps": _STRING_ARRAY,
            "optional_gaps": _STRING_ARRAY,
            "marginal_value_of_more_search": {"type": "number", "minimum": 0, "maximum": 1},
            "recommended_stop": {"type": "boolean"},
            "query_database_status": {"type": "string", "enum": ["more_available", "exhausted", "uncertain"]},
            "exhaustion_reason": {"type": "string"},
            "recommended_query_change": {"type": "string"},
            "queries_attempted": {"type": "array", "items": {"type": "string"}},
            "observed_gaps": _STRING_ARRAY,
            "observed_failures": _STRING_ARRAY},
            "required": ["matched_goal_count", "best_goal_relevance", "success_criteria_met", "completion_status", "critical_gaps", "optional_gaps", "marginal_value_of_more_search", "recommended_stop", "query_database_status", "exhaustion_reason", "recommended_query_change", "queries_attempted", "observed_gaps", "observed_failures"], "additionalProperties": False},
    },
    "required": ["accepted_evidence", "rejected_evidence", "question_information_gain", "goal_evaluation"],
    "additionalProperties": False,
}

# Backward-compatible name retained for live schema probes and old clients.
ROUND_MEMORY_SCHEMA = STEP_MEMORY_SCHEMA

ANSWER_MEMORY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "claim_id": {"type": "string"}, "claim": {"type": "string"},
            "status": {"type": "string", "enum": ["provisional", "supported", "contested"]},
            "supporting_chunk_ids": _STRING_ARRAY, "contradicting_chunk_ids": _STRING_ARRAY,
            "direct_support_chunk_ids": _STRING_ARRAY, "partial_support_chunk_ids": _STRING_ARRAY,
            "analog_support_chunk_ids": _STRING_ARRAY,
            "subject_entity": {"type": "string"}, "claim_scope": {"type": "string"},
            "support_level": {"type": "string", "enum": ["direct", "partial", "analog", "unsupported"]},
            "source_rounds": {"type": "array", "items": {"type": "integer", "minimum": 0}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "safety_status": {"type": "string"}},
            "required": ["claim_id", "claim", "status", "supporting_chunk_ids", "contradicting_chunk_ids", "direct_support_chunk_ids", "partial_support_chunk_ids", "analog_support_chunk_ids", "subject_entity", "claim_scope", "support_level", "source_rounds", "confidence", "safety_status"],
            "additionalProperties": False}},
        "informative_rounds": {"type": "array", "items": {"type": "integer", "minimum": 0}},
    },
    "required": ["claims", "informative_rounds"],
    "additionalProperties": False,
}


def _clamp_score(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, score))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _compact_evidence_rows(items: List[Dict[str, Any]], limit: int = 12, text_limit: int = 500) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in items[:limit]:
        row = {key: item.get(key) for key in (
            "chunk_id", "doc_id", "evidence_level", "pmid", "title",
            "relevance_signals", "llm_relevance_score", "llm_credibility_score", "llm_summary",
            "evidence_role", "overall_role", "target_entity_match", "supports_dimensions", "unreported_dimensions",
            "mismatched_dimensions", "claim_scope", "support_level", "source_round",
            "dimension_findings", "additional_findings",
        )}
        text = " ".join(str(item.get("text") or "").split())
        row["text"] = text[:text_limit]
        rows.append(row)
    return rows


class EvidenceReviewAgent:
    def __init__(
        self,
        llm_client: Optional["LLMClient"] = None,
        use_llm: bool = False,
        max_evidence_for_llm: int = 15,
    ) -> None:
        self.llm_client = llm_client
        self.use_llm = use_llm
        self.max_evidence_for_llm = max(1, max_evidence_for_llm)

    def _classify_assessed_evidence(
        self,
        assessed: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        supporting: List[Dict[str, Any]] = []
        contradicting: List[Dict[str, Any]] = []
        insufficient: List[Dict[str, Any]] = []
        safety_risks: List[Dict[str, Any]] = []

        for item in assessed:
            signals = item.get("relevance_signals") or {}
            if signals.get("is_supporting"):
                supporting.append(item)
            if signals.get("is_contradicting"):
                contradicting.append(item)
            if signals.get("is_safety_risk"):
                safety_risks.append(item)
            if not signals.get("is_supporting") and not signals.get("is_contradicting"):
                insufficient.append(item)
        return supporting, contradicting, insufficient, safety_risks

    def _derive_verdict_from_signals(
        self,
        supporting: List[Dict[str, Any]],
        contradicting: List[Dict[str, Any]],
        safety_risks: List[Dict[str, Any]],
        use_llm_verdicts: bool,
    ) -> str:
        support_count = len(supporting)
        counter_count = len(contradicting)
        risk_count = len(safety_risks)

        if use_llm_verdicts:
            if support_count == 0 and counter_count == 0 and risk_count == 0:
                return "insufficient_evidence"
            if counter_count + risk_count > max(1, support_count):
                return "contradicts"
            if support_count == 0:
                return "contradicts"
            if support_count >= 2 and counter_count == 0 and risk_count == 0:
                return "strongly_supports"
            return "partially_supports"

        if support_count == 0 and counter_count == 0 and risk_count == 0:
            return "insufficient_evidence"
        if counter_count + risk_count > max(1, support_count):
            return "needs_caution_due_to_contradiction_or_risk"
        if support_count == 0:
            return "not_supported"
        return "supported_with_monitoring"

    def _review_rules(self, route_result: Dict[str, Any]) -> Dict[str, Any]:
        contradiction = route_result.get("contradiction_check", {})
        evidence_layering = route_result.get("evidence_layering", {})
        assessed = list(evidence_layering.get("assessed_evidence", []))

        supporting, contradicting, insufficient, safety_risks = self._classify_assessed_evidence(assessed)
        verdict = contradiction.get("verdict")
        if verdict not in RULE_CITATION_VERDICTS:
            verdict = self._derive_verdict_from_signals(
                supporting=supporting,
                contradicting=contradicting,
                safety_risks=safety_risks,
                use_llm_verdicts=False,
            )

        return {
            "verdict": verdict,
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting,
            "insufficient_evidence": insufficient,
            "safety_risks": safety_risks,
            "layer_distribution": evidence_layering.get("layer_distribution", []),
            "top_supporting_hit_ids": contradiction.get("top_supporting_hit_ids", []),
            "top_counter_hit_ids": contradiction.get("top_counter_hit_ids", []),
            "review_mode": "rules",
        }

    def _build_llm_user_prompt_with_context(
        self,
        route_result: Dict[str, Any],
        assessed: List[Dict[str, Any]],
    ) -> str:
        evidence_payload: List[Dict[str, Any]] = []
        for item in assessed[: self.max_evidence_for_llm]:
            text = str(item.get("text") or "").strip().replace("\n", " ")
            if len(text) > 500:
                text = f"{text[:497]}..."
            evidence_payload.append(
                {
                    "chunk_id": str(item.get("chunk_id") or ""),
                    "evidence_level": str(item.get("evidence_level") or "case_report_evidence"),
                    "pmid": item.get("pmid"),
                    "title": item.get("title"),
                    "text": text,
                    "rule_signals": item.get("relevance_signals") or {},
                }
            )

        payload = {
            "main_question": route_result.get("main_question") or route_result.get("query") or "",
            "problem_representation": route_result.get("problem_representation") or {},
            "current_plan_step": route_result.get("plan_step") or {},
            "executed_query": route_result.get("planned_query") or route_result.get("query") or "",
            "rerank_goal": route_result.get("rerank_goal") or (route_result.get("evidence_layering") or {}).get("query") or "",
            "evidence": evidence_payload,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _merge_llm_assessments(
        self,
        assessed: List[Dict[str, Any]],
        llm_payload: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], str]:
        assessments = llm_payload.get("evidence_assessments") or []
        if not isinstance(assessments, list):
            assessments = []

        by_chunk_id: Dict[str, Dict[str, Any]] = {}
        for entry in assessments:
            if not isinstance(entry, dict):
                continue
            chunk_id = str(entry.get("chunk_id") or "").strip()
            if chunk_id:
                by_chunk_id[chunk_id] = entry

        merged: List[Dict[str, Any]] = []
        for item in assessed:
            row = dict(item)
            chunk_id = str(row.get("chunk_id") or "").strip()
            llm_entry = by_chunk_id.get(chunk_id)
            # Omission is the model's explicit discard decision.  Do not keep
            # an unreviewed placeholder in insufficient_evidence: it bloats
            # StepMemory and can be mistaken for a clinically meaningful row.
            if not llm_entry:
                continue

            role = str(llm_entry.get("overall_role") or llm_entry.get("evidence_role") or "insufficient")
            dimension_relations = {
                str(finding.get("relation_to_target") or "").lower()
                for finding in llm_entry.get("dimension_findings") or []
                if isinstance(finding, dict)
            }
            if role == "insufficient":
                if "supports" in dimension_relations and "contradicts" in dimension_relations:
                    role = "mixed"
                elif "analog" in dimension_relations:
                    role = "analog_support"
                elif "supports" in dimension_relations:
                    role = "partial_support"
                elif "contradicts" in dimension_relations:
                    role = "counter"
            if role in {"irrelevant", "insufficient"} and not any(
                _as_bool(llm_entry.get(key))
                for key in ("is_supporting", "is_contradicting", "is_safety_risk")
            ):
                continue
            signals = {
                "is_supporting": _as_bool(llm_entry.get("is_supporting")) or role in {"direct_support", "partial_support", "analog_support", "mixed"} or "supports" in dimension_relations,
                "is_contradicting": _as_bool(llm_entry.get("is_contradicting")) or role in {"counter", "mixed"} or "contradicts" in dimension_relations,
                "is_safety_risk": _as_bool(llm_entry.get("is_safety_risk")) or role == "risk",
            }
            row["relevance_signals"] = signals
            row["llm_relevance_score"] = _clamp_score(llm_entry.get("relevance_score"))
            row["llm_credibility_score"] = _clamp_score(llm_entry.get("credibility_score"))
            summary = str(llm_entry.get("summary") or "").strip()
            if summary:
                row["llm_summary"] = summary
            for key in (
                "evidence_role", "overall_role", "target_entity_match", "supports_dimensions",
                "mismatched_dimensions", "unreported_dimensions", "claim_scope",
                "dimension_findings", "additional_findings",
            ):
                if key in llm_entry:
                    row[key] = llm_entry.get(key)
            # Keep the legacy single-role lane consumable while preserving the
            # richer mixed interpretation for AnswerMemory and generation.
            row["overall_role"] = role
            row["evidence_role"] = "partial_support" if role == "mixed" else role
            merged.append(row)

        supporting, contradicting, insufficient, safety_risks = self._classify_assessed_evidence(merged)
        verdict = str(llm_payload.get("verdict") or "").strip()
        if verdict not in LLM_CITATION_VERDICTS:
            verdict = self._derive_verdict_from_signals(
                supporting=supporting,
                contradicting=contradicting,
                safety_risks=safety_risks,
                use_llm_verdicts=True,
            )
        return merged, verdict

    def _review_with_llm(self, route_result: Dict[str, Any]) -> Dict[str, Any]:
        if self.llm_client is None:
            raise ValueError("llm_client is required for LLM citation review")

        evidence_layering = route_result.get("evidence_layering", {})
        contradiction = route_result.get("contradiction_check", {})
        assessed = list(evidence_layering.get("assessed_evidence", []))
        if not assessed:
            return {
                "verdict": "insufficient_evidence",
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "insufficient_evidence": [],
                "safety_risks": [],
                "layer_distribution": evidence_layering.get("layer_distribution", []),
                "top_supporting_hit_ids": contradiction.get("top_supporting_hit_ids", []),
                "top_counter_hit_ids": contradiction.get("top_counter_hit_ids", []),
                "review_mode": "llm",
            }

        batch_size = configured_batch_size("EVIDENCE_REVIEW_BATCH_SIZE", 4)

        def review_part(part: List[Dict[str, Any]], part_number: int, part_count: int) -> Dict[str, Any]:
            user_prompt = self._build_llm_user_prompt_with_context(
                route_result=route_result,
                assessed=part,
            )
            user_prompt += (
                f"\n\n这是同一个 EvidenceReview 任务的第 {part_number}/{part_count} 部分。"
                "只审核本部分证据，并返回一个自身完整的 JSON/function 参数对象；不要续写上一部分。"
            )
            payload = _call_structured_llm(
                self.llm_client,
                system=_CITATION_SYSTEM_PROMPT,
                user=user_prompt,
                function_name="submit_evidence_review",
                description="Submit the retained evidence assessments for this bounded part.",
                parameters=EVIDENCE_REVIEW_SCHEMA,
                temperature=0.2,
                max_output_tokens=8000,
                strict=False,
            )
            if payload.get("verdict") not in LLM_CITATION_VERDICTS or not isinstance(
                payload.get("evidence_assessments"), list
            ):
                raise ValueError("LLM evidence review failed semantic validation")
            return payload

        part_payloads, recovered_from_truncation = run_adaptive(
            assessed,
            batch_size=batch_size,
            call_part=review_part,
        )
        llm_payload = {
            # Derive the task-wide verdict from the retained evidence after
            # deterministic merging; per-part verdicts are not comparable.
            "verdict": "",
            "evidence_assessments": [
                item
                for payload in part_payloads
                for item in (payload.get("evidence_assessments") or [])
                if isinstance(item, dict)
            ],
        }

        merged, verdict = self._merge_llm_assessments(assessed=assessed, llm_payload=llm_payload)
        supporting, contradicting, insufficient, safety_risks = self._classify_assessed_evidence(merged)

        return {
            "verdict": verdict,
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting,
            "insufficient_evidence": insufficient,
            "safety_risks": safety_risks,
            "layer_distribution": evidence_layering.get("layer_distribution", []),
            "top_supporting_hit_ids": contradiction.get("top_supporting_hit_ids", []),
            "top_counter_hit_ids": contradiction.get("top_counter_hit_ids", []),
            "review_mode": "llm",
            "structured_output_recovery": recovery_trace(
                input_items=len(assessed),
                part_count=len(part_payloads),
                batch_size=batch_size,
                initial_attempt_failed=recovered_from_truncation,
            ),
            "omitted_evidence_ids": [
                str(item.get("chunk_id") or "")
                for item in assessed
                if str(item.get("chunk_id") or "") not in {
                    str(row.get("chunk_id") or "") for row in merged
                }
            ],
            "llm_evidence_assessments": [
                {
                    "chunk_id": item.get("chunk_id"),
                    "relevance_score": item.get("llm_relevance_score"),
                    "credibility_score": item.get("llm_credibility_score"),
                    "is_supporting": (item.get("relevance_signals") or {}).get("is_supporting"),
                    "is_contradicting": (item.get("relevance_signals") or {}).get("is_contradicting"),
                    "is_safety_risk": (item.get("relevance_signals") or {}).get("is_safety_risk"),
                    "summary": item.get("llm_summary"),
                }
                for item in merged
            ],
        }

    def review(self, route_result: Dict[str, Any]) -> Dict[str, Any]:
        if self.use_llm and self.llm_client is not None:
            try:
                return self._review_with_llm(route_result=route_result)
            except Exception as exc:
                if os.environ.get("STRICT_LLM_PIPELINE") == "1":
                    raise
                logger.warning("LLM citation review failed, falling back to rules: %s", exc)
                return self._review_rules(route_result=route_result)
        return self._review_rules(route_result=route_result)


class CitationAgent(EvidenceReviewAgent):
    """Resolve answer claims back to database evidence and original source files.

    ``review`` is inherited temporarily for API compatibility.  The retrieval loop
    uses :class:`EvidenceReviewAgent`; new code should use ``trace_claims`` here.
    """

    @staticmethod
    def _citation_for(item: Dict[str, Any]) -> Dict[str, Any]:
        citation = item.get("citation_json") or item.get("citation") or {}
        if not isinstance(citation, dict):
            citation = {}
        metadata = item.get("metadata_json") or item.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            "chunk_id": item.get("chunk_id") or item.get("id"),
            "doc_id": item.get("doc_id") or metadata.get("doc_id"),
            "case_id": item.get("case_id") or metadata.get("case_id"),
            "event_id": item.get("event_id") or metadata.get("event_id"),
            "pmid": item.get("pmid") or metadata.get("pmid"),
            "title": item.get("title") or metadata.get("title"),
            "source_file": citation.get("source_file") or item.get("source_file") or metadata.get("source_file"),
            "field_path": citation.get("field_path"),
            "start_char": citation.get("start_char"),
            "end_char": citation.get("end_char"),
            "chunk_order": citation.get("chunk_order"),
            "evidence_level": item.get("evidence_level") or metadata.get("evidence_level"),
            "quoted_span": str(item.get("text") or "").strip()[:500],
        }

    def trace_claims(
        self,
        claims: List[Dict[str, Any]],
        evidence_by_id: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        claim_citations: List[Dict[str, Any]] = []
        unresolved: List[str] = []
        for claim in claims:
            claim_id = str(claim.get("claim_id") or "")
            refs = _unique_list(
                [str(value) for value in (claim.get("supporting_chunk_ids") or [])]
                + [str(value) for value in (claim.get("contradicting_chunk_ids") or [])]
            )
            citations = [self._citation_for(evidence_by_id[ref]) for ref in refs if ref in evidence_by_id]
            missing = [ref for ref in refs if ref not in evidence_by_id]
            if not citations:
                unresolved.append(claim_id)
            claim_citations.append(
                {"claim_id": claim_id, "claim": claim.get("claim"), "citations": citations, "missing_refs": missing}
            )
        return {"claim_citations": claim_citations, "unresolved_claim_ids": unresolved}


REPLANNER_MEMORY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "short_memory": {
            "type": "object",
            "properties": {
                "active_step_id": {"type": "string"},
                "recent_information_gain": {"type": "array", "items": {"type": "string"}},
                "recent_effective_strategies": {"type": "array", "items": {"type": "string"}},
                "recent_failed_strategies": {"type": "array", "items": {"type": "string"}},
                "current_critical_gaps": {"type": "array", "items": {"type": "string"}},
                "recent_conflicts": {"type": "array", "items": {"type": "string"}},
                "avoid_repeating": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["active_step_id", "recent_information_gain", "recent_effective_strategies", "recent_failed_strategies", "current_critical_gaps", "recent_conflicts", "avoid_repeating"],
            "additionalProperties": False,
        },
        "long_memory": {
            "type": "object",
            "properties": {
                "step_progress": {"type": "array", "items": {"type": "string"}},
                "global_key_findings": {"type": "array", "items": {"type": "string"}},
                "global_critical_gaps": {"type": "array", "items": {"type": "string"}},
                "cross_step_constraints": {"type": "array", "items": {"type": "string"}},
                "unresolved_conflicts": {"type": "array", "items": {"type": "string"}},
                "search_strategy_lessons": {"type": "array", "items": {"type": "string"}},
                "protected_safety_information": {"type": "array", "items": {"type": "string"}},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["step_progress", "global_key_findings", "global_critical_gaps", "cross_step_constraints", "unresolved_conflicts", "search_strategy_lessons", "protected_safety_information", "evidence_ids"],
            "additionalProperties": False,
        },
    },
    "required": ["short_memory", "long_memory"],
    "additionalProperties": False,
}


class ReplannerMemoryAgent:
    """Periodically rewrite compact short/long memories for the Replanner."""

    SYSTEM_PROMPT = """你维护供临床检索 Replanner 使用的长短期记忆。把旧记忆与尚未合并的 StepMemory 重新提炼成自包含的新版本；新版本完全替代旧版本，因此不机械追加。省略重复、已解决、低价值和被新信息替代的内容，这种提炼就是遗忘。短期记忆只保留近期会影响下一次检索的变化；长期记忆保留全局步骤进展、关键结论、关键缺口、跨步骤约束、未解决冲突、检索策略经验和临床安全信息。关键结论、冲突与安全信息必须保留真实 evidence/chunk id。不要虚构证据。必须通过 maintain_replanner_memory 提交。"""

    def __init__(self, llm_client: Optional["LLMClient"] = None, use_llm: bool = True, consolidate_every: int = 3, token_threshold: int = 2800) -> None:
        self.llm_client = llm_client
        self.use_llm = use_llm
        self.consolidate_every = max(1, consolidate_every)
        self.token_threshold = max(256, token_threshold)
        self.preemptive_threshold = max(256, int(self.token_threshold * 0.85))

    @staticmethod
    def estimate_tokens(value: Any) -> int:
        # A conservative dependency-free estimate for mixed Chinese/JSON text.
        return max(1, len(json.dumps(value, ensure_ascii=False, separators=(",", ":"))) // 2)

    def should_consolidate(self, pending: List[Dict[str, Any]], short_memory: Dict[str, Any], long_memory: Dict[str, Any]) -> bool:
        return len(pending) >= self.consolidate_every or self.estimate_tokens({"short": short_memory, "long": long_memory, "pending": pending}) >= self.preemptive_threshold

    @staticmethod
    def _extend_unique(target: List[str], values: List[Any], limit: int = 8) -> None:
        seen = {str(item) for item in target}
        for value in values:
            item = str(value).strip()
            if not item or item in seen:
                continue
            target.append(item)
            seen.add(item)
            if len(target) >= limit:
                break

    def _rule_consolidate(
        self,
        *,
        active_step_id: str,
        short_memory: Dict[str, Any],
        long_memory: Dict[str, Any],
        pending_step_memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        recent = [dict(item) for item in pending_step_memories]
        short = {
            "active_step_id": active_step_id,
            "recent_information_gain": list(short_memory.get("recent_information_gain") or []),
            "recent_effective_strategies": list(short_memory.get("recent_effective_strategies") or []),
            "recent_failed_strategies": list(short_memory.get("recent_failed_strategies") or []),
            "current_critical_gaps": list(short_memory.get("current_critical_gaps") or []),
            "recent_conflicts": list(short_memory.get("recent_conflicts") or []),
            "avoid_repeating": list(short_memory.get("avoid_repeating") or []),
        }
        long = {
            "step_progress": list(long_memory.get("step_progress") or []),
            "global_key_findings": list(long_memory.get("global_key_findings") or []),
            "global_critical_gaps": list(long_memory.get("global_critical_gaps") or []),
            "cross_step_constraints": list(long_memory.get("cross_step_constraints") or []),
            "unresolved_conflicts": list(long_memory.get("unresolved_conflicts") or []),
            "search_strategy_lessons": list(long_memory.get("search_strategy_lessons") or []),
            "protected_safety_information": list(long_memory.get("protected_safety_information") or []),
            "evidence_ids": list(long_memory.get("evidence_ids") or []),
        }
        for memory in recent:
            round_number = memory.get("round")
            plan_step_id = str(memory.get("plan_step_id") or active_step_id)
            goal = memory.get("goal_evaluation") or {}
            gain = memory.get("question_information_gain") or {}
            actions = memory.get("retrieval_actions") or []
            action = actions[0] if actions else {}
            query = str(action.get("query") or "")
            selected_tools = [str(name) for name in action.get("selected_tools") or []]
            long["step_progress"].append(f"{plan_step_id}@r{round_number}:{goal.get('completion_status') or 'unknown'}")
            self._extend_unique(long["global_key_findings"], [*(gain.get("new_facts") or []), *(goal.get("observed_gaps") or [])])
            self._extend_unique(long["global_critical_gaps"], [*(goal.get("critical_gaps") or []), *(goal.get("optional_gaps") or [])], limit=12)
            self._extend_unique(long["cross_step_constraints"], [goal.get("database_coverage_status"), goal.get("query_direction_status"), goal.get("exhaustion_reason"), goal.get("recommended_query_change")])
            self._extend_unique(long["unresolved_conflicts"], list(goal.get("observed_failures") or []), limit=12)
            self._extend_unique(long["search_strategy_lessons"], [f"{plan_step_id}:{','.join(selected_tools[:3])}", goal.get("search_attempt_status"), goal.get("query_direction_status"), goal.get("database_coverage_status"), goal.get("recommended_query_change")], limit=12)
            if goal.get("query_direction_status") == "locally_exhausted":
                self._extend_unique(short["recent_failed_strategies"], [f"query_locally_exhausted:{query}", goal.get("recommended_query_change")], limit=8)
            if (gain.get("score") or 0) >= 0.5 or gain.get("new_chunk_ids"):
                self._extend_unique(short["recent_information_gain"], [f"{plan_step_id}:{query}"], limit=8)
                self._extend_unique(short["recent_effective_strategies"], selected_tools[:3], limit=8)
            else:
                self._extend_unique(short["recent_failed_strategies"], [f"{plan_step_id}:{query}"], limit=8)
            self._extend_unique(short["current_critical_gaps"], list(goal.get("critical_gaps") or [])[:4], limit=8)
            self._extend_unique(short["recent_conflicts"], list(goal.get("observed_failures") or [])[:4], limit=8)
            self._extend_unique(short["avoid_repeating"], [query, goal.get("recommended_query_change")], limit=12)
            for item in (memory.get("accepted_evidence") or [])[:6]:
                chunk_id = str(item.get("chunk_id") or "")
                if chunk_id:
                    self._extend_unique(long["evidence_ids"], [chunk_id], limit=60)
                    if str(item.get("evidence_role") or "") == "risk":
                        self._extend_unique(long["protected_safety_information"], [chunk_id], limit=20)
        return {"short_memory": short, "long_memory": long, "consolidated": True, "consumed_count": len(recent), "memory_mode": "rules"}

    def maintain(self, *, main_question: str, plan: Dict[str, Any], active_step_id: str, short_memory: Dict[str, Any], long_memory: Dict[str, Any], pending_step_memories: List[Dict[str, Any]], force: bool = False) -> Dict[str, Any]:
        unchanged = {"short_memory": dict(short_memory), "long_memory": dict(long_memory), "consolidated": False, "consumed_count": 0}
        if not pending_step_memories or (not force and not self.should_consolidate(pending_step_memories, short_memory, long_memory)):
            return unchanged
        estimated_tokens = self.estimate_tokens({"short": short_memory, "long": long_memory, "pending": pending_step_memories})
        if estimated_tokens >= self.preemptive_threshold:
            return self._rule_consolidate(
                active_step_id=active_step_id,
                short_memory=short_memory,
                long_memory=long_memory,
                pending_step_memories=pending_step_memories,
            )
        if not self.use_llm or self.llm_client is None:
            return self._rule_consolidate(
                active_step_id=active_step_id,
                short_memory=short_memory,
                long_memory=long_memory,
                pending_step_memories=pending_step_memories,
            )
        payload = {
            "main_question": main_question,
            "plan_steps": plan.get("steps") or [],
            "active_step_id": active_step_id,
            "previous_short_memory": short_memory,
            "previous_long_memory": long_memory,
            "new_step_memories": pending_step_memories,
        }
        try:
            result = self.llm_client.call_function(
                system=self.SYSTEM_PROMPT,
                user=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                function_name="maintain_replanner_memory",
                description="Rewrite compact short-term and long-term Replanner memory.",
                parameters=REPLANNER_MEMORY_SCHEMA,
                temperature=0.1,
                max_output_tokens=8000,
            )
            return {"short_memory": dict(result.get("short_memory") or {}), "long_memory": dict(result.get("long_memory") or {}), "consolidated": True, "consumed_count": len(pending_step_memories)}
        except Exception as exc:
            logger.warning("Replanner memory consolidation failed; retaining previous memory: %s", exc)
            return self._rule_consolidate(
                active_step_id=active_step_id,
                short_memory=short_memory,
                long_memory=long_memory,
                pending_step_memories=pending_step_memories,
            )


class StepMemoryAgent:
    """Summarize a batch of deterministic micro-retrieval ledger entries."""

    SYSTEM_PROMPT = """你是 StepMemoryAgent，负责把本轮检索过程和 EvidenceReview 结果登记成可供 Replanner 使用的轮次记录。你不重新解释医学证据，也不重新撰写证据结论。

EvidenceReview 已负责判断每条证据的 evidence_role、target_entity_match、supports_dimensions、mismatched_dimensions、unreported_dimensions 和 claim_scope。你应默认继承这些字段。只有在输入内部存在清楚、可定位的错误时，才修改发生错误的单个字段，并在 reason 中说明“原值、修改值和依据”；不得因为句子不够流畅或不够完整而重写 claim_scope，不得把整条 partial/analog 证据改造成新的临床结论。

你的主要职责是：
- 根据真实chunk_id登记接受与拒绝；
- 汇总本轮新增信息、冲突与已解决问题；
- 评估当前Planning Step的完成度和剩余缺口；
- 保存ExecutionAgent对具体query是否穷尽的判断。

你不决定下一步动作。区分已达到最低目标但仍有关键缺口minimally_met，与证据已充分sufficiently_met。具体query穷尽不代表整个Planning Step完成。必须通过commit_step_memory提交。"""

    @staticmethod
    def build_micro_ledger(route_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        ledger: List[Dict[str, Any]] = []
        search_tools = {"structured_search", "dense_search", "bm25_search", "hybrid_search"}
        for call in route_result.get("tool_trace") or []:
            name = str(call.get("tool_name") or "")
            if name == "execute_retrieval_batch":
                args, result = call.get("arguments") or {}, call.get("result") or {}
                actions = [item for item in args.get("actions") or [] if isinstance(item, dict)]
                action_results = [item for item in result.get("action_results") or [] if isinstance(item, dict)]
                for index, action in enumerate(actions):
                    tool = str(action.get("tool") or "")
                    if tool not in search_tools:
                        continue
                    tool_result = action_results[index] if index < len(action_results) else {}
                    ledger.append({
                        "attempt": len(ledger) + 1,
                        "tool": tool,
                        "query": action.get("query"),
                        "constraints": action.get("constraints") or {},
                        "spaces": action.get("spaces") or [],
                        "requested_top_k": action.get("top_k"),
                        "hit_count": tool_result.get("hit_count"),
                        "known_candidate_count": tool_result.get("known_candidate_count"),
                        "top_hit_ids": [str(hit.get("id") or "") for hit in (tool_result.get("hits") or [])[:5]],
                        "error": tool_result.get("error"),
                    })
                continue
            if name not in search_tools:
                continue
            args, result = call.get("arguments") or {}, call.get("result") or {}
            ledger.append({
                "attempt": len(ledger) + 1,
                "tool": name,
                "query": args.get("query"),
                "constraints": args.get("constraints") or {},
                "hit_count": result.get("hit_count", len(result.get("hits") or [])),
                "known_candidate_count": result.get("known_candidate_count"),
                "top_hit_ids": [str(hit.get("id") or "") for hit in (result.get("hits") or [])[:5]],
                "error": result.get("error"),
            })
        if not ledger:
            for name in route_result.get("selected_tools") or []:
                if name in search_tools:
                    ledger.append({"attempt": len(ledger) + 1, "tool": name, "query": route_result.get("query"), "constraints": route_result.get("constraints") or {}, "hit_count": None, "error": None})
        return ledger

    def __init__(self, llm_client: Optional["LLMClient"] = None, use_llm: bool = True, min_relevance: float = 0.35) -> None:
        # EvidenceReview owns the medical interpretation. StepMemory is always
        # deterministic so a second LLM cannot reinterpret the same evidence.
        self.llm_client = None
        self.use_llm = False
        self.min_relevance = min_relevance

    def maintain(
        self,
        *,
        round_number: int,
        main_question: str,
        query: str,
        route_result: Dict[str, Any],
        evidence_review: Dict[str, Any],
        seen_chunk_ids: Set[str],
        plan_step: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.use_llm and self.llm_client is not None:
            try:
                compact_review = {
                    "verdict": evidence_review.get("verdict"),
                    "layer_distribution": evidence_review.get("layer_distribution") or [],
                }
                for key in ("supporting_evidence", "contradicting_evidence", "safety_risks", "insufficient_evidence"):
                    compact_review[key] = _compact_evidence_rows(evidence_review.get(key) or [])
                micro_ledger = self.build_micro_ledger(route_result)
                payload = {
                    "round": round_number,
                    "main_question": main_question,
                    "executed_query": query,
                    "active_plan_step": plan_step or {},
                    "step_execution_report": {
                        key: (route_result.get("step_execution_report") or {}).get(key)
                        for key in ("execution_status", "summary", "evidence_found", "unresolved_gaps", "query_database_status", "exhaustion_reason", "recommended_query_change", "queries_attempted")
                        if key in (route_result.get("step_execution_report") or {})
                    },
                    "micro_retrieval_ledger": micro_ledger,
                    "retrieval_plan": {
                        "query_type": route_result.get("query_type"),
                        "selected_tools": route_result.get("selected_tools") or [],
                        "constraints": route_result.get("constraints") or {},
                        "rerank_goal": route_result.get("rerank_goal"),
                    },
                    "evidence_review": compact_review,
                    "previously_seen_chunk_ids": sorted(seen_chunk_ids)[-20:],
                }
                prompt = (
                    f"输入：\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
                    "输出 JSON：{\"accepted_evidence\":[{\"chunk_id\":\"\",\"evidence_role\":\"direct_support|partial_support|analog_support|counter|risk\",\"target_entity_match\":\"exact|partial|class_analog|different|unknown\",\"supports_dimensions\":[],\"unreported_dimensions\":[],\"mismatched_dimensions\":[],\"claim_scope\":\"证据实际能支持的最窄论断\",\"reason\":\"\"}],"
                    "\"rejected_evidence\":[{\"chunk_id\":\"\",\"reason\":\"\"}],"
                    "\"question_information_gain\":{\"score\":0.0,\"new_facts\":[],\"resolved_questions\":[],\"new_conflicts\":[],\"new_chunk_ids\":[]},"
                    "\"goal_evaluation\":{\"matched_goal_count\":0,\"best_goal_relevance\":0.0,\"success_criteria_met\":false,\"completion_status\":\"not_met|minimally_met|sufficiently_met\",\"critical_gaps\":[],\"optional_gaps\":[],\"marginal_value_of_more_search\":0.0,\"recommended_stop\":false,\"query_database_status\":\"more_available|exhausted|uncertain\",\"exhaustion_reason\":\"\",\"recommended_query_change\":\"\",\"queries_attempted\":[],\"observed_gaps\":[],\"observed_failures\":[]}}。"
                    "score 范围0-1；只允许使用输入中存在的 chunk_id；只记录观察，不得建议下一轮动作。"
                    "accepted_evidence默认逐字段复制EvidenceReview结果。只修正可明确指出依据的错误字段；不得整体改写claim_scope。"
                )
                parsed = _call_structured_llm(
                    self.llm_client,
                    system=self.SYSTEM_PROMPT,
                    user=prompt,
                    function_name="commit_step_memory",
                    description="Commit one planning step's micro-retrieval summary and information gain.",
                    parameters=STEP_MEMORY_SCHEMA,
                    temperature=0.2,
                    max_output_tokens=8000,
                )
                if not parsed:
                    raise ValueError("invalid round memory JSON")
                by_id: Dict[str, Dict[str, Any]] = {}
                for group in ("supporting_evidence", "contradicting_evidence", "safety_risks", "insufficient_evidence"):
                    for item in evidence_review.get(group) or []:
                        if item.get("chunk_id"):
                            by_id[str(item["chunk_id"])] = dict(item)
                accepted = []
                for decision in parsed.get("accepted_evidence") or []:
                    chunk_id = str(decision.get("chunk_id") or "")
                    if chunk_id not in by_id:
                        continue
                    item = dict(by_id[chunk_id])
                    item["evidence_role"] = str(decision.get("evidence_role") or "partial_support")
                    for key in ("target_entity_match", "supports_dimensions", "unreported_dimensions", "mismatched_dimensions", "claim_scope"):
                        item[key] = decision.get(key)
                    item["memory_reason"] = str(decision.get("reason") or "")
                    item["source_round"] = round_number
                    accepted.append(item)
                q_gain = dict(parsed.get("question_information_gain") or {})
                q_gain["score"] = _clamp_score(q_gain.get("score"))
                q_gain["new_chunk_ids"] = [str(x) for x in q_gain.get("new_chunk_ids") or [] if str(x) in by_id and str(x) not in seen_chunk_ids]
                goal_evaluation = dict(parsed.get("goal_evaluation") or {})
                execution_report = route_result.get("step_execution_report") or {}
                for key, default in (("query_database_status", "uncertain"), ("exhaustion_reason", ""), ("recommended_query_change", ""), ("queries_attempted", [])):
                    goal_evaluation.setdefault(key, execution_report.get(key, default))
                return {
                    "round": round_number,
                    "plan_step_id": str((plan_step or {}).get("step_id") or ""),
                    "step_goal": str((plan_step or {}).get("goal") or ""),
                    "main_question": main_question,
                    "retrieval_actions": [{"query": query, "query_type": route_result.get("query_type"), "selected_tools": list(route_result.get("selected_tools") or []), "constraints": dict(route_result.get("constraints") or {}), "rerank_goal": route_result.get("rerank_goal"), "expansion": (route_result.get("planner_metadata") or {}).get("expansion") or {}}],
                    "retrieved_count": len(by_id),
                    "accepted_evidence": accepted,
                    "rejected_evidence": list(parsed.get("rejected_evidence") or []),
                    "question_information_gain": q_gain,
                    "goal_evaluation": goal_evaluation,
                    "micro_retrieval_ledger": micro_ledger,
                    "step_execution_report": dict(route_result.get("step_execution_report") or {}),
                    "tool_trace": list(route_result.get("tool_trace") or []),
                    "memory_mode": "llm",
                    "memory_type": "step_memory",
                }
            except Exception as exc:
                logger.warning("LLM round memory failed, falling back to rules: %s", exc)
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        execution_report = dict(route_result.get("step_execution_report") or {})
        report_accepted_ids = {
            str(value) for value in execution_report.get("accepted_evidence_ids") or []
            if str(value)
        }
        groups = (
            ("support_candidate", evidence_review.get("supporting_evidence") or []),
            ("counter", evidence_review.get("contradicting_evidence") or []),
            ("risk", evidence_review.get("safety_risks") or []),
            ("insufficient", evidence_review.get("insufficient_evidence") or []),
        )
        accepted_ids: Set[str] = set()
        rejected_ids: Set[str] = set()
        for role, rows in groups:
            for source in rows:
                item = dict(source)
                reviewed_role = str(item.get("evidence_role") or "")
                item_role = reviewed_role or role
                if item_role == "support_candidate":
                    # Rule mode may preserve direct evidence when upstream data
                    # explicitly establishes exact entity match and no conflict.
                    item_role = (
                        "direct_support"
                        if item.get("target_entity_match") == "exact" and not (item.get("mismatched_dimensions") or [])
                        else "partial_support"
                    )
                chunk_id = str(item.get("chunk_id") or "")
                relevance = float(item.get("llm_relevance_score", 1.0 if item_role != "insufficient" else 0.0) or 0.0)
                report_rejected = bool(report_accepted_ids) and chunk_id not in report_accepted_ids
                # A structured EvidenceReview role is authoritative. Low
                # relevance is often expected for a clinically useful analog
                # or safety warning and must not erase its bounded scope.
                legacy_below_threshold = not reviewed_role and relevance < self.min_relevance
                if not chunk_id or item_role == "insufficient" or legacy_below_threshold or report_rejected:
                    if chunk_id not in rejected_ids and chunk_id not in accepted_ids:
                        rejected.append({"chunk_id": chunk_id, "reason": "相关性或信息增益不足"})
                        rejected_ids.add(chunk_id)
                    continue
                if chunk_id in accepted_ids:
                    continue
                accepted_ids.add(chunk_id)
                item["evidence_role"] = item_role
                item["source_round"] = round_number
                accepted.append(item)

        new_ids = [item["chunk_id"] for item in accepted if item["chunk_id"] not in seen_chunk_ids]
        support_count = sum(
            item["evidence_role"] in {"direct_support", "partial_support", "analog_support"}
            or any(str(finding.get("relation_to_target") or "").lower() == "supports" for finding in item.get("dimension_findings") or [] if isinstance(finding, dict))
            for item in accepted
        )
        counter_count = sum(
            item["evidence_role"] == "counter"
            or any(str(finding.get("relation_to_target") or "").lower() == "contradicts" for finding in item.get("dimension_findings") or [] if isinstance(finding, dict))
            for item in accepted
        )
        risk_count = sum(item["evidence_role"] == "risk" for item in accepted)
        execution_goal = dict(execution_report.get("goal_evaluation") or {})
        missing: List[str] = [
            str(value) for value in execution_goal.get("observed_gaps") or []
            if str(value)
        ]
        if not support_count:
            missing.append("supporting_case_evidence")
        if not counter_count and not execution_goal.get("success_criteria_met"):
            missing.append("counter_evidence_for_balance")
        # Safety is an independent planning lane. Do not inject a safety/DDI
        # gap into a direct outcome, analog-case or mechanism step merely
        # because the overall user query is treatment advice.
        evidence_lane = str((plan_step or {}).get("evidence_lane") or "")
        if evidence_lane == "ddi_safety" and not risk_count:
            missing.append("safety_or_ddi_risk_evidence")
        gain_score = min(1.0, (len(new_ids) + counter_count + risk_count) / max(1, len(accepted) + 1))
        success_criteria_met = bool(execution_goal.get("success_criteria_met")) and bool(accepted)
        completion_status = (
            "sufficiently_met"
            if success_criteria_met and str(execution_report.get("execution_status") or "") == "success"
            else "minimally_met" if accepted else "not_met"
        )
        result = {
            "round": round_number,
            "plan_step_id": str((plan_step or {}).get("step_id") or ""),
            "step_goal": str((plan_step or {}).get("goal") or ""),
            "main_question": main_question,
            "retrieval_actions": [{
                "query": query,
                "query_type": route_result.get("query_type"),
                "selected_tools": list(route_result.get("selected_tools") or []),
                "constraints": dict(route_result.get("constraints") or {}),
                "rerank_goal": route_result.get("rerank_goal"),
                "expansion": (route_result.get("planner_metadata") or {}).get("expansion") or {},
            }],
            "retrieved_count": len((route_result.get("evidence_layering") or {}).get("assessed_evidence") or []),
            "accepted_evidence": accepted,
            "rejected_evidence": rejected,
            "question_information_gain": {
                "score": round(gain_score, 3),
                "new_chunk_ids": new_ids,
                "support_count": support_count,
                "counter_count": counter_count,
                "risk_count": risk_count,
            },
            "goal_evaluation": {
                "matched_goal_count": int(execution_goal.get("matched_goal_count") or len(accepted)),
                "best_goal_relevance": max(
                    float(execution_goal.get("best_goal_relevance") or 0.0),
                    max([float(item.get("llm_relevance_score") or 0.0) for item in accepted] or [0.0]),
                ),
                "success_criteria_met": success_criteria_met,
                "completion_status": completion_status,
                "critical_gaps": _unique_list(missing),
                "optional_gaps": [],
                "marginal_value_of_more_search": 0.5 if missing else 0.1,
                "recommended_stop": completion_status == "sufficiently_met",
                "query_database_status": str(execution_report.get("query_database_status") or "uncertain"),
                "exhaustion_reason": str(execution_report.get("exhaustion_reason") or ""),
                "recommended_query_change": str(execution_report.get("recommended_query_change") or ""),
                "queries_attempted": list(execution_report.get("queries_attempted") or []),
                "observed_gaps": _unique_list(missing),
                "observed_failures": list(execution_goal.get("observed_failures") or (["no_new_evidence"] if not new_ids else [])),
            },
        }
        result["step_execution_report"] = dict(route_result.get("step_execution_report") or {})
        result["tool_trace"] = list(route_result.get("tool_trace") or [])
        result["micro_retrieval_ledger"] = self.build_micro_ledger(route_result)
        result["retrieval_funnel"] = {
            "retrieved_count": result["retrieved_count"],
            "reranked_count": len(((route_result.get("results") or {}).get("rerank_candidates") or [])),
            "fetched_count": len(((route_result.get("results") or {}).get("fetch_evidence") or [])),
            "reviewed_count": sum(len(evidence_review.get(key) or []) for key in ("supporting_evidence", "contradicting_evidence", "insufficient_evidence")),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
        }
        ledger = result["micro_retrieval_ledger"]
        tool_errors = [row for row in ledger if row.get("error")]
        hit_count = sum(int(row.get("hit_count") or 0) for row in ledger)
        funnel = result["retrieval_funnel"]
        if ledger and len(tool_errors) == len(ledger):
            attempt_status = "tool_error"
        elif hit_count == 0:
            attempt_status = "no_hits"
        elif funnel["fetched_count"] == 0:
            attempt_status = "fetch_failed"
        elif funnel["accepted_count"] == 0 and funnel["reviewed_count"] > 0:
            attempt_status = "review_rejected"
        elif funnel["accepted_count"] > 0:
            attempt_status = "useful_evidence"
        else:
            attempt_status = "hits_irrelevant"
        legacy_query_status = str(execution_report.get("query_database_status") or "uncertain")
        result["retrieval_state"] = {
            "search_attempt_status": attempt_status,
            "query_direction_status": (
                "locally_exhausted" if legacy_query_status == "exhausted"
                else "untried_alternatives" if attempt_status in {"no_hits", "hits_irrelevant", "review_rejected"}
                else "uncertain"
            ),
            # One round can prove presence, never corpus-wide absence.
            "database_coverage_status": "evidence_found" if accepted else "unknown",
            "tools_attempted": list(dict.fromkeys(str(row.get("tool") or "") for row in ledger if row.get("tool"))),
            "distinct_queries_attempted": list(dict.fromkeys(str(row.get("query") or "") for row in ledger if row.get("query"))),
            "tool_error_count": len(tool_errors),
        }
        result["goal_evaluation"].update(result["retrieval_state"])
        rejection_reasons: Dict[str, int] = {}
        for item in [*(evidence_review.get("insufficient_evidence") or []), *[row for row in accepted if row.get("mismatched_dimensions")]]:
            for reason in item.get("mismatched_dimensions") or item.get("unreported_dimensions") or ["insufficient_relevance"]:
                key = str(reason).strip()[:120]
                if key:
                    rejection_reasons[key] = rejection_reasons.get(key, 0) + 1
        result["rejection_summary"] = rejection_reasons
        result["memory_mode"] = "rules"
        result["memory_type"] = "step_memory"
        return result


# Read-only compatibility for existing integrations and old session tests.
RoundMemoryAgent = StepMemoryAgent


class AnswerMemoryAgent:
    """Accumulate claim-oriented answer memory across informative rounds."""

    SYSTEM_PROMPT = """你是 AnswerMemoryAgent，负责维护跨检索轮次的论断索引，不负责重新阅读证据、重新做医学归因或从弱证据推导新结论。EvidenceReview 已经完成单条证据的医学事实提取和支持边界判断；StepMemory 已登记本轮采用的审核结果。

每条 claim 在内部必须满足以下“软 Schema”。它是粒度检查表，不要求你在函数参数中新增字段：
{
  "population": "证据实际研究的人群；未知则留空，不得补成目标患者",
  "intervention": "证据实际使用的药物或方案",
  "comparator": "证据明确报告的对照；未报告则留空",
  "outcome": "一个且仅一个结局维度",
  "direction": "该结局的原始方向",
  "timepoint": "证据明确报告的时间点；未报告则留空",
  "setting": "治疗线次、疾病部位或关键分子背景",
  "limitations": "不匹配和未报告的边界"
}
原子性要求：一条 claim 只能有一组 population × intervention × outcome × direction × timepoint。任一字段不同就输出两条 claim；只有软 Schema 各字段一致、仅措辞重复时，才允许合并 chunk_id。不得用分号把两条不同 claim_scope 串成一条 claim。

你的职责仅包括：
1. 保存仍与主问题有关的已审核 claim_scope 及真实 chunk_id；
2. 合并真正重复的论断，但不得把不同证据的药物、结局、严重程度、时间或处置拼成一个新论断；
3. 将新证据追加到已有论断，或在出现直接反证时把状态更新为 contested；
4. 维护 direct、partial、analog 和 counter 的证据绑定、来源轮次与置信度；
5. 删除已被更强证据明确取代的重复记忆。

不要承担这些任务：重新解释证据原文、改善 claim_scope 的医学表述、把类比证据改写成目标患者结论、补充指南知识、预测最终答案。对于 partial 或 analog 证据，使用 EvidenceReview 已给出的 claim_scope 作为语义底稿；可以在不改变主体、结局、方向、严重程度、时间和处置的前提下做简短压缩，但不能扩大含义。supporting_chunk_ids 仅是兼容字段，必须等于 direct_support_chunk_ids。必须通过 commit_answer_memory 提交，不生成面向用户的最终回答。"""

    def __init__(self, llm_client: Optional["LLMClient"] = None, use_llm: bool = True) -> None:
        self.llm_client = llm_client
        self.use_llm = use_llm

    def update(self, memory: Dict[str, Any], round_memory: Dict[str, Any]) -> Dict[str, Any]:
        if self.use_llm and self.llm_client is not None and round_memory.get("accepted_evidence"):
            try:
                compact_existing = {
                    "claims": memory.get("claims") or [],
                    "informative_rounds": memory.get("informative_rounds") or [],
                    "evidence_chunk_ids": list((memory.get("evidence_by_id") or {}).keys())[-50:],
                }
                compact_round = dict(round_memory)
                compact_round["accepted_evidence"] = _compact_evidence_rows(
                    round_memory.get("accepted_evidence") or [], text_limit=500
                )
                compact_round.pop("tool_trace", None)
                compact_round.pop("rejected_evidence", None)
                compact_round["retrieval_actions"] = [{
                    "query": action.get("query"),
                    "selected_tools": action.get("selected_tools") or [],
                    "rerank_goal": action.get("rerank_goal"),
                } for action in (round_memory.get("retrieval_actions") or [])[:1]]
                payload = {"existing_answer_memory": compact_existing, "new_round_memory": compact_round}
                prompt = (
                    f"输入：\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
                    "输出 JSON：{\"claims\":[{\"claim_id\":\"claim_01\",\"claim\":\"具体临床论断\","
                    "\"status\":\"provisional|supported|contested\",\"supporting_chunk_ids\":[],"
                    "\"contradicting_chunk_ids\":[],\"direct_support_chunk_ids\":[],\"partial_support_chunk_ids\":[],\"analog_support_chunk_ids\":[],"
                    "\"subject_entity\":\"论断主体药物或方案\",\"claim_scope\":\"证据支持的最窄范围\",\"support_level\":\"direct|partial|analog|unsupported\",\"source_rounds\":[],\"confidence\":0.0,"
                    "\"safety_status\":\"pending\"}],\"informative_rounds\":[]}。"
                    "必须保留仍有效的旧 claim；不得引用 evidence_by_id 中不存在的 chunk_id；confidence 范围0-1。"
                    "EvidenceReview的claim_scope是新论断的事实来源。先按软Schema逐字段比较，再决定是否合并。"
                    "跨证据只可去重和追加引用；population、intervention、outcome、direction、timepoint任一不同必须拆成不同claim。"
                )
                parsed = _call_structured_llm(
                    self.llm_client,
                    system=self.SYSTEM_PROMPT,
                    user=prompt,
                    function_name="commit_answer_memory",
                    description="Commit the updated cross-round answer claims and their evidence bindings.",
                    parameters=ANSWER_MEMORY_SCHEMA,
                    temperature=0.2,
                    max_output_tokens=8000,
                )
                if not parsed or not isinstance(parsed.get("claims"), list):
                    raise ValueError("invalid answer memory JSON")
                evidence_by_id = dict(memory.get("evidence_by_id") or {})
                for item in round_memory.get("accepted_evidence") or []:
                    if item.get("chunk_id"):
                        evidence_by_id[str(item["chunk_id"])] = dict(item)
                valid = set(evidence_by_id)
                claims = []
                for index, claim in enumerate(parsed["claims"], start=1):
                    if not isinstance(claim, dict) or not str(claim.get("claim") or "").strip():
                        continue
                    row = dict(claim)
                    row["claim_id"] = str(row.get("claim_id") or f"claim_{index:02d}")
                    row["supporting_chunk_ids"] = [str(x) for x in row.get("supporting_chunk_ids") or [] if str(x) in valid]
                    row["contradicting_chunk_ids"] = [str(x) for x in row.get("contradicting_chunk_ids") or [] if str(x) in valid]
                    for key in ("direct_support_chunk_ids", "partial_support_chunk_ids", "analog_support_chunk_ids"):
                        row[key] = [str(x) for x in row.get(key) or [] if str(x) in valid]
                    # Enforce the evidence review classification even if the
                    # memory model attempts to promote a weaker item.
                    role_by_id = {
                        evidence_id: str((evidence_by_id.get(evidence_id) or {}).get("evidence_role") or "")
                        for evidence_id in valid
                    }
                    row["direct_support_chunk_ids"] = [x for x in row["direct_support_chunk_ids"] if role_by_id.get(x) == "direct_support"]
                    row["partial_support_chunk_ids"] = [x for x in row["partial_support_chunk_ids"] if role_by_id.get(x) == "partial_support"]
                    row["analog_support_chunk_ids"] = [x for x in row["analog_support_chunk_ids"] if role_by_id.get(x) == "analog_support"]
                    row["supporting_chunk_ids"] = list(row["direct_support_chunk_ids"])
                    if row["direct_support_chunk_ids"]:
                        row["support_level"] = "direct"
                    elif row["partial_support_chunk_ids"]:
                        row["support_level"] = "partial"
                    elif row["analog_support_chunk_ids"]:
                        row["support_level"] = "analog"
                    else:
                        row["support_level"] = "unsupported"
                    if row["support_level"] in {"partial", "analog"}:
                        weak_ids = (
                            row["partial_support_chunk_ids"]
                            if row["support_level"] == "partial"
                            else row["analog_support_chunk_ids"]
                        )
                        weak_rows = [evidence_by_id[evidence_id] for evidence_id in weak_ids if evidence_id in evidence_by_id]
                        immutable_scopes = _unique_list([
                            str(item.get("claim_scope") or "").strip()
                            for item in weak_rows
                            if str(item.get("claim_scope") or "").strip()
                        ])
                        # A weak evidence ID without an Evidence Review scope
                        # cannot authorize any generated clinical proposition.
                        if not immutable_scopes:
                            row["partial_support_chunk_ids"] = []
                            row["analog_support_chunk_ids"] = []
                            row["support_level"] = "unsupported"
                        elif len(immutable_scopes) > 1:
                            # A merged weak claim would erase population/drug/
                            # outcome bindings. Retry through the atomic rule
                            # path instead of manufacturing a semicolon claim.
                            raise ValueError("answer memory merged distinct weak evidence scopes")
                        else:
                            row["claim"] = immutable_scopes[0]
                            row["claim_scope"] = row["claim"]
                            row["evidence_scopes"] = immutable_scopes
                            row["supports_dimensions"] = _unique_list([
                                str(value)
                                for item in weak_rows
                                for value in (item.get("supports_dimensions") or [])
                                if str(value)
                            ])
                            row["mismatched_dimensions"] = _unique_list([
                                str(value)
                                for item in weak_rows
                                for value in (item.get("mismatched_dimensions") or [])
                                if str(value)
                            ])
                            row["unreported_dimensions"] = _unique_list([
                                str(value)
                                for item in weak_rows
                                for value in (item.get("unreported_dimensions") or [])
                                if str(value)
                            ])
                            row["unsupported_dimensions"] = _unique_list([
                                *row["mismatched_dimensions"],
                                *row["unreported_dimensions"],
                            ])
                    row["status"] = "supported" if row["support_level"] == "direct" and not row["contradicting_chunk_ids"] else "contested" if row["contradicting_chunk_ids"] else "provisional"
                    row["source_rounds"] = _normalize_round_numbers(row.get("source_rounds"))
                    row["confidence"] = _clamp_score(row.get("confidence"))
                    row.setdefault("safety_status", "pending")
                    claims.append(row)
                return {"claims": claims, "evidence_by_id": evidence_by_id, "informative_rounds": _normalize_round_numbers(parsed.get("informative_rounds")), "memory_mode": "llm"}
            except Exception as exc:
                logger.warning("LLM answer memory failed, falling back to rules: %s", exc)
        evidence_by_id = dict(memory.get("evidence_by_id") or {})
        claims_by_key = {
            str(item.get("claim_key") or f"scope::{str(item.get('claim_scope') or item.get('claim') or '').strip().lower()}"): dict(item)
            for item in (memory.get("claims") or [])
            if isinstance(item, dict)
        }
        round_number = int(round_memory.get("round") or 0)
        for item in round_memory.get("accepted_evidence") or []:
            chunk_id = str(item.get("chunk_id") or "")
            if not chunk_id:
                continue
            evidence_by_id[chunk_id] = dict(item)
            role = str(item.get("evidence_role") or "partial_support")
            scope = str(item.get("claim_scope") or "").strip()
            subject = str(item.get("subject_entity") or "").strip().lower()
            # The reviewed scope is the semantic atom. Different scopes never
            # share an AnswerMemory bucket, even when both point in the same
            # broad answer direction.
            key = (
                f"clinical_risk::{subject}::{scope.lower()}"
                if role == "risk"
                else f"answer_atom::{subject}::{scope.lower() or chunk_id}"
            )
            claim = claims_by_key.setdefault(key, {
                "claim_id": f"claim_{len(claims_by_key) + 1:02d}",
                "claim_key": key,
                "claim": scope or ("存在需要在最终回答中综合说明的临床风险" if role == "risk" else "当前检索证据对主问题形成有限支持并需结合反证解释"),
                "status": "provisional",
                "supporting_chunk_ids": [],
                "contradicting_chunk_ids": [],
                "direct_support_chunk_ids": [],
                "partial_support_chunk_ids": [],
                "analog_support_chunk_ids": [],
                "subject_entity": "",
                "claim_scope": str(item.get("claim_scope") or ""),
                "support_level": "unsupported",
                "source_rounds": [],
                "confidence": 0.0,
                "safety_status": "pending",
            })
            target = {
                "direct_support": "direct_support_chunk_ids",
                "partial_support": "partial_support_chunk_ids",
                "analog_support": "analog_support_chunk_ids",
            }.get(role, "contradicting_chunk_ids" if role == "counter" else "partial_support_chunk_ids")
            claim[target] = _unique_list(list(claim[target]) + [chunk_id])
            if role in {"partial_support", "analog_support"}:
                if scope:
                    scopes = _unique_list([*(claim.get("evidence_scopes") or []), scope])
                    claim["evidence_scopes"] = scopes
                    claim["claim"] = scopes[0]
                    claim["claim_scope"] = claim["claim"]
                claim["supports_dimensions"] = _unique_list([
                    *(claim.get("supports_dimensions") or []),
                    *[str(value) for value in (item.get("supports_dimensions") or []) if str(value)],
                ])
                claim["mismatched_dimensions"] = _unique_list([
                    *(claim.get("mismatched_dimensions") or []),
                    *[str(value) for value in (item.get("mismatched_dimensions") or []) if str(value)],
                ])
                claim["unreported_dimensions"] = _unique_list([
                    *(claim.get("unreported_dimensions") or []),
                    *[str(value) for value in (item.get("unreported_dimensions") or []) if str(value)],
                ])
                claim["unsupported_dimensions"] = _unique_list([
                    *(claim.get("mismatched_dimensions") or []),
                    *(claim.get("unreported_dimensions") or []),
                ])
            claim["source_rounds"] = sorted(set(list(claim["source_rounds"]) + [round_number]))
        for claim in claims_by_key.values():
            claim["supporting_chunk_ids"] = list(claim["direct_support_chunk_ids"])
            support = len(claim["direct_support_chunk_ids"])
            counter = len(claim["contradicting_chunk_ids"])
            claim["confidence"] = round(support / max(1, support + counter), 3)
            claim["support_level"] = "direct" if support else "partial" if claim["partial_support_chunk_ids"] else "analog" if claim["analog_support_chunk_ids"] else "unsupported"
            if claim["support_level"] in {"partial", "analog"} and not claim.get("evidence_scopes"):
                claim["partial_support_chunk_ids"] = []
                claim["analog_support_chunk_ids"] = []
                claim["support_level"] = "unsupported"
            claim["status"] = "supported" if support and not counter else "provisional"
            if counter:
                claim["status"] = "contested"
        informative_rounds = list(memory.get("informative_rounds") or [])
        if (round_memory.get("question_information_gain") or {}).get("new_chunk_ids"):
            informative_rounds.append(round_number)
        return {
            "claims": list(claims_by_key.values()),
            "evidence_by_id": evidence_by_id,
            "informative_rounds": sorted(set(informative_rounds)),
            "memory_mode": "rules",
        }

    def apply_safety_review(self, memory: Dict[str, Any], safety_result: Dict[str, Any]) -> Dict[str, Any]:
        reviews = {
            str(item.get("claim_id")): item
            for item in (safety_result.get("claim_reviews") or [])
            if isinstance(item, dict) and item.get("claim_id")
        }
        critical_refs: Set[str] = set()
        for issue in safety_result.get("issues") or []:
            if issue.get("severity") in {"high", "critical"}:
                critical_refs.update(str(ref) for ref in (issue.get("evidence_refs") or []))
        for claim in memory.get("claims") or []:
            review = reviews.get(str(claim.get("claim_id")))
            if review:
                decision = str(review.get("decision") or "revise")
                claim["safety_status"] = decision
                claim["safety_violations"] = list(review.get("violations") or [])
                if review.get("required_revision"):
                    claim["required_revision"] = str(review["required_revision"])
                if review.get("safe_claim"):
                    claim["safe_claim"] = str(review["safe_claim"])
                if decision == "reject":
                    claim["status"] = "rejected_by_safety_gate"
                elif decision == "revise":
                    claim["status"] = "safety_limited"
                continue
            refs = set(claim.get("direct_support_chunk_ids") or claim.get("supporting_chunk_ids") or []) | set(claim.get("contradicting_chunk_ids") or [])
            if refs & critical_refs or safety_result.get("requires_human_review"):
                claim["safety_status"] = "revise"
                claim["status"] = "safety_limited"
                claim["required_revision"] = "降低结论确定性，明确风险、证据边界并要求临床复核"
            else:
                claim["safety_status"] = "approved_with_boundary"
        memory["safety_review"] = safety_result
        return memory


class MainAgentLoop:
    def __init__(
        self,
        router: RetrievalRouter,
        citation_agent: Optional[EvidenceReviewAgent] = None,
        evidence_review_agent: Optional[EvidenceReviewAgent] = None,
        round_memory_agent: Optional[StepMemoryAgent] = None,
        step_memory_agent: Optional[StepMemoryAgent] = None,
        replanner_memory_agent: Optional[ReplannerMemoryAgent] = None,
        answer_memory_agent: Optional[AnswerMemoryAgent] = None,
        planning_agent: Optional[MultiStepPlanningAgent] = None,
        execution_agent: Optional[RetrievalExecutionAgent] = None,
        max_steps: int = 3,
        max_total_steps: int = 128,
        max_budget_extension: int = 32,
        max_attempts_per_plan_step: int = 128,
        final_llm_call_reserve: int = 0,
        max_llm_calls_per_round: int = 0,
        top_k: int = 10,
        external_knowledge_enabled: bool = True,
    ) -> None:
        self.router = router
        # citation_agent remains accepted as a compatibility argument; its old
        # retrieval-review responsibility now belongs to EvidenceReviewAgent.
        self.evidence_review_agent = evidence_review_agent or citation_agent or EvidenceReviewAgent()
        shared_llm = getattr(self.evidence_review_agent, "llm_client", None)
        # StepMemory is a deterministic assembly of the execution report,
        # evidence review and retrieval trace. Those inputs already contain the
        # medical judgments, so another summarization call only adds latency.
        self.step_memory_agent = step_memory_agent or round_memory_agent or StepMemoryAgent(
            llm_client=None,
            use_llm=False,
        )
        self.round_memory_agent = self.step_memory_agent
        memory_llm = shared_llm or getattr(router, "llm_client", None)
        self.replanner_memory_agent = replanner_memory_agent or ReplannerMemoryAgent(llm_client=memory_llm, use_llm=memory_llm is not None)
        self.answer_memory_agent = answer_memory_agent or AnswerMemoryAgent(llm_client=shared_llm, use_llm=shared_llm is not None)
        planner_llm = getattr(router, "llm_client", None)
        self.planning_agent = planning_agent or MultiStepPlanningAgent(
            llm_client=planner_llm,
            use_llm=planner_llm is not None and getattr(router, "use_llm", False),
            external_knowledge_enabled=external_knowledge_enabled,
        )
        self.execution_agent = execution_agent
        if self.execution_agent is None and planner_llm is not None:
            self.execution_agent = RetrievalExecutionAgent(
                router=router,
                llm_client=planner_llm,
                external_knowledge_enabled=external_knowledge_enabled,
            )
        self.external_knowledge_enabled = bool(external_knowledge_enabled)
        self.max_steps = max(1, max_steps)
        self.max_total_steps = max(self.max_steps, max_total_steps)
        self.max_budget_extension = max(0, max_budget_extension)
        self.max_attempts_per_plan_step = max(1, max_attempts_per_plan_step)
        self.final_llm_call_reserve = max(0, final_llm_call_reserve)
        self.max_llm_calls_per_round = max(0, max_llm_calls_per_round)
        self.top_k = max(1, top_k)

    def _extract_entities(self, constraints: Dict[str, Any]) -> Dict[str, List[str]]:
        entities: Dict[str, List[str]] = {}
        for key, value in constraints.items():
            if isinstance(value, list):
                entities[key] = _unique_list([str(item) for item in value])
            elif value:
                entities[key] = [str(value)]
        return entities

    def _qwen_overflow_rescue(
        self,
        route_result: Dict[str, Any],
        initial_review: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Inspect the next Qwen rank window only when the first window is empty."""
        verdict = str(initial_review.get("verdict") or "")
        supporting = initial_review.get("supporting_evidence") or []
        overflow = list(route_result.get("qwen_overflow_hits") or [])[:8]
        if supporting or verdict not in {"insufficient_evidence", "not_supported"} or not overflow:
            return initial_review
        existing = list((route_result.get("results") or {}).get("fetch_evidence") or [])
        existing_ids = {str(row.get("chunk_id") or "") for row in existing}
        ids = [str(hit.get("id") or "") for hit in overflow if str(hit.get("id") or "") not in existing_ids]
        if not ids:
            return initial_review
        fetched = self.router.tools.fetch_evidence(chunk_ids=ids, limit=len(ids))
        if not fetched:
            return initial_review
        route_result.setdefault("results", {}).setdefault("fetch_evidence", []).extend(fetched)
        route_result["qwen_rescue"] = {
            "used": True,
            "source_rank_window": "9-16",
            "candidate_count": len(ids),
            "fetched_count": len(fetched),
        }
        evidence = self.router._assess_evidence(route_result["results"]["fetch_evidence"])
        route_result["evidence_layering"] = {
            "query": route_result.get("rerank_goal") or "",
            "layer_distribution": evidence["layer_distribution"],
            "assessed_evidence": evidence["assessed_items"],
        }
        route_result["contradiction_check"] = self.router._contradiction_check(
            [], [], evidence,
        )
        return self.evidence_review_agent.review(route_result=route_result)

    @staticmethod
    def _expand_allowed_strategies(
        *,
        action: str,
        query_type: str,
        evidence_lane: str,
        requested_strategy: str,
    ) -> List[str]:
        if action != "expand_current_step":
            return []
        strategy = str(requested_strategy or "").strip()
        if not strategy or strategy == "none":
            return []
        if evidence_lane == "ddi_safety":
            ordered = ["drug_alias", "case_ddi", "pk_relation", "ddi_rule"]
            if strategy == "mutation_drug":
                ordered.insert(1, "mutation_drug")
            return ordered
        return [strategy]

    def _build_missing_information(
        self,
        query_type: Optional[str],
        citation_review: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> List[str]:
        missing: List[str] = []
        support_count = len(citation_review.get("supporting_evidence", []))
        counter_count = len(citation_review.get("contradicting_evidence", []))
        risk_count = len(citation_review.get("safety_risks", []))

        if support_count == 0:
            missing.append("supporting_case_evidence")
        if counter_count == 0:
            missing.append("counter_evidence_for_balance")
        ddi_is_explicit = bool(constraints.get("ddi_terms")) or bool(re.search(r"(?:DDI|drug interaction|相互作用|CYP|P-gp|BCRP)", " ".join(str(x) for x in constraints.get("ddi_terms") or []), re.IGNORECASE))
        if ddi_is_explicit and risk_count == 0:
            missing.append("safety_or_ddi_risk_evidence")

        # Missing entity fields are not automatically retrieval gaps. Planner
        # decides what matters from the open information_needs representation.

        return _unique_list(missing)

    def _collect_evidence_rows(self, citation_review: Dict[str, Any], limit: int = 12) -> List[RetrievalEvidence]:
        all_items: List[Dict[str, Any]] = []
        all_items.extend(citation_review.get("supporting_evidence", []))
        all_items.extend(citation_review.get("contradicting_evidence", []))
        all_items.extend(citation_review.get("insufficient_evidence", []))

        dedup: Dict[str, RetrievalEvidence] = {}
        for item in all_items:
            chunk_id = str(item.get("chunk_id") or "").strip()
            if not chunk_id:
                continue
            if chunk_id in dedup:
                continue
            dedup[chunk_id] = RetrievalEvidence(
                chunk_id=chunk_id,
                evidence_level=str(item.get("evidence_level") or "case_report_evidence"),
                text=str(item.get("text") or ""),
                pmid=item.get("pmid"),
                title=item.get("title"),
                relevance_signals=dict(item.get("relevance_signals") or {}),
            )
            if len(dedup) >= limit:
                break
        return list(dedup.values())

    @staticmethod
    def _direct_step_decision(
        *,
        active_step: Dict[str, Any],
        route_plan: RoutePlan,
        first_execution: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        execution = dict(first_execution or {})
        return {
            "action": "retry_current_step",
            "active_step_id": str(active_step.get("step_id") or ""),
            "search_query": str(execution.get("search_query") or active_step.get("goal") or route_plan.search_query),
            "rerank_goal": str(execution.get("rerank_goal") or active_step.get("rerank_goal") or active_step.get("goal") or ""),
            "selected_tools": list(execution.get("selected_tools") or route_plan.selected_tools)[:2],
            "constraints": {},
            "expansion": {"strategy": "none", "source": "", "added_terms": [], "relation_type": ""},
            "plan_changes": [],
            "avoid_repeating": [],
            "decision_rationale": "Execute the untried plan step directly; no Replanner decision is needed.",
            "requested_budget_extension": 0,
            "budget_extension_reason": "",
            "decision_mode": "direct_plan_execution",
        }

    def run(self, query: str, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        initial_context = initial_context or {}
        initial_llm_enabled = bool(
            getattr(self.router, "use_llm", False)
            and getattr(self.router, "llm_client", None) is not None
        )
        if initial_llm_enabled:
            understanding = self.router.understand_query(query)
            retrieval_adapter = self.router.adapt_problem_representation(understanding)
            proposed_constraints = dict(retrieval_adapter.get("hard_constraints") or {})
            route_plan = RoutePlan(
                query_type=None,
                # Execution Agent performs tool selection from its registered
                # retrieval functions; this list is only a capability hint.
                selected_tools=["structured_search", "dense_search", "bm25_search", "hybrid_search", "fetch_evidence"],
                constraints=proposed_constraints,
                search_query=query,
                planner_metadata={
                    "mode": "separate_query_understanding_then_planning",
                },
                constraint_layers=understanding,
            )
            retrieval_plan = self.planning_agent.create_plan(
                main_question=query,
                query_type=route_plan.query_type,
                constraints=route_plan.constraints,
                context=initial_context,
                initial_total_budget=self.max_steps,
                hard_total_budget=self.max_total_steps,
                problem_representation=understanding,
            )
            first_execution: Dict[str, Any] = {}
        else:
            route_plan = self.router.plan(query, planner_context=initial_context, main_question=query)
            retrieval_plan = self.planning_agent.create_plan(
                main_question=query,
                query_type=route_plan.query_type,
                constraints=route_plan.constraints,
                context=initial_context,
                initial_total_budget=self.max_steps,
                hard_total_budget=self.max_total_steps,
                problem_representation=dict(route_plan.constraint_layers or {}),
            )
            first_execution = {}
        direct_untried_step_execution = initial_llm_enabled
        prior_constraints = dict(initial_context.get("confirmed_constraints") or {})
        prior_constraints.update(route_plan.constraints)
        try:
            requested_initial_budget = int(retrieval_plan.get("initial_total_budget") or self.max_steps)
        except (TypeError, ValueError):
            requested_initial_budget = self.max_steps
        total_budget = max(1, min(self.max_total_steps, requested_initial_budget))
        retrieval_plan["initial_total_budget"] = total_budget
        active_step_index = 0
        state = QueryState(
            original_query=query,
            query_type=route_plan.query_type,
            confirmed_constraints=prior_constraints,
            identified_entities=self._extract_entities(prior_constraints),
            search_hints=dict(route_plan.constraint_layers or {}),
        )
        steps: List[AgentLoopStep] = []
        seen_chunk_ids: Set[str] = set()
        last_route_result: Dict[str, Any] = {}
        replan_decisions: List[Dict[str, Any]] = []
        budget_events: List[Dict[str, Any]] = []
        step_budget_status: Dict[str, Dict[str, Any]] = {}
        round_memories: List[Dict[str, Any]] = [dict(x) for x in (initial_context.get("step_memories") or initial_context.get("round_memories") or []) if isinstance(x, dict)]
        replanner_short_memory: Dict[str, Any] = dict(initial_context.get("replanner_short_memory") or {})
        replanner_long_memory: Dict[str, Any] = dict(initial_context.get("replanner_long_memory") or {})
        pending_replanner_memories: List[Dict[str, Any]] = [dict(x) for x in (initial_context.get("pending_replanner_step_memories") or []) if isinstance(x, dict)]
        if round_memories and not replanner_short_memory and not replanner_long_memory and not pending_replanner_memories:
            pending_replanner_memories = list(round_memories)
        replanner_memory_events: List[Dict[str, Any]] = [dict(x) for x in (initial_context.get("replanner_memory_events") or []) if isinstance(x, dict)]
        answer_memory: Dict[str, Any] = dict(initial_context.get("answer_memory") or {})
        answer_memory.setdefault("claims", [])
        answer_memory.setdefault("evidence_by_id", {})
        structured_output_recoveries: List[Dict[str, Any]] = []
        answer_memory.setdefault("informative_rounds", [])
        seen_chunk_ids.update(str(x) for x in answer_memory["evidence_by_id"])
        prior_round_count = max(
            [int(item.get("round") or 0) for item in round_memories] or [0]
        )

        rounds_used = 0
        while rounds_used < self.max_total_steps:
            calls_remaining = getattr(getattr(self.router, "llm_client", None), "calls_remaining", None)
            remaining = calls_remaining() if callable(calls_remaining) else None
            required_for_next_round = self.final_llm_call_reserve + self.max_llm_calls_per_round
            if (
                remaining is not None
                and self.max_llm_calls_per_round
                and remaining < required_for_next_round
            ):
                state.stop_condition = "llm_budget_reserved_for_final_generation"
                budget_events.append({
                    "event": "llm_budget_reserved_for_final_generation",
                    "rounds_used": rounds_used,
                    "calls_remaining": remaining,
                    "reserved_final_calls": self.final_llm_call_reserve,
                    "estimated_next_round_calls": self.max_llm_calls_per_round,
                })
                break
            plan_steps = retrieval_plan.get("steps") or []
            if active_step_index >= len(plan_steps):
                state.stop_condition = "retrieval_plan_completed"
                break
            active_before_replan = dict(plan_steps[active_step_index])
            active_step_id = str(active_before_replan.get("step_id") or f"S{active_step_index + 1}")
            active_memories = [memory for memory in round_memories if memory.get("plan_step_id") == active_step_id]
            if active_memories:
                latest_goal = active_memories[-1].get("goal_evaluation") or {}
                # Do not auto-advance at a step boundary. Replanner must get a
                # chance to add a newly discovered high-value step before the
                # orchestrator moves on.
            try:
                raw_attempt_budget = int(active_before_replan.get("attempt_budget") or 2)
            except (TypeError, ValueError):
                raw_attempt_budget = 2
            latest_active_goal = (
                (active_memories[-1].get("goal_evaluation") or {})
                if active_memories else {}
            )
            # Initial Planner may allocate one attempt even when the first
            # query fails but the database is explicitly reported as having
            # more available directions. Permit up to two Replanner-driven
            # rewrites in that case. The global round budget and the coverage
            # guard above still protect later plan steps.
            replanner_retry_eligible = bool(
                active_memories
                and latest_active_goal.get("completion_status") == "not_met"
                and latest_active_goal.get("database_coverage_status") != "absence_supported"
                and latest_active_goal.get("search_attempt_status") in {"no_hits", "hits_irrelevant", "review_rejected", "fetch_failed"}
                and str(latest_active_goal.get("recommended_query_change") or "").strip()
            )
            if replanner_retry_eligible:
                raw_attempt_budget = max(raw_attempt_budget, 3)
            attempt_budget = max(1, min(self.max_attempts_per_plan_step, raw_attempt_budget))
            attempts_used = len(active_memories)
            step_is_complete = bool(
                latest_active_goal.get("completion_status") == "sufficiently_met"
                or latest_active_goal.get("recommended_stop") is True
                or (
                    latest_active_goal.get("completion_status") == "minimally_met"
                    and not (latest_active_goal.get("critical_gaps") or [])
                )
            )
            if attempts_used >= attempt_budget and not step_is_complete:
                step_budget_status[active_step_id] = {
                    "attempt_budget": attempt_budget,
                    "attempts_used": attempts_used,
                    "status": "budget_exhausted_partial",
                }
                budget_events.append({
                    "event": "step_budget_exhausted",
                    "step_id": active_step_id,
                    "attempt_budget": attempt_budget,
                    "attempts_used": attempts_used,
                    "rounds_used": rounds_used,
                })
                active_step_index += 1
                continue
            if rounds_used >= total_budget and not active_memories:
                state.stop_condition = "total_budget_exhausted"
                break
            round_number = prior_round_count + rounds_used + 1
            budget_state = {
                "initial_total_budget": retrieval_plan.get("initial_total_budget"),
                "current_total_budget": total_budget,
                "hard_total_budget": self.max_total_steps,
                "rounds_used": rounds_used,
                "rounds_remaining": total_budget - rounds_used,
                "active_step_attempt_budget": attempt_budget,
                "active_step_attempts_used": attempts_used,
                "remaining_plan_steps": len(plan_steps) - active_step_index,
                "max_extension_per_request": self.max_budget_extension,
            }
            current_memory_timeline = _memory_timeline(
                current_round=round_number,
                round_memories=round_memories,
                compaction_events=replanner_memory_events,
            )
            if not active_memories and direct_untried_step_execution:
                decision = self._direct_step_decision(
                    active_step=active_before_replan,
                    route_plan=route_plan,
                    first_execution=first_execution if rounds_used == 0 else None,
                )
            elif active_memories:
                latest_goal = active_memories[-1].get("goal_evaluation") or {}
                if (
                    latest_goal.get("database_coverage_status") == "absence_supported"
                    and int(latest_goal.get("matched_goal_count") or 0) == 0
                ):
                    step_budget_status[active_step_id] = {
                        "attempt_budget": attempt_budget,
                        "attempts_used": attempts_used,
                        "status": "query_exhausted",
                    }
                    active_step_index += 1
                    continue
                attempted = {
                    " ".join(str(value).lower().split())
                    for value in latest_goal.get("queries_attempted") or []
                }
                replacement = str(latest_goal.get("recommended_query_change") or "").strip()
                if (
                    latest_goal.get("query_direction_status") == "locally_exhausted"
                    and replacement
                    and " ".join(replacement.lower().split()) not in attempted
                ):
                    decision = self._direct_step_decision(
                        active_step=active_before_replan,
                        route_plan=route_plan,
                    )
                    decision.update({
                        "action": "expand_current_step",
                        "search_query": replacement,
                        "decision_rationale": "Execute the distinct recommended reformulation without another planning call.",
                        "decision_mode": "deterministic_exhaustion_reformulation",
                    })
                else:
                    # Only partial/ambiguous evidence reaches the Replanner.
                    decision = self.planning_agent.replan(
                        main_question=query,
                        plan=retrieval_plan,
                        active_step_index=active_step_index,
                        recent_memories=round_memories,
                        answer_memory=answer_memory,
                        base_query_type=state.query_type,
                        base_constraints=state.confirmed_constraints,
                        budget_state=budget_state,
                        replanner_short_memory=replanner_short_memory,
                        replanner_long_memory=replanner_long_memory,
                        memory_timeline=current_memory_timeline,
                    )
            else:
                decision = self.planning_agent.replan(
                    main_question=query,
                    plan=retrieval_plan,
                    active_step_index=active_step_index,
                    recent_memories=round_memories,
                    answer_memory=answer_memory,
                    base_query_type=state.query_type,
                    base_constraints=state.confirmed_constraints,
                    budget_state=budget_state,
                    replanner_short_memory=replanner_short_memory,
                    replanner_long_memory=replanner_long_memory,
                    memory_timeline=current_memory_timeline,
                )
            try:
                requested_extension = int(decision.get("requested_budget_extension") or 0)
            except (TypeError, ValueError):
                requested_extension = 0
            proposed_plan_changes = [
                dict(item) for item in (decision.get("plan_changes") or [])
                if isinstance(item, dict)
            ][:2]
            if str(decision.get("action") or "") == "revise_plan" and proposed_plan_changes:
                # A newly inserted step must be executable at least once. The
                # reserved budget is used first; request extension only for the
                # uncovered portion.
                rounds_after_current_boundary = max(0, total_budget - rounds_used)
                required_new_rounds = len(proposed_plan_changes)
                requested_extension = max(
                    requested_extension,
                    max(0, required_new_rounds - rounds_after_current_boundary),
                )
            granted_extension = min(
                max(0, requested_extension),
                self.max_budget_extension,
                self.max_total_steps - total_budget,
            )
            if granted_extension:
                previous_budget = total_budget
                total_budget += granted_extension
                budget_events.append({
                    "event": "agent_budget_extension",
                    "step_id": active_step_id,
                    "requested": requested_extension,
                    "granted": granted_extension,
                    "previous_total_budget": previous_budget,
                    "new_total_budget": total_budget,
                    "reason": str(decision.get("budget_extension_reason") or ""),
                })
            action = str(decision.get("action") or "retry_current_step")
            if not self.external_knowledge_enabled and action == "expand_current_step":
                action = "retry_current_step"
                decision["action"] = action
                decision["expansion"] = {
                    "strategy": "none",
                    "source": "",
                    "added_terms": [],
                    "relation_type": "",
                }
                decision["decision_rationale"] = (
                    f"{str(decision.get('decision_rationale') or '').strip()} "
                    "External knowledge is disabled for this ablation; use a case-corpus query retry."
                ).strip()
            if not active_memories and action in {"advance_to_next_step", "stop"}:
                action = "retry_current_step"
                decision["action"] = action
                decision["active_step_id"] = active_step_id
                decision["decision_rationale"] = (
                    "The highest-priority active step has not been executed. "
                    "Run its first information-gain retrieval before advancing or stopping."
                )
            if active_memories:
                latest_goal = active_memories[-1].get("goal_evaluation") or {}
                if (
                    latest_goal.get("completion_status") == "sufficiently_met"
                    or latest_goal.get("recommended_stop") is True
                ) and action != "revise_plan":
                    step_budget_status[active_step_id] = {
                        "attempt_budget": active_before_replan.get("attempt_budget") or 2,
                        "attempts_used": len(active_memories),
                        "status": "completed",
                    }
                    action = "advance_to_next_step"
                    decision["action"] = action
                    decision["decision_rationale"] = "StepMemory marked the step sufficiently complete; advance without another micro-retrieval batch."
                elif latest_goal.get("database_coverage_status") == "absence_supported" and int(latest_goal.get("matched_goal_count") or 0) == 0:
                    action = "advance_to_next_step" if active_step_index + 1 < len(plan_steps) else "stop"
                    decision["action"] = action
                    decision["decision_rationale"] = "Current query has no matching case evidence; stop extending the current small plan."
                elif latest_goal.get("query_direction_status") == "locally_exhausted":
                    attempted = {" ".join(str(x).lower().split()) for x in latest_goal.get("queries_attempted") or []}
                    proposed = " ".join(str(decision.get("search_query") or "").lower().split())
                    if proposed and proposed in attempted:
                        replacement = str(latest_goal.get("recommended_query_change") or "").strip()
                        if replacement and " ".join(replacement.lower().split()) not in attempted:
                            decision["search_query"] = replacement
                            decision["action"] = "expand_current_step"
                            action = "expand_current_step"
                            decision["decision_rationale"] = "Current database query was exhausted; execute the proposed reformulation instead of repeating it."
                        else:
                            action = "advance_to_next_step"
                            decision["action"] = action
                            decision["decision_rationale"] = "Current database query was exhausted and no distinct reformulation was available; advance instead of repeating it."
            if action == "revise_plan" and proposed_plan_changes:
                normalize_steps = getattr(self.planning_agent, "_normalize_steps", None)
                insertion = (
                    normalize_steps(proposed_plan_changes, max_steps=2)
                    if callable(normalize_steps)
                    else proposed_plan_changes
                )
                existing_step_ids = {
                    str(item.get("step_id") or "")
                    for item in (retrieval_plan.get("steps") or [])
                }
                insertion = [
                    item for item in insertion
                    if str(item.get("step_id") or "") not in existing_step_ids
                ]
                if insertion:
                    retrieval_plan["steps"][active_step_index + 1:active_step_index + 1] = insertion
                    decision["plan_changes"] = insertion
                    budget_events.append({
                        "event": "replanner_steps_added",
                        "after_step_id": active_step_id,
                        "added_step_ids": [str(item.get("step_id") or "") for item in insertion],
                        "rounds_used": rounds_used,
                    })
                    # At a completed boundary, proceed directly to the newly
                    # inserted step. If current evidence is still insufficient,
                    # retain the current step and use the revised query first.
                    if active_memories and step_is_complete:
                        action = "advance_to_next_step"
                        decision["action"] = action
                else:
                    action = "advance_to_next_step" if step_is_complete else "retry_current_step"
                    decision["action"] = action
            if action == "stop":
                replan_decisions.append(dict(decision))
                state.stop_condition = "replanner_stop"
                break
            # Replanner gets one budget-free decision at the boundary. It may
            # request an extension, but retrieval cannot execute unless granted.
            if rounds_used >= total_budget and action not in {"advance_to_next_step", "stop"}:
                replan_decisions.append(dict(decision))
                state.stop_condition = "total_budget_exhausted"
                break
            if action == "advance_to_next_step":
                active_step_index += 1
                if active_step_index >= len(retrieval_plan.get("steps") or []):
                    state.stop_condition = "retrieval_plan_completed"
                    break
                if direct_untried_step_execution:
                    decision = self._direct_step_decision(
                        active_step=dict((retrieval_plan.get("steps") or [])[active_step_index]),
                        route_plan=route_plan,
                    )
                else:
                    decision = self.planning_agent.replan(
                        main_question=query,
                        plan=retrieval_plan,
                        active_step_index=active_step_index,
                        recent_memories=round_memories,
                        answer_memory=answer_memory,
                        base_query_type=state.query_type,
                        base_constraints=state.confirmed_constraints,
                        budget_state={**budget_state, "current_total_budget": total_budget, "rounds_remaining": total_budget - rounds_used},
                        replanner_short_memory=replanner_short_memory,
                        replanner_long_memory=replanner_long_memory,
                        memory_timeline=current_memory_timeline,
                    )
                action = str(decision.get("action") or "retry_current_step")
            if rounds_used >= total_budget:
                replan_decisions.append(dict(decision))
                state.stop_condition = "total_budget_exhausted"
                break
            active_step = dict((retrieval_plan.get("steps") or [])[active_step_index])
            current_query = str(decision.get("search_query") or active_step.get("goal") or query).strip()
            rerank_goal = str(decision.get("rerank_goal") or active_step.get("rerank_goal") or active_step.get("goal") or query).strip()
            selected_tools = [
                str(name) for name in decision.get("selected_tools") or []
                if str(name) in {"structured_search", "dense_search", "bm25_search", "hybrid_search", "fetch_evidence"}
            ] or list(route_plan.selected_tools)
            if any(name != "fetch_evidence" for name in selected_tools) and "fetch_evidence" not in selected_tools:
                selected_tools.append("fetch_evidence")
            execution_constraints = dict(state.confirmed_constraints)
            # Replanner constraints are retrieval proposals, not confirmed
            # patient facts. Candidate entities stay in query/expansion and may
            # not become hard structured filters here.
            execution_plan = RoutePlan(
                query_type=None,
                selected_tools=selected_tools,
                constraints=execution_constraints,
                search_query=current_query,
                planner_metadata={
                    "plan_step": active_step,
                    "replan_action": action,
                    "decision_rationale": decision.get("decision_rationale"),
                    "proposed_constraints": decision.get("constraints") or {},
                    "constraint_policy": "confirmed_only",
                    "expansion": decision.get("expansion") or {},
                    "avoid_repeating": decision.get("avoid_repeating") or [],
                },
            )
            expansion_decision = dict(decision.get("expansion") or {})
            expansion_policy = {
                "enabled": action == "expand_current_step",
                "level": 1 if action == "expand_current_step" else 0,
                "allowed_strategies": self._expand_allowed_strategies(
                    action=action,
                    query_type=None,
                    evidence_lane=str(active_step.get("evidence_lane") or "direct_case"),
                    requested_strategy=str(expansion_decision.get("strategy") or ""),
                ),
                "source": str(expansion_decision.get("source") or ""),
                "relation_type": str(expansion_decision.get("relation_type") or ""),
            }
            if self.execution_agent is not None:
                try:
                    route_result = self.execution_agent.run(
                        main_question=query,
                        plan_step=active_step,
                        route_plan=execution_plan,
                        rerank_goal=rerank_goal,
                        top_k=self.top_k,
                        expansion=expansion_policy,
                        previously_seen_chunk_ids=sorted(seen_chunk_ids),
                    )
                except Exception as exc:
                    logger.warning("Execution agent failed, falling back to deterministic router: %s", exc)
                    route_result = self.router.run(
                        query=current_query,
                        top_k=self.top_k,
                        main_question=query,
                        route_plan=execution_plan,
                        rerank_goal=rerank_goal,
                    )
                    route_result["execution_fallback"] = {
                        "reason": type(exc).__name__,
                        "detail": str(exc)[:500],
                    }
            else:
                route_result = self.router.run(
                    query=current_query,
                    top_k=self.top_k,
                    main_question=query,
                    route_plan=execution_plan,
                    rerank_goal=rerank_goal,
                )
            route_result["plan_step"] = active_step
            route_result["replan_decision"] = decision
            route_result["main_question"] = query
            route_result["problem_representation"] = dict(state.search_hints or {})
            last_route_result = route_result
            rerank_backend = str(route_result.get("rerank_backend") or "")
            if rerank_backend == "QwenWithLLMFallbackReranker":
                # Qwen mode first uses the deterministic evidence signals. A
                # review LLM is reserved for rounds whose first evidence
                # window has no supporting material.
                rule_review = self.evidence_review_agent._review_rules(route_result)
                if rule_review.get("supporting_evidence"):
                    rule_review["review_mode"] = "qwen_rules_confident"
                    citation_review = rule_review
                else:
                    citation_review = self.evidence_review_agent.review(route_result=route_result)
            else:
                citation_review = self.evidence_review_agent.review(route_result=route_result)
            citation_review = self._qwen_overflow_rescue(route_result, citation_review)
            if isinstance(citation_review.get("structured_output_recovery"), dict):
                structured_output_recoveries.append({
                    "stage": "evidence_review",
                    "round": round_number,
                    **citation_review["structured_output_recovery"],
                })
            round_memory = self.step_memory_agent.maintain(
                round_number=round_number,
                main_question=query,
                query=current_query,
                route_result=route_result,
                evidence_review=citation_review,
                seen_chunk_ids=seen_chunk_ids,
                plan_step=active_step,
            )
            round_memory.setdefault("memory_id", f"agent-round:{round_number}")
            round_memory.setdefault("created_at", _utc_now())
            round_memory.setdefault("agent_round", round_number)
            round_memories.append(round_memory)
            pending_replanner_memories.append(round_memory)
            should_compact = self.replanner_memory_agent.should_consolidate(
                pending_replanner_memories,
                replanner_short_memory,
                replanner_long_memory,
            )
            # Keep the two newest rounds verbatim in the model-visible tail;
            # only older complete rounds are replaced by historical memory.
            compactable_memories = pending_replanner_memories[:-2] if should_compact else []
            compaction_id = str(uuid.uuid4()) if compactable_memories else ""
            event_emitter = getattr(getattr(self.router, "llm_client", None), "emit_trace_event", None)
            tokens_before = self.replanner_memory_agent.estimate_tokens({
                "short": replanner_short_memory,
                "long": replanner_long_memory,
                "pending": compactable_memories,
            }) if compactable_memories else 0
            if compactable_memories and callable(event_emitter):
                event_emitter("compaction/start", {
                    "compaction_id": compaction_id,
                    "current_agent_round": round_number + 1,
                    "covered_rounds": [int(item.get("round") or 0) for item in compactable_memories],
                    "source_memory_ids": [str(item.get("memory_id") or "") for item in compactable_memories],
                })
            trace_reader = getattr(getattr(self.router, "llm_client", None), "trace_events", None)
            trace_before_compaction = len(trace_reader()) if compactable_memories and callable(trace_reader) else 0
            memory_update = self.replanner_memory_agent.maintain(
                main_question=query,
                plan=retrieval_plan,
                active_step_id=str(active_step.get("step_id") or ""),
                short_memory=replanner_short_memory,
                long_memory=replanner_long_memory,
                pending_step_memories=compactable_memories,
                force=bool(compactable_memories),
            )
            if memory_update.get("consolidated"):
                replanner_short_memory = dict(memory_update.get("short_memory") or {})
                replanner_long_memory = dict(memory_update.get("long_memory") or {})
                consumed = int(memory_update.get("consumed_count") or 0)
                pending_replanner_memories = pending_replanner_memories[consumed:]
                covered = [int(item.get("round") or 0) for item in compactable_memories[:consumed]]
                tokens_after = self.replanner_memory_agent.estimate_tokens({"short": replanner_short_memory, "long": replanner_long_memory})
                compaction_llm_event: Dict[str, Any] = {}
                if callable(trace_reader):
                    new_llm_events = trace_reader()[trace_before_compaction:]
                    compaction_llm_event = next((
                        dict(item) for item in reversed(new_llm_events)
                        if item.get("operation") == "function:maintain_replanner_memory"
                    ), {})
                compaction_event = {
                    "compaction_id": compaction_id,
                    "created_at": _utc_now(),
                    "current_agent_round": round_number + 1,
                    "latest_completed_agent_round": round_number,
                    "covered_rounds": covered,
                    "covered_round_range": {"start": min(covered), "end": max(covered)} if covered else {"start": None, "end": None},
                    "source_memory_ids": [str(item.get("memory_id") or "") for item in compactable_memories[:consumed]],
                    "recent_full_rounds": [int(item.get("round") or 0) for item in pending_replanner_memories[-2:]],
                    "consumed_step_memories": consumed,
                    "estimated_tokens_before": tokens_before,
                    "estimated_tokens_after": tokens_after,
                    "memory_mode": memory_update.get("memory_mode") or ("llm" if self.replanner_memory_agent.use_llm else "rules"),
                    "summary": {"short_memory": replanner_short_memory, "long_memory": replanner_long_memory},
                    "summarizer_call": {
                        "call_id": compaction_llm_event.get("call_id"),
                        "model": compaction_llm_event.get("model"),
                        "usage": {
                            key: compaction_llm_event.get(key)
                            for key in ("input_tokens", "output_tokens", "total_tokens")
                            if compaction_llm_event.get(key) is not None
                        },
                        "raw_output": compaction_llm_event.get("response"),
                    } if compaction_llm_event else None,
                }
                replanner_memory_events.append(compaction_event)
                if callable(event_emitter):
                    event_emitter("compaction/summary", compaction_event)
                    event_emitter("compaction/end", {"compaction_id": compaction_id, "status": "complete"})
            answer_memory = self.answer_memory_agent.update(answer_memory, round_memory)
            rounds_used += 1

            # route_result only echoes execution constraints; do not promote
            # retrieval hypotheses into confirmed patient facts.
            state.identified_entities = self._extract_entities(state.confirmed_constraints)
            evidence_rows = self._collect_evidence_rows(citation_review=citation_review)
            new_chunk_count = 0
            for row in evidence_rows:
                if row.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(row.chunk_id)
                state.retrieved_evidence.append(row)
                new_chunk_count += 1

            missing_information = self._build_missing_information(
                query_type=state.query_type,
                citation_review=citation_review,
                constraints=state.confirmed_constraints,
            )
            current_goal = round_memory.get("goal_evaluation") or {}
            next_targets = [str(item) for item in [*(current_goal.get("critical_gaps") or []), *(current_goal.get("optional_gaps") or current_goal.get("observed_gaps") or [])]]

            state.loop_count = rounds_used
            state.missing_information = missing_information
            state.next_retrieval_targets = next_targets
            state.stop_condition = "running"
            replan_decisions.append(dict(decision))

            steps.append(
                AgentLoopStep(
                    step=rounds_used,
                    query=current_query,
                    query_type=route_result.get("query_type", state.query_type),
                    selected_tools=list(route_result.get("selected_tools", [])),
                    citation_verdict=citation_review.get("verdict", "insufficient_evidence"),
                    contradiction_check=dict(route_result.get("contradiction_check", {})),
                    missing_information=missing_information,
                    next_retrieval_targets=next_targets,
                    new_chunk_count=new_chunk_count,
                    plan_step_id=str(active_step.get("step_id") or ""),
                    step_goal=str(active_step.get("goal") or ""),
                    replan_action=action,
                    rerank_goal=rerank_goal,
                )
            )

            completion_status = str(current_goal.get("completion_status") or ("minimally_met" if current_goal.get("success_criteria_met") else "not_met"))
            completed = completion_status == "sufficiently_met" or current_goal.get("recommended_stop") is True
            if completed:
                step_budget_status[active_step_id] = {
                    "attempt_budget": attempt_budget,
                    "attempts_used": attempts_used + 1,
                    "status": "completed",
                }
                # Even when this was the last initial step, keep the loop alive
                # for one boundary Replanner decision. It may append a newly
                # discovered high-value step or explicitly finish the plan.
                if (
                    active_step_index + 1 >= len(plan_steps)
                    and rounds_used >= self.max_total_steps
                ):
                    active_step_index += 1
                    state.stop_condition = "retrieval_plan_completed"
                    break
            elif completion_status == "minimally_met":
                step_budget_status[active_step_id] = {
                    "attempt_budget": attempt_budget,
                    "attempts_used": attempts_used + 1,
                    "status": "minimally_met",
                }

        if state.stop_condition == "running":
            plan_steps = retrieval_plan.get("steps") or []
            if active_step_index >= len(plan_steps):
                state.stop_condition = "retrieval_plan_completed"
            elif rounds_used >= total_budget:
                state.stop_condition = "total_budget_exhausted"
            elif rounds_used >= self.max_total_steps:
                state.stop_condition = "hard_total_budget_exhausted"

        cumulative_rows = list(answer_memory.get("evidence_by_id", {}).values())
        final_safety_review = self.router.safety_gate.evaluate(
            query=query,
            query_type=state.query_type,
            constraints=state.confirmed_constraints,
            retrieval_results={"fetch_evidence": cumulative_rows},
            answer_memory=answer_memory,
        )
        if isinstance(final_safety_review.get("structured_output_recovery"), dict):
            structured_output_recoveries.append({
                "stage": "safety_reflection",
                **final_safety_review["structured_output_recovery"],
            })
        answer_memory = self.answer_memory_agent.apply_safety_review(answer_memory, final_safety_review)
        citations = CitationAgent().trace_claims(
            answer_memory.get("claims") or [], answer_memory.get("evidence_by_id") or {}
        )

        final_memory_timeline = _memory_timeline(
            current_round=prior_round_count + rounds_used + 1,
            round_memories=round_memories,
            compaction_events=replanner_memory_events,
        )
        return {
            "query": query,
            "query_type": state.query_type,
            "state": state.to_dict(),
            "loop_steps": [asdict(item) for item in steps],
            "round_memories": round_memories,
            "step_memories": round_memories,
            "replanner_short_memory": replanner_short_memory,
            "replanner_long_memory": replanner_long_memory,
            "pending_replanner_step_memories": pending_replanner_memories,
            "replanner_memory_events": replanner_memory_events,
            "memory_timeline": final_memory_timeline,
            "retrieval_plan": retrieval_plan,
            "replan_decisions": replan_decisions,
            "active_plan_step_index": active_step_index,
            "budget_state": {
                "initial_total_budget": retrieval_plan.get("initial_total_budget"),
                "final_total_budget": total_budget,
                "hard_total_budget": self.max_total_steps,
                "rounds_used": rounds_used,
                "rounds_remaining": max(0, total_budget - rounds_used),
                "max_extension_per_request": self.max_budget_extension,
            },
            "budget_events": budget_events,
            "step_budget_status": step_budget_status,
            "answer_memory": answer_memory,
            "structured_output_recoveries": structured_output_recoveries,
            "final_safety_review": final_safety_review,
            "citations": citations,
            "last_route_result": last_route_result,
        }
