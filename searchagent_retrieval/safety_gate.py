from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from .structured_output_recovery import configured_batch_size, recovery_trace, run_adaptive, unique_strings

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)

SAFETY_REFLECTION_SYSTEM_PROMPT = """你是 ClinicalSafetyGate 的临床安全自反思模块。输入 claims 已由程序预筛，只包含可能进入 AnswerContext 且具有证据 ID 的候选。你必须逐条审核这些候选 claim 是否与主问题、患者约束、支持证据、反证、毒性、禁忌和证据等级冲突。严格保留治疗先后、事件发生时间和因果边界：基线异常、既往治疗事件、当前治疗后事件和未来预测不得互换；较晚进展不得改写成较早时间窗无效；联合治疗结局不得无依据归因于单药。规则安全问题是不可删除的底线。你可以批准、降级、改写或否决 claim，但不得扩大证据结论，不得给出处方决定。必须通过 submit_safety_reflection 函数提交结果。"""

SAFETY_REFLECTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "claim_reviews": {"type": "array", "items": {"type": "object", "properties": {
            "claim_id": {"type": "string"}, "decision": {"type": "string", "enum": ["approve", "revise", "reject"]},
            "violations": {"type": "array", "items": {"type": "string"}}, "required_revision": {"type": "string"},
            "safe_claim": {"type": "string"}, "evidence_refs": {"type": "array", "items": {"type": "string"}}},
            "required": ["claim_id", "decision", "violations", "required_revision", "safe_claim", "evidence_refs"], "additionalProperties": False}},
        "cross_claim_conflicts": {"type": "array", "items": {"type": "string"}},
        "additional_issues": {"type": "array", "items": {"type": "object", "properties": {
            "code": {"type": "string"}, "severity": {"type": "string", "enum": ["medium", "high", "critical"]},
            "title": {"type": "string"}, "message": {"type": "string"}, "recommendation": {"type": "string"},
            "evidence_refs": {"type": "array", "items": {"type": "string"}}},
            "required": ["code", "severity", "title", "message", "recommendation", "evidence_refs"], "additionalProperties": False}},
        "reflection_summary": {"type": "string"},
    },
    "required": ["claim_reviews", "cross_claim_conflicts", "additional_issues", "reflection_summary"],
    "additionalProperties": False,
}


SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

DOSE_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|μg|ug|qd|bid|tid|q\d+h))\b", re.IGNORECASE)
SEVERE_TOXICITY_PATTERN = re.compile(
    r"\b(grade\s*[34]|fatal|death|life[-\s]?threatening|qt\s*prolong|hepatotox|pneumonitis|interstitial|cardiotox)\b",
    re.IGNORECASE,
)
DDI_PATTERN = re.compile(r"\b(ddi|drug interaction|interaction|cyp3a4|cyp2d6|p-gp|bcrp|禁忌|相互作用)\b", re.IGNORECASE)
GUIDELINE_PATTERN = re.compile(r"\b(guideline|nccn|esmo|asco|说明书|药品标签)\b", re.IGNORECASE)


@dataclass
class SafetyIssue:
    code: str
    severity: str
    title: str
    message: str
    recommendation: str
    evidence_refs: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "recommendation": self.recommendation,
            "evidence_refs": self.evidence_refs,
        }


def _extract_hits_text(results: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    for tool_name in ("structured_search", "dense_search", "bm25_search", "hybrid_search"):
        entries = results.get(tool_name, [])
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text)
    return texts


def _extract_evidence_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = results.get("fetch_evidence", [])
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _max_severity(issues: Sequence[SafetyIssue]) -> str:
    max_level = 0
    max_name = "low"
    for issue in issues:
        level = SEVERITY_ORDER.get(issue.severity, 0)
        if level > max_level:
            max_level = level
            max_name = issue.severity
    return max_name


def _claim_evidence_ids(claim: Dict[str, Any]) -> List[str]:
    values: List[object] = []
    for key in (
        "direct_support_chunk_ids",
        "supporting_chunk_ids",
        "partial_support_chunk_ids",
        "analog_support_chunk_ids",
        "contradicting_chunk_ids",
    ):
        values.extend(claim.get(key) or [])
    return unique_strings(values)


def _claim_enters_answer_context(claim: Dict[str, Any]) -> bool:
    """Mirror the evidence-bound lanes consumed by AnswerContext."""
    if not isinstance(claim, dict):
        return False
    status = str(claim.get("status") or "").strip().lower()
    if status in {"rejected_by_safety_gate", "irrelevant", "rejected", "discarded"}:
        return False
    role = str(claim.get("overall_role") or claim.get("evidence_role") or "").strip().lower()
    if role in {"irrelevant", "insufficient"}:
        return False
    if not str(claim.get("claim_id") or "").strip() or not _claim_evidence_ids(claim):
        return False
    level = str(claim.get("support_level") or "").strip().lower()
    if level == "direct":
        return bool(claim.get("direct_support_chunk_ids") or claim.get("supporting_chunk_ids"))
    if level == "partial":
        return bool(claim.get("partial_support_chunk_ids"))
    if level == "analog":
        return bool(claim.get("analog_support_chunk_ids"))
    return False


class ClinicalSafetyGate:
    """Rule-based safety gate for medication-assist scenarios."""

    def __init__(self, llm_client: Optional["LLMClient"] = None, use_llm: bool = True) -> None:
        self.llm_client = llm_client
        self.use_llm = use_llm

    def evaluate(
        self,
        query: str,
        query_type: Optional[str],
        constraints: Dict[str, Any],
        retrieval_results: Dict[str, Any],
        answer_memory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        hits_text = _extract_hits_text(retrieval_results)
        evidence_rows = _extract_evidence_rows(retrieval_results)
        evidence_text = "\n".join(
            [row.get("text", "") for row in evidence_rows if isinstance(row.get("text"), str)]
            + hits_text
        )
        query_and_evidence = f"{query}\n{evidence_text}"
        issues: List[SafetyIssue] = []

        if len(evidence_rows) < 3:
            issues.append(
                SafetyIssue(
                    code="insufficient_evidence",
                    severity="high",
                    title="证据数量不足",
                    message=f"仅检索到 {len(evidence_rows)} 条可溯源证据，难以支撑临床用药决策。",
                    recommendation="补充检索更多病例、反证信息与外部指南/说明书后再给出建议。",
                    evidence_refs=[],
                )
            )

        if DOSE_PATTERN.search(query):
            issues.append(
                SafetyIssue(
                    code="dose_recommendation_risk",
                    severity="high",
                    title="涉及剂量建议",
                    message="问题包含明确剂量或给药频率，单靠病例检索存在较高误导风险。",
                    recommendation="必须核对药品说明书、肝肾功能、合并用药，并由临床医生最终确认。",
                    evidence_refs=[],
                )
            )

        ddi_requested = bool(constraints.get("ddi_terms")) or bool(DDI_PATTERN.search(query))
        if ddi_requested:
            ddi_evidence = [row for row in evidence_rows if "ddi" in str(row.get("evidence_level", "")).lower()]
            if not ddi_evidence:
                issues.append(
                    SafetyIssue(
                        code="missing_ddi_evidence",
                        severity="high",
                        title="缺少可靠 DDI 证据",
                        message="问题涉及药物相互作用，但当前证据主要来自病例库，未见明确 DDI 规则/标签依据。",
                        recommendation="补充 DDI 规则库或说明书来源，再评估是否存在禁忌/需调剂量情形。",
                        evidence_refs=[str(row.get("chunk_id")) for row in evidence_rows[:3]],
                    )
                )

        severe_rows = [
            row for row in evidence_rows if SEVERE_TOXICITY_PATTERN.search(str(row.get("text", "")))
        ]
        if severe_rows:
            issues.append(
                SafetyIssue(
                    code="severe_toxicity_signal",
                    severity="critical",
                    title="检出严重毒性信号",
                    message="证据中出现 Grade 3/4、致死或器官严重毒性相关描述，需要优先安全处置。",
                    recommendation="优先提示停药/减量/替代方案评估，并建议 MDT 或专科会诊。",
                    evidence_refs=[str(row.get("chunk_id")) for row in severe_rows[:5]],
                )
            )

        evidence_levels = {str(row.get("evidence_level", "")) for row in evidence_rows}
        only_low_level = evidence_levels and evidence_levels.issubset(
            {"case_report_evidence", "structured_extraction_evidence"}
        )
        if only_low_level and not GUIDELINE_PATTERN.search(query_and_evidence):
            issues.append(
                SafetyIssue(
                    code="low_evidence_hierarchy",
                    severity="medium",
                    title="证据等级偏低",
                    message="当前证据以病例报告及结构化抽取为主，不能直接外推为指南级推荐。",
                    recommendation="在回答中明确证据边界，并建议同步查阅指南、说明书和真实世界数据。",
                    evidence_refs=[str(row.get("chunk_id")) for row in evidence_rows[:3]],
                )
            )

        risk_level = _max_severity(issues)
        requires_human_review = SEVERITY_ORDER.get(risk_level, 0) >= SEVERITY_ORDER["high"]
        recommended_actions = [
            "区分“病例库证据”与“模型背景知识”",
            "输出支持证据、反证与证据不足点",
            "提示结果不能替代指南、说明书与 MDT 个体化决策",
        ]
        if requires_human_review:
            recommended_actions.append("将最终处方决策交由临床医生复核")

        result = {
            "risk_level": risk_level,
            "requires_human_review": requires_human_review,
            "checks": {
                "evidence_row_count": len(evidence_rows),
                "has_ddi_intent": ddi_requested,
                "has_dose_intent": bool(DOSE_PATTERN.search(query)),
                "severe_toxicity_signal_count": len(severe_rows),
                "evidence_levels": sorted([level for level in evidence_levels if level]),
            },
            "issues": [issue.to_dict() for issue in issues],
            "recommended_actions": recommended_actions,
            "safety_boundary": (
                "本结果仅用于医生检索辅助，不能替代药品说明书、临床指南、MDT 讨论与患者个体化评估。"
            ),
        }
        all_claims = [row for row in ((answer_memory or {}).get("claims") or []) if isinstance(row, dict)]
        eligible_claims = [row for row in all_claims if _claim_enters_answer_context(row)]
        if self.use_llm and self.llm_client is not None and eligible_claims:
            try:
                compact_evidence = {}
                referenced_ids = {
                    chunk_id for claim in eligible_claims for chunk_id in _claim_evidence_ids(claim)
                }
                for chunk_id, item in (answer_memory.get("evidence_by_id") or {}).items():
                    if str(chunk_id) not in referenced_ids or not isinstance(item, dict):
                        continue
                    compact_evidence[chunk_id] = {
                        "evidence_level": item.get("evidence_level"),
                        "text": " ".join(str(item.get("text") or "").split())[:500],
                        "relevance_signals": item.get("relevance_signals") or {},
                    }
                compact_rule_result = {
                    "risk_level": result.get("risk_level"),
                    "requires_human_review": result.get("requires_human_review"),
                    "issues": result.get("issues") or [],
                    "safety_boundary": result.get("safety_boundary"),
                }
                claim_index = [
                    {
                        "claim_id": str(row.get("claim_id") or ""),
                        "claim": str(row.get("safe_claim") or row.get("claim") or "")[:240],
                    }
                    for row in eligible_claims
                ]
                batch_size = configured_batch_size("SAFETY_REFLECTION_BATCH_SIZE", 4)

                def reflect_part(part: List[Dict[str, Any]], part_number: int, part_count: int) -> Dict[str, Any]:
                    part_ids = {str(row.get("claim_id") or "") for row in part}
                    part_evidence = {
                        chunk_id: row
                        for chunk_id, row in compact_evidence.items()
                        if any(chunk_id in _claim_evidence_ids(claim) for claim in part)
                    }
                    reflection_input = {
                        "query": query,
                        "constraints": constraints,
                        "rule_safety_result": compact_rule_result,
                        "all_eligible_claim_index": claim_index,
                        "claims_to_review_in_this_part": part,
                        "evidence_by_id": part_evidence,
                    }
                    prompt = (
                        f"输入：\n{json.dumps(reflection_input, ensure_ascii=False, indent=2)}\n\n"
                        f"这是同一安全审核的第 {part_number}/{part_count} 部分。"
                        "每个 claims_to_review_in_this_part claim 必须恰有一条 review；"
                        "claim_reviews 不得输出其他 claim_id。all_eligible_claim_index 仅用于发现跨claim矛盾。"
                        "返回一个自身完整的函数参数对象；规则安全结果不可撤销。"
                    )
                    payload = self.llm_client.call_function(
                        system=SAFETY_REFLECTION_SYSTEM_PROMPT,
                        user=prompt,
                        function_name="submit_safety_reflection",
                        description="Submit safety decisions for this bounded claim part.",
                        parameters=SAFETY_REFLECTION_SCHEMA,
                        temperature=0.1,
                        max_output_tokens=8000,
                    )
                    reviews = [row for row in (payload.get("claim_reviews") or []) if isinstance(row, dict)]
                    returned_ids = {str(row.get("claim_id") or "") for row in reviews}
                    if returned_ids != part_ids:
                        raise ValueError(
                            f"safety reflection claim IDs mismatch: expected={sorted(part_ids)} returned={sorted(returned_ids)}"
                        )
                    return payload

                reflections, recovered_from_truncation = run_adaptive(
                    eligible_claims,
                    batch_size=batch_size,
                    call_part=reflect_part,
                )
                result["claim_reviews"] = [
                    row
                    for reflection in reflections
                    for row in (reflection.get("claim_reviews") or [])
                    if isinstance(row, dict)
                ]
                result["cross_claim_conflicts"] = unique_strings([
                    value
                    for reflection in reflections
                    for value in (reflection.get("cross_claim_conflicts") or [])
                ])
                result["reflection_summary"] = "；".join(unique_strings([
                    reflection.get("reflection_summary") for reflection in reflections
                ]))
                seen_issues = set()
                for issue in [
                    row
                    for reflection in reflections
                    for row in (reflection.get("additional_issues") or [])
                ]:
                    if isinstance(issue, dict) and issue.get("severity") in {"medium", "high", "critical"}:
                        signature = json.dumps(issue, ensure_ascii=False, sort_keys=True)
                        if signature not in seen_issues:
                            seen_issues.add(signature)
                            result["issues"].append(issue)
                if any(item.get("decision") in {"revise", "reject"} for item in result["claim_reviews"] if isinstance(item, dict)):
                    result["requires_human_review"] = True
                    if result["risk_level"] in {"low", "medium"}:
                        result["risk_level"] = "high"
                result["review_mode"] = "rules+llm"
                result["claim_prefilter"] = {
                    "input_claims": len(all_claims),
                    "reviewed_claims": len(eligible_claims),
                    "excluded_claims": len(all_claims) - len(eligible_claims),
                }
                result["structured_output_recovery"] = recovery_trace(
                    input_items=len(eligible_claims),
                    part_count=len(reflections),
                    batch_size=batch_size,
                    initial_attempt_failed=recovered_from_truncation,
                )
            except Exception as exc:
                if os.environ.get("STRICT_LLM_PIPELINE") == "1":
                    raise
                logger.warning("LLM clinical safety reflection failed, retaining rule result: %s", exc)
                result["review_mode"] = "rules"
        else:
            result["review_mode"] = "rules"
            if answer_memory:
                result["claim_prefilter"] = {
                    "input_claims": len(all_claims),
                    "reviewed_claims": 0,
                    "excluded_claims": len(all_claims),
                }
        return result
