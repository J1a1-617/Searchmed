from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from .structured_output_recovery import configured_batch_size, recovery_trace, run_adaptive, unique_strings

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)

_STRINGS = {"type": "array", "items": {"type": "string"}}
_VALIDATED_FINDING = {
    "type": "object",
    "properties": {
        "claim_id": {"type": "string"},
        "text": {"type": "string"},
        "evidence_ids": _STRINGS,
    },
    "required": ["claim_id", "text", "evidence_ids"],
    "additionalProperties": False,
}
_BOUNDED_FINDING = {
    "type": "object",
    "properties": {
        "claim_id": {"type": "string"},
        "text": {"type": "string", "description": "Copy the input claim text verbatim. Partial/analog evidence may not be paraphrased or enriched."},
        "support_level": {"type": "string", "enum": ["partial", "analog"]},
        "evidence_ids": _STRINGS,
    },
    "required": ["claim_id", "text", "support_level", "evidence_ids"],
    "additionalProperties": False,
}
ANSWER_CONTEXT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "case_context": {"type": "array", "description": "Confirmed patient facts needed to interpret the answer.", "items": {"type": "string"}},
        "key_findings": {"type": "array", "description": "Directly supported findings, bound to stable claim IDs. Text may be a faithful paraphrase.", "items": _VALIDATED_FINDING},
        "partial_or_analog_findings": {"type": "array", "description": "Partial or analog findings bound to their real evidence IDs. They may inform an uncertain prediction but must not be stated as direct target-specific facts.", "items": _BOUNDED_FINDING},
        "unresolved_gaps": {"type": "array", "description": "Important questions not resolved by available evidence.", "items": {"type": "string"}},
        "conflicts_and_limitations": {"type": "array", "description": "Contradictions, weak analogies, and evidence limitations.", "items": {"type": "string"}},
        "safety_boundaries": {"type": "array", "description": "Clinical risks and claims the final answer must not overstate.", "items": {"type": "string"}},
        "prohibited_attributions": {"type": "array", "description": "Explicit entity or causal attributions forbidden by evidence mismatch.", "items": {"type": "string"}},
        "evidence_ids": {"type": "array", "description": "Real retained chunk or event IDs supporting the context.", "items": {"type": "string"}},
    },
    "required": ["case_context", "key_findings", "partial_or_analog_findings", "unresolved_gaps", "conflicts_and_limitations", "safety_boundaries", "prohibited_attributions", "evidence_ids"],
    "additionalProperties": False,
}


class AnswerContextAgent:
    """Produce a bounded, final-answer-specific context without raw session history."""

    SYSTEM_PROMPT = """你是 AnswerContextAgent。你只为最终回答提炼上下文。输入已经过字段白名单压缩；请进一步合并重复内容。key_findings 的每项必须保留输入中的稳定 claim_id，并只引用该 claim 的 direct_support_chunk_ids；直接证据的text可以忠实同义改写，不要求逐字复制。

部分或类比证据必须放入 partial_or_analog_findings，保留 claim_id、support_level 以及对应的 partial_support_chunk_ids 或 analog_support_chunk_ids。弱证据的text必须逐字复制输入中同一claim_id的claim，不得改写、概括、补充主语、疗效方向、毒性、剂量、时间、因果关系或适用人群；如果原claim本身不足以表达安全结论，可以省略该项或把限制写入conflicts_and_limitations，不能“帮助它写完整”。部分或类比证据只能作为有边界的参考，不能改写成目标药物的直接事实。实体不匹配时必须写入 prohibited_attributions。

治疗时间线是结论边界，不是可压缩掉的修饰语：必须区分基线、既往治疗、当前治疗、各随访时点与未来预测；保留治疗先后和事件相对治疗的发生时间。较晚进展不能改写成较早时间窗无效，治疗后的事件不能倒置成基线事实，联合方案结局不能无依据归因于单药。不得补充输入中不存在的医学事实或引用。必须通过 summarize_answer_context 提交。"""

    def __init__(self, llm_client: Optional["LLMClient"] = None, use_llm: bool = True) -> None:
        self.llm_client = llm_client
        self.use_llm = use_llm

    @staticmethod
    def _compact_claims(memory: Dict[str, Any], limit: int = 10) -> list[Dict[str, Any]]:
        rows = []
        for claim in (memory.get("claims") or [])[:limit]:
            if not isinstance(claim, dict) or claim.get("status") == "rejected_by_safety_gate":
                continue
            rows.append({
                "claim_id": str(claim.get("claim_id") or ""),
                "claim": str(claim.get("safe_claim") or claim.get("claim") or "")[:500],
                "status": claim.get("status"),
                "confidence": claim.get("confidence"),
                "supporting_chunk_ids": (claim.get("supporting_chunk_ids") or [])[:3],
                "direct_support_chunk_ids": (claim.get("direct_support_chunk_ids") or [])[:3],
                "partial_support_chunk_ids": (claim.get("partial_support_chunk_ids") or [])[:3],
                "analog_support_chunk_ids": (claim.get("analog_support_chunk_ids") or [])[:3],
                "subject_entity": claim.get("subject_entity"),
                "claim_scope": claim.get("claim_scope"),
                "support_level": claim.get("support_level"),
                "evidence_scopes": (claim.get("evidence_scopes") or [])[:4],
                "supports_dimensions": (claim.get("supports_dimensions") or [])[:8],
                "mismatched_dimensions": (claim.get("mismatched_dimensions") or [])[:8],
                "unreported_dimensions": (claim.get("unreported_dimensions") or [])[:8],
                "unsupported_dimensions": (claim.get("unsupported_dimensions") or [])[:12],
                "contradicting_chunk_ids": (claim.get("contradicting_chunk_ids") or [])[:3],
                "required_revision": str(claim.get("required_revision") or "")[:300],
            })
        return rows

    def summarize(self, *, query: str, loop_result: Dict[str, Any], safety_result: Dict[str, Any]) -> Dict[str, Any]:
        prior = loop_result.get("prior_session_context") or {}
        payload = {
            "query": query,
            "confirmed_constraints": (loop_result.get("state") or {}).get("confirmed_constraints") or {},
            "replanner_long_memory": loop_result.get("replanner_long_memory") or prior.get("replanner_long_memory") or {},
            "replanner_short_memory": loop_result.get("replanner_short_memory") or prior.get("replanner_short_memory") or {},
            "current_claims": self._compact_claims(loop_result.get("answer_memory") or {}),
            "prior_claims": self._compact_claims(prior.get("answer_memory") or {}, limit=6),
            "safety": {
                "risk_level": safety_result.get("risk_level"),
                "issues": [{"severity": row.get("severity"), "title": str(row.get("title") or "")[:240]} for row in (safety_result.get("issues") or [])[:6] if isinstance(row, dict)],
                "recommended_actions": [str(x)[:240] for x in (safety_result.get("recommended_actions") or [])[:6]],
            },
        }
        direct_claims = [row for row in payload["current_claims"] if row.get("support_level") == "direct" and row.get("direct_support_chunk_ids")]
        weak_claims = [row for row in payload["current_claims"] if row.get("support_level") in {"partial", "analog"}]
        fallback = {
            "case_context": [json.dumps(payload["confirmed_constraints"], ensure_ascii=False, separators=(",", ":"))],
            "key_findings": [{"claim_id": row["claim_id"], "text": row["claim"], "evidence_ids": list(row["direct_support_chunk_ids"])} for row in direct_claims[:6] if row["claim"] and row["claim_id"]],
            "partial_or_analog_findings": [
                {
                    "claim_id": row["claim_id"],
                    "text": row["claim"],
                    "support_level": row["support_level"],
                    "evidence_ids": list(
                        row["partial_support_chunk_ids"]
                        if row["support_level"] == "partial"
                        else row["analog_support_chunk_ids"]
                    ),
                }
                for row in weak_claims[:6]
                if row["claim"] and row["claim_id"] and (
                    row["partial_support_chunk_ids"] or row["analog_support_chunk_ids"]
                )
            ],
            "unresolved_gaps": list((payload["replanner_long_memory"] or {}).get("global_critical_gaps") or [])[:8],
            "conflicts_and_limitations": list((payload["replanner_long_memory"] or {}).get("unresolved_conflicts") or [])[:8],
            "safety_boundaries": [row["title"] for row in payload["safety"]["issues"]],
            "prohibited_attributions": [
                (
                    f"不得把仅由部分或类比证据支持的论断写成目标实体的已证实事实：{row['claim']}"
                    + (
                        f"；不得扩展至这些不匹配或未报告维度：{', '.join(row.get('unsupported_dimensions') or [])}"
                        if row.get("unsupported_dimensions") else ""
                    )
                )
                for row in weak_claims[:6] if row["claim"]
            ],
            "evidence_ids": list(dict.fromkeys(x for row in payload["current_claims"] for x in [*(row["direct_support_chunk_ids"] or []), *(row["partial_support_chunk_ids"] or []), *(row["analog_support_chunk_ids"] or []), *(row["contradicting_chunk_ids"] or [])]))[:30],
        }
        if not self.use_llm or self.llm_client is None:
            return fallback
        try:
            batch_size = configured_batch_size("ANSWER_CONTEXT_BATCH_SIZE", 4)

            def summarize_part(part: list[Dict[str, Any]], part_number: int, part_count: int) -> Dict[str, Any]:
                part_payload = dict(payload)
                part_payload["current_claims"] = part
                part_payload["all_current_claim_index"] = [
                    {
                        "claim_id": row.get("claim_id"),
                        "support_level": row.get("support_level"),
                        "claim": str(row.get("claim") or "")[:200],
                    }
                    for row in payload["current_claims"]
                ]
                part_payload["structured_output_part"] = {
                    "part_number": part_number,
                    "part_count": part_count,
                    "instruction": (
                        "只为本部分 current_claims 生成 findings；仍返回完整 Schema。"
                        "全局限制字段只写本部分能确定的内容，不要续写其他部分。"
                    ),
                }
                result = self.llm_client.call_function(
                    system=self.SYSTEM_PROMPT,
                    user=json.dumps(part_payload, ensure_ascii=False, separators=(",", ":")),
                    function_name="summarize_answer_context",
                    description="Submit a complete bounded context object for this claim part.",
                    parameters=ANSWER_CONTEXT_SCHEMA,
                    temperature=0.1,
                    max_output_tokens=5000,
                )
                missing = [key for key in ANSWER_CONTEXT_SCHEMA["required"] if key not in result]
                if missing:
                    raise ValueError(f"missing required structured fields: {missing}")
                return result

            parts, recovered_from_truncation = run_adaptive(
                payload["current_claims"] or [{}],
                batch_size=batch_size,
                call_part=summarize_part,
            )
            string_fields = (
                "case_context",
                "unresolved_gaps",
                "conflicts_and_limitations",
                "safety_boundaries",
                "prohibited_attributions",
                "evidence_ids",
            )
            result: Dict[str, Any] = {
                key: unique_strings([
                    value for part in parts for value in (part.get(key) or [])
                ])
                for key in string_fields
            }
            for key in ("key_findings", "partial_or_analog_findings"):
                rows = []
                seen = set()
                for part in parts:
                    for row in part.get(key) or []:
                        if not isinstance(row, dict):
                            continue
                        signature = json.dumps(row, ensure_ascii=False, sort_keys=True)
                        if signature in seen:
                            continue
                        seen.add(signature)
                        rows.append(row)
                result[key] = rows
            result["structured_output_recovery"] = recovery_trace(
                input_items=len(payload["current_claims"]),
                part_count=len(parts),
                batch_size=batch_size,
                initial_attempt_failed=recovered_from_truncation,
            )
            # Validate stable IDs and evidence bindings, while allowing faithful
            # paraphrases of a golden claim.
            allowed = {
                row["claim_id"]: set(row["direct_support_chunk_ids"] or [])
                for row in direct_claims if row["claim_id"]
            }
            validated = []
            invalid_direct_texts = []
            for finding in result.get("key_findings") or []:
                if not isinstance(finding, dict):
                    invalid_direct_texts.append(str(finding))
                    continue
                claim_id = str(finding.get("claim_id") or "")
                evidence_ids = [str(x) for x in finding.get("evidence_ids") or []]
                if claim_id not in allowed or not evidence_ids or not set(evidence_ids).issubset(allowed[claim_id]):
                    invalid_direct_texts.append(str(finding.get("text") or ""))
                    continue
                validated.append({"claim_id": claim_id, "text": str(finding.get("text") or ""), "evidence_ids": evidence_ids})
            result["key_findings"] = validated
            weak_allowed = {}
            for row in weak_claims:
                claim_id = row.get("claim_id")
                level = row.get("support_level")
                ids = row.get("partial_support_chunk_ids") if level == "partial" else row.get("analog_support_chunk_ids")
                if claim_id and level in {"partial", "analog"}:
                    weak_allowed[(claim_id, level)] = {
                        "evidence_ids": set(ids or []),
                        "source_text": str(row.get("claim") or ""),
                    }
            validated_weak = []
            for finding in result.get("partial_or_analog_findings") or []:
                if not isinstance(finding, dict):
                    continue
                claim_id = str(finding.get("claim_id") or "")
                level = str(finding.get("support_level") or "")
                evidence_ids = [str(x) for x in finding.get("evidence_ids") or []]
                binding = weak_allowed.get((claim_id, level)) or {}
                allowed_ids = binding.get("evidence_ids") or set()
                if not evidence_ids or not allowed_ids or not set(evidence_ids).issubset(allowed_ids):
                    continue
                validated_weak.append({
                    "claim_id": claim_id,
                    # Never trust generated prose for weak evidence. A valid
                    # claim/evidence binding only authorizes retaining the
                    # original bounded claim, not creating a new description.
                    "text": str(binding.get("source_text") or ""),
                    "support_level": level,
                    "evidence_ids": evidence_ids,
                })
            result["partial_or_analog_findings"] = validated_weak
            if invalid_direct_texts:
                result["conflicts_and_limitations"] = list(dict.fromkeys([
                    *(result.get("conflicts_and_limitations") or []),
                    *(f"未通过直接证据绑定校验：{text}" for text in invalid_direct_texts if text),
                ]))
            result["prohibited_attributions"] = list(dict.fromkeys([*(result.get("prohibited_attributions") or []), *fallback["prohibited_attributions"]]))
            return result
        except Exception as exc:
            if os.environ.get("STRICT_LLM_PIPELINE") == "1":
                raise
            logger.warning("Answer context summarization failed; using compact deterministic context: %s", exc)
            return fallback
