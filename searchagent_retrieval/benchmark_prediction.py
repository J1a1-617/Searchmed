from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient

logger = logging.getLogger(__name__)

_BENCHMARK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_benefit": {
            "type": "string",
            "description": "Predicted net clinical benefit category across efficacy and harm.",
            "enum": ["明显获益", "有限获益或稳定", "无明显获益", "进展或有害"],
        },
        "body_lesion_recist": {"type": "string", "description": "Predicted extracranial RECIST response; NA when not assessable.", "enum": ["CR", "PR", "SD", "PD", "NA"]},
        "cns_lm_recist": {"type": "string", "description": "Predicted CNS/leptomeningeal response; NA when not assessable.", "enum": ["CR", "PR", "SD", "PD", "NA"]},
        "csf_trajectory": {"type": "string", "description": "Predicted cerebrospinal-fluid tumor marker/cytology trajectory.", "enum": ["转阴", "下降", "稳定", "上升", "未评估"]},
        "symptom_trajectory": {"type": "string", "description": "Predicted direction of clinically relevant symptoms.", "enum": ["明显改善", "部分改善", "无变化", "加重"]},
        "toxicity": {
            "type": "object",
            "properties": {
                "max_grade": {"type": "integer", "minimum": 0, "maximum": 5},
                "event": {"type": "string"},
                "requires_dose_modification": {"type": "boolean"},
            },
            "required": ["max_grade", "event", "requires_dose_modification"],
            "additionalProperties": False,
        },
        "confidence": {"type": "string", "description": "Confidence calibrated to evidence directness, consistency, and missing data.", "enum": ["高", "中", "低"]},
        "rationale": {"type": "array", "description": "Concise evidence-linked reasons; do not describe workflow.", "items": {"type": "string"}},
        "cited_evidence": {
            "type": "array",
            "description": "Only verifiable retrieved evidence available before the benchmark cutoff.",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "content": {"type": "string"},
                    "pub_date": {"type": "string"},
                },
                "required": ["type", "content", "pub_date"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "overall_benefit",
        "body_lesion_recist",
        "cns_lm_recist",
        "csf_trajectory",
        "symptom_trajectory",
        "toxicity",
        "confidence",
        "rationale",
        "cited_evidence",
    ],
    "additionalProperties": False,
}


class BenchmarkPredictionGenerator:
    """Convert agent loop evidence into the benchmark JSON schema."""

    SYSTEM_PROMPT = (
        "你是 Predictive Clinical Benchmark 的输出适配器。"
        "你只能基于输入中的病例信息、检索到的证据、答案记忆和安全门结果，"
        "输出严格符合指定 JSON Schema 的单个 JSON 对象。"
        "禁止输出任何额外解释、Markdown、编号或前后缀文本。"
        "cited_evidence只能改写输入evidence中的真实正文与日期；输入没有可核验证据时必须返回空数组。"
        "不要编造不存在的随访结局；无法判断的维度必须选择 schema 允许的保守值。"
        "只有标记为直接支持的 clinical_claims 和 direct evidence 可作为目标药物/方案的已证实事实；"
        "bounded_evidence 是部分匹配或类比证据，可以在明确保留药物、方案、人群或结局差异的前提下参与不确定预测，"
        "但不得升级成目标实体的直接事实；counter evidence 只能用于反证或表达不确定性。必须遵守 prohibited_attributions。"
    )

    SKILL_ROUTING_PROMPT = (
        "Skills 是可选的专项能力，不是普通临床预测的必经步骤。"
        "只能从候选 Skill 状态中选择；候选为空时必须直接提交结果。"
        "只有 Skill 的作用位置与当前 Generate 阶段匹配，且它能提供当前输入中尚未执行的专项处理时，才调用 load_skill。"
        "仅因为任务要求判断获益/不获益、没有检索到直接证据、或既往方案下曾进展，不构成加载 Skill 的理由。"
    )

    def __init__(self, llm_client: Optional[LLMClient] = None, use_llm: bool = True) -> None:
        self.llm_client = llm_client
        self.use_llm = use_llm
        self.last_generation_source = "not_run"
        self.last_error: Optional[str] = None
        self.last_input_payload: Optional[Dict[str, Any]] = None

    @staticmethod
    def _compact_evidence(
        loop_result: Dict[str, Any],
        limit: int = 8,
        allowed_direct_ids: Optional[set[str]] = None,
    ) -> List[Dict[str, str]]:
        """Return information-bearing evidence text, never workflow metadata."""
        answer_memory = loop_result.get("answer_memory") or {}
        evidence_by_id = answer_memory.get("evidence_by_id") or {}
        ordered_ids: List[tuple[str, str, str]] = []
        seen_ids = set()

        def add_id(value: Any, role: str, claim_scope: str = "") -> None:
            evidence_id = str(value or "").strip()
            if evidence_id and evidence_id not in seen_ids:
                seen_ids.add(evidence_id)
                ordered_ids.append((evidence_id, role, claim_scope))

        for claim in answer_memory.get("claims") or []:
            if not isinstance(claim, dict) or claim.get("status") == "rejected_by_safety_gate":
                continue
            direct_ids = claim.get("direct_support_chunk_ids")
            if direct_ids is None:  # compatibility with pre-migration sessions
                direct_ids = claim.get("supporting_chunk_ids") or []
            for evidence_id in direct_ids or []:
                if allowed_direct_ids is None or str(evidence_id) in allowed_direct_ids:
                    add_id(evidence_id, "direct", str(claim.get("claim_scope") or ""))
            for evidence_id in claim.get("contradicting_chunk_ids") or []:
                add_id(evidence_id, "counter", str(claim.get("claim_scope") or ""))

        rows: List[Dict[str, str]] = []
        seen_text = set()
        for evidence_id, role, claim_scope in ordered_ids:
            evidence = evidence_by_id.get(evidence_id)
            if not isinstance(evidence, dict):
                continue
            text = " ".join(str(evidence.get("text") or "").split())
            if not text:
                continue
            text_key = text.lower()
            if text_key in seen_text:
                continue
            seen_text.add(text_key)
            citation = evidence.get("citation_json")
            citation = citation if isinstance(citation, dict) else {}
            rows.append({
                "id": evidence_id,
                "evidence_role": role,
                "claim_scope": claim_scope,
                "type": str(
                    evidence.get("evidence_level")
                    or evidence.get("chunk_type")
                    or "case"
                ),
                "content": text[:1600],
                "pub_date": str(
                    evidence.get("pub_date")
                    or evidence.get("publication_date")
                    or citation.get("pub_date")
                    or citation.get("publication_date")
                    or citation.get("date")
                    or "NA"
                ),
            })
            if len(rows) >= limit:
                break
        return rows[:limit]

    @staticmethod
    def _compact_bounded_evidence(
        loop_result: Dict[str, Any],
        limit: int = 10,
        allowed_ids: Optional[set[str]] = None,
    ) -> List[Dict[str, str]]:
        """Expose partial/analog evidence with an explicit non-direct boundary."""
        memory = loop_result.get("answer_memory") or {}
        evidence_by_id = memory.get("evidence_by_id") or {}
        rows: List[Dict[str, str]] = []
        seen = set()
        for claim in memory.get("claims") or []:
            if not isinstance(claim, dict) or claim.get("status") == "rejected_by_safety_gate":
                continue
            level = str(claim.get("support_level") or "")
            if level not in {"partial", "analog"}:
                continue
            ids = claim.get("partial_support_chunk_ids") if level == "partial" else claim.get("analog_support_chunk_ids")
            for evidence_id in ids or []:
                evidence_id = str(evidence_id or "").strip()
                if allowed_ids is not None and evidence_id not in allowed_ids:
                    continue
                if not evidence_id or evidence_id in seen:
                    continue
                evidence = evidence_by_id.get(evidence_id)
                if not isinstance(evidence, dict):
                    continue
                content = " ".join(str(evidence.get("text") or "").split())
                if not content:
                    continue
                seen.add(evidence_id)
                citation = evidence.get("citation_json")
                citation = citation if isinstance(citation, dict) else {}
                rows.append({
                    "id": evidence_id,
                    "support_level": level,
                    "claim_scope": str(claim.get("claim_scope") or ""),
                    "subject_entity": str(claim.get("subject_entity") or ""),
                    "content": content[:1600],
                    "pub_date": str(evidence.get("pub_date") or evidence.get("publication_date") or citation.get("pub_date") or citation.get("publication_date") or citation.get("date") or "NA"),
                })
                if len(rows) >= limit:
                    return rows
        return rows

    @staticmethod
    def _information_context(answer_context_summary: Dict[str, Any]) -> Dict[str, Any]:
        allowed = (
            "case_context",
            "key_findings",
            "partial_or_analog_findings",
            "unresolved_gaps",
            "conflicts_and_limitations",
            "safety_boundaries",
            "prohibited_attributions",
        )
        result: Dict[str, Any] = {}
        for key in allowed:
            values = answer_context_summary.get(key) or []
            if key in {"key_findings", "partial_or_analog_findings"}:
                result[key] = [dict(value) if isinstance(value, dict) else str(value) for value in values]
            else:
                result[key] = [str(value) for value in values if str(value).strip()]
        return result

    @staticmethod
    def _clinical_claims(
        loop_result: Dict[str, Any],
        allowed_claim_ids: Optional[set[str]] = None,
    ) -> List[str]:
        claims: List[str] = []
        for row in (loop_result.get("answer_memory") or {}).get("claims") or []:
            if not isinstance(row, dict) or row.get("status") == "rejected_by_safety_gate":
                continue
            if allowed_claim_ids is not None and str(row.get("claim_id") or "") not in allowed_claim_ids:
                continue
            direct_ids = row.get("direct_support_chunk_ids")
            if direct_ids is None:
                direct_ids = row.get("supporting_chunk_ids") or []
            if not direct_ids or str(row.get("support_level") or "direct") != "direct":
                continue
            claim = str(row.get("safe_claim") or row.get("claim") or "").strip()
            if claim and claim not in claims:
                claims.append(claim[:500])
        return claims[:10]

    @staticmethod
    def _fallback(loop_result: Dict[str, Any], safety_result: Dict[str, Any]) -> Dict[str, Any]:
        risk_level = str(safety_result.get("risk_level") or "").lower()
        issues = [str(item.get("title") or "") for item in (safety_result.get("issues") or []) if isinstance(item, dict)]
        claims = loop_result.get("answer_memory", {}).get("claims") or []
        supported = [c for c in claims if isinstance(c, dict) and c.get("status") in {"supported", "contested"}]
        overall_benefit = "有限获益或稳定"
        if "critical" in risk_level or "high" in risk_level or any("progress" in x.lower() or "risk" in x.lower() for x in issues):
            overall_benefit = "进展或有害"
        elif supported:
            overall_benefit = "明显获益"
        return {
            "overall_benefit": overall_benefit,
            "body_lesion_recist": "NA",
            "cns_lm_recist": "NA",
            "csf_trajectory": "未评估",
            "symptom_trajectory": "无变化",
            "toxicity": {"max_grade": 0, "event": "无", "requires_dose_modification": False},
            "confidence": "低",
            "rationale": [
                "规则回退：未能获得可靠的结构化预测输出。",
                "仅依据当前检索证据与安全门结果进行保守填充。",
            ],
            "cited_evidence": [],
        }

    def generate(
        self,
        *,
        benchmark_prompt: str,
        query: str,
        loop_result: Dict[str, Any],
        safety_result: Dict[str, Any],
        answer_context_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.use_llm or self.llm_client is None:
            self.last_generation_source = "fallback"
            self.last_error = "LLM benchmark prediction disabled or unavailable"
            return self._fallback(loop_result, safety_result)

        validated_direct_ids = {
            str(evidence_id)
            for finding in answer_context_summary.get("key_findings") or []
            if isinstance(finding, dict)
            for evidence_id in finding.get("evidence_ids") or []
        }
        validated_direct_claim_ids = {
            str(finding.get("claim_id") or "")
            for finding in answer_context_summary.get("key_findings") or []
            if isinstance(finding, dict) and str(finding.get("claim_id") or "")
        }
        validated_bounded_ids = {
            str(evidence_id)
            for finding in answer_context_summary.get("partial_or_analog_findings") or []
            if isinstance(finding, dict)
            for evidence_id in finding.get("evidence_ids") or []
        }
        payload = {
            "case_information": benchmark_prompt,
            "answer_information": self._information_context(answer_context_summary),
            "safety_information": {
                "risk_level": safety_result.get("risk_level"),
                "issues": [
                    str(row.get("title") or "")
                    for row in (safety_result.get("issues") or [])
                    if isinstance(row, dict) and str(row.get("title") or "").strip()
                ][:8],
                "recommended_actions": [
                    str(value)
                    for value in (safety_result.get("recommended_actions") or [])
                    if str(value).strip()
                ][:8],
            },
            "clinical_claims": self._clinical_claims(
                loop_result,
                allowed_claim_ids=validated_direct_claim_ids,
            ),
            "evidence": self._compact_evidence(
                loop_result,
                allowed_direct_ids=validated_direct_ids,
            ),
            "bounded_evidence": self._compact_bounded_evidence(
                loop_result,
                allowed_ids=validated_bounded_ids,
            ),
        }
        if loop_result.get("skill_runtime"):
            payload["skill_runtime"] = loop_result["skill_runtime"]
        self.last_input_payload = payload
        user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        try:
            load_skill_schema = {
                "type": "object",
                "properties": {"skill_id": {"type": "string"}},
                "required": ["skill_id"],
                "additionalProperties": False,
            }
            skill_state = payload.get("skill_runtime") or {}
            candidate_skills = [
                row for row in skill_state.get("candidate_skills") or []
                if isinstance(row, dict) and str(row.get("skill_id") or "").strip()
            ]
            allowed_skill_ids = {str(row["skill_id"]) for row in candidate_skills}
            loaded_skill_ids: List[str] = []

            def execute_skill(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
                if name != "load_skill":
                    return {"ok": False, "error": "unknown_skill_tool"}
                skill_id = str(arguments.get("skill_id") or "").strip()
                if not skill_id or "/" in skill_id or "\\" in skill_id or ".." in skill_id:
                    return {"ok": False, "error": "invalid_skill_id"}
                if skill_id not in allowed_skill_ids:
                    return {"ok": False, "error": "skill_not_in_candidates", "skill_id": skill_id}
                skill_path = Path(__file__).resolve().parents[1] / "skill_evolution" / "skills" / skill_id / "SKILL.md"
                if not skill_path.is_file():
                    # Remote deployments may keep skills beside the repository.
                    skill_path = Path.cwd() / "skill_evolution" / "skills" / skill_id / "SKILL.md"
                if not skill_path.is_file():
                    return {"ok": False, "error": "skill_not_found", "skill_id": skill_id}
                if skill_id not in loaded_skill_ids:
                    loaded_skill_ids.append(skill_id)
                return {"ok": True, "skill_id": skill_id, "instructions": skill_path.read_text(encoding="utf-8")[:12000]}

            if candidate_skills and hasattr(self.llm_client, "run_function_tool_loop"):
                result = self.llm_client.run_function_tool_loop(
                    system=self.SYSTEM_PROMPT + "\n\n" + self.SKILL_ROUTING_PROMPT + "\n候选 Skill 状态：" + json.dumps(skill_state, ensure_ascii=False),
                    user=user_prompt,
                    tools=[
                        {"type": "function", "name": "load_skill", "description": "Load a relevant reusable Skill before answering.", "strict": True, "parameters": load_skill_schema},
                        {"type": "function", "name": "submit_predictive_benchmark_result", "description": "Submit a single predictive clinical benchmark JSON result.", "strict": True, "parameters": _BENCHMARK_SCHEMA},
                    ],
                    execute_tool=execute_skill,
                    terminal_tool_name="submit_predictive_benchmark_result",
                    max_turns=3,
                    temperature=0.0,
                    max_output_tokens=8000,
                )["report"]
            else:
                # With no stage-eligible candidates, skip the Skill tool loop.
                # This both prevents accidental mounting and saves one LLM call.
                result = self.llm_client.call_function(
                    system=self.SYSTEM_PROMPT,
                    user=user_prompt,
                    function_name="submit_predictive_benchmark_result",
                    description="Submit a single predictive clinical benchmark JSON result.",
                    parameters=_BENCHMARK_SCHEMA,
                    temperature=0.0,
                    max_output_tokens=8000,
                )
            if skill_state:
                skill_state["selected_skill_ids"] = list(loaded_skill_ids)
                skill_state["loaded_skill_ids"] = list(loaded_skill_ids)
                skill_state["validation"] = "passed"
            self.last_generation_source = "llm"
            self.last_error = None
            return result
        except Exception as exc:
            if os.environ.get("STRICT_LLM_PIPELINE") == "1":
                raise
            logger.warning("Benchmark prediction generation failed; using fallback: %s", exc)
            self.last_generation_source = "fallback"
            self.last_error = f"{type(exc).__name__}: {exc}"
            return self._fallback(loop_result, safety_result)
