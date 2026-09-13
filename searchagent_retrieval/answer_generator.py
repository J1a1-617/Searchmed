from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient

logger = logging.getLogger(__name__)

ANSWER_SYSTEM_PROMPT = (
    "你是肿瘤临床检索助手。基于检索到的病例证据生成带引用、分层和安全边界的回答。"
    "必须区分病例库证据与模型背景知识，不能给出超出证据范围的处方建议。"
    "引用只能来自输入的 CitationAgent 引用映射；禁止补充、猜测或凭模型记忆生成 PMID、DOI、标题、URL 或原文引句。"
)

ANSWER_USER_TEMPLATE = """医生问题：
{query}

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

部分/类比证据（仅供参考，不得直接归因）：
{weak_evidence}

弱证据剔除声明：
{weak_exclusion_notice}

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

要求：每条证据都必须显示“证据关系等级”（直接支持/部分支持/类比参考/反对或风险/未支持）和来源证据类型；
直接结论和支持证据只能使用“直接支持”证据。部分支持、类比参考、未支持证据不得用于证明目标药物、目标方案或目标时间窗的疗效，
不得把其中的数字改写成目标问题的结果。出现药物、联合方案、患者人群、对照或结局时间窗不一致时，必须在“弱证据剔除声明”中说明该证据已从主结论中剔除；
它只能作为背景参考，必要时可明确写“不能外推”。每条关键论断必须对应 citation_map 中的 chunk_id；引用章节只能逐项复制 citation_map 中存在的来源；
禁止引用模型背景知识；证据不足处明确说明；结尾重申不能替代指南与 MDT。"""


def _truncate(text: str, max_len: int = 280) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_len:
        return normalized
    return f"{normalized[: max_len - 3]}..."


def _format_evidence_rows(rows: List[Dict[str, Any]], limit: int = 6) -> str:
    if not rows:
        return "（无）"
    lines: List[str] = []
    for item in rows[:limit]:
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("chunk_id") or item.get("id") or "unknown"
        pmid = item.get("pmid") or "N/A"
        title = item.get("title") or "N/A"
        source_level = item.get("evidence_level") or "case_report_evidence"
        relation = str(item.get("support_level") or "").lower()
        role = str(item.get("evidence_role") or item.get("overall_role") or "").lower()
        if relation == "direct" or role == "direct_support":
            relation_label = "直接支持"
        elif relation == "partial" or role == "partial_support":
            relation_label = "部分支持"
        elif relation == "analog" or role == "analog_support":
            relation_label = "类比参考"
        elif role in {"counter", "risk"} or (item.get("relevance_signals") or {}).get("is_contradicting"):
            relation_label = "反对/风险"
        elif (item.get("relevance_signals") or {}).get("is_supporting"):
            relation_label = "直接支持"
        else:
            relation_label = "未支持"
        text = _truncate(str(item.get("text") or ""))
        lines.append(f"- [{chunk_id}] 证据关系等级={relation_label}；来源类型={source_level}；PMID={pmid}；标题={title}\n  {text}")
    return "\n".join(lines)


def _weak_exclusion_notice(evidence: Dict[str, List[Dict[str, Any]]]) -> str:
    weak = evidence.get("weak") or []
    if not weak:
        return "无弱证据，未触发剔除。"
    supporting = evidence.get("supporting") or []
    if supporting:
        return "已将上述弱证据从直接结论和支持证据中剔除；仅可作为背景参考，不得直接归因或外推。"
    return "未发现直接支持证据；上述弱证据全部从主结论中剔除，只能用于说明证据缺口，不能据此给出目标疗效或数值。"


class AnswerGenerator:
    def __init__(self, llm_client: Optional[LLMClient] = None, use_llm: bool = True) -> None:
        self.llm_client = llm_client
        self.use_llm = use_llm

    def generate(
        self,
        query: str,
        loop_result: Dict[str, Any],
        safety_result: Dict[str, Any],
    ) -> str:
        allowed_ids, allowed_pmids = self._citation_inventory(loop_result)
        if not allowed_ids:
            return self._generate_fallback(query=query, loop_result=loop_result, safety_result=safety_result)
        if self.use_llm and self.llm_client is not None:
            try:
                answer = self._generate_with_llm(query=query, loop_result=loop_result, safety_result=safety_result)
                self._validate_answer_citations(answer, allowed_ids=allowed_ids, allowed_pmids=allowed_pmids)
                return answer
            except Exception as exc:
                logger.warning("LLM answer generation failed, falling back to template: %s", exc)
        return self._generate_fallback(query=query, loop_result=loop_result, safety_result=safety_result)

    @staticmethod
    def _citation_inventory(loop_result: Dict[str, Any]) -> tuple[set[str], set[str]]:
        chunk_ids: set[str] = set()
        pmids: set[str] = set()
        for row in (loop_result.get("citations") or {}).get("claim_citations") or []:
            for citation in row.get("citations") or []:
                if citation.get("chunk_id"):
                    chunk_ids.add(str(citation["chunk_id"]))
                if citation.get("pmid"):
                    pmids.add(str(citation["pmid"]))
        return chunk_ids, pmids

    @staticmethod
    def _validate_answer_citations(answer: str, allowed_ids: set[str], allowed_pmids: set[str]) -> None:
        mentioned_pmids = set(re.findall(r"PMID\s*[:=]?\s*(\d{5,10})", answer, re.IGNORECASE))
        if mentioned_pmids - allowed_pmids:
            raise ValueError(f"answer contains unverified PMID(s): {sorted(mentioned_pmids - allowed_pmids)}")
        mentioned_chunks = set(re.findall(r"\b\d{5,10}#chunk-\d+\b", answer))
        if mentioned_chunks - allowed_ids:
            raise ValueError("answer contains chunk IDs absent from CitationAgent")
        if re.search(r"\bdoi\s*[:=]|https?://", answer, re.IGNORECASE):
            raise ValueError("answer contains unverified DOI or URL")

    def _collect_evidence(self, loop_result: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        allowed_ids, _ = self._citation_inventory(loop_result)
        state = loop_result.get("state") or {}
        retrieved = state.get("retrieved_evidence") or []
        supporting: List[Dict[str, Any]] = []
        weak: List[Dict[str, Any]] = []
        contradicting: List[Dict[str, Any]] = []

        for item in retrieved:
            if not isinstance(item, dict):
                continue
            if str(item.get("chunk_id") or item.get("id") or "") not in allowed_ids:
                continue
            signals = item.get("relevance_signals") or {}
            if signals.get("is_contradicting") or signals.get("is_safety_risk"):
                contradicting.append(item)
            elif str(item.get("support_level") or "").lower() in {"partial", "analog"} or str(item.get("evidence_role") or "").lower() in {"partial_support", "analog_support"}:
                weak.append(item)
            elif signals.get("is_supporting"):
                supporting.append(item)
            else:
                weak.append(item)

        last_route = loop_result.get("last_route_result") or {}
        assessed = (last_route.get("evidence_layering") or {}).get("assessed_evidence") or []
        if not supporting and isinstance(assessed, list):
            for item in assessed:
                if str(item.get("chunk_id") or item.get("id") or "") not in allowed_ids:
                    continue
                signals = item.get("relevance_signals") or {}
                if str(item.get("support_level") or "").lower() in {"partial", "analog"} or str(item.get("evidence_role") or "").lower() in {"partial_support", "analog_support"}:
                    weak.append(item)
                elif signals.get("is_supporting"):
                    supporting.append(item)
                if signals.get("is_contradicting") or signals.get("is_safety_risk"):
                    contradicting.append(item)

        return {"supporting": supporting, "weak": weak, "contradicting": contradicting}

    def _generate_with_llm(
        self,
        query: str,
        loop_result: Dict[str, Any],
        safety_result: Dict[str, Any],
    ) -> str:
        state = loop_result.get("state") or {}
        evidence = self._collect_evidence(loop_result)
        last_route = loop_result.get("last_route_result") or {}
        layer_distribution = (last_route.get("evidence_layering") or {}).get("layer_distribution") or []

        issues = safety_result.get("issues") or []
        issue_lines = [
            f"{item.get('severity', 'unknown')}: {item.get('title', '')}"
            for item in issues[:5]
            if isinstance(item, dict)
        ]

        answer_context = loop_result.get("answer_context_summary") or {}
        prior_context_text = json.dumps(answer_context, ensure_ascii=False, separators=(",", ":")) if answer_context else "（无）"

        claims = []
        for claim in (loop_result.get("answer_memory") or {}).get("claims") or []:
            if claim.get("status") == "rejected_by_safety_gate" or claim.get("safety_status") == "reject":
                continue
            row = {
                "claim_id": claim.get("claim_id"),
                "claim": _truncate(str(claim.get("safe_claim") or claim.get("claim") or ""), 500),
                "status": claim.get("status"),
                "confidence": claim.get("confidence"),
                "supporting_chunk_ids": (claim.get("supporting_chunk_ids") or [])[:3],
                "contradicting_chunk_ids": (claim.get("contradicting_chunk_ids") or [])[:3],
                "safety_boundary": _truncate(str(claim.get("required_revision") or ""), 280),
            }
            claims.append(row)
        claims.sort(
            key=lambda row: (
                str(row.get("status") or "") in {"contested", "safety_limited"},
                float(row.get("confidence") or 0.0),
            ),
            reverse=True,
        )
        claims = claims[:10]
        selected_claim_ids = {str(row.get("claim_id") or "") for row in claims}
        compact_citations = []
        for row in (loop_result.get("citations") or {}).get("claim_citations") or []:
            if str(row.get("claim_id") or "") not in selected_claim_ids:
                continue
            compact_citations.append({
                "claim_id": row.get("claim_id"),
                "citations": [{
                    "chunk_id": citation.get("chunk_id"),
                    "pmid": citation.get("pmid"),
                    "title": _truncate(str(citation.get("title") or ""), 180),
                    "source_file": _truncate(str(citation.get("source_file") or ""), 220),
                } for citation in (row.get("citations") or [])[:2]],
            })
        user_prompt = ANSWER_USER_TEMPLATE.format(
            query=query,
            prior_context=prior_context_text,
            answer_claims=json.dumps(claims, ensure_ascii=False, separators=(",", ":")),
            citation_map=json.dumps(compact_citations, ensure_ascii=False, separators=(",", ":")),
            layer_distribution=json.dumps(layer_distribution, ensure_ascii=False, indent=2),
            risk_level=safety_result.get("risk_level", "unknown"),
            safety_issues="\n".join(f"- {line}" for line in issue_lines) or "（无）",
            recommended_actions="\n".join(f"- {item}" for item in (safety_result.get("recommended_actions") or [])),
            supporting_evidence=_format_evidence_rows(evidence["supporting"]),
            weak_evidence=_format_evidence_rows(evidence["weak"]),
            weak_exclusion_notice=_weak_exclusion_notice(evidence),
            counter_evidence=_format_evidence_rows(evidence["contradicting"]),
        )
        return self.llm_client.chat(
            system=ANSWER_SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.3,
            max_output_tokens=5000,
        )

    def _generate_fallback(
        self,
        query: str,
        loop_result: Dict[str, Any],
        safety_result: Dict[str, Any],
    ) -> str:
        state = loop_result.get("state") or {}
        evidence = self._collect_evidence(loop_result)
        loop_steps = loop_result.get("loop_steps") or []
        verdict = loop_steps[-1].get("citation_verdict", "insufficient_evidence") if loop_steps else "insufficient_evidence"

        sections = [
            "## 1. 直接结论",
            f"基于当前病例库检索，审核结论为 `{verdict}`。以下内容为规则模板汇总，需结合临床判断。",
            "",
            "## 2. 关键病例证据",
            _format_evidence_rows(evidence["supporting"], limit=4),
            "",
            "## 3. 支持证据",
            _format_evidence_rows(evidence["supporting"]),
            "",
            "## 4. 部分/类比证据（仅供参考）",
            _format_evidence_rows(evidence["weak"]),
            "",
            "## 5. 弱证据剔除声明",
            _weak_exclusion_notice(evidence),
            "",
            "## 6. 反证/风险/毒性",
            _format_evidence_rows(evidence["contradicting"]),
            "",
            "## 7. 证据可信度说明",
            f"- 多跳轮次: {len(loop_steps)}",
            f"- 停止条件: {state.get('stop_condition')}",
            "",
            "## 8. 临床安全边界",
            f"- 风险等级: {safety_result.get('risk_level', 'unknown')}",
            f"- 安全边界: {safety_result.get('safety_boundary', '结果仅用于检索辅助，不能替代指南与 MDT。')}",
            "",
            "## 9. 引用（PMID/标题/片段）",
        ]

        for item in (evidence["supporting"] + evidence["weak"] + evidence["contradicting"])[:8]:
            chunk_id = item.get("chunk_id") or item.get("id") or "unknown"
            pmid = item.get("pmid") or "N/A"
            title = item.get("title") or "N/A"
            snippet = _truncate(str(item.get("text") or ""), 160)
            sections.append(f"- {chunk_id} | PMID={pmid} | {title}\n  {snippet}")

        citation_rows = (loop_result.get("citations") or {}).get("claim_citations") or []
        if citation_rows:
            sections.extend(["", "### 可追溯来源"])
            for row in citation_rows:
                for citation in row.get("citations") or []:
                    sections.append(
                        f"- {row.get('claim_id')} → {citation.get('chunk_id')} | "
                        f"source_file={citation.get('source_file') or 'N/A'}"
                    )

        sections.append("")
        sections.append(f"> 原始问题: {query}")
        return "\n".join(sections)
