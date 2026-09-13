from __future__ import annotations

import re
import os
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import LLMClient
from .tools import SearchHit

RERANK_SYSTEM_PROMPT = (
    "你是肿瘤临床证据分类器。只根据候选文本判断证据类型及七个匹配维度，"
    "不要计算总分或排序。论文题名、DOI、PMID和参考文献列表属于citation_pointer，"
    "不能作为疗效、安全性或结局证据。必须调用submit_reranking。"
)

RERANK_USER_TEMPLATE = """医生问题：
{query}

跨批次固定校准锚点（不是候选，不要输出它们）：
A. 目标患者、药物/方案、时间窗和结局全部明确匹配的原始结果：所有match=3，match_status=exact，hard_mismatches=[]。
B. 人群和结局相关，但目标单药而当期为抗肿瘤联合方案：regimen_match=1、attribution最多1，match_status=hard_mismatch，hard_mismatches=[wrong_regimen]。
C. 只报告目标时间窗之外的晚期进展/耐药：timepoint_match=0，match_status=hard_mismatch，hard_mismatches=[wrong_time_window]；不能当早期反证。
D. 只有标题/参考文献指针：全部match=0，match_status=hard_mismatch，hard_mismatches=[non_result_text]。

候选证据（id 与摘要）：
{candidates}

逐条分类并评分：
{{
  "rankings": [
    {{"id": "<id>", "evidence_type": "<类型>", "match_status": "exact|partial|hard_mismatch", "hard_mismatches": [], "population_match": 0-3, "intervention_match": 0-3, "regimen_match": 0-3, "timepoint_match": 0-3, "outcome_match": 0-3, "directness": 0-3, "attribution": 0-3, "reason": "<不超过40字>"}}
  ]
}}

要求：
1. 类型只能是 primary_result、case_result、review_result、background、citation_pointer、irrelevant。
2. 七个维度均用整数：0=完全不匹配或完全缺失，1=相关但未覆盖指定细节，2=覆盖部分指定要素，3=明确且完整匹配。
3. outcome_match只看正文实际报告的结局，标题不算。正文报告ORR/PFS/OS等相关临床结局、但缺指定时间点/影像/症状/CSF时应为1，不得记0；覆盖部分指定结局为2，完整覆盖为3。
4. regimen_match单独衡量方案结构。问题要求单药而候选患者当期接受两种或更多抗肿瘤药时最多1分；目标联合方案缺失任一核心药时最多1分。腰池引流、抗感染、对症支持不算联合抗肿瘤方案；仅在背景或既往/后续治疗中提到联合也不惩罚当期单药。
5. timepoint_match只衡量结局时点与问题目标窗口。例如问题问8–12周：12周PR是早期直接支持；11个月后获得性耐药/进展只描述持久性，不是早期疗效反证。候选同时报告早期缓解与晚期进展时，应按早期时点评分，reason必须分开两个时点。只有在目标时间窗内明确PD/无改善，才能作为该时窗的反证。
6. directness衡量正文是否直接报告结果；attribution衡量结局能否归因于目标干预。目标单药与当期抗肿瘤联合方案不匹配时，attribution最多1分。
7. 参考文献、仅题名/作者/期刊/DOI/PMID的内容必须标citation_pointer，各维度填0。
8. 覆盖全部id且不重复；保持输入顺序，不排序；reason只写最关键的命中或缺失，不超过40字。
9. 不使用候选之外的知识，不参考source、hybrid_score或候选顺序。"""

RERANK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"rankings": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "evidence_type": {
                "type": "string",
                "enum": ["primary_result", "case_result", "review_result", "background", "citation_pointer", "irrelevant"],
            },
            "match_status": {"type": "string", "enum": ["exact", "partial", "hard_mismatch"]},
            "hard_mismatches": {"type": "array", "items": {"type": "string", "enum": [
                "wrong_drug", "wrong_regimen", "wrong_population", "wrong_time_window",
                "wrong_outcome", "non_result_text", "attribution_impossible",
            ]}},
            "population_match": {"type": "integer", "minimum": 0, "maximum": 3},
            "intervention_match": {"type": "integer", "minimum": 0, "maximum": 3},
            "regimen_match": {"type": "integer", "minimum": 0, "maximum": 3},
            "timepoint_match": {"type": "integer", "minimum": 0, "maximum": 3},
            "outcome_match": {"type": "integer", "minimum": 0, "maximum": 3},
            "directness": {"type": "integer", "minimum": 0, "maximum": 3},
            "attribution": {"type": "integer", "minimum": 0, "maximum": 3},
            "reason": {"type": "string"},
        },
        "required": ["id", "evidence_type", "match_status", "hard_mismatches", "population_match", "intervention_match", "regimen_match", "timepoint_match", "outcome_match", "directness", "attribution", "reason"],
        "additionalProperties": False,
    }}},
    "required": ["rankings"],
    "additionalProperties": False,
}


_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.+-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]{2,}")
_CLINICAL_SIGNAL_RE = re.compile(
    r"\b(?:orr|pfs|os|dcr|cr|pr|sd|pd|rano|recist|ctcae|csf|response|survival|toxicity|"
    r"adverse|cytology|symptom|improv\w*|progress\w*)\b|"
    r"\d+(?:\.\d+)?\s*(?:%|mg|months?|weeks?|days?|年|月|周|天)",
    re.IGNORECASE,
)


def _looks_like_citation_pointer(text: str) -> bool:
    lowered = text.lower()
    markers = sum(
        marker in lowered
        for marker in ("[doi]", "[pubmed]", "[google scholar]", "pmid:", "doi:")
    )
    return markers >= 2


def _query_terms(query: str) -> List[str]:
    terms = []
    for match in _TERM_RE.findall(query.lower()):
        if len(match) >= 2 and match not in terms:
            terms.append(match)
    return terms


def _query_aware_excerpt(text: str, query: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized

    terms = _query_terms(query)
    lowered = normalized.lower()
    centers: List[Tuple[int, int]] = []
    for term in terms:
        start = 0
        while len(centers) < 40:
            position = lowered.find(term, start)
            if position < 0:
                break
            centers.append((position, 3))
            start = position + max(1, len(term))
    for match in _CLINICAL_SIGNAL_RE.finditer(normalized):
        centers.append((match.start(), 2))

    # Keep a small leading window for document/section context, then select
    # high-value windows around query terms and clinical outcomes.
    lead_chars = min(180, max_chars // 4)
    ranges: List[Tuple[int, int]] = [(0, lead_chars)]
    window_radius = max(90, min(220, max_chars // 5))
    for center, weight in sorted(centers, key=lambda item: (-item[1], item[0])):
        start = max(0, center - window_radius)
        end = min(len(normalized), center + window_radius)
        ranges.append((start, end))

    selected: List[Tuple[int, int]] = []
    used = 0
    for start, end in ranges:
        if any(start >= old_start and end <= old_end for old_start, old_end in selected):
            continue
        available = max_chars - used - (5 if selected else 0)
        if available <= 40:
            break
        end = min(end, start + available)
        if end - start <= 20:
            continue
        selected.append((start, end))
        used += end - start

    if len(selected) == 1 and selected[0][0] == 0:
        return normalized[: max_chars - 3] + "..."
    selected.sort()
    excerpt = " ... ".join(normalized[start:end].strip() for start, end in selected)
    if len(excerpt) > max_chars:
        return excerpt[: max_chars - 3].rstrip() + "..."
    return excerpt


def _estimate_rerank_candidate_chars(hit_id: str, source: str, score: float, excerpt: str) -> int:
    # Rough upper bound for the serialized candidate line in the rerank prompt.
    return len(hit_id) + len(source) + len(f"{score:.4f}") + len(excerpt) + 32


class LLMReranker:
    # At most two independent batches fit the prompt budget. Run them serially
    # to avoid API queueing/rate-limit amplification and make benchmark traces
    # deterministic.
    # Scores remain comparable because the LLM only assigns fixed dimensions;
    # the final scalar score is computed deterministically in this process.
    HYBRID_RERANK_POOL = 16
    BATCH_SIZE = 8
    MAX_REQUESTS = 2
    CANDIDATE_MAX_CHARS = 500
    MAX_RERANK_INPUT_CHARS = 4500
    DIMENSION_WEIGHTS = {
        "population_match": 0.125,
        "intervention_match": 0.125,
        "regimen_match": 0.20,
        "timepoint_match": 0.15,
        "outcome_match": 0.20,
        "directness": 0.10,
        "attribution": 0.10,
    }
    TYPE_CAPS = {
        "primary_result": 1.00,
        "case_result": 0.75,
        "review_result": 0.85,
        "background": 0.35,
        "citation_pointer": 0.15,
        "irrelevant": 0.10,
    }

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def rerank(self, query: str, hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        if not hits:
            return []

        candidates = hits[: self.HYBRID_RERANK_POOL]
        batches = [
            self._fit_single_request(query=query, hits=candidates[start:start + self.BATCH_SIZE])
            for start in range(0, len(candidates), self.BATCH_SIZE)
        ][: self.MAX_REQUESTS]
        batches = [batch for batch in batches if batch]
        if not batches:
            return candidates[:top_k]

        ranked: List[SearchHit] = []
        for batch in batches:
            try:
                ranked.extend(self._rerank_with_llm(query=query, hits=batch))
            except Exception as exc:
                if os.environ.get("STRICT_LLM_PIPELINE") == "1":
                    raise
                ranked.extend(self._fallback_hits(batch, exc))
        ranked.sort(
            key=lambda item: (
                bool(item.metadata.get("rerank_rule_pointer_signal")),
                -item.score,
            )
        )
        return ranked[:top_k]

    def _fit_single_request(self, query: str, hits: List[SearchHit]) -> List[SearchHit]:
        """Keep the highest retrieval-ranked candidates that fit one LLM call."""
        processable: List[SearchHit] = []
        used_chars = 0
        for hit in hits:
            normalized = " ".join(hit.text.split())
            excerpt = _query_aware_excerpt(normalized, query, self.CANDIDATE_MAX_CHARS)
            estimated = _estimate_rerank_candidate_chars(hit.id, hit.source, hit.score, excerpt)
            if processable and used_chars + estimated > self.MAX_RERANK_INPUT_CHARS:
                break
            processable.append(hit)
            used_chars += estimated
        return processable

    @staticmethod
    def _fallback_hits(hits: List[SearchHit], exc: Exception) -> List[SearchHit]:
        fallback: List[SearchHit] = []
        for hit in hits:
            metadata = dict(hit.metadata)
            is_pointer = _looks_like_citation_pointer(hit.text)
            metadata.update({
                "llm_rerank_status": "fallback_error",
                "llm_rerank_error": f"{type(exc).__name__}: {exc}",
                "rerank_rule_pointer_signal": is_pointer,
                "llm_evidence_type": "citation_pointer" if is_pointer else "",
            })
            fallback.append(SearchHit(
                id=hit.id,
                score=min(hit.score, 0.15) if is_pointer else hit.score,
                source=hit.source,
                text=hit.text,
                metadata=metadata,
            ))
        return fallback

    def _rerank_with_llm(self, query: str, hits: List[SearchHit]) -> List[SearchHit]:
        candidate_lines = []
        excerpt_info_by_id: Dict[str, Dict[str, Any]] = {}
        for index, hit in enumerate(hits, start=1):
            normalized = " ".join(hit.text.split())
            char_limit = self.CANDIDATE_MAX_CHARS
            excerpt = _query_aware_excerpt(normalized, query, char_limit)
            excerpt_info_by_id[hit.id] = {
                "rerank_original_chars": len(normalized),
                "rerank_input_chars": len(excerpt),
                "rerank_was_truncated": len(excerpt) < len(normalized),
                "rerank_excerpt_strategy": "full" if len(normalized) <= char_limit else "query_aware",
                "rerank_rule_pointer_signal": _looks_like_citation_pointer(normalized),
            }
            candidate_lines.append(
                f"{index}. id={hit.id}\n"
                f"   source={hit.source} hybrid_score={hit.score:.4f}\n"
                f"   text={excerpt}"
            )

        user_prompt = RERANK_USER_TEMPLATE.format(
            query=query,
            candidates="\n".join(candidate_lines),
        )
        payload = self.llm_client.call_function(
            system=RERANK_SYSTEM_PROMPT,
            user=user_prompt,
            function_name="submit_reranking",
            description="Submit relevance scores for every candidate evidence item.",
            parameters=RERANK_SCHEMA,
            temperature=0.1,
            # Eight candidates with twelve structured fields can exhaust a
            # 6k reasoning+output allowance on GPT-5 before the function call
            # is completed. A larger ceiling is cheaper than repeating the
            # same batch up to three times after incomplete responses.
            max_output_tokens=8000,
        )
        rankings = payload.get("rankings") or []

        score_by_id: Dict[str, float] = {}
        ranking_info_by_id: Dict[str, Dict[str, Any]] = {}
        for item in rankings:
            if not isinstance(item, dict):
                continue
            hit_id = str(item.get("id") or "").strip()
            if not hit_id:
                continue
            evidence_type = str(item.get("evidence_type") or "").strip()
            match_status = str(item.get("match_status") or "partial").strip()
            hard_mismatches = {
                str(value) for value in item.get("hard_mismatches") or [] if str(value)
            }
            reason = str(item.get("reason") or "").strip()
            dimensions = {
                name: self._dimension_value(item.get(name))
                for name in self.DIMENSION_WEIGHTS
            }
            # Backward compatibility for cached/test responses produced by
            # the previous five-dimension schema.
            if "regimen_match" not in item:
                dimensions["regimen_match"] = dimensions["intervention_match"]
            if "timepoint_match" not in item:
                dimensions["timepoint_match"] = dimensions["outcome_match"]
            # Backward-compatible deterministic derivation when an older
            # compatible endpoint omits the newly required hard-match fields.
            if dimensions["intervention_match"] == 0:
                hard_mismatches.add("wrong_drug")
            if dimensions["regimen_match"] == 0:
                hard_mismatches.add("wrong_regimen")
            if dimensions["timepoint_match"] == 0:
                hard_mismatches.add("wrong_time_window")
            if dimensions["outcome_match"] == 0:
                hard_mismatches.add("wrong_outcome")
            rule_pointer = bool(excerpt_info_by_id.get(hit_id, {}).get("rerank_rule_pointer_signal"))
            if rule_pointer:
                evidence_type = "citation_pointer"
                dimensions = {name: 0 for name in dimensions}
                hard_mismatches.add("non_result_text")
                match_status = "hard_mismatch"
            if hard_mismatches:
                match_status = "hard_mismatch"
            score = self._deterministic_score(evidence_type, dimensions, hard_mismatches)
            score_by_id[hit_id] = score
            ranking_info_by_id[hit_id] = {
                "evidence_type": evidence_type,
                "reason": reason[:80],
                "dimensions": dimensions,
                "match_status": match_status,
                "hard_mismatches": sorted(hard_mismatches),
            }

        reranked: List[SearchHit] = []
        seen = set()
        for item in rankings:
            if not isinstance(item, dict):
                continue
            hit_id = str(item.get("id") or "").strip()
            if not hit_id or hit_id in seen:
                continue
            seen.add(hit_id)
            hit = next((candidate for candidate in hits if candidate.id == hit_id), None)
            if hit is None:
                continue
            llm_score = score_by_id.get(hit_id, 0.0)
            reranked.append(self._with_llm_score(
                hit, llm_score, excerpt_info_by_id.get(hit_id), ranking_info_by_id.get(hit_id)
            ))

        for hit in hits:
            if hit.id in seen:
                continue
            reranked.append(self._with_llm_score(
                hit,
                score_by_id.get(hit.id, 0.0),
                excerpt_info_by_id.get(hit.id),
                ranking_info_by_id.get(hit.id),
            ))

        reranked.sort(key=lambda item: item.metadata.get("llm_relevance_score", 0.0), reverse=True)
        return reranked

    @staticmethod
    def _dimension_value(value: Any) -> int:
        try:
            return max(0, min(3, int(value)))
        except (TypeError, ValueError):
            return 0

    def _deterministic_score(
        self,
        evidence_type: str,
        dimensions: Dict[str, int],
        hard_mismatches: Optional[set[str]] = None,
    ) -> float:
        score = sum(
            self.DIMENSION_WEIGHTS[name] * dimensions.get(name, 0) / 3.0
            for name in self.DIMENSION_WEIGHTS
        )
        cap = self.TYPE_CAPS.get(evidence_type, 0.10)
        if dimensions.get("outcome_match", 0) == 0:
            cap = min(cap, 0.35)
        if dimensions.get("population_match", 0) == 0 or dimensions.get("intervention_match", 0) == 0:
            cap = min(cap, 0.30)
        if dimensions.get("regimen_match", 0) == 0:
            cap = min(cap, 0.30)
        elif dimensions.get("regimen_match", 0) == 1:
            cap = min(cap, 0.55)
        if dimensions.get("timepoint_match", 0) == 0:
            cap = min(cap, 0.35)
        if dimensions.get("attribution", 0) == 0:
            cap = min(cap, 0.60)
        mismatch_caps = {
            "wrong_drug": 0.25,
            "wrong_regimen": 0.55,
            "wrong_population": 0.40,
            "wrong_time_window": 0.35,
            "wrong_outcome": 0.30,
            "non_result_text": 0.15,
            "attribution_impossible": 0.30,
        }
        for mismatch in hard_mismatches or set():
            cap = min(cap, mismatch_caps.get(mismatch, cap))
        return round(max(0.0, min(score, cap)), 4)

    @staticmethod
    def _with_llm_score(
        hit: SearchHit,
        llm_score: float,
        excerpt_info: Optional[Dict[str, Any]] = None,
        ranking_info: Optional[Dict[str, Any]] = None,
    ) -> SearchHit:
        metadata = dict(hit.metadata)
        metadata["llm_relevance_score"] = llm_score
        metadata["llm_rerank_status"] = "completed"
        metadata["hybrid_score"] = hit.score
        metadata.update(excerpt_info or {})
        metadata["llm_evidence_type"] = (ranking_info or {}).get("evidence_type", "")
        metadata["llm_rerank_reason"] = (ranking_info or {}).get("reason", "")
        metadata["llm_rerank_dimensions"] = (ranking_info or {}).get("dimensions", {})
        metadata["llm_match_status"] = (ranking_info or {}).get("match_status", "")
        metadata["llm_hard_mismatches"] = (ranking_info or {}).get("hard_mismatches", [])
        return SearchHit(
            id=hit.id,
            score=llm_score,
            source=hit.source,
            text=hit.text,
            metadata=metadata,
        )
