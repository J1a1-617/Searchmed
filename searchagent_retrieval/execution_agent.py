from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from .llm_client import LLMClientError
from .router import RetrievalRouter, RoutePlan
from .text_utils import retrieval_terms
from .tools import SearchHit

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)

_STRING_ARRAY = {"type": "array", "items": {"type": "string"}}


def _short_string_array(max_items: int) -> Dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string", "maxLength": 160},
        "maxItems": max_items,
    }
_CONSTRAINTS = {
    "type": "object",
    "description": "Confirmed patient facts only; values here are hard filters.",
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

STEP_REPORT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "execution_status": {"type": "string", "description": "Outcome of this atomic retrieval step only.", "enum": ["success", "partial", "no_evidence", "budget_exhausted"]},
        "summary": {"type": "string", "description": "Short factual result without copying evidence text.", "maxLength": 320},
        "accepted_evidence_ids": {**_short_string_array(12), "description": "Only real IDs returned by tools and relevant to the step goal."},
        "rejected_evidence_ids": {**_short_string_array(8), "description": "Real candidate IDs rejected as irrelevant, weak, or contradictory."},
        "queries_attempted": _short_string_array(2),
        "query_database_status": {"type": "string", "description": "Whether this exact query direction has remaining retrieval value.", "enum": ["more_available", "exhausted", "uncertain"]},
        "exhaustion_reason": {"type": "string", "maxLength": 180},
        "recommended_query_change": {"type": "string", "maxLength": 240},
        "goal_evaluation": {"type": "object", "properties": {
            "matched_goal_count": {"type": "integer", "minimum": 0},
            "best_goal_relevance": {"type": "number", "minimum": 0, "maximum": 1},
            "success_criteria_met": {"type": "boolean"},
            "observed_gaps": _short_string_array(4),
            "observed_failures": _short_string_array(4),
        }, "required": ["matched_goal_count", "best_goal_relevance", "success_criteria_met", "observed_gaps", "observed_failures"], "additionalProperties": False},
    },
    "required": ["execution_status", "summary", "accepted_evidence_ids", "rejected_evidence_ids", "queries_attempted", "query_database_status", "exhaustion_reason", "recommended_query_change", "goal_evaluation"],
    "additionalProperties": False,
}

MINIMAL_STEP_REPORT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "execution_status": {"type": "string", "enum": ["success", "partial", "no_evidence", "budget_exhausted"]},
        "accepted_evidence_ids": _short_string_array(12),
        "query_database_status": {"type": "string", "enum": ["more_available", "exhausted", "uncertain"]},
        "matched_goal_count": {"type": "integer", "minimum": 0},
        "success_criteria_met": {"type": "boolean"},
    },
    "required": [
        "execution_status",
        "accepted_evidence_ids",
        "query_database_status",
        "matched_goal_count",
        "success_criteria_met",
    ],
    "additionalProperties": False,
}


def _function_tool(name: str, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "strict": True, "parameters": parameters}}


class RetrievalExecutionAgent:
    """Execute one atomic retrieval step with a constrained real function-call loop."""

    TERMINAL_TOOL = "submit_step_execution_report"
    RECOVERY_TOOL = "submit_minimal_step_execution_report"
    SYSTEM_PROMPT = """你是 Retrieval Execution Agent，只负责当前一个原子检索步骤。正常模式严格两阶段。

第一次调用execute_retrieval_batch：
1. 先读取current_step.goal、rerank_goal和success_criteria，识别本步骤不可丢失的检索核心：癌种、当前干预、特异转移部位、必要的治疗顺序和一个宽泛结局词。主驱动、精确亚组或既往药物只有在success_criteria明确要求时才是硬核心；“优先某亚组”属于rerank偏好，不得成为所有召回query的必需条件。不得把leptomeningeal metastases/LMD/CSF泛化成brain或intracranial，也不得用候选替代药物取代患者实际药物。
2. 严格区分召回与精排：search query负责找到可能相关的病例，rerank_goal负责核对精确时间点、ORR/DCR、剂量和全部success_criteria。不要把6/8/12 weeks、two cycles、ORR、DCR等所有细粒度标准同时塞入query；它们常不在相关chunk中，会显著降低召回。query只保留一个如response/outcome的宽泛结局词。
3. 再根据检索需求选择一到两个真正互补的工具：
   - dense_search：语义相似病例或自然语言表达差异较大时使用。query写成约20–45词的一条自然医学语义句，包含核心概念和治疗时序，不罗列评估指标。
   - bm25_search：精确药名、突变、罕见部位、缩写或结局词决定相关性时使用。query写成约8–16个紧凑关键词，加入语料常见英文名/缩写，例如leptomeningeal metastases、LMD、CSF；不要写完整问题或时间点枚举。
   - structured_search：仅当已确认硬字段可有效缩小病例时使用；只传输入中的confirmed constraints，不得把候选药物、替代药物或推测实体写成硬过滤。
   - hybrid_search：只在一个中等长度query能够同时服务语义和词项检索时使用。它已包含dense、BM25和structured分量，通常不要再与dense_search或bm25_search重复组合；存在罕见精确术语时优先显式选择dense_search+bm25_search。
4. 为每个已选工具分别写最适配的query，不得复制同一通用query。Planner给出的工具和query是本轮检索意图；若planner_metadata.replan_action表示重试，且Replanner在query或decision_rationale中明确放宽/删除某个可选过滤条件，不得仅因原始病例包含该条件就把它重新加入query。工具可以调整，但必须保留Replanner本轮的放宽策略。

程序收到actions后才会使用dense/hybrid query执行Step3定位并回到Step2生成seed candidates，随后并发检索、规则截断、rerank和读取证据。正常模式只调用execute_retrieval_batch一次；程序会根据真实工具结果生成执行报告，不需要你再次判断或提交终止报告。query禁止PubMed字段标签、AND/OR/NOT、括号表达式、注释和多行查询。失败后的下一轮改写由Replanner负责。不得改变Plan Step或扩大未授权范围。扩展模式会暴露专用知识工具，并继续受工具预算约束。"""

    def __init__(self, router: RetrievalRouter, llm_client: Optional["LLMClient"], max_turns: int = 3, max_hits_per_tool_result: int = 12, max_search_calls: int = 2, max_fetch_calls: int = 1, external_knowledge_enabled: bool = True) -> None:
        self.router = router
        self.tools = router.tools
        self.llm_client = llm_client
        self.max_turns = max(2, max_turns)
        self.max_hits_per_tool_result = max(1, max_hits_per_tool_result)
        self.max_search_calls = max(1, max_search_calls)
        self.max_fetch_calls = max(1, max_fetch_calls)
        self.external_knowledge_enabled = bool(external_knowledge_enabled)
        self.data_root = Path(__file__).resolve().parent.parent / "data"
        self._knowledge_cache: Dict[str, List[Dict[str, Any]]] = {}

    @staticmethod
    def _complete_constraints(value: Dict[str, Any]) -> Dict[str, Any]:
        result = {"cancer_type": "", "gene_alterations": [], "drugs": [], "responses": [], "toxicities": [], "metastatic_sites": [], "ddi_terms": []}
        result.update({key: item for key, item in value.items() if key in result})
        return result

    @staticmethod
    def _is_recoverable_terminal_protocol_error(exc: Exception) -> bool:
        message = str(exc)
        return any(fragment in message for fragment in (
            "Tool arguments are not valid JSON",
            "Expected exactly one tool call",
            "hit max_turns before terminal tool",
            "did not produce terminal report",
            "different Azure OpenAI resource",
            "previous_response_not_found",
            "Previous response with id",
        ))

    @staticmethod
    def _expand_minimal_report(payload: Dict[str, Any], queries_attempted: List[str]) -> Dict[str, Any]:
        matched = max(0, int(payload.get("matched_goal_count") or 0))
        return {
            "execution_status": str(payload.get("execution_status") or "partial"),
            "summary": "",
            "accepted_evidence_ids": list(payload.get("accepted_evidence_ids") or [])[:12],
            "rejected_evidence_ids": [],
            "queries_attempted": list(dict.fromkeys(queries_attempted))[-2:],
            "query_database_status": str(payload.get("query_database_status") or "uncertain"),
            "exhaustion_reason": "",
            "recommended_query_change": "",
            "goal_evaluation": {
                "matched_goal_count": matched,
                "best_goal_relevance": 0.0,
                "success_criteria_met": bool(payload.get("success_criteria_met")),
                "observed_gaps": [],
                "observed_failures": ["terminal_report_recovered"],
            },
        }

    @staticmethod
    def _hit_payload(hit: SearchHit) -> Dict[str, Any]:
        return {
            "id": hit.id,
            "score": round(float(hit.score), 6),
            "source": hit.source,
            "event_id": hit.metadata.get("event_id"),
            "chunk_type": hit.metadata.get("chunk_type"),
            "text": " ".join(hit.text.split())[:280],
        }

    @staticmethod
    def _normalize_term(value: Any) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _append_unique_terms(target: List[str], seen: Set[str], values: List[Any]) -> None:
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            target.append(text)

    def _tool_definitions(self, lane: str, expansion: Dict[str, Any]) -> List[Dict[str, Any]]:
        search_tools = [
            _function_tool("structured_search", "Filter cases by confirmed structured clinical fields. Use only literal patient facts; never add alternatives or inferred entities.", {"type": "object", "properties": {"constraints": _CONSTRAINTS, "top_k": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": ["constraints", "top_k"], "additionalProperties": False}),
            _function_tool("dense_search", "Semantic similar-case retrieval for paraphrases and treatment sequences. Use one natural clinical sentence preserving every discriminative concept in the active step.", {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 50}, "spaces": {"type": "array", "items": {"type": "string", "enum": ["case_semantic", "event_semantic", "structured_semantic"]}}}, "required": ["query", "top_k", "spaces"], "additionalProperties": False}),
            _function_tool("bm25_search", "Exact term retrieval for drug names, mutations, rare metastatic sites, abbreviations and outcome terms. Use compact keywords with aliases such as LMD/CSF; no Boolean syntax.", {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": ["query", "top_k"], "additionalProperties": False}),
            _function_tool("hybrid_search", "Single-query combination of structured, dense and BM25 retrieval. Use alone when one query suits all components; do not pair redundantly with dense/BM25 when tool-specific queries are needed.", {"type": "object", "properties": {"query": {"type": "string"}, "constraints": _CONSTRAINTS, "top_k": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": ["query", "constraints", "top_k"], "additionalProperties": False}),
            _function_tool("rerank_candidates", "Rerank already retrieved candidate IDs against the atomic step goal.", {"type": "object", "properties": {"goal": {"type": "string"}, "candidate_ids": _STRING_ARRAY, "top_k": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": ["goal", "candidate_ids", "top_k"], "additionalProperties": False}),
            _function_tool("fetch_evidence", "Fetch complete auditable evidence chunks for retrieved chunk IDs.", {"type": "object", "properties": {"chunk_ids": _STRING_ARRAY, "limit": {"type": "integer", "minimum": 1, "maximum": 30}}, "required": ["chunk_ids", "limit"], "additionalProperties": False}),
            _function_tool("fetch_structured_events", "Fetch original detailed Step3 timeline events after event retrieval; prefer it before opening long Step2 text when structured facts are sufficient.", {"type": "object", "properties": {"event_ids": _STRING_ARRAY, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": ["event_ids", "limit"], "additionalProperties": False}),
        ]
        if self.external_knowledge_enabled and expansion.get("enabled"):
            search_tools.append(_function_tool("expand_query_with_knowledge", "Expand the current query with structured local drug aliases, mutation-drug mappings, PK relations, direct case-level DDI relations, and DDI mechanism terms.", {"type": "object", "properties": {"query": {"type": "string"}, "drug_names": _STRING_ARRAY, "mutation_names": _STRING_ARRAY, "strategies": {"type": "array", "items": {"type": "string", "enum": ["drug_alias", "pk_relation", "ddi_rule", "mutation_drug", "case_ddi"]}}, "top_k_terms": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": ["query", "drug_names", "mutation_names", "strategies", "top_k_terms"], "additionalProperties": False}))
        if self.external_knowledge_enabled and lane == "ddi_safety" and expansion.get("enabled"):
            strategies = set(expansion.get("allowed_strategies") or [])
            if "drug_alias" in strategies:
                search_tools.append(_function_tool("normalize_drug_names", "Resolve generic, brand and alias names for the same drug identity.", {"type": "object", "properties": {"drug_names": _STRING_ARRAY}, "required": ["drug_names"], "additionalProperties": False}))
            if "pk_relation" in strategies:
                search_tools.append(_function_tool("lookup_pk_relations", "Look up drug enzyme, transporter and pharmacokinetic relations.", {"type": "object", "properties": {"drug_names": _STRING_ARRAY}, "required": ["drug_names"], "additionalProperties": False}))
            if "ddi_rule" in strategies:
                search_tools.append(_function_tool("lookup_ddi_rules", "Look up DDI rules for enzymes or transporters.", {"type": "object", "properties": {"enzymes_transporters": _STRING_ARRAY}, "required": ["enzymes_transporters"], "additionalProperties": False}))
            if "case_ddi" in strategies:
                search_tools.append(_function_tool("lookup_case_ddi_relations", "Look up direct case-level DDI relations from the curated case_ddi database.", {"type": "object", "properties": {"drug_names": _STRING_ARRAY, "mutation_names": _STRING_ARRAY, "interaction_types": _STRING_ARRAY, "top_k": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": ["drug_names", "mutation_names", "interaction_types", "top_k"], "additionalProperties": False}))
        if self.external_knowledge_enabled and expansion.get("enabled") and "mutation_drug" in set(expansion.get("allowed_strategies") or []):
            search_tools.append(_function_tool("lookup_mutation_drug_relations", "Look up mutation-to-drug mappings for expansion.", {"type": "object", "properties": {"mutation_names": _STRING_ARRAY}, "required": ["mutation_names"], "additionalProperties": False}))
        search_tools.append(_function_tool(self.TERMINAL_TOOL, "Finish this atomic retrieval step and submit its factual execution report.", STEP_REPORT_SCHEMA))
        return search_tools

    def _two_stage_tool_definitions(self) -> List[Dict[str, Any]]:
        action_schema = {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": "Choose by mechanism: dense=semantic sequence, BM25=exact rare terms/aliases, structured=confirmed fields, hybrid=one-query combined retrieval.",
                    "enum": ["structured_search", "dense_search", "bm25_search", "hybrid_search"],
                },
                "query": {
                    "type": "string",
                    "description": "Recall-oriented tool query written after choosing tool. Preserve core disease/driver/treatment/site concepts but leave exact timing, ORR/DCR and full success criteria to rerank_goal. Dense: 20-45 word clinical sentence; BM25: 8-16 exact terms/aliases; hybrid: one concise compromise. Empty only for structured_search.",
                },
                "constraints": _CONSTRAINTS,
                "top_k": {"type": "integer", "minimum": 1, "maximum": 30},
                "spaces": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["case_semantic", "event_semantic", "structured_semantic"],
                    },
                },
            },
            "required": ["tool", "query", "constraints", "top_k", "spaces"],
            "additionalProperties": False,
        }
        batch_schema = {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "description": "One or two complementary searches. First choose each tool, then independently write the query best suited to that tool; do not copy one generic query across tools.",
                    "minItems": 1,
                    "maxItems": 2,
                    "items": action_schema,
                },
            },
            "required": ["actions"],
            "additionalProperties": False,
        }
        return [
            _function_tool(
                "execute_retrieval_batch",
                "Execute at most two complementary local searches; the program then reranks and fetches evidence automatically.",
                batch_schema,
            ),
            _function_tool(
                self.TERMINAL_TOOL,
                "Finish this atomic retrieval step and submit its factual execution report.",
                STEP_REPORT_SCHEMA,
            ),
        ]

    def _load_json(self, name: str) -> List[Dict[str, Any]]:
        if name not in self._knowledge_cache:
            payload = json.loads((self.data_root / name).read_text(encoding="utf-8"))
            self._knowledge_cache[name] = [dict(row) for row in payload] if isinstance(payload, list) else []
        return [dict(row) for row in self._knowledge_cache[name]]

    def _lookup_drug_name_rows(self, drug_names: List[str]) -> List[Dict[str, Any]]:
        wanted = {self._normalize_term(value) for value in drug_names if self._normalize_term(value)}
        rows: List[Dict[str, Any]] = []
        for index, row in enumerate(self._load_json("all_drug_name_map.json")):
            names = [row.get("drug_name"), row.get("generic_name"), row.get("brand_name"), *(row.get("alias") or [])]
            if wanted & {self._normalize_term(value) for value in names if self._normalize_term(value)}:
                rows.append({**row, "id": f"drug_name_map:{index}"})
        return rows

    def _lookup_pk_relation_rows(self, drug_names: List[str]) -> List[Dict[str, Any]]:
        wanted = {self._normalize_term(value) for value in drug_names if self._normalize_term(value)}
        rows: List[Dict[str, Any]] = []
        for index, row in enumerate(self._load_json("all_drug_pk_relation.json")):
            names = [row.get("drug_name"), row.get("related_pk_drug")]
            if wanted & {self._normalize_term(value) for value in names if self._normalize_term(value)}:
                rows.append({**row, "id": f"drug_pk_relation:{index}"})
        return rows

    def _lookup_ddi_rule_rows(self, enzymes_transporters: List[str]) -> List[Dict[str, Any]]:
        wanted = {self._normalize_term(value) for value in enzymes_transporters if self._normalize_term(value)}
        rows: List[Dict[str, Any]] = []
        for index, row in enumerate(self._load_json("all_ddi_rule.json")):
            if self._normalize_term(row.get("enzyme_transporter")) in wanted:
                rows.append({**row, "id": f"ddi_rule:{index}"})
        return rows

    def _lookup_mutation_drug_rows(self, mutation_names: List[str]) -> List[Dict[str, Any]]:
        wanted = {self._normalize_term(value) for value in mutation_names if self._normalize_term(value)}
        rows: List[Dict[str, Any]] = []
        for index, row in enumerate(self._load_json("mutation_drug_map_min.json")):
            mutation = self._normalize_term(row.get("mutation"))
            if not mutation:
                continue
            if mutation in wanted or any(token and token in mutation for token in wanted):
                rows.append({**row, "id": f"mutation_drug_map:{index}"})
        return rows

    def _lookup_case_ddi_rows(
        self,
        drug_names: List[str],
        mutation_names: List[str],
        interaction_types: Optional[List[str]] = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        wanted_drugs = {self._normalize_term(value) for value in drug_names if self._normalize_term(value)}
        wanted_mutations = {self._normalize_term(value) for value in mutation_names if self._normalize_term(value)}
        wanted_interactions = {self._normalize_term(value) for value in interaction_types or [] if self._normalize_term(value)}
        payload = json.loads((self.data_root / "case_ddi.json").read_text(encoding="utf-8"))
        case_rows = payload.get("case_ddis") if isinstance(payload, dict) else []
        rows: List[Dict[str, Any]] = []
        for index, row in enumerate(case_rows if isinstance(case_rows, list) else []):
            drugs = {self._normalize_term(value) for value in row.get("drugs") or [] if self._normalize_term(value)}
            alterations = {
                self._normalize_term(value)
                for value in ((row.get("mutation") or {}).get("alterations") or [])
                if self._normalize_term(value)
            }
            interaction = self._normalize_term(row.get("interaction_type"))
            if wanted_drugs and not (wanted_drugs & drugs):
                continue
            if wanted_mutations and not any(
                wanted == alt or wanted in alt
                for wanted in wanted_mutations
                for alt in alterations
            ):
                continue
            if wanted_interactions and interaction not in wanted_interactions:
                continue
            rows.append({**row, "id": f"case_ddi:{index}"})
            if len(rows) >= max(1, int(top_k)):
                break
        return rows

    def _expand_query_with_knowledge(
        self,
        *,
        query: str,
        drug_names: List[str],
        mutation_names: List[str],
        strategies: List[str],
        top_k_terms: int,
    ) -> Dict[str, Any]:
        allowed = [str(name) for name in strategies if str(name)]
        added_terms: List[str] = []
        seen_terms: Set[str] = {
            self._normalize_term(value)
            for value in [*drug_names, *mutation_names, query]
            if self._normalize_term(value)
        }
        knowledge_rows: List[Dict[str, Any]] = []
        alias_rows: List[Dict[str, Any]] = []
        pk_rows: List[Dict[str, Any]] = []
        ddi_rows: List[Dict[str, Any]] = []
        mutation_rows: List[Dict[str, Any]] = []
        case_ddi_rows: List[Dict[str, Any]] = []

        if "drug_alias" in allowed:
            alias_rows = self._lookup_drug_name_rows(drug_names)
            knowledge_rows.extend(alias_rows)
            for row in alias_rows:
                self._append_unique_terms(added_terms, seen_terms, [row.get("drug_name"), row.get("generic_name"), row.get("brand_name"), *(row.get("alias") or [])])

        if "mutation_drug" in allowed:
            mutation_rows = self._lookup_mutation_drug_rows(mutation_names)
            knowledge_rows.extend(mutation_rows)
            for row in mutation_rows:
                self._append_unique_terms(added_terms, seen_terms, [row.get("mutation"), *(row.get("drugs") or [])])

        if "case_ddi" in allowed:
            case_ddi_rows = self._lookup_case_ddi_rows(drug_names, mutation_names, top_k=top_k_terms)
            knowledge_rows.extend(case_ddi_rows)
            for row in case_ddi_rows:
                self._append_unique_terms(
                    added_terms,
                    seen_terms,
                    [
                        *(row.get("drugs") or []),
                        row.get("interaction_type"),
                        ((row.get("outcome") or {}).get("response") if isinstance(row.get("outcome"), dict) else ""),
                        *(((row.get("outcome") or {}).get("adverse_events") or []) if isinstance(row.get("outcome"), dict) else []),
                    ],
                )

        expanded_drugs = list(drug_names)
        self._append_unique_terms(expanded_drugs, set(), added_terms)
        if "pk_relation" in allowed:
            pk_rows = self._lookup_pk_relation_rows(expanded_drugs)
            knowledge_rows.extend(pk_rows)
            for row in pk_rows:
                self._append_unique_terms(added_terms, seen_terms, [row.get("drug_name"), row.get("related_pk_drug"), row.get("enzyme_transporter"), row.get("pk_interaction_type"), row.get("role")])

        if "ddi_rule" in allowed:
            ddi_rows = self._lookup_ddi_rule_rows([str(row.get("enzyme_transporter") or "") for row in pk_rows])
            knowledge_rows.extend(ddi_rows)
            for row in ddi_rows:
                self._append_unique_terms(added_terms, seen_terms, [row.get("enzyme_transporter"), row.get("role1"), row.get("role2"), row.get("ddi_likelihood")])

        limited_terms = added_terms[: max(1, int(top_k_terms))]
        expanded_query = " ".join([query.strip(), *limited_terms]).strip()
        return {
            "ok": True,
            "expanded_query": expanded_query,
            "added_terms": limited_terms,
            "knowledge_rows": knowledge_rows,
            "matched_sources": sorted({str(row.get("id") or "").split(":", 1)[0] for row in knowledge_rows if row.get("id")}),
            "alias_count": len(alias_rows),
            "pk_relation_count": len(pk_rows),
            "ddi_rule_count": len(ddi_rows),
            "mutation_drug_count": len(mutation_rows),
            "case_ddi_count": len(case_ddi_rows),
        }

    def run(
        self,
        *,
        main_question: str,
        plan_step: Dict[str, Any],
        route_plan: RoutePlan,
        rerank_goal: str,
        top_k: int,
        expansion: Optional[Dict[str, Any]] = None,
        previously_seen_chunk_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self.llm_client is None or not hasattr(self.llm_client, "run_function_tool_loop"):
            raise RuntimeError("Execution agent requires an LLM client with function tool loop support")
        lane = str(plan_step.get("evidence_lane") or "direct_case")
        expansion = dict(expansion or {})
        two_stage_mode = not bool(expansion.get("enabled"))
        registered_tools = (
            self._two_stage_tool_definitions()
            if two_stage_mode
            else self._tool_definitions(lane, expansion)
        )
        hits_by_id: Dict[str, SearchHit] = {}
        evidence_by_id: Dict[str, Dict[str, Any]] = {}
        knowledge_rows: Dict[str, Dict[str, Any]] = {}
        result_buckets: Dict[str, List[Dict[str, Any]]] = {}
        retrieval_rankings: Dict[str, List[str]] = {}
        used_tools: List[str] = []
        queries_attempted: List[str] = []
        call_counts: Dict[str, int] = {}
        expanded_query_override: Optional[str] = None
        seed_query = ""
        seed_result: Dict[str, Any] = {
            "ok": True, "hit_count": 0, "hits": [], "known_candidate_count": 0,
        }

        def emit_stage(stage: str, data: Dict[str, Any], started: Optional[float] = None) -> None:
            emitter = getattr(self.llm_client, "emit_tool_stage", None)
            if not callable(emitter):
                return
            duration_ms = None
            if started is not None:
                duration_ms = (time.perf_counter() - started) * 1000
            emitter(stage, data, duration_ms=duration_ms)

        def effective_query(raw_query: str) -> str:
            query_text = str(raw_query or "").strip()
            if expanded_query_override and query_text == str(route_plan.search_query or "").strip():
                query_text = expanded_query_override
            return query_text

        def remember_hits(name: str, hits: List[SearchHit]) -> Dict[str, Any]:
            channel_ids = retrieval_rankings.setdefault(name, [])
            for hit in hits:
                if hit.id not in channel_ids:
                    channel_ids.append(hit.id)
                existing = hits_by_id.get(hit.id)
                if existing is None or hit.score > existing.score:
                    hits_by_id[hit.id] = hit
            rows = [self._hit_payload(hit) for hit in hits[: self.max_hits_per_tool_result]]
            result_buckets.setdefault(name, []).extend(rows)
            return {"ok": True, "hit_count": len(hits), "hits": rows, "known_candidate_count": len(hits_by_id)}

        def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal expanded_query_override, seed_query, seed_result
            if name != "execute_retrieval_batch":
                used_tools.append(name)
            call_counts[name] = call_counts.get(name, 0) + 1
            if name == "execute_retrieval_batch":
                actions = [dict(item) for item in args.get("actions") or []][:2]

                # Tool selection and tool-specific query generation happen in
                # the model action first. Only then may the program use the
                # semantic query for Step3 document location and Step2 detail
                # recovery. This avoids executing a generic Planner query
                # before the Execution Agent has made its decision.
                semantic_action = next(
                    (
                        action for action in actions
                        if str(action.get("tool") or "") in {"dense_search", "hybrid_search"}
                        and str(action.get("query") or "").strip()
                    ),
                    None,
                )
                seed_search = getattr(self.tools, "step3_then_step2_search", None)
                if semantic_action is not None and callable(seed_search):
                    stage_started = time.perf_counter()
                    seed_query = str(semantic_action.get("query") or "").strip()
                    # Step3 is a document locator, not the final evidence
                    # ranking. Keep a modestly wider seed pool before Step2
                    # recovery and LLM reranking; the case-level regression
                    # target for this path appears at rank 16.
                    seed_hits = seed_search(query=seed_query, top_k=max(32, top_k * 2))
                    seed_result = remember_hits("step3_then_step2_search", seed_hits)
                    used_tools.append("step3_then_step2_search")
                    call_counts["step3_then_step2_search"] = 1
                    queries_attempted.append(seed_query)
                    emit_stage("step3_then_step2_search", {
                        "input": {"query": seed_query, "top_k": max(32, top_k * 2)},
                        "output": seed_result,
                    }, stage_started)

                def execute_action(action: Dict[str, Any]) -> Dict[str, Any]:
                    tool_name = str(action.get("tool") or "")
                    common_top_k = min(30, max(1, int(action.get("top_k") or top_k)))
                    if tool_name == "structured_search":
                        return execute_tool(tool_name, {
                            "constraints": action.get("constraints") or {},
                            "top_k": common_top_k,
                        })
                    if tool_name == "dense_search":
                        return execute_tool(tool_name, {
                            "query": action.get("query") or route_plan.search_query,
                            "top_k": common_top_k,
                            "spaces": action.get("spaces") or [
                                "case_semantic", "event_semantic", "structured_semantic",
                            ],
                        })
                    if tool_name == "bm25_search":
                        return execute_tool(tool_name, {
                            "query": action.get("query") or route_plan.search_query,
                            "top_k": common_top_k,
                        })
                    if tool_name == "hybrid_search":
                        return execute_tool(tool_name, {
                            "query": action.get("query") or route_plan.search_query,
                            "constraints": action.get("constraints") or {},
                            "top_k": common_top_k,
                        })
                    return {"ok": False, "error": "unsupported_batch_tool"}

                # Execute retrieval calls serially. This avoids API/resource
                # contention and keeps tool traces deterministic across modes.
                action_results = []
                for action in actions:
                    stage_started = time.perf_counter()
                    action_result = execute_action(action)
                    action_results.append(action_result)
                    emit_stage(str(action.get("tool") or "retrieval_action"), {
                        "input": action,
                        "output": action_result,
                    }, stage_started)

                # A chunk hit also identifies a potentially useful document.
                # Open a few top documents and recover their best Step2 detail
                # chunks before reranking. This lets an exact-term hit on a
                # diagnosis chunk expose the adjacent treatment/outcome chunk.
                ranked_existing = sorted(
                    hits_by_id.values(), key=lambda item: item.score, reverse=True,
                )
                detail_doc_ids: List[str] = []
                for hit in ranked_existing:
                    doc_id = str(
                        hit.metadata.get("doc_id") or hit.id.split("#", 1)[0]
                    ).strip()
                    if doc_id and doc_id not in detail_doc_ids:
                        detail_doc_ids.append(doc_id)
                    if len(detail_doc_ids) >= 10:
                        break
                detail_query = " ".join([
                    rerank_goal,
                    *[
                        str(action.get("query") or "")
                        for action in actions
                        if str(action.get("query") or "").strip()
                    ],
                ]).lower()
                detail_terms = set(retrieval_terms(detail_query, max_terms=64))
                detail_hits: List[SearchHit] = []
                for doc_id in detail_doc_ids:
                    rows = self.tools.fetch_evidence(doc_id=doc_id, limit=300)
                    aligned_selector = getattr(self.tools, "_select_event_aligned_details", None)
                    if callable(aligned_selector):
                        aligned_rows = aligned_selector(
                            rows,
                            query=detail_query,
                            locator_texts=[
                                hit.text for hit in ranked_existing
                                if str(hit.metadata.get("doc_id") or hit.id.split("#", 1)[0]) == doc_id
                            ][:3],
                            locator_event_types=[],
                            per_document=3,
                        )
                        for score, row, evidence_slot in aligned_rows:
                            detail_hits.append(SearchHit(
                                id=str(row["chunk_id"]), score=float(score), source="document_detail",
                                text=str(row.get("text") or ""), metadata={
                                    **row,
                                    "target_event_types": self.tools._target_event_types(detail_query),
                                    "event_aligned_slot": evidence_slot,
                                    "retrieval_pipeline": "retrieved_document_to_step2_event_aligned_detail",
                                },
                            ))
                        continue
                    scored_rows: List[tuple[float, Dict[str, Any]]] = []
                    for row in rows:
                        if row.get("chunk_type") != "case_text_chunk":
                            continue
                        text = " ".join(str(row.get("text") or "").split())
                        lowered = text.lower()
                        lexical = sum(term in lowered for term in detail_terms) / max(
                            1, len(detail_terms)
                        )
                        clinical = min(
                            0.25,
                            0.025 * len(re.findall(
                                r"\b(?:pemetrexed|carboplatin|cisplatin|osimertinib|"
                                r"almonertinib|leptomeningeal|lmd|csf|response|"
                                r"progress|orr|dcr|recist|pr|sd|pd)\b",
                                lowered,
                            )),
                        )
                        scored_rows.append((lexical + clinical, row))
                    scored_rows.sort(key=lambda item: item[0], reverse=True)
                    for score, row in scored_rows[:3]:
                        detail_hits.append(SearchHit(
                            id=str(row["chunk_id"]),
                            score=float(score),
                            source="document_detail",
                            text=str(row.get("text") or ""),
                            metadata={
                                **row,
                                "retrieval_pipeline": "retrieved_document_to_step2_detail",
                            },
                        ))
                if detail_hits:
                    detail_hits.sort(key=lambda item: item.score, reverse=True)
                    remember_hits("document_detail", detail_hits)
                emit_stage("document_detail_recovery", {
                    "input": {"document_ids": detail_doc_ids, "query": detail_query},
                    "output": {
                        "hit_count": len(detail_hits),
                        "hit_ids": [hit.id for hit in detail_hits],
                    },
                })

                # Fuse channel ranks with reciprocal-rank fusion. This keeps
                # dense/BM25/Step3 agreement instead of comparing incompatible
                # raw scores or discarding an ID's secondary-source matches.
                rrf_scores: Dict[str, float] = {}
                rrf_sources: Dict[str, List[str]] = {}
                rrf_k = 60.0
                for channel, ranked_ids in retrieval_rankings.items():
                    if channel == "rerank_candidates":
                        continue
                    for rank, hit_id in enumerate(ranked_ids, start=1):
                        rrf_scores[hit_id] = rrf_scores.get(hit_id, 0.0) + 1.0 / (rrf_k + rank)
                        rrf_sources.setdefault(hit_id, []).append(channel)
                fused_hits: List[SearchHit] = []
                for hit_id, rrf_score in rrf_scores.items():
                    hit = hits_by_id.get(hit_id)
                    if hit is None:
                        continue
                    metadata = dict(hit.metadata)
                    metadata["rrf_score"] = rrf_score
                    metadata["retrieval_sources"] = list(dict.fromkeys(rrf_sources.get(hit_id) or []))
                    fused_hits.append(SearchHit(
                        id=hit.id, score=rrf_score, source="rrf", text=hit.text, metadata=metadata,
                    ))
                fused_hits.sort(key=lambda item: item.score, reverse=True)
                emit_stage("rrf_fusion", {
                    "input": {"rankings": retrieval_rankings, "rrf_k": rrf_k},
                    "output": {
                        "candidate_ids": [hit.id for hit in fused_hits],
                        "scores": {hit.id: hit.score for hit in fused_hits},
                    },
                })
                # Reserve room for complementary channels before filling by
                # fused rank. Dense similarity often over-ranks a treatment-
                # adjacent combination regimen, while BM25 can surface an
                # exact single-agent/drug/dose case. Both must reach reranking.
                detail_ids = {hit.id for hit in detail_hits}
                reserved_ids: List[str] = []
                for channel in ("bm25_search", "dense_search", "step3_then_step2_search", "document_detail"):
                    for hit_id in (retrieval_rankings.get(channel) or [])[:3]:
                        if hit_id not in reserved_ids:
                            reserved_ids.append(hit_id)
                reserved_hits = [hits_by_id[hit_id] for hit_id in reserved_ids if hit_id in hits_by_id]
                detail_priority = [hit for hit in fused_hits if hit.id in detail_ids and hit.id not in reserved_ids][:5]
                selected_ids = {hit.id for hit in [*reserved_hits, *detail_priority]}
                candidate_pool = [*reserved_hits, *detail_priority]
                candidate_pool.extend(
                    hit for hit in fused_hits
                    if hit.id not in selected_ids
                )
                candidate_pool = candidate_pool[:16]
                rerank_started = time.perf_counter()
                ranked = self.tools.rerank(
                    query=rerank_goal,
                    hits=candidate_pool,
                    top_k=min(max(1, top_k), 16),
                ) if candidate_pool else []
                rerank_result = remember_hits("rerank_candidates", ranked)
                used_tools.append("rerank_candidates")
                rerank_backend = str(getattr(self.tools, "last_rerank_backend", "none"))
                rerank_stage = "qwen_rerank" if "qwen" in rerank_backend.lower() else "rerank_candidates"
                emit_stage(rerank_stage, {
                    "input": {
                        "goal": rerank_goal,
                        "candidate_ids": [hit.id for hit in candidate_pool],
                    },
                    "output": {
                        "backend": rerank_backend,
                        "ranked_candidates": rerank_result.get("hits") or [],
                        "overflow_ids": [hit.id for hit in getattr(self.tools, "last_qwen_overflow", [])],
                    },
                }, rerank_started)
                candidate_chunk_ids = self.router._collect_chunk_ids(ranked, top_k=min(top_k, 12))
                fetch_started = time.perf_counter()
                fetched = self.tools.fetch_evidence(
                    chunk_ids=candidate_chunk_ids,
                    limit=min(max(1, top_k), 12),
                ) if candidate_chunk_ids else []
                for row in fetched:
                    if row.get("chunk_id"):
                        evidence_by_id[str(row["chunk_id"])] = dict(row)
                if fetched:
                    used_tools.append("fetch_evidence")
                    result_buckets.setdefault("fetch_evidence", []).extend(fetched)
                emit_stage("fetch_evidence", {
                    "input": {"chunk_ids": candidate_chunk_ids, "limit": min(max(1, top_k), 12)},
                    "output": {
                        "requested_count": len(candidate_chunk_ids),
                        "fetched_count": len(fetched),
                        "evidence": fetched,
                    },
                }, fetch_started)
                return {
                    "ok": True,
                    "action_results": action_results,
                    "reranked_candidates": rerank_result.get("hits") or [],
                    "qwen_overflow_hits": [hit.__dict__ for hit in getattr(self.tools, "last_qwen_overflow", [])],
                    "knowledge_context": [hit.__dict__ for hit in getattr(self.tools, "last_knowledge_hits", [])],
                    "rerank_backend": getattr(self.tools, "last_rerank_backend", "none"),
                    "fetched_evidence_ids": list(evidence_by_id)[:12],
                    "instruction": "Submit the terminal execution report now.",
                }
            if name in {"structured_search", "dense_search", "bm25_search", "hybrid_search"}:
                search_count = sum(call_counts.get(tool_name, 0) for tool_name in {"structured_search", "dense_search", "bm25_search", "hybrid_search"})
                if search_count > self.max_search_calls:
                    raise RuntimeError("search_call_budget_exceeded")
            if name in {"fetch_evidence", "fetch_structured_events"} and call_counts[name] > self.max_fetch_calls:
                raise RuntimeError("fetch_call_budget_exceeded")
            if name == "structured_search":
                return remember_hits(name, self.tools.structured_search(self._complete_constraints(dict(args["constraints"])), top_k=int(args["top_k"])))
            if name == "dense_search":
                query = effective_query(str(args["query"])); queries_attempted.append(query)
                return remember_hits(name, self.tools.dense_search(query=query, top_k=int(args["top_k"]), spaces=list(args["spaces"])))
            if name == "bm25_search":
                query = effective_query(str(args["query"])); queries_attempted.append(query)
                return remember_hits(name, self.tools.bm25_search(query=query, top_k=int(args["top_k"])))
            if name == "hybrid_search":
                query = effective_query(str(args["query"])); queries_attempted.append(query)
                return remember_hits(name, self.tools.hybrid_search(query=query, constraints=self._complete_constraints(dict(args["constraints"])), top_k=int(args["top_k"])))
            if name == "rerank_candidates":
                ids = [str(value) for value in args["candidate_ids"] if str(value) in hits_by_id]
                candidates = [hits_by_id[value] for value in ids] if ids else list(hits_by_id.values())
                ranked = self.tools.rerank(query=str(args["goal"]), hits=candidates[:top_k], top_k=int(args["top_k"]))
                return remember_hits(name, ranked)
            if name == "fetch_evidence":
                ids = [str(value) for value in args["chunk_ids"] if str(value) in hits_by_id or "#chunk-" in str(value)]
                rows = self.tools.fetch_evidence(chunk_ids=ids, limit=int(args["limit"]))
                for row in rows:
                    if row.get("chunk_id"):
                        evidence_by_id[str(row["chunk_id"])] = dict(row)
                compact = [{"chunk_id": row.get("chunk_id"), "evidence_level": row.get("evidence_level"), "text": " ".join(str(row.get("text") or "").split())[:360]} for row in rows]
                result_buckets.setdefault(name, []).extend(compact)
                return {"ok": True, "requested_count": len(ids), "fetched_count": len(rows), "evidence": compact}
            if name == "fetch_structured_events":
                event_ids = [str(value) for value in args["event_ids"] if str(value)]
                rows = self.tools.fetch_structured_events(event_ids=event_ids, limit=int(args["limit"]))
                for row in rows:
                    evidence_by_id[str(row["chunk_id"])] = dict(row)
                compact = [{"event_id": row.get("event_id"), "text": " ".join(str(row.get("text") or "").split())[:360]} for row in rows]
                result_buckets.setdefault(name, []).extend(compact)
                return {"ok": True, "requested_count": len(event_ids), "fetched_count": len(rows), "events": compact}
            if name == "expand_query_with_knowledge":
                expanded = self._expand_query_with_knowledge(
                    query=str(args["query"]),
                    drug_names=[str(value) for value in args["drug_names"]],
                    mutation_names=[str(value) for value in args["mutation_names"]],
                    strategies=[str(value) for value in args["strategies"]],
                    top_k_terms=int(args["top_k_terms"]),
                )
                expanded_query_override = str(expanded.get("expanded_query") or "").strip() or None
                for row in expanded["knowledge_rows"]:
                    if row.get("id"):
                        knowledge_rows[str(row["id"])] = dict(row)
                compact = {
                    "expanded_query": expanded["expanded_query"],
                    "added_terms": expanded["added_terms"],
                    "matched_sources": expanded["matched_sources"],
                    "alias_count": expanded["alias_count"],
                    "pk_relation_count": expanded["pk_relation_count"],
                    "ddi_rule_count": expanded["ddi_rule_count"],
                    "mutation_drug_count": expanded["mutation_drug_count"],
                    "case_ddi_count": expanded["case_ddi_count"],
                }
                result_buckets.setdefault(name, []).append(compact)
                return {"ok": True, **compact}
            if name == "normalize_drug_names":
                rows = self._lookup_drug_name_rows([str(value) for value in args["drug_names"]])
                for item in rows:
                    knowledge_rows[item["id"]] = item
                result_buckets.setdefault(name, []).extend(rows)
                return {"ok": True, "match_count": len(rows), "mappings": rows[:10]}
            if name == "lookup_pk_relations":
                rows = self._lookup_pk_relation_rows([str(value) for value in args["drug_names"]])
                for item in rows:
                    knowledge_rows[item["id"]] = item
                result_buckets.setdefault(name, []).extend(rows)
                return {"ok": True, "match_count": len(rows), "relations": rows[:15]}
            if name == "lookup_ddi_rules":
                rows = self._lookup_ddi_rule_rows([str(value) for value in args["enzymes_transporters"]])
                for item in rows:
                    knowledge_rows[item["id"]] = item
                result_buckets.setdefault(name, []).extend(rows)
                return {"ok": True, "match_count": len(rows), "rules": rows[:15]}
            if name == "lookup_mutation_drug_relations":
                rows = self._lookup_mutation_drug_rows([str(value) for value in args["mutation_names"]])
                for item in rows:
                    knowledge_rows[item["id"]] = item
                result_buckets.setdefault(name, []).extend(rows)
                return {"ok": True, "match_count": len(rows), "relations": rows[:15]}
            if name == "lookup_case_ddi_relations":
                rows = self._lookup_case_ddi_rows(
                    [str(value) for value in args["drug_names"]],
                    [str(value) for value in args["mutation_names"]],
                    [str(value) for value in args["interaction_types"]],
                    int(args["top_k"]),
                )
                for item in rows:
                    knowledge_rows[item["id"]] = item
                result_buckets.setdefault(name, []).extend(rows)
                return {"ok": True, "match_count": len(rows), "relations": rows[:15]}
            raise ValueError(f"Unsupported execution tool: {name}")

        prompt = json.dumps({
            "main_question": main_question,
            "current_step": plan_step,
            "planner_query_hint": route_plan.search_query,
            "planner_suggested_tools": route_plan.selected_tools,
            "rerank_goal": rerank_goal,
            "constraints": self._complete_constraints(route_plan.constraints),
            "expansion_policy": expansion,
            "previously_seen_chunk_ids": list(previously_seen_chunk_ids or [])[-50:],
            "seed_strategy": "After you select dense_search or hybrid_search, its tool-specific query will drive Step3-to-Step2 seed retrieval.",
            "budgets": {"max_turns": self.max_turns, "max_search_calls": self.max_search_calls, "max_fetch_calls": self.max_fetch_calls, "top_k": top_k, "max_hits_per_tool_result": self.max_hits_per_tool_result},
        }, ensure_ascii=False, indent=2)
        terminal_recovery: Optional[Dict[str, Any]] = None
        def run_execution_loop(output_tokens: int) -> Dict[str, Any]:
            return self.llm_client.run_function_tool_loop(
                system=self.SYSTEM_PROMPT,
                user=prompt,
                tools=registered_tools,
                execute_tool=execute_tool,
                terminal_tool_name=self.TERMINAL_TOOL,
                max_turns=2 if two_stage_mode else self.max_turns,
                # This Responses-compatible gateway accepts string
                # tool_choice values. The system contract and two-stage tool
                # schema require execute_retrieval_batch on the first turn.
                initial_tool_choice="required" if two_stage_mode else None,
                temperature=0.1,
                max_output_tokens=output_tokens,
            )

        if two_stage_mode:
            batch_definition = self._two_stage_tool_definitions()[0]["function"]
            batch_arguments = self.llm_client.call_function(
                system=self.SYSTEM_PROMPT,
                user=prompt,
                function_name="execute_retrieval_batch",
                description=str(batch_definition["description"]),
                parameters=dict(batch_definition["parameters"]),
                temperature=0.1,
                max_output_tokens=6000,
                max_attempts=2,
            )
            execute_via_pipeline = getattr(self.llm_client, "execute_tool_call", None)
            if callable(execute_via_pipeline):
                batch_result = execute_via_pipeline(
                    tool="execute_retrieval_batch",
                    arguments=batch_arguments,
                    handler=lambda: execute_tool("execute_retrieval_batch", batch_arguments),
                    operation="retrieval_execution",
                    turn=1,
                )
            else:
                # Compatibility for lightweight test/dry-run clients. Real
                # clients always use the durable pipeline above.
                batch_result = execute_tool("execute_retrieval_batch", batch_arguments)
            loop_result = {
                "report": {},
                "tool_trace": [{
                    "turn": 1,
                    "tool_name": "execute_retrieval_batch",
                    "arguments": batch_arguments,
                    "result": batch_result,
                }],
                "turns": 1,
            }
        else:
            try:
                # Expanded knowledge mode still permits a bounded multi-tool
                # interaction; normal retrieval no longer enters this loop.
                loop_result = run_execution_loop(6000)
            except LLMClientError as exc:
                if not self._is_recoverable_terminal_protocol_error(exc):
                    raise
                loop_result = run_execution_loop(9000)
                terminal_recovery = {
                    "used": False,
                    "protocol_retry": True,
                    "reason": str(exc)[:300],
                }

        if not evidence_by_id and hits_by_id:
            fallback_ids = self.router._collect_chunk_ids(self.router._dedupe_hits(list(hits_by_id.values())), top_k=top_k)
            for row in self.tools.fetch_evidence(chunk_ids=fallback_ids, limit=top_k):
                if row.get("chunk_id"):
                    evidence_by_id[str(row["chunk_id"])] = dict(row)
            if fallback_ids:
                used_tools.append("fetch_evidence_guardrail")

        for item_id, row in knowledge_rows.items():
            source = item_id.split(":", 1)[0].strip().lower()
            text = json.dumps(row, ensure_ascii=False)
            source_files = {
                "drug_name_map": "all_drug_name_map.json",
                "drug_pk_relation": "all_drug_pk_relation.json",
                "ddi": "all_ddi_rule.json",
                "ddi_rule": "all_ddi_rule.json",
                "mutation_drug_map": "mutation_drug_map_min.json",
                "case_ddi": "case_ddi.json",
            }
            source_file = source_files.get(source, "case_ddi.json")
            evidence_by_id[item_id] = {
                "chunk_id": item_id,
                "doc_id": source,
                "case_id": f"{source}#knowledge",
                "chunk_type": "ddi_rule_chunk",
                "evidence_level": "drug_label_or_ddi_rule_evidence",
                "text": text,
                "citation_json": {"source_layer": "drug_label_or_ddi_rule", "source_file": str(self.data_root / source_file)},
                "metadata_json": {"source": source},
            }

        evidence = list(evidence_by_id.values())
        assessment = self.router._assess_evidence(evidence)
        deduped_hits = self.router._dedupe_hits(list(hits_by_id.values()))
        report = dict(loop_result.get("report") or {})
        valid_ids = set(evidence_by_id)
        if two_stage_mode:
            batch_result = dict((loop_result.get("tool_trace") or [{}])[0].get("result") or {})
            fetched_count = len(evidence_by_id)
            candidate_count = len(hits_by_id)
            execution_status = (
                "partial" if not batch_result.get("ok", True)
                else "success" if fetched_count
                else "partial" if candidate_count
                else "no_evidence"
            )
            operational_failures = [
                str(item.get("error"))
                for item in batch_result.get("action_results") or []
                if isinstance(item, dict) and item.get("error")
            ]
            if batch_result.get("error"):
                operational_failures.append(str(batch_result["error"]))
            candidate_evidence_ids = list(evidence_by_id)[:12]
            report = {
                "report_source": "deterministic_tool_result",
                "execution_status": execution_status,
                "summary": (
                    f"retrieval completed: candidates={candidate_count}, fetched={fetched_count}"
                    if batch_result.get("ok", True)
                    else "retrieval execution failed"
                ),
                # Compatibility alias only. EvidenceReview, not this report,
                # decides whether any candidate is medically acceptable.
                "accepted_evidence_ids": candidate_evidence_ids,
                "candidate_evidence_ids": candidate_evidence_ids,
                "rejected_evidence_ids": [],
                "queries_attempted": list(dict.fromkeys(queries_attempted))[-2:],
                "query_database_status": "uncertain",
                "exhaustion_reason": "",
                "recommended_query_change": "",
                "goal_evaluation": {
                    "matched_goal_count": 0,
                    "best_goal_relevance": 0.0,
                    "success_criteria_met": False,
                    "observed_gaps": [],
                    "observed_failures": list(dict.fromkeys(operational_failures)),
                },
            }
        report["accepted_evidence_ids"] = [str(value) for value in report.get("accepted_evidence_ids") or [] if str(value) in valid_ids]
        report["rejected_evidence_ids"] = [str(value) for value in report.get("rejected_evidence_ids") or [] if str(value) in valid_ids]
        report["queries_attempted"] = list(dict.fromkeys([*queries_attempted, *[str(value) for value in report.get("queries_attempted") or []]]))
        report.setdefault("query_database_status", "uncertain")
        report.setdefault("exhaustion_reason", "")
        report.setdefault("recommended_query_change", "")
        return {
            "query_type": route_plan.query_type,
            "selected_tools": list(dict.fromkeys(used_tools)),
            "constraints": route_plan.constraints,
            "planner_metadata": {
                **(route_plan.planner_metadata or {}),
                "execution_mode": "function_tool_loop",
                "seed_query": seed_query,
                "seed_result": seed_result,
            },
            "planned_query": route_plan.search_query,
            "rerank_goal": rerank_goal,
            "results": {**result_buckets, "fetch_evidence": evidence},
            "evidence_layering": {"query": rerank_goal, "layer_distribution": assessment["layer_distribution"], "assessed_evidence": assessment["assessed_items"]},
            "contradiction_check": self.router._contradiction_check(deduped_hits, [], assessment),
            "clinical_safety_gate": self.router.safety_gate.evaluate(query=main_question, query_type=route_plan.query_type, constraints=route_plan.constraints, retrieval_results={"fetch_evidence": evidence}),
            "step_execution_report": report,
            "tool_trace": loop_result.get("tool_trace") or [],
            "tool_loop_turns": loop_result.get("turns"),
            "terminal_recovery": terminal_recovery or {"used": False},
        }
