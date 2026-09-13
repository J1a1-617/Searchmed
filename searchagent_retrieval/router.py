from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .safety_gate import ClinicalSafetyGate
from .tools import RetrievalTools, SearchHit

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return None
    return value if isinstance(value, dict) else None

ALLOWED_TOOLS = frozenset({"structured_search", "dense_search", "bm25_search", "hybrid_search", "fetch_evidence"})

COMMON_GENE_SYMBOLS = frozenset({
    "AKT1", "ALK", "BRAF", "BRCA1", "BRCA2", "CDK4", "CDK6", "EGFR", "ERBB2", "ERBB3",
    "ERBB4", "FGFR1", "FGFR2", "FGFR3", "HRAS", "KIT", "KRAS", "MAP2K1", "MAP2K2", "MDM2",
    "MDM4", "MET", "MYC", "NRAS", "NTRK1", "NTRK2", "NTRK3", "PDGFRA", "PIK3CA", "PIK3CB",
    "PIK3CD", "PTEN", "RB1", "RET", "ROS1", "STK11", "TP53",
})

GENE_ALTERATION_NOISE = frozenset({
    "AI", "AN", "CNS", "CT", "CSF", "DDI", "ECOG", "LUAD", "MRI", "NA", "NSCLC",
    "PET", "PS", "PR", "CR", "SD", "PD", "SCLC", "TKI",
})

PLANNER_SYSTEM_PROMPT = """你是肿瘤临床检索 PlannerAgent。你必须根据主问题、既往检索轮次记忆、已获得证据、失败方向和剩余缺口，动态制定本轮检索计划。你的目标是最大化对主问题的信息增益，同时主动寻找反证、安全风险和不可违背的临床约束。constraints 只能放用户明确提供或历史 Session 已确认的患者事实。候选药物、可能耐药机制、待检索毒性和检索指标不是已确认约束，不得写入 constraints，只能写入 search_query 或 targeted_information_gain。

必须通过 submit_retrieval_plan 函数提交计划，不要生成函数调用之外的回答。不得虚构数据库字段或工具。可用工具仅有：structured_search、dense_search、bm25_search、hybrid_search、fetch_evidence。"""

QUERY_UNDERSTANDING_SYSTEM_PROMPT = """你是肿瘤临床 QueryUnderstanding 模块。你的唯一职责是把本次用户原始问题忠实转换为检索前的开放问题表示。不要把问题归入一个固定类别，不要制定检索步骤、选择工具、生成检索query或回答医学问题，也不要使用指南或医学常识补全原文未提供的信息。不要参考历史memory或AnswerMemory。

先在内部区分四类时间角色，但不要输出额外字段：
1. 基线患者事实：诊断、转移、分子改变、治疗开始前症状/影像/检查；
2. 既往治疗及已经观察到的疗效或毒性；
3. 当前/计划评估的index treatment；
4. 用户要求预测的未来结局。未来结局不是已知事实，绝不能进入patient_facts。

治疗时间线属于核心语义，不能为了缩短 JSON 而省略。凡原文给出治疗先后、开始/停用、相对治疗的事件时间或多个随访时点，必须保留在对应的 patient_facts、treatment_context、target_outcomes 或 additional_context 中；不得倒置先后，也不得把时间上的先后自动写成药物因果。

用通用临床问题框架识别值得后续 Planner 关注的维度，但只记录与当前问题相关的项：
- 患者匹配：原发癌种/组织学、分子改变、转移部位、治疗线次与既往暴露；
- 当前干预：药物或联合方案、剂量/频率/途径、开始时间，以及哪一个才是 index treatment；
- 结局与时间：用户真正要判断的疗效、进展、毒性、停药、剂量调整或相互作用，以及预测窗口；
- 因果与可比性：需要区分基线异常、既往事件、当前治疗后事件和尚未发生的预测目标；联合方案不自动等于药物相互作用；
- 证据问题：用简短自然语言列出哪些信息会真正改变答案，如直接匹配病例、相反结局、时间窗口、特定器官疗效、安全性或药代信息。这些是信息需求，不是检索步骤。

软 Schema 字段语义：
- patient_facts：原文明确的患者事实，如 cancer_type、gene_alterations、metastatic_sites、observed_responses、observed_toxicities；
- treatment_context：区分 current_treatments、prior_treatments 和 regimen_details，不得混淆当前与既往药物；
- target_outcomes：用户要判断的未来或未知结局，可包含 time_window；
- information_needs：最多5项，描述检索需要回答的证据问题，不是建议补做的患者检查清单；
- ambiguities：最多3项，只记录原文已经表现出的未知、疑似或矛盾；
- additional_context：可选的开放对象，存放上述字段无法自然表达但确实有用的原文信息。

实体语义和禁止项：
- cancer_type：只写原文明确的原发肿瘤类型/组织学；不要把“脑膜转移、骨转移”拼入癌种，也不要自行标准化成原文未给出的亚型。
- gene_alterations：仅写明确的基因/融合/突变/扩增等分子改变，例如EGFR 19del、EGFR L858R、MET amplification。ALT、AST、ALP、TBIL、肌酐、血常规、ECOG/PS、药物剂量、日期以及CR/PR/SD/PD均不是基因改变。
- current_treatments：只放当前/计划评估的index treatment；联合方案必须保留每个当前药物。既往已停用药物放入prior_treatments。剂量、频率和途径放入regimen_details，不拼进药名。
- responses：仅放时间切点前已经明确观察到的CR/PR/SD/PD；“反应欠佳、可能有效、预计PR、预测8–12周结局”不是已确认response。
- toxicities：仅放时间切点前已经发生且原文明示为不良反应/毒性的事件。异常化验若未明确归因为药物毒性，放入clinical_terms，不要擅自诊断为肝损伤。
- metastatic_sites：仅放明确的转移部位，如脑、脑膜、骨、肝；不要把原发部位当转移。
- ddi_terms：仅放原文明示的相互作用、酶、转运体或PK事实；不要因为方案含多个药物就自动生成ddi_terms。

patient_facts 中可被检索程序作为硬约束的事实必须高精度：只放原文明示、时间角色明确、可安全硬过滤的患者事实。既往治疗、模糊检查、疑似转移、未知耐药状态、待预测结局以及医学推测不得作为硬约束。若原文有“实际采用的治疗方案”，它是当前index treatment的权威来源。宁可少填，也不得推测或补全。

ambiguities只允许记录原文中明确出现的未知状态、疑似表述或相互矛盾的信息。不要罗列原文完全未提及的检查、治疗或风险因素。不同输入字段对同一明确事实的表述差异，不要自动描述成矛盾。

information_needs 应面向当前病例库检索，例如“相同驱动和治疗序列下的早期颅内疗效”“相同方案的相反结局或早期毒性”。不要写“需要检测T790M”“需要补做影像”“需要提供肝肾功能”等患者检查建议；除非用户原问题明确询问还缺哪些临床信息。原文没有提到某项检测、放疗、激素或合并用药时，也不要把它们写成 ambiguities。

示例边界：
- “EGFR 19del，吉非替尼后脑转移进展，现用奥希替尼80mg，预测8–12周结局”：current_treatments仅为奥希替尼；吉非替尼写入prior_treatments；脑写入metastatic_sites；结局与8–12周写入target_outcomes；不得猜测T790M。
- “奥希替尼联合贝伐珠单抗后的疗效”：联合方案本身不是DDI；两药都是当前方案时均进入current_treatments。
- “ALT 90、AST 118，是否为药物性肝损伤”：ALT/AST绝不进入gene_alterations；因为因果尚未确认，异常值放入patient_facts的检查事实，待判断因果写入target_outcomes。

只返回一个符合用户模板的JSON对象，不输出解释、Markdown、思维过程或函数调用。"""

QUERY_UNDERSTANDING_USER_TEMPLATE = """用户原始问题：
{query}

请返回一个 JSON 对象。这是软 Schema，不是封闭 Schema：保留 patient_facts、treatment_context、target_outcomes、information_needs、ambiguities 这些稳定交接键，但允许子字段和 additional_context 根据问题自由扩展。

- patient_facts 是开放对象；
- treatment_context 是开放对象，但须明确区分 current_treatments 与 prior_treatments；
- target_outcomes、information_needs、ambiguities 使用自然语言列表；
- additional_context 是可选开放对象。

要求：
1. 不做问题类型分类。
2. patient_facts 只放原文明示且时间角色明确的事实。
3. current_treatments只放当前/计划评估方案；既往药物放入prior_treatments。
4. information_needs最多5项，只写检索需要回答的证据问题；不要写成医学指南、研究大纲、患者检查清单或“还需要提供什么病历”。
5. ambiguities最多3项；只记录原文明确呈现的未知、疑似或矛盾，不要枚举原文未提及的检查/治疗。
6. 只提取和组织问题；不要使用指南或背景知识提前回答。
7. 不要编造不存在的实体，不要输出 JSON 之外的解释文本。"""

PLANNER_USER_TEMPLATE = """主问题：
{main_question}

当前候选查询：
{query}

Planner 上下文（包含既往逐轮记忆和累计回答记忆）：
{planner_context}

请输出：
{{
  "search_query": "本轮实际执行的检索表达式",
  "selected_tools": ["工具名"],
  "constraints": {{
    "cancer_type": "可选",
    "gene_alterations": [],
    "drugs": [],
    "responses": [],
    "toxicities": [],
    "metastatic_sites": [],
    "ddi_terms": []
  }},
  "plan_rationale": "为何本轮这样检索",
  "targeted_information_gain": ["本轮要解决的具体缺口"],
  "avoid_repeating": ["不应重复的既往失败方向"],
  "stop_if": ["本轮结束后可停止的条件"]
}}

要求：计划必须参考 Planner 上下文；优先补齐尚未解决且会改变最终回答的缺口；至少考虑一种反证或安全风险检索；search_query 不得混入回答内容。"""

PLANNER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "search_query": {"type": "string", "description": "One concise local-index query for this round; no PubMed Boolean syntax or answer prose."},
        "selected_tools": {"type": "array", "description": "Ordered retrieval tools needed for this round.", "items": {"type": "string", "enum": sorted(ALLOWED_TOOLS)}},
        "constraints": {"type": "object", "description": "Hard filters supported by explicit patient facts only; exclude hypotheses and desired outcomes.", "properties": {
            "cancer_type": {"type": "string"},
            "gene_alterations": {"type": "array", "items": {"type": "string"}},
            "drugs": {"type": "array", "items": {"type": "string"}},
            "responses": {"type": "array", "items": {"type": "string"}},
            "toxicities": {"type": "array", "items": {"type": "string"}},
            "metastatic_sites": {"type": "array", "items": {"type": "string"}},
            "ddi_terms": {"type": "array", "items": {"type": "string"}},
        }, "required": ["cancer_type", "gene_alterations", "drugs", "responses", "toxicities", "metastatic_sites", "ddi_terms"], "additionalProperties": False},
        "plan_rationale": {"type": "string", "description": "Why this round has the highest marginal information value."},
        "targeted_information_gain": {"type": "array", "description": "Specific unknowns this round can resolve.", "items": {"type": "string"}},
        "avoid_repeating": {"type": "array", "items": {"type": "string"}},
        "stop_if": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["search_query", "selected_tools", "constraints", "plan_rationale", "targeted_information_gain", "avoid_repeating", "stop_if"],
    "additionalProperties": False,
}

EVIDENCE_LAYER_PRIORITY = {
    "guideline_evidence": 5,
    "clinical_trial_evidence": 4,
    "cohort_or_observational_evidence": 3,
    "drug_label_or_ddi_rule_evidence": 3,
    "review_evidence": 2,
    "structured_extraction_evidence": 2,
    "case_report_evidence": 1,
    "other_literature_evidence": 1,
    "model_prior_knowledge": 0,
}

SUPPORTING_PATTERNS = [
    re.compile(r"\b(pr|cr|response|respond|benefit|effective|improv(ed|ement)?)\b", re.IGNORECASE),
    re.compile(r"(缓解|获益|有效|改善|病情稳定|控制)"),
]

CONTRADICTING_PATTERNS = [
    re.compile(r"\b(pd|progression|resistance|resistant|fail(ed|ure)?|contraindicat|toxicit|adverse|fatal|death)\b", re.IGNORECASE),
    re.compile(r"(进展|耐药|无效|失败|禁忌|毒性|不良反应|致死|死亡|风险)"),
]

SAFETY_RISK_PATTERNS = [
    re.compile(r"\b(ddi|interaction|contraindicat|toxicit|adverse|grade\s*[3-5]|fatal|death)\b", re.IGNORECASE),
    re.compile(r"(相互作用|禁忌|毒性|不良反应|严重|死亡|风险|警示)"),
]

COUNTER_EVIDENCE_HINTS = {
    "general": "反证 毒性 不良反应 进展 耐药 禁忌 风险",
    "ddi": "禁忌 相互作用 DDI CYP P-gp 严重不良反应",
    "mutation_drug": "耐药 进展 无效 毒性 不良反应",
    "mechanism": "不支持 冲突 反证 无效 进展",
    "similar_case": "预后差 进展 毒性 不良反应",
    "treatment_advice": "禁忌 进展 耐药 毒性 不良反应",
}


@dataclass
class RoutePlan:
    query_type: Optional[str]  # Deprecated compatibility field; no routing decision may depend on it.
    selected_tools: List[str]
    constraints: Dict[str, Any]
    constraint_layers: Optional[Dict[str, Any]] = None
    search_query: str = ""
    planner_metadata: Optional[Dict[str, Any]] = None


def _normalize_compact(text: str) -> str:
    return " ".join(str(text or "").split()).upper().replace(" ", "")


def _looks_like_gene_alteration(text: str) -> bool:
    compact = _normalize_compact(text)
    if len(compact) < 3 or compact in GENE_ALTERATION_NOISE:
        return False
    if compact in COMMON_GENE_SYMBOLS:
        return True
    for symbol in COMMON_GENE_SYMBOLS:
        if compact.startswith(symbol) and len(compact) > len(symbol):
            tail = compact[len(symbol):]
            if not tail:
                return True
            if re.search(r"(EXON\d+|\d+DEL|\d+DELETION|\d+INS|\d+DUP|\d+AMP|AMPLIFICATION|FUSION|REARRANGEMENT|MUTATION|L858R|T790M|C797S|G719X|S768I)", tail):
                return True
            if re.search(r"\d", tail):
                return True
    if re.fullmatch(r"[A-Z0-9]{3,12}", compact) and re.search(r"\d", compact):
        return True
    return False


def _unique_terms(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


class RetrievalRouter:
    # Match the LLM reranker's one-call candidate pool.  Larger recall pools are
    # still available inside hybrid_search, but are cut by normalized retrieval
    # score before any text is sent to the LLM.
    HYBRID_RERANK_POOL = 64

    def __init__(self, retrieval_tools: RetrievalTools, llm_client: Optional["LLMClient"] = None, use_llm: bool = True) -> None:
        self.tools = retrieval_tools
        self.llm_client = llm_client
        self.use_llm = use_llm
        self.safety_gate = ClinicalSafetyGate(llm_client=llm_client, use_llm=use_llm)

    def _rerank_hybrid_hits(self, query: str, hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        pool_size = max(top_k, self.HYBRID_RERANK_POOL)
        pool = hits[:pool_size]
        return self.tools.rerank(query=query, hits=pool, top_k=top_k)

    @staticmethod
    def _compact_planner_context(context: Dict[str, Any]) -> Dict[str, Any]:
        compact = dict(context)
        rounds = []
        for memory in (context.get("round_memories") or [])[-3:]:
            if not isinstance(memory, dict):
                continue
            row = {key: memory.get(key) for key in (
                "round", "plan_step_id", "step_goal", "retrieval_actions",
                "question_information_gain", "goal_evaluation",
            )}
            rounds.append(row)
        compact["round_memories"] = rounds
        answer = context.get("answer_memory") or {}
        compact["answer_memory"] = {
            "claims": answer.get("claims") or [],
            "informative_rounds": answer.get("informative_rounds") or [],
            "evidence_chunk_ids": list((answer.get("evidence_by_id") or {}).keys())[-30:],
        }
        return compact

    def _search_with_tools(self, query: str, constraints: Dict[str, Any], selected_tools: List[str], top_k: int) -> List[SearchHit]:
        collected: List[SearchHit] = []
        if "structured_search" in selected_tools:
            collected.extend(self.tools.structured_search(constraints, top_k=top_k))
        if "dense_search" in selected_tools:
            collected.extend(self.tools.dense_search(query=query, top_k=top_k))
        if "bm25_search" in selected_tools:
            collected.extend(self.tools.bm25_search(query=query, top_k=top_k))
        if "hybrid_search" in selected_tools:
            rerank_pool = max(top_k, self.HYBRID_RERANK_POOL)
            hybrid_hits = self.tools.hybrid_search(query=query, constraints=constraints, top_k=rerank_pool)
            collected.extend(self._rerank_hybrid_hits(query=query, hits=hybrid_hits, top_k=top_k))
        return collected

    def _dedupe_hits(self, hits: List[SearchHit]) -> List[SearchHit]:
        merged: Dict[str, SearchHit] = {}
        for hit in hits:
            if hit.id not in merged:
                merged[hit.id] = hit
                continue
            existing = merged[hit.id]
            if hit.score > existing.score:
                merged[hit.id] = hit
            else:
                existing.metadata.update(hit.metadata)
                if len(hit.text) > len(existing.text):
                    existing.text = hit.text
        deduped = list(merged.values())
        deduped.sort(key=lambda item: item.score, reverse=True)
        return deduped

    def _collect_chunk_ids(self, hits: List[SearchHit], top_k: int) -> List[str]:
        chunk_ids: List[str] = []
        seen = set()
        for hit in hits:
            if "#chunk-" not in hit.id:
                continue
            if hit.id in seen:
                continue
            seen.add(hit.id)
            chunk_ids.append(hit.id)
            if len(chunk_ids) >= max(top_k, 15):
                break
        return chunk_ids

    def _extract_constraint_layers(self, query: str) -> Dict[str, Any]:
        hard_constraints = self.extract_constraints(query)
        lower = query.lower()
        soft_hints: Dict[str, List[str]] = {
            "clinical_terms": [],
            "retrieval_focus": [],
        }
        expansion_hints: List[str] = []

        clinical_terms = []
        for token in ("progression", "progressed", "resistance", "resistant", "after", "following", "subsequent", "post", "switch", "line", "toxicity", "adverse", "response", "benefit"):
            if token in lower:
                clinical_terms.append(token)
        if re.search(r"(进展|耐药|失败|后续|换药|毒性|不良反应|获益|缓解|相似|类似|同类|机制|原因)", query):
            clinical_terms.extend(re.findall(r"(进展|耐药|失败|后续|换药|毒性|不良反应|获益|缓解|相似|类似|同类|机制|原因)", query))
        soft_hints["clinical_terms"] = _unique_terms(clinical_terms)

        focus_terms: List[str] = []
        if hard_constraints.get("drugs"):
            focus_terms.extend([f"drug:{drug}" for drug in hard_constraints["drugs"]])
        if hard_constraints.get("gene_alterations"):
            focus_terms.extend([f"gene:{gene}" for gene in hard_constraints["gene_alterations"]])
        if hard_constraints.get("responses"):
            focus_terms.extend([f"response:{resp}" for resp in hard_constraints["responses"]])
        if hard_constraints.get("ddi_terms"):
            focus_terms.extend([f"ddi:{term}" for term in hard_constraints["ddi_terms"]])
        soft_hints["retrieval_focus"] = _unique_terms(focus_terms)

        if any(term in lower for term in ("similar", "similar case", "相似", "类似", "同类", "同靶点", "同通路")):
            expansion_hints.append("broaden_to_similar_case")
        if any(term in lower for term in ("resistance", "resistant", "耐药", "进展", "progression", "progressed")):
            expansion_hints.append("broaden_to_resistance_mechanism")
        if any(term in lower for term in ("mechanism", "机制", "pathway", "原因")):
            expansion_hints.append("broaden_to_mechanism")
        if hard_constraints.get("drugs") and any(term in lower for term in ("同类", "类似", "同靶点", "同通路")):
            expansion_hints.extend([f"same_class_drug:{drug}" for drug in hard_constraints["drugs"]])
        if hard_constraints.get("gene_alterations") and any(term in lower for term in ("同类", "类似", "同通路")):
            expansion_hints.extend([f"same_family_gene:{gene}" for gene in hard_constraints["gene_alterations"]])

        return {
            "hard_constraints": hard_constraints,
            "soft_hints": soft_hints,
            "expansion_hints": _unique_terms(expansion_hints),
        }

    def understand_query(
        self,
        query: str,
        planner_context: Optional[Dict[str, Any]] = None,
        main_question: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.use_llm or self.llm_client is None:
            raise RuntimeError("QueryUnderstanding requires an LLM; category/rule fallback is disabled")
        # Initial understanding is intentionally a pure function of the
        # untouched user question. Planning memory belongs to Planner.
        prompt = QUERY_UNDERSTANDING_USER_TEMPLATE.format(query=query)
        payload: Optional[Dict[str, Any]] = None
        last_output_error: Optional[Exception] = None
        for attempt in range(3):
            retry_instruction = ""
            if attempt:
                retry_instruction = (
                    "\n\n上一次输出无法解析。请重新输出一个完整、合法的 JSON 对象；"
                    "不要使用 Markdown 代码块，不要添加 JSON 之外的文字；"
                    "不要通过删除治疗时间线、随访时点或因果边界来缩短输出。"
                )
            try:
                raw = self.llm_client.chat(
                    system=QUERY_UNDERSTANDING_SYSTEM_PROMPT,
                    user=prompt + retry_instruction,
                    temperature=0.1,
                    max_output_tokens=5000,
                )
            except Exception as exc:
                # An empty model output is the same recoverable whole-object
                # failure as malformed JSON. Transport/auth/model errors must
                # still escape to the case-level retry policy.
                if "empty content" not in str(exc).lower():
                    raise
                last_output_error = exc
                continue
            parsed = _parse_json_object(raw)
            if isinstance(parsed, dict):
                payload = parsed
                break
            last_output_error = ValueError("QueryUnderstanding output was not valid JSON")
        if payload is None:
            raise ValueError("QueryUnderstanding did not return one JSON object after 3 attempts") from last_output_error
        patient_facts = payload.get("patient_facts") if isinstance(payload.get("patient_facts"), dict) else {}
        treatment_context = payload.get("treatment_context") if isinstance(payload.get("treatment_context"), dict) else {}
        target_outcomes = payload.get("target_outcomes") if isinstance(payload.get("target_outcomes"), list) else []
        information_needs = payload.get("information_needs") if isinstance(payload.get("information_needs"), list) else []
        ambiguities = payload.get("ambiguities") if isinstance(payload.get("ambiguities"), list) else []
        normalized = {
            "patient_facts": dict(patient_facts),
            "treatment_context": dict(treatment_context),
            "target_outcomes": target_outcomes,
            "information_needs": _unique_terms([str(item) for item in information_needs if str(item).strip()])[:5],
            "ambiguities": _unique_terms([str(item) for item in ambiguities if str(item).strip()])[:3],
            "additional_context": payload.get("additional_context") if isinstance(payload.get("additional_context"), dict) else {},
        }
        # Soft schema: preserve useful model extensions, but never leak the
        # retired retrieval-adapter fields from a malformed/legacy response.
        retired_fields = {"query_type", "hard_constraints", "soft_hints", "ambiguity_notes"}
        for key, value in payload.items():
            if key not in normalized and key not in retired_fields:
                normalized[key] = value
        return normalized

    @staticmethod
    def adapt_problem_representation(problem: Dict[str, Any]) -> Dict[str, Any]:
        """Mechanically adapt understood facts to current retrieval interfaces."""
        patient_facts = problem.get("patient_facts") if isinstance(problem.get("patient_facts"), dict) else {}
        treatment_context = problem.get("treatment_context") if isinstance(problem.get("treatment_context"), dict) else {}
        target_outcomes = problem.get("target_outcomes") if isinstance(problem.get("target_outcomes"), list) else []
        information_needs = problem.get("information_needs") if isinstance(problem.get("information_needs"), list) else []

        def string_values(value: Any) -> List[str]:
            values = value if isinstance(value, list) else ([] if value in (None, "") else [value])
            result: List[str] = []
            for item in values:
                if isinstance(item, dict):
                    preferred = item.get("name") or item.get("drug") or item.get("regimen") or item.get("value")
                    text = str(preferred or json.dumps(item, ensure_ascii=False, separators=(",", ":"))).strip()
                else:
                    text = str(item).strip()
                if text:
                    result.append(text)
            return _unique_terms(result)

        def observed_values(value: Any, preferred_keys: tuple[str, ...]) -> List[str]:
            values = value if isinstance(value, list) else ([] if value in (None, "") else [value])
            result: List[str] = []
            for item in values:
                if isinstance(item, dict):
                    preferred = next((item.get(key) for key in preferred_keys if item.get(key)), None)
                    text = str(preferred or "").strip()
                else:
                    text = str(item).strip()
                if text:
                    result.append(text)
            return _unique_terms(result)

        current_treatments = string_values(
            treatment_context.get("current_treatments")
            or treatment_context.get("index_treatment")
            or treatment_context.get("current_drugs")
        )
        prior_treatments = string_values(treatment_context.get("prior_treatments"))
        regimen_details = string_values(treatment_context.get("regimen_details"))
        clinical_terms = _unique_terms([
            *[f"既往治疗:{item}" for item in prior_treatments],
            *regimen_details,
        ])
        return {
            "hard_constraints": {
                "cancer_type": str(patient_facts.get("cancer_type") or patient_facts.get("diagnosis") or ""),
                "gene_alterations": string_values(patient_facts.get("gene_alterations") or patient_facts.get("molecular_profile")),
                "drugs": current_treatments,
                "responses": observed_values(patient_facts.get("observed_responses") or patient_facts.get("responses"), ("response", "best_response", "value")),
                "toxicities": observed_values(patient_facts.get("observed_toxicities") or patient_facts.get("toxicities"), ("toxicity", "event", "value")),
                "metastatic_sites": string_values(patient_facts.get("metastatic_sites")),
                "ddi_terms": string_values(patient_facts.get("ddi_terms") or patient_facts.get("interaction_terms")),
            },
            "soft_hints": {
                "clinical_terms": clinical_terms,
                "retrieval_focus": _unique_terms([
                    *string_values(target_outcomes),
                    *[str(item) for item in information_needs if str(item).strip()],
                ]),
            },
        }

    def _detect_signals(self, text: str) -> Dict[str, bool]:
        signal_text = text or ""
        is_supporting = any(pattern.search(signal_text) for pattern in SUPPORTING_PATTERNS)
        is_contradicting = any(pattern.search(signal_text) for pattern in CONTRADICTING_PATTERNS)
        is_safety_risk = any(pattern.search(signal_text) for pattern in SAFETY_RISK_PATTERNS)
        return {
            "is_supporting": is_supporting,
            "is_contradicting": is_contradicting,
            "is_safety_risk": is_safety_risk,
        }

    def _assess_evidence(self, evidence_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        layers: Dict[str, Dict[str, Any]] = {}
        support_count = 0
        contradiction_count = 0
        safety_risk_count = 0
        assessed_items: List[Dict[str, Any]] = []
        for row in evidence_rows:
            item = dict(row)
            level = str(item.get("evidence_level") or "case_report_evidence")
            signals = self._detect_signals(str(item.get("text") or ""))
            item["relevance_signals"] = signals
            assessed_items.append(item)

            if signals["is_supporting"]:
                support_count += 1
            if signals["is_contradicting"]:
                contradiction_count += 1
            if signals["is_safety_risk"]:
                safety_risk_count += 1

            layer = layers.setdefault(
                level,
                {
                    "count": 0,
                    "priority": EVIDENCE_LAYER_PRIORITY.get(level, 0),
                    "sample_chunk_ids": [],
                },
            )
            layer["count"] += 1
            if item.get("chunk_id") and len(layer["sample_chunk_ids"]) < 3:
                layer["sample_chunk_ids"].append(item["chunk_id"])

        ordered_layers = sorted(layers.items(), key=lambda pair: (pair[1]["priority"], pair[1]["count"]), reverse=True)
        return {
            "layer_distribution": [
                {
                    "evidence_level": level,
                    "count": info["count"],
                    "priority": info["priority"],
                    "sample_chunk_ids": info["sample_chunk_ids"],
                }
                for level, info in ordered_layers
            ],
            "support_count": support_count,
            "contradiction_count": contradiction_count,
            "safety_risk_count": safety_risk_count,
            "assessed_items": assessed_items,
        }

    def _build_counter_evidence_query(self, query: str, query_type: Optional[str] = None) -> str:
        hint = COUNTER_EVIDENCE_HINTS["general"]
        return f"{query} {hint}"

    def _contradiction_check(
        self,
        support_hits: List[SearchHit],
        counter_hits: List[SearchHit],
        evidence_assessment: Dict[str, Any],
    ) -> Dict[str, Any]:
        support_count = evidence_assessment.get("support_count", 0)
        contradiction_count = evidence_assessment.get("contradiction_count", 0)
        safety_risk_count = evidence_assessment.get("safety_risk_count", 0)
        if contradiction_count == 0 and safety_risk_count == 0 and support_count == 0:
            verdict = "insufficient_evidence"
        elif contradiction_count + safety_risk_count > max(1, support_count):
            verdict = "needs_caution_due_to_contradiction_or_risk"
        elif support_count == 0 and (counter_hits or contradiction_count > 0):
            verdict = "not_supported"
        else:
            verdict = "supported_with_monitoring"

        return {
            "verdict": verdict,
            "supporting_hit_count": len(support_hits),
            "counter_hit_count": len(counter_hits),
            "supporting_evidence_signals": support_count,
            "contradicting_evidence_signals": contradiction_count,
            "safety_risk_signals": safety_risk_count,
            "top_supporting_hit_ids": [hit.id for hit in support_hits[:5]],
            "top_counter_hit_ids": [hit.id for hit in counter_hits[:5]],
        }

    def extract_constraints(self, query: str) -> Dict[str, Any]:
        constraints: Dict[str, Any] = {
            "gene_alterations": [],
            "drugs": [],
            "responses": [],
            "toxicities": [],
            "metastatic_sites": [],
            "ddi_terms": [],
        }
        lower = query.lower()
        for token in ["luad", "nsclc", "sclc", "lung adenocarcinoma"]:
            if token in lower:
                constraints["cancer_type"] = token.upper() if token in {"luad", "nsclc", "sclc"} else token
                break
        for token in ["pr", "pd", "sd", "cr"]:
            if re.search(rf"\b{token}\b", lower):
                constraints["responses"].append(token.upper())
        mutation_matches = re.findall(
            r"\b[A-Z]{2,8}\s*(?:exon\s*\d+\s*(?:del|deletion|ins)?|(?:\d+|[A-Z])\d+[A-Z]?|[A-Z]?\d+[A-Z]?|del(?:etion)?|ins|dup|amp|amplification|fusion|rearrangement|mutation)?\b",
            query,
        )
        constraints["gene_alterations"].extend(
            [match.strip() for match in mutation_matches if match.strip() and _looks_like_gene_alteration(match)]
        )
        for drug in ["osimertinib", "gefitinib", "afatinib", "alectinib", "crizotinib", "temozolomide", "olaparib", "atezolizumab"]:
            if drug in lower:
                constraints["drugs"].append(drug)
        if "ddi" in lower or "interaction" in lower or "相互作用" in query:
            constraints["ddi_terms"].append("drug interaction")
        return constraints

    @staticmethod
    def _merge_explicit_constraints(
        fallback: Dict[str, Any], proposed: Dict[str, Any], source_text: str
    ) -> Dict[str, Any]:
        """Keep LLM-extracted hard constraints only when literally grounded."""
        merged = dict(fallback)
        normalized_source = " ".join(source_text.lower().split())
        for key in ("gene_alterations", "drugs", "responses", "toxicities", "metastatic_sites", "ddi_terms"):
            current = list(merged.get(key) or [])
            for value in proposed.get(key) or []:
                item = str(value).strip()
                if not item or item in current:
                    continue
                if key == "gene_alterations" and not _looks_like_gene_alteration(item):
                    continue
                if " ".join(item.lower().split()) in normalized_source:
                    current.append(item)
            merged[key] = current
        cancer_type = str(proposed.get("cancer_type") or "").strip()
        if cancer_type and " ".join(cancer_type.lower().split()) in normalized_source:
            merged["cancer_type"] = cancer_type
        return merged

    def plan(self, query: str, planner_context: Optional[Dict[str, Any]] = None, main_question: Optional[str] = None) -> RoutePlan:
        understanding = self.understand_query(query, planner_context=planner_context, main_question=main_question)
        retrieval_adapter = self.adapt_problem_representation(understanding)
        if not self.use_llm or self.llm_client is None:
            raise RuntimeError("Planner requires an LLM; category/rule fallback is disabled")
        try:
            prompt = PLANNER_USER_TEMPLATE.format(
                main_question=main_question or query,
                query=query,
                planner_context=json.dumps(
                    {
                        **self._compact_planner_context(planner_context or {}),
                        "query_understanding": understanding,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            payload = self.llm_client.call_function(
                system=PLANNER_SYSTEM_PROMPT,
                user=prompt,
                function_name="submit_retrieval_plan",
                description="Submit the next retrieval plan for the clinical question.",
                parameters=PLANNER_SCHEMA,
                temperature=0.2,
                max_output_tokens=8000,
            )
            tools = [str(item) for item in (payload.get("selected_tools") or []) if str(item) in ALLOWED_TOOLS]
            if not tools:
                raise ValueError("Planner returned no valid retrieval tool")
            if any(tool in tools for tool in {"structured_search", "dense_search", "bm25_search", "hybrid_search"}) and "fetch_evidence" not in tools:
                tools.append("fetch_evidence")
            constraints = dict(retrieval_adapter.get("hard_constraints") or {})
            if isinstance(payload.get("constraints"), dict):
                constraints = self._merge_explicit_constraints(
                    constraints,
                    payload["constraints"],
                    f"{main_question or query} {query}",
                )
            plan = RoutePlan(
                query_type=None,
                selected_tools=tools,
                constraints=constraints,
                constraint_layers=understanding,
            )
            plan.search_query = str(payload.get("search_query") or query).strip() or query  # type: ignore[attr-defined]
            plan.planner_metadata = {  # type: ignore[attr-defined]
                key: payload.get(key) for key in ("plan_rationale", "targeted_information_gain", "avoid_repeating", "stop_if")
            }
            if plan.constraint_layers is None:
                plan.constraint_layers = understanding
            return plan
        except Exception:
            raise

    def run(
        self,
        query: str,
        top_k: int = 10,
        planner_context: Optional[Dict[str, Any]] = None,
        main_question: Optional[str] = None,
        route_plan: Optional[RoutePlan] = None,
        rerank_goal: Optional[str] = None,
    ) -> Dict[str, Any]:
        plan = route_plan or self.plan(query, planner_context=planner_context, main_question=main_question)
        execution_query = getattr(plan, "search_query", "") or query
        ranking_query = (rerank_goal or execution_query).strip()
        results: Dict[str, Any] = {
            "query_type": plan.query_type,
            "selected_tools": plan.selected_tools,
            "constraints": plan.constraints,
            "constraint_layers": plan.constraint_layers or {},
            "query_understanding": plan.constraint_layers or {},
            "planner_metadata": getattr(plan, "planner_metadata", {}),
            "planned_query": execution_query,
            "rerank_goal": ranking_query,
            "results": {},
        }
        collected: List[SearchHit] = []

        if "structured_search" in plan.selected_tools:
            structured = self.tools.structured_search(plan.constraints, top_k=top_k)
            results["results"]["structured_search"] = [hit.__dict__ for hit in structured]
            collected.extend(structured)

        if "dense_search" in plan.selected_tools:
            dense = self.tools.dense_search(query=execution_query, top_k=top_k)
            results["results"]["dense_search"] = [hit.__dict__ for hit in dense]
            collected.extend(dense)

        if "bm25_search" in plan.selected_tools:
            bm25 = self.tools.bm25_search(query=execution_query, top_k=top_k)
            results["results"]["bm25_search"] = [hit.__dict__ for hit in bm25]
            collected.extend(bm25)

        if "hybrid_search" in plan.selected_tools:
            rerank_pool = max(top_k, self.HYBRID_RERANK_POOL)
            hybrid = self.tools.hybrid_search(query=execution_query, constraints=plan.constraints, top_k=rerank_pool)
            hybrid = self._rerank_hybrid_hits(query=ranking_query, hits=hybrid, top_k=top_k)
            results["results"]["hybrid_search"] = [hit.__dict__ for hit in hybrid]
            overflow = getattr(self.tools, "last_qwen_overflow", [])
            if overflow:
                results["qwen_overflow_hits"] = [hit.__dict__ for hit in overflow]
            knowledge = getattr(self.tools, "last_knowledge_hits", [])
            if knowledge:
                results["knowledge_context"] = [hit.__dict__ for hit in knowledge]
            results["rerank_backend"] = getattr(self.tools, "last_rerank_backend", "none")
            collected.extend(hybrid)

        deduped_support_hits = self._dedupe_hits(collected)

        counter_query = self._build_counter_evidence_query(query=execution_query, query_type=plan.query_type)
        counter_selected_tools = [tool for tool in plan.selected_tools if tool in {"dense_search", "bm25_search", "hybrid_search"}]
        if not counter_selected_tools:
            counter_selected_tools = ["bm25_search", "dense_search"]
        counter_hits = self._search_with_tools(
            query=counter_query,
            constraints=plan.constraints,
            selected_tools=counter_selected_tools,
            top_k=top_k,
        )
        deduped_counter_hits = self._dedupe_hits(counter_hits)
        results["results"]["counter_evidence_search"] = {
            "query": counter_query,
            "selected_tools": counter_selected_tools,
            "hits": [hit.__dict__ for hit in deduped_counter_hits[:top_k]],
        }

        if "fetch_evidence" in plan.selected_tools:
            # prioritize fetching both supporting and counter-evidence chunks.
            chunk_ids = self._collect_chunk_ids(deduped_support_hits + deduped_counter_hits, top_k=top_k)
            evidence = self.tools.fetch_evidence(chunk_ids=chunk_ids, limit=top_k)
            results["results"]["fetch_evidence"] = evidence
            evidence_assessment = self._assess_evidence(evidence_rows=evidence)
        else:
            evidence_assessment = self._assess_evidence(evidence_rows=[])

        results["evidence_layering"] = {
            "query": ranking_query,
            "layer_distribution": evidence_assessment["layer_distribution"],
            "assessed_evidence": evidence_assessment["assessed_items"],
        }
        results["contradiction_check"] = self._contradiction_check(
            support_hits=deduped_support_hits,
            counter_hits=deduped_counter_hits,
            evidence_assessment=evidence_assessment,
        )
        results["clinical_safety_gate"] = self.safety_gate.evaluate(
            query=query,
            query_type=plan.query_type,
            constraints=plan.constraints,
            retrieval_results=results["results"],
        )

        return results
