from __future__ import annotations

import json
import os
from pathlib import Path

from searchagent_retrieval.agent_loop import EvidenceReviewAgent, StepMemoryAgent
from searchagent_retrieval.llm_client import LLMClient


ROOT = Path(__file__).resolve().parents[1]
QUESTION = "EGFR 19del肺腺癌接受奥希替尼80mg单药后，8–12周的全身病灶疗效如何？"
TEXT = (
    "The patient received osimertinib 80 mg once daily as first-line therapy. "
    "Symptoms improved rapidly. A chest CT evaluation 12 weeks later showed a partial response, "
    "with the lung lesion decreasing to 38 x 22 mm. After 11 months of disease control, "
    "CT showed an enlarged lung mass and acquired progression."
)


def load_env() -> None:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    load_env()
    route = {
        "main_question": QUESTION,
        "problem_representation": {
            "patient_facts": ["EGFR 19del肺腺癌", "奥希替尼80mg单药"],
            "target_outcomes": ["8–12周全身病灶疗效"],
        },
        "query": "osimertinib 80 mg monotherapy EGFR lung adenocarcinoma early response",
        "planned_query": "osimertinib 80 mg monotherapy EGFR lung adenocarcinoma early response",
        "rerank_goal": "只评估8–12周全身病灶疗效；晚期耐药只限定持久性",
        "plan_step": {"step_id": "S1", "goal": "评估早期疗效", "rerank_goal": "8–12周全身病灶疗效", "evidence_lane": "direct_case"},
        "selected_tools": ["dense_search", "bm25_search", "fetch_evidence"],
        "constraints": {},
        "evidence_layering": {"query": "8–12周全身病灶疗效", "layer_distribution": [], "assessed_evidence": [{
            "chunk_id": "34532491#chunk-000004", "doc_id": "34532491", "text": TEXT,
            "evidence_level": "case_report_evidence",
            "relevance_signals": {"is_supporting": True, "is_contradicting": True, "is_safety_risk": False},
        }]},
        "contradiction_check": {},
        "step_execution_report": {"execution_status": "partial", "accepted_evidence_ids": ["34532491#chunk-000004"], "query_database_status": "uncertain", "queries_attempted": [], "goal_evaluation": {"matched_goal_count": 1, "success_criteria_met": False, "observed_gaps": []}},
        "tool_trace": [],
    }
    llm = LLMClient(timeout=300, max_retries=1, structured_attempts=1)
    review = EvidenceReviewAgent(llm_client=llm, use_llm=True)._review_with_llm(route)
    memory = StepMemoryAgent().maintain(round_number=1, main_question=QUESTION, query=route["query"], route_result=route, evidence_review=review, seen_chunk_ids=set(), plan_step=route["plan_step"])
    print(json.dumps({"review": review, "accepted_evidence": memory["accepted_evidence"], "counter_count": memory["question_information_gain"]["counter_count"], "support_count": memory["question_information_gain"]["support_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
