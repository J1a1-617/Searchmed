from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from searchagent_retrieval.agent_loop import EvidenceReviewAgent, StepMemoryAgent
from searchagent_retrieval.llm_client import LLMClient


def load_env() -> None:
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


FIXTURES = [
    {
        "name": "cross_drug_hepatotoxicity",
        "question": "伏美替尼120mg再挑战后8-12周是否发生肝损伤并导致减量或停药？",
        "goal": "判断目标方案早期肝毒性及是否导致治疗中断",
        "chunk_id": "capmatinib-hepatotoxicity",
        "text": (
            "A patient receiving capmatinib developed severe liver toxicity. "
            "Capmatinib was discontinued and liver tests gradually improved."
        ),
        "expected": {
            "must_preserve": ["capmatinib", "liver"],
            "must_not_infer": ["伏美替尼", "120mg", "8-12周"],
            "expected_role": ["analog_support", "partial_support", "risk"],
        },
    },
    {
        "name": "organ_specific_mixed_outcome",
        "question": "奥希替尼80mg单药后8-12周，脑膜转移和全身病灶是否总体获益？",
        "goal": "分别判断早期CNS和全身疗效，不能混成总体反应",
        "chunk_id": "mixed-cns-systemic",
        "text": (
            "After osimertinib plus capmatinib, chest lesions shrank at three months, "
            "but neurologic symptoms progressed because leptomeningeal disease worsened."
        ),
        "expected": {
            "must_preserve": ["chest", "leptomeningeal"],
            "must_not_infer": ["总体PR", "总体获益", "奥希替尼80mg单药有效"],
            "expected_role": ["partial_support", "analog_support", "counter"],
        },
    },
    {
        "name": "multi_outcome_toxicity_and_response",
        "question": "奥希替尼联合化疗后是否同时改善CNS和全身病灶，且未导致停药？",
        "goal": "分别审核CNS、全身疗效和停药安全性三个结局维度",
        "chunk_id": "mixed-response-toxicity",
        "text": (
            "After osimertinib plus chemotherapy, extracranial lesions achieved a partial response, "
            "while leptomeningeal disease progressed. Grade 3 thrombocytopenia required temporary "
            "treatment interruption, followed by dose reduction."
        ),
        "expected": {
            "must_preserve": ["extracranial", "leptomeningeal", "thrombocytopenia", "interruption", "dose reduction"],
            "must_not_infer": ["全身总体获益", "CNS获益", "未导致停药"],
            "expected_role": ["mixed"],
        },
    },
]


def route_result(fixture: Dict[str, Any]) -> Dict[str, Any]:
    item = {
        "chunk_id": fixture["chunk_id"],
        "text": fixture["text"],
        "evidence_level": "case_report_evidence",
        "relevance_signals": {
            "is_supporting": True,
            "is_contradicting": False,
            "is_safety_risk": fixture["name"] == "cross_drug_hepatotoxicity",
        },
    }
    return {
        "query": fixture["question"],
        "selected_tools": ["dense_search", "fetch_evidence"],
        "constraints": {},
        "rerank_goal": fixture["goal"],
        "main_question": fixture["question"],
        "problem_representation": {
            "patient_facts": [fixture["question"]],
            "target_outcomes": [fixture["goal"]],
        },
        "plan_step": {
            "step_id": "S1",
            "goal": fixture["goal"],
            "rerank_goal": fixture["goal"],
            "evidence_lane": "direct_case",
        },
        "planned_query": fixture["question"],
        "evidence_layering": {
            "query": fixture["question"],
            "assessed_evidence": [item],
            "layer_distribution": [],
        },
        "contradiction_check": {"verdict": "supported_with_monitoring"},
        "step_execution_report": {
            "execution_status": "success",
            "accepted_evidence_ids": [fixture["chunk_id"]],
            "query_database_status": "uncertain",
            "queries_attempted": [fixture["question"]],
            "goal_evaluation": {
                "matched_goal_count": 1,
                "success_criteria_met": False,
                "observed_gaps": [],
            },
        },
        "tool_trace": [],
    }


def no_review_payload(fixture: Dict[str, Any]) -> Dict[str, Any]:
    # C intentionally supplies only the raw candidate and relevance flags.
    item = route_result(fixture)["evidence_layering"]["assessed_evidence"][0]
    return {
        "verdict": "partially_supports",
        "supporting_evidence": [item],
        "contradicting_evidence": [],
        "safety_risks": [],
        "insufficient_evidence": [],
        "layer_distribution": [],
    }


def compact(memory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "memory_mode": memory.get("memory_mode"),
        "accepted_evidence": [
            {
                key: row.get(key)
                for key in (
                    "chunk_id", "evidence_role", "target_entity_match",
                    "supports_dimensions", "mismatched_dimensions",
                    "unreported_dimensions", "claim_scope", "memory_reason",
                )
            }
            for row in memory.get("accepted_evidence") or []
        ],
        "rejected_evidence": memory.get("rejected_evidence") or [],
        "goal_evaluation": memory.get("goal_evaluation") or {},
    }


def run_step(
    *,
    fixture: Dict[str, Any],
    review: Dict[str, Any],
    llm: LLMClient | None,
    use_llm: bool,
) -> tuple[Dict[str, Any], float, int]:
    before_calls = llm.calls_used() if llm else 0
    started = time.perf_counter()
    result = StepMemoryAgent(llm_client=llm, use_llm=use_llm).maintain(
        round_number=1,
        main_question=fixture["question"],
        query=fixture["question"],
        route_result=route_result(fixture),
        evidence_review=review,
        seen_chunk_ids=set(),
        plan_step={
            "step_id": "S1",
            "goal": fixture["goal"],
            "rerank_goal": fixture["goal"],
            "evidence_lane": "direct_case",
            "success_criteria": ["one bounded evidence item"],
        },
    )
    after_calls = llm.calls_used() if llm else 0
    return compact(result), round(time.perf_counter() - started, 3), after_calls - before_calls


def main() -> None:
    load_env()
    llm = LLMClient(timeout=300, max_retries=1, structured_attempts=1)
    rows = []
    for fixture in FIXTURES:
        llm.begin_trace(f"{fixture['name']}-review")
        started = time.perf_counter()
        review = EvidenceReviewAgent(llm_client=llm, use_llm=True)._review_with_llm(
            route_result(fixture)
        )
        review_seconds = round(time.perf_counter() - started, 3)
        review_calls = llm.calls_used()

        a_memory, a_step_seconds, a_step_calls = run_step(
            fixture=fixture, review=review, llm=llm, use_llm=True,
        )
        rows.append({
            "fixture": fixture["name"], "group": "A_review_plus_llm_step",
            "expected": fixture["expected"],
            "review": review, "memory": a_memory,
            "llm_calls": review_calls + a_step_calls,
            "seconds": round(review_seconds + a_step_seconds, 3),
        })

        b_memory, b_step_seconds, b_step_calls = run_step(
            fixture=fixture, review=review, llm=None, use_llm=False,
        )
        rows.append({
            "fixture": fixture["name"], "group": "B_review_plus_rule_step",
            "expected": fixture["expected"],
            "review": review, "memory": b_memory,
            "llm_calls": review_calls + b_step_calls,
            "seconds": round(review_seconds + b_step_seconds, 3),
        })

        llm.begin_trace(f"{fixture['name']}-no-review")
        c_memory, c_seconds, c_calls = run_step(
            fixture=fixture, review=no_review_payload(fixture), llm=llm, use_llm=True,
        )
        rows.append({
            "fixture": fixture["name"], "group": "C_no_review_plus_llm_step",
            "expected": fixture["expected"],
            "review": None, "memory": c_memory,
            "llm_calls": c_calls, "seconds": c_seconds,
        })
    print(json.dumps({"experiment": "evidence_review_step_memory_ablation", "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
