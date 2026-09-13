from __future__ import annotations

import json
import time

from experiments.ablate_evidence_review_step_memory import FIXTURES, load_env, route_result
from searchagent_retrieval.agent_loop import EvidenceReviewAgent
from searchagent_retrieval.llm_client import LLMClient


def main(repeats: int = 2) -> None:
    load_env()
    llm = LLMClient(timeout=300, max_retries=1, structured_attempts=1)
    rows = []
    for repeat in range(1, repeats + 1):
        for fixture in FIXTURES:
            started = time.perf_counter()
            review = EvidenceReviewAgent(llm_client=llm, use_llm=True)._review_with_llm(
                route_result(fixture)
            )
            items = []
            seen = set()
            for group in ("supporting_evidence", "contradicting_evidence", "safety_risks", "insufficient_evidence"):
                for item in review.get(group) or []:
                    if item.get("chunk_id") not in seen:
                        seen.add(item.get("chunk_id"))
                        items.append(item)
            text = json.dumps(items, ensure_ascii=False).lower()
            expected = fixture["expected"]
            rows.append({
                "repeat": repeat,
                "fixture": fixture["name"],
                "seconds": round(time.perf_counter() - started, 3),
                "verdict": review.get("verdict"),
                "overall_roles": [item.get("overall_role") for item in items],
                "dimension_count": sum(len(item.get("dimension_findings") or []) for item in items),
                "preserved": {term: term.lower() in text for term in expected["must_preserve"]},
                "forbidden_inferred": {term: term.lower() in text for term in expected["must_not_infer"]},
                "claim_scopes": [item.get("claim_scope") for item in items],
            })
    print(json.dumps({"experiment": "soft_schema_repeat", "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
