from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from searchagent_retrieval.llm_client import LLMClient
from searchagent_retrieval.llm_rerank import LLMReranker
from searchagent_retrieval.tools import RetrievalTools, SearchHit


CATEGORIES = {
    "direct_lm": ("LM", "120 mg"),
    "toxicity": ("ALT/AST", "dose-modification"),
    "pk": ("CSF:plasma", "PK"),
    "subgroup": ("19del", "ORR/DCR"),
    "resistance": ("C797S", "T790M"),
    "hematotoxicity": ("血液毒性", "剂量调整"),
    "cns_analog": ("BLOOM", "CSF"),
    "intrathecal_safety": ("intrathecal pemetrexed 40 mg", "grade ≥3"),
}


def _historical_steps(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen = set()
    for path in sorted(root.glob("**/session/*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for step in payload.get("loop_steps") or []:
            if not isinstance(step, dict):
                continue
            query = str(step.get("query") or "").strip()
            goal = str(step.get("rerank_goal") or "").strip()
            tools = step.get("selected_tools") or []
            key = (query, goal)
            if query and goal and "hybrid_search" in tools and key not in seen:
                seen.add(key)
                rows.append({"source": str(path), "query": query, "rerank_goal": goal})
    return rows


def _choose(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen = []
    used = set()
    for category, needles in CATEGORIES.items():
        match = next(
            (
                row for row in rows
                if id(row) not in used
                and all(needle.lower() in row["rerank_goal"].lower() for needle in needles)
            ),
            None,
        )
        if match:
            used.add(id(match))
            chosen.append({"category": category, **match})
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--index", type=Path, default=Path("indexes"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=16)
    args = parser.parse_args()

    selected = _choose(_historical_steps(args.root))
    retrieval = RetrievalTools(args.index, reranker_backend="none")
    llm = LLMClient()
    reranker = LLMReranker(llm)
    output: dict[str, Any] = {"pool_size": args.pool_size, "groups": []}
    try:
        for index, row in enumerate(selected, 1):
            hybrid = retrieval.hybrid_search(row["query"], constraints={}, top_k=args.pool_size)
            llm_ranked = reranker.rerank(row["rerank_goal"], hybrid, args.pool_size)
            by_id = {hit.id: hit for hit in llm_ranked}
            candidates = []
            for hit in hybrid:
                ranked = by_id.get(hit.id)
                metadata = dict((ranked or hit).metadata)
                candidates.append({
                    "id": hit.id,
                    "text": hit.text,
                    "hybrid_score": hit.score,
                    "llm_score": ranked.score if ranked else None,
                    "llm_status": metadata.get("llm_rerank_status"),
                    "llm_evidence_type": metadata.get("llm_evidence_type"),
                    "llm_match_status": metadata.get("llm_match_status"),
                    "llm_hard_mismatches": metadata.get("llm_hard_mismatches") or [],
                    "llm_dimensions": metadata.get("llm_rerank_dimensions") or {},
                    "llm_reason": metadata.get("llm_rerank_reason"),
                })
            output["groups"].append({**row, "candidates": candidates})
            args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))
            print(f"completed {index}/{len(selected)} {row['category']}", flush=True)
    finally:
        retrieval.close()


if __name__ == "__main__":
    main()
