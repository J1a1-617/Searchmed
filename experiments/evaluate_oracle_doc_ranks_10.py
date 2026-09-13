#!/usr/bin/env python3
"""Audit first-round retrieval by the rank of the best available evidence document."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from searchagent_retrieval.tools import RetrievalTools


TRACE_ROOT = ROOT / "benchmark_results/searchagent_current10_20260729_artifacts/current10_20260729"

# These are corpus-oracle documents, selected by reading the complete Step1/2
# article, not by treating the retrieval output or benchmark ground truth as a
# relevance label. An empty list means the corpus has no document supporting
# the defining regimen/context combination.
ORACLES: dict[str, dict[str, Any]] = {
    "case_1_node_1": {"docs": ["40134592"], "availability": "partial", "note": "Icotinib→brain progression→osimertinib, but SBRT is added and no clean early monotherapy outcome."},
    "case_1_node_2": {"docs": ["38711856", "36188633"], "availability": "direct_counter", "note": "Contains LM and osimertinib 80→160 mg with lack of neurologic/CSF benefit before additional therapy."},
    "case_1_node_3": {"docs": [], "availability": "absent", "note": "No furmonertinib+AZD3759, C797S LM outcome document in the corpus."},
    "case_2_node_1": {"docs": [], "availability": "absent", "note": "No furmonertinib→aumolertinib switch after hepatotoxicity with CNS outcome."},
    "case_2_node_2": {"docs": [], "availability": "absent", "note": "No named FGFR inhibitor+pemetrexed/platinum CNS outcome matching this regimen."},
    "case_2_node_3": {"docs": ["41918647", "41836239"], "availability": "partial", "note": "Direct furmonertinib CNS/LM response cases, but doses/context differ from 120 mg monotherapy."},
    "case_2_node_4": {"docs": ["41918647", "41836239"], "availability": "partial", "note": "Efficacy analogs exist; matching 120 mg rechallenge hepatotoxicity evidence does not."},
    "case_3_node_1": {"docs": ["40463905"], "availability": "partial", "note": "EGFR-TKI response cohort including icotinib; not an exact 19del+TP53 Y220H early-response cohort."},
    "case_3_node_2": {"docs": ["41430146", "41466843"], "availability": "partial", "note": "Post-EGFR-TKI platinum/pemetrexed evidence exists, but not the exact short-course patient context."},
    "case_3_node_3": {"docs": [], "availability": "absent", "note": "No exact osimertinib+anlotinib post-gefitinib brain-metastasis outcome study; aumolertinib+anlotinib is only an analog."},
}


def doc_id(hit: Any) -> str:
    return str(hit.metadata.get("doc_id") or hit.id.split("#", 1)[0])


def ranks(hits: list[Any], targets: set[str]) -> dict[str, Any]:
    if not targets:
        return {"best_chunk_rank": None, "best_document_rank": None, "matched_doc": None}
    seen: set[str] = set()
    best_chunk = best_doc = None
    matched = None
    for chunk_rank, hit in enumerate(hits, 1):
        document = doc_id(hit)
        if document not in seen:
            seen.add(document)
            document_rank = len(seen)
        else:
            document_rank = None
        if document in targets:
            if best_chunk is None:
                best_chunk = chunk_rank
                matched = document
            if document_rank is not None and best_doc is None:
                best_doc = document_rank
            if best_chunk is not None and best_doc is not None:
                break
    return {"best_chunk_rank": best_chunk, "best_document_rank": best_doc, "matched_doc": matched}


def serialized_positions(rows: list[dict[str, Any]], targets: set[str]) -> dict[str, Any]:
    for index, row in enumerate(rows, 1):
        metadata = row.get("metadata") or {}
        document = str(metadata.get("doc_id") or row.get("doc_id") or row.get("id", "").split("#", 1)[0])
        if document in targets:
            return {"rank": index, "matched_doc": document, "chunk_id": row.get("id") or row.get("chunk_id")}
    return {"rank": None, "matched_doc": None, "chunk_id": None}


def main() -> None:
    benchmark = {
        row["instance_id"]: row
        for row in json.loads((ROOT / "predictive_clinical_benchmark/benchmark_multinode.json").read_text(encoding="utf-8"))
    }
    tools = RetrievalTools(ROOT / "indexes_step3_v2", embed_model_path=ROOT / "models/bge-large-zh-v1.5")
    cases = []
    try:
        for case_id, oracle in ORACLES.items():
            session_path = TRACE_ROOT / case_id / "session" / f"current10_20260729__{case_id}.json"
            session = json.loads(session_path.read_text(encoding="utf-8"))
            round_one = next(
                (row for row in session["round_memories"] if row.get("tool_trace")),
                None,
            )
            if round_one is None:
                raise RuntimeError(f"No executed retrieval round in {case_id}")
            trace = round_one["tool_trace"][0]
            actions = trace["arguments"]["actions"]
            targets = set(oracle["docs"])
            cutoff = str(benchmark[case_id].get("time_cutoff") or "")[:10]
            target_metadata = tools.structured.get_document_metadata(list(targets))
            eligible_targets = sorted(
                target for target in targets
                if str((target_metadata.get(target) or {}).get("pub_date") or "")[:10]
                and str((target_metadata.get(target) or {}).get("pub_date"))[:10] <= cutoff
            )
            semantic = next((a for a in actions if a["tool"] in {"dense_search", "hybrid_search"}), actions[0])
            lexical = next((a for a in actions if a["tool"] == "bm25_search"), actions[-1])

            dense_hits = tools.dense_search(
                semantic["query"], top_k=30000,
                spaces=semantic.get("spaces") or ["case_semantic", "event_semantic", "structured_semantic"],
            )
            bm25_hits = tools.bm25_search(lexical["query"], top_k=30000)
            step_hits = tools.step3_then_step2_search(semantic["query"], top_k=120)
            result = trace.get("result") or {}
            reranked = result.get("reranked_candidates") or []
            rerank_ids = [str(row.get("id") or row.get("chunk_id")) for row in reranked if row.get("id") or row.get("chunk_id")]
            replayed_fetch = tools.fetch_evidence(chunk_ids=rerank_ids, limit=min(12, len(rerank_ids)))
            replayed_ids = [str(row.get("chunk_id")) for row in replayed_fetch]

            case = {
                "case_id": case_id,
                **oracle,
                "time_cutoff": cutoff,
                "oracle_publication_dates": {
                    target: (target_metadata.get(target) or {}).get("pub_date")
                    for target in sorted(targets)
                },
                "cutoff_eligible_oracle_docs": eligible_targets,
                "availability_at_cutoff": (
                    oracle["availability"] if eligible_targets else "absent_at_cutoff"
                ),
                "semantic_query": semantic["query"],
                "lexical_query": lexical["query"],
                "dense": ranks(dense_hits, targets),
                "bm25": ranks(bm25_hits, targets),
                "step3_then_step2_top120": ranks(step_hits, targets),
                "post_rerank_top10": serialized_positions(reranked, targets),
                "fetched_after_order_fix": serialized_positions(replayed_fetch, targets),
                "fetch_preserves_rerank_order": replayed_ids == rerank_ids[: len(replayed_ids)],
                "old_fetched_ids": result.get("fetched_evidence_ids") or [],
                "current_fetched_ids": replayed_ids,
            }
            cases.append(case)
            print(case_id, oracle["availability"], "dense", case["dense"], "bm25", case["bm25"], "step", case["step3_then_step2_top120"], "rerank", case["post_rerank_top10"])
    finally:
        tools.close()

    available = [row for row in cases if row["availability"] != "absent"]
    cutoff_available = [row for row in cases if row["cutoff_eligible_oracle_docs"]]
    summary = {
        "case_count": len(cases),
        "corpus_available_or_partial": len(available),
        "corpus_absent": len(cases) - len(available),
        "cutoff_eligible_available": len(cutoff_available),
        "absent_at_cutoff": len(cases) - len(cutoff_available),
        "oracle_doc_dense_top10": sum((row["dense"]["best_document_rank"] or 10**9) <= 10 for row in available),
        "oracle_doc_dense_top50": sum((row["dense"]["best_document_rank"] or 10**9) <= 50 for row in available),
        "oracle_doc_bm25_top10": sum((row["bm25"]["best_document_rank"] or 10**9) <= 10 for row in available),
        "oracle_doc_bm25_top50": sum((row["bm25"]["best_document_rank"] or 10**9) <= 50 for row in available),
        "oracle_doc_step3_top120": sum(row["step3_then_step2_top120"]["best_document_rank"] is not None for row in available),
        "oracle_doc_post_rerank_top10": sum(row["post_rerank_top10"]["rank"] is not None for row in available),
        "fetch_order_preserved_cases": sum(row["fetch_preserves_rerank_order"] for row in cases),
    }
    output = {"metric_definition": "Rank of manually verified best-available corpus document; absent cases are excluded from retrieval recall denominators.", "summary": summary, "cases": cases}
    out = ROOT / "benchmark_results/oracle_doc_rank_10case_20260803.json"
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(out)


if __name__ == "__main__":
    main()
