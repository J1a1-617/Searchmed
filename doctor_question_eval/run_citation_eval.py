from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from searchagent_retrieval.router import RetrievalRouter
from searchagent_retrieval.tools import RetrievalTools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run citation-accuracy evaluation on doctor-question set."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("doctor_question_eval/doctor_questions_citation_eval.json"),
        help="Path to doctor-question citation eval set.",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=Path("indexes"),
        help="Index root path used by retrieval stack.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k setting passed into router.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("doctor_question_eval/citation_eval_results.json"),
        help="Output JSON file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-question logs.",
    )
    return parser.parse_args()


def _safe_set(items: Sequence[Optional[str]]) -> Set[str]:
    return {item for item in items if isinstance(item, str) and item}


def _anchor_integrity_ratio(evidence_rows: List[Dict[str, Any]]) -> float:
    if not evidence_rows:
        return 0.0
    valid = 0
    for row in evidence_rows:
        citation = row.get("citation_json")
        has_anchor = bool(row.get("chunk_id")) and bool(row.get("doc_id"))
        has_path = isinstance(citation, dict) and bool(citation.get("source_file")) and bool(citation.get("field_path"))
        if has_anchor and has_path:
            valid += 1
    return valid / len(evidence_rows)


def _first_match_rank(ranked_ids: List[str], gold_ids: Set[str]) -> Optional[int]:
    for idx, cid in enumerate(ranked_ids, start=1):
        if cid in gold_ids:
            return idx
    return None


def _resolve_pmids_from_doc_ids(router: RetrievalRouter, doc_ids: Set[str]) -> Set[str]:
    if not doc_ids:
        return set()
    placeholders = ",".join(["?"] * len(doc_ids))
    rows = router.tools.structured.conn.execute(
        f"SELECT doc_id, pmid FROM documents WHERE doc_id IN ({placeholders})",
        tuple(sorted(doc_ids)),
    ).fetchall()
    resolved = set()
    for row in rows:
        pmid = row["pmid"]
        if isinstance(pmid, str) and pmid:
            resolved.add(pmid)
    return resolved


def evaluate_question(
    router: RetrievalRouter,
    item: Dict[str, Any],
    top_k: int,
) -> Dict[str, Any]:
    gold = item.get("gold_citations", [])
    gold_chunk_ids = _safe_set([entry.get("chunk_id") for entry in gold])
    gold_doc_ids = _safe_set([entry.get("doc_id") for entry in gold])
    gold_pmids = _safe_set([entry.get("pmid") for entry in gold])

    result = router.run(query=item["doctor_question"], top_k=top_k)
    evidence_rows = result.get("results", {}).get("fetch_evidence", []) or []

    retrieved_chunk_ids = [str(row.get("chunk_id", "")) for row in evidence_rows if row.get("chunk_id")]
    retrieved_doc_ids = _safe_set([row.get("doc_id") for row in evidence_rows])
    retrieved_pmids = _safe_set([row.get("pmid") for row in evidence_rows])
    if not retrieved_pmids and retrieved_doc_ids:
        retrieved_pmids = _resolve_pmids_from_doc_ids(router=router, doc_ids=retrieved_doc_ids)

    chunk_hits = gold_chunk_ids.intersection(set(retrieved_chunk_ids))
    doc_hits = gold_doc_ids.intersection(retrieved_doc_ids)
    pmid_hits = gold_pmids.intersection(retrieved_pmids)
    first_rank = _first_match_rank(retrieved_chunk_ids, gold_chunk_ids)

    chunk_recall = len(chunk_hits) / len(gold_chunk_ids) if gold_chunk_ids else 0.0
    doc_recall = len(doc_hits) / len(gold_doc_ids) if gold_doc_ids else 0.0
    pmid_recall = len(pmid_hits) / len(gold_pmids) if gold_pmids else 0.0
    precision_at_fetch = len(chunk_hits) / len(retrieved_chunk_ids) if retrieved_chunk_ids else 0.0
    anchor_integrity = _anchor_integrity_ratio(evidence_rows)

    return {
        "question_id": item.get("question_id"),
        "capability": item.get("capability"),
        "doctor_question": item.get("doctor_question"),
        "query_type": result.get("query_type"),
        "selected_tools": result.get("selected_tools", []),
        "gold_chunk_ids": sorted(gold_chunk_ids),
        "gold_doc_ids": sorted(gold_doc_ids),
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_doc_ids": sorted(retrieved_doc_ids),
        "chunk_hit": bool(chunk_hits),
        "doc_hit": bool(doc_hits),
        "pmid_hit": bool(pmid_hits),
        "chunk_recall": round(chunk_recall, 4),
        "doc_recall": round(doc_recall, 4),
        "pmid_recall": round(pmid_recall, 4),
        "precision_at_fetch": round(precision_at_fetch, 4),
        "first_match_rank": first_rank,
        "anchor_integrity": round(anchor_integrity, 4),
        "matched_chunk_ids": sorted(chunk_hits),
        "matched_doc_ids": sorted(doc_hits),
    }


def summarize_metrics(per_question: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(per_question)
    if n == 0:
        return {
            "n_questions": 0,
            "chunk_hit_rate": 0.0,
            "doc_hit_rate": 0.0,
            "pmid_hit_rate": 0.0,
            "mean_chunk_recall": 0.0,
            "mean_doc_recall": 0.0,
            "mean_pmid_recall": 0.0,
            "mean_precision_at_fetch": 0.0,
            "mrr_chunk": 0.0,
            "mean_anchor_integrity": 0.0,
        }

    chunk_hits = [1.0 if row["chunk_hit"] else 0.0 for row in per_question]
    doc_hits = [1.0 if row["doc_hit"] else 0.0 for row in per_question]
    pmid_hits = [1.0 if row["pmid_hit"] else 0.0 for row in per_question]
    chunk_recalls = [row["chunk_recall"] for row in per_question]
    doc_recalls = [row["doc_recall"] for row in per_question]
    pmid_recalls = [row["pmid_recall"] for row in per_question]
    precisions = [row["precision_at_fetch"] for row in per_question]
    anchor_integrities = [row["anchor_integrity"] for row in per_question]
    reciprocal_ranks = []
    for row in per_question:
        rank = row.get("first_match_rank")
        reciprocal_ranks.append(1.0 / rank if isinstance(rank, int) and rank > 0 else 0.0)

    return {
        "n_questions": n,
        "chunk_hit_rate": round(sum(chunk_hits) / n, 4),
        "doc_hit_rate": round(sum(doc_hits) / n, 4),
        "pmid_hit_rate": round(sum(pmid_hits) / n, 4),
        "mean_chunk_recall": round(sum(chunk_recalls) / n, 4),
        "mean_doc_recall": round(sum(doc_recalls) / n, 4),
        "mean_pmid_recall": round(sum(pmid_recalls) / n, 4),
        "mean_precision_at_fetch": round(sum(precisions) / n, 4),
        "mrr_chunk": round(sum(reciprocal_ranks) / n, 4),
        "mean_anchor_integrity": round(sum(anchor_integrities) / n, 4),
    }


def summarize_by_capability(per_question: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in per_question:
        grouped.setdefault(row.get("capability", "unknown"), []).append(row)
    return {cap: summarize_metrics(rows) for cap, rows in grouped.items()}


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    with args.dataset.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("Dataset top-level must be a JSON array.")

    tools = RetrievalTools(index_root=args.index_root)
    try:
        router = RetrievalRouter(retrieval_tools=tools)
        per_question: List[Dict[str, Any]] = []
        for idx, item in enumerate(dataset, start=1):
            row = evaluate_question(router=router, item=item, top_k=args.top_k)
            per_question.append(row)
            if not args.quiet:
                print(
                    f"[{idx:02d}/{len(dataset)}] {row['question_id']} "
                    f"chunk_hit={row['chunk_hit']} "
                    f"doc_hit={row['doc_hit']} "
                    f"rank={row['first_match_rank']}"
                )
    finally:
        tools.close()

    summary = summarize_metrics(per_question)
    by_capability = summarize_by_capability(per_question)

    payload = {
        "config": {
            "dataset": str(args.dataset),
            "index_root": str(args.index_root),
            "top_k": args.top_k,
            "n_questions": len(dataset),
        },
        "summary": summary,
        "by_capability": by_capability,
        "per_question": per_question,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n=== Citation Eval Summary ===")
    for key in (
        "n_questions",
        "chunk_hit_rate",
        "doc_hit_rate",
        "pmid_hit_rate",
        "mean_chunk_recall",
        "mean_precision_at_fetch",
        "mrr_chunk",
        "mean_anchor_integrity",
    ):
        print(f"{key}: {summary[key]}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
