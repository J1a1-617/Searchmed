"""Deterministic gold-evidence availability and stage-loss auditing.

The curator must never infer database coverage from a failed retrieval.  This
module resolves explicit gold anchors against the index and follows them
through persisted artifacts.  Cases without explicit anchors remain
``unknown`` rather than being mislabeled as database misses.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

STAGE_FILES = (
    ("retrieval", "retrieval_results.json"),
    ("evidence_review", "evidence_review.json"),
    ("answer_context", "answer_context.json"),
    ("generate", "final_prediction.json"),
)

RETRIEVAL_SUBSTAGES = {
    "candidate_retrieval": {"dense_search", "hybrid_search", "structured_search", "sparse_search", "seed_candidates"},
    "rerank": {"rerank_candidates", "reranked_hits", "local_rerank", "llm_rerank"},
    "fetch": {"fetch_evidence", "fetched_evidence"},
}


def _strings_for_keys(value: Any, keys: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in keys:
                if isinstance(item, (str, int)):
                    found.add(str(item))
                elif isinstance(item, list):
                    found.update(str(row) for row in item if isinstance(row, (str, int)))
            found.update(_strings_for_keys(item, keys))
    elif isinstance(value, list):
        for item in value:
            found.update(_strings_for_keys(item, keys))
    return {item for item in found if item}


def gold_anchors(case: dict[str, Any]) -> dict[str, list[str]]:
    """Read only explicit gold anchors; outcome labels are not evidence gold."""
    rows: list[Any] = []
    for key in ("gold_citations", "gold_evidence", "gold_evidence_groups"):
        value = case.get(key)
        if isinstance(value, list):
            rows.extend(value)
    ground_truth = case.get("ground_truth") or {}
    if isinstance(ground_truth, dict):
        for key in ("gold_citations", "gold_evidence", "gold_evidence_groups"):
            value = ground_truth.get(key)
            if isinstance(value, list):
                rows.extend(value)
    packed = {"chunk_ids": set(), "doc_ids": set(), "pmids": set(), "group_ids": set()}
    for row in rows:
        if isinstance(row, str):
            packed["chunk_ids"].add(row)
            continue
        if not isinstance(row, dict):
            continue
        for source, target in (
            ("chunk_id", "chunk_ids"), ("doc_id", "doc_ids"),
            ("pmid", "pmids"), ("group_id", "group_ids"),
        ):
            value = row.get(source)
            if value:
                packed[target].add(str(value))
        for member in row.get("members") or row.get("citations") or []:
            if isinstance(member, dict):
                if member.get("chunk_id"):
                    packed["chunk_ids"].add(str(member["chunk_id"]))
                if member.get("doc_id"):
                    packed["doc_ids"].add(str(member["doc_id"]))
                if member.get("pmid"):
                    packed["pmids"].add(str(member["pmid"]))
    return {key: sorted(values) for key, values in packed.items()}


def gold_evidence_texts(case: dict[str, Any]) -> list[str]:
    """Return textual evidence descriptions, never the outcome label itself."""
    ground_truth = case.get("ground_truth") or {}
    if not isinstance(ground_truth, dict):
        return []
    rows: list[str] = []
    for key in ("key_evidence_items", "evidence_items", "gold_evidence_texts"):
        value = ground_truth.get(key)
        if isinstance(value, list):
            rows.extend(str(item).strip() for item in value if str(item).strip())
    reason = str(ground_truth.get("benefit_rebalance_reason") or "").strip()
    if reason:
        rows.append(reason)
    return list(dict.fromkeys(rows))


def _hit_record(hit: Any, channel: str, query: str) -> dict[str, Any]:
    metadata = dict(hit.metadata or {})
    return {
        "channel": channel,
        "query": query,
        "id": str(hit.id),
        "chunk_id": str(metadata.get("chunk_id") or hit.id),
        "doc_id": str(metadata.get("doc_id") or str(hit.id).split("#", 1)[0]),
        "pmid": str(metadata.get("pmid") or ""),
        "score": float(hit.score),
        "text": str(hit.text or ""),
        "metadata": metadata,
    }


def _semantic_overlap(query: str, text: str) -> dict[str, Any]:
    """A conservative pre-filter; Codex remains the final semantic verifier."""
    # Keep this module importable by the lightweight worker environment. The
    # full retrieval stack (numpy/torch) is imported only when reverse search
    # actually runs.
    q_terms = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9+.-]{1,}|[\u4e00-\u9fff]{2,8}", query))
    t_lower = text.lower()
    matched = sorted(term for term in q_terms if term.lower() in t_lower)
    lexical = len(matched) / max(1, len(q_terms))
    # Dates and response/progression statements are especially discriminative
    # for the benchmark's longitudinal gold evidence.
    q_dates = set(re.findall(r"20\d{2}[-年/.]\d{1,2}(?:[-月/.]\d{1,2})?", query))
    date_hit = any(date.replace("年", "-").replace("月", "-").rstrip("日") in text.replace("年", "-").replace("月", "-").replace("日", "") for date in q_dates)
    outcome_terms = set(re.findall(r"进展|增多|增大|新发|缓解|缩小|稳定|复发|progress(?:ion|ed)?|response|\bPD\b|\bPR\b|\bSD\b", query, re.I))
    outcome_hit = any(term.lower() in t_lower for term in outcome_terms)
    return {
        "lexical_overlap": round(lexical, 4),
        "matched_terms": matched[:30],
        "date_hit": date_hit,
        "outcome_hit": outcome_hit,
        "prefilter_pass": lexical >= 0.35 or (date_hit and outcome_hit),
    }


def _reverse_search_gold(case: dict[str, Any], index_root: Path, top_k: int = 30) -> dict[str, Any]:
    texts = gold_evidence_texts(case)
    result: dict[str, Any] = {
        "queries": texts,
        "channels": {},
        "semantic_verification": "required_by_codex",
        "prefiltered_candidates": [],
    }
    if not texts or not (index_root / "structured.db").is_file():
        result["status"] = "unavailable" if texts else "no_gold_text"
        return result
    try:
        from searchagent_retrieval.tools import RetrievalTools
        tools = RetrievalTools(index_root, reranker_backend="none")
    except Exception as exc:
        result["status"] = "retrieval_backend_unavailable"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    records: list[dict[str, Any]] = []
    try:
        for query in texts:
            channel_hits = {
                "dense": tools.dense_search(query, top_k=top_k),
                "bm25": tools.bm25_search(query, top_k=top_k),
                "step3_then_step2": tools.step3_then_step2_search(query, top_k=top_k),
                # Structured rows are included as a separate coverage channel;
                # empty constraints intentionally avoid inventing patient facts.
                "structured": tools.structured_search({}, top_k=top_k),
            }
            for channel, hits in channel_hits.items():
                result["channels"].setdefault(channel, 0)
                result["channels"][channel] += len(hits)
                for hit in hits:
                    row = _hit_record(hit, channel, query)
                    row["semantic_prefilter"] = _semantic_overlap(query, row["text"])
                    records.append(row)
    finally:
        tools.close()
    records.sort(key=lambda row: (bool(row["semantic_prefilter"]["prefilter_pass"]), row["semantic_prefilter"]["lexical_overlap"], row["score"]), reverse=True)
    result["prefiltered_candidates"] = [row for row in records if row["semantic_prefilter"]["prefilter_pass"]][:60]
    result["status"] = "candidates_found" if result["prefiltered_candidates"] else "no_candidate"
    return result


def _db_presence(index_db: Path, anchors: dict[str, list[str]]) -> dict[str, Any]:
    result = {"chunk_ids": [], "doc_ids": [], "pmids": []}
    if not index_db.is_file():
        return {**result, "index_status": "unavailable"}
    with sqlite3.connect(str(index_db)) as conn:
        for key, table, column in (
            ("chunk_ids", "evidence_chunks", "chunk_id"),
            ("doc_ids", "documents", "doc_id"),
            ("pmids", "documents", "pmid"),
        ):
            values = anchors[key]
            if not values:
                continue
            placeholders = ",".join("?" for _ in values)
            query = f"SELECT DISTINCT {column} FROM {table} WHERE {column} IN ({placeholders})"
            result[key] = sorted(str(row[0]) for row in conn.execute(query, values))
    return {**result, "index_status": "checked"}


def _matches(anchors: dict[str, list[str]], payload: Any) -> dict[str, list[str]]:
    observed = {
        "chunk_ids": _strings_for_keys(payload, {"chunk_id", "chunk_ids", "evidence_id", "evidence_ids"}),
        "doc_ids": _strings_for_keys(payload, {"doc_id", "doc_ids"}),
        "pmids": _strings_for_keys(payload, {"pmid", "pmids"}),
    }
    return {key: sorted(set(anchors[key]).intersection(observed[key])) for key in observed}


def _named_payloads(value: Any, names: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in names:
                found.append(item)
            found.extend(_named_payloads(item, names))
    elif isinstance(value, list):
        for item in value:
            found.extend(_named_payloads(item, names))
    return found


def audit_case(case: dict[str, Any], artifact_dir: Path, index_db: Path) -> dict[str, Any]:
    anchors = gold_anchors(case)
    has_gold = any(anchors[key] for key in ("chunk_ids", "doc_ids", "pmids"))
    if not has_gold:
        reverse = _reverse_search_gold(case, index_db.parent)
        reverse_status = reverse.get("status")
        if reverse_status == "candidates_found":
            gold_status, failure = "text_candidates_pending_codex", "pending_semantic_verification"
        elif reverse_status == "no_candidate":
            gold_status, failure = "text_search_no_candidate", "db_missing_gold_text_candidate"
        else:
            gold_status, failure = "unknown_no_explicit_gold", "unknown"
        return {
            "instance_id": case.get("instance_id") or case.get("question_id"),
            "gold_status": gold_status,
            "failure_class": failure,
            "skill_learning_eligible": False,
            "retrieval_skill_learning_eligible": False,
            "reason": "Gold text was reverse-searched across all index channels; Codex must verify semantic identity before stage attribution.",
            "anchors": anchors,
            "gold_evidence_texts": gold_evidence_texts(case),
            "gold_reverse_search": reverse,
            "artifact_path": str(artifact_dir),
        }
    presence = _db_presence(index_db, anchors)
    db_hit = any(presence[key] for key in ("chunk_ids", "doc_ids", "pmids"))
    stages: dict[str, Any] = {}
    retrieval_payload: Any = {}
    for stage, filename in STAGE_FILES:
        path = artifact_dir / filename
        payload = {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        matches = _matches(anchors, payload)
        stages[stage] = {"artifact": str(path), "available": path.is_file(), "gold_matches": matches,
                         "hit": any(matches.values())}
        if stage == "retrieval":
            retrieval_payload = payload
    retrieval_substages: dict[str, Any] = {}
    for substage, names in RETRIEVAL_SUBSTAGES.items():
        payloads = _named_payloads(retrieval_payload, names)
        matches = _matches(anchors, payloads)
        retrieval_substages[substage] = {
            "available": bool(payloads), "gold_matches": matches, "hit": any(matches.values()),
        }
    stages["retrieval_substages"] = retrieval_substages
    if not db_hit:
        failure = "db_missing_gold"
    elif not stages["retrieval"]["hit"]:
        failure = "retrieval_miss"
    elif retrieval_substages["candidate_retrieval"]["available"] and not retrieval_substages["candidate_retrieval"]["hit"]:
        failure = "candidate_retrieval_miss"
    elif retrieval_substages["rerank"]["available"] and retrieval_substages["candidate_retrieval"]["hit"] and not retrieval_substages["rerank"]["hit"]:
        failure = "rerank_drop"
    elif retrieval_substages["fetch"]["available"] and (retrieval_substages["rerank"]["hit"] or retrieval_substages["candidate_retrieval"]["hit"]) and not retrieval_substages["fetch"]["hit"]:
        failure = "fetch_drop"
    elif not stages["evidence_review"]["hit"]:
        failure = "evidence_review_drop"
    elif not stages["answer_context"]["hit"]:
        failure = "answer_context_drop"
    elif not stages["generate"]["hit"]:
        failure = "generation_citation_drop"
    else:
        failure = "gold_reached_generation"
    return {
        "instance_id": case.get("instance_id") or case.get("question_id"),
        "gold_status": "explicit",
        "anchors": anchors,
        "db_presence": presence,
        "stages": stages,
        "failure_class": failure,
        "skill_learning_eligible": failure not in {"db_missing_gold", "gold_reached_generation"},
        "retrieval_skill_learning_eligible": failure not in {"db_missing_gold", "gold_reached_generation"},
        "artifact_path": str(artifact_dir),
    }


def audit_cases(cases: Iterable[dict[str, Any]], artifact_by_id: dict[str, Path], index_db: Path) -> list[dict[str, Any]]:
    return [audit_case(case, artifact_by_id.get(str(case.get("instance_id") or case.get("question_id")), Path("")), index_db)
            for case in cases]
