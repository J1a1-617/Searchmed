from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from skill_evolution.evidence_audit import audit_case


def _index(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE documents (doc_id TEXT PRIMARY KEY, pmid TEXT)")
        conn.execute("CREATE TABLE evidence_chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT)")
        conn.execute("INSERT INTO documents VALUES ('doc-1', 'pmid-1')")
        conn.execute("INSERT INTO evidence_chunks VALUES ('gold-1', 'doc-1')")


def test_no_explicit_gold_is_unknown_not_db_missing(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    _index(db)
    result = audit_case({"instance_id": "case-1", "ground_truth": {"overall_benefit": "明显获益"}}, tmp_path, db)
    assert result["failure_class"] == "unknown"
    assert result["skill_learning_eligible"] is False


def test_gold_is_attributed_to_first_stage_that_drops_it(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    _index(db)
    (tmp_path / "retrieval_results.json").write_text(json.dumps({"chunk_id": "gold-1"}))
    (tmp_path / "evidence_review.json").write_text(json.dumps({"chunk_id": "other"}))
    result = audit_case({"instance_id": "case-1", "gold_citations": [{"chunk_id": "gold-1"}]}, tmp_path, db)
    assert result["failure_class"] == "evidence_review_drop"
    assert result["db_presence"]["chunk_ids"] == ["gold-1"]
    assert result["skill_learning_eligible"] is True


def test_gold_absent_from_db_is_data_coverage_not_skill(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    _index(db)
    result = audit_case({"instance_id": "case-1", "gold_citations": [{"chunk_id": "missing"}]}, tmp_path, db)
    assert result["failure_class"] == "db_missing_gold"
    assert result["skill_learning_eligible"] is False


def test_gold_drop_is_localized_to_rerank(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    _index(db)
    payload = {"results": {"dense_search": [{"chunk_id": "gold-1"}], "rerank_candidates": [{"chunk_id": "other"}]}}
    (tmp_path / "retrieval_results.json").write_text(json.dumps(payload))
    result = audit_case({"instance_id": "case-1", "gold_citations": [{"chunk_id": "gold-1"}]}, tmp_path, db)
    assert result["failure_class"] == "rerank_drop"
    assert result["stages"]["retrieval_substages"]["candidate_retrieval"]["hit"] is True
