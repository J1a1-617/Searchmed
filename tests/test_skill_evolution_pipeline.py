from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from skill_evolution.scripts.curate_event_batch import materialize
from skill_evolution.scripts.run_evolution_worker import process


def test_materialized_skill_stays_candidate_until_ab(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text('{"skills": []}')
    synthesis = {"result": {"skills": [{
        "skill_id": "new-open-pattern", "name": "New pattern",
        "problem_pattern": "A newly discovered recurring failure",
        "activation_rules": ["check the binding"],
        "forbidden_behaviors": ["do not guess"],
        "target_call_stages": ["evidence_review"],
        "evidence_from_cases": ["case-1", "case-2"],
        "source_failure_signatures": ["novel_signature"],
        "ab_test_cases": ["case-1"], "ab_success_metric": "restore gold",
        "dedup_key": "new-open-pattern",
    }]}}
    created = materialize(synthesis, registry, tmp_path / "skills", tmp_path / "synthesis.json")
    row = json.loads(registry.read_text())["skills"][0]
    assert created == ["new-open-pattern"]
    assert row["status"] == "candidate"
    assert row["ab_status"] == "pending_source_error_ab"
    assert (tmp_path / "skills/new-open-pattern/SKILL.md").is_file()


def test_worker_consumes_event_once_and_attributes_retrieval_miss(tmp_path: Path) -> None:
    events = tmp_path / "events"
    artifacts = tmp_path / "artifacts"
    events.mkdir()
    artifacts.mkdir()
    db = tmp_path / "index.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE documents (doc_id TEXT PRIMARY KEY, pmid TEXT)")
        conn.execute("CREATE TABLE evidence_chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT)")
        conn.execute("INSERT INTO documents VALUES ('doc-1', 'pmid-1')")
        conn.execute("INSERT INTO evidence_chunks VALUES ('gold-1', 'doc-1')")
    event = {
        "type": "benchmark.case.completed", "event_id": "event-1",
        "artifact_path": str(artifacts),
        "case": {"instance_id": "case-1", "gold_citations": [{"chunk_id": "gold-1"}], "ground_truth": {"overall_benefit": "明显获益"}},
        "prediction": {"overall_benefit": "无明显获益"},
    }
    (events / "event-1.json").write_text(json.dumps(event))
    args = Args()
    args.events = events
    args.state = tmp_path / "state.json"
    args.audits = tmp_path / "audits.json"
    args.index_db = db
    args.curate_every = 99
    args.curator_command = ""
    args.model = "gpt-6"
    args.registry = tmp_path / "registry.json"
    first = process(args)
    second = process(args)
    assert first["new_events"] == 1
    assert second["new_events"] == 0
    assert json.loads(args.audits.read_text())[0]["failure_class"] == "retrieval_miss"


class Args:
    pass
