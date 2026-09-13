#!/usr/bin/env python3
"""Create a compact, evidence-backed audit of SearchAgent benchmark trajectories."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _session_id(path: Path, payload: dict[str, Any]) -> str:
    raw = str(payload.get("session_id") or path.stem)
    return raw.removeprefix("smoke_")


def _checkpoint_rows(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("instance_id"):
                rows[str(row["instance_id"])] = row
    return rows


def _all_evidence(session: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for memory in session.get("round_memories") or session.get("step_memories") or []:
        if isinstance(memory, dict):
            rows.extend(row for row in memory.get("accepted_evidence") or [] if isinstance(row, dict))
    return rows


def audit_session(path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    session = _read_json(path)
    evidence = _all_evidence(session)
    roles = Counter(str(row.get("evidence_role") or "unknown") for row in evidence)
    different_analogs = [
        str(row.get("chunk_id") or "")
        for row in evidence
        if row.get("evidence_role") == "analog_support"
        and row.get("target_entity_match") == "different"
    ]
    low_analog = [
        str(row.get("chunk_id") or "")
        for row in evidence
        if row.get("evidence_role") == "analog_support"
        and float(row.get("llm_relevance_score") or 0.0) < 0.35
    ]
    claims = [row for row in ((session.get("answer_memory") or {}).get("claims") or []) if isinstance(row, dict)]
    merged_claims = [
        str(row.get("claim_id") or "")
        for row in claims
        if len(row.get("evidence_scopes") or []) > 1
    ]
    context = session.get("answer_context_summary") or {}
    final_answer = session.get("final_answer")
    if isinstance(final_answer, str):
        try:
            final_answer = json.loads(final_answer)
        except json.JSONDecodeError:
            pass
    llm_summary = ((checkpoint.get("telemetry") or {}).get("llm_summary") or {})
    stage_calls = {
        str(name): int((stats or {}).get("calls") or 0)
        for name, stats in llm_summary.items()
        if isinstance(stats, dict)
    }
    stage_seconds = {
        str(name): round(float((stats or {}).get("duration_seconds") or 0.0), 3)
        for name, stats in llm_summary.items()
        if isinstance(stats, dict)
    }
    return {
        "instance_id": _session_id(path, session),
        "session_path": str(path),
        "run_status": checkpoint.get("run_status"),
        "final_prediction_source": checkpoint.get("final_prediction_source"),
        "failed_stages": checkpoint.get("failed_stages") or [],
        "plan_steps": len((session.get("retrieval_plan") or {}).get("steps") or []),
        "executed_rounds": len(session.get("round_memories") or []),
        "replans": len(session.get("replan_decisions") or []),
        "accepted_evidence": len(evidence),
        "evidence_roles": dict(roles),
        "analog_different_ids": different_analogs,
        "low_relevance_analog_ids": low_analog,
        "claims": len(claims),
        "merged_claim_ids": merged_claims,
        "context_direct_findings": len(context.get("key_findings") or []),
        "context_weak_findings": len(context.get("partial_or_analog_findings") or []),
        "context_evidence_ids": len(context.get("evidence_ids") or []),
        "final_citations": len(final_answer.get("cited_evidence") or []) if isinstance(final_answer, dict) else None,
        "stage_calls": stage_calls,
        "stage_seconds": stage_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoints = _checkpoint_rows(args.checkpoint)
    session_paths: list[Path] = []
    for root in args.sessions:
        session_paths.extend(root.rglob("smoke_case_*.json"))
    audits = []
    for path in sorted(session_paths):
        session = _read_json(path)
        instance_id = _session_id(path, session)
        audits.append(audit_session(path, checkpoints.get(instance_id, {})))

    occurrence: dict[str, list[str]] = defaultdict(list)
    for row in audits:
        if row["analog_different_ids"]:
            occurrence["analog_different_accepted"].append(row["instance_id"])
        if row["low_relevance_analog_ids"]:
            occurrence["low_relevance_analog_accepted"].append(row["instance_id"])
        if row["merged_claim_ids"]:
            occurrence["answer_memory_merged_scopes"].append(row["instance_id"])
        if row["accepted_evidence"] and not row["context_direct_findings"]:
            occurrence["accepted_but_no_direct_context"].append(row["instance_id"])
        if row["context_evidence_ids"] and row["final_citations"] == 0:
            occurrence["context_evidence_not_cited"].append(row["instance_id"])

    result = {
        "case_count": len(audits),
        "cases": audits,
        "cross_case_occurrences": dict(occurrence),
        "skill_candidates": {
            name: ids for name, ids in occurrence.items() if len(set(ids)) >= 2
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "case_count": len(audits), "skill_candidates": result["skill_candidates"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
