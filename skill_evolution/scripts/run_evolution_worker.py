#!/usr/bin/env python3
"""Consume durable benchmark-completion events and launch Skill curation.

Designed for a supervisor/systemd job.  ``--once`` makes it cron-friendly and
easy to test. Events are append-only JSON files; processed IDs live in a
durable state file, so restarts do not repeat GPT calls.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from skill_evolution.evidence_audit import audit_cases


POSITIVE_BENEFIT_LABELS = {"明显获益", "有限获益或稳定"}


def binary_outcome_error(truth: str, prediction: str) -> bool:
    """Use the benchmark's declared benefit/non-benefit decision boundary."""
    if not truth or not prediction:
        return False
    return (truth in POSITIVE_BENEFIT_LABELS) != (prediction in POSITIVE_BENEFIT_LABELS)


def load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def process(args: argparse.Namespace) -> dict[str, Any]:
    state = load(args.state, {"processed_event_ids": [], "pending_eligible": []})
    processed = set(str(x) for x in state.get("processed_event_ids") or [])
    events = []
    for path in sorted(args.events.glob("*.json")):
        event = load(path, {})
        event_id = str(event.get("event_id") or path.stem)
        if event_id not in processed and event.get("type") == "benchmark.case.completed":
            events.append((event_id, event))
    new_audits = []
    for event_id, event in events:
        case = event.get("case") or {}
        truth = str((case.get("ground_truth") or {}).get("overall_benefit") or "")
        prediction = str((event.get("prediction") or {}).get("overall_benefit") or "")
        outcome_error = binary_outcome_error(truth, prediction)
        if outcome_error:
            audit = audit_cases([case], {str(case.get("instance_id") or case.get("question_id")): Path(str(event.get("artifact_path") or ""))}, args.index_db)[0]
        else:
            audit = {
                "instance_id": case.get("instance_id") or case.get("question_id"),
                "gold_status": "not_audited_correct_outcome",
                "failure_class": "none",
                "skill_learning_eligible": False,
                "retrieval_skill_learning_eligible": False,
                "reason": "Correct binary outcome; expensive gold reverse-search is reserved for source errors.",
                "artifact_path": str(event.get("artifact_path") or ""),
            }
        audit["outcome_error"] = outcome_error
        audit["general_skill_learning_eligible"] = outcome_error
        # A wrong prediction may still teach a reasoning/format Skill when gold
        # evidence is absent. It may not teach a retrieval Skill without the
        # separate retrieval eligibility flag.
        audit["skill_learning_eligible"] = bool(audit.get("retrieval_skill_learning_eligible") or outcome_error)
        if outcome_error and audit.get("gold_status") != "explicit":
            audit["attribution_constraint"] = (
                "await_codex_gold_candidate_verification; retrieval attribution prohibited until a candidate is accepted"
                if audit.get("gold_status") == "text_candidates_pending_codex"
                else "general_reasoning_or_structure_only; retrieval attribution prohibited"
            )
        audit["event_id"] = event_id
        new_audits.append(audit)
        processed.add(event_id)
    audit_rows = load(args.audits, [])
    audit_rows.extend(new_audits)
    atomic_write(args.audits, audit_rows)
    pending = list(dict.fromkeys([*(state.get("pending_eligible") or []), *[
        row["event_id"] for row in new_audits if row.get("skill_learning_eligible")
    ]]))
    launched = False
    command_result = None
    if len(pending) >= args.curate_every:
        env = os.environ.copy()
        env["SKILL_CURATOR_MODEL"] = args.model
        command = args.curator_command or [
            sys.executable, "skill_evolution/scripts/curate_event_batch.py",
            "--events", str(args.events), "--audits", str(args.audits),
            "--registry", str(args.registry), "--model", args.model,
        ]
        if isinstance(command, str):
            import shlex
            command = shlex.split(command)
        for event_id in pending:
            command.extend(["--pending-event-id", event_id])
        completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
        command_result = {"returncode": completed.returncode, "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-2000:]}
        launched = True
        if completed.returncode == 0:
            pending = []
    state = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "processed_event_ids": sorted(processed),
        "pending_eligible": pending,
        "last_failure_classes": dict(Counter(row.get("failure_class") for row in new_audits)),
        "last_curator_launch": command_result,
    }
    atomic_write(args.state, state)
    return {"new_events": len(events), "new_audits": len(new_audits), "pending_eligible": len(pending), "curator_launched": launched}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, default=Path("skill_evolution/events"))
    ap.add_argument("--state", type=Path, default=Path("skill_evolution/registry/worker_state.json"))
    ap.add_argument("--audits", type=Path, default=Path("skill_evolution/registry/evidence_availability_audits.json"))
    ap.add_argument("--index-db", type=Path, default=Path("indexes/structured.db"))
    ap.add_argument("--model", default=os.environ.get("SKILL_CURATOR_MODEL") or "gpt-6-astra")
    ap.add_argument("--curate-every", type=int, default=2)
    ap.add_argument("--curator-command", default="")
    ap.add_argument("--registry", type=Path, default=Path("skill_evolution/registry/registry.json"))
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    args.events.mkdir(parents=True, exist_ok=True)
    while True:
        print(json.dumps(process(args), ensure_ascii=False), flush=True)
        if args.once:
            return 0
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
