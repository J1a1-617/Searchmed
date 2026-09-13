#!/usr/bin/env python3
"""Turn audited completion events into persisted GPT-curated candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def safe_id(value: Any) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", str(value).lower()).strip("-")[:80] or "candidate-skill"


def skill_markdown(candidate: dict[str, Any]) -> str:
    stages = ",".join(candidate.get("target_call_stages") or ["generate"])
    uses = "; ".join(candidate.get("activation_rules") or [])
    forbids = "\n".join(f"- {row}" for row in candidate.get("forbidden_behaviors") or [])
    rules = "\n".join(f"{index}. {row}" for index, row in enumerate(candidate.get("activation_rules") or [], 1))
    return (
        "---\n"
        f"name: {safe_id(candidate.get('skill_id'))}\n"
        f"description: {candidate.get('name') or candidate.get('problem_pattern') or ''}\n"
        "metadata:\n"
        f"  call_stage: {stages}\n"
        f"  when_to_use: {uses}\n"
        "  when_not_to_use: 未出现对应结构性失败信号，或证据审计为 unknown/db_missing_gold\n"
        "---\n\n"
        f"# {candidate.get('name') or candidate.get('skill_id')}\n\n"
        f"问题模式：{candidate.get('problem_pattern') or ''}\n\n"
        "## 执行规则\n\n" + (rules or "1. 保持原有行为。") + "\n\n"
        "## 禁止行为\n\n" + (forbids or "- 不得越过证据边界。") + "\n"
    )


def persist_gold_resolutions(audits_path: Path, resolutions: list[dict[str, Any]]) -> None:
    """Merge Codex semantic decisions into the durable per-case audits."""
    rows = load(audits_path, [])
    by_id = {str(row.get("instance_id")): row for row in resolutions if isinstance(row, dict)}
    for row in rows:
        resolution = by_id.get(str(row.get("instance_id")))
        if not resolution:
            continue
        row["codex_gold_resolution"] = resolution
        decision = resolution.get("decision")
        loss = resolution.get("first_loss_stage")
        if decision == "db_gold_found":
            row["gold_status"] = "text_resolved_by_codex"
            row["failure_class"] = f"{loss}_drop" if loss not in {None, "none", "unknown"} else "gold_reached_generation"
            row["retrieval_skill_learning_eligible"] = loss in {
                "candidate_retrieval", "rerank", "fetch", "evidence_review",
                "answer_memory", "answer_context", "generate",
            }
        elif decision == "db_missing_gold":
            row["gold_status"] = "text_resolved_by_codex"
            row["failure_class"] = "db_missing_gold"
            row["retrieval_skill_learning_eligible"] = False
        elif decision == "unknown":
            row["failure_class"] = "unknown"
            row["retrieval_skill_learning_eligible"] = False
    atomic_write(audits_path, rows)


def materialize(synthesis: dict[str, Any], registry_path: Path, skills_root: Path, artifact: Path) -> list[str]:
    registry = load(registry_path, {"skills": [], "cohorts": [], "eval_runs": []})
    created = []
    for raw in ((synthesis.get("result") or {}).get("skills") or []):
        if not isinstance(raw, dict):
            continue
        skill_id = safe_id(raw.get("skill_id"))
        canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
        existing = next((row for row in registry.get("skills") or [] if row.get("skill_id") == skill_id), None)
        record = {
            "skill_id": skill_id,
            "title": raw.get("name"),
            "problem_pattern": raw.get("problem_pattern"),
            "rules": raw.get("activation_rules") or [],
            "forbidden_behaviors": raw.get("forbidden_behaviors") or [],
            "call_stages": raw.get("target_call_stages") or [],
            "source_cases": raw.get("evidence_from_cases") or [],
            "source_failure_signatures": raw.get("source_failure_signatures") or [],
            "ab_test_cases": raw.get("ab_test_cases") or [],
            "ab_success_metric": raw.get("ab_success_metric"),
            "dedup_key": raw.get("dedup_key"),
            "skill_hash": digest,
            "status": "candidate",
            "ab_status": "pending_source_error_ab",
            "synthesis_artifact": str(artifact),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if existing is None:
            registry.setdefault("skills", []).append(record)
            created.append(skill_id)
        elif existing.get("skill_hash") != digest:
            existing.update(record)
            existing["status"] = "candidate_update_pending_ab"
            created.append(skill_id)
        skill_path = skills_root / skill_id / "SKILL.md"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(skill_markdown(raw), encoding="utf-8")
    registry["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write(registry_path, registry)
    return created


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--audits", type=Path, required=True)
    ap.add_argument("--pending-event-id", action="append", default=[])
    ap.add_argument("--registry", type=Path, default=Path("skill_evolution/registry/registry.json"))
    ap.add_argument("--skills-root", type=Path, default=Path("skill_evolution/skills"))
    ap.add_argument("--output-root", type=Path, default=Path("skill_evolution/registry/auto_curations"))
    ap.add_argument("--model", default=os.environ.get("SKILL_CURATOR_MODEL") or "gpt-6-astra")
    ap.add_argument("--curator-backend", choices=["codex", "llm"], default="codex")
    args = ap.parse_args()
    wanted = set(args.pending_event_id)
    events = [load(path, {}) for path in sorted(args.events.glob("*.json"))]
    events = [row for row in events if str(row.get("event_id")) in wanted]
    audits = [row for row in load(args.audits, []) if str(row.get("event_id")) in wanted]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.output_root / f"curation_{stamp}"
    cohort = {
        "cohort_id": f"auto_{stamp}",
        "source_batches": sorted({str(row.get("run_id")) for row in events}),
        "source_instance_ids": [str((row.get("case") or {}).get("instance_id") or (row.get("case") or {}).get("question_id")) for row in events],
        "source_error_instance_ids": [str(row.get("instance_id")) for row in audits if row.get("skill_learning_eligible")],
    }
    audit_summary = {"case_count": len(audits), "cases": audits, "open_failure_classes": sorted({str(row.get("failure_class")) for row in audits})}
    failure_inventory = {"wrong_cases": [{
        "instance_id": str((event.get("case") or {}).get("instance_id") or (event.get("case") or {}).get("question_id")),
        "ground_truth": (event.get("case") or {}).get("ground_truth"),
        "prediction": event.get("prediction"),
        "artifact_path": event.get("artifact_path"),
        "evidence_failure_class": next((row.get("failure_class") for row in audits if row.get("event_id") == event.get("event_id")), "unknown"),
    } for event in events]}
    atomic_write(root / "cohort.json", cohort)
    atomic_write(root / "audit.json", audit_summary)
    atomic_write(root / "failure_inventory.json", failure_inventory)
    output = root / "synthesis.json"
    if args.curator_backend == "codex":
        raw_output = root / "codex_candidates.json"
        command = [sys.executable, "skill_evolution/scripts/run_codex_curator.py",
                   "--cohort", str(root / "cohort.json"), "--audit", str(root / "audit.json"),
                   "--failure-inventory", str(root / "failure_inventory.json"),
                   "--evidence-audits", str(args.audits), "--model", args.model,
                   "--schema", str(root / "candidate_schema.json"), "--trace", str(root / "codex_trace.jsonl"),
                   "--output", str(raw_output)]
    else:
        raw_output = output
        command = [sys.executable, "skill_evolution/scripts/synthesize_skills.py", "--cohort", str(root / "cohort.json"), "--audit", str(root / "audit.json"), "--failure-inventory", str(root / "failure_inventory.json"), "--evidence-audits", str(args.audits), "--model", args.model, "--output", str(output)]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        return completed.returncode
    if args.curator_backend == "codex":
        synthesis = {"status": "generated", "curator_backend": "codex", "curator_model": args.model,
                     "result": load(raw_output, {})}
        atomic_write(output, synthesis)
    else:
        synthesis = load(output, {})
    persist_gold_resolutions(args.audits, (synthesis.get("result") or {}).get("gold_evidence_resolutions") or [])
    created = materialize(synthesis, args.registry, args.skills_root, output)
    atomic_write(root / "ab_queue.json", {"status": "pending", "skills": created, "source_cases": cohort["source_error_instance_ids"]})
    print(json.dumps({"status": "curated", "model": args.model, "created_or_updated": created, "root": str(root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
