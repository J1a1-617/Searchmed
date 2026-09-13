#!/usr/bin/env python3
"""Automate skill accumulation, deduplication, A/B gates, and held-out eval splits.

This controller is deliberately artifact-first: it never activates a candidate
skill unless an A/B artifact marked ``pass`` is present. It can therefore run in
dry-run mode while the API is unavailable and resume without losing provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from skill_evolution.scripts.update_skill_failure_pipeline import (
    _data_rows,
    _successful_predictions,
    build_state as build_failure_state,
)


def load_json(path: Path, default: Any) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value
    except (OSError, json.JSONDecodeError):
        return default


def session_instance(path: Path, value: dict[str, Any]) -> str:
    raw = str(value.get("session_id") or path.stem)
    return raw.removeprefix("smoke_").removesuffix("_recovered")


def batch_rows(manifest: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    rows = []
    for batch in manifest.get("batches") or []:
        path = Path(str(batch.get("path") or ""))
        if not path.is_absolute():
            path = root / path
        ids = [str(x) for x in batch.get("instance_ids") or []]
        rows.append({"batch_index": int(batch.get("batch_index")), "path": path, "instance_ids": ids})
    return rows


def completed_ids(sessions_root: Path) -> set[str]:
    found: set[str] = set()
    for path in sessions_root.rglob("final_prediction.json"):
        value = load_json(path, {})
        if value.get("run_status") != "complete" or value.get("final_prediction_source") != "llm":
            continue
        parent = path.parent
        case_dir = parent.parent if parent.name.startswith("attempt_") else parent
        if case_dir.name:
            found.add(case_dir.name)
    for path in sessions_root.rglob("*.json"):
        value = load_json(path, {})
        if isinstance(value, dict) and value.get("final_answer"):
            found.add(session_instance(path, value))
    return found


def candidate_rules(audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    occurrences: dict[str, set[str]] = defaultdict(set)
    for audit in audits:
        for name, ids in (audit.get("cross_case_occurrences") or {}).items():
            if ids:
                occurrences[name].add(str(audit.get("batch_id") or "unknown"))
    specs = [
        ("clinical-evidence-applicability", "弱类比证据门控", "low_relevance_analog_accepted", [
            "比较治疗方案、人群、终点和时间窗",
            "无直接证据时最多保留两个类比证据",
            "类比证据不能决定目标实体的具体结局",
        ]),
        ("claim-scope-preservation", "Claim 粒度保持", "answer_memory_merged_scopes", [
            "不同药物、人群、终点和时间点不得合并为一个 claim",
            "疗效、CNS、症状、毒性分别保留",
        ]),
        ("evidence-context-citation", "上下文证据绑定", "context_evidence_not_cited", [
            "进入 Generate 的证据必须有稳定 claim/evidence ID",
            "无法绑定的证据从最终上下文移除",
        ]),
        ("temporal-evidence-boundary", "时间切点边界", "temporal_leakage", [
            "文献和病例证据必须不晚于题目 cutoff",
            "晚期结局不能反推早期窗口",
        ]),
    ]
    result = []
    for skill_id, title, trigger, rules in specs:
        if occurrences.get(trigger):
            result.append({"skill_id": skill_id, "title": title, "trigger": trigger, "rules": rules})
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--sessions", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cohort-size", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--ab-dir", type=Path, default=None)
    ap.add_argument("--auto-synthesize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--benchmark-data", type=Path, action="append", default=[])
    ap.add_argument("--benchmark-result", type=Path, action="append", default=[])
    ap.add_argument("--checkpoint-root", type=Path, action="append", default=[])
    args = ap.parse_args()

    manifest = load_json(args.manifest, {})
    batches = batch_rows(manifest, args.root)
    done = completed_ids(args.sessions)
    completed_batches = [b for b in batches if b["instance_ids"] and set(b["instance_ids"]).issubset(done)]
    ordered_completed_ids = [iid for batch in batches for iid in batch["instance_ids"] if iid in done]
    completed_cases = len(ordered_completed_ids)
    cohort_count = completed_cases // max(1, args.cohort_size)
    out_root = args.root / "skill_evolution" / "registry"
    out_root.mkdir(parents=True, exist_ok=True)
    registry_path = out_root / "registry.json"
    registry = load_json(registry_path, {"skills": [], "cohorts": [], "eval_runs": []})
    failure_inventory_path = out_root / "wrong_case_pipeline.json"
    failure_state: dict[str, Any] = {}
    if args.benchmark_data and (args.benchmark_result or args.checkpoint_root):
        failure_state = build_failure_state(
            cases=_data_rows(args.benchmark_data),
            predictions=_successful_predictions(args.benchmark_result, args.checkpoint_root),
            registry=registry,
        )
        failure_inventory_path.write_text(
            json.dumps(failure_state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    audits: list[dict[str, Any]] = []
    audit_root = args.root / "skill_evolution" / "analysis"
    for path in audit_root.rglob("audit*.json"):
        value = load_json(path, {})
        if isinstance(value, dict) and value.get("case_count"):
            # Preserve provenance even for legacy audits that predate an
            # explicit batch_id field (e.g. audit_batch001.json).
            value["batch_id"] = value.get("batch_id") or path.stem.replace("audit_", "")
            audits.append(value)
    candidates = candidate_rules(audits)
    now = datetime.now(timezone.utc).isoformat()
    events = []

    if cohort_count > len(registry.get("cohorts") or []):
        for cohort_no in range(len(registry.get("cohorts") or []) + 1, cohort_count + 1):
            source_ids = ordered_completed_ids[: cohort_no * args.cohort_size]
            source_id_set = set(source_ids)
            selected = [b for b in batches if source_id_set.intersection(b["instance_ids"])]
            source_batches = [int(b["batch_index"]) for b in selected]
            holdout = [iid for b in batches for iid in b["instance_ids"] if iid not in source_id_set]
            source_error_ids = [
                row["instance_id"] for row in failure_state.get("wrong_cases") or []
                if row.get("instance_id") in set(source_ids)
            ]
            cohort = {"cohort_id": f"cohort_{cohort_no:03d}", "created_at": now, "source_batches": source_batches, "source_instance_ids": source_ids, "source_error_instance_ids": source_error_ids, "holdout_instance_ids": holdout[:args.eval_every], "candidate_skills": [c["skill_id"] for c in candidates], "ab_status": "pending"}
            (out_root / "cohorts").mkdir(exist_ok=True)
            (out_root / "cohorts" / f"cohort_{cohort_no:03d}.json").write_text(json.dumps(cohort, ensure_ascii=False, indent=2) + "\n")
            registry.setdefault("cohorts", []).append(cohort)
            events.append({"event": "cohort_created", "cohort_id": cohort["cohort_id"], "source_batches": source_batches, "ab_required_before_activation": True})
            if args.auto_synthesize:
                synthesis_path = out_root / "synthesis" / f"{cohort['cohort_id']}.json"
                audit_paths = sorted(audit_root.rglob("audit*.json"))
                command = [
                    os.environ.get("PYTHON", "python"),
                    str(args.root / "skill_evolution/scripts/synthesize_skills.py"),
                    "--cohort", str(out_root / "cohorts" / f"{cohort['cohort_id']}.json"),
                    "--output", str(synthesis_path),
                ]
                for audit_path in audit_paths:
                    command.extend(["--audit", str(audit_path)])
                command.extend(["--reports", str(audit_root)])
                if failure_inventory_path.is_file():
                    command.extend(["--failure-inventory", str(failure_inventory_path)])
                completed = subprocess.run(command, cwd=args.root, env=os.environ.copy(), check=False, capture_output=True, text=True)
                events.append({"event": "skill_synthesis", "cohort_id": cohort["cohort_id"], "status": "generated" if completed.returncode == 0 else "blocked", "output": str(synthesis_path), "stderr": completed.stderr[-500:]})

    for candidate in candidates:
        canonical = json.dumps({"skill_id": candidate["skill_id"], "rules": candidate["rules"]}, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
        existing = next((x for x in registry.get("skills") or [] if x.get("skill_id") == candidate["skill_id"]), None)
        source_labels = sorted({str(a.get("batch_id")) for a in audits if candidate["trigger"] in (a.get("cross_case_occurrences") or {})})
        if existing is None:
            existing = {**candidate, "skill_hash": digest, "source_batches": source_labels or sorted({int(b["batch_index"]) for b in completed_batches}), "status": "candidate", "ab_status": "pending", "created_at": now}
            registry.setdefault("skills", []).append(existing)
            events.append({"event": "skill_candidate_created", "skill_id": candidate["skill_id"], "ab_required_before_activation": True})
        elif existing.get("skill_hash") != digest:
            existing["status"] = "candidate_update_pending_ab"
            existing["skill_hash"] = digest
            existing["source_batches"] = sorted(set(existing.get("source_batches") or []) | set(source_labels) | {int(b["batch_index"]) for b in completed_batches}, key=str)
            events.append({"event": "skill_update_pending_ab", "skill_id": candidate["skill_id"]})

    if completed_cases >= args.eval_every:
        eval_record = {"eval_id": f"eval_{completed_cases:03d}", "created_at": now, "train_batches": [int(b["batch_index"]) for b in completed_batches], "eval_instance_ids": [iid for b in batches if b not in completed_batches for iid in b["instance_ids"]], "status": "pending", "reason": "100-case holdout evaluation required"}
        if not any(x.get("eval_id") == eval_record["eval_id"] for x in registry.get("eval_runs") or []):
            registry.setdefault("eval_runs", []).append(eval_record)
            events.append({"event": "heldout_eval_required", "eval_id": eval_record["eval_id"]})

    registry["updated_at"] = now
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"completed_cases": completed_cases, "completed_batches": [b["batch_index"] for b in completed_batches], "candidate_skills": [c["skill_id"] for c in candidates], "wrong_case_count": failure_state.get("wrong_case_count"), "wrong_case_coverage": failure_state.get("coverage"), "source_error_ab_jobs": len(failure_state.get("ab_jobs") or []), "events": events, "registry": str(registry_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
