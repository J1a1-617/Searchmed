#!/usr/bin/env python3
"""Inventory binary benchmark errors and create provenance-safe Skill A/B jobs."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


POSITIVE_BENEFIT = {"明显获益", "有限获益或稳定"}


def _load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _data_rows(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        values = _load(path, [])
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, dict) and value.get("instance_id"):
                rows[str(value["instance_id"])] = value
    return rows


def _successful_predictions(result_paths: Iterable[Path], checkpoint_roots: Iterable[Path]) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for path in result_paths:
        payload = _load(path, {})
        for row in payload.get("per_instance") or []:
            if isinstance(row, dict) and row.get("instance_id") and isinstance(row.get("parsed_output"), dict):
                predictions[str(row["instance_id"])] = {
                    "prediction": row["parsed_output"],
                    "source": str(path),
                    "completion": "complete_llm",
                }
    for root in checkpoint_roots:
        for path in sorted(root.rglob("case_*.json")):
            payload = _load(path, {})
            result = payload.get("result") or {}
            prediction = result.get("parsed_output") if isinstance(result, dict) else None
            if payload.get("status") != "success" or not isinstance(prediction, dict):
                continue
            predictions[str(payload.get("instance_id"))] = {
                "prediction": prediction,
                "source": str(path),
                "completion": "complete_llm",
            }
    return predictions


def _binary(label: str) -> str:
    return "positive" if str(label) in POSITIVE_BENEFIT else "negative"


def build_state(
    *,
    cases: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    registry: dict[str, Any],
    artifact_by_case: dict[str, str] | None = None,
) -> dict[str, Any]:
    skills = [row for row in registry.get("skills") or [] if isinstance(row, dict)]
    wrong_cases = []
    for instance_id, record in sorted(predictions.items()):
        case = cases.get(instance_id)
        if not case:
            continue
        truth_label = str((case.get("ground_truth") or {}).get("overall_benefit") or "")
        prediction_label = str((record.get("prediction") or {}).get("overall_benefit") or "")
        if _binary(truth_label) == _binary(prediction_label):
            continue
        source_skills = [
            str(skill.get("skill_id"))
            for skill in skills
            if instance_id in {str(value) for value in skill.get("source_cases") or []}
        ]
        validated_skills = [
            str(skill.get("skill_id"))
            for skill in skills
            if instance_id in {str(value) for value in skill.get("ab_validated_source_cases") or []}
        ]
        wrong_cases.append({
            "instance_id": instance_id,
            "ground_truth": truth_label,
            "prediction": prediction_label,
            "ground_truth_binary": _binary(truth_label),
            "prediction_binary": _binary(prediction_label),
            "result_source": record.get("source"),
            "artifact_path": (artifact_by_case or {}).get(instance_id),
            "completion": record.get("completion"),
            "source_skills": source_skills,
            "validated_skills": validated_skills,
            "skill_state": "validated" if validated_skills else ("candidate" if source_skills else "unassigned"),
        })

    wrong_ids = {row["instance_id"] for row in wrong_cases}
    jobs = []
    for skill in skills:
        skill_id = str(skill.get("skill_id") or "")
        source_errors = [str(value) for value in skill.get("source_cases") or [] if str(value) in wrong_ids]
        if not skill_id or not source_errors or str(skill.get("status") or "").startswith("merged"):
            continue
        jobs.append({
            "skill_id": skill_id,
            "status": "pending_source_error_ab",
            "source_error_cases": source_errors,
            "required_call_stages": [str(value) for value in skill.get("call_stages") or []],
            "arm_a": {"dynamic_skills_enabled": False},
            "arm_b": {"dynamic_skills_enabled": True, "skill_allowlist": [skill_id]},
            "activation_gate": {
                "baseline_must_reproduce_error": True,
                "intervention_must_correct_error": True,
                "must_be_retrieved": True,
                "must_be_selected": True,
                "must_be_loaded": True,
                "must_run_at_declared_stage": True,
                "fallback_allowed": False,
            },
        })

    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metric": "binary_overall_benefit",
        "eligible_completion": "complete_llm_only",
        "completed_predictions": len(predictions),
        "wrong_case_count": len(wrong_cases),
        "wrong_cases": wrong_cases,
        "coverage": {
            "unassigned": sum(row["skill_state"] == "unassigned" for row in wrong_cases),
            "candidate": sum(row["skill_state"] == "candidate" for row in wrong_cases),
            "validated": sum(row["skill_state"] == "validated" for row in wrong_cases),
        },
        "ab_jobs": jobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--result", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint-root", type=Path, action="append", default=[])
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact_by_case: dict[str, str] = {}
    artifact_mtime: dict[str, float] = {}
    for root in args.artifact_root:
        for path in root.rglob("final_prediction.json"):
            payload = _load(path, {})
            if payload.get("run_status") != "complete" or payload.get("final_prediction_source") != "llm":
                continue
            parent = path.parent
            case_dir = parent.parent if parent.name.startswith("attempt_") else parent
            instance_id = case_dir.name
            if path.stat().st_mtime >= artifact_mtime.get(instance_id, 0.0):
                artifact_mtime[instance_id] = path.stat().st_mtime
                artifact_by_case[instance_id] = str(parent)
    state = build_state(
        cases=_data_rows(args.data),
        predictions=_successful_predictions(args.result, args.checkpoint_root),
        registry=_load(args.registry, {"skills": []}),
        artifact_by_case=artifact_by_case,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "completed_predictions": state["completed_predictions"],
        "wrong_case_count": state["wrong_case_count"],
        "coverage": state["coverage"],
        "ab_jobs": len(state["ab_jobs"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
