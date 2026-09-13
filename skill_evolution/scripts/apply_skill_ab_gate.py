#!/usr/bin/env python3
"""Apply a source-error A/B artifact to the Skill registry activation gate."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def apply_gate(registry: dict[str, Any], summary: dict[str, Any], artifact: str) -> dict[str, Any]:
    gate = summary.get("source_error_gate") or {}
    skill_id = str(gate.get("skill_id") or "")
    skill = next((row for row in registry.get("skills") or [] if row.get("skill_id") == skill_id), None)
    if not skill:
        raise ValueError(f"skill not found in registry: {skill_id}")
    pairs = [row for row in gate.get("pairs") or [] if isinstance(row, dict)]
    eligible = [row for row in pairs if row.get("baseline_wrong_reproduced")]
    passed = bool(eligible) and gate.get("status") == "pass" and all(row.get("pass") for row in eligible)
    efficacy_passed = bool(eligible) and (gate.get("efficacy_gate") or {}).get("status") == "pass"
    callability_passed = bool(eligible) and (gate.get("callability_gate") or {}).get("status") == "pass"
    # Backward compatibility for older artifacts that only recorded pair.pass.
    if "efficacy_gate" not in gate:
        efficacy_passed = bool(eligible) and all(row.get("intervention_correct") for row in eligible)
    if "callability_gate" not in gate:
        callability_passed = bool(eligible) and all(row.get("mounted_at_required_stage") for row in eligible)
    passed = passed and efficacy_passed and callability_passed
    skill["ab_status"] = "source_error_ab_pass" if passed else "source_error_ab_failed"
    skill["ab_result"] = artifact
    skill["efficacy_gate"] = "pass" if efficacy_passed else "fail"
    skill["callability_gate"] = "pass" if callability_passed else "fail"
    skill["ab_validated_source_cases"] = [row["instance_id"] for row in eligible if row.get("pass")]
    if passed:
        skill["status"] = "active"
    elif skill.get("status") == "active":
        skill["status"] = "candidate_update_pending_ab"
    else:
        skill["status"] = "candidate"
    registry["updated_at"] = datetime.now(timezone.utc).isoformat()
    return {"skill_id": skill_id, "passed": passed, "eligible_source_cases": len(eligible)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = apply_gate(registry, summary, str(args.summary))
    args.registry.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
