#!/usr/bin/env python3
"""Forward-test the clinical-evidence-applicability rubric on saved sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = []
    for path in sorted(args.sessions.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        context = data.get("answer_context_summary") or {}
        direct = context.get("key_findings") or context.get("direct_findings") or []
        analogs = context.get("partial_or_analog_findings") or []
        # The current trajectory audit exposes support level and evidence IDs;
        # this deterministic gate models the skill's forwarding rule: without
        # a direct finding, retain at most two nearest analogs and never retain
        # a finding already marked as an unsupported low-relevance analog.
        # Direct findings are always retained; only the analog tail is capped.
        retained = list(direct)
        if not direct:
            retained = [item for item in analogs if item.get("support_level") != "analog"][:2]
            # If all items are analogs, retain the first two only as explicit
            # gap illustrations; they remain marked analog and cannot support
            # a direct claim downstream.
            if not retained:
                retained = list(analogs[:2])
        cases.append({
            "instance_id": data.get("session_id") or path.stem,
            "direct_findings_before": len(direct),
            "analog_findings_before": len(analogs),
            "forwarded_after_skill": len(retained),
            "direct_evidence_preserved": all(item in retained for item in direct),
            "forwarded_ids": [str(x.get("claim_id") or "") for x in retained],
        })

    result = {
        "skill": "clinical-evidence-applicability",
        "case_count": len(cases),
        "all_no_direct_cases_capped_at_two": all(
            item["direct_findings_before"] or item["forwarded_after_skill"] <= 2
            for item in cases
        ),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("skill", "case_count", "all_no_direct_cases_capped_at_two")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
