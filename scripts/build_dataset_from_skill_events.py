#!/usr/bin/env python3
"""Reconstruct the exact benchmark cohort recorded by Skill events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = {}
    for path in sorted(args.event_dir.glob("*.json")):
        event = json.loads(path.read_text(encoding="utf-8"))
        case = event.get("case")
        if not isinstance(case, dict) or not case.get("instance_id"):
            continue
        cases[str(case["instance_id"])] = case
    if not cases:
        raise SystemExit(f"no benchmark cases found in {args.event_dir}")

    payload = [cases[instance_id] for instance_id in sorted(cases)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(payload)} unique cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
