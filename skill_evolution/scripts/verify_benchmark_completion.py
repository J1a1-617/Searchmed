#!/usr/bin/env python3
"""Require one complete checkpoint for every source instance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--agent-system-id", required=True)
    parser.add_argument("--start-batch", type=int, default=5)
    parser.add_argument("--end-batch", type=int, default=11)
    args = parser.parse_args()
    expected = set()
    for number in range(args.start_batch, args.end_batch + 1):
        path = args.data_root / f"batch_{number:03d}.json"
        for row in json.loads(path.read_text(encoding="utf-8")):
            expected.add(str(row["instance_id"]))
    completed = set()
    for path in args.checkpoint_root.rglob("case_*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("status") == "success" and row.get("agent_system_id") == args.agent_system_id:
            completed.add(str(row.get("instance_id")))
    missing = sorted(expected - completed)
    print(json.dumps({"expected": len(expected), "complete": len(expected & completed), "missing": missing}, ensure_ascii=False))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
