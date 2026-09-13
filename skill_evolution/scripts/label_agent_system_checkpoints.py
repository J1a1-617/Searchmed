#!/usr/bin/env python3
"""Attach an explicit Agent system identifier to a dedicated run root."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--agent-system-id", required=True)
    parser.add_argument("--modified-after-epoch", type=float, default=0.0)
    args = parser.parse_args()
    updated = 0
    for path in args.root.rglob("*.json"):
        if path.name != "manifest.json" and not path.name.startswith("case_"):
            continue
        if path.name != "manifest.json" and path.stat().st_mtime < args.modified_after_epoch:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        payload["agent_system_id"] = args.agent_system_id
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        updated += 1
    print(json.dumps({"agent_system_id": args.agent_system_id, "updated_files": updated}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
