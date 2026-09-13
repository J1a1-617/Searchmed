#!/usr/bin/env python3
"""Select a diverse benchmark batch without reusing prior patients."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CASE_RE = re.compile(r"^(case_\d+)_node_\d+$")


def _load(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array: {path}")
    return [row for row in value if isinstance(row, dict)]


def _case_key(instance_id: str) -> str:
    match = CASE_RE.match(instance_id)
    return match.group(1) if match else instance_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    excluded_ids: set[str] = set()
    excluded_cases: set[str] = set()
    for path in args.exclude:
        for row in _load(path):
            instance_id = str(row.get("instance_id") or "")
            if instance_id:
                excluded_ids.add(instance_id)
                excluded_cases.add(_case_key(instance_id))

    selected: list[dict[str, Any]] = []
    selected_cases: set[str] = set()
    for row in _load(args.source):
        instance_id = str(row.get("instance_id") or "")
        case_key = _case_key(instance_id)
        if (
            not instance_id
            or instance_id in excluded_ids
            or case_key in excluded_cases
            or case_key in selected_cases
        ):
            continue
        selected.append(row)
        selected_cases.add(case_key)
        if len(selected) >= args.size:
            break

    if len(selected) != args.size:
        raise RuntimeError(f"requested {args.size} diverse cases, found {len(selected)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "instance_ids": [row["instance_id"] for row in selected]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
