#!/usr/bin/env python3
"""Build deterministic diverse batches covering every untested benchmark node."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


CASE_RE = re.compile(r"^(case_\d+)_node_(\d+)$")


def _load(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array: {path}")
    return [row for row in value if isinstance(row, dict) and row.get("instance_id")]


def _case_and_node(instance_id: str) -> tuple[str, int]:
    match = CASE_RE.match(instance_id)
    return (match.group(1), int(match.group(2))) if match else (instance_id, 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=3)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    excluded: set[str] = set()
    for path in args.exclude:
        excluded.update(str(row["instance_id"]) for row in _load(path))

    groups: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    source = _load(args.source)
    source_order: dict[str, int] = {}
    for position, row in enumerate(source):
        instance_id = str(row["instance_id"])
        if instance_id in excluded:
            continue
        case_key, node = _case_and_node(instance_id)
        source_order.setdefault(case_key, position)
        groups[case_key].append((node, position, row))  # type: ignore[arg-type]
    for case_key, entries in list(groups.items()):
        groups[case_key] = deque(row for _, _, row in sorted(entries, key=lambda item: (item[0], item[1])))

    case_order = sorted(groups, key=lambda key: source_order[key])
    queue: list[dict[str, Any]] = []
    while groups:
        for case_key in list(case_order):
            entries = groups.get(case_key)
            if not entries:
                groups.pop(case_key, None)
                case_order.remove(case_key)
                continue
            queue.append(entries.popleft())
            if not entries:
                groups.pop(case_key, None)
                case_order.remove(case_key)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_batches = []
    for offset in range(0, len(queue), args.batch_size):
        batch_index = args.start_index + offset // args.batch_size
        rows = queue[offset : offset + args.batch_size]
        path = args.output_root / f"batch_{batch_index:03d}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest_batches.append({
            "batch_index": batch_index,
            "path": str(path),
            "instance_ids": [str(row["instance_id"]) for row in rows],
        })

    manifest = {
        "source_count": len(source),
        "excluded_count": len(excluded),
        "queued_count": len(queue),
        "batch_size": args.batch_size,
        "batches": manifest_batches,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "queued_count": len(queue), "batch_count": len(manifest_batches)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
