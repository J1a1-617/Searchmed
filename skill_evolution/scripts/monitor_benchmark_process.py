#!/usr/bin/env python3
"""Poll a nohup benchmark and terminate a genuinely stale child process."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import signal
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def child_pids(pid: int) -> list[int]:
    """Return every live descendant, not just the direct wrapper shell."""
    descendants = []
    pending = [pid]
    seen = {pid}
    while pending:
        parent = pending.pop(0)
        children = Path(f"/proc/{parent}/task/{parent}/children")
        if not children.is_file():
            continue
        try:
            values = children.read_text().split()
        except OSError:
            continue
        for value in values:
            if not value.isdigit():
                continue
            child = int(value)
            if child in seen:
                continue
            seen.add(child)
            descendants.append(child)
            pending.append(child)
    return descendants


def checkpoint_summary(root: Path, agent_system_id: Optional[str]) -> dict:
    counts: Counter[str] = Counter()
    latest_path = None
    latest_mtime = 0.0
    for path in root.rglob("case_*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            counts["broken"] += 1
            continue
        if agent_system_id and record.get("agent_system_id") != agent_system_id:
            continue
        counts[str(record.get("status") or "unknown")] += 1
        mtime = path.stat().st_mtime
        if mtime > latest_mtime:
            latest_path, latest_mtime = path, mtime
    return {
        "status_counts": dict(counts),
        "complete_cases": counts["success"],
        "latest_checkpoint": str(latest_path) if latest_path else None,
        "latest_checkpoint_age_seconds": round(max(0.0, time.time() - latest_mtime), 1) if latest_mtime else None,
    }


def timing_summary(log_path: Path) -> dict:
    if not log_path.is_file():
        return {"completed_with_telemetry": 0}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows = re.findall(
        r"total=([0-9.]+)s\s+llm_calls=(\d+)\s+llm_errors=(\d+)\s+tokens=(\d+)",
        text,
    )
    if not rows:
        return {"completed_with_telemetry": 0}
    seconds = [float(row[0]) for row in rows]
    return {
        "completed_with_telemetry": len(rows),
        "case_seconds_median": round(statistics.median(seconds), 1),
        "case_seconds_max": round(max(seconds), 1),
        "llm_calls_total": sum(int(row[1]) for row in rows),
        "llm_errors_total": sum(int(row[2]) for row in rows),
        "tokens_total": sum(int(row[3]) for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--agent-system-id", default=None)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--stale-seconds", type=int, default=2100)
    args = parser.parse_args()
    stale_events = 0
    last_killed_child = None
    while Path(f"/proc/{args.pid}").exists():
        log_mtime = args.log.stat().st_mtime if args.log.is_file() else 0.0
        stale = max(0.0, time.time() - log_mtime)
        progress = checkpoint_summary(args.checkpoint_root, args.agent_system_id)
        state = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "queue_pid": args.pid,
            "child_pids": child_pids(args.pid),
            **progress,
            "timing": timing_summary(args.log),
            "log_stale_seconds": round(stale, 1),
            "stale_events": stale_events,
            "status": "stale" if stale > args.stale_seconds else "running",
        }
        atomic_write(args.state, state)
        if stale > args.stale_seconds:
            # Terminate leaf workers before their wrapper shells.
            for pid in reversed(state["child_pids"]):
                if pid == last_killed_child:
                    continue
                try:
                    os.kill(pid, signal.SIGTERM)
                    last_killed_child = pid
                    stale_events += 1
                except ProcessLookupError:
                    pass
        time.sleep(max(10, args.interval))
    atomic_write(args.state, {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "queue_pid": args.pid,
        "stale_events": stale_events,
        **checkpoint_summary(args.checkpoint_root, args.agent_system_id),
        "timing": timing_summary(args.log),
        "status": "finished",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
