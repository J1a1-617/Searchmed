#!/usr/bin/env python3
"""Resume the GPU queue with per-case retries while preserving every attempt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, Iterable, List


STATUS_RANK = {"error": 0, "failed": 0, "degraded": 1, "complete": 2}


def process_env(pid: int) -> Dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    result: Dict[str, str] = {}
    for item in raw.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            result[key.decode(errors="replace")] = value.decode(errors="replace")
    return result


def rows_in(path: Path) -> Iterable[Dict[str, Any]]:
    for checkpoint in sorted(path.rglob("ckpt_*.jsonl")):
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("instance_id"):
                yield row


def best_row(path: Path, instance_id: str) -> Dict[str, Any] | None:
    candidates = [
        row for row in rows_in(path)
        if str(row.get("instance_id")) == instance_id
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: STATUS_RANK.get(str(row.get("run_status")), 0))


def write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    root = args.root.resolve()
    captured = process_env(args.wait_pid)
    env = os.environ.copy()
    for key in ("DEFAULT_BASE_URL", "DEFAULT_OPENAI_API_KEY", "DEFAULT_OPENAI_MODEL"):
        if captured.get(key):
            env[key] = captured[key]
    env["PYTHONPATH"] = f"{root}:{root / 'benchmark_package' / 'code'}"

    print(f"waiting_for_pid={args.wait_pid}", flush=True)
    while Path(f"/proc/{args.wait_pid}").exists():
        time.sleep(30)
    print("previous_case_or_batch_finished", flush=True)

    data_root = root / "benchmark_package/data"
    sources: List[Path] = [data_root / "skill_dev_batch_002.json"]
    sources.extend(sorted((data_root / "full_queue").glob("batch_*.json")))
    output_root = root / "results/full_retry_queue"
    output_root.mkdir(parents=True, exist_ok=True)
    runner = root / "benchmark_package/scripts/smoke_agent_comutation.py"
    python = Path("/home/visitor/yangijiayi_legacy_20260810/conda-envs/yangijiayi/bin/python")

    for batch_number, source in enumerate(sources, start=2):
        batch_id = f"batch_{batch_number:03d}"
        batch_out = output_root / batch_id
        batch_out.mkdir(parents=True, exist_ok=True)
        attempts_root = batch_out / "attempts"
        selected_path = batch_out / "selected.jsonl"
        cases = json.loads(source.read_text(encoding="utf-8"))
        for case in cases:
            instance_id = str(case["instance_id"])
            existing = best_row(root / "results/full_serial_queue" / batch_id, instance_id)
            if existing is not None and str(existing.get("run_status")) == "complete":
                write_jsonl(selected_path, {"instance_id": instance_id, "selected_from": "original", "selected": existing})
                continue

            case_attempts = attempts_root / instance_id
            case_attempts.mkdir(parents=True, exist_ok=True)
            attempt_rows: List[Dict[str, Any]] = []
            if existing is not None:
                attempt_rows.append(existing)
            start_attempt = len(attempt_rows) + 1
            for attempt in range(start_attempt, args.max_attempts + 1):
                attempt_out = case_attempts / f"attempt_{attempt:02d}"
                attempt_out.mkdir(parents=True, exist_ok=True)
                input_path = attempt_out / "input.json"
                input_path.write_text(json.dumps([case], ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"{batch_id} {instance_id}: attempt {attempt}/{args.max_attempts}", flush=True)
                command = [
                    str(python), str(runner), "--data", str(input_path), "--max-cases", "1",
                    "--index-root", str(root / "indexes"),
                    "--embed-model-path", str(root / "models/bge-large-zh-v1.5"),
                    "--top-k", "8", "--max-steps", "2", "--max-total-steps", "2",
                    "--temporal-filter-mode", "cutoff", "--out-dir", str(attempt_out),
                ]
                subprocess.run(command, cwd=root, env=env, check=False)
                row = best_row(attempt_out, instance_id)
                if row is not None:
                    attempt_rows.append(row)
                if row is not None and str(row.get("run_status")) == "complete":
                    break

            selected = max(
                attempt_rows,
                key=lambda row: STATUS_RANK.get(str(row.get("run_status")), 0),
            ) if attempt_rows else {"instance_id": instance_id, "run_status": "error", "error": "no checkpoint"}
            write_jsonl(selected_path, {
                "instance_id": instance_id,
                "selected_from": "retry_attempts",
                "attempt_count": len(attempt_rows),
                "selected": selected,
                "all_attempt_statuses": [str(row.get("run_status")) for row in attempt_rows],
            })

    (output_root / "queue_status.json").write_text(
        json.dumps({"status": "complete", "max_attempts": args.max_attempts}, indent=2),
        encoding="utf-8",
    )
    print("retry_queue_complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
