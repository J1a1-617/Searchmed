#!/usr/bin/env python3
"""Run all benchmark batches through one resumable, strictly serial queue.

The supervisor may be started while an earlier benchmark PID is active. It
captures that process's API environment, waits for it to finish, and then runs
each remaining batch one after another. Existing successful checkpoint rows are
skipped on restart; errored rows are retried.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, Iterable, List, Set


DONE_STATUSES = {"complete", "degraded"}


def _read_process_environment(pid: int) -> Dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    environment: Dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        environment[key.decode(errors="replace")] = value.decode(errors="replace")
    return environment


def _checkpoint_rows(output_dir: Path) -> Iterable[Dict[str, Any]]:
    for path in sorted(output_dir.glob("ckpt_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _completed_ids(output_dir: Path) -> Set[str]:
    return {
        str(row.get("instance_id"))
        for row in _checkpoint_rows(output_dir)
        if row.get("instance_id") and str(row.get("run_status")) in DONE_STATUSES
    }


def _write_state(path: Path, **state: Any) -> None:
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    captured = _read_process_environment(args.wait_pid)
    child_env = os.environ.copy()
    for key in ("DEFAULT_BASE_URL", "DEFAULT_OPENAI_API_KEY", "DEFAULT_OPENAI_MODEL"):
        if captured.get(key):
            child_env[key] = captured[key]
    child_env["PYTHONPATH"] = f"{root}:{root / 'benchmark_package' / 'code'}"

    print(f"waiting_for_pid={args.wait_pid}", flush=True)
    while Path(f"/proc/{args.wait_pid}").exists():
        time.sleep(30)
    print("previous_batch_finished", flush=True)

    output_root = root / "results/full_serial_queue"
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "queue_state.json"
    runner = root / "benchmark_package/scripts/smoke_agent_comutation.py"
    python = Path("/home/visitor/yangijiayi_legacy_20260810/conda-envs/yangijiayi/bin/python")

    # First revisit the just-finished batch. Its existing checkpoint makes the
    # retry set contain only errored/interrupted cases (for example case09), so
    # batch_001 ends with ten usable trajectories instead of nine successes and
    # one silently omitted error.
    queue = [(
        "batch_001_recovery",
        root / "benchmark_package/data/skill_dev_batch_001_remaining_04_10.json",
        root / "results/local_rerun_04_10",
    )]
    later_batches = [root / "benchmark_package/data/skill_dev_batch_002.json"]
    later_batches.extend(sorted((root / "benchmark_package/data/full_queue").glob("batch_*.json")))
    queue.extend(
        (f"batch_{queue_index:03d}", batch_path, output_root / f"batch_{queue_index:03d}")
        for queue_index, batch_path in enumerate(later_batches, start=2)
    )

    for batch_id, batch_path, output_dir in queue:
        output_dir.mkdir(parents=True, exist_ok=True)
        cases = json.loads(batch_path.read_text(encoding="utf-8"))
        completed = _completed_ids(output_dir)
        remaining = [row for row in cases if str(row.get("instance_id")) not in completed]
        remaining_path = output_dir / "remaining.json"
        remaining_path.write_text(
            json.dumps(remaining, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_state(
            state_path,
            status="running" if remaining else "batch_already_complete",
            active_batch=batch_id,
            source=str(batch_path),
            batch_total=len(cases),
            already_complete=len(completed),
            remaining=len(remaining),
        )
        if not remaining:
            print(f"{batch_id}: already complete", flush=True)
            continue

        print(f"{batch_id}: starting {len(remaining)} cases", flush=True)
        command = [
            str(python), str(runner),
            "--data", str(remaining_path),
            "--max-cases", str(len(remaining)),
            "--index-root", str(root / "indexes"),
            "--embed-model-path", str(root / "models/bge-large-zh-v1.5"),
            "--top-k", "8",
            "--max-steps", "2",
            "--max-total-steps", "2",
            "--temporal-filter-mode", "cutoff",
            "--out-dir", str(output_dir),
        ]
        return_code = subprocess.run(
            command, cwd=root, env=child_env, check=False,
        ).returncode
        _write_state(
            state_path,
            status="batch_finished" if return_code == 0 else "batch_process_error",
            active_batch=batch_id,
            source=str(batch_path),
            return_code=return_code,
            completed_after_run=len(_completed_ids(output_dir)),
            batch_total=len(cases),
        )
        print(f"{batch_id}: process_return_code={return_code}", flush=True)

    _write_state(state_path, status="complete", active_batch=None)
    print("serial_queue_complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
