from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def load_local_env(path: Path) -> dict[str, str]:
    env = dict(os.environ)
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    # The suite favors bounded, comparable runs. Production defaults remain
    # configurable in .env, but one slow provider request must not stall 10 cases.
    env["LLM_TIMEOUT"] = os.environ.get("E2E_LLM_TIMEOUT", "45")
    env["LLM_MAX_RETRIES"] = os.environ.get("E2E_LLM_MAX_RETRIES", "0")
    env["LLM_MAX_OUTPUT_TOKENS"] = os.environ.get("E2E_LLM_MAX_OUTPUT_TOKENS", "1800")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run diverse SearchAgent end-to-end cases.")
    parser.add_argument("--questions", type=Path, default=Path("e2e_questions.json"))
    parser.add_argument("--output-root", type=Path, default=Path("e2e_workflows"))
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--case-timeout", type=int, default=600)
    args = parser.parse_args()

    cases = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = cases[max(0, args.start - 1) : args.end]
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = load_local_env(Path(".env"))
    summary_path = args.output_root / "summary.json"
    summary = []
    if args.start > 1 and summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = [item for item in existing if int(str(item.get("id", "0")).split("_", 1)[0]) < args.start]

    for index, case in enumerate(selected, start=args.start):
        case_id = str(case["id"])
        workflow_path = args.output_root / f"{case_id}_workflow.txt"
        stderr_path = args.output_root / f"{case_id}_stderr.txt"
        command = [
            sys.executable,
            "-u",
            "run_agent_loop.py",
            "--query",
            str(case["query"]),
            "--max-steps",
            str(args.max_steps),
            "--top-k",
            str(args.top_k),
            "--session-id",
            f"e2e_{case_id}",
            "--print-workflow",
        ]
        print(f"[{index}/{args.end}] START {case_id}: {case['query']}", flush=True)
        started = time.monotonic()
        status = "completed"
        returncode = None
        with workflow_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_file:
            try:
                completed = subprocess.run(
                    command,
                    cwd=Path.cwd(),
                    env=env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    timeout=args.case_timeout,
                    check=False,
                )
                returncode = completed.returncode
                if returncode != 0:
                    status = "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
        elapsed = round(time.monotonic() - started, 2)
        stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
        summary.append(
            {
                "id": case_id,
                "query": case["query"],
                "status": status,
                "returncode": returncode,
                "elapsed_seconds": elapsed,
                "workflow_file": str(workflow_path),
                "stderr_file": str(stderr_path),
                "fallback_count": stderr_text.lower().count("falling back")
                + stderr_text.lower().count("retaining rule result"),
            }
        )
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[{index}/{args.end}] END {case_id}: {status}, {elapsed}s", flush=True)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
