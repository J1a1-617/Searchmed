"""Run A1-A4 LLM-Judge scoring for successful existing benchmark checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval.metrics import compute_composite_score
from eval.prompts import (
    format_judge_prompt_A1,
    format_judge_prompt_A2,
    format_judge_prompt_A3,
    format_judge_prompt_A4,
)
from eval.runner import run_benchmark
from searchagent_retrieval.llm_client import LLMClient


FORMATTERS = {
    "A1": format_judge_prompt_A1,
    "A2": format_judge_prompt_A2,
    "A3": format_judge_prompt_A3,
    "A4": format_judge_prompt_A4,
}
_local = threading.local()


def atomic_write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def client() -> LLMClient:
    current = getattr(_local, "client", None)
    if current is None:
        current = LLMClient(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model_name=os.environ.get("MODEL_NAME", "gpt-5"),
            timeout=180,
            max_retries=0,
            structured_attempts=1,
        )
        _local.client = current
    return current


def judge(prompt: str, model: str, attempts: int = 3) -> dict:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            parsed = client().call_function(
                system=(
                    "你是严格、客观的临床评测员。根据用户提供的病例、标准答案、"
                    "模型输出和评分细则评分，并且必须调用 submit_judge_score。"
                ),
                user=prompt,
                function_name="submit_judge_score",
                description="提交当前辅助维度的整数评分和简短理由。",
                parameters={
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 5,
                        },
                        "brief_reason": {"type": "string"},
                    },
                    "required": ["score", "brief_reason"],
                    "additionalProperties": False,
                },
                temperature=0.0,
                max_output_tokens=2048,
                max_attempts=1,
            )
            if not isinstance(parsed, dict) or "score" not in parsed:
                raise ValueError("Judge response did not contain a parseable score")
            parsed["score"] = max(0, min(5, int(parsed["score"])))
            return parsed
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(str(last_error))


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--judge-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    artifact_root = Path(args.artifact_root)
    judge_checkpoint = Path(args.judge_checkpoint)
    judge_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict] = {}
    for path in checkpoint_dir.glob("*.json"):
        if path.name == "manifest.json":
            continue
        try:
            record = load_json(path)
            if isinstance(record, dict) and record.get("status") == "success":
                records[str(record["instance_id"])] = record
        except (OSError, json.JSONDecodeError):
            continue

    valid_ids: set[str] = set()
    for instance_id in records:
        telemetry_path = artifact_root / instance_id / "telemetry.json"
        if not telemetry_path.exists():
            valid_ids.add(instance_id)
            continue
        try:
            telemetry = load_json(telemetry_path)
            calls = telemetry.get("llm_calls") or []
            tokens = sum(int(call.get("total_tokens") or 0) for call in calls)
            successful = sum(call.get("status") != "error" for call in calls)
            if tokens > 0 and successful > 0:
                valid_ids.add(instance_id)
        except (OSError, json.JSONDecodeError):
            continue

    raw_instances = load_json(Path(args.data))
    if isinstance(raw_instances, dict):
        for key in ("instances", "data", "questions"):
            if key in raw_instances:
                raw_instances = raw_instances[key]
                break
    instances = [
        instance for instance in raw_instances
        if str(instance["instance_id"]) in valid_ids
    ]
    instance_by_id = {str(instance["instance_id"]): instance for instance in instances}

    scores: dict[str, dict] = {}
    if judge_checkpoint.exists():
        loaded = load_json(judge_checkpoint)
        if isinstance(loaded, dict):
            scores = loaded

    tasks = []
    for instance_id in instance_by_id:
        model_output = json.dumps(
            records[instance_id]["result"]["parsed_output"],
            ensure_ascii=False,
        )
        for dimension, formatter in FORMATTERS.items():
            if dimension not in scores.get(instance_id, {}):
                tasks.append(
                    (instance_id, dimension, formatter(instance_by_id[instance_id], model_output))
                )

    print(
        f"[JUDGE] valid_cases={len(instances)} completed_items="
        f"{sum(len(value) for value in scores.values())} pending_items={len(tasks)}",
        flush=True,
    )
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(judge, prompt, args.model): (instance_id, dimension)
            for instance_id, dimension, prompt in tasks
        }
        for future in as_completed(futures):
            instance_id, dimension = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"[ERROR] {instance_id} {dimension}: {exc}", flush=True)
                continue
            with lock:
                scores.setdefault(instance_id, {})[dimension] = result
                atomic_write(judge_checkpoint, scores)
            print(
                f"[OK] {instance_id} {dimension} score={result['score']}",
                flush=True,
            )

    missing = [
        (instance_id, dimension)
        for instance_id in instance_by_id
        for dimension in FORMATTERS
        if dimension not in scores.get(instance_id, {})
    ]
    if missing:
        print(f"[INCOMPLETE] missing_items={len(missing)}; rerun to resume", flush=True)
        raise SystemExit(2)

    judged_records = {}
    for instance_id in instance_by_id:
        record = json.loads(json.dumps(records[instance_id], ensure_ascii=False))
        record["result"]["auxiliary"] = {
            dimension: int(scores[instance_id][dimension]["score"])
            for dimension in FORMATTERS
        }
        judged_records[instance_id] = record

    result = run_benchmark(
        instances,
        model_fn=lambda _: (_ for _ in ()).throw(RuntimeError("unexpected model call")),
        resume_records=judged_records,
        verbose=False,
    )
    result["judge_details"] = scores
    result["composite"] = compute_composite_score(
        result["global"], result["global"]["auxiliary_avg"]
    )
    atomic_write(Path(args.output), result)
    print(json.dumps({
        "global": result["global"],
        "composite": result["composite"],
        "meta": result["meta"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
