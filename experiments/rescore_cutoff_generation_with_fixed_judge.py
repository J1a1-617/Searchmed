#!/usr/bin/env python3
"""Rescore saved GPT-4o/GPT-5 predictions with one reliable fixed judge."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from predictive_clinical_benchmark.eval.parser import parse_model_output
from predictive_clinical_benchmark.eval.runner import run_benchmark
from searchagent_retrieval.llm_client import LLMClient


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmark_results/cutoff_generation_gpt4o_vs_gpt5_5.json"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("predictive_clinical_benchmark/benchmark_multinode.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark_results/cutoff_generation_gpt4o_vs_gpt5_5_rescored.json"
        ),
    )
    parser.add_argument("--judge-model", default="gpt-4o")
    args = parser.parse_args()

    source = load_json(args.input)
    case_ids = source["experiment"]["case_ids"]
    models = source["experiment"]["models"]
    instance_by_id = {
        str(row["instance_id"]): row for row in load_json(args.data)
    }
    instances = [instance_by_id[case_id] for case_id in case_ids]

    outputs: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name in models:
        outputs[model_name] = {}
        for row in source["results"][model_name]["per_instance"]:
            prediction = dict(row["parsed_output"])
            prediction.pop("_ground_truth", None)
            outputs[model_name][str(row["instance_id"])] = prediction

    configured_keys = [
        value.strip()
        for value in os.environ.get("OPENAI_API_KEYS", "").split(",")
        if value.strip()
    ]
    judge = LLMClient(
        api_key=configured_keys[0] if configured_keys else None,
        model_name=args.judge_model,
        timeout=float(os.environ.get("LLM_TIMEOUT") or 900),
        max_retries=int(os.environ.get("LLM_MAX_RETRIES") or 10),
        structured_attempts=1,
    )
    judge.begin_trace("cutoff-generation-reliable-fixed-judge")

    def judge_fn(prompt: str) -> dict[str, Any]:
        raw = judge.chat(
            system="你是严格的临床预测评测员。只按用户提供的评分规则输出要求的JSON。",
            user=prompt,
            temperature=0.0,
            max_output_tokens=4000,
        )
        parsed = parse_model_output(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("score"), int):
            raise RuntimeError("judge returned no integer score")
        return parsed

    rescored: dict[str, Any] = {}
    for model_name in models:
        def instance_model_fn(_prompt: str, instance: dict[str, Any]) -> str:
            return json.dumps(
                outputs[model_name][str(instance["instance_id"])],
                ensure_ascii=False,
            )

        rescored[model_name] = run_benchmark(
            instances=instances,
            model_fn=lambda _: "",
            instance_model_fn=instance_model_fn,
            llm_judge_fn=judge_fn,
            verbose=False,
            workers=1,
        )
        print(f"[{model_name}] rescored", flush=True)

    source["experiment"]["judge_model"] = args.judge_model
    source["experiment"]["judge_rescore"] = (
        "Predictions and cutoff-safe evidence unchanged; A1-A4 replaced using "
        "one fixed reliable judge."
    )
    source["results"] = rescored
    source["judge_audit"] = judge.trace_events()
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(source, handle, ensure_ascii=False, indent=2)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
