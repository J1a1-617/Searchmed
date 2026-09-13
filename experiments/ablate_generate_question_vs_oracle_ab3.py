#!/usr/bin/env python3
"""Ablate the same Generate component with question-only vs full oracle GT."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predictive_clinical_benchmark.eval.prompts import construct_inference_prompt
from predictive_clinical_benchmark.eval.runner import run_benchmark
from searchagent_retrieval.benchmark_prediction import BenchmarkPredictionGenerator
from searchagent_retrieval.llm_client import LLMClient

CASE_IDS = ("case_1_node_2", "case_3_node_2", "case_5_node_1")
OUTPUT = ROOT / "benchmark_results/ab3_same_generate_question_vs_full_oracle_20260805.json"


def load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def score(instances: list[dict[str, Any]], outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return run_benchmark(
        instances=instances,
        model_fn=lambda _: "",
        instance_model_fn=lambda _prompt, inst: json.dumps(
            outputs[str(inst["instance_id"])], ensure_ascii=False
        ),
        verbose=False,
        workers=1,
    )


def main() -> None:
    all_instances = load(ROOT / "predictive_clinical_benchmark/benchmark_multinode.json")
    by_id = {str(row["instance_id"]): row for row in all_instances}
    instances = [by_id[case_id] for case_id in CASE_IDS]
    llm = LLMClient(
        timeout=float(os.environ.get("LLM_TIMEOUT") or 900),
        max_retries=int(os.environ.get("LLM_MAX_RETRIES") or 10),
        structured_attempts=int(os.environ.get("LLM_STRUCTURED_ATTEMPTS") or 2),
    )
    generator = BenchmarkPredictionGenerator(llm_client=llm, use_llm=True)
    conditions: dict[str, Any] = {}

    for condition in ("question_only", "full_oracle"):
        outputs: dict[str, dict[str, Any]] = {}
        audit: dict[str, Any] = {}
        for instance in instances:
            case_id = str(instance["instance_id"])
            prompt = construct_inference_prompt(instance)
            if condition == "full_oracle":
                prompt += (
                    "\n\n[ORACLE ABLATION ONLY — KNOWN TRUE FOLLOW-UP LABELS]\n"
                    + json.dumps(instance.get("ground_truth") or {}, ensure_ascii=False)
                    + "\nFor this controlled ablation, reproduce these known labels in the required output schema."
                )
            llm.begin_trace(f"same-generate-{condition}__{case_id}")
            output = generator.generate(
                benchmark_prompt=prompt,
                query="",
                loop_result={"answer_memory": {"claims": [], "evidence_by_id": {}}},
                safety_result={},
                answer_context_summary={},
            )
            if generator.last_generation_source != "llm":
                raise RuntimeError(f"{condition}/{case_id}: {generator.last_error}")
            events = llm.trace_events()
            outputs[case_id] = output
            audit[case_id] = {
                "prediction": output,
                "llm_calls": events,
                "llm_call_count": len(events),
                "duration_seconds": sum(float(e.get("duration_seconds") or 0) for e in events),
                "total_tokens": sum(int(e.get("total_tokens") or 0) for e in events),
            }
            print(f"[{condition}/{case_id}] calls={len(events)}", flush=True)
        conditions[condition] = {"audit": audit, "results": score(instances, outputs)}

    payload = {
        "experiment": {
            "name": "same_generate_question_only_vs_full_ground_truth_oracle",
            "case_ids": list(CASE_IDS),
            "retrieval_or_context_used": False,
            "warning": "full_oracle intentionally exposes post-cutoff ground truth and is only a Generate upper-bound diagnostic.",
        },
        "conditions": conditions,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[saved] {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
