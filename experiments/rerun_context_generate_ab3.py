#!/usr/bin/env python3
"""Re-run only AnswerContext and Generate on three saved agent sessions."""

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
from searchagent_retrieval.answer_context import AnswerContextAgent
from searchagent_retrieval.benchmark_prediction import BenchmarkPredictionGenerator
from searchagent_retrieval.llm_client import LLMClient

CASE_IDS = ("case_1_node_2", "case_3_node_2", "case_5_node_1")
SESSION_ROOT = ROOT / "benchmark_results/ab3_agent_gpt5_compact_review_20260805_agent_artifacts/run_20260805_161900"
OUTPUT = ROOT / "benchmark_results/ab3_context_generate_bounded_20260805.json"


def load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    instances = load(ROOT / "predictive_clinical_benchmark/benchmark_multinode.json")
    by_id = {str(row["instance_id"]): row for row in instances}
    selected = [by_id[case_id] for case_id in CASE_IDS]
    llm = LLMClient(
        timeout=float(os.environ.get("LLM_TIMEOUT") or 900),
        max_retries=int(os.environ.get("LLM_MAX_RETRIES") or 10),
        structured_attempts=int(os.environ.get("LLM_STRUCTURED_ATTEMPTS") or 2),
    )
    context_agent = AnswerContextAgent(llm_client=llm, use_llm=True)
    generator = BenchmarkPredictionGenerator(llm_client=llm, use_llm=True)
    outputs: dict[str, dict[str, Any]] = {}
    audits: dict[str, Any] = {}

    for instance in selected:
        case_id = str(instance["instance_id"])
        session_path = next((SESSION_ROOT / case_id / "session").glob("*.json"))
        session = load(session_path)
        # Current runtime shape contains state.confirmed_constraints; saved
        # sessions persist that field at the top level.
        loop_result = dict(session)
        loop_result["state"] = {
            "confirmed_constraints": session.get("confirmed_constraints") or {}
        }
        safety = session.get("final_safety_review") or {}
        query = str(session.get("original_query") or "")
        llm.begin_trace(f"context-generate-only__{case_id}")
        context = context_agent.summarize(
            query=query,
            loop_result=loop_result,
            safety_result=safety,
        )
        output = generator.generate(
            benchmark_prompt=construct_inference_prompt(instance),
            query=query,
            loop_result=loop_result,
            safety_result=safety,
            answer_context_summary=context,
        )
        if generator.last_generation_source != "llm":
            raise RuntimeError(f"{case_id}: Generate failed: {generator.last_error}")
        events = llm.trace_events()
        outputs[case_id] = output
        audits[case_id] = {
            "source_session": str(session_path.relative_to(ROOT)),
            "answer_context": context,
            "generate_payload": generator.last_input_payload,
            "prediction": output,
            "llm_calls": events,
            "llm_call_count": len(events),
            "elapsed_seconds": sum(float(event.get("duration_seconds") or 0) for event in events),
        }
        print(
            f"[{case_id}] context+generate complete; calls={len(events)}; "
            f"bounded={len((generator.last_input_payload or {}).get('bounded_evidence') or [])}",
            flush=True,
        )

    def predict(_prompt: str, instance: dict[str, Any]) -> str:
        return json.dumps(outputs[str(instance["instance_id"])], ensure_ascii=False)

    results = run_benchmark(
        instances=selected,
        model_fn=lambda _: "",
        instance_model_fn=predict,
        verbose=False,
        workers=1,
    )
    payload = {
        "experiment": {
            "name": "context_generate_only_with_bounded_partial_analog_evidence",
            "case_ids": list(CASE_IDS),
            "retrieval_rerank_memory_rerun": False,
            "components_rerun": ["AnswerContextAgent", "BenchmarkPredictionGenerator"],
        },
        "audit": audits,
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[saved] {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
