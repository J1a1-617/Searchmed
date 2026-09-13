#!/usr/bin/env python3
"""Compare generation with the real case-7 AnswerMemory versus no DB evidence."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from predictive_clinical_benchmark.eval.prompts import construct_inference_prompt
from predictive_clinical_benchmark.eval.runner import run_benchmark
from searchagent_retrieval.benchmark_prediction import BenchmarkPredictionGenerator
from searchagent_retrieval.llm_client import LLMClient


def main() -> None:
    case_id = "case_7_node_1"
    instances = json.loads(
        (REPO_ROOT / "predictive_clinical_benchmark/benchmark_multinode.json").read_text()
    )
    instance = next(row for row in instances if row["instance_id"] == case_id)
    session = json.loads(
        (
            REPO_ROOT
            / "fullcap_tests/predictive_benchmark_full/agent_artifacts/"
            "ig_priority_20260722/case_7_node_1/session/"
            "ig_priority_20260722__case_7_node_1.json"
        ).read_text()
    )
    real_loop_result = {"answer_memory": session["answer_memory"]}
    no_db_loop_result = {
        "answer_memory": {
            "claims": [],
            "evidence_by_id": {},
            "informative_rounds": [],
        }
    }

    results = {}
    for condition, loop_result in (
        ("real_retrieved_evidence", real_loop_result),
        ("no_db_evidence", no_db_loop_result),
    ):
        llm = LLMClient(timeout=900, max_retries=15, structured_attempts=2)
        llm.begin_trace(f"{condition}__{case_id}")
        generator = BenchmarkPredictionGenerator(llm_client=llm, use_llm=True)
        prediction = generator.generate(
            benchmark_prompt=construct_inference_prompt(instance),
            query="",
            loop_result=copy.deepcopy(loop_result),
            safety_result={},
            answer_context_summary={},
        )
        if generator.last_generation_source != "llm":
            raise RuntimeError(f"{condition}: {generator.last_error}")
        evaluation = run_benchmark(
            instances=[copy.deepcopy(instance)],
            model_fn=lambda _prompt, output=prediction: json.dumps(
                output, ensure_ascii=False
            ),
            verbose=False,
            workers=1,
        )
        results[condition] = {
            "prediction": prediction,
            "evaluation": evaluation,
            "llm_trace": llm.trace_events(),
            "generator_input_evidence": generator._compact_evidence(loop_result),
        }
        print(f"[{condition}] complete", flush=True)

    output = REPO_ROOT / "benchmark_results/case_7_node_1_real_vs_no_db.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
