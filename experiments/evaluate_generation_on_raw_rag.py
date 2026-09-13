#!/usr/bin/env python3
"""Evaluate the existing benchmark generation agent on raw RAG hits only.

This is an external ablation harness. It does not modify or execute the main
SearchAgent loop. For each selected completed case, it extracts the top raw
retrieval hits from the saved workflow trace, removes all downstream claims,
memory summaries, evidence reviews, and safety outputs, then calls the existing
BenchmarkPredictionGenerator and scores the outputs with the existing benchmark.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from predictive_clinical_benchmark.eval.prompts import construct_inference_prompt
from predictive_clinical_benchmark.eval.runner import run_benchmark
from searchagent_retrieval.benchmark_prediction import BenchmarkPredictionGenerator
from searchagent_retrieval.llm_client import LLMClient


DEFAULT_CASE_IDS = (
    "case_7_node_1",
    "case_7_node_2",
    "case_6_node_2",
    "case_6_node_3",
    "case_6_node_4",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def raw_rag_hits(trace: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    """Collect unique raw search hits in trace/rank order."""
    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for step in trace.get("steps") or []:
        if step.get("name") != "MultiStepPlanAndRetrieval":
            continue
        for call in step.get("tool_trace") or []:
            tool_name = str(call.get("tool_name") or "")
            if tool_name not in {
                "hybrid_search",
                "dense_search",
                "bm25_search",
                "structured_search",
            }:
                continue
            for hit in (call.get("result") or {}).get("hits") or []:
                if not isinstance(hit, dict):
                    continue
                evidence_id = str(hit.get("chunk_id") or hit.get("id") or "").strip()
                text = " ".join(str(hit.get("text") or "").split())
                if not evidence_id or not text or evidence_id in seen:
                    continue
                seen.add(evidence_id)
                hits.append(copy.deepcopy(hit))
                if len(hits) >= limit:
                    return hits
    return hits


def evidence_only_loop_result(hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the minimum input accepted by the unmodified generator."""
    evidence_by_id: dict[str, dict[str, Any]] = {}
    for hit in hits:
        evidence_id = str(hit.get("chunk_id") or hit.get("id"))
        evidence_by_id[evidence_id] = {
            **hit,
            "text": str(hit.get("text") or ""),
            "chunk_type": str(hit.get("chunk_type") or "case"),
            "evidence_level": str(hit.get("evidence_level") or hit.get("chunk_type") or "case"),
            "pub_date": str(
                hit.get("pub_date")
                or hit.get("publication_date")
                or hit.get("date")
                or "NA"
            ),
        }
    return {
        "answer_memory": {
            "claims": [],
            "evidence_by_id": evidence_by_id,
            "informative_rounds": [],
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("predictive_clinical_benchmark/benchmark_multinode.json"),
    )
    parser.add_argument(
        "--artifact-run",
        type=Path,
        default=Path(
            "benchmark_results/searchagent_full_135_retry_20260727_artifacts/"
            "run_20260727_150010"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "benchmark_results/searchagent_full_135_retry_20260727_checkpoints"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/raw_rag_generation_ablation_5.json"),
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--case-ids", nargs="+", default=list(DEFAULT_CASE_IDS))
    args = parser.parse_args()

    all_instances = load_json(args.data)
    instance_by_id = {str(row["instance_id"]): row for row in all_instances}
    instances = [instance_by_id[case_id] for case_id in args.case_ids]

    api_keys = [
        value.strip()
        for value in os.environ.get("OPENAI_API_KEYS", "").split(",")
        if value.strip()
    ]
    api_key = api_keys[0] if api_keys else None
    llm = LLMClient(
        api_key=api_key,
        timeout=float(os.environ.get("LLM_TIMEOUT") or 900),
        max_retries=int(os.environ.get("LLM_MAX_RETRIES") or 10),
        structured_attempts=int(os.environ.get("LLM_STRUCTURED_ATTEMPTS") or 1),
    )
    generator = BenchmarkPredictionGenerator(llm_client=llm, use_llm=True)

    generated: dict[str, dict[str, Any]] = {}
    evidence_audit: dict[str, Any] = {}
    generation_audit: dict[str, Any] = {}
    for instance in instances:
        case_id = str(instance["instance_id"])
        trace = load_json(args.artifact_run / case_id / "workflow_trace.json")
        hits = raw_rag_hits(trace, limit=max(1, args.top_k))
        if not hits:
            raise RuntimeError(f"{case_id}: no raw RAG hits in workflow trace")
        loop_result = evidence_only_loop_result(hits)
        llm.begin_trace(f"raw-rag-generation-ablation__{case_id}")
        prediction = generator.generate(
            benchmark_prompt=construct_inference_prompt(instance),
            query="",
            loop_result=loop_result,
            safety_result={},
            answer_context_summary={},
        )
        if generator.last_generation_source != "llm":
            raise RuntimeError(
                f"{case_id}: generation fell back: {generator.last_error}"
            )
        generated[case_id] = prediction
        evidence_audit[case_id] = {
            "raw_hit_count": len(hits),
            "chunk_ids": [
                str(hit.get("chunk_id") or hit.get("id")) for hit in hits
            ],
            "tools": sorted(
                {
                    str(hit.get("source") or "unknown")
                    for hit in hits
                }
            ),
        }
        generation_audit[case_id] = {
            "source": generator.last_generation_source,
            "llm_calls": llm.trace_events(),
        }
        print(f"[{case_id}] generated from {len(hits)} raw RAG hits", flush=True)

    def raw_rag_model_fn(_prompt: str, instance: dict[str, Any]) -> str:
        return json.dumps(generated[str(instance["instance_id"])], ensure_ascii=False)

    ablation_results = run_benchmark(
        instances=copy.deepcopy(instances),
        model_fn=lambda _: "",
        instance_model_fn=raw_rag_model_fn,
        verbose=False,
        workers=1,
    )

    baseline_records = {
        case_id: load_json(args.checkpoint_dir / f"{case_id}.json")
        for case_id in args.case_ids
    }
    baseline_results = run_benchmark(
        instances=copy.deepcopy(instances),
        model_fn=lambda _: "",
        verbose=False,
        workers=1,
        resume_records=baseline_records,
    )

    payload = {
        "experiment": {
            "name": "raw_rag_evidence_to_existing_benchmark_generation_agent",
            "case_ids": args.case_ids,
            "top_k": args.top_k,
            "removed_inputs": [
                "agent clinical claims",
                "step/replanner/answer memory summaries",
                "evidence review",
                "safety gate output",
                "answer context summary",
            ],
            "retained_inputs": [
                "benchmark case information available at cutoff",
                "raw RAG hit text in retrieval rank order",
            ],
        },
        "evidence_audit": evidence_audit,
        "generation_audit": generation_audit,
        "ablation": ablation_results,
        "full_agent_baseline_same_5": baseline_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
