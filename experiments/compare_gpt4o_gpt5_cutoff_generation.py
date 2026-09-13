#!/usr/bin/env python3
"""Compare GPT-4o and GPT-5 generation on identical cutoff-safe RAG evidence.

The harness uses the repository's current deterministic Hybrid retrieval with a
hard publication cutoff, freezes the resulting evidence per case, and feeds the
same evidence to the current BenchmarkPredictionGenerator under each model.
It also runs one fixed judge model for A1-A4 so composite scores share a common
evaluation protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from predictive_clinical_benchmark.eval.parser import parse_model_output
from predictive_clinical_benchmark.eval.prompts import (
    construct_inference_prompt,
    format_judge_prompt_A1,
    format_judge_prompt_A2,
    format_judge_prompt_A3,
    format_judge_prompt_A4,
)
from predictive_clinical_benchmark.eval.runner import run_benchmark
from searchagent_retrieval.benchmark_prediction import BenchmarkPredictionGenerator
from searchagent_retrieval.llm_client import LLMClient
from searchagent_retrieval.tools import RetrievalTools


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


def flatten_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        rows: list[str] = []
        for nested in value.values():
            rows.extend(flatten_values(nested))
        return rows
    if isinstance(value, list):
        rows = []
        for nested in value:
            rows.extend(flatten_values(nested))
        return rows
    text = str(value or "").strip()
    return [text] if text else []


def retrieval_inputs(instance: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    inp = instance.get("input") or {}
    background = inp.get("disease_background") or {}
    molecular = background.get("molecular_profile") or {}
    current = inp.get("current_status") or {}
    treatment = inp.get("planned_treatment") or {}
    drugs = [
        str(row.get("name") or "").strip()
        for row in treatment.get("drugs") or []
        if isinstance(row, dict) and str(row.get("name") or "").strip()
    ]
    mutations = flatten_values(molecular)
    sites = [str(value) for value in background.get("metastatic_sites") or []]
    query_parts = [
        str(background.get("diagnosis") or ""),
        " ".join(mutations),
        " ".join(drugs),
        str(treatment.get("combination_strategy") or ""),
        " ".join(sites),
        str(current.get("imaging") or ""),
        "8-12周 疗效 RECIST 症状 毒性 不良事件",
    ]
    query = " ".join(" ".join(query_parts).split())
    constraints = {
        "cancer_type": str(background.get("diagnosis") or ""),
        "gene_alterations": mutations,
        "drugs": drugs,
        "metastatic_sites": sites,
        "responses": [],
        "toxicities": [],
        "ddi_terms": [],
    }
    return query, constraints


def evidence_loop_result(hits: list[Any]) -> dict[str, Any]:
    evidence_by_id: dict[str, dict[str, Any]] = {}
    for hit in hits:
        metadata = dict(hit.metadata or {})
        evidence_by_id[str(hit.id)] = {
            "text": str(hit.text or ""),
            "chunk_type": str(metadata.get("chunk_type") or "case"),
            "evidence_level": str(metadata.get("evidence_level") or "case"),
            "pub_date": str(metadata.get("pub_date") or ""),
            "citation_json": {
                "date": str(metadata.get("pub_date") or ""),
                "pmid": str(metadata.get("pmid") or ""),
                "title": str(metadata.get("title") or ""),
            },
        }
    return {
        "answer_memory": {
            "claims": [],
            "informative_rounds": [],
            "evidence_by_id": evidence_by_id,
        }
    }


def safe_model_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("predictive_clinical_benchmark/benchmark_multinode.json"),
    )
    parser.add_argument("--index-root", type=Path, default=Path("indexes"))
    parser.add_argument(
        "--embed-model-path",
        type=Path,
        default=Path("models/bge-large-zh-v1.5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/cutoff_generation_gpt4o_vs_gpt5_5.json"),
    )
    parser.add_argument("--case-ids", nargs="+", default=list(DEFAULT_CASE_IDS))
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "gpt-5"])
    parser.add_argument("--judge-model", default="gpt-5")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    all_instances = load_json(args.data)
    instance_by_id = {str(row["instance_id"]): row for row in all_instances}
    instances = [instance_by_id[case_id] for case_id in args.case_ids]

    tools = RetrievalTools(
        index_root=args.index_root,
        embed_model_path=args.embed_model_path,
        llm_client=None,
    )
    evidence_by_case: dict[str, dict[str, Any]] = {}
    retrieval_audit: dict[str, Any] = {}
    try:
        for instance in instances:
            case_id = str(instance["instance_id"])
            cutoff = str(instance["time_cutoff"])
            query, constraints = retrieval_inputs(instance)
            tools.configure_temporal_filter(mode="cutoff", cutoff=cutoff)
            cutoff_safe_pool = tools.hybrid_search(
                query=query,
                constraints=constraints,
                top_k=max(100, args.top_k),
            )
            hits = cutoff_safe_pool[: max(1, args.top_k)]
            if not hits:
                raise RuntimeError(f"{case_id}: no cutoff-safe retrieval hits")
            evidence_by_case[case_id] = evidence_loop_result(hits)
            retrieval_audit[case_id] = {
                "cutoff": cutoff,
                "query": query,
                "hit_count": len(hits),
                "hits": [
                    {
                        "chunk_id": hit.id,
                        "score": hit.score,
                        "pub_date": hit.metadata.get("pub_date"),
                        "title": hit.metadata.get("title"),
                        "text": hit.text,
                    }
                    for hit in hits
                ],
            }
            print(f"[{case_id}] retrieved {len(hits)} cutoff-safe hits", flush=True)
    finally:
        tools.close()

    configured_keys = [
        value.strip()
        for value in os.environ.get("OPENAI_API_KEYS", "").split(",")
        if value.strip()
    ]
    api_key = configured_keys[0] if configured_keys else None
    outputs_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    generation_audit: dict[str, Any] = {}
    for model_name in args.models:
        llm = LLMClient(
            api_key=api_key,
            model_name=model_name,
            timeout=float(os.environ.get("LLM_TIMEOUT") or 900),
            max_retries=int(os.environ.get("LLM_MAX_RETRIES") or 10),
            structured_attempts=int(os.environ.get("LLM_STRUCTURED_ATTEMPTS") or 1),
        )
        generator = BenchmarkPredictionGenerator(llm_client=llm, use_llm=True)
        model_outputs: dict[str, dict[str, Any]] = {}
        model_audit: dict[str, Any] = {}
        for instance in instances:
            case_id = str(instance["instance_id"])
            llm.begin_trace(f"cutoff-generation__{safe_model_name(model_name)}__{case_id}")
            output = generator.generate(
                benchmark_prompt=construct_inference_prompt(instance),
                query="",
                loop_result=evidence_by_case[case_id],
                safety_result={},
                answer_context_summary={},
            )
            if generator.last_generation_source != "llm":
                raise RuntimeError(
                    f"{model_name}/{case_id}: {generator.last_error}"
                )
            model_outputs[case_id] = output
            model_audit[case_id] = llm.trace_events()
            print(f"[{model_name}/{case_id}] generated", flush=True)
        outputs_by_model[model_name] = model_outputs
        generation_audit[model_name] = model_audit

    judge_llm = LLMClient(
        api_key=api_key,
        model_name=args.judge_model,
        timeout=float(os.environ.get("LLM_TIMEOUT") or 900),
        max_retries=int(os.environ.get("LLM_MAX_RETRIES") or 10),
        structured_attempts=1,
    )
    judge_llm.begin_trace("cutoff-generation-fixed-judge")

    def judge_fn(prompt: str) -> dict[str, Any]:
        raw = judge_llm.chat(
            system="你是严格的临床预测评测员。只按用户提供的评分规则输出要求的JSON。",
            user=prompt,
            temperature=0.0,
            max_output_tokens=2000,
        )
        parsed = parse_model_output(raw)
        return parsed if isinstance(parsed, dict) else {"score": 0}

    results_by_model: dict[str, Any] = {}
    for model_name in args.models:
        model_outputs = outputs_by_model[model_name]

        def instance_model_fn(_prompt: str, instance: dict[str, Any]) -> str:
            return json.dumps(
                model_outputs[str(instance["instance_id"])],
                ensure_ascii=False,
            )

        results_by_model[model_name] = run_benchmark(
            instances=instances,
            model_fn=lambda _: "",
            instance_model_fn=instance_model_fn,
            llm_judge_fn=judge_fn,
            verbose=False,
            workers=1,
        )
        print(f"[{model_name}] scoring complete", flush=True)

    payload = {
        "experiment": {
            "name": "same_cutoff_rag_evidence_gpt4o_vs_gpt5_generation",
            "case_ids": args.case_ids,
            "models": args.models,
            "judge_model": args.judge_model,
            "top_k": args.top_k,
            "time_filter": "publication_date <= instance.time_cutoff",
            "same_evidence_for_all_models": True,
            "post_cutoff_ground_truth_used_as_input": False,
        },
        "retrieval_audit": retrieval_audit,
        "generation_audit": generation_audit,
        "judge_audit": judge_llm.trace_events(),
        "results": results_by_model,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
