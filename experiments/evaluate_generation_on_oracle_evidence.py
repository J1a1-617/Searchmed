#!/usr/bin/env python3
"""Oracle-evidence upper-bound test for the current benchmark generator.

This external harness does not modify or run SearchAgent retrieval. It converts
the benchmark follow-up facts into an evidence bundle (without directly passing
the target category/RECIST labels), calls the current BenchmarkPredictionGenerator,
and scores the resulting predictions with the repository's evaluator.

The experiment intentionally leaks post-cutoff outcome evidence. It measures
evidence interpretation and schema generation, not real predictive performance.
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


def evidence_date(text: str) -> str:
    match = re.search(r"\b(20\d{2})[-/.年](\d{1,2})", text)
    if not match:
        return "NA"
    return f"{match.group(1)}-{int(match.group(2)):02d}"


def oracle_evidence(instance: dict[str, Any]) -> list[dict[str, str]]:
    """Create clinical follow-up facts without directly copying target labels."""
    gt = instance.get("ground_truth") or {}
    rows: list[dict[str, str]] = []
    for index, item in enumerate(gt.get("key_evidence_items") or [], start=1):
        text = " ".join(str(item).split())
        if text:
            rows.append({
                "id": f"oracle-followup-{index}",
                "type": "oracle_followup",
                "text": text,
                "pub_date": evidence_date(text),
            })

    pfs = gt.get("pfs_months")
    if pfs is not None:
        rows.append({
            "id": "oracle-pfs",
            "type": "oracle_followup",
            "text": f"随访记录的无进展生存期为{pfs}个月。",
            "pub_date": "NA",
        })

    toxicity = gt.get("toxicity") or {}
    if toxicity.get("has_toxicity_record"):
        grade = int(toxicity.get("max_grade") or 0)
        event = str(toxicity.get("event") or "未注明事件")
        rows.append({
            "id": "oracle-toxicity",
            "type": "oracle_followup",
            "text": f"随访记录到最高{grade}级不良事件：{event}。",
            "pub_date": "NA",
        })

    na_flags = gt.get("na_flags") or {}
    missing_names = {
        "body_lesion_recist": "体部病灶影像学疗效",
        "cns_lm_recist": "颅内或脑膜病灶影像学疗效",
        "csf_trajectory": "脑脊液指标变化",
        "symptom_trajectory": "症状变化",
    }
    missing = [
        display
        for key, display in missing_names.items()
        if bool(na_flags.get(key))
    ]
    if missing:
        rows.append({
            "id": "oracle-not-assessed",
            "type": "oracle_followup",
            "text": "以下维度在随访中未评估：" + "、".join(missing) + "。",
            "pub_date": "NA",
        })
    return rows


def loop_result_from_evidence(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "answer_memory": {
            "claims": [],
            "informative_rounds": [],
            "evidence_by_id": {
                row["id"]: {
                    "text": row["text"],
                    "evidence_level": row["type"],
                    "pub_date": row["pub_date"],
                }
                for row in rows
            },
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
        "--output",
        type=Path,
        default=Path("benchmark_results/oracle_evidence_generation_ablation_5.json"),
    )
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
    llm = LLMClient(
        api_key=api_keys[0] if api_keys else None,
        timeout=float(os.environ.get("LLM_TIMEOUT") or 900),
        max_retries=int(os.environ.get("LLM_MAX_RETRIES") or 10),
        structured_attempts=int(os.environ.get("LLM_STRUCTURED_ATTEMPTS") or 1),
    )
    generator = BenchmarkPredictionGenerator(llm_client=llm, use_llm=True)

    outputs: dict[str, dict[str, Any]] = {}
    audits: dict[str, Any] = {}
    for instance in instances:
        case_id = str(instance["instance_id"])
        evidence = oracle_evidence(instance)
        llm.begin_trace(f"oracle-evidence-generation-ablation__{case_id}")
        output = generator.generate(
            benchmark_prompt=construct_inference_prompt(instance),
            query="",
            loop_result=loop_result_from_evidence(evidence),
            safety_result={},
            answer_context_summary={},
        )
        if generator.last_generation_source != "llm":
            raise RuntimeError(f"{case_id}: {generator.last_error}")
        outputs[case_id] = output
        audits[case_id] = {
            "oracle_evidence": evidence,
            "generation_source": generator.last_generation_source,
            "llm_calls": llm.trace_events(),
        }
        print(f"[{case_id}] generated from {len(evidence)} oracle facts", flush=True)

    def instance_model_fn(_prompt: str, instance: dict[str, Any]) -> str:
        return json.dumps(outputs[str(instance["instance_id"])], ensure_ascii=False)

    results = run_benchmark(
        instances=instances,
        model_fn=lambda _: "",
        instance_model_fn=instance_model_fn,
        verbose=False,
        workers=1,
    )
    payload = {
        "experiment": {
            "name": "current_generator_oracle_evidence_upper_bound",
            "case_ids": args.case_ids,
            "warning": (
                "Post-cutoff ground-truth follow-up facts are intentionally leaked. "
                "This measures evidence interpretation, not prediction or retrieval."
            ),
            "direct_target_labels_provided": False,
            "m7_interpretable": False,
        },
        "audit": audits,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
