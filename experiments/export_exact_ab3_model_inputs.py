#!/usr/bin/env python3
"""Export exact benchmark prompts and final Agent Generate inputs for PPT audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predictive_clinical_benchmark.eval.prompts import construct_inference_prompt, construct_agent_query
from searchagent_retrieval.benchmark_prediction import BenchmarkPredictionGenerator

CASE_IDS = ("case_1_node_2", "case_3_node_2", "case_5_node_1")
RESULT = ROOT / "benchmark_results/ab3_context_generate_bounded_20260805.json"
OUT_JSON = ROOT / "benchmark_results/ab3_exact_model_inputs.json"
OUT_MD = ROOT / "benchmark_results/ab3_exact_model_inputs.md"


def main() -> None:
    instances = json.loads((ROOT / "predictive_clinical_benchmark/benchmark_multinode.json").read_text(encoding="utf-8"))
    by_id = {str(row["instance_id"]): row for row in instances}
    audit = json.loads(RESULT.read_text(encoding="utf-8"))["audit"]
    marker = "\n请按以下 JSON 格式输出预测结果"
    exported = {}
    md = ["# 三道病例的逐字模型输入", "", "说明：以下内容不做摘要或改写。"]
    for case_id in CASE_IDS:
        full_prompt = construct_inference_prompt(by_id[case_id])
        prefix, suffix = full_prompt.split(marker, 1)
        benchmark_suffix = "请按以下 JSON 格式输出预测结果" + suffix
        final_payload = audit[case_id]["generate_payload"]
        exported[case_id] = {
            "benchmark_prompt_full": full_prompt,
            "benchmark_case_specific_prefix": prefix,
            "benchmark_common_output_suffix": benchmark_suffix,
            "agent_retrieval_query": construct_agent_query(by_id[case_id]),
            "agent_generate_system_prompt": BenchmarkPredictionGenerator.SYSTEM_PROMPT,
            "agent_generate_user_payload": final_payload,
            "agent_generate_user_json_exact": json.dumps(final_payload, ensure_ascii=False, separators=(",", ":")),
            "agent_generate_function_name": "submit_predictive_benchmark_result",
        }
        md.extend([
            "", f"## {case_id}",
            "", "### A. 完整 benchmark prompt", "", "```text", full_prompt, "```",
            "", "### B. Agent 检索阶段 query", "", "```text", construct_agent_query(by_id[case_id]), "```",
            "", "### C. Agent 最终 Generate system prompt", "", "```text", BenchmarkPredictionGenerator.SYSTEM_PROMPT, "```",
            "", "### D. Agent 最终 Generate user payload（逐字 JSON）", "", "```json",
            json.dumps(final_payload, ensure_ascii=False, indent=2), "```",
            "", "### E. 强制 function", "", "```text", "submit_predictive_benchmark_result", "```",
        ])
    OUT_JSON.write_text(json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
