#!/usr/bin/env python3
"""LLM-driven synthesis of reusable skills from one completed cohort.

This is intentionally separate from activation: synthesis creates candidates;
the A/B gate must pass before registry status changes to active.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from searchagent_retrieval.llm_client import LLMClient


SKILL_SCHEMA = {
    "type": "object",
    "properties": {
        "gold_evidence_resolutions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "instance_id": {"type": "string"},
                    "decision": {"type": "string", "enum": ["explicit_anchor", "db_gold_found", "db_missing_gold", "unknown"]},
                    "accepted_candidate_ids": {"type": "array", "items": {"type": "string"}},
                    "first_loss_stage": {"type": "string", "enum": ["none", "candidate_retrieval", "rerank", "fetch", "evidence_review", "answer_memory", "answer_context", "generate", "unknown"]},
                    "reason": {"type": "string"},
                },
                "required": ["instance_id", "decision", "accepted_candidate_ids", "first_loss_stage", "reason"],
                "additionalProperties": False,
            },
        },
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "name": {"type": "string"},
                    "problem_pattern": {"type": "string"},
                    "evidence_from_cases": {"type": "array", "items": {"type": "string"}},
                    "activation_rules": {"type": "array", "items": {"type": "string"}},
                    "forbidden_behaviors": {"type": "array", "items": {"type": "string"}},
                    "ab_test_cases": {"type": "array", "items": {"type": "string"}},
                    "ab_success_metric": {"type": "string"},
                    "target_call_stages": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["query_understanding", "planner", "execution", "rerank", "evidence_review", "answer_memory", "safety_reflection", "answer_context", "generate"]},
                    },
                    "source_failure_signatures": {"type": "array", "items": {"type": "string"}},
                    "dedup_key": {"type": "string"},
                },
                "required": ["skill_id", "name", "problem_pattern", "evidence_from_cases", "activation_rules", "forbidden_behaviors", "ab_test_cases", "ab_success_metric", "target_call_stages", "source_failure_signatures", "dedup_key"],
                "additionalProperties": False,
            },
        },
        "shared_patterns": {"type": "array", "items": {"type": "string"}},
        "discarded_patterns": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["gold_evidence_resolutions", "skills", "shared_patterns", "discarded_patterns"],
    "additionalProperties": False,
}


SYSTEM = """你是 Skill Evolution Agent。输入是一个已经完成的 benchmark cohort（通常20道题）的轨迹审计、失败案例和当前上下文。
你的任务是只提取跨病例、可复用的 agent 能力缺陷，并生成多个最小化、可测试的 Skill 候选。
不要把单个病例的医学事实写成 Skill；不要把 API 错误、偶发超时或数据库缺失伪装成通用 Skill。
每个 Skill 必须有明确触发条件、禁止行为、可执行规则、来源错题、实际作用位置、A/B 样本和成功指标。
来源错题必须来自 wrong_case_inventory；不能用原本已经答对的回归题替代来源错题有效性测试。
target_call_stages 必须是能力真正产生作用的位置，不是发现错误的位置或方便挂载的位置。
相似规则要合并，互不相关的问题要拆成不同 Skill。必须通过 submit_skill_candidates 提交严格 JSON。"""

SYSTEM += """
问题空间是开放的，绝不能只从预定义错误类型或既有四个 Skill 模板中选择；允许发现新的、跨病例可复用的 Agent 能力缺陷。
但在判断检索故障前必须读取 evidence_availability_audits：
- text_candidates_pending_codex 表示已用黄金证据文字反查 DB；你必须逐条核验候选是否表达同一患者事实，而不是仅仅主题相似；
- 核验通过后，使用候选 chunk/doc/PMID 对照 retrieval_results、rerank、fetch、evidence_review、answer_context 和 generate，明确记录首次丢失阶段；
- 候选均不等价时标记 db_missing_gold；无法确定时保持 unknown，禁止猜测；
- db_missing_gold 是数据覆盖问题，不得生成检索 Skill；
- candidate_retrieval_miss / rerank_drop / fetch_drop / evidence_review_drop / answer_context_drop / generation_citation_drop 才分别归因给对应阶段；笼统 retrieval_miss 只能说明整个检索 artifact 未见 gold，不能擅自细分；
- gold_reached_generation 说明 gold 已到最终生成层，错误应从推理或任务标签口径分析。
不得把 ground-truth 结局标签本身当作 gold evidence。"""

SYSTEM += """
general_skill_learning_eligible 与 retrieval_skill_learning_eligible 必须分别遵守：前者允许从明确错题中归纳推理、记忆、结构化输出等问题；后者为 false 时，禁止生成或更新任何检索召回、rerank、fetch、证据筛选归因的 Skill。"""

SYSTEM += """
对每个来源错题必须输出 gold_evidence_resolutions。若 gold_status=text_candidates_pending_codex，逐条比较黄金文字与候选全文：只有同一患者、同一治疗阶段、同一时间点和同一结局事实才算 db_gold_found；主题或疗法相似不算。若找到等价候选，再用其 chunk_id/doc_id/pmid 对照原始 Session 的 candidate retrieval、rerank、fetch、EvidenceReview、AnswerMemory、AnswerContext、Generate，填写 first_loss_stage。若未找到等价候选，decision=db_missing_gold 且不得生成检索 Skill；证据不足则 decision=unknown。"""


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_source_traces(wrong_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traces = []
    for row in wrong_cases:
        root_value = str(row.get("artifact_path") or "")
        root = Path(root_value) if root_value else None
        artifacts: dict[str, Any] = {}
        if root and root.is_dir():
            for name in ("final_prediction.json", "retrieval_results.json", "evidence_review.json", "answer_context.json", "session_memory.json"):
                path = root / name
                if not path.is_file():
                    continue
                value = load(path)
                encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                artifacts[name] = encoded[:4000]
        traces.append({
            "instance_id": row.get("instance_id"),
            "ground_truth": row.get("ground_truth"),
            "prediction": row.get("prediction"),
            "artifact_path": root_value or None,
            "artifacts": artifacts,
        })
    return traces


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--audit", type=Path, action="append", required=True)
    ap.add_argument("--reports", type=Path, action="append", default=[])
    ap.add_argument("--failure-inventory", type=Path, default=None)
    ap.add_argument("--evidence-audits", type=Path, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cohort = load(args.cohort)
    audits = [load(path) for path in args.audit]
    reports = []
    for root in args.reports:
        for path in sorted(root.rglob("*.md")):
            reports.append({"path": str(path), "text": path.read_text(encoding="utf-8")[:5000]})
    failure_inventory = load(args.failure_inventory) if args.failure_inventory and args.failure_inventory.is_file() else {}
    evidence_audits = load(args.evidence_audits) if args.evidence_audits and args.evidence_audits.is_file() else []
    all_wrong_cases = failure_inventory.get("wrong_cases", []) if isinstance(failure_inventory, dict) else []
    cohort_error_ids = {str(value) for value in cohort.get("source_error_instance_ids") or []}
    wrong_cases = [
        row for row in all_wrong_cases
        if not cohort_error_ids or str(row.get("instance_id")) in cohort_error_ids
    ][:20]
    payload = {
        "cohort_id": cohort.get("cohort_id"),
        "source_batches": cohort.get("source_batches") or [],
        "source_instance_ids": cohort.get("source_instance_ids") or [],
        "audits": audits,
        "case_reports": reports[:30],
        "wrong_case_inventory": wrong_cases,
        "source_error_traces": compact_source_traces(wrong_cases),
        "evidence_availability_audits": evidence_audits,
        "constraints": {
            "min_cross_case_occurrence": 2,
            "max_skills": 8,
            "must_preserve_source_cases": True,
            "must_define_ab_before_activation": True,
            "ab_arm_a_must_reproduce_source_error": True,
            "ab_arm_b_must_load_only_candidate_skill": True,
            "must_validate_declared_call_stage": True,
        },
    }
    try:
        curator_model = args.model or os.environ.get("SKILL_CURATOR_MODEL") or "gpt-6"
        llm = LLMClient(model_name=curator_model, timeout=180, max_retries=1, structured_attempts=2)
        llm.begin_trace(f"skill-synthesis-{cohort.get('cohort_id') or 'unknown'}")
        result = llm.call_function(
            system=SYSTEM,
            user=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            function_name="submit_skill_candidates",
            description="Submit reusable skill candidates extracted from one benchmark cohort.",
            parameters=SKILL_SCHEMA,
            temperature=0.0,
            max_output_tokens=8000,
        )
        record = {
            "status": "generated",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cohort_id": cohort.get("cohort_id"),
            "source_batches": cohort.get("source_batches") or [],
            "source_instance_ids": cohort.get("source_instance_ids") or [],
            "result": result,
            "ab_status": "required_before_activation",
            "trace": llm.trace_events(),
            "curator_model": curator_model,
        }
    except Exception as exc:
        record = {
            "status": "synthesis_failed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cohort_id": cohort.get("cohort_id"),
            "source_batches": cohort.get("source_batches") or [],
            "source_instance_ids": cohort.get("source_instance_ids") or [],
            "error": f"{type(exc).__name__}: {exc}",
            "ab_status": "blocked",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "cohort_id": record.get("cohort_id"), "output": str(args.output)}, ensure_ascii=False))
    return 0 if record["status"] == "generated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
