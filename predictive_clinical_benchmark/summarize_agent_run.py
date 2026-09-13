#!/usr/bin/env python3
"""Aggregate per-case SearchAgent telemetry and workflow paths for a benchmark run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def component_for(operation: str) -> str:
    if operation == "function:submit_query_understanding":
        return "QueryUnderstanding"
    if operation == "function:submit_retrieval_plan":
        return "RetrievalPlanner"
    if operation == "function:submit_multistep_plan":
        return "MultiStepPlanner"
    if operation == "function:submit_replan_decision":
        return "Replanner"
    if operation.startswith("tool_loop:submit_step_execution_report"):
        return "ExecutionAgent"
    if operation == "function:submit_reranking":
        return "Rerank"
    if operation == "function:submit_evidence_review":
        return "EvidenceReviewAgent"
    if operation == "function:commit_step_memory":
        return "StepMemoryAgent"
    if operation == "function:maintain_replanner_memory":
        return "ReplannerMemoryAgent"
    if operation == "function:commit_answer_memory":
        return "AnswerMemoryAgent"
    if operation == "function:submit_safety_reflection":
        return "SafetyGate"
    if operation == "function:summarize_answer_context":
        return "AnswerContextAgent"
    if operation == "function:submit_predictive_benchmark_result":
        return "BenchmarkPredictionAgent"
    if operation == "chat":
        return "AnswerGenerator"
    return operation


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    position = (len(ordered) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 4)
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(value, 4)


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "sum": 0, "mean": None, "median": None, "p90": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(values),
        "sum": round(sum(values), 4),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def workflow_metrics(workflow_path: Path) -> dict[str, Any]:
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    steps = {
        str(step.get("name") or ""): step
        for step in workflow.get("steps") or []
        if isinstance(step, dict)
    }
    retrieval = steps.get("MultiStepPlanAndRetrieval") or {}
    rerank = steps.get("Rerank") or {}
    review = steps.get("EvidenceReviewAgent") or {}
    memory = steps.get("StepMemoryAgent") or {}
    safety = steps.get("SafetyGate") or {}
    answer = steps.get("AnswerGenerator") or {}
    persistence = steps.get("SessionPersistence") or {}

    tool_trace = list(retrieval.get("tool_trace") or [])
    tool_calls: dict[str, int] = defaultdict(int)
    fetched_evidence_count = 0
    for item in tool_trace:
        tool_name = str(item.get("tool_name") or "unknown")
        tool_calls[tool_name] += 1
        result = item.get("result") or {}
        fetched_evidence_count += int(result.get("fetched_count") or 0)

    loop_steps_value = retrieval.get("loop_steps")
    loop_steps = list(loop_steps_value) if isinstance(loop_steps_value, list) else []
    loop_step_count = int(loop_steps_value or 0) if not isinstance(loop_steps_value, list) else len(loop_steps)
    step_memories = list(memory.get("step_memories") or [])
    completion_statuses: dict[str, int] = defaultdict(int)
    information_gain_scores: list[float] = []
    accepted_evidence_ids: set[str] = set()
    for item in step_memories:
        goal = item.get("goal_evaluation") or {}
        completion_statuses[str(goal.get("completion_status") or "unknown")] += 1
        gain = item.get("question_information_gain") or {}
        if gain.get("score") is not None:
            information_gain_scores.append(float(gain["score"]))
        for evidence in item.get("accepted_evidence") or []:
            evidence_id = str(evidence.get("chunk_id") or "")
            if evidence_id:
                accepted_evidence_ids.add(evidence_id)

    budget = retrieval.get("budget_state") or {}
    initial_budget = int(budget.get("initial_total_budget") or 0)
    final_budget = int(budget.get("final_total_budget") or 0)
    rounds_used = int(budget.get("rounds_used") or loop_step_count)
    return {
        "query_type": (steps.get("QueryUnderstanding") or {}).get("query_type"),
        "plan_step_count": len((retrieval.get("retrieval_plan") or {}).get("steps") or []),
        "retrieval_rounds": rounds_used,
        "replan_decisions": len(retrieval.get("replan_decisions") or []),
        "stop_condition": retrieval.get("stop_condition"),
        "initial_budget": initial_budget,
        "final_budget": final_budget,
        "budget_used_ratio": round(rounds_used / final_budget, 4) if final_budget else None,
        "budget_extension": final_budget - initial_budget,
        "tool_loop_turns_last_round": retrieval.get("tool_loop_turns"),
        "tool_calls_last_round": dict(tool_calls),
        "tool_calls_last_round_total": sum(tool_calls.values()),
        "tool_hit_counts_last_round": retrieval.get("tool_hit_counts") or {},
        "fetched_evidence_last_round": fetched_evidence_count,
        "new_chunks_total": sum(int(item.get("new_chunk_count") or 0) for item in loop_steps),
        "accepted_unique_evidence": len(accepted_evidence_ids),
        "step_memory_count": len(step_memories),
        "completion_statuses": dict(completion_statuses),
        "information_gain": distribution(information_gain_scores),
        "rerank_enabled_last_round": bool(rerank.get("enabled")),
        "rerank_pool_size_last_round": rerank.get("pool_size"),
        "rerank_top_k_last_round": rerank.get("top_k"),
        "supporting_evidence": int(review.get("supporting_count") or 0),
        "contradicting_evidence": int(review.get("contradicting_count") or 0),
        "safety_risk_evidence": int(review.get("safety_risk_count") or 0),
        "evidence_verdict": review.get("verdict"),
        "answer_claim_count": len(memory.get("answer_claims") or []),
        "safety_risk_level": safety.get("risk_level"),
        "safety_issue_count": int(safety.get("issue_count") or 0),
        "requires_human_review": bool(safety.get("requires_human_review")),
        "has_final_answer": bool(answer.get("has_final_answer")),
        "session_saved": bool(persistence.get("saved_session")),
    }


def summarize_case(case_root: Path) -> dict[str, Any]:
    telemetry_path = case_root / "telemetry.json"
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    calls = list(telemetry.get("llm_calls") or [])
    components: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0,
            "errors": 0,
            "duration_seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "operations": [],
        }
    )
    for call in calls:
        operation = str(call.get("operation") or "unknown")
        row = components[component_for(operation)]
        row["calls"] += 1
        row["errors"] += int(call.get("status") == "error")
        row["duration_seconds"] += float(call.get("duration_seconds") or 0.0)
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            row[key] += int(call.get(key) or 0)
        row["operations"].append(
            {
                "operation": operation,
                "status": call.get("status"),
                "duration_seconds": call.get("duration_seconds"),
                "input_tokens": call.get("input_tokens"),
                "output_tokens": call.get("output_tokens"),
                "total_tokens": call.get("total_tokens"),
                "error_type": call.get("error_type"),
            }
        )
    for row in components.values():
        row["duration_seconds"] = round(row["duration_seconds"], 4)

    stage_timings = dict(telemetry.get("stage_timings") or {})
    total_duration = sum(float(call.get("duration_seconds") or 0.0) for call in calls)
    total_tokens = sum(int(call.get("total_tokens") or 0) for call in calls)
    status_counts: dict[str, int] = defaultdict(int)
    for call in calls:
        status_counts[str(call.get("status") or "unknown")] += 1
    completed_calls = status_counts.get("completed", 0)
    error_calls = status_counts.get("error", 0)
    incomplete_calls = status_counts.get("incomplete", 0)
    for row in components.values():
        row["duration_share"] = round(row["duration_seconds"] / total_duration, 4) if total_duration else None
        row["token_share"] = round(row["total_tokens"] / total_tokens, 4) if total_tokens else None
    workflow_path = case_root / "workflow_trace.json"
    return {
        "case_id": case_root.name,
        "total_llm_calls": len(calls),
        "total_llm_errors": sum(int(call.get("status") == "error") for call in calls),
        "completed_llm_calls": completed_calls,
        "incomplete_llm_calls": incomplete_calls,
        "llm_status_counts": dict(status_counts),
        "llm_completion_rate": round(completed_calls / len(calls), 4) if calls else None,
        "run_quality": "pure_llm" if error_calls == 0 and incomplete_calls == 0 else "degraded_with_fallback_or_incomplete",
        "total_llm_duration_seconds": round(total_duration, 4),
        "mean_llm_call_seconds": round(total_duration / len(calls), 4) if calls else None,
        "total_tokens": total_tokens,
        "mean_tokens_per_call": round(total_tokens / len(calls), 2) if calls else None,
        "tokens_per_llm_second": round(total_tokens / total_duration, 2) if total_duration else None,
        "wall_clock_seconds": stage_timings.get("total"),
        "llm_time_wall_ratio": round(total_duration / float(stage_timings["total"]), 4) if stage_timings.get("total") else None,
        "stage_timings": stage_timings,
        "components": dict(components),
        "workflow_metrics": workflow_metrics(workflow_path),
        "workflow_json": str((case_root / "workflow_trace.json").resolve()),
        "workflow_text": str((case_root / "workflow.txt").resolve()),
        "telemetry": str(telemetry_path.resolve()),
    }


def global_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    components: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "errors": 0, "duration_seconds": 0.0, "total_tokens": 0}
    )
    stop_conditions: dict[str, int] = defaultdict(int)
    safety_levels: dict[str, int] = defaultdict(int)
    evidence_verdicts: dict[str, int] = defaultdict(int)
    tool_calls: dict[str, int] = defaultdict(int)
    for case in cases:
        for name, row in case["components"].items():
            target = components[name]
            for key in ("calls", "errors", "duration_seconds", "total_tokens"):
                target[key] += row[key]
        workflow = case["workflow_metrics"]
        stop_conditions[str(workflow.get("stop_condition") or "unknown")] += 1
        safety_levels[str(workflow.get("safety_risk_level") or "unknown")] += 1
        evidence_verdicts[str(workflow.get("evidence_verdict") or "unknown")] += 1
        for name, count in workflow.get("tool_calls_last_round", {}).items():
            tool_calls[name] += int(count)
    for row in components.values():
        row["duration_seconds"] = round(row["duration_seconds"], 4)
        row["mean_seconds_per_call"] = round(row["duration_seconds"] / row["calls"], 4) if row["calls"] else None
        row["error_rate"] = round(row["errors"] / row["calls"], 4) if row["calls"] else None
        row["mean_tokens_per_call"] = round(row["total_tokens"] / row["calls"], 2) if row["calls"] else None

    total_component_duration = sum(row["duration_seconds"] for row in components.values())
    total_component_tokens = sum(row["total_tokens"] for row in components.values())
    for row in components.values():
        row["duration_share"] = round(row["duration_seconds"] / total_component_duration, 4) if total_component_duration else None
        row["token_share"] = round(row["total_tokens"] / total_component_tokens, 4) if total_component_tokens else None

    calls = [float(case["total_llm_calls"]) for case in cases]
    errors = [float(case["total_llm_errors"]) for case in cases]
    walls = [float(case["wall_clock_seconds"]) for case in cases if case.get("wall_clock_seconds") is not None]
    llm_times = [float(case["total_llm_duration_seconds"]) for case in cases]
    tokens = [float(case["total_tokens"]) for case in cases]
    rounds = [float(case["workflow_metrics"]["retrieval_rounds"]) for case in cases]
    total_calls = int(sum(calls))
    total_errors = int(sum(errors))
    return {
        "completed_cases": len(cases),
        "llm_calls": distribution(calls),
        "llm_errors": distribution(errors),
        "llm_error_free_rate": round((total_calls - total_errors) / total_calls, 4) if total_calls else None,
        "pure_llm_cases": sum(int(case["run_quality"] == "pure_llm") for case in cases),
        "degraded_cases": sum(int(case["run_quality"] != "pure_llm") for case in cases),
        "wall_clock_seconds": distribution(walls),
        "llm_duration_seconds": distribution(llm_times),
        "tokens": distribution(tokens),
        "retrieval_rounds": distribution(rounds),
        "components": dict(components),
        "tool_calls_last_round": dict(tool_calls),
        "stop_conditions": dict(stop_conditions),
        "safety_risk_levels": dict(safety_levels),
        "evidence_verdicts": dict(evidence_verdicts),
    }


def to_markdown(report: dict[str, Any]) -> str:
    global_ = report["global"]
    lines = [
        "# SearchAgent benchmark call report",
        "",
        f"Completed cases: {report['completed_cases']}",
        f"Total LLM calls: {int(global_['llm_calls']['sum'])}",
        f"LLM error-free call rate: {global_['llm_error_free_rate']}",
        f"Pure-LLM cases: {global_['pure_llm_cases']}",
        f"Degraded cases: {global_['degraded_cases']}",
        f"Mean wall clock: {global_['wall_clock_seconds']['mean']} s",
        f"Median wall clock: {global_['wall_clock_seconds']['median']} s",
        f"P95 wall clock: {global_['wall_clock_seconds']['p95']} s",
        f"Total tokens: {int(global_['tokens']['sum'])}",
        "",
    ]
    for case in report["cases"]:
        lines.extend(
            [
                f"## {case['case_id']}",
                "",
                f"- Calls: {case['total_llm_calls']}",
                f"- Errors: {case['total_llm_errors']}",
                f"- Wall clock: {case['wall_clock_seconds']} s",
                f"- LLM duration sum: {case['total_llm_duration_seconds']} s",
                f"- Tokens: {case['total_tokens']}",
                f"- LLM completion rate: {case['llm_completion_rate']}",
                f"- Run quality: {case['run_quality']}",
                f"- Retrieval rounds: {case['workflow_metrics']['retrieval_rounds']}",
                f"- Stop condition: {case['workflow_metrics']['stop_condition']}",
                f"- New chunks: {case['workflow_metrics']['new_chunks_total']}",
                f"- Accepted evidence: {case['workflow_metrics']['accepted_unique_evidence']}",
                f"- Evidence verdict: {case['workflow_metrics']['evidence_verdict']}",
                f"- Safety level: {case['workflow_metrics']['safety_risk_level']}",
                f"- Workflow: `{case['workflow_text']}`",
                "",
                "| Component | Calls | Errors | Seconds | Tokens |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, component in case["components"].items():
            lines.append(
                f"| {name} | {component['calls']} | {component['errors']} | "
                f"{component['duration_seconds']} | {component['total_tokens']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    case_roots = sorted(
        path.parent
        for path in args.artifact_root.glob("*/telemetry.json")
        if (path.parent / "workflow_trace.json").exists()
    )
    cases = [summarize_case(case_root) for case_root in case_roots]
    report = {
        "artifact_root": str(args.artifact_root.resolve()),
        "completed_cases": len(cases),
        "global": global_summary(cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(to_markdown(report), encoding="utf-8")
    print(f"completed_cases={len(cases)} output={args.output}")


if __name__ == "__main__":
    main()
