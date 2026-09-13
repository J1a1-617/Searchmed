from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _truncate(text: str, max_len: int = 120) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_len:
        return normalized
    return f"{normalized[: max_len - 3]}..."


def _count_tool_hits(results: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for tool_name, payload in results.items():
        if tool_name == "counter_evidence_search" and isinstance(payload, dict):
            hits = payload.get("hits") or []
            counts[tool_name] = len(hits) if isinstance(hits, list) else 0
            continue
        if isinstance(payload, list):
            counts[tool_name] = len(payload)
    return counts


def _summarize_rerank_hits(hybrid_hits: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    summarized: List[Dict[str, Any]] = []
    for hit in hybrid_hits[:limit]:
        if not isinstance(hit, dict):
            continue
        metadata = hit.get("metadata") or {}
        summarized.append(
            {
                "id": hit.get("id"),
                "llm_relevance_score": metadata.get("llm_relevance_score"),
                "llm_evidence_type": metadata.get("llm_evidence_type"),
                "llm_rerank_dimensions": metadata.get("llm_rerank_dimensions") or {},
                "llm_rerank_reason": metadata.get("llm_rerank_reason"),
                "llm_rerank_status": metadata.get("llm_rerank_status"),
                "rerank_backend": metadata.get("rerank_backend"),
                "rerank_model": metadata.get("rerank_model"),
                "rerank_device": metadata.get("rerank_device"),
                "rerank_relevance_score": metadata.get("rerank_relevance_score"),
                "rerank_original_chars": metadata.get("rerank_original_chars"),
                "rerank_input_chars": metadata.get("rerank_input_chars"),
                "rerank_excerpt_strategy": metadata.get("rerank_excerpt_strategy"),
                "hybrid_score": metadata.get("hybrid_score", hit.get("score")),
                "text_preview": _truncate(str(hit.get("text") or "")),
            }
        )
    return summarized


def _count_evidence_signals(assessed: List[Dict[str, Any]]) -> Dict[str, int]:
    support = 0
    contradict = 0
    safety = 0
    for item in assessed:
        signals = item.get("relevance_signals") or {}
        if signals.get("is_supporting"):
            support += 1
        if signals.get("is_contradicting"):
            contradict += 1
        if signals.get("is_safety_risk"):
            safety += 1
    return {
        "supporting": support,
        "contradicting": contradict,
        "safety_risks": safety,
        "total_assessed": len(assessed),
    }


class WorkflowTrace:
    """Record and render the end-to-end LLM workflow for a single query."""

    def __init__(self) -> None:
        self.query: str = ""
        self.steps: List[Dict[str, Any]] = []

    def build(
        self,
        query: str,
        loop_result: Dict[str, Any],
        safety_result: Dict[str, Any],
        final_answer: Optional[str] = None,
        session_id: Optional[str] = None,
        session_loaded: bool = False,
        session_saved: bool = False,
    ) -> "WorkflowTrace":
        self.query = query
        self.steps = []

        state = loop_result.get("state") or {}
        last_route = loop_result.get("last_route_result") or {}
        loop_steps = loop_result.get("loop_steps") or []
        retrieval_results = last_route.get("results") or {}
        evidence_layering = last_route.get("evidence_layering") or {}
        assessed = evidence_layering.get("assessed_evidence") or []
        hybrid_hits = retrieval_results.get("hybrid_search") or []
        rerank_enabled = any(
            isinstance(hit, dict) and (
                (hit.get("metadata") or {}).get("llm_relevance_score") is not None
                or (hit.get("metadata") or {}).get("rerank_relevance_score") is not None
            )
            for hit in hybrid_hits
        )

        self.steps.append(
            {
                "step": 1,
                "name": "QueryUnderstanding",
                "query_type": loop_result.get("query_type") or state.get("query_type"),
                "constraints": state.get("confirmed_constraints") or last_route.get("constraints") or {},
                "identified_entities": state.get("identified_entities") or {},
            }
        )

        self.steps.append(
            {
                "step": 2,
                "name": "MultiStepPlanAndRetrieval",
                "retrieval_plan": loop_result.get("retrieval_plan") or {},
                "replan_decisions": loop_result.get("replan_decisions") or [],
                "budget_state": loop_result.get("budget_state") or {},
                "budget_events": loop_result.get("budget_events") or [],
                "step_budget_status": loop_result.get("step_budget_status") or {},
                "execution_mode": (last_route.get("planner_metadata") or {}).get("execution_mode", "deterministic_router"),
                "tool_loop_turns": last_route.get("tool_loop_turns"),
                "tool_trace": last_route.get("tool_trace") or [],
                "selected_tools": last_route.get("selected_tools") or [],
                "tool_hit_counts": _count_tool_hits(retrieval_results),
                "loop_steps": len(loop_steps),
                "stop_condition": state.get("stop_condition"),
            }
        )

        self.steps.append(
            {
                "step": 3,
                "name": "Rerank",
                "enabled": rerank_enabled,
                "atomic_rerank_goal": last_route.get("rerank_goal"),
                "pool_size": min(len(hybrid_hits), 20),
                "top_k": len(hybrid_hits),
                "top_hits": _summarize_rerank_hits(hybrid_hits if isinstance(hybrid_hits, list) else []),
            }
        )

        last_citation_verdict = "insufficient_evidence"
        if loop_steps:
            last_citation_verdict = loop_steps[-1].get("citation_verdict", last_citation_verdict)
        signal_counts = _count_evidence_signals(assessed if isinstance(assessed, list) else [])

        self.steps.append(
            {
                "step": 4,
                "name": "EvidenceReviewAgent",
                "verdict": last_citation_verdict,
                "supporting_count": signal_counts["supporting"],
                "contradicting_count": signal_counts["contradicting"],
                "safety_risk_count": signal_counts["safety_risks"],
                "layer_distribution": evidence_layering.get("layer_distribution") or [],
            }
        )

        self.steps.append(
            {
                "step": 5,
                "name": "StepMemoryAgent",
                "step_memories": loop_result.get("step_memories") or loop_result.get("round_memories") or [],
                "answer_claims": (loop_result.get("answer_memory") or {}).get("claims") or [],
                "replanner_short_memory": loop_result.get("replanner_short_memory") or {},
                "replanner_long_memory": loop_result.get("replanner_long_memory") or {},
                "replanner_memory_events": loop_result.get("replanner_memory_events") or [],
            }
        )

        issues = safety_result.get("issues") or []
        self.steps.append(
            {
                "step": 6,
                "name": "SafetyGate",
                "risk_level": safety_result.get("risk_level"),
                "requires_human_review": safety_result.get("requires_human_review"),
                "issue_count": len(issues),
                "key_issues": [
                    {
                        "code": item.get("code"),
                        "severity": item.get("severity"),
                        "title": item.get("title"),
                    }
                    for item in issues[:5]
                    if isinstance(item, dict)
                ],
                "recommended_actions": safety_result.get("recommended_actions") or [],
            }
        )

        self.steps.append(
            {
                "step": 7,
                "name": "AnswerGenerator",
                "final_answer_preview": _truncate(final_answer or "", max_len=400),
                "has_final_answer": bool(final_answer and str(final_answer).strip()),
                "answer_context_summary": loop_result.get("answer_context_summary") or {},
                "skill_runtime": loop_result.get("skill_runtime") or {},
                "structured_output_recoveries": loop_result.get("structured_output_recoveries") or [],
            }
        )

        self.steps.append(
            {
                "step": 8,
                "name": "SessionPersistence",
                "session_id": session_id,
                "loaded_existing_session": session_loaded,
                "saved_session": session_saved,
            }
        )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "steps": self.steps,
        }

    def to_text(self) -> str:
        lines: List[str] = [
            "=" * 72,
            "SearchAgent LLM 工作流",
            "=" * 72,
            f"问题: {self.query}",
            "",
        ]

        for step in self.steps:
            name = step.get("name", "Unknown")
            index = step.get("step", "?")
            lines.append(f"[步骤 {index}] {name}")
            lines.append("-" * 48)

            if name == "QueryUnderstanding":
                lines.append(f"  问题类型: {step.get('query_type')}")
                constraints = step.get("constraints") or {}
                if constraints:
                    lines.append(f"  约束: {constraints}")
                entities = step.get("identified_entities") or {}
                if entities:
                    lines.append(f"  实体: {entities}")

            elif name == "MultiStepPlanAndRetrieval":
                plan_steps = (step.get("retrieval_plan") or {}).get("steps") or []
                lines.append(f"  原子检索步骤: {len(plan_steps)}")
                for plan_step in plan_steps:
                    status = (step.get("step_budget_status") or {}).get(str(plan_step.get("step_id"))) or {}
                    lines.append(
                        f"    - {plan_step.get('step_id')}: {plan_step.get('goal')} "
                        f"[单步预算={plan_step.get('attempt_budget', 2)}, "
                        f"已用={status.get('attempts_used', 0)}, 状态={status.get('status', 'pending')}]"
                    )
                budget = step.get("budget_state") or {}
                lines.append(
                    "  总预算: "
                    f"初始={budget.get('initial_total_budget')}, "
                    f"最终={budget.get('final_total_budget')}, "
                    f"硬上限={budget.get('hard_total_budget')}, "
                    f"已用={budget.get('rounds_used')}, "
                    f"剩余={budget.get('rounds_remaining')}"
                )
                for event in step.get("budget_events") or []:
                    if event.get("event") == "agent_budget_extension":
                        lines.append(
                            f"    Agent扩容: 请求={event.get('requested')}, "
                            f"批准={event.get('granted')}, 新总预算={event.get('new_total_budget')}, "
                            f"理由={event.get('reason')}"
                        )
                    elif event.get("event") == "step_budget_exhausted":
                        lines.append(
                            f"    单步预算耗尽: {event.get('step_id')} "
                            f"({event.get('attempts_used')}/{event.get('attempt_budget')})"
                        )
                lines.append(f"  执行模式: {step.get('execution_mode')}; Tool Loop轮数: {step.get('tool_loop_turns')}")
                for call in step.get("tool_trace") or []:
                    lines.append(f"    工具调用: {call.get('tool_name')} ({call.get('tool_call_id', '')})")
                lines.append(f"  选用工具: {', '.join(step.get('selected_tools') or [])}")
                counts = step.get("tool_hit_counts") or {}
                if counts:
                    count_text = ", ".join(f"{tool}={count}" for tool, count in counts.items())
                    lines.append(f"  各工具召回: {count_text}")
                lines.append(f"  多跳轮次: {step.get('loop_steps')}")
                lines.append(f"  停止条件: {step.get('stop_condition')}")

            elif name == "Rerank":
                enabled = step.get("enabled")
                lines.append(f"  LLM Rerank: {'已启用' if enabled else '未启用/回退 hybrid 分数'}")
                lines.append(f"  本轮原子评分目标: {step.get('atomic_rerank_goal')}")
                lines.append(f"  候选池: {step.get('pool_size')} -> top {step.get('top_k')}")
                for hit in step.get("top_hits") or []:
                    score = hit.get("llm_relevance_score")
                    score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"
                    lines.append(f"    - {hit.get('id')} | llm={score_text} | {_truncate(hit.get('text_preview', ''), 80)}")

            elif name == "EvidenceReviewAgent":
                lines.append(f"  审核结论: {step.get('verdict')}")
                lines.append(
                    "  证据统计: "
                    f"支持={step.get('supporting_count')} "
                    f"反对={step.get('contradicting_count')} "
                    f"风险={step.get('safety_risk_count')}"
                )
                for layer in step.get("layer_distribution") or []:
                    lines.append(
                        f"    - {layer.get('evidence_level')}: count={layer.get('count')}, priority={layer.get('priority')}"
                    )

            elif name == "StepMemoryAgent":
                for memory in step.get("step_memories") or []:
                    gain = memory.get("question_information_gain") or {}
                    goal = memory.get("goal_evaluation") or {}
                    ledger = memory.get("micro_retrieval_ledger") or []
                    lines.append(
                        f"  StepMemory {memory.get('round')} [{memory.get('plan_step_id')}]: micro_attempts={len(ledger)}, gain={gain.get('score')}, "
                        f"new={len(gain.get('new_chunk_ids') or [])}"
                    )
                    for attempt in ledger:
                        lines.append(f"    Micro {attempt.get('attempt')}: {attempt.get('tool')} hits={attempt.get('hit_count')} query={_truncate(attempt.get('query') or '', 90)}")
                    lines.append(f"    完成度: {goal.get('completion_status')}; 关键缺口: {goal.get('critical_gaps') or []}; 继续价值: {goal.get('marginal_value_of_more_search')}")
                    lines.append(f"    数据库查询状态: {goal.get('query_database_status', 'uncertain')}; 原因: {_truncate(goal.get('exhaustion_reason') or '', 180)}")
                lines.append(f"  累计回答论断: {len(step.get('answer_claims') or [])}")
                events = step.get("replanner_memory_events") or []
                lines.append(f"  Replanner记忆提炼次数: {len(events)}")
                for event in events:
                    lines.append(f"    - round={event.get('round')} 合并StepMemory={event.get('consumed_step_memories')} 提炼后估算tokens={event.get('estimated_tokens_after')}")
                lines.append(f"  Replanner短期记忆: {_truncate(json.dumps(step.get('replanner_short_memory') or {}, ensure_ascii=False), 300)}")
                lines.append(f"  Replanner长期记忆: {_truncate(json.dumps(step.get('replanner_long_memory') or {}, ensure_ascii=False), 500)}")

            elif name == "SafetyGate":
                lines.append(f"  风险等级: {step.get('risk_level')}")
                lines.append(f"  需人工复核: {step.get('requires_human_review')}")
                for issue in step.get("key_issues") or []:
                    lines.append(f"    - [{issue.get('severity')}] {issue.get('title')} ({issue.get('code')})")

            elif name == "AnswerGenerator":
                context = step.get("answer_context_summary") or {}
                lines.append(f"  AnswerContext: findings={len(context.get('key_findings') or [])}, gaps={len(context.get('unresolved_gaps') or [])}, safety={len(context.get('safety_boundaries') or [])}, evidence_ids={len(context.get('evidence_ids') or [])}")
                if step.get("has_final_answer"):
                    lines.append("  最终回答摘要:")
                    lines.append(f"    {step.get('final_answer_preview')}")
                else:
                    lines.append("  最终回答: （未生成）")

            elif name == "SessionPersistence":
                session_id = step.get("session_id")
                if session_id:
                    lines.append(f"  Session ID: {session_id}")
                    lines.append(f"  加载历史: {'是' if step.get('loaded_existing_session') else '否'}")
                    lines.append(f"  保存状态: {'是' if step.get('saved_session') else '否'}")
                else:
                    lines.append("  未使用 session 持久化")

            lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines)
