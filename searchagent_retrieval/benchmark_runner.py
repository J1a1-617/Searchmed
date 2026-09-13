from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .agent_loop import AnswerMemoryAgent, EvidenceReviewAgent, MainAgentLoop, ReplannerMemoryAgent
from .answer_context import AnswerContextAgent
from .benchmark_prediction import BenchmarkPredictionGenerator
from .execution_agent import RetrievalExecutionAgent
from .llm_client import LLMClient, LLMClientError
from .router import RetrievalRouter
from .safety_gate import ClinicalSafetyGate
from .session_memory import SessionMemoryStore
from .tools import RetrievalTools
from .trajectory import TrajectoryEventStore
from .workflow_trace import WorkflowTrace
from .skill_runtime import SkillCatalog


class PredictiveBenchmarkRuntime:
    """Long-lived, single-worker runtime.

    A runtime owns its encoder/index handles and must only be used by the worker
    thread that created it. Agent state is still rebuilt for every benchmark case.
    """

    def __init__(
        self,
        *,
        index_root: Path,
        embed_model_path: Optional[Path] = None,
        llm_timeout: Optional[float] = None,
        llm_max_retries: Optional[int] = None,
        llm_structured_attempts: Optional[int] = None,
        llm_max_calls_per_case: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        account_slot: int = 0,
        reranker_backend: str = "qwen",
        reranker_model: Optional[str] = None,
        reranker_device: Optional[str] = None,
        external_knowledge_enabled: bool = True,
    ) -> None:
        self.llm = LLMClient(
            api_key=api_key,
            base_url=base_url,
            timeout=llm_timeout,
            max_retries=llm_max_retries,
            structured_attempts=llm_structured_attempts,
            max_calls_per_trace=llm_max_calls_per_case,
            stage_call_budgets={
                "planning": 2,
                "retrieval": 28,
                "finalization": 7,
            },
        )
        self.account_slot = max(0, int(account_slot))
        self.external_knowledge_enabled = bool(external_knowledge_enabled)
        resolved_reranker_model = reranker_model or os.environ.get("RERANKER_MODEL") or None
        resolved_reranker_device = reranker_device or os.environ.get("RERANKER_DEVICE") or None
        self.tools = RetrievalTools(
            index_root=index_root,
            embed_model_path=embed_model_path,
            llm_client=self.llm,
            reranker_backend=reranker_backend,
            reranker_model=resolved_reranker_model,
            reranker_device=resolved_reranker_device,
            external_knowledge_enabled=self.external_knowledge_enabled,
        )

    def close(self) -> None:
        self.tools.close()


def run_predictive_benchmark_case(
    *,
    benchmark_prompt: str,
    agent_query: Optional[str] = None,
    index_root: Path,
    embed_model_path: Optional[Path] = None,
    top_k: int = 10,
    max_steps: int = 4,
    max_total_steps: int = 4,
    max_budget_extension: int = 0,
    session_id: Optional[str] = None,
    session_root: Path = Path("sessions"),
    runtime: Optional[PredictiveBenchmarkRuntime] = None,
    temporal_filter_mode: str = "off",
    time_cutoff: Optional[str] = None,
    external_knowledge_enabled: bool = True,
) -> Dict[str, Any]:
    # Official benchmark runs must not silently convert failed LLM stages into
    # rule/template fallbacks. Retries happen inside LLMClient; exhaustion is a
    # failed case and is recorded as such.
    os.environ.setdefault("STRICT_LLM_PIPELINE", "1")
    # Benchmark artifacts must remain replayable for later agentic-training
    # work: retain the exact prompts and model/tool outputs in telemetry.
    os.environ.setdefault("LLM_TRACE_FULL", "1")
    owns_runtime = runtime is None
    runtime = runtime or PredictiveBenchmarkRuntime(
        index_root=index_root,
        embed_model_path=embed_model_path,
        external_knowledge_enabled=external_knowledge_enabled,
    )
    llm = runtime.llm
    trace_id = session_id or "benchmark-case"
    case_started = time.perf_counter()
    stage_timings: Dict[str, float] = {}
    session_store = SessionMemoryStore(root=session_root)
    if session_root.name == "session":
        trajectory_path = session_root.parent / "trajectory.jsonl"
    else:
        safe_trace_id = SessionMemoryStore.sanitize_session_id(trace_id)
        trajectory_path = session_root / f"{safe_trace_id}.trajectory.jsonl"
    trajectory = TrajectoryEventStore(
        trajectory_path,
        context={"session_id": trace_id, "trajectory_kind": "predictive_benchmark"},
    )
    llm.begin_trace(trace_id, event_sink=trajectory.append)
    trajectory.append("session/start", {
        "benchmark_prompt": benchmark_prompt,
        "agent_query": agent_query or benchmark_prompt,
        "time_cutoff": time_cutoff,
        "temporal_filter_mode": temporal_filter_mode,
        "top_k": top_k,
        "max_steps": max_steps,
        "max_total_steps": max_total_steps,
        "max_budget_extension": max_budget_extension,
        "external_knowledge_enabled": runtime.external_knowledge_enabled,
    })
    trajectory.append("query/received", {
        "benchmark_prompt": benchmark_prompt,
        "retrieval_query": agent_query or benchmark_prompt,
    })
    terminal_status = "error"
    terminal_error: Optional[Dict[str, str]] = None
    prior_context: Dict[str, Any] = {}
    session_loaded = False
    if session_id:
        prior_session = session_store.load(session_id)
        if prior_session:
            session_loaded = True
            prior_context = session_store.get_planner_context(session_id)

    tools = runtime.tools
    tools.configure_temporal_filter(mode=temporal_filter_mode, cutoff=time_cutoff)
    try:
        router = RetrievalRouter(retrieval_tools=tools, llm_client=llm, use_llm=True)
        loop = MainAgentLoop(
            router=router,
            evidence_review_agent=EvidenceReviewAgent(llm_client=llm, use_llm=True),
            # EvidenceReview owns per-evidence medical interpretation;
            # StepMemory and AnswerMemory remain deterministic bookkeeping so
            # retrieval cannot spend extra calls reinterpreting evidence.
            answer_memory_agent=AnswerMemoryAgent(llm_client=None, use_llm=False),
            replanner_memory_agent=ReplannerMemoryAgent(llm_client=None, use_llm=False),
            execution_agent=RetrievalExecutionAgent(
                router=router,
                llm_client=llm,
                max_turns=5,
                max_hits_per_tool_result=max(12, top_k),
                max_search_calls=2,
                max_fetch_calls=1,
                external_knowledge_enabled=runtime.external_knowledge_enabled,
            ),
            planning_agent=MultiStepPlanningAgent(
                llm_client=llm,
                use_llm=True,
                external_knowledge_enabled=runtime.external_knowledge_enabled,
            ),
            max_steps=max_steps,
            max_total_steps=max_total_steps,
            max_budget_extension=max_budget_extension,
            max_attempts_per_plan_step=3,
            final_llm_call_reserve=3,
            max_llm_calls_per_round=6,
            top_k=top_k,
            external_knowledge_enabled=runtime.external_knowledge_enabled,
        )
        started = time.perf_counter()
        retrieval_query = agent_query or benchmark_prompt
        loop_result = loop.run(query=retrieval_query, initial_context=prior_context)
        stage_timings["agent_loop"] = round(time.perf_counter() - started, 4)
        loop_result["original_query"] = benchmark_prompt
        loop_result["agent_query"] = retrieval_query
        trajectory.append("agent/loop_completed", {"loop_result": loop_result})
        completed_route = loop_result.get("last_route_result") or {}
        trajectory.append("retrieval/result", {
            "selected_tools": completed_route.get("selected_tools") or [],
            "tool_trace": completed_route.get("tool_trace") or [],
            "results": completed_route.get("results") or {},
            "planned_query": completed_route.get("planned_query"),
            "rerank_goal": completed_route.get("rerank_goal"),
        })
        trajectory.append("rerank/result", {
            "backend": completed_route.get("rerank_backend"),
            "goal": completed_route.get("rerank_goal"),
            "candidates": (completed_route.get("results") or {}).get("rerank_candidates") or [],
        })
        trajectory.append("evidence/review", {
            "evidence_layering": completed_route.get("evidence_layering") or {},
            "loop_steps": loop_result.get("loop_steps") or [],
        })
        trajectory.append("memory/update", {
            "step_memories": loop_result.get("step_memories") or [],
            "round_memories": loop_result.get("round_memories") or [],
            "answer_memory": loop_result.get("answer_memory") or {},
            "replanner_short_memory": loop_result.get("replanner_short_memory") or {},
            "replanner_long_memory": loop_result.get("replanner_long_memory") or {},
        })
        skill_catalog = SkillCatalog(embed_model_path=embed_model_path)
        if prior_context:
            loop_result["prior_session_context"] = prior_context

        latest_constraints = loop_result.get("state", {}).get("confirmed_constraints", {})
        safety_gate = ClinicalSafetyGate()
        latest_route = loop_result.get("last_route_result", {})
        safety_result: dict = dict(loop_result.get("final_safety_review") or {})
        if isinstance(latest_route, dict):
            safety_result = safety_result or dict(latest_route.get("clinical_safety_gate") or {})
        if not safety_result:
            safety_result = safety_gate.evaluate(
                query=retrieval_query,
                query_type=loop_result.get("query_type", "similar_case"),
                constraints=latest_constraints,
                retrieval_results=latest_route.get("results", {}) if isinstance(latest_route, dict) else {},
            )
        trajectory.append("safety/review", {"safety_result": safety_result})
        started = time.perf_counter()
        answer_context_summary = AnswerContextAgent(llm_client=llm, use_llm=True).summarize(
            query=retrieval_query,
            loop_result=loop_result,
            safety_result=safety_result,
        )
        operational_activations = list(loop_result.get("structured_output_recoveries") or [])
        if isinstance(answer_context_summary.get("structured_output_recovery"), dict):
            operational_activations.append({
                "stage": "answer_context",
                **answer_context_summary["structured_output_recovery"],
            })
        loop_result["structured_output_recoveries"] = operational_activations
        dynamic_skills_enabled = str(
            os.environ.get("DYNAMIC_SKILLS_ENABLED", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}
        lineage_signals, lineage_gaps = skill_catalog.lineage_anomaly_signals(
            answer_memory=loop_result.get("answer_memory") or {},
            answer_context=answer_context_summary,
        )
        if dynamic_skills_enabled:
            loop_result["skill_runtime"] = skill_catalog.mount_for_signals(
                stage="generate",
                goal="preserve_clinical_evidence_lineage",
                signals=lineage_signals,
                gaps=lineage_gaps,
            )
        else:
            loop_result["skill_runtime"] = {
                "stage": "generate",
                "query": "",
                "candidate_skills": [],
                "selected_skill_ids": [],
                "loaded_skill_ids": [],
                "validation": "disabled_for_ablation",
                "signals": lineage_signals,
                "gaps": lineage_gaps,
            }
        loop_result["skill_runtime"]["dynamic_skills_enabled"] = dynamic_skills_enabled
        loop_result["skill_runtime"]["operational_activations"] = operational_activations
        loop_result["answer_context_summary"] = answer_context_summary
        trajectory.append("answer/context", {"answer_context_summary": answer_context_summary})
        stage_timings["answer_context"] = round(time.perf_counter() - started, 4)
        started = time.perf_counter()
        prediction_generator = BenchmarkPredictionGenerator(llm_client=llm, use_llm=True)
        benchmark_output = prediction_generator.generate(
            benchmark_prompt=benchmark_prompt,
            query=retrieval_query,
            loop_result=loop_result,
            safety_result=safety_result,
            answer_context_summary=answer_context_summary,
        )
        stage_timings["benchmark_prediction"] = round(time.perf_counter() - started, 4)
        trajectory.append("answer/final", {
            "benchmark_output": benchmark_output,
            "generation_source": prediction_generator.last_generation_source,
            "generation_error": prediction_generator.last_error,
        })
        stage_timings["total"] = round(time.perf_counter() - case_started, 4)
        llm_calls = llm.trace_events()
        failed_stages = list(dict.fromkeys(
            str(event.get("operation") or "unknown")
            for event in llm_calls
            if event.get("status") == "error"
        ))
        fallback_stages = list(failed_stages)
        final_prediction_source = prediction_generator.last_generation_source
        if final_prediction_source == "fallback":
            fallback_stages.append("function:submit_predictive_benchmark_result")
            fallback_stages = list(dict.fromkeys(fallback_stages))
            run_status = "failed"
        elif fallback_stages:
            run_status = "degraded"
        else:
            run_status = "complete"
        llm_summary: Dict[str, Dict[str, Any]] = {}
        for event in llm_calls:
            operation = str(event.get("operation") or "unknown")
            bucket = llm_summary.setdefault(operation, {"calls": 0, "errors": 0, "duration_seconds": 0.0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            bucket["calls"] += 1
            bucket["errors"] += int(event.get("status") == "error")
            bucket["duration_seconds"] = round(float(bucket["duration_seconds"]) + float(event.get("duration_seconds") or 0), 4)
            for token_key in ("input_tokens", "output_tokens", "total_tokens"):
                bucket[token_key] += int(event.get(token_key) or 0)
        trace = WorkflowTrace().build(
            query=benchmark_prompt,
            loop_result=loop_result,
            safety_result=safety_result,
            final_answer=json.dumps(benchmark_output, ensure_ascii=False),
            session_id=session_id,
            session_loaded=session_loaded,
            session_saved=bool(session_id),
        )
        output = {
            **loop_result,
            "answer_context_summary": answer_context_summary,
            "safety_gate": safety_result,
            "benchmark_output": benchmark_output,
            "run_status": run_status,
            "final_prediction_source": final_prediction_source,
            "failed_stages": failed_stages,
            "fallback_stages": fallback_stages,
            "final_prediction_error": prediction_generator.last_error,
            "final_answer": json.dumps(benchmark_output, ensure_ascii=False),
            "workflow_trace": trace.to_dict(),
            "trajectory": {
                "path": str(trajectory_path),
                "schema_version": TrajectoryEventStore.SCHEMA_VERSION,
            },
            "telemetry": {
                "account_slot": runtime.account_slot,
                "run_status": run_status,
                "final_prediction_source": final_prediction_source,
                "failed_stages": failed_stages,
                "fallback_stages": fallback_stages,
                "temporal_filter": {
                    "mode": tools.temporal_filter_mode,
                    "cutoff": tools.temporal_cutoff,
                },
                "external_knowledge_enabled": runtime.external_knowledge_enabled,
                "stage_timings": stage_timings,
                "llm_summary": llm_summary,
                "llm_calls": llm_calls,
                "trajectory_path": str(trajectory_path),
            },
        }
        if session_id:
            snapshot_path = session_store.save(
                session_id=session_id,
                state={
                    "original_query": benchmark_prompt,
                    "confirmed_constraints": loop_result["state"].get("confirmed_constraints") or {},
                    "identified_entities": loop_result["state"].get("identified_entities") or {},
                    "round_memories": loop_result.get("round_memories") or [],
                    "step_memories": loop_result.get("step_memories") or [],
                    "replanner_short_memory": loop_result.get("replanner_short_memory") or {},
                    "replanner_long_memory": loop_result.get("replanner_long_memory") or {},
                    "memory_timeline": loop_result.get("memory_timeline") or {},
                    "pending_replanner_step_memories": loop_result.get("pending_replanner_step_memories") or [],
                    "replanner_memory_events": loop_result.get("replanner_memory_events") or [],
                    "answer_memory": loop_result.get("answer_memory") or {},
                    "answer_context_summary": answer_context_summary,
                    "skill_runtime": loop_result.get("skill_runtime") or {},
                    "retrieval_plan": loop_result.get("retrieval_plan") or {},
                    "replan_decisions": loop_result.get("replan_decisions") or [],
                    "active_plan_step_index": loop_result.get("active_plan_step_index") or 0,
                    "budget_state": loop_result.get("budget_state") or {},
                    "budget_events": loop_result.get("budget_events") or [],
                    "step_budget_status": loop_result.get("step_budget_status") or {},
                    "final_safety_review": safety_result or {},
                    "citations": loop_result.get("citations") or {},
                    "loop_steps": loop_result.get("loop_steps") or [],
                    "final_answer": json.dumps(benchmark_output, ensure_ascii=False),
                    "run_status": run_status,
                    "final_prediction_source": final_prediction_source,
                    "failed_stages": failed_stages,
                    "fallback_stages": fallback_stages,
                    "workflow_trace": trace.to_dict(),
                },
            )
            trajectory.append("session/snapshot", {"path": str(snapshot_path)})
        terminal_status = run_status
        return output
    except BaseException as exc:
        terminal_error = {"type": type(exc).__name__, "message": str(exc)}
        trajectory.append("session/error", terminal_error)
        raise
    finally:
        trajectory.append("session/end", {
            "status": terminal_status,
            "error": terminal_error,
            "duration_seconds": round(time.perf_counter() - case_started, 4),
            "llm_calls": llm.calls_used(),
        })
        llm.set_trace_event_sink(None)
        trajectory.close()
        # A worker runtime is reused by later cases. Never leak one case's
        # cutoff into the next request or into general, non-benchmark search.
        tools.configure_temporal_filter(mode="off")
        if owns_runtime:
            runtime.close()
