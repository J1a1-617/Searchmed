from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from searchagent_retrieval.agent_loop import EvidenceReviewAgent, MainAgentLoop
from searchagent_retrieval.answer_generator import AnswerGenerator
from searchagent_retrieval.answer_context import AnswerContextAgent
from searchagent_retrieval.llm_client import LLMClient, LLMClientError
from searchagent_retrieval.router import RetrievalRouter
from searchagent_retrieval.safety_gate import ClinicalSafetyGate
from searchagent_retrieval.session_memory import SessionMemoryStore
from searchagent_retrieval.tools import RetrievalTools
from searchagent_retrieval.trajectory import TrajectoryEventStore
from searchagent_retrieval.workflow_trace import WorkflowTrace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SearchAgent multi-agent loop with LLM workflow.")
    parser.add_argument("--query", type=str, required=True, help="Doctor question.")
    parser.add_argument("--index-root", type=Path, default=Path("indexes"), help="Index root path.")
    parser.add_argument(
        "--embed-model-path",
        type=Path,
        default=None,
        help="Embedding model directory used to encode dense-search queries.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for each retrieval tool.")
    parser.add_argument(
        "--reranker-backend",
        choices=("llm", "local", "cross_encoder", "qwen", "qwen_then_llm", "qwen_with_llm_fallback", "none"),
        default="qwen",
    )
    parser.add_argument("--reranker-model", type=str, default=None, help="Local CrossEncoder model name or path.")
    parser.add_argument("--reranker-device", choices=("cuda", "mps", "cpu"), default=None, help="Defaults to CUDA, then MPS, then CPU.")
    parser.add_argument("--max-steps", type=int, default=4, help="Fallback initial-budget hint; the Initial Planner chooses the actual budget.")
    parser.add_argument("--max-total-steps", type=int, default=128, help="High safety hard cap after agent-requested budget extensions.")
    parser.add_argument("--max-budget-extension", type=int, default=32, help="High safety cap for one Replanner extension request.")
    parser.add_argument("--session-id", type=str, default=None, help="Multi-turn session ID.")
    parser.add_argument(
        "--session-root",
        type=Path,
        default=Path("sessions"),
        help="Directory for persisted session JSON files.",
    )
    parser.add_argument(
        "--trajectory-path",
        type=Path,
        default=None,
        help="Append-only JSONL trajectory path; defaults beside session snapshots.",
    )
    parser.add_argument(
        "--print-workflow",
        action="store_true",
        help="Print human-readable LLM workflow trace to stdout.",
    )
    parser.add_argument("--print-full-json", action="store_true", help="Also print the complete audit JSON after the readable workflow.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("LLM_TRACE_FULL", "1")

    try:
        llm = LLMClient()
    except LLMClientError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    session_store = SessionMemoryStore(root=args.session_root)
    trace_id = args.session_id or f"agent-{uuid.uuid4()}"
    trajectory_path = args.trajectory_path or (
        args.session_root / f"{SessionMemoryStore.sanitize_session_id(trace_id)}.trajectory.jsonl"
    )
    trajectory = TrajectoryEventStore(
        trajectory_path,
        context={"session_id": trace_id, "trajectory_kind": "interactive_agent"},
    )
    llm.begin_trace(trace_id, event_sink=trajectory.append)
    trajectory.append("session/start", {
        "query": args.query,
        "top_k": args.top_k,
        "reranker_backend": args.reranker_backend,
        "max_steps": args.max_steps,
        "max_total_steps": args.max_total_steps,
    })
    trajectory.append("query/received", {"query": args.query})
    terminal_status = "error"
    terminal_error = None
    session_loaded = False
    prior_context: dict = {}
    if args.session_id:
        prior_session = session_store.load(args.session_id)
        if prior_session:
            session_loaded = True
            prior_context = session_store.get_planner_context(args.session_id)

    tools = None
    try:
        tools = RetrievalTools(
            index_root=args.index_root,
            embed_model_path=args.embed_model_path,
            llm_client=llm,
            reranker_backend=args.reranker_backend,
            reranker_model=args.reranker_model,
            reranker_device=args.reranker_device,
        )
        router = RetrievalRouter(retrieval_tools=tools, llm_client=llm, use_llm=True)
        loop = MainAgentLoop(
            router=router,
            evidence_review_agent=EvidenceReviewAgent(llm_client=llm, use_llm=True),
            max_steps=args.max_steps,
            max_total_steps=args.max_total_steps,
            max_budget_extension=args.max_budget_extension,
            top_k=args.top_k,
        )
        loop_result = loop.run(query=args.query, initial_context=prior_context)
        loop_result["original_query"] = args.query
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
        if prior_context:
            loop_result["prior_session_context"] = prior_context

        latest_constraints = loop_result["state"].get("confirmed_constraints", {})
        safety_gate = ClinicalSafetyGate()
        latest_route = loop_result.get("last_route_result", {})
        safety_result: dict = dict(loop_result.get("final_safety_review") or {})
        if isinstance(latest_route, dict):
            safety_result = safety_result or dict(latest_route.get("clinical_safety_gate") or {})
        if not safety_result:
            safety_result = safety_gate.evaluate(
                query=args.query,
                query_type=loop_result.get("query_type", "similar_case"),
                constraints=latest_constraints,
                retrieval_results=latest_route.get("results", {}) if isinstance(latest_route, dict) else {},
            )
        trajectory.append("safety/review", {"safety_result": safety_result})

        loop_result["answer_context_summary"] = AnswerContextAgent(llm_client=llm, use_llm=True).summarize(
            query=args.query, loop_result=loop_result, safety_result=safety_result
        )
        trajectory.append("answer/context", {
            "answer_context_summary": loop_result["answer_context_summary"],
        })
        final_answer = AnswerGenerator(llm_client=llm, use_llm=True).generate(
            query=args.query,
            loop_result=loop_result,
            safety_result=safety_result,
        )
        trajectory.append("answer/final", {"final_answer": final_answer})

        session_saved = bool(args.session_id)

        trace = WorkflowTrace().build(
            query=args.query,
            loop_result=loop_result,
            safety_result=safety_result,
            final_answer=final_answer,
            session_id=args.session_id,
            session_loaded=session_loaded,
            session_saved=session_saved,
        )

        if args.session_id:
            snapshot_path = session_store.save(session_id=args.session_id, state={
                "original_query": args.query,
                "confirmed_constraints": loop_result["state"].get("confirmed_constraints") or {},
                "identified_entities": loop_result["state"].get("identified_entities") or {},
                "round_memories": loop_result.get("round_memories") or [],
                "step_memories": loop_result.get("step_memories") or [],
                "replanner_short_memory": loop_result.get("replanner_short_memory") or {},
                "replanner_long_memory": loop_result.get("replanner_long_memory") or {},
                "pending_replanner_step_memories": loop_result.get("pending_replanner_step_memories") or [],
                "replanner_memory_events": loop_result.get("replanner_memory_events") or [],
                "memory_timeline": loop_result.get("memory_timeline") or {},
                "answer_memory": loop_result.get("answer_memory") or {},
                "answer_context_summary": loop_result.get("answer_context_summary") or {},
                "retrieval_plan": loop_result.get("retrieval_plan") or {},
                "replan_decisions": loop_result.get("replan_decisions") or [],
                "active_plan_step_index": loop_result.get("active_plan_step_index") or 0,
                "budget_state": loop_result.get("budget_state") or {},
                "budget_events": loop_result.get("budget_events") or [],
                "step_budget_status": loop_result.get("step_budget_status") or {},
                "final_safety_review": loop_result.get("final_safety_review") or {},
                "citations": loop_result.get("citations") or {},
                "loop_steps": loop_result.get("loop_steps") or [],
                "final_answer": final_answer,
                "workflow_trace": trace.to_dict(),
            })
            trajectory.append("session/snapshot", {"path": str(snapshot_path)})

        if args.print_workflow:
            print(trace.to_text())
            print()

        output = {
            **loop_result,
            "safety_gate": safety_result,
            "final_answer": final_answer,
            "workflow_trace": trace.to_dict(),
            "trajectory": {"path": str(trajectory_path), "schema_version": 1},
        }
        if not args.print_workflow or args.print_full_json:
            print(json.dumps(output, ensure_ascii=False, indent=2))
        terminal_status = "complete"
    except BaseException as exc:
        terminal_error = {"type": type(exc).__name__, "message": str(exc)}
        trajectory.append("session/error", terminal_error)
        raise
    finally:
        trajectory.append("session/end", {
            "status": terminal_status,
            "error": terminal_error,
            "llm_calls": llm.calls_used(),
        })
        llm.set_trace_event_sink(None)
        trajectory.close()
        if tools is not None:
            tools.close()


if __name__ == "__main__":
    main()
