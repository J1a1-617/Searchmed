from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class SessionSnapshot:
    session_id: str
    confirmed_constraints: Dict[str, Any] = field(default_factory=dict)
    identified_entities: Dict[str, List[str]] = field(default_factory=dict)
    round_memories: List[Dict[str, Any]] = field(default_factory=list)
    step_memories: List[Dict[str, Any]] = field(default_factory=list)
    replanner_short_memory: Dict[str, Any] = field(default_factory=dict)
    replanner_long_memory: Dict[str, Any] = field(default_factory=dict)
    pending_replanner_step_memories: List[Dict[str, Any]] = field(default_factory=list)
    replanner_memory_events: List[Dict[str, Any]] = field(default_factory=list)
    memory_timeline: Dict[str, Any] = field(default_factory=dict)
    answer_memory: Dict[str, Any] = field(default_factory=dict)
    answer_context_summary: Dict[str, Any] = field(default_factory=dict)
    skill_runtime: Dict[str, Any] = field(default_factory=dict)
    final_safety_review: Dict[str, Any] = field(default_factory=dict)
    citations: Dict[str, Any] = field(default_factory=dict)
    loop_steps: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    workflow_trace: Dict[str, Any] = field(default_factory=dict)
    budget_state: Dict[str, Any] = field(default_factory=dict)
    budget_events: List[Dict[str, Any]] = field(default_factory=list)
    step_budget_status: Dict[str, Any] = field(default_factory=dict)
    retrieval_plan: Dict[str, Any] = field(default_factory=dict)
    replan_decisions: List[Dict[str, Any]] = field(default_factory=list)
    active_plan_step_index: int = 0
    original_query: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], session_id: str) -> "SessionSnapshot":
        # Read-only migration for old session files: retain useful evidence and
        # legacy summaries as context without persisting the obsolete schema.
        answer_memory = dict(payload.get("answer_memory") or {})
        if not answer_memory and payload.get("retrieved_evidence"):
            evidence = {
                str(item.get("chunk_id")): dict(item)
                for item in payload.get("retrieved_evidence") or []
                if isinstance(item, dict) and item.get("chunk_id")
            }
            answer_memory = {"claims": [], "evidence_by_id": evidence, "informative_rounds": []}
        legacy_context = [str(x) for x in (payload.get("long_memory") or []) if str(x).strip()]
        if legacy_context:
            answer_memory["legacy_context"] = legacy_context
        return cls(
            session_id=session_id,
            confirmed_constraints=dict(payload.get("confirmed_constraints") or {}),
            identified_entities={
                key: list(value) if isinstance(value, list) else [str(value)]
                for key, value in (payload.get("identified_entities") or {}).items()
            },
            round_memories=[dict(x) for x in (payload.get("round_memories") or []) if isinstance(x, dict)],
            step_memories=[dict(x) for x in (payload.get("step_memories") or payload.get("round_memories") or []) if isinstance(x, dict)],
            replanner_short_memory=dict(payload.get("replanner_short_memory") or {}),
            replanner_long_memory=dict(payload.get("replanner_long_memory") or {}),
            pending_replanner_step_memories=[dict(x) for x in (payload.get("pending_replanner_step_memories") or []) if isinstance(x, dict)],
            replanner_memory_events=[dict(x) for x in (payload.get("replanner_memory_events") or []) if isinstance(x, dict)],
            memory_timeline=dict(payload.get("memory_timeline") or {}),
            answer_memory=answer_memory,
            answer_context_summary=dict(payload.get("answer_context_summary") or {}),
            skill_runtime=dict(payload.get("skill_runtime") or {}),
            final_safety_review=dict(payload.get("final_safety_review") or {}),
            citations=dict(payload.get("citations") or {}),
            loop_steps=[dict(x) for x in (payload.get("loop_steps") or []) if isinstance(x, dict)],
            final_answer=str(payload.get("final_answer") or ""),
            workflow_trace=dict(payload.get("workflow_trace") or {}),
            budget_state=dict(payload.get("budget_state") or {}),
            budget_events=[dict(x) for x in (payload.get("budget_events") or []) if isinstance(x, dict)],
            step_budget_status=dict(payload.get("step_budget_status") or {}),
            retrieval_plan=dict(payload.get("retrieval_plan") or {}),
            replan_decisions=[dict(x) for x in (payload.get("replan_decisions") or []) if isinstance(x, dict)],
            active_plan_step_index=int(payload.get("active_plan_step_index") or 0),
            original_query=str(payload.get("original_query") or ""),
            updated_at=str(payload.get("updated_at") or ""),
        )


class SessionMemoryStore:
    """Persist planner and answer memory for multi-turn sessions."""

    def __init__(self, root: Union[Path, str] = "sessions") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sanitize_session_id(session_id: str) -> str:
        cleaned = re.sub(r"[^\w\-]", "_", session_id.strip())
        if not cleaned:
            raise ValueError("session_id cannot be empty after sanitization.")
        return cleaned

    def _session_path(self, session_id: str) -> Path:
        return self.root / f"{self.sanitize_session_id(session_id)}.json"

    def exists(self, session_id: str) -> bool:
        return self._session_path(session_id).is_file()

    def load_snapshot(self, session_id: str) -> Optional[SessionSnapshot]:
        path = self._session_path(session_id)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SessionSnapshot.from_dict(payload, self.sanitize_session_id(session_id)) if isinstance(payload, dict) else None

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        snapshot = self.load_snapshot(session_id)
        return snapshot.to_dict() if snapshot else None

    def get_planner_context(self, session_id: str) -> Dict[str, Any]:
        snapshot = self.load_snapshot(session_id)
        if snapshot is None:
            return {}
        return {
            "previous_query": snapshot.original_query,
            "confirmed_constraints": snapshot.confirmed_constraints,
            "identified_entities": snapshot.identified_entities,
            "round_memories": snapshot.round_memories,
            "step_memories": snapshot.step_memories,
            "replanner_short_memory": snapshot.replanner_short_memory,
            "replanner_long_memory": snapshot.replanner_long_memory,
            "pending_replanner_step_memories": snapshot.pending_replanner_step_memories,
            "memory_timeline": snapshot.memory_timeline,
            "answer_memory": snapshot.answer_memory,
            "final_safety_review": snapshot.final_safety_review,
            "retrieval_plan": snapshot.retrieval_plan,
            "replan_decisions": snapshot.replan_decisions,
            "active_plan_step_index": snapshot.active_plan_step_index,
            "budget_state": snapshot.budget_state,
            "step_budget_status": snapshot.step_budget_status,
        }

    def save(self, session_id: str, state: Dict[str, Any]) -> Path:
        safe_id = self.sanitize_session_id(session_id)
        snapshot = SessionSnapshot.from_dict(
            {**state, "updated_at": datetime.now(timezone.utc).isoformat()}, safe_id
        )
        path = self._session_path(safe_id)
        path.write_text(json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path
