import json
import tempfile
import unittest
from pathlib import Path

from searchagent_retrieval.session_memory import SessionMemoryStore


class SessionMemoryStoreTests(unittest.TestCase):
    def test_persists_new_memory_schema(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = SessionMemoryStore(root)
            store.save("s1", {
                "original_query": "q",
                "round_memories": [{"round": 1}],
                "loop_steps": [{"step": 1}],
                "final_answer": "测试回答",
                "workflow_trace": {"has_final_answer": True},
                "answer_memory": {"claims": [{"claim_id": "c1"}], "evidence_by_id": {}},
                "skill_runtime": {
                    "candidate_skills": [{"skill_id": "clinical-evidence-lineage"}],
                    "loaded_skill_ids": ["clinical-evidence-lineage"],
                },
                "final_safety_review": {"risk_level": "low"},
                "citations": {"claim_citations": []},
                "retrieval_plan": {"steps": [{"step_id": "S1"}]},
                "replan_decisions": [{"action": "retry_current_step"}],
                "active_plan_step_index": 0,
                "memory_timeline": {
                    "current_agent_round": 4,
                    "compacted_rounds": [1],
                    "recent_full_rounds": [2, 3],
                },
            })
            payload = store.load("s1")
            self.assertEqual(payload["round_memories"][0]["round"], 1)
            self.assertEqual(payload["loop_steps"][0]["step"], 1)
            self.assertEqual(payload["final_answer"], "测试回答")
            self.assertEqual(payload["answer_memory"]["claims"][0]["claim_id"], "c1")
            self.assertEqual(
                payload["skill_runtime"]["loaded_skill_ids"],
                ["clinical-evidence-lineage"],
            )
            self.assertEqual(payload["retrieval_plan"]["steps"][0]["step_id"], "S1")
            self.assertEqual(payload["memory_timeline"]["current_agent_round"], 4)
            self.assertEqual(
                store.get_planner_context("s1")["memory_timeline"]["recent_full_rounds"],
                [2, 3],
            )
            self.assertNotIn("long_memory", payload)

    def test_reads_legacy_session_into_answer_memory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "old.json"
            path.write_text(json.dumps({
                "long_memory": ["旧摘要"],
                "retrieved_evidence": [{"chunk_id": "e1", "text": "evidence"}],
            }, ensure_ascii=False), encoding="utf-8")
            context = SessionMemoryStore(root).get_planner_context("old")
            self.assertEqual(context["answer_memory"]["legacy_context"], ["旧摘要"])
            self.assertIn("e1", context["answer_memory"]["evidence_by_id"])


if __name__ == "__main__":
    unittest.main()
