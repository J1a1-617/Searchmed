import unittest

from searchagent_retrieval.agent_loop import _memory_timeline


class MemoryTimelineTests(unittest.TestCase):
    def test_timeline_distinguishes_current_compacted_and_recent_rounds(self):
        timeline = _memory_timeline(
            current_round=6,
            round_memories=[
                {"round": value, "memory_id": f"agent-round:{value}"}
                for value in range(1, 6)
            ],
            compaction_events=[{"covered_rounds": [1, 2, 3]}],
        )

        self.assertEqual(timeline["current_agent_round"], 6)
        self.assertEqual(timeline["latest_completed_agent_round"], 5)
        self.assertEqual(timeline["compacted_rounds"], [1, 2, 3])
        self.assertEqual(timeline["recent_full_rounds"], [4, 5])
        self.assertIn("不是患者治疗周期", timeline["instruction"])


if __name__ == "__main__":
    unittest.main()
