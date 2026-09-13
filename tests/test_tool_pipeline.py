import unittest

from searchagent_retrieval.tool_pipeline import ToolExecutionPipeline


class ToolExecutionPipelineTests(unittest.TestCase):
    def test_one_call_has_stages_and_exactly_one_terminal_result(self):
        events = []
        pipeline = ToolExecutionPipeline(
            lambda event_type, data: events.append((event_type, data))
        )

        def handler():
            pipeline.stage("dense_search", {"output": {"hit_count": 3}})
            pipeline.stage("qwen_rerank", {"output": {"candidate_ids": ["c1"]}})
            return {"ok": True, "fetched_evidence_ids": ["c1"]}

        result = pipeline.execute(
            call_id="call-1",
            tool="execute_retrieval_batch",
            arguments={"actions": [{"tool": "dense_search"}]},
            handler=handler,
            turn=1,
            operation="tool_loop:submit_step_execution_report",
        )

        self.assertEqual(result["fetched_evidence_ids"], ["c1"])
        self.assertEqual(
            [event_type for event_type, _ in events],
            ["tool/call", "tool/stage", "tool/stage", "tool/result"],
        )
        self.assertTrue(all(data["call_id"] == "call-1" for _, data in events))
        self.assertEqual(sum(kind == "tool/result" for kind, _ in events), 1)

    def test_exception_is_normalized_and_still_has_one_result(self):
        events = []
        pipeline = ToolExecutionPipeline(
            lambda event_type, data: events.append((event_type, data))
        )

        result = pipeline.execute(
            call_id="call-error",
            tool="execute_retrieval_batch",
            arguments={},
            handler=lambda: (_ for _ in ()).throw(RuntimeError("backend down")),
            turn=1,
            operation="test",
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "RuntimeError")
        terminal = [data for kind, data in events if kind == "tool/result"]
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["status"], "error")


if __name__ == "__main__":
    unittest.main()
