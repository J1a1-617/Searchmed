import json
import tempfile
import unittest
from pathlib import Path

from searchagent_retrieval.trajectory import TrajectoryEventStore


class TrajectoryEventStoreTests(unittest.TestCase):
    def test_appends_ordered_jsonl_and_redacts_sensitive_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl"
            with TrajectoryEventStore(path, context={"session_id": "s1"}) as store:
                store.append("session/start", {"query": "q", "api_key": "secret"})
                store.append("llm/request", {"headers": {"Authorization": "Bearer secret"}})

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["seq"] for row in rows], [1, 2])
            self.assertEqual([row["type"] for row in rows], ["session/start", "llm/request"])
            self.assertEqual(rows[0]["session_id"], "s1")
            self.assertEqual(rows[0]["data"]["api_key"], "[REDACTED]")
            self.assertEqual(rows[1]["data"]["headers"]["Authorization"], "[REDACTED]")

    def test_reopen_continues_sequence_after_existing_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.jsonl"
            with TrajectoryEventStore(path) as store:
                store.append("session/start")
            with path.open("a", encoding="utf-8") as handle:
                handle.write("incomplete-json")
            with TrajectoryEventStore(path) as store:
                event = store.append("session/end")
            self.assertEqual(event["seq"], 2)


if __name__ == "__main__":
    unittest.main()
