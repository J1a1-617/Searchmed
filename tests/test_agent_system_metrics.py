import unittest

from skill_evolution.scripts.summarize_agent_system_metrics import compute_metrics


class AgentSystemMetricsTests(unittest.TestCase):
    def test_class_and_total_accuracy(self):
        cases = {
            "p1": {"ground_truth": {"overall_benefit": "明显获益"}},
            "p2": {"ground_truth": {"overall_benefit": "有限获益或稳定"}},
            "n1": {"ground_truth": {"overall_benefit": "进展或有害"}},
        }
        result = compute_metrics(cases, {
            "p1": "明显获益",
            "p2": "无明显获益",
            "n1": "进展或有害",
        })
        self.assertEqual(result["positive"], {"correct": 1, "count": 2, "accuracy": 0.5})
        self.assertEqual(result["negative"], {"correct": 1, "count": 1, "accuracy": 1.0})
        self.assertEqual(result["overall"]["correct"], 2)


if __name__ == "__main__":
    unittest.main()
