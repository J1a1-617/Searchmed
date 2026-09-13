import unittest

from skill_evolution.scripts.update_skill_failure_pipeline import build_state


class SkillFailurePipelineTests(unittest.TestCase):
    def test_wrong_cases_are_never_implicitly_marked_validated(self):
        cases = {
            "case_a": {"ground_truth": {"overall_benefit": "明显获益"}},
            "case_b": {"ground_truth": {"overall_benefit": "进展或有害"}},
        }
        predictions = {
            "case_a": {"prediction": {"overall_benefit": "无明显获益"}, "completion": "complete_llm"},
            "case_b": {"prediction": {"overall_benefit": "有限获益或稳定"}, "completion": "complete_llm"},
        }
        registry = {"skills": [{
            "skill_id": "candidate-skill",
            "status": "candidate",
            "source_cases": ["case_a"],
            "call_stages": ["generate"],
        }]}

        state = build_state(cases=cases, predictions=predictions, registry=registry)

        by_id = {row["instance_id"]: row for row in state["wrong_cases"]}
        self.assertEqual(by_id["case_a"]["skill_state"], "candidate")
        self.assertEqual(by_id["case_b"]["skill_state"], "unassigned")
        self.assertEqual(state["coverage"], {"unassigned": 1, "candidate": 1, "validated": 0})
        self.assertEqual(state["ab_jobs"][0]["source_error_cases"], ["case_a"])
        self.assertFalse(state["ab_jobs"][0]["activation_gate"]["fallback_allowed"])


if __name__ == "__main__":
    unittest.main()
