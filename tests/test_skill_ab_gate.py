import unittest

from skill_evolution.scripts.apply_skill_ab_gate import apply_gate


class SkillABGateTests(unittest.TestCase):
    def test_pass_requires_source_error_flip_and_correct_mount(self):
        registry = {"skills": [{"skill_id": "s1", "status": "candidate"}]}
        summary = {"source_error_gate": {"skill_id": "s1", "status": "pass", "pairs": [{
            "instance_id": "case_a",
            "baseline_wrong_reproduced": True,
            "intervention_correct": True,
            "mounted_at_required_stage": True,
            "pass": True,
        }]}}
        result = apply_gate(registry, summary, "artifact.json")
        self.assertTrue(result["passed"])
        self.assertEqual(registry["skills"][0]["status"], "active")
        self.assertEqual(registry["skills"][0]["ab_validated_source_cases"], ["case_a"])

    def test_regression_case_that_was_already_correct_cannot_activate(self):
        registry = {"skills": [{"skill_id": "s1", "status": "candidate"}]}
        summary = {"source_error_gate": {"skill_id": "s1", "status": "fail", "pairs": [{
            "instance_id": "case_regression",
            "baseline_wrong_reproduced": False,
            "intervention_correct": True,
            "mounted_at_required_stage": True,
            "pass": False,
        }]}}
        result = apply_gate(registry, summary, "artifact.json")
        self.assertFalse(result["passed"])
        self.assertEqual(registry["skills"][0]["status"], "candidate")

    def test_effective_answer_without_skill_call_cannot_activate(self):
        registry = {"skills": [{"skill_id": "s1", "status": "candidate"}]}
        summary = {"source_error_gate": {
            "skill_id": "s1",
            "status": "fail",
            "efficacy_gate": {"status": "pass"},
            "callability_gate": {"status": "fail"},
            "pairs": [{
                "instance_id": "case_a",
                "baseline_wrong_reproduced": True,
                "intervention_correct": True,
                "mounted_at_required_stage": False,
                "pass": False,
            }],
        }}
        result = apply_gate(registry, summary, "artifact.json")
        self.assertFalse(result["passed"])
        self.assertEqual(registry["skills"][0]["efficacy_gate"], "pass")
        self.assertEqual(registry["skills"][0]["callability_gate"], "fail")
        self.assertEqual(registry["skills"][0]["status"], "candidate")

    def test_skill_call_without_correcting_source_error_cannot_activate(self):
        registry = {"skills": [{"skill_id": "s1", "status": "candidate"}]}
        summary = {"source_error_gate": {
            "skill_id": "s1",
            "status": "fail",
            "efficacy_gate": {"status": "fail"},
            "callability_gate": {"status": "pass"},
            "pairs": [{
                "instance_id": "case_a",
                "baseline_wrong_reproduced": True,
                "intervention_correct": False,
                "mounted_at_required_stage": True,
                "pass": False,
            }],
        }}
        result = apply_gate(registry, summary, "artifact.json")
        self.assertFalse(result["passed"])
        self.assertEqual(registry["skills"][0]["efficacy_gate"], "fail")
        self.assertEqual(registry["skills"][0]["callability_gate"], "pass")
        self.assertEqual(registry["skills"][0]["status"], "candidate")


if __name__ == "__main__":
    unittest.main()
