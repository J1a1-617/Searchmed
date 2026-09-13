import unittest

from searchagent_retrieval.safety_gate import ClinicalSafetyGate


class ClinicalSafetyGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ClinicalSafetyGate()

    def test_flags_high_risk_for_dose_ddi_and_severe_toxicity(self) -> None:
        retrieval_results = {
            "structured_search": [],
            "dense_search": [{"id": "doc#chunk-1", "text": "osimertinib interaction with voriconazole"}],
            "bm25_search": [],
            "hybrid_search": [],
            "fetch_evidence": [
                {
                    "chunk_id": "doc#chunk-1",
                    "evidence_level": "case_report_evidence",
                    "text": "Patient developed grade 4 hepatotoxicity and treatment-related death.",
                },
                {
                    "chunk_id": "doc#chunk-2",
                    "evidence_level": "structured_extraction_evidence",
                    "text": "Serious adverse reaction with CYP3A4 inhibitor co-administration.",
                },
            ],
        }
        output = self.gate.evaluate(
            query="奥希替尼 80 mg qd 与伏立康唑是否有相互作用？",
            query_type="ddi",
            constraints={"ddi_terms": ["drug interaction"]},
            retrieval_results=retrieval_results,
        )
        issue_codes = {item["code"] for item in output["issues"]}
        self.assertTrue(output["requires_human_review"])
        self.assertEqual(output["risk_level"], "critical")
        self.assertIn("dose_recommendation_risk", issue_codes)
        self.assertIn("missing_ddi_evidence", issue_codes)
        self.assertIn("severe_toxicity_signal", issue_codes)

    def test_keeps_low_risk_with_sufficient_higher_level_evidence(self) -> None:
        retrieval_results = {
            "structured_search": [{"id": "doc#case-1", "text": "NSCLC case summary"}],
            "dense_search": [],
            "bm25_search": [],
            "hybrid_search": [],
            "fetch_evidence": [
                {"chunk_id": "doc#chunk-1", "evidence_level": "guideline_evidence", "text": "NCCN guideline recommends regimen A."},
                {
                    "chunk_id": "doc#chunk-2",
                    "evidence_level": "drug_label_or_ddi_rule_evidence",
                    "text": "Label: no clinically significant interaction with drug B.",
                },
                {"chunk_id": "doc#chunk-3", "evidence_level": "clinical_trial_evidence", "text": "Phase III trial showed improved PFS."},
            ],
        }
        output = self.gate.evaluate(
            query="EGFR 突变患者有哪些推荐方案？",
            query_type="treatment_advice",
            constraints={"ddi_terms": []},
            retrieval_results=retrieval_results,
        )
        self.assertFalse(output["requires_human_review"])
        self.assertEqual(output["risk_level"], "low")
        self.assertEqual(output["issues"], [])

    def test_llm_reflection_reviews_claims_without_removing_rule_issues(self) -> None:
        class MockLLM:
            def call_function(self, system, user, temperature=0.3, **kwargs):
                return {"claim_reviews":[{"claim_id":"c1","decision":"revise","violations":["证据等级低"],"required_revision":"降低确定性","safe_claim":"病例层面可能获益","evidence_refs":["e1"]}],"cross_claim_conflicts":[],"additional_issues":[],"reflection_summary":"需降级"}

        gate = ClinicalSafetyGate(llm_client=MockLLM(), use_llm=True)
        output = gate.evaluate(
            query="治疗建议",
            query_type="treatment_advice",
            constraints={},
            retrieval_results={"fetch_evidence": [{"chunk_id": "e1", "evidence_level": "case_report_evidence", "text": "response"}]},
            answer_memory={"claims": [{
                "claim_id": "c1", "claim": "确定获益", "support_level": "direct",
                "direct_support_chunk_ids": ["e1"],
            }], "evidence_by_id": {"e1": {"text": "response"}}},
        )
        self.assertEqual(output["review_mode"], "rules+llm")
        self.assertEqual(output["claim_reviews"][0]["safe_claim"], "病例层面可能获益")
        self.assertIn("insufficient_evidence", {item["code"] for item in output["issues"]})

    def test_llm_reflection_prefilters_claims_that_cannot_enter_context(self) -> None:
        captured = []

        class MockLLM:
            def call_function(self, user, **kwargs):
                captured.append(user)
                return {
                    "claim_reviews": [{
                        "claim_id": "keep", "decision": "approve", "violations": [],
                        "required_revision": "", "safe_claim": "保留", "evidence_refs": ["e1"],
                    }],
                    "cross_claim_conflicts": [], "additional_issues": [], "reflection_summary": "通过",
                }

        claims = [
            {"claim_id": "keep", "claim": "保留", "support_level": "direct", "direct_support_chunk_ids": ["e1"]},
            {"claim_id": "rejected", "status": "rejected_by_safety_gate", "support_level": "direct", "direct_support_chunk_ids": ["e2"]},
            {"claim_id": "irrelevant", "status": "irrelevant", "support_level": "direct", "direct_support_chunk_ids": ["e3"]},
            {"claim_id": "no-evidence", "support_level": "direct", "direct_support_chunk_ids": []},
            {"claim_id": "unsupported", "support_level": "unsupported", "contradicting_chunk_ids": ["e4"]},
        ]
        output = ClinicalSafetyGate(llm_client=MockLLM(), use_llm=True).evaluate(
            query="q", query_type=None, constraints={}, retrieval_results={"fetch_evidence": []},
            answer_memory={"claims": claims, "evidence_by_id": {"e1": {"text": "证据"}}},
        )
        self.assertEqual([row["claim_id"] for row in output["claim_reviews"]], ["keep"])
        self.assertEqual(output["claim_prefilter"], {"input_claims": 5, "reviewed_claims": 1, "excluded_claims": 4})
        self.assertNotIn('"claim_id": "rejected"', captured[0])


if __name__ == "__main__":
    unittest.main()
