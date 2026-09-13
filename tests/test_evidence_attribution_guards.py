import unittest

from searchagent_retrieval.agent_loop import AnswerMemoryAgent, EvidenceReviewAgent, StepMemoryAgent
from searchagent_retrieval.answer_context import AnswerContextAgent
from searchagent_retrieval.benchmark_prediction import BenchmarkPredictionGenerator


class _PromotingMemoryLLM:
    """Simulate an LLM incorrectly promoting a different-drug analogy."""

    def call_function(self, **kwargs):
        return {
            "claims": [{
                "claim_id": "claim_01",
                "claim": "伏美替尼会导致严重肝毒性",
                "status": "supported",
                "supporting_chunk_ids": ["capmatinib-1"],
                "contradicting_chunk_ids": [],
                "direct_support_chunk_ids": ["capmatinib-1"],
                "partial_support_chunk_ids": [],
                "analog_support_chunk_ids": ["capmatinib-1"],
                "subject_entity": "伏美替尼",
                "claim_scope": "目标药物肝毒性",
                "support_level": "direct",
                "source_rounds": [1],
                "confidence": 0.95,
                "safety_status": "pending",
            }],
            "informative_rounds": [1],
        }


class EvidenceAttributionGuardTests(unittest.TestCase):
    def test_step_memory_is_always_deterministic(self):
        class _MustNotBeCalled:
            def call_function(self, **kwargs):
                raise AssertionError("StepMemory must not call an LLM")

        agent = StepMemoryAgent(llm_client=_MustNotBeCalled(), use_llm=True)
        self.assertFalse(agent.use_llm)
        self.assertIsNone(agent.llm_client)

    def test_rule_step_memory_keeps_reviewed_low_score_analog_once(self):
        evidence = {
            "chunk_id": "analog-low",
            "text": "Capmatinib caused severe liver toxicity and was discontinued.",
            "llm_relevance_score": 0.2,
            "evidence_role": "analog_support",
            "target_entity_match": "different",
            "supports_dimensions": ["liver toxicity", "discontinuation"],
            "mismatched_dimensions": ["drug"],
            "unreported_dimensions": ["time window"],
            "claim_scope": "Capmatinib caused severe liver toxicity and was discontinued.",
            "relevance_signals": {
                "is_supporting": True, "is_contradicting": False, "is_safety_risk": True,
            },
        }
        memory = StepMemoryAgent().maintain(
            round_number=1,
            main_question="伏美替尼是否导致肝损伤并停药",
            query="q",
            route_result={"step_execution_report": {"accepted_evidence_ids": ["analog-low"]}},
            evidence_review={
                "supporting_evidence": [evidence],
                "contradicting_evidence": [],
                "safety_risks": [evidence],
                "insufficient_evidence": [],
            },
            seen_chunk_ids=set(),
            plan_step={"evidence_lane": "ddi_safety"},
        )
        self.assertEqual(len(memory["accepted_evidence"]), 1)
        self.assertEqual(memory["accepted_evidence"][0]["evidence_role"], "analog_support")
        self.assertEqual(memory["rejected_evidence"], [])

    def test_evidence_review_keeps_original_compact_assessment(self):
        assessed = [{"chunk_id": "gold-1", "text": "伏美替尼治疗后ALT升高"}]
        payload = {"verdict": "partially_supports", "evidence_assessments": [{
            "chunk_id": "gold-1", "relevance_score": 0.9, "credibility_score": 0.9,
            "is_supporting": True, "is_contradicting": False, "is_safety_risk": True,
            "evidence_role": "direct_support", "target_entity_match": "exact",
            "supports_dimensions": ["ALT elevation"], "mismatched_dimensions": [],
            "unreported_dimensions": ["dose"],
            "claim_scope": "伏美替尼治疗后ALT升高",
            "summary": "目标药物和毒性结局直接匹配",
        }]}
        merged, _ = EvidenceReviewAgent()._merge_llm_assessments(assessed, payload)
        self.assertTrue(merged[0]["relevance_signals"]["is_supporting"])
        self.assertEqual(merged[0]["llm_summary"], "目标药物和毒性结局直接匹配")
        self.assertEqual(merged[0]["claim_scope"], "伏美替尼治疗后ALT升高")
        self.assertEqual(merged[0]["unreported_dimensions"], ["dose"])

    def test_evidence_review_omission_discards_irrelevant_input(self):
        assessed = [
            {"chunk_id": "keep", "text": "direct"},
            {"chunk_id": "drop", "text": "irrelevant"},
        ]
        payload = {"verdict": "partially_supports", "evidence_assessments": [{
            "chunk_id": "keep", "claim_scope": "direct", "overall_role": "direct_support",
        }]}
        merged, _ = EvidenceReviewAgent()._merge_llm_assessments(assessed, payload)
        self.assertEqual([row["chunk_id"] for row in merged], ["keep"])

    def test_evidence_review_splits_large_structured_output(self):
        import json

        class _ReviewLLM:
            def __init__(self):
                self.calls = 0

            def call_function(self, user, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return {"verdict": "invalid", "evidence_assessments": []}
                data = json.loads(user.split("\n\n这是同一个 EvidenceReview", 1)[0])
                return {
                    "verdict": "partially_supports",
                    "evidence_assessments": [{
                        "chunk_id": row["chunk_id"], "claim_scope": row["text"],
                        "overall_role": "partial_support", "is_supporting": True,
                    } for row in data["evidence"]],
                }

        llm = _ReviewLLM()
        assessed = [{"chunk_id": f"e{i}", "text": f"fact {i}"} for i in range(9)]
        output = EvidenceReviewAgent(llm_client=llm, use_llm=True)._review_with_llm({
            "evidence_layering": {"assessed_evidence": assessed}, "contradiction_check": {},
        })
        self.assertEqual(llm.calls, 4)
        self.assertEqual(len(output["supporting_evidence"]), 9)
        self.assertTrue(output["structured_output_recovery"]["activated"])

    def test_agent_prompts_have_non_overlapping_responsibilities(self):
        self.assertIn("一次性的医学解释和支持边界判断", __import__(
            "searchagent_retrieval.agent_loop", fromlist=["_CITATION_SYSTEM_PROMPT"]
        )._CITATION_SYSTEM_PROMPT)
        self.assertIn("默认继承这些字段", StepMemoryAgent.SYSTEM_PROMPT)
        self.assertIn("才修改发生错误的单个字段", StepMemoryAgent.SYSTEM_PROMPT)
        self.assertIn("不负责重新阅读证据", AnswerMemoryAgent.SYSTEM_PROMPT)
        self.assertIn("不得把不同证据", AnswerMemoryAgent.SYSTEM_PROMPT)

    def test_rule_memory_preserves_explicit_exact_direct_evidence(self):
        evidence = {
            "chunk_id": "gold-1", "llm_relevance_score": 0.9,
            "target_entity_match": "exact", "mismatched_dimensions": [],
            "unreported_dimensions": ["dose"], "claim_scope": "伏美替尼可导致ALT升高",
        }
        memory = StepMemoryAgent(use_llm=False).maintain(
            round_number=1, main_question="q", query="q",
            route_result={"step_execution_report": {}},
            evidence_review={"supporting_evidence": [evidence]},
            seen_chunk_ids=set(), plan_step={},
        )
        self.assertEqual(memory["accepted_evidence"][0]["evidence_role"], "direct_support")

    def test_different_drug_cannot_be_promoted_to_direct_support(self):
        round_memory = {
            "round": 1,
            "accepted_evidence": [{
                "chunk_id": "capmatinib-1",
                "text": "Capmatinib caused grade 3 ALT elevation.",
                "evidence_role": "analog_support",
                "target_entity_match": "different",
                "claim_scope": "Capmatinib hepatotoxicity only",
                "supports_dimensions": ["ALT elevation"],
                "mismatched_dimensions": ["drug"],
                "unreported_dimensions": ["time", "treatment discontinuation"],
            }],
            "question_information_gain": {"new_chunk_ids": ["capmatinib-1"]},
        }
        memory = AnswerMemoryAgent(
            llm_client=_PromotingMemoryLLM(), use_llm=True
        ).update({}, round_memory)
        claim = memory["claims"][0]

        self.assertEqual(claim["direct_support_chunk_ids"], [])
        self.assertEqual(claim["supporting_chunk_ids"], [])
        self.assertEqual(claim["analog_support_chunk_ids"], ["capmatinib-1"])
        self.assertEqual(claim["support_level"], "analog")
        self.assertEqual(claim["claim"], "Capmatinib hepatotoxicity only")
        self.assertEqual(
            claim["unsupported_dimensions"],
            ["drug", "time", "treatment discontinuation"],
        )
        self.assertNotEqual(claim["status"], "supported")

        loop_result = {
            "state": {"confirmed_constraints": {"drugs": ["伏美替尼"]}},
            "answer_memory": memory,
            "replanner_long_memory": {},
            "replanner_short_memory": {},
        }
        context = AnswerContextAgent(use_llm=False).summarize(
            query="伏美替尼肝毒性", loop_result=loop_result, safety_result={}
        )
        self.assertEqual(context["key_findings"], [])
        self.assertEqual(context["partial_or_analog_findings"][0]["text"], "Capmatinib hepatotoxicity only")
        self.assertEqual(context["partial_or_analog_findings"][0]["support_level"], "analog")
        self.assertEqual(context["partial_or_analog_findings"][0]["evidence_ids"], ["capmatinib-1"])
        self.assertTrue(context["prohibited_attributions"])
        self.assertIn("treatment discontinuation", context["prohibited_attributions"][0])
        self.assertEqual(BenchmarkPredictionGenerator._clinical_claims(loop_result), [])
        self.assertEqual(BenchmarkPredictionGenerator._compact_evidence(loop_result), [])
        self.assertEqual(
            BenchmarkPredictionGenerator._compact_bounded_evidence(loop_result)[0]["support_level"],
            "analog",
        )

    def test_context_allows_paraphrase_when_claim_id_and_evidence_match(self):
        class _ContextLLM:
            def call_function(self, **kwargs):
                return {
                    "case_context": [],
                    "key_findings": [{
                        "claim_id": "claim_gold",
                        "text": "治疗期间观察到转氨酶升高",
                        "evidence_ids": ["gold-1"],
                    }],
                    "partial_or_analog_findings": [], "unresolved_gaps": [],
                    "conflicts_and_limitations": [], "safety_boundaries": [],
                    "prohibited_attributions": [], "evidence_ids": ["gold-1"],
                }

        loop_result = {
            "state": {"confirmed_constraints": {}},
            "answer_memory": {"claims": [{
                "claim_id": "claim_gold", "claim": "伏美替尼可能引起ALT/AST升高",
                "status": "supported", "confidence": 0.9, "support_level": "direct",
                "direct_support_chunk_ids": ["gold-1"], "supporting_chunk_ids": ["gold-1"],
                "partial_support_chunk_ids": [], "analog_support_chunk_ids": [],
                "contradicting_chunk_ids": [],
            }]},
            "replanner_long_memory": {}, "replanner_short_memory": {},
        }
        context = AnswerContextAgent(llm_client=_ContextLLM(), use_llm=True).summarize(
            query="q", loop_result=loop_result, safety_result={}
        )
        self.assertEqual(context["key_findings"][0]["claim_id"], "claim_gold")
        self.assertEqual(context["key_findings"][0]["text"], "治疗期间观察到转氨酶升高")

    def test_context_cannot_enrich_weak_evidence_text(self):
        class _WeakEnrichingContextLLM:
            def call_function(self, **kwargs):
                return {
                    "case_context": [],
                    "key_findings": [],
                    "partial_or_analog_findings": [{
                        "claim_id": "weak_01",
                        "text": "伏美替尼120mg可在8周内导致严重肝损伤并需要停药",
                        "support_level": "analog",
                        "evidence_ids": ["other-drug-1"],
                    }],
                    "unresolved_gaps": [],
                    "conflicts_and_limitations": [],
                    "safety_boundaries": [],
                    "prohibited_attributions": [],
                    "evidence_ids": ["other-drug-1"],
                }

        source_claim = "其他EGFR-TKI病例报告过转氨酶升高"
        loop_result = {
            "state": {"confirmed_constraints": {"drugs": ["伏美替尼"]}},
            "answer_memory": {"claims": [{
                "claim_id": "weak_01",
                "claim": source_claim,
                "status": "provisional",
                "confidence": 0.4,
                "support_level": "analog",
                "direct_support_chunk_ids": [],
                "supporting_chunk_ids": [],
                "partial_support_chunk_ids": [],
                "analog_support_chunk_ids": ["other-drug-1"],
                "contradicting_chunk_ids": [],
            }]},
            "replanner_long_memory": {},
            "replanner_short_memory": {},
        }
        context = AnswerContextAgent(
            llm_client=_WeakEnrichingContextLLM(), use_llm=True
        ).summarize(query="伏美替尼肝损伤", loop_result=loop_result, safety_result={})

        finding = context["partial_or_analog_findings"][0]
        self.assertEqual(finding["text"], source_claim)
        self.assertNotIn("120mg", finding["text"])
        self.assertNotIn("8周", finding["text"])
        self.assertNotIn("需要停药", finding["text"])

    def test_context_splits_claims_and_merges_complete_parts(self):
        import json

        class _ContextLLM:
            def __init__(self):
                self.calls = 0

            def call_function(self, user, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return {"case_context": []}
                data = json.loads(user)
                return {
                    "case_context": ["按治疗先后解释"],
                    "key_findings": [{
                        "claim_id": row["claim_id"], "text": row["claim"],
                        "evidence_ids": row["direct_support_chunk_ids"],
                    } for row in data["current_claims"]],
                    "partial_or_analog_findings": [], "unresolved_gaps": [],
                    "conflicts_and_limitations": [], "safety_boundaries": [],
                    "prohibited_attributions": [],
                    "evidence_ids": [cid for row in data["current_claims"] for cid in row["direct_support_chunk_ids"]],
                }

        claims = [{
            "claim_id": f"c{i}", "claim": f"第{i}个时间点事实", "status": "supported",
            "support_level": "direct", "direct_support_chunk_ids": [f"e{i}"],
            "supporting_chunk_ids": [f"e{i}"], "partial_support_chunk_ids": [],
            "analog_support_chunk_ids": [], "contradicting_chunk_ids": [],
        } for i in range(6)]
        llm = _ContextLLM()
        output = AnswerContextAgent(llm_client=llm, use_llm=True).summarize(
            query="q", loop_result={
                "state": {"confirmed_constraints": {}}, "answer_memory": {"claims": claims},
                "replanner_long_memory": {}, "replanner_short_memory": {},
            }, safety_result={},
        )
        self.assertEqual(llm.calls, 3)
        self.assertEqual(len(output["key_findings"]), 6)
        self.assertEqual(output["case_context"], ["按治疗先后解释"])
        self.assertTrue(output["structured_output_recovery"]["activated"])


if __name__ == "__main__":
    unittest.main()
