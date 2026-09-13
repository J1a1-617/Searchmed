import json
import unittest

from searchagent_retrieval.benchmark_prediction import BenchmarkPredictionGenerator


class _CapturingLLM:
    def __init__(self):
        self.user_payload = None

    def call_function(self, **kwargs):
        self.user_payload = json.loads(kwargs["user"])
        return {
            "overall_benefit": "有限获益或稳定",
            "body_lesion_recist": "SD",
            "cns_lm_recist": "NA",
            "csf_trajectory": "未评估",
            "symptom_trajectory": "无变化",
            "toxicity": {
                "max_grade": 0,
                "event": "无",
                "requires_dose_modification": False,
            },
            "confidence": "低",
            "rationale": [],
            "cited_evidence": [],
        }


class BenchmarkPredictionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.loop_result = {
            "answer_memory": {
                "claims": [{
                    "claim": "奥希替尼具有颅内活性",
                    "safe_claim": "奥希替尼具有颅内活性",
                    "status": "supported",
                    "supporting_chunk_ids": ["doc-1#chunk-1"],
                    "contradicting_chunk_ids": [],
                }],
                "evidence_by_id": {
                    "doc-1#chunk-1": {
                        "chunk_id": "doc-1#chunk-1",
                        "text": "研究在第六周评估颅内病灶，并报告客观缓解。",
                        "evidence_level": "trial_evidence",
                        "citation_json": {"pub_date": "2020-01-02"},
                        "source_file": "internal-process-field.json",
                    }
                },
            },
            "retrieval_plan": {"must_not": "reach generator"},
            "replan_decisions": [{"must_not": "reach generator"}],
            "loop_steps": [{"must_not": "reach generator"}],
        }

    def test_compact_evidence_contains_text_not_locator_metadata(self):
        rows = BenchmarkPredictionGenerator._compact_evidence(self.loop_result)

        self.assertEqual(rows, [{
            "id": "doc-1#chunk-1",
            "evidence_role": "direct",
            "claim_scope": "",
            "type": "trial_evidence",
            "content": "研究在第六周评估颅内病灶，并报告客观缓解。",
            "pub_date": "2020-01-02",
        }])

    def test_generator_payload_contains_information_fields_only(self):
        llm = _CapturingLLM()
        generator = BenchmarkPredictionGenerator(llm_client=llm, use_llm=True)

        generator.generate(
            benchmark_prompt="病例信息",
            query="不应重复传入的检索query",
            loop_result=self.loop_result,
            safety_result={
                "risk_level": "medium",
                "issues": [{"title": "证据有限", "code": "process-code"}],
                "recommended_actions": ["临床复核"],
            },
            answer_context_summary={
                "case_context": ["EGFR突变肺癌"],
                "key_findings": [{
                    "claim_id": "claim-1",
                    "text": "可能疾病稳定",
                    "evidence_ids": ["doc-1#chunk-1"],
                }],
                "unresolved_gaps": ["缺少队列"],
                "conflicts_and_limitations": [],
                "safety_boundaries": ["不可替代医生"],
                "evidence_ids": ["doc-1#chunk-1"],
            },
        )

        payload = llm.user_payload
        self.assertEqual(
            set(payload),
            {
                "case_information",
                "answer_information",
                "safety_information",
                "clinical_claims",
                "evidence",
                "bounded_evidence",
            },
        )
        self.assertNotIn("retrieval_summary", payload)
        self.assertNotIn("retrieval_plan", json.dumps(payload, ensure_ascii=False))
        self.assertNotIn("source_file", json.dumps(payload, ensure_ascii=False))
        self.assertEqual(
            payload["evidence"][0]["content"],
            "研究在第六周评估颅内病灶，并报告客观缓解。",
        )
        self.assertEqual(generator.last_generation_source, "llm")
        self.assertIsNone(generator.last_error)

    def test_generator_marks_fallback_source_after_llm_failure(self):
        class _FailingLLM:
            def call_function(self, **kwargs):
                raise RuntimeError("upstream unavailable")

        generator = BenchmarkPredictionGenerator(llm_client=_FailingLLM(), use_llm=True)
        result = generator.generate(
            benchmark_prompt="病例信息",
            query="query",
            loop_result=self.loop_result,
            safety_result={},
            answer_context_summary={},
        )

        self.assertEqual(generator.last_generation_source, "fallback")
        self.assertIn("RuntimeError", generator.last_error)
        self.assertIn("规则回退", result["rationale"][0])

    def test_only_context_validated_bounded_evidence_reaches_generator(self):
        llm = _CapturingLLM()
        generator = BenchmarkPredictionGenerator(llm_client=llm, use_llm=True)
        loop_result = {
            "answer_memory": {
                "claims": [{
                    "claim_id": "weak-1", "claim": "部分支持", "support_level": "partial",
                    "partial_support_chunk_ids": ["keep", "drop"],
                }],
                "evidence_by_id": {
                    "keep": {"text": "经Context验证的证据"},
                    "drop": {"text": "未被Context选中的证据"},
                },
            }
        }
        generator.generate(
            benchmark_prompt="病例", query="q", loop_result=loop_result,
            safety_result={},
            answer_context_summary={
                "partial_or_analog_findings": [{
                    "claim_id": "weak-1", "text": "部分支持",
                    "support_level": "partial", "evidence_ids": ["keep"],
                }]
            },
        )
        self.assertEqual([row["id"] for row in llm.user_payload["bounded_evidence"]], ["keep"])


if __name__ == "__main__":
    unittest.main()
