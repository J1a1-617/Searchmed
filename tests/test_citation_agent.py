import unittest
from typing import Any, Dict
from unittest.mock import MagicMock

from searchagent_retrieval.agent_loop import CitationAgent


def _sample_route_result() -> Dict[str, Any]:
    return {
        "query_type": "mutation_drug",
        "contradiction_check": {
            "verdict": "supported_with_monitoring",
            "top_supporting_hit_ids": ["doc#chunk-1"],
            "top_counter_hit_ids": ["doc#chunk-2"],
        },
        "evidence_layering": {
            "query": "EGFR突变能否用奥希替尼？",
            "layer_distribution": [{"evidence_level": "case_report_evidence", "count": 2}],
            "assessed_evidence": [
                {
                    "chunk_id": "doc#chunk-1",
                    "evidence_level": "case_report_evidence",
                    "text": "Patient with EGFR mutation achieved partial response on osimertinib.",
                    "pmid": "12345",
                    "title": "Case A",
                    "relevance_signals": {
                        "is_supporting": True,
                        "is_contradicting": False,
                        "is_safety_risk": False,
                    },
                },
                {
                    "chunk_id": "doc#chunk-2",
                    "evidence_level": "case_report_evidence",
                    "text": "Disease progression after osimertinib treatment.",
                    "pmid": "67890",
                    "title": "Case B",
                    "relevance_signals": {
                        "is_supporting": False,
                        "is_contradicting": True,
                        "is_safety_risk": False,
                    },
                },
            ],
        },
    }


class CitationAgentTests(unittest.TestCase):
    def test_rules_review_classifies_evidence(self) -> None:
        agent = CitationAgent(use_llm=False)
        review = agent.review(_sample_route_result())

        self.assertEqual(review["review_mode"], "rules")
        self.assertEqual(len(review["supporting_evidence"]), 1)
        self.assertEqual(len(review["contradicting_evidence"]), 1)
        self.assertEqual(review["verdict"], "supported_with_monitoring")

    def test_llm_review_enriches_evidence_and_verdict(self) -> None:
        llm_client = MagicMock()
        llm_client.call_function.return_value = {
          "verdict": "partially_supports",
          "evidence_assessments": [
            {
              "chunk_id": "doc#chunk-1",
              "relevance_score": 0.92,
              "credibility_score": 0.81,
              "is_supporting": True,
              "is_contradicting": False,
              "is_safety_risk": False,
              "summary": "病例显示EGFR突变患者对奥希替尼有缓解。"
            },
            {
              "chunk_id": "doc#chunk-2",
              "relevance_score": 0.75,
              "credibility_score": 0.7,
              "is_supporting": False,
              "is_contradicting": True,
              "is_safety_risk": False,
              "summary": "病例提示后续进展，构成反证。"
            }
          ]
        }

        agent = CitationAgent(llm_client=llm_client, use_llm=True)
        review = agent.review(_sample_route_result())

        self.assertEqual(review["review_mode"], "llm")
        self.assertEqual(review["verdict"], "partially_supports")
        self.assertEqual(review["supporting_evidence"][0]["llm_relevance_score"], 0.92)
        self.assertEqual(
            review["supporting_evidence"][0]["llm_summary"],
            "病例显示EGFR突变患者对奥希替尼有缓解。",
        )
        self.assertEqual(len(review["llm_evidence_assessments"]), 2)
        llm_client.call_function.assert_called_once()

    def test_llm_failure_falls_back_to_rules(self) -> None:
        llm_client = MagicMock()
        llm_client.call_function.side_effect = RuntimeError("api down")

        agent = CitationAgent(llm_client=llm_client, use_llm=True)
        review = agent.review(_sample_route_result())

        self.assertEqual(review["review_mode"], "rules")
        self.assertEqual(review["verdict"], "supported_with_monitoring")

    def test_llm_invalid_json_falls_back_to_rules(self) -> None:
        llm_client = MagicMock()
        llm_client.call_function.return_value = {"unexpected": []}

        agent = CitationAgent(llm_client=llm_client, use_llm=True)
        review = agent.review(_sample_route_result())

        self.assertEqual(review["review_mode"], "rules")


if __name__ == "__main__":
    unittest.main()
