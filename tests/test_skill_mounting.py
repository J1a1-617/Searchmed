import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from searchagent_retrieval.benchmark_prediction import BenchmarkPredictionGenerator
from searchagent_retrieval.skill_runtime import SkillCatalog


def _prediction():
    return {
        "overall_benefit": "有限获益或稳定",
        "body_lesion_recist": "SD",
        "cns_lm_recist": "NA",
        "csf_trajectory": "未评估",
        "symptom_trajectory": "无变化",
        "toxicity": {"max_grade": 0, "event": "无", "requires_dose_modification": False},
        "confidence": "低",
        "rationale": [],
        "cited_evidence": [],
    }


class _LoopCapableLLM:
    def __init__(self):
        self.direct_calls = 0
        self.loop_calls = 0

    def call_function(self, **kwargs):
        self.direct_calls += 1
        return _prediction()

    def run_function_tool_loop(self, **kwargs):
        self.loop_calls += 1
        raise AssertionError("Skill tool loop must not run without stage-eligible candidates")


class _CountingEmbeddingCatalog(SkillCatalog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.document_embedding_calls = 0
        self.query_embedding_calls = 0

    def _encode_skill_documents(self, texts):
        self.document_embedding_calls += 1
        matrix = np.zeros((len(texts), 4), dtype="float32")
        matrix[:, 0] = 1.0
        return matrix

    def _encode_query(self, query):
        self.query_embedding_calls += 1
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype="float32")


class SkillMountingTests(unittest.TestCase):
    def test_ab_allowlist_exposes_only_target_skill(self):
        previous = os.environ.get("DYNAMIC_SKILL_ALLOWLIST")
        try:
            os.environ["DYNAMIC_SKILL_ALLOWLIST"] = "clinical-evidence-lineage"
            ids = {skill.skill_id for skill in SkillCatalog().skills}
            self.assertEqual(ids, {"clinical-evidence-lineage"})
        finally:
            if previous is None:
                os.environ.pop("DYNAMIC_SKILL_ALLOWLIST", None)
            else:
                os.environ["DYNAMIC_SKILL_ALLOWLIST"] = previous

    def test_retired_and_merged_skills_are_not_discovered(self):
        ids = {skill.skill_id for skill in SkillCatalog().skills}
        self.assertNotIn("clinical-evidence-applicability", ids)
        self.assertNotIn("evidence-absence-calibration", ids)

    def test_generate_retrieves_unified_lineage_skill(self):
        catalog = SkillCatalog(allow_candidates=True)
        for case_id in ("case_12_node_1", "case_16_node_1", "case_01_node_2"):
            state = catalog.mount(
                query=f"{case_id} prior-regimen progression and missing direct evidence",
                stage="generate",
            )
            self.assertEqual(
                [row["skill_id"] for row in state["candidate_skills"]],
                ["clinical-evidence-lineage"],
            )

    def test_lineage_query_uses_abstract_anomaly_signals(self):
        catalog = SkillCatalog(allow_candidates=True)
        signals, gaps = catalog.lineage_anomaly_signals(
            answer_memory={
                "claims": [{
                    "claim_id": "claim_1",
                    "evidence_scopes": ["osimertinib CNS response", "afatinib systemic progression"],
                    "direct_support_chunk_ids": ["e1"],
                    "support_level": "direct",
                }],
            },
            answer_context={
                "key_findings": [],
                "partial_or_analog_findings": [],
                "evidence_ids": [],
                "prohibited_attributions": ["do not transfer the prior regimen outcome"],
            },
        )
        state = catalog.mount_for_signals(
            stage="generate",
            goal="preserve_clinical_evidence_lineage",
            signals=signals,
            gaps=gaps,
        )

        self.assertIn("claim_scope_collision", signals)
        self.assertIn("unexplained_evidence_drop", signals)
        self.assertIn("cross_intervention_attribution", signals)
        self.assertNotIn("osimertinib", state["query"])
        self.assertNotIn("afatinib", state["query"])
        self.assertEqual(state["candidate_skills"][0]["skill_id"], "clinical-evidence-lineage")

    def test_no_lineage_signal_skips_skill_retrieval(self):
        state = SkillCatalog().mount_for_signals(
            stage="generate",
            goal="preserve_clinical_evidence_lineage",
            signals=[],
        )
        self.assertEqual(state["candidate_skills"], [])
        self.assertEqual(state["validation"], "not_needed")

    def test_skill_documents_are_persisted_and_runtime_only_embeds_query(self):
        with TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            model = Path(directory) / "fake-bge-large-zh-v1.5"
            first = _CountingEmbeddingCatalog(embed_model_path=model, embedding_cache_dir=cache, allow_candidates=True)
            first.retrieve("scope collision", stage="generate")
            self.assertEqual(first.document_embedding_calls, 1)
            self.assertEqual(first.query_embedding_calls, 1)

            second = _CountingEmbeddingCatalog(embed_model_path=model, embedding_cache_dir=cache, allow_candidates=True)
            second.retrieve("binding break", stage="generate")
            self.assertEqual(second.document_embedding_calls, 0)
            self.assertEqual(second.query_embedding_calls, 1)

    def test_empty_candidate_set_skips_load_skill_loop(self):
        llm = _LoopCapableLLM()
        generator = BenchmarkPredictionGenerator(llm_client=llm, use_llm=True)
        loop_result = {
            "answer_memory": {"claims": [], "evidence_by_id": {}},
            "skill_runtime": {
                "stage": "generate",
                "query": "ordinary benefit prediction",
                "candidate_skills": [],
                "selected_skill_ids": [],
                "loaded_skill_ids": [],
                "validation": "pending",
            },
        }

        generator.generate(
            benchmark_prompt="case",
            query="query",
            loop_result=loop_result,
            safety_result={},
            answer_context_summary={},
        )

        self.assertEqual(llm.direct_calls, 1)
        self.assertEqual(llm.loop_calls, 0)
        self.assertEqual(loop_result["skill_runtime"]["loaded_skill_ids"], [])
        self.assertEqual(loop_result["skill_runtime"]["validation"], "passed")


if __name__ == "__main__":
    unittest.main()
