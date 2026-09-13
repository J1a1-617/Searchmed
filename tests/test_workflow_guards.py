import unittest

from searchagent_retrieval.answer_generator import AnswerGenerator
from searchagent_retrieval.answer_context import AnswerContextAgent
from searchagent_retrieval.llm_rerank import LLMReranker
from searchagent_retrieval.planning import MultiStepPlanningAgent
from searchagent_retrieval.router import (
    QUERY_UNDERSTANDING_SYSTEM_PROMPT,
    QUERY_UNDERSTANDING_USER_TEMPLATE,
    RetrievalRouter,
)
from searchagent_retrieval.workflow_trace import WorkflowTrace
from searchagent_retrieval.tools import SearchHit


class MockLLM:
    def __init__(self, response) -> None:
        if isinstance(response, list):
            self.responses = list(response)
        else:
            self.responses = [response]
        self.calls = 0

    def chat(self, system: str, user: str, temperature: float = 0.3, **kwargs) -> str:
        self.calls += 1
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]

    def call_function(self, system: str, user: str, temperature: float = 0.3, **kwargs):
        self.calls += 1
        response = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        return __import__("json").loads(response)


class StubRetrievalTools:
    def structured_search(self, constraints, top_k=20):
        return []

    def dense_search(self, query, top_k=20, spaces=None):
        return []

    def bm25_search(self, query, top_k=20):
        return []

    def hybrid_search(self, query, constraints=None, top_k=20, weights=None):
        return []

    def rerank(self, query, hits, top_k):
        return []

    def fetch_evidence(self, *args, **kwargs):
        return []


class WorkflowGuardTests(unittest.TestCase):
    def test_query_understanding_prompt_separates_prediction_from_advice(self) -> None:
        self.assertIn("不要把问题归入一个固定类别", QUERY_UNDERSTANDING_SYSTEM_PROMPT)
        self.assertIn("结局与时间", QUERY_UNDERSTANDING_SYSTEM_PROMPT)
        self.assertIn("治疗时间线属于核心语义", QUERY_UNDERSTANDING_SYSTEM_PROMPT)
        self.assertIn("联合方案本身不是DDI", QUERY_UNDERSTANDING_SYSTEM_PROMPT)

    def test_query_understanding_prompt_preserves_treatment_roles_and_lab_boundaries(self) -> None:
        self.assertIn("只放当前/计划评估的index treatment", QUERY_UNDERSTANDING_SYSTEM_PROMPT)
        self.assertIn("既往已停用药物放入prior_treatments", QUERY_UNDERSTANDING_SYSTEM_PROMPT)
        self.assertIn("ALT/AST绝不进入gene_alterations", QUERY_UNDERSTANDING_SYSTEM_PROMPT)
        self.assertIn("不得猜测T790M", QUERY_UNDERSTANDING_SYSTEM_PROMPT)
        self.assertIn("当前index treatment的权威来源", QUERY_UNDERSTANDING_SYSTEM_PROMPT)
        self.assertIn("不要罗列原文完全未提及", QUERY_UNDERSTANDING_SYSTEM_PROMPT)
        self.assertIn("current_treatments只放当前/计划评估方案", QUERY_UNDERSTANDING_USER_TEMPLATE)

    def test_soft_schema_is_pure_and_adapter_is_separate(self) -> None:
        llm = MockLLM([
            __import__("json").dumps({
                "patient_facts": {
                    "cancer_type": "肺腺癌",
                    "gene_alterations": ["EGFR 19del"],
                    "metastatic_sites": ["脑"],
                },
                "treatment_context": {
                    "current_treatments": [{"name": "奥希替尼", "dose": "80mg"}],
                    "prior_treatments": ["吉非替尼"],
                    "regimen_details": ["80mg qd"],
                },
                "target_outcomes": ["8-12周颅内疗效"],
                "information_needs": ["直接匹配病例", "相反结局"],
                "ambiguities": [],
                "additional_context": {"time_window_weeks": [8, 12]},
                "model_extension": {"useful": True},
            }),
        ])
        router = RetrievalRouter(retrieval_tools=object(), llm_client=llm, use_llm=True)  # type: ignore[arg-type]
        understanding = router.understand_query("test")
        self.assertNotIn("hard_constraints", understanding)
        self.assertNotIn("soft_hints", understanding)
        adapter = router.adapt_problem_representation(understanding)
        self.assertEqual(adapter["hard_constraints"]["drugs"], ["奥希替尼"])
        self.assertEqual(adapter["hard_constraints"]["gene_alterations"], ["EGFR 19del"])
        self.assertIn("既往治疗:吉非替尼", adapter["soft_hints"]["clinical_terms"])
        self.assertEqual(understanding["model_extension"], {"useful": True})

    def test_query_understanding_retries_once_after_invalid_json(self) -> None:
        llm = MockLLM([
            "I cannot produce JSON right now.",
            '{"patient_facts":{},"treatment_context":{"current_treatments":[],"prior_treatments":[]},"target_outcomes":[],"information_needs":[],"ambiguities":[]}',
        ])
        router = RetrievalRouter(retrieval_tools=object(), llm_client=llm, use_llm=True)  # type: ignore[arg-type]

        understanding = router.understand_query("test")

        self.assertEqual(llm.calls, 2)
        self.assertEqual(understanding["patient_facts"], {})

    def test_query_understanding_retries_whole_task_after_empty_content(self) -> None:
        class EmptyThenJson:
            def __init__(self):
                self.calls = 0

            def chat(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("LLM returned empty content.")
                return '{"patient_facts":{},"treatment_context":{},"target_outcomes":[],"information_needs":[],"ambiguities":[]}'

        llm = EmptyThenJson()
        router = RetrievalRouter(retrieval_tools=object(), llm_client=llm, use_llm=True)  # type: ignore[arg-type]
        understanding = router.understand_query("test")
        self.assertEqual(llm.calls, 2)
        self.assertEqual(understanding["patient_facts"], {})

    def test_query_understanding_raises_after_three_invalid_json_outputs(self) -> None:
        llm = MockLLM(["not json", "still not json", "also not json"])
        router = RetrievalRouter(retrieval_tools=object(), llm_client=llm, use_llm=True)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "after 3 attempts"):
            router.understand_query("test")

        self.assertEqual(llm.calls, 3)

    def test_readable_workflow_renders_replanner_memories(self) -> None:
        trace = WorkflowTrace().build(
            query="q",
            loop_result={"state": {}, "replanner_short_memory": {"active_step_id": "S1"}, "replanner_long_memory": {"step_progress": ["S1"]}},
            safety_result={},
            final_answer="answer",
        )
        rendered = trace.to_text()
        self.assertIn("Replanner短期记忆", rendered)
        self.assertIn("AnswerContext", rendered)

    def test_answer_context_does_not_forward_raw_session_history(self) -> None:
        raw_marker = "RAW_TOOL_TRACE_MUST_NOT_REACH_FINAL_PROMPT"
        loop_result = {
            "state": {"confirmed_constraints": {"cancer_type": "NSCLC"}},
            "prior_session_context": {"step_memories": [{"tool_trace": [raw_marker]}]},
            "answer_memory": {"claims": []},
            "replanner_long_memory": {},
            "replanner_short_memory": {},
        }
        summary = AnswerContextAgent(use_llm=False).summarize(query="q", loop_result=loop_result, safety_result={})
        self.assertNotIn(raw_marker, __import__("json").dumps(summary))

    def test_planner_forces_fetch_evidence_for_retrieval(self) -> None:
        llm = MockLLM([
            '{"patient_facts":{},"treatment_context":{"current_treatments":[],"prior_treatments":[]},"target_outcomes":[],"information_needs":[],"ambiguities":[]}',
            '{"query_type":"mechanism","search_query":"q","selected_tools":["hybrid_search"],"constraints":{}}',
        ])
        router = RetrievalRouter(retrieval_tools=object(), llm_client=llm, use_llm=True)  # type: ignore[arg-type]
        plan = router.plan("q")
        self.assertEqual(plan.selected_tools, ["hybrid_search", "fetch_evidence"])

    def test_planner_does_not_promote_candidate_hypotheses_to_patient_constraints(self) -> None:
        payload = {
            "query_type": "similar_case",
            "search_query": "EGFR L858R osimertinib resistance",
            "selected_tools": ["hybrid_search"],
            "constraints": {
                "cancer_type": "NSCLC",
                "gene_alterations": ["EGFR L858R", "MET amplification", "C797S"],
                "drugs": ["osimertinib", "amivantamab"],
                "responses": ["ORR"], "toxicities": [], "metastatic_sites": [], "ddi_terms": [],
            },
            "plan_rationale": "test", "targeted_information_gain": [],
            "avoid_repeating": [], "stop_if": [],
        }
        llm = MockLLM([
            '{"patient_facts":{"cancer_type":"NSCLC","gene_alterations":["EGFR L858R"]},"treatment_context":{"current_treatments":["osimertinib"],"prior_treatments":[]},"target_outcomes":["resistance"],"information_needs":[],"ambiguities":[]}',
            __import__("json").dumps(payload),
        ])
        router = RetrievalRouter(retrieval_tools=object(), llm_client=llm, use_llm=True)  # type: ignore[arg-type]

        plan = router.plan(
            "EGFR L858R NSCLC patient progressed after osimertinib",
            main_question="EGFR L858R NSCLC patient progressed after osimertinib",
        )

        self.assertIn("EGFR L858R", plan.constraints["gene_alterations"])
        self.assertNotIn("MET amplification", plan.constraints["gene_alterations"])
        self.assertNotIn("C797S", plan.constraints["gene_alterations"])
        self.assertNotIn("amivantamab", plan.constraints["drugs"])

    def test_query_understanding_filters_clinical_noise_from_gene_constraints(self) -> None:
        router = RetrievalRouter(retrieval_tools=object(), llm_client=None, use_llm=False)  # type: ignore[arg-type]
        constraints = router.extract_constraints("EGFR L858R NSCLC ECOG 1 after osimertinib PR with NA AI")
        self.assertIn("EGFR L858R", constraints["gene_alterations"])
        self.assertNotIn("ECOG", constraints["gene_alterations"])
        self.assertNotIn("AI", constraints["gene_alterations"])
        self.assertNotIn("NA", constraints["gene_alterations"])
        self.assertIn("PR", constraints["responses"])

    def test_query_understanding_comes_directly_from_llm(self) -> None:
        llm = MockLLM([
            __import__("json").dumps({
                "patient_facts": {
                    "cancer_type": "NSCLC",
                    "gene_alterations": ["EGFR L858R"],
                    "metastatic_sites": [],
                },
                "treatment_context": {"current_treatments": ["osimertinib"], "prior_treatments": []},
                "target_outcomes": ["resistance"],
                "information_needs": ["direct resistance evidence"],
                "ambiguities": [],
            }),
        ])
        router = RetrievalRouter(retrieval_tools=object(), llm_client=llm, use_llm=True)  # type: ignore[arg-type]
        understanding = router.understand_query("EGFR L858R osimertinib resistance")
        self.assertNotIn("query_type", understanding)
        self.assertIn("EGFR L858R", understanding["patient_facts"]["gene_alterations"])
        self.assertNotIn("hard_constraints", understanding)
        self.assertNotIn("expansion_hints", understanding)

    def test_router_splits_hard_constraints_and_soft_hints(self) -> None:
        router = RetrievalRouter(retrieval_tools=object(), llm_client=None, use_llm=False)  # type: ignore[arg-type]
        layers = router._extract_constraint_layers("EGFR L858R NSCLC after osimertinib resistance with PR and similar case")
        self.assertIn("EGFR L858R", layers["hard_constraints"]["gene_alterations"])
        self.assertIn("osimertinib", layers["soft_hints"]["retrieval_focus"][0])
        self.assertIn("PR", layers["hard_constraints"]["responses"])
        self.assertIn("broaden_to_resistance_mechanism", layers["expansion_hints"])
        self.assertIn("broaden_to_similar_case", layers["expansion_hints"])

    def test_router_run_includes_query_understanding(self) -> None:
        router = RetrievalRouter(retrieval_tools=StubRetrievalTools(), llm_client=None, use_llm=False)
        with self.assertRaisesRegex(RuntimeError, "rule fallback is disabled"):
            router.run("EGFR L858R osimertinib resistance")

    def test_replanner_expands_after_repeated_no_gain_observations(self) -> None:
        planner = MultiStepPlanningAgent(use_llm=False)
        plan = {
            "steps": [
                {"step_id": "S1", "goal": "direct evidence", "rerank_goal": "direct outcome", "evidence_lane": "direct_case", "success_criteria": ["one"], "attempt_budget": 2},
                {"step_id": "S2", "goal": "counter evidence", "rerank_goal": "opposite outcome", "evidence_lane": "direct_case", "success_criteria": ["one"], "attempt_budget": 2},
            ]
        }
        memories = [
            {"plan_step_id": "S1", "question_information_gain": {"new_chunk_ids": []}, "goal_evaluation": {"success_criteria_met": False}, "retrieval_actions": [{"query": f"q{i}"}]}
            for i in range(2)
        ]
        decision = planner.replan(
            main_question="q", plan=plan, active_step_index=0,
            recent_memories=memories, answer_memory={},
            base_query_type="similar_case", base_constraints={},
        )
        self.assertEqual(decision["action"], "retry_current_step")

    def test_replanner_retries_a_single_locally_exhausted_query(self) -> None:
        planner = MultiStepPlanningAgent(use_llm=False)
        plan = {
            "steps": [
                {"step_id": "S1", "goal": "goal one", "rerank_goal": "rank one", "evidence_lane": "direct_case", "success_criteria": ["one"], "attempt_budget": 2},
                {"step_id": "S2", "goal": "goal two", "rerank_goal": "rank two", "evidence_lane": "direct_case", "success_criteria": ["two"], "attempt_budget": 2},
            ]
        }
        memories = [
            {
                "plan_step_id": "S1",
                "question_information_gain": {"new_chunk_ids": []},
                "goal_evaluation": {"matched_goal_count": 0, "query_database_status": "exhausted", "success_criteria_met": False},
                "retrieval_actions": [{"query": "q1"}],
            }
        ]
        decision = planner.replan(
            main_question="q", plan=plan, active_step_index=0,
            recent_memories=memories, answer_memory={},
            base_query_type="similar_case", base_constraints={},
        )
        self.assertEqual(decision["action"], "retry_current_step")

    def test_replanner_reviews_completed_step_boundary_for_dynamic_plan(self) -> None:
        llm = MockLLM("{}")
        planner = MultiStepPlanningAgent(llm_client=llm, use_llm=True)
        plan = {
            "steps": [
                {"step_id": "S1", "goal": "goal one", "rerank_goal": "rank one", "evidence_lane": "direct_case", "success_criteria": ["one"], "attempt_budget": 2},
                {"step_id": "S2", "goal": "goal two", "rerank_goal": "rank two", "evidence_lane": "direct_case", "success_criteria": ["two"], "attempt_budget": 2},
            ]
        }
        decision = planner.replan(
            main_question="q",
            plan=plan,
            active_step_index=0,
            recent_memories=[{
                "plan_step_id": "S1",
                "question_information_gain": {"new_chunk_ids": []},
                "goal_evaluation": {
                    "completion_status": "sufficiently_met",
                    "recommended_stop": True,
                    "matched_goal_count": 2,
                },
                "retrieval_actions": [{"query": "q1"}],
            }],
            answer_memory={},
            base_query_type="similar_case",
            base_constraints={},
        )
        self.assertEqual(decision["action"], "advance_to_next_step")
        self.assertEqual(llm.calls, 1)

    def test_no_citations_skips_llm_and_returns_evidence_limited_answer(self) -> None:
        llm = MockLLM("PMID: 12345678 invented")
        answer = AnswerGenerator(llm_client=llm, use_llm=True).generate(
            query="q",
            loop_result={
                "query_type": "mechanism",
                "state": {"retrieved_evidence": [], "stop_condition": "no_new_evidence"},
                "loop_steps": [],
                "citations": {"claim_citations": []},
            },
            safety_result={"risk_level": "high", "safety_boundary": "boundary"},
        )
        self.assertEqual(llm.calls, 0)
        self.assertNotIn("12345678", answer)

    def test_fallback_answer_does_not_embed_or_require_raw_prior_context(self) -> None:
        marker = "RAW_PRIOR_SESSION"
        answer = AnswerGenerator(use_llm=False).generate(
            query="q",
            loop_result={"query_type": "similar_case", "state": {}, "loop_steps": [], "citations": {}, "prior_session_context": {"step_memories": [marker]}},
            safety_result={},
        )
        self.assertNotIn(marker, answer)

    def test_unverified_llm_pmid_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AnswerGenerator._validate_answer_citations(
                "引用 PMID: 99999999", allowed_ids={"1#chunk-000001"}, allowed_pmids={"11111111"}
            )

    def test_rerank_uses_one_small_budgeted_llm_call(self) -> None:
        llm = MockLLM('{"rankings":[{"id":"h1","evidence_type":"primary_result","population_match":3,"intervention_match":3,"outcome_match":3,"directness":3,"attribution":3,"reason":"best"}]}')
        reranker = LLMReranker(llm)
        original_budget = LLMReranker.MAX_RERANK_INPUT_CHARS
        original_candidate_chars = LLMReranker.CANDIDATE_MAX_CHARS
        try:
            LLMReranker.MAX_RERANK_INPUT_CHARS = 260
            LLMReranker.CANDIDATE_MAX_CHARS = 220
            hits = [
                SearchHit(id=f"h{i}", score=1.0 - i * 0.1, source="dense", text=" ".join(["evidence"] * 35), metadata={})
                for i in range(1, 4)
            ]
            ranked = reranker.rerank(query="EGFR L858R osimertinib", hits=hits, top_k=3)
            statuses = [hit.metadata.get("llm_rerank_status") for hit in ranked]
            self.assertIn("completed", statuses)
            self.assertNotIn("deferred_budget", statuses)
            self.assertLess(len(ranked), len(hits))
            self.assertEqual(llm.calls, 1)
        finally:
            LLMReranker.MAX_RERANK_INPUT_CHARS = original_budget
            LLMReranker.CANDIDATE_MAX_CHARS = original_candidate_chars


if __name__ == "__main__":
    unittest.main()
