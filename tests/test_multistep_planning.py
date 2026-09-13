import unittest
import json

from searchagent_retrieval.planning import (
    EXPANSION_STRATEGIES,
    REPLAN_SCHEMA,
    MultiStepPlanningAgent,
)


class _OverPlanningLLM:
    def call_function(self, **kwargs):
        return {
            "plan_rationale": "research outline",
            "initial_total_budget": 18,
            "steps": [
                {
                    "step_id": f"S{index}",
                    "goal": f"specific candidate branch {index}",
                    "rerank_goal": f"rank branch {index}",
                    "evidence_lane": "direct_case",
                    "success_criteria": ["one hit"],
                    "attempt_budget": 2,
                }
                for index in range(1, 10)
            ],
        }


class _BudgetChoosingLLM:
    def call_function(self, **kwargs):
        return {
            "plan_rationale": "model-selected budget",
            "initial_total_budget": 3,
            "steps": [
                {"step_id": "S1", "goal": "direct evidence", "rerank_goal": "direct", "evidence_lane": "direct_case", "success_criteria": ["hit"], "attempt_budget": 2},
                {"step_id": "S2", "goal": "counter evidence", "rerank_goal": "counter", "evidence_lane": "direct_case", "success_criteria": ["hit"], "attempt_budget": 1},
            ],
        }


class _CapturingReplannerLLM:
    def __init__(self):
        self.kwargs = {}

    def call_function(self, **kwargs):
        self.kwargs = kwargs
        return {
            "action": "retry_current_step",
            "active_step_id": "S1",
            "search_query": "EGFR突变肺腺癌 一代TKI后脑转移 奥希替尼 颅内反应",
            "rerank_goal": "优先评估6-12周颅内ORR/DCR和症状改善",
            "selected_tools": ["dense_search", "bm25_search", "fetch_evidence"],
            "constraints": {
                "cancer_type": "NSCLC", "gene_alterations": ["EGFR 19del"],
                "drugs": ["osimertinib"], "responses": [], "toxicities": [],
                "metastatic_sites": ["brain"], "ddi_terms": [],
            },
            "expansion": {"strategy": "none", "source": "", "added_terms": [], "relation_type": ""},
            "plan_changes": [], "avoid_repeating": ["AURA3 intracranial ORR MRI 6 weeks"],
            "decision_rationale": "broaden for the local case corpus",
            "requested_budget_extension": 0, "budget_extension_reason": "",
        }


class _CombinedPlanningLLM:
    def __init__(self):
        self.calls = 0
        self.function_name = ""

    def call_function(self, **kwargs):
        self.calls += 1
        self.function_name = kwargs["function_name"]
        return {
            "query_type": "mutation_drug",
            "hard_constraints": {
                "cancer_type": "NSCLC",
                "gene_alterations": ["EGFR L858R"],
                "drugs": ["osimertinib"],
                "responses": [],
                "toxicities": [],
                "metastatic_sites": [],
                "ddi_terms": [],
            },
            "soft_hints": {"clinical_terms": ["progression"], "retrieval_focus": ["response"]},
            "expansion_hints": [],
            "ambiguity_notes": [],
            "plan_rationale": "direct evidence first",
            "initial_total_budget": 2,
            "steps": [
                {"step_id": "S1", "goal": "direct response", "rerank_goal": "direct outcome", "evidence_lane": "direct_case", "success_criteria": ["one"], "attempt_budget": 1},
                {"step_id": "S2", "goal": "counter evidence", "rerank_goal": "progression", "evidence_lane": "direct_case", "success_criteria": ["one"], "attempt_budget": 1},
            ],
            "first_execution": {
                "search_query": "EGFR L858R osimertinib progression response",
                "rerank_goal": "direct post-progression outcome",
                "selected_tools": ["dense_search", "bm25_search"],
            },
        }


class MultiStepPlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = MultiStepPlanningAgent(use_llm=False)
        self.constraints = {
            "cancer_type": "NSCLC",
            "gene_alterations": ["KRAS G12C"],
            "drugs": ["KRAZATI"],
            "responses": [],
            "toxicities": [],
            "metastatic_sites": [],
            "ddi_terms": [],
        }

    @staticmethod
    def _plan() -> dict:
        return {
            "plan_rationale": "test fixture",
            "initial_total_budget": 4,
            "steps": [
                {"step_id": "S1", "goal": "direct evidence", "rerank_goal": "direct outcome", "evidence_lane": "direct_case", "success_criteria": ["one"], "attempt_budget": 2},
                {"step_id": "S2", "goal": "counter evidence", "rerank_goal": "opposite outcome", "evidence_lane": "direct_case", "success_criteria": ["one"], "attempt_budget": 2},
            ],
        }

    def test_initial_plan_has_no_rule_fallback(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "rule fallback is disabled"):
            self.agent.create_plan("question", None, self.constraints, {})

    def test_replan_advances_only_after_current_step_succeeds(self) -> None:
        plan = self._plan()
        memory = {"plan_step_id": "S1", "goal_evaluation": {"success_criteria_met": True}, "question_information_gain": {"new_chunk_ids": ["c1"]}}
        decision = self.agent.replan(main_question="question", plan=plan, active_step_index=0, recent_memories=[memory], answer_memory={}, base_query_type="treatment_advice", base_constraints=self.constraints)
        self.assertEqual(decision["action"], "advance_to_next_step")

    def test_replan_expands_with_aliases_after_two_no_gain_rounds(self) -> None:
        plan = self._plan()
        memories = [
            {"plan_step_id": "S1", "goal_evaluation": {"success_criteria_met": False}, "question_information_gain": {"new_chunk_ids": []}, "retrieval_actions": [{"query": "q1"}]},
            {"plan_step_id": "S1", "goal_evaluation": {"success_criteria_met": False}, "question_information_gain": {"new_chunk_ids": []}, "retrieval_actions": [{"query": "q2"}]},
        ]
        decision = self.agent.replan(main_question="question", plan=plan, active_step_index=0, recent_memories=memories, answer_memory={}, base_query_type="treatment_advice", base_constraints=self.constraints)
        self.assertEqual(decision["action"], "retry_current_step")
        self.assertEqual(decision["expansion"]["strategy"], "none")
        self.assertEqual(decision["expansion"]["added_terms"], [])

    def test_one_exhausted_query_is_rewritten_not_treated_as_database_absence(self) -> None:
        plan = self._plan()
        memory = {
            "plan_step_id": "S1",
            "goal_evaluation": {
                "success_criteria_met": False,
                "query_database_status": "exhausted",
                "queries_attempted": ["old exact query"],
                "recommended_query_change": "broader cohort query",
            },
            "question_information_gain": {"new_chunk_ids": []},
            "retrieval_actions": [{"query": "old exact query"}],
        }
        decision = self.agent.replan(main_question="question", plan=plan, active_step_index=0, recent_memories=[memory], answer_memory={}, base_query_type="treatment_advice", base_constraints=self.constraints)
        self.assertEqual(decision["action"], "retry_current_step")
        # The legacy execution label no longer proves corpus absence.  A local
        # query failure must be rewritten and retried.
        memory["goal_evaluation"]["query_direction_status"] = "locally_exhausted"
        decision = self.agent.replan(main_question="question", plan=plan, active_step_index=0, recent_memories=[memory], answer_memory={}, base_query_type="treatment_advice", base_constraints=self.constraints)
        self.assertTrue(decision["search_query"].startswith("broader cohort query"))
        self.assertNotEqual(decision["search_query"], "old exact query")
        self.assertIn("old exact query", decision["avoid_repeating"])

    def test_multiple_queries_and_complementary_tools_can_support_absence(self) -> None:
        plan = self._plan()
        memories = [
            {
                "plan_step_id": "S1",
                "goal_evaluation": {"success_criteria_met": False, "matched_goal_count": 0},
                "question_information_gain": {"new_chunk_ids": []},
                "retrieval_actions": [{"query": "q1"}],
                "micro_retrieval_ledger": [{"tool": "dense_search", "query": "q1", "hit_count": 0}],
            },
            {
                "plan_step_id": "S1",
                "goal_evaluation": {"success_criteria_met": False, "matched_goal_count": 0},
                "question_information_gain": {"new_chunk_ids": []},
                "retrieval_actions": [{"query": "q2"}],
                "micro_retrieval_ledger": [{"tool": "bm25_search", "query": "q2", "hit_count": 0}],
            },
        ]
        decision = self.agent.replan(main_question="question", plan=plan, active_step_index=0, recent_memories=memories, answer_memory={}, base_query_type="treatment_advice", base_constraints=self.constraints)
        self.assertEqual(decision["action"], "advance_to_next_step")
        self.assertEqual(memories[-1]["goal_evaluation"]["database_coverage_status"], "absence_supported")

    def test_initial_planner_rejects_nine_step_research_outline(self) -> None:
        agent = MultiStepPlanningAgent(llm_client=_OverPlanningLLM(), use_llm=True)

        with self.assertRaisesRegex(ValueError, "too many steps"):
            agent.create_plan("question", "similar_case", self.constraints, {})

    def test_cli_initial_budget_is_hint_not_model_minimum(self) -> None:
        agent = MultiStepPlanningAgent(llm_client=_BudgetChoosingLLM(), use_llm=True)

        plan = agent.create_plan(
            "question", "similar_case", self.constraints, {},
            initial_total_budget=12, hard_total_budget=128,
        )

        self.assertEqual(plan["initial_total_budget"], 3)
        self.assertEqual([step["attempt_budget"] for step in plan["steps"]], [2, 1])

    def test_replan_strategy_schema_matches_implemented_expansions(self) -> None:
        strategy_schema = REPLAN_SCHEMA["properties"]["expansion"]["properties"]["strategy"]
        self.assertEqual(set(strategy_schema["enum"]), set(EXPANSION_STRATEGIES))
        self.assertIn("case_ddi", strategy_schema["enum"])
        self.assertIn("mutation_drug", strategy_schema["enum"])
        self.assertNotIn("therapeutic_class", strategy_schema["enum"])

    def test_mutation_drug_expansion_returns_mapped_drugs(self) -> None:
        terms = self.agent._drug_knowledge_terms(
            "mutation_drug",
            [],
            ["AKT1 E17K"],
        )
        self.assertIn("AKT1 E17K", terms)
        self.assertIn("anlotinib", terms)

    def test_replanner_prompt_is_grounded_in_local_case_database(self) -> None:
        llm = _CapturingReplannerLLM()
        agent = MultiStepPlanningAgent(llm_client=llm, use_llm=True)
        plan = {
            "steps": [{
                "step_id": "S1", "goal": "early CNS response",
                "rerank_goal": "6-12 week intracranial response",
                "evidence_lane": "direct_case", "success_criteria": ["direct evidence"],
                "attempt_budget": 3,
            }]
        }
        memory = {
            "plan_step_id": "S1",
            "question_information_gain": {"new_chunk_ids": []},
            "goal_evaluation": {
                "completion_status": "not_met", "query_database_status": "more_available",
                "matched_goal_count": 0,
                "critical_gaps": ["early intracranial outcome"],
            },
            "retrieval_actions": [{"query": "AURA3 intracranial ORR MRI 6 weeks"}],
        }
        decision = agent.replan(
            main_question="EGFR 19del NSCLC after icotinib with brain progression",
            plan=plan, active_step_index=0, recent_memories=[memory], answer_memory={},
            base_query_type="similar_case", base_constraints=self.constraints,
        )

        prompt = json.loads(llm.kwargs["user"])
        self.assertIn("case-report corpus", prompt["database_profile"]["primary_content"])
        self.assertIn("remove at least two", prompt["query_contract"]["zero_match_change"])
        self.assertIn("not a comprehensive PubMed/trial registry", prompt["database_profile"]["primary_content"])
        self.assertIn("不得仅凭模型记忆强制加入", llm.kwargs["system"])
        self.assertNotIn("AURA3", decision["search_query"])

    def test_replanner_consumes_retrieval_state_and_micro_diagnostics(self) -> None:
        llm = _CapturingReplannerLLM()
        agent = MultiStepPlanningAgent(llm_client=llm, use_llm=True)
        plan = {"steps": [{
            "step_id": "S1", "goal": "direct response", "rerank_goal": "early response",
            "evidence_lane": "direct_case", "success_criteria": ["direct evidence"], "attempt_budget": 3,
        }]}
        memory = {
            "plan_step_id": "S1", "round": 1,
            "retrieval_state": {
                "search_attempt_status": "no_hits", "query_direction_status": "locally_exhausted",
                "database_coverage_status": "unknown", "tools_attempted": ["dense_search"],
                "distinct_queries_attempted": ["old query"], "tool_error_count": 0,
            },
            "goal_evaluation": {
                "completion_status": "not_met", "matched_goal_count": 0,
                "search_attempt_status": "no_hits", "query_direction_status": "locally_exhausted",
                "database_coverage_status": "unknown", "recommended_query_change": "broader query",
            },
            "question_information_gain": {"new_chunk_ids": []},
            "retrieval_actions": [{"query": "old query", "selected_tools": ["dense_search"]}],
            "micro_retrieval_ledger": [{"tool": "dense_search", "query": "old query", "hit_count": 0}],
        }
        agent.replan(
            main_question="question", plan=plan, active_step_index=0, recent_memories=[memory],
            answer_memory={}, base_query_type="similar_case", base_constraints=self.constraints,
        )
        prompt = json.loads(llm.kwargs["user"])
        self.assertEqual(prompt["retrieval_state"]["database_coverage_status"], "unknown")
        latest = prompt["latest_step_memory"][-1]
        self.assertEqual(latest["retrieval_state"]["query_direction_status"], "locally_exhausted")
        self.assertEqual(latest["retrieval_diagnostics"][0]["hit_count"], 0)


if __name__ == "__main__":
    unittest.main()
