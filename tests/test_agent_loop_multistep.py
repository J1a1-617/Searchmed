import json
import unittest

from searchagent_retrieval.agent_loop import EvidenceReviewAgent, MainAgentLoop
from searchagent_retrieval.router import RetrievalRouter, RoutePlan


class _SafetyGate:
    def evaluate(self, **kwargs):
        return {"risk_level": "low", "issues": [], "claim_reviews": []}


class _Router:
    llm_client = None
    use_llm = False

    def __init__(self):
        self.calls = []
        self.safety_gate = _SafetyGate()

    def plan(self, query, **kwargs):
        return RoutePlan("similar_case", ["bm25_search", "fetch_evidence"], {"drugs": ["drug-a"]})

    def run(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        plan = kwargs["route_plan"]
        return {
            "query_type": plan.query_type,
            "selected_tools": plan.selected_tools,
            "constraints": plan.constraints,
            "planner_metadata": plan.planner_metadata,
            "rerank_goal": kwargs["rerank_goal"],
            "results": {"fetch_evidence": []},
            "evidence_layering": {"query": kwargs["rerank_goal"], "layer_distribution": [], "assessed_evidence": []},
            "contradiction_check": {},
        }


class _ScriptedPlanningAgent:
    def __init__(self):
        self.calls = 0

    def create_plan(self, **kwargs):
        return {"plan_rationale": "two atomic goals", "steps": [
            {"step_id": "S1", "goal": "goal one", "rerank_goal": "rank only goal one", "evidence_lane": "direct_case", "success_criteria": ["one"]},
            {"step_id": "S2", "goal": "goal two", "rerank_goal": "rank only goal two", "evidence_lane": "ddi_safety", "success_criteria": ["two"]},
        ]}

    def replan(self, **kwargs):
        self.calls += 1
        index = kwargs["active_step_index"]
        if self.calls == 2:
            return {"action": "advance_to_next_step"}
        return {
            "action": "retry_current_step",
            "active_step_id": f"S{index + 1}",
            "search_query": f"query-for-step-{index + 1}",
            "rerank_goal": f"rank only goal {'one' if index == 0 else 'two'}",
            "selected_tools": ["bm25_search", "fetch_evidence"],
            "constraints": {},
            "expansion": {"strategy": "none", "source": "", "added_terms": [], "relation_type": ""},
            "plan_changes": [],
            "avoid_repeating": [],
            "decision_rationale": "scripted",
        }


class _ExecutionAgent:
    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        plan = kwargs["route_plan"]
        return {
            "query_type": plan.query_type,
            "selected_tools": ["bm25_search", "fetch_evidence"],
            "constraints": plan.constraints,
            "planner_metadata": {"execution_mode": "function_tool_loop"},
            "rerank_goal": kwargs["rerank_goal"],
            "results": {"fetch_evidence": []},
            "evidence_layering": {"query": kwargs["rerank_goal"], "layer_distribution": [], "assessed_evidence": []},
            "contradiction_check": {},
            "step_execution_report": {"execution_status": "no_evidence"},
            "tool_trace": [{"tool_name": "bm25_search"}],
        }


class _RemainingBudget:
    def __init__(self, remaining):
        self.remaining = remaining

    def calls_remaining(self):
        return self.remaining


class _BudgetPlanningAgent:
    def __init__(self, *, extend_at_boundary=False):
        self.extend_at_boundary = extend_at_boundary

    def create_plan(self, **kwargs):
        if self.extend_at_boundary:
            return {"plan_rationale": "extension test", "initial_total_budget": 1, "steps": [
                {"step_id": "S1", "goal": "goal one", "rerank_goal": "rank one", "attempt_budget": 3}
            ]}
        return {"plan_rationale": "step budget test", "initial_total_budget": 2, "steps": [
            {"step_id": "S1", "goal": "goal one", "rerank_goal": "rank one", "attempt_budget": 1},
            {"step_id": "S2", "goal": "goal two", "rerank_goal": "rank two", "attempt_budget": 1},
        ]}

    def replan(self, **kwargs):
        budget = kwargs.get("budget_state") or {}
        extension = 2 if self.extend_at_boundary and budget.get("rounds_remaining") == 0 else 0
        index = kwargs["active_step_index"]
        return {
            "action": "retry_current_step",
            "search_query": f"budget-query-{index + 1}",
            "rerank_goal": f"budget-rank-{index + 1}",
            "selected_tools": ["bm25_search", "fetch_evidence"],
            "constraints": {},
            "requested_budget_extension": extension,
            "budget_extension_reason": "finish high-value active step" if extension else "",
        }


class _SingleStepPlanningAgent:
    def create_plan(self, **kwargs):
        return {"plan_rationale": "single step", "initial_total_budget": 1, "steps": [
            {"step_id": "S1", "goal": "goal one", "rerank_goal": "rank one", "evidence_lane": "direct_case", "success_criteria": ["one"], "attempt_budget": 1},
        ]}

    def replan(self, **kwargs):
        return {
            "action": "retry_current_step",
            "active_step_id": "S1",
            "search_query": "q1",
            "rerank_goal": "rank one",
            "selected_tools": ["bm25_search", "fetch_evidence"],
            "constraints": {"drugs": ["hallucinated-drug"]},
            "expansion": {"strategy": "none", "source": "", "added_terms": [], "relation_type": ""},
            "plan_changes": [],
            "avoid_repeating": [],
            "decision_rationale": "scripted",
        }


class _SuccessRoundMemory:
    def maintain(self, **kwargs):
        step = kwargs.get("plan_step") or {}
        return {
            "round": kwargs["round_number"],
            "plan_step_id": step.get("step_id"),
            "accepted_evidence": [],
            "question_information_gain": {"new_chunk_ids": []},
            "goal_evaluation": {"success_criteria_met": True, "completion_status": "sufficiently_met", "recommended_stop": True},
            "retrieval_actions": [],
            "tool_trace": [],
        }


class _CombinedInitialPlanningAgent:
    def __init__(self):
        self.replan_calls = 0

    def create_plan(self, **kwargs):
        return {
            "plan_rationale": "direct",
            "initial_total_budget": 1,
            "steps": [{
                "step_id": "S1", "goal": "direct outcome", "rerank_goal": "direct response",
                "evidence_lane": "direct_case", "success_criteria": ["one"],
                "attempt_budget": 1,
            }],
        }

    def replan(self, **kwargs):
        self.replan_calls += 1
        raise AssertionError("The first attempt must not call Replanner")


class AgentLoopMultiStepTests(unittest.TestCase):
    def test_separate_initial_plan_executes_first_round_without_replanner(self):
        class _UnderstandingLLM:
            def chat(self, **kwargs):
                return json.dumps({
                    "query_type": "mutation_drug",
                    "hard_constraints": {
                        "cancer_type": "NSCLC", "gene_alterations": ["EGFR L858R"],
                        "drugs": ["osimertinib"], "responses": [], "toxicities": [],
                        "metastatic_sites": [], "ddi_terms": [],
                    },
                    "soft_hints": {"clinical_terms": [], "retrieval_focus": []},
                    "expansion_hints": [], "ambiguity_notes": [],
                })

        planning = _CombinedInitialPlanningAgent()
        executor = _ExecutionAgent()
        router = RetrievalRouter(retrieval_tools=object(), llm_client=_UnderstandingLLM(), use_llm=True)  # type: ignore[arg-type]
        loop = MainAgentLoop(
            router=router,
            evidence_review_agent=EvidenceReviewAgent(use_llm=False),
            planning_agent=planning,
            execution_agent=executor,
            max_steps=1,
            max_total_steps=1,
        )

        result = loop.run("EGFR L858R NSCLC progressed after osimertinib")

        self.assertEqual(planning.replan_calls, 0)
        self.assertEqual(executor.calls[0]["route_plan"].search_query, "direct outcome")
        self.assertEqual(result["replan_decisions"][0]["decision_mode"], "direct_plan_execution")

    def test_each_plan_step_uses_its_own_query_and_rerank_goal(self):
        router = _Router()
        loop = MainAgentLoop(
            router=router,
            evidence_review_agent=EvidenceReviewAgent(use_llm=False),
            planning_agent=_ScriptedPlanningAgent(),
            max_steps=2,
        )

        result = loop.run("main question")

        self.assertEqual([call["query"] for call in router.calls], ["query-for-step-1", "query-for-step-2"])
        self.assertEqual([call["rerank_goal"] for call in router.calls], ["rank only goal one", "rank only goal two"])
        self.assertEqual([row["plan_step_id"] for row in result["round_memories"]], ["S1", "S2"])
        self.assertEqual([row["replan_action"] for row in result["loop_steps"]], ["retry_current_step", "retry_current_step"])

    def test_main_loop_delegates_atomic_step_to_execution_agent(self):
        router = _Router()
        executor = _ExecutionAgent()
        loop = MainAgentLoop(
            router=router,
            evidence_review_agent=EvidenceReviewAgent(use_llm=False),
            planning_agent=_ScriptedPlanningAgent(),
            execution_agent=executor,
            max_steps=1,
        )

        result = loop.run("main question")

        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(router.calls, [])
        self.assertEqual(result["round_memories"][0]["tool_trace"][0]["tool_name"], "bm25_search")

    def test_main_loop_preserves_final_call_reserve_before_starting_round(self):
        router = _Router()
        router.llm_client = _RemainingBudget(10)
        executor = _ExecutionAgent()
        loop = MainAgentLoop(
            router=router,
            evidence_review_agent=EvidenceReviewAgent(use_llm=False),
            planning_agent=_ScriptedPlanningAgent(),
            execution_agent=executor,
            max_steps=2,
            final_llm_call_reserve=3,
            max_llm_calls_per_round=8,
        )
        result = loop.run("main question")
        self.assertEqual(executor.calls, [])
        self.assertEqual(result["state"]["stop_condition"], "llm_budget_reserved_for_final_generation")
        self.assertEqual(result["budget_events"][0]["calls_remaining"], 10)

    def test_ddi_expansion_policy_exposes_multiple_related_strategies_in_one_round(self):
        class _DdiPlanningAgent:
            def create_plan(self, **kwargs):
                return {"plan_rationale": "ddi step", "initial_total_budget": 1, "steps": [
                    {"step_id": "S1", "goal": "ddi goal", "rerank_goal": "ddi rank", "evidence_lane": "ddi_safety", "success_criteria": ["one"], "attempt_budget": 1},
                ]}

            def replan(self, **kwargs):
                return {
                    "action": "expand_current_step",
                    "active_step_id": "S1",
                    "search_query": "osimertinib voriconazole interaction",
                    "rerank_goal": "ddi rank",
                    "selected_tools": ["bm25_search", "fetch_evidence"],
                    "constraints": {},
                    "expansion": {"strategy": "case_ddi", "source": "case_ddi.json", "added_terms": [], "relation_type": "direct_case_ddi_relation"},
                    "plan_changes": [],
                    "avoid_repeating": [],
                    "decision_rationale": "scripted",
                }

        router = _Router()
        executor = _ExecutionAgent()
        loop = MainAgentLoop(
            router=router,
            evidence_review_agent=EvidenceReviewAgent(use_llm=False),
            planning_agent=_DdiPlanningAgent(),
            execution_agent=executor,
            max_steps=1,
        )

        loop.run("main question")

        expansion = executor.calls[0]["expansion"]
        self.assertEqual(expansion["allowed_strategies"], ["drug_alias", "case_ddi", "pk_relation", "ddi_rule"])

    def test_per_step_budget_prevents_one_step_from_consuming_all_rounds(self):
        router = _Router()
        loop = MainAgentLoop(
            router=router,
            evidence_review_agent=EvidenceReviewAgent(use_llm=False),
            planning_agent=_BudgetPlanningAgent(),
            max_steps=2,
        )

        result = loop.run("main question")

        self.assertEqual([row["plan_step_id"] for row in result["loop_steps"]], ["S1", "S2"])
        self.assertEqual(result["step_budget_status"]["S1"]["status"], "budget_exhausted_partial")
        self.assertTrue(any(event["event"] == "step_budget_exhausted" for event in result["budget_events"]))

    def test_total_budget_boundary_allows_replanner_extension(self):
        router = _Router()
        loop = MainAgentLoop(
            router=router,
            evidence_review_agent=EvidenceReviewAgent(use_llm=False),
            planning_agent=_BudgetPlanningAgent(extend_at_boundary=True),
            max_steps=1,
            max_total_steps=4,
            max_budget_extension=2,
        )

        result = loop.run("main question")

        self.assertEqual(result["budget_state"]["initial_total_budget"], 1)
        self.assertEqual(result["budget_state"]["final_total_budget"], 3)
        self.assertEqual(result["budget_state"]["rounds_used"], 3)
        self.assertTrue(any(event["event"] == "agent_budget_extension" for event in result["budget_events"]))
        self.assertEqual(len(result["round_memories"]), 3)
        self.assertEqual(result["memory_timeline"]["current_agent_round"], 4)
        self.assertEqual(result["memory_timeline"]["compacted_rounds"], [1])
        self.assertEqual(result["memory_timeline"]["recent_full_rounds"], [2, 3])
        self.assertEqual(result["replanner_memory_events"][0]["covered_rounds"], [1])
        self.assertEqual(result["replanner_memory_events"][0]["recent_full_rounds"], [2, 3])

    def test_sufficient_step_advances_without_extra_micro_batch(self):
        router = _Router()
        loop = MainAgentLoop(
            router=router,
            evidence_review_agent=EvidenceReviewAgent(use_llm=False),
            round_memory_agent=_SuccessRoundMemory(),
            planning_agent=_BudgetPlanningAgent(),
            max_steps=4,
        )

        result = loop.run("main question")

        self.assertEqual([row["plan_step_id"] for row in result["loop_steps"]], ["S1", "S2"])
        self.assertEqual(result["step_budget_status"]["S1"]["status"], "completed")
        self.assertEqual(result["step_budget_status"]["S2"]["status"], "completed")
        self.assertFalse(any(event["event"] == "step_budget_exhausted" for event in result["budget_events"]))

    def test_replanner_can_insert_and_execute_new_step_at_boundary(self):
        class _DynamicPlanningAgent:
            def create_plan(self, **kwargs):
                return {"plan_rationale": "minimal skeleton", "initial_total_budget": 2, "steps": [{
                    "step_id": "S1", "goal": "direct outcome", "rerank_goal": "direct",
                    "evidence_lane": "direct_case", "success_criteria": ["hit"], "attempt_budget": 1,
                }]}

            def replan(self, **kwargs):
                index = kwargs["active_step_index"]
                memories = kwargs.get("recent_memories") or []
                active_id = (kwargs["plan"]["steps"])[index]["step_id"]
                active_memories = [m for m in memories if m.get("plan_step_id") == active_id]
                base = {
                    "active_step_id": active_id, "selected_tools": ["bm25_search", "fetch_evidence"],
                    "constraints": {}, "expansion": {"strategy": "none", "source": "", "added_terms": [], "relation_type": ""},
                    "avoid_repeating": [], "requested_budget_extension": 0, "budget_extension_reason": "",
                }
                if active_id == "S1" and active_memories:
                    return {**base, "action": "revise_plan", "search_query": "new safety branch",
                            "rerank_goal": "target-drug safety", "decision_rationale": "new high-value gap",
                            "plan_changes": [{
                                "step_id": "S_NEW", "goal": "target-drug safety evidence",
                                "rerank_goal": "same-drug hepatotoxicity only", "evidence_lane": "ddi_safety",
                                "success_criteria": ["same-drug safety evidence"], "attempt_budget": 1,
                            }]}
                return {**base, "action": "retry_current_step", "search_query": active_id,
                        "rerank_goal": active_id, "decision_rationale": "execute", "plan_changes": []}

        loop = MainAgentLoop(
            router=_Router(), evidence_review_agent=EvidenceReviewAgent(use_llm=False),
            round_memory_agent=_SuccessRoundMemory(), planning_agent=_DynamicPlanningAgent(),
            max_steps=2, max_total_steps=3, max_budget_extension=1,
        )
        result = loop.run("main question")

        self.assertEqual([row["plan_step_id"] for row in result["loop_steps"]], ["S1", "S_NEW"], result)
        self.assertTrue(any(event["event"] == "replanner_steps_added" for event in result["budget_events"]))
        self.assertEqual([step["step_id"] for step in result["retrieval_plan"]["steps"]], ["S1", "S_NEW"])

    def test_plan_completion_is_not_reported_as_budget_exhaustion(self):
        router = _Router()
        loop = MainAgentLoop(
            router=router,
            evidence_review_agent=EvidenceReviewAgent(use_llm=False),
            round_memory_agent=_SuccessRoundMemory(),
            planning_agent=_SingleStepPlanningAgent(),
            max_steps=1,
        )

        result = loop.run("main question")

        self.assertEqual(result["state"]["stop_condition"], "retrieval_plan_completed")
        self.assertEqual(result["budget_state"]["rounds_used"], 1)
        self.assertEqual(result["step_budget_status"]["S1"]["status"], "completed")
        self.assertEqual(router.calls[0]["route_plan"].constraints, {"drugs": ["drug-a"]})
        self.assertEqual(result["state"]["confirmed_constraints"], {"drugs": ["drug-a"]})


if __name__ == "__main__":
    unittest.main()
