import unittest

from searchagent_retrieval.agent_loop import AnswerMemoryAgent, CitationAgent, ReplannerMemoryAgent, RoundMemoryAgent, StepMemoryAgent


class MemoryArchitectureTests(unittest.TestCase):
    def test_micro_ledger_expands_two_stage_batch_diagnostics(self) -> None:
        route = {
            "tool_trace": [{
                "tool_name": "execute_retrieval_batch",
                "arguments": {"actions": [
                    {"tool": "dense_search", "query": "semantic query", "top_k": 20, "spaces": ["case_semantic"]},
                    {"tool": "bm25_search", "query": "EGFR L858R LMD osimertinib", "top_k": 16},
                ]},
                "result": {"action_results": [
                    {"hit_count": 20, "known_candidate_count": 20, "hits": [{"id": "d1"}]},
                    {"hit_count": 8, "known_candidate_count": 25, "hits": [{"id": "b1"}]},
                ]},
            }],
        }
        ledger = StepMemoryAgent.build_micro_ledger(route)
        self.assertEqual([row["tool"] for row in ledger], ["dense_search", "bm25_search"])
        self.assertEqual([row["query"] for row in ledger], ["semantic query", "EGFR L858R LMD osimertinib"])
        self.assertEqual([row["hit_count"] for row in ledger], [20, 8])

    def test_replanner_memory_consolidates_after_configured_batches(self) -> None:
        response = '{"short_memory":{"active_step_id":"S2","recent_information_gain":["new"],"recent_effective_strategies":[],"recent_failed_strategies":[],"current_critical_gaps":[],"recent_conflicts":[],"avoid_repeating":[]},"long_memory":{"step_progress":["S1 done"],"global_key_findings":[],"global_critical_gaps":[],"cross_step_constraints":[],"unresolved_conflicts":[],"search_strategy_lessons":[],"protected_safety_information":[],"evidence_ids":["c1"]}}'
        llm = self.MockLLM(response)
        agent = ReplannerMemoryAgent(llm_client=llm, consolidate_every=3, token_threshold=100000)
        pending = [{"round": n, "plan_step_id": "S2"} for n in range(1, 4)]
        result = agent.maintain(main_question="q", plan={"steps": []}, active_step_id="S2", short_memory={}, long_memory={}, pending_step_memories=pending)
        self.assertTrue(result["consolidated"])
        self.assertEqual(result["consumed_count"], 3)
        self.assertEqual(result["long_memory"]["evidence_ids"], ["c1"])

    def test_replanner_memory_consolidates_early_when_token_threshold_is_exceeded(self) -> None:
        response = '{"short_memory":{"active_step_id":"S1","recent_information_gain":[],"recent_effective_strategies":[],"recent_failed_strategies":[],"current_critical_gaps":[],"recent_conflicts":[],"avoid_repeating":[]},"long_memory":{"step_progress":[],"global_key_findings":[],"global_critical_gaps":[],"cross_step_constraints":[],"unresolved_conflicts":[],"search_strategy_lessons":[],"protected_safety_information":[],"evidence_ids":[]}}'
        llm = self.MockLLM(response)
        agent = ReplannerMemoryAgent(llm_client=llm, consolidate_every=10, token_threshold=10)
        result = agent.maintain(main_question="q", plan={"steps": []}, active_step_id="S1", short_memory={}, long_memory={}, pending_step_memories=[{"large": "x" * 1000}])
        self.assertTrue(result["consolidated"])

    def test_replanner_memory_uses_rule_compaction_when_pending_is_too_large(self) -> None:
        llm = self.MockLLM('{"short_memory":{},"long_memory":{}}')
        agent = ReplannerMemoryAgent(llm_client=llm, consolidate_every=10, token_threshold=256)
        result = agent.maintain(
            main_question="q",
            plan={"steps": []},
            active_step_id="S1",
            short_memory={},
            long_memory={},
            pending_step_memories=[
                {"round": 1, "plan_step_id": "S1", "question_information_gain": {"score": 0.9, "new_chunk_ids": ["c1"]}, "goal_evaluation": {"critical_gaps": ["gap1"], "query_database_status": "exhausted", "recommended_query_change": "q2"}, "retrieval_actions": [{"query": "q1", "selected_tools": ["bm25_search"]}], "accepted_evidence": [{"chunk_id": "c1", "evidence_role": "risk"}]},
                {"round": 2, "plan_step_id": "S1", "question_information_gain": {"score": 0.1, "new_chunk_ids": []}, "goal_evaluation": {"critical_gaps": ["gap2"], "query_database_status": "exhausted", "recommended_query_change": "q3"}, "retrieval_actions": [{"query": "q2", "selected_tools": ["dense_search"]}], "accepted_evidence": []},
            ],
        )
        self.assertTrue(result["consolidated"])
        self.assertEqual(result["memory_mode"], "rules")
        self.assertEqual(llm.calls, [])
        self.assertIn("gap1", result["long_memory"]["global_critical_gaps"])
        self.assertIn("c1", result["long_memory"]["evidence_ids"])

    def test_step_memory_builds_deterministic_micro_retrieval_ledger(self) -> None:
        ledger = StepMemoryAgent.build_micro_ledger({
            "tool_trace": [
                {"tool_name": "hybrid_search", "arguments": {"query": "q1"}, "result": {"hit_count": 8}},
                {"tool_name": "bm25_search", "arguments": {"query": "q2"}, "result": {"hit_count": 6}},
                {"tool_name": "rerank_candidates", "arguments": {}, "result": {}},
                {"tool_name": "dense_search", "arguments": {"query": "q3"}, "result": {"hit_count": 7}},
            ]
        })

        self.assertEqual([row["tool"] for row in ledger], ["hybrid_search", "bm25_search", "dense_search"])
        self.assertEqual([row["attempt"] for row in ledger], [1, 2, 3])
        self.assertEqual([row["hit_count"] for row in ledger], [8, 6, 7])

    def test_step_memory_separates_local_query_failure_from_database_coverage(self) -> None:
        result = RoundMemoryAgent().maintain(
            round_number=1,
            main_question="question",
            query="over-specific query",
            route_result={
                "query_type": "similar_case",
                "selected_tools": ["dense_search"],
                "constraints": {},
                "tool_trace": [{
                    "tool_name": "dense_search",
                    "arguments": {"query": "over-specific query"},
                    "result": {"hit_count": 0},
                }],
                "step_execution_report": {
                    "query_database_status": "exhausted",
                    "recommended_query_change": "broader query",
                },
                "evidence_layering": {},
            },
            evidence_review={
                "supporting_evidence": [], "contradicting_evidence": [],
                "safety_risks": [], "insufficient_evidence": [],
            },
            seen_chunk_ids=set(),
        )
        self.assertEqual(result["retrieval_state"]["search_attempt_status"], "no_hits")
        self.assertEqual(result["retrieval_state"]["query_direction_status"], "locally_exhausted")
        self.assertEqual(result["retrieval_state"]["database_coverage_status"], "unknown")
    class MockLLM:
        def __init__(self, response):
            self.response = response
            self.calls = []

        def chat(self, system, user, temperature=0.3, **kwargs):
            self.calls.append({"system": system, "user": user, "temperature": temperature})
            return self.response

        def call_function(self, system, user, temperature=0.3, **kwargs):
            self.calls.append({"system": system, "user": user, "temperature": temperature, **kwargs})
            return __import__("json").loads(self.response)

    def test_round_memory_records_information_gain(self) -> None:
        route = {
            "query_type": "mutation_drug",
            "selected_tools": ["bm25_search"],
            "constraints": {"drugs": ["osimertinib"]},
            "evidence_layering": {"assessed_evidence": [{"chunk_id": "c1"}]},
        }
        review = {
            "supporting_evidence": [{"chunk_id": "c1", "text": "partial response"}],
            "contradicting_evidence": [],
            "safety_risks": [],
            "insufficient_evidence": [],
        }
        result = RoundMemoryAgent().maintain(
            round_number=1,
            main_question="question",
            query="query",
            route_result=route,
            evidence_review=review,
            seen_chunk_ids=set(),
        )
        self.assertEqual(result["question_information_gain"]["new_chunk_ids"], ["c1"])
        self.assertIn("counter_evidence_for_balance", result["goal_evaluation"]["observed_gaps"])
        self.assertNotIn("planner_information_gain", result)

    def test_answer_memory_accumulates_rounds_and_safety_revises_claim(self) -> None:
        agent = AnswerMemoryAgent()
        memory = {"claims": [], "evidence_by_id": {}, "informative_rounds": []}
        for round_number, chunk_id, role in [(1, "c1", "support"), (2, "c2", "counter")]:
            memory = agent.update(memory, {
                "round": round_number,
                "accepted_evidence": [{"chunk_id": chunk_id, "evidence_role": role}],
                "question_information_gain": {"new_chunk_ids": [chunk_id]},
            })
        self.assertEqual(memory["informative_rounds"], [1, 2])
        self.assertEqual(set(memory["evidence_by_id"]), {"c1", "c2"})
        agent.apply_safety_review(memory, {"requires_human_review": True, "issues": []})
        self.assertEqual(memory["claims"][0]["safety_status"], "revise")

    def test_answer_memory_rule_fallback_keeps_distinct_scopes_atomic(self) -> None:
        result = AnswerMemoryAgent(use_llm=False).update(
            {"claims": [], "evidence_by_id": {}, "informative_rounds": []},
            {
                "round": 1,
                "accepted_evidence": [
                    {
                        "chunk_id": "c1",
                        "evidence_role": "analog_support",
                        "subject_entity": "osimertinib",
                        "claim_scope": "EGFR患者使用奥希替尼后脑转移部分缓解",
                    },
                    {
                        "chunk_id": "c2",
                        "evidence_role": "analog_support",
                        "subject_entity": "crizotinib",
                        "claim_scope": "ALK患者使用克唑替尼后PFS较短",
                    },
                ],
                "question_information_gain": {"new_chunk_ids": ["c1", "c2"]},
            },
        )

        self.assertEqual(len(result["claims"]), 2)
        self.assertEqual(
            {claim["claim"] for claim in result["claims"]},
            {
                "EGFR患者使用奥希替尼后脑转移部分缓解",
                "ALK患者使用克唑替尼后PFS较短",
            },
        )
        self.assertTrue(all("；" not in claim["claim"] for claim in result["claims"]))

    def test_citation_agent_traces_claim_to_source_file(self) -> None:
        result = CitationAgent().trace_claims(
            [{"claim_id": "claim_01", "claim": "x", "supporting_chunk_ids": ["c1"]}],
            {"c1": {
                "chunk_id": "c1",
                "doc_id": "d1",
                "text": "source span",
                "citation_json": {"source_file": "raw/d1.json", "field_path": "text", "start_char": 10},
            }},
        )
        citation = result["claim_citations"][0]["citations"][0]
        self.assertEqual(citation["source_file"], "raw/d1.json")
        self.assertEqual(citation["start_char"], 10)

    def test_round_memory_is_deterministic_even_when_llm_is_supplied(self) -> None:
        llm = self.MockLLM('{"accepted_evidence":[{"chunk_id":"c1","evidence_role":"support","reason":"新增疗效"}],"rejected_evidence":[],"question_information_gain":{"score":0.8,"new_facts":["PR"],"resolved_questions":[],"new_conflicts":[],"new_chunk_ids":["c1"]},"goal_evaluation":{"matched_goal_count":1,"best_goal_relevance":0.8,"success_criteria_met":true,"observed_gaps":[],"observed_failures":[]}}')
        result = RoundMemoryAgent(llm_client=llm, use_llm=True).maintain(
            round_number=1,
            main_question="q",
            query="q1",
            route_result={"query_type": "mutation_drug", "selected_tools": [], "constraints": {}, "evidence_layering": {}},
            evidence_review={"supporting_evidence": [{"chunk_id": "c1", "text": "PR"}]},
            seen_chunk_ids=set(),
        )
        self.assertEqual(result["memory_mode"], "rules")
        self.assertEqual(result["accepted_evidence"][0]["chunk_id"], "c1")
        self.assertNotIn("planner_information_gain", result)

    def test_answer_memory_llm_creates_specific_claim(self) -> None:
        llm = self.MockLLM('{"claims":[{"claim_id":"claim_01","claim":"病例中观察到部分缓解","status":"provisional","supporting_chunk_ids":["c1"],"contradicting_chunk_ids":[],"source_rounds":[1],"confidence":0.7,"safety_status":"pending"}],"informative_rounds":[1]}')
        result = AnswerMemoryAgent(llm_client=llm, use_llm=True).update(
            {"claims": [], "evidence_by_id": {}, "informative_rounds": []},
            {"round": 1, "accepted_evidence": [{"chunk_id": "c1", "evidence_role": "support"}], "question_information_gain": {"new_chunk_ids": ["c1"]}},
        )
        self.assertEqual(result["memory_mode"], "llm")
        self.assertEqual(result["claims"][0]["claim"], "病例中观察到部分缓解")

    def test_answer_memory_llm_accepts_structured_round_numbers(self) -> None:
        llm = self.MockLLM('{"claims":[{"claim_id":"claim_01","claim":"病例中观察到部分缓解","status":"provisional","supporting_chunk_ids":["c1"],"contradicting_chunk_ids":[],"source_rounds":[{"round":1}],"confidence":0.7,"safety_status":"pending"}],"informative_rounds":[{"round_number":1},"2",{"unexpected":3}]}')
        result = AnswerMemoryAgent(llm_client=llm, use_llm=True).update(
            {"claims": [], "evidence_by_id": {}, "informative_rounds": []},
            {"round": 1, "accepted_evidence": [{"chunk_id": "c1", "evidence_role": "support"}], "question_information_gain": {"new_chunk_ids": ["c1"]}},
        )
        self.assertEqual(result["memory_mode"], "llm")
        self.assertEqual(result["informative_rounds"], [1, 2])
        self.assertEqual(result["claims"][0]["source_rounds"], [1])


if __name__ == "__main__":
    unittest.main()
