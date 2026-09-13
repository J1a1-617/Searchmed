import unittest
import json
import tempfile
from pathlib import Path

from searchagent_retrieval.execution_agent import RetrievalExecutionAgent
from searchagent_retrieval.llm_client import LLMClientError
from searchagent_retrieval.router import RetrievalRouter, RoutePlan
from searchagent_retrieval.tools import SearchHit


class _FakeTools:
    def __init__(self):
        self.last_bm25_query = ""

    def bm25_search(self, query, top_k):
        self.last_bm25_query = query
        return [SearchHit("doc#chunk-1", 7.0, "bm25", "progressive disease after treatment", {})]

    def rerank(self, query, hits, top_k):
        hit = hits[0]
        return [SearchHit(hit.id, 0.95, hit.source, hit.text, {"llm_relevance_score": 0.95})]

    def fetch_evidence(self, chunk_ids=None, limit=20, **kwargs):
        return [{
            "chunk_id": "doc#chunk-1", "doc_id": "doc", "case_id": "case-1",
            "evidence_level": "case_report_evidence",
            "text": "progressive disease after osimertinib treatment",
            "citation_json": {}, "metadata_json": {},
        }]


class _ScriptedLoopLLM:
    def call_function(self, **kwargs):
        assert kwargs["function_name"] == "execute_retrieval_batch"
        return {"actions": [{
            "tool": "bm25_search", "query": "osimertinib resistance",
            "constraints": {
                "cancer_type": "", "gene_alterations": [], "drugs": [],
                "responses": [], "toxicities": [], "metastatic_sites": [],
                "ddi_terms": [],
            },
            "top_k": 10, "spaces": [],
        }]}

    def run_function_tool_loop(self, *, execute_tool, **kwargs):
        first = execute_tool("bm25_search", {"query": "osimertinib resistance", "top_k": 10})
        self.assert_result(first["hit_count"] == 1)
        second = execute_tool("rerank_candidates", {"goal": "find resistance", "candidate_ids": ["doc#chunk-1"], "top_k": 5})
        self.assert_result(second["hits"][0]["score"] == 0.95)
        third = execute_tool("fetch_evidence", {"chunk_ids": ["doc#chunk-1"], "limit": 5})
        self.assert_result(third["fetched_count"] == 1)
        return {
            "report": {
                "execution_status": "success", "summary": "found resistance evidence",
                "accepted_evidence_ids": ["doc#chunk-1"], "rejected_evidence_ids": [],
                "queries_attempted": ["osimertinib resistance"],
                "goal_evaluation": {"matched_goal_count": 1, "best_goal_relevance": 0.95, "success_criteria_met": True, "observed_gaps": [], "observed_failures": []},
            },
            "tool_trace": [{"tool_name": "bm25_search"}, {"tool_name": "rerank_candidates"}, {"tool_name": "fetch_evidence"}],
            "turns": 4,
        }

    @staticmethod
    def assert_result(value):
        if not value:
            raise AssertionError("unexpected tool result")


class _ExpansionLoopLLM:
    def run_function_tool_loop(self, *, execute_tool, **kwargs):
        expanded = execute_tool("expand_query_with_knowledge", {
            "query": "Tagrisso resistance",
            "drug_names": ["Tagrisso"],
            "mutation_names": [],
            "strategies": ["drug_alias"],
            "top_k_terms": 5,
        })
        self.assert_result("osimertinib" in " ".join(expanded["added_terms"]).lower())
        bm25 = execute_tool("bm25_search", {"query": expanded["expanded_query"], "top_k": 10})
        self.assert_result(bm25["hit_count"] == 1)
        evidence = execute_tool("fetch_evidence", {"chunk_ids": ["doc#chunk-1"], "limit": 5})
        self.assert_result(evidence["fetched_count"] == 1)
        return {
            "report": {
                "execution_status": "success",
                "summary": "expanded query and found evidence",
                "accepted_evidence_ids": ["doc#chunk-1"],
                "rejected_evidence_ids": [],
                "queries_attempted": [expanded["expanded_query"]],
                "goal_evaluation": {
                    "matched_goal_count": 1,
                    "best_goal_relevance": 0.8,
                    "success_criteria_met": True,
                    "observed_gaps": [],
                    "observed_failures": [],
                },
            },
            "tool_trace": [{"tool_name": "expand_query_with_knowledge"}, {"tool_name": "bm25_search"}, {"tool_name": "fetch_evidence"}],
            "turns": 3,
        }

    @staticmethod
    def assert_result(value):
        if not value:
            raise AssertionError("unexpected tool result")


class _RecoveringLoopLLM:
    def __init__(self):
        self.recovery_calls = 0

    def run_function_tool_loop(self, *, execute_tool, **kwargs):
        execute_tool("bm25_search", {"query": "osimertinib resistance", "top_k": 10})
        raise LLMClientError("Tool arguments are not valid JSON: Unterminated string")

    def call_function(self, **kwargs):
        self.recovery_calls += 1
        assert kwargs["function_name"] == "submit_minimal_step_execution_report"
        assert kwargs["max_attempts"] == 1
        assert "progressive disease after treatment" not in kwargs["user"]
        return {
            "execution_status": "partial",
            "accepted_evidence_ids": ["doc#chunk-1"],
            "query_database_status": "more_available",
            "matched_goal_count": 1,
            "success_criteria_met": False,
        }


class _TwoStageLoopLLM:
    def __init__(self):
        self.selection_calls = 0

    def call_function(self, **kwargs):
        self.selection_calls += 1
        assert kwargs["function_name"] == "execute_retrieval_batch"
        return {"actions": [{
            "tool": "bm25_search",
            "query": "osimertinib resistance",
            "constraints": {
                "cancer_type": "", "gene_alterations": [], "drugs": [],
                "responses": [], "toxicities": [], "metastatic_sites": [],
                "ddi_terms": [],
            },
            "top_k": 10,
            "spaces": [],
        }]}

    def run_function_tool_loop(self, *, execute_tool, tools, max_turns, initial_tool_choice, **kwargs):
        names = {tool["function"]["name"] for tool in tools}
        assert names == {"execute_retrieval_batch", "submit_step_execution_report"}
        assert max_turns == 2
        assert initial_tool_choice == "required"
        result = execute_tool("execute_retrieval_batch", {
            "actions": [{
                "tool": "bm25_search",
                "query": "osimertinib resistance",
                "constraints": {
                    "cancer_type": "", "gene_alterations": [], "drugs": [],
                    "responses": [], "toxicities": [], "metastatic_sites": [],
                    "ddi_terms": [],
                },
                "top_k": 10,
                "spaces": [],
            }],
        })
        assert result["fetched_evidence_ids"] == ["doc#chunk-1"]
        return {
            "report": {
                "execution_status": "success",
                "summary": "found evidence",
                "accepted_evidence_ids": ["doc#chunk-1"],
                "rejected_evidence_ids": [],
                "queries_attempted": ["osimertinib resistance"],
                "query_database_status": "more_available",
                "exhaustion_reason": "",
                "recommended_query_change": "",
                "goal_evaluation": {
                    "matched_goal_count": 1,
                    "best_goal_relevance": 0.95,
                    "success_criteria_met": True,
                    "observed_gaps": [],
                    "observed_failures": [],
                },
            },
            "tool_trace": [{"tool_name": "execute_retrieval_batch", "result": result}],
            "turns": 2,
        }


class _ToolFirstTools(_FakeTools):
    def __init__(self):
        super().__init__()
        self.seed_query = ""
        self.last_dense_query = ""

    def step3_then_step2_search(self, query, top_k):
        self.seed_query = query
        return []

    def dense_search(self, query, top_k, spaces):
        self.last_dense_query = query
        return []


class _ToolFirstLoopLLM:
    def __init__(self, retrieval_tools):
        self.retrieval_tools = retrieval_tools

    def call_function(self, *, user, **kwargs):
        assert self.retrieval_tools.seed_query == ""
        prompt = json.loads(user)
        assert "planner_query_hint" in prompt
        dense_query = (
            "EGFR L858R lung adenocarcinoma with leptomeningeal metastases "
            "after TKI progression treated with pemetrexed carboplatin"
        )
        bm25_query = "EGFR L858R NSCLC LMD CSF pemetrexed carboplatin"
        return {"actions": [
            {
                "tool": "dense_search", "query": dense_query,
                "constraints": {
                    "cancer_type": "", "gene_alterations": [], "drugs": [],
                    "responses": [], "toxicities": [], "metastatic_sites": [],
                    "ddi_terms": [],
                },
                "top_k": 10, "spaces": ["case_semantic", "event_semantic"],
            },
            {
                "tool": "bm25_search", "query": bm25_query,
                "constraints": {
                    "cancer_type": "", "gene_alterations": [], "drugs": [],
                    "responses": [], "toxicities": [], "metastatic_sites": [],
                    "ddi_terms": [],
                },
                "top_k": 10, "spaces": [],
            },
        ]}

    def run_function_tool_loop(self, *, execute_tool, user, **kwargs):
        # No Planner query may execute before the model chooses tools and
        # writes tool-specific action queries.
        assert self.retrieval_tools.seed_query == ""
        prompt = json.loads(user)
        assert "planner_query_hint" in prompt
        assert "step3_then_step2_seed" not in prompt
        dense_query = (
            "EGFR L858R lung adenocarcinoma with leptomeningeal metastases "
            "after TKI progression treated with pemetrexed carboplatin"
        )
        bm25_query = "EGFR L858R NSCLC LMD CSF pemetrexed carboplatin"
        result = execute_tool("execute_retrieval_batch", {"actions": [
            {
                "tool": "dense_search", "query": dense_query,
                "constraints": {
                    "cancer_type": "", "gene_alterations": [], "drugs": [],
                    "responses": [], "toxicities": [], "metastatic_sites": [],
                    "ddi_terms": [],
                },
                "top_k": 10, "spaces": ["case_semantic", "event_semantic"],
            },
            {
                "tool": "bm25_search", "query": bm25_query,
                "constraints": {
                    "cancer_type": "", "gene_alterations": [], "drugs": [],
                    "responses": [], "toxicities": [], "metastatic_sites": [],
                    "ddi_terms": [],
                },
                "top_k": 10, "spaces": [],
            },
        ]})
        assert self.retrieval_tools.seed_query == dense_query
        assert self.retrieval_tools.last_dense_query == dense_query
        assert self.retrieval_tools.last_bm25_query == bm25_query
        return {
            "report": {
                "execution_status": "success", "summary": "",
                "accepted_evidence_ids": result["fetched_evidence_ids"],
                "rejected_evidence_ids": [], "queries_attempted": [],
                "query_database_status": "more_available",
                "exhaustion_reason": "", "recommended_query_change": "",
                "goal_evaluation": {
                    "matched_goal_count": 1, "best_goal_relevance": 1.0,
                    "success_criteria_met": True, "observed_gaps": [],
                    "observed_failures": [],
                },
            },
            "tool_trace": [], "turns": 2,
        }


class ExecutionAgentTests(unittest.TestCase):
    def test_tools_are_selected_before_tool_specific_queries_and_seed(self):
        tools = _ToolFirstTools()
        agent = RetrievalExecutionAgent(
            RetrievalRouter(tools, use_llm=False),
            _ToolFirstLoopLLM(tools),
        )
        result = agent.run(
            main_question="EGFR L858R肺腺癌伴脑膜转移，TKI进展后使用培美曲塞卡铂",
            plan_step={
                "step_id": "S1", "goal": "检索脑膜转移疗效",
                "rerank_goal": "寻找LMD或CSF疗效", "evidence_lane": "direct_case",
                "success_criteria": ["找到相关证据"],
            },
            route_plan=RoutePlan(
                query_type="treatment_advice",
                selected_tools=["dense_search", "bm25_search"],
                constraints={},
                search_query="通用Planner检索意图",
            ),
            rerank_goal="寻找LMD或CSF疗效",
            top_k=5,
        )
        self.assertNotEqual(tools.last_dense_query, tools.last_bm25_query)
        self.assertEqual(result["planner_metadata"]["seed_query"], tools.last_dense_query)

    def test_normal_execution_uses_two_stage_batch_and_auto_rerank(self):
        router = RetrievalRouter(_FakeTools(), use_llm=False)
        agent = RetrievalExecutionAgent(router, _TwoStageLoopLLM(), max_turns=5)

        result = agent.run(
            main_question="question",
            plan_step={"step_id": "S1", "goal": "resistance", "rerank_goal": "find resistance", "evidence_lane": "direct_case", "success_criteria": ["one"]},
            route_plan=RoutePlan("mutation_drug", ["bm25_search"], {}, "osimertinib resistance", {}),
            rerank_goal="find resistance",
            top_k=5,
        )

        self.assertEqual(result["tool_loop_turns"], 1)
        self.assertEqual(agent.llm_client.selection_calls, 1)
        self.assertIn("rerank_candidates", result["selected_tools"])
        self.assertEqual(result["results"]["fetch_evidence"][0]["chunk_id"], "doc#chunk-1")
        self.assertEqual(result["step_execution_report"]["report_source"], "deterministic_tool_result")

    def test_real_tool_dispatch_builds_auditable_route_result(self):
        router = RetrievalRouter(_FakeTools(), use_llm=False)
        agent = RetrievalExecutionAgent(router, _ScriptedLoopLLM())
        result = agent.run(
            main_question="question",
            plan_step={"step_id": "S2", "goal": "resistance", "rerank_goal": "find resistance", "evidence_lane": "direct_case", "success_criteria": ["one"]},
            route_plan=RoutePlan("mutation_drug", ["bm25_search", "fetch_evidence"], {}, "osimertinib resistance", {}),
            rerank_goal="find resistance",
            top_k=5,
        )
        self.assertEqual(agent.max_turns, 3)
        self.assertEqual(result["planner_metadata"]["execution_mode"], "function_tool_loop")
        self.assertEqual(result["results"]["fetch_evidence"][0]["chunk_id"], "doc#chunk-1")
        self.assertEqual(result["step_execution_report"]["accepted_evidence_ids"], ["doc#chunk-1"])
        self.assertEqual(result["selected_tools"], ["bm25_search", "rerank_candidates", "fetch_evidence"])

    def test_lane_and_expansion_policy_limit_visible_tools(self):
        router = RetrievalRouter(_FakeTools(), use_llm=False)
        agent = RetrievalExecutionAgent(router, _ScriptedLoopLLM())
        direct_names = {tool["function"]["name"] for tool in agent._tool_definitions("direct_case", {"enabled": True, "allowed_strategies": ["ddi_rule"]})}
        ddi_names = {tool["function"]["name"] for tool in agent._tool_definitions("ddi_safety", {"enabled": True, "allowed_strategies": ["drug_alias", "pk_relation", "ddi_rule", "case_ddi"]})}
        self.assertNotIn("lookup_ddi_rules", direct_names)
        self.assertIn("expand_query_with_knowledge", direct_names)
        self.assertIn("normalize_drug_names", ddi_names)
        self.assertIn("lookup_pk_relations", ddi_names)
        self.assertIn("lookup_ddi_rules", ddi_names)
        self.assertIn("lookup_case_ddi_relations", ddi_names)

    def test_expand_query_tool_participates_in_agent_tool_calling(self):
        fake_tools = _FakeTools()
        router = RetrievalRouter(fake_tools, use_llm=False)
        agent = RetrievalExecutionAgent(router, _ExpansionLoopLLM())
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            (data_root / "all_drug_name_map.json").write_text(json.dumps([
                {"drug_name": "osimertinib", "generic_name": "osimertinib", "brand_name": "Tagrisso", "alias": ["泰瑞沙"]}
            ], ensure_ascii=False), encoding="utf-8")
            (data_root / "all_drug_pk_relation.json").write_text("[]", encoding="utf-8")
            (data_root / "all_ddi_rule.json").write_text("[]", encoding="utf-8")
            (data_root / "mutation_drug_map_min.json").write_text("[]", encoding="utf-8")
            agent.data_root = data_root
            result = agent.run(
                main_question="question",
                plan_step={"step_id": "S2", "goal": "resistance", "rerank_goal": "find resistance", "evidence_lane": "direct_case", "success_criteria": ["one"]},
                route_plan=RoutePlan("mutation_drug", ["bm25_search", "fetch_evidence"], {}, "Tagrisso resistance", {}),
                rerank_goal="find resistance",
                top_k=5,
                expansion={"enabled": True, "allowed_strategies": ["drug_alias"]},
            )
        self.assertIn("expand_query_with_knowledge", result["selected_tools"])
        self.assertIn("osimertinib", fake_tools.last_bm25_query.lower())
        self.assertEqual(result["step_execution_report"]["accepted_evidence_ids"], ["doc#chunk-1"])

    def test_lookup_case_ddi_relations_returns_direct_case_matches(self):
        router = RetrievalRouter(_FakeTools(), use_llm=False)
        agent = RetrievalExecutionAgent(router, _ScriptedLoopLLM())
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            (data_root / "all_drug_name_map.json").write_text("[]", encoding="utf-8")
            (data_root / "all_drug_pk_relation.json").write_text("[]", encoding="utf-8")
            (data_root / "all_ddi_rule.json").write_text("[]", encoding="utf-8")
            (data_root / "mutation_drug_map_min.json").write_text("[]", encoding="utf-8")
            (data_root / "case_ddi.json").write_text(json.dumps({
                "metadata": {},
                "case_ddis": [
                    {
                        "pmid": "1",
                        "drugs": ["osimertinib", "voriconazole"],
                        "interaction_type": "CYP_Inhibition",
                        "mutation": {"alterations": ["EGFR L858R"]},
                        "outcome": {"response": "PR", "adverse_events": ["rash"]},
                    }
                ],
            }, ensure_ascii=False), encoding="utf-8")
            agent.data_root = data_root
            rows = agent._lookup_case_ddi_rows(["osimertinib"], ["EGFR"], ["CYP_Inhibition"], top_k=5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["interaction_type"], "CYP_Inhibition")

    def test_curated_ddi_id_cannot_override_canonical_source_id(self):
        router = RetrievalRouter(_FakeTools(), use_llm=False)
        agent = RetrievalExecutionAgent(router, _ScriptedLoopLLM())
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            (data_root / "all_ddi_rule.json").write_text(json.dumps([
                {
                    "id": "DDI",
                    "enzyme_transporter": "CYP3A4",
                    "role1": "substrate",
                    "role2": "inhibitor",
                }
            ]), encoding="utf-8")
            agent.data_root = data_root
            rows = agent._lookup_ddi_rule_rows(["cyp3a4"])
        self.assertEqual(rows[0]["id"], "ddi_rule:0")

    def test_terminal_report_schema_has_strict_size_bounds(self):
        router = RetrievalRouter(_FakeTools(), use_llm=False)
        agent = RetrievalExecutionAgent(router, _ScriptedLoopLLM())
        terminal = next(
            tool for tool in agent._tool_definitions("direct_case", {"enabled": False})
            if tool["function"]["name"] == "submit_step_execution_report"
        )
        schema = terminal["function"]["parameters"]
        self.assertEqual(schema["properties"]["summary"]["maxLength"], 320)
        self.assertEqual(schema["properties"]["accepted_evidence_ids"]["maxItems"], 12)
        self.assertEqual(schema["properties"]["queries_attempted"]["maxItems"], 2)

    def test_cross_resource_previous_response_error_is_recoverable(self):
        error = LLMClientError(
            "Error code: 500 - The requested item was created under a different Azure OpenAI resource"
        )
        self.assertTrue(RetrievalExecutionAgent._is_recoverable_terminal_protocol_error(error))


if __name__ == "__main__":
    unittest.main()
