import json
import unittest

from searchagent_retrieval.bm25_index import BM25Index
from searchagent_retrieval.execution_agent import RetrievalExecutionAgent
from searchagent_retrieval.planning import MultiStepPlanningAgent
from searchagent_retrieval.tools import RetrievalTools


class _ExpansionChoosingLLM:
    def __init__(self) -> None:
        self.kwargs = {}

    def call_function(self, **kwargs):
        self.kwargs = kwargs
        return {
            "action": "expand_current_step",
            "active_step_id": "S1",
            "search_query": "JS212 alias EGFR lung cancer response",
            "rerank_goal": "direct response",
            "selected_tools": ["dense_search", "bm25_search", "fetch_evidence"],
            "constraints": {
                "cancer_type": "lung adenocarcinoma",
                "gene_alterations": ["EGFR exon 19Del"],
                "drugs": ["JS212"],
                "responses": [],
                "toxicities": [],
                "metastatic_sites": [],
                "ddi_terms": [],
            },
            "expansion": {
                "strategy": "drug_alias",
                "source": "all_drug_name_map.json",
                "added_terms": ["invented alias"],
                "relation_type": "same_drug_identity",
            },
            "plan_changes": [],
            "avoid_repeating": [],
            "decision_rationale": "try an alias",
            "requested_budget_extension": 0,
            "budget_extension_reason": "",
        }


class _RouterStub:
    tools = object()


class ExternalKnowledgeAblationTests(unittest.TestCase):
    def test_bm25_subset_recomputes_statistics_and_removes_rules(self) -> None:
        index = BM25Index()
        index.add("case#1", "EGFR lung response", {"source": "case"})
        index.add(
            "ddi_rule#1",
            "EGFR interaction rule",
            {"source": "rule", "chunk_type": "ddi_rule_chunk"},
        )

        subset = index.filtered(
            lambda doc: not RetrievalTools._is_external_knowledge_metadata(doc["metadata"])
        )

        self.assertEqual([doc["doc_id"] for doc in subset.docs], ["case#1"])
        self.assertEqual(subset.doc_freq, {"egfr": 1, "lung": 1, "response": 1})
        self.assertEqual(subset.avg_len, 3.0)

    def test_disabled_replanner_converts_knowledge_expansion_to_query_retry(self) -> None:
        llm = _ExpansionChoosingLLM()
        agent = MultiStepPlanningAgent(
            llm_client=llm,
            use_llm=True,
            external_knowledge_enabled=False,
        )
        plan = {
            "steps": [{
                "step_id": "S1",
                "goal": "direct JS212 response",
                "rerank_goal": "direct response",
                "evidence_lane": "direct_case",
                "success_criteria": ["one direct outcome"],
                "attempt_budget": 2,
            }]
        }
        memory = {
            "plan_step_id": "S1",
            "goal_evaluation": {"completion_status": "minimally_met", "matched_goal_count": 1},
            "question_information_gain": {"new_chunk_ids": ["case#1"]},
            "retrieval_actions": [{"query": "old query"}],
        }
        decision = agent.replan(
            main_question="question",
            plan=plan,
            active_step_index=0,
            recent_memories=[memory],
            answer_memory={},
            base_query_type=None,
            base_constraints={"drugs": ["JS212"], "gene_alterations": ["EGFR exon 19Del"]},
        )

        self.assertEqual(decision["action"], "retry_current_step")
        self.assertEqual(decision["expansion"]["strategy"], "none")
        self.assertEqual(decision["expansion"]["added_terms"], [])
        prompt = json.loads(llm.kwargs["user"])
        self.assertEqual(prompt["database_profile"]["external_knowledge"], "disabled for this ablation; use only the case-report corpus")
        self.assertIn("本次消融已禁用", llm.kwargs["system"])

    def test_disabled_execution_agent_never_exposes_knowledge_tools(self) -> None:
        agent = RetrievalExecutionAgent(
            router=_RouterStub(),
            llm_client=None,
            external_knowledge_enabled=False,
        )
        definitions = agent._tool_definitions(
            "ddi_safety",
            {"enabled": True, "allowed_strategies": ["drug_alias", "case_ddi", "pk_relation", "ddi_rule"]},
        )
        names = {item["function"]["name"] for item in definitions}
        self.assertFalse(names & {
            "expand_query_with_knowledge",
            "normalize_drug_names",
            "lookup_pk_relations",
            "lookup_ddi_rules",
            "lookup_mutation_drug_relations",
            "lookup_case_ddi_relations",
        })


if __name__ == "__main__":
    unittest.main()
