from searchagent_retrieval.llm_rerank import (
    LLMReranker,
    _looks_like_citation_pointer,
    _query_aware_excerpt,
)
from searchagent_retrieval.tools import SearchHit
def test_standard_chunk_is_not_truncated() -> None:
    text = "A" * 700 + " BLOOM osimertinib 160 mg ORR 62% PFS 8.6 months OS 11 months"
    assert len(text) < 1200
    assert _query_aware_excerpt(text, "BLOOM osimertinib 160 mg ORR", 1200) == text


def test_query_aware_excerpt_preserves_late_outcome() -> None:
    text = "background " * 180 + "BLOOM osimertinib 160 mg ORR 62% median PFS 8.6 months"
    excerpt = _query_aware_excerpt(text, "BLOOM osimertinib 160 mg ORR PFS", 700)
    assert len(excerpt) <= 700
    assert "BLOOM osimertinib 160 mg" in excerpt
    assert "ORR 62%" in excerpt


def test_reference_list_is_detected_for_compact_input() -> None:
    text = "Study title. [DOI] [PMC free article] [PubMed] [Google Scholar]"
    assert _looks_like_citation_pointer(text)


def test_pointer_score_is_capped_by_code() -> None:
    class FakeClient:
        def call_function(self, **kwargs):
            return {
                "rankings": [{
                    "id": "ref-1",
                    "evidence_type": "citation_pointer",
                    "population_match": 3,
                    "intervention_match": 3,
                    "outcome_match": 3,
                    "directness": 3,
                    "attribution": 3,
                    "reason": "citation_pointer: title only",
                }]
            }

    hit = SearchHit(
        id="ref-1",
        score=10.0,
        source="bm25",
        text="Study title and citation only.",
        metadata={},
    )
    reranked = LLMReranker(FakeClient()).rerank("clinical outcome", [hit], 1)
    assert reranked[0].score == 0.15
    assert reranked[0].metadata["llm_evidence_type"] == "citation_pointer"


def test_llm_failure_moves_reference_pointer_after_body_evidence() -> None:
    class FailingClient:
        def call_function(self, **kwargs):
            raise RuntimeError("connection failed")

    pointer = SearchHit(
        id="ref",
        score=25.0,
        source="bm25",
        text="Study title. [DOI] [PubMed] [Google Scholar]",
        metadata={},
    )
    body = SearchHit(
        id="body",
        score=20.0,
        source="bm25",
        text="LM patients received osimertinib 160 mg and ORR was 62%.",
        metadata={},
    )
    reranked = LLMReranker(FailingClient()).rerank("LM osimertinib outcome", [pointer, body], 2)
    assert [hit.id for hit in reranked] == ["body", "ref"]
    assert reranked[1].score == 0.15
    assert reranked[1].metadata["llm_rerank_status"] == "fallback_error"


def test_reranker_uses_bounded_serial_batches_and_program_scores() -> None:
    class BatchClient:
        def __init__(self):
            self.calls = 0

        def call_function(self, **kwargs):
            self.calls += 1
            ids = [line.split("id=", 1)[1] for line in kwargs["user"].splitlines() if "id=" in line]
            return {"rankings": [{
                "id": hit_id,
                "evidence_type": "primary_result",
                "population_match": 3,
                "intervention_match": 3,
                "outcome_match": 3,
                "directness": 3,
                "attribution": 3,
                "reason": "direct outcome",
            } for hit_id in ids]}

    client = BatchClient()
    hits = [SearchHit(str(i), float(i), "bm25", "Direct clinical outcome text.", {}) for i in range(10)]
    ranked = LLMReranker(client).rerank("clinical outcome", hits, 10)
    assert client.calls == 2
    assert all(hit.score == 1.0 for hit in ranked)
    assert all(hit.metadata["llm_rerank_dimensions"]["outcome_match"] == 3 for hit in ranked)


def test_regimen_mismatch_caps_otherwise_strong_combination_candidate() -> None:
    reranker = LLMReranker(object())
    exact = {
        "population_match": 3, "intervention_match": 3, "regimen_match": 3,
        "timepoint_match": 3, "outcome_match": 3, "directness": 3, "attribution": 3,
    }
    combination_for_monotherapy = {
        "population_match": 3, "intervention_match": 3, "regimen_match": 1,
        "timepoint_match": 3, "outcome_match": 3, "directness": 3, "attribution": 1,
    }
    assert reranker._deterministic_score("case_result", exact) == 0.75
    assert reranker._deterministic_score("case_result", combination_for_monotherapy) <= 0.55
    assert reranker._deterministic_score(
        "case_result", exact, {"wrong_drug", "wrong_time_window"}
    ) <= 0.25


def test_serial_batches_share_calibration_anchors_and_emit_hard_matches() -> None:
    class AnchoredClient:
        def __init__(self):
            self.prompts = []

        def call_function(self, **kwargs):
            self.prompts.append(kwargs["user"])
            ids = [line.split("id=", 1)[1] for line in kwargs["user"].splitlines() if "id=" in line]
            return {"rankings": [{
                "id": hit_id, "evidence_type": "case_result", "match_status": "hard_mismatch",
                "hard_mismatches": ["wrong_regimen"], "population_match": 3,
                "intervention_match": 3, "regimen_match": 1, "timepoint_match": 3,
                "outcome_match": 3, "directness": 3, "attribution": 1,
                "reason": "target monotherapy but current combination",
            } for hit_id in ids]}

    client = AnchoredClient()
    hits = [SearchHit(str(i), float(i), "dense", "clinical result", {}) for i in range(10)]
    ranked = LLMReranker(client).rerank("target monotherapy at 8-12 weeks", hits, 10)
    assert len(client.prompts) == 2
    assert all("跨批次固定校准锚点" in prompt for prompt in client.prompts)
    assert all(hit.score <= 0.55 for hit in ranked)
    assert all(hit.metadata["llm_match_status"] == "hard_mismatch" for hit in ranked)
    assert all(hit.metadata["llm_hard_mismatches"] == ["wrong_regimen"] for hit in ranked)


def test_two_rerank_batches_run_serially() -> None:
    class SerialClient:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.calls = 0

        def call_function(self, **kwargs):
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            ids = [line.split("id=", 1)[1] for line in kwargs["user"].splitlines() if "id=" in line]
            result = {"rankings": [{
                "id": hit_id,
                "evidence_type": "case_result",
                "population_match": 2,
                "intervention_match": 2,
                "outcome_match": 2,
                "directness": 2,
                "attribution": 2,
                "reason": "serial batch",
            } for hit_id in ids]}
            self.active -= 1
            return result

    hits = [SearchHit(str(i), float(i), "bm25", "Direct clinical outcome text.", {}) for i in range(12)]
    client = SerialClient()
    ranked = LLMReranker(client).rerank("clinical outcome", hits, 12)
    assert client.calls == 2
    assert client.max_active == 1
    assert len(ranked) == 12
    assert all(hit.metadata["llm_rerank_status"] == "completed" for hit in ranked)
