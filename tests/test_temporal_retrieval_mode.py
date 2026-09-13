from types import SimpleNamespace

import pytest

from searchagent_retrieval.tools import RetrievalTools


class _Documents:
    def get_document_metadata(self, doc_ids):
        rows = {
            "past": {"doc_id": "past", "pub_date": "2021-01-01", "pmid": "1", "publication_date_source": "citation_publication_date"},
            "future": {"doc_id": "future", "pub_date": "2025-01-01", "pmid": "2", "publication_date_source": "citation_publication_date"},
            "undated": {"doc_id": "undated", "pub_date": None, "pmid": "3", "publication_date_source": "citation_publication_date"},
        }
        return {doc_id: rows[doc_id] for doc_id in doc_ids if doc_id in rows}


class _Vector:
    def __init__(self):
        self.requested_top_k = None

    def search(self, *, query, top_k, spaces):
        self.requested_top_k = top_k
        return [
            SimpleNamespace(doc_id="future#1", score=0.99, text="future", metadata={"doc_id": "future"}),
            SimpleNamespace(doc_id="undated#1", score=0.98, text="undated", metadata={"doc_id": "undated"}),
            SimpleNamespace(doc_id="past#1", score=0.80, text="past", metadata={"doc_id": "past"}),
        ]


def _tools():
    tools = RetrievalTools.__new__(RetrievalTools)
    tools.structured = _Documents()
    tools.vector = _Vector()
    tools.temporal_filter_mode = "off"
    tools.temporal_cutoff = None
    return tools


def test_cutoff_mode_overfetches_then_removes_future_and_undated_hits() -> None:
    tools = _tools()
    tools.configure_temporal_filter("cutoff", "2022-08")

    hits = tools.dense_search("EGFR osimertinib response", top_k=2)

    assert tools.temporal_cutoff == "2022-08-31"
    assert tools.vector.requested_top_k > 2
    assert [hit.id for hit in hits] == ["past#1"]
    assert hits[0].metadata["pub_date"] == "2021-01-01"


def test_off_mode_preserves_normal_retrieval() -> None:
    tools = _tools()
    tools.configure_temporal_filter("off")

    hits = tools.dense_search("EGFR osimertinib response", top_k=2)

    assert tools.vector.requested_top_k == 2
    assert [hit.id for hit in hits] == ["future#1", "undated#1"]


def test_cutoff_mode_requires_valid_cutoff() -> None:
    tools = _tools()
    with pytest.raises(ValueError):
        tools.configure_temporal_filter("cutoff")
