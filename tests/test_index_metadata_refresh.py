import json

from searchagent_retrieval.index_builder import _refresh_vector_item_metadata


def test_refresh_vector_metadata_keeps_vectors_and_rank_items(tmp_path) -> None:
    vector_dir = tmp_path / "vector"
    vector_dir.mkdir()
    metadata_path = vector_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps({
            "space_items": {
                "case_semantic": [
                    {"doc_id": "d1#chunk-1", "text": "body", "metadata": {"evidence_level": "case_report_evidence"}}
                ]
            }
        }),
        encoding="utf-8",
    )

    count = _refresh_vector_item_metadata(
        vector_dir,
        {"d1#chunk-1": {"evidence_level": "review_evidence", "document_type": "review"}},
    )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    item = payload["space_items"]["case_semantic"][0]

    assert count == 1
    assert item["doc_id"] == "d1#chunk-1"
    assert item["text"] == "body"
    assert item["metadata"]["evidence_level"] == "review_evidence"
    assert item["metadata"]["document_type"] == "review"
