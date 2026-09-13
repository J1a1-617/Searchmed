from searchagent_retrieval.tools import RetrievalTools


def _row(chunk_id: str, text: str) -> dict:
    return {"chunk_id": chunk_id, "doc_id": "doc-1", "chunk_type": "case_text_chunk", "text": text}


def test_early_response_target_does_not_let_long_term_resistance_displace_response() -> None:
    rows = [
        _row("resistance-1", "After 18 months the disease developed acquired resistance and progressed."),
        _row("resistance-2", "Long-term follow-up showed further progression after resistance."),
        _row("early", "At the first assessment after 6 weeks, intracranial lesions achieved partial response and symptoms improved."),
        _row("toxicity", "Treatment caused grade 2 rash as an adverse event."),
    ]
    selected = RetrievalTools._select_event_aligned_details(
        rows,
        query="EGFR lung cancer brain metastasis osimertinib early intracranial response",
        locator_texts=["treatment response: osimertinib partial response"],
        locator_event_types=["treatment", "response"],
        per_document=3,
    )
    ids = [row[1]["chunk_id"] for row in selected]
    slots = [row[2] for row in selected]
    assert ids[0] == "early"
    assert slots.count("early_response") <= 1
    assert slots.count("target_outcome") <= 1
    assert slots.count("toxicity_or_counter") <= 1
    assert not ({"resistance-1", "resistance-2"} <= set(ids))


def test_progression_target_can_select_resistance_as_target_outcome() -> None:
    rows = [
        _row("early", "Initial partial response was documented."),
        _row("resistance", "After 18 months acquired resistance caused disease progression."),
    ]
    selected = RetrievalTools._select_event_aligned_details(
        rows, query="osimertinib acquired resistance progression", locator_texts=[],
        locator_event_types=["progression"], per_document=3,
    )
    assert selected[0][1]["chunk_id"] == "resistance"
    assert selected[0][2] == "target_outcome"
