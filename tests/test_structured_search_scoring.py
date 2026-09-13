from pathlib import Path

from searchagent_retrieval.structured_index import StructuredIndex


def _build_index(tmp_path: Path) -> StructuredIndex:
    index = StructuredIndex(tmp_path / "structured.db")
    for number in range(1, 4):
        index.conn.execute("INSERT INTO documents(doc_id, title) VALUES(?, ?)", (f"d{number}", f"case {number}"))
        index.conn.execute("INSERT INTO case_profiles(case_id, doc_id, cancer_type) VALUES(?, ?, ?)", (f"c{number}", f"d{number}", "NSCLC"))
        index.conn.execute("INSERT INTO timeline_events(event_id, doc_id, case_id) VALUES(?, ?, ?)", (f"e{number}", f"d{number}", f"c{number}"))
    index.conn.executemany(
        "INSERT INTO entity_index(ref_id, ref_type, entity_type, entity_value) VALUES(?, 'event', ?, ?)",
        [("e1", "drugs", "osimertinib"), ("e1", "responses", "pr"), ("e2", "drugs", "osimertinib")],
    )
    index.conn.commit()
    return index


def test_and_ranks_above_or_and_zero_match_is_excluded(tmp_path: Path) -> None:
    index = _build_index(tmp_path)
    try:
        rows = index.structured_search({"cancer_type": "NSCLC", "drugs": ["osimertinib"], "responses": ["PR"]}, limit=10)
    finally:
        index.close()
    assert [row["case_id"] for row in rows] == ["c1", "c2"]
    assert rows[0]["structured_match_mode"] == "and"
    assert rows[0]["structured_score"] == 1.0
    assert rows[1]["structured_match_mode"] == "or"
    assert 0.0 < rows[1]["structured_score"] < 1.0
    assert rows[1]["constraint_coverage"] == 0.666667


def test_unconstrained_browse_has_zero_score(tmp_path: Path) -> None:
    index = _build_index(tmp_path)
    try:
        rows = index.structured_search({}, limit=10)
    finally:
        index.close()
    assert rows
    assert {row["structured_score"] for row in rows} == {0.0}
    assert {row["structured_match_mode"] for row in rows} == {"unconstrained"}


def test_cancer_type_uses_canonical_compatible_matching(tmp_path: Path) -> None:
    index = _build_index(tmp_path)
    try:
        rows = index.structured_search(
            {"cancer_type": "非小细胞肺癌（肺腺癌）IV期", "drugs": ["osimertinib"]},
            limit=10,
        )
    finally:
        index.close()

    assert [row["case_id"] for row in rows] == ["c2", "c1"]
    assert {row["cancer_type_match_mode"] for row in rows} <= {
        "normalized_compatible",
        "canonical_compatible",
    }


def test_cancer_type_does_not_merge_conflicting_histologies(tmp_path: Path) -> None:
    index = _build_index(tmp_path)
    try:
        index.conn.execute(
            "UPDATE case_profiles SET cancer_type = 'lung squamous cell carcinoma'"
        )
        index.conn.commit()
        rows = index.structured_search(
            {"cancer_type": "肺腺癌", "drugs": ["osimertinib"]},
            limit=10,
        )
    finally:
        index.close()

    assert rows == []


def test_cutoff_is_applied_inside_structured_search_and_evidence_fetch(tmp_path: Path) -> None:
    index = _build_index(tmp_path)
    try:
        index.conn.executemany(
            "UPDATE documents SET date = ?, pmid = ?, metadata_json = ? WHERE doc_id = ?",
            [
                ("2021-06-01", "1", '{"publication_date_source":"citation_publication_date"}', "d1"),
                ("2024-06-01", "2", '{"publication_date_source":"citation_publication_date"}', "d2"),
                (None, "3", '{"publication_date_source":"citation_publication_date"}', "d3"),
            ],
        )
        index.conn.executemany(
            """
            INSERT INTO evidence_chunks(
                chunk_id, doc_id, case_id, chunk_type, evidence_level, text
            ) VALUES(?, ?, ?, 'case_text_chunk', 'case_report_evidence', ?)
            """,
            [
                ("ch1", "d1", "c1", "past evidence"),
                ("ch2", "d2", "c2", "future evidence"),
                ("ch3", "d3", "c3", "undated evidence"),
            ],
        )
        index.conn.commit()

        cases = index.structured_search({}, cutoff="2022-12-31", limit=10)
        evidence = index.fetch_evidence(
            chunk_ids=["ch1", "ch2", "ch3"],
            cutoff="2022-12-31",
            limit=10,
        )
    finally:
        index.close()

    assert [row["case_id"] for row in cases] == ["c1"]
    assert [row["chunk_id"] for row in evidence] == ["ch1"]
    assert evidence[0]["pub_date"] == "2021-06-01"
    assert evidence[0]["pmid"] == "1"


def test_fetch_preserves_ranked_chunk_and_event_order(tmp_path: Path) -> None:
    index = _build_index(tmp_path)
    try:
        index.conn.executemany(
            """
            INSERT INTO evidence_chunks(
                chunk_id, doc_id, case_id, event_id, chunk_type, evidence_level, text
            ) VALUES(?, ?, ?, ?, 'case_text_chunk', 'case_report_evidence', ?)
            """,
            [
                ("ch1", "d1", "c1", "e1", "one"),
                ("ch2", "d2", "c2", "e2", "two"),
                ("ch3", "d3", "c3", "e3", "three"),
            ],
        )
        index.conn.commit()
        evidence = index.fetch_evidence(chunk_ids=["ch3", "ch1", "ch2"], limit=2)
        events = index.fetch_structured_events(["e3", "e1", "e2"], limit=2)
    finally:
        index.close()

    assert [row["chunk_id"] for row in evidence] == ["ch3", "ch1"]
    assert [row["event_id"] for row in events] == ["e3", "e1"]
