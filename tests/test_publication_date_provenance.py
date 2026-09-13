import json
from pathlib import Path

from searchagent_retrieval.schema_parser import parse_unified_case_record


def test_article_meta_date_overrides_patient_timeline_date(tmp_path: Path) -> None:
    html = tmp_path / "123.html"
    html.write_text(
        '<html><head><meta name="citation_publication_date" content="2021 Aug"></head></html>',
        encoding="utf-8",
    )
    step1 = tmp_path / "123_tr.json"
    step1.write_text(
        json.dumps(
            {
                "pmid": "123",
                "title": "Example",
                "date": "2016-06-01",
                "source_file": str(html),
                "text": "A patient was diagnosed in June 2016.",
            }
        ),
        encoding="utf-8",
    )

    step2 = tmp_path / "123_tr_cleaned.json"
    step2.write_text(
        json.dumps(
            {
                "provenance": {"source_pmid": "123"},
                "metadata": {"date": "2015-09-01"},
                "cleaned_text": "The patient began treatment in September 2015.",
            }
        ),
        encoding="utf-8",
    )

    record = parse_unified_case_record(step1, step2, tmp_path / "missing3.json")

    assert record is not None
    assert record.document.date == "2021-08-31"
    assert record.document.metadata["publication_date_source"] == "citation_publication_date"


def test_missing_source_uses_normalized_step1_date(tmp_path: Path) -> None:
    step1 = tmp_path / "123_tr.json"
    step1.write_text(
        json.dumps({"pmid": "123", "date": "2020-02", "text": "text"}),
        encoding="utf-8",
    )

    record = parse_unified_case_record(step1, tmp_path / "missing2.json", tmp_path / "missing3.json")

    assert record is not None
    assert record.document.date == "2020-02-29"
    assert record.document.metadata["publication_date_source"] == "step1_date_fallback"
