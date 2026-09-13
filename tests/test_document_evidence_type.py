from searchagent_retrieval.schema_parser import _build_document, _text_evidence_level


def test_rejected_step2_cohort_is_not_relabelled_as_case_report() -> None:
    document = _build_document(
        step1={"title": "A retrospective cohort of patients with NSCLC"},
        step2={
            "is_case_report": False,
            "rejection_reason": "这是队列研究和统计分析，非个案报道",
            "metadata": {},
        },
        step3=None,
        doc_id="cohort-1",
    )

    assert document.document_type == "cohort_or_observational_study"
    assert _text_evidence_level(document.document_type) == "cohort_or_observational_evidence"


def test_rejected_step2_review_is_not_relabelled_as_case_report() -> None:
    document = _build_document(
        step1={"title": "Systematic review of EGFR inhibitors"},
        step2={"is_case_report": False, "rejection_reason": "综述", "metadata": {}},
        step3=None,
        doc_id="review-1",
    )

    assert document.document_type == "review"
    assert _text_evidence_level(document.document_type) == "review_evidence"


def test_positive_step2_case_report_remains_case_evidence() -> None:
    document = _build_document(
        step1={"title": "Clinical response in one patient"},
        step2={"is_case_report": True, "metadata": {}},
        step3=None,
        doc_id="case-1",
    )

    assert document.document_type == "case_report"
    assert _text_evidence_level(document.document_type) == "case_report_evidence"
