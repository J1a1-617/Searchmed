import json
from pathlib import Path

from predictive_clinical_benchmark.eval.prompts import construct_agent_query, construct_inference_prompt
from searchagent_retrieval.schema_parser import parse_unified_case_record
from searchagent_retrieval.structured_index import StructuredIndex


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_step3_event_and_case_fields_become_retrievable_evidence(tmp_path: Path) -> None:
    step1 = tmp_path / "case_tr.json"
    step2 = tmp_path / "case_tr_cleaned.json"
    step3 = tmp_path / "case_tr_cleaned.json.json"
    _write_json(step1, {"title": "Case", "source_file": "case.json"})
    _write_json(step2, {"text": "Long source article.", "provenance": {"source_filename": "case.json"}})
    _write_json(step3, {
        "provenance": {"source_filename": "case.json"},
        "baseline_clinical_profile": {"cancer_type": "NSCLC", "histology": "adenocarcinoma"},
        "timeline_events": [{
            "time": "2024-01", "event_type": "treatment + response + toxicity",
            "molecular": {"alterations": ["EGFR L858R"]},
            "treatment": {"line": "1L", "regimen": ["osimertinib"], "dosage": {"osimertinib": "80 mg qd"}},
            "response": {"best": "PR", "duration_months": 12},
            "safety": {"adverse_events": ["Grade 1 rash"], "dose_adjustments": ["none"]},
            "metastatic_sites": ["brain"],
        }],
        "outcome_summary": {"survival_months": 24},
        "knowledge_annotation": {"key_findings": ["durable response"]},
    })

    record = parse_unified_case_record(step1, step2, step3)
    assert record is not None
    event_chunk = next(chunk for chunk in record.evidence_chunks if chunk.event_id)
    assert "80 mg qd" in event_chunk.text
    assert "response_duration_months=12" in event_chunk.text
    assert "Grade 1 rash" in event_chunk.text
    structured_chunk = next(chunk for chunk in record.evidence_chunks if chunk.chunk_type == "structured_field_chunk")
    assert "survival_months" in structured_chunk.text

    index = StructuredIndex(tmp_path / "structured.db")
    index.add_record(record)
    rows = index.fetch_structured_events([event_chunk.event_id or ""])
    index.close()
    assert rows[0]["raw_json"]
    assert "safety" in rows[0]["raw_json"]
    assert "osimertinib" in rows[0]["summary_text"]


def test_benchmark_agent_query_excludes_output_schema() -> None:
    instance = {
        "time_cutoff": "2025-01-01",
        "input": {
            "disease_background": {"diagnosis": "NSCLC", "metastatic_sites": [], "molecular_profile": {"primary_mutation": "EGFR L858R"}},
            "prior_treatment_timeline": [],
            "current_status": {"symptoms": [], "imaging": "N/A", "csf": "N/A", "performance_status": "0"},
            "planned_treatment": {"drugs": [{"name": "osimertinib", "dose": "80 mg", "route": "po"}]},
        },
    }
    full_prompt = construct_inference_prompt(instance)
    agent_query = construct_agent_query(instance)
    assert "overall_benefit" in full_prompt
    assert "overall_benefit" not in agent_query
    assert "EGFR L858R" in agent_query
