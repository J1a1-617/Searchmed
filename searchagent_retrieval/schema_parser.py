from __future__ import annotations

import json
import re
from calendar import monthrange
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .models import (
    CaseProfile,
    CitationAnchor,
    Document,
    EntitySet,
    EvidenceChunk,
    TimelineEvent,
    UnifiedCaseRecord,
)
from .text_utils import chunk_text, normalize_text

GENE_PATTERN = re.compile(r"\b[A-Z0-9]{2,8}\b")
REFERENCE_SECTION_HEADING = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\d+(?:\.\d+)*[.)]?\s*)?"
    r"(?:references?(?:\s+and\s+notes)?|bibliography|"
    r"参\s*考\s*文\s*献|参\s*考\s*资\s*料)\s*:?[ \t]*$"
)
REFERENCE_POINTER = re.compile(
    r"(?i)(?:\[\s*(?:doi|pubmed)\s*\]|google\s+scholar|pmc\s+free\s+article)"
)

TRIAL_DOCUMENT_PATTERN = re.compile(
    r"\b(?:randomi[sz]ed|phase\s*[i1-4v]+|clinical\s+trial|controlled\s+trial)\b|临床试验|随机对照",
    re.IGNORECASE,
)
COHORT_DOCUMENT_PATTERN = re.compile(
    r"\b(?:cohort|retrospective|prospective|observational|real[- ]world|case[- ]control|cross[- ]sectional)\b|队列|回顾性|前瞻性|观察性|统计分析",
    re.IGNORECASE,
)
REVIEW_DOCUMENT_PATTERN = re.compile(
    r"\b(?:systematic\s+review|meta[- ]analysis|literature\s+review|review)\b|系统综述|荟萃分析|综述",
    re.IGNORECASE,
)
CASE_REPORT_DOCUMENT_PATTERN = re.compile(
    r"\b(?:case\s+report|case\s+presentation|a\s+case\s+of)\b|病例报告|个案报道",
    re.IGNORECASE,
)

PUBLICATION_DATE_META_PATTERN = re.compile(
    r'<meta\b[^>]*\bname=["\']citation_publication_date["\'][^>]*\bcontent=["\']([^"\']+)["\'][^>]*>',
    re.IGNORECASE,
)
PUBLICATION_DATE_META_REVERSED_PATTERN = re.compile(
    r'<meta\b[^>]*\bcontent=["\']([^"\']+)["\'][^>]*\bname=["\']citation_publication_date["\'][^>]*>',
    re.IGNORECASE,
)


def _normalize_publication_date(value: object) -> Optional[str]:
    """Normalize bibliographic dates conservatively for cutoff filtering.

    Month- or year-only dates use the end of that period. This prevents an
    article with unknown publication day from leaking into a cutoff earlier
    in the same month/year.
    """
    raw = re.sub(r"\s+", " ", str(value or "").strip())
    if not raw:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw
    match = re.fullmatch(r"(\d{4})[-/](\d{1,2})", raw)
    if match:
        year, month = (int(part) for part in match.groups())
        return f"{year:04d}-{month:02d}-{monthrange(year, month)[1]:02d}"
    if re.fullmatch(r"\d{4}", raw):
        return f"{int(raw):04d}-12-31"
    for fmt in ("%Y %b", "%Y %B", "%b %Y", "%B %Y"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return f"{parsed.year:04d}-{parsed.month:02d}-{monthrange(parsed.year, parsed.month)[1]:02d}"
        except ValueError:
            pass
    for fmt in ("%Y %b %d", "%Y %B %d", "%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def _publication_date_from_source(step1: Dict) -> Optional[str]:
    """Read the article publication date, never a patient timeline date."""
    source_file = str(step1.get("source_file") or "").strip()
    if not source_file:
        return None
    path = Path(source_file)
    if not path.is_file():
        return None
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    match = PUBLICATION_DATE_META_PATTERN.search(html) or PUBLICATION_DATE_META_REVERSED_PATTERN.search(html)
    return _normalize_publication_date(match.group(1)) if match else None


def _infer_document_type(step1: Optional[Dict], step2: Optional[Dict], step3: Optional[Dict]) -> str:
    """Classify the source conservatively; the presence of Step1 does not imply a case report."""
    if step3:
        declared = str((step3.get("provenance") or {}).get("document_type") or "").strip()
        if declared:
            return declared
    if step2 and step2.get("is_case_report") is True:
        return "case_report"

    title = str(((step2 or {}).get("metadata") or {}).get("title") or (step1 or {}).get("title") or "")
    rejection = str((step2 or {}).get("rejection_reason") or "")
    # A negative Step2 decision is authoritative; use its reason/title to retain
    # the study design without relabelling the article as a case report.
    # The title is more specific than Step2's current generic rejection label
    # ("cohort/statistical analysis"), so classify it first.
    if TRIAL_DOCUMENT_PATTERN.search(title):
        return "clinical_trial"
    if REVIEW_DOCUMENT_PATTERN.search(title):
        return "review"
    if COHORT_DOCUMENT_PATTERN.search(title):
        return "cohort_or_observational_study"
    if TRIAL_DOCUMENT_PATTERN.search(rejection):
        return "clinical_trial"
    if REVIEW_DOCUMENT_PATTERN.search(rejection):
        return "review"
    if COHORT_DOCUMENT_PATTERN.search(rejection):
        return "cohort_or_observational_study"
    if not step2 or step2.get("is_case_report") is not False:
        if CASE_REPORT_DOCUMENT_PATTERN.search(title):
            return "case_report"
    return "other_literature"


def _text_evidence_level(document_type: Optional[str]) -> str:
    normalized = str(document_type or "").lower()
    if normalized == "case_report":
        return "case_report_evidence"
    if normalized == "clinical_trial":
        return "clinical_trial_evidence"
    if normalized == "cohort_or_observational_study":
        return "cohort_or_observational_evidence"
    if normalized == "review":
        return "review_evidence"
    return "other_literature_evidence"


def strip_reference_section(text: str) -> str:
    """Remove a trailing bibliography while preserving in-body citations."""
    matches = list(REFERENCE_SECTION_HEADING.finditer(text or ""))
    if not matches:
        return text or ""
    return (text or "")[: matches[-1].start()].rstrip()


def is_reference_list_chunk(text: str, *, min_pointers: int = 3) -> bool:
    """Detect citation-list chunks that lack a recognizable section heading."""
    return len(REFERENCE_POINTER.findall(text or "")) >= min_pointers


def _safe_load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _stem_from_file(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_tr_cleaned"):
        return stem[: -len("_tr_cleaned")]
    if stem.endswith("_tr"):
        return stem[: -len("_tr")]
    return stem


def _parse_event_types(raw_type: str) -> List[str]:
    if not raw_type:
        return ["other"]
    segments = [seg.strip().lower() for seg in raw_type.split("+")]
    normalized = []
    for seg in segments:
        if seg in {"diagnosis", "molecular", "treatment", "response", "progression", "toxicity", "metastasis", "ddi", "outcome", "other"}:
            normalized.append(seg)
            continue
        if "metasta" in seg:
            normalized.append("metastasis")
        elif "tox" in seg or "adverse" in seg:
            normalized.append("toxicity")
        elif "molec" in seg or "mutation" in seg:
            normalized.append("molecular")
        elif "diagn" in seg:
            normalized.append("diagnosis")
        elif "treat" in seg or "therapy" in seg:
            normalized.append("treatment")
        elif "resp" in seg:
            normalized.append("response")
        elif "prog" in seg:
            normalized.append("progression")
        else:
            normalized.append("other")
    if not normalized:
        normalized = ["other"]
    # dedupe with order preserved
    deduped: List[str] = []
    seen = set()
    for item in normalized:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _extract_genes_and_alterations(texts: Iterable[str]) -> Tuple[List[str], List[str]]:
    genes: List[str] = []
    alterations: List[str] = []
    seen_gene = set()
    seen_alteration = set()
    for text in texts:
        if not text:
            continue
        for match in GENE_PATTERN.findall(text):
            if len(match) > 1 and match.lower() not in seen_gene:
                seen_gene.add(match.lower())
                genes.append(match)
        lower = text.lower()
        if any(token in lower for token in ("mutation", "del", "fusion", "amplification", "ins", "exon")):
            key = text.strip().lower()
            if key not in seen_alteration:
                seen_alteration.add(key)
                alterations.append(text.strip())
    return genes, alterations


def _build_event_summary(case_profile: CaseProfile, event: Dict) -> str:
    """Create a Step3 event representation suitable for retrieval and citation."""
    parts: List[str] = []
    if case_profile.cancer_type:
        parts.append(case_profile.cancer_type)
    if case_profile.histology:
        parts.append(case_profile.histology)
    molecular = event.get("molecular") or {}
    treatment = event.get("treatment") or {}
    response = event.get("response") or {}
    alterations = molecular.get("alterations") or []
    if alterations:
        parts.append(" / ".join(alterations))
    line = treatment.get("line")
    if line:
        parts.append(f"{line}")
    regimen = treatment.get("regimen") or []
    if regimen:
        parts.append(" + ".join(regimen))
    best = response.get("best")
    if best:
        parts.append(f"response={best}")
    time = event.get("time")
    if time:
        parts.append(f"time={time}")
    dosage = treatment.get("dosage") or {}
    if dosage:
        parts.append("dosage=" + json.dumps(dosage, ensure_ascii=False, sort_keys=True))
    duration = response.get("duration_months")
    if duration is not None:
        parts.append(f"response_duration_months={duration}")
    safety = event.get("safety") or {}
    toxicities = event.get("toxicities") or []
    adverse_events = safety.get("adverse_events") or []
    toxicity_texts = [
        str(item.get("event") or item) if isinstance(item, dict) else str(item)
        for item in [*toxicities, *adverse_events]
        if item
    ]
    if toxicity_texts:
        parts.append("toxicities=" + " / ".join(dict.fromkeys(toxicity_texts)))
    dose_adjustments = safety.get("dose_adjustments") or []
    if dose_adjustments:
        parts.append("dose_adjustments=" + json.dumps(dose_adjustments, ensure_ascii=False))
    metastatic_sites = event.get("metastatic_sites") or []
    if metastatic_sites:
        parts.append("metastatic_sites=" + " / ".join(str(item) for item in metastatic_sites))
    return ", ".join(parts)


def _build_structured_case_summary(step3: Optional[Dict]) -> str:
    """Expose Step3-only baseline, outcome, and annotation facts as evidence."""
    if not step3:
        return ""
    sections: List[str] = []
    for key in ("baseline_clinical_profile", "outcome_summary", "knowledge_annotation"):
        value = step3.get(key) or {}
        if value:
            sections.append(f"{key}=" + json.dumps(value, ensure_ascii=False, sort_keys=True))
    return "\n".join(sections)


def _build_document(step1: Optional[Dict], step2: Optional[Dict], step3: Optional[Dict], doc_id: str) -> Document:
    source_layer = "step3_structured" if step3 else ("step2_cleaned" if step2 else "step1_extracted")
    pmid = None
    title = None
    date = None
    source_file = None
    source_url = None
    hostname = None
    document_type = None
    metadata: Dict = {}

    if step1:
        pmid = step1.get("pmid") or pmid
        title = step1.get("title") or title
        # Step1's historical ``date`` field was sometimes extracted from the
        # patient timeline. Prefer the publisher/PubMed citation metadata from
        # the source article so temporal retrieval filters publication time.
        source_publication_date = _publication_date_from_source(step1)
        date = source_publication_date or _normalize_publication_date(step1.get("date")) or date
        source_file = step1.get("source_file") or source_file
        source_url = step1.get("url") or source_url
        hostname = step1.get("hostname") or hostname
        metadata["step1_fingerprint"] = step1.get("fingerprint")
        metadata["publication_date_source"] = (
            "citation_publication_date" if source_publication_date else "step1_date_fallback"
        )

    if step2:
        prov = step2.get("provenance", {})
        md = step2.get("metadata", {})
        pmid = prov.get("source_pmid") or pmid
        title = md.get("title") or title
        # Step2 metadata describes the extracted case and may contain a
        # diagnosis/treatment date. It must never overwrite a bibliographic
        # publication date already established from Step1.
        if not date:
            date = _normalize_publication_date(md.get("publication_date"))
        metadata["is_case_report"] = step2.get("is_case_report")
        metadata["rejection_reason"] = step2.get("rejection_reason")

    if step3:
        prov = step3.get("provenance", {})
        pmid = prov.get("source_pmid") or pmid
        source_file = prov.get("source_filename") or source_file
        metadata["step3_confidence_score"] = prov.get("confidence_score")

    document_type = _infer_document_type(step1, step2, step3)

    return Document(
        doc_id=doc_id,
        pmid=pmid if pmid else None,
        title=title,
        date=date,
        source_layer=source_layer,  # type: ignore[arg-type]
        source_file=source_file,
        source_url=source_url,
        hostname=hostname,
        language="en",
        document_type=document_type,
        metadata=metadata,
    )


def _build_case_profile(doc_id: str, step3: Optional[Dict]) -> CaseProfile:
    case_id = f"{doc_id}#case-1"
    if not step3:
        return CaseProfile(case_id=case_id, doc_id=doc_id)

    profile_raw = step3.get("baseline_clinical_profile")
    profile = profile_raw if isinstance(profile_raw, dict) else {}
    return CaseProfile(
        case_id=case_id,
        doc_id=doc_id,
        age=profile.get("age"),
        sex=profile.get("sex"),
        smoking_status=profile.get("smoking_status"),
        pack_years=profile.get("pack_years"),
        diagnosis_date=profile.get("diagnosis_date"),
        cancer_type=profile.get("cancer_type"),
        histology=profile.get("histology"),
        tnm_stage_system=profile.get("tnm_stage_system"),
        tnm_stage=profile.get("tnm_stage"),
        overall_stage=profile.get("overall_stage"),
        primary_site=profile.get("baseline_primary_site"),
        metastatic_sites_at_baseline=profile.get("metastatic_sites_at_baseline", []),
        presenting_symptoms=profile.get("presenting_symptoms", []),
        baseline_molecular_alterations=(
            profile.get("baseline_molecular_alterations", [])
            or [item for event in step3.get("timeline_events", []) for item in ((event.get("molecular") or {}).get("alterations") or [])]
        ),
        metadata={},
    )


def _build_timeline_events(doc: Document, profile: CaseProfile, step3: Optional[Dict]) -> List[TimelineEvent]:
    if not step3:
        return []
    events: List[TimelineEvent] = []
    for idx, event in enumerate(step3.get("timeline_events", []), start=1):
        event_id = f"{profile.case_id}#event-{idx:03d}"
        molecular = event.get("molecular") or {}
        treatment = event.get("treatment") or {}
        response = event.get("response") or {}
        metastasis = event.get("metastatic_sites") or []
        toks = event.get("toxicities") or []

        alterations = molecular.get("alterations") or []
        regimen = treatment.get("regimen") or []
        response_best = response.get("best")
        genes, gene_alterations = _extract_genes_and_alterations(alterations)

        entities = EntitySet(
            cancer_types=[profile.cancer_type] if profile.cancer_type else [],
            histologies=[profile.histology] if profile.histology else [],
            genes=genes,
            gene_alterations=gene_alterations,
            drugs=regimen,
            lines_of_therapy=[treatment.get("line")] if treatment.get("line") else [],
            responses=[response_best] if response_best else [],
            toxicities=[t.get("event") for t in toks if t.get("event")],
            metastatic_sites=metastasis,
            ddi_terms=[],
        ).normalize()

        events.append(
            TimelineEvent(
                event_id=event_id,
                doc_id=doc.doc_id,
                case_id=profile.case_id,
                time=str(event.get("time") or "unknown"),
                event_types=_parse_event_types(event.get("event_type", "")),
                molecular=molecular,
                treatment=treatment,
                response=response,
                toxicities=toks,
                metastatic_sites=metastasis,
                summary_text=_build_event_summary(profile, event),
                entities=entities,
                metadata={
                    "raw_event_type": event.get("event_type"),
                    "raw_step3_event": event,
                },
            )
        )
    return events


def _build_evidence_chunks(
    doc: Document,
    profile: CaseProfile,
    events: List[TimelineEvent],
    step1: Optional[Dict],
    step2: Optional[Dict],
    step3: Optional[Dict],
) -> List[EvidenceChunk]:
    chunks: List[EvidenceChunk] = []
    text_evidence_level = _text_evidence_level(doc.document_type)
    chunk_seq = 0

    def next_chunk_id() -> str:
        nonlocal chunk_seq
        chunk_seq += 1
        return f"{doc.doc_id}#chunk-{chunk_seq:06d}"

    if step2 and step2.get("text"):
        source_text = strip_reference_section(str(step2["text"]))
        for idx, text_chunk in enumerate(chunk_text(source_text, chunk_size=900, overlap=120)):
            if is_reference_list_chunk(text_chunk):
                continue
            chunks.append(
                EvidenceChunk(
                    chunk_id=next_chunk_id(),
                    doc_id=doc.doc_id,
                    case_id=profile.case_id,
                    chunk_type="case_text_chunk",
                    text=text_chunk,
                    text_for_embedding=text_chunk,
                    evidence_level=text_evidence_level,  # type: ignore[arg-type]
                    citation=CitationAnchor(
                        source_layer="step2_cleaned",
                        source_file=(step2.get("provenance") or {}).get("source_filename"),
                        field_path="text",
                        chunk_order=idx,
                    ),
                )
            )
    elif step1 and step1.get("text"):
        source_text = strip_reference_section(str(step1["text"]))
        for idx, text_chunk in enumerate(chunk_text(source_text, chunk_size=900, overlap=120)):
            if is_reference_list_chunk(text_chunk):
                continue
            chunks.append(
                EvidenceChunk(
                    chunk_id=next_chunk_id(),
                    doc_id=doc.doc_id,
                    case_id=profile.case_id,
                    chunk_type="case_text_chunk",
                    text=text_chunk,
                    text_for_embedding=text_chunk,
                    evidence_level=text_evidence_level,  # type: ignore[arg-type]
                    citation=CitationAnchor(
                        source_layer="step1_extracted",
                        source_file=step1.get("source_file"),
                        field_path="text",
                        chunk_order=idx,
                    ),
                )
            )

    for event in events:
        chunks.append(
            EvidenceChunk(
                chunk_id=next_chunk_id(),
                doc_id=doc.doc_id,
                case_id=profile.case_id,
                event_id=event.event_id,
                chunk_type="event_chunk",
                text=event.summary_text or "",
                text_for_embedding=event.summary_text or "",
                evidence_level="structured_extraction_evidence",
                entities=event.entities,
                citation=CitationAnchor(
                    source_layer="step3_structured",
                    source_file=(step3.get("provenance") or {}).get("source_filename") if step3 else None,
                    field_path=f"timeline_events[{event.event_id.split('-')[-1]}]",
                ),
            )
        )
    structured_case_summary = _build_structured_case_summary(step3)
    if structured_case_summary:
        chunks.append(
            EvidenceChunk(
                chunk_id=next_chunk_id(),
                doc_id=doc.doc_id,
                case_id=profile.case_id,
                chunk_type="structured_field_chunk",
                text=structured_case_summary,
                text_for_embedding=structured_case_summary,
                evidence_level="structured_extraction_evidence",
                citation=CitationAnchor(
                    source_layer="step3_structured",
                    source_file=(step3.get("provenance") or {}).get("source_filename") if step3 else None,
                    field_path="baseline_clinical_profile,outcome_summary,knowledge_annotation",
                ),
            )
        )
    return chunks


def parse_unified_case_record(
    step1_path: Path,
    step2_path: Path,
    step3_path: Path,
) -> Optional[UnifiedCaseRecord]:
    step1 = _safe_load_json(step1_path)
    step2 = _safe_load_json(step2_path)
    step3 = _safe_load_json(step3_path)
    if not step1 and not step2 and not step3:
        return None

    doc_id = _stem_from_file(step1_path if step1 else (step2_path if step2 else step3_path))
    document = _build_document(step1=step1, step2=step2, step3=step3, doc_id=doc_id)
    profile = _build_case_profile(doc_id=doc_id, step3=step3)
    events = _build_timeline_events(doc=document, profile=profile, step3=step3)
    chunks = _build_evidence_chunks(
        doc=document,
        profile=profile,
        events=events,
        step1=step1,
        step2=step2,
        step3=step3,
    )
    return UnifiedCaseRecord(
        document=document,
        case_profile=profile,
        timeline_events=events,
        evidence_chunks=chunks,
    )


def load_records_from_data_root(data_root: Path) -> List[UnifiedCaseRecord]:
    step1_dir = data_root / "step1_extracted"
    step2_dir = data_root / "step2_cleaned"
    step3_dir = data_root / "step3_structured"
    stems = set()
    for path in step1_dir.glob("*.json"):
        stems.add(_stem_from_file(path))
    for path in step2_dir.glob("*.json"):
        stems.add(_stem_from_file(path))
    for path in step3_dir.glob("*.json"):
        stems.add(_stem_from_file(path))

    records: List[UnifiedCaseRecord] = []
    for stem in sorted(stems):
        step1_path = step1_dir / f"{stem}_tr.json"
        step2_path = step2_dir / f"{stem}_tr_cleaned.json"
        step3_path = step3_dir / f"{stem}_tr_cleaned.json.json"
        record = parse_unified_case_record(step1_path=step1_path, step2_path=step2_path, step3_path=step3_path)
        if record:
            records.append(record)
    return records


def record_to_event_text(event: TimelineEvent) -> str:
    parts = [event.summary_text or "", event.entities.as_text()]
    return normalize_text(" | ".join([p for p in parts if p]))
