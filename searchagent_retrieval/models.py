from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


SourceLayer = Literal[
    "step1_extracted",
    "step2_cleaned",
    "step3_structured",
    "drug_label_or_ddi_rule",
    "external_guideline",
    "clinical_trial",
    "model_prior_knowledge",
]

EventType = Literal[
    "diagnosis",
    "molecular",
    "treatment",
    "response",
    "progression",
    "toxicity",
    "metastasis",
    "ddi",
    "outcome",
    "other",
]

ChunkType = Literal[
    "case_text_chunk",
    "event_chunk",
    "structured_field_chunk",
    "ddi_rule_chunk",
]

EvidenceLevel = Literal[
    "case_report_evidence",
    "cohort_or_observational_evidence",
    "review_evidence",
    "other_literature_evidence",
    "structured_extraction_evidence",
    "drug_label_or_ddi_rule_evidence",
    "guideline_evidence",
    "clinical_trial_evidence",
    "model_prior_knowledge",
]


def _normalize_list(value: Optional[List[str]]) -> List[str]:
    if not value:
        return []
    dedup: List[str] = []
    seen = set()
    for item in value:
        if not item:
            continue
        key = item.strip()
        if not key:
            continue
        if key.lower() in seen:
            continue
        seen.add(key.lower())
        dedup.append(key)
    return dedup


@dataclass
class EntitySet:
    cancer_types: List[str] = field(default_factory=list)
    histologies: List[str] = field(default_factory=list)
    genes: List[str] = field(default_factory=list)
    gene_alterations: List[str] = field(default_factory=list)
    drugs: List[str] = field(default_factory=list)
    lines_of_therapy: List[str] = field(default_factory=list)
    responses: List[str] = field(default_factory=list)
    toxicities: List[str] = field(default_factory=list)
    metastatic_sites: List[str] = field(default_factory=list)
    ddi_terms: List[str] = field(default_factory=list)

    def normalize(self) -> "EntitySet":
        self.cancer_types = _normalize_list(self.cancer_types)
        self.histologies = _normalize_list(self.histologies)
        self.genes = _normalize_list(self.genes)
        self.gene_alterations = _normalize_list(self.gene_alterations)
        self.drugs = _normalize_list(self.drugs)
        self.lines_of_therapy = _normalize_list(self.lines_of_therapy)
        self.responses = _normalize_list(self.responses)
        self.toxicities = _normalize_list(self.toxicities)
        self.metastatic_sites = _normalize_list(self.metastatic_sites)
        self.ddi_terms = _normalize_list(self.ddi_terms)
        return self

    def as_text(self) -> str:
        parts: List[str] = []
        for name, values in asdict(self).items():
            if values:
                parts.append(f"{name}: {', '.join(values)}")
        return " | ".join(parts)


@dataclass
class Document:
    doc_id: str
    source_layer: SourceLayer
    pmid: Optional[str] = None
    title: Optional[str] = None
    date: Optional[str] = None
    source_file: Optional[str] = None
    source_url: Optional[str] = None
    hostname: Optional[str] = None
    language: Optional[str] = None
    document_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseProfile:
    case_id: str
    doc_id: str
    age: Optional[int] = None
    sex: Optional[str] = None
    smoking_status: Optional[str] = None
    pack_years: Optional[float] = None
    diagnosis_date: Optional[str] = None
    cancer_type: Optional[str] = None
    histology: Optional[str] = None
    tnm_stage_system: Optional[str] = None
    tnm_stage: Optional[str] = None
    overall_stage: Optional[str] = None
    primary_site: Optional[str] = None
    metastatic_sites_at_baseline: List[str] = field(default_factory=list)
    presenting_symptoms: List[str] = field(default_factory=list)
    baseline_molecular_alterations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimelineEvent:
    event_id: str
    doc_id: str
    case_id: str
    time: str
    event_types: List[EventType]
    molecular: Dict[str, Any] = field(default_factory=dict)
    treatment: Dict[str, Any] = field(default_factory=dict)
    response: Dict[str, Any] = field(default_factory=dict)
    toxicities: List[Dict[str, Any]] = field(default_factory=list)
    metastatic_sites: List[str] = field(default_factory=list)
    summary_text: Optional[str] = None
    entities: EntitySet = field(default_factory=EntitySet)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CitationAnchor:
    source_layer: SourceLayer
    source_file: Optional[str] = None
    field_path: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    chunk_order: Optional[int] = None


@dataclass
class EvidenceChunk:
    chunk_id: str
    doc_id: str
    case_id: str
    chunk_type: ChunkType
    text: str
    evidence_level: EvidenceLevel
    citation: CitationAnchor
    event_id: Optional[str] = None
    text_for_embedding: Optional[str] = None
    language: Optional[str] = None
    entities: EntitySet = field(default_factory=EntitySet)
    relevance_signals: Dict[str, bool] = field(
        default_factory=lambda: {
            "is_supporting": False,
            "is_contradicting": False,
            "is_safety_risk": False,
        }
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedCaseRecord:
    document: Document
    case_profile: CaseProfile
    timeline_events: List[TimelineEvent]
    evidence_chunks: List[EvidenceChunk]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document": asdict(self.document),
            "case_profile": asdict(self.case_profile),
            "timeline_events": [asdict(event) for event in self.timeline_events],
            "evidence_chunks": [asdict(chunk) for chunk in self.evidence_chunks],
        }
