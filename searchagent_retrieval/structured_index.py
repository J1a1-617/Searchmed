from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import EvidenceChunk, TimelineEvent, UnifiedCaseRecord


def _normalize_cancer_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("非小细胞肺癌", " nsclc ")
    text = text.replace("非小细胞癌", " nsclc ")
    text = text.replace("肺腺癌", " lung adenocarcinoma ")
    text = text.replace("肺鳞癌", " lung squamous ")
    text = text.replace("小细胞肺癌", " sclc ")
    text = re.sub(r"\b(?:stage|期|分期)\s*(?:i{1,3}|iv|[1-4])\b", " ", text)
    return " ".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text))


def _cancer_labels(value: Any) -> set[str]:
    text = _normalize_cancer_text(value)
    labels: set[str] = set()
    if not text:
        return labels
    if any(term in text for term in ("lung adenocarcinoma", "luad")):
        labels.update({"lung", "nsclc", "luad"})
    elif any(term in text for term in ("lung squamous", "lusc")):
        labels.update({"lung", "nsclc", "lusc"})
    elif "nsclc" in text or "non small cell lung" in text:
        labels.update({"lung", "nsclc"})
    elif "sclc" in text or "small cell lung" in text:
        labels.update({"lung", "sclc"})
    elif "lung" in text or "肺癌" in text:
        labels.add("lung")
    labels.add(text)
    return labels


def _cancer_type_match(query_value: Any, stored_value: Any) -> tuple[bool, str]:
    query_text = _normalize_cancer_text(query_value)
    stored_text = _normalize_cancer_text(stored_value)
    if not query_text:
        return True, "unconstrained"
    if not stored_text:
        return False, "missing"
    if query_text == stored_text:
        return True, "normalized_exact"
    if len(query_text) >= 4 and len(stored_text) >= 4 and (
        query_text in stored_text or stored_text in query_text
    ):
        return True, "normalized_compatible"
    query_labels = _cancer_labels(query_value)
    stored_labels = _cancer_labels(stored_value)
    # Never collapse adenocarcinoma and squamous carcinoma merely because both
    # belong to NSCLC.
    query_histology = query_labels & {"luad", "lusc"}
    stored_histology = stored_labels & {"luad", "lusc"}
    if query_histology and stored_histology and query_histology != stored_histology:
        return False, "histology_conflict"
    if query_labels & stored_labels & {"luad", "lusc"}:
        return True, "canonical_exact"
    if "nsclc" in query_labels and "nsclc" in stored_labels:
        return True, "canonical_compatible"
    if query_labels == {"lung", query_text} and "lung" in stored_labels:
        return True, "organ_compatible"
    return False, "different_cancer"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  pmid TEXT,
  title TEXT,
  date TEXT,
  source_layer TEXT,
  source_file TEXT,
  document_type TEXT,
  metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS case_profiles (
  case_id TEXT PRIMARY KEY,
  doc_id TEXT,
  cancer_type TEXT,
  histology TEXT,
  overall_stage TEXT,
  tnm_stage TEXT,
  diagnosis_date TEXT,
  metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS timeline_events (
  event_id TEXT PRIMARY KEY,
  doc_id TEXT,
  case_id TEXT,
  event_time TEXT,
  event_types_json TEXT,
  summary_text TEXT,
  entities_json TEXT,
  raw_json TEXT
);

CREATE TABLE IF NOT EXISTS evidence_chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT,
  case_id TEXT,
  event_id TEXT,
  chunk_type TEXT,
  evidence_level TEXT,
  text TEXT,
  entities_json TEXT,
  citation_json TEXT,
  metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS entity_index (
  ref_id TEXT,
  ref_type TEXT,
  entity_type TEXT,
  entity_value TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title);
CREATE INDEX IF NOT EXISTS idx_case_profiles_doc_id ON case_profiles(doc_id);
CREATE INDEX IF NOT EXISTS idx_events_case_id ON timeline_events(case_id);
CREATE INDEX IF NOT EXISTS idx_events_doc_id ON timeline_events(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON evidence_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_case_id ON evidence_chunks(case_id);
CREATE INDEX IF NOT EXISTS idx_chunks_event_id ON evidence_chunks(event_id);
CREATE INDEX IF NOT EXISTS idx_entity_idx ON entity_index(entity_type, entity_value);
"""


class StructuredIndex:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        document_columns = {
            str(row["name"])
            for row in self.conn.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "document_type" not in document_columns:
            self.conn.execute("ALTER TABLE documents ADD COLUMN document_type TEXT")
        self.conn.commit()

    def reset(self) -> None:
        self.conn.executescript(
            """
            DELETE FROM documents;
            DELETE FROM case_profiles;
            DELETE FROM timeline_events;
            DELETE FROM evidence_chunks;
            DELETE FROM entity_index;
            """
        )
        self.conn.commit()

    def add_record(self, record: UnifiedCaseRecord) -> None:
        doc = record.document
        profile = record.case_profile
        self.conn.execute(
            """
            INSERT OR REPLACE INTO documents(doc_id, pmid, title, date, source_layer, source_file, document_type, metadata_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc.doc_id,
                doc.pmid,
                doc.title,
                doc.date,
                doc.source_layer,
                doc.source_file,
                doc.document_type,
                json.dumps(doc.metadata, ensure_ascii=False),
            ),
        )
        self.conn.execute(
            """
            INSERT OR REPLACE INTO case_profiles(case_id, doc_id, cancer_type, histology, overall_stage, tnm_stage, diagnosis_date, metadata_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile.case_id,
                profile.doc_id,
                profile.cancer_type,
                profile.histology,
                profile.overall_stage,
                profile.tnm_stage,
                profile.diagnosis_date,
                json.dumps(profile.metadata, ensure_ascii=False),
            ),
        )
        for event in record.timeline_events:
            self._add_event(event)
        for chunk in record.evidence_chunks:
            self._add_chunk(chunk)
        self.conn.commit()

    def _add_event(self, event: TimelineEvent) -> None:
        raw_event = event.metadata.get("raw_step3_event") if isinstance(event.metadata, dict) else None
        if raw_event is None:
            raw_event = asdict(event)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO timeline_events(event_id, doc_id, case_id, event_time, event_types_json, summary_text, entities_json, raw_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.doc_id,
                event.case_id,
                event.time,
                json.dumps(event.event_types, ensure_ascii=False),
                event.summary_text,
                json.dumps(asdict(event.entities), ensure_ascii=False),
                json.dumps(raw_event, ensure_ascii=False),
            ),
        )
        self._index_entities(ref_id=event.event_id, ref_type="event", entity_dict=asdict(event.entities))

    def _add_chunk(self, chunk: EvidenceChunk) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO evidence_chunks(chunk_id, doc_id, case_id, event_id, chunk_type, evidence_level, text, entities_json, citation_json, metadata_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.case_id,
                chunk.event_id,
                chunk.chunk_type,
                chunk.evidence_level,
                chunk.text,
                json.dumps(asdict(chunk.entities), ensure_ascii=False),
                json.dumps(asdict(chunk.citation), ensure_ascii=False),
                json.dumps(chunk.metadata, ensure_ascii=False),
            ),
        )
        self._index_entities(ref_id=chunk.chunk_id, ref_type="chunk", entity_dict=asdict(chunk.entities))

    def _index_entities(self, ref_id: str, ref_type: str, entity_dict: Dict[str, List[str]]) -> None:
        for entity_type, values in entity_dict.items():
            for value in values:
                self.conn.execute(
                    """
                    INSERT INTO entity_index(ref_id, ref_type, entity_type, entity_value)
                    VALUES(?, ?, ?, ?)
                    """,
                    (ref_id, ref_type, entity_type, value.lower()),
                )

    def structured_search(
        self,
        constraints: Dict[str, Any],
        limit: int = 50,
        cutoff: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Apply hard profile AND filters, then rank by entity coverage."""
        clauses = []
        params: List[Any] = []
        if cutoff:
            # Benchmark temporal mode is deliberately strict: undated documents
            # or dates not derived from bibliographic metadata cannot be proven
            # to have existed at the prediction time.
            clauses.append("d.date IS NOT NULL AND d.date != '' AND d.date <= ?")
            clauses.append("json_extract(d.metadata_json, '$.publication_date_source') = 'citation_publication_date'")
            params.append(cutoff)
        cancer_type = constraints.get("cancer_type")
        for key in ("histology", "overall_stage"):
            value = constraints.get(key)
            if value:
                clauses.append(f"cp.{key} = ?")
                params.append(value)

        base_sql = """
        SELECT cp.case_id, cp.doc_id, d.title, d.pmid, d.date AS pub_date,
               cp.cancer_type, cp.histology, cp.overall_stage
        FROM case_profiles cp
        JOIN documents d ON d.doc_id = cp.doc_id
        """
        if clauses:
            base_sql += " WHERE " + " AND ".join(clauses)
        rows = self.conn.execute(base_sql, params).fetchall()
        results = [dict(row) for row in rows]
        if cancer_type:
            compatible = []
            for item in results:
                matched, mode = _cancer_type_match(cancer_type, item.get("cancer_type"))
                if matched:
                    item["cancer_type_match_mode"] = mode
                    compatible.append(item)
            results = compatible

        entity_terms = {
            "gene_alterations": constraints.get("gene_alterations", []),
            "drugs": constraints.get("drugs", []),
            "responses": constraints.get("responses", []),
            "toxicities": constraints.get("toxicities", []),
            "metastatic_sites": constraints.get("metastatic_sites", []),
            "ddi_terms": constraints.get("ddi_terms", []),
        }
        requested_entities = [
            (entity_type, str(value).strip())
            for entity_type, values in entity_terms.items()
            for value in values or []
            if str(value).strip()
        ]
        hard_count = len(clauses) + int(bool(cancer_type))
        if not requested_entities:
            # Zero is intentional: hybrid min-max normalization must not turn
            # an unconstrained browse into a full structured-channel vote.
            score = 0.0 if hard_count == 0 else 1.0
            for item in results:
                item.update({
                    "structured_score": score,
                    "structured_match_mode": "unconstrained" if hard_count == 0 else "and",
                    "matched_constraint_count": hard_count,
                    "requested_constraint_count": hard_count,
                    "constraint_coverage": 0.0 if hard_count == 0 else 1.0,
                })
            results.sort(key=lambda item: item["case_id"], reverse=True)
            return results[:limit]

        candidate_ids = {str(item["case_id"]) for item in results}
        matched: Dict[str, set[tuple[str, str]]] = {case_id: set() for case_id in candidate_ids}
        for entity_type, value in requested_entities:
            rows = self.conn.execute(
                """
                SELECT DISTINCT CASE
                    WHEN e.ref_type = 'event' THEN te.case_id
                    WHEN e.ref_type = 'chunk' THEN ec.case_id
                END AS case_id
                FROM entity_index e
                LEFT JOIN timeline_events te ON e.ref_type = 'event' AND te.event_id = e.ref_id
                LEFT JOIN evidence_chunks ec ON e.ref_type = 'chunk' AND ec.chunk_id = e.ref_id
                WHERE e.entity_type = ? AND e.entity_value LIKE ?
                """,
                (entity_type, f"%{value.lower()}%"),
            ).fetchall()
            criterion = (entity_type, value.lower())
            for row in rows:
                case_id = str(row["case_id"] or "")
                if case_id in matched:
                    matched[case_id].add(criterion)

        requested_total = hard_count + len(requested_entities)
        scored: List[Dict[str, Any]] = []
        for item in results:
            entity_matches = matched[str(item["case_id"])]
            if not entity_matches:
                continue
            matched_total = hard_count + len(entity_matches)
            coverage = matched_total / requested_total
            all_matched = len(entity_matches) == len(requested_entities)
            item.update({
                "structured_score": round(1.0 if all_matched else 0.25 + 0.5 * coverage, 6),
                "structured_match_mode": "and" if all_matched else "or",
                "matched_constraint_count": matched_total,
                "requested_constraint_count": requested_total,
                "constraint_coverage": round(coverage, 6),
                "matched_entity_constraints": [
                    {"entity_type": key, "value": value}
                    for key, value in sorted(entity_matches)
                ],
            })
            scored.append(item)
        scored.sort(key=lambda item: (item["structured_score"], item["matched_constraint_count"], item["case_id"]), reverse=True)
        return scored[:limit]

    def fetch_evidence(
        self,
        doc_id: Optional[str] = None,
        chunk_ids: Optional[List[str]] = None,
        event_ids: Optional[List[str]] = None,
        limit: int = 20,
        cutoff: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # Preserve the caller's ranked order. SQL IN has set semantics and does
        # not guarantee that rows follow the order of its parameters.
        requested_chunk_ids = list(dict.fromkeys(chunk_ids or []))
        requested_event_ids = list(dict.fromkeys(event_ids or []))
        clauses = []
        params: List[Any] = []
        if doc_id:
            clauses.append("ec.doc_id = ?")
            params.append(doc_id)
        if requested_chunk_ids:
            placeholders = ",".join(["?"] * len(requested_chunk_ids))
            clauses.append(f"ec.chunk_id IN ({placeholders})")
            params.extend(requested_chunk_ids)
        if requested_event_ids:
            placeholders = ",".join(["?"] * len(requested_event_ids))
            clauses.append(f"ec.event_id IN ({placeholders})")
            params.extend(requested_event_ids)
        if cutoff:
            clauses.append("d.date IS NOT NULL AND d.date != '' AND d.date <= ?")
            clauses.append("json_extract(d.metadata_json, '$.publication_date_source') = 'citation_publication_date'")
            params.append(cutoff)
        sql = """
        SELECT ec.*, d.pmid, d.title, d.date AS pub_date,
               d.document_type,
               d.source_layer AS document_source_layer,
               d.source_file AS document_source_file
        FROM evidence_chunks ec
        LEFT JOIN documents d ON d.doc_id = ec.doc_id
        """
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        rows = self.conn.execute(sql, params).fetchall()
        results = [dict(row) for row in rows]
        if requested_chunk_ids:
            chunk_rank = {value: rank for rank, value in enumerate(requested_chunk_ids)}
            results.sort(key=lambda row: chunk_rank.get(str(row.get("chunk_id")), len(chunk_rank)))
        elif requested_event_ids:
            event_rank = {value: rank for rank, value in enumerate(requested_event_ids)}
            results.sort(key=lambda row: event_rank.get(str(row.get("event_id")), len(event_rank)))
        else:
            results.sort(key=lambda row: str(row.get("chunk_id") or ""))
        return results[:limit]

    def fetch_structured_events(
        self,
        event_ids: List[str],
        limit: int = 20,
        cutoff: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch original Step3 event JSON, not merely a case-level match."""
        if not event_ids:
            return []
        cutoff_clause = ""
        requested_event_ids = list(dict.fromkeys(event_ids))
        params: List[Any] = [*requested_event_ids]
        if cutoff:
            cutoff_clause = (
                " AND d.date IS NOT NULL AND d.date != '' AND d.date <= ?"
                " AND json_extract(d.metadata_json, '$.publication_date_source') = 'citation_publication_date'"
            )
            params.append(cutoff)
        placeholders = ",".join(["?"] * len(requested_event_ids))
        rows = self.conn.execute(
            f"""
            SELECT te.*, d.pmid, d.title, d.date AS pub_date, d.source_file
            FROM timeline_events te
            LEFT JOIN documents d ON d.doc_id = te.doc_id
            WHERE te.event_id IN ({placeholders})
            {cutoff_clause}
            """,
            params,
        ).fetchall()
        event_rank = {value: rank for rank, value in enumerate(requested_event_ids)}
        results = [dict(row) for row in rows]
        results.sort(key=lambda row: event_rank.get(str(row.get("event_id")), len(event_rank)))
        return results[:limit]

    def get_document_metadata(self, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Return publication metadata in one query for retrieval-time filtering."""
        unique_ids = list(dict.fromkeys(str(value) for value in doc_ids if str(value)))
        if not unique_ids:
            return {}
        placeholders = ",".join(["?"] * len(unique_ids))
        rows = self.conn.execute(
            f"""SELECT doc_id, pmid, title, date AS pub_date,
                       json_extract(metadata_json, '$.publication_date_source') AS publication_date_source
                FROM documents WHERE doc_id IN ({placeholders})""",
            unique_ids,
        ).fetchall()
        return {str(row["doc_id"]): dict(row) for row in rows}
