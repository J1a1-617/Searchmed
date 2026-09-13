from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bm25_index import BM25Index
from .models import CitationAnchor, EntitySet, EvidenceChunk
from .schema_parser import load_records_from_data_root
from .structured_index import StructuredIndex
from .vector_index import DEFAULT_EMBED_MODEL_PATH, VectorEncoder, VectorIndex


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _refresh_vector_item_metadata(vector_dir: Path, metadata_by_chunk_id: Dict[str, Dict]) -> int:
    """Refresh non-vector metadata without recomputing unchanged embeddings."""
    metadata_path = vector_dir / "metadata.json"
    if not metadata_path.exists():
        return 0
    payload = _load_json(metadata_path)
    refreshed = 0
    for items in (payload.get("space_items") or {}).values():
        for item in items:
            current = item.get("metadata") or {}
            chunk_id = str(current.get("chunk_id") or item.get("doc_id") or "")
            replacement = metadata_by_chunk_id.get(chunk_id)
            if replacement is None:
                continue
            item["metadata"] = {**current, **replacement}
            refreshed += 1
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return refreshed


def _build_ddi_rule_chunks(data_root: Path) -> List[EvidenceChunk]:
    chunks: List[EvidenceChunk] = []
    chunk_seq = 0
    drug_name_map = _load_json(data_root / "all_drug_name_map.json")
    ddi_rules = _load_json(data_root / "all_ddi_rule.json")
    pk_rules = _load_json(data_root / "all_drug_pk_relation.json")
    mutation_drug = _load_json(data_root / "mutation_drug_map_min.json")

    def next_chunk_id(prefix: str) -> str:
        nonlocal chunk_seq
        chunk_seq += 1
        return f"{prefix}#chunk-{chunk_seq:06d}"

    for item in drug_name_map:
        aliases = item.get("alias", [])
        text = (
            f"drug_name={item.get('drug_name')}; generic_name={item.get('generic_name')}; "
            f"brand_name={item.get('brand_name')}; aliases={', '.join(aliases)}"
        )
        chunks.append(
            EvidenceChunk(
                chunk_id=next_chunk_id("drug_name_map"),
                doc_id="drug_name_map",
                case_id="drug_name_map#case-1",
                chunk_type="ddi_rule_chunk",
                text=text,
                text_for_embedding=text,
                evidence_level="drug_label_or_ddi_rule_evidence",
                entities=EntitySet(
                    drugs=[item.get("drug_name", ""), item.get("generic_name", "")], ddi_terms=aliases
                ).normalize(),
                citation=CitationAnchor(
                    source_layer="drug_label_or_ddi_rule",
                    source_file=str(data_root / "all_drug_name_map.json"),
                    field_path="[]",
                ),
                metadata={"source": "all_drug_name_map"},
            )
        )

    for item in ddi_rules:
        text = (
            f"enzyme_or_transporter={item.get('enzyme_transporter')}; "
            f"role1={item.get('role1')}; role2={item.get('role2')}; "
            f"ddi_likelihood={item.get('ddi_likelihood')}; notes={item.get('notes')}"
        )
        chunks.append(
            EvidenceChunk(
                chunk_id=next_chunk_id("ddi_rule"),
                doc_id="ddi_rule",
                case_id="ddi_rule#case-1",
                chunk_type="ddi_rule_chunk",
                text=text,
                text_for_embedding=text,
                evidence_level="drug_label_or_ddi_rule_evidence",
                entities=EntitySet(
                    ddi_terms=[item.get("enzyme_transporter", ""), item.get("ddi_likelihood", "")]
                ).normalize(),
                citation=CitationAnchor(
                    source_layer="drug_label_or_ddi_rule",
                    source_file=str(data_root / "all_ddi_rule.json"),
                    field_path="[]",
                ),
                metadata={"source": "all_ddi_rule"},
            )
        )

    for item in pk_rules:
        text = (
            f"drug={item.get('drug_name')}; role={item.get('role')}; "
            f"enzyme_or_transporter={item.get('enzyme_transporter')}; "
            f"interaction_type={item.get('pk_interaction_type')}; "
            f"related_pk_drug={item.get('related_pk_drug')}; adme={item.get('adme_desc')}"
        )
        entities = EntitySet(
            drugs=[item.get("drug_name", ""), item.get("related_pk_drug", "")],
            ddi_terms=[item.get("enzyme_transporter", ""), item.get("pk_interaction_type", ""), item.get("role", "")],
        ).normalize()
        chunks.append(
            EvidenceChunk(
                chunk_id=next_chunk_id("drug_pk"),
                doc_id="drug_pk_relation",
                case_id="drug_pk_relation#case-1",
                chunk_type="ddi_rule_chunk",
                text=text,
                text_for_embedding=text,
                evidence_level="drug_label_or_ddi_rule_evidence",
                entities=entities,
                citation=CitationAnchor(
                    source_layer="drug_label_or_ddi_rule",
                    source_file=str(data_root / "all_drug_pk_relation.json"),
                    field_path="[]",
                ),
                metadata={"source": "all_drug_pk_relation"},
            )
        )

    for item in mutation_drug:
        mutation = item.get("mutation", "")
        drugs = item.get("drugs", [])
        text = f"mutation={mutation}; recommended_or_reported_drugs={', '.join(drugs)}"
        chunks.append(
            EvidenceChunk(
                chunk_id=next_chunk_id("mutation_drug"),
                doc_id="mutation_drug_map",
                case_id="mutation_drug_map#case-1",
                chunk_type="ddi_rule_chunk",
                text=text,
                text_for_embedding=text,
                evidence_level="drug_label_or_ddi_rule_evidence",
                entities=EntitySet(gene_alterations=[mutation], drugs=drugs).normalize(),
                citation=CitationAnchor(
                    source_layer="drug_label_or_ddi_rule",
                    source_file=str(data_root / "mutation_drug_map_min.json"),
                    field_path="[]",
                ),
                metadata={"source": "mutation_drug_map_min"},
            )
        )

    return chunks


def _embed_space_streaming(
    vector_index: VectorIndex,
    encoder: VectorEncoder,
    entries: List[Dict],
    space: str,
    batch_size: int = 128,
) -> Tuple[int, float]:
    """Encode a list of entries into a single space with progress output.

    Returns (num_embedded, elapsed_seconds).
    """
    start = time.time()
    total = len(entries)
    if total == 0:
        return 0, 0.0

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = entries[batch_start:batch_end]
        texts = [entry["text"] for entry in batch]
        vectors = encoder.encode_many(texts=texts, batch_size=batch_size)
        vector_index._append(space=space, vectors=vectors, items=batch)
        print(f"  [{space}] embedded {batch_end}/{total} chunks", flush=True)

    return total, time.time() - start


def build_indexes(
    case_data_root: Path,
    external_data_root: Path,
    output_root: Path,
    skip_structured: bool = False,
    skip_embedding: bool = False,
    skip_bm25: bool = False,
    limit_records: Optional[int] = None,
    record_offset: int = 0,
    append_embedding: bool = False,
    refresh_embedding_spaces: Optional[Sequence[str]] = None,
    embed_model_path: Optional[Path] = None,
    require_sentence_transformers: bool = False,
) -> Dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    records = load_records_from_data_root(case_data_root)
    ddi_rule_chunks = _build_ddi_rule_chunks(external_data_root)

    structured_db_path = output_root / "structured.db"
    vector_dir = output_root / "vector"
    bm25_path = output_root / "bm25.json"

    structured: Optional[StructuredIndex] = None
    if skip_structured and structured_db_path.exists():
        print(f"Skipping structured rebuild; reusing {structured_db_path}", flush=True)
    else:
        structured = StructuredIndex(structured_db_path)
        structured.reset()

    bm25_index: Optional[BM25Index] = None
    if skip_bm25 and bm25_path.exists():
        print(f"Skipping BM25 rebuild; reusing {bm25_path}", flush=True)
        bm25_index = BM25Index.load(bm25_path)
    else:
        bm25_index = BM25Index()

    vector_index: Optional[VectorIndex] = None
    dense_entries_by_space: Dict[str, List[Dict]] = {
        "case_semantic": [],
        "event_semantic": [],
        "structured_semantic": [],
    }
    metadata_by_chunk_id: Dict[str, Dict] = {}

    embed_records = records
    if limit_records is not None:
        embed_records = records[record_offset : record_offset + limit_records]
        print(
            f"Embedding subset: records[{record_offset}:{record_offset + limit_records}] "
            f"({len(embed_records)} of {len(records)} records)",
            flush=True,
        )

    if not skip_embedding:
        if append_embedding and (vector_dir / "metadata.json").exists():
            vector_index = VectorIndex.load(
                vector_dir,
                model_path=embed_model_path or DEFAULT_EMBED_MODEL_PATH,
            )
            if require_sentence_transformers and vector_index.encoder.backend != "sentence_transformers":
                raise RuntimeError(
                    f"SentenceTransformer model is required but could not be loaded from "
                    f"{embed_model_path or DEFAULT_EMBED_MODEL_PATH}"
                )
            refresh_spaces = set(refresh_embedding_spaces or [])
            invalid_spaces = refresh_spaces.difference(dense_entries_by_space)
            if invalid_spaces:
                raise ValueError(f"Unknown embedding spaces: {sorted(invalid_spaces)}")
            for space in sorted(refresh_spaces):
                old_count = len(vector_index.space_items.get(space, []))
                vector_index.space_items.pop(space, None)
                vector_index.space_vectors.pop(space, None)
                print(f"Refreshing embedding space {space} ({old_count} old chunks discarded)", flush=True)
            existing = vector_index.embedded_doc_ids()
            print(f"Appending to existing vector index ({len(existing)} embedded chunks)", flush=True)
        else:
            encoder = VectorEncoder(
                model_path=embed_model_path or DEFAULT_EMBED_MODEL_PATH,
                require_sentence_transformers=require_sentence_transformers,
            )
            vector_index = VectorIndex(encoder=encoder)

    phase = "BM25" if skip_structured else "structured and BM25"
    if not skip_bm25:
        print(f"Building {phase} indexes...", flush=True)
    for record in records:
        if structured is not None:
            structured.add_record(record)
        for chunk in record.evidence_chunks:
            meta = {
                "source": "case",
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "case_id": chunk.case_id,
                "event_id": chunk.event_id,
                "chunk_type": chunk.chunk_type,
                "evidence_level": chunk.evidence_level,
                "document_type": record.document.document_type,
                "source_layer": chunk.citation.source_layer,
                "source_file": chunk.citation.source_file,
                "field_path": chunk.citation.field_path,
                "start_char": chunk.citation.start_char,
                "end_char": chunk.citation.end_char,
                "chunk_order": chunk.citation.chunk_order,
                "pmid": record.document.pmid,
                "title": record.document.title,
            }
            metadata_by_chunk_id[chunk.chunk_id] = meta
            if not skip_bm25 and bm25_index is not None:
                bm25_index.add(doc_id=chunk.chunk_id, text=chunk.text, metadata=meta)

    if not skip_bm25 and bm25_index is not None:
        for chunk in ddi_rule_chunks:
            meta = {
                "source": "rule",
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "case_id": chunk.case_id,
                "event_id": chunk.event_id,
                "chunk_type": chunk.chunk_type,
                "evidence_level": chunk.evidence_level,
                "rule_source": chunk.metadata.get("source"),
            }
            bm25_index.add(doc_id=chunk.chunk_id, text=chunk.text, metadata=meta)

    if not skip_embedding and vector_index is not None:
        embedded_ids = vector_index.embedded_doc_ids() if append_embedding else set()
        for record in embed_records:
            for chunk in record.evidence_chunks:
                if chunk.chunk_id in embedded_ids:
                    continue
                meta = {
                    "source": "case",
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "case_id": chunk.case_id,
                    "event_id": chunk.event_id,
                    "chunk_type": chunk.chunk_type,
                    "evidence_level": chunk.evidence_level,
                    "document_type": record.document.document_type,
                    "source_layer": chunk.citation.source_layer,
                    "source_file": chunk.citation.source_file,
                    "field_path": chunk.citation.field_path,
                    "start_char": chunk.citation.start_char,
                    "end_char": chunk.citation.end_char,
                    "chunk_order": chunk.citation.chunk_order,
                    "pmid": record.document.pmid,
                    "title": record.document.title,
                }
                if chunk.text_for_embedding and chunk.chunk_type in {"case_text_chunk", "event_chunk", "structured_field_chunk"}:
                    dense_space = {
                        "case_text_chunk": "case_semantic",
                        "event_chunk": "event_semantic",
                        "structured_field_chunk": "structured_semantic",
                    }[chunk.chunk_type]
                    dense_entries_by_space[dense_space].append(
                        {
                            "doc_id": chunk.chunk_id,
                            "text": chunk.text_for_embedding,
                            "metadata": {**meta, "embedding_space": dense_space},
                        }
                    )

    vector_status = "pending"
    if vector_dir.joinpath("metadata.json").exists():
        vector_status = "partial" if limit_records is not None else "ready"

    if skip_embedding:
        print("Skipping embedding; reusing existing vector index if present.", flush=True)
        if vector_dir.joinpath("metadata.json").exists():
            refreshed = _refresh_vector_item_metadata(vector_dir, metadata_by_chunk_id)
            print(f"Refreshed metadata for {refreshed} existing vector chunks.", flush=True)
            vector_status = "partial" if limit_records is not None else "ready"
    elif vector_index is not None:
        print(
            f"Embedding {len(dense_entries_by_space['case_semantic'])} case text chunks and "
            f"{len(dense_entries_by_space['event_semantic'])} event chunks and "
            f"{len(dense_entries_by_space['structured_semantic'])} structured field chunks with medical model...",
            flush=True,
        )
        for space, entries in dense_entries_by_space.items():
            if not entries:
                print(f"  [{space}] no new chunks", flush=True)
                continue
            count, elapsed = _embed_space_streaming(
                vector_index=vector_index,
                encoder=vector_index.encoder,
                entries=entries,
                space=space,
                batch_size=128,
            )
            print(f"  [{space}] done: {count} chunks in {elapsed:.1f}s", flush=True)
        vector_index.save(vector_dir)
        total_embedded = len(vector_index.embedded_doc_ids())
        vector_status = "ready" if limit_records is None and record_offset == 0 else "partial"
        if limit_records is None and total_embedded > 0:
            vector_status = "ready"

    if not skip_bm25 and bm25_index is not None:
        print(f"Saving BM25 index ({len(bm25_index.docs)} docs)...", flush=True)
        bm25_index.save(bm25_path)
    if structured is not None:
        structured.close()

    bm25_doc_count = len(bm25_index.docs) if bm25_index is not None else 0
    if skip_bm25 and bm25_path.exists():
        bm25_doc_count = len(BM25Index.load(bm25_path).docs)

    embedded_counts = {}
    if vector_dir.joinpath("metadata.json").exists():
        loaded = VectorIndex.load(
            vector_dir,
            model_path=embed_model_path or DEFAULT_EMBED_MODEL_PATH,
        )
        embedded_counts = {space: len(items) for space, items in loaded.space_items.items()}

    manifest = {
        "record_count": len(records),
        "rule_chunk_count": len(ddi_rule_chunks),
        "bm25_doc_count": bm25_doc_count,
        "structured_db": str(structured_db_path),
        "vector_dir": str(vector_dir),
        "bm25_path": str(bm25_path),
        "dense_spaces": ["case_semantic", "event_semantic", "structured_semantic"],
        "vector_status": vector_status,
        "structured_status": "ready" if structured_db_path.exists() else "missing",
        "bm25_status": "ready" if bm25_path.exists() else "missing",
        "embedded_record_offset": record_offset,
        "embedded_record_limit": limit_records,
        "embedded_chunk_counts": embedded_counts,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "manifest": str(manifest_path),
        "structured_db": str(structured_db_path),
        "vector_dir": str(vector_dir),
        "bm25_path": str(bm25_path),
    }
