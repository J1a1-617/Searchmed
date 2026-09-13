from __future__ import annotations

import json
import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .bm25_index import BM25Index
from .query_adapter import adapt_bm25_query, adapt_dense_query
from .structured_index import StructuredIndex
from .text_utils import min_max_normalize, retrieval_terms, weighted_sum
from .vector_index import DEFAULT_EMBED_MODEL_PATH, VectorIndex

if TYPE_CHECKING:
    from .llm_client import LLMClient


@dataclass
class SearchHit:
    id: str
    score: float
    source: str
    text: str
    metadata: Dict[str, Any]


class RetrievalTools:
    STEP3_LOCATOR_POOL = 180
    STEP3_DOCUMENT_POOL = 30
    STEP2_DETAILS_PER_DOCUMENT = 4

    @staticmethod
    def _target_event_types(text: str) -> List[str]:
        lowered = str(text or "").lower()
        patterns = {
            "early_response": r"早期|early|initial|首次|first assessment|症状改善|response|缓解|\b(?:cr|pr|orr|dcr)\b",
            "progression": r"耐药|进展|复发|resistan|progress|relapse|\bpd\b",
            "toxicity": r"毒性|不良反应|安全性|toxicit|adverse|\bae\b",
            "ddi": r"相互作用|合并用药|interaction|\bddi\b|cyp3a|抑制剂|诱导剂",
            "outcome": r"结局|生存|随访|outcome|survival|\bpfs\b|\bos\b|死亡|death",
        }
        return [name for name, pattern in patterns.items() if re.search(pattern, lowered)] or ["target_outcome"]

    @classmethod
    def _select_event_aligned_details(
        cls,
        rows: List[Dict[str, Any]],
        *,
        query: str,
        locator_texts: Optional[List[str]] = None,
        locator_event_types: Optional[List[str]] = None,
        per_document: int = 3,
    ) -> List[tuple[float, Dict[str, Any], str]]:
        """Choose Step2 spans aligned with the located Step3 event, not merely its PMID."""
        anchor_text = " ".join([query, *(locator_texts or [])])
        anchor_terms = set(retrieval_terms(adapt_bm25_query(anchor_text), max_terms=80))
        target_types = list(dict.fromkeys([*(locator_event_types or []), *cls._target_event_types(query)]))
        wants_early = "early_response" in target_types
        wants_progression = "progression" in target_types
        scored: List[tuple[float, Dict[str, Any], str]] = []
        for row in rows:
            if row.get("chunk_type") != "case_text_chunk":
                continue
            text = " ".join(str(row.get("text") or "").split())
            lowered = text.lower()
            lexical = sum(term in lowered for term in anchor_terms) / max(1, len(anchor_terms))
            is_early = bool(re.search(r"早期|初始|首次|early|initial|first|\b(?:cr|pr|orr|dcr)\b|缓解|改善", lowered))
            is_progression = bool(re.search(r"耐药|进展|复发|resistan|progress|relapse|\bpd\b", lowered))
            is_toxicity = bool(re.search(r"毒性|不良反应|toxicit|adverse|\bae\b", lowered))
            is_outcome = bool(re.search(r"疗效|结局|随访|response|outcome|survival|\bpfs\b|\bos\b|症状", lowered))
            if is_early:
                bucket = "early_response"
            elif is_toxicity or (is_progression and not wants_progression):
                bucket = "toxicity_or_counter"
            else:
                bucket = "target_outcome"
            alignment = 0.0
            if wants_early and is_early:
                alignment += 0.35
            if wants_progression and is_progression:
                alignment += 0.30
            if "toxicity" in target_types and is_toxicity:
                alignment += 0.30
            if "outcome" in target_types and is_outcome:
                alignment += 0.20
            # Long-term resistance is a useful counter at most once, but must
            # not outrank an early-response span for an early-response target.
            mismatch_penalty = 0.35 if wants_early and is_progression and not is_early else 0.0
            score = lexical + alignment - mismatch_penalty
            scored.append((score, row, bucket))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected: List[tuple[float, Dict[str, Any], str]] = []
        used_buckets = set()
        for item in scored:
            if item[2] in used_buckets:
                continue
            selected.append(item)
            used_buckets.add(item[2])
            if len(selected) >= per_document:
                break
        return selected

    def __init__(
        self,
        index_root: Path,
        embed_model_path: Optional[Path] = None,
        llm_client: Optional["LLMClient"] = None,
        reranker_backend: str = "llm",
        reranker_model: Optional[str] = None,
        reranker_device: Optional[str] = None,
        external_knowledge_enabled: bool = True,
    ) -> None:
        self.index_root = index_root
        self.external_knowledge_enabled = bool(external_knowledge_enabled)
        self.structured = StructuredIndex(index_root / "structured.db")
        self.vector = VectorIndex.load_optional(
            index_root / "vector",
            model_path=embed_model_path or DEFAULT_EMBED_MODEL_PATH,
        )
        self.bm25 = BM25Index.load(index_root / "bm25.json")
        if not self.external_knowledge_enabled:
            self.bm25 = self.bm25.filtered(
                lambda doc: not self._is_external_knowledge_metadata(doc.get("metadata") or {})
            )
        self.temporal_filter_mode = "off"
        self.temporal_cutoff: Optional[str] = None
        self.reranker = None
        self.last_qwen_overflow: List[SearchHit] = []
        self.last_knowledge_hits: List[SearchHit] = []
        self.last_rerank_backend: str = "none"
        backend = str(reranker_backend or "none").lower()
        # ``local``/``cross_encoder`` use a sequence-classification model.
        # Qwen3-Reranker is a causal-LM yes/no scorer and must never be loaded
        # through CrossEncoder (which would create an untrained score head).
        if backend in {"local", "cross_encoder"}:
            from .local_rerank import DEFAULT_RERANK_MODEL, LocalCrossEncoderReranker

            self.reranker = LocalCrossEncoderReranker(
                model_name_or_path=reranker_model or DEFAULT_RERANK_MODEL,
                device=reranker_device,
            )
        elif backend == "qwen":
            from .local_rerank import DEFAULT_QWEN_RERANK_MODEL, Qwen3Reranker

            self.reranker = Qwen3Reranker(
                reranker_model or DEFAULT_QWEN_RERANK_MODEL,
                device=reranker_device,
            )
        elif backend == "qwen_then_llm":
            if llm_client is None:
                raise ValueError("qwen_then_llm backend requires llm_client")
            from .local_rerank import DEFAULT_QWEN_RERANK_MODEL, QwenThenLLMReranker

            self.reranker = QwenThenLLMReranker(
                llm_client,
                model_name_or_path=reranker_model or DEFAULT_QWEN_RERANK_MODEL,
                device=reranker_device,
            )
        elif backend == "qwen_with_llm_fallback":
            if llm_client is None:
                raise ValueError("qwen_with_llm_fallback backend requires llm_client")
            from .local_rerank import DEFAULT_QWEN_RERANK_MODEL, QwenWithLLMFallbackReranker

            self.reranker = QwenWithLLMFallbackReranker(
                llm_client,
                model_name_or_path=reranker_model or DEFAULT_QWEN_RERANK_MODEL,
                device=reranker_device,
            )
        elif backend == "llm" and llm_client is not None:
            from .llm_rerank import LLMReranker

            self.reranker = LLMReranker(llm_client)
        elif backend not in {"llm", "none"}:
            raise ValueError(f"Unsupported reranker backend: {reranker_backend!r}")

    @staticmethod
    def _is_external_knowledge_metadata(metadata: Dict[str, Any]) -> bool:
        source = str(metadata.get("source") or "").strip().lower()
        chunk_type = str(metadata.get("chunk_type") or "").strip().lower()
        evidence_level = str(metadata.get("evidence_level") or "").strip().lower()
        rule_source = str(metadata.get("rule_source") or "").strip().lower()
        return bool(
            source == "rule"
            or chunk_type == "ddi_rule_chunk"
            or evidence_level == "drug_label_or_ddi_rule_evidence"
            or rule_source in {
                "all_drug_name_map",
                "all_drug_pk_relation",
                "all_ddi_rule",
                "mutation_drug_map_min",
                "case_ddi",
            }
        )

    def close(self) -> None:
        self.structured.close()

    @staticmethod
    def _normalize_cutoff(value: Optional[str]) -> Optional[str]:
        raw = str(value or "").strip()
        if not raw:
            return None
        if re.fullmatch(r"\d{4}", raw):
            return f"{raw}-12-31"
        match = re.fullmatch(r"(\d{4})-(\d{1,2})", raw)
        if match:
            year, month = int(match.group(1)), int(match.group(2))
            return f"{year:04d}-{month:02d}-{monthrange(year, month)[1]:02d}"
        match = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", raw)
        if match:
            return date(*(int(value) for value in match.groups())).isoformat()
        raise ValueError(f"Unsupported temporal cutoff: {value!r}")

    def configure_temporal_filter(self, mode: str = "off", cutoff: Optional[str] = None) -> None:
        """Configure retrieval-time publication filtering for the current case."""
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"off", "cutoff"}:
            raise ValueError("temporal filter mode must be 'off' or 'cutoff'")
        normalized_cutoff = self._normalize_cutoff(cutoff)
        if normalized_mode == "cutoff" and not normalized_cutoff:
            raise ValueError("cutoff mode requires a temporal cutoff")
        self.temporal_filter_mode = normalized_mode
        self.temporal_cutoff = normalized_cutoff if normalized_mode == "cutoff" else None

    def _candidate_pool_size(self, top_k: int) -> int:
        if self.temporal_filter_mode != "cutoff":
            return top_k
        return min(500, max(top_k * 4, top_k + 40))

    def _filter_hits_by_time(self, hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        if self.temporal_filter_mode != "cutoff":
            return hits[:top_k]
        doc_ids = [
            str(hit.metadata.get("doc_id") or hit.id.split("#", 1)[0]).strip()
            for hit in hits
        ]
        metadata_by_doc = self.structured.get_document_metadata(doc_ids)
        filtered: List[SearchHit] = []
        for hit, doc_id in zip(hits, doc_ids):
            document = metadata_by_doc.get(doc_id)
            pub_date = str((document or {}).get("pub_date") or "").strip()
            date_source = str((document or {}).get("publication_date_source") or "").strip()
            if (
                date_source != "citation_publication_date"
                or not pub_date
                or pub_date[:10] > str(self.temporal_cutoff)
            ):
                continue
            hit.metadata = {**hit.metadata, **(document or {})}
            filtered.append(hit)
            if len(filtered) >= top_k:
                break
        return filtered

    def structured_search(self, constraints: Dict[str, Any], top_k: int = 20) -> List[SearchHit]:
        rows = self.structured.structured_search(
            constraints=constraints,
            limit=top_k,
            cutoff=self.temporal_cutoff,
        )
        hits: List[SearchHit] = []
        for row in rows:
            text = (
                f"doc={row.get('doc_id')} title={row.get('title')} "
                f"cancer={row.get('cancer_type')} histology={row.get('histology')} "
                f"stage={row.get('overall_stage')}"
            )
            hits.append(
                SearchHit(
                    id=row["case_id"],
                    score=float(row.get("structured_score", 1.0)),
                    source="structured",
                    text=text,
                    metadata=dict(row),
                )
            )
        return hits

    def dense_search(self, query: str, top_k: int = 20, spaces: Optional[List[str]] = None) -> List[SearchHit]:
        effective_query = adapt_dense_query(query)
        rows = self.vector.search(
            query=effective_query,
            top_k=self._candidate_pool_size(top_k),
            spaces=spaces or ["case_semantic", "event_semantic", "structured_semantic"],
        )
        hits = [
            SearchHit(
                id=row.doc_id,
                score=float(row.score),
                source="dense",
                text=row.text,
                metadata=row.metadata,
            )
            for row in rows
        ]
        return self._filter_hits_by_time(hits, top_k)

    def step3_then_step2_search(self, query: str, top_k: int = 12) -> List[SearchHit]:
        """Locate dense Step3 facts first, then rank detailed Step2 chunks only inside matched documents."""
        dense_query = adapt_dense_query(query)
        lexical_query = adapt_bm25_query(query)
        locator_hits = self.dense_search(
            query=dense_query,
            top_k=max(self.STEP3_LOCATOR_POOL, top_k * 4),
            spaces=["structured_semantic", "event_semantic"],
        )
        query_terms = set(retrieval_terms(lexical_query, max_terms=64))
        locators_by_doc: Dict[str, List[SearchHit]] = {}
        for hit in locator_hits:
            doc_id = str(hit.metadata.get("doc_id") or "").strip()
            if not doc_id:
                continue
            locators_by_doc.setdefault(doc_id, []).append(hit)

        def locator_document_score(locators: List[SearchHit]) -> float:
            dense_score = max(float(hit.score) for hit in locators)
            locator_text = " ".join(hit.text.lower() for hit in locators)
            lexical_score = sum(1 for term in query_terms if term in locator_text) / max(1, len(query_terms))
            return 0.70 * dense_score + 0.30 * lexical_score

        ranked_documents = sorted(
            locators_by_doc.items(),
            key=lambda item: locator_document_score(item[1]),
            reverse=True,
        )[: max(self.STEP3_DOCUMENT_POOL, top_k)]

        candidates: List[SearchHit] = []
        for doc_id, locators in ranked_documents:
            detail_rows = self.fetch_evidence(doc_id=doc_id, limit=300)
            locator_score = locator_document_score(locators)
            locator_ids = [hit.id for hit in locators[:3]]
            locator_event_types = list(dict.fromkeys(
                event_type
                for hit in locators
                for event_type in (hit.metadata.get("event_types") or self._target_event_types(hit.text))
            ))
            scored_rows = self._select_event_aligned_details(
                detail_rows, query=query, locator_texts=[hit.text for hit in locators[:3]],
                locator_event_types=locator_event_types,
                per_document=min(3, self.STEP2_DETAILS_PER_DOCUMENT),
            )
            for detail_score, row, evidence_slot in scored_rows:
                score = 0.55 * locator_score + 0.45 * detail_score
                candidates.append(SearchHit(
                    id=str(row["chunk_id"]),
                    score=float(score),
                    source="step3_then_step2",
                    text=str(row.get("text") or ""),
                    metadata={
                        **row,
                        "step3_locator_ids": locator_ids,
                        "step3_locator_score": locator_score,
                        "target_event_types": locator_event_types,
                        "event_aligned_slot": evidence_slot,
                        "retrieval_pipeline": "step3_locator_to_step2_detail",
                    },
                ))
        candidates.sort(key=lambda hit: hit.score, reverse=True)
        # Preserve document recall before spending remaining slots on sibling
        # chunks from the same report. Otherwise four strong chunks from a few
        # documents can recreate the document-level cutoff we just widened.
        first_by_doc: List[SearchHit] = []
        remaining: List[SearchHit] = []
        seen_docs = set()
        for hit in candidates:
            doc_id = str(hit.metadata.get("doc_id") or hit.id.split("#", 1)[0])
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                first_by_doc.append(hit)
            else:
                remaining.append(hit)
        return [*first_by_doc, *remaining][:top_k]

    def bm25_search(self, query: str, top_k: int = 20) -> List[SearchHit]:
        effective_query = adapt_bm25_query(query)
        rows = self.bm25.search(query=effective_query, top_k=self._candidate_pool_size(top_k))
        hits = [
            SearchHit(
                id=row.doc_id,
                score=float(row.score),
                source="bm25",
                text=row.text,
                metadata=row.metadata,
            )
            for row in rows
        ]
        return self._filter_hits_by_time(hits, top_k)

    def hybrid_search(
        self,
        query: str,
        constraints: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[SearchHit]:
        constraints = constraints or {}
        weights = weights or {"structured": 0.35, "dense": 0.40, "bm25": 0.25}
        structured_hits = self.structured_search(constraints=constraints, top_k=top_k)
        dense_hits = self.dense_search(query=query, top_k=top_k * 2, spaces=["case_semantic", "event_semantic", "structured_semantic"])
        bm25_hits = self.bm25_search(query=query, top_k=top_k * 2)

        merged: Dict[str, Dict[str, Any]] = {}
        for hit in structured_hits + dense_hits + bm25_hits:
            merged.setdefault(
                hit.id,
                {
                    "id": hit.id,
                    "text": hit.text,
                    "metadata": hit.metadata,
                    "scores": {},
                },
            )
            merged[hit.id]["scores"][hit.source] = max(hit.score, merged[hit.id]["scores"].get(hit.source, 0.0))
            if len(hit.text) > len(merged[hit.id]["text"]):
                merged[hit.id]["text"] = hit.text
            merged[hit.id]["metadata"].update(hit.metadata)

        scores_by_source: Dict[str, Dict[str, float]] = {}
        for item_id, item in merged.items():
            for source, score in item["scores"].items():
                scores_by_source.setdefault(source, {})[item_id] = score
        normalized_by_source = {
            source: min_max_normalize(source_scores)
            for source, source_scores in scores_by_source.items()
        }

        results: List[SearchHit] = []
        for item_id, item in merged.items():
            normalized_scores = {
                source: normalized_by_source[source][item_id]
                for source in item["scores"]
            }
            score = weighted_sum(normalized_scores, weights)
            results.append(
                SearchHit(
                    id=item["id"],
                    score=score,
                    source="hybrid",
                    text=item["text"],
                    metadata={
                        **item["metadata"],
                        "component_scores": item["scores"],
                        "normalized_component_scores": normalized_scores,
                    },
                )
            )
        results.sort(key=lambda hit: hit.score, reverse=True)
        return results[:top_k]

    def rerank(self, query: str, hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        if not hits:
            self.last_knowledge_hits = []
            self.last_qwen_overflow = []
            self.last_rerank_backend = "none"
            return []
        knowledge_prefixes = ("mutation_drug#", "drug_alias#", "drug_pk_relation#", "ddi_rule#", "case_ddi#")
        knowledge_hits = [
            hit for hit in hits
            if str(hit.id).lower().startswith(knowledge_prefixes)
            or str(hit.metadata.get("chunk_type") or "") == "ddi_rule_chunk"
        ]
        evidence_hits = [hit for hit in hits if hit not in knowledge_hits]
        self.last_knowledge_hits = knowledge_hits
        if not evidence_hits:
            self.last_qwen_overflow = []
            self.last_rerank_backend = "none"
            return []
        if self.reranker is None:
            self.last_qwen_overflow = []
            self.last_rerank_backend = "none"
            return evidence_hits[:top_k]
        self.last_rerank_backend = type(self.reranker).__name__
        ranked = self.reranker.rerank(query=query, hits=evidence_hits, top_k=top_k)
        qwen = getattr(self.reranker, "qwen", None)
        pool = getattr(qwen, "last_ranked_pool", None)
        self.last_qwen_overflow = list(pool[len(ranked):]) if isinstance(pool, list) else []
        return ranked

    def fetch_evidence(
        self,
        pmid: Optional[str] = None,
        doc_id: Optional[str] = None,
        chunk_ids: Optional[List[str]] = None,
        event_ids: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        if pmid and not doc_id:
            row = self.structured.conn.execute("SELECT doc_id FROM documents WHERE pmid = ?", (pmid,)).fetchone()
            if row:
                doc_id = row["doc_id"]
        rows = self.structured.fetch_evidence(
            doc_id=doc_id,
            chunk_ids=chunk_ids,
            event_ids=event_ids,
            limit=limit,
            cutoff=self.temporal_cutoff,
        )
        parsed = []
        for row in rows:
            row = dict(row)
            for field in ("entities_json", "citation_json", "metadata_json"):
                if isinstance(row.get(field), str):
                    row[field] = json.loads(row[field])
            parsed.append(row)
        return parsed

    def fetch_structured_events(self, event_ids: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        rows = self.structured.fetch_structured_events(
            event_ids=event_ids,
            limit=limit,
            cutoff=self.temporal_cutoff,
        )
        parsed: List[Dict[str, Any]] = []
        for row in rows:
            raw = json.loads(row.get("raw_json") or "{}")
            entities = json.loads(row.get("entities_json") or "{}")
            parsed.append({
                "chunk_id": row["event_id"],
                "event_id": row["event_id"],
                "doc_id": row["doc_id"],
                "case_id": row["case_id"],
                "chunk_type": "event_chunk",
                "evidence_level": "structured_extraction_evidence",
                "text": row.get("summary_text") or "",
                "raw_event": raw,
                "entities_json": entities,
                "citation_json": {
                    "source_layer": "step3_structured",
                    "source_file": row.get("source_file"),
                    "field_path": "timeline_events",
                    "pmid": row.get("pmid"),
                    "title": row.get("title"),
                    "date": row.get("pub_date"),
                },
                "metadata_json": {
                    "event_time": row.get("event_time"),
                    "event_types": json.loads(row.get("event_types_json") or "[]"),
                },
            })
        return parsed
