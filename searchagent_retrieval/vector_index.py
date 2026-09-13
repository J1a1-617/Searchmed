from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .text_utils import tokenize

DEFAULT_EMBED_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "bge-large-zh-v1.5"
)
BGE_LARGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


@dataclass
class VectorHit:
    doc_id: str
    score: float
    text: str
    metadata: Dict


class VectorEncoder:
    def __init__(
        self,
        model_path: Path = DEFAULT_EMBED_MODEL_PATH,
        fallback_dim: int = 512,
        require_sentence_transformers: bool = False,
    ) -> None:
        self.model_path = model_path
        self.fallback_dim = fallback_dim
        self.require_sentence_transformers = require_sentence_transformers
        self._backend = "hashing"
        self._model = None
        self._init_model()

    @property
    def backend(self) -> str:
        return self._backend

    def _init_model(self) -> None:
        if not self.model_path.exists():
            if self.require_sentence_transformers:
                raise RuntimeError(f"Embedding model path does not exist: {self.model_path}")
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(str(self.model_path), local_files_only=True)
            self._backend = "sentence_transformers"
        except Exception as exc:
            self._model = None
            self._backend = "hashing"
            if self.require_sentence_transformers:
                raise RuntimeError(
                    f"Failed to load SentenceTransformer model from {self.model_path}: {exc}"
                ) from exc

    def encode(self, text: str) -> np.ndarray:
        """Encode a corpus document without a retrieval instruction."""
        if self._backend == "sentence_transformers" and self._model is not None:
            vector = self._model.encode([text], normalize_embeddings=True)[0]
            return np.asarray(vector, dtype=np.float32)
        return self._hash_encode(text)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a retrieval query with BGE's official query instruction."""
        if self._backend == "sentence_transformers" and self._model is not None:
            instructed = f"{BGE_LARGE_ZH_QUERY_INSTRUCTION}{text}"
            vector = self._model.encode([instructed], normalize_embeddings=True)[0]
            return np.asarray(vector, dtype=np.float32)
        # The deterministic fallback is not a BGE model. Prefixing it would add
        # unrelated tokens and reduce lexical overlap with corpus documents.
        return self._hash_encode(text)

    def encode_many(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.fallback_dim), dtype=np.float32)
        if self._backend == "sentence_transformers" and self._model is not None:
            vectors = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vectors, dtype=np.float32)
        vectors = [self._hash_encode(text) for text in texts]
        if not vectors:
            return np.zeros((0, self.fallback_dim), dtype=np.float32)
        return np.asarray(vectors, dtype=np.float32)

    def _hash_encode(self, text: str) -> np.ndarray:
        vec = np.zeros((self.fallback_dim,), dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.fallback_dim
            sign = -1.0 if int(digest[8:10], 16) % 2 == 0 else 1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


class VectorIndex:
    def __init__(self, encoder: Optional[VectorEncoder] = None) -> None:
        self.encoder = encoder or VectorEncoder()
        self.space_vectors: Dict[str, np.ndarray] = {}
        self.space_items: Dict[str, List[Dict]] = {}

    def _get_dim(self) -> int:
        for matrix in self.space_vectors.values():
            if matrix.size > 0:
                return int(matrix.shape[1])
        return self.encoder.fallback_dim

    def add(self, doc_id: str, text: str, metadata: Optional[Dict] = None, space: str = "default") -> None:
        vector = self.encoder.encode(text)
        self._append(space=space, vectors=vector.reshape(1, -1), items=[{"doc_id": doc_id, "text": text, "metadata": metadata or {}}])

    def add_many(self, entries: List[Dict], space: str = "default", batch_size: int = 128) -> None:
        if not entries:
            return
        texts = [entry["text"] for entry in entries]
        vectors = self.encoder.encode_many(texts=texts, batch_size=batch_size)
        self._append(space=space, vectors=vectors, items=entries)

    def _append(self, space: str, vectors: np.ndarray, items: List[Dict]) -> None:
        if vectors.size == 0 or not items:
            return
        if space not in self.space_vectors or self.space_vectors[space].size == 0:
            self.space_vectors[space] = vectors
            self.space_items[space] = []
        else:
            self.space_vectors[space] = np.vstack([self.space_vectors[space], vectors])
        self.space_items[space].extend(
            [
                {
                    "doc_id": item["doc_id"],
                    "text": item["text"],
                    "metadata": item.get("metadata", {}),
                }
                for item in items
            ]
        )

    def search(self, query: str, top_k: int = 20, spaces: Optional[List[str]] = None) -> List[VectorHit]:
        if not self.space_items:
            return []
        spaces = spaces or sorted(self.space_items.keys())
        query_vector = self.encoder.encode_query(query)
        if query_vector.shape[0] != self._get_dim():
            return []

        results: List[VectorHit] = []
        for space in spaces:
            vectors = self.space_vectors.get(space)
            items = self.space_items.get(space, [])
            if vectors is None or vectors.size == 0 or not items:
                continue
            scores = vectors @ query_vector
            ranked_indices = np.argsort(scores)[::-1][:top_k]
            for idx in ranked_indices:
                score = float(scores[idx])
                if score <= 0:
                    continue
                item = items[int(idx)]
                metadata = dict(item["metadata"])
                metadata["dense_space"] = space
                results.append(
                    VectorHit(
                        doc_id=item["doc_id"],
                        score=score,
                        text=item["text"],
                        metadata=metadata,
                    )
                )
        results.sort(key=lambda hit: hit.score, reverse=True)
        return results[:top_k]

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        vectors_dir = dir_path / "vectors"
        vectors_dir.mkdir(parents=True, exist_ok=True)
        for space, matrix in self.space_vectors.items():
            safe_name = space.replace("/", "_")
            np.save(vectors_dir / f"{safe_name}.npy", matrix)
        payload = {
            "backend": self.encoder.backend,
            "spaces": sorted(self.space_items.keys()),
            "space_items": self.space_items,
            "fallback_dim": self.encoder.fallback_dim,
        }
        (dir_path / "metadata.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, dir_path: Path, model_path: Path = DEFAULT_EMBED_MODEL_PATH) -> "VectorIndex":
        metadata_path = dir_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(str(metadata_path))
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        encoder = VectorEncoder(model_path=model_path, fallback_dim=int(payload.get("fallback_dim", 512)))
        index = cls(encoder=encoder)
        stored_backend = payload.get("backend")
        if stored_backend == "sentence_transformers" and encoder.backend != "sentence_transformers":
            raise RuntimeError(
                "Vector index was built with sentence_transformers, but the query encoder "
                f"could not load the model at {model_path}. Pass the same embedding model path."
            )
        # Backward compatibility with old single-space format.
        if "space_items" not in payload:
            old_items = payload.get("items", [])
            old_vectors_path = dir_path / "vectors.npy"
            old_vectors = (
                np.load(old_vectors_path)
                if old_vectors_path.exists()
                else np.zeros((0, encoder.fallback_dim), dtype=np.float32)
            )
            index.space_items = {"default": old_items}
            index.space_vectors = {"default": old_vectors}
            return index

        spaces = payload.get("spaces", [])
        index.space_items = payload.get("space_items", {})
        index.space_vectors = {}
        vectors_dir = dir_path / "vectors"
        for space in spaces:
            safe_name = space.replace("/", "_")
            vector_path = vectors_dir / f"{safe_name}.npy"
            if vector_path.exists():
                index.space_vectors[space] = np.load(vector_path)
            else:
                index.space_vectors[space] = np.zeros((0, encoder.fallback_dim), dtype=np.float32)
        dimensions = {int(matrix.shape[1]) for matrix in index.space_vectors.values() if matrix.ndim == 2 and matrix.size}
        if dimensions:
            probe_dim = int(encoder.encode("dimension check").shape[0])
            if dimensions != {probe_dim}:
                raise RuntimeError(
                    f"Query encoder dimension {probe_dim} does not match stored vector dimensions {sorted(dimensions)}."
                )
        return index

    @classmethod
    def load_optional(cls, dir_path: Path, model_path: Path = DEFAULT_EMBED_MODEL_PATH) -> "VectorIndex":
        """Load an on-disk vector index, or return an empty in-memory index."""
        try:
            return cls.load(dir_path, model_path=model_path)
        except FileNotFoundError:
            encoder = VectorEncoder(model_path=model_path)
            return cls(encoder=encoder)

    def embedded_doc_ids(self) -> set[str]:
        doc_ids: set[str] = set()
        for items in self.space_items.values():
            for item in items:
                doc_id = item.get("doc_id")
                if doc_id:
                    doc_ids.add(str(doc_id))
        return doc_ids
