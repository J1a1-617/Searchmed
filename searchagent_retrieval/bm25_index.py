from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .text_utils import tokenize


@dataclass
class BM25Hit:
    doc_id: str
    score: float
    text: str
    metadata: Dict


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: List[Dict] = []
        self.doc_freq: Dict[str, int] = {}
        self.avg_len = 0.0
        self.term_freqs: List[Dict[str, int]] = []

    def add(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> None:
        tokens = tokenize(text)
        if not tokens:
            return
        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        for token in tf:
            self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

        self.docs.append(
            {
                "doc_id": doc_id,
                "text": text,
                "length": len(tokens),
                "metadata": metadata or {},
            }
        )
        self.term_freqs.append(tf)
        total_len = sum(item["length"] for item in self.docs)
        self.avg_len = total_len / len(self.docs)

    def _idf(self, term: str) -> float:
        n_docs = len(self.docs)
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 20) -> List[BM25Hit]:
        if not self.docs:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: List[float] = [0.0 for _ in self.docs]
        for term in query_tokens:
            idf = self._idf(term)
            if idf == 0:
                continue
            for idx, tf in enumerate(self.term_freqs):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                doc_len = self.docs[idx]["length"]
                denom = f + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_len, 1e-6))
                scores[idx] += idf * f * (self.k1 + 1) / denom

        ranked = sorted(
            [
                BM25Hit(
                    doc_id=self.docs[idx]["doc_id"],
                    score=score,
                    text=self.docs[idx]["text"],
                    metadata=self.docs[idx]["metadata"],
                )
                for idx, score in enumerate(scores)
                if score > 0
            ],
            key=lambda item: item.score,
            reverse=True,
        )
        return ranked[:top_k]

    def filtered(self, predicate: Callable[[Dict], bool]) -> "BM25Index":
        """Return an exact corpus-level subset without re-tokenizing documents.

        Recomputing document frequencies and average length is important for
        ablations: filtering only the final top-k would still let removed
        documents change BM25 IDF values and displace eligible candidates.
        """
        result = BM25Index(k1=self.k1, b=self.b)
        for doc, term_freq in zip(self.docs, self.term_freqs):
            if not predicate(doc):
                continue
            result.docs.append(dict(doc))
            copied_tf = dict(term_freq)
            result.term_freqs.append(copied_tf)
            for token in copied_tf:
                result.doc_freq[token] = result.doc_freq.get(token, 0) + 1
        if result.docs:
            result.avg_len = sum(item["length"] for item in result.docs) / len(result.docs)
        return result

    def save(self, path: Path) -> None:
        payload = {
            "k1": self.k1,
            "b": self.b,
            "docs": self.docs,
            "doc_freq": self.doc_freq,
            "avg_len": self.avg_len,
            "term_freqs": self.term_freqs,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        payload = json.loads(path.read_text(encoding="utf-8"))
        idx = cls(k1=payload["k1"], b=payload["b"])
        idx.docs = payload["docs"]
        idx.doc_freq = {k: int(v) for k, v in payload["doc_freq"].items()}
        idx.avg_len = float(payload["avg_len"])
        idx.term_freqs = [{k: int(v) for k, v in tf.items()} for tf in payload["term_freqs"]]
        return idx
