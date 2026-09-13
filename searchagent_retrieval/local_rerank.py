from __future__ import annotations

from typing import Any, List, Optional

from .tools import SearchHit


DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_QWEN_RERANK_MODEL = "Qwen/Qwen3-Reranker-4B"


def _auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalCrossEncoderReranker:
    """Generic multilingual cross-encoder reranker with GPU/MPS support."""

    RERANK_POOL = 16

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_RERANK_MODEL,
        device: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Local reranking requires sentence-transformers; install requirements.txt"
            ) from exc

        self.model_name_or_path = str(model_name_or_path)
        self.device = device or _auto_device()
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(32, int(max_length))
        self.model = CrossEncoder(
            self.model_name_or_path,
            device=self.device,
            max_length=self.max_length,
            trust_remote_code=False,
        )

    def rerank(self, query: str, hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        if not hits:
            return []
        candidates = hits[: self.RERANK_POOL]
        scores = self.model.predict(
            [(query, hit.text) for hit in candidates],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            apply_softmax=False,
        )
        reranked: List[SearchHit] = []
        for hit, raw_score in zip(candidates, scores):
            score = float(raw_score)
            metadata: dict[str, Any] = dict(hit.metadata)
            metadata.update(
                {
                    "rerank_backend": "local_cross_encoder",
                    "rerank_model": self.model_name_or_path,
                    "rerank_device": self.device,
                    "rerank_status": "completed",
                    "rerank_relevance_score": score,
                    "hybrid_score": hit.score,
                }
            )
            reranked.append(
                SearchHit(
                    id=hit.id,
                    score=score,
                    source=hit.source,
                    text=hit.text,
                    metadata=metadata,
                )
            )
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]


class Qwen3Reranker:
    """Qwen3 instruction reranker. Scores query-document pairs on CUDA."""

    RERANK_POOL = 64
    INSTRUCTION = (
        "Rank oncology evidence by exact clinical applicability. Require matching cancer population, "
        "target intervention, monotherapy versus combination regimen, requested time window, reported "
        "outcome, direct result text, and outcome attribution. Demote citations, background text, wrong "
        "regimens, and outcomes outside the requested time window."
    )

    def __init__(self, model_name_or_path: str, device: Optional[str] = None, batch_size: int = 8) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Qwen reranking requires torch and transformers") from exc
        self.model_name_or_path = str(model_name_or_path)
        self.device = device or _auto_device()
        self.batch_size = max(1, int(batch_size))
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side="left")
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                dtype=torch.bfloat16,
                device_map={"": 0},
            ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path).to(self.device).eval()
        self.false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.last_ranked_pool: List[SearchHit] = []
        self.prefix = (
            '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
            'and the Instruct provided. Note that the answer can only be "yes" or "no".'
            '<|im_end|>\n<|im_start|>user\n'
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def _scores(self, query: str, hits: List[SearchHit]) -> List[float]:
        scores: List[float] = []
        torch = self._torch
        for start in range(0, len(hits), self.batch_size):
            batch = hits[start:start + self.batch_size]
            texts = [
                self.prefix
                + f"<Instruct>: {self.INSTRUCTION}\n<Query>: {query}\n<Document>: {hit.text}"
                + self.suffix
                for hit in batch
            ]
            inputs = self.tokenizer(
                texts, padding=True, truncation=True, max_length=2048, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits[:, -1, :]
                probabilities = torch.softmax(
                    torch.stack([logits[:, self.false_id], logits[:, self.true_id]], dim=1), dim=1
                )[:, 1].float().cpu().tolist()
            scores.extend(float(value) for value in probabilities)
        return scores

    def rerank(self, query: str, hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        if not hits:
            return []
        candidates = hits[: self.RERANK_POOL]
        scored = self._scores(query, candidates)
        reranked: List[SearchHit] = []
        for hit, score in zip(candidates, scored):
            metadata: dict[str, Any] = dict(hit.metadata)
            metadata.update({
                "qwen_rerank_score": score,
                "qwen_rerank_status": "completed",
                "qwen_rerank_model": self.model_name_or_path,
                "qwen_rerank_device": self.device,
                "hybrid_score": hit.score,
            })
            reranked.append(SearchHit(hit.id, score, hit.source, hit.text, metadata))
        reranked.sort(key=lambda item: item.score, reverse=True)
        self.last_ranked_pool = list(reranked)
        return reranked[:top_k]


class QwenThenLLMReranker:
    """Expand recall with Hybrid, compress with Qwen, then call LLM at most twice."""

    QWEN_TOP_K = 8
    MAX_LLM_REQUESTS = 2

    def __init__(self, llm_client: Any, model_name_or_path: str, device: Optional[str] = None) -> None:
        from .llm_rerank import LLMReranker

        self.qwen = Qwen3Reranker(model_name_or_path, device=device)
        self.llm = LLMReranker(llm_client)
        # Keep the safety cap local to this composite backend as well, so a
        # future change to the base reranker cannot increase API fan-out.
        self.llm.MAX_REQUESTS = self.MAX_LLM_REQUESTS

    def rerank(self, query: str, hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        qwen_hits = self.qwen.rerank(query, hits, top_k=self.QWEN_TOP_K)
        ranked = self.llm.rerank(query, qwen_hits, top_k=top_k)
        for hit in ranked:
            hit.metadata["rerank_backend"] = "qwen_then_llm"
            hit.metadata["qwen_prefilter_count"] = len(qwen_hits)
            hit.metadata["llm_final_candidate_count"] = len(qwen_hits)
            hit.metadata["llm_max_requests"] = self.MAX_LLM_REQUESTS
        return ranked


class QwenWithLLMFallbackReranker(QwenThenLLMReranker):
    """Use Qwen normally; EvidenceReview owns the evidence-driven LLM fallback."""

    QWEN_FALLBACK_THRESHOLD = 0.50

    def rerank(self, query: str, hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        if not hits:
            return []
        qwen_hits = self.qwen.rerank(query, hits, top_k=self.QWEN_TOP_K)
        ranked = qwen_hits[:top_k]
        fallback = False
        reason = "deferred_to_evidence_review"
        for hit in ranked:
            hit.metadata["rerank_backend"] = "qwen_with_llm_fallback"
            hit.metadata["qwen_prefilter_count"] = len(qwen_hits)
            hit.metadata["qwen_fallback_used"] = fallback
            hit.metadata["qwen_fallback_reason"] = reason
            hit.metadata["qwen_fallback_threshold"] = self.QWEN_FALLBACK_THRESHOLD
            hit.metadata["llm_max_requests"] = self.MAX_LLM_REQUESTS
            if fallback:
                hit.metadata["llm_final_candidate_count"] = len(qwen_hits)
        return ranked
