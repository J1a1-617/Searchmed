from __future__ import annotations

import json
import os
from pathlib import Path

from searchagent_retrieval.llm_client import LLMClient
from searchagent_retrieval.llm_rerank import LLMReranker
from searchagent_retrieval.tools import RetrievalTools, SearchHit


ROOT = Path(__file__).resolve().parents[1]
QUERY = (
    "EGFR-mutant lung adenocarcinoma with leptomeningeal metastases treated with "
    "single-agent furmonertinib, reporting early CNS radiographic, CSF or symptom response. "
    "The target antitumor regimen is furmonertinib monotherapy."
)
TARGETS = {"40519289", "41836239"}


def load_env() -> None:
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def doc(hit: SearchHit) -> str:
    return str(hit.metadata.get("doc_id") or hit.id.split("#", 1)[0])


def explicit_combo(hit: SearchHit) -> bool:
    text = " ".join(hit.text.lower().split())
    return any(term in text for term in (
        "osimertinib plus", "furmonertinib plus", "combined with", "combination therapy of",
        "gefitinib plus", "almonertinib plus", "aumolertinib plus",
    ))


def unique(hits: list[SearchHit]) -> list[SearchHit]:
    output, seen = [], set()
    for hit in hits:
        if hit.id not in seen:
            seen.add(hit.id)
            output.append(hit)
    return output


def summarize(name: str, hits: list[SearchHit]) -> dict:
    ranks = {}
    for index, hit in enumerate(hits, 1):
        if doc(hit) in TARGETS and doc(hit) not in ranks:
            ranks[doc(hit)] = index
    return {
        "name": name,
        "target_ranks": ranks,
        "explicit_combo_top5": sum(explicit_combo(hit) for hit in hits[:5]),
        "top10": [{"rank": i, "id": hit.id, "doc_id": doc(hit), "score": round(hit.score, 4), "explicit_combo": explicit_combo(hit), "text": " ".join(hit.text.split())[:180]} for i, hit in enumerate(hits[:10], 1)],
    }


def main() -> None:
    load_env()
    tools = RetrievalTools(ROOT / "indexes")
    dense = tools.dense_search(QUERY, top_k=30, spaces=["case_semantic", "event_semantic", "structured_semantic"])
    bm25 = tools.bm25_search("furmonertinib monotherapy single-agent leptomeningeal LMD CSF response", top_k=30)
    old_pool = unique(dense)[:16]
    new_pool = unique([*bm25[:3], *dense[:3], *dense, *bm25])[:16]
    llm = LLMClient(timeout=300, max_retries=1, structured_attempts=1)
    reranker = LLMReranker(llm)
    old_ranked = reranker.rerank(QUERY, old_pool, top_k=16)
    new_ranked = reranker.rerank(QUERY, new_pool, top_k=16)
    print(json.dumps({
        "query": QUERY,
        "before_recall": summarize("old_dense_dominated_pool", old_pool),
        "after_recall": summarize("new_reserved_bm25_dense_pool", new_pool),
        "before_rerank": summarize("old_pool_new_reranker", old_ranked),
        "after_rerank": summarize("new_pool_new_reranker", new_ranked),
    }, ensure_ascii=False))
    tools.close()


if __name__ == "__main__":
    main()
