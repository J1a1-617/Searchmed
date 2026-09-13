from __future__ import annotations

import argparse, json, time
from pathlib import Path

from searchagent_retrieval.local_rerank import Qwen3Reranker
from searchagent_retrieval.tools import RetrievalTools


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    data = json.loads(Path(args.data).read_text())[: args.limit]
    started = time.perf_counter()
    tools = RetrievalTools(Path(args.index), reranker_backend="none")
    qwen = Qwen3Reranker(args.model, device=args.device)
    rows, errors = [], []
    try:
        for case in data:
            case_started = time.perf_counter()
            try:
                query = json.dumps(case.get("input") or {}, ensure_ascii=False)
                hybrid = tools.hybrid_search(query, {}, top_k=64)
                ranked = qwen.rerank(query, hybrid, top_k=8)
                ids = [hit.id for hit in ranked]
                fetched = tools.fetch_evidence(chunk_ids=ids, limit=8)
                rows.append({"instance_id": case.get("instance_id"), "hybrid": len(hybrid), "qwen": len(ranked), "fetched": len(fetched), "seconds": round(time.perf_counter() - case_started, 3)})
            except Exception as exc:
                errors.append({"instance_id": case.get("instance_id"), "error": f"{type(exc).__name__}: {exc}"})
    finally:
        tools.close()
    summary = {"requested": len(data), "completed": len(rows), "errors": len(errors), "empty_qwen": sum(x["qwen"] == 0 for x in rows), "empty_fetch": sum(x["fetched"] == 0 for x in rows), "avg_seconds": round(sum(x["seconds"] for x in rows) / max(1, len(rows)), 3), "elapsed_seconds": round(time.perf_counter() - started, 3), "rows": rows, "errors_detail": errors}
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in {"rows", "errors_detail"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
