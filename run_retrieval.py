from __future__ import annotations

import argparse
import json
from pathlib import Path

from searchagent_retrieval.router import RetrievalRouter
from searchagent_retrieval.tools import RetrievalTools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SearchAgent retrieval router.")
    parser.add_argument("--query", type=str, required=True, help="Doctor question.")
    parser.add_argument("--index-root", type=Path, default=Path("indexes"), help="Index root path.")
    parser.add_argument(
        "--embed-model-path",
        type=Path,
        default=None,
        help="Embedding model directory used to encode dense-search queries.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k results per tool.")
    parser.add_argument("--reranker-backend", choices=("local", "none"), default="none")
    parser.add_argument("--reranker-model", type=str, default=None)
    parser.add_argument("--reranker-device", choices=("cuda", "mps", "cpu"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tools = RetrievalTools(
        index_root=args.index_root,
        embed_model_path=args.embed_model_path,
        reranker_backend=args.reranker_backend,
        reranker_model=args.reranker_model,
        reranker_device=args.reranker_device,
    )
    try:
        router = RetrievalRouter(retrieval_tools=tools)
        result = router.run(query=args.query, top_k=args.top_k)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        tools.close()


if __name__ == "__main__":
    main()
