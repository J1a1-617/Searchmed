from __future__ import annotations

import argparse
import json
from pathlib import Path

from searchagent_retrieval.index_builder import build_indexes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SearchAgent retrieval indexes.")
    parser.add_argument(
        "--case-data-root",
        type=Path,
        default=Path("case_pipeline_steps_data"),
        help="Path to case pipeline data root (step1/2/3 dirs).",
    )
    parser.add_argument(
        "--external-data-root",
        type=Path,
        default=Path("data"),
        help="Path to external rule data root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("indexes"),
        help="Output directory for built indexes.",
    )
    parser.add_argument(
        "--skip-structured",
        action="store_true",
        help="Reuse existing structured.db instead of rebuilding it.",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip vector recomputation while refreshing existing vector metadata and other indexes.",
    )
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Reuse existing bm25.json instead of rebuilding it.",
    )
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Only embed chunks from the first N records (or N records after --record-offset).",
    )
    parser.add_argument(
        "--record-offset",
        type=int,
        default=0,
        help="Start embedding from this record index when using --limit-records.",
    )
    parser.add_argument(
        "--append-embedding",
        action="store_true",
        help="Append new embeddings to an existing vector index instead of rebuilding it.",
    )
    parser.add_argument(
        "--refresh-embedding-space",
        action="append",
        choices=["case_semantic", "event_semantic", "structured_semantic"],
        default=[],
        help=(
            "When appending, discard and recompute this embedding space. "
            "Repeat the option to refresh more than one space."
        ),
    )
    parser.add_argument(
        "--embed-model-path",
        type=Path,
        default=None,
        help="SentenceTransformer model directory. Overrides the development-machine default.",
    )
    parser.add_argument(
        "--require-sentence-transformers",
        action="store_true",
        help="Fail immediately instead of silently falling back to the hashing encoder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_indexes(
        case_data_root=args.case_data_root,
        external_data_root=args.external_data_root,
        output_root=args.output_root,
        skip_structured=args.skip_structured,
        skip_embedding=args.skip_embedding,
        skip_bm25=args.skip_bm25,
        limit_records=args.limit_records,
        record_offset=args.record_offset,
        append_embedding=args.append_embedding,
        refresh_embedding_spaces=args.refresh_embedding_space,
        embed_model_path=args.embed_model_path,
        require_sentence_transformers=args.require_sentence_transformers,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
