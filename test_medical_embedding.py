from __future__ import annotations

from pathlib import Path

from searchagent_retrieval.schema_parser import load_records_from_data_root
from searchagent_retrieval.vector_index import VectorEncoder, VectorIndex


def main() -> None:
    print("Loading encoder...")
    encoder = VectorEncoder()
    print(f"Encoder backend: {encoder.backend}")
    if encoder.backend != "sentence_transformers":
        print("ERROR: not using medical embedding model")
        return

    print("Loading a few records...")
    records = load_records_from_data_root(Path("case_pipeline_steps_data"))[:10]
    print(f"Loaded {len(records)} records")

    index = VectorIndex(encoder=encoder)
    for record in records:
        for chunk in record.evidence_chunks:
            if not chunk.text_for_embedding:
                continue
            space = "case_semantic" if chunk.chunk_type == "case_text_chunk" else "event_semantic"
            index.add(
                doc_id=chunk.chunk_id,
                text=chunk.text_for_embedding,
                metadata={"doc_id": chunk.doc_id, "chunk_type": chunk.chunk_type, "space": space},
                space=space,
            )

    out_dir = Path("test_indexes")
    out_dir.mkdir(parents=True, exist_ok=True)
    index.save(out_dir)
    print(f"Saved index to {out_dir}")

    query = "EGFR exon 19 deletion 患者使用 osimertinib 后进展"
    hits = index.search(query=query, top_k=5, spaces=["case_semantic", "event_semantic"])
    print(f"\nTop hits for query: {query}")
    for hit in hits:
        print(f"  {hit.doc_id} | {hit.metadata.get('dense_space')} | score={hit.score:.4f}")


if __name__ == "__main__":
    main()
