from __future__ import annotations

from pathlib import Path

from searchagent_retrieval.models import Document
from searchagent_retrieval.schema_parser import load_records_from_data_root
from searchagent_retrieval.vector_index import VectorEncoder, VectorIndex


def main() -> None:
    data_root = Path("case_pipeline_steps_data")
    records = load_records_from_data_root(data_root)
    print(f"loaded records: {len(records)}")

    encoder = VectorEncoder()
    print(f"encoder backend: {encoder.backend}")

    index = VectorIndex(encoder=encoder)
    sample = records[:3]
    for record in sample:
        for chunk in record.evidence_chunks:
            if not chunk.text_for_embedding:
                continue
            space = "case_semantic" if chunk.chunk_type == "case_text_chunk" else "event_semantic"
            index.add(
                doc_id=chunk.chunk_id,
                text=chunk.text_for_embedding,
                metadata={
                    "doc_id": chunk.doc_id,
                    "chunk_type": chunk.chunk_type,
                    "space": space,
                },
                space=space,
            )

    out_dir = Path("indexes_verify")
    out_dir.mkdir(parents=True, exist_ok=True)
    index.save(out_dir / "vector")
    print("saved vector index")

    # verify load and search
    loaded = VectorIndex.load(out_dir / "vector")
    print(f"loaded backend: {loaded.encoder.backend}")
    print(f"spaces: {sorted(loaded.space_items.keys())}")
    print(f"case_semantic items: {len(loaded.space_items.get('case_semantic', []))}")
    print(f"event_semantic items: {len(loaded.space_items.get('event_semantic', []))}")

    query = "EGFR exon 19 deletion osimertinib resistance"
    hits = loaded.search(query=query, top_k=5, spaces=["case_semantic", "event_semantic"])
    for hit in hits:
        print(f"score={hit.score:.4f} space={hit.metadata.get('dense_space')} id={hit.doc_id}")
        print(f"  text={hit.text[:120]}")


if __name__ == "__main__":
    main()
