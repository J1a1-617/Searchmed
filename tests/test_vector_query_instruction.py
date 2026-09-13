import tempfile
import unittest
from pathlib import Path

import numpy as np

from searchagent_retrieval.vector_index import (
    BGE_LARGE_ZH_QUERY_INSTRUCTION,
    VectorEncoder,
    VectorIndex,
)


class _RecordingSentenceTransformer:
    def __init__(self) -> None:
        self.inputs = []

    def encode(self, texts, **kwargs):
        self.inputs.append((list(texts), dict(kwargs)))
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class _SearchEncoder:
    backend = "test"
    fallback_dim = 2

    def __init__(self) -> None:
        self.document_inputs = []
        self.query_inputs = []

    def encode(self, text):
        self.document_inputs.append(text)
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def encode_many(self, texts, batch_size=128):
        self.document_inputs.extend(texts)
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_query(self, text):
        self.query_inputs.append(text)
        return np.asarray([1.0, 0.0], dtype=np.float32)


class VectorQueryInstructionTests(unittest.TestCase):
    def test_sentence_transformer_instruction_is_query_only(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = VectorEncoder(model_path=Path(directory))
        model = _RecordingSentenceTransformer()
        encoder._backend = "sentence_transformers"
        encoder._model = model

        encoder.encode("文档正文")
        encoder.encode_query("奥希替尼脑膜转移")

        self.assertEqual(model.inputs[0][0], ["文档正文"])
        self.assertEqual(
            model.inputs[1][0],
            [f"{BGE_LARGE_ZH_QUERY_INSTRUCTION}奥希替尼脑膜转移"],
        )
        self.assertTrue(model.inputs[0][1]["normalize_embeddings"])
        self.assertTrue(model.inputs[1][1]["normalize_embeddings"])

    def test_vector_search_uses_query_encoder(self):
        encoder = _SearchEncoder()
        index = VectorIndex(encoder=encoder)
        index.add("doc-1", "病例文档")

        hits = index.search("检索问题", top_k=1)

        self.assertEqual(encoder.document_inputs, ["病例文档"])
        self.assertEqual(encoder.query_inputs, ["检索问题"])
        self.assertEqual([hit.doc_id for hit in hits], ["doc-1"])

    def test_hash_fallback_does_not_add_bge_instruction(self):
        encoder = VectorEncoder(model_path=Path("/path/that/does/not/exist"))

        np.testing.assert_array_equal(
            encoder.encode_query("原始查询"),
            encoder._hash_encode("原始查询"),
        )


if __name__ == "__main__":
    unittest.main()
