import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from searchagent_retrieval.tools import RetrievalTools


class RerankerBackendSelectionTests(unittest.TestCase):
    @patch("searchagent_retrieval.local_rerank.Qwen3Reranker")
    @patch("searchagent_retrieval.tools.BM25Index.load")
    @patch("searchagent_retrieval.tools.VectorIndex.load_optional")
    @patch("searchagent_retrieval.tools.StructuredIndex")
    def test_qwen_backend_uses_qwen_causal_lm_scorer(
        self,
        structured_index,
        vector_load,
        bm25_load,
        qwen_reranker,
    ):
        expected = MagicMock()
        qwen_reranker.return_value = expected

        tools = RetrievalTools(
            Path("/tmp/nonexistent-index-for-mocked-test"),
            reranker_backend="qwen",
            reranker_model="/models/Qwen3-Reranker-4B",
            reranker_device="cuda",
        )

        self.assertIs(tools.reranker, expected)
        qwen_reranker.assert_called_once_with(
            "/models/Qwen3-Reranker-4B",
            device="cuda",
        )


if __name__ == "__main__":
    unittest.main()
