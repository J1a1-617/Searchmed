import unittest

from searchagent_retrieval.text_utils import min_max_normalize
from searchagent_retrieval.tools import RetrievalTools, SearchHit


def _hit(hit_id: str, score: float, source: str) -> SearchHit:
    return SearchHit(id=hit_id, score=score, source=source, text=hit_id, metadata={})


class HybridScoreNormalizationTests(unittest.TestCase):
    def test_min_max_normalize_handles_regular_equal_and_empty_scores(self) -> None:
        self.assertEqual(min_max_normalize({"a": 10.0, "b": 20.0}), {"a": 0.0, "b": 1.0})
        self.assertEqual(min_max_normalize({"a": 2.0, "b": 2.0}), {"a": 1.0, "b": 1.0})
        self.assertEqual(min_max_normalize({"a": 0.0}), {"a": 0.0})
        self.assertEqual(min_max_normalize({}), {})

    def test_hybrid_search_weights_normalized_channel_scores(self) -> None:
        tools = RetrievalTools.__new__(RetrievalTools)
        tools.structured_search = lambda **kwargs: [
            _hit("a", 100.0, "structured"),
            _hit("b", 50.0, "structured"),
        ]
        tools.dense_search = lambda **kwargs: [
            _hit("a", 0.1, "dense"),
            _hit("b", 0.9, "dense"),
        ]
        tools.bm25_search = lambda **kwargs: [
            _hit("a", 1000.0, "bm25"),
            _hit("b", 10.0, "bm25"),
        ]

        results = tools.hybrid_search(query="q", top_k=2)

        self.assertEqual([hit.id for hit in results], ["a", "b"])
        self.assertAlmostEqual(results[0].score, 0.60)
        self.assertAlmostEqual(results[1].score, 0.40)
        self.assertEqual(
            results[0].metadata["component_scores"],
            {"structured": 100.0, "dense": 0.1, "bm25": 1000.0},
        )
        self.assertEqual(
            results[0].metadata["normalized_component_scores"],
            {"structured": 1.0, "dense": 0.0, "bm25": 1.0},
        )


if __name__ == "__main__":
    unittest.main()
