import unittest

from searchagent_retrieval.query_adapter import adapt_bm25_query, adapt_dense_query


class QueryAdapterTests(unittest.TestCase):
    def test_removes_pubmed_syntax_and_comments(self):
        raw = (
            '(osimertinib[Title/Abstract]) AND ("brain metastases" OR CNS) '
            'AND (2020:2024[dp])\n# Chinese alternatives\n奥希替尼 AND 脑转移'
        )

        dense = adapt_dense_query(raw)
        bm25 = adapt_bm25_query(raw)

        for adapted in (dense, bm25):
            self.assertNotIn("AND", adapted)
            self.assertNotIn("OR", adapted)
            self.assertNotIn("[Title/Abstract]", adapted)
            self.assertNotIn("[dp]", adapted)
            self.assertNotIn("#", adapted)
            self.assertIn("osimertinib", adapted)
            self.assertIn("brain", adapted)

    def test_deduplicates_terms_and_caps_query(self):
        query = " AND ".join(["osimertinib"] * 80)

        self.assertEqual(adapt_dense_query(query), "osimertinib")
        self.assertEqual(adapt_bm25_query(query), "osimertinib")

    def test_dense_query_does_not_drop_late_clinical_terms(self):
        query = " ".join(
            [f"clinicalterm{i}" for i in range(60)]
            + ["osimertinib", "brain metastases", "脑转移"]
        )

        adapted = adapt_dense_query(query)

        self.assertIn("clinicalterm59", adapted)
        self.assertIn("osimertinib", adapted)
        self.assertIn("脑转移", adapted)


if __name__ == "__main__":
    unittest.main()
