import unittest

import numpy as np

from src.retrieval.eval_utils import dcg, evaluate_retrieval


class RetrievalMetricsTest(unittest.TestCase):
    def test_dcg_and_retrieval_scores(self):
        self.assertEqual(dcg([], 10), 0.0)
        self.assertEqual(dcg([1, 0, 1], 2), 1.0)
        actual = dcg(np.array([1, 0, 1], dtype=np.float32), 3)
        self.assertAlmostEqual(actual, 1 + 1 / np.log2(3), places=12)
        recall_any, recall_all, ndcg = evaluate_retrieval(
            [2, 0, 1], ["a", "c"], ["a", "b", "c"], 2
        )
        self.assertEqual((recall_any, recall_all), (1.0, 1.0))
        self.assertAlmostEqual(ndcg, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
