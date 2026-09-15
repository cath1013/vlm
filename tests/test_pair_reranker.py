import unittest

from traffic_llm.pair_reranker import (
    FEATURE_NAMES, PairReranker, pair_features,
)


class PairRerankerTest(unittest.TestCase):
    def test_features_are_symmetric_in_actor_order(self):
        self.assertEqual(
            pair_features("EGO_a", "T003", 0.0, 2),
            pair_features("T003", "EGO_a", 0.0, 2),
        )

    def test_model_round_trip(self):
        n = len(FEATURE_NAMES)
        model = PairReranker(
            weights=[0.1] * n, bias=-0.2, means=[0.0] * n,
            scales=[1.0] * n, threshold=0.6,
        )
        restored = PairReranker.from_dict(model.to_dict())
        self.assertAlmostEqual(
            model.predict_proba("V001", "V002", 0.0, 1),
            restored.predict_proba("V001", "V002", 0.0, 1),
        )


if __name__ == "__main__":
    unittest.main()
