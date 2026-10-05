import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from examples.evaluate_pair_reranker_v4_0_2 import (
    DEFAULT_BUCKET1_MODEL, DEFAULT_BUCKET2_MODEL, bucket_pools,
    evaluate_window, evaluation_truth, load_frozen_models, metric_report,
)


def gt(*, collision_bucket=None, scorable1=True, scorable2=True):
    return {"expected": [
        {"k": k, "scorable": scorable,
         "accident_expected": collision_bucket == k,
         "involved_vehicles": ([{"actor_ids": ["a", "a_alias"]},
                                {"actor_ids": ["b"]}]
                               if collision_bucket == k else [])}
        for k, scorable in ((1, scorable1), (2, scorable2))]}


def candidate(a, b, score, contact=False):
    return {"actor_a": a, "actor_b": b, "features": [score],
            "predicted_contact": contact}


class FakeModel:
    def __init__(self, threshold):
        self.threshold = threshold

    def predict_features(self, features):
        return features[0]


class CombinedV4EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.models = {1: FakeModel(0.6), 2: FakeModel(0.8)}

    def test_bucket1_collision_valid_when_bucket2_unscorable(self):
        row = evaluate_window(gt(collision_bucket=1, scorable2=False),
                              {1: [], 2: []}, self.models)
        self.assertTrue(row["actual"])
        self.assertEqual(row["positive_bucket"], 1)

    def test_negative_requires_both_buckets_scorable(self):
        self.assertIsNone(evaluation_truth(gt(scorable2=False)))
        self.assertIsNone(evaluation_truth(gt(scorable1=False)))
        self.assertEqual(evaluation_truth(gt()), (None, []))

    def test_combined_prediction_is_or_and_pair_can_come_from_either_bucket(self):
        rows = [
            evaluate_window(gt(collision_bucket=1),
                            {1: [candidate("x", "y", 0.7)],
                             2: [candidate("a_alias", "b", 0.9, True)]},
                            self.models),
            evaluate_window(gt(collision_bucket=2),
                            {1: [candidate("a", "b", 0.7, True)],
                             2: [candidate("x", "y", 0.2)]}, self.models),
            evaluate_window(gt(), {1: [], 2: []}, self.models),
        ]
        self.assertEqual([(r["bucket1_prediction"], r["bucket2_prediction"],
                           r["predicted"]) for r in rows],
                         [(True, True, True), (True, False, True),
                          (False, False, False)])
        self.assertTrue(all(r["correct_pair_hit"] for r in rows[:2]))
        report = metric_report(rows, "predicted", "correct_pair_hit")
        self.assertEqual(report["confusion"], {"tp": 2, "fp": 0, "tn": 1, "fn": 0})
        self.assertEqual(report["correct_gt_actor_pair_hits"], 2)
        self.assertEqual(report["gt_pair_candidate_coverage"], 1.0)
        baseline = metric_report(rows, "baseline_predicted", "baseline_correct_pair_hit")
        self.assertEqual(baseline["correct_gt_actor_pair_hits"], 2)

    def test_interval_filter_precedes_top_50(self):
        pairs = [SimpleNamespace(interval_index=2, actor_a=str(i)) for i in range(50)]
        pairs += [SimpleNamespace(interval_index=1, actor_a="target")]
        pools = bucket_pools(pairs)
        self.assertEqual([p.actor_a for p in pools[1]], ["target"])
        self.assertEqual(len(pools[2]), 50)

    def test_frozen_thresholds_are_loaded_unchanged(self):
        models = load_frozen_models(DEFAULT_BUCKET1_MODEL, DEFAULT_BUCKET2_MODEL)
        for bucket, path in ((1, DEFAULT_BUCKET1_MODEL), (2, DEFAULT_BUCKET2_MODEL)):
            saved = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(models[bucket].threshold, saved["decision_threshold"])
        # Scores on opposite sides of the saved gate retain their decisions.
        for bucket, model in models.items():
            score = model.threshold
            rows = {1: [], 2: []}
            rows[bucket] = [candidate("x", "y", score)]
            original = model.predict_features
            try:
                model.predict_features = lambda _: score
                self.assertTrue(evaluate_window(gt(), rows, models)["predicted"])
                model.predict_features = lambda _: score / 2
                self.assertFalse(evaluate_window(gt(), rows, models)["predicted"])
            finally:
                model.predict_features = original


if __name__ == "__main__":
    unittest.main()
