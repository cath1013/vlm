import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.train_pair_reranker_v2 import load_records, validation_report


class BucketPopulationTest(unittest.TestCase):
    def test_bucket_one_ignores_later_collision_and_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            rows = [
                {"scenario": "later", "actual": True,
                 "positive_intervals": [2], "target_interval": None,
                 "candidates": [
                     {"actor_a": "a", "actor_b": "b", "features": [1.0],
                      "label": True, "interval_index": 1,
                      "minimum_clearance_m": 0.0, "predicted_contact": True},
                     {"actor_a": "a", "actor_b": "b", "features": [2.0],
                      "label": True, "interval_index": 2,
                      "minimum_clearance_m": 0.0, "predicted_contact": True}]},
                {"scenario": "now", "actual": True,
                 "positive_intervals": [1], "target_interval": None,
                 "candidates": [
                     {"actor_a": "c", "actor_b": "d", "features": [3.0],
                      "label": True, "interval_index": 1,
                      "minimum_clearance_m": 0.0, "predicted_contact": True}]},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            samples, windows = load_records(path, target_interval=1)
            self.assertEqual([sample["y"] for sample in samples], [0.0, 1.0])
            self.assertEqual([window["actual"] for window in windows], [False, True])
            report = validation_report(samples, windows, np.array([0.1, 0.9]), 0.5)
            self.assertEqual(report["confusion"], {"tp": 1, "fp": 0, "tn": 1, "fn": 0})
            self.assertEqual(report["baseline_before_verifier"]["confusion"],
                             {"tp": 1, "fp": 1, "tn": 0, "fn": 0})
            self.assertEqual(report["correct_gt_actor_pair_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
