import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from examples import calibrate_pair_reranker_v4_0_2 as calibration
from examples.train_pair_reranker_v2 import fold_of


def doc(scenario, window, bucket, positive=None, candidates=None):
    return {"scenario": scenario, "window": window, "target_interval": bucket,
            "actual": positive == bucket,
            "positive_intervals": [] if positive is None else [positive],
            "candidates": candidates or []}


def scored(actual, score1, score2, pair1=-np.inf, pair2=-np.inf):
    return {"actual": actual, "positive_bucket": 1 if actual else None,
            "scores": {1: score1, 2: score2},
            "pair_scores": {1: pair1, 2: pair2},
            "gt_pair_covered": np.isfinite(pair1) or np.isfinite(pair2)}


class CalibrationTest(unittest.TestCase):
    def test_scenario_grouped_oof_has_no_training_leakage(self):
        # Multiple candidates per scenario and multiple scenarios per fold.
        scenarios = []
        for fold in range(5):
            scenarios.extend(s for s in (f"scenario-{i}" for i in range(200))
                             if fold_of(s, 5) == fold)
        samples = [{"scenario": s, "x": [float(i)] * len(calibration.FEATURE_NAMES_V4),
                    "y": 1.0} for i, s in enumerate(scenarios) for _ in range(2)]
        windows = [{"scenario": s} for s in scenarios]
        trained = []

        def fake_fit(x, y, hidden, weight_decay, seed):
            trained.append(set(x[:, 0]))
            return trained[-1], None, None

        def fake_predict(model, means, scales, x):
            self.assertFalse(model & set(x[:, 0]))
            return np.full(len(x), 0.75)

        with patch.object(calibration, "fit", side_effect=fake_fit), patch.object(
                calibration, "predict", side_effect=fake_predict):
            scores, audit = calibration.generate_oof(samples, windows, 32, 0.001)
        self.assertEqual(len(trained), 5)
        np.testing.assert_array_equal(scores, np.full(len(samples), 0.75))
        for row in audit:
            self.assertFalse(set(row["training_scenarios"]) & set(row["held_out_scenarios"]))
            for s in row["held_out_scenarios"]:
                self.assertEqual(fold_of(s, 5), row["fold"])

    def test_censor_aware_population_and_alignment(self):
        b1 = [doc("s", "positive1", 1, 1), doc("s", "positive2", 1, 2),
              doc("s", "negative", 1), doc("s", "censored", 1),
              doc("s", "late_collision", 1, 3)]
        b2 = [doc("s", "negative", 2), doc("s", "positive2", 2, 2),
              doc("s", "only2", 2)]
        population = calibration.build_population(
            calibration.index_records(b1, 1), calibration.index_records(b2, 2))
        self.assertEqual(calibration.population_counts(population),
                         {"valid_0_2_windows": 3, "positive_windows": 2, "negative_windows": 1})
        self.assertEqual([r["key"][1] for r in population], ["negative", "positive1", "positive2"])
        with self.assertRaisesRegex(ValueError, "census mismatch"):
            calibration.check_population(population)

    def test_sweep_keeps_bucket1_fixed_and_prefers_specificity(self):
        rows = [scored(True, .9, .1), scored(True, .1, .9),
                scored(False, .1, .6), scored(False, .1, .2)]
        original, threshold, selected, sweep = calibration.sweep_bucket2(rows, .8, .5)
        self.assertEqual(threshold, .9)
        self.assertEqual(original["confusion"]["fp"], 1)
        self.assertEqual(selected["confusion"]["fp"], 0)
        self.assertEqual(selected["recall"], original["recall"])
        self.assertEqual({r["bucket1_threshold"] for r in sweep}, {.8})
        self.assertGreater(len({r["bucket2_threshold"] for r in sweep}), 1)
        # Lowering bucket2 to improve recall must not beat a higher-specificity point.
        self.assertTrue(any(r["eligible"] for r in sweep))

    def test_selection_cannot_reduce_recall_and_retains_original_without_improvement(self):
        rows = [scored(True, .1, .6), scored(False, .1, .9)]
        original, threshold, selected, sweep = calibration.sweep_bucket2(rows, .8, .5)
        self.assertEqual(threshold, .5)
        self.assertGreaterEqual(selected["recall"], original["recall"])
        self.assertTrue(any(r["specificity"] > original["specificity"] and not r["eligible"]
                            for r in sweep))

    def test_selection_tie_chooses_higher_threshold(self):
        rows = [scored(True, .9, .95), scored(False, .1, .6),
                scored(False, .9, .7), scored(False, .9, .8)]
        _, threshold, selected, _ = calibration.sweep_bucket2(rows, .85, .5)
        self.assertEqual(threshold, float(np.nextafter(.95, np.inf)))
        self.assertEqual(selected["recall"], 1)

    def test_specificity_beats_higher_recall_when_both_meet_floor(self):
        rows = [scored(True, .1, .9), scored(True, .1, .4),
                scored(False, .1, .6), scored(False, .1, .2)]
        original, threshold, selected, sweep = calibration.sweep_bucket2(rows, .8, .5)
        self.assertEqual(original["recall"], .5)
        self.assertEqual(threshold, .9)
        self.assertEqual(selected["specificity"], 1)
        higher_recall = [r for r in sweep if r["recall"] > selected["recall"]]
        self.assertTrue(higher_recall)
        self.assertTrue(all(r["specificity"] < selected["specificity"] for r in higher_recall))

    def test_pair_metric_uses_above_threshold_pairs_from_either_bucket(self):
        candidates = [{"actor_a": "b", "actor_b": "a_alias", "interval_index": 2,
                       "label": False}]
        d1, d2 = doc("s", "w", 1, 1), doc("s", "w", 2, 1, candidates)
        population = calibration.build_population({("s", "w"): d1}, {("s", "w"): d2})
        calibration.attach_scores(population, {1: {}, 2: {("s", "w"): np.array([.9])}},
                                  {("s", "w"): [["a", "a_alias"], ["b"]]})
        before = calibration.combined_metrics(population, .8, .9)
        after = calibration.combined_metrics(population, .8, .91)
        self.assertEqual(before["correct_gt_actor_pair_recall"], 1)
        self.assertEqual(after["correct_gt_actor_pair_recall"], 0)
        self.assertEqual(before["alarms"]["bucket2_only"], 1)

    def test_training_config_does_not_consult_validation_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps({"training": {"hidden": 16, "weight_decay": .01,
                                                      "folds": 5, "feature_version": 4,
                                                      "target_interval": 2},
                                        "held_out_validation": {"hidden": 99}}))
            self.assertEqual(calibration.load_training_config(path, 2),
                             {"hidden": 16, "weight_decay": .01})

    def test_cli_refuses_to_overwrite_models_or_inputs_before_reading(self):
        for output in (str(calibration.BASE / "model_bucket1/pair_reranker_v4.json"),
                       str(calibration.BASE / "train_bucket2.jsonl"),
                       str(calibration.BASE / "train_bucket1.manifest.json")):
            with self.subTest(output=output), patch.object(calibration, "read_jsonl") as read:
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    calibration.main(["--out", output])
                read.assert_not_called()

    def test_recovery_rejects_non_train_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "TRAIN directory"):
                calibration.recover_train_gt([], {"dataset_root": tmp}, tmp)

    def test_missing_gt_groups_cannot_silently_use_interval_labels(self):
        population = calibration.build_population(
            {("s", "w"): doc("s", "w", 1, 1)}, {})
        with self.assertRaisesRegex(ValueError, "missing GT actor groups"):
            calibration.attach_scores(population, {1: {}, 2: {}}, {})

    def test_validation_paths_rejected_before_access(self):
        with patch.object(calibration, "read_jsonl") as read:
            with self.assertRaisesRegex(ValueError, "validation input paths"):
                calibration.main(["--train-bucket1", "/never/read/val104/data.jsonl",
                                  "--out", "/tmp/calibration-test.json"])
            read.assert_not_called()

    def test_cli_writes_diagnostics_with_embedded_train_gt_and_preserves_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = ["--out", str(root / "diagnostic.json")]
            counts = {"valid_0_2_windows": 3, "positive_windows": 2, "negative_windows": 1}
            model_paths = []
            for bucket, hidden in ((1, 16), (2, 32)):
                data = root / f"train_bucket{bucket}.jsonl"
                records = []
                for window, positive, score1, score2 in (
                        ("w1", 1, .9, .1), ("w2", 2, .1, .9), ("w3", None, .1, .6)):
                    candidate = {"actor_a": "a", "actor_b": "b",
                                 "interval_index": bucket, "label": positive == bucket,
                                 "minimum_clearance_m": 0,
                                 "features": [score1 if bucket == 1 else score2] * len(calibration.FEATURE_NAMES_V4)}
                    record = doc("s", window, bucket, positive, [candidate])
                    if positive:
                        record["gt_actor_groups"] = [["a"], ["b"]]
                    records.append(record)
                data.write_text("\n".join(json.dumps(r) for r in records) + "\n")
                data.with_suffix(".manifest.json").write_text(json.dumps(
                    {"split": "train", "feature_version": 4, "predictor_mode": "joint_scene",
                     "target_interval": bucket}))
                report = root / f"report{bucket}.json"
                report.write_text(json.dumps({"training": {"hidden": hidden, "weight_decay": .001,
                                                           "folds": 5, "feature_version": 4,
                                                           "target_interval": bucket}}))
                model = root / f"model{bucket}.json"
                model.write_text("{\"decision_threshold\": 0.5}")
                model_paths.append(model)
                argv.extend([f"--train-bucket{bucket}", str(data), f"--bucket{bucket}-report",
                             str(report), f"--bucket{bucket}-model", str(model)])
            before = [p.read_bytes() for p in model_paths]
            with patch.object(calibration, "EXPECTED_COUNTS", counts), patch.object(
                    calibration, "load_frozen_models", return_value={
                        1: SimpleNamespace(threshold=.8), 2: SimpleNamespace(threshold=.5)}), patch.object(
                    calibration, "generate_oof", side_effect=lambda samples, windows, **config: (
                        np.array([s["x"][0] for s in samples]), [])) as oof, patch.object(
                    calibration, "recover_train_gt") as recover:
                calibration.main(argv)
            recover.assert_not_called()
            self.assertEqual([call.kwargs["hidden"] for call in oof.call_args_list], [16, 32])
            result = json.loads((root / "diagnostic.json").read_text())
            self.assertEqual(result["train_population_counts"], counts)
            self.assertEqual(result["original_bucket1_threshold"], result["selected_bucket1_threshold"])
            self.assertEqual(result["selected_bucket2_threshold"], .9)
            self.assertEqual(result["absolute_fp_reduction"], 1)
            self.assertEqual(result["percent_fp_reduction"], 100)
            self.assertEqual(result["tp_change"], 0)
            self.assertEqual(result["fn_change"], 0)
            self.assertEqual(result["correct_pair_recall_change"], 0)
            self.assertEqual(result["bucket2_only_alarms_before"], 2)
            self.assertEqual(result["bucket2_only_alarms_after"], 1)
            self.assertEqual([p.read_bytes() for p in model_paths], before)


if __name__ == "__main__":
    unittest.main()
