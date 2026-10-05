import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from examples import train_pair_reranker_v4_window as experiment
from examples.train_pair_reranker_v2 import fit, fold_of
from traffic_llm.pair_reranker_v2 import FEATURE_NAMES_V4, PairRerankerV4


def window(start, stop, actual, scenario="s"):
    return {"start": start, "stop": stop, "actual": actual, "scenario": scenario}


def config(tp=116, fp=5, pair_recall=.5, f1=.5, weights=(0.0, 0.0)):
    return {"lambda_window": weights[0], "lambda_rank": weights[1],
            "combined_oof": {"confusion": {"tp": tp, "fp": fp},
                             "correct_gt_actor_pair_recall": pair_recall, "f1": f1}}


class WindowObjectiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def losses(self, values, labels, windows):
        logits = torch.tensor(values, dtype=torch.float32, requires_grad=True)
        layout = experiment.make_window_layout(windows, labels)
        losses = experiment.window_and_ranking_losses(logits, layout)
        return logits, *losses

    def test_negative_window_uses_highest_logit_and_only_its_gradient(self):
        logits, loss, rank = self.losses([-4, 2, 0], [0, 0, 0], [window(0, 3, False)])
        self.assertAlmostEqual(loss.item(), F.softplus(torch.tensor(2.)).item())
        self.assertEqual(rank.item(), 0)
        loss.backward()
        self.assertEqual(logits.grad[0].item(), 0)
        self.assertGreater(logits.grad[1].item(), 0)
        self.assertEqual(logits.grad[2].item(), 0)

    def test_positive_window_uses_highest_true_gt_candidate_never_wrong(self):
        logits, loss, _ = self.losses([5, -1, -2], [0, 1, 1], [window(0, 3, True)])
        self.assertAlmostEqual(loss.item(), F.softplus(torch.tensor(1.)).item())
        loss.backward()
        self.assertEqual(logits.grad[0].item(), 0)
        self.assertLess(logits.grad[1].item(), 0)
        self.assertEqual(logits.grad[2].item(), 0)

    def test_gt_absent_positive_gets_no_window_or_rank_target(self):
        logits, loss, rank = self.losses([-3, 5], [0, 0], [window(0, 2, True)])
        self.assertEqual(loss.item(), 0)
        self.assertEqual(rank.item(), 0)
        (loss + rank).backward()
        torch.testing.assert_close(logits.grad, torch.zeros(2), rtol=0, atol=0)
        # Pair BCE still treats those wrong candidates as negatives.
        labels = torch.zeros(2)
        total = experiment.training_objective(logits, labels, torch.tensor(1.),
                                              experiment.make_window_layout([window(0, 2, True)], [0, 0]), 1., .5)
        torch.testing.assert_close(total, nn.BCEWithLogitsLoss()(logits, labels))

    def test_ranking_penalizes_the_hardest_wrong_pair(self):
        logits, _, rank = self.losses([0, 3, 1], [1, 0, 0], [window(0, 3, True)])
        self.assertAlmostEqual(rank.item(), F.softplus(torch.tensor(4.)).item())
        rank.backward()
        self.assertLess(logits.grad[0].item(), 0)
        self.assertGreater(logits.grad[1].item(), 0)
        self.assertEqual(logits.grad[2].item(), 0)
        _, _, easier = self.losses([2, 0, -1], [1, 0, 0], [window(0, 3, True)])
        self.assertGreater(rank.item(), easier.item())

    def test_ranking_skips_window_without_wrong_pair(self):
        _, loss, rank = self.losses([0, 3], [1, 1], [window(0, 2, True)])
        self.assertAlmostEqual(loss.item(), F.softplus(torch.tensor(-3.)).item())
        self.assertEqual(rank.item(), 0)

    def test_window_loss_balances_group_means_instead_of_window_counts(self):
        _, loss, _ = self.losses([0, 2, -1, -2, 2, 4], [0, 0, 0, 0, 1, 0],
                                [window(0, 2, False), window(2, 4, False), window(4, 6, True)])
        negative = (F.softplus(torch.tensor(2.)) + F.softplus(torch.tensor(-1.))) / 2
        expected = (negative + F.softplus(torch.tensor(-2.))) / 2
        torch.testing.assert_close(loss, expected)

    def test_empty_windows_have_no_loss(self):
        _, loss, rank = self.losses([], [], [window(0, 0, False), window(0, 0, True)])
        self.assertEqual(loss.item(), 0)
        self.assertEqual(rank.item(), 0)

    def test_zero_weights_are_exact_weighted_pair_objective(self):
        z = torch.tensor([-2., 1., 3.], requires_grad=True)
        labels, weight = torch.tensor([0., 1., 0.]), torch.tensor(2.)
        actual = experiment.training_objective(z, labels, weight, None, 0., 0.)
        expected = nn.BCEWithLogitsLoss(pos_weight=weight)(z, labels)
        self.assertTrue(torch.equal(actual, expected))
        torch.testing.assert_close(torch.autograd.grad(actual, z)[0],
                                   torch.autograd.grad(expected, z)[0], rtol=0, atol=0)

    def test_zero_weight_fit_exactly_reproduces_original_model_and_normalization(self):
        x = np.random.default_rng(3).normal(size=(8, len(FEATURE_NAMES_V4)))
        x[:, -1] = 2.0  # Original constant-feature scale clamp is preserved.
        y = np.array([0., 1., 0., 0., 1., 0., 0., 0.])
        old = fit(x, y, 32, .001, seed=123, epochs=6)
        control = experiment.fit_window(x, y, [window(0, 8, True)], 0., 0., seed=123, epochs=6)
        np.testing.assert_array_equal(old[1], control[1])
        np.testing.assert_array_equal(old[2], control[2])
        for name, value in old[0].state_dict().items():
            torch.testing.assert_close(value, control[0].state_dict()[name], rtol=0, atol=0)
        self.assertEqual(tuple(control[0].hidden.weight.shape), (32, len(FEATURE_NAMES_V4)))

    def test_nonzero_fit_preserves_normalization_and_is_finite(self):
        x = np.random.default_rng(7).normal(size=(9, len(FEATURE_NAMES_V4)))
        y = np.array([0., 0., 0., 1., 0., 0., 0., 0., 0.])
        windows = [window(0, 3, False), window(3, 6, True), window(6, 9, True)]
        model, means, scales = experiment.fit_window(x, y, windows, .5, .25, 123, epochs=5)
        np.testing.assert_array_equal(means, x.mean(axis=0))
        np.testing.assert_array_equal(scales, x.std(axis=0))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_fold_isolation_is_scenario_level_and_window_indices_are_remapped(self):
        scenarios = [s for fold in range(5)
                     for s in [s for s in (f"scenario-{i}" for i in range(100))
                               if fold_of(s, 5) == fold][:2]]
        samples, windows = [], []
        for scenario in scenarios:
            for _ in range(2):
                start = len(samples)
                samples.append({"scenario": scenario, "x": [float(start)] * len(FEATURE_NAMES_V4), "y": 1.})
                windows.append(window(start, start + 1, True, scenario))
        fitted = []

        def fake_fit(x, y, subset, lambda_window, lambda_rank, seed):
            self.assertEqual([w["start"] for w in subset], list(range(len(x))))
            self.assertEqual([w["stop"] for w in subset], list(range(1, len(x) + 1)))
            fitted.append(set(x[:, 0]))
            return fitted[-1], None, None

        def fake_predict(model, means, scales, x):
            self.assertFalse(model & set(x[:, 0]))
            return np.full(len(x), .75)

        with patch.object(experiment, "fit_window", side_effect=fake_fit), patch.object(
                experiment, "predict", side_effect=fake_predict), contextlib.redirect_stdout(io.StringIO()):
            scores, audit = experiment.generate_oof(samples, windows, .25, .1)
        self.assertEqual(len(fitted), 5)
        np.testing.assert_array_equal(scores, np.full(len(samples), .75))
        for row in audit:
            self.assertFalse(set(row["training_scenarios"]) & set(row["held_out_scenarios"]))
            self.assertTrue(all(fold_of(s, 5) == row["fold"] for s in row["held_out_scenarios"]))

    def test_selection_never_accepts_tp_below_116(self):
        infeasible, feasible = config(tp=115, fp=0, pair_recall=1.), config(tp=116, fp=50)
        self.assertIs(experiment.select_configuration([infeasible, feasible]), feasible)
        with self.assertRaisesRegex(ValueError, "TP >= 116"):
            experiment.select_configuration([infeasible])

    def test_selection_minimizes_fp_before_pair_recall(self):
        fewer_fp, more_hits = config(fp=4, pair_recall=.1), config(fp=5, pair_recall=.9)
        self.assertIs(experiment.select_configuration([more_hits, fewer_fp]), fewer_fp)

    def test_selection_ties_use_pair_recall_then_f1_then_lower_loss_weights(self):
        low_pair = config(pair_recall=.4, f1=.9)
        high_pair = config(pair_recall=.5, f1=.3, weights=(1., .5))
        self.assertIs(experiment.select_configuration([low_pair, high_pair]), high_pair)
        high_f1 = config(pair_recall=.5, f1=.4, weights=(1., .5))
        self.assertIs(experiment.select_configuration([high_pair, high_f1]), high_f1)
        simpler = config(pair_recall=.5, f1=.4, weights=(.25, .1))
        self.assertIs(experiment.select_configuration([high_f1, simpler]), simpler)

    def test_bucket2_report_counts_gt_absent_and_score_statistics(self):
        samples = [{"y": 1.}, {"y": 0.}, {"y": 0.}, {"y": 0.}, {"y": 0.}]
        windows = [window(0, 2, True), window(2, 3, True), window(3, 5, False), window(5, 5, False)]
        report = experiment.bucket2_metrics(np.array([.8, .9, .95, .1, .4]), samples, windows, .7)
        self.assertEqual(report["correct_gt_actor_pair_hits"], 1)
        self.assertEqual(report["correct_gt_actor_pair_recall"], .5)
        self.assertEqual(report["gt_pair_candidate_coverage_count"], 1)
        self.assertEqual(report["conditional_correct_pair_recall"], 1)
        self.assertEqual(report["positive_windows_with_gt_pair_absent"], 1)
        self.assertEqual(report["negative_window_score_mean"], .2)
        self.assertEqual(report["negative_window_score_max"], .4)

    def test_paths_protect_frozen_v4_and_reject_validation_before_reading(self):
        with self.assertRaisesRegex(ValueError, "existing V4"):
            experiment.validate_paths([], experiment.BASE / "model_bucket2")
        with self.assertRaisesRegex(ValueError, "validation"):
            experiment.validate_paths(["/never/read/val104/data.jsonl"], experiment.DEFAULT_OUT)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "pair_reranker_v4.json").symlink_to(experiment.BASE / "model_bucket1/pair_reranker_v4.json")
            with self.assertRaisesRegex(ValueError, "existing V4"):
                experiment.validate_paths([], out)

    def test_cli_saves_separate_model_and_never_modifies_bucket1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp), Path(tmp) / "window_model"
            argv = ["--out", str(out), "--train-gt", str(root / "gt.jsonl")]
            paths = []
            for bucket in (1, 2):
                data = root / f"train_bucket{bucket}.jsonl"
                records = []
                for label, positive in (("w1", 1), ("w2", 2), ("w3", None)):
                    score = .9 if bucket == positive else (.6 if label == "w3" and bucket == 2 else .1)
                    pairs = [("a", "b", True, .1), ("x", "y", False, .9)] if bucket == 2 and label == "w2" else [
                        ("a", "b", True, score) if bucket == positive else ("x", "y", False, score)]
                    records.append({"scenario": "s", "window": label, "actual": bucket == positive,
                                    "target_interval": bucket, "positive_intervals": [positive] if positive else [],
                                    "candidates": [{"actor_a": a, "actor_b": b, "label": gt,
                                                    "features": [v] * len(FEATURE_NAMES_V4),
                                                    "minimum_clearance_m": 0., "interval_index": bucket}
                                                   for a, b, gt, v in pairs]})
                data.write_text("".join(json.dumps(d) + "\n" for d in records))
                data.with_suffix(".manifest.json").write_text(json.dumps({"split": "train", "feature_version": 4,
                                                                          "predictor_mode": "joint_scene", "target_interval": bucket}))
                model = experiment.MLP(len(FEATURE_NAMES_V4), 32)
                doc = PairRerankerV4([0.] * len(FEATURE_NAMES_V4), [1.] * len(FEATURE_NAMES_V4),
                                     model.hidden.weight.detach().tolist(), model.hidden.bias.detach().tolist(),
                                     model.output.weight.detach()[0].tolist(), model.output.bias.item(),
                                     .8 if bucket == 1 else .5).to_dict()
                doc["training"] = {"target_interval": bucket, "hidden": 32, "weight_decay": .001, "folds": 5}
                saved = root / f"frozen_model{bucket}.json"
                saved.write_text(json.dumps(doc))
                paths.append(saved)
                argv.extend(["--bucket1-train" if bucket == 1 else "--train", str(data),
                             "--bucket1-model" if bucket == 1 else "--current-bucket2-model", str(saved)])
            (root / "gt.jsonl").write_text("".join(json.dumps({"split": "train", "scenario": "s", "window": label,
                                                                 "gt_actor_groups": [["a"], ["b"]]}) + "\n"
                                                     for label in ("w1", "w2")))
            before = [p.read_bytes() for p in paths]
            control = np.array([.1, .1, .9, .6])
            improved = np.array([.1, .8, .5, .2])
            final = experiment.MLP(len(FEATURE_NAMES_V4), 32)
            with (patch.object(experiment, "TP_FLOOR", 2), patch.object(experiment.calibration, "EXPECTED_COUNTS",
                    {"valid_0_2_windows": 3, "positive_windows": 2, "negative_windows": 1}), patch.object(
                    experiment.calibration, "generate_oof", return_value=(np.array([.9, .1, .1]), [])), patch.object(
                    experiment, "generate_oof", side_effect=[(control, []), (improved, []), (improved, []), (improved, [])]), patch.object(
                    experiment, "fit_window", return_value=(final, np.zeros(len(FEATURE_NAMES_V4)), np.ones(len(FEATURE_NAMES_V4)))) as final_fit,
                    contextlib.redirect_stdout(io.StringIO())):
                experiment.main(argv)
            self.assertEqual([p.read_bytes() for p in paths], before)
            report = json.loads((out / "training_report_v4_window.json").read_text())
            self.assertEqual(report["bucket1_threshold"], .8)
            self.assertEqual(report["current_v4_baseline"]["bucket1_threshold"], .8)
            self.assertEqual(report["selected_oof"]["objective_index"], 1)
            self.assertEqual(final_fit.call_args.args[3:5], (.25, .10))
            saved = PairRerankerV4.load(str(out / "pair_reranker_v4.json"))
            self.assertEqual(saved.threshold, .8)
            self.assertEqual(report["training"]["target_interval"], 2)
            self.assertEqual(len(report["cross_validation"]), 4)
            with np.load(out / "oof_candidate_scores.npz") as arrays:
                np.testing.assert_array_equal(arrays["current_bucket2"], control)
                np.testing.assert_array_equal(arrays["objective_1"], improved)
            rows = [json.loads(line) for line in (out / "oof_windows.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["objectives"]["1"]["combined_correct_pair_hit"] for row in rows[:2]))


if __name__ == "__main__":
    unittest.main()
