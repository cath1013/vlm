"""Censor-aware pair hazard checks."""
import copy
import json
import math
import random
import tempfile
from unittest.mock import patch, Mock
import unittest
from pathlib import Path

import torch

from examples.train_collision_risk_net import (aggregate_window, collate, fit,
                                               identifiable, report,
                                               positive_weight, supervision_counts, eligible_rows,
                                               folds_by_scenario, main, read_rows,
                                               select_threshold, threshold_candidates, binary_metrics, predict)
from traffic_llm.collision_risk_net import (CollisionRiskNet, cumulative_risk,
                                            hazard_supervision,
                                            hazard_to_event_probabilities)
from traffic_llm.predict_model import (N_CANDIDATE_FEATURES, N_GLOBAL_FEATURES,
                                       N_HISTORY_FEATURES, N_HISTORY_STEPS,
                                       N_INTERACTION_FEATURES)
from traffic_llm.predict_nets import JointSceneMotionNetV2


def gt(bucket, end, groups=("A", "B")):
    return {"data_end_s": end, "expected": [
        {"k": k, "interval_end_s": float(k), "accident_expected": k == bucket,
         "involved_vehicles": ([{"actor_ids": [groups[0]]},
                                {"actor_ids": [groups[1]]}] if k == bucket else [])}
        for k in range(1, 6)]}


def inputs(mask):
    b, a = mask.shape
    g = torch.randn(b, a, N_GLOBAL_FEATURES)
    g[..., 0] = 2.0
    g[..., 4] = 0.0
    g[..., 5] = 1.0
    return (g, torch.randn(b, a, 1, N_CANDIDATE_FEATURES),
            mask.unsqueeze(-1).clone(),
            torch.randn(b, a, N_HISTORY_STEPS, N_HISTORY_FEATURES),
            torch.randn(b, a, 1, N_INTERACTION_FEATURES),
            mask.unsqueeze(-1).clone(), mask, torch.randn(b, a, 2))


def actor(name):
    return {"actor_id": name, "global": [0.0] * N_GLOBAL_FEATURES,
            "history": [[0.0] * N_HISTORY_FEATURES for _ in range(N_HISTORY_STEPS)],
            "candidates": [], "interactions": [], "origin_enu": [0.0, 0.0]}


def row(name, bucket, end, groups=("A", "B")):
    labels, targets, masks, observed = hazard_supervision(["A", "B"], gt(bucket, end, groups))
    return {"window_id": name, "scenario_id": name, "dataset_split": "train",
            "gt_bucket": bucket, "gt_positive": bool(bucket),
            "gt_pair_present": bool(labels[0]), "actors": [actor("A"), actor("B")],
            "pair_labels": labels, "hazard_target": targets,
            "hazard_mask": masks, "observed_negative_buckets": observed}


class HazardTests(unittest.TestCase):
    def test_bucket_three_and_post_event(self):
        labels, targets, masks, observed = hazard_supervision(["A", "B"], gt(3, 2.3))
        self.assertEqual(labels, [3])
        self.assertEqual(targets[0], [0, 0, 1, 0, 0])
        self.assertEqual(masks[0], [True, True, True, False, False])
        self.assertEqual(observed, [True, True, False, False, False])

    def test_censored_negative_and_absent_gt_pair(self):
        labels, targets, masks, _ = hazard_supervision(["A", "B"], gt(0, 2.0))
        self.assertEqual(targets[0], [0] * 5)
        self.assertEqual(masks[0], [True, True, False, False, False])
        labels, targets, masks, _ = hazard_supervision(["A", "B"], gt(3, 2.4, ("X", "Y")))
        self.assertEqual(labels, [0])
        self.assertEqual(targets[0], [0] * 5)
        self.assertEqual(masks[0], [True, True, False, False, False])

    def test_non_gt_collision_bucket_requires_full_observation(self):
        labels, targets, masks, _ = hazard_supervision(["A", "B", "C"], gt(3, 2.4))
        self.assertEqual(labels, [3, 0, 0])
        self.assertEqual(masks[1], [True, True, False, False, False])
        self.assertEqual(masks[2], [True, True, False, False, False])
        labels, _, masks, _ = hazard_supervision(["A", "B", "C"], gt(3, 5.0))
        self.assertEqual(masks[1], [True, True, True, False, False])

    def test_stable_identity_groups(self):
        collision = gt(3, 5.0)
        collision["expected"][2]["involved_vehicles"] = [
            {"actor_ids": ["A", "D"]}, {"actor_ids": ["B"]}]
        labels, _, _, _ = hazard_supervision(["A", "B", "C", "D"], collision)
        self.assertEqual(labels, [3, 0, 0, 0, 3, 0])

    def test_pair_symmetry_and_frozen_eval(self):
        torch.manual_seed(7)
        model = CollisionRiskNet(JointSceneMotionNetV2(
            hidden_dim=16, scene_layers=1, interaction_hidden=8), hidden_dim=16)
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        model.eval()
        batch = inputs(torch.ones(1, 2, dtype=torch.bool))
        with torch.no_grad():
            original, mask = model(*batch)
            swapped, _ = model(*(t[:, [1, 0]] for t in batch))
        self.assertEqual(original.shape, (1, 1, 4))
        self.assertTrue(mask.item())
        torch.testing.assert_close(original, swapped, atol=1e-5, rtol=1e-5)
        self.assertEqual(model.head[-1].out_features, 1)

    def test_survival_probability_math(self):
        hazards = torch.tensor([[0.2, 0.3, 0.4, 0.5, 0.6]])
        events, survival = hazard_to_event_probabilities(hazards)
        torch.testing.assert_close(events[0, :3], torch.tensor([.2, .24, .224]))
        torch.testing.assert_close(events.sum(-1) + survival, torch.ones(1))
        torch.testing.assert_close(cumulative_risk(hazards)[0, 1], torch.tensor(.44))

    def test_reporting_supported_horizons_and_coverage(self):
        for k in (2, 3, 4):
            rows = [row("p", 2, 2), row("absent", 2, 2, ("X", "Y")),
                    row("late", 5, 5), row("n", 0, 1), row("excluded", 0, 0)]
            records = [aggregate_window(r, [[.8] * k], k) for r in rows]
            metrics = report(records, rows, {str(h): .5 for h in range(1, k + 1)}, k)
            self.assertEqual(set(metrics["per_horizon"]), {str(h) for h in range(1, k + 1)})
            self.assertEqual(metrics["window_count"], 4)
            for risks in metrics["cumulative_collision_risk_by_window"].values():
                self.assertEqual(set(risks), {str(h) for h in range(1, k + 1)})
                self.assertAlmostEqual(risks["2"], .96)
            self.assertEqual(len(metrics["timing_confusion"]), k)
            self.assertEqual(metrics["gt_positive_windows"], 2)
            at2 = metrics["per_horizon"]["2"]
            self.assertEqual(at2["identifiable_windows"], 3)
            self.assertEqual((at2["positives"], at2["negatives"]), (2, 1))
            self.assertEqual((at2["tp"], at2["fp"], at2["tn"], at2["fn"]), (2, 1, 0, 0))
            self.assertEqual(at2["correct_gt_actor_pair_hits"], 1)
            self.assertEqual(at2["candidate_coverage"], .5)
            self.assertEqual(at2["correct_gt_actor_pair_recall"], .5)
            self.assertEqual(at2["conditional_correct_pair_recall"], 1)
            self.assertFalse(identifiable(rows[3], 2))
            self.assertNotIn("bucket_5_event_count", metrics)

    def test_shapes_and_fixed_absolute_time_encoding(self):
        batch = inputs(torch.ones(1, 2, dtype=torch.bool))
        models = []
        for k in (2, 3, 4):
            model = CollisionRiskNet(JointSceneMotionNetV2(
                hidden_dim=16, scene_layers=1, interaction_hidden=8),
                hidden_dim=16, dropout=0, max_horizon=k).eval()
            models.append(model)
            captured = []
            hook = model.head.register_forward_pre_hook(
                lambda module, args: captured.append(args[0][..., -1].clone()))
            with torch.no_grad():
                logits, _ = model(*batch)
                empty, _ = model(*inputs(torch.ones(1, 1, dtype=torch.bool)))
            hook.remove()
            self.assertEqual(logits.shape, (1, 1, k))
            self.assertEqual(empty.shape, (1, 0, k))
            torch.testing.assert_close(captured[0][0, 0], torch.tensor([.25, .5, .75, 1.][:k]))
            self.assertEqual(model.spec["max_horizon"], k)
            self.assertEqual(model.head[-1].out_features, 1)
        # Identical parameters imply identical hazard logits for shared seconds.
        for model in models[1:]:
            model.load_state_dict(models[0].state_dict())
            with torch.no_grad():
                torch.testing.assert_close(model(*batch)[0][..., :2], models[0](*batch)[0])

    def test_prefix_supervision_exclusion_and_no_tail_use(self):
        rows = [row("event5", 5, 5), row("censored", 0, 1), row("none", 0, 0)]
        original = copy.deepcopy(rows)
        for k in (2, 3, 4):
            self.assertEqual([r["window_id"] for r in eligible_rows(rows, k)], ["event5", "censored"])
            batch = collate(eligible_rows(rows, k), k)
            self.assertEqual(batch["hazard_target"].shape, (2, 1, k))
            self.assertEqual(batch["hazard_mask"].shape, (2, 1, k))
            for i, r in enumerate(rows[:2]):
                self.assertEqual(batch["hazard_target"][i, 0].tolist(), r["hazard_target"][0][:k])
                self.assertEqual(batch["hazard_mask"][i, 0].tolist(), r["hazard_mask"][0][:k])
            self.assertFalse(batch["hazard_mask"][1, 0, 1:].any())
            changed = copy.deepcopy(rows)
            for r in changed:
                r["hazard_target"][0][k:] = [1] * (5-k)
                r["hazard_mask"][0][k:] = [True] * (5-k)
            self.assertEqual(supervision_counts(rows, k), supervision_counts(changed, k))
            self.assertEqual(positive_weight(rows, k), positive_weight(changed, k))
            other = collate(changed[:2], k)
            torch.testing.assert_close(batch["hazard_target"], other["hazard_target"])
            torch.testing.assert_close(batch["hazard_mask"], other["hazard_mask"])
        self.assertEqual(rows, original)
        for first in (3, 4, 5):
            late_only = row("late_only", 0, 5)
            late_only["hazard_mask"] = [[i >= first for i in range(1, 6)]]
            for k in (2, 3, 4):
                self.assertEqual(bool(eligible_rows([late_only], k)), first <= k)


    def test_cli_same_folds_across_horizons_and_checkpoint_metadata(self):
        rows = [row(str(i), 2, 5) for i in range(10)]
        rows.append(row("excluded_scenario", 0, 0))
        expected = folds_by_scenario(rows, 7)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            source.write_text("\n".join(json.dumps(r) for r in rows))
            self.assertEqual(read_rows(source, "train"), rows)
            for k in (2, 3, 4):
                out = Path(directory) / f"model_h{k}"
                model = Mock(max_horizon=k)
                model.spec = {"max_horizon": k}
                model.state_dict.return_value = {}
                def predictions(model, held, device, batch_size):
                    return [aggregate_window(r, [[.4] * k], k) for r in held]
                with patch("examples.train_collision_risk_net.verify_manifest"), \
                     patch("examples.train_collision_risk_net.fit", return_value=model) as fitting, \
                     patch("examples.train_collision_risk_net.predict", side_effect=predictions), \
                     patch("builtins.print"):
                    main(["--train", str(source), "--v2-checkpoint", "v2.pt",
                          "--out", str(out), "--seed", "7", "--max-horizon", str(k),
                          "--lr-grid", ".001"])
                result = json.loads((out / "report.json").read_text())
                self.assertEqual(result["fold_assignments"], expected)
                self.assertEqual(set(result["cv_train"]["per_horizon"]), {str(h) for h in range(1,k+1)})
                saved = torch.load(out / "collision_risk_final.pt", weights_only=False)
                self.assertEqual(saved["max_horizon"], k)
                self.assertEqual(saved["threshold_by_horizon"], result["threshold_by_horizon"])
                self.assertEqual(saved["threshold_by_horizon"], result["cv_train"]["threshold_by_horizon"])
                self.assertEqual(saved["train_scenario_ids"], sorted(expected))
                for call in fitting.call_args_list:
                    self.assertEqual(call.kwargs["max_horizon"], k)
                    self.assertNotIn("excluded_scenario", [r["scenario_id"] for r in call.args[0]])

    def test_oof_score_candidates_and_independent_horizons(self):
        rows = [row("p1", 1, 5), row("p2", 2, 5), row("n", 0, 5),
                row("censored", 0, 1)]
        scores = [[.213, .431, .613, .813], [.117, .389, .587, .789],
                  [.181, .277, .511, .711], [.101, .999, .999, .999]]
        records = [aggregate_window(r, [[.1] * 4], 4) for r in rows]
        for record, risks in zip(records, scores):
            record["horizon_risks"] = risks
        candidates = threshold_candidates([.213, .181, .213, 1.0])
        self.assertEqual(candidates, [0.0, .181, .213, 1.0, math.nextafter(1.0, math.inf)])
        thresholds = select_threshold(records, rows, 4)
        self.assertEqual(thresholds, {"1": .213, "2": .389, "3": .587, "4": .789})
        changed = copy.deepcopy(records)
        for i, record in enumerate(changed):
            record["horizon_risks"][2:] = [.9 + i * .01, .95 + i * .01]
        self.assertEqual(select_threshold(changed, rows, 4)["2"], thresholds["2"])
        # CRN-2/3/4 use only the identifiable 2s scores for their 2s threshold.
        for k in (2, 3):
            shorter = copy.deepcopy(records)
            for record in shorter:
                record["max_horizon"] = k
                record["horizon_risks"] = record["horizon_risks"][:k]
            self.assertEqual(select_threshold(shorter, rows, k)["2"], thresholds["2"])

    def test_threshold_sweep_matches_balanced_f1_precision_higher_tie_break(self):
        rng = random.Random(19)
        rows = [row(str(i), 1 if i % 2 else 0, 5) for i in range(12)]
        for _ in range(30):
            records = [aggregate_window(r, [[.1] * 2], 2) for r in rows]
            for record in records:
                record["horizon_risks"] = [rng.choice([.123, .287, .431, .619]) for _ in range(2)]
            thresholds = select_threshold(records, rows, 2)
            for h in (1, 2):
                candidates = threshold_candidates([r["horizon_risks"][h - 1] for r in records])
                def key(t):
                    metrics = binary_metrics([bool(r["gt_bucket"]) for r in rows],
                                             [r["horizon_risks"][h - 1] >= t for r in records])
                    return tuple(metrics[name] if metrics[name] is not None else -1
                                 for name in ("balanced_accuracy", "f1", "precision")) + (t,)
                self.assertEqual(thresholds[str(h)], max(candidates, key=key))
        # Equal classification metrics at thresholds 0 and .123 prefer .123.
        for record in records:
            record["horizon_risks"] = [.123, .123]
        self.assertEqual(select_threshold(records, rows, 2), {"1": .123, "2": .123})
        for record in records:
            record["horizon_risks"] = [1.0, 1.0]
        negatives = [row(str(i), 0, 5) for i in range(12)]
        for record in records:
            record["gt_bucket"] = 0
        # With no positives, BA is undefined; defined F1/precision then decide.
        self.assertEqual(select_threshold(records, negatives, 2), {"1": 1.0, "2": 1.0})
        self.assertEqual(select_threshold([], [], 2),
                         {"1": math.nextafter(1.0, math.inf), "2": math.nextafter(1.0, math.inf)})
        tied_rows = [row("p1", 1, 5), row("n1", 0, 5),
                     row("p2", 1, 5), row("n2", 0, 5)]
        tied_records = [aggregate_window(r, [[.1] * 2], 2) for r in tied_rows]
        for record, score in zip(tied_records, [.9, .8, .7, .6]):
            record["horizon_risks"] = [score, score]
        # .9 and .7 tie on BA; .7 has the higher F1 despite lower precision.
        self.assertEqual(select_threshold(tied_records, tied_rows, 2), {"1": .7, "2": .7})

    def test_checkpoint_evaluation_uses_frozen_horizon_thresholds(self):
        rows = [row("p1", 1, 5), row("p2", 2, 5), row("n", 0, 5)]
        for r in rows:
            r["dataset_split"] = "val"
        records = [aggregate_window(r, [[.1] * 2], 2) for r in rows]
        for r in records:
            r["horizon_risks"] = [.4, .6]
        frozen = {"1": .3, "2": .7}
        saved = {"max_horizon": 2, "v2_checkpoint": "unused.pt", "hidden_dim": 16,
                 "dropout": 0, "state_dict": {}, "threshold_by_horizon": frozen,
                 "train_scenario_ids": ["train_only"]}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(saved, checkpoint)
            out = Path(directory) / "evaluation"
            with patch("examples.train_collision_risk_net.CollisionRiskNet"), \
                 patch("examples.train_collision_risk_net.load_v2"), \
                 patch("examples.train_collision_risk_net.verify_manifest"), \
                 patch("examples.train_collision_risk_net.read_rows", return_value=rows), \
                 patch("examples.train_collision_risk_net.predict", return_value=records), \
                 patch("examples.train_collision_risk_net.select_threshold") as selecting, \
                 patch("builtins.print"):
                main(["--evaluate-checkpoint", str(checkpoint), "--val", "synthetic.jsonl",
                      "--out", str(out), "--max-horizon", "2"])
            selecting.assert_not_called()
            metrics = json.loads((out / "val104_report.json").read_text())
        self.assertEqual(metrics["threshold_by_horizon"], frozen)
        self.assertEqual(metrics["per_horizon"]["1"]["threshold"], .3)
        self.assertEqual(metrics["per_horizon"]["2"]["threshold"], .7)
        self.assertEqual((metrics["per_horizon"]["1"]["tp"], metrics["per_horizon"]["1"]["fp"]), (1, 2))
        self.assertEqual((metrics["per_horizon"]["2"]["fn"], metrics["per_horizon"]["2"]["tn"]), (2, 1))

    def test_fit_cli_rejects_val_before_reading(self):
        with patch("examples.train_collision_risk_net.read_rows") as reading, \
             patch("examples.train_collision_risk_net.verify_manifest") as manifests, \
             patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                main(["--train", "train483.jsonl", "--val", "val104.jsonl",
                      "--v2-checkpoint", "v2.pt", "--out", "unused", "--max-horizon", "2"])
        reading.assert_not_called()
        manifests.assert_not_called()

    def test_small_smoke_fit_and_counts(self):
        rows = [row("p", 3, 2.4), row("n", 0, 2.0)]
        self.assertEqual(supervision_counts(rows), {
            "windows_used": 2,
            "supervised_hazard_entries_per_bucket": [2, 2, 1, 0],
            "positive_hazard_entries_per_bucket": [0, 0, 1, 0],
            "negative_hazard_entries_per_bucket": [2, 2, 0, 0],
            "censored_hazard_entries_per_bucket": [0, 0, 1, 2]})
        self.assertEqual(collate(rows)["hazard_mask"].shape, (2, 1, 4))
        self.assertEqual(positive_weight(rows), 4.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "v2.pt"
            torch.save(JointSceneMotionNetV2(
                hidden_dim=16, scene_layers=1, interaction_hidden=8), checkpoint)
            for k in (2, 3, 4):
                model = fit(rows + [row("excluded", 0, 0)], checkpoint, "cpu",
                            epochs=1, batch_size=2, lr=.001, hidden_dim=16,
                            dropout=0, seed=0, max_horizon=k)
                self.assertFalse(model.backbone.training)
                self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
                records = predict(model, rows + [row("excluded", 0, 0)], "cpu", 2)
                self.assertEqual(len(records), 2)
                self.assertTrue(all(len(r["horizon_risks"]) == k for r in records))
                thresholds = select_threshold(records, rows, k)
                self.assertEqual(set(thresholds), {str(h) for h in range(1, k + 1)})
                for h in range(1, k + 1):
                    scores = [r["horizon_risks"][h - 1] for r, source in zip(records, rows)
                              if identifiable(source, h)]
                    self.assertIn(thresholds[str(h)], threshold_candidates(scores))


if __name__ == "__main__":
    unittest.main()
