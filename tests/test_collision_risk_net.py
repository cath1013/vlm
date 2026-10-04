"""Small structural checks for the observation-only pair experiment."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from examples.train_collision_risk_net import (aggregate_window, decide_window,
                                               report, select_threshold,
                                               verify_manifest)
from examples.collect_collision_risk_dataset import (add_scenario_census,
                                                     add_scenario_diagnostics,
                                                     add_window_to_census,
                                                     collect_scenario,
                                                     empty_horizon_census,
                                                     finalize_horizon_census,
                                                     horizon_supervision_status,
                                                     main as collection_main,
                                                     window_supervision_status)
from traffic_llm.collision_risk_net import (CollisionRiskNet, ground_truth_bucket,
                                            pair_labels, valid_pairs)
from traffic_llm.predict_model import (N_CANDIDATE_FEATURES, N_GLOBAL_FEATURES,
                                       N_HISTORY_FEATURES, N_HISTORY_STEPS,
                                       N_INTERACTION_FEATURES)
from traffic_llm.predict_nets import JointSceneMotionNetV2


def inputs(mask):
    B, A = mask.shape
    g = torch.randn(B, A, N_GLOBAL_FEATURES)
    g[..., 0] = 2.0
    g[..., 4] = 0.0
    g[..., 5] = 1.0
    c = torch.randn(B, A, 1, N_CANDIDATE_FEATURES)
    cm = mask.unsqueeze(-1).clone()
    h = torch.randn(B, A, N_HISTORY_STEPS, N_HISTORY_FEATURES)
    x = torch.randn(B, A, 1, N_INTERACTION_FEATURES)
    xm = mask.unsqueeze(-1).clone()
    origin = torch.randn(B, A, 2)
    return g, c, cm, h, x, xm, mask, origin


class CollisionRiskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = CollisionRiskNet(JointSceneMotionNetV2(
            hidden_dim=16, scene_layers=1, interaction_hidden=8), hidden_dim=16).eval()

    def test_actor_order_symmetry(self):
        batch = inputs(torch.ones(1, 2, dtype=torch.bool))
        with torch.no_grad():
            original, _ = self.model(*batch)
            permuted = tuple(t[:, [1, 0]] for t in batch)
            swapped, _ = self.model(*permuted)
        torch.testing.assert_close(original, swapped, atol=1e-5, rtol=1e-5)
        self.assertTrue(all(not p.requires_grad for p in self.model.backbone.parameters()))

    def test_padded_pairs_are_masked(self):
        batch = inputs(torch.tensor([[True, True, False], [True, False, False]]))
        with torch.no_grad():
            logits, mask = self.model(*batch)
        self.assertEqual(mask.tolist(), [[True, False, False], [False, False, False]])
        self.assertEqual(logits.shape, (2, 3, 6))
        self.assertTrue(torch.equal(logits[~mask], torch.zeros_like(logits[~mask])))
        i, j, direct_mask = valid_pairs(batch[6])
        self.assertEqual((i.tolist(), j.tolist()), ([0, 0, 1], [1, 2, 2]))
        self.assertTrue(torch.equal(mask, direct_mask))

    def test_bucket_labels_use_distinct_identity_groups(self):
        actors = ["A", "B", "C", "D"]
        expected = [{"k": k, "accident_expected": k == 3,
                     "involved_vehicles": ([{"actor_ids": ["A", "D"]},
                                             {"actor_ids": ["B"]}] if k == 3 else [])}
                    for k in range(1, 6)]
        labels = pair_labels(actors, {"expected": expected})
        # AB, AC, AD, BC, BD, CD. A and D are the same physical vehicle.
        self.assertEqual(labels, [3, 0, 0, 0, 3, 0])
        self.assertEqual(pair_labels(actors, {"expected": [
            {"k": k, "accident_expected": False} for k in range(1, 6)]}), [0] * 6)

    def test_gt_bucket_survives_missing_collision_pair(self):
        gt = {"expected": [{"k": k, "accident_expected": k == 4,
                            "involved_vehicles": ([{"actor_ids": ["missing_A"]},
                                                    {"actor_ids": ["missing_B"]}]
                                                   if k == 4 else [])}
                           for k in range(1, 6)]}
        self.assertEqual(ground_truth_bucket(gt), 4)
        self.assertEqual(pair_labels(["A", "B"], gt), [0])

    def test_window_aggregation_uses_max_pair_score_and_selected_bucket(self):
        row = {"window_id": "s:0", "scenario_id": "s", "gt_bucket": 2,
               "actors": [{"actor_id": x} for x in ("A", "B", "C")],
               "pair_labels": [0, 2, 0], "gt_pair_present": True}
        result = aggregate_window(row, [
            [.15, .05, .70, .04, .03, .03],  # AB: score .85, bucket 2
            [.10, .05, .05, .70, .05, .05],  # AC: score .90, bucket 3
            [.70, .10, .05, .05, .05, .05],
        ])
        self.assertAlmostEqual(result["score"], .90)
        self.assertEqual(result["selected_pair"], ("A", "C"))
        self.assertEqual(decide_window(result, .89), 3)
        self.assertEqual(decide_window(result, .91), 0)

    def test_threshold_uses_windows_and_report_counts_missing_pair(self):
        def window(name, bucket, score):
            row = {"window_id": name, "scenario_id": name, "gt_bucket": bucket,
                   "actors": [{"actor_id": "A"}, {"actor_id": "B"}],
                   "pair_labels": [bucket if bucket else 0],
                   "gt_pair_present": bool(bucket)}
            return row, aggregate_window(row, [[1 - score, score, 0, 0, 0, 0]])
        data = [window("p1", 1, .8), window("p2", 1, .75),
                window("n1", 0, .6), window("n2", 0, .1)]
        rows, records = zip(*data)
        self.assertEqual(select_threshold(records), .65)
        metrics = report(records, rows, .65)
        self.assertEqual(metrics["unit"], "window")
        self.assertEqual(metrics["overall_0_5_s"]["confusion_matrix"], [[2, 0], [0, 2]])
        # Adding more pairs to one negative window must not give it extra
        # votes during OOF threshold selection.
        crowded = dict(rows[2], actors=[{"actor_id": x} for x in ("A", "B", "C")],
                       pair_labels=[0, 0, 0])
        crowded_prediction = aggregate_window(crowded, [[.4, .6, 0, 0, 0, 0]] * 3)
        self.assertEqual(select_threshold([records[0], records[1],
                                           crowded_prediction, records[3]]), .65)
        absent = {"window_id": "absent", "scenario_id": "s", "gt_bucket": 3,
                  "actors": [{"actor_id": "A"}], "pair_labels": [],
                  "gt_pair_present": False}
        absent_prediction = aggregate_window(absent, [])
        self.assertEqual(absent_prediction["score"], 0)
        missing_metrics = report([absent_prediction], [absent], .65)
        self.assertEqual(missing_metrics["overall_0_5_s"]["confusion_matrix"], [[0, 0], [1, 0]])
        self.assertEqual(missing_metrics["horizon_confusion_6x6"][3][0], 1)
        self.assertEqual(missing_metrics["visible_pair_coverage_ceiling"], 0)

    def test_full_train_requires_official_train483_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            manifest_path = path.with_suffix(".manifest.json")
            manifest = {"source_split": "train", "observation_mode": "sensor3d",
                        "window_s": 5, "stride_s": 1, "horizon_s": 5,
                        "limit": 0, "available_scenarios": 483,
                        "selected_scenarios": 483, "counts": {"failed": []}}
            def check():
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                return verify_manifest(path, "train", smoke=False)
            self.assertEqual(check()["selected_scenarios"], 483)
            for field in ("available_scenarios", "selected_scenarios"):
                manifest[field] = 482
                with self.assertRaisesRegex(ValueError, "official train483"):
                    check()
                manifest[field] = 483
            manifest["limit"] = 1
            with self.assertRaisesRegex(ValueError, "no limit or failed scenarios"):
                check()
            manifest["limit"] = 0
            manifest["counts"]["failed"] = ["s"]
            with self.assertRaisesRegex(ValueError, "no limit or failed scenarios"):
                check()

    def test_collection_diagnostics_keep_existing_supervision_rule(self):
        def gt(bucket, scorable):
            return {"expected": [{"k": k, "accident_expected": k == bucket,
                                  "scorable": value}
                                 for k, value in enumerate(scorable, 1)]}
        incomplete = gt(0, [True, True, False, False, False])
        known_collision = gt(3, [True, True, True, False, False])
        complete_negative = gt(0, [True] * 5)
        self.assertEqual(window_supervision_status(incomplete), (0, False, True))
        self.assertEqual(window_supervision_status(known_collision), (3, True, False))
        self.assertEqual(window_supervision_status(complete_negative), (0, True, False))

        snapshots = [SimpleNamespace(t=0.0), SimpleNamespace(t=10.0)]
        windows = [SimpleNamespace(index=i, t_end=float(i + 5),
                                   last=SimpleNamespace(actors=[])) for i in range(2)]
        result = SimpleNamespace(snapshots=lambda rate_hz: iter(snapshots), network=None)
        runner = SimpleNamespace(build=lambda *a, **kw: result)
        scenario = SimpleNamespace(scenario="s", scenario_type="t", town="town",
                                   scenario_id="sid", split="train", agents=[], meta=None)
        cfg = SimpleNamespace(deepaccident=None)
        with patch("examples.collect_collision_risk_dataset.find_xodr", return_value=None), \
             patch("examples.collect_collision_risk_dataset.estimate_collision", return_value=None), \
             patch("examples.collect_collision_risk_dataset.build_windows",
                   return_value=(windows, {"dropped_incomplete": 2,
                                           "dropped_after_collision": 1})), \
             patch("examples.collect_collision_risk_dataset.window_ground_truth",
                   side_effect=[incomplete, complete_negative]):
            rows, diagnostics = collect_scenario(runner, scenario, cfg, None, "maps", 2.0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(diagnostics, {
            "scenario_id": "sid", "windows_from_build_windows": 2,
            "windows_kept": 1, "fully_supervised_5s_windows": 1,
            "dropped_incomplete_observation": 2, "dropped_after_collision": 1,
            "dropped_incomplete_future_5s": 1,
            "collision_time_s": None, "data_start_s": 0.0, "data_end_s": 10.0,
            "horizon_census": finalize_horizon_census(
                self._census_for(incomplete, complete_negative))})

    @staticmethod
    def _census_for(*gt_rows):
        census = empty_horizon_census()
        for gt in gt_rows:
            add_window_to_census(census, gt)
        return census

    def test_horizon_keep_drop_rules_and_scenario_counts(self):
        def gt(bucket, scorable):
            return {"expected": [{"k": k, "accident_expected": k == bucket,
                                  "scorable": value}
                                 for k, value in enumerate(scorable, 1)]}
        negative_partial = gt(0, [True, True, False, False, False])
        positive_late = gt(4, [True, True, True, True, False])
        self.assertEqual([horizon_supervision_status(negative_partial, h)
                          for h in range(1, 6)], [
            (True, False, True, False), (True, False, True, False),
            (False, False, False, True), (False, False, False, True),
            (False, False, False, True)])
        self.assertEqual([horizon_supervision_status(positive_late, h)
                          for h in range(1, 6)], [
            (True, False, True, False), (True, False, True, False),
            (True, False, True, False), (True, True, False, False),
            (True, True, False, False)])
        per_scenario = finalize_horizon_census(
            self._census_for(negative_partial, positive_late))
        self.assertEqual(per_scenario["3"]["kept_windows"], 1)
        self.assertEqual(per_scenario["3"]["dropped_incomplete_future"], 1)
        self.assertEqual(per_scenario["3"]["retention_fraction"], .5)
        self.assertEqual(per_scenario["5"]["positive_windows"], 1)
        total = empty_horizon_census()
        add_scenario_census(total, per_scenario)
        add_scenario_census(total, finalize_horizon_census(self._census_for(negative_partial)))
        finalize_horizon_census(total)
        self.assertEqual(total["5"]["total_build_windows"], 3)
        self.assertEqual(total["5"]["kept_windows"], 1)
        self.assertEqual(total["5"]["scenarios_with_at_least_one_kept_window"], 1)
        self.assertEqual(total["5"]["scenarios_with_at_least_one_positive_window"], 1)
        self.assertEqual(total["2"]["scenarios_with_at_least_one_negative_window"], 2)

    def test_census_only_skips_actor_encoding(self):
        snapshots = [SimpleNamespace(t=0.0), SimpleNamespace(t=10.0)]
        window = SimpleNamespace(index=0, t_end=5.0,
                                 last=SimpleNamespace(actors=[SimpleNamespace(actor_id="A")]))
        result = SimpleNamespace(snapshots=lambda rate_hz: iter(snapshots), network=None)
        runner = SimpleNamespace(build=lambda *a, **kw: result)
        scenario = SimpleNamespace(scenario="s", scenario_type="t", town="town",
                                   scenario_id="sid", split="train", agents=[], meta=None)
        gt = {"expected": [{"k": k, "accident_expected": False, "scorable": True}
                           for k in range(1, 6)]}
        with patch("examples.collect_collision_risk_dataset.find_xodr", return_value=None), \
             patch("examples.collect_collision_risk_dataset.estimate_collision", return_value=None), \
             patch("examples.collect_collision_risk_dataset.build_windows",
                   return_value=([window], {"dropped_incomplete": 0,
                                            "dropped_after_collision": 0})), \
             patch("examples.collect_collision_risk_dataset.window_ground_truth", return_value=gt), \
             patch("examples.collect_collision_risk_dataset.build_context",
                   side_effect=AssertionError("context built")), \
             patch("examples.collect_collision_risk_dataset.encode",
                   side_effect=AssertionError("encoded")):
            rows, diagnostics = collect_scenario(
                runner, scenario, SimpleNamespace(deepaccident=None), None,
                "maps", 2.0, census_only=True)
        self.assertEqual(rows, [])
        self.assertEqual(diagnostics["windows_kept"], 1)
        self.assertEqual(diagnostics["horizon_census"]["5"]["negative_windows"], 1)

    def test_census_only_writes_json_without_pair_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "census.json"
            runner = SimpleNamespace(list_scenarios=lambda: [])
            with patch("examples.collect_collision_risk_dataset.DeepAccidentRunner",
                       return_value=runner):
                collection_main(["--root", directory, "--carla-maps", directory,
                                 "--split", "train", "--out", str(out), "--census-only"])
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(report["census_only"])
            self.assertEqual(set(report["horizon_census"]), {"1", "2", "3", "4", "5"})
            self.assertFalse(out.with_suffix(".manifest.json").exists())

    def test_aggregate_collection_diagnostics(self):
        counts = {key: 0 for key in (
            "total_build_windows", "total_kept_windows", "dropped_incomplete_future_5s",
            "dropped_incomplete_observation", "dropped_after_collision",
            "scenarios_with_zero_build_windows",
            "scenarios_with_build_windows_but_zero_fully_supervised_5s_windows",
            "scenarios_with_kept_windows")}
        for built, kept, fully, dropped, observation, collision in (
            (0, 0, 0, 0, 1, 0), (2, 1, 0, 1, 2, 1), (3, 2, 2, 1, 0, 2)):
            add_scenario_diagnostics(counts, {
                "windows_from_build_windows": built, "windows_kept": kept,
                "fully_supervised_5s_windows": fully,
                "dropped_incomplete_observation": observation,
                "dropped_after_collision": collision,
                "dropped_incomplete_future_5s": dropped})
        self.assertEqual(counts, {
            "total_build_windows": 5, "total_kept_windows": 3,
            "dropped_incomplete_observation": 3, "dropped_after_collision": 3,
            "dropped_incomplete_future_5s": 2,
            "scenarios_with_zero_build_windows": 1,
            "scenarios_with_build_windows_but_zero_fully_supervised_5s_windows": 1,
            "scenarios_with_kept_windows": 2})


if __name__ == "__main__":
    unittest.main()
