"""Deterministic pilot selection and preflight tests."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from dataclasses import dataclass


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "run_pilot.py",
)
SPEC = importlib.util.spec_from_file_location("run_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pilot)


@dataclass(frozen=True)
class FakeScenario:
    scenario_id: str
    split: str
    outcome: str
    town: str
    scenario_type: str
    scenario: str


class TestPilotSelection(unittest.TestCase):
    def _scenarios(self):
        rows = []
        for outcome in ("accident", "normal"):
            for i in range(12):
                rows.append(
                    FakeScenario(
                        scenario_id=f"type{i % 3}_{outcome}/Town0{i % 4}_{i}",
                        split="val",
                        outcome=outcome,
                        town=f"Town0{i % 4}",
                        scenario_type=f"type{i % 3}_{outcome}",
                        scenario=f"Town0{i % 4}_{i}",
                    )
                )
        rows.append(
            FakeScenario("train/x", "train", "accident", "Town99", "train", "x")
        )
        return rows

    def test_selection_is_balanced_validation_only_and_deterministic(self):
        rows = self._scenarios()
        a = pilot.select_balanced(rows, seed=7)
        b = pilot.select_balanced(reversed(rows), seed=7)
        self.assertEqual([s.scenario_id for s in a], [s.scenario_id for s in b])
        self.assertEqual(sum(s.outcome == "accident" for s in a), 5)
        self.assertEqual(sum(s.outcome == "normal" for s in a), 5)
        self.assertTrue(all(s.split == "val" for s in a))
        self.assertGreaterEqual(len({s.town for s in a}), 3)
        self.assertGreaterEqual(len({s.scenario_type for s in a}), 4)

    def test_selection_rejects_an_undersized_outcome(self):
        rows = [s for s in self._scenarios() if s.outcome == "normal"]
        with self.assertRaisesRegex(ValueError, "accident"):
            pilot.select_balanced(rows)


class TestPilotPreflight(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        scenarios = []
        for outcome in ("accident", "normal"):
            for i in range(5):
                scenarios.append(
                    {
                        "ref": f"{outcome}_{i}",
                        "id": f"type{i % 2}_{outcome}/Town0{i % 3}_{i}",
                        "scenario": f"Town0{i % 3}_{i}",
                        "scenario_type": f"type{i % 2}_{outcome}",
                        "dataset_split": "val",
                        "outcome": outcome,
                        "town": f"Town0{i % 3}",
                    }
                )
        self.selection = {
            "generation": {
                "map_built_from_scenario_trajectories": False,
                "llm_calls_made": 0,
                "conditions": {
                    "waypointnet": {
                        "mode": "waypoints",
                        "fallback": "constant_velocity",
                        "model_sha256": "test-hash",
                    }
                },
            },
            "scenarios": scenarios,
        }
        for condition in pilot.CONDITIONS:
            for record in scenarios:
                self._write_case(condition, record)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_json(self, path, value):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(value, handle)

    def _write_case(self, condition, record):
        directory = pilot._scenario_dir(self.root, condition, record)
        collision = {
            "occurred": record["outcome"] == "accident",
            "time_s": 8.0 if record["outcome"] == "accident" else None,
        }
        ground_truth = {
            "window": {"label": "0-5", "t_start_s": 0.0, "t_end_s": 5.0},
            "scenario": record,
            "collision": collision,
            "expected": [{"k": k, "accident_expected": False} for k in range(1, 6)],
        }
        self._write_json(os.path.join(directory, "ground_truth_0-5.json"), ground_truth)
        self._write_json(
            os.path.join(directory, "llm_payload_0-5.json"),
            {"condition": condition},
        )
        manifest = {
            "scenario": {
                "ref": record["ref"],
                "id": record["id"],
                "split": record["scenario_type"],
                "dataset_split": "val",
                "outcome": record["outcome"],
                "town": record["town"],
            },
            "config": {
                "predictor": condition,
                "map_source": f"OpenDRIVE ({record['town']}.xodr)",
                "map_built_from_scenario_trajectories": False,
                "window_s": 5.0,
                "stride_s": 1.0,
                "horizon_s": 5.0,
                "snapshot_rate_hz": 2.0,
            },
            "collision": collision,
            "windows": [
                {
                    "label": "0-5",
                    "index": 0,
                    "t_start_s": 0.0,
                    "t_end_s": 5.0,
                    "ground_truth": "ground_truth_0-5.json",
                    "payload": "llm_payload_0-5.json",
                    "has_accident_in_horizon": False,
                    "n_snapshots": 11,
                }
            ],
        }
        self._write_json(os.path.join(directory, "manifest.json"), manifest)

    def test_complete_paired_fixture_passes(self):
        report = pilot.preflight_pilot(self.root, self.selection)
        self.assertEqual(report["status"], "pass", report["errors"])
        self.assertEqual(report["summary"]["paired_scenarios"], 10)
        self.assertEqual(report["llm_calls_made"], 0)

    def test_window_at_collision_is_rejected(self):
        record = self.selection["scenarios"][0]
        directory = pilot._scenario_dir(self.root, "constant_velocity", record)
        path = os.path.join(directory, "manifest.json")
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["windows"][0]["t_end_s"] = 8.0
        self._write_json(path, manifest)
        report = pilot.preflight_pilot(self.root, self.selection)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("at or after collision" in e for e in report["errors"]))


if __name__ == "__main__":
    unittest.main()
