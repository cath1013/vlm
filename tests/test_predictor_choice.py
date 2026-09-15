# -*- coding: utf-8 -*-
"""`run_deepaccident.py` 의 예측기 선택 — 기본값·폴백·모드 판정.

여기서 막으려는 것은 **조용히 다른 예측기를 쓰는 것**이다. payload 의 `예상경로`
절이 예측기마다 완전히 달라지므로(규칙: "좌회전 50%, 우회전 50%" / waypoints: 단일
경로), 어느 것을 썼는지 모르면 결과를 해석할 수 없다.
"""

import argparse
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

import run_deepaccident as rd  # noqa: E402


def ns(**kw):
    base = dict(predictor=None, no_predictor=False, predictor_mode=None,
                predictor_device="cpu", windows_dir=None, eval_out=None,
                payload_out=None, road_map=None, allow_trajectory_map=False)
    base.update(kw)
    return argparse.Namespace(**base)


class TestResolvePredictor(unittest.TestCase):
    def test_default_is_the_waypoint_model(self):
        """기본 예측기는 좌표 회귀 모델이다."""
        self.assertTrue(rd.DEFAULT_PREDICTOR.endswith("waypointnet_best.pt"))

    def test_no_predictor_forces_the_rule(self):
        path, mode, why = rd.resolve_predictor(ns(no_predictor=True))
        self.assertIsNone(path)
        self.assertIn("--no-predictor", why)

    def test_missing_default_falls_back_and_says_why(self):
        """기본 모델이 없으면 규칙 기반으로 내려가되 이유를 남긴다."""
        saved = rd.DEFAULT_PREDICTOR
        rd.DEFAULT_PREDICTOR = os.path.join(tempfile.gettempdir(), "no_such.pt")
        try:
            path, mode, why = rd.resolve_predictor(ns())
            self.assertIsNone(path)
            self.assertIn("기본 모델 없음", why)
        finally:
            rd.DEFAULT_PREDICTOR = saved

    def test_explicit_missing_path_is_an_error_not_a_fallback(self):
        """직접 지정한 파일이 없으면 조용히 규칙 기반으로 바꾸면 안 된다."""
        with self.assertRaises(SystemExit):
            rd.resolve_predictor(ns(predictor="/nonexistent/model.pt"))

    def test_explicit_mode_overrides_inference(self):
        p = os.path.join(ROOT, "out/predict_model/waypointnet_best.pt")
        if not os.path.isfile(p):
            self.skipTest("학습된 모델 없음")
        _, mode, _ = rd.resolve_predictor(ns(predictor=p,
                                             predictor_mode="rank"))
        self.assertEqual(mode, "rank")


class TestInferPredictorMode(unittest.TestCase):
    """모델과 모드를 손으로 짝지으면 어긋난다 — 파일에서 판정해야 한다."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch 없음")

    def _p(self, name):
        p = os.path.join(ROOT, "out/predict_model", name)
        if not os.path.isfile(p):
            self.skipTest(f"{name} 없음")
        return p

    def test_waypointnet_is_waypoints(self):
        self.assertEqual(
            rd.infer_predictor_mode(self._p("waypointnet_best.pt")), "waypoints")

    def test_ranknet_is_rank(self):
        self.assertEqual(
            rd.infer_predictor_mode(self._p("ranknet_best.pt")), "rank")

    def test_unknown_class_is_rejected(self):
        """모르는 모델이면 임의로 고르지 말고 멈춘다."""
        import torch
        from torch import nn

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.pt")
            torch.save(nn.Linear(3, 1), p)
            with self.assertRaises(SystemExit):
                rd.infer_predictor_mode(p)


class TestExperimentMapPolicy(unittest.TestCase):
    def test_window_generation_requires_an_external_map(self):
        with self.assertRaises(SystemExit) as cm:
            rd.enforce_experiment_map_policy(ns(windows_dir="out/windows"), None)
        self.assertIn("--carla-maps", str(cm.exception))

    def test_opendrive_is_accepted(self):
        rd.enforce_experiment_map_policy(
            ns(windows_dir="out/windows"), "/maps/Town05.xodr"
        )

    def test_geojson_is_accepted(self):
        rd.enforce_experiment_map_policy(
            ns(windows_dir="out/windows", road_map="map.geojson"), None
        )

    def test_explicit_exploratory_override_is_accepted(self):
        rd.enforce_experiment_map_policy(
            ns(windows_dir="out/windows", allow_trajectory_map=True), None
        )

    def test_non_experiment_scene_conversion_keeps_the_fallback(self):
        rd.enforce_experiment_map_policy(ns(), None)


if __name__ == "__main__":
    unittest.main()
