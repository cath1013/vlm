# -*- coding: utf-8 -*-
"""과거 궤적 입력 — 인코딩·마스킹·모델 연결.

여기서 막으려는 것 셋.
  1. 이력이 없는 표본을 "정지해 있었다"로 오해하는 것 (valid 플래그가 그 구별이다)
  2. 전부 패딩인 행에서 NaN 이 나는 것 (transformer 어텐션이 실제로 그랬다)
  3. 이력을 받도록 학습된 모델을 2-인자로 불러 조용히 성능이 떨어지는 것
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from traffic_llm.predict_model import (  # noqa: E402
    HISTORY_DT_S,
    N_HISTORY_FEATURES,
    N_HISTORY_STEPS,
    PredictContext,
    encode,
    history_features,
)
from traffic_llm.schemas import ActorState  # noqa: E402


def actor(hist=None, xy=(100.0, 200.0)):
    return ActorState(
        actor_id="V1", kind="observed", cls="car", world_xy=xy,
        heading_deg=0.0, speed_mps=10.0, accel_mps2=0.0,
        track_history=hist or [],
    )


def ctx_with(hist, t_s=10.0):
    return PredictContext(actor=actor(hist), horizon_s=5.0, t_s=t_s)


class TestHistoryFeatures(unittest.TestCase):
    def test_shape_is_fixed_regardless_of_observation_rate(self):
        """관측 주기가 달라도 모양이 같아야 계약이 성립한다."""
        for dt in (0.5, 0.1, 1.0):
            hist = [(10.0 - k * dt, 100.0 - k * dt * 10, 200.0)
                    for k in range(int(6 / dt) + 1)][::-1]
            rows = history_features(ctx_with(hist))
            self.assertEqual(len(rows), N_HISTORY_STEPS)
            self.assertTrue(all(len(r) == N_HISTORY_FEATURES for r in rows))

    def test_no_history_is_marked_invalid_not_zero_motion(self):
        """이력이 없으면 valid=0 이어야 한다 — 0 만 채우면 '정지'로 읽힌다."""
        rows = history_features(ctx_with([]))
        self.assertTrue(all(r[-1] == 0.0 for r in rows))

    def test_partial_history_marks_only_the_missing_steps(self):
        """트랙이 2초 전에 생겼으면 오래된 단계만 invalid 다."""
        hist = [(9.0, 90.0, 200.0), (10.0, 100.0, 200.0)]
        rows = history_features(ctx_with(hist, t_s=10.0))
        valid = [r[-1] for r in rows]
        self.assertEqual(valid, [0.0, 0.0, 0.0, 0.0, 1.0],
                         "t-1 만 관측됐다")

    def test_coordinates_are_relative_to_now(self):
        """절대 ENU 를 넣으면 모델이 지역 좌표계 원점을 외운다."""
        hist = [(10.0 - k, 100.0 - 10.0 * k, 200.0) for k in range(6)][::-1]
        rows = history_features(ctx_with(hist))
        # t-1 은 현재보다 10m 뒤 → de = -10
        self.assertAlmostEqual(rows[-1][0], -10.0, places=3)
        self.assertAlmostEqual(rows[-1][1], 0.0, places=3)
        # 같은 궤적을 통째로 평행이동해도 특징이 같아야 한다
        moved = [(t, e + 5000.0, n - 3000.0) for t, e, n in hist]
        a2 = actor(moved, xy=(100.0 + 5000.0, 200.0 - 3000.0))
        rows2 = history_features(
            PredictContext(actor=a2, horizon_s=5.0, t_s=10.0))
        for r1, r2 in zip(rows, rows2):
            for v1, v2 in zip(r1, r2):
                self.assertAlmostEqual(v1, v2, places=3)

    def test_oldest_step_gets_a_velocity(self):
        """t-5 의 속도는 t-6 과의 차분이다 — 이력을 1초 더 받아야 채워진다."""
        hist = [(10.0 - k, 100.0 - 10.0 * k, 200.0) for k in range(7)][::-1]
        rows = history_features(ctx_with(hist))
        self.assertAlmostEqual(rows[0][2], 10.0, places=3,
                               msg="가장 오래된 단계도 속도가 있어야 한다")

    def test_row_order_is_oldest_first(self):
        """순환신경망에 그대로 넣으려면 시간 순서여야 한다."""
        hist = [(10.0 - k, 100.0 - 10.0 * k, 200.0) for k in range(7)][::-1]
        rows = history_features(ctx_with(hist))
        des = [r[0] for r in rows]
        self.assertEqual(des, sorted(des), "de 가 증가 = 과거→현재")

    def test_encode_includes_history(self):
        enc = encode(ctx_with([(10.0, 100.0, 200.0)]))
        self.assertIn("history", enc)
        self.assertEqual(len(enc["history"]), N_HISTORY_STEPS)

    def test_grid_spacing_matches_the_target(self):
        """과거·미래가 같은 1초 격자여야 대칭 구조가 된다."""
        self.assertEqual(HISTORY_DT_S, 1.0)


class TestHistoryEncoderModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch 없음")

    def test_all_kinds_are_finite_including_empty_history(self):
        """전부 패딩인 행에서 NaN 이 나면 손실 전체가 오염된다."""
        import torch
        from traffic_llm.predict_nets import HistoryEncoder

        h = torch.zeros(3, N_HISTORY_STEPS, N_HISTORY_FEATURES)
        h[0, :, -1] = 1.0          # 전부 유효
        h[1, 3:, -1] = 1.0         # 최근 2개만
        for kind in HistoryEncoder.KINDS:
            enc = HistoryEncoder(kind=kind)
            enc.train()
            out = enc(h)
            self.assertTrue(torch.isfinite(out).all(), f"{kind} 출력 NaN")
            out.sum().backward()
            for p in enc.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all(),
                                    f"{kind} 기울기 NaN")

    def test_empty_history_encodes_to_zero_for_sequence_kinds(self):
        """계열 인코더는 '이력 없음'을 0 으로 낸다 (플래그로 구별한다)."""
        import torch
        from traffic_llm.predict_nets import HistoryEncoder

        h = torch.zeros(1, N_HISTORY_STEPS, N_HISTORY_FEATURES)
        for kind in ("gru", "transformer"):
            enc = HistoryEncoder(kind=kind)
            enc.eval()
            with torch.no_grad():
                self.assertAlmostEqual(float(enc(h).norm()), 0.0, places=5,
                                       msg=kind)

    def test_rejects_unknown_kind(self):
        from traffic_llm.predict_nets import HistoryEncoder

        with self.assertRaises(ValueError):
            HistoryEncoder(kind="lstm-ish")


class TestWaypointNetHistory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch 없음")

    def _shapes(self, kind):
        import torch
        from traffic_llm.predict_nets import WaypointNet

        m = WaypointNet(history_encoder=kind)
        m.eval()
        g = torch.zeros(N_HISTORY_FEATURES * 0 + 22)
        g[0] = 10.0
        c = torch.zeros(3, 12)
        c[:, 0] = 1 / 3
        h = torch.zeros(N_HISTORY_STEPS, N_HISTORY_FEATURES)
        h[:, -1] = 1.0
        with torch.no_grad():
            out = m(g, c, h) if m.accepts_history else m(g, c)
        return m, out

    def test_all_kinds_produce_the_inference_shape(self):
        for kind in ("gru", "transformer", "mlp", "none"):
            m, out = self._shapes(kind)
            wp, logits = out
            self.assertEqual(tuple(wp.shape), (5, 2), kind)
            self.assertEqual(tuple(logits.shape), (5,), kind)

    def test_none_does_not_accept_history(self):
        """이력을 쓰지 않는 설정은 3번째 인자를 요구하지 않아야 한다."""
        m, _ = self._shapes("none")
        self.assertFalse(m.accepts_history)

    def test_missing_history_is_an_error_not_a_silent_zero(self):
        """이력을 받도록 학습된 모델을 2-인자로 부르면 멈춰야 한다."""
        import torch
        from traffic_llm.predict_nets import WaypointNet

        m = WaypointNet(history_encoder="gru")
        g = torch.zeros(22)
        c = torch.zeros(2, 12)
        c[:, 0] = 0.5
        with self.assertRaises(ValueError) as cm:
            m(g, c)
        self.assertIn("과거 궤적", str(cm.exception))

    def test_history_changes_the_output(self):
        """이력을 넣었는데 출력이 그대로면 연결이 끊긴 것이다."""
        import torch
        from traffic_llm.predict_nets import WaypointNet

        torch.manual_seed(0)
        m = WaypointNet(history_encoder="gru", residual=False)
        m.eval()
        g = torch.zeros(22)
        g[0] = 10.0
        c = torch.zeros(2, 12)
        c[:, 0] = 0.5
        h1 = torch.zeros(N_HISTORY_STEPS, N_HISTORY_FEATURES)
        h1[:, -1] = 1.0
        h2 = h1.clone()
        h2[:, 0] = 20.0          # 다른 과거 궤적
        with torch.no_grad():
            a = m(g, c, h1)[0]
            b = m(g, c, h2)[0]
        self.assertFalse(torch.allclose(a, b), "이력이 출력에 반영되지 않는다")

    def test_spec_records_the_history_setting(self):
        """저장된 모델에서 무엇으로 학습했는지 알 수 있어야 한다."""
        from traffic_llm.predict_nets import WaypointNet

        m = WaypointNet(history_encoder="transformer", history_hidden=32)
        self.assertEqual(m.spec["history_encoder"], "transformer")
        self.assertEqual(m.spec["history_hidden"], 32)
        self.assertEqual(m.spec["n_history_steps"], N_HISTORY_STEPS)


if __name__ == "__main__":
    unittest.main()
