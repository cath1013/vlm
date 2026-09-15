"""InteractionWaypointNet 입력 계약과 관측 관계 회귀 테스트."""
import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from traffic_llm.predict_model import (
    N_CANDIDATE_FEATURES,
    N_GLOBAL_FEATURES,
    N_HISTORY_FEATURES,
    N_HISTORY_STEPS,
    N_INTERACTION_FEATURES,
    PredictContext,
    interaction_features,
)
from traffic_llm.schemas import ActorState


def actor(actor_id, *, kind="ego", xy=(0.0, 0.0), heading=0.0,
          speed=10.0, observed_by=None):
    return ActorState(
        actor_id=actor_id, kind=kind, cls="car", world_xy=xy,
        heading_deg=heading, speed_mps=speed, accel_mps2=-1.0,
        observed_by=observed_by or [],
    )


class TestInteractionFeatures(unittest.TestCase):
    def test_reverse_observed_by_becomes_directed_pair_features(self):
        # A observes B iff B.observed_by contains A's observer id ('a').
        a = actor("EGO_a", xy=(0.0, 0.0), observed_by=["b"])
        b = actor("EGO_b", xy=(0.0, 10.0), observed_by=["a"])
        row = interaction_features(PredictContext(
            actor=a, horizon_s=5.0, neighbors=[a, b]))[0]
        self.assertEqual(len(row), N_INTERACTION_FEATURES)
        self.assertAlmostEqual(row[0], 10.0)  # target heading 0°: north = forward
        self.assertEqual(row[11:16], [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_non_observer_actor_is_marked_unknown_not_unobserved(self):
        target = actor("EGO_a")
        other = actor("V1", kind="observed", xy=(0.0, 10.0))
        row = interaction_features(PredictContext(
            actor=target, horizon_s=5.0, neighbors=[other]))[0]
        self.assertEqual(row[12:16], [0.0, 0.0, 1.0, 0.0])


class TestInteractionWaypointNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch 없음")

    def test_forward_accepts_padded_and_empty_neighbor_sets(self):
        import torch
        from traffic_llm.predict_nets import InteractionWaypointNet

        model = InteractionWaypointNet(hidden=32, history_hidden=16, dropout=0.0)
        g = torch.zeros(2, N_GLOBAL_FEATURES)
        c = torch.zeros(2, 1, N_CANDIDATE_FEATURES)
        c[:, 0, 0] = 1.0
        h = torch.zeros(2, N_HISTORY_STEPS, N_HISTORY_FEATURES)
        interaction = torch.zeros(2, 2, N_INTERACTION_FEATURES)
        interaction[0, 0, -1] = 1.0
        mask = interaction[..., -1] > 0.5
        model.fit_normalizer(g, c, c[..., 0] > 0, interaction, mask)
        out, logits = model(g, c, h, interaction, mask)
        self.assertEqual(tuple(out.shape), (2, 5, 2))
        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(logits).all())

        empty = torch.zeros(1, 0, N_INTERACTION_FEATURES)
        out, _ = model(g[:1], c[:1], h[:1], empty)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
