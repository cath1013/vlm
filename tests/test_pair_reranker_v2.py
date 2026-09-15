import unittest

from traffic_llm.pair_reranker_v2 import (
    FEATURE_NAMES_V2, FEATURE_NAMES_V3, PairRerankerV2, PairRerankerV3,
    load_pair_reranker, pair_features_v2, pair_features_v3,
)
from traffic_llm.schemas import ActorState, PredictedPath
from traffic_llm.swept_path import swept_pair_clearances


def make_actor(actor_id, x, vx):
    actor = ActorState(
        actor_id=actor_id, kind="ego" if actor_id.startswith("EGO_") else "observed",
        cls="car", world_xy=(x, 0.0), heading_deg=90.0 if vx > 0 else 270.0,
        speed_mps=abs(vx), accel_mps2=0.0, track_age_s=2.0,
        track_history=[(0.0, x - vx, 0.0), (1.0, x, 0.0)],
    )
    actor.predictions = [PredictedPath(
        maneuver="straight", probability=1.0,
        waypoints=[(x, 0.0), (x + vx, 0.0)], horizon_s=1.0,
    )]
    return actor


class PairRerankerV2FeatureTest(unittest.TestCase):
    def test_feature_length_and_actor_symmetry(self):
        a, b = make_actor("EGO_a", -6.0, 6.0), make_actor("V001", 6.0, -6.0)
        pair = swept_pair_clearances([a, b], horizon_s=1.0)[0]
        reverse = swept_pair_clearances([b, a], horizon_s=1.0)[0]
        x = pair_features_v2(a, b, pair, 1.0)
        y = pair_features_v2(b, a, reverse, 1.0)
        self.assertEqual(len(x), len(FEATURE_NAMES_V2))
        self.assertEqual(x, y)

    def test_saved_model_round_trip_and_probability_range(self):
        n = len(FEATURE_NAMES_V2)
        model = PairRerankerV2(
            means=[0.0] * n, scales=[1.0] * n,
            hidden_weights=[[0.01] * n, [-0.01] * n],
            hidden_bias=[0.0, 0.0], output_weights=[0.5, -0.5],
            output_bias=0.0, threshold=0.25,
        )
        value = model.predict_features([1.0] * n)
        self.assertGreater(value, 0.0)
        self.assertLess(value, 1.0)
        self.assertEqual(model.to_dict()["model_type"], "dynamic_actor_pair_mlp_v2")

    def test_v3_features_preserve_symmetry_and_model_round_trip(self):
        a, b = make_actor("EGO_a", -6.0, 6.0), make_actor("V001", 6.0, -6.0)
        pair = swept_pair_clearances([a, b], horizon_s=1.0)[0]
        reverse = swept_pair_clearances([b, a], horizon_s=1.0)[0]
        self.assertEqual(pair_features_v3(a, b, pair, 1.0),
                         pair_features_v3(b, a, reverse, 1.0))
        n = len(FEATURE_NAMES_V3)
        model = PairRerankerV3(
            means=[0.0] * n, scales=[1.0] * n,
            hidden_weights=[[0.01] * n], hidden_bias=[0.0],
            output_weights=[0.5], output_bias=0.0, threshold=0.25,
        )
        self.assertEqual(model.to_dict()["model_type"], "risk_aware_actor_pair_mlp_v3")
        self.assertEqual(len(pair_features_v3(a, b, pair, 1.0)), n)
        self.assertTrue(model.uses_threshold_gate)


if __name__ == "__main__":
    unittest.main()
