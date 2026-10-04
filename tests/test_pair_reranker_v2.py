import json
import tempfile
import unittest
from pathlib import Path

from traffic_llm.pair_reranker_v2 import (
    FEATURE_NAMES_V2, FEATURE_NAMES_V3, FEATURE_NAMES_V4,
    PairRerankerV2, PairRerankerV3, PairRerankerV4,
    load_pair_reranker, pair_features_for_model, pair_features_v2,
    pair_features_v3, pair_features_v4,
)
from traffic_llm.schemas import ActorState, Interaction, PredictedPath, RoadPlacement
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

    def test_v4_route_eta_crossing_and_checkpoint(self):
        a, b = make_actor("EGO_a", -6.0, 6.0), make_actor("V001", 6.0, -6.0)
        def placement(distance):
            return RoadPlacement("road", "road", 0.0, 0.0, "east", 90.0,
                                 1, 2, None, distance, "junction")
        a.placement, b.placement = placement(24.0), placement(12.0)
        pair = swept_pair_clearances([a, b], horizon_s=1.0)[0]
        interaction = Interaction("crossing", b.actor_id, a.actor_id,
                                  conflict="orthogonal")
        values = pair_features_v4(a, b, pair, 1.0, [interaction])
        reverse = pair_features_v4(b, a, pair, 1.0, [interaction])
        self.assertEqual(values, reverse)
        self.assertEqual(len(values), len(FEATURE_NAMES_V4))
        got = dict(zip(FEATURE_NAMES_V4, values))
        self.assertEqual(got["same_next_junction"], 1.0)
        self.assertEqual(got["route_eta_available"], 1.0)
        self.assertEqual(got["eta_gap_s"], 2.0)
        self.assertEqual(got["has_crossing_interaction"], 1.0)
        self.assertEqual(got["crossing_conflict_orthogonal"], 1.0)
        b.placement.next_junction_id = "different_junction"
        different = dict(zip(FEATURE_NAMES_V4,
                             pair_features_v4(a, b, pair, 1.0, [])))
        self.assertEqual(different["same_next_junction"], 0.0)
        self.assertEqual(different["route_eta_available"], 0.0)
        for name in ("eta_a_s", "eta_b_s", "eta_gap_s", "min_eta_s"):
            self.assertEqual(different[name], 0.0)
        b.placement.next_junction_id = "junction"
        b.speed_mps = 0.0
        missing = dict(zip(FEATURE_NAMES_V4,
                           pair_features_v4(a, b, pair, 1.0, [])))
        self.assertEqual(missing["route_eta_available"], 0.0)
        self.assertEqual(missing["eta_gap_s"], 0.0)
        n = len(FEATURE_NAMES_V4)
        model = PairRerankerV4([0.0] * n, [1.0] * n, [[0.0] * n],
                               [0.0], [0.0], 0.0, 0.5)
        self.assertEqual(pair_features_for_model(model, a, b, pair, 1.0, []),
                         pair_features_v4(a, b, pair, 1.0, []))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(model.to_dict()))
            self.assertIsInstance(load_pair_reranker(str(path)), PairRerankerV4)


if __name__ == "__main__":
    unittest.main()
