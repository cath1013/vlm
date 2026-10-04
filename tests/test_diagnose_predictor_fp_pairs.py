import unittest
from types import SimpleNamespace
from unittest.mock import patch

from examples import diagnose_predictor_fp_pairs as diagnose
from examples import audit_gt_carla_boxes as raw_audit
from traffic_llm.schemas import ActorState, PredictedPath


class DiagnosePredictorFpPairsTest(unittest.TestCase):
    def test_same_actor_pair_is_compared_in_predictor_and_gt(self):
        row = {"scenario_type": "type", "scenario": "scene", "window": "0-5", "bucket": 1,
               "cutoff_s": 5., "actor_pair": ["actor-a", "actor-b"]}
        pred = [SimpleNamespace(actor_id="actor-a", cls="car", speed_mps=1., predictions=[]),
                SimpleNamespace(actor_id="actor-b", cls="car", speed_mps=2., predictions=[])]
        gt = [SimpleNamespace(actor_id="actor-a", cls="car", speed_mps=1., predictions=[]),
              SimpleNamespace(actor_id="actor-b", cls="car", speed_mps=2., predictions=[])]
        context = ({"0-5": SimpleNamespace(last=SimpleNamespace(actors=pred))},
                   {"0-5": SimpleNamespace(last=SimpleNamespace(actors=gt))}, {},
                   SimpleNamespace(actor_cap=lambda _n: 2), {})
        calls = []
        def evaluate(actors, pair, *_args):
            calls.append((actors, pair))
            return None, (None, None), [next(a for a in actors if a.actor_id == aid) for aid in pair]
        with patch.object(diagnose, "rank_actors", return_value=(pred, None)), \
             patch.object(diagnose.geometry, "identities", return_value={}), \
             patch.object(diagnose, "_evaluate_pair", side_effect=evaluate), \
             patch.object(diagnose, "_trajectory_payload", return_value={}):
            diagnose.diagnose_fp_row(row, context, 1.)
        self.assertEqual([pair for _actors, pair in calls], [("actor-a", "actor-b"), ("actor-a", "actor-b")])
        self.assertIs(calls[0][0], pred)
        self.assertIs(calls[1][0], gt)

    def test_noncontact_gt_pair_is_retained(self):
        # The regular contact reporter suppresses static non-contact pairs;
        # this diagnostic must nevertheless retain the requested GT pair.
        a = ActorState("A", "observed", "car", (0., 0.), 90., 0., 0.)
        b = ActorState("B", "observed", "car", (20., 0.), 90., 0., 0.)
        for actor, xy in ((a, (0., 0.)), (b, (20., 0.))):
            actor.predictions = [PredictedPath("straight", 1., [xy, xy], .1,
                                               waypoint_times_s=[0., .1])]
        def box(carla_id):
            return raw_audit.WorldBox(carla_id, (0., 0.), 0., (0., 1.), 4., 2., 1.)
        frames = {1: {1: box(1), 2: box(2)}, 2: {1: box(1), 2: box(2)}}
        pair, _paths, _actors = diagnose._evaluate_pair([a, b], ("A", "B"), frames,
                                                         {"A": {1}, "B": {2}}, 0., .1)
        self.assertIsNotNone(pair)
        self.assertFalse(pair.predicted_contact)

    def test_center_distances_come_from_xy(self):
        result = diagnose.center_distance_trajectories_from_xy(
            [0., .1], [[0., 0.], [3., 4.]], [[3., 4.], [0., 0.]],
            [[0., 0.], [0., 0.]], [[0., 5.], [12., 5.]])
        self.assertEqual(result["predictor_center_distance_m"], [5., 5.])
        self.assertEqual(result["gt_center_distance_m"], [5., 13.])
        self.assertEqual(result["predictor_relative_xy_m"][0], [-3., -4.])

    def test_only_input_scenarios_are_selected(self):
        rows = [{"scenario_type": "a", "scenario": "one", "window": "w", "bucket": 1, "cutoff_s": 0., "actor_pair": ["A", "B"]}]
        cohort = [{"scenario_type": "a", "scenario": "one"}, {"scenario_type": "a", "scenario": "two"}]
        self.assertEqual(diagnose.select_affected_scenarios(rows, cohort), [cohort[0]])

    def test_summary_row_count_matches_valid_input_rows(self):
        base = {"diagnostic_status": "ok", "predictor_same_pair": diagnose._metric_payload(None),
                "gt_same_pair": diagnose._metric_payload(None)}
        summary = diagnose.summarize([base, dict(base)], 2, 1)
        self.assertEqual(summary["successfully_diagnosed_rows"], 2)
        self.assertEqual(summary["missing_unavailable_rows"], 0)

    def test_none_gt_evaluation_is_unavailable_not_a_gt_noncontact(self):
        row = {"diagnostic_status": "gt_pair_unavailable",
               "predictor_same_pair": diagnose._metric_payload(None),
               "gt_same_pair": diagnose._metric_payload(None)}
        summary = diagnose.summarize([row], 1, 1)
        self.assertEqual(summary["missing_unavailable_rows"], 1)
        self.assertEqual(summary["gt_same_pair_noncontacts"], 0)


if __name__ == "__main__":
    unittest.main()
