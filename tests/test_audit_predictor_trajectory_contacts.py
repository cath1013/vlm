import unittest

from examples import audit_gt_carla_boxes as raw_audit
from examples import audit_predictor_trajectory_contacts as audit
from traffic_llm.schemas import ActorState, PredictedPath


def actor(actor_id, xy, future):
    value = ActorState(actor_id, "observed", "car", xy, 90., 4., 0.)
    value.predictions = [PredictedPath("straight", 1., [xy, future], .1,
                                       waypoint_times_s=[0., .1])]
    return value


def box(carla_id, length=2., width=1.):
    return raw_audit.WorldBox(carla_id, (99., 99.), 0., (0., 1.), length, width, 1.)


def swept(a="A", b="B"):
    return audit.geometry.SweptPair(a, b, 0., .1, 1, True, 1., "straight", "straight",
                                    1., .1, .1, 0., 0., "NEW_CONTACT", False, False)


def record(predicted, oracle, *, positive=False, pred_hit=False, oracle_hit=False):
    return {"scenario_type": "accident", "scenario": "scene", "window": "w", "bucket": 1,
            "cutoff_s": 0., "gt_positive": positive, "target_carla_ids": [1, 2],
            "identities": {"A": {1}, "B": {2}, "C": {3}, "D": {4}},
            "pairs": {audit.PREDICTED_XY_ORACLE_GEOMETRY: predicted,
                      audit.GT_TRAJECTORY_ORACLE_GEOMETRY: oracle},
            "hits": {audit.PREDICTED_XY_ORACLE_GEOMETRY: pred_hit,
                     audit.GT_TRAJECTORY_ORACLE_GEOMETRY: oracle_hit}}


class PredictorTrajectoryContactsTest(unittest.TestCase):
    def test_predictor_xy_is_used_not_raw_or_gt_xy(self):
        # Raw centers intentionally disagree with predictor XY; raw boxes are
        # consulted only for yaw/dimensions, so the predicted convergence wins.
        a, b = actor("A", (0., 0.), (2., 0.)), actor("B", (5., 0.), (2., 0.))
        frames = {1: {1: box(1), 2: box(2)}, 2: {1: box(1), 2: box(2)}}
        pairs, _ = audit._contact_pairs([a, b], {"A": {1}, "B": {2}}, frames, 0., .1)
        self.assertIn(frozenset(("A", "B")), pairs)

    def test_raw_yaw_substitution_does_not_modify_predictor_xy(self):
        a = actor("A", (1., 2.), (4., 6.))
        pose = audit.geometry._future_pose(a, a.predictions[0], .1, lambda _aid, _t: (0., 1.))[0]
        self.assertEqual(pose(.1)[:2], (4., 6.))
        self.assertEqual(pose(.1)[2:], (0., 1.))

    def test_exact_dimensions_are_used(self):
        a, b = actor("A", (0., 0.), (0., 0.)), actor("B", (3.5, 0.), (3.5, 0.))
        frames = {1: {1: box(1, 2., 1.), 2: box(2, 2., 1.)},
                  2: {1: box(1, 2., 1.), 2: box(2, 2., 1.)}}
        pairs, coverage = audit._contact_pairs([a, b], {"A": {1}, "B": {2}}, frames, 0., .1)
        self.assertNotIn(frozenset(("A", "B")), pairs)
        self.assertEqual(coverage["exact_size_available"], coverage["exact_size_queries"])

    def test_trajectory_fp_classification(self):
        p = swept()
        rows = audit.trajectory_differences([record({frozenset(("A", "B")): p}, {}, positive=False)])
        self.assertEqual([row["difference_category"] for row in rows], ["TRAJECTORY_FP"])

    def test_trajectory_fn_classification(self):
        p = swept()
        rows = audit.trajectory_differences([record({}, {frozenset(("A", "B")): p}, positive=True)])
        self.assertIn("TRAJECTORY_FN", [row["difference_category"] for row in rows])

    def test_target_pair_miss_classification(self):
        rows = audit.trajectory_differences([record({frozenset(("C", "D")): swept("C", "D")},
                                                   {frozenset(("A", "B")): swept()}, positive=True,
                                                   pred_hit=False, oracle_hit=True)])
        self.assertIn("TRAJECTORY_TARGET_PAIR_MISS", [row["difference_category"] for row in rows])


if __name__ == "__main__":
    unittest.main()
