import unittest

from examples import audit_gt_carla_boxes as raw_audit
from examples import audit_gt_heading_source_ablation as audit
from traffic_llm.schemas import ActorState, PredictedPath
import traffic_llm.swept_path as swept_path


def actor():
    a = ActorState("A", "observed", "car", (1.0, 2.0), 90.0, 4.0, 0.0)
    a.predictions = [PredictedPath("straight", 1.0, [(1.0, 2.0), (3.0, 5.0)], 1.0,
                                   waypoint_times_s=[0.0, 1.0])]
    return a


def box(heading):
    return raw_audit.WorldBox(7, (99.0, 99.0), 0.0, heading, 4.0, 2.0, 1.0)


class HeadingSourceAblationTest(unittest.TestCase):
    def test_raw_heading_is_direct_enu_vector(self):
        heading = audit._raw_heading({1: {7: box((0.0, 1.0))}}, {"A": {7}}, 0.0, "A", 0.0)
        self.assertEqual(heading, (0.0, 1.0))

    def test_missing_raw_heading_is_unknown_not_interpolated(self):
        frames = {1: {7: box((1.0, 0.0))}, 3: {7: box((0.0, 1.0))}}
        self.assertIsNone(audit._raw_heading(frames, {"A": {7}}, 0.0, "A", 0.1))
        self.assertIsNone(audit._raw_heading(frames, {"A": {7, 8}}, 0.0, "A", 0.0))

    def test_missing_heading_coverage_counters(self):
        lookup = audit._HeadingLookup({1: {7: box((1.0, 0.0))}}, {"A": {7}}, 0.0)
        self.assertIsNotNone(lookup("A", 0.0))
        self.assertIsNone(lookup("A", 0.1))
        self.assertIsNone(lookup("A", 0.1))  # cached: one logical query
        self.assertEqual(lookup.report(), {
            "raw_heading_queries": 2, "raw_heading_available": 1,
            "raw_heading_missing": 1, "raw_heading_coverage": 0.5,
            "actor_pairs_skipped_missing_heading_at_t0": 0,
            "path_pairs_with_any_missing_future_heading": 0,
        })

    def test_removed_fp_with_complete_yaw_is_separated_from_missing_yaw(self):
        self.assertEqual(audit._removed_fp_coverage_category(True, True),
                         "FP_removed_with_complete_raw_heading")
        self.assertEqual(audit._removed_fp_coverage_category(True, False),
                         "FP_removed_with_missing_raw_heading")
        self.assertIsNone(audit._removed_fp_coverage_category(False, True))
        counts = audit._removed_fp_coverage_counts([
            {"FP_removed_coverage_category": "FP_removed_with_complete_raw_heading"},
            {"FP_removed_coverage_category": "FP_removed_with_missing_raw_heading"},
        ])
        self.assertEqual(counts["FP_removed_with_complete_raw_heading"], 1)
        self.assertEqual(counts["FP_removed_with_missing_raw_heading"], 1)

    def test_heading_substitution_preserves_production_xy(self):
        a = actor()
        tangent = swept_path._motion_fn(a, a.predictions[0], 1.0)[0]
        poses = audit._future_pose(a, a.predictions[0], 1.0, lambda _aid, _t: (0.0, 1.0))[0]
        for t in (0.0, 0.1, 0.5, 1.0):
            self.assertEqual(poses(t)[:2], tangent(t)[:2])
            self.assertEqual(poses(t)[2:], (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
