import unittest
from unittest.mock import patch

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


def pair():
    return audit.SweptPair("A", "B", 0.0, 0.1, 1, True, 1.0, "straight", "straight",
                           1.0, 0.1, 0.1, 0.0, 0.0, "NEW_CONTACT", False, False)


def fp_record(generic, exact):
    actors = {aid: actor() for aid in ("A", "B")}
    for aid, value in actors.items():
        value.actor_id = aid
    frames = {1: {7: box((1.0, 0.0)), 8: raw_audit.WorldBox(8, (0., 0.), 0., (1., 0.), 2., 1., 1.)}}
    return {"scenario_type": "normal", "scenario": "scene", "window": "w", "bucket": 0,
            "cutoff_s": 0., "gt_positive": False, "target_carla_ids": [],
            "identities": {"A": {7}, "B": {8}}, "actors": actors, "frame_boxes": frames,
            "pairs": {audit.RAW_BOX_YAW_GENERIC_SIZE: generic,
                      audit.RAW_BOX_YAW_EXACT_SIZE: exact}}


class HeadingSourceAblationTest(unittest.TestCase):
    def test_residual_raw_yaw_fp_categories(self):
        pair = audit.SweptPair(
            "A", "B", 0.0, 0.5, 1, True, 0.5, "straight", "left",
            1.0, 0.5, 0.1, 2.0, 3.0, "NEW_CONTACT", False, False,
        )

        def record(scenario_type, identities, target=(1, 2)):
            actors = {aid: actor() for aid in ("A", "B")}
            for aid, value in actors.items():
                value.actor_id = aid
            return {
                "scenario_type": scenario_type, "scenario": "scene", "window": "w",
                "bucket": 0, "cutoff_s": 1.0, "gt_positive": False,
                "target_carla_ids": list(target), "identities": identities,
                "actors": actors, "pairs": {audit.RAW_BOX_YAW: {frozenset(("A", "B")): pair}},
            }

        cases = [
            (record("normal", {"A": {9}, "B": {10}}), "NORMAL_SCENARIO_CONTACT"),
            (record("accident", {"A": {1}, "B": {2}}), "TARGET_PAIR_WRONG_BUCKET"),
            (record("accident", {"A": {1}, "B": {3}}), "ONE_TARGET_ACTOR_PLUS_OTHER"),
            (record("accident", {"A": {3}, "B": {4}}), "UNRELATED_ACTOR_PAIR"),
            (record("accident", {"A": {1, 3}, "B": {2}}), "IDENTITY_OR_TARGET_UNKNOWN"),
        ]
        for record_, expected in cases:
            with self.subTest(expected=expected):
                rows = audit.residual_raw_yaw_fp_rows([record_])
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["category"], expected)

    def test_raw_heading_is_direct_enu_vector(self):
        heading = audit._raw_heading({1: {7: box((0.0, 1.0))}}, {"A": {7}}, 0.0, "A", 0.0)
        self.assertEqual(heading, (0.0, 1.0))

    def test_missing_raw_heading_is_unknown_not_interpolated(self):
        frames = {1: {7: box((1.0, 0.0))}, 3: {7: box((0.0, 1.0))}}
        self.assertIsNone(audit._raw_heading(frames, {"A": {7}}, 0.0, "A", 0.1))
        self.assertIsNone(audit._raw_heading(frames, {"A": {7, 8}}, 0.0, "A", 0.0))

    def test_exact_footprint_uses_world_box_dimensions_and_requires_one_source_id(self):
        frames = {1: {7: raw_audit.WorldBox(7, (0., 0.), 0., (1., 0.), 7.5, 2.25, 1.)}}
        self.assertEqual(audit._exact_footprint(frames, {"A": {7}}, 0., "A", 0.), (7.5, 2.25))
        self.assertIsNone(audit._exact_footprint(frames, {"A": {7, 8}}, 0., "A", 0.))

    def test_generic_footprint_uses_actor_footprint_m(self):
        a, b = actor(), actor()
        b.actor_id, b.world_xy = "B", (20., 2.)
        b.predictions = [PredictedPath("straight", 1., [b.world_xy, b.world_xy], 1.,
                                       waypoint_times_s=[0., 1.])]
        frames = {f: {7: box((1., 0.)), 8: raw_audit.WorldBox(8, (0., 0.), 0., (1., 0.), 2., 1., 1.)}
                  for f in range(1, 12)}
        with patch.object(audit, "actor_footprint_m", return_value=(9., 3.)) as footprint:
            audit.raw_yaw_swept_pair_clearances([a, b], frame_boxes=frames,
                identities_by_actor={"A": {7}, "B": {8}}, cutoff_s=0., horizon_s=1.)
        self.assertGreater(footprint.call_count, 0)
        self.assertTrue(all(call.args[0] in (a, b) for call in footprint.call_args_list))

    def test_missing_exact_box_is_unavailable_not_generic_fallback(self):
        lookup = audit._ExactFootprintLookup({1: {7: box((1., 0.))}}, {"A": {7}}, 0.)
        self.assertIsNone(lookup(actor(), .1))
        self.assertEqual(lookup.report()["exact_size_missing"], 1)

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

    def test_generic_overlap_but_exact_separation_is_removed_fp(self):
        a, b = actor(), actor()
        a.world_xy, b.actor_id, b.world_xy = (0., 0.), "B", (5., 0.)
        a.predictions = [PredictedPath("straight", 1., [(0., 0.), (0., 0.)], 1.,
                                       waypoint_times_s=[0., .1])]
        b.predictions = [PredictedPath("straight", 1., [(5., 0.), (3.5, 0.)], 1.,
                                       waypoint_times_s=[0., .1])]
        frames = {1: {7: raw_audit.WorldBox(7, (0., 0.), 0., (1., 0.), 2., 1., 1.),
                      8: raw_audit.WorldBox(8, (5., 0.), 0., (1., 0.), 2., 1., 1.)},
                  2: {7: raw_audit.WorldBox(7, (0., 0.), 0., (1., 0.), 2., 1., 1.),
                      8: raw_audit.WorldBox(8, (3.5, 0.), 0., (1., 0.), 2., 1., 1.)}}
        args = dict(frame_boxes=frames, identities_by_actor={"A": {7}, "B": {8}}, cutoff_s=0., horizon_s=.1)
        generic = audit.raw_yaw_swept_pair_clearances([a, b], **args)
        exact = audit.raw_yaw_swept_pair_clearances([a, b], **args, exact_size=True)
        self.assertTrue(generic[0].predicted_contact)
        self.assertFalse(exact[0].predicted_contact)
        record = fp_record({frozenset(("A", "B")): generic[0]}, {})
        self.assertEqual(audit.exact_size_fp_comparison([record])[0]["status"], "REMOVED_BY_EXACT_SIZE")

    def test_exact_size_surviving_fp_is_classified(self):
        p = pair()
        record = fp_record({frozenset(("A", "B")): p}, {frozenset(("A", "B")): p})
        self.assertEqual(audit.exact_size_fp_comparison([record])[0]["status"], "SURVIVES_EXACT_SIZE")


if __name__ == "__main__":
    unittest.main()
