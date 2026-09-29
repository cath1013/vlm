import unittest
from types import SimpleNamespace
from unittest.mock import patch

from examples import audit_gt_carla_boxes as raw_audit
from examples import audit_gt_xy_source_ablation as audit
from traffic_llm.schemas import ActorState, PredictedPath
from traffic_llm.swept_path import actor_footprint_m
import traffic_llm.swept_path as swept_path


def actor(actor_id="A"):
    a = ActorState(actor_id, "observed", "car", (1., 2.), 90., 4., 0.)
    a.predictions = [PredictedPath("straight", 1., [(1., 2.), (3., 5.)], 1.,
                                   waypoint_times_s=[0., 1.])]
    return a


def box(center=(9., 8.), heading=(0., 1.)):
    return raw_audit.WorldBox(7, center, 0., heading, 4., 2., 1.)


class XySourceAblationTest(unittest.TestCase):
    def test_production_gt_xy_preserves_existing_motion_xy(self):
        a = actor()
        production = swept_path._motion_fn(a, a.predictions[0], 1.)[0]
        raw_heading_pose = audit.heading_audit._future_pose(
            a, a.predictions[0], 1., lambda _aid, _t: (0., 1.))[0]
        self.assertEqual(raw_heading_pose(.5)[:2], production(.5)[:2])

    def test_raw_xy_replaces_xy_but_keeps_raw_heading(self):
        a = actor()
        pose = audit._raw_xy_pose(a, a.predictions[0], 1., lambda _aid, _t: box())[0](.5)
        self.assertEqual(pose, (9., 8., 0., 1.))

    def test_generic_dimensions_and_pair_universe_are_unchanged(self):
        a, b = actor("A"), actor("B")
        self.assertEqual(actor_footprint_m(a), actor_footprint_m(a))
        self.assertEqual(audit._actor_pair_universe([a, b]), {frozenset(("A", "B"))})

    def test_missing_raw_xy_is_unknown_not_interpolated(self):
        lookup = audit._RawBoxLookup({1: {7: box()}, 3: {7: box()}}, {"A": {7}}, 0.)
        self.assertIsNone(lookup("A", .1))
        self.assertEqual(lookup.report()["raw_xy_missing"], 1)

    def test_complete_and_missing_fp_removals_are_distinct(self):
        self.assertEqual(audit._removed_category(True, True), "FP_removed_with_complete_raw_xy")
        self.assertEqual(audit._removed_category(True, False), "FP_removed_with_missing_raw_xy")
        self.assertIsNone(audit._removed_category(False, True))

    def test_raw_frame_cache_loads_each_absolute_frame_once(self):
        scenario = SimpleNamespace(agents={"ego": SimpleNamespace(frames=[1])})
        with patch.object(audit.raw_audit, "merge_frame_boxes", return_value=({7: box()}, {})) as merge:
            cache = audit._FrameBoxCache(scenario)
            self.assertIsNotNone(cache.get(1).get(7))
            self.assertIsNotNone(cache.get(1).get(7))
        merge.assert_called_once_with(scenario, 1)

    def test_production_gt_xy_raw_yaw_baseline_gate_is_exact(self):
        audit._assert_production_baseline(dict(audit.EXPECTED))
        changed = dict(audit.EXPECTED, FP=59)
        with self.assertRaises(RuntimeError):
            audit._assert_production_baseline(changed)


if __name__ == "__main__":
    unittest.main()
