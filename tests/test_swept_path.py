import unittest

from traffic_llm.schemas import ActorState, PredictedPath
from traffic_llm.swept_path import swept_pair_clearances


def actor(actor_id, xy, points, heading=90.0, speed=5.0, cls="car"):
    a = ActorState(
        actor_id=actor_id,
        kind="observed",
        cls=cls,
        world_xy=xy,
        heading_deg=heading,
        speed_mps=speed,
        accel_mps2=0.0,
    )
    a.predictions = [
        PredictedPath(
            maneuver="straight",
            probability=1.0,
            waypoints=points,
            horizon_s=float(max(0, len(points) - 1)),
        )
    ]
    return a


class SweptPathTest(unittest.TestCase):
    def test_interpolation_detects_crossing_between_integer_waypoints(self):
        # Centres cross at t=0.5 although every integer-time centre is distinct.
        a = actor("A", (-5.0, 0.0), [(-5.0, 0.0), (5.0, 0.0)])
        b = actor("B", (0.0, -5.0), [(0.0, -5.0), (0.0, 5.0)], heading=0.0)
        pair = swept_pair_clearances([a, b], horizon_s=1.0)[0]
        self.assertTrue(pair.predicted_contact)
        self.assertGreater(pair.time_after_observation_s, 0.0)
        self.assertLess(pair.time_after_observation_s, 1.0)
        self.assertEqual(pair.interval_index, 1)
        self.assertIsNotNone(pair.first_contact_s)
        self.assertGreater(pair.contact_duration_s, 0.0)
        self.assertGreaterEqual(pair.clearance_at_horizon_m, 0.0)

    def test_parallel_paths_report_physical_gap(self):
        a = actor("A", (0.0, 0.0), [(0.0, 0.0), (10.0, 0.0)])
        b = actor("B", (0.0, 4.0), [(0.0, 4.0), (10.0, 4.0)])
        pair = swept_pair_clearances([a, b], horizon_s=1.0)[0]
        self.assertFalse(pair.predicted_contact)
        self.assertAlmostEqual(pair.minimum_clearance_m, 4.0 - 1.93, places=2)

    def test_known_stopped_actor_is_extended_but_unknown_is_not(self):
        moving = actor("M", (-8.0, 0.0), [(-8.0, 0.0), (0.0, 0.0)])
        stopped = actor("S", (0.0, 0.0), [(0.0, 0.0)], speed=0.0)
        self.assertTrue(
            swept_pair_clearances([moving, stopped], horizon_s=1.0)[0].predicted_contact
        )
        unknown = actor("U", (0.0, 0.0), [(0.0, 0.0)], speed=None)
        self.assertEqual(swept_pair_clearances([moving, unknown], horizon_s=1.0), [])

    def test_paths_already_touching_are_not_future_conflicts(self):
        a = actor("A", (0.0, 0.0), [(0.0, 0.0), (1.0, 0.0)])
        b = actor("B", (0.0, 0.0), [(0.0, 0.0), (-1.0, 0.0)], heading=270.0)
        self.assertEqual(swept_pair_clearances([a, b], horizon_s=1.0), [])

    def test_static_static_pair_is_excluded(self):
        a = actor("A", (0.0, 0.0), [(0.0, 0.0)], speed=0.0)
        b = actor("B", (8.0, 0.0), [(8.0, 0.0)], speed=0.0)
        self.assertEqual(swept_pair_clearances([a, b], horizon_s=1.0), [])


if __name__ == "__main__":
    unittest.main()
