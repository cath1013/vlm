import unittest

from traffic_llm.schemas import ActorState, PredictedPath
from traffic_llm.swept_path import swept_pair_clearances


def actor(actor_id, xy, points, heading=90.0, speed=5.0, cls="car", history=None):
    a = ActorState(
        actor_id=actor_id,
        kind="observed",
        cls=cls,
        world_xy=xy,
        heading_deg=heading,
        speed_mps=speed,
        accel_mps2=0.0,
        track_history=history or [],
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

    def test_timestamped_paths_use_actual_subsecond_offsets(self):
        """Timestamped oracle samples must not be interpreted as one second apart."""
        a = actor("A", (-4.0, 0.0), [(-4.0, 0.0), (-2.0, 0.0), (0.0, 0.0)])
        b = actor("B", (4.0, 0.0), [(4.0, 0.0), (2.0, 0.0), (0.0, 0.0)],
                  heading=270.0)
        for vehicle in (a, b):
            vehicle.predictions[0].waypoint_times_s = [0.0, 0.1, 0.2]

        pair = swept_pair_clearances([a, b], horizon_s=5.0)[0]
        self.assertTrue(pair.predicted_contact)
        self.assertIsNotNone(pair.first_contact_s)
        self.assertLessEqual(pair.first_contact_s, 0.2)
        self.assertLessEqual(pair.time_after_observation_s, 0.2)
        self.assertEqual(pair.interval_index, 1)

    def test_parallel_paths_report_physical_gap(self):
        a = actor("A", (0.0, 0.0), [(0.0, 0.0), (10.0, 0.0)])
        b = actor("B", (0.0, 4.0), [(0.0, 4.0), (10.0, 4.0)])
        pair = swept_pair_clearances([a, b], horizon_s=1.0)[0]
        self.assertFalse(pair.predicted_contact)
        self.assertAlmostEqual(pair.minimum_clearance_m, 4.0 - 1.93, places=2)

    def test_contact_margin_changes_only_contact_classification(self):
        # The paths close from a safe initial gap to a positive final clearance.
        # The geometry itself must stay the same when the decision threshold moves.
        a = actor("A", (0.0, 0.0), [(0.0, 0.0), (10.0, 0.0)])
        b = actor("B", (0.0, 4.0), [(0.0, 4.0), (10.0, 2.5)])
        zero = swept_pair_clearances([a, b], horizon_s=1.0, contact_margin_m=0.0)[0]
        one = swept_pair_clearances([a, b], horizon_s=1.0, contact_margin_m=1.0)[0]
        self.assertFalse(zero.predicted_contact)
        self.assertTrue(one.predicted_contact)
        self.assertGreater(one.minimum_clearance_m, 0.0)
        self.assertLessEqual(one.minimum_clearance_m, 1.0)
        self.assertEqual(one.minimum_clearance_m, zero.minimum_clearance_m)

    def test_known_stopped_actor_is_extended_but_unknown_is_not(self):
        moving = actor("M", (-8.0, 0.0), [(-8.0, 0.0), (0.0, 0.0)])
        stopped = actor("S", (0.0, 0.0), [(0.0, 0.0)], speed=0.0)
        self.assertTrue(
            swept_pair_clearances([moving, stopped], horizon_s=1.0)[0].predicted_contact
        )
        unknown = actor("U", (0.0, 0.0), [(0.0, 0.0)], speed=None)
        self.assertEqual(swept_pair_clearances([moving, unknown], horizon_s=1.0), [])

    def test_paths_already_touching_without_history_are_uncertain(self):
        a = actor("A", (0.0, 0.0), [(0.0, 0.0), (1.0, 0.0)])
        b = actor("B", (0.0, 0.0), [(0.0, 0.0), (-1.0, 0.0)], heading=270.0)
        pair = swept_pair_clearances([a, b], horizon_s=1.0)[0]
        self.assertEqual(pair.contact_event, "RECONTACT_UNCERTAIN")
        self.assertFalse(pair.predicted_contact)

    def test_static_static_pair_is_excluded(self):
        a = actor("A", (0.0, 0.0), [(0.0, 0.0)], speed=0.0)
        b = actor("B", (8.0, 0.0), [(8.0, 0.0)], speed=0.0)
        self.assertEqual(swept_pair_clearances([a, b], horizon_s=1.0), [])

    @staticmethod
    def _timed_path(vehicle, points):
        vehicle.predictions[0].waypoints = points
        vehicle.predictions[0].waypoint_times_s = [0.1 * i for i in range(len(points))]

    def test_boundary_onset_from_previous_2hz_sample_is_predicted(self):
        a = actor("A", (0.0, 0.0), [(0.0, 0.0), (0.0, 0.0)],
                  history=[(-0.5, 0.0, 0.0), (0.0, 0.0, 0.0)])
        b = actor("B", (4.0, 0.0), [(4.0, 0.0), (4.0, 0.0)],
                  history=[(-0.5, 5.0, 0.0), (0.0, 4.0, 0.0)])
        self._timed_path(a, a.predictions[0].waypoints)
        self._timed_path(b, b.predictions[0].waypoints)
        pair = swept_pair_clearances([a, b], horizon_s=0.1)[0]
        self.assertEqual(pair.contact_event, "BOUNDARY_ONSET")
        self.assertTrue(pair.predicted_contact)
        self.assertEqual(pair.first_contact_s, 0.0)
        self.assertEqual(pair.interval_index, 1)

    def test_history_hole_cannot_manufacture_boundary_onset(self):
        a = actor("A", (0.0, 0.0), [(0.0, 0.0), (0.0, 0.0)],
                  history=[(-1.0, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.0, 0.0, 0.0)])
        b = actor("B", (4.0, 0.0), [(4.0, 0.0), (4.0, 0.0)],
                  history=[(-1.0, 5.0, 0.0), (0.0, 4.0, 0.0)])
        self._timed_path(a, a.predictions[0].waypoints)
        self._timed_path(b, b.predictions[0].waypoints)
        # The latest jointly valid state at -1.0 is separated, but B has a
        # 0.5-second-cadence history hole before the touching cutoff.
        pair = swept_pair_clearances([a, b], horizon_s=0.1, contact_lookback_s=1.1)[0]
        self.assertFalse(pair.contact_before_observation)
        self.assertEqual(pair.contact_event, "RECONTACT_UNCERTAIN")
        self.assertFalse(pair.predicted_contact)

    def test_preexisting_persistent_contact_is_not_predicted(self):
        a = actor("A", (0.0, 0.0), [(0.0, 0.0), (0.0, 0.0)],
                  history=[(-0.5, 0.0, 0.0), (0.0, 0.0, 0.0)])
        b = actor("B", (4.0, 0.0), [(4.0, 0.0), (4.0, 0.0)],
                  history=[(-0.5, 4.0, 0.0), (0.0, 4.0, 0.0)])
        self._timed_path(a, a.predictions[0].waypoints)
        self._timed_path(b, b.predictions[0].waypoints)
        pair = swept_pair_clearances([a, b], horizon_s=0.1)[0]
        self.assertEqual(pair.contact_event, "PREEXISTING_PERSISTENT")
        self.assertFalse(pair.predicted_contact)

    def test_static_actors_keep_confirmed_boundary_onset(self):
        a = actor("A", (0.0, 0.0), [(0.0, 0.0)], speed=0.0,
                  history=[(-0.5, 0.0, 0.0), (0.0, 0.0, 0.0)])
        b = actor("B", (4.0, 0.0), [(4.0, 0.0)], speed=0.0,
                  history=[(-0.5, 5.0, 0.0), (0.0, 4.0, 0.0)])
        pair = swept_pair_clearances([a, b], horizon_s=0.1)[0]
        self.assertEqual(pair.contact_event, "BOUNDARY_ONSET")
        self.assertTrue(pair.predicted_contact)

    def test_later_contact_is_new_contact(self):
        a = actor("A", (0.0, 0.0), [(0.0, 0.0), (0.0, 0.0)])
        b = actor("B", (6.0, 0.0), [(6.0, 0.0), (4.0, 0.0)])
        self._timed_path(a, a.predictions[0].waypoints)
        self._timed_path(b, b.predictions[0].waypoints)
        pair = swept_pair_clearances([a, b], horizon_s=0.1)[0]
        self.assertEqual(pair.contact_event, "NEW_CONTACT")
        self.assertTrue(pair.predicted_contact)
        self.assertEqual(pair.first_contact_s, 0.1)

    def test_recontact_is_predicted(self):
        a = actor("A", (0.0, 0.0), [(0.0, 0.0)] * 3,
                  history=[(-0.1, 0.0, 0.0), (0.0, 0.0, 0.0)])
        b = actor("B", (4.0, 0.0), [(4.0, 0.0), (6.0, 0.0), (4.0, 0.0)],
                  history=[(-0.1, 4.0, 0.0), (0.0, 4.0, 0.0)])
        self._timed_path(a, a.predictions[0].waypoints)
        self._timed_path(b, b.predictions[0].waypoints)
        pair = swept_pair_clearances([a, b], horizon_s=0.2)[0]
        self.assertEqual(pair.contact_event, "RECONTACT")
        self.assertTrue(pair.predicted_contact)
        self.assertEqual(pair.first_contact_s, 0.2)

    def test_regular_one_second_waypoints_are_not_internal_missing_data(self):
        a = actor("A", (0.0, 0.0), [(0.0, 0.0)] * 3,
                  history=[(-0.5, 0.0, 0.0), (0.0, 0.0, 0.0)])
        b = actor("B", (4.0, 0.0), [(4.0, 0.0), (6.0, 0.0), (4.0, 0.0)],
                  history=[(-0.5, 4.0, 0.0), (0.0, 4.0, 0.0)])
        pair = swept_pair_clearances([a, b], horizon_s=2.0)[0]
        self.assertEqual(pair.contact_event, "RECONTACT")
        self.assertTrue(pair.predicted_contact)

    def test_timestamped_internal_hole_makes_recontact_uncertain(self):
        a = actor("A", (0.0, 0.0), [(0.0, 0.0)] * 3,
                  history=[(-0.5, 0.0, 0.0), (0.0, 0.0, 0.0)])
        b = actor("B", (4.0, 0.0), [(4.0, 0.0), (6.0, 0.0), (4.0, 0.0)],
                  history=[(-0.5, 4.0, 0.0), (0.0, 4.0, 0.0)])
        for vehicle in (a, b):
            vehicle.predictions[0].waypoint_times_s = [0.0, 0.1, 0.4]
        pair = swept_pair_clearances([a, b], horizon_s=0.4)[0]
        self.assertEqual(pair.contact_event, "RECONTACT_UNCERTAIN")
        self.assertFalse(pair.predicted_contact)

    def test_recontact_with_missing_history_is_uncertain(self):
        a = actor("A", (0.0, 0.0), [(0.0, 0.0)] * 3,
                  history=[(0.0, 0.0, 0.0)])
        b = actor("B", (4.0, 0.0), [(4.0, 0.0), (6.0, 0.0), (4.0, 0.0)],
                  history=[(-0.1, 4.0, 0.0), (0.0, 4.0, 0.0)])
        self._timed_path(a, a.predictions[0].waypoints)
        self._timed_path(b, b.predictions[0].waypoints)
        pair = swept_pair_clearances([a, b], horizon_s=0.2)[0]
        self.assertEqual(pair.contact_event, "RECONTACT_UNCERTAIN")
        self.assertFalse(pair.predicted_contact)

    def test_never_contacts_is_none(self):
        a = actor("A", (0.0, 0.0), [(0.0, 0.0), (0.0, 0.0)])
        b = actor("B", (6.0, 0.0), [(6.0, 0.0), (6.0, 0.0)])
        self._timed_path(a, a.predictions[0].waypoints)
        self._timed_path(b, b.predictions[0].waypoints)
        pair = swept_pair_clearances([a, b], horizon_s=0.1)[0]
        self.assertEqual(pair.contact_event, "NONE")
        self.assertFalse(pair.predicted_contact)

    def test_legacy_mode_keeps_any_future_contact_behavior(self):
        a = actor("A", (0.0, 0.0), [(0.0, 0.0), (0.0, 0.0)])
        b = actor("B", (4.0, 0.0), [(4.0, 0.0), (4.0, 0.0)])
        self._timed_path(a, a.predictions[0].waypoints)
        self._timed_path(b, b.predictions[0].waypoints)
        pair = swept_pair_clearances([a, b], horizon_s=0.1, exclude_touching_now=False)[0]
        self.assertTrue(pair.predicted_contact)


if __name__ == "__main__":
    unittest.main()
