import csv
import json
import random
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from examples import plot_v4_crn4_error_cases as plot
from traffic_llm.accident_qa import TimeWindow
from traffic_llm.schemas import ActorState, PredictedPath


def fixture():
    def actor(aid, cid, start, end, horizon):
        value = ActorState(aid, "observed", "car", (start, 0.), 90., 4., 0.,
                           source_track_ids=[cid],
                           track_history=[(4.4, start, 0.), (5., start, 0.)])
        value.predictions = [PredictedPath(
            "straight", 1., [(start, 0.), (end, 0.)], horizon,
            waypoint_times_s=[0., horizon])]
        return value
    a, b = actor("A", 1, 0., 0., 2.), actor("B", 2, 10., 0., 2.)
    ga, gb = actor("A", 1, 0., 0., 5.), actor("B", 2, 10., 0., 5.)
    def window(actors):
        return TimeWindow(0, 0., 5., [SimpleNamespace(t=5., actors=actors)])
    frames = {frame: {cid: plot.boxes.WorldBox(cid, (99., 99.), 0., (1., 0.), 2., 1., 1.)
                      for cid in (1, 2)} for frame in range(1, 102)}
    case = dict(window_id="type/scene:0:5.000", scenario_id="type/scene",
                comparison_group="V4_TP_to_CRN_FN", gt_bucket="4",
                crn_risk_2s=.4, crn_threshold_2s=.9,
                crn_pairs=[("A", "B")], v4_pairs=[("A", "B")])
    collision = SimpleNamespace(occurred=True, carla_ids=[1, 2], time_s=9.)
    return case, window([a, b]), window([ga, gb]), frames, collision


class CasePlotTest(unittest.TestCase):
    def test_pairs_accept_saved_formats_and_reject_invalid_ids(self):
        self.assertEqual(plot.parse_pairs("('A', 'B')", single=True), [("A", "B")])
        self.assertEqual(plot.parse_pairs('[["A","B"]]'), [("A", "B")])
        self.assertEqual(plot.parse_pairs("[]"), [])
        with self.assertRaises(ValueError):
            plot.parse_pairs("[[1, 2]]")

    def test_id_only_cases_use_frozen_ledgers_and_seeded_shuffle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["window_id", "scenario_id"])
                writer.writeheader()
                for wid in list({r["window_id"]: r for r in plot.read_csv(
                        plot.ANALYSIS / "v4_vs_crn4_2s_all_windows.csv")})[:8][::-1]:
                    writer.writerow(dict(window_id=wid, scenario_id=wid.rsplit(":", 2)[0]))
            rows = plot.load_cases(path)
            requested = [r["window_id"] for r in plot.read_csv(path)]
            expected = requested.copy()
            random.Random(20261006).shuffle(expected)
            self.assertEqual([r["window_id"] for r in rows], expected)
            self.assertNotEqual(expected, requested)
            self.assertEqual(rows, plot.load_cases(path))
            other = requested.copy()
            random.Random(17).shuffle(other)
            self.assertEqual([r["window_id"] for r in plot.load_cases(path, blind_seed=17)], other)
            self.assertNotEqual(other, expected)
            limited = requested[:3]
            random.Random(17).shuffle(limited)
            self.assertEqual([r["window_id"] for r in plot.load_cases(path, 3, 17)], limited)
            self.assertTrue(all(r["crn_pairs"] for r in rows))

    def test_strict_v4_join_and_frozen_values_override_case_values(self):
        wid = "type/scene:0:5.000"
        frozen = dict(window_id=wid, selected_pairs=[["A", "B"]], gt_pair_covered=False,
                      baseline_predicted=True, correct_pair_hit=False)
        with tempfile.TemporaryDirectory() as directory:
            ledger, cases = Path(directory) / "v4.jsonl", Path(directory) / "cases.csv"
            ledger.write_text(json.dumps(frozen) + "\n")
            row = dict(window_id=wid, scenario_id="type/scene", crn4_selected_pair="('A', 'B')",
                       crn4_correct_pair_hit="False", v4_selected_pairs='[["wrong","pair"]]',
                       v4_gt_pair_candidate_covered="True", v4_baseline_predicted="False",
                       v4_correct_gt_pair_hit="True")
            with cases.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            with patch.object(plot, "V4_LEDGER", ledger):
                result = plot.load_cases(cases)[0]
                self.assertEqual(result["v4_pairs"], [("A", "B")])
                self.assertIs(result["v4_gt_pair_candidate_covered"], False)
                self.assertIs(result["v4_baseline_predicted"], True)
                self.assertIs(result["v4_correct_gt_pair_hit"], False)
                self.assertIs(result["crn_correct_gt_pair_hit"], False)
                ledger.write_text("")
                with self.assertRaisesRegex(KeyError, "missing from frozen V4 ledger"):
                    plot.load_cases(cases)

    def test_per_group_limit_selects_each_group_reproducibly(self):
        requested = plot.read_csv(plot.ANALYSIS / "v4_vs_crn4_2s_all_windows.csv")[:12]
        for i, row in enumerate(requested):
            row["comparison_group"] = f"group_{i % 3}"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(requested[0]))
                writer.writeheader()
                writer.writerows(requested)
            selected = plot.load_cases(path, blind_seed=17, per_group_limit=1)
            self.assertEqual(len(selected), 3)
            self.assertEqual(sorted(r["comparison_group"] for r in selected),
                             ["group_0", "group_1", "group_2"])
            self.assertEqual(selected, plot.load_cases(path, blind_seed=17, per_group_limit=1))
            rng, expected = random.Random(17), []
            for group in ("group_0", "group_1", "group_2"):
                members = [r for r in requested if r["comparison_group"] == group]
                rng.shuffle(members)
                expected.append(members[0])
            rng.shuffle(expected)
            self.assertEqual([r["window_id"] for r in selected],
                             [r["window_id"] for r in expected])
            other = plot.load_cases(path, blind_seed=19, per_group_limit=1)
            self.assertNotEqual([r["window_id"] for r in selected],
                                [r["window_id"] for r in other])
            self.assertEqual(len(plot.load_cases(path, per_group_limit=99)), 12)
            filtered = plot.load_cases(path, groups=["group_2", "group_1"], per_group_limit=1,
                                       blind_seed=17)
            self.assertEqual(sorted(r["comparison_group"] for r in filtered), ["group_1", "group_2"])
            self.assertEqual(filtered, plot.load_cases(
                path, groups=["group_1", "group_2"], per_group_limit=1, blind_seed=17))
            # The first input row is group_0: filtering must precede --limit.
            limited = plot.load_cases(path, groups=["group_2"], limit=1)
            self.assertEqual([r["window_id"] for r in limited], [requested[2]["window_id"]])
            with self.assertRaisesRegex(ValueError, "have no cases: absent"):
                plot.load_cases(path, groups=["group_1", "absent"], per_group_limit=1)

    def test_limit_and_per_group_limit_are_rejected_together(self):
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            plot.load_cases("unused.csv", limit=1, per_group_limit=1)
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            plot.main(["--root", "unused", "--carla-maps", "unused",
                       "--limit", "1", "--per-group-limit", "1"])
        for limit in (0, -1):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "must be positive"):
                plot.load_cases("unused.csv", per_group_limit=limit)

    def test_clearance_uses_exact_footprints_and_raw_yaw_not_raw_centers(self):
        case, window, gt, frames, collision = fixture()
        *_, metadata, ttc = plot.prepare_case(case, window, gt, {}, collision, frames)
        self.assertEqual(metadata["predicted_min_clearance_m"], 0.)
        self.assertAlmostEqual(metadata["predicted_min_clearance_time_s"], 1.6)
        self.assertAlmostEqual(metadata["gt_min_clearance_0_2_m"], 4.)
        self.assertEqual(metadata["gt_min_clearance_0_2_time_s"], 2.)
        self.assertEqual(metadata["gt_min_clearance_2_5_m"], 0.)
        self.assertEqual(metadata["gt_first_contact_time_s"], 4.)
        self.assertFalse(metadata["collision_within_2s"])
        self.assertTrue(metadata["collision_after_2s"])
        self.assertEqual(ttc, 4.)
        self.assertEqual(window.last.actors[1].predictions[0].waypoints[-1], (0., 0.))

    def test_missing_actor_and_truncated_paths_fail(self):
        for missing in ("actor", "path"):
            case, window, gt, frames, collision = fixture()
            if missing == "actor":
                case["v4_pairs"].append(("A", "missing"))
            else:
                gt.last.actors[1].predictions[0].waypoint_times_s = [0.]
                gt.last.actors[1].predictions[0].waypoints = [(10., 0.)]
            with self.subTest(missing=missing), self.assertRaises((ValueError, KeyError)):
                plot.prepare_case(case, window, gt, {}, collision, frames)

    def test_gt_data_end_is_explicit_and_never_extrapolated(self):
        case, window, gt, frames, collision = fixture()
        for actor in gt.last.actors:
            actor.predictions[0].waypoint_times_s[-1] = 1.
        _, _, _, truth, metadata, _ = plot.prepare_case(case, window, gt, {}, collision, frames)
        self.assertEqual(max(t for t, gap in truth["samples"][("A", "B")]), 1.)
        self.assertIsNone(metadata["gt_min_clearance_2_5_m"])
        self.assertIsNone(metadata["gt_min_clearance_2_5_time_s"])

    def test_missing_geometry_retains_xy_and_marks_contact_coverage_uncertain(self):
        import matplotlib.figure
        case, window, gt, frames, collision = fixture()
        # Predicted separation cannot be confirmed across a missing raw box.
        window.last.actors[1].predictions[0].waypoints[-1] = (10., 0.)
        del frames[68][2]  # cutoff=5 s, missing raw geometry at +1.7 s
        actors, identities, predicted, truth, metadata, ttc = plot.prepare_case(
            case, window, gt, {}, collision, frames)
        self.assertEqual(predicted["xy_paths"]["B"][0][1](1.7)[:2], (10., 0.))
        self.assertIsNone(predicted["paths"]["B"][0][1](1.7))
        self.assertFalse(metadata["predicted_geometry_complete_0_2"])
        self.assertFalse(metadata["gt_geometry_complete_0_2"])
        self.assertTrue(metadata["gt_geometry_complete_2_5"])
        for key in ("predicted_missing_geometry_times_s", "gt_missing_geometry_times_s"):
            self.assertEqual(json.loads(metadata[key]), [1.7])
        self.assertNotIn(1.7, [t for t, gap in predicted["samples"][("A", "B")]])
        start, end = predicted["missing_intervals"][("A", "B")][0]
        self.assertLess(start, 1.7)
        self.assertGreater(end, 1.7)
        self.assertEqual(predicted["events"][("A", "B")], ("GEOMETRY_UNCERTAIN", None))
        self.assertEqual(metadata["gt_bucket"], 4)
        self.assertEqual(ttc, 4.)
        captured = []
        queried_geometry = {}
        original_footprint = plot.boxes.oriented_footprint
        def track_size(original):
            def size(actor, t):
                dimensions = original(actor, t)
                queried_geometry.update(actor=actor.actor_id, t=t, dimensions=dimensions)
                return dimensions
            return size
        predicted["size"] = track_size(predicted["size"])
        truth["size"] = track_size(truth["size"])
        def footprint(center, heading, length, width):
            self.assertNotEqual((queried_geometry["actor"], queried_geometry["t"]), ("B", 1.7))
            self.assertEqual((length, width), queried_geometry["dimensions"])
            return original_footprint(center, heading, length, width)
        def save(fig, *args, **kwargs):
            captured.append("\n".join(text.get_text() for text in fig.texts))
            # XY at the missing sample remains drawn in the prediction panel.
            self.assertTrue(any(len(line.get_xdata()) == 21
                                for line in fig.axes[1].lines))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(matplotlib.figure.Figure, "savefig", save), \
                patch.object(plot.boxes, "oriented_footprint", side_effect=footprint) as draw:
            # Even a requested contact footprint at the gap must be skipped.
            predicted["events"][("A", "B")] = ("GEOMETRY_UNCERTAIN", 1.7)
            plot.render_case("case_001", actors, identities, predicted, truth, metadata, ttc,
                             Path(directory))
            self.assertGreater(draw.call_count, 0)
        self.assertIn("Geometry coverage incomplete: +1.7 s", captured[0])
        self.assertIn("available-sample minimum (coverage incomplete)", captured[0])

    def test_entirely_missing_geometry_still_renders_xy_without_footprints(self):
        import matplotlib.figure
        import matplotlib.pyplot  # Initialize pyplot before mocking Figure.savefig.
        case, window, gt, _, collision = fixture()
        actors, identities, predicted, truth, metadata, ttc = plot.prepare_case(
            case, window, gt, {}, collision, {})
        self.assertIsNone(metadata["predicted_min_clearance_m"])
        self.assertIsNone(metadata["gt_min_clearance_0_2_m"])
        self.assertIsNone(metadata["gt_min_clearance_2_5_m"])
        self.assertEqual(metadata["gt_bucket"], 4)
        self.assertEqual(predicted["events"][("A", "B")], ("GEOMETRY_UNCERTAIN", None))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(matplotlib.figure.Figure, "savefig") as save, \
                patch.object(plot.boxes, "oriented_footprint") as draw:
            plot.render_case("case_001", actors, identities, predicted, truth, metadata, ttc,
                             Path(directory))
            self.assertEqual(save.call_count, 2)
            draw.assert_not_called()

    def test_gt_collision_pair_contact_is_separate_from_listed_pairs_and_dataset(self):
        from copy import deepcopy
        case, window, gt, frames, collision = fixture()
        for win, horizon in ((window, 2.), (gt, 1.)):
            c = deepcopy(win.last.actors[1])
            c.actor_id, c.source_track_ids = "C", [3]
            c.predictions[0].waypoint_times_s[-1] = horizon
            win.last.actors.append(c)
        for frame in frames.values():
            frame[3] = plot.boxes.WorldBox(3, (99., 99.), 0., (1., 0.), 2., 1., 1.)
        case["crn_pairs"] = [("A", "C")]
        case["gt_bucket"] = "5"
        collision.time_s = 9.2
        *_, metadata, ttc = plot.prepare_case(case, window, gt, {}, collision, frames)
        self.assertEqual(metadata["gt_first_contact_time_s"], 4.)
        self.assertEqual(metadata["listed_pairs_gt_first_contact_time_s"], .8)
        self.assertAlmostEqual(ttc, 4.2)
        self.assertEqual(metadata["gt_bucket"], 5)
        case["gt_bucket"] = "0"
        *_, metadata, ttc = plot.prepare_case(case, window, gt, {}, None, frames)
        self.assertIsNone(metadata["gt_first_contact_time_s"])
        self.assertEqual(metadata["listed_pairs_gt_first_contact_time_s"], .8)
        self.assertIsNone(ttc)

    def test_unobservable_gt_collision_pair_keeps_case_and_dataset_timing(self):
        import matplotlib.figure
        import matplotlib.pyplot
        from copy import deepcopy
        for gt_ids in ([1, 99], [98, 99]):
            case, window, gt, frames, collision = fixture()
            collision.carla_ids = gt_ids
            historical = deepcopy(window.last.actors[1])
            historical.actor_id, historical.source_track_ids = "historical_gt", [99]
            window.snapshots.insert(0, SimpleNamespace(t=4.5, actors=[historical]))
            actors, identities, predicted, truth, metadata, ttc = plot.prepare_case(
                case, window, gt, {}, collision, frames)
            self.assertFalse(metadata["gt_collision_pair_observable_at_t0"])
            self.assertEqual(json.loads(metadata["gt_pair"]), gt_ids)
            self.assertEqual(metadata["gt_bucket"], 4)
            self.assertEqual(ttc, 4.)
            self.assertEqual(set(actors), {"A", "B"})
            self.assertEqual(list(predicted["samples"]), [("A", "B")])
            self.assertEqual(list(truth["samples"]), [("A", "B")])
            self.assertIsNone(metadata["gt_first_contact_time_s"])
            self.assertEqual(json.loads(metadata["crn_selected_pair"]), ["A", "B"])
            self.assertEqual(json.loads(metadata["v4_selected_pairs"]), [["A", "B"]])
            texts = []
            original = matplotlib.figure.Figure.savefig
            def save(fig, path, **kwargs):
                texts.append("\n".join(text.get_text() for text in fig.texts))
                return original(fig, path, **kwargs)
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                for kind in ("blind", "diagnostic"):
                    (out / kind).mkdir()
                with patch.object(matplotlib.figure.Figure, "savefig", save):
                    plot.render_case("case_001", actors, identities, predicted, truth, metadata, ttc, out)
                for kind in ("blind", "diagnostic"):
                    self.assertTrue((out / kind / "case_001.png").is_file())
            self.assertIn("GT collision pair observable at t=0: False", texts[1])
        case, window, gt, frames, collision = fixture()
        *_, metadata, _ = plot.prepare_case(case, window, gt, {}, collision, frames)
        self.assertTrue(metadata["gt_collision_pair_observable_at_t0"])

    def test_continue_on_error_preserves_successes_logs_failures_and_skips_pdfs(self):
        from copy import deepcopy
        from PIL import Image
        export_pdfs = plot.export_case_pdfs
        case, window, gt, frames, collision = fixture()
        cases, windows = [], {}
        for i in range(3):
            row, win = deepcopy(case), deepcopy(window)
            row["window_id"], win.index = f"type/scene:{i}:5.000", i
            cases.append(row)
            windows[str(i)] = win
        scenario = SimpleNamespace(scenario_id="type/scene", scenario="scene", scenario_type="type",
                                   town="Town01", split="val", root="selected/val",
                                   agents={"agent": SimpleNamespace(frames=list(frames))})
        validation = [scenario] + [SimpleNamespace(scenario_id=f"unused/{i}", split="val")
                                   for i in range(103)]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(plot, "load_cases", return_value=cases), \
                patch.object(plot, "DeepAccidentRunner") as runner, \
                patch.object(plot, "replay_windows", return_value=(windows, {gt.label: gt}, {}, None)), \
                patch.object(plot.boxes, "merge_frame_boxes", side_effect=lambda sc, f: (frames[f], {})), \
                patch.object(plot, "estimate_collision", return_value=collision), \
                patch.object(plot, "export_case_pdfs") as pdf, patch("builtins.print") as log:
            runner.return_value.list_scenarios.return_value = validation
            out = Path(directory) / "continued"
            def write_pngs(case_id, *args, **kwargs):
                if case_id == "case_002":
                    raise ValueError("synthetic render failure")
                for kind in ("blind", "diagnostic"):
                    Image.new("RGB", (16, 8)).save(args[-1] / kind / f"{case_id}.png")
            argv = ["--root", directory, "--carla-maps", directory, "--out", str(out)]
            with patch.object(plot, "render_case", side_effect=write_pngs) as render:
                with self.assertRaisesRegex(RuntimeError, "1 case.*failed; PDFs skipped"):
                    plot.main(argv + ["--continue-on-error"])
                self.assertEqual([call.args[0] for call in render.call_args_list],
                                 ["case_001", "case_002", "case_003"])
            pdf.assert_not_called()
            self.assertEqual(plot.read_csv(out / "failed_cases.csv"), [dict(
                case_id="case_002", window_id=cases[1]["window_id"],
                exception_type="ValueError", error_message="synthetic render failure")])
            self.assertTrue(any("FAILED case_002" in call.args[0] for call in log.call_args_list))
            for name in ("case_plot_metadata.csv", "case_pair_metadata.csv", "case_id_mapping.csv"):
                self.assertEqual([r["case_id"] for r in plot.read_csv(out / name)], ["case_001", "case_003"])
            for kind in ("blind", "diagnostic"):
                self.assertTrue((out / kind / "case_001.png").is_file())
                self.assertTrue((out / kind / "case_003.png").is_file())
                self.assertFalse((out / f"{kind}_cases.pdf").exists())
            # Default mode still stops at the failing case.
            fast_out = Path(directory) / "fail_fast"
            with patch.object(plot, "render_case", side_effect=write_pngs) as render:
                with self.assertRaisesRegex(RuntimeError, "case_002.*synthetic render failure"):
                    plot.main(argv[:-1] + [str(fast_out)])
                self.assertEqual(render.call_count, 2)
            self.assertFalse((fast_out / "failed_cases.csv").exists())
            # No failures with the flag leaves PDF assembly unchanged.
            good_out = Path(directory) / "all_good"
            def good_pngs(case_id, *args, **kwargs):
                for kind in ("blind", "diagnostic"):
                    Image.new("RGB", (16, 8)).save(args[-1] / kind / f"{case_id}.png")
            with patch.object(plot, "render_case", side_effect=good_pngs), \
                    patch.object(plot, "export_case_pdfs", wraps=export_pdfs) as good_pdf:
                plot.main(argv[:-1] + [str(good_out), "--continue-on-error"])
            good_pdf.assert_called_once_with(good_out, ["case_001", "case_002", "case_003"])
            for kind in ("blind", "diagnostic"):
                self.assertEqual(len(re.findall(rb"/Type /Page\b", (
                    good_out / f"{kind}_cases.pdf").read_bytes())), 3)
            self.assertFalse((good_out / "failed_cases.csv").exists())

    def test_render_blinds_labels_and_shares_limits(self):
        import matplotlib.figure
        case, window, gt, frames, collision = fixture()
        actors, identities, predicted, truth, metadata, ttc = plot.prepare_case(
            case, window, gt, {}, collision, frames)
        captured = []
        original = matplotlib.figure.Figure.savefig
        def save(fig, path, **kwargs):
            self.assertEqual(fig.axes[1].get_title(), "JointScene V2 predicted future (0–2 s)")
            self.assertEqual(fig.axes[2].get_title(), "GT future (available data up to 5 s)")
            self.assertEqual([len(ax.child_axes) for ax in fig.axes], [0, 1, 1])
            for ax in fig.axes[1:]:
                inset = ax.child_axes[0]
                self.assertEqual(len(inset.patches), 2)
                self.assertGreaterEqual(len(inset.lines), 2)
                self.assertTrue(any("A / B" in text.get_text() for text in inset.texts))
            captured.append((Path(path).parent.name,
                             "\n".join(t.get_text() for t in fig.texts),
                             [(ax.get_xlim(), ax.get_ylim(), ax.get_aspect()) for ax in fig.axes]))
            return original(fig, path, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for kind in ("blind", "diagnostic"):
                (out / kind).mkdir()
            with patch.object(matplotlib.figure.Figure, "savefig", save):
                plot.render_case("case_001", actors, identities, predicted, truth, metadata, ttc, out)
            for kind in ("blind", "diagnostic"):
                self.assertTrue((out / kind / "case_001.png").read_bytes().startswith(b"\x89PNG"))
        blind, diagnostic = captured
        self.assertNotIn("V4", blind[1])
        self.assertNotIn("CRN", blind[1])
        self.assertNotIn("TP", blind[1])
        self.assertNotIn("FN", blind[1])
        self.assertIn(case["comparison_group"], diagnostic[1])
        for _, _, limits in captured:
            self.assertEqual(limits[0], limits[1])
            self.assertEqual(limits[1], limits[2])
            self.assertEqual(limits[0][2], 1.)

    def test_render_footprint_times_are_pair_local_and_within_path_coverage(self):
        from copy import deepcopy
        import matplotlib.figure
        case, window, gt, frames, collision = fixture()
        actors, identities, predicted, truth, metadata, ttc = plot.prepare_case(
            case, window, gt, {}, collision, frames)
        c = deepcopy(actors["B"])
        c.actor_id, c.source_track_ids = "C", [3]
        actors["C"], identities["C"] = c, {3}
        for frame in frames.values():
            frame[3] = plot.boxes.WorldBox(3, (99., 99.), 0., (1., 0.), 2., 1., 1.)
        calls = []
        def short_pose(original, name):
            def pose(t):
                self.assertLessEqual(t, 1., f"{name} evaluated past coverage")
                calls.append((name, t))
                return original(t)
            return pose
        short_a = short_pose(truth["paths"]["A"][0][1], "short A")
        short_c = short_pose(truth["paths"]["B"][0][1], "short C")
        truth["paths"]["A"].append((None, short_a, 1.))
        truth["xy_paths"]["A"].append((None, short_a, 1.))
        truth["paths"]["C"] = [(None, short_c, 1.)]
        truth["xy_paths"]["C"] = [(None, short_c, 1.)]
        truth["samples"][("A", "C")] = [(.1, 9.), (.8, 0.), (1., 1.)]
        truth["events"][("A", "C")] = ("NEW_CONTACT", 1.5)
        truth["chosen"][("A", "C")] = (short_a, short_c)
        predicted["paths"]["C"] = predicted["paths"]["B"]
        predicted["xy_paths"]["C"] = predicted["xy_paths"]["B"]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(matplotlib.figure.Figure, "savefig"):
            plot.render_case("case_001", actors, identities, predicted, truth, metadata, ttc,
                             Path(directory))
        self.assertIn(("short A", .8), calls)
        self.assertNotIn(("short A", 4.), calls)
        self.assertNotIn(("short A", 1.5), calls)

    def test_pair_metadata_reports_each_pairs_own_minima_and_events(self):
        case, window, gt, frames, collision = fixture()
        _, _, predicted, truth, metadata, _ = plot.prepare_case(case, window, gt, {}, collision, frames)
        unchanged = metadata.copy()
        predicted["samples"][("A", "C")] = [(.2, 6.), (2., 3.)]
        truth["samples"][("A", "C")] = [(.1, 9.), (2., 8.), (3., 4.), (4., 5.)]
        predicted["events"][("A", "C")] = ("GEOMETRY_UNCERTAIN", None)
        truth["events"][("A", "C")] = ("NONE", None)
        records = plot.pair_metadata("case_002", predicted, truth)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(set(record) == set(plot.PAIR_FIELDS) for record in records))
        a, b = records
        self.assertEqual(json.loads(a["actor_pair"]), ["A", "B"])
        self.assertEqual(a["case_id"], "case_002")
        self.assertEqual((a["predicted_min_clearance_m"], a["predicted_min_clearance_time_s"]), (0., 1.6))
        self.assertEqual((a["gt_min_clearance_0_2_m"], a["gt_min_clearance_0_2_time_s"]), (4., 2.))
        self.assertEqual((a["gt_min_clearance_2_5_m"], a["gt_min_clearance_2_5_time_s"]), (0., 4.))
        self.assertEqual((a["predicted_contact_event"], a["predicted_contact_time_s"]), ("NEW_CONTACT", 1.6))
        self.assertEqual((a["gt_contact_event"], a["gt_contact_time_s"]), ("NEW_CONTACT", 4.))
        self.assertEqual(json.loads(b["actor_pair"]), ["A", "C"])
        self.assertEqual(b["predicted_min_clearance_m"], 3.)
        self.assertEqual((b["gt_min_clearance_2_5_m"], b["gt_min_clearance_2_5_time_s"]), (4., 3.))
        self.assertEqual(b["predicted_contact_event"], "GEOMETRY_UNCERTAIN")
        self.assertIsNone(b["predicted_contact_time_s"])
        self.assertEqual(metadata, unchanged)

    def test_inset_selects_lowest_clearance_pair_without_changing_main_limits(self):
        import matplotlib.pyplot as plt
        from copy import deepcopy
        case, window, gt, frames, collision = fixture()
        actors, identities, predicted, _, _, _ = plot.prepare_case(case, window, gt, {}, collision, frames)
        actors["C"] = deepcopy(actors["B"])
        actors["C"].actor_id = "C"
        identities["C"] = {2}
        predicted["paths"]["C"] = predicted["paths"]["B"]
        predicted["xy_paths"]["C"] = predicted["xy_paths"]["B"]
        predicted["chosen"][("A", "C")] = predicted["chosen"][("A", "B")]
        predicted["samples"][("A", "B")] = [(1., 3.)]
        predicted["samples"][("A", "C")] = [(1.6, 0.)]
        fig, ax = plt.subplots()
        try:
            ax.set(xlim=(-50., 50.), ylim=(-30., 30.))
            before = ax.get_xlim(), ax.get_ylim()
            plot.interaction_inset(ax, actors, predicted, dict(A="red", B="blue", C="green"), "--")
            self.assertEqual((ax.get_xlim(), ax.get_ylim()), before)
            inset = ax.child_axes[0]
            self.assertEqual(len(inset.patches), 2)
            self.assertTrue(any("A / C" in text.get_text() for text in inset.texts))
            self.assertLess(inset.get_xlim()[1] - inset.get_xlim()[0], 100.)
            self.assertEqual(inset.get_aspect(), 1.)
        finally:
            plt.close(fig)

    def test_discovery_ignores_other_splits_but_rejects_duplicate_val_ids(self):
        validation = [SimpleNamespace(scenario_id=f"type/scene_{i:03d}", split="val")
                      for i in reversed(range(104))]
        other_splits = [SimpleNamespace(scenario_id=validation[0].scenario_id, split=split)
                        for split in ("train", "test")]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(plot, "load_cases", return_value=[
                    dict(scenario_id="missing", window_id="missing:0:5.000")]), \
                patch.object(plot, "DeepAccidentRunner") as runner:
            argv = ["--root", directory, "--carla-maps", directory,
                    "--out", str(Path(directory) / "plots")]
            runner.return_value.list_scenarios.return_value = other_splits + validation
            # Discovery succeeds despite duplicate IDs outside val; the
            # deliberately absent requested case stops before any replay.
            with self.assertRaisesRegex(KeyError, "scenario not found: missing"):
                plot.main(argv)
            duplicates = validation[:-1] + [validation[0]]
            runner.return_value.list_scenarios.return_value = other_splits + duplicates
            with self.assertRaisesRegex(ValueError, "duplicate validation scenario_id"):
                plot.main(argv)
            runner.return_value.list_scenarios.return_value = other_splits + validation[:-1]
            with self.assertRaisesRegex(ValueError, "requires 104 scenarios; found 103"):
                plot.main(argv)

    def test_pdf_export_preserves_page_order_resolution_and_directory_separation(self):
        from PIL import Image
        import matplotlib
        colors = {"blind": [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
                  "diagnostic": [(0, 0, 0), (255, 255, 255), (128, 128, 128)]}
        sizes = [(16, 8), (20, 10), (24, 12)]
        with tempfile.TemporaryDirectory() as directory, matplotlib.rc_context({"pdf.compression": 0}):
            out = Path(directory)
            for kind in ("blind", "diagnostic"):
                (out / kind).mkdir()
                for i, (size, color) in enumerate(zip(sizes, colors[kind]), 1):
                    Image.new("RGB", size, color).save(out / kind / f"case_{i:03d}.png")
            with patch.object(plot, "render_case", side_effect=AssertionError("must not rerender trajectories")):
                plot.export_case_pdfs(out, ["case_003", "case_001", "case_002"])
            for kind in ("blind", "diagnostic"):
                data = (out / f"{kind}_cases.pdf").read_bytes()
                self.assertEqual(len(re.findall(rb"/Type /Page\b", data)), 3)
                pages = re.findall(rb"/MediaBox \[ 0 0 ([\d.]+) ([\d.]+) \]", data)
                self.assertEqual([(float(w), float(h)) for w, h in pages], sizes)
                images = [(header, pixels) for header, pixels in re.findall(
                    rb"<<(.*?)>>\s*stream\n(.*?)\nendstream", data, re.S)
                          if b"/Subtype /Image" in header]
                self.assertEqual(len(images), 3)
                for (header, pixels), (width, height), color in zip(images, sizes, colors[kind]):
                    self.assertIn(f"/Width {width}".encode(), header)
                    self.assertIn(f"/Height {height}".encode(), header)
                    self.assertEqual(pixels, bytes(color) * width * height)
                self.assertNotIn(b"/DCTDecode", data)

    def test_pdf_export_fails_on_missing_case_png(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "blind").mkdir()
            (out / "diagnostic").mkdir()
            Image.new("RGB", (16, 8)).save(out / "blind" / "case_001.png")
            with self.assertRaisesRegex(FileNotFoundError, "diagnostic/case_001.png"):
                plot.export_case_pdfs(out, ["case_001"])
            self.assertFalse((out / "blind_cases.pdf").exists())
            self.assertFalse((out / "diagnostic_cases.pdf").exists())

    def test_main_writes_metadata_mapping_and_refuses_overwrites(self):
        case, window, gt, frames, collision = fixture()
        scenario = SimpleNamespace(scenario_id="type/scene", scenario="scene",
                                   scenario_type="type", town="Town01", split="val",
                                   agents={"agent": SimpleNamespace(frames=list(frames))})
        with tempfile.TemporaryDirectory() as directory:
            scenario.root = str(Path(directory) / "val")
            validation = [scenario] + [SimpleNamespace(scenario_id=f"unused/{i}", split="val")
                                       for i in range(103)]
            path, out = Path(directory) / "cases.csv", Path(directory) / "plots"
            row = dict(window_id=case["window_id"], scenario_id=case["scenario_id"],
                       gt_bucket="4", analysis_group=case["comparison_group"],
                       crn4_selected_pair="('A', 'B')", v4_selected_pairs='[["A","B"]]',
                       crn4_risk_2s="0.4", crn4_threshold_2s="0.9", crn4_correct_pair_hit="True")
            frozen = Path(directory) / "v4.jsonl"
            frozen.write_text(json.dumps(dict(
                window_id=case["window_id"], selected_pairs=[["A", "B"]],
                gt_pair_covered=True, baseline_predicted=False, correct_pair_hit=True)) + "\n")
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            argv = ["--cases", str(path), "--root", directory, "--carla-maps", directory,
                    "--out", str(out), "--limit", "1", "--blind-seed", "17", "--no-pdf"]
            with patch.object(plot, "V4_LEDGER", frozen), \
                    patch.object(plot, "load_cases", wraps=plot.load_cases) as load, \
                    patch.object(plot, "DeepAccidentRunner") as runner, \
                    patch.object(plot, "replay_windows", return_value=(
                        {window.label: window}, {gt.label: gt}, {}, None)) as replay, \
                    patch.object(plot.boxes, "merge_frame_boxes", side_effect=lambda sc, f: (frames[f], {})), \
                    patch.object(plot, "estimate_collision", return_value=collision), \
                    patch.object(plot, "render_case") as render, patch("builtins.print"):
                runner.return_value.list_scenarios.return_value = validation
                plot.main(argv)
                load.assert_called_once_with(path, 1, 17, None, None)
                self.assertEqual(runner.call_args.args[0], directory)
                self.assertEqual(replay.call_args.args[0], scenario.root)
                render.assert_called_once()
                self.assertFalse((out / "blind_cases.pdf").exists())
                self.assertFalse((out / "diagnostic_cases.pdf").exists())
                saved = plot.read_csv(out / "case_plot_metadata.csv")
                self.assertEqual(list(saved[0]), plot.FIELDS)
                self.assertEqual(saved[0]["case_id"], "case_001")
                self.assertEqual(saved[0]["comparison_group"], case["comparison_group"])
                self.assertEqual(json.loads(saved[0]["gt_pair"]), [1, 2])
                self.assertEqual(saved[0]["gt_first_contact_time_s"], "4.0")
                self.assertNotIn("window_id", saved[0])
                self.assertEqual(saved[0]["v4_gt_pair_candidate_covered"], "True")
                self.assertEqual(saved[0]["v4_baseline_predicted"], "False")
                self.assertEqual(saved[0]["v4_correct_gt_pair_hit"], "True")
                self.assertEqual(saved[0]["crn_correct_gt_pair_hit"], "True")
                self.assertEqual(plot.read_csv(out / "case_id_mapping.csv"), [
                    dict(case_id="case_001", window_id=case["window_id"])])
                pairs = plot.read_csv(out / "case_pair_metadata.csv")
                self.assertEqual(len(pairs), 1)
                self.assertEqual(list(pairs[0]), plot.PAIR_FIELDS)
                self.assertEqual(pairs[0]["case_id"], "case_001")
                self.assertEqual(json.loads(pairs[0]["actor_pair"]), ["A", "B"])
                self.assertEqual(pairs[0]["predicted_contact_time_s"], "1.6")
                with self.assertRaises(FileExistsError):
                    plot.main(argv)
                grouped_argv = argv.copy()
                grouped_argv[grouped_argv.index("--limit")] = "--per-group-limit"
                with self.assertRaises(FileExistsError):
                    plot.main(grouped_argv)
                load.assert_called_with(path, None, 17, 1, None)
                grouped_argv += ["--groups", case["comparison_group"]]
                with self.assertRaises(FileExistsError):
                    plot.main(grouped_argv)
                load.assert_called_with(path, None, 17, 1, [case["comparison_group"]])
                with patch.object(plot, "replay_windows", return_value=({}, {}, {}, None)):
                    with self.assertRaisesRegex(RuntimeError, "cannot be reconstructed"):
                        missing_argv = argv.copy()
                        missing_argv[missing_argv.index("--out") + 1] = str(Path(directory) / "missing")
                        plot.main(missing_argv)
                from PIL import Image
                pdf_out = Path(directory) / "pdf_plots"
                pdf_argv = [arg for arg in argv if arg != "--no-pdf"]
                pdf_argv[pdf_argv.index("--out") + 1] = str(pdf_out)
                def write_pngs(case_id, *args, **kwargs):
                    destination = args[-1]
                    for kind in ("blind", "diagnostic"):
                        Image.new("RGB", (16, 8)).save(destination / kind / f"{case_id}.png")
                with patch.object(plot, "render_case", side_effect=write_pngs):
                    plot.main(pdf_argv)
                for kind in ("blind", "diagnostic"):
                    self.assertEqual(len(re.findall(rb"/Type /Page\b", (
                        pdf_out / f"{kind}_cases.pdf").read_bytes())), 1)


if __name__ == "__main__":
    unittest.main()
