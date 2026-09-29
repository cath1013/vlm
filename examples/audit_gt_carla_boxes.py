"""Audit raw DeepAccident CARLA 3D label boxes as a collision oracle.

This deliberately bypasses every prediction, re-ranking, and LLM path.  It
uses only per-instance raw label dimensions and poses, while using a completed
GT run solely for its selected scenarios and bucket definitions.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from traffic_llm.deepaccident import (  # noqa: E402
    SELF_ID, load_calib, parse_label_file, scan_scenarios,
)


FRAME_RATE_HZ = 10.0
DUPLICATE_POSITION_TOLERANCE_M = 0.25


@dataclass(frozen=True)
class WorldBox:
    carla_id: int
    center: tuple[float, float]
    center_z: float
    heading: tuple[float, float]
    length: float
    width: float
    height: float
    # These are raw-label attributes retained alongside the geometry for the
    # optional pipeline-matched eligibility filters.  Defaults keep WorldBox a
    # small, geometry-only value object for the unrestricted audit mode.
    speed_mps: float = 0.0
    camera_visible: bool = False
    is_self: bool = False
    cls: str = ""
    raw_yaw_rad: float = float("nan")
    n_lidar_pts: int = 0

    def polygon(self) -> list[tuple[float, float]]:
        return oriented_footprint(self.center, self.heading, self.length, self.width)


def oriented_footprint(center, heading, length: float, width: float):
    """Return a labelled box's own oriented world-ENU footprint."""
    hx, hy = heading
    norm = math.hypot(hx, hy)
    if norm <= 1e-12:
        raise ValueError("box heading has zero length")
    hx, hy = hx / norm, hy / norm
    sx, sy = -hy, hx
    half_l, half_w = length / 2.0, width / 2.0
    return [
        (center[0] + a * hx + b * sx, center[1] + a * hy + b * sy)
        for a, b in ((-half_l, -half_w), (-half_l, half_w),
                     (half_l, half_w), (half_l, -half_w))
    ]


def _cross(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p) -> bool:
    return (abs(_cross(a, b, p)) <= 1e-9
            and min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9
            and min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9)


def _segments_intersect(a, b, c, d) -> bool:
    ab_c, ab_d = _cross(a, b, c), _cross(a, b, d)
    cd_a, cd_b = _cross(c, d, a), _cross(c, d, b)
    if ((ab_c > 1e-9 and ab_d < -1e-9) or (ab_c < -1e-9 and ab_d > 1e-9)) and (
        (cd_a > 1e-9 and cd_b < -1e-9) or (cd_a < -1e-9 and cd_b > 1e-9)
    ):
        return True
    return (_on_segment(a, b, c) or _on_segment(a, b, d)
            or _on_segment(c, d, a) or _on_segment(c, d, b))


def _point_segment_distance(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx * dx + dy * dy
    if denom <= 1e-18:
        return math.dist(p, a)
    u = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / denom))
    return math.dist(p, (a[0] + u * dx, a[1] + u * dy))


def _inside_convex_polygon(p, polygon) -> bool:
    crosses = [_cross(a, b, p) for a, b in zip(polygon, polygon[1:] + polygon[:1])]
    return all(v >= -1e-9 for v in crosses) or all(v <= 1e-9 for v in crosses)


def polygon_clearance(a, b) -> float:
    """Minimum 2D clearance of two convex footprints; zero includes touching."""
    edges_a = list(zip(a, a[1:] + a[:1]))
    edges_b = list(zip(b, b[1:] + b[:1]))
    if any(_segments_intersect(*ea, *eb) for ea in edges_a for eb in edges_b):
        return 0.0
    if _inside_convex_polygon(a[0], b) or _inside_convex_polygon(b[0], a):
        return 0.0
    return min(
        min(_point_segment_distance(p, *edge) for p in a for edge in edges_b),
        min(_point_segment_distance(p, *edge) for p in b for edge in edges_a),
    )


def boxes_contact(a: WorldBox, b: WorldBox, margin_m: float = 0.0) -> bool:
    horizontal_contact = polygon_clearance(a.polygon(), b.polygon()) <= margin_m + 1e-9
    a_bottom, a_top = a.center_z - a.height / 2.0, a.center_z + a.height / 2.0
    b_bottom, b_top = b.center_z - b.height / 2.0, b.center_z + b.height / 2.0
    return horizontal_contact and max(a_bottom, b_bottom) <= min(a_top, b_top)


def raw_box_to_world(obj, lidar_to_world: np.ndarray, carla_id: int) -> WorldBox:
    """Transform a raw lidar-label box to ENU without class-size fallbacks."""
    p = lidar_to_world @ np.array([obj.x, obj.y, obj.z, 1.0])
    direction = lidar_to_world[:3, :3] @ np.array([
        math.cos(obj.yaw), math.sin(obj.yaw), 0.0,
    ])
    # CARLA x/y is east/south; the audit's 2D world frame is east/north.
    return WorldBox(
        carla_id=carla_id,
        center=(float(p[0]), -float(p[1])),
        center_z=float(p[2]),
        heading=(float(direction[0]), -float(direction[1])),
        length=float(obj.length),
        width=float(obj.width),
        height=float(obj.height),
    )


def resolve_carla_id(obj, scenario, agent: str):
    if obj.obj_id == -1:
        return None
    return scenario.meta.agent_id_of(agent) if obj.obj_id == SELF_ID else obj.obj_id


def merge_frame_boxes(scenario, frame: int):
    """Merge all observer measurements into one CARLA-ID keyed frame."""
    boxes: dict[int, WorldBox] = {}
    diagnostics = Counter()
    self_ids = {
        scenario.meta.agent_id_of(agent)
        for agent in scenario.agents
        if scenario.meta.agent_id_of(agent) is not None
    }
    for agent, series in scenario.agents.items():
        if frame not in series.frames:
            continue
        calib = load_calib(series.calib_paths[frame])
        lidar_to_world = np.asarray(calib["ego_to_world"], dtype=float) @ np.asarray(
            calib["lidar_to_ego"], dtype=float
        )
        for obj in parse_label_file(series.label_paths[frame]).objects:
            diagnostics["objects_boxes_processed"] += 1
            carla_id = resolve_carla_id(obj, scenario, agent)
            if carla_id is None:
                continue
            box = replace(
                raw_box_to_world(obj, lidar_to_world, carla_id),
                speed_mps=math.hypot(getattr(obj, "vx", 0.0), getattr(obj, "vy", 0.0)),
                camera_visible=bool(getattr(obj, "camera_visible", False)),
                is_self=(obj.obj_id == SELF_ID or carla_id in self_ids),
                cls=str(getattr(obj, "cls", "")),
                raw_yaw_rad=float(getattr(obj, "yaw", float("nan"))),
                n_lidar_pts=int(getattr(obj, "n_lidar_pts", 0)),
            )
            old = boxes.get(carla_id)
            if old is None:
                boxes[carla_id] = box
                continue
            # Geometry continues to use the first observation, as it did in
            # the original raw-box audit.  Visibility is an any-observer fact;
            # a self observation makes this CARLA actor eligible as ego.
            boxes[carla_id] = replace(
                old,
                camera_visible=old.camera_visible or box.camera_visible,
                is_self=old.is_self or box.is_self,
            )
            diagnostics["duplicate_carla_id_observations_merged"] += 1
            if math.dist(old.center, box.center) > DUPLICATE_POSITION_TOLERANCE_M:
                diagnostics["duplicate_position_inconsistencies"] += 1
    return boxes, diagnostics


def _box_diagnostic(box: WorldBox) -> dict:
    """Raw-label attributes and the already-transformed audit pose."""
    return {
        "carla_id": box.carla_id,
        "class": box.cls,
        "center_xyz_m": [box.center[0], box.center[1], box.center_z],
        "length_width_height_m": [box.length, box.width, box.height],
        "raw_yaw_rad": box.raw_yaw_rad,
        "world_yaw_rad": math.atan2(box.heading[1], box.heading[0]),
        "speed_mps": box.speed_mps,
        "camera_visible": box.camera_visible,
        "n_lidar_pts": box.n_lidar_pts,
    }


def _pair_diagnostic(a: WorldBox, b: WorldBox) -> dict:
    gap = polygon_clearance(a.polygon(), b.polygon())
    a_z = (a.center_z - a.height / 2.0, a.center_z + a.height / 2.0)
    b_z = (b.center_z - b.height / 2.0, b.center_z + b.height / 2.0)
    vertical_overlap = max(a_z[0], b_z[0]) <= min(a_z[1], b_z[1])
    return {
        "actors": [_box_diagnostic(a), _box_diagnostic(b)],
        "center_distance_xy_m": math.dist(a.center, b.center),
        # polygon_clearance is the audit's exact BEV routine: zero means the
        # rectangles touch or overlap; it does not calculate penetration depth.
        "minimum_bev_gap_m": gap,
        "bev_rectangles_intersect": gap <= 1e-9,
        "vertical_z_intervals_m": [list(a_z), list(b_z)],
        "vertical_z_intervals_overlap": vertical_overlap,
        "audit_3d_contact": boxes_contact(a, b, margin_m=0.0),
    }


def _draw_box(ax, box: WorldBox, *, color: str, emphasized: bool) -> None:
    from matplotlib.patches import Polygon  # lazy: audit itself has no plot dependency

    polygon = box.polygon()
    ax.add_patch(Polygon(
        polygon, closed=True, fill=emphasized, facecolor=color if emphasized else "none",
        edgecolor=color, alpha=0.18 if emphasized else 0.7,
        linewidth=2.6 if emphasized else 1.0,
    ))
    if emphasized:
        ax.text(
            box.center[0], box.center[1], f"{box.carla_id} ({box.cls})", color=color,
            ha="center", va="center", fontsize=9, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1},
        )
    else:
        ax.text(box.center[0], box.center[1], str(box.carla_id), color="0.25",
                ha="center", va="center", fontsize=7)


def _render_pair_frame(frame: int, boxes: dict[int, WorldBox], pair, radius_m: float, out: Path) -> dict:
    """Render one diagnostic PNG from the exact boxes already used by the audit."""
    import matplotlib.pyplot as plt  # optional diagnostic dependency

    a, b = (boxes[carla_id] for carla_id in pair)
    diagnostic = _pair_diagnostic(a, b)
    midpoint = ((a.center[0] + b.center[0]) / 2.0, (a.center[1] + b.center[1]) / 2.0)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {a.carla_id: "tab:red", b.carla_id: "tab:blue"}
    for box in boxes.values():
        if math.dist(box.center, midpoint) <= radius_m:
            _draw_box(ax, box, color=colors.get(box.carla_id, "0.45"),
                      emphasized=box.carla_id in colors)
    ax.set_xlim(midpoint[0] - radius_m, midpoint[0] + radius_m)
    ax.set_ylim(midpoint[1] - radius_m, midpoint[1] + radius_m)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"Raw CARLA boxes — frame {frame:04d}")
    # One-metre scale reference in the lower-left of the requested view.
    sx, sy = midpoint[0] - radius_m + 1.0, midpoint[1] - radius_m + 1.0
    ax.plot((sx, sx + 1.0), (sy, sy), color="black", linewidth=3)
    ax.text(sx + 0.5, sy + radius_m * 0.05, "1 m", ha="center", va="bottom", fontsize=8)
    details = []
    for box in (a, b):
        d = _box_diagnostic(box)
        details.append(
            f"{d['carla_id']} ({d['class']})\n"
            f"xyz=({d['center_xyz_m'][0]:.2f}, {d['center_xyz_m'][1]:.2f}, {d['center_xyz_m'][2]:.2f}) m\n"
            f"LWH=({d['length_width_height_m'][0]:.2f}, {d['length_width_height_m'][1]:.2f}, {d['length_width_height_m'][2]:.2f}) m\n"
            f"yaw raw/world={d['raw_yaw_rad']:.3f}/{d['world_yaw_rad']:.3f} rad; v={d['speed_mps']:.2f} m/s\n"
            f"camera_visible={d['camera_visible']}; lidar_pts={d['n_lidar_pts']}"
        )
    ax.text(1.02, 1.0, "\n\n".join(details), transform=ax.transAxes, va="top",
            fontsize=8, family="monospace")
    fig.savefig(out / f"frame_{frame:04d}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return diagnostic


def plot_pair_diagnostics(scenarios, pair, frames, plot_dir: Path, radius_m: float) -> None:
    """Find one raw scenario containing a pair and emit independent plot artifacts."""
    if radius_m <= 0:
        raise ValueError("--plot-radius-m must be positive")
    pair = tuple(map(int, pair))
    if pair[0] == pair[1]:
        raise ValueError("--plot-pair requires two distinct CARLA IDs")
    matching = []
    for scenario in scenarios.values():
        requested = {frame: merge_frame_boxes(scenario, frame)[0] for frame in frames
                     if any(frame in series.frames for series in scenario.agents.values())}
        if all(set(pair) <= set(boxes) for boxes in requested.values()) and len(requested) == len(frames):
            matching.append((scenario, requested))
    if not matching:
        raise KeyError(f"no raw scenario contains pair {pair} at every requested frame {list(frames)}")
    if len(matching) != 1:
        names = [f"{s.scenario_type}/{s.scenario}" for s, _ in matching]
        raise KeyError(f"pair {pair} is ambiguous across raw scenarios: {names}")
    scenario, requested = matching[0]
    plot_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "scenario": {"split": scenario.split, "scenario_type": scenario.scenario_type,
                     "scenario": scenario.scenario},
        "pair": list(pair), "plot_radius_m": radius_m, "frames": {},
    }
    for frame in frames:
        output["frames"][str(frame)] = _render_pair_frame(
            frame, requested[frame], pair, radius_m, plot_dir
        )
    (plot_dir / "pair_diagnostics.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def in_interval(t: float, start: float, end: float) -> bool:
    """Ground-truth buckets are exactly open on the left and closed on the right."""
    return t > start + 1e-9 and t <= end + 1e-9


def target_pair_hit(contacting_pairs, involved_carla_ids) -> bool:
    return (len(involved_carla_ids) == 2
            and frozenset(map(int, involved_carla_ids)) in contacting_pairs)


def scenario_key(scenario) -> tuple[str, str, str]:
    return scenario.split, scenario.scenario_type, scenario.scenario


def scenario_row_key(row: dict) -> tuple[str, str, str]:
    return row["dataset_split"], row["scenario_type"], row["scenario"]


def _scenario_buckets(gt_dir: Path):
    rows = []
    manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))
    for window in manifest.get("windows", []):
        gt = json.loads((gt_dir / window["ground_truth"]).read_text(encoding="utf-8"))
        for expected in gt.get("expected", []):
            if expected.get("scorable"):
                rows.append((gt["window"]["label"], float(gt["window"]["t_end_s"]), expected))
    return rows


def _frame_at_time(frame_boxes, t_end: float):
    """Return raw boxes at the exact 10 Hz observation cutoff, if present."""
    frame = int(round(t_end * FRAME_RATE_HZ)) + 1
    if math.isclose((frame - 1) / FRAME_RATE_HZ, t_end, abs_tol=1e-9):
        return frame_boxes.get(frame, {})
    return {}


def _pair_visible_at_frame(pair, boxes) -> bool:
    return all(boxes[carla_id].is_self or boxes[carla_id].camera_visible for carla_id in pair)


def audit_scenario(
    scenario, scenario_row: dict, gt_run: str, margin_m: float,
    require_visible: bool = False, exclude_static_pairs: bool = False,
    exclude_touching_now: bool = False,
):
    gt_dir = Path(gt_run) / scenario_row["directory"]
    buckets = _scenario_buckets(gt_dir)
    contacts_by_frame = {}
    boxes_by_frame = {}
    diagnostics = Counter()
    frames = sorted({frame for series in scenario.agents.values() for frame in series.frames})
    for frame in frames:
        boxes, counts = merge_frame_boxes(scenario, frame)
        boxes_by_frame[frame] = boxes
        diagnostics.update(counts)
        diagnostics["raw_frames_processed"] += 1
        pairs = {
            frozenset((a.carla_id, b.carla_id))
            for a, b in itertools.combinations(boxes.values(), 2)
            if boxes_contact(a, b, margin_m)
        }
        contacts_by_frame[frame] = pairs

    records = []
    # A swept-path candidate is removed once per prediction window, even if it
    # would have contacted on several 10 Hz frames or future buckets.
    removed_pairs = {
        "pairs_removed_not_visible": set(),
        "pairs_removed_static_static": set(),
        "pairs_removed_touching_at_observation": set(),
    }
    for window_label, t_end, expected in buckets:
        start, end = float(expected["interval_start_s"]), float(expected["interval_end_s"])
        observation_boxes = _frame_at_time(boxes_by_frame, t_end)
        touching_now = {
            frozenset((a.carla_id, b.carla_id))
            for a, b in itertools.combinations(observation_boxes.values(), 2)
            if boxes_contact(a, b, margin_m)
        } if exclude_touching_now else set()
        frame_rows = []
        pairs = set()
        available = False
        for frame, frame_pairs in contacts_by_frame.items():
            t = (frame - 1) / FRAME_RATE_HZ
            if in_interval(t, start, end):
                available = True
                eligible_pairs = set()
                for pair in frame_pairs:
                    if require_visible and not _pair_visible_at_frame(pair, boxes_by_frame[frame]):
                        key = (window_label, pair)
                        if key not in removed_pairs["pairs_removed_not_visible"]:
                            diagnostics["pairs_removed_not_visible"] += 1
                            removed_pairs["pairs_removed_not_visible"].add(key)
                        continue
                    observed_pair = [observation_boxes.get(carla_id) for carla_id in pair]
                    if (exclude_static_pairs and all(observed_pair)
                            and all(abs(box.speed_mps) <= 0.5 for box in observed_pair)):
                        key = (window_label, pair)
                        if key not in removed_pairs["pairs_removed_static_static"]:
                            diagnostics["pairs_removed_static_static"] += 1
                            removed_pairs["pairs_removed_static_static"].add(key)
                        continue
                    if exclude_touching_now and pair in touching_now:
                        key = (window_label, pair)
                        if key not in removed_pairs["pairs_removed_touching_at_observation"]:
                            diagnostics["pairs_removed_touching_at_observation"] += 1
                            removed_pairs["pairs_removed_touching_at_observation"].add(key)
                        continue
                    eligible_pairs.add(pair)
                if eligible_pairs:
                    encoded = [sorted(pair) for pair in sorted(eligible_pairs, key=lambda p: sorted(p))]
                    frame_rows.append({"frame": frame, "time_s": t, "pairs": encoded})
                    pairs.update(eligible_pairs)
        if not available:
            diagnostics["buckets_with_no_raw_frame_available"] += 1
        gt_positive = bool(expected["accident_expected"])
        records.append({
            "scenario": scenario_row["scenario"],
            "scenario_type": scenario_row["scenario_type"],
            "window_label": window_label,
            "k": int(expected["k"]),
            "interval_start_s": start,
            "interval_end_s": end,
            "gt_positive": gt_positive,
            "involved_carla_ids": list(expected.get("involved_carla_ids") or []),
            "pred_any_contact": bool(pairs),
            "target_pair_hit": target_pair_hit(pairs, expected.get("involved_carla_ids") or []),
            "contacting_pairs": [sorted(pair) for pair in sorted(pairs, key=lambda p: sorted(p))],
            "frames_with_contact": frame_rows,
        })
    return records, diagnostics


def summarize(records, diagnostics, n_scenarios: int, margin_m: float):
    counts = Counter()
    fp_windows, fp_scenarios = set(), set()
    for row in records:
        if row["gt_positive"]:
            counts["positive_GT_buckets"] += 1
            counts["GT_actor_pair_hits"] += int(row["target_pair_hit"])
        if row["pred_any_contact"] and row["gt_positive"]:
            counts["TP_any_contact"] += 1
        elif row["pred_any_contact"]:
            counts["FP_contact"] += 1
            counts["FP_contact_pair_rows"] += len(row["contacting_pairs"])
            fp_windows.add((row["scenario"], row["window_label"]))
            fp_scenarios.add(row["scenario"])
        elif row["gt_positive"]:
            counts["FN_no_contact"] += 1
        else:
            counts["TN_no_contact"] += 1
    tp, fp = counts["TP_any_contact"], counts["FP_contact"]
    positives = counts["positive_GT_buckets"]
    return {
        "margin_m": margin_m,
        "n_buckets": len(records),
        "TP_any_contact": tp,
        "FP_contact": fp,
        "TN_no_contact": counts["TN_no_contact"],
        "FN_no_contact": counts["FN_no_contact"],
        "any_contact_precision": None if not tp + fp else tp / (tp + fp),
        "any_contact_recall": None if not positives else tp / positives,
        "positive_GT_buckets": positives,
        "GT_actor_pair_hits": counts["GT_actor_pair_hits"],
        "GT_actor_pair_recall": None if not positives else counts["GT_actor_pair_hits"] / positives,
        "FP_contact_pair_rows": counts["FP_contact_pair_rows"],
        "windows_with_FP_contact": len(fp_windows),
        "scenarios_with_FP_contact": len(fp_scenarios),
        "scenarios_processed": n_scenarios,
        "raw_frames_processed": diagnostics["raw_frames_processed"],
        "objects_boxes_processed": diagnostics["objects_boxes_processed"],
        "duplicate_carla_id_observations_merged": diagnostics["duplicate_carla_id_observations_merged"],
        "duplicate_position_inconsistencies": diagnostics["duplicate_position_inconsistencies"],
        "buckets_with_no_raw_frame_available": diagnostics["buckets_with_no_raw_frame_available"],
        "pairs_removed_not_visible": diagnostics["pairs_removed_not_visible"],
        "pairs_removed_static_static": diagnostics["pairs_removed_static_static"],
        "pairs_removed_touching_at_observation": diagnostics["pairs_removed_touching_at_observation"],
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="DeepAccident root")
    ap.add_argument("--gt-run", required=True, help="existing GT validation run")
    ap.add_argument("--margin-m", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--require-visible", action="store_true",
                    help="require a non-self actor to be camera-visible in each raw frame")
    ap.add_argument("--exclude-static-pairs", action="store_true",
                    help="exclude pairs whose raw cutoff speeds are both <= 0.5 m/s")
    ap.add_argument("--exclude-touching-now", action="store_true",
                    help="exclude pairs whose raw 3D boxes touch at a window cutoff")
    ap.add_argument("--plot-pair", nargs=2, type=int, metavar=("CARLA_ID_A", "CARLA_ID_B"),
                    help="write raw-box diagnostic plots for this CARLA actor pair")
    ap.add_argument("--plot-frames", nargs="+", type=int, metavar="FRAME",
                    help="raw 10 Hz frame numbers to plot")
    ap.add_argument("--plot-dir", type=Path, help="directory for diagnostic PNGs and JSON")
    ap.add_argument("--plot-radius-m", type=float, default=12.0,
                    help="BEV radius around the pair midpoint (default: 12 m)")
    args = ap.parse_args(argv)
    plot_options = (args.plot_pair, args.plot_frames, args.plot_dir)
    if any(option is not None for option in plot_options) and not all(plot_options):
        ap.error("--plot-pair, --plot-frames, and --plot-dir must be supplied together")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    gt_run = Path(args.gt_run)
    cohort = json.loads((gt_run / "batch_manifest.json").read_text(encoding="utf-8"))
    scenario_rows = cohort["scenarios"]
    scenarios = {scenario_key(s): s for s in scan_scenarios(args.root)}
    missing = [
        f"{row['scenario_type']}/{row['scenario']}"
        for row in scenario_rows
        if scenario_row_key(row) not in scenarios
    ]
    if missing:
        raise KeyError(f"raw scenarios not found: {missing[:10]}")
    if args.plot_pair:
        cohort_scenarios = {
            scenario_row_key(row): scenarios[scenario_row_key(row)] for row in scenario_rows
        }
        plot_pair_diagnostics(
            cohort_scenarios, args.plot_pair, args.plot_frames, args.plot_dir,
            args.plot_radius_m,
        )
    all_records, diagnostics = [], Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                audit_scenario,
                scenarios[scenario_row_key(row)],
                row, str(gt_run), args.margin_m, args.require_visible,
                args.exclude_static_pairs, args.exclude_touching_now,
            ): row
            for row in scenario_rows
        }
        for done, future in enumerate(as_completed(futures), 1):
            records, counts = future.result()
            all_records.extend(records)
            diagnostics.update(counts)
            print(f"[{done}/{len(scenario_rows)}] complete {futures[future]['scenario']}", flush=True)
    all_records.sort(key=lambda r: (r["scenario_type"], r["scenario"], r["window_label"], r["k"]))
    summary = summarize(all_records, diagnostics, len(scenario_rows), args.margin_m)
    if summary["n_buckets"] != 624 or summary["positive_GT_buckets"] != 69:
        raise RuntimeError(
            "GT bucket set does not match the required oracle evaluation: "
            f"n_buckets={summary['n_buckets']}, positive_GT_buckets={summary['positive_GT_buckets']}"
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_records),
        encoding="utf-8",
    )
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
