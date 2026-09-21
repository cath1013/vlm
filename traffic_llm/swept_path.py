"""Deterministic pair geometry over trajectory-predictor paths.

The language model should not have to estimate simultaneous contact by visually
comparing dozens of waypoint arrays.  This module performs that arithmetic only:
it interpolates each supplied path, places a class-sized oriented footprint at
each sample, and reports the closest future clearance for every actor pair.

No constant-velocity fallback, TTC heuristic, road relation, or ground-truth
collision information is used here.  A clearance of zero means the two predicted
footprints touch or overlap; a positive value is the physical gap in metres.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .config import DEFAULT_CLASS_SIZES
from .schemas import ActorState, PredictedPath

Point = Tuple[float, float]
Pose = Tuple[float, float, float, float]  # e, n, heading unit e, heading unit n

DEFAULT_LENGTH_M = 4.60
DEFAULT_WIDTH_M = 1.93
STATIONARY_MPS = 0.5


@dataclass(frozen=True)
class SweptPair:
    actor_a: str
    actor_b: str
    minimum_clearance_m: float
    time_after_observation_s: float
    interval_index: int
    predicted_contact: bool
    joint_path_probability: float
    path_a_maneuver: str
    path_b_maneuver: str
    clearance_at_observation_m: float
    first_contact_s: Optional[float]
    contact_duration_s: float
    clearance_at_1s_m: float
    clearance_at_horizon_m: float


def actor_footprint_m(actor: ActorState) -> Tuple[float, float]:
    """Return ``(length, width)`` in metres for an actor class."""
    size = DEFAULT_CLASS_SIZES.get(actor.cls)
    if size is None:
        return DEFAULT_LENGTH_M, DEFAULT_WIDTH_M
    return size.length_m, size.width_m


def _unit(dx: float, dy: float, fallback: Tuple[float, float]) -> Tuple[float, float]:
    norm = math.hypot(dx, dy)
    return (dx / norm, dy / norm) if norm > 1e-8 else fallback


def _fallback_heading(actor: ActorState) -> Tuple[float, float]:
    if actor.heading_deg is not None:
        angle = math.radians(actor.heading_deg)
        return math.sin(angle), math.cos(angle)
    history = actor.track_history or []
    if len(history) >= 2:
        return _unit(
            history[-1][1] - history[-2][1],
            history[-1][2] - history[-2][2],
            (1.0, 0.0),
        )
    return 1.0, 0.0


def _canonical_points(
    actor: ActorState, path: PredictedPath
) -> Tuple[List[Point], Optional[List[float]]]:
    """Normalize paths to a t=0 anchor, retaining optional actual timestamps."""
    points = list(path.waypoints or [])
    times = list(path.waypoint_times_s) if path.waypoint_times_s is not None else None
    if times is not None and len(times) != len(points):
        return [], None
    if not points:
        return [], times
    if math.dist(points[0], actor.world_xy) > 0.5:
        points.insert(0, actor.world_xy)
        if times is not None:
            times.insert(0, 0.0)
    return points, times


def _motion_fn(
    actor: ActorState, path: PredictedPath, horizon_s: float
) -> Optional[Tuple[Callable[[float], Pose], float]]:
    points, times = _canonical_points(actor, path)
    fallback = _fallback_heading(actor)
    if len(points) == 1:
        # A known stopped actor remains a valid collision target.  An unknown-speed
        # one-point path is a withheld prediction, not evidence that it stays still.
        if actor.speed_mps is None or abs(actor.speed_mps) > STATIONARY_MPS:
            return None
        points.extend([points[0]] * max(1, int(math.ceil(horizon_s))))
        if times is not None:
            times.extend(float(i) for i in range(1, len(points)))
    if len(points) < 2:
        return None

    if times is None:
        times = [float(i) for i in range(len(points))]
    if times[0] != 0.0 or any(b <= a for a, b in zip(times, times[1:])):
        return None
    available_s = min(times[-1], float(horizon_s))
    if available_s <= 0:
        return None

    def pose(t: float) -> Pose:
        clamped = max(0.0, min(t, available_s))
        if clamped >= times[-1]:
            i = len(points) - 2
            frac = 1.0
        else:
            i = max(0, min(len(points) - 2, bisect_right(times, clamped) - 1))
            frac = (clamped - times[i]) / (times[i + 1] - times[i])
        p, q = points[i], points[i + 1]
        heading = _unit(q[0] - p[0], q[1] - p[1], fallback)
        return (
            p[0] + (q[0] - p[0]) * frac,
            p[1] + (q[1] - p[1]) * frac,
            heading[0],
            heading[1],
        )

    return pose, available_s


def _corners(pose: Pose, length_m: float, width_m: float) -> List[Point]:
    cx, cy, ux, uy = pose
    px, py = -uy, ux
    hl, hw = length_m / 2.0, width_m / 2.0
    return [
        (cx + ux * hl + px * hw, cy + uy * hl + py * hw),
        (cx + ux * hl - px * hw, cy + uy * hl - py * hw),
        (cx - ux * hl - px * hw, cy - uy * hl - py * hw),
        (cx - ux * hl + px * hw, cy - uy * hl + py * hw),
    ]


def _separated(a: Sequence[Point], b: Sequence[Point]) -> bool:
    for polygon in (a, b):
        for i, p in enumerate(polygon):
            q = polygon[(i + 1) % len(polygon)]
            nx, ny = -(q[1] - p[1]), q[0] - p[0]
            pa = [nx * x + ny * y for x, y in a]
            pb = [nx * x + ny * y for x, y in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return True
    return False


def _point_segment_distance(p: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx * dx + dy * dy
    u = 0.0 if denom <= 1e-12 else max(
        0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / denom)
    )
    return math.hypot(p[0] - (a[0] + u * dx), p[1] - (a[1] + u * dy))


def polygon_clearance(a: Sequence[Point], b: Sequence[Point]) -> float:
    """Exact convex-polygon gap; zero means touch or overlap."""
    if not _separated(a, b):
        return 0.0
    best = float("inf")
    for points, edges in ((a, b), (b, a)):
        for p in points:
            for i, q in enumerate(edges):
                best = min(
                    best,
                    _point_segment_distance(p, q, edges[(i + 1) % len(edges)]),
                )
    return best


def _footprint_clearance(
    pose_a: Pose, extent_a: Tuple[float, float],
    pose_b: Pose, extent_b: Tuple[float, float],
) -> float:
    return polygon_clearance(
        _corners(pose_a, *extent_a), _corners(pose_b, *extent_b)
    )


def swept_pair_clearances(
    actors: Sequence[ActorState],
    horizon_s: float = 5.0,
    sample_dt_s: float = 0.1,
    contact_margin_m: float = 0.0,
    exclude_touching_now: bool = True,
    exclude_static_pairs: bool = True,
) -> List[SweptPair]:
    """Return actor pairs ordered by minimum predictor-derived footprint gap."""
    if sample_dt_s <= 0:
        raise ValueError("sample_dt_s must be positive")
    if horizon_s <= 0:
        return []

    candidates = []
    for actor in actors:
        paths = []
        for path in actor.predictions:
            motion = _motion_fn(actor, path, horizon_s)
            if motion is not None:
                paths.append((path, *motion))
        if paths:
            candidates.append((actor, paths, actor_footprint_m(actor)))

    results: List[SweptPair] = []
    for i, (actor_a, paths_a, extent_a) in enumerate(candidates):
        for actor_b, paths_b, extent_b in candidates[i + 1:]:
            if exclude_static_pairs and all(
                abs(actor.speed_mps or 0.0) <= STATIONARY_MPS
                for actor in (actor_a, actor_b)
            ):
                continue

            best = None
            for path_a, pose_a, available_a in paths_a:
                for path_b, pose_b, available_b in paths_b:
                    available = min(available_a, available_b, horizon_s)
                    if available < sample_dt_s - 1e-9:
                        continue
                    now_gap = _footprint_clearance(
                        pose_a(0.0), extent_a, pose_b(0.0), extent_b
                    )
                    if exclude_touching_now and now_gap <= contact_margin_m:
                        continue
                    n_steps = int(math.floor(available / sample_dt_s + 1e-9))
                    min_gap, min_t = float("inf"), sample_dt_s
                    sampled_gaps = []
                    for step in range(1, n_steps + 1):
                        t = step * sample_dt_s
                        gap = _footprint_clearance(
                            pose_a(t), extent_a, pose_b(t), extent_b
                        )
                        sampled_gaps.append((t, gap))
                        if gap < min_gap:
                            min_gap, min_t = gap, t
                    row = (
                        min_gap, min_t, now_gap, path_a, path_b, sampled_gaps
                    )
                    if best is None or row[:2] < best[:2]:
                        best = row

            if best is None:
                continue
            gap, at_s, now_gap, path_a, path_b, sampled_gaps = best
            contacts = [
                t for t, sampled_gap in sampled_gaps
                if sampled_gap <= contact_margin_m
            ]
            gap_at_1s = min(
                sampled_gaps,
                key=lambda item: abs(item[0] - min(1.0, sampled_gaps[-1][0])),
            )[1]
            results.append(
                SweptPair(
                    actor_a=actor_a.actor_id,
                    actor_b=actor_b.actor_id,
                    minimum_clearance_m=gap,
                    time_after_observation_s=at_s,
                    interval_index=max(1, int(math.ceil(at_s - 1e-9))),
                    predicted_contact=gap <= contact_margin_m,
                    joint_path_probability=path_a.probability * path_b.probability,
                    path_a_maneuver=path_a.maneuver,
                    path_b_maneuver=path_b.maneuver,
                    clearance_at_observation_m=now_gap,
                    first_contact_s=contacts[0] if contacts else None,
                    contact_duration_s=len(contacts) * sample_dt_s,
                    clearance_at_1s_m=gap_at_1s,
                    clearance_at_horizon_m=sampled_gaps[-1][1],
                )
            )

    return sorted(
        results,
        key=lambda row: (
            row.minimum_clearance_m,
            row.time_after_observation_s,
            row.actor_a,
            row.actor_b,
        ),
    )
