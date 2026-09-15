"""Dynamic features and JSON inference model for actor-pair re-ranking v2."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .pair_reranker import PAIR_TYPES, PREFIXES, actor_prefix
from .predict_model import observes_actor
from .schemas import ActorState
from .swept_path import SweptPair, actor_footprint_m


def _velocity(actor: ActorState) -> Tuple[float, float]:
    history = actor.track_history or []
    if len(history) >= 2:
        a, b = history[-2], history[-1]
        dt = b[0] - a[0]
        if dt > 1e-6:
            return (b[1] - a[1]) / dt, (b[2] - a[2]) / dt
    if actor.speed_mps is not None and actor.heading_deg is not None:
        angle = math.radians(actor.heading_deg)
        return actor.speed_mps * math.sin(angle), actor.speed_mps * math.cos(angle)
    return 0.0, 0.0


def _path_stats(actor: ActorState) -> Tuple[float, float, float, float]:
    if not actor.predictions:
        return 0.0, 0.0, 0.0, 0.0
    path = max(actor.predictions, key=lambda item: item.probability)
    points = list(path.waypoints or [])
    if points and math.dist(points[0], actor.world_xy) > 0.5:
        points.insert(0, actor.world_xy)
    if len(points) < 2:
        return 0.0, 0.0, 0.0, 0.0
    steps = [math.dist(a, b) for a, b in zip(points, points[1:])]
    displacement = math.dist(points[0], points[-1])
    speed_delta = steps[-1] - steps[0]
    first = (points[1][0] - points[0][0], points[1][1] - points[0][1])
    last = (points[-1][0] - points[-2][0], points[-1][1] - points[-2][1])
    denom = math.hypot(*first) * math.hypot(*last)
    curvature = 0.0 if denom < 1e-8 else math.acos(max(
        -1.0, min(1.0, (first[0] * last[0] + first[1] * last[1]) / denom)
    )) / math.pi
    return displacement, sum(steps) / len(steps), speed_delta, curvature


ACTOR_STAT_NAMES = (
    "speed_mps", "abs_accel_mps2", "position_quality", "range_m",
    "track_age_s", "path_displacement_m", "path_mean_step_m",
    "path_speed_delta_m", "path_curvature_fraction", "length_m", "width_m",
)

PAIR_BASE_NAMES = (
    "minimum_clearance_m", "log1p_minimum_clearance", "predicted_contact",
    "time_fraction", "time_fraction_squared", "clearance_now_m",
    "clearance_reduction_m", "clearance_at_1s_m", "clearance_at_horizon_m",
    "first_contact_fraction", "contact_duration_fraction",
    "joint_path_probability", "center_distance_now_m", "relative_speed_mps",
    "closing_speed_mps", "cv_time_to_closest_s", "heading_cosine",
    "same_road", "same_lane", "both_have_road", "any_ego", "both_ego",
    "both_speed_known", "both_accel_known", "both_range_known",
)
FEATURE_NAMES_V2 = (
    PAIR_BASE_NAMES
    + tuple(f"{name}_{suffix}" for name in ACTOR_STAT_NAMES
            for suffix in ("min", "max", "absdiff"))
    + tuple(f"pair_type_{name}" for name in PAIR_TYPES)
)

# V2 learns which *actor pair* is involved in an eventual collision.  V3 adds
# features that distinguish an unavoidable collision from a geometrical
# intersection of two deterministic forecast paths.  In particular, signed
# braking/stop state, contact recovery, and the availability (not merely the
# value) of directed observation are needed for the normal-scene hard
# negatives that dominate the Gemini false positives.
RISK_FEATURE_NAMES_V3 = (
    "clearance_recovery_from_min_m",
    "clearance_change_1s_to_horizon_m",
    "contact_within_1s",
    "signed_accel_min_mps2",
    "signed_accel_max_mps2",
    "signed_accel_absdiff_mps2",
    "both_braking",
    "either_braking",
    "both_stopped",
    "observation_available_count",
    "directed_observation_count",
    "mutually_observed",
    "known_nonmutual_observation",
)
FEATURE_NAMES_V3 = FEATURE_NAMES_V2 + RISK_FEATURE_NAMES_V3


def _actor_stats(actor: ActorState) -> List[float]:
    path = _path_stats(actor)
    length, width = actor_footprint_m(actor)
    return [
        min(abs(actor.speed_mps or 0.0), 50.0),
        min(abs(actor.accel_mps2 or 0.0), 15.0),
        max(0.0, min(actor.position_quality, 1.0)),
        min(actor.observed_range_m or 0.0, 200.0),
        min(actor.track_age_s, 20.0),
        min(path[0], 200.0), min(path[1], 50.0),
        max(-30.0, min(path[2], 30.0)), path[3], length, width,
    ]


def pair_features_v2(
    actor_a: ActorState, actor_b: ActorState, pair: SweptPair,
    horizon_s: float = 5.0,
) -> List[float]:
    """Symmetric joint features from two actors and their swept-path result."""
    horizon = max(float(horizon_s), 1e-6)
    gap = min(max(pair.minimum_clearance_m, 0.0), 30.0)
    now_gap = min(max(pair.clearance_at_observation_m, 0.0), 50.0)
    va, vb = _velocity(actor_a), _velocity(actor_b)
    rv = (vb[0] - va[0], vb[1] - va[1])
    rp = (actor_b.world_xy[0] - actor_a.world_xy[0],
          actor_b.world_xy[1] - actor_a.world_xy[1])
    center_dist = math.hypot(*rp)
    rel_speed = math.hypot(*rv)
    closing = 0.0 if center_dist < 1e-6 else -(
        rp[0] * rv[0] + rp[1] * rv[1]
    ) / center_dist
    rv2 = rv[0] * rv[0] + rv[1] * rv[1]
    cv_t = 10.0 if rv2 < 1e-8 else max(
        0.0, min(10.0, -(rp[0] * rv[0] + rp[1] * rv[1]) / rv2)
    )
    speed_product = math.hypot(*va) * math.hypot(*vb)
    heading_cos = 0.0 if speed_product < 1e-8 else max(
        -1.0, min(1.0, (va[0] * vb[0] + va[1] * vb[1]) / speed_product)
    )
    road_a = actor_a.placement
    road_b = actor_b.placement
    same_road = bool(road_a and road_b and road_a.road_id == road_b.road_id)
    same_lane = bool(same_road and road_a.lane_index is not None
                     and road_a.lane_index == road_b.lane_index)
    ego_a, ego_b = actor_a.kind == "ego", actor_b.kind == "ego"
    first_contact = pair.first_contact_s or 0.0
    base = [
        gap, math.log1p(gap), float(pair.predicted_contact),
        pair.time_after_observation_s / horizon,
        (pair.time_after_observation_s / horizon) ** 2,
        now_gap, max(-30.0, min(now_gap - gap, 50.0)),
        min(pair.clearance_at_1s_m, 50.0),
        min(pair.clearance_at_horizon_m, 50.0),
        first_contact / horizon,
        min(pair.contact_duration_s / horizon, 1.0),
        max(0.0, min(pair.joint_path_probability, 1.0)),
        min(center_dist, 250.0), min(rel_speed, 80.0),
        max(-50.0, min(closing, 50.0)), cv_t, heading_cos,
        float(same_road), float(same_lane), float(bool(road_a and road_b)),
        float(ego_a or ego_b), float(ego_a and ego_b),
        float(actor_a.speed_mps is not None and actor_b.speed_mps is not None),
        float(actor_a.accel_mps2 is not None and actor_b.accel_mps2 is not None),
        float(actor_a.observed_range_m is not None
              and actor_b.observed_range_m is not None),
    ]
    actor_features = []
    for a, b in zip(_actor_stats(actor_a), _actor_stats(actor_b)):
        actor_features.extend((min(a, b), max(a, b), abs(a - b)))
    pa, pb = sorted((actor_prefix(actor_a.actor_id), actor_prefix(actor_b.actor_id)),
                    key=PREFIXES.index)
    pair_type = f"{pa}-{pb}"
    return base + actor_features + [float(name == pair_type) for name in PAIR_TYPES]


def pair_features_v3(
    actor_a: ActorState, actor_b: ActorState, pair: SweptPair,
    horizon_s: float = 5.0,
) -> List[float]:
    """V2 features plus risk-gating features for collision-pair ranking.

    The additional observation fields preserve ``unknown`` separately from
    false: only ego actors provide a driver-observation relation in this data.
    """
    base = pair_features_v2(actor_a, actor_b, pair, horizon_s)
    accel_a = max(-15.0, min(float(actor_a.accel_mps2 or 0.0), 15.0))
    accel_b = max(-15.0, min(float(actor_b.accel_mps2 or 0.0), 15.0))
    stopped_a = float((actor_a.speed_mps or 0.0) <= 0.5)
    stopped_b = float((actor_b.speed_mps or 0.0) <= 0.5)
    braking_a, braking_b = accel_a <= -0.5, accel_b <= -0.5
    a_sees_b, b_sees_a = observes_actor(actor_a, actor_b), observes_actor(actor_b, actor_a)
    availability = float(a_sees_b is not None) + float(b_sees_a is not None)
    directed = float(a_sees_b is True) + float(b_sees_a is True)
    mutual = a_sees_b is True and b_sees_a is True
    known_nonmutual = availability == 2.0 and not mutual
    gap = max(float(pair.minimum_clearance_m), 0.0)
    extras = [
        max(-50.0, min(float(pair.clearance_at_horizon_m) - gap, 50.0)),
        max(-50.0, min(float(pair.clearance_at_horizon_m) - float(pair.clearance_at_1s_m), 50.0)),
        float(pair.predicted_contact and pair.time_after_observation_s <= 1.0),
        min(accel_a, accel_b), max(accel_a, accel_b), abs(accel_a - accel_b),
        float(braking_a and braking_b), float(braking_a or braking_b),
        float(stopped_a and stopped_b), availability, directed, float(mutual),
        float(known_nonmutual),
    ]
    return base + extras


@dataclass
class PairRerankerV2:
    means: List[float]
    scales: List[float]
    hidden_weights: List[List[float]]
    hidden_bias: List[float]
    output_weights: List[float]
    output_bias: float
    threshold: float

    def predict_features(self, values: Sequence[float]) -> float:
        normalized = [(x - m) / s for x, m, s in zip(values, self.means, self.scales)]
        hidden = [max(0.0, b + sum(w * x for w, x in zip(row, normalized)))
                  for row, b in zip(self.hidden_weights, self.hidden_bias)]
        logit = self.output_bias + sum(w * x for w, x in zip(self.output_weights, hidden))
        if logit >= 0:
            z = math.exp(-logit); return 1.0 / (1.0 + z)
        z = math.exp(logit); return z / (1.0 + z)

    def to_dict(self) -> dict:
        return {
            "model_type": "dynamic_actor_pair_mlp_v2",
            "feature_names": list(FEATURE_NAMES_V2),
            "means": self.means, "scales": self.scales,
            "hidden_weights": self.hidden_weights, "hidden_bias": self.hidden_bias,
            "output_weights": self.output_weights, "output_bias": self.output_bias,
            "decision_threshold": self.threshold,
        }

    @classmethod
    def load(cls, path: str) -> "PairRerankerV2":
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        if doc.get("model_type") != "dynamic_actor_pair_mlp_v2":
            raise ValueError("unsupported v2 pair re-ranker")
        return cls(
            means=doc["means"], scales=doc["scales"],
            hidden_weights=doc["hidden_weights"], hidden_bias=doc["hidden_bias"],
            output_weights=doc["output_weights"], output_bias=doc["output_bias"],
            threshold=doc["decision_threshold"],
        )


@dataclass
class PairRerankerV3(PairRerankerV2):
    """Risk-aware V3 MLP with the same compact JSON inference format."""

    # V2 is deliberately rank-only for backward-compatible experiments.  V3
    # is trained as a risk classifier, so its selected threshold is an actual
    # payload exposure gate rather than metadata for offline reporting.
    uses_threshold_gate: bool = True

    def to_dict(self) -> dict:
        doc = super().to_dict()
        doc["model_type"] = "risk_aware_actor_pair_mlp_v3"
        doc["feature_names"] = list(FEATURE_NAMES_V3)
        return doc

    @classmethod
    def load(cls, path: str) -> "PairRerankerV3":
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        if doc.get("model_type") != "risk_aware_actor_pair_mlp_v3":
            raise ValueError("unsupported v3 pair re-ranker")
        if len(doc.get("feature_names") or []) != len(FEATURE_NAMES_V3):
            raise ValueError("v3 pair re-ranker feature schema mismatch")
        return cls(
            means=doc["means"], scales=doc["scales"],
            hidden_weights=doc["hidden_weights"], hidden_bias=doc["hidden_bias"],
            output_weights=doc["output_weights"], output_bias=doc["output_bias"],
            threshold=doc["decision_threshold"],
        )


def load_pair_reranker(path: str) -> PairRerankerV2:
    """Load either V2 or V3 without making callers inspect checkpoint JSON."""
    with open(path, encoding="utf-8") as handle:
        model_type = json.load(handle).get("model_type")
    if model_type == "dynamic_actor_pair_mlp_v2":
        return PairRerankerV2.load(path)
    if model_type == "risk_aware_actor_pair_mlp_v3":
        return PairRerankerV3.load(path)
    raise ValueError(f"unsupported pair re-ranker model_type: {model_type!r}")


def pair_features_for_model(
    model: PairRerankerV2, actor_a: ActorState, actor_b: ActorState,
    pair: SweptPair, horizon_s: float = 5.0,
) -> List[float]:
    return (pair_features_v3 if isinstance(model, PairRerankerV3) else pair_features_v2)(
        actor_a, actor_b, pair, horizon_s
    )
