"""Small, dependency-free logistic re-ranker for predicted actor pairs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import List, Sequence

PREFIXES = ("EGO", "V", "T", "M", "C", "P", "N", "OTHER")
PAIR_TYPES = tuple(
    f"{a}-{b}"
    for i, a in enumerate(PREFIXES)
    for b in PREFIXES[i:]
)
FEATURE_NAMES = (
    "minimum_clearance_m_clipped",
    "log1p_minimum_clearance",
    "predicted_contact",
    "interval_fraction",
    "interval_fraction_squared",
    "any_ego",
    "both_ego",
) + tuple(f"pair_type_{name}" for name in PAIR_TYPES)


def actor_prefix(actor_id: str) -> str:
    if actor_id.startswith("EGO_"):
        return "EGO"
    prefix = actor_id[:1].upper()
    return prefix if prefix in PREFIXES else "OTHER"


def pair_features(
    actor_a: str, actor_b: str, clearance_m: float, interval_index: int,
    horizon_buckets: int = 5,
) -> List[float]:
    """Common features available in both the old train cache and compact v3."""
    gap = max(0.0, min(float(clearance_m), 20.0))
    k = max(1, min(int(interval_index), horizon_buckets))
    frac = k / max(1, horizon_buckets)
    pa, pb = sorted((actor_prefix(actor_a), actor_prefix(actor_b)),
                    key=PREFIXES.index)
    pair_type = f"{pa}-{pb}"
    base = [
        gap,
        math.log1p(gap),
        float(gap <= 1e-9),
        frac,
        frac * frac,
        float("EGO" in (pa, pb)),
        float(pa == pb == "EGO"),
    ]
    return base + [float(name == pair_type) for name in PAIR_TYPES]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass
class PairReranker:
    weights: List[float]
    bias: float
    means: List[float]
    scales: List[float]
    threshold: float
    feature_names: Sequence[str] = FEATURE_NAMES

    def predict_proba(
        self, actor_a: str, actor_b: str, clearance_m: float,
        interval_index: int, horizon_buckets: int = 5,
    ) -> float:
        values = pair_features(
            actor_a, actor_b, clearance_m, interval_index, horizon_buckets
        )
        logit = self.bias + sum(
            weight * ((value - mean) / scale)
            for weight, value, mean, scale in zip(
                self.weights, values, self.means, self.scales
            )
        )
        return _sigmoid(logit)

    def to_dict(self) -> dict:
        return {
            "model_type": "standardized_logistic_actor_pair_reranker_v1",
            "feature_names": list(self.feature_names),
            "weights": self.weights,
            "bias": self.bias,
            "means": self.means,
            "scales": self.scales,
            "decision_threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, doc: dict) -> "PairReranker":
        if doc.get("model_type") != "standardized_logistic_actor_pair_reranker_v1":
            raise ValueError("unsupported pair re-ranker model")
        return cls(
            weights=list(doc["weights"]),
            bias=float(doc["bias"]),
            means=list(doc["means"]),
            scales=list(doc["scales"]),
            threshold=float(doc["decision_threshold"]),
            feature_names=tuple(doc["feature_names"]),
        )

    @classmethod
    def load(cls, path: str) -> "PairReranker":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
