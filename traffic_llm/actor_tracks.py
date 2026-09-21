"""Shared actor-trajectory target helpers.

The motion-training dataset and evaluation-only oracle use the same sampling
contract so a ``+1s`` target always means the same thing in both places.
"""

from __future__ import annotations


def future_track(
    per_actor: dict, actor_id: str, t0: float, horizon_s: float, dt: float = 1.0
):
    """Return actual positions at ``t0, t0 + dt, ...`` until unavailable.

    The nearest source sample may be at most 0.3 seconds away. A missing
    sample truncates the track instead of extrapolating it.
    """
    samples = per_actor.get(actor_id, [])
    if not samples:
        return [], False
    out = []
    k = 0.0
    while k <= horizon_s + 1e-6:
        want = t0 + k
        best = min(samples, key=lambda s: abs(s[0] - want))
        if abs(best[0] - want) > 0.3:
            return out, False
        out.append(best[1])
        k += dt
    return out, True
