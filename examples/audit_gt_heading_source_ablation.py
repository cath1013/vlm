#!/usr/bin/env python3
"""Evaluation-only heading-source ablation for the GT-oracle production replay.

This final evaluation-only diagnostic keeps replay/GT positions and raw CARLA
yaw fixed, comparing generic actor footprints with exact per-instance raw
CARLA WorldBox dimensions.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples import audit_gt_carla_boxes as raw_audit  # noqa: E402
from examples import audit_gt_state_source_ablation as state_audit  # noqa: E402
from traffic_llm.accident_qa import build_actor_lookup  # noqa: E402
from traffic_llm.swept_path import SweptPair, _contact_event, actor_footprint_m, swept_pair_clearances  # noqa: E402
import traffic_llm.swept_path as swept_path  # noqa: E402

PATH_TANGENT_HEADING = "PATH_TANGENT_HEADING"  # retained for old diagnostic helpers
RAW_BOX_YAW_GENERIC_SIZE = "RAW_BOX_YAW_GENERIC_SIZE"
RAW_BOX_YAW_EXACT_SIZE = "RAW_BOX_YAW_EXACT_SIZE"
# Compatibility for consumers of the previous heading-only audit output.
RAW_BOX_YAW = RAW_BOX_YAW_GENERIC_SIZE
LOOKBACK_S = 0.6
EXPECTED_GENERIC = {
    "TP": 45, "FP": 18, "TN": 537, "FN": 24,
    "GT_actor_pair_hits": 45, "FP_contact_pair_rows": 19,
    "windows_with_FP_contact": 18, "scenarios_with_FP_contact": 8,
}


def identities(window, agent_ids):
    out = defaultdict(set)
    for cid, aids in build_actor_lookup(window, agent_ids).items():
        for aid in aids:
            out[aid].add(int(cid))
    return out


def _raw_heading(frame_boxes, identities_by_actor, cutoff_s, actor_id, t):
    """Return only an exact, unambiguous raw ENU heading; never interpolate it."""
    ids = identities_by_actor.get(actor_id, set())
    frame = int(round((cutoff_s + t) * raw_audit.FRAME_RATE_HZ)) + 1
    if (len(ids) != 1 or not math.isclose(state_audit.raw_time(frame), cutoff_s + t,
                                          abs_tol=1e-8)):
        return None
    box = frame_boxes.get(frame, {}).get(next(iter(ids)))
    if box is None or math.hypot(*box.heading) <= 1e-12:
        return None
    # WorldBox.heading is already a unit vector in world ENU; do not route it
    # through ActorState.heading_deg (or any CARLA-degree conversion).
    return box.heading


def _exact_footprint(frame_boxes, identities_by_actor, cutoff_s, actor_id, t):
    """Return exact raw WorldBox dimensions at this exact timestamp, or None."""
    ids = identities_by_actor.get(actor_id, set())
    frame = int(round((cutoff_s + t) * raw_audit.FRAME_RATE_HZ)) + 1
    if (len(ids) != 1 or not math.isclose(state_audit.raw_time(frame), cutoff_s + t,
                                          abs_tol=1e-8)):
        return None
    box = frame_boxes.get(frame, {}).get(next(iter(ids)))
    return None if box is None else (box.length, box.width)


class _HeadingLookup:
    """Cache exact heading availability and count each actor/time query once."""
    def __init__(self, frame_boxes, identities_by_actor, cutoff_s):
        self.frame_boxes, self.identities, self.cutoff_s = frame_boxes, identities_by_actor, cutoff_s
        self.cache = {}
        self.skipped_t0_pairs, self.future_missing_path_pairs = set(), set()

    def __call__(self, actor_id, t):
        key = (actor_id, round(t, 9))
        if key not in self.cache:
            self.cache[key] = _raw_heading(self.frame_boxes, self.identities, self.cutoff_s, actor_id, t)
        return self.cache[key]

    def report(self):
        available = sum(value is not None for value in self.cache.values())
        total = len(self.cache)
        return {"raw_heading_queries": total, "raw_heading_available": available,
                "raw_heading_missing": total - available,
                "raw_heading_coverage": None if not total else available / total,
                "actor_pairs_skipped_missing_heading_at_t0": len(self.skipped_t0_pairs),
                "path_pairs_with_any_missing_future_heading": len(self.future_missing_path_pairs)}


class _ExactFootprintLookup:
    """Cache raw dimensions and account for each exact actor/time query once."""
    def __init__(self, frame_boxes, identities_by_actor, cutoff_s):
        self.frame_boxes, self.identities, self.cutoff_s = frame_boxes, identities_by_actor, cutoff_s
        self.cache = {}
        self.skipped_t0_pairs, self.future_missing_path_pairs = set(), set()

    def __call__(self, actor, t):
        key = (actor.actor_id, round(t, 9))
        if key not in self.cache:
            self.cache[key] = _exact_footprint(self.frame_boxes, self.identities, self.cutoff_s,
                                               actor.actor_id, t)
        return self.cache[key]

    def report(self):
        available = sum(value is not None for value in self.cache.values())
        total = len(self.cache)
        return {"exact_size_queries": total, "exact_size_available": available,
                "exact_size_missing": total - available,
                "exact_size_coverage": None if not total else available / total,
                "actor_pairs_skipped_missing_exact_size_at_t0": len(self.skipped_t0_pairs),
                "path_pairs_with_any_missing_future_exact_size": len(self.future_missing_path_pairs)}


def _future_pose(actor, path, horizon_s, heading_at):
    """Production interpolation with a heading lookup substituted at the end."""
    motion = swept_path._motion_fn(actor, path, horizon_s)
    if motion is None:
        return None
    tangent_pose, available, gaps = motion

    def pose(t):
        x, y, _hx, _hy = tangent_pose(t)
        h = heading_at(actor.actor_id, t)
        return None if h is None else (x, y, h[0], h[1])
    return pose, available, gaps


def _history_pose(actor, heading_at):
    """Production history XY interpolation, with only the footprint yaw replaced."""
    tangent_pose, times, gaps = swept_path._history_pose_fn(actor)
    if tangent_pose is None:
        return None, times, gaps

    def pose(t):
        p = tangent_pose(t)
        h = heading_at(actor.actor_id, t)
        return None if p is None or h is None else (p[0], p[1], h[0], h[1])
    return pose, times, gaps


def _raw_history_before(a, b, heading_at, footprint_at, margin):
    pa, ta, ga = _history_pose(a, heading_at)
    pb, tb, gb = _history_pose(b, heading_at)
    if pa is None or pb is None:
        return None, True
    times = sorted({t for t in (*ta, *tb) if -LOOKBACK_S - 1e-9 <= t < -1e-9})
    valid, unavailable = [], False
    for t in times:
        aa, bb = pa(t), pb(t)
        ea, eb = footprint_at(a, t), footprint_at(b, t)
        if aa is not None and bb is not None and ea is not None and eb is not None:
            valid.append((t, swept_path._footprint_clearance(aa, ea, bb, eb) <= margin))
        elif aa is not None and bb is not None:
            unavailable = True
    if not valid:
        return None, True
    latest_t, latest = valid[-1]
    gapped = any(t > latest_t + 1e-9 and (pa(t) is None or pb(t) is None) for t in times)
    gapped |= any(start >= latest_t - 1e-9 and end > latest_t + 1e-9 for start, end in (*ga, *gb))
    return latest, gapped or unavailable


def raw_yaw_swept_pair_clearances(actors, *, frame_boxes, identities_by_actor, cutoff_s,
                                  horizon_s, sample_dt_s=.1, contact_margin_m=0., coverage=None,
                                  selected_paths=None, exact_size=False):
    """Local raw-yaw evaluator; only ``exact_size`` changes the footprint source."""
    lookup = _HeadingLookup(frame_boxes, identities_by_actor, cutoff_s)
    size_lookup = _ExactFootprintLookup(frame_boxes, identities_by_actor, cutoff_s) if exact_size else None
    heading_at = lookup
    footprint_at = size_lookup if size_lookup is not None else lambda actor, _t: actor_footprint_m(actor)
    candidates = []
    for actor in actors:
        paths = [(p, *m) for p in actor.predictions
                 if (m := _future_pose(actor, p, horizon_s, heading_at)) is not None]
        if paths:
            candidates.append((actor, paths))
    result = []
    for i, (a, paths_a) in enumerate(candidates):
        for b, paths_b in candidates[i + 1:]:
            best = None
            for path_a, pose_a, avail_a, gaps_a in paths_a:
                for path_b, pose_b, avail_b, gaps_b in paths_b:
                    available = min(avail_a, avail_b, horizon_s)
                    if available < sample_dt_s - 1e-9:
                        continue
                    path_key = (a.actor_id, b.actor_id, id(path_a), id(path_b))
                    future_times = [step * sample_dt_s for step in range(1, int(math.floor(available / sample_dt_s + 1e-9)) + 1)]
                    if any(heading_at(x.actor_id, t) is None for x in (a, b) for t in future_times):
                        lookup.future_missing_path_pairs.add(path_key)
                    if size_lookup and any(footprint_at(x, t) is None for x in (a, b) for t in future_times):
                        size_lookup.future_missing_path_pairs.add(path_key)
                    now_a, now_b = pose_a(0.), pose_b(0.)
                    ea, eb = footprint_at(a, 0.), footprint_at(b, 0.)
                    if now_a is None or now_b is None or ea is None or eb is None:
                        # No source yaw at the boundary makes the state unknown.
                        lookup.skipped_t0_pairs.add((a.actor_id, b.actor_id))
                        if size_lookup and (ea is None or eb is None):
                            size_lookup.skipped_t0_pairs.add((a.actor_id, b.actor_id))
                        continue
                    now_gap = swept_path._footprint_clearance(now_a, ea, now_b, eb)
                    samples, missing = [], []
                    for step in range(1, int(math.floor(available / sample_dt_s + 1e-9)) + 1):
                        t = step * sample_dt_s; aa, bb = pose_a(t), pose_b(t)
                        ea_t, eb_t = footprint_at(a, t), footprint_at(b, t)
                        if aa is None or bb is None or ea_t is None or eb_t is None:
                            missing.append((max(0., t - sample_dt_s / 2), min(available, t + sample_dt_s / 2)))
                        else:
                            samples.append((t, swept_path._footprint_clearance(aa, ea_t, bb, eb_t)))
                    if not samples:
                        continue
                    row = (min(g for _t, g in samples), min(samples, key=lambda x: x[1])[0], now_gap,
                           path_a, path_b, samples, available, [*gaps_a, *gaps_b, *missing], bool(missing))
                    if best is None or row[:2] < best[:2]: best = row
            if best is None: continue
            gap, at, now_gap, pa, pb, samples, available, gaps, missing = best
            before, history_gapped = _raw_history_before(a, b, heading_at, footprint_at, contact_margin_m)
            event, first = _contact_event(now_gap <= contact_margin_m, before, history_gapped,
                [(t, g <= contact_margin_m) for t, g in samples],
                available >= horizon_s - 1e-9 and not missing, gaps)
            predicted = event in {"NEW_CONTACT", "BOUNDARY_ONSET", "RECONTACT"}
            if all(abs(x.speed_mps or 0.) <= swept_path.STATIONARY_MPS for x in (a, b)) and not predicted:
                continue
            contacts = [t for t, g in samples if g <= contact_margin_m]
            at_1 = min(samples, key=lambda x: abs(x[0] - min(1., samples[-1][0])))[1]
            result.append(SweptPair(a.actor_id, b.actor_id, gap, at,
                1 if event == "BOUNDARY_ONSET" else max(1, int(math.ceil((first if first is not None else at) - 1e-9))),
                predicted, pa.probability * pb.probability, pa.maneuver, pb.maneuver, now_gap,
                first, len(contacts) * sample_dt_s, at_1, samples[-1][1], event,
                now_gap <= contact_margin_m, before))
            if selected_paths is not None:
                selected_paths[frozenset((a.actor_id, b.actor_id))] = (pa, pb)
    if coverage is not None:
        coverage.update(lookup.report())
        if size_lookup:
            coverage.update(size_lookup.report())
    return sorted(result, key=lambda p: (p.minimum_clearance_m, p.time_after_observation_s, p.actor_a, p.actor_b))


def _target_hit(pairs, target, actor_ids):
    return len(target) == 2 and any(
        (target[0] in actor_ids.get(a, set()) and target[1] in actor_ids.get(b, set())) or
        (target[1] in actor_ids.get(a, set()) and target[0] in actor_ids.get(b, set()))
        for a, b in (tuple(p) for p in pairs))


def _paths_recorded_by_swept(actors, swept_pairs):
    """Resolve a production SweptPair through its recorded maneuver/probability."""
    actors_by_id = {actor.actor_id: actor for actor in actors}
    out = {}
    for pair in swept_pairs:
        a, b = actors_by_id[pair.actor_a], actors_by_id[pair.actor_b]
        choices = [(pa, pb) for pa in a.predictions for pb in b.predictions
                   if pa.maneuver == pair.path_a_maneuver
                   and pb.maneuver == pair.path_b_maneuver
                   and math.isclose(pa.probability * pb.probability,
                                    pair.joint_path_probability, rel_tol=1e-12, abs_tol=1e-12)]
        if choices:
            out[frozenset((pair.actor_a, pair.actor_b))] = choices[0]
    return out


def audit_one(root, row, conf, maps, raw_scenario):
    groups = state_audit.gt_groups(Path(conf["gt_run"]) / row["directory"])
    windows, agent_ids, wc = state_audit.replay_windows(root, row, conf, maps)
    frames = sorted({f for series in raw_scenario.agents.values() for f in series.frames})
    frame_boxes = {f: raw_audit.merge_frame_boxes(raw_scenario, f)[0] for f in frames}
    records = []
    for (label, cutoff), expected_rows in groups.items():
        win = windows.get(label)
        if not win: raise KeyError(f"missing replay window {row['scenario']}/{label}")
        selected, _ = state_audit.rank_actors(win.last, wc.actor_cap(len(win.last.actors)))
        ids = identities(win, agent_ids)
        generic_coverage, exact_coverage, generic_paths, exact_paths = {}, {}, {}, {}
        generic = raw_yaw_swept_pair_clearances(selected, frame_boxes=frame_boxes, identities_by_actor=ids,
            cutoff_s=cutoff, horizon_s=float(conf["horizon_s"]), coverage=generic_coverage,
            selected_paths=generic_paths)
        exact = raw_yaw_swept_pair_clearances(selected, frame_boxes=frame_boxes, identities_by_actor=ids,
            cutoff_s=cutoff, horizon_s=float(conf["horizon_s"]), coverage=exact_coverage,
            selected_paths=exact_paths, exact_size=True)
        by_condition = {
            RAW_BOX_YAW_GENERIC_SIZE: {frozenset((p.actor_a, p.actor_b)): p for p in generic if p.predicted_contact},
            RAW_BOX_YAW_EXACT_SIZE: {frozenset((p.actor_a, p.actor_b)): p for p in exact if p.predicted_contact},
        }
        for e in expected_rows:
            bucket = int(e["k"]); target = tuple(map(int, e.get("involved_carla_ids") or []))
            pairs = {c: {pair: p for pair, p in values.items() if p.interval_index == bucket}
                     for c, values in by_condition.items()}
            records.append({"scenario_type": row["scenario_type"], "scenario": row["scenario"], "window": label,
                "bucket": bucket, "cutoff_s": cutoff, "gt_positive": bool(e["accident_expected"]),
                "target_carla_ids": sorted(target),
                "pairs": pairs, "hits": {c: _target_hit(v, target, ids) for c, v in pairs.items()},
                "identities": ids, "actors": {a.actor_id: a for a in selected}, "frame_boxes": frame_boxes,
                "coverage": {RAW_BOX_YAW_GENERIC_SIZE: generic_coverage,
                             RAW_BOX_YAW_EXACT_SIZE: exact_coverage},
                "selected_paths": {RAW_BOX_YAW_GENERIC_SIZE: generic_paths,
                                   RAW_BOX_YAW_EXACT_SIZE: exact_paths}})
    return records


def summarize(records, condition):
    c, windows, scenarios = Counter(), set(), set()
    coverage = Counter(); seen_coverage = set()
    for r in records:
        coverage_key = (r["scenario_type"], r["scenario"], r["window"])
        if condition in (RAW_BOX_YAW_GENERIC_SIZE, RAW_BOX_YAW_EXACT_SIZE) and coverage_key not in seen_coverage:
            seen_coverage.add(coverage_key)
            coverage.update({key: r["coverage"][condition][key] for key in
                             ("raw_heading_queries", "raw_heading_available", "raw_heading_missing",
                              "actor_pairs_skipped_missing_heading_at_t0",
                              "path_pairs_with_any_missing_future_heading")})
            if condition == RAW_BOX_YAW_EXACT_SIZE:
                coverage.update({key: r["coverage"][condition][key] for key in
                                 ("exact_size_queries", "exact_size_available", "exact_size_missing",
                                  "actor_pairs_skipped_missing_exact_size_at_t0",
                                  "path_pairs_with_any_missing_future_exact_size")})
        pairs, pos = r["pairs"][condition], r["gt_positive"]
        if pos: c["positive"] += 1; c["hits"] += int(r["hits"][condition])
        if pairs and pos: c["TP"] += 1
        elif pairs: c["FP"] += 1; c["FP_contact_pair_rows"] += len(pairs); windows.add((r["scenario_type"],r["scenario"],r["window"])); scenarios.add((r["scenario_type"],r["scenario"]))
        elif pos: c["FN"] += 1
        else: c["TN"] += 1
    result = {"n_buckets":len(records), "positive_GT_buckets":c["positive"], "TP":c["TP"], "FP":c["FP"], "TN":c["TN"], "FN":c["FN"], "precision":None if not c["TP"]+c["FP"] else c["TP"]/(c["TP"]+c["FP"]), "recall":c["TP"]/c["positive"], "GT_actor_pair_hits":c["hits"], "GT_actor_pair_recall":c["hits"]/c["positive"], "FP_contact_pair_rows":c["FP_contact_pair_rows"], "windows_with_FP_contact":len(windows), "scenarios_with_FP_contact":len(scenarios)}
    if condition in (RAW_BOX_YAW_GENERIC_SIZE, RAW_BOX_YAW_EXACT_SIZE):
        total = coverage["raw_heading_queries"]
        result.update(coverage)
        result["raw_heading_coverage"] = None if not total else coverage["raw_heading_available"] / total
    if condition == RAW_BOX_YAW_EXACT_SIZE:
        total = coverage["exact_size_queries"]
        result["exact_size_coverage"] = None if not total else coverage["exact_size_available"] / total
    return result


def _bucket_keys(records, condition, predicate):
    return {(r["scenario_type"],r["scenario"],r["window"],r["bucket"]) for r in records if predicate(r, bool(r["pairs"][condition]), r["hits"][condition])}


def pairwise(records, a=RAW_BOX_YAW_GENERIC_SIZE, b=RAW_BOX_YAW_EXACT_SIZE):
    af, bf = _bucket_keys(records,a,lambda r,p,h:not r["gt_positive"] and p), _bucket_keys(records,b,lambda r,p,h:not r["gt_positive"] and p)
    at, bt = _bucket_keys(records,a,lambda r,p,h:r["gt_positive"] and p), _bucket_keys(records,b,lambda r,p,h:r["gt_positive"] and p)
    ah, bh = _bucket_keys(records,a,lambda r,p,h:r["gt_positive"] and h), _bucket_keys(records,b,lambda r,p,h:r["gt_positive"] and h)
    return {"FP removed":len(af-bf), "FP introduced":len(bf-af), "TP lost":len(at-bt), "TP gained":len(bt-at), "GT-pair hits lost":len(ah-bh), "GT-pair hits gained":len(bh-ah)}


def _diagnostic_pose(r, aid, t, path):
    actor = r["actors"][aid]
    if path is None:
        return None
    motion = swept_path._motion_fn(actor, path, max(0.1, t))
    tangent = motion[0](max(0., t)) if motion else None
    raw = _raw_heading(r["frame_boxes"], r["identities"], r["cutoff_s"], aid, t)
    angle = None if tangent is None or raw is None else math.degrees(math.acos(max(-1., min(1., tangent[2]*raw[0]+tangent[3]*raw[1]))))
    return {"production_xy_m":None if tangent is None else [tangent[0],tangent[1]], "path_tangent_heading_enu":None if tangent is None else [tangent[2],tangent[3]], "raw_carla_heading_enu":None if raw is None else list(raw), "angular_difference_deg":angle}


def _required_future_times(t):
    return [step * .1 for step in range(int(math.floor(max(0., t) / .1 + 1e-9)) + 1)]


def _missing_raw_heading_times(r, aid, times):
    return [t for t in times if _raw_heading(r["frame_boxes"], r["identities"], r["cutoff_s"], aid, t) is None]


def _history_heading_times(actor):
    history = sorted(actor.track_history or [])
    if not history:
        return []
    now = history[-1][0]
    return [t - now for t, _x, _y in history if -LOOKBACK_S - 1e-9 <= t - now < -1e-9]


def _removed_fp_coverage_category(removed, complete):
    if not removed:
        return None
    return "FP_removed_with_complete_raw_heading" if complete else "FP_removed_with_missing_raw_heading"


def _removed_fp_coverage_counts(rows):
    counts = Counter(row["FP_removed_coverage_category"] for row in rows
                     if row["FP_removed_coverage_category"] is not None)
    return {"FP_removed_with_complete_raw_heading": counts["FP_removed_with_complete_raw_heading"],
            "FP_removed_with_missing_raw_heading": counts["FP_removed_with_missing_raw_heading"]}


def residual_raw_yaw_fp_rows(records, condition=RAW_BOX_YAW_GENERIC_SIZE):
    """Return every condition contact pair from a GT-negative bucket.

    Accident targets are scenario-level labels.  A target is usable only when
    the records for that scenario provide one unambiguous two-CARLA-ID pair.
    """
    scenario_targets = defaultdict(set)
    for r in records:
        target = tuple(r["target_carla_ids"])
        if len(target) == 2:
            scenario_targets[(r["scenario_type"], r["scenario"])].add(target)

    rows = []
    for r in records:
        if r["gt_positive"]:
            continue
        key = (r["scenario_type"], r["scenario"])
        targets = scenario_targets[key]
        target = next(iter(targets)) if len(targets) == 1 else None
        for pair, p in r["pairs"][condition].items():
            actor_pair = [p.actor_a, p.actor_b]
            source_carla_ids = [sorted(r["identities"][actor_id]) for actor_id in actor_pair]
            actors = [r["actors"][actor_id] for actor_id in actor_pair]
            if r["scenario_type"] == "normal" or r["scenario_type"].endswith("_normal"):
                category = "NORMAL_SCENARIO_CONTACT"
            elif target is None or any(len(ids) != 1 for ids in source_carla_ids):
                category = "IDENTITY_OR_TARGET_UNKNOWN"
            else:
                predicted = {ids[0] for ids in source_carla_ids}
                target_ids = set(target)
                if predicted == target_ids:
                    category = "TARGET_PAIR_WRONG_BUCKET"
                elif len(predicted & target_ids) == 1:
                    category = "ONE_TARGET_ACTOR_PLUS_OTHER"
                else:
                    category = "UNRELATED_ACTOR_PAIR"
            rows.append({
                "scenario_type": r["scenario_type"], "scenario": r["scenario"],
                "window": r["window"], "bucket": r["bucket"], "cutoff_s": r["cutoff_s"],
                "actor_pair": actor_pair, "source_carla_ids": source_carla_ids,
                "scenario_target_carla_ids": None if target is None else list(target),
                "category": category,
                "event": p.contact_event, "first_contact_time_s": p.first_contact_s,
                "minimum_clearance_m": p.minimum_clearance_m,
                "clearance_at_observation_m": p.clearance_at_observation_m,
                "clearance_at_1s_m": p.clearance_at_1s_m,
                "clearance_at_horizon_m": p.clearance_at_horizon_m,
                "contact_duration_s": p.contact_duration_s,
                "contact_at_observation": p.contact_at_observation,
                "contact_before_observation": p.contact_before_observation,
                "path_a_maneuver": p.path_a_maneuver,
                "path_b_maneuver": p.path_b_maneuver,
                "joint_path_probability": p.joint_path_probability,
                "actor_classes": [actor.cls for actor in actors],
                "actor_speeds_mps": [actor.speed_mps for actor in actors],
                "actor_footprints_m": [list(actor_footprint_m(actor)) for actor in actors],
            })
    return rows


def exact_size_fp_comparison(records):
    """Compare all generic residual FP rows with exact-size residual FP rows."""
    generic = residual_raw_yaw_fp_rows(records, RAW_BOX_YAW_GENERIC_SIZE)
    exact = residual_raw_yaw_fp_rows(records, RAW_BOX_YAW_EXACT_SIZE)
    key = lambda row: (row["scenario_type"], row["scenario"], row["window"],
                       row["bucket"], tuple(row["actor_pair"]))
    generic_by_key, exact_by_key = {key(row): row for row in generic}, {key(row): row for row in exact}
    record_by_bucket = {(r["scenario_type"], r["scenario"], r["window"], r["bucket"]): r for r in records}
    rows = []
    for pair_key in sorted(set(generic_by_key) | set(exact_by_key)):
        g, e = generic_by_key.get(pair_key), exact_by_key.get(pair_key)
        base = g or e
        status = "SURVIVES_EXACT_SIZE" if g and e else (
            "REMOVED_BY_EXACT_SIZE" if g else "INTRODUCED_BY_EXACT_SIZE")
        r = record_by_bucket[pair_key[:4]]
        relevant_t = next((x["first_contact_time_s"] for x in (g, e)
                           if x is not None and x["first_contact_time_s"] is not None), 0.)
        aids = base["actor_pair"]
        exact_dimensions = [_exact_footprint(r["frame_boxes"], r["identities"], r["cutoff_s"], aid, relevant_t)
                            for aid in aids]
        rows.append({
            "scenario_type": base["scenario_type"], "scenario": base["scenario"],
            "window": base["window"], "bucket": base["bucket"], "cutoff_s": base["cutoff_s"],
            "actor_pair": aids, "source_carla_ids": base["source_carla_ids"],
            "scenario_target_carla_ids": base["scenario_target_carla_ids"],
            "residual_category": base["category"], "generic_event": None if g is None else g["event"],
            "exact_event": None if e is None else e["event"],
            "generic_first_contact_time_s": None if g is None else g["first_contact_time_s"],
            "exact_first_contact_time_s": None if e is None else e["first_contact_time_s"],
            "generic_minimum_clearance_m": None if g is None else g["minimum_clearance_m"],
            "exact_minimum_clearance_m": None if e is None else e["minimum_clearance_m"],
            "generic_contact_duration_s": None if g is None else g["contact_duration_s"],
            "exact_contact_duration_s": None if e is None else e["contact_duration_s"],
            "generic_actor_footprints_m": None if g is None else g["actor_footprints_m"],
            "exact_actor_footprints_at_relevant_time_m": [None if x is None else list(x) for x in exact_dimensions],
            "status": status,
        })
    return rows


def fp_pair_event_changes(rows):
    events = ("NEW_CONTACT", "BOUNDARY_ONSET", "RECONTACT")
    return {
        "FP removed": {event: sum(row["status"] == "REMOVED_BY_EXACT_SIZE" and
                                    row["generic_event"] == event for row in rows) for event in events},
        "FP introduced": {event: sum(row["status"] == "INTRODUCED_BY_EXACT_SIZE" and
                                      row["exact_event"] == event for row in rows) for event in events},
    }


def changed_pairs(records):
    rows = []
    for r in records:
        a, b = r["pairs"][PATH_TANGENT_HEADING], r["pairs"][RAW_BOX_YAW]
        for pair in set(a) ^ set(b):
            pa, pb = a.get(pair), b.get(pair); p = pa or pb; t = p.first_contact_s if p.first_contact_s is not None else 0.
            aids = sorted(pair); fp_removed = not r["gt_positive"] and pa is not None and pb is None
            fp_introduced = not r["gt_positive"] and pb is not None and pa is None
            future_times = _required_future_times(t)
            missing_by_actor = {aid: _missing_raw_heading_times(r, aid, future_times) for aid in aids}
            complete = not any(missing_by_actor.values())
            history_required = any(p and p.contact_event in {"BOUNDARY_ONSET", "RECONTACT"} for p in (pa, pb))
            history_missing = {aid: _missing_raw_heading_times(r, aid, _history_heading_times(r["actors"][aid])) for aid in aids}
            complete_history = (not any(history_missing.values())) if history_required else None
            tangent_paths = r["selected_paths"][PATH_TANGENT_HEADING].get(pair, (None, None))
            raw_paths = r["selected_paths"][RAW_BOX_YAW].get(pair, (None, None))
            rows.append({"scenario":r["scenario"],"window":r["window"],"bucket":r["bucket"],"actor_pair":aids,
                "source_carla_ids":[sorted(r["identities"].get(x,set())) for x in aids],"GT_positive":r["gt_positive"],
                "PATH_TANGENT_event":None if pa is None else pa.contact_event,"PATH_TANGENT_first_contact_time_s":None if pa is None else pa.first_contact_s,
                "RAW_BOX_YAW_event":None if pb is None else pb.contact_event,"RAW_BOX_YAW_first_contact_time_s":None if pb is None else pb.first_contact_s,
                "FP_removed":fp_removed,"FP_introduced":fp_introduced,"relevant_contact_time_s":t,
                "raw_heading_complete_to_relevant_contact":complete,
                "missing_raw_heading_times_s":missing_by_actor,
                "raw_heading_complete_history":complete_history,
                "missing_raw_heading_history_times_s":history_missing if history_required else None,
                "FP_removed_coverage_category":_removed_fp_coverage_category(fp_removed, complete and (complete_history is not False)),
                "PATH_TANGENT_actors":[_diagnostic_pose(r,x,t,path) for x,path in zip(aids,tangent_paths)],
                "RAW_BOX_YAW_actors":[_diagnostic_pose(r,x,t,path) for x,path in zip(aids,raw_paths)]})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--root",required=True); ap.add_argument("--carla-maps",required=True); ap.add_argument("--gt-run",default="out/experiments/gt_future_waypointnet_0m_event_val104"); ap.add_argument("--out",required=True); ap.add_argument("--workers",type=int,default=1)
    args = ap.parse_args(argv); run = Path(args.gt_run); cohort = json.loads((run/"batch_manifest.json").read_text()); conf = dict(cohort["config"],gt_run=str(run))
    scenarios = {raw_audit.scenario_key(s):s for s in state_audit.scan_scenarios(args.root)}
    missing = [r["scenario"] for r in cohort["scenarios"] if raw_audit.scenario_row_key(r) not in scenarios]
    if missing: raise KeyError(f"raw scenarios not found: {missing[:10]}")
    records=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fs={pool.submit(audit_one,args.root,r,conf,args.carla_maps,scenarios[raw_audit.scenario_row_key(r)]):r for r in cohort["scenarios"]}
        for i,f in enumerate(as_completed(fs),1): records.extend(f.result()); print(f"[{i}/{len(fs)}] complete {fs[f]['scenario']}",flush=True)
    summaries = {c: summarize(records, c) for c in (RAW_BOX_YAW_GENERIC_SIZE, RAW_BOX_YAW_EXACT_SIZE)}
    generic_summary = summaries[RAW_BOX_YAW_GENERIC_SIZE]
    if (generic_summary["n_buckets"] != 624 or generic_summary["positive_GT_buckets"] != 69 or
            {k: generic_summary[k] for k in EXPECTED_GENERIC} != EXPECTED_GENERIC):
        raise RuntimeError(f"RAW_BOX_YAW_GENERIC_SIZE is not faithful: {generic_summary}")
    fp_rows = exact_size_fp_comparison(records)
    categories = Counter(row["residual_category"] for row in fp_rows if row["status"] != "REMOVED_BY_EXACT_SIZE")
    events = Counter(row["exact_event"] for row in fp_rows if row["exact_event"] is not None)
    removed = sum(row["status"] == "REMOVED_BY_EXACT_SIZE" for row in fp_rows)
    surviving = sum(row["status"] == "SURVIVES_EXACT_SIZE" for row in fp_rows)
    introduced = sum(row["status"] == "INTRODUCED_BY_EXACT_SIZE" for row in fp_rows)
    doc = {
        "conditions": summaries,
        "pairwise": {f"{RAW_BOX_YAW_GENERIC_SIZE} -> {RAW_BOX_YAW_EXACT_SIZE}": pairwise(records),
                     "FP_pair_row_changes_by_event": fp_pair_event_changes(fp_rows)},
        "generic_residual_FP_pair_rows": len(residual_raw_yaw_fp_rows(records, RAW_BOX_YAW_GENERIC_SIZE)),
        "generic_residual_removed_by_exact_size": removed,
        "generic_residual_surviving_exact_size": surviving,
        "exact_size_FP_pair_rows_introduced": introduced,
        "exact_size_residual_categories": {category: categories[category] for category in
            ("NORMAL_SCENARIO_CONTACT", "TARGET_PAIR_WRONG_BUCKET", "ONE_TARGET_ACTOR_PLUS_OTHER",
             "UNRELATED_ACTOR_PAIR", "IDENTITY_OR_TARGET_UNKNOWN")},
        "exact_size_residual_events": {event: events[event] for event in
            ("NEW_CONTACT", "BOUNDARY_ONSET", "RECONTACT")},
        "exact_size_coverage": summaries[RAW_BOX_YAW_EXACT_SIZE]["exact_size_coverage"],
        "contact_lookback_s": LOOKBACK_S, "sample_dt_s": .1, "margin_m": 0.,
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(doc, indent=2) + "\n")
    (out / "exact_size_fp_comparison.jsonl").write_text("".join(json.dumps(x) + "\n" for x in fp_rows))
    # Keep the previous residual listing available under its established name.
    generic_residual = residual_raw_yaw_fp_rows(records, RAW_BOX_YAW_GENERIC_SIZE)
    (out / "raw_box_yaw_fp_pairs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in generic_residual))
    print(json.dumps(doc, indent=2))


if __name__ == "__main__": raise SystemExit(main())
