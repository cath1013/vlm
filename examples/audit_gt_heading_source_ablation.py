#!/usr/bin/env python3
"""Evaluation-only heading-source ablation for the GT-oracle production replay.

``PATH_TANGENT_HEADING`` calls the production swept-path implementation
unchanged.  ``RAW_BOX_YAW`` retains those replay/GT positions exactly, but
uses the identity-matched raw CARLA WorldBox ENU heading at every pose.
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

PATH_TANGENT_HEADING = "PATH_TANGENT_HEADING"
RAW_BOX_YAW = "RAW_BOX_YAW"
LOOKBACK_S = 0.6
EXPECTED = {"TP": 52, "FP": 193, "TN": 362, "FN": 17, "GT_actor_pair_hits": 41,
            "FP_contact_pair_rows": 337, "windows_with_FP_contact": 150,
            "scenarios_with_FP_contact": 47}


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


def _raw_history_before(a, ea, b, eb, heading_at, margin):
    pa, ta, ga = _history_pose(a, heading_at)
    pb, tb, gb = _history_pose(b, heading_at)
    if pa is None or pb is None:
        return None, True
    times = sorted({t for t in (*ta, *tb) if -LOOKBACK_S - 1e-9 <= t < -1e-9})
    valid = []
    for t in times:
        aa, bb = pa(t), pb(t)
        if aa is not None and bb is not None:
            valid.append((t, swept_path._footprint_clearance(aa, ea, bb, eb) <= margin))
    if not valid:
        return None, True
    latest_t, latest = valid[-1]
    gapped = any(t > latest_t + 1e-9 and (pa(t) is None or pb(t) is None) for t in times)
    gapped |= any(start >= latest_t - 1e-9 and end > latest_t + 1e-9 for start, end in (*ga, *gb))
    return latest, gapped


def raw_yaw_swept_pair_clearances(actors, *, frame_boxes, identities_by_actor, cutoff_s,
                                  horizon_s, sample_dt_s=.1, contact_margin_m=0., coverage=None,
                                  selected_paths=None):
    """A local heading-only variant of swept_path; XY/event rules mirror production."""
    lookup = _HeadingLookup(frame_boxes, identities_by_actor, cutoff_s)
    heading_at = lookup
    candidates = []
    for actor in actors:
        paths = [(p, *m) for p in actor.predictions
                 if (m := _future_pose(actor, p, horizon_s, heading_at)) is not None]
        if paths:
            candidates.append((actor, paths, actor_footprint_m(actor)))
    result = []
    for i, (a, paths_a, ea) in enumerate(candidates):
        for b, paths_b, eb in candidates[i + 1:]:
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
                    now_a, now_b = pose_a(0.), pose_b(0.)
                    if now_a is None or now_b is None:
                        # No source yaw at the boundary makes the state unknown.
                        lookup.skipped_t0_pairs.add((a.actor_id, b.actor_id))
                        continue
                    now_gap = swept_path._footprint_clearance(now_a, ea, now_b, eb)
                    samples, missing = [], []
                    for step in range(1, int(math.floor(available / sample_dt_s + 1e-9)) + 1):
                        t = step * sample_dt_s; aa, bb = pose_a(t), pose_b(t)
                        if aa is None or bb is None:
                            missing.append((max(0., t - sample_dt_s / 2), min(available, t + sample_dt_s / 2)))
                        else:
                            samples.append((t, swept_path._footprint_clearance(aa, ea, bb, eb)))
                    if not samples:
                        continue
                    row = (min(g for _t, g in samples), min(samples, key=lambda x: x[1])[0], now_gap,
                           path_a, path_b, samples, available, [*gaps_a, *gaps_b, *missing], bool(missing))
                    if best is None or row[:2] < best[:2]: best = row
            if best is None: continue
            gap, at, now_gap, pa, pb, samples, available, gaps, missing = best
            before, history_gapped = _raw_history_before(a, ea, b, eb, heading_at, contact_margin_m)
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
        tangent = swept_pair_clearances(selected, horizon_s=float(conf["horizon_s"]), sample_dt_s=.1,
            contact_margin_m=0., exclude_touching_now=True, exclude_static_pairs=True, contact_lookback_s=LOOKBACK_S)
        coverage, raw_paths = {}, {}
        raw_yaw = raw_yaw_swept_pair_clearances(selected, frame_boxes=frame_boxes, identities_by_actor=ids,
            cutoff_s=cutoff, horizon_s=float(conf["horizon_s"]), coverage=coverage, selected_paths=raw_paths)
        tangent_paths = _paths_recorded_by_swept(selected, tangent)
        by_condition = {PATH_TANGENT_HEADING: {frozenset((p.actor_a, p.actor_b)): p for p in tangent if p.predicted_contact},
                        RAW_BOX_YAW: {frozenset((p.actor_a, p.actor_b)): p for p in raw_yaw if p.predicted_contact}}
        for e in expected_rows:
            bucket = int(e["k"]); target = tuple(map(int, e.get("involved_carla_ids") or []))
            pairs = {c: {pair: p for pair, p in values.items() if p.interval_index == bucket}
                     for c, values in by_condition.items()}
            records.append({"scenario_type": row["scenario_type"], "scenario": row["scenario"], "window": label,
                "bucket": bucket, "cutoff_s": cutoff, "gt_positive": bool(e["accident_expected"]),
                "pairs": pairs, "hits": {c: _target_hit(v, target, ids) for c, v in pairs.items()},
                "identities": ids, "actors": {a.actor_id: a for a in selected}, "frame_boxes": frame_boxes,
                "heading_coverage": coverage, "selected_paths": {PATH_TANGENT_HEADING: tangent_paths,
                                                                      RAW_BOX_YAW: raw_paths}})
    return records


def summarize(records, condition):
    c, windows, scenarios = Counter(), set(), set()
    coverage = Counter(); seen_coverage = set()
    for r in records:
        coverage_key = (r["scenario_type"], r["scenario"], r["window"])
        if condition == RAW_BOX_YAW and coverage_key not in seen_coverage:
            seen_coverage.add(coverage_key)
            coverage.update({key: r["heading_coverage"][key] for key in
                             ("raw_heading_queries", "raw_heading_available", "raw_heading_missing",
                              "actor_pairs_skipped_missing_heading_at_t0",
                              "path_pairs_with_any_missing_future_heading")})
        pairs, pos = r["pairs"][condition], r["gt_positive"]
        if pos: c["positive"] += 1; c["hits"] += int(r["hits"][condition])
        if pairs and pos: c["TP"] += 1
        elif pairs: c["FP"] += 1; c["FP_contact_pair_rows"] += len(pairs); windows.add((r["scenario_type"],r["scenario"],r["window"])); scenarios.add((r["scenario_type"],r["scenario"]))
        elif pos: c["FN"] += 1
        else: c["TN"] += 1
    result = {"n_buckets":len(records), "positive_GT_buckets":c["positive"], "TP":c["TP"], "FP":c["FP"], "TN":c["TN"], "FN":c["FN"], "precision":None if not c["TP"]+c["FP"] else c["TP"]/(c["TP"]+c["FP"]), "recall":c["TP"]/c["positive"], "GT_actor_pair_hits":c["hits"], "GT_actor_pair_recall":c["hits"]/c["positive"], "FP_contact_pair_rows":c["FP_contact_pair_rows"], "windows_with_FP_contact":len(windows), "scenarios_with_FP_contact":len(scenarios)}
    if condition == RAW_BOX_YAW:
        total = coverage["raw_heading_queries"]
        result.update(coverage)
        result["raw_heading_coverage"] = None if not total else coverage["raw_heading_available"] / total
    return result


def _bucket_keys(records, condition, predicate):
    return {(r["scenario_type"],r["scenario"],r["window"],r["bucket"]) for r in records if predicate(r, bool(r["pairs"][condition]), r["hits"][condition])}


def pairwise(records):
    a, b = PATH_TANGENT_HEADING, RAW_BOX_YAW
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
    summaries={c:summarize(records,c) for c in (PATH_TANGENT_HEADING,RAW_BOX_YAW)}
    if {k:summaries[PATH_TANGENT_HEADING][k] for k in EXPECTED} != EXPECTED:
        raise RuntimeError(f"PATH_TANGENT_HEADING is not faithful to production: {summaries[PATH_TANGENT_HEADING]}")
    changed=changed_pairs(records)
    event_counts = {"FP_removed": {e: 0 for e in ("NEW_CONTACT", "BOUNDARY_ONSET", "RECONTACT")},
                    "FP_introduced": {e: 0 for e in ("NEW_CONTACT", "BOUNDARY_ONSET", "RECONTACT")}}
    for x in changed:
        if x["FP_removed"] and x["PATH_TANGENT_event"] in event_counts["FP_removed"]:
            event_counts["FP_removed"][x["PATH_TANGENT_event"]] += 1
        if x["FP_introduced"] and x["RAW_BOX_YAW_event"] in event_counts["FP_introduced"]:
            event_counts["FP_introduced"][x["RAW_BOX_YAW_event"]] += 1
    removed_coverage = _removed_fp_coverage_counts(changed)
    doc={"conditions":summaries,"pairwise":{f"{PATH_TANGENT_HEADING} -> {RAW_BOX_YAW}":pairwise(records)},"changed_FP_pairs_by_event_type":event_counts,**removed_coverage,"contact_lookback_s":LOOKBACK_S,"sample_dt_s":.1,"margin_m":0.}
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); (out/"summary.json").write_text(json.dumps(doc,indent=2)+"\n"); (out/"heading_source_changed_pairs.jsonl").write_text("".join(json.dumps(x)+"\n" for x in changed)); print(json.dumps(doc,indent=2))


if __name__ == "__main__": raise SystemExit(main())
