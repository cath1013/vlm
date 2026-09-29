#!/usr/bin/env python3
"""Evaluation-only XY-source ablation after the raw-CARLA-yaw correction."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples import audit_gt_carla_boxes as raw_audit  # noqa: E402
from examples import audit_gt_heading_source_ablation as heading_audit  # noqa: E402
from examples import audit_gt_state_source_ablation as state_audit  # noqa: E402
from traffic_llm.accident_qa import WindowConfig, build_windows, ground_truth_future_snapshots  # noqa: E402
from traffic_llm.carla_map import find_xodr  # noqa: E402
from traffic_llm.config import PipelineConfig  # noqa: E402
from traffic_llm.da_runner import DeepAccidentRunner  # noqa: E402
from traffic_llm.deepaccident import estimate_collision  # noqa: E402
from traffic_llm.schemas import PredictedPath  # noqa: E402
from traffic_llm.swept_path import SweptPair, _contact_event, actor_footprint_m  # noqa: E402
import traffic_llm.swept_path as swept_path  # noqa: E402

PRODUCTION_GT_XY_RAW_YAW = "PRODUCTION_GT_XY_RAW_YAW"
RAW_CARLA_XY_RAW_YAW = "RAW_CARLA_XY_RAW_YAW"
LOOKBACK_S = .6
EXPECTED = {"TP": 46, "FP": 58, "TN": 497, "FN": 23, "GT_actor_pair_hits": 45,
            "FP_contact_pair_rows": 79, "windows_with_FP_contact": 52,
            "scenarios_with_FP_contact": 23}


def _gt_placeholder_predictor(context):
    """Create the required path container; GT replacement supplies all geometry."""
    actor = context.actor
    return [PredictedPath("GT-oracle placeholder", 1.0, [actor.world_xy], context.horizon_s)]


def replay_windows_profiled(root, row, conf, maps, *, legacy_replay=False):
    """Faithful replay with fine-grained timings and an optional no-model path."""
    timings = {}
    start = time.perf_counter(); cfg = PipelineConfig(); cfg.deepaccident.observation_mode = conf["mode"]
    timings["pipeline_config_s"] = time.perf_counter() - start
    start = time.perf_counter()
    if legacy_replay:
        from traffic_llm.predict_model import TorchPredictor
        cfg.predictor = TorchPredictor(conf["predictor"], mode=conf.get("predictor_mode") or "waypoints")
    else:
        cfg.predictor = _gt_placeholder_predictor
    timings["predictor_construction_s"] = time.perf_counter() - start
    start = time.perf_counter()
    if legacy_replay:
        # TorchPredictor normally loads lazily on the first snapshot.  Loading
        # it here changes only timing attribution, not its model or outputs.
        cfg.predictor._load()
    timings["predictor_loading_s"] = time.perf_counter() - start
    start = time.perf_counter(); runner = DeepAccidentRunner(root, cfg)
    timings["runner_construction_s"] = time.perf_counter() - start
    start = time.perf_counter(); xodr = find_xodr(row["town"], [maps])
    timings["find_xodr_s"] = time.perf_counter() - start
    if not xodr: raise FileNotFoundError(f"OpenDRIVE map for {row['town']} not found below {maps}")
    start = time.perf_counter(); built = runner.build(row["scenario"], row["scenario_type"], opendrive_path=xodr)
    timings["runner_build_s"] = time.perf_counter() - start
    start = time.perf_counter(); online = list(built.snapshots(rate_hz=2.0))
    timings["online_snapshots_2hz_s"] = time.perf_counter() - start
    start = time.perf_counter(); raw = list(built.snapshots(rate_hz=cfg.deepaccident.frame_rate_hz))
    timings["raw_snapshots_frame_rate_s"] = time.perf_counter() - start
    start = time.perf_counter(); oracle = ground_truth_future_snapshots(online, float(conf["horizon_s"]), raw_snapshots=raw)
    timings["ground_truth_future_snapshots_s"] = time.perf_counter() - start
    wc = WindowConfig(window_s=float(conf["window_s"]), stride_s=float(conf["stride_s"]), horizon_s=float(conf["horizon_s"]), snapshot_rate_hz=2.0, history_stride_s=float(conf["history_stride_s"]), warmup=False, full_window=False, future_source="ground_truth", swept_sample_dt_s=.1, swept_contact_margin_m=0.)
    start = time.perf_counter(); collision = estimate_collision(built.scenario, cfg.deepaccident)
    timings["estimate_collision_s"] = time.perf_counter() - start
    start = time.perf_counter(); windows, _ = build_windows(oracle, wc, collision_time_s=collision.time_s if collision and collision.occurred else None)
    timings["build_windows_s"] = time.perf_counter() - start
    timings["replay_total_s"] = sum(timings.values())
    agent_ids = {a: built.scenario.meta.agent_id_of(a) for a in built.scenario.agents}
    return {w.label: w for w in windows}, agent_ids, wc, timings


def _assert_replay_equivalent(legacy, lightweight):
    """Compare every audit-relevant window/ActorState field before using no-op replay."""
    lwindows, lids, lwc, _lt = legacy; owindows, oids, owc, _ot = lightweight
    if lids != oids or lwc != owc or list(lwindows) != list(owindows):
        raise AssertionError("replay window or identity metadata differs")
    for label in lwindows:
        a, b = lwindows[label], owindows[label]
        if not (math.isclose(a.t_end, b.t_end, abs_tol=1e-9) and math.isclose(a.t_start, b.t_start, abs_tol=1e-9)):
            raise AssertionError(f"window time differs: {label}")
        aa, bb = a.last.actors, b.last.actors
        if len(aa) != len(bb): raise AssertionError(f"actor count differs: {label}")
        for x, y in zip(aa, bb):
            if (x.actor_id, x.cls, x.track_history) != (y.actor_id, y.cls, y.track_history):
                raise AssertionError(f"actor identity/history differs: {label}/{x.actor_id}")
            for xv, yv in (*zip(x.world_xy, y.world_xy), (x.heading_deg, y.heading_deg), (x.speed_mps, y.speed_mps)):
                if xv is None or yv is None:
                    if xv != yv: raise AssertionError(f"actor scalar differs: {label}/{x.actor_id}")
                elif not math.isclose(xv, yv, abs_tol=1e-9): raise AssertionError(f"actor scalar differs: {label}/{x.actor_id}")
            lselected, _ = heading_audit.rank_actors(a.last, lwc.actor_cap(len(aa)))
            oselected, _ = heading_audit.rank_actors(b.last, owc.actor_cap(len(bb)))
            if [x.actor_id for x in lselected] != [x.actor_id for x in oselected]:
                raise AssertionError(f"actor selection differs: {label}")
            if heading_audit.identities(a, lids) != heading_audit.identities(b, oids):
                raise AssertionError(f"CARLA identities differ: {label}")
            if len(x.predictions) != len(y.predictions): raise AssertionError(f"prediction count differs: {label}/{x.actor_id}")
            for px, py in zip(x.predictions, y.predictions):
                if px.waypoints != py.waypoints or px.waypoint_times_s != py.waypoint_times_s:
                    raise AssertionError(f"GT paths differ: {label}/{x.actor_id}")


class _FrameBoxCache:
    """Scenario-local, lazy frame cache shared by both XY conditions.

    Keys are absolute raw frames.  A frame is parsed at most once even when it
    is needed by multiple cutoffs, history samples, or diagnostic rows.
    """
    def __init__(self, scenario):
        self.scenario = scenario
        self.valid_frames = {f for series in scenario.agents.values() for f in series.frames}
        self.loaded, self.load_s = {}, 0.0

    def get(self, frame, default=None):
        if frame not in self.valid_frames:
            return default
        if frame not in self.loaded:
            start = time.perf_counter()
            self.loaded[frame] = _FrameBoxes(raw_audit.merge_frame_boxes(self.scenario, frame)[0])
            self.load_s += time.perf_counter() - start
        return self.loaded[frame]


class _FrameBoxes:
    """Memoize absolute ``(frame, CARLA ID)`` pose retrieval for both conditions."""
    def __init__(self, boxes):
        self.boxes, self.pose_cache = boxes, {}

    def get(self, carla_id, default=None):
        if carla_id not in self.pose_cache:
            self.pose_cache[carla_id] = self.boxes.get(carla_id)
        value = self.pose_cache[carla_id]
        return default if value is None else value


class _RawBoxLookup:
    """Exact identity/frame lookup.  Missing boxes remain unknown, never sampled nearby."""
    def __init__(self, frame_boxes, identities, cutoff_s, pose_cache=None):
        self.frame_boxes, self.identities, self.cutoff_s, self.cache = frame_boxes, identities, cutoff_s, {}
        self.pose_cache = pose_cache if pose_cache is not None else {}
        self.skipped_t0_pairs, self.future_missing_path_pairs = set(), set()

    def __call__(self, actor_id, t):
        key = actor_id, round(t, 9)
        if key not in self.cache:
            ids = self.identities.get(actor_id, set())
            frame = int(round((self.cutoff_s + t) * raw_audit.FRAME_RATE_HZ)) + 1
            if len(ids) != 1 or not math.isclose(state_audit.raw_time(frame), self.cutoff_s + t, abs_tol=1e-8):
                self.cache[key] = None
            else:
                pose_key = frame, next(iter(ids))
                if pose_key not in self.pose_cache:
                    self.pose_cache[pose_key] = self.frame_boxes.get(frame, {}).get(pose_key[1])
                self.cache[key] = self.pose_cache[pose_key]
        return self.cache[key]

    def report(self):
        n = len(self.cache); available = sum(b is not None for b in self.cache.values())
        return {"raw_xy_queries": n, "raw_xy_available": available, "raw_xy_missing": n - available,
                "raw_xy_coverage": None if not n else available / n,
                "actor_pairs_skipped_missing_raw_xy_at_t0": len(self.skipped_t0_pairs),
                "path_pairs_with_any_missing_future_raw_xy": len(self.future_missing_path_pairs)}


def _raw_xy_pose(actor, path, horizon_s, box_at):
    motion = swept_path._motion_fn(actor, path, horizon_s)
    if motion is None: return None
    tangent, available, gaps = motion
    def pose(t):
        box = box_at(actor.actor_id, t)
        return None if box is None else (box.center[0], box.center[1], box.heading[0], box.heading[1])
    return pose, available, gaps


def _raw_xy_history_pose(actor, box_at):
    production_pose, times, gaps = swept_path._history_pose_fn(actor)
    if production_pose is None: return None, times, gaps
    def pose(t):
        box = box_at(actor.actor_id, t)
        return None if box is None else (box.center[0], box.center[1], box.heading[0], box.heading[1])
    return pose, times, gaps


def _history_before(a, ea, b, eb, box_at, margin):
    pa, ta, ga = _raw_xy_history_pose(a, box_at); pb, tb, gb = _raw_xy_history_pose(b, box_at)
    if pa is None or pb is None: return None, True
    times = sorted({t for t in (*ta, *tb) if -LOOKBACK_S - 1e-9 <= t < -1e-9})
    valid = [(t, swept_path._footprint_clearance(pa(t), ea, pb(t), eb) <= margin)
             for t in times if pa(t) is not None and pb(t) is not None]
    if not valid: return None, True
    latest_t, latest = valid[-1]
    gapped = any(t > latest_t + 1e-9 and (pa(t) is None or pb(t) is None) for t in times)
    gapped |= any(start >= latest_t - 1e-9 and end > latest_t + 1e-9 for start, end in (*ga, *gb))
    return latest, gapped


def _actor_pair_universe(actors):
    return {frozenset((a.actor_id, b.actor_id)) for i, a in enumerate(actors) for b in actors[i + 1:]}


def raw_xy_swept_pair_clearances(actors, *, frame_boxes, identities, cutoff_s, horizon_s,
                                 sample_dt_s=.1, contact_margin_m=0., coverage=None, selected_paths=None,
                                 pose_cache=None):
    """Local XY-only substitution; dimensions, timestamps and events mirror swept_path."""
    lookup = _RawBoxLookup(frame_boxes, identities, cutoff_s, pose_cache)
    candidates = []
    for actor in actors:
        paths = [(path, *motion) for path in actor.predictions
                 if (motion := _raw_xy_pose(actor, path, horizon_s, lookup)) is not None]
        if paths: candidates.append((actor, paths, actor_footprint_m(actor)))
    result = []
    for i, (a, paths_a, ea) in enumerate(candidates):
        for b, paths_b, eb in candidates[i + 1:]:
            best = None
            for path_a, pose_a, avail_a, gaps_a in paths_a:
                for path_b, pose_b, avail_b, gaps_b in paths_b:
                    available = min(avail_a, avail_b, horizon_s)
                    if available < sample_dt_s - 1e-9: continue
                    future = [step * sample_dt_s for step in range(1, int(math.floor(available / sample_dt_s + 1e-9)) + 1)]
                    if any(lookup(x.actor_id, t) is None for x in (a, b) for t in future):
                        lookup.future_missing_path_pairs.add((a.actor_id, b.actor_id, id(path_a), id(path_b)))
                    aa, bb = pose_a(0.), pose_b(0.)
                    if aa is None or bb is None:
                        lookup.skipped_t0_pairs.add((a.actor_id, b.actor_id)); continue
                    now_gap = swept_path._footprint_clearance(aa, ea, bb, eb)
                    samples, missing = [], []
                    for t in future:
                        aa, bb = pose_a(t), pose_b(t)
                        if aa is None or bb is None:
                            missing.append((max(0., t - sample_dt_s / 2), min(available, t + sample_dt_s / 2)))
                        else: samples.append((t, swept_path._footprint_clearance(aa, ea, bb, eb)))
                    if not samples: continue
                    row = (min(g for _t, g in samples), min(samples, key=lambda x: x[1])[0], now_gap,
                           path_a, path_b, samples, available, [*gaps_a, *gaps_b, *missing], bool(missing))
                    if best is None or row[:2] < best[:2]: best = row
            if best is None: continue
            gap, at, now_gap, pa, pb, samples, available, gaps, missing = best
            before, history_gapped = _history_before(a, ea, b, eb, lookup, contact_margin_m)
            event, first = _contact_event(now_gap <= contact_margin_m, before, history_gapped,
                [(t, g <= contact_margin_m) for t, g in samples], available >= horizon_s - 1e-9 and not missing, gaps)
            predicted = event in {"NEW_CONTACT", "BOUNDARY_ONSET", "RECONTACT"}
            if all(abs(x.speed_mps or 0.) <= swept_path.STATIONARY_MPS for x in (a, b)) and not predicted: continue
            contacts = [t for t, g in samples if g <= contact_margin_m]
            at_1 = min(samples, key=lambda x: abs(x[0] - min(1., samples[-1][0])))[1]
            pair = SweptPair(a.actor_id,b.actor_id,gap,at,1 if event == "BOUNDARY_ONSET" else max(1,int(math.ceil((first if first is not None else at)-1e-9)),),predicted,pa.probability*pb.probability,pa.maneuver,pb.maneuver,now_gap,first,len(contacts)*sample_dt_s,at_1,samples[-1][1],event,now_gap<=contact_margin_m,before)
            result.append(pair)
            if selected_paths is not None: selected_paths[frozenset((a.actor_id,b.actor_id))] = (pa,pb)
    if coverage is not None: coverage.update(lookup.report())
    return sorted(result, key=lambda p:(p.minimum_clearance_m,p.time_after_observation_s,p.actor_a,p.actor_b))


def _target_hit(pairs, target, identities):
    return len(target) == 2 and any((target[0] in identities.get(a,set()) and target[1] in identities.get(b,set())) or (target[1] in identities.get(a,set()) and target[0] in identities.get(b,set())) for a,b in map(tuple,pairs))


def audit_one(root, row, conf, maps, raw_scenario, *, legacy_replay=False, verify_replay_equivalence=False):
    total_start = time.perf_counter()
    groups = state_audit.gt_groups(Path(conf["gt_run"]) / row["directory"])
    replay = replay_windows_profiled(root, row, conf, maps, legacy_replay=legacy_replay)
    if verify_replay_equivalence:
        legacy = replay if legacy_replay else replay_windows_profiled(root, row, conf, maps, legacy_replay=True)
        lightweight = replay_windows_profiled(root, row, conf, maps, legacy_replay=False) if legacy_replay else replay
        _assert_replay_equivalent(legacy, lightweight)
    windows, agent_ids, wc, replay_profile = replay
    replay_s = replay_profile["replay_total_s"]
    # Loading is intentionally lazy: most scenarios do not need every raw 10-Hz
    # frame.  The cache is shared across all windows and both conditions.
    frame_boxes = _FrameBoxCache(raw_scenario)
    pose_cache, records = {}, []
    geometry_start = time.perf_counter()
    for (label,cutoff), expected_rows in groups.items():
        win=windows.get(label)
        if not win: raise KeyError(f"missing replay window {row['scenario']}/{label}")
        selected,_=state_audit.rank_actors(win.last,wc.actor_cap(len(win.last.actors))); ids=heading_audit.identities(win,agent_ids)
        prod_paths={}; prod=heading_audit.raw_yaw_swept_pair_clearances(selected,frame_boxes=frame_boxes,identities_by_actor=ids,cutoff_s=cutoff,horizon_s=float(conf['horizon_s']),selected_paths=prod_paths)
        coverage={}; raw_paths={}; raw=raw_xy_swept_pair_clearances(selected,frame_boxes=frame_boxes,identities=ids,cutoff_s=cutoff,horizon_s=float(conf['horizon_s']),coverage=coverage,selected_paths=raw_paths,pose_cache=pose_cache)
        by={PRODUCTION_GT_XY_RAW_YAW:{frozenset((p.actor_a,p.actor_b)):p for p in prod if p.predicted_contact},RAW_CARLA_XY_RAW_YAW:{frozenset((p.actor_a,p.actor_b)):p for p in raw if p.predicted_contact}}
        for e in expected_rows:
            bucket=int(e['k']); pairs={c:{pair:p for pair,p in ps.items() if p.interval_index==bucket} for c,ps in by.items()}; target=tuple(map(int,e.get('involved_carla_ids') or []))
            records.append({'scenario_type':row['scenario_type'],'scenario':row['scenario'],'window':label,'bucket':bucket,'cutoff_s':cutoff,'gt_positive':bool(e['accident_expected']),'pairs':pairs,'hits':{c:_target_hit(ps,target,ids) for c,ps in pairs.items()},'identities':ids,'actors':{a.actor_id:a for a in selected},'frame_boxes':frame_boxes,'coverage':coverage,'selected_paths':{PRODUCTION_GT_XY_RAW_YAW:prod_paths,RAW_CARLA_XY_RAW_YAW:raw_paths},'actor_pair_universe':_actor_pair_universe(selected)})
    total_s = time.perf_counter() - total_start
    timing = {"scenario": row["scenario"], "replay_s": replay_s, "replay_profile_s": replay_profile,
              "raw_box_loading_s": frame_boxes.load_s,
              # Raw loading is part of the geometry phase; subtract it here so
              # the two reported components sum to the total useful work.
              "geometry_s": time.perf_counter() - geometry_start - frame_boxes.load_s,
              "total_s": total_s, "raw_frames_loaded": len(frame_boxes.loaded),
              "raw_pose_cache_entries": len(pose_cache)}
    return records, timing


def summarize(records, condition):
    c,ws,ss=Counter(),set(),set(); cov=Counter(); seen=set()
    for r in records:
        key=r['scenario_type'],r['scenario'],r['window']
        if condition == RAW_CARLA_XY_RAW_YAW and key not in seen:
            seen.add(key); cov.update({k:r['coverage'][k] for k in ('raw_xy_queries','raw_xy_available','raw_xy_missing','actor_pairs_skipped_missing_raw_xy_at_t0','path_pairs_with_any_missing_future_raw_xy')})
        ps,pos=r['pairs'][condition],r['gt_positive']
        if pos: c['positive']+=1;c['hits']+=int(r['hits'][condition])
        if ps and pos:c['TP']+=1
        elif ps:c['FP']+=1;c['FP_contact_pair_rows']+=len(ps);ws.add(key);ss.add(key[:2])
        elif pos:c['FN']+=1
        else:c['TN']+=1
    out={'n_buckets':len(records),'positive_GT_buckets':c['positive'],'TP':c['TP'],'FP':c['FP'],'TN':c['TN'],'FN':c['FN'],'precision':None if not c['TP']+c['FP'] else c['TP']/(c['TP']+c['FP']),'recall':None if not c['positive'] else c['TP']/c['positive'],'GT_actor_pair_hits':c['hits'],'GT_actor_pair_recall':None if not c['positive'] else c['hits']/c['positive'],'FP_contact_pair_rows':c['FP_contact_pair_rows'],'windows_with_FP_contact':len(ws),'scenarios_with_FP_contact':len(ss)}
    if condition==RAW_CARLA_XY_RAW_YAW:
        out.update(cov); out['raw_xy_coverage']=None if not cov['raw_xy_queries'] else cov['raw_xy_available']/cov['raw_xy_queries']
    return out


def _keys(records,cond,pred): return {(r['scenario_type'],r['scenario'],r['window'],r['bucket']) for r in records if pred(r,bool(r['pairs'][cond]),r['hits'][cond])}
def pairwise(records):
    a,b=PRODUCTION_GT_XY_RAW_YAW,RAW_CARLA_XY_RAW_YAW
    af,bf=_keys(records,a,lambda r,p,h:not r['gt_positive'] and p),_keys(records,b,lambda r,p,h:not r['gt_positive'] and p); at,bt=_keys(records,a,lambda r,p,h:r['gt_positive'] and p),_keys(records,b,lambda r,p,h:r['gt_positive'] and p); ah,bh=_keys(records,a,lambda r,p,h:r['gt_positive'] and h),_keys(records,b,lambda r,p,h:r['gt_positive'] and h)
    return {'FP removed':len(af-bf),'FP introduced':len(bf-af),'TP lost':len(at-bt),'TP gained':len(bt-at),'GT-pair hits lost':len(ah-bh),'GT-pair hits gained':len(bh-ah)}


def _times_to(t): return [i*.1 for i in range(int(math.floor(max(0.,t)/.1+1e-9))+1)]
def _box(r,aid,t): return _RawBoxLookup(r['frame_boxes'],r['identities'],r['cutoff_s'])(aid,t)
def _history_times(actor):
    h=sorted(actor.track_history or []); now=h[-1][0] if h else 0.; return [t-now for t,_x,_y in h if -LOOKBACK_S-1e-9<=t-now<-1e-9]
def _xy_pose(r,aid,t,path):
    actor=r['actors'][aid]; motion=swept_path._motion_fn(actor,path,max(.1,t)) if path else None; prod=motion[0](max(0.,t)) if motion else None; box=_box(r,aid,t)
    return {'production_gt_xy':None if prod is None else [prod[0],prod[1]],'raw_carla_xy':None if box is None else list(box.center),'xy_error_m':None if prod is None or box is None else math.dist(prod[:2],box.center),'raw_carla_heading_enu':None if box is None else list(box.heading)}
def _removed_category(removed,complete): return None if not removed else ('FP_removed_with_complete_raw_xy' if complete else 'FP_removed_with_missing_raw_xy')


def _assert_production_baseline(summary):
    if {k: summary[k] for k in EXPECTED} != EXPECTED:
        raise RuntimeError(f'PRODUCTION_GT_XY_RAW_YAW is not faithful: {summary}')


def changed_pairs(records):
    out=[]
    for r in records:
        a,b=r['pairs'][PRODUCTION_GT_XY_RAW_YAW],r['pairs'][RAW_CARLA_XY_RAW_YAW]
        for pair in set(a)^set(b):
            pa,pb=a.get(pair),b.get(pair); p=pa or pb;t=p.first_contact_s if p.first_contact_s is not None else 0.; aids=sorted(pair); missing={aid:[x for x in _times_to(t) if _box(r,aid,x) is None] for aid in aids}; complete=not any(missing.values()); history_required=any(x and x.contact_event in {'BOUNDARY_ONSET','RECONTACT'} for x in (pa,pb)); history_missing={aid:[x for x in _history_times(r['actors'][aid]) if _box(r,aid,x) is None] for aid in aids}; complete_history=not any(history_missing.values()) if history_required else None; removed=not r['gt_positive'] and pa is not None and pb is None
            paths=r['selected_paths'][PRODUCTION_GT_XY_RAW_YAW].get(pair) or r['selected_paths'][RAW_CARLA_XY_RAW_YAW].get(pair,(None,None))
            out.append({'scenario':r['scenario'],'window':r['window'],'bucket':r['bucket'],'GT_positive':r['gt_positive'],'production_actor_ids':aids,'source_carla_ids':[sorted(r['identities'].get(x,set())) for x in aids],'PRODUCTION_GT_XY_RAW_YAW_event':None if pa is None else pa.contact_event,'PRODUCTION_GT_XY_RAW_YAW_first_contact_time_s':None if pa is None else pa.first_contact_s,'RAW_CARLA_XY_RAW_YAW_event':None if pb is None else pb.contact_event,'RAW_CARLA_XY_RAW_YAW_first_contact_time_s':None if pb is None else pb.first_contact_s,'relevant_contact_time_s':t,'actors':[_xy_pose(r,aid,t,path) for aid,path in zip(aids,paths)],'raw_xy_complete_to_relevant_contact':complete,'missing_raw_xy_times_s':missing,'raw_xy_complete_history':complete_history,'missing_raw_xy_history_times_s':history_missing if history_required else None,'FP_removed_with_complete_raw_xy':removed and complete and complete_history is not False,'FP_removed_with_missing_raw_xy':removed and not (complete and complete_history is not False),'FP_introduced':not r['gt_positive'] and pb is not None and pa is None})
    return out


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',required=True);ap.add_argument('--carla-maps',required=True);ap.add_argument('--gt-run',default='out/experiments/gt_future_waypointnet_0m_event_val104');ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--legacy-replay',action='store_true',help='force the original TorchPredictor replay (the safe default)');ap.add_argument('--lightweight-replay',action='store_true',help='use the no-model GT-oracle replay after equivalence validation');ap.add_argument('--verify-replay-equivalence',action='store_true',help='run both replay paths and compare every audit-relevant state');ap.add_argument('--max-scenarios',type=int,help='limit scenarios, for profiling/smoke checks only');ap.add_argument('--scenario',help='run the unique cohort row with this scenario name');ap.add_argument('--directory',help='run the cohort row whose directory exactly matches this value');args=ap.parse_args(argv)
    if args.legacy_replay and args.lightweight_replay: raise ValueError('--legacy-replay and --lightweight-replay are mutually exclusive')
    legacy_replay = not args.lightweight_replay
    run=Path(args.gt_run); cohort=json.loads((run/'batch_manifest.json').read_text());conf=dict(cohort['config'],gt_run=str(run)); scenarios={raw_audit.scenario_key(s):s for s in state_audit.scan_scenarios(args.root)}; missing=[r['scenario'] for r in cohort['scenarios'] if raw_audit.scenario_row_key(r) not in scenarios]
    if missing: raise KeyError(f'raw scenarios not found: {missing[:10]}')
    if args.directory is not None:
        scenario_rows = [row for row in cohort['scenarios'] if row['directory'] == args.directory]
        if not scenario_rows:
            raise KeyError(f'--directory not found in cohort: {args.directory}')
        if len(scenario_rows) != 1:
            raise KeyError(f'--directory is not unique in cohort: {args.directory}')
    elif args.scenario is not None:
        scenario_rows = [row for row in cohort['scenarios'] if row['scenario'] == args.scenario]
        if not scenario_rows:
            raise KeyError(f'--scenario not found in cohort: {args.scenario}')
        if len(scenario_rows) != 1:
            matches = [f"{row['directory']} ({row['scenario_type']})" for row in scenario_rows]
            raise KeyError(f"--scenario matches multiple cohort rows: {args.scenario}; use --directory. Matches: {matches}")
    else:
        scenario_rows = cohort['scenarios'][:args.max_scenarios] if args.max_scenarios is not None else cohort['scenarios']
    if args.scenario is None and args.directory is None and args.max_scenarios is not None and args.max_scenarios <= 0: raise ValueError('--max-scenarios must be positive')
    records=[]; timings=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fs={pool.submit(audit_one,args.root,r,conf,args.carla_maps,scenarios[raw_audit.scenario_row_key(r)],legacy_replay=legacy_replay,verify_replay_equivalence=args.verify_replay_equivalence):r for r in scenario_rows}
        for i,f in enumerate(as_completed(fs),1):
            scenario_records, timing = f.result(); records.extend(scenario_records); timings.append(timing)
            profile = ' '.join(f'{key}={value:.2f}' for key, value in timing['replay_profile_s'].items())
            print(f'[{i}/{len(fs)}] complete {fs[f]["scenario"]} {profile} raw_box_loading_s={timing["raw_box_loading_s"]:.2f} geometry_s={timing["geometry_s"]:.2f} total_s={timing["total_s"]:.2f}',flush=True)
    summaries={c:summarize(records,c) for c in (PRODUCTION_GT_XY_RAW_YAW,RAW_CARLA_XY_RAW_YAW)}
    if args.max_scenarios is None and args.scenario is None and args.directory is None:
        _assert_production_baseline(summaries[PRODUCTION_GT_XY_RAW_YAW])
    changed=changed_pairs(records); event={kind:{e:0 for e in ('NEW_CONTACT','BOUNDARY_ONSET','RECONTACT')} for kind in ('FP_removed','FP_introduced')}
    for x in changed:
        if x['FP_removed_with_complete_raw_xy'] or x['FP_removed_with_missing_raw_xy']:
            e=x['PRODUCTION_GT_XY_RAW_YAW_event']; event['FP_removed'][e]=event['FP_removed'].get(e,0)+1
        if x['FP_introduced']:
            e=x['RAW_CARLA_XY_RAW_YAW_event']; event['FP_introduced'][e]=event['FP_introduced'].get(e,0)+1
    errors=sorted(v['xy_error_m'] for x in changed for v in x['actors']
                  if (x['FP_removed_with_complete_raw_xy'] or x['FP_removed_with_missing_raw_xy']
                      or x['FP_introduced']) and v['xy_error_m'] is not None); q=lambda p:None if not errors else errors[min(len(errors)-1,int(math.ceil(p*len(errors))-1))]
    runtimes=sorted(t['total_s'] for t in timings)
    runtime_summary={'n_scenarios':len(timings),'mean_s':None if not timings else sum(runtimes)/len(runtimes),'median_s':None if not runtimes else runtimes[(len(runtimes)-1)//2] if len(runtimes)%2 else (runtimes[len(runtimes)//2-1]+runtimes[len(runtimes)//2])/2}
    doc={'conditions':summaries,'pairwise':{f'{PRODUCTION_GT_XY_RAW_YAW} -> {RAW_CARLA_XY_RAW_YAW}':pairwise(records)},'changed_FP_pairs_by_event_type':event,'FP_removed_with_complete_raw_xy':sum(x['FP_removed_with_complete_raw_xy'] for x in changed),'FP_removed_with_missing_raw_xy':sum(x['FP_removed_with_missing_raw_xy'] for x in changed),'changed_pair_xy_error_m':{'median':q(.5),'p95':q(.95),'maximum':None if not errors else errors[-1]},'scenario_timing':timings,'aggregate_scenario_runtime_s':runtime_summary,'contact_lookback_s':LOOKBACK_S,'sample_dt_s':.1,'margin_m':0.}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(doc,indent=2)+'\n');(out/'xy_source_changed_pairs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in changed));print(json.dumps(doc,indent=2))
if __name__=='__main__':raise SystemExit(main())
