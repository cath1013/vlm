#!/usr/bin/env python3
"""Evaluation-only state-source ablation for a saved GT-oracle run.

RAW_STATE reads raw CARLA 10 Hz boxes. PRODUCTION_STATE replays the normal
online pipeline at full precision, substitutes only GT future coordinates, and
uses swept_path directly without candidate caps or re-ranking.
"""
from __future__ import annotations
import argparse, itertools, json, math, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples import audit_gt_carla_boxes as raw_audit  # noqa: E402
from traffic_llm.accident_qa import WindowConfig, build_actor_lookup, build_windows, ground_truth_future_snapshots  # noqa: E402
from traffic_llm.carla_map import find_xodr  # noqa: E402
from traffic_llm.config import DEFAULT_CLASS_SIZES, PipelineConfig  # noqa: E402
from traffic_llm.da_runner import DeepAccidentRunner  # noqa: E402
from traffic_llm.deepaccident import CLASS_MAP, estimate_collision, scan_scenarios  # noqa: E402
from traffic_llm.predict_model import TorchPredictor  # noqa: E402
from traffic_llm.serialize import rank_actors  # noqa: E402
from traffic_llm.swept_path import _contact_event, swept_pair_clearances  # noqa: E402

RAW_STATE, PRODUCTION_STATE_ALL_PAIRS, PRODUCTION_STATE_SAVED_VIEW, LOOKBACK_S = (
    "RAW_STATE", "PRODUCTION_STATE_ALL_PAIRS", "PRODUCTION_STATE_SAVED_VIEW", 0.6,
)
EXPECTED = {"TP": 52, "FP": 193, "TN": 362, "FN": 17, "GT_actor_pair_hits": 41}
EXPECTED_DIAGNOSTICS = {"FP_contact_pair_rows": 326, "windows_with_FP_contact": 150, "scenarios_with_FP_contact": 47}

def raw_time(frame): return (frame - 1) / raw_audit.FRAME_RATE_HZ
def raw_contact(a, b):
    sa = DEFAULT_CLASS_SIZES[CLASS_MAP.get(a.cls, a.cls)]
    sb = DEFAULT_CLASS_SIZES[CLASS_MAP.get(b.cls, b.cls)]
    return raw_audit.polygon_clearance(raw_audit.oriented_footprint(a.center, a.heading, sa.length_m, sa.width_m), raw_audit.oriented_footprint(b.center, b.heading, sb.length_m, sb.width_m)) <= 1e-9
def raw_event(boxes, frames, pair, cutoff, cutoff_frame, horizon):
    samples = [{"frame": f, "time_s": raw_time(f) - cutoff, "contact": raw_contact(boxes[f][pair[0]], boxes[f][pair[1]]) if set(pair) <= set(boxes[f]) else None} for f in frames]
    index = next(i for i, row in enumerate(samples) if row["frame"] == cutoff_frame)
    now, before, future = samples[index]["contact"], samples[:index], samples[index + 1:]
    if now is None:
        return {"event": "RECONTACT_UNCERTAIN", "time_s": None}
    valid_before = [row for row in before if row["contact"] is not None]
    latest = valid_before[-1] if valid_before else None
    gapped = latest is None or any(row["contact"] is None and row["time_s"] > latest["time_s"] + 1e-9 for row in before)
    gaps, gap_start = [], None
    for row in future:
        if row["contact"] is None and gap_start is None: gap_start = row["time_s"]
        if row["contact"] is not None and gap_start is not None:
            gaps.append((gap_start, row["time_s"])); gap_start = None
    if gap_start is not None: gaps.append((gap_start, horizon))
    event, at = _contact_event(now, None if latest is None else latest["contact"], gapped,
                               [(row["time_s"], row["contact"]) for row in future if row["contact"] is not None],
                               bool(future and future[-1]["time_s"] >= horizon - 1e-9), gaps)
    return {"event": event, "time_s": at}
def pairs_in_bucket(events, start, end):
    return {p: e for p, e in events.items() if (e["event"] == "BOUNDARY_ONSET" and math.isclose(start, 0.0, abs_tol=1e-9)) or (e["time_s"] is not None and raw_audit.in_interval(e["time_s"], start, end))}
def gt_groups(gt_dir):
    out = defaultdict(list)
    for label, cutoff, expected in raw_audit._scenario_buckets(gt_dir): out[(label, cutoff)].append(expected)
    return out

def replay_windows(root, row, conf, maps):
    """Reconstruct full-precision online ActorStates, then only replace futures."""
    cfg = PipelineConfig(); cfg.deepaccident.observation_mode = conf["mode"]
    cfg.predictor = TorchPredictor(conf["predictor"], mode=conf.get("predictor_mode") or "waypoints")
    runner = DeepAccidentRunner(root, cfg); xodr = find_xodr(row["town"], [maps])
    if not xodr: raise FileNotFoundError(f"OpenDRIVE map for {row['town']} not found below {maps}")
    built = runner.build(row["scenario"], row["scenario_type"], opendrive_path=xodr)
    online = list(built.snapshots(rate_hz=2.0))
    raw = list(built.snapshots(rate_hz=cfg.deepaccident.frame_rate_hz))
    oracle = ground_truth_future_snapshots(online, float(conf["horizon_s"]), raw_snapshots=raw)
    wc = WindowConfig(window_s=float(conf["window_s"]), stride_s=float(conf["stride_s"]), horizon_s=float(conf["horizon_s"]), snapshot_rate_hz=2.0, history_stride_s=float(conf["history_stride_s"]), warmup=False, full_window=False, future_source="ground_truth", swept_sample_dt_s=.1, swept_contact_margin_m=0.)
    collision = estimate_collision(built.scenario, cfg.deepaccident)
    windows, _ = build_windows(oracle, wc, collision_time_s=collision.time_s if collision and collision.occurred else None)
    agent_ids = {a: built.scenario.meta.agent_id_of(a) for a in built.scenario.agents}
    return {w.label: w for w in windows}, agent_ids, wc

def identities(window, agent_ids):
    result = defaultdict(set)
    for cid, aids in build_actor_lookup(window, agent_ids).items():
        for aid in aids: result[aid].add(int(cid))
    return result

def audit_one(root, row, conf, maps, raw_scenario):
    gt_dir = Path(conf["gt_run"]) / row["directory"]; groups = gt_groups(gt_dir)
    windows, agent_ids, wc = replay_windows(root, row, conf, maps)
    frames = sorted({f for series in raw_scenario.agents.values() for f in series.frames})
    boxes = {f: raw_audit.merge_frame_boxes(raw_scenario, f)[0] for f in frames}; records = []
    for (label, cutoff), expected_rows in groups.items():
        win = windows.get(label)
        if not win: raise KeyError(f"missing replay window {row['scenario']}/{label}")
        horizon = float(conf["horizon_s"]); cutoff_frame = int(round(cutoff * raw_audit.FRAME_RATE_HZ)) + 1
        raw_frames = [f for f in frames if -LOOKBACK_S - 1e-9 <= raw_time(f) - cutoff <= horizon + 1e-9]
        candidates = set().union(*(set(itertools.combinations(boxes[f], 2)) for f in raw_frames))
        raw_events = {frozenset(p): raw_event(boxes, raw_frames, p, cutoff, cutoff_frame, horizon) for p in candidates}
        # Match compact_window_json() selection while retaining the original,
        # full-precision ActorState objects for all geometry.
        selected_actors, _ = rank_actors(win.last, wc.actor_cap(len(win.last.actors)))
        # The all-pairs view has no pair cap or re-ranker after actor selection.
        swept = swept_pair_clearances(selected_actors, horizon_s=horizon, sample_dt_s=.1, contact_margin_m=0., exclude_touching_now=True, exclude_static_pairs=True, contact_lookback_s=LOOKBACK_S)
        candidate_pool = swept[:wc.swept_candidate_pool_cap]
        saved_view = candidate_pool[:wc.swept_pair_cap]
        prod = {frozenset((p.actor_a, p.actor_b)): p for p in swept if p.predicted_contact}
        saved = {frozenset((p.actor_a, p.actor_b)): p for p in saved_view if p.predicted_contact}
        actor_ids = identities(win, agent_ids); actors = {a.actor_id: a for a in win.last.actors}
        for e in expected_rows:
            start, end = float(e["interval_start_s"]) - cutoff, float(e["interval_end_s"]) - cutoff
            rp = pairs_in_bucket(raw_events, start, end); pp = {pair: p for pair, p in prod.items() if p.interval_index == int(e["k"])}
            target_carla_ids = frozenset(map(int, e.get("involved_carla_ids") or []))
            def target_hit(predicted_pairs):
                return len(target_carla_ids) == 2 and any(
                (tuple(target_carla_ids)[0] in actor_ids.get(aid_a, set())
                 and tuple(target_carla_ids)[1] in actor_ids.get(aid_b, set()))
                or (tuple(target_carla_ids)[1] in actor_ids.get(aid_a, set())
                    and tuple(target_carla_ids)[0] in actor_ids.get(aid_b, set()))
                for aid_a, aid_b in (tuple(pair) for pair in predicted_pairs)
                )
            saved_pp = {pair: p for pair, p in saved.items() if p.interval_index == int(e["k"])}
            legacy_target = frozenset(e.get("involved_actor_ids") or [])
            legacy_saved_hit = len(legacy_target) >= 2 and legacy_target in saved_pp
            records.append({"scenario_type": row["scenario_type"], "scenario": row["scenario"], "window": label, "bucket": int(e["k"]), "gt_positive": bool(e["accident_expected"]), "raw": rp, "production": pp, "saved_view": saved_pp, "raw_hit": raw_audit.target_pair_hit(rp, e.get("involved_carla_ids") or []), "production_hit": target_hit(pp), "saved_view_hit": target_hit(saved_pp), "legacy_saved_hit": legacy_saved_hit, "identities": actor_ids, "actors": actors, "cutoff_boxes": boxes.get(cutoff_frame, {})})
    return records

def summarize(records, state):
    c, ws, ss = Counter(), set(), set()
    for r in records:
        pairs = r["raw"] if state == RAW_STATE else r["production"] if state == PRODUCTION_STATE_ALL_PAIRS else r["saved_view"]
        hit = r["raw_hit"] if state == RAW_STATE else r["production_hit"] if state == PRODUCTION_STATE_ALL_PAIRS else r["saved_view_hit"]
        pred, pos = bool(pairs), r["gt_positive"]
        if pos:
            c["positive"] += 1
            c["GT_actor_pair_hits_carla_identity"] += int(hit)
            if state == PRODUCTION_STATE_SAVED_VIEW: c["GT_actor_pair_hits_legacy"] += int(r["legacy_saved_hit"])
        if pred and pos: c["TP"] += 1
        elif pred: c["FP"] += 1; c["FP_contact_pair_rows"] += len(pairs); ws.add((r["scenario_type"], r["scenario"], r["window"])); ss.add((r["scenario_type"], r["scenario"]))
        elif pos: c["FN"] += 1
        else: c["TN"] += 1
    result = {"n_buckets": len(records), "positive_GT_buckets": c["positive"], "TP": c["TP"], "FP": c["FP"], "TN": c["TN"], "FN": c["FN"], "precision": None if not c["TP"] + c["FP"] else c["TP"] / (c["TP"] + c["FP"]), "recall": c["TP"] / c["positive"], "GT_actor_pair_hits_carla_identity": c["GT_actor_pair_hits_carla_identity"], "GT_actor_pair_recall_carla_identity": c["GT_actor_pair_hits_carla_identity"] / c["positive"], "FP_contact_pair_rows": c["FP_contact_pair_rows"], "windows_with_FP_contact": len(ws), "scenarios_with_FP_contact": len(ss)}
    if state == PRODUCTION_STATE_SAVED_VIEW:
        result["GT_actor_pair_hits_legacy"] = c["GT_actor_pair_hits_legacy"]
        result["GT_actor_pair_recall_legacy"] = c["GT_actor_pair_hits_legacy"] / c["positive"]
    return result

def bkeys(records, state, selector):
    return {(r["scenario_type"], r["scenario"], r["window"], r["bucket"]) for r in records if selector(r, bool(r["raw"] if state == RAW_STATE else r["production"] if state == PRODUCTION_STATE_ALL_PAIRS else r["saved_view"]), r["raw_hit"] if state == RAW_STATE else r["production_hit"] if state == PRODUCTION_STATE_ALL_PAIRS else r["saved_view_hit"])}
def changes(records):
    a, b = RAW_STATE, PRODUCTION_STATE_ALL_PAIRS
    af, bf = bkeys(records,a,lambda r,p,h:not r["gt_positive"] and p), bkeys(records,b,lambda r,p,h:not r["gt_positive"] and p)
    at, bt = bkeys(records,a,lambda r,p,h:r["gt_positive"] and p), bkeys(records,b,lambda r,p,h:r["gt_positive"] and p)
    ah, bh = bkeys(records,a,lambda r,p,h:r["gt_positive"] and h), bkeys(records,b,lambda r,p,h:r["gt_positive"] and h)
    first = {"FP removed":len(af-bf),"FP introduced":len(bf-af),"TP lost":len(at-bt),"TP gained":len(bt-at),"GT-pair hits lost":len(ah-bh),"GT-pair hits gained":len(bh-ah)}
    a, b = PRODUCTION_STATE_ALL_PAIRS, PRODUCTION_STATE_SAVED_VIEW
    af, bf = bkeys(records,a,lambda r,p,h:not r["gt_positive"] and p), bkeys(records,b,lambda r,p,h:not r["gt_positive"] and p)
    at, bt = bkeys(records,a,lambda r,p,h:r["gt_positive"] and p), bkeys(records,b,lambda r,p,h:r["gt_positive"] and p)
    ah, bh = bkeys(records,a,lambda r,p,h:r["gt_positive"] and h), bkeys(records,b,lambda r,p,h:r["gt_positive"] and h)
    second = {"FP removed":len(af-bf),"FP introduced":len(bf-af),"TP lost":len(at-bt),"TP gained":len(bt-at),"GT-pair hits lost":len(ah-bh),"GT-pair hits gained":len(bh-ah)}
    return {"RAW_STATE -> PRODUCTION_STATE_ALL_PAIRS": first, "PRODUCTION_STATE_ALL_PAIRS -> PRODUCTION_STATE_SAVED_VIEW": second}

def counterpart(aid, row):
    actor, boxes = row["actors"][aid], row["cutoff_boxes"]; exact = [boxes[c] for c in row["identities"].get(aid, set()) if c in boxes]
    if exact: return min(exact, key=lambda b: math.dist(b.center, actor.world_xy)), "source_track"
    same = [b for b in boxes.values() if CLASS_MAP.get(b.cls, b.cls) == actor.cls]
    return (min(same, key=lambda b: math.dist(b.center, actor.world_xy)) if same else None), "nearest_fallback"
def introduced_rows(records):
    output = []
    for r in records:
        if r["gt_positive"]: continue
        for pair, p in r["production"].items():
            aids = sorted(pair); matches = [counterpart(a, r) for a in aids]; boxes = [x[0] for x in matches]; raw_pair = frozenset(b.carla_id for b in boxes if b)
            if len(raw_pair) == 2 and raw_pair in r["raw"]: continue
            actors = [r["actors"][a] for a in aids]
            output.append({"scenario":r["scenario"],"window":r["window"],"bucket":r["bucket"],"actor_pair":aids,"event":p.contact_event,"first_contact_time_s":p.first_contact_s,"raw_matched_actor_pair":sorted(raw_pair) if len(raw_pair)==2 else None,"identity_match":"source_track" if all(k=="source_track" for _,k in matches) else "nearest_fallback","raw_current_positions_m":[None if b is None else list(b.center) for b in boxes],"production_current_positions_m":[list(a.world_xy) for a in actors],"current_position_error_m":[None if b is None else math.dist(b.center,a.world_xy) for a,b in zip(actors,boxes)],"raw_yaw_deg":[None if b is None else math.degrees(math.atan2(b.heading[1],b.heading[0])) for b in boxes],"production_heading_deg":[a.heading_deg for a in actors]})
    return output

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--root",required=True); ap.add_argument("--carla-maps",required=True); ap.add_argument("--gt-run",default="out/experiments/gt_future_waypointnet_0m_event_val104"); ap.add_argument("--out",required=True); ap.add_argument("--workers",type=int,default=1); ap.add_argument("--faithfulness-max-delta",type=int,default=5)
    args=ap.parse_args(argv); run=Path(args.gt_run); cohort=json.loads((run/"batch_manifest.json").read_text()); conf=dict(cohort["config"],gt_run=str(run))
    scenarios={raw_audit.scenario_key(s):s for s in scan_scenarios(args.root)}; missing=[r["scenario"] for r in cohort["scenarios"] if raw_audit.scenario_row_key(r) not in scenarios]
    if missing: raise KeyError(f"raw scenarios not found: {missing[:10]}")
    records=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fs={pool.submit(audit_one,args.root,r,conf,args.carla_maps,scenarios[raw_audit.scenario_row_key(r)]):r for r in cohort["scenarios"]}
        for i,f in enumerate(as_completed(fs),1): records.extend(f.result()); print(f"[{i}/{len(fs)}] complete {fs[f]['scenario']}",flush=True)
    summaries={RAW_STATE:summarize(records,RAW_STATE),PRODUCTION_STATE_ALL_PAIRS:summarize(records,PRODUCTION_STATE_ALL_PAIRS),PRODUCTION_STATE_SAVED_VIEW:summarize(records,PRODUCTION_STATE_SAVED_VIEW)}
    for s in summaries.values():
        if s["n_buckets"]!=624 or s["positive_GT_buckets"]!=69: raise RuntimeError(f"unexpected GT bucket set: {s}")
    delta={k:(summaries[PRODUCTION_STATE_SAVED_VIEW]["GT_actor_pair_hits_legacy"] if k=="GT_actor_pair_hits" else summaries[PRODUCTION_STATE_SAVED_VIEW][k])-v for k,v in EXPECTED.items()}
    diagnostic_delta={k:summaries[PRODUCTION_STATE_SAVED_VIEW][k]-v for k,v in EXPECTED_DIAGNOSTICS.items()}
    faithful=all(abs(v)<=args.faithfulness_max_delta for v in delta.values())
    doc={"conditions":summaries,"pairwise":changes(records),"expected_saved_run":EXPECTED,"reconstructed_production_minus_expected":delta,"expected_saved_run_diagnostics":EXPECTED_DIAGNOSTICS,"reconstructed_production_diagnostic_deltas":diagnostic_delta,"faithful_enough":faithful,"contact_lookback_s":LOOKBACK_S,"sample_dt_s":.1,"margin_m":0.}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/"summary.json").write_text(json.dumps(doc,indent=2)+"\n");(out/"state_source_changed_fp.jsonl").write_text("".join(json.dumps(x)+"\n" for x in introduced_rows(records)));print(json.dumps(doc,indent=2))
    if not faithful: raise RuntimeError("reconstructed production state is too far from saved-run metrics; state-source interpretation is not faithful enough")
if __name__=="__main__": raise SystemExit(main())
