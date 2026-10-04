#!/usr/bin/env python3
"""Predictor-XY contact attribution against the exact-geometry GT trajectory oracle.

The predictor paths are never replaced in ``PREDICTED_XY_ORACLE_GEOMETRY``.
Only raw CARLA yaw and per-instance raw CARLA dimensions are supplied to the
local 2-D swept-box evaluator.  The separate GT-path evaluation is a reference
for attribution, not an input to predictor geometry.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples import audit_gt_carla_boxes as raw_audit  # noqa: E402
from examples import audit_gt_heading_source_ablation as geometry  # noqa: E402
from examples import audit_gt_state_source_ablation as state_audit  # noqa: E402
from examples.run_deepaccident import infer_predictor_mode  # noqa: E402
from traffic_llm.accident_qa import WindowConfig, build_windows, ground_truth_future_snapshots  # noqa: E402
from traffic_llm.carla_map import find_xodr  # noqa: E402
from traffic_llm.config import PipelineConfig  # noqa: E402
from traffic_llm.da_runner import DeepAccidentRunner  # noqa: E402
from traffic_llm.deepaccident import estimate_collision  # noqa: E402
from traffic_llm.predict_model import JointSceneTorchPredictor, TorchPredictor  # noqa: E402
from traffic_llm.serialize import rank_actors  # noqa: E402

PREDICTED_XY_ORACLE_GEOMETRY = "PREDICTED_XY_ORACLE_GEOMETRY"
GT_TRAJECTORY_ORACLE_GEOMETRY = "GT_TRAJECTORY_ORACLE_GEOMETRY"
LOOKBACK_S = .6
GT_REFERENCE = {"TP": 63, "FP": 16, "TN": 539, "FN": 6, "GT_actor_pair_hits": 63}


def _predictor(path, mode, device):
    return (JointSceneTorchPredictor(path, device=device) if mode == "joint_scene"
            else TorchPredictor(path, mode=mode, device=device))


def replay_windows(root, row, conf, maps, predictor_path, predictor_mode, device):
    """Build predictor and GT-reference windows from the same observed snapshots."""
    cfg = PipelineConfig()
    cfg.deepaccident.observation_mode = conf["mode"]
    cfg.predictor = _predictor(predictor_path, predictor_mode, device)
    runner = DeepAccidentRunner(root, cfg)
    xodr = find_xodr(row["town"], [maps])
    if not xodr:
        raise FileNotFoundError(f"OpenDRIVE map for {row['town']} not found below {maps}")
    built = runner.build(row["scenario"], row["scenario_type"], opendrive_path=xodr)
    online = list(built.snapshots(rate_hz=2.0))
    raw = list(built.snapshots(rate_hz=cfg.deepaccident.frame_rate_hz))
    collision = estimate_collision(built.scenario, cfg.deepaccident)
    wc = WindowConfig(window_s=float(conf["window_s"]), stride_s=float(conf["stride_s"]),
        horizon_s=float(conf["horizon_s"]), snapshot_rate_hz=2.0,
        history_stride_s=float(conf["history_stride_s"]), warmup=False, full_window=False,
        future_source="predictor", swept_sample_dt_s=.1, swept_contact_margin_m=0.)
    predictor_windows, _ = build_windows(online, wc,
        collision_time_s=collision.time_s if collision and collision.occurred else None)
    # This deepcopy-based helper changes only this reference stream; ``online``
    # (and therefore predictor windows) retain their checkpoint outputs.
    gt_snapshots = ground_truth_future_snapshots(online, wc.horizon_s, raw_snapshots=raw)
    gt_wc = WindowConfig(**{**wc.__dict__, "future_source": "ground_truth"})
    gt_windows, _ = build_windows(gt_snapshots, gt_wc,
        collision_time_s=collision.time_s if collision and collision.occurred else None)
    agent_ids = {a: built.scenario.meta.agent_id_of(a) for a in built.scenario.agents}
    return ({w.label: w for w in predictor_windows}, {w.label: w for w in gt_windows}, agent_ids, wc)


def _target_hit(pairs, target, identities):
    return len(target) == 2 and any(
        (target[0] in identities.get(a, set()) and target[1] in identities.get(b, set())) or
        (target[1] in identities.get(a, set()) and target[0] in identities.get(b, set()))
        for a, b in (tuple(pair) for pair in pairs))


def _contact_pairs(actors, identities, frame_boxes, cutoff_s, horizon_s):
    """Use predictor/GT path XY verbatim; geometry supplies only raw yaw/size."""
    coverage, paths = {}, {}
    pairs = geometry.raw_yaw_swept_pair_clearances(
        actors, frame_boxes=frame_boxes, identities_by_actor=identities, cutoff_s=cutoff_s,
        horizon_s=horizon_s, sample_dt_s=.1, contact_margin_m=0., coverage=coverage,
        selected_paths=paths, exact_size=True)
    return {frozenset((p.actor_a, p.actor_b)): p for p in pairs if p.predicted_contact}, coverage


def audit_one(root, row, conf, maps, raw_scenario, predictor_path, predictor_mode, device):
    groups = state_audit.gt_groups(Path(conf["gt_run"]) / row["directory"])
    predicted_windows, gt_windows, agent_ids, wc = replay_windows(
        root, row, conf, maps, predictor_path, predictor_mode, device)
    frames = sorted({f for series in raw_scenario.agents.values() for f in series.frames})
    frame_boxes = {f: raw_audit.merge_frame_boxes(raw_scenario, f)[0] for f in frames}
    records = []
    for (label, cutoff), expected_rows in groups.items():
        pred_win, gt_win = predicted_windows.get(label), gt_windows.get(label)
        if pred_win is None or gt_win is None:
            raise KeyError(f"missing replay window {row['scenario']}/{label}")
        # Selection is calculated once from the production/predictor state and
        # transferred by actor id to the GT reference stream.
        selected, _ = rank_actors(pred_win.last, wc.actor_cap(len(pred_win.last.actors)))
        gt_by_id = {actor.actor_id: actor for actor in gt_win.last.actors}
        if any(actor.actor_id not in gt_by_id for actor in selected):
            raise KeyError(f"GT reference actor mismatch {row['scenario']}/{label}")
        gt_selected = [gt_by_id[actor.actor_id] for actor in selected]
        identities = geometry.identities(pred_win, agent_ids)
        predicted, pred_coverage = _contact_pairs(selected, identities, frame_boxes, cutoff, float(conf["horizon_s"]))
        gt_oracle, gt_coverage = _contact_pairs(gt_selected, identities, frame_boxes, cutoff, float(conf["horizon_s"]))
        for expected in expected_rows:
            bucket, target = int(expected["k"]), tuple(map(int, expected.get("involved_carla_ids") or []))
            by_condition = {
                PREDICTED_XY_ORACLE_GEOMETRY: {pair: p for pair, p in predicted.items() if p.interval_index == bucket},
                GT_TRAJECTORY_ORACLE_GEOMETRY: {pair: p for pair, p in gt_oracle.items() if p.interval_index == bucket},
            }
            records.append({"scenario_type": row["scenario_type"], "scenario": row["scenario"],
                "window": label, "bucket": bucket, "cutoff_s": cutoff,
                "gt_positive": bool(expected["accident_expected"]), "target_carla_ids": sorted(target),
                "pairs": by_condition,
                "hits": {condition: _target_hit(pairs, target, identities) for condition, pairs in by_condition.items()},
                "identities": identities, "coverage": {PREDICTED_XY_ORACLE_GEOMETRY: pred_coverage,
                                                          GT_TRAJECTORY_ORACLE_GEOMETRY: gt_coverage}})
    return records


def summarize(records, condition):
    counts, windows, scenarios, coverage, seen = Counter(), set(), set(), Counter(), set()
    coverage_keys = ("raw_heading_queries", "raw_heading_available", "raw_heading_missing",
                     "actor_pairs_skipped_missing_heading_at_t0", "path_pairs_with_any_missing_future_heading",
                     "exact_size_queries", "exact_size_available", "exact_size_missing",
                     "actor_pairs_skipped_missing_exact_size_at_t0", "path_pairs_with_any_missing_future_exact_size")
    for record in records:
        window_key = record["scenario_type"], record["scenario"], record["window"]
        if window_key not in seen:
            seen.add(window_key)
            coverage.update({key: record["coverage"][condition][key] for key in coverage_keys})
        pairs, positive = record["pairs"][condition], record["gt_positive"]
        if positive:
            counts["positive"] += 1
            counts["hits"] += int(record["hits"][condition])
        if positive and pairs: counts["TP"] += 1
        elif not positive and pairs:
            counts["FP"] += 1; counts["FP_contact_pair_rows"] += len(pairs)
            windows.add(window_key); scenarios.add(window_key[:2])
        elif positive: counts["FN"] += 1
        else: counts["TN"] += 1
    result = {"n_buckets": len(records), "positive_GT_buckets": counts["positive"],
        "TP": counts["TP"], "FP": counts["FP"], "TN": counts["TN"], "FN": counts["FN"],
        "precision": None if not counts["TP"] + counts["FP"] else counts["TP"] / (counts["TP"] + counts["FP"]),
        "recall": None if not counts["positive"] else counts["TP"] / counts["positive"],
        "GT_actor_pair_hits": counts["hits"],
        "GT_actor_pair_recall": None if not counts["positive"] else counts["hits"] / counts["positive"],
        "FP_contact_pair_rows": counts["FP_contact_pair_rows"], "windows_with_FP_contact": len(windows),
        "scenarios_with_FP_contact": len(scenarios), **coverage}
    for prefix in ("raw_heading", "exact_size"):
        result[f"{prefix}_coverage"] = None if not coverage[f"{prefix}_queries"] else coverage[f"{prefix}_available"] / coverage[f"{prefix}_queries"]
    return result


def horizon_breakdown(records, condition):
    """Contact metrics by exact 1-s bucket and cumulative prediction horizon."""
    max_bucket = max((int(r["bucket"]) for r in records), default=0)

    by_bucket = {}
    cumulative = {}

    for h in range(1, max_bucket + 1):
        exact = [r for r in records if int(r["bucket"]) == h]
        upto = [r for r in records if int(r["bucket"]) <= h]

        by_bucket[str(h)] = summarize(exact, condition)
        cumulative[str(h)] = summarize(upto, condition)

    return {
        "by_bucket": by_bucket,
        "cumulative": cumulative,
    }


def _pair_payload(pair, identities):
    if pair is None: return None
    return {"actor_pair": [pair.actor_a, pair.actor_b],
            "carla_pair": [sorted(identities.get(pair.actor_a, set())), sorted(identities.get(pair.actor_b, set()))],
            "first_contact_time_s": pair.first_contact_s, "minimum_clearance_m": pair.minimum_clearance_m}


def trajectory_differences(records):
    rows = []
    for r in records:
        predicted, oracle = r["pairs"][PREDICTED_XY_ORACLE_GEOMETRY], r["pairs"][GT_TRAJECTORY_ORACLE_GEOMETRY]
        categories = []
        if not r["gt_positive"] and predicted and not oracle: categories.append("TRAJECTORY_FP")
        if r["gt_positive"] and oracle and not predicted: categories.append("TRAJECTORY_FN")
        if r["hits"][GT_TRAJECTORY_ORACLE_GEOMETRY] and not r["hits"][PREDICTED_XY_ORACLE_GEOMETRY]: categories.append("TRAJECTORY_TARGET_PAIR_MISS")
        if r["hits"][PREDICTED_XY_ORACLE_GEOMETRY] and not r["hits"][GT_TRAJECTORY_ORACLE_GEOMETRY]: categories.append("TRAJECTORY_TARGET_PAIR_RECOVERED")
        if predicted and oracle and set(predicted) != set(oracle): categories.append("DIFFERENT_ACTOR_PAIRS")
        for category in categories:
            pp = next(iter(predicted.values()), None); op = next(iter(oracle.values()), None)
            rows.append({"scenario_type": r["scenario_type"], "scenario": r["scenario"], "window": r["window"],
                "bucket": r["bucket"], "GT_label": r["gt_positive"], "target_CARLA_pair": r["target_carla_ids"],
                "predictor_actor_CARLA_pair": _pair_payload(pp, r["identities"]),
                "GT_oracle_actor_CARLA_pair": _pair_payload(op, r["identities"]),
                "predictor_first_contact_time_s": None if pp is None else pp.first_contact_s,
                "GT_oracle_first_contact_time_s": None if op is None else op.first_contact_s,
                "predictor_minimum_clearance_m": None if pp is None else pp.minimum_clearance_m,
                "GT_oracle_minimum_clearance_m": None if op is None else op.minimum_clearance_m,
                "difference_category": category})
    return rows


def predictor_fp_pairs(records):
    rows = []
    for r in records:
        if r["gt_positive"]: continue
        for pair in r["pairs"][PREDICTED_XY_ORACLE_GEOMETRY].values():
            rows.append({"scenario_type": r["scenario_type"], "scenario": r["scenario"], "window": r["window"],
                "bucket": r["bucket"], "cutoff_s": r["cutoff_s"], **_pair_payload(pair, r["identities"])})
    return rows


def predictor_fn_buckets(records):
    return [{"scenario_type": r["scenario_type"], "scenario": r["scenario"], "window": r["window"],
             "bucket": r["bucket"], "cutoff_s": r["cutoff_s"], "target_CARLA_pair": r["target_carla_ids"],
             "GT_oracle_pairs": [_pair_payload(p, r["identities"]) for p in r["pairs"][GT_TRAJECTORY_ORACLE_GEOMETRY].values()]}
            for r in records if r["gt_positive"] and not r["pairs"][PREDICTED_XY_ORACLE_GEOMETRY]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--carla-maps", required=True)
    parser.add_argument("--predictor", required=True); parser.add_argument("--predictor-mode", choices=("waypoints", "joint_scene"))
    parser.add_argument("--predictor-device", default="cpu")
    parser.add_argument("--gt-run", default="out/experiments/gt_future_waypointnet_0m_event_val104")
    parser.add_argument("--out", required=True); parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    mode = args.predictor_mode or infer_predictor_mode(args.predictor)
    run = Path(args.gt_run); cohort = json.loads((run / "batch_manifest.json").read_text())
    conf = dict(cohort["config"], gt_run=str(run))
    scenarios = {raw_audit.scenario_key(s): s for s in state_audit.scan_scenarios(args.root)}
    missing = [row["scenario"] for row in cohort["scenarios"] if raw_audit.scenario_row_key(row) not in scenarios]
    if missing: raise KeyError(f"raw scenarios not found: {missing[:10]}")
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(audit_one, args.root, row, conf, args.carla_maps,
                   scenarios[raw_audit.scenario_row_key(row)], args.predictor, mode, args.predictor_device): row
                   for row in cohort["scenarios"]}
        for index, future in enumerate(as_completed(futures), 1):
            records.extend(future.result()); print(f"[{index}/{len(futures)}] complete {futures[future]['scenario']}", flush=True)
    summaries = {condition: summarize(records, condition) for condition in
                 (PREDICTED_XY_ORACLE_GEOMETRY, GT_TRAJECTORY_ORACLE_GEOMETRY)}
    oracle = summaries[GT_TRAJECTORY_ORACLE_GEOMETRY]
    if {key: oracle[key] for key in GT_REFERENCE} != GT_REFERENCE:
        raise RuntimeError(
            f"GT trajectory oracle does not reproduce trusted reference: {oracle}"
        )
    for condition, summary in summaries.items():
        if summary["n_buckets"] != 624 or summary["positive_GT_buckets"] != 69:
            raise RuntimeError(f"unexpected cohort bucket set for {condition}: {summary}")
    diffs, fps, fns = trajectory_differences(records), predictor_fp_pairs(records), predictor_fn_buckets(records)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    doc = {"condition": PREDICTED_XY_ORACLE_GEOMETRY, "predictor": str(Path(args.predictor).resolve()),
           "predictor_mode": mode, "conditions": summaries, "GT_trajectory_oracle_reference": GT_REFERENCE,
           "predictor_vs_trusted_GT_trajectory_oracle": {
               key: summaries[PREDICTED_XY_ORACLE_GEOMETRY][key] - value for key, value in GT_REFERENCE.items()},
           "coverage": {key: summaries[PREDICTED_XY_ORACLE_GEOMETRY][key] for key in
               ("raw_heading_queries", "raw_heading_available", "raw_heading_missing", "raw_heading_coverage",
                "exact_size_queries", "exact_size_available", "exact_size_missing", "exact_size_coverage")},
           "difference_categories": dict(Counter(row["difference_category"] for row in diffs)),
           "horizon_breakdown": {
               condition: horizon_breakdown(records, condition)
               for condition in (
                   PREDICTED_XY_ORACLE_GEOMETRY,
                   GT_TRAJECTORY_ORACLE_GEOMETRY,
               )
           },
           "contact_lookback_s": LOOKBACK_S, "sample_dt_s": .1, "contact_margin_m": 0.}
    (out / "summary.json").write_text(json.dumps(doc, indent=2) + "\n")
    (out / "trajectory_contact_differences.jsonl").write_text("".join(json.dumps(row) + "\n" for row in diffs))
    (out / "predictor_fp_pairs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in fps))
    (out / "predictor_fn_buckets.jsonl").write_text("".join(json.dumps(row) + "\n" for row in fns))
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
