#!/usr/bin/env python3
"""Diagnose saved predictor FP pairs against the *same* pair on GT futures.

This is deliberately a narrow replay: its input is the FP-pair JSONL emitted
by ``audit_predictor_trajectory_contacts.py`` and it rebuilds only scenarios
named by that file.  It is diagnostic-only and does not alter the trusted
audit's selection, geometry, or outputs.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples import audit_gt_carla_boxes as raw_audit  # noqa: E402
from examples import audit_gt_heading_source_ablation as geometry  # noqa: E402
from examples import audit_gt_state_source_ablation as state_audit  # noqa: E402
from examples import audit_predictor_trajectory_contacts as trusted  # noqa: E402
from examples.run_deepaccident import infer_predictor_mode  # noqa: E402
from traffic_llm.serialize import rank_actors  # noqa: E402
import traffic_llm.swept_path as swept_path  # noqa: E402

SAMPLE_DT_S = 0.1
CONTACT_MARGIN_M = 0.0
REQUIRED_FP_KEYS = ("scenario_type", "scenario", "window", "bucket", "cutoff_s", "actor_pair")


def read_fp_pairs(path):
    """Read and structurally validate rows before loading or replaying a cohort."""
    rows = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if all(key in value for key in REQUIRED_FP_KEYS) and len(value["actor_pair"]) == 2:
            rows.append(value)
        else:
            rows.append({"_invalid": True, "_line_number": line_number, "_raw": value})
    return rows


def valid_fp_rows(rows):
    return [row for row in rows if not row.get("_invalid")]


def select_affected_scenarios(fp_rows, cohort_rows):
    """Return only manifest rows referred to by valid FP input rows."""
    wanted = {(r["scenario_type"], r["scenario"]) for r in valid_fp_rows(fp_rows)}
    return [r for r in cohort_rows if (r["scenario_type"], r["scenario"]) in wanted]


def _pair_key(row):
    return (row["scenario_type"], row["scenario"], row["window"], int(row["bucket"]),
            tuple(row["actor_pair"]))


def _metric_payload(pair):
    """Use false/null fields for an unavailable non-contact evaluation."""
    if pair is None:
        return {"predicted_contact": False, "contact": False, "contact_event": None,
                "first_contact_time_s": None, "minimum_clearance_m": None,
                "time_of_minimum_clearance_s": None, "clearance_at_1s_m": None,
                "clearance_at_horizon_m": None}
    return {"predicted_contact": bool(pair.predicted_contact), "contact": bool(pair.predicted_contact),
            "contact_event": pair.contact_event, "first_contact_time_s": pair.first_contact_s,
            "minimum_clearance_m": pair.minimum_clearance_m,
            "time_of_minimum_clearance_s": pair.time_after_observation_s,
            "clearance_at_1s_m": pair.clearance_at_1s_m,
            "clearance_at_horizon_m": pair.clearance_at_horizon_m}


def _unavailable_row(base, status, source=None):
    """Keep the JSONL schema useful even when a saved FP cannot be replayed."""
    source = source or {}
    return {**base, "diagnostic_status": status,
            "carla_ids": source.get("carla_pair", [None, None]),
            "actor_classes": [None, None], "observed_speed_mps": [None, None],
            "predictor_same_pair": _metric_payload(None), "gt_same_pair": _metric_payload(None),
            "times_s": [], "predictor_center_distance_m": [], "gt_center_distance_m": [],
            "predictor_relative_xy_m": [], "gt_relative_xy_m": []}


def _evaluate_pair(actors, actor_pair, frame_boxes, identities, cutoff_s, horizon_s):
    """Evaluate exactly one named pair, retaining it even when it is not a contact."""
    by_id = {a.actor_id: a for a in actors}
    if any(aid not in by_id for aid in actor_pair):
        return None, (None, None), [by_id.get(aid) for aid in actor_pair]
    paths = {}
    pair_actors = [by_id[actor_pair[0]], by_id[actor_pair[1]]]
    kwargs = dict(frame_boxes=frame_boxes, identities_by_actor=identities, cutoff_s=cutoff_s, horizon_s=horizon_s,
                  sample_dt_s=SAMPLE_DT_S, contact_margin_m=CONTACT_MARGIN_M,
                  selected_paths=paths, exact_size=True)
    pairs = geometry.raw_yaw_swept_pair_clearances(pair_actors,
        **kwargs)
    key = frozenset(actor_pair)
    result = next((p for p in pairs if frozenset((p.actor_a, p.actor_b)) == key), None)
    # The shared evaluator suppresses static *non-contact* pairs for normal
    # contact reporting.  Here such a pair is the requested diagnostic target,
    # so re-run copies with only the reporting filter bypassed; XY/yaw/sizes
    # and the original observed speeds remain unchanged in the output.
    if result is None and all(abs(a.speed_mps or 0.) <= swept_path.STATIONARY_MPS for a in pair_actors):
        diagnostic_actors = [copy.copy(a) for a in pair_actors]
        for actor in diagnostic_actors:
            actor.speed_mps = swept_path.STATIONARY_MPS + 1e-6
        paths = {}
        kwargs["selected_paths"] = paths
        pairs = geometry.raw_yaw_swept_pair_clearances(diagnostic_actors, **kwargs)
        result = next((p for p in pairs if frozenset((p.actor_a, p.actor_b)) == key), None)
    return result, paths.get(key, (None, None)), pair_actors


def center_distance_trajectories_from_xy(times_s, predictor_xy_a, predictor_xy_b, gt_xy_a, gt_xy_b):
    """Distances deliberately use centers/trajectory XY, never footprint clearance."""
    def vectors(a, b):
        relative, distances = [], []
        for pa, pb in zip(a, b):
            if pa is None or pb is None:
                relative.append(None); distances.append(None)
            else:
                dx, dy = pa[0] - pb[0], pa[1] - pb[1]
                relative.append([dx, dy]); distances.append(math.hypot(dx, dy))
        return relative, distances
    pred_relative, pred_distance = vectors(predictor_xy_a, predictor_xy_b)
    gt_relative, gt_distance = vectors(gt_xy_a, gt_xy_b)
    return {"times_s": times_s, "predictor_center_distance_m": pred_distance,
            "gt_center_distance_m": gt_distance, "predictor_relative_xy_m": pred_relative,
            "gt_relative_xy_m": gt_relative}


def _path_xy(actor, path, times_s, horizon_s):
    if actor is None:
        return [None] * len(times_s)
    path = path or (actor.predictions[0] if actor.predictions else None)
    motion = None if path is None else swept_path._motion_fn(actor, path, horizon_s)
    if motion is None:
        return [None] * len(times_s)
    pose, available, _gaps = motion
    return [None if t > available + 1e-9 else list(pose(t)[:2]) for t in times_s]


def _trajectory_payload(pred_actors, pred_paths, gt_actors, gt_paths, horizon_s):
    times = [round(i * SAMPLE_DT_S, 9) for i in range(int(math.floor(horizon_s / SAMPLE_DT_S + 1e-9)) + 1)]
    return center_distance_trajectories_from_xy(
        times, _path_xy(pred_actors[0], pred_paths[0], times, horizon_s),
        _path_xy(pred_actors[1], pred_paths[1], times, horizon_s),
        _path_xy(gt_actors[0], gt_paths[0], times, horizon_s),
        _path_xy(gt_actors[1], gt_paths[1], times, horizon_s))


def _scenario_context(root, manifest_row, conf, maps, raw_scenario, predictor, mode, device):
    pred_windows, gt_windows, agent_ids, wc = trusted.replay_windows(
        root, manifest_row, conf, maps, predictor, mode, device)
    frames = sorted({f for series in raw_scenario.agents.values() for f in series.frames})
    frame_boxes = {f: raw_audit.merge_frame_boxes(raw_scenario, f)[0] for f in frames}
    return pred_windows, gt_windows, agent_ids, wc, frame_boxes


def diagnose_fp_row(row, context, horizon_s):
    """Produce one output row for one input row; never search for another GT pair."""
    pred_windows, gt_windows, agent_ids, wc, frame_boxes = context
    label, actor_pair = row["window"], tuple(row["actor_pair"])
    pred_win, gt_win = pred_windows.get(label), gt_windows.get(label)
    base = {key: row[key] for key in ("scenario_type", "scenario", "window", "bucket", "cutoff_s")}
    base["actor_pair"] = list(actor_pair)
    if pred_win is None or gt_win is None:
        return _unavailable_row(base, "missing_window", row)
    selected, _ = rank_actors(pred_win.last, wc.actor_cap(len(pred_win.last.actors)))
    gt_by_id = {a.actor_id: a for a in gt_win.last.actors}
    if any(a.actor_id not in gt_by_id for a in selected):
        return _unavailable_row(base, "missing_gt_selected_actor", row)
    # _evaluate_pair resolves only ``actor_pair`` by ID, so retain the actual
    # GT actor collection rather than rebuilding a predictor-shaped list.
    gt_selected = gt_win.last.actors
    identities = geometry.identities(pred_win, agent_ids)
    pp, pred_paths, pred_actors = _evaluate_pair(selected, actor_pair, frame_boxes, identities, row["cutoff_s"], horizon_s)
    gp, gt_paths, gt_actors = _evaluate_pair(gt_selected, actor_pair, frame_boxes, identities, row["cutoff_s"], horizon_s)
    if any(a is None for a in pred_actors) or any(a is None for a in gt_actors):
        unavailable = _unavailable_row(base, "missing_actor_pair", row)
        unavailable["carla_ids"] = [sorted(identities.get(a, set())) for a in actor_pair]
        return unavailable
    if pp is None or gp is None:
        unavailable = _unavailable_row(base, "predictor_pair_unavailable" if pp is None else "gt_pair_unavailable", row)
        unavailable["carla_ids"] = [sorted(identities.get(a, set())) for a in actor_pair]
        unavailable["actor_classes"] = [a.cls for a in pred_actors]
        unavailable["observed_speed_mps"] = [a.speed_mps for a in pred_actors]
        return unavailable
    return {**base, "diagnostic_status": "ok", "carla_ids": [sorted(identities.get(a, set())) for a in actor_pair],
            "actor_classes": [a.cls for a in pred_actors], "observed_speed_mps": [a.speed_mps for a in pred_actors],
            "predictor_same_pair": _metric_payload(pp), "gt_same_pair": _metric_payload(gp),
            **_trajectory_payload(pred_actors, pred_paths, gt_actors, gt_paths, horizon_s)}


def summarize(rows, input_row_count, affected_scenarios, valid_row_count=None):
    good = [r for r in rows if r.get("diagnostic_status") == "ok"]
    gt_metrics = [r["gt_same_pair"] for r in good]
    clearances = [m["minimum_clearance_m"] for m in gt_metrics if m["minimum_clearance_m"] is not None]
    bins = Counter()
    for value in clearances:
        bins["<=0 m" if value <= 0 else "0-0.5 m" if value <= .5 else "0.5-1 m" if value <= 1 else "1-2 m" if value <= 2 else "2-5 m" if value <= 5 else ">5 m"] += 1
    timing = [r["predictor_same_pair"]["first_contact_time_s"] - r["gt_same_pair"]["first_contact_time_s"]
              for r in good if r["predictor_same_pair"]["predicted_contact"] and r["gt_same_pair"]["predicted_contact"]
              and r["predictor_same_pair"]["first_contact_time_s"] is not None and r["gt_same_pair"]["first_contact_time_s"] is not None]
    predictor_first = [r["predictor_same_pair"]["first_contact_time_s"] for r in good
                       if r["predictor_same_pair"]["first_contact_time_s"] is not None]
    return {"input_fp_pair_rows": input_row_count,
            "valid_input_fp_pair_rows": input_row_count if valid_row_count is None else valid_row_count,
            "invalid_input_rows": 0 if valid_row_count is None else input_row_count - valid_row_count,
            "unique_affected_scenarios": affected_scenarios,
            "successfully_diagnosed_rows": len(good), "missing_unavailable_rows": len(rows) - len(good),
            "gt_same_pair_contacts": sum(m["contact"] for m in gt_metrics),
            "gt_same_pair_noncontacts": sum(not m["contact"] for m in gt_metrics),
            "gt_same_pair_minimum_clearance_distribution": {key: bins[key] for key in ("<=0 m", "0-0.5 m", "0.5-1 m", "1-2 m", "2-5 m", ">5 m")},
            "median_gt_same_pair_minimum_clearance_m": statistics.median(clearances) if clearances else None,
            "median_predictor_first_contact_time_s": statistics.median(predictor_first) if predictor_first else None,
            "predictor_minus_gt_first_contact_time_s": {"count": len(timing), "median": statistics.median(timing) if timing else None}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--carla-maps", required=True)
    parser.add_argument("--predictor", required=True); parser.add_argument("--predictor-device", default="cpu")
    parser.add_argument("--predictor-mode", choices=("waypoints", "joint_scene"))
    parser.add_argument("--fp-pairs", required=True)
    parser.add_argument("--gt-run", default="out/experiments/gt_future_waypointnet_0m_event_val104")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    fp_rows = read_fp_pairs(args.fp_pairs)  # Must happen before manifest/replay work.
    valid_rows = valid_fp_rows(fp_rows)
    run = Path(args.gt_run); cohort = json.loads((run / "batch_manifest.json").read_text())
    conf = dict(cohort["config"], gt_run=str(run)); chosen = select_affected_scenarios(fp_rows, cohort["scenarios"])
    raw = {raw_audit.scenario_key(s): s for s in state_audit.scan_scenarios(args.root)}
    manifest = {(r["scenario_type"], r["scenario"]): r for r in chosen}
    mode = args.predictor_mode or infer_predictor_mode(args.predictor)
    contexts, rows = {}, []
    for fp in valid_rows:
        scenario_key = (fp["scenario_type"], fp["scenario"])
        manifest_row = manifest.get(scenario_key)
        if manifest_row is None or raw_audit.scenario_row_key(manifest_row) not in raw:
            base = {k: fp[k] for k in REQUIRED_FP_KEYS if k != "actor_pair"}
            rows.append(_unavailable_row({**base, "actor_pair": fp["actor_pair"]}, "missing_scenario", fp))
            continue
        if scenario_key not in contexts:
            contexts[scenario_key] = _scenario_context(args.root, manifest_row, conf, args.carla_maps,
                raw[raw_audit.scenario_row_key(manifest_row)], args.predictor, mode, args.predictor_device)
        rows.append(diagnose_fp_row(fp, contexts[scenario_key], float(conf["horizon_s"])))
    summary = summarize(rows, len(fp_rows), len(chosen), len(valid_rows))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "diagnosed_predictor_fp_pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
