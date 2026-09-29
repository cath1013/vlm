#!/usr/bin/env python3
"""Evaluation-only raw-geometry ablation for the fixed GT validation cohort.

The three conditions share the same raw CARLA poses, frames, GT buckets, and
CONTACT_ONSET state machine.  They differ only in the requested footprint and
vertical-overlap choices; this script neither regenerates GT nor invokes the
predictor, re-ranker, scorer, or an LLM.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples import audit_gt_carla_boxes as raw_audit  # noqa: E402
from traffic_llm.config import DEFAULT_CLASS_SIZES  # noqa: E402
from traffic_llm.deepaccident import CLASS_MAP, scan_scenarios  # noqa: E402


GENERIC_2D = "GENERIC_2D"
EXACT_SIZE_2D = "EXACT_SIZE_2D"
EXACT_SIZE_3D = "EXACT_SIZE_3D"
CONDITIONS = (GENERIC_2D, EXACT_SIZE_2D, EXACT_SIZE_3D)
LOOKBACK_S = 0.3


def frame_time(frame: int) -> float:
    return (frame - 1) / raw_audit.FRAME_RATE_HZ


def generic_box(box: raw_audit.WorldBox) -> raw_audit.WorldBox:
    """Change dimensions only; position, yaw/heading, and raw height remain intact."""
    try:
        size = DEFAULT_CLASS_SIZES[CLASS_MAP.get(box.cls, box.cls)]
    except KeyError as exc:
        raise KeyError(
            f"no DEFAULT_CLASS_SIZES entry for raw CARLA class {box.cls!r} "
            f"(actor {box.carla_id})"
        ) from exc
    return replace(box, length=size.length_m, width=size.width_m)


def vertical_overlap(a: raw_audit.WorldBox, b: raw_audit.WorldBox) -> bool:
    return max(a.center_z - a.height / 2.0, b.center_z - b.height / 2.0) <= min(
        a.center_z + a.height / 2.0, b.center_z + b.height / 2.0
    ) + 1e-9


def contact(a: raw_audit.WorldBox, b: raw_audit.WorldBox, condition: str) -> bool:
    """The only condition-dependent contact test (all use a 0.0 m margin)."""
    if condition == GENERIC_2D:
        a, b = generic_box(a), generic_box(b)
    horizontal = raw_audit.polygon_clearance(a.polygon(), b.polygon()) <= 1e-9
    return horizontal and (condition != EXACT_SIZE_3D or vertical_overlap(a, b))


def sample_pair_states(boxes_by_frame, frames, pair, condition: str) -> list[dict[str, Any]]:
    """Raw state samples; missing actor frames deliberately remain unknown."""
    samples = []
    for frame in frames:
        boxes = boxes_by_frame[frame]
        if not set(pair) <= set(boxes):
            samples.append({"frame": frame, "time_s": frame_time(frame), "contact": None})
            continue
        a, b = (boxes[carla_id] for carla_id in pair)
        samples.append({
            "frame": frame,
            "time_s": frame_time(frame),
            "contact": contact(a, b, condition),
        })
    return samples


def onset_event(samples: list[dict[str, Any]], cutoff_frame: int) -> dict[str, Any]:
    """CONTACT_ONSET semantics with missing-dependent onset/recontact suppression."""
    cutoff_index = next(i for i, row in enumerate(samples) if row["frame"] == cutoff_frame)
    cutoff = samples[cutoff_index]
    before, future = samples[:cutoff_index], samples[cutoff_index + 1:]
    contact_now = cutoff["contact"]
    if contact_now is None:
        return {"event": "NONE", "time_s": None, "uncertain": True}

    valid_before = [row for row in before if row["contact"] is not None]
    last_before = valid_before[-1] if valid_before else None
    # A missing state after the latest jointly-valid pre-cutoff state means a
    # boundary onset cannot be established from separated -> contact.
    missing_before_tail = (last_before is None or any(
        row["contact"] is None and row["time_s"] > last_before["time_s"] + 1e-9
        for row in before
    ))

    if not contact_now:
        first = next((row for row in future if row["contact"] is True), None)
        return {"event": "NEW_CONTACT" if first else "NONE",
                "time_s": None if first is None else first["time_s"], "uncertain": False}

    if last_before is not None and last_before["contact"] is False and not missing_before_tail:
        return {"event": "BOUNDARY_ONSET", "time_s": cutoff["time_s"], "uncertain": False}
    if last_before is None or missing_before_tail:
        return {"event": "RECONTACT_UNCERTAIN", "time_s": None, "uncertain": True}

    # Contact already existed at the cutoff.  It is not a predicted event
    # unless a fully observed separation is followed by a fully observed contact.
    separated = False
    missing_after_separation = False
    for row in future:
        state = row["contact"]
        if state is None:
            if separated:
                missing_after_separation = True
            continue
        if state is False:
            separated = True
        elif state is True and separated:
            if missing_after_separation:
                return {"event": "RECONTACT_UNCERTAIN", "time_s": None, "uncertain": True}
            return {"event": "RECONTACT", "time_s": row["time_s"], "uncertain": False}
    return {"event": "PREEXISTING_PERSISTENT", "time_s": None, "uncertain": False}


def event_pairs_for_bucket(pair_events, cutoff_s: float, start_s: float, end_s: float):
    pairs = {}
    for pair, event in pair_events.items():
        at = event["time_s"]
        if event["event"] == "BOUNDARY_ONSET" and math.isclose(start_s, cutoff_s, abs_tol=1e-9):
            pairs[pair] = event
        elif at is not None and raw_audit.in_interval(at, start_s, end_s):
            pairs[pair] = event
    return pairs


def audit_scenario(scenario, scenario_row: dict, gt_run: str) -> list[dict[str, Any]]:
    frames = sorted({frame for series in scenario.agents.values() for frame in series.frames})
    boxes_by_frame = {frame: raw_audit.merge_frame_boxes(scenario, frame)[0] for frame in frames}
    gt_dir = Path(gt_run) / scenario_row["directory"]
    horizon_ends = raw_horizon_ends(gt_dir)
    grouped = defaultdict(list)
    for label, cutoff_s, expected in raw_audit._scenario_buckets(gt_dir):
        grouped[(label, cutoff_s)].append(expected)

    records = []
    for (label, cutoff_s), expected_rows in grouped.items():
        cutoff_frame = int(round(cutoff_s * raw_audit.FRAME_RATE_HZ)) + 1
        if cutoff_frame not in boxes_by_frame:
            raise KeyError(f"raw cutoff frame {cutoff_frame} absent for {scenario_row['scenario']}")
        horizon_end = horizon_ends[label]
        sequence_frames = [frame for frame in frames if cutoff_s - LOOKBACK_S - 1e-9
                           <= frame_time(frame) <= horizon_end + 1e-9]
        # The union ensures every condition evaluates precisely the same raw
        # actor-pair trajectories; only its geometry test determines an event.
        candidate_pairs = set()
        for frame in sequence_frames:
            candidate_pairs.update(frozenset(pair) for pair in itertools.combinations(boxes_by_frame[frame], 2))
        events = {
            condition: {
                pair: onset_event(sample_pair_states(boxes_by_frame, sequence_frames, pair, condition), cutoff_frame)
                for pair in candidate_pairs
            }
            for condition in CONDITIONS
        }
        for expected in expected_rows:
            start, end = float(expected["interval_start_s"]), float(expected["interval_end_s"])
            predictions = {}
            for condition in CONDITIONS:
                pairs = event_pairs_for_bucket(events[condition], cutoff_s, start, end)
                predictions[condition] = {
                    "predict_contact": bool(pairs),
                    "contact_pairs": [
                        {"carla_ids": sorted(pair), "contact_event": event["event"],
                         "contact_time_s": event["time_s"]}
                        for pair, event in sorted(pairs.items(), key=lambda item: sorted(item[0]))
                    ],
                    "target_pair_hit": raw_audit.target_pair_hit(
                        set(pairs), expected.get("involved_carla_ids") or []
                    ),
                }
            records.append({
                "dataset_split": scenario_row["dataset_split"],
                "scenario_type": scenario_row["scenario_type"],
                "scenario": scenario_row["scenario"],
                "window_label": label,
                "k": int(expected["k"]),
                "gt_positive": bool(expected["accident_expected"]),
                "involved_carla_ids": list(expected.get("involved_carla_ids") or []),
                "predictions": predictions,
            })
    return records


def raw_horizon_ends(gt_dir: Path) -> dict[str, float]:
    manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))
    ends = {}
    for window in manifest.get("windows", []):
        gt = json.loads((gt_dir / window["ground_truth"]).read_text(encoding="utf-8"))
        if gt.get("expected"):
            ends[gt["window"]["label"]] = max(float(row["interval_end_s"]) for row in gt["expected"])
    return ends


def summary_for(records, condition: str) -> dict[str, Any]:
    counts = Counter()
    fp_windows, fp_scenarios = set(), set()
    for row in records:
        prediction, positive = row["predictions"][condition], row["gt_positive"]
        predicted = prediction["predict_contact"]
        if positive:
            counts["positive_GT_buckets"] += 1
            counts["GT_actor_pair_hits"] += int(prediction["target_pair_hit"])
        if predicted and positive:
            counts["TP_any_contact"] += 1
        elif predicted:
            counts["FP_contact"] += 1
            counts["FP_contact_pair_rows"] += len(prediction["contact_pairs"])
            fp_windows.add((row["scenario_type"], row["scenario"], row["window_label"]))
            fp_scenarios.add((row["scenario_type"], row["scenario"]))
        elif positive:
            counts["FN_no_contact"] += 1
        else:
            counts["TN_no_contact"] += 1
    tp, fp, positives = counts["TP_any_contact"], counts["FP_contact"], counts["positive_GT_buckets"]
    return {
        "n_buckets": len(records), "TP_any_contact": tp, "FP_contact": fp,
        "TN_no_contact": counts["TN_no_contact"], "FN_no_contact": counts["FN_no_contact"],
        "precision": None if not tp + fp else tp / (tp + fp),
        "recall": None if not positives else tp / positives,
        "positive_GT_buckets": positives, "GT_actor_pair_hits": counts["GT_actor_pair_hits"],
        "GT_actor_pair_recall": None if not positives else counts["GT_actor_pair_hits"] / positives,
        "FP_contact_pair_rows": counts["FP_contact_pair_rows"],
        "windows_with_FP_contact": len(fp_windows), "scenarios_with_FP_contact": len(fp_scenarios),
    }


def changed_buckets(records, before: str, after: str) -> dict[str, int]:
    def keys(predicate):
        return {(r["scenario_type"], r["scenario"], r["window_label"], r["k"])
                for r in records if predicate(r)}
    old_pred = lambda r: r["predictions"][before]["predict_contact"]
    new_pred = lambda r: r["predictions"][after]["predict_contact"]
    old_hit = lambda r: r["predictions"][before]["target_pair_hit"]
    new_hit = lambda r: r["predictions"][after]["target_pair_hit"]
    old_fp, new_fp = keys(lambda r: not r["gt_positive"] and old_pred(r)), keys(lambda r: not r["gt_positive"] and new_pred(r))
    old_tp, new_tp = keys(lambda r: r["gt_positive"] and old_pred(r)), keys(lambda r: r["gt_positive"] and new_pred(r))
    old_hits, new_hits = keys(lambda r: r["gt_positive"] and old_hit(r)), keys(lambda r: r["gt_positive"] and new_hit(r))
    return {
        "FP buckets removed": len(old_fp - new_fp), "FP buckets introduced": len(new_fp - old_fp),
        "TP buckets lost": len(old_tp - new_tp), "TP buckets gained": len(new_tp - old_tp),
        "GT-pair hits lost": len(old_hits - new_hits), "GT-pair hits gained": len(new_hits - old_hits),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="DeepAccident root")
    parser.add_argument("--gt-run", default="out/experiments/gt_future_waypointnet_1m_noreranker_val104")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    gt_run = Path(args.gt_run)
    cohort = json.loads((gt_run / "batch_manifest.json").read_text(encoding="utf-8"))
    scenario_rows = cohort["scenarios"]
    scenarios = {raw_audit.scenario_key(s): s for s in scan_scenarios(args.root)}
    missing = [f"{row['scenario_type']}/{row['scenario']}" for row in scenario_rows
               if raw_audit.scenario_row_key(row) not in scenarios]
    if missing:
        raise KeyError(f"raw scenarios not found: {missing[:10]}")
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(audit_scenario, scenarios[raw_audit.scenario_row_key(row)], row, str(gt_run)): row
                   for row in scenario_rows}
        for done, future in enumerate(as_completed(futures), 1):
            records.extend(future.result())
            print(f"[{done}/{len(scenario_rows)}] complete {futures[future]['scenario']}", flush=True)
    records.sort(key=lambda row: (row["scenario_type"], row["scenario"], row["window_label"], row["k"]))
    summaries = {condition: summary_for(records, condition) for condition in CONDITIONS}
    for condition, summary in summaries.items():
        if summary["n_buckets"] != 624 or summary["positive_GT_buckets"] != 69:
            raise RuntimeError(f"unexpected GT bucket set for {condition}: {summary}")
    output = {
        "conditions": summaries,
        "pairwise_changes": {
            "A -> B": changed_buckets(records, GENERIC_2D, EXACT_SIZE_2D),
            "B -> C": changed_buckets(records, EXACT_SIZE_2D, EXACT_SIZE_3D),
        },
        "contact_semantics": "CONTACT_ONSET: NEW_CONTACT after separation; valid BOUNDARY_ONSET retained; persistent pre-existing contact suppressed; missing-dependent recontact/boundary cases suppressed as uncertain.",
        "margin_m": 0.0,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_bucket.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    (out / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
