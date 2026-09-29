"""Compare raw-box collision rules, including event-based contact onset.

This is an evaluation-only companion to :mod:`audit_gt_carla_boxes`.  It
imports that module's raw-label merge and exact 3D box-contact geometry rather
than maintaining a second implementation.
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
from traffic_llm.deepaccident import scan_scenarios  # noqa: E402


RULE_ANY = "ANY_OVERLAP"
RULE_EXCLUDE = "EXCLUDE_TOUCHING_NOW"
RULE_ONSET = "CONTACT_ONSET"
RULES = (RULE_ANY, RULE_EXCLUDE, RULE_ONSET)


def _frame_time(frame: int) -> float:
    return (frame - 1) / raw_audit.FRAME_RATE_HZ


def _vertical_gap(a, b) -> float:
    """Exact gap between the two vertical box intervals."""
    a_lo, a_hi = a.center_z - a.height / 2.0, a.center_z + a.height / 2.0
    b_lo, b_hi = b.center_z - b.height / 2.0, b.center_z + b.height / 2.0
    return max(0.0, a_lo - b_hi, b_lo - a_hi)


def _exact_3d_clearance(a, b) -> float:
    """Exact clearance of upright boxes from existing exact planar geometry."""
    return math.hypot(raw_audit.polygon_clearance(a.polygon(), b.polygon()),
                      _vertical_gap(a, b))


def _sample_pair_states(boxes_by_frame, frames, pair, separation_epsilon_m: float):
    """Return the raw contact-state sequence, with optional exact-gap hysteresis.

    At epsilon zero this is precisely the boolean contact transition.  For a
    positive epsilon, an existing CONTACT state is retained until the exact
    3D clearance (planar exact gap plus vertical interval gap) exceeds epsilon.
    """
    state = None
    samples = []
    for frame in frames:
        boxes = boxes_by_frame.get(frame, {})
        if not set(pair) <= set(boxes):
            samples.append({"frame": frame, "time_s": _frame_time(frame), "state": "MISSING"})
            state = None
            continue
        a, b = (boxes[carla_id] for carla_id in pair)
        geometry_contact = raw_audit.boxes_contact(a, b, margin_m=0.0)
        clearance = _exact_3d_clearance(a, b)
        if geometry_contact:
            state, label = True, "CONTACT"
        elif clearance > separation_epsilon_m:
            state, label = False, "SEPARATED"
        else:
            label = ("CONTACT_HYSTERESIS" if state is True else
                     "SEPARATED_HYSTERESIS" if state is False else "UNCERTAIN")
        samples.append({
            "frame": frame,
            "time_s": _frame_time(frame),
            "state": label,
            "contact_state": state,
            "geometry_contact": geometry_contact,
            "bev_gap_m": raw_audit.polygon_clearance(a.polygon(), b.polygon()),
            "vertical_gap_m": _vertical_gap(a, b),
            "exact_3d_clearance_m": clearance,
        })
    return samples


def _onset_events(samples, cutoff_frame: int):
    """Classify future raw contact onsets for a pair at one prediction cutoff."""
    cutoff_index = next(i for i, row in enumerate(samples) if row["frame"] == cutoff_frame)
    cutoff = samples[cutoff_index]
    before = samples[:cutoff_index]
    future = samples[cutoff_index + 1:]
    if "contact_state" not in cutoff:
        return {
            "contact_at_cutoff": False, "contact_during_lookback": False,
            "preexisting_persistent": False, "new_contact": False,
            "boundary_onset": False, "recontact": False, "no_contact": True,
            "first_new_contact_frame": None, "first_new_contact_time_s": None,
            "events": [],
        }
    contact_at_cutoff = bool(cutoff.get("contact_state"))
    contact_during_lookback = any(row.get("contact_state") for row in before)
    previous_valid_state = next(
        (row["contact_state"] for row in reversed(before)
         if row.get("contact_state") is not None),
        None,
    )
    boundary_onset = contact_at_cutoff and previous_valid_state is False

    events = []
    previous = cutoff["contact_state"]
    separated_after_cutoff = False
    for row in future:
        if "contact_state" not in row:
            previous = None
            continue
        current = row["contact_state"]
        if current is False:
            separated_after_cutoff = True
        if previous is False and current is True:
            events.append({
                "type": "RECONTACT" if contact_at_cutoff and separated_after_cutoff else "NEW_CONTACT",
                "frame": row["frame"], "time_s": row["time_s"],
            })
        previous = current

    new_events = [event for event in events if event["type"] == "NEW_CONTACT"]
    recontact_events = [event for event in events if event["type"] == "RECONTACT"]
    preexisting_persistent = (
        contact_at_cutoff and contact_during_lookback and not boundary_onset
        and bool(future) and all(row.get("contact_state") is True for row in future)
    )
    first = (new_events + recontact_events)[0] if (new_events or recontact_events) else None
    if boundary_onset and first is None:
        first = {"type": "BOUNDARY_ONSET", "frame": cutoff_frame, "time_s": cutoff["time_s"]}
    no_contact = not (new_events or boundary_onset or recontact_events)
    return {
        "contact_at_cutoff": contact_at_cutoff,
        "contact_during_lookback": contact_during_lookback,
        "preexisting_persistent": preexisting_persistent,
        "new_contact": bool(new_events),
        "boundary_onset": boundary_onset,
        "recontact": bool(recontact_events),
        "no_contact": no_contact,
        "first_new_contact_frame": None if first is None else first["frame"],
        "first_new_contact_time_s": None if first is None else first["time_s"],
        "events": events,
    }


def _window_pair_records(boxes_by_frame, contacts_by_frame, all_frames, cutoff_s, horizon_end_s,
                         lookback_s, separation_epsilon_m):
    cutoff_frame = int(round(cutoff_s * raw_audit.FRAME_RATE_HZ)) + 1
    if cutoff_frame not in boxes_by_frame:
        return {}
    sequence_frames = [
        frame for frame in all_frames
        if cutoff_s - lookback_s - 1e-9 <= _frame_time(frame) <= horizon_end_s + 1e-9
    ]
    candidate_pairs = set().union(*(contacts_by_frame.get(frame, set()) for frame in sequence_frames)) \
        if sequence_frames else set()
    result = {}
    for pair in candidate_pairs:
        samples = _sample_pair_states(boxes_by_frame, sequence_frames, pair, separation_epsilon_m)
        event = _onset_events(samples, cutoff_frame)
        event["involved_carla_ids"] = sorted(pair)
        event["contact_state_sequence"] = samples
        result[pair] = event
    return result


def _pairs_in_interval(contacts_by_frame, start_s: float, end_s: float):
    pairs = set()
    for frame, frame_pairs in contacts_by_frame.items():
        if raw_audit.in_interval(_frame_time(frame), start_s, end_s):
            pairs.update(frame_pairs)
    return pairs


def _onset_pairs_for_bucket(pair_records, cutoff_s: float, start_s: float, end_s: float):
    pairs = set()
    for pair, event in pair_records.items():
        # A boundary onset exists at the cutoff and is a candidate for the
        # first future bucket, even though that bucket is open at the left.
        if event["boundary_onset"] and math.isclose(start_s, cutoff_s, abs_tol=1e-9):
            pairs.add(pair)
        if any(raw_audit.in_interval(row["time_s"], start_s, end_s) for row in event["events"]):
            pairs.add(pair)
    return pairs


def _window_horizon_ends(gt_dir: Path):
    """Read all expected intervals, including unscorable ones, for sequence length."""
    manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))
    ends = {}
    for window in manifest.get("windows", []):
        gt = json.loads((gt_dir / window["ground_truth"]).read_text(encoding="utf-8"))
        expected = gt.get("expected", [])
        if expected:
            ends[gt["window"]["label"]] = max(float(row["interval_end_s"]) for row in expected)
    return ends


def audit_scenario_onsets(scenario, scenario_row, gt_run, lookback_s, separation_epsilon_m):
    """Evaluate all three rules for one scenario with shared raw geometry."""
    frames = sorted({frame for series in scenario.agents.values() for frame in series.frames})
    boxes_by_frame, contacts_by_frame = {}, {}
    for frame in frames:
        boxes, _ = raw_audit.merge_frame_boxes(scenario, frame)
        boxes_by_frame[frame] = boxes
        contacts_by_frame[frame] = {
            frozenset((a.carla_id, b.carla_id))
            for a, b in itertools.combinations(boxes.values(), 2)
            if raw_audit.boxes_contact(a, b, margin_m=0.0)
        }

    gt_dir = Path(gt_run) / scenario_row["directory"]
    horizon_ends = _window_horizon_ends(gt_dir)
    grouped = defaultdict(list)
    for label, cutoff_s, expected in raw_audit._scenario_buckets(gt_dir):
        grouped[(label, cutoff_s)].append(expected)
    records = []
    for (window_label, cutoff_s), expected_rows in grouped.items():
        horizon_end_s = horizon_ends.get(
            window_label, max(float(expected["interval_end_s"]) for expected in expected_rows)
        )
        pair_records = _window_pair_records(
            boxes_by_frame, contacts_by_frame, frames, cutoff_s, horizon_end_s,
            lookback_s, separation_epsilon_m,
        )
        cutoff_contacts = contacts_by_frame.get(
            int(round(cutoff_s * raw_audit.FRAME_RATE_HZ)) + 1, set()
        )
        for expected in expected_rows:
            start_s, end_s = float(expected["interval_start_s"]), float(expected["interval_end_s"])
            any_pairs = _pairs_in_interval(contacts_by_frame, start_s, end_s)
            excluded_pairs = {pair for pair in any_pairs if pair not in cutoff_contacts}
            onset_pairs = _onset_pairs_for_bucket(pair_records, cutoff_s, start_s, end_s)
            by_rule = {RULE_ANY: any_pairs, RULE_EXCLUDE: excluded_pairs, RULE_ONSET: onset_pairs}
            records.append({
                "scenario": scenario_row["scenario"],
                "scenario_type": scenario_row["scenario_type"],
                "window_label": window_label,
                "k": int(expected["k"]),
                "interval_start_s": start_s,
                "interval_end_s": end_s,
                "gt_positive": bool(expected["accident_expected"]),
                "involved_carla_ids": list(expected.get("involved_carla_ids") or []),
                "predictions": {
                    rule: {
                        "predict_contact": bool(pairs),
                        "contacting_pairs": [sorted(pair) for pair in sorted(pairs, key=lambda p: sorted(p))],
                        "target_pair_hit": raw_audit.target_pair_hit(
                            pairs, expected.get("involved_carla_ids") or []
                        ),
                    }
                    for rule, pairs in by_rule.items()
                },
                "pair_records": [pair_records[pair] for pair in sorted(pair_records, key=lambda p: sorted(p))],
            })
    return records


def _rule_summary(records, rule):
    counts = Counter()
    fp_windows, fp_scenarios = set(), set()
    for row in records:
        prediction = row["predictions"][rule]
        predicted, positive = prediction["predict_contact"], row["gt_positive"]
        if positive:
            counts["positive_GT_buckets"] += 1
            counts["GT_actor_pair_hits"] += int(prediction["target_pair_hit"])
        if predicted and positive:
            counts["TP"] += 1
        elif predicted:
            counts["FP"] += 1
            fp_windows.add((row["scenario_type"], row["scenario"], row["window_label"]))
            fp_scenarios.add((row["scenario_type"], row["scenario"]))
        elif positive:
            counts["FN"] += 1
        else:
            counts["TN"] += 1
    tp, fp, positives = counts["TP"], counts["FP"], counts["positive_GT_buckets"]
    return {
        "n_buckets": len(records), "TP": tp, "FP": fp, "TN": counts["TN"], "FN": counts["FN"],
        "precision": None if not tp + fp else tp / (tp + fp),
        "recall": None if not positives else tp / positives,
        "positive_GT_buckets": positives,
        "GT_actor_pair_hits": counts["GT_actor_pair_hits"],
        "GT_actor_pair_recall": None if not positives else counts["GT_actor_pair_hits"] / positives,
        "FP_buckets": fp,
        "unique_FP_windows": len(fp_windows),
        "unique_FP_scenario_instances": len(fp_scenarios),
    }


def _case_rows(records, predicate):
    return [row for row in records if predicate(row)]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="DeepAccident root")
    ap.add_argument("--gt-run", required=True, help="existing GT validation run")
    ap.add_argument("--out", required=True, help="diagnostic output directory")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lookback-s", type=float, default=0.3)
    ap.add_argument("--separation-epsilon-m", type=float, default=0.0)
    args = ap.parse_args(argv)
    if args.lookback_s < 0 or args.separation_epsilon_m < 0:
        ap.error("--lookback-s and --separation-epsilon-m must be non-negative")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    gt_run = Path(args.gt_run)
    cohort = json.loads((gt_run / "batch_manifest.json").read_text(encoding="utf-8"))
    scenario_rows = cohort["scenarios"]
    scenarios = {raw_audit.scenario_key(s): s for s in scan_scenarios(args.root)}
    missing = [
        f"{row['scenario_type']}/{row['scenario']}" for row in scenario_rows
        if raw_audit.scenario_row_key(row) not in scenarios
    ]
    if missing:
        raise KeyError(f"raw scenarios not found: {missing[:10]}")

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(audit_scenario_onsets, scenarios[raw_audit.scenario_row_key(row)], row,
                        str(gt_run), args.lookback_s, args.separation_epsilon_m): row
            for row in scenario_rows
        }
        for done, future in enumerate(as_completed(futures), 1):
            records.extend(future.result())
            print(f"[{done}/{len(scenario_rows)}] complete {futures[future]['scenario']}", flush=True)
    records.sort(key=lambda row: (row["scenario_type"], row["scenario"], row["window_label"], row["k"]))

    summaries = {rule: _rule_summary(records, rule) for rule in RULES}
    any_summary = summaries[RULE_ANY]
    if any_summary["n_buckets"] != 624 or any_summary["positive_GT_buckets"] != 69:
        raise RuntimeError(f"unexpected GT bucket set: {any_summary}")
    expected_any = {"TP": 66, "FP": 119, "TN": 436, "FN": 3}
    found_any = {key: any_summary[key] for key in expected_any}
    if found_any != expected_any:
        raise RuntimeError(
            "ANY_OVERLAP disagrees with the unrestricted raw-box audit; "
            f"expected={expected_any}, found={found_any}. This is an implementation discrepancy."
        )

    lost_tp = _case_rows(records, lambda row: row["gt_positive"]
                         and row["predictions"][RULE_ANY]["predict_contact"]
                         and not row["predictions"][RULE_EXCLUDE]["predict_contact"])
    recovered_tp = [row for row in lost_tp if row["predictions"][RULE_ONSET]["predict_contact"]]
    fp_removed = _case_rows(records, lambda row: not row["gt_positive"]
                          and row["predictions"][RULE_ANY]["predict_contact"]
                          and not row["predictions"][RULE_ONSET]["predict_contact"])
    fp_retained = _case_rows(records, lambda row: not row["gt_positive"]
                           and row["predictions"][RULE_ANY]["predict_contact"]
                           and row["predictions"][RULE_ONSET]["predict_contact"])
    event_counts = Counter()
    event_counted = set()
    for row in records:
        for pair in row["pair_records"]:
            key = (row["scenario_type"], row["scenario"], row["window_label"],
                   tuple(pair["involved_carla_ids"]))
            if key in event_counted:
                continue
            event_counted.add(key)
            for flag, label in (("new_contact", "NEW_CONTACT"), ("boundary_onset", "BOUNDARY_ONSET"),
                                ("preexisting_persistent", "PREEXISTING_PERSISTENT"),
                                ("recontact", "RECONTACT")):
                event_counts[label] += int(pair[flag])
    summary = {
        "lookback_s": args.lookback_s,
        "separation_epsilon_m": args.separation_epsilon_m,
        "separation_note": "Positive hysteresis uses exact 3D clearance derived from the existing exact BEV gap and vertical interval gap.",
        "rules": summaries,
        "comparisons": {
            "TPs_lost_by_EXCLUDE_TOUCHING_NOW_relative_to_ANY_OVERLAP": len(lost_tp),
            "lost_TPs_recovered_by_CONTACT_ONSET": len(recovered_tp),
            "FPs_removed_by_CONTACT_ONSET_relative_to_ANY_OVERLAP": len(fp_removed),
            "FPs_retained_by_CONTACT_ONSET": len(fp_retained),
        },
        "event_type_counts": dict(event_counts),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    (out / "lost_tp_cases.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in lost_tp), encoding="utf-8")
    (out / "recovered_tp_cases.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in recovered_tp), encoding="utf-8")
    (out / "fp_removed_by_onset.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in fp_removed), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
