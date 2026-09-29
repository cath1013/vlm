"""Compare estimated DeepAccident collision time with raw CARLA-box contact onset.

This is evaluation-only.  It deliberately imports the raw per-instance box
conversion and exact 3D contact predicate from ``audit_gt_carla_boxes``;
neither the pipeline collision estimate nor any generated ground truth is
changed.  The supplied GT run provides the validation cohort and its existing
prediction-window manifests.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples import audit_gt_carla_boxes as raw_audit  # noqa: E402
from traffic_llm.deepaccident import estimate_collision, scan_scenarios  # noqa: E402


def frame_time_s(frame: int) -> float:
    """Raw DeepAccident frame number on the repository observation clock."""
    return (frame - 1) / raw_audit.FRAME_RATE_HZ


def _json_number(value):
    return None if value is None or not math.isfinite(value) else value


def _pair_samples(scenario, pair):
    """Read the designated pair at every available raw frame.

    A frame is valid only when both designated CARLA IDs have an exact raw
    box after the same multi-observer merge used by the raw-box audit.
    """
    frames = sorted({frame for series in scenario.agents.values() for frame in series.frames})
    samples = []
    for frame in frames:
        boxes, _ = raw_audit.merge_frame_boxes(scenario, frame)
        if not set(pair) <= set(boxes):
            samples.append({"frame": frame, "time_s": frame_time_s(frame), "state": "missing"})
            continue
        a, b = (boxes[carla_id] for carla_id in pair)
        samples.append({
            "frame": frame,
            "time_s": frame_time_s(frame),
            "state": "contact" if raw_audit.boxes_contact(a, b, margin_m=0.0) else "separated",
        })
    return samples


def find_first_onset(samples):
    """Find the first observed separated -> contact transition.

    If contact is already present before any observed separated state,
    the original onset is not observable. Later recontacts must not be
    mistaken for the first collision onset.
    """
    previous_valid = None
    seen_separated = False

    for index, sample in enumerate(samples):
        if sample["state"] == "missing":
            continue

        # Contact before we have ever observed separation means the
        # original onset occurred before our observable sequence.
        if sample["state"] == "contact" and not seen_separated:
            previous = next(
                (
                    row
                    for row in reversed(samples[:index])
                    if row["state"] != "missing"
                ),
                None,
            )
            return {
                "first_contact_frame": None,
                "first_contact_time_s": None,
                "previous_valid_frame": (
                    None if previous is None else previous["frame"]
                ),
                "previous_valid_time_s": (
                    None if previous is None else previous["time_s"]
                ),
                "previous_valid_state": (
                    None if previous is None else previous["state"]
                ),
                "missing_frames_immediately_before_onset": (
                    index > 0
                    and any(
                        row["state"] == "missing"
                        for row in samples[:index]
                    )
                ),
                "onset_not_observable": True,
                "no_exact_contact_found": False,
            }

        if sample["state"] == "separated":
            seen_separated = True

        elif (
            sample["state"] == "contact"
            and previous_valid is not None
            and previous_valid["state"] == "separated"
        ):
            missing_before = (
                sample["frame"] > previous_valid["frame"] + 1
                or any(
                    row["state"] == "missing"
                    for row in samples[
                        previous_valid["index"] + 1:index
                    ]
                )
            )

            return {
                "first_contact_frame": sample["frame"],
                "first_contact_time_s": sample["time_s"],
                "previous_valid_frame": previous_valid["frame"],
                "previous_valid_time_s": previous_valid["time_s"],
                "previous_valid_state": previous_valid["state"],
                "missing_frames_immediately_before_onset": missing_before,
                "onset_not_observable": False,
                "no_exact_contact_found": False,
            }

        previous_valid = {**sample, "index": index}

    return {
        "first_contact_frame": None,
        "first_contact_time_s": None,
        "previous_valid_frame": (
            None if previous_valid is None else previous_valid["frame"]
        ),
        "previous_valid_time_s": (
            None if previous_valid is None else previous_valid["time_s"]
        ),
        "previous_valid_state": (
            None if previous_valid is None else previous_valid["state"]
        ),
        "missing_frames_immediately_before_onset": False,
        "onset_not_observable": False,
        "no_exact_contact_found": True,
    }


def _window_ends(gt_dir: Path):
    """Existing generated prediction windows and their observation cutoffs."""
    manifest = json.loads((gt_dir / "manifest.json").read_text(encoding="utf-8"))
    result = []
    for window in manifest.get("windows", []):
        ground_truth = json.loads((gt_dir / window["ground_truth"]).read_text(encoding="utf-8"))
        result.append((ground_truth["window"]["label"], float(ground_truth["window"]["t_end_s"])))
    return result


def _bucket_after_cutoff(time_s: float, cutoff_s: float):
    """1-based bucket under the repository's open-left, closed-right rule."""
    if time_s <= cutoff_s + 1e-9:
        return None
    return int(math.ceil(time_s - cutoff_s - 1e-9))


def audit_scenario(scenario, row, gt_run: str):
    collision = estimate_collision(scenario)
    pair = tuple(map(int, collision.carla_ids))
    if len(pair) != 2:
        raise ValueError(f"designated collision pair is not a two-ID pair: {pair}")
    samples = _pair_samples(scenario, pair)
    onset = find_first_onset(samples)
    current_time = _json_number(collision.time_s)
    onset_time = onset["first_contact_time_s"]
    delta = None if current_time is None or onset_time is None else current_time - onset_time

    existing_windows = _window_ends(Path(gt_run) / row["directory"])
    dropped_with_onset = 0
    newly_dropped = 0
    bucket_changed = 0
    for _, cutoff in existing_windows:
        exact_bucket = None if onset_time is None else _bucket_after_cutoff(onset_time, cutoff)
        current_bucket = None if current_time is None else _bucket_after_cutoff(current_time, cutoff)
        dropped_with_onset += exact_bucket is None and onset_time is not None
        newly_dropped += (
            onset_time is not None
            and exact_bucket is None
            and current_bucket is not None
        )
        bucket_changed += (exact_bucket is not None and current_bucket is not None
                           and exact_bucket != current_bucket)
    return {
        "scenario_type": scenario.scenario_type,
        "scenario": scenario.scenario,
        "carla_ids": list(pair),
        "current_collision_frame": collision.frame,
        "current_collision_time_s": current_time,
        "estimation_method": collision.method,
        "min_distance_m": _json_number(collision.min_distance_m),
        "exact_contact_definition": "raw CARLA oriented 3D boxes: BEV rectangles touch/overlap and vertical intervals overlap",
        **onset,
        "delta_s": delta,
        "raw_frame_counts": dict(Counter(sample["state"] for sample in samples)),
        "existing_prediction_windows": len(existing_windows),
        "existing_prediction_windows_dropped_with_exact_onset": dropped_with_onset,
        "existing_prediction_windows_newly_dropped_relative_to_current_time": newly_dropped,
        "existing_prediction_windows_with_changed_1s_bucket": bucket_changed,
    }


def _histogram(deltas):
    bins = Counter(math.floor(delta * 10 + 1e-9) / 10 for delta in deltas)
    return {
        f"[{start:.1f}, {start + 0.1:.1f})": bins[start]
        for start in sorted(bins)
    }


def summarize(records):
    deltas = [row["delta_s"] for row in records if row["delta_s"] is not None]
    return {
        "number_of_accident_scenarios": len(records),
        "number_with_observable_exact_onset": sum(not row["onset_not_observable"]
                                                   and not row["no_exact_contact_found"] for row in records),
        "number_with_no_exact_contact_found": sum(row["no_exact_contact_found"] for row in records),
        "number_with_onset_not_observable": sum(row["onset_not_observable"] for row in records),
        "delta_s_distribution": {
            "n": len(deltas),
            "min": None if not deltas else min(deltas),
            "max": None if not deltas else max(deltas),
            "mean": None if not deltas else statistics.mean(deltas),
            "median": None if not deltas else statistics.median(deltas),
            "count_lt_0": sum(delta < -1e-9 for delta in deltas),
            "count_eq_0": sum(math.isclose(delta, 0.0, abs_tol=1e-9) for delta in deltas),
            "count_gt_0": sum(delta > 1e-9 for delta in deltas),
        },
        "delta_s_histogram_0.1s": _histogram(deltas),
        "number_of_cases_where_difference_changes_1s_bucket": sum(
            row["existing_prediction_windows_with_changed_1s_bucket"] > 0 for row in records
        ),
        "number_of_existing_prediction_windows_with_changed_1s_bucket": sum(
            row["existing_prediction_windows_with_changed_1s_bucket"] for row in records
        ),
        "number_of_existing_prediction_windows_dropped_if_exact_onset_replaces_current_time": sum(
            row["existing_prediction_windows_newly_dropped_relative_to_current_time"] for row in records
        ),
        "number_of_existing_prediction_windows_at_or_after_exact_onset": sum(
            row["existing_prediction_windows_dropped_with_exact_onset"] for row in records
        ),
        "bucket_convention": "For a window ending at t_end, future bucket k is (t_end + k - 1, t_end + k]; collision at or before t_end drops that prediction window.",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="DeepAccident root")
    parser.add_argument("--gt-run", required=True,
                        help="existing validation GT run (cohort and window manifests only)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    gt_run = Path(args.gt_run)
    cohort = json.loads((gt_run / "batch_manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in cohort["scenarios"]
            if row.get("dataset_split") == "val" and row.get("outcome") == "accident"]
    scenarios = {raw_audit.scenario_key(s): s for s in scan_scenarios(args.root)}
    missing = [f"{row['scenario_type']}/{row['scenario']}" for row in rows
               if raw_audit.scenario_row_key(row) not in scenarios]
    if missing:
        raise KeyError(f"raw scenarios not found: {missing[:10]}")

    # The manifest's outcome is only a cohort selector.  Metadata remains the
    # authority for the designated pair, so mislabeled rows are not silently
    # treated as accidents.
    rows = [
        row for row in rows
        if scenarios[raw_audit.scenario_row_key(row)].meta.collision_occurred
        and scenarios[raw_audit.scenario_row_key(row)].meta.collision_id_a is not None
        and scenarios[raw_audit.scenario_row_key(row)].meta.collision_id_b is not None
    ]
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(audit_scenario, scenarios[raw_audit.scenario_row_key(row)], row, str(gt_run)): row
            for row in rows
        }
        for done, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            print(f"[{done}/{len(rows)}] complete {futures[future]['scenario']}", flush=True)
    records.sort(key=lambda row: (row["scenario_type"], row["scenario"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_scenario.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    summary = summarize(records)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
