#!/usr/bin/env python3
"""Audit contact-event types in a saved compact validation output.

This is read-only with respect to generated payloads and ground truth.  It
writes ``summary.json`` and ``fp_contact_rows.jsonl`` to ``--out`` (the input
directory by default).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


EVENTS = ("NEW_CONTACT", "BOUNDARY_ONSET", "RECONTACT")


def compact_scene(payload_path: Path) -> dict[str, Any]:
    """Extract the compact scene JSON from an LLM request payload."""
    request = json.loads(payload_path.read_text(encoding="utf-8"))
    for content in request.get("contents", []):
        for part in content.get("parts", []):
            value = part.get("text")
            if not isinstance(value, str):
                continue
            try:
                scene = json.loads(value)
            except json.JSONDecodeError:
                continue
            if "predicted_pair_columns" in scene and "predicted_closest_pairs" in scene:
                return scene
    raise ValueError(f"compact scene not found in {payload_path}")


def pair_rows(scene: dict[str, Any], payload_path: Path) -> Iterable[dict[str, Any]]:
    columns = scene.get("predicted_pair_columns")
    if not isinstance(columns, list):
        raise ValueError(f"missing predicted_pair_columns: {payload_path}")
    required = {
        "actor_a", "actor_b", "minimum_clearance_m",
        "clearance_at_observation_m", "interval_index", "predicted_contact",
        "contact_event", "first_contact_s", "contact_duration_s",
    }
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"missing pair columns in {payload_path}: {', '.join(missing)}")
    if len(set(columns)) != len(columns):
        raise ValueError(f"duplicate predicted_pair_columns in {payload_path}")

    for number, values in enumerate(scene.get("predicted_closest_pairs", [])):
        if len(values) != len(columns):
            raise ValueError(
                f"pair row {number} has {len(values)} values for {len(columns)} columns: "
                f"{payload_path}"
            )
        yield dict(zip(columns, values))


def event_counts(counter: Counter[str]) -> dict[str, int]:
    return {event: counter[event] for event in EVENTS}


def json_number(value: Any) -> Any:
    """Keep nulls null and make emitted floating-point diagnostics tidy."""
    return None if value is None else round(float(value), 3)


def audit(root: Path, out_dir: Path) -> dict[str, Any]:
    totals = Counter()
    fp_rows_by_event: Counter[str] = Counter()
    fp_buckets_by_event: Counter[str] = Counter()
    all_negative_by_event: Counter[str] = Counter()
    all_positive_by_event: Counter[str] = Counter()
    target_hit_buckets_by_event: Counter[str] = Counter()
    target_hit_rows_by_event: Counter[str] = Counter()
    fp_records: list[dict[str, Any]] = []

    gt_paths = sorted(root.rglob("ground_truth_*.json"))
    if not gt_paths:
        raise FileNotFoundError(f"no ground_truth_*.json files under {root}")

    for gt_path in gt_paths:
        payload_path = gt_path.with_name(
            gt_path.name.replace("ground_truth_", "llm_payload_", 1)
        )
        if not payload_path.is_file():
            raise FileNotFoundError(f"missing corresponding payload: {payload_path}")
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        scene = compact_scene(payload_path)
        rows_by_k: dict[int, list[dict[str, Any]]] = {}
        for row in pair_rows(scene, payload_path):
            if not bool(row["predicted_contact"]):
                continue
            event = row["contact_event"]
            if event not in EVENTS:
                raise AssertionError(
                    f"predicted_contact=True has invalid contact_event {event!r} "
                    f"in {payload_path} ({row['actor_a']}, {row['actor_b']})"
                )
            rows_by_k.setdefault(int(row["interval_index"]), []).append(row)

        scenario = gt.get("scenario", {}).get("id")
        window_label = gt.get("window", {}).get("label")
        for expected in gt.get("expected", []):
            if not expected.get("scorable", False):
                continue
            totals["n_buckets"] += 1
            k = int(expected["k"])
            gt_positive = bool(expected["accident_expected"])
            rows = rows_by_k.get(k, [])
            by_event = {event: [row for row in rows if row["contact_event"] == event]
                        for event in EVENTS}

            if gt_positive:
                totals["positive_gt_buckets"] += 1
                for event, event_rows in by_event.items():
                    all_positive_by_event[event] += len(event_rows)

                # A target pair is an exact unordered pair of the two GT actor ids.
                target = frozenset(expected.get("involved_actor_ids") or [])
                hits = [row for row in rows
                        if len(target) == 2
                        and frozenset((row["actor_a"], row["actor_b"])) == target]
                if hits:
                    totals["gt_actor_pair_hits"] += 1
                    for event in {row["contact_event"] for row in hits}:
                        target_hit_buckets_by_event[event] += 1
                for row in hits:
                    target_hit_rows_by_event[row["contact_event"]] += 1
                continue

            for event, event_rows in by_event.items():
                all_negative_by_event[event] += len(event_rows)
            if not rows:
                continue

            totals["fp_buckets"] += 1
            totals["fp_pair_rows"] += len(rows)
            for event, event_rows in by_event.items():
                fp_rows_by_event[event] += len(event_rows)
                if event_rows:
                    fp_buckets_by_event[event] += 1
            for row in rows:
                fp_records.append({
                    "scenario": scenario,
                    "window_label": window_label,
                    "k": k,
                    "actor_a": row["actor_a"],
                    "actor_b": row["actor_b"],
                    "contact_event": row["contact_event"],
                    "minimum_clearance_m": json_number(row["minimum_clearance_m"]),
                    "clearance_at_observation_m": json_number(
                        row["clearance_at_observation_m"]
                    ),
                    "first_contact_s": json_number(row["first_contact_s"]),
                    "contact_duration_s": json_number(row["contact_duration_s"]),
                })

    summary = {
        "n_buckets": totals["n_buckets"],
        "positive_gt_buckets": totals["positive_gt_buckets"],
        "fp_buckets": totals["fp_buckets"],
        "fp_pair_rows": totals["fp_pair_rows"],
        "fp_pair_rows_by_event": event_counts(fp_rows_by_event),
        "fp_buckets_with_event": event_counts(fp_buckets_by_event),
        "gt_actor_pair_hits": totals["gt_actor_pair_hits"],
        "gt_actor_pair_hits_by_event": event_counts(target_hit_buckets_by_event),
        "all_predicted_contact_pair_rows_by_gt_bucket": {
            "gt_negative": event_counts(all_negative_by_event),
            "gt_positive": event_counts(all_positive_by_event),
        },
        # Extra row-level counterpart to the required bucket-level target-hit totals.
        "gt_actor_pair_hit_rows_by_event": event_counts(target_hit_rows_by_event),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "fp_contact_rows.jsonl").open("w", encoding="utf-8") as output:
        for record in fp_records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_output_dir", type=Path)
    parser.add_argument("--out", type=Path,
                        help="directory for summary.json and fp_contact_rows.jsonl "
                             "(default: validation output directory)")
    args = parser.parse_args()
    root = args.validation_output_dir.resolve()
    summary = audit(root, args.out.resolve() if args.out else root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
