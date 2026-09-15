"""Evaluate compact-v3 swept-path pair geometry without calling an LLM.

The input is a validation-batch directory produced by run_validation_batch.py.
Only the deterministic ``predicted_closest_pairs`` rows and held-out ground-truth
files are read.  The script writes ``swept_geometry_scores.json`` beside the batch
manifest and prints the same report.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def _compact_doc(payload: dict) -> dict:
    """Extract the first user-data block from supported provider request shapes."""
    if "contents" in payload:  # Gemini
        text = payload["contents"][0]["parts"][0]["text"]
    elif "messages" in payload:  # Claude or OpenAI
        user = next(m for m in payload["messages"] if m.get("role") == "user")
        content = user["content"]
        if isinstance(content, str):
            text = content
        else:
            block = next(x for x in content if x.get("type") == "text")
            text = block["text"]
    else:
        raise ValueError("unsupported provider payload")
    return json.loads(text)


def _actual_pair_groups(gt: dict):
    positive = next(
        (row for row in gt.get("expected", []) if row.get("accident_expected")),
        None,
    )
    if positive is None:
        return []
    groups = []
    for vehicle in positive.get("involved_vehicles") or []:
        groups.append(set(vehicle.get("actor_ids") or []))
    return groups


def _pair_matches(pair, groups) -> bool:
    if len(groups) < 2:
        return False
    a, b = pair[0], pair[1]
    return any(
        (a in groups[i] and b in groups[j]) or
        (b in groups[i] and a in groups[j])
        for i in range(len(groups))
        for j in range(i + 1, len(groups))
    )


def _division(n, d):
    return n / d if d else 0.0


def evaluate(base: Path, thresholds) -> dict:
    cohort = json.loads((base / "batch_manifest.json").read_text(encoding="utf-8"))
    records = []
    for scenario in cohort["scenarios"]:
        directory = base / scenario["directory"]
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for window in manifest["windows"]:
            payload = json.loads(
                (directory / window["payload"]).read_text(encoding="utf-8")
            )
            gt = json.loads(
                (directory / window["ground_truth"]).read_text(encoding="utf-8")
            )
            compact = _compact_doc(payload)
            columns = compact["predicted_pair_columns"]
            index = {name: i for i, name in enumerate(columns)}
            records.append({
                "scenario": scenario["scenario"],
                "window": window["label"],
                "actual": any(x.get("accident_expected") for x in gt["expected"]),
                "actual_k": next(
                    (x["k"] for x in gt["expected"] if x.get("accident_expected")),
                    None,
                ),
                "groups": _actual_pair_groups(gt),
                "pairs": compact["predicted_closest_pairs"],
                "index": index,
            })

    reports = []
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        correct_bucket = early_bucket = late_bucket = 0
        actor_pair_hits = 0
        positive_with_any_contact = 0
        for row in records:
            ix = row["index"]
            hits = [
                p for p in row["pairs"]
                if float(p[ix["minimum_clearance_m"]]) <= threshold
            ]
            predicted = bool(hits)
            actual = row["actual"]
            if predicted and actual:
                tp += 1
            elif predicted:
                fp += 1
            elif actual:
                fn += 1
            else:
                tn += 1

            if actual and hits:
                positive_with_any_contact += 1
                ks = {int(p[ix["interval_index"]]) for p in hits}
                actual_k = row["actual_k"]
                correct_bucket += int(actual_k in ks)
                early_bucket += int(any(k < actual_k for k in ks))
                late_bucket += int(any(k > actual_k for k in ks))
                actor_pair_hits += int(any(
                    _pair_matches(p, row["groups"]) for p in hits
                ))

        precision = _division(tp, tp + fp)
        recall = _division(tp, tp + fn)
        reports.append({
            "clearance_threshold_m": threshold,
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "accuracy": _division(tp + tn, len(records)),
            "precision": precision,
            "recall": recall,
            "f1": _division(2 * precision * recall, precision + recall),
            "positive_windows_with_any_geometry_hit": positive_with_any_contact,
            "positive_windows_with_correct_actor_pair": actor_pair_hits,
            "positive_windows_with_hit_in_true_bucket": correct_bucket,
            "positive_windows_with_any_earlier_hit": early_bucket,
            "positive_windows_with_any_later_hit": late_bucket,
        })

    zero_hit_windows = []
    false_pair_classes = Counter()
    false_contact_buckets = Counter()
    for row in records:
        ix = row["index"]
        contacts = [
            p for p in row["pairs"]
            if float(p[ix["minimum_clearance_m"]]) <= 0.0
        ]
        if not contacts:
            continue
        zero_hit_windows.append(len(contacts))
        if not row["actual"]:
            false_contact_buckets.update(
                int(p[ix["interval_index"]]) for p in contacts
            )
            # Actor id prefixes are a useful compact diagnostic without copying
            # full payload actor tables into the report.
            false_pair_classes.update(
                "-".join(sorted((p[0].split("_")[0], p[1].split("_")[0])))
                for p in contacts
            )

    return {
        "source": str(base.resolve()),
        "method": "compact_geometry_v3 predicted_closest_pairs; no LLM calls",
        "n_windows": len(records),
        "n_positive": sum(r["actual"] for r in records),
        "n_negative": sum(not r["actual"] for r in records),
        "zero_clearance_diagnostics": {
            "windows_with_contact": len(zero_hit_windows),
            "mean_contact_pairs_per_firing_window": (
                sum(zero_hit_windows) / len(zero_hit_windows)
                if zero_hit_windows else 0.0
            ),
            "false_contact_interval_counts": dict(sorted(false_contact_buckets.items())),
            "false_contact_actor_prefix_pairs": dict(false_pair_classes.most_common()),
        },
        "thresholds": reports,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir")
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0]
    )
    args = parser.parse_args(argv)
    base = Path(args.batch_dir)
    report = evaluate(base, args.thresholds)
    (base / "swept_geometry_scores.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
