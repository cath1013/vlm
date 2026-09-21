from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def compact_from_payload(path: Path) -> dict:
    req = json.loads(path.read_text(encoding="utf-8"))

    for content in req.get("contents", []):
        for part in content.get("parts", []):
            text = part.get("text")
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text.startswith("{"):
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            if obj.get("representation") == "compact_geometry_v3":
                return obj

    raise ValueError(f"compact payload not found: {path}")


def load_buckets(root: Path):
    buckets = []

    for gt_path in root.glob("val/*/*/ground_truth_*.json"):
        gt = json.loads(gt_path.read_text(encoding="utf-8"))

        payload_path = gt_path.with_name(
            gt_path.name.replace("ground_truth_", "llm_payload_")
        )
        compact = compact_from_payload(payload_path)

        cols = compact["predicted_pair_columns"]
        ci = {name: i for i, name in enumerate(cols)}

        contacts_by_k = {}

        for row in compact.get("predicted_closest_pairs", []):
            if not bool(row[ci["predicted_contact"]]):
                continue

            k = int(row[ci["interval_index"]])

            contacts_by_k.setdefault(k, []).append({
                "a": str(row[ci["actor_a"]]),
                "b": str(row[ci["actor_b"]]),
                "time": float(row[ci["time_after_observation_s"]]),
                "obs_clearance": float(row[ci["clearance_at_observation_m"]]),
                "min_clearance": float(row[ci["minimum_clearance_m"]]),
            })

        outcome = gt["scenario"]["outcome"]

        for exp in gt.get("expected", []):
            if not exp.get("scorable", False):
                continue

            k = int(exp["k"])

            buckets.append({
                "scenario": gt["scenario"]["id"],
                "outcome": outcome,
                "window": gt["window"]["label"],
                "k": k,
                "gt_positive": bool(exp["accident_expected"]),
                "gt_ids": frozenset(exp.get("involved_actor_ids") or []),
                "contacts": contacts_by_k.get(k, []),
            })

    return buckets


def evaluate(buckets, max_time, max_obs_clearance):
    tp = fp = tn = fn = 0
    gt_pair_hits = 0
    positive = 0

    normal_fp = 0
    normal_fp_windows = set()
    normal_fp_scenarios = set()

    accepted_rows = 0

    for b in buckets:
        accepted = [
            c for c in b["contacts"]
            if c["time"] <= max_time
            and c["obs_clearance"] <= max_obs_clearance
        ]

        pred = bool(accepted)
        gt = b["gt_positive"]

        accepted_rows += len(accepted)

        if gt:
            positive += 1

            if pred:
                tp += 1
            else:
                fn += 1

            if len(b["gt_ids"]) >= 2:
                for c in accepted:
                    pair = frozenset((c["a"], c["b"]))
                    if pair == b["gt_ids"]:
                        gt_pair_hits += 1
                        break
        else:
            if pred:
                fp += 1
            else:
                tn += 1

            if pred and b["outcome"] == "normal":
                normal_fp += 1
                normal_fp_windows.add(
                    (b["scenario"], b["window"])
                )
                normal_fp_scenarios.add(b["scenario"])

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (
        2 * precision * recall / max(1e-12, precision + recall)
    )

    pair_recall = gt_pair_hits / max(1, positive)

    return {
        "max_time_s": max_time,
        "max_observation_clearance_m": max_obs_clearance,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "GT_pair_hits": gt_pair_hits,
        "GT_pair_recall": pair_recall,
        "normal_FP_buckets": normal_fp,
        "normal_FP_windows": len(normal_fp_windows),
        "normal_FP_scenarios": len(normal_fp_scenarios),
        "accepted_contact_rows": accepted_rows,
    }


def pct(x):
    return f"{100*x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "root",
        nargs="?",
        default="out/experiments/contact_audit_control_val104",
    )
    ap.add_argument(
        "--out",
        default="out/contact_gate_sweep.csv",
    )
    args = ap.parse_args()

    buckets = load_buckets(Path(args.root))

    print(f"Loaded {len(buckets)} scorable buckets")

    # Baseline = effectively no gate.
    baseline = evaluate(
        buckets,
        max_time=999.0,
        max_obs_clearance=999999.0,
    )

    print("\n=== BASELINE / NO GATE ===")
    print(
        f"TP={baseline['TP']} FP={baseline['FP']} "
        f"TN={baseline['TN']} FN={baseline['FN']}"
    )
    print(
        f"precision={pct(baseline['precision'])} "
        f"recall={pct(baseline['recall'])} "
        f"F1={pct(baseline['f1'])}"
    )
    print(
        f"GT-pair recall={pct(baseline['GT_pair_recall'])} "
        f"({baseline['GT_pair_hits']} hits)"
    )
    print(
        f"normal FP buckets={baseline['normal_FP_buckets']} "
        f"windows={baseline['normal_FP_windows']} "
        f"scenarios={baseline['normal_FP_scenarios']}"
    )

    times = [
        1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 999.0
    ]

    clearances = [
        5.0, 7.5, 10.0, 12.5, 15.0,
        20.0, 25.0, 30.0, 40.0, 60.0, 999999.0
    ]

    rows = []

    for t in times:
        for d in clearances:
            rows.append(evaluate(buckets, t, d))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Prefer configurations that retain most of the baseline GT-pair recall.
    base_pair_recall = baseline["GT_pair_recall"]
    base_recall = baseline["recall"]

    candidates = [
        r for r in rows
        if r["GT_pair_recall"] >= base_pair_recall - 0.05
        and r["recall"] >= base_recall - 0.05
    ]

    candidates.sort(
        key=lambda r: (
            r["normal_FP_buckets"],
            -r["GT_pair_recall"],
            -r["precision"],
        )
    )

    print("\n=== BEST GATES WITH <=5pp RECALL LOSS ===")

    if not candidates:
        print("No gate satisfies the <=5 percentage-point recall constraint.")
    else:
        for r in candidates[:15]:
            t = (
                "none"
                if r["max_time_s"] > 100
                else f"{r['max_time_s']:.1f}s"
            )
            d = (
                "none"
                if r["max_observation_clearance_m"] > 1000
                else f"{r['max_observation_clearance_m']:.1f}m"
            )

            print(
                f"time<={t:5s} current_clearance<={d:6s} | "
                f"normalFP={r['normal_FP_buckets']:3d} | "
                f"precision={pct(r['precision']):>6s} "
                f"recall={pct(r['recall']):>6s} "
                f"GTpair={pct(r['GT_pair_recall']):>6s}"
            )

    print(f"\nFull sweep written to: {out}")


if __name__ == "__main__":
    main()
