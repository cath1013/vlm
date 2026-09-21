from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def compact_from_payload(path: Path) -> dict:
    """Gemini request wrapper 안의 compact_geometry_v3 JSON을 꺼낸다."""
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


def load_run(root: Path):
    """
    Returns one record per scorable GT bucket.

    Key:
        (scenario_id, window_label, k)
    """
    records = {}

    for gt_path in root.glob("val/*/*/ground_truth_*.json"):
        gt = json.loads(gt_path.read_text(encoding="utf-8"))

        scenario = gt["scenario"]["id"]
        outcome = gt["scenario"]["outcome"]
        label = gt["window"]["label"]

        payload_path = gt_path.with_name(
            gt_path.name.replace("ground_truth_", "llm_payload_")
        )
        if not payload_path.is_file():
            raise FileNotFoundError(payload_path)

        compact = compact_from_payload(payload_path)

        cols = compact["predicted_pair_columns"]
        col = {name: i for i, name in enumerate(cols)}
        raw_pairs = compact.get("predicted_closest_pairs", [])

        contact_pairs_by_k = {}
        for row in raw_pairs:
            if not row[col["predicted_contact"]]:
                continue

            k = int(row[col["interval_index"]])
            a = str(row[col["actor_a"]])
            b = str(row[col["actor_b"]])
            pair = frozenset((a, b))

            contact_pairs_by_k.setdefault(k, []).append({
                "pair": pair,
                "a": a,
                "b": b,
                "clearance": float(row[col["minimum_clearance_m"]]),
                "time_s": float(row[col["time_after_observation_s"]]),
            })

        cap = int(
            compact.get("predicted_pair_method", {}).get(
                "pairs_serialized_cap", 0
            ) or 0
        )

        # If every serialized row is a contact and we hit the cap,
        # there could theoretically be additional contact pairs that
        # were truncated. Window/bucket "any contact" remains valid,
        # but pair counts should then be treated as a lower bound.
        possible_truncation = (
            cap > 0
            and len(raw_pairs) >= cap
            and sum(
                bool(r[col["predicted_contact"]])
                for r in raw_pairs
            ) >= cap
        )

        for exp in gt.get("expected", []):
            if not exp.get("scorable", False):
                continue

            k = int(exp["k"])
            gt_positive = bool(exp["accident_expected"])
            gt_ids = frozenset(exp.get("involved_actor_ids") or [])

            contacts = contact_pairs_by_k.get(k, [])
            pred_any = bool(contacts)

            target_pair_hit = False
            if gt_positive and len(gt_ids) >= 2:
                target_pair_hit = any(
                    p["pair"] == gt_ids for p in contacts
                )

            key = (scenario, label, k)
            records[key] = {
                "scenario": scenario,
                "outcome": outcome,
                "label": label,
                "k": k,
                "gt_positive": gt_positive,
                "gt_actor_ids": sorted(gt_ids),
                "pred_any_contact": pred_any,
                "target_pair_hit": target_pair_hit,
                "contact_pairs": contacts,
                "possible_pair_truncation": possible_truncation,
            }

    return records


def summarize(records):
    c = Counter()
    fp_pairs = 0
    scenarios_with_fp = set()
    windows_with_fp = set()
    truncated = set()

    for key, r in records.items():
        gt = r["gt_positive"]
        pred = r["pred_any_contact"]

        if gt and pred:
            c["tp_any"] += 1
        elif gt and not pred:
            c["fn_any"] += 1
        elif not gt and pred:
            c["fp"] += 1
        else:
            c["tn"] += 1

        if gt:
            c["positive_buckets"] += 1
            if r["target_pair_hit"]:
                c["target_pair_hits"] += 1

        if not gt and pred:
            scenarios_with_fp.add(r["scenario"])
            windows_with_fp.add((r["scenario"], r["label"]))
            fp_pairs += len(r["contact_pairs"])

        if r["possible_pair_truncation"]:
            truncated.add((r["scenario"], r["label"]))

    precision = c["tp_any"] / max(1, c["tp_any"] + c["fp"])
    recall = c["tp_any"] / max(1, c["tp_any"] + c["fn_any"])
    target_recall = (
        c["target_pair_hits"] / max(1, c["positive_buckets"])
    )

    return {
        "n_buckets": len(records),
        "TP_any_contact": c["tp_any"],
        "FP_contact": c["fp"],
        "TN_no_contact": c["tn"],
        "FN_no_contact": c["fn_any"],
        "any_contact_precision": precision,
        "any_contact_recall": recall,
        "positive_GT_buckets": c["positive_buckets"],
        "GT_actor_pair_hits": c["target_pair_hits"],
        "GT_actor_pair_recall": target_recall,
        "FP_contact_pair_rows": fp_pairs,
        "windows_with_FP_contact": len(windows_with_fp),
        "scenarios_with_FP_contact": len(scenarios_with_fp),
        "possibly_truncated_windows": len(truncated),
    }


def print_summary(name, s):
    print(f"\n=== {name} ===")
    for k, v in s.items():
        if isinstance(v, float):
            print(f"{k:30s} {v:.4f}")
        else:
            print(f"{k:30s} {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("control")
    ap.add_argument("pair005")
    ap.add_argument("--changed-out", default="out/contact_changed_buckets.jsonl")
    args = ap.parse_args()

    control = load_run(Path(args.control))
    pair005 = load_run(Path(args.pair005))

    if set(control) != set(pair005):
        only_a = set(control) - set(pair005)
        only_b = set(pair005) - set(control)
        raise SystemExit(
            f"bucket sets differ: control-only={len(only_a)}, "
            f"pair005-only={len(only_b)}"
        )

    s0 = summarize(control)
    s1 = summarize(pair005)

    print_summary("CONTROL", s0)
    print_summary("PAIR LOSS 0.05", s1)

    base_fp = {
        k for k, r in control.items()
        if (not r["gt_positive"]) and r["pred_any_contact"]
    }
    new_fp = {
        k for k, r in pair005.items()
        if (not r["gt_positive"]) and r["pred_any_contact"]
    }

    resolved = base_fp - new_fp
    introduced = new_fp - base_fp
    persistent = base_fp & new_fp

    base_target = {
        k for k, r in control.items()
        if r["gt_positive"] and r["target_pair_hit"]
    }
    new_target = {
        k for k, r in pair005.items()
        if r["gt_positive"] and r["target_pair_hit"]
    }

    print("\n=== DIRECT COMPARISON ===")
    print(f"Baseline FP buckets          : {len(base_fp)}")
    print(f"Pair005 FP buckets           : {len(new_fp)}")
    print(f"Resolved FP buckets          : {len(resolved)}")
    print(f"Newly introduced FP buckets  : {len(introduced)}")
    print(f"Persistent FP buckets        : {len(persistent)}")
    print()
    print(f"Baseline GT-pair hits         : {len(base_target)}")
    print(f"Pair005 GT-pair hits          : {len(new_target)}")
    print(f"GT-pair hits lost             : {len(base_target - new_target)}")
    print(f"GT-pair hits gained           : {len(new_target - base_target)}")

    out = Path(args.changed_out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for status, keys in (
            ("resolved_fp", resolved),
            ("new_fp", introduced),
            ("persistent_fp", persistent),
            ("lost_gt_pair_hit", base_target - new_target),
            ("gained_gt_pair_hit", new_target - base_target),
        ):
            for key in sorted(keys):
                row = {
                    "status": status,
                    "key": list(key),
                    "control": control[key],
                    "pair005": pair005[key],
                }
                f.write(json.dumps(
                    row,
                    ensure_ascii=False,
                    default=lambda o: (
                        sorted(o)
                        if isinstance(o, (set, frozenset))
                        else str(o)
                    ),
                ) + "\n")

    print(f"\nChanged buckets written to: {out}")


if __name__ == "__main__":
    main()
