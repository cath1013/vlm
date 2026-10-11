#!/usr/bin/env python3
"""Score frozen CRN5-A/V4 agreement without fitting, V4 replay, or API calls.

Run from any directory with Python and the project's inference dependencies.
Outputs go to a new directory; an existing output directory is never overwritten.
Both rules retain the CRN5-A selected pair when positive. GT is used only for
cohort eligibility and scoring, never to select pairs or make decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FROZEN = ROOT / "out/collision_risk_hazard/frozen/crn5_a_w01"
COLLECTOR = ROOT / "out/collision_risk_hazard/run_20261006_window_balanced/val104.jsonl"
CACHE = ROOT / "out/experiments/frozen_val104_replay_cache_v1.json"
CHECKPOINT_SHA256 = "9946c3480885dd3e7260c835c069553cd3f7cb5b382cb35e743919f795b2fddd"
EXPECTED = dict(TP=41, FP=29, TN=137, FN=12, correct_gt_pair_hits=31)
RULES = ("crn5_and_v4_any_pair", "crn5_and_v4_exact_pair")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcome(truth, prediction):
    return ("TP" if prediction else "FN") if truth else ("FP" if prediction else "TN")


def score(rows, prediction_key):
    counts = Counter(outcome(r["gt_positive_2s"], r[prediction_key]) for r in rows)
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    divide = lambda a, b: a / b if b else None
    return {**{k: counts[k] for k in ("TP", "FP", "TN", "FN")},
            "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
            "F1": divide(2 * tp, 2 * tp + fp + fn),
            "correct_gt_pair_hits": sum(r[prediction_key] and r["selected_pair_gt_hit"] for r in rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=Path, default=ROOT / "out/experiments" /
                        ("crn5_v4_agreement_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    args = parser.parse_args(argv)
    require(args.batch_size > 0, "batch size must be positive")
    require(not args.out.exists(), "Output directory must be new")

    import torch
    from examples import train_collision_risk_net as evaluator
    from traffic_llm.collision_risk_net import CollisionRiskNet, load_v2

    checkpoint = FROZEN / "collision_risk_final.pt"
    require(sha256(checkpoint) == CHECKPOINT_SHA256, "Frozen CRN5-A checkpoint changed")
    cache_hash = sha256(CACHE)
    document = json.loads(CACHE.read_text(encoding="utf-8"))
    cases = document["cases"]
    cached = {c["window_id"]: c for c in cases}
    require(len(cases) == len(cached) == 219, "Cache must contain 219 unique windows")
    saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
    threshold = saved["threshold_by_horizon"]["2"]
    oof = json.loads((FROZEN / "train_oof_2s_report.json").read_text())
    require(threshold == oof["threshold"], "Checkpoint threshold differs from saved train-OOF threshold")
    evaluator.verify_manifest(COLLECTOR, "val", smoke=False)
    collector_rows = evaluator.read_rows(COLLECTOR, "val")
    require(not set(saved["train_scenario_ids"]) & {r["scenario_id"] for r in collector_rows},
            "Train/validation scenario overlap")
    cohort = {r["window_id"]: r for r in collector_rows if evaluator.identifiable(r, 2)}
    require(len(cohort) == 219 and set(cohort) == set(cached),
            "Identifiable collector windows must match all 219 cached window_ids")
    model = CollisionRiskNet(load_v2(saved["v2_checkpoint"], args.device),
                             saved["hidden_dim"], saved["dropout"],
                             max_horizon=saved["max_horizon"]).to(args.device)
    model.load_state_dict(saved["state_dict"])
    with torch.inference_mode():
        predictions = evaluator.predict(model, collector_rows, args.device, args.batch_size)
    records = {r["window_id"]: r for r in predictions}
    require(set(cohort) <= set(records), "Missing CRN5-A predictions")
    rows = []
    for wid in sorted(cohort):
        record, source = records[wid], cohort[wid]
        pair = record["selected_pairs"][1]
        truth = 0 < source["gt_bucket"] <= 2
        rows.append(dict(window_id=wid, scenario_id=source["scenario_id"],
                         gt_bucket=source["gt_bucket"], gt_positive_2s=truth,
                         crn5_risk_2s=record["horizon_risks"][1], crn5_threshold_2s=threshold,
                         crn5_selected_pair=pair,
                         crn5_positive=record["horizon_risks"][1] >= threshold,
                         selected_pair_gt_hit=truth and record["selected_pair_gt_labels"][1] == source["gt_bucket"]))
    baseline = score(rows, "crn5_positive")
    require(all(baseline[k] == v for k, v in EXPECTED.items()),
            f"CRN5-A baseline mismatch: {baseline}; expected {EXPECTED}")

    # Only frozen accepted pairs and CRN predictions determine the rules.
    for row in rows:
        case = cached[row["window_id"]]
        accepted = case["v4"]["evidence"]["accepted_interactions"]
        require(case["v4"]["decision"] == bool(accepted), "Inconsistent cached V4 decision")
        require(all(p["k"] in (1, 2) for p in accepted), "V4 acceptance outside 2-second horizon")
        pairs = sorted({tuple(sorted((p["actor_a"], p["actor_b"]))) for p in accepted})
        selected = row["crn5_selected_pair"]
        exact = selected is not None and tuple(sorted(selected)) in pairs
        row.update(v4_positive=bool(pairs), v4_accepted_pairs=pairs, v4_accepts_crn5_pair=exact)
        row[RULES[0]] = row["crn5_positive"] and bool(pairs)
        row[RULES[1]] = row["crn5_positive"] and exact
        row["crn5_outcome"] = outcome(row["gt_positive_2s"], row["crn5_positive"])
        for rule in RULES:
            row[rule + "_transition"] = row["crn5_outcome"] + "->" + outcome(row["gt_positive_2s"], row[rule])
    summaries = {}
    for rule in RULES:
        metrics = score(rows, rule)
        metrics.update(FPs_eliminated=baseline["FP"] - metrics["FP"],
                       TPs_lost=baseline["TP"] - metrics["TP"],
                       transitions=dict(sorted(Counter(r[rule + "_transition"] for r in rows).items())))
        summaries[rule] = metrics
    require(sha256(CACHE) == cache_hash, "Replay cache changed during evaluation")
    summary = dict(window_count=len(rows), horizon_s=2, threshold_source="saved train-OOF",
                   threshold_2s=threshold, baseline=baseline, rules=summaries,
                   pair_scoring="Positive decisions retain the CRN5-A maximum-risk selected pair",
                   sources={"checkpoint": str(checkpoint), "checkpoint_sha256": CHECKPOINT_SHA256,
                            "collector": str(COLLECTOR), "collector_sha256": sha256(COLLECTOR),
                            "v4_cache": str(CACHE), "v4_cache_sha256": cache_hash},
                   device=args.device, batch_size=args.batch_size)
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "per_window.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, tuple)) or v is None else v
                             for k, v in row.items()})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved diagnostics to {args.out}")


if __name__ == "__main__":
    main()
