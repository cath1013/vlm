"""Evaluate frozen bucket 1 and 2 V4 pair verifiers on official val104.

This script only performs inference. Each bucket gets its own clearance-ranked
top-50 pool, matching the interval-specific V4 collection procedure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from traffic_llm.accident_qa import WindowConfig, build_windows, window_ground_truth  # noqa: E402
from traffic_llm.carla_map import find_xodr  # noqa: E402
from traffic_llm.config import PipelineConfig  # noqa: E402
from traffic_llm.da_runner import DeepAccidentRunner  # noqa: E402
from traffic_llm.deepaccident import estimate_collision  # noqa: E402
from traffic_llm.pair_reranker_v2 import PairRerankerV4, pair_features_v4  # noqa: E402
from traffic_llm.predict_model import JointSceneTorchPredictor  # noqa: E402
from traffic_llm.serialize import rank_actors  # noqa: E402
from traffic_llm.swept_path import swept_pair_clearances  # noqa: E402

DEFAULT_BUCKET1_MODEL = ROOT / "out/pair_reranker_v4/model_bucket1/pair_reranker_v4.json"
DEFAULT_BUCKET2_MODEL = ROOT / "out/pair_reranker_v4/model_bucket2/pair_reranker_v4.json"
CANDIDATE_POOL_CAP = 50


def evaluation_truth(gt):
    """Return (positive bucket, identity groups), or None if censored."""
    rows = {row["k"]: row for row in gt.get("expected", [])}
    positive = next((k for k in (1, 2)
                     if k in rows and rows[k].get("accident_expected")), None)
    if positive is not None:
        return positive, [set(vehicle.get("actor_ids") or [])
                          for vehicle in rows[positive].get("involved_vehicles") or []]
    if all(k in rows and rows[k].get("scorable") for k in (1, 2)):
        return None, []
    return None


def bucket_pools(pairs):
    """Filter each interval before applying its candidate cap."""
    return {k: [pair for pair in pairs if pair.interval_index == k][:CANDIDATE_POOL_CAP]
            for k in (1, 2)}


def matches_gt_pair(candidate, groups):
    a, b = candidate["actor_a"], candidate["actor_b"]
    return any((a in groups[i] and b in groups[j]) or
               (b in groups[i] and a in groups[j])
               for i in range(len(groups)) for j in range(i + 1, len(groups)))


def evaluate_window(gt, candidates, models):
    truth = evaluation_truth(gt)
    if truth is None:
        return None
    positive_bucket, groups = truth
    covered = positive_bucket is not None and any(
        matches_gt_pair(candidate, groups)
        for bucket in (1, 2) for candidate in candidates[bucket]
    )
    selected = {bucket: [candidate for candidate in candidates[bucket]
                         if models[bucket].predict_features(candidate["features"])
                         >= models[bucket].threshold]
                for bucket in (1, 2)}
    baseline = [candidate for bucket in (1, 2) for candidate in candidates[bucket]
                if candidate["predicted_contact"]]
    chosen = selected[1] + selected[2]
    return {
        "actual": positive_bucket is not None,
        "positive_bucket": positive_bucket,
        "bucket1_prediction": bool(selected[1]),
        "bucket2_prediction": bool(selected[2]),
        "predicted": bool(chosen),
        "baseline_predicted": bool(baseline),
        "gt_pair_covered": covered,
        "correct_pair_hit": positive_bucket is not None and any(
            matches_gt_pair(candidate, groups) for candidate in chosen),
        "baseline_correct_pair_hit": positive_bucket is not None and any(
            matches_gt_pair(candidate, groups) for candidate in baseline),
    }


def metric_report(rows, prediction_key, hit_key):
    n = len(rows)
    positives = sum(row["actual"] for row in rows)
    negatives = n - positives
    tp = sum(row["actual"] and row[prediction_key] for row in rows)
    fp = sum(not row["actual"] and row[prediction_key] for row in rows)
    tn = negatives - fp
    fn = positives - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    specificity = tn / negatives if negatives else 0.0
    hits = sum(row[hit_key] for row in rows)
    coverage = sum(row["gt_pair_covered"] for row in rows)
    return {
        "valid_0_2_windows": n,
        "positive_windows": positives,
        "negative_windows": negatives,
        "bucket1_positives": sum(row["positive_bucket"] == 1 for row in rows),
        "bucket2_positives": sum(row["positive_bucket"] == 2 for row in rows),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": (tp + tn) / n if n else 0.0,
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "correct_gt_actor_pair_hits": hits,
        "correct_gt_actor_pair_recall": hits / positives if positives else 0.0,
        "gt_pair_candidate_coverage_count": coverage,
        "gt_pair_candidate_coverage": coverage / positives if positives else 0.0,
        "correct_pair_recall_given_coverage": hits / coverage if coverage else 0.0,
    }


def load_frozen_models(bucket1_path, bucket2_path):
    models = {1: PairRerankerV4.load(str(bucket1_path)),
              2: PairRerankerV4.load(str(bucket2_path))}
    for bucket, path in ((1, bucket1_path), (2, bucket2_path)):
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if doc.get("training", {}).get("target_interval") != bucket:
            raise ValueError(f"{path} is not a bucket {bucket} model")
        if models[bucket].threshold != doc["decision_threshold"]:
            raise ValueError(f"{path} decision threshold was not preserved")
    return models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="DeepAccident root")
    parser.add_argument("--carla-maps", required=True)
    parser.add_argument("--predictor", required=True, help="V2 JointScene checkpoint")
    parser.add_argument("--bucket1-model", default=str(DEFAULT_BUCKET1_MODEL))
    parser.add_argument("--bucket2-model", default=str(DEFAULT_BUCKET2_MODEL))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True, help="output JSON")
    args = parser.parse_args(argv)

    models = load_frozen_models(args.bucket1_model, args.bucket2_model)
    cfg = PipelineConfig()
    cfg.deepaccident.observation_mode = "sensor3d"
    cfg.predictor = JointSceneTorchPredictor(args.predictor, device=args.device)
    runner = DeepAccidentRunner(args.root, cfg)
    scenarios = sorted((s for s in runner.list_scenarios() if s.split == "val"),
                       key=lambda s: s.scenario_id)
    if len(scenarios) != 104:
        raise ValueError(f"official val104 requires 104 scenarios; found {len(scenarios)}")
    wcfg = WindowConfig(window_s=5.0, stride_s=1.0, horizon_s=5.0,
                        snapshot_rate_hz=2.0, warmup=False, full_window=False)
    rows = []
    for index, scenario in enumerate(scenarios, 1):
        xodr = find_xodr(scenario.town, [args.carla_maps])
        result = runner.build(scenario.scenario, scenario.scenario_type,
                              opendrive_path=xodr)
        snapshots = list(result.snapshots(rate_hz=2.0))
        collision = estimate_collision(scenario, cfg.deepaccident)
        collision_time = collision.time_s if collision and collision.occurred else None
        windows, _ = build_windows(snapshots, wcfg, collision_time_s=collision_time)
        agent_ids = {agent: scenario.meta.agent_id_of(agent) for agent in scenario.agents}
        for window in windows:
            gt = window_ground_truth(
                window, wcfg, collision=collision, agent_carla_ids=agent_ids,
                scenario_id=scenario.scenario_id, scenario_split=scenario.scenario_type,
                dataset_split=scenario.split, town=scenario.town,
                data_end_s=snapshots[-1].t,
            )
            if evaluation_truth(gt) is None:
                continue
            actors, _ = rank_actors(window.last, wcfg.actor_cap(len(window.last.actors)))
            actor_by_id = {actor.actor_id: actor for actor in actors}
            pairs = swept_pair_clearances(
                actors, horizon_s=wcfg.horizon_s,
                sample_dt_s=wcfg.swept_sample_dt_s,
                contact_margin_m=wcfg.swept_contact_margin_m,
                exclude_touching_now=wcfg.swept_exclude_touching_now,
                exclude_static_pairs=wcfg.swept_exclude_static_pairs,
            )
            candidates = {1: [], 2: []}
            for bucket, pool in bucket_pools(pairs).items():
                for pair in pool:
                    candidates[bucket].append({
                        "actor_a": pair.actor_a, "actor_b": pair.actor_b,
                        "predicted_contact": bool(pair.predicted_contact),
                        "features": pair_features_v4(
                            actor_by_id[pair.actor_a], actor_by_id[pair.actor_b],
                            pair, wcfg.horizon_s, window.last.interactions),
                    })
            rows.append(evaluate_window(gt, candidates, models))
        print(f"[{index:03d}/{len(scenarios):03d}] {scenario.scenario_id} "
              f"valid_windows={len(rows)}", flush=True)

    report = {
        "number_of_val_scenarios": len(scenarios),
        "setup": {"split": "val", "observation_mode": "sensor3d",
                  "window_s": 5.0, "stride_s": 1.0, "predictor_mode": "joint_scene",
                  "feature_version": 4, "candidate_pool_cap_per_bucket": CANDIDATE_POOL_CAP,
                  "deepaccident_root": str(Path(args.root).resolve()),
                  "carla_maps": str(Path(args.carla_maps).resolve()),
                  "device": args.device,
                  "predictor": str(Path(args.predictor).resolve()),
                  "bucket1_model": str(Path(args.bucket1_model).resolve()),
                  "bucket2_model": str(Path(args.bucket2_model).resolve()),
                  "bucket1_decision_threshold": models[1].threshold,
                  "bucket2_decision_threshold": models[2].threshold},
        "combined_verifier": metric_report(rows, "predicted", "correct_pair_hit"),
        "baseline_before_verifier": metric_report(
            rows, "baseline_predicted", "baseline_correct_pair_hit"),
        "diagnostics": {
            "bucket1_predicted_positive_windows": sum(r["bucket1_prediction"] for r in rows),
            "bucket2_predicted_positive_windows": sum(r["bucket2_prediction"] for r in rows),
            "positive_by_bucket1_only": sum(r["bucket1_prediction"] and not r["bucket2_prediction"] for r in rows),
            "positive_by_bucket2_only": sum(r["bucket2_prediction"] and not r["bucket1_prediction"] for r in rows),
            "positive_by_both": sum(r["bucket1_prediction"] and r["bucket2_prediction"] for r in rows),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
