"""Calibrate only bucket 2 using scenario-grouped official TRAIN OOF scores.

The saved models are read only. Missing GT actor groups are recovered from
official TRAIN data using the collection pipeline. Alternatively, supply a
TRAIN-only --train-gt JSONL sidecar (split, scenario, window, gt_actor_groups).
Interval-specific labels cannot identify a GT pair from the other bucket.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.train_pair_reranker_v2 import fit, fold_of, load_records, predict  # noqa: E402
from examples.evaluate_pair_reranker_v4_0_2 import (  # noqa: E402
    DeepAccidentRunner, JointSceneTorchPredictor, PipelineConfig, WindowConfig,
    build_windows, estimate_collision, evaluation_truth, find_xodr,
    load_frozen_models, matches_gt_pair, metric_report, window_ground_truth,
)
from traffic_llm.pair_reranker_v2 import FEATURE_NAMES_V4  # noqa: E402

BASE = ROOT / "out/pair_reranker_v4"
EXPECTED_COUNTS = {"valid_0_2_windows": 1106, "positive_windows": 152,
                   "negative_windows": 954}
FOLDS = 5
SEED = 20260902


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def window_key(doc):
    return doc["scenario"], doc["window"]


def index_records(records, bucket):
    indexed = {}
    for doc in records:
        key = window_key(doc)
        if key in indexed:
            raise ValueError(f"duplicate bucket {bucket} window: {key}")
        if doc.get("target_interval") != bucket:
            raise ValueError(f"{key}: expected target_interval={bucket}")
        if "positive_intervals" not in doc:
            raise ValueError(f"{key}: missing interval-specific GT")
        if any(c["interval_index"] != bucket for c in doc["candidates"]):
            raise ValueError(f"{key}: candidate from another bucket")
        if bool(doc["actual"]) != (bucket in doc["positive_intervals"]):
            raise ValueError(f"{key}: inconsistent bucket GT")
        indexed[key] = doc
    return indexed


def build_population(bucket1, bucket2):
    """Presence in an interval collection means that bucket was scorable.

    The collector omits unscorable buckets. Keep a known early collision even
    if the other bucket is missing; require both records for a negative.
    """
    population = []
    for key in sorted(bucket1.keys() | bucket2.keys()):
        docs = {1: bucket1.get(key), 2: bucket2.get(key)}
        observed = [doc for doc in docs.values() if doc is not None]
        if len(observed) == 2 and (observed[0]["positive_intervals"] !=
                                   observed[1]["positive_intervals"]):
            raise ValueError(f"{key}: bucket GT labels disagree")
        positive_bucket = next((b for b in (1, 2)
                                if b in observed[0]["positive_intervals"]), None)
        if positive_bucket is None and len(observed) != 2:
            continue
        if positive_bucket is not None and docs[positive_bucket] is None:
            raise ValueError(f"{key}: missing scorable collision bucket")
        population.append({"key": key, "actual": positive_bucket is not None,
                           "positive_bucket": positive_bucket, "docs": docs})
    return population


def population_counts(population):
    positives = sum(row["actual"] for row in population)
    return {"valid_0_2_windows": len(population), "positive_windows": positives,
            "negative_windows": len(population) - positives}


def check_population(population):
    counts = population_counts(population)
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"TRAIN census mismatch: expected {EXPECTED_COUNTS}, got {counts}")
    return counts


def generate_oof(samples, windows, hidden, weight_decay):
    """Reuse the trainer's hashing, fitting, normalization, and per-fold seed."""
    x = np.asarray([s["x"] for s in samples], dtype=np.float64)
    y = np.asarray([s["y"] for s in samples], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES_V4) or not np.isfinite(x).all():
        raise ValueError("invalid V4 feature matrix")
    assignments = np.asarray([fold_of(s["scenario"], FOLDS) for s in samples])
    oof = np.full(len(samples), np.nan, dtype=np.float64)
    audit = []
    for fold in range(FOLDS):
        train_ix, test_ix = assignments != fold, assignments == fold
        training_scenarios = {s["scenario"] for s, keep in zip(samples, train_ix) if keep}
        held_out_scenarios = {w["scenario"] for w in windows
                              if fold_of(w["scenario"], FOLDS) == fold}
        if training_scenarios & held_out_scenarios:
            raise ValueError("scenario leakage in OOF training")
        if not train_ix.any() or not y[train_ix].any():
            raise ValueError(f"fold {fold}: no positive training candidates")
        model, means, scales = fit(x[train_ix], y[train_ix], hidden, weight_decay,
                                  SEED + fold)
        if test_ix.any():
            oof[test_ix] = predict(model, means, scales, x[test_ix])
        audit.append({"fold": fold, "seed": SEED + fold,
                      "training_scenarios": sorted(training_scenarios),
                      "held_out_scenarios": sorted(held_out_scenarios),
                      "training_candidates": int(train_ix.sum()),
                      "held_out_candidates": int(test_ix.sum())})
        print(f"  fold {fold + 1}/{FOLDS}: {int(test_ix.sum())} OOF candidates", flush=True)
    if not np.isfinite(oof).all():
        raise ValueError("missing or nonfinite OOF predictions")
    return oof, audit


def attach_scores(population, bucket_scores, gt_groups):
    for row in population:
        groups = gt_groups.get(row["key"])
        if row["actual"] and groups is None:
            groups = next((d["gt_actor_groups"] for d in row["docs"].values()
                           if d is not None and "gt_actor_groups" in d), None)
        if row["actual"] and groups is None:
            raise ValueError(f"{row['key']}: missing GT actor groups; supply --train-gt")
        groups = [set(group) for group in (groups or [])]
        row["scores"], row["pair_scores"] = {}, {}
        for bucket in (1, 2):
            doc = row["docs"][bucket]
            scores = bucket_scores[bucket].get(row["key"], np.asarray([]))
            candidates = doc["candidates"] if doc is not None else []
            if len(scores) != len(candidates):
                raise ValueError("OOF candidate alignment mismatch")
            row["scores"][bucket] = float(max(scores, default=-np.inf))
            row["pair_scores"][bucket] = float(max(
                (score for candidate, score in zip(candidates, scores)
                 if matches_gt_pair(candidate, groups)), default=-np.inf))
        row["gt_pair_covered"] = row["actual"] and any(
            np.isfinite(score) for score in row["pair_scores"].values())


def combined_metrics(population, bucket1_threshold, bucket2_threshold):
    rows = []
    for row in population:
        p1 = row["scores"][1] >= bucket1_threshold
        p2 = row["scores"][2] >= bucket2_threshold
        rows.append({**row, "predicted": p1 or p2,
                     "bucket1_prediction": p1, "bucket2_prediction": p2,
                     "correct_pair_hit": row["actual"] and (
                         row["pair_scores"][1] >= bucket1_threshold or
                         row["pair_scores"][2] >= bucket2_threshold)})
    report = metric_report(rows, "predicted", "correct_pair_hit")
    report["alarms"] = {
        "bucket1_only": sum(r["bucket1_prediction"] and not r["bucket2_prediction"] for r in rows),
        "bucket2_only": sum(r["bucket2_prediction"] and not r["bucket1_prediction"] for r in rows),
        "both": sum(r["bucket1_prediction"] and r["bucket2_prediction"] for r in rows),
    }
    return report


def sweep_bucket2(population, bucket1_threshold, original_bucket2_threshold):
    original = combined_metrics(population, bucket1_threshold, original_bucket2_threshold)
    scores = [r["scores"][2] for r in population if np.isfinite(r["scores"][2])]
    thresholds = set(scores) | {original_bucket2_threshold}
    # >= includes the largest score. Also consider the boundary with no bucket2 alarms.
    if scores:
        thresholds.add(float(np.nextafter(max(scores), np.inf)))
    sweep = []
    for threshold in sorted(thresholds):
        result = combined_metrics(population, bucket1_threshold, threshold)
        sweep.append({"bucket1_threshold": bucket1_threshold,
                      "bucket2_threshold": threshold,
                      "eligible": result["recall"] >= original["recall"], **result})
    eligible = [row for row in sweep if row["eligible"]]
    best = max(eligible, key=lambda r: (r["specificity"], r["f1"], r["precision"],
                                       r["bucket2_threshold"]))
    threshold = (best["bucket2_threshold"] if best["specificity"] > original["specificity"]
                 else original_bucket2_threshold)
    selected = combined_metrics(population, bucket1_threshold, threshold)
    return original, threshold, selected, sweep


def load_training_config(report_path, bucket):
    # Validation fields in this existing report are never consulted.
    training = json.loads(Path(report_path).read_text(encoding="utf-8"))["training"]
    if (training["folds"] != FOLDS or training["feature_version"] != 4 or
            training["target_interval"] != bucket):
        raise ValueError(f"{report_path}: incompatible training configuration")
    return {"hidden": int(training["hidden"]),
            "weight_decay": float(training["weight_decay"])}


def recover_scenario_gt(scenario_id, wanted, train_root, predictor, carla_maps):
    train_root = Path(train_root).resolve()
    if train_root.name != "train" or not train_root.is_dir():
        raise ValueError("GT recovery requires an explicit official TRAIN directory")
    torch.set_num_threads(1)
    cfg = PipelineConfig()
    cfg.deepaccident.observation_mode = "sensor3d"
    cfg.predictor = JointSceneTorchPredictor(predictor, device="cpu")
    runner = DeepAccidentRunner(str(train_root), cfg)
    wcfg = WindowConfig(window_s=5.0, stride_s=1.0, horizon_s=5.0,
                        snapshot_rate_hz=2.0, warmup=False, full_window=False)
    groups_by_key = {}
    scenario_type, scenario_name = scenario_id.split("/", 1)
    town = scenario_name.split("_", 1)[0]
    xodr = find_xodr(town, [str(carla_maps)])
    if xodr is None:
        raise ValueError(f"missing collection OpenDRIVE map for {town}")
    result = runner.build(scenario_name, scenario_type, opendrive_path=xodr)
    scenario = result.scenario
    if scenario.split != "train" or scenario.scenario_id != scenario_id:
        raise ValueError("GT recovery encountered a non-TRAIN scenario")
    snapshots = list(result.snapshots(rate_hz=2.0))
    collision = estimate_collision(scenario, cfg.deepaccident)
    windows, _ = build_windows(snapshots, wcfg, collision_time_s=(
        collision.time_s if collision and collision.occurred else None))
    agent_ids = {agent: scenario.meta.agent_id_of(agent) for agent in scenario.agents}
    for window in windows:
        if window.label not in wanted:
            continue
        row = wanted[window.label]
        gt = window_ground_truth(window, wcfg, collision=collision,
                                 agent_carla_ids=agent_ids, data_end_s=snapshots[-1].t)
        truth = evaluation_truth(gt)
        if truth is None or truth[0] != row["positive_bucket"]:
            raise ValueError(f"{row['key']}: reconstructed GT disagrees with collection")
        groups = truth[1]
        for bucket, doc in row["docs"].items():
            if doc is not None and any(bool(c["label"]) != (
                    bucket == row["positive_bucket"] and matches_gt_pair(c, groups))
                    for c in doc["candidates"]):
                raise ValueError(f"{row['key']}: reconstructed pair identities disagree with labels")
        groups_by_key[row["key"]] = [sorted(group) for group in groups]
    if any((scenario_id, label) not in groups_by_key for label in wanted):
        raise ValueError(f"{scenario_id}: unable to reconstruct all positive windows")
    return groups_by_key


def recover_train_gt(population, manifest, carla_maps, workers=4):
    """Reconstruct positive TRAIN windows; never scan the dataset parent.

    Actor IDs are pipeline identities, not CARLA IDs, so metadata alone does
    not preserve the evaluator's alias-aware pair definition. Independent
    scenarios can be reconstructed concurrently without changing their IDs.
    """
    root = Path(manifest["dataset_root"]).resolve()
    train_root = root if root.name == "train" else (root / "train").resolve()
    if train_root.name != "train" or not train_root.is_dir():
        raise ValueError("GT recovery requires an explicit official TRAIN directory")
    requested = {}
    for row in population:
        if row["actual"]:
            requested.setdefault(row["key"][0], {})[row["key"][1]] = row
    groups_by_key = {}
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(recover_scenario_gt, scenario_id, wanted, train_root,
                               manifest["predictor"], carla_maps): scenario_id
                   for scenario_id, wanted in sorted(requested.items())}
        for index, future in enumerate(as_completed(futures), 1):
            groups_by_key.update(future.result())
            print(f"GT identities {index}/{len(requested)}: {futures[future]}", flush=True)
    return groups_by_key, {"source": "official_train_pipeline",
                           "train_root": str(train_root), "carla_maps": str(carla_maps),
                           "predictor": manifest["predictor"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for bucket in (1, 2):
        parser.add_argument(f"--train-bucket{bucket}", default=str(BASE / f"train_bucket{bucket}.jsonl"))
        parser.add_argument(f"--bucket{bucket}-report", default=str(BASE / f"model_bucket{bucket}/training_report_v4.json"))
        parser.add_argument(f"--bucket{bucket}-model", default=str(BASE / f"model_bucket{bucket}/pair_reranker_v4.json"))
    parser.add_argument("--train-gt", help="TRAIN-only JSONL with GT actor identity groups")
    parser.add_argument("--carla-maps", default=str(ROOT / "carla_map"),
                        help="collection maps for recovering missing TRAIN GT identities")
    parser.add_argument("--gt-workers", type=int, default=4,
                        help="processes for independent TRAIN identity recovery")
    parser.add_argument("--out", required=True, help="diagnostic output JSON; never a model path")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    inputs = [Path(getattr(args, f"train_bucket{b}")) for b in (1, 2)]
    inputs += [path.with_suffix(".manifest.json") for path in inputs]
    inputs += [Path(getattr(args, f"bucket{b}_{kind}")) for b in (1, 2) for kind in ("report", "model")]
    if args.train_gt:
        inputs.append(Path(args.train_gt))
    if any(part == "val" or part.startswith("val_") or "val104" in part
           for path in inputs for part in path.resolve().parts):
        raise ValueError("validation input paths are forbidden in TRAIN calibration")
    out = Path(args.out)
    if out.name == "pair_reranker_v4.json" or out.resolve() in {p.resolve() for p in inputs}:
        raise ValueError("output must not overwrite a model or an input")
    if args.threads < 1 or args.gt_workers < 1:
        raise ValueError("--threads and --gt-workers must be positive")
    torch.set_num_threads(args.threads)
    datasets, manifests = {}, {}
    for bucket in (1, 2):
        path = Path(getattr(args, f"train_bucket{bucket}"))
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("split") != "train" or manifest.get("feature_version") != 4 or
                manifest.get("predictor_mode") != "joint_scene" or
                manifest.get("target_interval") != bucket):
            raise ValueError(f"{path}: requires official TRAIN JointScene V4 bucket {bucket}")
        datasets[bucket] = index_records(read_jsonl(path), bucket)
        manifests[bucket] = manifest
    population = build_population(datasets[1], datasets[2])
    counts = check_population(population)
    print(f"TRAIN population: {counts}", flush=True)
    gt_groups = {}
    if args.train_gt:
        for doc in read_jsonl(args.train_gt):
            if doc.get("split") != "train" or window_key(doc) not in datasets[1].keys() | datasets[2].keys():
                raise ValueError("GT sidecar must contain only matching official TRAIN windows")
            if window_key(doc) in gt_groups:
                raise ValueError("duplicate GT sidecar window")
            gt_groups[window_key(doc)] = doc["gt_actor_groups"]
    missing_gt = [row for row in population if row["actual"] and
                  row["key"] not in gt_groups and not any(
                      d is not None and "gt_actor_groups" in d for d in row["docs"].values())]
    gt_source = {"source": "train_sidecar" if args.train_gt else "train_records"}
    if missing_gt:
        if args.train_gt:
            raise ValueError("TRAIN GT sidecar is missing positive windows")
        if any(manifests[1][key] != manifests[2][key] for key in ("dataset_root", "predictor")):
            raise ValueError("bucket collection sources disagree")
        recovered, gt_source = recover_train_gt(missing_gt, manifests[1], args.carla_maps,
                                               args.gt_workers)
        gt_groups.update(recovered)
    models = load_frozen_models(args.bucket1_model, args.bucket2_model)
    thresholds = {b: models[b].threshold for b in (1, 2)}
    bucket_scores, audits, configs = {}, {}, {}
    for bucket in (1, 2):
        configs[bucket] = load_training_config(getattr(args, f"bucket{bucket}_report"), bucket)
        samples, windows = load_records(getattr(args, f"train_bucket{bucket}"), bucket)
        print(f"Bucket {bucket}: {configs[bucket]}", flush=True)
        scores, audits[bucket] = generate_oof(samples, windows, **configs[bucket])
        bucket_scores[bucket] = {key: scores[w["start"]:w["stop"]]
                                 for key, w in zip(datasets[bucket], windows)}
    attach_scores(population, bucket_scores, gt_groups)
    original, threshold, selected, sweep = sweep_bucket2(population, thresholds[1], thresholds[2])
    fp_reduction = original["confusion"]["fp"] - selected["confusion"]["fp"]
    report = {
        "train_population_counts": counts,
        "setup": {"split": "train", "folds": FOLDS, "seed": SEED,
                  "epochs": 350, "threads": args.threads,
                  "selected_configurations": configs,
                  "inputs": [str(p.resolve()) for p in inputs],
                  "gt_identity_source": gt_source,
                  "pair_definition": "any above-threshold candidate in either bucket matching two distinct GT identity groups"},
        "original_bucket1_threshold": thresholds[1],
        "original_bucket2_threshold": thresholds[2],
        "selected_bucket1_threshold": thresholds[1],
        "selected_bucket2_threshold": threshold,
        "original_combined_oof_metrics": original,
        "selected_combined_oof_metrics": selected,
        "absolute_fp_reduction": fp_reduction,
        "percent_fp_reduction": 100 * fp_reduction / original["confusion"]["fp"] if original["confusion"]["fp"] else 0.0,
        "tp_change": selected["confusion"]["tp"] - original["confusion"]["tp"],
        "fn_change": selected["confusion"]["fn"] - original["confusion"]["fn"],
        "correct_pair_recall_change": selected["correct_gt_actor_pair_recall"] - original["correct_gt_actor_pair_recall"],
        "bucket2_only_alarms_before": original["alarms"]["bucket2_only"],
        "bucket2_only_alarms_after": selected["alarms"]["bucket2_only"],
        "recall_floor": original["recall"],
        "selection_rule": "recall >= original recall; maximize specificity, then F1, precision, and bucket2 threshold; retain original if specificity does not improve",
        "threshold_sweep": sweep, "oof_fold_audit": audits,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("threshold_sweep", "oof_fold_audit")}, indent=2))


if __name__ == "__main__":
    main()
