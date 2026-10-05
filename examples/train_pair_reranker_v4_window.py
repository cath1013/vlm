"""Controlled bucket-2 V4 window-objective experiment using official TRAIN only.

Architecture, features, normalization, optimizer, epochs and scenario folds are
unchanged. Select each threshold and the objective on combined TRAIN OOF
predictions with frozen bucket1 threshold and TP >= 116. Existing model files
are read only. Exact combined pair metrics reuse TRAIN GT identity recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples import calibrate_pair_reranker_v4_0_2 as calibration  # noqa: E402
from examples.train_pair_reranker_v2 import MLP, fit, fold_of, load_records, metrics, predict  # noqa: E402
from traffic_llm.pair_reranker_v2 import FEATURE_NAMES_V4, PairRerankerV4  # noqa: E402

BASE = ROOT / "out/pair_reranker_v4"
DEFAULT_OUT = ROOT / "out/pair_reranker_v4_window/model_bucket2"
HIDDEN = 32
WEIGHT_DECAY = 0.001
EPOCHS = 350
FOLDS = 5
SEED = 20260902
RANKING_MARGIN = 1.0
TP_FLOOR = 116
OBJECTIVE_GRID = ((0.0, 0.0), (0.25, 0.10), (0.50, 0.25), (1.00, 0.50))
OUTPUT_NAMES = ("pair_reranker_v4.json", "training_report_v4_window.json",
                "oof_diagnostics.json", "oof_candidate_scores.npz", "oof_windows.jsonl",
                "train_gt_actor_groups.jsonl", "train_gt_actor_groups.meta.json")


@dataclass
class WindowLayout:
    indices: torch.Tensor
    valid: torch.Tensor
    gt_positive: torch.Tensor
    negative_rows: torch.Tensor
    positive_rows: torch.Tensor
    ranking_rows: torch.Tensor


def make_window_layout(windows, labels):
    """Precompute padded indices so full-batch losses need no Python window loop."""
    labels = np.asarray(labels, dtype=bool)
    width = max((w["stop"] - w["start"] for w in windows), default=0)
    indices = np.zeros((len(windows), width), dtype=np.int64)
    valid = np.zeros_like(indices, dtype=bool)
    for row, window in enumerate(windows):
        start, stop = window["start"], window["stop"]
        if not 0 <= start <= stop <= len(labels):
            raise ValueError("invalid window candidate bounds")
        indices[row, :stop - start] = np.arange(start, stop)
        valid[row, :stop - start] = True
    gt = valid & labels[indices] if indices.size else valid.copy()
    actual = np.asarray([w["actual"] for w in windows], dtype=bool)
    positive = actual & gt.any(axis=1)
    return WindowLayout(
        indices=torch.from_numpy(indices), valid=torch.from_numpy(valid),
        gt_positive=torch.from_numpy(gt),
        negative_rows=torch.from_numpy(~actual & valid.any(axis=1)),
        positive_rows=torch.from_numpy(positive),
        ranking_rows=torch.from_numpy(positive & (valid & ~gt).any(axis=1)),
    )


def window_and_ranking_losses(logits, layout, ranking_margin=RANKING_MARGIN):
    zero = logits.sum() * 0.0
    if not layout.indices.numel():
        return zero, zero
    padded = logits[layout.indices]
    hard = padded.masked_fill(~layout.valid, -torch.inf).max(dim=1).values
    positive = padded.masked_fill(~layout.gt_positive, -torch.inf).max(dim=1).values
    wrong = padded.masked_fill(~(layout.valid & ~layout.gt_positive), -torch.inf).max(dim=1).values
    groups = []
    if layout.negative_rows.any():
        groups.append(F.softplus(hard[layout.negative_rows]).mean())
    if layout.positive_rows.any():
        groups.append(F.softplus(-positive[layout.positive_rows]).mean())
    window_loss = torch.stack(groups).mean() if groups else zero
    ranking_loss = (F.softplus(ranking_margin + wrong[layout.ranking_rows]
                               - positive[layout.ranking_rows]).mean()
                    if layout.ranking_rows.any() else zero)
    return window_loss, ranking_loss


def training_objective(logits, labels, pos_weight, layout, lambda_window, lambda_rank):
    pair_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    if lambda_window == 0.0 and lambda_rank == 0.0:
        return pair_loss
    window_loss, ranking_loss = window_and_ranking_losses(logits, layout)
    return pair_loss + lambda_window * window_loss + lambda_rank * ranking_loss


def fit_window(x, y, windows, lambda_window, lambda_rank, seed, epochs=EPOCHS):
    if lambda_window == 0.0 and lambda_rank == 0.0:
        # Delegate the control to the original fitting code for exact equality.
        return fit(x, y, HIDDEN, WEIGHT_DECAY, seed, epochs=epochs)
    torch.manual_seed(seed)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-6] = 1.0
    tx = torch.tensor((x - means) / scales, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    positives = max(float(y.sum()), 1.0)
    pos_weight = torch.tensor((len(y) - positives) / positives)
    layout = make_window_layout(windows, y)
    model = MLP(x.shape[1], HIDDEN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=WEIGHT_DECAY)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = training_objective(model(tx), ty, pos_weight, layout, lambda_window, lambda_rank)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite window training loss")
        loss.backward()
        optimizer.step()
    return model, means, scales


def training_windows_for_fold(windows, fold):
    result, cursor = [], 0
    for window in windows:
        if fold_of(window["scenario"], FOLDS) == fold:
            continue
        size = window["stop"] - window["start"]
        result.append({**window, "start": cursor, "stop": cursor + size})
        cursor += size
    return result


def generate_oof(samples, windows, lambda_window, lambda_rank):
    x = np.asarray([s["x"] for s in samples], dtype=np.float64)
    y = np.asarray([s["y"] for s in samples], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES_V4) or not np.isfinite(x).all():
        raise ValueError("invalid FEATURE_NAMES_V4 matrix")
    assignments = np.asarray([fold_of(s["scenario"], FOLDS) for s in samples])
    scores = np.full(len(samples), np.nan, dtype=np.float64)
    audit = []
    for fold in range(FOLDS):
        train_ix, test_ix = assignments != fold, assignments == fold
        training_windows = training_windows_for_fold(windows, fold)
        train_scenarios = {s["scenario"] for s, keep in zip(samples, train_ix) if keep}
        held_scenarios = {w["scenario"] for w in windows
                          if fold_of(w["scenario"], FOLDS) == fold}
        if train_scenarios & held_scenarios or any(
                w["scenario"] in held_scenarios for w in training_windows):
            raise ValueError("scenario leakage in OOF training")
        if sum(w["stop"] - w["start"] for w in training_windows) != int(train_ix.sum()):
            raise ValueError("fold window/candidate alignment mismatch")
        if not train_ix.any() or not y[train_ix].any():
            raise ValueError(f"fold {fold}: no positive training candidates")
        model, means, scales = fit_window(x[train_ix], y[train_ix], training_windows,
                                         lambda_window, lambda_rank, SEED + fold)
        if test_ix.any():
            scores[test_ix] = predict(model, means, scales, x[test_ix])
        audit.append({"fold": fold, "seed": SEED + fold,
                      "training_scenarios": sorted(train_scenarios),
                      "held_out_scenarios": sorted(held_scenarios),
                      "training_candidates": int(train_ix.sum()),
                      "held_out_candidates": int(test_ix.sum())})
        print(f"  fold {fold + 1}/{FOLDS}: {int(test_ix.sum())} held-out candidates", flush=True)
    if not np.isfinite(scores).all():
        raise ValueError("missing or nonfinite OOF scores")
    return scores, audit


def bucket2_metrics(scores, samples, windows, threshold):
    maxima, pair_maxima, actual, covered = [], [], [], []
    negative_scores = []
    empty_negatives = 0
    for window in windows:
        start, stop = window["start"], window["stop"]
        window_scores = scores[start:stop]
        maximum = float(max(window_scores, default=-np.inf))
        gt_scores = [scores[i] for i in range(start, stop) if samples[i]["y"] > 0.5]
        maxima.append(maximum)
        pair_maxima.append(float(max(gt_scores, default=-np.inf)))
        actual.append(bool(window["actual"]))
        covered.append(bool(window["actual"] and gt_scores))
        if not window["actual"]:
            # An empty pool cannot alarm. Its maximum probability is reported as 0.
            negative_scores.append(maximum if np.isfinite(maximum) else 0.0)
            empty_negatives += int(stop == start)
    report = metrics(actual, np.asarray(maxima) >= threshold)
    positives = sum(actual)
    coverage = sum(covered)
    hits = int(np.sum(np.asarray(actual) & (np.asarray(pair_maxima) >= threshold)))
    report.update({"windows": len(windows), "positive_windows": positives,
                   "negative_windows": len(windows) - positives,
                   "correct_gt_actor_pair_hits": hits,
                   "correct_gt_actor_pair_recall": hits / positives if positives else 0.0,
                   "gt_pair_candidate_coverage_count": coverage,
                   "gt_pair_candidate_coverage": coverage / positives if positives else 0.0,
                   "conditional_correct_pair_recall": hits / coverage if coverage else 0.0,
                   "positive_windows_with_gt_pair_absent": positives - coverage,
                   "negative_window_score_mean": float(np.mean(negative_scores)) if negative_scores else 0.0,
                   "negative_window_score_max": max(negative_scores, default=0.0),
                   "negative_windows_without_candidates": empty_negatives})
    return report


def selection_key(row):
    result = row["combined_oof"]
    return (-result["confusion"]["fp"], result["correct_gt_actor_pair_recall"], result["f1"],
            -(row["lambda_window"] + row["lambda_rank"]),
            -row["lambda_window"], -row["lambda_rank"])


def select_configuration(rows, tp_floor=TP_FLOOR):
    eligible = [row for row in rows if row["combined_oof"]["confusion"]["tp"] >= tp_floor]
    if not eligible:
        raise ValueError(f"no objective satisfies combined OOF TP >= {tp_floor}")
    return max(eligible, key=selection_key)


def choose_combined_threshold(population, bucket1_threshold, original_bucket2_threshold):
    scores = [r["scores"][2] for r in population if np.isfinite(r["scores"][2])]
    thresholds = set(scores) | {original_bucket2_threshold}
    if scores:
        thresholds.add(float(np.nextafter(max(scores), np.inf)))
    sweep, best = [], None
    for threshold in sorted(thresholds):
        result = calibration.combined_metrics(population, bucket1_threshold, threshold)
        eligible = result["confusion"]["tp"] >= TP_FLOOR
        sweep.append({"bucket2_threshold": threshold, "eligible": eligible,
                      **result["confusion"], "precision": result["precision"],
                      "recall": result["recall"], "specificity": result["specificity"],
                      "f1": result["f1"], "balanced_accuracy": result["balanced_accuracy"],
                      "correct_gt_actor_pair_hits": result["correct_gt_actor_pair_hits"],
                      "correct_gt_actor_pair_recall": result["correct_gt_actor_pair_recall"]})
        key = (-result["confusion"]["fp"], result["correct_gt_actor_pair_recall"], result["f1"], threshold)
        if eligible and (best is None or key > best[0]):
            best = key, threshold, result
    if best is None:
        raise ValueError(f"no threshold satisfies combined OOF TP >= {TP_FLOOR}")
    return best[1], best[2], sweep


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_paths(inputs, out):
    for path in inputs:
        if any(part == "val" or part.startswith("val_") or "val104" in part
               for part in Path(path).resolve().parts):
            raise ValueError("validation input paths are forbidden")
    protected = {Path(p).resolve() for p in inputs}
    old = BASE.resolve()
    for target in [Path(out), *(Path(out) / name for name in OUTPUT_NAMES)]:
        resolved = target.resolve()
        if resolved == old or old in resolved.parents or resolved in protected:
            raise ValueError("output cannot overwrite existing V4 artifacts or inputs")


def load_gt_groups(population, datasets, manifests, source_hashes, args, out):
    path = Path(args.train_gt) if args.train_gt else out / "train_gt_actor_groups.jsonl"
    meta_path = out / "train_gt_actor_groups.meta.json"
    metadata = None
    if not args.train_gt and path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("source_train_sha256") != source_hashes:
            metadata = None
    groups = {}
    if args.train_gt or metadata is not None:
        train_keys = datasets[1].keys() | datasets[2].keys()
        for doc in calibration.read_jsonl(path):
            key = calibration.window_key(doc)
            if doc.get("split") != "train" or key not in train_keys or key in groups:
                raise ValueError("GT sidecar requires unique matching TRAIN windows")
            groups[key] = doc["gt_actor_groups"]
        source = metadata["gt_identity_source"] if metadata else {"source": "train_sidecar", "path": str(path.resolve())}
    else:
        if any(manifests[1][key] != manifests[2][key] for key in ("dataset_root", "predictor")):
            raise ValueError("bucket collection sources disagree")
        groups, source = calibration.recover_train_gt(population, manifests[1], args.carla_maps,
                                                      workers=args.gt_workers)
        with path.open("w", encoding="utf-8") as handle:
            for (scenario, window), actor_groups in sorted(groups.items()):
                handle.write(json.dumps({"split": "train", "scenario": scenario, "window": window,
                                         "gt_actor_groups": actor_groups}) + "\n")
        meta_path.write_text(json.dumps({"source_train_sha256": source_hashes,
                                        "gt_identity_source": source}, indent=2) + "\n", encoding="utf-8")
    for row in population:
        if not row["actual"]:
            continue
        if row["key"] not in groups:
            raise ValueError("TRAIN GT sidecar is missing positive windows")
        identity_groups = [set(group) for group in groups[row["key"]]]
        for bucket, doc in row["docs"].items():
            if doc is not None and any(bool(c["label"]) != (
                    bucket == row["positive_bucket"] and calibration.matches_gt_pair(c, identity_groups))
                    for c in doc["candidates"]):
                raise ValueError("TRAIN GT identities disagree with candidate labels")
    return groups, source


def score_map(datasets, windows, scores):
    if len(datasets) != len(windows):
        raise ValueError("OOF window alignment mismatch")
    return {key: scores[w["start"]:w["stop"]] for key, w in zip(datasets, windows)}


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(BASE / "train_bucket2.jsonl"))
    parser.add_argument("--bucket1-train", default=str(BASE / "train_bucket1.jsonl"))
    parser.add_argument("--bucket1-model", default=str(BASE / "model_bucket1/pair_reranker_v4.json"))
    parser.add_argument("--current-bucket2-model", default=str(BASE / "model_bucket2/pair_reranker_v4.json"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--train-gt", help="optional TRAIN-only GT identity JSONL")
    parser.add_argument("--carla-maps", default=str(ROOT / "carla_map"))
    parser.add_argument("--gt-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    train_paths = {1: Path(args.bucket1_train), 2: Path(args.train)}
    model_paths = {1: Path(args.bucket1_model), 2: Path(args.current_bucket2_model)}
    inputs = [*train_paths.values(), *model_paths.values(),
              *(p.with_suffix(".manifest.json") for p in train_paths.values())]
    if args.train_gt:
        inputs.append(Path(args.train_gt))
    out = Path(args.out)
    validate_paths(inputs, out)
    if args.threads < 1 or args.gt_workers < 1:
        raise ValueError("threads and GT workers must be positive")
    torch.set_num_threads(args.threads)
    input_hashes = {str(p.resolve()): sha256(p) for p in inputs}
    datasets, manifests, sample_sets, window_sets = {}, {}, {}, {}
    for bucket, path in train_paths.items():
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("split") != "train" or manifest.get("feature_version") != 4 or
                manifest.get("predictor_mode") != "joint_scene" or manifest.get("target_interval") != bucket):
            raise ValueError(f"requires official TRAIN JointScene V4 bucket {bucket}")
        manifests[bucket] = manifest
        datasets[bucket] = calibration.index_records(calibration.read_jsonl(path), bucket)
        sample_sets[bucket], window_sets[bucket] = load_records(path, target_interval=bucket)
    population = calibration.build_population(datasets[1], datasets[2])
    counts = calibration.check_population(population)
    models = calibration.load_frozen_models(model_paths[1], model_paths[2])
    frozen_thresholds = {b: models[b].threshold for b in (1, 2)}
    model_docs = {b: json.loads(path.read_text(encoding="utf-8")) for b, path in model_paths.items()}
    config2 = model_docs[2]["training"]
    if config2["hidden"] != HIDDEN or config2["weight_decay"] != WEIGHT_DECAY or config2["folds"] != FOLDS:
        raise ValueError("current bucket2 configuration differs from fixed experiment control")
    config1 = model_docs[1]["training"]
    if config1["folds"] != FOLDS:
        raise ValueError("bucket1 must use the original five-fold configuration")
    print(f"TRAIN census: {counts}", flush=True)
    print("Bucket1: original pair objective, configuration and frozen threshold", flush=True)
    bucket1_scores, bucket1_audit = calibration.generate_oof(
        sample_sets[1], window_sets[1], config1["hidden"], config1["weight_decay"])
    all_scores, audits = {}, {"bucket1": bucket1_audit}
    for index, (lambda_window, lambda_rank) in enumerate(OBJECTIVE_GRID):
        print(f"Objective {index}: lambda_window={lambda_window}, lambda_rank={lambda_rank}", flush=True)
        all_scores[index], audits[f"objective_{index}"] = generate_oof(
            sample_sets[2], window_sets[2], lambda_window, lambda_rank)
    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(b): input_hashes[str(path.resolve())] for b, path in train_paths.items()}
    gt_groups, gt_source = load_gt_groups(population, datasets, manifests, source_hashes, args, out)
    bucket1_map = score_map(datasets[1], window_sets[1], bucket1_scores)
    calibration.attach_scores(population, {
        1: bucket1_map, 2: score_map(datasets[2], window_sets[2], all_scores[0])}, gt_groups)
    baseline = {"name": "current_v4_saved_thresholds", "bucket1_threshold": frozen_thresholds[1],
                "bucket2_threshold": frozen_thresholds[2],
                "bucket2_oof": bucket2_metrics(all_scores[0], sample_sets[2], window_sets[2], frozen_thresholds[2]),
                "combined_oof": calibration.combined_metrics(population, frozen_thresholds[1], frozen_thresholds[2])}
    if baseline["combined_oof"]["confusion"]["tp"] < TP_FLOOR:
        raise ValueError("current V4 OOF control does not reproduce the TP floor")
    comparisons, sweeps, window_diagnostics = [], {}, {}
    for row in population:
        window_diagnostics[row["key"]] = {
            "scenario": row["key"][0], "window": row["key"][1],
            "fold": fold_of(row["key"][0], FOLDS), "actual": row["actual"],
            "positive_bucket": row["positive_bucket"],
            "bucket1_score": finite_or_none(row["scores"][1]),
            "bucket1_gt_pair_score": finite_or_none(row["pair_scores"][1]),
            "bucket1_alarm": row["scores"][1] >= frozen_thresholds[1], "objectives": {},
        }
    for index, (lambda_window, lambda_rank) in enumerate(OBJECTIVE_GRID):
        calibration.attach_scores(population, {
            1: bucket1_map, 2: score_map(datasets[2], window_sets[2], all_scores[index])}, gt_groups)
        threshold, combined, sweeps[index] = choose_combined_threshold(
            population, frozen_thresholds[1], frozen_thresholds[2])
        entry = {"objective_index": index, "lambda_window": lambda_window, "lambda_rank": lambda_rank,
                 "threshold": threshold, "combined_tp_floor_met": combined["confusion"]["tp"] >= TP_FLOOR,
                 "bucket2_oof": bucket2_metrics(all_scores[index], sample_sets[2], window_sets[2], threshold),
                 "combined_oof": combined}
        comparisons.append(entry)
        print(json.dumps(entry), flush=True)
        for row in population:
            window_diagnostics[row["key"]]["objectives"][str(index)] = {
                "bucket2_score": finite_or_none(row["scores"][2]),
                "bucket2_gt_pair_score": finite_or_none(row["pair_scores"][2]),
                "bucket2_alarm": row["scores"][2] >= threshold,
                "combined_alarm": row["scores"][1] >= frozen_thresholds[1] or row["scores"][2] >= threshold,
                "combined_correct_pair_hit": row["actual"] and (
                    row["pair_scores"][1] >= frozen_thresholds[1] or row["pair_scores"][2] >= threshold),
                "gt_pair_covered": row["gt_pair_covered"],
            }
            if index == 0:
                window_diagnostics[row["key"]]["current_v4"] = {
                    "bucket2_alarm": row["scores"][2] >= frozen_thresholds[2],
                    "combined_alarm": row["scores"][1] >= frozen_thresholds[1] or row["scores"][2] >= frozen_thresholds[2],
                    "combined_correct_pair_hit": row["actual"] and (
                        row["pair_scores"][1] >= frozen_thresholds[1] or row["pair_scores"][2] >= frozen_thresholds[2]),
                }
    selected = select_configuration(comparisons, tp_floor=TP_FLOOR)
    print(f"Selected objective {selected['objective_index']}; fitting TRAIN-only final model", flush=True)
    samples, windows = sample_sets[2], window_sets[2]
    x = np.asarray([s["x"] for s in samples], dtype=np.float64)
    y = np.asarray([s["y"] for s in samples], dtype=np.float64)
    final, means, scales = fit_window(x, y, windows, selected["lambda_window"], selected["lambda_rank"], SEED)
    trained = PairRerankerV4(
        means=means.tolist(), scales=scales.tolist(),
        hidden_weights=final.hidden.weight.detach().numpy().tolist(),
        hidden_bias=final.hidden.bias.detach().numpy().tolist(),
        output_weights=final.output.weight.detach().numpy()[0].tolist(),
        output_bias=float(final.output.bias.detach().numpy()[0]), threshold=selected["threshold"])
    model_doc = trained.to_dict()
    training = {"source": str(train_paths[2].resolve()), "split": "train",
                "n_scenarios": len({s["scenario"] for s in samples}), "n_windows": len(windows),
                "n_candidate_pairs": len(samples), "n_positive_pairs": int(y.sum()),
                "feature_version": 4, "target_interval": 2, "hidden": HIDDEN,
                "weight_decay": WEIGHT_DECAY, "folds": FOLDS, "seed": SEED, "epochs": EPOCHS,
                "lambda_window": selected["lambda_window"], "lambda_rank": selected["lambda_rank"],
                "ranking_margin": RANKING_MARGIN, "combined_tp_floor": TP_FLOOR,
                "experiment": "v4_window"}
    model_doc["training"] = training
    # Verify frozen models and all inputs before publishing this separate result.
    if any(sha256(path) != digest for path, digest in input_hashes.items()):
        raise ValueError("an input changed during training")
    report = {"model": str((out / OUTPUT_NAMES[0]).resolve()), "training": training,
              "train_population_counts": counts, "current_v4_baseline": baseline,
              "cross_validation": comparisons, "selected_oof": selected,
              "bucket1_threshold": frozen_thresholds[1], "input_sha256": input_hashes,
              "gt_identity_source": gt_source,
              "selection": {"split": "train", "combined_tp_floor": TP_FLOOR,
                            "order": ["min combined FP", "max correct-GT-pair recall", "max F1", "min loss weights"],
                            "threshold_rule": "same combined ranking with TP floor; higher threshold breaks remaining threshold ties",
                            "bucket1_oof": "original configuration/objective retrained in scenario-held-out folds",
                            "missing_bucket_policy": "unscorable/missing bucket has no candidates and cannot alarm"},
              "diagnostics": {"summary": str((out / OUTPUT_NAMES[2]).resolve()),
                              "candidate_scores": str((out / OUTPUT_NAMES[3]).resolve()),
                              "windows": str((out / OUTPUT_NAMES[4]).resolve())}}
    (out / OUTPUT_NAMES[0]).write_text(json.dumps(model_doc, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    (out / OUTPUT_NAMES[1]).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(out / OUTPUT_NAMES[3], bucket1=bucket1_scores, current_bucket2=all_scores[0],
                        **{f"objective_{i}": scores for i, scores in all_scores.items()})
    diagnostics = {"train_population_counts": counts, "current_v4": baseline,
                   "v4_window": selected, "objectives": comparisons, "threshold_sweeps": sweeps,
                   "bucket1_threshold": frozen_thresholds[1], "oof_fold_audit": audits,
                   "input_sha256": input_hashes,
                   "candidate_score_layout": "arrays follow load_records input candidate order; objectives use bucket2 input",
                   "candidate_scores": str((out / OUTPUT_NAMES[3]).resolve()),
                   "windows": str((out / OUTPUT_NAMES[4]).resolve())}
    (out / OUTPUT_NAMES[2]).write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (out / OUTPUT_NAMES[4]).open("w", encoding="utf-8") as handle:
        for row in window_diagnostics.values():
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps({"model": report["model"], "selected": selected, "current_v4": baseline}, indent=2))


if __name__ == "__main__":
    main()
