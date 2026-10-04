"""Train dynamic pair re-ranker v2 with scenario-grouped cross-validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from traffic_llm.pair_reranker_v2 import (  # noqa: E402
    FEATURE_NAMES_V2, FEATURE_NAMES_V3, FEATURE_NAMES_V4,
    PairRerankerV2, PairRerankerV3, PairRerankerV4,
)


def load_records(path, target_interval=None):
    samples, windows = [], []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            if target_interval is not None:
                if "positive_intervals" not in doc:
                    raise ValueError(f"{path} lacks interval-specific GT labels")
                if doc.get("target_interval") not in (None, target_interval):
                    raise ValueError(f"{path} was collected for another interval")
            start = len(samples)
            for candidate in doc["candidates"]:
                if (target_interval is not None and
                        candidate["interval_index"] != target_interval):
                    continue
                samples.append({
                    "x": candidate["features"],
                    "y": float(candidate["label"] and (
                        target_interval is None or
                        target_interval in doc["positive_intervals"])),
                    "scenario": doc["scenario"], "window": len(windows),
                    "pair": (candidate["actor_a"], candidate["actor_b"]),
                    "gap": candidate["minimum_clearance_m"],
                    "predicted_contact": candidate.get("predicted_contact",
                                                       candidate["minimum_clearance_m"] <= 1e-9),
                })
            windows.append({
                "start": start, "stop": len(samples),
                "actual": (doc["actual"] if target_interval is None else
                           target_interval in doc["positive_intervals"]),
                "scenario": doc["scenario"],
            })
    return samples, windows


def fold_of(scenario, folds):
    return int.from_bytes(hashlib.sha256(scenario.encode()).digest()[:4], "big") % folds


class MLP(nn.Module):
    def __init__(self, n_features, hidden):
        super().__init__()
        self.hidden = nn.Linear(n_features, hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.output(torch.relu(self.hidden(x))).squeeze(-1)


def fit(x, y, hidden, weight_decay, seed, epochs=350):
    torch.manual_seed(seed)
    means = x.mean(axis=0); scales = x.std(axis=0)
    scales[scales < 1e-6] = 1.0
    tx = torch.tensor((x - means) / scales, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    model = MLP(x.shape[1], hidden)
    positives = max(float(y.sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(y) - positives) / positives)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.01, weight_decay=weight_decay
    )
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(tx), ty)
        loss.backward()
        optimizer.step()
    return model, means, scales


def predict(model, means, scales, x):
    model.eval()
    with torch.no_grad():
        tx = torch.tensor((x - means) / scales, dtype=torch.float32)
        return torch.sigmoid(model(tx)).numpy()


def metrics(actual, predicted):
    a = np.asarray(actual, dtype=bool); p = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(a & p)); fp = int(np.sum(~a & p))
    tn = int(np.sum(~a & ~p)); fn = int(np.sum(a & ~p))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": (tp + tn) / len(a), "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision, "recall": recall, "specificity": specificity, "f1": f1,
    }


def threshold_metrics(scores, samples, windows, threshold):
    actual, predicted = [], []
    for window in windows:
        actual.append(window["actual"])
        predicted.append(bool(np.any(scores[window["start"]:window["stop"]] >= threshold)))
    return metrics(actual, predicted)


def choose_threshold(scores, samples, windows, recall_target=0.70):
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 1001)))
    rows = [(float(t), threshold_metrics(scores, samples, windows, t))
            for t in candidates]
    feasible = [row for row in rows if row[1]["recall"] >= recall_target]
    pool = feasible or rows
    return max(pool, key=lambda row: (
        row[1]["balanced_accuracy"], row[1]["f1"], row[1]["precision"]
    ))


def validation_report(samples, windows, scores, threshold):
    report = threshold_metrics(scores, samples, windows, threshold)
    correct = selected_positive = 0
    for window in windows:
        chosen = [i for i in range(window["start"], window["stop"])
                  if scores[i] >= threshold]
        if window["actual"] and chosen:
            selected_positive += 1
            correct += int(any(samples[i]["y"] > 0.5 for i in chosen))
    raw = []
    for window in windows:
        raw.append(any(samples[i]["gap"] <= 1e-9
                       for i in range(window["start"], window["stop"])))
    positives = sum(bool(w["actual"]) for w in windows)
    report["positive_windows_selected"] = selected_positive
    report["positive_windows_with_correct_actor_pair"] = correct
    report["correct_gt_actor_pair_recall"] = correct / positives if positives else 0.0
    baseline = [any(samples[i]["predicted_contact"]
                    for i in range(w["start"], w["stop"])) for w in windows]
    baseline_correct = sum(bool(w["actual"]) and any(
        samples[i]["predicted_contact"] and samples[i]["y"] > 0.5
        for i in range(w["start"], w["stop"])) for w in windows)
    report["baseline_before_verifier"] = {
        **metrics([w["actual"] for w in windows], baseline),
        "correct_gt_actor_pair_recall": baseline_correct / positives if positives else 0.0,
        "positive_windows_with_correct_actor_pair": baseline_correct,
    }
    report["raw_zero_clearance"] = metrics([w["actual"] for w in windows], raw)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", default="out/pair_reranker_v2/train_pairs_top50.jsonl"
    )
    parser.add_argument(
        "--validation", default="out/pair_reranker_v2/validation_pairs_top50.jsonl"
    )
    parser.add_argument("--out", default="out/pair_reranker_v2_top50")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--recall-target", type=float, default=0.70)
    parser.add_argument("--feature-version", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--target-interval", type=int, choices=range(1, 6), default=None)
    args = parser.parse_args(argv)

    if args.feature_version == 4:
        train_manifest = json.loads(Path(args.train).with_suffix(".manifest.json").read_text())
        val_manifest = json.loads(Path(args.validation).with_suffix(".manifest.json").read_text())
        if train_manifest.get("split") != "train" or val_manifest.get("split") != "val":
            raise SystemExit("V4 requires official-train fitting and held-out val evaluation")
        if (train_manifest.get("feature_version") != 4 or
                val_manifest.get("feature_version") != 4 or
                train_manifest.get("predictor_mode") != "joint_scene" or
                val_manifest.get("predictor_mode") != "joint_scene"):
            raise SystemExit("V4 requires JointScene feature-version 4 collections")
        if (train_manifest.get("target_interval") != args.target_interval or
                val_manifest.get("target_interval") != args.target_interval):
            raise SystemExit("collection and training target intervals must match")
    samples, windows = load_records(args.train, args.target_interval)
    x = np.asarray([s["x"] for s in samples], dtype=np.float64)
    expected_features = {2: FEATURE_NAMES_V2, 3: FEATURE_NAMES_V3,
                         4: FEATURE_NAMES_V4}[args.feature_version]
    if x.ndim != 2 or x.shape[1] != len(expected_features):
        raise SystemExit(
            f"feature-version {args.feature_version} expects {len(expected_features)} "
            f"features, dataset has {x.shape[1] if x.ndim == 2 else 'invalid'}"
        )
    y = np.asarray([s["y"] for s in samples], dtype=np.float64)
    if not len(samples) or not y.any():
        raise SystemExit("training population has no positive pair examples")
    v_samples, v_windows = load_records(args.validation, args.target_interval)
    if {s["scenario"] for s in samples} & {s["scenario"] for s in v_samples}:
        raise SystemExit("train and validation scenarios overlap")
    vx = np.asarray([s["x"] for s in v_samples], dtype=np.float64)
    if vx.ndim != 2 or vx.shape[1] != len(expected_features):
        raise SystemExit("validation feature schema mismatch")
    configs = ((16, 1e-3), (32, 1e-3), (32, 1e-2))
    cv = []
    best = None
    for hidden, decay in configs:
        oof = np.zeros(len(samples), dtype=np.float64)
        for fold in range(args.folds):
            train_ix = np.asarray([
                fold_of(s["scenario"], args.folds) != fold for s in samples
            ])
            val_ix = ~train_ix
            model, means, scales = fit(
                x[train_ix], y[train_ix], hidden, decay, 20260902 + fold
            )
            oof[val_ix] = predict(model, means, scales, x[val_ix])
        threshold, result = choose_threshold(
            oof, samples, windows, args.recall_target
        )
        row = {"hidden": hidden, "weight_decay": decay,
               "threshold": threshold, **result}
        cv.append(row)
        key = (result["balanced_accuracy"], result["f1"], result["precision"])
        if best is None or key > best[0]:
            best = (key, hidden, decay, threshold, result)

    _, hidden, decay, threshold, selected_cv = best
    final, means, scales = fit(x, y, hidden, decay, 20260902)
    validation_scores = predict(final, means, scales, vx)
    held_out = validation_report(
        v_samples, v_windows, validation_scores, threshold
    )
    model_cls = {2: PairRerankerV2, 3: PairRerankerV3,
                 4: PairRerankerV4}[args.feature_version]
    trained = model_cls(
        means=means.tolist(), scales=scales.tolist(),
        hidden_weights=final.hidden.weight.detach().numpy().tolist(),
        hidden_bias=final.hidden.bias.detach().numpy().tolist(),
        output_weights=final.output.weight.detach().numpy()[0].tolist(),
        output_bias=float(final.output.bias.detach().numpy()[0]),
        threshold=threshold,
    )
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model_doc = trained.to_dict()
    model_doc["training"] = {
        "source": str(Path(args.train).resolve()),
        "n_scenarios": len({s["scenario"] for s in samples}),
        "n_windows": len(windows), "n_candidate_pairs": len(samples),
        "n_positive_pairs": int(y.sum()), "folds": args.folds,
        "recall_target": args.recall_target, "hidden": hidden,
        "weight_decay": decay,
        "feature_version": args.feature_version,
        "target_interval": args.target_interval,
    }
    model_path = out / f"pair_reranker_v{args.feature_version}.json"
    model_path.write_text(json.dumps(model_doc, indent=1), encoding="utf-8")
    report = {
        "model": str(model_path.resolve()), "training": model_doc["training"],
        "cross_validation": cv, "selected_oof": selected_cv,
        "held_out_validation": held_out,
        "validation_source": str(Path(args.validation).resolve()),
    }
    (out / f"training_report_v{args.feature_version}.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
