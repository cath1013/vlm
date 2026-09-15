"""Train a first actor-pair re-ranker and evaluate it on a compact-v3 cohort.

Training uses the cached DeepAccident train split only. Hyperparameters and the
decision threshold are selected from scenario-grouped out-of-fold predictions.
The validation batch is loaded only after the final model has been fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from traffic_llm.pair_reranker import (  # noqa: E402
    FEATURE_NAMES, PairReranker, pair_features,
)


def _groups(gt):
    positive = next(
        (row for row in gt.get("expected", []) if row.get("accident_expected")),
        None,
    )
    if not positive:
        return []
    return [set(v.get("actor_ids") or [])
            for v in positive.get("involved_vehicles") or []]


def _matches(pair, groups):
    a, b = pair
    return any(
        (a in groups[i] and b in groups[j]) or
        (b in groups[i] and a in groups[j])
        for i in range(len(groups)) for j in range(i + 1, len(groups))
    )


def load_train(path: Path):
    samples, windows = [], []
    with path.open(encoding="utf-8") as handle:
        for window_id, line in enumerate(handle):
            doc = json.loads(line)
            gt = doc["gt"]
            scenario = gt["scenario"]["id"]
            groups = _groups(gt)
            actual = bool(groups)
            # The cache stores the closest pair separately in every 1-second
            # bucket. Collapse repeated pairs to their minimum gap.
            by_pair = {}
            for k, (gap, pair) in enumerate(doc["gaps"], 1):
                if gap is None or pair is None:
                    continue
                key = tuple(sorted(pair))
                candidate = (float(gap), k)
                if key not in by_pair or candidate < by_pair[key]:
                    by_pair[key] = candidate
            start = len(samples)
            for pair, (gap, k) in by_pair.items():
                samples.append({
                    "x": pair_features(pair[0], pair[1], gap, k),
                    "y": float(_matches(pair, groups)),
                    "scenario": scenario,
                    "window_id": window_id,
                    "pair": pair,
                })
            windows.append({
                "start": start, "stop": len(samples), "actual": actual,
                "scenario": scenario,
            })
    return samples, windows


def _fold(scenario: str, folds: int) -> int:
    digest = hashlib.sha256(scenario.encode()).digest()
    return int.from_bytes(digest[:4], "big") % folds


def fit_logistic(x, y, l2=1e-3, steps=1800, lr=0.03):
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-6] = 1.0
    z = (x - means) / scales
    weights = np.zeros(z.shape[1], dtype=np.float64)
    bias = 0.0
    mw = np.zeros_like(weights); vw = np.zeros_like(weights)
    mb = vb = 0.0
    positives = max(float(y.sum()), 1.0)
    pos_weight = (len(y) - positives) / positives
    row_weight = np.where(y > 0.5, pos_weight, 1.0)
    row_weight /= row_weight.mean()
    for step in range(1, steps + 1):
        logits = np.clip(z @ weights + bias, -40.0, 40.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        err = row_weight * (probs - y)
        gw = z.T @ err / len(y) + l2 * weights
        gb = float(err.mean())
        mw = 0.9 * mw + 0.1 * gw
        vw = 0.999 * vw + 0.001 * gw * gw
        mb = 0.9 * mb + 0.1 * gb
        vb = 0.999 * vb + 0.001 * gb * gb
        weights -= lr * (mw / (1 - 0.9 ** step)) / (
            np.sqrt(vw / (1 - 0.999 ** step)) + 1e-8
        )
        bias -= lr * (mb / (1 - 0.9 ** step)) / (
            math.sqrt(vb / (1 - 0.999 ** step)) + 1e-8
        )
    return weights, bias, means, scales


def probabilities(x, params):
    weights, bias, means, scales = params
    logits = np.clip(((x - means) / scales) @ weights + bias, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _metrics(actual, predicted):
    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(actual & predicted)); fp = int(np.sum(~actual & predicted))
    tn = int(np.sum(~actual & ~predicted)); fn = int(np.sum(actual & ~predicted))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": (tp + tn) / len(actual),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision, "recall": recall, "f1": f1,
    }


def choose_threshold(scores, samples, windows):
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 501)))
    best = None
    for threshold in candidates:
        actual, predicted = [], []
        for window in windows:
            actual.append(window["actual"])
            predicted.append(bool(np.any(
                scores[window["start"]:window["stop"]] >= threshold
            )))
        metrics = _metrics(actual, predicted)
        key = (metrics["balanced_accuracy"], metrics["f1"], metrics["precision"])
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    return best[1], best[2]


def _compact_doc(payload):
    if "contents" in payload:
        return json.loads(payload["contents"][0]["parts"][0]["text"])
    user = next(m for m in payload["messages"] if m.get("role") == "user")
    content = user["content"]
    text = content if isinstance(content, str) else next(
        x["text"] for x in content if x.get("type") == "text"
    )
    return json.loads(text)


def evaluate_validation(base: Path, model: PairReranker):
    cohort = json.loads((base / "batch_manifest.json").read_text(encoding="utf-8"))
    actual, predicted, raw = [], [], []
    pair_correct = 0
    positive_selected = 0
    for scenario in cohort["scenarios"]:
        directory = base / scenario["directory"]
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for window in manifest["windows"]:
            payload = json.loads((directory / window["payload"]).read_text(encoding="utf-8"))
            gt = json.loads((directory / window["ground_truth"]).read_text(encoding="utf-8"))
            doc = _compact_doc(payload)
            ix = {name: i for i, name in enumerate(doc["predicted_pair_columns"])}
            scored = []
            for pair in doc["predicted_closest_pairs"]:
                score = model.predict_proba(
                    pair[0], pair[1], pair[ix["minimum_clearance_m"]],
                    pair[ix["interval_index"]],
                )
                scored.append((score, pair))
            selected = [p for score, p in scored if score >= model.threshold]
            groups = _groups(gt)
            is_actual = bool(groups)
            actual.append(is_actual)
            predicted.append(bool(selected))
            raw.append(any(float(p[ix["minimum_clearance_m"]]) <= 0.0
                           for _, p in scored))
            if is_actual and selected:
                positive_selected += 1
                pair_correct += int(any(_matches((p[0], p[1]), groups) for p in selected))
    report = _metrics(actual, predicted)
    report["positive_windows_selected"] = positive_selected
    report["positive_windows_with_correct_actor_pair"] = pair_correct
    report["raw_zero_clearance"] = _metrics(actual, raw)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-records", default="out/baselines/records_train_pred.jsonl")
    parser.add_argument("--validation-batch", default="out/experiments/waypointnet_geometry_v3_audit20")
    parser.add_argument("--out", default="out/pair_reranker")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)

    samples, windows = load_train(Path(args.train_records))
    x = np.asarray([s["x"] for s in samples], dtype=np.float64)
    y = np.asarray([s["y"] for s in samples], dtype=np.float64)
    oof = np.zeros(len(samples), dtype=np.float64)
    l2_grid = (1e-4, 1e-3, 1e-2, 1e-1)
    cv_rows = []
    best_l2 = None; best_score = -1.0
    for l2 in l2_grid:
        fold_scores = np.zeros(len(samples), dtype=np.float64)
        for fold in range(args.folds):
            train_ix = np.array([_fold(s["scenario"], args.folds) != fold for s in samples])
            val_ix = ~train_ix
            params = fit_logistic(x[train_ix], y[train_ix], l2=l2)
            fold_scores[val_ix] = probabilities(x[val_ix], params)
        threshold, metrics = choose_threshold(fold_scores, samples, windows)
        cv_rows.append({"l2": l2, "threshold": threshold, **metrics})
        if metrics["balanced_accuracy"] > best_score:
            best_l2, best_score, oof = l2, metrics["balanced_accuracy"], fold_scores

    threshold, oof_metrics = choose_threshold(oof, samples, windows)
    params = fit_logistic(x, y, l2=best_l2)
    weights, bias, means, scales = params
    model = PairReranker(
        weights=weights.tolist(), bias=float(bias), means=means.tolist(),
        scales=scales.tolist(), threshold=threshold,
    )
    validation = evaluate_validation(Path(args.validation_batch), model)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model_doc = model.to_dict()
    model_doc["training"] = {
        "source": str(Path(args.train_records).resolve()),
        "n_scenarios": len({s["scenario"] for s in samples}),
        "n_windows": len(windows), "n_candidate_pairs": len(samples),
        "n_positive_pairs": int(y.sum()), "folds": args.folds,
        "selected_l2": best_l2,
    }
    (out / "pair_reranker.json").write_text(
        json.dumps(model_doc, indent=1), encoding="utf-8"
    )
    report = {
        "model": str((out / "pair_reranker.json").resolve()),
        "train": model_doc["training"],
        "cross_validation": cv_rows,
        "selected_oof": oof_metrics,
        "held_out_validation": validation,
        "validation_source": str(Path(args.validation_batch).resolve()),
    }
    (out / "training_report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
