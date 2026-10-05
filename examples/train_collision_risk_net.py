"""Five-fold scenario CV, final train fit, and one held-out val evaluation."""
from __future__ import annotations

import argparse
from functools import partial
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader

from traffic_llm.collision_risk_net import (CollisionRiskNet, cumulative_risk,
                                            hazard_to_event_probabilities, load_v2)
from traffic_llm.predict_model import (N_CANDIDATE_FEATURES, N_GLOBAL_FEATURES,
                                       N_HISTORY_FEATURES, N_HISTORY_STEPS,
                                       N_INTERACTION_FEATURES)


def read_rows(path, split):
    with Path(path).open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    if not rows or any(row.get("dataset_split") != split for row in rows):
        raise ValueError(f"{path} must contain nonempty official {split} data only")
    ids = [r["window_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate window identifiers")
    for row in rows:
        n = len(row["actors"])
        if len(row["pair_labels"]) != n * (n - 1) // 2:
            raise ValueError(f"Pair label count mismatch: {row['window_id']}")
        bucket = row.get("gt_bucket")
        if type(bucket) is not int or not 0 <= bucket <= 5:
            raise ValueError(f"Missing or invalid gt_bucket: {row['window_id']}")
        if bool(bucket) != row["gt_positive"] or any(k > 0 for k in row["pair_labels"]) != row["gt_pair_present"]:
            raise ValueError(f"Inconsistent window labels: {row['window_id']}")
        if any(k not in (0, bucket) for k in row["pair_labels"]):
            raise ValueError(f"Pair bucket disagrees with window: {row['window_id']}")
        if len(row.get("hazard_target", [])) != len(row["pair_labels"]) or len(row.get("hazard_mask", [])) != len(row["pair_labels"]):
            raise ValueError(f"Missing pair hazards: {row['window_id']}")
        if len(row.get("observed_negative_buckets", [])) != 5:
            raise ValueError(f"Missing observed intervals: {row['window_id']}")
        for label, target, mask in zip(row["pair_labels"], row["hazard_target"], row["hazard_mask"]):
            if len(target) != 5 or len(mask) != 5 or any(x not in (0, 1) for x in target):
                raise ValueError(f"Invalid hazards: {row['window_id']}")
            expected_mask = ([k < bucket for k in range(5)] if label else
                             [observed and (not bucket or k < bucket)
                              for k, observed in enumerate(row["observed_negative_buckets"])])
            expected_target = [int(label == k + 1) for k in range(5)]
            if mask != expected_mask or target != expected_target:
                raise ValueError(f"Inconsistent hazard supervision: {row['window_id']}")
    return rows


def verify_manifest(path, split, *, smoke):
    manifest_path = Path(path).with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {"source_split": split, "observation_mode": "sensor3d",
                "window_s": 5, "stride_s": 1, "horizon_s": 5}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{manifest_path}: expected {key}={value!r}")
    if not smoke:
        if manifest.get("limit") != 0 or manifest["counts"]["failed"]:
            raise ValueError(f"{manifest_path}: full run requires no limit or failed scenarios")
        if split == "train" and (manifest.get("available_scenarios") != 483
                                 or manifest.get("selected_scenarios") != 483):
            raise ValueError(f"{manifest_path}: expected the complete official train483 set")
        if split == "val" and (manifest.get("available_scenarios") != 104
                               or manifest.get("selected_scenarios") != 104):
            raise ValueError(f"{manifest_path}: expected the complete official val104 set")
    return manifest


def eligible_rows(rows, max_horizon=4):
    """Keep only windows with supervised cells in the model's prefix."""
    if max_horizon not in (2, 3, 4):
        raise ValueError("max_horizon must be 2, 3, or 4")
    return [r for r in rows if any(any(m[:max_horizon]) for m in r["hazard_mask"])]


def collate(rows, max_horizon=4):
    B = len(rows)
    A = max(1, max(len(r["actors"]) for r in rows))
    C = max(1, max((len(a["candidates"]) for r in rows for a in r["actors"]), default=0))
    I = max(1, max((len(a["interactions"]) for r in rows for a in r["actors"]), default=0))
    out = {"global_features": torch.zeros(B, A, N_GLOBAL_FEATURES),
           "candidates": torch.zeros(B, A, C, N_CANDIDATE_FEATURES),
           "candidate_mask": torch.zeros(B, A, C, dtype=torch.bool),
           "history": torch.zeros(B, A, N_HISTORY_STEPS, N_HISTORY_FEATURES),
           "interactions": torch.zeros(B, A, I, N_INTERACTION_FEATURES),
           "interaction_mask": torch.zeros(B, A, I, dtype=torch.bool),
           "actor_mask": torch.zeros(B, A, dtype=torch.bool),
           "origin": torch.zeros(B, A, 2),
           "hazard_target": torch.zeros(B, A * (A - 1) // 2, max_horizon),
           "hazard_mask": torch.zeros(B, A * (A - 1) // 2, max_horizon, dtype=torch.bool),
           "rows": rows}
    for b, row in enumerate(rows):
        for a, actor in enumerate(row["actors"]):
            out["actor_mask"][b, a] = True
            out["global_features"][b, a] = torch.as_tensor(actor["global"])
            out["history"][b, a] = torch.as_tensor(actor["history"])
            out["origin"][b, a] = torch.as_tensor(actor["origin_enu"])
            nc, ni = len(actor["candidates"]), len(actor["interactions"])
            if nc:
                out["candidates"][b, a, :nc] = torch.as_tensor(actor["candidates"])
                out["candidate_mask"][b, a, :nc] = True
            if ni:
                out["interactions"][b, a, :ni] = torch.as_tensor(actor["interactions"])
                out["interaction_mask"][b, a, :ni] = True
        local_i, local_j = torch.triu_indices(len(row["actors"]), len(row["actors"]), 1)
        for p, (i, j) in enumerate(zip(local_i.tolist(), local_j.tolist())):
            slot = i * (2 * A - i - 1) // 2 + (j - i - 1)
            out["hazard_target"][b, slot] = torch.as_tensor(row["hazard_target"][p][:max_horizon])
            out["hazard_mask"][b, slot] = torch.as_tensor(row["hazard_mask"][p][:max_horizon])
    return out


INPUT_KEYS = ("global_features", "candidates", "candidate_mask", "history",
              "interactions", "interaction_mask", "actor_mask", "origin")


def forward(model, batch, device):
    return model(*(batch[k].to(device) for k in INPUT_KEYS))


def folds_by_scenario(rows, seed):
    scenarios = sorted({r["scenario_id"] for r in rows})
    if len(scenarios) < 5:
        raise ValueError("Five-fold CV requires at least five train scenarios")
    random.Random(seed).shuffle(scenarios)
    assignments = {sid: i % 5 for i, sid in enumerate(scenarios)}
    return assignments


def supervision_counts(rows, max_horizon=4):
    rows = eligible_rows(rows, max_horizon)
    positive = [0] * max_horizon
    negative = [0] * max_horizon
    censored = [0] * max_horizon
    for row in rows:
        for targets, masks in zip(row["hazard_target"], row["hazard_mask"]):
            for k in range(max_horizon):
                if not masks[k]:
                    censored[k] += 1
                elif targets[k]:
                    positive[k] += 1
                else:
                    negative[k] += 1
    return {"windows_used": len(rows), "supervised_hazard_entries_per_bucket":
            [p + n for p, n in zip(positive, negative)],
            "positive_hazard_entries_per_bucket": positive,
            "negative_hazard_entries_per_bucket": negative,
            "censored_hazard_entries_per_bucket": censored}


def positive_weight(rows, max_horizon=4):
    counts = supervision_counts(rows, max_horizon)
    positives = sum(counts["positive_hazard_entries_per_bucket"])
    negatives = sum(counts["negative_hazard_entries_per_bucket"])
    return negatives / positives if positives else 1.0


def fit(rows, checkpoint, device, *, epochs, batch_size, lr, hidden_dim,
        dropout, seed, max_horizon=4):
    rows = eligible_rows(rows, max_horizon)
    if not rows:
        raise ValueError("No supervised windows within max_horizon")
    torch.manual_seed(seed)
    model = CollisionRiskNet(load_v2(checkpoint, device), hidden_dim, dropout, max_horizon=max_horizon).to(device)
    pos_weight = torch.tensor(positive_weight(rows, max_horizon), device=device)
    loader = DataLoader(rows, batch_size=batch_size, shuffle=True, collate_fn=partial(collate, max_horizon=max_horizon),
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        loss_sum = n = 0
        for batch in loader:
            logits, pair_mask = forward(model, batch, device)
            mask = batch["hazard_mask"].to(device) & pair_mask.unsqueeze(-1)
            if not mask.any():
                continue
            targets = batch["hazard_target"].to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[mask], targets[mask], pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().item() * int(mask.sum())
            n += int(mask.sum())
        print(f"epoch {epoch + 1}/{epochs}: masked BCE={loss_sum / max(1, n):.5f}", flush=True)
    return model


@torch.no_grad()
def predict(model, rows, device, batch_size):
    model.eval()
    rows = eligible_rows(rows, model.max_horizon)
    records = []
    for batch in DataLoader(rows, batch_size=batch_size,
                            collate_fn=partial(collate, max_horizon=model.max_horizon)):
        logits, mask = forward(model, batch, device)
        probs = logits.sigmoid().cpu()
        for b, row in enumerate(batch["rows"]):
            valid = mask[b].cpu()
            records.append(aggregate_window(row, probs[b, valid].tolist(), model.max_horizon))
    return records


def aggregate_window(row, pair_probabilities, max_horizon=4):
    """Select the maximum cumulative-risk pair independently at each horizon."""
    if len(pair_probabilities) != len(row["pair_labels"]):
        raise ValueError(f"Pair prediction count mismatch: {row['window_id']}")
    actor_ids = [a["actor_id"] for a in row["actors"]]
    pairs = [(actor_ids[i], actor_ids[j])
             for i in range(len(actor_ids)) for j in range(i + 1, len(actor_ids))]
    if pair_probabilities:
        hazards = torch.as_tensor(pair_probabilities, dtype=torch.float64)
        if hazards.shape[-1] != max_horizon:
            raise ValueError("Predictions must contain exactly max_horizon hazards")
        risks = cumulative_risk(hazards).tolist()
        events, survival = hazard_to_event_probabilities(hazards)
        best = [max(range(len(pairs)), key=lambda p: risks[p][h]) for h in range(max_horizon)]
        scores = [risks[best[h]][h] for h in range(max_horizon)]
    else:
        best, scores, events, survival = [None] * max_horizon, [0.0] * max_horizon, None, None
    return {"window_id": row["window_id"], "scenario_id": row["scenario_id"],
            "max_horizon": max_horizon, "gt_bucket": row["gt_bucket"], "horizon_risks": scores,
            "selected_pairs": [pairs[p] if p is not None else None for p in best],
            "selected_pair_gt_labels": [row["pair_labels"][p] if p is not None else None for p in best],
            "selected_pair_events": [events[p].tolist() if p is not None else None for p in best],
            "selected_pair_survival": [float(survival[p]) if p is not None else None for p in best],
            "pair_event_probabilities": events.tolist() if events is not None else [],
            "pair_survival": survival.tolist() if survival is not None else []}


def decide_window(record, threshold, horizon):
    return record["horizon_risks"][horizon - 1] >= threshold


def binary_metrics(truth, pred):
    tp = sum(a and b for a, b in zip(truth, pred))
    tn = sum(not a and not b for a, b in zip(truth, pred))
    fp = sum(not a and b for a, b in zip(truth, pred))
    fn = sum(a and not b for a, b in zip(truth, pred))
    div = lambda a, b: a / b if b else None
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "accuracy": div(tp + tn, tp + tn + fp + fn),
            "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
            "specificity": div(tn, tn + fp), "f1": div(2 * tp, 2 * tp + fp + fn),
            "balanced_accuracy": ((div(tp, tp + fn) + div(tn, tn + fp)) / 2
                                  if tp + fn and tn + fp else None)}


def identifiable(row, horizon):
    return (0 < row["gt_bucket"] <= horizon or
            all(row["observed_negative_buckets"][:horizon]))


def threshold_candidates(scores):
    """Unique OOF risks plus all-positive and all-negative boundaries (>=)."""
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("OOF cumulative risks must be finite probabilities")
    return sorted({0.0, math.nextafter(1.0, math.inf), *scores})


def select_threshold(records, rows, max_horizon=4):
    """Select independent thresholds using identifiable TRAIN-OOF windows only.

    Sweep unique scores in descending order, grouping ties, in O(N log N)
    per horizon. Undefined metrics rank below defined metrics; this also gives
    deterministic thresholds for horizons lacking one or both classes.
    """
    rows = eligible_rows(rows, max_horizon)
    by_id = {r["window_id"]: r for r in rows}
    records = [r for r in records if r["window_id"] in by_id]
    thresholds = {}
    for h in range(1, max_horizon + 1):
        valid = [r for r in records if identifiable(by_id[r["window_id"]], h)]
        scored = sorted(((r["horizon_risks"][h - 1], 0 < r["gt_bucket"] <= h)
                         for r in valid), reverse=True)
        candidates = threshold_candidates([score for score, _ in scored])
        positives = sum(truth for _, truth in scored)
        negatives = len(scored) - positives
        tp = fp = index = 0
        best_key = None
        for threshold in reversed(candidates):
            while index < len(scored) and scored[index][0] >= threshold:
                tp += int(scored[index][1])
                fp += int(not scored[index][1])
                index += 1
            fn, tn = positives - tp, negatives - fp
            balanced = ((tp / positives + tn / negatives) / 2
                        if positives and negatives else -1.0)
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else -1.0
            precision = tp / (tp + fp) if tp + fp else -1.0
            key = (balanced, f1, precision, threshold)
            if best_key is None or key > best_key:
                best_key = key
                thresholds[str(h)] = threshold
    return thresholds


def report(records, rows, threshold_by_horizon, max_horizon=4):
    if set(threshold_by_horizon) != {str(h) for h in range(1, max_horizon + 1)}:
        raise ValueError("One frozen threshold per supported horizon is required")
    rows = eligible_rows(rows, max_horizon)
    eligible_ids = {r["window_id"] for r in rows}
    records = [r for r in records if r["window_id"] in eligible_ids]
    if any(r["max_horizon"] != max_horizon for r in records):
        raise ValueError("Prediction horizon differs from report horizon")
    if len(records) != len(rows) or {r["window_id"] for r in records} != {r["window_id"] for r in rows}:
        raise ValueError("One prediction per evaluation window is required")
    by_id = {r["window_id"]: r for r in rows}
    positive_windows = [r for r in rows if 0 < r["gt_bucket"] <= max_horizon]
    present = sum(r["gt_pair_present"] for r in positive_windows)
    safe_div = lambda a, b: a / b if b else None
    horizons = {}
    for h in range(1, max_horizon + 1):
        valid = [r for r in records if identifiable(by_id[r["window_id"]], h)]
        truth = [0 < r["gt_bucket"] <= h for r in valid]
        pred = [decide_window(r, threshold_by_horizon[str(h)], h) for r in valid]
        positive_count = sum(truth)
        negative_count = len(valid) - positive_count
        metrics = binary_metrics(truth, pred)
        if negative_count == 0:
            metrics["precision"] = None
            metrics["f1"] = None
        hits = sum(t and p and r["selected_pair_gt_labels"][h - 1] == r["gt_bucket"]
                   for r, t, p in zip(valid, truth, pred))
        covered = sum(by_id[r["window_id"]]["gt_pair_present"]
                      for r, t in zip(valid, truth) if t)
        metrics.update({"threshold": threshold_by_horizon[str(h)],
                        "identifiable_windows": len(valid),
                        "candidate_coverage": safe_div(covered, positive_count),
                        "gt_pair_present_windows": covered,
                        "conditional_correct_pair_recall": safe_div(hits, covered),
                        "valid_windows": len(valid), "positives": positive_count,
                        "negatives": negative_count, "correct_gt_actor_pair_hits": hits,
                        "correct_gt_actor_pair_recall": safe_div(hits, positive_count)})
        horizons[str(h)] = metrics
    timing = [[0] * max_horizon for _ in range(max_horizon)]
    for r in records:
        if 0 < r["gt_bucket"] <= max_horizon:
            events = r["selected_pair_events"][max_horizon - 1]
            predicted = 1 + max(range(max_horizon), key=lambda k: events[k]) if events else None
            if predicted is not None:
                timing[r["gt_bucket"] - 1][predicted - 1] += 1
    exact = sum(timing[k][k] for k in range(max_horizon))
    near = sum(timing[i][j] for i in range(max_horizon)
               for j in range(max_horizon) if abs(i - j) <= 1)
    timing_available = sum(sum(line) for line in timing)
    return {"max_horizon": max_horizon, "model": f"CRN-{max_horizon}",
            "threshold_by_horizon": threshold_by_horizon, "unit": "window", "window_count": len(records),
            "pair_count": sum(len(row["pair_labels"]) for row in rows),
            "supervision": supervision_counts(rows, max_horizon), "per_horizon": horizons,
            "cumulative_collision_risk_by_window": {
                r["window_id"]: {str(h): r["horizon_risks"][h - 1]
                                 for h in range(1, max_horizon + 1)} for r in records},
            "timing_confusion": timing,
            "timing_available_positive_windows": timing_available,
            "timing_unavailable_positive_windows": len(positive_windows) - timing_available,
            "timing_exact_bucket_accuracy": safe_div(exact, len(positive_windows)),
            "timing_within_one_bucket_accuracy": safe_div(near, len(positive_windows)),
            "survival_selected_pair": [r["selected_pair_survival"][max_horizon - 1] for r in records],
            "gt_positive_windows": len(positive_windows),
            "gt_pair_present_windows": present,
            "visible_pair_coverage_ceiling": safe_div(present, len(positive_windows))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", help="official train JSONL for CV and final fitting")
    ap.add_argument("--val", help="official val104 JSONL, evaluation mode only")
    ap.add_argument("--evaluate-checkpoint", help="frozen collision_risk_final.pt for val evaluation")
    ap.add_argument("--v2-checkpoint", help="frozen V2 encoder for fitting")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-horizon", type=int, choices=(2, 3, 4), default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-grid", default="0.001,0.0003",
                    help="learning-rate candidates selected by train CV")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="allow a limited val subset and label the report as a smoke run")
    args = ap.parse_args(argv)
    if args.evaluate_checkpoint:
        if not args.val or args.train:
            ap.error("evaluation requires --val and no --train")
        saved = torch.load(args.evaluate_checkpoint, map_location=args.device, weights_only=False)
        if saved["max_horizon"] != args.max_horizon:
            ap.error("--max-horizon must match checkpoint metadata")
        model = CollisionRiskNet(load_v2(saved["v2_checkpoint"], args.device),
                                 saved["hidden_dim"], saved["dropout"],
                                 max_horizon=saved["max_horizon"]).to(args.device)
        model.load_state_dict(saved["state_dict"])
        verify_manifest(args.val, "val", smoke=args.smoke)
        val = read_rows(args.val, "val")
        overlap = set(saved["train_scenario_ids"]) & {r["scenario_id"] for r in val}
        if overlap:
            raise ValueError(f"Train/val scenario overlap: {sorted(overlap)[:5]}")
        metrics = report(predict(model, val, args.device, args.batch_size), val,
                         saved["threshold_by_horizon"], args.max_horizon)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "val104_report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return
    if not args.train or not args.v2_checkpoint or args.val:
        ap.error("fitting requires --train and --v2-checkpoint, without --val")
    if args.epochs < 1 or args.batch_size < 1:
        ap.error("epochs and batch size must be positive")
    learning_rates = [float(x) for x in args.lr_grid.split(",") if x.strip()]
    if not learning_rates or any(x <= 0 for x in learning_rates):
        ap.error("--lr-grid must contain positive learning rates")
    verify_manifest(args.train, "train", smoke=args.smoke)
    train = read_rows(args.train, "train")
    # Assign from the complete source population before horizon-specific filtering.
    assignments = folds_by_scenario(train, args.seed)
    train = eligible_rows(train, args.max_horizon)
    cv_candidates = []
    for lr in learning_rates:
        oof = []
        for fold in range(5):
            fitting = [r for r in train if assignments[r["scenario_id"]] != fold]
            held = [r for r in train if assignments[r["scenario_id"]] == fold]
            print(f"lr={lr} fold {fold + 1}/5: {len(fitting)} fit, {len(held)} held", flush=True)
            model = fit(fitting, args.v2_checkpoint, args.device, epochs=args.epochs,
                        batch_size=args.batch_size, lr=lr, hidden_dim=args.hidden_dim,
                        dropout=args.dropout, seed=args.seed + fold, max_horizon=args.max_horizon)
            oof.extend(predict(model, held, args.device, args.batch_size))
            del model
        threshold_by_horizon = select_threshold(oof, train, args.max_horizon)
        cv_candidates.append({"lr": lr, "threshold_by_horizon": threshold_by_horizon,
                              "report": report(oof, train, threshold_by_horizon, args.max_horizon)})
    chosen = max(cv_candidates,
                 key=lambda c: (sum(
                     m["balanced_accuracy"] for m in c["report"]["per_horizon"].values()
                     if m["balanced_accuracy"] is not None) /
                     max(1, sum(m["balanced_accuracy"] is not None
                                for m in c["report"]["per_horizon"].values())), -c["lr"]))
    threshold_by_horizon = chosen["threshold_by_horizon"]
    cv_report = chosen["report"]
    model = fit(train, args.v2_checkpoint, args.device, epochs=args.epochs,
                batch_size=args.batch_size, lr=chosen["lr"], hidden_dim=args.hidden_dim,
                dropout=args.dropout, seed=args.seed, max_horizon=args.max_horizon)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "v2_checkpoint": str(Path(args.v2_checkpoint).resolve()),
                "hidden_dim": args.hidden_dim, "dropout": args.dropout,
                "max_horizon": model.max_horizon, "model_spec": model.spec,
                "threshold_by_horizon": threshold_by_horizon, "lr": chosen["lr"],
                "train_scenario_ids": sorted(assignments)}, out / "collision_risk_final.pt")
    result = {"config": vars(args), "smoke_run": args.smoke,
              "fold_assignments": assignments, "cv_candidates": cv_candidates,
              "selected_lr": chosen["lr"],
              "threshold_by_horizon": threshold_by_horizon,
              "decision_rule": "maximum pair cumulative hazard risk at each horizon >= horizon-specific train-OOF threshold",
              "cv_train": cv_report}
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"cv_train": cv_report}, indent=2))


if __name__ == "__main__":
    main()
