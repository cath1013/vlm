"""Five-fold scenario CV, final train fit, and one held-out val evaluation."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader

from traffic_llm.collision_risk_net import CollisionRiskNet, load_v2
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


def collate(rows):
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
           "labels": torch.zeros(B, A * (A - 1) // 2, dtype=torch.long),
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
        for i, j, label in zip(local_i.tolist(), local_j.tolist(), row["pair_labels"]):
            slot = i * (2 * A - i - 1) // 2 + (j - i - 1)
            out["labels"][b, slot] = label
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


def class_weights(rows):
    counts = torch.bincount(torch.tensor([k for r in rows for k in r["pair_labels"]],
                                         dtype=torch.long), minlength=6).float()
    # Absent classes have zero weight. Mean normalization leaves the learning
    # rate interpretable while inverse frequency balances observed classes.
    weights = torch.where(counts > 0, 1 / counts.clamp(min=1), 0)
    return weights / weights[weights > 0].mean()


def fit(rows, checkpoint, device, *, epochs, batch_size, lr, hidden_dim,
        dropout, seed):
    torch.manual_seed(seed)
    model = CollisionRiskNet(load_v2(checkpoint, device), hidden_dim, dropout).to(device)
    weights = class_weights(rows).to(device)
    loader = DataLoader(rows, batch_size=batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        loss_sum = n = 0
        for batch in loader:
            logits, mask = forward(model, batch, device)
            if not mask.any():
                continue
            labels = batch["labels"].to(device)
            loss = nn.functional.cross_entropy(logits[mask], labels[mask], weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss) * int(mask.sum())
            n += int(mask.sum())
        print(f"epoch {epoch + 1}/{epochs}: weighted CE={loss_sum / max(1, n):.5f}", flush=True)
    return model


@torch.no_grad()
def predict(model, rows, device, batch_size):
    model.eval()
    records = []
    for batch in DataLoader(rows, batch_size=batch_size, collate_fn=collate):
        logits, mask = forward(model, batch, device)
        probs = logits.softmax(-1).cpu()
        for b, row in enumerate(batch["rows"]):
            valid = mask[b].cpu()
            records.append(aggregate_window(row, probs[b, valid].tolist()))
    return records


def aggregate_window(row, pair_probabilities):
    """Keep only the highest collision-score pair for this observed window."""
    if len(pair_probabilities) != len(row["pair_labels"]):
        raise ValueError(f"Pair prediction count mismatch: {row['window_id']}")
    if not pair_probabilities:
        return {"window_id": row["window_id"], "scenario_id": row["scenario_id"],
                "gt_bucket": row["gt_bucket"], "score": 0.0,
                "selected_pair": None, "selected_pair_gt_label": None,
                "selected_probabilities": None}
    scores = [sum(p[1:]) for p in pair_probabilities]
    best = max(range(len(scores)), key=scores.__getitem__)
    actor_ids = [a["actor_id"] for a in row["actors"]]
    # Pair labels and valid-pair model logits both follow torch.triu_indices.
    pairs = [(actor_ids[i], actor_ids[j])
             for i in range(len(actor_ids)) for j in range(i + 1, len(actor_ids))]
    return {"window_id": row["window_id"], "scenario_id": row["scenario_id"],
            "gt_bucket": row["gt_bucket"], "score": scores[best],
            "selected_pair": pairs[best],
            "selected_pair_gt_label": row["pair_labels"][best],
            "selected_probabilities": pair_probabilities[best]}


def decide_window(record, threshold):
    if record["score"] < threshold or record["selected_probabilities"] is None:
        return 0
    probabilities = record["selected_probabilities"]
    return 1 + max(range(5), key=lambda k: probabilities[k + 1])


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
            "specificity": div(tn, tn + fp), "f1": div(2 * tp, 2 * tp + fp + fn)}


def select_threshold(records):
    # Each OOF window contributes once, regardless of its number of pairs.
    candidates = [i / 100 for i in range(5, 100, 5)]
    def key(t):
        pred = [decide_window(r, t) > 0 for r in records]
        f1 = binary_metrics([r["gt_bucket"] > 0 for r in records], pred)["f1"]
        return ((f1 if f1 is not None else -1), -t)
    return max(candidates, key=key)


def report(records, rows, threshold):
    if len(records) != len(rows) or {r["window_id"] for r in records} != {r["window_id"] for r in rows}:
        raise ValueError("One prediction per evaluation window is required")
    truth = [r["gt_bucket"] for r in records]
    pred = [decide_window(r, threshold) for r in records]
    matrix = [[0] * 6 for _ in range(6)]
    for a, b in zip(truth, pred):
        matrix[a][b] += 1
    positive_windows = [r for r in rows if r["gt_bucket"] > 0]
    present = sum(r["gt_pair_present"] for r in positive_windows)
    pair_hits = sum(p > 0 and r["gt_bucket"] > 0 and r["selected_pair_gt_label"] > 0
                    for r, p in zip(records, pred) if r["selected_pair_gt_label"] is not None)
    safe_div = lambda a, b: a / b if b else None
    return {"threshold": threshold, "unit": "window", "window_count": len(records),
            "pair_count": sum(len(row["pair_labels"]) for row in rows),
            "overall_0_5_s": binary_metrics([x > 0 for x in truth], [x > 0 for x in pred]),
            "per_horizon": {f"{k-1}-{k}s": binary_metrics([x == k for x in truth],
                                                             [x == k for x in pred])
                            for k in range(1, 6)},
            "combined_0_2_s": binary_metrics([x in (1, 2) for x in truth],
                                               [x in (1, 2) for x in pred]),
            "combined_0_5_s": binary_metrics([x > 0 for x in truth], [x > 0 for x in pred]),
            "horizon_confusion_6x6": matrix,
            "gt_positive_windows": len(positive_windows),
            "gt_pair_present_windows": present,
            "visible_pair_coverage_ceiling": safe_div(present, len(positive_windows)),
            "correct_gt_actor_pair_recall": safe_div(pair_hits, len(positive_windows)),
            "correct_gt_actor_pair_recall_given_coverage": safe_div(pair_hits, present),
            "correct_gt_actor_pair_hits": pair_hits}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True, help="official train JSONL")
    ap.add_argument("--val", required=True, help="official val104 JSONL")
    ap.add_argument("--v2-checkpoint", required=True)
    ap.add_argument("--out", required=True)
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
    if args.epochs < 1 or args.batch_size < 1:
        ap.error("epochs and batch size must be positive")
    learning_rates = [float(x) for x in args.lr_grid.split(",") if x.strip()]
    if not learning_rates or any(x <= 0 for x in learning_rates):
        ap.error("--lr-grid must contain positive learning rates")
    verify_manifest(args.train, "train", smoke=args.smoke)
    train = read_rows(args.train, "train")
    assignments = folds_by_scenario(train, args.seed)
    val_path = Path(args.val)
    if Path(args.train).resolve() == val_path.resolve():
        ap.error("train and val files must differ")
    # The validation file is read only after CV and final fitting are frozen.
    cv_candidates = []
    for lr in learning_rates:
        oof = []
        for fold in range(5):
            fitting = [r for r in train if assignments[r["scenario_id"]] != fold]
            held = [r for r in train if assignments[r["scenario_id"]] == fold]
            print(f"lr={lr} fold {fold + 1}/5: {len(fitting)} fit, {len(held)} held", flush=True)
            model = fit(fitting, args.v2_checkpoint, args.device, epochs=args.epochs,
                        batch_size=args.batch_size, lr=lr, hidden_dim=args.hidden_dim,
                        dropout=args.dropout, seed=args.seed + fold)
            oof.extend(predict(model, held, args.device, args.batch_size))
            del model
        threshold = select_threshold(oof)
        cv_candidates.append({"lr": lr, "threshold": threshold,
                              "report": report(oof, train, threshold)})
    chosen = max(cv_candidates,
                 key=lambda c: ((c["report"]["overall_0_5_s"]["f1"] or -1), -c["lr"]))
    threshold = chosen["threshold"]
    cv_report = chosen["report"]
    model = fit(train, args.v2_checkpoint, args.device, epochs=args.epochs,
                batch_size=args.batch_size, lr=chosen["lr"], hidden_dim=args.hidden_dim,
                dropout=args.dropout, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "v2_checkpoint": str(Path(args.v2_checkpoint).resolve()),
                "hidden_dim": args.hidden_dim, "dropout": args.dropout,
                "threshold": threshold, "lr": chosen["lr"]}, out / "collision_risk_final.pt")
    verify_manifest(val_path, "val", smoke=args.smoke)
    val = read_rows(val_path, "val")
    overlap = {r["scenario_id"] for r in train} & {r["scenario_id"] for r in val}
    if overlap:
        raise ValueError(f"Train/val scenario overlap: {sorted(overlap)[:5]}")
    val_records = predict(model, val, args.device, args.batch_size)
    result = {"config": vars(args), "smoke_run": args.smoke,
              "fold_assignments": assignments, "cv_candidates": cv_candidates,
              "selected_lr": chosen["lr"],
              "decision_rule": "maximum pair positive probability >= threshold; selected pair's largest positive class",
              "cv_train": cv_report, "val104": report(val_records, val, threshold)}
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"cv_train": cv_report, "val104": result["val104"]}, indent=2))


if __name__ == "__main__":
    main()
