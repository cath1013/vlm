from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from traffic_llm.predict_model import (
    N_HISTORY_FEATURES,
    N_HISTORY_STEPS,
    N_INTERACTION_FEATURES,
)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def window_end(label: str) -> float:
    # "0-5" -> 5.0, "2-7" -> 7.0
    return float(label.rsplit("-", 1)[1])


def world_gt(row):
    e0, n0 = row["origin_enu"]
    return [
        [e0 + float(p[0]), n0 + float(p[1])]
        for p in row["target_offsets"]
    ]


def load_model(path, device):
    model = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    model = model.to(device)
    model.eval()
    return model


def infer_world(model, row, device):
    g = torch.tensor(
        row["global"], dtype=torch.float32, device=device
    )
    c = torch.tensor(
        row["candidates"], dtype=torch.float32, device=device
    )

    hist = row.get("history")
    if hist:
        h = torch.tensor(hist, dtype=torch.float32, device=device)
    else:
        h = torch.zeros(
            (N_HISTORY_STEPS, N_HISTORY_FEATURES),
            dtype=torch.float32,
            device=device,
        )

    interactions = row.get("interactions") or []
    if interactions:
        inter = torch.tensor(
            interactions, dtype=torch.float32, device=device
        )
    else:
        inter = torch.zeros(
            (0, N_INTERACTION_FEATURES),
            dtype=torch.float32,
            device=device,
        )

    with torch.no_grad():
        if getattr(model, "accepts_interactions", False):
            out = model(g, c, h, inter)
        elif getattr(model, "accepts_history", False):
            out = model(g, c, h)
        else:
            out = model(g, c)

    if isinstance(out, (tuple, list)):
        out = out[0]

    offsets = out.reshape(-1, 2).detach().cpu().tolist()

    e0, n0 = row["origin_enu"]

    # Model output is t=1..K. Add current position as t=0.
    pts = [[float(e0), float(n0)]]
    pts += [
        [float(e0) + float(de), float(n0) + float(dn)]
        for de, dn in offsets
    ]
    return pts


def trajectory_stats(pred, gt):
    K = min(len(pred), len(gt))
    if K <= 1:
        return {}

    errs = [dist(pred[k], gt[k]) for k in range(1, K)]

    steps = [
        dist(pred[k], pred[k - 1])
        for k in range(1, K)
    ]
    gt_steps = [
        dist(gt[k], gt[k - 1])
        for k in range(1, K)
    ]

    return {
        "ade_m": sum(errs) / len(errs),
        "fde_m": errs[-1],
        "travel_m": sum(steps),
        "gt_travel_m": sum(gt_steps),
        "travel_error_m": sum(steps) - sum(gt_steps),
        "final_displacement_m": dist(pred[0], pred[K - 1]),
        "gt_final_displacement_m": dist(gt[0], gt[K - 1]),
        "step_distance_m": steps,
        "gt_step_distance_m": gt_steps,
    }


def mean_path_shift(a, b):
    K = min(len(a), len(b))
    if K <= 1:
        return 0.0
    vals = [dist(a[k], b[k]) for k in range(1, K)]
    return sum(vals) / len(vals)


def pair_distances(a, b):
    K = min(len(a), len(b))
    return [dist(a[k], b[k]) for k in range(K)]


def interp(path, t):
    if not path:
        return None

    if t <= 0:
        return path[0]

    if t >= len(path) - 1:
        return path[-1]

    lo = int(math.floor(t))
    hi = lo + 1
    alpha = t - lo

    return [
        path[lo][0] * (1.0 - alpha) + path[hi][0] * alpha,
        path[lo][1] * (1.0 - alpha) + path[hi][1] * alpha,
    ]


def distance_at_time(a, b, t):
    pa = interp(a, t)
    pb = interp(b, t)
    if pa is None or pb is None:
        return None
    return dist(pa, pb)


def load_cases(path):
    cases = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            if r.get("status") != "new_fp":
                continue

            sid, window, k = r["key"]

            # Only normal scenarios for this audit.
            if "_normal/" not in sid:
                continue

            for contact in r["pair005"]["contact_pairs"]:
                cases.append({
                    "scenario": sid,
                    "window": window,
                    "t_s": window_end(window),
                    "k": int(k),
                    "actor_a": contact["a"],
                    "actor_b": contact["b"],
                    "contact_time_s": float(contact["time_s"]),
                    "payload_clearance_m": float(contact["clearance"]),
                })

    return cases


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--changed",
        default="out/contact_changed_buckets.jsonl",
    )
    ap.add_argument(
        "--samples",
        default="out/predict_dataset_interaction_v1/samples.jsonl",
    )
    ap.add_argument(
        "--control",
        default=(
            "out/predict_model_interaction_pair_control/"
            "interaction_waypointnet_best.pt"
        ),
    )
    ap.add_argument(
        "--pair005",
        default=(
            "out/predict_model_interaction_pair005/"
            "interaction_waypointnet_best.pt"
        ),
    )
    ap.add_argument(
        "--out",
        default="out/new_fp_trajectory_audit.jsonl",
    )
    ap.add_argument("--device", default="cpu")

    args = ap.parse_args()

    cases = load_cases(args.changed)
    print(f"Normal new-FP contact pairs: {len(cases)}")

    # We only need rows involved in those FP contacts.
    wanted = set()
    for c in cases:
        wanted.add(
            (c["scenario"], round(c["t_s"], 3), c["actor_a"])
        )
        wanted.add(
            (c["scenario"], round(c["t_s"], 3), c["actor_b"])
        )

    print(f"Required actor rows: {len(wanted)}")
    print("Scanning samples.jsonl ...")

    rows = {}

    with open(args.samples, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            key = (
                r.get("scenario_id"),
                round(float(r.get("t_s", -999)), 3),
                r.get("actor_id"),
            )

            if key in wanted:
                rows[key] = r

    print(f"Found actor rows: {len(rows)}/{len(wanted)}")

    missing = sorted(wanted - set(rows))
    if missing:
        print(f"Missing actor rows: {len(missing)}")
        for x in missing[:20]:
            print("  MISSING", x)

    print("Loading checkpoints ...")

    control_model = load_model(args.control, args.device)
    pair_model = load_model(args.pair005, args.device)

    pred_cache = {}

    def get_predictions(key):
        if key in pred_cache:
            return pred_cache[key]

        row = rows[key]

        gt = world_gt(row)
        control = infer_world(control_model, row, args.device)
        pair005 = infer_world(pair_model, row, args.device)

        result = {
            "gt": gt,
            "control": control,
            "pair005": pair005,
            "control_stats": trajectory_stats(control, gt),
            "pair005_stats": trajectory_stats(pair005, gt),
            "control_to_pair005_mean_shift_m": mean_path_shift(
                control, pair005
            ),
        }

        pred_cache[key] = result
        return result

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with out_path.open("w", encoding="utf-8") as f:
        for c in cases:
            ka = (
                c["scenario"],
                round(c["t_s"], 3),
                c["actor_a"],
            )
            kb = (
                c["scenario"],
                round(c["t_s"], 3),
                c["actor_b"],
            )

            if ka not in rows or kb not in rows:
                skipped += 1
                continue

            A = get_predictions(ka)
            B = get_predictions(kb)

            gt_pair = pair_distances(A["gt"], B["gt"])
            control_pair = pair_distances(
                A["control"], B["control"]
            )
            pair005_pair = pair_distances(
                A["pair005"], B["pair005"]
            )

            contact_t = c["contact_time_s"]

            row = {
                **c,
                "actor_a_data": A,
                "actor_b_data": B,

                "pair_center_distance_m": {
                    "gt_by_integer_second": gt_pair,
                    "control_by_integer_second": control_pair,
                    "pair005_by_integer_second": pair005_pair,

                    "gt_min": min(gt_pair) if gt_pair else None,
                    "control_min": (
                        min(control_pair) if control_pair else None
                    ),
                    "pair005_min": (
                        min(pair005_pair) if pair005_pair else None
                    ),

                    "gt_at_contact_time": distance_at_time(
                        A["gt"], B["gt"], contact_t
                    ),
                    "control_at_contact_time": distance_at_time(
                        A["control"], B["control"], contact_t
                    ),
                    "pair005_at_contact_time": distance_at_time(
                        A["pair005"], B["pair005"], contact_t
                    ),
                },

                "ade_change_pair005_minus_control": {
                    "actor_a": (
                        A["pair005_stats"].get("ade_m", 0.0)
                        - A["control_stats"].get("ade_m", 0.0)
                    ),
                    "actor_b": (
                        B["pair005_stats"].get("ade_m", 0.0)
                        - B["control_stats"].get("ade_m", 0.0)
                    ),
                },
            }

            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )
            written += 1

    print()
    print(f"Written pairs : {written}")
    print(f"Skipped pairs : {skipped}")
    print(f"Output        : {out_path}")


if __name__ == "__main__":
    main()
