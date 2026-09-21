"""Control-vs-JointScene predictor-only transition and GT-oracle audit."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compare_predictor_contacts import compact_from_payload, load_run  # noqa: E402
from traffic_llm.schemas import ActorState, PredictedPath  # noqa: E402
from traffic_llm.swept_path import swept_pair_clearances  # noqa: E402


def path_for(root: Path, record: dict) -> Path:
    return (root / "val" / record["outcome"] / record["scenario"].split("/", 1)[1]
            / f"llm_payload_{record['label']}.json")


def actor_map(root: Path, record: dict) -> dict:
    compact = compact_from_payload(path_for(root, record))
    return {a["id"]: a for a in compact["actors"]}


def parse_path(actor: dict):
    current = actor["current_state"]
    paths = actor.get("future_paths") or []
    if not paths:
        return None
    # JointScene V1 is deterministic; retain the most likely path for generic
    # control compatibility.
    row = max(paths, key=lambda p: float(p[1]))
    return [[float(current[0]), float(current[1])]] + [
        [float(x), float(y)] for x, y in row[3]
    ]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def stats(pred, gt):
    valid = [(p, g) for p, g in zip(pred[1:], gt[1:]) if g is not None]
    if not valid:
        return None
    errors = [dist(p, g) for p, g in valid]
    pred_travel = sum(dist(pred[i - 1], pred[i]) for i in range(1, len(pred)))
    gt_travel = sum(dist(gt[i - 1], gt[i]) for i in range(1, len(gt))
                    if gt[i - 1] is not None and gt[i] is not None)
    return {"ade_m": sum(errors) / len(errors), "fde_m": errors[-1],
            "travel_error_m": pred_travel - gt_travel,
            "gt_travel_m": gt_travel}


def gt_path(row):
    e0, n0 = row["origin_enu"]
    return [[float(e0), float(n0)]] + [
        ([float(e0) + float(off[0]), float(n0) + float(off[1])]
         if valid else None)
        for off, valid in zip(row["target_offsets"], row["target_mask"])
    ]


def gt_oracle(aid, bid, ra, rb):
    pa, pb = gt_path(ra), gt_path(rb)
    # GT can end before +5 s near a scenario boundary.  Oracle replacement is
    # still valid up to the common observed future horizon; never extrapolate.
    horizon = 0
    for k, (xa, xb) in enumerate(zip(pa[1:], pb[1:]), 1):
        if xa is None or xb is None:
            break
        horizon = k
    if not horizon:
        return None
    pa, pb = pa[:horizon + 1], pb[:horizon + 1]
    def actor(actor_id, row, points):
        g = row["global"]
        heading = math.degrees(math.atan2(float(g[4]), float(g[5]))) % 360.0
        state = ActorState(actor_id, "observed", row["actor_class"], tuple(points[0]),
                           heading, float(g[0]), float(g[2]))
        state.predictions = [PredictedPath("GT oracle", 1.0,
                                           [tuple(x) for x in points], float(horizon))]
        return state
    pairs = swept_pair_clearances([actor(aid, ra, pa), actor(bid, rb, pb)], float(horizon))
    if not pairs:
        return {"contact": False, "minimum_clearance_m": None,
                "contact_time_s": None, "available_horizon_s": horizon}
    pair = pairs[0]
    return {"contact": bool(pair.predicted_contact),
            "minimum_clearance_m": pair.minimum_clearance_m,
            "contact_time_s": pair.time_after_observation_s,
            "available_horizon_s": horizon}


def _turn_error(pred, gt):
    if not pred or not gt or gt[-1] is None:
        return False
    pdx, pdy = pred[-1][0] - pred[0][0], pred[-1][1] - pred[0][1]
    gdx, gdy = gt[-1][0] - gt[0][0], gt[-1][1] - gt[0][1]
    if math.hypot(pdx, pdy) < 2.0 or math.hypot(gdx, gdy) < 2.0:
        return False
    dot = (pdx * gdx + pdy * gdy) / max(math.hypot(pdx, pdy) * math.hypot(gdx, gdy), 1e-9)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot)))) >= 45.0


def label_error(sa, sb, pred_a, gt_a, pred_b, gt_b):
    if sa is None or sb is None:
        return "other"
    travel = max(sa["travel_error_m"], sb["travel_error_m"])
    if travel >= 3.0:
        if min(sa["gt_travel_m"], sb["gt_travel_m"]) <= 1.0:
            return "missed_stop_or_slowdown"
        if _turn_error(pred_a, gt_a) or _turn_error(pred_b, gt_b):
            return "turn_shape_error"
        return "over_travel_ge_3m"
    a, b = sa["ade_m"], sb["ade_m"]
    if max(a, b) >= 1.5 * max(min(a, b), 0.25):
        return "one_actor_dominant_error"
    if min(a, b) >= 1.0:
        return "two_actor_pair_inconsistency"
    return "other"


def contacts(record):
    return record["contact_pairs"] if record["pred_any_contact"] else []


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default="out/experiments/contact_audit_control_val104")
    ap.add_argument("--joint", default="out/experiments/contact_audit_joint_scene_v1_val104")
    ap.add_argument("--scenes", default="out/predict_dataset_scene_v1/scenes.jsonl")
    ap.add_argument("--out", default="out/joint_scene_transition_audit.jsonl")
    ap.add_argument("--summary", default="out/joint_scene_transition_summary.json")
    args = ap.parse_args(argv)
    control, joint = load_run(Path(args.control)), load_run(Path(args.joint))
    if set(control) != set(joint):
        raise SystemExit("control/joint bucket sets differ")

    rows_needed = set()
    new_keys = []
    for key in control:
        c, j = control[key], joint[key]
        if not j["gt_positive"] and j["pred_any_contact"] and not c["pred_any_contact"]:
            new_keys.append(key)
            for item in contacts(j):
                rows_needed |= {(key[0], float(key[1].rsplit("-", 1)[1]), item["a"]),
                                (key[0], float(key[1].rsplit("-", 1)[1]), item["b"])}
    rows = {}
    with open(args.scenes, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sid, t = row["scenario_id"], float(row["t_s"])
            for actor in row["actors"]:
                key = (sid, t, actor["actor_id"])
                if key in rows_needed:
                    rows[key] = actor

    transition = Counter()
    pair_transition = Counter()
    for key in control:
        c, j = control[key], joint[key]
        if not c["gt_positive"]:
            transition[("negative", "FP" if c["pred_any_contact"] else "TN",
                        "FP" if j["pred_any_contact"] else "TN")] += 1
        else:
            transition[("positive", "TP" if c["pred_any_contact"] else "FN",
                        "TP" if j["pred_any_contact"] else "FN")] += 1
            pair_transition[("hit" if c["target_pair_hit"] else "miss",
                             "hit" if j["target_pair_hit"] else "miss")] += 1

    payload_cache = {}
    records, oracle = [], Counter()
    for key in new_keys:
        c, j = control[key], joint[key]
        if key not in payload_cache:
            payload_cache[key] = actor_map(Path(args.joint), j)
        payload = payload_cache[key]
        t = float(key[1].rsplit("-", 1)[1])
        for contact in contacts(j):
            aid, bid = contact["a"], contact["b"]
            ra, rb = rows.get((key[0], t, aid)), rows.get((key[0], t, bid))
            pa, pb = payload.get(aid), payload.get(bid)
            pred_a, pred_b = (parse_path(pa) if pa else None), (parse_path(pb) if pb else None)
            sa = stats(pred_a, gt_path(ra)) if pred_a and ra else None
            sb = stats(pred_b, gt_path(rb)) if pred_b and rb else None
            outcome = gt_oracle(aid, bid, ra, rb) if ra and rb else None
            oracle_class = ("geometry_contact_layer_issue" if outcome and outcome["contact"]
                            else "predictor_trajectory_error" if outcome else "oracle_unavailable")
            detail = (label_error(sa, sb, pred_a, gt_path(ra), pred_b, gt_path(rb))
                      if oracle_class == "predictor_trajectory_error" else None)
            oracle[(oracle_class, detail)] += 1
            records.append({
                "scenario_id": key[0], "window": key[1], "observation_time_s": t,
                "bucket": key[2], "predicted_contact_pair": [aid, bid],
                "predicted_contact_time_s": contact["time_s"],
                "predicted_minimum_clearance_m": contact["clearance"],
                "actor_a": {"class": pa.get("class") if pa else None,
                            "has_route_candidate": ra.get("has_route_candidate") if ra else None,
                            "speed_mps": ra["global"][0] if ra else None, "trajectory": sa},
                "actor_b": {"class": pb.get("class") if pb else None,
                            "has_route_candidate": rb.get("has_route_candidate") if rb else None,
                            "speed_mps": rb["global"][0] if rb else None, "trajectory": sb},
                "gt_gt_oracle": outcome, "classification": oracle_class,
                "trajectory_error_detail": detail,
            })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"transitions": {"|".join(k): v for k, v in transition.items()},
               "gt_pair_transitions": {"|".join(k): v for k, v in pair_transition.items()},
               "new_fp_buckets": len(new_keys), "new_fp_contact_rows": len(records),
               "oracle": {"|".join(str(x) for x in k): v for k, v in oracle.items()},
               "missing_scene_rows": len(rows_needed - set(rows)), "detail_path": args.out}
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
