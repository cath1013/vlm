from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from traffic_llm.carla_map import find_xodr
from traffic_llm.config import PipelineConfig
from traffic_llm.da_runner import DeepAccidentRunner


def euclid(a, b):
    if a is None or b is None:
        return None
    return math.hypot(float(a[0]) - float(b[0]),
                      float(a[1]) - float(b[1]))


def window_end(label: str) -> float:
    return float(label.rsplit("-", 1)[1])


# ----------------------------------------------------------------------
# Read compact_geometry_v3 from saved Gemini request payload
# ----------------------------------------------------------------------

def compact_from_payload(path: Path) -> dict:
    req = json.loads(path.read_text(encoding="utf-8"))

    for content in req.get("contents", []):
        for part in content.get("parts", []):
            text = part.get("text")

            if not isinstance(text, str):
                continue

            text = text.strip()

            if not text.startswith("{"):
                continue

            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue

            if obj.get("representation") == "compact_geometry_v3":
                return obj

    raise ValueError(f"compact_geometry_v3 not found: {path}")


def actor_map(compact: dict):
    return {
        a["id"]: a
        for a in compact.get("actors", [])
    }


def parse_actor_paths(actor: dict):
    """
    Compact future_path columns:

      maneuver
      probability
      to_roads
      waypoints_1s_enu_m
      truncated

    waypoints start at +1 s, so prepend current world position as t=0.
    """
    current = actor["current_state"]
    current_xy = [float(current[0]), float(current[1])]

    out = []

    for i, p in enumerate(actor.get("future_paths", [])):
        maneuver = p[0]
        probability = float(p[1])
        to_roads = p[2]
        future = p[3]
        truncated = bool(p[4])

        pts = [current_xy]

        for xy in future:
            pts.append([float(xy[0]), float(xy[1])])

        out.append({
            "index": i,
            "maneuver": maneuver,
            "probability": probability,
            "to_roads": to_roads,
            "truncated": truncated,
            "points": pts,
        })

    return out


def interp_path(points, t):
    """
    Predictor path:
      index 0 = current
      index 1 = +1 s
      ...
    """
    if not points:
        return None

    if t < -1e-9:
        return None

    if t > len(points) - 1 + 1e-9:
        return None

    if abs(t - round(t)) < 1e-9:
        i = int(round(t))
        if 0 <= i < len(points):
            return points[i]

    lo = int(math.floor(t))
    hi = lo + 1

    if lo < 0 or hi >= len(points):
        return None

    alpha = t - lo

    return [
        points[lo][0] * (1.0 - alpha) + points[hi][0] * alpha,
        points[lo][1] * (1.0 - alpha) + points[hi][1] * alpha,
    ]


def best_pair_combo(paths_a, paths_b, t):
    """
    We do not have the internal path indices recorded in predicted_closest_pairs.

    For diagnostics, select the A/B path combination with the smallest
    centre-to-centre distance at the recorded contact time.

    For WaypointNet actors there is normally only one path anyway.
    """
    best = None

    for pa in paths_a:
        xa = interp_path(pa["points"], t)

        if xa is None:
            continue

        for pb in paths_b:
            xb = interp_path(pb["points"], t)

            if xb is None:
                continue

            d = euclid(xa, xb)

            row = {
                "distance_m": d,
                "actor_a_path_index": pa["index"],
                "actor_b_path_index": pb["index"],
                "actor_a_maneuver": pa["maneuver"],
                "actor_b_maneuver": pb["maneuver"],
                "actor_a_probability": pa["probability"],
                "actor_b_probability": pb["probability"],
                "actor_a_point": xa,
                "actor_b_point": xb,
                "actor_a_path": pa,
                "actor_b_path": pb,
            }

            if best is None or d < best["distance_m"]:
                best = row

    return best


def prediction_signature(paths):
    """
    Used only to determine whether an actor's saved prediction is effectively
    unchanged between the two checkpoints.

    An unchanged prediction is useful for spotting cases where one side likely
    went through the same fallback path in both runs.
    """
    sig = []

    for p in paths:
        pts = [
            [round(float(x), 4), round(float(y), 4)]
            for x, y in p["points"]
        ]

        sig.append({
            "maneuver": p["maneuver"],
            "probability": round(float(p["probability"]), 6),
            "truncated": p["truncated"],
            "points": pts,
        })

    return sig


# ----------------------------------------------------------------------
# New-FP cases
# ----------------------------------------------------------------------

def load_cases(path: Path):
    cases = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if row.get("status") != "new_fp":
                continue

            sid, label, k = row["key"]

            # We are diagnosing the normal-scene deterioration first.
            if "_normal/" not in sid:
                continue

            contacts = row["pair005"]["contact_pairs"]

            for contact in contacts:
                cases.append({
                    "scenario_id": sid,
                    "scenario_type": sid.split("/", 1)[0],
                    "scenario": sid.split("/", 1)[1],
                    "window": label,
                    "t_end": window_end(label),
                    "k": int(k),
                    "actor_a": contact["a"],
                    "actor_b": contact["b"],
                    "contact_time_s": float(contact["time_s"]),
                    "payload_clearance_m": float(contact["clearance"]),
                })

    return cases


# ----------------------------------------------------------------------
# Raw DeepAccident GT reconstruction
# ----------------------------------------------------------------------

def actor_series(snaps, actor_id: str, start: float, end: float):
    series = []

    for snap in snaps:
        if snap.t < start - 1e-6 or snap.t > end + 1e-6:
            continue

        actor = next(
            (a for a in snap.actors if a.actor_id == actor_id),
            None,
        )

        if actor is None:
            continue

        series.append((
            float(snap.t),
            [float(actor.world_xy[0]), float(actor.world_xy[1])],
        ))

    return series


def interp_series(series, t, max_bracket_s=1.01):
    """
    Interpolate raw sensor3d positions.

    Snapshots are generated at 2 Hz, so ideal spacing is 0.5 s.
    We permit up to 1.01 s across a missing intermediate observation,
    but never extrapolate beyond the first/last observed point.
    """
    if not series:
        return None

    times = [x[0] for x in series]

    i = bisect.bisect_left(times, t)

    if i < len(series) and abs(series[i][0] - t) < 1e-6:
        return series[i][1]

    if i == 0 or i >= len(series):
        return None

    t0, p0 = series[i - 1]
    t1, p1 = series[i]

    if t1 - t0 > max_bracket_s:
        return None

    if not (t0 <= t <= t1):
        return None

    alpha = (t - t0) / max(t1 - t0, 1e-12)

    return [
        p0[0] * (1.0 - alpha) + p1[0] * alpha,
        p0[1] * (1.0 - alpha) + p1[1] * alpha,
    ]


def gt_integer_path(series, t_end, horizon=5):
    return [
        interp_series(series, t_end + float(k))
        for k in range(horizon + 1)
    ]


# ----------------------------------------------------------------------
# Trajectory statistics
# ----------------------------------------------------------------------

def trajectory_stats(pred_points, gt_points):
    if not pred_points or not gt_points:
        return None

    n = min(len(pred_points), len(gt_points))

    errs = []

    for k in range(1, n):
        if pred_points[k] is None or gt_points[k] is None:
            continue

        errs.append(euclid(pred_points[k], gt_points[k]))

    if not errs:
        return None

    # Travel is evaluated only across consecutive timesteps that exist in both.
    pred_travel = 0.0
    gt_travel = 0.0
    travel_steps = 0

    for k in range(1, n):
        if (
            pred_points[k - 1] is None
            or pred_points[k] is None
            or gt_points[k - 1] is None
            or gt_points[k] is None
        ):
            continue

        pred_travel += euclid(pred_points[k - 1], pred_points[k])
        gt_travel += euclid(gt_points[k - 1], gt_points[k])
        travel_steps += 1

    return {
        "n_ade_steps": len(errs),
        "ade_m": sum(errs) / len(errs),
        "fde_m": errs[-1],
        "n_travel_steps": travel_steps,
        "pred_travel_m": pred_travel,
        "gt_travel_m": gt_travel,
        "travel_error_m": pred_travel - gt_travel,
    }


def path_pair_distances(pa, pb):
    n = min(len(pa), len(pb))
    vals = []

    for k in range(n):
        if pa[k] is None or pb[k] is None:
            vals.append(None)
        else:
            vals.append(euclid(pa[k], pb[k]))

    return vals


def finite_min(values):
    vals = [x for x in values if x is not None]
    return min(vals) if vals else None


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        default="/home/sryu/inclab-nas/DeepAccident",
    )
    ap.add_argument(
        "--carla-maps",
        default="carla_map",
    )
    ap.add_argument(
        "--changed",
        default="out/contact_changed_buckets.jsonl",
    )
    ap.add_argument(
        "--control-run",
        default="out/experiments/contact_audit_control_val104",
    )
    ap.add_argument(
        "--pair-run",
        default="out/experiments/contact_audit_pair005_val104",
    )
    ap.add_argument(
        "--out",
        default="out/new_fp_raw_replay.jsonl",
    )
    ap.add_argument(
        "--summary",
        default="out/new_fp_raw_replay_summary.json",
    )
    ap.add_argument(
        "--rate",
        type=float,
        default=2.0,
    )

    args = ap.parse_args()

    changed = Path(args.changed)
    control_root = Path(args.control_run)
    pair_root = Path(args.pair_run)

    cases = load_cases(changed)

    print(f"Normal new-FP contact pairs: {len(cases)}")

    unique_scenarios = sorted({
        (c["scenario_type"], c["scenario"])
        for c in cases
    })

    print(f"Unique raw scenarios to rebuild: {len(unique_scenarios)}")

    # Raw reconstruction config.
    cfg = PipelineConfig()
    cfg.deepaccident.observation_mode = "sensor3d"

    runner = DeepAccidentRunner(args.root, cfg)

    raw_cache = {}

    for idx, (scenario_type, scenario) in enumerate(unique_scenarios, 1):
        town = scenario.split("_", 1)[0]

        xodr = find_xodr(town, [args.carla_maps])

        if not xodr:
            raise RuntimeError(
                f"OpenDRIVE not found for {town} under {args.carla_maps}"
            )

        print(
            f"[{idx:02d}/{len(unique_scenarios)}] "
            f"raw GT rebuild: {scenario_type}/{scenario}",
            flush=True,
        )

        res = runner.build(
            scenario,
            scenario_type,
            opendrive_path=xodr,
        )

        snaps = list(res.snapshots(rate_hz=args.rate))

        raw_cache[(scenario_type, scenario)] = snaps

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []

    payload_missing = 0
    actor_missing = 0
    gt_contact_missing = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for i, c in enumerate(cases, 1):
            sc = c["scenario"]
            label = c["window"]

            control_payload = (
                control_root
                / "val"
                / "normal"
                / sc
                / f"llm_payload_{label}.json"
            )

            pair_payload = (
                pair_root
                / "val"
                / "normal"
                / sc
                / f"llm_payload_{label}.json"
            )

            if not control_payload.is_file() or not pair_payload.is_file():
                payload_missing += 1
                print(
                    f"WARNING payload missing: {c['scenario_id']} {label}"
                )
                continue

            control_compact = compact_from_payload(control_payload)
            pair_compact = compact_from_payload(pair_payload)

            control_actors = actor_map(control_compact)
            pair_actors = actor_map(pair_compact)

            aid = c["actor_a"]
            bid = c["actor_b"]

            if (
                aid not in control_actors
                or bid not in control_actors
                or aid not in pair_actors
                or bid not in pair_actors
            ):
                actor_missing += 1
                print(
                    f"WARNING payload actor missing: "
                    f"{c['scenario_id']} {label} {aid}/{bid}"
                )
                continue

            ca_paths = parse_actor_paths(control_actors[aid])
            cb_paths = parse_actor_paths(control_actors[bid])

            pa_paths = parse_actor_paths(pair_actors[aid])
            pb_paths = parse_actor_paths(pair_actors[bid])

            contact_t = c["contact_time_s"]

            control_combo = best_pair_combo(
                ca_paths,
                cb_paths,
                contact_t,
            )

            pair_combo = best_pair_combo(
                pa_paths,
                pb_paths,
                contact_t,
            )

            if control_combo is None or pair_combo is None:
                actor_missing += 1
                print(
                    f"WARNING no valid path combo: "
                    f"{c['scenario_id']} {label} {aid}/{bid}"
                )
                continue

            snaps = raw_cache[(c["scenario_type"], c["scenario"])]

            raw_start = c["t_end"]
            raw_end = c["t_end"] + 5.0

            series_a = actor_series(
                snaps, aid, raw_start, raw_end
            )
            series_b = actor_series(
                snaps, bid, raw_start, raw_end
            )

            gt_a = gt_integer_path(
                series_a,
                c["t_end"],
                horizon=5,
            )

            gt_b = gt_integer_path(
                series_b,
                c["t_end"],
                horizon=5,
            )

            gt_a_contact = interp_series(
                series_a,
                c["t_end"] + contact_t,
            )

            gt_b_contact = interp_series(
                series_b,
                c["t_end"] + contact_t,
            )

            gt_contact_distance = euclid(
                gt_a_contact,
                gt_b_contact,
            )

            if gt_contact_distance is None:
                gt_contact_missing += 1

            control_a_path = control_combo["actor_a_path"]["points"]
            control_b_path = control_combo["actor_b_path"]["points"]

            pair_a_path = pair_combo["actor_a_path"]["points"]
            pair_b_path = pair_combo["actor_b_path"]["points"]

            control_pair_integer = path_pair_distances(
                control_a_path,
                control_b_path,
            )

            pair_pair_integer = path_pair_distances(
                pair_a_path,
                pair_b_path,
            )

            gt_pair_integer = path_pair_distances(
                gt_a,
                gt_b,
            )

            control_a_stats = trajectory_stats(
                control_a_path,
                gt_a,
            )
            control_b_stats = trajectory_stats(
                control_b_path,
                gt_b,
            )

            pair_a_stats = trajectory_stats(
                pair_a_path,
                gt_a,
            )
            pair_b_stats = trajectory_stats(
                pair_b_path,
                gt_b,
            )

            unchanged_a = (
                prediction_signature(ca_paths)
                == prediction_signature(pa_paths)
            )

            unchanged_b = (
                prediction_signature(cb_paths)
                == prediction_signature(pb_paths)
            )

            record = {
                **c,

                "prediction_source": (
                    "saved validation compact payloads; "
                    "therefore fallback behaviour is preserved"
                ),

                "raw_gt_source": (
                    "DeepAccident sensor3d snapshots rebuilt "
                    "from raw validation scenario"
                ),

                "control": {
                    "actor_a_n_paths": len(ca_paths),
                    "actor_b_n_paths": len(cb_paths),

                    "selected_actor_a_path_index":
                        control_combo["actor_a_path_index"],

                    "selected_actor_b_path_index":
                        control_combo["actor_b_path_index"],

                    "selected_actor_a_maneuver":
                        control_combo["actor_a_maneuver"],

                    "selected_actor_b_maneuver":
                        control_combo["actor_b_maneuver"],

                    "center_distance_at_pair005_contact_time_m":
                        control_combo["distance_m"],

                    "integer_second_pair_distance_m":
                        control_pair_integer,

                    "minimum_integer_second_pair_distance_m":
                        finite_min(control_pair_integer),

                    "actor_a_path":
                        control_a_path,

                    "actor_b_path":
                        control_b_path,

                    "actor_a_stats_vs_raw_gt":
                        control_a_stats,

                    "actor_b_stats_vs_raw_gt":
                        control_b_stats,
                },

                "pair005": {
                    "actor_a_n_paths": len(pa_paths),
                    "actor_b_n_paths": len(pb_paths),

                    "selected_actor_a_path_index":
                        pair_combo["actor_a_path_index"],

                    "selected_actor_b_path_index":
                        pair_combo["actor_b_path_index"],

                    "selected_actor_a_maneuver":
                        pair_combo["actor_a_maneuver"],

                    "selected_actor_b_maneuver":
                        pair_combo["actor_b_maneuver"],

                    "center_distance_at_contact_time_m":
                        pair_combo["distance_m"],

                    "integer_second_pair_distance_m":
                        pair_pair_integer,

                    "minimum_integer_second_pair_distance_m":
                        finite_min(pair_pair_integer),

                    "actor_a_path":
                        pair_a_path,

                    "actor_b_path":
                        pair_b_path,

                    "actor_a_stats_vs_raw_gt":
                        pair_a_stats,

                    "actor_b_stats_vs_raw_gt":
                        pair_b_stats,
                },

                "raw_gt": {
                    "actor_a_series_2hz":
                        series_a,

                    "actor_b_series_2hz":
                        series_b,

                    "actor_a_integer_second_path":
                        gt_a,

                    "actor_b_integer_second_path":
                        gt_b,

                    "integer_second_pair_distance_m":
                        gt_pair_integer,

                    "minimum_integer_second_pair_distance_m":
                        finite_min(gt_pair_integer),

                    "actor_a_at_pair005_contact_time":
                        gt_a_contact,

                    "actor_b_at_pair005_contact_time":
                        gt_b_contact,

                    "center_distance_at_pair005_contact_time_m":
                        gt_contact_distance,

                    "actor_a_complete_0_to_5":
                        all(x is not None for x in gt_a),

                    "actor_b_complete_0_to_5":
                        all(x is not None for x in gt_b),
                },

                "change": {
                    "pair005_minus_control_center_distance_at_contact_m":
                        pair_combo["distance_m"]
                        - control_combo["distance_m"],

                    "actor_a_prediction_unchanged":
                        unchanged_a,

                    "actor_b_prediction_unchanged":
                        unchanged_b,

                    "actor_a_ade_change_m":
                        None
                        if not control_a_stats or not pair_a_stats
                        else (
                            pair_a_stats["ade_m"]
                            - control_a_stats["ade_m"]
                        ),

                    "actor_b_ade_change_m":
                        None
                        if not control_b_stats or not pair_b_stats
                        else (
                            pair_b_stats["ade_m"]
                            - control_b_stats["ade_m"]
                        ),

                    "actor_a_travel_error_change_m":
                        None
                        if not control_a_stats or not pair_a_stats
                        else (
                            pair_a_stats["travel_error_m"]
                            - control_a_stats["travel_error_m"]
                        ),

                    "actor_b_travel_error_change_m":
                        None
                        if not control_b_stats or not pair_b_stats
                        else (
                            pair_b_stats["travel_error_m"]
                            - control_b_stats["travel_error_m"]
                        ),
                },
            }

            records.append(record)

            fout.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    center_changes = [
        r["change"][
            "pair005_minus_control_center_distance_at_contact_m"
        ]
        for r in records
    ]

    gt_contact_vals = [
        r["raw_gt"][
            "center_distance_at_pair005_contact_time_m"
        ]
        for r in records
        if r["raw_gt"][
            "center_distance_at_pair005_contact_time_m"
        ] is not None
    ]

    control_contact_vals = [
        r["control"][
            "center_distance_at_pair005_contact_time_m"
        ]
        for r in records
    ]

    pair_contact_vals = [
        r["pair005"][
            "center_distance_at_contact_time_m"
        ]
        for r in records
    ]

    both_gt_complete = [
        r for r in records
        if (
            r["raw_gt"]["actor_a_complete_0_to_5"]
            and r["raw_gt"]["actor_b_complete_0_to_5"]
        )
    ]

    closer = sum(x < -1e-6 for x in center_changes)
    farther = sum(x > 1e-6 for x in center_changes)
    same = len(center_changes) - closer - farther

    both_changed = sum(
        (
            not r["change"]["actor_a_prediction_unchanged"]
            and not r["change"]["actor_b_prediction_unchanged"]
        )
        for r in records
    )

    one_changed = sum(
        (
            r["change"]["actor_a_prediction_unchanged"]
            != r["change"]["actor_b_prediction_unchanged"]
        )
        for r in records
    )

    neither_changed = sum(
        (
            r["change"]["actor_a_prediction_unchanged"]
            and r["change"]["actor_b_prediction_unchanged"]
        )
        for r in records
    )

    summary = {
        "normal_new_fp_contact_pairs_requested": len(cases),
        "records_written": len(records),
        "payload_missing": payload_missing,
        "payload_actor_or_path_missing": actor_missing,

        "raw_gt_contact_distance_available":
            len(gt_contact_vals),

        "raw_gt_contact_distance_missing":
            gt_contact_missing,

        "raw_gt_complete_0_to_5_pairs":
            len(both_gt_complete),

        "pair005_closer_than_control_at_contact":
            closer,

        "pair005_farther_than_control_at_contact":
            farther,

        "same_center_distance_at_contact":
            same,

        "mean_center_distance_change_pair005_minus_control_m":
            mean(center_changes) if center_changes else None,

        "mean_raw_gt_center_distance_at_contact_m":
            mean(gt_contact_vals) if gt_contact_vals else None,

        "mean_control_center_distance_at_contact_m":
            mean(control_contact_vals)
            if control_contact_vals else None,

        "mean_pair005_center_distance_at_contact_m":
            mean(pair_contact_vals)
            if pair_contact_vals else None,

        "prediction_change_pattern": {
            "both_actors_changed": both_changed,
            "only_one_actor_changed": one_changed,
            "neither_actor_changed": neither_changed,
        },

        "notes": [
            (
                "Saved compact payload waypoints are serialized/rounded "
                "representations of the validation predictions. The original "
                "predicted-contact label and contact time remain those produced "
                "by the validation contact checker."
            ),
            (
                "An actor prediction unchanged between control and pair005 is "
                "consistent with checkpoint-independent fallback behaviour, but "
                "this script does not label it as fallback with certainty."
            ),
        ],
    }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 68)
    print("RAW NEW-FP AUDIT SUMMARY")
    print("=" * 68)

    for key, value in summary.items():
        if key in ("notes", "prediction_change_pattern"):
            continue
        print(f"{key:48s} {value}")

    print()
    print("prediction change pattern:")
    for key, value in summary["prediction_change_pattern"].items():
        print(f"  {key:38s} {value}")

    print()
    print(f"Detailed JSONL : {out_path}")
    print(f"Summary JSON   : {summary_path}")


if __name__ == "__main__":
    main()
