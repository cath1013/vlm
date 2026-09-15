"""Diagnose saved scorable FP buckets without API calls or pipeline changes.

Output JSON to stdout. Later payload histories are retrospective sensor estimates,
not ground-truth trajectories and never input to the original prediction.
"""
import json
import math
from collections import Counter, defaultdict

from audit_payload_fp import CURRENT, read_response


def analyze():
    windows = []
    histories = defaultdict(dict)
    conflicts = 0
    for path in sorted(CURRENT.glob("val/*/*/responses/response_*.json")):
        response, payload = read_response(path)
        scene = json.loads(payload["contents"][0]["parts"][0]["text"])
        label = scene["observation_window"]["label"]
        gt = json.loads((path.parent.parent / f"ground_truth_{label}.json").read_text())
        scenario = str(path.parent.parent)
        for actor in scene["actors"]:
            history = histories[(scenario, actor["id"])]
            for row in actor["history"]:
                if row[0] in history and history[row[0]] != row:
                    conflicts += 1
                history[row[0]] = row
        windows.append((path, response, scene, gt))
    cases = []
    for path, response, scene, gt in windows:
        expected = {r["k"]: r for r in gt["expected"]}
        true_ids = {aid for r in gt["expected"] if r["accident_expected"]
                    for aid in r["involved_actor_ids"]}
        end = scene["observation_window"]["t_end_s"]
        actors = {a["id"]: a for a in scene["actors"]}
        pairs = [dict(zip(scene["predicted_pair_columns"], row))
                 for row in scene["predicted_closest_pairs"]]
        for pred in response["answer"]["predictions"]:
            truth = expected[pred["k"]]
            if not (truth["scorable"] and not truth["accident_expected"]
                    and pred["accident_expected"]):
                continue
            ids = set(pred["involved_actor_ids"])
            contacts = [dict(p, rank=i + 1) for i, p in enumerate(pairs)
                        if p["predicted_contact"] and {p["actor_a"], p["actor_b"]} <= ids]
            state = {aid: dict(zip(scene["current_state_columns"], actors[aid]["current_state"]))
                     for aid in ids if aid in actors}
            actor_diagnostics = []
            for aid in sorted(ids & actors.keys()):
                a = actors[aid]
                future = [dict(zip(scene["future_path_columns"], f)) for f in a["future_paths"]]
                if not future:
                    continue
                best = max(future, key=lambda f: f["probability"])
                points = best["waypoints_1s_enu_m"]
                observed = histories[(str(path.parent.parent), aid)]
                samples = []
                previous = [state[aid]["e_m"], state[aid]["n_m"]]
                # Only samples through this FP bucket's end, not later behavior.
                for k, point in enumerate(points[:pred["k"]], 1):
                    row = observed.get(end + k)
                    speed = math.dist(previous, point) * 3.6
                    if row:
                        prior_row = observed.get(end + k - 1)
                        samples.append({"t": end + k, "predicted_xy": point,
                                        "observed_xy": row[1:3],
                                        "position_error_m": round(math.dist(point, row[1:3]), 2),
                                        "predicted_step_speed_kph": round(speed, 2),
                                        "observed_step_speed_kph": None if prior_row is None else round(
                                            math.dist(prior_row[1:3], row[1:3]) * 3.6, 2),
                                        "observed_speed_kph": row[3]})
                    previous = point
                actor_diagnostics.append({"id": aid, "current": state[aid],
                                          "history": a["history"], "path_maneuver": best["maneuver"],
                                          "samples": samples})
            mutual = any(a.startswith("EGO_") and b.startswith("EGO_")
                         and a[4:] in state[b]["observed_by"]
                         and b[4:] in state[a]["observed_by"]
                         for a in state for b in state if a != b)
            missed_stop = any(s["observed_speed_kph"] is not None
                              and s["observed_speed_kph"] <= 1
                              and s["predicted_step_speed_kph"] >= 10
                              for a in actor_diagnostics for s in a["samples"])
            large_error = any(s["position_error_m"] >= 5
                              for a in actor_diagnostics for s in a["samples"])
            stop_egos = [a for a in actor_diagnostics if a["id"].startswith("EGO_")
                         and any(s["observed_speed_kph"] is not None and s["observed_speed_kph"] <= 1
                                 and s["predicted_step_speed_kph"] >= 10 for s in a["samples"])]
            spatial_stop = any(s["observed_speed_kph"] is not None and s["observed_speed_kph"] <= 1
                               and s["observed_step_speed_kph"] is not None
                               and s["observed_step_speed_kph"] <= 2
                               and s["predicted_step_speed_kph"] >= 10
                               for a in stop_egos for s in a["samples"])
            braking_stop = any(a["current"]["accel_mps2"] is not None
                               and a["current"]["accel_mps2"] <= -0.5 for a in stop_egos)
            cases.append({"response": str(path), "scenario": path.parent.parent.name,
                          "outcome": gt["scenario"]["outcome"], "window": scene["observation_window"]["label"],
                          "prediction": pred, "gt_bucket": truth,
                          "collision_time_s": (gt.get("collision") or {}).get("time_s"),
                          "gt_positive_actor_ids": sorted(true_ids),
                          "exact_event_pair_wrong_time": len(ids) == 2 and ids == true_ids,
                          "contact_pairs": contacts, "mutual_ego_subset": mutual,
                          "missed_stop_screen": missed_stop, "large_error_screen": large_error,
                          "ego_missed_stop_screen": bool(stop_egos),
                          "ego_spatial_stop_screen": spatial_stop,
                          "ego_stop_with_current_braking": braking_stop,
                          "actors": actor_diagnostics})
    counts = Counter()
    for index, c in enumerate(cases, 1):
        c["case_id"] = f"FP{index:03d}"
        counts[c["outcome"]] += 1
        for key in ["exact_event_pair_wrong_time", "mutual_ego_subset", "missed_stop_screen", "large_error_screen",
                    "ego_missed_stop_screen", "ego_spatial_stop_screen", "ego_stop_with_current_braking"]:
            counts[key] += c[key]
        counts["with_future_samples"] += any(a["samples"] for a in c["actors"])
        counts["with_all_actor_bucket_end_samples"] += bool(c["actors"]) and len(c["actors"]) == len(set(c["prediction"]["involved_actor_ids"])) and all(
            any(s["t"] == c["gt_bucket"]["interval_end_s"] for s in a["samples"]) for a in c["actors"])
        counts["contact_same_bucket"] += any(p["interval_index"] == c["prediction"]["k"] for p in c["contact_pairs"])
    return {"counts": dict(counts), "fp_buckets": len(cases),
            "fp_windows": len({c["response"] for c in cases}),
            "fp_scenarios": len({(c["outcome"], c["scenario"]) for c in cases}),
            "overlapping_history_conflicts": conflicts, "cases": cases}


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
