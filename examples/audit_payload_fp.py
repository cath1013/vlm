"""Read-only audit of saved compact responses; no API calls or regeneration.

Run from the repository root: .venv/bin/python examples/audit_payload_fp.py
Counts are bucket-level, not independent scenarios or causal effect estimates.
"""
import json
from collections import Counter
from itertools import combinations
from pathlib import Path


ROOT = Path("out/experiments")
CURRENT = ROOT / "waypointnet_compact_v2_fixed_val104_gemini3flash_20260909"


def read_response(path):
    response = json.loads(path.read_text())
    payload = response["payload"]
    if isinstance(payload, str):
        payload = json.loads((path.parent.parent / payload).read_text())
    return response, payload


def audit():
    counts = Counter()
    examples = []
    for path in sorted(CURRENT.glob("val/*/*/responses/response_*.json")):
        response, payload = read_response(path)
        scene = json.loads(payload["contents"][0]["parts"][0]["text"])
        label = scene["observation_window"]["label"]
        gt = json.loads((path.parent.parent / f"ground_truth_{label}.json").read_text())
        expected = {row["k"]: row for row in gt["expected"]}
        actors = {a["id"]: dict(zip(scene["current_state_columns"], a["current_state"]))
                  for a in scene["actors"]}
        pairs = [dict(zip(scene["predicted_pair_columns"], row))
                 for row in scene["predicted_closest_pairs"]]
        counts["windows"] += 1
        for prediction in response["answer"]["predictions"]:
            truth = expected[prediction["k"]]
            scorable = truth["scorable"]
            positive = prediction["accident_expected"]
            if scorable:
                counts[("T" if positive == truth["accident_expected"] else "F")
                       + ("P" if positive else "N")] += 1
            if not positive:
                continue
            ids = set(prediction["involved_actor_ids"])
            mutual = [(a, b) for a, b in combinations(sorted(ids & actors.keys()), 2)
                      if a.startswith("EGO_") and b.startswith("EGO_")
                      and a[4:] in actors[b]["observed_by"]
                      and b[4:] in actors[a]["observed_by"]]
            contacts = [p for p in pairs if p["predicted_contact"]
                        and {p["actor_a"], p["actor_b"]} <= ids]
            groups = ["all_alerts"]
            if gt["scenario"]["outcome"] == "normal":
                groups.append("normal_folder_alerts")
            if scorable:
                groups.append("scorable_TP" if truth["accident_expected"] else "scorable_FP")
            for group in groups:
                counts[group] += 1
                counts[group + "_contact_subset"] += bool(contacts)
                counts[group + "_contact_subset_same_bucket"] += any(
                    p["interval_index"] == prediction["k"] for p in contacts)
                counts[group + "_mutual_ego_subset"] += bool(mutual)
                counts[group + "_exact_two_ids"] += len(ids) == 2
                counts[group + "_exact_two_mutual_ego"] += bool(mutual) and len(ids) == 2
            if "scorable_FP" in groups and len(ids) == 2 and mutual:
                examples.append({"response": str(path), "prediction": prediction,
                                 "gt_bucket": truth, "outcome": gt["scenario"]["outcome"],
                                 "data_end_s": gt["data_end_s"],
                                 "actors": {aid: actors[aid] for aid in sorted(ids)},
                                 "contact_pairs": contacts})
    return {"counts": dict(sorted(counts.items())), "exact_mutual_FP_examples": examples}


def compare_compact_runs():
    older = ROOT / "waypointnet_compact_val104"
    newer = ROOT / "waypointnet_compact_reranker_v2_val104"
    counts = Counter()
    changed_fields = Counter()
    for old in sorted(older.glob("val/*/*/responses/response_*.json")):
        new = newer / old.relative_to(older)
        if not new.exists():
            continue
        _, a = read_response(old)
        _, b = read_response(new)
        x = json.loads(a["contents"][0]["parts"][0]["text"])
        y = json.loads(b["contents"][0]["parts"][0]["text"])
        counts["matched_windows"] += 1
        counts["identical_actors"] += x["actors"] == y["actors"]
        counts["identical_system_prompt"] += a.get("systemInstruction") == b.get("systemInstruction")
        for key in x.keys() | y.keys():
            if x.get(key) != y.get(key):
                changed_fields[key] += 1
    return {"counts": dict(counts), "changed_top_level_fields": dict(changed_fields)}


if __name__ == "__main__":
    result = audit()
    result["older_compact_vs_v2"] = compare_compact_runs()
    print(json.dumps(result, ensure_ascii=False, indent=2))
