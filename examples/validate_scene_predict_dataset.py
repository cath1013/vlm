"""Fast structural and count validation for a scene motion JSONL dataset."""
from __future__ import annotations

import argparse
import json
import math
import os

SUPPORTED_SPLITS = {"train", "val"}


def _finite(value):
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(x) for x in value)
    return True


def validate(data_dir):
    scenes_path = os.path.join(data_dir, "scenes.jsonl")
    report_path = os.path.join(data_dir, "dataset_report.json")
    counts = {"n_scene_samples": 0, "actor_instances": 0, "with_route": 0,
              "without_route": 0, "full_future": 0, "partial_future": 0,
              "zero_future": 0}
    scenarios, scenes_by_split, actors_by_split = set(), {}, {}
    with open(scenes_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            scene = json.loads(line)
            actors = scene.get("actors") or []
            assert actors, f"line {line_no}: empty scene"
            ids = [a["actor_id"] for a in actors]
            assert len(ids) == len(set(ids)), f"line {line_no}: duplicate actor_id"
            counts["n_scene_samples"] += 1
            scenarios.add(scene["scenario_id"])
            split = scene["dataset_split"]
            assert split in SUPPORTED_SPLITS, (
                f"line {line_no}: unsupported dataset_split {split!r}; "
                "default scene dataset must contain only train/val"
            )
            scenes_by_split[split] = scenes_by_split.get(split, 0) + 1
            actors_by_split[split] = actors_by_split.get(split, 0) + len(actors)
            for actor in actors:
                assert len(actor["target_mask"]) == 5, f"line {line_no}: target_mask"
                assert len(actor["target_offsets"]) == 5 and all(
                    len(p) == 2 for p in actor["target_offsets"]
                ), f"line {line_no}: target_offsets"
                assert _finite(actor["global"]) and _finite(actor["history"])
                assert _finite(actor["candidates"]) and _finite(actor["interactions"])
                assert _finite(actor["origin_enu"]) and _finite(actor["target_offsets"])
                counts["actor_instances"] += 1
                counts["with_route" if actor["has_route_candidate"] else "without_route"] += 1
                n = sum(actor["target_mask"])
                counts["full_future" if n == 5 else "partial_future" if n else "zero_future"] += 1
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    counts["n_scenarios"] = len(scenarios)
    for key, value in counts.items():
        assert report.get(key) == value, f"report {key}: {report.get(key)} != {value}"
    assert report.get("scenes_by_split") == scenes_by_split
    assert report.get("actors_by_split") == actors_by_split
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/predict_dataset_scene_v1")
    args = ap.parse_args(argv)
    counts = validate(args.data)
    print(f"OK: {counts['n_scene_samples']:,} scenes, {counts['actor_instances']:,} actors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
