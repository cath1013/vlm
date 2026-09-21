"""Safely repair scene-dataset split provenance without recomputing features.

The authoritative split is the descriptor returned by ``runner.list_scenarios``.
This utility changes only ``dataset_split`` in each JSONL record after every
scenario id has been resolved uniquely to that descriptor.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import make_scene_predict_dataset as scene_data

SUPPORTED_SPLITS = {"train", "val"}


def authoritative_split_map(scenarios):
    """Return exact scenario_id -> source split, rejecting relevant ambiguity.

    DeepAccident_mini intentionally duplicates 20 official-train identifiers.
    Those descriptors were excluded by default generation, so an official
    descriptor is authoritative over its mini duplicate.  Two official
    descriptors for one id remain an unsafe ambiguity and are rejected.
    """
    grouped = {}
    for scenario in scenarios:
        grouped.setdefault(scenario.scenario_id, []).append(scenario)
    mapping = {}
    for sid, descriptors in grouped.items():
        official = [s for s in descriptors if s.split in SUPPORTED_SPLITS]
        if len(official) > 1:
            raise ValueError(f"ambiguous authoritative scenario_id: {sid}")
        if official:
            mapping[sid] = official[0].split
        elif len(descriptors) == 1:
            # Preserve this as an explicit unsupported-source failure later.
            mapping[sid] = descriptors[0].split
        else:
            raise ValueError(f"ambiguous authoritative scenario_id: {sid}")
    return mapping


def _empty_counts():
    return {
        "n_scene_samples": 0, "actor_instances": 0, "with_route": 0,
        "without_route": 0, "full_future": 0, "partial_future": 0,
        "zero_future": 0,
    }


def _add_actor_counts(counts, actors):
    counts["actor_instances"] += len(actors)
    for actor in actors:
        counts["with_route" if actor["has_route_candidate"] else "without_route"] += 1
        n_future = sum(actor["target_mask"])
        counts["full_future" if n_future == 5 else
               "partial_future" if n_future else "zero_future"] += 1


def _subtract_counts(total, part):
    result = dict(total)
    for key, value in part.items():
        result[key] = result.get(key, 0) - value
        if result[key] == 0:
            del result[key]
    return result


def repair_dataset(data_dir, scenarios, *, apply=False):
    """Preflight (or atomically apply) a split-only repair.

    Returns a report and verification summary.  No files are changed unless
    ``apply=True`` and the entire source JSONL has passed validation.
    """
    mapping = authoritative_split_map(scenarios)
    source_splits = scene_data.split_counts(scenarios)
    scenes_path = os.path.join(data_dir, "scenes.jsonl")
    report_path = os.path.join(data_dir, "dataset_report.json")
    with open(report_path, encoding="utf-8") as f:
        old_report = json.load(f)

    counts = _empty_counts()
    scenes_by_split, actors_by_split = {}, {}
    seen_scenarios, correction_pairs, corrections = set(), {}, 0
    temp_scene = None
    out_file = None
    try:
        if apply:
            fd, temp_scene = tempfile.mkstemp(
                prefix=".scenes.repaired.", suffix=".jsonl", dir=data_dir, text=True
            )
            out_file = os.fdopen(fd, "w", encoding="utf-8")
        with open(scenes_path, encoding="utf-8") as inp:
            for line_no, line in enumerate(inp, 1):
                if not line.strip():
                    continue
                scene = json.loads(line)
                sid = scene.get("scenario_id")
                if sid not in mapping:
                    raise ValueError(f"unresolved scenario_id at line {line_no}: {sid!r}")
                source_split = mapping[sid]
                if source_split not in SUPPORTED_SPLITS:
                    raise ValueError(
                        f"unsupported authoritative split at line {line_no}: "
                        f"{sid!r} -> {source_split!r}"
                    )
                old_split = scene.get("dataset_split")
                if old_split != source_split:
                    corrections += 1
                    key = f"{old_split}->{source_split}"
                    correction_pairs[key] = correction_pairs.get(key, 0) + 1
                # This is deliberately the only mutation to a scene record.
                scene["dataset_split"] = source_split
                actors = scene.get("actors") or []
                counts["n_scene_samples"] += 1
                _add_actor_counts(counts, actors)
                seen_scenarios.add(sid)
                scenes_by_split[source_split] = scenes_by_split.get(source_split, 0) + 1
                actors_by_split[source_split] = actors_by_split.get(source_split, 0) + len(actors)
                if out_file is not None:
                    out_file.write(json.dumps(scene, ensure_ascii=False) + "\n")
        if out_file is not None:
            out_file.close()
            out_file = None

        included_splits = {}
        for sid in seen_scenarios:
            split = mapping[sid]
            included_splits[split] = included_splits.get(split, 0) + 1
        excluded_splits = _subtract_counts(source_splits, included_splits)
        report = {
            "n_scenarios": len(seen_scenarios),
            **counts,
            "n_failed": old_report.get("n_failed", 0),
            "scenes_by_split": scenes_by_split,
            "actors_by_split": actors_by_split,
            "source_splits": source_splits,
            "included_splits": included_splits,
            "excluded_splits": excluded_splits,
            # A repair does not rebuild scenarios, so it cannot independently
            # recompute BuildResult provenance.  Keep any generator diagnostic
            # and record the directly observed JSONL corrections separately.
            "built_split_mismatches": old_report.get(
                "built_split_mismatches", {"count": 0, "by_pair": {}, "details": []}
            ),
            "repair_split_corrections": {
                "count": corrections, "by_pair": correction_pairs,
            },
            "config": old_report.get("config", {}),
        }
        summary = {
            "scene_count_before": counts["n_scene_samples"],
            "scene_count_after": counts["n_scene_samples"],
            "actor_count_before": counts["actor_instances"],
            "actor_count_after": counts["actor_instances"],
            "corrections": corrections,
            "correction_pairs": correction_pairs,
            "all_splits_supported": set(scenes_by_split) <= SUPPORTED_SPLITS,
        }
        if not summary["all_splits_supported"]:
            raise AssertionError("repair left unsupported scene splits")
        if apply:
            fd, temp_report = tempfile.mkstemp(
                prefix=".dataset_report.repaired.", suffix=".json", dir=data_dir, text=True
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=1)
                f.write("\n")
            # Each replacement is atomic.  Both temporary files are complete
            # before the existing JSONL is replaced.
            os.replace(temp_scene, scenes_path)
            temp_scene = None
            os.replace(temp_report, report_path)
        return report, summary
    finally:
        if out_file is not None:
            out_file.close()
        if temp_scene is not None:
            try:
                os.unlink(temp_scene)
            except FileNotFoundError:
                pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Repair scene JSONL split provenance")
    ap.add_argument("--root", required=True, help="DeepAccident root used for generation")
    ap.add_argument("--data", default="out/predict_dataset_scene_v1")
    ap.add_argument("--apply", action="store_true", help="atomically replace scenes.jsonl/report")
    args = ap.parse_args(argv)

    from traffic_llm.config import PipelineConfig
    from traffic_llm.da_runner import DeepAccidentRunner

    scenarios = DeepAccidentRunner(args.root, PipelineConfig()).list_scenarios()
    report, summary = repair_dataset(args.data, scenarios, apply=args.apply)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.apply:
        from validate_scene_predict_dataset import validate
        validate(args.data)
        print("OK: repaired dataset passed structural/provenance validation")
    else:
        print("preflight only; re-run with --apply to atomically replace the dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
