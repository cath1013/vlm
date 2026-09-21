"""Scene-level motion-prediction dataset generator.

Unlike ``make_predict_dataset.py``, one JSONL line represents all actors that
are observable in one ``(scenario_id, t_s)`` snapshot.  Actors without map
candidates or future supervision remain as context; only a scene with no
future target at all is omitted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_llm.config import PipelineConfig
from traffic_llm.da_runner import DeepAccidentRunner
from traffic_llm.predict_model import encode
from traffic_llm.prediction import build_context

import make_predict_dataset as actor_data

SUPPORTED_SPLITS = ("train", "val")


def split_counts(scenarios):
    out = {}
    for scenario in scenarios:
        out[scenario.split] = out.get(scenario.split, 0) + 1
    return out


def select_source_scenarios(scenarios, requested_split=None):
    """Default experiment pool is official train+val, never mini/unknown."""
    source = split_counts(scenarios)
    if requested_split is not None:
        if requested_split not in SUPPORTED_SPLITS:
            raise SystemExit(f"--split 은 {SUPPORTED_SPLITS} 중 하나여야 합니다: {requested_split}")
        included = [s for s in scenarios if s.split == requested_split]
    else:
        included = [s for s in scenarios if s.split in SUPPORTED_SPLITS]
    included_counts = split_counts(included)
    excluded_counts = dict(source)
    for split, count in included_counts.items():
        excluded_counts[split] -= count
        if excluded_counts[split] == 0:
            del excluded_counts[split]
    return included, source, included_counts, excluded_counts


def build_scene_records(res, cfg, rate_hz: float, *, dataset_split: str,
                        split_mismatches=None):
    """Return one record per observed snapshot, retaining every actor."""
    built_split = res.scenario.split
    if built_split != dataset_split and split_mismatches is not None:
        key = f"{dataset_split}->{built_split}"
        split_mismatches["count"] += 1
        split_mismatches["by_pair"][key] = (
            split_mismatches["by_pair"].get(key, 0) + 1
        )
        split_mismatches["details"].append({
            "scenario_id": res.scenario.scenario_id,
            "source_split": dataset_split,
            "built_split": built_split,
        })
    snaps = list(res.snapshots(rate_hz=rate_hz))
    per_actor = {}
    for snap in snaps:
        for actor in snap.actors:
            per_actor.setdefault(actor.actor_id, []).append((snap.t, actor.world_xy))

    K = int(round(cfg.prediction_horizon_s))
    scenes, counts = [], {
        "actor_instances": 0, "with_route": 0, "without_route": 0,
        "full_future": 0, "partial_future": 0, "zero_future": 0,
    }
    sid = res.scenario.scenario_id
    for snap in snaps:
        actors = []
        scene_counts = {key: 0 for key in counts}
        for actor in snap.actors:
            ctx = build_context(
                actor, res.network, cfg.prediction_horizon_s,
                t_s=snap.t, scenario_id=sid, neighbors=snap.actors,
            )
            track, complete = actor_data.future_track(
                per_actor, actor.actor_id, snap.t, cfg.prediction_horizon_s
            )
            future = track[1:K + 1]
            mask = [True] * len(future) + [False] * (K - len(future))
            e0, n0 = actor.world_xy
            offsets = [[round(p[0] - e0, 3), round(p[1] - n0, 3)] for p in future]
            offsets += [[0.0, 0.0]] * (K - len(offsets))
            enc = encode(ctx)
            has_route = bool(ctx.candidates)
            maneuver = None
            maneuver_ambiguous = True
            matched = False
            if has_route and len(track) >= 2:
                bi, err, errs = actor_data.match_candidate(track, ctx.candidates)
                _, _, maneuver_ambiguous = actor_data.tie_analysis(errs, ctx.candidates)
                maneuver = ctx.candidates[bi].maneuver
                matched = err <= actor_data.MATCH_TOLERANCE_M
            actors.append({
                "actor_id": actor.actor_id,
                "actor_class": actor.cls,
                "global": [round(v, 6) for v in enc["global"]],
                "history": [[round(v, 6) for v in row] for row in enc["history"]],
                "candidates": [[round(v, 6) for v in row] for row in enc["candidates"]],
                "has_route_candidate": has_route,
                "interactions": [[round(v, 6) for v in row] for row in enc["interactions"]],
                "origin_enu": [round(e0, 3), round(n0, 3)],
                "target_offsets": offsets,
                "target_mask": mask,
                "future_complete": bool(complete and len(future) == K),
                "maneuver": maneuver,
                "maneuver_ambiguous": maneuver_ambiguous,
                "matched": matched,
            })
            scene_counts["actor_instances"] += 1
            scene_counts["with_route" if has_route else "without_route"] += 1
            n_future = sum(mask)
            if n_future == K:
                scene_counts["full_future"] += 1
            elif n_future:
                scene_counts["partial_future"] += 1
            else:
                scene_counts["zero_future"] += 1
        if any(any(a["target_mask"]) for a in actors):
            _add_counts(counts, scene_counts)
            scenes.append({
                "scenario_id": sid,
                # ``dataset_split`` is the descriptor split that passed the
                # source inclusion filter.  BuildResult.scenario can carry a
                # different root/split and must never overwrite provenance.
                "dataset_split": dataset_split,
                "town": res.scenario.town,
                "t_s": round(snap.t, 3),
                "actors": actors,
            })
    return scenes, counts


def _add_counts(total, part):
    for key, value in part.items():
        total[key] = total.get(key, 0) + value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DeepAccident scene motion dataset generator")
    ap.add_argument("--root", required=True)
    ap.add_argument("--carla-maps", default=None)
    ap.add_argument("--allow-trajectory-map", action="store_true")
    ap.add_argument("--out", default="out/predict_dataset_scene_v1")
    ap.add_argument("--split", default=None)
    ap.add_argument("--type", default=None)
    ap.add_argument("--town", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--mode", choices=["sensor3d", "camera"], default="sensor3d")
    args = ap.parse_args(argv)

    cfg = PipelineConfig()
    cfg.deepaccident.observation_mode = args.mode
    runner = DeepAccidentRunner(args.root, cfg)
    scenarios = runner.list_scenarios(
        scenario_types=[args.type] if args.type else None,
        towns=[args.town] if args.town else None,
    )
    scenarios, source_splits, included_splits, excluded_splits = select_source_scenarios(
        scenarios, args.split
    )
    print(f"source splits: {source_splits}")
    print(f"included splits: {included_splits}")
    if excluded_splits:
        print(f"excluded splits: {excluded_splits}")
    if args.limit:
        scenarios = scenarios[:args.limit]
    if not scenarios:
        raise SystemExit("조건에 맞는 시나리오가 없습니다")
    map_provenance = actor_data.validate_map_coverage(
        scenarios, args.carla_maps, args.allow_trajectory_map
    )
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "scenes.jsonl")
    total = {"n_scenarios": 0, "n_scene_samples": 0, "n_failed": 0}
    totals = {}
    scenes_by_split, actors_by_split = {}, {}
    split_mismatches = {"count": 0, "by_pair": {}, "details": []}
    with open(out_path, "w", encoding="utf-8") as f:
        for index, scenario in enumerate(scenarios, 1):
            xodr = None
            if args.carla_maps:
                candidate = os.path.join(args.carla_maps, f"{scenario.town}.xodr")
                xodr = candidate if os.path.isfile(candidate) else None
            try:
                res = runner.build(scenario.scenario, scenario.scenario_type,
                                   opendrive_path=xodr)
                scenes, counts = build_scene_records(
                    res, cfg, args.rate, dataset_split=scenario.split,
                    split_mismatches=split_mismatches,
                )
            except Exception as exc:
                total["n_failed"] += 1
                print(f"[{index}/{len(scenarios)}] {scenario.scenario} 실패: {exc}")
                continue
            total["n_scenarios"] += 1
            total["n_scene_samples"] += len(scenes)
            _add_counts(totals, counts)
            for scene in scenes:
                f.write(json.dumps(scene, ensure_ascii=False) + "\n")
                split = scene["dataset_split"]
                scenes_by_split[split] = scenes_by_split.get(split, 0) + 1
                actors_by_split[split] = actors_by_split.get(split, 0) + len(scene["actors"])
    report = {
        **total, **totals,
        "scenes_by_split": scenes_by_split,
        "actors_by_split": actors_by_split,
        "source_splits": source_splits,
        "included_splits": included_splits,
        "excluded_splits": excluded_splits,
        "built_split_mismatches": split_mismatches,
        "config": {"prediction_horizon_s": cfg.prediction_horizon_s,
                   "snapshot_rate_hz": args.rate, "observation_mode": args.mode,
                   "root": os.path.abspath(args.root), **map_provenance},
    }
    with open(os.path.join(args.out, "dataset_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"scenes {total['n_scene_samples']:,} → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
