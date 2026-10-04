"""Collect top-k dynamic actor-pair features from DeepAccident scenarios."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from traffic_llm.accident_qa import (  # noqa: E402
    WindowConfig, build_windows, window_ground_truth,
)
from traffic_llm.carla_map import find_xodr  # noqa: E402
from traffic_llm.config import PipelineConfig  # noqa: E402
from traffic_llm.da_runner import DeepAccidentRunner  # noqa: E402
from traffic_llm.deepaccident import estimate_collision  # noqa: E402
from traffic_llm.pair_reranker_v2 import (  # noqa: E402
    FEATURE_NAMES_V2, FEATURE_NAMES_V3, FEATURE_NAMES_V4,
    pair_features_v2, pair_features_v3, pair_features_v4,
)
from traffic_llm.predict_model import JointSceneTorchPredictor, TorchPredictor  # noqa: E402
from traffic_llm.serialize import rank_actors  # noqa: E402
from traffic_llm.swept_path import swept_pair_clearances  # noqa: E402


def _groups(gt, target_interval=None):
    positive = next(
        (row for row in gt.get("expected", []) if row.get("accident_expected")
         and (target_interval is None or row.get("k") == target_interval)),
        None,
    )
    if not positive:
        return []
    return [set(v.get("actor_ids") or [])
            for v in positive.get("involved_vehicles") or []]


def _matches(a, b, groups):
    return any(
        (a in groups[i] and b in groups[j]) or
        (b in groups[i] and a in groups[j])
        for i in range(len(groups)) for j in range(i + 1, len(groups))
    )


def _positive_intervals(gt):
    """Intervals containing the labelled collision, if any."""
    return {int(row["k"]) for row in gt.get("expected", [])
            if row.get("accident_expected")}


def _scenario_filter(path):
    if not path:
        return None
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(row["scenario"], row["scenario_type"]) for row in doc["scenarios"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--carla-maps", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--scenario-list", default=None,
                        help="optional validation batch_manifest.json")
    parser.add_argument("--predictor", default="out/predict_model/waypointnet_best.pt")
    parser.add_argument("--predictor-mode", choices=("waypoints", "joint_scene"),
                        default="waypoints")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--candidate-pool-cap", "--pair-cap", dest="candidate_pool_cap",
        type=int, default=50,
        help="clearance-ranked pairs retained for re-ranking (default: 50); "
             "--pair-cap is a backward-compatible alias",
    )
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for deterministic subset selection")
    parser.add_argument("--feature-version", type=int, choices=(2, 3, 4), default=2,
                        help="V3 uses risk features and time-aligned pair labels")
    parser.add_argument("--target-interval", type=int, choices=range(1, 6),
                        default=None, help="restrict labels and candidates to one 1-second bucket")
    args = parser.parse_args(argv)
    feature_names = {2: FEATURE_NAMES_V2, 3: FEATURE_NAMES_V3,
                     4: FEATURE_NAMES_V4}[args.feature_version]
    feature_fn = {2: pair_features_v2, 3: pair_features_v3,
                  4: pair_features_v4}[args.feature_version]

    cfg = PipelineConfig()
    cfg.deepaccident.observation_mode = "sensor3d"
    cfg.predictor = (JointSceneTorchPredictor(args.predictor, device=args.device)
                     if args.predictor_mode == "joint_scene" else
                     TorchPredictor(args.predictor, mode="waypoints", device=args.device))
    runner = DeepAccidentRunner(args.root, cfg)
    allowed = _scenario_filter(args.scenario_list)
    scenarios = [s for s in runner.list_scenarios()
                 if s.split == args.split and (
                     allowed is None or (s.scenario, s.scenario_type) in allowed
                 )]
    scenarios.sort(key=lambda s: s.scenario_id)
    if args.limit and args.limit < len(scenarios):
        random.Random(args.seed).shuffle(scenarios)
        scenarios = scenarios[:args.limit]
    if allowed is not None and len(scenarios) != len(allowed):
        missing = sorted(allowed - {
            (s.scenario, s.scenario_type) for s in scenarios
        })
        raise SystemExit(f"scenario-list mismatch; missing={missing}")

    # Pair training must use the same fixed-length observation windows as the
    # intended end-to-end comparison.  Expanding-prefix windows would change
    # the task and invalidate the learned ranking threshold.
    wcfg = WindowConfig(window_s=5.0, stride_s=1.0, horizon_s=5.0,
                        snapshot_rate_hz=args.rate, warmup=False, full_window=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"scenarios": 0, "windows": 0, "pairs": 0, "positive_pairs": 0,
              "positive_windows": 0, "positive_candidate_windows": 0,
              "failed": 0}
    failures = []
    with out.open("w", encoding="utf-8") as handle:
        for index, scenario in enumerate(scenarios, 1):
            try:
                xodr = find_xodr(scenario.town, [args.carla_maps])
                result = runner.build(
                    scenario.scenario, scenario.scenario_type, opendrive_path=xodr
                )
                snapshots = list(result.snapshots(rate_hz=args.rate))
                collision = estimate_collision(scenario, cfg.deepaccident)
                collision_time = (
                    collision.time_s if collision and collision.occurred else None
                )
                windows, _ = build_windows(
                    snapshots, wcfg, collision_time_s=collision_time
                )
                agent_ids = {
                    agent: scenario.meta.agent_id_of(agent)
                    for agent in scenario.agents
                }
                for window in windows:
                    gt = window_ground_truth(
                        window, wcfg, collision=collision,
                        agent_carla_ids=agent_ids,
                        scenario_id=scenario.scenario_id,
                        scenario_split=scenario.scenario_type,
                        dataset_split=scenario.split, town=scenario.town,
                        data_end_s=snapshots[-1].t,
                    )
                    if not gt.get("expected"):
                        continue
                    if args.target_interval is not None and not any(
                        row.get("k") == args.target_interval and row.get("scorable")
                        for row in gt["expected"]
                    ):
                        continue
                    groups = _groups(gt, args.target_interval)
                    positive_intervals = _positive_intervals(gt)
                    actors, _ = rank_actors(
                        window.last, wcfg.actor_cap(len(window.last.actors))
                    )
                    actor_by_id = {a.actor_id: a for a in actors}
                    pairs = swept_pair_clearances(
                        actors, horizon_s=wcfg.horizon_s,
                        sample_dt_s=wcfg.swept_sample_dt_s,
                        contact_margin_m=wcfg.swept_contact_margin_m,
                        exclude_touching_now=wcfg.swept_exclude_touching_now,
                        exclude_static_pairs=wcfg.swept_exclude_static_pairs,
                    )
                    if args.target_interval is not None:
                        pairs = [p for p in pairs if p.interval_index == args.target_interval]
                    pairs = pairs[:args.candidate_pool_cap]
                    candidates = []
                    for pair in pairs:
                        label = _matches(pair.actor_a, pair.actor_b, groups)
                        # V2's pair identity target is useful for exposure
                        # ranking.  V3 must additionally learn *when* that
                        # pair is risky; otherwise every projected contact of
                        # an eventual collision pair becomes a positive.
                        if args.feature_version in (3, 4) or args.target_interval is not None:
                            label = label and pair.interval_index in positive_intervals
                        candidates.append({
                            "actor_a": pair.actor_a, "actor_b": pair.actor_b,
                            "label": label,
                            "minimum_clearance_m": round(pair.minimum_clearance_m, 6),
                            "predicted_contact": bool(pair.predicted_contact),
                            "interval_index": pair.interval_index,
                            "features": feature_fn(
                                actor_by_id[pair.actor_a], actor_by_id[pair.actor_b],
                                pair, wcfg.horizon_s,
                                *([window.last.interactions] if args.feature_version == 4 else []),
                            ),
                        })
                        counts["positive_pairs"] += int(label)
                    actual = bool(groups)
                    counts["windows"] += 1
                    counts["pairs"] += len(candidates)
                    counts["positive_windows"] += int(actual)
                    counts["positive_candidate_windows"] += int(
                        actual and any(row["label"] for row in candidates)
                    )
                    handle.write(json.dumps({
                        "scenario": scenario.scenario_id,
                        "window": window.label,
                        "actual": actual,
                        "positive_intervals": sorted(positive_intervals),
                        "target_interval": args.target_interval,
                        "candidates": candidates,
                    }, separators=(",", ":")) + "\n")
                    handle.flush()
                counts["scenarios"] += 1
            except Exception as exc:
                counts["failed"] += 1
                failures.append({"scenario": scenario.scenario_id,
                                 "error": f"{type(exc).__name__}: {exc}"})
            print(
                f"[{index:03d}/{len(scenarios):03d}] {scenario.scenario} "
                f"windows={counts['windows']} pairs={counts['pairs']} "
                f"failed={counts['failed']}", flush=True,
            )

    manifest = {
        "dataset_root": os.path.abspath(args.root), "split": args.split,
        "predictor": os.path.abspath(args.predictor),
        "predictor_mode": args.predictor_mode,
        "target_interval": args.target_interval,
        "candidate_pool_cap": args.candidate_pool_cap,
        "payload_pair_cap": wcfg.swept_pair_cap,
        "selection": {"limit": args.limit, "seed": args.seed},
        "feature_version": args.feature_version,
        "feature_names": list(feature_names), "counts": counts,
        "failures": failures,
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
