"""Collect observation-only pair-risk scenes from official DeepAccident splits.

Run separately for train and val; the latter is held out by the trainer.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traffic_llm.accident_qa import WindowConfig, build_windows, window_ground_truth
from traffic_llm.carla_map import find_xodr
from traffic_llm.collision_risk_net import ground_truth_bucket, pair_labels
from traffic_llm.config import PipelineConfig
from traffic_llm.da_runner import DeepAccidentRunner
from traffic_llm.deepaccident import estimate_collision
from traffic_llm.predict_model import encode
from traffic_llm.prediction import build_context


def window_supervision_status(gt):
    """Return (GT bucket, keep, incomplete-negative-horizon drop)."""
    expected = gt["expected"]
    bucket = ground_truth_bucket(gt)
    complete = len(expected) == 5 and all(x["scorable"] for x in expected)
    keep = len(expected) == 5 and (bucket > 0 or complete)
    dropped_incomplete = len(expected) == 5 and bucket == 0 and not complete
    return bucket, keep, dropped_incomplete


def horizon_supervision_status(gt, horizon):
    """Return (keep, positive, negative, dropped) for a 1..5 second horizon."""
    if not 1 <= horizon <= 5:
        raise ValueError("horizon must be 1..5 seconds")
    expected = gt["expected"][:horizon]
    positive = any(row["accident_expected"] for row in expected)
    negative = (len(expected) == horizon and all(
        row["scorable"] and not row["accident_expected"] for row in expected))
    keep = positive or negative
    return keep, positive, negative, not keep


def empty_horizon_census():
    return {str(h): {"total_build_windows": 0, "kept_windows": 0,
                     "positive_windows": 0, "negative_windows": 0,
                     "dropped_incomplete_future": 0,
                     "scenarios_with_at_least_one_kept_window": 0,
                     "scenarios_with_at_least_one_positive_window": 0,
                     "scenarios_with_at_least_one_negative_window": 0,
                     "retention_fraction": None}
            for h in range(1, 6)}


def add_window_to_census(census, gt):
    for horizon in range(1, 6):
        row = census[str(horizon)]
        keep, positive, negative, dropped = horizon_supervision_status(gt, horizon)
        row["total_build_windows"] += 1
        row["kept_windows"] += int(keep)
        row["positive_windows"] += int(positive)
        row["negative_windows"] += int(negative)
        row["dropped_incomplete_future"] += int(dropped)


def finalize_horizon_census(census):
    for row in census.values():
        total = row["total_build_windows"]
        row["retention_fraction"] = row["kept_windows"] / total if total else None
    return census


def add_scenario_census(total, scenario):
    for horizon in range(1, 6):
        target, source = total[str(horizon)], scenario[str(horizon)]
        for key in ("total_build_windows", "kept_windows", "positive_windows",
                    "negative_windows", "dropped_incomplete_future"):
            target[key] += source[key]
        for key, count in (("kept", "kept_windows"), ("positive", "positive_windows"),
                           ("negative", "negative_windows")):
            target[f"scenarios_with_at_least_one_{key}_window"] += int(source[count] > 0)


def add_scenario_diagnostics(counts, diagnostics):
    built = diagnostics["windows_from_build_windows"]
    kept = diagnostics["windows_kept"]
    counts["total_build_windows"] += built
    counts["total_kept_windows"] += kept
    counts["dropped_incomplete_observation"] += diagnostics["dropped_incomplete_observation"]
    counts["dropped_after_collision"] += diagnostics["dropped_after_collision"]
    counts["dropped_incomplete_future_5s"] += diagnostics["dropped_incomplete_future_5s"]
    counts["scenarios_with_zero_build_windows"] += int(built == 0)
    counts["scenarios_with_build_windows_but_zero_fully_supervised_5s_windows"] += int(
        built > 0 and diagnostics["fully_supervised_5s_windows"] == 0)
    counts["scenarios_with_kept_windows"] += int(kept > 0)


def collect_scenario(runner, scenario, cfg, wcfg, maps, rate, census_only=False):
    result = runner.build(scenario.scenario, scenario.scenario_type,
                          opendrive_path=find_xodr(scenario.town, [maps]))
    # The converter's default rule predictor constructs actor contexts. Its
    # paths are irrelevant to window GT, so skip that work for a census.
    prediction_context = (patch("traffic_llm.pipeline.predict", return_value=[])
                          if census_only else nullcontext())
    with prediction_context:
        snapshots = list(result.snapshots(rate_hz=rate))
    collision = estimate_collision(scenario, cfg.deepaccident)
    collision_time = collision.time_s if collision and collision.occurred else None
    windows, build_info = build_windows(snapshots, wcfg, collision_time_s=collision_time)
    diagnostics = {
        "scenario_id": scenario.scenario_id,
        "windows_from_build_windows": len(windows),
        "windows_kept": 0,
        "fully_supervised_5s_windows": 0,
        "dropped_incomplete_observation": build_info["dropped_incomplete"],
        "dropped_after_collision": build_info["dropped_after_collision"],
        "dropped_incomplete_future_5s": 0,
        "collision_time_s": collision_time,
        "data_start_s": snapshots[0].t if snapshots else None,
        "data_end_s": snapshots[-1].t if snapshots else None,
        "horizon_census": empty_horizon_census(),
    }
    agent_ids = {agent: scenario.meta.agent_id_of(agent) for agent in scenario.agents}
    rows = []
    for window in windows:
        gt = window_ground_truth(
            window, wcfg, collision=collision, agent_carla_ids=agent_ids,
            scenario_id=scenario.scenario_id, scenario_split=scenario.scenario_type,
            dataset_split=scenario.split, town=scenario.town,
            data_end_s=snapshots[-1].t,
        )
        add_window_to_census(diagnostics["horizon_census"], gt)
        # Negative windows need a fully observed horizon. A known collision
        # bucket remains supervised even when recording ends before +5 s.
        gt_bucket, keep, dropped_incomplete = window_supervision_status(gt)
        positive = gt_bucket > 0
        diagnostics["fully_supervised_5s_windows"] += int(
            len(gt["expected"]) == 5 and all(x["scorable"] for x in gt["expected"]))
        diagnostics["dropped_incomplete_future_5s"] += int(dropped_incomplete)
        if not keep:
            continue
        diagnostics["windows_kept"] += 1
        if census_only:
            continue
        actors = sorted(window.last.actors, key=lambda a: a.actor_id)
        encoded = []
        for actor in actors:
            context = build_context(actor, result.network, 5.0, t_s=window.t_end,
                                    scenario_id=scenario.scenario_id, neighbors=actors)
            item = encode(context)
            encoded.append({"actor_id": actor.actor_id, "global": item["global"],
                            "history": item["history"], "candidates": item["candidates"],
                            "interactions": item["interactions"],
                            "origin_enu": list(actor.world_xy)})
        labels = pair_labels([a["actor_id"] for a in encoded], gt)
        rows.append({"scenario_id": scenario.scenario_id,
                     "window_id": f"{scenario.scenario_id}:{window.index}:{window.t_end:.3f}",
                     "dataset_split": scenario.split, "t_end_s": window.t_end,
                     "gt_bucket": gt_bucket,
                     "gt_positive": positive, "gt_pair_present": any(x > 0 for x in labels),
                     "actors": encoded, "pair_labels": labels})
    finalize_horizon_census(diagnostics["horizon_census"])
    return rows, diagnostics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--carla-maps", required=True)
    ap.add_argument("--split", choices=("train", "val"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0, help="scenario limit for smoke runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--census-only", action="store_true",
                    help="write horizon-supervision JSON without actor features or pair JSONL")
    args = ap.parse_args(argv)
    if args.rate <= 0:
        ap.error("--rate must be positive")
    cfg = PipelineConfig()
    cfg.deepaccident.observation_mode = "sensor3d"
    runner = DeepAccidentRunner(args.root, cfg)
    scenarios = sorted((s for s in runner.list_scenarios() if s.split == args.split),
                       key=lambda s: s.scenario_id)
    available_scenarios = len(scenarios)
    if args.limit:
        random.Random(args.seed).shuffle(scenarios)
        scenarios = scenarios[:args.limit]
    wcfg = WindowConfig(window_s=5.0, stride_s=1.0, horizon_s=5.0,
                        snapshot_rate_hz=args.rate, warmup=False, full_window=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"scenarios": 0, "windows": 0, "pairs": 0,
              "gt_positive_windows": 0, "gt_pair_present_windows": 0,
              "scenarios_with_zero_build_windows": 0,
              "scenarios_with_build_windows_but_zero_fully_supervised_5s_windows": 0,
              "scenarios_with_kept_windows": 0,
              "total_build_windows": 0, "total_kept_windows": 0,
              "dropped_incomplete_observation": 0, "dropped_after_collision": 0,
              "dropped_incomplete_future_5s": 0, "failed": []}
    scenario_diagnostics = []
    horizon_census = empty_horizon_census()
    output_context = nullcontext(None) if args.census_only else out.open("w", encoding="utf-8")
    with output_context as file:
        for scenario in scenarios:
            try:
                rows, diagnostics = collect_scenario(
                    runner, scenario, cfg, wcfg, args.carla_maps, args.rate,
                    census_only=args.census_only)
            except Exception as exc:
                counts["failed"].append({"scenario_id": scenario.scenario_id,
                                         "error": f"{type(exc).__name__}: {exc}"})
                continue
            counts["scenarios"] += 1
            scenario_diagnostics.append(diagnostics)
            add_scenario_diagnostics(counts, diagnostics)
            add_scenario_census(horizon_census, diagnostics["horizon_census"])
            for row in rows:
                if file is not None:
                    file.write(json.dumps(row, separators=(",", ":")) + "\n")
                counts["windows"] += 1
                counts["pairs"] += len(row["pair_labels"])
                counts["gt_positive_windows"] += int(row["gt_positive"])
                counts["gt_pair_present_windows"] += int(row["gt_pair_present"])
            print(json.dumps(diagnostics), flush=True)
    counts["visible_pair_coverage_ceiling"] = (
        counts["gt_pair_present_windows"] / counts["gt_positive_windows"]
        if counts["gt_positive_windows"] else None)
    report = {"source_split": args.split, "observation_mode": "sensor3d",
              "window_s": 5, "stride_s": 1, "horizon_s": 5,
              "rate_hz": args.rate, "limit": args.limit,
              "available_scenarios": available_scenarios,
              "selected_scenarios": len(scenarios), "census_only": args.census_only,
              "counts": counts,
              "horizon_census": finalize_horizon_census(horizon_census),
              "scenario_diagnostics": scenario_diagnostics}
    report_path = out if args.census_only else out.with_suffix(".manifest.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "horizon_census": report["horizon_census"]}, indent=2))


if __name__ == "__main__":
    main()
