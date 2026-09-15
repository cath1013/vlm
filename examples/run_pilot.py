#!/usr/bin/env python3
"""Generate and preflight a fixed, balanced DeepAccident pilot.

The pilot contains five validation accidents and five validation normal traces.
Every scenario is converted twice with the same external OpenDRIVE map and window
configuration: once with a pure constant-velocity predictor and once with the
trained WaypointNet.  This script only writes LLM request payloads and ground
truth; it never sends a provider request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_llm.accident_qa import WindowConfig, write_window_set
from traffic_llm.config import PipelineConfig
from traffic_llm.da_runner import DeepAccidentRunner
from traffic_llm.deepaccident import DeepAccidentScenario, estimate_collision
from traffic_llm.predict_model import TorchPredictor
from traffic_llm.prediction import constant_velocity_predict
from traffic_llm.serialize import scenario_ref


CONDITIONS = ("constant_velocity", "waypointnet")
SELECTION_FILE = "pilot_selection.json"
PREFLIGHT_FILE = "pilot_preflight.json"


def _stable_rank(seed: int, scenario_id: str) -> str:
    raw = f"{seed}:{scenario_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_group(
    candidates: Sequence[DeepAccidentScenario], n: int, seed: int
) -> List[DeepAccidentScenario]:
    """Deterministically spread a group across towns and scenario types."""
    remaining = list(candidates)
    chosen: List[DeepAccidentScenario] = []
    town_counts: Counter = Counter()
    type_counts: Counter = Counter()
    while remaining and len(chosen) < n:
        pick = min(
            remaining,
            key=lambda s: (
                town_counts[s.town] + type_counts[s.scenario_type],
                town_counts[s.town],
                type_counts[s.scenario_type],
                _stable_rank(seed, s.scenario_id),
                s.scenario_id,
            ),
        )
        chosen.append(pick)
        remaining.remove(pick)
        town_counts[pick.town] += 1
        type_counts[pick.scenario_type] += 1
    return chosen


def select_balanced(
    scenarios: Iterable[DeepAccidentScenario],
    n_accident: int = 5,
    n_normal: int = 5,
    seed: int = 20260830,
) -> List[DeepAccidentScenario]:
    """Choose an exact validation-only outcome balance with stable diversity."""
    val = [s for s in scenarios if s.split == "val"]
    groups = {
        outcome: [s for s in val if s.outcome == outcome]
        for outcome in ("accident", "normal")
    }
    needs = {"accident": n_accident, "normal": n_normal}
    for outcome, need in needs.items():
        if len(groups[outcome]) < need:
            raise ValueError(
                f"validation {outcome} scenarios: need {need}, found {len(groups[outcome])}"
            )
    selected: List[DeepAccidentScenario] = []
    for offset, outcome in enumerate(("accident", "normal")):
        selected.extend(_select_group(groups[outcome], needs[outcome], seed + offset))
    return selected


def scenario_record(scenario: DeepAccidentScenario) -> dict:
    return {
        "ref": scenario_ref(scenario.scenario_id),
        "id": scenario.scenario_id,
        "scenario": scenario.scenario,
        "scenario_type": scenario.scenario_type,
        "dataset_split": scenario.split,
        "outcome": scenario.outcome,
        "town": scenario.town,
    }


def selection_document(
    selected: Sequence[DeepAccidentScenario],
    seed: int,
    root: str,
    maps_dir: str = "",
    waypointnet_path: str = "",
    device: str = "cpu",
) -> dict:
    counts = Counter(s.outcome for s in selected)
    return {
        "schema_version": 1,
        "selection_method": "deterministic greedy diversity over validation split",
        "seed": seed,
        "dataset_root": os.path.abspath(root),
        "generation": {
            "observation_mode": "sensor3d",
            "map_directory": os.path.abspath(maps_dir) if maps_dir else "",
            "map_built_from_scenario_trajectories": False,
            "window_s": 5.0,
            "stride_s": 1.0,
            "horizon_s": 5.0,
            "snapshot_rate_hz": 2.0,
            "early_credit_s": 2.0,
            "llm_calls_made": 0,
            "conditions": {
                "constant_velocity": {
                    "implementation": (
                        "traffic_llm.prediction.constant_velocity_predict"
                    ),
                },
                "waypointnet": {
                    "model_path": (
                        os.path.abspath(waypointnet_path) if waypointnet_path else ""
                    ),
                    "model_sha256": (
                        _file_sha256(waypointnet_path)
                        if waypointnet_path and os.path.isfile(waypointnet_path)
                        else ""
                    ),
                    "mode": "waypoints",
                    "device": device,
                    "fallback": "constant_velocity",
                },
            },
        },
        "counts": dict(sorted(counts.items())),
        "scenarios": [scenario_record(s) for s in selected],
    }


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, value: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _scenario_dir(out_root: str, condition: str, record: Mapping[str, object]) -> str:
    return os.path.join(out_root, "conditions", condition, str(record["ref"]))


def _manifest_signature(manifest: Mapping[str, object]) -> dict:
    """Fields that must be prediction-condition invariant."""
    return {
        "scenario": manifest.get("scenario"),
        "collision": manifest.get("collision"),
        "windows": [
            {
                key: row.get(key)
                for key in (
                    "label",
                    "index",
                    "t_start_s",
                    "t_end_s",
                    "ground_truth",
                    "has_accident_in_horizon",
                    "n_snapshots",
                )
            }
            for row in manifest.get("windows", [])
        ],
    }


def preflight_pilot(out_root: str, selection: Mapping[str, object]) -> dict:
    """Validate all invariants needed before any LLM evaluation."""
    errors: List[str] = []
    warnings: List[str] = []
    scenarios = list(selection.get("scenarios", []))
    outcome_counts = Counter(row.get("outcome") for row in scenarios)
    ids = [row.get("id") for row in scenarios]

    if len(scenarios) != 10:
        errors.append(f"selection must contain 10 scenarios, found {len(scenarios)}")
    if outcome_counts != Counter({"accident": 5, "normal": 5}):
        errors.append(f"selection outcome balance is not 5/5: {dict(outcome_counts)}")
    if len(set(ids)) != len(ids):
        errors.append("selection contains duplicate scenario ids")
    if any(row.get("dataset_split") != "val" for row in scenarios):
        errors.append("selection contains a non-validation scenario")
    towns = {row.get("town") for row in scenarios}
    types = {row.get("scenario_type") for row in scenarios}
    generation = selection.get("generation", {})
    condition_provenance = generation.get("conditions", {})
    if len(towns) < 2:
        errors.append("selection does not cover multiple towns")
    if len(types) < 2:
        errors.append("selection does not cover multiple scenario types")
    if generation.get("map_built_from_scenario_trajectories") is not False:
        errors.append("pilot generation provenance does not prohibit trajectory maps")
    if generation.get("llm_calls_made") != 0:
        errors.append("pilot generation provenance does not record zero LLM calls")
    waypoint_provenance = condition_provenance.get("waypointnet", {})
    if waypoint_provenance.get("mode") != "waypoints":
        errors.append("WaypointNet provenance does not record waypoints mode")
    if waypoint_provenance.get("fallback") != "constant_velocity":
        errors.append("WaypointNet provenance does not record constant-velocity fallback")
    if not waypoint_provenance.get("model_sha256"):
        errors.append("WaypointNet provenance is missing the model SHA-256")

    manifests: Dict[str, Dict[str, dict]] = {condition: {} for condition in CONDITIONS}
    total_windows: Counter = Counter()
    total_positive: Counter = Counter()
    different_payload_pairs = 0
    total_payload_pairs = 0

    for condition in CONDITIONS:
        for record in scenarios:
            sid = str(record.get("id"))
            scenario_dir = _scenario_dir(out_root, condition, record)
            manifest_path = os.path.join(scenario_dir, "manifest.json")
            if not os.path.isfile(manifest_path):
                errors.append(f"missing manifest: {condition}/{record.get('ref')}")
                continue
            manifest = _read_json(manifest_path)
            manifests[condition][sid] = manifest
            cfg = manifest.get("config", {})
            sc = manifest.get("scenario", {})

            expected_predictor = condition
            if cfg.get("predictor") != expected_predictor:
                errors.append(
                    f"{condition}/{record.get('ref')}: predictor={cfg.get('predictor')!r}"
                )
            if not str(cfg.get("map_source", "")).startswith("OpenDRIVE ("):
                errors.append(
                    f"{condition}/{record.get('ref')}: external OpenDRIVE map not recorded"
                )
            if cfg.get("map_built_from_scenario_trajectories") is not False:
                errors.append(
                    f"{condition}/{record.get('ref')}: trajectory-derived map flag is not false"
                )
            wanted_cfg = {
                "window_s": 5.0,
                "stride_s": 1.0,
                "horizon_s": 5.0,
                "snapshot_rate_hz": 2.0,
            }
            for key, want in wanted_cfg.items():
                if cfg.get(key) != want:
                    errors.append(
                        f"{condition}/{record.get('ref')}: {key}={cfg.get(key)!r}, expected {want}"
                    )
            for key, want in (
                ("id", sid),
                ("dataset_split", "val"),
                ("outcome", record.get("outcome")),
                ("town", record.get("town")),
            ):
                if sc.get(key) != want:
                    errors.append(
                        f"{condition}/{record.get('ref')}: scenario {key} mismatch"
                    )

            collision = manifest.get("collision", {})
            should_collide = record.get("outcome") == "accident"
            if bool(collision.get("occurred")) != should_collide:
                errors.append(
                    f"{condition}/{record.get('ref')}: collision label disagrees with outcome"
                )
            collision_time = collision.get("time_s")
            windows = manifest.get("windows", [])
            if not windows:
                errors.append(f"{condition}/{record.get('ref')}: no windows generated")
            total_windows[condition] += len(windows)
            total_positive[condition] += sum(
                bool(row.get("has_accident_in_horizon")) for row in windows
            )
            for window in windows:
                if (
                    collision.get("occurred")
                    and collision_time is not None
                    and float(window.get("t_end_s", 0.0)) >= float(collision_time) - 1e-6
                ):
                    errors.append(
                        f"{condition}/{record.get('ref')}/{window.get('label')}: "
                        "window ends at or after collision"
                    )
                payload_path = os.path.join(scenario_dir, str(window.get("payload")))
                if not os.path.isfile(payload_path):
                    errors.append(
                        f"{condition}/{record.get('ref')}/{window.get('label')}: missing payload"
                    )
                gt_path = os.path.join(scenario_dir, str(window.get("ground_truth")))
                if not os.path.isfile(gt_path):
                    errors.append(
                        f"{condition}/{record.get('ref')}/{window.get('label')}: missing ground truth"
                    )
                    continue
                gt = _read_json(gt_path)
                expected = gt.get("expected", [])
                if [row.get("k") for row in expected] != [1, 2, 3, 4, 5]:
                    errors.append(
                        f"{condition}/{record.get('ref')}/{window.get('label')}: "
                        "ground truth does not contain buckets 1..5"
                    )

    paired = 0
    for record in scenarios:
        sid = str(record.get("id"))
        left = manifests[CONDITIONS[0]].get(sid)
        right = manifests[CONDITIONS[1]].get(sid)
        if left is None or right is None:
            continue
        paired += 1
        if _manifest_signature(left) != _manifest_signature(right):
            errors.append(f"{record.get('ref')}: condition manifests are not window-aligned")
            continue
        left_dir = _scenario_dir(out_root, CONDITIONS[0], record)
        right_dir = _scenario_dir(out_root, CONDITIONS[1], record)
        scenario_payload_differences = 0
        for window in left.get("windows", []):
            name = str(window.get("ground_truth"))
            if _read_json(os.path.join(left_dir, name)) != _read_json(
                os.path.join(right_dir, name)
            ):
                errors.append(f"{record.get('ref')}/{name}: ground truth differs by predictor")
            payload_name = str(window.get("payload"))
            left_payload = os.path.join(left_dir, payload_name)
            right_payload = os.path.join(right_dir, payload_name)
            if os.path.isfile(left_payload) and os.path.isfile(right_payload):
                total_payload_pairs += 1
                if _read_json(left_payload) != _read_json(right_payload):
                    different_payload_pairs += 1
                    scenario_payload_differences += 1
        if scenario_payload_differences == 0:
            errors.append(
                f"{record.get('ref')}: predictor conditions produced no distinct payload"
            )

    if total_windows["constant_velocity"] != total_windows["waypointnet"]:
        errors.append(f"condition window totals differ: {dict(total_windows)}")
    if total_positive["constant_velocity"] != total_positive["waypointnet"]:
        errors.append(f"condition positive-window totals differ: {dict(total_positive)}")

    return {
        "status": "pass" if not errors else "fail",
        "llm_calls_made": 0,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "n_selected": len(scenarios),
            "outcomes": dict(sorted(outcome_counts.items())),
            "towns": sorted(str(v) for v in towns),
            "scenario_types": sorted(str(v) for v in types),
            "paired_scenarios": paired,
            "windows_by_condition": dict(total_windows),
            "positive_windows_by_condition": dict(total_positive),
            "different_payload_pairs": different_payload_pairs,
            "total_payload_pairs": total_payload_pairs,
        },
        "checks": [
            "exactly five validation accidents and five validation normals",
            "unique scenarios with town/type diversity",
            "external OpenDRIVE map provenance only",
            "5 s observation / 1 s stride / 5 s forecast at 2 Hz",
            "no observation window ending at or after collision",
            "all ground-truth files contain forecast buckets 1..5",
            "condition windows and ground truth are exactly paired",
            "each scenario has predictor-dependent payload content",
            "WaypointNet model hash and constant-velocity fallback are recorded",
            "no LLM/provider request was made",
        ],
    }


def generate_condition(
    root: str,
    maps_dir: str,
    out_root: str,
    selected: Sequence[DeepAccidentScenario],
    condition: str,
    waypointnet_path: str,
    device: str,
) -> None:
    cfg = PipelineConfig()
    cfg.deepaccident.observation_mode = "sensor3d"
    if condition == "constant_velocity":
        cfg.predictor = constant_velocity_predict
    elif condition == "waypointnet":
        cfg.predictor = TorchPredictor(
            waypointnet_path,
            mode="waypoints",
            device=device,
            fallback=constant_velocity_predict,
        )
    else:  # pragma: no cover - internal invariant
        raise ValueError(condition)

    runner = DeepAccidentRunner(root, cfg)
    wcfg = WindowConfig(
        window_s=5.0,
        stride_s=1.0,
        horizon_s=5.0,
        snapshot_rate_hz=2.0,
        early_credit_s=2.0,
    )
    for index, scenario in enumerate(selected, 1):
        xodr = os.path.join(maps_dir, f"{scenario.town}.xodr")
        print(
            f"[{condition} {index}/{len(selected)}] {scenario.scenario_id} "
            f"({scenario.town})",
            flush=True,
        )
        result = runner.build(
            scenario.scenario,
            scenario.scenario_type,
            opendrive_path=xodr,
        )
        snapshots = list(result.snapshots(rate_hz=wcfg.snapshot_rate_hz))
        collision = estimate_collision(result.scenario, cfg.deepaccident)
        agent_ids = {
            agent: result.scenario.meta.agent_id_of(agent)
            for agent in result.scenario.agents
        }
        record = scenario_record(result.scenario)
        write_window_set(
            snapshots,
            out_dir=_scenario_dir(out_root, condition, record),
            scfg=cfg.serialize,
            cfg=wcfg,
            collision=collision,
            agent_carla_ids=agent_ids,
            scenario_id=result.scenario.scenario_id,
            scenario_split=result.scenario.scenario_type,
            dataset_split=result.scenario.split,
            town=result.scenario.town,
            also_write_text=False,
            predictor_note=condition,
            map_source=result.map_source,
            map_built_from_scenario_trajectories=False,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="DeepAccident dataset root")
    parser.add_argument("--carla-maps", required=True, help="directory containing Town*.xodr")
    parser.add_argument("--out", default="out/pilot_10", help="pilot output root")
    parser.add_argument(
        "--waypointnet",
        default="out/predict_model/waypointnet_best.pt",
        help="trained WaypointNet model",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch inference device")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate existing pilot outputs without regenerating them",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out_root = os.path.abspath(args.out)
    selection_path = os.path.join(out_root, SELECTION_FILE)

    if args.preflight_only:
        if not os.path.isfile(selection_path):
            raise SystemExit(f"selection file not found: {selection_path}")
        selection = _read_json(selection_path)
    else:
        if not os.path.isdir(args.root):
            raise SystemExit(f"dataset root not found: {args.root}")
        if not os.path.isdir(args.carla_maps):
            raise SystemExit(f"CARLA map directory not found: {args.carla_maps}")
        if not os.path.isfile(args.waypointnet):
            raise SystemExit(f"WaypointNet model not found: {args.waypointnet}")

        probe = DeepAccidentRunner(args.root, PipelineConfig())
        selected = select_balanced(probe.list_scenarios(), seed=args.seed)
        missing_maps = sorted(
            {
                scenario.town
                for scenario in selected
                if not os.path.isfile(
                    os.path.join(args.carla_maps, f"{scenario.town}.xodr")
                )
            }
        )
        if missing_maps:
            raise SystemExit(
                "external OpenDRIVE maps are required; missing: " + ", ".join(missing_maps)
            )

        selection = selection_document(
            selected,
            args.seed,
            args.root,
            maps_dir=args.carla_maps,
            waypointnet_path=args.waypointnet,
            device=args.device,
        )
        _write_json(selection_path, selection)
        print(f"Fixed selection written: {selection_path}", flush=True)
        for condition in CONDITIONS:
            generate_condition(
                args.root,
                args.carla_maps,
                out_root,
                selected,
                condition,
                args.waypointnet,
                args.device,
            )

    report = preflight_pilot(out_root, selection)
    report_path = os.path.join(out_root, PREFLIGHT_FILE)
    _write_json(report_path, report)
    print(
        f"Preflight {report['status'].upper()}: {report_path} "
        f"({len(report['errors'])} errors)",
        flush=True,
    )
    for error in report["errors"]:
        print(f"  ERROR: {error}", flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
