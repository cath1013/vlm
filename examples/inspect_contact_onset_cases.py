"""Inspect manually selected CONTACT_ONSET cases with exact raw CARLA boxes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples import audit_gt_carla_boxes as raw_audit  # noqa: E402
from traffic_llm.deepaccident import scan_scenarios  # noqa: E402


def _time_s(frame: int) -> float:
    return (frame - 1) / raw_audit.FRAME_RATE_HZ


def _load_cases(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list):
        raise ValueError("--cases must contain a JSON array or an object with a 'cases' array")
    for case in cases:
        required = {"scenario_type", "scenario", "pair", "frames", "note"}
        missing = required - set(case)
        if missing:
            raise ValueError(f"case is missing fields: {sorted(missing)}")
        if len(case["pair"]) != 2 or case["pair"][0] == case["pair"][1]:
            raise ValueError(f"case pair must contain two distinct CARLA IDs: {case['pair']}")
    return cases


def _pair_frame_diagnostic(frame: int, boxes, pair):
    actors = [boxes.get(carla_id) for carla_id in pair]
    row = {"frame": frame, "time_s": _time_s(frame), "actor_missing": any(actor is None for actor in actors)}
    if row["actor_missing"]:
        row["actors"] = [None if actor is None else raw_audit._box_diagnostic(actor) for actor in actors]
        return row
    row.update(raw_audit._pair_diagnostic(*actors))
    return row


def _plot_frame(frame: int, boxes, pair, radius_m: float, path: Path):
    """Draw only existing audit polygons; this function adds no geometry logic."""
    import matplotlib.pyplot as plt

    actors = [boxes.get(carla_id) for carla_id in pair]
    missing = any(actor is None for actor in actors)
    state = "MISSING" if missing else ("CONTACT" if raw_audit.boxes_contact(*actors) else "SEPARATED")
    fig, ax = plt.subplots(figsize=(8, 8))
    present = [actor for actor in actors if actor is not None]
    if present:
        center = (sum(actor.center[0] for actor in present) / len(present),
                  sum(actor.center[1] for actor in present) / len(present))
        colors = ("tab:red", "tab:blue")
        for actor, color in zip(actors, colors):
            if actor is not None:
                raw_audit._draw_box(ax, actor, color=color, emphasized=True)
        ax.set_xlim(center[0] - radius_m, center[0] + radius_m)
        ax.set_ylim(center[1] - radius_m, center[1] + radius_m)
        sx, sy = center[0] - radius_m + 1.0, center[1] - radius_m + 1.0
        ax.plot((sx, sx + 1.0), (sy, sy), color="black", linewidth=3)
        ax.text(sx + 0.5, sy + radius_m * 0.05, "1 m", ha="center", va="bottom", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Both requested actors are missing", transform=ax.transAxes,
                ha="center", va="center")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"Raw CARLA boxes — frame {frame:04d} ({state})")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _contact_episodes(frame_rows):
    """Contiguous raw-frame contact episodes; missing frames break an episode."""
    episodes, active = [], []
    previous_frame = None
    for row in frame_rows:
        contact = not row["actor_missing"] and row["audit_3d_contact"]
        if contact and previous_frame is not None and row["frame"] == previous_frame + 1:
            active.append(row)
        elif contact:
            if active:
                episodes.append(active)
            active = [row]
        elif active:
            episodes.append(active)
            active = []
        previous_frame = row["frame"]
    if active:
        episodes.append(active)
    return episodes


def _episode_summary(all_rows, requested_frames):
    requested = {row["frame"] for row in all_rows if row["frame"] in requested_frames}
    episodes = [episode for episode in _contact_episodes(all_rows)
                if any(row["frame"] in requested for row in episode)]
    contact_frames = [row["frame"] for row in all_rows
                      if row["frame"] in requested and not row["actor_missing"]
                      and row["audit_3d_contact"]]
    missing_frames = [row["frame"] for row in all_rows
                      if row["frame"] in requested and row["actor_missing"]]
    summary = {
        "contact_frames_among_requested_frames": contact_frames,
        "missing_frames": missing_frames,
        "number_of_consecutive_contact_frames": None,
        "first_contact_frame": None,
        "first_contact_time_s": None,
        "last_contact_frame": None,
        "last_contact_time_s": None,
        "contact_duration_raw_10hz_frames": None,
        "gap_immediately_before_first_contact_m": None,
        "gap_immediately_after_last_contact_m": None,
    }
    if not episodes:
        return summary
    episode = episodes[0]
    first, last = episode[0], episode[-1]
    summary.update({
        "number_of_consecutive_contact_frames": len(episode),
        "first_contact_frame": first["frame"], "first_contact_time_s": first["time_s"],
        "last_contact_frame": last["frame"], "last_contact_time_s": last["time_s"],
        "contact_duration_raw_10hz_frames": len(episode),
    })
    by_frame = {row["frame"]: row for row in all_rows}
    before, after = by_frame.get(first["frame"] - 1), by_frame.get(last["frame"] + 1)
    if before is not None and not before["actor_missing"]:
        summary["gap_immediately_before_first_contact_m"] = before["minimum_bev_gap_m"]
    if after is not None and not after["actor_missing"]:
        summary["gap_immediately_after_last_contact_m"] = after["minimum_bev_gap_m"]
    return summary


def inspect_case(root: str, case: dict, out: Path, radius_m: float):
    scenario_type, scenario_name = case["scenario_type"], case["scenario"]
    # scan_scenarios reads metadata to locate the requested scenario; raw labels
    # and boxes are loaded below only for this selected scenario.
    matches = [scenario for scenario in scan_scenarios(root, scenario_types=[scenario_type])
               if scenario.scenario == scenario_name]
    if len(matches) != 1:
        raise KeyError(f"expected one scenario {scenario_type}/{scenario_name}, found {len(matches)}")
    scenario = matches[0]
    pair = tuple(map(int, case["pair"]))
    frames = sorted(set(map(int, case["frames"])))
    all_frames = sorted({frame for series in scenario.agents.values() for frame in series.frames})
    boxes_by_frame = {frame: raw_audit.merge_frame_boxes(scenario, frame)[0] for frame in all_frames}
    all_rows = [_pair_frame_diagnostic(frame, boxes_by_frame[frame], pair) for frame in all_frames]

    case_dir = out / f"{scenario_type}__{scenario_name}__{pair[0]}_{pair[1]}"
    case_dir.mkdir(parents=True, exist_ok=True)
    requested_rows = []
    for frame in frames:
        if frame not in boxes_by_frame:
            row = {"frame": frame, "time_s": _time_s(frame), "actor_missing": True, "actors": [None, None]}
            requested_rows.append(row)
            _plot_frame(frame, {}, pair, radius_m, case_dir / f"frame_{frame:04d}.png")
            continue
        row = _pair_frame_diagnostic(frame, boxes_by_frame[frame], pair)
        requested_rows.append(row)
        _plot_frame(frame, boxes_by_frame[frame], pair, radius_m, case_dir / f"frame_{frame:04d}.png")
    (case_dir / "pair_diagnostics.json").write_text(
        json.dumps({"scenario_type": scenario_type, "scenario": scenario_name, "pair": list(pair),
                    "note": case["note"], "frames": requested_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    actor_classes = {
        str(carla_id): next((boxes_by_frame[frame][carla_id].cls for frame in frames
                             if carla_id in boxes_by_frame.get(frame, {})), None)
        for carla_id in pair
    }
    episode = _episode_summary(all_rows, set(frames))
    episode["missing_frames"] = [row["frame"] for row in requested_rows if row["actor_missing"]]
    summary = {"scenario_type": scenario_type, "scenario": scenario_name, "pair": list(pair),
               "actor_classes": actor_classes, "requested_frames": frames,
               **episode}
    (case_dir / "case_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return case_dir


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="DeepAccident root")
    ap.add_argument("--cases", required=True, type=Path, help="JSON case list")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--plot-radius-m", type=float, default=12.0)
    args = ap.parse_args(argv)
    if args.plot_radius_m <= 0:
        ap.error("--plot-radius-m must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    for case in _load_cases(args.cases):
        result = inspect_case(args.root, case, args.out, args.plot_radius_m)
        print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
