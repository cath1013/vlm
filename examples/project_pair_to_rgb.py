"""Project two raw DeepAccident CARLA boxes onto their original RGB frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from traffic_llm.config import DeepAccidentConfig  # noqa: E402
from traffic_llm.deepaccident import (  # noqa: E402
    _box_corners_lidar, load_calib, parse_label_file, resolve_cameras, scan_scenarios,
)
from examples import audit_gt_carla_boxes as raw_audit  # noqa: E402


# These connect the corner order returned by the existing _box_corners_lidar.
BOX_EDGES = ((0, 2), (2, 6), (6, 4), (4, 0), (1, 3), (3, 7), (7, 5), (5, 1),
             (0, 1), (2, 3), (4, 5), (6, 7))


def _project_box(obj, calib, camera: str, image_size):
    """Use DeepAccident's existing raw corners and camera calibration convention."""
    width, height = image_size
    corners = _box_corners_lidar(obj)
    camera_corners = np.asarray(calib[f"lidar_to_{camera}"], dtype=float) @ corners
    depth = camera_corners[0, :]
    valid = depth > 0.5  # Same near-plane convention as DeepAccidentPerception.
    if not np.any(valid):
        return {"projected": False, "reason": "behind_camera", "uv": None, "valid": valid}
    uvw = np.asarray(calib[f"intrinsic_{camera}"], dtype=float) @ camera_corners[:3, valid]
    uv = np.full((8, 2), np.nan, dtype=float)
    uv[valid, 0] = uvw[0, :] / uvw[2, :]
    uv[valid, 1] = uvw[1, :] / uvw[2, :]
    x1, y1 = float(np.nanmin(uv[:, 0])), float(np.nanmin(uv[:, 1]))
    x2, y2 = float(np.nanmax(uv[:, 0])), float(np.nanmax(uv[:, 1]))
    if x2 < 0 or x1 > width or y2 < 0 or y1 > height:
        return {"projected": False, "reason": "outside_image", "uv": uv, "valid": valid}
    return {"projected": True, "reason": None, "uv": uv, "valid": valid,
            "bbox_xyxy": [x1, y1, x2, y2]}


def _font():
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except OSError:
        return ImageFont.load_default()


def _draw_projected_box(draw, projection, actor_id: int, color):
    uv, valid = projection["uv"], projection["valid"]
    for start, end in BOX_EDGES:
        if valid[start] and valid[end]:
            draw.line([tuple(uv[start]), tuple(uv[end])], fill=color, width=4)
    points = uv[valid]
    anchor = (max(0, float(np.nanmin(points[:, 0]))), max(0, float(np.nanmin(points[:, 1]) - 44)))
    draw.text(anchor, str(actor_id), fill=color, font=_font(), stroke_width=2, stroke_fill="black")


def _actor_for_id(label_frame, scenario, observer: str, carla_id: int):
    for obj in label_frame.objects:
        if raw_audit.resolve_carla_id(obj, scenario, observer) == carla_id:
            return obj
    return None


def _scenario(root: str, scenario_type: str, scenario_name: str, camera: str):
    cfg = DeepAccidentConfig(camera=camera)
    matches = [scenario for scenario in scan_scenarios(root, cfg=cfg, scenario_types=[scenario_type])
               if scenario.scenario == scenario_name]
    if len(matches) != 1:
        raise KeyError(f"expected one scenario {scenario_type}/{scenario_name}, found {len(matches)}")
    return matches[0]


def project_frames(root, scenario_type, scenario_name, observer, camera, pair, frames, out: Path):
    scenario = _scenario(root, scenario_type, scenario_name, camera)
    if observer not in scenario.agents:
        raise KeyError(f"observer {observer!r} is unavailable; choices: {sorted(scenario.agents)}")
    series = scenario.agents[observer]
    out.mkdir(parents=True, exist_ok=True)
    summary_frames = []
    for frame in frames:
        row = {"frame": frame, "actor_ids": list(pair), "observer": observer, "camera": camera,
               "actors": []}
        if frame not in series.frames:
            row["image_available"] = False
            row["actors"] = [{"carla_id": actor_id, "present": False, "projected_into_image": False,
                              "reason": "frame_unavailable"} for actor_id in pair]
            summary_frames.append(row)
            continue
        image_path = series.image_paths.get(frame)
        if image_path is None:
            row["image_available"] = False
            row["actors"] = [{"carla_id": actor_id, "present": False, "projected_into_image": False,
                              "reason": "camera_image_unavailable"} for actor_id in pair]
            summary_frames.append(row)
            continue
        image = Image.open(image_path).convert("RGB")  # preserve native source dimensions
        calib = load_calib(series.calib_paths[frame])
        resolve_cameras(calib, [camera])  # retain the repository's calibration validation
        labels = parse_label_file(series.label_paths[frame])
        draw = ImageDraw.Draw(image)
        for actor_id, color in zip(pair, ((255, 40, 40), (40, 130, 255))):
            obj = _actor_for_id(labels, scenario, observer, actor_id)
            actor = {"carla_id": actor_id, "present": obj is not None, "projected_into_image": False}
            if obj is None:
                actor["reason"] = "absent_from_observer_label"
            else:
                projection = _project_box(obj, calib, camera, image.size)
                actor["projected_into_image"] = projection["projected"]
                actor["reason"] = projection["reason"]
                if projection["projected"]:
                    _draw_projected_box(draw, projection, actor_id, color)
            row["actors"].append(actor)
        row["image_available"] = True
        image.save(out / f"frame_{frame:04d}.jpg", quality=95)
        summary_frames.append(row)
    (out / "projection_summary.json").write_text(
        json.dumps({"scenario_type": scenario_type, "scenario": scenario_name, "observer": observer,
                    "camera": camera, "pair": list(pair), "frames": summary_frames}, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenario-type", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--observer", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--pair", required=True, type=int, nargs=2, metavar=("ID1", "ID2"))
    ap.add_argument("--frames", required=True, type=int, nargs="+", metavar="FRAME")
    ap.add_argument("--out", required=True, type=Path)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    project_frames(args.root, args.scenario_type, args.scenario, args.observer, args.camera,
                   tuple(args.pair), args.frames, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
