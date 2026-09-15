"""Render source camera and BEV frames for one DeepAccident scenario.

Example:
  .venv/bin/python examples/render_deepaccident_montage.py \
    --root /home/sryu/inclab-nas/DeepAccident --scenario Town03_... \
    --scenario-type type1_subtype2_normal --frames 50,60,70,80 --out out.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--scenario-type", required=True)
    parser.add_argument("--agent", default="ego_vehicle")
    parser.add_argument("--frames", default="50,60,70,80",
                        help="one-based 10 Hz source frame indices")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frames = [int(x) for x in args.frames.split(",")]
    base = Path(args.root) / "val" / args.scenario_type / args.agent
    front = base / "Camera_Front" / args.scenario
    bev = base / "BEV_instance_camera" / args.scenario
    front_size, cell_w, cell_h = (480, 270), 480, 300
    canvas = Image.new("RGB", (cell_w * len(frames), cell_h * 2 + 36), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f"{args.scenario}  |  actual source frames (10 Hz)", fill="black")
    for col, frame in enumerate(frames):
        stem = f"{args.scenario}_{frame:03d}"
        rgb = Image.open(front / f"{stem}.jpg").convert("RGB")
        rgb.thumbnail(front_size)
        x = col * cell_w + (cell_w - rgb.width) // 2
        canvas.paste(rgb, (x, 36))
        draw.text((col * cell_w + 8, 36 + front_size[1] + 4),
                  f"Front  t={frame / 10:.1f}s", fill="black")
        arr = np.load(bev / f"{stem}.npz")["data"]
        bird = Image.fromarray(arr).convert("RGB")
        bird.thumbnail((270, 270))
        bx = col * cell_w + (cell_w - bird.width) // 2
        by = 36 + cell_h
        canvas.paste(bird, (bx, by))
        draw.text((col * cell_w + 8, by + 272),
                  f"BEV  t={frame / 10:.1f}s", fill="black")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(out)


if __name__ == "__main__":
    main()
