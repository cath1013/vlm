"""iitp 액터로 **vlm 형식의 gap table** 을 만든다 — 등속판과 학습예측판 둘 다.

왜
    vlm 리포트가 자기 천장의 원인을 스스로 지목한다:

        "인지 개선은 거의 소진됐다. 0.76~0.80 의 벽은 예측 모델 —
         1초 등속 외삽 — 이지 센싱이 아니다."

    iitp 에는 그 등속보다 나은 예측기가 있다 (WaypointNet, ADE 3.01m vs 등속
    4.13m). 그런데 조건 G·H·J 는 전부 **텍스트를 더하는** 실험이었고 표 자체는
    vlm 의 등속판 그대로였다 — 지목된 병목을 건드리지 않았다.

    이 스크립트는 표를 바꾼다. 액터·좌표계·시점을 고정하고 **외삽만** 교체하므로,
    두 판의 차이는 예측기에서만 온다.

vlm 규약을 그대로 따른다 (`demo.pairwise_gaps`)
    - 회전 사각형 사이 실제 간격. 0.00m 면 차체가 닿는다
    - `(a, b, gap_now, min_gap, at_t)` 5-튜플, min_gap 오름차순
    - **둘 다 정지한 쌍은 제외** — 주차 행렬이 표를 채우는 것을 막는다

    .venv/bin/python examples/export_gaptable_for_vlm.py \\
        --root /home/sryu/inclab-nas/DeepAccident --carla-maps ./carla_map \\
        --split val --horizon 1.0 --out out/vlm_scene/gaps_val_h1.0.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines_accident_qa import _corners, _extent, _motion_fn, _poly_gap  # noqa: E402
from export_scene_for_vlm import decision_frames  # noqa: E402
from traffic_llm.carla_map import find_xodr  # noqa: E402
from traffic_llm.config import PipelineConfig  # noqa: E402
from traffic_llm.da_runner import DeepAccidentRunner  # noqa: E402

DT = 0.05
MIN_SPEED = 0.5   # m/s — 이보다 느리면 '정지'


def gap_table(snap, mode: str, horizon: float) -> List[Tuple[str, str, float, float, float]]:
    acts = [a for a in snap.actors if a.world_xy is not None]
    if len(acts) < 2:
        return []
    n = len(acts)
    ids = [a.actor_id for a in acts]
    half = [(l / 2.0, w / 2.0) for l, w in (_extent(a) for a in acts)]
    motion = [_motion_fn(a, mode) for a in acts]
    # 속력은 관측 이력에서 — 라벨 속도 컬럼은 신뢰할 수 없다는 규약을 따른다
    speed = []
    for m in motion:
        p0, p1 = m(0.0), m(0.1)
        speed.append(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / 0.1)

    steps = int(round(horizon / DT)) + 1
    polys = [[None] * steps for _ in range(n)]
    for i in range(n):
        for k in range(steps):
            x, y, u = motion[i](k * DT)
            polys[i][k] = _corners(x, y, u[0], u[1], *half[i])

    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if speed[i] <= MIN_SPEED and speed[j] <= MIN_SPEED:
                continue          # 둘 다 정지 — vlm 규약과 동일하게 제외
            now = _poly_gap(polys[i][0], polys[j][0])
            best, when = float("inf"), 0.0
            for k in range(steps):
                g = _poly_gap(polys[i][k], polys[j][k])
                if g < best:
                    best, when = g, k * DT
            out.append((ids[i], ids[j], round(now, 2), round(best, 2), round(when, 2)))
    out.sort(key=lambda r: (r[3], r[2]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--carla-maps", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--horizon", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--predictor", default="out/predict_model/waypointnet_best.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = PipelineConfig()
    from traffic_llm.predict_model import TorchPredictor
    cfg.predictor = TorchPredictor(args.predictor, mode="waypoints")
    print(f"경로 예측기: {os.path.basename(args.predictor)}", flush=True)

    runner = DeepAccidentRunner(args.root, cfg)
    scs = [s for s in runner.list_scenarios() if s.split == args.split]
    scs.sort(key=lambda s: f"{s.scenario_type}__{s.scenario}")
    if args.limit:
        scs = scs[: args.limit]
    print(f"시나리오 {len(scs)}개 · 지평 {args.horizon}s", flush=True)

    out: Dict[str, dict] = {}
    t0 = time.time()
    skipped = []
    for i, sc in enumerate(scs, 1):
        uid = f"{sc.scenario_type}__{sc.scenario}"
        try:
            want = decision_frames(sc.frames(), args.horizon)
            if not want:
                skipped.append(uid); continue
            xodr = find_xodr(sc.town, [args.carla_maps]) if args.carla_maps else None
            res = runner.build(sc.scenario, sc.scenario_type,
                               opendrive_path=xodr, frames=want)
            snaps = list(res.snapshots())
            if len(snaps) < 2:
                skipped.append(uid); continue
            last = snaps[-1]
            out[uid] = {
                "uid": uid,
                "decision_frame": want[-1],
                "n_actors": len(last.actors),
                "cv": gap_table(last, "cv", args.horizon),
                "predicted": gap_table(last, "predicted", args.horizon),
            }
        except Exception as e:
            skipped.append(f"{uid}: {type(e).__name__}: {e}")
        if i % 10 == 0 or i == len(scs):
            print(f"  [{i}/{len(scs)}] 성공 {len(out)} · 건너뜀 {len(skipped)} "
                  f"· {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 두 판이 실제로 다른지 — 같으면 실험이 성립하지 않는다
    diff = same = 0
    for v in out.values():
        a = {(r[0], r[1]): r[3] for r in v["cv"]}
        b = {(r[0], r[1]): r[3] for r in v["predicted"]}
        for k in set(a) & set(b):
            if abs(a[k] - b[k]) > 0.05: diff += 1
            else: same += 1
    print(f"\n저장 {args.out}  ({len(out)}개, 건너뜀 {len(skipped)})")
    npair = sum(len(v['cv']) for v in out.values())
    print(f"  쌍 {npair:,}개 · 두 판이 다른 쌍 {diff:,} / 같은 쌍 {same:,} "
          f"({diff/(diff+same)*100:.1f}% 변경)" if diff + same else "")
    for tag in ("cv", "predicted"):
        z = sum(1 for v in out.values() for r in v[tag] if r[3] <= 0.0)
        print(f"  {tag:10s} min_gap 0.00m 인 쌍 {z:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
