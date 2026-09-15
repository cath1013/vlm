"""iitp 의 장면 브리핑을 **vlm 실험의 결정 시점에 맞춰** 내보낸다.

왜
    vlm 의 중앙 조건들(A~E)은 관측자 산문 아니면 gap table 만 받는다. 그런데
    iitp 파이프라인은 같은 데이터에서 **지도 정합된 수치**를 더 만든다 —
    도로/차선 배정, 다음 교차로까지 거리, 제한속도, 수치 가속도, 예상 경로,
    그리고 TTC·헤드웨이·도달시간차가 붙은 상호작용 목록. vlm 실험은 그걸 한 번도
    본 적이 없다.

    vlm 의 핵심 발견은 **범주형 판정은 신호를 파괴하고 숫자는 보존한다**는 것이다
    (조건 A~C 0.53~0.64 vs D·E 0.728). iitp 브리핑에는 둘이 섞여 있으므로
    두 판을 낸다.

시간 정렬
    vlm 의 결정 시점은 `마지막 프레임 − horizon/0.1` 이고 이력은 1.0초 2Hz 3프레임
    (`demo.timestamps`). 여기서는 **그 프레임들만** 지정해 창을 만든다 — 기존 창을
    재활용하면 최대 0.5초 어긋나는데, 새로 만들면 오차가 0 이다.

    .venv/bin/python examples/export_scene_for_vlm.py \\
        --root /home/sryu/inclab-nas/DeepAccident --carla-maps ./carla_map \\
        --split val --horizon 1.0 --out out/vlm_scene/val_h1.0.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_llm.accident_qa import TimeWindow, WindowConfig, render_window_text  # noqa: E402
from traffic_llm.carla_map import find_xodr  # noqa: E402
from traffic_llm.config import PipelineConfig  # noqa: E402
from traffic_llm.da_runner import DeepAccidentRunner  # noqa: E402
from traffic_llm.deepaccident import estimate_collision  # noqa: E402

STEP_FRAMES = 5   # 0.5s @ 10Hz — vlm 과 동일
N_HISTORY = 3     # 1.0s 이력 (DeepAccident receptive_field=3 과 맞춤)


def decision_frames(frames: List[int], horizon_s: float) -> Optional[List[int]]:
    """vlm 과 같은 결정 시점과 이력 프레임."""
    t = frames[-1] - int(round(horizon_s / 0.1))
    idxs = [t - i * STEP_FRAMES for i in range(N_HISTORY - 1, -1, -1)]
    if idxs[0] < frames[0]:
        return None                      # 이력이 클립 시작보다 앞선다
    return [i for i in idxs if i in set(frames)] if all(
        i in set(frames) for i in idxs) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--carla-maps", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--horizon", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--predictor", default="out/predict_model/waypointnet_best.pt")
    ap.add_argument("--no-predictor", action="store_true")
    ap.add_argument("--lang", default="en", choices=("en", "ko"),
                    help="브리핑 언어. vlm 프롬프트가 영어이므로 기본은 en — "
                         "한국어로 섞으면 조건 차이가 정보 때문인지 언어 전환 "
                         "때문인지 구분되지 않는다.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = PipelineConfig()
    cfg.serialize.language = args.lang
    if not args.no_predictor and os.path.exists(args.predictor):
        from traffic_llm.predict_model import TorchPredictor
        cfg.predictor = TorchPredictor(args.predictor, mode="waypoints")
        print(f"경로 예측기: {os.path.basename(args.predictor)}", flush=True)
    else:
        print("경로 예측기: 규칙 기반", flush=True)

    wcfg = WindowConfig(window_s=float(N_HISTORY - 1) * STEP_FRAMES * 0.1,
                        stride_s=1.0, horizon_s=args.horizon)

    runner = DeepAccidentRunner(args.root, cfg)
    scs = [s for s in runner.list_scenarios() if s.split == args.split]
    scs.sort(key=lambda s: f"{s.scenario_type}__{s.scenario}")
    if args.limit:
        scs = scs[: args.limit]
    print(f"시나리오 {len(scs)}개 · 지평 {args.horizon}s · 언어 {args.lang}",
          flush=True)

    out: Dict[str, dict] = {}
    t0 = time.time()
    skipped = []
    for i, sc in enumerate(scs, 1):
        uid = f"{sc.scenario_type}__{sc.scenario}"
        try:
            frames = sc.frames()
            want = decision_frames(frames, args.horizon)
            if not want:
                skipped.append((uid, "이력이 클립 시작보다 앞섬"))
                continue
            xodr = find_xodr(sc.town, [args.carla_maps]) if args.carla_maps else None
            res = runner.build(sc.scenario, sc.scenario_type,
                               opendrive_path=xodr, frames=want)
            snaps = list(res.snapshots())
            if len(snaps) < 2:
                skipped.append((uid, f"스냅샷 {len(snaps)}개"))
                continue
            win = TimeWindow(index=0, t_start=snaps[0].t, t_end=snaps[-1].t,
                             snapshots=snaps)
            text = render_window_text(win, cfg.serialize, wcfg)
            col = estimate_collision(sc, cfg.deepaccident)
            m = sc.meta
            out[uid] = {
                "uid": uid,
                "scenario": sc.scenario,
                "scenario_type": sc.scenario_type,
                "town": sc.town,
                "decision_frame": want[-1],
                "history_frames": want,
                "last_frame": frames[-1],
                "horizon_s": args.horizon,
                "t_end_s": snaps[-1].t,
                "n_actors": len(snaps[-1].actors),
                "text": text,
                # 채점용 — payload 에는 넣지 않는다
                "gt": {
                    "collision": bool(m.collision_occurred),
                    "relative_direction": m.spawn_relation or None,
                    "impact_zone": m.collision_position or None,
                    "involved": [c for c in (m.collision_cls_a, m.collision_cls_b) if c],
                    "collider_ids": [x for x in (m.collision_id_a, m.collision_id_b)
                                     if x is not None],
                    "intensity": m.collision_intensity,
                    "time_s": col.time_s if (col and col.occurred) else None,
                },
            }
        except Exception as e:
            skipped.append((uid, f"{type(e).__name__}: {e}"))
        if i % 10 == 0 or i == len(scs):
            print(f"  [{i}/{len(scs)}] 성공 {len(out)} · 건너뜀 {len(skipped)} "
                  f"· {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    n_col = sum(1 for v in out.values() if v["gt"]["collision"])
    sizes = sorted(len(v["text"]) for v in out.values())
    print(f"\n저장 {args.out}")
    print(f"  {len(out)}개 (충돌 {n_col} / 무충돌 {len(out) - n_col})")
    if sizes:
        print(f"  브리핑 길이: 중앙값 {sizes[len(sizes)//2]:,}자 · "
              f"최대 {sizes[-1]:,}자")
    if skipped:
        print(f"  건너뜀 {len(skipped)}개")
        for uid, why in skipped[:5]:
            print(f"    {uid}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
