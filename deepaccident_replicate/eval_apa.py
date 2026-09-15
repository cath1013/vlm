"""저자들의 APA 프로토콜로 우리 재구현을 채점한다.

우리 채점기(`score_modes`)로 재면 이식한 규칙도 학습한 헤드도 우연을 못 넘는데
논문은 APA 69.5 를 보고한다. 두 숫자가 같은 것을 재고 있는지 가리는 실험이다.

    .venv/bin/python deepaccident_replicate/eval_apa.py \
        --cache out/da_motion/frames_val.npz \
        --model out/da_motion/da_head_best.pt

캐시에서 바로 돌린다 — 프레임 액터 배열이면 충분하고, 시나리오를 다시 읽을 필요가
없다. **미래 4프레임이 모두 녹화된 표본만** 쓴다: 저자들의 변환기가 과거·미래가 다
있는 표본만 만들기 때문이다 (`carla_converter.py`).

세 줄을 함께 낸다. 같은 예측을 다른 기준으로 재는 것이므로 차이는 전부 기준의 차이다.

    apa_faithful   공개 코드 그대로 (마지막 프레임만·FP 는 평균 표본만·빗나감은 FP)
    apa_debugged   알려진 결함 셋을 고친 판본
    apa_cv         등속 외삽에 같은 프로토콜 — 학습이 무엇을 더하는지 격리한다
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepaccident_replicate.apa import (
    ApaConfig,
    ApaCounts,
    aggregate,
    find_accident,
    score_window,
)
from deepaccident_replicate.da_motion import (
    FRAME_DT_S,
    N_FUTURE,
    N_PAST,
    BevGrid,
    rasterize,
    read_tracks,
)
from deepaccident_replicate.train_da_motion import _to_frame, load_cache


def _yaw_from_step(prev: np.ndarray, cur: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """변위 방향으로 방위를 다시 잡는다. 거의 안 움직였으면 이전 방위를 쓴다."""
    d = cur - prev
    moved = np.hypot(d[:, 0], d[:, 1]) > 0.05
    return np.where(moved, np.arctan2(d[:, 1], d[:, 0]), fallback)


def gt_accidents(frames, c: int, cfg: ApaConfig, ids: List[str]) -> List[Optional[dict]]:
    """미래 프레임별 GT 사고. **관측 시점에 있던 차량만** 본다.

    저자들도 GT 운동에 예측과 **같은 후처리**를 돌려 정답을 만든다
    (`multi_gpu_test.py:556-560`). 즉 정답은 "라벨에 적힌 충돌" 이 아니라
    "GT 인스턴스 두 개가 마지막 미래 프레임에서 닿아 있음" 이다.
    """
    out: List[Optional[dict]] = []
    known = set(ids)
    for k in range(1, N_FUTURE + 1):
        rec = frames[c + k]
        fr = _to_frame(rec)
        keep = [i for i, a in enumerate(fr.ids) if a in known]
        if len(keep) < 2:
            out.append(None)
            continue
        out.append(find_accident(
            fr.xy[keep], fr.yaw[keep], fr.size[keep],
            [fr.ids[i] for i in keep], cfg.dist_threshold_gt_m))
    return out


def pred_accidents(tracks: np.ndarray, cur, cfg: ApaConfig) -> List[Optional[dict]]:
    """예측 궤적 → 미래 프레임별 예측 사고."""
    out: List[Optional[dict]] = []
    prev = cur.xy
    for k in range(tracks.shape[0]):
        pos = tracks[k]
        yaw = _yaw_from_step(prev, pos, cur.yaw)
        out.append(find_accident(pos, yaw, cur.size, cur.ids,
                                 cfg.dist_threshold_pred_m))
        prev = pos
    return out


def cv_tracks(cur) -> np.ndarray:
    """등속 외삽 궤적 (F, N, 2). 학습이 무엇을 더하는지 재는 바닥이다.

    **ego 자기 운동을 빼야 한다.** 프레임마다 원점이 그 시점의 ego 이므로, 단순히
    `xy + v·t` 로 두면 예측은 관측 프레임 원점에, 정답은 미래 프레임 원점에 있게 되어
    10 m/s 로 2초면 20 m 가 통째로 어긋난다 (그렇게 재면 TP 가 0 이 나온다).
    학습된 모델은 흐름 라벨이 이 오프셋을 이미 흡수하고 있어 문제가 없다.
    """
    ego = 0
    for i, a in enumerate(cur.ids):
        if a == "EGO_ego_vehicle":
            ego = i
            break
    else:
        ego = int(np.argmin(np.hypot(cur.xy[:, 0], cur.xy[:, 1])))
    v_ego = cur.vel[ego]
    out = np.zeros((N_FUTURE, len(cur), 2), dtype=np.float32)
    for k in range(1, N_FUTURE + 1):
        t = k * FRAME_DT_S
        out[k - 1] = cur.xy + (cur.vel - v_ego) * t
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="프레임 캐시 npz")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--samples", type=int, default=6,
                    help="운동 표본 수. 원본은 5 + 분포평균 = 6")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    import torch

    from deepaccident_replicate.da_motion import build_net

    grid = BevGrid()
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    net = build_net(width=ck.get("width", 64))
    net.load_state_dict(ck["state_dict"])
    net.eval().to(args.device)
    print(f"모델 {args.model} (epoch {ck.get('epoch')})")

    _, frames_list, _ = load_cache(args.cache)
    # 저자들의 프로토콜: 과거·미래가 **모두 녹화된** 표본만
    index = []
    for si, frames in enumerate(frames_list):
        n = len(frames)
        for c in range(N_PAST - 1, n - N_FUTURE):
            if all(frames[k] is not None
                   for k in range(c - N_PAST + 1, c + N_FUTURE + 1)):
                index.append((si, c))
    if args.limit:
        index = index[: args.limit]
    print(f"표본 {len(index)}개 (과거 {N_PAST} + 미래 {N_FUTURE} 프레임이 모두 있는 것)")

    cfgs = {"apa_faithful": ApaConfig(), "apa_debugged": ApaConfig.debugged()}
    totals = {k: ApaCounts.zeros(len(c.tp_position_thresholds_m))
              for k, c in cfgs.items()}
    totals["apa_cv"] = ApaCounts.zeros(len(TP := ApaConfig().tp_position_thresholds_m))
    n_gt_acc = 0
    t0 = time.time()

    for n, (si, c) in enumerate(index, 1):
        frames = frames_list[si]
        cur = _to_frame(frames[c])
        past = np.stack([rasterize(_to_frame(frames[k]), grid)
                         for k in range(c - N_PAST + 1, c + 1)], axis=0)
        x = torch.from_numpy(past).unsqueeze(0).to(args.device)
        with torch.no_grad():
            flows, _, _, _, _ = net(x, n_samples=args.samples)
        flows = flows[0].cpu().numpy()

        base = ApaConfig()
        gts = gt_accidents(frames, c, base, cur.ids)
        n_gt_acc += 1 if gts[-1] is not None else 0

        preds = [pred_accidents(read_tracks(flows[s], cur, grid), cur, base)
                 for s in range(flows.shape[0])]
        for key, cfg in cfgs.items():
            totals[key].add(score_window(gts, preds, cfg))
        # 등속 외삽 — 표본 하나(결정적)라 평균 표본이자 유일한 표본이다
        totals["apa_cv"].add(
            score_window(gts, [pred_accidents(cv_tracks(cur), cur, base)],
                         ApaConfig()))

        if n % 200 == 0 or n == len(index):
            print(f"  [{n}/{len(index)}] {time.time()-t0:.0f}s", flush=True)

    print(f"\nGT 사고가 있는 창 {n_gt_acc}/{len(index)} "
          f"({n_gt_acc/max(len(index),1):.3f})")
    results = {}
    for key in ("apa_faithful", "apa_debugged", "apa_cv"):
        cfg = cfgs.get(key, ApaConfig())
        r = aggregate(totals[key], cfg)
        results[key] = r
        print(f"\n=== {key} · {r['rule']} ===")
        print(f"  APA {r['APA']}")
        print(f"{'D[m]':>5s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'APA':>7s} "
              f"{'id오차':>7s} {'위치오차':>8s} {'시간오차':>8s}")
        for p in r["per_threshold"]:
            f = lambda v: "      -" if v is None else f"{v:7.3f}"  # noqa: E731
            print(f"{p['threshold_m']:5.0f} {p['TP']:5d} {p['FP']:5d} {p['FN']:5d} "
                  f"{p['APA']:7.4f} {f(p['id_error'])} {f(p['position_error_m'])} "
                  f"{f(p['time_error_s'])}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"cache": args.cache, "model": args.model,
                       "n_windows": len(index), "n_gt_accident": n_gt_acc,
                       "samples": args.samples, "results": results},
                      fh, ensure_ascii=False, indent=1)
        print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
