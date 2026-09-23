"""단기 경로 예측.

두 모델을 결합한다.
  - 지도 제약 예측: 현재 차선 중심선을 따라 진행, 교차로에서 분기 확률 부여
  - 등속 예측    : 도로 매칭이 실패한 경우의 폴백 (constant velocity)
"""

from __future__ import annotations

import math
from typing import List, Tuple

from .geometry import heading_to_unit
from .roadmap import RoadNetwork
from .schemas import ActorState, PredictedPath


def predict(
    actor: ActorState,
    network: RoadNetwork,
    horizon_s: float,
) -> List[PredictedPath]:
    if actor.speed_mps is None:
        # 속도 미확정(추적 이력 부족)을 '정지'로 단정하면 움직이는 차를
        # 멈춰 있다고 보고하게 된다. 판정을 보류한다.
        return [
            PredictedPath(
                maneuver="속도 미확정 — 경로 예측 보류",
                probability=1.0,
                waypoints=[actor.world_xy],
                horizon_s=horizon_s,
            )
        ]

    v = actor.speed_mps
    if v < 0.5:
        return [
            PredictedPath(
                maneuver="정지 유지",
                probability=1.0,
                waypoints=[actor.world_xy],
                horizon_s=horizon_s,
            )
        ]

    horizon_m = v * horizon_s

    if actor.placement is None:
        return [_constant_velocity(actor, horizon_s)]

    options = network.downstream_paths(actor.placement, horizon_m)
    if not options:
        return [_constant_velocity(actor, horizon_s)]

    out: List[PredictedPath] = []
    for label, prob, poly, to_road in options:
        if len(poly) < 2:
            continue
        # 경로는 **차량이 있는 곳에서** 시작해야 한다. 지도 샘플링은 s[m] 지점의
        # 도로 좌표를 쓰므로, 곡선 구간이나 도로 밖(보행자)에서는 첫 점이 실제
        # 위치와 몇 m 떨어진다. 그러면 경로가 엉뚱한 데서 출발하는 것처럼 보인다.
        if math.dist(actor.world_xy, poly[0]) > 0.5:
            poly = [actor.world_xy] + poly
        pts, truncated = _resample_by_time(poly, v, horizon_s)
        out.append(
            PredictedPath(
                maneuver=label,
                probability=prob,
                waypoints=pts,
                horizon_s=horizon_s,
                to_roads=[to_road] if to_road else [],
                truncated=truncated,
            )
        )
    return _merge_identical(out) or [_constant_velocity(actor, horizon_s)]


def _merge_identical(paths: List[PredictedPath]) -> List[PredictedPath]:
    """예측 구간 안에서 기하가 같은 경로를 하나로 합친다.

    교차로가 같은 방향으로 진출로를 여럿 열어 두면(CARLA 는 차로마다 별개
    road 로 쪼갠다) 라벨·확률·좌표가 똑같은 항목이 여러 개 나간다. 보는 쪽에는
    중복으로만 보이고, 실제로는 **같은 하나의 기동**이므로 확률을 합쳐야 한다.
    구분은 진입 가능한 도로 목록으로 남긴다.

    비교 기준은 출력 해상도(0.1m)다. 그보다 미세한 차이는 payload 에 나타나지
    않으므로 다른 경로라고 부를 근거가 없다.
    """
    merged: List[PredictedPath] = []
    index: dict = {}
    for p in paths:
        key = (p.maneuver, tuple((round(w[0], 1), round(w[1], 1)) for w in p.waypoints))
        j = index.get(key)
        if j is None:
            index[key] = len(merged)
            merged.append(p)
            continue
        first = merged[j]
        first.probability += p.probability
        for name in p.to_roads:
            if name not in first.to_roads:
                first.to_roads.append(name)
        first.truncated = first.truncated or p.truncated
    merged.sort(key=lambda p: -p.probability)
    return merged


def _constant_velocity(actor: ActorState, horizon_s: float) -> PredictedPath:
    v = actor.speed_mps or 0.0
    h = actor.heading_deg
    if h is None:
        return PredictedPath("불확실", 1.0, [actor.world_xy], horizon_s)
    fe, fn = heading_to_unit(h)
    pts = [
        (actor.world_xy[0] + fe * v * k, actor.world_xy[1] + fn * v * k)
        for k in range(int(horizon_s) + 1)
    ]
    return PredictedPath("등속 직진(지도 미매칭)", 1.0, pts, horizon_s)


def _resample_by_time(
    poly: List[Tuple[float, float]], v: float, horizon_s: float, dt: float = 1.0
) -> Tuple[List[Tuple[float, float]], bool]:
    """폴리라인을 1초 간격 웨이포인트로 재샘플링 (현재 속도 유지 가정).

    Returns: (웨이포인트, 도로망 끝에서 잘렸는가)

    폴리라인이 예측 구간보다 짧으면 **거기서 멈춘다**. 예전에는 남은 시각을
    마지막 점으로 채웠는데, 그러면 같은 좌표가 반복되어 "그 자리에 정지한다"는
    예측처럼 보인다. 실제 의미는 "이 앞은 도로망이 없어 예측할 수 없다"이므로
    짧은 목록 + truncated 플래그로 알린다.
    """
    cum = [0.0]
    for i in range(len(poly) - 1):
        cum.append(cum[-1] + math.dist(poly[i], poly[i + 1]))
    total = cum[-1]

    out: List[Tuple[float, float]] = []
    truncated = False
    k = 0.0
    while k <= horizon_s + 1e-6:
        target = v * k
        if target > total + 1e-6:
            # 도로 끝을 넘어섰다. 끝점을 한 번만 찍고 중단한다.
            truncated = True
            if not out or math.dist(out[-1], poly[-1]) > 1e-6:
                out.append(poly[-1])
            break
        # target 이 속한 세그먼트 탐색
        j = 0
        while j < len(cum) - 2 and cum[j + 1] < target:
            j += 1
        seg = cum[j + 1] - cum[j]
        r = (target - cum[j]) / seg if seg > 1e-9 else 0.0
        a, b = poly[j], poly[min(j + 1, len(poly) - 1)]
        out.append((a[0] + r * (b[0] - a[0]), a[1] + r * (b[1] - a[1])))
        k += dt
    return out, truncated
