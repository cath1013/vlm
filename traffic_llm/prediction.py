"""단기 경로 예측.

두 모델을 결합한다.
  - 지도 제약 예측: 현재 차선 중심선을 따라 진행, 교차로에서 분기 확률 부여
  - 등속 예측    : 도로 매칭이 실패한 경우의 폴백 (constant velocity)

예측기는 **교체할 수 있다.** `predict(..., predictor=fn)` 으로 학습 모델을 끼우면
같은 입력(`PredictContext`)을 받아 같은 형태(`List[PredictedPath]`)를 돌려준다.
입력·출력 명세와 학습 데이터 생성은 `predict_model.py` 와
`docs/predict_model_io.md` 참고.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .geometry import heading_to_unit
from .predict_model import Candidate, PredictContext, Predictor
from .roadmap import RoadNetwork
from .schemas import ActorState, PredictedPath


def build_context(
    actor: ActorState,
    network: RoadNetwork,
    horizon_s: float,
    t_s: Optional[float] = None,
    scenario_id: str = "",
    neighbors: Optional[List[ActorState]] = None,
) -> PredictContext:
    """예측기에 넘길 입력을 모은다.

    후보 목록까지 여기서 만든다 — 규칙 기반과 학습 모델이 **같은 후보**를 보게
    해야 학습·추론의 입력이 어긋나지 않는다.
    """
    cands: List[Candidate] = []
    v = actor.speed_mps
    if v is not None and v >= 0.5 and actor.placement is not None:
        for label, prior, poly, to_road in network.downstream_paths(
            actor.placement, v * horizon_s
        ):
            if len(poly) >= 2:
                cands.append(Candidate(label, prior, poly, to_road))
    # 점이 1개뿐인 후보를 버렸으면 그 확률이 사라진다. 규칙 기반은 `prior` 를 그대로
    # 확률로 쓰므로 재분배하지 않으면 합이 1 미만이 된다 — 실측으로 0.032 까지
    # 떨어졌고(전체의 0.015%), 그러면 페이로드가 "직진 3%" 만 내보내 읽는 쪽이
    # "어디로도 가지 않는다"로 해석한다. 버린 쪽이 있으면 남은 것으로 정규화한다.
    total = sum(c.prior for c in cands)
    if cands and total > 0.0 and abs(total - 1.0) > 1e-9:
        for c in cands:
            c.prior /= total
    return PredictContext(
        actor=actor,
        horizon_s=horizon_s,
        candidates=cands,
        t_s=t_s,
        scenario_id=scenario_id,
        neighbors=list(neighbors or ()),
    )


def predict(
    actor: ActorState,
    network: RoadNetwork,
    horizon_s: float,
    predictor: Optional[Predictor] = None,
    neighbors: Optional[List[ActorState]] = None,
) -> List[PredictedPath]:
    """경로 예측. `predictor` 를 주면 규칙 기반 대신 그것을 쓴다."""
    ctx = build_context(actor, network, horizon_s, neighbors=neighbors)
    return (predictor or rule_predict)(ctx)


def candidate_to_path(
    ctx: PredictContext, cand: Candidate, probability: float
) -> Optional[PredictedPath]:
    """후보 + 확률 → `PredictedPath` (시간축 재샘플링 포함).

    학습 모델(rank 모드)도 이 함수를 쓴다 — 좌표 생성 규칙이 규칙 기반과 같아야
    출력을 비교할 수 있다.
    """
    poly = cand.polyline
    if len(poly) < 2:
        return None
    v = ctx.actor.speed_mps or 0.0
    # 경로는 **차량이 있는 곳에서** 시작해야 한다 (아래 predict 주석 참고)
    if math.dist(ctx.actor.world_xy, poly[0]) > 0.5:
        poly = [ctx.actor.world_xy] + poly
    pts, truncated = _resample_by_time(poly, v, ctx.horizon_s)
    return PredictedPath(
        maneuver=cand.maneuver,
        probability=probability,
        waypoints=pts,
        horizon_s=ctx.horizon_s,
        to_roads=[cand.to_road] if cand.to_road else [],
        truncated=truncated,
    )


def rule_predict(ctx: PredictContext) -> List[PredictedPath]:
    """규칙 기반 예측 (기본). 지도 제약 + 차선 기반 사전확률."""
    actor = ctx.actor
    horizon_s = ctx.horizon_s
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

    if not ctx.candidates:
        # 도로 미매칭이거나 후보를 못 만든 경우 (지평이 도로 끝을 못 넘김 등)
        return [_constant_velocity(actor, horizon_s)]

    out: List[PredictedPath] = []
    for cand in ctx.candidates:
        # 좌표 생성은 학습 모델(rank 모드)과 **같은 함수**를 쓴다 — 규칙과 모델의
        # 출력이 기하 처리에서 갈리면 둘을 비교할 수 없다.
        pth = candidate_to_path(ctx, cand, cand.prior)
        if pth is not None:
            out.append(pth)
    return merge_identical(out) or [_constant_velocity(actor, horizon_s)]


def constant_velocity_predict(ctx: PredictContext) -> List[PredictedPath]:
    """지도 후보를 보지 않는 순수 등속 직선 베이스라인.

    규칙 예측의 도로 미매칭 폴백과 달리 실험 조건으로 직접 선택할 수 있는
    ``Predictor`` 이다. 속도나 방위가 아직 확정되지 않은 첫 관측에서는 정지를
    가정하지 않고 현재 위치 한 점만 내보낸다.
    """
    actor = ctx.actor
    if actor.speed_mps is None or actor.heading_deg is None:
        return [
            PredictedPath(
                maneuver="등속 예측 보류(운동 상태 미확정)",
                probability=1.0,
                waypoints=[actor.world_xy],
                horizon_s=ctx.horizon_s,
            )
        ]
    return [
        _constant_velocity(
            actor,
            ctx.horizon_s,
            maneuver="등속 직진",
        )
    ]


def merge_identical(paths: List[PredictedPath]) -> List[PredictedPath]:
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


def _constant_velocity(
    actor: ActorState,
    horizon_s: float,
    maneuver: str = "등속 직진(지도 미매칭)",
) -> PredictedPath:
    v = actor.speed_mps or 0.0
    h = actor.heading_deg
    if h is None:
        return PredictedPath("불확실", 1.0, [actor.world_xy], horizon_s)
    fe, fn = heading_to_unit(h)
    pts = [
        (actor.world_xy[0] + fe * v * k, actor.world_xy[1] + fn * v * k)
        for k in range(int(horizon_s) + 1)
    ]
    return PredictedPath(maneuver, 1.0, pts, horizon_s)


def _resample_by_time(
    poly: List[Tuple[float, float]], v: float, horizon_s: float, dt: float = 1.0
) -> Tuple[List[Tuple[float, float]], bool]:
    """폴리라인을 1초 간격 웨이포인트로 재샘플링 (현재 속도 유지 가정).

    출력의 첫 점은 t=0 현재 위치이고, 뒤가 t=+1s, +2s, ... 이다.

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
