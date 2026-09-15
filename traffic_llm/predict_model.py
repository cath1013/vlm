"""경로 예측기 교체 지점 — 규칙 기반(기본)과 학습 모델.

`prediction.predict()` 는 기본적으로 지도 제약 규칙으로 경로를 만든다. 이 모듈은
같은 입력을 받아 같은 형태를 돌려주는 **교체 가능한 예측기**를 정의하고, PyTorch 로
저장한 모델을 그 자리에 끼울 수 있게 한다.

    from traffic_llm.predict_model import TorchPredictor
    cfg.predictor = TorchPredictor("model.pt")      # 없으면 규칙 기반

설계 규약
    - 입력은 `PredictContext` 하나로 모은다. 규칙 기반이 쓰는 것과 **정확히 같은**
      정보이며, 그것만으로 학습 데이터를 만들 수 있어야 한다 (그래야 학습과 추론의
      입력이 어긋나지 않는다).
    - 출력은 `List[PredictedPath]` 다. 하류(직렬화·BEV·상호작용)가 그대로 쓴다.
    - 모델은 **후보 순위 매기기**(rank)를 기본으로 한다. 지도에서 나온 기하를 그대로
      쓰고 확률만 학습하므로, 학습 데이터가 적어도 동작하고 노면을 벗어나지 않는다.
      좌표를 직접 회귀하려면 `mode="waypoints"` 를 쓴다.
    - torch 는 **지연 임포트**한다. 이 패키지는 torch 없이도 동작해야 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .schemas import ActorState, PredictedPath, RoadPlacement

# ---------------------------------------------------------------- 입력 계약

# 기동 라벨 → 인덱스. 모델 입출력의 순서를 고정한다.
MANEUVERS: Tuple[str, ...] = ("직진", "좌회전", "우회전", "유턴")
MANEUVER_INDEX: Dict[str, int] = {m: i for i, m in enumerate(MANEUVERS)}

# 클래스 → 인덱스 (원-핫). `fusion.CLASS_GROUP` 과 같은 집합.
CLASSES: Tuple[str, ...] = (
    "car", "van", "truck", "bus", "motorcycle", "bicycle", "person",
)
CLASS_INDEX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

N_GLOBAL_FEATURES = 22

# waypoints 모드에서 기동을 **모델이 직접 분류**할 때의 클래스.
# 마지막 칸은 "어느 후보도 따라가지 않는다" — 실제 궤적의 21% 가 그렇고(§4.1),
# 그것을 표현할 자리가 없으면 모델은 억지로 네 기동 중 하나를 골라야 한다.
OFF_CANDIDATE_LABEL = "후보 밖 경로"
MANEUVER_CLASSES: Tuple[str, ...] = MANEUVERS + (OFF_CANDIDATE_LABEL,)

# `global_features` 안의 위치. 모델이 등속 기준선을 스스로 만들 때 쓰므로 이름을
# 붙여 둔다 — 숫자를 모델 코드에 박아 두면 인코딩 순서를 바꿀 때 조용히 어긋난다.
IDX_SPEED = 0
IDX_ACCEL = 2
IDX_HEADING_SIN = 4
IDX_HEADING_COS = 5
N_CANDIDATE_FEATURES = 12

# ---- 과거 궤적 인코딩 -------------------------------------------------------
# 과거를 **고정 격자**로 리샘플링한다. 관측 주기(2Hz/10Hz)에 따라 길이가 달라지면
# 모델 입력 모양이 데이터셋마다 바뀌어 계약이 성립하지 않는다.
# 격자는 목표(`target_offsets`)와 **같은 1초 간격**이라 과거 5초 → 미래 5초가
# 대칭이 된다.
HISTORY_DT_S = 1.0
N_HISTORY_STEPS = 5           # t-5 … t-1 (현재 시각은 원점이므로 넣지 않는다)
N_HISTORY_FEATURES = 5        # de, dn, vde, vdn, valid

# InteractionWaypointNet 입력. 각 행은 target 차량 좌표계에서 본 주변 차량 한 대다.
# 마지막 valid 는 패딩과 실제 정지/0 값을 구분한다.
N_INTERACTION_FEATURES = 29
MAX_INTERACTION_NEIGHBORS = 12


@dataclass
class Candidate:
    """지도에서 나온 후보 경로 하나 (`roadmap.downstream_paths` 의 출력)."""

    maneuver: str
    prior: float  # 규칙 기반 사전확률
    polyline: List[Tuple[float, float]]  # ENU, 5m 간격
    to_road: Optional[str]


@dataclass
class PredictContext:
    """예측기가 받는 입력 전부.

    규칙 기반 `predict()` 가 쓰는 것과 **정확히 같은** 정보다. 학습 데이터도 이
    구조에서 만들므로 학습·추론의 입력이 어긋나지 않는다.
    """

    actor: ActorState
    horizon_s: float
    # 지도가 낸 후보. 비어 있으면 도로 미매칭이거나 교차로에 닿지 않은 것이다.
    candidates: List[Candidate] = field(default_factory=list)
    # 부가 맥락 (학습에 쓸 수 있게 남긴다). 예측에 필수는 아니다.
    t_s: Optional[float] = None
    scenario_id: str = ""
    # 같은 시각에 융합된 다른 교통 참여자. 미래 경로나 GT는 절대 넣지 않는다.
    neighbors: List[ActorState] = field(default_factory=list)

    # ------------------------------------------------------------ 파생값
    @property
    def placement(self) -> Optional[RoadPlacement]:
        return self.actor.placement

    @property
    def horizon_m(self) -> float:
        return (self.actor.speed_mps or 0.0) * self.horizon_s


def _sin_cos(deg: Optional[float]) -> Tuple[float, float]:
    if deg is None:
        return (0.0, 0.0)
    r = math.radians(deg)
    return (math.sin(r), math.cos(r))


def global_features(ctx: PredictContext) -> List[float]:
    """액터·도로 상태를 고정 길이 벡터로. 길이 `N_GLOBAL_FEATURES`.

    각도는 sin/cos 쌍으로 넣는다 — 0°/360° 가 붙어 있어 스칼라로 주면 모델이
    경계에서 큰 오차를 학습한다. 결측은 0 으로 채우고 **있음/없음 플래그를 따로
    둔다** (0 이 "값이 0" 인지 "모른다" 인지 구별해야 한다).
    """
    a = ctx.actor
    p = ctx.placement
    hs, hc = _sin_cos(a.heading_deg)
    out: List[float] = [
        a.speed_mps if a.speed_mps is not None else 0.0,   # 0  속도 [m/s]
        1.0 if a.speed_mps is not None else 0.0,           # 1  속도 있음
        a.accel_mps2 if a.accel_mps2 is not None else 0.0,  # 2  가속도
        1.0 if a.accel_mps2 is not None else 0.0,          # 3  가속도 있음
        hs, hc,                                            # 4,5 방위
        1.0 if a.heading_deg is not None else 0.0,         # 6  방위 있음
        ctx.horizon_s,                                     # 7  예측 지평 [s]
        a.track_age_s,                                     # 8  추적 나이 [s]
        a.confidence,                                      # 9  존재 확신도
        a.position_quality,                                # 10 위치 정확도
    ]
    # 11~17: 클래스 원-핫
    onehot = [0.0] * len(CLASSES)
    idx = CLASS_INDEX.get(a.cls)
    if idx is not None:
        onehot[idx] = 1.0
    out += onehot
    # 18~21: 도로 상태
    if p is None:
        out += [0.0, 0.0, 0.0, 0.0]
    else:
        lane_frac = (
            (p.lane_index / p.lane_count) if (p.lane_index and p.lane_count) else 0.0
        )
        out += [
            p.lateral_offset_m,                                    # 18 횡오프셋
            lane_frac,                                             # 19 차선 위치 비율
            (p.dist_to_next_junction_m
             if p.dist_to_next_junction_m is not None else 200.0),  # 20 교차로 거리
            1.0 if p.direction_confident else 0.0,                 # 21 방향 확신
        ]
    assert len(out) == N_GLOBAL_FEATURES, len(out)
    return out


def candidate_features(ctx: PredictContext, cand: Candidate) -> List[float]:
    """후보 하나를 고정 길이 벡터로. 길이 `N_CANDIDATE_FEATURES`."""
    a = ctx.actor
    poly = cand.polyline
    length = sum(
        math.dist(poly[i], poly[i + 1]) for i in range(len(poly) - 1)
    ) if len(poly) > 1 else 0.0
    # 후보의 진행 방향(첫 구간)과 액터 방위의 차이
    if len(poly) > 1:
        de, dn = poly[1][0] - poly[0][0], poly[1][1] - poly[0][1]
        first = math.degrees(math.atan2(de, dn)) % 360.0
    else:
        first = a.heading_deg or 0.0
    turn = ((first - (a.heading_deg or first)) + 180.0) % 360.0 - 180.0
    ts, tc = _sin_cos(turn)
    onehot = [0.0] * len(MANEUVERS)
    mi = MANEUVER_INDEX.get(cand.maneuver)
    if mi is not None:
        onehot[mi] = 1.0
    out = [
        cand.prior,                                   # 0 규칙 기반 사전확률
        ts, tc,                                       # 1,2 회전각
        turn / 180.0,                                 # 3 회전각 (정규화 스칼라)
        length,                                       # 4 후보 경로 길이 [m]
        length / max(ctx.horizon_m, 1e-6),            # 5 지평 대비 길이 비율
        float(len(poly)),                             # 6 정점 수
        1.0 if cand.to_road else 0.0,                 # 7 진입 도로 이름 있음
    ] + onehot                                        # 8~11 기동 원-핫
    assert len(out) == N_CANDIDATE_FEATURES, len(out)
    return out


def history_features(ctx: PredictContext) -> List[List[float]]:
    """과거 궤적을 `N_HISTORY_STEPS × N_HISTORY_FEATURES` 행렬로.

    행 순서는 **오래된 것 → 최근**(t-5 … t-1)이라 순환신경망·트랜스포머에 그대로
    넣을 수 있다. 좌표는 **현재 위치 기준 상대값**이다 — 절대 ENU 를 넣으면 모델이
    지역 좌표계의 원점을 외운다 (출력 쪽과 같은 이유).

    | # | 값 |
    |---|---|
    | 0,1 | de, dn — 현재 위치 기준 상대좌표 [m] |
    | 2,3 | vde, vdn — 그 시각의 1초 속도 벡터 [m/s] |
    | 4 | valid — 관측이 있었는가 (0/1) |

    관측이 없는 시각은 0 으로 채우고 `valid=0` 을 둔다. 트랙이 방금 생겼으면 전부
    0 일 수 있는데, 그것 자체가 "이력이 없다"는 정보다 — 0 으로만 채우면 모델이
    "정지해 있었다"로 오해한다.
    """
    e0, n0 = ctx.actor.world_xy
    hist = ctx.actor.track_history or []
    t_now = hist[-1][0] if hist else (ctx.t_s or 0.0)

    def at(dt_back: float):
        """t_now - dt_back 에 가장 가까운 관측. 0.5초 넘게 벌어지면 없는 것으로 본다."""
        if not hist:
            return None
        want = t_now - dt_back
        best = min(hist, key=lambda x: abs(x[0] - want))
        return best if abs(best[0] - want) <= HISTORY_DT_S * 0.5 + 1e-9 else None

    rows: List[List[float]] = []
    for k in range(N_HISTORY_STEPS, 0, -1):          # 5,4,3,2,1 → 오래된 것부터
        cur = at(float(k) * HISTORY_DT_S)
        if cur is None:
            rows.append([0.0] * N_HISTORY_FEATURES)
            continue
        prv = at(float(k + 1) * HISTORY_DT_S)
        if prv is None:
            vde = vdn = 0.0
        else:
            dt = cur[0] - prv[0]
            vde = (cur[1] - prv[1]) / dt if dt > 1e-6 else 0.0
            vdn = (cur[2] - prv[2]) / dt if dt > 1e-6 else 0.0
        rows.append([cur[1] - e0, cur[2] - n0, vde, vdn, 1.0])
    assert len(rows) == N_HISTORY_STEPS
    return rows


def _observer_aliases(actor: ActorState) -> set[str]:
    """actor 자신을 가리키는 fusion observer id 후보.

    observed_by 는 observer id(예: ``ego_vehicle``)를 저장하고 actor id는
    ``EGO_ego_vehicle`` 형식이다. 일반 observed actor는 관측자일 수 없으므로
    빈 집합을 반환한다. 빈 값은 '보지 못함'이 아니라 아래 availability flag로
    구별된다.
    """
    if actor.kind != "ego":
        return set()
    aliases = {actor.actor_id}
    if actor.actor_id.startswith("EGO_"):
        aliases.add(actor.actor_id[4:])
    return aliases


def observes_actor(observer: ActorState, subject: ActorState) -> Optional[bool]:
    """observer가 subject를 센서로 관측했는지, 알 수 없으면 None을 반환한다."""
    aliases = _observer_aliases(observer)
    if not aliases:
        return None
    return bool(aliases & set(subject.observed_by))


def interaction_features(
    ctx: PredictContext,
    max_neighbors: int = MAX_INTERACTION_NEIGHBORS,
) -> List[List[float]]:
    """Target 기준 주변 차량 상태를 고정 길이 feature 행으로 만든다.

    관측 관계는 ``A sees B``를 ``B.observed_by``에서 역으로 읽는다. 이는 V2X
    센서 관측일 뿐 운전자의 제동 의도는 아니며, 모델이 위치·속도·가속도와 함께
    조건부 미래 행동으로 학습한다.
    """
    target = ctx.actor
    tx, ty = target.world_xy
    heading = math.radians(target.heading_deg or 0.0)
    # ENU heading: 0°=north, 90°=east. forward와 left 단위벡터.
    forward = (math.sin(heading), math.cos(heading))
    left = (-math.cos(heading), math.sin(heading))

    def velocity(actor: ActorState) -> Tuple[float, float]:
        if actor.speed_mps is None or actor.heading_deg is None:
            return (0.0, 0.0)
        r = math.radians(actor.heading_deg)
        return (actor.speed_mps * math.sin(r), actor.speed_mps * math.cos(r))

    tvx, tvy = velocity(target)
    raw = [a for a in ctx.neighbors if a.actor_id != target.actor_id]
    raw.sort(key=lambda a: math.dist(target.world_xy, a.world_xy))
    out: List[List[float]] = []
    for other in raw[:max_neighbors]:
        de, dn = other.world_xy[0] - tx, other.world_xy[1] - ty
        ovx, ovy = velocity(other)
        rel_vx, rel_vy = ovx - tvx, ovy - tvy
        rel_forward = de * forward[0] + dn * forward[1]
        rel_left = de * left[0] + dn * left[1]
        vel_forward = rel_vx * forward[0] + rel_vy * forward[1]
        vel_left = rel_vx * left[0] + rel_vy * left[1]
        heading_known = target.heading_deg is not None and other.heading_deg is not None
        delta = math.radians((other.heading_deg or 0.0) - (target.heading_deg or 0.0))
        target_seen_value = observes_actor(target, other)
        other_seen_value = observes_actor(other, target)
        target_sees_other = bool(target_seen_value)
        other_sees_target = bool(other_seen_value)
        target_can_observe = target_seen_value is not None
        other_can_observe = other_seen_value is not None
        tp, op = target.placement, other.placement
        same_road = bool(tp and op and tp.road_id == op.road_id)
        same_lane = bool(same_road and tp.lane_index is not None
                         and tp.lane_index == op.lane_index)
        same_direction = bool(tp and op and tp.direction_label == op.direction_label)
        onehot = [0.0] * len(CLASSES)
        cls = CLASS_INDEX.get(other.cls)
        if cls is not None:
            onehot[cls] = 1.0
        out.append([
            rel_forward, rel_left, vel_forward, vel_left,
            other.speed_mps or 0.0, 1.0 if other.speed_mps is not None else 0.0,
            other.accel_mps2 or 0.0, 1.0 if other.accel_mps2 is not None else 0.0,
            math.sin(delta), math.cos(delta), 1.0 if heading_known else 0.0,
            1.0 if target_sees_other else 0.0,
            1.0 if other_sees_target else 0.0,
            1.0 if target_sees_other and other_sees_target else 0.0,
            1.0 if target_can_observe else 0.0,
            1.0 if other_can_observe else 0.0,
            1.0 if same_road else 0.0, 1.0 if same_lane else 0.0,
            1.0 if same_direction else 0.0,
            other.confidence, other.position_quality,
            *onehot,
            1.0,
        ])
    assert all(len(row) == N_INTERACTION_FEATURES for row in out)
    return out


def encode(ctx: PredictContext) -> Dict[str, object]:
    """학습·추론 공용 인코딩.

    `global` 은 길이 22 벡터, `candidates` 는 후보 수 × 12 행렬,
    `history` 는 5 × 5 행렬(과거 5초, 1초 간격)이다.
    후보가 없으면 `candidates` 가 빈 목록이다 (모델을 호출하지 않는다).
    """
    return {
        "global": global_features(ctx),
        "candidates": [candidate_features(ctx, c) for c in ctx.candidates],
        "history": history_features(ctx),
        "interactions": interaction_features(ctx),
    }


# ---------------------------------------------------------------- 예측기

# 예측기 인터페이스: PredictContext → PredictedPath 목록
Predictor = Callable[[PredictContext], List[PredictedPath]]


class TorchPredictor:
    """PyTorch 로 저장한 모델을 예측기로 쓴다.

    두 가지 모드를 지원한다.

    `mode="rank"` (기본, 권장)
        모델이 후보마다 점수를 내고 그것을 softmax 해 확률로 쓴다. 기하는 지도에서
        나온 것을 그대로 쓰므로 **노면을 벗어나지 않고**, 학습 데이터가 적어도
        동작한다. 규칙 기반의 사전확률을 학습된 확률로 바꾸는 것이다.

        forward: (global[22], candidates[N,12]) → scores[N]

    `mode="waypoints"`
        모델이 지평까지의 좌표를 직접 낸다. 지도 제약이 없으므로 자유롭지만
        학습 데이터가 많이 필요하고 노면을 벗어날 수 있다.

        forward: (global[22], candidates[N,12]) → offsets[K,2]
        출력은 **현재 위치 기준 상대 좌표**다 (절대 좌표를 회귀하면 지역 좌표계의
        원점 위치를 외우게 된다).

    모델을 못 불러오거나 후보가 없으면 `fallback`(기본: 규칙 기반)으로 넘긴다 —
    조용히 빈 예측을 내지 않는다.
    """

    def __init__(
        self,
        model_path: str,
        mode: str = "rank",
        device: str = "cpu",
        fallback: Optional[Predictor] = None,
    ):
        if mode not in ("rank", "waypoints"):
            raise ValueError(f"mode 는 'rank' 또는 'waypoints' 여야 합니다: {mode}")
        self.model_path = model_path
        self.mode = mode
        self.device = device
        self.fallback = fallback
        self._model = None
        self._torch = None
        self._wants_hist: Optional[bool] = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
        except ImportError as e:  # pragma: no cover - 선택 의존성
            raise ImportError(
                "TorchPredictor 는 torch 가 필요합니다.  pip install torch"
            ) from e
        self._torch = torch
        obj = torch.load(self.model_path, map_location=self.device, weights_only=False)
        # state_dict 만 저장한 경우는 모델 구조를 알 수 없으므로 거부한다 —
        # 조용히 잘못된 예측을 내는 것보다 낫다.
        if isinstance(obj, dict):
            raise ValueError(
                f"{self.model_path} 에 state_dict 만 있습니다. "
                "`torch.save(model, path)` 로 모델 객체를 저장하거나 "
                "TorchPredictor 를 상속해 _load() 를 재정의하십시오."
            )
        obj.eval()
        self._model = obj.to(self.device)
        return self._model

    def __call__(self, ctx: PredictContext) -> List[PredictedPath]:
        if not ctx.candidates:
            return self._fallback(ctx)
        model = self._load()
        torch = self._torch
        enc = encode(ctx)
        g = torch.tensor(enc["global"], dtype=torch.float32, device=self.device)
        c = torch.tensor(enc["candidates"], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self._wants_history(model):
                h = torch.tensor(
                    enc["history"], dtype=torch.float32, device=self.device
                )
                if self._wants_interactions(model):
                    i = torch.tensor(
                        enc["interactions"], dtype=torch.float32, device=self.device
                    )
                    out = model(g, c, h, i)
                else:
                    out = model(g, c, h)
            else:
                if self._wants_interactions(model):
                    i = torch.tensor(
                        enc["interactions"], dtype=torch.float32, device=self.device
                    )
                    out = model(g, c, None, i)
                else:
                    out = model(g, c)
        if self.mode == "rank":
            return self._from_scores(ctx, out)
        return self._from_waypoints(ctx, out)

    def _wants_history(self, model) -> bool:
        """모델이 과거 궤적을 3번째 인자로 받는가.

        `forward` 서명을 보고 정한다 — 예외를 잡아 판단하면 모델 **내부**의
        TypeError 까지 "이력을 안 받는다"로 오해한다. 판정 결과는 캐시한다
        (액터마다 스냅샷마다 부르는 자리다).

        TorchScript 처럼 서명을 읽을 수 없으면 `accepts_history` 속성을 보고,
        그것도 없으면 2-인자로 본다 — 과거 궤적 없이도 동작하는 쪽이 안전하다.
        """
        if self._wants_hist is None:
            flag = getattr(model, "accepts_history", None)
            if flag is not None:
                self._wants_hist = bool(flag)
            else:
                import inspect

                # nn.Module 이면 `forward`, 평범한 호출체면 그 자체를 본다.
                # 바인딩된 메서드라 `self` 는 세지 않는다.
                target = getattr(model, "forward", model)
                try:
                    params = inspect.signature(target).parameters
                    self._wants_hist = len(params) >= 3
                except (TypeError, ValueError):
                    self._wants_hist = False
        return self._wants_hist

    def _wants_interactions(self, model) -> bool:
        """InteractionWaypointNet 여부는 명시 attribute로 판정한다."""
        return bool(getattr(model, "accepts_interactions", False))

    def _fallback(self, ctx: PredictContext) -> List[PredictedPath]:
        if self.fallback is None:
            from .prediction import rule_predict

            return rule_predict(ctx)
        return self.fallback(ctx)

    def _from_scores(self, ctx: PredictContext, out) -> List[PredictedPath]:
        torch = self._torch
        scores = out.reshape(-1)
        if scores.numel() != len(ctx.candidates):
            raise ValueError(
                f"rank 모드는 후보 수({len(ctx.candidates)})만큼의 점수를 기대합니다 "
                f"— 받은 것 {tuple(out.shape)}"
            )
        probs = torch.softmax(scores, dim=0).tolist()
        from .prediction import candidate_to_path, merge_identical

        paths = [
            candidate_to_path(ctx, cand, prob)
            for cand, prob in zip(ctx.candidates, probs)
        ]
        # **잘라내지 않는다.** softmax 뒤에 상위 k개만 남기면 확률 합이 1 미만이
        # 되는데, payload 는 "합이 1" 을 규약으로 삼는다 (실측으로 합이 0.45 까지
        # 떨어졌다). 규칙 기반도 잘라내지 않고 전부 낸다 — 자연어 표시에서만
        # 상위 3개를 보인다.
        return merge_identical([p for p in paths if p is not None])

    def _from_waypoints(self, ctx: PredictContext, out) -> List[PredictedPath]:
        # 모델은 좌표만 내거나, (좌표, 기동 로짓) 을 낸다. 후자가 권장 형태다 —
        # 기동을 좌표에서 사후에 되짚으면 정확도가 규칙 기반보다 낮았다
        # (실측 94.5% 대 98.6%, docs/predict_model_io.md §5.8).
        logits = None
        if isinstance(out, (tuple, list)):
            if len(out) != 2:
                raise ValueError(
                    f"waypoints 모델은 좌표 하나 또는 (좌표, 기동 로짓) 두 개를 "
                    f"내야 합니다 — 받은 것 {len(out)}개"
                )
            out, logits = out
        arr = out.reshape(-1, 2).tolist()
        e0, n0 = ctx.actor.world_xy
        pts = [(e0 + de, n0 + dn) for de, dn in arr]
        if not pts or math.dist(pts[0], ctx.actor.world_xy) > 0.5:
            pts = [ctx.actor.world_xy] + pts

        if logits is not None:
            label, to_roads = self._label_from_logits(ctx, pts, logits)
        else:
            label, to_roads = self._describe(ctx, pts)
        return [
            PredictedPath(
                maneuver=label,
                probability=1.0,
                waypoints=pts,
                horizon_s=ctx.horizon_s,
                to_roads=to_roads,
                truncated=False,
            )
        ]

    def _label_from_logits(
        self, ctx: PredictContext, pts, logits
    ) -> Tuple[str, List[str]]:
        """모델이 분류한 기동. 진입 도로는 그 기동의 후보에서 고른다.

        도로 이름은 고정 어휘가 아니라 지도마다 다르므로 모델이 분류할 수 없다.
        그래서 **모델이 고른 기동**과 같은 기동을 가진 후보 중에서 예측 궤적에 가장
        가까운 것의 이름을 쓴다. "후보 밖" 이면 도로 이름을 붙이지 않는다 — 따라가지
        않는 도로를 적으면 그냥 오정보다.
        """
        vals = logits.reshape(-1).tolist()
        if len(vals) != len(MANEUVER_CLASSES):
            raise ValueError(
                f"기동 로짓은 {len(MANEUVER_CLASSES)}개여야 합니다 "
                f"({MANEUVER_CLASSES}) — 받은 것 {len(vals)}개"
            )
        label = MANEUVER_CLASSES[max(range(len(vals)), key=lambda i: vals[i])]
        if label == OFF_CANDIDATE_LABEL:
            return label, []
        same = [c for c in ctx.candidates if c.maneuver == label]
        if not same:
            # 모델이 지도에 없는 기동을 골랐다. 감추지 않고 그렇게 적는다.
            return f"{label}(해당 후보 없음)", []
        best = min(same, key=lambda c: self._cross_track(pts, c.polyline))
        return label, ([best.to_road] if best.to_road else [])

    @staticmethod
    def _cross_track(pts, poly) -> float:
        """예측 궤적과 폴리라인의 평균 횡거리. 폴리라인 끝을 넘어간 점은 제외한다."""
        from .geometry import project_point_to_polyline

        if len(poly) < 2:
            return float("inf")
        total = sum(math.dist(poly[i], poly[i + 1])
                    for i in range(len(poly) - 1))
        ds = []
        for p in pts[1:]:
            s, _, d, _ = project_point_to_polyline(p, poly)
            if s < total - 1e-6:
                ds.append(d)
        return (sum(ds) / len(ds)) if ds else float("inf")

    # 예측 궤적이 어느 후보 경로에서도 이보다 벗어나면 "후보 밖"으로 본다.
    # 차로 폭(3.5m)의 절반 — `make_predict_dataset.MATCH_TOLERANCE_M` 과 같은 기준.
    OFF_CANDIDATE_M = 3.0

    def _describe(self, ctx: PredictContext, pts) -> Tuple[str, List[str]]:
        """예측한 좌표에 **실제로 맞는** 기동 라벨과 진입 도로.

        사전확률이 가장 높은 후보의 라벨을 그대로 베끼면 안 된다. 모델이 그 후보와
        다른 방향을 내면 라벨이 좌표와 모순된다 — 실측으로 다후보 표본의 **8.9%**
        가 그랬고, 370건은 "우회전" 이라 적으면서 좌회전 궤적을 냈다. 자연어 브리핑과
        BEV 를 같이 읽는 쪽에서 이것은 그냥 오정보다.

        거리는 **점–폴리라인**(횡방향)으로 잰다. 같은 시각끼리 점–점으로 재면 속도
        차이가 섞인다 — 모델은 가속을 배우지만 후보는 등속 가정이므로, 같은 길을
        가면서도 몇 미터씩 벌어져 "다른 길" 로 오판된다 (실측으로 그렇게 됐다).
        기동 라벨이 답해야 하는 것은 **어느 길로 가는가**이고 언제 도착하는가가 아니다.

        어느 후보도 가깝지 않으면 감추지 않고 그렇게 적는다 — 지도를 벗어나는
        차선 변경·노면 이탈이 실제로 21% 다 (docs/predict_model_io.md §4.1).
        """
        best, best_d = None, float("inf")
        for cand in ctx.candidates:
            d = self._cross_track(pts, cand.polyline)
            if d < best_d:
                best, best_d = cand, d
        if best is None:
            return "경로 예측(후보 없음)", []
        if best_d > self.OFF_CANDIDATE_M:
            # 거릿값을 라벨에 넣지 않는다 — 1m 단위로 라벨이 쪼개져 어휘가 20종을
            # 넘고, 읽는 쪽이 같은 상황을 다른 상황으로 본다.
            return f"후보 밖 경로(가장 가까운 기동 {best.maneuver})", []
        return best.maneuver, ([best.to_road] if best.to_road else [])


class ScriptedRankPredictor(TorchPredictor):
    """TorchScript(`torch.jit.save`) 로 저장한 rank 모델용."""

    def _load(self):
        if self._model is not None:
            return self._model
        import torch

        self._torch = torch
        m = torch.jit.load(self.model_path, map_location=self.device)
        m.eval()
        self._model = m
        return m
