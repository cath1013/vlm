"""운동학 추정: 월드 궤적 → 속도·가속도·방위각·기동 분류, 그리고 상호작용 분석."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from .geometry import heading_to_unit, unit_to_heading, wrap180
from .schemas import ActorState, Interaction, RoadPlacement


# 단안 위치 추정 오차 규모 [m]. **중앙값이 아니라 상위 꼬리(p90)** 를 쓴다 —
# 이 값은 "실제로 움직였는가"를 판정하는 관문이고, 관문은 오탐을 막아야 하므로
# 최악에 가까운 잡음을 기준으로 삼아야 한다. 중앙값(0~70m 약 2m)을 쓰면
# 사실상 정지한 차량의 ±9m 진동이 실제 이동으로 통과해 방위가 180° 뒤집히고
# 속도가 63km/h 로 보고된다 (Town01 V001 에서 관측된 결함).
#
# DeepAccident mini 실측 p90: 0-15m 5.2m, 15-30m 8.9m, 30-45m 6.6m,
# 45-70m 9.1m, 70-130m 23.4m. 원거리에서 선형보다 빠르게 커진다 (평면노면 IPM
# 은 지평선에 가까워질수록 1픽셀이 수 m 에 대응한다) → 2차항을 둔다.
# 근거리 하한은 절단 bbox 로 접지점을 못 쓰는 구간(0~15m p90 5.2m)에 맞춘다.
SIGMA_FLOOR_M = 5.0
SIGMA_RANGE_COEFF = 0.06
SIGMA_RANGE_COEFF2 = 0.0015
# 이 거리를 넘는 관측에서는 궤적으로 **방위·진행방향을 주장하지 않는다**.
# 지평선 근처 역투영은 bbox 가 매끄럽게 흘러도 속도가 수십 m/s 로 잘못 나오고,
# 그 오차가 매끄럽기 때문에 잔차로도 거리척도로도 걸러지지 않는다
# (실측: 73m 트럭의 실제 2m/s 가 24.6m/s 로, 방향은 반대로 추정됨).
# 도로에 매칭된 차량은 하류가 도로축으로 그리므로 표현 손실은 없다.
MAX_RANGE_FOR_HEADING_M = 70.0


def position_sigma_m(range_m: float) -> float:
    """관측거리에 대응하는 위치 오차 규모 [m] (p90 근사)."""
    r = max(range_m, 0.0)
    return max(SIGMA_FLOOR_M, SIGMA_RANGE_COEFF * r + SIGMA_RANGE_COEFF2 * r * r)


@dataclass
class TrackHistory:
    """단일 참여자의 최근 궤적. EMA 로 잡음을 줄인다."""

    actor_id: str
    # 표본 개수 상한. 이력은 **시간으로 잘라** 쓰므로(아래 `recent`) 개수 상한은
    # 넉넉하면 된다. 5초 이력을 10Hz 까지 담을 수 있게 잡았다 — 예전 값 12 는
    # 2Hz 에서 6초였고, 예측 모델에 과거 궤적을 넣기 시작하면서 부족해졌다.
    maxlen: int = 64
    # (t, e, n, 관측거리) — 관측거리 0 은 ego(텔레메트리 기반, 편향 없음)
    samples: Deque[Tuple[float, float, float, float]] = field(default_factory=deque)
    # (t, 횡오프셋, 관측거리, 차선번호) — 관측거리로 측거 편향을 걸러내고,
    # 차선번호 변화로 실제 차선변경과 추정치 표류를 구분한다
    lat_offsets: Deque[Tuple[float, float, float, Optional[int]]] = field(
        default_factory=deque
    )
    placement_key: Optional[Tuple[str, str]] = None  # (road_id, direction)
    speed_ema: Optional[float] = None
    heading_ema: Optional[float] = None
    # 방위가 마지막으로 유의하게 확정된 시각. 정지 후에도 잠시 유지하기 위함.
    heading_valid_t: Optional[float] = None
    prev_speed: Optional[float] = None
    accel: Optional[float] = None
    first_t: Optional[float] = None
    # 클래스 상한을 넘는 함의 속도로 폐기한 표본 수 (위치 품질 진단용)
    noisy_steps: int = 0

    def __post_init__(self):
        self.samples = deque(maxlen=self.maxlen)
        self.lat_offsets = deque(maxlen=self.maxlen)

    def update(
        self,
        t: float,
        world_xy: Tuple[float, float],
        obs_range_m: Optional[float] = None,
        alpha: float = 0.5,
        stable_delta_m: float = 2.0,
        reliable_range_m: float = 45.0,
        max_speed_mps: Optional[float] = None,
        max_accel_mps2: float = 9.0,
    ) -> None:
        """위치 이력 갱신 → 속도/가속도/방위각 추정.

        가속도는 연속한 두 관측의 거리가 안정적일 때만 산출한다. 접근 중인
        원거리 차량은 프레임마다 측거 편향이 달라 겉보기 속도가 급변하므로,
        그 미분을 가속도로 보고하면 존재하지 않는 급감속을 만들어낸다.

        max_speed_mps: 이 클래스가 낼 수 있는 최대 속도. 두 관측 간 함의 속도가
          이를 넘으면 실제 이동이 아니라 위치 추정 잡음이므로(원거리 단안
          역투영은 수십 m 씩 튄다) 속도 갱신을 건너뛴다. 이 관문이 없으면
          트럭이 396km/h 로 보고되어 LLM 입력이 쓸모없어진다.
        max_accel_mps2: 물리적으로 가능한 최대 속도 변화율 [m/s²]. 0.9g 로,
          어떤 도로차량의 급제동·급가속보다 크다. 클래스 상한만으로는 상한
          바로 아래(트럭 118km/h)의 잡음 표본을 막을 수 없으므로 가속도
          타당성으로 한 번 더 걸른다.
        """
        if self.first_t is None:
            self.first_t = t
        r = obs_range_m or 0.0
        self.samples.append((t, world_xy[0], world_xy[1], r))

        if len(self.samples) < 2:
            return
        # 최근 두 샘플 간 변위로 순간 속도 산출 후 EMA 평활
        # (방위는 프레임 간 변위가 아니라 순변위로 따로 구한다 — heading_estimate)
        t0, e0, n0, r0 = self.samples[-2]
        t1, e1, n1, r1 = self.samples[-1]
        dt = t1 - t0
        if dt <= 1e-3:
            return
        de, dn = e1 - e0, n1 - n0
        step = math.hypot(de, dn)
        v = step / dt

        implausible = max_speed_mps is not None and v > max_speed_mps
        if (
            self.speed_ema is not None
            and abs(v - self.speed_ema) / dt > max_accel_mps2
        ):
            implausible = True
        if implausible:
            # 위치 잡음 지배 구간 — 속도·가속도 갱신을 건너뛴다. 방위는 순변위
            # 기반이라 이 표본 하나에 흔들리지 않으므로 계속 갱신한다.
            self.noisy_steps += 1
            self.accel = None
            self._update_heading(t)
            return

        self.prev_speed = self.speed_ema
        self.speed_ema = v if self.speed_ema is None else (
            alpha * v + (1 - alpha) * self.speed_ema
        )

        lo, hi = min(r0, r1), max(r0, r1)
        # 측거 편향은 거리에 비례하므로 프레임 간 편향 변화량은 거리 변화량에
        # 비례한다. 따라서 비율이 아니라 절대 변화량으로 판정한다.
        range_stable = (
            lo <= 1.0  # ego — 텔레메트리 기반, 편향 없음
            or hi <= reliable_range_m  # 근거리: 편향 자체가 작음
            or abs(r1 - r0) <= stable_delta_m  # 일정 간격 추종 등
        )
        if self.prev_speed is not None and range_stable:
            a = (self.speed_ema - self.prev_speed) / dt
            self.accel = a if self.accel is None else 0.5 * a + 0.5 * self.accel
        elif not range_stable:
            self.accel = None  # 판정 보류

        self._update_heading(t)

    def heading_from_fit(self, k_sigma: float = 2.0) -> Optional[float]:
        """최소자승 속도의 방향. 잡음과 구별되지 않으면 None.

        프레임 간 변위 방향을 EMA 로 누적하면 안 된다. 단안 측거 잡음은 주로
        시선방향 성분이므로, 관측자 쪽으로 곧게 달리는 차량은 프레임마다 전진과
        후퇴가 번갈아 나타나 변위 방향이 180° 씩 뒤집힌다. 그 오염이 EMA 에
        남아, 북향 트럭이 60° 를 바라보다 3초에 걸쳐 회전하는 모습이 된다
        (Town05 T013 에서 관측된 결함).

        관문은 **거리 기반 sigma_pos** 로 잡는다: 창 동안의 이동거리가 위치 오차
        규모의 k_sigma 배를 넘어야 한다. 잔차 기반 불확실도를 쓰면 매끄럽게
        표류하는 측거 편향이 잔차를 키우지 않은 채 방향을 뒤집는다 (실측:
        잔차 기준으로 바꾸자 180° 반전이 9%→17% 로 늘었다).
        """
        fit = self.velocity_fit()
        if fit is None:
            return None
        ve, vn, span, _, sigma_pos = fit
        if not math.isfinite(sigma_pos):
            return None
        if math.hypot(ve, vn) * span < k_sigma * sigma_pos:
            return None
        return unit_to_heading(ve, vn)

    def axis_component(
        self, axis_bearing_deg: float
    ) -> Optional[Tuple[float, float, float]]:
        """도로축 방향 속도 성분 (v_along, 경과시간, 거리기반 sigma_pos)."""
        fit = self.velocity_fit()
        if fit is None:
            return None
        ve, vn, span, _, sigma_pos = fit
        ue, un = heading_to_unit(axis_bearing_deg)
        return ve * ue + vn * un, span, sigma_pos

    def velocity_fit(
        self, window_s: float = 4.0, min_samples: int = 4
    ) -> Optional[Tuple[float, float, float, float, float]]:
        """최근 창의 최소자승 속도 (ve, vn, 경과시간, 속도 불확실도 σ_v).

        프레임 간 차분은 잡음을 그대로 속도로 옮긴다: 위치 잡음이 0.5초 간격에
        만드는 겉보기 속도는 수십 m/s 에 달해, 사실상 정지한 차량이 63km/h 로
        보고된다 (Town01 V001). 창 전체를 최소자승으로 적합하면 잡음이 표본 수
        만큼 상쇄된다.

        불확실도를 두 가지로 돌려준다. 오차 기제가 다르기 때문이다.
          - sigma_v (잔차 기반): 프레임 간 **산포**가 속도 **크기**의 정밀도를
            제한한다. 위치 오차의 큰 부분은 창 안에서 거의 일정한 측거 편향이고
            그것은 기울기에서 상쇄되므로, 위치 오차 크기(p90)를 그대로 쓰면
            과하게 보수적이다. 이 트랙이 실제로 얼마나 튀는지는 잔차가 말해 준다.
          - sigma_pos (거리 기반 p90): 측거 편향은 **매끄럽게 표류**하므로 잔차를
            키우지 않으면서 기울기의 **부호**를 뒤집을 수 있다. 방위·진행방향
            판정은 이쪽을 기준으로 해야 한다 (잔차만 보면 매끄럽게 뒤집힌
            궤적을 정상으로 판정한다).
        """
        if len(self.samples) < min_samples:
            return None
        t_end = self.samples[-1][0]
        win = [sm for sm in self.samples if t_end - sm[0] <= window_s + 1e-9]
        if len(win) < min_samples:
            return None
        n = len(win)
        tm = sum(sm[0] for sm in win) / n
        stt = sum((sm[0] - tm) ** 2 for sm in win)
        if stt < 1e-9:
            return None
        em = sum(sm[1] for sm in win) / n
        nm = sum(sm[2] for sm in win) / n
        ve = sum((sm[0] - tm) * (sm[1] - em) for sm in win) / stt
        vn = sum((sm[0] - tm) * (sm[2] - nm) for sm in win) / stt
        # 잔차 표준편차 (두 축 합산, 자유도 n-2)
        ss = 0.0
        for t, e, nn, _ in win:
            ss += (e - (em + ve * (t - tm))) ** 2 + (nn - (nm + vn * (t - tm))) ** 2
        sigma_res = math.sqrt(ss / max(2 * (n - 2), 1))
        sigma_v = sigma_res / math.sqrt(stt)
        rng = max(sm[3] for sm in win)
        sigma_pos = position_sigma_m(rng)
        if rng > MAX_RANGE_FOR_HEADING_M:
            # 방위·방향 관문을 통과할 수 없게 만든다 (속도 관문은 잔차 기준이라
            # 영향받지 않는다 — 원거리 속도는 여전히 보고하되 방향은 주장 안 함)
            sigma_pos = float("inf")
        return ve, vn, win[-1][0] - win[0][0], sigma_v, sigma_pos

    def speed_from_fit(
        self, max_speed_mps: Optional[float] = None, k_sigma: float = 2.0
    ) -> Optional[float]:
        """최소자승 속도의 크기. 잡음과 구별되지 않으면 None (= 속도 미확정).

        판정은 추정값이 자기 **불확실도**의 k_sigma 배를 넘는가로 한다. 창을
        여러 개 시도해 '유의한 첫 창'을 고르면 안 된다 — 유의성을 보고할 값
        자체로 판정하므로, 잡음으로 부풀려진 추정이 관문을 통과하는 선택 편향이
        생긴다 (실측: 트랙 초반 4.4m/s 차량이 26m/s 로 보고됨).

        정지로 단정하지도, 잡음값을 내지도 않는다 — 정지로 보고하면 LLM 이
        정지차량으로 오해하고, 잡음값을 내면 없는 주행을 만든다.
        """
        fit = self.velocity_fit()
        if fit is None:
            return None
        ve, vn, _, sigma_v, _ = fit
        v = math.hypot(ve, vn)
        if v < k_sigma * sigma_v:
            return None
        if max_speed_mps is not None and v > max_speed_mps:
            return None  # 클래스가 낼 수 없는 속도 = 잡음 지배
        return v

    def _update_heading(self, t: float, alpha: float = 0.6) -> None:
        """적합 방위로 EMA 갱신. 유의하지 않으면 직전 값을 유지한다.

        정지 중에도 값을 유지하는 이유: 적신호에 멈춘 차량의 방위는 멈추기 전
        주행에서 이미 알고 있다. 유효 시각(heading_valid_t)을 기록해 두어
        하류가 너무 낡은 값을 쓰지 않게 한다.
        """
        est = self.heading_from_fit()
        if est is None:
            return
        self.heading_valid_t = t
        if self.heading_ema is None:
            self.heading_ema = est
        else:
            # 각도 EMA — 0/360 경계 처리
            self.heading_ema = (
                self.heading_ema + alpha * wrap180(est - self.heading_ema)
            ) % 360.0

    def heading_at(self, t: float, hold_s: float = 2.0) -> Optional[float]:
        """이 시각에 보고할 방위. 유의한 추정이 hold_s 안에 있었을 때만.

        hold_s 가 길면 낡은 방위가 남는다 — 보행자는 2초면 방향을 바꾸고,
        정지한 차량은 도로축으로 그리는 것이 더 정확하다.
        """
        if self.heading_ema is None or self.heading_valid_t is None:
            return None
        return self.heading_ema if t - self.heading_valid_t <= hold_s else None

    def record_placement(
        self,
        t: float,
        placement: Optional[RoadPlacement],
        obs_range_m: Optional[float] = None,
    ) -> None:
        """확정된 도로 매칭의 횡오프셋 기록.

        매칭된 도로나 진행방향이 바뀌면 이력을 버린다. 그렇지 않으면
        부호 규약이 다른 값이 섞여 허위 차선변경으로 오분류된다.
        """
        if placement is None:
            self.lat_offsets.clear()
            self.placement_key = None
            return
        key = (placement.road_id, placement.direction_label)
        if key != self.placement_key:
            self.lat_offsets.clear()
            self.placement_key = key
        self.lat_offsets.append(
            (
                t,
                placement.lateral_offset_m,
                obs_range_m if obs_range_m else 0.0,
                placement.lane_index,
            )
        )

    @property
    def age_s(self) -> float:
        if self.first_t is None or not self.samples:
            return 0.0
        return self.samples[-1][0] - self.first_t

    def recent(self, t: float, span_s: float) -> List[Tuple[float, float, float]]:
        """[t-span_s, t] 구간의 (시각, e, n). 오래된 것 → 최근 순.

        예측 모델에 넣을 과거 궤적의 출처다. EMA 로 매끈하게 만든 값이 아니라
        **관측된 위치 그대로**를 준다 — 평활은 모델이 할 일이고, 여기서 미리 하면
        급감속·차선변경 같은 신호가 지워진다.
        """
        return [
            (sm[0], sm[1], sm[2])
            for sm in self.samples
            if t - sm[0] <= span_s + 1e-9
        ]

    def lateral_trend(
        self,
        window_s: float = 2.5,
        max_range_delta_m: float = 8.0,
        reliable_range_m: float = 45.0,
        noise_floor_m: float = 0.05,
    ) -> Optional[Tuple[float, float, bool]]:
        """최근 window_s 구간의 (횡방향 변화율 [m/s], 총 변위 [m], 차선번호 변화).

        전체 이력이 아니라 **최근 구간**만 본다. 차선변경은 2~4초에 걸쳐
        일어나므로, 기동 이전의 평탄한 구간까지 포함해 단조성을 검사하면
        실제 차선변경을 놓친다.

        세 조건을 모두 만족해야 값을 반환한다 (아니면 None = 판정 보류).
          1) 최근 창에 4개 이상 샘플, 0.8초 이상 — 짧은 기울기는 잡음
          2) 단조성 — 잡음 수준(noise_floor_m) 이상의 변화만 세고, 그중
             70% 이상이 같은 방향이어야 한다
          3) 관측거리 안정성 — 단안 횡오프셋에는 거리 의존 편향이 있어
             접근/이탈 중인 원거리 차량의 오프셋은 시계열 비교가 불가능하다.
             (원거리 차량의 실제 차선변경은 놓치지만, 없는 기동을 만들어
              LLM 에 넘기는 것보다 낫다)
        """
        if len(self.lat_offsets) < 4:
            return None
        t_end = self.lat_offsets[-1][0]
        win = [x for x in self.lat_offsets if t_end - x[0] <= window_s]
        if len(win) < 4:
            return None

        t0, o0, _, lane0 = win[0]
        t1, o1, _, lane1 = win[-1]
        dt = t1 - t0
        if dt < 0.8:
            return None

        ranges = [r for _, _, r, _ in win if r > 1.0]
        if (
            ranges
            and max(ranges) > reliable_range_m
            and (max(ranges) - min(ranges)) > max_range_delta_m
        ):
            return None

        disp = o1 - o0
        sign = 1 if disp > 0 else -1
        diffs = [win[i + 1][1] - win[i][1] for i in range(len(win) - 1)]
        significant = [d for d in diffs if abs(d) > noise_floor_m]
        if len(significant) < 2:
            return None
        if sum(1 for d in significant if d * sign > 0) < 0.7 * len(significant):
            return None
        lane_changed = (
            lane0 is not None and lane1 is not None and lane0 != lane1
        )
        return (disp / dt, disp, lane_changed)


class KinematicsTracker:
    """actor_id 별 궤적 이력을 유지하며 상태를 채운다."""

    def __init__(
        self,
        timeout_s: float = 2.5,
        lane_width_m: float = 3.25,
        max_speed_by_class: Optional[Dict[str, float]] = None,
        max_speed_mps: Optional[float] = None,
        # `ActorState.track_history` 에 붙일 과거 궤적의 길이 [s].
        # 모델이 쓰는 이력은 과거 5초(`N_HISTORY_STEPS × HISTORY_DT_S`)지만
        # **1초를 더 받는다** — 가장 오래된 시각(t-5)의 속도를 t-6 과의 차분으로
        # 구하기 때문이다. 5초만 받으면 그 행의 속도가 0 이 되어 모델이 "그때
        # 멈춰 있었다"로 오해한다 (실측으로 그렇게 나왔다).
        history_s: float = 6.0,
    ):
        self.timeout_s = timeout_s
        self.history_s = history_s
        self.lane_width_m = lane_width_m
        self.max_speed_by_class = max_speed_by_class or {}
        self.max_speed_mps = max_speed_mps
        self.tracks: Dict[str, TrackHistory] = {}

    def speed_limit_of(self, cls: str) -> Optional[float]:
        return self.max_speed_by_class.get(cls, self.max_speed_mps)

    def update(self, t: float, actor: ActorState) -> None:
        """1단계: 궤적으로 속도·가속도·방위각 추정 (도로 매칭 정제 전)."""
        h = self.tracks.setdefault(actor.actor_id, TrackHistory(actor.actor_id))
        h.update(
            t,
            actor.world_xy,
            actor.observed_range_m,
            max_speed_mps=self.speed_limit_of(actor.cls),
        )

        # ego 는 텔레메트리 값이 우선 (GPS/IMU 가 영상 추정보다 정확)
        if actor.speed_mps is None:
            actor.speed_mps = h.speed_from_fit(self.speed_limit_of(actor.cls))
        if actor.heading_deg is None:
            # 순변위가 위치 잡음 규모를 유의하게 넘을 때만 방위를 공개한다.
            # 아니면 미확정으로 남기고, 하류(BEV·직렬화)가 도로 방향으로
            # 가정하거나 방향 없이 표시하게 한다.
            actor.heading_deg = h.heading_at(t)
            if actor.heading_deg is not None:
                actor.heading_source = "trajectory"
        actor.accel_mps2 = h.accel
        actor.track_age_s = h.age_s

    def direction_hint(self, actor_id: str) -> Optional[float]:
        """도로 선택용 방위 힌트."""
        h = self.tracks.get(actor_id)
        return h.heading_from_fit() if h else None

    def axis_direction(
        self, actor_id: str, axis_bearing_deg: float, k_sigma: float = 2.0
    ) -> Optional[bool]:
        """도로축 기준 진행 부호. True=역방향, None=판정 불가.

        축방향 **이동거리**(성분 × 경과시간)가 위치 오차 규모의 k_sigma 배를
        넘을 때만 확정한다. 관문이 없으면 잡음으로 절반은 뒤집혀, 조감도에서
        차량이 도로를 거꾸로 달리는 것으로 그려진다.

        확정하지 못하면 하류가 방위를 **미확정**으로 둔다. 사각형은 180° 뒤집어도
        같은 모양이므로, 도로축에 맞춰 그리는 한 표현 손실이 없다.
        """
        h = self.tracks.get(actor_id)
        if h is None:
            return None
        comp = h.axis_component(axis_bearing_deg)
        if comp is None:
            return None
        along, span, sigma_pos = comp
        if not math.isfinite(sigma_pos):
            return None
        if abs(along) * span < k_sigma * sigma_pos:
            return None
        return along < 0

    def finalize(self, t: float, actor: ActorState) -> None:
        """2단계: 확정된 도로 매칭으로 횡오프셋 기록·기동 분류·과거 궤적 부착."""
        h = self.tracks.setdefault(actor.actor_id, TrackHistory(actor.actor_id))
        h.record_placement(t, actor.placement, actor.observed_range_m)
        actor.maneuver = classify_maneuver(actor, h, self.lane_width_m)
        # 경로 예측이 바로 다음에 불리므로 여기서 이력을 붙인다.
        actor.track_history = h.recent(t, self.history_s)

    def prune(self, t: float) -> None:
        stale = [
            k
            for k, v in self.tracks.items()
            if v.samples and t - v.samples[-1][0] > self.timeout_s
        ]
        for k in stale:
            del self.tracks[k]


def classify_maneuver(
    actor: ActorState, hist: TrackHistory, lane_width_m: float = 3.25
) -> str:
    """기동 분류: 정지 / 차선유지 / 차선변경 / 회전 / 가감속.

    관측차량(ego)은 GPS 방위각을 쓰므로 임계값을 낮게, 주변차량은 궤적
    미분으로 방위각을 추정하므로 임계값을 높이고 최소 추적시간을 요구한다.
    """
    v = actor.speed_mps
    if v is None:
        # 속도를 아직 모르는 것과 정지한 것은 다르다. 첫 관측 시점의 차량을
        # '정지'로 보고하면 LLM 이 정지차량으로 오해한다.
        return "속도 미확정"
    if v < 0.6:
        return "정지"

    # 도로 방위각 대비 헤딩 편차 → 회전 판정
    if actor.placement is not None and actor.heading_deg is not None:
        if actor.kind == "ego":
            thresh, min_age = 25.0, 0.0
        else:
            thresh, min_age = 35.0, 1.0
        dev = wrap180(actor.heading_deg - actor.placement.bearing_deg)
        if abs(dev) > thresh and hist.age_s >= min_age:
            return "좌회전 중" if dev > 0 else "우회전 중"

    trend = hist.lateral_trend()
    if trend is not None:
        rate, disp, lane_changed = trend
        # 임계값은 관측거리에 따라 완화 (원거리일수록 횡방향 추정 잡음 증가)
        rng = actor.observed_range_m or 0.0
        rate_thresh = 0.35 + 0.005 * rng
        # 변위 조건: 차선번호가 실제로 바뀌었으면 반차선폭, 아니면 0.8차선폭.
        # 차선번호가 그대로인데 오프셋만 움직이는 것은 추정치 표류일 가능성이
        # 높으므로 더 큰 변위를 요구한다.
        disp_thresh = (0.5 if lane_changed else 0.8) * lane_width_m
        if abs(rate) > rate_thresh and abs(disp) > disp_thresh:
            return "차선변경(좌)" if rate > 0 else "차선변경(우)"

    # 가감속 판정 — 주변차량은 측거 편향이 속도 추정에 섞이므로 임계값을 높인다
    if actor.accel_mps2 is not None:
        if actor.kind == "ego":
            acc_hi, acc_lo, min_age = 1.0, -1.5, 0.0
        else:
            acc_hi, acc_lo, min_age = 2.0, -2.5, 1.5
        if hist.age_s >= min_age:
            if actor.accel_mps2 > acc_hi:
                return "가속 중"
            if actor.accel_mps2 < acc_lo:
                return "감속 중"
    return "차선유지"


# ---------------------------------------------------------------- 상호작용


def analyze_interactions(
    actors: List[ActorState],
    same_lane_lateral_m: float = 2.2,
    max_gap_m: float = 60.0,
) -> List[Interaction]:
    """선행/후행 관계, 헤드웨이·TTC, 교차 충돌 가능성 산출."""
    out: List[Interaction] = []
    by_road: Dict[Tuple[str, str], List[ActorState]] = {}
    for a in actors:
        if a.placement is None:
            continue
        by_road.setdefault(
            (a.placement.road_id, a.placement.direction_label), []
        ).append(a)

    # 1) 동일 도로·동일 방향의 종방향 추종 관계
    for group in by_road.values():
        group.sort(key=lambda x: x.placement.s_m)
        forward = group[0].placement.bearing_deg
        # s 증가 방향이 진행방향과 반대인 경우 정렬 뒤집기
        ref = group[0].placement
        if abs(wrap180(forward - ref.bearing_deg)) > 90.0:
            group.reverse()

        for i in range(len(group) - 1):
            rear, lead = group[i], group[i + 1]
            if (
                abs(rear.placement.lateral_offset_m - lead.placement.lateral_offset_m)
                > same_lane_lateral_m
            ):
                continue  # 다른 차선
            gap = abs(lead.placement.s_m - rear.placement.s_m)
            if gap > max_gap_m:
                continue
            v_rear = rear.speed_mps or 0.0
            v_lead = lead.speed_mps or 0.0
            headway = gap / v_rear if v_rear > 0.5 else None
            closing = v_rear - v_lead
            ttc = gap / closing if closing > 0.3 else None
            out.append(
                Interaction(
                    kind="following",
                    subject_id=rear.actor_id,
                    object_id=lead.actor_id,
                    gap_m=gap,
                    headway_s=headway,
                    ttc_s=ttc,
                    note="선행차 추종",
                )
            )

    # 2) 교차 경로 — 같은 교차로에 비슷한 시각 도달하고 경로가 실제로 교차할 때만
    for i, a in enumerate(actors):
        for b in actors[i + 1 :]:
            if a.placement is None or b.placement is None:
                continue
            if a.placement.road_id == b.placement.road_id:
                continue
            jid_a = a.placement.next_junction_id
            jid_b = b.placement.next_junction_id
            if jid_a is None or jid_a != jid_b:
                continue
            da = a.placement.dist_to_next_junction_m
            db = b.placement.dist_to_next_junction_m
            if da is None or db is None or max(da, db) > 70.0:
                continue
            va, vb = (a.speed_mps or 0.0), (b.speed_mps or 0.0)
            ta = da / va if va > 0.5 else None
            tb = db / vb if vb > 0.5 else None
            if ta is None or tb is None or abs(ta - tb) > 3.0:
                continue

            conflict, reason, conflict_kind, turn_probs = _paths_conflict(a, b)
            if not conflict:
                continue
            out.append(
                Interaction(
                    kind="crossing",
                    subject_id=a.actor_id,
                    object_id=b.actor_id,
                    ttc_s=min(ta, tb),
                    note=(
                        f"교차로 {jid_a}, 도달시간차 {abs(ta - tb):.1f}s | {reason}"
                    ),
                    junction_id=jid_a,
                    arrival_gap_s=abs(ta - tb),
                    conflict=conflict_kind,
                    turn_probs=turn_probs,
                )
            )

    # 3) 차선변경 경합 — 인접 차선으로 이동 중인 차와 그 차선 점유차
    changers = [a for a in actors if a.maneuver.startswith("차선변경")]
    for c in changers:
        if c.placement is None or c.placement.lane_index is None:
            continue
        target = c.placement.lane_index + (
            -1 if "좌" in c.maneuver else 1
        )
        for other in actors:
            if other.actor_id == c.actor_id or other.placement is None:
                continue
            if other.placement.road_id != c.placement.road_id:
                continue
            if other.placement.lane_index != target:
                continue
            gap = abs(other.placement.s_m - c.placement.s_m)
            if gap > 35.0:
                continue
            out.append(
                Interaction(
                    kind="lane_change_conflict",
                    subject_id=c.actor_id,
                    object_id=other.actor_id,
                    gap_m=gap,
                    note=f"{target}차선 진입 경합",
                    target_lane=target,
                )
            )
    return out


def _turn_probability(actor: ActorState, labels: Tuple[str, ...]) -> float:
    """지정한 기동들의 합산 확률. 예측이 없으면 0."""
    return sum(
        p.probability for p in actor.predictions if p.maneuver in labels
    )


def _paths_conflict(
    a: ActorState, b: ActorState, turn_prob_thresh: float = 0.3
) -> Tuple[bool, str, str, List[Tuple[str, float]]]:
    """두 차량의 교차로 통과 경로가 실제로 상충하는지 판정.

    같은 교차로에 동시 도달한다는 사실만으로는 상충이 아니다.
      - 직교 접근  → 상충 (한쪽이 다른 쪽 경로를 가로지른다)
      - 대향 접근  → 양쪽 모두 직진이면 상충 아님. 한쪽이 좌회전/유턴할
                     가능성이 유의미하면 대향 직진 경로를 가로지르므로 상충으로
                     본다. 최상위 경로만 보면 확률 37% 의 좌회전 상충이 묻히므로
                     합산 확률로 판정한다.
                     임계값(기본 0.3)은 차선 정보가 없을 때의 균등 사전확률
                     (좌/우 각 0.2)보다 높게 둔다. 그렇지 않으면 정보가 전혀
                     없는 상태에서 모든 대향 쌍이 상충으로 보고된다.
      - 동방향 접근 → 종방향 추종이므로 following 항목에서 다룬다.
    """
    if a.placement is None or b.placement is None:
        return (False, "", "", [])
    diff = abs(wrap180(a.placement.bearing_deg - b.placement.bearing_deg))

    if diff < 30.0:  # 같은 방향
        return (False, "", "", [])
    if diff > 150.0:  # 대향
        probs: List[Tuple[str, float]] = []
        for act in (a, b):
            p = _turn_probability(act, ("좌회전", "유턴"))
            if p >= turn_prob_thresh:
                probs.append((act.actor_id, p))
        if probs:
            # 회전 확률은 차선 위치에서 유도한 사전확률이며, 관측된 의도가
            # 아니다 (방향지시등·감속을 보지 않음). LLM 이 과신하지 않도록 명시.
            turners = ", ".join(f"{aid} {p * 100:.0f}%" for aid, p in probs)
            return (
                True,
                "대향 좌회전 상충 가능 — 좌회전/유턴 확률(차선위치 기반 "
                f"사전확률): {turners}",
                "oncoming_turn",
                probs,
            )
        return (False, "", "", [])
    return (True, "직교 진입 상충 예상", "orthogonal", [])


def risk_score(actor: ActorState, interactions: List[Interaction]) -> float:
    """LLM 입력 토큰 예산 배분용 중요도. 클수록 우선 포함."""
    score = 0.0
    if actor.kind == "ego":
        score += 100.0
    for it in interactions:
        if actor.actor_id not in (it.subject_id, it.object_id):
            continue
        if it.ttc_s is not None:
            score += max(0.0, 30.0 - it.ttc_s * 3.0)
        if it.headway_s is not None and it.headway_s < 2.0:
            score += 10.0
        if it.kind in ("crossing", "lane_change_conflict"):
            score += 8.0
    score += actor.confidence * 2.0
    if actor.maneuver not in ("차선유지", "정지"):
        score += 5.0
    return score
