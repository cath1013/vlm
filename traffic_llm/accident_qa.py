"""사고 예측 질의 생성: 슬라이딩 시간 윈도우 → LLM payload + 정답 파일.

길이 T 의 윈도우를 stride 간격으로 밀며(예: T=5, stride=1 → 0~5, 1~6, 2~7 …),
각 윈도우의 교통 상황을 LLM 입력으로 만들고 **마지막 관측 시점 이후 미래 N초를
1초 구간별로** 사고 발생 여부·관련 차량·이유를 묻는다.

    윈도우당 파일 2개
        llm_payload_0-5.json    ← LLM 요청 본문 (정답 없음)
        ground_truth_0-5.json   ← 채점용 정답 (LLM 에 주지 않음)
    그리고 전체를 묶는 manifest.json

시간 구간 규약 (질문·정답·채점이 모두 같은 정의를 쓴다)
    k번째 구간 = 마지막 관측 시점 t_end 이후 (k-1, k] 초.
    즉 k=1 은 (t_end, t_end+1], k=2 는 (t_end+1, t_end+2] …
    1초 단위로 "그 시점"을 물으면 경계가 모호해지므로 구간으로 정의한다.

정답 leakaage 방지
    - 시나리오 id 는 분할명("..._accident")을 담으므로 payload 에서는 불투명
      토큰으로 치환한다 (serialize.scenario_ref).
    - 충돌 정보는 payload 에 넣지 않는다.
    - 충돌이 관측 윈도우 **안에서** 이미 일어난 윈도우는 예측 문제가 아니므로
      기본적으로 버린다 (drop_windows_after_collision).
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import i18n, providers
from .config import SerializeConfig
from .kinematics import risk_score
from .predict_model import observes_actor
from .schemas import ActorState, CollisionTruth, SceneSnapshot
from .swept_path import actor_footprint_m, swept_pair_clearances
from .serialize import (
    MPS_TO_KPH,
    scenario_ref,
    to_bev_ascii,
    to_json,
    to_text,
)


# ---------------------------------------------------------------- 설정


@dataclass
class WindowConfig:
    """윈도우/질의 설정."""

    window_s: float = 5.0  # T — 관측 구간 길이
    stride_s: float = 1.0  # 윈도우 이동 간격
    horizon_s: float = 5.0  # N — 미래 예측 구간
    snapshot_rate_hz: float = 2.0  # 윈도우 안 스냅샷 밀도
    history_stride_s: float = 1.0  # 이력 표시 간격 (토큰 절약)
    # 수록 차량 수 상한. None(기본) 이면 **관측된 차량 수에 맞춰** 정한다 —
    # 고정 상한은 관측이 늘어날수록 담는 비율이 줄어들어, 카메라를 6대로 늘렸을 때
    # 실제로 사고 당사자가 목록 밖으로 밀려났다. `actor_cap()` 참조.
    max_actors: Optional[int] = None
    # 접근 쌍 표시 수 상한. None 이면 차량 수에 맞춰 정한다. 쌍의 수는 차량 수의
    # 제곱으로 늘어나므로 고정 상한이 특히 위험하다 (43쌍→88쌍인데 상한 6 고정).
    max_closing_pairs: Optional[int] = None
    # 상한을 넘더라도 이 시간 안에 접촉이 예상되는 쌍은 반드시 담는다.
    # 순위로 자르면 임박한 쌍이 밀려날 수 있는데, 그것이 바로 놓치면 안 되는
    # 쌍이다. None(기본) 이면 **질문 지평의 절반**(horizon_s × 0.5) — 지평을
    # 바꾸면 같이 움직인다. 0 이면 순수 상위 N.
    # 실측(Town05): 지평 5초에서 절반(2.5초)이면 사고 2.2·3.2초 전 윈도우까지
    # 당사자 쌍을 담고 쌍 수는 13→18~22개다. 지평 전체(5초)로 넓히면 32~40개로
    # 늘지만 이 시나리오에서 추가로 잡히는 당사자 쌍은 없다.
    closing_must_include_contact_s: Optional[float] = None
    closing_must_include_contact_frac: float = 0.5
    # 상한 자동 산정 계수 — 아래 actor_cap()/pair_cap() 에서 쓴다
    actor_cap_floor: int = 15
    actor_cap_ceiling: int = 40
    pair_cap_floor: int = 6
    pair_cap_ceiling: int = 20
    pair_cap_per_actor: float = 0.5
    include_bev_last: bool = True  # 마지막 시점 BEV 포함
    # 구조화 JSON 블록을 payload 에 넣는가. **기본 꺼짐**.
    # 바로 위 자연어 브리핑과 내용이 거의 같다 — 다섯 블록(window, last_snapshot,
    # closing_pairs, perception, trajectories) 전부가 브리핑의 대응 절과 짝을
    # 이루면서 입력의 79% 를 차지한다. 켜면 payload 가 약 4.5배가 된다
    # (실측 26K → 117K자).
    # JSON 에만 있던 값(방위각 숫자, 지도 규약)은 자연어 쪽으로 옮겨 두었으므로
    # 꺼도 정보가 사라지지 않는다. 툴 호출·정량 파싱이 필요할 때만 켠다.
    include_json_block: bool = False
    # 궤적 표본마다 관측 관계(누가 봤는가·관측거리·확신도)를 함께 담는다.
    # 이것이 없으면 "누가 누구를 인지했는가"가 마지막 시각만 남아, 시야에
    # 들어오고 나가는 변화를 읽을 수 없다.
    # 관측자 이름을 표본마다 반복하므로 입력이 약 24% 커진다 (실측 57.3K→71.4K자).
    # "없으면 직전과 같다" 같은 규약으로 줄일 수도 있지만, 그러면 목록이 없는 것을
    # "인지하지 못했다"로 오해할 수 있어 매번 명시한다. 토큰이 부담되면 False.
    trajectory_observation: bool = True
    # 관측자 관점 요약(perception 블록). 액터별 observed_by 를 역변환한 것이지만,
    # 그 역변환을 모델에게 맡기면 자주 틀린다 — V2X 사각지대 보완이 이 관계에
    # 달려 있으므로 명시적으로 준다.
    include_perception: bool = True
    # LLM 에 보내는 장면 표현. standard 는 기존 자연어 브리핑을 그대로 보존한다.
    # compact 는 관측 궤적 + 현재 상태 + 예측기의 미래 웨이포인트만 구조화해 보내고,
    # 접근쌍 외삽/TTC·헤드웨이 상호작용/ASCII BEV 같은 파생 위험 단서는 뺀다.
    payload_profile: str = "standard"  # 'standard' | 'compact'
    # Re-ranker가 검사할 내부 후보 pool과 compact payload에 최종 수록할 pair 수를
    # 분리한다. Clearance만으로 먼저 12개를 잘랐을 때 실제 충돌 pair recall이
    # train 56.8%, validation 59.6%로 제한됐다. 내부에서는 50개를 유지하고,
    # re-ranking 후 payload에는 12개만 보내 입력 크기를 보존한다.
    swept_candidate_pool_cap: int = 50
    swept_pair_cap: int = 12
    swept_sample_dt_s: float = 0.1
    swept_contact_margin_m: float = 0.0
    swept_exclude_touching_now: bool = True
    swept_exclude_static_pairs: bool = True
    # Optional learned pair selector.  The geometry collector always builds the
    # clearance-ranked candidate pool first; when this is set, compact payloads
    # re-rank that pool and serialize only the highest-scoring pairs.
    pair_reranker_model: object | None = field(default=None, repr=False)
    pair_reranker_note: str = ""
    history_mode: str = "compact"  # 'compact' | 'full'(스냅샷 JSON 전부)
    # 기본은 **고정 길이** sliding window다. window_s=5, stride=1 →
    # 0-5, 1-6, 2-7, … . 짧은 선두 창을 만들지 않는다.
    # expanding-prefix(0-1, 0-2, …)가 필요할 때만 warmup=True로 명시한다.
    warmup: bool = False
    # 기본은 이동 창만 만든다. 데이터 전체 구간 창(예: 0-9)을 추가하려면
    # full_window=True를 명시한다.
    full_window: bool = False
    require_full_window: bool = True  # 창 끝에 실제 스냅샷이 있어야 함
    drop_windows_after_collision: bool = True
    # 사고 발생 시점보다 이만큼 **먼저** 예측한 것은 정답으로 인정한다 [s].
    # 사고 예측의 목적은 회피할 시간을 버는 것이므로, 조금 이른 경보는 맞은
    # 것으로 보는 편이 과제에 맞는다. **늦은 예측은 여전히 오답이다** — 이미
    # 일어난 사고를 예측이라 부를 수 없다.
    # 0 이면 구간이 정확히 일치해야만 정답 (엄격 채점).
    early_credit_s: float = 2.0
    # None 이면 SerializeConfig.model → provider 기본값 순으로 정해진다
    model: Optional[str] = None
    # 응답 토큰 상한. 적응형 추론(thinking)이 이 예산을 함께 쓰므로 넉넉해야
    # 한다 — 16000 으로는 실제 호출에서 추론이 예산을 다 먹고 JSON 이 중간에
    # 잘렸다(stop_reason=max_tokens, 출력 16000 토큰 중 본문 1,410자만 남음).
    # 구조화 JSON 블록을 끄면 입력이 짧아지는 만큼 모델이 추론을 더 하는 경향이
    # 있어 상한이 더 중요해진다. 본문 자체는 2~3K자면 충분하다.
    max_tokens: int = 32000
    effort: str = "high"

    @property
    def n_horizon_buckets(self) -> int:
        return max(int(round(self.horizon_s)), 1)

    def actor_cap(self, n_actors: int) -> int:
        """이 장면에서 수록할 차량 수.

        고정값을 주면 그대로 쓴다. None 이면 **관측된 전부**를 담되 상한을 둔다 —
        관측을 늘려 놓고 담는 수를 그대로 두면 새로 보인 차량이 기존 차량을
        밀어내기만 한다 (카메라 6대로 늘렸을 때 26대 중 11대가 생략됐고, 그
        여파로 사고 당사자 쌍이 접근쌍 목록에서 사라졌다).
        """
        if self.max_actors is not None:
            return self.max_actors
        return max(self.actor_cap_floor, min(n_actors, self.actor_cap_ceiling))

    def pair_cap(self, n_actors: int) -> int:
        """이 장면에서 수록할 접근 쌍 수. 차량 수에 비례해 늘린다."""
        if self.max_closing_pairs is not None:
            return self.max_closing_pairs
        want = int(round(self.pair_cap_per_actor * n_actors))
        return max(self.pair_cap_floor, min(want, self.pair_cap_ceiling))

    @property
    def must_include_contact_s(self) -> float:
        """상한을 넘겨도 담아야 하는 접촉 임박 기준 [s]."""
        if self.closing_must_include_contact_s is not None:
            return self.closing_must_include_contact_s
        return self.horizon_s * self.closing_must_include_contact_frac


def _fmt_t(t: float) -> str:
    """파일명·라벨용 시각 표기. 정수면 소수점을 붙이지 않는다."""
    if abs(t - round(t)) < 1e-6:
        return str(int(round(t)))
    return f"{t:g}"


@dataclass
class TimeWindow:
    index: int
    t_start: float
    t_end: float
    snapshots: List[SceneSnapshot]

    @property
    def label(self) -> str:
        return f"{_fmt_t(self.t_start)}-{_fmt_t(self.t_end)}"

    @property
    def last(self) -> SceneSnapshot:
        return self.snapshots[-1]

    def actors_at_end(self) -> List[ActorState]:
        return self.last.actors


# ---------------------------------------------------------------- 윈도우 분할


def build_windows(
    snapshots: Sequence[SceneSnapshot],
    cfg: Optional[WindowConfig] = None,
    collision_time_s: Optional[float] = None,
) -> Tuple[List[TimeWindow], Dict[str, object]]:
    """스냅샷 시퀀스를 관측 윈도우로 자른다.

    기본은 **고정 길이**(`warmup=False`) sliding window다. `window_s` 길이의
    창을 만든 뒤 `stride_s`만큼 이동한다.

        window_s=5, stride=1 → 0-1, 0-2, 0-3, 0-4, 0-5, 1-6, 2-7, …

    `warmup=True`를 명시하면 데이터 시작부터 창이 자라는 expanding-prefix
    모드(0-1, 0-2, …, 0-5, 1-6, …)로 전환된다.

    `full_window=True`면 마지막에 **데이터 전체 구간** 창을 하나 더 붙인다
    (9초 데이터면 `0-9`). 이동 창은 `window_s` 로 관측을 잘라 내므로 "관측을 더
    줬으면 맞혔을까" 를 답할 수 없는데, 전체 구간 창이 그 상한을 준다. 데이터가
    `window_s` 보다 짧아 마지막 누적 창이 이미 전체 구간이면 넣지 않는다.

    Returns: (윈도우 목록, 진단 정보)
    """
    cfg = cfg or WindowConfig()
    snaps = sorted(snapshots, key=lambda s: s.t)
    info: Dict[str, object] = {
        "n_snapshots": len(snaps),
        "dropped_incomplete": 0,
        "dropped_after_collision": 0,
        "warmup": bool(cfg.warmup),
    }
    if not snaps:
        return ([], info)

    t0, t1 = snaps[0].t, snaps[-1].t
    info["data_range_s"] = [round(t0, 3), round(t1, 3)]

    out: List[TimeWindow] = []
    idx = 0
    eps = 1e-6
    # 창의 **끝**을 stride 간격으로 밀고, 시작은 최대 길이만큼 뒤로 잡는다.
    # warmup 이면 데이터 시작 앞으로는 못 가므로 초반 창이 짧아진다.
    end = t0 + cfg.stride_s if cfg.warmup else t0 + cfg.window_s
    while end <= t1 + eps:
        start = end - cfg.window_s
        if start < t0 - eps:
            if not cfg.warmup:
                end += cfg.stride_s
                continue
            start = t0
        inside = [s for s in snaps if start - eps <= s.t <= end + eps]
        # 창 끝에 실제 스냅샷이 있어야 한다 — 마지막 관측 시점이 질문의 기준이다
        if cfg.require_full_window and (
            not inside or inside[-1].t < end - eps
        ):
            info["dropped_incomplete"] = int(info["dropped_incomplete"]) + 1
            end += cfg.stride_s
            continue
        if len(inside) < 2:
            info["dropped_incomplete"] = int(info["dropped_incomplete"]) + 1
            end += cfg.stride_s
            continue
        # 충돌이 관측 구간 안에서 이미 일어났으면 '예측' 문제가 아니다
        if (
            cfg.drop_windows_after_collision
            and collision_time_s is not None
            and collision_time_s <= end + eps
        ):
            info["dropped_after_collision"] = (
                int(info["dropped_after_collision"]) + 1
            )
            end += cfg.stride_s
            continue

        out.append(TimeWindow(index=idx, t_start=start, t_end=end, snapshots=inside))
        idx += 1
        end += cfg.stride_s

    info["full_window"] = False
    if cfg.full_window:
        # 이미 같은 구간의 창이 있으면 넣지 않는다 (데이터가 window_s 보다 짧은 경우).
        dup = any(
            abs(w.t_start - t0) < eps and abs(w.t_end - t1) < eps for w in out
        )
        after = (
            cfg.drop_windows_after_collision
            and collision_time_s is not None
            and collision_time_s <= t1 + eps
        )
        if dup:
            info["full_window_skipped"] = "duplicate"
        elif after:
            # 관측 구간 안에서 이미 사고가 났으면 '예측' 문제가 아니다 —
            # 이동 창과 같은 규칙을 적용한다.
            info["full_window_skipped"] = "collision_inside"
        elif len(snaps) < 2:
            info["full_window_skipped"] = "too_few_snapshots"
        else:
            out.append(
                TimeWindow(index=idx, t_start=t0, t_end=t1, snapshots=list(snaps))
            )
            info["full_window"] = True

    info["n_windows"] = len(out)
    return (out, info)


# ---------------------------------------------------------------- 접근 쌍


@dataclass
class ClosingPair:
    """윈도우 동안 거리가 줄어든 차량 쌍."""

    a: str
    b: str
    d_start: float
    d_end: float
    closing_rate_mps: float
    linear_contact_s: Optional[float]  # 현재 접근율이 유지될 때 접촉까지 [s]
    same_road: bool
    note: str = ""


def closing_pairs(win: TimeWindow, cfg: WindowConfig) -> List[ClosingPair]:
    """윈도우 동안 서로 접근한 차량 쌍을 거리 변화로 추출.

    선형 외삽값은 **관측된 거리 변화율을 그대로 연장한 것**이며 실제 경로
    예측이 아니다. 회전·제동을 반영하지 않으므로 참고값으로만 쓴다.
    """
    first, last = win.snapshots[0], win.snapshots[-1]
    pos_first = {a.actor_id: a.world_xy for a in first.actors}
    span = last.t - first.t
    if span <= 1e-6:
        return []

    road_of = {
        a.actor_id: (a.placement.road_id if a.placement else None)
        for a in last.actors
    }
    out: List[ClosingPair] = []
    acts = last.actors
    for i, a in enumerate(acts):
        for b in acts[i + 1 :]:
            if a.actor_id not in pos_first or b.actor_id not in pos_first:
                continue
            d0 = math.dist(pos_first[a.actor_id], pos_first[b.actor_id])
            d1 = math.dist(a.world_xy, b.world_xy)
            if d1 >= d0 - 0.5:
                continue  # 접근하지 않음
            rate = (d0 - d1) / span
            if rate < 0.3:
                continue
            contact = d1 / rate if rate > 1e-6 else None
            if contact is not None and contact > 3.0 * cfg.horizon_s:
                continue  # 예측 구간과 무관하게 먼 미래
            out.append(
                ClosingPair(
                    a=a.actor_id,
                    b=b.actor_id,
                    d_start=d0,
                    d_end=d1,
                    closing_rate_mps=rate,
                    linear_contact_s=contact,
                    same_road=(
                        road_of.get(a.actor_id) is not None
                        and road_of.get(a.actor_id) == road_of.get(b.actor_id)
                    ),
                )
            )
    out.sort(key=lambda p: (p.linear_contact_s if p.linear_contact_s else 1e9))
    cap = cfg.pair_cap(len(acts))
    if len(out) <= cap:
        return out
    kept = out[:cap]
    # 상한을 넘겼더라도 접촉이 임박한 쌍은 버리지 않는다. 순위로만 자르면
    # 새로 보이게 된 무관한 쌍이 임박한 쌍을 밀어낼 수 있고, 실제로 그렇게
    # 사고 당사자 쌍이 payload 에서 빠졌다.
    thr = cfg.must_include_contact_s
    if thr > 0:
        urgent = [
            p
            for p in out[cap:]
            if p.linear_contact_s is not None and p.linear_contact_s <= thr
        ]
        kept = kept + urgent
    return kept


# ---------------------------------------------------------------- 렌더링


def _history_times(win: TimeWindow, cfg: WindowConfig) -> List[float]:
    """이력 표시에 쓸 시각 목록. 마지막 시점은 항상 포함."""
    ts = [s.t for s in win.snapshots]
    if cfg.history_stride_s <= 0:
        return ts
    picked: List[float] = []
    for t in ts:
        if not picked or t - picked[-1] >= cfg.history_stride_s - 1e-6:
            picked.append(t)
    if ts[-1] not in picked:
        picked.append(ts[-1])
    return picked


def _actor_row(a: ActorState, lang: str = "ko") -> str:
    t = i18n.tx(lang)
    pos = f"({a.world_xy[0]:7.1f},{a.world_xy[1]:7.1f})"
    spd = (
        f"{a.speed_mps * MPS_TO_KPH:5.0f}km/h"
        if a.speed_mps is not None
        else t["row_speed_unknown"]
    )
    if a.placement is not None:
        p = a.placement
        lane = (
            t["row_lane"].format(idx=p.lane_index, count=p.lane_count)
            if p.lane_index
            else t["row_lane_unknown"]
        )
        road = f"{p.road_name} {i18n.label(p.direction_label, lang)} {lane}"
        junc = (
            t["row_junction"].format(d=p.dist_to_next_junction_m)
            if p.dist_to_next_junction_m is not None
            and p.dist_to_next_junction_m < 200
            else ""
        )
    else:
        road, junc = t["no_road"], ""
    return f"{pos} {spd}  {road}{junc}  {i18n.label(a.maneuver, lang)}"


def render_window_text(
    win: TimeWindow, scfg: SerializeConfig, cfg: WindowConfig
) -> str:
    """윈도우 전체를 자연어로 재려한다.

    구성: 개요 → 관측자/인프라 → 차량별 시간 경과 → 접근 중인 쌍 →
          마지막 시점 상세(상호작용·BEV 포함)
    """
    lang = scfg.language
    t = i18n.tx(lang)
    last = win.last
    lines: List[str] = []
    lines.append(
        t["win_header"].format(
            label=win.label,
            t0=_fmt_t(win.t_start),
            t1=_fmt_t(win.t_end),
            n=len(win.snapshots),
        )
    )
    if last.scenario is not None:
        sc = last.scenario
        bits = []
        if sc.source:
            bits.append(t["bg_source"].format(v=sc.source))
        if sc.town:
            bits.append(t["bg_town"].format(v=sc.town))
        for k, v in sc.available.items():
            if v:
                bits.append(f"{k}={v}")
        if bits:
            lines.append(t["bg_prefix"] + ", ".join(bits))
    if last.map_context.get("map_source"):
        lines.append(
            t["map_note"].format(
                src=i18n.map_source(last.map_context["map_source"], lang)
            )
        )

    # 중요도 상위 차량만 (토큰 예산)
    ranked = sorted(last.actors, key=lambda a: -risk_score(a, last.interactions))
    kept = ranked[: cfg.actor_cap(len(last.actors))]
    dropped = len(ranked) - len(kept)
    kept_ids = {a.actor_id for a in kept}
    egos = [a for a in kept if a.kind == "ego"]
    others = [a for a in kept if a.kind != "ego"]
    counts = t["win_counts"].format(ego=len(egos), other=len(others))
    if last.infrastructure:
        counts += t["counts_infra"].format(n=len(last.infrastructure))
    if dropped:
        counts += t["win_counts_dropped"].format(n=dropped)
    lines.append(counts)

    if last.infrastructure:
        lines.append(t["sec_infra"])
        for si in last.infrastructure:
            lines.append(
                t["win_infra_row"].format(
                    sid=si.infra_id, h=si.height_m, n=si.n_observed
                )
            )

    # ---- 관측 관계 (누가 누구를 인지했는가)
    if cfg.include_perception:
        perc = perception_summary(win, cfg, lang)
        lines.append(t["sec_perception"])
        for o in perc["observers"]:
            kind = t["kind_infra" if o["kind"] == "infrastructure" else "kind_vehicle"]
            row = t["perc_row"].format(
                oid=o["id"], kind=kind, n=o["n_observed_now"]
            )
            if o["n_observed_now"] == 0:
                row = f"- {o['id']}({kind}): " + t["perc_none"].lstrip("- ")
            elif o["only_this_observer"]:
                row += t["perc_only"].format(
                    ids=", ".join(o["only_this_observer"])
                )
            # 관측자 이름('ego_vehicle')과 그 본체의 액터 id('EGO_ego_vehicle')는
            # 다른 문자열이다. 매핑을 주지 않으면 답변에 관측자 이름을 써서
            # 차량 지목이 전부 오답 처리된다 (실제 호출에서 그렇게 나왔다).
            if o.get("self_actor_id"):
                row += t["perc_self"].format(aid=o["self_actor_id"])
            if o.get("n_cameras"):
                row += t["perc_cams"].format(
                    n=o["n_cameras"], cov=o.get("azimuth_coverage_deg")
                )
            lines.append(row)

    # ---- 차량별 시간 경과
    times = _history_times(win, cfg)
    by_actor: Dict[str, List[Tuple[float, ActorState]]] = {}
    for sn in win.snapshots:
        if sn.t not in times:
            continue
        for a in sn.actors:
            if a.actor_id in kept_ids:
                by_actor.setdefault(a.actor_id, []).append((sn.t, a))

    lines.append(
        t["sec_history"].format(
            t0=_fmt_t(times[0]),
            t1=_fmt_t(times[-1]),
            stride=cfg.history_stride_s,
        )
    )
    for a in kept:
        hist = by_actor.get(a.actor_id, [])
        if not hist:
            continue
        role = t["role_ego"] if a.kind == "ego" else a.cls
        obs = ",".join(a.observed_by) if a.observed_by else "-"
        head = t["hist_head"].format(aid=a.actor_id, role=role, obs=obs)
        if a.kind != "ego":
            head += t["hist_conf"].format(conf=a.confidence)
            if a.observed_range_m is not None:
                head += t["hist_range"].format(r=a.observed_range_m)
        lines.append(head)
        for tt, st in hist:
            lines.append(f"   t={_fmt_t(tt):>5s}  {_actor_row(st, lang)}")
        if len(hist) < len(times):
            lines.append(
                t["hist_partial"].format(seen=len(hist), total=len(times))
            )

    # 윈도우 도중 사라진 차량 — 놓친 것인지 이탈인지 LLM 이 알아야 한다
    end_ids = {a.actor_id for a in last.actors}
    seen_all: Set[str] = {a.actor_id for sn in win.snapshots for a in sn.actors}
    gone = sorted(seen_all - end_ids)
    if gone:
        lines.append(t["win_gone"].format(ids=", ".join(gone)))

    # ---- 접근 중인 쌍
    pairs = closing_pairs(win, cfg)
    lines.append(t["sec_pairs"])
    if not pairs:
        lines.append(t["none_pairs"])
    for pr in pairs:
        eta = (
            f"{pr.linear_contact_s:.1f}s"
            if pr.linear_contact_s is not None
            else "-"
        )
        row = t["pair_row"].format(
            a=pr.a,
            b=pr.b,
            d0=pr.d_start,
            d1=pr.d_end,
            rate=pr.closing_rate_mps,
            eta=eta,
        )
        if pr.same_road:
            row += t["pair_same_road"]
        lines.append(row + ")")
    if pairs:
        lines.append(t["pair_note"])

    # ---- 마지막 시점 상세
    lines.append(t["sec_last"].format(t=_fmt_t(win.t_end)))
    detail_cfg = SerializeConfig(
        language=scfg.language,
        max_actors=cfg.actor_cap(len(win.last.actors)),
        include_bev_ascii=cfg.include_bev_last,
        bev_range_m=scfg.bev_range_m,
        round_digits=scfg.round_digits,
        redact_scenario_id=scfg.redact_scenario_id,
    )
    body = to_text(last, detail_cfg)
    # 중복 헤더 제거 (윈도우 헤더가 이미 있다)
    body_lines = body.split("\n")
    trimmed = [L for L in body_lines if not L.startswith("# ")]
    lines.append("\n".join(trimmed).strip())

    return "\n".join(lines)


def compact_window_json(
    win: TimeWindow, scfg: SerializeConfig, cfg: WindowConfig
) -> dict:
    """Geometry-first window representation for the compact payload profile.

    This retains the inputs needed to compare the LLM with the selected trajectory
    predictor: observed actor history and the predictor's numerical future paths.
    Derived hazard summaries are intentionally absent so the language model cannot
    turn a candidate conflict (for example, a small TTC) into an accident by wording
    alone.
    """
    detail_cfg = SerializeConfig(
        language=scfg.language,
        max_actors=cfg.actor_cap(len(win.last.actors)),
        include_bev_ascii=False,
        bev_range_m=scfg.bev_range_m,
        round_digits=scfg.round_digits,
        redact_scenario_id=scfg.redact_scenario_id,
    )
    last_json = to_json(win.last, detail_cfg)
    current_by_id = {a["id"]: a for a in last_json["actors"]}
    kept_ids = set(current_by_id)
    times = set(_history_times(win, cfg))
    histories: Dict[str, List[list]] = {aid: [] for aid in kept_ids}
    for snap in win.snapshots:
        if snap.t not in times:
            continue
        for actor in snap.actors:
            if actor.actor_id not in kept_ids:
                continue
            # Column names are declared once below instead of repeated for every
            # sample.  Observer membership at each historical instant was a major
            # source of payload growth and is not trajectory geometry; the current
            # observer set and quality remain in current_state.
            row = [
                round(snap.t, 3),
                round(actor.world_xy[0], 1),
                round(actor.world_xy[1], 1),
                None
                if actor.speed_mps is None
                else round(actor.speed_mps * MPS_TO_KPH, 1),
            ]
            histories[actor.actor_id].append(row)

    actors: List[dict] = []
    for aid, source in current_by_id.items():
        road = source.get("road")
        current = [
            *source["position_enu_m"],
            source["heading_deg"],
            source["speed_kph"],
            source["accel_mps2"],
            source["maneuver"],
            None if road is None else road["name"],
            None if road is None else road["direction"],
            None if road is None else road["direction_confident"],
            None if road is None else road["lane"],
            None if road is None else road["lane_count"],
            None if road is None else road["lateral_offset_m"],
            source["observed_by"],
            source["confidence"],
            source["position_quality"],
            source["observed_range_m"],
            source["track_age_s"],
        ]
        futures = [
            [
                path["maneuver"],
                path["probability"],
                path["to_roads"],
                path["waypoints_1s_enu_m"],
                path["truncated"],
            ]
            for path in source["predicted_paths"]
        ]
        actors.append(
            {
                "id": aid,
                "role": source["role"],
                "class": source["class"],
                "history": histories.get(aid, []),
                "current_state": current,
                # Numerical 1-second future geometry from WaypointNet (or the
                # predictor recorded in the manifest), not a prose risk label.
                "future_paths": futures,
            }
        )

    original_by_id = {a.actor_id: a for a in win.last.actors}
    selected_actors = [original_by_id[aid] for aid in current_by_id]
    selected_by_id = {a.actor_id: a for a in selected_actors}
    candidate_pool = swept_pair_clearances(
        selected_actors,
        horizon_s=cfg.horizon_s,
        sample_dt_s=cfg.swept_sample_dt_s,
        contact_margin_m=cfg.swept_contact_margin_m,
        exclude_touching_now=cfg.swept_exclude_touching_now,
        exclude_static_pairs=cfg.swept_exclude_static_pairs,
    )[: max(0, cfg.swept_candidate_pool_cap)]
    if cfg.pair_reranker_model is not None:
        # Import lazily to keep the base payload generator independent of torch
        # and to avoid a module-level dependency cycle.
        from .pair_reranker_v2 import pair_features_for_model

        scored = []
        actor_by_id = {a.actor_id: a for a in selected_actors}
        for pair in candidate_pool:
            score = float(cfg.pair_reranker_model.predict_features(
                pair_features_for_model(
                    cfg.pair_reranker_model,
                    actor_by_id[pair.actor_a], actor_by_id[pair.actor_b],
                    pair, cfg.horizon_s,
                )
            ))
            scored.append((score, pair))
        scored.sort(key=lambda item: (
            -item[0], item[1].minimum_clearance_m,
            item[1].time_after_observation_s,
            item[1].actor_a, item[1].actor_b,
        ))
        if getattr(cfg.pair_reranker_model, "uses_threshold_gate", False):
            scored = [item for item in scored
                      if item[0] >= cfg.pair_reranker_model.threshold]
        swept = [pair for _, pair in scored[: max(0, cfg.swept_pair_cap)]]
    else:
        swept = candidate_pool[: max(0, cfg.swept_pair_cap)]

    map_ctx = win.last.map_context or {}
    end_ids = {a.actor_id for a in win.last.actors}
    seen_ids = {a.actor_id for s in win.snapshots for a in s.actors}

    def pair_observation(pair):
        a, b = selected_by_id[pair.actor_a], selected_by_id[pair.actor_b]
        a_sees_b, b_sees_a = observes_actor(a, b), observes_actor(b, a)
        return [
            a_sees_b, b_sees_a,
            a_sees_b is True and b_sees_a is True,
            a_sees_b is not None, b_sees_a is not None,
        ]

    return {
        "representation": "compact_geometry_v3",
        "observation_window": {
            "label": win.label,
            "t_start_s": round(win.t_start, 3),
            "t_end_s": round(win.t_end, 3),
            "history_sample_times_s": sorted(times),
        },
        "coordinate_convention": {
            "frame": "local_ENU",
            "unit": "metre",
            "future_waypoint_start_s": 1,
            "future_waypoint_step_s": 1,
        },
        "history_columns": ["t_s", "e_m", "n_m", "speed_kph"],
        "current_state_columns": [
            "e_m", "n_m", "heading_deg", "speed_kph", "accel_mps2",
            "maneuver", "road", "direction", "direction_confident", "lane",
            "lane_count", "lateral_offset_m", "observed_by",
            "existence_confidence", "position_quality", "range_m", "track_age_s",
        ],
        "future_path_columns": [
            "maneuver", "probability", "to_roads", "waypoints_1s_enu_m",
            "truncated",
        ],
        "class_footprints_m": {
            cls: [round(length, 2), round(width, 2)]
            for cls in sorted({a.cls for a in selected_actors})
            for length, width in [
                actor_footprint_m(next(a for a in selected_actors if a.cls == cls))
            ]
        },
        "predicted_pair_method": {
            "source": "supplied_future_paths_only",
            "footprint": "oriented_rectangle",
            "sample_dt_s": cfg.swept_sample_dt_s,
            "contact_margin_m": cfg.swept_contact_margin_m,
            "exclude_touching_at_observation": cfg.swept_exclude_touching_now,
            "exclude_static_static": cfg.swept_exclude_static_pairs,
            "internal_candidate_pool_cap": cfg.swept_candidate_pool_cap,
            "pairs_serialized_cap": cfg.swept_pair_cap,
            "selection": (
                "v2_pair_reranker"
                if cfg.pair_reranker_model is not None
                else "minimum_clearance"
            ),
            "pair_reranker": cfg.pair_reranker_note or None,
        },
        "predicted_pair_columns": [
            "actor_a", "actor_b", "minimum_clearance_m",
            "time_after_observation_s", "interval_index", "predicted_contact",
            "joint_path_probability", "clearance_at_observation_m",
            "a_observes_b", "b_observes_a", "mutually_observed",
            "a_observation_available", "b_observation_available",
        ],
        "predicted_closest_pairs": [
            [
                pair.actor_a,
                pair.actor_b,
                round(pair.minimum_clearance_m, 2),
                round(pair.time_after_observation_s, 1),
                pair.interval_index,
                pair.predicted_contact,
                round(pair.joint_path_probability, 3),
                round(pair.clearance_at_observation_m, 2),
                *pair_observation(pair),
            ]
            for pair in swept
        ],
        "map_conventions": {
            "source": i18n.map_source(map_ctx.get("map_source"), scfg.language),
            "drive_side": map_ctx.get("drive_side", "right"),
            "lane_numbering": map_ctx.get("lane_numbering", "from_median"),
            "lane_width_m": map_ctx.get("lane_width_m"),
        },
        "actor_count_total": len(win.last.actors),
        "actors_omitted": last_json["actors_omitted"],
        "actors": actors,
        "actors_seen_but_absent_at_end": sorted(seen_ids - end_ids),
        "infrastructure_columns": [
            "id", "e_m", "n_m", "observations_contributed"
        ],
        "infrastructure": [
            [
                item["id"],
                *item["position_enu_m"],
                item["observations_contributed"],
            ]
            for item in last_json["infrastructure"]
        ],
    }


def render_window_input(
    win: TimeWindow, scfg: SerializeConfig, cfg: WindowConfig
) -> str:
    """Render the selected payload profile's primary user-data block."""
    if cfg.payload_profile == "standard":
        return render_window_text(win, scfg, cfg)
    if cfg.payload_profile == "compact":
        return json.dumps(
            compact_window_json(win, scfg, cfg),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    raise ValueError(
        f"지원하지 않는 payload_profile: {cfg.payload_profile!r} "
        "(가능: standard, compact)"
    )


def perception_summary(
    win: TimeWindow, cfg: WindowConfig, lang: str = "ko"
) -> dict:
    """관측자 관점 요약 — 어느 관측자가 어떤 차량을 인지했는가.

    액터별 `observed_by` 를 역변환한 것이다. 파생 가능한 정보를 중복해 담는
    이유는, 이 역변환이 V2X 추론의 핵심인데 15개 객체에 흩어진 관계를 모델이
    직접 뒤집으면 자주 틀리기 때문이다. 특히 `only_this_observer` 는 그
    관측자가 **혼자** 본 차량, 즉 다른 관측자의 사각지대를 메운 몫이다.

    `observed_now` 에서 관측자 **자기 자신**은 뺀다 ("내가 무엇을 보는가"이므로).
    """
    last = win.last
    mc = last.map_context or {}
    vehicle_ids = list(mc.get("observer_ids") or [])
    infra_ids = list(mc.get("infrastructure_ids") or [])

    # 마지막 시각: 관측자 → 본 액터
    now: Dict[str, List[str]] = {}
    seen_by_count: Dict[str, int] = {}
    for a in last.actors:
        seen_by_count[a.actor_id] = len(a.observed_by)
        for obs in a.observed_by:
            now.setdefault(obs, []).append(a.actor_id)

    # 윈도우 전체: 관측자 → 한 번이라도 본 액터
    ever: Dict[str, Set[str]] = {}
    for s in win.snapshots:
        for a in s.actors:
            for obs in a.observed_by:
                ever.setdefault(obs, set()).add(a.actor_id)

    cams = mc.get("observer_cameras") or {}
    observers: List[dict] = []
    for oid in vehicle_ids + infra_ids:
        self_aid = f"EGO_{oid}"
        has_self = any(a.actor_id == self_aid for a in last.actors)
        seen = [x for x in now.get(oid, []) if x != self_aid]
        row = {
            "id": oid,
            "kind": "infrastructure" if oid in infra_ids else "vehicle",
            "self_actor_id": self_aid if has_self else None,
            "n_observed_now": len(seen),
            "observed_now": seen,
            # 이 관측자만 본 차량 = 다른 관측자의 사각지대를 메운 몫
            "only_this_observer": [
                x for x in seen if seen_by_count.get(x) == 1
            ],
            "n_observed_in_window": len(
                {x for x in ever.get(oid, set()) if x != self_aid}
            ),
        }
        # 카메라 구성. "관측되지 않았다"를 해석하는 데 필요하다 — 전방 1대뿐인
        # 관측자가 뒤차를 못 보는 것은 당연하고, 360° 를 덮는 관측자가 못 보는
        # 것은 가려졌다는 뜻이다.
        cc = cams.get(oid)
        if cc:
            row["n_cameras"] = cc.get("n_cameras")
            row["azimuth_coverage_deg"] = cc.get("azimuth_coverage_deg")
        observers.append(row)
    return {
        "observers": observers,
        "note": i18n.tx(lang)["perception_note"],
    }


def window_json(
    win: TimeWindow, scfg: SerializeConfig, cfg: WindowConfig
) -> dict:
    """윈도우의 구조화 표현. history_mode='full' 이면 전 스냅샷 JSON 포함."""
    detail_cfg = SerializeConfig(
        language=scfg.language,
        max_actors=cfg.actor_cap(len(win.last.actors)),
        include_bev_ascii=False,
        bev_range_m=scfg.bev_range_m,
        round_digits=scfg.round_digits,
        redact_scenario_id=scfg.redact_scenario_id,
    )
    out: dict = {
        "window": {
            "label": win.label,
            "t_start_s": round(win.t_start, 3),
            "t_end_s": round(win.t_end, 3),
            "n_snapshots": len(win.snapshots),
            "snapshot_times_s": [round(s.t, 3) for s in win.snapshots],
        },
        "last_snapshot": to_json(win.last, detail_cfg),
        "closing_pairs": [
            {
                "a": p.a,
                "b": p.b,
                "distance_start_m": round(p.d_start, 2),
                "distance_end_m": round(p.d_end, 2),
                "closing_rate_mps": round(p.closing_rate_mps, 2),
                "linear_contact_s": None
                if p.linear_contact_s is None
                else round(p.linear_contact_s, 2),
                "same_road": p.same_road,
            }
            for p in closing_pairs(win, cfg)
        ],
    }
    if cfg.include_perception:
        out["perception"] = perception_summary(win, cfg, scfg.language)

    if cfg.history_mode == "full":
        out["snapshots"] = [to_json(s, detail_cfg) for s in win.snapshots]
    else:
        # 궤적만 압축해 담는다
        traj: Dict[str, List[dict]] = {}
        times = set(_history_times(win, cfg))
        for s in win.snapshots:
            if s.t not in times:
                continue
            for a in s.actors:
                row = {
                    "t_s": round(s.t, 3),
                    "pos_enu_m": [
                        round(a.world_xy[0], 1),
                        round(a.world_xy[1], 1),
                    ],
                    "speed_kph": None
                    if a.speed_mps is None
                    else round(a.speed_mps * MPS_TO_KPH, 0),
                    "lane": None if a.placement is None else a.placement.lane_index,
                    "road": None if a.placement is None else a.placement.road_name,
                    "maneuver": i18n.label(a.maneuver, scfg.language),
                }
                if cfg.trajectory_observation:
                    # 이 시각에 이 차량을 본 관측자들. 목록에 없는 관측자는 그
                    # 시각에 이 차량을 인지하지 못한 것이다 (사각지대·화각 밖).
                    row["observed_by"] = list(a.observed_by)
                    row["range_m"] = (
                        None
                        if a.observed_range_m is None
                        else round(a.observed_range_m, 0)
                    )
                    row["confidence"] = round(a.confidence, 2)
                traj.setdefault(a.actor_id, []).append(row)
        out["trajectories"] = traj
    return out


# ---------------------------------------------------------------- 질의


# 하위호환 별칭(한국어). 언어별 스키마는 i18n.prediction_schema(lang).
PREDICTION_SCHEMA: dict = i18n.prediction_schema("ko")


# 하위호환 별칭. 언어별 프롬프트는 i18n.system_prompt('accident', lang) 로 얻는다.
SYSTEM_PROMPT_ACCIDENT_KO = """당신은 협력형 자율주행(V2X) 교통 상황 분석 전문가입니다.
여러 대의 차량에 탑재된 카메라와 노변 인프라 센서가 수집한 관측을 융합하고 도로 지도와
정합한 교통 상황 시계열을 입력받아, **미래 구간별 사고 발생 가능성**을 판단합니다.

데이터 규약:
- 위치는 지역 ENU 평면 좌표 [m], e=동쪽, n=북쪽입니다. 방위각은 진북 기준 시계방향.
- 차선 번호는 1이 중앙선쪽입니다.
- 위치·속도는 단안 카메라 추정값이므로 오차가 있습니다. 관측거리 40m 초과 또는
  position_quality 0.5 미만 객체는 위치 불확실성이 큽니다.
- confidence(존재 확신도)와 position_quality(위치 정확도)는 다른 값입니다.
  존재는 확실한데 위치만 부정확한 경우가 있습니다.
- '속도 미확정'은 정지를 뜻하지 않습니다. 추적 이력이 짧아 판정을 보류한 것입니다.
- maneuver 가 '차선유지'인 것은 차선변경이 없었다는 뜻이 아니라, 관측 조건상
  판정 근거가 부족하다는 뜻일 수 있습니다.
- predicted_paths 의 확률은 도로 구조와 차선 위치에서 유도한 **사전확률**입니다.
  방향지시등·제동등을 관측한 값이 아니므로 회전 의도를 단정하지 마십시오.
- '선형연장 접촉까지' 값은 관측된 거리 변화율을 그대로 늘린 참고값입니다.
- infrastructure 는 고정 센서이며 교통 참여자가 아닙니다. 정차 차량으로 보지 마십시오.
- map_source 가 '궤적 합성'이면 도로지도가 통행 궤적에서 역추정된 것입니다.
  차량이 지나가지 않은 차선·도로는 지도에 없고, 차선 수는 하한입니다.
- 이 데이터에는 **신호등 상태, 정지선, 차선 표시 종류(실선/점선)가 없습니다.**
  신호 준수 여부를 전제로 판단하지 말고, 필요하면 그 한계를 밝히십시오.

판단 원칙:
- 데이터에 등장한 actor id 만 사용하고, 없는 차량을 만들지 마십시오.
- 근거는 관측된 기하·운동 정보(거리, 접근율, 차선, 교차로 도달시간, 헤드웨이,
  TTC)에 두십시오. 추측을 사실처럼 쓰지 마십시오.
- 사고가 예상되지 않는 구간은 accident_expected=false 로 두고 involved_actor_ids 를
  비우십시오. 위험을 과장해 모든 구간을 사고로 표시하지 마십시오.
- 반대로, 관측된 상충이 명확하면(임박한 TTC, 교차로 동시 진입, 차선변경 경합)
  주저하지 말고 사고로 표시하고 관련 차량을 모두 나열하십시오."""


def build_question(
    win: TimeWindow, cfg: WindowConfig, lang: str = "ko"
) -> str:
    t = i18n.tx(lang)
    n = cfg.n_horizon_buckets
    t_end = win.t_end
    rows = [
        t["q_bucket_row"].format(
            k=k, lo=_fmt_t(t_end + k - 1), hi=_fmt_t(t_end + k)
        )
        for k in range(1, n + 1)
    ]
    return (
        t["q_intro"].format(
            # **설정값이 아니라 이 창의 실제 길이**를 쓴다. 누적 창(0-1, 0-2 …)과
            # 전체 구간 창은 `cfg.window_s` 와 길이가 다르므로, 설정값을 쓰면
            # "위 5초 관측 구간(t=0~1초)" 처럼 자기 모순인 문장이 나간다.
            win=_fmt_t(win.t_end - win.t_start),
            t0=_fmt_t(win.t_start),
            t1=_fmt_t(t_end),
            horizon=_fmt_t(cfg.horizon_s),
        )
        + "\n\n"
        + t["q_bucket_head"]
        + "\n"
        + "\n".join(rows)
        + "\n\n"
        + t["q_answer_all"].format(n=n)
        + "\n"
        + t["q_rule_negative"]
        + "\n"
        + t["q_rule_positive"]
        + "\n"
        + t["q_rule_ids"]
        + "\n"
        + t["q_rule_limits"]
    )


def build_window_payload(
    win: TimeWindow,
    scfg: Optional[SerializeConfig] = None,
    cfg: Optional[WindowConfig] = None,
) -> dict:
    """LLM API 요청 본문. 정답은 포함하지 않는다.

    형식은 `scfg.provider` ('claude' | 'openai' | 'gemini')가 정한다. 내용
    (브리핑·구조화 JSON·질문·프롬프트·스키마)은 provider 와 무관하게 같다.
    """
    scfg = scfg or SerializeConfig()
    cfg = cfg or WindowConfig()

    if cfg.payload_profile == "compact" and cfg.include_json_block:
        raise ValueError(
            "payload_profile='compact' 자체가 구조화 표현이므로 "
            "include_json_block=True 와 함께 쓸 수 없습니다"
        )
    blocks: List[str] = [render_window_input(win, scfg, cfg)]
    if cfg.include_json_block:
        blocks.append(
            i18n.tx(scfg.language)["json_label"]
            + "\n```json\n"
            + json.dumps(
                window_json(win, scfg, cfg), ensure_ascii=False, indent=1
            )
            + "\n```"
        )
    blocks.append(build_question(win, cfg, scfg.language))

    return providers.build_request(
        scfg.provider,
        system=i18n.system_prompt(
            "accident_compact"
            if cfg.payload_profile == "compact"
            else "accident",
            scfg.language,
        ),
        blocks=blocks,
        schema=i18n.prediction_schema(scfg.language, cfg.n_horizon_buckets),
        model=cfg.model or scfg.model,
        max_tokens=cfg.max_tokens,
        effort=cfg.effort,
        extra=scfg.provider_extra,
    )


# ---------------------------------------------------------------- 정답


def build_actor_lookup(
    win: TimeWindow, agent_carla_ids: Optional[Dict[str, int]] = None
) -> Dict[int, List[str]]:
    """CARLA(원본 트랙) id → 파이프라인 actor id 목록.

    - 관측차량(ego)의 신원은 **관측자 이름 ↔ CARLA actor id 대응이 정답**이다.
      흡수된 검출 id 로 역추적하면 근처를 지나던 다른 차량 id 가 섞일 수 있다.
    - 주변차량은 ActorState.source_track_ids 로 찾는다.
    - 융합이 실패해 같은 물리 차량이 ego 액터와 별도 주변차량으로 동시에
      나타날 수 있다. 그 경우 한 CARLA id 에 actor id 가 여러 개 매핑되며,
      채점에서는 그중 하나만 지목해도 맞은 것으로 본다.
    """
    ego_actor_ids: Set[str] = set()
    out: Dict[int, Set[str]] = {}
    if agent_carla_ids:
        for agent, cid in agent_carla_ids.items():
            if cid is None:
                continue
            aid = f"EGO_{agent}"
            ego_actor_ids.add(aid)
            out.setdefault(cid, set()).add(aid)

    for s in win.snapshots:
        for a in s.actors:
            if a.actor_id in ego_actor_ids or a.kind == "ego":
                continue  # ego 신원은 위에서 확정
            for tid in a.source_track_ids:
                out.setdefault(tid, set()).add(a.actor_id)
    return {k: sorted(v) for k, v in out.items()}


def bucket_of(win: TimeWindow, t: float) -> int:
    """시각 t 가 속한 구간 번호. 구간 k = (t_end + k-1, t_end + k].

    상한 포함이므로 `ceil` 이다. t 가 t_end 이하면 0 이하가 되어 지평 밖이다.
    """
    return math.ceil(t - win.t_end - 1e-9)


def credit_window(
    win: TimeWindow, cfg: WindowConfig, collision_time_s: Optional[float]
) -> dict:
    """'사고'로 답해도 정답인 구간 범위.

    사고 예측의 목적은 회피할 시간을 버는 것이므로, 실제 시점보다 조금 이른
    경보는 맞은 것으로 본다. 늦은 것은 오답이다 — 이미 일어난 사고를 예측이라
    부를 수 없다.

    `k_from` 은 `충돌시각 − early_credit_s` 가 속한 구간, `k_to` 는 충돌 시각이
    속한 구간이다. 둘 다 지평 [1, N] 으로 자른다. 충돌이 없거나 지평을 벗어나면
    빈 범위(`k_true=None`)를 돌려준다.
    """
    n = cfg.n_horizon_buckets
    x = max(0.0, float(cfg.early_credit_s))
    if collision_time_s is None:
        return {"early_credit_s": x, "k_true": None, "k_from": None, "k_to": None}
    k_true = bucket_of(win, collision_time_s)
    if k_true < 1 or k_true > n:
        # 충돌이 이 윈도우의 질문 지평 밖이다 (앞이거나 뒤).
        return {
            "early_credit_s": x,
            "k_true": k_true if 1 <= k_true <= n else None,
            "k_from": None,
            "k_to": None,
        }
    k_from = max(1, bucket_of(win, collision_time_s - x))
    return {
        "early_credit_s": x,
        "k_true": k_true,
        "k_from": min(k_from, k_true),
        "k_to": k_true,
    }


def window_ground_truth(
    win: TimeWindow,
    cfg: Optional[WindowConfig] = None,
    collision: Optional[CollisionTruth] = None,
    agent_carla_ids: Optional[Dict[str, int]] = None,
    scenario_id: str = "",
    scenario_split: str = "",
    # 데이터셋 분할 (`DeepAccident_mini` / `train` / `val`). 산출물이 어느
    # 데이터에서 나온 것인지 파일만 보고 알 수 있어야 한다.
    dataset_split: str = "",
    town: str = "",
    lang: str = "ko",
    # 원본 데이터의 마지막 시각. 이 시각을 넘는 구간은 **채점에서 제외**한다 —
    # 관측이 끝난 뒤의 '사고 없음' 은 확인된 사실이 아니라 질문 형식을 맞추려고
    # 채운 값이다. None 이면 전 구간을 채점한다(예전 동작).
    data_end_s: Optional[float] = None,
) -> dict:
    """윈도우 하나에 대한 채점용 정답.

    collision 이 None 이거나 occurred=False 면 전 구간 '사고 없음'이 정답이다.
    """
    cfg = cfg or WindowConfig()
    n = cfg.n_horizon_buckets
    lookup = build_actor_lookup(win, agent_carla_ids)

    occurred = bool(collision.occurred) if collision else False
    c_time = collision.time_s if collision else None
    c_ids = tuple(collision.carla_ids or ()) if collision else ()

    # 차량 단위로 정답을 구성한다. 융합 실패로 한 물리 차량이 여러 actor id 로
    # 나뉠 수 있으므로, 그중 하나만 지목해도 맞은 것으로 채점해야 공정하다.
    involved_vehicles: List[dict] = []
    for cid in c_ids:
        aids = lookup.get(cid, [])
        involved_vehicles.append(
            {
                "carla_id": cid,
                "actor_ids": aids,  # 이 중 하나만 지목해도 정답
                "observed_in_window": bool(aids),
            }
        )
    involved_actor_ids = sorted(
        {a for v in involved_vehicles for a in v["actor_ids"]}
    )
    unobserved = [v["carla_id"] for v in involved_vehicles if not v["actor_ids"]]

    expected: List[dict] = []
    for k in range(1, n + 1):
        lo = win.t_end + (k - 1)
        hi = win.t_end + k
        # 구간은 (lo, hi] — 하한은 배타적, 상한은 포함. 하한에 여유를 주면
        # 경계값이 두 구간에 동시에 걸린다.
        hit = (
            occurred
            and c_time is not None
            and c_time > lo + 1e-9
            and c_time <= hi + 1e-9
        )
        # 이 구간을 채점에 쓸 수 있는가.
        #   - 구간 전체가 원본 데이터 안에 있으면 확인된 사실이다.
        #   - 데이터 끝을 넘더라도 **충돌이 그 구간에 있으면** 답을 안다.
        #   - 그 밖(데이터가 끝난 뒤)은 '사고 없음'을 확인할 수 없다.
        if data_end_s is None:
            scorable, why = True, ""
        elif hi <= data_end_s + 1e-9:
            scorable, why = True, ""
        elif hit:
            scorable, why = True, "collision_known"
        else:
            scorable, why = False, "beyond_data_end"
        expected.append(
            {
                "k": k,
                "interval_s": f"({_fmt_t(lo)}, {_fmt_t(hi)}]",
                "interval_start_s": round(lo, 3),
                "interval_end_s": round(hi, 3),
                "accident_expected": hit,
                "scorable": scorable,
                "unscorable_reason": why if not scorable else "",
                "involved_vehicles": involved_vehicles if hit else [],
                "involved_actor_ids": involved_actor_ids if hit else [],
                "involved_carla_ids": list(c_ids) if hit else [],
                "unobserved_carla_ids": unobserved if hit else [],
                "all_involved_observed": (not unobserved) if hit else True,
            }
        )

    ttc = None if c_time is None else round(c_time - win.t_end, 3)
    return {
        # 조기 예측 인정 범위. 채점이 파일만으로 재현되도록 정답에 적어 둔다.
        "credit": credit_window(win, cfg, c_time if occurred else None),
        # 원본 데이터의 마지막 시각. 이 뒤의 구간은 채점에서 뺀다.
        "data_end_s": None if data_end_s is None else round(data_end_s, 3),
        "n_scorable": sum(1 for e in expected if e["scorable"]),
        "window": {
            "index": win.index,
            "label": win.label,
            "t_start_s": round(win.t_start, 3),
            "t_end_s": round(win.t_end, 3),
            "n_snapshots": len(win.snapshots),
        },
        "scenario": {
            "ref": scenario_ref(scenario_id),
            "id": scenario_id,
            # `split` 은 시나리오 유형(type1_subtype2_accident 등),
            # `dataset_split` 은 데이터셋 분할(mini/train/val)이다.
            "split": scenario_split,
            "dataset_split": dataset_split,
            "outcome": (
                "accident" if scenario_split.endswith("_accident")
                else ("normal" if scenario_split.endswith("_normal") else "")
            ),
            "town": town,
        },
        "config": {
            "window_s": cfg.window_s,
            "stride_s": cfg.stride_s,
            "horizon_s": cfg.horizon_s,
            "bucket_rule": i18n.tx(lang)["bucket_rule"],
        },
        "collision": collision.to_dict()
        if collision is not None
        else {"occurred": False},
        "time_to_collision_from_window_end_s": ttc,
        "expected": expected,
        "notes": _gt_notes(occurred, c_time, win, cfg, unobserved, lang),
    }


def _gt_notes(
    occurred: bool,
    c_time: Optional[float],
    win: TimeWindow,
    cfg: WindowConfig,
    unobserved: Sequence[int],
    lang: str = "ko",
) -> List[str]:
    t = i18n.tx(lang)
    notes: List[str] = []
    if not occurred:
        notes.append(t["gt_no_collision"])
        return notes
    if c_time is None:
        notes.append(t["gt_time_unknown"])
        return notes
    lead = c_time - win.t_end
    if lead > cfg.horizon_s:
        notes.append(
            t["gt_beyond_horizon"].format(horizon=cfg.horizon_s, lead=lead)
        )
    else:
        notes.append(t["gt_lead"].format(lead=lead))
    if unobserved:
        notes.append(t["gt_unobserved"].format(ids=list(unobserved)))
    return notes


# ---------------------------------------------------------------- 파일 출력


def write_window_set(
    snapshots: Sequence[SceneSnapshot],
    out_dir: str,
    scfg: Optional[SerializeConfig] = None,
    cfg: Optional[WindowConfig] = None,
    collision: Optional[CollisionTruth] = None,
    agent_carla_ids: Optional[Dict[str, int]] = None,
    scenario_id: str = "",
    scenario_split: str = "",
    # 데이터셋 분할 (`DeepAccident_mini` / `train` / `val`). 산출물이 어느
    # 데이터에서 나온 것인지 파일만 보고 알 수 있어야 한다.
    dataset_split: str = "",
    town: str = "",
    also_write_text: bool = True,
    # 경로 예측기 식별자. 규칙 기반과 학습 모델의 산출물을 파일만 보고 구별할 수
    # 있어야 한다 — 페이로드의 `예상경로` 절이 바뀌므로 섞이면 비교가 무의미해진다.
    # **번역하지 않는 기계 식별자**로 둔다 (manifest 는 언어별로 나가고, 산문을
    # 넣으면 영어 산출물에 한글이 섞인다 — 실제로 테스트가 잡았다).
    predictor_note: str = "",
    # 지도 출처와 미래 궤적 사용 여부. 사고 예측 실험에서는 지도 자체가 시나리오의
    # 미래 궤적으로 합성되면 온라인 예측 조건을 어기므로 manifest 에 반드시 남긴다.
    # 직접 호출한 오래된 코드도 출처 누락을 숨기지 않도록 기본값은 명시적인
    # `unspecified` 다.
    map_source: str = "unspecified",
    map_built_from_scenario_trajectories: Optional[bool] = None,
) -> dict:
    """윈도우별 payload/정답 파일과 manifest 를 쓴다.

    파일명
        llm_payload_<label>.json      LLM 요청 본문
        ground_truth_<label>.json     채점용 정답
        window_<label>.txt            사람이 읽는 사본 (also_write_text)
        manifest.json                 전체 목록·설정·요약
    """
    scfg = scfg or SerializeConfig()
    cfg = cfg or WindowConfig()
    os.makedirs(out_dir, exist_ok=True)
    # 이 함수가 만드는 파일만 지운다. 창 구성이 바뀌면(누적 창 도입, 지평 변경)
    # 예전 라벨의 파일이 남아 새 것과 섞이는데, 호출자가 디렉터리째 지우면
    # `responses/` 의 LLM 응답까지 날아간다 — 실제로 두 번 그렇게 잃었다.
    for pat in ("llm_payload_*.json", "ground_truth_*.json", "window_*.txt"):
        for stale in glob.glob(os.path.join(out_dir, pat)):
            os.remove(stale)

    c_time = collision.time_s if (collision and collision.occurred) else None
    windows, info = build_windows(snapshots, cfg, collision_time_s=c_time)
    # 원본 데이터의 마지막 시각 — 이 뒤의 구간은 정답을 확인할 수 없다
    data_end = max((s.t for s in snapshots), default=None)

    entries: List[dict] = []
    n_pos = 0
    for win in windows:
        payload = build_window_payload(win, scfg, cfg)
        gt = window_ground_truth(
            win,
            cfg,
            collision=collision,
            agent_carla_ids=agent_carla_ids,
            scenario_id=scenario_id,
            scenario_split=scenario_split,
            dataset_split=dataset_split,
            data_end_s=data_end,
            town=town,
            lang=scfg.language,
        )
        p_path = os.path.join(out_dir, f"llm_payload_{win.label}.json")
        g_path = os.path.join(out_dir, f"ground_truth_{win.label}.json")
        with open(p_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        # 입력 크기 파악용. 정확한 토큰 수는 messages.count_tokens 로 재야 하지만,
        # 윈도우 설정을 조절할 때의 상대 비교에는 문자 수가 충분하다.
        n_chars = providers.request_chars(scfg.provider, payload)
        with open(g_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=1)
        if also_write_text:
            with open(
                os.path.join(out_dir, f"window_{win.label}.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(render_window_input(win, scfg, cfg))

        any_pos = any(e["accident_expected"] for e in gt["expected"])
        n_pos += 1 if any_pos else 0
        entries.append(
            {
                "label": win.label,
                "index": win.index,
                "t_start_s": round(win.t_start, 3),
                "t_end_s": round(win.t_end, 3),
                "payload": os.path.basename(p_path),
                "ground_truth": os.path.basename(g_path),
                "has_accident_in_horizon": any_pos,
                "n_snapshots": len(win.snapshots),
                "input_chars": n_chars,
            }
        )

    manifest = {
        "scenario": {
            "ref": scenario_ref(scenario_id),
            "id": scenario_id,
            # `split` 은 시나리오 유형(type1_subtype2_accident 등),
            # `dataset_split` 은 데이터셋 분할(mini/train/val)이다.
            "split": scenario_split,
            "dataset_split": dataset_split,
            "outcome": (
                "accident" if scenario_split.endswith("_accident")
                else ("normal" if scenario_split.endswith("_normal") else "")
            ),
            "town": town,
        },
        "config": {
            "window_s": cfg.window_s,
            "stride_s": cfg.stride_s,
            "horizon_s": cfg.horizon_s,
            "snapshot_rate_hz": cfg.snapshot_rate_hz,
            "history_stride_s": cfg.history_stride_s,
            # 자동 산정이면 실제 적용값과 규칙을 함께 남긴다
            "max_actors": cfg.max_actors,
            "max_actors_effective": [
                cfg.actor_cap(len(w.last.actors)) for w in windows
            ],
            "max_closing_pairs": cfg.max_closing_pairs,
            "max_closing_pairs_effective": [
                cfg.pair_cap(len(w.last.actors)) for w in windows
            ],
            "history_mode": cfg.history_mode,
            "payload_profile": cfg.payload_profile,
            "swept_candidate_pool_cap": cfg.swept_candidate_pool_cap,
            "swept_pair_cap": cfg.swept_pair_cap,
            "swept_sample_dt_s": cfg.swept_sample_dt_s,
            "swept_contact_margin_m": cfg.swept_contact_margin_m,
            "swept_exclude_touching_now": cfg.swept_exclude_touching_now,
            "swept_exclude_static_pairs": cfg.swept_exclude_static_pairs,
            "pair_reranker": cfg.pair_reranker_note or None,
            "warmup": cfg.warmup,
            "full_window": cfg.full_window,
            "predictor": predictor_note or "rule",
            "map_source": map_source or "unspecified",
            "map_built_from_scenario_trajectories": (
                map_built_from_scenario_trajectories
            ),
            "language": scfg.language,
            # payload 파일은 **요청 본문 그대로**이므로 provider·endpoint 같은
            # 메타정보는 여기에만 적는다. Gemini 는 모델이 URL 경로에 들어가
            # 본문에 없으므로 특히 이 기록이 필요하다.
            "provider": providers.normalize_provider(scfg.provider),
            "model": providers.resolve_model(
                scfg.provider, cfg.model or scfg.model
            ),
            "endpoint": providers.endpoint_for(
                scfg.provider, cfg.model or scfg.model
            ),
            "bucket_rule": i18n.tx(scfg.language)["bucket_rule"],
        },
        "collision": collision.to_dict()
        if collision is not None
        else {"occurred": False},
        "windows": entries,
        "summary": {
            "n_windows": len(entries),
            "n_windows_with_accident": n_pos,
            "n_windows_without_accident": len(entries) - n_pos,
            "input_chars_max": max((e["input_chars"] for e in entries), default=0),
            "input_chars_mean": round(
                sum(e["input_chars"] for e in entries) / len(entries)
            )
            if entries
            else 0,
            **{k: v for k, v in info.items() if k != "n_windows"},
        },
        "output_schema": i18n.prediction_schema(
            scfg.language, cfg.n_horizon_buckets
        ),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return manifest


# ---------------------------------------------------------------- 채점


def _resolve_credit(
    ground_truth: dict,
    exp: Dict[int, dict],
    override_s: Optional[float],
) -> dict:
    """조기 예측 인정 범위를 정한다 → {early_credit_s, k_true, buckets}.

    정답 파일에 적힌 `credit` 를 쓰되, `override_s` 가 주어지면 그 값으로 범위를
    다시 계산한다. 구간은 1초 단위이므로 시각이 아니라 구간 번호로 셀 수 있다:
    충돌 구간 `k_true` 에서 `ceil(x)` 만큼 앞까지 인정한다.

    `credit` 가 없는 예전 정답 파일은 엄격 채점(0초)으로 떨어진다 — 조용히
    관대해지지 않는다.
    """
    gt_credit = ground_truth.get("credit") or {}
    k_true = gt_credit.get("k_true")
    if k_true is None:
        # 예전 파일 호환: '사고'인 구간을 정답에서 찾는다
        hits = [k for k, e in exp.items() if e.get("accident_expected")]
        k_true = hits[0] if hits else None

    x = gt_credit.get("early_credit_s", 0.0) if override_s is None else override_s
    x = max(0.0, float(x or 0.0))

    if k_true is None:
        return {"early_credit_s": x, "k_true": None, "buckets": set()}

    if override_s is None and gt_credit.get("k_from") is not None:
        k_from = int(gt_credit["k_from"])
    else:
        # 충돌 시각을 알면 `충돌시각 − x` 가 속한 구간으로 정한다 — 정답 파일을
        # 만들 때(credit_window)와 **같은 계산**이어야 다시 채점한 결과가 일치한다.
        ttc = ground_truth.get("time_to_collision_from_window_end_s")
        if ttc is not None:
            k_from = math.ceil(ttc - x - 1e-9)
        else:
            # 시각을 모르면 구간 폭(1초) 단위로 근사한다
            k_from = k_true - math.ceil(x - 1e-9)
    lo = max(min(exp) if exp else 1, k_from)
    return {
        "early_credit_s": x,
        "k_true": k_true,
        "buckets": {k for k in range(lo, k_true + 1) if k in exp},
    }


def score_response(
    response: dict,
    ground_truth: dict,
    early_credit_s: Optional[float] = None,
) -> dict:
    """Score one response at **bucket resolution**.

    ``early_credit_s`` is used to select an accepted early bucket.  To keep a
    five-bucket horizon at a five-bucket denominator, early scoring assigns the
    single positive target to the exact bucket (strict) or to the representative
    credited bucket (early).  Other buckets are still counted as TN/FP, so an
    early alarm is not duplicated into several TP entries.
    """
    exp = {e["k"]: e for e in ground_truth.get("expected", [])}
    preds = response.get("predictions", []) or []
    got = {p.get("k"): p for p in preds}
    issues = response_issues(preds, sorted(exp))

    credit = _resolve_credit(ground_truth, exp, early_credit_s)
    window = credit["buckets"]  # 정답으로 인정하는 구간 집합 (없으면 빈 집합)
    k_true = credit["k_true"]

    # 원본 데이터가 끝난 뒤의 구간은 채점하지 않는다. 그 구간의 '사고 없음' 은
    # 확인된 사실이 아니라 질문 형식을 맞추려고 채운 값이다 — 채점에 쓰면 모델이
    # 관측되지 않은 미래를 맞혔다고 점수를 주게 된다.
    scorable_ks = {k for k, e in exp.items() if e.get("scorable", True)}
    window = {k for k in window if k in scorable_ks}

    # Pick the representative alarm for event diagnostics.  Exact is preferred
    # when present; otherwise the earliest alarm in the accepted early range.
    event: Optional[dict] = None
    if window:
        hits = [
            k
            for k in sorted(window)
            if got.get(k) and bool(got[k].get("accident_expected"))
        ]
        # 정확히 맞힌 구간이 있으면 그것을, 없으면 가장 이른 경보를 대표로 쓴다
        credited = k_true if k_true in hits else (hits[0] if hits else None)
        event = {
            "k_true": k_true,
            "credit_buckets": sorted(window),
            "credited_k": credited,
            "detected": credited is not None,
            "timing": (
                "none"
                if credited is None
                else ("exact" if credited == k_true else "early")
            ),
            # 대표 경보가 실제 충돌보다 몇 초 이른가 (구간 상한 기준)
            "lead_s": (
                None
                if credited is None or ground_truth.get(
                    "time_to_collision_from_window_end_s"
                ) is None
                else round(
                    ground_truth["time_to_collision_from_window_end_s"] - credited,
                    3,
                )
            ),
        }

    # One positive target per accident window.  Strict keeps the actual bucket;
    # early moves that target to the credited bucket when an accepted alarm is
    # present.  This preserves a bucket denominator without counting the same
    # event multiple times.
    if k_true is None or k_true not in scorable_ks:
        target_k: Optional[int] = None
    elif credit["early_credit_s"] > 0:
        target_k = event["credited_k"] if event and event.get("credited_k") is not None else k_true
    else:
        target_k = k_true

    rows: List[dict] = []
    tp = fp = tn = fn = 0
    for k in sorted(exp):
        e = exp[k]
        p = got.get(k)
        actual_truth = bool(e.get("accident_expected"))
        in_window = k in window
        if k not in scorable_ks:
            # 상태만 남기고 어떤 집계에도 넣지 않는다
            rows.append(
                {
                    "k": k,
                    "status": "UNSCORABLE",
                    "truth": actual_truth,
                    "scoring_truth": None,
                    "pred": None if p is None else bool(p.get("accident_expected")),
                    "scorable": False,
                    "unscorable_reason": e.get("unscorable_reason", ""),
                    "reason": "" if p is None else p.get("reason", ""),
                }
            )
            continue
        pred = bool(p and p.get("accident_expected"))
        scoring_truth = target_k is not None and k == target_k
        if scoring_truth and pred:
            status = "TP"
            tp += 1
        elif scoring_truth and not pred:
            status = "missing" if p is None else "FN"
            fn += 1
        elif not scoring_truth and pred:
            if k_true is not None and k > k_true:
                status = "FP_LATE"
            elif in_window and event and event.get("credited_k") != k:
                status = "FP_EARLY_DUP"
            else:
                status = "FP"
            fp += 1
        else:
            status = "TN"
            tn += 1

        row = {
            "k": k,
            "status": status,
            # ``truth`` is the label used for this mode.  ``actual_truth`` is
            # retained so reports can still see the physical collision bucket.
            "truth": scoring_truth,
            "actual_truth": actual_truth,
            "scoring_truth": scoring_truth,
            "pred": pred,
            "in_credit_window": in_window,
            "reason": "" if p is None else p.get("reason", ""),
            "confidence": "" if p is None else p.get("confidence", ""),
        }
        # 차량 지목은 **대표 경보 구간**에서만 채점한다. 정답의 차량 정보는
        # 실제 충돌 구간(k_true)에만 들어 있으므로 그것을 기준으로 쓴다.
        score_vehicles = pred and event is not None and event.get("credited_k") == k
        if score_vehicles:
            e = exp.get(k_true, e)
            pred_ids = set(p.get("involved_actor_ids", []) or [])
            # 차량 단위 재현율: 융합이 한 차량을 여러 actor id 로 쪼갰을 수
            # 있으므로 그중 하나만 지목해도 맞은 것으로 본다.
            vehicles = e.get("involved_vehicles") or [
                {"carla_id": None, "actor_ids": e.get("involved_actor_ids", [])}
            ]
            matched, missed = [], []
            acceptable: Set[str] = set()
            for v in vehicles:
                aids = set(v.get("actor_ids", []))
                acceptable |= aids
                if aids & pred_ids:
                    matched.append(v.get("carla_id"))
                else:
                    missed.append(v.get("carla_id"))
            scorable = [v for v in vehicles if v.get("actor_ids")]
            row["vehicle_recall"] = (
                round(len(matched) / len(scorable), 3) if scorable else None
            )
            row["actor_precision"] = (
                round(len(pred_ids & acceptable) / len(pred_ids), 3)
                if pred_ids
                else None
            )
            row["matched_carla_ids"] = matched
            row["missed_carla_ids"] = missed
            row["extra_actor_ids"] = sorted(pred_ids - acceptable)
            row["all_involved_observed"] = e.get("all_involved_observed", True)
        rows.append(row)

    total = tp + fp + tn + fn
    return {
        "window": ground_truth.get("window", {}),
        "scenario_ref": ground_truth.get("scenario", {}).get("ref", ""),
        "counts": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "status": _status_summary({"TP": tp, "FP": fp, "TN": tn, "FN": fn}),
        "accuracy": round((tp + tn) / total, 3) if total else None,
        "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
        "recall": round(tp / (tp + fn), 3) if (tp + fn) else None,
        # 조기 예측 인정 기준과 그 결과. `event` 는 충돌이 이 윈도우의 지평 안에
        # 있을 때만 있다 (그때만 인정 범위가 정의된다).
        "credit": {
            "early_credit_s": credit["early_credit_s"],
            "buckets": sorted(window),
        },
        # 실제로 채점한 구간과 제외한 구간. 데이터가 끝난 뒤의 구간은 정답을
        # 확인할 수 없어 뺀다.
        "scored_buckets": sorted(scorable_ks),
        "n_buckets": len(scorable_ks),
        "excluded_buckets": sorted(set(exp) - scorable_ks),
        "data_end_s": ground_truth.get("data_end_s"),
        "event": event,
        "per_interval": rows,
        # 응답 형식 문제. 채점은 k 로 짝지으므로 중복은 조용히 무시되고 누락은
        # 정답이 '사고'일 때만 FN 으로 잡힌다 — 즉 형식 오류가 점수에 거의
        # 드러나지 않는다. 실제로 모델이 같은 k 를 두 번 낸 응답이 있었고
        # ("동일 구간 중복 표기 방지용 확인 항목"이라 적어 두기까지 했다),
        # 그것이 채점 결과만 봐서는 보이지 않았다.
        "response_issues": issues,
    }


def response_issues(
    predictions: Sequence[dict], expected_ks: Sequence[int]
) -> List[dict]:
    """응답 예측 목록의 형식 문제를 찾는다.

    스키마(JSON Schema)는 배열 길이와 항목 형태만 강제할 수 있고 "k 가 1..N 을
    정확히 한 번씩 덮는다"는 제약은 표현하지 못한다. 그래서 채점 단계에서 검사해
    보고한다 — 조용히 넘기면 구간을 빠뜨린 응답이 정상으로 보인다.
    """
    out: List[dict] = []
    ks = [p.get("k") for p in predictions]
    seen: Dict[object, int] = {}
    for k in ks:
        seen[k] = seen.get(k, 0) + 1

    dups = sorted(
        (k for k, c in seen.items() if c > 1 and k is not None),
        key=lambda v: (v is None, v),
    )
    if dups:
        out.append(
            {
                "kind": "duplicate_k",
                "detail": {int(k): seen[k] for k in dups},
                "note": "같은 구간을 여러 번 답했다. 채점은 첫 항목만 쓴다.",
            }
        )
    missing = [k for k in expected_ks if k not in seen]
    if missing:
        out.append(
            {
                "kind": "missing_k",
                "detail": missing,
                "note": "답하지 않은 구간이 있다.",
            }
        )
    extra = sorted(
        k for k in seen if k is not None and k not in set(expected_ks)
    )
    if extra:
        out.append(
            {
                "kind": "unexpected_k",
                "detail": extra,
                "note": "질문에 없는 구간을 답했다.",
            }
        )
    if any(k is None for k in ks):
        out.append(
            {
                "kind": "missing_k_field",
                "detail": sum(1 for k in ks if k is None),
                "note": "k 필드가 없는 항목이 있다.",
            }
        )
    return out


def aggregate_scores(scores: Iterable[dict]) -> dict:
    """여러 윈도우 채점 결과 합산."""
    tp = fp = tn = fn = 0
    recalls: List[float] = []
    precisions: List[float] = []
    issues: Dict[str, int] = {}
    timing: Dict[str, int] = {"exact": 0, "early": 0, "none": 0}
    leads: List[float] = []
    n = n_events = n_buckets = 0
    for s in scores:
        n += 1
        n_buckets += int(s.get("n_buckets", len(s.get("scored_buckets", []))) or 0)
        ev = s.get("event")
        if ev:
            n_events += 1
            timing[ev.get("timing", "none")] = (
                timing.get(ev.get("timing", "none"), 0) + 1
            )
            if ev.get("lead_s") is not None:
                leads.append(ev["lead_s"])
        c = s.get("counts", {})
        tp += c.get("TP", 0)
        fp += c.get("FP", 0)
        tn += c.get("TN", 0)
        fn += c.get("FN", 0)
        for iss in s.get("response_issues", []) or []:
            issues[iss.get("kind", "?")] = issues.get(iss.get("kind", "?"), 0) + 1
        for row in s.get("per_interval", []):
            if row.get("vehicle_recall") is not None:
                recalls.append(row["vehicle_recall"])
            if row.get("actor_precision") is not None:
                precisions.append(row["actor_precision"])
    total = tp + fp + tn + fn
    return {
        "n_windows": n,
        "n_buckets": n_buckets,
        "counts": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "accuracy": round((tp + tn) / total, 3) if total else None,
        "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
        "recall": round(tp / (tp + fn), 3) if (tp + fn) else None,
        "vehicle_recall_mean": round(sum(recalls) / len(recalls), 3)
        if recalls
        else None,
        "actor_precision_mean": round(sum(precisions) / len(precisions), 3)
        if precisions
        else None,
        # 사건 단위 요약. 충돌이 질문 지평 안에 있는 윈도우만 센다.
        # exact = 실제 구간을 맞힘, early = 인정 범위 안에서 이르게 맞힘,
        # none = 인정 범위 안에서 한 번도 사고로 답하지 않음(미검출).
        "events": {
            "n": n_events,
            **timing,
            "detection_rate": round(
                (timing["exact"] + timing["early"]) / n_events, 3
            )
            if n_events
            else None,
            "mean_lead_s": round(sum(leads) / len(leads), 3) if leads else None,
        },
        # 형식 문제가 몇 창에서 났는가. 점수에는 거의 드러나지 않으므로
        # 따로 세어 보고한다.
        "response_issue_counts": issues,
    }


# ------------------------------------------------- 다중 기준 채점 (score_modes)
#
# 같은 응답을 여러 기준으로 **동시에** 채점한다. 주 비교축은 시나리오 하나당
# 한 건인 `binary_scenario` 이고, 각 고정 창의 관측 시점 변화는 `binary_window`,
# 창 안의 세부 시간 위치는 `binary_bucket` 으로 분리한다.
#
#   binary_scenario  한 시나리오에서 유효한 창을 합쳐 TP/FP/TN/FN 한 건
#   binary_window   고정 observation window 하나당 TP/FP/TN/FN 한 건
#   binary_bucket   미래 지평의 각 bucket을 독립적으로 TP/FP/TN/FN 채점
#   strict          동일한 bucket 해상도에서 실제 사고 bucket만 양성
#   early           인정 범위 안에서 대표 bucket 하나를 양성으로 옮긴 뒤 나머지도 채점
#   weighted        첫 경보와 실제 bucket 거리의 부분 점수 + bucket confusion 보조값
#
# strict·early 는 `score_response` 를 그대로 재사용한다 — 채점 규칙이 한 곳에만
# 있어야 모드 간 숫자 차이가 '기준의 차이'로만 설명된다.

# `binary` 는 과거 산출물과의 호환을 위한 별칭이며 새 코드는 명시적인 이름을 쓴다.
MODES: Tuple[str, ...] = (
    "binary_scenario", "binary_window", "strict", "early", "weighted", "binary_bucket"
)
SUPPORTED_MODES: Tuple[str, ...] = MODES + ("binary",)


def _status_summary(counts: Dict[str, int]) -> str:
    """Return a compact status for compatibility with old per-window reports."""
    active = [k for k in ("TP", "FP", "TN", "FN") if counts.get(k, 0)]
    return active[0] if len(active) == 1 else ("mixed" if active else "UNSCORABLE")


def _exp_got(
    response: dict, ground_truth: dict
) -> Tuple[Dict[int, dict], Dict[int, dict], Set[int]]:
    """정답·응답·채점가능 구간을 한 번에 꺼낸다.

    `score_response` 와 **같은 규약**이어야 한다: k 로 짝짓고(중복은 마지막이
    남는다 — 형식 문제는 `response_issues` 가 따로 잡는다), 원본 데이터가 끝난
    뒤의 구간(`scorable=False`)은 어떤 집계에도 넣지 않는다.
    """
    exp = {e["k"]: e for e in ground_truth.get("expected", [])}
    got = {p.get("k"): p for p in (response.get("predictions", []) or [])}
    scorable = {k for k, e in exp.items() if e.get("scorable", True)}
    return exp, got, scorable


def _positive_ks(got: Dict[int, dict], scorable: Set[int]) -> List[int]:
    """'사고'로 답한 채점가능 구간들 (오름차순)."""
    return sorted(
        k
        for k in scorable
        if got.get(k) is not None and bool(got[k].get("accident_expected"))
    )


def _k_true(ground_truth: dict, exp: Dict[int, dict], scorable: Set[int]) -> Optional[int]:
    """충돌이 속한 구간. 지평 밖이거나 충돌이 없으면 None.

    정답 파일의 `credit.k_true` 를 쓰고, 없는 예전 파일은 '사고' 구간에서 찾는다
    (`_resolve_credit` 와 같은 대체 경로다).
    """
    k = (ground_truth.get("credit") or {}).get("k_true")
    if k is not None:
        return int(k)
    hits = [k for k in sorted(scorable) if exp[k].get("accident_expected")]
    return hits[0] if hits else None


def score_binary_bucket(response: dict, ground_truth: dict) -> dict:
    """Binary confusion at bucket resolution.

    A five-second horizon with five one-second buckets contributes five
    independent TN/FP/FN/TP entries.  The physical accident label remains only
    on its actual bucket; an alarm in another bucket is therefore a FP.
    """
    exp, got, scorable = _exp_got(response, ground_truth)
    if not exp:
        # 정답 행 자체가 없으면 윈도우 판정을 정의할 수 없다.
        return {
            "scorable": False,
            "status": "UNSCORABLE",
            "truth": None,
            "pred": None,
            "counts": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
            "n_scorable": 0,
            "n_positive_buckets": 0,
            "positive_ks": [],
            "n_buckets": 0,
        }
    if not scorable:
        return {
            "scorable": False,
            "status": "UNSCORABLE",
            "truth": None,
            "pred": None,
            "counts": {s: 0 for s in ("TP", "FP", "TN", "FN")},
            "n_scorable": 0,
            "n_buckets": 0,
            "n_positive_buckets": 0,
            "positive_ks": [],
            "per_interval": [],
        }
    counts = {s: 0 for s in ("TP", "FP", "TN", "FN")}
    per_interval: List[dict] = []
    pos: List[int] = []
    for k in sorted(scorable):
        e = exp[k]
        p = got.get(k)
        truth = bool(e.get("accident_expected"))
        pred = bool(p and p.get("accident_expected"))
        status = "TP" if truth and pred else "FN" if truth else "FP" if pred else "TN"
        counts[status] += 1
        if pred:
            pos.append(k)
        per_interval.append({
            "k": k,
            "status": "missing" if p is None and truth else status,
            "truth": truth,
            "pred": pred,
            "scorable": True,
            "reason": "" if p is None else p.get("reason", ""),
        })
    # Keep the historical window-level status as a convenience field while
    # ``counts`` is now unambiguously bucket-level.  Consumers migrating from
    # the old reports can use this field; metrics must use the counts.
    window_truth = any(bool(e.get("accident_expected")) for e in exp.values())
    all_positive = _positive_ks(got, set(exp))
    window_pred = bool(all_positive)
    window_status = "TP" if window_truth and window_pred else "FN" if window_truth else "FP" if window_pred else "TN"
    return {
        "scorable": True,
        "status": window_status,
        "bucket_status": _status_summary(counts),
        "truth": window_truth,
        "pred": window_pred,
        "counts": counts,
        "n_scorable": len(scorable),
        "n_buckets": len(scorable),
        "n_positive_buckets": len(all_positive),
        "positive_ks": all_positive,
        "per_interval": per_interval,
    }


def score_binary(response: dict, ground_truth: dict) -> dict:
    """Backward-compatible alias for :func:`score_binary_bucket`.

    Older scripts imported ``score_binary`` directly.  New reports should use
    the explicit ``binary_bucket`` name so that it cannot be confused with the
    one-row-per-window or one-row-per-scenario axes.
    """
    return score_binary_bucket(response, ground_truth)


def score_binary_window(response: dict, ground_truth: dict) -> dict:
    """Score one fixed observation window as one binary sample.

    A window is eligible when its future label is known.  A collision inside
    the horizon is known even if the recording ends immediately afterwards;
    a no-collision window needs the complete horizon, otherwise it is censored
    and excluded.  Once eligible, any alarm in the requested horizon counts,
    including a timing bucket beyond recording end.  Bucket scoring separately
    excludes those unknown timing labels.
    """
    exp, got, scorable = _exp_got(response, ground_truth)
    if not exp:
        return {
            "scorable": False,
            "status": "UNSCORABLE",
            "truth": None,
            "pred": None,
            "counts": {s: 0 for s in ("TP", "FP", "TN", "FN")},
            "n_windows": 0,
            "n_scorable": 0,
            "n_buckets": 0,
            "per_window": [],
        }

    truth = any(bool(e.get("accident_expected")) for e in exp.values())
    # A collision bucket is explicitly marked scorable by window_ground_truth
    # (`collision_known`), even when later no-collision buckets are censored.
    if truth:
        # Confirm the positive window label before collapsing predictions.
        if not any(exp[k].get("accident_expected") for k in scorable):
            return {
                "scorable": False,
                "status": "UNSCORABLE",
                "truth": True,
                "pred": None,
                "counts": {s: 0 for s in ("TP", "FP", "TN", "FN")},
                "n_windows": 0,
                "n_scorable": 0,
                "n_buckets": len(scorable),
                "per_window": [],
            }
    else:
        # For a negative binary window, every requested future bucket must be
        # observed.  Partial no-accident futures are right-censored.
        if len(scorable) != len(exp):
            return {
                "scorable": False,
                "status": "UNSCORABLE",
                "truth": False,
                "pred": None,
                "counts": {s: 0 for s in ("TP", "FP", "TN", "FN")},
                "n_windows": 0,
                "n_scorable": 0,
                "n_buckets": len(scorable),
                "per_window": [],
            }

    # Window eligibility and prediction timing are distinct: the observed
    # collision proves the horizon-level positive even if later timing labels
    # are unknown.  Only requested buckets count (ignore out-of-horizon IDs).
    eligible_ks = set(exp)

    pred = any(
        bool(got.get(k) and got[k].get("accident_expected"))
        for k in eligible_ks
    )
    status = "TP" if truth and pred else "FN" if truth else "FP" if pred else "TN"
    counts = {s: 0 for s in ("TP", "FP", "TN", "FN")}
    counts[status] = 1
    row = {
        "status": status,
        "truth": truth,
        "pred": pred,
        "scorable": True,
        "eligible_buckets": sorted(eligible_ks),
        "n_buckets": len(scorable),
    }
    return {
        "window": ground_truth.get("window", {}),
        "scenario_ref": ground_truth.get("scenario", {}).get("ref", ""),
        "scenario_outcome": ground_truth.get("scenario", {}).get("outcome", ""),
        "scorable": True,
        "status": status,
        "truth": truth,
        "pred": pred,
        "counts": counts,
        "n_windows": 1,
        "n_scorable": 1,
        "n_buckets": len(scorable),
        "eligible_buckets": sorted(eligible_ks),
        "n_positive_buckets": sum(
            bool(got.get(k) and got[k].get("accident_expected"))
            for k in eligible_ks
        ),
        "per_window": [row],
    }


def score_weighted(
    response: dict,
    ground_truth: dict,
    late_decay: float = 0.2,
    early_decay: float = 0.0,
) -> dict:
    """실제 구간과의 거리로 부분 점수 [0, 1].

    Δ = 대표 경보 구간 − 충돌 구간.
        Δ ≤ 0 (이른 경보)  →  1.0 − early_decay·|Δ|   (기본 early_decay=0 → 1.0)
        Δ > 0 (늦은 경보)  →  1.0 − late_decay·Δ      (0.8 / 0.6 / 0.4 …)

    이르게 부른 것을 깎지 않는 것이 기본값인 이유: 사고 예측의 목적은 회피할
    시간을 버는 것이므로 먼저 부른 경보는 목적을 달성한 것이다. 늦은 경보는
    `early` 모드에서 그냥 오답(FP)이지만 여기서는 **부분 점수**를 받는다 —
    "시점은 늦어도 미래 사고 자체는 미리 맞혔다" 를 숫자로 남기기 위해서다.

    **대표 경보는 첫 경보(가장 이른 양성)다.** `score_response` 의 `credited_k`
    (정확히 맞힌 칸이 있으면 그것)를 쓰면 범위를 칠한 답이 항상 Δ=0 이 되어
    가중이 무의미해진다. 실제로 시스템이 움직이는 시점도 첫 경보다.

    충돌이 지평 밖이면 거리 가중 점수는 정의되지 않는다 (`applicable=False`).
    그래도 bucket confusion과 bucket-level 오경보율은 모든 윈도우에서 기록한다.
    """
    exp, got, scorable = _exp_got(response, ground_truth)
    bucket = score_binary(response, ground_truth)
    bucket_counts = bucket.get("counts", {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
    bucket_total = sum(bucket_counts.values())
    bucket_fa_den = bucket_counts["FP"] + bucket_counts["TN"]
    bucket_false_alarm_rate = (
        round(bucket_counts["FP"] / bucket_fa_den, 3) if bucket_fa_den else None
    )
    pos = _positive_ks(got, scorable)
    alarm_k = pos[0] if pos else None
    k_true = _k_true(ground_truth, exp, scorable) if scorable else None

    if k_true is None or k_true not in scorable:
        # 이 윈도우의 지평 안에 (채점 가능한) 충돌이 없다.
        return {
            "applicable": False,
            "score": None,
            "k_true": k_true,
            "alarm_k": alarm_k,
            "delta": None,
            "detected": None,
            "false_alarm": alarm_k is not None,
            "scorable": bool(scorable),
            "counts": bucket_counts,
            "n_buckets": bucket_total,
            "bucket_accuracy": round((bucket_counts["TP"] + bucket_counts["TN"]) / bucket_total, 3) if bucket_total else None,
            "bucket_false_alarm_rate": bucket_false_alarm_rate,
            "per_interval": bucket.get("per_interval", []),
        }

    if alarm_k is None:
        return {
            "applicable": True,
            "score": 0.0,
            "k_true": k_true,
            "alarm_k": None,
            "delta": None,
            "detected": False,
            "false_alarm": False,
            "scorable": True,
            "counts": bucket_counts,
            "n_buckets": bucket_total,
            "bucket_accuracy": round((bucket_counts["TP"] + bucket_counts["TN"]) / bucket_total, 3) if bucket_total else None,
            "bucket_false_alarm_rate": bucket_false_alarm_rate,
            "per_interval": bucket.get("per_interval", []),
        }

    delta = alarm_k - k_true
    decay = (late_decay * delta) if delta > 0 else (early_decay * -delta)
    score = min(1.0, max(0.0, 1.0 - decay))
    return {
        "applicable": True,
        "score": round(score, 3),
        "k_true": k_true,
        "alarm_k": alarm_k,
        "delta": delta,
        "detected": True,
        "false_alarm": False,
        "scorable": True,
        "counts": bucket_counts,
        "n_buckets": bucket_total,
        "bucket_accuracy": round((bucket_counts["TP"] + bucket_counts["TN"]) / bucket_total, 3) if bucket_total else None,
        "bucket_false_alarm_rate": bucket_false_alarm_rate,
        "per_interval": bucket.get("per_interval", []),
        "decay": {"late": late_decay, "early": early_decay},
    }


def score_modes(
    response: dict,
    ground_truth: dict,
    *,
    credit_s: Optional[float] = None,
    late_decay: float = 0.2,
    early_decay: float = 0.0,
    modes: Sequence[str] = MODES,
) -> dict:
    """같은 응답을 여러 기준으로 동시에 채점 → {mode: 결과}.

    입력이 동일하므로 모드 간 숫자 차이는 전부 **기준의 차이**다.

    `credit_s` 는 `early` 모드의 인정 폭을 덮어쓴다. None 이면 정답 파일의
    `credit.early_credit_s` (기본 2초)를 쓴다. `strict` 는 항상 0 이다.
    """
    unknown = [m for m in modes if m not in SUPPORTED_MODES]
    if unknown:
        raise ValueError(
            f"알 수 없는 채점 모드: {unknown} (가능: {list(SUPPORTED_MODES)})"
        )

    out: dict = {
        "window": ground_truth.get("window", {}),
        "scenario_ref": ground_truth.get("scenario", {}).get("ref", ""),
        "scenario_outcome": ground_truth.get("scenario", {}).get("outcome", ""),
        "modes": list(modes),
        "response_issues": response_issues(
            response.get("predictions", []) or [],
            sorted(e["k"] for e in ground_truth.get("expected", [])),
        ),
    }
    # `binary_scenario` is an aggregate-only axis: it needs all windows from a
    # scenario.  The per-window record carries enough metadata for
    # `aggregate_modes` to derive it later.
    if "binary_window" in modes or "binary_scenario" in modes:
        out["binary_window"] = score_binary_window(response, ground_truth)
    if "binary_bucket" in modes:
        out["binary_bucket"] = score_binary_bucket(response, ground_truth)
    # Keep the old key when explicitly requested (and for the default call
    # used by older clients).  It remains the bucket-level result, not the new
    # primary scenario-level metric.
    if "binary" in modes or modes == MODES:
        out["binary"] = score_binary_bucket(response, ground_truth)
    if "strict" in modes:
        out["strict"] = score_response(response, ground_truth, 0.0)
    if "early" in modes:
        out["early"] = score_response(response, ground_truth, credit_s)
    if "weighted" in modes:
        out["weighted"] = score_weighted(
            response, ground_truth, late_decay, early_decay
        )
    return out


def _aggregate_binary_window(rows: Sequence[dict]) -> dict:
    """Aggregate one binary confusion entry per eligible fixed window."""
    counts = {s: 0 for s in ("TP", "FP", "TN", "FN")}
    n = 0
    for b in rows:
        if not b.get("scorable"):
            continue
        n += 1
        for s in counts:
            counts[s] += int((b.get("counts") or {}).get(s, 0) or 0)
    total = sum(counts.values())
    prec = counts["TP"] / (counts["TP"] + counts["FP"]) if counts["TP"] + counts["FP"] else None
    rec = counts["TP"] / (counts["TP"] + counts["FN"]) if counts["TP"] + counts["FN"] else None
    return {
        "n_windows": n,
        "counts": counts,
        "accuracy": round((counts["TP"] + counts["TN"]) / total, 3) if total else None,
        "precision": round(prec, 3) if prec is not None else None,
        "recall": round(rec, 3) if rec is not None else None,
        "f1": round(2 * prec * rec / (prec + rec), 3)
        if prec is not None and rec is not None and (prec + rec)
        else None,
        "n_scorable_windows": n,
        "n_censored_windows": len(rows) - n,
    }


def _aggregate_binary_scenario(rows: Sequence[dict]) -> dict:
    """Collapse per-window binary results to one outcome per scenario.

    Accident scenarios use only windows whose future horizon contains the known
    collision.  Normal scenarios require at least one uncensored window and
    are positive if any eligible window raises an alarm.  This implements the
    meeting rule that a test case contributes one result, not one result per
    overlapping window.
    """
    groups: Dict[str, List[dict]] = {}
    for row in rows:
        key = (
            row.get("scenario_ref")
            or row.get("scenario")
            or row.get("scenario_id")
            or "__single_scenario__"
        )
        groups.setdefault(str(key), []).append(row)

    counts = {s: 0 for s in ("TP", "FP", "TN", "FN")}
    per_scenario: List[dict] = []
    censored = 0
    for key, group in sorted(groups.items()):
        outcome = next(
            (
                r.get("scenario_outcome") or r.get("outcome")
                for r in group
                if r.get("scenario_outcome") or r.get("outcome")
            ),
            None,
        )
        accident = str(outcome).lower() == "accident" if outcome else any(
            bool((r.get("binary_window") or {}).get("truth")) for r in group
        )
        bw = [r.get("binary_window", {}) for r in group]
        if accident:
            eligible = [r for r in bw if r.get("scorable") and r.get("truth")]
        else:
            eligible = [r for r in bw if r.get("scorable")]
        if not eligible:
            censored += 1
            per_scenario.append({
                "scenario_ref": key,
                "outcome": "accident" if accident else "normal",
                "status": "UNSCORABLE",
                "scorable": False,
                "n_windows": len(group),
                "n_eligible_windows": 0,
            })
            continue
        pred = any(bool(r.get("pred")) for r in eligible)
        status = "TP" if accident and pred else "FN" if accident else "FP" if pred else "TN"
        counts[status] += 1
        per_scenario.append({
            "scenario_ref": key,
            "outcome": "accident" if accident else "normal",
            "status": status,
            "truth": accident,
            "pred": pred,
            "scorable": True,
            "n_windows": len(group),
            "n_eligible_windows": len(eligible),
            "n_positive_windows": sum(bool(r.get("pred")) for r in eligible),
        })

    total = sum(counts.values())
    prec = counts["TP"] / (counts["TP"] + counts["FP"]) if counts["TP"] + counts["FP"] else None
    rec = counts["TP"] / (counts["TP"] + counts["FN"]) if counts["TP"] + counts["FN"] else None
    return {
        "n_scenarios": len(groups),
        "n_scorable_scenarios": total,
        "n_censored_scenarios": censored,
        "counts": counts,
        "accuracy": round((counts["TP"] + counts["TN"]) / total, 3) if total else None,
        "precision": round(prec, 3) if prec is not None else None,
        "recall": round(rec, 3) if rec is not None else None,
        "f1": round(2 * prec * rec / (prec + rec), 3)
        if prec is not None and rec is not None and (prec + rec)
        else None,
        "per_scenario": per_scenario,
    }


def _aggregate_binary(rows: Sequence[dict]) -> dict:
    tp = fp = tn = fn = 0
    sprayed: List[int] = []
    n = n_buckets = 0
    for b in rows:
        if not b.get("scorable"):
            continue
        n += 1
        n_buckets += int(b.get("n_buckets", b.get("n_scorable", 0)) or 0)
        c = b.get("counts", {})
        tp += c.get("TP", 0)
        fp += c.get("FP", 0)
        tn += c.get("TN", 0)
        fn += c.get("FN", 0)
        sprayed.append(b.get("n_positive_buckets", 0))
    total = tp + fp + tn + fn
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    return {
        "n_windows": n,
        "n_buckets": n_buckets,
        "counts": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "accuracy": round((tp + tn) / total, 3) if total else None,
        "precision": round(prec, 3) if prec is not None else None,
        "recall": round(rec, 3) if rec is not None else None,
        "f1": round(2 * prec * rec / (prec + rec), 3)
        if prec and rec
        else None,
        # 몇 칸을 '사고'로 칠했는가. bucket confusion에서는 도배가 FP로 반영되지만,
        # 이 값은 경보 성향을 별도로 읽기 쉽게 남긴다.
        "mean_positive_buckets": round(sum(sprayed) / len(sprayed), 3)
        if sprayed
        else None,
    }


def _aggregate_weighted(rows: Sequence[dict]) -> dict:
    scores: List[float] = []
    n_detected = n_missed = 0
    n_no_event = n_false_alarm = 0
    bucket_counts = {s: 0 for s in ("TP", "FP", "TN", "FN")}
    n_buckets = 0
    for w in rows:
        c = w.get("counts", {})
        for s in bucket_counts:
            bucket_counts[s] += int(c.get(s, 0) or 0)
        n_buckets += int(w.get("n_buckets", sum(c.values())) or 0)
        if not w.get("scorable"):
            continue
        if not w.get("applicable"):
            n_no_event += 1
            n_false_alarm += 1 if w.get("false_alarm") else 0
            continue
        scores.append(float(w.get("score") or 0.0))
        if w.get("detected"):
            n_detected += 1
        else:
            n_missed += 1
    n = len(scores)
    return {
        # 충돌이 지평 안에 있는 윈도우만. 부분 점수는 여기서만 정의된다.
        "n_event_windows": n,
        "n_buckets": n_buckets,
        "counts": bucket_counts,
        "bucket_accuracy": round(
            (bucket_counts["TP"] + bucket_counts["TN"]) / n_buckets, 3
        ) if n_buckets else None,
        "mean_score": round(sum(scores) / n, 3) if n else None,
        "n_detected": n_detected,
        "n_missed": n_missed,
        "detection_rate": round(n_detected / n, 3) if n else None,
        # 충돌이 없는(또는 지평 밖인) 윈도우. 위 평균 점수는 이쪽을 전혀 보지
        # 않으므로 오경보율을 반드시 같이 읽어야 한다.
        "n_no_event_windows": n_no_event,
        "false_alarms": n_false_alarm,
        # Window-level alarm presence is retained for compatibility; the main
        # false-alarm rate is now the bucket-level confusion rate.
        "window_false_alarm_rate": round(n_false_alarm / n_no_event, 3)
        if n_no_event else None,
        "false_alarm_rate": round(
            bucket_counts["FP"] / (bucket_counts["FP"] + bucket_counts["TN"]), 3
        ) if (bucket_counts["FP"] + bucket_counts["TN"]) else None,
    }


def aggregate_modes(
    rows: Iterable[dict], modes: Optional[Sequence[str]] = None
) -> dict:
    """`score_modes` 결과 여러 개를 모드별로 합산.

    `modes` 를 주지 않으면 첫 행에 실제로 들어 있는 모드를 쓴다.
    """
    rows = list(rows)
    if modes is None:
        modes = rows[0].get("modes", MODES) if rows else MODES
    out: dict = {"n_windows": len(rows)}
    # Explicit axes.  The legacy `binary` alias is retained as bucket-level
    # output so old reports remain readable, but new callers should consume the
    # named fields below.
    window_rows = [r["binary_window"] for r in rows if "binary_window" in r]
    bucket_rows = [r["binary_bucket"] for r in rows if "binary_bucket" in r]
    legacy_rows = [r["binary"] for r in rows if "binary" in r]
    if "binary_window" in modes or window_rows:
        out["binary_window"] = _aggregate_binary_window(window_rows)
    if "binary_bucket" in modes or bucket_rows:
        out["binary_bucket"] = _aggregate_binary(bucket_rows)
    if "binary_scenario" in modes or window_rows:
        out["binary_scenario"] = _aggregate_binary_scenario(rows)
    if "binary" in modes or legacy_rows:
        out["binary"] = _aggregate_binary(legacy_rows)
    if not rows:
        # Preserve the old empty-result shape for callers that initialize a
        # report before any windows have completed.
        zero = {"counts": {s: 0 for s in ("TP", "FP", "TN", "FN")},
                "n_windows": 0, "n_buckets": 0, "accuracy": None,
                "precision": None, "recall": None, "f1": None}
        out.setdefault("binary", dict(zero))
        out.setdefault("binary_window", {
            **zero, "n_scorable_windows": 0, "n_censored_windows": 0
        })
        out.setdefault("binary_bucket", dict(zero))
        out.setdefault("binary_scenario", {
            "n_scenarios": 0, "n_scorable_scenarios": 0,
            "n_censored_scenarios": 0, "counts": dict(zero["counts"]),
            "accuracy": None, "precision": None, "recall": None,
            "f1": None, "per_scenario": [],
        })
    for m in ("strict", "early"):
        if m in modes:
            out[m] = aggregate_scores([r[m] for r in rows if m in r])
    if "weighted" in modes:
        out["weighted"] = _aggregate_weighted(
            [r["weighted"] for r in rows if "weighted" in r]
        )
    return out
