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

정답 누수 방지
    - 시나리오 id 는 분할명("..._accident")을 담으므로 payload 에서는 불투명
      토큰으로 치환한다 (serialize.scenario_ref).
    - 충돌 정보는 payload 에 넣지 않는다.
    - 충돌이 관측 윈도우 **안에서** 이미 일어난 윈도우는 예측 문제가 아니므로
      기본적으로 버린다 (drop_windows_after_collision).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import i18n, providers
from .config import SerializeConfig
from .kinematics import risk_score
from .schemas import ActorState, CollisionTruth, SceneSnapshot
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
    history_mode: str = "compact"  # 'compact' | 'full'(스냅샷 JSON 전부)
    require_full_window: bool = True  # 데이터가 부족한 윈도우는 버림
    drop_windows_after_collision: bool = True
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
    """스냅샷 시퀀스를 슬라이딩 윈도우로 자른다.

    Returns: (윈도우 목록, 진단 정보)
    """
    cfg = cfg or WindowConfig()
    snaps = sorted(snapshots, key=lambda s: s.t)
    info: Dict[str, object] = {
        "n_snapshots": len(snaps),
        "dropped_incomplete": 0,
        "dropped_after_collision": 0,
    }
    if not snaps:
        return ([], info)

    t0, t1 = snaps[0].t, snaps[-1].t
    info["data_range_s"] = [round(t0, 3), round(t1, 3)]

    out: List[TimeWindow] = []
    idx = 0
    start = t0
    eps = 1e-6
    while start + cfg.window_s <= t1 + cfg.stride_s * 0.5 + eps:
        end = start + cfg.window_s
        inside = [s for s in snaps if start - eps <= s.t <= end + eps]
        if cfg.require_full_window and (
            not inside or inside[-1].t < end - eps or inside[0].t > start + eps
        ):
            info["dropped_incomplete"] = int(info["dropped_incomplete"]) + 1
            start += cfg.stride_s
            continue
        if len(inside) < 2:
            info["dropped_incomplete"] = int(info["dropped_incomplete"]) + 1
            start += cfg.stride_s
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
            start += cfg.stride_s
            continue

        out.append(TimeWindow(index=idx, t_start=start, t_end=end, snapshots=inside))
        idx += 1
        start += cfg.stride_s

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
            win=_fmt_t(cfg.window_s),
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

    blocks: List[str] = [render_window_text(win, scfg, cfg)]
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
        system=i18n.system_prompt("accident", scfg.language),
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


def window_ground_truth(
    win: TimeWindow,
    cfg: Optional[WindowConfig] = None,
    collision: Optional[CollisionTruth] = None,
    agent_carla_ids: Optional[Dict[str, int]] = None,
    scenario_id: str = "",
    scenario_split: str = "",
    town: str = "",
    lang: str = "ko",
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
        expected.append(
            {
                "k": k,
                "interval_s": f"({_fmt_t(lo)}, {_fmt_t(hi)}]",
                "interval_start_s": round(lo, 3),
                "interval_end_s": round(hi, 3),
                "accident_expected": hit,
                "involved_vehicles": involved_vehicles if hit else [],
                "involved_actor_ids": involved_actor_ids if hit else [],
                "involved_carla_ids": list(c_ids) if hit else [],
                "unobserved_carla_ids": unobserved if hit else [],
                "all_involved_observed": (not unobserved) if hit else True,
            }
        )

    ttc = None if c_time is None else round(c_time - win.t_end, 3)
    return {
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
            "split": scenario_split,
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
    town: str = "",
    also_write_text: bool = True,
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

    c_time = collision.time_s if (collision and collision.occurred) else None
    windows, info = build_windows(snapshots, cfg, collision_time_s=c_time)

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
                f.write(render_window_text(win, scfg, cfg))

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
            "split": scenario_split,
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


def score_response(response: dict, ground_truth: dict) -> dict:
    """LLM 응답 1건을 정답과 비교.

    response 는 PREDICTION_SCHEMA 를 따르는 dict (구조화 출력 결과).
    구간별 정오와 차량 지목 정확도를 함께 낸다.
    """
    exp = {e["k"]: e for e in ground_truth.get("expected", [])}
    preds = response.get("predictions", []) or []
    got = {p.get("k"): p for p in preds}
    issues = response_issues(preds, sorted(exp))

    rows: List[dict] = []
    tp = fp = tn = fn = 0
    for k in sorted(exp):
        e = exp[k]
        p = got.get(k)
        truth = bool(e["accident_expected"])
        if p is None:
            rows.append({"k": k, "status": "missing", "truth": truth})
            if truth:
                fn += 1
            continue
        pred = bool(p.get("accident_expected"))
        if pred and truth:
            tp += 1
            status = "TP"
        elif pred and not truth:
            fp += 1
            status = "FP"
        elif not pred and truth:
            fn += 1
            status = "FN"
        else:
            tn += 1
            status = "TN"

        pred_ids = set(p.get("involved_actor_ids", []) or [])
        row = {
            "k": k,
            "status": status,
            "truth": truth,
            "pred": pred,
            "reason": p.get("reason", ""),
            "confidence": p.get("confidence", ""),
        }
        if truth and pred:
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
        "accuracy": round((tp + tn) / total, 3) if total else None,
        "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
        "recall": round(tp / (tp + fn), 3) if (tp + fn) else None,
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
    n = 0
    for s in scores:
        n += 1
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
        # 형식 문제가 몇 창에서 났는가. 점수에는 거의 드러나지 않으므로
        # 따로 세어 보고한다.
        "response_issue_counts": issues,
    }
