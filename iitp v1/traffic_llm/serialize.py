"""SceneSnapshot → LLM 입력 직렬화.

세 가지 표현을 제공한다.
  - to_json    : 구조화 JSON (툴/함수 호출, 정량 평가용)
  - to_text    : 자연어 상황 브리핑 (LLM 추론 입력으로 가장 효과적)
  - to_bev_ascii: 텍스트 BEV 지도 (공간 관계 보조)

토큰 예산이 있으므로 중요도(risk_score) 상위 max_actors 대만 포함하고,
잘라낸 대수는 본문에 명시한다 — LLM 이 "전부 봤다"고 오인하지 않도록.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Optional, Tuple

from .config import SerializeConfig
from .geometry import world_to_ego
from .kinematics import risk_score
from . import i18n, providers
from .schemas import ActorState, SceneSnapshot

MPS_TO_KPH = 3.6


def _r(x: Optional[float], digits: int = 1) -> Optional[float]:
    return None if x is None else round(float(x), digits)


def _round_probs(values: List[float], digits: int = 2) -> List[float]:
    """합이 1.00 으로 보이도록 반올림한다 (최대잉여법).

    각각 독립적으로 반올림하면 표시값의 합이 1 을 벗어난다 — 0.3417 짜리 3개는
    0.34×3 = 1.02 가 된다. 내부 계산은 정확히 1.0 인데 payload 만 어긋나 보이면
    확률 자체를 못 믿게 된다. 잔여를 큰 순서로 한 단위씩 배분해 맞춘다.
    """
    if not values:
        return []
    scale = 10 ** digits
    scaled = [v * scale for v in values]
    floors = [math.floor(s) for s in scaled]
    short = int(round(sum(scaled))) - sum(floors)
    # 잔여가 큰 항목부터 1 단위씩 올린다
    order = sorted(range(len(values)), key=lambda i: -(scaled[i] - floors[i]))
    for i in order[: max(0, short)]:
        floors[i] += 1
    return [f / scale for f in floors]


def rank_actors(snap: SceneSnapshot, limit: int) -> Tuple[List[ActorState], int]:
    """중요도 순 정렬 후 상위 limit 대 반환. (선택된 목록, 잘라낸 대수)"""
    scored = sorted(
        snap.actors, key=lambda a: -risk_score(a, snap.interactions)
    )
    return scored[:limit], max(0, len(scored) - limit)


# ---------------------------------------------------------------- JSON


def to_json(
    snap: SceneSnapshot, cfg: SerializeConfig, include_ground_truth: bool = False
) -> dict:
    """LLM 입력용 구조화 JSON.

    include_ground_truth 는 기본 False 다. 시나리오 정답(사고 발생 여부, 충돌
    주체·시각)을 입력에 넣으면 사고 예측 과제에서 정답 누수가 된다. 평가용
    레코드가 필요하면 to_evaluation_record() 를 쓴다.
    """
    lang = cfg.language
    _t = i18n.tx(lang)
    actors, dropped = rank_actors(snap, cfg.max_actors)
    d = cfg.round_digits
    kept = {a.actor_id for a in actors}

    out = {
        "timestamp_s": _r(snap.t, 2),
        "frame_idx": snap.frame_idx,
        "area": snap.area_name,
        "actor_count_total": len(snap.actors),
        "actor_count_included": len(actors),
        "actors_omitted": dropped,
        "actors": [
            {
                "id": a.actor_id,
                "role": (
                    _t["role_ego"] if a.kind == "ego" else _t["role_other"]
                ),
                "class": a.cls,
                "position_enu_m": [_r(a.world_xy[0], d), _r(a.world_xy[1], d)],
                "heading_deg": _r(a.heading_deg, 0),
                "speed_kph": _r((a.speed_mps or 0) * MPS_TO_KPH, 0)
                if a.speed_mps is not None
                else None,
                "accel_mps2": _r(a.accel_mps2, 2),
                "maneuver": i18n.label(a.maneuver, lang),
                "road": None
                if a.placement is None
                else {
                    "name": a.placement.road_name,
                    "direction": i18n.label(a.placement.direction_label, lang),
                    # 진행 방향을 확신할 수 없으면 direction·lane 이 반대일 수
                    # 있다. 단안 궤적으로는 방향이 뒤집히는 일이 있어, 확신
                    # 여부를 함께 넘겨 LLM 이 감안하게 한다.
                    "direction_confident": a.placement.direction_confident,
                    "lane": a.placement.lane_index,
                    "lane_count": a.placement.lane_count,
                    "lateral_offset_m": _r(a.placement.lateral_offset_m, d),
                    "speed_limit_kph": a.placement.speed_limit_kph,
                    "dist_to_next_junction_m": _r(
                        a.placement.dist_to_next_junction_m, 0
                    ),
                    "next_junction": a.placement.next_junction_id,
                },
                "predicted_paths": [
                    {
                        "maneuver": i18n.label(p.maneuver, lang),
                        # 같은 기동으로 갈 수 있는 도로가 둘 이상인 교차로가
                        # 있다. 진입 도로를 함께 주지 않으면 라벨·확률이 같은
                        # 항목이 여러 개 보여 중복으로 읽힌다.
                        "to_roads": p.to_roads,
                        "probability": prob,
                        # 0.1m 단위. 1m 로 반올림하면 차로 폭(3.5m)의 30% 라
                        # 좌표가 같은 값으로 뭉쳐 회전 궤적이 계단처럼 보인다.
                        "waypoints_1s_enu_m": [
                            [_r(w[0], 1), _r(w[1], 1)] for w in p.waypoints
                        ],
                        # 도로망 끝에서 잘렸다 = 그 앞은 예측 불가 (정지 아님)
                        "truncated": p.truncated,
                    }
                    for p, prob in zip(
                        a.predictions,
                        _round_probs([p.probability for p in a.predictions]),
                    )
                ],
                "observed_by": a.observed_by,
                "confidence": _r(a.confidence, 2),
                "position_quality": _r(a.position_quality, 2),
                "observed_range_m": _r(a.observed_range_m, 0),
                "track_age_s": _r(a.track_age_s, 1),
            }
            for a in actors
        ],
        "interactions": [
            {
                "type": it.kind,
                "subject": it.subject_id,
                "object": it.object_id,
                "gap_m": _r(it.gap_m, d),
                "headway_s": _r(it.headway_s, d),
                "ttc_s": _r(it.ttc_s, d),
                "note": i18n.interaction_note(it, lang),
            }
            for it in snap.interactions
            if it.subject_id in kept and it.object_id in kept
        ],
        "infrastructure": [
            {
                "id": s.infra_id,
                "position_enu_m": [_r(s.world_xy[0], d), _r(s.world_xy[1], d)],
                "mount_height_m": _r(s.height_m, 1),
                "heading_deg": _r(s.heading_deg, 0),
                "observations_contributed": s.n_observed,
                "road": None
                if s.placement is None
                else {
                    "name": s.placement.road_name,
                    "direction": i18n.label(s.placement.direction_label, lang),
                },
            }
            for s in snap.infrastructure
        ],
        "map_context": _localized_map_context(snap.map_context, lang),
    }

    if snap.scenario is not None:
        out["scenario"] = {
            "id": scenario_ref(snap.scenario.scenario_id)
            if cfg.redact_scenario_id
            else snap.scenario.scenario_id,
            "source": snap.scenario.source,
            "town": snap.scenario.town,
            **snap.scenario.available,
        }
        if include_ground_truth:
            out["scenario"]["ground_truth"] = snap.scenario.ground_truth
    return out


def _localized_map_context(ctx: dict, lang: str) -> dict:
    """map_context 사본. 지도 출처 키를 요청 언어의 문장으로 바꾼다."""
    if not ctx:
        return dict(ctx or {})
    out = dict(ctx)
    if out.get("map_source"):
        out["map_source"] = i18n.map_source(out["map_source"], lang)
    return out


def scenario_ref(scenario_id: str) -> str:
    """시나리오 id → 불투명 참조 토큰.

    DeepAccident 의 id 는 "type1_subtype1_accident/Town03_..." 처럼 분할명을
    담고 있어 사고 예측 과제의 정답을 노출한다. 결정적 해시로 치환해 payload
    와 정답 파일을 짝지을 수는 있게 하면서 분할명은 감춘다.
    """
    if not scenario_id:
        return ""
    h = hashlib.sha1(scenario_id.encode("utf-8")).hexdigest()[:12]
    return f"scn_{h}"


def to_evaluation_record(snap: SceneSnapshot, cfg: SerializeConfig) -> dict:
    """채점용 레코드. 정답을 포함하므로 LLM 입력에 넣지 말 것."""
    return {
        "timestamp_s": _r(snap.t, 2),
        "frame_idx": snap.frame_idx,
        "scenario_id": snap.scenario.scenario_id if snap.scenario else "",
        "ground_truth": dict(snap.scenario.ground_truth) if snap.scenario else {},
        "actor_ids": [a.actor_id for a in snap.actors],
        "interaction_count": len(snap.interactions),
        "min_ttc_s": min(
            (it.ttc_s for it in snap.interactions if it.ttc_s is not None),
            default=None,
        ),
    }


# ---------------------------------------------------------------- 자연어


def _map_conventions(ctx: dict, lang: str) -> str:
    """지도 규약 한 줄: 통행측 · 차선 번호 기준 · 차선폭 · 도로/교차로 수."""
    t = i18n.tx(lang)
    bits = [
        t["conv_drive_right"]
        if (ctx.get("drive_side") or "right") == "right"
        else t["conv_drive_left"],
        t["conv_lane_median"]
        if (ctx.get("lane_numbering") or "from_median") == "from_median"
        else t["conv_lane_curb"],
    ]
    if ctx.get("lane_width_m"):
        bits.append(t["conv_lane_width"].format(w=ctx["lane_width_m"]))
    if ctx.get("road_count"):
        bits.append(
            t["conv_map_size"].format(
                roads=ctx["road_count"], junctions=ctx.get("junction_count", 0)
            )
        )
    return t["conv_prefix"] + " | ".join(bits)


def _describe_actor(a: ActorState, cfg: SerializeConfig) -> str:
    lang = cfg.language
    t = i18n.tx(lang)
    bits: List[str] = []
    if a.placement is not None:
        p = a.placement
        lane = (
            t["lane_of"].format(count=p.lane_count, idx=p.lane_index)
            if p.lane_index is not None
            else t["lane_unknown"]
        )
        arrow = "" if p.direction_confident else t["dir_unsure"]
        direction = i18n.label(p.direction_label, lang)
        bits.append(f"{p.road_name} {direction}{arrow} {lane}")
        if p.speed_limit_kph:
            bits.append(t["speed_limit"].format(v=p.speed_limit_kph))
        if p.dist_to_next_junction_m is not None and p.dist_to_next_junction_m < 150:
            bits.append(t["to_junction"].format(d=p.dist_to_next_junction_m))
    else:
        bits.append(t["no_road"])

    if a.speed_mps is not None:
        bits.append(f"{a.speed_mps * MPS_TO_KPH:.0f}km/h")
    # 방위각(숫자)은 구조화 JSON 에만 있던 값이다. 도로 진행방향 라벨("북서행")은
    # 8방위로 뭉개져 있어 충돌 기하를 따질 수 없다 — 정지 차량 옆을 지나갈 때
    # 측방 여유가 방위 차이 몇 도에서 갈린다. JSON 블록을 끄면 이 값이 통째로
    # 사라지므로 자연어에도 싣는다.
    if a.heading_deg is not None:
        bits.append(t["heading"].format(h=a.heading_deg))
    if a.accel_mps2 is not None and abs(a.accel_mps2) > 0.5:
        bits.append(f"{a.accel_mps2:+.1f}m/s²")
    bits.append(i18n.label(a.maneuver, lang))

    if a.predictions:
        top = max(a.predictions, key=lambda p: p.probability)
        if len(a.predictions) == 1:
            bits.append(t["pred_one"].format(m=i18n.label(top.maneuver, lang)))
        else:
            alts = ", ".join(
                t["pred_alt"].format(
                    m=i18n.label(p.maneuver, lang), pct=p.probability * 100
                )
                for p in sorted(a.predictions, key=lambda x: -x.probability)[:3]
            )
            bits.append(t["pred_many"].format(alts=alts))

    if a.kind == "observed":
        obs = t["seen_by"].format(
            who=", ".join(a.observed_by),
            conf=a.confidence,
            pq=a.position_quality,
        )
        if a.observed_range_m is not None:
            obs += t["seen_range"].format(r=a.observed_range_m)
        obs += ")"
        bits.append(obs)
        if (a.observed_range_m or 0) > 40.0 or a.position_quality < 0.5:
            bits.append(t["uncertain"])
    return " | ".join(bits)


def describe_interaction(it, cfg: SerializeConfig) -> str:
    """상호작용 한 줄. 설명문은 구조화 필드에서 언어별로 만든다."""
    t = i18n.tx(cfg.language)
    parts = [
        f"{it.subject_id} → {it.object_id}",
        i18n.interaction_note(it, cfg.language) or it.kind,
    ]
    if it.gap_m is not None:
        parts.append(t["gap"].format(g=it.gap_m))
    if it.headway_s is not None:
        parts.append(t["headway"].format(h=it.headway_s))
    if it.ttc_s is not None:
        parts.append(f"TTC {it.ttc_s:.1f}s")
    return " | ".join(parts)


def to_text(snap: SceneSnapshot, cfg: SerializeConfig) -> str:
    lang = cfg.language
    t = i18n.tx(lang)
    actors, dropped = rank_actors(snap, cfg.max_actors)
    egos = [a for a in actors if a.kind == "ego"]
    others = [a for a in actors if a.kind == "observed"]
    kept = {a.actor_id for a in actors}

    lines: List[str] = []
    header = t["snap_header"].format(t=snap.t)
    if snap.frame_idx is not None:
        header += t["snap_frame"].format(f=snap.frame_idx)
    if snap.area_name:
        header += t["snap_area"].format(area=snap.area_name)
    header += ")"
    lines.append(header)

    if snap.scenario is not None:
        sc = snap.scenario
        bits = []
        if sc.source:
            bits.append(t["bg_source"].format(v=sc.source))
        if sc.town:
            bits.append(t["bg_town"].format(v=sc.town))
        for k, v in sc.available.items():
            if v:
                bits.append(f"{k}={v}")
        lines.append(t["bg_prefix"] + ", ".join(bits))

    counts = t["counts"].format(ego=len(egos), other=len(others))
    if snap.infrastructure:
        counts += t["counts_infra"].format(n=len(snap.infrastructure))
    counts += t["counts_total"].format(total=len(snap.actors), kept=len(actors))
    counts += t["counts_dropped"].format(n=dropped) if dropped else ")"
    lines.append(counts)

    if snap.map_context.get("map_source"):
        lines.append(
            t["map_note"].format(
                src=i18n.map_source(snap.map_context["map_source"], lang)
            )
        )
    # 지도 규약. "1차선"이 중앙선쪽인지 가장자리쪽인지, 통행이 좌측인지 우측인지를
    # 모르면 차선 번호와 좌/우 표현을 반대로 읽는다. 구조화 JSON 의 map_context
    # 에만 있던 값이라, 블록을 끄면 규약을 알 길이 없어진다.
    lines.append(_map_conventions(snap.map_context, lang))

    lines.append(t["sec_ego"])
    for a in egos:
        lines.append(f"- {a.actor_id}: {_describe_actor(a, cfg)}")

    lines.append(t["sec_others"])
    if not others:
        lines.append(t["none_others"])
    for a in others:
        # 가장 가까운 관측차량 기준 상대위치를 함께 제공
        rel = ""
        if egos:
            ref = min(egos, key=lambda e: math.dist(e.world_xy, a.world_xy))
            if ref.heading_deg is not None:
                x, y = world_to_ego(a.world_xy, ref.world_xy, ref.heading_deg)
                side = (
                    t["side_left"]
                    if y > 1.0
                    else (t["side_right"] if y < -1.0 else t["side_center"])
                )
                rel = t["rel_pos"].format(
                    ref=ref.actor_id,
                    fb=t["fb_front"] if x >= 0 else t["fb_back"],
                    x=abs(x),
                    side=side,
                    y=abs(y),
                )
        lines.append(f"- {a.actor_id}({a.cls}): {_describe_actor(a, cfg)}{rel}")

    if snap.infrastructure:
        lines.append(t["sec_infra"])
        for si in snap.infrastructure:
            bits = [t["infra_height"].format(h=si.height_m)]
            if si.placement is not None:
                bits.append(t["infra_near"].format(road=si.placement.road_name))
            if si.heading_deg is not None:
                bits.append(t["infra_aim"].format(b=si.heading_deg))
            bits.append(t["infra_nobs"].format(n=si.n_observed))
            lines.append(f"- {si.infra_id}: {' | '.join(bits)}")

    shown = [
        it
        for it in snap.interactions
        if it.subject_id in kept and it.object_id in kept
    ]
    lines.append(t["sec_inter"])
    if not shown:
        lines.append(t["none_inter"])
    for it in sorted(
        shown, key=lambda x: (x.ttc_s if x.ttc_s is not None else 1e9)
    ):
        lines.append("- " + describe_interaction(it, cfg))

    if cfg.include_bev_ascii and egos:
        lines.append(t["sec_bev"].format(ref=egos[0].actor_id))
        lines.append(to_bev_ascii(snap, egos[0], cfg))

    return "\n".join(lines)


# ---------------------------------------------------------------- ASCII BEV


def to_bev_ascii(
    snap: SceneSnapshot,
    ref: ActorState,
    cfg: SerializeConfig,
    cols: int = 31,
    rows: int = 21,
) -> str:
    """기준차량 중심 조감도. 위=전방, 왼쪽=좌측."""
    lang = cfg.language
    t = i18n.tx(lang)
    if ref.heading_deg is None:
        return t["bev_no_heading"]

    rng = cfg.bev_range_m
    grid = [["." for _ in range(cols)] for _ in range(rows)]
    labels: List[str] = []
    # 기호는 액터마다 유일해야 한다 — 같은 기호가 두 번 나오면 범례를 읽을 수 없다
    symbols = "123456789ABCDFGHJKLMNPQRSTUVWXYZ"
    next_sym = 0

    for a in snap.actors:
        x, y = world_to_ego(a.world_xy, ref.world_xy, ref.heading_deg)
        if abs(x) > rng or abs(y) > rng / 2:
            continue
        r = int((rng - x) / (2 * rng) * (rows - 1))
        c = int((rng / 2 - y) / rng * (cols - 1))
        r = min(max(r, 0), rows - 1)
        c = min(max(c, 0), cols - 1)
        if a.actor_id == ref.actor_id:
            mark = "@"
        elif next_sym < len(symbols):
            mark = symbols[next_sym]
            next_sym += 1
        else:
            mark = "*"  # 기호 소진
        if grid[r][c] != ".":
            # 격자 해상도 한계로 겹친 경우 — 범례에 표시해 오해를 막는다
            labels.append(
                f"{mark}={a.actor_id}({x:+.0f},{y:+.0f}, {t['bev_overlap']})"
            )
            continue
        grid[r][c] = mark
        role = t["bev_ref"] if a.kind == "ego" else a.cls
        labels.append(f"{mark}={a.actor_id}/{role}({x:+.0f},{y:+.0f})")

    body = "\n".join("".join(row) for row in grid)
    legend = "  ".join(labels)
    return (
        f"```\n{body}\n```\n"
        + t["bev_legend"].format(n=rng, e=rng / 2)
        + legend
    )


# ---------------------------------------------------------------- 프롬프트 조립

# 하위호환 별칭. 언어별 프롬프트는 i18n.system_prompt('snapshot', lang) 로 얻는다.
SYSTEM_PROMPT_KO = """당신은 협력형 자율주행(V2X) 교통 상황 분석 전문가입니다.
여러 대의 차량이 탑재 카메라로 수집한 영상과 해당 지역 도로 지도를 융합한
구조화 교통 상황 데이터를 입력받습니다.

데이터 규약:
- 위치는 지역 ENU 평면 좌표 [m], e=동쪽, n=북쪽입니다.
- 방위각은 진북 기준 시계방향입니다.
- 차선 번호는 1이 중앙선쪽입니다.
- confidence 는 **차량이 존재한다는 확신도**, position_quality 는 **위치 추정의
  정확도**입니다. 둘은 다릅니다: 바로 앞의 차량은 존재가 확실하지만 bbox 가
  절단되어 위치가 부정확할 수 있습니다. 여러 관측자가 본 객체는 둘 다 높습니다.
- actors_omitted 가 0보다 크면 일부 차량이 토큰 예산으로 생략된 것입니다.
  생략된 차량에 대해 단정적 판단을 하지 마십시오.
- 위치/속도는 단안 카메라 추정값이므로 오차가 있습니다. 특히 원거리
  (40m 초과) 및 confidence 0.5 미만 객체는 불확실성을 명시하십시오.
- predicted_paths 의 확률은 **도로 구조와 차선 위치에서 유도한 사전확률**입니다.
  방향지시등·제동등·운전자 의도를 관측한 값이 아니므로, 특정 차량이 회전할
  것이라고 단정하지 마십시오.
- maneuver 가 '차선유지' 인 것은 차선변경이 없었다는 뜻이 아니라, 관측 조건상
  차선변경으로 판정할 근거가 부족하다는 뜻일 수 있습니다. 특히 관측거리가
  급변하는 원거리 차량은 기동 판정이 보류됩니다.
- 이 데이터에는 신호등 상태, 정지선, 차선 표시 종류(실선/점선)가 포함되어
  있지 않습니다. 해당 정보가 필요한 판단에는 그 한계를 밝히십시오.
- infrastructure 항목은 노변 고정 센서입니다. 교통 참여자가 아니므로 정차
  차량으로 해석하지 마십시오. 인프라가 본 객체는 관측차량 사각지대를 보완합니다.
- map_context.map_source 가 '궤적 합성'이면 도로지도가 관측된 통행 궤적에서
  역추정된 것입니다. 차량이 지나가지 않은 차선·도로는 지도에 없고, 차선 수는
  하한입니다. '일방통행' 표기도 반대 방향이 관측되지 않았다는 뜻일 수 있습니다."""


def build_messages(
    snap: SceneSnapshot,
    question: str,
    cfg: SerializeConfig,
    include_json: Optional[bool] = None,
) -> dict:
    """LLM API 요청 본문(dict) 생성.

    형식은 `cfg.provider` ('claude' | 'openai' | 'gemini')가 정한다.
    Claude 경로는 system 블록에 cache_control 을 두어 반복 호출 시 프리픽스를
    캐싱한다 (스냅샷은 매 호출 달라지므로 캐시 브레이크포인트 뒤에 온다).
    """
    if include_json is None:
        include_json = cfg.include_json_block
    blocks = [to_text(snap, cfg)]
    if include_json:
        blocks.append(
            i18n.tx(cfg.language)["json_label"]
            + "\n```json\n"
            + json.dumps(to_json(snap, cfg), ensure_ascii=False, indent=1)
            + "\n```"
        )
    blocks.append(i18n.tx(cfg.language)["question_label"].format(q=question))

    return providers.build_request(
        cfg.provider,
        system=i18n.system_prompt("snapshot", cfg.language),
        blocks=blocks,
        schema=None,  # 단일 스냅샷 질의는 자유 서술 응답
        model=cfg.model,
        extra=cfg.provider_extra,
    )
