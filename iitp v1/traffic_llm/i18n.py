"""LLM 입력 문자열의 한국어/영어 전환.

기본 언어는 한국어이고, `SerializeConfig.language='en'` 이면 **모델에게 보내는
모든 문자열**이 영어로 나온다 (자연어 브리핑, 윈도우 텍스트, 질문, system
프롬프트, JSON 안의 라벨과 스키마 설명).

설계 규약
    - 파이프라인이 데이터 모델에 저장하는 라벨(진행방향, 기동, 예측 경로)의
      **정본은 한국어**다. 여기서 직렬화 시점에 번역한다. 파이프라인 내부를
      키로 바꾸면 로직·테스트가 전부 흔들리는데 얻는 것이 없다 — 라벨은 이
      패키지만 생성하는 닫힌 집합이고, 번역은 출력 경계 한 곳에서만 필요하다.
    - 사전에 없는 값은 **그대로 통과**시킨다. 도로명처럼 데이터에서 온 문자열은
      번역 대상이 아니다.
    - 숫자가 섞인 문장은 문자열 치환이 아니라 **형식 템플릿**(TX)으로 만든다.
    - 코드 주석·콘솔 출력·문서는 계속 한국어를 쓴다. 번역 대상은 모델이 읽는
      문자열뿐이다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

LANGS: Tuple[str, ...] = ("ko", "en")
DEFAULT_LANG = "ko"


def normalize_lang(lang: Optional[str]) -> str:
    """알 수 없는 값은 기본 언어로. 대소문자·지역코드(en-US)를 허용한다."""
    if not lang:
        return DEFAULT_LANG
    base = str(lang).strip().lower().replace("_", "-").split("-")[0]
    return base if base in LANGS else DEFAULT_LANG


# ---------------------------------------------------------------- 라벨 사전
#
# 정본(한국어) → 언어별 표기. 파이프라인이 만드는 닫힌 집합만 담는다.

LABELS: Dict[str, Dict[str, str]] = {
    # roadmap.direction_label
    "북행": {"en": "northbound"},
    "북동행": {"en": "northeastbound"},
    "동행": {"en": "eastbound"},
    "남동행": {"en": "southeastbound"},
    "남행": {"en": "southbound"},
    "남서행": {"en": "southwestbound"},
    "서행": {"en": "westbound"},
    "북서행": {"en": "northwestbound"},
    # kinematics.classify_maneuver
    "속도 미확정": {"en": "speed unknown"},
    "정지": {"en": "stopped"},
    "좌회전 중": {"en": "turning left"},
    "우회전 중": {"en": "turning right"},
    "차선변경(좌)": {"en": "changing lane (left)"},
    "차선변경(우)": {"en": "changing lane (right)"},
    "가속 중": {"en": "accelerating"},
    "감속 중": {"en": "decelerating"},
    "차선유지": {"en": "keeping lane"},
    "미확인": {"en": "unclassified"},
    # prediction / roadmap turn labels
    "직진": {"en": "straight"},
    "좌회전": {"en": "left turn"},
    "우회전": {"en": "right turn"},
    "유턴": {"en": "U-turn"},
    "정지 유지": {"en": "remains stopped"},
    "불확실": {"en": "uncertain"},
    "속도 미확정 — 경로 예측 보류": {
        "en": "speed unknown — path prediction withheld"
    },
    "등속 직진(지도 미매칭)": {
        "en": "constant-speed straight (no road match)"
    },
}

# 지도 출처는 고정 열거값이므로 **키**로 저장하고 여기서 문장을 만든다.
# 데이터 모델에 산문을 넣으면 언어 전환이 문자열 치환이 되어 깨진다.
MAP_SOURCES: Dict[str, Dict[str, str]] = {
    "inferred_from_trajectories": {
        "ko": "궤적 합성 (차선 수는 관측 하한)",
        "en": "synthesized from trajectories (lane counts are lower bounds)",
    },
    "geojson": {"ko": "GeoJSON 도로망", "en": "GeoJSON road network"},
    "opendrive": {"ko": "OpenDRIVE (.xodr)", "en": "OpenDRIVE (.xodr)"},
}


def label(value: Optional[str], lang: str = DEFAULT_LANG) -> str:
    """라벨 번역. 사전에 없으면 원문 그대로."""
    if value is None:
        return ""
    if normalize_lang(lang) == "ko":
        return value
    entry = LABELS.get(value)
    return entry.get(normalize_lang(lang), value) if entry else value


def map_source(value: Optional[str], lang: str = DEFAULT_LANG) -> str:
    """지도 출처 키 → 문장. 키가 아니면(파일명 등) 원문 그대로."""
    if not value:
        return ""
    entry = MAP_SOURCES.get(value)
    if entry is None:
        return value
    return entry.get(normalize_lang(lang), entry["ko"])


# ---------------------------------------------------------------- 상호작용 설명


def interaction_note(it, lang: str = DEFAULT_LANG) -> str:
    """상호작용 설명문. 구조화 필드에서 언어별로 만든다.

    구조화 필드가 없는 Interaction(직접 만든 것 등)은 저장된 note 를 그대로 쓴다.
    """
    t = tx(lang)
    kind = getattr(it, "kind", "")
    if kind == "following":
        return t["it_following"]
    if kind == "crossing":
        bits = []
        jid = getattr(it, "junction_id", None)
        if jid:
            bits.append(t["it_junction"].format(jid=jid))
        gap = getattr(it, "arrival_gap_s", None)
        if gap is not None:
            bits.append(t["it_arrival_gap"].format(gap=gap))
        head = ", ".join(bits)
        conflict = getattr(it, "conflict", "") or ""
        probs = getattr(it, "turn_probs", None) or []
        if conflict == "orthogonal":
            tail = t["it_orthogonal"]
        elif conflict == "oncoming_turn":
            tail = t["it_oncoming_turn"].format(
                probs=", ".join(
                    t["it_turn_prob"].format(aid=a, pct=p * 100) for a, p in probs
                )
            )
        else:
            return head or getattr(it, "note", "") or kind
        return f"{head} | {tail}" if head else tail
    if kind == "lane_change_conflict":
        lane = getattr(it, "target_lane", None)
        if lane is not None:
            return t["it_lane_conflict"].format(lane=lane)
    return getattr(it, "note", "") or kind


# ---------------------------------------------------------------- 형식 템플릿

_TX_KO: Dict[str, str] = {
    # --- serialize._describe_actor
    "lane_of": "{count}차선 중 {idx}차선",
    "lane_unknown": "차선 미확정",
    "dir_unsure": "(진행방향 미확정)",
    "speed_limit": "제한속도 {v:.0f}km/h",
    "heading": "방위 {h:.0f}°",
    "conv_prefix": "※ 지도 규약: ",
    "conv_drive_right": "우측통행",
    "conv_drive_left": "좌측통행",
    "conv_lane_median": "1차선은 중앙선쪽",
    "conv_lane_curb": "1차선은 가장자리쪽",
    "conv_lane_width": "차선폭 {w:.2f}m",
    "conv_map_size": "도로 {roads}개·교차로 {junctions}개",
    "to_junction": "다음 교차로까지 {d:.0f}m",
    "no_road": "도로 미매칭",
    "pred_one": "예상경로: {m}",
    "pred_many": "예상경로: {alts}",
    "pred_alt": "{m} {pct:.0f}%",
    "seen_by": "관측: {who} (존재확신 {conf:.2f}, 위치정확도 {pq:.2f}",
    "seen_range": ", 관측거리 {r:.0f}m",
    "uncertain": "※위치·속도 불확실",
    # --- serialize.to_text
    "snap_header": "# 교통 상황 스냅샷 (t = {t:.1f}s",
    "snap_frame": ", frame {f}",
    "snap_area": ", 지역: {area}",
    "bg_prefix": "배경: ",
    "bg_source": "출처 {v}",
    "bg_town": "지도 {v}",
    "counts": "관측차량 {ego}대, 주변차량 {other}대",
    "counts_infra": ", 노변 인프라 {n}기",
    "counts_total": " (전체 차량 {total}대 중 상위 {kept}대 수록",
    "counts_dropped": ", {n}대 생략)",
    "map_note": (
        "※ 도로지도 출처: {src} — "
        "차량이 지나가지 않은 차선·도로는 지도에 없을 수 있습니다."
    ),
    "sec_ego": "\n## 관측차량 (탑재 카메라로 주변을 관측)",
    "sec_others": "\n## 주변차량",
    "none_others": "- 검출된 주변차량 없음",
    "rel_pos": " [{ref} 기준 {fb} {x:.0f}m, {side}측 {y:.1f}m]",
    "fb_front": "전방",
    "fb_back": "후방",
    "side_left": "좌",
    "side_right": "우",
    "side_center": "정면",
    "sec_infra": "\n## 노변 인프라 센서 (V2X, 교통 참여자 아님)",
    "infra_height": "설치높이 {h:.1f}m",
    "infra_near": "{road} 인접",
    "infra_aim": "지향 {b:.0f}°",
    "infra_nobs": "이 시각 관측 {n}건",
    "sec_inter": "\n## 상호작용 및 주의 상황",
    "none_inter": "- 특이사항 없음",
    "gap": "간격 {g:.1f}m",
    "headway": "헤드웨이 {h:.1f}s",
    "sec_bev": "\n## BEV 개략도 (기준: {ref})",
    "bev_no_heading": "(기준차량 방위각 미확정 — BEV 생략)",
    "bev_legend": (
        "@=기준차량 | 축척: 세로 ±{n:.0f}m(전/후), 가로 ±{e:.0f}m(좌/우) | "
    ),
    "bev_ref": "관측차량",
    "bev_overlap": "격자중복",
    "role_ego": "관측차량",
    "role_other": "주변차량",
    # --- 상호작용 설명
    "it_following": "선행차 추종",
    "it_junction": "교차로 {jid}",
    "it_arrival_gap": "도달시간차 {gap:.1f}s",
    "it_orthogonal": "직교 진입 상충 예상",
    "it_oncoming_turn": (
        "대향 좌회전 상충 가능 — 좌회전/유턴 확률"
        "(차선위치 기반 사전확률): {probs}"
    ),
    "it_turn_prob": "{aid} {pct:.0f}%",
    "it_lane_conflict": "{lane}차선 진입 경합",
    # --- serialize.build_messages
    "json_label": "구조화 데이터:",
    "question_label": "질문: {q}",
    # --- accident_qa._actor_row
    "row_speed_unknown": " 미확정 ",
    "row_lane": "{idx}/{count}차선",
    "row_lane_unknown": "차선미정",
    "row_junction": " 교차로 {d:.0f}m",
    # --- accident_qa.render_window_text
    "win_header": "# 관측 윈도우 {label} (t={t0}~{t1}초, 스냅샷 {n}개)",
    "win_counts": "마지막 시점 기준 관측차량 {ego}대, 주변차량 {other}대",
    "win_counts_dropped": " (중요도 하위 {n}대 생략)",
    "win_infra_row": "- {sid}: 설치높이 {h:.1f}m, 이 시각 관측 {n}건",
    "sec_history": "\n## 차량별 시간 경과 ({t0}~{t1}초, {stride:g}초 간격)",
    "hist_head": "### {aid} ({role}) — 관측: {obs}",
    # 액터 행과 **같은 낱말**을 쓴다. system 프롬프트가 존재확신과 위치정확도는
    # 다르다고 명시하는데, 여기서 "신뢰도"라고 쓰면 어느 쪽인지 알 수 없다.
    "hist_conf": ", 존재확신 {conf:.2f}",
    "hist_range": ", 관측거리 {r:.0f}m",
    "hist_partial": "   (이 윈도우에서 {seen}/{total} 시점만 관측됨)",
    "role_ego": "관측차량",
    "win_gone": (
        "\n※ 윈도우 중간까지 관측되다 마지막 시점에는 없는 차량: {ids} "
        "(시야 이탈 또는 관측 실패 — 사라졌다고 단정할 수 없음)"
    ),
    "sec_pairs": "\n## 윈도우 동안 서로 접근한 차량 쌍",
    "none_pairs": "- 접근율 0.3m/s 이상으로 가까워진 쌍 없음",
    "pair_row": (
        "- {a} ↔ {b}: 거리 {d0:.1f}m → {d1:.1f}m "
        "(접근율 {rate:.1f}m/s, 선형연장 접촉까지 {eta}"
    ),
    "pair_same_road": ", 동일 도로",
    "pair_note": (
        "  ※ 선형연장 값은 관측된 거리 변화율을 그대로 늘린 참고값이며, "
        "회전·제동·신호를 반영한 경로 예측이 아닙니다."
    ),
    "sec_last": "\n## 마지막 관측 시점 상세 (t={t}초)",
    # --- accident_qa.build_question
    "q_intro": (
        "위 {win}초 관측 구간(t={t0}~{t1}초)을 근거로, **마지막 관측 시점 이후 "
        "미래 {horizon}초**의 사고 발생 가능성을 1초 구간별로 판단하십시오."
    ),
    "q_bucket_head": "구간 정의 (k번째 구간 = 마지막 관측 시점 이후 (k-1, k] 초):",
    "q_bucket_row": "  k={k}: ({lo}, {hi}] 초",
    "q_answer_all": "각 구간 k=1..{n} 을 **모두** 답하십시오.",
    "q_rule_negative": (
        "- 사고가 예상되지 않으면: accident_expected=false, "
        "involved_actor_ids=[], reason 에 안전하다고 본 근거를 한 문장."
    ),
    "q_rule_positive": (
        "- 사고가 예상되면: accident_expected=true, involved_actor_ids 에 "
        "관련 차량 id 를 모두, reason 에 어떤 상충 때문인지 구체적으로 "
        "(예: 교차로 직교 진입 상충, 선행차 추종 중 TTC 부족, 차선변경 경합)."
    ),
    "q_rule_ids": (
        "- involved_actor_ids 에는 **액터 id** 를 쓰십시오 (관측차량은 "
        "EGO_ 로 시작, 그 외는 V/T/P/N 등으로 시작). '관측 관계' 절의 관측자 "
        "이름(예: ego_vehicle)은 액터 id 가 아닙니다 — 그 관측자의 본체를 "
        "가리키려면 같은 줄에 적힌 본체 액터 id(예: EGO_ego_vehicle)를 쓰십시오."
    ),
    "q_rule_limits": (
        "- 판단에 영향을 준 데이터 한계가 있으면 data_limitations 에 적으십시오."
    ),
    # --- 관측 관계 (perception 블록 · 자연어 절)
    "perception_note": (
        "observed_now 에 없는 관측자는 그 시각에 그 차량을 인지하지 못했다 "
        "(사각지대·화각 밖·가려짐). only_this_observer 는 그 관측자만 본 차량, "
        "즉 다른 관측자의 사각지대를 메운 몫이다. n_cameras 와 "
        "azimuth_coverage_deg 는 그 관측자의 카메라 대수와 덮는 방위 범위다 — "
        "360° 를 덮는 관측자가 못 본 차량은 화각 밖이 아니라 가려진 것이다."
    ),
    "sec_perception": "\n## 관측 관계 (누가 누구를 인지했는가)",
    "perc_row": "- {oid}({kind}): {n}대 인지",
    # 관측자 이름과 그 관측자 **본체**의 액터 id 는 다른 문자열이다
    # ('ego_vehicle' vs 'EGO_ego_vehicle'). 이 매핑이 없으면 답변에 관측자
    # 이름을 쓰게 되고, 차량 지목이 전부 오답 처리된다 (실제로 그랬다).
    "perc_self": " | 본체 액터 id {aid}",
    "perc_cams": " | 카메라 {n}대 {cov}° 커버",
    "perc_only": " | 단독 인지 {ids}",
    "perc_none": "- 이 시각에 인지한 차량 없음",
    "kind_vehicle": "차량",
    "kind_infra": "인프라",
    # --- 정답/manifest 주석 (모델에는 가지 않지만 산출물 전체가 한 언어여야 한다)
    "bucket_rule": "k번째 구간 = 마지막 관측 시점 이후 (k-1, k] 초",
    "gt_no_collision": "이 시나리오에는 충돌 기록이 없다 — 전 구간 정답은 '사고 없음'.",
    "gt_time_unknown": "충돌 시각을 추정하지 못했다 — 채점에서 제외하는 것이 안전하다.",
    "gt_beyond_horizon": (
        "충돌은 예측 구간({horizon:g}초) 밖 {lead:.1f}초 뒤에 일어난다 — "
        "전 구간 정답은 '사고 없음'."
    ),
    "gt_lead": "충돌은 마지막 관측 시점 {lead:.1f}초 뒤에 일어난다.",
    "gt_unobserved": (
        "충돌 주체 중 {ids} 는 이 윈도우에서 관측되지 않았다 — LLM 이 이름을 "
        "댈 수 없으므로 차량 지목 정확도 채점에서 감안해야 한다."
    ),
    # --- 스키마 설명
    "sch_predictions": (
        "미래 1초 구간별 예측. k=1..N 을 **각각 정확히 한 번씩** 포함해야 한다 "
        "(같은 k 를 두 번 넣거나 빠뜨리지 말 것)."
    ),
    "sch_n_buckets": "이 질문의 구간 수는 N={n} 이므로 항목이 정확히 {n}개여야 한다.",
    "sch_k": "마지막 관측 시점 이후 몇 번째 1초 구간인가 (1부터)",
    "sch_interval": "절대 시각 구간, 예: '(8.0, 9.0]'",
    "sch_expected": "이 구간에 사고가 발생할 것으로 보는가",
    "sch_involved": (
        "사고 관련 차량의 **액터 id** (EGO_ 로 시작하는 관측차량 또는 "
        "V/T/P/N 등으로 시작하는 주변차량). 관측자 이름이 아니다. "
        "accident_expected 가 false 면 빈 배열."
    ),
    "sch_reason": "판단 근거. 사고 없음이면 안전하다고 본 이유를 한 문장으로.",
    "sch_overall": "윈도우 전체에 대한 2~3문장 요약",
    "sch_limits": "이 판단에 영향을 준 데이터 한계(있으면). 없으면 빈 문자열.",
}

_TX_EN: Dict[str, str] = {
    "lane_of": "lane {idx} of {count}",
    "lane_unknown": "lane unknown",
    "dir_unsure": "(travel direction unconfirmed)",
    "speed_limit": "speed limit {v:.0f}km/h",
    "heading": "heading {h:.0f} deg",
    "conv_prefix": "NOTE map conventions: ",
    "conv_drive_right": "right-hand traffic",
    "conv_drive_left": "left-hand traffic",
    "conv_lane_median": "lane 1 is the median-side lane",
    "conv_lane_curb": "lane 1 is the curb-side lane",
    "conv_lane_width": "lane width {w:.2f}m",
    "conv_map_size": "{roads} roads, {junctions} junctions",
    "to_junction": "{d:.0f}m to next junction",
    "no_road": "no road match",
    "pred_one": "predicted path: {m}",
    "pred_many": "predicted paths: {alts}",
    "pred_alt": "{m} {pct:.0f}%",
    "seen_by": "seen by {who} (existence {conf:.2f}, position quality {pq:.2f}",
    "seen_range": ", range {r:.0f}m",
    "uncertain": "NOTE: position/speed uncertain",
    "snap_header": "# Traffic snapshot (t = {t:.1f}s",
    "snap_frame": ", frame {f}",
    "snap_area": ", area: {area}",
    "bg_prefix": "Context: ",
    "bg_source": "source {v}",
    "bg_town": "map {v}",
    "counts": "{ego} observer vehicle(s), {other} other road user(s)",
    "counts_infra": ", {n} roadside sensor(s)",
    "counts_total": " (top {kept} of {total} road users included",
    "counts_dropped": ", {n} omitted)",
    "map_note": (
        "NOTE: road map source: {src} — lanes and roads that no vehicle drove "
        "may be missing from the map."
    ),
    "sec_ego": "\n## Observer vehicles (perceive via onboard cameras)",
    "sec_others": "\n## Other road users",
    "none_others": "- none detected",
    "rel_pos": " [rel. to {ref}: {fb} {x:.0f}m, {side} {y:.1f}m]",
    "fb_front": "ahead",
    "fb_back": "behind",
    "side_left": "left",
    "side_right": "right",
    "side_center": "dead ahead",
    "sec_infra": "\n## Roadside sensors (V2X, not road users)",
    "infra_height": "mounted {h:.1f}m",
    "infra_near": "beside {road}",
    "infra_aim": "aimed {b:.0f}°",
    "infra_nobs": "{n} detection(s) at this time",
    "sec_inter": "\n## Interactions and hazards",
    "none_inter": "- nothing notable",
    "gap": "gap {g:.1f}m",
    "headway": "headway {h:.1f}s",
    "sec_bev": "\n## BEV sketch (reference: {ref})",
    "bev_no_heading": "(reference vehicle heading unknown — BEV omitted)",
    "bev_legend": (
        "@=reference | scale: vertical +/-{n:.0f}m (front/back), "
        "horizontal +/-{e:.0f}m (left/right) | "
    ),
    "bev_ref": "observer",
    "bev_overlap": "cell shared",
    "role_ego": "observer vehicle",
    "role_other": "other road user",
    "it_following": "car following",
    "it_junction": "junction {jid}",
    "it_arrival_gap": "arrival time gap {gap:.1f}s",
    "it_orthogonal": "orthogonal entry conflict expected",
    "it_oncoming_turn": (
        "possible conflict with an oncoming left turn — left-turn/U-turn "
        "probability (prior from lane position): {probs}"
    ),
    "it_turn_prob": "{aid} {pct:.0f}%",
    "it_lane_conflict": "contention for entering lane {lane}",
    "json_label": "Structured data:",
    "question_label": "Question: {q}",
    "row_speed_unknown": " unknown ",
    "row_lane": "lane {idx}/{count}",
    "row_lane_unknown": "lane n/a",
    "row_junction": " junction {d:.0f}m",
    "win_header": (
        "# Observation window {label} (t={t0}-{t1}s, {n} snapshots)"
    ),
    "win_counts": (
        "At the last timestep: {ego} observer vehicle(s), "
        "{other} other road user(s)"
    ),
    "win_counts_dropped": " ({n} lowest-importance omitted)",
    "win_infra_row": "- {sid}: mounted {h:.1f}m, {n} detection(s) at this time",
    "sec_history": "\n## Per-vehicle time course ({t0}-{t1}s, every {stride:g}s)",
    "hist_head": "### {aid} ({role}) — seen by: {obs}",
    "hist_conf": ", confidence {conf:.2f}",
    "hist_range": ", range {r:.0f}m",
    "hist_partial": "   (observed at only {seen}/{total} timesteps in this window)",
    "role_ego": "observer vehicle",
    "win_gone": (
        "\nNOTE: observed earlier in the window but absent at the last "
        "timestep: {ids} (left the field of view or detection failed — "
        "do not conclude they are gone)"
    ),
    "sec_pairs": "\n## Vehicle pairs that closed on each other during the window",
    "none_pairs": "- no pair closed at 0.3m/s or more",
    "pair_row": (
        "- {a} <-> {b}: distance {d0:.1f}m -> {d1:.1f}m "
        "(closing rate {rate:.1f}m/s, linear-extrapolated contact in {eta}"
    ),
    "pair_same_road": ", same road",
    "pair_note": (
        "  NOTE: the linear-extrapolated value simply extends the observed "
        "range rate; it is not a path prediction accounting for turning, "
        "braking or signals."
    ),
    "sec_last": "\n## Last observed timestep in detail (t={t}s)",
    "q_intro": (
        "Based on the {win}s observation window above (t={t0}-{t1}s), judge "
        "the likelihood of an accident in each 1-second interval over the "
        "**{horizon}s following the last observed timestep**."
    ),
    "q_bucket_head": (
        "Interval definition (the k-th interval is (k-1, k] seconds after the "
        "last observed timestep):"
    ),
    "q_bucket_row": "  k={k}: ({lo}, {hi}] s",
    "q_answer_all": "Answer **all** intervals k=1..{n}.",
    "q_rule_negative": (
        "- If no accident is expected: accident_expected=false, "
        "involved_actor_ids=[], and one sentence in reason for why it looks safe."
    ),
    "q_rule_positive": (
        "- If an accident is expected: accident_expected=true, list every "
        "involved vehicle id in involved_actor_ids, and state the specific "
        "conflict in reason (e.g. orthogonal entry conflict at a junction, "
        "insufficient TTC while following, lane-change contention)."
    ),
    "q_rule_ids": (
        "- Use **actor ids** in involved_actor_ids (observer vehicles start "
        "with EGO_; others start with V/T/P/N). The observer names in the "
        "'Perception relations' section (e.g. ego_vehicle) are NOT actor ids — "
        "to refer to an observer's own vehicle, use the own actor id given on "
        "the same line (e.g. EGO_ego_vehicle)."
    ),
    "q_rule_limits": (
        "- If any data limitation affected your judgement, record it in "
        "data_limitations."
    ),
    "bucket_rule": (
        "the k-th interval is (k-1, k] seconds after the last observed timestep"
    ),
    "gt_no_collision": (
        "This scenario has no recorded collision — the answer is 'no accident' "
        "for every interval."
    ),
    "gt_time_unknown": (
        "The collision time could not be estimated — safest to exclude this "
        "window from scoring."
    ),
    "gt_beyond_horizon": (
        "The collision happens {lead:.1f}s later, beyond the {horizon:g}s "
        "prediction horizon — the answer is 'no accident' for every interval."
    ),
    "gt_lead": (
        "The collision happens {lead:.1f}s after the last observed timestep."
    ),
    "gt_unobserved": (
        "Of the vehicles involved in the collision, {ids} were not observed in "
        "this window — the model cannot name them, so allow for that when "
        "scoring vehicle-identification accuracy."
    ),
    "perception_note": (
        "An observer absent from observed_now did not perceive that vehicle at "
        "that time (blind spot, outside the field of view, or occluded). "
        "only_this_observer lists vehicles seen by that observer alone — its "
        "contribution to covering the other observers' blind spots. n_cameras "
        "and azimuth_coverage_deg give that observer's camera count and the "
        "azimuth range they cover — a vehicle missed by an observer with 360 "
        "degree coverage is occluded, not out of frame."
    ),
    "sec_perception": "\n## Perception relations (who perceived whom)",
    "perc_row": "- {oid} ({kind}): perceives {n}",
    "perc_self": " | own actor id {aid}",
    "perc_cams": " | {n} cameras, {cov} deg coverage",
    "perc_only": " | sole observer of {ids}",
    "perc_none": "- perceives nothing at this time",
    "kind_vehicle": "vehicle",
    "kind_infra": "infrastructure",
    "sch_predictions": (
        "Per-1-second-interval predictions. Include every k=1..N **exactly "
        "once** — do not repeat or omit a k."
    ),
    "sch_n_buckets": (
        "This question has N={n} intervals, so return exactly {n} items."
    ),
    "sch_k": (
        "Which 1-second interval after the last observed timestep (starting at 1)"
    ),
    "sch_interval": "Absolute time interval, e.g. '(8.0, 9.0]'",
    "sch_expected": "Do you expect an accident in this interval",
    "sch_involved": (
        "**Actor ids** of the vehicles involved (observer vehicles start with "
        "EGO_; others start with V/T/P/N). Not observer names. Empty array if "
        "accident_expected is false."
    ),
    "sch_reason": (
        "Grounds for the judgement. If no accident, one sentence on why it "
        "looks safe."
    ),
    "sch_overall": "2-3 sentence summary of the whole window",
    "sch_limits": (
        "Data limitations that affected this judgement, if any. Empty string "
        "if none."
    ),
}

_TX: Dict[str, Dict[str, str]] = {"ko": _TX_KO, "en": _TX_EN}


def tx(lang: str = DEFAULT_LANG) -> Dict[str, str]:
    """언어별 형식 템플릿 사전. 없는 키는 한국어로 되돌린다."""
    lg = normalize_lang(lang)
    if lg == "ko":
        return _TX_KO
    table = dict(_TX_KO)
    table.update(_TX[lg])
    return table


# ---------------------------------------------------------------- system 프롬프트

SYSTEM_PROMPT_SNAPSHOT: Dict[str, str] = {
    "ko": """당신은 협력형 자율주행(V2X) 교통 상황 분석 전문가입니다.
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
- road.direction_confident 가 false 면 진행 방향(및 차선 번호)이 반대일 수
  있습니다. 그 차량의 진행 방향을 전제로 단정하지 마십시오.
- 이 데이터에는 신호등 상태, 정지선, 차선 표시 종류(실선/점선)가 포함되어
  있지 않습니다. 해당 정보가 필요한 판단에는 그 한계를 밝히십시오.
- infrastructure 항목은 노변 고정 센서입니다. 교통 참여자가 아니므로 정차
  차량으로 해석하지 마십시오. 인프라가 본 객체는 관측차량 사각지대를 보완합니다.""",
    "en": """You are an expert analyst of cooperative-driving (V2X) traffic situations.
Your input is structured traffic-situation data produced by fusing onboard-camera
video from several vehicles with the road map of the area.

Data conventions:
- Positions are local ENU plane coordinates in metres: e=east, n=north.
- Headings are degrees clockwise from true north.
- Lane numbering starts at 1 on the centreline side.
- confidence is **how certain it is that the vehicle exists**; position_quality
  is **how accurate its position estimate is**. These differ: a vehicle right in
  front certainly exists, but a truncated bbox can make its position poor. An
  object seen by several observers scores high on both.
- If actors_omitted is greater than 0, some vehicles were dropped to fit the
  token budget. Do not make definitive claims about vehicles you cannot see.
- Positions and speeds come from monocular estimation and carry error. State the
  uncertainty explicitly for distant objects (beyond 40m) and for objects with
  confidence below 0.5.
- Probabilities in predicted_paths are **priors derived from road geometry and
  lane position**. They are not observations of turn signals, brake lights or
  driver intent, so do not assert that a particular vehicle will turn.
- maneuver = 'keeping lane' does not mean no lane change happened; it can mean
  the observing conditions gave insufficient grounds to call one. Manoeuvre
  classification is withheld for distant vehicles whose range changes rapidly.
- If road.direction_confident is false, the travel direction (and therefore the
  lane number) may be reversed. Do not build conclusions on that vehicle's
  direction of travel.
- This data contains no traffic-signal state, no stop lines, and no lane-marking
  types (solid/dashed). Where a judgement would need them, say so.
- infrastructure entries are fixed roadside sensors. They are not road users, so
  do not read them as parked vehicles. What they see fills the observer
  vehicles' blind spots.""",
}

SYSTEM_PROMPT_ACCIDENT: Dict[str, str] = {
    "ko": """당신은 협력형 자율주행(V2X) 교통 상황 분석 전문가입니다.
여러 대의 차량에 탑재된 카메라와 노변 인프라 센서가 수집한 관측을 융합하고 도로 지도와
정합한 교통 상황 시계열을 입력받아, **미래 구간별 사고 발생 가능성**을 판단합니다.

데이터 규약:
- 위치는 지역 ENU 평면 좌표 [m], e=동쪽, n=북쪽입니다. 방위각은 진북 기준 시계방향.
- 차선 번호는 1이 중앙선쪽입니다.
- 위치·속도는 센서 추정값이므로 오차가 있습니다. '관측거리' 40m 초과 또는
  '위치정확도'(position_quality) 0.5 미만 객체는 위치 불확실성이 큽니다.
- '존재확신'(confidence)과 '위치정확도'(position_quality)는 다른 값입니다.
  존재는 확실한데 위치만 부정확한 경우가 있습니다.
- '속도 미확정'은 정지를 뜻하지 않습니다. 추적 이력이 짧아 판정을 보류한 것입니다.
- 기동이 '차선유지'인 것은 차선변경이 없었다는 뜻이 아니라, 관측 조건상
  판정 근거가 부족하다는 뜻일 수 있습니다.
- '예상경로'(predicted_paths)의 확률은 도로 구조와 차선 위치에서 유도한
  **사전확률**입니다. 방향지시등·제동등을 관측한 값이 아니므로 회전 의도를
  단정하지 마십시오. '도로망 끝에서 잘림' 표시는 그 앞을 예측할 수 없다는 뜻이며
  정지 예측이 아닙니다.
- '선형연장 접촉까지' 값은 관측된 거리 변화율을 그대로 늘린 참고값입니다.
  회전·제동을 반영하지 않으므로 경로 예측이 아닙니다.
- '(진행방향 미확정)' 표시가 붙은 차량은 진행 방향과 차선 번호가 반대일 수 있습니다.
- '노변 인프라 센서' 절의 항목은 고정 센서이며 교통 참여자가 아닙니다. 정차
  차량으로 보지 마십시오.
- '지도 규약' 줄에 통행측·차선번호 기준·차선폭이 있습니다. 그 규약대로 읽으십시오.
- '도로지도 출처'가 '궤적 합성'이면 도로지도가 통행 궤적에서 역추정된 것입니다.
  차량이 지나가지 않은 차선·도로는 지도에 없고, 차선 수는 하한입니다.
- '관측 관계' 절은 어느 관측자가 어떤 차량을 인지했는지 보여 줍니다. 목록에 없는
  관측자는 그 시각에 그 차량을 인지하지 못했습니다 — 방위 360°를 덮는 관측자가
  못 본 차량은 화각 밖이 아니라 가려진 것입니다.
- 이 데이터에는 **신호등 상태, 정지선, 차선 표시 종류(실선/점선)가 없습니다.**
  신호 준수 여부를 전제로 판단하지 말고, 필요하면 그 한계를 밝히십시오.

판단 원칙:
- 데이터에 등장한 actor id 만 사용하고, 없는 차량을 만들지 마십시오.
- 근거는 관측된 기하·운동 정보(거리, 접근율, 차선, 교차로 도달시간, 헤드웨이,
  TTC)에 두십시오. 추측을 사실처럼 쓰지 마십시오.
- 사고가 예상되지 않는 구간은 accident_expected=false 로 두고 involved_actor_ids 를
  비우십시오. 위험을 과장해 모든 구간을 사고로 표시하지 마십시오.
- 반대로, 관측된 상충이 명확하면(임박한 TTC, 교차로 동시 진입, 차선변경 경합)
  주저하지 말고 사고로 표시하고 관련 차량을 모두 나열하십시오.""",
    "en": """You are an expert analyst of cooperative-driving (V2X) traffic situations.
Your input is a traffic-situation time series built by fusing observations from
several vehicles' onboard cameras and roadside infrastructure sensors and matching
them to the road map. Judge the **likelihood of an accident in each future
interval**.

Data conventions:
- Positions are local ENU plane coordinates in metres: e=east, n=north. Headings
  are degrees clockwise from true north.
- Lane numbering starts at 1 on the centreline side.
- Positions and speeds are sensor estimates and carry error. Objects beyond
  40m 'range', or with 'position quality' below 0.5, have large positional
  uncertainty.
- 'existence' (certainty the vehicle is there) and 'position quality'
  (accuracy of its location) are different values. An object can certainly
  exist yet be poorly located.
- 'speed unknown' does not mean stopped. It means the tracking history was too
  short to commit to a value.
- A maneuver of 'keeping lane' does not mean no lane change happened; it can
  mean the observing conditions gave insufficient grounds to call one.
- Probabilities under 'predicted path' are **priors** derived from road geometry
  and lane position. They are not observations of turn signals or brake lights,
  so do not assert turning intent. A 'truncated at the road-network edge' note
  means the path beyond it cannot be predicted — it is not a stop prediction.
- 'linear-extrapolated contact in' simply extends the observed range rate. It
  ignores turning and braking, so it is not a path prediction.
- A '(travel direction unconfirmed)' marker means the travel direction and lane
  number may be reversed.
- Entries in the 'Roadside infrastructure' section are fixed sensors, not road
  users. Do not read them as parked vehicles.
- The 'map conventions' line gives the traffic side, lane-numbering basis and
  lane width. Read the scene by those conventions.
- If the road-map source says it was synthesized from trajectories, the map was
  inferred from observed travel: lanes and roads nobody drove are absent, and
  lane counts are lower bounds.
- The 'Perception relations' section shows which observer perceived which
  vehicle. An observer absent from a line did not perceive that vehicle at that
  time — a vehicle missed by an observer covering 360 degrees is occluded, not
  out of frame.
- This data contains **no traffic-signal state, no stop lines, and no
  lane-marking types (solid/dashed)**. Do not assume signal compliance; where a
  judgement would need them, say so.

Judgement principles:
- Use only actor ids that appear in the data. Do not invent vehicles.
- Ground your reasoning in the observed geometry and motion (distance, closing
  rate, lane, time to junction, headway, TTC). Do not present guesses as facts.
- For intervals where no accident is expected, set accident_expected=false and
  leave involved_actor_ids empty. Do not inflate risk by marking every interval
  as an accident.
- Conversely, when an observed conflict is clear (imminent TTC, simultaneous
  junction entry, lane-change contention), mark it as an accident without
  hesitation and list every vehicle involved.""",
}


def system_prompt(kind: str, lang: str = DEFAULT_LANG) -> str:
    """kind: 'snapshot' | 'accident'."""
    table = (
        SYSTEM_PROMPT_ACCIDENT if kind == "accident" else SYSTEM_PROMPT_SNAPSHOT
    )
    return table[normalize_lang(lang)]


def prediction_schema(
    lang: str = DEFAULT_LANG, n_buckets: Optional[int] = None
) -> dict:
    """사고 예측 응답 스키마. description 을 요청 언어로 낸다.

    `n_buckets` 를 주면 구간 수를 **설명문에 적어** 모델에게 알린다.

    스키마 제약(`minItems`/`maxItems`/`minimum`)으로 못박지 **않는다**. OpenAI
    strict 모드가 그 키워드를 지원하지 않아, 넣으면 provider 별로 스키마가
    달라진다 — 내용이 provider 와 무관하게 같다는 이 프로젝트의 규약이 깨진다.
    대신 설명문에서 요구하고, 실제 위반은 채점 단계
    (`accident_qa.response_issues`)가 잡아 보고한다.
    """
    t = tx(lang)
    desc = t["sch_predictions"]
    if n_buckets:
        desc += " " + t["sch_n_buckets"].format(n=n_buckets)
    return {
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "description": desc,
                "items": {
                    "type": "object",
                    "properties": {
                        "k": {"type": "integer", "description": t["sch_k"]},
                        "interval_s": {
                            "type": "string",
                            "description": t["sch_interval"],
                        },
                        "accident_expected": {
                            "type": "boolean",
                            "description": t["sch_expected"],
                        },
                        "involved_actor_ids": {
                            "type": "array",
                            "description": t["sch_involved"],
                            "items": {"type": "string"},
                        },
                        "reason": {
                            "type": "string",
                            "description": t["sch_reason"],
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
                    },
                    "required": [
                        "k",
                        "interval_s",
                        "accident_expected",
                        "involved_actor_ids",
                        "reason",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            },
            "overall_assessment": {
                "type": "string",
                "description": t["sch_overall"],
            },
            "data_limitations": {
                "type": "string",
                "description": t["sch_limits"],
            },
        },
        "required": ["predictions", "overall_assessment", "data_limitations"],
        "additionalProperties": False,
    }
