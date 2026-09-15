"""사고 예측 QA 의 **무모델 기준선**.

LLM 이 이겨야 하는 바닥을 잰다. 지금까지 예측기(rule/rank/waypoints)별 LLM 정확도는
서로 비교됐지만 **모델 없는 정책과 비교된 적이 없다**. 옆 프로젝트(`~/vlm`)에서는
"쌍별 등속 외삽 최소간격에 임계값 하나"가 여섯 개 LLM 조건 전부와 동률(0.728)이었다.
같은 검사를 여기서도 해야 한다.

네 가지 정책
    never    모든 구간 '사고 없음'
    always   모든 구간 '사고' (관련 차량은 그 시점 최근접 쌍)
    random   구간마다 동전 던지기 (시드 고정)
    gap      **등속 외삽 최소간격에 임계값 하나** — 이것이 진짜 기준선이다

`gap` 은 모든 액터를 등속으로 외삽하고 각 시점에서 **회전 사각형 사이의 실제
간격**을 잰다. 0 이면 두 차체가 닿는다. 구간마다 그 최솟값을 구해 임계값 하나로
자른다 — 이것이 예측기 전부다.

중심거리에 '절반 길이 합 + 여유' 를 쓰지 않는 이유: 승용차 둘이면 그 기준이 약
6.6m 라, 차선에 줄서 있는 차량이 전부 접촉으로 잡힌다. 그 기준은 정답이 **이미
알려진 충돌 쌍**의 최근접 시점을 찾을 때 쓰는 것이지 일반 판정용이 아니다.

속도는 `speed_mps`·`heading_deg` 가 아니라 **`track_history` 의 위치 차분**으로
구한다 — DeepAccident 라벨의 속도 컬럼은 신뢰할 수 없고, 그 규약을 여기서도 지킨다.

임계값은 **train 에서 정하고 val 에 그대로 적용한다.**

    # train 에서 임계값 적합 → val 채점
    .venv/bin/python examples/baselines_accident_qa.py \
        --root /home/sryu/inclab-nas/DeepAccident \
        --carla-maps ./carla_map --fit-split train --eval-split val

    # 인지 모드 ablation (이상적 3D vs 단안 역투영)
    .venv/bin/python examples/baselines_accident_qa.py ... --mode camera
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepaccident_replicate.da_motion import STATIONARY_MPS
from deepaccident_replicate.da_baseline import VARIANT_NAMES as DA_VARIANT_NAMES
from deepaccident_replicate.da_baseline import DeepAccidentRule
from deepaccident_replicate.da_baseline import build_response as da_response
from deepaccident_replicate.da_baseline import saturation as da_saturation
from deepaccident_replicate.da_baseline import variants as da_variants
from traffic_llm.accident_qa import (  # noqa: E402
    WindowConfig,
    MODES,
    SUPPORTED_MODES,
    aggregate_modes,
    aggregate_scores,
    build_windows,
    score_modes,
    score_response,
    window_ground_truth,
)
from traffic_llm.carla_map import find_xodr  # noqa: E402
from traffic_llm.config import PipelineConfig  # noqa: E402
from traffic_llm.da_runner import DeepAccidentRunner  # noqa: E402
from traffic_llm.deepaccident import CLASS_SIZES, estimate_collision  # noqa: E402
from traffic_llm.schemas import ActorState, SceneSnapshot  # noqa: E402

DEFAULT_LENGTH_M = 4.6
DEFAULT_WIDTH_M = 1.9
SAMPLE_DT_S = 0.1


# ------------------------------------------------------------ 운동 상태

def _velocity(a: ActorState) -> Tuple[float, float]:
    """등속 외삽에 쓸 속도 [m/s]. 위치 차분이 1순위."""
    h = a.track_history or []
    if len(h) >= 2:
        (t0, e0, n0), (t1, e1, n1) = h[-2], h[-1]
        dt = t1 - t0
        if dt > 1e-6:
            return ((e1 - e0) / dt, (n1 - n0) / dt)
    # 폴백 — 방위각은 진북 기준 시계방향이므로 (e, n) = (sin, cos)
    if a.speed_mps and a.heading_deg is not None:
        r = math.radians(a.heading_deg)
        return (a.speed_mps * math.sin(r), a.speed_mps * math.cos(r))
    return (0.0, 0.0)


def _extent(a: ActorState) -> Tuple[float, float]:
    """(길이, 폭) [m]."""
    sz = CLASS_SIZES.get(a.cls)
    return (sz.length_m, sz.width_m) if sz else (DEFAULT_LENGTH_M, DEFAULT_WIDTH_M)


def _corners(cx, cy, ux, uy, half_l, half_w):
    """진행방향 단위벡터 (ux,uy) 로 회전한 사각형의 네 꼭짓점."""
    px, py = -uy, ux
    return [
        (cx + ux * half_l + px * half_w, cy + uy * half_l + py * half_w),
        (cx + ux * half_l - px * half_w, cy + uy * half_l - py * half_w),
        (cx - ux * half_l - px * half_w, cy - uy * half_l - py * half_w),
        (cx - ux * half_l + px * half_w, cy - uy * half_l + py * half_w),
    ]


def _seg_point_dist(px, py, ax, ay, bx, by) -> float:
    ex, ey = bx - ax, by - ay
    L2 = ex * ex + ey * ey
    t = 0.0 if L2 <= 1e-12 else max(0.0, min(1.0, ((px - ax) * ex + (py - ay) * ey) / L2))
    return math.hypot(px - (ax + t * ex), py - (ay + t * ey))


def _separated(A, B) -> bool:
    """분리축 정리 — 두 볼록다각형이 떨어져 있는가."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            nx, ny = -(by - ay), bx - ax
            pa = [nx * x + ny * y for x, y in A]
            pb = [nx * x + ny * y for x, y in B]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return True
    return False


def _poly_gap(A, B) -> float:
    """두 사각형 사이 최단거리 [m]. 겹치면 0."""
    if not _separated(A, B):
        return 0.0
    best = float("inf")
    for P, Q in ((A, B), (B, A)):
        n = len(Q)
        for px, py in P:
            for i in range(n):
                ax, ay = Q[i]
                bx, by = Q[(i + 1) % n]
                d = _seg_point_dist(px, py, ax, ay, bx, by)
                if d < best:
                    best = d
    return best


# ------------------------------------------------------------ gap table

def _motion_fn(a: ActorState, mode: str):
    """t[s] → (e, n, 진행방향 단위벡터). 등속 또는 예측 궤적."""
    p0 = a.world_xy
    vx, vy = _velocity(a)

    def unit_of(dx, dy, fallback):
        sp = math.hypot(dx, dy)
        return (dx / sp, dy / sp) if sp > 1e-6 else fallback

    sp0 = math.hypot(vx, vy)
    if sp0 > 0.3:
        base_u = (vx / sp0, vy / sp0)
    elif a.heading_deg is not None:
        r = math.radians(a.heading_deg)
        base_u = (math.sin(r), math.cos(r))
    else:
        base_u = (1.0, 0.0)

    if mode == "predicted" and a.predictions:
        # 확률 최고 경로. 웨이포인트는 1초 간격 (e, n) 이고 t=1 부터다.
        best = max(a.predictions, key=lambda p: p.probability)
        wps = list(best.waypoints or [])
        # 점이 2개 미만이면 궤적 정보가 없다 (도로망 끝에서 잘렸거나 정지 판정).
        # 실측으로 액터-프레임의 약 24% 가 여기 해당한다 — 그 경우 등속으로 돌아간다.
        if len(wps) >= 2:
            # 웨이포인트는 1초 간격이다. 구현에 따라 **첫 점이 현재 위치**일 수도
            # 있고(실측: WaypointNet 은 그렇다) t=1 부터일 수도 있다. 앞에 현재
            # 위치를 무조건 덧붙이면 전체가 1초씩 밀린다.
            pts = wps if math.dist(wps[0], p0) < 0.5 else [p0] + wps
            #                                             index i ↔ t = i 초
            def f(t: float):
                if t <= 0:
                    return (p0[0], p0[1], base_u)
                if t >= len(pts) - 1:             # 궤적 끝 — 마지막 방향으로 등속 연장
                    q, r_ = pts[-2], pts[-1]
                    u = unit_of(r_[0] - q[0], r_[1] - q[1], base_u)
                    dt = t - (len(pts) - 1)
                    step = math.dist(q, r_)
                    return (r_[0] + u[0] * step * dt, r_[1] + u[1] * step * dt, u)
                i = int(t)
                f_ = t - i
                q, r_ = pts[i], pts[i + 1]
                u = unit_of(r_[0] - q[0], r_[1] - q[1], base_u)
                return (q[0] + (r_[0] - q[0]) * f_, q[1] + (r_[1] - q[1]) * f_, u)
            return f

    def g(t: float):
        return (p0[0] + vx * t, p0[1] + vy * t, base_u)
    return g


def bucket_min_gaps(
    snap: SceneSnapshot, n_buckets: int, mode: str = "cv",
    exclude_touching_now: bool = False, exclude_static_pairs: bool = False,
) -> List[Tuple[float, Optional[Tuple[str, str]]]]:
    """구간 k=1..N 각각의 (최소 간격 [m], 그 쌍).

    각 시점에서 **회전 사각형 사이의 실제 간격**을 잰다. 0 이면 두 차체가 닿는다.

    중심거리가 아니라 차체 간격을 쓰는 이유: 중심거리에 '절반 길이 합 + 여유' 를
    적용하면 차선에 줄서 있는 차량이 전부 접촉으로 잡힌다. 그 기준은 정답이
    **이미 알려진 충돌 쌍**의 최근접 시점을 찾을 때 쓰는 것이지 일반 판정용이 아니다.

    mode
        'cv'         등속 외삽 (기본). `~/vlm` 의 gap table 과 같은 방법이다.
        'predicted'  파이프라인의 예측 경로(기본 WaypointNet)를 따라간다.
                     vlm 의 천장 분석이 "벽은 인지가 아니라 예측 모델"이라 지목한
                     자리를 바꿔 끼우는 것이다. 예측이 없는 액터는 등속으로 폴백.

    두 필터가 후보 쌍을 줄인다 (기본은 둘 다 꺼짐 — 예전 동작 유지).

    `exclude_static_pairs`
        **둘 다 정지한 쌍**을 뺀다. 주차된 차 두 대는 사고를 낼 수 없다.
        DeepAccident 라벨은 주차 차량 박스가 서로 겹쳐 있어 (중심거리 3~4 m 인데
        차 길이 4.5~5.4 m) 최소 간격이 영구히 0 이 된다. val 라벨 400프레임 실측:
        겹치는 쌍 537개 중 **526개가 정지-정지**, 정지-이동 0개, 이동-이동 11개
        (실제 충돌). 이 필터로 겹치는 프레임이 **31 % → 2.8 %** 로 떨어진다.
        한쪽만 정지한 쌍은 남긴다 — 신호 대기 중 추돌당하는 것은 사고다.
        치수를 객체별 라벨 값으로 바꿔도 겹침은 거의 그대로다 (0.310 → 0.305).
        즉 문제는 치수 출처가 아니라 정지 차량 쌍이다.

    `exclude_touching_now`
        관측 시점에 이미 붙어 있는 쌍을 뺀다. 사고 예측은 벌어져 있다가 닫히는
        쌍의 문제다.
    """
    acts = [a for a in snap.actors if a.world_xy is not None]
    if len(acts) < 2:
        return [(float("inf"), None)] * n_buckets

    n = len(acts)
    ids = [a.actor_id for a in acts]
    half = [(l / 2.0, w / 2.0) for l, w in (_extent(a) for a in acts)]
    radius = [math.hypot(hl, hw) for hl, hw in half]
    motion = [_motion_fn(a, mode) for a in acts]

    banned: Set[Tuple[int, int]] = set()
    if exclude_touching_now or exclude_static_pairs:
        now = [m(0.0) for m in motion]
        moving = [abs(a.speed_mps or 0.0) > STATIONARY_MPS for a in acts]
        for i in range(n):
            for j in range(i + 1, n):
                if exclude_static_pairs and not (moving[i] or moving[j]):
                    banned.add((i, j))
                    continue
                if not exclude_touching_now:
                    continue
                A = _corners(now[i][0], now[i][1], now[i][2][0], now[i][2][1], *half[i])
                B = _corners(now[j][0], now[j][1], now[j][2][0], now[j][2][1], *half[j])
                if _poly_gap(A, B) <= 0.0:
                    banned.add((i, j))

    steps = int(round(1.0 / SAMPLE_DT_S))
    out: List[Tuple[float, Optional[Tuple[str, str]]]] = []
    for k in range(1, n_buckets + 1):
        best, best_pair = float("inf"), None
        for st in range(1, steps + 1):
            t = (k - 1) + st * SAMPLE_DT_S
            pos = [m(t) for m in motion]
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) in banned:
                        continue
                    lo = math.hypot(pos[i][0] - pos[j][0],
                                    pos[i][1] - pos[j][1]) - radius[i] - radius[j]
                    if lo >= best:
                        continue  # 외접원 하한이 이미 최소보다 크다 — 계산 생략
                    A = _corners(pos[i][0], pos[i][1], pos[i][2][0], pos[i][2][1], *half[i])
                    B = _corners(pos[j][0], pos[j][1], pos[j][2][0], pos[j][2][1], *half[j])
                    g = _poly_gap(A, B)
                    if g < best:
                        best, best_pair = g, (ids[i], ids[j])
        out.append((best, best_pair))
    return out


# ------------------------------------------------------------ 정책

RULE_POLICIES = ("never", "always", "random", "gap")
DA_POLICIES = DA_VARIANT_NAMES              # da, da_any, da_h5, da_fit
POLICIES = RULE_POLICIES + DA_POLICIES


def _resp(preds: List[dict]) -> dict:
    return {"predictions": preds, "data_limitations": ""}


def policy_responses(
    t_e: float, n_buckets: int, gaps, threshold_m: float, rng: random.Random,
    only: Optional[Sequence[str]] = None,
    da_rules: Optional[Dict[str, DeepAccidentRule]] = None,
) -> Dict[str, dict]:
    """네 정책의 응답을 PREDICTION_SCHEMA 형태로.

    `win` 이 아니라 마지막 관측 시각 `t_e` 만 받는다 — 저장된 레코드에서 재실행할 때
    스냅샷 없이도 그대로 재생되어야 하기 때문이다.

    `only` 로 일부 정책만 만들 수 있다. 임계값 스윕은 `gap` 하나만 쓰는데,
    DeepAccident 규칙은 임계값과 무관하므로 격자마다 다시 만들 이유가 없다.
    """
    want = set(only) if only else set(POLICIES)
    closest = gaps[0][1] if gaps else None

    def interval(k: int) -> str:
        return f"({t_e + k - 1:.1f}, {t_e + k:.1f}]"

    out: Dict[str, List[dict]] = {p: [] for p in RULE_POLICIES}
    for k in range(1, n_buckets + 1):
        iv = interval(k)
        gap, pair = gaps[k - 1]
        hit = gap <= threshold_m

        if "never" in want:
            out["never"].append(
                {"k": k, "interval_s": iv, "accident_expected": False,
                 "involved_actor_ids": [], "reason": "무조건 사고 없음"})
        if "always" in want:
            out["always"].append(
                {"k": k, "interval_s": iv, "accident_expected": True,
                 "involved_actor_ids": list(closest) if closest else [],
                 "reason": "무조건 사고"})
        # 시드 재현성을 위해 `random` 을 만들지 않을 때도 난수는 소비한다
        r = rng.random() < 0.5
        if "random" in want:
            out["random"].append(
                {"k": k, "interval_s": iv, "accident_expected": r,
                 "involved_actor_ids": list(closest) if (r and closest) else [],
                 "reason": "동전 던지기"})
        if "gap" in want:
            out["gap"].append(
                {"k": k, "interval_s": iv, "accident_expected": bool(hit),
                 "involved_actor_ids": list(pair) if (hit and pair) else [],
                 "reason": f"등속 외삽 최소간격 {gap:.2f}m (임계 {threshold_m:.2f}m)"})

    resps = {name: _resp(preds) for name, preds in out.items() if preds}
    # DeepAccident 후처리 규칙. **같은 gaps**(같은 예측)를 받으므로 예측이 통제되고
    # 판정 규칙만 달라진다 — 그것이 이 비교의 요점이다.
    for name, rule in (da_rules if da_rules is not None else da_variants()).items():
        if name in want:
            resps[name] = da_response(t_e, n_buckets, gaps, rule)
    return resps


# ------------------------------------------------------------ 실행

def windows_of(runner: DeepAccidentRunner, sc, cfg: PipelineConfig,
               wcfg: WindowConfig, rate_hz: float, maps: Optional[str],
               extrap: str = "cv", exclude_touching_now: bool = False,
               exclude_static_pairs: bool = False):
    """한 시나리오 → [(window, ground_truth, gaps)]"""
    xodr = find_xodr(sc.town, [maps]) if maps else None
    res = runner.build(sc.scenario, sc.scenario_type, opendrive_path=xodr)
    snaps = list(res.snapshots(rate_hz=rate_hz))
    if len(snaps) < 2:
        return []
    collision = estimate_collision(sc, cfg.deepaccident)
    ct = collision.time_s if (collision and collision.occurred) else None
    wins, _ = build_windows(snaps, wcfg, collision_time_s=ct)
    agent_ids = {ag: sc.meta.agent_id_of(ag) for ag in sc.agents}

    rows = []
    for w in wins:
        gt = window_ground_truth(
            w, wcfg, collision=collision, agent_carla_ids=agent_ids,
            scenario_id=sc.scenario_id, scenario_split=sc.scenario_type,
            dataset_split=sc.split, town=sc.town, data_end_s=snaps[-1].t,
        )
        n_b = len(gt.get("expected") or [])
        if n_b == 0:
            continue
        rows.append((w.snapshots[-1].t, gt,
                     bucket_min_gaps(w.snapshots[-1], n_b, extrap,
                                     exclude_touching_now,
                                     exclude_static_pairs)))
    return rows


def evaluate(rows, threshold_m: float, seed: int = 0,
             only: Optional[Sequence[str]] = None,
             da_rules: Optional[Dict[str, DeepAccidentRule]] = None) -> Dict[str, dict]:
    names = list(only) if only else list(POLICIES)
    per: Dict[str, List[dict]] = {p: [] for p in names}
    rng = random.Random(seed)
    for t_e, gt, gaps in rows:
        n_b = len(gt["expected"])
        resps = policy_responses(t_e, n_b, gaps, threshold_m, rng, only=names,
                                 da_rules=da_rules)
        for name, r in resps.items():
            per[name].append(score_response(r, gt))
    return {name: aggregate_scores(s) for name, s in per.items()}


def evaluate_modes(
    rows, threshold_m: float, seed: int = 0,
    modes: Sequence[str] = MODES, late_decay: float = 0.2,
    early_decay: float = 0.0,
    da_rules: Optional[Dict[str, DeepAccidentRule]] = None,
) -> Dict[str, dict]:
    """정책별 **다중 기준** 채점.

    기준선도 LLM 과 같은 축에서 읽혀야 한다 — 특히 `binary_scenario` 는
    DeepAccident 비교용 주 지표이므로, "임계값 하나짜리 정책이 이미 몇 점인가"
    를 그 축에서 알아야 LLM 숫자를 해석할 수 있다.
    """
    names = list(POLICIES) if da_rules is None else (
        list(RULE_POLICIES) + list(da_rules))
    per: Dict[str, List[dict]] = {p: [] for p in names}
    rng = random.Random(seed)
    for t_e, gt, gaps in rows:
        n_b = len(gt["expected"])
        resps = policy_responses(t_e, n_b, gaps, threshold_m, rng, only=names,
                                 da_rules=da_rules)
        for name, r in resps.items():
            per[name].append(
                score_modes(r, gt, late_decay=late_decay,
                            early_decay=early_decay, modes=modes)
            )
    return {name: aggregate_modes(s, modes) for name, s in per.items()}


def dump_records(rows, path: str) -> None:
    """창별 (마지막 시각, 정답, 구간별 간격) 을 JSONL 로.

    수집(시나리오 읽기·융합·지도매칭)이 비싸므로 **한 번 모으면 다시 안 모은다**.
    임계값 재스윕, 구간별 분해, 지평 절단이 전부 이 파일에서 재실행된다.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t_e, gt, gaps in rows:
            f.write(json.dumps({
                "t_end_s": t_e, "gt": gt,
                "gaps": [[None if g == float("inf") else g,
                          list(pair) if pair else None] for g, pair in gaps],
            }, ensure_ascii=False) + "\n")


def load_records(path: str):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        gaps = [(float("inf") if g is None else g,
                 tuple(pair) if pair else None) for g, pair in d["gaps"]]
        rows.append((d["t_end_s"], d["gt"], gaps))
    return rows


def _prf(counts: dict) -> Tuple[float, float, float, float]:
    """정밀도·재현율·F1·균형정확도."""
    tp, fp, tn, fn = (counts[k] for k in ("TP", "FP", "TN", "FN"))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return p, r, f1, (r + spec) / 2.0


def fit_threshold(rows, grid: Sequence[float],
                  metric: str = "balanced") -> Tuple[float, float]:
    """train 에서 지표가 최대가 되는 임계값.

    **F1 로 고르면 안 된다.** 양성 비율이 12% 대라 F1 최대화가 재현율 쪽으로 끌려가
    오탐이 폭증하는 임계값을 뽑는다 (실측: F1 최고인 0.00m 에서 TP 154 / FP 1644).
    기본은 균형정확도(민감도·특이도 평균)다.
    """
    best_t, best_v = grid[0], -1.0
    for t in grid:
        agg = evaluate(rows, t, only=("gap",))["gap"]
        c = agg["counts"]
        p, r, f1, bal = _prf(c)
        v = {"f1": f1, "balanced": bal, "accuracy": agg["accuracy"]}[metric]
        print(f"    임계 {t:6.2f}m → {metric} {v:.3f}  "
              f"(정확도 {agg['accuracy']:.3f} F1 {f1:.3f} "
              f"TP {c['TP']} FP {c['FP']} FN {c['FN']})")
        if v > best_v:
            best_t, best_v = t, v
    return best_t, best_v


def fit_da_threshold(rows, grid: Sequence[float],
                     metric: str = "balanced") -> Tuple[float, float]:
    """DeepAccident 규칙의 거리 임계값을 train 에서 적합한다.

    **바이너리 축에서 고른다.** 그 축이 DeepAccident 와 판정 단위가 같은 유일한
    축이고(창당 1건), 이 규칙이 창당 구간 하나만 표시하므로 구간별 정확도는
    표시하지 않은 구간의 TN 으로 부풀려진다.

    저자들의 상수 2.5m 는 그들의 BEV 픽셀 공간(5px)에 맞춘 값이다. 우리 장면에서
    모든 쌍을 5초간 훑으면 그 값이 거의 항상 걸리므로, 상수를 그대로 쓴 줄만
    보고하면 규칙을 불리하게 만든 비교가 된다.
    """
    from traffic_llm.accident_qa import score_binary

    best_t, best_v = grid[0], -1.0
    for t in grid:
        rule = DeepAccidentRule(dist_threshold_m=t, horizon_s=5.0, frame_rule="any")
        tp = fp = tn = fn = 0
        for t_e, gt, gaps in rows:
            n_b = len(gt["expected"])
            b = score_binary(da_response(t_e, n_b, gaps, rule), gt)
            if not b["scorable"]:
                continue
            c = b["counts"]
            tp += c["TP"]; fp += c["FP"]; tn += c["TN"]; fn += c["FN"]
        p, r, f1, bal = _prf({"TP": tp, "FP": fp, "TN": tn, "FN": fn})
        total = tp + fp + tn + fn
        acc = (tp + tn) / total if total else 0.0
        v = {"f1": f1, "balanced": bal, "accuracy": acc}[metric]
        print(f"    DA 임계 {t:6.2f}m → {metric} {v:.3f}  "
              f"(정확도 {acc:.3f} F1 {f1:.3f} TP {tp} FP {fp} FN {fn})")
        if v > best_v:
            best_t, best_v = t, v
    return best_t, best_v


def collect(root: str, split: str, cfg, wcfg, limit: Optional[int],
            maps: Optional[str], rate_hz: float = 2.0, seed: int = 0,
            extrap: str = "cv", exclude_touching_now: bool = False,
            exclude_static_pairs: bool = False):
    runner = DeepAccidentRunner(root, cfg)
    scs = [s for s in runner.list_scenarios() if s.split == split]
    if limit and limit < len(scs):
        # 앞에서 자르면 시나리오 유형이 한쪽으로 쏠린다 (목록이 유형순이다).
        # 시드 고정 셔플로 사고/정상과 유형을 고루 섞는다.
        rnd = random.Random(seed)
        scs = sorted(scs, key=lambda x: x.scenario_id)
        rnd.shuffle(scs)
        scs = scs[:limit]
    n_acc = sum(1 for x in scs if x.meta.collision_occurred)
    print(f"  시나리오 {len(scs)}개 (충돌 {n_acc} / 무충돌 {len(scs)-n_acc})")
    rows = []
    for i, sc in enumerate(scs, 1):
        try:
            rows.extend(windows_of(runner, sc, cfg, wcfg, rate_hz, maps, extrap,
                                   exclude_touching_now, exclude_static_pairs))
        except Exception as e:  # 개별 시나리오 실패가 전체를 막지 않게
            print(f"  [{i}/{len(scs)}] {sc.scenario} 실패: {type(e).__name__}: {e}")
            continue
        if i % 10 == 0 or i == len(scs):
            print(f"  [{i}/{len(scs)}] 누적 창 {len(rows)}개")
    return rows


def report(title: str, aggs: Dict[str, dict]) -> None:
    print(f"\n=== {title} ===")
    print(f"{'정책':10s} {'정확도':>7s} {'균형정확':>8s} {'정밀도':>7s} {'재현율':>7s} "
          f"{'F1':>6s} {'TP':>5s} {'FP':>5s} {'TN':>5s} {'FN':>5s}")
    for name in POLICIES:
        a = aggs.get(name)
        if a is None:
            continue
        c = a["counts"]
        p, r, f1, bal = _prf(c)
        print(f"{name:10s} {a['accuracy']:7.3f} {bal:8.3f} {p:7.3f} {r:7.3f} "
              f"{f1:6.3f} {c['TP']:5d} {c['FP']:5d} {c['TN']:5d} {c['FN']:5d}")


def report_modes(title: str, aggs: Dict[str, dict], modes: Sequence[str]) -> None:
    """모드별 한 줄 요약. 같은 응답이므로 열 사이 차이는 기준의 차이다."""
    print(f"\n=== {title} (다중 기준) ===")
    head = f"{'정책':10s}"
    if "binary_scenario" in modes:
        head += f" {'BS정확':>6s} {'BS정밀':>6s} {'BS재현':>6s}"
    if "binary_window" in modes:
        head += f" {'BW정확':>6s} {'BW재현':>6s}"
    if "binary" in modes:  # legacy alias
        head += f" {'B정확':>6s} {'B정밀':>6s} {'B재현':>6s} {'B칸수':>6s}"
    if "strict" in modes:
        head += f" {'S정확':>6s}"
    if "early" in modes:
        head += f" {'E정확':>6s} {'E검출':>6s}"
    if "weighted" in modes:
        head += f" {'W점수':>6s} {'W오경':>6s}"
    print(head)
    for name in POLICIES:
        a = aggs.get(name)
        if a is None:
            continue
        line = f"{name:10s}"
        if "binary_scenario" in modes:
            b = a["binary_scenario"]
            line += (f" {_f(b['accuracy'])} {_f(b['precision'])} "
                     f"{_f(b['recall'])}")
        if "binary_window" in modes:
            b = a["binary_window"]
            line += f" {_f(b['accuracy'])} {_f(b['recall'])}"
        if "binary" in modes:
            b = a["binary"]
            line += (f" {_f(b['accuracy'])} {_f(b['precision'])} "
                     f"{_f(b['recall'])} {_f(b['mean_positive_buckets'])}")
        if "strict" in modes:
            line += f" {_f(a['strict']['accuracy'])}"
        if "early" in modes:
            line += (f" {_f(a['early']['accuracy'])} "
                     f"{_f(a['early']['events']['detection_rate'])}")
        if "weighted" in modes:
            w = a["weighted"]
            line += f" {_f(w['mean_score'])} {_f(w['false_alarm_rate'])}"
        print(line)


def _f(v) -> str:
    return "     -" if v is None else f"{v:6.3f}"


def report_da_saturation(rows, da_rules) -> None:
    """DeepAccident 규칙이 이 데이터에서 변별력이 있는가.

    **표보다 먼저 읽어야 하는 줄이다.** 발화율이 1 에 가까우면 그 정책의 정확도는
    규칙의 변별력이 아니라 기저율이다.
    """
    gaps_list = [g for _, _, g in rows]
    print(f"\n=== DeepAccident 규칙 포화 진단 ===")
    print("  규칙의 원출력 기준이다 — 위 표는 데이터 끝 이후 구간을 채점에서 빼므로 "
          "발화율이 표의 'B칸수' 보다 조금 높다.")
    print(f"{'정책':10s} {'규칙':>22s} {'발화율':>7s} {'차체겹침':>9s}  판정")
    any_bad = False
    for name, rule in da_rules.items():
        d = da_saturation(gaps_list, rule)
        bad = d["degenerate"]
        any_bad = any_bad or bad
        verdict = "사실상 상수 — 정확도는 기저율" if bad else "변별력 있음"
        print(f"{name:10s} {d['rule']:>22s} {d['fire_rate']:7.3f} "
              f"{d['touching_rate']:9.3f}  {verdict}")
    if any_bad:
        print("  ! 외삽한 차체가 거의 모든 창에서 겹친다. 거리 임계 하나로는 "
              "이 데이터에서 판정이 되지 않는다 —")
        print("    위 표의 해당 줄을 'DeepAccident 성능' 으로 읽으면 안 된다.")


def per_bucket_table(rows) -> None:
    """구간(k)별 신호 세기.

    **집계만 보면 안 된다.** 실측(train 120 시나리오, 구간 2,955개):
    k=1 은 AUC 0.806 으로 풀리지만 k=4·k=5 는 0.5 미만 — 우연보다 못하다.
    5초 지평의 뒤쪽은 운동학이 아니라 아직 일어나지 않은 기동이 결정하기 때문이고,
    같은 곡선이 `~/vlm` 의 horizon_sweep 에서 독립적으로 측정됐다
    (1.0s 0.801 / 2.0s 0.603 / 3.0s 0.565).

    그래서 전체 평균은 **답할 수 있는 구간과 없는 구간을 섞은 값**이다.
    """
    from collections import defaultdict
    by_k = defaultdict(list)
    for _t_e, gt, gaps in rows:
        for i, e in enumerate(gt.get("expected") or []):
            if not e.get("scorable", True) or i >= len(gaps):
                continue
            g = gaps[i][0]
            if g is None or g == float("inf"):
                continue
            by_k[e.get("k", i + 1)].append((g, bool(e.get("accident_expected"))))

    print(f"\n{'구간':>4s} {'n':>6s} {'양성률':>7s} {'AUC':>7s}  판정")
    for k in sorted(by_k):
        vals = by_k[k]
        pos = sum(1 for _, l in vals if l)
        neg = len(vals) - pos
        if pos == 0 or neg == 0:
            continue
        # 위험도 = -간격. 순위합으로 AUC.
        order = sorted(range(len(vals)), key=lambda i: -vals[i][0])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]][0] == vals[order[i]][0]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                ranks[order[t]] = avg
            i = j + 1
        srank = sum(r for r, (_, l) in zip(ranks, vals) if l)
        a = (srank - pos * (pos + 1) / 2.0) / (pos * neg)
        verdict = ("신호 있음" if a >= 0.65 else
                   "약함" if a >= 0.55 else "우연 수준")
        print(f"{k:4d} {len(vals):6d} {pos/len(vals):7.3f} {a:7.3f}  {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--carla-maps", default=None)
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--eval-split", default="val")
    ap.add_argument("--mode", default="sensor3d", choices=("sensor3d", "camera"))
    ap.add_argument("--limit-fit", type=int, default=None)
    ap.add_argument("--limit-eval", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="주면 적합을 건너뛰고 이 값을 쓴다 [m]")
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--stride", type=float, default=1.0)
    ap.add_argument("--horizon", type=float, default=5.0)
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--out", default=None, help="결과 JSON 저장 경로")
    ap.add_argument("--modes", default=None,
                    help="다중 기준 채점을 함께 낸다. 쉼표 구분 "
                         f"({','.join(MODES)}) 또는 'all'")
    ap.add_argument("--late-decay", type=float, default=0.2,
                    help="weighted 모드에서 늦은 경보 한 칸당 감점")
    ap.add_argument("--early-decay", type=float, default=0.0,
                    help="weighted 모드에서 이른 경보 한 칸당 감점")
    ap.add_argument("--records-out", default=None,
                    help="창별 원본(정답+구간별 간격) JSONL. 재분석용")
    ap.add_argument("--from-records", default=None,
                    help="수집 대신 이 JSONL 을 읽는다 (재실행 없이 재분석)")
    ap.add_argument("--fit-records", default=None, help="적합용 레코드 JSONL")
    ap.add_argument("--fit-metric", default="balanced",
                    choices=("balanced", "accuracy", "f1"))
    ap.add_argument("--da-threshold", type=float, default=None,
                    help="DeepAccident 규칙의 거리 임계값을 직접 지정 [m]. "
                         "생략하면 train 에서 적합한다 (da_fit 줄)")
    ap.add_argument("--exclude-touching", action="store_true",
                    help="관측 시점에 이미 붙어 있는 쌍을 후보에서 뺀다. "
                         "DeepAccident 라벨의 주차 차량 박스가 서로 겹쳐 있어 "
                         "이것 없이는 최소 간격이 영구히 0 이다. **레코드를 다시 "
                         "수집해야 적용된다** (--from-records 는 저장된 값을 쓴다)")
    ap.add_argument("--exclude-static", action="store_true",
                    help="둘 다 정지한 쌍을 후보에서 뺀다. 주차 차량 두 대는 사고를 "
                         "낼 수 없다 — 실측으로 겹치는 쌍의 98%%가 이것이다. "
                         "**레코드를 다시 수집해야 적용된다**")
    ap.add_argument("--no-da-fit", action="store_true",
                    help="da_fit 줄을 만들지 않는다. 저자들의 상수 2.5m 줄만 "
                         "보고하면 규칙을 불리하게 만든 비교가 되므로 기본은 적합이다")
    ap.add_argument("--extrapolation", default="cv", choices=("cv", "predicted"),
                    help="gap table 을 등속으로 낼지 예측 경로로 낼지")
    ap.add_argument("--predictor", default="out/predict_model/waypointnet_best.pt",
                    help="--extrapolation predicted 일 때 쓸 학습 예측기")
    ap.add_argument("--predictor-mode", default="waypoints")
    ap.add_argument("--no-predictor", action="store_true",
                    help="규칙 기반(지도 제약) 예측을 쓴다")
    args = ap.parse_args()

    cfg = PipelineConfig()
    cfg.deepaccident.observation_mode = args.mode
    wcfg = WindowConfig(window_s=args.window, stride_s=args.stride,
                        horizon_s=args.horizon)

    da_thr: Optional[float] = args.da_threshold

    if args.extrapolation == "predicted" and not args.no_predictor:
        from traffic_llm.predict_model import TorchPredictor
        cfg.predictor = TorchPredictor(args.predictor, mode=args.predictor_mode)
        pnote = f"{os.path.basename(args.predictor)} (mode={args.predictor_mode})"
    else:
        pnote = "규칙 기반" if args.extrapolation == "predicted" else "미사용(등속)"
    print(f"인지 모드: {args.mode} · 외삽: {args.extrapolation} · 예측기: {pnote}",
          flush=True)

    if args.threshold is None:
        print(f"\n[1] 임계값 적합 — {args.fit_split} (지표 {args.fit_metric})",
              flush=True)
        if args.fit_records and os.path.exists(args.fit_records):
            fit_rows = load_records(args.fit_records)
            print(f"  레코드에서 적재: {args.fit_records}", flush=True)
        else:
            fit_rows = collect(args.root, args.fit_split, cfg, wcfg,
                               args.limit_fit, args.carla_maps, args.rate,
                               extrap=args.extrapolation,
                               exclude_touching_now=args.exclude_touching,
                               exclude_static_pairs=args.exclude_static)
            if args.fit_records:
                dump_records(fit_rows, args.fit_records)
                print(f"  레코드 저장: {args.fit_records}", flush=True)
        print(f"  창 {len(fit_rows)}개", flush=True)
        grid = [round(x * 0.1, 2) for x in range(0, 31)]  # 0.0 … 3.0 m
        thr, v = fit_threshold(fit_rows, grid, args.fit_metric)
        print(f"  → 선택 임계값 {thr:.2f}m ({args.fit_metric} {v:.3f})", flush=True)
        if args.da_threshold is None and not args.no_da_fit:
            print(f"\n[1b] DeepAccident 규칙 임계값 적합 — {args.fit_split} "
                  f"(바이너리 축, 지표 {args.fit_metric})", flush=True)
            da_grid = [round(x * 0.25, 2) for x in range(0, 21)]  # 0.0 … 5.0 m
            da_thr, da_v = fit_da_threshold(fit_rows, da_grid, args.fit_metric)
            print(f"  → 선택 임계값 {da_thr:.2f}m ({args.fit_metric} {da_v:.3f})",
                  flush=True)
    else:
        thr = args.threshold
        print(f"\n[1] 임계값 지정 {thr:.2f}m (적합 건너뜀)", flush=True)
    if args.da_threshold is not None:
        print(f"[1b] DeepAccident 규칙 임계값 지정 {da_thr:.2f}m (적합 건너뜀)",
              flush=True)

    print(f"\n[2] 채점 — {args.eval_split}", flush=True)
    if args.from_records and os.path.exists(args.from_records):
        ev_rows = load_records(args.from_records)
        print(f"  레코드에서 적재: {args.from_records}", flush=True)
    else:
        ev_rows = collect(args.root, args.eval_split, cfg, wcfg, args.limit_eval,
                          args.carla_maps, args.rate, extrap=args.extrapolation,
                          exclude_touching_now=args.exclude_touching,
                          exclude_static_pairs=args.exclude_static)
        if args.records_out:
            dump_records(ev_rows, args.records_out)
            print(f"  레코드 저장: {args.records_out}", flush=True)
    print(f"  창 {len(ev_rows)}개", flush=True)
    da_rules = da_variants(da_thr)
    aggs = evaluate(ev_rows, thr, da_rules=da_rules)
    title = f"{args.eval_split} · 인지={args.mode} · 임계 {thr:.2f}m"
    if da_thr is not None:
        title += f" · DA 임계 {da_thr:.2f}m"
    report(title, aggs)

    maggs = None
    report_da_saturation(ev_rows, da_rules)

    if args.modes:
        modes = tuple(MODES) if args.modes == "all" else tuple(
            m.strip() for m in args.modes.split(",") if m.strip()
        )
        bad = [m for m in modes if m not in SUPPORTED_MODES]
        if bad:
            ap.error(f"알 수 없는 채점 모드: {bad} (가능: {list(SUPPORTED_MODES)})")
        maggs = evaluate_modes(ev_rows, thr, modes=modes,
                               late_decay=args.late_decay,
                               early_decay=args.early_decay,
                               da_rules=da_rules)
        report_modes(title, maggs, modes)

    per_bucket_table(ev_rows)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"mode": args.mode, "threshold_m": thr,
                   "da_threshold_m": da_thr,
                   "da_rules": {k: vars(v) for k, v in da_rules.items()},
                   "fit_split": args.fit_split, "eval_split": args.eval_split,
                   "n_windows": len(ev_rows), "aggregate": aggs,
                   **({"aggregate_modes": maggs} if maggs else {})},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
