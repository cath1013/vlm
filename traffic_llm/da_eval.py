"""DeepAccident 정답 대비 파이프라인 정확도 평가.

DeepAccident 의 로컬 트랙 id 는 CARLA actor id 이며 모든 관측자에서 동일하다.
따라서 최근접 매칭이 아니라 **신원 기반 매칭**으로 평가할 수 있다 — 원거리
큰 오차를 '유령 객체'로 오분류하지 않으므로 왜곡이 없다.

    from traffic_llm.da_eval import evaluate_scenario
    m = evaluate_scenario(root, "Town03", "type1_subtype1_normal", mode="camera")
    print(m.report())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import PipelineConfig
from .da_runner import BuildResult, DeepAccidentRunner
from .deepaccident import ground_truth_tracks
from .geometry import wrap180


def _pct(v: Sequence[float], q: float) -> float:
    if not v:
        return float("nan")
    s = sorted(v)
    i = min(int(q * len(s)), len(s) - 1)
    return s[i]


def _median(v: Sequence[float]) -> float:
    return _pct(v, 0.5)


@dataclass
class Metrics:
    """평가 지표. 거리 구간별로 나눠 봐야 단안 특성이 드러난다."""

    scenario_id: str = ""
    mode: str = ""
    n_snapshots: int = 0
    n_matched: int = 0
    n_unmatched_actor: int = 0  # 정답에 대응 id 가 없는 액터
    n_missed_gt: int = 0  # 검출은 됐으나 액터가 되지 못한 객체 (파이프라인 손실)
    pos_err: List[float] = field(default_factory=list)
    radial_bias: List[float] = field(default_factory=list)  # +면 과대추정
    lateral_err: List[float] = field(default_factory=list)
    heading_err: List[float] = field(default_factory=list)
    speed_err: List[float] = field(default_factory=list)
    by_range: Dict[str, List[float]] = field(default_factory=dict)
    localization_rate: float = float("nan")
    lane_assigned_rate: float = float("nan")

    def report(self) -> str:
        L = [
            f"{self.scenario_id}  [{self.mode}]",
            f"  스냅샷 {self.n_snapshots}, 신원매칭 {self.n_matched}, "
            f"미매칭 액터 {self.n_unmatched_actor}, "
            f"검출→액터 손실 {self.n_missed_gt}",
        ]
        if self.pos_err:
            L.append(
                f"  위치오차   중앙 {_median(self.pos_err):5.2f}m  "
                f"p90 {_pct(self.pos_err, .9):5.2f}m  최대 {max(self.pos_err):5.2f}m"
            )
        if self.radial_bias:
            L.append(
                f"  반경편향   중앙 {_median(self.radial_bias):+5.2f}m  "
                f"(+ = 과대추정)"
            )
        if self.lateral_err:
            L.append(f"  횡방향오차 중앙 {_median(self.lateral_err):5.2f}m")
        if self.heading_err:
            L.append(f"  방위오차   중앙 {_median(self.heading_err):5.1f}°")
        if self.speed_err:
            L.append(f"  속도오차   중앙 {_median(self.speed_err):5.2f}m/s")
        for k in sorted(self.by_range, key=lambda z: int(z.split("-")[0])):
            v = self.by_range[k]
            L.append(
                f"    {k:>9s}m n={len(v):4d} 중앙 {_median(v):5.2f}m "
                f"p90 {_pct(v, .9):5.2f}m"
            )
        if not math.isnan(self.localization_rate):
            L.append(
                f"  도로매칭률 {self.localization_rate:5.1%}, "
                f"차선배정률 {self.lane_assigned_rate:5.1%}"
            )
        return "\n".join(L)


RANGE_BINS = ((0, 15), (15, 30), (30, 45), (45, 70), (70, 130))


def evaluate_build(
    res: BuildResult,
    cfg: PipelineConfig,
    rate_hz: float = 2.0,
    gt_max_range_m: float = 60.0,
) -> Metrics:
    """구성된 시나리오를 평가한다.

    gt_max_range_m: 이 거리 안의 정답 객체만 '놓쳤다'고 센다. 그보다 먼
    객체는 전면 카메라 화각·가려짐으로 못 보는 것이 정상이다.
    """
    conv = res.converter
    m = Metrics(scenario_id=res.scenario.scenario_id, mode=cfg.deepaccident.observation_mode)
    gt_all = ground_truth_tracks(res.scenario, cfg.deepaccident)

    # 프레임별로 어떤 트랙 id 가 실제 검출되었는지 (파이프라인 손실 계산용)
    frame_detections: Dict[int, set] = {}
    for src in conv.sources.values():
        for det_list in src._det_by_time.values():
            for d in det_list:
                frame_detections.setdefault(d.frame_idx, set()).add(d.track_id)

    # 정답 방위/속도: 프레임 간 위치 차분으로 산출
    gt_prev: Dict[int, Dict[int, Tuple[float, float]]] = {}
    frames_sorted = sorted(gt_all)
    dt = 1.0 / cfg.deepaccident.frame_rate_hz

    n_loc = n_lane = n_actor = 0
    for snap in conv.run(rate_hz=rate_hz):
        m.n_snapshots += 1
        f = snap.frame_idx
        if f is None or f not in gt_all:
            continue
        gt = gt_all[f]

        # 관측자(ego) 위치 — 놓친 정답 판정에 필요
        obs_pos = [a.world_xy for a in snap.actors if a.kind == "ego"]
        obs_pos += [i.world_xy for i in snap.infrastructure]

        seen_ids = set()
        for a in snap.actors:
            n_actor += 1
            # ego 로 흡수된 검출도 '처리됨'으로 센다 (관측차량끼리 서로를
            # 검출한 경우 — 파이프라인 손실이 아니다)
            if a.kind == "ego":
                seen_ids.update(a.source_track_ids)
            if a.placement is not None:
                n_loc += 1
                if a.placement.lane_index is not None:
                    n_lane += 1
            if a.kind != "observed":
                continue
            # 이 스냅샷의 원본 검출 id 로 매칭한다. 레지스트리 누적 매핑을
            # 쓰면 시간이 지나며 다른 id 가 섞여 잘못된 정답과 비교된다.
            cand = [i for i in a.source_track_ids if i in gt]
            if not cand:
                m.n_unmatched_actor += 1
                continue
            oid = cand[0]
            seen_ids.add(oid)
            truth = gt[oid]
            err = math.dist(a.world_xy, truth)
            m.n_matched += 1
            m.pos_err.append(err)

            rng = a.observed_range_m or 0.0
            for lo, hi in RANGE_BINS:
                if lo <= rng < hi:
                    m.by_range.setdefault(f"{lo}-{hi}", []).append(err)
                    break

            # 반경/접선 분해 (가장 가까운 관측자 기준)
            if obs_pos:
                o = min(obs_pos, key=lambda p: math.dist(p, truth))
                r_gt = math.dist(o, truth)
                r_est = math.dist(o, a.world_xy)
                m.radial_bias.append(r_est - r_gt)
                m.lateral_err.append(
                    max(0.0, math.sqrt(max(err * err - (r_est - r_gt) ** 2, 0.0)))
                )

            # 방위/속도 정답 = 정답 궤적 차분
            i = frames_sorted.index(f) if f in frames_sorted else -1
            if i > 0:
                pf = frames_sorted[i - 1]
                if oid in gt_all.get(pf, {}):
                    p0 = gt_all[pf][oid]
                    de, dn = truth[0] - p0[0], truth[1] - p0[1]
                    span = (f - pf) * dt
                    if span > 0:
                        v_gt = math.hypot(de, dn) / span
                        if a.speed_mps is not None:
                            m.speed_err.append(abs(a.speed_mps - v_gt))
                        if a.heading_deg is not None and math.hypot(de, dn) > 0.3:
                            h_gt = math.degrees(math.atan2(de, dn)) % 360.0
                            m.heading_err.append(abs(wrap180(a.heading_deg - h_gt)))

        # 파이프라인 손실: 이 프레임에 검출은 있었는데 액터가 되지 못한 객체.
        # (관측자 반경 안의 모든 정답을 세면 전면 카메라 화각 밖·가려짐까지
        #  '놓침'으로 잡혀 지표가 무의미해진다 — 센서 커버리지는 인지 백엔드
        #  통계로 따로 본다)
        detected_ids = frame_detections.get(f, set())
        m.n_missed_gt += len(detected_ids - seen_ids)

    if n_actor:
        m.localization_rate = n_loc / n_actor
        m.lane_assigned_rate = n_lane / n_actor
    return m


def evaluate_scenario(
    root: str,
    scenario: str,
    scenario_type: Optional[str] = None,
    mode: str = "camera",
    rate_hz: float = 2.0,
    cfg: Optional[PipelineConfig] = None,
) -> Metrics:
    cfg = cfg or PipelineConfig()
    cfg.deepaccident.observation_mode = mode
    runner = DeepAccidentRunner(root, cfg)
    res = runner.build(scenario, scenario_type)
    return evaluate_build(res, cfg, rate_hz=rate_hz)


def aggregate(metrics: Sequence[Metrics]) -> Metrics:
    """여러 시나리오 지표를 합산."""
    out = Metrics(scenario_id=f"{len(metrics)}개 시나리오 합산",
                  mode=metrics[0].mode if metrics else "")
    for m in metrics:
        out.n_snapshots += m.n_snapshots
        out.n_matched += m.n_matched
        out.n_unmatched_actor += m.n_unmatched_actor
        out.n_missed_gt += m.n_missed_gt
        out.pos_err += m.pos_err
        out.radial_bias += m.radial_bias
        out.lateral_err += m.lateral_err
        out.heading_err += m.heading_err
        out.speed_err += m.speed_err
        for k, v in m.by_range.items():
            out.by_range.setdefault(k, []).extend(v)
    rates = [m.localization_rate for m in metrics if not math.isnan(m.localization_rate)]
    lanes = [m.lane_assigned_rate for m in metrics if not math.isnan(m.lane_assigned_rate)]
    if rates:
        out.localization_rate = sum(rates) / len(rates)
    if lanes:
        out.lane_assigned_rate = sum(lanes) / len(lanes)
    return out
