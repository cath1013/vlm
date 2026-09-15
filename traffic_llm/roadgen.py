"""궤적 기반 도로망 합성.

도로 지도(GeoJSON/OpenDRIVE)가 없을 때, 관측된 차량 궤적으로부터 도로망을
역추정한다. DeepAccident 같은 시뮬레이션 데이터에는 별도 지도 파일이 없지만
모든 차량의 정확한 궤적이 있으므로, 통행 궤적을 차로(corridor)로 묶고
차로를 도로로 묶어 중심선과 차선 수를 복원할 수 있다.

절차
    1) 궤적을 방위각이 일정한 구간으로 분할 (회전 지점에서 끊음)
    2) 동일 직선·동일 방향 구간을 묶어 **차로(corridor)** 생성
    3) 평행한 차로들을 묶어 **도로(Road)** 생성, 방향별 차선 수 산출
    4) 도로 축을 서로 교차시켜 **교차로** 위치를 찾고 그 지점에서 도로 분할

한계 (합성 도로망은 관측된 통행만 반영한다)
    - 차량이 지나가지 않은 차선·도로는 존재하지 않는 것으로 나온다.
      한 방향만 관측되면 그 방향만 있는 일방통행으로 표기한다(중앙선 위치를
      추정할 근거가 없으므로 관측 차로들의 중심을 중심선으로 삼는다).
    - 따라서 차선 수는 **하한**이다. Road.inferred=True 로 표시하며 직렬화
      단계에서 LLM 에 이 불확실성을 명시한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import LaneConfig, RoadGenConfig
from .geometry import LocalENU, unit_to_heading, wrap180, wrap360
from .roadmap import Road, RoadNetwork

XY = Tuple[float, float]

AXIS_LABELS = [
    (0.0, "남북"),
    (45.0, "북동-남서"),
    (90.0, "동서"),
    (135.0, "북서-남동"),
]


def _axis_label(bearing_deg: float) -> str:
    """축 방향 이름. 왕복 도로는 방향이 둘이므로 축(180도 모듈로)으로 표기."""
    a = bearing_deg % 180.0
    best = min(AXIS_LABELS, key=lambda z: min(abs(a - z[0]), 180.0 - abs(a - z[0])))
    return best[1]


# ---------------------------------------------------------------- 궤적 분할


@dataclass
class TrackSegment:
    """방위각이 거의 일정한 궤적 구간."""

    track_id: str
    points: List[XY]
    bearing_deg: float
    length_m: float

    @property
    def start(self) -> XY:
        return self.points[0]

    @property
    def end(self) -> XY:
        return self.points[-1]


def split_track_into_segments(
    track_id: str,
    points: Sequence[XY],
    cfg: RoadGenConfig,
    min_step_m: float = 0.5,
) -> List[TrackSegment]:
    """궤적을 방위각 일정 구간으로 분할. 정지 구간은 버린다."""
    # 정지/미세이동 제거 — 방위각 계산이 불안정해진다
    pts: List[XY] = []
    for p in points:
        if not pts or math.dist(pts[-1], p) >= min_step_m:
            pts.append((float(p[0]), float(p[1])))
    if len(pts) < 3:
        return []

    bearings = [
        unit_to_heading(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        for i in range(len(pts) - 1)
    ]

    segs: List[TrackSegment] = []
    start = 0
    ref = bearings[0]
    for i in range(1, len(bearings)):
        if abs(wrap180(bearings[i] - ref)) > cfg.heading_bin_deg:
            segs.append(_make_segment(track_id, pts[start : i + 1]))
            start = i
            ref = bearings[i]
        else:
            # 구간 내 평균으로 기준 방위각을 갱신 (완만한 곡선 추종)
            ref = ref + 0.2 * wrap180(bearings[i] - ref)
    segs.append(_make_segment(track_id, pts[start:]))

    return [
        s
        for s in segs
        if s is not None and s.length_m >= cfg.min_corridor_len_m
        and len(s.points) >= cfg.min_points_per_corridor
    ]


def _make_segment(track_id: str, pts: Sequence[XY]) -> Optional[TrackSegment]:
    if len(pts) < 2:
        return None
    length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    bearing = unit_to_heading(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
    return TrackSegment(track_id, list(pts), bearing, length)


# ---------------------------------------------------------------- 차로


@dataclass
class Corridor:
    """동일 직선·동일 진행방향의 통행 축 = 차로 1개."""

    origin: np.ndarray  # 축 위의 한 점
    direction: np.ndarray  # 단위 진행벡터
    bearing_deg: float
    s_min: float
    s_max: float
    points: List[XY] = field(default_factory=list)
    track_ids: set = field(default_factory=set)

    def project(self, p: XY) -> Tuple[float, float]:
        """(종방향 s, 좌측(+) 횡방향 offset)."""
        d = np.array(p, dtype=float) - self.origin
        s = float(d @ self.direction)
        left = np.array([-self.direction[1], self.direction[0]])
        return (s, float(d @ left))

    @property
    def length_m(self) -> float:
        return self.s_max - self.s_min

    def center_point(self) -> XY:
        mid = self.origin + self.direction * (0.5 * (self.s_min + self.s_max))
        return (float(mid[0]), float(mid[1]))

    def endpoint(self, s: float) -> XY:
        p = self.origin + self.direction * s
        return (float(p[0]), float(p[1]))


def _fit_axis(points: Sequence[XY], bearing_hint: float) -> Tuple[np.ndarray, np.ndarray]:
    """점군에 직선 적합 (PCA). 진행방향은 bearing_hint 와 같은 쪽으로 맞춘다."""
    P = np.asarray(points, dtype=float)
    origin = P.mean(axis=0)
    if len(P) >= 2:
        u, s, vt = np.linalg.svd(P - origin, full_matrices=False)
        d = vt[0]
    else:  # pragma: no cover
        d = np.array([0.0, 1.0])
    n = np.linalg.norm(d)
    d = d / n if n > 1e-9 else np.array([0.0, 1.0])
    # 방향 정렬: hint 와 반대면 뒤집는다
    hint = np.array(
        [math.sin(math.radians(bearing_hint)), math.cos(math.radians(bearing_hint))]
    )
    if float(d @ hint) < 0:
        d = -d
    return origin, d


def build_corridors(
    segments: Sequence[TrackSegment], cfg: RoadGenConfig
) -> List[Corridor]:
    """궤적 구간을 차로로 병합."""
    corridors: List[Corridor] = []
    # 긴 구간부터 처리해 축 적합이 안정적이도록
    for seg in sorted(segments, key=lambda s: -s.length_m):
        placed = False
        for c in corridors:
            if abs(wrap180(seg.bearing_deg - c.bearing_deg)) > cfg.heading_bin_deg:
                continue
            offs = [c.project(p)[1] for p in seg.points]
            if max(abs(o) for o in offs) > cfg.corridor_width_m:
                continue
            ss = [c.project(p)[0] for p in seg.points]
            c.s_min = min(c.s_min, min(ss))
            c.s_max = max(c.s_max, max(ss))
            c.points.extend(seg.points)
            c.track_ids.add(seg.track_id)
            placed = True
            break
        if placed:
            continue
        origin, d = _fit_axis(seg.points, seg.bearing_deg)
        c = Corridor(
            origin=origin,
            direction=d,
            bearing_deg=unit_to_heading(float(d[0]), float(d[1])),
            s_min=0.0,
            s_max=0.0,
            points=list(seg.points),
            track_ids={seg.track_id},
        )
        ss = [c.project(p)[0] for p in seg.points]
        c.s_min, c.s_max = min(ss), max(ss)
        corridors.append(c)

    return [c for c in corridors if c.length_m >= cfg.min_corridor_len_m]


# ---------------------------------------------------------------- 도로


@dataclass
class RoadGroup:
    """평행 차로들의 묶음 = 도로 1개."""

    axis_origin: np.ndarray
    axis_dir: np.ndarray  # 기준(정) 방향
    corridors_fwd: List[Corridor] = field(default_factory=list)
    corridors_bwd: List[Corridor] = field(default_factory=list)

    @property
    def bearing_deg(self) -> float:
        return unit_to_heading(float(self.axis_dir[0]), float(self.axis_dir[1]))

    def lateral_of(self, c: Corridor) -> float:
        """차로의 기준축 대비 좌측(+) 오프셋.

        중심점 하나가 아니라 구성 점들의 중앙값을 쓴다. 축 적합에 미세한
        각도 오차가 있을 때 중심점만 보면 종방향 위치에 따라 오프셋이
        달라져 같은 차선이 여러 차선으로 세어진다.
        """
        left = np.array([-self.axis_dir[1], self.axis_dir[0]])
        pts = c.points if c.points else [c.center_point()]
        offs = [float((np.array(p, dtype=float) - self.axis_origin) @ left) for p in pts]
        return float(np.median(offs))

    def refit_axis(self) -> None:
        """구성 차로의 모든 점으로 축을 재적합.

        축 방향에 각도 오차가 있으면 오차 × 종방향거리 만큼 횡오프셋이
        왜곡된다 (110m 도로에서 1도면 1.9m — 차선 하나 폭). 그룹이 확정된
        뒤 전체 점으로 다시 맞춰 이 오차를 줄인다.
        """
        pts: List[XY] = []
        for c in self.corridors_fwd + self.corridors_bwd:
            pts.extend(c.points)
        if len(pts) < 2:
            return
        origin, d = _fit_axis(pts, self.bearing_deg)
        self.axis_origin, self.axis_dir = origin, d

    def s_range(self) -> Tuple[float, float]:
        lo, hi = math.inf, -math.inf
        for c in self.corridors_fwd + self.corridors_bwd:
            for p in (c.endpoint(c.s_min), c.endpoint(c.s_max)):
                s = float((np.array(p) - self.axis_origin) @ self.axis_dir)
                lo, hi = min(lo, s), max(hi, s)
        return (lo, hi)


def group_corridors_into_roads(
    corridors: Sequence[Corridor], cfg: RoadGenConfig
) -> List[RoadGroup]:
    groups: List[RoadGroup] = []
    for c in sorted(corridors, key=lambda z: -z.length_m):
        placed = False
        for g in groups:
            diff = abs(wrap180(c.bearing_deg - g.bearing_deg))
            same = diff <= cfg.heading_bin_deg
            opposite = abs(diff - 180.0) <= cfg.heading_bin_deg
            if not (same or opposite):
                continue
            if abs(g.lateral_of(c)) > cfg.road_group_width_m:
                continue
            # 종방향으로 겹치지 않으면 다른 도로 (같은 축의 먼 구간)
            gs = g.s_range()
            cs0 = float((np.array(c.endpoint(c.s_min)) - g.axis_origin) @ g.axis_dir)
            cs1 = float((np.array(c.endpoint(c.s_max)) - g.axis_origin) @ g.axis_dir)
            lo, hi = min(cs0, cs1), max(cs0, cs1)
            if hi < gs[0] - cfg.extend_m or lo > gs[1] + cfg.extend_m:
                continue
            (g.corridors_fwd if same else g.corridors_bwd).append(c)
            placed = True
            break
        if not placed:
            groups.append(
                RoadGroup(
                    axis_origin=np.array(c.center_point(), dtype=float),
                    axis_dir=c.direction.copy(),
                    corridors_fwd=[c],
                )
            )
    return groups


def split_group_by_lateral_gaps(
    g: RoadGroup, max_gap_m: float
) -> List[RoadGroup]:
    """횡방향 공백이 큰 곳에서 도로 그룹을 분리.

    차로들을 횡오프셋 순으로 늘어놓고, 사이 간격이 max_gap_m 을 넘으면 그
    사이에는 차선이 들어갈 수 없으므로 다른 도로로 본다. 넓은 도로 그룹
    윈도(road_group_width_m) 하나로는 나란한 별개 도로를 구분할 수 없다.
    """
    members = [(g.lateral_of(c), c, True) for c in g.corridors_fwd]
    members += [(g.lateral_of(c), c, False) for c in g.corridors_bwd]
    if len(members) <= 1:
        return [g]
    members.sort(key=lambda z: z[0])

    clusters: List[List[Tuple[float, Corridor, bool]]] = [[members[0]]]
    for m in members[1:]:
        if m[0] - clusters[-1][-1][0] > max_gap_m:
            clusters.append([m])
        else:
            clusters[-1].append(m)

    if len(clusters) == 1:
        return [g]

    out: List[RoadGroup] = []
    for cl in clusters:
        ng = RoadGroup(
            axis_origin=g.axis_origin.copy(),
            axis_dir=g.axis_dir.copy(),
            corridors_fwd=[c for _, c, is_f in cl if is_f],
            corridors_bwd=[c for _, c, is_f in cl if not is_f],
        )
        ng.refit_axis()
        out.append(ng)
    return out


def _merge_lane_offsets(
    lats: Sequence[float], min_sep_m: float
) -> List[float]:
    """차로 횡오프셋들을 차선 단위로 병합.

    같은 차선 안에서도 회전 접근·차폭 차이로 궤적이 흔들려 여러 차로로
    분리될 수 있다. min_sep_m 보다 가까운 오프셋은 한 차선으로 본다.
    """
    out: List[float] = []
    for x in sorted(lats):
        if not out or x - out[-1] >= min_sep_m:
            out.append(x)
        else:
            out[-1] = 0.5 * (out[-1] + x)  # 같은 차선 → 중심 갱신
    return out


def _group_to_road(
    g: RoadGroup, idx: int, lane_cfg: LaneConfig, cfg: RoadGenConfig
) -> Optional[Road]:
    """RoadGroup → Road. 중심선 위치와 차선 수를 결정한다."""
    fwd = sorted(g.corridors_fwd, key=g.lateral_of)
    bwd = sorted(g.corridors_bwd, key=g.lateral_of)
    if not fwd and not bwd:
        return None

    min_sep = cfg.lane_separation_frac * lane_cfg.lane_width_m
    lanes_f_lat = _merge_lane_offsets([g.lateral_of(c) for c in fwd], min_sep)
    lanes_b_lat = _merge_lane_offsets([g.lateral_of(c) for c in bwd], min_sep)

    n_dirs = (1 if fwd else 0) + (1 if bwd else 0)
    if n_dirs == 2:
        # 왕복 도로: 중앙선은 두 진행방향 차선군 사이
        # 우측통행이면 정방향 차선은 중심선 오른쪽(오프셋 음수)에 있다
        center_lat = 0.5 * (max(lanes_f_lat) + min(lanes_b_lat))
        oneway = False
        notes = "왕복 통행 관측"
    else:
        # 한 방향만 관측: 중앙선 위치를 알 근거가 없으므로 관측 차선군의
        # 중심을 중심선으로 두고 일방통행으로 표기한다 (미관측 차선을
        # 임의로 만들지 않는다)
        only = lanes_f_lat or lanes_b_lat
        center_lat = 0.5 * (min(only) + max(only))
        oneway = True
        notes = "단일 방향만 관측 — 반대편 차선은 미관측"

    left = np.array([-g.axis_dir[1], g.axis_dir[0]])
    origin = g.axis_origin + left * center_lat
    s0, s1 = g.s_range()
    if not math.isfinite(s0) or s1 - s0 < cfg.min_corridor_len_m:
        return None
    s0 -= cfg.extend_m
    s1 += cfg.extend_m

    n = max(int((s1 - s0) / cfg.resample_m), 1)
    poly = [
        (
            float(origin[0] + g.axis_dir[0] * (s0 + (s1 - s0) * k / n)),
            float(origin[1] + g.axis_dir[1] * (s0 + (s1 - s0) * k / n)),
        )
        for k in range(n + 1)
    ]

    lanes_f = len(lanes_f_lat)
    lanes_b = len(lanes_b_lat)
    if oneway:
        lanes_f = max(lanes_f, lanes_b, 1)
        lanes_b = 0

    bearing = g.bearing_deg
    name = f"도로{idx}({_axis_label(bearing)})"
    length = sum(math.dist(poly[i], poly[i + 1]) for i in range(len(poly) - 1))
    return Road(
        road_id=f"gen_{idx}",
        name=name,
        poly=poly,
        oneway=oneway,
        lanes_forward=lanes_f,
        lanes_backward=lanes_b,
        speed_limit_kph=None,
        highway="inferred",
        length_m=length,
        inferred=True,
        observed_directions=n_dirs,
        notes=notes,
    )


# ---------------------------------------------------------------- 교차로 분할


def _segment_intersection(
    p1: XY, p2: XY, p3: XY, p4: XY
) -> Optional[Tuple[float, float, float, float]]:
    """두 선분 교점. 반환 (x, y, t1, t2), t 는 각 선분의 매개변수 0~1."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if abs(den) < 1e-9:
        return None
    t1 = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / den
    t2 = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / den
    return (x1 + t1 * (x2 - x1), y1 + t1 * (y2 - y1), t1, t2)


def split_roads_at_intersections(
    roads: List[Road], cfg: RoadGenConfig, min_angle_deg: float = 25.0
) -> List[Road]:
    """도로 축이 서로 교차하는 지점에서 도로를 분할.

    RoadNetwork 은 도로 끝점을 클러스터링해 교차로를 만들기 때문에, 교차점에서
    미리 끊어 두면 교차로가 자동으로 인식된다.
    """
    # 도로별 분할 지점 (종방향 거리 s)
    cuts: Dict[str, List[float]] = {r.road_id: [] for r in roads}

    for i, a in enumerate(roads):
        for b in roads[i + 1 :]:
            ang = abs(wrap180(_road_bearing(a) - _road_bearing(b)))
            ang = min(ang, 180.0 - ang)
            if ang < min_angle_deg:
                continue
            hit = _segment_intersection(a.poly[0], a.poly[-1], b.poly[0], b.poly[-1])
            if hit is None:
                continue
            x, y, t1, t2 = hit
            # 두 도로 구간 내부(약간의 여유 포함)에서 만나야 교차로
            if not (-0.02 <= t1 <= 1.02 and -0.02 <= t2 <= 1.02):
                continue
            cuts[a.road_id].append(t1 * a.length_m)
            cuts[b.road_id].append(t2 * b.length_m)

    out: List[Road] = []
    for r in roads:
        cs = sorted(
            s
            for s in cuts[r.road_id]
            if cfg.min_corridor_len_m * 0.4 < s < r.length_m - cfg.min_corridor_len_m * 0.4
        )
        # 서로 너무 가까운 분할점은 하나로 병합
        merged: List[float] = []
        for s in cs:
            if not merged or s - merged[-1] > cfg.junction_snap_m:
                merged.append(s)
        if not merged:
            out.append(r)
            continue

        bounds = [0.0] + merged + [r.length_m]
        for k in range(len(bounds) - 1):
            sub = _subpoly(r, bounds[k], bounds[k + 1], cfg.resample_m)
            if len(sub) < 2:
                continue
            length = sum(math.dist(sub[i], sub[i + 1]) for i in range(len(sub) - 1))
            if length < cfg.min_road_len_m:
                continue  # 교차로 근처 파편 — 지도에 넣으면 매칭을 방해한다
            out.append(
                Road(
                    road_id=f"{r.road_id}_{k}",
                    name=r.name,
                    poly=sub,
                    oneway=r.oneway,
                    lanes_forward=r.lanes_forward,
                    lanes_backward=r.lanes_backward,
                    speed_limit_kph=r.speed_limit_kph,
                    highway=r.highway,
                    length_m=length,
                    inferred=r.inferred,
                    observed_directions=r.observed_directions,
                    notes=r.notes,
                )
            )
    return out


def _road_bearing(r: Road) -> float:
    return unit_to_heading(r.poly[-1][0] - r.poly[0][0], r.poly[-1][1] - r.poly[0][1])


def _subpoly(r: Road, s0: float, s1: float, step: float) -> List[XY]:
    n = max(int((s1 - s0) / step), 1)
    return [r.point_at(s0 + (s1 - s0) * k / n) for k in range(n + 1)]


# ---------------------------------------------------------------- 진입점


@dataclass
class RoadGenReport:
    """합성 결과 요약 — 무엇을 근거로 만들었는지 설명할 수 있어야 한다."""

    n_tracks: int = 0
    n_segments: int = 0
    n_corridors: int = 0
    n_road_groups: int = 0
    n_roads: int = 0
    n_junctions: int = 0
    one_way_only_roads: int = 0
    expected_road_type: str = ""

    def summary(self) -> str:
        s = (
            f"궤적 {self.n_tracks}개 → 구간 {self.n_segments} → 차로 {self.n_corridors}"
            f" → 도로 {self.n_roads}(교차로 {self.n_junctions})"
        )
        if self.one_way_only_roads:
            s += f", 단일방향만 관측된 도로 {self.one_way_only_roads}개"
        if self.expected_road_type:
            s += f" | meta 기대값: {self.expected_road_type}"
        return s


def synthesize_road_network(
    tracks: Dict[str, Sequence[XY]],
    lane_cfg: Optional[LaneConfig] = None,
    cfg: Optional[RoadGenConfig] = None,
    enu: Optional[LocalENU] = None,
    expected_road_type: str = "",
) -> Tuple[RoadNetwork, RoadGenReport]:
    """궤적 딕셔너리 {track_id: [(e,n), ...]} → RoadNetwork.

    enu 를 주면 RoadNetwork 이 위경도 변환에 사용한다 (파이프라인이 텔레메트리
    위경도를 ENU 로 바꿀 때 동일한 원점을 써야 한다).
    """
    lane_cfg = lane_cfg or LaneConfig()
    cfg = cfg or RoadGenConfig()
    enu = enu or LocalENU(0.0, 0.0)

    rep = RoadGenReport(n_tracks=len(tracks), expected_road_type=expected_road_type)

    segments: List[TrackSegment] = []
    for tid, pts in tracks.items():
        segments.extend(split_track_into_segments(str(tid), pts, cfg))
    rep.n_segments = len(segments)

    corridors = build_corridors(segments, cfg)
    rep.n_corridors = len(corridors)

    groups = group_corridors_into_roads(corridors, cfg)
    for g in groups:
        g.refit_axis()  # 그룹 확정 후 전체 점으로 축 재적합 (횡오프셋 정확도)
    # 횡방향 공백이 큰 곳에서 나란한 별개 도로를 분리
    max_gap = cfg.max_lane_gap_frac * lane_cfg.lane_width_m
    split: List[RoadGroup] = []
    for g in groups:
        split.extend(split_group_by_lateral_gaps(g, max_gap))
    groups = split
    rep.n_road_groups = len(groups)

    roads: List[Road] = []
    for i, g in enumerate(groups, start=1):
        r = _group_to_road(g, i, lane_cfg, cfg)
        if r is not None:
            roads.append(r)
    rep.one_way_only_roads = sum(1 for r in roads if r.observed_directions == 1)

    roads = split_roads_at_intersections(roads, cfg)
    rep.n_roads = len(roads)

    if not roads:
        raise ValueError(
            "궤적에서 도로를 추출하지 못했습니다. 궤적이 너무 짧거나 "
            "min_corridor_len_m 이 큽니다."
        )

    net = RoadNetwork(roads, lane_cfg, enu)
    net.junctions = {}
    net._build_junctions(snap_m=cfg.junction_snap_m)
    rep.n_junctions = sum(1 for j in net.junctions.values() if j.is_intersection)
    return net, rep
