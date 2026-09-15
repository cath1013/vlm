"""BEV(조감도) 시계열 이미지 렌더링.

지도 위에 차량을 방위각대로 회전한 사각형으로 그려, 시나리오의 시간 변화를
`<시나리오이름>_<timestamp>.jpg` 이미지 집합으로 출력한다.
(DeepAccident 저장소 figs/First_video.gif 오른쪽 패널과 같은 형태)

    renderer = BevRenderer(network, BevConfig())
    paths = renderer.render_sequence(snapshots, out_dir, "Town03_scenario00024")

의존성
    Pillow (JPEG/PNG 출력). 없으면 SVG 로 폴백한다 — SVG 는 순수 텍스트라
    추가 의존성이 없고 벡터 품질이며 브라우저에서 바로 열린다.

그리는 것
    - 도로 노면: 중심선을 차선 수만큼 좌우로 옵셋한 다각형
    - 차선 경계선(흰색 파선) / 중앙선(노란 실선, 왕복 도로만)
    - 차량: 클래스 치수 그대로의 회전 사각형 + 진행방향 삼각 표식
    - 관측차량(ego)은 색을 달리하고 테두리를 굵게, 노변 인프라는 마름모
    - 최근 궤적 꼬리, 예상 경로(파선), 위험 쌍(빨간 연결선 + TTC)
    - HUD: 시나리오명, 시각, 프레임, 축척바, 북 방향, 범례
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import DEFAULT_CLASS_SIZES, ClassSize
from .geometry import heading_to_unit
from .roadmap import Road, RoadNetwork
from .schemas import ActorState, InfraState, SceneSnapshot

try:
    from PIL import Image, ImageDraw, ImageFont

    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False

XY = Tuple[float, float]
RGB = Tuple[int, int, int]


# ---------------------------------------------------------------- 설정


@dataclass
class BevConfig:
    """BEV 렌더링 설정."""

    width_px: int = 1200
    height_px: int = 900
    margin_m: float = 25.0  # 액터 외곽에서 남길 여유
    min_extent_m: float = 70.0  # 최소 시야 폭 (너무 확대되지 않도록)
    max_extent_m: float = 220.0  # 최대 시야 폭 (차량이 점이 되지 않도록)
    # 시야 기준: 'ego'  — 관측차량 궤적만으로 시야를 정한다 (기본).
    #            'all'  — 인프라가 본 원거리 주차차량까지 모두 포함 → 시야가 넓어짐
    # 인프라는 100m 넘는 차량도 관측하므로 'all' 은 화면이 과도하게 넓어진다.
    focus: str = "ego"
    # 시야 고정: 시퀀스 전체를 담는 고정 시야를 쓰면 프레임 간 화면이 흔들리지
    # 않아 애니메이션으로 보기 좋다. False 면 프레임마다 자동 맞춤.
    fixed_view: bool = True
    center: Optional[XY] = None  # 직접 지정 (None이면 자동)
    extent_m: Optional[float] = None  # 가로 시야 폭 (None이면 자동)
    # 라벨: 'all' | 'ego_and_risky'(관측차량 + 위험 상호작용 차량) | 'none'
    label_mode: str = "all"
    label_backdrop: bool = True  # 라벨 뒤에 반투명 배경 (도로 위 가독성)

    draw_road_surface: bool = True
    draw_lane_lines: bool = True
    draw_road_names: bool = False
    draw_ids: bool = True
    draw_speed: bool = True
    # 예상 경로(최상위 후보)를 점선으로 그린다. **기본 꺼짐**.
    # 켜면 액터 25대 전부의 점선이 겹치는데, 보행자·인도 경로는 노면이 그려지지
    # 않는 곳을 가로질러 정체를 알 수 없는 대각선 조각으로 보인다. 차량 방위
    # 미확정 표시(점선 테두리)와도 선이 헷갈린다. 켜면 범례에 항목을 넣는다.
    draw_predictions: bool = False
    # 예상 경로를 그릴 때, 이 속도 미만인 액터는 건너뛴다 [m/s]. 정지·저속
    # 물체의 예상 경로는 길이가 거의 0 인 점 뭉치라 정보가 없다.
    prediction_min_speed_mps: float = 1.5
    # 예상 경로를 그릴 때, 도로에 매칭되지 않은 액터는 건너뛴다. 그 경로는
    # 방위를 그대로 외삽한 값이어서 노면을 벗어나 뻗는다.
    prediction_require_road: bool = True
    draw_trails: bool = True
    trail_s: float = 3.0
    # 궤적 꼬리 잡음 제거. 두 이력점 사이 함의 속도가
    #   min(클래스 지속속도 상한, max(trail_noise_speed_mps, factor × 추정속도))
    # 를 넘으면 실제 이동이 아니라 단안 위치 잡음이므로 그 지점 이전 이력은
    # 그리지 않는다. 없으면 원거리 관측의 수십 m 튐이 차량에서 뻗어나가는 긴
    # 직선으로, 보행자의 위치 잡음이 인도 위를 가로지르는 지그재그로 보인다.
    trail_noise_speed_mps: float = 2.0  # 정지 물체에도 허용할 최소 잡음 여유
    trail_noise_speed_factor: float = 2.5
    # 클래스별 지속 주행 속도 상한 [m/s]. FusionConfig.report_speed_by_class 와
    # 같은 뜻이며, 렌더러가 설정 객체에 의존하지 않도록 값만 복사해 둔다.
    trail_speed_by_class: Dict[str, float] = field(
        default_factory=lambda: {
            "person": 2.5,
            "bicycle": 8.0,
            "motorcycle": 33.0,
            "car": 40.0,
            "van": 33.0,
            "truck": 28.0,
            "bus": 25.0,
        }
    )
    draw_risk_links: bool = True
    risk_ttc_s: float = 4.0  # 이 TTC 이하 상호작용을 위험선으로 표시
    draw_legend: bool = True
    draw_scalebar: bool = True

    jpeg_quality: int = 88
    image_format: str = "jpg"  # 'jpg' | 'png' | 'svg'
    class_sizes: Dict[str, ClassSize] = field(
        default_factory=lambda: dict(DEFAULT_CLASS_SIZES)
    )

    # 색
    bg: RGB = (24, 26, 30)
    road_fill: RGB = (58, 62, 68)
    road_edge: RGB = (92, 98, 106)
    lane_line: RGB = (150, 156, 164)
    center_line: RGB = (196, 176, 72)
    text: RGB = (232, 234, 238)
    text_dim: RGB = (150, 156, 164)
    risk: RGB = (232, 76, 60)
    trail: RGB = (120, 130, 145)


# 관측차량(ego) 팔레트 — 관측자별로 색을 달리해 추적하기 쉽게
EGO_PALETTE: List[RGB] = [
    (86, 190, 255),   # 하늘
    (120, 230, 160),  # 연두
    (255, 190, 90),   # 주황
    (215, 150, 255),  # 보라
    (255, 130, 170),  # 분홍
]

CLASS_COLOR: Dict[str, RGB] = {
    "car": (200, 205, 212),
    "van": (170, 200, 225),
    "truck": (235, 150, 110),
    "bus": (235, 170, 90),
    "motorcycle": (240, 225, 120),
    "bicycle": (200, 230, 140),
    "person": (130, 235, 190),
}
INFRA_COLOR: RGB = (255, 105, 180)


def _hex(c: RGB) -> str:
    return "#%02x%02x%02x" % c


# ---------------------------------------------------------------- 좌표 변환


class ViewTransform:
    """월드(ENU) → 픽셀. 북쪽이 위, 동쪽이 오른쪽."""

    def __init__(
        self, center: XY, extent_m: float, width_px: int, height_px: int
    ):
        self.center = center
        self.width_px = width_px
        self.height_px = height_px
        # 가로 시야를 extent_m 로 맞추고 세로는 종횡비대로
        self.mpp = extent_m / max(width_px, 1)
        self.extent_m = extent_m

    @property
    def extent_n_m(self) -> float:
        return self.mpp * self.height_px

    def to_px(self, p: XY) -> Tuple[float, float]:
        dx = (p[0] - self.center[0]) / self.mpp
        dy = (p[1] - self.center[1]) / self.mpp
        return (self.width_px / 2.0 + dx, self.height_px / 2.0 - dy)

    def m_to_px(self, m: float) -> float:
        return m / self.mpp

    def visible(self, p: XY, pad_m: float = 20.0) -> bool:
        return (
            abs(p[0] - self.center[0]) <= self.extent_m / 2 + pad_m
            and abs(p[1] - self.center[1]) <= self.extent_n_m / 2 + pad_m
        )


# ---------------------------------------------------------------- 기하 유틸


def _left_normals(poly: Sequence[XY]) -> List[XY]:
    """각 정점의 좌측 법선(인접 세그먼트 평균)."""
    n = len(poly)
    if n < 2:
        # 점 하나로는 방향이 없다 — 옵셋 0 과 같게 두어 호출부가 터지지 않게
        return [(0.0, 0.0)] * n
    out: List[XY] = []
    for i in range(n):
        if i == 0:
            a, b = poly[0], poly[1]
        elif i == n - 1:
            a, b = poly[-2], poly[-1]
        else:
            a, b = poly[i - 1], poly[i + 1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy) or 1.0
        out.append((-dy / L, dx / L))
    return out


def offset_polyline(poly: Sequence[XY], offset_m: float) -> List[XY]:
    """폴리라인을 좌측(+)으로 offset_m 만큼 평행 이동."""
    nrm = _left_normals(poly)
    return [
        (p[0] + nrm[i][0] * offset_m, p[1] + nrm[i][1] * offset_m)
        for i, p in enumerate(poly)
    ]


def road_surface_polygon(road: Road, lane_width_m: float, drive_side: str = "right"):
    """도로 노면 다각형 (좌측 경계 + 역순 우측 경계).

    차선 배정 규약과 같은 기준을 쓴다:
      - 왕복 도로: 진행차로는 중심선의 통행측(우측통행이면 우측), 대향차로는 반대
      - 일방통행: 중심선이 차도 중앙
    """
    left_off, right_off = road_side_offsets(road, lane_width_m, drive_side)
    left = offset_polyline(road.poly, left_off)
    right = offset_polyline(road.poly, right_off)
    return left + list(reversed(right))


def road_side_offsets(
    road: Road, lane_width_m: float, drive_side: str = "right"
) -> Tuple[float, float]:
    """(좌측 경계 오프셋, 우측 경계 오프셋) [m]. 좌측이 +."""
    w = lane_width_m
    if road.oneway:
        half = max(road.lanes_forward, 1) * w / 2.0
        return half, -half
    f = max(road.lanes_forward, 1) * w
    b = max(road.lanes_backward, 1) * w
    # 진행차로는 통행측(우측통행이면 중심선 우측 = 오프셋 음수)
    return (b, -f) if drive_side == "right" else (f, -b)


def road_surface_quads(
    road: Road, lane_width_m: float, drive_side: str = "right"
) -> List[List[XY]]:
    """노면을 **세그먼트별 사각형**으로. 곡률이 급한 구간에서도 안전하다.

    좌·우 경계 폴리라인 하나로 큰 다각형을 만들면, 곡률 반경보다 오프셋이 클 때
    내측 경계가 스스로 교차해 노면이 쐐기 모양으로 찌그러진다 (교차로 진입부에서
    도로가 끊긴 것처럼 보이는 원인). 세그먼트별 사각형은 곡선 내측에서 서로
    조금 겹칠 뿐이어서 빈틈이 생기지 않는다.
    """
    if len(road.poly) < 2:
        return []
    lo, ro = road_side_offsets(road, lane_width_m, drive_side)
    left = offset_polyline(road.poly, lo)
    right = offset_polyline(road.poly, ro)
    return [
        [left[i], left[i + 1], right[i + 1], right[i]]
        for i in range(len(road.poly) - 1)
    ]


def convex_hull(pts: Sequence[XY]) -> List[XY]:
    """볼록껍질 (모노톤 체인). scipy 없이 쓰기 위한 최소 구현."""
    ps = sorted(set(pts))
    if len(ps) <= 2:
        return list(ps)

    def cross(o: XY, a: XY, b: XY) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[XY] = []
    for pt in ps:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)
    upper: List[XY] = []
    for pt in reversed(ps):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)
    return lower[:-1] + upper[:-1]


def point_in_convex(q: XY, hull: Sequence[XY]) -> bool:
    """볼록 다각형 내부 판정. 모든 변에 대해 같은 쪽이면 내부.

    원(중심+최대반경)으로 근사하면 안 된다 — 교차로 덮개는 접근로 방향으로 길게
    늘어난 형태라, 최대 반경을 쓰면 도로를 따라 수십 m 를 과도하게 덮어 차선
    표시가 통째로 사라진다.
    """
    n = len(hull)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        cr = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
        if cr > 1e-9:
            if sign < 0:
                return False
            sign = 1
        elif cr < -1e-9:
            if sign > 0:
                return False
            sign = -1
    return True


def clip_segment_convex(
    a: XY, b: XY, hull: Sequence[XY]
) -> Optional[Tuple[float, float]]:
    """선분 a→b 가 볼록 다각형(반시계) 내부에 있는 매개변수 구간 (t0, t1).

    Cyrus–Beck. 볼록 다각형이므로 내부 구간은 항상 하나다. 겹치지 않으면 None.
    꼭짓점 단위로 자르면 안 되기 때문에 필요하다 — 폴리라인이 거칠면(합성
    도로망은 300m 도로가 정점 2개) 한쪽 끝만 교차로 안에 들어가도 남는 점이
    하나뿐이어서 선 전체가 사라진다.
    """
    n = len(hull)
    if n < 3:
        return None
    d = (b[0] - a[0], b[1] - a[1])
    tlo, thi = 0.0, 1.0
    for i in range(n):
        v0, v1 = hull[i], hull[(i + 1) % n]
        e = (v1[0] - v0[0], v1[1] - v0[1])
        # 반시계 다각형의 내부는 각 변의 왼쪽 → cross(e, q - v0) >= 0
        c = e[0] * (a[1] - v0[1]) - e[1] * (a[0] - v0[0])
        m = e[0] * d[1] - e[1] * d[0]
        if abs(m) < 1e-12:
            if c < 0:
                return None  # 이 변의 바깥에서 평행 — 전 구간 외부
            continue
        t = -c / m
        if m > 0:
            tlo = max(tlo, t)
        else:
            thi = min(thi, t)
        if tlo >= thi:
            return None
    return (tlo, thi)


def subtract_intervals(
    intervals: Sequence[Tuple[float, float]], lo: float = 0.0, hi: float = 1.0
) -> List[Tuple[float, float]]:
    """[lo, hi] 에서 구간들의 합집합을 뺀 나머지 구간들."""
    out: List[Tuple[float, float]] = []
    cur = lo
    for s0, s1 in sorted(intervals):
        if s1 <= cur:
            continue
        if s0 > cur:
            out.append((cur, min(s0, hi)))
        cur = max(cur, s1)
        if cur >= hi:
            return [(x, y) for x, y in out if y - x > 1e-9]
    if cur < hi:
        out.append((cur, hi))
    return [(x, y) for x, y in out if y - x > 1e-9]


def rect_corners(
    center: XY, heading_deg: float, length_m: float, width_m: float
) -> List[XY]:
    """진행방향으로 회전한 사각형 4개 코너 (전좌, 전우, 후우, 후좌)."""
    fe, fn = heading_to_unit(heading_deg)
    le, ln = (-fn, fe)  # 좌측 단위벡터
    hl, hw = length_m / 2.0, width_m / 2.0
    return [
        (center[0] + fe * hl + le * hw, center[1] + fn * hl + ln * hw),
        (center[0] + fe * hl - le * hw, center[1] + fn * hl - ln * hw),
        (center[0] - fe * hl - le * hw, center[1] - fn * hl - ln * hw),
        (center[0] - fe * hl + le * hw, center[1] - fn * hl + ln * hw),
    ]


def heading_arrow(center: XY, heading_deg: float, length_m: float) -> List[XY]:
    """차량 앞쪽 진행방향 삼각형."""
    fe, fn = heading_to_unit(heading_deg)
    le, ln = (-fn, fe)
    tip = (center[0] + fe * length_m * 0.62, center[1] + fn * length_m * 0.62)
    base = (center[0] + fe * length_m * 0.18, center[1] + fn * length_m * 0.18)
    w = length_m * 0.16
    return [
        tip,
        (base[0] + le * w, base[1] + ln * w),
        (base[0] - le * w, base[1] - ln * w),
    ]


# ---------------------------------------------------------------- 렌더러


class BevRenderer:
    """도로망 + 스냅샷 → BEV 이미지."""

    def __init__(
        self,
        network: Optional[RoadNetwork],
        cfg: Optional[BevConfig] = None,
        lane_width_m: Optional[float] = None,
        drive_side: str = "right",
    ):
        self.network = network
        self.cfg = cfg or BevConfig()
        if network is not None:
            self.lane_width_m = lane_width_m or network.lane_cfg.lane_width_m
            self.drive_side = drive_side or network.lane_cfg.drive_side
        else:
            self.lane_width_m = lane_width_m or 3.5
            self.drive_side = drive_side
        self._surfaces: Optional[List[Tuple[Road, List[List[XY]]]]] = None
        self._junctions: Optional[
            List[Tuple[List[XY], XY, Tuple[float, float, float, float]]]
        ] = None
        self._markings: Optional[
            List[Tuple[Road, List[Tuple[str, List[List[XY]]]]]]
        ] = None
        self._font_cache: Dict[int, object] = {}

    # ------------------------------------------------------------ 준비

    def _road_surfaces(self) -> List[Tuple[Road, List[List[XY]]]]:
        """도로별 노면 사각형 목록 (세그먼트 단위)."""
        if self._surfaces is None:
            self._surfaces = []
            if self.network is not None:
                for r in self.network.roads.values():
                    if len(r.poly) < 2:
                        continue
                    self._surfaces.append(
                        (
                            r,
                            road_surface_quads(
                                r, self.lane_width_m, self.drive_side
                            ),
                        )
                    )
        return self._surfaces

    def _junction_blankets(
        self,
    ) -> List[Tuple[List[XY], XY, Tuple[float, float, float, float]]]:
        """교차로별 (덮개 다각형, 중심, 경계상자).

        교차로 내부는 여러 개의 좁은 연결로로 표현되므로, 그것들의 노면을 그대로
        이어 붙이면 사이사이 빈틈이 남아 도로가 끊긴 것처럼 보인다. 교차로에
        접하는 도로들의 **노면 단면 꼭짓점**을 모아 볼록껍질로 한 장 덮는다.

        같은 다각형으로 차선 표시를 끊는다 — 실제 교차로 안에는 차선·중앙선
        표시가 없다. 경계상자는 내부 판정 전 빠른 배제용이다.
        """
        if self._junctions is not None:
            return self._junctions
        self._junctions = []
        if self.network is None:
            return self._junctions
        for j in self.network.junctions.values():
            if not j.is_intersection:
                continue
            pts: List[XY] = [j.xy]
            for rid in j.road_ids:
                road = self.network.roads.get(rid)
                if road is None or len(road.poly) < 2:
                    continue
                lo, ro = road_side_offsets(
                    road, self.lane_width_m, self.drive_side
                )
                left = offset_polyline(road.poly, lo)
                right = offset_polyline(road.poly, ro)
                half = max(abs(lo), abs(ro))
                if road.in_junction:
                    # 교차로 내부 연결로는 전 구간이 교차로 안이다
                    idxs = list(range(len(road.poly)))
                else:
                    # 일반 도로는 교차로에 닿은 끝점 단면만
                    idxs = [
                        0
                        if math.dist(road.poly[0], j.xy)
                        <= math.dist(road.poly[-1], j.xy)
                        else len(road.poly) - 1
                    ]
                for i in idxs:
                    pts.append(left[i])
                    pts.append(right[i])
                # 끝점 단면을 교차로 안쪽으로 도로 반폭만큼 연장한다.
                # 단면만 모으면 껍질이 마름모가 되어 교차로 네 귀퉁이가 빈다
                # (접근로가 교차로 중심까지 닿지 않는 지도에서 실제로 뚫린다).
                i = idxs[-1] if not road.in_junction else None
                if i is not None and len(road.poly) >= 2:
                    nb = road.poly[1] if i == 0 else road.poly[-2]
                    ux, uy = road.poly[i][0] - nb[0], road.poly[i][1] - nb[1]
                    ln = math.hypot(ux, uy)
                    if ln > 1e-6:
                        ux, uy = ux / ln * half, uy / ln * half
                        pts.append((left[i][0] + ux, left[i][1] + uy))
                        pts.append((right[i][0] + ux, right[i][1] + uy))
            hull = convex_hull(pts)
            if len(hull) < 3:
                continue
            xs = [q[0] for q in hull]
            ys = [q[1] for q in hull]
            self._junctions.append(
                (hull, j.xy, (min(xs), min(ys), max(xs), max(ys)))
            )
        return self._junctions

    def _outside_junctions(self, pts: Sequence[XY]) -> List[List[XY]]:
        """폴리라인에서 교차로 덮개 **밖**의 연속 구간들만.

        차선선·경계선을 교차로 안까지 그으면 교차로를 가로지르는 줄이 되어
        노면이 갈라져 보인다. 자르는 단위는 정점이 아니라 **선분**이다 (정점
        단위로 자르면 거친 폴리라인에서 선이 통째로 사라진다).
        """
        blankets = self._junction_blankets()
        if not blankets or len(pts) < 2:
            return [list(pts)] if len(pts) >= 2 else []
        runs: List[List[XY]] = []
        cur: List[XY] = []
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            lo_x, hi_x = min(a[0], b[0]), max(a[0], b[0])
            lo_y, hi_y = min(a[1], b[1]), max(a[1], b[1])
            inside: List[Tuple[float, float]] = []
            for hull, _, bb in blankets:
                if hi_x < bb[0] or lo_x > bb[2] or hi_y < bb[1] or lo_y > bb[3]:
                    continue  # 경계상자 빠른 배제
                iv = clip_segment_convex(a, b, hull)
                if iv is not None:
                    inside.append(iv)

            def at(t: float) -> XY:
                return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

            for t0, t1 in subtract_intervals(inside):
                p0, p1 = at(t0), at(t1)
                if cur and t0 <= 1e-9 and math.dist(cur[-1], p0) < 1e-6:
                    cur.append(p1)  # 직전 조각과 이어진다
                else:
                    if len(cur) >= 2:
                        runs.append(cur)
                    cur = [p0, p1]
                if t1 < 1.0 - 1e-9:
                    # 교차로 경계에서 끊긴다 — 다음 조각은 새 구간
                    if len(cur) >= 2:
                        runs.append(cur)
                    cur = []
        if len(cur) >= 2:
            runs.append(cur)
        return runs

    def _font(self, size: int):
        """트루타입 폰트를 찾고, 없으면 Pillow 기본 폰트.

        이미지 안의 모든 문자열은 라틴 문자다 — 이미지가 어느 환경에서 열릴지,
        어떤 폰트가 깔려 있을지 알 수 없는데 한글을 넣으면 두부(□)가 된다.
        코드 주석·콘솔 출력은 한글을 그대로 쓴다.
        """
        if size in self._font_cache:
            return self._font_cache[size]
        f = None
        if _HAS_PIL:
            # 리눅스 배포판 기본 폰트를 반드시 포함한다. 트루타입을 못 찾으면
            # Pillow 비트맵 기본 폰트로 떨어지는데, 그것은 범례의 em dash(—)
            # 같은 비 ASCII 문자를 두부(▯)로 그린다 (실제로 `EGO▯` 가 나왔다).
            for path in (
                "C:/Windows/Fonts/malgun.ttf",
                "C:/Windows/Fonts/gulim.ttc",
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
            ):
                if os.path.exists(path):
                    try:
                        f = ImageFont.truetype(path, size)
                        break
                    except Exception:
                        pass
            if f is None:
                try:
                    f = ImageFont.load_default(size=size)
                except TypeError:  # 구버전 Pillow
                    f = ImageFont.load_default()
        self._font_cache[size] = f
        return f

    # ------------------------------------------------------------ 시야

    def compute_view(self, snapshots: Sequence[SceneSnapshot]) -> ViewTransform:
        """스냅샷들을 모두 담는 시야를 계산."""
        cfg = self.cfg
        pts: List[XY] = []
        for s in snapshots:
            if cfg.focus == "ego":
                pts.extend(a.world_xy for a in s.actors if a.kind == "ego")
            else:
                pts.extend(a.world_xy for a in s.actors)
                pts.extend(i.world_xy for i in s.infrastructure)
        if not pts and cfg.focus == "ego":
            # 관측차량이 없으면(인프라 단독 관측 등) 전체로 되돌린다
            for s in snapshots:
                pts.extend(a.world_xy for a in s.actors)
                pts.extend(i.world_xy for i in s.infrastructure)
        if not pts:
            center = cfg.center or (0.0, 0.0)
            return ViewTransform(
                center, cfg.extent_m or cfg.min_extent_m, cfg.width_px, cfg.height_px
            )

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = cfg.center[0] if cfg.center else (min(xs) + max(xs)) / 2.0
        cy = cfg.center[1] if cfg.center else (min(ys) + max(ys)) / 2.0

        if cfg.extent_m:
            extent = cfg.extent_m
        else:
            need_e = (max(xs) - min(xs)) + 2 * cfg.margin_m
            need_n = (max(ys) - min(ys)) + 2 * cfg.margin_m
            aspect = cfg.width_px / max(cfg.height_px, 1)
            # 세로 요구를 가로 기준으로 환산해 둘 다 담기게
            extent = max(need_e, need_n * aspect, cfg.min_extent_m)
            extent = min(extent, cfg.max_extent_m)
        return ViewTransform((cx, cy), extent, cfg.width_px, cfg.height_px)

    def view_fits(self, snapshots: Sequence[SceneSnapshot]) -> bool:
        """고정 시야로 시퀀스 전체를 담을 수 있는가.

        차량이 max_extent_m 보다 멀리 이동하는 시나리오에서 고정 시야를 쓰면
        대부분의 프레임이 빈 화면이 되고 차량은 구석에 몰린다. 그때는 프레임별
        시야로 따라가야 한다.
        """
        if self.cfg.extent_m or self.cfg.center:
            return True  # 사용자가 시야를 직접 지정했으면 그 뜻을 따른다
        v = self.compute_view(snapshots)
        pad = self.cfg.margin_m
        return all(
            v.visible(a.world_xy, pad)
            for s in snapshots
            for a in s.actors
            if self.cfg.focus != "ego" or a.kind == "ego"
        )

    # ------------------------------------------------------------ 단일 프레임

    def render(
        self,
        snapshot: SceneSnapshot,
        out_path: str,
        view: Optional[ViewTransform] = None,
        trails: Optional[Dict[str, List[XY]]] = None,
        title: str = "",
        ego_colors: Optional[Dict[str, RGB]] = None,
    ) -> str:
        view = view or self.compute_view([snapshot])
        ego_colors = ego_colors or self.assign_ego_colors([snapshot])
        fmt = os.path.splitext(out_path)[1].lstrip(".").lower() or self.cfg.image_format
        if fmt == "svg" or not _HAS_PIL:
            return self._render_svg(
                snapshot, out_path, view, trails, title, ego_colors
            )
        return self._render_pil(snapshot, out_path, view, trails, title, ego_colors)

    def assign_ego_colors(
        self, snapshots: Sequence[SceneSnapshot]
    ) -> Dict[str, RGB]:
        ids = sorted(
            {a.actor_id for s in snapshots for a in s.actors if a.kind == "ego"}
        )
        return {aid: EGO_PALETTE[i % len(EGO_PALETTE)] for i, aid in enumerate(ids)}

    def _actor_size(self, a: ActorState) -> Tuple[float, float]:
        sz = self.cfg.class_sizes.get(a.cls)
        if sz is None:
            sz = self.cfg.class_sizes.get("car", ClassSize(1.9, 4.6, 1.5))
        return (sz.length_m, sz.width_m)

    def _actor_color(self, a: ActorState, ego_colors: Dict[str, RGB]) -> RGB:
        if a.kind == "ego":
            return ego_colors.get(a.actor_id, EGO_PALETTE[0])
        return CLASS_COLOR.get(a.cls, (200, 205, 212))

    # ------------------------------------------------------------ PIL 경로

    def _render_pil(
        self,
        snap: SceneSnapshot,
        out_path: str,
        view: ViewTransform,
        trails: Optional[Dict[str, List[XY]]],
        title: str,
        ego_colors: Dict[str, RGB],
    ) -> str:
        cfg = self.cfg
        img = Image.new("RGB", (cfg.width_px, cfg.height_px), cfg.bg)
        dr = ImageDraw.Draw(img, "RGBA")
        P = view.to_px

        # --- 도로 (교차로 덮개 → 노면 → 차선 표시 순)
        if cfg.draw_road_surface:
            for hull, c, _ in self._junction_blankets():
                if not view.visible(c, 60.0):
                    continue
                dr.polygon([P(q) for q in hull], fill=cfg.road_fill)
            for road, quads in self._road_surfaces():
                if not any(view.visible(p, 60.0) for p in road.poly):
                    continue
                for quad in quads:
                    dr.polygon([P(q) for q in quad], fill=cfg.road_fill)
        if cfg.draw_lane_lines:
            self._draw_lane_lines(dr, view)

        # --- 궤적 꼬리
        if cfg.draw_trails and trails:
            for aid, pts in trails.items():
                if len(pts) < 2:
                    continue
                col = ego_colors.get(aid, cfg.trail)
                dr.line(
                    [P(p) for p in pts],
                    fill=col + (150,),
                    width=max(int(view.m_to_px(0.5)), 2),
                    joint="curve",
                )

        # --- 예상 경로
        if cfg.draw_predictions:
            for a in snap.actors:
                if not self._show_prediction(a, cfg):
                    continue
                top = max(a.predictions, key=lambda p: p.probability)
                col = self._actor_color(a, ego_colors)
                self._dashed(dr, [P(p) for p in top.waypoints], col + (160,), 2)

        # --- 위험 연결선
        if cfg.draw_risk_links:
            pos = {a.actor_id: a.world_xy for a in snap.actors}
            for it in snap.interactions:
                if it.ttc_s is None or it.ttc_s > cfg.risk_ttc_s:
                    continue
                if it.subject_id not in pos or it.object_id not in pos:
                    continue
                p0, p1 = P(pos[it.subject_id]), P(pos[it.object_id])
                dr.line([p0, p1], fill=cfg.risk + (200,), width=2)
                mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
                dr.text(
                    (mx + 4, my - 14),
                    f"TTC {it.ttc_s:.1f}s",
                    fill=cfg.risk,
                    font=self._font(13),
                )

        # --- 차량 (라벨은 도형 위에 오도록 두 번 순회)
        risky = self._risky_ids(snap)
        blocked = self._hud_rects(len(ego_colors))
        for a in sorted(snap.actors, key=lambda x: x.kind != "ego"):
            self._draw_actor(dr, a, view, ego_colors)
        for a in sorted(snap.actors, key=lambda x: x.kind != "ego"):
            self._draw_actor_label(dr, a, view, ego_colors, risky, blocked)

        # --- 노변 인프라
        for s in snap.infrastructure:
            self._draw_infra(dr, s, view)

        # --- HUD
        self._draw_hud(dr, snap, view, title, ego_colors)

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        ext = os.path.splitext(out_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            img.save(out_path, "JPEG", quality=cfg.jpeg_quality, optimize=True)
        else:
            img.save(out_path)
        return out_path

    def _road_markings(
        self,
    ) -> List[Tuple[Road, List[Tuple[str, List[List[XY]]]]]]:
        """도로별 표시 선 (종류, 교차로 밖 구간들).

        프레임과 무관한 정적 기하이므로 한 번만 계산해 캐시한다 (교차로 내부
        판정이 프레임마다 반복되면 시퀀스 렌더링이 수 배 느려진다).
        """
        if self._markings is not None:
            return self._markings
        self._markings = []
        w = self.lane_width_m
        for road, _ in self._road_surfaces():
            # 교차로 내부 연결로에는 차선 표시가 없다. 좁은 리본마다 테두리를
            # 그리면 교차로가 방사형으로 찢어져 보인다.
            if road.in_junction:
                continue
            if road.oneway:
                lanes = max(road.lanes_forward, 1)
                half = lanes * w / 2.0
                edges = [half, -half]
                inner = [half - k * w for k in range(1, lanes)]
            else:
                f = max(road.lanes_forward, 1)
                b = max(road.lanes_backward, 1)
                if self.drive_side == "right":
                    edges = [b * w, -f * w]
                    inner = [k * w for k in range(1, b)] + [
                        -k * w for k in range(1, f)
                    ]
                else:
                    edges = [f * w, -b * w]
                    inner = [k * w for k in range(1, f)] + [
                        -k * w for k in range(1, b)
                    ]
            items: List[Tuple[str, List[List[XY]]]] = []
            for off in edges:
                items.append(
                    (
                        "edge",
                        self._outside_junctions(
                            offset_polyline(road.poly, off)
                        ),
                    )
                )
            for off in inner:
                items.append(
                    (
                        "lane",
                        self._outside_junctions(
                            offset_polyline(road.poly, off)
                        ),
                    )
                )
            if not road.oneway:
                items.append(("center", self._outside_junctions(road.poly)))
            self._markings.append((road, items))
        return self._markings

    def _draw_lane_lines(self, dr, view: ViewTransform) -> None:
        cfg = self.cfg
        P = view.to_px
        for road, items in self._road_markings():
            if not any(view.visible(p, 60.0) for p in road.poly):
                continue
            for kind, runs in items:
                for run in runs:
                    px = [P(q) for q in run]
                    if kind == "edge":
                        dr.line(px, fill=cfg.road_edge, width=1)
                    elif kind == "lane":
                        self._dashed(
                            dr, px, cfg.lane_line + (170,), 1,
                            dash_px=9, gap_px=9,
                        )
                    else:
                        dr.line(px, fill=cfg.center_line, width=1)

    def _draw_actor(
        self,
        dr,
        a: ActorState,
        view: ViewTransform,
        ego_colors: Dict[str, RGB],
    ) -> None:
        cfg = self.cfg
        if not view.visible(a.world_xy, 10.0):
            return
        length, width = self._actor_size(a)
        heading, assumed = self._actor_heading(a)
        col = self._actor_color(a, ego_colors)
        P = view.to_px

        if heading is None:
            # 방위각도 도로 매칭도 없음 → 원으로 (방향을 아는 척하지 않는다)
            r = max(view.m_to_px(max(length, width) / 2), 3)
            c = P(a.world_xy)
            dr.ellipse(
                [c[0] - r, c[1] - r, c[0] + r, c[1] + r],
                fill=col + (170,),
                outline=col,
            )
        else:
            corners = [P(p) for p in rect_corners(a.world_xy, heading, length, width)]
            alpha = 235 if a.kind == "ego" else 200
            dr.polygon(corners, fill=col + (alpha,))
            outline_w = 3 if a.kind == "ego" else 1
            # 도로 방향으로 가정한 방위는 점선 테두리로 구분한다
            edge = (20, 22, 26)
            if assumed:
                self._dashed(dr, corners + [corners[0]], edge + (220,), outline_w,
                             dash_px=4, gap_px=3)
            else:
                dr.line(corners + [corners[0]], fill=edge, width=outline_w)
            if view.m_to_px(length) > 12 and not assumed:
                arrow = [P(p) for p in heading_arrow(a.world_xy, heading, length)]
                dr.polygon(arrow, fill=(20, 22, 26, 200))

    def _actor_heading(self, a: ActorState) -> Tuple[Optional[float], bool]:
        """(그리기에 쓸 방위각, 도로 방향으로 가정했는지).

        궤적이 짧거나 정지 중이면 방위각 추정이 안 되는데, 그때마다 원을 그리면
        조감도에서 차량 방향을 전혀 읽을 수 없다. 도로에 매칭된 차량은 그 지점의
        도로 진행 방위(bearing_deg)를 쓰고 **점선 테두리**로 가정임을 표시한다.
        """
        if a.heading_deg is not None:
            return a.heading_deg, False
        if a.placement is not None:
            return a.placement.bearing_deg, True
        return None, False

    def _hud_rects(self, n_ego: int) -> List[Tuple[float, float, float, float]]:
        """HUD·범례가 불투명하게 덮는 화면 영역 (x0, y0, x1, y1).

        이 안에 그려진 액터 라벨은 나중에 HUD 로 덮여 반쯤 잘린 글자로 남는다.
        읽을 수 없는 라벨은 그리지 않는 편이 낫다.
        """
        cfg = self.cfg
        rects = [(0.0, 0.0, float(cfg.width_px), 54.0)]
        if cfg.draw_legend and n_ego:
            # 범례 항목 = 관측차량 n_ego + 고정 2행(+ 예상경로 켜면 1행)
            n_extra = 3 if cfg.draw_predictions else 2
            rects.append(
                (0.0, 56.0, 260.0, 62.0 + 18.0 * (n_ego + n_extra) + 6.0)
            )
        return rects

    def _risky_ids(self, snap: SceneSnapshot) -> set:
        """위험 상호작용에 연루된 액터 id."""
        out = set()
        for it in snap.interactions:
            if it.ttc_s is not None and it.ttc_s <= self.cfg.risk_ttc_s:
                out.add(it.subject_id)
                out.add(it.object_id)
        return out

    def _draw_actor_label(
        self,
        dr,
        a: ActorState,
        view: ViewTransform,
        ego_colors: Dict[str, RGB],
        risky: set,
        blocked: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> None:
        cfg = self.cfg
        if cfg.label_mode == "none" or not cfg.draw_ids:
            return
        if cfg.label_mode == "ego_and_risky" and a.kind != "ego" and (
            a.actor_id not in risky
        ):
            return
        if not view.visible(a.world_xy, 10.0):
            return
        length, width = self._actor_size(a)
        label = short_id(a.actor_id)
        if cfg.draw_speed and a.speed_mps is not None:
            label += f" {a.speed_mps * 3.6:.0f}km/h"
        size = 13 if a.kind == "ego" else 12
        font = self._font(size)
        c = view.to_px(a.world_xy)
        off = max(view.m_to_px(max(length, width) / 2) + 5, 9)
        x, y = c[0] + off, c[1] - size / 2 - 2
        for bx0, by0, bx1, by1 in blocked or ():
            if x <= bx1 and y <= by1 and y + size >= by0:
                return  # HUD·범례가 덮을 자리 — 잘린 글자를 남기지 않는다
        if cfg.label_backdrop:
            try:
                bb = dr.textbbox((x, y), label, font=font)
                dr.rectangle(
                    [bb[0] - 3, bb[1] - 1, bb[2] + 3, bb[3] + 1],
                    fill=(12, 14, 18, 175),
                )
            except Exception:  # pragma: no cover - 구버전 Pillow
                pass
        col = cfg.text if (a.kind == "ego" or a.actor_id in risky) else cfg.text_dim
        dr.text((x, y), label, fill=col, font=font)

    def _draw_infra(self, dr, s: InfraState, view: ViewTransform) -> None:
        if not view.visible(s.world_xy, 10.0):
            return
        P = view.to_px
        c = P(s.world_xy)
        r = 9
        dr.polygon(
            [(c[0], c[1] - r), (c[0] + r, c[1]), (c[0], c[1] + r), (c[0] - r, c[1])],
            fill=INFRA_COLOR + (220,),
            outline=(20, 22, 26),
        )
        if self.cfg.draw_ids:
            dr.text(
                (c[0] + r + 3, c[1] - 7),
                f"{s.infra_id} (infra)",
                fill=INFRA_COLOR,
                font=self._font(12),
            )

    def _draw_hud(
        self,
        dr,
        snap: SceneSnapshot,
        view: ViewTransform,
        title: str,
        ego_colors: Dict[str, RGB],
    ) -> None:
        cfg = self.cfg
        f14, f13, f12 = self._font(15), self._font(13), self._font(12)
        # 상단 반투명 띠
        dr.rectangle([0, 0, cfg.width_px, 52], fill=(0, 0, 0, 150))
        head = title or snap.area_name or ""
        dr.text((12, 8), head, fill=cfg.text, font=f14)
        sub = f"t = {snap.t:.1f}s"
        if snap.frame_idx is not None:
            sub += f"  |  frame {snap.frame_idx}"
        n_ego = sum(1 for a in snap.actors if a.kind == "ego")
        sub += (
            f"  |  observers {n_ego}   road users {len(snap.actors) - n_ego}"
            f"   infra {len(snap.infrastructure)}"
        )
        dr.text((12, 30), sub, fill=cfg.text_dim, font=f12)

        if cfg.draw_scalebar:
            self._draw_scalebar(dr, view)
        # 북 방향
        x0, y0 = cfg.width_px - 34, 74
        dr.line([(x0, y0 + 22), (x0, y0 - 12)], fill=cfg.text, width=2)
        dr.polygon(
            [(x0, y0 - 18), (x0 - 5, y0 - 8), (x0 + 5, y0 - 8)], fill=cfg.text
        )
        dr.text((x0 - 4, y0 + 24), "N", fill=cfg.text, font=f12)

        if cfg.draw_legend:
            y = 62
            rows = [
                (short_id(aid), aid.replace("EGO_", ""), col)
                for aid, col in ego_colors.items()
            ]
            # 점선으로 그리는 것이 있으면 범례에 넣는다 — 설명 없는 점선은
            # 지도 표시인지 예측인지 알 수 없어 오류로 읽힌다.
            extras = ["observed road user", "roadside infrastructure"]
            if cfg.draw_predictions:
                extras.append("predicted path (dashed)")
            w = 40 + max(
                [int(dr.textlength(s, font=f12)) for s in extras]
                + [
                    int(dr.textlength(f"{sh} — {full}", font=f12))
                    for sh, full, _ in rows
                ]
            )
            dr.rectangle(
                [8, y - 4, 8 + w, y + 18 * (len(rows) + len(extras)) + 4],
                fill=(0, 0, 0, 130),
            )
            for sh, full, col in rows:
                dr.rectangle([14, y + 3, 30, y + 13], fill=col)
                dr.text((36, y), f"{sh} — {full}", fill=cfg.text, font=f12)
                y += 18
            dr.rectangle([14, y + 3, 30, y + 13], fill=CLASS_COLOR["car"])
            dr.text((36, y), "observed road user", fill=cfg.text_dim, font=f12)
            y += 18
            dr.polygon(
                [(22, y + 1), (30, y + 8), (22, y + 15), (14, y + 8)],
                fill=INFRA_COLOR,
            )
            dr.text((36, y), "roadside infrastructure", fill=cfg.text_dim, font=f12)
            if cfg.draw_predictions:
                y += 18
                self._dashed(
                    dr, [(14, y + 8), (30, y + 8)], CLASS_COLOR["car"] + (200,), 2,
                    dash_px=5, gap_px=3,
                )
                dr.text(
                    (36, y), "predicted path (dashed)", fill=cfg.text_dim, font=f12
                )

    def _draw_scalebar(self, dr, view: ViewTransform) -> None:
        cfg = self.cfg
        # 화면 폭의 1/6 에 가까운 "좋은" 길이 선택
        target_m = view.extent_m / 6.0
        nice = [5, 10, 20, 25, 50, 100, 200, 500]
        bar_m = min(nice, key=lambda v: abs(v - target_m))
        px = view.m_to_px(bar_m)
        x1 = cfg.width_px - 24
        x0 = x1 - px
        y = cfg.height_px - 24
        dr.line([(x0, y), (x1, y)], fill=cfg.text, width=2)
        dr.line([(x0, y - 5), (x0, y + 5)], fill=cfg.text, width=2)
        dr.line([(x1, y - 5), (x1, y + 5)], fill=cfg.text, width=2)
        dr.text(
            ((x0 + x1) / 2 - 14, y - 20), f"{bar_m} m", fill=cfg.text,
            font=self._font(12),
        )

    @staticmethod
    def _show_prediction(a, cfg: BevConfig) -> bool:
        """이 액터의 예상 경로를 그릴 가치가 있는가.

        걸러내는 것이 셋이다.
          - 후보가 없거나 점이 1개 — 그릴 선이 없다
          - 저속·정지 — 경로가 점 뭉치라 정보가 없고 차량 사각형과 겹친다
          - 도로 미매칭 — 방위를 그대로 외삽한 값이라 노면 밖으로 뻗는다.
            보행자가 인도를 걷는 경우가 대표적이고, 노면이 그려지지 않는 곳에
            정체 불명의 대각선 조각으로 남는다.
        """
        if not a.predictions:
            return False
        top = max(a.predictions, key=lambda p: p.probability)
        if len(top.waypoints) < 2:
            return False
        if a.speed_mps is None or a.speed_mps < cfg.prediction_min_speed_mps:
            return False
        if cfg.prediction_require_road and a.placement is None:
            return False
        return True

    @staticmethod
    def _dashed(dr, pts, color, width, dash_px: int = 8, gap_px: int = 6) -> None:
        """파선 폴리라인."""
        carry = 0.0
        draw_on = True
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            seg = math.hypot(x1 - x0, y1 - y0)
            if seg < 1e-6:
                continue
            pos = 0.0
            while pos < seg:
                step = (dash_px if draw_on else gap_px) - carry
                step = min(step, seg - pos)
                t0, t1 = pos / seg, (pos + step) / seg
                if draw_on:
                    dr.line(
                        [
                            (x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0),
                            (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1),
                        ],
                        fill=color,
                        width=width,
                    )
                pos += step
                carry += step
                if carry >= (dash_px if draw_on else gap_px) - 1e-9:
                    carry = 0.0
                    draw_on = not draw_on

    # ------------------------------------------------------------ SVG 폴백

    def _render_svg(
        self,
        snap: SceneSnapshot,
        out_path: str,
        view: ViewTransform,
        trails: Optional[Dict[str, List[XY]]],
        title: str,
        ego_colors: Dict[str, RGB],
    ) -> str:
        cfg = self.cfg
        P = view.to_px
        out = os.path.splitext(out_path)[0] + ".svg"
        L: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{cfg.width_px}" '
            f'height="{cfg.height_px}" viewBox="0 0 {cfg.width_px} {cfg.height_px}">',
            f'<rect width="100%" height="100%" fill="{_hex(cfg.bg)}"/>',
        ]

        def path(pts, **kw):
            d = " ".join(
                ("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}"
                for i, (x, y) in enumerate(pts)
            )
            attrs = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in kw.items())
            L.append(f'<path d="{d}" {attrs}/>')

        if cfg.draw_road_surface:
            for hull, c, _ in self._junction_blankets():
                if not view.visible(c, 60.0):
                    continue
                path(
                    [P(q) for q in hull] + [P(hull[0])],
                    fill=_hex(cfg.road_fill),
                    stroke="none",
                )
            for road, quads in self._road_surfaces():
                if not any(view.visible(p, 60.0) for p in road.poly):
                    continue
                for quad in quads:
                    path(
                        [P(q) for q in quad] + [P(quad[0])],
                        fill=_hex(cfg.road_fill),
                        stroke="none",
                    )
        if cfg.draw_lane_lines:
            for road, items in self._road_markings():
                if not any(view.visible(p, 60.0) for p in road.poly):
                    continue
                for kind, runs in items:
                    if kind != "center":
                        continue
                    for run in runs:
                        path(
                            [P(q) for q in run],
                            fill="none",
                            stroke=_hex(cfg.center_line),
                            stroke_width="1",
                        )

        if trails:
            for aid, pts in trails.items():
                if len(pts) < 2:
                    continue
                path(
                    [P(p) for p in pts],
                    fill="none",
                    stroke=_hex(ego_colors.get(aid, cfg.trail)),
                    stroke_width="2",
                    stroke_opacity="0.6",
                )

        for a in snap.actors:
            if not view.visible(a.world_xy, 10.0):
                continue
            col = _hex(self._actor_color(a, ego_colors))
            length, width = self._actor_size(a)
            if a.heading_deg is None:
                c = P(a.world_xy)
                r = max(view.m_to_px(max(length, width) / 2), 3)
                L.append(
                    f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="{r:.1f}" '
                    f'fill="{col}" fill-opacity="0.7"/>'
                )
            else:
                corners = [
                    P(p)
                    for p in rect_corners(a.world_xy, a.heading_deg, length, width)
                ]
                path(
                    corners + [corners[0]],
                    fill=col,
                    fill_opacity="0.85",
                    stroke="#14161a",
                    stroke_width="2" if a.kind == "ego" else "1",
                )
            if cfg.draw_ids:
                c = P(a.world_xy)
                off = max(view.m_to_px(max(length, width) / 2) + 4, 8)
                L.append(
                    f'<text x="{c[0] + off:.1f}" y="{c[1]:.1f}" font-size="12" '
                    f'fill="{_hex(cfg.text)}">{a.actor_id}</text>'
                )

        for s in snap.infrastructure:
            c = P(s.world_xy)
            L.append(
                f'<polygon points="{c[0]:.1f},{c[1]-9:.1f} {c[0]+9:.1f},{c[1]:.1f} '
                f'{c[0]:.1f},{c[1]+9:.1f} {c[0]-9:.1f},{c[1]:.1f}" '
                f'fill="{_hex(INFRA_COLOR)}"/>'
            )

        head = title or snap.area_name or ""
        L.append(
            f'<text x="12" y="24" font-size="15" fill="{_hex(cfg.text)}">'
            f"{head}</text>"
        )
        L.append(
            f'<text x="12" y="42" font-size="12" fill="{_hex(cfg.text_dim)}">'
            f"t = {snap.t:.1f}s</text>"
        )
        L.append("</svg>")

        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(L))
        return out

    # ------------------------------------------------------------ 시퀀스

    def render_sequence(
        self,
        snapshots: Sequence[SceneSnapshot],
        out_dir: str,
        scenario_name: str,
        title: str = "",
        image_format: Optional[str] = None,
    ) -> List[str]:
        """스냅샷 시퀀스 → `<scenario_name>_<timestamp>.<ext>` 이미지 집합.

        기본적으로 시퀀스 전체를 담는 **고정 시야**를 쓴다. 프레임마다 시야를
        다시 맞추면 화면이 흔들려 시간 변화를 읽기 어렵다.
        """
        snaps = sorted(snapshots, key=lambda s: s.t)
        if not snaps:
            return []
        fmt = (image_format or self.cfg.image_format).lstrip(".")
        # 고정 시야로 담을 수 없는 긴 이동 시나리오는 프레임별 시야로 자동 전환
        fixed = self.cfg.fixed_view and self.view_fits(snaps)
        view = self.compute_view(snaps) if fixed else None
        ego_colors = self.assign_ego_colors(snaps)
        os.makedirs(out_dir, exist_ok=True)

        history: Dict[str, List[Tuple[float, XY]]] = {}
        paths: List[str] = []
        for snap in snaps:
            for a in snap.actors:
                history.setdefault(a.actor_id, []).append((snap.t, a.world_xy))
            state = {a.actor_id: a for a in snap.actors}
            trails = {
                aid: _plausible_trail(
                    [
                        (t, xy)
                        for (t, xy) in pts
                        if snap.t - t <= self.cfg.trail_s + 1e-9
                        and t <= snap.t + 1e-9
                    ],
                    state[aid].speed_mps if aid in state else None,
                    self.cfg.trail_noise_speed_mps,
                    self.cfg.trail_noise_speed_factor,
                    self.cfg.trail_speed_by_class.get(
                        state[aid].cls if aid in state else ""
                    ),
                )
                for aid, pts in history.items()
            }
            stamp = _stamp(snap.t)
            name = f"{scenario_name}_{stamp}.{fmt}"
            out_path = os.path.join(out_dir, name)
            v = view or self.compute_view([snap])
            paths.append(
                self.render(
                    snap,
                    out_path,
                    view=v,
                    trails=trails,
                    title=title or scenario_name,
                    ego_colors=ego_colors,
                )
            )
        return paths


# 관측차량 actor id → 지도 위 짧은 표시명. 전체 이름(EGO_other_vehicle_behind)을
# 차량 옆에 쓰면 라벨이 서로 겹쳐 아무것도 읽을 수 없다. 범례에 전체 이름이 있다.
EGO_SHORT: Dict[str, str] = {
    "EGO_ego_vehicle": "EGO",
    "EGO_ego_vehicle_behind": "EGO-B",
    "EGO_other_vehicle": "OTH",
    "EGO_other_vehicle_behind": "OTH-B",
}


def _plausible_trail(
    pts: Sequence[Tuple[float, XY]],
    speed_mps: Optional[float],
    floor_mps: float,
    factor: float,
    cap_mps: Optional[float] = None,
) -> List[XY]:
    """이력점 목록 → 현재 위치에서 거꾸로 이어지는 타당한 구간만.

    뒤에서부터 훑어 함의 속도가 상한을 넘는 첫 단계에서 멈춘다. 어느 쪽 점이
    틀렸는지는 알 수 없으므로, 가장 최근의 연속 구간만 남긴다.

    cap_mps: 이 클래스가 낼 수 있는 지속 주행 속도. 추정속도 비례항만 쓰면
      잡음으로 부풀려진 속도가 자기 자신의 잡음 궤적을 정당화한다
      (보행자 추정속도 5m/s → 허용 12.5m/s → 잡음 그대로 통과).
    """
    if len(pts) < 2:
        return [xy for _, xy in pts]
    if speed_mps is None:
        # 속도 미확정 — 클래스 상한만으로 판단한다
        limit = cap_mps if cap_mps is not None else floor_mps
    else:
        limit = max(floor_mps, factor * speed_mps)
        if cap_mps is not None:
            limit = min(limit, cap_mps)
    out = [pts[-1][1]]
    for i in range(len(pts) - 1, 0, -1):
        dt = pts[i][0] - pts[i - 1][0]
        d = math.dist(pts[i][1], pts[i - 1][1])
        if dt > 1e-6 and d / dt > limit:
            break
        out.append(pts[i - 1][1])
    out.reverse()
    return out


def short_id(actor_id: str) -> str:
    """지도 위 라벨용 짧은 식별자."""
    if actor_id in EGO_SHORT:
        return EGO_SHORT[actor_id]
    return actor_id[4:] if actor_id.startswith("EGO_") else actor_id


def _stamp(t: float) -> str:
    """파일명용 타임스탬프. 정렬이 사전순과 일치하도록 0 채움 (예: 003.5s)."""
    return f"{t:07.2f}s".replace(".", "p")


def write_index_html(
    paths: Sequence[str], out_path: str, title: str = "BEV 시계열"
) -> str:
    """생성된 이미지들을 순서대로 훑어보는 간단한 HTML 인덱스.

    이미지 뷰어를 따로 열지 않고 시간 변화를 확인할 수 있게 한다.
    """
    rows = "\n".join(
        f'<figure><img src="{os.path.basename(p)}" loading="lazy">'
        f"<figcaption>{os.path.basename(p)}</figcaption></figure>"
        for p in paths
    )
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>{title}</title><style>
body{{background:#15171b;color:#e8eaee;font-family:system-ui,sans-serif;margin:16px}}
h1{{font-size:18px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:10px}}
figure{{margin:0}} img{{width:100%;border-radius:6px;display:block}}
figcaption{{font-size:12px;color:#9aa0a8;margin-top:4px}}
</style></head><body><h1>{title} — {len(paths)}프레임</h1>
<div class="grid">
{rows}
</div></body></html>"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
