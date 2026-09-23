"""CARLA 타운 지도 지원 (OpenDRIVE .xodr 임포트).

DeepAccident 는 지도 파일을 배포에 포함하지 않지만, 시나리오 이름의 접두어가
CARLA 기본 타운(Town01~Town10HD)이므로 CARLA 설치본의 .xodr 을 쓰면 궤적
합성보다 정확한 도로망을 얻을 수 있다.

    <CARLA>/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr

좌표계
    CARLA 런타임 월드는 좌수계(y=남)지만 **.xodr 의 y 는 그 부호를 뒤집은**
    우수계다. 따라서 .xodr 좌표 (x, y) 는 본 패키지의 ENU (e, n) 과 그대로
    일치한다. (CARLA 월드 좌표는 e = x, n = -y 로 변환해야 한다 —
    deepaccident.carla_to_enu 참조)

지원 범위
    planView 기하: line, arc, spiral(수치적분), paramPoly3
    lanes: laneSection 별 driving 차선 수, 차선폭, 제한속도
    교차로: RoadNetwork 의 끝점 클러스터링을 재사용 (junction 요소는 참고만)
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import LaneConfig
from .geometry import LocalENU
from .roadmap import Road, RoadNetwork

XY = Tuple[float, float]

# CARLA 기본 타운 요약 — 시나리오 town 접두어로 조회한다.
CARLA_TOWNS: Dict[str, str] = {
    "Town01": "소규모 격자형, 왕복 2차선, 교차로 다수, 다리 1개",
    "Town02": "소규모 주거지, 왕복 2차선",
    "Town03": "대규모 도심, 다차선·로터리·터널·고가",
    "Town04": "산악 배경, 무한 8자 고속도로 + 소도시",
    "Town05": "격자형 도심, 다차선 왕복, 고가도로",
    "Town06": "장거리 다차선 고속도로, 미시간 좌회전",
    "Town07": "농촌, 좁은 도로, 곡물창고",
    "Town10HD": "고화질 도심, 다차선 대로",
}


def town_description(town: str) -> str:
    return CARLA_TOWNS.get(town, "")


def default_xodr_path(carla_root: str, town: str) -> str:
    """CARLA 설치 경로에서 타운 .xodr 예상 위치."""
    return os.path.join(
        carla_root, "CarlaUE4", "Content", "Carla", "Maps", "OpenDrive", f"{town}.xodr"
    )


def find_xodr(town: str, search_dirs: List[str]) -> Optional[str]:
    """여러 후보 디렉터리에서 <town>.xodr 을 찾는다."""
    for d in search_dirs:
        if not d:
            continue
        cand = os.path.join(d, f"{town}.xodr")
        if os.path.isfile(cand):
            return cand
        cand2 = default_xodr_path(d, town)
        if os.path.isfile(cand2):
            return cand2
    return None


# ---------------------------------------------------------------- planView


@dataclass
class Geometry:
    """planView 기하 요소 하나."""

    s: float
    x: float
    y: float
    hdg: float
    length: float
    kind: str = "line"
    curvature: float = 0.0
    curv_start: float = 0.0
    curv_end: float = 0.0
    poly: Optional[Tuple[float, ...]] = None  # paramPoly3 계수 (aU..dU, aV..dV)
    p_range_normalized: bool = True

    def point_at(self, ds: float) -> XY:
        """요소 시작점 기준 ds 만큼 진행한 지점."""
        ds = max(0.0, min(ds, self.length))
        if self.kind == "line":
            return (
                self.x + ds * math.cos(self.hdg),
                self.y + ds * math.sin(self.hdg),
            )
        if self.kind == "arc":
            k = self.curvature
            if abs(k) < 1e-12:
                return (
                    self.x + ds * math.cos(self.hdg),
                    self.y + ds * math.sin(self.hdg),
                )
            h1 = self.hdg + k * ds
            return (
                self.x + (math.sin(h1) - math.sin(self.hdg)) / k,
                self.y + (-math.cos(h1) + math.cos(self.hdg)) / k,
            )
        if self.kind == "spiral":
            # 곡률이 선형 변화 — 수치적분 (오일러 나선)
            n = max(int(ds / 0.5), 8)
            step = ds / n
            x, y, h = self.x, self.y, self.hdg
            dk = (self.curv_end - self.curv_start) / (self.length or 1.0)
            for i in range(n):
                s_i = (i + 0.5) * step
                k = self.curv_start + dk * s_i
                x += step * math.cos(h)
                y += step * math.sin(h)
                h += k * step
            return (x, y)
        if self.kind == "poly3" and self.poly:
            aU, bU, cU, dU, aV, bV, cV, dV = self.poly
            p = ds / self.length if (self.p_range_normalized and self.length) else ds
            u = aU + bU * p + cU * p * p + dU * p ** 3
            v = aV + bV * p + cV * p * p + dV * p ** 3
            ch, sh = math.cos(self.hdg), math.sin(self.hdg)
            return (self.x + u * ch - v * sh, self.y + u * sh + v * ch)
        # 미지 형식 → 직선 근사
        return (self.x + ds * math.cos(self.hdg), self.y + ds * math.sin(self.hdg))


def _parse_geometry(el: ET.Element) -> Geometry:
    g = Geometry(
        s=float(el.get("s", 0.0)),
        x=float(el.get("x", 0.0)),
        y=float(el.get("y", 0.0)),
        hdg=float(el.get("hdg", 0.0)),
        length=float(el.get("length", 0.0)),
    )
    if el.find("line") is not None:
        g.kind = "line"
    elif (a := el.find("arc")) is not None:
        g.kind = "arc"
        g.curvature = float(a.get("curvature", 0.0))
    elif (sp := el.find("spiral")) is not None:
        g.kind = "spiral"
        g.curv_start = float(sp.get("curvStart", 0.0))
        g.curv_end = float(sp.get("curvEnd", 0.0))
    elif (pp := el.find("paramPoly3")) is not None:
        g.kind = "poly3"
        g.poly = tuple(
            float(pp.get(k, 0.0))
            for k in ("aU", "bU", "cU", "dU", "aV", "bV", "cV", "dV")
        )
        g.p_range_normalized = pp.get("pRange", "normalized") != "arcLength"
    return g


def _sample_reference_line(
    geoms: List[Geometry], total_length: float, step_m: float
) -> List[XY]:
    """planView 전체를 step_m 간격으로 샘플링."""
    if not geoms:
        return []
    geoms = sorted(geoms, key=lambda g: g.s)
    pts: List[XY] = []
    n = max(int(total_length / step_m), 1)
    for i in range(n + 1):
        s = min(total_length * i / n, total_length)
        g = geoms[0]
        for cand in geoms:
            if cand.s <= s + 1e-9:
                g = cand
            else:
                break
        pts.append(g.point_at(s - g.s))
    # 중복 제거
    out: List[XY] = []
    for p in pts:
        if not out or math.dist(out[-1], p) > 1e-6:
            out.append(p)
    return out


# ---------------------------------------------------------------- lanes


def _count_driving_lanes(road_el: ET.Element) -> Tuple[int, int, Optional[float]]:
    """(정방향 차선수, 역방향 차선수, 차선폭).

    OpenDRIVE 규약: 기준선 진행방향 오른쪽(음수 id) 차선이 정방향,
    왼쪽(양수 id) 차선이 역방향이다. driving 타입만 센다.
    차선 수가 구간마다 다르면 최대값을 쓴다 (교차로 접근부의 확장 반영).
    """
    best_r = best_l = 0
    widths: List[float] = []
    lanes = road_el.find("lanes")
    if lanes is None:
        return (1, 1, None)
    for sec in lanes.findall("laneSection"):
        for side, key in (("right", "r"), ("left", "l")):
            grp = sec.find(side)
            if grp is None:
                continue
            n = 0
            for lane in grp.findall("lane"):
                if lane.get("type") != "driving":
                    continue
                n += 1
                w = lane.find("width")
                if w is not None:
                    a = float(w.get("a", 0.0))
                    if a > 0.5:
                        widths.append(a)
            if key == "r":
                best_r = max(best_r, n)
            else:
                best_l = max(best_l, n)
    width = None
    if widths:
        widths.sort()
        width = widths[len(widths) // 2]
    return (best_r, best_l, width)


def _road_speed_limit_kph(road_el: ET.Element) -> Optional[float]:
    for t in road_el.findall(".//type"):
        sp = t.find("speed")
        if sp is None:
            continue
        try:
            v = float(sp.get("max", "nan"))
        except ValueError:
            continue
        if math.isnan(v):
            continue
        unit = (sp.get("unit") or "m/s").lower()
        if unit in ("mph",):
            return v * 1.60934
        if unit in ("km/h", "kph"):
            return v
        return v * 3.6  # m/s
    for lane in road_el.findall(".//lane"):
        sp = lane.find("speed")
        if sp is not None:
            try:
                v = float(sp.get("max", "nan"))
            except ValueError:
                continue
            if not math.isnan(v):
                unit = (sp.get("unit") or "m/s").lower()
                return v if unit in ("km/h", "kph") else (
                    v * 1.60934 if unit == "mph" else v * 3.6
                )
    return None


# ---------------------------------------------------------------- 진입점


def load_opendrive(
    path: str,
    lane_cfg: Optional[LaneConfig] = None,
    enu: Optional[LocalENU] = None,
    step_m: float = 4.0,
    include_junction_roads: bool = True,
    min_length_m: float = 3.0,
    flip_y: bool = False,
) -> RoadNetwork:
    """OpenDRIVE(.xodr) → RoadNetwork.

    flip_y: .xodr 의 y 부호를 뒤집는다. CARLA .xodr 은 이미 우수계이므로
      기본 False. 다른 도구에서 만든 좌수계 파일이면 True 로 둔다.
    include_junction_roads: junction 내부 연결로(junction 속성이 -1 이 아닌
      도로)를 포함할지. 포함하면 교차로 내부 경로가 이어지지만 도로 수가
      크게 늘어난다.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"OpenDRIVE 파일이 없습니다: {path}")
    lane_cfg = lane_cfg or LaneConfig()
    enu = enu or LocalENU(0.0, 0.0)

    tree = ET.parse(path)
    root = tree.getroot()

    roads: List[Road] = []
    lane_widths: List[float] = []
    for r in root.findall("road"):
        junction = r.get("junction", "-1")
        if junction not in ("-1", "", None) and not include_junction_roads:
            continue
        try:
            total = float(r.get("length", 0.0))
        except ValueError:
            continue
        if total < min_length_m:
            continue
        pv = r.find("planView")
        if pv is None:
            continue
        geoms = [_parse_geometry(g) for g in pv.findall("geometry")]
        pts = _sample_reference_line(geoms, total, step_m)
        if len(pts) < 2:
            continue
        if flip_y:
            pts = [(x, -y) for x, y in pts]

        lanes_f, lanes_b, w = _count_driving_lanes(r)
        if w:
            lane_widths.append(w)
        if lanes_f == 0 and lanes_b == 0:
            continue  # driving 차선이 없는 도로 (보도 등)
        oneway = lanes_b == 0 or lanes_f == 0
        if lanes_f == 0:
            # driving 차선이 모두 기준선 **왼쪽**(양수 id)에 있는 일방통행.
            # OpenDRIVE 에서 왼쪽 차선의 통행방향은 -s 이므로, 폴리라인을 뒤집어
            # '정방향 = 통행방향'을 유지한다. 뒤집지 않고 lanes_f 로 옮기기만
            # 하면 진행 방위가 180° 반대가 되어(실측 반전율 57%) 조감도에서
            # 차량이 도로를 거꾸로 달리고, 차선번호·진행방향 라벨도 뒤집힌다.
            pts = list(reversed(pts))
            lanes_f, lanes_b = lanes_b, 0
        elif oneway:
            lanes_b = 0

        rid = r.get("id") or f"road{len(roads)}"
        name = r.get("name") or ""
        length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        roads.append(
            Road(
                road_id=f"od_{rid}",
                name=name or f"도로{rid}",
                poly=pts,
                oneway=oneway,
                lanes_forward=max(lanes_f, 1),
                lanes_backward=lanes_b,
                speed_limit_kph=_road_speed_limit_kph(r),
                highway="opendrive",
                length_m=length,
                inferred=False,
                observed_directions=2 if lanes_b else 1,
                in_junction=junction not in ("-1", ""),
                notes="OpenDRIVE" + (" (교차로 내부)" if junction not in ("-1", "") else ""),
            )
        )

    if not roads:
        raise ValueError(f"{path}: driving 차선을 가진 road 를 찾지 못했습니다")

    # 파일에 기재된 차선폭을 설정에 반영 (CARLA 는 보통 3.5m)
    if lane_widths:
        lane_widths.sort()
        lane_cfg.lane_width_m = lane_widths[len(lane_widths) // 2]

    return RoadNetwork(roads, lane_cfg, enu)


def opendrive_summary(path: str) -> str:
    """파일을 읽지 않고 개요만 (진단용)."""
    tree = ET.parse(path)
    root = tree.getroot()
    n_road = len(root.findall("road"))
    n_junc = len(root.findall("junction"))
    hdr = root.find("header")
    name = hdr.get("name", "") if hdr is not None else ""
    geo = hdr.find("geoReference") if hdr is not None else None
    return (
        f"{os.path.basename(path)}: road {n_road}, junction {n_junc}"
        + (f", name={name}" if name else "")
        + (f", geoRef={(geo.text or '').strip()[:60]}" if geo is not None else "")
    )
