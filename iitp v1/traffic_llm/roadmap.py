"""도로 지도 로딩 및 위치 매칭.

입력 포맷: GeoJSON FeatureCollection (LineString). OSM 태그 관례를 따른다.
  properties: name, lanes, lanes:forward, lanes:backward, oneway, maxspeed, highway

osmnx / OpenDRIVE / lanelet2 를 쓰는 경우 load_* 함수만 교체하면 나머지는 그대로 동작한다.
(교체 지점: RoadNetwork.from_geojson → 동일한 Road 객체 리스트를 만들면 됨)
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import LaneConfig
from .geometry import (
    LocalENU,
    bearing_of_segment,
    project_point_to_polyline,
    wrap180,
)
from .schemas import RoadPlacement

DIRECTION_LABELS_KO = [
    (0, "북행"),
    (45, "북동행"),
    (90, "동행"),
    (135, "남동행"),
    (180, "남행"),
    (225, "남서행"),
    (270, "서행"),
    (315, "북서행"),
]


def direction_label(bearing_deg: float, lang: str = "ko") -> str:
    idx = int(((bearing_deg % 360) + 22.5) // 45) % 8
    label = DIRECTION_LABELS_KO[idx][1]
    if lang == "en":
        return [
            "northbound",
            "northeastbound",
            "eastbound",
            "southeastbound",
            "southbound",
            "southwestbound",
            "westbound",
            "northwestbound",
        ][idx]
    return label


def _parse_maxspeed(raw) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().lower()
    try:
        if "mph" in s:
            return float(s.replace("mph", "").strip()) * 1.60934
        return float(s.split()[0])
    except (ValueError, IndexError):
        return None


@dataclass
class Road:
    """단일 도로 구간 (OSM way 하나에 대응)."""

    road_id: str
    name: str
    poly: List[Tuple[float, float]]  # ENU 폴리라인
    oneway: bool = False
    lanes_forward: int = 1
    lanes_backward: int = 1
    speed_limit_kph: Optional[float] = None
    highway: str = "unclassified"
    node_start: str = ""
    node_end: str = ""
    length_m: float = 0.0
    # 궤적에서 합성된 도로인지. True 면 차선 수·중앙선 위치가 관측된 통행에서
    # 추정된 값이므로 미관측 차선이 누락될 수 있다.
    inferred: bool = False
    # 교차로 **내부** 연결로인지 (OpenDRIVE road 의 junction 속성이 -1 이 아님).
    # 교차로 안에는 차선 표시가 없고, 연결로 하나하나는 폭이 좁은 리본이라
    # 그대로 테두리·차선선을 그리면 교차로가 방사형으로 찢어져 보인다.
    in_junction: bool = False
    observed_directions: int = 0  # 합성 시 관측된 진행방향 수 (1 또는 2)
    notes: str = ""

    def bearing_at(self, seg_idx: int) -> float:
        i = min(max(seg_idx, 0), len(self.poly) - 2)
        return bearing_of_segment(self.poly[i], self.poly[i + 1])

    def point_at(self, s: float) -> Tuple[float, float]:
        """시작점부터 s[m] 지점의 좌표 (선형보간)."""
        acc = 0.0
        for i in range(len(self.poly) - 1):
            a, b = self.poly[i], self.poly[i + 1]
            L = math.dist(a, b)
            if acc + L >= s:
                t = (s - acc) / L if L > 1e-9 else 0.0
                return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            acc += L
        return self.poly[-1]


@dataclass
class Junction:
    junction_id: str
    xy: Tuple[float, float]
    road_ids: List[str] = field(default_factory=list)

    @property
    def is_intersection(self) -> bool:
        return len(self.road_ids) >= 3


class RoadNetwork:
    """도로망 + 위치 매칭 + 하류 경로 열거."""

    def __init__(self, roads: List[Road], lane_cfg: LaneConfig, enu: LocalENU):
        self.roads: Dict[str, Road] = {r.road_id: r for r in roads}
        self.lane_cfg = lane_cfg
        self.enu = enu
        self.junctions: Dict[str, Junction] = {}
        self._build_junctions()

    # ------------------------------------------------------------ 로딩

    @classmethod
    def from_geojson(
        cls,
        path: str,
        lane_cfg: Optional[LaneConfig] = None,
        origin: Optional[Tuple[float, float]] = None,
    ) -> "RoadNetwork":
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)

        feats = [
            ft
            for ft in gj.get("features", [])
            if ft.get("geometry", {}).get("type") == "LineString"
        ]
        if not feats:
            raise ValueError(f"{path}: LineString feature 가 없습니다")

        if origin is None:
            pts = [c for ft in feats for c in ft["geometry"]["coordinates"]]
            origin = (
                sum(p[1] for p in pts) / len(pts),
                sum(p[0] for p in pts) / len(pts),
            )
        enu = LocalENU(origin[0], origin[1])

        roads: List[Road] = []
        for i, ft in enumerate(feats):
            props = ft.get("properties", {}) or {}
            coords = ft["geometry"]["coordinates"]  # [lon, lat] 순서
            poly = [enu.to_enu(lat=c[1], lon=c[0]) for c in coords]
            oneway = str(props.get("oneway", "no")).lower() in ("yes", "true", "1")
            total = int(props.get("lanes", 2) or 2)
            lf = props.get("lanes:forward")
            lb = props.get("lanes:backward")
            if oneway:
                lanes_f, lanes_b = total, 0
            else:
                lanes_f = int(lf) if lf else max(total // 2, 1)
                lanes_b = int(lb) if lb else max(total - lanes_f, 1)
            rid = str(props.get("id") or props.get("osmid") or f"road_{i}")
            roads.append(
                Road(
                    road_id=rid,
                    name=props.get("name") or props.get("ref") or f"도로{i}",
                    poly=poly,
                    oneway=oneway,
                    lanes_forward=lanes_f,
                    lanes_backward=lanes_b,
                    speed_limit_kph=_parse_maxspeed(props.get("maxspeed")),
                    highway=props.get("highway", "unclassified"),
                    length_m=sum(
                        math.dist(poly[k], poly[k + 1]) for k in range(len(poly) - 1)
                    ),
                )
            )
        return cls(roads, lane_cfg or LaneConfig(), enu)

    # ------------------------------------------------------------ 교차로 구성

    def _build_junctions(self, snap_m: float = 8.0) -> None:
        """도로 끝점들을 근접 클러스터링하여 노드 생성."""
        endpoints: List[Tuple[str, str, Tuple[float, float]]] = []
        for r in self.roads.values():
            endpoints.append((r.road_id, "start", r.poly[0]))
            endpoints.append((r.road_id, "end", r.poly[-1]))

        clusters: List[Tuple[Tuple[float, float], List[Tuple[str, str]]]] = []
        for rid, which, xy in endpoints:
            for c_xy, members in clusters:
                if math.dist(c_xy, xy) <= snap_m:
                    members.append((rid, which))
                    break
            else:
                clusters.append((xy, [(rid, which)]))

        for k, (xy, members) in enumerate(clusters):
            jid = f"J{k}"
            rids = sorted({rid for rid, _ in members})
            self.junctions[jid] = Junction(jid, xy, rids)
            for rid, which in members:
                road = self.roads[rid]
                if which == "start":
                    road.node_start = jid
                else:
                    road.node_end = jid

    # ------------------------------------------------------------ 위치 매칭

    def locate(
        self,
        world_xy: Tuple[float, float],
        heading_deg: Optional[float],
        max_dist_m: float = 25.0,
        reverse: Optional[bool] = None,
        trust_heading_direction: bool = False,
    ) -> Optional[RoadPlacement]:
        """월드 좌표(+방위각)를 도로망에 매칭.

        heading_deg 는 두 가지로 쓴다.
          - 어느 도로에 속하는지 고르는 **정렬 점수**: 방향과 무관하게 도로축과의
            각도(min(정방향차, 역방향차))를 쓴다. 뒤집힌 방위 추정 때문에 올바른
            도로가 탈락하지 않도록 하기 위함이다.
          - 진행 **방향**(순/역) 판정: 긴 창 순변위에서 얻은 힌트를 기대한다.
            방향은 1비트이므로 큰 변위로 판정하면 안정적이다.

        방위 힌트가 없을 때만 **통행측**으로 방향을 판별한다 (우측통행이면
        진행차로가 중심선 우측). 다만 궤적에서 합성한 도로(inferred)는 중심선
        자체가 추정값이어서 통행측이 뒤바뀔 수 있으므로 쓰지 않는다.

        reverse 를 명시하면 그 값을 그대로 쓴다 (호출부가 궤적의 축방향 부호를
        검증해 이미 방향을 정한 경우).

        trust_heading_direction=True 면 heading_deg 로 방향까지 정한다. 텔레메트리
        (GPS/IMU)나 3D 센서가 준 방위일 때만 켠다 — 단안 궤적 방위로 방향을
        정하면 뒤집힌 추정이 그대로 확정된다.
        """
        best_score = float("inf")
        best: Optional[RoadPlacement] = None

        for road in self.roads.values():
            s, lat_off, dist, seg = project_point_to_polyline(world_xy, road.poly)
            half_width = (
                max(road.lanes_forward + road.lanes_backward, 1)
                * self.lane_cfg.lane_width_m
                / 2.0
                + 4.0
            )
            if dist > max(max_dist_m, half_width):
                continue

            fwd_bearing = road.bearing_at(seg)
            diff_f = diff_b = None
            axis_pen = 0.0
            if heading_deg is not None:
                diff_f = abs(wrap180(heading_deg - fwd_bearing))
                diff_b = abs(wrap180(heading_deg - (fwd_bearing + 180.0)))
                axis_pen = min(diff_f, diff_b)  # 방향 무관 — 도로축 정렬도
                if axis_pen > 70.0:  # 이 도로와 무관한 방향
                    continue

            if reverse is None:
                rev, confident, dir_src = self._is_reverse(
                    road, lat_off, diff_f, diff_b, trust_heading_direction
                )
            else:
                rev, confident, dir_src = reverse, True, "trajectory"

            score = dist + axis_pen * 0.25
            if score >= best_score:
                continue

            travel_bearing = wrap180(fwd_bearing + (180.0 if rev else 0.0)) % 360.0
            # 진행방향 기준으로 횡오프셋 부호 반전
            travel_lat = -lat_off if rev else lat_off
            lane_count = road.lanes_backward if rev else road.lanes_forward
            lane_count = max(lane_count, 1)
            lane_idx = self._lane_index(road, travel_lat, lane_count, rev)
            jid, jdist = self._next_junction(road, s, rev)

            best_score = score
            best = RoadPlacement(
                road_id=road.road_id,
                road_name=road.name,
                s_m=s,
                lateral_offset_m=travel_lat,
                direction_label=direction_label(travel_bearing),
                bearing_deg=travel_bearing,
                lane_index=lane_idx,
                lane_count=lane_count,
                speed_limit_kph=road.speed_limit_kph,
                dist_to_next_junction_m=jdist,
                next_junction_id=jid,
                is_oneway=road.oneway,
                confidence=max(0.0, 1.0 - dist / max(max_dist_m, 1.0)),
                axis_bearing_deg=fwd_bearing,
                direction_confident=confident,
                direction_source=dir_src,
            )
        return best

    def _is_reverse(
        self,
        road: Road,
        lat_off: float,
        diff_f: Optional[float],
        diff_b: Optional[float],
        trust_heading: bool = False,
        side_frac: float = 0.35,
    ) -> Tuple[bool, bool, str]:
        """(역방향인가, 확신하는가, 무엇으로 정했는가).

        판단 순서
          1) 신뢰할 수 있는 방위(trust_heading) — 텔레메트리·3D 센서. 가장 정확
             하므로 일방통행 판정보다 먼저 본다.
          2) 일방통행 표기
          3) 통행측 — 우측통행에서 진행차로는 중심선 우측(lat_off < 0)이다.
             중심선에서 차선폭 × side_frac 이상 떨어져 있어야 신뢰한다.
             합성 도로망(inferred)은 중심선이 추정값이라 제외한다.
          4) 아무 단서도 없으면 순방향
        """
        w = self.lane_cfg.lane_width_m
        if trust_heading and diff_f is not None and diff_b is not None:
            # 실측 방위(텔레메트리·3D 센서)가 있으면 그것이 가장 정확하다.
            # 일방통행 판정보다 먼저 본다 — 교차로 연결로처럼 차도 경계가
            # 애매한 일방통행에서 확신을 잃지 않도록.
            return diff_b < diff_f, True, "heading"
        if road.oneway:
            # 통행방향 = 폴리라인 정방향 (임포터가 그렇게 맞춘다). 다만 차량이
            # 이 도로의 차도 폭 안에 있어야 신뢰한다 — 나란한 다른 도로의 차량이
            # 잘못 매칭된 것이면 방향 주장도 근거가 없다.
            span = max(road.lanes_forward, 1) * w
            inside = -span - 0.5 * w <= lat_off <= 0.5 * w
            # 합성 도로망은 폴리라인 방향이 관측 궤적에서 정해지므로 통행방향과
            # 반대일 수 있다 (실측 반전율 25%) — 확신하지 않는다.
            return False, inside and not road.inferred, "oneway"
        margin = w * side_frac
        if not road.inferred and abs(lat_off) >= margin:
            # 우측통행: 순방향 차로는 중심선 우측 = lat_off < 0.
            # 합성 도로망은 중심선이 참 중앙선이 아닐 수 있어 제외한다.
            rev = lat_off > 0 if self.lane_cfg.drive_side == "right" else lat_off < 0
            # **이 도로의 차도 안**에 있을 때만 통행측을 신뢰한다. CARLA 의 분리
            # 도로는 방향별로 별개 road 이므로, 기준선에서 차도 폭을 넘어 떨어진
            # 차량은 나란한 다른 도로에 속한다 — 실측 반전율이 |횡오프셋| 7m 이하
            # 2~12% 에서 7m 초과 54% 로 급증한다.
            lanes_side = road.lanes_backward if lat_off > 0 else road.lanes_forward
            if abs(lat_off) <= max(lanes_side, 1) * w:
                return rev, True, "lane_side"
        # 단서 없음 — 도로 정방향으로 두되 확신하지 않는다고 알린다
        return False, False, "default"

    def _lane_index(
        self, road: Road, travel_lat: float, lane_count: int, reverse: bool
    ) -> Optional[int]:
        """진행방향 기준 횡오프셋 → 차선 번호."""
        w = self.lane_cfg.lane_width_m
        if road.oneway:
            # 중심선이 차도 중앙 → 좌측 끝에서 세어 들어옴
            k_from_left = int((lane_count * w / 2.0 - travel_lat) // w) + 1
        else:
            # 우측통행: 진행차로는 중심선 우측 (travel_lat <= 0)
            if self.lane_cfg.drive_side == "right":
                k_from_left = int((-travel_lat) // w) + 1
            else:
                k_from_left = int(travel_lat // w) + 1
        k_from_left = min(max(k_from_left, 1), lane_count)
        if self.lane_cfg.numbering == "from_median":
            return k_from_left  # 1 = 중앙선쪽
        return lane_count - k_from_left + 1  # 1 = 가장자리쪽

    def _next_junction(
        self, road: Road, s: float, reverse: bool
    ) -> Tuple[Optional[str], Optional[float]]:
        if reverse:
            return (road.node_start or None, s)
        return (road.node_end or None, max(road.length_m - s, 0.0))

    # ------------------------------------------------------------ 하류 경로

    def downstream_paths(
        self,
        placement: RoadPlacement,
        horizon_m: float,
    ) -> List[Tuple[str, float, List[Tuple[float, float]], Optional[str]]]:
        """현재 위치에서 horizon_m 앞까지의 후보 경로.

        Returns: [(기동라벨, 확률, 폴리라인, 진입도로이름), ...]
        교차로에 도달하지 않으면 '직진' 하나만 반환한다.

        확률은 **기동 단위의 사전확률**이다. 한 기동으로 갈 수 있는 도로가 둘
        이상이면 그 기동의 몫을 도로 수로 나눈다. 도로마다 독립적으로 가중치를
        주면 교차로가 진출로를 여럿 열어 두었다는 이유만으로 그 기동이 과대
        평가된다 (오른쪽 진출로 2개 = 우회전 0.4 vs 좌회전 0.2).
        """
        road = self.roads.get(placement.road_id)
        if road is None:
            return []

        # 진행 방향은 **차량이 있는 지점의** 도로축과 비교해 정한다.
        # 폴리라인 첫 구간(bearing_at(0))과 비교하면, 곡선 도로에서 차량이 s 축을
        # 따라 멀리 있을 때 국소 방위가 첫 구간과 90° 넘게 달라 판정이 뒤집힌다
        # (실측 22%의 경로가 방위와 180° 반대로 나갔다). axis_bearing_deg 는
        # locate() 가 저장한 국소 도로축이다.
        axis = placement.axis_bearing_deg
        if not axis:  # 0.0 = 미기재 (구 데이터·직접 생성한 placement)
            axis = road.bearing_at(self._seg_at(road, placement.s_m))
        reverse = abs(wrap180(placement.bearing_deg - axis)) > 90.0
        # 중심선이 아니라 **차량이 달리는 차로**를 따라간다. 중심선을 그대로 쓰면
        # 첫 웨이포인트가 현재 위치에서 차로 폭만큼(실측 5~10m) 떨어져, 경로가
        # 엉뚱한 곳에서 시작하는 것처럼 보인다.
        lat = placement.lateral_offset_m
        along = self._sample_along(road, placement.s_m, horizon_m, reverse, lat)
        remaining = horizon_m - self._poly_len(along)
        jid = placement.next_junction_id

        if remaining <= 5.0 or jid is None:
            return [("직진", 1.0, along, road.name)]

        junction = self.junctions.get(jid)
        if junction is None or not junction.is_intersection:
            return [("직진", 1.0, along, road.name)]

        # 교차로에서 진출 가능한 도로별 기동 분류
        options: List[Tuple[str, float, List[Tuple[float, float]], Optional[str]]] = []
        seen_geom: set = set()
        for rid in junction.road_ids:
            if rid == road.road_id:
                continue
            nxt = self.roads[rid]
            out_reverse = nxt.node_end == jid  # 교차로가 끝점이면 역방향 진출
            out_bearing = nxt.bearing_at(len(nxt.poly) - 2 if out_reverse else 0)
            if out_reverse:
                out_bearing = (out_bearing + 180.0) % 360.0
            turn = wrap180(out_bearing - placement.bearing_deg)
            if abs(turn) < 30.0:
                label, weight = "직진", 0.6
            elif abs(turn) > 150.0:
                label, weight = "유턴", 0.02
            elif turn > 0:
                label, weight = "좌회전", 0.2
            else:
                label, weight = "우회전", 0.2
            start_s = nxt.length_m if out_reverse else 0.0
            # 진출로에서도 같은 횡오프셋을 유지한다. 어느 차로로 나갈지는 모르지만,
            # 우측통행에서 주행 차로는 중심선의 오른쪽(음수)이므로 중심선으로
            # 되돌리는 것보다 통행측을 지키는 편이 맞고 이음새도 끊기지 않는다.
            tail = self._sample_along(nxt, start_s, remaining, out_reverse, lat)
            poly = along + tail
            # 같은 기하를 두 번 내보내지 않는다. CARLA 교차로는 한 도로를
            # 차선구간별로 여러 road 로 쪼개 두기도 해서, 그대로 두면 완전히
            # 같은 경로가 중복 후보가 되고 그 기동의 확률만 부풀려진다.
            key = (label, tuple((round(x, 2), round(y, 2)) for x, y in poly))
            if key in seen_geom:
                continue
            seen_geom.add(key)
            options.append((label, weight, poly, nxt.name))

        if not options:
            return [("직진", 1.0, along, road.name)]

        # 차선 위치로 가중치 보정 (1차선=좌회전 유리, 최외측=우회전 유리)
        li, lc = placement.lane_index, placement.lane_count
        by_label: Dict[str, float] = {}
        for label, w, _poly, _nm in options:
            if li is not None and lc > 1:
                from_median = li if self.lane_cfg.numbering == "from_median" else lc - li + 1
                if label == "좌회전":
                    w *= 2.0 if from_median == 1 else 0.4
                elif label == "우회전":
                    w *= 2.0 if from_median == lc else 0.4
            by_label[label] = w  # 같은 라벨은 같은 가중치 — 덮어써도 무방

        n_by_label = Counter(label for label, _w, _p, _n in options)
        total = sum(by_label.values()) or 1.0
        out = [
            (label, by_label[label] / total / n_by_label[label], poly, name)
            for label, _w, poly, name in options
        ]
        out.sort(key=lambda x: -x[1])
        return out

    def _sample_along(
        self,
        road: Road,
        s0: float,
        length_m: float,
        reverse: bool,
        lateral_m: float = 0.0,
    ) -> List[Tuple[float, float]]:
        """도로를 따라 5m 간격으로 샘플링. lateral_m 만큼 횡방향 평행이동.

        `lateral_m` 은 **진행 방향 기준** 좌(+)/우(−) 오프셋
        (`RoadPlacement.lateral_offset_m` 규약)이다. 0 이면 중심선 그대로다.
        """
        step = 5.0
        pts: List[Tuple[float, float]] = []
        d = 0.0
        while d <= length_m:
            s = s0 - d if reverse else s0 + d
            if s < 0 or s > road.length_m:
                break
            p = road.point_at(s)
            if abs(lateral_m) > 1e-6:
                p = self._offset_point(road, s, reverse, lateral_m)
            pts.append(p)
            d += step
        return pts

    def _offset_point(
        self, road: Road, s: float, reverse: bool, lateral_m: float
    ) -> Tuple[float, float]:
        """s 지점을 진행 방향 기준 좌(+)/우(−)로 lateral_m 평행이동."""
        brg = road.bearing_at(self._seg_at(road, s))
        if reverse:
            brg = (brg + 180.0) % 360.0
        # 진행 방향 단위벡터 (e, n) = (sin, cos), 좌측 법선은 그것을 -90° 회전
        rad = math.radians(brg)
        fe, fn = math.sin(rad), math.cos(rad)
        le, ln = -fn, fe  # 좌측 법선
        px, py = road.point_at(s)
        return (px + le * lateral_m, py + ln * lateral_m)

    @staticmethod
    def _seg_at(road: Road, s: float) -> int:
        """s[m] 지점이 속한 폴리라인 구간 번호."""
        acc = 0.0
        for i in range(len(road.poly) - 1):
            L = math.dist(road.poly[i], road.poly[i + 1])
            if acc + L >= s:
                return i
            acc += L
        return max(len(road.poly) - 2, 0)

    @staticmethod
    def _poly_len(poly: List[Tuple[float, float]]) -> float:
        return sum(math.dist(poly[i], poly[i + 1]) for i in range(len(poly) - 1))
