"""기하 변환: 픽셀 ↔ 차량좌표 ↔ 월드(ENU) 좌표.

핵심 가정: 검출된 차량의 bbox 하단 중앙이 평평한 노면에 접한다 (flat-ground IPM).
경사로/요철에서는 오차가 커지므로 bbox 폭 기반 추정치와 가중 결합한다.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .config import CameraConfig, ClassSize
from .schemas import BBox

try:  # 정밀 좌표변환 (선택)
    from pyproj import Transformer

    _HAS_PYPROJ = True
except ImportError:  # pragma: no cover
    _HAS_PYPROJ = False

EARTH_R = 6378137.0


# ---------------------------------------------------------------- 각도 유틸


def wrap360(deg: float) -> float:
    """[0, 360) 으로 정규화. 부동소수 오차로 360.0 이 나오지 않도록 보정."""
    v = deg % 360.0
    return 0.0 if v >= 360.0 - 1e-9 else v


def wrap180(deg: float) -> float:
    """[-180, 180) 로 정규화. 방위각 차이 계산에 사용."""
    return (deg + 180.0) % 360.0 - 180.0


def heading_to_unit(heading_deg: float) -> Tuple[float, float]:
    """진북기준 시계방향 방위각 → ENU 단위벡터 (e, n)."""
    r = math.radians(heading_deg)
    return (math.sin(r), math.cos(r))


def unit_to_heading(e: float, n: float) -> float:
    return wrap360(math.degrees(math.atan2(e, n)))


def bearing_of_segment(p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
    """ENU 선분의 진행 방위각."""
    return unit_to_heading(p1[0] - p0[0], p1[1] - p0[1])


# ---------------------------------------------------------------- 지역 ENU 평면


class LocalENU:
    """위경도 ↔ 지역 평면(m) 변환. 원점 근방 수 km 범위에서 사용."""

    def __init__(self, origin_lat: float, origin_lon: float):
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self._fwd = None
        self._inv = None
        if _HAS_PYPROJ:
            # 원점 중심 azimuthal equidistant — 국지 영역에서 왜곡이 작다
            crs = (
                f"+proj=aeqd +lat_0={origin_lat} +lon_0={origin_lon} "
                "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
            )
            self._fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            self._inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def to_enu(self, lat: float, lon: float) -> Tuple[float, float]:
        if self._fwd is not None:
            e, n = self._fwd.transform(lon, lat)
            return (float(e), float(n))
        # 등거리 근사 (pyproj 미설치 시)
        n = math.radians(lat - self.origin_lat) * EARTH_R
        e = (
            math.radians(lon - self.origin_lon)
            * EARTH_R
            * math.cos(math.radians(self.origin_lat))
        )
        return (e, n)

    def to_geo(self, e: float, n: float) -> Tuple[float, float]:
        if self._inv is not None:
            lon, lat = self._inv.transform(e, n)
            return (float(lat), float(lon))
        lat = self.origin_lat + math.degrees(n / EARTH_R)
        lon = self.origin_lon + math.degrees(
            e / (EARTH_R * math.cos(math.radians(self.origin_lat)))
        )
        return (lat, lon)


# ---------------------------------------------------------------- 카메라 모델


class CameraModel:
    """카메라 1대의 투영/역투영. 관측자가 여러 대를 달면 카메라마다 하나씩 만든다."""

    def __init__(self, cfg: CameraConfig):
        if not (cfg.height_m > 0.0):
            # 지상고가 0이면 노면 교점이 원점으로 붕괴해 모든 거리 추정이
            # 0이 된다. 조용히 잘못된 값을 내지 않도록 여기서 막는다.
            raise ValueError(
                f"카메라 지상고가 양수여야 합니다 (height_m={cfg.height_m}). "
                "외부 파라미터에서 지상고를 잘못 복원했을 가능성이 큽니다."
            )
        if not (cfg.fx > 0.0 and cfg.fy > 0.0):
            raise ValueError(f"초점거리가 양수여야 합니다 (fx={cfg.fx}, fy={cfg.fy})")
        self.cfg = cfg
        self.K = np.array(
            [[cfg.fx, 0.0, cfg.cx], [0.0, cfg.fy, cfg.cy], [0.0, 0.0, 1.0]]
        )
        self.K_inv = np.linalg.inv(self.K)
        self.R = self._rotation()  # 카메라 좌표 → 차량 좌표

    def _rotation(self) -> np.ndarray:
        cfg = self.cfg
        # 기본 정렬: cam_z→veh_x(전방), cam_x→veh_-y(우측), cam_y→veh_-z(하방)
        R0 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        p, y, r = (
            math.radians(cfg.pitch_deg),
            math.radians(cfg.yaw_deg),
            math.radians(cfg.roll_deg),
        )
        # 차량 좌표계에서의 회전: yaw(z) * pitch(y) * roll(x)
        Rz = np.array(
            [[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]]
        )
        Ry = np.array(
            [[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]]
        )
        Rx = np.array(
            [[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]]
        )
        return Rz @ Ry @ Rx @ R0

    # ---- 역투영: 픽셀 → 노면 위 차량좌표

    def pixel_to_ground(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """노면(z = -height_m) 과의 교점. 지평선 위 픽셀이면 None."""
        d_cam = self.K_inv @ np.array([u, v, 1.0])
        d = self.R @ d_cam
        if d[2] >= -1e-6:  # 아래로 향하지 않음 → 노면과 만나지 않음
            return None
        s = -self.cfg.height_m / d[2]
        return (float(s * d[0]), float(s * d[1]))  # (x_fwd, y_left)

    def bbox_to_ego(
        self,
        bbox: BBox,
        cls: str,
        class_sizes: Dict[str, "ClassSize"],
    ) -> Tuple[Tuple[float, float], float]:
        """bbox → (차량좌표 (x_fwd, y_left), 신뢰도 0~1).

        **접지점 IPM 을 주 추정기로 쓴다.** DeepAccident 정답과 비교한 결과
        (n≈9900, 절단 bbox 제외) 상대오차 중앙값이 다음과 같았다:

            거리구간    IPM     높이기반   폭기반
            15-30m     -9.8%   -19.8%    -39.4%
            45-70m     -3.8%   -13.4%    -38.7%

        폭 기반 추정이 크게 나쁜 이유: 2D bbox 는 3D 박스 8개 코너의 축정렬
        헐이므로, 차량을 비스듬히 보면 겉보기 폭이 (길이+폭)/√2 까지 커진다.
        실폭으로 나누면 거리가 최대 2.5배 과소추정된다. 높이는 방위와 거의
        무관하므로 IPM 을 쓸 수 없을 때의 보조 단서로 적합하다.

        추가로 **근접면 → 중심 보정**을 적용한다. bbox 하단은 3D 박스에서
        카메라에 가장 가까운 접지 모서리이므로, 객체 중심은 시선 방향으로
        대략 발자국 반경만큼 더 멀다. 이 보정으로 15-70m 구간 편향이
        -9.8%/-3.8% → -5.2%/-0.8% 로 줄었다.
        """
        x1, y1, x2, y2 = bbox
        u_c, v_b = (x1 + x2) / 2.0, y2
        size = class_sizes.get(cls)
        w_real = size.width_m if size else 1.9
        l_real = size.length_m if size else 4.6
        h_real = size.height_m if size else 1.5
        # 방위 미상 시 접지 모서리에서 중심까지의 기대 거리
        near_face_offset = 0.25 * (w_real + l_real)

        h_px = max(y2 - y1, 1.0)
        d_height = self.cfg.fy * h_real / h_px

        cfgc = self.cfg
        truncated_bottom = y2 >= cfgc.height - 2
        truncated_side = x1 <= 1 or x2 >= cfgc.width - 1

        ground = None if truncated_bottom else self.pixel_to_ground(u_c, v_b)

        if ground is None:
            # 접지점을 쓸 수 없음 → 광선 방향 + 높이 기반 거리
            d_cam = self.K_inv @ np.array([u_c, (y1 + y2) / 2.0, 1.0])
            d = self.R @ d_cam
            horiz = math.hypot(d[0], d[1]) or 1.0
            q = 0.3 if not truncated_side else 0.2
            return ((d_height * d[0] / horiz, d_height * d[1] / horiz), q)

        d_ground = math.hypot(*ground) or 1e-6
        # 접지점이 유효하면 IPM 을 주로 쓰고, 높이 기반은 극단적 불일치를
        # 완화하는 정도로만 섞는다 (측면 절단 시 비중 상향).
        w_h = 0.25 if truncated_side else 0.1
        d_fused = (1.0 - w_h) * (d_ground + near_face_offset) + w_h * d_height
        scale = d_fused / d_ground
        quality = 0.55 if truncated_side else min(
            1.0, 0.95 / (1.0 + (d_ground / 90.0) ** 2)
        )
        return ((ground[0] * scale, ground[1] * scale), quality)

    # ---- 정투영 (검증/시각화/합성 데이터 생성용)

    def ego_to_pixel(
        self, x_fwd: float, y_left: float, z_up: float = 0.0
    ) -> Optional[Tuple[float, float]]:
        d_veh = np.array([x_fwd, y_left, z_up - self.cfg.height_m])
        d_cam = self.R.T @ d_veh
        if d_cam[2] <= 0.1:  # 카메라 뒤쪽
            return None
        p = self.K @ d_cam
        return (float(p[0] / p[2]), float(p[1] / p[2]))


# ---------------------------------------------------------------- 차량 ↔ 월드


def ego_to_world(
    ego_xy: Tuple[float, float],
    origin_enu: Tuple[float, float],
    heading_deg: float,
) -> Tuple[float, float]:
    """관측차량 기준 (x_fwd, y_left) → 월드 (e, n)."""
    fe, fn = heading_to_unit(heading_deg)
    le, ln = (-fn, fe)  # 전방벡터를 반시계 90도 회전 = 좌측
    x, y = ego_xy
    return (origin_enu[0] + x * fe + y * le, origin_enu[1] + x * fn + y * ln)


def world_to_ego(
    world_xy: Tuple[float, float],
    origin_enu: Tuple[float, float],
    heading_deg: float,
) -> Tuple[float, float]:
    """월드 (e, n) → 관측차량 기준 (x_fwd, y_left)."""
    fe, fn = heading_to_unit(heading_deg)
    le, ln = (-fn, fe)
    de, dn = world_xy[0] - origin_enu[0], world_xy[1] - origin_enu[1]
    return (de * fe + dn * fn, de * le + dn * ln)


def project_point_to_polyline(
    pt: Tuple[float, float], poly: list
) -> Tuple[float, float, float, int]:
    """점을 폴리라인에 정사영.

    Returns: (s, 횡오프셋(좌+), 수직거리, 세그먼트 인덱스)
    횡오프셋 부호는 폴리라인 진행방향 기준 왼쪽이 +.
    """
    best = (0.0, 0.0, float("inf"), 0)
    s_acc = 0.0
    for i in range(len(poly) - 1):
        a = np.array(poly[i], dtype=float)
        b = np.array(poly[i + 1], dtype=float)
        ab = b - a
        L = float(np.linalg.norm(ab))
        if L < 1e-9:
            continue
        t = float(np.dot(np.array(pt) - a, ab) / (L * L))
        t_c = min(max(t, 0.0), 1.0)
        proj = a + t_c * ab
        d = float(np.linalg.norm(np.array(pt) - proj))
        if d < best[2]:
            u = ab / L
            rel = np.array(pt) - proj
            # 좌측 법선 = u 를 반시계 90도 회전
            left = np.array([-u[1], u[0]])
            best = (s_acc + t_c * L, float(np.dot(rel, left)), d, i)
        s_acc += L
    return best
