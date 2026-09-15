"""DeepAccident 데이터셋 어댑터.

데이터셋: https://deepaccident.github.io/  (CARLA 기반 V2X 사고 예측 데이터셋)
참조 구현: https://github.com/tianqi-wang1996/DeepAccident
           (tools/data_converter/carla_converter.py 의 레이블 파싱 규약을 따름)

디렉터리 구조
    <root>/<scenario_type>/
        meta/<scenario>.txt
        <agent>/calib/<scenario>/<scenario>_NNN.pkl
        <agent>/label/<scenario>/<scenario>_NNN.txt
        <agent>/Camera_Front/<scenario>/<scenario>_NNN.jpg
        <agent>/lidar01/<scenario>/<scenario>_NNN.npz
    scenario_type: type1_subtype1_accident / _normal / type1_subtype2_accident / _normal
    agent: ego_vehicle, ego_vehicle_behind, other_vehicle, other_vehicle_behind,
           infrastructure

레이블 .txt (프레임당 1개)
    1행:  <ego_speed_x> <ego_speed_y>            ← **월드(CARLA) 프레임**
    이후: cls x y z l w h yaw vx vy id npts vis  ← 13개 필드
      - x y z    : 해당 센서의 **라이다 프레임** 박스 중심 [m]
      - l w h    : 길이/폭/높이 [m]
      - yaw      : 라이다 프레임 기준 방위 [rad]
      - vx vy    : **센서 프레임** 속도 [m/s]  (ego 자기행 id=-100 만 월드 프레임)
      - id       : CARLA actor id. -100=자기 자신, -1=추적 불가(원거리)
      - npts     : 박스 내 라이다 포인트 수
      - vis      : 6개 카메라 **전체** 기준 가시성 (전면 카메라 전용 아님)

calib .pkl
    ego_to_world (4x4), lidar_to_ego (4x4),
    intrinsic_<Camera> (3x3), lidar_to_<Camera> (4x4)
    world = ego_to_world @ lidar_to_ego @ p_lidar   (센서 간 일치 확인됨)

좌표계 변환
    CARLA 월드는 **좌수계** (x=동, y=남, z=상). 본 패키지의 ENU 는 우수계이므로
    e = x_carla, n = -y_carla 로 반사 변환한다. 반사이므로 회전 성분의 부호가
    바뀐다: yaw_enu = -yaw_carla, vy_enu = -vy_carla.
    (공식 컨버터도 같은 이유로 yaw 와 vy 를 음수화한다.)

내부 카메라 행렬은 축이 치환된 형태로 저장되어 있다:
    K = [[cx, f, 0], [cy, 0, -f], [1, 0, 0]]
    → u = cx + f*(y_right/depth),  v = cy - f*(z_up/depth)
  즉 카메라 프레임이 (x=깊이, y=우, z=상)이며, 광학적으로는 표준 핀홀과 동일해
  fx=fy=f, cx, cy 를 그대로 CameraConfig 에 넣으면 된다.
"""

from __future__ import annotations

import glob
import math
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import CameraConfig, ClassSize, DeepAccidentConfig
from .geometry import LocalENU, wrap360
from .perception import PerceptionBackend
from .schemas import BBox, CollisionTruth, Detection, EgoSample, ScenarioContext

# 데이터셋 클래스 → 본 패키지 클래스 (공식 CLASSES 순서: car truck van cyclist
# motorcycle pedestrian)
CLASS_MAP = {
    "car": "car",
    "truck": "truck",
    "van": "van",
    "bus": "bus",
    "cyclist": "bicycle",
    "motorcycle": "motorcycle",
    "pedestrian": "person",
}

# 클래스별 치수 [m] — 데이터셋 레이블 실측 중앙값 (n=250,378).
#   car 4.60×1.93×1.52, truck 8.47×2.89×3.83, van 4.48×1.99×2.04,
#   motorcycle 2.21×0.86×1.53, cyclist 1.64×0.37×1.77, pedestrian 0.38×0.38×1.86
CLASS_SIZES: Dict[str, ClassSize] = {
    "car": ClassSize(1.93, 4.60, 1.52),
    "truck": ClassSize(2.89, 8.47, 3.83),
    "van": ClassSize(1.99, 4.48, 2.04),
    "bus": ClassSize(2.90, 11.00, 3.20),
    "motorcycle": ClassSize(0.86, 2.21, 1.53),
    "bicycle": ClassSize(0.37, 1.64, 1.77),
    "person": ClassSize(0.38, 0.38, 1.86),
}

AGENT_ROLE_KO = {
    "ego_vehicle": "자차",
    "ego_vehicle_behind": "자차 후속차",
    "other_vehicle": "상대차",
    "other_vehicle_behind": "상대차 후속차",
    "infrastructure": "노변 인프라",
}

SELF_ID = -100
UNTRACKED_ID = -1


# ---------------------------------------------------------------- CARLA ↔ ENU


def carla_to_enu(x: float, y: float) -> Tuple[float, float]:
    """CARLA 월드(좌수계) → ENU(우수계). e=동, n=북."""
    return (x, -y)


def carla_matrix_to_enu_heading(R: np.ndarray) -> float:
    """CARLA 회전행렬 → ENU 방위각 [deg]. x축(전방)을 사용."""
    fwd = R[:, 0]
    e, n = carla_to_enu(float(fwd[0]), float(fwd[1]))
    return wrap360(math.degrees(math.atan2(e, n)))


def carla_yaw_to_enu_heading(yaw_rad: float) -> float:
    """CARLA yaw(rad, x축 기준 반시계) → ENU 방위각 [deg]."""
    # CARLA 좌수계에서 yaw 는 +x(동)에서 +y(남)으로 증가한다.
    e, n = carla_to_enu(math.cos(yaw_rad), math.sin(yaw_rad))
    return wrap360(math.degrees(math.atan2(e, n)))


# ---------------------------------------------------------------- 메타


@dataclass
class ScenarioMeta:
    """meta/<scenario>.txt 파싱 결과.

    1행: weather id1 cls1 id2 cls2 intensity spawn_relation collision_position n_sim
      사고가 없으면 weather 이후 필드가 모두 -1.
      n_sim 은 시뮬레이션 총 프레임(저장 프레임 + 워밍업 10)이다.
    """

    scenario: str = ""
    scenario_type: str = ""
    town: str = ""
    weather: str = ""
    is_accident_split: bool = False  # 디렉터리 이름이 *_accident 인가
    collision_occurred: bool = False  # 실제 충돌 기록이 있는가
    collision_id_a: Optional[int] = None
    collision_cls_a: Optional[str] = None
    collision_id_b: Optional[int] = None
    collision_cls_b: Optional[str] = None
    collision_intensity: Optional[float] = None
    spawn_relation: str = ""
    collision_position: str = ""
    n_simulated_frames: Optional[int] = None
    colliding_agents: Tuple[str, ...] = ()
    agent_ids: Tuple[int, ...] = ()
    road_type: str = ""
    another_vehicle_spawn_side: str = ""
    ego_vehicle_direction: str = ""
    other_vehicle_direction: str = ""
    n_frames: int = 0

    def agent_id_of(self, agent: str) -> Optional[int]:
        """에이전트 이름 → CARLA actor id. agent_ids 순서는 고정이다."""
        order = (
            "ego_vehicle",
            "ego_vehicle_behind",
            "other_vehicle",
            "other_vehicle_behind",
        )
        if agent not in order:
            return None
        i = order.index(agent)
        return self.agent_ids[i] if i < len(self.agent_ids) else None

    def to_context(self) -> ScenarioContext:
        """LLM 입력용/평가용 정보를 명시적으로 분리한 컨텍스트."""
        return ScenarioContext(
            scenario_id=f"{self.scenario_type}/{self.scenario}",
            source="DeepAccident",
            town=self.town,
            # 주행 시점에 알 수 있는 것만
            available={
                "weather": self.weather,
                "road_type": self.road_type,
            },
            # 사후에만 알 수 있는 정답 — 직렬화에서 기본 제외
            ground_truth={
                "is_accident_split": self.is_accident_split,
                "collision_occurred": self.collision_occurred,
                "colliding_agents": list(self.colliding_agents),
                "collision_ids": [self.collision_id_a, self.collision_id_b],
                "collision_intensity": self.collision_intensity,
                "collision_position": self.collision_position,
                "spawn_relation": self.spawn_relation,
                "ego_vehicle_direction": self.ego_vehicle_direction,
                "other_vehicle_direction": self.other_vehicle_direction,
                # 사고 분할에서는 기록이 충돌 시점에 끊기므로 마지막 프레임이
                # 충돌 시각에 해당한다.
                "collision_frame_approx": self.n_frames
                if (self.is_accident_split and self.collision_occurred)
                else None,
            },
        )


def _parse_int(tok: str) -> Optional[int]:
    """meta 의 id 필드 파싱. CARLA actor id 는 1부터이므로 0 이하는 미지 값."""
    try:
        v = int(tok)
    except ValueError:
        return None
    return None if v <= 0 else v


def parse_meta(path: str, scenario_type: str, n_frames: int = 0) -> ScenarioMeta:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [L.rstrip("\n") for L in f if L.strip()]

    scenario = os.path.splitext(os.path.basename(path))[0]
    m = ScenarioMeta(
        scenario=scenario,
        scenario_type=scenario_type,
        town=scenario.split("_")[0],
        is_accident_split=scenario_type.endswith("_accident"),
        n_frames=n_frames,
    )
    if not lines:
        return m

    h = lines[0].split()
    m.weather = h[0] if h else ""
    if len(h) >= 9:
        m.collision_id_a = _parse_int(h[1])
        m.collision_cls_a = h[2] if h[2] != "-1" else None
        m.collision_id_b = _parse_int(h[3])
        m.collision_cls_b = h[4] if h[4] != "-1" else None
        try:
            inten = float(h[5])
            m.collision_intensity = None if inten < 0 else inten
        except ValueError:
            m.collision_intensity = None
        m.spawn_relation = h[6] if h[6] != "-1" else ""
        m.collision_position = h[7] if h[7] != "-1" else ""
        try:
            m.n_simulated_frames = int(h[8])
        except ValueError:
            m.n_simulated_frames = None

    for L in lines[1:]:
        s = L.strip()
        if ":" not in s:
            continue
        key, val = s.split(":", 1)
        key, val = key.strip(), val.strip()
        if key == "colliding agents":
            m.colliding_agents = tuple(
                v for v in val.split() if v and v != "none"
            )
        elif key == "agents id":
            ids = []
            for v in val.split():
                try:
                    ids.append(int(v))
                except ValueError:
                    pass
            m.agent_ids = tuple(ids)
        elif key == "road_type":
            m.road_type = val
        elif key == "another_vehicle_spawn_side":
            m.another_vehicle_spawn_side = val
        elif key == "ego_vehicle_direction":
            m.ego_vehicle_direction = val
        elif key == "other_vehicle_direction":
            m.other_vehicle_direction = val

    m.collision_occurred = bool(m.colliding_agents) and m.collision_id_a is not None
    return m


# ---------------------------------------------------------------- 레이블 / calib


@dataclass
class LabelObject:
    """레이블 한 행. 좌표는 해당 센서의 라이다 프레임."""

    cls_raw: str
    cls: str
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float  # 라이다 프레임, rad
    vx: float
    vy: float
    obj_id: int
    n_lidar_pts: int
    camera_visible: bool

    @property
    def is_self(self) -> bool:
        return self.obj_id == SELF_ID

    @property
    def is_untracked(self) -> bool:
        return self.obj_id == UNTRACKED_ID

    @property
    def range_m(self) -> float:
        return math.hypot(self.x, self.y)


@dataclass
class LabelFrame:
    ego_speed_world: Tuple[float, float]  # 월드(CARLA) 프레임 자기 속도
    objects: List[LabelObject] = field(default_factory=list)

    @property
    def self_object(self) -> Optional[LabelObject]:
        for o in self.objects:
            if o.is_self:
                return o
        return None


def parse_label_file(path: str) -> LabelFrame:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [L.rstrip("\n") for L in f]
    if not lines:
        return LabelFrame((0.0, 0.0))

    hdr = lines[0].split()
    ego_v = (
        (float(hdr[0]), float(hdr[1])) if len(hdr) >= 2 else (0.0, 0.0)
    )
    objs: List[LabelObject] = []
    for L in lines[1:]:
        t = L.split()
        if len(t) < 13:
            continue
        raw = t[0]
        objs.append(
            LabelObject(
                cls_raw=raw,
                cls=CLASS_MAP.get(raw, raw),
                x=float(t[1]),
                y=float(t[2]),
                z=float(t[3]),
                length=float(t[4]),
                width=float(t[5]),
                height=float(t[6]),
                yaw=float(t[7]),
                vx=float(t[8]),
                vy=float(t[9]),
                obj_id=int(t[-3]),
                n_lidar_pts=int(t[-2]),
                camera_visible=(t[-1] == "True"),
            )
        )
    return LabelFrame(ego_v, objs)


def load_calib(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- 카메라 설정


def available_cameras(calib: dict) -> List[str]:
    """calib 에 내·외부 파라미터가 **둘 다** 있는 카메라 이름 (정렬).

    DeepAccident 는 관측자마다 6대(Front, FrontLeft, FrontRight, Back,
    BackLeft, BackRight)를 달고 yaw 0/±55/±110/180° 로 전방위를 덮는다.
    한쪽만 있는 카메라는 쓸 수 없으므로 교집합을 취한다.
    """
    intr = {k[len("intrinsic_") :] for k in calib if k.startswith("intrinsic_")}
    extr = {k[len("lidar_to_") :] for k in calib if k.startswith("lidar_to_")}
    return sorted(intr & extr)


def resolve_cameras(
    calib: dict, requested: Optional[Sequence[str]] = None
) -> List[str]:
    """쓸 카메라 목록을 정한다. None 이면 있는 것 전부.

    요청한 이름 중 calib 에 없는 것은 조용히 버리지 않는다 — 오타 하나로
    카메라가 통째로 빠지면 "측방 차량이 원래 안 보인다"는 잘못된 결론에
    이르기 때문이다.
    """
    have = available_cameras(calib)
    if requested is None:
        return have
    names = list(requested)
    missing = [n for n in names if n not in have]
    if missing:
        raise ValueError(
            f"calib 에 없는 카메라를 요청했습니다: {missing} (존재: {have})"
        )
    return [n for n in have if n in set(names)]


def camera_world_z(calib: dict, camera: str = "Camera_Front") -> float:
    """카메라 광학중심의 CARLA 월드 z [m]."""
    M = np.asarray(calib[f"lidar_to_{camera}"], dtype=float)
    R, t = M[:3, :3], M[:3, 3]
    origin_lidar = -R.T @ t  # 카메라 원점(라이다 좌표) = -Rᵀt
    L2E = np.asarray(calib["lidar_to_ego"], dtype=float)
    E2W = np.asarray(calib["ego_to_world"], dtype=float)
    p_ego = L2E @ np.array([*origin_lidar, 1.0])
    p_world = E2W @ p_ego
    return float(p_world[2])


def camera_config_from_calib(
    calib: dict,
    camera: str = "Camera_Front",
    image_size: Tuple[int, int] = (1600, 900),
    ground_z: float = 0.0,
    name: Optional[str] = None,
) -> CameraConfig:
    """calib 의 내부/외부 파라미터 → 본 패키지 CameraConfig.

    저장된 K 는 축이 치환되어 있다(모듈 docstring 참조):
        f = K[0,1], cx = K[0,0], cy = K[1,0]

    지상고는 **월드 좌표 z 에서 노면 z 를 뺀 값**으로 구한다. 센서 종류에
    따라 높이가 실려 있는 행렬이 다르기 때문이다:
      - 차량   : ego 원점이 노면이고 높이는 lidar_to_ego 에 들어 있다
      - 인프라 : lidar_to_ego 가 항등행렬이고 높이는 ego_to_world 에 들어 있다
    lidar_to_ego 만 보면 인프라 지상고가 0 이 되어 노면 교점이 원점으로
    붕괴하므로, 반드시 전체 변환 사슬을 통과시켜야 한다.

    CARLA 는 y=우측이므로 yaw 부호를 반전해 본 패키지 규약(y=좌측)에 맞춘다.
    """
    K = np.asarray(calib[f"intrinsic_{camera}"], dtype=float)
    f = abs(float(K[0, 1]))
    cx, cy = float(K[0, 0]), float(K[1, 0])

    M = np.asarray(calib[f"lidar_to_{camera}"], dtype=float)
    R = M[:3, :3]
    fwd = R[0, :]  # 카메라 광축(라이다 좌표) = R 의 첫 행

    height = camera_world_z(calib, camera) - ground_z

    yaw = math.degrees(math.atan2(-float(fwd[1]), float(fwd[0])))
    norm = float(np.linalg.norm(fwd)) or 1.0
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(fwd[2]) / norm))))

    return CameraConfig(
        width=image_size[0],
        height=image_size[1],
        fx=f,
        fy=f,
        cx=cx,
        cy=cy,
        height_m=height,
        pitch_deg=pitch,
        yaw_deg=yaw,
        roll_deg=0.0,
        # 검출의 `Detection.camera` 와 짝지어 역투영 파라미터를 고르는 열쇠.
        # 기본값을 calib 키 그대로 두어 다중 카메라에서 이름이 겹치지 않는다.
        name=name if name is not None else camera,
    )


# ---------------------------------------------------------------- 시나리오


def estimate_ground_z(
    calib: dict,
    label: LabelFrame,
    default: float = 0.0,
    local_radius_m: float = 50.0,
) -> float:
    """센서 아래 노면의 CARLA 월드 z 를 추정.

    타운마다 고도가 다르고(mini 데이터에서 Town03 은 3~12m) 한 장면 안에서도
    기복이 있으므로, 노면을 0 으로 가정하면 지상고가 크게 틀린다.

    두 경로를 쓴다.
      - **차량**: 자기 자신 행(id=-100)의 박스 밑면. mini 데이터 전 타운에서
        `ego_to_world[2,3]` 과 1cm 이내로 일치함을 확인했다(= ego 원점이 노면).
      - **인프라**: 자기 행이 없으므로 `local_radius_m` 안의 객체 박스 밑면
        중앙값. 전 범위 중앙값을 쓰면 다른 고도의 원거리 객체에 끌려간다
        (Town03 에서 3.84 vs 실제 7.99).
    """
    L2W = np.asarray(calib["ego_to_world"], dtype=float) @ np.asarray(
        calib["lidar_to_ego"], dtype=float
    )

    def bottom(o: LabelObject) -> float:
        p = L2W @ np.array([o.x, o.y, o.z, 1.0])
        return float(p[2]) - o.height / 2.0

    self_obj = label.self_object
    if self_obj is not None and self_obj.height > 0.1:
        return bottom(self_obj)

    near = [
        bottom(o)
        for o in label.objects
        if o.height > 0.1 and o.range_m <= local_radius_m
    ]
    if near:
        return float(np.median(near))
    allb = [bottom(o) for o in label.objects if o.height > 0.1]
    if allb:
        return float(np.median(allb))
    # 최후: ego 원점 높이 (차량이면 노면과 같다)
    return float(np.asarray(calib["ego_to_world"], dtype=float)[2, 3]) or default


def estimate_ground_z_multiframe(
    scenario: "DeepAccidentScenario",
    agent: str,
    frames: Optional[Sequence[int]] = None,
    max_frames: int = 24,
    local_radius_m: float = 50.0,
) -> float:
    """여러 프레임을 모아 노면 z 를 추정 (정지 센서용).

    인프라는 자기 행(id=-100)이 없어 근거리 객체 박스 밑면에 의존하는데,
    한 프레임에 근거리 객체가 몇 개 없으면(기복 있는 지형에서 특히) 중앙값이
    흔들린다. mini 데이터의 Town07 인프라는 단일 프레임 추정에서 지상고가
    1.86m 로 나왔으나(다른 타운은 3.4~3.8m), 전 프레임을 모으면 정상화된다.
    """
    series = scenario.agents[agent]
    fr = list(frames) if frames is not None else series.frames
    if len(fr) > max_frames:
        step = max(len(fr) // max_frames, 1)
        fr = fr[::step][:max_frames]

    bottoms: List[float] = []
    for f in fr:
        cal = load_calib(series.calib_paths[f])
        lf = parse_label_file(series.label_paths[f])
        self_obj = lf.self_object
        if self_obj is not None and self_obj.height > 0.1:
            # 차량이면 자기 행 하나로 충분하다
            return estimate_ground_z(cal, lf, local_radius_m=local_radius_m)
        L2W = np.asarray(cal["ego_to_world"], dtype=float) @ np.asarray(
            cal["lidar_to_ego"], dtype=float
        )
        for o in lf.objects:
            if o.height <= 0.1 or o.range_m > local_radius_m:
                continue
            p = L2W @ np.array([o.x, o.y, o.z, 1.0])
            bottoms.append(float(p[2]) - o.height / 2.0)
    if bottoms:
        return float(np.median(bottoms))
    cal = load_calib(series.calib_paths[fr[0]])
    return estimate_ground_z(cal, parse_label_file(series.label_paths[fr[0]]))


@dataclass
class AgentSeries:
    """한 에이전트의 프레임별 calib/label 경로와 파생 상태."""

    agent: str
    frames: List[int]
    calib_paths: Dict[int, str]
    label_paths: Dict[int, str]
    image_paths: Dict[int, str] = field(default_factory=dict)
    is_static: bool = False

    @property
    def role(self) -> str:
        return "infrastructure" if self.agent == "infrastructure" else "vehicle"


@dataclass
class DeepAccidentScenario:
    """하나의 (scenario_type, scenario) 단위."""

    root: str
    scenario_type: str
    scenario: str
    meta: ScenarioMeta
    agents: Dict[str, AgentSeries]

    @property
    def scenario_id(self) -> str:
        return f"{self.scenario_type}/{self.scenario}"

    @property
    def town(self) -> str:
        return self.meta.town

    @property
    def split(self) -> str:
        """이 시나리오가 속한 분할 이름 (`DeepAccident_mini` / `train` / `val`).

        `root` 는 스캔 단위(분할 디렉터리)를 가리키므로 그 이름이 곧 분할이다.
        분할 상위 디렉터리를 루트로 줘도(607개 한꺼번에 스캔) 시나리오마다 어느
        분할에서 왔는지 남는다 — 산출물 경로와 manifest 에 이 값을 쓴다.
        """
        return os.path.basename(os.path.normpath(self.root))

    @property
    def outcome(self) -> str:
        """`accident` 또는 `normal`. 시나리오 유형 이름의 접미에서 읽는다."""
        return "accident" if self.scenario_type.endswith("_accident") else "normal"

    def frames(self) -> List[int]:
        """모든 에이전트가 데이터를 가진 프레임 (교집합)."""
        sets = [set(a.frames) for a in self.agents.values() if a.frames]
        if not sets:
            return []
        common = set.intersection(*sets)
        return sorted(common)


def _frame_index(path: str) -> Optional[int]:
    stem = os.path.splitext(os.path.basename(path))[0]
    tail = stem.rsplit("_", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return None


def _has_scenario_types(path: str) -> bool:
    """이 디렉터리가 `<유형>/meta/*.txt` 를 직접 담고 있는가."""
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    for st in entries:
        if glob.glob(os.path.join(path, st, "meta", "*.txt")):
            return True
    return False


def scan_scenarios(
    root: str,
    cfg: Optional[DeepAccidentConfig] = None,
    scenario_types: Optional[Sequence[str]] = None,
    towns: Optional[Sequence[str]] = None,
) -> List[DeepAccidentScenario]:
    """데이터셋 루트를 스캔해 시나리오 목록을 만든다.

    루트는 두 가지 배치를 모두 받는다.

      A) 분할 디렉터리를 직접 가리킴 — `<root>/<시나리오유형>/meta/*.txt`
         (`DeepAccident_mini`, `train`, `val` 각각을 루트로 줄 때)
      B) 분할들의 **상위** 디렉터리 — `<root>/<분할>/<시나리오유형>/meta/*.txt`
         (`.../DeepAccident` 를 루트로 주면 mini·train·val 을 한꺼번에 스캔)

    B 를 지원하는 이유: 실제 배포 데이터가 `DeepAccident/{DeepAccident_mini,
    train,val,test}` 로 나뉘어 있어, 상위 경로를 그대로 주는 것이 자연스럽다.
    지원하지 않으면 조용히 0개가 나와 데이터가 없는 것처럼 보인다.
    """
    cfg = cfg or DeepAccidentConfig()
    if not os.path.isdir(root):
        raise FileNotFoundError(f"데이터셋 루트가 없습니다: {root}")

    # 배치 A 가 아니면 한 단계 내려가 각 분할을 스캔한다 (배치 B).
    if not _has_scenario_types(root):
        nested: List[DeepAccidentScenario] = []
        for d in sorted(os.listdir(root)):
            sub = os.path.join(root, d)
            if os.path.isdir(sub) and _has_scenario_types(sub):
                nested.extend(
                    scan_scenarios(sub, cfg, scenario_types, towns)
                )
        return nested

    wanted_agents = list(cfg.agents)
    if cfg.use_infrastructure:
        wanted_agents.append("infrastructure")

    out: List[DeepAccidentScenario] = []
    for st in sorted(os.listdir(root)):
        st_dir = os.path.join(root, st)
        if not os.path.isdir(st_dir):
            continue
        if scenario_types and st not in scenario_types:
            continue
        meta_dir = os.path.join(st_dir, "meta")
        if not os.path.isdir(meta_dir):
            continue

        for mf in sorted(glob.glob(os.path.join(meta_dir, "*.txt"))):
            sc = os.path.splitext(os.path.basename(mf))[0]
            if towns and sc.split("_")[0] not in towns:
                continue

            agents: Dict[str, AgentSeries] = {}
            for ag in wanted_agents:
                cal_dir = os.path.join(st_dir, ag, "calib", sc)
                lab_dir = os.path.join(st_dir, ag, "label", sc)
                if not (os.path.isdir(cal_dir) and os.path.isdir(lab_dir)):
                    continue
                cps, lps = {}, {}
                for p in glob.glob(os.path.join(cal_dir, "*.pkl")):
                    i = _frame_index(p)
                    if i is not None:
                        cps[i] = p
                for p in glob.glob(os.path.join(lab_dir, "*.txt")):
                    i = _frame_index(p)
                    if i is not None:
                        lps[i] = p
                frames = sorted(set(cps) & set(lps))
                if not frames:
                    continue
                img_dir = os.path.join(st_dir, ag, cfg.camera, sc)
                ips = {}
                if os.path.isdir(img_dir):
                    for p in glob.glob(os.path.join(img_dir, "*")):
                        i = _frame_index(p)
                        if i is not None:
                            ips[i] = p
                agents[ag] = AgentSeries(
                    agent=ag,
                    frames=frames,
                    calib_paths=cps,
                    label_paths=lps,
                    image_paths=ips,
                    is_static=(ag == "infrastructure"),
                )

            if not agents:
                continue
            n_frames = max(len(a.frames) for a in agents.values())
            meta = parse_meta(mf, st, n_frames=n_frames)
            out.append(
                DeepAccidentScenario(
                    root=root,
                    scenario_type=st,
                    scenario=sc,
                    meta=meta,
                    agents=agents,
                )
            )
    return out


def find_scenario(
    root: str,
    scenario: str,
    scenario_type: Optional[str] = None,
    cfg: Optional[DeepAccidentConfig] = None,
) -> DeepAccidentScenario:
    """이름으로 시나리오 하나를 찾는다. 부분 문자열 매칭을 허용한다."""
    cands = scan_scenarios(root, cfg)
    matches = [
        s
        for s in cands
        if (scenario == s.scenario or scenario in s.scenario)
        and (scenario_type is None or s.scenario_type == scenario_type)
    ]
    if not matches:
        names = sorted({f"{s.scenario_type}/{s.scenario}" for s in cands})
        raise KeyError(
            f"시나리오를 찾을 수 없습니다: {scenario!r} "
            f"(type={scenario_type!r})\n사용 가능: {names[:10]}"
        )
    if len(matches) > 1 and scenario_type is None:
        raise KeyError(
            f"시나리오 {scenario!r} 가 여러 분할에 있습니다. scenario_type 을 "
            f"지정하십시오: {[m.scenario_type for m in matches]}"
        )
    return matches[0]


# ---------------------------------------------------------------- 자세/궤적


@dataclass
class SensorPose:
    """한 프레임의 센서 자세 (ENU)."""

    frame: int
    t: float
    world_xy: Tuple[float, float]  # ENU
    world_z: float
    heading_deg: float
    speed_mps: float
    ego_to_world: np.ndarray
    lidar_to_ego: np.ndarray
    # 자기 레이블 행(id=-100)의 **박스 중심** ENU 위치. ego_to_world 의 평행이동
    # 성분은 차량 기준점이고 박스 중심과 최대 0.3m 어긋난다 — 다른 모든 액터는
    # 박스 중심으로 표현되므로, 관측차량도 같은 규약을 써야 거리·TTC·조감도
    # 사각형이 일관된다. 정지 센서(인프라)에는 자기 행이 없어 None 이다.
    box_center_xy: Optional[Tuple[float, float]] = None

    @property
    def lidar_to_world(self) -> np.ndarray:
        return self.ego_to_world @ self.lidar_to_ego


def read_poses(
    scenario: DeepAccidentScenario,
    agent: str,
    cfg: Optional[DeepAccidentConfig] = None,
    frames: Optional[Sequence[int]] = None,
) -> Dict[int, SensorPose]:
    """에이전트의 프레임별 자세를 읽는다.

    속도는 레이블 1행(월드 프레임 자기 속도)에서 얻고, 없으면 위치 미분한다.
    인프라(정지 센서)는 항상 0 이다.
    """
    cfg = cfg or DeepAccidentConfig()
    series = scenario.agents[agent]
    fr = list(frames) if frames is not None else series.frames
    dt = 1.0 / cfg.frame_rate_hz

    poses: Dict[int, SensorPose] = {}
    for f in fr:
        cal = load_calib(series.calib_paths[f])
        E2W = np.asarray(cal["ego_to_world"], dtype=float)
        L2E = np.asarray(cal["lidar_to_ego"], dtype=float)
        e, n = carla_to_enu(float(E2W[0, 3]), float(E2W[1, 3]))
        heading = carla_matrix_to_enu_heading(E2W[:3, :3])

        box_center = None
        if series.is_static:
            speed = 0.0
        else:
            lf = parse_label_file(series.label_paths[f])
            speed = math.hypot(*lf.ego_speed_world)
            L2W = E2W @ L2E
            for o in lf.objects:
                if o.is_self:
                    wp = L2W @ np.array([o.x, o.y, o.z, 1.0])
                    box_center = carla_to_enu(float(wp[0]), float(wp[1]))
                    break

        poses[f] = SensorPose(
            frame=f,
            t=(f - 1) * dt,
            world_xy=(e, n),
            world_z=float(E2W[2, 3]),
            heading_deg=heading,
            speed_mps=speed,
            ego_to_world=E2W,
            lidar_to_ego=L2E,
            box_center_xy=box_center,
        )
    return poses


def synthesize_telemetry(
    scenario: DeepAccidentScenario,
    agent: str,
    cfg: Optional[DeepAccidentConfig] = None,
    enu: Optional[LocalENU] = None,
    frames: Optional[Sequence[int]] = None,
) -> List[EgoSample]:
    """GPS/IMU 텔레메트리를 데이터셋 자세로부터 합성한다.

    DeepAccident 에는 GPS 채널이 없으므로 `ego_to_world` 를 GPS+나침반 상당으로
    변환한다. cfg.gps_noise_m / heading_noise_deg 가 0보다 크면 잡음을 주입해
    실제 GPS 오차 조건을 재현한다 (재현성을 위해 random_seed 사용).

    위치는 자기 레이블 행의 **박스 중심**을 쓴다. `ego_to_world` 평행이동은
    차량 기준점이어서 박스 중심과 최대 0.3m 어긋나는데, 다른 액터는 모두 박스
    중심으로 표현되므로 그대로 두면 관측차량만 규약이 달라진다.
    """
    cfg = cfg or DeepAccidentConfig()
    enu = enu or LocalENU(*cfg.geo_origin)
    poses = read_poses(scenario, agent, cfg, frames)
    rng = random.Random(f"{cfg.random_seed}:{scenario.scenario_id}:{agent}")

    out: List[EgoSample] = []
    prev_heading = None
    prev_t = None
    for f in sorted(poses):
        p = poses[f]
        # 관측차량 위치는 **박스 중심**으로 보고한다 (다른 모든 액터와 같은 규약).
        e, n = p.box_center_xy or p.world_xy
        if cfg.gps_noise_m > 0:
            e += rng.gauss(0.0, cfg.gps_noise_m)
            n += rng.gauss(0.0, cfg.gps_noise_m)
        heading = p.heading_deg
        if cfg.heading_noise_deg > 0:
            heading = wrap360(heading + rng.gauss(0.0, cfg.heading_noise_deg))
        lat, lon = enu.to_geo(e, n)

        yaw_rate = 0.0
        if prev_heading is not None and prev_t is not None and p.t > prev_t:
            d = (heading - prev_heading + 180.0) % 360.0 - 180.0
            yaw_rate = d / (p.t - prev_t)
        prev_heading, prev_t = heading, p.t

        out.append(
            EgoSample(
                t=p.t,
                lat=lat,
                lon=lon,
                heading_deg=heading,
                speed_mps=p.speed_mps,
                yaw_rate_dps=yaw_rate,
                alt=p.world_z,
            )
        )
    return out


def write_telemetry_csv(samples: Iterable[EgoSample], path: str) -> None:
    """합성 텔레메트리를 파이프라인 CSV 형식으로 저장."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("t,lat,lon,heading_deg,speed_mps,yaw_rate_dps,alt\n")
        for s in samples:
            f.write(
                f"{s.t:.4f},{s.lat:.9f},{s.lon:.9f},{s.heading_deg:.4f},"
                f"{s.speed_mps:.4f},{s.yaw_rate_dps:.4f},{s.alt:.3f}\n"
            )


# ---------------------------------------------------------------- 관측 생성


def _box_corners_lidar(o: LabelObject) -> np.ndarray:
    """라이다 프레임 3D 박스 8개 코너 (4x8 동차좌표)."""
    l2, w2, h2 = o.length / 2.0, o.width / 2.0, o.height / 2.0
    xs = np.array([l2, l2, l2, l2, -l2, -l2, -l2, -l2])
    ys = np.array([w2, w2, -w2, -w2, w2, w2, -w2, -w2])
    zs = np.array([h2, -h2, h2, -h2, h2, -h2, h2, -h2])
    c, s = math.cos(o.yaw), math.sin(o.yaw)
    X = o.x + c * xs - s * ys
    Y = o.y + s * xs + c * ys
    Z = o.z + zs
    return np.vstack([X, Y, Z, np.ones_like(X)])


class DeepAccidentPerception(PerceptionBackend):
    """DeepAccident 레이블 → Detection 변환 백엔드.

    observation_mode:
      'camera'   — 3D 박스를 카메라에 투영해 2D bbox 생성. 파이프라인의 단안
                   역투영이 그대로 동작하므로 실제 배포 경로를 재현한다.
      'sensor3d' — GT 3D 위치/방위/속도를 Detection 에 실어 보낸다. 단안 역투영
                   을 건너뛰므로 지도매칭·융합·상호작용 계층만 평가할 때 쓴다.

    두 모드 모두 **카메라마다** 절두체 컬링과 가려짐 판정을 하고, 한 대라도 본
    객체를 남긴다. 카메라별로 화상면과 가려짐 관계가 다르므로 격자도 따로
    쓴다 — 전방 카메라에서 앞차에 가려진 차량이 측방 카메라에는 드러난다.

    데이터셋의 `vis` 플래그는 6개 카메라 OR 이다. 6대를 모두 쓰면 이 플래그의
    범위와 우리 판정 범위가 일치하므로 정확한 관문이 되고, 일부만 쓰면
    필요조건으로만 유효하다.
    """

    def __init__(
        self,
        scenario: DeepAccidentScenario,
        cfg: Optional[DeepAccidentConfig] = None,
    ):
        self.scenario = scenario
        self.cfg = cfg or DeepAccidentConfig()
        # 통계 — 왜 관측이 줄었는지 설명할 수 있어야 한다.
        # 판정 순서: 자기/추적불가/클래스/사거리 → 기하 컬링 → 가시성/가려짐
        # (기하 컬링을 먼저 해야 'occluded' 가 "화면 안에서 가려짐"을 뜻한다)
        #
        # `dropped_*` 중 속성 필터(self/untracked/class/range/lidar_pts/
        # not_visible_any_cam)는 **객체당 1회**, 기하 필터(behind/outside_fov/
        # small_bbox/occluded)는 **(객체, 카메라) 쌍당 1회** 센다. 카메라가
        # k대면 한 객체가 기하 필터에서 최대 k번 탈락하므로, 객체 단위로 읽어야
        # 하는 값은 `dropped_no_camera`(어느 카메라도 보지 못한 객체 수)다.
        self.stats: Dict[str, int] = {
            "total_rows": 0,
            "kept": 0,
            "n_cameras": 0,
            "dropped_self": 0,
            "dropped_untracked": 0,
            "dropped_class": 0,
            "dropped_range": 0,
            "dropped_behind": 0,
            "dropped_outside_fov": 0,
            "dropped_small_bbox": 0,
            "dropped_not_visible_any_cam": 0,
            "dropped_lidar_pts": 0,
            "dropped_occluded": 0,
            "dropped_no_camera": 0,
            # 기준(전면) 카메라만 썼다면 놓쳤을 관측 수 — 다중 카메라의 이득
            "seen_only_by_non_primary": 0,
        }
        # 카메라별 기여 관측 수
        self.per_camera: Dict[str, int] = {}

    # PerceptionBackend 인터페이스 — video_path 는 "<agent>" 문자열로 쓴다
    def run(self, video_path: str, t0: float) -> List[Detection]:
        return self.detections_for(video_path, t0=t0)

    def detections_for(
        self,
        agent: str,
        t0: float = 0.0,
        frames: Optional[Sequence[int]] = None,
    ) -> List[Detection]:
        cfg = self.cfg
        series = self.scenario.agents[agent]
        fr = list(frames) if frames is not None else series.frames
        dt = 1.0 / cfg.frame_rate_hz
        keep_classes = set(CLASS_SIZES)
        if not cfg.include_pedestrians:
            keep_classes.discard("person")

        out: List[Detection] = []
        for f in fr:
            cal = load_calib(series.calib_paths[f])
            lf = parse_label_file(series.label_paths[f])
            cams = resolve_cameras(cal, cfg.cameras)
            if not cams:
                raise ValueError(
                    f"{agent}: calib 에서 쓸 카메라를 찾지 못했습니다 "
                    f"(요청 {cfg.cameras}, 존재 {available_cameras(cal)})"
                )
            self.stats["n_cameras"] = len(cams)
            primary = cfg.camera if cfg.camera in cams else cams[0]
            L2W = np.asarray(cal["ego_to_world"], dtype=float) @ np.asarray(
                cal["lidar_to_ego"], dtype=float
            )
            img_w = 1600
            img_h = 900
            t = t0 + (f - 1) * dt

            # --- 1단계: 속성 필터 (카메라와 무관하므로 객체당 한 번만)
            objs: List[LabelObject] = []
            for o in lf.objects:
                self.stats["total_rows"] += 1
                if o.is_self:
                    self.stats["dropped_self"] += 1
                    continue
                if o.is_untracked and not cfg.include_untracked:
                    self.stats["dropped_untracked"] += 1
                    continue
                if o.cls not in keep_classes:
                    self.stats["dropped_class"] += 1
                    continue
                if o.range_m > cfg.max_range_m:
                    self.stats["dropped_range"] += 1
                    continue
                # 어느 카메라에도 안 보인다고 데이터셋이 표시한 객체
                if cfg.require_camera_visibility and not o.camera_visible:
                    self.stats["dropped_not_visible_any_cam"] += 1
                    continue
                if o.n_lidar_pts < cfg.min_lidar_pts:
                    self.stats["dropped_lidar_pts"] += 1
                    continue
                objs.append(o)

            # --- 2단계: 카메라별 투영 + 가려짐 판정
            # 카메라마다 화상면과 가려짐 관계가 다르므로 격자도 따로 쓴다 —
            # 전방 카메라에서 앞차에 가려진 차량이 측방 카메라에는 드러난다.
            best: Dict[int, Tuple[float, str, BBox]] = {}
            for cam in cams:
                K = np.asarray(cal[f"intrinsic_{cam}"], dtype=float)
                L2C = np.asarray(cal[f"lidar_to_{cam}"], dtype=float)
                projected = self._project_to_camera(objs, K, L2C, img_w, img_h, cfg)
                for free, o, bbox in self._unoccluded(projected, img_w, img_h, cfg):
                    key = id(o)
                    prev = best.get(key)
                    # 여러 대가 본 객체는 **가장 잘 보이는** 카메라의 bbox 를
                    # 쓴다. 화면 끝에서 절단된 bbox 보다 온전히 담긴 bbox 가
                    # 단안 역투영 정확도가 높다.
                    if prev is None or free > prev[0]:
                        best[key] = (free, cam, bbox)

            # --- 3단계: Detection 생성 (객체당 하나)
            by_key = {id(o): o for o in objs}
            for key, (free, cam, bbox) in best.items():
                o = by_key[key]
                det = Detection(
                    t=t,
                    frame_idx=f,
                    track_id=self._stable_track_id(o, f),
                    cls=o.cls,
                    conf=round(min(1.0, 0.5 + 0.5 * free), 3),
                    bbox=bbox,
                    n_lidar_pts=o.n_lidar_pts,
                    camera=cam,
                )
                if cfg.observation_mode == "sensor3d":
                    det.ego_xy_measured = (o.x, -o.y)  # CARLA y=우 → 내부 y=좌
                    # 절대 월드 좌표는 **전체 4x4 변환**으로 구한다. 라이다
                    # 프레임 오프셋을 yaw 만으로 되돌리면 경사로에서 어긋난다.
                    wp = L2W @ np.array([o.x, o.y, o.z, 1.0])
                    det.world_xy_measured = carla_to_enu(
                        float(wp[0]), float(wp[1])
                    )
                    det.heading_measured = self._object_heading(o, L2W)
                    det.speed_measured = math.hypot(o.vx, o.vy)
                out.append(det)
                self.stats["kept"] += 1
                self.per_camera[cam] = self.per_camera.get(cam, 0) + 1
                if cam != primary:
                    self.stats["seen_only_by_non_primary"] += 1
            self.stats["dropped_no_camera"] += len(objs) - len(best)

        out.sort(key=lambda d: (d.t, d.track_id))
        return out

    @staticmethod
    def _project_to_camera(
        objs: Sequence[LabelObject],
        K: np.ndarray,
        L2C: np.ndarray,
        img_w: int,
        img_h: int,
        cfg: DeepAccidentConfig,
    ) -> List[Tuple[float, LabelObject, Optional[BBox], str]]:
        """3D 박스를 한 카메라에 투영. (최근접깊이, 객체, 화면내 bbox, 탈락사유).

        탈락한 항목은 bbox=None 과 사유 문자열을 달고 그대로 돌려준다 —
        통계 누적은 호출자가 카메라별로 한다.
        """
        out: List[Tuple[float, LabelObject, Optional[BBox], str]] = []
        for o in objs:
            corners = _box_corners_lidar(o)
            cc = L2C @ corners
            depth = cc[0, :]
            if float(np.max(depth)) <= 0.5:
                out.append((0.0, o, None, "behind"))
                continue
            # 카메라 앞쪽 코너만 투영 (뒤쪽 코너는 발산)
            m = depth > 0.5
            uvw = K @ cc[:3, m]
            u = uvw[0, :] / uvw[2, :]
            v = uvw[1, :] / uvw[2, :]
            x1, x2 = float(np.min(u)), float(np.max(u))
            y1, y2 = float(np.min(v)), float(np.max(v))
            # 절두체 컬링 — 화면과 전혀 겹치지 않으면 폐기
            if x2 < 0 or x1 > img_w or y2 < 0 or y1 > img_h:
                out.append((0.0, o, None, "fov"))
                continue
            cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
            cx2, cy2 = min(x2, float(img_w)), min(y2, float(img_h))
            if min(cx2 - cx1, cy2 - cy1) < cfg.min_bbox_px:
                out.append((0.0, o, None, "small"))
                continue
            out.append((float(np.min(depth[m])), o, (cx1, cy1, cx2, cy2), ""))
        return out

    _DROP_STAT = {
        "behind": "dropped_behind",
        "fov": "dropped_outside_fov",
        "small": "dropped_small_bbox",
    }

    def _unoccluded(
        self,
        projected: Sequence[Tuple[float, LabelObject, Optional[BBox], str]],
        img_w: int,
        img_h: int,
        cfg: DeepAccidentConfig,
    ) -> List[Tuple[float, LabelObject, BBox]]:
        """깊이 정렬 가려짐 판정 → (노출면적비, 객체, bbox).

        가까운 것부터 격자를 점유시키고, 남은 노출 면적이 기준 미만이면
        폐기한다. 실제 카메라 검출기가 볼 수 없는 객체를 파이프라인에 넣지
        않기 위한 것이다. 격자는 **이 카메라 전용**이다.
        """
        keep: List[Tuple[float, LabelObject, BBox]] = []
        for depth, o, bbox, reason in projected:
            if bbox is None:
                self.stats[self._DROP_STAT[reason]] += 1
                continue
            keep.append((depth, o, bbox))

        keep.sort(key=lambda z: z[0])
        gp = max(cfg.occlusion_grid_px, 1.0)
        gw, gh = int(img_w / gp) + 1, int(img_h / gp) + 1
        occ = np.zeros((gh, gw), dtype=bool)

        out: List[Tuple[float, LabelObject, BBox]] = []
        for _depth, o, bbox in keep:
            cx1, cy1, cx2, cy2 = bbox
            g0, g1 = int(cx1 / gp), max(int(cx2 / gp), int(cx1 / gp) + 1)
            r0, r1 = int(cy1 / gp), max(int(cy2 / gp), int(cy1 / gp) + 1)
            g1, r1 = min(g1, gw), min(r1, gh)
            cell = occ[r0:r1, g0:g1]
            if cell.size == 0:
                self.stats["dropped_occluded"] += 1
                continue
            free = float(np.count_nonzero(~cell)) / float(cell.size)
            if free < cfg.min_visible_frac:
                self.stats["dropped_occluded"] += 1
                continue
            occ[r0:r1, g0:g1] = True
            out.append((free, o, bbox))
        return out

    @staticmethod
    def _stable_track_id(o: LabelObject, frame: int) -> int:
        """추적 가능한 객체는 CARLA id, 불가한 객체는 프레임별 유일 id.

        id=-1 은 프레임마다 다른 객체를 가리키므로 그대로 쓰면 트랙이 섞인다.
        """
        if not o.is_untracked:
            return o.obj_id
        h = int(abs(hash((round(o.x, 1), round(o.y, 1)))) % 100000)
        return -(frame * 1000000 + h)

    @staticmethod
    def _object_heading(o: LabelObject, lidar_to_world: np.ndarray) -> float:
        """레이블 yaw(라이다 프레임) → 절대 ENU 방위각 [deg]."""
        d = np.array([math.cos(o.yaw), math.sin(o.yaw), 0.0])
        d_world = lidar_to_world[:3, :3] @ d
        e, n = carla_to_enu(float(d_world[0]), float(d_world[1]))
        return wrap360(math.degrees(math.atan2(e, n)))


# ---------------------------------------------------------------- GT 참조


def ground_truth_headings(
    scenario: DeepAccidentScenario,
    cfg: Optional[DeepAccidentConfig] = None,
    frames: Optional[Sequence[int]] = None,
) -> Dict[int, Dict[int, float]]:
    """프레임별 {CARLA id: ENU 방위각 [deg]} — 레이블 yaw 기준.

    방위 평가에는 궤적 차분보다 이것을 써야 한다. 정답 궤적을 0.1초 간격으로
    차분하면 저속 객체의 '정답' 방위 자체가 잡음이라(3m/s 차량이 한 프레임에
    0.3m 이동) 추정치의 오차를 과대평가한다. 레이블 yaw 는 정지 중에도 정확하다.
    """
    cfg = cfg or DeepAccidentConfig()
    out: Dict[int, Dict[int, float]] = {}
    for agent, series in scenario.agents.items():
        self_id = scenario.meta.agent_id_of(agent)
        fr = list(frames) if frames is not None else series.frames
        for f in fr:
            cal = load_calib(series.calib_paths[f])
            L2W = np.asarray(cal["ego_to_world"], dtype=float) @ np.asarray(
                cal["lidar_to_ego"], dtype=float
            )
            per = out.setdefault(f, {})
            for o in parse_label_file(series.label_paths[f]).objects:
                if o.is_untracked:
                    continue
                oid = o.obj_id
                if oid == SELF_ID:
                    if self_id is None:
                        continue
                    oid = self_id
                per[oid] = DeepAccidentPerception._object_heading(o, L2W)
    return out


def merged_ground_truth_tracks(
    scenario: DeepAccidentScenario,
    cfg: Optional[DeepAccidentConfig] = None,
    frames: Optional[Sequence[int]] = None,
) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """모든 관측자의 레이블을 합친 프레임별 {CARLA id: ENU 위치}.

    한 관측자의 레이블만 쓰면 사거리·가려짐 때문에 빠지는 객체가 있다.
    센서 간 월드 좌표는 정확히 일치하므로(검증됨) 단순 병합해도 된다.
    """
    cfg = cfg or DeepAccidentConfig()
    out: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for ag in scenario.agents:
        per = ground_truth_tracks(scenario, cfg, reference_agent=ag, frames=frames)
        for f, d in per.items():
            out.setdefault(f, {}).update(d)
    return out


def estimate_collision(
    scenario: DeepAccidentScenario,
    cfg: Optional[DeepAccidentConfig] = None,
    contact_slack_m: float = 2.0,
) -> CollisionTruth:
    """충돌 시각·주체를 데이터에서 추정한다.

    meta 는 충돌 주체 id 와 강도만 주고 **시각은 주지 않는다**. 사고 분할은
    기록이 충돌 시점에 끊기므로 마지막 프레임이 근사값이지만, 두 주체의 궤적
    최근접 시점을 찾으면 더 정확하다.

    접촉 판정 기준은 두 객체의 **절반 길이 합 + contact_slack_m** 이다. 고정
    임계값을 쓰면 큰 차량이 걸린 충돌을 놓친다 (트럭은 길이 8.5m 이므로 중심
    간 6.4m 도 접촉이다). 기준을 넘으면 궤적으로 접촉을 확인할 수 없다고 보고
    마지막 프레임으로 되돌린다.
    """
    cfg = cfg or DeepAccidentConfig()
    m = scenario.meta
    if not m.collision_occurred or m.collision_id_a is None:
        return CollisionTruth(occurred=False, method="none")

    ids = tuple(i for i in (m.collision_id_a, m.collision_id_b) if i is not None)
    classes = tuple(
        c for c in (m.collision_cls_a, m.collision_cls_b) if c is not None
    )
    frames = scenario.frames()
    dt = 1.0 / cfg.frame_rate_hz
    truth = CollisionTruth(
        occurred=True,
        carla_ids=ids,
        classes=classes,
        agent_roles=m.colliding_agents,
        intensity=m.collision_intensity,
    )

    if len(ids) == 2 and frames:
        # 접촉 기준: 두 객체 절반 길이 합 + 여유
        half_len = 0.0
        for c in (m.collision_cls_a, m.collision_cls_b):
            size = CLASS_SIZES.get(CLASS_MAP.get(c or "", ""))
            half_len += (size.length_m if size else 4.6) / 2.0
        contact_m = half_len + contact_slack_m

        tracks = merged_ground_truth_tracks(scenario, cfg, frames)
        best_f, best_d = None, float("inf")
        for f in frames:
            per = tracks.get(f, {})
            if ids[0] in per and ids[1] in per:
                d = math.dist(per[ids[0]], per[ids[1]])
                if d < best_d:
                    best_f, best_d = f, d
        if best_f is not None and best_d <= contact_m:
            truth.frame = best_f
            truth.time_s = (best_f - 1) * dt
            truth.min_distance_m = best_d
            truth.method = "trajectory"
            return truth
        if best_f is not None:
            truth.min_distance_m = best_d

    # 궤적으로 접촉을 확인할 수 없음 → 기록이 끊긴 마지막 프레임을 충돌 시각으로
    if frames:
        truth.frame = frames[-1]
        truth.time_s = (frames[-1] - 1) * dt
        truth.method = "last_frame"
    return truth


def ground_truth_tracks(
    scenario: DeepAccidentScenario,
    cfg: Optional[DeepAccidentConfig] = None,
    reference_agent: str = "ego_vehicle",
    frames: Optional[Sequence[int]] = None,
) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """프레임별 {CARLA id: ENU 위치} — 정확도 평가와 도로망 합성의 기준.

    reference_agent 의 레이블을 월드로 변환해 사용한다. 자기 자신(id=-100)은
    해당 에이전트의 actor id 로 치환한다.
    """
    cfg = cfg or DeepAccidentConfig()
    series = scenario.agents[reference_agent]
    fr = list(frames) if frames is not None else series.frames
    self_id = scenario.meta.agent_id_of(reference_agent)

    out: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for f in fr:
        cal = load_calib(series.calib_paths[f])
        L2W = np.asarray(cal["ego_to_world"], dtype=float) @ np.asarray(
            cal["lidar_to_ego"], dtype=float
        )
        lf = parse_label_file(series.label_paths[f])
        per: Dict[int, Tuple[float, float]] = {}
        for o in lf.objects:
            if o.is_untracked:
                continue
            oid = o.obj_id
            if oid == SELF_ID:
                if self_id is None:
                    continue
                oid = self_id
            p = L2W @ np.array([o.x, o.y, o.z, 1.0])
            per[oid] = carla_to_enu(float(p[0]), float(p[1]))
        out[f] = per
    return out
