"""데이터 모델 정의.

좌표계 규약 (전 모듈 공통):
  - 픽셀      : (u, v), 좌상단 원점, u→오른쪽, v→아래
  - 차량 좌표 : (x_fwd, y_left, z_up) [m], 카메라 광학중심을 원점으로 하는 오른손 좌표계
  - 월드 좌표 : (e, n) [m], 지역 ENU 평면. e=동쪽(+), n=북쪽(+)
  - 방위각    : heading_deg [0,360), 진북 기준 시계방향 (0=북, 90=동)
  - 시간      : t [s], 모든 차량이 공유하는 절대 시각 (unix epoch 또는 세션 상대시각)
  - 속도      : m/s (직렬화 단계에서만 km/h로 변환)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2) 픽셀


@dataclass
class EgoSample:
    """차량 자체 텔레메트리 한 샘플 (GPS/IMU)."""

    t: float
    lat: float
    lon: float
    heading_deg: float
    speed_mps: float
    yaw_rate_dps: float = 0.0
    alt: float = 0.0


@dataclass
class Detection:
    """한 프레임에서 검출·추적된 객체 하나.

    `*_measured` 필드는 3D 센서(라이다, 스테레오, 시뮬레이터 GT)로 직접
    측정된 값이다. 주어지면 단안 역투영·궤적 미분을 건너뛰고 그 값을 쓴다.
    단안 카메라만 있는 경우에는 모두 None 이며 bbox 로부터 추정한다.
    """

    t: float
    frame_idx: int
    track_id: int
    cls: str  # 'car', 'truck', 'bus', 'van', 'motorcycle', 'bicycle', 'person'
    conf: float
    bbox: BBox
    ego_xy_measured: Optional[Tuple[float, float]] = None
    # 3D 센서가 준 **절대 월드 좌표**(ENU). 있으면 파이프라인이 그대로 쓴다.
    # ego_xy_measured 를 관측차량 자세로 2D 회전해 월드로 되돌리면 경사로에서
    # pitch/roll 성분이 빠져 원거리 객체가 수 m 어긋난다 (Town05 실측 2.4m).
    world_xy_measured: Optional[Tuple[float, float]] = None  # (x_fwd, y_left) [m]
    heading_measured: Optional[float] = None  # 절대 방위각 [deg]
    speed_measured: Optional[float] = None  # 대지 속도 [m/s]
    n_lidar_pts: Optional[int] = None  # 라이다 포인트 수 (가시성 지표)
    # 이 검출을 만든 카메라 이름. 관측자가 카메라를 여러 대 달고 있으면 bbox 를
    # 역투영할 때 **그 카메라의** 내·외부 파라미터를 써야 한다. None 이면
    # 관측자의 기준(첫) 카메라로 본다.
    camera: Optional[str] = None


@dataclass
class Observation:
    """관측자 1대가 특정 시각에 본 객체 1개 (월드 좌표로 투영 완료)."""

    t: float
    observer_id: str
    local_track_id: int
    cls: str
    conf: float
    ego_xy: Tuple[float, float]  # (x_fwd, y_left) 관측자 기준 [m]
    world_xy: Tuple[float, float]  # (e, n) [m]
    distance_m: float
    bbox: BBox
    range_quality: float = 1.0  # 0~1, 거리 추정 신뢰도 (원거리/절단 bbox일수록 낮음)
    heading_deg: Optional[float] = None  # 3D 센서가 준 절대 방위각
    speed_mps: Optional[float] = None  # 3D 센서가 준 대지 속도
    from_3d_sensor: bool = False  # True면 단안 역투영을 쓰지 않은 관측
    camera: Optional[str] = None  # 이 관측을 만든 카메라 이름
    # 이 관측을 만든 **검출 자체의 시각**. t 는 스냅샷 시각이므로, 관측자가
    # 그 시각에 표본이 없어 이웃 프레임 검출을 쓴 경우 둘이 다르다. 정확한 값을
    # 고를 때 낡은 관측을 구별해야 한다 (0.1초는 8m/s 차량에 0.8m 다).
    measured_t: Optional[float] = None


@dataclass
class RoadPlacement:
    """도로망 위에서의 위치."""

    road_id: str
    road_name: str
    s_m: float  # 도로 시작점부터의 종방향 거리
    lateral_offset_m: float  # 진행방향 기준 좌(+)/우(-) 횡방향 오프셋
    direction_label: str  # '북행', '남행', ...
    bearing_deg: float  # 해당 지점 도로 진행 방위각
    lane_index: Optional[int]  # 1 = 중앙선쪽 차선 (한국식 차선번호)
    lane_count: int
    speed_limit_kph: Optional[float]
    dist_to_next_junction_m: Optional[float]
    next_junction_id: Optional[str]
    is_oneway: bool = False
    confidence: float = 1.0
    # 도로축 방위 (진행 방향과 무관한 도로 기하). bearing_deg 는 진행 방향을
    # 반영한 값이고, 이것은 항상 도로 폴리라인의 정방향이다.
    axis_bearing_deg: float = 0.0
    # 진행 **방향**(순/역)을 확신할 수 있는가. 일방통행이거나, 궤적의 축방향
    # 부호가 일관되거나, 실측 지도에서 통행측이 분명할 때 True. False 면
    # bearing_deg·lane_index·direction_label 이 반대일 수 있다.
    direction_confident: bool = True
    # 방향을 무엇으로 정했는가: 'oneway' | 'heading'(실측 방위) |
    # 'lane_side'(통행측) | 'trajectory'(궤적 축방향 부호) | 'default'(단서 없음)
    direction_source: str = ""


@dataclass
class PredictedPath:
    """단기 예측 경로 하나."""

    maneuver: str  # '직진', '좌회전', '우회전', '차선변경(좌)', ...
    probability: float
    waypoints: List[Tuple[float, float]]  # 1초 간격 (e, n)
    horizon_s: float
    # 이 경로가 진입할 수 있는 도로 이름들. 한 교차로에서 같은 기동(예: 우회전)
    # 으로 갈 수 있는 도로가 둘 이상일 때 경로를 구별하는 유일한 단서다 —
    # 이것이 없으면 라벨과 확률이 같은 항목이 여러 개 보여 중복으로 오해된다.
    # 예측 구간 안에서 기하가 같은 진출로들은 한 경로로 합치므로 목록이 된다.
    to_roads: List[str] = field(default_factory=list)
    # 웨이포인트가 예측 구간 끝까지 가지 못하고 도로망 끝에서 멈췄는가.
    # (도로망 밖으로 나가는 경우. 마지막 점을 반복해 채우지 않는다)
    truncated: bool = False


@dataclass
class ActorState:
    """융합 후 확정된 교통 참여자 1대의 상태."""

    actor_id: str
    kind: str  # 'ego'(관측 차량) | 'observed'(관측된 주변 차량)
    cls: str
    world_xy: Tuple[float, float]
    heading_deg: Optional[float]
    speed_mps: Optional[float]
    accel_mps2: Optional[float]
    placement: Optional[RoadPlacement] = None
    maneuver: str = "미확인"
    predictions: List[PredictedPath] = field(default_factory=list)
    observed_by: List[str] = field(default_factory=list)  # 이 차를 본 관측차량 id
    # 이 차량이 **존재한다**는 확신도. 검출기 신뢰도에서 오며, 여러 관측자가
    # 같은 차량을 보면 올라간다.
    confidence: float = 1.0
    # **위치 추정의 정확도**. 존재 확신도와 별개다. 극근접·절단 bbox 는 접지점을
    # 쓸 수 없어 위치가 부정확하지만 차량이 있다는 사실 자체는 확실하다.
    # 둘을 한 값에 섞으면 바로 앞의 차량이 저신뢰로 보고된다.
    position_quality: float = 1.0
    track_age_s: float = 0.0
    # 방위의 출처. 'telemetry'(GPS/IMU) | 'measured'(3D 센서) |
    # 'trajectory'(궤적 미분) | 'road'(도로축) | None(미확정).
    # 궤적 미분 방위는 원거리에서 시선방향 측거 잡음 때문에 180° 뒤집히므로,
    # 도로에 매칭되면 도로축으로 대체한다. 무엇에서 나온 값인지 알아야 하류가
    # 신뢰도를 판단할 수 있다.
    heading_source: Optional[str] = None
    observed_range_m: Optional[float] = None  # 가장 가까운 관측차량까지 거리
    # 이 액터를 만든 원본 검출의 로컬 트랙 id (관측자별 추적기 id).
    # 원시 검출까지 역추적하거나, 트랙 id 가 전역 유일한 데이터셋에서
    # 정답과 신원 기반 매칭을 할 때 쓴다.
    source_track_ids: List[int] = field(default_factory=list)


@dataclass
class Interaction:
    """두 참여자 사이의 상호작용/위험 관계."""

    kind: str  # 'following' | 'crossing' | 'lane_change_conflict' | 'merge'
    subject_id: str
    object_id: str
    gap_m: Optional[float] = None
    headway_s: Optional[float] = None
    ttc_s: Optional[float] = None
    # 설명문(한국어 정본). 출력 언어가 영어면 아래 구조화 필드로 다시 만든다.
    note: str = ""
    # --- 설명문을 언어별로 재구성하기 위한 구조화 필드 ---
    junction_id: Optional[str] = None  # crossing: 상충이 일어날 교차로
    arrival_gap_s: Optional[float] = None  # crossing: 두 차의 도달시간차
    # crossing 상충 유형: 'orthogonal' | 'oncoming_turn'
    conflict: str = ""
    # oncoming_turn 일 때 (actor_id, 좌회전/유턴 합산 확률)
    turn_probs: List[Tuple[str, float]] = field(default_factory=list)
    target_lane: Optional[int] = None  # lane_change_conflict: 진입하려는 차선


@dataclass
class InfraState:
    """노변 인프라 센서(고정 관측자) 1기의 상태.

    교통 참여자가 아니므로 ActorState 와 분리한다. 인프라를 차량 목록에
    섞으면 LLM 이 존재하지 않는 정차 차량으로 오해한다.
    """

    infra_id: str
    world_xy: Tuple[float, float]
    height_m: float
    heading_deg: Optional[float] = None  # 센서 지향 방위각
    n_observed: int = 0  # 이 시각에 이 센서가 기여한 관측 수
    placement: Optional[RoadPlacement] = None  # 설치 지점의 도로 (있으면)


@dataclass
class ScenarioContext:
    """시나리오 배경 정보.

    `available` 은 주행 시점에 실제로 알 수 있는 정보(기상, 도로 형태 등),
    `ground_truth` 는 사후에만 알 수 있는 정답(사고 발생 여부, 충돌 주체,
    충돌 시각)이다. 사고 **예측** 과제에서 ground_truth 를 LLM 입력에 넣으면
    정답 누수가 되므로 직렬화 단계에서 기본적으로 제외한다.
    """

    scenario_id: str = ""
    source: str = ""  # 'DeepAccident' 등
    town: str = ""
    available: Dict[str, object] = field(default_factory=dict)
    ground_truth: Dict[str, object] = field(default_factory=dict)


@dataclass
class CollisionTruth:
    """충돌 정답. 사고 예측 과제의 채점 기준.

    데이터셋(DeepAccident)에서 추정하거나 합성 시나리오에서 직접 구성한다.
    `time_s` 는 관측 시각축(스냅샷 t)과 같은 기준이어야 한다.
    """

    occurred: bool = False
    carla_ids: Tuple[int, ...] = ()  # 충돌 주체의 원본 트랙 id
    classes: Tuple[str, ...] = ()
    agent_roles: Tuple[str, ...] = ()  # 'ego', 'other_behind' 등
    frame: Optional[int] = None
    time_s: Optional[float] = None
    min_distance_m: Optional[float] = None
    intensity: Optional[float] = None
    method: str = "none"  # 'trajectory' | 'last_frame' | 'scripted' | 'none'

    def to_dict(self) -> Dict[str, object]:
        return {
            "occurred": self.occurred,
            "carla_ids": list(self.carla_ids),
            "classes": list(self.classes),
            "agent_roles": list(self.agent_roles),
            "frame": self.frame,
            "time_s": None if self.time_s is None else round(self.time_s, 3),
            "min_distance_m": None
            if self.min_distance_m is None
            else round(self.min_distance_m, 3),
            "intensity": self.intensity,
            "estimation_method": self.method,
        }


@dataclass
class SceneSnapshot:
    """특정 시각의 전체 교통 상황 — LLM 입력의 단위."""

    t: float
    actors: List[ActorState]
    interactions: List[Interaction]
    area_name: str = ""
    map_context: Dict[str, object] = field(default_factory=dict)
    infrastructure: List[InfraState] = field(default_factory=list)
    scenario: Optional[ScenarioContext] = None
    frame_idx: Optional[int] = None
