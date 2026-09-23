"""파이프라인 설정값."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class CameraConfig:
    """카메라 1대의 내·외부 파라미터.

    관측자 1기가 카메라를 여러 대 달 수 있고, 그때는 이 객체를 카메라마다
    하나씩 만들어 목록으로 넘긴다. 방향은 `yaw_deg` 로 구분한다 — 전방 0°,
    좌측 +, 우측 −, 후방 180°.

    fx/fy/cx/cy 를 모르면 hfov_deg 로부터 근사 (from_fov 사용).
    height_m: 노면에서 카메라 광학중심까지 높이.
    pitch_deg: 아래로 향할 때 +. yaw_deg: 좌회전(+, 차량 전방 기준 왼쪽).
    name: 카메라 식별자. 검출 결과의 `Detection.camera` 와 짝지어 역투영에
          쓸 파라미터를 고르는 열쇠다.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    height_m: float = 1.35
    pitch_deg: float = 2.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    name: str = "front"

    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        hfov_deg: float = 60.0,
        **kwargs,
    ) -> "CameraConfig":
        import math

        fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(
            width=width,
            height=height,
            fx=fx,
            fy=fx,  # 정사각 픽셀 가정
            cx=width / 2.0,
            cy=height / 2.0,
            **kwargs,
        )


@dataclass
class LaneConfig:
    lane_width_m: float = 3.25
    drive_side: str = "right"  # 'right'(한국/미국) | 'left'
    # 차선 번호 규약: 'from_median' = 1차선이 중앙선쪽 (한국),
    #                'from_curb'   = 1차선이 가장자리쪽 (미국식 lane 1)
    numbering: str = "from_median"


@dataclass
class ClassSize:
    """클래스별 실물 치수 사전값 [m]."""

    width_m: float
    length_m: float
    height_m: float


# 기본 사전값. DeepAccident 레이블 실측 중앙값을 반영했다 (car 4.60×1.93×1.52,
# truck 8.47×2.89×3.83 등). 대상 차종 분포가 다르면 조정한다.
DEFAULT_CLASS_SIZES: Dict[str, ClassSize] = {
    "car": ClassSize(1.93, 4.60, 1.52),
    "truck": ClassSize(2.89, 8.47, 3.83),
    "van": ClassSize(1.99, 4.48, 2.04),
    "bus": ClassSize(2.90, 11.0, 3.20),
    "motorcycle": ClassSize(0.86, 2.21, 1.53),
    "bicycle": ClassSize(0.37, 1.64, 1.77),
    "person": ClassSize(0.38, 0.38, 1.86),
}


@dataclass
class PerceptionConfig:
    sample_hz: float = 2.0  # 프레임 샘플링 주기
    min_conf: float = 0.35
    model_name: str = "yolo11m.pt"  # ultralytics 가중치 (없으면 mock 사용)
    tracker: str = "bytetrack.yaml"
    vehicle_classes: tuple = ("car", "truck", "bus", "motorcycle", "bicycle")
    # 클래스별 치수 사전값 — 단안 거리 추정의 보조 단서
    class_size: Dict[str, ClassSize] = field(
        default_factory=lambda: dict(DEFAULT_CLASS_SIZES)
    )

    @property
    def class_width_m(self) -> Dict[str, float]:
        """하위호환용 폭 사전값 조회."""
        return {k: v.width_m for k, v in self.class_size.items()}


@dataclass
class FusionConfig:
    """다중 차량 관측 융합 파라미터.

    단안 측거 오차는 거리의 제곱에 가깝게 증가한다. 고정 게이트를 쓰면
    원거리에서 같은 차량이 2대로 중복 계수되므로 게이트를 거리에 비례해
    넓힌다: gate = max_match_dist_m + range_gate_coeff * 관측거리.
    """

    max_match_dist_m: float = 6.0  # 근거리 기준 게이트
    range_gate_coeff: float = 0.14  # 관측거리 1m 당 게이트 증가량
    max_gate_m: float = 25.0  # 게이트 상한 (과도한 병합 방지)
    # 거리 추정 신뢰도(range_quality)가 낮은 관측에 허용할 추가 게이트 [m].
    # 극근접·절단 bbox 는 접지점을 못 써서 오차가 수 m 에 달하므로, 신뢰도가
    # 낮을수록 더 멀리까지 같은 물체로 볼 수 있어야 병합에 실패하지 않는다.
    quality_gate_m: float = 4.0
    # 로컬 트랙 id 가 전 관측자에서 전역 유일한가. **기본 True.**
    #
    # 이 프로젝트의 전제는 V2X 로 객체 id 를 공유하는 협력형 자율주행이다.
    # 같은 id 면 같은 물체이므로 위치와 무관하게 병합해 신원을 정확히 유지한다 —
    # 극근거리 측거 오차로 한 차량이 두 액터로 갈라지는 것을 막는다.
    # DeepAccident 의 CARLA actor id 도 전 관측자에서 동일하다.
    #
    # **카메라별 독립 추적기**(YOLO+ByteTrack 등)를 쓰면 각자 1번부터 번호를
    # 매기므로 이 가정이 깨진다. 그때는 False 로 두어야 한다. False 로 두는 것을
    # 잊어도 조용히 틀리지는 않는다 — 같은 id 관측이 `id_collision_dist_m` 보다
    # 흩어져 있으면 위치로 쪼개고 `GlobalTrackRegistry.id_collisions` 를 올린다.
    global_track_ids: bool = True
    # 같은 트랙 id 가 이 거리보다 흩어져 나타나면 "전역 유일" 가정 위반으로 본다.
    #
    # 임계는 실측으로 정했다. DeepAccident 를 camera 모드(단안 역투영)로 돌려
    # **id 가 진짜 전역인** 다관측 클러스터 355건의 위치 산포를 재면
    # p50 5.6m · p90 12.4m · p99 39.9m · 최대 66.0m 다 (sensor3d 는 GT 라 항상 0).
    # 반면 진짜 id 충돌은 서로 무관한 차량이므로 수백 m 떨어진다.
    # 두 분포 사이에 두어야 오탐(정상을 쪼갬)과 미탐이 모두 없다 —
    # 40m 로 두면 실데이터에서 3건이 잘못 쪼개졌다. 0 이면 검사하지 않는다.
    id_collision_dist_m: float = 150.0
    class_mismatch_penalty_m: float = 3.0
    ego_match_dist_m: float = 8.0  # 관측객체가 다른 관측차량 본체인지 판정
    max_range_m: float = 120.0  # 이 거리를 넘는 관측은 신뢰불가로 폐기
    track_timeout_s: float = 2.5
    # 액터 id 재사용을 허용할 최대 함의 속도 [m/s]. 직전 위치에서 지금 위치까지
    # 이 속도를 넘어야 도달 가능하다면 다른 차량으로 본다. 오연관으로 액터가
    # 순간이동하며 속도가 200km/h 로 튀는 것을 막는다.
    # 40m/s = 144km/h — 도심 시나리오의 실제 속도보다 충분히 높고, 단안 위치
    # 잡음(2~3m)이 0.5초 간격에 더하는 겉보기 속도(~6m/s)도 흡수한다.
    max_speed_mps: float = 40.0
    # 클래스별 상한 [m/s]. 보행자에게 차량과 같은 상한을 주면 오연관으로
    # 100km/h 로 달리는 보행자가 생긴다. 없는 클래스는 max_speed_mps 를 쓴다.
    #
    # 이 값은 **신원 관문용**이므로 넉넉하다 — 단안 위치 잡음(2~4m)이 0.5초
    # 간격에 더하는 겉보기 속도를 흡수해야 같은 물체가 두 액터로 갈라지지 않는다.
    max_speed_by_class: Dict[str, float] = field(
        default_factory=lambda: {
            "person": 8.0,  # 전력질주 수준
            "bicycle": 13.0,
            "motorcycle": 40.0,
            "car": 40.0,
            "van": 36.0,
            "truck": 33.0,
            "bus": 30.0,
        }
    )
    # 클래스별 **지속 주행 속도** 상한 [m/s]. 신원 관문(max_speed_by_class)보다
    # 좁고, 속도를 *보고*할 때 쓴다. 두 역할을 한 값으로 겸하면 안 된다:
    # 보행자는 1~2m/s 로 걷는데 신원 관문 상한(8m/s)을 보고에도 쓰면 위치 잡음이
    # 그대로 "28km/h 로 달리는 보행자"로 나가고, 그 잡음 궤적이 조감도에서
    # 인도 위를 지그재그로 가로지르는 선으로 그려진다.
    report_speed_by_class: Dict[str, float] = field(
        default_factory=lambda: {
            "person": 2.5,  # 빠른 걸음~조깅. CARLA 워커는 약 1.4m/s
            "bicycle": 8.0,
            "motorcycle": 33.0,
            "car": 40.0,
            "van": 33.0,
            "truck": 28.0,  # 100km/h
            "bus": 25.0,
        }
    )


@dataclass
class SerializeConfig:
    """LLM 입력 직렬화 파라미터."""

    language: str = "ko"  # 'ko' | 'en'
    # LLM provider. payload 파일의 형식이 이것으로 결정된다.
    #   'claude' (기본) — Anthropic Messages API
    #   'openai'        — OpenAI Chat Completions
    #   'gemini'        — Google Gemini generateContent
    # 내용(브리핑·JSON·질문·프롬프트·스키마)은 provider 와 무관하게 같다.
    provider: str = "claude"
    # 모델 id. None 이면 provider 기본값 (claude 만 기본값이 있다).
    model: Optional[str] = None
    # provider 고유 파라미터를 덮어쓸 dict. 추론 예산처럼 버전에 따라 달라지는
    # 값은 여기로 넣는다 (검증하지 않은 값을 코드에 박지 않기 위함).
    provider_extra: Optional[Dict[str, object]] = None
    max_actors: int = 20  # 토큰 예산: 중요도 상위 N대만 포함
    # 구조화 JSON 블록을 payload 에 넣는가 (build_messages 기본값).
    # 자연어 브리핑과 내용이 겹치고 입력의 대부분을 차지하므로 기본은 꺼짐.
    include_json_block: bool = False
    include_bev_ascii: bool = True
    bev_range_m: float = 80.0
    round_digits: int = 1
    # 시나리오 식별자를 불투명 토큰으로 치환한다. DeepAccident 의 시나리오 id 는
    # 분할명("...._accident" / "..._normal")을 포함하므로 그대로 넣으면 사고
    # 예측 과제에서 정답이 새어 나간다. 실제 id 는 평가 레코드에만 남긴다.
    redact_scenario_id: bool = True


@dataclass
class DeepAccidentConfig:
    """DeepAccident 데이터셋 어댑터 설정."""

    # 사용할 카메라. None(기본) 이면 calib 에 있는 **카메라 전부**를 쓴다 —
    # DeepAccident 는 관측자마다 6대(Front, FrontLeft, FrontRight, Back,
    # BackLeft, BackRight)를 달고 있어 yaw 0/±55/±110/180° 로 전방위를 덮는다.
    # 전면 1대만 쓰면 측방·후방 차량을 통째로 놓친다.
    # 특정 카메라만 쓰려면 이름 목록을 준다 (예: ("Camera_Front",)).
    cameras: Optional[Tuple[str, ...]] = None
    # 기준 카메라. 노면고도 추정과 관측자 설치높이 표시에 쓴다.
    camera: str = "Camera_Front"
    frame_rate_hz: float = 10.0  # 데이터셋 기록 주기
    # 관측 생성 모드:
    #   'sensor3d' (기본) — 데이터셋 레이블의 3D 위치·방위(yaw)·속도(vx,vy)를
    #              **그대로** 쓴다. "각 차량이 자기 센서로 주변 차량의 상태를
    #              정확히 수집했다"는 전제에 해당한다. 어느 차량이 보이는지는
    #              여전히 센서 가시성(화각·가려짐·사거리)으로 정한다 — 값만
    #              정확하고, 보이지 않는 차량은 모른다.
    #   'camera' — GT 3D 박스를 카메라에 투영해 2D bbox 만 남기고, 파이프라인의
    #              단안 역투영으로 위치를 **추정**한다. 단안 인지 오차가 하류
    #              (차선 배정·기동 판정·사고 예측)에 미치는 영향을 볼 때 쓴다.
    #              실측 위치오차 중앙값 2.3m, 원거리에서는 7m 를 넘는다.
    observation_mode: str = "sensor3d"
    # 라이다 포인트 수 필터. 라이다는 사거리·밀도가 제한되어 원거리 객체는
    # 화면에 선명해도 npts=0 이 되므로 카메라 가시성 대용으로 쓰면 안 된다.
    # 기본 0(미적용). 라이다 기반 실험에서만 올린다.
    min_lidar_pts: int = 0
    # 데이터셋 vis 플래그는 6개 카메라 OR 이다. 카메라 6대를 모두 쓰면 이 플래그가
    # 곧 "어느 카메라에도 안 보인다"와 같은 범위이므로 정확한 가시성 관문이 된다.
    # 일부 카메라만 쓸 때는 **필요조건**으로만 유효하다 (보인다고 표시된 객체가
    # 선택한 카메라에는 안 보일 수 있고, 그건 절두체 컬링이 걸러낸다).
    require_camera_visibility: bool = True
    # 깊이 정렬 가려짐 판정에서 살아남아야 하는 최소 노출 면적 비율
    min_visible_frac: float = 0.35
    occlusion_grid_px: float = 10.0  # 가려짐 판정 격자 해상도
    min_bbox_px: float = 12.0  # 화면상 최소 bbox 크기
    max_range_m: float = 120.0  # 이 거리 초과 관측 폐기
    include_untracked: bool = False  # id=-1 (추적 불가) 객체 포함 여부
    include_pedestrians: bool = True  # 보행자를 관측 대상에 포함
    agents: tuple = (
        "ego_vehicle",
        "ego_vehicle_behind",
        "other_vehicle",
        "other_vehicle_behind",
    )
    use_infrastructure: bool = True
    # 합성 텔레메트리의 기준 위경도. CARLA 좌표는 가상이므로 값 자체는
    # 임의이며, ENU 왕복 변환이 일관되기만 하면 된다.
    geo_origin: Tuple[float, float] = (0.0, 0.0)
    # GPS 잡음 주입 (합성 텔레메트리를 현실적으로 만들 때). 0이면 완전 정확.
    gps_noise_m: float = 0.0
    heading_noise_deg: float = 0.0
    random_seed: int = 0
    # CARLA 실측 차선폭 [m]. mini 데이터의 같은 방향 궤적 간격 분포가 3.50m 에
    # 강하게 집중한다 (n=33, 중앙값 3.50). LaneConfig 기본값(3.25)과 다르므로
    # 어댑터가 LaneConfig 를 이 값으로 맞춘다.
    lane_width_m: float = 3.5


@dataclass
class RoadGenConfig:
    """궤적 기반 도로망 합성 설정."""

    heading_bin_deg: float = 20.0  # 같은 진행방향으로 묶을 각도 허용범위
    corridor_width_m: float = 2.0  # 같은 차로로 볼 횡방향 허용범위
    min_corridor_len_m: float = 12.0  # 이보다 짧은 궤적은 도로 근거로 쓰지 않음
    min_points_per_corridor: int = 4
    road_group_width_m: float = 22.0  # 하나의 도로로 묶을 최대 횡방향 폭
    junction_snap_m: float = 12.0  # 교차점 클러스터링 반경
    extend_m: float = 8.0  # 관측 구간 양끝을 조금 연장 (도로가 끊겨 보이지 않게)
    resample_m: float = 4.0  # 중심선 재샘플 간격
    # 별개 차선으로 셀 최소 횡방향 간격 = 차선폭 × 이 계수.
    # 같은 차선 안에서의 횡방향 흔들림(회전 접근 등)을 별도 차선으로 세지 않기 위함.
    lane_separation_frac: float = 0.6
    # 같은 도로로 볼 최대 차선 간 공백 = 차선폭 × 이 계수. 이보다 넓게 벌어진
    # 차로는 사이에 차선이 들어갈 수 없으므로 다른 도로로 분리한다.
    # (DeepAccident 실측: 같은 방향 차선 간격 3.5m, 대향 최소 3.5m)
    max_lane_gap_frac: float = 1.6
    min_road_len_m: float = 10.0  # 교차로 분할 후 이보다 짧은 조각은 버린다


@dataclass
class PipelineConfig:
    lane: LaneConfig = field(default_factory=LaneConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    serialize: SerializeConfig = field(default_factory=SerializeConfig)
    deepaccident: DeepAccidentConfig = field(default_factory=DeepAccidentConfig)
    roadgen: RoadGenConfig = field(default_factory=RoadGenConfig)
    prediction_horizon_s: float = 5.0
    area_name: str = ""
