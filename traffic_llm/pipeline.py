"""전체 파이프라인 오케스트레이션.

    영상 N개 + 텔레메트리 N개 + 도로지도 1개
        → 시각 동기화 스냅샷 스트림
        → LLM 입력(JSON / 텍스트 / 메시지 페이로드)

사용 예:
    conv = TrafficSceneConverter(PipelineConfig(), network)
    conv.add_vehicle("V1", "front1.mp4", "tele1.csv", cam_cfg)
    conv.add_vehicle("V2", "front2.mp4", "tele2.csv", cam_cfg)
    for snap in conv.run(rate_hz=2.0):
        print(to_text(snap, cfg.serialize))
"""

from __future__ import annotations

import bisect
import csv
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from .config import CameraConfig, PipelineConfig
from .fusion import GlobalTrackRegistry, class_group, fuse_observations
from .geometry import CameraModel, ego_to_world, wrap180, wrap360
from .kinematics import KinematicsTracker, analyze_interactions
from .perception import PerceptionBackend, YoloPerception, build_backend
from .prediction import predict
from .roadmap import RoadNetwork
from .schemas import (
    Detection,
    EgoSample,
    InfraState,
    Observation,
    ScenarioContext,
    SceneSnapshot,
)


def load_telemetry(path: str) -> List[EgoSample]:
    """CSV 텔레메트리 로딩.

    필수 컬럼: t, lat, lon, heading_deg, speed_mps
    선택 컬럼: yaw_rate_dps, alt
    speed 컬럼명이 speed_kph 인 경우도 허용한다.
    """
    out: List[EgoSample] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if "speed_mps" in row and row["speed_mps"] not in (None, ""):
                speed = float(row["speed_mps"])
            elif row.get("speed_kph"):
                speed = float(row["speed_kph"]) / 3.6
            else:
                speed = 0.0
            out.append(
                EgoSample(
                    t=float(row["t"]),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    heading_deg=wrap360(float(row["heading_deg"])),
                    speed_mps=speed,
                    yaw_rate_dps=float(row.get("yaw_rate_dps") or 0.0),
                    alt=float(row.get("alt") or 0.0),
                )
            )
    out.sort(key=lambda s: s.t)
    if not out:
        raise ValueError(f"{path}: 텔레메트리 샘플이 없습니다")
    return out


def interp_ego(samples: List[EgoSample], ts: List[float], t: float) -> Optional[EgoSample]:
    """시각 t 의 자차 상태를 선형보간. 데이터 범위를 벗어나면 None."""
    if t < ts[0] - 0.5 or t > ts[-1] + 0.5:
        return None
    i = bisect.bisect_left(ts, t)
    if i == 0:
        return samples[0]
    if i >= len(samples):
        return samples[-1]
    a, b = samples[i - 1], samples[i]
    dt = b.t - a.t
    if dt <= 1e-9:
        return a
    r = (t - a.t) / dt
    # 방위각은 최단 회전 방향으로 보간
    dh = wrap180(b.heading_deg - a.heading_deg)
    return EgoSample(
        t=t,
        lat=a.lat + r * (b.lat - a.lat),
        lon=a.lon + r * (b.lon - a.lon),
        heading_deg=wrap360(a.heading_deg + r * dh),
        speed_mps=a.speed_mps + r * (b.speed_mps - a.speed_mps),
        yaw_rate_dps=a.yaw_rate_dps + r * (b.yaw_rate_dps - a.yaw_rate_dps),
        alt=a.alt + r * (b.alt - a.alt),
    )


@dataclass
class VehicleSource:
    """관측자 1기의 입력 묶음.

    role='vehicle'         : 카메라를 1대 이상 가진 주행 차량 (교통 참여자)
    role='infrastructure'  : 노변 고정 센서 (교통 참여자가 아님)

    카메라는 **k대**까지 달 수 있다. `cameras[0]` 이 기준 카메라이고,
    `Detection.camera` 가 어느 카메라의 것인지 지정한다 — 역투영은 반드시 그
    카메라의 파라미터로 해야 한다. 후방 카메라의 검출을 전방 파라미터로 풀면
    yaw 가 180° 틀려 차량이 반대편에 놓인다.
    """

    vehicle_id: str
    video_path: str
    cameras: List[CameraConfig]
    telemetry: List[EgoSample]
    detections: List[Detection] = field(default_factory=list)
    t_offset: float = 0.0  # 영상 시각 보정 (동기화 오차 보상)
    role: str = "vehicle"
    mount_height_m: float = 0.0  # 인프라 설치 높이 (표시용)
    # 이 관측자 본체가 다른 관측자의 검출에서 갖는 트랙 id.
    # 트랙 id 가 전역 유일한 경우(V2X 객체 id 공유, 시뮬레이터 GT)에만 의미가
    # 있으며, 거리 게이트 없이 정확한 신원 판정을 가능하게 한다.
    self_track_id: Optional[int] = None
    _models: Dict[str, CameraModel] = field(default_factory=dict)
    _tele_ts: List[float] = field(default_factory=list)
    _det_by_time: Dict[float, List[Detection]] = field(default_factory=dict)

    @property
    def is_infrastructure(self) -> bool:
        return self.role == "infrastructure"

    @property
    def camera(self) -> CameraConfig:
        """기준 카메라. 관측자 위치·설치높이 표시 등 카메라 1대만 필요한 곳에서 쓴다."""
        return self.cameras[0]

    def prepare(self) -> None:
        if not self.cameras:
            raise ValueError(f"{self.vehicle_id}: 카메라가 하나도 없습니다")
        self._models = {c.name: CameraModel(c) for c in self.cameras}
        self._tele_ts = [s.t for s in self.telemetry]
        self._det_by_time = {}
        for d in self.detections:
            self._det_by_time.setdefault(round(d.t + self.t_offset, 2), []).append(d)
        self._det_times = sorted(self._det_by_time)

    def detections_near(self, t: float, tol: float) -> List[Detection]:
        """t 에 가장 가까운 프레임의 검출 결과."""
        if not self._det_times:
            return []
        i = bisect.bisect_left(self._det_times, t)
        cands = [j for j in (i - 1, i) if 0 <= j < len(self._det_times)]
        if not cands:
            return []
        best = min(cands, key=lambda j: abs(self._det_times[j] - t))
        if abs(self._det_times[best] - t) > tol:
            return []
        return self._det_by_time[self._det_times[best]]

    @property
    def model(self) -> CameraModel:
        """기준 카메라의 모델."""
        return self.model_for(None)

    def model_for(self, camera: Optional[str]) -> CameraModel:
        """검출을 만든 카메라의 모델. 이름이 없거나 모르면 기준 카메라.

        모르는 이름을 조용히 기준 카메라로 떨어뜨린다 — 카메라 목록과 검출의
        camera 필드가 어긋나는 것은 설정 오류이지만, 그 때문에 파이프라인 전체가
        멈추는 것보다 기준 카메라로 처리하고 계속 도는 편이 낫다. 이름이 맞으면
        정확한 파라미터가 쓰이므로 정상 경로는 영향받지 않는다.
        """
        assert self._models, "prepare() 를 먼저 호출하세요"
        if camera is not None:
            m = self._models.get(camera)
            if m is not None:
                return m
        return self._models[self.cameras[0].name]


class TrafficSceneConverter:
    """다중 차량 영상 + 도로지도 → LLM 입력용 교통 상황 스냅샷."""

    def __init__(
        self,
        cfg: PipelineConfig,
        network: RoadNetwork,
        backend: Optional[PerceptionBackend] = None,
    ):
        self.cfg = cfg
        self.network = network
        self.backend = backend or build_backend(cfg.perception)
        self.sources: Dict[str, VehicleSource] = {}
        self.scenario: Optional[ScenarioContext] = None  # 데이터셋 배경 정보
        self.reset()

    def reset(self) -> None:
        """프레임 간 누적 상태(신원 레지스트리, 궤적 이력)를 초기화한다.

        관측차량 등록(sources)과 지도는 입력이므로 유지한다.

        `run()` 은 시나리오를 처음부터 스트리밍하는 것이므로 매 호출에 이것을
        부른다. 초기화하지 않으면 두 번째 호출이 첫 호출의 이력을 안은 채 t=0
        부터 다시 시작해, 방위·속도 추정이 9초 전 표본과 섞여 망가진다. 실제로
        CLI 가 JSONL 은 1회차, 윈도우·BEV 는 2·3회차로 만들어 조감도의 차량
        방향이 도로와 어긋났다.
        """
        cfg = self.cfg
        self.registry = GlobalTrackRegistry(cfg.fusion)
        self.kinematics = KinematicsTracker(
            cfg.fusion.track_timeout_s,
            cfg.lane.lane_width_m,
            # 속도 추정에는 **보고용** 상한을 쓴다. 신원 관문 상한은 잡음 흡수를
            # 위해 넉넉하므로, 그것으로 속도를 걸러내면 잡음이 그대로 보고된다.
            max_speed_by_class=cfg.fusion.report_speed_by_class,
            max_speed_mps=cfg.fusion.max_speed_mps,
        )
        self.dropped_far = 0  # max_range_m 초과로 폐기된 관측 수

    # ------------------------------------------------------------ 입력 등록

    def add_vehicle(
        self,
        vehicle_id: str,
        video_path,
        telemetry_path: str,
        camera,
        t_offset: float = 0.0,
        detections: Optional[List[Detection]] = None,
        role: str = "vehicle",
        telemetry: Optional[List[EgoSample]] = None,
        mount_height_m: float = 0.0,
        self_track_id: Optional[int] = None,
    ) -> None:
        """관측자를 등록한다.

        `camera` 는 `CameraConfig` 하나 또는 **여러 대의 목록**이다. 여러 대면
        `video_path` 도 같은 길이의 목록(카메라 순서대로) 또는
        `{카메라이름: 경로}` 로 줄 수 있다 — 카메라마다 검출을 따로 돌리고
        결과에 `Detection.camera` 를 채운다. 경로를 하나만 주면 그 영상을
        기준 카메라의 것으로 본다.

        telemetry 를 직접 주면 CSV 로딩을 건너뛴다 (합성 텔레메트리 사용 시).
        role='infrastructure' 면 노변 고정 센서로 취급해 교통 참여자 목록에서
        제외한다.
        """
        tele = telemetry if telemetry is not None else load_telemetry(telemetry_path)
        if not tele:
            raise ValueError(f"{vehicle_id}: 텔레메트리가 비어 있습니다")
        cams = [camera] if isinstance(camera, CameraConfig) else list(camera)
        if not cams:
            raise ValueError(f"{vehicle_id}: 카메라를 하나 이상 주십시오")
        if len({c.name for c in cams}) != len(cams):
            raise ValueError(
                f"{vehicle_id}: 카메라 이름이 중복됩니다 "
                f"({[c.name for c in cams]}). 이름으로 역투영 파라미터를 "
                "고르므로 유일해야 합니다."
            )
        videos = self._video_map(vehicle_id, video_path, cams)

        src = VehicleSource(
            vehicle_id=vehicle_id,
            video_path=videos[cams[0].name],
            cameras=cams,
            telemetry=tele,
            t_offset=t_offset,
            role=role,
            mount_height_m=mount_height_m,
            self_track_id=self_track_id,
        )
        if detections is not None:
            src.detections = detections
        else:
            src.detections = self._detect_all(vehicle_id, cams, videos, tele[0].t)
        src.prepare()
        self.sources[vehicle_id] = src

    @staticmethod
    def _video_map(
        vehicle_id: str, video_path, cams: List[CameraConfig]
    ) -> Dict[str, str]:
        """카메라 이름 → 영상 경로."""
        names = [c.name for c in cams]
        if isinstance(video_path, dict):
            missing = [n for n in names if n not in video_path]
            if missing:
                raise ValueError(
                    f"{vehicle_id}: 카메라 {missing} 의 영상 경로가 없습니다"
                )
            return {n: video_path[n] for n in names}
        if isinstance(video_path, (list, tuple)):
            if len(video_path) != len(cams):
                raise ValueError(
                    f"{vehicle_id}: 영상 {len(video_path)}개와 카메라 "
                    f"{len(cams)}대의 수가 다릅니다"
                )
            return dict(zip(names, video_path))
        # 경로 하나 — 카메라가 여러 대여도 기준 카메라의 영상으로 본다
        return {names[0]: video_path}

    def _detect_all(
        self,
        vehicle_id: str,
        cams: List[CameraConfig],
        videos: Dict[str, str],
        t0: float,
    ) -> List[Detection]:
        """카메라별로 검출을 돌리고 결과에 카메라 이름을 새긴다."""
        out: List[Detection] = []
        for cam in cams:
            path = videos.get(cam.name)
            if path is None:
                # 영상이 없는 카메라는 건너뛴다. 카메라 목록에는 남겨 둔다 —
                # 다른 관측자가 준 검출을 역투영할 때 파라미터가 필요하다.
                continue
            # 사전 검출 결과를 쓰는 백엔드(JsonPerception)는 영상 파일이 없어도
            # 되므로, 영상 존재 여부는 영상을 직접 읽는 백엔드에만 요구한다.
            if isinstance(self.backend, YoloPerception) and not (
                path and os.path.exists(path)
            ):
                raise FileNotFoundError(
                    f"{vehicle_id}/{cam.name}: 영상을 찾을 수 없습니다 ({path}). "
                    "사전 계산된 검출 결과를 쓰려면 JsonPerception 백엔드를 "
                    "사용하거나 detections 인자를 넘기십시오."
                )
            for det in self.backend.run(path, t0=t0):
                if det.camera is None:
                    det.camera = cam.name
                out.append(det)
        out.sort(key=lambda d: (d.t, d.track_id))
        return out

    def add_infrastructure(
        self,
        infra_id: str,
        camera: CameraConfig,
        telemetry: List[EgoSample],
        detections: List[Detection],
        mount_height_m: float = 0.0,
        t_offset: float = 0.0,
    ) -> None:
        """노변 고정 센서를 등록한다 (V2X 인프라 관측)."""
        self.add_vehicle(
            infra_id,
            video_path=f"<{infra_id}>",
            telemetry_path="",
            camera=camera,
            t_offset=t_offset,
            detections=detections,
            role="infrastructure",
            telemetry=telemetry,
            mount_height_m=mount_height_m or camera.height_m,
        )

    # ------------------------------------------------------------ 실행

    def time_range(self) -> Tuple[float, float]:
        starts = [s.telemetry[0].t for s in self.sources.values()]
        ends = [s.telemetry[-1].t for s in self.sources.values()]
        # 모든 차량이 데이터를 가진 구간(교집합)만 처리 — 부분 관측 왜곡 방지
        return (max(starts), min(ends))

    def run(
        self,
        rate_hz: Optional[float] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        reset: bool = True,
    ) -> Iterator[SceneSnapshot]:
        """[t_start, t_end] 를 rate_hz 로 훑어 스냅샷을 스트리밍한다.

        reset=True(기본)면 매 호출에 누적 상태를 초기화하므로 **같은 입력에 대해
        몇 번 호출해도 같은 결과**가 나온다. 구간을 나눠 이어서 처리할 때만
        reset=False 로 연속성을 유지한다.
        """
        if not self.sources:
            raise RuntimeError("등록된 관측차량이 없습니다")
        if reset:
            self.reset()
        rate = rate_hz or self.cfg.perception.sample_hz
        dt = 1.0 / rate
        lo, hi = self.time_range()
        t = t_start if t_start is not None else lo
        end = t_end if t_end is not None else hi
        if t > end:
            raise ValueError(
                f"공통 시간 구간이 없습니다 (차량별 텔레메트리 시각 확인: {lo:.1f}~{hi:.1f})"
            )

        tol = dt * 0.75
        while t <= end + 1e-6:
            snap = self.snapshot_at(t, tol)
            if snap is not None:
                yield snap
            t += dt

    def _refine_placement(self, a: ActorState) -> None:
        """도로 매칭을 다시 하며 진행 방향을 확정한다.

        1차 매칭은 방위를 모르고 한 것이므로 도로만 맞고 방향은 미정일 수 있다.
        여기서 도로축을 알았으니 궤적의 **축방향 부호**를 여러 창에서 교차 확인해
        방향을 정한다. 방향이 확정되지 않으면 placement 의 방향 관련 필드
        (진행 방위·차선번호·진행방향 라벨)를 신뢰할 수 없다고 표시한다.
        """
        if a.placement is None:
            return
        if a.heading_source in ("telemetry", "measured"):
            # 실측 방위가 있으면 그것으로 방향을 정한다 (가장 정확)
            refined = self.network.locate(
                a.world_xy, a.heading_deg, trust_heading_direction=True
            )
            if refined is not None:
                a.placement = refined
            return
        hint = self.kinematics.direction_hint(a.actor_id)
        base = self.network.locate(a.world_xy, hint)
        if base is None:
            return
        # 판정근거를 **실측 신뢰도 순**으로 쓴다 (DeepAccident mini 실측 반전율):
        #   통행측 7% < 궤적 축방향 부호 9% < 일방통행 표기 19~25%
        # 통행측은 지도 기하만 쓰므로 원거리 측거 편향에 영향받지 않는다.
        if base.direction_source == "lane_side" and base.direction_confident:
            a.placement = base
            return
        rev = self.kinematics.axis_direction(a.actor_id, base.axis_bearing_deg)
        if rev is not None:
            refined = self.network.locate(a.world_xy, hint, reverse=rev)
            if refined is not None:
                a.placement = refined
                return
        a.placement = base

    def _adopt_road_heading(self, a: ActorState) -> None:
        """궤적 미분 방위를 **도로축 방위**로 대체한다.

        단안 궤적 방위는 원거리에서 시선방향 측거 잡음 때문에 180° 뒤집히는 일이
        흔하다 (DeepAccident 레이블 yaw 대비 90° 초과가 10%). 도로에 매칭된
        차량은 거의 항상 차로를 따라가므로 지도의 도로축이 궤적보다 정확하고,
        진행 방향(순/역)은 통행측으로 판별하므로 뒤집힌 궤적에 오염되지 않는다.

        텔레메트리(GPS/IMU)나 3D 센서가 준 방위는 궤적보다 정확하므로 건드리지
        않는다. 도로 매칭이 없으면(주차장·도로 밖) 궤적 값을 그대로 둔다.
        보행자·자전거는 차로를 따르지 않으므로(인도를 걷거나 횡단한다) 대상에서
        제외한다 — 도로축을 씌우면 오히려 방향이 틀어진다.
        """
        if a.placement is None:
            return
        if a.heading_source in ("telemetry", "measured"):
            return
        if class_group(a.cls) == "vru":
            return
        if not a.placement.direction_confident:
            # 진행 방향을 모르면 방위를 **미확정**으로 둔다. 절반의 확률로
            # 180° 틀린 값을 내보내는 것보다 낫고, 조감도는 도로축에 맞춘
            # 사각형(점선 테두리, 화살표 없음)으로 그리므로 표현 손실도 없다
            # — 사각형은 180° 뒤집어도 같은 모양이다.
            a.heading_deg = None
            a.heading_source = None
            return
        a.heading_deg = a.placement.bearing_deg
        a.heading_source = "road"

    def snapshot_at(self, t: float, tol: float = 0.4) -> Optional[SceneSnapshot]:
        """단일 시각의 교통 상황 구성."""
        ego_states: Dict[str, EgoSample] = {}
        ego_world: Dict[str, Tuple[float, float]] = {}
        observations: List[Observation] = []

        for vid, src in self.sources.items():
            samp = interp_ego(src.telemetry, src._tele_ts, t)
            if samp is None:
                continue
            origin = self.network.enu.to_enu(samp.lat, samp.lon)
            ego_states[vid] = samp
            ego_world[vid] = origin

            for det in src.detections_near(t, tol):
                if det.cls not in self.cfg.perception.vehicle_classes:
                    continue
                if det.ego_xy_measured is not None:
                    # 3D 센서(라이다/스테레오/시뮬레이터 GT)가 준 위치 —
                    # 단안 역투영을 건너뛴다
                    ego_xy = det.ego_xy_measured
                    quality = 1.0
                    from_3d = True
                else:
                    # 검출을 만든 **그** 카메라의 파라미터로 역투영한다.
                    # 기준 카메라로 풀면 후방·측방 검출의 yaw 가 어긋난다.
                    ego_xy, quality = src.model_for(det.camera).bbox_to_ego(
                        det.bbox, det.cls, self.cfg.perception.class_size
                    )
                    from_3d = False
                dist = math.hypot(*ego_xy)
                if dist > self.cfg.fusion.max_range_m:
                    self.dropped_far += 1  # 신뢰불가 원거리 관측 폐기
                    continue
                if det.world_xy_measured is not None:
                    # 3D 센서가 절대 좌표를 준 경우. 2D 회전으로 되돌리지
                    # 않는다 — 경사로에서 pitch/roll 성분이 빠져 어긋난다.
                    world_xy = det.world_xy_measured
                else:
                    world_xy = ego_to_world(ego_xy, origin, samp.heading_deg)
                observations.append(
                    Observation(
                        t=t,
                        observer_id=vid,
                        local_track_id=det.track_id,
                        cls=det.cls,
                        conf=det.conf,
                        ego_xy=ego_xy,
                        world_xy=world_xy,
                        distance_m=dist,
                        bbox=det.bbox,
                        range_quality=quality,
                        heading_deg=det.heading_measured,
                        speed_mps=det.speed_measured,
                        from_3d_sensor=from_3d,
                        measured_t=det.t,
                        camera=det.camera,
                    )
                )

        if not ego_states:
            return None

        roles = {vid: src.role for vid, src in self.sources.items()}
        self_ids = {
            vid: src.self_track_id
            for vid, src in self.sources.items()
            if src.self_track_id is not None
        }
        actors = fuse_observations(
            t,
            observations,
            ego_states,
            ego_world,
            self.registry,
            self.cfg.fusion,
            observer_roles=roles,
            observer_self_ids=self_ids or None,
        )

        # 순서가 중요하다:
        #   1) 임시 매칭 (주변차량은 아직 방위각 미상)
        #   2) 궤적으로 속도·방위각 추정
        #   3) 방위각을 반영해 재매칭 → 진행방향/차선 확정
        #   4) 확정된 매칭으로 횡오프셋 기록 및 기동 분류
        #   5) 경로 예측
        for a in actors:
            a.placement = self.network.locate(a.world_xy, a.heading_deg)
        for a in actors:
            self.kinematics.update(t, a)
        for a in actors:
            # 관측차량(ego)도 재매칭한다. 1차 매칭(위 locate)은 방위를 넘기지만
            # trust_heading_direction 없이 부르므로, 일방통행 도로에서 차도 폭을
            # 벗어나 있으면 '방향 미확정 + 폴리라인 정방향'으로 떨어져 실측
            # 방위와 180° 어긋난 진행방향 라벨이 나간다. ego 는 텔레메트리 방위를
            # 가진 가장 정확한 차량인데 그 근거를 안 쓰고 있었다.
            self._refine_placement(a)
            self._adopt_road_heading(a)
            self.kinematics.finalize(t, a)
            a.predictions = predict(
                a,
                self.network,
                self.cfg.prediction_horizon_s,
                predictor=self.cfg.predictor,
                neighbors=actors,
            )

        self.kinematics.prune(t)
        interactions = analyze_interactions(actors)

        # 노변 인프라는 교통 참여자가 아니므로 별도 목록으로 낸다
        infra: List[InfraState] = []
        n_obs_by_observer: Dict[str, int] = {}
        for o in observations:
            n_obs_by_observer[o.observer_id] = (
                n_obs_by_observer.get(o.observer_id, 0) + 1
            )
        for vid, src in self.sources.items():
            if not src.is_infrastructure or vid not in ego_world:
                continue
            infra.append(
                InfraState(
                    infra_id=vid,
                    world_xy=ego_world[vid],
                    height_m=src.mount_height_m or src.camera.height_m,
                    heading_deg=ego_states[vid].heading_deg,
                    n_observed=n_obs_by_observer.get(vid, 0),
                    # 인프라 방위도 설치 시 측정한 값이므로 신뢰한다
                    placement=self.network.locate(
                        ego_world[vid],
                        ego_states[vid].heading_deg,
                        trust_heading_direction=True,
                    ),
                )
            )

        inferred_roads = sum(1 for r in self.network.roads.values() if r.inferred)
        map_ctx: Dict[str, object] = {
            "road_count": len(self.network.roads),
            "junction_count": sum(
                1 for j in self.network.junctions.values() if j.is_intersection
            ),
            "observer_ids": sorted(
                v for v, s in self.sources.items() if not s.is_infrastructure
            ),
            "infrastructure_ids": sorted(
                v for v, s in self.sources.items() if s.is_infrastructure
            ),
            "lane_numbering": self.cfg.lane.numbering,
            "drive_side": self.cfg.lane.drive_side,
            "lane_width_m": self.cfg.lane.lane_width_m,
            # 관측자별 카메라 구성. "관측되지 않았다"의 의미가 카메라 배치에
            # 따라 다르다 — 전방 1대뿐인 관측자가 뒤차를 못 보는 것은 당연하고,
            # 360° 를 덮는 관측자가 못 보는 것은 가려짐이라는 뜻이다.
            "observer_cameras": {
                vid: self._camera_coverage(src)
                for vid, src in sorted(self.sources.items())
            },
        }
        if inferred_roads:
            map_ctx["inferred_roads"] = inferred_roads
            # 산문이 아니라 **키**를 저장한다 (i18n.MAP_SOURCES 에서 문장을 만든다)
            map_ctx["map_source"] = "inferred_from_trajectories"

        return SceneSnapshot(
            t=t,
            actors=actors,
            interactions=interactions,
            area_name=self.cfg.area_name,
            map_context=map_ctx,
            infrastructure=infra,
            scenario=self.scenario,
            frame_idx=self._frame_of(t),
        )

    @staticmethod
    def _camera_coverage(src: VehicleSource) -> dict:
        """관측자의 카메라 구성 요약: 대수, 방위 커버리지, 화각 합.

        커버리지는 카메라별 [yaw − hfov/2, yaw + hfov/2] 구간을 1° 격자에
        합집합으로 채워 센다. 화각이 겹치면 중복 계산되지 않는다 —
        단순히 hfov 를 더하면 6대 × 70° = 420° 처럼 360° 를 넘는다.
        """
        covered = set()
        for c in src.cameras:
            hfov = 2.0 * math.degrees(math.atan((c.width / 2.0) / c.fx))
            lo = c.yaw_deg - hfov / 2.0
            for k in range(int(round(hfov))):
                covered.add(int(round(lo + k)) % 360)
        return {
            "n_cameras": len(src.cameras),
            "names": [c.name for c in src.cameras],
            # 전방 0°, 좌측 +, 우측 −, 후방 180° 기준으로 덮는 방위 범위 [deg]
            "azimuth_coverage_deg": len(covered),
        }

    def _frame_of(self, t: float) -> Optional[int]:
        """시각 t 에 가장 가까운 원본 프레임 번호 (추적성 확보용)."""
        best: Optional[int] = None
        best_d = float("inf")
        for src in self.sources.values():
            for ts, dets in src._det_by_time.items():
                d = abs(ts - t)
                if d < best_d and dets:
                    best_d, best = d, dets[0].frame_idx
        return best
