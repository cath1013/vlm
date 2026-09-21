"""DeepAccident 시나리오 → 교통 상황 스냅샷 스트림.

어댑터(deepaccident) + 도로망 합성(roadgen) + 파이프라인(pipeline) 을 묶어,
DeepAccident 시나리오 하나를 LLM 입력 스냅샷 시퀀스로 바꾼다.

    runner = DeepAccidentRunner(root, cfg)
    result = runner.build("Town01", "type1_subtype1_normal")
    for snap in result.snapshots():
        print(to_text(snap, cfg.serialize))

지도 우선순위
    1) road_map_path 를 주면 그 GeoJSON 을 사용
    2) opendrive_path 를 주면 CARLA .xodr 을 사용
    3) 둘 다 없으면 관측 궤적에서 합성 (DeepAccident 는 지도 파일을 포함하지
       않으므로 기본 경로)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .config import LaneConfig, PipelineConfig
from .deepaccident import (
    AGENT_ROLE_KO,
    CLASS_SIZES,
    DeepAccidentPerception,
    DeepAccidentScenario,
    camera_config_from_calib,
    estimate_ground_z_multiframe,
    find_scenario,
    ground_truth_tracks,
    load_calib,
    parse_label_file,
    read_poses,
    resolve_cameras,
    scenario_actor_classes,
    scan_scenarios,
    synthesize_telemetry,
)
from .geometry import LocalENU
from .pipeline import TrafficSceneConverter
from .roadgen import RoadGenReport, synthesize_road_network
from .roadmap import RoadNetwork
from .schemas import SceneSnapshot


@dataclass
class BuildResult:
    """시나리오 1개에 대한 구성 결과."""

    scenario: DeepAccidentScenario
    converter: TrafficSceneConverter
    network: RoadNetwork
    roadgen_report: Optional[RoadGenReport]
    map_source: str
    perception_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    n_detections: Dict[str, int] = field(default_factory=dict)
    # 관측자별로 등록한 카메라 이름 (기준 카메라가 맨 앞)
    cameras: Dict[str, List[str]] = field(default_factory=dict)

    def snapshots(
        self, rate_hz: Optional[float] = None, **kw
    ) -> Iterator[SceneSnapshot]:
        return self.converter.run(rate_hz=rate_hz, **kw)

    def summary(self) -> str:
        det = ", ".join(f"{k}:{v}" for k, v in sorted(self.n_detections.items()))
        s = (
            f"{self.scenario.scenario_id}\n"
            f"  지도: {self.map_source} (도로 {len(self.network.roads)}개, "
            f"교차로 {sum(1 for j in self.network.junctions.values() if j.is_intersection)}개)\n"
            f"  관측자: {len(self.converter.sources)}기 | 검출 {det}"
        )
        if self.roadgen_report:
            s += f"\n  합성: {self.roadgen_report.summary()}"
        return s


class DeepAccidentRunner:
    def __init__(self, root: str, cfg: Optional[PipelineConfig] = None):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"DeepAccident 루트가 없습니다: {root}")
        self.root = root
        self.cfg = cfg or PipelineConfig()
        # 데이터셋 실측 차선폭을 파이프라인 전역에 반영
        self.cfg.lane.lane_width_m = self.cfg.deepaccident.lane_width_m
        # DeepAccident 의 로컬 트랙 id 는 CARLA actor id 로 전 관측자에서 동일하다.
        # 이제 기본값과 같지만(FusionConfig.global_track_ids=True) 데이터셋의
        # 성질이므로 명시해 둔다 — 기본값이 바뀌어도 이 경로는 영향받지 않는다.
        self.cfg.fusion.global_track_ids = True
        # 데이터셋 클래스에 맞춘 치수 사전값과 대상 클래스
        self.cfg.perception.class_size = dict(CLASS_SIZES)
        classes = ["car", "truck", "van", "bus", "motorcycle", "bicycle"]
        if self.cfg.deepaccident.include_pedestrians:
            classes.append("person")
        self.cfg.perception.vehicle_classes = tuple(classes)
        self.enu = LocalENU(*self.cfg.deepaccident.geo_origin)

    # ------------------------------------------------------------ 조회

    def list_scenarios(self, **kw) -> List[DeepAccidentScenario]:
        return scan_scenarios(self.root, self.cfg.deepaccident, **kw)

    # ------------------------------------------------------------ 도로망

    def build_network(
        self,
        scenario: DeepAccidentScenario,
        road_map_path: Optional[str] = None,
        opendrive_path: Optional[str] = None,
    ) -> Tuple[RoadNetwork, Optional[RoadGenReport], str]:
        lane_cfg = self.cfg.lane
        if road_map_path:
            net = RoadNetwork.from_geojson(
                road_map_path, lane_cfg, origin=self.cfg.deepaccident.geo_origin
            )
            return net, None, f"GeoJSON ({os.path.basename(road_map_path)})"
        if opendrive_path:
            from .carla_map import load_opendrive

            net = load_opendrive(opendrive_path, lane_cfg, self.enu)
            return net, None, f"OpenDRIVE ({os.path.basename(opendrive_path)})"

        tracks = self.collect_tracks(scenario)
        net, rep = synthesize_road_network(
            tracks,
            lane_cfg,
            self.cfg.roadgen,
            self.enu,
            expected_road_type=scenario.meta.road_type,
        )
        return net, rep, "궤적 합성 (차선 수는 관측 하한)"

    def collect_tracks(
        self, scenario: DeepAccidentScenario
    ) -> Dict[str, List[Tuple[float, float]]]:
        """도로망 합성 재료: 관측자 자세 + 레이블 객체 궤적 (ENU)."""
        cfg = self.cfg.deepaccident
        tracks: Dict[str, List[Tuple[float, float]]] = {}
        for ag, series in scenario.agents.items():
            if series.is_static:
                continue  # 정지 센서는 궤적이 없다
            poses = read_poses(scenario, ag, cfg)
            tracks[f"agent:{ag}"] = [poses[f].world_xy for f in sorted(poses)]

        ref = (
            "ego_vehicle"
            if "ego_vehicle" in scenario.agents
            else next(iter(scenario.agents))
        )
        gt = ground_truth_tracks(scenario, cfg, reference_agent=ref)
        per: Dict[int, List[Tuple[float, float]]] = {}
        for f in sorted(gt):
            for oid, xy in gt[f].items():
                per.setdefault(oid, []).append(xy)
        for oid, pts in per.items():
            if len(pts) >= 5:
                tracks[f"obj:{oid}"] = pts
        return tracks

    # ------------------------------------------------------------ 구성

    def build(
        self,
        scenario: str,
        scenario_type: Optional[str] = None,
        road_map_path: Optional[str] = None,
        opendrive_path: Optional[str] = None,
        frames: Optional[Sequence[int]] = None,
    ) -> BuildResult:
        da = self.cfg.deepaccident
        sc = find_scenario(self.root, scenario, scenario_type, da)

        net, rep, map_source = self.build_network(sc, road_map_path, opendrive_path)
        self.cfg.area_name = f"{sc.town} ({sc.meta.road_type or '도로형태 미기재'})"

        backend = DeepAccidentPerception(sc, da)
        conv = TrafficSceneConverter(self.cfg, net, backend)
        conv.scenario = sc.meta.to_context()

        result = BuildResult(
            scenario=sc,
            converter=conv,
            network=net,
            roadgen_report=rep,
            map_source=map_source,
        )

        common = list(frames) if frames is not None else sc.frames()
        actor_classes = scenario_actor_classes(sc, frames=common)
        for ag, series in sc.agents.items():
            f0 = common[0] if common else series.frames[0]
            cal = load_calib(series.calib_paths[f0])
            # 노면 고도를 레이블 박스 밑면에서 추정해 지상고를 맞춘다.
            # (인프라는 높이가 ego_to_world 에 있어 사슬 전체가 필요하고,
            #  자기 행이 없어 여러 프레임을 모아야 안정적이다)
            ground_z = estimate_ground_z_multiframe(sc, ag, frames=common)
            # 카메라 **전부**를 등록한다. 검출은 어느 카메라의 것인지
            # (`Detection.camera`) 달고 오므로, 역투영에 쓸 파라미터를
            # 파이프라인이 카메라별로 고를 수 있어야 한다.
            names = resolve_cameras(cal, da.cameras)
            cams = [
                camera_config_from_calib(cal, n, ground_z=ground_z, name=n)
                for n in names
            ]
            # 기준 카메라를 목록 맨 앞에 둔다 — 설치높이 표시 등 1대만
            # 필요한 곳이 이것을 쓴다.
            cams.sort(key=lambda c: (c.name != da.camera, c.name))
            result.cameras[ag] = [c.name for c in cams]
            tele = synthesize_telemetry(sc, ag, da, self.enu, frames=common)
            dets = backend.detections_for(ag, t0=0.0, frames=common)
            result.n_detections[ag] = len(dets)

            if series.is_static:
                conv.add_infrastructure(
                    ag,
                    camera=cams,
                    telemetry=tele,
                    detections=dets,
                    mount_height_m=cams[0].height_m,
                )
            else:
                conv.add_vehicle(
                    ag,
                    video_path=f"<{ag}>",
                    telemetry_path="",
                    camera=cams,
                    detections=dets,
                    telemetry=tele,
                    # CARLA actor id 를 알려주면 다른 관측자가 이 차량을 검출한
                    # 것을 거리 게이트 없이 정확히 흡수할 수 있다
                    self_track_id=sc.meta.agent_id_of(ag),
                    self_class=actor_classes.get(sc.meta.agent_id_of(ag)),
                )
        result.perception_stats["all"] = dict(backend.stats)
        return result

    def agent_label(self, agent: str) -> str:
        return AGENT_ROLE_KO.get(agent, agent)
