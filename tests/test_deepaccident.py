"""DeepAccident 어댑터 / 도로망 합성 / OpenDRIVE 임포트 테스트.

데이터셋이 없는 환경에서도 대부분 실행된다. 실제 데이터가 필요한 테스트는
DEEPACCIDENT_ROOT 환경변수(또는 아래 기본 경로)가 있을 때만 실행되고,
없으면 skip 된다.

    set DEEPACCIDENT_ROOT=C:\\path\\to\\DeepAccident_mini
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from traffic_llm.carla_map import (
    CARLA_TOWNS,
    Geometry,
    load_opendrive,
    opendrive_summary,
    town_description,
)
from traffic_llm.config import (
    DeepAccidentConfig,
    LaneConfig,
    PipelineConfig,
    RoadGenConfig,
)
from traffic_llm.deepaccident import (
    CLASS_SIZES,
    SELF_ID,
    UNTRACKED_ID,
    camera_config_from_calib,
    camera_world_z,
    carla_matrix_to_enu_heading,
    carla_to_enu,
    estimate_ground_z,
    parse_label_file,
    parse_meta,
)
from traffic_llm.geometry import LocalENU
from traffic_llm.roadgen import (
    build_corridors,
    split_group_by_lateral_gaps,
    split_track_into_segments,
    synthesize_road_network,
)

DA_ROOT = os.environ.get(
    "DEEPACCIDENT_ROOT", r"C:/Users/ylim/Downloads/DeepAccident_mini"
)
HAS_DATA = os.path.isdir(DA_ROOT)
skip_no_data = unittest.skipUnless(
    HAS_DATA, f"DeepAccident 데이터가 없습니다 ({DA_ROOT})"
)


# ---------------------------------------------------------------- 포맷 파싱


class TestFormatParsing(unittest.TestCase):
    def test_parse_label_fields(self):
        txt = (
            "0.0034441 -5.6166797\n"
            "car -38.698 -19.789 -0.882 4.181 1.994 1.381 -0.000518 0.0 0.0 20001 0 True\n"
            "car -0.0026 -1.4e-05 -1.2216 4.674 1.812 1.442 0.0 0.00344 -5.6167 -100 1394 False\n"
            "pedestrian 21.8 -1.46 -0.84 0.375 0.375 1.86 2.19 0.0 0.0 -1 0 True\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                        encoding="utf-8") as f:
            f.write(txt)
            p = f.name
        try:
            lf = parse_label_file(p)
        finally:
            os.unlink(p)

        self.assertAlmostEqual(lf.ego_speed_world[0], 0.0034441)
        self.assertAlmostEqual(lf.ego_speed_world[1], -5.6166797)
        self.assertEqual(len(lf.objects), 3)

        o = lf.objects[0]
        self.assertEqual(o.cls_raw, "car")
        self.assertAlmostEqual(o.x, -38.698)
        self.assertAlmostEqual(o.length, 4.181)
        self.assertAlmostEqual(o.width, 1.994)
        self.assertAlmostEqual(o.height, 1.381)
        self.assertEqual(o.obj_id, 20001)
        self.assertEqual(o.n_lidar_pts, 0)
        self.assertTrue(o.camera_visible)

        self.assertTrue(lf.objects[1].is_self)
        self.assertEqual(lf.self_object.obj_id, SELF_ID)
        self.assertTrue(lf.objects[2].is_untracked)
        self.assertEqual(lf.objects[2].obj_id, UNTRACKED_ID)
        # pedestrian → person 으로 매핑
        self.assertEqual(lf.objects[2].cls, "person")

    def test_parse_meta_accident(self):
        txt = (
            "WetNoon 1334 car 1342 truck 14599.48 same front 55\n"
            " colliding agents: ego none\n"
            " agents id: 1334 1335 1333 1338\n"
            " road_type: three-way junction\n"
            " another_vehicle_spawn_side: left\n"
            " ego_vehicle_direction: straight\n"
            " other_vehicle_direction: left\n"
        )
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "Town02_type001_subtype0001_scenario00013.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(txt)
            m = parse_meta(p, "type1_subtype1_accident", n_frames=45)

        self.assertEqual(m.town, "Town02")
        self.assertEqual(m.weather, "WetNoon")
        self.assertTrue(m.is_accident_split)
        self.assertTrue(m.collision_occurred)
        self.assertEqual(m.collision_id_a, 1334)
        self.assertEqual(m.collision_cls_b, "truck")
        self.assertAlmostEqual(m.collision_intensity, 14599.48)
        self.assertEqual(m.colliding_agents, ("ego",))
        self.assertEqual(m.agent_ids, (1334, 1335, 1333, 1338))
        self.assertEqual(m.road_type, "three-way junction")
        self.assertEqual(m.agent_id_of("ego_vehicle"), 1334)
        self.assertEqual(m.agent_id_of("other_vehicle_behind"), 1338)
        self.assertIsNone(m.agent_id_of("infrastructure"))
        # n_sim = 저장 프레임 + 워밍업 10
        self.assertEqual(m.n_simulated_frames, 55)

    def test_parse_meta_normal_has_no_collision(self):
        txt = (
            "ClearNoon -1 -1 -1 -1 -1 -1 -1 110\n"
            " colliding agents: none none\n"
            " agents id: 384 395 382 396\n"
            " road_type: three-way junction\n"
        )
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "Town01_x.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(txt)
            m = parse_meta(p, "type1_subtype1_normal", n_frames=100)
        self.assertFalse(m.collision_occurred)
        self.assertFalse(m.is_accident_split)
        self.assertEqual(m.colliding_agents, ())
        self.assertIsNone(m.collision_id_a)

    def test_ground_truth_leak_is_separated(self):
        """사고 정답은 available 이 아니라 ground_truth 에만 있어야 한다."""
        txt = (
            "WetNoon 1334 car 1342 truck 14599.48 same front 55\n"
            " colliding agents: ego none\n"
            " agents id: 1334 1335 1333 1338\n"
            " road_type: three-way junction\n"
        )
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "Town02_x.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(txt)
            ctx = parse_meta(p, "type1_subtype1_accident", 45).to_context()

        avail = " ".join(f"{k}={v}" for k, v in ctx.available.items())
        for leak in ("collision", "ego", "1334", "front"):
            self.assertNotIn(leak, avail, f"정답 누수: {leak}")
        self.assertTrue(ctx.ground_truth["collision_occurred"])
        self.assertEqual(ctx.ground_truth["collision_frame_approx"], 45)
        self.assertIn("weather", ctx.available)
        self.assertIn("road_type", ctx.available)


# ---------------------------------------------------------------- 좌표 변환


class TestCarlaCoords(unittest.TestCase):
    def test_carla_to_enu_flips_y(self):
        self.assertEqual(carla_to_enu(10.0, 5.0), (10.0, -5.0))
        self.assertEqual(carla_to_enu(-3.0, -7.0), (-3.0, 7.0))

    def test_heading_from_matrix(self):
        # CARLA x축(전방)이 월드 -y (=ENU 북) → 방위각 0
        R = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertAlmostEqual(carla_matrix_to_enu_heading(R), 0.0, places=6)
        # x축이 월드 +x (=ENU 동) → 90도
        R = np.eye(3)
        self.assertAlmostEqual(carla_matrix_to_enu_heading(R), 90.0, places=6)
        # x축이 월드 +y (=ENU 남) → 180도
        R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertAlmostEqual(carla_matrix_to_enu_heading(R), 180.0, places=6)

    def _fake_calib(self, ego_z: float, lidar_z: float, cam_dz: float):
        """차량형(높이가 lidar_to_ego) / 인프라형(높이가 ego_to_world) 모두 생성."""
        E2W = np.eye(4)
        E2W[2, 3] = ego_z
        L2E = np.eye(4)
        L2E[2, 3] = lidar_z
        L2C = np.eye(4)
        L2C[0, 3] = -2.3368  # 카메라가 라이다보다 2.34m 전방
        L2C[2, 3] = cam_dz
        K = np.array([[800.0, 1142.5184, 0.0], [450.0, 0.0, -1142.5184],
                      [1.0, 0.0, 0.0]])
        return {
            "ego_to_world": E2W,
            "lidar_to_ego": L2E,
            "lidar_to_Camera_Front": L2C,
            "intrinsic_Camera_Front": K,
        }

    def test_camera_world_z_uses_full_chain(self):
        # 차량: 높이가 lidar_to_ego 에 있다
        c = self._fake_calib(ego_z=0.0, lidar_z=1.9419, cam_dz=0.3)
        self.assertAlmostEqual(camera_world_z(c), 1.9419 - 0.3, places=4)
        # 인프라: lidar_to_ego 가 항등, 높이가 ego_to_world 에 있다
        c = self._fake_calib(ego_z=3.679, lidar_z=0.0, cam_dz=0.0)
        self.assertAlmostEqual(camera_world_z(c), 3.679, places=4)

    def test_camera_config_intrinsics_and_pose(self):
        c = self._fake_calib(0.0, 1.9419, 0.3)
        cam = camera_config_from_calib(c, "Camera_Front", ground_z=0.0)
        self.assertAlmostEqual(cam.fx, 1142.5184, places=3)
        self.assertAlmostEqual(cam.fy, 1142.5184, places=3)
        self.assertAlmostEqual(cam.cx, 800.0)
        self.assertAlmostEqual(cam.cy, 450.0)
        self.assertAlmostEqual(cam.height_m, 1.6419, places=3)
        self.assertAlmostEqual(cam.yaw_deg, 0.0, places=6)
        self.assertAlmostEqual(cam.pitch_deg, 0.0, places=6)

    def test_infrastructure_height_not_zero(self):
        """인프라 지상고를 lidar_to_ego 만으로 구하면 0 이 되어 IPM 이 붕괴한다."""
        c = self._fake_calib(ego_z=3.679, lidar_z=0.0, cam_dz=0.0)
        cam = camera_config_from_calib(c, "Camera_Front", ground_z=0.0)
        self.assertGreater(cam.height_m, 3.0)

    def test_camera_model_rejects_zero_height(self):
        from traffic_llm.config import CameraConfig
        from traffic_llm.geometry import CameraModel

        bad = CameraConfig.from_fov(1600, 900, hfov_deg=70.0, height_m=0.0)
        with self.assertRaises(ValueError):
            CameraModel(bad)

    def test_estimate_ground_z_prefers_self_row(self):
        """자기 행이 있으면 원거리 객체 고도에 끌려가지 않아야 한다."""
        c = self._fake_calib(ego_z=3.0, lidar_z=1.94, cam_dz=0.3)
        txt = (
            "0 0\n"
            # 자기 자신: 박스 밑면이 노면 = ego z(3.0).
            # 라이다가 1.94m 위이므로 중심은 -1.94 + 높이/2 = -1.19
            "car 0 0 -1.19 4.6 1.9 1.5 0 0 0 -100 100 False\n"
            # 원거리에 8m 높은 객체 (고가도로)
            "car 100 0 6.0 4.6 1.9 1.5 0 0 0 501 5 True\n"
            "car 110 0 6.0 4.6 1.9 1.5 0 0 0 502 5 True\n"
            "car 115 0 6.0 4.6 1.9 1.5 0 0 0 503 5 True\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                        encoding="utf-8") as f:
            f.write(txt)
            p = f.name
        try:
            lf = parse_label_file(p)
        finally:
            os.unlink(p)
        gz = estimate_ground_z(c, lf)
        self.assertAlmostEqual(gz, 3.0, delta=0.3)

    def test_class_sizes_are_plausible(self):
        for name, s in CLASS_SIZES.items():
            self.assertGreater(s.width_m, 0.0, name)
            self.assertGreater(s.length_m, 0.0, name)
            self.assertGreater(s.height_m, 0.0, name)
        self.assertGreater(CLASS_SIZES["truck"].length_m,
                           CLASS_SIZES["car"].length_m)
        self.assertLess(CLASS_SIZES["person"].width_m,
                        CLASS_SIZES["car"].width_m)


# ---------------------------------------------------------------- 도로망 합성


class TestRoadGen(unittest.TestCase):
    LW = 3.5

    def _straight(self, e: float, n0: float, n1: float, step: float = 4.0):
        n = int(abs(n1 - n0) / step)
        sgn = 1 if n1 > n0 else -1
        return [(e, n0 + sgn * step * k) for k in range(n + 1)]

    def _ew(self, n: float, e0: float, e1: float, step: float = 4.0):
        k = int(abs(e1 - e0) / step)
        sgn = 1 if e1 > e0 else -1
        return [(e0 + sgn * step * i, n) for i in range(k + 1)]

    def test_split_track_at_turn(self):
        # 북진 후 동진으로 꺾이는 궤적 → 2개 구간
        pts = self._straight(0.0, -60.0, 0.0) + self._ew(0.0, 0.0, 60.0)
        segs = split_track_into_segments("t1", pts, RoadGenConfig())
        self.assertEqual(len(segs), 2)
        self.assertAlmostEqual(segs[0].bearing_deg, 0.0, delta=8)
        self.assertAlmostEqual(segs[1].bearing_deg, 90.0, delta=8)

    def test_two_way_road_lane_counts(self):
        """왕복 1차선씩 관측 → lanes 1/1, 양방향."""
        tracks = {
            "a": self._straight(-0.5 * self.LW, -80.0, 80.0),   # 북행
            "b": self._straight(0.5 * self.LW, 80.0, -80.0),    # 남행
        }
        net, rep = synthesize_road_network(
            tracks, LaneConfig(lane_width_m=self.LW), RoadGenConfig()
        )
        self.assertEqual(len(net.roads), 1)
        r = next(iter(net.roads.values()))
        self.assertEqual((r.lanes_forward, r.lanes_backward), (1, 1))
        self.assertFalse(r.oneway)
        self.assertTrue(r.inferred, "합성 도로는 inferred 로 표시되어야 한다")
        self.assertEqual(r.observed_directions, 2)

    def test_one_direction_only_marked_oneway_with_note(self):
        tracks = {"a": self._straight(0.0, -80.0, 80.0)}
        net, _ = synthesize_road_network(
            tracks, LaneConfig(lane_width_m=self.LW), RoadGenConfig()
        )
        r = next(iter(net.roads.values()))
        self.assertTrue(r.oneway)
        self.assertEqual(r.observed_directions, 1)
        self.assertIn("미관측", r.notes)

    def test_multilane_same_direction(self):
        tracks = {
            "a": self._straight(-0.5 * self.LW, -80.0, 80.0),
            "b": self._straight(-1.5 * self.LW, -80.0, 80.0),
            "c": self._straight(0.5 * self.LW, 80.0, -80.0),
        }
        net, _ = synthesize_road_network(
            tracks, LaneConfig(lane_width_m=self.LW), RoadGenConfig()
        )
        r = max(net.roads.values(), key=lambda z: z.length_m)
        self.assertEqual(r.lanes_forward, 2)
        self.assertEqual(r.lanes_backward, 1)

    def test_within_lane_wander_is_not_a_second_lane(self):
        """같은 차선 안의 흔들림(±0.8m)을 별도 차선으로 세면 안 된다."""
        base = self._straight(0.0, -80.0, 80.0)
        wobble = [(0.8 if i % 2 else -0.8, p[1]) for i, p in enumerate(base)]
        net, _ = synthesize_road_network(
            {"a": base, "b": wobble},
            LaneConfig(lane_width_m=self.LW),
            RoadGenConfig(),
        )
        r = max(net.roads.values(), key=lambda z: z.length_m)
        self.assertEqual(r.lanes_forward, 1)

    def test_parallel_far_road_is_separate(self):
        """20m 떨어진 나란한 도로는 같은 도로로 묶이면 안 된다."""
        tracks = {
            "a": self._straight(0.0, -80.0, 80.0),
            "b": self._straight(20.0, -80.0, 80.0),
        }
        net, _ = synthesize_road_network(
            tracks, LaneConfig(lane_width_m=self.LW), RoadGenConfig()
        )
        self.assertGreaterEqual(len(net.roads), 2)
        for r in net.roads.values():
            self.assertLessEqual(r.lanes_forward, 1)

    def test_junction_detected_and_roads_split(self):
        tracks = {
            "ns_a": self._straight(-0.5 * self.LW, -90.0, 90.0),
            "ns_b": self._straight(0.5 * self.LW, 90.0, -90.0),
            "ew_a": self._ew(-0.5 * self.LW, -90.0, 90.0),
            "ew_b": self._ew(0.5 * self.LW, 90.0, -90.0),
        }
        net, rep = synthesize_road_network(
            tracks, LaneConfig(lane_width_m=self.LW), RoadGenConfig()
        )
        self.assertGreaterEqual(rep.n_junctions, 1)
        # 교차로에서 분할되어 도로가 4개 이상
        self.assertGreaterEqual(len(net.roads), 4)

    def test_localization_on_synthesized_map(self):
        tracks = {
            "a": self._straight(-0.5 * self.LW, -80.0, 80.0),
            "b": self._straight(0.5 * self.LW, 80.0, -80.0),
        }
        net, _ = synthesize_road_network(
            tracks, LaneConfig(lane_width_m=self.LW), RoadGenConfig()
        )
        p = net.locate((-0.5 * self.LW, 0.0), 0.0)
        self.assertIsNotNone(p)
        self.assertEqual(p.lane_index, 1)
        self.assertEqual(p.direction_label, "북행")

    def test_empty_tracks_raises(self):
        with self.assertRaises(ValueError):
            synthesize_road_network({}, LaneConfig(), RoadGenConfig())

    def test_lateral_gap_split(self):
        from traffic_llm.roadgen import RoadGroup, _fit_axis

        pts_a = self._straight(0.0, -50.0, 50.0)
        pts_b = self._straight(3.5, -50.0, 50.0)
        pts_c = self._straight(20.0, -50.0, 50.0)
        cors = build_corridors(
            [s for tid, p in (("a", pts_a), ("b", pts_b), ("c", pts_c))
             for s in split_track_into_segments(tid, p, RoadGenConfig())],
            RoadGenConfig(),
        )
        self.assertEqual(len(cors), 3)
        origin, d = _fit_axis(pts_a, 0.0)
        g = RoadGroup(axis_origin=origin, axis_dir=d, corridors_fwd=list(cors))
        parts = split_group_by_lateral_gaps(g, 1.6 * self.LW)
        self.assertEqual(len(parts), 2, "20m 떨어진 차로는 분리되어야 한다")


# ---------------------------------------------------------------- OpenDRIVE

MINI_XODR = """<?xml version="1.0" standalone="yes"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="TestTown" north="0" south="0">
    <geoReference><![CDATA[+proj=tmerc +lat_0=0 +lon_0=0]]></geoReference>
  </header>
  <road name="Main" length="100.0" id="1" junction="-1">
    <type s="0"><speed max="50" unit="km/h"/></type>
    <planView>
      <geometry s="0" x="0" y="0" hdg="1.5707963" length="100.0"><line/></geometry>
    </planView>
    <lanes>
      <laneSection s="0">
        <left>
          <lane id="1" type="driving"><width sOffset="0" a="3.5"/></lane>
        </left>
        <center><lane id="0" type="none"/></center>
        <right>
          <lane id="-1" type="driving"><width sOffset="0" a="3.5"/></lane>
          <lane id="-2" type="driving"><width sOffset="0" a="3.5"/></lane>
          <lane id="-3" type="sidewalk"><width sOffset="0" a="2.0"/></lane>
        </right>
      </laneSection>
    </lanes>
  </road>
  <road name="Cross" length="80.0" id="2" junction="-1">
    <type s="0"><speed max="30" unit="mph"/></type>
    <planView>
      <geometry s="0" x="-40" y="50" hdg="0.0" length="80.0"><line/></geometry>
    </planView>
    <lanes>
      <laneSection s="0">
        <center><lane id="0" type="none"/></center>
        <right>
          <lane id="-1" type="driving"><width sOffset="0" a="3.5"/></lane>
        </right>
      </laneSection>
    </lanes>
  </road>
  <road name="Curve" length="50.0" id="3" junction="-1">
    <planView>
      <geometry s="0" x="0" y="200" hdg="0.0" length="50.0">
        <arc curvature="0.02"/>
      </geometry>
    </planView>
    <lanes>
      <laneSection s="0">
        <center><lane id="0" type="none"/></center>
        <right><lane id="-1" type="driving"><width sOffset="0" a="3.5"/></lane></right>
      </laneSection>
    </lanes>
  </road>
  <junction id="900" name="J"/>
</OpenDRIVE>
"""


class TestOpenDrive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "TestTown.xodr")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(MINI_XODR)

    def test_geometry_line(self):
        g = Geometry(s=0, x=0, y=0, hdg=math.pi / 2, length=10.0, kind="line")
        p = g.point_at(10.0)
        self.assertAlmostEqual(p[0], 0.0, places=6)
        self.assertAlmostEqual(p[1], 10.0, places=6)

    def test_geometry_arc_curves(self):
        g = Geometry(s=0, x=0, y=0, hdg=0.0, length=50.0, kind="arc",
                     curvature=0.02)
        p = g.point_at(50.0)
        # 곡률 0.02 → 반지름 50m, 50m 진행하면 1 라디안 회전
        self.assertAlmostEqual(p[0], 50.0 * math.sin(1.0), delta=0.2)
        self.assertAlmostEqual(p[1], 50.0 * (1 - math.cos(1.0)), delta=0.2)

    def test_geometry_arc_zero_curvature_is_line(self):
        g = Geometry(s=0, x=1, y=2, hdg=0.0, length=10.0, kind="arc",
                     curvature=0.0)
        p = g.point_at(10.0)
        self.assertAlmostEqual(p[0], 11.0, places=6)
        self.assertAlmostEqual(p[1], 2.0, places=6)

    def test_load_roads_and_lane_counts(self):
        net = load_opendrive(self.path, LaneConfig(), LocalENU(0.0, 0.0))
        self.assertEqual(len(net.roads), 3)
        main = net.roads["od_1"]
        # driving 만 세고 sidewalk 는 제외 → 정방향 2, 역방향 1
        self.assertEqual(main.lanes_forward, 2)
        self.assertEqual(main.lanes_backward, 1)
        self.assertFalse(main.oneway)
        self.assertFalse(main.inferred, "OpenDRIVE 도로는 추정이 아니다")
        self.assertAlmostEqual(main.speed_limit_kph, 50.0, places=3)

    def test_mph_speed_conversion(self):
        net = load_opendrive(self.path, LaneConfig(), LocalENU(0.0, 0.0))
        self.assertAlmostEqual(net.roads["od_2"].speed_limit_kph,
                               30 * 1.60934, places=3)

    def test_oneway_when_no_left_lanes(self):
        net = load_opendrive(self.path, LaneConfig(), LocalENU(0.0, 0.0))
        self.assertTrue(net.roads["od_2"].oneway)
        self.assertEqual(net.roads["od_2"].lanes_backward, 0)

    def test_lane_width_from_file(self):
        lc = LaneConfig(lane_width_m=3.25)
        load_opendrive(self.path, lc, LocalENU(0.0, 0.0))
        self.assertAlmostEqual(lc.lane_width_m, 3.5, places=3)

    def test_localization_on_opendrive_map(self):
        net = load_opendrive(self.path, LaneConfig(), LocalENU(0.0, 0.0))
        # Main 은 북행(hdg=π/2), 우측 차선은 동쪽(e>0)
        p = net.locate((1.75, 50.0), 0.0)
        self.assertIsNotNone(p)
        self.assertEqual(p.road_id, "od_1")
        self.assertEqual(p.direction_label, "북행")

    def test_junction_detected_between_crossing_roads(self):
        net = load_opendrive(self.path, LaneConfig(), LocalENU(0.0, 0.0))
        # Main(북행 0~100) 과 Cross(y=50, 동행) 가 (0,50) 에서 만난다.
        # 끝점 클러스터링 방식이므로 T자 접점은 Cross 의 끝점이 아니다 —
        # 여기서는 요약 정보만 확인한다.
        self.assertIsInstance(net.junctions, dict)

    def test_summary(self):
        s = opendrive_summary(self.path)
        self.assertIn("road 3", s)
        self.assertIn("junction 1", s)
        self.assertIn("TestTown", s)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_opendrive(os.path.join(self.tmp, "nope.xodr"))

    def test_town_registry(self):
        self.assertIn("Town01", CARLA_TOWNS)
        self.assertIn("Town10HD", CARLA_TOWNS)
        self.assertTrue(town_description("Town03"))
        self.assertEqual(town_description("NoSuchTown"), "")


# ---------------------------------------------------------------- 실데이터


@skip_no_data
class TestWithRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from traffic_llm.deepaccident import scan_scenarios

        cls.scenarios = scan_scenarios(DA_ROOT, DeepAccidentConfig())

    def test_scan_finds_scenarios_and_agents(self):
        self.assertGreater(len(self.scenarios), 0)
        s = self.scenarios[0]
        self.assertIn("ego_vehicle", s.agents)
        self.assertIn("infrastructure", s.agents)
        self.assertGreater(len(s.frames()), 10)

    def test_world_positions_agree_across_sensors(self):
        """같은 객체를 두 센서가 본 월드 좌표는 일치해야 한다."""
        from traffic_llm.deepaccident import ground_truth_tracks

        s = next(x for x in self.scenarios if "ego_vehicle" in x.agents
                 and "infrastructure" in x.agents)
        cfg = DeepAccidentConfig()
        frames = s.frames()[:12]
        a = ground_truth_tracks(s, cfg, "ego_vehicle", frames)
        b = ground_truth_tracks(s, cfg, "infrastructure", frames)
        compared = 0
        for f in frames:
            for oid in set(a[f]) & set(b[f]):
                d = math.dist(a[f][oid], b[f][oid])
                self.assertLess(d, 0.05, f"frame {f} id {oid}: {d:.3f}m 불일치")
                compared += 1
        self.assertGreater(compared, 20, "비교 표본이 너무 적다")

    def test_telemetry_synthesis_roundtrip(self):
        """합성 텔레메트리의 위경도를 ENU 로 되돌리면 원 좌표와 같아야 한다.

        보고 위치는 자기 레이블 행의 **박스 중심**이다 (다른 모든 액터와 같은
        규약). ego_to_world 평행이동은 차량 기준점이라 박스 중심과 최대 0.3m
        어긋나므로, 그것과 비교하면 안 된다.
        """
        from traffic_llm.deepaccident import read_poses, synthesize_telemetry

        s = self.scenarios[0]
        cfg = DeepAccidentConfig()
        enu = LocalENU(*cfg.geo_origin)
        frames = s.frames()[:20]
        poses = read_poses(s, "ego_vehicle", cfg, frames)
        tel = synthesize_telemetry(s, "ego_vehicle", cfg, enu, frames)
        self.assertEqual(len(tel), len(frames))
        for samp, f in zip(tel, sorted(poses)):
            e, n = enu.to_enu(samp.lat, samp.lon)
            ref = poses[f].box_center_xy or poses[f].world_xy
            self.assertAlmostEqual(e, ref[0], places=2)
            self.assertAlmostEqual(n, ref[1], places=2)
            self.assertGreaterEqual(samp.heading_deg, 0.0)
            self.assertLess(samp.heading_deg, 360.0)

    def test_pose_box_centre_is_read(self):
        """주행 관측자는 자기 박스 중심을 읽어야 하고, 정지 센서는 없다."""
        from traffic_llm.deepaccident import read_poses

        s = self.scenarios[0]
        cfg = DeepAccidentConfig()
        f = s.frames()[0]
        p = read_poses(s, "ego_vehicle", cfg, [f])[f]
        self.assertIsNotNone(p.box_center_xy)
        # 차량 기준점과 박스 중심은 1m 안에서 다르다 (같은 차량, 다른 기준)
        self.assertLess(math.dist(p.box_center_xy, p.world_xy), 1.0)
        if "infrastructure" in s.agents:
            q = read_poses(s, "infrastructure", cfg, [f])[f]
            self.assertIsNone(q.box_center_xy)

    def test_all_camera_heights_plausible(self):
        """전 시나리오·전 관측자의 카메라 지상고가 물리적으로 타당해야 한다.

        런너와 동일하게 다중 프레임 노면 추정을 쓴다. 단일 프레임 추정은
        근거리 객체가 적은 장면(Town07 인프라)에서 흔들린다.
        """
        from traffic_llm.deepaccident import (
            estimate_ground_z_multiframe,
            load_calib,
        )

        checked = 0
        for s in self.scenarios:
            for ag, series in s.agents.items():
                f0 = series.frames[0]
                cal = load_calib(series.calib_paths[f0])
                gz = estimate_ground_z_multiframe(s, ag)
                cam = camera_config_from_calib(cal, "Camera_Front", ground_z=gz)
                self.assertGreater(cam.height_m, 1.0, f"{s.scenario}/{ag}")
                self.assertLess(cam.height_m, 8.0, f"{s.scenario}/{ag}")
                if series.is_static:
                    self.assertGreater(cam.height_m, 2.5,
                                       f"인프라 지상고가 너무 낮다: {ag}")
                checked += 1
        self.assertGreater(checked, 20)

    def test_sensor3d_mode_is_exact(self):
        """GT 3D 관측 모드에서는 좌표변환·융합 오차가 0 이어야 한다."""
        from traffic_llm.da_eval import evaluate_build
        from traffic_llm.da_runner import DeepAccidentRunner

        cfg = PipelineConfig()
        cfg.deepaccident.observation_mode = "sensor3d"
        r = DeepAccidentRunner(DA_ROOT, cfg)
        res = r.build(self.scenarios[0].scenario, self.scenarios[0].scenario_type)
        m = evaluate_build(res, cfg, rate_hz=2.0)
        self.assertGreater(m.n_matched, 20)
        errs = sorted(m.pos_err)
        self.assertLess(errs[len(errs) // 2], 0.05)
        self.assertLess(errs[int(0.9 * len(errs))], 0.2)

    def test_camera_mode_bias_is_small(self):
        """단안 모드의 반경 편향 중앙값이 ±1.5m 안이어야 한다."""
        from traffic_llm.da_eval import evaluate_build
        from traffic_llm.da_runner import DeepAccidentRunner

        cfg = PipelineConfig()
        cfg.deepaccident.observation_mode = "camera"
        r = DeepAccidentRunner(DA_ROOT, cfg)
        res = r.build(self.scenarios[0].scenario, self.scenarios[0].scenario_type)
        m = evaluate_build(res, cfg, rate_hz=2.0)
        self.assertGreater(m.n_matched, 10)
        bias = sorted(m.radial_bias)
        median = bias[len(bias) // 2]
        self.assertLess(abs(median), 1.5, f"반경 편향 과다: {median:+.2f}m")

    def test_infrastructure_is_not_a_traffic_actor(self):
        from traffic_llm.da_runner import DeepAccidentRunner

        cfg = PipelineConfig()
        r = DeepAccidentRunner(DA_ROOT, cfg)
        s = next(x for x in self.scenarios if "infrastructure" in x.agents)
        res = r.build(s.scenario, s.scenario_type)
        seen_infra = False
        for snap in res.snapshots(rate_hz=2.0):
            for a in snap.actors:
                self.assertNotIn(
                    "infrastructure", a.actor_id,
                    "인프라가 교통 참여자 목록에 들어갔다",
                )
            if snap.infrastructure:
                seen_infra = True
                for i in snap.infrastructure:
                    self.assertGreater(i.height_m, 2.0)
        self.assertTrue(seen_infra, "인프라 상태가 스냅샷에 없다")

    def test_ego_actors_not_overwritten_by_observations(self):
        """관측 클러스터가 ego actor id 를 가져가면 관측차량이 사라진다."""
        from traffic_llm.da_runner import DeepAccidentRunner

        cfg = PipelineConfig()
        r = DeepAccidentRunner(DA_ROOT, cfg)
        s = self.scenarios[0]
        res = r.build(s.scenario, s.scenario_type)
        n_vehicles = sum(
            1 for a in s.agents.values() if not a.is_static
        )
        for snap in res.snapshots(rate_hz=2.0):
            egos = [a for a in snap.actors if a.kind == "ego"]
            self.assertEqual(len(egos), n_vehicles,
                             f"t={snap.t}: 관측차량 수가 {len(egos)}")
            for a in snap.actors:
                if a.actor_id.startswith("EGO_"):
                    self.assertEqual(a.kind, "ego", a.actor_id)

    def test_unknown_speed_is_not_reported_as_stopped(self):
        """첫 관측 시점의 차량을 '정지'로 단정하지 않아야 한다."""
        from traffic_llm.da_runner import DeepAccidentRunner

        cfg = PipelineConfig()
        r = DeepAccidentRunner(DA_ROOT, cfg)
        s = self.scenarios[0]
        res = r.build(s.scenario, s.scenario_type)
        first = next(iter(res.snapshots(rate_hz=2.0)))
        for a in first.actors:
            if a.kind == "observed" and a.speed_mps is None:
                self.assertEqual(a.maneuver, "속도 미확정")
                self.assertTrue(
                    any("미확정" in p.maneuver for p in a.predictions),
                    "속도 미확정인데 경로를 단정했다",
                )

    def test_serialized_input_has_no_ground_truth(self):
        from traffic_llm.da_runner import DeepAccidentRunner
        from traffic_llm.serialize import to_evaluation_record, to_json

        cfg = PipelineConfig()
        r = DeepAccidentRunner(DA_ROOT, cfg)
        s = next(x for x in self.scenarios
                 if x.scenario_type.endswith("_accident")
                 and x.meta.collision_occurred)
        res = r.build(s.scenario, s.scenario_type)
        snap = next(iter(res.snapshots(rate_hz=2.0)))

        d = to_json(snap, cfg.serialize)
        self.assertIn("scenario", d)
        self.assertNotIn("ground_truth", d["scenario"])
        blob = str(d)
        self.assertNotIn("collision_occurred", blob)
        self.assertNotIn("colliding_agents", blob)

        # 정답은 평가 레코드에만
        ev = to_evaluation_record(snap, cfg.serialize)
        self.assertTrue(ev["ground_truth"]["collision_occurred"])

        # 요청 시에는 포함 가능
        d2 = to_json(snap, cfg.serialize, include_ground_truth=True)
        self.assertIn("ground_truth", d2["scenario"])

    def test_untracked_objects_excluded_by_default(self):
        """id=-1 은 프레임마다 다른 객체이므로 기본 제외해야 한다."""
        from traffic_llm.deepaccident import DeepAccidentPerception

        s = self.scenarios[0]
        p = DeepAccidentPerception(s, DeepAccidentConfig())
        dets = p.detections_for("ego_vehicle")
        self.assertGreater(p.stats["total_rows"], 0)
        for d in dets:
            self.assertNotEqual(d.track_id, UNTRACKED_ID)

    ATTR_DROPS = (
        "dropped_self",
        "dropped_untracked",
        "dropped_class",
        "dropped_range",
        "dropped_not_visible_any_cam",
        "dropped_lidar_pts",
    )

    def test_perception_filter_order_is_geometry_first(self):
        """가려짐 통계가 '화면 안에서 가려짐'을 뜻해야 한다."""
        from traffic_llm.deepaccident import DeepAccidentPerception

        s = self.scenarios[0]
        p = DeepAccidentPerception(s, DeepAccidentConfig(cameras=("Camera_Front",)))
        p.detections_for("ego_vehicle")
        st = p.stats
        # 전면 카메라 화각 밖 객체가 가려짐으로 분류되지 않았는지:
        # 기하 컬링 수가 가려짐 수보다 훨씬 커야 정상이다
        self.assertGreater(st["dropped_behind"], st["dropped_occluded"])

    def test_stats_account_for_every_label_row(self):
        """모든 레이블 행이 어딘가에 계상되어야 한다 (객체 단위 항등식).

        기하 필터(behind/fov/small/occluded)는 **(객체, 카메라) 쌍**마다 세므로
        카메라가 k대면 한 객체가 최대 k번 탈락한다. 따라서 기하 항을 더하면
        전체 행 수를 넘는다 — 객체 단위로 닫히는 항등식은
        `total_rows = 속성탈락 + kept + dropped_no_camera` 다.
        """
        from traffic_llm.deepaccident import DeepAccidentPerception

        s = self.scenarios[0]
        for cams in (("Camera_Front",), None):  # 1대 / 전부
            p = DeepAccidentPerception(s, DeepAccidentConfig(cameras=cams))
            p.detections_for("ego_vehicle")
            st = p.stats
            attr = sum(st[k] for k in self.ATTR_DROPS)
            self.assertEqual(
                st["total_rows"],
                attr + st["kept"] + st["dropped_no_camera"],
                f"통계 합이 전체 행 수와 맞지 않는다 (cameras={cams})",
            )
            self.assertGreater(st["kept"], 0)

    def test_all_cameras_see_more_than_front_alone(self):
        """카메라 6대가 전면 1대보다 더 많이 본다 — 다중 카메라의 요점."""
        from traffic_llm.deepaccident import DeepAccidentPerception

        s = self.scenarios[0]
        front = DeepAccidentPerception(s, DeepAccidentConfig(cameras=("Camera_Front",)))
        n_front = len(front.detections_for("ego_vehicle"))
        allcam = DeepAccidentPerception(s, DeepAccidentConfig())  # None = 전부
        n_all = len(allcam.detections_for("ego_vehicle"))
        self.assertEqual(front.stats["n_cameras"], 1)
        self.assertGreater(allcam.stats["n_cameras"], 1)
        self.assertGreater(n_all, n_front)
        # 전면만 썼다면 놓쳤을 관측이 실제로 있어야 한다
        self.assertGreater(allcam.stats["seen_only_by_non_primary"], 0)

    def test_detections_carry_their_camera(self):
        from traffic_llm.deepaccident import DeepAccidentPerception, available_cameras
        from traffic_llm.deepaccident import load_calib

        s = self.scenarios[0]
        p = DeepAccidentPerception(s, DeepAccidentConfig())
        dets = p.detections_for("ego_vehicle")
        series = s.agents["ego_vehicle"]
        have = set(available_cameras(load_calib(series.calib_paths[series.frames[0]])))
        used = {d.camera for d in dets}
        self.assertTrue(used, "카메라 이름이 채워지지 않았다")
        self.assertTrue(used <= have, f"모르는 카메라: {used - have}")
        # 여러 카메라가 실제로 기여해야 한다
        self.assertGreater(len(used), 1)
        self.assertEqual(sum(p.per_camera.values()), len(dets))

    def test_unknown_camera_name_is_rejected(self):
        """오타 하나로 카메라가 통째로 빠지면 잘못된 결론에 이른다."""
        from traffic_llm.deepaccident import DeepAccidentPerception

        s = self.scenarios[0]
        p = DeepAccidentPerception(s, DeepAccidentConfig(cameras=("Camera_Fron",)))
        with self.assertRaises(ValueError):
            p.detections_for("ego_vehicle")


if __name__ == "__main__":
    unittest.main(verbosity=2)


@unittest.skipUnless(HAS_DATA, "DeepAccident 데이터셋 없음")
class TestNestedSplitRoot(unittest.TestCase):
    """루트가 분할들의 상위 디렉터리일 때도 스캔되는지.

    실제 배포 데이터는 `DeepAccident/{DeepAccident_mini,train,val,test}` 로
    나뉘어 있다. 상위 경로를 그대로 주면 예전에는 조용히 0개가 나와 데이터가
    없는 것처럼 보였다.
    """

    @classmethod
    def setUpClass(cls):
        cls.parent = os.path.dirname(os.path.abspath(DA_ROOT))
        if not os.path.isdir(cls.parent):
            raise unittest.SkipTest("상위 디렉터리 없음")

    def test_split_dir_still_works(self):
        from traffic_llm.deepaccident import scan_scenarios

        self.assertTrue(scan_scenarios(DA_ROOT, DeepAccidentConfig()))

    def test_parent_dir_finds_at_least_the_split(self):
        from traffic_llm.deepaccident import scan_scenarios

        direct = scan_scenarios(DA_ROOT, DeepAccidentConfig())
        nested = scan_scenarios(self.parent, DeepAccidentConfig())
        self.assertGreaterEqual(len(nested), len(direct))
        # 분할 디렉터리에서 찾은 시나리오는 상위에서도 모두 나와야 한다
        ids = {s.scenario for s in nested}
        for s in direct:
            self.assertIn(s.scenario, ids)

    def test_no_duplicate_scenarios_from_nesting(self):
        """같은 시나리오가 두 번 나오면 채점·집계가 어긋난다."""
        from traffic_llm.deepaccident import scan_scenarios

        nested = scan_scenarios(self.parent, DeepAccidentConfig())
        keys = [(s.scenario_type, s.scenario, s.root) for s in nested]
        self.assertEqual(len(keys), len(set(keys)))

    def test_type_filter_applies_through_nesting(self):
        from traffic_llm.deepaccident import scan_scenarios

        got = scan_scenarios(
            self.parent, DeepAccidentConfig(),
            scenario_types=["type1_subtype2_accident"],
        )
        self.assertTrue(got)
        self.assertEqual(
            {s.scenario_type for s in got}, {"type1_subtype2_accident"}
        )

    def test_town_filter_applies_through_nesting(self):
        from traffic_llm.deepaccident import scan_scenarios

        got = scan_scenarios(self.parent, DeepAccidentConfig(), towns=["Town05"])
        self.assertTrue(got)
        self.assertEqual({s.town for s in got}, {"Town05"})

    def test_empty_dir_yields_nothing_without_error(self):
        """zip 만 있고 추출되지 않은 분할(test)이 있어도 오류가 아니다."""
        import tempfile
        from traffic_llm.deepaccident import scan_scenarios

        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "type1_subtype1_accident"))
            self.assertEqual(scan_scenarios(d, DeepAccidentConfig()), [])
