"""파이프라인 단위/통합 테스트.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_llm.config import (
    CameraConfig,
    FusionConfig,
    LaneConfig,
    PerceptionConfig,
    PipelineConfig,
    SerializeConfig,
)
from traffic_llm.geometry import (
    CameraModel,
    LocalENU,
    ego_to_world,
    heading_to_unit,
    project_point_to_polyline,
    unit_to_heading,
    world_to_ego,
    wrap180,
)
from traffic_llm.kinematics import TrackHistory, analyze_interactions, classify_maneuver
from traffic_llm.perception import JsonPerception
from traffic_llm.pipeline import TrafficSceneConverter
from traffic_llm.roadmap import Road, RoadNetwork, direction_label
from dataclasses import replace

from traffic_llm.schemas import ActorState, Detection, EgoSample, RoadPlacement
from traffic_llm.serialize import to_json, to_text

LANE_W = 3.25
ORIGIN = (39.9612, -83.0007)


def make_camera() -> CameraConfig:
    return CameraConfig.from_fov(1920, 1080, hfov_deg=62.0, height_m=1.35, pitch_deg=2.0)


def write_test_map(path: str) -> None:
    enu = LocalENU(*ORIGIN)

    def geo(e, n):
        lat, lon = enu.to_geo(e, n)
        return [lon, lat]

    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": "ns_s",
                    "name": "High St",
                    "lanes": 4,
                    "lanes:forward": 2,
                    "lanes:backward": 2,
                    "maxspeed": "50",
                },
                "geometry": {"type": "LineString",
                             "coordinates": [geo(0, -300), geo(0, 0)]},
            },
            {
                "type": "Feature",
                "properties": {
                    "id": "ns_n",
                    "name": "High St",
                    "lanes": 4,
                    "lanes:forward": 2,
                    "lanes:backward": 2,
                    "maxspeed": "50",
                },
                "geometry": {"type": "LineString",
                             "coordinates": [geo(0, 0), geo(0, 300)]},
            },
            {
                "type": "Feature",
                "properties": {
                    "id": "ew_w",
                    "name": "Broad St",
                    "lanes": 4,
                    "lanes:forward": 2,
                    "lanes:backward": 2,
                    "maxspeed": "40 mph",
                },
                "geometry": {"type": "LineString",
                             "coordinates": [geo(-300, 0), geo(0, 0)]},
            },
            {
                "type": "Feature",
                "properties": {
                    "id": "ew_e",
                    "name": "Broad St",
                    "lanes": 4,
                    "lanes:forward": 2,
                    "lanes:backward": 2,
                    "maxspeed": "40 mph",
                },
                "geometry": {"type": "LineString",
                             "coordinates": [geo(0, 0), geo(300, 0)]},
            },
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f)


class TestGeometry(unittest.TestCase):
    def test_heading_roundtrip(self):
        for h in (0.0, 45.0, 90.0, 180.0, 271.0, 359.0):
            e, n = heading_to_unit(h)
            self.assertAlmostEqual(unit_to_heading(e, n), h, places=6)

    def test_heading_conventions(self):
        # 0도 = 북(+n), 90도 = 동(+e)
        self.assertAlmostEqual(heading_to_unit(0.0)[1], 1.0, places=9)
        self.assertAlmostEqual(heading_to_unit(90.0)[0], 1.0, places=9)

    def test_wrap180(self):
        self.assertAlmostEqual(wrap180(350.0), -10.0)
        self.assertAlmostEqual(wrap180(-350.0), 10.0)
        self.assertAlmostEqual(wrap180(190.0), -170.0)

    def test_ego_world_roundtrip(self):
        origin = (100.0, -50.0)
        for heading in (0.0, 37.0, 90.0, 200.0, 315.0):
            for ego in ((10.0, 2.0), (-5.0, -3.5), (60.0, 0.0)):
                w = ego_to_world(ego, origin, heading)
                back = world_to_ego(w, origin, heading)
                self.assertAlmostEqual(back[0], ego[0], places=6)
                self.assertAlmostEqual(back[1], ego[1], places=6)

    def test_ego_to_world_directions(self):
        # 북향(0도) 차량의 전방 10m 는 원점 기준 북쪽 10m
        w = ego_to_world((10.0, 0.0), (0.0, 0.0), 0.0)
        self.assertAlmostEqual(w[0], 0.0, places=6)
        self.assertAlmostEqual(w[1], 10.0, places=6)
        # 북향 차량의 좌측 3m 는 서쪽 3m (e = -3)
        w = ego_to_world((0.0, 3.0), (0.0, 0.0), 0.0)
        self.assertAlmostEqual(w[0], -3.0, places=6)

    def test_local_enu_roundtrip(self):
        enu = LocalENU(*ORIGIN)
        for e, n in ((0.0, 0.0), (150.0, -230.0), (-1000.0, 800.0)):
            lat, lon = enu.to_geo(e, n)
            e2, n2 = enu.to_enu(lat, lon)
            self.assertAlmostEqual(e2, e, places=2)
            self.assertAlmostEqual(n2, n, places=2)

    def test_projection_roundtrip(self):
        """ego_to_pixel → bbox_to_ego 역투영이 원래 위치를 복원해야 한다."""
        cam = make_camera()
        model = CameraModel(cam)
        sizes = PerceptionConfig().class_size
        for x, y in ((15.0, 0.0), (30.0, 3.5), (45.0, -2.0), (8.0, 1.0)):
            w = sizes["car"].width_m
            bl = model.ego_to_pixel(x, y + w / 2, 0.0)
            br = model.ego_to_pixel(x, y - w / 2, 0.0)
            tl = model.ego_to_pixel(x, y + w / 2, 1.55)
            self.assertIsNotNone(bl)
            self.assertIsNotNone(br)
            self.assertIsNotNone(tl)
            bbox = (min(bl[0], br[0]), tl[1], max(bl[0], br[0]), bl[1])
            (rx, ry), quality = model.bbox_to_ego(bbox, "car", sizes)
            # 접지점은 차량의 '가까운 면'이므로 역투영에는 근접면→중심 보정이
            # 들어간다. 여기서는 후면 접지점을 투영했으므로 그 보정만큼
            # 앞쪽으로 나온다 (0.25*(w+l) ≈ 1.6m).
            offset = 0.25 * (sizes["car"].width_m + sizes["car"].length_m)
            self.assertLess(abs(rx - x - offset), max(1.5, 0.08 * x),
                            f"종방향 오차 과다: {rx} vs {x}+{offset:.2f}")
            self.assertLess(abs(ry - y), 1.2, f"횡방향 오차 과다: {ry} vs {y}")
            self.assertGreater(quality, 0.0)

    def test_pixel_above_horizon_returns_none(self):
        model = CameraModel(make_camera())
        self.assertIsNone(model.pixel_to_ground(960.0, 10.0))

    def test_project_point_to_polyline_sign(self):
        poly = [(0.0, 0.0), (0.0, 100.0)]  # 북쪽으로 진행
        # 서쪽(e<0)에 있는 점은 진행방향 기준 왼쪽 → 양수
        s, off, dist, seg = project_point_to_polyline((-3.0, 50.0), poly)
        self.assertAlmostEqual(s, 50.0, places=6)
        self.assertGreater(off, 0.0)
        self.assertAlmostEqual(dist, 3.0, places=6)


class TestRoadmap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.map_path = os.path.join(cls.tmp, "m.geojson")
        write_test_map(cls.map_path)
        cls.net = RoadNetwork.from_geojson(cls.map_path, LaneConfig(), origin=ORIGIN)

    def test_loaded(self):
        self.assertEqual(len(self.net.roads), 4)
        inter = [j for j in self.net.junctions.values() if j.is_intersection]
        self.assertEqual(len(inter), 1, "중앙 교차로 1개가 인식되어야 한다")

    def test_maxspeed_mph_parsing(self):
        broad = self.net.roads["ew_e"]
        self.assertAlmostEqual(broad.speed_limit_kph, 40 * 1.60934, places=3)
        high = self.net.roads["ns_n"]
        self.assertAlmostEqual(high.speed_limit_kph, 50.0, places=6)

    def test_direction_label(self):
        self.assertEqual(direction_label(0.0), "북행")
        self.assertEqual(direction_label(90.0), "동행")
        self.assertEqual(direction_label(180.0), "남행")
        self.assertEqual(direction_label(270.0), "서행")

    def test_lane_assignment_northbound(self):
        """우측통행 북행: 1차선(중앙선쪽) e=+1.6, 2차선 e=+4.9."""
        p1 = self.net.locate((0.5 * LANE_W, -100.0), 0.0)
        self.assertIsNotNone(p1)
        self.assertEqual(p1.direction_label, "북행")
        self.assertEqual(p1.lane_index, 1)
        self.assertEqual(p1.lane_count, 2)

        p2 = self.net.locate((1.5 * LANE_W, -100.0), 0.0)
        self.assertEqual(p2.lane_index, 2)

    def test_lane_assignment_southbound(self):
        """남행은 중앙선 서쪽(e<0)."""
        p = self.net.locate((-0.5 * LANE_W, 100.0), 180.0)
        self.assertIsNotNone(p)
        self.assertEqual(p.direction_label, "남행")
        self.assertEqual(p.lane_index, 1)

    def test_lane_side_decides_direction(self):
        """방위각 없이도 통행측으로 진행방향이 정해져야 한다.

        우측통행에서 진행차로는 중심선 우측이므로, 차량이 중심선 어느 쪽에
        있는지가 방향을 말해 준다. 이것이 기본 판정 근거다.
        """
        west = (-0.5 * LANE_W, 100.0)  # 북향 도로의 진행방향 좌측 → 남행
        east = (0.5 * LANE_W, 100.0)   # 진행방향 우측 → 북행
        pw = self.net.locate(west, None)
        pe = self.net.locate(east, None)
        self.assertEqual(pw.direction_label, "남행")
        self.assertEqual(pe.direction_label, "북행")
        for pl in (pw, pe):
            self.assertTrue(pl.direction_confident)
            self.assertEqual(pl.direction_source, "lane_side")

    def test_untrusted_heading_does_not_flip_direction(self):
        """단안 궤적 방위로 방향을 정하면 뒤집힌 추정이 그대로 확정된다.

        방위는 도로 **선택**에만 쓰고 방향은 통행측으로 정한다. 그래서 180°
        뒤집힌 방위를 줘도 진행방향은 바뀌지 않아야 한다.
        """
        xy = (-0.5 * LANE_W, 100.0)  # 통행측으로는 남행
        for h in (0.0, 180.0, None):
            self.assertEqual(self.net.locate(xy, h).direction_label, "남행")

    def test_trusted_heading_decides_direction(self):
        """텔레메트리·3D 센서 방위는 신뢰하므로 방향을 정할 수 있다."""
        xy = (-0.5 * LANE_W, 100.0)
        p_n = self.net.locate(xy, 0.0, trust_heading_direction=True)
        p_s = self.net.locate(xy, 180.0, trust_heading_direction=True)
        self.assertEqual(p_n.direction_label, "북행")
        self.assertEqual(p_s.direction_label, "남행")
        self.assertEqual(p_n.direction_source, "heading")

    def test_direction_not_confident_far_from_carriageway(self):
        """차도 폭을 넘어 떨어진 차량은 통행측을 신뢰할 수 없다.

        CARLA 의 분리도로는 방향별로 별개 road 이므로, 기준선에서 차도 폭보다
        멀리 있는 차량은 나란한 다른 도로에 속한다 — 실측 반전율이 |횡오프셋|
        7m 이하 2~12% 에서 7m 초과 54% 로 급증한다.
        """
        near = self.net.locate((-0.5 * LANE_W, 100.0), None)
        self.assertTrue(near.direction_confident)
        far = self.net.locate((-3.4 * LANE_W, 100.0), None)
        self.assertIsNotNone(far)
        self.assertFalse(far.direction_confident)
        self.assertEqual(far.direction_source, "default")

    def test_axis_bearing_is_direction_independent(self):
        """axis_bearing_deg 는 진행방향과 무관한 도로 기하다."""
        for xy in ((-0.5 * LANE_W, 100.0), (0.5 * LANE_W, 100.0)):
            self.assertAlmostEqual(
                self.net.locate(xy, None).axis_bearing_deg, 0.0, delta=1.0
            )

    def test_lane_numbering_from_curb(self):
        net = RoadNetwork.from_geojson(
            self.map_path, LaneConfig(numbering="from_curb"), origin=ORIGIN
        )
        # 중앙선쪽 차선은 from_curb 규약에서 마지막 번호
        p = net.locate((0.5 * LANE_W, -100.0), 0.0)
        self.assertEqual(p.lane_index, 2)

    def test_dist_to_next_junction(self):
        p = self.net.locate((0.5 * LANE_W, -50.0), 0.0)
        self.assertIsNotNone(p.dist_to_next_junction_m)
        self.assertAlmostEqual(p.dist_to_next_junction_m, 50.0, delta=2.0)

    def test_off_road_returns_none(self):
        self.assertIsNone(self.net.locate((500.0, 500.0), 0.0))

    def test_perpendicular_heading_rejected(self):
        """도로 위에 있어도 방위각이 직각이면 그 도로에 매칭되지 않아야 한다."""
        p = self.net.locate((0.5 * LANE_W, -100.0), 90.0)
        if p is not None:
            self.assertNotEqual(p.road_id, "ns_s")

    def test_downstream_paths_straight(self):
        p = self.net.locate((0.5 * LANE_W, -200.0), 0.0)
        opts = self.net.downstream_paths(p, 50.0)
        self.assertEqual(len(opts), 1)
        self.assertEqual(opts[0][0], "직진")

    def test_downstream_paths_at_junction(self):
        """교차로 직전에서는 직진/좌회전/우회전 후보가 나와야 한다."""
        p = self.net.locate((0.5 * LANE_W, -20.0), 0.0)
        opts = self.net.downstream_paths(p, 100.0)
        labels = {o[0] for o in opts}
        self.assertIn("직진", labels)
        self.assertIn("좌회전", labels)
        self.assertIn("우회전", labels)
        self.assertAlmostEqual(sum(o[1] for o in opts), 1.0, places=6)

    def test_lane_biases_turn_probability(self):
        """1차선(중앙선쪽)은 좌회전 확률이 최외측 차선보다 높아야 한다."""
        p_inner = self.net.locate((0.5 * LANE_W, -20.0), 0.0)
        p_outer = self.net.locate((1.5 * LANE_W, -20.0), 0.0)
        left_inner = dict(
            (o[0], o[1]) for o in self.net.downstream_paths(p_inner, 100.0)
        ).get("좌회전", 0.0)
        left_outer = dict(
            (o[0], o[1]) for o in self.net.downstream_paths(p_outer, 100.0)
        ).get("좌회전", 0.0)
        self.assertGreater(left_inner, left_outer)


class TestKinematics(unittest.TestCase):
    def _placement(
        self,
        off: float,
        lane: int,
        bearing: float = 0.0,
        direction: str = "북행",
    ) -> RoadPlacement:
        return RoadPlacement(
            road_id="r", road_name="R", s_m=0.0, lateral_offset_m=off,
            direction_label=direction, bearing_deg=bearing, lane_index=lane,
            lane_count=2, speed_limit_kph=50.0,
            dist_to_next_junction_m=100.0, next_junction_id="J0",
        )

    def test_speed_from_trajectory(self):
        h = TrackHistory("a")
        for k in range(6):
            h.update(k * 0.5, (0.0, 5.0 * k * 0.5))  # 5 m/s 북진
        self.assertAlmostEqual(h.speed_ema, 5.0, delta=0.3)
        self.assertAlmostEqual(h.heading_ema, 0.0, delta=1.0)

    def test_stationary_gives_no_heading(self):
        h = TrackHistory("a")
        for k in range(6):
            h.update(k * 0.5, (0.0, 0.0))
        self.assertIsNone(h.heading_ema)

    def test_real_lane_change_detected(self):
        """근거리에서 3.25m 횡이동 + 차선번호 변화 → 차선변경으로 분류."""
        h = TrackHistory("a")
        actor = ActorState("a", "observed", "car", (0.0, 0.0), 0.0, 12.0, None,
                           observed_range_m=25.0)
        offs = [-4.88, -4.88, -4.33, -3.79, -3.25, -2.71, -2.17]
        for k, off in enumerate(offs):
            t = k * 0.5
            h.update(t, (0.0, 12.0 * t), 25.0)
            lane = 2 if off < -0.5 * LANE_W else 1
            h.record_placement(t, self._placement(off, lane), 25.0)
        actor.placement = self._placement(offs[-1], 1)
        actor.track_age_s = h.age_s
        self.assertEqual(classify_maneuver(actor, h, LANE_W), "차선변경(좌)")

    def test_range_bias_drift_not_a_lane_change(self):
        """접근 중 원거리 차량의 측거 편향 수렴은 차선변경이 아니다."""
        h = TrackHistory("a")
        # 관측거리 118→52m, 오프셋이 편향 때문에 -7.6→-2.8 로 수렴
        data = [(-7.6, 107.0), (-6.6, 96.0), (-5.7, 85.0), (-4.9, 74.0),
                (-3.3, 63.0), (-2.8, 52.0)]
        for k, (off, rng) in enumerate(data):
            t = k * 0.5
            h.update(t, (off, 60.0 - 13.0 * t), rng)
            h.record_placement(t, self._placement(off, 2, 180.0, "남행"), rng)
        actor = ActorState(
            "a", "observed", "car", (-2.8, 27.0), 180.0, 13.0, None,
            placement=self._placement(-2.8, 1, 180.0, "남행"),
            observed_range_m=52.0,
        )
        self.assertEqual(classify_maneuver(actor, h, LANE_W), "차선유지")

    def test_accel_suppressed_when_range_unstable(self):
        """거리가 급변하는 원거리 관측에서는 가속도를 보고하지 않는다."""
        h = TrackHistory("a")
        for k, rng in enumerate((110.0, 96.0, 85.0, 74.0)):
            h.update(k * 0.5, (0.0, 60.0 - 13.0 * k * 0.5), rng)
        self.assertIsNone(h.accel)

    def test_accel_computed_at_close_range(self):
        h = TrackHistory("a")
        # 근거리에서 실제 가속: 5 → 9 m/s
        speeds = [5.0, 6.0, 7.0, 8.0, 9.0]
        n = 0.0
        for k, v in enumerate(speeds):
            n += v * 0.5
            h.update(k * 0.5, (0.0, n), 20.0)
        self.assertIsNotNone(h.accel)
        self.assertGreater(h.accel, 0.5)

    def test_placement_change_clears_history(self):
        h = TrackHistory("a")
        h.record_placement(0.0, self._placement(-4.0, 2), 20.0)
        h.record_placement(0.5, self._placement(-4.0, 2), 20.0)
        self.assertEqual(len(h.lat_offsets), 2)
        other = self._placement(4.0, 1)
        other.direction_label = "남행"
        h.record_placement(1.0, other, 20.0)
        self.assertEqual(len(h.lat_offsets), 1, "방향이 바뀌면 이력을 버려야 한다")

    def test_stopped(self):
        h = TrackHistory("a")
        actor = ActorState("a", "ego", "car", (0.0, 0.0), 0.0, 0.2, 0.0)
        self.assertEqual(classify_maneuver(actor, h, LANE_W), "정지")

    def test_turn_detected_for_ego(self):
        h = TrackHistory("a")
        actor = ActorState(
            "a", "ego", "car", (0.0, 0.0), 40.0, 8.0, 0.0,
            placement=self._placement(-2.0, 1),
        )
        self.assertEqual(classify_maneuver(actor, h, LANE_W), "좌회전 중")

    def test_following_interaction(self):
        rear = ActorState("R", "ego", "car", (1.6, 0.0), 0.0, 20.0, 0.0,
                          placement=self._placement(-1.6, 1))
        lead = ActorState("L", "observed", "car", (1.6, 30.0), 0.0, 10.0, 0.0,
                          placement=self._placement(-1.6, 1))
        lead.placement.s_m = 30.0
        its = analyze_interactions([rear, lead])
        follow = [i for i in its if i.kind == "following"]
        self.assertEqual(len(follow), 1)
        self.assertEqual(follow[0].subject_id, "R")
        self.assertEqual(follow[0].object_id, "L")
        self.assertAlmostEqual(follow[0].gap_m, 30.0, places=3)
        self.assertAlmostEqual(follow[0].headway_s, 1.5, places=3)
        self.assertAlmostEqual(follow[0].ttc_s, 3.0, places=3)

    def test_different_lanes_not_following(self):
        a = ActorState("A", "ego", "car", (1.6, 0.0), 0.0, 20.0, 0.0,
                       placement=self._placement(-1.6, 1))
        b = ActorState("B", "observed", "car", (4.9, 30.0), 0.0, 10.0, 0.0,
                       placement=self._placement(-4.9, 2))
        b.placement.s_m = 30.0
        its = analyze_interactions([a, b])
        self.assertEqual([i for i in its if i.kind == "following"], [])


class TestFusionAndPipeline(unittest.TestCase):
    """합성 데이터로 end-to-end 검증 (examples/make_demo_data.py 와 동일 시나리오)."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
            ),
        )
        import make_demo_data as demo

        cls.demo = demo
        cls.tmp = tempfile.mkdtemp()
        demo.OUT = cls.tmp
        os.makedirs(cls.tmp, exist_ok=True)

        cls.map_path = os.path.join(cls.tmp, "roads.geojson")
        demo.write_map(cls.map_path)
        enu = LocalENU(*ORIGIN)
        # 충돌 없는 기본 배치 8초 — 이 클래스의 검사는 차선변경·추종 판정이다
        cls.duration = 8.0
        actors = demo.build_actors(with_collision=False)
        cam = make_camera()
        cls.cam = cam

        cls.specs = []
        for obs in actors[:3]:
            oid = obs[0]
            tele = os.path.join(cls.tmp, f"{oid}_tele.csv")
            det = os.path.join(cls.tmp, f"{oid}_det.json")
            demo.write_telemetry(
                tele, obs[2], obs[3], obs[4], enu, cls.duration
            )
            demo.write_detections(
                det, obs, [a for a in actors if a[0] != oid], cam, cls.duration
            )
            cls.specs.append((oid, f"{oid}.mp4", tele, det))

        from traffic_llm.perception import JsonPerception
        from traffic_llm.pipeline import TrafficSceneConverter

        cfg = PipelineConfig(area_name="테스트")
        cfg.perception.sample_hz = 2.0
        cls.cfg = cfg
        net = RoadNetwork.from_geojson(cls.map_path, cfg.lane, origin=ORIGIN)
        backend = JsonPerception(cfg.perception, {s[1]: s[3] for s in cls.specs})
        conv = TrafficSceneConverter(cfg, net, backend)
        for oid, video, tele, _ in cls.specs:
            conv.add_vehicle(oid, video, tele, cam,
                             detections=backend.run(video, t0=0.0))
        cls.snaps = list(conv.run(rate_hz=2.0))

    def test_snapshots_produced(self):
        self.assertGreater(len(self.snaps), 10)

    def test_no_duplicate_actors(self):
        """진실값은 최대 6대. 융합 실패로 유령 차량이 생기면 안 된다."""
        for s in self.snaps:
            self.assertLessEqual(
                len(s.actors), 6,
                f"t={s.t}: 액터 {len(s.actors)}대 — 중복 계수 의심",
            )
            ids = [a.actor_id for a in s.actors]
            self.assertEqual(len(ids), len(set(ids)))

    def test_three_egos_always_present(self):
        for s in self.snaps:
            egos = [a for a in s.actors if a.kind == "ego"]
            self.assertEqual(len(egos), 3)

    def test_ego_lane_assignment_correct(self):
        """V1=2차선 북행, V2=1차선 북행, V3=1차선 동행 (진실값)."""
        expect = {
            "EGO_V1": ("High St", "북행", 2),
            "EGO_V2": ("High St", "북행", 1),
            "EGO_V3": ("Broad St", "동행", 1),
        }
        for s in self.snaps:
            for a in s.actors:
                if a.actor_id not in expect:
                    continue
                name, direction, lane = expect[a.actor_id]
                self.assertIsNotNone(a.placement, f"{a.actor_id} 도로 미매칭")
                self.assertEqual(a.placement.road_name, name)
                self.assertEqual(a.placement.direction_label, direction)
                self.assertEqual(a.placement.lane_index, lane)

    def test_ego_speed_from_telemetry(self):
        """관측차량 속도는 텔레메트리 값 그대로여야 한다."""
        truth = {"EGO_V1": 12.0, "EGO_V2": 9.0, "EGO_V3": 14.0}
        for s in self.snaps:
            for a in s.actors:
                if a.actor_id in truth:
                    self.assertAlmostEqual(a.speed_mps, truth[a.actor_id], places=2)

    def test_observed_vehicle_position_accuracy(self):
        """단안 역투영 위치 정확도.

        5~45m 구간을 검사한다. 5m 미만은 차량이 화면을 가득 채워 bbox 가
        절단되고, 근접면→중심 보정량(차량 약 1.6m)이 거리의 절반에 달해
        상대오차가 구조적으로 커진다 — 전면 카메라의 물리적 한계이므로
        별도 기준으로 다룬다.
        """
        actors = self.demo.build_actors(with_collision=False)
        truth = {a[0]: a[2] for a in actors}
        errs = []
        for s in self.snaps:
            for a in s.actors:
                if a.kind != "observed":
                    continue
                rng = a.observed_range_m or 99.0
                if not (5.0 <= rng <= 45.0):
                    continue
                best = min(math.dist(a.world_xy, fn(s.t)) for fn in truth.values())
                errs.append(best)
        self.assertGreater(len(errs), 5, "근거리 관측 샘플이 너무 적다")
        errs.sort()
        median = errs[len(errs) // 2]
        self.assertLess(median, 1.5, f"중앙 오차 과다: {median:.2f}m")
        self.assertLess(max(errs), 3.5, f"최대 오차 과다: {max(errs):.2f}m")

    def test_no_false_maneuvers_for_straight_drivers(self):
        """직진 유지 차량에 허위 차선변경/회전이 붙지 않아야 한다."""
        for s in self.snaps:
            for a in s.actors:
                if a.kind != "ego":
                    continue
                self.assertEqual(
                    a.maneuver, "차선유지", f"t={s.t} {a.actor_id}: {a.maneuver}"
                )

    def test_real_lane_change_is_detected(self):
        """X3 의 실제 2→1차선 변경이 어느 시점에는 검출되어야 한다."""
        found = [
            (s.t, a.actor_id, a.maneuver)
            for s in self.snaps
            for a in s.actors
            if a.maneuver.startswith("차선변경")
        ]
        self.assertTrue(found, "실제 차선변경이 전혀 검출되지 않았다 (과도 억제)")
        self.assertTrue(
            all("좌" in m for _, _, m in found),
            f"방향 오분류: {found}",
        )

    def test_multi_observer_fusion_raises_confidence(self):
        """2대 이상이 본 차량은 존재 확신도가 높아야 한다."""
        multi = [a for s in self.snaps for a in s.actors
                 if a.kind == "observed" and len(a.observed_by) > 1]
        single = [a for s in self.snaps for a in s.actors
                  if a.kind == "observed" and len(a.observed_by) == 1]
        self.assertTrue(multi, "다중 관측 융합 사례가 없다")
        for a in multi:
            self.assertGreater(a.confidence, 0.9, a.actor_id)
        if single:
            self.assertLessEqual(
                max(x.confidence for x in single),
                max(x.confidence for x in multi),
            )

    def test_existence_and_position_confidence_are_separate(self):
        """존재 확신도와 위치 정확도는 별개 값이어야 한다.

        극근접·절단 bbox 는 위치가 부정확하지만 차량이 있다는 사실은 확실하다.
        한 값에 섞으면 바로 앞의 차량이 저신뢰로 보고된다.
        """
        seen = False
        for s in self.snaps:
            for a in s.actors:
                if a.kind != "observed":
                    continue
                self.assertGreaterEqual(a.position_quality, 0.0)
                self.assertLessEqual(a.position_quality, 1.0)
                if a.position_quality < 0.6:
                    # 위치가 부정확해도 존재 확신도는 떨어지지 않는다
                    self.assertGreater(
                        a.confidence, a.position_quality, a.actor_id
                    )
                    seen = True
        self.assertTrue(seen, "위치 정확도가 낮은 관측 표본이 없다")

    def test_serialization_roundtrip(self):
        cfg = SerializeConfig(max_actors=5)
        snap = self.snaps[len(self.snaps) // 2]
        d = to_json(snap, cfg)
        json.dumps(d, ensure_ascii=False)  # 직렬화 가능해야 함
        self.assertEqual(d["actor_count_total"], len(snap.actors))
        self.assertLessEqual(d["actor_count_included"], 5)
        self.assertEqual(
            d["actors_omitted"], max(0, len(snap.actors) - 5)
        )
        txt = to_text(snap, cfg)
        self.assertIn("교통 상황 스냅샷", txt)
        self.assertIn("관측차량", txt)

    def test_truncation_is_disclosed(self):
        """토큰 예산으로 잘린 차량 수가 본문에 명시되어야 한다."""
        cfg = SerializeConfig(max_actors=2)
        snap = max(self.snaps, key=lambda s: len(s.actors))
        txt = to_text(snap, cfg)
        dropped = len(snap.actors) - 2
        self.assertIn(f"{dropped}대 생략", txt)

    def test_interactions_reference_included_actors_only(self):
        cfg = SerializeConfig(max_actors=3)
        snap = max(self.snaps, key=lambda s: len(s.actors))
        d = to_json(snap, cfg)
        ids = {a["id"] for a in d["actors"]}
        for it in d["interactions"]:
            self.assertIn(it["subject"], ids)
            self.assertIn(it["object"], ids)

    def test_predictions_present_and_normalized(self):
        for s in self.snaps:
            for a in s.actors:
                self.assertTrue(a.predictions, f"{a.actor_id} 예측 경로 없음")
                total = sum(p.probability for p in a.predictions)
                self.assertAlmostEqual(total, 1.0, places=5)

    def test_llm_payload_shape(self):
        from traffic_llm.serialize import build_messages

        payload = build_messages(self.snaps[5], "위험 요소는?", SerializeConfig())
        self.assertEqual(payload["model"], "claude-opus-5")
        self.assertEqual(payload["thinking"]["type"], "adaptive")
        self.assertNotIn("temperature", payload)
        self.assertNotIn("budget_tokens", payload.get("thinking", {}))
        self.assertEqual(payload["messages"][-1]["role"], "user")
        self.assertEqual(
            payload["system"][0]["cache_control"]["type"], "ephemeral"
        )


class TestClassAwareGating(unittest.TestCase):
    """클래스별 상한을 쓰는 신원·속도 관문.

    관문이 없으면 보행자 관측이 근접만으로 오토바이 트랙을 물려받아
    100km/h 로 달리는 보행자가 만들어진다 (실데이터에서 관측된 결함).
    """

    def setUp(self):
        from traffic_llm.fusion import GlobalTrackRegistry

        # 운동 관문은 **트랙 id 가 전역 유일하지 않을 때** 신원을 지키는 장치다.
        # 전역 유일하면(기본값) 같은 id = 같은 물체이므로 관문을 걸지 않는다 —
        # 걸면 가려짐으로 오래 끊긴 차량이 여러 액터로 쪼개진다. 이 클래스는
        # 관문 자체를 시험하므로 그 조건을 명시한다.
        self.cfg = FusionConfig()
        self.cfg.global_track_ids = False
        self.reg = GlobalTrackRegistry(self.cfg)

    def test_default_assumes_shared_object_ids(self):
        """기본값은 V2X 로 객체 id 를 공유하는 전제다."""
        self.assertTrue(FusionConfig().global_track_ids)

    def test_class_groups(self):
        from traffic_llm.fusion import class_compatible, class_group

        self.assertTrue(class_compatible("car", "truck"))
        self.assertTrue(class_compatible("person", "bicycle"))
        self.assertFalse(class_compatible("person", "car"))
        self.assertFalse(class_compatible("bicycle", "motorcycle"))
        # 모르는 클래스에는 제약을 걸지 않는다
        self.assertTrue(class_compatible("trailer", "car"))
        self.assertEqual(class_group("motorcycle"), "vehicle")

    def test_pedestrian_does_not_inherit_vehicle_track(self):
        v = self.reg.resolve(0.0, ("obs", 1), (0.0, 0.0), "motorcycle")
        # 같은 자리에 보행자 관측 — 근접하지만 같은 물체일 수 없다
        p = self.reg.resolve(0.5, ("obs", 2), (0.5, 0.0), "person")
        self.assertNotEqual(v, p)
        self.assertTrue(p.startswith("P"))

    def test_vehicle_class_confusion_still_merges(self):
        a = self.reg.resolve(0.0, ("obs_a", 1), (0.0, 0.0), "car")
        b = self.reg.resolve(0.5, ("obs_b", 2), (1.0, 0.0), "van")
        self.assertEqual(a, b)

    def test_speed_cap_is_per_class(self):
        self.assertLess(self.reg._max_speed("person"), 10.0)
        self.assertEqual(self.reg._max_speed("car"), self.cfg.max_speed_mps)
        # 사전에 없는 클래스는 전역 상한
        self.assertEqual(self.reg._max_speed("trailer"), self.cfg.max_speed_mps)

    def test_id_prefix_by_class(self):
        self.assertTrue(
            self.reg.resolve(0.0, ("o", 1), (0.0, 0.0), "person").startswith("P")
        )
        self.assertTrue(
            self.reg.resolve(0.0, ("o", 2), (60.0, 0.0), "van").startswith("N")
        )
        self.assertTrue(
            self.reg.resolve(0.0, ("o", 3), (120.0, 0.0), "bicycle").startswith("C")
        )

    def test_pedestrian_teleport_gets_new_id(self):
        first = self.reg.resolve(0.0, ("obs", 7), (0.0, 0.0), "person")
        # 0.5초에 20m — 보행자 상한(8m/s)의 5배
        second = self.reg.resolve(0.5, ("obs", 7), (20.0, 0.0), "person")
        self.assertNotEqual(first, second)

    def test_authoritative_ids_survive_observation_gap(self):
        """전역 유일 트랙 id 면 가려짐으로 오래 끊겼다 돌아와도 같은 액터."""
        from traffic_llm.fusion import GlobalTrackRegistry

        cfg = FusionConfig()
        cfg.global_track_ids = True
        reg = GlobalTrackRegistry(cfg)
        a = reg.resolve_cluster(0.0, [("obs_a", 42)], (0.0, 0.0), "car")
        # track_timeout_s 를 훨씬 넘긴 뒤, 다른 관측자가 멀리서 다시 관측
        b = reg.resolve_cluster(9.0, [("obs_b", 42)], (150.0, 0.0), "car")
        self.assertEqual(a, b)

    def test_non_authoritative_ids_do_not_survive_gap(self):
        a = self.reg.resolve_cluster(0.0, [("obs_a", 42)], (0.0, 0.0), "car")
        b = self.reg.resolve_cluster(9.0, [("obs_b", 42)], (150.0, 0.0), "car")
        self.assertNotEqual(a, b)


class TestSpeedNoiseRejection(unittest.TestCase):
    """위치 잡음 지배 구간에서 속도 추정을 건너뛰는지."""

    def test_class_cap_rejects_absurd_step(self):
        h = TrackHistory("P001")
        h.update(0.0, (0.0, 0.0), 40.0, max_speed_mps=8.0)
        h.update(0.5, (1.0, 0.0), 40.0, max_speed_mps=8.0)
        v_before = h.speed_ema
        # 0.5초에 30m = 60m/s — 보행자에게 불가능
        h.update(1.0, (31.0, 0.0), 40.0, max_speed_mps=8.0)
        self.assertEqual(h.speed_ema, v_before)
        self.assertIsNone(h.accel)
        self.assertEqual(h.noisy_steps, 1)

    def test_accel_gate_rejects_step_under_class_cap(self):
        """클래스 상한 아래라도 물리적 가속도를 넘는 변화는 잡음."""
        h = TrackHistory("V001")
        for k in range(3):
            h.update(k * 0.5, (0.0, 5.0 * k), 30.0, max_speed_mps=40.0)
        v_before = h.speed_ema
        # 0.5초에 16m = 32m/s (상한 40 아래) 이지만 가속도 44m/s² 는 불가능
        h.update(1.5, (0.0, 26.0), 30.0, max_speed_mps=40.0)
        self.assertEqual(h.speed_ema, v_before)
        self.assertEqual(h.noisy_steps, 1)

    def test_noisy_step_does_not_stop_heading_tracking(self):
        """잡음 표본 하나로 속도 갱신은 건너뛰어도 방위 추정은 계속된다.

        방위는 프레임 간 변위가 아니라 창 전체 적합에서 나오므로 표본 하나에
        흔들리지 않는다.
        """
        h = TrackHistory("V001")
        for k in range(5):
            h.update(k * 0.5, (0.0, 6.0 * k), 20.0, max_speed_mps=40.0)
        h.update(2.5, (0.0, 90.0), 20.0, max_speed_mps=40.0)  # 잡음 (그 순간 큰 점프)
        self.assertGreater(h.noisy_steps, 0)
        self.assertIsNotNone(h.heading_ema)

    def test_report_cap_is_tighter_than_identity_cap(self):
        """두 역할을 한 값으로 겸하면 안 된다.

        신원 관문 상한은 잡음 흡수를 위해 넉넉해야 하고, 보고 상한은 실제 지속
        주행 속도여야 한다. 겸하면 위치 잡음이 '28km/h 로 달리는 보행자'로 나간다.
        """
        cfg = FusionConfig()
        for cls in ("person", "bicycle", "truck", "bus"):
            self.assertLess(
                cfg.report_speed_by_class[cls],
                cfg.max_speed_by_class[cls],
                f"{cls}: 보고 상한이 신원 관문 상한보다 좁아야 한다",
            )
        self.assertLessEqual(cfg.report_speed_by_class["person"], 3.0)

    def test_plausible_step_is_accepted(self):
        h = TrackHistory("V001")
        h.update(0.0, (0.0, 0.0), 30.0, max_speed_mps=40.0)
        h.update(0.5, (0.0, 6.0), 30.0, max_speed_mps=40.0)
        self.assertAlmostEqual(h.speed_ema, 12.0, places=6)
        self.assertEqual(h.noisy_steps, 0)


class TestVelocityFit(unittest.TestCase):
    """속도·방위는 최소자승 적합에서 나오고, 각자 맞는 불확실도로 관문을 건다.

    프레임 간 차분은 잡음을 그대로 속도로 옮긴다: 48m 거리의 위치 잡음이
    0.5초 간격에 만드는 겉보기 속도는 20m/s 를 넘어, 사실상 정지한 차량이
    63km/h 로 보고되고 방위가 180° 뒤집힌다 (Town01 V001 에서 관측된 결함).
    """

    @staticmethod
    def track(points, rng=20.0):
        h = TrackHistory("X")
        for k, (e, n) in enumerate(points):
            h.update(k * 0.5, (e, n), rng, max_speed_mps=40.0)
        return h

    def test_straight_motion_speed_and_heading(self):
        # 북향 6m/0.5s = 12m/s, 잡음 없음
        h = self.track([(0.0, 6.0 * k) for k in range(6)])
        self.assertAlmostEqual(h.speed_from_fit(40.0), 12.0, places=6)
        self.assertAlmostEqual(h.heading_from_fit(), 0.0, places=6)

    def test_stationary_with_noise_reports_nothing(self):
        """위치 잡음만 있는 정지 물체는 속도·방위 모두 미확정이어야 한다."""
        noise = [(0.0, 0.0), (2.5, -1.8), (-2.1, 2.4), (1.4, -2.6), (-1.1, 1.2),
                 (2.0, 0.4)]
        h = self.track(noise, rng=48.0)
        self.assertIsNone(h.speed_from_fit(40.0))
        self.assertIsNone(h.heading_from_fit())

    def test_needs_minimum_samples(self):
        h = self.track([(0.0, 0.0), (0.0, 6.0)])
        self.assertIsNone(h.velocity_fit())
        self.assertIsNone(h.speed_from_fit(40.0))

    def test_class_cap_rejects_implausible_fit(self):
        h = self.track([(0.0, 30.0 * k) for k in range(5)])  # 60m/s
        self.assertIsNone(h.speed_from_fit(40.0))
        self.assertAlmostEqual(h.speed_from_fit(None), 60.0, places=6)

    def test_two_uncertainties_are_reported(self):
        """속도 크기는 잔차로, 방향은 거리척도로 판정한다 (오차 기제가 다르다)."""
        h = self.track([(0.0, 6.0 * k) for k in range(6)], rng=60.0)
        ve, vn, span, sigma_v, sigma_pos = h.velocity_fit()
        self.assertAlmostEqual(vn, 12.0, places=6)
        self.assertLess(sigma_v, 0.5)          # 매끄러운 궤적 → 잔차 작음
        self.assertGreater(sigma_pos, 5.0)     # 60m 거리 → 위치 오차 규모 큼

    def test_far_range_withholds_heading_but_not_speed(self):
        """지평선 근처 역투영은 매끄러워도 방향이 틀린다 — 방위를 주장하지 않는다."""
        h = self.track([(0.0, 6.0 * k) for k in range(6)], rng=110.0)
        self.assertIsNotNone(h.speed_from_fit(40.0))
        self.assertIsNone(h.heading_from_fit())

    def test_axis_component_projects_onto_road(self):
        h = self.track([(0.0, 6.0 * k) for k in range(6)])
        along, span, sigma_pos = h.axis_component(0.0)     # 북향 도로
        self.assertAlmostEqual(along, 12.0, places=6)
        back, _, _ = h.axis_component(180.0)               # 남향 도로
        self.assertAlmostEqual(back, -12.0, places=6)
        cross, _, _ = h.axis_component(90.0)               # 직교
        self.assertAlmostEqual(cross, 0.0, places=6)

    def test_position_sigma_grows_faster_than_linear(self):
        from traffic_llm.kinematics import position_sigma_m

        self.assertAlmostEqual(position_sigma_m(0.0), 5.0)
        self.assertGreater(position_sigma_m(100.0), 2 * position_sigma_m(50.0))


class TestAxisDirection(unittest.TestCase):
    """도로축 진행 부호는 확신할 때만 낸다."""

    def tracker(self, points, rng=20.0):
        from traffic_llm.kinematics import KinematicsTracker

        tr = KinematicsTracker(max_speed_by_class={"car": 40.0})
        for k, (e, n) in enumerate(points):
            a = ActorState(
                actor_id="V1", kind="observed", cls="car", world_xy=(e, n),
                heading_deg=None, speed_mps=None, accel_mps2=None,
                observed_by=["O"], observed_range_m=rng,
            )
            tr.update(k * 0.5, a)
        return tr

    def test_forward_and_reverse(self):
        tr = self.tracker([(0.0, 6.0 * k) for k in range(6)])
        self.assertIs(tr.axis_direction("V1", 0.0), False)    # 북향 축, 북행
        self.assertIs(tr.axis_direction("V1", 180.0), True)   # 남향 축 기준 역방향

    def test_undetermined_when_noise_dominates(self):
        tr = self.tracker(
            [(0.0, 0.0), (1.8, -1.4), (-1.5, 2.0), (1.0, -2.2), (-0.8, 1.0)],
            rng=48.0,
        )
        self.assertIsNone(tr.axis_direction("V1", 0.0))

    def test_undetermined_when_crossing_the_axis(self):
        """축과 직교로 움직이면 축방향 부호를 정할 수 없다."""
        tr = self.tracker([(6.0 * k, 0.0) for k in range(6)])
        self.assertIsNone(tr.axis_direction("V1", 0.0))

    def test_unknown_actor(self):
        tr = self.tracker([(0.0, 6.0 * k) for k in range(6)])
        self.assertIsNone(tr.axis_direction("없는액터", 0.0))


class TestScriptedCollision(unittest.TestCase):
    """합성 데모의 충돌 정답 구성."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "examples",
            ),
        )
        import make_demo_data as demo

        cls.demo = demo

    def test_collision_detected_and_timed(self):
        actors = self.demo.build_actors(with_collision=True)
        truth = self.demo.scripted_collision(actors)
        self.assertTrue(truth.occurred)
        self.assertEqual(truth.method, "scripted")
        self.assertEqual(
            set(truth.carla_ids),
            {self.demo.TRACK_ID["V1"], self.demo.TRACK_ID["X4"]},
        )
        # 두 궤적은 실제로 겹친다 (설계상 측면충돌)
        self.assertLess(truth.min_distance_m, 1.0)
        self.assertIsNotNone(truth.time_s)

    def test_no_collision_when_partner_absent(self):
        actors = self.demo.build_actors(with_collision=False)
        truth = self.demo.scripted_collision(actors)
        self.assertFalse(truth.occurred)

    def test_lead_in_delays_collision_without_changing_geometry(self):
        base = self.demo.scripted_collision(
            self.demo.build_actors(True, lead_in_s=0.0)
        )
        shifted = self.demo.scripted_collision(
            self.demo.build_actors(True, lead_in_s=6.0)
        )
        self.assertAlmostEqual(shifted.time_s - base.time_s, 6.0, places=2)
        self.assertAlmostEqual(
            shifted.min_distance_m, base.min_distance_m, places=6
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestPathPredictionQuality(unittest.TestCase):
    """예측 경로가 LLM 에게 오해 없이 전달되는지.

    payload 를 실제 LLM 에 보냈을 때 "우회전/우회전 33%씩" 같은 중복 항목과
    끝에서 정체된 웨이포인트를 근거 불충분으로 지적받아 고친 부분이다.
    """

    def test_identical_geometry_merged_into_one_path(self):
        """같은 기동으로 같은 경로가 여러 번 나오면 하나로 합치고 확률을 더한다."""
        from traffic_llm.prediction import merge_identical
        from traffic_llm.schemas import PredictedPath

        wp = [(0.0, 0.0), (0.0, 10.0)]
        merged = merge_identical(
            [
                PredictedPath("우회전", 0.2, list(wp), 5.0, to_roads=["A"]),
                PredictedPath("우회전", 0.2, list(wp), 5.0, to_roads=["B"]),
                PredictedPath("직진", 0.6, [(0.0, 0.0), (0.0, 20.0)], 5.0,
                              to_roads=["C"]),
            ]
        )
        self.assertEqual(len(merged), 2)
        by = {p.maneuver: p for p in merged}
        self.assertAlmostEqual(by["우회전"].probability, 0.4, places=6)
        self.assertEqual(by["우회전"].to_roads, ["A", "B"])
        # 확률 순 정렬
        self.assertEqual(merged[0].maneuver, "직진")

    def test_sub_output_resolution_differences_are_the_same_path(self):
        """0.1m 미만 차이는 payload 에 나타나지 않으므로 같은 경로로 본다."""
        from traffic_llm.prediction import merge_identical
        from traffic_llm.schemas import PredictedPath

        merged = merge_identical(
            [
                PredictedPath("직진", 0.5, [(0.0, 0.0), (0.0, 10.0)], 5.0,
                              to_roads=["A"]),
                PredictedPath("직진", 0.5, [(0.0, 0.0), (0.001, 10.004)], 5.0,
                              to_roads=["B"]),
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0].probability, 1.0, places=6)

    def test_distinct_geometry_kept_separate(self):
        from traffic_llm.prediction import merge_identical
        from traffic_llm.schemas import PredictedPath

        merged = merge_identical(
            [
                PredictedPath("우회전", 0.2, [(0.0, 0.0), (5.0, 5.0)], 5.0,
                              to_roads=["A"]),
                PredictedPath("우회전", 0.2, [(0.0, 0.0), (9.0, 2.0)], 5.0,
                              to_roads=["B"]),
            ]
        )
        self.assertEqual(len(merged), 2)

    def test_waypoints_stop_at_road_end_instead_of_repeating(self):
        """도로가 짧으면 끝점을 반복해 채우지 않는다 — 정지 예측으로 읽힌다."""
        from traffic_llm.prediction import _resample_by_time

        poly = [(0.0, 0.0), (0.0, 12.0)]  # 12m 뿐
        pts, truncated = _resample_by_time(poly, v=10.0, horizon_s=5.0)
        self.assertTrue(truncated)
        self.assertLessEqual(len(pts), 3)  # 0s, 1s, 끝점
        self.assertEqual(len(pts), len(set(pts)), f"좌표 반복: {pts}")
        self.assertAlmostEqual(pts[-1][1], 12.0, places=6)

    def test_waypoints_not_truncated_when_road_is_long_enough(self):
        from traffic_llm.prediction import _resample_by_time

        poly = [(0.0, 0.0), (0.0, 200.0)]
        pts, truncated = _resample_by_time(poly, v=10.0, horizon_s=5.0)
        self.assertFalse(truncated)
        self.assertEqual(len(pts), 6)  # 0~5초
        self.assertAlmostEqual(pts[-1][1], 50.0, places=6)

    def test_downstream_paths_report_target_road(self):
        """모든 후보가 진입 도로 이름을 달고 나오고, 확률 합이 1 이어야 한다.

        도로 **이름**은 유일하지 않다 — 교차로 양쪽이 같은 길인 경우가 흔해
        좌회전과 우회전의 진입 도로명이 같을 수 있다. 후보를 구별하는 것은
        (기동 라벨, 기하) 쌍이고, 도로명은 그 위의 보조 단서다.
        """
        tmp = tempfile.mkdtemp()
        mp = os.path.join(tmp, "m.geojson")
        write_test_map(mp)
        net = RoadNetwork.from_geojson(mp, LaneConfig(), origin=ORIGIN)
        p = net.locate((0.5 * LANE_W, -20.0), 0.0)
        opts = net.downstream_paths(p, 100.0)
        self.assertAlmostEqual(sum(o[1] for o in opts), 1.0, places=6)
        self.assertTrue(all(o[3] for o in opts), "진입 도로 이름이 없다")
        keys = {(o[0], tuple((round(x, 2), round(y, 2)) for x, y in o[2]))
                for o in opts}
        self.assertEqual(len(keys), len(opts), "같은 라벨+기하 후보가 중복")


class TestProbabilityRounding(unittest.TestCase):
    """표시 확률의 합이 1.00 을 벗어나지 않는지.

    내부 계산은 정확히 1.0 인데 개별 반올림 때문에 payload 만 1.02 로 보이면
    확률 자체를 신뢰할 수 없게 된다.
    """

    def test_thirds_sum_to_one(self):
        from traffic_llm.serialize import _round_probs

        r = _round_probs([1 / 3, 1 / 3, 1 / 3])
        self.assertAlmostEqual(sum(r), 1.0, places=6)

    def test_uneven_split_sums_to_one(self):
        from traffic_llm.serialize import _round_probs

        for vals in (
            [0.3417, 0.3417, 0.3166],
            [0.555, 0.370, 0.075],
            [0.125, 0.125, 0.125, 0.125, 0.5],
            [0.9999, 0.0001],
        ):
            r = _round_probs(vals)
            self.assertAlmostEqual(sum(r), 1.0, places=6, msg=str(vals))
            self.assertEqual(len(r), len(vals))

    def test_largest_remainder_gets_the_extra_unit(self):
        from traffic_llm.serialize import _round_probs

        # 0.336 의 잔여(0.6)가 0.332 의 잔여(0.2)보다 크므로 그쪽이 올라간다
        r = _round_probs([0.336, 0.332, 0.332])
        self.assertEqual(r, [0.34, 0.33, 0.33])

    def test_single_and_empty(self):
        from traffic_llm.serialize import _round_probs

        self.assertEqual(_round_probs([]), [])
        self.assertEqual(_round_probs([1.0]), [1.0])

    def test_serialized_actor_probabilities_sum_to_one(self):
        from traffic_llm.schemas import ActorState, PredictedPath, SceneSnapshot

        actor = ActorState(
            actor_id="V001", kind="observed", cls="car",
            world_xy=(0.0, 0.0), heading_deg=0.0, speed_mps=10.0,
            accel_mps2=0.0, placement=None, observed_by=["v1"],
            predictions=[
                PredictedPath("직진", 1 / 3, [(0.0, 1.0)], 5.0, to_roads=["A"]),
                PredictedPath("좌회전", 1 / 3, [(0.0, 2.0)], 5.0, to_roads=["B"]),
                PredictedPath("우회전", 1 / 3, [(0.0, 3.0)], 5.0, to_roads=["C"]),
            ],
        )
        snap = SceneSnapshot(
            t=0.0, actors=[actor], interactions=[], area_name="X",
            map_context={}, scenario=None, frame_idx=1,
        )
        d = to_json(snap, SerializeConfig())
        paths = d["actors"][0]["predicted_paths"]
        self.assertEqual(len(paths), 3)
        self.assertAlmostEqual(
            sum(p["probability"] for p in paths), 1.0, places=6
        )


class TestMultiCamera(unittest.TestCase):
    """관측자 1기가 카메라 k대를 다는 경우.

    핵심은 **검출을 만든 카메라의 파라미터로 역투영**하는 것이다. 후방 카메라의
    bbox 를 전방 파라미터로 풀면 yaw 가 180° 틀려 차량이 반대편에 놓인다
    (실측: 올바른 파라미터 중앙값 3.1m vs 전방 일괄 22.6m).
    """

    @staticmethod
    def _cams():
        front = CameraConfig.from_fov(
            1600, 900, hfov_deg=70.0, height_m=1.5, name="front"
        )
        back = CameraConfig.from_fov(
            1600, 900, hfov_deg=70.0, height_m=1.5, yaw_deg=180.0, name="back"
        )
        return [front, back]

    @staticmethod
    def _tele():
        return [
            EgoSample(t=float(i) * 0.5, lat=ORIGIN[0], lon=ORIGIN[1],
                      heading_deg=0.0, speed_mps=0.0)
            for i in range(6)
        ]

    def _conv(self):
        tmp = tempfile.mkdtemp()
        mp = os.path.join(tmp, "m.geojson")
        write_test_map(mp)
        net = RoadNetwork.from_geojson(mp, LaneConfig(), origin=ORIGIN)
        return TrafficSceneConverter(PipelineConfig(), net, JsonPerception(
            PerceptionConfig(), {}
        ))

    def test_back_camera_detection_lands_behind(self):
        """후방 카메라 검출은 관측자 **뒤쪽**(x_fwd<0)에 놓여야 한다."""
        conv = self._conv()
        cams = self._cams()
        # 화면 중앙 하단의 bbox — 두 카메라 모두 같은 픽셀 위치
        bbox = (760.0, 600.0, 840.0, 680.0)
        dets = [
            Detection(t=0.0, frame_idx=1, track_id=1, cls="car", conf=0.9,
                      bbox=bbox, camera="front"),
            Detection(t=0.0, frame_idx=1, track_id=2, cls="car", conf=0.9,
                      bbox=bbox, camera="back"),
        ]
        conv.add_vehicle("v1", video_path="<v1>", telemetry_path="",
                         camera=cams, detections=dets, telemetry=self._tele())
        src = conv.sources["v1"]
        xy_f, _ = src.model_for("front").bbox_to_ego(
            bbox, "car", conv.cfg.perception.class_size
        )
        xy_b, _ = src.model_for("back").bbox_to_ego(
            bbox, "car", conv.cfg.perception.class_size
        )
        self.assertGreater(xy_f[0], 0.0, "전방 카메라 검출이 앞에 놓이지 않았다")
        self.assertLess(xy_b[0], 0.0, "후방 카메라 검출이 뒤에 놓이지 않았다")
        # 같은 픽셀이므로 크기는 같고 방향만 반대여야 한다
        self.assertAlmostEqual(abs(xy_f[0]), abs(xy_b[0]), places=6)

    def test_unknown_camera_falls_back_to_primary(self):
        conv = self._conv()
        conv.add_vehicle("v1", video_path="<v1>", telemetry_path="",
                         camera=self._cams(), detections=[],
                         telemetry=self._tele())
        src = conv.sources["v1"]
        self.assertIs(src.model_for("nope"), src.model_for("front"))
        self.assertIs(src.model_for(None), src.model_for("front"))

    def test_single_camera_still_accepted(self):
        """CameraConfig 하나를 그대로 넘기는 기존 호출이 계속 동작해야 한다."""
        conv = self._conv()
        cam = CameraConfig.from_fov(1600, 900, hfov_deg=70.0, height_m=1.5)
        conv.add_vehicle("v1", video_path="<v1>", telemetry_path="",
                         camera=cam, detections=[], telemetry=self._tele())
        src = conv.sources["v1"]
        self.assertEqual(len(src.cameras), 1)
        self.assertIs(src.camera, cam)

    def test_duplicate_camera_names_rejected(self):
        """이름으로 파라미터를 고르므로 중복은 조용히 넘어가면 안 된다."""
        conv = self._conv()
        dup = [
            CameraConfig.from_fov(1600, 900, height_m=1.5, name="c"),
            CameraConfig.from_fov(1600, 900, height_m=1.5, name="c"),
        ]
        with self.assertRaises(ValueError):
            conv.add_vehicle("v1", video_path="<v1>", telemetry_path="",
                             camera=dup, detections=[], telemetry=self._tele())

    def test_video_list_length_must_match(self):
        conv = self._conv()
        with self.assertRaises(ValueError):
            conv.add_vehicle("v1", video_path=["a.mp4"], telemetry_path="",
                             camera=self._cams(), telemetry=self._tele(),
                             detections=[])

    def test_video_dict_requires_every_camera(self):
        conv = self._conv()
        with self.assertRaises(ValueError):
            conv.add_vehicle("v1", video_path={"front": "a.mp4"},
                             telemetry_path="", camera=self._cams(),
                             telemetry=self._tele(), detections=[])

    def test_observation_records_its_camera(self):
        conv = self._conv()
        dets = [
            Detection(t=0.0, frame_idx=1, track_id=1, cls="car", conf=0.9,
                      bbox=(760.0, 600.0, 840.0, 680.0), camera="back",
                      ego_xy_measured=(-12.0, 0.0)),
        ]
        conv.add_vehicle("v1", video_path="<v1>", telemetry_path="",
                         camera=self._cams(), detections=dets,
                         telemetry=self._tele())
        snap = conv.snapshot_at(0.0)
        self.assertIsNotNone(snap)

    def test_camera_coverage_is_a_union_not_a_sum(self):
        """화각을 단순히 더하면 6대×70°=420° 로 360° 를 넘는다."""
        conv = self._conv()
        cams = [
            CameraConfig.from_fov(1600, 900, hfov_deg=70.0, height_m=1.5,
                                  yaw_deg=y, name=f"c{i}")
            for i, y in enumerate((0.0, 55.0, -55.0, 110.0, -110.0, 180.0))
        ]
        conv.add_vehicle("v1", video_path="<v1>", telemetry_path="",
                         camera=cams, detections=[], telemetry=self._tele())
        cov = TrafficSceneConverter._camera_coverage(conv.sources["v1"])
        self.assertEqual(cov["n_cameras"], 6)
        self.assertLessEqual(cov["azimuth_coverage_deg"], 360)
        self.assertGreater(cov["azimuth_coverage_deg"], 300)

    def test_overlapping_cameras_do_not_double_count_coverage(self):
        conv = self._conv()
        same = [
            CameraConfig.from_fov(1600, 900, hfov_deg=70.0, height_m=1.5,
                                  yaw_deg=0.0, name=f"c{i}")
            for i in range(3)
        ]
        conv.add_vehicle("v1", video_path="<v1>", telemetry_path="",
                         camera=same, detections=[], telemetry=self._tele())
        cov = TrafficSceneConverter._camera_coverage(conv.sources["v1"])
        self.assertEqual(cov["n_cameras"], 3)
        self.assertLess(cov["azimuth_coverage_deg"], 80)

    def test_snapshot_exposes_observer_cameras(self):
        conv = self._conv()
        conv.add_vehicle("v1", video_path="<v1>", telemetry_path="",
                         camera=self._cams(), detections=[],
                         telemetry=self._tele())
        snap = conv.snapshot_at(0.0)
        self.assertIsNotNone(snap)
        oc = snap.map_context["observer_cameras"]
        self.assertEqual(oc["v1"]["n_cameras"], 2)
        self.assertEqual(oc["v1"]["names"], ["front", "back"])


class TestPathGeometry(unittest.TestCase):
    """예상 경로가 차량이 있는 곳에서, 차량이 보는 방향으로 시작하는지.

    payload 를 실제 LLM 에 보냈을 때 두 가지를 지적받아 고친 부분이다.
      - 경로가 방위와 180° 반대로 나간다 (실측 22%)
      - 첫 웨이포인트가 현재 위치에서 5~10m 떨어져 있다 (실측 30%)
    """

    @staticmethod
    def _curved_network():
        """s 축을 따라 방위가 90° 도는 도로 하나.

        폴리라인 **첫 구간**의 방위와 비교하면 뒤쪽 구간에 있는 차량의 진행
        방향 판정이 뒤집힌다. 그것이 원래 버그였다.
        """
        import math as _m

        # 반경 60m, 90° 원호 (길이 ≈ 94m) — 지평 30m 를 담을 여유가 있어야
        # 경로가 도로 끝에서 잘리지 않는다
        pts = [
            (60.0 * _m.sin(_m.radians(a)), 60.0 * (1 - _m.cos(_m.radians(a))))
            for a in range(0, 91, 5)
        ]
        road = Road(
            road_id="c1", name="Curve", poly=pts, lanes_forward=1,
            lanes_backward=1, speed_limit_kph=50.0,
            length_m=sum(math.dist(pts[i], pts[i + 1])
                         for i in range(len(pts) - 1)),
        )
        return RoadNetwork([road], LaneConfig(), LocalENU(*ORIGIN))

    def test_direction_uses_local_axis_not_first_segment(self):
        """곡선 도로 끝쪽에서 진행 방향이 뒤집히지 않아야 한다."""
        net = self._curved_network()
        road = net.roads["c1"]
        # 폴리라인 중후반 (방위가 첫 구간과 45° 넘게 다르다)
        s = road.length_m * 0.6
        p = net.locate(road.point_at(s), None)
        self.assertIsNotNone(p)
        local = road.bearing_at(RoadNetwork._seg_at(road, p.s_m))
        first = road.bearing_at(0)
        self.assertGreater(
            abs(wrap180(local - first)), 45.0, "픽스처가 곡선이 아니다"
        )
        # 국소 도로축과 같은 방위로 달리는 차량 → 경로도 같은 방향이어야 한다
        opts = net.downstream_paths(
            RoadPlacement(
                road_id="c1", road_name="Curve", s_m=s, lateral_offset_m=0.0,
                direction_label="", bearing_deg=local, lane_index=1,
                lane_count=1, speed_limit_kph=50.0,
                dist_to_next_junction_m=None, next_junction_id=None,
                axis_bearing_deg=local,
            ),
            30.0,
        )
        self.assertTrue(opts)
        poly = opts[0][2]
        self.assertGreaterEqual(len(poly), 2)
        step = math.degrees(
            math.atan2(poly[1][0] - poly[0][0], poly[1][1] - poly[0][1])
        ) % 360
        self.assertLess(
            abs(wrap180(step - local)), 45.0,
            f"경로가 방위({local:.0f}°)와 어긋난다 ({step:.0f}°)",
        )

    def test_reverse_direction_goes_backwards(self):
        net = self._curved_network()
        road = net.roads["c1"]
        s = road.length_m * 0.6
        local = road.bearing_at(RoadNetwork._seg_at(road, s))
        back = (local + 180.0) % 360.0
        opts = net.downstream_paths(
            RoadPlacement(
                road_id="c1", road_name="Curve", s_m=s, lateral_offset_m=0.0,
                direction_label="", bearing_deg=back, lane_index=1,
                lane_count=1, speed_limit_kph=50.0,
                dist_to_next_junction_m=None, next_junction_id=None,
                axis_bearing_deg=local,
            ),
            30.0,
        )
        poly = opts[0][2]
        step = math.degrees(
            math.atan2(poly[1][0] - poly[0][0], poly[1][1] - poly[0][1])
        ) % 360
        self.assertLess(abs(wrap180(step - back)), 45.0)

    def test_path_follows_the_lane_not_the_centerline(self):
        """횡오프셋을 반영해야 경로가 차량이 달리는 차로를 따른다."""
        net = self._curved_network()
        road = net.roads["c1"]
        s = road.length_m * 0.5
        local = road.bearing_at(RoadNetwork._seg_at(road, s))
        pl = RoadPlacement(
            road_id="c1", road_name="Curve", s_m=s, lateral_offset_m=-1.75,
            direction_label="", bearing_deg=local, lane_index=1, lane_count=1,
            speed_limit_kph=50.0, dist_to_next_junction_m=None,
            next_junction_id=None, axis_bearing_deg=local,
        )
        center = net.downstream_paths(replace(pl, lateral_offset_m=0.0), 30.0)
        offset = net.downstream_paths(pl, 30.0)
        d = math.dist(center[0][2][0], offset[0][2][0])
        self.assertAlmostEqual(d, 1.75, delta=0.2)

    def test_prediction_starts_at_the_actor_position(self):
        """경로의 첫 웨이포인트는 차량의 현재 위치여야 한다."""
        from traffic_llm.prediction import predict
        from traffic_llm.schemas import ActorState

        net = self._curved_network()
        road = net.roads["c1"]
        s = road.length_m * 0.5
        local = road.bearing_at(RoadNetwork._seg_at(road, s))
        # 도로에서 6m 벗어난 위치 (인도 위 보행자 같은 경우)
        rad = math.radians(local)
        base = road.point_at(s)
        pos = (base[0] - math.cos(rad) * 6.0, base[1] + math.sin(rad) * 6.0)
        a = ActorState(
            actor_id="P1", kind="observed", cls="person", world_xy=pos,
            heading_deg=local, speed_mps=2.0, accel_mps2=0.0,
            placement=RoadPlacement(
                road_id="c1", road_name="Curve", s_m=s, lateral_offset_m=-6.0,
                direction_label="", bearing_deg=local, lane_index=1,
                lane_count=1, speed_limit_kph=50.0,
                dist_to_next_junction_m=None, next_junction_id=None,
                axis_bearing_deg=local,
            ),
            observed_by=["v1"],
        )
        paths = predict(a, net, horizon_s=5.0)
        self.assertTrue(paths)
        for p in paths:
            self.assertAlmostEqual(
                math.dist(p.waypoints[0], pos), 0.0, places=6,
                msg=f"{p.maneuver} 경로가 {math.dist(p.waypoints[0], pos):.1f}m 떨어져 시작",
            )


class TestGlobalTrackIdAssumption(unittest.TestCase):
    """"트랙 id 가 전역 유일하다"는 기본 전제와 그 위반 탐지.

    이 프로젝트는 V2X 로 객체 id 를 공유하는 협력형 자율주행을 전제하므로
    `global_track_ids` 기본값이 True 다. 카메라별 독립 추적기(YOLO+ByteTrack)는
    각자 1번부터 번호를 매겨 이 전제가 깨지는데, 그대로 두면 **수백 m 떨어진 두
    차량이 한 액터로 병합되고 하나가 조용히 사라진다.**
    """

    @staticmethod
    def _obs(oid, tid, xy, dist, cls="car"):
        from traffic_llm.schemas import Observation

        return Observation(
            t=0.0, observer_id=oid, local_track_id=tid, cls=cls, conf=0.9,
            ego_xy=(dist, 0.0), world_xy=xy, distance_m=dist,
            bbox=(0.0, 0.0, 10.0, 10.0), range_quality=0.9,
        )

    def _fuse(self, obs, cfg):
        from traffic_llm.fusion import GlobalTrackRegistry, fuse_observations
        from traffic_llm.schemas import EgoSample

        ego = {
            o.observer_id: EgoSample(0.0, 0.0, 0.0, 0.0, 10.0) for o in obs
        }
        world = {o.observer_id: (-30.0, 0.0) for o in obs}
        reg = GlobalTrackRegistry(cfg)
        actors = fuse_observations(0.0, list(obs), ego, world, reg, cfg)
        return [a for a in actors if a.kind == "observed"], reg

    def test_default_is_true(self):
        self.assertTrue(FusionConfig().global_track_ids)

    def test_same_id_merges_across_observers(self):
        """전제가 맞으면 같은 id 는 위치 게이트와 무관하게 한 물체다."""
        cfg = FusionConfig()
        obs = [self._obs("A", 1, (0.0, 0.0), 20.0),
               self._obs("B", 1, (8.0, 0.0), 25.0)]
        actors, reg = self._fuse(obs, cfg)
        self.assertEqual(len(actors), 1)
        self.assertEqual(actors[0].observed_by, ["A", "B"])
        self.assertEqual(reg.id_collisions, 0)

    def test_id_collision_is_split_not_merged(self):
        """전제가 깨지면 쪼개고 카운터를 올린다 — 조용히 병합하지 않는다."""
        cfg = FusionConfig()
        obs = [self._obs("A", 1, (0.0, 0.0), 20.0),
               self._obs("B", 1, (400.0, 300.0), 25.0)]
        actors, reg = self._fuse(obs, cfg)
        self.assertEqual(len(actors), 2, "서로 다른 차량이 병합됐다")
        self.assertEqual(reg.id_collisions, 1)
        # 두 차량이 모두 남아 있어야 한다 (하나가 사라지면 안 된다)
        xs = sorted(round(a.world_xy[0]) for a in actors)
        self.assertEqual(xs, [0, 400])

    def test_measurement_disagreement_is_not_a_collision(self):
        """단안 측거 불일치(실측 최대 66m)를 충돌로 오판하면 안 된다."""
        cfg = FusionConfig()
        self.assertGreater(cfg.id_collision_dist_m, 66.0)
        obs = [self._obs("A", 1, (0.0, 0.0), 100.0),
               self._obs("B", 1, (66.0, 0.0), 100.0)]
        actors, reg = self._fuse(obs, cfg)
        self.assertEqual(reg.id_collisions, 0)
        self.assertEqual(len(actors), 1)

    def test_check_can_be_disabled(self):
        cfg = FusionConfig()
        cfg.id_collision_dist_m = 0.0
        obs = [self._obs("A", 1, (0.0, 0.0), 20.0),
               self._obs("B", 1, (400.0, 300.0), 25.0)]
        actors, reg = self._fuse(obs, cfg)
        self.assertEqual(len(actors), 1)
        self.assertEqual(reg.id_collisions, 0)

    def test_non_global_mode_uses_position_only(self):
        """False 로 두면 id 를 신원 근거로 쓰지 않는다."""
        cfg = FusionConfig()
        cfg.global_track_ids = False
        obs = [self._obs("A", 1, (0.0, 0.0), 20.0),
               self._obs("B", 1, (400.0, 300.0), 25.0)]
        actors, reg = self._fuse(obs, cfg)
        self.assertEqual(len(actors), 2)
        self.assertEqual(reg.id_collisions, 0, "검사는 전역 모드에서만 돈다")


class TestTurnLabelConvention(unittest.TestCase):
    """좌/우회전 라벨이 실제 방향과 맞는지.

    방위각은 **진북 기준 시계방향**(북 0°, 동 90°)이다. 따라서 방위가 커지는 것 =
    시계방향 = 차량 기준 **우회전**이다. 부호를 반대로 쓰면 좌/우가 통째로
    뒤집히는데, 라벨 자체는 그대로 나오므로 "세 라벨이 다 있는지"만 보는 테스트는
    이것을 잡지 못한다 (실제로 못 잡았고, 실데이터에서 41건이 뒤집혀 있었다).
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        mp = os.path.join(cls.tmp, "m.geojson")
        write_test_map(mp)
        cls.net = RoadNetwork.from_geojson(mp, LaneConfig(), origin=ORIGIN)

    def _options(self, bearing_deg):
        """교차로 직전에서 주어진 방위로 달리는 차량의 후보 경로."""
        p = self.net.locate((0.5 * LANE_W, -20.0), bearing_deg)
        self.assertIsNotNone(p)
        p = replace(p, bearing_deg=bearing_deg, axis_bearing_deg=bearing_deg)
        return self.net.downstream_paths(p, 100.0)

    @staticmethod
    def _final_bearing(poly):
        e, n = poly[-1][0] - poly[-2][0], poly[-1][1] - poly[-2][1]
        return unit_to_heading(e, n)

    def test_north_bound_east_exit_is_a_right_turn(self):
        """북행 차량이 동쪽으로 나가면 우회전이다."""
        for label, prob, poly, _name in self._options(0.0):
            if label in ("직진", "유턴"):
                continue
            d = wrap180(self._final_bearing(poly) - 0.0)
            expect = "우회전" if d > 0 else "좌회전"
            self.assertEqual(
                label, expect,
                f"방위 0° 에서 {d:+.0f}° 회전인데 라벨이 {label} 이다",
            )

    def test_label_matches_geometry_for_every_heading(self):
        """네 방위 모두에서 라벨과 실제 회전 방향이 일치해야 한다."""
        for bearing in (0.0, 90.0, 180.0, 270.0):
            for label, _p, poly, _n in self._options(bearing):
                if label in ("직진", "유턴") or len(poly) < 2:
                    continue
                d = wrap180(self._final_bearing(poly) - bearing)
                if abs(d) < 30.0:
                    continue
                expect = "우회전" if d > 0 else "좌회전"
                self.assertEqual(label, expect, f"방위 {bearing}° / {d:+.0f}°")

    def test_clockwise_is_right_by_definition(self):
        """규약 자체를 못박는다 — 방위 증가 = 시계방향 = 우회전."""
        self.assertAlmostEqual(unit_to_heading(0.0, 1.0), 0.0, places=6)   # 북
        self.assertAlmostEqual(unit_to_heading(1.0, 0.0), 90.0, places=6)  # 동
        # 북에서 동으로 = 오른쪽. 외적 z 성분이 음수여야 한다.
        fe, fn = heading_to_unit(0.0)     # 북 전방
        de, dn = heading_to_unit(90.0)    # 동 이동
        self.assertLess(fe * dn - fn * de, 0.0, "북→동이 좌측으로 판정된다")
        self.assertGreater(wrap180(90.0 - 0.0), 0.0, "북→동의 방위차가 음수다")

    def test_lane_one_still_favours_left_turn(self):
        """1차선(중앙선쪽)이 좌회전에 유리한 보정은 그대로 유효하다.

        우측통행에서 1차선은 중앙선쪽 = 가장 왼쪽 차로이므로 좌회전 대기 차로다.
        """
        inner = self.net.locate((0.5 * LANE_W, -20.0), 0.0)
        outer = self.net.locate((1.5 * LANE_W, -20.0), 0.0)
        left_in = dict(
            (o[0], o[1]) for o in self.net.downstream_paths(inner, 100.0)
        ).get("좌회전", 0.0)
        left_out = dict(
            (o[0], o[1]) for o in self.net.downstream_paths(outer, 100.0)
        ).get("좌회전", 0.0)
        self.assertGreater(left_in, left_out)


class TestPluggablePredictor(unittest.TestCase):
    """예측기 교체 지점.

    기본은 규칙 기반이고, `PredictContext → List[PredictedPath]` 형태면 무엇이든
    끼울 수 있다. 학습 모델과 규칙 기반이 **같은 입력**을 봐야 학습·추론의 특징이
    어긋나지 않는다.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        mp = os.path.join(cls.tmp, "m.geojson")
        write_test_map(mp)
        cls.net = RoadNetwork.from_geojson(mp, LaneConfig(), origin=ORIGIN)

    def _actor(self, xy=(0.5 * LANE_W, -20.0), speed=10.0, heading=0.0):
        from traffic_llm.schemas import ActorState

        return ActorState(
            actor_id="V001", kind="observed", cls="car", world_xy=xy,
            heading_deg=heading, speed_mps=speed, accel_mps2=0.0,
            placement=self.net.locate(xy, heading), observed_by=["v1"],
        )

    def test_default_is_rule_based(self):
        from traffic_llm.prediction import predict, rule_predict, build_context

        a = self._actor()
        auto = predict(a, self.net, 5.0)
        manual = rule_predict(build_context(a, self.net, 5.0))
        self.assertEqual(
            [(p.maneuver, round(p.probability, 6)) for p in auto],
            [(p.maneuver, round(p.probability, 6)) for p in manual],
        )

    def test_custom_predictor_is_used(self):
        from traffic_llm.prediction import predict
        from traffic_llm.schemas import PredictedPath

        called = {}

        def fake(ctx):
            called["ctx"] = ctx
            return [PredictedPath("직진", 1.0, [(0.0, 0.0), (0.0, 1.0)], ctx.horizon_s)]

        out = predict(self._actor(), self.net, 5.0, predictor=fake)
        self.assertEqual(len(out), 1)
        self.assertIn("ctx", called)
        self.assertTrue(called["ctx"].candidates, "후보를 만들어 넘겨야 한다")

    def test_constant_velocity_predictor_ignores_map_candidates(self):
        from traffic_llm.prediction import build_context, constant_velocity_predict

        ctx = build_context(self._actor(xy=(1.0, 2.0), speed=2.0, heading=90.0), self.net, 3.0)
        self.assertTrue(ctx.candidates, "테스트 전제: 지도 후보가 있어야 한다")
        paths = constant_velocity_predict(ctx)
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].maneuver, "등속 직진")
        expected = [(1.0, 2.0), (3.0, 2.0), (5.0, 2.0), (7.0, 2.0)]
        for got, want in zip(paths[0].waypoints, expected):
            self.assertAlmostEqual(got[0], want[0], places=6)
            self.assertAlmostEqual(got[1], want[1], places=6)

    def test_constant_velocity_does_not_treat_unknown_speed_as_stopped(self):
        from traffic_llm.prediction import build_context, constant_velocity_predict

        ctx = build_context(self._actor(speed=None), self.net, 5.0)
        path = constant_velocity_predict(ctx)[0]
        self.assertIn("보류", path.maneuver)
        self.assertEqual(path.waypoints, [ctx.actor.world_xy])

    def test_context_carries_what_the_rule_uses(self):
        from traffic_llm.prediction import build_context

        a = self._actor()
        ctx = build_context(a, self.net, 5.0, t_s=1.5, scenario_id="s/x")
        self.assertIs(ctx.actor, a)
        self.assertEqual(ctx.horizon_s, 5.0)
        self.assertEqual(ctx.t_s, 1.5)
        self.assertAlmostEqual(ctx.horizon_m, 50.0, places=6)
        self.assertTrue(ctx.candidates)
        for c in ctx.candidates:
            self.assertGreaterEqual(len(c.polyline), 2)
            self.assertGreater(c.prior, 0.0)

    def test_no_candidates_when_stopped(self):
        from traffic_llm.prediction import build_context

        ctx = build_context(self._actor(speed=0.1), self.net, 5.0)
        self.assertEqual(ctx.candidates, [])

    def test_no_candidates_without_placement(self):
        from traffic_llm.prediction import build_context

        a = self._actor()
        a.placement = None
        self.assertEqual(build_context(a, self.net, 5.0).candidates, [])

    def test_feature_vector_lengths_are_fixed(self):
        from traffic_llm.predict_model import (
            N_CANDIDATE_FEATURES, N_GLOBAL_FEATURES, encode,
        )
        from traffic_llm.prediction import build_context

        ctx = build_context(self._actor(), self.net, 5.0)
        enc = encode(ctx)
        self.assertEqual(len(enc["global"]), N_GLOBAL_FEATURES)
        self.assertEqual(len(enc["candidates"]), len(ctx.candidates))
        for row in enc["candidates"]:
            self.assertEqual(len(row), N_CANDIDATE_FEATURES)

    def test_missing_values_get_a_flag_not_just_zero(self):
        """0 이 '값이 0' 인지 '모른다' 인지 구별되어야 한다."""
        from traffic_llm.predict_model import global_features
        from traffic_llm.prediction import build_context

        a = self._actor()
        known = global_features(build_context(a, self.net, 5.0))
        a2 = self._actor()
        a2.speed_mps = None
        a2.heading_deg = None
        a2.accel_mps2 = None
        unknown = global_features(build_context(a2, self.net, 5.0))
        self.assertEqual(known[1], 1.0)      # 속도 있음
        self.assertEqual(unknown[1], 0.0)
        self.assertEqual(known[6], 1.0)      # 방위 있음
        self.assertEqual(unknown[6], 0.0)
        self.assertEqual(unknown[3], 0.0)    # 가속도 있음

    def test_features_are_finite(self):
        from traffic_llm.predict_model import encode
        from traffic_llm.prediction import build_context

        for speed in (0.6, 10.0, 30.0):
            enc = encode(build_context(self._actor(speed=speed), self.net, 5.0))
            for v in enc["global"]:
                self.assertTrue(math.isfinite(v))
            for row in enc["candidates"]:
                for v in row:
                    self.assertTrue(math.isfinite(v))

    def test_candidate_to_path_matches_rule_output(self):
        """rank 모드가 쓰는 좌표 생성이 규칙 기반과 같아야 한다."""
        from traffic_llm.prediction import (
            build_context, candidate_to_path, rule_predict,
        )

        ctx = build_context(self._actor(), self.net, 5.0)
        rule = {p.maneuver: p for p in rule_predict(ctx)}
        for cand in ctx.candidates:
            p = candidate_to_path(ctx, cand, cand.prior)
            self.assertIsNotNone(p)
            if p.maneuver in rule:
                # 같은 기동의 규칙 출력과 첫 웨이포인트가 같아야 한다
                self.assertAlmostEqual(
                    p.waypoints[0][0], rule[p.maneuver].waypoints[0][0], places=6
                )

    def test_torch_predictor_needs_torch_or_reports_clearly(self):
        """torch 가 없으면 조용히 실패하지 않고 설치 안내를 낸다."""
        from traffic_llm.predict_model import TorchPredictor
        from traffic_llm.prediction import build_context

        ctx = build_context(self._actor(), self.net, 5.0)
        p = TorchPredictor("/nonexistent/model.pt")
        try:
            import torch  # noqa: F401
        except ImportError:
            with self.assertRaises(ImportError) as cm:
                p(ctx)
            self.assertIn("torch", str(cm.exception))
            return
        with self.assertRaises(Exception):
            p(ctx)

    def test_torch_predictor_rejects_bad_mode(self):
        from traffic_llm.predict_model import TorchPredictor

        with self.assertRaises(ValueError):
            TorchPredictor("m.pt", mode="nope")

    def test_torch_predictor_falls_back_without_candidates(self):
        """후보가 없으면 모델을 부르지 않고 폴백한다 (모델 파일이 없어도 동작)."""
        from traffic_llm.predict_model import TorchPredictor
        from traffic_llm.prediction import build_context

        ctx = build_context(self._actor(speed=0.1), self.net, 5.0)
        out = TorchPredictor("/nonexistent/model.pt")(ctx)
        self.assertEqual([p.maneuver for p in out], ["정지 유지"])

    def test_dropped_candidates_do_not_leak_probability(self):
        """폴리라인이 짧아 버려진 후보의 확률은 남은 후보로 재분배해야 한다.

        규칙 기반은 `prior` 를 그대로 확률로 쓴다. 재분배하지 않으면 합이 1 미만이
        되고(실측 0.032), 페이로드가 "직진 3%" 만 내보내 읽는 쪽이 "어디로도 가지
        않는다"로 해석한다.
        """
        from traffic_llm.prediction import build_context, rule_predict
        from traffic_llm import prediction as pmod

        a = self._actor()
        real = self.net.downstream_paths

        def with_a_degenerate_exit(placement, dist):
            out = list(real(placement, dist))
            self.assertTrue(out, "테스트 전제: 후보가 있어야 한다")
            # 첫 후보를 점 1개로 만든다 = build_context 가 버리는 형태
            label, prior, poly, to_road = out[0]
            out[0] = (label, prior, [poly[0]], to_road)
            return out

        self.net.downstream_paths = with_a_degenerate_exit
        try:
            ctx = build_context(a, self.net, 5.0)
            self.assertTrue(ctx.candidates, "나머지 후보는 남아야 한다")
            self.assertAlmostEqual(
                sum(c.prior for c in ctx.candidates), 1.0, places=9
            )
            paths = rule_predict(ctx)
            self.assertAlmostEqual(
                sum(p.probability for p in paths), 1.0, places=6
            )
        finally:
            self.net.downstream_paths = real

    def test_renormalization_preserves_ranking(self):
        """재분배는 균일 배율이므로 후보 순위를 바꾸지 않는다 (베이스라인 불변)."""
        from traffic_llm.prediction import build_context

        ctx = build_context(self._actor(), self.net, 5.0)
        before = [c.prior for c in ctx.candidates]
        order = sorted(range(len(before)), key=lambda i: -before[i])
        scaled = [p / sum(before) * 1.0 for p in before]
        self.assertEqual(
            order, sorted(range(len(scaled)), key=lambda i: -scaled[i])
        )

    def test_rank_probabilities_sum_to_one(self):
        """rank 모드의 확률 합은 1 이어야 한다.

        softmax 결과를 상위 k개로 자르면 합이 1 미만이 된다 (실측 0.45). payload·
        자연어·JSON 이 모두 "합 1" 을 전제로 하므로 예측기 단계에서 자르지 않는다.
        후보 수가 자연어 표시 상한(3개)보다 많은 경우로 확인한다.
        """
        try:
            import torch
        except ImportError:
            self.skipTest("torch 없음")
        from traffic_llm.predict_model import TorchPredictor
        from traffic_llm.prediction import build_context

        class Fixed(TorchPredictor):
            """점수를 고정해 확률 계산만 검사한다 (모델 파일 불필요)."""

            def _load(self):
                self._torch = torch
                # 후보 수만큼 서로 다른 점수. 격차를 작게 둔다 — 한 후보가
                # 확률을 독점하면 뒤를 잘라내도 합이 1 에 가까워 검사가 무력해진다.
                return lambda g, c: torch.arange(
                    float(c.shape[0]), dtype=torch.float32
                ) * 0.2

        from traffic_llm.predict_model import Candidate

        # 표시 상한(3개)보다 많은 후보를 만든다. 합성 지도는 3개뿐이므로 후보를
        # 직접 구성한다 — 기하는 서로 달라야 merge_identical 이 합치지 않는다.
        ctx = build_context(self._actor(), self.net, 5.0)
        ctx.candidates = [
            Candidate("직진", 0.2, [(0.0, 0.0), (float(i + 1) * 7.0, 40.0)], f"R{i}")
            for i in range(5)
        ]
        paths = Fixed("unused.pt")(ctx)
        self.assertTrue(paths)
        self.assertAlmostEqual(sum(p.probability for p in paths), 1.0, places=6)

    # ---------------------------------------------- waypoints 모드의 라벨
    def _wp_predictor(self, offsets):
        """주어진 상대좌표를 그대로 내는 waypoints 예측기 (torch 불필요)."""
        from traffic_llm.predict_model import TorchPredictor

        class Fixed(TorchPredictor):
            def _load(self):
                return None

            def __call__(inner, ctx):        # noqa: N805
                class Out:
                    def reshape(self, *a):
                        return self

                    def tolist(self):
                        return offsets
                return inner._from_waypoints(ctx, Out())

        return Fixed("unused.pt", mode="waypoints")

    def _cand_offsets(self, ctx, idx):
        """후보 idx 의 웨이포인트를 액터 기준 상대좌표로."""
        from traffic_llm.prediction import candidate_to_path

        pth = candidate_to_path(ctx, ctx.candidates[idx], 0.0)
        e0, n0 = ctx.actor.world_xy
        return [[w[0] - e0, w[1] - n0] for w in pth.waypoints[1:]]

    def test_waypoint_label_follows_the_geometry_not_the_prior(self):
        """라벨을 사전확률 최고 후보에서 베끼면 좌표와 모순된다 (실측 8.9%)."""
        from traffic_llm.prediction import build_context

        ctx = build_context(self._actor(), self.net, 5.0)
        priors = [c.prior for c in ctx.candidates]
        top = priors.index(max(priors))
        # 사전확률 1순위가 **아닌**, 기동 라벨이 다른 후보를 고른다
        other = next(
            (i for i, c in enumerate(ctx.candidates)
             if c.maneuver != ctx.candidates[top].maneuver), None
        )
        self.assertIsNotNone(other, "테스트 전제: 기동이 다른 후보가 있어야 한다")
        out = self._wp_predictor(self._cand_offsets(ctx, other))(ctx)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].maneuver, ctx.candidates[other].maneuver,
                         "좌표가 가리키는 후보의 라벨이어야 한다")
        self.assertNotEqual(out[0].maneuver, ctx.candidates[top].maneuver)

    def test_waypoint_off_candidate_is_reported_not_hidden(self):
        """어느 후보도 안 따라가면 그렇게 적어야 한다 — 실제로 21% 다."""
        from traffic_llm.prediction import build_context

        ctx = build_context(self._actor(), self.net, 5.0)
        # 도로와 무관한 옆방향 궤적
        far = [[80.0 + 10.0 * k, -300.0] for k in range(5)]
        out = self._wp_predictor(far)(ctx)
        self.assertIn("후보 밖", out[0].maneuver)
        self.assertEqual(out[0].to_roads, [],
                         "따라가지 않는 도로 이름을 붙이면 안 된다")

    def test_waypoint_label_survives_a_speed_difference(self):
        """같은 길을 다른 속도로 가면 여전히 그 길의 라벨이어야 한다.

        같은 시각끼리 점–점으로 재면 가속만으로 몇 미터씩 벌어져 "후보 밖" 으로
        오판된다. 기동 라벨은 **어느 길인가**를 답해야 한다.
        """
        from traffic_llm.prediction import build_context

        ctx = build_context(self._actor(), self.net, 5.0)
        base = self._cand_offsets(ctx, 0)
        fast = [[p[0] * 1.6, p[1] * 1.6] for p in base]   # 60% 빠르게
        out = self._wp_predictor(fast)(ctx)
        self.assertNotIn("후보 밖", out[0].maneuver)
        self.assertEqual(out[0].maneuver, ctx.candidates[0].maneuver)

    def _wp_predictor_with_logits(self, offsets, logits):
        """(좌표, 기동 로짓) 을 내는 예측기 — 권장 형태."""
        from traffic_llm.predict_model import TorchPredictor

        class Vec(list):
            def reshape(self, *a):
                return self

            def tolist(self):
                return list(self)

        class Fixed(TorchPredictor):
            def _load(self):
                return None

            def __call__(inner, ctx):        # noqa: N805
                return inner._from_waypoints(ctx, (Vec(offsets), Vec(logits)))

        return Fixed("unused.pt", mode="waypoints")

    def test_maneuver_comes_from_the_model_when_it_reports_one(self):
        """기동 로짓을 주면 **모델의 분류**를 쓴다 (좌표에서 되짚지 않는다)."""
        from traffic_llm.predict_model import MANEUVER_CLASSES
        from traffic_llm.prediction import build_context

        ctx = build_context(self._actor(), self.net, 5.0)
        # 좌표는 후보 0 을 따라가지만 모델은 '우회전' 이라고 분류한다
        offs = self._cand_offsets(ctx, 0)
        want = "우회전"
        logits = [9.0 if m == want else 0.0 for m in MANEUVER_CLASSES]
        out = self._wp_predictor_with_logits(offs, logits)(ctx)
        self.assertEqual(out[0].maneuver, want)

    def test_off_candidate_class_suppresses_the_road_name(self):
        """'후보 밖' 이면 진입 도로를 붙이지 않는다 — 따라가지 않는 도로다."""
        from traffic_llm.predict_model import (
            MANEUVER_CLASSES,
            OFF_CANDIDATE_LABEL,
        )
        from traffic_llm.prediction import build_context

        ctx = build_context(self._actor(), self.net, 5.0)
        logits = [9.0 if m == OFF_CANDIDATE_LABEL else 0.0
                  for m in MANEUVER_CLASSES]
        out = self._wp_predictor_with_logits(
            self._cand_offsets(ctx, 0), logits)(ctx)
        self.assertEqual(out[0].maneuver, OFF_CANDIDATE_LABEL)
        self.assertEqual(out[0].to_roads, [])

    def test_wrong_logit_count_is_an_error_not_a_silent_label(self):
        from traffic_llm.prediction import build_context

        ctx = build_context(self._actor(), self.net, 5.0)
        with self.assertRaises(ValueError) as cm:
            self._wp_predictor_with_logits(
                self._cand_offsets(ctx, 0), [1.0, 2.0])(ctx)
        self.assertIn("기동 로짓", str(cm.exception))

    def test_model_maneuver_with_no_such_candidate_is_flagged(self):
        """모델이 지도에 없는 기동을 고르면 감추지 않고 그렇게 적는다."""
        from traffic_llm.predict_model import MANEUVER_CLASSES
        from traffic_llm.prediction import build_context

        ctx = build_context(self._actor(), self.net, 5.0)
        present = {c.maneuver for c in ctx.candidates}
        absent = next((m for m in ("유턴", "우회전", "좌회전")
                       if m not in present), None)
        if absent is None:
            self.skipTest("이 지도에는 모든 기동 후보가 있다")
        logits = [9.0 if m == absent else 0.0 for m in MANEUVER_CLASSES]
        out = self._wp_predictor_with_logits(
            self._cand_offsets(ctx, 0), logits)(ctx)
        self.assertIn("해당 후보 없음", out[0].maneuver)
        self.assertEqual(out[0].to_roads, [])

    def test_pipeline_config_carries_the_predictor(self):
        self.assertIsNone(PipelineConfig().predictor)
