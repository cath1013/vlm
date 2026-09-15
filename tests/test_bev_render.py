"""BEV 시계열 렌더링 테스트.

이미지의 '보기 좋음'은 검사할 수 없으므로, 그림이 옳으려면 반드시 성립해야
하는 것들을 검사한다: 기하 유틸의 수치, 좌표변환의 왕복, 파일명 규약과 정렬,
잡음 궤적 절단, Pillow 없을 때의 SVG 폴백.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_llm import bev_render
from traffic_llm.bev_render import (
    BevConfig,
    BevRenderer,
    ViewTransform,
    _plausible_trail,
    _stamp,
    clip_segment_convex,
    convex_hull,
    heading_arrow,
    offset_polyline,
    point_in_convex,
    rect_corners,
    road_side_offsets,
    road_surface_polygon,
    road_surface_quads,
    short_id,
    subtract_intervals,
    write_index_html,
)
from traffic_llm.config import LaneConfig
from traffic_llm.geometry import LocalENU
from traffic_llm.roadmap import Road, RoadNetwork
from traffic_llm.schemas import (
    ActorState,
    InfraState,
    Interaction,
    RoadPlacement,
    SceneSnapshot,
)


def make_network() -> RoadNetwork:
    """남북 1차선 왕복 도로 하나 + 동서 도로 하나."""
    roads = [
        Road(
            road_id="ns",
            name="High St",
            poly=[(0.0, -100.0), (0.0, 100.0)],
            lanes_forward=2,
            lanes_backward=2,
            length_m=200.0,
        ),
        Road(
            road_id="ew",
            name="Broad St",
            poly=[(-100.0, 0.0), (100.0, 0.0)],
            lanes_forward=1,
            lanes_backward=1,
            length_m=200.0,
        ),
    ]
    return RoadNetwork(roads, LaneConfig(), LocalENU(39.9612, -83.0007))


def make_crossing_network() -> RoadNetwork:
    """원점에서 만나는 4개 도로 — 끝점 클러스터링으로 교차로가 생긴다.

    make_network() 의 두 도로는 원점에서 **교차만** 하고 끝점이 ±100m 에 있어
    교차로 노드가 만들어지지 않는다. 교차로 관련 검사에는 이 망을 쓴다.
    """
    def road(rid, name, a, b):
        return Road(
            road_id=rid,
            name=name,
            poly=[a, b],
            lanes_forward=2,
            lanes_backward=2,
            length_m=math.dist(a, b),
        )

    roads = [
        road("high_s", "High St", (0.0, -150.0), (0.0, 0.0)),
        road("high_n", "High St", (0.0, 0.0), (0.0, 150.0)),
        road("broad_w", "Broad St", (-150.0, 0.0), (0.0, 0.0)),
        road("broad_e", "Broad St", (0.0, 0.0), (150.0, 0.0)),
    ]
    return RoadNetwork(roads, LaneConfig(), LocalENU(39.9612, -83.0007))


def actor(
    aid: str,
    xy,
    heading=0.0,
    cls: str = "car",
    kind: str = "observed",
    speed: float = 10.0,
    placement: RoadPlacement = None,
) -> ActorState:
    return ActorState(
        actor_id=aid,
        kind=kind,
        cls=cls,
        world_xy=xy,
        heading_deg=heading,
        speed_mps=speed,
        accel_mps2=None,
        observed_by=["V1"],
        placement=placement,
    )


def snapshot(t: float, actors, frame: int = 0) -> SceneSnapshot:
    return SceneSnapshot(
        t=t,
        actors=list(actors),
        interactions=[],
        area_name="test",
        frame_idx=frame,
    )


class TestGeometryHelpers(unittest.TestCase):
    def test_rect_corners_preserves_center_and_size(self):
        for heading in (0.0, 37.0, 90.0, 180.0, 271.5):
            c = rect_corners((10.0, -5.0), heading, 4.6, 1.9)
            self.assertEqual(len(c), 4)
            ce = sum(p[0] for p in c) / 4
            cn = sum(p[1] for p in c) / 4
            self.assertAlmostEqual(ce, 10.0, places=6)
            self.assertAlmostEqual(cn, -5.0, places=6)
            # 앞변(0-1)은 폭, 옆변(1-2)은 길이
            self.assertAlmostEqual(math.dist(c[0], c[1]), 1.9, places=6)
            self.assertAlmostEqual(math.dist(c[1], c[2]), 4.6, places=6)

    def test_rect_corners_heading_north(self):
        # 방위 0° = 북 → 전방 코너는 북쪽(n>0)
        c = rect_corners((0.0, 0.0), 0.0, 4.0, 2.0)
        self.assertAlmostEqual(c[0][1], 2.0, places=6)
        self.assertAlmostEqual(c[2][1], -2.0, places=6)

    def test_rect_corners_heading_east(self):
        # 방위 90° = 동 → 전방 코너는 동쪽(e>0)
        c = rect_corners((0.0, 0.0), 90.0, 4.0, 2.0)
        self.assertAlmostEqual(c[0][0], 2.0, places=6)
        self.assertAlmostEqual(c[2][0], -2.0, places=6)

    def test_heading_arrow_points_forward(self):
        tip, a, b = heading_arrow((0.0, 0.0), 90.0, 5.0)
        self.assertGreater(tip[0], a[0])
        self.assertGreater(tip[0], b[0])
        # 밑변 두 점은 진행축에 대해 대칭
        self.assertAlmostEqual(a[1], -b[1], places=6)

    def test_offset_polyline_shifts_left(self):
        # 북향 직선의 좌측은 서쪽(e<0)
        out = offset_polyline([(0.0, 0.0), (0.0, 10.0)], 3.0)
        self.assertEqual(len(out), 2)
        for p in out:
            self.assertAlmostEqual(p[0], -3.0, places=6)
        # 음수 오프셋은 우측(동쪽)
        out = offset_polyline([(0.0, 0.0), (0.0, 10.0)], -3.0)
        for p in out:
            self.assertAlmostEqual(p[0], 3.0, places=6)

    def test_offset_polyline_degenerate(self):
        self.assertEqual(offset_polyline([(1.0, 2.0)], 3.0), [(1.0, 2.0)])

    def test_road_surface_polygon_covers_all_lanes(self):
        net = make_network()
        poly = road_surface_polygon(net.roads["ns"], 3.5, "right")
        self.assertEqual(len(poly), 4)  # 2점 도로 → 좌 2 + 우 2
        # 폭 = (전방 2 + 후방 2) 차선 × 3.5 = 14m
        es = [p[0] for p in poly]
        self.assertAlmostEqual(max(es) - min(es), 14.0, places=6)

    def test_road_surface_polygon_oneway_centered(self):
        road = Road(road_id="x", name="x", poly=[(0.0, 0.0), (0.0, 10.0)],
                    oneway=True, lanes_forward=3, length_m=10.0)
        poly = road_surface_polygon(road, 3.0, "right")
        es = [p[0] for p in poly]
        self.assertAlmostEqual(max(es), 4.5, places=6)
        self.assertAlmostEqual(min(es), -4.5, places=6)


class TestViewTransform(unittest.TestCase):
    def setUp(self):
        self.v = ViewTransform((10.0, 20.0), extent_m=100.0,
                               width_px=200, height_px=100)

    def test_center_maps_to_image_center(self):
        self.assertEqual(self.v.to_px((10.0, 20.0)), (100.0, 50.0))

    def test_north_is_up_east_is_right(self):
        self.assertGreater(self.v.to_px((20.0, 20.0))[0], 100.0)  # 동 → 오른쪽
        self.assertLess(self.v.to_px((10.0, 30.0))[1], 50.0)      # 북 → 위쪽

    def test_scale_and_extent(self):
        self.assertAlmostEqual(self.v.mpp, 0.5)
        self.assertAlmostEqual(self.v.m_to_px(10.0), 20.0)
        self.assertAlmostEqual(self.v.extent_n_m, 50.0)

    def test_visible_respects_padding(self):
        self.assertTrue(self.v.visible((10.0, 20.0), pad_m=0.0))
        # 가로 반폭 50m 밖, 여유 0 → 보이지 않음
        self.assertFalse(self.v.visible((70.0, 20.0), pad_m=0.0))
        self.assertTrue(self.v.visible((70.0, 20.0), pad_m=20.0))


class TestStampAndNames(unittest.TestCase):
    def test_stamp_is_zero_padded_and_sortable(self):
        stamps = [_stamp(t) for t in (0.0, 0.5, 2.0, 9.5, 10.0, 100.25)]
        self.assertEqual(stamps[0], "0000p00s")
        self.assertEqual(stamps[1], "0000p50s")
        # 사전순 정렬이 시간순과 일치해야 한다 (이미지 뷰어 정렬)
        self.assertEqual(stamps, sorted(stamps))

    def test_stamp_has_no_dot(self):
        # 파일명에 점이 더 있으면 확장자 판별이 애매해진다
        self.assertNotIn(".", _stamp(3.25))

    def test_short_id(self):
        self.assertEqual(short_id("EGO_ego_vehicle"), "EGO")
        self.assertEqual(short_id("EGO_other_vehicle_behind"), "OTH-B")
        self.assertEqual(short_id("V007"), "V007")
        # 등록되지 않은 관측차량 이름도 EGO_ 접두어는 벗긴다
        self.assertEqual(short_id("EGO_bus_42"), "bus_42")


class TestPlausibleTrail(unittest.TestCase):
    def test_keeps_smooth_history(self):
        pts = [(0.0, (0.0, 0.0)), (0.5, (0.0, 5.0)), (1.0, (0.0, 10.0))]
        out = _plausible_trail(pts, speed_mps=10.0, floor_mps=8.0, factor=2.5)
        self.assertEqual(len(out), 3)

    def test_cuts_at_position_jump(self):
        # 두 번째 → 세 번째 사이 40m/0.5s = 80m/s 는 잡음
        pts = [
            (0.0, (0.0, 0.0)),
            (0.5, (0.0, 5.0)),
            (1.0, (0.0, 45.0)),
            (1.5, (0.0, 50.0)),
        ]
        out = _plausible_trail(pts, speed_mps=10.0, floor_mps=8.0, factor=2.5)
        self.assertEqual(out, [(0.0, 45.0), (0.0, 50.0)])

    def test_floor_allows_slow_actor_noise(self):
        # 정지 차량(속도 0)도 floor_mps 만큼은 위치 잡음을 허용한다
        pts = [(0.0, (0.0, 0.0)), (0.5, (0.0, 3.0))]
        out = _plausible_trail(pts, speed_mps=0.0, floor_mps=8.0, factor=2.5)
        self.assertEqual(len(out), 2)

    def test_single_point(self):
        self.assertEqual(
            _plausible_trail([(0.0, (1.0, 2.0))], None, 8.0, 2.5), [(1.0, 2.0)]
        )

    def test_class_cap_beats_inflated_speed(self):
        """잡음으로 부풀려진 속도가 자기 잡음 궤적을 정당화하지 못해야 한다.

        보행자 추정속도 5m/s(잡음) × factor 2.5 = 12.5m/s 를 허용하면 인도 위
        지그재그가 그대로 그려진다. 클래스 상한 2.5m/s 가 이겨야 한다.
        """
        pts = [(0.0, (0.0, 0.0)), (0.5, (3.0, 0.0)), (1.0, (3.5, 0.0))]
        loose = _plausible_trail(pts, 5.0, 2.0, 2.5, cap_mps=None)
        self.assertEqual(len(loose), 3)
        capped = _plausible_trail(pts, 5.0, 2.0, 2.5, cap_mps=2.5)
        self.assertEqual(capped, [(3.0, 0.0), (3.5, 0.0)])

    def test_unknown_speed_uses_class_cap(self):
        pts = [(0.0, (0.0, 0.0)), (0.5, (1.0, 0.0)), (1.0, (2.0, 0.0))]
        # 0.5초에 1m = 2m/s — 차량 상한 아래, 보행자 상한 아래
        self.assertEqual(len(_plausible_trail(pts, None, 2.0, 2.5, 40.0)), 3)
        # 상한을 1m/s 로 낮추면 끊긴다
        self.assertEqual(len(_plausible_trail(pts, None, 2.0, 2.5, 1.0)), 1)

    def test_bev_config_has_class_caps(self):
        cfg = BevConfig()
        self.assertLess(cfg.trail_speed_by_class["person"], 4.0)
        self.assertGreater(cfg.trail_speed_by_class["car"], 20.0)


class TestInImageTextIsLatin(unittest.TestCase):
    """이미지 안 문자열은 모두 라틴 문자여야 한다.

    이미지가 어느 환경에서 열릴지, 어떤 폰트가 깔려 있을지 알 수 없다.
    한글을 넣으면 폰트가 없는 환경에서 두부(□)로 렌더링된다.
    """

    @staticmethod
    def _rich_snapshot():
        """HUD·범례·라벨·인프라·TTC·축척바를 모두 태우는 스냅샷."""
        snap = snapshot(
            2.5,
            [
                actor("EGO_ego_vehicle", (0.0, 0.0), kind="ego"),
                actor("EGO_other_vehicle_behind", (-8.0, 0.0), kind="ego"),
                actor("V002", (6.0, 4.0), heading=180.0),
                actor("P003", (-6.0, 3.0), heading=None, cls="person", speed=1.2),
            ],
            frame=25,
        )
        snap.interactions.append(
            Interaction(
                subject_id="EGO_ego_vehicle",
                object_id="V002",
                kind="crossing",
                ttc_s=1.8,
                gap_m=7.0,
                note="junction conflict",
            )
        )
        snap.infrastructure.append(
            InfraState(
                infra_id="infra_1",
                world_xy=(10.0, 9.0),
                height_m=5.5,
                heading_deg=200.0,
                n_observed=2,
            )
        )
        return snap

    def test_no_hangul_is_ever_drawn(self):
        """실제로 그려지는 모든 문자열을 가로채 검사한다.

        소스의 dr.text(...) 만 정규식으로 훑으면 f-string 으로 미리 조립해 변수로
        넘기는 문구(HUD 카운트 등)를 놓친다.
        """
        from PIL import ImageDraw

        seen = []
        orig = ImageDraw.ImageDraw.text

        def spy(self, xy, text="", *a, **kw):
            seen.append(str(text))
            return orig(self, xy, text, *a, **kw)

        ImageDraw.ImageDraw.text = spy
        try:
            r = BevRenderer(make_network(), BevConfig(width_px=700, height_px=520))
            with tempfile.TemporaryDirectory() as d:
                r.render_sequence([self._rich_snapshot()], d, "s",
                                  title="Town05 - scenario00036")
        finally:
            ImageDraw.ImageDraw.text = orig

        self.assertTrue(seen, "아무 문자열도 그리지 않았다 — 검사가 무의미하다")
        for text in seen:
            hangul = [ch for ch in text if "가" <= ch <= "힣"]
            self.assertFalse(hangul, f"이미지에 한글이 들어간다: {text!r}")
        # 검사 범위 확인 — HUD·범례·라벨·인프라·TTC 가 모두 태워졌는가
        joined = " | ".join(seen)
        for expect in ("observers", "road users", "infra", "TTC", "EGO"):
            self.assertIn(expect, joined, f"검사에 {expect} 문구가 안 잡혔다")

    def test_legend_and_hud_render(self):
        snaps = [
            snapshot(
                0.0,
                [
                    actor("EGO_ego_vehicle", (0.0, 0.0), kind="ego"),
                    actor("V002", (6.0, 4.0)),
                ],
            )
        ]
        r = BevRenderer(make_network(), BevConfig(width_px=500, height_px=380))
        with tempfile.TemporaryDirectory() as d:
            out = r.render_sequence(snaps, d, "s")[0]
            self.assertTrue(os.path.isfile(out))

    def test_svg_fallback_text_is_latin(self):
        orig = bev_render._HAS_PIL
        bev_render._HAS_PIL = False
        try:
            r = BevRenderer(make_network(), BevConfig(width_px=400, height_px=300))
            snaps = [snapshot(1.5, [actor("EGO_other_vehicle_behind", (0.0, 0.0),
                                         kind="ego")])]
            with tempfile.TemporaryDirectory() as d:
                path = r.render_sequence(snaps, d, "s")[0]
                with open(path, encoding="utf-8") as f:
                    body = f.read()
            hangul = [ch for ch in body if "가" <= ch <= "힣"]
            self.assertFalse(hangul, "SVG 안에 한글이 들어간다")
        finally:
            bev_render._HAS_PIL = orig


class TestActorHeadingFallback(unittest.TestCase):
    def setUp(self):
        self.r = BevRenderer(make_network(), BevConfig())

    def _placement(self, bearing: float) -> RoadPlacement:
        return RoadPlacement(
            road_id="ns",
            road_name="High St",
            s_m=10.0,
            lateral_offset_m=-1.75,
            direction_label="북행",
            bearing_deg=bearing,
            lane_index=1,
            lane_count=2,
            speed_limit_kph=50.0,
            dist_to_next_junction_m=None,
            next_junction_id=None,
        )

    def test_measured_heading_is_not_assumed(self):
        h, assumed = self.r._actor_heading(actor("V1", (0.0, 0.0), heading=42.0))
        self.assertAlmostEqual(h, 42.0)
        self.assertFalse(assumed)

    def test_falls_back_to_road_bearing(self):
        a = actor("V1", (0.0, 0.0), heading=None,
                  placement=self._placement(357.0))
        h, assumed = self.r._actor_heading(a)
        self.assertAlmostEqual(h, 357.0)
        self.assertTrue(assumed)

    def test_no_heading_no_placement(self):
        h, assumed = self.r._actor_heading(actor("V1", (0.0, 0.0), heading=None))
        self.assertIsNone(h)
        self.assertFalse(assumed)


class TestComputeView(unittest.TestCase):
    def test_ego_focus_ignores_distant_observed_actors(self):
        """인프라가 본 원거리 차량이 시야를 끌어당기지 않아야 한다."""
        snaps = [
            snapshot(
                0.0,
                [
                    actor("EGO_V1", (0.0, 0.0), kind="ego"),
                    actor("X", (400.0, 400.0)),  # 아주 먼 주차 차량
                ],
            )
        ]
        r = BevRenderer(make_network(), BevConfig(focus="ego", margin_m=10.0))
        view = r.compute_view(snaps)
        self.assertLess(view.extent_m, 200.0)
        r_all = BevRenderer(make_network(), BevConfig(focus="all", margin_m=10.0))
        self.assertGreater(r_all.compute_view(snaps).extent_m, view.extent_m)

    def test_extent_is_clamped(self):
        cfg = BevConfig(min_extent_m=80.0, max_extent_m=150.0, focus="all")
        r = BevRenderer(make_network(), cfg)
        tight = r.compute_view([snapshot(0.0, [actor("A", (0.0, 0.0))])])
        self.assertAlmostEqual(tight.extent_m, 80.0)
        wide = r.compute_view(
            [snapshot(0.0, [actor("A", (-500.0, 0.0)), actor("B", (500.0, 0.0))])]
        )
        self.assertAlmostEqual(wide.extent_m, 150.0)

    def test_ego_colors_are_distinct(self):
        snaps = [
            snapshot(
                0.0,
                [
                    actor("EGO_a", (0.0, 0.0), kind="ego"),
                    actor("EGO_b", (5.0, 0.0), kind="ego"),
                    actor("V1", (10.0, 0.0)),
                ],
            )
        ]
        cols = BevRenderer(make_network(), BevConfig()).assign_ego_colors(snaps)
        self.assertEqual(set(cols), {"EGO_a", "EGO_b"})
        self.assertEqual(len(set(cols.values())), 2)


class TestRenderSequence(unittest.TestCase):
    def _snaps(self):
        out = []
        for k in range(4):
            t = k * 0.5
            out.append(
                snapshot(
                    t,
                    [
                        actor("EGO_V1", (0.0, -20.0 + 6.0 * k), kind="ego"),
                        actor("V002", (3.5, 10.0 - 5.0 * k), heading=180.0),
                        actor("P003", (-8.0, 0.0), heading=None, cls="person",
                              speed=1.2),
                    ],
                    frame=k * 5 + 1,
                )
            )
        # 위험 상호작용이 있는 프레임 하나 — 위험 연결선 경로를 태운다
        out[-1].interactions.append(
            Interaction(
                subject_id="EGO_V1",
                object_id="V002",
                kind="정면 접근",
                ttc_s=1.5,
                gap_m=8.0,
                note="교차로 상충",
            )
        )
        out[-1].infrastructure.append(
            InfraState(
                infra_id="infra_1",
                world_xy=(12.0, 12.0),
                height_m=5.5,
                heading_deg=210.0,
                n_observed=3,
            )
        )
        return out

    def test_filenames_and_count(self):
        snaps = self._snaps()
        r = BevRenderer(make_network(), BevConfig(width_px=400, height_px=300))
        with tempfile.TemporaryDirectory() as d:
            paths = r.render_sequence(snaps, d, "Town03_scenario00024")
            self.assertEqual(len(paths), len(snaps))
            names = [os.path.basename(p) for p in paths]
            self.assertEqual(names[0], "Town03_scenario00024_0000p00s.jpg")
            self.assertEqual(names[1], "Town03_scenario00024_0000p50s.jpg")
            self.assertEqual(names, sorted(names))  # 사전순 = 시간순
            for p in paths:
                self.assertTrue(os.path.isfile(p))
                self.assertGreater(os.path.getsize(p), 0)

    def test_png_format(self):
        r = BevRenderer(make_network(), BevConfig(width_px=300, height_px=200,
                                                 image_format="png"))
        with tempfile.TemporaryDirectory() as d:
            paths = r.render_sequence(self._snaps()[:1], d, "s")
            self.assertTrue(paths[0].endswith(".png"))

    def test_empty_sequence(self):
        r = BevRenderer(make_network(), BevConfig())
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(r.render_sequence([], d, "s"), [])

    def test_no_network_still_renders(self):
        """도로망 없이도(지도 미제공) 액터만 그려야 한다."""
        r = BevRenderer(None, BevConfig(width_px=300, height_px=200))
        with tempfile.TemporaryDirectory() as d:
            paths = r.render_sequence(self._snaps()[:1], d, "s")
            self.assertTrue(os.path.isfile(paths[0]))

    def test_svg_fallback_without_pillow(self):
        snaps = self._snaps()
        orig = bev_render._HAS_PIL
        bev_render._HAS_PIL = False
        try:
            r = BevRenderer(make_network(), BevConfig(width_px=400, height_px=300))
            with tempfile.TemporaryDirectory() as d:
                paths = r.render_sequence(snaps[:2], d, "s")
                self.assertTrue(all(p.endswith(".svg") for p in paths))
                with open(paths[0], encoding="utf-8") as f:
                    body = f.read()
                self.assertIn("<svg", body)
                self.assertIn("</svg>", body)
        finally:
            bev_render._HAS_PIL = orig

    def test_index_html_lists_images_in_order(self):
        snaps = self._snaps()
        r = BevRenderer(make_network(), BevConfig(width_px=300, height_px=200))
        with tempfile.TemporaryDirectory() as d:
            paths = r.render_sequence(snaps, d, "s")
            idx = write_index_html(paths, os.path.join(d, "index.html"))
            with open(idx, encoding="utf-8") as f:
                html = f.read()
            pos = [html.index(os.path.basename(p)) for p in paths]
            self.assertEqual(pos, sorted(pos))

    def test_fixed_view_is_shared_across_frames(self):
        """고정 시야면 프레임마다 화면이 흔들리지 않아야 한다."""
        snaps = self._snaps()
        r = BevRenderer(make_network(), BevConfig(fixed_view=True))
        v = r.compute_view(snaps)
        for s in snaps:
            self.assertNotEqual(r.compute_view([s]).center, None)
        # 전체 시퀀스 시야는 개별 프레임 시야보다 좁지 않다
        self.assertGreaterEqual(v.extent_m, min(
            r.compute_view([s]).extent_m for s in snaps
        ))


class TestConvexAndClipping(unittest.TestCase):
    SQ = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]  # 반시계

    def test_convex_hull_is_counter_clockwise(self):
        hull = convex_hull([(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)])
        self.assertEqual(len(hull), 4)
        # 부호 있는 면적 > 0 이면 반시계 (clip_segment_convex 가 이를 가정한다)
        area = sum(
            hull[i][0] * hull[(i + 1) % 4][1] - hull[(i + 1) % 4][0] * hull[i][1]
            for i in range(4)
        )
        self.assertGreater(area, 0)

    def test_convex_hull_drops_interior_points(self):
        hull = convex_hull([(0, 0), (4, 0), (4, 4), (0, 4), (2, 2), (1, 3)])
        self.assertNotIn((2, 2), hull)
        self.assertEqual(len(hull), 4)

    def test_point_in_convex(self):
        self.assertTrue(point_in_convex((5.0, 5.0), self.SQ))
        self.assertFalse(point_in_convex((15.0, 5.0), self.SQ))
        self.assertFalse(point_in_convex((-0.1, 5.0), self.SQ))
        self.assertTrue(point_in_convex((0.0, 5.0), self.SQ))  # 변 위

    def test_point_in_convex_degenerate(self):
        self.assertFalse(point_in_convex((0.0, 0.0), [(0.0, 0.0), (1.0, 1.0)]))

    def test_clip_segment_fully_inside(self):
        iv = clip_segment_convex((2.0, 2.0), (8.0, 8.0), self.SQ)
        self.assertIsNotNone(iv)
        self.assertAlmostEqual(iv[0], 0.0)
        self.assertAlmostEqual(iv[1], 1.0)

    def test_clip_segment_fully_outside(self):
        self.assertIsNone(
            clip_segment_convex((20.0, 20.0), (30.0, 30.0), self.SQ)
        )

    def test_clip_segment_parallel_outside(self):
        # 사각형 위쪽에 평행 — 어떤 t 에도 내부가 아니다
        self.assertIsNone(
            clip_segment_convex((-5.0, 20.0), (15.0, 20.0), self.SQ)
        )

    def test_clip_segment_crossing(self):
        # y=5 를 따라 왼쪽 밖에서 오른쪽 밖까지 (길이 20, 내부는 0~10)
        iv = clip_segment_convex((-5.0, 5.0), (15.0, 5.0), self.SQ)
        self.assertAlmostEqual(iv[0], 0.25)
        self.assertAlmostEqual(iv[1], 0.75)

    def test_clip_segment_one_endpoint_inside(self):
        iv = clip_segment_convex((5.0, 5.0), (25.0, 5.0), self.SQ)
        self.assertAlmostEqual(iv[0], 0.0)
        self.assertAlmostEqual(iv[1], 0.25)

    def test_subtract_intervals(self):
        self.assertEqual(subtract_intervals([]), [(0.0, 1.0)])
        self.assertEqual(
            subtract_intervals([(0.25, 0.75)]), [(0.0, 0.25), (0.75, 1.0)]
        )
        self.assertEqual(subtract_intervals([(0.0, 1.0)]), [])
        # 겹치는 구간은 합집합으로
        self.assertEqual(
            subtract_intervals([(0.1, 0.4), (0.3, 0.6)]),
            [(0.0, 0.1), (0.6, 1.0)],
        )

    def test_subtract_intervals_drops_slivers(self):
        self.assertEqual(subtract_intervals([(0.0, 1.0 - 1e-12)]), [])


class TestRoadSurfaceQuads(unittest.TestCase):
    def test_quad_count_and_width(self):
        road = Road(road_id="x", name="x",
                    poly=[(0.0, 0.0), (0.0, 10.0), (0.0, 20.0)],
                    lanes_forward=2, lanes_backward=2, length_m=20.0)
        quads = road_surface_quads(road, 3.5)
        self.assertEqual(len(quads), 2)
        for q in quads:
            self.assertEqual(len(q), 4)
            es = [p[0] for p in q]
            self.assertAlmostEqual(max(es) - min(es), 14.0, places=6)

    def test_sharp_corner_does_not_pinch(self):
        """급한 꺾임에서 노면이 쐐기로 찌그러지지 않아야 한다.

        단일 다각형(좌 경계 + 역순 우 경계)은 곡률 반경보다 오프셋이 크면 내측
        경계가 스스로 교차해 면적이 사라진다. 세그먼트별 사각형은 안전하다.
        """
        road = Road(road_id="x", name="x",
                    poly=[(0.0, 0.0), (6.0, 0.0), (6.0, 6.0)],
                    lanes_forward=2, lanes_backward=2, length_m=12.0)

        def area(poly):
            n = len(poly)
            return abs(sum(
                poly[i][0] * poly[(i + 1) % n][1]
                - poly[(i + 1) % n][0] * poly[i][1]
                for i in range(n)
            )) / 2.0

        quads = road_surface_quads(road, 3.5)
        self.assertEqual(len(quads), 2)
        for q in quads:
            # 각 사각형은 대략 세그먼트길이 × 도로폭 만큼의 면적을 가진다
            self.assertGreater(area(q), 6.0 * 14.0 * 0.5)

    def test_side_offsets_follow_drive_side(self):
        road = Road(road_id="x", name="x", poly=[(0.0, 0.0), (0.0, 10.0)],
                    lanes_forward=2, lanes_backward=1, length_m=10.0)
        lo, ro = road_side_offsets(road, 3.0, "right")
        self.assertAlmostEqual(lo, 3.0)   # 대향 1차선 = 좌측
        self.assertAlmostEqual(ro, -6.0)  # 진행 2차선 = 우측
        lo, ro = road_side_offsets(road, 3.0, "left")
        self.assertAlmostEqual(lo, 6.0)
        self.assertAlmostEqual(ro, -3.0)

    def test_oneway_is_centered(self):
        road = Road(road_id="x", name="x", poly=[(0.0, 0.0), (0.0, 10.0)],
                    oneway=True, lanes_forward=3, length_m=10.0)
        lo, ro = road_side_offsets(road, 3.0)
        self.assertAlmostEqual(lo, 4.5)
        self.assertAlmostEqual(ro, -4.5)


class TestJunctionBlanket(unittest.TestCase):
    """교차로가 끊긴 것처럼 보이지 않아야 한다."""

    def setUp(self):
        self.net = make_crossing_network()
        self.r = BevRenderer(self.net, BevConfig(), lane_width_m=3.25)

    def test_blanket_covers_junction_centre(self):
        blankets = self.r._junction_blankets()
        self.assertEqual(len(blankets), 1)  # 4지 교차로 하나
        hull, c, bb = blankets[0]
        self.assertAlmostEqual(c[0], 0.0, places=6)
        self.assertAlmostEqual(c[1], 0.0, places=6)
        self.assertTrue(point_in_convex((0.0, 0.0), hull))
        # 교차로 사각형의 네 귀퉁이까지 덮어야 한다 (4차선 × 3.25 = 반폭 6.5m).
        # 끝점 단면만 모으면 껍질이 마름모가 되어 귀퉁이가 빈다.
        for corner in ((6.0, 6.0), (-6.0, 6.0), (6.0, -6.0), (-6.0, -6.0)):
            self.assertTrue(
                point_in_convex(corner, hull), f"귀퉁이 {corner} 가 안 덮인다"
            )
        self.assertFalse(point_in_convex((40.0, 40.0), hull))

    def test_two_road_node_is_not_blanketed(self):
        """도로 2개가 이어지는 지점은 교차로가 아니다 (덮개 대상 아님)."""
        roads = [
            Road(road_id="a", name="s", poly=[(0.0, -50.0), (0.0, 0.0)],
                 lanes_forward=1, lanes_backward=1, length_m=50.0),
            Road(road_id="b", name="s", poly=[(0.0, 0.0), (0.0, 50.0)],
                 lanes_forward=1, lanes_backward=1, length_m=50.0),
        ]
        net = RoadNetwork(roads, LaneConfig(), LocalENU(39.9612, -83.0007))
        self.assertEqual(
            BevRenderer(net, BevConfig())._junction_blankets(), []
        )

    def test_markings_stop_at_junction(self):
        """차선 표시가 교차로를 가로지르지 않아야 한다."""
        blankets = self.r._junction_blankets()
        hull = blankets[0][0]
        any_seg = False
        for road, items in self.r._road_markings():
            for _, runs in items:
                self.assertTrue(runs, f"{road.road_id}: 표시 선이 통째로 사라졌다")
                for run in runs:
                    # 잘린 끝점은 덮개 경계에 정확히 놓이므로, 실제로 그려지는
                    # 선분의 **중간점**이 교차로 안에 없는지를 본다.
                    for i in range(len(run) - 1):
                        mid = (
                            (run[i][0] + run[i + 1][0]) / 2,
                            (run[i][1] + run[i + 1][1]) / 2,
                        )
                        any_seg = True
                        self.assertFalse(
                            point_in_convex(mid, hull),
                            f"{road.road_id}: 표시 선이 교차로를 가로지른다 {mid}",
                        )
        self.assertTrue(any_seg)

    def test_coarse_polyline_keeps_markings(self):
        """정점 2개짜리 도로도 표시 선이 남아야 한다 (정점 단위 클리핑 회귀)."""
        centers = [
            runs
            for road, items in self.r._road_markings()
            for kind, runs in items
            if kind == "center" and road.road_id == "high_s"
        ]
        self.assertEqual(len(centers), 1)
        run = centers[0][0]
        self.assertAlmostEqual(run[0][1], -150.0, places=3)
        self.assertLess(run[-1][1], 0.0)      # 교차로 앞에서 끊긴다
        self.assertGreater(run[-1][1], -30.0)  # 그러나 통째로 사라지지는 않는다

    def test_junction_internal_roads_get_no_markings(self):
        """교차로 내부 연결로에는 차선 표시를 그리지 않는다."""
        roads = list(make_crossing_network().roads.values())
        roads.append(
            Road(road_id="conn", name="link",
                 poly=[(0.0, -6.0), (3.0, -3.0), (6.0, 0.0)],
                 oneway=True, lanes_forward=1, length_m=8.5,
                 in_junction=True)
        )
        net = RoadNetwork(roads, LaneConfig(), LocalENU(39.9612, -83.0007))
        r = BevRenderer(net, BevConfig())
        ids = {road.road_id for road, _ in r._road_markings()}
        self.assertNotIn("conn", ids)
        # 노면 자체는 그린다 (교차로 안을 채우기 위해)
        self.assertIn("conn", {road.road_id for road, _ in r._road_surfaces()})

    def test_renders_with_junction(self):
        snaps = [snapshot(0.0, [actor("EGO_V1", (0.0, -20.0), kind="ego")])]
        with tempfile.TemporaryDirectory() as d:
            out = self.r.render_sequence(snaps, d, "s")[0]
            self.assertGreater(os.path.getsize(out), 0)


if __name__ == "__main__":
    unittest.main()


class TestPredictionOverlay(unittest.TestCase):
    """예상 경로 점선.

    기본이 꺼짐이다. 켜면 액터 전부의 점선이 겹치고, 특히 도로에 매칭되지 않은
    액터(인도를 걷는 보행자 등)의 경로는 노면이 그려지지 않는 곳을 가로질러
    정체를 알 수 없는 대각선 조각으로 남는다.
    """

    @staticmethod
    def _actor(**kw):
        from traffic_llm.schemas import ActorState, PredictedPath, RoadPlacement

        base = dict(
            actor_id="V001", kind="observed", cls="car",
            world_xy=(0.0, 0.0), heading_deg=0.0, speed_mps=10.0,
            accel_mps2=0.0, observed_by=["v1"],
            predictions=[
                PredictedPath("직진", 1.0, [(0.0, 0.0), (0.0, 10.0)], 5.0)
            ],
            placement=RoadPlacement(
                road_id="r1", road_name="r", s_m=0.0, lateral_offset_m=-1.6,
                direction_label="북행", bearing_deg=0.0, lane_index=1,
                lane_count=2, speed_limit_kph=50.0,
                dist_to_next_junction_m=50.0, next_junction_id="J0",
            ),
        )
        base.update(kw)
        return ActorState(**base)

    def test_off_by_default(self):
        self.assertFalse(BevConfig().draw_predictions)

    def test_shown_for_a_moving_road_matched_actor(self):
        cfg = BevConfig(draw_predictions=True)
        self.assertTrue(BevRenderer._show_prediction(self._actor(), cfg))

    def test_skipped_when_not_road_matched(self):
        """도로 미매칭 경로는 방위 외삽이라 노면을 벗어난다."""
        cfg = BevConfig(draw_predictions=True)
        self.assertFalse(
            BevRenderer._show_prediction(self._actor(placement=None), cfg)
        )

    def test_skipped_when_slow_or_stopped(self):
        cfg = BevConfig(draw_predictions=True)
        for v in (None, 0.0, cfg.prediction_min_speed_mps - 0.1):
            self.assertFalse(
                BevRenderer._show_prediction(self._actor(speed_mps=v), cfg), v
            )

    def test_skipped_when_path_has_no_length(self):
        from traffic_llm.schemas import PredictedPath

        cfg = BevConfig(draw_predictions=True)
        self.assertFalse(
            BevRenderer._show_prediction(self._actor(predictions=[]), cfg)
        )
        self.assertFalse(
            BevRenderer._show_prediction(
                self._actor(
                    predictions=[PredictedPath("정지 유지", 1.0, [(0.0, 0.0)], 5.0)]
                ),
                cfg,
            )
        )

    def test_road_requirement_can_be_relaxed(self):
        cfg = BevConfig(draw_predictions=True, prediction_require_road=False)
        self.assertTrue(
            BevRenderer._show_prediction(self._actor(placement=None), cfg)
        )

    def test_legend_reserves_an_extra_row_when_enabled(self):
        """점선을 그리면 범례 항목이 늘어 HUD 예약 영역도 커져야 한다."""
        off = BevRenderer(None, BevConfig(draw_predictions=False))
        on = BevRenderer(None, BevConfig(draw_predictions=True))
        h_off = off._hud_rects(4)[-1][3]
        h_on = on._hud_rects(4)[-1][3]
        self.assertGreater(h_on, h_off)


class TestFontResolution(unittest.TestCase):
    """트루타입 폰트를 찾는지.

    못 찾으면 Pillow 비트맵 기본 폰트로 떨어지고, 그것은 범례의 em dash(—) 를
    두부(▯)로 그린다 — 리눅스에서 실제로 `EGO▯ ego_vehicle` 이 나왔다.
    """

    def test_truetype_is_found_on_this_platform(self):
        try:
            from PIL import ImageFont
        except ImportError:
            self.skipTest("Pillow 없음")
        r = BevRenderer(None, BevConfig())
        f = r._font(12)
        self.assertIsNotNone(f)
        self.assertIsInstance(
            f, ImageFont.FreeTypeFont,
            "트루타입을 못 찾아 비트맵 폰트로 떨어졌다 — 비 ASCII 문자가 깨진다",
        )

    def test_legend_dash_is_renderable(self):
        """범례에 쓰는 문자가 폰트에 있는지 (폭이 0 이면 못 그리는 것)."""
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            self.skipTest("Pillow 없음")
        r = BevRenderer(None, BevConfig())
        dr = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        f = r._font(12)
        for ch in ("—", "±", "°", "↔"):
            self.assertGreater(dr.textlength(ch, font=f), 0.0, ch)
