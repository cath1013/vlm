"""DeepAccident 운동·사고 예측 재구현.

이 모델은 **비교 대상**이다. 원본의 설정값(격자·프레임 수·표본 수)이 조용히
어긋나면 비교가 무의미해지므로 값으로 잠근다.
"""

import math
import unittest

import numpy as np

from deepaccident_replicate.da_motion import (
    CHANNELS,
    FRAME_DT_S,
    N_CHANNELS,
    N_FUTURE,
    N_PAST,
    N_SAMPLES,
    VEHICLE_CLASSES,
    BevGrid,
    FrameActors,
    ego_frame,
    flow_label,
    frames_to_buckets,
    instance_masks,
    poly_gap,
    rasterize,
    read_tracks,
    track_gaps,
)


def actors(*specs) -> FrameActors:
    """(id, x, y, yaw_deg, speed, length, width) 들 → FrameActors."""
    ids, xy, yaw, vel, size = [], [], [], [], []
    for aid, x, y, yd, sp, l, w in specs:
        th = math.radians(yd)
        ids.append(aid)
        xy.append((x, y))
        yaw.append(th)
        vel.append((sp * math.cos(th), sp * math.sin(th)))
        size.append((l, w))
    return FrameActors(
        ids=ids,
        xy=np.asarray(xy, dtype=np.float32).reshape(-1, 2),
        yaw=np.asarray(yaw, dtype=np.float32),
        vel=np.asarray(vel, dtype=np.float32).reshape(-1, 2),
        size=np.asarray(size, dtype=np.float32).reshape(-1, 2),
    )


class TestGridMatchesTheConfig(unittest.TestCase):
    """`configs/DeepAccident_tiny.py` 에서 읽어온 값."""

    def test_motion_grid_is_100m_at_half_metre(self):
        g = BevGrid()
        self.assertEqual((g.range_m, g.res_m), (50.0, 0.5))
        self.assertEqual(g.size, 200)

    def test_three_past_four_future_at_2hz(self):
        self.assertEqual((N_PAST, N_FUTURE), (3, 4))
        self.assertEqual(FRAME_DT_S, 0.5)

    def test_six_samples(self):
        # `_base_motion_head.py:315-320` — 표본 5개 + 분포 평균
        self.assertEqual(N_SAMPLES, 6)

    def test_only_vehicles(self):
        # ConvertMotionLabels(only_vehicle=True)
        self.assertIn("car", VEHICLE_CLASSES)
        self.assertNotIn("pedestrian", VEHICLE_CLASSES)
        self.assertNotIn("cyclist", VEHICLE_CLASSES)

    def test_centre_maps_to_grid_centre(self):
        g = BevGrid()
        self.assertEqual(g.to_px(0.0, 0.0), (100.0, 100.0))

    def test_range_check(self):
        g = BevGrid()
        self.assertTrue(g.in_range(49.9, -49.9))
        self.assertFalse(g.in_range(50.1, 0.0))


class TestRasterize(unittest.TestCase):
    def test_channel_order_is_fixed(self):
        self.assertEqual(CHANNELS,
                         ("occupancy", "vx", "vy", "cos_yaw", "sin_yaw"))
        self.assertEqual(N_CHANNELS, 5)

    def test_footprint_area_is_about_right(self):
        g = BevGrid()
        r = rasterize(actors(("A", 0, 0, 0, 0, 4.6, 1.93)), g)
        px = r[0].sum()
        expect = 4.6 * 1.93 / (g.res_m ** 2)  # 약 35.5 px
        self.assertAlmostEqual(px, expect, delta=0.25 * expect)

    def test_velocity_only_inside_the_footprint(self):
        g = BevGrid()
        r = rasterize(actors(("A", 0, 0, 0, 10.0, 4.6, 1.93)), g)
        occ = r[0] > 0
        self.assertAlmostEqual(float(r[1][occ].mean()), 10.0, places=3)
        self.assertEqual(float(r[1][~occ].sum()), 0.0)

    def test_heading_channels(self):
        g = BevGrid()
        r = rasterize(actors(("A", 0, 0, 90, 1.0, 4.6, 1.93)), g)
        occ = r[0] > 0
        self.assertAlmostEqual(float(r[3][occ].mean()), 0.0, places=5)
        self.assertAlmostEqual(float(r[4][occ].mean()), 1.0, places=5)

    def test_actor_outside_the_grid_is_not_drawn(self):
        g = BevGrid()
        r = rasterize(actors(("A", 400, 0, 0, 0, 4.6, 1.93)), g)
        self.assertEqual(float(r[0].sum()), 0.0)

    def test_rotated_box_covers_similar_area(self):
        g = BevGrid()
        a = rasterize(actors(("A", 0, 0, 0, 0, 4.6, 1.93)), g)[0].sum()
        b = rasterize(actors(("A", 0, 0, 37, 0, 4.6, 1.93)), g)[0].sum()
        self.assertAlmostEqual(a, b, delta=0.25 * a)


class TestEgoFrame(unittest.TestCase):
    class _A:
        def __init__(self, aid, x, y, cls="car", kind="observed"):
            self.actor_id, self.world_xy, self.cls, self.kind = aid, (x, y), cls, kind
            self.heading_deg, self.speed_mps = 0.0, 1.0

    class _S:
        def __init__(self, actors):
            self.actors, self.t = actors, 0.0

    def test_origin_is_the_ego(self):
        s = self._S([self._A("EGO_ego_vehicle", 100, 200, kind="ego"),
                     self._A("V1", 110, 200)])
        fr = ego_frame(s, BevGrid())
        self.assertEqual(fr.ids[0], "EGO_ego_vehicle")
        np.testing.assert_allclose(fr.xy[0], [0, 0], atol=1e-5)
        np.testing.assert_allclose(fr.xy[1], [10, 0], atol=1e-5)

    def test_far_actors_are_dropped(self):
        """±50 m 밖은 BEV 격자에 없다 — 원본의 실제 한계다."""
        s = self._S([self._A("EGO_ego_vehicle", 0, 0, kind="ego"),
                     self._A("V1", 400, 0)])
        self.assertEqual(ego_frame(s, BevGrid()).ids, ["EGO_ego_vehicle"])

    def test_non_vehicles_are_dropped(self):
        s = self._S([self._A("EGO_ego_vehicle", 0, 0, kind="ego"),
                     self._A("P1", 5, 0, cls="pedestrian")])
        self.assertEqual(ego_frame(s, BevGrid()).ids, ["EGO_ego_vehicle"])

    def test_without_an_ego_there_is_no_frame(self):
        s = self._S([self._A("V1", 0, 0)])
        self.assertIsNone(ego_frame(s, BevGrid()))


class TestFlowLabel(unittest.TestCase):
    def test_displacement_is_written_on_the_actor_pixels(self):
        g = BevGrid()
        cur = actors(("A", 0, 0, 0, 0, 4.6, 1.93))
        fut = actors(("A", 6, 2, 0, 0, 4.6, 1.93))
        flow, valid = flow_label(cur, fut, g)
        m = valid > 0
        self.assertTrue(m.any())
        self.assertAlmostEqual(float(flow[0][m].mean()), 6.0, places=4)
        self.assertAlmostEqual(float(flow[1][m].mean()), 2.0, places=4)

    def test_actors_that_vanish_are_not_supervised(self):
        """미래에 사라진 차를 0 으로 맞히게 하면 '아무도 안 움직인다' 로 수렴한다."""
        g = BevGrid()
        cur = actors(("A", 0, 0, 0, 0, 4.6, 1.93), ("B", 20, 0, 0, 0, 4.6, 1.93))
        fut = actors(("A", 3, 0, 0, 0, 4.6, 1.93))
        flow, valid = flow_label(cur, fut, g)
        masks = instance_masks(cur, g)
        self.assertEqual(float(valid[masks["B"]].sum()), 0.0)
        self.assertTrue(float(valid[masks["A"]].mean()) > 0.9)


class TestPolyGap(unittest.TestCase):
    @staticmethod
    def _rect(x, y, l=4.0, w=2.0):
        return np.array([[x + l / 2, y + w / 2], [x + l / 2, y - w / 2],
                         [x - l / 2, y - w / 2], [x - l / 2, y + w / 2]],
                        dtype=np.float32)

    def test_overlap_is_zero(self):
        self.assertEqual(poly_gap(self._rect(0, 0), self._rect(1, 0)), 0.0)

    def test_clearance_along_the_axis(self):
        self.assertAlmostEqual(poly_gap(self._rect(0, 0), self._rect(10, 0)), 6.0, 4)

    def test_lateral_clearance(self):
        self.assertAlmostEqual(poly_gap(self._rect(0, 0), self._rect(0, 5)), 3.0, 4)

    def test_touching_is_zero(self):
        self.assertAlmostEqual(poly_gap(self._rect(0, 0), self._rect(4, 0)), 0.0, 4)


class TestReadTracks(unittest.TestCase):
    def test_flow_is_read_as_actor_displacement(self):
        g = BevGrid()
        fr = actors(("A", 0, 0, 0, 0, 4.6, 1.93))
        flow = np.zeros((N_FUTURE, 2, g.size, g.size), dtype=np.float32)
        for k in range(N_FUTURE):
            flow[k, 0] = 2.0 * (k + 1)
        tr = read_tracks(flow, fr, g)
        self.assertEqual(tr.shape, (N_FUTURE, 1, 2))
        np.testing.assert_allclose(tr[:, 0, 0], [2, 4, 6, 8], atol=1e-4)

    def test_each_actor_reads_its_own_pixels(self):
        g = BevGrid()
        fr = actors(("A", -10, 0, 0, 0, 4.6, 1.93), ("B", 10, 0, 0, 0, 4.6, 1.93))
        flow = np.zeros((1, 2, g.size, g.size), dtype=np.float32)
        masks = instance_masks(fr, g)
        flow[0, 0][masks["A"]] = 5.0
        flow[0, 0][masks["B"]] = -5.0
        tr = read_tracks(flow, fr, g)
        self.assertAlmostEqual(float(tr[0, 0, 0]), -5.0, places=3)
        self.assertAlmostEqual(float(tr[0, 1, 0]), 5.0, places=3)


class TestTrackGaps(unittest.TestCase):
    def test_closing_pair_is_found(self):
        fr = actors(("A", 0, 0, 0, 8, 4.0, 2.0), ("B", 30, 0, 180, 8, 4.0, 2.0))
        tracks = np.array([[[10, 0], [20, 0]], [[14, 0], [16, 0]]],
                          dtype=np.float32)
        gaps = track_gaps(tracks, fr)
        self.assertAlmostEqual(gaps[0][0], 6.0, places=3)
        self.assertAlmostEqual(gaps[1][0], 0.0, places=3)
        self.assertEqual(set(gaps[1][1]), {"A", "B"})

    def test_pairs_already_touching_are_excluded(self):
        """DeepAccident 라벨의 주차 차량 박스는 서로 겹쳐 있다 — 빼지 않으면
        최소 간격이 영구히 0 이라 어떤 임계도 무의미해진다."""
        # 전부 움직이게 둬서 **접촉 필터만** 시험한다 (정지쌍 필터는 따로 시험)
        fr = actors(("A", 0, 0, 0, 8, 5.0, 2.0), ("B", 3, 0, 0, 8, 5.0, 2.0),
                    ("C", 40, 0, 0, 8, 4.0, 2.0))
        tracks = np.array([[[0, 0], [3, 0], [40, 0]]], dtype=np.float32)
        # B 앞끝 x=5.5, C 뒤끝 x=38 → 32.5
        self.assertAlmostEqual(
            track_gaps(tracks, fr, exclude_touching_now=True,
                       exclude_static_pairs=False)[0][0], 32.5, places=3)
        self.assertEqual(
            track_gaps(tracks, fr, exclude_touching_now=False,
                       exclude_static_pairs=False)[0][0], 0.0)

    def test_two_parked_cars_are_not_an_accident(self):
        """겹치는 쌍의 98 %가 정지-정지다 — 주차된 차 두 대는 사고를 낼 수 없다."""
        parked = actors(("A", 0, 0, 0, 0.0, 5.0, 2.0), ("B", 3, 0, 0, 0.0, 5.0, 2.0))
        tracks = np.array([[[0, 0], [3, 0]]], dtype=np.float32)
        self.assertEqual(track_gaps(tracks, parked,
                                    exclude_static_pairs=True)[0][0], float("inf"))
        # 두 필터를 다 끄면 이 쌍이 최소 간격 0 으로 남는다 — 그것이 문제였다
        self.assertEqual(track_gaps(tracks, parked, exclude_touching_now=False,
                                    exclude_static_pairs=False)[0][0], 0.0)

    def test_one_moving_into_a_stopped_car_still_counts(self):
        """신호 대기 중 추돌당하는 것은 사고다 — 한쪽만 정지한 쌍은 남긴다."""
        fr = actors(("A", 0, 0, 0, 0.0, 4.0, 2.0), ("B", 30, 0, 180, 10.0, 4.0, 2.0))
        tracks = np.array([[[0, 0], [4, 0]]], dtype=np.float32)
        self.assertAlmostEqual(
            track_gaps(tracks, fr, exclude_static_pairs=True)[0][0], 0.0, places=3)

    def test_single_actor_has_no_pairs(self):
        fr = actors(("A", 0, 0, 0, 0, 4.0, 2.0))
        gaps = track_gaps(np.zeros((2, 1, 2), dtype=np.float32), fr)
        self.assertEqual(gaps, [(float("inf"), None)] * 2)


class TestFramesToBuckets(unittest.TestCase):
    def test_frames_fold_into_one_second_buckets(self):
        """2 Hz 미래 4프레임 → 구간 1 은 0.5·1.0초, 구간 2 는 1.5·2.0초."""
        s = [[(9.0, None), (1.0, ("A", "B")), (9.0, None), (9.0, None)]]
        out = frames_to_buckets(s, 5)
        self.assertAlmostEqual(out[0][0], 1.0)
        self.assertEqual(out[0][1], ("A", "B"))

    def test_beyond_the_prediction_horizon_is_infinite(self):
        """2초까지만 예측하므로 3·4·5 구간은 보지 않는다."""
        s = [[(1.0, None)] * 4]
        out = frames_to_buckets(s, 5)
        self.assertEqual([g for g, _ in out[2:]], [float("inf")] * 3)

    def test_samples_fold_by_minimum(self):
        """원본은 표본 아무거나 사고를 가리키면 사고로 본다."""
        s = [[(9.0, None)] * 4, [(0.2, ("A", "B"))] + [(9.0, None)] * 3]
        out = frames_to_buckets(s, 2)
        self.assertAlmostEqual(out[0][0], 0.2)


class TestNetShapes(unittest.TestCase):
    def test_outputs_are_full_resolution(self):
        """1/8 해상도(4 m/셀)에서 읽으면 차 한 대가 한 셀보다 작아 접촉 판정이 안 된다."""
        import torch

        from deepaccident_replicate.da_motion import build_net

        g = BevGrid()
        net = build_net(width=16)
        x = torch.zeros(1, N_PAST, N_CHANNELS, g.size, g.size)
        with torch.no_grad():
            flows, occs, accs, mu, logvar = net(x, n_samples=2)
        self.assertEqual(tuple(flows.shape), (1, 2, N_FUTURE, 2, g.size, g.size))
        self.assertEqual(tuple(occs.shape), (1, 2, N_FUTURE, 1, g.size, g.size))
        self.assertEqual(tuple(accs.shape), (1, 2, N_FUTURE, 1, g.size, g.size))
        self.assertEqual(mu.shape, logvar.shape)

    def test_last_sample_is_the_distribution_mean(self):
        """원본이 `samples[-1] = mu` 로 두고 운동 지표는 평균만 쓴다."""
        import torch

        from deepaccident_replicate.da_motion import build_net

        g = BevGrid()
        net = build_net(width=16)
        x = torch.zeros(1, N_PAST, N_CHANNELS, g.size, g.size)
        with torch.no_grad():
            feat = net.encode(x)
            zs, mu, _ = net.sample_latents(feat, 4)
        torch.testing.assert_close(zs[:, -1], mu)
        self.assertFalse(torch.equal(zs[:, 0], mu))


if __name__ == "__main__":
    unittest.main()


class TestHeadingConvention(unittest.TestCase):
    """`heading_deg` 는 진북 기준 시계방향 방위각이다 (`geometry.heading_to_unit`).

    `_rect_corners` 는 수학 각도를 쓰므로 `ego_frame` 이 θ = π/2 − h 로 바꿔야 한다.
    빠뜨리면 모든 차체가 90° 어긋나 접촉 판정이 통째로 틀린다.
    """

    class _A:
        def __init__(self, aid, x, y, heading, kind="observed"):
            self.actor_id, self.world_xy, self.cls, self.kind = aid, (x, y), "car", kind
            self.heading_deg, self.speed_mps = heading, 10.0

    class _S:
        def __init__(self, actors):
            self.actors, self.t = actors, 0.0

    def _fr(self, heading):
        return ego_frame(self._S([self._A("EGO_ego_vehicle", 0, 0, heading, "ego")]),
                         BevGrid())

    def test_north_heading_points_along_plus_y(self):
        fr = self._fr(0.0)  # 진북
        self.assertAlmostEqual(float(fr.vel[0, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(fr.vel[0, 1]), 10.0, places=4)

    def test_east_heading_points_along_plus_x(self):
        fr = self._fr(90.0)  # 진동
        self.assertAlmostEqual(float(fr.vel[0, 0]), 10.0, places=4)
        self.assertAlmostEqual(float(fr.vel[0, 1]), 0.0, places=4)

    def test_footprint_is_long_along_the_heading(self):
        """진북을 보는 차는 BEV 에서 세로로 길어야 한다."""
        g = BevGrid()
        occ = rasterize(self._fr(0.0), g)[0]
        ys, xs = np.nonzero(occ)
        self.assertGreater(np.ptp(ys), np.ptp(xs))
        occ_e = rasterize(self._fr(90.0), g)[0]
        ys, xs = np.nonzero(occ_e)
        self.assertGreater(np.ptp(xs), np.ptp(ys))


class TestCollisionHead(unittest.TestCase):
    """사고 헤드 — 모든 쌍의 최소 간격 대신 **모델이 지목한 자리**를 쓴다.

    후처리 규칙은 이 데이터에서 어떤 임계값에서도 판정을 못 했다 (val 816창 스윕에서
    균형정확도 최대 0.541, 우연 0.500). 원인은 모델이 아니라 양(quantity)이다 —
    기록된 실제 미래에서도 표본의 89 %가 어떤 쌍이든 2.5 m 안에 든다.
    """

    def test_heatmap_peaks_at_the_collision_point(self):
        from deepaccident_replicate.da_motion import collision_heatmap_label

        g = BevGrid()
        h = collision_heatmap_label(g, (10.0, -6.0))
        row, col = divmod(int(h.argmax()), g.size)
        self.assertAlmostEqual((col + 0.5) * g.res_m - g.range_m, 10.0, delta=0.5)
        self.assertAlmostEqual((row + 0.5) * g.res_m - g.range_m, -6.0, delta=0.5)
        # 픽셀 중심이 정확히 사고 지점에 오지는 않으므로 최댓값은 1 에 근접한다
        self.assertGreater(float(h.max()), 0.98)

    def test_no_collision_gives_an_empty_map(self):
        from deepaccident_replicate.da_motion import collision_heatmap_label

        self.assertEqual(float(collision_heatmap_label(BevGrid(), None).sum()), 0.0)

    def test_a_point_outside_the_grid_is_dropped(self):
        from deepaccident_replicate.da_motion import collision_heatmap_label

        self.assertEqual(
            float(collision_heatmap_label(BevGrid(), (400.0, 0.0)).sum()), 0.0)

    def test_readout_finds_the_strongest_frame_and_place(self):
        from deepaccident_replicate.da_motion import read_accident

        g = BevGrid()
        logits = np.full((2, N_FUTURE, g.size, g.size), -9.0, dtype=np.float32)
        cx, cy = g.to_px(12.0, -4.0)
        logits[0, 2, int(cy), int(cx)] = 5.0
        v = read_accident(logits, g, threshold=0.5)
        self.assertTrue(v["accident"])
        self.assertEqual(v["frame"], 2)
        self.assertAlmostEqual(v["xy"][0], 12.0, delta=0.6)
        self.assertAlmostEqual(v["xy"][1], -4.0, delta=0.6)

    def test_samples_fold_by_maximum(self):
        """원본은 표본 아무거나 사고를 가리키면 사고로 본다."""
        from deepaccident_replicate.da_motion import read_accident

        g = BevGrid()
        logits = np.full((3, N_FUTURE, g.size, g.size), -9.0, dtype=np.float32)
        logits[2, 1, 100, 100] = 4.0  # 세 표본 중 하나만 사고를 가리킨다
        self.assertTrue(read_accident(logits, g, 0.5)["accident"])

    def test_quiet_map_is_no_accident(self):
        from deepaccident_replicate.da_motion import read_accident

        g = BevGrid()
        logits = np.full((2, N_FUTURE, g.size, g.size), -9.0, dtype=np.float32)
        self.assertFalse(read_accident(logits, g, 0.5)["accident"])

    def test_response_marks_the_bucket_of_the_named_frame(self):
        """미래 프레임 3(= 2.0초)은 1초 구간 2 에 든다."""
        from deepaccident_replicate.da_motion import head_response

        g = BevGrid()
        logits = np.full((1, N_FUTURE, g.size, g.size), -9.0, dtype=np.float32)
        logits[0, 3, 100, 100] = 5.0
        r = head_response(5.0, 5, logits, g)
        hits = [p["k"] for p in r["predictions"] if p["accident_expected"]]
        self.assertEqual(hits, [2])

    def test_response_names_the_two_nearest_vehicles(self):
        from deepaccident_replicate.da_motion import head_response

        g = BevGrid()
        fr = actors(("A", 12, 0, 0, 8, 4.6, 1.93), ("B", 14, 0, 0, 8, 4.6, 1.93),
                    ("C", -40, 0, 0, 8, 4.6, 1.93))
        logits = np.full((1, N_FUTURE, g.size, g.size), -9.0, dtype=np.float32)
        cx, cy = g.to_px(13.0, 0.0)
        logits[0, 1, int(cy), int(cx)] = 5.0
        r = head_response(5.0, 5, logits, g, fr)
        hit = [p for p in r["predictions"] if p["accident_expected"]][0]
        self.assertEqual(set(hit["involved_actor_ids"]), {"A", "B"})

    def test_beyond_the_prediction_horizon_is_marked_as_unseen(self):
        from deepaccident_replicate.da_motion import head_response

        g = BevGrid()
        logits = np.full((1, N_FUTURE, g.size, g.size), -9.0, dtype=np.float32)
        r = head_response(5.0, 5, logits, g)
        beyond = [p for p in r["predictions"] if p["k"] > 2]
        self.assertTrue(all("지평" in p["reason"] for p in beyond))

    def test_response_flows_through_score_modes(self):
        from traffic_llm.accident_qa import score_modes

        from deepaccident_replicate.da_motion import head_response

        g = BevGrid()
        logits = np.full((1, N_FUTURE, g.size, g.size), -9.0, dtype=np.float32)
        logits[0, 3, 100, 100] = 5.0   # 미래 프레임 3 = 2.0초 → 구간 2
        from traffic_llm.accident_qa import (WindowConfig, build_windows,
                                             window_ground_truth)
        from traffic_llm.schemas import CollisionTruth
        from tests.test_accident_qa import make_snapshots, pick

        cfg = WindowConfig(window_s=5.0, horizon_s=5.0)
        wins, _ = build_windows(make_snapshots(n=21, dt=0.5), cfg)
        w = pick(wins, "0-5")  # t_end = 5.0
        gt = window_ground_truth(  # 충돌 6.2초 → k_true = 2
            w, cfg, CollisionTruth(occurred=True, carla_ids=(7001,),
                                   time_s=6.2, frame=0),
            {"v1": 1}, "s", "sp", "T")
        m = score_modes(head_response(5.0, 5, logits, g), gt)
        self.assertEqual(m["binary"]["status"], "TP")
        self.assertEqual(m["strict"]["counts"]["TP"], 1)
