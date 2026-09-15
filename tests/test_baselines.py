"""무모델 기준선(`examples/baselines_accident_qa.py`)의 기하 검증.

`gap` 정책의 값은 **회전 사각형 사이의 실제 간격**에서 나온다. 그 계산이 틀리면
기준선 숫자 전체가 무효가 되므로, 손으로 답을 알 수 있는 배치들로 고정한다.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "examples"))

from baselines_accident_qa import (  # noqa: E402
    _corners,
    _motion_fn,
    _poly_gap,
    _seg_point_dist,
    _separated,
    _velocity,
    bucket_min_gaps,
)
from traffic_llm.schemas import (  # noqa: E402
    ActorState,
    PredictedPath,
    SceneSnapshot,
)

E = (1.0, 0.0)  # 동쪽을 보는 단위벡터


def car(cx, cy, unit=E, length=4.6, width=1.9):
    return _corners(cx, cy, unit[0], unit[1], length / 2.0, width / 2.0)


class TestCorners(unittest.TestCase):
    def test_axis_aligned_extent(self):
        """동쪽을 보는 4.6×1.9 차량의 꼭짓점은 ±2.3, ±0.95."""
        c = car(0.0, 0.0)
        xs = sorted({round(p[0], 6) for p in c})
        ys = sorted({round(p[1], 6) for p in c})
        self.assertEqual(xs, [-2.3, 2.3])
        self.assertEqual(ys, [-0.95, 0.95])

    def test_rotated_90_swaps_extent(self):
        """북쪽을 보면 길이축이 n 방향으로 간다."""
        c = _corners(0.0, 0.0, 0.0, 1.0, 2.3, 0.95)
        xs = sorted({round(p[0], 6) for p in c})
        ys = sorted({round(p[1], 6) for p in c})
        self.assertEqual(xs, [-0.95, 0.95])
        self.assertEqual(ys, [-2.3, 2.3])

    def test_area_preserved_under_rotation(self):
        """어떤 각도로 돌려도 넓이는 길이×폭이다 (신발끈 공식)."""
        for deg in (0, 17, 45, 90, 143, 271):
            r = math.radians(deg)
            c = _corners(5.0, -3.0, math.cos(r), math.sin(r), 2.3, 0.95)
            a = 0.0
            for i in range(4):
                x1, y1 = c[i]
                x2, y2 = c[(i + 1) % 4]
                a += x1 * y2 - x2 * y1
            self.assertAlmostEqual(abs(a) / 2.0, 4.6 * 1.9, places=6)


class TestSegPointDist(unittest.TestCase):
    def test_perpendicular_foot_inside(self):
        self.assertAlmostEqual(_seg_point_dist(0, 3, -5, 0, 5, 0), 3.0)

    def test_clamps_to_endpoint(self):
        """수선의 발이 선분 밖이면 끝점까지의 거리."""
        self.assertAlmostEqual(_seg_point_dist(10, 0, -5, 0, 5, 0), 5.0)
        self.assertAlmostEqual(_seg_point_dist(8, 3, -5, 0, 5, 0), math.sqrt(18.0))

    def test_degenerate_segment(self):
        self.assertAlmostEqual(_seg_point_dist(3, 4, 0, 0, 0, 0), 5.0)


class TestSeparated(unittest.TestCase):
    def test_far_apart(self):
        self.assertTrue(_separated(car(0, 0), car(50, 0)))

    def test_overlapping(self):
        self.assertFalse(_separated(car(0, 0), car(1.0, 0)))

    def test_touching_is_not_separated(self):
        """정확히 맞닿은 두 차량 — 분리되지 않은 것으로 본다."""
        self.assertFalse(_separated(car(0, 0), car(4.6, 0)))

    def test_rotation_only_overlap(self):
        """축정렬 경계상자로는 겹치지만 실제로는 떨어진 배치.

        45° 로 돌린 두 차량을 대각으로 놓으면 AABB 는 겹치는데 차체는 떨어진다.
        분리축 정리를 제대로 쓰는지 확인하는 사례다.
        """
        u = (math.cos(math.radians(45)), math.sin(math.radians(45)))
        a = _corners(0.0, 0.0, u[0], u[1], 2.3, 0.95)
        b = _corners(0.0, 4.0, u[0], u[1], 2.3, 0.95)
        self.assertTrue(_separated(a, b))


class TestPolyGap(unittest.TestCase):
    def test_bumper_to_bumper(self):
        """같은 방향 두 차량, 중심거리 10m → 간격 10 − 4.6 = 5.4m."""
        self.assertAlmostEqual(_poly_gap(car(0, 0), car(10, 0)), 5.4, places=6)

    def test_side_by_side(self):
        """나란한 두 차량, 횡거리 3m → 간격 3 − 1.9 = 1.1m."""
        self.assertAlmostEqual(_poly_gap(car(0, 0), car(0, 3.0)), 1.1, places=6)

    def test_overlap_is_zero(self):
        self.assertEqual(_poly_gap(car(0, 0), car(2.0, 0)), 0.0)

    def test_touching_is_zero(self):
        self.assertEqual(_poly_gap(car(0, 0), car(4.6, 0)), 0.0)

    def test_symmetric(self):
        a, b = car(0, 0), car(7.3, 2.1)
        self.assertAlmostEqual(_poly_gap(a, b), _poly_gap(b, a), places=9)

    def test_perpendicular_t_bone(self):
        """직교 배치 — 동쪽 차 앞범퍼(x=2.3)와 북쪽 차 좌측면(x=10−0.95).

        간격 = (10 − 0.95) − 2.3 = 6.75m
        """
        a = car(0, 0)                              # 동쪽
        b = _corners(10.0, 0.0, 0.0, 1.0, 2.3, 0.95)  # 북쪽
        self.assertAlmostEqual(_poly_gap(a, b), 6.75, places=6)

    def test_never_negative(self):
        for dx in (0.0, 1.0, 4.6, 5.0, 20.0):
            self.assertGreaterEqual(_poly_gap(car(0, 0), car(dx, 0)), 0.0)

    def test_larger_vehicle_closes_gap(self):
        """트럭(8.47m)은 승용차보다 같은 중심거리에서 간격이 좁다."""
        d_car = _poly_gap(car(0, 0), car(12, 0))
        d_truck = _poly_gap(car(0, 0, length=8.47, width=2.89), car(12, 0))
        self.assertLess(d_truck, d_car)


class TestVelocity(unittest.TestCase):
    def _actor(self, **kw):
        base = dict(actor_id="V1", kind="observed", cls="car",
                    world_xy=(0.0, 0.0), heading_deg=None, speed_mps=None,
                    accel_mps2=None)
        base.update(kw)
        return ActorState(**base)

    def test_prefers_track_history_difference(self):
        """위치 차분이 1순위 — 라벨 속도 컬럼을 믿지 않는다는 규약."""
        a = self._actor(track_history=[(0.0, 0.0, 0.0), (0.5, 5.0, 0.0)],
                        heading_deg=0.0, speed_mps=99.0)
        vx, vy = _velocity(a)
        self.assertAlmostEqual(vx, 10.0)   # 5m / 0.5s — heading/speed 가 아니라
        self.assertAlmostEqual(vy, 0.0)

    def test_falls_back_to_heading_and_speed(self):
        """이력이 없으면 방위각+속력. 방위는 진북 기준 시계방향 → (sin, cos)."""
        a = self._actor(heading_deg=90.0, speed_mps=10.0)  # 동쪽
        vx, vy = _velocity(a)
        self.assertAlmostEqual(vx, 10.0, places=6)
        self.assertAlmostEqual(vy, 0.0, places=6)

    def test_zero_when_nothing_known(self):
        self.assertEqual(_velocity(self._actor()), (0.0, 0.0))

    def test_ignores_zero_duration_history(self):
        a = self._actor(track_history=[(1.0, 0.0, 0.0), (1.0, 5.0, 0.0)])
        self.assertEqual(_velocity(a), (0.0, 0.0))


class TestBucketMinGaps(unittest.TestCase):
    def _snap(self, actors, t=0.0):
        return SceneSnapshot(t=t, actors=actors, interactions=[])

    def _mover(self, aid, xy, hist):
        return ActorState(actor_id=aid, kind="observed", cls="car", world_xy=xy,
                          heading_deg=None, speed_mps=None, accel_mps2=None,
                          track_history=hist)

    def test_head_on_closes_then_passes_through(self):
        """마주 보고 접근하는 두 차.

        접촉 전까지는 간격이 줄지만, **등속 외삽은 두 차가 서로를 통과하게
        둔다** — 실제로는 부딪혀 멈출 상황이다. 그래서 접촉 이후 구간의
        간격은 의미가 없다. 첫 접촉 구간까지만 읽어야 한다.

        정답도 충돌 시점에서 녹화가 끊기고 그 뒤 구간은 채점에서 빠지므로
        실사용에서는 문제가 되지 않지만, 이 성질을 모르면 값을 오독한다.
        """
        a = self._mover("A", (0.0, 0.0), [(-1.0, -10.0, 0.0), (0.0, 0.0, 0.0)])
        b = self._mover("B", (40.0, 0.0), [(-1.0, 50.0, 0.0), (0.0, 40.0, 0.0)])
        gaps = bucket_min_gaps(self._snap([a, b]), 4)
        vals = [g for g, _ in gaps]
        # 접근 20m/s, 중심거리 40m, 차체 4.6m → t=1 에 중심거리 20m
        self.assertAlmostEqual(vals[0], 20.0 - 4.6, places=6)
        self.assertEqual(vals[1], 0.0)          # t≈1.77 에 접촉
        self.assertEqual(vals[2], 0.0)          # t=2 에 겹침 — 통과 중
        self.assertGreater(vals[3], 0.0)        # 통과 후 다시 벌어진다
        self.assertTrue(all(p == ("A", "B") for _, p in gaps))

    def test_gap_decreases_while_approaching(self):
        """접촉 전 구간에서는 단조 감소한다."""
        a = self._mover("A", (0.0, 0.0), [(-1.0, -5.0, 0.0), (0.0, 0.0, 0.0)])
        b = self._mover("B", (80.0, 0.0), [(-1.0, 85.0, 0.0), (0.0, 80.0, 0.0)])
        vals = [g for g, _ in bucket_min_gaps(self._snap([a, b]), 4)]
        self.assertEqual(vals, sorted(vals, reverse=True))
        self.assertTrue(all(v > 0.0 for v in vals))

    def test_contact_reaches_zero(self):
        """접근 속도 20m/s, 중심거리 40m → 2초 안에 닿는다."""
        a = self._mover("A", (0.0, 0.0), [(-1.0, -10.0, 0.0), (0.0, 0.0, 0.0)])
        b = self._mover("B", (40.0, 0.0), [(-1.0, 50.0, 0.0), (0.0, 40.0, 0.0)])
        gaps = bucket_min_gaps(self._snap([a, b]), 3)
        self.assertGreater(gaps[0][0], 0.0)
        self.assertEqual(gaps[2][0], 0.0)

    def test_parallel_never_touches(self):
        """같은 속도로 나란히 달리는 두 차 — 간격이 일정하게 유지된다."""
        a = self._mover("A", (0.0, 0.0), [(-1.0, -10.0, 0.0), (0.0, 0.0, 0.0)])
        b = self._mover("B", (0.0, 5.0), [(-1.0, -10.0, 5.0), (0.0, 0.0, 5.0)])
        gaps = bucket_min_gaps(self._snap([a, b]), 3)
        for g, _ in gaps:
            # CLASS_SIZES['car'].width_m == 1.93 (절반씩 0.965)
            self.assertAlmostEqual(g, 5.0 - 1.93, places=6)

    def test_single_actor_has_no_pair(self):
        a = self._mover("A", (0.0, 0.0), [(0.0, 0.0, 0.0)])
        gaps = bucket_min_gaps(self._snap([a]), 3)
        self.assertEqual([p for _, p in gaps], [None, None, None])
        self.assertTrue(all(math.isinf(g) for g, _ in gaps))

    def test_reports_the_closest_pair_not_just_any(self):
        """세 대 중 실제로 가장 가까워지는 쌍을 지목해야 한다."""
        a = self._mover("A", (0.0, 0.0), [(-1.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
        b = self._mover("B", (0.0, 60.0), [(-1.0, 0.0, 60.0), (0.0, 0.0, 60.0)])
        c = self._mover("C", (0.0, 63.0), [(-1.0, 0.0, 63.0), (0.0, 0.0, 63.0)])
        gaps = bucket_min_gaps(self._snap([a, b, c]), 2)
        for _, pair in gaps:
            self.assertEqual(set(pair), {"B", "C"})


if __name__ == "__main__":
    unittest.main()


class TestPredictedMotion(unittest.TestCase):
    """`--extrapolation predicted` — 예측 경로를 따라가는 gap table.

    웨이포인트 정렬이 핵심이다. WaypointNet 은 **첫 점이 현재 위치**이고 이후
    1초 간격인데, 앞에 현재 위치를 무조건 덧붙이면 전체가 1초씩 밀린다.
    """

    def _actor(self, aid, xy, wps=None, hist=None, prob=1.0):
        preds = []
        if wps is not None:
            preds = [PredictedPath(maneuver="직진", probability=prob,
                                   waypoints=list(wps), horizon_s=float(len(wps) - 1))]
        return ActorState(actor_id=aid, kind="observed", cls="car", world_xy=xy,
                          heading_deg=None, speed_mps=None, accel_mps2=None,
                          track_history=hist or [], predictions=preds)

    def test_waypoints_starting_at_current_position(self):
        """첫 점 = 현재 위치. t초에는 wps[t] 에 있어야 한다."""
        wps = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
        f = _motion_fn(self._actor("A", (0.0, 0.0), wps), "predicted")
        for t, want in ((0.0, 0.0), (1.0, 10.0), (2.0, 20.0), (2.5, 25.0)):
            self.assertAlmostEqual(f(t)[0], want, places=6, msg=f"t={t}")

    def test_waypoints_starting_at_t1(self):
        """첫 점이 t=1 인 구현이면 현재 위치를 앞에 붙여 맞춘다."""
        wps = [(10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
        f = _motion_fn(self._actor("A", (0.0, 0.0), wps), "predicted")
        for t, want in ((0.0, 0.0), (1.0, 10.0), (2.0, 20.0)):
            self.assertAlmostEqual(f(t)[0], want, places=6, msg=f"t={t}")

    def test_turning_path_changes_heading(self):
        """경로가 꺾이면 진행방향 단위벡터도 따라 꺾인다."""
        wps = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        f = _motion_fn(self._actor("A", (0.0, 0.0), wps), "predicted")
        self.assertAlmostEqual(f(0.5)[2][0], 1.0, places=6)   # 동쪽
        self.assertAlmostEqual(f(1.5)[2][1], 1.0, places=6)   # 북쪽

    def test_falls_back_to_cv_without_predictions(self):
        a = self._actor("A", (0.0, 0.0), None,
                        hist=[(-1.0, -10.0, 0.0), (0.0, 0.0, 0.0)])
        f = _motion_fn(a, "predicted")
        self.assertAlmostEqual(f(2.0)[0], 20.0, places=6)

    def test_cv_mode_ignores_predictions(self):
        """mode='cv' 는 예측이 있어도 등속으로 간다 — 두 조건을 섞으면 안 된다."""
        wps = [(0.0, 0.0), (0.0, 99.0)]     # 예측은 북쪽
        a = self._actor("A", (0.0, 0.0), wps,
                        hist=[(-1.0, -10.0, 0.0), (0.0, 0.0, 0.0)])  # 관측은 동쪽
        f = _motion_fn(a, "cv")
        self.assertAlmostEqual(f(1.0)[0], 10.0, places=6)
        self.assertAlmostEqual(f(1.0)[1], 0.0, places=6)

    def test_picks_highest_probability_path(self):
        a = self._actor("A", (0.0, 0.0), [(0.0, 0.0), (1.0, 0.0)], prob=0.3)
        a.predictions.append(PredictedPath(maneuver="좌회전", probability=0.7,
                                           waypoints=[(0.0, 0.0), (0.0, 5.0)],
                                           horizon_s=1.0))
        f = _motion_fn(a, "predicted")
        self.assertAlmostEqual(f(1.0)[1], 5.0, places=6)   # 확률 0.7 쪽

    def test_extends_past_horizon(self):
        """예측 구간을 넘어서면 마지막 방향으로 등속 연장한다."""
        wps = [(0.0, 0.0), (10.0, 0.0)]
        f = _motion_fn(self._actor("A", (0.0, 0.0), wps), "predicted")
        self.assertAlmostEqual(f(3.0)[0], 30.0, places=6)

    def test_gap_table_uses_predicted_path(self):
        """예측이 서로 멀어지는 방향이면 등속보다 간격이 넓게 나온다."""
        hist_a = [(-1.0, -10.0, 0.0), (0.0, 0.0, 0.0)]
        hist_b = [(-1.0, 50.0, 0.0), (0.0, 40.0, 0.0)]
        # 등속이면 정면 충돌. 예측은 둘 다 옆으로 빠진다.
        a = self._actor("A", (0.0, 0.0), [(0.0, 0.0), (0.0, 20.0), (0.0, 40.0)], hist=hist_a)
        b = self._actor("B", (40.0, 0.0), [(40.0, 0.0), (40.0, -20.0), (40.0, -40.0)], hist=hist_b)
        snap = SceneSnapshot(t=0.0, actors=[a, b], interactions=[])
        cv = bucket_min_gaps(snap, 2, "cv")
        pr = bucket_min_gaps(snap, 2, "predicted")
        self.assertEqual(cv[1][0], 0.0)          # 등속: 접촉
        self.assertGreater(pr[1][0], 10.0)       # 예측: 멀찍이 갈라짐

    def test_single_waypoint_falls_back_to_cv(self):
        """점이 1개뿐인 예측(도로망 끝에서 잘림/정지)은 궤적 정보가 없다.

        실측으로 액터-프레임의 약 24% 가 여기 해당하고, 가드가 없으면
        `pts[-2]` 에서 IndexError 로 시나리오 전체가 실패한다.
        """
        a = self._actor("A", (0.0, 0.0), [(0.0, 0.0)],
                        hist=[(-1.0, -10.0, 0.0), (0.0, 0.0, 0.0)])
        f = _motion_fn(a, "predicted")
        self.assertAlmostEqual(f(2.0)[0], 20.0, places=6)   # 등속으로 폴백

    def test_empty_waypoints_falls_back_to_cv(self):
        a = self._actor("A", (0.0, 0.0), [],
                        hist=[(-1.0, -5.0, 0.0), (0.0, 0.0, 0.0)])
        f = _motion_fn(a, "predicted")
        self.assertAlmostEqual(f(2.0)[0], 10.0, places=6)


class TestExcludeTouchingPairs(unittest.TestCase):
    """관측 시점에 이미 붙어 있는 쌍은 후보가 아니다.

    DeepAccident 라벨에는 주차·정차 차량의 박스가 서로 겹쳐 있다 (중심거리 3~4 m 인데
    차 길이가 4.5~5.4 m). 그 쌍을 두면 최소 간격이 영구히 0 이라 거리 임계가 판정을
    못 한다 — val 실측으로 5초 지평에서 창의 93~98 %, **기록된 실제 미래에서도 86 %**
    가 그랬다. 예측 탓이 아니라 라벨의 성질이다.
    """

    @staticmethod
    def _snap(*specs):
        from traffic_llm.schemas import ActorState, SceneSnapshot

        acts = [
            ActorState(actor_id=aid, kind="observed", cls="car",
                       world_xy=(x, y), heading_deg=h, speed_mps=sp,
                       accel_mps2=0.0)
            for aid, x, y, h, sp in specs
        ]
        return SceneSnapshot(t=0.0, actors=acts, interactions=[])

    def test_overlapping_parked_pair_is_dropped(self):
        # A·B 는 중심거리 3 m 인 정차 차량 (차 길이 4.6 m) → 이미 관통 상태
        # heading 0° 는 진북(+y) — 줄 서 있는 차는 y 축을 따라 놓인다
        snap = self._snap(("A", 0.0, 0.0, 0.0, 0.0), ("B", 0.0, 3.0, 0.0, 0.0),
                          ("C", 0.0, 60.0, 180.0, 0.0))
        keep = bucket_min_gaps(snap, 3, "cv", exclude_touching_now=False)
        drop = bucket_min_gaps(snap, 3, "cv", exclude_touching_now=True)
        self.assertEqual(keep[0][0], 0.0)
        self.assertGreater(drop[0][0], 10.0)

    def test_a_real_closing_pair_survives_the_filter(self):
        """정면으로 접근하는 쌍은 지금 떨어져 있으므로 걸러지지 않는다."""
        snap = self._snap(("A", 0.0, 0.0, 0.0, 10.0), ("B", 0.0, 40.0, 180.0, 10.0))
        g = bucket_min_gaps(snap, 3, "cv", exclude_touching_now=True)
        self.assertEqual(g[1][0], 0.0)
        self.assertEqual(set(g[1][1]), {"A", "B"})

    def test_default_keeps_the_old_behaviour(self):
        snap = self._snap(("A", 0.0, 0.0, 0.0, 0.0), ("B", 0.0, 3.0, 0.0, 0.0))
        self.assertEqual(bucket_min_gaps(snap, 2, "cv")[0][0], 0.0)


class TestExcludeStaticPairs(unittest.TestCase):
    """둘 다 정지한 쌍은 사고 후보가 아니다.

    val 라벨 400프레임 실측: 겹치는 쌍 537개 중 **526개가 정지-정지**(주차 차량),
    정지-이동 0개, 이동-이동 11개(실제 충돌). 이 필터 하나로 겹치는 프레임이
    31 % → 2.8 % 로 떨어진다. 치수를 객체별 라벨 값으로 바꿔도 겹침은 거의 그대로다
    (0.310 → 0.305) — 문제는 치수 출처가 아니라 정지 차량 쌍이다.
    """

    @staticmethod
    def _snap(*specs):
        from traffic_llm.schemas import ActorState, SceneSnapshot

        return SceneSnapshot(
            t=0.0, interactions=[],
            actors=[ActorState(actor_id=aid, kind="observed", cls="car",
                               world_xy=(x, y), heading_deg=h, speed_mps=sp,
                               accel_mps2=0.0)
                    for aid, x, y, h, sp in specs])

    def test_two_parked_cars_are_dropped(self):
        # heading 0° 는 진북(+y) — 줄 서 있는 주차 차량
        snap = self._snap(("A", 0.0, 0.0, 0.0, 0.0), ("B", 0.0, 3.0, 0.0, 0.0))
        keep = bucket_min_gaps(snap, 2, "cv", exclude_static_pairs=False)
        drop = bucket_min_gaps(snap, 2, "cv", exclude_static_pairs=True)
        self.assertEqual(keep[0][0], 0.0)
        self.assertEqual(drop[0][0], float("inf"))

    def test_a_moving_car_closing_on_a_stopped_one_survives(self):
        """신호 대기 중 추돌당하는 것은 사고다."""
        snap = self._snap(("A", 0.0, 0.0, 0.0, 0.0), ("B", 0.0, 40.0, 180.0, 12.0))
        g = bucket_min_gaps(snap, 4, "cv", exclude_static_pairs=True)
        self.assertEqual(min(x for x, _ in g), 0.0)

    def test_default_keeps_the_old_behaviour(self):
        snap = self._snap(("A", 0.0, 0.0, 0.0, 0.0), ("B", 0.0, 3.0, 0.0, 0.0))
        self.assertEqual(bucket_min_gaps(snap, 2, "cv")[0][0], 0.0)
