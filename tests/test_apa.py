"""APA — DeepAccident 사고 예측 지표의 이식본.

이 지표는 **비교의 기준**이므로 원본 상수와 결함이 조용히 어긋나면 안 된다.
결함(마지막 프레임만·FP 는 평균 표본만·빗나감은 FP)은 발표된 숫자를 만든 코드의
일부이므로 **재현되는지도** 시험한다.
"""

import unittest

import numpy as np

from deepaccident_replicate.apa import (
    DIST_THRESHOLD_GT_M,
    DIST_THRESHOLD_PRED_M,
    MAX_TIME_ERROR_S,
    TP_POSITION_THRESHOLDS_M,
    ApaConfig,
    ApaCounts,
    aggregate,
    find_accident,
    score_window,
)


def acc(ids, pa, pb, gap=0.0) -> dict:
    return {"ids": tuple(ids),
            "pos": (np.array(pa, dtype=float), np.array(pb, dtype=float)),
            "gap_m": gap}


def scene(*specs):
    """(id, x, y, yaw, length, width) 들 → find_accident 인자."""
    ids = [s[0] for s in specs]
    pos = np.array([[s[1], s[2]] for s in specs], dtype=np.float32)
    yaw = np.array([s[3] for s in specs], dtype=np.float32)
    size = np.array([[s[4], s[5]] for s in specs], dtype=np.float32)
    return pos, yaw, size, ids


class TestConstants(unittest.TestCase):
    """원본 코드에서 읽어온 값 (`multi_gpu_test.py`)."""

    def test_gt_threshold_is_essentially_contact(self):
        # :526 dist_threshold_gt = 0.001 px, 0.5 m/px
        self.assertAlmostEqual(DIST_THRESHOLD_GT_M, 0.0005)

    def test_prediction_threshold_is_five_pixels(self):
        # :527 dist_threshold_pred = 5 px
        self.assertEqual(DIST_THRESHOLD_PRED_M, 2.5)

    def test_position_thresholds(self):
        self.assertEqual(TP_POSITION_THRESHOLDS_M, (5.0, 10.0, 15.0))

    def test_time_error_cap(self):
        self.assertEqual(MAX_TIME_ERROR_S, 1.5)

    def test_defaults_reproduce_the_released_code(self):
        c = ApaConfig()
        self.assertTrue(c.last_frame_only)
        self.assertTrue(c.fp_from_mean_only)
        self.assertTrue(c.miss_is_fp_not_fn)

    def test_debugged_turns_all_three_off(self):
        c = ApaConfig.debugged()
        self.assertFalse(c.last_frame_only)
        self.assertFalse(c.fp_from_mean_only)
        self.assertFalse(c.miss_is_fp_not_fn)


class TestFindAccident(unittest.TestCase):
    def test_closest_pair_below_threshold(self):
        p, y, s, i = scene(("A", 0, 0, 0, 4, 2), ("B", 4.5, 0, 0, 4, 2),
                           ("C", 40, 0, 0, 4, 2))
        v = find_accident(p, y, s, i, 2.5)
        self.assertEqual(set(v["ids"]), {"A", "B"})

    def test_nothing_when_all_far(self):
        p, y, s, i = scene(("A", 0, 0, 0, 4, 2), ("B", 40, 0, 0, 4, 2))
        self.assertIsNone(find_accident(p, y, s, i, 2.5))

    def test_gt_threshold_needs_actual_contact(self):
        """GT 임계는 0.0005 m — 2 m 떨어진 쌍은 GT 사고가 아니다."""
        p, y, s, i = scene(("A", 0, 0, 0, 4, 2), ("B", 6.0, 0, 0, 4, 2))
        self.assertIsNone(find_accident(p, y, s, i, DIST_THRESHOLD_GT_M))
        self.assertIsNotNone(find_accident(p, y, s, i, DIST_THRESHOLD_PRED_M))

    def test_single_actor(self):
        p, y, s, i = scene(("A", 0, 0, 0, 4, 2))
        self.assertIsNone(find_accident(p, y, s, i, 2.5))


class TestScoreWindow(unittest.TestCase):
    N = len(TP_POSITION_THRESHOLDS_M)

    def test_exact_match_is_tp(self):
        gt = [None, None, None, acc(("A", "B"), (0, 0), (5, 0))]
        pred = [[None, None, None, acc(("A", "B"), (0, 0), (5, 0))]]
        c = score_window(gt, pred)
        self.assertEqual(c.tp, [1] * self.N)
        self.assertEqual(c.fp, [0] * self.N)
        self.assertEqual(c.id_err, [0] * self.N)

    def test_position_error_is_the_sum_of_both_agents(self):
        """임계 {5,10,15} m 는 두 차량 오차의 **합**이다 — 차량당 {2.5,5,7.5} m."""
        gt = [None, None, None, acc(("A", "B"), (0, 0), (5, 0))]
        pred = [[None, None, None, acc(("A", "B"), (3, 0), (8, 0))]]  # 합 6 m
        c = score_window(gt, pred)
        self.assertEqual(c.tp, [0, 1, 1])   # D=5 는 놓치고 10·15 는 맞는다
        self.assertEqual(c.fp, [1, 0, 0])

    def test_swapped_pair_order_does_not_penalise(self):
        gt = [None, None, None, acc(("A", "B"), (0, 0), (5, 0))]
        pred = [[None, None, None, acc(("B", "A"), (5, 0), (0, 0))]]
        self.assertEqual(score_window(gt, pred).tp, [1] * self.N)

    def test_wrong_ids_at_the_right_place_raise_id_error(self):
        gt = [None, None, None, acc(("A", "B"), (0, 0), (5, 0))]
        pred = [[None, None, None, acc(("C", "D"), (0, 0), (5, 0))]]
        c = score_window(gt, pred)
        self.assertEqual(c.tp, [1] * self.N)
        self.assertEqual(c.id_err, [1] * self.N)

    def test_no_prediction_at_all_is_fn(self):
        gt = [None, None, None, acc(("A", "B"), (0, 0), (5, 0))]
        c = score_window(gt, [[None] * 4])
        self.assertEqual(c.fn, [1] * self.N)
        self.assertEqual(c.fp, [0] * self.N)

    def test_no_gt_and_no_prediction_counts_nothing(self):
        """APA 는 true negative 를 세지 않는다 — 이것이 지표의 가장 약한 지점이다."""
        c = score_window([None] * 4, [[None] * 4])
        self.assertEqual((c.tp, c.fp, c.fn), ([0] * 3, [0] * 3, [0] * 3))


class TestReproducedDefects(unittest.TestCase):
    """원본의 결함 셋. 발표된 숫자를 만든 코드이므로 기본값으로 재현된다."""

    def test_only_the_last_frame_is_scored(self):
        """중간 프레임의 사고는 원본에서 보이지 않는다 (`:553` 의 루프 버그)."""
        gt = [acc(("A", "B"), (0, 0), (5, 0)), None, None, None]
        pred = [[acc(("A", "B"), (0, 0), (5, 0)), None, None, None]]
        faithful = score_window(gt, pred, ApaConfig())
        fixed = score_window(gt, pred, ApaConfig.debugged())
        self.assertEqual(faithful.tp, [0, 0, 0])   # 마지막 프레임엔 아무것도 없다
        self.assertEqual(fixed.tp, [1, 1, 1])

    def test_time_error_is_identically_zero_in_the_faithful_version(self):
        """예측과 정답을 같은 프레임에서만 고르므로 시간 오차가 0 일 수밖에 없다."""
        gt = [None, None, None, acc(("A", "B"), (0, 0), (5, 0))]
        pred = [[acc(("A", "B"), (0, 0), (5, 0))] * 4]
        self.assertEqual(score_window(gt, pred, ApaConfig()).time_err, [0.0] * 3)

    def test_false_alarms_from_stochastic_samples_are_free(self):
        """GT 사고가 없을 때 FP 는 **평균 표본(마지막)** 만 본다 (`:646`)."""
        gt = [None] * 4
        noisy = [[None, None, None, acc(("A", "B"), (0, 0), (5, 0))],  # 표본 0
                 [None] * 4]                                           # 평균 표본
        self.assertEqual(score_window(gt, noisy, ApaConfig()).fp, [0, 0, 0])
        self.assertEqual(score_window(gt, noisy, ApaConfig.debugged()).fp, [1, 1, 1])

    def test_a_missed_accident_is_fp_not_fn(self):
        """GT 사고가 있는데 예측이 임계 밖이면 FP 만 오른다 (FN 줄은 주석 처리)."""
        gt = [None, None, None, acc(("A", "B"), (0, 0), (5, 0))]
        pred = [[None, None, None, acc(("A", "B"), (100, 0), (105, 0))]]
        faithful = score_window(gt, pred, ApaConfig())
        fixed = score_window(gt, pred, ApaConfig.debugged())
        self.assertEqual((faithful.fp, faithful.fn), ([1] * 3, [0] * 3))
        self.assertEqual((fixed.fp, fixed.fn), ([1] * 3, [1] * 3))


class TestAggregate(unittest.TestCase):
    def test_apa_formula(self):
        """APA = TP / (TP + 0.5·FP + 0.5·FN) (`single_gpu_test.py:772`)."""
        c = ApaCounts.zeros(3)
        c.tp = [6, 6, 6]
        c.fp = [4, 4, 4]
        c.fn = [2, 2, 2]
        r = aggregate(c)
        self.assertAlmostEqual(r["per_threshold"][0]["APA"], 6 / (6 + 2 + 1), places=4)
        self.assertAlmostEqual(r["APA"], 6 / 9, places=4)

    def test_perfect_is_one(self):
        c = ApaCounts.zeros(3)
        c.tp = [5, 5, 5]
        self.assertEqual(aggregate(c)["APA"], 1.0)

    def test_no_events_is_zero_not_undefined(self):
        self.assertEqual(aggregate(ApaCounts.zeros(3))["APA"], 0.0)

    def test_tp_metrics_are_averaged_over_true_positives(self):
        c = ApaCounts.zeros(3)
        c.tp = [4, 4, 4]
        c.pos_err = [8.0, 8.0, 8.0]
        c.id_err = [1, 1, 1]
        p = aggregate(c)["per_threshold"][0]
        self.assertAlmostEqual(p["position_error_m"], 2.0)
        self.assertAlmostEqual(p["id_error"], 0.25)

    def test_result_records_that_tn_is_ignored(self):
        self.assertIn("true negative", aggregate(ApaCounts.zeros(3))["note"])


if __name__ == "__main__":
    unittest.main()
