"""DeepAccident 사고 판정 규칙 이식본.

이 규칙은 **비교 대상**이므로, 우리에게 유리한 쪽으로 어긋나면 비교 자체가 무의미해
진다. 그래서 원본 코드의 상수·부등호·지평을 값으로 잠근다.
"""

import math
import unittest

from traffic_llm.accident_qa import score_binary, score_modes
from deepaccident_replicate.da_baseline import (
    DA_DIST_THRESHOLD_PRED_M,
    DA_HORIZON_S,
    VARIANT_NAMES,
    DeepAccidentRule,
    build_response,
    judge,
    looked_at,
    variants,
)

INF = float("inf")


def gaps(*vals):
    """구간별 (최소 간격, 쌍). None 이면 무한대(쌍 없음)."""
    return [
        (INF, None) if v is None else (float(v), ("V001", "V002")) for v in vals
    ]


class TestRuleConstants(unittest.TestCase):
    """원본 코드에서 읽어온 값. 바꾸려면 출처를 함께 바꿔야 한다."""

    def test_prediction_threshold_is_five_pixels(self):
        # multi_gpu_test.py:527 dist_threshold_pred = 5 (px), 0.5 m/px
        self.assertEqual(DA_DIST_THRESHOLD_PRED_M, 2.5)

    def test_default_horizon_is_two_seconds(self):
        # configs/DeepAccident_tiny.py — 2 Hz × 4 미래 프레임
        self.assertEqual(DA_HORIZON_S, 2.0)

    def test_defaults_match_the_released_code(self):
        r = DeepAccidentRule()
        self.assertEqual(r.dist_threshold_m, 2.5)
        self.assertEqual(r.horizon_s, 2.0)

    def test_bad_frame_rule_raises(self):
        with self.assertRaises(ValueError):
            DeepAccidentRule(frame_rule="middle")

    def test_bad_horizon_raises(self):
        with self.assertRaises(ValueError):
            DeepAccidentRule(horizon_s=0.0)


class TestLookedAt(unittest.TestCase):
    def test_horizon_limits_what_the_rule_sees(self):
        """2초 지평이면 3·4·5 구간은 보지 않는다 — 우리와의 구조적 차이다."""
        self.assertEqual(looked_at(DeepAccidentRule(horizon_s=2.0), 5), [1, 2])
        self.assertEqual(looked_at(DeepAccidentRule(horizon_s=5.0), 5), [1, 2, 3, 4, 5])

    def test_last_rule_keeps_only_the_final_bucket(self):
        r = DeepAccidentRule(horizon_s=2.0, frame_rule="last")
        self.assertEqual(looked_at(r, 5), [2])

    def test_horizon_cannot_exceed_the_question(self):
        self.assertEqual(looked_at(DeepAccidentRule(horizon_s=5.0), 2), [1, 2])

    def test_fractional_horizon_rounds_up(self):
        self.assertEqual(looked_at(DeepAccidentRule(horizon_s=1.5), 5), [1, 2])


class TestJudge(unittest.TestCase):
    def test_fires_below_threshold(self):
        v = judge(gaps(9, 1.0, 9, 9, 9), DeepAccidentRule(horizon_s=5.0))
        self.assertTrue(v["accident"])
        self.assertEqual(v["k"], 2)

    def test_silent_above_threshold(self):
        v = judge(gaps(9, 9, 9, 9, 9), DeepAccidentRule(horizon_s=5.0))
        self.assertFalse(v["accident"])
        self.assertIsNone(v["k"])

    def test_comparison_is_strict(self):
        """원본은 `preddist < dist_threshold_pred` 다 — 같은 값은 사고가 아니다."""
        r = DeepAccidentRule(horizon_s=5.0)
        self.assertFalse(judge(gaps(2.5, 9, 9, 9, 9), r)["accident"])
        self.assertTrue(judge(gaps(2.499, 9, 9, 9, 9), r)["accident"])

    def test_picks_the_closest_bucket(self):
        v = judge(gaps(2.0, 0.3, 1.0, 9, 9), DeepAccidentRule(horizon_s=5.0))
        self.assertEqual(v["k"], 2)
        self.assertEqual(v["gap_m"], 0.3)

    def test_beyond_horizon_is_invisible(self):
        """2초 지평이면 4번 구간의 접촉은 보이지 않는다."""
        g = gaps(9, 9, 9, 0.1, 9)
        self.assertFalse(judge(g, DeepAccidentRule(horizon_s=2.0))["accident"])
        self.assertTrue(judge(g, DeepAccidentRule(horizon_s=5.0))["accident"])

    def test_last_rule_ignores_earlier_buckets(self):
        """공개 코드는 마지막 프레임만 채점한다 — 그 전에 붙어도 못 본다."""
        g = gaps(0.1, 9, 9, 9, 9)
        r_last = DeepAccidentRule(horizon_s=2.0, frame_rule="last")
        r_any = DeepAccidentRule(horizon_s=2.0, frame_rule="any")
        self.assertFalse(judge(g, r_last)["accident"])
        self.assertTrue(judge(g, r_any)["accident"])

    def test_all_infinite_gaps(self):
        v = judge(gaps(None, None), DeepAccidentRule(horizon_s=2.0))
        self.assertFalse(v["accident"])
        self.assertIsNone(v["gap_m"])

    def test_empty_gaps(self):
        v = judge([], DeepAccidentRule())
        self.assertFalse(v["accident"])
        self.assertEqual(v["looked_at"], [])


class TestResponse(unittest.TestCase):
    def test_marks_exactly_one_bucket(self):
        """DeepAccident 는 최소 거리 시점 하나를 낼 뿐 구간별 답을 내지 않는다."""
        r = build_response(5.0, 5, gaps(9, 0.5, 0.2, 9, 9),
                           DeepAccidentRule(horizon_s=5.0))
        hits = [p["k"] for p in r["predictions"] if p["accident_expected"]]
        self.assertEqual(hits, [3])

    def test_marks_nothing_when_silent(self):
        r = build_response(5.0, 5, gaps(9, 9, 9, 9, 9),
                           DeepAccidentRule(horizon_s=5.0))
        self.assertFalse(any(p["accident_expected"] for p in r["predictions"]))

    def test_fills_every_bucket_of_the_question(self):
        """지평 밖도 답을 채운다 — '보지 않았다' 가 곧 '사고 없음' 이다."""
        r = build_response(5.0, 5, gaps(9, 0.1, 9, 9, 9),
                           DeepAccidentRule(horizon_s=2.0))
        self.assertEqual([p["k"] for p in r["predictions"]], [1, 2, 3, 4, 5])
        beyond = [p for p in r["predictions"] if p["k"] > 2]
        self.assertTrue(all("지평" in p["reason"] for p in beyond))

    def test_intervals_match_our_convention(self):
        r = build_response(5.0, 3, gaps(9, 9, 9), DeepAccidentRule(horizon_s=5.0))
        self.assertEqual(r["predictions"][0]["interval_s"], "(5.0, 6.0]")
        self.assertEqual(r["predictions"][2]["interval_s"], "(7.0, 8.0]")

    def test_names_the_pair_it_fired_on(self):
        r = build_response(5.0, 5, gaps(9, 0.4, 9, 9, 9),
                           DeepAccidentRule(horizon_s=5.0))
        hit = [p for p in r["predictions"] if p["accident_expected"]][0]
        self.assertEqual(hit["involved_actor_ids"], ["V001", "V002"])

    def test_records_that_the_perception_stack_is_missing(self):
        """가중치가 없다는 사실이 결과물에 남아야 한다."""
        r = build_response(5.0, 2, gaps(9, 9))
        self.assertIn("V2XFormer", r["data_limitations"])


class TestScoresLikeAnLlmAnswer(unittest.TestCase):
    """같은 채점기를 통과해야 같은 표에 오른다."""

    @staticmethod
    def _gt(collision_s):
        from traffic_llm.accident_qa import WindowConfig, build_windows, window_ground_truth
        from traffic_llm.schemas import CollisionTruth
        from tests.test_accident_qa import make_snapshots, pick

        snaps = make_snapshots(n=21, dt=0.5)
        cfg = WindowConfig(window_s=5.0, horizon_s=5.0)
        wins, _ = build_windows(snaps, cfg)
        w = pick(wins, "0-5")  # t_end = 5.0
        coll = CollisionTruth(occurred=True, carla_ids=(7001,),
                              time_s=collision_s, frame=0)
        return window_ground_truth(w, cfg, coll, {"v1": 1}, "s", "sp", "T")

    def test_flows_through_score_modes(self):
        gt = self._gt(8.2)  # k_true = 4
        r = build_response(5.0, 5, gaps(9, 9, 9, 0.2, 9),
                           DeepAccidentRule(horizon_s=5.0))
        m = score_modes(r, gt)
        self.assertEqual(m["binary"]["status"], "TP")
        self.assertEqual(m["strict"]["counts"]["TP"], 1)
        self.assertEqual(m["weighted"]["score"], 1.0)

    def test_short_horizon_misses_a_late_collision(self):
        """2초만 보는 규칙은 4번 구간의 사고를 놓친다 — 비교의 핵심 축이다."""
        gt = self._gt(8.2)  # k_true = 4
        r = build_response(5.0, 5, gaps(9, 9, 9, 0.2, 9),
                           DeepAccidentRule(horizon_s=2.0))
        self.assertEqual(score_binary(r, gt)["status"], "FN")

    def test_marks_one_bucket_so_binary_is_the_fair_axis(self):
        """구간 하나만 표시하므로 구간별 정확도는 미표시 TN 으로 부풀려진다."""
        gt = self._gt(8.2)
        r = build_response(5.0, 5, gaps(9, 9, 9, 9, 9),
                           DeepAccidentRule(horizon_s=5.0))
        m = score_modes(r, gt)
        self.assertEqual(m["binary"]["status"], "FN")     # 못 맞혔다
        self.assertGreaterEqual(m["strict"]["counts"]["TN"], 3)  # 그래도 TN 이 쌓인다


class TestVariants(unittest.TestCase):
    def test_default_set(self):
        v = variants()
        self.assertEqual(sorted(v), ["da", "da_any", "da_h5"])
        self.assertEqual(v["da"].frame_rule, "last")
        self.assertEqual(v["da_h5"].horizon_s, 5.0)

    def test_fitted_variant_is_added_on_demand(self):
        v = variants(0.75)
        self.assertIn("da_fit", v)
        self.assertEqual(v["da_fit"].dist_threshold_m, 0.75)
        self.assertEqual(v["da_fit"].horizon_s, 5.0)

    def test_variant_names_covers_everything_variants_can_make(self):
        self.assertEqual(set(variants(1.0)), set(VARIANT_NAMES))

    def test_each_variant_changes_one_thing(self):
        v = variants()
        self.assertEqual(v["da"].dist_threshold_m, v["da_any"].dist_threshold_m)
        self.assertEqual(v["da"].horizon_s, v["da_any"].horizon_s)
        self.assertEqual(v["da_any"].frame_rule, v["da_h5"].frame_rule)


if __name__ == "__main__":
    unittest.main()
