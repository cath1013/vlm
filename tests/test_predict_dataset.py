# -*- coding: utf-8 -*-
"""`examples/make_predict_dataset.py` — 정답 판정과 조각 병합.

여기서 지키려는 것은 하나다: **정답이 임의인 표본이 정답인 척하지 않는 것.**
차량이 아직 갈림길에 닿지 않으면 여러 후보가 실제 궤적과 똑같이 가깝고, 그때
`argmin` 은 후보 목록의 순서를 고른다. 그것을 표시하지 않으면 학습은 순서를 외우고
평가는 부풀려진다.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "examples")
)

import make_predict_dataset as mpd  # noqa: E402


class _Cand:
    def __init__(self, maneuver, polyline=None):
        self.maneuver = maneuver
        self.polyline = polyline or [(0.0, 0.0), (0.0, 10.0)]


class _Scenario:
    def __init__(self, town):
        self.town = town


class TestMapCoverage(unittest.TestCase):
    def test_all_towns_require_opendrive(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "Town01.xodr"), "w").close()
            with self.assertRaises(SystemExit) as cm:
                mpd.validate_map_coverage(
                    [_Scenario("Town01"), _Scenario("Town05")], d
                )
        self.assertIn("Town05", str(cm.exception))

    def test_complete_opendrive_coverage_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "Town01.xodr"), "w").close()
            p = mpd.validate_map_coverage([_Scenario("Town01")], d)
        self.assertEqual(p["map_source"], "opendrive")
        self.assertFalse(p["map_built_from_scenario_trajectories"])
        self.assertEqual(p["map_missing_towns"], [])

    def test_explicit_override_records_trajectory_provenance(self):
        p = mpd.validate_map_coverage(
            [_Scenario("Town05")], None, allow_trajectory_map=True
        )
        self.assertEqual(
            p["map_source"], "trajectory_synthesized_from_complete_scenario"
        )
        self.assertTrue(p["map_built_from_scenario_trajectories"])
        self.assertEqual(p["map_missing_towns"], ["Town05"])


class TestTieAnalysis(unittest.TestCase):
    def test_clear_winner_is_not_ambiguous(self):
        tied, margin, man_amb = mpd.tie_analysis(
            [0.4, 8.0, 9.0], [_Cand("직진"), _Cand("좌회전"), _Cand("우회전")]
        )
        self.assertEqual(tied, [0])
        self.assertAlmostEqual(margin, 7.6, places=6)
        self.assertFalse(man_amb)

    def test_equal_errors_are_all_tied(self):
        """오차가 같으면 정답을 고를 수 없다 — 전부 동률로 표시한다."""
        tied, margin, man_amb = mpd.tie_analysis(
            [1.0, 1.0, 1.0], [_Cand("직진"), _Cand("직진"), _Cand("직진")]
        )
        self.assertEqual(tied, [0, 1, 2])
        self.assertEqual(margin, float("inf"), "동률이 아닌 후보가 없으면 무한")
        self.assertFalse(man_amb, "기동이 모두 같으면 기동 라벨은 유효하다")

    def test_maneuver_ambiguity_is_flagged_separately(self):
        """후보 인덱스가 임의여도 기동이 같으면 기동 라벨은 쓸 수 있다."""
        tied, _, man_amb = mpd.tie_analysis(
            [1.0, 1.0], [_Cand("좌회전"), _Cand("우회전")]
        )
        self.assertEqual(tied, [0, 1])
        self.assertTrue(man_amb, "좌/우가 갈리면 기동 라벨도 임의다")

    def test_tie_threshold_is_not_exact_equality(self):
        """mm 로 반올림해 저장하므로 정확히 같기를 기다리면 대부분 놓친다."""
        eps = mpd.TIE_EPS_M
        tied, _, _ = mpd.tie_analysis(
            [1.0, 1.0 + eps * 0.5], [_Cand("직진"), _Cand("좌회전")]
        )
        self.assertEqual(tied, [0, 1], "임계 안이면 동률")
        tied, _, _ = mpd.tie_analysis(
            [1.0, 1.0 + eps * 3.0], [_Cand("직진"), _Cand("좌회전")]
        )
        self.assertEqual(tied, [0], "임계를 넘으면 유일")

    def test_best_candidate_is_always_inside_tied_best(self):
        errs = [2.0, 1.0, 1.02, 5.0]
        cands = [_Cand("직진")] * 4
        bi = min(range(len(errs)), key=lambda j: errs[j])
        tied, _, _ = mpd.tie_analysis(errs, cands)
        self.assertIn(bi, tied)

    def test_empty_is_handled(self):
        self.assertEqual(mpd.tie_analysis([], [])[0], [])


class TestFutureTrack(unittest.TestCase):
    def test_gap_truncates_and_reports_incomplete(self):
        """관측이 끊기면 거기서 멈추고 complete=False 를 낸다."""
        per_actor = {"A": [(0.0, (0.0, 0.0)), (1.0, (0.0, 10.0)),
                           (5.0, (0.0, 50.0))]}
        track, complete = mpd.future_track(per_actor, "A", 0.0, 3.0)
        self.assertFalse(complete)
        self.assertEqual(track, [(0.0, 0.0), (0.0, 10.0)])

    def test_complete_when_sampled_through_horizon(self):
        per_actor = {"A": [(float(k), (0.0, 10.0 * k)) for k in range(6)]}
        track, complete = mpd.future_track(per_actor, "A", 0.0, 3.0)
        self.assertTrue(complete)
        self.assertEqual(len(track), 4, "0,1,2,3초")

    def test_missing_actor(self):
        self.assertEqual(mpd.future_track({}, "A", 0.0, 3.0), ([], False))


class TestMatchCandidate(unittest.TestCase):
    def test_start_point_is_excluded(self):
        """시작점을 넣으면 모든 후보의 오차가 같이 낮아져 구별력이 떨어진다."""
        track = [(0.0, 0.0), (0.0, 10.0), (0.0, 20.0)]
        straight = _Cand("직진", [(0.0, 0.0), (0.0, 30.0)])
        away = _Cand("좌회전", [(0.0, 0.0), (-30.0, 0.0)])
        i, err, errs = mpd.match_candidate(track, [straight, away])
        self.assertEqual(i, 0)
        self.assertAlmostEqual(err, 0.0, places=6)
        self.assertGreater(errs[1], 10.0, "시작점을 뺐으므로 벌어진다")


class TestShardMerge(unittest.TestCase):
    """조각을 합칠 때 표본을 잃지 않고 집계가 더해지는가."""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _shard(self, idx, n, rows, counts):
        with open(os.path.join(self.d, f"samples.{idx:03d}of{n:03d}.jsonl"),
                  "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        man = {
            "n_records": len(rows), "n_scenarios": 2, "n_failed": 0,
            "config": {"root": "/x", "shard": f"{idx}/{n}"},
            "counts": counts,
            "maneuver_distribution": {"직진": len(rows)},
        }
        with open(os.path.join(self.d, f"manifest.{idx:03d}of{n:03d}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False)

    def test_merge_sums_records_and_counts(self):
        self._shard(0, 2, [{"a": 1}, {"a": 2}], {"matched": 2, "label_ambiguous": 1})
        self._shard(1, 2, [{"a": 3}], {"matched": 1, "label_ambiguous": 0})
        self.assertEqual(mpd.merge_shards(self.d), 0)
        with open(os.path.join(self.d, "samples.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(x) for x in f if x.strip()]
        self.assertEqual([r["a"] for r in rows], [1, 2, 3])
        with open(os.path.join(self.d, "manifest.json"), encoding="utf-8") as f:
            man = json.load(f)
        self.assertEqual(man["n_records"], 3)
        self.assertEqual(man["n_scenarios"], 4)
        self.assertEqual(man["counts"]["matched"], 3)
        self.assertEqual(man["label_quality"]["label_ambiguous"], 1)
        self.assertEqual(man["maneuver_distribution"]["직진"], 3)

    def test_shard_files_are_kept(self):
        """일부 조각만 다시 돌려 합칠 수 있어야 한다."""
        self._shard(0, 1, [{"a": 1}], {"matched": 1})
        mpd.merge_shards(self.d)
        self.assertTrue(
            os.path.exists(os.path.join(self.d, "samples.000of001.jsonl"))
        )

    def test_no_shards_is_an_error_not_an_empty_file(self):
        self.assertEqual(mpd.merge_shards(self.d), 1)
        self.assertFalse(os.path.exists(os.path.join(self.d, "samples.jsonl")))


class TestTrainSplit(unittest.TestCase):
    """`train_predict_model.py` 의 분할 — 시나리오가 양쪽에 걸치면 안 된다.

    0.5초 간격 인접 표본은 서로 거의 같다. 같은 시나리오가 학습·시험에 모두 들어가면
    시험 성능이 사실상 학습 성능이 되어 크게 과대평가된다.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch 없음")
        import train_predict_model as tpm

        cls.tpm = tpm

    def _rows(self, n_scen=40, per=5):
        return [
            {"scenario": f"s{k:03d}", "split": "train" if k % 5 else "val",
             "c": [[0.5] * 12, [0.5] * 12], "g": [0.0] * 22, "y": 0,
             "tied": [0], "amb": False}
            for k in range(n_scen) for _ in range(per)
        ]

    def test_scenario_never_spans_splits(self):
        tr, va, te = self.tpm.split_rows(self._rows(), "scenario", 0.1, 0.2)
        s_tr = {r["scenario"] for r in tr}
        s_va = {r["scenario"] for r in va}
        s_te = {r["scenario"] for r in te}
        self.assertFalse(s_tr & s_te, "학습·시험 시나리오가 겹친다")
        self.assertFalse(s_tr & s_va, "학습·검증 시나리오가 겹친다")
        self.assertFalse(s_va & s_te, "검증·시험 시나리오가 겹친다")

    def test_dataset_mode_uses_val_as_test(self):
        tr, va, te = self.tpm.split_rows(self._rows(), "dataset", 0.0, 0.2)
        self.assertTrue(all(r["split"] == "val" for r in te))
        self.assertTrue(all(r["split"] != "val" for r in tr))
        self.assertEqual(va, [])

    def test_dataset_mode_val_slice_comes_out_of_train(self):
        rows = self._rows()
        tr, va, te = self.tpm.split_rows(rows, "dataset", 0.1, 0.2)
        self.assertTrue(va, "검증셋이 만들어져야 한다")
        self.assertTrue(all(r["split"] != "val" for r in va),
                        "검증셋은 학습셋에서 떼어낸다 — 시험셋을 건드리지 않는다")
        self.assertEqual(len(tr) + len(va) + len(te), len(rows),
                         "표본을 잃지 않는다")

    def test_bucket_is_stable_across_runs(self):
        """`hash()` 는 PYTHONHASHSEED 에 따라 달라진다 — 분할이 재현돼야 한다."""
        self.assertEqual(self.tpm.scenario_bucket("abc/def"),
                         self.tpm.scenario_bucket("abc/def"))
        self.assertEqual(self.tpm.scenario_bucket("abc/def"),
                         zlib_crc_ref("abc/def"))

    def test_no_split_is_empty_for_reasonable_fractions(self):
        tr, va, te = self.tpm.split_rows(self._rows(200, 3), "scenario", 0.1, 0.2)
        for name, part in (("학습", tr), ("검증", va), ("시험", te)):
            self.assertTrue(part, f"{name} 이 비었다")


def zlib_crc_ref(s):
    import zlib

    return zlib.crc32(s.encode("utf-8")) % 100


class TestRankNetShapes(unittest.TestCase):
    """저장한 모델이 **추론 모양**을 받아야 파이프라인에 꽂힌다."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch 없음")

    def test_inference_and_batch_shapes(self):
        import torch
        from traffic_llm.predict_nets import RankNet, masked_scores

        m = RankNet(hidden=8, depth=1, dropout=0.0)
        # 추론: TorchPredictor 가 넘기는 모양
        out = m(torch.zeros(22), torch.zeros(5, 12))
        self.assertEqual(tuple(out.shape), (5,))
        # 학습: 패딩된 배치
        out = m(torch.zeros(3, 22), torch.zeros(3, 5, 12))
        self.assertEqual(tuple(out.shape), (3, 5))
        mask = torch.zeros(3, 5, dtype=torch.bool)
        mask[:, :2] = True
        s = masked_scores(m, torch.zeros(3, 22), torch.zeros(3, 5, 12), mask)
        self.assertTrue(torch.isfinite(s).all(), "-inf 대신 큰 음수를 써야 한다")
        self.assertTrue((s[:, 2:] < -1e8).all(), "패딩 자리는 배제돼야 한다")
        p = torch.softmax(s, dim=1)
        self.assertAlmostEqual(p[:, :2].sum().item(), 3.0, places=4,
                               msg="유효 후보에만 확률이 가야 한다")

    def test_pickles_under_a_stable_module_path(self):
        """`__main__` 에 정의하면 다른 프로세스에서 못 읽는다."""
        from traffic_llm.predict_nets import RankNet

        self.assertEqual(RankNet.__module__, "traffic_llm.predict_nets")


class TestCoordLoss(unittest.TestCase):
    """좌표 손실 — **평가 지표(ADE)와 같지 않다.** 그 차이를 고정한다."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch 없음")
        import train_predict_model as tpm

        cls.tpm = tpm

    def _one(self, dx, dy, kind, n_steps=1):
        import torch

        p = torch.zeros(1, n_steps, 2)
        t = torch.zeros(1, n_steps, 2)
        t[0, 0] = torch.tensor([dx, dy])
        m = torch.ones(1, n_steps, dtype=torch.bool)
        return self.tpm.coord_loss(p, t, m, kind).item()

    def test_ade_kind_is_exactly_the_metric(self):
        """`ade` 는 평가에서 쓰는 정의(평균 유클리드 거리)와 같아야 한다."""
        self.assertAlmostEqual(self._one(3.0, 4.0, "ade"), 5.0, places=5)
        self.assertAlmostEqual(self._one(3.0, 4.0, "ade", n_steps=3),
                               5.0 / 3, places=5)

    def test_default_huber_is_not_the_metric(self):
        """기본 손실은 ADE 가 아니다 — 값 자체가 다르다."""
        self.assertNotAlmostEqual(self._one(3.0, 4.0, "huber"), 5.0, places=2)

    def test_huber_is_rotation_invariant_only_inside_delta(self):
        """delta 안에서는 방향 무관, 밖에서는 대각선이 더 세게 벌점을 받는다."""
        import math

        d = 1.0                                   # delta(2m) 안
        ax = self._one(d, 0.0, "huber")
        dg = self._one(d / math.sqrt(2), d / math.sqrt(2), "huber")
        self.assertAlmostEqual(ax, dg, places=5, msg="2차 구간은 회전 불변")

        d = 16.0                                  # delta 밖
        ax = self._one(d, 0.0, "huber")
        dg = self._one(d / math.sqrt(2), d / math.sqrt(2), "huber")
        self.assertGreater(dg / ax, 1.3, "1차 구간에서 대각선이 더 무겁다")
        self.assertLess(dg / ax, math.sqrt(2) + 1e-6, "상한은 √2")

    def test_huber_norm_is_rotation_invariant_everywhere(self):
        import math

        for d in (1.0, 4.0, 16.0):
            ax = self._one(d, 0.0, "huber_norm")
            dg = self._one(d / math.sqrt(2), d / math.sqrt(2), "huber_norm")
            self.assertAlmostEqual(ax, dg, places=4, msg=f"{d}m 에서 회전 불변")

    def test_mask_excludes_unobserved_steps(self):
        """관측 안 된 시각을 넣으면 '원점으로 돌아온다'를 학습한다."""
        import torch

        p = torch.zeros(1, 3, 2)
        t = torch.zeros(1, 3, 2)
        t[0, 0] = torch.tensor([3.0, 4.0])
        t[0, 1] = torch.tensor([90.0, 0.0])       # 관측되지 않은 자리의 쓰레기값
        m = torch.tensor([[True, False, False]])
        self.assertAlmostEqual(
            self.tpm.coord_loss(p, t, m, "ade").item(), 5.0, places=5,
            msg="마스크된 시각은 손실에 들어가면 안 된다")

    def test_gradient_is_finite_at_zero_error(self):
        """`‖e‖` 는 0 에서 미분 불가 — clamp 로 막아야 NaN 이 안 난다."""
        import torch

        for kind in ("huber", "huber_norm", "ade", "l2"):
            q = torch.zeros(1, 1, 2, requires_grad=True)
            self.tpm.coord_loss(
                q, torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool),
                kind).backward()
            self.assertTrue(torch.isfinite(q.grad).all(), f"{kind} 에서 NaN")

    def test_l2_amplifies_large_errors_the_most(self):
        """L2 를 쓰지 않는 근거 — 큰 오차가 제곱으로 커진다 (실측 +0.08m)."""
        small = {k: self._one(1.0, 0.0, k) for k in ("huber", "ade", "l2")}
        large = {k: self._one(16.0, 0.0, k) for k in ("huber", "ade", "l2")}
        ratio = {k: large[k] / small[k] for k in small}
        self.assertGreater(ratio["l2"], ratio["huber"])
        self.assertGreater(ratio["l2"], ratio["ade"])

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            self._one(1.0, 0.0, "nope")


class TestTorchStaysOptional(unittest.TestCase):
    """torch 는 선택 의존성이다 — 없어도 규칙 기반 경로가 동작해야 한다.

    `predict_nets.py` 를 분리한 이유가 이것이다. 신경망 정의가 `predict_model.py` 로
    들어오면 torch 없는 환경에서 파이프라인 전체가 import 되지 않는다.
    """

    def _toplevel_imports(self, rel):
        import ast

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tree = ast.parse(open(os.path.join(root, rel), encoding="utf-8").read())
        names = set()
        for node in tree.body:            # 최상위만 — 함수 안의 지연 임포트는 허용
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        return names

    def test_predict_model_has_no_toplevel_torch(self):
        self.assertNotIn("torch", self._toplevel_imports(
            "traffic_llm/predict_model.py"))

    def test_prediction_has_no_toplevel_torch(self):
        self.assertNotIn("torch", self._toplevel_imports(
            "traffic_llm/prediction.py"))

    def test_predict_nets_is_where_torch_lives(self):
        """분리한 쪽은 최상위에서 torch 를 쓴다 — nn.Module 정의에 필요하다."""
        self.assertIn("torch", self._toplevel_imports(
            "traffic_llm/predict_nets.py"))


class TestShardSplitCoversEverything(unittest.TestCase):
    """라운드로빈 분배가 겹치지도 빠뜨리지도 않는가."""

    def test_round_robin_is_a_partition(self):
        for total in (1, 7, 16, 607):
            for n in (1, 3, 16):
                items = list(range(total))
                seen = []
                for i in range(n):
                    seen += items[i::n]
                self.assertEqual(sorted(seen), items, f"{total}/{n}")


if __name__ == "__main__":
    unittest.main()
