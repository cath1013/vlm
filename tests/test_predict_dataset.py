# -*- coding: utf-8 -*-
"""`examples/make_predict_dataset.py` — 정답 판정과 조각 병합.

여기서 지키려는 것은 하나다: **정답이 임의인 표본이 정답인 척하지 않는 것.**
차량이 아직 갈림길에 닿지 않으면 여러 후보가 실제 궤적과 똑같이 가깝고, 그때
`argmin` 은 후보 목록의 순서를 고른다. 그것을 표시하지 않으면 학습은 순서를 외우고
평가는 부풀려진다.
"""

import json
import math
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
import make_scene_predict_dataset as smpd  # noqa: E402
import repair_scene_predict_dataset_splits as srps  # noqa: E402


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


class TestSceneDatasetRecords(unittest.TestCase):
    def test_keeps_no_route_and_partial_or_zero_future_actors(self):
        """Scene generator must not inherit actor-row filtering semantics."""
        from types import SimpleNamespace

        a0 = SimpleNamespace(actor_id="A", cls="car", world_xy=(0.0, 0.0))
        b0 = SimpleNamespace(actor_id="B", cls="truck", world_xy=(5.0, 0.0))
        a1 = SimpleNamespace(actor_id="A", cls="car", world_xy=(1.0, 0.0))
        a2 = SimpleNamespace(actor_id="A", cls="car", world_xy=(2.0, 0.0))
        a3 = SimpleNamespace(actor_id="A", cls="car", world_xy=(3.0, 0.0))
        snaps = [SimpleNamespace(t=float(i), actors=actors)
                 for i, actors in enumerate(([a0, b0], [a1], [a2], [a3]))]
        res = SimpleNamespace(
            snapshots=lambda rate_hz: iter(snaps), network=object(),
            scenario=SimpleNamespace(scenario_id="type_normal/s", split="train", town="Town01"),
        )
        old_context, old_encode = smpd.build_context, smpd.encode
        try:
            smpd.build_context = lambda actor, *a, **kw: SimpleNamespace(candidates=[])
            smpd.encode = lambda ctx: {"global": [0.0] * 22,
                                       "history": [[0.0] * 5 for _ in range(5)],
                                       "candidates": [], "interactions": []}
            cfg = SimpleNamespace(prediction_horizon_s=5.0)
            scenes, _ = smpd.build_scene_records(res, cfg, 1.0, dataset_split="train")
        finally:
            smpd.build_context, smpd.encode = old_context, old_encode
        first = scenes[0]["actors"]
        self.assertEqual([x["actor_id"] for x in first], ["A", "B"])
        self.assertFalse(first[1]["has_route_candidate"])
        self.assertEqual(first[0]["target_mask"], [True, True, True, False, False])
        self.assertEqual(first[1]["target_mask"], [False] * 5)

    def test_source_split_is_preserved_when_built_result_split_differs(self):
        from types import SimpleNamespace

        a0 = SimpleNamespace(actor_id="A", cls="car", world_xy=(0.0, 0.0))
        a1 = SimpleNamespace(actor_id="A", cls="car", world_xy=(1.0, 0.0))
        res = SimpleNamespace(
            snapshots=lambda rate_hz: iter([
                SimpleNamespace(t=0.0, actors=[a0]),
                SimpleNamespace(t=1.0, actors=[a1]),
            ]),
            network=object(),
            scenario=SimpleNamespace(
                scenario_id="type_normal/s", split="DeepAccident_mini", town="Town01"
            ),
        )
        old_context, old_encode = smpd.build_context, smpd.encode
        try:
            smpd.build_context = lambda actor, *a, **kw: SimpleNamespace(candidates=[])
            smpd.encode = lambda ctx: {"global": [0.0] * 22,
                                       "history": [[0.0] * 5 for _ in range(5)],
                                       "candidates": [], "interactions": []}
            mismatches = {"count": 0, "by_pair": {}, "details": []}
            scenes, _ = smpd.build_scene_records(
                res, SimpleNamespace(prediction_horizon_s=1.0), 1.0,
                dataset_split="train", split_mismatches=mismatches,
            )
        finally:
            smpd.build_context, smpd.encode = old_context, old_encode
        self.assertEqual(scenes[0]["dataset_split"], "train")
        self.assertEqual(mismatches["count"], 1)
        self.assertEqual(mismatches["by_pair"], {"train->DeepAccident_mini": 1})
        self.assertEqual(mismatches["details"][0]["scenario_id"], "type_normal/s")

    def test_default_scene_source_excludes_mini_and_unknown_splits(self):
        from types import SimpleNamespace

        scenarios = [SimpleNamespace(split=s) for s in (
            "train", "train", "val", "DeepAccident_mini", "mystery"
        )]
        included, source, include_counts, excluded = smpd.select_source_scenarios(scenarios)
        self.assertEqual([s.split for s in included], ["train", "train", "val"])
        self.assertEqual(source["DeepAccident_mini"], 1)
        self.assertEqual(include_counts, {"train": 2, "val": 1})
        self.assertEqual(excluded, {"DeepAccident_mini": 1, "mystery": 1})

    def test_explicit_scene_source_split_is_official_only(self):
        from types import SimpleNamespace

        scenarios = [SimpleNamespace(split=s) for s in ("train", "val", "DeepAccident_mini")]
        included, _, counts, _ = smpd.select_source_scenarios(scenarios, "train")
        self.assertEqual([s.split for s in included], ["train"])
        self.assertEqual(counts, {"train": 1})
        with self.assertRaises(SystemExit):
            smpd.select_source_scenarios(scenarios, "DeepAccident_mini")


class TestSceneSplitRepair(unittest.TestCase):
    def _scenario(self, sid, split):
        from types import SimpleNamespace
        return SimpleNamespace(scenario_id=sid, split=split)

    def _scene(self, sid, split, actor_id="a"):
        return {"scenario_id": sid, "dataset_split": split, "actors": [{
            "actor_id": actor_id, "has_route_candidate": False,
            "target_mask": [False] * 5,
        }]}

    def test_repair_uses_authoritative_source_split_and_only_changes_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            scenes = [self._scene("type/s", "DeepAccident_mini")]
            with open(os.path.join(d, "scenes.jsonl"), "w", encoding="utf-8") as f:
                for scene in scenes:
                    f.write(json.dumps(scene) + "\n")
            with open(os.path.join(d, "dataset_report.json"), "w", encoding="utf-8") as f:
                json.dump({"n_failed": 0, "config": {"x": 1}}, f)
            report, summary = srps.repair_dataset(
                d, [self._scenario("type/s", "train")], apply=True
            )
            with open(os.path.join(d, "scenes.jsonl"), encoding="utf-8") as f:
                repaired = json.loads(f.readline())
        self.assertEqual(repaired["dataset_split"], "train")
        self.assertEqual(repaired["actors"], scenes[0]["actors"])
        self.assertEqual(summary["corrections"], 1)
        self.assertEqual(report["repair_split_corrections"]["by_pair"],
                         {"DeepAccident_mini->train": 1})

    def test_repair_fails_for_unresolved_ambiguous_or_unsupported_source(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "scenes.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps(self._scene("missing", "train")) + "\n")
            with open(os.path.join(d, "dataset_report.json"), "w", encoding="utf-8") as f:
                json.dump({}, f)
            with self.assertRaisesRegex(ValueError, "unresolved"):
                srps.repair_dataset(d, [self._scenario("other", "train")])
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                srps.repair_dataset(d, [self._scenario("missing", "train"),
                                        self._scenario("missing", "val")])
            with self.assertRaisesRegex(ValueError, "unsupported"):
                srps.repair_dataset(d, [self._scenario("missing", "DeepAccident_mini")])


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


class TestPairSeparationLoss(unittest.TestCase):
    """정상 장면 pair 보조 손실의 한쪽 방향 margin을 고정한다."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch 없음")
        import train_predict_model as tpm

        cls.tpm = tpm

    def _loss(self, predicted_b_x):
        import torch

        class FixedModel(torch.nn.Module):
            accepts_history = False
            accepts_interactions = False

            def forward(self, g, c):
                return g

        # 두 actor의 GT world 위치는 (0, 0), (10, 0) 이다. model output은
        # 각 actor의 origin-relative offset이고, test에서는 origin을 0으로 둔다.
        d = {
            "g": torch.tensor([[[0.0, 0.0], [0.0, 0.0]],
                               [[predicted_b_x, 0.0], [predicted_b_x, 0.0]]]),
            "c": torch.zeros(2, 1, 1),
            "hist": torch.zeros(2, 1, 1),
            "interaction": torch.zeros(2, 0, 1),
            "interaction_mask": torch.zeros(2, 0, dtype=torch.bool),
        }
        w = {
            "tgt": torch.zeros(2, 2, 2),
            "tmask": torch.ones(2, 2, dtype=torch.bool),
            "origin": torch.tensor([[0.0, 0.0], [10.0, 0.0]]),
        }
        return self.tpm.pair_separation_loss(
            FixedModel(), d, w, torch.tensor([[0, 1]]), tolerance_m=0.5,
            near_m=10.0,
        ).item()

    def test_zero_when_prediction_is_not_too_close(self):
        # GT 10m, tolerance .5m: 9.5m 이상이면 벌점이 없어야 한다.
        self.assertEqual(self._loss(-0.5), 0.0)
        self.assertEqual(self._loss(2.0), 0.0)

    def test_positive_when_prediction_is_much_too_close(self):
        # actor B의 world 위치는 10 + (-8) = 2m → GT보다 8m 가까워진다.
        self.assertGreater(self._loss(-8.0), 0.0)

    def test_far_gt_timesteps_are_ignored(self):
        import torch

        class FixedModel(torch.nn.Module):
            accepts_history = False
            accepts_interactions = False

            def forward(self, g, c):
                return g

        # t=1: GT/pred 모두 8m라 벌점 없음. t=2: GT 30m, pred 5m로
        # 매우 가깝지만 GT가 near_m 밖이므로 이 timestep은 완전히 무시해야 한다.
        d = {
            "g": torch.tensor([[[0.0, 0.0], [0.0, 0.0]],
                               [[8.0, 0.0], [5.0, 0.0]]]),
            "c": torch.zeros(2, 1, 1),
            "hist": torch.zeros(2, 1, 1),
            "interaction": torch.zeros(2, 0, 1),
            "interaction_mask": torch.zeros(2, 0, dtype=torch.bool),
        }
        w = {
            "tgt": torch.tensor([[[0.0, 0.0], [0.0, 0.0]],
                                 [[8.0, 0.0], [30.0, 0.0]]]),
            "tmask": torch.ones(2, 2, dtype=torch.bool),
            "origin": torch.zeros(2, 2),
        }
        got = self.tpm.pair_separation_loss(
            FixedModel(), d, w, torch.tensor([[0, 1]]), tolerance_m=0.5,
            near_m=10.0,
        )
        self.assertEqual(got.item(), 0.0)

    def test_accident_rows_are_excluded_from_pair_index(self):
        def row(sid, actor, x):
            return {"scenario_id": sid, "t_s": 1.0, "actor_id": actor,
                    "tgt": [[0.0, 0.0], [0.0, 0.0]],
                    "origin_enu": [x, 0.0]}

        rows = [
            row("type1_subtype1_normal/normal_scene", "A", 0.0),
            row("type1_subtype1_normal/normal_scene", "B", 5.0),
            row("type1_subtype1_accident/accident_scene", "C", 0.0),
            row("type1_subtype1_accident/accident_scene", "D", 5.0),
        ]
        self.assertEqual(self.tpm.build_normal_pair_index(rows, 1, 10.0), [(0, 1)])

    def test_zero_pair_weight_keeps_base_loss_exactly(self):
        import torch

        base = torch.tensor(3.0, requires_grad=True)
        pair = torch.tensor(7.0, requires_grad=True)
        got = self.tpm.combine_waypoint_losses(base, pair, 0.0)
        self.assertIs(got, base)
        got.backward()
        self.assertEqual(base.grad.item(), 1.0)
        self.assertIsNone(pair.grad)


class TestPairRelativeLoss(unittest.TestCase):
    """GT-relative auxiliary loss는 거리 증가가 아니라 상대 기하를 맞춘다."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch 없음")
        import train_predict_model as tpm

        cls.tpm = tpm

    @staticmethod
    def _model():
        import torch

        class FixedModel(torch.nn.Module):
            accepts_history = False
            accepts_interactions = False

            def forward(self, g, c):
                return g

        return FixedModel()

    @staticmethod
    def _data(pred_a, pred_b, gt_a, gt_b):
        import torch

        K = len(pred_a)
        return (
            {"g": torch.tensor([pred_a, pred_b], dtype=torch.float32),
             "c": torch.zeros(2, 1, 1),
             "hist": torch.zeros(2, 1, 1),
             "interaction": torch.zeros(2, 0, 1),
             "interaction_mask": torch.zeros(2, 0, dtype=torch.bool)},
            {"tgt": torch.tensor([gt_a, gt_b], dtype=torch.float32),
             "tmask": torch.ones(2, K, dtype=torch.bool),
             "origin": torch.zeros(2, 2)},
        )

    def test_translation_invariant_when_relative_trajectory_matches(self):
        import torch

        # 두 actor 모두 동일한 [5, 4]m absolute error를 갖지만 상대 벡터는 GT와 같다.
        d, w = self._data(
            [[5.0, 4.0]], [[13.0, 4.0]], [[0.0, 0.0]], [[8.0, 0.0]]
        )
        got = self.tpm.pair_relative_loss(
            self._model(), d, w, torch.tensor([[0, 1]]), near_m=10.0
        )
        self.assertEqual(got.item(), 0.0)

    def test_positive_when_relative_geometry_is_wrong(self):
        import torch

        # GT는 8m, 예측은 3m separation이다.
        d, w = self._data(
            [[0.0, 0.0]], [[3.0, 0.0]], [[0.0, 0.0]], [[8.0, 0.0]]
        )
        got = self.tpm.pair_relative_loss(
            self._model(), d, w, torch.tensor([[0, 1]]), near_m=10.0
        )
        self.assertGreater(got.item(), 0.0)

    def test_far_gt_timesteps_are_ignored(self):
        import torch

        # t=2의 prediction은 크게 틀렸지만 GT 30m는 near_m 밖이다.
        d, w = self._data(
            [[0.0, 0.0], [0.0, 0.0]],
            [[8.0, 0.0], [5.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[8.0, 0.0], [30.0, 0.0]],
        )
        got = self.tpm.pair_relative_loss(
            self._model(), d, w, torch.tensor([[0, 1]]), near_m=10.0
        )
        self.assertEqual(got.item(), 0.0)

    def test_relative_pair_index_includes_normal_and_accident_only_same_time(self):
        def row(sid, t_s, actor, x):
            return {"scenario_id": sid, "t_s": t_s, "actor_id": actor,
                    "tgt": [[0.0, 0.0], [0.0, 0.0]],
                    "origin_enu": [x, 0.0]}

        rows = [
            row("type1_subtype1_accident/a", 1.0, "A", 0.0),
            row("type1_subtype1_accident/a", 1.0, "B", 8.0),
            row("type1_subtype1_normal/n", 1.0, "C", 0.0),
            row("type1_subtype1_normal/n", 1.0, "D", 8.0),
            row("type1_subtype1_normal/n", 2.0, "E", 0.0),
            row("type1_subtype1_normal/other", 1.0, "F", 8.0),
        ]
        self.assertEqual(
            self.tpm.build_relative_pair_index(rows, 1, 10.0), [(0, 1), (2, 3)]
        )

    def test_zero_relative_weight_keeps_base_loss_exactly(self):
        import torch

        base = torch.tensor(3.0, requires_grad=True)
        relative = torch.tensor(7.0, requires_grad=True)
        got = self.tpm.combine_waypoint_losses(
            base, None, 0.0, relative, 0.0
        )
        self.assertIs(got, base)
        got.backward()
        self.assertEqual(base.grad.item(), 1.0)
        self.assertIsNone(relative.grad)


class TestJointSceneMotionNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch 없음")
        import train_scene_predict_model as stpm
        from traffic_llm.predict_nets import JointSceneMotionNet

        cls.stpm, cls.Net = stpm, JointSceneMotionNet

    def _scene(self, actors):
        return {"scenario_id": "s", "dataset_split": "train", "t_s": 0.0,
                "town": "Town01", "actors": actors}

    def _actor(self, n_cand=1, n_inter=1, target_mask=None):
        target_mask = target_mask or [True] * 5
        return {"actor_id": "x", "actor_class": "car", "global": [0.0] * 22,
                "history": [[0.0] * 5 for _ in range(5)],
                "candidates": [[1.0] + [0.0] * 11 for _ in range(n_cand)],
                "has_route_candidate": bool(n_cand),
                "interactions": [[0.0] * 28 + [1.0] for _ in range(n_inter)],
                "origin_enu": [0.0, 0.0], "target_offsets": [[0.0, 0.0]] * 5,
                "target_mask": target_mask, "future_complete": all(target_mask),
                "maneuver": "직진", "maneuver_ambiguous": False, "matched": True}

    def test_collate_preserves_scene_axes_and_masks(self):
        batch = self.stpm.scene_collate([
            self._scene([self._actor(1, 2), self._actor(0, 0)]),
            self._scene([self._actor(3, 1)]),
        ])
        self.assertEqual(tuple(batch["global"].shape), (2, 2, 22))
        self.assertEqual(tuple(batch["candidates"].shape), (2, 2, 3, 12))
        self.assertEqual(tuple(batch["interactions"].shape), (2, 2, 2, 29))
        self.assertTrue(batch["actor_mask"][0, 1])
        self.assertFalse(batch["actor_mask"][1, 1])
        self.assertFalse(batch["candidate_mask"][0, 1].any())
        self.assertFalse(batch["interaction_mask"][0, 1].any())

    def test_official_split_policy_excludes_mini_unknown_and_keeps_val_held_out(self):
        scenes = ([{"scenario_id": f"train/{i}", "dataset_split": "train"} for i in range(100)]
                  + [{"scenario_id": "val/only", "dataset_split": "val"},
                     {"scenario_id": "mini/only", "dataset_split": "DeepAccident_mini"},
                     {"scenario_id": "unknown/only", "dataset_split": "other"}])
        train, internal_val, test = self.stpm.split_scenes(scenes, "dataset", .1, .2)
        self.assertTrue(all(x["dataset_split"] == "train" for x in train + internal_val))
        self.assertEqual([x["scenario_id"] for x in test], ["val/only"])
        all_ids = [{x["scenario_id"] for x in part} for part in (train, internal_val, test)]
        self.assertFalse(all_ids[0] & all_ids[1])
        self.assertFalse(all_ids[0] & all_ids[2])
        self.assertFalse(all_ids[1] & all_ids[2])
        self.assertNotIn("mini/only", set().union(*all_ids))
        self.assertNotIn("unknown/only", set().union(*all_ids))

    def _inputs(self, A=2):
        import torch
        return (torch.randn(1, A, 22), torch.ones(1, A, 1, 12),
                torch.ones(1, A, 1, dtype=torch.bool), torch.zeros(1, A, 5, 5),
                torch.zeros(1, A, 1, 29), torch.ones(1, A, 1, dtype=torch.bool),
                torch.ones(1, A, dtype=torch.bool), torch.arange(A * 2).reshape(1, A, 2).float())

    def test_output_shape_padding_permutation_translation_and_gradients(self):
        import torch

        torch.manual_seed(7)
        model = self.Net(hidden_dim=32, scene_layers=2, num_heads=4, dropout=0.0)
        model.eval()
        inputs = self._inputs(2)
        offsets, logits = model(*inputs)
        self.assertEqual(tuple(offsets.shape), (1, 2, 5, 2))
        self.assertEqual(tuple(logits.shape), (1, 2, 5))
        # A fully masked appended actor cannot affect valid actors.
        padded = list(inputs)
        for ix in (0,):
            padded[ix] = torch.cat([padded[ix], torch.zeros(1, 1, 22)], 1)
        padded[1] = torch.cat([padded[1], torch.zeros(1, 1, 1, 12)], 1)
        padded[2] = torch.cat([padded[2], torch.zeros(1, 1, 1, dtype=torch.bool)], 1)
        padded[3] = torch.cat([padded[3], torch.zeros(1, 1, 5, 5)], 1)
        padded[4] = torch.cat([padded[4], torch.zeros(1, 1, 1, 29)], 1)
        padded[5] = torch.cat([padded[5], torch.zeros(1, 1, 1, dtype=torch.bool)], 1)
        padded[6] = torch.cat([padded[6], torch.zeros(1, 1, dtype=torch.bool)], 1)
        padded[7] = torch.cat([padded[7], torch.zeros(1, 1, 2)], 1)
        self.assertTrue(torch.allclose(offsets, model(*padded)[0][:, :2], atol=1e-6))
        perm = torch.tensor([1, 0])
        permuted = [x[:, perm] for x in inputs]
        self.assertTrue(torch.allclose(offsets[:, perm], model(*permuted)[0], atol=1e-6))
        translated = list(inputs); translated[7] = translated[7] + torch.tensor([100.0, -50.0])
        self.assertTrue(torch.allclose(offsets, model(*translated)[0], atol=1e-5))
        model.train()
        loss = model(*inputs)[0].square().mean(); loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_loss_masks_invalid_targets_and_padded_actors(self):
        import torch

        model = self.Net(hidden_dim=32, scene_layers=1, num_heads=4, dropout=0.0).eval()
        batch = self.stpm.scene_collate([self._scene([
            self._actor(target_mask=[True, False, False, False, False]), self._actor()
        ])])
        batch["actor_mask"][0, 1] = False
        loss_a, _, _ = self.stpm.scene_loss(model, batch)
        changed = dict(batch); changed["target"] = batch["target"].clone()
        changed["target"][0, 0, 1:] = 999.0
        changed["target"][0, 1] = -999.0
        loss_b, _, _ = self.stpm.scene_loss(model, changed)
        self.assertAlmostEqual(loss_a.item(), loss_b.item(), places=6)

    def test_all_ignored_maneuvers_are_finite(self):
        import torch

        model = self.Net(hidden_dim=32, scene_layers=1, num_heads=4, dropout=0.0)
        batch = self.stpm.scene_collate([self._scene([self._actor(0, 0)])])
        self.assertTrue((batch["maneuver"] == -100).all())
        loss, _, _ = self.stpm.scene_loss(model, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                            if p.grad is not None))

    def test_evaluate_averages_actor_ade_not_future_points(self):
        import torch

        class Fixed(torch.nn.Module):
            def forward(self, g, *unused):
                return g[..., :10].reshape(g.shape[0], g.shape[1], 5, 2)

        batch = self.stpm.scene_collate([self._scene([
            self._actor(target_mask=[True, False, False, False, False]),
            self._actor(target_mask=[True, True, True, False, False]),
        ])])
        batch["global"][0, 0, :10] = torch.tensor([1.0, 0.0] * 5)
        batch["global"][0, 1, :10] = torch.tensor([3.0, 0.0] * 5)
        got = self.stpm.evaluate(Fixed(), [batch], "cpu")
        self.assertAlmostEqual(got["ade"], 2.0, places=6)
        self.assertAlmostEqual(got["fde"], 2.0, places=6)
        self.assertEqual(got["trajectories"], 2)
        self.assertEqual(got["points"], 4)

    def test_empty_and_padded_interactions_are_invariant(self):
        import torch

        torch.manual_seed(3)
        model = self.Net(hidden_dim=32, scene_layers=1, num_heads=4, dropout=0.0).eval()
        base = self._inputs(1)
        empty = list(base); empty[4] = torch.zeros(1, 1, 1, 29); empty[5] = torch.zeros(1, 1, 1, dtype=torch.bool)
        padded = list(empty); padded[4] = torch.randn(1, 1, 4, 29); padded[5] = torch.zeros(1, 1, 4, dtype=torch.bool)
        self.assertTrue(torch.allclose(model(*empty)[0], model(*padded)[0], atol=1e-6))
        one = list(empty); one[4][:, :, 0, -1] = 1.0; one[5][:, :, 0] = True
        one_padded = list(one); one_padded[4] = torch.cat([one[4], torch.randn(1, 1, 3, 29)], 2)
        one_padded[5] = torch.cat([one[5], torch.zeros(1, 1, 3, dtype=torch.bool)], 2)
        self.assertTrue(torch.allclose(model(*one)[0], model(*one_padded)[0], atol=1e-6))

    def test_initialization_is_constant_velocity_rollout(self):
        import torch

        model = self.Net(hidden_dim=32, scene_layers=1, num_heads=4, dropout=0.0).eval()
        x = list(self._inputs(2))
        x[0].zero_()
        # ENU heading sin/cos are global indices 4/5; use 2m/s north and 3m/s east.
        x[0][0, 0, 0], x[0][0, 0, 5] = 2.0, 1.0
        x[0][0, 1, 0], x[0][0, 1, 4] = 3.0, 1.0
        out = model(*x)[0]
        want = torch.tensor([[[[0.0, 2.0], [0.0, 4.0], [0.0, 6.0], [0.0, 8.0], [0.0, 10.0]],
                              [[3.0, 0.0], [6.0, 0.0], [9.0, 0.0], [12.0, 0.0], [15.0, 0.0]]]])
        self.assertTrue(torch.allclose(out, want, atol=1e-6))


class TestJointSceneRuntimePredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch 없음")

    @staticmethod
    def _contexts(with_neighbors=True):
        from traffic_llm.predict_model import Candidate, PredictContext
        from traffic_llm.schemas import ActorState

        actors = [
            ActorState("route", "ego", "car", (0.0, 0.0), 0.0, 4.0, 0.0),
            ActorState("no_route", "observed", "truck", (5.0, 0.0), 0.0, 0.0, 0.0),
            ActorState("third", "observed", "bicycle", (10.0, 0.0), 0.0, 2.0, 0.0),
        ]
        candidate = Candidate("직진", 1.0, [(0.0, 0.0), (0.0, 20.0)], "R")
        return [
            PredictContext(actor=a, horizon_s=5.0,
                           candidates=[candidate] if a.actor_id == "route" else [],
                           neighbors=actors if with_neighbors else [])
            for a in actors
        ]

    def _predictor(self):
        import torch
        from traffic_llm.predict_model import JointSceneTorchPredictor

        class CountingModel(torch.nn.Module):
            accepts_scene = True

            def __init__(self):
                super().__init__()
                self.calls = 0
                self.last_shapes = None

            def forward(self, g, c, cm, h, i, im, am, origin):
                self.calls += 1
                self.last_shapes = tuple(x.shape for x in (g, c, cm, h, i, im, am, origin))
                b, a = g.shape[:2]
                offsets = torch.zeros(b, a, 5, 2, device=g.device)
                logits = torch.zeros(b, a, 5, device=g.device)
                return offsets, logits

        predictor = JointSceneTorchPredictor("unused.pt")
        predictor._torch = torch
        predictor._model = CountingModel()
        return predictor

    def test_one_scene_forward_maps_all_ids_and_keeps_no_route_actor(self):
        predictor = self._predictor()
        contexts = self._contexts()
        paths = predictor.predict_scene(contexts)
        self.assertEqual(predictor.forward_calls, 1)
        self.assertEqual(predictor._model.calls, 1)
        self.assertEqual(set(paths), {"route", "no_route", "third"})
        self.assertEqual(len(paths), 3, "runtime padding must not create outputs")
        self.assertTrue(all(len(path.waypoints) == 6 for path in paths.values()))
        self.assertTrue(all(all(math.isfinite(v) for point in path.waypoints for v in point)
                            for path in paths.values()))
        shapes = predictor._model.last_shapes
        self.assertEqual(tuple(shapes[0]), (1, 3, 22))
        self.assertEqual(tuple(shapes[1]), (1, 3, 1, 12))
        self.assertEqual(tuple(shapes[3]), (1, 3, 5, 5))
        self.assertEqual(tuple(shapes[4]), (1, 3, 2, 29))
        self.assertEqual(tuple(shapes[6]), (1, 3))
        with self.assertRaisesRegex(RuntimeError, "predict_scene"):
            predictor(contexts[0])

    def test_zero_interaction_scene_needs_no_future_gt(self):
        predictor = self._predictor()
        paths = predictor.predict_scene(self._contexts(with_neighbors=False)[:1])
        self.assertEqual(predictor.forward_calls, 1)
        self.assertEqual(set(paths), {"route"})
        self.assertEqual(tuple(predictor._model.last_shapes[4]), (1, 1, 1, 29))
        self.assertEqual(tuple(predictor._model.last_shapes[5]), (1, 1, 1))

    def test_selected_joint_scene_checkpoint_loads(self):
        from traffic_llm.predict_model import JointSceneTorchPredictor

        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "out/predict_model_joint_scene_v1/joint_scene_motionnet_best.pt",
        )
        self.assertTrue(os.path.isfile(path), path)
        predictor = JointSceneTorchPredictor(path)
        self.assertTrue(getattr(predictor._load(), "accepts_scene", False))

    def test_v2_uses_the_same_single_scene_runtime_dispatch(self):
        from traffic_llm.predict_model import JointSceneTorchPredictor
        from traffic_llm.predict_nets import JointSceneMotionNetV2

        model = JointSceneMotionNetV2(hidden_dim=32, scene_layers=1, num_heads=4,
                                      dropout=0.0).eval()
        original_forward = model.forward
        calls = [0]

        def counted_forward(*args, **kwargs):
            calls[0] += 1
            return original_forward(*args, **kwargs)

        model.forward = counted_forward
        predictor = JointSceneTorchPredictor("unused-v2.pt")
        import torch
        predictor._torch, predictor._model = torch, model
        paths = predictor.predict_scene(self._contexts())
        self.assertEqual(calls[0], 1)
        self.assertEqual(predictor.forward_calls, 1)
        self.assertEqual(set(paths), {"route", "no_route", "third"})


class TestJointSceneMotionNetV2(TestJointSceneMotionNet):
    """V2-only checks; inherited V1 tests keep the baseline path exercised."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch 없음")
        import train_scene_predict_model as stpm
        from traffic_llm.predict_nets import JointSceneMotionNetV2

        cls.stpm, cls.Net = stpm, JointSceneMotionNetV2

    def _v2(self):
        return self.Net(hidden_dim=32, scene_layers=2, num_heads=4, dropout=0.0)

    def test_zero_velocity_update_prior_is_persistent_constant_velocity(self):
        import torch

        model = self._v2().eval()
        x = list(self._inputs(2))
        x[0].zero_()
        # Existing ENU convention: [speed, ..., heading_sin, heading_cos].
        x[0][0, 0, 0], x[0][0, 0, 5] = 2.0, 1.0
        x[0][0, 1, 0], x[0][0, 1, 4] = 3.0, 1.0
        got = model(*x)[0]
        want = torch.tensor([[[[0., 2.], [0., 4.], [0., 6.], [0., 8.], [0., 10.]],
                              [[3., 0.], [6., 0.], [9., 0.], [12., 0.], [15., 0.]]]])
        self.assertTrue(torch.allclose(got, want, atol=1e-6))

    def test_velocity_state_does_not_reinject_original_cv_after_stopping(self):
        import torch

        updates = torch.tensor([[[[0., -2.], [0., -2.], [0., 0.], [0., 0.], [0., 0.]]]])
        got = self.Net.integrate_velocity_updates(
            torch.zeros(1, 1, 2), torch.tensor([[[0., 4.]]]), updates,
            torch.ones(1, 1, dtype=torch.bool))
        # v: 2, 0, 0, 0, 0.  Fixed-CV V1 would resume moving after each step.
        want = torch.tensor([[[[0., 2.], [0., 2.], [0., 2.], [0., 2.], [0., 2.]]]])
        self.assertTrue(torch.equal(got, want))

    def _edge_inputs(self):
        import torch

        positions = torch.tensor([[[0., 0.], [0., 8.], [6., 0.]]])
        velocities = torch.tensor([[[0., 2.], [0., -1.], [1., 0.]]])
        route = torch.tensor([[True, False, True]])
        global_features = torch.zeros(1, 3, 22)
        global_features[..., 4] = torch.tensor([[0., 0., 1.]])
        global_features[..., 5] = torch.tensor([[1., -1., 0.]])
        global_features[0, 0, 11] = 1.
        global_features[0, 1, 12] = 1.
        global_features[0, 2, 13] = 1.
        return positions, velocities, route, global_features

    def test_explicit_edges_keep_neighbor_identity_and_closing_sign(self):
        import torch
        p, v, route, g = self._edge_inputs()
        edge = self.Net.edge_features(p, v, route, g)
        self.assertFalse(torch.allclose(edge[0, 0, 1], edge[0, 0, 2]))
        # Actor 0 and actor 1 approach: j moves toward i along the radial line.
        self.assertGreater(edge[0, 0, 1, self.Net.EDGE_CLOSING_SPEED].item(), 0.)
        # Actor 0 and actor 2 have a different (separating) radial motion.
        self.assertLess(edge[0, 0, 2, self.Net.EDGE_CLOSING_SPEED].item(), 0.)

    def test_dynamic_edges_and_no_route_flags(self):
        import torch

        p, v, route, g = self._edge_inputs()
        before = self.Net.edge_features(p, v, route, g)
        after = self.Net.edge_features(p + v, v, route, g)
        self.assertFalse(torch.allclose(before[..., :6], after[..., :6]))
        self.assertEqual(before[0, 0, 1, self.Net.EDGE_HAS_ROUTE_I].item(), 1.)
        self.assertEqual(before[0, 0, 1, self.Net.EDGE_HAS_ROUTE_J].item(), 0.)
        self.assertEqual(before[0, 1, 2, self.Net.EDGE_HAS_ROUTE_I].item(), 0.)
        self.assertEqual(before[0, 1, 2, self.Net.EDGE_HAS_ROUTE_J].item(), 1.)
        self.assertEqual(before[0, 1, 1, self.Net.EDGE_HAS_ROUTE_I].item(), 0.)
        self.assertEqual(before[0, 1, 1, self.Net.EDGE_HAS_ROUTE_J].item(), 0.)

    def test_padded_actor_cannot_send_or_receive_and_mixed_scene_is_finite(self):
        import torch

        model = self._v2().eval()
        base = list(self._inputs(3))
        base[2][0, 1] = False  # no-route actor
        base[1][0, 1].zero_()
        base[5][0, 2] = False  # zero-interaction actor
        base[4][0, 2].zero_()
        got = model(*base)[0]
        padded = list(base)
        for ix, shape in ((0, (1, 1, 22)), (1, (1, 1, 1, 12)), (2, (1, 1, 1)),
                          (3, (1, 1, 5, 5)), (4, (1, 1, 1, 29)),
                          (5, (1, 1, 1)), (6, (1, 1)), (7, (1, 1, 2))):
            value = torch.zeros(shape, dtype=(torch.bool if ix in (2, 5, 6) else torch.float32))
            padded[ix] = torch.cat([padded[ix], value], dim=1)
        self.assertTrue(torch.isfinite(got).all())
        self.assertTrue(torch.allclose(got, model(*padded)[0][:, :3], atol=1e-6))
        model.train()
        loss = model(*base)[0].square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None))

    def test_edge_normalization_defaults_and_applies_saved_scale(self):
        import torch

        model = self._v2()
        self.assertTrue(torch.equal(model.e_mu, torch.zeros(model.EDGE_DIM)))
        self.assertTrue(torch.equal(model.e_sd, torch.ones(model.EDGE_DIM)))
        model.e_mu.copy_(torch.arange(model.EDGE_DIM, dtype=torch.float32))
        model.e_sd.copy_(torch.full((model.EDGE_DIM,), 2.0))
        raw = torch.arange(model.EDGE_DIM, dtype=torch.float32).reshape(1, 1, 1, -1) + 4
        self.assertTrue(torch.equal(model.normalize_edges(raw), torch.full_like(raw, 2.0)))

    def test_edge_fit_uses_only_valid_directed_nonself_observation_edges(self):
        import torch

        model = self._v2()
        batch = {
            "global": torch.zeros(1, 3, 22),
            "candidates": torch.zeros(1, 3, 1, 12),
            "candidate_mask": torch.tensor([[[True], [False], [True]]]),
            "history": torch.zeros(1, 3, 5, 5),
            "interactions": torch.zeros(1, 3, 1, 29),
            "interaction_mask": torch.zeros(1, 3, 1, dtype=torch.bool),
            "actor_mask": torch.tensor([[True, True, False]]),
            "origin": torch.tensor([[[0., 0.], [4., 0.], [999., 999.]]]),
            "scenes": [],
        }
        # The padded actor has deliberately extreme state values.  It cannot
        # affect the fitted statistic because its directed edges are excluded.
        batch["global"][0, 2, 0] = 999.
        self.stpm.fit_normalizer(model, [batch], "cpu")
        velocity = model.initial_velocity(batch["global"], batch["actor_mask"])
        edges = model.edge_features(batch["origin"], velocity,
                                    batch["candidate_mask"].any(-1), batch["global"])
        valid = model.edge_valid_mask(batch["actor_mask"])
        expected = edges[valid]
        mean = expected.mean(0)
        sd = expected.std(0, unbiased=False).clamp(min=1e-3)
        self.assertTrue(torch.allclose(model.e_mu, mean))
        self.assertTrue(torch.allclose(model.e_sd, sd))

    def test_edge_normalization_persists_and_v1_has_no_edge_buffers(self):
        import torch
        from traffic_llm.predict_nets import JointSceneMotionNet

        model = self._v2()
        model.e_mu.copy_(torch.arange(model.EDGE_DIM, dtype=torch.float32))
        model.e_sd.copy_(torch.arange(model.EDGE_DIM, dtype=torch.float32) + 1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "v2.pt")
            torch.save(model, path)
            loaded = torch.load(path, weights_only=False)
        self.assertTrue(torch.equal(loaded.e_mu, model.e_mu))
        self.assertTrue(torch.equal(loaded.e_sd, model.e_sd))
        self.assertFalse(hasattr(JointSceneMotionNet(hidden_dim=32), "e_mu"))


class TestSceneGradientAccumulation(unittest.TestCase):
    """Microbatch accumulation is tested independently from scene_loss."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch 없음")
        import train_scene_predict_model as stpm
        cls.stpm = stpm

    @staticmethod
    def _model():
        import torch
        return torch.nn.Linear(1, 1, bias=False)

    @staticmethod
    def _mse_loss(model, batch):
        import torch
        return torch.nn.functional.mse_loss(model(batch["x"]), batch["y"]), None, None

    @staticmethod
    def _counting_sgd(params, lr):
        import torch

        class CountingSGD(torch.optim.SGD):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.steps, self.grads = 0, []

            def step(self, closure=None):
                self.steps += 1
                self.grads.append(next(iter(self.param_groups[0]["params"])).grad.detach().clone())
                return super().step(closure)

        return CountingSGD(params, lr=lr)

    def test_accumulation_one_preserves_one_step_per_microbatch(self):
        import torch

        model = self._model()
        optimizer = self._counting_sgd(model.parameters(), .1)
        loader = [{"x": torch.ones(1, 1), "y": torch.zeros(1, 1)} for _ in range(3)]
        got = self.stpm.train_one_epoch(model, loader, optimizer, "cpu",
                                        grad_accum_steps=1, loss_fn=self._mse_loss)
        self.assertEqual(optimizer.steps, 3)
        self.assertEqual(got["optimizer_steps"], 3)

    def test_eight_microbatches_of_eight_accumulated_by_four_make_two_updates(self):
        import torch

        model = self._model()
        optimizer = self._counting_sgd(model.parameters(), .1)
        # Eight physical batches * eight scenes; each update covers four
        # physical batches, i.e. 32 scenes.
        loader = [{"x": torch.ones(8, 1), "y": torch.zeros(8, 1)} for _ in range(8)]
        got = self.stpm.train_one_epoch(model, loader, optimizer, "cpu",
                                        grad_accum_steps=4, loss_fn=self._mse_loss)
        self.assertEqual(optimizer.steps, 2)
        self.assertEqual(got["optimizer_steps"], 2)

    def test_final_partial_window_steps_and_uses_its_actual_divisor(self):
        import torch

        model = self._model()
        with torch.no_grad():
            model.weight.zero_()
        optimizer = self._counting_sgd(model.parameters(), 1.0)

        def linear_loss(m, batch):
            return (m.weight * batch["value"]).sum(), None, None

        loader = [{"value": torch.tensor([[1.]])} for _ in range(4)]
        loader.append({"value": torch.tensor([[10.]])})
        got = self.stpm.train_one_epoch(model, loader, optimizer, "cpu",
                                        grad_accum_steps=4, loss_fn=linear_loss)
        self.assertEqual(optimizer.steps, 2)
        self.assertEqual(got["optimizer_steps"], 2)
        self.assertAlmostEqual(optimizer.grads[0].item(), 1.0)
        # The one-item final window is divided by one, not by four.
        self.assertAlmostEqual(optimizer.grads[1].item(), 10.0)

    def test_optimization_loss_is_mean_of_unscaled_microbatch_losses(self):
        import torch

        model = self._model()
        optimizer = self._counting_sgd(model.parameters(), 0.0)

        def fixed_loss(m, batch):
            return m.weight.sum() * 0 + batch["loss"], None, None

        loader = [{"loss": torch.tensor(v)} for v in (1.0, 3.0, 8.0)]
        got = self.stpm.train_one_epoch(model, loader, optimizer, "cpu",
                                        grad_accum_steps=2, loss_fn=fixed_loss)
        self.assertAlmostEqual(got["optimization_loss"], 4.0)

    def test_deterministic_microbatch_update_is_close_to_one_batch_update(self):
        import torch

        # This fixture has equal valid counts, so it is nearly identical. In
        # real scene_loss runs exact equality is not guaranteed: it averages
        # each microbatch independently over variable valid targets/maneuvers.
        torch.manual_seed(4)
        full = self._model()
        micro = self._model()
        micro.load_state_dict(full.state_dict())
        x = torch.tensor([[1.], [2.], [3.], [4.]])
        y = torch.tensor([[2.], [1.], [0.], [-1.]])
        full_opt = torch.optim.SGD(full.parameters(), lr=.05)
        micro_opt = torch.optim.SGD(micro.parameters(), lr=.05)
        self.stpm.train_one_epoch(full, [{"x": x, "y": y}], full_opt, "cpu",
                                  grad_accum_steps=1, loss_fn=self._mse_loss)
        self.stpm.train_one_epoch(micro, [{"x": x[i:i + 1], "y": y[i:i + 1]}
                                          for i in range(4)], micro_opt, "cpu",
                                  grad_accum_steps=4, loss_fn=self._mse_loss)
        self.assertTrue(torch.allclose(full.weight, micro.weight, atol=1e-6, rtol=1e-5))


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
