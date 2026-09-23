"""사고 예측 질의 생성(슬라이딩 윈도우 + 정답) 테스트."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_llm.accident_qa import (
    PREDICTION_SCHEMA,
    WindowConfig,
    aggregate_scores,
    build_actor_lookup,
    build_question,
    build_window_payload,
    build_windows,
    closing_pairs,
    render_window_text,
    score_response,
    window_ground_truth,
    window_json,
    write_window_set,
)
from dataclasses import replace

from traffic_llm import i18n
from traffic_llm.config import SerializeConfig
from traffic_llm.schemas import (
    ActorState,
    RoadPlacement,
    ScenarioContext,
    SceneSnapshot,
)

DA_ROOT = os.environ.get(
    "DEEPACCIDENT_ROOT", r"C:/Users/ylim/Downloads/DeepAccident_mini"
)
HAS_DATA = os.path.isdir(DA_ROOT)


def placement(off: float = -1.6, lane: int = 1, s: float = 0.0) -> RoadPlacement:
    return RoadPlacement(
        road_id="r1", road_name="도로1", s_m=s, lateral_offset_m=off,
        direction_label="북행", bearing_deg=0.0, lane_index=lane, lane_count=2,
        speed_limit_kph=50.0, dist_to_next_junction_m=100.0 - s,
        next_junction_id="J0",
    )


def make_snapshots(
    n: int = 21, dt: float = 0.5, scenario: bool = True
) -> list:
    """북행 자차 + 접근하는 대향차 시퀀스."""
    out = []
    ctx = (
        ScenarioContext(
            scenario_id="type1_subtype1_accident/TownX_scenario1",
            source="TestSource",
            town="TownX",
            available={"weather": "ClearNoon", "road_type": "four-way junction"},
            ground_truth={"collision_occurred": True},
        )
        if scenario
        else None
    )
    for i in range(n):
        t = i * dt
        ego = ActorState(
            actor_id="EGO_v1", kind="ego", cls="car",
            world_xy=(0.0, -50.0 + 10.0 * t), heading_deg=0.0,
            speed_mps=10.0, accel_mps2=0.0, placement=placement(s=10.0 * t),
            observed_by=["v1"], source_track_ids=[],
        )
        opp = ActorState(
            actor_id="V002", kind="observed", cls="car",
            world_xy=(3.3, 60.0 - 12.0 * t), heading_deg=180.0,
            speed_mps=12.0, accel_mps2=0.0, placement=placement(off=1.6, s=5.0),
            observed_by=["v1"], confidence=0.9, position_quality=0.8,
            observed_range_m=abs(110.0 - 22.0 * t), source_track_ids=[7001],
        )
        out.append(
            SceneSnapshot(
                t=t, actors=[ego, opp], interactions=[],
                area_name="TownX", map_context={"road_count": 2},
                scenario=ctx, frame_idx=i + 1,
            )
        )
    return out


class TestWindowing(unittest.TestCase):
    def test_sliding_windows_labels(self):
        snaps = make_snapshots(n=21, dt=0.5)  # 0 ~ 10초
        cfg = WindowConfig(window_s=5.0, stride_s=1.0, horizon_s=5.0)
        wins, info = build_windows(snaps, cfg)
        labels = [w.label for w in wins]
        self.assertEqual(labels[:4], ["0-5", "1-6", "2-7", "3-8"])
        self.assertEqual(labels[-1], "5-10")
        for w in wins:
            self.assertAlmostEqual(w.t_end - w.t_start, 5.0, places=6)
            self.assertAlmostEqual(w.snapshots[0].t, w.t_start, places=6)
            self.assertAlmostEqual(w.snapshots[-1].t, w.t_end, places=6)
        self.assertEqual(info["n_windows"], len(wins))

    def test_stride_and_window_length_respected(self):
        snaps = make_snapshots(n=21, dt=0.5)
        wins, _ = build_windows(
            snaps, WindowConfig(window_s=3.0, stride_s=2.0)
        )
        self.assertEqual([w.label for w in wins], ["0-3", "2-5", "4-7", "6-9"])

    def test_incomplete_window_dropped(self):
        snaps = make_snapshots(n=7, dt=0.5)  # 0 ~ 3초
        wins, info = build_windows(snaps, WindowConfig(window_s=5.0))
        self.assertEqual(wins, [])
        self.assertEqual(info["n_windows"], 0)

    def test_windows_after_collision_dropped(self):
        """충돌이 관측 구간 안에 들면 예측 문제가 아니므로 버린다."""
        snaps = make_snapshots(n=21, dt=0.5)
        cfg = WindowConfig(window_s=5.0, stride_s=1.0)
        wins, info = build_windows(snaps, cfg, collision_time_s=7.0)
        self.assertTrue(wins)
        for w in wins:
            self.assertLess(w.t_end, 7.0)
        self.assertGreater(info["dropped_after_collision"], 0)

    def test_keep_windows_after_collision_when_disabled(self):
        snaps = make_snapshots(n=21, dt=0.5)
        cfg = WindowConfig(window_s=5.0, stride_s=1.0,
                           drop_windows_after_collision=False)
        wins, _ = build_windows(snaps, cfg, collision_time_s=7.0)
        self.assertTrue(any(w.t_end >= 7.0 for w in wins))

    def test_empty_input(self):
        wins, info = build_windows([], WindowConfig())
        self.assertEqual(wins, [])
        self.assertEqual(info["n_snapshots"], 0)


class TestClosingPairs(unittest.TestCase):
    def test_approaching_pair_detected(self):
        snaps = make_snapshots(n=11, dt=0.5)
        wins, _ = build_windows(snaps, WindowConfig(window_s=5.0, stride_s=1.0))
        pairs = closing_pairs(wins[0], WindowConfig())
        self.assertTrue(pairs, "접근 중인 쌍을 찾지 못했다")
        p = pairs[0]
        self.assertEqual({p.a, p.b}, {"EGO_v1", "V002"})
        self.assertLess(p.d_end, p.d_start)
        # 자차 10m/s 북행 + 대향 12m/s 남행 → 접근율 약 22m/s
        self.assertAlmostEqual(p.closing_rate_mps, 22.0, delta=1.0)
        self.assertIsNotNone(p.linear_contact_s)

    def test_diverging_pair_not_reported(self):
        snaps = make_snapshots(n=11, dt=0.5)
        # 시간을 뒤집어 서로 멀어지게
        for s in snaps:
            for a in s.actors:
                if a.actor_id == "V002":
                    a.world_xy = (3.3, 200.0 + 12.0 * s.t)
        wins, _ = build_windows(snaps, WindowConfig(window_s=5.0))
        self.assertEqual(closing_pairs(wins[0], WindowConfig()), [])


class TestRendering(unittest.TestCase):
    def setUp(self):
        self.snaps = make_snapshots(n=21, dt=0.5)
        self.cfg = WindowConfig(window_s=5.0, stride_s=1.0, horizon_s=5.0)
        self.wins, _ = build_windows(self.snaps, self.cfg)
        self.scfg = SerializeConfig()

    def test_text_has_window_history_and_detail(self):
        txt = render_window_text(self.wins[2], self.scfg, self.cfg)
        self.assertIn("관측 윈도우 2-7", txt)
        self.assertIn("차량별 시간 경과", txt)
        self.assertIn("서로 접근한 차량 쌍", txt)
        self.assertIn("마지막 관측 시점 상세", txt)
        self.assertIn("EGO_v1", txt)
        self.assertIn("V002", txt)
        # 이력이 여러 시점 담겨 있어야 한다
        self.assertGreaterEqual(txt.count("t="), 6)

    def test_history_stride_reduces_rows(self):
        dense = WindowConfig(window_s=5.0, history_stride_s=0.5)
        sparse = WindowConfig(window_s=5.0, history_stride_s=2.0)
        a = render_window_text(self.wins[0], self.scfg, dense)
        b = render_window_text(self.wins[0], self.scfg, sparse)
        self.assertGreater(a.count("t="), b.count("t="))

    def test_json_compact_has_trajectories(self):
        d = window_json(self.wins[0], self.scfg, self.cfg)
        self.assertIn("trajectories", d)
        self.assertIn("last_snapshot", d)
        self.assertIn("closing_pairs", d)
        self.assertNotIn("snapshots", d)
        self.assertIn("EGO_v1", d["trajectories"])

    def test_json_full_has_all_snapshots(self):
        cfg = WindowConfig(window_s=5.0, history_mode="full")
        d = window_json(self.wins[0], self.scfg, cfg)
        self.assertIn("snapshots", d)
        self.assertEqual(len(d["snapshots"]), len(self.wins[0].snapshots))

    def test_scenario_id_redacted_in_payload(self):
        """분할명('..._accident')이 payload 에 노출되면 정답 누수다.

        JSON 블록을 켜든 끄든 지켜져야 한다. 시나리오 참조 토큰(`scn_`)은
        JSON 블록에만 있으므로, 끈 상태에서는 없는 것이 정상이다 — payload 와
        정답을 짝짓는 것은 파일명과 manifest 다.
        """
        for on in (True, False):
            cfg = replace(self.cfg, include_json_block=on)
            blob = json.dumps(
                build_window_payload(self.wins[0], self.scfg, cfg),
                ensure_ascii=False,
            )
            self.assertNotIn("type1_subtype1_accident", blob)
            self.assertNotIn("_normal", blob)
            self.assertEqual("scn_" in blob, on)

    def test_no_ground_truth_in_payload(self):
        payload = build_window_payload(self.wins[0], self.scfg, self.cfg)
        blob = json.dumps(payload, ensure_ascii=False)
        for leak in ("collision_occurred", "ground_truth", "colliding"):
            self.assertNotIn(leak, blob, f"정답 누수: {leak}")


class TestQuestionAndSchema(unittest.TestCase):
    def setUp(self):
        self.snaps = make_snapshots(n=21, dt=0.5)
        self.cfg = WindowConfig(window_s=5.0, stride_s=1.0, horizon_s=5.0)
        self.wins, _ = build_windows(self.snaps, self.cfg)

    def test_question_lists_all_buckets(self):
        q = build_question(self.wins[0], self.cfg)  # t_end=5
        for k, iv in ((1, "(5, 6]"), (2, "(6, 7]"), (5, "(9, 10]")):
            self.assertIn(f"k={k}", q)
            self.assertIn(iv, q)
        self.assertIn("사고가 예상되지 않으면", q)
        self.assertIn("involved_actor_ids", q)

    def test_horizon_bucket_count(self):
        cfg = WindowConfig(window_s=5.0, horizon_s=3.0)
        self.assertEqual(cfg.n_horizon_buckets, 3)
        q = build_question(self.wins[0], cfg)
        self.assertIn("k=3", q)
        self.assertNotIn("k=4", q)

    def test_payload_shape(self):
        p = build_window_payload(self.wins[0], SerializeConfig(), self.cfg)
        self.assertEqual(p["model"], "claude-opus-5")
        self.assertEqual(p["thinking"], {"type": "adaptive"})
        self.assertNotIn("temperature", p)
        self.assertNotIn("budget_tokens", p.get("thinking", {}))
        self.assertEqual(p["output_config"]["effort"], "high")
        fmt = p["output_config"]["format"]
        self.assertEqual(fmt["type"], "json_schema")
        # 스키마는 질문의 구간 수를 설명문에 담으므로 그것까지 넣고 비교한다.
        # (구간 수를 스키마 **제약**으로 못박지는 않는다 — OpenAI strict 모드가
        #  minItems/maximum 을 지원하지 않아 provider 별로 스키마가 갈린다)
        self.assertEqual(
            fmt["schema"],
            i18n.prediction_schema("ko", self.cfg.n_horizon_buckets),
        )
        self.assertIn(
            str(self.cfg.n_horizon_buckets),
            fmt["schema"]["properties"]["predictions"]["description"],
        )
        self.assertEqual(
            p["system"][0]["cache_control"]["type"], "ephemeral"
        )
        self.assertEqual(p["messages"][-1]["role"], "user")

    def test_schema_is_strict_compatible(self):
        """구조화 출력 스키마 제약: additionalProperties=false, required 명시."""

        def check(node):
            if not isinstance(node, dict):
                return
            if node.get("type") == "object":
                self.assertIn("additionalProperties", node)
                self.assertFalse(node["additionalProperties"])
                self.assertIn("required", node)
                self.assertEqual(
                    sorted(node["required"]),
                    sorted(node.get("properties", {})),
                    "required 가 properties 전체를 담아야 한다",
                )
            for key in ("properties", "items"):
                v = node.get(key)
                if isinstance(v, dict):
                    if key == "items":
                        check(v)
                    else:
                        for sub in v.values():
                            check(sub)

        check(PREDICTION_SCHEMA)
        # 지원되지 않는 제약이 없어야 한다
        blob = json.dumps(PREDICTION_SCHEMA)
        for bad in ("minimum", "maximum", "minLength", "maxLength", "$ref"):
            self.assertNotIn(bad, blob)


class TestGroundTruth(unittest.TestCase):
    class FakeCollision:
        def __init__(self, occurred=True, time_s=7.4, ids=(7001, 7002)):
            self.occurred = occurred
            self.time_s = time_s
            self.carla_ids = ids

        def to_dict(self):
            return {
                "occurred": self.occurred,
                "time_s": self.time_s,
                "carla_ids": list(self.carla_ids),
            }

    def setUp(self):
        self.snaps = make_snapshots(n=21, dt=0.5)
        self.cfg = WindowConfig(window_s=5.0, stride_s=1.0, horizon_s=5.0)
        self.wins, _ = build_windows(self.snaps, self.cfg)

    def test_bucket_assignment(self):
        """충돌 7.4초, t_end=5 → (7,8] 구간인 k=3 만 True."""
        win = next(w for w in self.wins if abs(w.t_end - 5.0) < 1e-6)
        gt = window_ground_truth(
            win, self.cfg, collision=self.FakeCollision(time_s=7.4),
            agent_carla_ids={"v1": 9001}, scenario_id="s/x",
        )
        flags = {e["k"]: e["accident_expected"] for e in gt["expected"]}
        self.assertEqual(flags, {1: False, 2: False, 3: True, 4: False, 5: False})
        self.assertAlmostEqual(gt["time_to_collision_from_window_end_s"], 2.4)

    def test_bucket_boundary_inclusive_upper(self):
        """구간은 (k-1, k] — 정확히 k 초면 그 구간에 속한다."""
        win = next(w for w in self.wins if abs(w.t_end - 5.0) < 1e-6)
        gt = window_ground_truth(
            win, self.cfg, collision=self.FakeCollision(time_s=6.0)
        )
        flags = {e["k"]: e["accident_expected"] for e in gt["expected"]}
        self.assertTrue(flags[1], "t_end+1.0 은 k=1 구간에 포함되어야 한다")
        self.assertFalse(flags[2])

    def test_collision_beyond_horizon_all_false(self):
        win = self.wins[0]
        gt = window_ground_truth(
            win, self.cfg, collision=self.FakeCollision(time_s=99.0)
        )
        self.assertFalse(any(e["accident_expected"] for e in gt["expected"]))
        self.assertTrue(any("예측 구간" in n for n in gt["notes"]))

    def test_no_collision_all_false(self):
        gt = window_ground_truth(self.wins[0], self.cfg, collision=None)
        self.assertFalse(any(e["accident_expected"] for e in gt["expected"]))
        self.assertFalse(gt["collision"]["occurred"])
        self.assertTrue(any("충돌 기록이 없다" in n for n in gt["notes"]))

    def test_actor_lookup_prefers_agent_mapping_for_ego(self):
        """ego 신원은 관측자↔CARLA id 대응이 정답이다."""
        win = self.wins[0]
        lookup = build_actor_lookup(win, {"v1": 9001})
        self.assertEqual(lookup[9001], ["EGO_v1"])
        self.assertEqual(lookup[7001], ["V002"])

    def test_involved_vehicles_grouped_per_carla_id(self):
        win = next(w for w in self.wins if abs(w.t_end - 5.0) < 1e-6)
        gt = window_ground_truth(
            win, self.cfg,
            collision=self.FakeCollision(time_s=7.4, ids=(7001, 9001)),
            agent_carla_ids={"v1": 9001},
        )
        hit = next(e for e in gt["expected"] if e["accident_expected"])
        by_id = {v["carla_id"]: v for v in hit["involved_vehicles"]}
        self.assertEqual(by_id[7001]["actor_ids"], ["V002"])
        self.assertEqual(by_id[9001]["actor_ids"], ["EGO_v1"])
        self.assertTrue(all(v["observed_in_window"] for v in by_id.values()))
        self.assertEqual(hit["unobserved_carla_ids"], [])

    def test_unobserved_collider_flagged(self):
        """관측되지 않은 충돌 주체는 표시되어야 한다 (지목 불가)."""
        win = next(w for w in self.wins if abs(w.t_end - 5.0) < 1e-6)
        gt = window_ground_truth(
            win, self.cfg,
            collision=self.FakeCollision(time_s=7.4, ids=(7001, 55555)),
        )
        hit = next(e for e in gt["expected"] if e["accident_expected"])
        self.assertIn(55555, hit["unobserved_carla_ids"])
        self.assertFalse(hit["all_involved_observed"])
        self.assertTrue(any("관측되지 않았다" in n for n in gt["notes"]))


class TestScoring(unittest.TestCase):
    def _gt(self, hit_k=3, ids=("V002",)):
        return {
            "window": {"label": "0-5"},
            "scenario": {"ref": "scn_x"},
            "expected": [
                {
                    "k": k,
                    "accident_expected": (k == hit_k),
                    "involved_vehicles": [
                        {"carla_id": 7001, "actor_ids": list(ids)}
                    ]
                    if k == hit_k
                    else [],
                    "involved_actor_ids": list(ids) if k == hit_k else [],
                    "all_involved_observed": True,
                }
                for k in range(1, 6)
            ],
        }

    def test_perfect_answer(self):
        gt = self._gt()
        resp = {
            "predictions": [
                {
                    "k": k,
                    "accident_expected": (k == 3),
                    "involved_actor_ids": ["V002"] if k == 3 else [],
                    "reason": "-",
                    "confidence": "high",
                }
                for k in range(1, 6)
            ]
        }
        s = score_response(resp, gt)
        self.assertEqual(s["counts"], {"TP": 1, "FP": 0, "TN": 4, "FN": 0})
        self.assertEqual(s["accuracy"], 1.0)
        row = next(r for r in s["per_interval"] if r["k"] == 3)
        self.assertEqual(row["vehicle_recall"], 1.0)
        self.assertEqual(row["actor_precision"], 1.0)

    def test_missed_accident(self):
        resp = {
            "predictions": [
                {"k": k, "accident_expected": False, "involved_actor_ids": [],
                 "reason": "-", "confidence": "low"}
                for k in range(1, 6)
            ]
        }
        s = score_response(resp, self._gt())
        self.assertEqual(s["counts"]["FN"], 1)
        self.assertEqual(s["recall"], 0.0)

    def test_false_alarm(self):
        resp = {
            "predictions": [
                {"k": k, "accident_expected": True,
                 "involved_actor_ids": ["EGO_v1"], "reason": "-",
                 "confidence": "high"}
                for k in range(1, 6)
            ]
        }
        s = score_response(resp, self._gt())
        self.assertEqual(s["counts"]["FP"], 4)
        self.assertEqual(s["counts"]["TP"], 1)
        row = next(r for r in s["per_interval"] if r["k"] == 3)
        self.assertEqual(row["vehicle_recall"], 0.0)  # 잘못된 차량 지목
        self.assertEqual(row["extra_actor_ids"], ["EGO_v1"])

    def test_duplicate_actor_ids_any_counts(self):
        """융합이 한 차량을 여러 id 로 쪼갠 경우 하나만 지목해도 정답."""
        gt = self._gt(ids=("V002", "V009"))
        resp = {
            "predictions": [
                {"k": 3, "accident_expected": True,
                 "involved_actor_ids": ["V009"], "reason": "-",
                 "confidence": "high"}
            ]
        }
        s = score_response(resp, gt)
        row = next(r for r in s["per_interval"] if r["k"] == 3)
        self.assertEqual(row["vehicle_recall"], 1.0)

    def test_missing_prediction_counts_as_miss(self):
        resp = {"predictions": [{"k": 1, "accident_expected": False,
                                 "involved_actor_ids": [], "reason": "-",
                                 "confidence": "low"}]}
        s = score_response(resp, self._gt())
        self.assertEqual(s["counts"]["FN"], 1)
        missing = [r for r in s["per_interval"] if r["status"] == "missing"]
        self.assertEqual(len(missing), 4)

    def test_aggregate(self):
        gt = self._gt()
        good = {"predictions": [
            {"k": k, "accident_expected": (k == 3),
             "involved_actor_ids": ["V002"] if k == 3 else [],
             "reason": "-", "confidence": "high"} for k in range(1, 6)]}
        agg = aggregate_scores([score_response(good, gt),
                                score_response(good, gt)])
        self.assertEqual(agg["n_windows"], 2)
        self.assertEqual(agg["counts"]["TP"], 2)
        self.assertEqual(agg["accuracy"], 1.0)
        self.assertEqual(agg["vehicle_recall_mean"], 1.0)


class TestFileOutput(unittest.TestCase):
    def test_write_window_set_files_and_manifest(self):
        snaps = make_snapshots(n=21, dt=0.5)
        cfg = WindowConfig(window_s=5.0, stride_s=1.0, horizon_s=5.0)
        with tempfile.TemporaryDirectory() as d:
            manifest = write_window_set(
                snaps, out_dir=d, scfg=SerializeConfig(), cfg=cfg,
                collision=None, scenario_id="split_x/TownX", town="TownX",
                scenario_split="split_x",
            )
            for lbl in ("0-5", "1-6", "5-10"):
                self.assertTrue(
                    os.path.isfile(os.path.join(d, f"llm_payload_{lbl}.json")),
                    lbl,
                )
                self.assertTrue(
                    os.path.isfile(os.path.join(d, f"ground_truth_{lbl}.json")),
                    lbl,
                )
                self.assertTrue(
                    os.path.isfile(os.path.join(d, f"window_{lbl}.txt")), lbl
                )
            self.assertTrue(os.path.isfile(os.path.join(d, "manifest.json")))

            # payload 와 정답은 서로 다른 파일이며 payload 에 정답이 없다
            p = json.load(open(os.path.join(d, "llm_payload_0-5.json"),
                               encoding="utf-8"))
            g = json.load(open(os.path.join(d, "ground_truth_0-5.json"),
                               encoding="utf-8"))
            # payload 에는 정답 구조가 없어야 한다 (스키마의 accident_expected
            # 필드명과 혼동하지 않도록 정답 전용 키로 검사)
            pblob = json.dumps(p, ensure_ascii=False)
            for gt_only in ("involved_carla_ids", "interval_start_s",
                            "unobserved_carla_ids", "collision"):
                self.assertNotIn(gt_only, pblob, gt_only)
            self.assertIn("expected", g)
            self.assertEqual(len(g["expected"]), 5)
            self.assertEqual(g["scenario"]["id"], "split_x/TownX")

            self.assertEqual(manifest["summary"]["n_windows"], 6)
            labels = [w["label"] for w in manifest["windows"]]
            self.assertEqual(labels[0], "0-5")
            self.assertIn("output_schema", manifest)

    def test_manifest_counts_accident_windows(self):
        class C:
            occurred = True
            time_s = 7.4
            carla_ids = (7001,)

            def to_dict(self):
                return {"occurred": True, "time_s": 7.4, "carla_ids": [7001]}

        snaps = make_snapshots(n=21, dt=0.5)
        cfg = WindowConfig(window_s=5.0, stride_s=1.0, horizon_s=5.0)
        with tempfile.TemporaryDirectory() as d:
            m = write_window_set(
                snaps, out_dir=d, cfg=cfg, collision=C(),
                scenario_id="x/y", town="TownX",
            )
            s = m["summary"]
            self.assertEqual(
                s["n_windows"],
                s["n_windows_with_accident"] + s["n_windows_without_accident"],
            )
            self.assertGreater(s["n_windows_with_accident"], 0)
            # 충돌(7.4s) 이후 t_end 인 윈도우는 제외되어야 한다
            for w in m["windows"]:
                self.assertLess(w["t_end_s"], 7.4)


@unittest.skipUnless(HAS_DATA, f"DeepAccident 데이터가 없습니다 ({DA_ROOT})")
class TestWithRealData(unittest.TestCase):
    def test_accident_scenario_windows_and_truth(self):
        from traffic_llm.config import PipelineConfig
        from traffic_llm.da_runner import DeepAccidentRunner
        from traffic_llm.deepaccident import estimate_collision

        cfg = PipelineConfig()
        r = DeepAccidentRunner(DA_ROOT, cfg)
        res = r.build(
            "Town10HD_type001_subtype0001_scenario00014",
            "type1_subtype1_accident",
        )
        collision = estimate_collision(res.scenario, cfg.deepaccident)
        self.assertTrue(collision.occurred)
        self.assertEqual(collision.method, "trajectory")

        wcfg = WindowConfig(window_s=5.0, stride_s=1.0, horizon_s=5.0)
        agent_ids = {
            ag: res.scenario.meta.agent_id_of(ag) for ag in res.scenario.agents
        }
        with tempfile.TemporaryDirectory() as d:
            m = write_window_set(
                list(res.snapshots(rate_hz=2.0)), out_dir=d,
                scfg=cfg.serialize, cfg=wcfg, collision=collision,
                agent_carla_ids=agent_ids,
                scenario_id=res.scenario.scenario_id,
                scenario_split=res.scenario.scenario_type,
                town=res.scenario.town,
            )
            self.assertGreater(m["summary"]["n_windows"], 0)
            for entry in m["windows"]:
                g = json.load(
                    open(os.path.join(d, entry["ground_truth"]),
                         encoding="utf-8")
                )
                hits = [e for e in g["expected"] if e["accident_expected"]]
                # 충돌은 정확히 한 구간에만 배정된다
                self.assertLessEqual(len(hits), 1)
                for e in hits:
                    # 충돌 주체는 meta 의 colliding agents 와 일치해야 한다
                    ids = {v["carla_id"] for v in e["involved_vehicles"]}
                    self.assertEqual(ids, set(collision.carla_ids))
                    for v in e["involved_vehicles"]:
                        self.assertTrue(
                            v["actor_ids"],
                            f"CARLA {v['carla_id']} 가 액터로 매핑되지 않았다",
                        )
                        # 융합이 신원을 쪼개지 않았는지
                        self.assertEqual(
                            len(v["actor_ids"]), 1,
                            f"CARLA {v['carla_id']} → {v['actor_ids']} (신원 분열)",
                        )

                p = json.load(
                    open(os.path.join(d, entry["payload"]), encoding="utf-8")
                )
                blob = json.dumps(p, ensure_ascii=False)
                # 분할명·충돌 정답이 새면 안 된다.
                # (스키마의 accident_expected 는 답변 형식이므로 정상)
                for leak in (
                    "type1_subtype1_accident",
                    "type1_subtype1_normal",
                    "_subtype",
                    "collision",
                    "colliding",
                    "ground_truth",
                    str(collision.time_s),
                ):
                    self.assertNotIn(leak, blob, f"정답 누수: {leak}")

    def test_normal_scenario_all_windows_negative(self):
        from traffic_llm.config import PipelineConfig
        from traffic_llm.da_runner import DeepAccidentRunner
        from traffic_llm.deepaccident import estimate_collision

        cfg = PipelineConfig()
        r = DeepAccidentRunner(DA_ROOT, cfg)
        res = r.build("Town01", "type1_subtype1_normal")
        collision = estimate_collision(res.scenario, cfg.deepaccident)
        self.assertFalse(collision.occurred)

        wcfg = WindowConfig(window_s=5.0, stride_s=1.0, horizon_s=5.0)
        with tempfile.TemporaryDirectory() as d:
            m = write_window_set(
                list(res.snapshots(rate_hz=2.0)), out_dir=d,
                scfg=cfg.serialize, cfg=wcfg, collision=collision,
                scenario_id=res.scenario.scenario_id,
                scenario_split=res.scenario.scenario_type,
                town=res.scenario.town,
            )
            self.assertGreater(m["summary"]["n_windows"], 3)
            self.assertEqual(m["summary"]["n_windows_with_accident"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestPerceptionRelations(unittest.TestCase):
    """구조화 데이터가 "누가 누구를 인지했는가"를 전달하는지.

    이 관계가 V2X 융합의 핵심이다. 관측차량 사각지대를 노변 인프라가 메우는지를
    판단하려면 관측자별 커버리지를 알아야 하는데, 액터별 observed_by 를 모델이
    직접 역변환하게 두면 자주 틀린다.
    """

    @staticmethod
    def _pl(off: float = -1.6, s: float = 0.0) -> RoadPlacement:
        """도로명을 ASCII 로 둔다 — 영어 출력의 한글 검사에서 지도 데이터(한글
        도로명)와 번역 누락을 구분하기 위해서다."""
        pl = placement(off=off, s=s)
        return replace(pl, road_name="road1")

    def _snaps(self):
        """관측자 2대 + 인프라 1기, 커버리지가 서로 다른 시퀀스.

        make_snapshots() 는 관측자가 하나뿐이라 이 관계를 시험할 수 없다. 여기서는
        일부러 겹치는 관측(V100)·단독 관측(V200=v2만, V300=인프라만)·시야에서
        사라지는 관측(V400 은 마지막 시각에 아무도 못 봄)을 만든다.
        """
        out = []
        for i in range(7):
            t = i * 0.5
            actors = [
                ActorState(
                    actor_id="EGO_v1", kind="ego", cls="car",
                    world_xy=(0.0, 10.0 * t), heading_deg=0.0, speed_mps=10.0,
                    accel_mps2=0.0, placement=self._pl(s=10.0 * t),
                    observed_by=["v1", "v2"], source_track_ids=[],
                ),
                ActorState(
                    actor_id="EGO_v2", kind="ego", cls="car",
                    world_xy=(3.3, 10.0 * t), heading_deg=0.0, speed_mps=10.0,
                    accel_mps2=0.0, placement=self._pl(off=1.6, s=10.0 * t),
                    observed_by=["v2"], source_track_ids=[],
                ),
                # 두 관측자가 함께 보는 차량 → 단독 관측이 아니다
                ActorState(
                    actor_id="V100", kind="observed", cls="car",
                    world_xy=(0.0, 40.0), heading_deg=0.0, speed_mps=8.0,
                    accel_mps2=0.0, placement=self._pl(s=40.0),
                    observed_by=["v1", "v2"], confidence=0.9,
                    position_quality=0.8, observed_range_m=20.0 + t,
                    source_track_ids=[1],
                ),
                # v2 만 보는 차량
                ActorState(
                    actor_id="V200", kind="observed", cls="car",
                    world_xy=(6.6, 30.0), heading_deg=0.0, speed_mps=9.0,
                    accel_mps2=0.0, placement=self._pl(off=4.8, s=30.0),
                    observed_by=["v2"], confidence=0.7,
                    position_quality=0.6, observed_range_m=45.0,
                    source_track_ids=[2],
                ),
                # 인프라만 보는 차량 (관측차량 사각지대)
                ActorState(
                    actor_id="V300", kind="observed", cls="car",
                    world_xy=(-20.0, 55.0), heading_deg=90.0, speed_mps=11.0,
                    accel_mps2=0.0, placement=self._pl(off=-4.8, s=55.0),
                    observed_by=["rsu1"], confidence=0.75,
                    position_quality=0.5, observed_range_m=60.0,
                    source_track_ids=[3],
                ),
            ]
            if i < 6:  # 마지막 시각에는 아무도 못 보는 차량
                actors.append(
                    ActorState(
                        actor_id="V400", kind="observed", cls="car",
                        world_xy=(0.0, 90.0), heading_deg=0.0, speed_mps=7.0,
                        accel_mps2=0.0, placement=self._pl(s=90.0),
                        observed_by=["v1"], confidence=0.5,
                        position_quality=0.4, observed_range_m=88.0,
                        source_track_ids=[4],
                    )
                )
            out.append(
                SceneSnapshot(
                    t=t, actors=actors, interactions=[], area_name="TownX",
                    map_context={
                        "road_count": 2,
                        "observer_ids": ["v1", "v2"],
                        "infrastructure_ids": ["rsu1"],
                    },
                    scenario=None, frame_idx=i + 1,
                )
            )
        return out

    def _win(self, **kw):
        from traffic_llm.accident_qa import build_windows

        cfg = WindowConfig(window_s=3.0, stride_s=1.0, horizon_s=3.0, **kw)
        wins, _ = build_windows(self._snaps(), cfg)
        return wins[0], cfg

    def test_fixture_exercises_overlapping_and_sole_coverage(self):
        """픽스처가 실제로 겹침·단독·소실을 모두 담고 있는지 (테스트의 전제)."""
        from traffic_llm.accident_qa import window_json

        win, cfg = self._win()
        d = window_json(win, SerializeConfig(), cfg)
        by = {o["id"]: o for o in d["perception"]["observers"]}
        self.assertEqual(set(by), {"v1", "v2", "rsu1"})
        self.assertIn("V100", by["v1"]["observed_now"])
        self.assertIn("V100", by["v2"]["observed_now"])
        self.assertEqual(by["v2"]["only_this_observer"], ["V200"])
        self.assertEqual(by["rsu1"]["only_this_observer"], ["V300"])
        # 마지막 시각에 사라진 V400 은 아무 관측자의 현재 목록에도 없다
        for o in by.values():
            self.assertNotIn("V400", o["observed_now"])
        # 그래도 윈도우 집계에는 남는다 — 이것이 now/window 를 나눈 이유다
        self.assertGreater(
            by["v1"]["n_observed_in_window"], by["v1"]["n_observed_now"]
        )

    def test_trajectory_samples_carry_observation(self):
        from traffic_llm.accident_qa import window_json

        win, cfg = self._win()
        d = window_json(win, SerializeConfig(), cfg)
        for aid, rows in d["trajectories"].items():
            for r in rows:
                self.assertIn("observed_by", r, aid)
                self.assertIn("range_m", r, aid)
                self.assertIn("confidence", r, aid)
                self.assertIsInstance(r["observed_by"], list)

    def test_trajectory_observation_can_be_disabled(self):
        from traffic_llm.accident_qa import window_json

        win, cfg = self._win(trajectory_observation=False)
        d = window_json(win, SerializeConfig(), cfg)
        row = next(iter(d["trajectories"].values()))[0]
        self.assertNotIn("observed_by", row)

    def test_perception_block_inverts_observed_by(self):
        """관측자 관점이 액터별 observed_by 와 정확히 일치해야 한다."""
        from traffic_llm.accident_qa import window_json

        win, cfg = self._win()
        d = window_json(win, SerializeConfig(), cfg)
        actors = d["last_snapshot"]["actors"]
        for o in d["perception"]["observers"]:
            expected = {
                a["id"]
                for a in actors
                if o["id"] in a["observed_by"] and a["id"] != o["self_actor_id"]
            }
            self.assertEqual(set(o["observed_now"]), expected, o["id"])
            self.assertEqual(o["n_observed_now"], len(o["observed_now"]))

    def test_perception_excludes_the_observer_itself(self):
        from traffic_llm.accident_qa import window_json

        win, cfg = self._win()
        d = window_json(win, SerializeConfig(), cfg)
        for o in d["perception"]["observers"]:
            if o["self_actor_id"]:
                self.assertNotIn(o["self_actor_id"], o["observed_now"])

    def test_only_this_observer_marks_unique_coverage(self):
        """단독 인지 = 다른 관측자의 사각지대를 메운 몫."""
        from traffic_llm.accident_qa import window_json

        win, cfg = self._win()
        d = window_json(win, SerializeConfig(), cfg)
        seen_by = {
            a["id"]: a["observed_by"] for a in d["last_snapshot"]["actors"]
        }
        for o in d["perception"]["observers"]:
            for aid in o["only_this_observer"]:
                self.assertEqual(seen_by[aid], [o["id"]], aid)
            # 단독 목록은 인지 목록의 부분집합
                self.assertIn(aid, o["observed_now"])

    def test_window_counts_are_at_least_current(self):
        from traffic_llm.accident_qa import window_json

        win, cfg = self._win()
        d = window_json(win, SerializeConfig(), cfg)
        for o in d["perception"]["observers"]:
            self.assertGreaterEqual(
                o["n_observed_in_window"], o["n_observed_now"], o["id"]
            )

    def test_infrastructure_is_marked_as_such(self):
        from traffic_llm.accident_qa import window_json

        win, cfg = self._win()
        d = window_json(win, SerializeConfig(), cfg)
        kinds = {o["id"]: o["kind"] for o in d["perception"]["observers"]}
        mc = win.last.map_context
        for oid in mc.get("infrastructure_ids") or []:
            self.assertEqual(kinds.get(oid), "infrastructure")
        for oid in mc.get("observer_ids") or []:
            self.assertEqual(kinds.get(oid), "vehicle")

    def test_perception_can_be_disabled(self):
        from traffic_llm.accident_qa import window_json

        win, cfg = self._win(include_perception=False)
        d = window_json(win, SerializeConfig(), cfg)
        self.assertNotIn("perception", d)

    def test_text_section_lists_observers(self):
        from traffic_llm.accident_qa import render_window_text

        win, cfg = self._win()
        txt = render_window_text(win, SerializeConfig(), cfg)
        self.assertIn("관측 관계", txt)
        for oid in win.last.map_context.get("observer_ids") or []:
            self.assertIn(oid, txt)

    def test_english_text_section(self):
        from traffic_llm.accident_qa import render_window_text

        win, cfg = self._win()
        txt = render_window_text(win, SerializeConfig(language="en"), cfg)
        self.assertIn("Perception relations", txt)
        self.assertNotRegex(txt, "[가-힣]")


class TestAdaptiveCaps(unittest.TestCase):
    """수록 상한이 관측량에 맞춰 늘어나는지.

    카메라를 6대로 늘려 관측 차량이 24→26대, 접근 쌍이 43→88개가 됐는데 상한이
    15대·6쌍으로 고정이어서 **사고 당사자 쌍이 payload 에서 사라졌다**. 고정
    상한은 관측이 늘수록 담는 비율만 줄인다.
    """

    def test_actor_cap_grows_with_scene(self):
        cfg = WindowConfig()
        self.assertEqual(cfg.actor_cap(5), cfg.actor_cap_floor)
        self.assertEqual(cfg.actor_cap(26), 26)
        self.assertEqual(cfg.actor_cap(200), cfg.actor_cap_ceiling)

    def test_pair_cap_grows_with_scene(self):
        cfg = WindowConfig()
        self.assertEqual(cfg.pair_cap(4), cfg.pair_cap_floor)
        self.assertEqual(cfg.pair_cap(26), 13)
        self.assertEqual(cfg.pair_cap(500), cfg.pair_cap_ceiling)

    def test_explicit_caps_win(self):
        cfg = WindowConfig(max_actors=7, max_closing_pairs=3)
        self.assertEqual(cfg.actor_cap(26), 7)
        self.assertEqual(cfg.pair_cap(26), 3)

    def test_must_include_threshold_follows_horizon(self):
        self.assertAlmostEqual(WindowConfig(horizon_s=5.0).must_include_contact_s, 2.5)
        self.assertAlmostEqual(WindowConfig(horizon_s=8.0).must_include_contact_s, 4.0)
        self.assertAlmostEqual(
            WindowConfig(closing_must_include_contact_s=1.0).must_include_contact_s,
            1.0,
        )

    def _crowded(self, n_extra: int = 8):
        """접촉까지 ~2초인 쌍 A/B 하나 + 그보다 빠른 무관한 쌍 여러 개.

        무관한 쌍이 순위를 채우면 A/B 가 상한 밖으로 밀려난다. 실제 데이터에서
        카메라를 늘렸을 때 정확히 이 일이 일어났다. 쌍은 윈도우 동안 **교차하지
        않고** 계속 접근해야 한다 — 교차해 버리면 거리가 다시 늘어 접근 쌍으로
        잡히지 않는다.
        """
        out = []
        for i in range(11):
            t = i * 0.5
            # A/B: 220m → 110m, 접근율 22m/s → 접촉 5.0s
            actors = [
                ActorState(
                    actor_id="A", kind="observed", cls="car",
                    world_xy=(0.0, -110.0 + 11.0 * t), heading_deg=0.0,
                    speed_mps=11.0, accel_mps2=0.0, placement=placement(),
                    observed_by=["v1"], source_track_ids=[1],
                ),
                ActorState(
                    actor_id="B", kind="observed", cls="car",
                    world_xy=(0.0, 110.0 - 11.0 * t), heading_deg=180.0,
                    speed_mps=11.0, accel_mps2=0.0, placement=placement(),
                    observed_by=["v1"], source_track_ids=[2],
                ),
            ]
            # X/Y: 160m → 30m, 접근율 26m/s → 접촉 1.2s (A/B 보다 빠르다)
            for k in range(n_extra):
                base = 3000.0 + 400.0 * k  # 서로 멀리 떨어뜨려 교차쌍을 줄인다
                actors += [
                    ActorState(
                        actor_id=f"X{k}", kind="observed", cls="car",
                        world_xy=(base, -80.0 + 13.0 * t), heading_deg=0.0,
                        speed_mps=13.0, accel_mps2=0.0, placement=placement(),
                        observed_by=["v1"], source_track_ids=[100 + k],
                    ),
                    ActorState(
                        actor_id=f"Y{k}", kind="observed", cls="car",
                        world_xy=(base, 80.0 - 13.0 * t), heading_deg=180.0,
                        speed_mps=13.0, accel_mps2=0.0, placement=placement(),
                        observed_by=["v1"], source_track_ids=[200 + k],
                    ),
                ]
            out.append(
                SceneSnapshot(
                    t=t, actors=actors, interactions=[], area_name="X",
                    map_context={"observer_ids": ["v1"]}, scenario=None,
                    frame_idx=i + 1,
                )
            )
        return out

    def test_fixture_crowds_the_slower_pair_out(self):
        """전제 확인 — 작은 고정 상한이면 A/B 가 순위에서 밀려난다."""
        from traffic_llm.accident_qa import closing_pairs

        cfg = WindowConfig(window_s=5.0, max_closing_pairs=3,
                           closing_must_include_contact_s=0.0)
        wins, _ = build_windows(self._crowded(), cfg)
        pairs = closing_pairs(wins[0], cfg)
        self.assertEqual(len(pairs), 3)
        self.assertNotIn({"A", "B"}, [{p.a, p.b} for p in pairs])

    def test_adaptive_cap_admits_more_pairs_than_the_floor(self):
        from traffic_llm.accident_qa import closing_pairs

        snaps = self._crowded()
        n = len(snaps[-1].actors)
        auto = WindowConfig(window_s=5.0)
        self.assertGreater(auto.pair_cap(n), auto.pair_cap_floor)
        fixed = WindowConfig(window_s=5.0, max_closing_pairs=auto.pair_cap_floor,
                             closing_must_include_contact_s=0.0)
        wins_a, _ = build_windows(snaps, auto)
        wins_f, _ = build_windows(snaps, fixed)
        self.assertGreater(
            len(closing_pairs(wins_a[0], auto)),
            len(closing_pairs(wins_f[0], fixed)),
        )

    def test_must_include_survives_even_past_the_cap(self):
        """상한이 작아도 접촉 임박 쌍은 버리지 않는다."""
        from traffic_llm.accident_qa import closing_pairs

        snaps = self._crowded()
        tight = WindowConfig(window_s=5.0, max_closing_pairs=3,
                             closing_must_include_contact_s=0.0)
        wins, _ = build_windows(snaps, tight)
        self.assertEqual(len(closing_pairs(wins[0], tight)), 3)
        # A/B 의 접촉 예상이 5.0s 이므로 임계를 그 위로 올리면 반드시 포함된다
        loose = WindowConfig(window_s=5.0, max_closing_pairs=3,
                             closing_must_include_contact_s=6.0)
        pairs = closing_pairs(wins[0], loose)
        self.assertGreater(len(pairs), 3, "임박 쌍이 추가되지 않았다")
        self.assertIn({"A", "B"}, [{p.a, p.b} for p in pairs])

    def test_manifest_records_effective_caps(self):
        import tempfile

        snaps = self._crowded(4)
        with tempfile.TemporaryDirectory() as d:
            write_window_set(
                snaps, d, SerializeConfig(), WindowConfig(window_s=5.0)
            )
            with open(os.path.join(d, "manifest.json"), encoding="utf-8") as f:
                man = json.load(f)
        c = man["config"]
        self.assertIsNone(c["max_actors"])
        self.assertIsNone(c["max_closing_pairs"])
        self.assertTrue(c["max_actors_effective"])
        self.assertTrue(all(v > 0 for v in c["max_actors_effective"]))
        self.assertTrue(all(v > 0 for v in c["max_closing_pairs_effective"]))


class TestResponseIssues(unittest.TestCase):
    """응답 형식 문제를 채점이 드러내는지.

    채점은 k 로 짝지으므로 중복은 조용히 무시되고, 누락은 정답이 '사고'일 때만
    FN 이 된다 — 형식 오류가 점수에 거의 드러나지 않는다. 실제로 모델이 같은 k 를
    두 번 낸 응답이 있었고("동일 구간 중복 표기 방지용 확인 항목"이라 적어 두기까지
    했다) 채점 결과만 봐서는 보이지 않았다.

    스키마 제약(minItems/maximum)으로 막지 않는 이유: OpenAI strict 모드가 그
    키워드를 지원하지 않아 provider 별로 스키마가 갈린다.
    """

    @staticmethod
    def _pred(k, acc=False):
        return {
            "k": k, "interval_s": f"({k - 1}, {k}]", "accident_expected": acc,
            "involved_actor_ids": [], "reason": "", "confidence": "low",
        }

    def test_clean_response_has_no_issues(self):
        from traffic_llm.accident_qa import response_issues

        self.assertEqual(
            response_issues([self._pred(k) for k in (1, 2, 3)], [1, 2, 3]), []
        )

    def test_duplicate_k_reported(self):
        from traffic_llm.accident_qa import response_issues

        iss = response_issues(
            [self._pred(k) for k in (1, 2, 3, 3)], [1, 2, 3]
        )
        kinds = {i["kind"] for i in iss}
        self.assertIn("duplicate_k", kinds)
        dup = next(i for i in iss if i["kind"] == "duplicate_k")
        self.assertEqual(dup["detail"], {3: 2})

    def test_missing_k_reported(self):
        from traffic_llm.accident_qa import response_issues

        iss = response_issues([self._pred(k) for k in (1, 3)], [1, 2, 3])
        m = next(i for i in iss if i["kind"] == "missing_k")
        self.assertEqual(m["detail"], [2])

    def test_unexpected_k_reported(self):
        from traffic_llm.accident_qa import response_issues

        iss = response_issues([self._pred(k) for k in (1, 2, 3, 9)], [1, 2, 3])
        m = next(i for i in iss if i["kind"] == "unexpected_k")
        self.assertEqual(m["detail"], [9])

    def test_missing_k_field_reported(self):
        from traffic_llm.accident_qa import response_issues

        bad = self._pred(1)
        del bad["k"]
        iss = response_issues([bad], [1])
        self.assertIn("missing_k_field", {i["kind"] for i in iss})

    def test_score_response_carries_issues(self):
        """중복이 점수를 바꾸지 않으면서도 보고되어야 한다."""
        snaps = make_snapshots(n=21, dt=0.5)
        cfg = WindowConfig(window_s=5.0, horizon_s=3.0)
        wins, _ = build_windows(snaps, cfg)
        gt = window_ground_truth(wins[0], cfg, collision=None)
        clean = {"predictions": [self._pred(k) for k in (1, 2, 3)]}
        dirty = {"predictions": [self._pred(k) for k in (1, 2, 3, 3)]}
        a, b = score_response(clean, gt), score_response(dirty, gt)
        self.assertEqual(a["counts"], b["counts"], "중복이 점수를 바꿨다")
        self.assertEqual(a["response_issues"], [])
        self.assertTrue(b["response_issues"])

    def test_aggregate_counts_issues(self):
        snaps = make_snapshots(n=21, dt=0.5)
        cfg = WindowConfig(window_s=5.0, horizon_s=3.0)
        wins, _ = build_windows(snaps, cfg)
        gt = window_ground_truth(wins[0], cfg, collision=None)
        dirty = {"predictions": [self._pred(k) for k in (1, 2, 3, 3)]}
        agg = aggregate_scores([score_response(dirty, gt) for _ in range(2)])
        self.assertEqual(agg["response_issue_counts"].get("duplicate_k"), 2)

    def test_schema_tells_the_model_the_bucket_count(self):
        cfg = WindowConfig(horizon_s=5.0)
        sch = i18n.prediction_schema("ko", cfg.n_horizon_buckets)
        desc = sch["properties"]["predictions"]["description"]
        self.assertIn("5", desc)
        # provider 이식성: strict 모드가 지원하지 않는 제약은 넣지 않는다
        blob = json.dumps(sch)
        for bad in ("minItems", "maxItems", "minimum", "maximum"):
            self.assertNotIn(bad, blob)


class TestJsonBlockOptional(unittest.TestCase):
    """구조화 JSON 블록은 선택이고 기본은 꺼짐.

    다섯 블록 전부가 자연어 브리핑의 대응 절과 짝을 이루면서 입력의 79%를
    차지한다. 끄면 payload 가 약 1/5 이 된다.

    다만 JSON 에만 있던 값(방위각 숫자, 지도 규약)은 자연어로 옮겨 두어야 한다 —
    그러지 않으면 "옵션"이 아니라 조용한 정보 손실이다.
    """

    def setUp(self):
        self.snaps = make_snapshots(n=21, dt=0.5)
        self.cfg = WindowConfig(window_s=5.0, stride_s=1.0, horizon_s=5.0)
        self.wins, _ = build_windows(self.snaps, self.cfg)
        self.scfg = SerializeConfig()

    def _blob(self, on: bool) -> str:
        cfg = replace(self.cfg, include_json_block=on)
        return json.dumps(
            build_window_payload(self.wins[0], self.scfg, cfg),
            ensure_ascii=False,
        )

    def test_default_is_off(self):
        self.assertFalse(WindowConfig().include_json_block)
        self.assertFalse(SerializeConfig().include_json_block)

    def test_block_absent_by_default_present_when_enabled(self):
        self.assertNotIn("```json", self._blob(False))
        self.assertIn("```json", self._blob(True))

    def test_turning_it_off_shrinks_the_payload(self):
        """줄어드는 양이 JSON 블록 크기만큼이어야 한다.

        고정 배수(예: 절반)로 검사하지 않는다 — 이 픽스처는 액터가 2대뿐이라
        system 프롬프트·질문 규칙 같은 **고정 오버헤드**가 본문보다 크고, 규칙
        문장 하나를 추가하면 배수가 흔들린다. 실데이터(액터 26대)에서는 실제로
        119K → 25K자, 4.7배다.
        """
        off, on = len(self._blob(False)), len(self._blob(True))
        self.assertLess(off, on)
        self.assertGreater(on - off, 3000, "JSON 블록이 제거되지 않았다")

    def test_briefing_survives(self):
        """브리핑과 질문은 그대로 있어야 한다."""
        blob = self._blob(False)
        self.assertIn("관측 윈도우", blob)
        self.assertIn("차량별 시간 경과", blob)
        self.assertIn("EGO_v1", blob)

    def test_map_conventions_are_in_the_text(self):
        """통행측·차선번호 기준은 JSON 의 map_context 에만 있던 값이다."""
        for lang, probe in (
            ("ko", ("지도 규약", "우측통행", "1차선은 중앙선쪽")),
            ("en", ("map conventions", "right-hand traffic", "median-side")),
        ):
            cfg = replace(self.cfg, include_json_block=False)
            txt = render_window_text(
                self.wins[0], SerializeConfig(language=lang), cfg
            )
            for s in probe:
                self.assertIn(s, txt, f"{lang}: {s}")

    def test_numeric_heading_is_in_the_text(self):
        """방위 라벨('북행')은 8방위로 뭉개져 충돌 기하를 따질 수 없다."""
        cfg = replace(self.cfg, include_json_block=False)
        txt = render_window_text(self.wins[0], self.scfg, cfg)
        self.assertRegex(txt, r"방위 \d+°")
        en = render_window_text(
            self.wins[0], SerializeConfig(language="en"), cfg
        )
        self.assertRegex(en, r"heading \d+ deg")
        # 영어 출력에 한국어 **템플릿**이 남지 않았는지. 도로명 같은 지도
        # 데이터의 한글은 번역 대상이 아니므로 전체 한글 검사는 하지 않는다
        # (그 검사는 ASCII 도로명 픽스처를 쓰는 TestPerceptionRelations 에 있다).
        for ko in ("지도 규약", "방위 ", "우측통행", "중앙선쪽"):
            self.assertNotIn(ko, en)

    def test_left_hand_traffic_and_curb_numbering_render(self):
        snap = self.wins[0].last
        snap.map_context = dict(snap.map_context)
        snap.map_context.update(
            {"drive_side": "left", "lane_numbering": "from_curb"}
        )
        txt = render_window_text(
            self.wins[0], self.scfg, replace(self.cfg, include_json_block=False)
        )
        self.assertIn("좌측통행", txt)
        self.assertIn("1차선은 가장자리쪽", txt)

    def test_single_snapshot_path_honours_the_config(self):
        from traffic_llm.serialize import build_messages

        snap = self.snaps[0]
        off = json.dumps(
            build_messages(snap, "q", SerializeConfig()), ensure_ascii=False
        )
        on = json.dumps(
            build_messages(snap, "q", SerializeConfig(include_json_block=True)),
            ensure_ascii=False,
        )
        self.assertNotIn("```json", off)
        self.assertIn("```json", on)
        # 명시 인자가 설정을 덮어쓴다
        forced = json.dumps(
            build_messages(snap, "q", SerializeConfig(), include_json=True),
            ensure_ascii=False,
        )
        self.assertIn("```json", forced)


class TestActorIdDisambiguation(unittest.TestCase):
    """관측자 이름과 액터 id 를 구별할 수 있는지.

    두 이름 체계가 payload 에 섞여 있다 — '관측 관계' 절은 관측자 이름
    ('ego_vehicle'), '관측차량' 절은 액터 id('EGO_ego_vehicle')를 쓴다. 매핑이
    JSON 블록의 self_actor_id 에만 있어서, JSON 을 기본 비활성으로 바꾼 뒤
    실제 호출에서 모델이 관측자 이름으로 답해 차량 지목이 전부 오답 처리됐다.
    """

    def _win(self):
        snaps = TestPerceptionRelations._snaps(TestPerceptionRelations())
        cfg = WindowConfig(window_s=3.0, stride_s=1.0, horizon_s=3.0)
        wins, _ = build_windows(snaps, cfg)
        return wins[0], cfg

    def test_perception_rows_carry_the_own_actor_id(self):
        win, cfg = self._win()
        for lang, needle in (("ko", "본체 액터 id"), ("en", "own actor id")):
            txt = render_window_text(win, SerializeConfig(language=lang), cfg)
            self.assertIn(needle, txt, lang)
            # 관측차량마다 본체 id 가 붙어야 한다
            for oid in win.last.map_context["observer_ids"]:
                self.assertRegex(
                    txt, rf"{oid}\b.*(본체 액터 id|own actor id) EGO_{oid}", lang
                )

    def test_infrastructure_has_no_own_actor(self):
        """인프라는 교통 참여자가 아니므로 본체 액터가 없다."""
        win, cfg = self._win()
        txt = render_window_text(win, SerializeConfig(), cfg)
        line = next(
            ln for ln in txt.splitlines() if ln.startswith("- rsu1")
        )
        self.assertNotIn("본체 액터 id", line)

    def test_question_tells_which_id_to_use(self):
        win, cfg = self._win()
        for lang, needle in (
            ("ko", "involved_actor_ids 에는 **액터 id**"),
            ("en", "Use **actor ids** in involved_actor_ids"),
        ):
            q = build_question(win, cfg, lang)
            self.assertIn(needle, q, lang)
            self.assertIn("EGO_", q, lang)

    def test_schema_says_actor_ids_not_observer_names(self):
        for lang, needle in (("ko", "액터 id"), ("en", "Actor ids")):
            sch = i18n.prediction_schema(lang, 5)
            desc = sch["properties"]["predictions"]["items"]["properties"][
                "involved_actor_ids"
            ]["description"]
            self.assertIn(needle, desc, lang)

    def test_mapping_present_without_the_json_block(self):
        """JSON 을 껐을 때도 매핑이 payload 에 있어야 한다 — 그것이 이 수정의 요점."""
        win, cfg = self._win()
        blob = json.dumps(
            build_window_payload(
                win, SerializeConfig(), replace(cfg, include_json_block=False)
            ),
            ensure_ascii=False,
        )
        self.assertNotIn("```json", blob)
        self.assertIn("EGO_v1", blob)
        self.assertIn("본체 액터 id", blob)
