"""사고 예측 질의 생성(슬라이딩 윈도우 + 정답) 테스트."""

from __future__ import annotations

import glob
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_llm.accident_qa import (
    PREDICTION_SCHEMA,
    WindowConfig,
    aggregate_modes,
    aggregate_scores,
    build_actor_lookup,
    build_question,
    build_window_payload,
    build_windows,
    closing_pairs,
    compact_window_json,
    ground_truth_future_snapshots,
    render_window_text,
    score_binary,
    score_modes,
    score_response,
    score_weighted,
    window_ground_truth,
    window_json,
    write_window_set,
)
from dataclasses import replace

from traffic_llm import i18n
from traffic_llm.config import SerializeConfig
from traffic_llm.schemas import (
    ActorState,
    PredictedPath,
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


def pick(wins, label: str):
    """라벨로 창을 고른다.

    인덱스로 고르면 창 생성 방식(누적/고정)이 바뀔 때 다른 창을 집는다 —
    기본이 누적(warmup)으로 바뀌면서 `wins[0]` 이 0-5 에서 0-1 이 되었다.
    """
    for w in wins:
        if w.label == label:
            return w
    raise AssertionError(f"창 {label} 이 없다: {[w.label for w in wins]}")


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
    def test_warmup_windows_grow_then_slide(self):
        """warmup=True면 누적 후 이동: 0-1, 0-2, …, 0-5, 1-6, …"""
        snaps = make_snapshots(n=21, dt=0.5)  # 0 ~ 10초
        # 이동 창 동작만 검사한다 — 전체 구간 창은 아래 TestFullWindow 에서 본다
        cfg = WindowConfig(window_s=5.0, stride_s=1.0, horizon_s=5.0,
                           warmup=True, full_window=False)
        wins, info = build_windows(snaps, cfg)
        labels = [w.label for w in wins]
        self.assertEqual(labels[:6], ["0-1", "0-2", "0-3", "0-4", "0-5"] + ["1-6"])
        self.assertEqual(labels[-1], "5-10")
        for w in wins:
            self.assertLessEqual(w.t_end - w.t_start, 5.0 + 1e-6)
            self.assertAlmostEqual(w.snapshots[0].t, w.t_start, places=6)
            self.assertAlmostEqual(w.snapshots[-1].t, w.t_end, places=6)
        # 창이 자라는 동안 스냅샷 수도 늘어난다
        counts = [len(w.snapshots) for w in wins]
        self.assertEqual(counts[:5], [3, 5, 7, 9, 11])
        self.assertEqual(info["n_windows"], len(wins))
        self.assertTrue(info["warmup"])

    def test_fixed_length_mode_keeps_old_behaviour(self):
        snaps = make_snapshots(n=21, dt=0.5)
        cfg = WindowConfig(window_s=5.0, stride_s=1.0, warmup=False,
                           full_window=False)
        wins, info = build_windows(snaps, cfg)
        self.assertEqual([w.label for w in wins][:4], ["0-5", "1-6", "2-7", "3-8"])
        for w in wins:
            self.assertAlmostEqual(w.t_end - w.t_start, 5.0, places=6)
        self.assertFalse(info["warmup"])

    def test_default_window_mode_is_fixed_without_full_window(self):
        cfg = WindowConfig(window_s=5.0, stride_s=1.0)
        self.assertFalse(cfg.warmup)
        self.assertFalse(cfg.full_window)

    def test_stride_and_window_length_respected(self):
        snaps = make_snapshots(n=21, dt=0.5)
        wins, _ = build_windows(
            snaps,
            WindowConfig(window_s=3.0, stride_s=2.0, warmup=False,
                         full_window=False),
        )
        self.assertEqual([w.label for w in wins], ["0-3", "2-5", "4-7", "6-9"])

    def test_short_data_still_yields_windows(self):
        """창 길이보다 짧은 데이터로도 실험할 수 있어야 한다 — 누적 창의 요점."""
        snaps = make_snapshots(n=7, dt=0.5)  # 0 ~ 3초
        wins, _ = build_windows(snaps, WindowConfig(window_s=5.0, warmup=True))
        self.assertEqual([w.label for w in wins], ["0-1", "0-2", "0-3"])

    def test_fixed_length_mode_drops_short_data(self):
        snaps = make_snapshots(n=7, dt=0.5)  # 0 ~ 3초
        wins, info = build_windows(
            snaps,
            WindowConfig(window_s=5.0, warmup=False, full_window=False),
        )
        self.assertEqual(wins, [])
        self.assertEqual(info["n_windows"], 0)

    # ------------------------------------------------ 전체 구간 창
    def test_full_window_is_added_last(self):
        """이동 창과 별개로 데이터 전체 구간 창이 하나 붙는다 (9초 → 0-9)."""
        snaps = make_snapshots(n=19, dt=0.5)          # 0 ~ 9초
        cfg = WindowConfig(window_s=5.0, stride_s=1.0, warmup=True, full_window=True)
        wins, info = build_windows(snaps, cfg)
        labels = [w.label for w in wins]
        self.assertEqual(labels[-1], "0-9", "전체 구간 창이 마지막에 온다")
        self.assertTrue(info["full_window"])
        full = wins[-1]
        self.assertAlmostEqual(full.t_start, 0.0, places=6)
        self.assertAlmostEqual(full.t_end, 9.0, places=6)
        self.assertEqual(len(full.snapshots), len(snaps),
                         "전체 구간이므로 스냅샷을 전부 담는다")
        # 이동 창 쪽은 그대로 window_s 를 넘지 않는다
        for w in wins[:-1]:
            self.assertLessEqual(w.t_end - w.t_start, 5.0 + 1e-6)

    def test_full_window_can_be_disabled(self):
        snaps = make_snapshots(n=19, dt=0.5)
        wins, info = build_windows(
            snaps, WindowConfig(window_s=5.0, stride_s=1.0, full_window=False)
        )
        self.assertNotIn("0-9", [w.label for w in wins])
        self.assertFalse(info["full_window"])

    def test_full_window_not_duplicated_for_short_data(self):
        """데이터가 window_s 이하면 마지막 누적 창이 이미 전체 구간이다."""
        snaps = make_snapshots(n=7, dt=0.5)           # 0 ~ 3초
        wins, info = build_windows(snaps, WindowConfig(window_s=5.0, warmup=True, full_window=True))
        labels = [w.label for w in wins]
        self.assertEqual(labels, ["0-1", "0-2", "0-3"], "같은 창을 두 번 만들지 않는다")
        self.assertFalse(info["full_window"])
        self.assertEqual(info["full_window_skipped"], "duplicate")

    def test_full_window_skipped_when_collision_is_inside(self):
        """관측 구간 안에서 이미 사고가 났으면 예측 문제가 아니다."""
        snaps = make_snapshots(n=21, dt=0.5)          # 0 ~ 10초
        cfg = WindowConfig(window_s=5.0, stride_s=1.0, full_window=True)
        wins, info = build_windows(snaps, cfg, collision_time_s=7.0)
        self.assertNotIn("0-10", [w.label for w in wins])
        self.assertFalse(info["full_window"])
        self.assertEqual(info["full_window_skipped"], "collision_inside")

    def test_full_window_kept_when_collision_is_after_data(self):
        """사고가 데이터 종료 이후면 전체 구간 창이 가장 강한 질문이 된다."""
        snaps = make_snapshots(n=19, dt=0.5)          # 0 ~ 9초
        cfg = WindowConfig(window_s=5.0, stride_s=1.0, full_window=True)
        wins, info = build_windows(snaps, cfg, collision_time_s=9.2)
        self.assertEqual([w.label for w in wins][-1], "0-9")
        self.assertTrue(info["full_window"])

    def test_full_window_question_states_its_real_length(self):
        """질문 문구가 설정값(5초)이 아니라 실제 관측 길이를 말해야 한다."""
        from traffic_llm.accident_qa import build_question

        snaps = make_snapshots(n=19, dt=0.5)
        cfg = WindowConfig(window_s=5.0, stride_s=1.0, warmup=True, full_window=True)
        wins, _ = build_windows(snaps, cfg)
        q_full = build_question(wins[-1], cfg, "ko")
        self.assertIn("9초 관측 구간", q_full)
        self.assertIn("t=0~9초", q_full)
        # 누적 창도 같은 규칙 — 0-1 창이 "5초" 라고 하면 자기 모순이다
        q_first = build_question(wins[0], cfg, "ko")
        self.assertIn("1초 관측 구간", q_first)
        self.assertNotIn("5초 관측 구간", q_first)

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
        txt = render_window_text(pick(self.wins, "2-7"), self.scfg, self.cfg)
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

    def test_compact_payload_keeps_history_and_numeric_future_geometry(self):
        win = pick(self.wins, "2-7")
        target = next(a for a in win.last.actors if a.actor_id == "V002")
        target.predictions = [
            PredictedPath(
                maneuver="직진",
                probability=1.0,
                # Internal path contract includes t=0 current position first.
                waypoints=[target.world_xy, (3.3, -36.0), (3.3, -48.0)],
                horizon_s=2.0,
            )
        ]
        cfg = replace(self.cfg, payload_profile="compact")
        doc = compact_window_json(win, self.scfg, cfg)
        self.assertEqual(doc["representation"], "compact_geometry_v3")
        self.assertEqual(doc["coordinate_convention"]["future_waypoint_start_s"], 1)
        actor = next(a for a in doc["actors"] if a["id"] == "V002")
        self.assertGreater(len(actor["history"]), 1)
        self.assertEqual(
            actor["future_paths"][0][3],
            [[3.3, -36.0], [3.3, -48.0]],
        )

        payload = build_window_payload(win, self.scfg, cfg)
        blob = json.dumps(payload, ensure_ascii=False)
        self.assertIn("compact_geometry_v3", blob)
        self.assertIn("future_paths", blob)
        self.assertIn("predicted_closest_pairs", blob)
        self.assertIn("class_footprints_m", blob)
        self.assertEqual(
            doc["predicted_pair_method"]["internal_candidate_pool_cap"], 50
        )
        self.assertEqual(
            doc["predicted_pair_method"]["pairs_serialized_cap"], 12
        )
        self.assertEqual(doc["predicted_pair_method"]["contact_margin_m"], 0.0)
        pair_cols = doc["predicted_pair_columns"]
        self.assertEqual(pair_cols[-10:-5], [
            "a_observes_b", "b_observes_a", "mutually_observed",
            "a_observation_available", "b_observation_available",
        ])
        self.assertEqual(pair_cols[-5:], [
            "contact_event", "contact_at_observation",
            "contact_before_observation", "first_contact_s",
            "contact_duration_s",
        ])
        self.assertTrue(all(len(row) == len(pair_cols)
                            for row in doc["predicted_closest_pairs"]))
        # These derived hazard representations belong only to the standard profile.
        self.assertNotIn("closing_pairs", blob)
        self.assertNotIn("서로 접근한 차량 쌍", blob)
        self.assertNotIn("상호작용 및 위험", blob)
        self.assertNotIn("```\n", blob)  # no ASCII BEV code block

    def test_compact_payload_records_positive_contact_margin(self):
        cfg = replace(self.cfg, payload_profile="compact", swept_contact_margin_m=1.0)
        doc = compact_window_json(pick(self.wins, "2-7"), self.scfg, cfg)
        self.assertEqual(doc["predicted_pair_method"]["contact_margin_m"], 1.0)
        payload = build_window_payload(
            pick(self.wins, "2-7"), SerializeConfig(language="en"), cfg
        )
        self.assertIn(
            "1.0 m or less counts as a contact/dangerous-collision candidate",
            json.dumps(payload),
        )

    def test_compact_payload_is_smaller_than_standard(self):
        win = pick(self.wins, "2-7")
        standard = build_window_payload(win, self.scfg, self.cfg)
        compact = build_window_payload(
            win, self.scfg, replace(self.cfg, payload_profile="compact")
        )
        self.assertLess(
            len(json.dumps(compact, ensure_ascii=False)),
            len(json.dumps(standard, ensure_ascii=False)),
        )

    def test_ground_truth_oracle_changes_only_futures_and_regenerates_pairs(self):
        """Oracle paths never rewrite observations and regenerate geometry."""
        snaps = []
        for t in range(5):
            a = ActorState(
                actor_id="A", kind="observed", cls="car", world_xy=(-10.0 + 5 * t, 0.0),
                heading_deg=90.0, speed_mps=5.0, accel_mps2=0.0,
                placement=placement(), observed_by=["v1"], maneuver="straight",
                predictions=[PredictedPath("straight", 1.0,
                    [(-10.0 + 5 * t, 0.0), (-11.0 + 5 * t, 0.0), (-12.0 + 5 * t, 0.0)],
                    2.0, to_roads=["r2"])],
            )
            b = ActorState(
                actor_id="B", kind="observed", cls="car", world_xy=(10.0 - 5 * t, 0.0),
                heading_deg=270.0, speed_mps=5.0, accel_mps2=0.0,
                placement=placement(), observed_by=["v2"], maneuver="straight",
                predictions=[PredictedPath("straight", 1.0,
                    [(10.0 - 5 * t, 0.0), (11.0 - 5 * t, 0.0), (12.0 - 5 * t, 0.0)],
                    2.0, to_roads=["r3"])],
            )
            snaps.append(SceneSnapshot(t=float(t), actors=[a, b], interactions=[]))

        oracle = ground_truth_future_snapshots(snaps, horizon_s=2.0)
        # Predictor mode is untouched because the helper returns a deep copy.
        self.assertEqual(snaps[1].actors[0].predictions[0].waypoints, [(-5.0, 0.0), (-6.0, 0.0), (-7.0, 0.0)])
        original, replaced = snaps[1].actors[0], oracle[1].actors[0]
        self.assertEqual(original.world_xy, replaced.world_xy)
        self.assertEqual(original.track_history, replaced.track_history)
        self.assertEqual(original.observed_by, replaced.observed_by)
        self.assertEqual(original.maneuver, replaced.maneuver)
        self.assertEqual(original.predictions[0].maneuver, replaced.predictions[0].maneuver)
        self.assertEqual(original.predictions[0].probability, replaced.predictions[0].probability)
        self.assertEqual(original.predictions[0].to_roads, replaced.predictions[0].to_roads)
        self.assertNotEqual(original.predictions[0].waypoints, replaced.predictions[0].waypoints)

        self.assertEqual(
            replaced.predictions[0].waypoints,
            [(-5.0, 0.0), (0.0, 0.0), (5.0, 0.0)],
        )

        cfg = WindowConfig(window_s=1.0, stride_s=1.0, horizon_s=2.0,
                           payload_profile="compact")
        predictor_win = pick(build_windows(snaps, cfg)[0], "0-1")
        oracle_win = pick(build_windows(oracle, cfg)[0], "0-1")
        predictor_pairs = compact_window_json(predictor_win, self.scfg, cfg)["predicted_closest_pairs"]
        oracle_pairs = compact_window_json(oracle_win, self.scfg, cfg)["predicted_closest_pairs"]
        self.assertFalse(predictor_pairs[0][5])
        self.assertTrue(oracle_pairs[0][5])
        self.assertNotEqual(predictor_pairs, oracle_pairs)

    def test_ground_truth_oracle_keeps_subsecond_samples_to_data_end(self):
        """A +0.6 s collision endpoint must not be lost to +1 s sampling."""
        def scene(t, a_x, b_x):
            actors = []
            for actor_id, x, heading in (("A", a_x, 90.0), ("B", b_x, 270.0)):
                actors.append(ActorState(
                    actor_id=actor_id, kind="observed", cls="car", world_xy=(x, 0.0),
                    heading_deg=heading, speed_mps=10.0, accel_mps2=0.0,
                    placement=placement(), observed_by=["v1"],
                    predictions=[PredictedPath("straight", 1.0,
                        [(x, 0.0), (x - 2.0 if actor_id == "A" else x + 2.0, 0.0)],
                        1.0)],
                ))
            return SceneSnapshot(t=t, actors=actors, interactions=[])

        observed = [scene(5.0, -14.0, 14.0), scene(6.0, -4.0, 4.0)]
        raw = [scene(6.0 + i / 10, -4.0 + i, 4.0 - i) for i in range(7)]
        oracle = ground_truth_future_snapshots(
            observed, horizon_s=5.0, raw_snapshots=raw
        )
        path = oracle[-1].actors[0].predictions[0]
        self.assertEqual(path.waypoint_times_s, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.assertEqual(path.waypoints[-1], (2.0, 0.0))
        self.assertTrue(path.truncated)

        cfg = WindowConfig(window_s=1.0, horizon_s=5.0,
                           payload_profile="compact", future_source="ground_truth")
        win = pick(build_windows(oracle, cfg)[0], "5-6")
        doc = compact_window_json(win, self.scfg, cfg)
        self.assertEqual(doc["representation"], "compact_geometry_v3_timestamped_gt")
        self.assertEqual(doc["future_path_columns"][3], "waypoints_timed_enu_m")
        actor = next(a for a in doc["actors"] if a["id"] == "A")
        self.assertEqual(actor["future_paths"][0][3][0], [0.1, -3.0, 0.0])
        self.assertEqual(actor["future_paths"][0][3][-1], [0.6, 2.0, 0.0])
        self.assertTrue(doc["predicted_closest_pairs"][0][5])
        self.assertLessEqual(doc["predicted_closest_pairs"][0][3], 0.6)
        payload = build_window_payload(win, self.scfg, cfg)
        blob = json.dumps(payload, ensure_ascii=False)
        self.assertIn("waypoints_timed_enu_m", blob)
        self.assertIn("1초 간격이라고 가정하지 마십시오", blob)

        english_payload = build_window_payload(
            win, SerializeConfig(language="en"), cfg
        )
        english_blob = json.dumps(english_payload, ensure_ascii=False)
        self.assertIn("waypoints_timed_enu_m", english_blob)
        self.assertIn("[seconds_after_observation, east_m, north_m]", english_blob)
        self.assertIn("do not assume one-second spacing", english_blob)
        self.assertNotIn(
            "Successive waypoints in a future path represent +1s, +2s",
            english_blob,
        )

    def test_compact_keeps_legacy_external_predictor_starting_at_plus_one(self):
        """An external predictor may already omit t=0; do not drop its +1 point."""
        win = pick(self.wins, "2-7")
        target = next(a for a in win.last.actors if a.actor_id == "V002")
        target.predictions = [
            PredictedPath(
                maneuver="직진",
                probability=1.0,
                waypoints=[(3.3, -36.0), (3.3, -48.0)],
                horizon_s=2.0,
            )
        ]
        doc = compact_window_json(
            win, self.scfg, replace(self.cfg, payload_profile="compact")
        )
        actor = next(a for a in doc["actors"] if a["id"] == "V002")
        self.assertEqual(
            actor["future_paths"][0][3],
            [[3.3, -36.0], [3.3, -48.0]],
        )

    def test_compact_rejects_duplicate_json_block(self):
        cfg = replace(
            self.cfg, payload_profile="compact", include_json_block=True
        )
        with self.assertRaisesRegex(ValueError, "include_json_block"):
            build_window_payload(self.wins[0], self.scfg, cfg)

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
        q = build_question(pick(self.wins, "0-5"), self.cfg)  # t_end=5
        for k, iv in ((1, "(5, 6]"), (2, "(6, 7]"), (5, "(9, 10]")):
            self.assertIn(f"k={k}", q)
            self.assertIn(iv, q)
        self.assertIn("사고가 예상되지 않으면", q)
        self.assertIn("involved_actor_ids", q)

    def test_horizon_bucket_count(self):
        cfg = WindowConfig(window_s=5.0, horizon_s=3.0)
        self.assertEqual(cfg.n_horizon_buckets, 3)
        q = build_question(pick(self.wins, "0-5"), cfg)
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
        self.assertEqual(len(missing), 1)

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
        cfg = WindowConfig(window_s=5.0, stride_s=1.0, horizon_s=5.0,
                           warmup=True, full_window=True)
        with tempfile.TemporaryDirectory() as d:
            manifest = write_window_set(
                snaps, out_dir=d, scfg=SerializeConfig(), cfg=cfg,
                collision=None, scenario_id="split_x/TownX", town="TownX",
                scenario_split="split_x",
            )
            # 0-10 은 데이터 전체 구간 창 (full_window=True를 명시)
            for lbl in ("0-5", "1-6", "5-10", "0-10"):
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

            # 누적 창 0-1 … 0-5, 1-6 … 5-10 = 10개 + 전체 구간 창 0-10 = 11개
            self.assertEqual(manifest["summary"]["n_windows"], 11)
            labels = [w["label"] for w in manifest["windows"]]
            self.assertEqual(labels[:2], ["0-1", "0-2"])
            self.assertIn("0-5", labels)
            self.assertEqual(labels[-1], "0-10", "전체 구간 창이 마지막에 온다")
            self.assertTrue(manifest["config"]["full_window"])
            self.assertEqual(manifest["config"]["map_source"], "unspecified")
            self.assertEqual(manifest["config"]["payload_profile"], "standard")
            self.assertIsNone(
                manifest["config"]["map_built_from_scenario_trajectories"]
            )
            self.assertIn("output_schema", manifest)

    def test_manifest_records_map_provenance(self):
        snaps = make_snapshots(n=5, dt=0.5)
        with tempfile.TemporaryDirectory() as d:
            manifest = write_window_set(
                snaps,
                out_dir=d,
                map_source="OpenDRIVE (Town05.xodr)",
                map_built_from_scenario_trajectories=False,
            )
        self.assertEqual(
            manifest["config"]["map_source"], "OpenDRIVE (Town05.xodr)"
        )
        self.assertFalse(
            manifest["config"]["map_built_from_scenario_trajectories"]
        )

    def test_manifest_records_explicit_trajectory_map_override(self):
        snaps = make_snapshots(n=5, dt=0.5)
        with tempfile.TemporaryDirectory() as d:
            manifest = write_window_set(
                snaps,
                out_dir=d,
                map_source="궤적 합성 (차선 수는 관측 하한)",
                map_built_from_scenario_trajectories=True,
            )
        self.assertTrue(
            manifest["config"]["map_built_from_scenario_trajectories"]
        )

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
        return pick(wins, "0-3"), cfg

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
        pairs = closing_pairs(pick(wins, "0-5"), cfg)
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
            len(closing_pairs(pick(wins_a, "0-5"), auto)),
            len(closing_pairs(pick(wins_f, "0-5"), fixed)),
        )

    def test_must_include_survives_even_past_the_cap(self):
        """상한이 작아도 접촉 임박 쌍은 버리지 않는다."""
        from traffic_llm.accident_qa import closing_pairs

        snaps = self._crowded()
        tight = WindowConfig(window_s=5.0, max_closing_pairs=3,
                             closing_must_include_contact_s=0.0)
        wins, _ = build_windows(snaps, tight)
        w = pick(wins, "0-5")
        self.assertEqual(len(closing_pairs(w, tight)), 3)
        # A/B 의 접촉 예상이 5.0s 이므로 임계를 그 위로 올리면 반드시 포함된다
        loose = WindowConfig(window_s=5.0, max_closing_pairs=3,
                             closing_must_include_contact_s=6.0)
        pairs = closing_pairs(w, loose)
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


class TestEarlyCredit(unittest.TestCase):
    """조기 예측 인정 채점.

    사고 예측의 목적은 회피할 시간을 버는 것이므로, 실제 충돌 시각보다 조금 이른
    경보는 정답으로 본다. **늦은 예측은 여전히 오답**이다 — 이미 일어난 사고를
    예측이라 부를 수 없다.
    """

    N = 5

    def _gt(self, collision_s, early_credit_s=2.0):
        from traffic_llm.schemas import CollisionTruth

        snaps = make_snapshots(n=21, dt=0.5)
        cfg = WindowConfig(
            window_s=5.0, horizon_s=float(self.N), early_credit_s=early_credit_s
        )
        wins, _ = build_windows(snaps, cfg)
        w = pick(wins, "0-5")  # t_end = 5.0
        coll = (
            None
            if collision_s is None
            else CollisionTruth(
                occurred=True, carla_ids=(7001,), time_s=collision_s, frame=0
            )
        )
        return window_ground_truth(w, cfg, coll, {"v1": 1}, "s", "sp", "T"), w

    @staticmethod
    def _resp(*accident_ks, ids=("V002",), n=5):
        return {
            "predictions": [
                {
                    "k": k,
                    "interval_s": "",
                    "accident_expected": k in accident_ks,
                    "involved_actor_ids": list(ids) if k in accident_ks else [],
                    "reason": "",
                    "confidence": "low",
                }
                for k in range(1, n + 1)
            ]
        }

    def test_ground_truth_records_the_credit_window(self):
        gt, w = self._gt(8.2)  # t_end=5.0 → k_true = ceil(3.2) = 4
        c = gt["credit"]
        self.assertEqual(c["k_true"], 4)
        self.assertEqual((c["k_from"], c["k_to"]), (2, 4))
        self.assertEqual(c["early_credit_s"], 2.0)
        # 정답 자체는 **실제 구간만** '사고'로 둔다 (정답을 넓히지 않는다)
        hits = [e["k"] for e in gt["expected"] if e["accident_expected"]]
        self.assertEqual(hits, [4])

    def test_exact_and_early_are_both_correct(self):
        gt, _ = self._gt(8.2)
        for k, timing in ((4, "exact"), (3, "early"), (2, "early")):
            s = score_response(self._resp(k), gt)
            self.assertEqual(s["counts"], {"TP": 1, "FP": 0, "TN": 4, "FN": 0}, k)
            self.assertEqual(s["event"]["timing"], timing, k)
            self.assertEqual(s["accuracy"], 1.0, k)

    def test_too_early_is_wrong(self):
        """인정 범위를 넘어선 조기 예측은 오답이다."""
        gt, _ = self._gt(8.2)
        s = score_response(self._resp(1), gt)  # 3초 이름 > 2초 인정
        self.assertEqual(s["counts"]["FP"], 1)
        self.assertEqual(s["counts"]["FN"], 1)
        self.assertFalse(s["event"]["detected"])

    def test_late_is_still_wrong(self):
        gt, _ = self._gt(8.2)
        s = score_response(self._resp(5), gt)  # 1초 늦음
        self.assertEqual(s["counts"]["TP"], 0)
        self.assertEqual(s["counts"]["FP"], 1)
        self.assertEqual(s["counts"]["FN"], 1)
        stat = [r["status"] for r in s["per_interval"] if r["k"] == 5][0]
        self.assertEqual(stat, "FP_LATE")

    def test_missing_the_event_is_fn(self):
        gt, _ = self._gt(8.2)
        s = score_response(self._resp(), gt)
        self.assertEqual(s["counts"], {"TP": 0, "FP": 0, "TN": 4, "FN": 1})
        self.assertEqual(s["event"]["timing"], "none")

    def test_event_counts_once_not_per_bucket(self):
        """인정 범위를 전부 '사고'로 칠해도 TP 는 1 이다.

        구간마다 세면 범위를 다 칠한 답이 정확히 한 구간만 맞힌 답보다 TP 를 더
        받아, 같은 정답에 다른 점수가 나온다.
        """
        gt, _ = self._gt(8.2)
        one = score_response(self._resp(4), gt)
        allw = score_response(self._resp(2, 3, 4), gt)
        self.assertEqual(one["counts"]["TP"], 1)
        self.assertEqual(allw["counts"]["TP"], 1)
        # 범위 밖까지 칠하면 그것만 FP 로 늘어난다
        over = score_response(self._resp(1, 2, 3, 4), gt)
        self.assertEqual(over["counts"]["TP"], 1)
        self.assertEqual(over["counts"]["FP"], 3)

    def test_exact_bucket_is_preferred_as_the_credited_call(self):
        gt, _ = self._gt(8.2)
        s = score_response(self._resp(2, 4), gt)
        self.assertEqual(s["event"]["credited_k"], 4)
        self.assertEqual(s["event"]["timing"], "exact")

    def test_zero_credit_is_strict(self):
        gt, _ = self._gt(8.2, early_credit_s=0.0)
        self.assertEqual((gt["credit"]["k_from"], gt["credit"]["k_to"]), (4, 4))
        self.assertEqual(score_response(self._resp(4), gt)["counts"]["TP"], 1)
        s = score_response(self._resp(3), gt)
        self.assertEqual(s["counts"]["TP"], 0)
        self.assertEqual(s["counts"]["FP"], 1)

    def test_override_rescoring(self):
        """같은 응답을 다른 기준으로 다시 채점할 수 있어야 한다."""
        gt, _ = self._gt(8.2, early_credit_s=0.0)
        r = self._resp(3)
        self.assertEqual(score_response(r, gt)["counts"]["TP"], 0)
        self.assertEqual(score_response(r, gt, 2.0)["counts"]["TP"], 1)

    def test_credit_boundary_follows_the_collision_time(self):
        """인정 경계는 `충돌시각 − x` 가 속한 구간이다 — 구간 번호 산술이 아니다.

        구간이 1초 폭이라 충돌이 구간 안 어디에 있느냐로 경계가 달라진다.
        충돌 8.2s(구간 4의 앞부분)면 0.5초 인정으로도 구간 3 이 들어오지만,
        충돌 8.9s(구간 4의 뒷부분)면 들어오지 않는다.
        """
        near, _ = self._gt(8.2, early_credit_s=0.5)   # 8.2-0.5=7.7 → 구간 3
        self.assertEqual(near["credit"]["k_from"], 3)
        far, _ = self._gt(8.9, early_credit_s=0.5)    # 8.9-0.5=8.4 → 구간 4
        self.assertEqual(far["credit"]["k_from"], 4)
        self.assertEqual(score_response(self._resp(3), near)["counts"]["TP"], 1)
        self.assertEqual(score_response(self._resp(3), far)["counts"]["FP"], 1)

    def test_override_matches_the_generated_window(self):
        """오버라이드로 계산한 범위가 정답 파일 생성 시와 같아야 한다."""
        for c_time in (5.5, 8.2, 8.9, 9.99):
            for x in (0.0, 0.5, 1.0, 2.0, 3.5):
                baked, _ = self._gt(c_time, early_credit_s=x)
                strict, _ = self._gt(c_time, early_credit_s=0.0)
                a = score_response(self._resp(2, 3, 4), baked)["credit"]["buckets"]
                b = score_response(self._resp(2, 3, 4), strict, x)["credit"]["buckets"]
                self.assertEqual(a, b, f"c_time={c_time} x={x}")

    def test_no_accident_scenario_is_unchanged(self):
        gt, _ = self._gt(None)
        self.assertIsNone(gt["credit"]["k_true"])
        s = score_response(self._resp(), gt)
        self.assertEqual(s["counts"], {"TP": 0, "FP": 0, "TN": self.N, "FN": 0})
        self.assertIsNone(s["event"])
        s2 = score_response(self._resp(2), gt)
        self.assertEqual(s2["counts"]["FP"], 1)
        self.assertEqual(s2["counts"]["TN"], self.N - 1)

    def test_credit_window_is_clipped_to_the_horizon(self):
        """충돌이 첫 구간이면 인정 범위가 지평 앞으로 넘어가지 않는다."""
        gt, _ = self._gt(5.5)  # t_end=5.0 → k_true = 1
        self.assertEqual((gt["credit"]["k_from"], gt["credit"]["k_to"]), (1, 1))
        s = score_response(self._resp(1), gt)
        self.assertEqual(s["counts"]["TP"], 1)

    def test_vehicle_scoring_uses_the_credited_call(self):
        """이르게 맞힌 구간의 차량 지목으로 채점해야 한다."""
        gt, _ = self._gt(8.2)
        s = score_response(self._resp(2, ids=("V002",)), gt)
        row = [r for r in s["per_interval"] if r["k"] == 2][0]
        self.assertEqual(row["vehicle_recall"], 1.0)
        bad = score_response(self._resp(2, ids=("EGO_v1",)), gt)
        row = [r for r in bad["per_interval"] if r["k"] == 2][0]
        self.assertEqual(row["vehicle_recall"], 0.0)

    def test_aggregate_reports_event_timing(self):
        gt, _ = self._gt(8.2)
        scores = [
            score_response(self._resp(4), gt),
            score_response(self._resp(3), gt),
            score_response(self._resp(), gt),
        ]
        agg = aggregate_scores(scores)
        ev = agg["events"]
        self.assertEqual(ev["n"], 3)
        self.assertEqual((ev["exact"], ev["early"], ev["none"]), (1, 1, 1))
        self.assertAlmostEqual(ev["detection_rate"], 0.667, places=2)

    def test_legacy_ground_truth_without_credit_is_strict(self):
        """`credit` 가 없는 예전 정답 파일은 조용히 관대해지지 않는다."""
        gt, _ = self._gt(8.2)
        del gt["credit"]
        s = score_response(self._resp(3), gt)
        self.assertEqual(s["counts"]["TP"], 0)
        self.assertEqual(s["counts"]["FP"], 1)


class TestUnscorableBeyondData(unittest.TestCase):
    """원본 데이터가 끝난 뒤의 구간은 채점하지 않는다.

    질문은 항상 미래 N초를 묻지만, 데이터가 9.2초에 끝났다면 (10,11] 구간의
    '사고 없음' 은 확인된 사실이 아니라 질문 형식을 맞추려고 채운 값이다.
    그것으로 점수를 주면 모델이 관측되지 않은 미래를 맞혔다고 계산된다.
    """

    def _gt(self, n_snaps, collision_s, window_label="0-5", horizon=5.0):
        from traffic_llm.schemas import CollisionTruth

        snaps = make_snapshots(n=n_snaps, dt=0.5)
        cfg = WindowConfig(window_s=5.0, stride_s=1.0, horizon_s=horizon)
        wins, _ = build_windows(snaps, cfg)
        w = pick(wins, window_label)
        coll = (
            None
            if collision_s is None
            else CollisionTruth(
                occurred=True, carla_ids=(7001,), time_s=collision_s, frame=0
            )
        )
        return window_ground_truth(
            w, cfg, coll, {"v1": 1}, "s", "sp", "T",
            data_end_s=max(s.t for s in snaps),
        ), cfg

    @staticmethod
    def _resp(*ks, n=5):
        return {
            "predictions": [
                {"k": k, "interval_s": "", "accident_expected": k in ks,
                 "involved_actor_ids": ["V002"] if k in ks else [],
                 "reason": "", "confidence": "low"}
                for k in range(1, n + 1)
            ]
        }

    def test_buckets_past_the_data_end_are_marked(self):
        # 데이터 0~5초(n=11), 창 0-5 → 구간 1~5 = (5,6]…(9,10] 전부 데이터 밖
        gt, _ = self._gt(11, None)
        self.assertEqual(gt["data_end_s"], 5.0)
        self.assertEqual([e["scorable"] for e in gt["expected"]], [False] * 5)
        self.assertEqual(gt["n_scorable"], 0)
        for e in gt["expected"]:
            self.assertEqual(e["unscorable_reason"], "beyond_data_end")

    def test_buckets_inside_the_data_are_scorable(self):
        # 데이터 0~10초(n=21), 창 0-5 → 구간 1~5 = (5,6]…(9,10] 전부 데이터 안
        gt, _ = self._gt(21, None)
        self.assertEqual(gt["data_end_s"], 10.0)
        self.assertEqual([e["scorable"] for e in gt["expected"]], [True] * 5)

    def test_collision_bucket_stays_scorable_past_the_end(self):
        """데이터가 충돌과 함께 끝나도 그 구간은 답을 안다."""
        # 데이터 0~9.5초(n=20), 충돌 9.2 → 창 3-8 의 구간 2 = (9,10] 은 데이터
        # 끝(9.5)을 넘지만 충돌을 담고 있다
        gt, _ = self._gt(20, 9.2, window_label="3-8")
        by = {e["k"]: e for e in gt["expected"]}
        self.assertTrue(by[2]["accident_expected"])
        self.assertTrue(by[2]["scorable"])
        self.assertEqual(by[2]["unscorable_reason"], "")
        # 그 뒤 구간들은 확인 불가
        self.assertFalse(by[3]["scorable"])
        self.assertFalse(by[4]["scorable"])

    def test_excluded_buckets_do_not_enter_the_counts(self):
        gt, _ = self._gt(20, 9.2, window_label="3-8")
        s = score_response(self._resp(2), gt)
        self.assertEqual(s["excluded_buckets"], [3, 4, 5])
        self.assertEqual(s["scored_buckets"], [1, 2])
        # 채점 가능한 두 bucket 모두 분모에 포함된다.
        self.assertEqual(s["counts"], {"TP": 1, "FP": 0, "TN": 1, "FN": 0})
        self.assertEqual(s["accuracy"], 1.0)

    def test_wrong_answers_in_excluded_buckets_are_not_penalised(self):
        """확인할 수 없는 구간의 오답으로 벌점을 주면 안 된다."""
        gt, _ = self._gt(20, 9.2, window_label="3-8")
        clean = score_response(self._resp(2), gt)
        noisy = score_response(self._resp(2, 4, 5), gt)  # 4,5 는 데이터 밖
        self.assertEqual(clean["counts"], noisy["counts"])
        stats = {r["k"]: r["status"] for r in noisy["per_interval"]}
        self.assertEqual(stats[4], "UNSCORABLE")
        self.assertEqual(stats[5], "UNSCORABLE")

    def test_unscorable_rows_record_what_the_model_said(self):
        """제외해도 모델이 뭐라고 답했는지는 남겨야 진단이 된다."""
        gt, _ = self._gt(20, 9.2, window_label="3-8")
        s = score_response(self._resp(2, 4), gt)
        row = [r for r in s["per_interval"] if r["k"] == 4][0]
        self.assertTrue(row["pred"])
        self.assertEqual(row["unscorable_reason"], "beyond_data_end")

    def test_credit_window_is_restricted_to_scorable(self):
        gt, _ = self._gt(20, 9.2, window_label="3-8")
        s = score_response(self._resp(1), gt)
        self.assertTrue(set(s["credit"]["buckets"]) <= {1, 2})

    def test_no_data_end_scores_everything(self):
        """data_end_s 를 주지 않으면 예전처럼 전 구간을 채점한다."""
        from traffic_llm.schemas import CollisionTruth

        snaps = make_snapshots(n=20, dt=0.5)
        cfg = WindowConfig(window_s=5.0, stride_s=1.0)
        w = pick(build_windows(snaps, cfg)[0], "3-8")
        gt = window_ground_truth(
            w, cfg,
            CollisionTruth(occurred=True, carla_ids=(7001,), time_s=9.2, frame=0),
            {"v1": 1}, "s", "sp", "T",
        )
        self.assertIsNone(gt["data_end_s"])
        self.assertTrue(all(e["scorable"] for e in gt["expected"]))
        s = score_response(self._resp(2), gt)
        self.assertEqual(s["excluded_buckets"], [])


class TestStaleOutputCleanup(unittest.TestCase):
    """재생성이 자기 파일만 지우는지.

    창 구성이 바뀌면(누적 창 도입, 지평 변경) 예전 라벨의 payload 가 남아 새
    것과 섞인다. 그래서 호출자가 디렉터리째 지우곤 했는데, 그러면 `responses/`
    의 LLM 응답까지 날아간다 — 실제로 두 번 잃었다.
    """

    def _write(self, d, cfg):
        return write_window_set(
            make_snapshots(n=21, dt=0.5), d, SerializeConfig(), cfg
        )

    def test_stale_payloads_are_removed(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._write(d, WindowConfig(window_s=5.0, warmup=False))
            before = {os.path.basename(f) for f in
                      glob.glob(os.path.join(d, "llm_payload_*.json"))}
            self.assertIn("llm_payload_0-5.json", before)
            # 누적 창으로 다시 만들면 0-1 … 이 생기고 0-5 도 여전히 있다
            self._write(d, WindowConfig(window_s=3.0, warmup=True))
            after = {os.path.basename(f) for f in
                     glob.glob(os.path.join(d, "llm_payload_*.json"))}
            self.assertIn("llm_payload_0-1.json", after)
            self.assertNotIn("llm_payload_0-5.json", after,
                             "예전 창의 payload 가 남았다")

    def test_responses_survive_regeneration(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._write(d, WindowConfig(window_s=5.0))
            rdir = os.path.join(d, "responses")
            os.makedirs(rdir)
            keep = os.path.join(rdir, "response_0-5.json")
            with open(keep, "w", encoding="utf-8") as f:
                json.dump({"answer": {}}, f)
            self._write(d, WindowConfig(window_s=4.0))
            self.assertTrue(os.path.isfile(keep), "LLM 응답이 삭제됐다")

    def test_other_files_are_left_alone(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            note = os.path.join(d, "NOTES.md")
            with open(note, "w", encoding="utf-8") as f:
                f.write("keep me")
            self._write(d, WindowConfig(window_s=5.0))
            self.assertTrue(os.path.isfile(note))


class TestPredictedPathTextRendering(unittest.TestCase):
    """자연어에서 같은 기동의 여러 경로를 구별할 수 있는지.

    라벨과 확률만 쓰면 `직진 50%, 직진 50%` 처럼 똑같이 보인다 — 실제로는 진출로가
    다른 별개 경로다. 구조화 JSON 블록이 기본 꺼짐이므로 여기서 구별하지 않으면
    구별할 방법이 없다.
    """

    @staticmethod
    def _actor(preds):
        return ActorState(
            actor_id="V001", kind="observed", cls="car", world_xy=(0.0, 0.0),
            heading_deg=0.0, speed_mps=10.0, accel_mps2=0.0,
            placement=placement(), observed_by=["v1"], predictions=preds,
        )

    def _text(self, preds, lang="ko"):
        from traffic_llm.schemas import SceneSnapshot
        from traffic_llm.serialize import to_text

        snap = SceneSnapshot(
            t=0.0, actors=[self._actor(preds)], interactions=[],
            area_name="X", map_context={}, scenario=None, frame_idx=1,
        )
        return to_text(snap, SerializeConfig(language=lang))

    @staticmethod
    def _p(maneuver, prob, roads):
        from traffic_llm.schemas import PredictedPath

        return PredictedPath(
            maneuver, prob, [(0.0, 0.0), (0.0, 10.0)], 5.0, to_roads=list(roads)
        )

    def test_duplicate_labels_get_the_target_road(self):
        txt = self._text([
            self._p("직진", 0.5, ["Road 2065"]),
            self._p("직진", 0.5, ["Road 2071"]),
        ])
        self.assertIn("Road 2065", txt)
        self.assertIn("Road 2071", txt)
        self.assertNotIn("직진 50%, 직진 50%", txt)

    def test_distinct_labels_stay_clean(self):
        """라벨이 다르면 이미 구별되므로 도로명을 붙이지 않는다."""
        txt = self._text([
            self._p("직진", 0.6, ["Road A"]),
            self._p("좌회전", 0.4, ["Road B"]),
        ])
        self.assertNotIn("Road A", txt)
        self.assertNotIn("Road B", txt)
        self.assertIn("직진 60%", txt)
        self.assertIn("좌회전 40%", txt)

    def test_single_path_needs_no_percentage(self):
        txt = self._text([self._p("직진", 1.0, ["Road A"])])
        self.assertIn("예상경로: 직진", txt)
        self.assertNotIn("100%", txt)

    def test_english_rendering(self):
        txt = self._text([
            self._p("직진", 0.5, ["Road 2065"]),
            self._p("직진", 0.5, ["Road 2071"]),
        ], lang="en")
        self.assertIn("straight 50%", txt)
        self.assertIn("Road 2065", txt)
        self.assertNotIn("직진", txt)

    def test_missing_to_roads_does_not_crash(self):
        txt = self._text([
            self._p("직진", 0.5, []),
            self._p("직진", 0.5, []),
        ])
        self.assertIn("직진 50%", txt)


class TestScoreModes(unittest.TestCase):
    """다중 기준 채점.

    같은 응답을 여러 기준으로 채점하므로, 모드 간 숫자 차이는 전부 **기준의
    차이**여야 한다. 구간별 답에서 바이너리는 유도되지만 그 반대는 불가능하다는
    것이 이 설계의 전제다.
    """

    N = 5

    def _gt(self, collision_s, early_credit_s=2.0, data_end_s=None):
        from traffic_llm.schemas import CollisionTruth

        snaps = make_snapshots(n=21, dt=0.5)
        cfg = WindowConfig(
            window_s=5.0, horizon_s=float(self.N), early_credit_s=early_credit_s
        )
        wins, _ = build_windows(snaps, cfg)
        w = pick(wins, "0-5")  # t_end = 5.0
        coll = (
            None
            if collision_s is None
            else CollisionTruth(
                occurred=True, carla_ids=(7001,), time_s=collision_s, frame=0
            )
        )
        return window_ground_truth(
            w, cfg, coll, {"v1": 1}, "s", "sp", "T", data_end_s=data_end_s
        )

    @staticmethod
    def _resp(*accident_ks, ids=("V002",), n=5):
        return {
            "predictions": [
                {
                    "k": k,
                    "interval_s": "",
                    "accident_expected": k in accident_ks,
                    "involved_actor_ids": list(ids) if k in accident_ks else [],
                    "reason": "",
                    "confidence": "low",
                }
                for k in range(1, n + 1)
            ]
        }

    # ---------------------------------------------------------- binary

    def test_binary_scores_each_bucket(self):
        gt = self._gt(8.2)  # k_true = 4
        for ks in ((4,), (1,), (5,), (1, 2, 3, 4, 5)):
            b = score_binary(self._resp(*ks), gt)
            self.assertEqual(b["status"], "TP", ks)
            self.assertEqual(sum(b["counts"].values()), 5, ks)

    def test_binary_example_is_tn_tn_tn_tp_tn(self):
        """사고 bucket 하나만 맞히면 분모 5에 TP 1/TN 4가 된다."""
        b = score_binary(self._resp(4), self._gt(8.2))
        self.assertEqual(
            [r["status"] for r in b["per_interval"]],
            ["TN", "TN", "TN", "TP", "TN"],
        )
        self.assertEqual(b["counts"], {"TP": 1, "FP": 0, "TN": 4, "FN": 0})
        self.assertEqual(b["n_buckets"], 5)

    def test_binary_counts_one_window_not_one_bucket(self):
        """판정 단위가 윈도우여야 DeepAccident 와 같은 축이 된다."""
        b = score_binary(self._resp(2, 3, 4), self._gt(8.2))
        self.assertEqual(sum(b["counts"].values()), 5)

    def test_binary_no_collision_window(self):
        gt = self._gt(None)
        self.assertEqual(score_binary(self._resp(), gt)["status"], "TN")
        self.assertEqual(score_binary(self._resp(3), gt)["status"], "FP")

    def test_binary_missed_is_fn(self):
        self.assertEqual(score_binary(self._resp(), self._gt(8.2))["status"], "FN")

    def test_binary_collision_beyond_horizon_is_negative(self):
        """충돌이 지평 밖이면 이 창의 정답은 '사고 없음' 이다."""
        gt = self._gt(12.0)  # t_end=5.0, 지평 5초 → k_true = 7 > 5
        self.assertEqual(score_binary(self._resp(), gt)["status"], "TN")
        self.assertEqual(score_binary(self._resp(5), gt)["status"], "FP")

    def test_binary_reports_spray(self):
        """도배를 벌하지 않으므로 몇 칸을 칠했는지 같이 내야 한다."""
        gt = self._gt(8.2)
        self.assertEqual(score_binary(self._resp(4), gt)["n_positive_buckets"], 1)
        self.assertEqual(
            score_binary(self._resp(1, 2, 3, 4, 5), gt)["n_positive_buckets"], 5
        )

    def test_binary_collapses_positive_from_unscorable_timing_bucket(self):
        """binary는 시점을 버리므로 요청 지평 어디의 경보든 양성이다."""
        gt = self._gt(None, data_end_s=7.0)  # k=3,4,5 는 채점 불가
        b = score_binary(self._resp(4), gt)
        self.assertEqual(b["n_scorable"], 2)
        self.assertEqual(b["status"], "FP")
        self.assertEqual(b["positive_ks"], [4])

    def test_binary_wrong_timing_still_detects_known_collision(self):
        gt = self._gt(6.2, data_end_s=7.0)  # 실제 충돌 k=2, k=4는 시점 채점 불가
        b = score_binary(self._resp(4), gt)
        self.assertEqual(b["status"], "TP")
        self.assertEqual(b["positive_ks"], [4])

    def test_binary_scores_window_when_all_timing_buckets_are_unscorable(self):
        """DeepAccident binary 축에서는 생성된 창마다 판정이 한 건이다."""
        gt = self._gt(None, data_end_s=5.0)
        self.assertEqual(gt["n_scorable"], 0)
        tn = score_binary(self._resp(), gt)
        fp = score_binary(self._resp(5), gt)
        self.assertFalse(tn["scorable"])
        self.assertEqual(tn["status"], "UNSCORABLE")
        self.assertEqual(fp["status"], "UNSCORABLE")
        self.assertEqual(tn["n_scorable"], 0)

    # ---------------------------------------------------------- weighted

    def test_weighted_early_gets_full_credit(self):
        gt = self._gt(8.2)  # k_true = 4
        for k in (1, 2, 3, 4):
            w = score_weighted(self._resp(k), gt)
            self.assertEqual(w["score"], 1.0, k)

    def test_weighted_late_decays_by_step(self):
        gt = self._gt(6.2)  # k_true = 2
        self.assertEqual(score_weighted(self._resp(3), gt)["score"], 0.8)
        self.assertEqual(score_weighted(self._resp(4), gt)["score"], 0.6)
        self.assertEqual(score_weighted(self._resp(5), gt)["score"], 0.4)

    def test_weighted_uses_first_alarm_not_the_lucky_bucket(self):
        """대표는 첫 경보다 — 범위를 칠했다고 Δ=0 이 되면 가중이 무의미하다."""
        gt = self._gt(6.2)  # k_true = 2
        w = score_weighted(self._resp(2, 3, 4, 5), gt)
        self.assertEqual(w["alarm_k"], 2)
        self.assertEqual(w["delta"], 0)
        late = score_weighted(self._resp(3, 4), gt)
        self.assertEqual(late["alarm_k"], 3)
        self.assertEqual(late["delta"], 1)

    def test_weighted_missed_is_zero(self):
        w = score_weighted(self._resp(), self._gt(8.2))
        self.assertEqual(w["score"], 0.0)
        self.assertFalse(w["detected"])

    def test_weighted_undefined_without_collision(self):
        gt = self._gt(None)
        w = score_weighted(self._resp(3), gt)
        self.assertFalse(w["applicable"])
        self.assertIsNone(w["score"])
        self.assertTrue(w["false_alarm"])
        self.assertFalse(score_weighted(self._resp(), gt)["false_alarm"])

    def test_weighted_early_decay_is_configurable(self):
        gt = self._gt(8.2)  # k_true = 4
        w = score_weighted(self._resp(1), gt, early_decay=0.2)
        self.assertEqual(w["delta"], -3)
        self.assertAlmostEqual(w["score"], 0.4)

    def test_weighted_never_goes_negative(self):
        gt = self._gt(5.5)  # k_true = 1
        self.assertEqual(score_weighted(self._resp(5), gt, late_decay=0.5)["score"], 0.0)

    # ---------------------------------------------------------- score_modes

    def test_meeting_example_window_0_5_collision_at_7(self):
        """회의 예시를 고정한다: 0~5초 관측, 실제 충돌 t=7 (정답 k=2).

        바이너리는 미래 5초 안의 경보면 맞고, strict 는 k=2만 맞으며,
        early 는 기본 2초 조기 경보(k=1)를 인정한다. k=3은 한 칸 늦어
        weighted 에서만 0.8의 부분 점수를 받는다.
        """
        gt = self._gt(7.0)
        cases = (
            # alarms, binary, strict TP, early TP, weighted
            ((),   "FN", 0, 0, 0.0),
            ((1,), "TP", 0, 1, 1.0),
            ((2,), "TP", 1, 1, 1.0),
            ((3,), "TP", 0, 0, 0.8),
        )
        for alarms, binary, strict_tp, early_tp, weighted in cases:
            with self.subTest(alarms=alarms):
                scores = score_modes(self._resp(*alarms), gt)
                self.assertEqual(scores["binary"]["status"], binary)
                self.assertEqual(scores["strict"]["counts"]["TP"], strict_tp)
                self.assertEqual(scores["early"]["counts"]["TP"], early_tp)
                self.assertEqual(scores["weighted"]["score"], weighted)

    def test_modes_disagree_where_the_criteria_disagree(self):
        """늦은 경보: binary 는 맞고, early 는 틀리고, weighted 는 부분 점수."""
        gt = self._gt(6.2)  # k_true = 2
        m = score_modes(self._resp(4), gt)
        self.assertEqual(m["binary"]["status"], "TP")
        self.assertEqual(m["early"]["counts"]["TP"], 0)
        self.assertEqual(m["early"]["counts"]["FP"], 1)
        self.assertEqual(m["weighted"]["score"], 0.6)

    def test_strict_is_stricter_than_early(self):
        gt = self._gt(8.2)  # k_true = 4, 인정 범위 2~4
        m = score_modes(self._resp(2), gt)
        self.assertEqual(m["strict"]["counts"]["TP"], 0)
        self.assertEqual(m["early"]["counts"]["TP"], 1)

    def test_early_mode_matches_score_response(self):
        """모드 래퍼가 기존 채점을 바꾸지 않는다."""
        gt = self._gt(8.2)
        r = self._resp(3)
        self.assertEqual(score_modes(r, gt)["early"], score_response(r, gt))

    def test_modes_subset_is_honored(self):
        m = score_modes(self._resp(3), self._gt(8.2), modes=("binary",))
        self.assertIn("binary", m)
        for k in ("strict", "early", "weighted"):
            self.assertNotIn(k, m)

    def test_named_binary_axes_are_separate(self):
        gt = self._gt(8.2)
        m = score_modes(self._resp(4), gt)
        self.assertEqual(m["binary_window"]["counts"],
                         {"TP": 1, "FP": 0, "TN": 0, "FN": 0})
        self.assertEqual(m["binary_bucket"]["counts"],
                         {"TP": 1, "FP": 0, "TN": 4, "FN": 0})
        # Window binary ignores timing within the future horizon.
        self.assertEqual(score_modes(self._resp(1), gt)["binary_window"]["status"], "TP")

    def test_binary_includes_alarm_after_recording_end_in_known_positive(self):
        gt = self._gt(6.0, data_end_s=6.0)
        gt["scenario"].update(ref="truncated-accident", outcome="accident")
        m = score_modes(self._resp(3), gt)
        self.assertEqual(m["binary_window"]["status"], "TP")
        self.assertEqual(m["binary_window"]["eligible_buckets"], [1, 2, 3, 4, 5])
        self.assertEqual(m["binary_window"]["n_positive_buckets"], 1)
        self.assertEqual(aggregate_modes([m])["binary_scenario"]["counts"],
                         {"TP": 1, "FP": 0, "TN": 0, "FN": 0})
        # Unknown timing labels remain excluded from bucket-level metrics.
        for mode in ("binary_bucket", "strict", "early"):
            self.assertEqual(m[mode]["counts"]["TP"], 0)
            self.assertEqual(m[mode]["counts"]["FN"], 1)

    def test_binary_known_positive_requires_alarm_inside_requested_horizon(self):
        gt = self._gt(6.0, data_end_s=6.0)
        for response in (self._resp(), self._resp(6, n=6)):
            with self.subTest(response=response):
                m = score_modes(response, gt)["binary_window"]
                self.assertEqual(m["status"], "FN")
                self.assertEqual(m["n_positive_buckets"], 0)

    def test_binary_truncated_negative_is_excluded_even_with_alarm(self):
        gt = self._gt(None, data_end_s=7.0)
        gt["scenario"].update(ref="truncated-normal", outcome="normal")
        for response in (self._resp(), self._resp(1), self._resp(5)):
            with self.subTest(response=response):
                m = score_modes(response, gt)
                self.assertFalse(m["binary_window"]["scorable"])
                self.assertEqual(m["binary_window"]["n_windows"], 0)
                self.assertEqual(aggregate_modes([m])["binary_scenario"]["counts"],
                                 {"TP": 0, "FP": 0, "TN": 0, "FN": 0})

    def test_binary_complete_negative_counts_full_horizon_alarm(self):
        gt = self._gt(None, data_end_s=10.0)
        self.assertEqual(score_modes(self._resp(5), gt)["binary_window"]["status"], "FP")
        self.assertEqual(score_modes(self._resp(), gt)["binary_window"]["status"], "TN")

    def test_binary_scenario_collapses_overlapping_windows(self):
        gt1 = self._gt(8.2)
        gt2 = self._gt(8.2)
        gt1["scenario"].update(ref="acc-1", outcome="accident")
        gt2["scenario"].update(ref="acc-1", outcome="accident")
        rows = [score_modes(self._resp(4), gt1),
                score_modes(self._resp(4), gt2)]
        a = aggregate_modes(rows)
        self.assertEqual(a["binary_window"]["counts"]["TP"], 2)
        self.assertEqual(a["binary_scenario"]["counts"],
                         {"TP": 1, "FP": 0, "TN": 0, "FN": 0})
        self.assertEqual(a["binary_scenario"]["n_scenarios"], 1)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            score_modes(self._resp(3), self._gt(8.2), modes=("bogus",))

    def test_credit_override_moves_only_the_early_mode(self):
        gt = self._gt(8.2)  # k_true = 4
        m = score_modes(self._resp(1), gt, credit_s=4.0)  # 1번까지 인정
        self.assertEqual(m["early"]["counts"]["TP"], 1)
        self.assertEqual(m["strict"]["counts"]["TP"], 0)

    # ---------------------------------------------------------- aggregate

    def test_aggregate_modes_sums_each_axis(self):
        acc, non = self._gt(6.2), self._gt(None)  # k_true = 2
        rows = [
            score_modes(self._resp(2), acc),   # 정확
            score_modes(self._resp(4), acc),   # 늦음 → 0.6
            score_modes(self._resp(), acc),    # 미검출 → 0.0
            score_modes(self._resp(3), non),   # 오경보
            score_modes(self._resp(), non),    # 정상
        ]
        a = aggregate_modes(rows)
        self.assertEqual(a["n_windows"], 5)
        self.assertEqual(
            a["binary"]["counts"], {"TP": 1, "FP": 2, "TN": 20, "FN": 2}
        )
        w = a["weighted"]
        self.assertEqual(w["n_event_windows"], 3)
        self.assertAlmostEqual(w["mean_score"], round((1.0 + 0.6 + 0.0) / 3, 3))
        self.assertEqual((w["n_detected"], w["n_missed"]), (2, 1))
        self.assertEqual(w["n_no_event_windows"], 2)
        self.assertEqual(w["false_alarms"], 1)
        self.assertEqual(w["false_alarm_rate"], round(2 / 22, 3))
        self.assertEqual(w["window_false_alarm_rate"], 0.5)

    def test_aggregate_modes_reports_spray_average(self):
        gt = self._gt(6.2)
        a = aggregate_modes([score_modes(self._resp(1, 2, 3), gt),
                             score_modes(self._resp(2), gt)])
        self.assertEqual(a["binary"]["mean_positive_buckets"], 2.0)

    def test_aggregate_modes_empty(self):
        a = aggregate_modes([])
        self.assertEqual(a["n_windows"], 0)
        self.assertEqual(a["binary"]["counts"]["TP"], 0)
