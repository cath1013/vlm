"""LLM 입력 언어 전환 테스트.

기본은 한국어, `SerializeConfig.language='en'` 이면 **모델에게 가는 모든
문자열**이 영어여야 한다. 사람이 눈으로 확인할 수 없는 항목(JSON 안의 라벨,
스키마 설명, 정답 주석)이 빠지기 쉬우므로, 산출물 전체를 훑어 한글이 남았는지
검사한다.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_llm import i18n
from traffic_llm.accident_qa import (
    WindowConfig,
    build_question,
    build_window_payload,
    build_windows,
    render_window_text,
    window_ground_truth,
    write_window_set,
)
from traffic_llm.config import SerializeConfig
from traffic_llm.schemas import (
    ActorState,
    CollisionTruth,
    InfraState,
    Interaction,
    PredictedPath,
    RoadPlacement,
    ScenarioContext,
    SceneSnapshot,
)
from traffic_llm.serialize import build_messages, to_json, to_text

HANGUL = re.compile("[가-힣]")


def hangul_in(obj) -> list:
    """자료구조 전체에서 한글이 든 문자열을 찾는다."""
    found = []

    def walk(x, path="$"):
        if isinstance(x, str):
            if HANGUL.search(x):
                found.append((path, x[:60]))
        elif isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and HANGUL.search(k):
                    found.append((path, k))
                walk(v, f"{path}.{k}")
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")

    walk(obj)
    return found


def placement(direction="북행", confident=True) -> RoadPlacement:
    return RoadPlacement(
        road_id="r1",
        road_name="High St",
        s_m=10.0,
        lateral_offset_m=-1.75,
        direction_label=direction,
        bearing_deg=0.0,
        lane_index=1,
        lane_count=2,
        speed_limit_kph=50.0,
        dist_to_next_junction_m=40.0,
        next_junction_id="J1",
        direction_confident=confident,
    )


def actor(aid, kind="observed", maneuver="차선유지", **kw) -> ActorState:
    a = ActorState(
        actor_id=aid,
        kind=kind,
        cls=kw.pop("cls", "car"),
        world_xy=kw.pop("xy", (0.0, 0.0)),
        heading_deg=kw.pop("heading", 0.0),
        speed_mps=kw.pop("speed", 10.0),
        accel_mps2=kw.pop("accel", -1.2),
        observed_by=["V1"],
        placement=kw.pop("placement", placement()),
        maneuver=maneuver,
    )
    a.predictions = kw.pop(
        "predictions",
        [
            PredictedPath("직진", 0.6, [(0.0, 10.0)], 3.0),
            PredictedPath("좌회전", 0.4, [(-5.0, 8.0)], 3.0),
        ],
    )
    a.observed_range_m = kw.pop("rng", 45.0)
    return a


def snapshot(t=0.0) -> SceneSnapshot:
    ego = actor("EGO_V1", kind="ego", maneuver="가속 중", xy=(0.0, -20.0))
    other = actor("V002", maneuver="차선변경(좌)", xy=(3.5, 5.0))
    ped = actor(
        "P003",
        cls="person",
        maneuver="속도 미확정",
        xy=(-8.0, 2.0),
        speed=None,
        placement=placement(direction="남행", confident=False),
        predictions=[PredictedPath("정지 유지", 1.0, [(-8.0, 2.0)], 3.0)],
    )
    snap = SceneSnapshot(
        t=t,
        actors=[ego, other, ped],
        interactions=[
            Interaction(
                kind="following",
                subject_id="EGO_V1",
                object_id="V002",
                gap_m=18.0,
                headway_s=1.8,
                ttc_s=4.2,
                note="선행차 추종",
            ),
            Interaction(
                kind="crossing",
                subject_id="EGO_V1",
                object_id="P003",
                ttc_s=2.5,
                note="교차로 J1, 도달시간차 0.4s | 직교 진입 상충 예상",
                junction_id="J1",
                arrival_gap_s=0.4,
                conflict="orthogonal",
            ),
            Interaction(
                kind="lane_change_conflict",
                subject_id="V002",
                object_id="EGO_V1",
                gap_m=12.0,
                note="1차선 진입 경합",
                target_lane=1,
            ),
        ],
        area_name="Columbus, OH",
        frame_idx=int(t * 10) + 1,
    )
    snap.infrastructure.append(
        InfraState(
            infra_id="infra_1",
            world_xy=(12.0, 12.0),
            height_m=5.5,
            heading_deg=210.0,
            n_observed=3,
            placement=placement(),
        )
    )
    snap.scenario = ScenarioContext(
        scenario_id="type1_subtype1_accident/Town05_x",
        source="DeepAccident",
        town="Town05",
        available={"weather": "ClearNight"},
        ground_truth={"collision": True},
    )
    snap.map_context = {"map_source": "inferred_from_trajectories"}
    return snap


class TestNormalize(unittest.TestCase):
    def test_defaults_to_korean(self):
        for v in (None, "", "fr", "ja", "xx-YY"):
            self.assertEqual(i18n.normalize_lang(v), "ko")

    def test_accepts_case_and_region(self):
        for v in ("en", "EN", "en-US", "en_GB"):
            self.assertEqual(i18n.normalize_lang(v), "en")

    def test_label_passthrough_for_unknown(self):
        # 도로명처럼 데이터에서 온 문자열은 번역 대상이 아니다
        self.assertEqual(i18n.label("High St", "en"), "High St")
        self.assertEqual(i18n.label(None, "en"), "")

    def test_label_korean_is_identity(self):
        self.assertEqual(i18n.label("북행", "ko"), "북행")

    def test_label_translates_closed_set(self):
        self.assertEqual(i18n.label("북행", "en"), "northbound")
        self.assertEqual(i18n.label("차선변경(좌)", "en"), "changing lane (left)")

    def test_every_label_has_english(self):
        missing = [k for k, v in i18n.LABELS.items() if not v.get("en")]
        self.assertEqual(missing, [], f"영어 표기 누락: {missing}")

    def test_map_source_is_key_based(self):
        self.assertTrue(
            i18n.map_source("inferred_from_trajectories", "en").startswith(
                "synthesized from trajectories"
            )
        )
        self.assertTrue(
            i18n.map_source("inferred_from_trajectories", "ko").startswith("궤적")
        )
        # 키가 아니면 원문 그대로 (파일명 등)
        self.assertEqual(i18n.map_source("Town05.xodr", "en"), "Town05.xodr")

    def test_every_map_source_has_both_languages(self):
        missing = [
            k for k, v in i18n.MAP_SOURCES.items() if not (v.get("ko") and v.get("en"))
        ]
        self.assertEqual(missing, [])

    def test_tx_falls_back_to_korean_key(self):
        # 영어 표에 없는 키는 한국어로 되돌아온다 (KeyError 로 죽지 않는다)
        table = i18n.tx("en")
        for k in i18n._TX_KO:
            self.assertIn(k, table)

    def test_every_template_key_translated(self):
        """영어 표에 빠진 키는 한국어가 그대로 출력되므로 전수 검사한다."""
        missing = sorted(set(i18n._TX_KO) - set(i18n._TX_EN))
        self.assertEqual(missing, [], f"영어 템플릿 누락: {missing}")

    def test_english_templates_have_no_hangul(self):
        bad = {k: v for k, v in i18n._TX_EN.items() if HANGUL.search(v)}
        self.assertEqual(bad, {}, f"영어 표에 한글: {bad}")

    def test_english_system_prompts_have_no_hangul(self):
        for kind in ("snapshot", "accident"):
            text = i18n.system_prompt(kind, "en")
            self.assertFalse(HANGUL.search(text), f"{kind} 프롬프트에 한글")
            self.assertTrue(HANGUL.search(i18n.system_prompt(kind, "ko")))

    def test_english_schema_has_no_hangul(self):
        self.assertEqual(hangul_in(i18n.prediction_schema("en")), [])
        self.assertTrue(hangul_in(i18n.prediction_schema("ko")))

    def test_schema_shape_is_language_independent(self):
        ko, en = i18n.prediction_schema("ko"), i18n.prediction_schema("en")
        self.assertEqual(ko["required"], en["required"])
        item_ko = ko["properties"]["predictions"]["items"]
        item_en = en["properties"]["predictions"]["items"]
        self.assertEqual(item_ko["required"], item_en["required"])
        self.assertEqual(
            sorted(item_ko["properties"]), sorted(item_en["properties"])
        )


class TestInteractionNote(unittest.TestCase):
    def test_following(self):
        it = Interaction(kind="following", subject_id="a", object_id="b")
        self.assertEqual(i18n.interaction_note(it, "ko"), "선행차 추종")
        self.assertEqual(i18n.interaction_note(it, "en"), "car following")

    def test_crossing_orthogonal(self):
        it = Interaction(
            kind="crossing", subject_id="a", object_id="b",
            junction_id="J1", arrival_gap_s=0.4, conflict="orthogonal",
        )
        en = i18n.interaction_note(it, "en")
        self.assertIn("junction J1", en)
        self.assertIn("0.4s", en)
        self.assertIn("orthogonal", en)
        self.assertFalse(HANGUL.search(en))

    def test_crossing_oncoming_turn(self):
        it = Interaction(
            kind="crossing", subject_id="a", object_id="b",
            junction_id="J2", arrival_gap_s=1.1, conflict="oncoming_turn",
            turn_probs=[("a", 0.37)],
        )
        en = i18n.interaction_note(it, "en")
        self.assertIn("a 37%", en)
        self.assertFalse(HANGUL.search(en))

    def test_lane_change(self):
        it = Interaction(
            kind="lane_change_conflict", subject_id="a", object_id="b",
            target_lane=2,
        )
        self.assertEqual(i18n.interaction_note(it, "en"), "contention for entering lane 2")

    def test_falls_back_to_stored_note(self):
        """구조화 필드가 없으면 저장된 note 를 쓴다 (직접 만든 Interaction)."""
        it = Interaction(kind="merge", subject_id="a", object_id="b", note="합류")
        self.assertEqual(i18n.interaction_note(it, "en"), "합류")


class TestSnapshotSerialization(unittest.TestCase):
    def setUp(self):
        self.snap = snapshot()

    def test_korean_is_default(self):
        cfg = SerializeConfig()
        self.assertEqual(cfg.language, "ko")
        self.assertTrue(HANGUL.search(to_text(self.snap, cfg)))

    def test_english_text_has_no_hangul(self):
        cfg = SerializeConfig(language="en")
        text = to_text(self.snap, cfg)
        leaks = HANGUL.findall(text)
        self.assertEqual(leaks, [], f"영어 브리핑에 한글: {set(leaks)}")

    def test_english_text_keeps_the_facts(self):
        text = to_text(self.snap, SerializeConfig(language="en"))
        for expect in (
            "northbound",
            "accelerating",
            "changing lane (left)",
            "speed unknown",
            "car following",
            "orthogonal",
            "travel direction unconfirmed",
            "High St",  # 도로명은 번역하지 않는다
            "EGO_V1",
        ):
            self.assertIn(expect, text)

    def test_english_json_has_no_hangul(self):
        blob = to_json(self.snap, SerializeConfig(language="en"))
        self.assertEqual(hangul_in(blob), [])

    def test_json_keys_are_language_independent(self):
        ko = to_json(self.snap, SerializeConfig(language="ko"))
        en = to_json(self.snap, SerializeConfig(language="en"))
        self.assertEqual(sorted(ko), sorted(en))
        self.assertEqual(
            sorted(ko["actors"][0]), sorted(en["actors"][0])
        )
        # 수치는 언어와 무관하게 같아야 한다
        self.assertEqual(
            ko["actors"][0]["position_enu_m"], en["actors"][0]["position_enu_m"]
        )

    def test_english_payload_has_no_hangul(self):
        payload = build_messages(
            self.snap, "Is a lane change safe now?", SerializeConfig(language="en")
        )
        self.assertEqual(hangul_in(payload), [])
        self.assertIn("Question:", payload["messages"][0]["content"][-1]["text"])

    def test_english_bev_legend(self):
        cfg = SerializeConfig(language="en", include_bev_ascii=True)
        text = to_text(self.snap, cfg)
        self.assertIn("@=reference", text)
        self.assertIn("BEV sketch", text)


class TestWindowSerialization(unittest.TestCase):
    def setUp(self):
        snaps = [snapshot(t / 2) for t in range(0, 13)]
        cfg = WindowConfig(window_s=3.0, stride_s=1.0, horizon_s=3.0)
        self.wins, _ = build_windows(snaps, cfg)
        self.cfg = cfg
        self.assertTrue(self.wins)

    def test_english_window_text_has_no_hangul(self):
        text = render_window_text(
            self.wins[0], SerializeConfig(language="en"), self.cfg
        )
        leaks = HANGUL.findall(text)
        self.assertEqual(leaks, [], f"영어 윈도우 텍스트에 한글: {set(leaks)}")

    def test_english_window_text_keeps_structure(self):
        text = render_window_text(
            self.wins[0], SerializeConfig(language="en"), self.cfg
        )
        for expect in (
            "Observation window",
            "Per-vehicle time course",
            "Roadside sensors",
            "Last observed timestep in detail",
        ):
            self.assertIn(expect, text)

    def test_english_question(self):
        q = build_question(self.wins[0], self.cfg, "en")
        self.assertFalse(HANGUL.search(q))
        self.assertIn("k=1:", q)
        self.assertIn("accident_expected=false", q)

    def test_korean_question_is_default(self):
        self.assertTrue(HANGUL.search(build_question(self.wins[0], self.cfg)))

    def test_english_window_payload_has_no_hangul(self):
        payload = build_window_payload(
            self.wins[0], SerializeConfig(language="en"), self.cfg
        )
        self.assertEqual(hangul_in(payload), [])

    def test_english_ground_truth_has_no_hangul(self):
        gt = window_ground_truth(
            self.wins[0],
            self.cfg,
            collision=CollisionTruth(
                occurred=True, carla_ids=(1, 2), time_s=7.0, method="scripted"
            ),
            lang="en",
        )
        self.assertEqual(hangul_in(gt), [])
        self.assertTrue(gt["notes"])

    def test_english_file_set_has_no_hangul(self):
        """실제로 쓰이는 파일 전체 — payload·정답·manifest 를 통째로 검사."""
        snaps = [snapshot(t / 2) for t in range(0, 13)]
        with tempfile.TemporaryDirectory() as d:
            write_window_set(
                snaps,
                out_dir=d,
                scfg=SerializeConfig(language="en"),
                cfg=self.cfg,
                collision=None,
                scenario_id="synthetic/x",
                also_write_text=True,
            )
            names = sorted(os.listdir(d))
            self.assertTrue(any(n.startswith("llm_payload_") for n in names))
            for name in names:
                path = os.path.join(d, name)
                with open(path, encoding="utf-8") as f:
                    body = f.read()
                leaks = set(HANGUL.findall(body))
                self.assertEqual(leaks, set(), f"{name} 에 한글: {leaks}")

    def test_korean_file_set_stays_korean(self):
        snaps = [snapshot(t / 2) for t in range(0, 13)]
        with tempfile.TemporaryDirectory() as d:
            write_window_set(
                snaps, out_dir=d, scfg=SerializeConfig(), cfg=self.cfg,
                collision=None, scenario_id="synthetic/x",
            )
            with open(
                os.path.join(d, "manifest.json"), encoding="utf-8"
            ) as f:
                man = json.load(f)
            self.assertEqual(man["config"]["language"], "ko")
            self.assertTrue(HANGUL.search(man["config"]["bucket_rule"]))


if __name__ == "__main__":
    unittest.main()


class TestComposedLabels(unittest.TestCase):
    """학습 예측기가 **합성**하는 라벨도 번역돼야 한다.

    `predict_model` 은 안쪽 기동 이름을 끼워 넣은 라벨을 만든다
    (`후보 밖 경로(가장 가까운 기동 좌회전)`). 사전 조회만으로는 안 걸려서
    영어 브리핑에 한글이 새어 나갔다 — 실측으로 val 104개 중 96개에
    323회 남아 있었다.
    """

    def test_off_candidate_translates_with_inner_maneuver(self):
        got = i18n.label("후보 밖 경로(가장 가까운 기동 좌회전)", "en")
        self.assertEqual(got, "off-candidate path (nearest maneuver: left turn)")

    def test_no_such_candidate_translates_with_inner(self):
        self.assertEqual(i18n.label("직진(해당 후보 없음)", "en"),
                         "straight (no such candidate)")

    def test_plain_composed_label(self):
        self.assertEqual(i18n.label("경로 예측(후보 없음)", "en"),
                         "path prediction (no candidate)")

    def test_korean_is_unchanged(self):
        for v in ("후보 밖 경로(가장 가까운 기동 좌회전)", "직진(해당 후보 없음)"):
            self.assertEqual(i18n.label(v, "ko"), v)

    def test_unknown_label_passes_through(self):
        self.assertEqual(i18n.label("알 수 없는 라벨", "en"), "알 수 없는 라벨")

    def test_no_hangul_left_in_english_output(self):
        import re
        for v in ("후보 밖 경로(가장 가까운 기동 유턴)", "직진(해당 후보 없음)",
                  "경로 예측(후보 없음)", "차선유지", "정지 유지"):
            self.assertIsNone(re.search(r"[가-힣]", i18n.label(v, "en")),
                              msg=f"{v!r} 번역에 한글이 남았다")
