"""provider 별 요청 본문 생성 테스트.

세 API 의 형식이 서로 다르므로, **내용은 같고 껍데기만 다른지**를 검사한다.
payload 파일은 요청 본문 그대로여야 하므로(그래야 `**payload` 로 보낼 수 있다)
API 가 모르는 키가 섞이지 않는지도 본다.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_llm import i18n, providers
from traffic_llm.accident_qa import (
    WindowConfig,
    build_window_payload,
    build_windows,
    write_window_set,
)
from traffic_llm.config import SerializeConfig
from traffic_llm.serialize import build_messages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_i18n import snapshot  # 동일한 합성 스냅샷 재사용

SCHEMA = i18n.prediction_schema("ko")
BLOCKS = ["브리핑 본문", "구조화 데이터:\n```json\n{}\n```", "질문: 위험한가?"]
SYSTEM = "당신은 분석 전문가입니다."


class TestNormalizeAndModel(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(providers.normalize_provider("anthropic"), "claude")
        self.assertEqual(providers.normalize_provider("ChatGPT"), "openai")
        self.assertEqual(providers.normalize_provider("GPT"), "openai")
        self.assertEqual(providers.normalize_provider("google"), "gemini")
        self.assertEqual(providers.normalize_provider(None), "claude")

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            providers.normalize_provider("llama")

    def test_claude_has_default_model(self):
        self.assertEqual(
            providers.resolve_model("claude", None), "claude-opus-5"
        )

    def test_other_providers_require_model(self):
        """검증하지 않은 모델 id 를 짐작해 넣지 않는다 — 명시를 요구한다."""
        for p in ("openai", "gemini"):
            with self.assertRaises(ValueError):
                providers.resolve_model(p, None)
            self.assertEqual(providers.resolve_model(p, "x-1"), "x-1")

    def test_endpoints(self):
        self.assertIn("api.anthropic.com", providers.endpoint_for("claude"))
        self.assertIn("api.openai.com", providers.endpoint_for("openai"))
        url = providers.endpoint_for("gemini", "gemini-2.5-pro")
        self.assertIn("gemini-2.5-pro:generateContent", url)


class TestSchemaTranslation(unittest.TestCase):
    def test_gemini_uppercases_types_and_drops_unsupported(self):
        g = providers.gemini_schema(SCHEMA)
        self.assertEqual(g["type"], "OBJECT")
        self.assertNotIn("additionalProperties", g)
        item = g["properties"]["predictions"]["items"]
        self.assertEqual(g["properties"]["predictions"]["type"], "ARRAY")
        self.assertEqual(item["type"], "OBJECT")
        self.assertNotIn("additionalProperties", item)
        self.assertEqual(item["properties"]["k"]["type"], "INTEGER")
        self.assertEqual(item["properties"]["accident_expected"]["type"], "BOOLEAN")
        self.assertEqual(
            item["properties"]["involved_actor_ids"]["items"]["type"], "STRING"
        )

    def test_gemini_keeps_semantics(self):
        g = providers.gemini_schema(SCHEMA)
        item = g["properties"]["predictions"]["items"]
        # 설명·열거·필수는 살아 있어야 한다
        self.assertTrue(item["properties"]["reason"]["description"])
        self.assertEqual(
            item["properties"]["confidence"]["enum"], ["low", "medium", "high"]
        )
        self.assertIn("k", item["required"])
        # 필드 순서를 고정해 파싱·비교가 흔들리지 않게 한다
        self.assertEqual(
            g["propertyOrdering"], list(SCHEMA["properties"].keys())
        )

    def test_openai_strict_requirements_hold(self):
        """strict 모드는 additionalProperties=false + 전 속성 required 를 요구한다."""
        w = providers.openai_schema(SCHEMA)
        self.assertTrue(w["strict"])
        item = w["schema"]["properties"]["predictions"]["items"]
        self.assertIs(item["additionalProperties"], False)
        self.assertEqual(sorted(item["required"]), sorted(item["properties"]))
        self.assertIs(w["schema"]["additionalProperties"], False)
        self.assertEqual(
            sorted(w["schema"]["required"]), sorted(w["schema"]["properties"])
        )

    def test_translation_does_not_mutate_input(self):
        before = json.dumps(SCHEMA, sort_keys=True)
        providers.gemini_schema(SCHEMA)
        providers.openai_schema(SCHEMA)
        self.assertEqual(json.dumps(SCHEMA, sort_keys=True), before)


class TestRequestShapes(unittest.TestCase):
    def build(self, provider, **kw):
        return providers.build_request(
            provider, system=SYSTEM, blocks=BLOCKS, schema=SCHEMA, **kw
        )

    def test_claude_shape(self):
        b = self.build("claude")
        self.assertEqual(b["model"], "claude-opus-5")
        self.assertEqual(b["thinking"], {"type": "adaptive"})
        self.assertNotIn("temperature", b)
        self.assertNotIn("budget_tokens", b["thinking"])
        self.assertEqual(
            b["system"][0]["cache_control"], {"type": "ephemeral"}
        )
        self.assertEqual(
            b["output_config"]["format"]["type"], "json_schema"
        )
        # 블록 구조를 유지한다
        self.assertEqual(len(b["messages"][0]["content"]), len(BLOCKS))

    def test_openai_shape(self):
        b = self.build("openai", model="gpt-x")
        self.assertEqual(b["model"], "gpt-x")
        self.assertEqual(b["messages"][0]["role"], "system")
        self.assertEqual(b["messages"][1]["role"], "user")
        # Chat Completions 호환성을 위해 블록을 하나의 문자열로 합친다
        self.assertIsInstance(b["messages"][1]["content"], str)
        for blk in BLOCKS:
            self.assertIn(blk, b["messages"][1]["content"])
        # 폐기 예정인 max_tokens 대신 max_completion_tokens
        self.assertIn("max_completion_tokens", b)
        self.assertNotIn("max_tokens", b)
        self.assertEqual(b["response_format"]["type"], "json_schema")

    def test_gemini_shape(self):
        b = self.build("gemini", model="gemini-x")
        # 모델은 URL 경로에 들어가므로 본문에 없다
        self.assertNotIn("model", b)
        self.assertEqual(len(b["contents"][0]["parts"]), len(BLOCKS))
        self.assertEqual(b["systemInstruction"]["parts"][0]["text"], SYSTEM)
        gen = b["generationConfig"]
        self.assertEqual(gen["responseMimeType"], "application/json")
        self.assertEqual(gen["responseSchema"]["type"], "OBJECT")
        self.assertEqual(gen["maxOutputTokens"], 16000)

    def test_content_is_identical_across_providers(self):
        """껍데기만 다르고 모델이 읽는 내용은 같아야 한다."""
        texts = {}
        for p, m in (("claude", None), ("openai", "x"), ("gemini", "y")):
            b = self.build(p, model=m)
            texts[p] = providers.request_texts(p, b)
        # system + 블록 3개
        for p, ts in texts.items():
            self.assertEqual(ts[0], SYSTEM, p)
            joined = "\n\n".join(ts[1:])
            for blk in BLOCKS:
                self.assertIn(blk, joined, p)

    def test_request_chars_is_comparable(self):
        sizes = {
            p: providers.request_chars(p, self.build(p, model="m"))
            for p in ("claude", "openai", "gemini")
        }
        # OpenAI 는 블록을 합칠 때 구분자가 들어가 조금 길다
        self.assertLessEqual(abs(sizes["claude"] - sizes["gemini"]), 0)
        self.assertLessEqual(sizes["openai"] - sizes["claude"], 8)

    def test_extra_deep_merges(self):
        b = self.build(
            "gemini",
            model="g",
            extra={"generationConfig": {"temperature": 0.2}},
        )
        # 기존 키를 지우지 않고 병합한다
        self.assertEqual(b["generationConfig"]["temperature"], 0.2)
        self.assertEqual(b["generationConfig"]["maxOutputTokens"], 16000)

    def test_extra_can_remove_a_field(self):
        """비추론 모델에는 reasoning_effort 를 지워야 한다."""
        b = self.build("openai", model="gpt-x", extra={"reasoning_effort": None})
        self.assertIsNone(b["reasoning_effort"])

    def test_schema_optional(self):
        for p, m in (("claude", None), ("openai", "x"), ("gemini", "y")):
            b = providers.build_request(
                p, system=SYSTEM, blocks=BLOCKS, schema=None, model=m
            )
            if p == "claude":
                self.assertNotIn("format", b["output_config"])
            elif p == "openai":
                self.assertNotIn("response_format", b)
            else:
                self.assertNotIn("responseSchema", b["generationConfig"])


class TestResponseParsing(unittest.TestCase):
    ANSWER = {"predictions": [], "overall_assessment": "x", "data_limitations": ""}

    def test_claude(self):
        resp = {
            "content": [
                {"type": "thinking", "thinking": "생각"},
                {"type": "text", "text": json.dumps(self.ANSWER)},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
        self.assertEqual(
            json.loads(providers.response_texts("claude", resp)[0]), self.ANSWER
        )
        self.assertEqual(
            providers.usage_of("claude", resp), {"input": 10, "output": 2}
        )

    def test_openai(self):
        resp = {
            "choices": [{"message": {"content": json.dumps(self.ANSWER)}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }
        self.assertEqual(
            json.loads(providers.response_texts("openai", resp)[0]), self.ANSWER
        )
        self.assertEqual(
            providers.usage_of("openai", resp), {"input": 7, "output": 3}
        )

    def test_gemini_skips_thought_parts(self):
        resp = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "추론 요약", "thought": True},
                            {"text": json.dumps(self.ANSWER)},
                        ]
                    }
                }
            ],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 4},
        }
        texts = providers.response_texts("gemini", resp)
        self.assertEqual(len(texts), 1)
        self.assertEqual(json.loads(texts[0]), self.ANSWER)
        self.assertEqual(
            providers.usage_of("gemini", resp), {"input": 9, "output": 4}
        )


class TestEndToEnd(unittest.TestCase):
    def test_build_messages_follows_provider(self):
        for p, m in (("claude", None), ("openai", "x"), ("gemini", "y")):
            cfg = SerializeConfig(provider=p, model=m, include_bev_ascii=False)
            b = build_messages(snapshot(), "위험한가?", cfg)
            if p == "gemini":
                self.assertIn("contents", b)
            elif p == "openai":
                self.assertIn("messages", b)
                self.assertIsInstance(b["messages"][1]["content"], str)
            else:
                self.assertIn("system", b)

    def test_window_payload_follows_provider(self):
        snaps = [snapshot(t / 2) for t in range(0, 13)]
        wcfg = WindowConfig(window_s=3.0, stride_s=1.0, horizon_s=3.0)
        wins, _ = build_windows(snaps, wcfg)
        for p, m in (("claude", None), ("openai", "x"), ("gemini", "y")):
            scfg = SerializeConfig(provider=p, model=m)
            b = build_window_payload(wins[0], scfg, wcfg)
            self.assertEqual(
                "contents" in b, p == "gemini", f"{p}: 형식 불일치"
            )

    def test_manifest_records_provider_and_endpoint(self):
        """payload 는 본문 그대로이므로 provider 정보는 manifest 에만 남는다.

        Gemini 는 모델이 URL 경로에 들어가 본문에 없으므로 이 기록이 없으면
        어떤 모델용 payload 인지 알 수 없게 된다.
        """
        snaps = [snapshot(t / 2) for t in range(0, 13)]
        wcfg = WindowConfig(window_s=3.0, stride_s=1.0, horizon_s=3.0)
        with tempfile.TemporaryDirectory() as d:
            man = write_window_set(
                snaps,
                out_dir=d,
                scfg=SerializeConfig(provider="gemini", model="gemini-x"),
                cfg=wcfg,
                collision=None,
                scenario_id="synthetic/x",
                also_write_text=False,
            )
            self.assertEqual(man["config"]["provider"], "gemini")
            self.assertEqual(man["config"]["model"], "gemini-x")
            self.assertIn("gemini-x:generateContent", man["config"]["endpoint"])
            # payload 본문에는 model 이 없어야 한다
            with open(
                os.path.join(d, man["windows"][0]["payload"]), encoding="utf-8"
            ) as f:
                body = json.load(f)
            self.assertNotIn("model", body)
            self.assertIn("contents", body)

    def test_input_chars_counted_for_every_provider(self):
        snaps = [snapshot(t / 2) for t in range(0, 13)]
        wcfg = WindowConfig(window_s=3.0, stride_s=1.0, horizon_s=3.0)
        for p, m in (("claude", None), ("openai", "x"), ("gemini", "y")):
            with tempfile.TemporaryDirectory() as d:
                man = write_window_set(
                    snaps, out_dir=d, scfg=SerializeConfig(provider=p, model=m),
                    cfg=wcfg, collision=None, scenario_id="synthetic/x",
                    also_write_text=False,
                )
                self.assertGreater(man["summary"]["input_chars_mean"], 100, p)


if __name__ == "__main__":
    unittest.main()
