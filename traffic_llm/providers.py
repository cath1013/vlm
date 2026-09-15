"""LLM provider 별 요청 본문 생성.

payload 내용(system 프롬프트, 텍스트 블록, 질문, 응답 스키마)은 provider 와
무관하게 한 번 만들고, 여기서 각 API 의 요청 형식으로 감싼다.

    build_request("gemini", model="...", system=..., blocks=[...], schema=...)

설계 규약
    - **payload 파일 = 요청 본문 그대로**. 그래야 `**payload` 로 바로 보낼 수
      있고, 파일에 API 가 모르는 키가 섞이지 않는다. provider·endpoint 같은
      메타정보는 manifest 에 적는다.
    - Gemini 는 모델을 **URL 경로**에 넣으므로 본문에 `model` 을 넣지 않는다.
      `endpoint_for()` 가 호출 URL 을 만들어 준다.
    - 응답 스키마는 provider 마다 방언이 다르다. 정본은 JSON Schema 이고
      (`i18n.prediction_schema`), Gemini 용은 여기서 OpenAPI 방언으로 옮긴다.

검증 상태
    Claude 경로는 이 저장소에서 실제로 쓰이는 형식이다. OpenAI·Gemini 형식은
    각 API 문서 기준으로 작성했고 **실제 호출로 검증하지 않았다**. 모델 id 와
    추론 관련 파라미터는 버전에 따라 달라지므로 `--model`/`extra` 로 지정한다.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence

PROVIDERS = ("claude", "openai", "gemini")

# provider 별 기본 모델. Claude 만 이 저장소에서 검증된 값이 있고, 나머지는
# 모델 id 가 버전마다 바뀌므로 사용자가 --model 로 지정해야 한다.
DEFAULT_MODEL: Dict[str, Optional[str]] = {
    "claude": "claude-opus-5",
    "openai": None,
    "gemini": None,
}

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
CLAUDE_ENDPOINT = "https://api.anthropic.com/v1/messages"

SCHEMA_NAME = "accident_prediction"


def normalize_provider(name: Optional[str]) -> str:
    p = (name or "claude").strip().lower()
    aliases = {
        "anthropic": "claude",
        "chatgpt": "openai",
        "gpt": "openai",
        "google": "gemini",
    }
    p = aliases.get(p, p)
    if p not in PROVIDERS:
        raise ValueError(
            f"지원하지 않는 provider: {name!r} (가능: {', '.join(PROVIDERS)})"
        )
    return p


def resolve_model(provider: str, model: Optional[str]) -> str:
    """모델 id 확정. 기본값이 없는 provider 는 명시를 요구한다.

    검증하지 않은 모델 id 를 임의로 박아 넣으면 호출이 알 수 없는 이유로
    실패하므로, 모르는 것은 짐작하지 않고 사용자에게 받는다.
    """
    provider = normalize_provider(provider)
    m = model or DEFAULT_MODEL.get(provider)
    if not m:
        raise ValueError(
            f"provider={provider} 는 기본 모델이 없습니다 — --model 로 "
            "모델 id 를 지정하십시오 (예: --model gemini-2.5-pro)."
        )
    return m


def endpoint_for(provider: str, model: Optional[str] = None) -> str:
    """호출 URL. Gemini 는 모델이 경로에 들어간다."""
    provider = normalize_provider(provider)
    if provider == "gemini":
        return GEMINI_ENDPOINT.format(model=resolve_model(provider, model))
    return OPENAI_ENDPOINT if provider == "openai" else CLAUDE_ENDPOINT


# ---------------------------------------------------------------- 스키마 변환

_GEMINI_TYPES = {
    "object": "OBJECT",
    "array": "ARRAY",
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
}


def gemini_schema(schema: dict) -> dict:
    """JSON Schema → Gemini `responseSchema` (OpenAPI 3.0 부분집합).

    차이점을 흡수한다.
      - `type` 이 대문자 열거값이다
      - `additionalProperties` 를 지원하지 않는다 (제거)
      - 필드 순서를 고정하려면 `propertyOrdering` 을 준다. 주지 않으면 순서가
        흔들려 파싱·비교가 불안정해진다.
    """
    out: Dict[str, Any] = {}
    t = schema.get("type")
    if isinstance(t, str):
        out["type"] = _GEMINI_TYPES.get(t, t.upper())
    for key in ("description", "enum", "format", "nullable", "minItems", "maxItems"):
        if key in schema:
            out[key] = copy.deepcopy(schema[key])
    if "properties" in schema:
        props = schema["properties"]
        out["properties"] = {k: gemini_schema(v) for k, v in props.items()}
        out["propertyOrdering"] = list(props.keys())
    if "required" in schema:
        out["required"] = list(schema["required"])
    if "items" in schema:
        out["items"] = gemini_schema(schema["items"])
    return out


def openai_schema(schema: dict) -> dict:
    """OpenAI structured outputs 용 래퍼.

    strict 모드는 모든 object 에 `additionalProperties: false` 와 **모든**
    속성이 `required` 에 있기를 요구한다. 이 프로젝트의 스키마는 이미 그렇게
    작성돼 있어 그대로 넘긴다.
    """
    return {
        "name": SCHEMA_NAME,
        "strict": True,
        "schema": copy.deepcopy(schema),
    }


# ---------------------------------------------------------------- 본문 생성


def _deep_merge(base: dict, extra: Optional[dict]) -> dict:
    """extra 를 base 에 병합. dict 는 재귀, 나머지는 덮어쓴다."""
    if not extra:
        return base
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def build_request(
    provider: str,
    *,
    system: str,
    blocks: Sequence[str],
    schema: Optional[dict] = None,
    model: Optional[str] = None,
    max_tokens: int = 16000,
    effort: str = "high",
    extra: Optional[dict] = None,
) -> dict:
    """provider 별 요청 본문.

    blocks: user 메시지를 이루는 텍스트 조각들 (브리핑 / 구조화 JSON / 질문).
      Claude·Gemini 는 여러 블록을 그대로 유지하고, OpenAI Chat Completions 는
      호환성을 위해 하나의 문자열로 합친다.
    extra: provider 고유 파라미터를 덮어쓸 dict. 추론 예산처럼 버전에 따라
      달라지는 값은 여기로 넣는다 (짐작해서 박아 넣지 않는다).
    """
    provider = normalize_provider(provider)
    blocks = [b for b in blocks if b]
    if provider == "claude":
        body = _claude(system, blocks, schema, model, max_tokens, effort)
    elif provider == "openai":
        body = _openai(system, blocks, schema, model, max_tokens, effort)
    else:
        body = _gemini(system, blocks, schema, model, max_tokens)
    return _deep_merge(body, extra)


def _claude(system, blocks, schema, model, max_tokens, effort) -> dict:
    out: Dict[str, Any] = {
        "model": resolve_model("claude", model),
        "max_tokens": max_tokens,
        # 4.6 이후 모델은 adaptive 를 쓴다. budget_tokens 는 Opus 5 에서
        # 400 오류로 거부되므로 넣지 않는다.
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort},
        "system": [
            {
                "type": "text",
                "text": system,
                # 윈도우를 연속 호출할 때 프리픽스를 캐싱한다
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": b} for b in blocks],
            }
        ],
    }
    if schema is not None:
        out["output_config"]["format"] = {
            "type": "json_schema",
            "schema": copy.deepcopy(schema),
        }
    return out


def _openai(system, blocks, schema, model, max_tokens, effort) -> dict:
    out: Dict[str, Any] = {
        "model": resolve_model("openai", model),
        "messages": [
            {"role": "system", "content": system},
            # Chat Completions 는 문자열 content 가 가장 호환성이 넓다.
            # 블록 경계는 빈 줄로 남긴다.
            {"role": "user", "content": "\n\n".join(blocks)},
        ],
        # max_tokens 는 최신 모델에서 폐기 예정이라 max_completion_tokens 를 쓴다
        "max_completion_tokens": max_tokens,
    }
    if schema is not None:
        out["response_format"] = {
            "type": "json_schema",
            "json_schema": openai_schema(schema),
        }
    if effort:
        # 추론 모델 전용. 비추론 모델에 보내면 거부되므로 필요 없으면
        # extra 로 {"reasoning_effort": null} 을 주어 지운다.
        out["reasoning_effort"] = effort
    return out


def _gemini(system, blocks, schema, model, max_tokens) -> dict:
    # 모델은 URL 경로에 들어간다 — 본문에 넣지 않는다 (endpoint_for 사용).
    resolve_model("gemini", model)  # 지정 여부만 여기서 검증
    gen: Dict[str, Any] = {"maxOutputTokens": max_tokens}
    if schema is not None:
        gen["responseMimeType"] = "application/json"
        gen["responseSchema"] = gemini_schema(schema)
    return {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [
            {"role": "user", "parts": [{"text": b} for b in blocks]}
        ],
        "generationConfig": gen,
    }


def request_texts(provider: str, body: dict) -> List[str]:
    """요청 본문에서 모델에게 가는 텍스트를 모두 뽑는다 (system 포함).

    입력 크기를 provider 무관하게 재기 위한 것이다. 정확한 토큰 수는 각 API 의
    토큰 계산 엔드포인트로 재야 하지만, 윈도우 설정을 비교할 때는 문자 수로
    충분하다.
    """
    provider = normalize_provider(provider)
    out: List[str] = []
    if provider == "claude":
        for b in body.get("system") or []:
            if isinstance(b, dict):
                out.append(b.get("text", ""))
        for m in body.get("messages") or []:
            c = m.get("content")
            if isinstance(c, str):
                out.append(c)
            elif isinstance(c, list):
                out += [
                    b.get("text", "") for b in c if isinstance(b, dict)
                ]
    elif provider == "openai":
        for m in body.get("messages") or []:
            c = m.get("content")
            if isinstance(c, str):
                out.append(c)
            elif isinstance(c, list):
                out += [
                    b.get("text", "") for b in c if isinstance(b, dict)
                ]
    else:  # gemini
        for part in (body.get("systemInstruction") or {}).get("parts") or []:
            if isinstance(part, dict):
                out.append(part.get("text", ""))
        for content in body.get("contents") or []:
            for part in content.get("parts") or []:
                if isinstance(part, dict):
                    out.append(part.get("text", ""))
    return [t for t in out if t]


def request_chars(provider: str, body: dict) -> int:
    return sum(len(t) for t in request_texts(provider, body))


# ---------------------------------------------------------------- 응답 파싱


def response_texts(provider: str, response: dict) -> List[str]:
    """응답 본문에서 모델이 낸 텍스트 조각들을 뽑는다 (추론 블록 제외)."""
    provider = normalize_provider(provider)
    out: List[str] = []
    if provider == "claude":
        for b in response.get("content") or []:
            if isinstance(b, dict) and b.get("type") == "text":
                out.append(b.get("text", ""))
    elif provider == "openai":
        for ch in response.get("choices") or []:
            msg = ch.get("message") or {}
            c = msg.get("content")
            if isinstance(c, str):
                out.append(c)
            elif isinstance(c, list):
                out += [
                    part.get("text", "")
                    for part in c
                    if isinstance(part, dict) and part.get("type") in ("text", "output_text")
                ]
    else:  # gemini
        for cand in response.get("candidates") or []:
            for part in (cand.get("content") or {}).get("parts") or []:
                if isinstance(part, dict) and "text" in part:
                    # thought=True 는 추론 요약이므로 답이 아니다
                    if not part.get("thought"):
                        out.append(part["text"])
    return [t for t in out if t]


def usage_of(provider: str, response: dict) -> Dict[str, Optional[int]]:
    """토큰 사용량을 provider 무관 형태로."""
    provider = normalize_provider(provider)
    if provider == "claude":
        u = response.get("usage") or {}
        return {"input": u.get("input_tokens"), "output": u.get("output_tokens")}
    if provider == "openai":
        u = response.get("usage") or {}
        return {
            "input": u.get("prompt_tokens"),
            "output": u.get("completion_tokens"),
        }
    u = response.get("usageMetadata") or {}
    return {
        "input": u.get("promptTokenCount"),
        "output": u.get("candidatesTokenCount"),
    }
