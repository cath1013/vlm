"""생성된 payload 를 LLM API 에 보내고 정답과 대조한다.

    # 윈도우 하나
    python examples/ask_llm.py out/windows_example/ko/accident/llm_payload_3-8.json

    # 디렉터리 전체 (정답이 있으면 채점까지)
    python examples/ask_llm.py out/windows_example/ko/accident --score

provider 는 같은 디렉터리의 manifest.json 에서 읽는다 (없으면 본문 형태로 추정).
payload 파일이 **요청 본문 그대로**이므로 여기서 내용을 덧붙이지 않는다 —
모델·추론 설정·출력 스키마는 이미 payload 안에 있다.

준비 (provider 별 환경변수)
    claude   ANTHROPIC_API_KEY   (pip install anthropic — 공식 SDK 사용)
    openai   OPENAI_API_KEY      (SDK 없이 HTTP POST)
    gemini   GEMINI_API_KEY      (SDK 없이 HTTP POST)

PowerShell:  $env:ANTHROPIC_API_KEY="sk-ant-..."
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

from traffic_llm import providers
from traffic_llm.accident_qa import (
    MODES,
    SUPPORTED_MODES,
    aggregate_modes,
    aggregate_scores,
    response_issues,
    score_modes,
    score_response,
)


ENV_KEY = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def expected_ks_from_payload(body: dict) -> list:
    """payload 의 응답 스키마 설명에서 구간 수 N 을 읽어 1..N 을 만든다.

    정답 파일이 없을 때도 "구간을 빠뜨렸는가"를 볼 수 있어야 한다.
    """
    blob = json.dumps(body, ensure_ascii=False)
    m = re.search(r"N=(\d+)", blob)
    return list(range(1, int(m.group(1)) + 1)) if m else []


def detect_provider(base: str, body: dict) -> str:
    """manifest 의 기록을 우선 쓰고, 없으면 본문 형태로 추정한다."""
    man = os.path.join(base, "manifest.json")
    if os.path.isfile(man):
        with open(man, encoding="utf-8") as f:
            cfg = (json.load(f).get("config") or {})
        if cfg.get("provider"):
            return providers.normalize_provider(cfg["provider"])
    if "contents" in body:
        return "gemini"
    if "response_format" in body or "max_completion_tokens" in body:
        return "openai"
    return "claude"


def api_key(provider: str) -> str:
    env = ENV_KEY[provider]
    key = os.environ.get(env)
    if not key:
        raise SystemExit(
            f"{env} 환경변수가 없습니다.\n"
            f'  PowerShell:  $env:{env}="..."'
        )
    return key


def call_claude(body: dict, stream: bool) -> dict:
    """공식 Anthropic SDK 로 호출하고 응답을 dict 로 돌려준다."""
    try:
        import anthropic
    except ImportError:
        raise SystemExit("anthropic SDK 가 없습니다.  pip install anthropic")
    client = anthropic.Anthropic(api_key=api_key("claude"))
    if stream:
        with client.messages.stream(**body) as st:
            msg = st.get_final_message()
    else:
        msg = client.messages.create(**body)
    return msg.model_dump() if hasattr(msg, "model_dump") else json.loads(msg.json())


def call_http(provider: str, body: dict, endpoint: str, timeout: float) -> dict:
    """OpenAI·Gemini 는 SDK 없이 요청 본문을 그대로 POST 한다."""
    import urllib.error
    import urllib.request

    key = api_key(provider)
    url = endpoint
    headers = {"Content-Type": "application/json"}
    if provider == "openai":
        headers["Authorization"] = f"Bearer {key}"
    else:  # gemini — 헤더로 키를 넘긴다 (URL 쿼리는 로그에 남는다)
        headers["x-goog-api-key"] = key
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:2000]
        raise SystemExit(f"{provider} API 오류 {e.code}\n{detail}")


def call(provider: str, body: dict, endpoint: str, stream: bool, timeout: float):
    """(응답 dict, 소요초). 입력이 5만 자를 넘으므로 Claude 는 기본 스트리밍."""
    t0 = time.time()
    if provider == "claude":
        resp = call_claude(body, stream)
    else:
        resp = call_http(provider, body, endpoint, timeout)
    return resp, time.time() - t0


def _set_max_tokens(provider: str, body: dict, n: int) -> None:
    """provider 별 응답 토큰 상한 키를 덮어쓴다."""
    key = {
        "claude": "max_tokens",
        "openai": "max_completion_tokens",
    }.get(provider)
    if key:
        body[key] = n
    else:  # gemini — generationConfig 안에 있다
        body.setdefault("generationConfig", {})["maxOutputTokens"] = n


def _brace_slice(t: str) -> Optional[str]:
    """가장 바깥 {...} 를 괄호 짝을 세어 잘라낸다 (문자열 리터럴 안은 무시).

    정규식으로 `\\{.*\\}` 를 잡으면 본문에 중괄호가 섞였을 때 엉뚱한 데서
    끊긴다. 모델이 JSON 앞뒤에 한 줄 설명을 붙여 보내는 경우를 위한 마지막
    수단이다.
    """
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return t[start : i + 1]
    return None


def extract_json(provider: str, resp: dict) -> dict:
    """응답에서 구조화 출력(JSON)을 꺼낸다.

    payload 가 JSON 스키마를 강제하므로 본문 텍스트가 곧 JSON 이다. 추론 블록은
    provider 모듈이 걸러 준다. 그래도 세 단계로 시도한다 — 그대로 파싱 →
    코드펜스 벗기기 → 괄호 짝 세어 잘라내기.

    실패하면 **받은 텍스트를 예외에 담는다**. 원문을 못 보면 응답이 잘렸는지,
    형식이 달랐는지, 애초에 빈 응답이었는지 구분할 수 없다.
    """
    texts = providers.response_texts(provider, resp)
    for text in texts:
        cands = [text.strip()]
        for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S):
            cands.append(m.group(1).strip())
        sliced = _brace_slice(text)
        if sliced:
            cands.append(sliced)
        for t in cands:
            if not t:
                continue
            try:
                return json.loads(t)
            except json.JSONDecodeError:
                continue
    detail = " / ".join(f"[{len(t)}자] {t[:300]!r}" for t in texts) or "(텍스트 없음)"
    stop = resp.get("stop_reason") or resp.get("finish_reason")
    if stop in ("max_tokens", "length"):
        # 흔한 원인이므로 따로 알려 준다. payload 가 요청 본문 그대로이므로
        # max_tokens 를 늘리려면 payload 를 다시 생성해야 한다.
        raise ValueError(
            f"응답이 토큰 상한에서 잘렸습니다 (stop_reason={stop}). "
            f"payload 의 max_tokens 를 올려 다시 생성하거나 "
            f"--max-tokens 로 이번 호출만 늘리십시오. 받은 텍스트: {detail}"
        )
    raise ValueError(
        f"응답에서 JSON 을 찾지 못했습니다. stop_reason={stop} · "
        f"받은 텍스트: {detail}"
    )


def show(label: str, answer: dict, gt, secs: float, usage: dict,
         issues=None, score=None) -> None:
    tok = ""
    if usage.get("input") is not None:
        tok = f", 입력 {usage['input']:,} · 출력 {usage.get('output') or 0:,} 토큰"
    print(f"\n{'=' * 74}\n윈도우 {label}   ({secs:.1f}초{tok})")
    print("=" * 74)
    exp = {e["k"]: e for e in (gt or {}).get("expected", [])}
    for p in answer.get("predictions", []):
        k = p.get("k")
        mark = "사고" if p.get("accident_expected") else "없음"
        line = f"  k={k} {p.get('interval_s',''):>10s} → {mark}"
        if p.get("involved_actor_ids"):
            line += f"  {p['involved_actor_ids']}"
        line += f"  [{p.get('confidence','')}]"
        if k in exp:
            truth = "사고" if exp[k]["accident_expected"] else "없음"
            line += f"   |  정답 {truth}"
            if exp[k]["accident_expected"]:
                line += f" {exp[k]['involved_actor_ids']}"
            line += "  " + ("O" if (truth == mark) else "X")
        print(line)
        print(f"        이유: {p.get('reason','')}")
    if answer.get("overall_assessment"):
        print(f"\n  종합: {answer['overall_assessment']}")
    if answer.get("data_limitations"):
        print(f"  데이터 한계: {answer['data_limitations']}")
    if score and score.get("event"):
        ev = score["event"]
        mark = {"exact": "정확", "early": "조기 인정", "none": "미검출"}[ev["timing"]]
        lead = "" if ev.get("lead_s") is None else f" (실제보다 {ev['lead_s']}초 이름)"
        print(f"\n  사건 판정: {mark}{lead}  "
              f"| 실제 구간 k={ev['k_true']} · 인정 범위 k={ev['credit_buckets']}"
              f" · 대표 경보 k={ev['credited_k']}")
    for iss in issues or []:
        # 형식 문제는 점수에 거의 드러나지 않는다 (중복은 무시되고, 누락은
        # 정답이 '사고'일 때만 FN 이 된다). 눈에 보이게 따로 찍는다.
        print(f"  [응답 형식] {iss.get('kind')}: {iss.get('detail')} "
              f"— {iss.get('note','')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="payload → LLM 호출 및 채점")
    ap.add_argument("target", help="payload json 파일 또는 윈도우 디렉터리")
    ap.add_argument("--score", action="store_true",
                    help="같은 디렉터리의 ground_truth 와 대조해 채점")
    ap.add_argument("--out", default=None,
                    help="응답 저장 디렉터리 (기본: payload 와 같은 곳/responses)")
    ap.add_argument("--provider", default=None,
                    choices=["claude", "openai", "gemini"],
                    help="생략하면 manifest.json 또는 본문 형태로 판단")
    ap.add_argument("--endpoint", default=None,
                    help="호출 URL 을 직접 지정 (기본: manifest 또는 provider 기본값)")
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="HTTP 타임아웃 [초] (openai·gemini)")
    ap.add_argument("--no-stream", action="store_true",
                    help="Claude 를 스트리밍 없이 호출 (긴 입력에서는 타임아웃 위험)")
    ap.add_argument("--limit", type=int, default=0,
                    help="디렉터리 처리 시 앞 N개만")
    ap.add_argument("--rescore", action="store_true",
                    help="LLM 을 **부르지 않고** 저장된 response_*.json 만 다시 "
                         "채점한다. 창을 하나씩 호출하면 scores.json 이 매번 "
                         "덮어써져 앞 창의 결과가 사라지므로, 전부 받은 뒤 이것으로 "
                         "합산한다. --early-credit 을 바꿔 다시 채점할 때도 쓴다")
    ap.add_argument("--skip-existing", action="store_true",
                    help="response_<창>.json 이 이미 있으면 호출하지 않고 "
                         "저장된 것을 쓴다. 중단된 배치를 이어서 돌릴 때 쓴다 — "
                         "이미 돈을 쓴 응답을 다시 받지 않는다")
    ap.add_argument("--early-credit", type=float, default=None,
                    help="조기 예측 인정 시간 [초]. 생략하면 정답 파일에 적힌 "
                         "값을 쓴다. 같은 응답을 여러 기준으로 다시 채점할 때 준다")
    ap.add_argument("--modes", default="all",
                    help="다중 기준 채점. 쉼표로 구분한다 "
                         f"({','.join(MODES)}). 예: --modes binary_window,early. "
                         "'all' 이면 전부이며 기본값이다. 같은 응답을 여러 "
                         "기준으로 채점해 나란히 본다")
    ap.add_argument("--late-decay", type=float, default=0.2,
                    help="weighted 모드에서 늦은 경보 한 칸당 감점 (기본 0.2 → "
                         "1.0/0.8/0.6/0.4)")
    ap.add_argument("--early-decay", type=float, default=0.0,
                    help="weighted 모드에서 이른 경보 한 칸당 감점 (기본 0 = "
                         "이른 경보는 깎지 않는다)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="이번 호출의 응답 토큰 상한을 덮어쓴다. payload 는 "
                         "요청 본문 그대로이므로 평소에는 건드리지 않지만, "
                         "잘린 응답을 payload 재생성 없이 다시 받을 때 쓴다")
    args = ap.parse_args(argv)

    modes = tuple(MODES) if args.modes == "all" else tuple(
        m.strip() for m in args.modes.split(",") if m.strip()
    )
    bad = [m for m in modes if m not in SUPPORTED_MODES]
    if bad:
        ap.error(f"알 수 없는 채점 모드: {bad} (가능: {list(SUPPORTED_MODES)})")

    if os.path.isdir(args.target):
        files = sorted(
            os.path.join(args.target, f)
            for f in os.listdir(args.target)
            if f.startswith("llm_payload_") and f.endswith(".json")
        )
        base = args.target
    else:
        files = [args.target]
        base = os.path.dirname(os.path.abspath(args.target))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"payload 파일이 없습니다: {args.target}")

    with open(files[0], encoding="utf-8") as f:
        first = json.load(f)
    provider = args.provider or detect_provider(base, first)

    endpoint = args.endpoint
    if endpoint is None:
        man = os.path.join(base, "manifest.json")
        if os.path.isfile(man):
            with open(man, encoding="utf-8") as f:
                endpoint = (json.load(f).get("config") or {}).get("endpoint")
    if endpoint is None:
        endpoint = providers.endpoint_for(provider, first.get("model"))

    out_dir = args.out or os.path.join(base, "responses")
    os.makedirs(out_dir, exist_ok=True)
    print(f"provider={provider}  endpoint={endpoint}  payload {len(files)}개")

    scores = []
    mode_scores = []
    for path in files:
        label = os.path.basename(path)[len("llm_payload_"):-len(".json")]
        with open(path, encoding="utf-8") as f:
            body = json.load(f)
        saved_path = os.path.join(out_dir, f"response_{label}.json")
        reuse = (args.rescore or args.skip_existing) and os.path.isfile(saved_path)
        if reuse:
            with open(saved_path, encoding="utf-8") as f:
                saved = json.load(f)
            answer = saved["answer"]
            usage = saved.get("usage")
            secs = saved.get("elapsed_s") or 0.0
            resp = None
        elif args.rescore:
            print(f"윈도우 {label}: 저장된 응답이 없어 건너뜀 (--rescore)")
            continue
        else:
            if args.max_tokens:
                _set_max_tokens(provider, body, args.max_tokens)
            resp, secs = call(
                provider, body, endpoint, not args.no_stream, args.timeout
            )
            usage = providers.usage_of(provider, resp)
            # 파싱 전에 원문을 남긴다. 호출은 이미 돈(토큰)을 썼는데 파싱에서
            # 죽으면 응답이 통째로 사라진다 — 실제로 그 일이 있었다.
            raw_path = os.path.join(out_dir, f"raw_{label}.json")
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(resp, f, ensure_ascii=False, indent=1)
            try:
                answer = extract_json(provider, resp)
            except ValueError as e:
                print(f"\n윈도우 {label}: {e}\n  원문 저장: {raw_path}")
                continue

        gt = None
        score = None
        gt_path = os.path.join(base, f"ground_truth_{label}.json")
        if args.score and os.path.isfile(gt_path):
            with open(gt_path, encoding="utf-8") as f:
                gt = json.load(f)
            score = score_response(answer, gt, args.early_credit)
            scores.append(score)
            if modes:
                mode_scores.append(
                    score_modes(
                        answer, gt,
                        credit_s=args.early_credit,
                        late_decay=args.late_decay,
                        early_decay=args.early_decay,
                        modes=modes,
                    )
                )
        # 정답이 없어도 형식 검사는 할 수 있다 (payload 의 구간 수를 쓴다)
        issues = (
            score["response_issues"]
            if score
            else response_issues(
                answer.get("predictions", []) or [],
                expected_ks_from_payload(body),
            )
        )

        show(label, answer, gt, secs, usage, issues, score)
        if reuse:
            # 저장된 응답을 그대로 쓴 것이므로 다시 쓰지 않는다 — usage·elapsed 를
            # 덮으면 실제 호출 기록이 흐려진다.
            continue
        with open(
            os.path.join(out_dir, f"response_{label}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "payload": os.path.basename(path),
                    "provider": provider,
                    "endpoint": endpoint,
                    "elapsed_s": round(secs, 2),
                    "usage": usage,
                    "response_issues": issues,
                    "answer": answer,
                },
                f,
                ensure_ascii=False,
                indent=1,
            )

    if scores:
        agg = aggregate_scores(scores)
        print(f"\n{'=' * 74}\n채점 합산 ({len(scores)}개 윈도우, provider={provider})")
        print("=" * 74)
        print(json.dumps(agg, ensure_ascii=False, indent=1))
        with open(
            os.path.join(out_dir, "scores.json"), "w", encoding="utf-8"
        ) as f:
            json.dump({"provider": provider, "per_window": scores,
                       "aggregate": agg}, f, ensure_ascii=False, indent=1)
    if mode_scores:
        magg = aggregate_modes(mode_scores, modes)
        print(f"\n{'=' * 74}\n다중 기준 채점 ({len(mode_scores)}개 윈도우, "
              f"모드={','.join(modes)})")
        print("=" * 74)
        print(json.dumps(magg, ensure_ascii=False, indent=1))
        with open(
            os.path.join(out_dir, "scores_modes.json"), "w", encoding="utf-8"
        ) as f:
            json.dump({"provider": provider, "modes": list(modes),
                       "late_decay": args.late_decay,
                       "early_decay": args.early_decay,
                       "per_window": mode_scores, "aggregate": magg},
                      f, ensure_ascii=False, indent=1)
    print(f"\n응답 저장: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
