"""생성된 payload 를 **로컬 모델**(transformers)로 답하게 한다.

`ask_llm.py` 의 로컬판이다. API 키가 없어도, 비용 없이, 원하는 만큼 반복 호출할 수
있으므로 **표본 잡음 측정과 규모 확대**가 가능해진다.

출력은 `ask_llm.py` 와 같은 형식(`response_<창>.json`)이라 기존 채점기가 그대로 쓴다.

    # 창 하나
    .venv-qwen/bin/python examples/ask_local.py out/windows_.../llm_payload_0-5.json

    # 디렉터리 전체, 같은 입력을 3회씩 (잡음 측정)
    ... examples/ask_local.py <디렉터리> --repeat 3 --out-dir out/local_run

주의 — 이 스크립트는 **`~/vlm/.venv-qwen`** 으로 돌린다 (transformers 가 거기 있다).
payload 생성과 채점은 iitp 의 `.venv` 로 한다. 두 환경은 파일로만 주고받는다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import List, Optional, Tuple

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

JSON_INSTRUCTION = """

--- 출력 형식 ---
설명 없이 **JSON 객체 하나만** 출력하십시오. 코드펜스도 붙이지 마십시오.

{"predictions": [{"k": <정수>, "interval_s": "<구간>", "accident_expected": <true|false>,
  "involved_actor_ids": ["<액터 id>", ...], "reason": "<한 문장>"}, ...],
 "data_limitations": "<없으면 빈 문자열>"}

질문에 제시된 k 를 **각각 정확히 한 번씩** 포함하십시오.
"""


def payload_texts(doc: dict) -> Tuple[str, str]:
    """payload JSON → (system, user) 평문."""
    sysblk = doc.get("system")
    if isinstance(sysblk, list):
        system = "\n".join(b.get("text", "") for b in sysblk if isinstance(b, dict))
    else:
        system = sysblk or ""
    parts: List[str] = []
    for m in doc.get("messages", []):
        c = m.get("content")
        if isinstance(c, list):
            parts += [b.get("text", "") for b in c if isinstance(b, dict)]
        elif isinstance(c, str):
            parts.append(c)
    return system, "\n\n".join(p for p in parts if p)


def extract_json(text: str) -> Optional[dict]:
    """생성문에서 첫 균형 JSON 객체를 꺼낸다."""
    t = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.M)
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="llm_payload_*.json 또는 그것들이 있는 디렉터리")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out-dir", default=None, help="기본은 payload 와 같은 곳")
    ap.add_argument("--repeat", type=int, default=1, help="같은 입력 반복 횟수")
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 이면 그리디 (결정론적)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-guided", action="store_true",
                    help="제약 디코딩을 끈다 (JSON 구조 오류를 감수)")
    args = ap.parse_args()

    if os.path.isdir(args.target):
        files = sorted(os.path.join(args.target, f) for f in os.listdir(args.target)
                       if f.startswith("llm_payload_") and f.endswith(".json"))
    else:
        files = [args.target]
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("payload 를 찾지 못했습니다.")
        return 1

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"모델 적재: {args.model}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print(f"  적재 완료 {time.time() - t0:.1f}s")

    # 제약 디코딩 — 스키마를 벗어나는 토큰을 아예 못 내게 한다.
    # 32B 가 predictions 배열을 ']' 로 닫지 않고 다음 키로 넘어가는 구조 오류를
    # 냈고, 1000건 넘게 돌릴 것이므로 사후 파싱 보정에 기대면 안 된다.
    prefix_fn = None
    if not args.no_guided:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # lm-format-enforcer 0.11.3 은 transformers 4.x 위치를 찾는다.
        # 5.x 에서 PreTrainedTokenizerBase 가 옮겨갔을 뿐이라 그 이름만 되돌려준다.
        import transformers as _tf
        import transformers.tokenization_utils as _tu
        if not hasattr(_tu, "PreTrainedTokenizerBase"):
            _tu.PreTrainedTokenizerBase = _tf.PreTrainedTokenizerBase

        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )
        from traffic_llm.accident_qa import PREDICTION_SCHEMA

        prefix_fn = build_transformers_prefix_allowed_tokens_fn(
            tok, JsonSchemaParser(PREDICTION_SCHEMA))
        print("  제약 디코딩: PREDICTION_SCHEMA 적용")

    out_dir = args.out_dir
    n_ok = n_fail = 0
    for path in files:
        doc = json.load(open(path, encoding="utf-8"))
        system, user = payload_texts(doc)
        label = os.path.basename(path)[len("llm_payload_"):-len(".json")]
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user + JSON_INSTRUCTION}]
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        n_in = enc["input_ids"].shape[-1]

        for r in range(1, args.repeat + 1):
            torch.manual_seed(args.seed + r)
            t1 = time.time()
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    prefix_allowed_tokens_fn=prefix_fn,
                    pad_token_id=tok.eos_token_id)
            text = tok.decode(gen[0][n_in:], skip_special_tokens=True)
            dt = time.time() - t1
            ans = extract_json(text)
            ok = bool(ans and ans.get("predictions"))
            n_ok += ok
            n_fail += (not ok)

            d = out_dir or os.path.dirname(path)
            suffix = "" if args.repeat == 1 else f"_r{r}"
            os.makedirs(d, exist_ok=True)
            json.dump({"payload": os.path.basename(path), "provider": "local",
                       "model": args.model, "elapsed_s": round(dt, 2),
                       "temperature": args.temperature, "seed": args.seed + r,
                       "usage": {"input": n_in, "output": int(gen.shape[-1] - n_in)},
                       "guided": not args.no_guided,
                       "parsed": ok, "answer": ans, "raw_text": text},
                      open(os.path.join(d, f"response_{label}{suffix}.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)
            n_pred = len(ans.get("predictions", [])) if ans else 0
            print(f"  {label}{suffix}: 입력 {n_in} 토큰 · 생성 "
                  f"{int(gen.shape[-1]-n_in)} · {dt:.1f}s · "
                  f"{'파싱 OK, 구간 %d개' % n_pred if ok else '파싱 실패'}")

    print(f"\n완료 — 성공 {n_ok} / 실패 {n_fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
