"""본 실험 전에 **모델이 원시 연산을 할 수 있는지** 싸게 잰다.

왜 필요한가
    `~/vlm` 은 본 실험 전에 `probe_bev_localise.py` 로 "이 모델이 BEV 에서 위치를
    짚을 수 있나"를 먼저 쟀고, Qwen3-VL-8B 는 9개 중 1개였다. 같은 리포트가
    이렇게도 적었다 — **"같은 모델들이 쌍 거리 계산도 못 한다. 4.5m 떨어진 행이
    있는 표를 주고 '30m 안에 움직이는 물체가 없다'고 답했다."**

    사고 예측은 그 원시 연산들 위에 얹힌 과제다. 밑이 안 되면 위도 안 된다.
    실제로 Qwen3-VL-32B 를 본 과제에 바로 넣었더니 5개 구간 전부 '사고' 라고 답했다.
    이 probe 를 먼저 돌렸으면 예측할 수 있었다.

무엇을 재는가 — 전부 합성 문제라 정답이 확실하고 데이터셋이 필요 없다.

    read       상태표에서 값 하나를 읽는다            (표를 읽을 수 있는가)
    distance   두 좌표 사이 거리를 낸다               (산술이 되는가)
    extrapol   등속으로 N초 뒤 위치를 낸다            (외삽이 되는가)
    closest    여러 대 중 가장 가까운 쌍을 고른다      (비교가 되는가)
    ruleout    명백히 안전한 장면을 '안전' 이라 한다   ← **늑대소년 검사**
    detect     명백히 위험한 장면을 '위험' 이라 한다

`ruleout` 과 `detect` 를 함께 봐야 한다. detect 만 높고 ruleout 이 낮으면
"무조건 위험" 이라 답하는 모델이고, 그건 본 과제에서 쓸 수 없다.

    ~/vlm/.venv-qwen/bin/python examples/probe_capability.py \
        --model Qwen/Qwen3-VL-32B-Instruct --n 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Callable, Dict, List, Optional, Tuple

SYSTEM = (
    "당신은 교통 상황 데이터를 읽고 계산하는 도구입니다.\n"
    "위치는 ENU 평면 [m], e=동쪽(+), n=북쪽(+). 방위각은 진북 기준 시계방향.\n"
    "추측하지 말고 주어진 수치로만 답하십시오."
)

SCHEMA_NUM = {"type": "object",
              "properties": {"answer": {"type": "number"}},
              "required": ["answer"]}
SCHEMA_STR = {"type": "object",
              "properties": {"answer": {"type": "string"}},
              "required": ["answer"]}
SCHEMA_BOOL = {"type": "object",
               "properties": {"answer": {"type": "boolean"}},
               "required": ["answer"]}


def _row(aid: str, t: float, e: float, n: float, kmh: float, extra: str = "") -> str:
    return f"   t={t:5.0f}  ({e:8.1f},{n:8.1f}) {kmh:6.0f}km/h  {extra}"


def _table(rng: random.Random, n_actors: int, n_steps: int = 4):
    """실제 payload 와 같은 모양의 상태표."""
    actors = []
    for i in range(n_actors):
        aid = f"V{i+1:03d}"
        e0, n0 = rng.uniform(-80, 80), rng.uniform(-80, 80)
        kmh = rng.choice([0, 12, 25, 34, 47, 58])
        hd = rng.choice([0, 90, 180, 270])
        actors.append({"id": aid, "e": e0, "n": n0, "kmh": kmh, "hd": hd})
    lines = []
    for a in actors:
        lines.append(f"### {a['id']} (승용차)")
        sp = a["kmh"] / 3.6
        ux, uy = math.sin(math.radians(a["hd"])), math.cos(math.radians(a["hd"]))
        for s in range(n_steps):
            lines.append(_row(a["id"], s, a["e"] + ux * sp * s, a["n"] + uy * sp * s,
                              a["kmh"], f"방위 {a['hd']}°"))
    return actors, "\n".join(lines)


# ------------------------------------------------------------ 문제 생성기

def gen_read(rng):
    actors, tbl = _table(rng, rng.randint(3, 6))
    a = rng.choice(actors)
    t = rng.randint(1, 3)
    q = (f"{tbl}\n\n{a['id']} 의 t={t} 시점 속도[km/h]는 얼마입니까? "
         f"숫자만 answer 에 넣으십시오.")
    return q, SCHEMA_NUM, float(a["kmh"]), "num", 0.5


def gen_distance(rng):
    e1, n1 = rng.uniform(-50, 50), rng.uniform(-50, 50)
    dx, dy = rng.uniform(-40, 40), rng.uniform(-40, 40)
    e2, n2 = e1 + dx, n1 + dy
    q = (f"차량 A 는 ({e1:.1f}, {n1:.1f}) 에, 차량 B 는 ({e2:.1f}, {n2:.1f}) 에 있습니다.\n"
         f"두 차량 중심 사이 거리[m]는 얼마입니까?")
    return q, SCHEMA_NUM, math.hypot(dx, dy), "num", 1.0


def gen_extrapol(rng):
    e, n = rng.uniform(-50, 50), rng.uniform(-50, 50)
    kmh = rng.choice([18, 36, 54, 72])
    hd = rng.choice([0, 90, 180, 270])
    dt = rng.choice([2, 3, 4])
    sp = kmh / 3.6
    ux, uy = math.sin(math.radians(hd)), math.cos(math.radians(hd))
    q = (f"차량이 ({e:.1f}, {n:.1f}) 에서 방위 {hd}° 로 {kmh}km/h 로 달립니다.\n"
         f"속도가 유지되면 {dt}초 뒤 **동쪽 좌표 e** 는 얼마입니까?")
    return q, SCHEMA_NUM, e + ux * sp * dt, "num", 1.0


def gen_closest(rng):
    n = rng.randint(4, 6)
    pts = []
    while len(pts) < n:
        p = (rng.uniform(-60, 60), rng.uniform(-60, 60))
        if all(math.dist(p, q) > 12 for q in pts):
            pts.append(p)
    # 한 쌍을 확실히 가깝게
    i = rng.randrange(n)
    j = (i + 1) % n
    pts[j] = (pts[i][0] + rng.uniform(3, 5), pts[i][1] + rng.uniform(-1, 1))
    ids = [f"V{k+1:03d}" for k in range(n)]
    body = "\n".join(f"  {ids[k]}  ({pts[k][0]:7.1f}, {pts[k][1]:7.1f})" for k in range(n))
    best, bd = None, 1e9
    for a in range(n):
        for b in range(a + 1, n):
            d = math.dist(pts[a], pts[b])
            if d < bd:
                best, bd = (ids[a], ids[b]), d
    q = (f"차량 위치:\n{body}\n\n"
         f"가장 가까운 두 차량은 무엇입니까? 'V001+V002' 형식으로 답하십시오.")
    return q, SCHEMA_STR, "+".join(sorted(best)), "pair", 0


def _scene(rng, dangerous: bool):
    """마지막 관측 시점까지의 **과거 이력**을 만든다.

    주의 — 미래를 그려 넣으면 안 된다. 접근 20m/s 로 미래 4초를 그리면 표 안에서
    두 차가 이미 교차해 멀어지고, 모델이 '충돌 안 함' 이라 답하는 것이 **옳은
    독해**가 된다. 이력은 t=-3..0 이고, t=0(마지막 관측)에서 여전히 접근 중이어야 한다.

    현재 판의 기하는 수치로 검산했다 (2026-08-26):
        detect  t=0 간격 21~22m, 접근 20m/s → t≈1.05s 에 중심거리 0.0~0.3m  충돌
        ruleout 같은 방향 평행, 간격 9~11m 가 5초 내내 유지            비충돌
    즉 **문제는 모호하지 않다.** 그럼에도 Qwen3-VL-32B 는 detect 8문항 전부를
    'false' 로 답한다 (detect 0.00 / ruleout 1.00) — 계산이 아니라 한쪽으로
    상수 응답을 하는 것이다. 같은 모델이 실제 과제(차량 26대)에서는 반대로
    5개 구간 전부 '사고' 라고 답했다. 장면의 밀도에 따라 방향만 바뀐다.
    """
    e, n = rng.uniform(-30, 30), rng.uniform(-30, 30)
    steps = [-3, -2, -1, 0]
    if dangerous:
        gap0 = rng.uniform(18, 26)      # t=0 에서의 간격 — 접근 20m/s 면 약 1초 뒤 접촉
        a = (e, n, 36.0, 90)            # 동쪽 36km/h = 10m/s
        b = (e + gap0, n, 36.0, 270)    # 마주 옴
    else:
        lat = rng.uniform(7, 12)        # 차로 간 여유, 계속 유지된다
        a = (e, n, 36.0, 90)
        b = (e + rng.uniform(-5, 5), n + lat, 36.0, 90)   # 같은 방향 나란히
    rows = []
    for tag, (x, y, kmh, hd) in (("A", a), ("B", b)):
        sp = kmh / 3.6
        ux, uy = math.sin(math.radians(hd)), math.cos(math.radians(hd))
        rows.append(f"### 차량 {tag} (승용차, 길이 4.6m 폭 1.9m)")
        for t in steps:                 # t<=0 : 과거 → 현재
            rows.append(_row(tag, t, x + ux * sp * t, y + uy * sp * t, kmh,
                             f"방위 {hd}°"))
    return "\n".join(rows)


def gen_ruleout(rng):
    q = (f"{_scene(rng, dangerous=False)}\n\n"
         f"위는 **t=0 까지의 과거 관측**입니다. 두 차량이 현재 속도·방향을 유지하면 "
         f"t=0 이후 5초 안에 **서로 충돌**합니까? true/false 로만 답하십시오.")
    return q, SCHEMA_BOOL, False, "bool", 0


def gen_detect(rng):
    q = (f"{_scene(rng, dangerous=True)}\n\n"
         f"위는 **t=0 까지의 과거 관측**입니다. 두 차량이 현재 속도·방향을 유지하면 "
         f"t=0 이후 5초 안에 **서로 충돌**합니까? true/false 로만 답하십시오.")
    return q, SCHEMA_BOOL, True, "bool", 0


TASKS: Dict[str, Callable] = {
    "read": gen_read, "distance": gen_distance, "extrapol": gen_extrapol,
    "closest": gen_closest, "ruleout": gen_ruleout, "detect": gen_detect,
}


def check(kind: str, got, want, tol) -> bool:
    if got is None:
        return False
    if kind == "num":
        try:
            return abs(float(got) - float(want)) <= tol
        except (TypeError, ValueError):
            return False
    if kind == "pair":
        return "+".join(sorted(str(got).replace(" ", "").split("+"))) == want
    if kind == "bool":
        return bool(got) is bool(want)
    return False


def _gemini_call(model: str, system: str, user: str, schema: dict,
                 key: str, timeout: int = 120) -> Optional[dict]:
    """구조화 출력으로 한 문항. 실패하면 None."""
    import urllib.request, urllib.error

    def clean(sc):
        """gemini 는 스키마 키워드를 일부만 받는다."""
        out = {"type": sc["type"].upper()}
        if "properties" in sc:
            out["properties"] = {k: clean(v) for k, v in sc["properties"].items()}
        if "required" in sc:
            out["required"] = sc["required"]
        return out

    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.0,
                             "responseMimeType": "application/json",
                             "responseSchema": clean(schema)},
    }
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        txt = d["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(txt)
    except Exception:
        return None


def run_gemini(args) -> int:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY 가 없습니다. `source env.sh` 후 다시 실행하십시오.")
        return 1
    model = args.model if "/" not in args.model else "gemini-2.5-flash"
    print(f"API 모델: {model}\n", flush=True)

    rng = random.Random(args.seed)
    results: Dict[str, List[bool]] = {}
    log = []
    for name in args.tasks.split(","):
        name = name.strip()
        if name not in TASKS:
            continue
        oks = []
        for i in range(args.n):
            q, schema, want, kind, tol = TASKS[name](rng)
            ans = _gemini_call(model, SYSTEM, q, schema, key)
            got = ans.get("answer") if isinstance(ans, dict) else None
            ok = check(kind, got, want, tol)
            oks.append(ok)
            log.append({"task": name, "i": i, "want": want, "got": got, "ok": ok})
        results[name] = oks
        bar = "".join("O" if o else "." for o in oks)
        print(f"  {name:9s} {sum(oks)/len(oks):5.2f}  {bar}", flush=True)

    _summary(model, results)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"model": model, "n": args.n, "seed": args.seed,
                   "results": results, "log": log},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n저장: {args.out}")
    return 0


def _summary(model: str, results: Dict[str, List[bool]]) -> None:
    print(f"\n{'=' * 56}\n{model}\n{'=' * 56}")
    prim = [t for t in ("read", "distance", "extrapol", "closest") if t in results]
    if prim:
        v = sum(sum(results[t]) for t in prim) / sum(len(results[t]) for t in prim)
        print(f"  원시 연산 (읽기·산술·외삽·비교)   {v:.2f}")
    if "ruleout" in results and "detect" in results:
        ro = sum(results["ruleout"]) / len(results["ruleout"])
        de = sum(results["detect"]) / len(results["detect"])
        print(f"  위험 탐지 (detect)               {de:.2f}")
        print(f"  안전 배제 (ruleout)              {ro:.2f}   ← 늑대소년 검사")
        if de >= 0.8 and ro <= 0.3:
            print("\n  ★ 무조건 '위험' 이라 답하는 모델이다. 본 과제에 쓸 수 없다.")
        elif de <= 0.3 and ro >= 0.8:
            print("\n  ★ 무조건 '안전' 이라 답하는 모델이다. 본 과제에 쓸 수 없다.")
        elif min(de, ro) >= 0.7:
            print("\n  ★ 양방향 판별이 된다. 본 실험을 진행할 만하다.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--n", type=int, default=8, help="과제당 문항 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--api", default=None, choices=("gemini",),
                    help="로컬 대신 API 로 묻는다 (키는 환경변수)")
    args = ap.parse_args()

    if args.api == "gemini":
        return run_gemini(args)

    import torch
    from transformers import AutoTokenizer
    import transformers as _tf
    import transformers.tokenization_utils as _tu
    if not hasattr(_tu, "PreTrainedTokenizerBase"):
        _tu.PreTrainedTokenizerBase = _tf.PreTrainedTokenizerBase
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    print(f"모델 적재: {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        from transformers import AutoModelForImageTextToText as _M
    except ImportError:
        from transformers import AutoModelForCausalLM as _M
    model = _M.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    fns = {id(s): build_transformers_prefix_allowed_tokens_fn(tok, JsonSchemaParser(s))
           for s in (SCHEMA_NUM, SCHEMA_STR, SCHEMA_BOOL)}
    print("  적재 완료\n", flush=True)

    rng = random.Random(args.seed)
    results: Dict[str, List[bool]] = {}
    log = []
    for name in args.tasks.split(","):
        name = name.strip()
        if name not in TASKS:
            continue
        oks = []
        for i in range(args.n):
            q, schema, want, kind, tol = TASKS[name](rng)
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": q}]
            prompt = tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt").to(model.device)
            n_in = enc["input_ids"].shape[-1]
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False,
                                     prefix_allowed_tokens_fn=fns[id(schema)],
                                     pad_token_id=tok.eos_token_id)
            txt = tok.decode(gen[0][n_in:], skip_special_tokens=True)
            try:
                got = json.loads(txt[txt.find("{"):txt.rfind("}") + 1]).get("answer")
            except Exception:
                got = None
            ok = check(kind, got, want, tol)
            oks.append(ok)
            log.append({"task": name, "i": i, "want": want, "got": got, "ok": ok})
        results[name] = oks
        acc = sum(oks) / len(oks)
        bar = "".join("O" if o else "." for o in oks)
        print(f"  {name:9s} {acc:5.2f}  {bar}", flush=True)

    print(f"\n{'=' * 56}")
    print(f"{args.model}")
    print(f"{'=' * 56}")
    prim = [t for t in ("read", "distance", "extrapol", "closest") if t in results]
    if prim:
        v = sum(sum(results[t]) for t in prim) / sum(len(results[t]) for t in prim)
        print(f"  원시 연산 (읽기·산술·외삽·비교)   {v:.2f}")
    if "ruleout" in results and "detect" in results:
        ro = sum(results["ruleout"]) / len(results["ruleout"])
        de = sum(results["detect"]) / len(results["detect"])
        print(f"  위험 탐지 (detect)               {de:.2f}")
        print(f"  안전 배제 (ruleout)              {ro:.2f}   ← 늑대소년 검사")
        if de >= 0.8 and ro <= 0.3:
            print("\n  ★ 무조건 '위험' 이라 답하는 모델이다. 본 과제에 쓸 수 없다.")
        elif min(de, ro) >= 0.7:
            print("\n  ★ 양방향 판별이 된다. 본 실험을 진행할 만하다.")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"model": args.model, "n": args.n, "seed": args.seed,
                   "results": {k: v for k, v in results.items()}, "log": log},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
