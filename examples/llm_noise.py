"""**같은 입력에 대한 LLM 표본 잡음**을 재고, 예측기 차이와 크기를 비교한다.

왜 필요한가
    2026-08-04 실험에서 예측기별 정확도가 0.643 / 0.679 / 0.714 로 올라갔는데,
    실제로 바뀐 것은 FP 가 8 → 7 → 6, 즉 **구간 판정 하나씩**이었다.
    LLM 이 같은 입력에도 판정 한두 개씩 흔들린다면 그 "개선" 은 전부 잡음이다.

    `rule_run2` 는 정확히 이 질문을 위해 **같은 payload 를 두 번째로 호출한**
    표본이다. 두 응답의 불일치가 **잡음 바닥**이고, 예측기를 바꿨을 때의 불일치가
    **신호**다. 신호가 바닥에 묻히는지 본다.

정답이 필요 없다
    두 응답을 서로 비교할 뿐이므로, payload 가 그 뒤 바뀌어 재채점이 불가능한
    것과 무관하게 측정할 수 있다. 결제된 응답을 그대로 쓴다.

    .venv/bin/python examples/llm_noise.py out/llm_responses_archive
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from typing import Dict, List, Optional, Tuple


def load_answers(d: str) -> Dict[str, Dict[int, dict]]:
    """디렉터리 → {창 라벨: {k: 예측}}"""
    out: Dict[str, Dict[int, dict]] = {}
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not (fn.startswith("response_") and fn.endswith(".json")):
            continue
        label = fn[len("response_"):-len(".json")]
        doc = json.load(open(os.path.join(d, fn), encoding="utf-8"))
        preds = ((doc.get("answer") or {}).get("predictions")) or []
        by_k: Dict[int, dict] = {}
        for p in preds:
            k = p.get("k")
            if isinstance(k, int) and k not in by_k:  # 중복 k 는 첫 것만
                by_k[k] = p
        out[label] = by_k
    return out


def compare(a: Dict[str, Dict[int, dict]],
            b: Dict[str, Dict[int, dict]]) -> Optional[dict]:
    """겹치는 창의 구간별 일치도."""
    labels = sorted(set(a) & set(b))
    if not labels:
        return None
    n = flips = id_diff = both_pos = 0
    detail: List[Tuple[str, int, bool, bool]] = []
    for lb in labels:
        for k in sorted(set(a[lb]) & set(b[lb])):
            pa, pb = a[lb][k], b[lb][k]
            va = bool(pa.get("accident_expected"))
            vb = bool(pb.get("accident_expected"))
            n += 1
            if va != vb:
                flips += 1
                detail.append((lb, k, va, vb))
            elif va and vb:
                both_pos += 1
                if set(pa.get("involved_actor_ids") or []) != set(
                        pb.get("involved_actor_ids") or []):
                    id_diff += 1
    return {"labels": labels, "n_buckets": n, "flips": flips,
            "agreement": (n - flips) / n if n else 0.0,
            "both_positive": both_pos, "id_mismatch": id_diff, "detail": detail}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", help="out/llm_responses_archive")
    args = ap.parse_args()

    conds = {}
    for name in ("rule", "rule_run2", "rank", "waypoints"):
        p = os.path.join(args.archive, name)
        ans = load_answers(p)
        if ans:
            conds[name] = ans
            print(f"  {name:11s} 창 {len(ans)}개")

    print("\n" + "=" * 74)
    print("잡음 바닥 — 같은 입력, 같은 예측기, 두 번 호출")
    print("=" * 74)
    noise = compare(conds.get("rule", {}), conds.get("rule_run2", {}))
    if not noise:
        print("  rule / rule_run2 겹치는 창이 없다.")
        return 1
    print(f"  창 {noise['labels']}  구간 {noise['n_buckets']}개")
    print(f"  판정 불일치 {noise['flips']}개  →  일치율 {noise['agreement']:.3f}")
    print(f"  둘 다 '사고' 인 구간 {noise['both_positive']}개 중 "
          f"관련차량 지목이 다른 것 {noise['id_mismatch']}개")
    for lb, k, va, vb in noise["detail"]:
        print(f"    창 {lb} k={k}: run1={'사고' if va else '없음'} / "
              f"run2={'사고' if vb else '없음'}")

    print("\n" + "=" * 74)
    print("신호 — 예측기를 바꿨을 때 (같은 창만 비교)")
    print("=" * 74)
    rows = []
    for x, y in itertools.combinations(["rule", "rank", "waypoints"], 2):
        if x not in conds or y not in conds:
            continue
        c = compare(conds[x], conds[y])
        if c:
            rows.append((f"{x} vs {y}", c))
            print(f"  {x:10s} vs {y:10s}  구간 {c['n_buckets']:3d}  "
                  f"불일치 {c['flips']:2d}  일치율 {c['agreement']:.3f}")

    # 같은 창만 놓고 잡음과 신호를 직접 견준다
    print("\n" + "=" * 74)
    print("직접 비교 — 잡음이 측정된 창(0-1..0-4)에 한정")
    print("=" * 74)
    sub = set(noise["labels"])
    def restrict(d):
        return {k: v for k, v in d.items() if k in sub}
    print(f"  {'비교':26s} {'구간':>5s} {'불일치':>6s} {'일치율':>7s}")
    base = f"rule vs rule_run2 (잡음)"
    print(f"  {base:26s} {noise['n_buckets']:5d} {noise['flips']:6d} "
          f"{noise['agreement']:7.3f}")
    for x, y in (("rule", "rank"), ("rule", "waypoints"), ("rank", "waypoints")):
        if x not in conds or y not in conds:
            continue
        c = compare(restrict(conds[x]), restrict(conds[y]))
        if c:
            lbl = f"{x} vs {y}"
            print(f"  {lbl:26s} {c['n_buckets']:5d} {c['flips']:6d} "
                  f"{c['agreement']:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
