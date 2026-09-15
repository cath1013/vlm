"""내보낸 장면 브리핑에서 **범주형 판정만** 걷어낸다 (수치는 그대로).

왜
    조건 G(gap table + iitp 브리핑 전체)가 vlm 의 천장 0.728 을 넘기는커녕
    0.631 로 내려갔다 — 오탐이 8 → 31 로 늘고 정밀도가 0.795 → 0.587 로 무너졌다.
    이는 vlm 이 A → B → C 에서 관측자 산문을 넣었을 때와 **같은 양상**이다:
    "모델은 모든 입력을 위험의 증거로 취급하고 위험하지 않다는 증거로는 결코
    취급하지 않는다."

    iitp 브리핑에는 숫자(TTC 1.8s, gap 47.3m)와 범주 라벨
    (`orthogonal entry conflict expected`, `keeping lane`,
     `predicted path: off-candidate path`)이 **섞여** 있다. G 는 그 섞인 것을
    쟀으므로, iitp 의 지도 정합 *수치* 가 쓸모 있는지는 아직 답이 없다.

    이 필터가 만드는 것이 조건 H 다. **G 와 단 한 가지만 다르다** — 범주 라벨의
    유무. 그래서 차이가 생기면 원인이 하나로 특정된다.

무엇을 지우는가 — 위험/상태에 대한 **판정**만
    1. 액터 줄의 맨 기동 필드      `| stopped |`, `| keeping lane |`
    2. 예상 경로 라벨              `| predicted path: remains stopped |`
    3. 상호작용 유형 라벨          `| car following |`, `| orthogonal entry conflict expected |`
    4. 불확실성 훈수               `| NOTE: position/speed uncertain`

무엇을 남기는가 — 측정값과 지도 사실
    도로·차선 번호, 제한속도, 교차로까지 거리, 속력, 방위, 가속도,
    TTC·헤드웨이·간격·도달시간차, 접근 쌍의 거리 변화와 접근율,
    존재확신·위치정확도·관측거리, 관측자 목록, 상대 위치, BEV 개략도.

    .venv/bin/python examples/strip_categorical.py \\
        out/vlm_scene/val_h1.0_en.json out/vlm_scene/val_h1.0_en_numeric.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from typing import List, Tuple

_DIGIT = re.compile(r"\d")
# 지워도 되는 '라벨' — 파이프 구획 안에 숫자가 하나도 없는 조각.
# 숫자가 있으면 측정값이므로 절대 건드리지 않는다.
_DROP_PREFIX = ("predicted path:", "NOTE:")


def strip_line(line: str, stats: collections.Counter) -> str:
    if not line.startswith("- ") or "|" not in line:
        return line
    head, *fields = [f.strip() for f in line.split("|")]
    kept: List[str] = []
    for f in fields:
        low = f.lower()
        if any(low.startswith(p.lower()) for p in _DROP_PREFIX):
            stats[f.split(":")[0].strip()] += 1
            continue
        if not _DIGIT.search(f):
            # 숫자 없는 조각 = 순수 라벨. 관측자 목록만 예외로 남긴다
            # (누가 봤는지는 위험 판정이 아니라 인지 출처다).
            # 예외 — 이것들은 위험 판정이 아니라 **신원·출처** 정보다.
            # `own actor id ...` 는 관측자 이름(ego_vehicle)과 액터 id
            # (EGO_ego_vehicle)를 잇는 유일한 단서라, 지우면 프롬프트가 둘을
            # 같은 차량으로 볼 수 없다.
            if (low.startswith("seen by") or low.startswith("perceives")
                    or low.startswith("own actor id")
                    or low.startswith("sole observer")):
                kept.append(f)
                continue
            stats[f if len(f) < 40 else f[:37] + "..."] += 1
            continue
        kept.append(f)
    return " | ".join([head] + kept)


def strip_text(text: str) -> Tuple[str, collections.Counter]:
    stats: collections.Counter = collections.Counter()
    out = [strip_line(ln, stats) for ln in text.split("\n")]
    return "\n".join(out), stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()

    d = json.load(open(args.src, encoding="utf-8"))
    total: collections.Counter = collections.Counter()
    before = after = 0
    for v in d.values():
        before += len(v["text"])
        v["text"], st = strip_text(v["text"])
        after += len(v["text"])
        total.update(st)

    json.dump(d, open(args.dst, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"{len(d)}개 · {before:,}자 → {after:,}자 "
          f"({(before - after) / before * 100:.1f}% 제거)")
    print("\n지운 라벨 (상위 20):")
    for lab, n in total.most_common(20):
        print(f"  {n:7,}  {lab}")
    n_kinds = len(total)
    print(f"\n  라벨 종류 {n_kinds}종 · 총 {sum(total.values()):,}개 제거")
    print(f"  저장: {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
