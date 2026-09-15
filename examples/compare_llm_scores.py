"""여러 경로 예측기 조건의 LLM 채점 결과를 나란히 비교한다.

같은 시나리오·같은 창·같은 채점 기준에서 **경로 예측기만 바꿔** 만든 payload 의 결과를
비교하는 것이 목적이다. 그래야 차이가 예측기에서 왔다고 말할 수 있다.

    python examples/compare_llm_scores.py \\
        rule=out/windows_example/DeepAccident_mini/ko/accident \\
        rank=out/windows_rank/DeepAccident_mini/ko \\
        waypoints=out/windows_waypoints/DeepAccident_mini/ko

각 디렉터리의 `responses/scores.json` 을 읽는다. 창을 하나씩 호출하면 그 파일이 매번
덮어써지므로, 먼저 `ask_llm.py <디렉터리> --score --rescore` 로 전부 합산해 두어야 한다.

채점 구조 — 창 종류에 따라 세는 방식이 다르다
    사고가 지평 안에 있는 창 → **사건 단위**로 한 번 센다 (TP 또는 FN).
        조기 인정 구간 안에서 한 번이라도 사고를 부르면 TP.
    사고가 없는 창 → **구간 단위**로 센다 (버킷마다 TN 또는 FP).
    그래서 TP+FN 은 사고 창 수와 같고, TN+FP 는 무사고 창의 채점 가능한 버킷 수다.
"""

from __future__ import annotations

import io
import json
import os
import sys
import unicodedata


def _w(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _pad(s: str, width: int, right: bool = False) -> str:
    fill = " " * max(0, width - _w(s))
    return (fill + s) if right else (s + fill)


def load(d: str):
    p = os.path.join(d, "responses", "scores.json")
    if not os.path.isfile(p):
        raise SystemExit(
            f"없음: {p}\n  먼저 `ask_llm.py {d} --score --rescore` 를 돌리십시오."
        )
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def predictor_of(d: str) -> str:
    p = os.path.join(d, "manifest.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return (json.load(f).get("config") or {}).get("predictor") or "?"
    return "?"


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1
    conds = []
    for spec in argv[1:]:
        name, _, d = spec.partition("=")
        if not d:
            d, name = name, os.path.basename(name.rstrip("/"))
        conds.append((name, d, load(d)))

    # 창 구성이 같은지 먼저 본다. 다르면 비교가 성립하지 않는다.
    sets = [tuple(w["window"]["label"] for w in s["per_window"]) for _, _, s in conds]
    if len(set(sets)) != 1:
        print("경고: 조건마다 채점된 창이 다릅니다 — 합산 비교가 성립하지 않습니다.",
              file=sys.stderr)
        for (n, _, _), ws in zip(conds, sets):
            print(f"  {n}: {len(ws)}개 {list(ws)}", file=sys.stderr)
    labels = sets[0]

    LW = max(_w(n) for n, _, _ in conds) + 2
    print(f"창 {len(labels)}개: {', '.join(labels)}\n")
    print(_pad("조건", LW) + _pad("예측기", 32)
          + "".join(_pad(k, 7, True) for k in ("TP", "FN", "FP", "TN"))
          + _pad("검출률", 9, True) + _pad("평균선행", 10, True)
          + _pad("오탐률", 9, True))
    for name, d, s in conds:
        a = s["aggregate"]
        c, ev = a["counts"], a["events"]
        lead = ev.get("mean_lead_s")
        neg = c["FP"] + c["TN"]
        print(_pad(name, LW) + _pad(predictor_of(d), 32)
              + "".join(_pad(str(c[k]), 7, True) for k in ("TP", "FN", "FP", "TN"))
              + _pad(f"{ev['detection_rate']:.0%}", 9, True)
              + _pad("-" if lead is None else f"{lead:+.2f}s", 10, True)
              + _pad("-" if not neg else f"{c['FP']/neg:.0%}", 9, True))

    # 창별로 어디가 갈렸는지. 합산만 보면 어느 창에서 달라졌는지 알 수 없다.
    print("\n창별 판정")
    head = _pad("창", 7) + _pad("정답", 9)
    for n, _, _ in conds:
        head += _pad(n, max(14, _w(n) + 2))
    print(head)
    for i, lab in enumerate(labels):
        # 지평 안에 사고가 없는 창은 `event` 자체가 없다 (사건 단위로 셀 것이 없다)
        ev0 = conds[0][2]["per_window"][i].get("event")
        truth = ev0["k_true"] if ev0 else None
        cells = ""
        for name, _, s in conds:
            w = s["per_window"][i]
            e, c = w.get("event"), w["counts"]
            if e and e.get("k_true") is not None:
                cell = (f"{e['timing']} k={e['credited_k']}"
                        if e["detected"] else "미검출")
            else:
                cell = f"오탐 {c['FP']}" if c["FP"] else "정상 OK"
            cells += _pad(cell, max(14, _w(name) + 2))
        print(_pad(lab, 7)
              + _pad("-" if truth is None else f"k={truth}", 9) + cells)

    print("\n읽는 법")
    print("  정답 k = 사고가 일어난 구간 번호, '-' 는 지평 안에 사고가 없는 창.")
    print("  early/exact = 조기·정시 검출. 검출률은 사고 창만 놓고 센다.")
    print("  오탐률 = 무사고 구간 중 사고라고 부른 비율 (FP/(FP+TN)).")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    raise SystemExit(main(sys.argv))
