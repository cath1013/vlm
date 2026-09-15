"""저장된 기준선 레코드를 구간(k)별로 분해한다.

묻는 것: **예측 지평 5초의 뒤쪽 구간이 애초에 답할 수 있는 질문인가.**

`~/vlm` 의 `horizon_sweep.py` 가 같은 방법(등속 외삽 + 회전 사각형 간격)으로
"몇 초 앞까지 운동학으로 결정되는가"를 쟀고, **2초에서 이미 AUC 0.603, 3초면
0.565(우연)** 였다. iitp 는 5초를 1초 구간 5개로 묻는다. 그렇다면 k=3,4,5 는
어떤 방법으로도 우연 수준이어야 한다 — 그것을 확인한다.

구간별로 낸다.
    양성률       그 구간에 실제로 사고가 있는 비율
    AUC          위험도(=-간격)가 정답을 얼마나 가르는가. 0.5 = 정보 없음
    임계값 성능   never 대비 균형정확도

    .venv/bin/python examples/analyze_records.py out/baselines/records_train_sensor3d.jsonl
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Tuple

INF = float("inf")


def load(path: str) -> List[Tuple[int, Optional[float], bool]]:
    """(k, gap, truth) — 채점 가능한 구간만."""
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        gaps = d["gaps"]
        for i, e in enumerate(d["gt"].get("expected") or []):
            if not e.get("scorable", True):
                continue
            k = e.get("k", i + 1)
            g = gaps[i][0] if i < len(gaps) else None
            rows.append((k, g, bool(e.get("accident_expected"))))
    return rows


def auc(scores: List[float], labels: List[bool]) -> Optional[float]:
    """순위 기반 AUC (동점은 평균 순위)."""
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    s = sum(r for r, l in zip(ranks, labels) if l)
    return (s - pos * (pos + 1) / 2.0) / (pos * neg)


def best_threshold(gaps: List[float], labels: List[bool]) -> Tuple[float, float]:
    """균형정확도를 최대화하는 임계값과 그 값."""
    best_t, best_v = 0.0, 0.0
    for t in [x * 0.1 for x in range(0, 51)]:
        tp = fp = tn = fn = 0
        for g, l in zip(gaps, labels):
            p = g <= t
            if l and p: tp += 1
            elif l and not p: fn += 1
            elif not l and p: fp += 1
            else: tn += 1
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        v = (sens + spec) / 2.0
        if v > best_v:
            best_t, best_v = t, v
    return best_t, best_v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+")
    args = ap.parse_args()

    for path in args.records:
        rows = load(path)
        print(f"\n{'=' * 78}\n{path}\n{'=' * 78}")
        print(f"채점 가능 구간 {len(rows)}개")
        print(f"\n{'구간':>4s} {'n':>6s} {'양성':>6s} {'양성률':>7s} {'AUC':>7s} "
              f"{'최적임계':>8s} {'균형정확':>8s} {'never':>7s}")
        for k in sorted({r[0] for r in rows}):
            sub = [r for r in rows if r[0] == k]
            finite = [(g, l) for _, g, l in sub if g is not None]
            if not finite:
                continue
            gaps = [g for g, _ in finite]
            labels = [l for _, l in finite]
            pos = sum(labels)
            # 간격은 **작을수록 위험**이므로 위험도 점수는 -gap 이다.
            a = auc([-g for g in gaps], labels)
            t, bal = best_threshold(gaps, labels)
            never = 1.0 - pos / len(labels)
            print(f"{k:4d} {len(labels):6d} {pos:6d} {pos/len(labels):7.3f} "
                  f"{(f'{a:.3f}' if a is not None else '  —  '):>7s} "
                  f"{t:8.2f} {bal:8.3f} {never:7.3f}")

        # 전체
        finite = [(g, l) for _, g, l in rows if g is not None]
        gaps = [g for g, _ in finite]
        labels = [l for _, l in finite]
        a = auc([-g for g in gaps], labels)
        t, bal = best_threshold(gaps, labels)
        pos = sum(labels)
        print(f"{'전체':>4s} {len(labels):6d} {pos:6d} {pos/len(labels):7.3f} "
              f"{(f'{a:.3f}' if a is not None else '  —  '):>7s} "
              f"{t:8.2f} {bal:8.3f} {1.0-pos/len(labels):7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
