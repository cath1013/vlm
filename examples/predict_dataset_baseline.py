"""생성한 지도학습 데이터에서 **규칙 기반 사전확률**의 베이스라인을 잰다.

학습 모델이 이겨야 하는 기준선이 무엇인지 정직하게 재는 것이 목적이다.

가장 중요한 것은 **정답 라벨이 임의인 표본을 빼고 재는 것**이다. 차량이 아직
갈림길에 닿지 않았으면 여러 후보가 실제 궤적과 똑같이 가깝고, 그때 `best_candidate`
는 목록 순서를 고른 것이다. 사전확률의 argmax 도 같은 순서로 동률을 깨므로 **둘이
구조적으로 일치한다** — 그 표본만 놓고 보면 정확도가 75.7% 로 나온다. 유일한 정답이
있는 표본만 보면 61.1% 다. 후자가 실제 성능이다 (전체 607 시나리오 실측).

세 수치를 따로 본다.

    후보 top-1   사전확률 최고 후보가 정답 후보와 같은 비율. **이것이 과제다.**
    기동 라벨    정답 후보의 기동 라벨을 맞히는 비율. 갈 수 있는 방향이 보통
                 하나뿐이라 이미 높다 — 여기엔 배울 것이 거의 없다.
    무작위       후보를 균등하게 고를 때의 기대 정확도 = mean(1/N).

    .venv/bin/python examples/predict_dataset_baseline.py out/predict_dataset
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

# 생성기와 같은 값을 쓴다 (구 버전 파일에는 플래그가 없어 여기서 다시 판정한다).
TIE_EPS_M = 0.05


def load(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def tied_of(r):
    """정답 동률 후보 인덱스. 저장돼 있으면 그것을 쓰고, 없으면 다시 판정한다."""
    if r.get("tied_best"):
        return r["tied_best"]
    errs = r["candidate_errors_m"]
    lo = min(errs)
    return [i for i, e in enumerate(errs) if e <= lo + TIE_EPS_M]


def main(argv):
    ddir = argv[1] if len(argv) > 1 else "out/predict_dataset"
    path = os.path.join(ddir, "samples.jsonl")
    if not os.path.exists(path):
        print(f"없음: {path}", file=sys.stderr)
        return 1

    n_all = n_unmatched = n_single = n_incomplete = 0
    scored = 0
    amb = amb_hit = 0
    uniq = uniq_hit = uniq_man = 0
    rand_sum = 0.0
    member_hit = 0
    by_n = Counter()
    by_n_hit = Counter()
    man_dist = Counter()
    by_len = Counter()
    by_len_hit = Counter()
    err_sum = 0.0
    splits = Counter()
    towns = Counter()
    man_amb = 0

    for r in load(path):
        n_all += 1
        splits[r.get("dataset_split") or "?"] += 1
        towns[r.get("town") or "?"] += 1
        if not r.get("future_complete", True):
            n_incomplete += 1
        if not r.get("matched"):
            n_unmatched += 1
            continue
        cands = r["candidates"]
        n = len(cands)
        if n < 2:
            n_single += 1
            continue
        scored += 1
        rand_sum += 1.0 / n
        err_sum += r.get("match_error_m") or 0.0
        man_dist[r["maneuver"]] += 1
        # 사전확률은 후보 특징 벡터의 0번 칸이다 (predict_model.candidate_features)
        priors = [c[0] for c in cands]
        pick = priors.index(max(priors))
        tied = tied_of(r)
        member_hit += pick in tied
        if len(tied) > 1:
            amb += 1
            amb_hit += pick == r["best_candidate"]
            if r.get("maneuver_ambiguous"):
                man_amb += 1
            continue  # 정답이 임의이므로 아래 집계에서 뺀다
        hit = pick == r["best_candidate"]
        uniq += 1
        uniq_hit += hit
        meta = r.get("candidate_meta") or []
        if meta:
            uniq_man += meta[pick].get("maneuver") == meta[r["best_candidate"]].get(
                "maneuver"
            )
        by_n[n] += 1
        by_n_hit[n] += hit
        L = len(r.get("target_offsets") or [])
        by_len[L] += 1
        by_len_hit[L] += hit

    if not scored:
        print("채점 가능한 표본이 없습니다.")
        return 1

    print(f"표본 총 {n_all:,}건   분할 {dict(splits)}   타운 {len(towns)}종")
    print(f"  정답 없음(matched=false)   {n_unmatched:,} ({n_unmatched/n_all:.1%})"
          f"  ← 실제 궤적이 어느 후보와도 3m 안에서 맞지 않음")
    print(f"  후보 1개뿐(고를 것 없음)    {n_single:,}")
    print(f"  미래 궤적 불완전(참고)      {n_incomplete:,}")
    print(f"  → 다후보 표본              {scored:,}")
    print(f"      정답 라벨 임의          {amb:,} ({amb/scored:.1%})"
          f"   그중 기동도 임의 {man_amb:,}")
    print(f"      정답 라벨 유일          {uniq:,} ({uniq/scored:.1%})"
          f"   ← 아래 수치의 모집단\n")

    print(f"후보 top-1 (사전확률)   {uniq_hit/uniq:.1%}   ({uniq_hit:,}/{uniq:,})")
    print(f"기동 라벨               {uniq_man/uniq:.1%}")
    print(f"무작위 기대치           {rand_sum/scored:.1%}")
    print(f"정답 후보 평균 오차     {err_sum/scored:.2f} m\n")

    print(f"참고 — 라벨 임의 표본을 넣으면 {(uniq_hit+amb_hit)/scored:.1%} 로 보인다"
          f" (임의 표본만 {amb_hit/amb:.1%}). 목록 순서가 일치해 생기는 값이므로"
          f" 성능이 아니다.")
    print(f"참고 — 동률 집합 안에 들면 정답으로 볼 때 {member_hit/scored:.1%}"
          f" (관대한 상한).\n")

    print("후보 수별 top-1 (라벨 유일 표본)")
    for n in sorted(by_n):
        print(f"  후보 {n}개: {by_n_hit[n]/by_n[n]:6.1%}  (n={by_n[n]:,}, "
              f"무작위 {1/n:.1%})")
    print("\n미래 관측 길이별 top-1 (라벨 유일 표본)")
    for L in sorted(by_len):
        print(f"  {L}초: {by_len_hit[L]/by_len[L]:6.1%}  (n={by_len[L]:,})")
    print("\n정답 기동 분포 (다후보 표본)")
    for m, c in man_dist.most_common():
        print(f"  {m}: {c:,} ({c/scored:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
