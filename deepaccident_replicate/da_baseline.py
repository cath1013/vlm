"""DeepAccident 사고 판정 규칙 — 같은 축에서 비교하기 위한 이식.

**무엇을 이식했고 무엇은 못 했는가.**

DeepAccident 의 사고 예측은 두 부분이다.

    (1) 카메라 → BEV → 인스턴스 미래 운동 예측        (V2XFormer, 학습된 모델)
    (2) 예측된 미래에서 사고를 읽어내는 후처리 규칙     (임계값 하나)

여기 있는 것은 **(2) 뿐**이다. (1) 은 이 저장소에서 돌릴 수 없다 — 공개 저장소에
DeepAccident 로 학습한 가중치가 없다. `README.md` 의 Model Zoo 절은 통째로 주석
처리돼 있고(`[//]: # (## Model Zoo)`), 그 안에 살아 있는 두 링크마저 nuScenes 용
BEVDet 가중치다(지표가 mAP/NDS/Map IoU). 즉 (1) 을 쓰려면 242GB 카메라 데이터로
20 에폭을 새로 학습해야 하고, 그렇게 해도 논문 Table 3·4 는 재현되지 않는다 —
공개된 융합 모듈은 CoBEVT 가 아니라 평균 풀링이다
(`projects/mmdet3d_plugin/models/basic_modules.py:637` 에서 조기 return).

그래서 이 비교는 **예측을 통제하고 판정 규칙만 바꾼다**:

    같은 관측 → 같은 미래 예측 → ┬─ DeepAccident 규칙 (임계값 하나)
                                 └─ 우리 파이프라인 (리포트 → LLM)

이것이 "LLM 이 임계값 하나보다 무엇을 더하는가" 에 답하는 통제 비교다. 학습된
인지·운동 모델까지 포함한 비교는 별개의 작업이며, 이 모듈은 그 자리에 다른 예측을
끼워 넣을 수 있게 열려 있다.

**규칙 자체는 원본 코드 그대로다.**

| 항목 | 값 | 출처 |
|---|---|---|
| 예측 쪽 사고 거리 임계 | 5 px × 0.5 m/px = **2.5 m** | `multi_gpu_test.py:527`, `:544` |
| 거리 정의 | 두 인스턴스 **폴리곤 사이 간격** (닿으면 0) | `poly_distance()` — `shapely` `Polygon.distance` |
| 비교 연산 | `preddist < threshold` (강부등호) | `multi_gpu_test.py:583` |
| 기본 미래 지평 | 2.0 초 (2Hz × 4 프레임) | `configs/DeepAccident_tiny.py` |
| 채점 프레임 | **마지막 프레임 하나만** | `multi_gpu_test.py:553` — 아래 참조 |

논문은 "1.0 m" 라고 적었지만 코드는 2.5 m 다. 코드를 따랐다 — 재현 대상은 발표된
숫자를 만든 코드이지 본문 문장이 아니다.

**마지막 프레임만 보는 것은 원본의 버그다.** `for t in range(sequence_length):` 안에서
`gt_accident` 가 매 반복 초기화되고, 그것을 쓰는 코드는 루프 **밖**(함수 본문 들여쓰기)
에 있어 새어 나온 마지막 `t` 를 읽는다. 그래서 발표된 APA 는 사실상 마지막 미래
프레임만 채점한 값이다. `frame_rule` 로 두 가지를 다 낼 수 있게 했다.

    "any"   미래 지평 안 아무 시점에서나 임계 미만이면 사고 (논문 서술)
    "last"  지평의 **마지막 구간**만 본다 (공개 코드의 실제 동작)

`"last"` 는 관대한 근사다: 우리 레코드가 갖고 있는 것은 구간별 **최솟값**이지 특정
프레임의 값이 아니므로, 원본의 "마지막 프레임 한 장" 보다 넓게 본다. 이 방향의 오차는
DeepAccident 에 유리하므로, 비교에서 우리 쪽이 유리해지는 착시를 만들지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# 원본 상수. 픽셀 임계를 미터로 환산해 둔다 (BEV 해상도 0.5 m/px).
DA_BEV_RESOLUTION_M_PER_PX = 0.5
DA_DIST_THRESHOLD_PRED_PX = 5.0
DA_DIST_THRESHOLD_PRED_M = DA_DIST_THRESHOLD_PRED_PX * DA_BEV_RESOLUTION_M_PER_PX  # 2.5
# 논문 본문이 말하는 값. 코드와 다르므로 비교용으로만 노출한다.
DA_DIST_THRESHOLD_PAPER_M = 1.0
DA_HORIZON_S = 2.0  # 2 Hz × 4 미래 프레임

FRAME_RULES = ("any", "last")


@dataclass
class DeepAccidentRule:
    """DeepAccident 후처리 규칙의 설정.

    기본값은 **공개 코드의 값**이다. 논문 본문 값으로 바꾸려면
    `dist_threshold_m=DA_DIST_THRESHOLD_PAPER_M`.
    """

    dist_threshold_m: float = DA_DIST_THRESHOLD_PRED_M
    horizon_s: float = DA_HORIZON_S
    frame_rule: str = "any"

    def __post_init__(self) -> None:
        if self.frame_rule not in FRAME_RULES:
            raise ValueError(
                f"frame_rule 은 {list(FRAME_RULES)} 중 하나여야 한다: {self.frame_rule!r}"
            )
        if self.horizon_s <= 0:
            raise ValueError(f"horizon_s 는 양수여야 한다: {self.horizon_s}")

    @property
    def n_horizon_buckets(self) -> int:
        """이 규칙이 들여다보는 구간 수. 구간은 1초 폭이다."""
        return max(1, int(math.ceil(self.horizon_s - 1e-9)))

    def label(self) -> str:
        return (
            f"da[{self.dist_threshold_m:g}m/{self.horizon_s:g}s/{self.frame_rule}]"
        )


def looked_at(cfg: DeepAccidentRule, n_buckets: int) -> List[int]:
    """이 규칙이 실제로 판정에 쓰는 구간 번호들.

    지평 밖은 **보지 않는다** — 그것이 DeepAccident 와 우리의 구조적 차이다.
    우리는 5초를 묻고 그쪽은 2초만 본다.
    """
    ks = list(range(1, min(cfg.n_horizon_buckets, n_buckets) + 1))
    if cfg.frame_rule == "last" and ks:
        return [ks[-1]]
    return ks


def judge(
    gaps: Sequence[Tuple[float, Optional[Tuple[str, str]]]],
    cfg: Optional[DeepAccidentRule] = None,
) -> dict:
    """구간별 최소 간격 → 사고 판정 1건.

    `gaps[k-1]` 은 구간 k 의 `(최소 간격 [m], 그 쌍)` 이다 —
    `examples/baselines_accident_qa.py:bucket_min_gaps` 가 만드는 것과 같은 형식이고,
    그 값은 **회전 사각형 사이의 간격**이라 DeepAccident 의 `poly_distance` 와 같은
    물리량이다 (닿으면 0).

    돌려주는 것
        accident   사고 있음/없음 — 이것이 DeepAccident 가 내는 전부다
        k          사고로 지목한 구간 (없으면 None)
        gap_m      그 구간의 최소 간격
        pair       그 구간의 최근접 쌍
        looked_at  판정에 쓴 구간들
    """
    cfg = cfg or DeepAccidentRule()
    ks = looked_at(cfg, len(gaps))
    best_k: Optional[int] = None
    best_gap = float("inf")
    best_pair: Optional[Tuple[str, str]] = None
    for k in ks:
        gap, pair = gaps[k - 1]
        gap = float("inf") if gap is None else float(gap)
        if gap < best_gap:
            best_k, best_gap, best_pair = k, gap, pair
    # 원본은 강부등호다 (`preddist < dist_threshold_pred`).
    accident = best_k is not None and best_gap < cfg.dist_threshold_m
    return {
        "accident": bool(accident),
        "k": best_k if accident else None,
        "gap_m": None if best_gap == float("inf") else round(best_gap, 3),
        "pair": list(best_pair) if (accident and best_pair) else [],
        "looked_at": ks,
        "rule": cfg.label(),
    }


def build_response(
    t_end: float,
    n_buckets: int,
    gaps: Sequence[Tuple[float, Optional[Tuple[str, str]]]],
    cfg: Optional[DeepAccidentRule] = None,
) -> dict:
    """판정을 `PREDICTION_SCHEMA` 형태의 응답으로.

    이 형태여야 LLM 응답과 **같은 채점기**(`score_modes`)를 통과해 같은 표에 오른다.

    사고로 판정하면 **지목한 구간 하나만** True 로 둔다. DeepAccident 는 최소 거리
    시점 하나를 사고 시점으로 내놓지 구간별 답을 내지 않으므로, 그것을 그대로 옮긴
    것이다. 지평 밖 구간은 전부 False 다 — 보지 않았으니 '사고 없음' 이라고 답한
    셈이 되고, 그것이 이 규칙의 실제 출력이다.
    """
    cfg = cfg or DeepAccidentRule()
    v = judge(gaps, cfg)
    hit_k = v["k"]
    preds: List[dict] = []
    for k in range(1, n_buckets + 1):
        hit = k == hit_k
        if k in v["looked_at"]:
            why = (
                f"예측 최소 간격 {v['gap_m']}m < 임계 {cfg.dist_threshold_m:g}m"
                if hit
                else f"예측 최소 간격이 임계 {cfg.dist_threshold_m:g}m 이상"
            )
        else:
            why = f"지평 {cfg.horizon_s:g}초 밖 — 이 규칙은 보지 않는다"
        preds.append(
            {
                "k": k,
                "interval_s": f"({t_end + k - 1:.1f}, {t_end + k:.1f}]",
                "accident_expected": hit,
                "involved_actor_ids": list(v["pair"]) if hit else [],
                "reason": why,
                "confidence": "medium",
            }
        )
    return {
        "predictions": preds,
        "overall_assessment": (
            f"DeepAccident 후처리 규칙 ({v['rule']}): "
            + ("사고 예측" if v["accident"] else "사고 없음")
        ),
        "data_limitations": (
            "학습된 인지·운동 모델(V2XFormer)이 아니라 사고 판정 규칙만 이식한 것이다. "
            "공개 저장소에 DeepAccident 학습 가중치가 없다."
        ),
        "deepaccident": v,
    }


def variants(fitted_threshold_m: Optional[float] = None) -> Dict[str, DeepAccidentRule]:
    """비교표에 함께 올릴 규칙 묶음.

    각 줄이 다른 질문에 답한다. 한 번에 하나씩만 바꾼다.

        da        공개 코드 그대로 (2.5m / 2초 / 마지막 구간)
        da_any    같은 지평, 지평 안 아무 시점이나 — 논문 서술대로면 얼마나 오르나
        da_h5     같은 규칙, **우리와 같은 5초 지평** — 차이가 지평 때문인가
                  규칙 때문인가를 가른다
        da_fit    5초 지평 + **train 에서 적합한 임계값** — 상수 2.5m 는 저자들의
                  BEV 픽셀 공간에 맞춘 값이라 우리 장면에 그대로 쓰면 규칙을
                  불리하게 만든다. 이 줄이 있어야 "규칙이 약한 것" 과 "상수가
                  안 맞는 것" 을 가를 수 있다 (`fitted_threshold_m` 을 주면 추가)
    """
    out = {
        "da": DeepAccidentRule(frame_rule="last"),
        "da_any": DeepAccidentRule(frame_rule="any"),
        "da_h5": DeepAccidentRule(horizon_s=5.0, frame_rule="any"),
    }
    if fitted_threshold_m is not None:
        out["da_fit"] = DeepAccidentRule(
            dist_threshold_m=float(fitted_threshold_m),
            horizon_s=5.0,
            frame_rule="any",
        )
    return out


#: `variants()` 가 만들 수 있는 모든 이름 (적합 변형 포함)
VARIANT_NAMES: Tuple[str, ...] = ("da", "da_any", "da_h5", "da_fit")


# ---------------------------------------------------------------- 포화 진단
#
# 이 규칙을 우리 데이터에 그대로 쓰면 거의 항상 발화한다. 원인은 임계값이 아니라
# **양(quantity) 자체**다: 모든 쌍을 몇 초간 외삽해 최소 간격을 재면, 뒤따라가는 차나
# 교차로에서 스치는 차의 사각형이 늘 겹친다. val 816창 실측 (세 외삽 방식 모두):
#
#     5초 지평에서 최소 간격이 0 인 창   93 ~ 98 %
#     5초 지평에서 최소 간격 < 2.5m      99.8 ~ 100 %
#     2초 지평에서 최소 간격 < 2.5m      98.5 ~ 99.6 %
#
# 그래서 임계값을 0.25m 로 낮춰도 판정이 거의 바뀌지 않는다 (train 적합에서
# 0.25m~5.0m 구간 내내 TP 155→156, FP 514→534). **임계값이 문제가 아니라는 뜻이고,
# 동시에 이 비교의 숫자를 "DeepAccident 성능" 이라고 부르면 안 된다는 뜻이다.**
#
# 이 진단을 결과와 함께 반드시 출력한다 — 발화율이 0.8 을 넘으면 그 줄의 정확도는
# 규칙의 변별력이 아니라 기저율을 재고 있다.


def saturation(
    gaps_list: Sequence[Sequence[Tuple[float, Optional[Tuple[str, str]]]]],
    cfg: Optional[DeepAccidentRule] = None,
) -> dict:
    """이 규칙이 이 데이터에서 변별력이 있는가.

    `fire_rate` 가 1 에 가까우면 규칙이 사실상 상수 함수다 — 그 줄의 정확도는
    기저율일 뿐이므로 비교표에서 그렇게 읽어야 한다.
    """
    cfg = cfg or DeepAccidentRule()
    n = fires = 0
    zero = 0
    for gaps in gaps_list:
        n += 1
        v = judge(gaps, cfg)
        fires += 1 if v["accident"] else 0
        zero += 1 if (v["gap_m"] is not None and v["gap_m"] <= 0.0) else 0
    return {
        "rule": cfg.label(),
        "n_windows": n,
        "fire_rate": round(fires / n, 3) if n else None,
        # 외삽한 차체가 실제로 겹친 창의 비율. 이것이 높으면 임계값을 어떻게 잡아도
        # 판정이 바뀌지 않는다.
        "touching_rate": round(zero / n, 3) if n else None,
        "degenerate": bool(n and fires / n > 0.8),
    }
