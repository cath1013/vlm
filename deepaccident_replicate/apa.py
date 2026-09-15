"""APA — DeepAccident 의 사고 예측 지표를 그대로 이식한 것.

**왜 필요한가.** 우리 채점기(`traffic_llm.accident_qa.score_modes`)로 재면 이식한
규칙도 학습한 헤드도 우연을 못 넘는다. 그런데 논문은 APA 69.5 를 보고한다. 두
숫자가 같은 것을 재고 있는지 확인하려면 **저자들의 지표를 그대로** 구현해서 우리
재구현에 씌워봐야 한다.

    APA 가 높게 나오면  → 지표가 "앞으로 사고가 날까" 를 재는 것이 아니다
    APA 도 낮으면       → 우리 재구현이나 입력 표현이 부족하다

두 답 모두 논문에 쓸 수 있다.

**우리 채점기와 무엇이 다른가.** 다섯 가지이고, 전부 원본 코드에서 읽었다.

| | 저자들 (APA) | 우리 (`score_modes`) |
|---|---|---|
| 정답 | GT 운동에 **같은 후처리**를 돌려 나온 "사고" | 라벨의 실제 충돌 시각 |
| 채점 프레임 | **마지막 미래 프레임 하나** | 지평 전체 |
| 무충돌 장면 | **점수에 안 들어감** (TN 을 세지 않는다) | TN 으로 센다 |
| 표본 6개 | TP 는 아무 표본이나, FP 는 **평균 표본만** | 대칭 |
| 빗나간 예측 | GT 사고가 있어도 **FP** (FN 아님) | FN |

정답이 예측과 같은 후처리를 통과한다는 점이 결정적이다. 규칙이 포화돼 있어도 양쪽이
같이 포화되므로 APA 는 높게 나올 수 있다.

**출처 (모두 `projects/mmdet3d_plugin/tools/`).**

| 항목 | 값 | 위치 |
|---|---|---|
| GT 접촉 임계 | 0.001 px × 0.5 m/px = **0.0005 m** | `multi_gpu_test.py:526` |
| 예측 접촉 임계 | 5 px × 0.5 m/px = **2.5 m** | `:527` |
| 위치 임계 D | **{5, 10, 15} m** — 두 차량 오차의 **합** | `:538` |
| APA 식 | `TP / (TP + 0.5·FP + 0.5·FN)`, D 평균 | `single_gpu_test.py:772` |
| 프레임당 초 | 0.5 (2 Hz) | `multi_gpu_test.py:534` |
| 시간 오차 상한 | 3 × 0.5 = 1.5 초 | `:551` |

**재현한 결함.** 원본의 다음 세 가지는 결과를 만든 코드의 일부이므로 기본값으로
재현한다. `faithful=False` 로 끄면 고쳐진 판본이 나온다.

    last_frame_only          `for t in range(sequence_length)` 안에서 `gt_accident`
                             가 매 반복 초기화되고, 그것을 쓰는 코드는 루프 **밖**에
                             있어 새어 나온 마지막 `t` 만 읽는다 (`:553`). 그래서
                             발표된 APA 는 사실상 마지막 미래 프레임만 채점한 값이고,
                             시간 오차는 항상 0 이다.
    fp_from_mean_only        GT 에 사고가 없을 때 FP 는 `pred_accident_total[-1]`
                             (평균 표본)만 본다 (`:646`). 반면 TP 는 모든 표본을
                             뒤진다 (`:618`). 확률 표본으로 낸 오경보는 공짜다.
    miss_is_fp_not_fn        GT 사고가 있는데 예측이 임계 밖이면 FP 만 올린다.
                             FN 을 올리는 줄은 주석 처리돼 있다 (`:641`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .da_motion import _rect_corners, poly_gap

# 원본 상수
BEV_RESOLUTION_M_PER_PX = 0.5
DIST_THRESHOLD_GT_PX = 0.001
DIST_THRESHOLD_PRED_PX = 5.0
DIST_THRESHOLD_GT_M = DIST_THRESHOLD_GT_PX * BEV_RESOLUTION_M_PER_PX  # 0.0005
DIST_THRESHOLD_PRED_M = DIST_THRESHOLD_PRED_PX * BEV_RESOLUTION_M_PER_PX  # 2.5
TP_POSITION_THRESHOLDS_M: Tuple[float, ...] = (5.0, 10.0, 15.0)
SECONDS_PER_SAMPLE = 0.5
MAX_TIME_ERROR_S = 3 * SECONDS_PER_SAMPLE  # 1.5


@dataclass
class ApaConfig:
    """APA 설정. 기본값은 **공개 코드 그대로**다."""

    dist_threshold_gt_m: float = DIST_THRESHOLD_GT_M
    dist_threshold_pred_m: float = DIST_THRESHOLD_PRED_M
    tp_position_thresholds_m: Tuple[float, ...] = TP_POSITION_THRESHOLDS_M
    seconds_per_sample: float = SECONDS_PER_SAMPLE
    max_time_error_s: float = MAX_TIME_ERROR_S
    # 원본의 결함들. 전부 True 가 "발표된 숫자를 만든 코드" 다.
    last_frame_only: bool = True
    fp_from_mean_only: bool = True
    miss_is_fp_not_fn: bool = True

    @classmethod
    def debugged(cls) -> "ApaConfig":
        """알려진 결함 셋을 고친 판본. 같은 예측에 이것도 함께 내야 정직하다."""
        return cls(last_frame_only=False, fp_from_mean_only=False,
                   miss_is_fp_not_fn=False)

    def label(self) -> str:
        flags = "".join(c for c, on in
                        (("L", self.last_frame_only), ("M", self.fp_from_mean_only),
                         ("P", self.miss_is_fp_not_fn)) if on)
        return f"apa[{self.dist_threshold_pred_m:g}m/{flags or 'fixed'}]"


# ---------------------------------------------------------------- 사고 후보


def find_accident(
    pos: np.ndarray,
    yaw: np.ndarray,
    size: np.ndarray,
    ids: Sequence[str],
    threshold_m: float,
) -> Optional[dict]:
    """한 프레임에서 **가장 가까운 쌍**이 임계 미만이면 그것이 사고다.

    원본과 같다: 모든 쌍의 폴리곤 거리를 재고, 임계 미만인 것들을 거리로 정렬해
    **가장 가까운 하나**를 고른다 (`multi_gpu_test.py:578-591`).

    돌려주는 것: `{ids, pos, gap_m}` — `pos` 는 두 차량의 (x, y) 다.
    """
    n = len(ids)
    if n < 2:
        return None
    corners = [
        _rect_corners(float(pos[i, 0]), float(pos[i, 1]), float(yaw[i]),
                      float(size[i, 0]), float(size[i, 1]))
        for i in range(n)
    ]
    best, pair = float("inf"), None
    for i in range(n):
        for j in range(i + 1, n):
            g = poly_gap(corners[i], corners[j])
            if g < best:
                best, pair = g, (i, j)
    if pair is None or not (best < threshold_m):
        return None
    i, j = pair
    return {
        "ids": (ids[i], ids[j]),
        "pos": (np.asarray(pos[i], dtype=float), np.asarray(pos[j], dtype=float)),
        "gap_m": float(best),
    }


def _pair_position_error(pred: dict, gt: dict) -> float:
    """두 차량 위치 오차의 **합** [m]. 쌍의 대응은 더 작은 쪽으로 잡는다.

    원본 `multi_gpu_test.py:620-630` 과 같다 — 평균이 아니라 합이고, 그래서
    임계 {5, 10, 15} m 는 차량당 {2.5, 5, 7.5} m 에 해당한다.
    """
    pa, pb = pred["pos"]
    ga, gb = gt["pos"]
    e1 = float(np.linalg.norm(pa - ga) + np.linalg.norm(pb - gb))
    e2 = float(np.linalg.norm(pa - gb) + np.linalg.norm(pb - ga))
    return min(e1, e2)


# ---------------------------------------------------------------- 창 하나 채점


@dataclass
class ApaCounts:
    """위치 임계값별 누적. 길이는 `tp_position_thresholds_m` 와 같다."""

    tp: List[int] = field(default_factory=list)
    fp: List[int] = field(default_factory=list)
    fn: List[int] = field(default_factory=list)
    id_err: List[int] = field(default_factory=list)
    pos_err: List[float] = field(default_factory=list)
    time_err: List[float] = field(default_factory=list)

    @classmethod
    def zeros(cls, n: int) -> "ApaCounts":
        return cls([0] * n, [0] * n, [0] * n, [0] * n, [0.0] * n, [0.0] * n)

    def add(self, other: "ApaCounts") -> None:
        for a, b in ((self.tp, other.tp), (self.fp, other.fp), (self.fn, other.fn),
                     (self.id_err, other.id_err), (self.pos_err, other.pos_err),
                     (self.time_err, other.time_err)):
            for i in range(len(a)):
                a[i] += b[i]


def score_window(
    gt_frames: Sequence[Optional[dict]],
    pred_frames: Sequence[Sequence[Optional[dict]]],
    cfg: Optional[ApaConfig] = None,
) -> ApaCounts:
    """창 하나 → 임계값별 TP/FP/FN.

    `gt_frames[t]`      그 미래 프레임의 GT 사고 (`find_accident` 결과 또는 None)
    `pred_frames[s][t]` 표본 s 의 그 프레임 예측 사고

    **마지막 표본이 분포 평균**이라는 원본 규약을 따른다 (`pred_frames[-1]`).
    """
    cfg = cfg or ApaConfig()
    n_thr = len(cfg.tp_position_thresholds_m)
    out = ApaCounts.zeros(n_thr)
    n_future = len(gt_frames)
    if n_future == 0:
        return out

    # 원본은 마지막 프레임 하나만 본다. 고친 판본은 GT 사고가 있는 가장 이른
    # 프레임을 쓴다 (없으면 마지막).
    if cfg.last_frame_only:
        t_gt = n_future - 1
    else:
        hits = [t for t, g in enumerate(gt_frames) if g is not None]
        t_gt = hits[0] if hits else n_future - 1
    gt = gt_frames[t_gt]

    # 표본별 예측. 원본은 같은 프레임에서 고르고, 고친 판본은 표본마다
    # **가장 이른** 예측 사고를 쓴다 (그래야 시간 오차가 0 이 아니게 된다).
    picks: List[Optional[Tuple[int, dict]]] = []
    for s in range(len(pred_frames)):
        if cfg.last_frame_only:
            p = pred_frames[s][t_gt] if t_gt < len(pred_frames[s]) else None
            picks.append((t_gt, p) if p is not None else None)
        else:
            got = next(((t, p) for t, p in enumerate(pred_frames[s]) if p is not None),
                       None)
            picks.append(got)
    valid = [s for s, p in enumerate(picks) if p is not None]

    for k, thr in enumerate(cfg.tp_position_thresholds_m):
        if gt is not None:
            if not valid:
                out.fn[k] += 1
                continue
            best_err, best_s = float("inf"), -1
            for s in valid:
                e = _pair_position_error(picks[s][1], gt)
                if e < best_err:
                    best_err, best_s = e, s
            if best_err <= thr:
                out.tp[k] += 1
                out.pos_err[k] += min(best_err, thr)
                dt = abs(picks[best_s][0] - t_gt) * cfg.seconds_per_sample
                out.time_err[k] += min(dt, cfg.max_time_error_s)
                if set(picks[best_s][1]["ids"]) != set(gt["ids"]):
                    out.id_err[k] += 1
            else:
                # 원본은 빗나간 예측을 FP 로만 센다 (FN 줄이 주석 처리돼 있다)
                out.fp[k] += 1
                if not cfg.miss_is_fp_not_fn:
                    out.fn[k] += 1
        else:
            # GT 에 사고가 없다. 원본은 **평균 표본만** 보고 FP 를 올린다.
            fired = (picks[-1] is not None) if cfg.fp_from_mean_only else bool(valid)
            if fired:
                out.fp[k] += 1
    return out


def aggregate(total: ApaCounts, cfg: Optional[ApaConfig] = None) -> dict:
    """누적 → APA 와 TP 지표.

    `APA = mean_d [ TP_d / (TP_d + 0.5·FP_d + 0.5·FN_d) ]`  (`single_gpu_test.py:772`)
    """
    cfg = cfg or ApaConfig()
    per: List[dict] = []
    accs: List[float] = []
    for k, thr in enumerate(cfg.tp_position_thresholds_m):
        tp, fp, fn = total.tp[k], total.fp[k], total.fn[k]
        den = tp + 0.5 * fp + 0.5 * fn
        acc = (tp / den) if den else 0.0
        accs.append(acc)
        per.append({
            "threshold_m": thr,
            "TP": tp, "FP": fp, "FN": fn,
            "APA": round(acc, 4),
            "id_error": round(total.id_err[k] / tp, 4) if tp else None,
            "position_error_m": round(total.pos_err[k] / tp, 4) if tp else None,
            "time_error_s": round(total.time_err[k] / tp, 4) if tp else None,
        })
    return {
        "rule": cfg.label(),
        "APA": round(sum(accs) / len(accs), 4) if accs else None,
        "per_threshold": per,
        # TN 이 어디에도 안 들어간다는 사실을 결과에 남긴다 — APA 의 가장 약한 지점이다
        "note": "APA 는 true negative 를 세지 않는다 (무충돌 장면은 점수에 기여하지 않는다)",
    }
