"""`make_predict_dataset.py` 로 만든 데이터로 경로 예측 모델을 학습한다.

두 가지 과제
    `--task rank` (기본) — 후보 경로들 중 **실제로 간 것**을 고른다. 기하는 지도에서
    나온 것을 그대로 쓰므로 학습하는 것은 확률뿐이고 노면을 벗어나지 않는다. 척도는
    top-1 정확도(클수록 좋다). 단, **후보 집합에 정답이 있어야만** 맞힐 수 있다.

    `--task waypoints` — 좌표를 직접 회귀한다. 후보를 고르지 않으므로 후보 집합의
    상한(실측 78.8%)에 걸리지 않고, `matched=false` 표본도 쓸 수 있어 학습 데이터가
    6.7배가 된다. 척도는 ADE/FDE(작을수록 좋다). 대가는 **다중 모드를 잃는 것**이다 —
    "직진 60% / 좌회전 40%" 같은 분포 대신 궤적 하나를 확률 1.0 으로 낸다.

    저장한 모델은 `TorchPredictor` 로 파이프라인에 그대로 꽂힌다.

    python examples/train_predict_model.py --data out/predict_dataset \\
        --epochs 100 --out out/predict_model
    python examples/train_predict_model.py --data out/predict_dataset \\
        --task waypoints --epochs 100 --out out/predict_model

무엇과 비교하는가
    같은 표본에 대해 **현재 `prediction.py` 의 규칙 기반**이 내는 성능을 함께 낸다.
    규칙 기반의 확률은 `Candidate.prior` 이고 그것이 후보 특징 벡터의 0번 칸에
    저장돼 있으므로(§4.2), 그 argmax 가 곧 `rule_predict` 의 1순위다. 두 방법이
    **같은 후보 집합, 같은 표본**을 보므로 비교가 성립한다.

    waypoints 과제에서는 규칙 기반이 실제로 내보내는 웨이포인트
    (`candidate_waypoints`, 데이터에 저장돼 있다)와 `target_offsets` 를 **같은 시각
    격자**에서 비교한다. 등속 직진과 "후보 중 최선"(후보 방식의 오라클)도 함께 낸다 —
    오라클을 넘는지가 후보 집합의 상한을 벗어났는지를 말해 준다.

분할
    기본은 데이터셋 자신의 분할을 쓴다 — `train`+`DeepAccident_mini` 로 학습하고
    `val` 로 시험한다. **표본 단위 무작위 분할을 쓰지 않는다**: 0.5초 간격 인접
    표본은 서로 거의 같아서 같은 시나리오가 양쪽에 들어가면 성능이 크게
    과대평가된다. `--split-by scenario` 는 시나리오 id 해시로 나눈다.

    학습셋에서 다시 일부를 떼어 **검증셋**으로 쓰고(기본 10%, 시나리오 단위),
    거기서 가장 좋은 epoch 의 가중치를 `ranknet_best.pt` 로 저장한다. 시험셋으로
    epoch 을 고르면 시험 성능이 낙관적으로 편향되므로 그렇게 하지 않는다.
    `--val-frac 0` 이면 검증셋 없이 마지막 epoch 만 저장한다.

보고하는 수치가 무엇에 조건부인가 — 반드시 알 것
    `matched=false` 를 제외하므로, 출력되는 top-1 은 **"정답이 후보에 있는 경우"로
    한정한 값**이다. 후보는 `roadmap.downstream_paths` 가 지도에서 만든 것이고 실제
    궤적이 그중 하나라는 보장이 없다 — 실측 21.4% 는 어느 후보와도 3m 안에서 맞지
    않는다(차선 변경·노면 이탈). 예측 구간이 길수록 늘어난다 (2초 4.5% → 6초 35.5%).
    후보 밖을 오답으로 세면 상한이 78.8% 로 막힌다. 규칙 기반도 같은 한계를 지므로
    비교 자체는 성립한다. docs/predict_model_io.md §4.1 · §5.6.

제외하는 표본
    `matched=false` (실제 궤적이 어느 후보와도 3m 안에서 맞지 않음) 와
    `label_ambiguous=true` (여러 후보가 실제 궤적과 똑같이 가까워 정답이 목록
    순서로 정해진 것) 는 뺀다. 후자를 넣으면 모델이 후보 순서를 외운다.
    `--include-ambiguous` 는 `tied_best` 를 균등 소프트 타깃으로 넣어 살려 쓴다
    (평가에서는 여전히 제외한다 — 채점 기준이 임의이므로).
"""

from __future__ import annotations

import argparse
import io
import json

import os
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

import torch  # noqa: E402
from torch import nn  # noqa: E402

from traffic_llm.predict_model import (  # noqa: E402
    MANEUVER_CLASSES,
    MANEUVERS,
    N_CANDIDATE_FEATURES,
    N_GLOBAL_FEATURES,
    N_HISTORY_FEATURES,
    N_HISTORY_STEPS,
    N_INTERACTION_FEATURES,
    OFF_CANDIDATE_LABEL,
)
from traffic_llm.predict_nets import (  # noqa: E402
    RankNet,
    WaypointNet,
    InteractionWaypointNet,
    masked_scores,
)

# 후보 기동 원-핫은 특징 벡터의 뒤 4칸이다 (candidate_features 참조).
MAN_ONEHOT_OFFSET = N_CANDIDATE_FEATURES - len(MANEUVERS)


# ------------------------------------------------------------------ 데이터

def read_samples(path: str, include_ambiguous: bool, task: str = "rank"):
    """JSONL → 학습에 쓸 표본 목록.

    `task="rank"` 는 후보 중 정답 인덱스를 고르는 문제다. 정답이 후보에 없는
    표본(`matched=false`)과 정답이 목록 순서로 정해진 표본(`label_ambiguous`)은
    쓸 수 없으므로 뺀다.

    `task="waypoints"` 는 좌표를 직접 내는 문제다. **아무것도 빼지 않는다** —
    `matched=false` 도 실제 궤적이라는 정답이 있고, 후보 인덱스의 모호성은 회귀
    정답과 무관하다. 후보 집합의 한계를 넘어서는 것이 이 방식의 존재 이유다.
    """
    rows = []
    n_all = n_unmatched = n_single = n_amb = n_short = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n_all += 1
            amb = bool(r.get("label_ambiguous"))
            if not r["matched"]:
                n_unmatched += 1
            if len(r["candidates"]) < 2:
                n_single += 1
            if amb:
                n_amb += 1
            if task == "rank":
                if not r["matched"] or len(r["candidates"]) < 2:
                    continue
                if amb and not include_ambiguous:
                    continue
            else:
                if "candidate_waypoints" not in r:
                    raise SystemExit(
                        "데이터에 candidate_waypoints 가 없습니다 — "
                        "make_predict_dataset.py 로 다시 생성하십시오."
                    )
                if len(r["target_offsets"]) < 2:
                    n_short += 1
                    continue
            rows.append({
                "g": r["global"],
                "c": r["candidates"],
                "y": r["best_candidate"],
                "tied": r.get("tied_best") or [r["best_candidate"]],
                "amb": amb,
                "matched": bool(r["matched"]),
                "man": r["maneuver"],
                "man_amb": bool(r.get("maneuver_ambiguous")),
                "hist": r.get("history"),
                "interaction": r.get("interactions") or [],
                "tgt": r["target_offsets"],
                "cw": r.get("candidate_waypoints") or [],
                "split": r.get("dataset_split") or "?",
                "scenario": r.get("scenario_id") or "",
                # Pair separation 학습은 같은 장면·시각의 서로 다른 actor를
                # 묶는다. target_offsets 는 이 origin 에 대한 ENU 상대좌표다.
                "scenario_id": r.get("scenario_id") or "",
                "t_s": r.get("t_s"),
                "actor_id": r.get("actor_id") or "",
                "actor_class": r.get("actor_class") or "",
                "origin_enu": r.get("origin_enu") or [0.0, 0.0],
                "town": r.get("town") or "?",
            })
    stats = {
        "n_all": n_all, "n_unmatched": n_unmatched, "n_single": n_single,
        "n_ambiguous": n_amb, "n_short_future": n_short, "n_kept": len(rows),
    }
    return rows, stats


def scenario_bucket(scenario_id: str, n: int = 100) -> int:
    """시나리오 id → 0..n-1. `hash()` 는 실행마다 달라지므로 crc32 를 쓴다."""
    return zlib.crc32(scenario_id.encode("utf-8")) % n


def split_rows(rows, mode: str, val_frac: float, test_frac: float):
    """(학습, 검증, 시험). 검증은 학습셋에서 시나리오 단위로 떼어낸다."""
    if mode == "dataset":
        tr = [r for r in rows if r["split"] != "val"]
        te = [r for r in rows if r["split"] == "val"]
        if not te:
            raise SystemExit(
                "`val` 분할 표본이 없습니다. --split-by scenario 를 쓰십시오."
            )
    else:
        cut = int(round(test_frac * 100))
        te = [r for r in rows if scenario_bucket(r["scenario"]) < cut]
        tr = [r for r in rows if scenario_bucket(r["scenario"]) >= cut]

    va: list = []
    if val_frac > 0:
        # 학습셋 안에서 다시 시나리오 단위로 뗀다. 오프셋을 주어 시험셋 분할과
        # 같은 경계를 재사용하지 않는다.
        lo = int(round(test_frac * 100)) if mode != "dataset" else 0
        hi = lo + int(round(val_frac * 100))
        va = [r for r in tr if lo <= scenario_bucket(r["scenario"]) < hi]
        tr = [r for r in tr if not (lo <= scenario_bucket(r["scenario"]) < hi)]
    return tr, va, te


def to_waypoint_tensors(rows, device, n_max, n_steps):
    """좌표 회귀용 텐서. t = 1..n_steps 만 담는다 (t=0 은 항상 원점).

    `tmask` 는 실제 궤적이 관측된 시각, `cwmask` 는 후보가 그 시각까지 닿는지다.
    비교는 **둘 다 값이 있는 시각**에서만 한다 — 후보가 도로 끝에서 잘린 구간을
    오차로 세면 규칙 기반을 부당하게 깎고, 무시하면 커버리지 차이가 감춰진다.
    그래서 오차와 커버리지를 따로 낸다.
    """
    b, K = len(rows), n_steps
    # 파이썬 리스트로 다 채운 뒤 **한 번에** 텐서로 만든다. 원소마다
    # `torch.tensor()` 를 부르면 151k×9×5 회가 되어 몇 분씩 걸린다.
    ZP = [0.0, 0.0]
    tgt_l, tm_l, cw_l, cm_l = [], [], [], []
    for r in rows:
        t = r["tgt"][1:K + 1]
        tgt_l.append(t + [ZP] * (K - len(t)))
        tm_l.append([True] * len(t) + [False] * (K - len(t)))
        rows_cw, rows_cm = [], []
        for wp in r["cw"][:n_max]:
            w = wp[1:K + 1]
            rows_cw.append(w + [ZP] * (K - len(w)))
            rows_cm.append([True] * len(w) + [False] * (K - len(w)))
        pad = n_max - len(rows_cw)
        rows_cw += [[ZP] * K] * pad
        rows_cm += [[False] * K] * pad
        cw_l.append(rows_cw)
        cm_l.append(rows_cm)
    # 기동 정답. 후보 밖이면 마지막 클래스, 기동이 모호하면 -100 (손실에서 무시).
    MI = {m: i for i, m in enumerate(MANEUVER_CLASSES)}
    OFFI = MI[OFF_CANDIDATE_LABEL]
    man_l = []
    for r in rows:
        if not r["matched"]:
            man_l.append(OFFI)
        elif r["man_amb"]:
            man_l.append(-100)          # 정답이 임의 → 학습·채점에서 제외
        else:
            man_l.append(MI.get(r["man"], -100))
    tgt = torch.tensor(tgt_l, dtype=torch.float32)
    tmask = torch.tensor(tm_l, dtype=torch.bool)
    cw = torch.tensor(cw_l, dtype=torch.float32)
    cwmask = torch.tensor(cm_l, dtype=torch.bool)
    return {
        "tgt": tgt.to(device), "tmask": tmask.to(device),
        "cw": cw.to(device), "cwmask": cwmask.to(device), "n_steps": K,
        "origin": torch.tensor([r["origin_enu"] for r in rows],
                               dtype=torch.float32, device=device),
        "man": torch.tensor(man_l, dtype=torch.long, device=device),
    }


def is_normal_scenario(scenario_id: str) -> bool:
    """Dataset scenario_id의 첫 경로 요소가 ``*_normal`` 인지 판별한다."""
    return scenario_id.split("/", 1)[0].endswith("_normal")


def build_normal_pair_index(rows, n_steps: int, near_m: float):
    """같은 정상 scene-time의 GT 근접 actor pair의 row index를 만든다.

    `target_offsets`의 0번은 현재 위치라 제외한다. 각 actor마다 미래 관측 길이가
    다를 수 있으므로 둘 다 존재하는 시각만 본다. 이 index는 training split에만
    만들어 검증/시험 데이터를 auxiliary loss에 누출하지 않는다.
    """
    groups = {}
    for i, row in enumerate(rows):
        sid = row["scenario_id"]
        if (not is_normal_scenario(sid)) or not row["actor_id"]:
            continue
        groups.setdefault((sid, row["t_s"]), []).append(i)

    pairs = []
    for indices in groups.values():
        for pos, ia in enumerate(indices):
            a = rows[ia]
            for ib in indices[pos + 1:]:
                b = rows[ib]
                # 방어적으로 같은 actor가 중복된 경우에는 pair가 되지 않게 한다.
                if a["actor_id"] == b["actor_id"]:
                    continue
                K = min(n_steps, len(a["tgt"]) - 1, len(b["tgt"]) - 1)
                if K <= 0:
                    continue
                a_off = torch.tensor(a["tgt"][1:K + 1], dtype=torch.float32)
                b_off = torch.tensor(b["tgt"][1:K + 1], dtype=torch.float32)
                a_world = a_off + torch.tensor(a["origin_enu"], dtype=torch.float32)
                b_world = b_off + torch.tensor(b["origin_enu"], dtype=torch.float32)
                if torch.linalg.vector_norm(a_world - b_world, dim=-1).min().item() <= near_m:
                    pairs.append((ia, ib))
    return pairs


def build_relative_pair_index(rows, n_steps: int, near_m: float):
    """같은 scene-time의 GT 근접 actor pair를 scenario 종류와 무관하게 만든다.

    정상/사고 여부는 relative trajectory supervision에 사용하지 않는다. raw
    `target_offsets`의 길이가 곧 각 row의 유효 미래 구간이므로, 두 actor에게
    공통으로 유효한 future timestep에서만 GT world 거리를 판정한다.
    """
    groups = {}
    for i, row in enumerate(rows):
        if not row["actor_id"]:
            continue
        groups.setdefault((row["scenario_id"], row["t_s"]), []).append(i)

    pairs = []
    for indices in groups.values():
        for pos, ia in enumerate(indices):
            a = rows[ia]
            for ib in indices[pos + 1:]:
                b = rows[ib]
                if a["actor_id"] == b["actor_id"]:
                    continue
                K = min(n_steps, len(a["tgt"]) - 1, len(b["tgt"]) - 1)
                if K <= 0:
                    continue
                a_off = torch.tensor(a["tgt"][1:K + 1], dtype=torch.float32)
                b_off = torch.tensor(b["tgt"][1:K + 1], dtype=torch.float32)
                a_world = a_off + torch.tensor(a["origin_enu"], dtype=torch.float32)
                b_world = b_off + torch.tensor(b["origin_enu"], dtype=torch.float32)
                if torch.linalg.vector_norm(a_world - b_world, dim=-1).min().item() <= near_m:
                    pairs.append((ia, ib))
    return pairs


def displacement(pred, tgt, mask):
    """(ADE, FDE, 유효표본 마스크). `mask` 가 True 인 시각만 센다."""
    d = ((pred - tgt) ** 2).sum(-1).clamp(min=0).sqrt()      # [B,K]
    cnt = mask.sum(-1)
    ok = cnt > 0
    ade = torch.where(ok, (d * mask).sum(-1) / cnt.clamp(min=1),
                      torch.zeros_like(cnt, dtype=d.dtype))
    # FDE: 유효한 마지막 시각
    idx = (mask.float() * torch.arange(
        mask.shape[1], device=mask.device, dtype=torch.float32)).argmax(-1)
    fde = d.gather(1, idx.unsqueeze(1)).squeeze(1)
    return ade, fde, ok


def eval_waypoints(model, d, w, chunk: int = 8192):
    """좌표 회귀 평가 — 모델 · 규칙 top-1 · 후보 최선(오라클) · 등속.

    등속을 넣는 이유: 5초 예측의 상당 부분은 "직진 유지" 이므로, 그것보다 나은지를
    보지 않으면 모델이 무엇을 배웠는지 알 수 없다.
    """
    model.eval()
    preds, mlogits = [], []
    with torch.no_grad():
        for i in range(0, d["n"], chunk):
            o = call_model(model, d, slice(i, i + chunk))
            if isinstance(o, tuple):
                o, lg = o
                mlogits.append(lg)
            preds.append(o)
    pred = torch.cat(preds)                                   # [B,K,2]
    man_pred = torch.cat(mlogits).argmax(-1) if mlogits else None

    tgt, tmask, cw, cwmask = w["tgt"], w["tmask"], w["cw"], w["cwmask"]
    prior = d["c"][..., 0]                                     # [B,N]
    top = prior.argmax(dim=1)
    bi = torch.arange(d["n"], device=pred.device)
    rule = cw[bi, top]                                         # [B,K,2]
    rmask = cwmask[bi, top]

    # 공통 지지 구간에서만 비교한다
    common = tmask & rmask
    m_ade, m_fde, ok = displacement(pred, tgt, common)
    r_ade, r_fde, _ = displacement(rule, tgt, common)

    # 등속: 첫 1초 변위를 그대로 반복 (실제 궤적의 첫 구간을 쓰지 않는다 —
    # 그러면 정답을 엿보는 것이다. 속도·방위에서 만든다)
    v = d["g"][:, 0]                                            # 속도 [m/s]
    hs, hc = d["g"][:, 4], d["g"][:, 5]                         # 방위 sin/cos
    step = torch.stack([v * hs, v * hc], dim=-1)                # ENU 1초 변위
    ks = torch.arange(1, w["n_steps"] + 1, device=pred.device,
                      dtype=step.dtype).reshape(1, -1, 1)
    cv = step.unsqueeze(1) * ks
    c_ade, c_fde, _ = displacement(cv, tgt, common)

    # 후보 중 실제 궤적에 가장 가까운 것 = 후보 방식의 상한(오라클)
    cm = tmask.unsqueeze(1) & cwmask                            # [B,N,K]
    dall = ((cw - tgt.unsqueeze(1)) ** 2).sum(-1).clamp(min=0).sqrt()
    cnt = cm.sum(-1)
    ade_all = torch.where(cnt > 0, (dall * cm).sum(-1) / cnt.clamp(min=1),
                          torch.full_like(dall[..., 0], 1e9))
    ade_all = ade_all.masked_fill(~d["mask"], 1e9)
    o_ade = ade_all.min(dim=1).values

    def mean(x, sel):
        s = ok & sel
        return x[s].mean().item() if s.any() else float("nan")

    all_ = torch.ones_like(ok)
    matched = d["matched"]
    out = {"n": int(ok.sum().item())}
    for name, sel in (("all", all_), ("matched", matched),
                      ("unmatched", ~matched)):
        out[name] = {
            "n": int((ok & sel).sum().item()),
            "model_ade": mean(m_ade, sel), "model_fde": mean(m_fde, sel),
            "rule_ade": mean(r_ade, sel), "rule_fde": mean(r_fde, sel),
            "oracle_ade": mean(o_ade, sel),
            "cv_ade": mean(c_ade, sel), "cv_fde": mean(c_fde, sel),
        }
    # --- 기동 분류. **좌표 정확도와 별개로 반드시 재야 한다.**
    if man_pred is not None:
        gt = w["man"]
        scoreable = gt >= 0                       # 정답이 임의인 표본 제외
        MI = {m: i for i, m in enumerate(MANEUVER_CLASSES)}
        OFFI = MI[OFF_CANDIDATE_LABEL]
        # 규칙 기반의 기동 = 사전확률 1순위 후보의 기동. 규칙은 "후보 밖" 을
        # 말할 수 없으므로 실제로 후보 밖이었던 표본에서는 무조건 틀린다.
        onehot = d["c"][..., MAN_ONEHOT_OFFSET:]
        cand_man = onehot.argmax(-1)                            # [B,N]
        top = d["c"][..., 0].masked_fill(~d["mask"], -1.0).argmax(1)
        rule_man = cand_man[torch.arange(d["n"], device=gt.device), top]
        on = scoreable & (gt != OFFI)             # 후보 안 (규칙도 채점 가능)
        out["maneuver"] = {
            "n": int(scoreable.sum().item()),
            "model_acc": (man_pred[scoreable] == gt[scoreable])
                         .float().mean().item(),
            "n_on_candidate": int(on.sum().item()),
            "model_acc_on_candidate": (man_pred[on] == gt[on])
                                      .float().mean().item(),
            "rule_acc_on_candidate": (rule_man[on] == gt[on])
                                     .float().mean().item(),
        }
        # "후보 밖" 판정의 정밀도·재현율
        said = man_pred == OFFI
        truly = gt == OFFI
        tp = int((said & truly & scoreable).sum().item())
        fp = int((said & ~truly & scoreable).sum().item())
        fn = int((~said & truly & scoreable).sum().item())
        out["off_candidate"] = {
            "base_rate": float(truly[scoreable].float().mean().item()),
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
            "n_flagged": tp + fp,
        }

    # 규칙 기반이 예측 시각을 얼마나 덮는가 (모델은 항상 100%)
    out["rule_step_coverage"] = (
        (rmask & tmask).sum().item() / max(1, tmask.sum().item())
    )
    return out


def to_tensors(rows, device, n_max=None):
    """가변 후보 수를 패딩해 하나의 텐서 묶음으로. 전부 GPU 에 올린다.

    표본 수가 수만 개, 후보가 10개 이하라 전체가 수십 MB 다 — epoch 마다 CPU→GPU
    복사를 하는 것보다 한 번 올려 두는 것이 빠르다.
    """
    if n_max is None:
        n_max = max(len(r["c"]) for r in rows)
    b = len(rows)
    g = torch.zeros(b, N_GLOBAL_FEATURES)
    c = torch.zeros(b, n_max, N_CANDIDATE_FEATURES)
    mask = torch.zeros(b, n_max, dtype=torch.bool)
    y = torch.zeros(b, dtype=torch.long)
    soft = torch.zeros(b, n_max)
    amb = torch.zeros(b, dtype=torch.bool)
    for i, r in enumerate(rows):
        n = len(r["c"])
        g[i] = torch.tensor(r["g"])
        c[i, :n] = torch.tensor(r["c"])
        mask[i, :n] = True
        y[i] = r["y"]
        for j in r["tied"]:
            soft[i, j] = 1.0 / len(r["tied"])
        amb[i] = r["amb"]
    # 과거 궤적 [B, T, F]. 데이터에 없으면(구 버전 파일) 전부 0 + valid=0 이라
    # "이력 없음"으로 학습된다 — 조용히 다른 값을 넣지 않는다.
    ZH = [[0.0] * N_HISTORY_FEATURES for _ in range(N_HISTORY_STEPS)]
    hist = torch.tensor(
        [(r["hist"] or ZH) for r in rows], dtype=torch.float32
    )
    n_interaction_max = max((len(r["interaction"]) for r in rows), default=0)
    interaction = torch.zeros(b, n_interaction_max, N_INTERACTION_FEATURES)
    interaction_mask = torch.zeros(b, n_interaction_max, dtype=torch.bool)
    for i, r in enumerate(rows):
        rows_i = r["interaction"][:n_interaction_max]
        if rows_i:
            if any(len(x) != N_INTERACTION_FEATURES for x in rows_i):
                raise ValueError("interaction feature dimension mismatch; regenerate dataset")
            interaction[i, :len(rows_i)] = torch.tensor(rows_i)
            interaction_mask[i, :len(rows_i)] = True
    return {
        "g": g.to(device), "c": c.to(device), "mask": mask.to(device),
        "y": y.to(device), "soft": soft.to(device), "amb": amb.to(device),
        "hist": hist.to(device),
        "interaction": interaction.to(device),
        "interaction_mask": interaction_mask.to(device),
        "n": b, "n_max": n_max,
        "towns": [r["town"] for r in rows],
        "matched": torch.tensor([r["matched"] for r in rows],
                                dtype=torch.bool, device=device),
    }


# ------------------------------------------------------------------ 평가

def evaluate(model, d, chunk: int = 8192):
    """모델·규칙 기반·무작위의 top-1 과 기동 정확도. 라벨 임의 표본은 뺀다."""
    keep = ~d["amb"]
    if keep.sum().item() == 0:
        return None
    g, c, mask, y = d["g"][keep], d["c"][keep], d["mask"][keep], d["y"][keep]
    n = g.shape[0]

    model.eval()
    picks = []
    with torch.no_grad():
        for i in range(0, n, chunk):
            s = masked_scores(model, g[i:i + chunk], c[i:i + chunk],
                              mask[i:i + chunk])
            picks.append(s.argmax(dim=1))
    pick = torch.cat(picks)

    # 규칙 기반: 후보 특징 0번 칸이 rule_predict 의 확률이다
    prior = c[..., 0].masked_fill(~mask, -1.0)
    rule = prior.argmax(dim=1)

    man = c[..., MAN_ONEHOT_OFFSET:].argmax(dim=-1)      # [n, n_max]
    idx = torch.arange(n, device=g.device)
    man_true = man[idx, y]

    n_cand = mask.sum(dim=1).float()
    return {
        "n": n,
        "model_top1": (pick == y).float().mean().item(),
        "rule_top1": (rule == y).float().mean().item(),
        "random": (1.0 / n_cand).mean().item(),
        "model_man": (man[idx, pick] == man_true).float().mean().item(),
        "rule_man": (man[idx, rule] == man_true).float().mean().item(),
    }


def per_town(model, d, chunk: int = 8192):
    """타운별 top-1. 전체 수치가 타운 구성에 얼마나 끌려가는지 보기 위한 것이다.

    한 타운만 나빠서 평균이 낮은 것인지, 고르게 낮은 것인지 구별해야 원인을 찾을 수
    있다. 학습·시험의 타운 구성이 다르면 전체 수치만으로는 성능 차이의 원인을 알 수 없다.
    """
    keep = ~d["amb"]
    idx_keep = [i for i, k in enumerate(keep.tolist()) if k]
    if not idx_keep:
        return []
    towns = [d["towns"][i] for i in idx_keep]
    g, c, mask, y = d["g"][keep], d["c"][keep], d["mask"][keep], d["y"][keep]

    model.eval()
    picks = []
    with torch.no_grad():
        for i in range(0, g.shape[0], chunk):
            picks.append(masked_scores(
                model, g[i:i + chunk], c[i:i + chunk], mask[i:i + chunk]
            ).argmax(dim=1))
    pick = torch.cat(picks)
    rule = c[..., 0].masked_fill(~mask, -1.0).argmax(dim=1)
    hit_m = (pick == y).tolist()
    hit_r = (rule == y).tolist()

    agg: dict = {}
    for t, hm, hr in zip(towns, hit_m, hit_r):
        a = agg.setdefault(t, [0, 0, 0])
        a[0] += 1
        a[1] += hm
        a[2] += hr
    return sorted(
        ({"town": t, "n": a[0], "model": a[1] / a[0], "rule": a[2] / a[0]}
         for t, a in agg.items()),
        key=lambda r: -r["n"],
    )


def call_model(model, d, sl):
    """이력을 받는 모델이면 3-인자로 부른다.

    `accepts_history` 는 `WaypointNet` 이 설정에 따라 True/False 로 둔다
    (`history_encoder="none"` 이면 False). 서명만 보고 판단하면 그 구별이 안 된다.
    """
    h = d["hist"][sl] if getattr(model, "accepts_history", False) else None
    if getattr(model, "accepts_interactions", False):
        return model(d["g"][sl], d["c"][sl], h,
                     d["interaction"][sl], d["interaction_mask"][sl])
    if h is not None:
        return model(d["g"][sl], d["c"][sl], h)
    return model(d["g"][sl], d["c"][sl])


def _w(s: str) -> int:
    """터미널 표시 폭. 한글·한자는 2칸을 차지하므로 문자 수로 맞추면 표가 어긋난다."""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _pad(s: str, width: int, right: bool = False) -> str:
    fill = " " * max(0, width - _w(s))
    return (fill + s) if right else (s + fill)


def loss_of(model, d, sl, soft: bool):
    s = masked_scores(model, d["g"][sl], d["c"][sl], d["mask"][sl])
    if soft:
        return nn.functional.cross_entropy(s, d["soft"][sl])
    return nn.functional.cross_entropy(s, d["y"][sl])


# 좌표 손실 종류. 평가 지표(ADE)와 **같지 않다** — 아래 주석과 §손실 문서 참조.
HUBER_DELTA_M = 2.0


def coord_loss(pred, tgt, mask, kind: str = "huber"):
    """관측된 시각에서만 재는 좌표 손실. [B,K,2] → 스칼라.

    네 가지를 고를 수 있다. 기본은 `huber` 다.

    `huber`  (기본) x·y 축별 Huber 를 더한다. delta=2m 이하는 2차, 그보다 크면 1차.
             2차 구간에서는 회전 불변이지만 1차 구간에서는 아니다 — 같은 크기의 오차라도
             대각선 방향이 최대 1.41배 세게 벌점을 받는다. 즉 **"ADE 를 그대로
             최소화한다"는 말은 사실이 아니다.**
    `huber_norm`  유클리드 거리에 Huber. **회전 불변**이고 ADE 에 더 가깝다.
    `ade`    평균 유클리드 거리 = **평가 지표 그 자체**를 손실로 쓴다.
    `l2`     평균 제곱오차.

    실측 (전체 데이터, 100 epoch, 시드 0~2 의 시험 ADE):

        huber  3.05 / 3.03 / 3.04 m      ← 기본
        ade    3.03 / 3.03 / 3.03 m      ← 사실상 같다
        l2     3.11 / 3.13 / 3.11 m      ← 확실히 나쁘다

    읽는 법: **huber 와 ade 는 구별되지 않는다**(차이 0.01m, 시드폭 0.02m 안). 지표를
    직접 최소화하는 것이 이론적으로 불리하다고 볼 근거가 이 규모에서는 없다 — 처음에
    "‖e‖ 의 기울기 크기가 오차와 무관해 미세 수렴이 나쁘다"고 적었으나 측정이
    뒷받침하지 않아 지웠다. 실제로 의미 있는 것은 **L2 를 쓰지 않는 것**이다
    (+0.08m, 시드폭 밖). 큰 오차(차선 변경·급정지)가 기울기를 지배하기 때문이다.
    기본을 huber 로 둔 이유는 성능 우위가 아니라 오차 0 에서 미분이 정의된다는 것뿐이다.

    어느 것을 쓰든 **마스크가 필요하다.** `target_offsets` 는 관측이 끊기면 짧아진다.
    없는 시각을 0 으로 두고 손실에 넣으면 "원점으로 돌아온다"를 학습한다.
    """
    if kind == "huber":
        per = nn.functional.huber_loss(
            pred, tgt, reduction="none", delta=HUBER_DELTA_M).sum(-1)
    else:
        # 0 에서 sqrt 의 기울기가 무한이 되는 것을 막는다 (오차 0 인 시각이 실제로 있다)
        dist = ((pred - tgt) ** 2).sum(-1).clamp(min=1e-12).sqrt()
        if kind == "huber_norm":
            per = nn.functional.huber_loss(
                dist, torch.zeros_like(dist), reduction="none",
                delta=HUBER_DELTA_M)
        elif kind == "ade":
            per = dist
        elif kind == "l2":
            per = dist ** 2
        else:
            raise ValueError(f"모르는 좌표 손실: {kind}")
    return (per * mask).sum() / mask.sum().clamp(min=1)


def wp_loss(model, d, w, sl, man_weight: float = 1.0, kind: str = "huber"):
    """좌표 손실 + 기동 분류 교차엔트로피.

    **평가 지표(ADE)를 그대로 손실로 쓰지 않는다.** 이유는 `coord_loss` 주석에 있다.
    또 하나의 차이: 이 손실은 (표본, 시각) 쌍 전체를 한 번에 평균하므로 관측 시각이
    많은 표본이 더 큰 비중을 갖는다. 반면 ADE 는 표본별로 평균한 뒤 표본 간 평균이라
    모든 표본이 같은 비중이다.

    기동을 **함께 학습한다.** 학습되지 않는 출력은 검증할 대상이 없다. 정답이 임의인
    표본(`maneuver_ambiguous`)은 -100 으로 두어 손실에서 빠진다.
    """
    out = call_model(model, d, sl)
    logits = None
    if isinstance(out, tuple):
        out, logits = out
    loss = coord_loss(out, w["tgt"][sl], w["tmask"][sl], kind)
    if logits is not None and man_weight > 0:
        loss = loss + man_weight * nn.functional.cross_entropy(
            logits, w["man"][sl], ignore_index=-100
        )
    return loss


def pair_separation_loss(
    model,
    d,
    w,
    pair_rows,
    tolerance_m: float,
    near_m: float,
):
    """정상 장면의 근접 pair가 GT보다 과도하게 가까워지는 것만 벌점 준다.

    각 model output과 target은 actor 자신의 현재 원점 기준 ENU 상대좌표다. 같은
    world ENU에서 거리를 재기 위해 각 actor의 `origin`을 더한다. GT보다 멀거나
    `tolerance_m` 안에서만 더 가까운 예측에는 0 손실이므로, GT 거리를 그대로
    맞추도록 강제하지 않는다.
    """
    ia, ib = pair_rows[:, 0], pair_rows[:, 1]

    def coords(indices):
        out = call_model(model, d, indices)
        return out[0] if isinstance(out, tuple) else out

    pred_a, pred_b = coords(ia), coords(ib)
    gt_a, gt_b = w["tgt"][ia], w["tgt"][ib]
    valid = w["tmask"][ia] & w["tmask"][ib]
    pred_a = pred_a + w["origin"][ia].unsqueeze(1)
    pred_b = pred_b + w["origin"][ib].unsqueeze(1)
    gt_a = gt_a + w["origin"][ia].unsqueeze(1)
    gt_b = gt_b + w["origin"][ib].unsqueeze(1)
    gt_dist = torch.linalg.vector_norm(gt_a - gt_b, dim=-1)
    pred_dist = torch.linalg.vector_norm(pred_a - pred_b, dim=-1)
    near_valid = valid & (gt_dist <= near_m)
    per_step = torch.relu(gt_dist - tolerance_m - pred_dist).square()
    return (per_step * near_valid).sum() / near_valid.sum().clamp(min=1)


def pair_relative_loss(model, d, w, pair_rows, near_m: float):
    """GT-near pair의 2D 상대 궤적을 맞춘다 (정상·사고 모두 포함)."""
    ia, ib = pair_rows[:, 0], pair_rows[:, 1]

    def coords(indices):
        out = call_model(model, d, indices)
        return out[0] if isinstance(out, tuple) else out

    pred_a, pred_b = coords(ia), coords(ib)
    gt_a, gt_b = w["tgt"][ia], w["tgt"][ib]
    valid = w["tmask"][ia] & w["tmask"][ib]
    pred_a = pred_a + w["origin"][ia].unsqueeze(1)
    pred_b = pred_b + w["origin"][ib].unsqueeze(1)
    gt_a = gt_a + w["origin"][ia].unsqueeze(1)
    gt_b = gt_b + w["origin"][ib].unsqueeze(1)
    pred_rel, gt_rel = pred_a - pred_b, gt_a - gt_b
    gt_dist = torch.linalg.vector_norm(gt_rel, dim=-1)
    near_valid = valid & (gt_dist <= near_m)
    per_xy = nn.functional.smooth_l1_loss(
        pred_rel, gt_rel, beta=1.0, reduction="none"
    )
    per_step = per_xy.mean(dim=-1)
    return (per_step * near_valid).sum() / near_valid.sum().clamp(min=1)


def combine_waypoint_losses(base_loss, pair_loss, pair_loss_weight: float,
                            relative_loss=None, relative_loss_weight: float = 0.0):
    """Auxiliary weight가 모두 0이면 base loss 객체를 그대로 돌려준다."""
    if ((pair_loss is None or pair_loss_weight == 0)
            and (relative_loss is None or relative_loss_weight == 0)):
        return base_loss
    loss = base_loss
    if pair_loss is not None and pair_loss_weight != 0:
        loss = loss + pair_loss_weight * pair_loss
    if relative_loss is not None and relative_loss_weight != 0:
        loss = loss + relative_loss_weight * relative_loss
    return loss


def report_waypoints(args, model, chosen, dev, name, TR, VA, TE,
                     tr, va, te, st, hist, best, final_path, best_path, scen):
    """좌표 회귀 보고. 후보 방식과 **다른 척도**로 재므로 표를 따로 낸다."""
    r_tr = eval_waypoints(model, *TR)
    r_va = eval_waypoints(model, *VA) if VA else None
    r_te = eval_waypoints(model, *TE)

    LW, CW = 22, 11
    width = LW + CW * (3 if r_va else 2)
    print(f"\n{'='*width}\n최종 성능 — {chosen}\n{'='*width}")
    print("ADE = 1..K초 평균 변위오차, FDE = 마지막 시각 오차. **작을수록 좋다.**")
    print(_pad("", LW) + _pad("학습", CW, True)
          + (_pad("검증", CW, True) if r_va else "") + _pad("시험", CW, True))

    def row(label, key, sub="all", fmt="{:.2f}m"):
        s = _pad(label, LW) + _pad(fmt.format(r_tr[sub][key]), CW, True)
        if r_va:
            s += _pad(fmt.format(r_va[sub][key]), CW, True)
        return s + _pad(fmt.format(r_te[sub][key]), CW, True)

    print(row("WaypointNet ADE", "model_ade"))
    print(row("규칙 기반 ADE", "rule_ade"))
    print(row("등속 ADE", "cv_ade"))
    print(row("후보 최선 ADE(오라클)", "oracle_ade"))
    print()
    print(row("WaypointNet FDE", "model_fde"))
    print(row("규칙 기반 FDE", "rule_fde"))
    print(row("등속 FDE", "cv_fde"))
    print()
    print(row("표본 수", "n", fmt="{:,}"))

    print(f"\n정답이 후보에 있는지로 나눠 본 시험 ADE — 이득이 어디서 오는가")
    print(_pad("", 22) + _pad("표본", 9, True) + _pad("WaypointNet", 13, True)
          + _pad("규칙", 9, True) + _pad("차이", 9, True))
    for key, label in (("matched", "정답이 후보에 있음"),
                       ("unmatched", "정답이 후보에 없음")):
        m = r_te[key]
        if not m["n"]:
            continue
        print(_pad(label, 22) + _pad(f"{m['n']:,}", 9, True)
              + _pad(f"{m['model_ade']:.2f}m", 13, True)
              + _pad(f"{m['rule_ade']:.2f}m", 9, True)
              + _pad(f"{m['model_ade']-m['rule_ade']:+.2f}m", 9, True))
    if "maneuver" in r_te:
        mv, mo = r_te["maneuver"], r_te["off_candidate"]
        print(f"\n기동 라벨 — 모델이 직접 분류한다 (좌표에서 되짚지 않는다)")
        print(_pad("  전체 정확도", 30)
              + _pad(f"{mv['model_acc']:.1%}", 10, True)
              + f"   (n={mv['n']:,}, '후보 밖' 포함 5종 분류)")
        print(_pad("  후보 안 표본만 — 모델", 30)
              + _pad(f"{mv['model_acc_on_candidate']:.1%}", 10, True)
              + f"   (n={mv['n_on_candidate']:,})")
        print(_pad("  후보 안 표본만 — 규칙 기반", 30)
              + _pad(f"{mv['rule_acc_on_candidate']:.1%}", 10, True)
              + "   ← 같은 표본, 비교 기준")
        print(f"\n'후보 밖' 판정 (기저 {mo['base_rate']:.1%})"
              f"  정밀도 {mo['precision']:.1%} · 재현율 {mo['recall']:.1%}"
              f" · 경보 {mo['n_flagged']:,}건")
        print("  규칙 기반은 이 판정을 **아예 할 수 없다** — 후보에만 확률을 나눈다.")
    print(f"\n규칙 기반의 시각 커버리지 {r_te['rule_step_coverage']:.1%} "
          f"— 후보가 도로 끝에서 잘려 예측하지 못한 시각이 있다 "
          f"(비교는 양쪽에 값이 있는 시각에서만 했다). 모델은 항상 100%.")

    meta = {
        "device": dev, "task": args.task, "epochs": args.epochs,
        "chosen": chosen, "best_epoch": best["epoch"],
        "hyper": {"hidden": args.hidden, "dropout": args.dropout,
                  "lr": args.lr, "weight_decay": args.weight_decay,
                  "batch_size": args.batch_size, "seed": args.seed,
                  "steps": args.steps, "wp_loss": args.wp_loss,
                  "man_weight": args.man_weight,
                  "pair_loss_weight": args.pair_loss_weight,
                  "pair_tolerance_m": args.pair_tolerance_m,
                  "pair_near_m": args.pair_near_m,
                  "pair_batch_size": args.pair_batch_size,
                  "relative_loss_weight": args.relative_loss_weight,
                  "relative_near_m": args.relative_near_m,
                  "relative_batch_size": args.relative_batch_size,
                  "history_encoder": args.history_encoder,
                  "history_hidden": args.history_hidden},
        "split": {"by": args.split_by, "val_frac": args.val_frac,
                  "test_frac": args.test_frac, "n_train": len(tr),
                  "n_val": len(va), "n_test": len(te),
                  "scenarios_train": scen(tr), "scenarios_test": scen(te)},
        "counts": st,
        "result": {"train": r_tr, "val": r_va, "test": r_te},
        "history": hist,
    }
    with open(os.path.join(args.out, f"train_report_{args.task}.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print(f"\n  {final_path}\n  {best_path}"
          f"\n  {os.path.join(args.out, f'train_report_{args.task}.json')}")
    print(f"\n쓰는 법:\n"
          f"  from traffic_llm.predict_model import TorchPredictor\n"
          f"  cfg.predictor = TorchPredictor('{best_path}', mode='waypoints')")
    return 0


# ------------------------------------------------------------------ 실행

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RankNet 경로 예측 모델 학습")
    ap.add_argument("--data", default="out/predict_dataset",
                    help="make_predict_dataset.py 의 출력 디렉터리")
    ap.add_argument("--task", choices=["rank", "waypoints"], default="rank",
                    help="rank: 후보 중 정답 인덱스 분류 (기본) / "
                         "waypoints: 좌표를 직접 회귀 — 후보 집합에 정답이 없는 "
                         "표본도 쓸 수 있고 평가는 ADE/FDE 로 한다")
    ap.add_argument("--architecture", choices=["waypoint", "interaction"],
                    default="waypoint",
                    help="waypoints: 기존 단독 WaypointNet 또는 주변 actor attention을 "
                         "쓰는 InteractionWaypointNet")
    ap.add_argument("--steps", type=int, default=5,
                    help="waypoints: 예측할 시각 수 (1..K초)")
    ap.add_argument("--history-encoder",
                    choices=["transformer", "gru", "mlp", "none"],
                    default="transformer",
                    help="waypoints: 과거 궤적(5초, 1초 간격) 인코더. "
                         "gru=기본, transformer=self-attention 1층, "
                         "mlp=평탄화(순서 구조 없음, 대조군), "
                         "none=이력을 쓰지 않음(예전 동작)")
    ap.add_argument("--history-hidden", type=int, default=64,
                    help="과거 궤적 인코더의 은닉 차원")
    ap.add_argument("--wp-loss", choices=["huber", "huber_norm", "ade", "l2"],
                    default="huber",
                    help="waypoints 좌표 손실. 기본 huber(축별, delta=2m). "
                         "ade 는 평가 지표를 그대로 손실로 쓴다 — 기울기 크기가 "
                         "오차와 무관해 0 근처 수렴이 나쁘다")
    ap.add_argument("--man-weight", type=float, default=1.0,
                    help="waypoints: 기동 분류 손실 가중치 (0 이면 좌표만)")
    ap.add_argument("--pair-loss-weight", type=float, default=0.0,
                    help="waypoints: 정상 장면 pair separation 보조 손실 가중치 (기본 0)")
    ap.add_argument("--pair-tolerance-m", type=float, default=0.5,
                    help="GT보다 이 값(m) 이상 가까워질 때만 pair 손실을 준다")
    ap.add_argument("--pair-near-m", type=float, default=10.0,
                    help="GT 미래가 이 거리(m) 안으로 오는 정상 pair만 표본으로 쓴다")
    ap.add_argument("--pair-batch-size", type=int, default=128,
                    help="각 actor minibatch와 별도로 뽑을 pair 수")
    ap.add_argument("--relative-loss-weight", type=float, default=0.0,
                    help="waypoints: GT 상대 궤적 보조 손실 가중치 (기본 0)")
    ap.add_argument("--relative-near-m", type=float, default=10.0,
                    help="GT 상대 궤적 손실을 적용할 GT pair 거리(m)")
    ap.add_argument("--relative-batch-size", type=int, default=128,
                    help="각 actor minibatch와 별도로 뽑을 relative pair 수")
    ap.add_argument("--save-every", type=int, default=10,
                    help="N epoch마다 중간 checkpoint 저장 (0 이면 저장 안 함)")
    ap.add_argument("--out", default="out/predict_model", help="모델 저장 위치")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu")
    ap.add_argument("--split-by", choices=["dataset", "scenario"],
                    default="dataset",
                    help="dataset: 데이터셋의 val 을 시험셋으로 / "
                         "scenario: 시나리오 id 해시로 분할")
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="--split-by scenario 일 때 시험셋 비율")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="학습셋에서 떼어낼 검증셋 비율 (0 이면 마지막 epoch 저장)")
    ap.add_argument("--include-ambiguous", action="store_true",
                    help="라벨 임의 표본을 소프트 타깃으로 학습에 포함")
    ap.add_argument("--log-every", type=int, default=5,
                    help="몇 epoch 마다 한 줄 찍을지")
    args = ap.parse_args(argv)
    if args.pair_loss_weight < 0:
        ap.error("--pair-loss-weight 는 0 이상이어야 합니다")
    if args.relative_loss_weight < 0:
        ap.error("--relative-loss-weight 는 0 이상이어야 합니다")
    if args.pair_tolerance_m < 0 or args.pair_near_m <= 0:
        ap.error("--pair-tolerance-m 은 0 이상, --pair-near-m 은 0 초과여야 합니다")
    if args.relative_near_m <= 0:
        ap.error("--relative-near-m 은 0 초과여야 합니다")
    if args.pair_batch_size <= 0:
        ap.error("--pair-batch-size 는 1 이상이어야 합니다")
    if args.relative_batch_size <= 0:
        ap.error("--relative-batch-size 는 1 이상이어야 합니다")
    if args.save_every < 0:
        ap.error("--save-every 는 0 이상이어야 합니다")
    if args.task != "waypoints" and args.pair_loss_weight != 0:
        ap.error("--pair-loss-weight 는 --task waypoints 에서만 사용할 수 있습니다")
    if args.task != "waypoints" and args.relative_loss_weight != 0:
        ap.error("--relative-loss-weight 는 --task waypoints 에서만 사용할 수 있습니다")
    if args.pair_loss_weight > 0 and args.relative_loss_weight > 0:
        raise ValueError(
            "--pair-loss-weight 와 --relative-loss-weight 는 동시에 사용할 수 없습니다"
        )

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"장치: {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))

    path = os.path.join(args.data, "samples.jsonl")
    if not os.path.exists(path):
        print(f"없음: {path}", file=sys.stderr)
        return 1
    t0 = time.time()
    rows, st = read_samples(path, args.include_ambiguous, args.task)
    print(f"과제: {args.task}"
          + ("  (후보 중 정답 인덱스 분류)" if args.task == "rank"
             else "  (좌표 직접 회귀 — ADE/FDE 로 평가)"))
    print(f"표본 {st['n_all']:,} 읽음 ({time.time()-t0:.0f}초) → 사용 {st['n_kept']:,}")
    if args.task == "rank":
        if args.architecture != "waypoint":
            ap.error("--architecture interaction 은 --task waypoints 에서만 지원합니다")
        print(f"  제외: 정답없음 {st['n_unmatched']:,} · 후보1개 {st['n_single']:,}"
              + (f" · 라벨임의 {st['n_ambiguous']:,}"
                 if not args.include_ambiguous else
                 f" (라벨임의 {st['n_ambiguous']:,} 는 소프트 타깃으로 포함)"))
    else:
        print(f"  포함: 정답이 후보에 없는 표본 {st['n_unmatched']:,} "
              f"({st['n_unmatched']/st['n_all']:.1%}) — **이것이 이 과제의 이유다**"
              f" · 미래 1점뿐 제외 {st['n_short_future']:,}")
    if not rows:
        print("쓸 표본이 없습니다.", file=sys.stderr)
        return 1
    if args.architecture == "interaction" and not any(r["interaction"] for r in rows):
        print("interaction feature가 없는 구 버전 데이터입니다 — "
              "examples/make_predict_dataset.py 로 데이터를 다시 생성하십시오.",
              file=sys.stderr)
        return 1

    tr, va, te = split_rows(rows, args.split_by, args.val_frac, args.test_frac)
    # 빈 분할로 진행하면 한참 뒤에 엉뚱한 곳에서 터진다. 여기서 멈춘다.
    for name, part in (("학습", tr), ("시험", te)):
        if not part:
            print(f"{name} 표본이 0개입니다 — 분할 설정을 확인하십시오 "
                  f"(--split-by {args.split_by}, --test-frac {args.test_frac}, "
                  f"--val-frac {args.val_frac}).", file=sys.stderr)
            return 1
    if args.task == "rank" and not any(not r["amb"] for r in te):
        print("시험 표본이 전부 라벨 임의입니다 — 채점할 수 없습니다.",
              file=sys.stderr)
        return 1
    n_max = max(len(r["c"]) for r in rows)
    d_tr = to_tensors(tr, dev, n_max)
    d_va = to_tensors(va, dev, n_max) if va else None
    d_te = to_tensors(te, dev, n_max)
    if args.task == "waypoints":
        w_tr = to_waypoint_tensors(tr, dev, n_max, args.steps)
        w_va = to_waypoint_tensors(va, dev, n_max, args.steps) if va else None
        w_te = to_waypoint_tensors(te, dev, n_max, args.steps)
        # Index 자체는 training split에만 만들며, sampling/추가 forward는 weight가
        # 0보다 클 때만 한다. 따라서 기본값은 기존의 난수 소비와 학습 경로도 같다.
        pair_index = build_normal_pair_index(tr, args.steps, args.pair_near_m)
        pair_index_tensor = (torch.tensor(pair_index, dtype=torch.long, device=dev)
                             if args.pair_loss_weight > 0 and pair_index else None)
        relative_pair_index = (build_relative_pair_index(
            tr, args.steps, args.relative_near_m
        ) if args.relative_loss_weight > 0 else [])
        # Auxiliary sampling은 global RNG를 소비하지 않도록 CPU generator를 쓴다.
        relative_pair_index_tensor = torch.tensor(
            relative_pair_index, dtype=torch.long
        ) if args.relative_loss_weight > 0 and relative_pair_index else None
        relative_generator = None
        if args.relative_loss_weight > 0:
            relative_generator = torch.Generator(device="cpu")
            relative_generator.manual_seed(args.seed + 100003)
        print(f"  정상 근접 pair {len(pair_index):,} "
              f"(GT 미래 {args.pair_near_m:g}m 이내)")
        if args.relative_loss_weight > 0:
            print(f"  상대 궤적 근접 pair {len(relative_pair_index):,} "
                  f"(정상·사고, GT 미래 {args.relative_near_m:g}m 이내)")

    def scen(rs):
        return len({r["scenario"] for r in rs})

    print(f"  학습 {len(tr):,} (시나리오 {scen(tr)})")
    print(f"  검증 {len(va):,} (시나리오 {scen(va)})" if va else "  검증 없음")
    print(f"  시험 {len(te):,} (시나리오 {scen(te)})")
    overlap = {r["scenario"] for r in tr} & {r["scenario"] for r in te}
    if overlap:
        print(f"경고: 학습·시험에 같은 시나리오가 {len(overlap)}개 겹칩니다 — "
              f"성능이 과대평가됩니다", file=sys.stderr)

    if args.task == "rank":
        model = RankNet(hidden=args.hidden, depth=args.depth,
                        dropout=args.dropout).to(dev)
        name = "ranknet"
    else:
        model_cls = InteractionWaypointNet if args.architecture == "interaction" else WaypointNet
        model = model_cls(hidden=args.hidden, n_steps=args.steps,
                          dropout=args.dropout,
                          history_encoder=args.history_encoder,
                          history_hidden=args.history_hidden).to(dev)
        name = "interaction_waypointnet" if args.architecture == "interaction" else "waypointnet"
    # 표준화 통계는 **학습셋만** 으로 정한다 (검증·시험을 보면 누출이다).
    normalizer = {"interaction": d_tr["interaction"],
                  "interaction_mask": d_tr["interaction_mask"]}
    if args.architecture == "interaction":
        model.fit_normalizer(d_tr["g"], d_tr["c"], d_tr["mask"], **normalizer)
    else:
        model.fit_normalizer(d_tr["g"], d_tr["c"], d_tr["mask"])
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\n{type(model).__name__} hidden={args.hidden} "
          f"dropout={args.dropout} · 파라미터 {n_par:,} "
          f"(입력 표준화 통계는 학습셋에서만 산출)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    if args.task == "rank":
        b_tr, b_te = evaluate(model, d_tr), evaluate(model, d_te)
        print(f"규칙 기반(predict.py) top-1 — 학습 {b_tr['rule_top1']:.2%} · "
              f"시험 {b_te['rule_top1']:.2%}\n")
    else:
        b_te = eval_waypoints(model, d_te, w_te)
        print(f"규칙 기반(predict.py) 시험 ADE {b_te['all']['rule_ade']:.2f}m · "
              f"FDE {b_te['all']['rule_fde']:.2f}m  "
              f"(등속 ADE {b_te['all']['cv_ade']:.2f}m)\n")

    os.makedirs(args.out, exist_ok=True)
    # `-inf` 로 시작해야 한다. 선택 지표는 rank 에서는 정확도(0~1)지만
    # waypoints 에서는 **-ADE**(음수)다. -1.0 으로 두면 -3.05 > -1.0 이 거짓이라
    # 최고 가중치가 한 번도 저장되지 않는다 (실측으로 그렇게 됐다).
    best = {"val": float("-inf"), "epoch": 0}
    hist = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(d_tr["n"], device=dev)
        tot = tot_base = tot_pair = tot_rel = 0.0
        n_pair_steps = n_rel_steps = 0
        for i in range(0, d_tr["n"], args.batch_size):
            sl = perm[i:i + args.batch_size]
            if args.task == "rank":
                loss = loss_of(model, d_tr, sl, args.include_ambiguous)
                base_loss = loss
                pair_loss = None
                relative_loss = None
            else:
                base_loss = wp_loss(model, d_tr, w_tr, sl,
                                    man_weight=args.man_weight,
                                    kind=args.wp_loss)
                pair_loss = None
                relative_loss = None
                if pair_index_tensor is not None:
                    pair_pick = torch.randint(
                        len(pair_index_tensor), (args.pair_batch_size,), device=dev
                    )
                    pair_rows = pair_index_tensor[pair_pick]
                    pair_loss = pair_separation_loss(
                        model, d_tr, w_tr, pair_rows, args.pair_tolerance_m,
                        args.pair_near_m,
                    )
                if relative_pair_index_tensor is not None:
                    # Base minibatch 뒤의 dropout 난수열은 auxiliary forward가
                    # 없었던 대조군과 같아야 한다. graph는 유지한 채 RNG만 복원한다.
                    pair_pick = torch.randint(
                        len(relative_pair_index_tensor),
                        (args.relative_batch_size,),
                        generator=relative_generator, device="cpu",
                    )
                    pair_rows = relative_pair_index_tensor[pair_pick].to(dev)
                    cpu_rng_state = torch.get_rng_state()
                    cuda_rng_states = (
                        torch.cuda.get_rng_state_all()
                        if str(dev).startswith("cuda") else None
                    )
                    try:
                        relative_loss = pair_relative_loss(
                            model, d_tr, w_tr, pair_rows, args.relative_near_m
                        )
                    finally:
                        torch.set_rng_state(cpu_rng_state)
                        if cuda_rng_states is not None:
                            torch.cuda.set_rng_state_all(cuda_rng_states)
                loss = combine_waypoint_losses(
                    base_loss, pair_loss, args.pair_loss_weight,
                    relative_loss, args.relative_loss_weight,
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(sl)
            tot_base += base_loss.item() * len(sl)
            if pair_loss is not None:
                tot_pair += pair_loss.item()
                n_pair_steps += 1
            if relative_loss is not None:
                tot_rel += relative_loss.item()
                n_rel_steps += 1
        sched.step()
        if args.task == "rank":
            m_tr = evaluate(model, d_tr)
            m_va = evaluate(model, d_va) if d_va else None
            # 클수록 좋다
            sel = (m_va or m_tr)["model_top1"]
            shown = [("학습", f"{m_tr['model_top1']:.2%}")]
            if m_va:
                shown.append(("검증", f"{m_va['model_top1']:.2%}"))
        else:
            m_tr = eval_waypoints(model, d_tr, w_tr)
            m_va = eval_waypoints(model, d_va, w_va) if d_va else None
            # ADE 는 **작을수록** 좋다 — 부호를 뒤집어 같은 비교식을 쓴다
            sel = -(m_va or m_tr)["all"]["model_ade"]
            shown = [("학습", f"{m_tr['all']['model_ade']:.2f}m")]
            if m_va:
                shown.append(("검증", f"{m_va['all']['model_ade']:.2f}m"))
        hist.append({"epoch": ep, "loss": tot / d_tr["n"],
                     "base_loss": tot_base / d_tr["n"],
                     "pair_loss": (tot_pair / n_pair_steps if n_pair_steps else 0.0),
                     "relative_loss": (tot_rel / n_rel_steps if n_rel_steps else 0.0),
                     "select": sel})
        if sel > best["val"]:
            best = {"val": sel, "epoch": ep}
            torch.save(model, os.path.join(args.out, f"{name}_best.pt"))
        if args.save_every > 0 and ep % args.save_every == 0:
            torch.save(model, os.path.join(args.out, f"{name}_epoch{ep:03d}.pt"))
        if ep % args.log_every == 0 or ep == 1 or ep == args.epochs:
            if args.task == "rank":
                te_s = f"{evaluate(model, d_te)['model_top1']:.2%}"
            else:
                te_s = f"{eval_waypoints(model, d_te, w_te)['all']['model_ade']:.2f}m"
            losses = (f"loss {tot/d_tr['n']:.4f}  base {tot_base/d_tr['n']:.4f}"
                      + (f"  pair {tot_pair/n_pair_steps:.4f}"
                         if n_pair_steps else "  pair 0.0000")
                      + (f"  rel {tot_rel/n_rel_steps:.4f}"
                         if n_rel_steps else "  rel 0.0000"))
            print(f"  epoch {ep:3d}  {losses}  "
                  + "  ".join(f"{k} {v}" for k, v in shown)
                  + f"  시험 {te_s}")
    print(f"\n{args.epochs} epoch, {time.time()-t0:.0f}초")

    final_path = os.path.join(args.out, f"{name}_final.pt")
    torch.save(model, final_path)
    best_path = os.path.join(args.out, f"{name}_best.pt")

    # 최종 보고는 **검증셋으로 고른** 가중치로 낸다.
    if os.path.exists(best_path) and d_va is not None:
        model = torch.load(best_path, map_location=dev, weights_only=False)
        chosen = f"{name}_best.pt (검증 최고, epoch {best['epoch']})"
    else:
        chosen = f"{name}_final.pt (마지막 epoch)"

    if args.task == "waypoints":
        return report_waypoints(args, model, chosen, dev, name,
                                (d_tr, w_tr), (d_va, w_va) if d_va else None,
                                (d_te, w_te), tr, va, te, st, hist, best,
                                final_path, best_path, scen)

    r_tr, r_te = evaluate(model, d_tr), evaluate(model, d_te)
    r_va = evaluate(model, d_va) if d_va else None
    LW, CW = 20, 11
    width = LW + CW * (3 if r_va else 2)
    print(f"\n{'='*width}\n최종 성능 — {chosen}\n{'='*width}")
    print(_pad("", LW) + _pad("학습", CW, True)
          + (_pad("검증", CW, True) if r_va else "")
          + _pad("시험", CW, True))

    def row(label, key, fmt="{:.2%}"):
        s = _pad(label, LW) + _pad(fmt.format(r_tr[key]), CW, True)
        if r_va:
            s += _pad(fmt.format(r_va[key]), CW, True)
        return s + _pad(fmt.format(r_te[key]), CW, True)

    print(row("RankNet top-1", "model_top1"))
    print(row("규칙 기반 top-1", "rule_top1"))
    print(row("무작위 기대치", "random"))
    print()
    print(row("RankNet 기동 라벨", "model_man"))
    print(row("규칙 기반 기동 라벨", "rule_man"))
    print()

    def gain(m):
        return m["model_top1"] - m["rule_top1"]

    print(_pad("규칙 대비 향상", LW) + _pad(f"{gain(r_tr):+.2%}", CW, True)
          + (_pad(f"{gain(r_va):+.2%}", CW, True) if r_va else "")
          + _pad(f"{gain(r_te):+.2%}", CW, True))
    print(row("표본 수", "n", "{:,}"))

    towns = per_town(model, d_te)
    if len(towns) > 1:
        print(f"\n타운별 시험 성능 — 전체 수치가 타운 구성에 끌려가는지 보기 위한 것")
        print(_pad("", 12) + _pad("표본", 9, True) + _pad("RankNet", 10, True)
              + _pad("규칙", 9, True) + _pad("향상", 9, True))
        for t in towns:
            print(_pad(t["town"], 12) + _pad(f"{t['n']:,}", 9, True)
                  + _pad(f"{t['model']:.1%}", 10, True)
                  + _pad(f"{t['rule']:.1%}", 9, True)
                  + _pad(f"{t['model']-t['rule']:+.1%}", 9, True))

    meta = {
        "device": dev,
        "task": args.task,
        "epochs": args.epochs,
        "chosen": chosen,
        "best_epoch": best["epoch"],
        "hyper": {"hidden": args.hidden, "depth": args.depth,
                  "dropout": args.dropout, "lr": args.lr,
                  "weight_decay": args.weight_decay,
                  "batch_size": args.batch_size, "seed": args.seed},
        "split": {"by": args.split_by, "val_frac": args.val_frac,
                  "test_frac": args.test_frac,
                  "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                  "scenarios_train": scen(tr), "scenarios_test": scen(te)},
        "excluded": st,
        "include_ambiguous": args.include_ambiguous,
        "result": {"train": r_tr, "val": r_va, "test": r_te},
        "test_by_town": towns,
        "history": hist,
    }
    with open(os.path.join(args.out, f"train_report_{args.task}.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print(f"\n  {final_path}")
    if d_va is not None:
        print(f"  {best_path}")
    print(f"  {os.path.join(args.out, f'train_report_{args.task}.json')}")
    print(f"\n쓰는 법:\n"
          f"  from traffic_llm.predict_model import TorchPredictor\n"
          f"  cfg.predictor = TorchPredictor('{best_path}')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
