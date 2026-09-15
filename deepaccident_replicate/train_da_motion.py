"""DeepAccident 운동 예측 모델 학습 — 돌아가는 비교 대상 만들기.

원본 저장소는 이 머신에서 돌지 않는다(가중치 미배포 + CUDA 10.2/mmcv 가 sm_120 을
지원하지 않음). `deepaccident_replicate/da_motion.py` 에 그 **방법**을 재구현했고, 이 스크립트가
그것을 실제 데이터로 학습시킨다.

두 단계다. 수집은 NFS 대역폭에 묶여 느리므로 **한 번만** 하고 캐시를 남긴다.

    # 1) 프레임 캐시 (시나리오별 액터 배열). train+val 한 번씩
    .venv/bin/python deepaccident_replicate/train_da_motion.py cache \
        --root /home/sryu/inclab-nas/DeepAccident --split train \
        --out out/da_motion/frames_train.npz

    # 2) 학습
    .venv/bin/python deepaccident_replicate/train_da_motion.py train \
        --train-cache out/da_motion/frames_train.npz \
        --val-cache   out/da_motion/frames_val.npz \
        --out out/da_motion/da_motion_best.pt

캐시는 래스터가 아니라 **액터 배열**을 담는다 (표본당 몇 KB). 래스터는 학습 중에
만든다 — 200×200×5 를 미리 저장하면 표본당 5 MB 라 감당이 안 된다.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_llm.config import PipelineConfig
from deepaccident_replicate.da_motion import (
    FRAME_DT_S,
    STATIONARY_MPS,
    collision_heatmap_label,
    poly_gap,
    _rect_corners,
    N_CHANNELS,
    N_FUTURE,
    N_PAST,
    BevGrid,
    FrameActors,
    ego_frame,
    flow_label,
    rasterize,
)
from traffic_llm.da_runner import DeepAccidentRunner
from traffic_llm.deepaccident import CLASS_SIZES, estimate_collision


# ------------------------------------------------------------ 캐시

def agent_ego_ids(snaps) -> List[str]:
    """스냅샷들에 등장하는 **관측 차량**들의 actor id.

    DeepAccident 는 차량 4대 + 인프라 1대로 기록하는데 인프라는 차량이 아니므로
    운동 예측의 자기 중심 좌표계가 될 수 없다. 실제로 나오는 것은
    `EGO_ego_vehicle`, `EGO_ego_vehicle_behind`, `EGO_other_vehicle`,
    `EGO_other_vehicle_behind` 넷이다.
    """
    seen: Dict[str, int] = {}
    for s in snaps:
        for a in s.actors:
            if a.kind == "ego" and a.world_xy is not None:
                seen[a.actor_id] = seen.get(a.actor_id, 0) + 1
    # 프레임 대부분에 등장하는 것만 (한두 프레임 스친 것은 중심으로 쓸 수 없다)
    need = max(1, int(0.8 * len(snaps)))
    return sorted(a for a, n in seen.items() if n >= need)


def cache_split(root: str, split: str, out_path: str, rate_hz: float,
                limit: Optional[int] = None, maps: Optional[str] = None) -> None:
    """시나리오 → 프레임별 액터 배열을 npz 하나로.

    저장 형태는 시나리오마다 가변 길이라 object 배열을 쓴다. 프레임마다
    `(ids, xy, yaw, vel, size)` 다.
    """
    cfg = PipelineConfig()
    cfg.deepaccident.observation_mode = "sensor3d"
    runner = DeepAccidentRunner(root, cfg)
    scs = [s for s in runner.list_scenarios() if s.split == split]
    if limit:
        scs = scs[:limit]
    grid = BevGrid()
    sizes = CLASS_SIZES
    print(f"{split}: 시나리오 {len(scs)}개 · 에이전트당 한 벌", flush=True)

    scen_names: List[str] = []
    scen_types: List[str] = []
    scen_agents: List[str] = []
    scen_frames: List[object] = []
    scen_coll: List[dict] = []
    t0 = time.time()
    for i, sc in enumerate(scs, 1):
        try:
            res = runner.build(sc.scenario, sc.scenario_type)
            snaps = list(res.snapshots(rate_hz=rate_hz))
        except Exception as e:
            print(f"  [{i}/{len(scs)}] {sc.scenario} 실패: {type(e).__name__}: {e}",
                  flush=True)
            continue
        # 충돌 정답. meta 만 읽으므로 값싸다 — 사고 헤드의 라벨이 된다.
        try:
            col = estimate_collision(sc, cfg.deepaccident)
        except Exception:
            col = None
        coll_rec = {
            "occurred": bool(col.occurred) if col else False,
            "time_s": (float(col.time_s)
                       if col and col.occurred and col.time_s is not None else None),
        }

        # **에이전트마다 한 벌.** 원본도 관측 차량 각각을 별개 표본으로 쓴다
        # (5 에이전트 × 전 프레임). 스냅샷은 시나리오당 한 번만 만들므로 NFS 비용이
        # 늘지 않는데 표본은 몇 배가 된다. 중심이 달라지면 ±50 m 창이 잘라내는
        # 장면도 달라지므로 같은 입력의 복제가 아니다.
        for aid in agent_ego_ids(snaps):
            frames = []
            for s in snaps:
                fr = ego_frame(s, grid, sizes, ego_id=aid)
                if fr is None:
                    frames.append(None)
                    continue
                frames.append({
                    "t": float(s.t),
                    "ids": np.array(fr.ids, dtype=object),
                    "xy": fr.xy, "yaw": fr.yaw, "vel": fr.vel, "size": fr.size,
                })
            if sum(1 for f in frames if f is not None) < N_PAST + N_FUTURE:
                continue
            scen_names.append(sc.scenario)
            scen_types.append(sc.scenario_type)
            scen_agents.append(aid)
            scen_frames.append(np.array(frames, dtype=object))
            scen_coll.append(dict(coll_rec))
        if i % 20 == 0 or i == len(scs):
            print(f"  [{i}/{len(scs)}] 담은 (시나리오×에이전트) {len(scen_names)}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        names=np.array(scen_names, dtype=object),
        types=np.array(scen_types, dtype=object),
        agents=np.array(scen_agents, dtype=object),
        frames=np.array(scen_frames, dtype=object),
        collisions=np.array(scen_coll, dtype=object),
        rate_hz=rate_hz,
    )
    n_f = sum(len(f) for f in scen_frames)
    print(f"저장: {out_path}  (시나리오×에이전트) {len(scen_names)}  "
          f"고유 시나리오 {len(set(zip(scen_types, scen_names)))}  프레임 {n_f}")


def cache_collisions(root: str, split: str, out_path: str) -> None:
    """시나리오별 충돌 정답만 따로 뽑는다.

    `estimate_collision` 은 meta·label 만 읽으므로 프레임 캐시를 다시 만들지 않아도
    된다 — 프레임 캐시는 스냅샷 파이프라인을 다 돌려야 해서 훨씬 비싸다.
    """
    cfg = PipelineConfig()
    cfg.deepaccident.observation_mode = "sensor3d"
    runner = DeepAccidentRunner(root, cfg)
    scs = [s for s in runner.list_scenarios() if s.split == split]
    names, types, coll = [], [], []
    for i, sc in enumerate(scs, 1):
        try:
            c = estimate_collision(sc, cfg.deepaccident)
        except Exception:
            c = None
        names.append(sc.scenario)
        types.append(sc.scenario_type)
        coll.append({
            "occurred": bool(c.occurred) if c else False,
            "time_s": (float(c.time_s)
                       if c and c.occurred and c.time_s is not None else None),
        })
        if i % 50 == 0 or i == len(scs):
            print(f"  [{i}/{len(scs)}]", flush=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, names=np.array(names, dtype=object),
                        types=np.array(types, dtype=object),
                        collisions=np.array(coll, dtype=object))
    n = sum(1 for c in coll if c["occurred"])
    n_t = sum(1 for c in coll if c["time_s"] is not None)
    print(f"저장: {out_path}  시나리오 {len(names)}  충돌 {n} (시각 있음 {n_t})")


def load_cache(path: str, coll_path: Optional[str] = None):
    """→ (이름, 프레임, 충돌정보).

    충돌정보는 캐시 안에 있으면 그것을 쓰고, `coll_path` 를 주면 **이름으로 맞춰**
    덮어쓴다. 프레임 캐시를 다시 만들지 않고 충돌 정답만 붙일 때 쓴다.
    """
    d = np.load(path, allow_pickle=True)
    names, frames = list(d["names"]), list(d["frames"])
    coll = (list(d["collisions"]) if "collisions" in d.files
            else [{"occurred": False, "time_s": None} for _ in names])
    if coll_path and os.path.exists(coll_path):
        e = np.load(coll_path, allow_pickle=True)
        # **시나리오 이름만으로는 유일하지 않다.** 같은 id 가 `*_accident` 와
        # `*_normal` 폴더에 각각 있어(val 104개 중 고유 이름 52개), 이름만 키로 쓰면
        # 한쪽이 다른 쪽을 덮어써 충돌 51개가 2개로 줄어든다. 유형까지 키에 넣는다.
        if "types" in d.files and "types" in e.files:
            key = [f"{t}/{n}" for t, n in zip(list(d["types"]), names)]
            lut = dict(zip([f"{t}/{n}" for t, n in
                            zip(list(e["types"]), list(e["names"]))],
                           list(e["collisions"])))
            missing = [k for k in key if k not in lut]
            if missing:
                print(f"  경고: 충돌 정답이 없는 시나리오 {len(missing)}개 "
                      f"(예: {missing[:2]}) — 무충돌로 둔다")
            coll = [lut.get(k, {"occurred": False, "time_s": None}) for k in key]
        elif len(e["names"]) == len(names) and list(e["names"]) == names:
            # 옛 프레임 캐시에는 유형이 없다. 두 캐시가 **같은 순서의 같은 목록**일
            # 때만 위치로 맞춘다 — 이름이 순서까지 일치하는지 확인한 뒤에만.
            print("  참고: 프레임 캐시에 시나리오 유형이 없어 **위치**로 맞춘다 "
                  "(이름 순서가 일치함을 확인했다)")
            coll = list(e["collisions"])
        else:
            raise SystemExit(
                "충돌 정답을 프레임 캐시에 맞출 수 없다. `cache` 를 다시 돌려 "
                "시나리오 유형을 포함시켜라."
            )
    return names, frames, coll


def _to_frame(rec) -> FrameActors:
    return FrameActors(ids=list(rec["ids"]), xy=rec["xy"], yaw=rec["yaw"],
                       vel=rec["vel"], size=rec["size"])


def sample_index(frames_list, coll_list=None) -> List[tuple]:
    """(시나리오 idx, 현재 프레임 idx) 목록.

    **미래 프레임이 다 있을 것을 요구하지 않는다.** DeepAccident 시나리오는 충돌
    순간에 녹화가 끝나므로, 미래 4프레임이 모두 있는 표본은 정의상 녹화 끝보다
    2초 이상 앞이고 따라서 **사고보다 2초 이상 앞**이다. 그것만 쓰면 사고가 지평
    안에 든 표본이 거의 없다 — 실측으로 train 21,220 프레임 중 양성이 90개(0.4 %)
    뿐이었고 사고 헤드가 전부 음성으로 붕괴했다.

    관측(과거 3프레임)만 요구하고, 없는 미래 프레임은 **운동 손실에서 가린다**.
    사고 라벨은 충돌 정답에서 오므로 프레임이 없어도 만들 수 있다. 이 완화로
    양성 표본이 97 → 624 개가 된다.
    """
    out = []
    for si, frames in enumerate(frames_list):
        n = len(frames)
        for c in range(N_PAST - 1, n):
            if any(frames[k] is None for k in range(c - N_PAST + 1, c + 1)):
                continue
            has_future = any(c + k < n and frames[c + k] is not None
                             for k in range(1, N_FUTURE + 1))
            ct = None
            if coll_list:
                cl = coll_list[si] or {}
                ct = cl.get("time_s") if cl.get("occurred") else None
            in_horizon = (ct is not None
                          and float(frames[c]["t"]) < ct
                          <= float(frames[c]["t"]) + N_FUTURE * FRAME_DT_S)
            if has_future or in_horizon:
                out.append((si, c))
    return out


# ------------------------------------------------------------ 데이터셋

def collision_xy(fr: Optional[FrameActors]) -> Optional[tuple]:
    """이 프레임에서 **움직이는** 쌍 중 가장 가까운 둘의 중점. 접촉이 없으면 None.

    충돌 **시각**은 정답에서 오고, 그 시각의 **자리**만 프레임에서 찾는다. 그
    순간에는 두 차가 실제로 붙어 있으므로 최근접 이동 쌍이 곧 충돌 쌍이다.
    정지-정지 쌍은 뺀다 — 주차 차량 박스가 서로 겹쳐 있어 그것이 늘 최근접이 된다.
    """
    if fr is None or len(fr) < 2:
        return None
    n = len(fr)
    sp = np.hypot(fr.vel[:, 0], fr.vel[:, 1])
    corners = [_rect_corners(float(fr.xy[i, 0]), float(fr.xy[i, 1]), float(fr.yaw[i]),
                             float(fr.size[i, 0]), float(fr.size[i, 1]))
               for i in range(n)]
    best, pair = float("inf"), None
    for i in range(n):
        for j in range(i + 1, n):
            if not (sp[i] > STATIONARY_MPS or sp[j] > STATIONARY_MPS):
                continue
            g = poly_gap(corners[i], corners[j])
            if g < best:
                best, pair = g, (i, j)
    if pair is None or best > 2.0:
        return None
    i, j = pair
    return (float((fr.xy[i, 0] + fr.xy[j, 0]) / 2),
            float((fr.xy[i, 1] + fr.xy[j, 1]) / 2))


def make_dataset(frames_list, index, grid: BevGrid, coll_list=None):
    import torch
    from torch.utils.data import Dataset

    class DAMotionDataset(Dataset):
        """표본 하나 = 과거 3프레임 래스터 + 미래 4프레임 흐름/점유 라벨."""

        def __len__(self):
            return len(index)

        def __getitem__(self, i):
            si, c = index[i]
            frames = frames_list[si]
            past = np.stack(
                [rasterize(_to_frame(frames[k]), grid)
                 for k in range(c - N_PAST + 1, c + 1)], axis=0
            )
            cur = _to_frame(frames[c])
            col = (coll_list[si] if coll_list else None) or {}
            ct = col.get("time_s") if col.get("occurred") else None
            t0 = float(frames[c]["t"])
            # 녹화가 충돌에서 끊기므로, 없는 미래 프레임의 사고 자리는 **마지막으로
            # 남아 있는 프레임**의 최근접 이동 쌍에서 가져온다. 그 프레임이 곧
            # 충돌 직전이다.
            last = None
            for k in range(min(N_FUTURE, len(frames) - 1 - c), 0, -1):
                if frames[c + k] is not None:
                    last = _to_frame(frames[c + k])
                    break
            flows, valids, occs, accs, fmask = [], [], [], [], []
            zero_f = np.zeros((2, grid.size, grid.size), dtype=np.float32)
            zero_1 = np.zeros((1, grid.size, grid.size), dtype=np.float32)
            zero_v = np.zeros((grid.size, grid.size), dtype=np.float32)
            for k in range(1, N_FUTURE + 1):
                rec = frames[c + k] if c + k < len(frames) else None
                have = rec is not None
                fmask.append(1.0 if have else 0.0)
                if have:
                    fut = _to_frame(rec)
                    fl, va = flow_label(cur, fut, grid)
                    flows.append(fl); valids.append(va)
                    occs.append(rasterize(fut, grid)[0:1])
                else:
                    fut = None
                    flows.append(zero_f); valids.append(zero_v); occs.append(zero_1)
                # 사고 열지도: 이 미래 프레임의 1초/2Hz 구간에 충돌 시각이 들면
                # 그 자리에 가우시안을 찍는다. 프레임이 있으면 그 프레임 좌표계에서,
                # 없으면 마지막 남은 프레임에서 자리를 찾는다.
                lo, hi = t0 + (k - 1) * FRAME_DT_S, t0 + k * FRAME_DT_S
                hit = ct is not None and lo < ct <= hi
                xy = collision_xy(fut if fut is not None else last) if hit else None
                accs.append(collision_heatmap_label(grid, xy)[None])
            return (
                torch.from_numpy(past),
                torch.from_numpy(np.stack(flows, 0)),
                torch.from_numpy(np.stack(valids, 0)),
                torch.from_numpy(np.stack(occs, 0)),
                torch.from_numpy(np.stack(accs, 0)),
                torch.tensor(fmask, dtype=torch.float32),
            )

    return DAMotionDataset()


def losses(flows, occs, accs, mu, logvar, gt_flow, gt_valid, gt_occ, gt_acc,
           fmask, kl_weight: float, acc_weight: float):
    """흐름 L1(유효 픽셀만) + 점유 BCE + KL.

    흐름 손실을 **유효 픽셀에만** 거는 것이 중요하다. 시야 밖으로 나간 차나 빈
    공간까지 0 으로 맞히게 하면 모델이 "아무도 안 움직인다" 로 수렴한다.

    표본 축은 **평균 표본(마지막)** 으로만 회귀한다 — 원본도 운동 지표는 분포
    평균으로만 잰다. 확률성은 사고 판정에서만 쓰인다.

    사고 열지도는 **focal 가중 BCE** 다. 양성 픽셀이 4만분의 몇이라 평범한 BCE 로는
    전부 0 으로 수렴한다 — CenterPoint 계열 중심 열지도와 같은 문제이고 같은 해법을
    쓴다. 양·음 항을 **각자의 개수로 나눈다**: 둘 다 양성 개수로 나누면 (표본당
    양성이 0 인 경우가 대부분이라) 음성 항이 16만 픽셀 합으로 폭주해 모든 출력을
    0 으로 밀어버린다. 실제로 그렇게 붕괴한 적이 있다.

    `fmask` 는 그 미래 프레임이 실제로 녹화돼 있는지다. 없는 프레임은 **운동 손실
    에서 가린다** — 녹화가 충돌에서 끊기므로 그 프레임을 0 으로 맞히게 하면 안 된다.
    사고 손실은 충돌 정답에서 오므로 가리지 않는다.
    """
    import torch
    import torch.nn.functional as F

    pred_flow = flows[:, -1]  # (B, F, 2, H, W) — 평균 표본
    pred_occ = occs[:, -1]
    fm = fmask.view(fmask.shape[0], fmask.shape[1], 1, 1, 1)
    v = gt_valid.unsqueeze(2) * fm  # (B, F, 1, H, W)
    denom = v.sum().clamp(min=1.0) * 2.0
    l_flow = ((pred_flow - gt_flow).abs() * v).sum() / denom
    occ_px = F.binary_cross_entropy_with_logits(pred_occ, gt_occ, reduction="none")
    l_occ = (occ_px * fm).sum() / (fm.expand_as(occ_px).sum().clamp(min=1.0))
    l_kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()

    pred_acc = accs[:, -1]
    p = torch.sigmoid(pred_acc).clamp(1e-4, 1 - 1e-4)
    pos = (gt_acc > 0.9).float()
    neg = 1.0 - pos
    # CenterNet 변형 focal loss: 양성 근처는 (1-y)^4 로 감쇠시켜 벌을 줄인다
    l_pos = -((1 - p) ** 2 * torch.log(p) * pos).sum() / pos.sum().clamp(min=1.0)
    l_neg = (-((1 - gt_acc) ** 4 * p ** 2 * torch.log(1 - p) * neg).sum()
             / neg.sum().clamp(min=1.0))
    l_acc = l_pos + l_neg

    total = l_flow + l_occ + kl_weight * l_kl + acc_weight * l_acc
    return total, {
        "flow": float(l_flow.detach()),
        "occ": float(l_occ.detach()),
        "acc": float(l_acc.detach()),
        "kl": float(l_kl.detach()),
    }


def train(args) -> int:
    import torch
    from torch.utils.data import DataLoader

    from deepaccident_replicate.da_motion import build_net

    grid = BevGrid()
    _, tr_frames, tr_coll = load_cache(args.train_cache, args.train_collisions)
    tr_index = sample_index(tr_frames, tr_coll)
    n_acc = sum(1 for c in tr_coll if c.get("occurred"))
    print(f"train 표본 {len(tr_index)}개 (시나리오 {len(tr_frames)}, 충돌 {n_acc})")
    va_frames, va_index, va_coll = None, None, None
    if args.val_cache and os.path.exists(args.val_cache):
        _, va_frames, va_coll = load_cache(args.val_cache, args.val_collisions)
        va_index = sample_index(va_frames, va_coll)
        print(f"val   표본 {len(va_index)}개 (시나리오 {len(va_frames)}, "
              f"충돌 {sum(1 for c in va_coll if c.get('occurred'))})")

    tr = DataLoader(make_dataset(tr_frames, tr_index, grid, tr_coll),
                    batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0)
    va = (DataLoader(make_dataset(va_frames, va_index, grid, va_coll),
                     batch_size=args.batch, shuffle=False,
                     num_workers=args.workers)
          if va_index else None)

    dev = torch.device(args.device)
    net = build_net(width=args.width).to(dev)
    print(f"파라미터 {sum(p.numel() for p in net.parameters())/1e6:.2f}M · {dev}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = float("inf")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ep in range(1, args.epochs + 1):
        net.train()
        agg: Dict[str, float] = {}
        n = 0
        t0 = time.time()
        for past, gt_flow, gt_valid, gt_occ, gt_acc, fmask in tr:
            past = past.to(dev, non_blocking=True)
            gt_flow = gt_flow.to(dev, non_blocking=True)
            gt_valid = gt_valid.to(dev, non_blocking=True)
            gt_occ = gt_occ.to(dev, non_blocking=True)
            gt_acc = gt_acc.to(dev, non_blocking=True)
            fmask = fmask.to(dev, non_blocking=True)
            flows, occs, accs, mu, logvar = net(past, n_samples=1)
            loss, parts = losses(flows, occs, accs, mu, logvar, gt_flow, gt_valid,
                                 gt_occ, gt_acc, fmask, args.kl_weight,
                                 args.acc_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            n += 1
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
        sched.step()
        tr_msg = " ".join(f"{k} {v/max(n,1):.4f}" for k, v in agg.items())

        vmsg, score = "", agg.get("flow", 0.0) / max(n, 1)
        if va is not None:
            net.eval()
            vagg: Dict[str, float] = {}
            m = 0
            with torch.no_grad():
                for past, gt_flow, gt_valid, gt_occ, gt_acc, fmask in va:
                    past = past.to(dev); gt_flow = gt_flow.to(dev)
                    gt_valid = gt_valid.to(dev); gt_occ = gt_occ.to(dev)
                    gt_acc = gt_acc.to(dev); fmask = fmask.to(dev)
                    flows, occs, accs, mu, logvar = net(past, n_samples=1)
                    _, parts = losses(flows, occs, accs, mu, logvar, gt_flow,
                                      gt_valid, gt_occ, gt_acc, fmask,
                                      args.kl_weight, args.acc_weight)
                    m += 1
                    for k, v in parts.items():
                        vagg[k] = vagg.get(k, 0.0) + v
            # **사고 헤드 손실로 고른다.** 비교 대상의 산출물은 사고 판정이지
            # 흐름이 아니다. 흐름만 보면 사고를 못 맞히는 체크포인트가 뽑힌다.
            score = vagg.get("acc", 0.0) / max(m, 1)
            vmsg = " | val " + " ".join(f"{k} {v/max(m,1):.4f}"
                                        for k, v in vagg.items())
        print(f"ep {ep:3d}/{args.epochs}  {tr_msg}{vmsg}  "
              f"{time.time()-t0:.0f}s", flush=True)

        if score < best:
            best = score
            torch.save({"state_dict": net.state_dict(), "width": args.width,
                        "grid": {"range_m": grid.range_m, "res_m": grid.res_m},
                        "n_past": N_PAST, "n_future": N_FUTURE,
                        "val_acc_loss": score,
                        "flow_l1_m": (vagg.get("flow", 0.0) / max(m, 1)
                                      if va is not None else None),
                        "epoch": ep}, args.out)
            print(f"    저장 {args.out} (사고 손실 {score:.4f})", flush=True)
    print(f"\n최저 사고 손실: {best:.4f} → {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    x = sub.add_parser("collisions", help="충돌 정답만 뽑는다 (값싸다)")
    x.add_argument("--root", required=True)
    x.add_argument("--split", default="val")
    x.add_argument("--out", required=True)

    c = sub.add_parser("cache", help="시나리오 → 프레임 액터 배열 캐시")
    c.add_argument("--root", required=True)
    c.add_argument("--split", default="train")
    c.add_argument("--out", required=True)
    c.add_argument("--rate", type=float, default=1.0 / FRAME_DT_S)  # 2 Hz
    c.add_argument("--limit", type=int, default=None)

    t = sub.add_parser("train", help="운동 예측 모델 학습")
    t.add_argument("--train-cache", required=True)
    t.add_argument("--val-cache", default=None)
    t.add_argument("--train-collisions", default=None,
                   help="충돌 정답 npz (`collisions` 하위명령). 프레임 캐시에 "
                        "충돌 정보가 없을 때 이름으로 맞춰 붙인다")
    t.add_argument("--val-collisions", default=None)
    t.add_argument("--out", default="out/da_motion/da_motion_best.pt")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--width", type=int, default=64)
    t.add_argument("--kl-weight", type=float, default=1e-3)
    t.add_argument("--acc-weight", type=float, default=1.0,
                   help="사고 열지도 손실 가중")
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--device", default="cuda")

    args = ap.parse_args(argv)
    if args.cmd == "collisions":
        cache_collisions(args.root, args.split, args.out)
        return 0
    if args.cmd == "cache":
        cache_split(args.root, args.split, args.out, args.rate, args.limit)
        return 0
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
