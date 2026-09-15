"""학습된 DeepAccident 재구현으로 사고 예측을 평가한다.

`train_da_motion.py` 가 만든 운동 예측 모델을 돌려 미래 인스턴스 위치를 얻고,
이식된 사고 판정 규칙(`deepaccident_replicate/da_baseline.py`)을 그대로 씌운다. 결과는
LLM 응답과 **같은 채점기**(`score_modes`)를 통과하므로 같은 표에서 읽힌다.

    .venv/bin/python deepaccident_replicate/eval_da_motion.py \
        --root /home/sryu/inclab-nas/DeepAccident --split val \
        --model out/da_motion/da_motion_best.pt \
        --out out/da_motion/eval_val.json

**같은 창에서 등속 외삽 기준선도 함께 낸다.** 학습된 운동 예측이 단순 외삽보다
무엇을 더하는지가 이 비교의 핵심이고, 그것은 같은 창에서 재야만 의미가 있다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_llm.accident_qa import (
    MODES,
    WindowConfig,
    aggregate_modes,
    build_windows,
    score_modes,
    window_ground_truth,
)
from traffic_llm.config import PipelineConfig
from deepaccident_replicate.da_baseline import DeepAccidentRule
from deepaccident_replicate.da_baseline import build_response as da_response
from deepaccident_replicate.da_baseline import saturation as da_saturation
from deepaccident_replicate.da_motion import (
    FRAME_DT_S,
    head_response,
    read_accident,
    N_FUTURE,
    N_PAST,
    BevGrid,
    ego_frame,
    frames_to_buckets,
    rasterize,
    read_tracks,
    track_gaps,
)
from traffic_llm.da_runner import DeepAccidentRunner
from traffic_llm.deepaccident import CLASS_SIZES, estimate_collision

# 등속 외삽 기준선은 `examples/` 의 스크립트에 있다 — 같은 창에서 재야 학습된 운동
# 예측이 무엇을 더하는지 격리된다.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))
from baselines_accident_qa import bucket_min_gaps  # noqa: E402


def load_model(path: str, device: str):
    import torch

    from deepaccident_replicate.da_motion import build_net

    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net(width=ck.get("width", 64))
    net.load_state_dict(ck["state_dict"])
    net.eval().to(device)
    return net, ck


def model_outputs(net, w, grid: BevGrid, device: str, n_samples: int,
                  exclude_touching: bool, exclude_static: bool):
    """창 하나 → 모델이 본 구간별 최소 간격.

    과거 `N_PAST` 프레임을 래스터화해 모델에 넣고, 표본마다 흐름을 액터 궤적으로
    읽어 인스턴스 사이 거리를 잰다. 표본 축은 **최솟값**으로 접는다 — 원본이
    "표본 아무거나 사고를 가리키면 사고" 규칙을 쓰기 때문이다.

    돌려주는 것은 `(gaps_fn, acc_logits, cur)` 이다. `gaps_fn(n_buckets)` 이 후처리
    규칙용 구간별 간격을, `acc_logits` 가 사고 헤드 출력을 준다.

    관측 프레임이 모자라거나 ego 를 못 찾으면 None 을 돌려준다 (그 창은 건너뛴다).
    """
    import torch

    ws = w.snapshots
    if len(ws) < N_PAST:
        return None
    frames = [ego_frame(s, grid, CLASS_SIZES) for s in ws[-N_PAST:]]
    if any(f is None for f in frames):
        return None
    cur = frames[-1]
    past = np.stack([rasterize(f, grid) for f in frames], axis=0)
    x = torch.from_numpy(past).unsqueeze(0).to(device)
    with torch.no_grad():
        flows, _, accs, _, _ = net(x, n_samples=n_samples)
    flows = flows[0].cpu().numpy()  # (S, F, 2, H, W)
    acc_logits = accs[0, :, :, 0].cpu().numpy()  # (S, F, H, W)

    per_sample = []
    for s in range(flows.shape[0]):
        tracks = read_tracks(flows[s], cur, grid)
        per_sample.append(track_gaps(tracks, cur, exclude_touching, exclude_static))

    def gaps_fn(n_buckets: int):
        return frames_to_buckets(per_sample, n_buckets, FRAME_DT_S)

    return gaps_fn, acc_logits, cur


def threshold_sweep(rows, base: DeepAccidentRule, grid) -> List[dict]:
    """접촉 임계값만 바꿔 다시 채점한다. 모델은 이미 돌았으므로 값싸다.

    **바이너리 축에서 본다** — 그 축만 DeepAccident 와 판정 단위가 같다(창당 1건).
    """
    from traffic_llm.accident_qa import score_binary

    out: List[dict] = []
    for t in grid:
        rule = DeepAccidentRule(dist_threshold_m=t, horizon_s=base.horizon_s,
                                frame_rule=base.frame_rule)
        tp = fp = tn = fn = 0
        fired = n = 0
        for t_e, gt, gaps in rows:
            n_b = len(gt["expected"])
            b = score_binary(da_response(t_e, n_b, gaps, rule), gt)
            if not b["scorable"]:
                continue
            n += 1
            fired += 1 if b["pred"] else 0
            c = b["counts"]
            tp += c["TP"]; fp += c["FP"]; tn += c["TN"]; fn += c["FN"]
        total = tp + fp + tn + fn
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        out.append({
            "threshold_m": t,
            "fire_rate": round(fired / n, 3) if n else None,
            "counts": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            "accuracy": round((tp + tn) / total, 3) if total else None,
            "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
            "recall": round(rec, 3),
            "balanced": round((rec + spec) / 2.0, 3),
        })
    return out


def head_threshold_sweep(rows, grid: BevGrid, grid_t) -> List[dict]:
    """사고 헤드 판정 임계값 스윕. 열지도는 이미 계산됐으므로 값싸다."""
    from traffic_llm.accident_qa import score_binary

    out: List[dict] = []
    for t in grid_t:
        tp = fp = tn = fn = 0
        fired = n = 0
        for t_e, gt, acc_logits, cur in rows:
            n_b = len(gt["expected"])
            b = score_binary(
                head_response(t_e, n_b, acc_logits, grid, cur, t), gt)
            if not b["scorable"]:
                continue
            n += 1
            fired += 1 if b["pred"] else 0
            c = b["counts"]
            tp += c["TP"]; fp += c["FP"]; tn += c["TN"]; fn += c["FN"]
        total = tp + fp + tn + fn
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        out.append({
            "threshold": t,
            "fire_rate": round(fired / n, 3) if n else None,
            "counts": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            "accuracy": round((tp + tn) / total, 3) if total else None,
            "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
            "recall": round(rec, 3),
            "balanced": round((rec + spec) / 2.0, 3),
        })
    return out


def run(args) -> int:
    cfg = PipelineConfig()
    cfg.deepaccident.observation_mode = args.mode
    wcfg = WindowConfig(window_s=args.window, stride_s=args.stride,
                        horizon_s=args.horizon)
    grid = BevGrid()
    net, ck = load_model(args.model, args.device)
    print(f"모델 {args.model}  (epoch {ck.get('epoch')}, "
          f"흐름 L1 {ck.get('flow_l1_m'):.3f} m)")
    print(f"관측 {N_PAST}프레임@{1/FRAME_DT_S:g}Hz → 미래 {N_FUTURE}프레임 "
          f"({N_FUTURE*FRAME_DT_S:g}초) · 표본 {args.samples}개")

    runner = DeepAccidentRunner(args.root, cfg)
    scs = [s for s in runner.list_scenarios() if s.split == args.split]
    if args.limit:
        scs = scs[: args.limit]
    n_acc = sum(1 for x in scs if x.meta.collision_occurred)
    print(f"시나리오 {len(scs)}개 (충돌 {n_acc} / 무충돌 {len(scs)-n_acc})", flush=True)

    rule = DeepAccidentRule(
        dist_threshold_m=args.da_threshold,
        horizon_s=min(args.horizon, N_FUTURE * FRAME_DT_S),
        frame_rule=args.frame_rule,
    )
    print(f"판정 규칙 {rule.label()}", flush=True)

    scored: Dict[str, List[dict]] = {"da_net": [], "da_cv": [], "da_head": []}
    net_gaps_all: List[list] = []
    head_rows: List[tuple] = []
    # 임계값 스윕용. 모델은 한 번만 돌리고 임계값만 바꿔 다시 채점한다.
    rows: List[tuple] = []
    n_win = n_skip = 0
    t0 = time.time()
    for i, sc in enumerate(scs, 1):
        try:
            res = runner.build(sc.scenario, sc.scenario_type)
            snaps = list(res.snapshots(rate_hz=1.0 / FRAME_DT_S))
        except Exception as e:
            print(f"  [{i}/{len(scs)}] {sc.scenario} 실패: {type(e).__name__}: {e}",
                  flush=True)
            continue
        if len(snaps) < 2:
            continue
        collision = estimate_collision(sc, cfg.deepaccident)
        ct = collision.time_s if (collision and collision.occurred) else None
        wins, _ = build_windows(snaps, wcfg, collision_time_s=ct)
        agent_ids = {ag: sc.meta.agent_id_of(ag) for ag in sc.agents}
        for w in wins:
            gt = window_ground_truth(
                w, wcfg, collision=collision, agent_carla_ids=agent_ids,
                scenario_id=sc.scenario_id, scenario_split=sc.scenario_type,
                dataset_split=sc.split, town=sc.town, data_end_s=snaps[-1].t,
            )
            n_b = len(gt.get("expected") or [])
            if n_b == 0:
                continue
            out = model_outputs(net, w, grid, args.device, args.samples,
                                not args.keep_touching, not args.keep_static)
            if out is None:
                n_skip += 1
                continue
            gaps_fn, acc_logits, cur = out
            g_net = gaps_fn(n_b)
            n_win += 1
            net_gaps_all.append(g_net)
            t_e = w.snapshots[-1].t
            rows.append((t_e, gt, g_net))
            scored["da_net"].append(
                score_modes(da_response(t_e, n_b, g_net, rule), gt))
            # 사고 헤드 — 모든 쌍의 최소 간격 대신 **모델이 지목한 자리**를 쓴다
            scored["da_head"].append(
                score_modes(head_response(t_e, n_b, acc_logits, grid, cur,
                                          args.head_threshold), gt))
            head_rows.append((t_e, gt, acc_logits, cur))
            # 같은 창의 등속 외삽 — 학습이 무엇을 더하는지 격리한다.
            # 겹침 필터도 **같은 설정**이어야 두 줄이 비교 가능하다.
            g_cv = bucket_min_gaps(w.snapshots[-1], n_b, "cv",
                                   exclude_touching_now=not args.keep_touching,
                                   exclude_static_pairs=not args.keep_static)
            scored["da_cv"].append(
                score_modes(da_response(t_e, n_b, g_cv, rule), gt))
        if i % 10 == 0 or i == len(scs):
            print(f"  [{i}/{len(scs)}] 창 {n_win}개 (건너뜀 {n_skip})  "
                  f"{time.time()-t0:.0f}s", flush=True)

    aggs = {k: aggregate_modes(v) for k, v in scored.items() if v}
    sat = da_saturation(net_gaps_all, rule)

    print(f"\n=== {args.split} · 학습된 DeepAccident 재구현 ===")
    print(f"{'':10s} {'B정확':>6s} {'B정밀':>6s} {'B재현':>6s} {'B칸수':>6s} "
          f"{'E검출':>6s} {'W점수':>6s} {'W오경':>6s}")
    for name in ("da_head", "da_net", "da_cv"):
        a = aggs.get(name)
        if not a:
            continue
        b, w_, e = a["binary"], a["weighted"], a["early"]["events"]
        f = lambda v: "     -" if v is None else f"{v:6.3f}"  # noqa: E731
        print(f"{name:10s} {f(b['accuracy'])} {f(b['precision'])} {f(b['recall'])} "
              f"{f(b['mean_positive_buckets'])} {f(e['detection_rate'])} "
              f"{f(w_['mean_score'])} {f(w_['false_alarm_rate'])}")
    print(f"\n포화 진단(da_net): 발화율 {sat['fire_rate']}  "
          f"차체겹침 {sat['touching_rate']}  "
          + ("사실상 상수" if sat["degenerate"] else "변별력 있음"))

    head_sweep = head_threshold_sweep(head_rows, grid,
                                      [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    print(f"\n=== 사고 헤드 판정 임계 스윕 (바이너리 축) ===")
    print(f"{'임계':>6s} {'발화율':>7s} {'정확도':>7s} {'정밀도':>7s} "
          f"{'재현율':>7s} {'균형정확':>8s}")
    for r in head_sweep:
        f = lambda v: "      -" if v is None else f"{v:7.3f}"  # noqa: E731
        print(f"{r['threshold']:6.2f} {r['fire_rate']:7.3f} {f(r['accuracy'])} "
              f"{f(r['precision'])} {f(r['recall'])} {r['balanced']:8.3f}")
    hbest = max(head_sweep, key=lambda r: r["balanced"])
    print(f"  → 균형정확도 최대: {hbest['threshold']:.2f} "
          f"({hbest['balanced']:.3f}, 발화율 {hbest['fire_rate']:.3f})")

    sweep = threshold_sweep(rows, rule, args.sweep_grid)
    print(f"\n=== 접촉 임계값 스윕 (da_net, 바이너리 축) ===")
    print("  저자들의 2.5 m 는 모델 오차를 흡수하려고 느슨하게 잡은 값이다. "
          "모든 쌍에 적용하면")
    print("  정상 주행이 전부 걸린다 — 실측으로 기록된 미래에서도 89 % 가 2.5 m 안에 든다.")
    print(f"{'임계[m]':>8s} {'발화율':>7s} {'정확도':>7s} {'정밀도':>7s} "
          f"{'재현율':>7s} {'균형정확':>8s}")
    for r in sweep:
        f = lambda v: "      -" if v is None else f"{v:7.3f}"  # noqa: E731
        print(f"{r['threshold_m']:8.2f} {r['fire_rate']:7.3f} {f(r['accuracy'])} "
              f"{f(r['precision'])} {f(r['recall'])} {r['balanced']:8.3f}")
    best = max(sweep, key=lambda r: r["balanced"])
    print(f"  → 균형정확도 최대: {best['threshold_m']:.2f} m "
          f"({best['balanced']:.3f}, 발화율 {best['fire_rate']:.3f})")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({
                "split": args.split, "model": args.model,
                "rule": rule.label(), "n_windows": n_win, "n_skipped": n_skip,
                "samples": args.samples,
                "exclude_touching_now": not args.keep_touching,
                "saturation": sat, "aggregate": aggs, "threshold_sweep": sweep,
                "head_threshold_sweep": head_sweep,
            }, fh, ensure_ascii=False, indent=1)
        print(f"\n저장: {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", default="sensor3d", choices=("sensor3d", "camera"))
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--stride", type=float, default=1.0)
    ap.add_argument("--horizon", type=float, default=5.0)
    ap.add_argument("--samples", type=int, default=6,
                    help="운동 표본 수. 원본은 5 + 분포평균 = 6")
    ap.add_argument("--da-threshold", type=float, default=2.5,
                    help="사고 접촉 임계 [m]. 원본 코드 값은 2.5 (5px × 0.5 m/px)")
    ap.add_argument("--frame-rule", default="any", choices=("any", "last"))
    ap.add_argument("--head-threshold", type=float, default=0.5,
                    help="사고 열지도 판정 임계 (시그모이드 확률)")
    ap.add_argument("--keep-static", action="store_true",
                    help="둘 다 정지한 쌍도 후보로 둔다. 주차 차량 두 대는 사고를 "
                         "낼 수 없으므로 기본은 제외다")
    ap.add_argument("--keep-touching", action="store_true",
                    help="관측 시점에 이미 붙어 있는 쌍도 후보로 둔다. "
                         "DeepAccident 라벨의 주차 차량 박스가 서로 겹쳐 있어 "
                         "기본은 제외다")
    ap.add_argument("--sweep-grid", default="0.0,0.1,0.25,0.5,1.0,1.5,2.0,2.5,3.5,5.0",
                    help="접촉 임계값 스윕 격자 [m]. 모델은 한 번만 돌린다")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    args.sweep_grid = [float(x) for x in args.sweep_grid.split(",") if x.strip()]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
