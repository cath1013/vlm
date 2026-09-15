"""경로 예측 모델용 지도학습 데이터 생성.

`prediction.predict()` 가 쓰는 입력(`PredictContext`)과, **시나리오 전체를 보고**
정한 출력 정답을 짝지어 JSONL 로 낸다.

정답을 어떻게 정하는가
    추론 시점에는 미래를 모르지만 데이터셋에는 시나리오 끝까지 다 있다. 그래서
    시각 t 의 액터가 **실제로 어디로 갔는지**를 t 이후 스냅샷에서 읽어, 그것을
    정답으로 쓴다.

      - `target_offsets`   : 실제 미래 궤적을 1초 간격 상대좌표로 (회귀 정답)
      - `best_candidate`   : 실제 궤적에 가장 가까운 후보의 인덱스 (분류 정답)
      - `maneuver`         : 그 후보의 기동 라벨
      - `match_error_m`    : 실제 궤적과 그 후보의 평균 거리 (정답의 신뢰도)

    실제 궤적이 어느 후보와도 멀면(`match_error_m` 이 큼) 액터가 후보 집합에 없는
    행동을 한 것이다 (차선 변경, 노면 이탈, 급정지). 그런 표본은 버리지 않고
    `matched=false` 로 표시해 남긴다 — 후보 집합의 한계를 학습에서 다룰 수 있어야
    하고, 조용히 버리면 데이터가 실제보다 쉬워 보인다.

정답이 하나로 정해지지 않는 표본
    차량이 아직 갈림길에 닿지 않았으면 **여러 후보가 실제 궤적과 똑같이 가깝다.**
    그때 `best_candidate` 는 후보 목록의 순서를 고른 것이다. 그런 표본을
    `tied_best` · `label_ambiguous` · `maneuver_ambiguous` · `margin_m` 으로
    표시한다 (`tie_analysis` 참조). 학습·평가에서 반드시 다뤄야 한다 —
    docs/predict_model_io.md §4.1.

실행
    python examples/make_predict_dataset.py --root <DeepAccident 루트> \\
        --carla-maps ./carla_map --out out/predict_dataset --limit 20

    # 특정 분할·유형만
    python examples/make_predict_dataset.py --root <루트> \\
        --carla-maps ./carla_map --split train \\
        --type type1_subtype2_accident --out out/predict_dataset

    # 전체를 16조각으로 나눠 동시에 (한 프로세스로는 몇 시간 걸린다)
    for i in $(seq 0 15); do
        python examples/make_predict_dataset.py --root <루트> \\
            --carla-maps ./carla_map --out out/predict_dataset --shard $i/16 &
    done; wait
    python examples/make_predict_dataset.py --merge --out out/predict_dataset
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

from traffic_llm.config import PipelineConfig
from traffic_llm.da_runner import DeepAccidentRunner
from traffic_llm.predict_model import (
    N_CANDIDATE_FEATURES,
    N_GLOBAL_FEATURES,
    N_HISTORY_FEATURES,
    N_HISTORY_STEPS,
    N_INTERACTION_FEATURES,
    encode,
)
from traffic_llm.prediction import build_context, candidate_to_path

# 실제 궤적과 후보의 평균 거리가 이 값을 넘으면 "후보 집합 밖의 행동"으로 본다.
# 차로 폭(3.5m)의 절반을 넘으면 다른 차로/다른 경로를 간 것이다.
MATCH_TOLERANCE_M = 3.0


def future_track(
    per_actor: dict, actor_id: str, t0: float, horizon_s: float, dt: float = 1.0
):
    """t0 이후 1초 간격 실제 위치. 관측이 끊기면 거기서 멈춘다.

    Returns: (점 목록, 지평까지 관측됐는가)
    """
    samples = per_actor.get(actor_id, [])
    if not samples:
        return [], False
    out = []
    k = 0.0
    while k <= horizon_s + 1e-6:
        want = t0 + k
        # want 에 가장 가까운 표본. 0.3초 넘게 벌어지면 관측이 끊긴 것이다.
        best = min(samples, key=lambda s: abs(s[0] - want))
        if abs(best[0] - want) > 0.3:
            return out, False
        out.append(best[1])
        k += dt
    return out, True


def polyline_distance(pt, poly) -> float:
    """점에서 폴리라인까지의 최단 거리."""
    if not poly:
        return float("inf")
    if len(poly) == 1:
        return math.dist(pt, poly[0])
    best = float("inf")
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-12:
            d = math.dist(pt, a)
        else:
            t = max(0.0, min(1.0, ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / L2))
            d = math.dist(pt, (ax + t * dx, ay + t * dy))
        best = min(best, d)
    return best


def match_candidate(track, candidates):
    """실제 궤적에 가장 가까운 후보. Returns: (인덱스, 평균거리, 후보별 거리).

    시작점은 두 경로가 같으므로 제외하고, **미래 부분만** 본다. 포함하면 모든
    후보의 거리가 같이 낮아져 구별력이 떨어진다.
    """
    if not track or not candidates:
        return None, float("inf"), []
    fut = track[1:] if len(track) > 1 else track
    errs = []
    for c in candidates:
        ds = [polyline_distance(p, c.polyline) for p in fut]
        errs.append(sum(ds) / len(ds))
    i = min(range(len(errs)), key=lambda j: errs[j])
    return i, errs[i], errs


# 두 후보의 평균거리 차이가 이보다 작으면 **구별할 수 없는 것**으로 본다.
# 좌표는 mm 단위로 반올림해 저장하므로 그보다 크게 잡는다.
TIE_EPS_M = 0.05


def validate_map_coverage(scenarios, carla_maps, allow_trajectory_map=False):
    """학습 후보 지도가 평가 시나리오의 미래 궤적과 독립적인지 확인한다.

    Returns a manifest-ready provenance dictionary. `allow_trajectory_map` 은 기존 산출물
    재현이나 탐색을 위한 명시적 예외이며, 일부 타운만 지도가 있으면 그 사실도 숨기지
    않고 `mixed` 로 기록한다.
    """
    towns = sorted({s.town for s in scenarios})
    available = []
    missing = []
    for town in towns:
        path = os.path.join(carla_maps, f"{town}.xodr") if carla_maps else ""
        (available if path and os.path.isfile(path) else missing).append(town)
    if missing and not allow_trajectory_map:
        shown = ", ".join(missing)
        raise SystemExit(
            "경로예측 학습 데이터에는 미래 궤적과 독립적인 OpenDRIVE 지도가 "
            f"필요합니다. 누락 타운: {shown}. --carla-maps <디렉터리>를 지정하거나, "
            "탐색용 합성을 의도한 경우에만 --allow-trajectory-map을 명시하십시오."
        )
    if not missing:
        source = "opendrive"
    elif not available:
        source = "trajectory_synthesized_from_complete_scenario"
    else:
        source = "mixed_opendrive_and_trajectory_synthesized"
    return {
        "map_source": source,
        "map_directory": os.path.abspath(carla_maps) if carla_maps else None,
        "map_built_from_scenario_trajectories": bool(missing),
        "map_towns": towns,
        "map_missing_towns": missing,
    }


def tie_analysis(errs, candidates):
    """정답 라벨이 실제로 하나로 정해지는지 본다.

    차량이 아직 갈림길에 닿지 않았으면 여러 후보가 실제 궤적과 **똑같이** 가깝다.
    그때 `argmin(errs)` 은 목록 순서를 고르는 것이지 정답을 고르는 것이 아니다.
    이것을 표시하지 않으면 학습은 순서를 외우고, 평가는 그 순서에 우연히 일치하는
    사전확률을 정답으로 세어 정확도가 부풀려진다
    (전체 데이터셋 실측: 임의 표본 75.7% vs 라벨 유일 표본 61.1%).

    Returns: (동률 인덱스 목록, 다음 후보까지의 여유 [m], 기동도 갈리는가)
    """
    if not errs:
        return [], 0.0, False
    lo = min(errs)
    tied = [i for i, e in enumerate(errs) if e <= lo + TIE_EPS_M]
    rest = [e for e in errs if e > lo + TIE_EPS_M]
    margin = (min(rest) - lo) if rest else float("inf")
    mans = {candidates[i].maneuver for i in tied}
    return tied, margin, len(mans) > 1


def build_records(res, cfg, rate_hz: float):
    """한 시나리오 → 표본 목록."""
    snaps = list(res.snapshots(rate_hz=rate_hz))
    if len(snaps) < 2:
        return [], {"snapshots": len(snaps)}

    # 액터별 (시각, 위치) 이력 — 미래를 읽기 위한 색인
    per_actor: dict = {}
    for s in snaps:
        for a in s.actors:
            per_actor.setdefault(a.actor_id, []).append((s.t, a.world_xy))

    horizon = cfg.prediction_horizon_s
    recs = []
    stats = {
        "snapshots": len(snaps),
        "actor_frames": 0,
        "no_candidates": 0,
        "future_incomplete": 0,
        "matched": 0,
        "unmatched": 0,
        "label_ambiguous": 0,
        "maneuver_ambiguous": 0,
        # 과거 궤적이 몇 단계 채워졌는가. 0 이면 방금 생긴 트랙이다.
        "history_steps_total": 0,
        "history_full": 0,
    }
    sid = res.scenario.scenario_id
    for s in snaps:
        for a in s.actors:
            stats["actor_frames"] += 1
            ctx = build_context(
                a, res.network, horizon, t_s=s.t, scenario_id=sid,
                neighbors=s.actors,
            )
            if not ctx.candidates:
                stats["no_candidates"] += 1
                continue
            track, complete = future_track(per_actor, a.actor_id, s.t, horizon)
            if len(track) < 2:
                stats["future_incomplete"] += 1
                continue
            if not complete:
                stats["future_incomplete"] += 1
            bi, err, errs = match_candidate(track, ctx.candidates)
            matched = err <= MATCH_TOLERANCE_M
            stats["matched" if matched else "unmatched"] += 1
            tied, margin, man_amb = tie_analysis(errs, ctx.candidates)
            if matched and len(ctx.candidates) > 1 and len(tied) > 1:
                stats["label_ambiguous"] += 1
                if man_amb:
                    stats["maneuver_ambiguous"] += 1

            enc = encode(ctx)
            n_hist = sum(1 for row in enc["history"] if row[-1] > 0.5)
            stats["history_steps_total"] += n_hist
            if n_hist == N_HISTORY_STEPS:
                stats["history_full"] += 1
            e0, n0 = a.world_xy
            cand_wps = []
            for cand in ctx.candidates:
                pth = candidate_to_path(ctx, cand, 0.0)
                cand_wps.append(
                    [] if pth is None else
                    [[round(w[0] - e0, 3), round(w[1] - n0, 3)]
                     for w in pth.waypoints]
                )
            recs.append(
                {
                    # --- 식별
                    "scenario_id": sid,
                    "dataset_split": res.scenario.split,
                    "town": res.scenario.town,
                    "t_s": round(s.t, 3),
                    "actor_id": a.actor_id,
                    "actor_class": a.cls,
                    # --- 입력 (predict 가 보는 것과 동일)
                    "global": [round(v, 6) for v in enc["global"]],
                    "candidates": [
                        [round(v, 6) for v in row] for row in enc["candidates"]
                    ],
                    # 과거 궤적 5×5 (t-5 … t-1, 현재 위치 기준 상대좌표).
                    # `target_offsets` 와 같은 1초 격자라 과거 5초 → 미래 5초가
                    # 대칭이다. 마지막 칸이 valid.
                    "history": [
                        [round(v, 6) for v in row] for row in enc["history"]
                    ],
                    # target 기준 주변 actor 상태. 마지막 값은 padding 방지 valid다.
                    # 미래 위치·경로는 포함하지 않아 예측 시점의 정보만 쓴다.
                    "interactions": [
                        [round(v, 6) for v in row]
                        for row in enc["interactions"]
                    ],
                    "candidate_meta": [
                        {
                            "maneuver": c.maneuver,
                            "prior": round(c.prior, 6),
                            "to_road": c.to_road,
                            "n_vertices": len(c.polyline),
                        }
                        for c in ctx.candidates
                    ],
                    # 규칙 기반이 실제로 내보내는 웨이포인트 (상대좌표, 1초 간격).
                    # `target_offsets` 와 **같은 시각 격자**이므로 index k 끼리 바로
                    # 비교해 ADE/FDE 를 낼 수 있다. 이것이 없으면 좌표를 직접 내는
                    # 모델(waypoints 모드)과 규칙 기반을 비교할 수 없다.
                    "candidate_waypoints": cand_wps,
                    # --- 정답 (시나리오 전체를 보고 정함)
                    "best_candidate": bi,
                    "maneuver": ctx.candidates[bi].maneuver,
                    "match_error_m": round(err, 3),
                    "candidate_errors_m": [round(v, 3) for v in errs],
                    "matched": matched,
                    "future_complete": complete,
                    # --- 정답의 신뢰도. 학습·평가에서 반드시 봐야 한다.
                    # `tied_best` 가 2개 이상이면 `best_candidate` 는 목록 순서로
                    # 고른 것이다. 단일 라벨 학습에서는 제외하거나, 집합 라벨
                    # (tied_best 균등분포)로 쓰는 것이 맞다.
                    "tied_best": tied,
                    "label_ambiguous": len(tied) > 1,
                    # 동률 후보들의 기동이 갈리면 기동 라벨조차 임의다.
                    "maneuver_ambiguous": man_amb,
                    # 정답 후보와 그다음 후보의 평균거리 차이. 클수록 분명하다.
                    "margin_m": (None if margin == float("inf")
                                 else round(margin, 3)),
                    # 상대 좌표로 낸다 — 절대 좌표를 회귀하면 지역 좌표계의
                    # 원점 위치를 외우게 된다.
                    "target_offsets": [
                        [round(p[0] - e0, 3), round(p[1] - n0, 3)] for p in track
                    ],
                    "origin_enu": [round(e0, 3), round(n0, 3)],
                }
            )
    return recs, stats


def merge_shards(out_dir: str) -> int:
    """`--shard` 로 나눠 만든 조각들을 `samples.jsonl` · `manifest.json` 으로 합친다.

    조각 파일은 지우지 않는다 — 일부 조각만 다시 돌려 합칠 수 있어야 한다.
    """
    import glob

    parts = sorted(glob.glob(os.path.join(out_dir, "samples.*of*.jsonl")))
    if not parts:
        print(f"합칠 조각이 없습니다: {out_dir}/samples.*of*.jsonl", file=sys.stderr)
        return 1
    expected = None
    total = 0
    dst = os.path.join(out_dir, "samples.jsonl")
    with open(dst, "w", encoding="utf-8") as w:
        for p in parts:
            with open(p, encoding="utf-8") as r:
                for line in r:
                    if line.strip():
                        w.write(line)
                        total += 1

    agg: dict = {}
    man: dict = {}
    by_man: dict = {}
    n_scen = n_failed = 0
    for p in parts:
        mp = p.replace("samples.", "manifest.").replace(".jsonl", ".json")
        if not os.path.exists(mp):
            print(f"경고: {os.path.basename(mp)} 없음 — 그 조각은 아직 끝나지 "
                  f"않았을 수 있습니다", file=sys.stderr)
            continue
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        man = man or m
        n_scen += m["n_scenarios"]
        n_failed += m["n_failed"]
        for k, v in (m.get("counts") or {}).items():
            agg[k] = agg.get(k, 0) + v
        for k, v in (m.get("maneuver_distribution") or {}).items():
            by_man[k] = by_man.get(k, 0) + v
        # 조각별 설정이 다르면 합친 결과의 의미가 없다.
        cfg_now = dict(m.get("config") or {})
        cfg_now.pop("shard", None)
        if expected is None:
            expected = cfg_now
        elif cfg_now != expected:
            print(f"경고: {os.path.basename(mp)} 의 설정이 다릅니다 — 합친 결과를 "
                  f"믿을 수 없습니다", file=sys.stderr)

    man["n_records"] = total
    man["n_scenarios"] = n_scen
    man["n_failed"] = n_failed
    man["counts"] = agg
    man["maneuver_distribution"] = dict(sorted(by_man.items(), key=lambda kv: -kv[1]))
    (man.setdefault("config", {}))["shard"] = f"merged from {len(parts)} shards"
    lq = man.setdefault("label_quality", {})
    lq["label_ambiguous"] = agg.get("label_ambiguous", 0)
    lq["maneuver_ambiguous"] = agg.get("maneuver_ambiguous", 0)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)

    print(f"조각 {len(parts)}개 → 표본 {total:,} · 시나리오 {n_scen}"
          + (f" (실패 {n_failed})" if n_failed else ""))
    m = agg.get("matched", 0)
    if m:
        a = agg.get("label_ambiguous", 0)
        print(f"  정답 라벨 임의 {a:,} ({a/m:.1%}) · "
              f"그중 기동도 임의 {agg.get('maneuver_ambiguous',0):,}")
    print(f"  {dst}\n  {os.path.join(out_dir,'manifest.json')}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="경로 예측 모델용 지도학습 데이터 생성"
    )
    if argv is None:
        argv = sys.argv[1:]
    if "--merge" in argv:
        ap2 = argparse.ArgumentParser()
        ap2.add_argument("--merge", action="store_true")
        ap2.add_argument("--out", default="out/predict_dataset")
        a2, _ = ap2.parse_known_args(argv)
        return merge_shards(a2.out)
    ap.add_argument("--merge", action="store_true",
                    help="--shard 로 만든 조각들을 samples.jsonl 로 합친다 "
                         "(--out 만 함께 준다)")
    ap.add_argument("--root", required=True, help="DeepAccident 루트")
    ap.add_argument("--carla-maps", default=None, help="OpenDrive(.xodr) 디렉터리")
    ap.add_argument(
        "--allow-trajectory-map", action="store_true",
        help="탐색용으로만 전체 시나리오 궤적에서 지도를 합성하도록 허용한다. "
             "기본적으로 학습 데이터 생성에는 모든 타운의 OpenDRIVE가 필수다",
    )
    ap.add_argument("--out", default="out/predict_dataset", help="출력 디렉터리")
    ap.add_argument("--split", default=None,
                    help="데이터셋 분할만 (DeepAccident_mini / train / val)")
    ap.add_argument("--type", default=None, help="시나리오 유형")
    ap.add_argument("--town", default=None, help="타운")
    ap.add_argument("--limit", type=int, default=0, help="시나리오 수 상한")
    ap.add_argument("--rate", type=float, default=2.0, help="스냅샷 주기 [Hz]")
    ap.add_argument("--mode", choices=["sensor3d", "camera"], default="sensor3d")
    ap.add_argument(
        "--shard", default=None, metavar="I/N",
        help="시나리오를 N 조각으로 나눠 I 번째만 처리한다 (0-based). "
             "여러 프로세스를 동시에 띄워 나눠 처리할 때 쓴다 — 출력 파일 이름에 "
             "조각 번호가 붙으므로 서로 덮어쓰지 않는다. 시나리오 목록은 정렬돼 "
             "있어 조각 분배가 실행마다 같다.",
    )
    args = ap.parse_args(argv)

    cfg = PipelineConfig()
    cfg.deepaccident.observation_mode = args.mode
    runner = DeepAccidentRunner(args.root, cfg)

    scenarios = runner.list_scenarios(
        scenario_types=[args.type] if args.type else None,
        towns=[args.town] if args.town else None,
    )
    if args.split:
        scenarios = [s for s in scenarios if s.split == args.split]
    if not scenarios:
        raise SystemExit("조건에 맞는 시나리오가 없습니다")
    if args.limit:
        scenarios = scenarios[: args.limit]

    shard = None
    if args.shard:
        try:
            si, sn = (int(x) for x in args.shard.split("/"))
        except ValueError:
            raise SystemExit("--shard 는 I/N 형식이어야 합니다 (예: 0/8)")
        if not (0 <= si < sn):
            raise SystemExit(f"--shard 범위 오류: 0 <= {si} < {sn} 이어야 합니다")
        shard = (si, sn)
        # 라운드로빈으로 나눈다 — 앞뒤를 잘라 나누면 타운·유형이 조각마다 쏠려
        # 조각별 처리 시간이 크게 달라진다.
        scenarios = scenarios[si::sn]
        if not scenarios:
            raise SystemExit(f"조각 {si}/{sn} 에 시나리오가 없습니다")

    map_provenance = validate_map_coverage(
        scenarios, args.carla_maps, args.allow_trajectory_map
    )

    os.makedirs(args.out, exist_ok=True)
    suffix = f".{shard[0]:03d}of{shard[1]:03d}" if shard else ""
    jsonl = os.path.join(args.out, f"samples{suffix}.jsonl")
    total = {"records": 0, "scenarios": 0, "failed": 0}
    agg = {}
    by_maneuver = {}
    print(f"시나리오 {len(scenarios)}개 → {jsonl}")

    with open(jsonl, "w", encoding="utf-8") as f:
        for i, sc in enumerate(scenarios, 1):
            xodr = None
            if args.carla_maps:
                p = os.path.join(args.carla_maps, f"{sc.town}.xodr")
                xodr = p if os.path.isfile(p) else None
            try:
                res = runner.build(
                    sc.scenario, sc.scenario_type, opendrive_path=xodr
                )
                recs, st = build_records(res, cfg, args.rate)
            except Exception as e:  # 한 시나리오 실패가 전체를 멈추게 하지 않는다
                total["failed"] += 1
                print(f"  [{i}/{len(scenarios)}] {sc.scenario} 실패: "
                      f"{type(e).__name__}: {e}")
                continue
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                if r["matched"]:
                    by_maneuver[r["maneuver"]] = by_maneuver.get(r["maneuver"], 0) + 1
            total["records"] += len(recs)
            total["scenarios"] += 1
            for k, v in st.items():
                agg[k] = agg.get(k, 0) + v
            if i % 10 == 0 or i == len(scenarios):
                print(f"  [{i}/{len(scenarios)}] 누적 표본 {total['records']:,}")

    manifest = {
        "n_records": total["records"],
        "n_scenarios": total["scenarios"],
        "n_failed": total["failed"],
        "config": {
            "prediction_horizon_s": cfg.prediction_horizon_s,
            "snapshot_rate_hz": args.rate,
            "observation_mode": args.mode,
            "match_tolerance_m": MATCH_TOLERANCE_M,
            "root": os.path.abspath(args.root),
            "split": args.split,
            "scenario_type": args.type,
            "shard": (f"{shard[0]}/{shard[1]}" if shard else None),
            **map_provenance,
        },
        "feature_spec": {
            "n_global": N_GLOBAL_FEATURES,
            "n_candidate": N_CANDIDATE_FEATURES,
            "n_history_steps": N_HISTORY_STEPS,
            "n_history_features": N_HISTORY_FEATURES,
            "n_interaction_features": N_INTERACTION_FEATURES,
            "doc": "docs/predict_model_io.md",
        },
        "counts": agg,
        "label_quality": {
            "tie_eps_m": TIE_EPS_M,
            "label_ambiguous": agg.get("label_ambiguous", 0),
            "maneuver_ambiguous": agg.get("maneuver_ambiguous", 0),
            "note": "label_ambiguous 표본의 best_candidate 는 목록 순서로 고른 "
                    "것이다. 단일 라벨 학습·평가에서 제외하거나 tied_best 를 "
                    "집합 라벨로 써야 한다 — docs/predict_model_io.md §5",
        },
        "maneuver_distribution": dict(
            sorted(by_maneuver.items(), key=lambda kv: -kv[1])
        ),
    }
    mpath = os.path.join(args.out, f"manifest{suffix}.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    fail_note = f" (실패 {total['failed']})" if total["failed"] else ""
    print(f"\n표본 {total['records']:,}개 · 시나리오 {total['scenarios']}개{fail_note}")
    print(f"  후보 없음 {agg.get('no_candidates',0):,} · "
          f"미래 불충분 {agg.get('future_incomplete',0):,}")
    print(f"  후보 일치 {agg.get('matched',0):,} · "
          f"불일치 {agg.get('unmatched',0):,} "
          f"(불일치는 matched=false 로 남긴다)")
    amb = agg.get("label_ambiguous", 0)
    m = agg.get("matched", 0)
    if m:
        print(f"  정답 라벨 임의 {amb:,} ({amb/m:.1%}) · "
              f"그중 기동도 임의 {agg.get('maneuver_ambiguous',0):,} "
              f"— 차량이 아직 갈림길에 닿지 않아 후보를 구별할 수 없는 표본")
    print(f"  기동 분포: {manifest['maneuver_distribution']}")
    print(f"  {jsonl}\n  {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
