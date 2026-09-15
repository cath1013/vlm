"""DeepAccident 시나리오 → LLM 입력 변환 및 정확도 검증.

    # 시나리오 목록
    python examples/run_deepaccident.py --root <DeepAccident_mini> --list

    # 한 시나리오를 LLM 입력으로 변환
    python examples/run_deepaccident.py --root <루트> \
        --scenario Town03 --type type1_subtype1_accident \
        --out out/da_scenes.jsonl --text-out out/da_scenes.txt --print

    # 정답 대비 정확도 평가 (두 관측 모드 비교)
    python examples/run_deepaccident.py --root <루트> --evaluate

    # CARLA 지도 사용 (있으면 궤적 합성보다 정확)
    python examples/run_deepaccident.py --root <루트> --scenario Town03 \
        --carla-maps "C:/CARLA/CarlaUE4/Content/Carla/Maps/OpenDrive"

    # 시간에 따른 BEV 변화를 지도 위 이미지 집합으로 (시나리오이름_타임스탬프.jpg)
    python examples/run_deepaccident.py --root <루트> --scenario Town03 \
        --type type1_subtype1_accident --carla-maps <지도디렉터리> \
        --bev-dir out/bev
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

from traffic_llm.accident_qa import WindowConfig, write_window_set
from traffic_llm.bev_render import BevConfig, BevRenderer, write_index_html
from traffic_llm.carla_map import find_xodr, town_description
from traffic_llm.config import PipelineConfig
from traffic_llm.da_eval import aggregate, evaluate_build
from traffic_llm.da_runner import DeepAccidentRunner
from traffic_llm.deepaccident import estimate_collision
from traffic_llm.serialize import (
    build_messages,
    to_evaluation_record,
    to_json,
    to_text,
)


def cmd_list(runner: DeepAccidentRunner) -> int:
    scs = runner.list_scenarios()
    print(f"시나리오 {len(scs)}개\n")
    print(f"{'분할':26s} {'시나리오':46s} {'프레임':>6s} {'관측자':>6s} "
          f"{'충돌':>5s} {'도로형태':16s} 기상")
    for s in scs:
        print(f"{s.scenario_type:26s} {s.scenario:46s} {len(s.frames()):6d} "
              f"{len(s.agents):6d} {'O' if s.meta.collision_occurred else '-':>5s} "
              f"{s.meta.road_type:16s} {s.meta.weather}")
    towns = sorted({s.town for s in scs})
    print("\n등장 CARLA 타운:")
    for t in towns:
        print(f"  {t:10s} {town_description(t)}")
    return 0


def cmd_evaluate(root: str, rate: float) -> int:
    picks = [
        ("Town01", "type1_subtype1_normal"),
        ("Town03", "type1_subtype1_normal"),
        ("Town05", "type1_subtype1_normal"),
        ("Town04", "type1_subtype2_normal"),
    ]
    for mode in ("sensor3d", "camera"):
        ms = []
        for town, st in picks:
            cfg = PipelineConfig()
            cfg.deepaccident.observation_mode = mode
            r = DeepAccidentRunner(root, cfg)
            try:
                res = r.build(town, st)
            except KeyError as e:
                print(f"  건너뜀 {town}/{st}: {e}")
                continue
            ms.append(evaluate_build(res, cfg, rate_hz=rate))
        if not ms:
            continue
        print("=" * 72)
        print(aggregate(ms).report())
    print("=" * 72)
    print(
        "sensor3d = GT 3D 위치를 그대로 사용 (좌표변환·융합·지도매칭 검증용)\n"
        "camera   = GT 3D 박스를 카메라에 투영한 2D bbox 만 사용 (실제 배포 경로)"
    )
    return 0


def out_subdir(
    base: str, sc, by_outcome: bool, flat: bool, label: str = ""
) -> str:
    """산출물 경로에 **출처**를 새긴다 → `<base>/<분할>[/<accident|normal>][/<label>]`.

    한 폴더에 여러 분할·트레이스의 결과가 섞이면 나중에 어느 데이터로 만든
    것인지 알 수 없다. 분할 상위 디렉터리를 루트로 주면(607개) 특히 그렇다.

    분할을 **맨 앞**에 두는 이유: 같은 데이터에서 나온 결과가 한자리에 모인다.
    `--out-label` 로 그 아래 구분(언어·provider 등)을 더할 수 있고,
    `--flat-out` 으로 전체를 끌 수 있다.
    """
    if flat:
        return os.path.join(base, label) if label else base
    parts = [base, sc.split]
    if by_outcome:
        parts.append(sc.outcome)
    if label:
        parts.append(label)
    return os.path.join(*parts)


# 기본 경로 예측기. **좌표 회귀 모델을 기본으로 쓴다** — 같은 시험셋에서 규칙 기반보다
# ADE 가 24% 낮고(3.05m 대 4.01m) 후보 집합의 상한도 넘는다 (docs/waypointnet.md).
# 파일이 없거나 torch 가 없으면 규칙 기반으로 조용히 내려간다 — 이 파이프라인은
# torch 없이도 동작해야 한다.
DEFAULT_PREDICTOR = os.path.join("out", "predict_model", "waypointnet_best.pt")


def infer_predictor_mode(path: str) -> str:
    """모델 파일에서 rank/waypoints 를 알아낸다.

    사용자가 모델과 모드를 손으로 짝지으면 어긋난다 — `ranknet_best.pt` 에
    `mode=waypoints` 를 주면 좌표 대신 후보 점수를 좌표로 해석해 조용히 엉뚱한
    궤적이 나간다. 클래스 이름으로 판정하는 것이 안전하다.
    """
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    name = type(obj).__name__
    if "Waypoint" in name:
        return "waypoints"
    if "Rank" in name:
        return "rank"
    raise SystemExit(
        f"{path} 의 모드를 알 수 없습니다 (클래스 {name}). "
        f"--predictor-mode 로 직접 지정하십시오."
    )


def resolve_predictor(args: argparse.Namespace):
    """(모델 경로 또는 None, 모드, 이유 문자열).

    기본은 `DEFAULT_PREDICTOR` 다. 없으면 규칙 기반으로 내려가고 **왜 그랬는지
    출력한다** — 조용히 다른 예측기를 쓰면 payload 가 달라진 이유를 알 수 없다.
    """
    if args.no_predictor:
        return (None, "", "--no-predictor 지정")
    path = args.predictor or DEFAULT_PREDICTOR
    explicit = bool(args.predictor)
    if not os.path.isfile(path):
        if explicit:
            raise SystemExit(f"예측기 파일이 없습니다: {path}")
        return (None, "", f"기본 모델 없음 ({path})")
    try:
        import torch  # noqa: F401
    except ImportError:
        if explicit:
            raise SystemExit("--predictor 를 쓰려면 torch 가 필요합니다")
        return (None, "", "torch 미설치")
    mode = args.predictor_mode or infer_predictor_mode(path)
    return (path, mode, "지정" if explicit else "기본값")


def enforce_experiment_map_policy(
    args: argparse.Namespace, opendrive_path: str | None
) -> None:
    """평가 산출물이 미래 궤적으로 만든 지도를 쓰지 못하게 막는다.

    일반적인 장면 변환/디버깅은 예전처럼 지도 합성을 허용한다. 하지만 LLM payload,
    윈도우 정답, 평가 레코드는 논문 수치로 이어질 수 있으므로 외부 지도(GeoJSON 또는
    OpenDRIVE)가 기본적으로 필수다. 탐색 목적으로만 명시적 override 를 허용한다.
    """
    experiment_output = any(
        getattr(args, name, None) for name in ("windows_dir", "eval_out", "payload_out")
    )
    if not experiment_output:
        return
    external_map = bool(getattr(args, "road_map", None) or opendrive_path)
    if external_map or getattr(args, "allow_trajectory_map", False):
        return
    raise SystemExit(
        "사고 예측/평가 산출물에는 미래 시나리오 궤적과 독립적인 지도가 필요합니다. "
        "--carla-maps <OpenDRIVE 디렉터리> 또는 --road-map <GeoJSON>을 지정하십시오. "
        "탐색용으로 미래 궤적 기반 지도 합성을 의도한 경우에만 "
        "--allow-trajectory-map을 명시하십시오."
    )


def cmd_build(args: argparse.Namespace) -> int:
    cfg = PipelineConfig()
    cfg.deepaccident.observation_mode = args.mode
    if args.cameras:
        cfg.deepaccident.cameras = tuple(
            c.strip() for c in args.cameras.split(",") if c.strip()
        )
    cfg.deepaccident.include_untracked = args.include_untracked
    # None(자동)이면 SerializeConfig 기본값을 그대로 둔다. 윈도우 경로는
    # WindowConfig.actor_cap() 이 장면마다 다시 정한다.
    if args.max_actors is not None:
        cfg.serialize.max_actors = args.max_actors
    cfg.serialize.language = args.language
    cfg.serialize.provider = args.provider
    cfg.serialize.model = args.model
    cfg.serialize.include_json_block = args.json_block
    if args.provider_extra:
        cfg.serialize.provider_extra = json.loads(args.provider_extra)
    predictor_note = ""
    path, mode, why = resolve_predictor(args)
    if path:
        # 학습한 예측기로 갈아끼운다. torch 는 여기서만 필요하므로 이 시점에
        # import 된다 (`TorchPredictor` 가 지연 임포트한다).
        from traffic_llm.predict_model import TorchPredictor

        cfg.predictor = TorchPredictor(
            path, mode=mode, device=args.predictor_device
        )
        predictor_note = f"{os.path.basename(path)} (mode={mode})"
        print(f"경로 예측기: {path} (mode={mode})  — {why}")
    else:
        print(f"경로 예측기: 규칙 기반 (지도 제약)  — {why}")

    pair_reranker = None
    pair_reranker_note = ""
    if args.pair_reranker_model:
        from traffic_llm.pair_reranker_v2 import load_pair_reranker

        pair_reranker = load_pair_reranker(args.pair_reranker_model)
        pair_reranker_note = (
            f"{os.path.basename(args.pair_reranker_model)} "
            f"(threshold={pair_reranker.threshold:.6g})"
        )
        print(
            f"Pair re-ranker: {args.pair_reranker_model} "
            f"(threshold={pair_reranker.threshold:.6g})"
        )
    runner = DeepAccidentRunner(args.root, cfg)

    xodr = None
    if args.carla_maps:
        sc_probe = runner.list_scenarios()
        town = next(
            (s.town for s in sc_probe if args.scenario in s.scenario), args.scenario
        )
        xodr = find_xodr(town, [args.carla_maps])
        if xodr:
            print(f"CARLA 지도 사용: {xodr}")
        else:
            print(f"경고: {town}.xodr 을 찾지 못해 궤적 합성으로 진행합니다")

    enforce_experiment_map_policy(args, xodr)

    res = runner.build(
        args.scenario, args.type, opendrive_path=xodr, road_map_path=args.road_map
    )
    print(res.summary())
    stats = {k: v for k, v in res.perception_stats.get("all", {}).items() if v}
    print(f"  인지 통계: {stats}")

    for p in (args.out, args.text_out, args.eval_out):
        if p:
            d = os.path.dirname(os.path.abspath(p))
            os.makedirs(d, exist_ok=True)

    # 스냅샷은 한 번만 만들어 재사용한다. res.snapshots() 는 부를 때마다
    # 파이프라인을 처음부터 다시 돌리므로, 출력마다 부르면 같은 일을 세 번 한다.
    snaps = list(res.snapshots(rate_hz=args.rate))

    fj = open(args.out, "w", encoding="utf-8") if args.out else None
    ft = open(args.text_out, "w", encoding="utf-8") if args.text_out else None
    fe = open(args.eval_out, "w", encoding="utf-8") if args.eval_out else None
    n = 0
    first_text = None
    try:
        for snap in snaps:
            if fj:
                fj.write(
                    json.dumps(to_json(snap, cfg.serialize), ensure_ascii=False) + "\n"
                )
            text = to_text(snap, cfg.serialize)
            if first_text is None:
                first_text = text
            if ft:
                ft.write(text + "\n\n" + "=" * 70 + "\n\n")
            if fe:
                fe.write(
                    json.dumps(
                        to_evaluation_record(snap, cfg.serialize), ensure_ascii=False
                    )
                    + "\n"
                )
            n += 1
    finally:
        for f in (fj, ft, fe):
            if f:
                f.close()

    print(f"\n스냅샷 {n}개 생성")
    if args.print_first and first_text:
        print("=" * 72)
        print(first_text)

    if args.windows_dir:
        wcfg = WindowConfig(
            window_s=args.window,
            stride_s=args.stride,
            horizon_s=args.horizon,
            snapshot_rate_hz=args.rate,
            history_stride_s=args.history_stride,
            max_actors=args.max_actors,
            max_closing_pairs=args.max_closing_pairs,
            include_json_block=args.json_block,
            payload_profile=args.payload_profile,
            early_credit_s=args.early_credit,
            warmup=args.warmup,
            full_window=args.full_window,
            history_mode=args.history_mode,
            pair_reranker_model=pair_reranker,
            pair_reranker_note=pair_reranker_note,
        )
        # 산출물이 어느 분할·어느 트레이스에서 온 것인지 경로에 남긴다.
        # BEV 와 **같은 구조**를 쓴다 (`<분할>/<accident|normal>`) — 같은 시나리오의
        # 사고/정상 트레이스는 이름이 같아서 outcome 을 폴더로 갈라 두지 않으면
        # 서로를 덮어쓴다.
        win_dir = out_subdir(
            args.windows_dir, res.scenario, True, args.flat_out, args.out_label
        )
        collision = estimate_collision(res.scenario, cfg.deepaccident)
        agent_ids = {
            ag: res.scenario.meta.agent_id_of(ag) for ag in res.scenario.agents
        }
        manifest = write_window_set(
            snaps,
            out_dir=win_dir,
            scfg=cfg.serialize,
            cfg=wcfg,
            collision=collision,
            agent_carla_ids=agent_ids,
            scenario_id=res.scenario.scenario_id,
            scenario_split=res.scenario.scenario_type,
            dataset_split=res.scenario.split,
            town=res.scenario.town,
            predictor_note=predictor_note,
            map_source=res.map_source,
            map_built_from_scenario_trajectories=(res.roadgen_report is not None),
        )
        s = manifest["summary"]
        mc = manifest["config"]
        print(
            f"\n윈도우 payload 생성: {win_dir}\n"
            f"  provider={mc['provider']}  model={mc['model']}\n"
            f"  endpoint {mc['endpoint']}\n"
            f"  T={args.window:g}s, stride={args.stride:g}s, 미래 N={args.horizon:g}s\n"
            f"  윈도우 {s['n_windows']}개 "
            f"(사고 포함 {s['n_windows_with_accident']}, "
            f"사고 없음 {s['n_windows_without_accident']})\n"
            f"  입력 크기: 평균 {s['input_chars_mean']:,}자, "
            f"최대 {s['input_chars_max']:,}자"
        )
        if s.get("dropped_after_collision"):
            print(
                f"  충돌이 관측 구간 안에 든 윈도우 {s['dropped_after_collision']}개는 "
                "예측 문제가 아니므로 제외"
            )
        c = manifest["collision"]
        if c.get("occurred"):
            print(
                f"  충돌 정답: t={c['time_s']}s (프레임 {c['frame']}, "
                f"방식 {c['estimation_method']}), 주체 CARLA id {c['carla_ids']}"
            )
        else:
            print("  충돌 정답: 없음 (전 구간 '사고 없음')")
        print(f"  파일: llm_payload_<구간>.json / ground_truth_<구간>.json / manifest.json")

    if args.bev_dir:
        bcfg = BevConfig(
            width_px=args.bev_width,
            height_px=args.bev_height,
            focus=args.bev_focus,
            label_mode=args.bev_labels,
            image_format=args.bev_format,
            fixed_view=not args.bev_follow,
            draw_predictions=args.bev_predictions,
        )
        rend = BevRenderer(
            res.network,
            bcfg,
            lane_width_m=cfg.lane.lane_width_m,
            drive_side=cfg.lane.drive_side,
        )
        bev_dir = out_subdir(
            args.bev_dir, res.scenario, True, args.flat_out, args.out_label
        )
        paths = rend.render_sequence(
            snaps,
            bev_dir,
            res.scenario.scenario,
            title=f"{res.scenario.town} · {res.scenario.scenario}",
        )
        idx = write_index_html(
            paths,
            os.path.join(bev_dir, "index.html"),
            title=f"BEV · {res.scenario.scenario}",
        )
        print(
            f"\nBEV 이미지 {len(paths)}개: {bev_dir}\n"
            f"  파일명 <시나리오>_<타임스탬프>.{args.bev_format} "
            f"(예: {os.path.basename(paths[0]) if paths else '-'})\n"
            f"  훑어보기: {idx}"
        )

    if args.payload_out:
        # 단일 스냅샷 기반 요청 본문 (윈도우 방식과 별개로 유지)
        mid = snaps[len(snaps) // 2]
        question = args.question or {
            "ko": "지금 가장 위험한 상황과 그 근거는?",
            "en": "What is the most dangerous situation right now, and why?",
        }[args.language]
        payload = build_messages(mid, question, cfg.serialize)
        with open(args.payload_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print(f"단일 스냅샷 요청 본문: {args.payload_out}")

    for label, path in (("JSONL", args.out), ("텍스트", args.text_out),
                        ("평가레코드", args.eval_out)):
        if path:
            print(f"  {label}: {path}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="DeepAccident → LLM 입력 변환")
    p.add_argument("--root", required=True, help="DeepAccident 데이터 루트")
    p.add_argument("--list", action="store_true", help="시나리오 목록만 출력")
    p.add_argument("--evaluate", action="store_true", help="정답 대비 정확도 평가")
    p.add_argument("--scenario", default="Town01", help="시나리오 이름(부분 일치)")
    p.add_argument("--type", default=None, help="분할 (type1_subtype1_accident 등)")
    p.add_argument("--mode", choices=["sensor3d", "camera"], default="sensor3d",
                   help="관측 생성 모드. sensor3d(기본)=레이블의 3D 위치·방위·"
                        "속도를 그대로 사용, camera=2D bbox 에서 단안 역투영으로 추정")
    p.add_argument("--cameras", default=None,
                   help="사용할 카메라 (쉼표 구분). 생략하면 calib 에 있는 "
                        "전부(DeepAccident 는 6대: Camera_Front, Camera_FrontLeft, "
                        "Camera_FrontRight, Camera_Back, Camera_BackLeft, "
                        "Camera_BackRight). 예: --cameras Camera_Front")
    p.add_argument("--rate", type=float, default=2.0, help="스냅샷 주기 [Hz]")
    p.add_argument("--max-actors", type=int, default=None,
                   help="수록 차량 수 상한. 생략하면 관측된 차량 수에 맞춰 "
                        "자동 산정한다 (관측 전부, 하한 15·상한 40). 고정하면 "
                        "관측이 늘어도 담는 수가 그대로여서 새로 보인 차량이 "
                        "기존 차량을 밀어낸다")
    p.add_argument("--max-closing-pairs", type=int, default=None,
                   help="접근 쌍 표시 수 상한. 생략하면 차량 수의 0.5배 "
                        "(하한 6·상한 20). 접촉이 임박한 쌍은 상한과 무관하게 "
                        "포함된다")
    p.add_argument("--provider", default="claude",
                   choices=["claude", "openai", "gemini"],
                   help="payload 형식. claude(기본)=Anthropic Messages, "
                        "openai=Chat Completions, gemini=generateContent")
    p.add_argument("--model", default=None,
                   help="모델 id. 생략하면 provider 기본값 — claude 만 기본값이 "
                        "있고 openai·gemini 는 반드시 지정해야 한다")
    p.add_argument("--provider-extra", default=None,
                   help='provider 고유 파라미터 JSON. 본문에 깊은 병합된다 '
                        '(예: \'{"reasoning_effort": null}\')')
    p.add_argument("--language", choices=["ko", "en"], default="ko",
                   help="LLM 입력 언어. 기본 한국어 (en 이면 payload·질문·"
                        "system 프롬프트·스키마 설명·정답 주석 모두 영어)")
    p.add_argument("--include-untracked", action="store_true",
                   help="id=-1 (추적 불가) 객체 포함")
    p.add_argument(
        "--predictor", default=None,
        help="학습한 경로 예측기(.pt). 생략하면 "
             f"{DEFAULT_PREDICTOR} 를 쓰고, 그 파일이나 torch 가 없으면 "
             "규칙 기반으로 내려간다. 예: out/predict_model/ranknet_best.pt",
    )
    p.add_argument(
        "--no-predictor", action="store_true",
        help="학습 모델을 쓰지 않고 규칙 기반(지도 제약)으로 강제한다",
    )
    p.add_argument(
        "--predictor-mode", choices=["rank", "waypoints"], default=None,
        help="--predictor 의 출력 형태. 생략하면 **모델 파일에서 판정한다** "
             "(WaypointNet→waypoints, RankNet→rank). rank=후보 순위(확률 분포 "
             "유지), waypoints=좌표 직접 회귀(궤적 하나, 확률 1.0)",
    )
    p.add_argument("--predictor-device", default="cpu",
                   help="예측기 실행 장치 (cpu/cuda). 스냅샷마다 액터별로 부르므로 "
                        "작은 모델은 cpu 가 더 빠르다")
    p.add_argument(
        "--pair-reranker-model", default=None,
        help="학습한 V2 actor-pair re-ranker(.json). compact payload의 내부 "
             "top-50 후보를 점수화해 top-12를 선택한다",
    )
    p.add_argument("--carla-maps", default=None,
                   help="CARLA OpenDrive 디렉터리 (있으면 .xodr 사용)")
    p.add_argument("--road-map", default=None, help="도로망 GeoJSON")
    p.add_argument(
        "--allow-trajectory-map", action="store_true",
        help="탐색용으로만 전체 시나리오 궤적에서 지도를 합성하도록 허용한다. "
             "기본적으로 사고예측 윈도우·평가·payload 생성에는 독립적인 "
             "OpenDRIVE/GeoJSON 지도가 필수다",
    )
    p.add_argument("--out", default=None, help="JSONL 출력")
    p.add_argument("--text-out", default=None, help="자연어 브리핑 출력")
    p.add_argument("--eval-out", default=None, help="평가용 정답 레코드 출력")
    p.add_argument("--payload-out", default=None,
                   help="단일 스냅샷 요청 본문 출력 (윈도우 방식과 별개)")
    p.add_argument("--question", default=None,
                   help="단일 스냅샷 질문. 기본값은 --language 에 맞춰 정해진다")

    g = p.add_argument_group("사고 예측 윈도우 (슬라이딩 T초 → 미래 N초 질의)")
    g.add_argument("--windows-dir", default=None,
                   help="윈도우별 payload/정답 출력 디렉터리")
    g.add_argument("--window", type=float, default=5.0, help="관측 윈도우 길이 T [s]")
    g.add_argument("--stride", type=float, default=1.0, help="윈도우 이동 간격 [s]")
    g.add_argument("--horizon", type=float, default=5.0, help="미래 예측 구간 N [s]")
    g.add_argument("--history-stride", type=float, default=1.0,
                   help="윈도우 내 이력 표시 간격 [s]")
    g.add_argument("--json-block", action="store_true",
                   help="구조화 JSON 블록을 payload 에 포함한다 (기본 꺼짐). "
                        "바로 위 자연어 브리핑과 내용이 겹치면서 입력의 79%%를 "
                        "차지한다 — 켜면 payload 가 약 4.5배가 된다. 툴 호출이나 "
                        "정량 파싱이 필요할 때만 켠다")
    g.add_argument(
        "--payload-profile", choices=["standard", "compact"], default="standard",
        help="standard=기존 전체 브리핑, compact=관측 궤적·현재 상태·예측 미래 "
             "웨이포인트만 포함 (접근쌍/TTC/상호작용/ASCII BEV 제외)",
    )
    g.add_argument("--warmup", action="store_true", default=False,
                   help="expanding-prefix 창을 사용한다 (0-1, 0-2, …, 0-5, 1-6, …). "
                        "기본값은 고정 길이 창이며, 짧은 선두 창은 만들지 않는다")
    # 이전 명령줄과의 호환성을 유지한다. 기본이 fixed이므로 --no-warmup은
    # 사실상 no-op이지만 기존 batch command를 그대로 재실행할 수 있어야 한다.
    g.add_argument("--no-warmup", action="store_false", dest="warmup",
                   help="고정 길이 창을 사용한다 (기본값; --warmup의 반대 옵션)")
    g.add_argument("--full-window", action="store_true", default=False,
                   help="이동 창 외에 데이터 전체 구간 창을 하나 더 추가한다")
    g.add_argument("--no-full-window", action="store_false", dest="full_window",
                   help="전체 구간 창을 추가하지 않는다 (기본값)")
    g.add_argument("--early-credit", type=float, default=5.0,
                   help="사고 발생 시점보다 이만큼 먼저 예측한 것을 정답으로 "
                        "인정한다 [초] (main protocol 기본 5초). 늦은 예측은 여전히 오답. 0 이면 구간이 "
                        "정확히 일치해야만 정답")
    g.add_argument("--history-mode", choices=["compact", "full"], default="compact",
                   help="compact=궤적 요약, full=전 스냅샷 JSON")
    b = p.add_argument_group("BEV 이미지 (시간에 따른 조감도 변화)")
    b.add_argument("--bev-dir", default=None,
                   help="BEV 이미지 출력 디렉터리 (<시나리오>_<타임스탬프>.jpg)")
    b.add_argument("--bev-width", type=int, default=1200)
    b.add_argument("--bev-height", type=int, default=900)
    b.add_argument("--bev-focus", choices=["ego", "all"], default="ego",
                   help="시야 기준. ego=관측차량 주변, all=모든 액터 포함")
    b.add_argument("--bev-labels", choices=["all", "ego_and_risky", "none"],
                   default="all", help="차량 라벨 표시 범위")
    b.add_argument("--bev-format", default="jpg", choices=["jpg", "png"])
    b.add_argument("--bev-follow", action="store_true",
                   help="프레임마다 시야를 다시 맞춘다 (기본: 시퀀스 고정 시야)")
    b.add_argument("--bev-predictions", action="store_true",
                   help="예상 경로를 점선으로 겹쳐 그린다 (기본 꺼짐 — 액터 전부의 "
                        "점선이 겹쳐 읽기 어렵다). 켜면 범례에 항목이 추가된다")
    p.add_argument("--out-label", default="",
                   help="산출물 경로의 분할 아래에 붙일 구분 이름 "
                        "(예: 'ko/accident', 'gemini'). 같은 데이터로 여러 설정을 "
                        "돌려 나란히 두고 비교할 때 쓴다")
    p.add_argument("--flat-out", action="store_true",
                   help="산출물을 분할·트레이스 하위폴더 없이 지정한 경로에 "
                        "그대로 쓴다 (기본은 <경로>/<분할>[/<accident|normal>])")
    p.add_argument("--print", dest="print_first", action="store_true",
                   help="첫 스냅샷 텍스트 출력")
    args = p.parse_args(argv)

    if args.list:
        return cmd_list(DeepAccidentRunner(args.root, PipelineConfig()))
    if args.evaluate:
        return cmd_evaluate(args.root, args.rate)
    return cmd_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
