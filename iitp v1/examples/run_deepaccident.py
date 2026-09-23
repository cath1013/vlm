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
            history_mode=args.history_mode,
        )
        collision = estimate_collision(res.scenario, cfg.deepaccident)
        agent_ids = {
            ag: res.scenario.meta.agent_id_of(ag) for ag in res.scenario.agents
        }
        manifest = write_window_set(
            snaps,
            out_dir=args.windows_dir,
            scfg=cfg.serialize,
            cfg=wcfg,
            collision=collision,
            agent_carla_ids=agent_ids,
            scenario_id=res.scenario.scenario_id,
            scenario_split=res.scenario.scenario_type,
            town=res.scenario.town,
        )
        s = manifest["summary"]
        mc = manifest["config"]
        print(
            f"\n윈도우 payload 생성: {args.windows_dir}\n"
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
        paths = rend.render_sequence(
            snaps,
            args.bev_dir,
            res.scenario.scenario,
            title=f"{res.scenario.town} · {res.scenario.scenario}",
        )
        idx = write_index_html(
            paths,
            os.path.join(args.bev_dir, "index.html"),
            title=f"BEV · {res.scenario.scenario}",
        )
        print(
            f"\nBEV 이미지 {len(paths)}개: {args.bev_dir}\n"
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
    p.add_argument("--carla-maps", default=None,
                   help="CARLA OpenDrive 디렉터리 (있으면 .xodr 사용)")
    p.add_argument("--road-map", default=None, help="도로망 GeoJSON")
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
