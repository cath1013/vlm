"""명령행 인터페이스.

    python -m traffic_llm.cli \
        --map data/columbus_roads.geojson \
        --vehicle V1:data/v1.mp4:data/v1_tele.csv \
        --vehicle V2:data/v2.mp4:data/v2_tele.csv \
        --hfov 60 --cam-height 1.35 --rate 2 \
        --out out/scenes.jsonl --text-out out/scenes.txt
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

from .config import CameraConfig, PipelineConfig
from .perception import JsonPerception, build_backend
from .pipeline import TrafficSceneConverter
from .roadmap import RoadNetwork
from .serialize import to_json, to_text


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="다중 차량 전면 카메라 영상 + 도로지도 → LLM 입력 변환"
    )
    p.add_argument("--map", required=True, help="도로망 GeoJSON 경로")
    p.add_argument(
        "--vehicle",
        action="append",
        required=True,
        metavar="ID:VIDEO:TELEMETRY[:DETECTIONS_JSON]",
        help="관측차량 정의. 여러 번 지정 가능",
    )
    p.add_argument("--hfov", type=float, default=60.0, help="수평 화각 [deg]")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--cam-height", type=float, default=1.35, help="카메라 지상고 [m]")
    p.add_argument("--cam-pitch", type=float, default=2.0, help="하향 피치 [deg]")
    p.add_argument("--language", choices=["ko", "en"], default="ko",
                   help="LLM 입력 언어 (기본 한국어)")
    p.add_argument("--rate", type=float, default=2.0, help="스냅샷 생성 주기 [Hz]")
    p.add_argument("--t-start", type=float, default=None)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--max-actors", type=int, default=20)
    p.add_argument("--area", default="", help="지역명 (프롬프트에 포함)")
    p.add_argument("--lane-numbering", choices=["from_median", "from_curb"],
                   default="from_median")
    p.add_argument("--drive-side", choices=["right", "left"], default="right")
    p.add_argument("--out", default=None, help="JSONL 출력 경로")
    p.add_argument("--text-out", default=None, help="자연어 브리핑 출력 경로")
    p.add_argument("--print-first", action="store_true", help="첫 스냅샷 텍스트 출력")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    cfg = PipelineConfig(area_name=args.area)
    cfg.perception.sample_hz = args.rate
    cfg.serialize.max_actors = args.max_actors
    cfg.serialize.language = args.language
    cfg.lane.numbering = args.lane_numbering
    cfg.lane.drive_side = args.drive_side

    network = RoadNetwork.from_geojson(args.map, cfg.lane)

    cam = CameraConfig.from_fov(
        width=args.width,
        height=args.height,
        hfov_deg=args.hfov,
        height_m=args.cam_height,
        pitch_deg=args.cam_pitch,
    )

    # 사전 검출 결과가 지정된 차량이 있으면 JSON 백엔드 사용
    det_paths = {}
    specs = []
    for spec in args.vehicle:
        parts = spec.split(":")
        if len(parts) < 3:
            raise SystemExit(f"--vehicle 형식 오류: {spec}")
        vid, video, tele = parts[0], parts[1], parts[2]
        det = parts[3] if len(parts) > 3 else None
        specs.append((vid, video, tele, det))
        if det:
            det_paths[video] = det

    backend = (
        JsonPerception(cfg.perception, det_paths)
        if det_paths
        else build_backend(cfg.perception)
    )
    conv = TrafficSceneConverter(cfg, network, backend)
    for vid, video, tele, _ in specs:
        conv.add_vehicle(vid, video, tele, cam)

    for path in (args.out, args.text_out):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    fj = open(args.out, "w", encoding="utf-8") if args.out else None
    ft = open(args.text_out, "w", encoding="utf-8") if args.text_out else None
    count = 0
    try:
        for snap in conv.run(rate_hz=args.rate, t_start=args.t_start, t_end=args.t_end):
            if fj:
                fj.write(
                    json.dumps(to_json(snap, cfg.serialize), ensure_ascii=False) + "\n"
                )
            text = to_text(snap, cfg.serialize)
            if ft:
                ft.write(text + "\n\n" + "=" * 70 + "\n\n")
            if count == 0 and args.print_first:
                print(text)
            count += 1
    finally:
        if fj:
            fj.close()
        if ft:
            ft.close()

    print(f"스냅샷 {count}개 생성 완료")
    if args.out:
        print(f"  JSONL: {args.out}")
    if args.text_out:
        print(f"  텍스트: {args.text_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
