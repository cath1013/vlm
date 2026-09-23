"""합성 데모 데이터 생성 + 파이프라인 종단 실행 + 사고 예측 윈도우 생성.

영상 없이도 파이프라인 전체를 검증하기 위해:
  1) 4지 교차로 도로망 GeoJSON 생성
  2) 가상 차량들의 월드 궤적 생성 (교차로 측면충돌 시나리오 포함)
  3) 관측차량 3대의 텔레메트리 CSV 생성
  4) 각 관측차량의 카메라 모델로 다른 차량을 픽셀로 정투영 → 검출 결과 JSON
     (역투영 파이프라인이 원래 위치를 복원하는지 확인 가능)
  5) 슬라이딩 윈도우 사고 예측 payload/정답 세트 생성

DeepAccident 없이 윈도우 기반 예측 실험을 돌려볼 수 있다. 기본값은 **충돌
포함** 시나리오이므로 양성/음성 윈도우가 모두 생긴다 (--no-collision 으로
전 구간 음성만 만들 수 있다).

실행 예
    python examples/make_demo_data.py
    python examples/make_demo_data.py --window 5 --stride 1 --horizon 5 --bev
    python examples/make_demo_data.py --no-collision --duration 12
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 기본 콘솔 코드페이지(cp949)에서 한글이 깨지지 않도록 UTF-8 로 고정
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

from traffic_llm.accident_qa import WindowConfig, write_window_set
from traffic_llm.config import CameraConfig, PipelineConfig
from traffic_llm.geometry import CameraModel, LocalENU, world_to_ego
from traffic_llm.perception import JsonPerception
from traffic_llm.pipeline import TrafficSceneConverter
from traffic_llm.roadmap import RoadNetwork
from traffic_llm.schemas import CollisionTruth
from traffic_llm.serialize import build_messages, to_json, to_text

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data")
ORIGIN = (39.9612, -83.0007)  # Columbus, OH 부근
FPS = 10.0
LANE_W = 3.25

CLASS_WIDTH = {"car": 1.93, "truck": 2.89, "bus": 2.90}
CLASS_LENGTH = {"car": 4.60, "truck": 8.47, "bus": 11.00}
CLASS_HEIGHT = {"car": 1.52, "truck": 3.83, "bus": 3.20}

# 관측 객체에 붙는 트랙 id. 모든 관측자가 같은 물체에 같은 id 를 쓴다 —
# V2X 로 객체 id 를 공유하는 시스템, 그리고 DeepAccident 의 CARLA actor id 와
# 같은 상황이며 `FusionConfig.global_track_ids` 기본값(True)의 전제다.
# 이 id 는 융합의 신원 근거이면서 채점용 신원으로도 쓰인다.
TRACK_ID = {"V1": 101, "V2": 102, "V3": 103, "X1": 111, "X2": 112, "X3": 113,
            "X4": 114}


# ---------------------------------------------------------------- 도로망


def write_map(path: str) -> None:
    """남북 간선(High St, 왕복 4차선) + 동서 간선(Broad St, 왕복 4차선) 교차."""
    enu = LocalENU(*ORIGIN)

    def geo(e: float, n: float):
        lat, lon = enu.to_geo(e, n)
        return [lon, lat]

    def road(rid: str, name: str, a, b):
        return {
            "type": "Feature",
            "properties": {
                "id": rid,
                "name": name,
                "lanes": 4,
                "lanes:forward": 2,
                "lanes:backward": 2,
                "maxspeed": "50",
                "highway": "primary",
            },
            "geometry": {"type": "LineString", "coordinates": [geo(*a), geo(*b)]},
        }

    fc = {
        "type": "FeatureCollection",
        "features": [
            road("high_s", "High St", (0, -300), (0, 0)),
            road("high_n", "High St", (0, 0), (0, 300)),
            road("broad_w", "Broad St", (-300, 0), (0, 0)),
            road("broad_e", "Broad St", (0, 0), (300, 0)),
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- 궤적


def northbound(lane_from_median: int, n0: float, v: float):
    """High St 북행.

    북행 진행방향 기준 좌측이 서쪽(e<0). 우측통행이면 진행 차로는
    진행방향 기준 우측 → e > 0 (동쪽). 1차선(중앙선쪽)은 e=+0.5*LANE_W.
    """
    e = (lane_from_median - 0.5) * LANE_W
    return lambda t: (e, n0 + v * t), 0.0, v


def southbound(lane_from_median: int, n0: float, v: float):
    """High St 남행 → 진행방향 기준 우측은 서쪽(e<0)."""
    e = -(lane_from_median - 0.5) * LANE_W
    return lambda t: (e, n0 - v * t), 180.0, v


def eastbound(lane_from_median: int, e0: float, v: float):
    """Broad St 동행 → 진행방향 기준 우측은 남쪽(n<0)."""
    n = -(lane_from_median - 0.5) * LANE_W
    return lambda t: (e0 + v * t, n), 90.0, v


def westbound(lane_from_median: int, e0: float, v: float):
    """Broad St 서행 → 진행방향 기준 우측은 북쪽(n>0)."""
    n = (lane_from_median - 0.5) * LANE_W
    return lambda t: (e0 - v * t, n), 270.0, v


def nb_lane_change(lane_from: int, lane_to: int, n0: float, v: float,
                   t_start: float = 2.0, dur: float = 3.0):
    """북행 차량의 실제 차선변경 (t_start ~ t_start+dur 사이 선형 이동)."""
    e_from = (lane_from - 0.5) * LANE_W
    e_to = (lane_to - 0.5) * LANE_W

    def pos(t):
        r = min(max((t - t_start) / dur, 0.0), 1.0)
        return (e_from + r * (e_to - e_from), n0 + v * t)

    return pos, 0.0, v


# V1(북행 2차선)이 교차로 정지선을 통과하는 시각. n(t) = -120 + 12t 이므로
# 서행 1차선(n=+1.625)을 지나는 시각은 t=10.135s 다. X4 는 그 순간 V1 의
# 차로 중심(e=+4.875)에 도달하도록 초기 위치를 역산한다 → 측면충돌.
V1_N0, V1_V = -120.0, 12.0
X4_V = 14.0


def x4_spawn_e() -> float:
    """X4(서행) 초기 e. V1 과 교차점에서 동시에 도달하도록 역산."""
    e_v1 = (2 - 0.5) * LANE_W  # V1 차로 중심 (북행 2차선)
    n_x4 = (1 - 0.5) * LANE_W  # X4 차로 중심 (서행 1차선)
    t_hit = (n_x4 - V1_N0) / V1_V
    return e_v1 + X4_V * t_hit


def build_actors(with_collision: bool, lead_in_s: float = 0.0):
    """(id, cls, pos(t), heading, speed) 목록. 앞 3대는 관측차량.

    lead_in_s: 전체 궤적을 이만큼 뒤로 미룬다. 상대 배치는 그대로 유지되고
      충돌 시각만 늦춰지므로, 사고가 예측 지평(N초) 밖에 있는 **음성 윈도우**가
      앞쪽에 생긴다. 0 이면 첫 윈도우부터 사고가 지평 안에 들어와 모든 윈도우가
      양성이 되어 평가에 쓸 수 없다.
    """
    actors = [
        ("V1", "car", *northbound(2, V1_N0, V1_V)),    # 관측차량, 2차선 북행
        ("V2", "car", *northbound(1, -60.0, 9.0)),     # 관측차량, 1차선 북행 (V1 전방)
        ("V3", "car", *eastbound(1, -140.0, 14.0)),    # 관측차량, Broad St 동행
        ("X1", "truck", *northbound(2, -70.0, 8.0)),   # V1 직전방 트럭
        ("X2", "car", *southbound(1, 80.0, 13.0)),     # 대향 차량
        # V1 근거리(약 25m) 전방에서 2→1차선 실제 차선변경 — 검출 검증용
        ("X3", "car", *nb_lane_change(2, 1, -95.0, 12.0, t_start=2.0, dur=3.0)),
    ]
    if with_collision:
        # Broad St 서행 — 신호 위반으로 교차로에 진입해 V1 좌측면을 충격
        actors.append(("X4", "car", *westbound(1, x4_spawn_e(), X4_V)))
    if lead_in_s > 0:
        actors = [
            (aid, cls, _delay(pos, lead_in_s), head, v)
            for aid, cls, pos, head, v in actors
        ]
    return actors


def _delay(pos_fn, lead_in_s: float):
    """궤적을 lead_in_s 초 뒤로 미룬 함수. 상대 배치는 보존된다."""
    return lambda t: pos_fn(t - lead_in_s)


def scripted_collision(
    actors, a_id: str = "V1", b_id: str = "X4", contact_slack_m: float = 2.0
) -> CollisionTruth:
    """두 궤적의 최근접 시점을 찾아 충돌 정답을 구성한다.

    접촉 기준은 deepaccident.estimate_collision 과 동일하게
    **두 객체 절반 길이 합 + contact_slack_m** 을 쓴다. 시각을 손으로 적어
    넣지 않고 궤적에서 산출하므로, 궤적 파라미터를 바꿔도 정답이 따라온다.
    """
    by_id = {a[0]: a for a in actors}
    if a_id not in by_id or b_id not in by_id:
        return CollisionTruth(occurred=False, method="none")
    A, B = by_id[a_id], by_id[b_id]
    contact_m = (
        CLASS_LENGTH[A[1]] / 2 + CLASS_LENGTH[B[1]] / 2 + contact_slack_m
    )
    step = 1.0 / (FPS * 10)  # 프레임보다 촘촘히 훑어 접촉 시각을 정확히
    best_t, best_d = None, float("inf")
    t = 0.0
    while t <= 30.0:
        d = math.dist(A[2](t), B[2](t))
        if d < best_d:
            best_t, best_d = t, d
        t += step
    if best_t is None or best_d > contact_m:
        return CollisionTruth(occurred=False, method="none")
    # 접촉 시각 = 간격이 접촉 기준 아래로 내려가는 첫 시점
    t = 0.0
    hit_t = best_t
    while t <= best_t:
        if math.dist(A[2](t), B[2](t)) <= contact_m:
            hit_t = t
            break
        t += step
    return CollisionTruth(
        occurred=True,
        carla_ids=(TRACK_ID[a_id], TRACK_ID[b_id]),
        classes=(A[1], B[1]),
        agent_roles=(a_id, b_id),
        frame=int(round(hit_t * FPS)) + 1,
        time_s=hit_t,
        min_distance_m=best_d,
        intensity=None,
        method="scripted",
    )


# ---------------------------------------------------------------- 출력 생성


def write_telemetry(
    path: str, pos_fn, heading: float, speed: float, enu: LocalENU,
    duration: float,
):
    rows = ["t,lat,lon,heading_deg,speed_mps,yaw_rate_dps"]
    n = int(round(duration * FPS)) + 1
    for k in range(n):
        t = k / FPS
        e, nn = pos_fn(t)
        lat, lon = enu.to_geo(e, nn)
        rows.append(f"{t:.3f},{lat:.8f},{lon:.8f},{heading:.2f},{speed:.3f},0.0")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


def write_detections(path: str, observer, others, cam: CameraConfig,
                     duration: float):
    """관측차량 시점에서 다른 차량을 픽셀로 정투영해 검출 결과 합성."""
    model = CameraModel(cam)
    obs_id, _, obs_pos, obs_head, _ = observer
    dets = []
    n_frames = int(round(duration * FPS)) + 1

    for k in range(n_frames):
        t = k / FPS
        origin = obs_pos(t)
        for aid, cls, pos_fn, head, _ in others:
            x, y = world_to_ego(pos_fn(t), origin, obs_head)
            if x < 3.0:  # 후방/근접은 전면 카메라에 안 잡힘
                continue
            # 3D 박스 8개 코너를 모두 투영해 축정렬 bbox 를 만든다.
            # 실제 검출기 bbox 는 차량 전체의 가시 외곽이므로, 후면만
            # 투영하면 접지점이 곧 중심이 되어 물리적으로 맞지 않는다.
            w_real = CLASS_WIDTH[cls]
            l_real = CLASS_LENGTH[cls]
            h_real = CLASS_HEIGHT[cls]
            rel_head = math.radians(head - obs_head)  # 관측자 기준 상대 방위
            c, s = math.cos(rel_head), math.sin(rel_head)
            us, vs = [], []
            ok = True
            for dl in (l_real / 2, -l_real / 2):
                for dw in (w_real / 2, -w_real / 2):
                    # 차량 좌표계: 전방 x, 좌측 y
                    ex = x + dl * c - dw * s
                    ey = y + dl * s + dw * c
                    for dz in (0.0, h_real):
                        p = model.ego_to_pixel(ex, ey, dz)
                        if p is None:
                            ok = False
                            break
                        us.append(p[0])
                        vs.append(p[1])
            if not ok or not us:
                continue
            x1, x2 = min(us), max(us)
            y1, y2 = min(vs), max(vs)
            if x2 < 0 or x1 > cam.width or y2 < 0 or y1 > cam.height:
                continue
            dets.append(
                {
                    "frame": k,
                    "t": t,
                    "track_id": TRACK_ID[aid],
                    "cls": cls,
                    "conf": 0.92,
                    "bbox": [
                        max(x1, 0.0),
                        max(y1, 0.0),
                        min(x2, float(cam.width)),
                        min(y2, float(cam.height)),
                    ],
                }
            )
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"fps": FPS, "detections": dets}, f, ensure_ascii=False)
    return len(dets)


# ---------------------------------------------------------------- 메인


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="합성 데모 데이터 + 사고 예측 윈도우 세트 생성"
    )
    p.add_argument("--out", default=OUT, help="출력 디렉터리")
    p.add_argument("--window", type=float, default=5.0,
                   help="윈도우 길이 T [초] (기본 5)")
    p.add_argument("--stride", type=float, default=1.0,
                   help="윈도우 이동 간격 [초] (기본 1)")
    p.add_argument("--horizon", type=float, default=5.0,
                   help="미래 예측 지평 N [초], 1초 구간으로 나눔 (기본 5)")
    p.add_argument("--rate", type=float, default=2.0,
                   help="스냅샷 샘플링 [Hz] (기본 2)")
    p.add_argument("--history-stride", type=float, default=1.0,
                   help="윈도우 안에서 이력으로 넣을 스냅샷 간격 [초]")
    p.add_argument("--history-mode", default="compact",
                   choices=("compact", "full"),
                   help="윈도우 이력 표현 (compact=요약, full=전체 스냅샷)")
    p.add_argument("--max-actors", type=int, default=10,
                   help="윈도우 payload 에 담을 최대 차량 수")
    p.add_argument("--duration", type=float, default=None,
                   help="시뮬레이션 길이 [초]. 기본: 충돌 직전까지 (충돌 "
                        "시나리오) 또는 12초 (충돌 없음)")
    p.add_argument("--no-collision", action="store_true",
                   help="충돌 차량(X4)을 넣지 않는다 — 전 구간 '사고 없음' 정답")
    p.add_argument("--lead-in", type=float, default=None,
                   help="충돌 전 여유 시간 [초]. 기본은 --horizon 과 같게 두어 "
                        "음성/양성 윈도우가 비슷한 수로 생긴다")
    p.add_argument("--language", choices=["ko", "en"], default="ko",
                   help="LLM 입력 언어 (기본 한국어)")
    p.add_argument("--provider", default="claude",
                   choices=["claude", "openai", "gemini"],
                   help="payload 형식 (기본 claude)")
    p.add_argument("--model", default=None,
                   help="모델 id. openai·gemini 는 반드시 지정해야 한다")
    p.add_argument("--bev", action="store_true",
                   help="시간별 BEV 이미지도 생성 (out/bev)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = args.out
    with_collision = not args.no_collision
    os.makedirs(out, exist_ok=True)
    enu = LocalENU(*ORIGIN)
    map_path = os.path.join(out, "roads.geojson")
    write_map(map_path)

    lead_in = args.horizon if args.lead_in is None else args.lead_in
    actors = build_actors(with_collision, lead_in_s=lead_in)
    collision = (
        scripted_collision(actors) if with_collision
        else CollisionTruth(occurred=False, method="none")
    )
    if args.duration is not None:
        duration = args.duration
    elif collision.occurred and collision.time_s is not None:
        # DeepAccident 의 사고 분할과 같이 충돌 시점에서 기록을 끊는다.
        # 프레임 격자로 내림 → 마지막 프레임은 충돌 직전이다.
        duration = math.floor(collision.time_s * FPS) / FPS
    else:
        duration = 12.0

    if collision.occurred:
        print(
            f"충돌 시나리오: {collision.agent_roles[0]}(북행 2차선) × "
            f"{collision.agent_roles[1]}(서행 1차선) 교차로 측면충돌\n"
            f"  충돌 시각 t={collision.time_s:.2f}s, 최근접 "
            f"{collision.min_distance_m:.2f}m, 기록 길이 {duration:.1f}s"
        )
    else:
        print(f"충돌 없는 시나리오, 기록 길이 {duration:.1f}s")

    observers = actors[:3]
    cam = CameraConfig.from_fov(
        1920, 1080, hfov_deg=62.0, height_m=1.35, pitch_deg=2.0
    )

    specs = []
    for obs in observers:
        oid = obs[0]
        tele_path = os.path.join(out, f"{oid}_tele.csv")
        det_path = os.path.join(out, f"{oid}_det.json")
        write_telemetry(tele_path, obs[2], obs[3], obs[4], enu, duration)
        others = [a for a in actors if a[0] != oid]
        n = write_detections(det_path, obs, others, cam, duration)
        print(f"  {oid}: 텔레메트리 + 검출 {n}건")
        specs.append((oid, f"{oid}.mp4", tele_path, det_path))

    # ---- 파이프라인 실행
    cfg = PipelineConfig(area_name="Columbus, OH (High St × Broad St)")
    cfg.perception.sample_hz = args.rate
    cfg.serialize.max_actors = 12
    cfg.serialize.language = args.language
    cfg.serialize.provider = args.provider
    cfg.serialize.model = args.model
    network = RoadNetwork.from_geojson(map_path, cfg.lane, origin=ORIGIN)
    print(
        f"\n도로 {len(network.roads)}개, "
        f"교차로 {sum(1 for j in network.junctions.values() if j.is_intersection)}개 로딩"
    )

    backend = JsonPerception(cfg.perception, {s[1]: s[3] for s in specs})
    conv = TrafficSceneConverter(cfg, network, backend)
    for oid, video, tele, det_path in specs:
        # 데모에는 실제 영상 파일이 없으므로 검출 결과를 직접 주입한다
        conv.add_vehicle(
            oid, video, tele, cam, detections=backend.run(video, t0=0.0),
            self_track_id=TRACK_ID[oid],
        )

    snaps = list(conv.run(rate_hz=args.rate))
    print(f"스냅샷 {len(snaps)}개 생성\n")

    mid = snaps[len(snaps) // 2]
    brief = to_text(mid, cfg.serialize)
    print("=" * 72)
    print(brief)
    print("=" * 72)
    # 콘솔 인코딩과 무관하게 확인할 수 있도록 파일로도 남긴다
    with open(os.path.join(out, "brief.txt"), "w", encoding="utf-8") as f:
        for s in snaps:
            f.write(to_text(s, cfg.serialize) + "\n\n" + "=" * 70 + "\n\n")

    jsonl = os.path.join(out, "scenes.jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for s in snaps:
            f.write(json.dumps(to_json(s, cfg.serialize), ensure_ascii=False) + "\n")

    question = {
        "ko": "V1이 지금 1차선으로 차선변경해도 안전한가?",
        "en": "Is it safe for V1 to change into lane 1 right now?",
    }[args.language]
    payload = build_messages(mid, question, cfg.serialize)
    with open(os.path.join(out, "llm_payload.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    # ---- 사고 예측용 슬라이딩 윈도우 payload/정답
    win_dir = os.path.join(out, "windows")
    wcfg = WindowConfig(
        window_s=args.window,
        stride_s=args.stride,
        horizon_s=args.horizon,
        snapshot_rate_hz=args.rate,
        history_stride_s=args.history_stride,
        history_mode=args.history_mode,
        max_actors=args.max_actors,
    )
    manifest = write_window_set(
        snaps,
        out_dir=win_dir,
        scfg=cfg.serialize,
        cfg=wcfg,
        collision=collision if collision.occurred else None,
        # 관측차량의 신원은 이름↔트랙 id 대응이 정답이다
        agent_carla_ids={oid: TRACK_ID[oid] for oid, *_ in specs},
        scenario_id="synthetic/columbus_demo"
        + ("_collision" if collision.occurred else "_normal"),
        scenario_split="synthetic",
        town="Columbus_demo",
    )
    s = manifest["summary"]
    print(
        f"\n사고 예측 윈도우: {win_dir}\n"
        f"  T={args.window:g}s, stride={args.stride:g}s, 미래 N={args.horizon:g}s"
        f" → 윈도우 {s['n_windows']}개"
        f" (사고 포함 {s['n_windows_with_accident']}, "
        f"사고 없음 {s['n_windows_without_accident']})\n"
        f"  입력 크기 평균 {s['input_chars_mean']}자 / 최대 "
        f"{s['input_chars_max']}자"
    )
    if manifest["windows"]:
        first = manifest["windows"][0]
        print(f"      예: {first['payload']} / {first['ground_truth']}")

    if args.bev:
        from traffic_llm.bev_render import BevConfig, BevRenderer, write_index_html

        bev_dir = os.path.join(out, "bev")
        rend = BevRenderer(
            network,
            BevConfig(width_px=1100, height_px=850, focus="all"),
            lane_width_m=cfg.lane.lane_width_m,
            drive_side=cfg.lane.drive_side,
        )
        paths = rend.render_sequence(
            snaps, bev_dir, "columbus_demo",
            title="Columbus demo · High St × Broad St",
        )
        write_index_html(paths, os.path.join(bev_dir, "index.html"))
        print(f"\nBEV 이미지 {len(paths)}개: {bev_dir}")

    print(f"\n출력: {jsonl}")
    print(f"      {os.path.join(out, 'llm_payload.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
