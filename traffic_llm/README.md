# traffic_llm

여러 대의 차량에 **탑재된 카메라 k대**(+ 노변 인프라 센서)의 영상과 해당 지역
도로 지도를 융합해, 전체 교통 상황(차량별 진행 차선·방향·속도·기동·예상 경로 및
차량 간 상호작용)을 LLM 입력으로 변환하는 파이프라인.

관측자 1기가 카메라를 몇 대든 달 수 있고, 검출마다 어느 카메라의 것인지
(`Detection.camera`) 기록되어 역투영에 **그 카메라의** 파라미터가 쓰인다.
DeepAccident 는 관측자마다 6대(전방·전측방±·후측방±·후방)를 달아 방위 360°를
덮는다.

**DeepAccident 데이터셋을 바로 입력으로 쓸 수 있다** → [DeepAccident 사용법](#deepaccident-데이터셋-사용)

## 파이프라인 구조

```
관측자 N기 ────► 검출·추적 (YOLO+ByteTrack / 사전계산 JSON / DeepAccident 레이블)
 × 카메라 k대         │  bbox (픽셀) + 카메라 이름  [+ 3D 센서면 측정 위치·방위·속도]
                     ▼
             역투영 — **그 카메라의** 파라미터로 (평면노면 IPM + 근접면 보정) [geometry.py]
                     │  차량좌표 (x_fwd, y_left)
텔레메트리 N개 ──────┤
 (GPS/IMU 또는 합성)  ▼
             관측자→월드 변환 (지역 ENU 평면)
                     │  월드좌표 (e, n)
                     ▼
             다중 관측자 융합 (거리적응 게이팅 + 최적할당)   [fusion.py]
                     │  중복제거된 액터 목록 (+ 인프라는 별도 분리)
도로 지도 ──────────►┤
 (GeoJSON/.xodr/합성) ▼
             도로/차선 매칭 → 운동학 추정 → 재매칭         [roadmap, kinematics]
                     │  차선번호, 속도, 가속도, 기동
                     ▼
             지도제약 경로예측 + 상호작용 분석              [prediction, kinematics]
                     │
                     ▼
             LLM 입력 직렬화 (자연어 / JSON / ASCII BEV)      [serialize.py]
                     │
                     ├──► 슬라이딩 윈도우 사고예측 질의 + 정답  [accident_qa.py]
                     │      └─ provider 별 요청 본문 조립       [providers.py]
                     └──► BEV 시계열 이미지 (지도 + 회전 사각형) [bev_render.py]
```

## 빠른 시작

### 1. 데모 (영상 없이 종단 실행)

```bash
pip install numpy
python examples/make_demo_data.py
```

합성 교차로 시나리오를 만들고 전체 파이프라인을 돌려 `examples/demo_data/` 에
결과를 남긴다. 카메라 정투영으로 검출값을 합성하므로 역투영 정확도를 직접
검증할 수 있다.

구성: 관측차량 3대(V1 북행 2차선, V2 북행 1차선, V3 Broad St 동행) +
주변차량 3대(선행 트럭, 대향차, 실제 차선변경 1건) +
**신호 위반 서행차 X4** — 교차로에서 V1 좌측면을 충격한다.

충돌이 있으므로 **사고 예측 윈도우의 양성/음성 표본이 모두** 나온다. 충돌
시각은 손으로 적지 않고 두 궤적의 최근접 시점에서 산출하며(접촉 기준은
DeepAccident 어댑터와 동일: 절반 길이 합 + 여유 2m), 기록은 DeepAccident 의
사고 분할처럼 충돌 직전에서 끊는다.

```bash
# 윈도우 설정을 바꿔 실험
python examples/make_demo_data.py --window 5 --stride 1 --horizon 5 --bev

# 전 구간 '사고 없음' 음성 세트만
python examples/make_demo_data.py --no-collision --duration 12
```

기본값(T=5s, stride=1s, N=5s)에서 윈도우 10개가 나오며 양성 5 / 음성 5 로
갈린다. `--lead-in` 이 충돌 전 여유 시간을 정한다 — 기본은 `--horizon` 과 같게
두어 앞쪽 윈도우에서는 사고가 예측 지평 밖에 있게 한다(0 이면 모든 윈도우가
양성이 되어 평가에 쓸 수 없다).

| 주요 옵션 | 뜻 |
|---|---|
| `--window/--stride/--horizon` | 윈도우 길이 T, 이동 간격, 미래 지평 N [초] |
| `--rate` | 스냅샷 샘플링 [Hz] |
| `--history-stride`, `--history-mode` | 윈도우 내 이력 간격 / `compact`\|`full` |
| `--payload-profile` | `standard`=기존 브리핑, `compact`=관측 궤적·현재 상태·미래 웨이포인트만 포함 |
| `--duration`, `--lead-in` | 기록 길이, 충돌 전 여유 |
| `--no-collision` | 충돌차 X4 제외 |
| `--bev` | 시간별 BEV 이미지도 생성 (`demo_data/bev/`) |

`--payload-profile compact`는 JSON 이력의 `--history-mode compact`와 별개다.
전자는 LLM 입력 전체에서 접근쌍 외삽, TTC/헤드웨이 상호작용 설명, ASCII BEV를
제외하는 실험 프로필이고, 후자는 선택적 JSON 블록 안의 과거 스냅샷 표현만 정한다.
compact payload 자체가 구조화 JSON이므로 `--json-block`과 함께 사용하지 않는다.

### 2. 실제 영상

```bash
pip install -r requirements.txt
python -m traffic_llm.cli \
    --map data/columbus_roads.geojson \
    --vehicle V1:data/v1.mp4:data/v1_tele.csv \
    --vehicle V2:data/v2.mp4:data/v2_tele.csv \
    --vehicle V3:data/v3.mp4:data/v3_tele.csv \
    --hfov 62 --cam-height 1.35 --cam-pitch 2.0 \
    --rate 2 --area "Columbus, OH" \
    --out out/scenes.jsonl --text-out out/scenes.txt --print-first
```

검출·추적을 미리 돌려둔 경우 4번째 필드로 JSON 경로를 붙이면 재사용한다:
`--vehicle V1:data/v1.mp4:data/v1_tele.csv:data/v1_det.json`

### 3. 라이브러리로 사용

```python
from traffic_llm import (
    CameraConfig, PipelineConfig, RoadNetwork,
    TrafficSceneConverter, build_messages, to_text,
)

cfg = PipelineConfig(area_name="Columbus, OH")
network = RoadNetwork.from_geojson("roads.geojson", cfg.lane)
cam = CameraConfig.from_fov(1920, 1080, hfov_deg=62.0, height_m=1.35, pitch_deg=2.0)

conv = TrafficSceneConverter(cfg, network)
conv.add_vehicle("V1", "v1.mp4", "v1_tele.csv", cam)
conv.add_vehicle("V2", "v2.mp4", "v2_tele.csv", cam)

for snap in conv.run(rate_hz=2.0):
    print(to_text(snap, cfg.serialize))
    payload = build_messages(snap, "지금 가장 위험한 상황은?", cfg.serialize)
    # import anthropic; anthropic.Anthropic().messages.create(**payload)
```

## DeepAccident 데이터셋 사용

[DeepAccident](https://deepaccident.github.io/) (CARLA 기반 V2X 사고 예측
데이터셋)를 별도 전처리 없이 입력으로 쓴다. 참조 구현:
[tianqi-wang1996/DeepAccident](https://github.com/tianqi-wang1996/DeepAccident)

```bash
# 시나리오 목록
python examples/run_deepaccident.py --root <DeepAccident_mini> --list

# 한 시나리오를 LLM 입력으로 변환
python examples/run_deepaccident.py --root <루트> \
    --scenario Town03 --type type1_subtype1_accident \
    --out out/scenes.jsonl --text-out out/scenes.txt \
    --eval-out out/eval.jsonl --payload-out out/payload.json --print

# 정답 대비 정확도 평가
python examples/run_deepaccident.py --root <루트> --evaluate

# CARLA 지도 위 BEV 시계열 이미지
python examples/run_deepaccident.py --root <루트> \
    --scenario Town05_type001_subtype0002 --type type1_subtype2_accident \
    --carla-maps C:/path/to/carla_map --bev-dir out/bev
```

라이브러리로:

```python
from traffic_llm import PipelineConfig
from traffic_llm.da_runner import DeepAccidentRunner
from traffic_llm.serialize import to_text

cfg = PipelineConfig()
cfg.deepaccident.observation_mode = "camera"   # 또는 "sensor3d"
runner = DeepAccidentRunner(root, cfg)
res = runner.build("Town03", "type1_subtype1_accident")
print(res.summary())
for snap in res.snapshots(rate_hz=2.0):
    print(to_text(snap, cfg.serialize))
```

### 데이터셋 규격 (역공학 + 공식 컨버터 교차검증)

| 항목 | 내용 |
|---|---|
| 관측자 | `ego_vehicle`, `ego_vehicle_behind`, `other_vehicle`, `other_vehicle_behind`, `infrastructure` (5기) |
| 기록 주기 | 10 Hz, 시나리오당 최대 100 프레임 |
| 레이블 필드 | `cls x y z l w h yaw vx vy id npts vis` (13개), 헤더행 = ego 속도 |
| 레이블 좌표계 | 해당 센서의 **라이다 프레임**. `world = ego_to_world @ lidar_to_ego @ p` |
| 속도 프레임 | 일반 객체는 **센서 프레임**, ego 자기행(`id=-100`)만 **월드 프레임** — 데이터셋 내 비일관성이므로 어댑터가 구분 처리 |
| id 특수값 | `-100` = 자기 자신, `-1` = 추적 불가(중앙값 90m, 기본 제외) |
| 카메라 | 1600×900. Front f=1142.5(hfov 70°), Back f=560(110°), 6방향 |
| `vis` 플래그 | **6개 카메라 OR** 기준. 6대를 모두 쓰면 판정 범위가 일치해 정확한 관문이 되고, 일부만 쓰면 필요조건으로만 유효하다. 어느 쪽이든 카메라별 절두체 컬링 + 깊이정렬 가려짐 판정을 따로 수행 |
| 차선폭 | 실측 3.50m (같은 방향 궤적 간격 분포 최빈, n=33) |

**좌표계**: CARLA 월드는 좌수계(x=동, y=남)다. 본 패키지 ENU(우수계)로는
`e = x, n = -y` 반사 변환하며, 반사이므로 `yaw`·`vy` 부호가 반전된다.
`.xodr` 파일의 y는 이미 부호가 뒤집혀 있어 ENU와 그대로 일치한다.

> **상세 문서**
> - [`docs/payload_modules.md`](../docs/payload_modules.md)
>   — payload 생성에 관여하는 **모듈별 알고리즘** (입출력·판정 규칙·왜 그렇게 했는지)
> - [`docs/payload_text_structure.md`](../docs/payload_text_structure.md)
>   — payload **자연어 설명 부분의 절 구조** (각 절의 형식·필드·읽는 법)
> - [`docs/predicted_path_algorithm.md`](../docs/predicted_path_algorithm.md)
>   — **`PredictedPath` 생성 알고리즘** (기동 판별·확률 산정·좌표 생성)
> - [`docs/predict_model_io.md`](../docs/predict_model_io.md)
>   — **예측 모델의 입력·출력 명세**와 지도학습 데이터 형식
> - [`docs/waypointnet.md`](../docs/waypointnet.md)
>   — **`WaypointNet` 의 내부 구조** (과거 궤적 인코더·후보 집계·등속 잔차·마스킹·
>   기동 헤드·손실)

### 관측 모드

| 모드 | 위치·방위·속도 | 용도 |
|---|---|---|
| `sensor3d` (기본) | 레이블의 3D 위치·yaw·(vx,vy)를 **그대로** | "각 차량이 자기 센서로 주변 상태를 정확히 수집했다"는 전제 |
| `camera` | 2D bbox 에서 단안 역투영으로 **추정** | 단안 인지 오차가 하류에 미치는 영향을 볼 때 |

기본은 `sensor3d` 다. 어느 차량이 보이는지는 **여전히 센서 가시성**으로 정한다
(카메라별 화각, 카메라별 깊이정렬 가려짐, 사거리 120m, 최소 bbox 크기). 값만
정확하고, 보이지 않는 차량은 모른다 — 그래야 V2X 융합이 의미를 갖는다.

카메라는 기본적으로 **calib 에 있는 전부**를 쓴다. `--cameras Camera_Front` 처럼
이름 목록을 주면 그 카메라만 쓴다.

`sensor3d` 에서 보고 값이 레이블과 **정확히** 일치하는지 검증했다
(Town05 사고 364 액터-프레임, Town03 정상 225):

| 지표 | Town05 | Town03 |
|---|---|---|
| 위치 오차 최대 | **0.0000m** | **0.0000m** |
| 방위 오차 최대 | 0.0105° | 0.1160° |
| 속도 오차 최대 | 0.0003 m/s | 0.1069 m/s |
| 180° 반전 | 0 / 1195 (4개 시나리오 합산) | |

정확해지기까지 세 가지를 고쳐야 했다. 값을 GT 로 바꾸는 것만으로는 부족했다.

- **절대 좌표를 그대로 실어 보낸다.** 라이다 프레임 오프셋을 관측차량 자세로
  **2D 회전**해 월드로 되돌리면 경사로에서 pitch/roll 성분이 빠진다. Town05 의
  경사 구간에서 같은 차량을 두 센서가 2.4m 어긋나게 보고했다. 이제
  `Detection.world_xy_measured` 에 전체 4×4 변환 결과를 담아 재구성하지 않는다.
- **관측차량 자신의 위치는 박스 중심으로.** `ego_to_world` 평행이동은 차량
  기준점이고 박스 중심과 최대 0.3m 어긋난다. 다른 액터는 모두 박스 중심이므로
  관측차량만 규약이 달라지면 거리·TTC·조감도 사각형이 어긋난다.
- **정확한 관측은 평균하지 않는다.** 평균은 어떤 센서도 보고하지 않은 값을
  만든다. 대표 하나를 고르는데, **거리보다 최신성이 먼저**다 — 가까운 관측자가
  그 시각에 표본이 없어 이웃 프레임(0.1초 전) 검출을 쓴 경우가 있고, 8m/s
  차량에 0.1초는 0.8m 다. `Observation.measured_t` 로 낡은 관측을 구별한다.

> 데이터셋 자체의 성질 두 가지. (1) 한 시나리오(Town03)에서 차량이 **자기**
> 기록한 속도와 **다른 관측자**가 기록한 속도가 최대 0.6m/s 다르다. 자기 상태는
> 원격 관측이 아니므로 자기 기록을 쓴다. (2) 어떤 시각에는 어느 센서도 표본이
> 없어 0.1~0.2초 낡은 관측만 남는다 (실측 331건 중 1건). 이때는 최신 것을 쓰되
> 정확히 일치하지 않을 수 있다.

### GPS / 도로지도가 없을 때

DeepAccident에는 GPS 채널도 지도 파일도 없다. 어댑터가 데이터에서 생성한다.

- **텔레메트리 합성**: `ego_to_world`를 GPS+나침반 상당으로 변환
  (`deepaccident.synthesize_telemetry`). `gps_noise_m` / `heading_noise_deg` 로
  실제 GPS 오차를 주입할 수 있다(시드 고정, 재현 가능).
- **도로지도**: 우선순위대로 선택
  1. `--road-map` GeoJSON
  2. `--carla-maps <OpenDrive 디렉터리>` → CARLA `.xodr` 임포트
     (`carla_map.load_opendrive`: line/arc/spiral/paramPoly3, driving 차선 수,
     제한속도)
  3. 없으면 **관측 궤적에서 합성** (`roadgen.synthesize_road_network`)

궤적 합성 도로망은 `Road.inferred=True` 로 표시되고, 직렬화 시 LLM 에
"차량이 지나가지 않은 차선·도로는 지도에 없을 수 있음"을 명시한다.
한 방향만 관측된 도로는 중앙선 위치를 알 근거가 없으므로 관측 차선군의 중심을
중심선으로 두고 일방통행으로 표기한다 — **미관측 차선을 임의로 만들지 않는다.**

### CARLA 지도 검증 (Town01–07 `.xodr`)

CARLA 배포판의 `OpenDrive/*.xodr` 을 그대로 읽어(별도 변환 없음) DeepAccident
궤적과 맞춰 보았다. 검증 기준은 **관측차량이 실제로 도로 위에 놓이는지**와
**차로 중심에서 얼마나 떨어지는지**다.

| 타운 | 임포트 도로 | 교차로 | 관측차량 도로매칭률 | 횡오프셋 중앙값 |
|---|---|---|---|---|
| Town01 | 98 | 36 | 100.0% | 2.00m |
| Town02 | 68 | 24 | 100.0% | 2.00m |
| Town03 | 237 | 86 | 100.0% | 1.75m |
| Town04 | 242 | 80 | 100.0% | 1.75m |
| Town05 | 256 | 76 | 100.0% | 1.75m |
| Town06 | 157 | 24 | (mini 데이터에 시나리오 없음) | — |
| Town07 | 220 | 83 | 100.0% | 1.75m |

횡오프셋 중앙값 1.75m 는 CARLA 차선폭 3.5m 의 **정확히 절반** — 차량이 차로
중심을 달린다는 뜻이고, 동시에 `.xodr` 의 y 축이 ENU 북방향과 같은 부호임을
확인해 준다(`flip_y` 불필요). 부호가 뒤집혀 있으면 궤적이 도로 밖으로 나가
매칭률이 무너진다.

Town10HD 는 CARLA 배포판 위치가 달라 `.xodr` 이 없는 경우가 많다 — 그때는
자동으로 궤적 합성 도로망으로 되돌아간다(매칭률 99.5%).

지원 geometry: `line`, `arc`, `spiral`(수치적분), `paramPoly3`.
`laneSection` 의 `driving` 차선만 세고 `type`/`speed` 로 제한속도를 읽는다.

### 신원 융합: 트랙 id 가 전역 유일하다는 전제 (기본값)

이 프로젝트는 **V2X 로 객체 id 를 공유하는 협력형 자율주행**을 전제한다. 따라서
`FusionConfig.global_track_ids` 기본값은 **True** 다. DeepAccident 의 트랙 id 도
CARLA actor id 로 전 관측자에서 전역 유일하므로 같은 조건이다.

이 전제 아래에서 트랙 id 를 신원의 근거로 쓴다:

- 조회를 (관측자, 트랙) 쌍이 아니라 **트랙 id 만**으로 한다. 쌍으로 조회하면
  다른 관측자가 같은 차량을 처음 볼 때 키가 없어 새 액터가 발급되고, 한 물체가
  관측자 수만큼 쪼개진다.
- 가려짐으로 `track_timeout_s` 보다 길게 끊겼다 돌아와도 같은 액터로 이어붙인다.
- 관측차량 본체 판정도 자기 트랙 id 로만 한다 — 거리 게이트(최대 25m)를 함께
  돌리면 관측차량 옆을 지나는 다른 차량이 본체로 흡수된다.

`mini` Town03 사고 시나리오에서 이 규칙을 적용해 액터 29개(정답 차량 19대가
쪼개진 상태) → **19개, 정답 차량과 1:1** 이 되었다.

id 재사용에는 두 관문이 걸린다.

- **동역학 그룹 일치** — `{car, van, truck, bus, motorcycle}` 와 `{person, bicycle}`
  는 서로 같은 물체가 될 수 없다. 없으면 근접한 보행자 관측이 오토바이 트랙을
  물려받아 **100km/h 로 달리는 보행자**가 나온다(실데이터에서 관측된 결함).
- **운동 타당성** — 직전 위치에서 지금 위치까지 필요한 속도가 클래스별 상한
  (`max_speed_by_class`, 보행자 8 m/s 등)을 넘으면 다른 물체다.
  이 관문은 `global_track_ids=False` 일 때만 돈다 — 전역 유일하면 같은 id 가 곧
  같은 물체이므로, 관문을 걸면 가려짐으로 오래 끊긴 차량이 여러 액터로 쪼개진다.

#### 전제가 깨지는 경우 — 조용히 틀리지 않게

**카메라별 독립 추적기**(YOLO+ByteTrack 등)는 관측자마다 1번부터 번호를 매기므로
서로 다른 차량이 같은 트랙 id 를 갖는다. 그 상태로 id 만 믿으면 **수백 m 떨어진 두
차량이 한 액터로 병합되고 하나가 조용히 사라진다** (합성 실험: 360m 떨어진 두 차량
→ 액터 1개).

그래서 같은 id 로 묶인 관측이 `id_collision_dist_m`(기본 150m)보다 흩어져 있으면
전제 위반으로 보고 **위치 기준으로 쪼갠 뒤** `GlobalTrackRegistry.id_collisions` 를
올린다. 카운터가 0 이 아니면 `global_track_ids=False` 로 두어야 한다는 신호다.

임계값은 실측으로 정했다. DeepAccident 를 `camera` 모드(단안 역투영)로 돌려
**id 가 진짜 전역인** 다관측 클러스터 355건의 위치 산포를 재면 다음과 같다.

| | p50 | p90 | p99 | 최대 |
|---|---|---|---|---|
| `sensor3d` (GT) | 0.0m | 0.0m | 0.0m | 0.0m |
| `camera` (단안) | 5.6m | 12.4m | 39.9m | **66.0m** |

정상 불일치가 최대 66m 이고 진짜 id 충돌은 수백 m 이므로 그 사이에 둔다. 40m 로
잡았을 때는 실데이터에서 3건이 잘못 쪼개졌다(고유 액터 23 → 27개). 150m 에서는
`sensor3d`·`camera` 양쪽 모두 오탐 0건이면서 합성 충돌은 잡힌다.

### 속도·방위 추정: 하나의 적합, 세 개의 관문

> 아래는 **`camera` 모드**(단안 추정)에만 해당한다. 기본 `sensor3d` 에서는
> 위치·방위·속도가 레이블 값이므로 이 추정기가 개입하지 않는다 (운동학 추정기는
> 값이 없을 때만 채운다). 일반 영상 입력에서는 항상 이 경로를 탄다.

속도와 방위는 모두 최근 궤적의 **최소자승 적합**에서 나온다. 프레임 간 차분은
위치 잡음을 그대로 속도로 옮긴다 — 48m 거리의 잡음이 0.5초 간격에 만드는 겉보기
속도는 20m/s 를 넘어, 사실상 정지한 차량이 63km/h 로 보고되고 방위가 180°
뒤집힌다 (Town01 V001 에서 관측). 창 전체를 적합하면 잡음이 표본 수만큼 상쇄된다.

**오차 기제가 둘이므로 불확실도도 둘이다.** 이걸 하나로 뭉치면 반드시 한쪽이
틀린다.

| 무엇을 판정 | 기준 불확실도 | 왜 |
|---|---|---|
| 속도 **크기** | 적합 **잔차** (`sigma_v`) | 위치 오차의 큰 부분은 창 안에서 거의 일정한 측거 편향이고, 그것은 기울기에서 상쇄된다. 이 트랙이 실제로 얼마나 튀는지는 잔차가 말해 준다 |
| 방위·진행 **방향** | 관측거리 기반 **p90** (`sigma_pos`) | 측거 편향은 **매끄럽게 표류**하므로 잔차를 키우지 않으면서 기울기의 부호를 뒤집는다. 잔차 기준으로 바꾸자 180° 반전이 9%→17% 로 늘었다 |

`sigma_pos` 는 실측 p90 위치오차에 맞춘 `max(5.0, 0.06r + 0.0015r²)` 이다
(0-15m 5.2m, 45-70m 9.1m, 70-130m 23.4m — 원거리에서 선형보다 빠르게 커진다).
**중앙값이 아니라 상위 꼬리를 쓴다** — 관문은 오탐을 막아야 하기 때문이다.

관측거리 70m 를 넘으면 궤적으로 **방위를 주장하지 않는다**. 지평선 근처 역투영은
bbox 가 매끄럽게 흘러도 속도가 수십 m/s 로 잘못 나오고, 그 오차가 매끄러워서
잔차로도 거리척도로도 걸러지지 않는다 (실측: 73m 트럭의 실제 2m/s 가 24.6m/s,
방향은 반대로). 속도는 잔차 기준이라 이 제약과 무관하게 계속 보고한다.

여기에 클래스별 관문이 더 붙는다. **신원 관문 상한과 보고 상한은 다른 값이어야
한다**: 신원 관문(`max_speed_by_class`)은 잡음을 흡수해야 하므로 넉넉하고
(보행자 8 m/s), 보고 상한(`report_speed_by_class`)은 실제 지속 주행 속도다
(보행자 2.5 m/s). 겸하면 위치 잡음이 그대로 "28km/h 로 달리는 보행자"로 나가고,
그 잡음 궤적이 조감도에서 인도 위를 지그재그로 가로지르는 선으로 그려진다.

### 진행 방향은 지도에서, 각도도 지도에서

도로에 매칭된 차량의 방위는 **도로축**을 쓴다 (`heading_source='road'`). 차량은
거의 항상 차로를 따라가므로 지도의 도로축이 단안 궤적보다 정확하다 — 레이블 yaw
대비 방위오차 중앙값이 15.6° → **0.3°** 로 떨어진다. 보행자·자전거는 차로를
따르지 않으므로 제외한다.

그러면 남는 문제는 **1비트**, 순방향인지 역방향인지다. 판정 근거를 실측 신뢰도
순으로 쓴다.

| 근거 | 반전율 (mini 실측) | 조건 |
|---|---|---|
| 텔레메트리·3D 센서 방위 | 0% | ego 또는 `sensor3d` 모드 |
| **통행측** (`lane_side`) | 7% | 실측 지도(`inferred=False`), 중심선에서 0.35차선 이상, **차도 폭 안** |
| 궤적 축방향 부호 (`trajectory`) | 9~14% | 축방향 이동거리 > 2·`sigma_pos` |
| 일방통행 표기 (`oneway`) | 19~25% | 위 셋이 모두 불가할 때만 |

- **통행측**: 우측통행에서 진행차로는 중심선 우측이다. 지도 기하만 쓰므로 원거리
  측거 편향에 영향받지 않는다. 단 **차도 폭 안**에 있어야 한다 — CARLA 의 분리
  도로는 방향별로 별개 road 이므로 기준선에서 차도 폭보다 멀리 있는 차량은 나란한
  다른 도로에 속한다 (실측 반전율이 |횡오프셋| 7m 이하 2~12% 에서 7m 초과 54% 로
  급증). 합성 도로망은 중심선 자체가 추정값이라 쓰지 않는다.
- **방위는 도로 선택에만, 방향에는 안 쓴다.** 뒤집힌 궤적 방위로 방향을 정하면
  그 반전이 그대로 확정된다. 도로 선택 점수는 방향과 무관한 `min(정방향차,
  역방향차)` 를 쓴다 — 뒤집힌 추정 때문에 올바른 도로가 탈락하지 않도록.
- 어느 근거로도 정하지 못하면 **방위를 미확정으로 둔다**. 절반의 확률로 180°
  틀린 값을 내보내는 것보다 낫고, 조감도는 도로축에 맞춘 사각형(점선 테두리,
  화살표 없음)으로 그리므로 표현 손실도 없다 — 사각형은 180° 뒤집어도 같은
  모양이다. `RoadPlacement.direction_confident=False` 로 표시되고, 텍스트에는
  "(진행방향 미확정)", JSON 에는 `road.direction_confident` 로 넘어간다.

> **OpenDRIVE 임포트 버그 하나**: driving 차선이 모두 기준선 **왼쪽**(양수 id)에
> 있는 일방통행 도로는 통행방향이 −s 다. 차선 수만 `lanes_forward` 로 옮기고
> 폴리라인을 그대로 두면 진행 방위가 180° 반대가 된다 (실측 반전율 57%).
> 임포터가 폴리라인을 뒤집어 '정방향 = 통행방향'을 유지한다.

### 인프라(노변 센서) 취급

인프라는 교통 참여자가 아니므로 `SceneSnapshot.infrastructure` 로 분리하고,
차량 목록·BEV에서 제외한다. 섞으면 LLM 이 정차 차량으로 오해한다.
인프라의 관측은 관측차량 사각지대를 보완하며, 융합 시 같은 차량을 본 관측자
목록에 함께 기록된다.

> 인프라 지상고는 `ego_to_world[2,3]` 에 들어 있고 `lidar_to_ego` 는 항등행렬이다.
> `lidar_to_ego` 만 보면 지상고가 0이 되어 IPM 노면 교점이 원점으로 붕괴한다
> (모든 거리 추정이 0). 어댑터는 변환 사슬 전체를 통과시키고,
> `CameraModel` 은 지상고 ≤ 0 을 예외로 막는다.

### 사고 정답 누수 방지

`meta` 의 충돌 정보(발생 여부·주체·강도·시각)는 사후에만 알 수 있는 정답이다.
사고 **예측** 과제에서 LLM 입력에 넣으면 정답 누수가 되므로:

- `ScenarioContext.available` — 주행 시점에 알 수 있는 것만(기상, 도로 형태)
- `ScenarioContext.ground_truth` — 정답. `to_json()` 은 **기본 제외**
- 채점은 `to_evaluation_record()` 로 별도 스트림에 기록

```python
to_json(snap, cfg.serialize)                            # 정답 없음 (LLM 입력)
to_json(snap, cfg.serialize, include_ground_truth=True) # 명시 요청 시만
to_evaluation_record(snap, cfg.serialize)               # 채점용
```

## 사고 예측 질의 (누적 관측 윈도우 → 미래 N초)

관측 윈도우마다 **마지막 관측 시점 이후 미래 N초를 1초 구간별로** 사고 발생 여부·
관련 차량·이유를 묻는 payload 를 생성한다. 정답은 별도 파일로 관리해 LLM 답변과
대조한다.

```bash
python examples/run_deepaccident.py --root <루트> \
    --scenario Town10HD_type001_subtype0001_scenario00014 \
    --type type1_subtype1_accident \
    --windows-dir out/windows --window 5 --stride 1 --horizon 5
```

### 창은 고정 길이로 이동한다 (기본)

`--window` 는 창의 고정 길이다. 첫 창부터 `--stride` 간격으로 이동하며,
짧은 선두 창은 버린다.

```
--window 5 --stride 1 → 0-5, 1-6, 2-7, 3-8, …
스냅샷 수(2Hz)              11   11   11   11
```

expanding-prefix가 필요하면 명시적으로 `--warmup`을 추가한다.
아래 누적 창 개수 표는 과거 expanding 실행과의 비교용이다.

| 분할 | 충돌 | 고정 길이 | 누적 창 |
|---|---|---|---|
| `DeepAccident_mini` | t=9.2s | 5개 | 9개 |
| `train` | t=5.6s | 1개 | 5개 |
| `val` | t=4.0s | 0개 | 3개 |

`val` 은 충돌이 t=4.0s 라 5초 창이 하나도 만들어지지 않았다(전부 "충돌이 관측 구간
안"으로 버려짐). 짧은 창에서는 속도·기동이 `미확정` 으로 나올 수 있다 — 2Hz 에서
표본 4개(1.5초)가 모여야 판정한다. 이것도 관측 조건의 일부다.

`--no-warmup`은 이전 명령과의 호환성을 위해 유지되며 현재 기본값과 동일하다.

### 출력 파일과 경로

산출물 경로에 **출처**를 새긴다 — `<경로>/<분할>[/<라벨>]`. 한 폴더에 여러 분할의
결과가 섞이면 어느 데이터로 만든 것인지 알 수 없다.

```
out/windows/DeepAccident_mini/          ← --windows-dir out/windows
  llm_payload_0-1.json    ← LLM 요청 본문 (정답 없음)
  ground_truth_0-1.json   ← 채점용 정답
  window_0-1.txt          ← 사람이 읽는 사본
  llm_payload_0-2.json / ground_truth_0-2.json / window_0-2.txt
  ...
  manifest.json           ← 윈도우 목록·설정·요약·응답 스키마
  responses/              ← ask_llm.py 가 쓰는 응답·채점 (재생성이 지우지 않는다)
```

경로는 `<base>/<분할>/<accident|normal>[/<label>]` 이다 — 윈도우와 BEV 가 **같은
구조**를 쓴다. 같은 시나리오의 사고/정상 트레이스는 이름이 같아서 outcome 을 폴더로
갈라 두지 않으면 서로를 덮어쓴다. `--out-label` 로 그 아래 구분(언어·provider 등)을
더할 수 있고, `--flat-out` 으로 자동 구성을 끈다.

재생성은 **자기가 만드는 파일만** 지운다(`llm_payload_*`, `ground_truth_*`,
`window_*`). 디렉터리째 지우면 `responses/` 의 LLM 응답이 날아가므로 그러지 않는다.

### 시간 구간 규약

**k번째 구간 = 마지막 관측 시점 이후 (k-1, k] 초.** 질문·정답·채점이 모두 같은
정의를 쓴다. 1초 단위로 "그 시점"을 물으면 경계가 모호해지므로 구간으로 정의했다
(하한 배타, 상한 포함). 예: t_end=6초, N=5 → k=1은 (6,7], k=5는 (10,11].

### payload 구성

1. **윈도우 개요** — 구간, 스냅샷 수, 배경(기상·도로형태), 지도 출처
2. **노변 인프라 센서** — 교통 참여자와 분리
3. **차량별 시간 경과** — 위치·속도·도로/차선·기동을 `history_stride_s` 간격으로
4. **윈도우 동안 서로 접근한 차량 쌍** — 거리 변화·접근율·선형연장 접촉시간
   (관측된 거리 변화율의 연장이며 경로 예측이 아님을 명시)
5. **마지막 관측 시점 상세** — 상호작용(헤드웨이·TTC·교차로 상충), BEV 개략도
6. **구조화 JSON** — 궤적 요약 + 마지막 스냅샷 (`--history-mode full`이면 전 스냅샷)
7. **질문** — 구간 정의와 답변 규칙

### 구조화 JSON 블록은 선택 (기본 꺼짐)

`--json-block` 으로 켠다. 끄는 것이 기본인 이유는 **자연어 브리핑과 내용이
겹치기** 때문이다 — 다섯 블록(`window`, `last_snapshot`, `closing_pairs`,
`perception`, `trajectories`) 전부가 브리핑의 대응 절과 짝을 이루면서 입력의
79% 를 차지한다.

| | payload 크기 (Town05, 창 9개 평균) |
|---|---|
| JSON 제외 (기본) | **23,379자** |
| JSON 포함 | 108,376자 (4.6배) |

**끄면 정보가 사라지는 값은 미리 자연어로 옮겼다.** 그러지 않으면 옵션이 아니라
조용한 손실이다. 옮긴 것은 둘이다.

- **방위각 숫자** — 액터 행에 `방위 346°` / `heading 346 deg`. 도로 진행방향
  라벨("북서행")은 8방위로 뭉개져 있어 충돌 기하를 따질 수 없다. 정지 차량 옆을
  지나갈 때 측방 여유가 방위 차이 몇 도에서 갈린다.
- **지도 규약** — `※ 지도 규약: 우측통행 | 1차선은 중앙선쪽 | 차선폭 3.50m |
  도로 256개·교차로 76개`. 통행측과 차선번호 기준을 모르면 차선 번호와 좌/우
  표현을 반대로 읽는다.

단일 스냅샷 경로(`build_messages`)도 `SerializeConfig.include_json_block` 을 따르고,
명시 인자(`include_json=True`)가 설정을 덮어쓴다.

### 수록 상한은 관측량에 따라 정한다

`--max-actors`·`--max-closing-pairs` 를 주지 않으면 장면의 차량 수에서 계산한다.

| 항목 | 규칙 | 하한 | 상한 |
|---|---|---|---|
| 수록 차량 | 관측된 전부 | 15 | 40 |
| 접근 쌍 | 차량 수 × 0.5 | 6 | 20 |

**고정 상한은 관측이 늘어날수록 담는 비율만 줄인다.** 카메라를 6대로 늘려 차량이
24→26대, 접근 쌍이 43→88개가 됐을 때 상한 6 고정이 **사고 당사자 쌍을 목록 밖으로
밀어냈다** — 모델이 놓친 게 아니라 payload 에 없었다.

여기에 순위와 무관한 안전장치가 하나 더 있다. **접촉이 임박한 쌍은 상한을 넘겨도
담는다** — `linear_contact_s` 가 질문 지평의 절반(기본 `horizon_s × 0.5` = 2.5초)
이하면 순위를 무시하고 포함한다. 순위로만 자르면 새로 보이게 된 무관한 쌍이 정작
임박한 쌍을 밀어낼 수 있다.

실측(Town05, 충돌 t=9.2s)에서 당사자 쌍의 순위는 충돌이 가까워질수록 단조롭게
개선되고, 접촉 2.5초 안에 드는 윈도우부터 반드시 포함된다.

| 윈도우 | 충돌까지 | 당사자 쌍 순위 | 선형 접촉 | 수록 |
|---|---|---|---|---|
| 0-5 | 4.2s | 46 / 85 | 5.8s | 제외 (지평 밖) |
| 1-6 | 3.2s | 22 / 95 | 2.5s | 포함 (임박 규칙) |
| 2-7 | 2.2s | 8 / 88 | 1.1s | 포함 |
| 3-8 | 1.2s | 2 / 91 | 0.8s | 포함 |
| 4-9 | 0.2s | 1 / 80 | 0.6s | 포함 |

윈도우별 실제 적용값은 manifest 의 `max_actors_effective` /
`max_closing_pairs_effective` 에 남는다 — 조용히 잘라내지 않는다.

토큰이 부담되면 값을 명시해 고정할 수 있다 (`--max-actors 15
--max-closing-pairs 6`). 실측으로 자동 상한은 payload 를 늘린다 (고정 15대·6쌍 대비 창당 약 25%).

응답은 **구조화 출력**(`output_config.format`)으로 스키마를 강제하므로 자동 채점이
가능하다:

```json
{"predictions": [{"k": 1, "interval_s": "(6, 7]", "accident_expected": false,
                  "involved_actor_ids": [], "reason": "...", "confidence": "low"}],
 "overall_assessment": "...", "data_limitations": "..."}
```

### 응답 형식 검사

채점은 `k` 로 정답과 짝지으므로 **형식 오류가 점수에 거의 드러나지 않는다** —
중복된 `k` 는 조용히 무시되고, 빠진 `k` 는 정답이 '사고'일 때만 FN 이 된다.
실제로 모델이 같은 구간을 두 번 답하고 그 항목에 "동일 구간 중복 표기 방지용
확인 항목"이라 적어 둔 응답이 있었는데, 채점 결과만 봐서는 보이지 않았다.

`score_response()` 가 `response_issues` 를, `aggregate_scores()` 가
`response_issue_counts` 를 함께 낸다. `ask_llm.py` 는 이것을 화면에 찍고 응답
파일에도 저장한다.

| 종류 | 뜻 |
|---|---|
| `duplicate_k` | 같은 구간을 여러 번 답했다 (채점은 첫 항목만 쓴다) |
| `missing_k` | 답하지 않은 구간이 있다 |
| `unexpected_k` | 질문에 없는 구간을 답했다 |
| `missing_k_field` | `k` 필드가 없는 항목이 있다 |

**스키마 제약으로 막지 않는 이유.** `minItems`/`maxItems`/`minimum` 을 쓰면
구간 수를 강제할 수 있지만 **OpenAI strict 모드가 그 키워드를 지원하지 않아**
provider 별로 스키마가 갈린다 — "내용은 provider 와 무관하게 같다"는 이
프로젝트의 규약이 깨진다(테스트로 고정해 둔 규약이다). 대신 스키마 설명문에
구간 수를 적어 요구하고(`"이 질문의 구간 수는 N=5 이므로 항목이 정확히 5개여야
한다"`), 실제 위반은 채점 단계에서 잡는다. 항목 간 유일성은 애초에 JSON Schema
로 표현할 수 없다.

### 정답 파일과 채점

```python
import json
from traffic_llm.accident_qa import aggregate_scores, score_response

gt = json.load(open("out/windows/ground_truth_1-6.json", encoding="utf-8"))
resp = json.loads(llm_response_text)          # 구조화 출력 결과
s = score_response(resp, gt)
print(s["counts"], s["accuracy"], s["recall"])
print(aggregate_scores([s, ...]))             # 여러 윈도우 합산
```

정답의 `expected[k]`는 구간별 `accident_expected`와 `involved_vehicles`를 담는다.
차량은 **CARLA id 단위로 묶어** 두므로, 융합이 한 물리 차량을 여러 actor id로
쪼갠 경우에도 그중 하나만 지목하면 맞은 것으로 채점한다(`vehicle_recall`).
관측되지 않은 충돌 주체는 `unobserved_carla_ids`로 표시해, LLM이 이름을 댈 수
없는 차량을 감점 근거로 쓰지 않도록 한다.

#### 조기 예측은 정답으로 인정한다 (`--early-credit`)

`--early-credit`는 사고 전에 허용할 최대 lead time이다. 현재 main evaluation
protocol은 **5초**를 사용한다. 라이브러리의 `WindowConfig` 기본값(2초)은 기존
ground-truth와 테스트의 재현성을 위해 유지하므로, 실험 실행 시에는 반드시
`--early-credit 5`를 명시하거나 main CLI 기본값을 사용한다.

사고 예측의 목적은 회피할 시간을 버는 것이므로, 실제 충돌보다 조금 이른 경보는 맞은
것으로 본다. **늦은 예측은 여전히 오답**이다 — 이미 일어난 사고를 예측이라 부를 수
없다(진단용으로 `FP_LATE` 로 표시).

```
인정 범위 = [ (충돌시각 − x) 가 속한 구간, 충돌 구간 ]
```

경계를 구간 번호 산술이 아니라 **충돌 시각**으로 정한다 — 구간이 1초 폭이라 충돌이
구간 안 어디에 있느냐로 경계가 달라진다.

`binary_bucket`·`strict`·`early`의 confusion metric은 bucket 단위로 센다. 예를
들어 충돌이 네 번째 bucket에 있고 응답이 `TN TN TN TP TN`이면 분모는 5,
TP=1/TN=4이다. `binary_window`는 창당 한 건, `binary_scenario`는 시나리오당
한 건만 센다.
`early`는 인정 범위 안의 대표 경보 bucket 하나를 양성 target으로 옮기고 나머지
bucket도 계속 TN/FP로 센다. 따라서 범위 전체를 칠해 TP를 중복해서 얻을 수 없다.

정답 파일은 넓히지 않는다 — `accident_expected=True` 는 실제 구간에만 두고 인정
범위는 `credit` 블록으로 적어, 파일만으로 채점이 재현된다.
`score_response(resp, gt, early_credit_s=x)` 로 다른 기준으로 재채점할 수 있다.

### 비교 대상 — DeepAccident

논문의 비교 대상인 DeepAccident 사고 예측은 **`deepaccident_replicate/`** 에 따로 두었다.
공개 저장소가 이 머신에서 돌지 않아(가중치 미배포 + CUDA 10.2/mmcv 가 sm_120 을 지원
하지 않음) 방법을 다시 구현했고, 비교 대상이 본체 파이프라인을 오염시키지 않도록
패키지를 분리했다 — 그쪽이 `traffic_llm` 에 의존하고 그 반대는 아니다.

산출물은 `traffic_llm.accident_qa.score_modes` 를 그대로 통과하므로 LLM 응답과 **같은
표**에서 읽힌다. 자세한 것은 `deepaccident_replicate/README.md`.

#### 다중 기준 채점 (`--modes`)

같은 응답을 여러 기준으로 **동시에** 채점한다. 논문 비교용 주 지표는 시나리오당
한 건인 `binary_scenario`이고, window/bucket과 event timing은 별도 진단 축이다.

| 모드 | 판정 단위 | 기준 |
|---|---|---|
| `binary_scenario` | **scenario** | 한 영상의 유효 window 중 하나라도 사고를 감지하면 TP |
| `binary_window` | **window** | 고정 observation window 하나를 TP/FP/TN/FN 한 건으로 집계 |
| `binary_bucket` | **bucket** | 실제 사고 bucket과 일치하는지; 모든 bucket을 TP/FP/TN/FN으로 집계 |
| `strict` | **bucket** | 실제 충돌 bucket만 양성 (`early_credit_s=0`) |
| `early` | **bucket** | 인정 범위의 대표 bucket을 양성 target으로 사용 (기본 2초) |
| `weighted` | bucket + 거리 | 첫 경보 거리의 부분 점수와 bucket confusion을 함께 기록 |

```python
from traffic_llm.accident_qa import aggregate_modes, score_modes

rows = [score_modes(resp, gt) for resp, gt in pairs]
agg = aggregate_modes(rows)
```

```bash
.venv/bin/python examples/ask_llm.py <창 디렉터리> --rescore --score --modes all
.venv/bin/python examples/baselines_accident_qa.py ... --modes all
```

`ask_llm.py` 는 `scores_modes.json` 을 따로 쓴다 (`scores.json` 은 그대로 둔다).

`binary_scenario`는 한 시나리오의 겹치는 window를 먼저 합친다. 사고 시나리오에서는
사고가 미래 horizon에 들어오는 유효 window 중 하나라도 경보하면 TP 1개이고, 정상
시나리오에서는 유효 window 중 하나라도 경보하면 FP 1개다. 따라서 같은 영상의
여러 window가 TP를 중복 생성하지 않는다. `binary_window`는 관측 시점별 결과를
보여주며, `binary_bucket`은 한 window 안의 `TN TN TN TP TN` 같은 상세 결과를
보여준다. 과거 산출물의 `binary` 키는 bucket 결과를 가리키는 호환 별칭이다.

**`weighted` 는 늦은 경보에 부분 점수를 준다.** `mean_score`는 첫 경보와
충돌 bucket 사이 거리로 계산하고, `false_alarm_rate`는 다른 모드와 마찬가지로
bucket-level FP/(FP+TN)이다. 기존 window-level 오경보율은
`window_false_alarm_rate`로 별도 보존한다.

```
Δ = 첫 경보 구간 − 충돌 구간
Δ ≤ 0 (이른 경보)  →  1.0 − early_decay·|Δ|   (기본 early_decay=0 → 1.0)
Δ > 0 (늦은 경보)  →  1.0 − late_decay·Δ      (기본 0.2 → 0.8 / 0.6 / 0.4)
```

대표를 **첫 경보**로 잡는 것이 중요하다. `score_response` 의 `credited_k`(정확히
맞힌 칸이 있으면 그것)를 쓰면 범위를 칠한 답이 항상 Δ=0 이 되어 가중이 무의미해진다.
실제로 시스템이 움직이는 시점도 첫 경보다.

충돌이 지평 밖인 창에는 부분 점수가 정의되지 않는다(`applicable=False`). 그 창은
오경보 여부만 세어 `false_alarm_rate` 로 따로 낸다 — `mean_score` 는 사고 창만 보므로
이 둘을 같이 읽지 않으면 "무조건 경보" 가 만점처럼 보인다.

**모드 간 숫자를 가로질러 비교하면 안 된다.** 판정 단위가 다르다 —
`binary_scenario`는 시나리오당 1건, `binary_window`는 창당 1건,
`binary_bucket`/`strict`/`early`는 bucket 단위다. 모드가 알려 주는 것은
**같은 응답이 기준을 바꾸면 얼마나 달라지는가**이지 절대값이 아니다.

#### 데이터 범위 밖 구간은 채점하지 않는다

질문은 항상 미래 N초를 묻지만, 데이터가 9.2초에 끝났다면 (10,11] 구간의 '사고 없음'
은 확인된 사실이 아니라 질문 형식을 맞추려고 채운 값이다. 그것으로 점수를 주면
창이 데이터 끝에 가까워질수록 TN 이 늘어 **아무것도 예측하지 않는 모델이 높은 점수**를
받는다.

정답의 각 구간에 `scorable` / `unscorable_reason` 을 둔다.

| 조건 | 판정 |
|---|---|
| 구간 전체가 데이터 안 | 채점 |
| 데이터 끝을 넘지만 **충돌이 그 구간에 있음** | 채점 (답을 안다) |
| 그 밖 | **제외** (`beyond_data_end`) |

두 번째 규칙이 없으면 정답 구간 자체가 빠진다 — 충돌 9.2s, 데이터 끝 9.5s 인 `3-8`
창의 실제 사고 구간은 (9,10] 로 데이터 끝을 넘는다.

Timing/bucket 평가에서는 제외된 구간을 집계에 넣지 않고 `UNSCORABLE` 로 기록한다.
단, `binary_window`는 window 정답이 확정되면 요청 horizon 전체의 경보를 사용한다.
관측 0~5초, 실제 사고 및 녹화 종료 6초, 예측 사고 8초이면 binary는 TP다.
8초의 timing label은 미확인이므로 bucket 평가에서는 제외하고, 놓친 실제 사고
bucket은 FN으로 남긴다. `binary_scenario`에도 이 window의 TP가 반영된다.
사고가 확인되지 않고 미래 녹화가 부족한 window는 binary에서도 제외하며,
요청 horizon 밖의 경보는 인정하지 않는다. 결과에
`scored_buckets` / `excluded_buckets` 가 함께 나온다.

실측 (충돌 t=9.2s, 데이터 끝 9.5s):

| 창 | 실제 k | 인정 범위 | 채점 구간 | 제외 |
|---|---|---|---|---|
| 0-5 | 5 | 3~5 | 1~5 | — |
| 2-7 | 3 | 1~3 | 1~3 | 4,5 |
| 3-8 | 2 | 1~2 | 1~2 | 3,4,5 |
| 4-9 | 1 | 1~1 | 1 | 2,3,4,5 |

`aggregate_scores` 는 사건 단위 요약(`events`: exact/early/none, `detection_rate`,
`mean_lead_s`)을 함께 낸다.

두 기준이 **다른 것을 센다**는 점을 유의한다. 실측 예에서 조기 인정은 점수를 올리고
(2-7 창 0.60 → 1.00), 확인 불가 구간 제외는 내린다(3-8 창 0.80 → 0.00) — 후자의 0.80
은 관측되지 않은 구간의 TN 4개가 만든 점수였다.

### 충돌 시각 추정

meta는 충돌 주체 id와 강도만 주고 **시각은 주지 않는다**. 사고 분할은 기록이
충돌 시점에 끊기므로 마지막 프레임이 근사값이지만, 두 주체의 궤적 최근접 시점을
찾으면 더 정확하다(`deepaccident.estimate_collision`). 접촉 기준은 두 객체의
**절반 길이 합 + 2m** 다 — 고정 임계값을 쓰면 트럭(길이 8.5m)이 걸린 충돌에서
중심 간 6.4m를 미접촉으로 오판한다. mini 데이터 8건 중 7건이 궤적으로 확정되고,
1건은 상대 id가 미기재(`0`)여서 마지막 프레임으로 되돌아간다.

### 정답 누수 차단 (사고 예측 과제의 전제)

| 누수 경로 | 차단 방식 |
|---|---|
`scenario.id`에 분할명(`..._accident`) | 불투명 토큰으로 치환 (`scn_<hash>`). 실제 id는 정답 파일에만 |
충돌 정보 | payload에 넣지 않음. `to_json`은 정답을 기본 제외 |
충돌이 관측 윈도우 안에 든 윈도우 | 예측 문제가 아니므로 기본 제외 (`drop_windows_after_collision`) |

테스트가 payload에서 `type1_subtype1_accident`, `collision`, `ground_truth`,
충돌 시각 문자열이 나타나지 않음을 검사한다.

> **평가 설계 시 유의점**: 사고 분할은 기록이 충돌 시점에 끊기므로 윈도우 수가
> 정상 분할보다 적고, 마지막 윈도우는 항상 충돌 1~2초 전이다. 정상 분할
> 시나리오를 음성 표본으로 함께 생성해 균형을 맞추는 것을 권한다.

### mini 데이터 실측 정확도

`DeepAccident_mini` 4개 시나리오(Town01/03/04/05, 스냅샷 80개, 2Hz)에서
CARLA actor id 기반 **신원 매칭**으로 측정:

| 지표 | `sensor3d` (기본) | `camera` |
|---|---|---|
| 위치오차 중앙값 | **0.00m** (최대 0.19m) | 2.26m (최대 68.8m) |
| 반경편향 중앙값 | +0.00m | +0.22m |
| 횡방향오차 중앙값 | 0.00m | 1.03m |
| 방위오차 중앙값 | **0.0°** | 5.9° |
| 속도오차 중앙값 | 0.01 m/s | 0.44 m/s |
| 도로매칭률 | 94.0% | 96.6% |
| 진행방향 확정률 | **100%** | 55% (실측 지도) |
| 180° 반전 | **0** | 42 / 413 |

`--evaluate` 는 두 모드를 나란히 돌린다. `sensor3d` 의 잔여 최대 0.19m 는
어느 센서도 그 시각에 표본이 없어 0.2초 낡은 관측을 쓴 1건에서 온다.

방위 정답으로는 레이블 `yaw`(`deepaccident.ground_truth_headings`)를 쓰는 것이
옳다 — 정답 궤적을 0.1초 간격으로 차분하면 저속 객체의 '정답' 자체가 잡음이다.
`camera` 열의 방위·진행방향 수치는 CARLA `.xodr` 을 준 경우다.

거리 구간별 위치오차 중앙값(`camera`): 0-15m 2.07m, 15-30m 1.95m,
30-45m 2.53m, 45-70m 2.15m, **70-130m 7.27m**.

`sensor3d` 가 전 구간 0.00m 인 것은 좌표변환·융합·지도매칭 사슬이 정확하다는
뜻이다. `camera` 의 오차는 단안 역투영 자체의 한계다. 재현:

```bash
python examples/run_deepaccident.py --root <루트> --evaluate
```

### 단안 거리 추정 방식 선택 근거

DeepAccident 정답과 비교해(n≈9,900, 절단 bbox 제외) 상대오차 중앙값:

| 거리구간 | IPM(접지점) | 높이 기반 | 폭 기반 |
|---|---|---|---|
| 15-30m | **-9.8%** | -19.8% | -39.4% |
| 45-70m | **-3.8%** | -13.4% | -38.7% |

**폭 기반 추정이 크게 나쁘다**: 2D bbox 는 3D 박스 8개 코너의 축정렬 헐이므로
차량을 비스듬히 보면 겉보기 폭이 (길이+폭)/√2 까지 커지고, 실폭으로 나누면
거리가 최대 2.5배 과소추정된다. 높이는 방위와 거의 무관해 IPM 을 쓸 수 없을 때의
보조 단서로 쓴다.

여기에 **근접면→중심 보정**(bbox 하단은 3D 박스의 가장 가까운 접지 모서리이므로
중심은 `0.25×(폭+길이)` 만큼 더 멀다)을 적용해 15-70m 편향을
-9.8%/-3.8% → -5.2%/-0.8% 로 줄였다.

## 경로 예측기 교체 (규칙 기반 / 학습 모델)

경로 예측은 기본이 **지도 제약 규칙**이고, 같은 입력을 받는 학습 모델로 바꿀 수 있다.

```python
from traffic_llm.config import PipelineConfig
from traffic_llm.predict_model import TorchPredictor

cfg = PipelineConfig()
cfg.predictor = TorchPredictor("model.pt")     # 지정하지 않으면 규칙 기반
```

> **`run_deepaccident.py` 의 기본 예측기는 학습된 좌표 회귀 모델이다**
> (`out/predict_model/waypointnet_best.pt`). 같은 시험셋에서 규칙 기반보다 ADE 가 24%
> 낮고 후보 집합의 상한도 넘기 때문이다(§5.7). 그 파일이나 torch 가 없으면 규칙 기반으로
> 내려가고 **왜 그랬는지 출력한다.** `--no-predictor` 로 규칙 기반을 강제할 수 있고,
> `--predictor-mode` 는 생략하면 모델 클래스에서 판정한다(WaypointNet→waypoints,
> RankNet→rank) — 모델과 모드를 손으로 짝지으면 어긋난다.
>
> **라이브러리 기본값은 여전히 규칙 기반이다** (`PipelineConfig.predictor = None`).
> `traffic_llm` 은 torch 없이도 동작해야 하므로 패키지 기본값을 학습 모델로 둘 수 없다.

예측기는 `PredictContext → List[PredictedPath]` 함수다. 입력은 **규칙 기반이 쓰는 것과
정확히 같은 정보**를 하나로 모은 것이고(액터 상태 + 지도가 낸 후보 경로), 출력은 하류가
그대로 쓰는 `PredictedPath` 목록이다.

기본 모드(`rank`)는 모델이 **후보에 점수만 매기고** 기하는 지도에서 나온 것을 쓴다.
노면을 벗어나지 않고, 학습 데이터가 적어도 최악의 경우 사전확률 근처로 수렴한다.
좌표를 직접 회귀하려면 `mode="waypoints"` 를 쓴다(현재 위치 기준 상대 좌표).

torch 는 **선택 의존성**이다 — 지연 임포트하므로 없어도 나머지가 동작하고, 없는 상태로
`TorchPredictor` 를 쓰면 설치 안내를 담은 `ImportError` 가 난다.

### 지도학습 데이터 생성

```bash
# 16조각으로 나눠 동시에 (한 프로세스로는 몇 시간, 16조각이면 ~15분)
for i in $(seq 0 15); do
    python examples/make_predict_dataset.py \
        --root /home/ylim/inclab-nas/DeepAccident --carla-maps ./carla_map \
        --out out/predict_dataset --shard $i/16 &
done; wait
python examples/make_predict_dataset.py --merge --out out/predict_dataset
```

전체 607 시나리오로 **표본 151,563개**(실패 0)가 나온다. 조각 하나만 확인하려면
`--shard 0/16` 를, 범위를 좁히려면 `--split` / `--type` / `--town` / `--limit` 를 쓴다.

정답은 **시나리오 전체를 보고** 정한다. 시각 t 의 액터가 실제로 어디로 갔는지를 t 이후
스냅샷에서 읽어, 후보 폴리라인까지의 평균 거리가 가장 작은 것을 정답 후보로 쓴다.
평균 거리가 3.0m(차로 폭의 절반)를 넘으면 후보 집합 밖의 행동(차선 변경·노면 이탈)이며,
**버리지 않고 `matched=false` 로 남긴다** — 조용히 버리면 데이터가 실제보다 쉬워 보인다.

레코드에는 입력 특징(`global`[22], `candidates`[N,12]), 분류 정답(`best_candidate`),
회귀 정답(`target_offsets`, 상대 좌표), 그리고 **정답의 신뢰도**(`match_error_m`,
`candidate_errors_m`)가 들어간다.

> **정답이 하나로 정해지지 않는 표본이 많다.** 차량이 아직 갈림길에 닿지 않았으면
> 여러 후보가 실제 궤적과 똑같이 가깝고, 그때 `best_candidate` 는 후보 목록의 순서를
> 고른 것이다. 그런 표본을 `tied_best[]` · `label_ambiguous` · `margin_m` 으로 표시한다.
> **학습에서 제외하거나 `tied_best` 를 집합 라벨로 써야 하고, 평가에서는 반드시
> 제외해야 한다** — 사전확률의 argmax 도 같은 순서로 동률을 깨므로 그냥 세면 정확도가
> 구조적으로 부풀려진다.

필드별 명세와 학습 예시는
[`docs/predict_model_io.md`](../docs/predict_model_io.md) 에 있다. 베이스라인은
`examples/predict_dataset_baseline.py` 로 다시 잴 수 있다.

### 학습으로 얻을 것이 어디에 있는가

DeepAccident **전체 607 시나리오**(표본 151,563개, 실패 0)에서 규칙 기반 사전확률의
성능을 쟀다. 후보가 2개 이상이고 `matched=true` 이며 **정답 라벨이 유일한** 22,309개
기준이다.

| 지표 | 값 |
|---|---|
| 사전확률 argmax = 정답 후보 | **61.1%** |
| 사전확률 argmax 의 기동 = 정답 기동 | **94.6%** |
| 무작위 선택 기대값 | 25.2% |
| 다후보 표본 중 정답 라벨이 임의 | 59.3% (채점에서 제외) |

**기동 판별은 거의 다 맞는다.** 틀리는 것은 같은 기동으로 갈 수 있는 진출로가 여럿일 때
어느 쪽인지이고 — 다후보 표본의 62% 가 "직진 후보 여럿" 이다 — 사전확률은 그것들에 같은
값을 주므로 구별하지 못한다. 학습 모델이 기여할 곳은 좌/우 판별이 아니라
**진출로 선택**이다.

> 라벨이 임의인 표본을 넣으면 69.8% 로 보이지만 그 값은 성능이 아니다 (사전확률의
> argmax 와 정답의 argmin 이 같은 목록 순서로 동률을 깬다). 자세한 근거와 분해는
> [`docs/predict_model_io.md`](../docs/predict_model_io.md) §5 에 있다.

### 학습

```bash
# 후보 순위 학습 (기본)
python examples/train_predict_model.py --data out/predict_dataset \
    --epochs 100 --out out/predict_model

# 좌표 직접 회귀
python examples/train_predict_model.py --data out/predict_dataset \
    --task waypoints --epochs 100 --out out/predict_model
```

GPU 가 있으면 자동으로 쓴다. **시나리오 단위**로 나눈다 — 0.5초 간격 인접 표본은 서로
거의 같아서 표본 단위 무작위 분할은 성능을 크게 과대평가한다. 시험셋으로 epoch 을 고르지
않기 위해 학습셋에서 검증셋을 따로 뗀다(기본 10%).

100 epoch 결과 (RTX PRO 6000 에서 rank 2초 / waypoints 24초):

**`--task rank`** — 후보 중 정답 인덱스를 고른다 (top-1, 클수록 좋다).

| top-1 | 학습 | 검증 | **시험** |
|---|---:|---:|---:|
| **RankNet** | 79.4% | 74.6% | **68.8%** |
| 규칙 기반 (`prediction.py`) | 61.4% | 58.7% | 61.0% |
| 향상 | +17.9%p | +15.9%p | **+7.9%p** |

시드 0~3 의 시험 top-1 은 68.7~69.7% 다 — 변동이 약 1%p 이므로 "69% 내외" 로 읽는다.

주요 옵션: `--wp-loss {huber,huber_norm,ade,l2}`(기본 huber — **평가 지표 ADE 가 손실이
아니다**, [`docs/waypointnet.md`](../docs/waypointnet.md) §9) · `--man-weight`(기동 손실
가중치) · `--history-encoder`(위) · `--split-by {dataset,scenario}` · `--val-frac`.

**`--task waypoints`** — 좌표를 직접 회귀한다 (ADE/FDE, **작을수록** 좋다).
`matched=false` 를 뺄 이유가 없으므로 학습 표본이 **114,169 개**(rank 의 6.7배)다.

| 시험셋 (23,453) | ADE | FDE |
|---|---:|---:|
| **WaypointNet** (과거 5초 궤적 + 트랜스포머) | **3.01 m** | **5.25 m** |
| WaypointNet (이력 없음) | 3.05 m | 5.30 m |
| 규칙 기반 (top-1 후보) | 4.01 m | 7.00 m |
| 등속 직진 | 4.13 m | 7.42 m |
| 후보 최선 = 후보 방식의 오라클 | 3.75 m | — |

**ADE −24%, 그리고 후보 오라클(3.75m)까지 넘는다.** 후보를 완벽히 골라도 도달할 수 없는
정확도다 — 후보 집합의 상한을 벗어나기 때문이다. 이득의 대부분은 **후보 방식이 구조적으로
맞힐 수 없는 21%** 에서 온다 (그 구간만 보면 8.26m → 4.78m, −42%).
대가는 **다중 모드를 잃는 것**이다 (궤적 하나, 확률 1.0).

입력에는 **과거 5초 궤적**(`ActorState.track_history`, 1초 간격 5단계)이 들어가고
`--history-encoder {transformer,gru,mlp,none}` 로 계열 구조를 고른다. 이력의 몫은
ADE 0.04m(1.2%)로 작지만 시드 3회에서 일관되고 이력 보유량을 따라간다 —
[`docs/waypointnet.md`](../docs/waypointnet.md) §2.5.

기동 라벨은 좌표와 함께 **모델이 분류한다**(`직진·좌회전·우회전·유턴 + 후보 밖 경로`).
정확도를 재보면 어디서 무엇을 얻고 잃는지가 분명하다.

| 기동 라벨 | 전체 | 후보 안 표본만 |
|---|---:|---:|
| 모델의 기동 분류 헤드 | **83.6%** | 91.8% |
| 사전확률 1순위 후보에서 읽기 | 77.0% | **98.6%** |

**기동은 지도에서, 좌표는 모델에서** 가 결론이다. 후보를 따라가는 차량의 기동만 필요하면
규칙/rank 가 낫고(98.6%), 모델 쪽 이점은 **"후보 밖" 을 판정할 수 있다는 것**이다
(정밀도 66.9% · 재현율 54.3% — 규칙 기반은 아예 불가능). 자세한 것은
[`docs/predict_model_io.md`](../docs/predict_model_io.md) §5.7–5.8 과 모델 내부 구조는
[`docs/waypointnet.md`](../docs/waypointnet.md) 에 있다.

> **이 수치는 "정답이 후보에 있는 경우"로 한정한 값이다.** 후보는 지도에서 만든 것이고
> 실제 궤적이 그중 하나라는 보장이 없다 — 레코드의 **21.4%** 는 어느 후보와도 3m 안에서
> 맞지 않는다(차선 변경·노면 이탈). 예측 구간이 길수록 늘어난다(2초 4.5% → 6초 35.5%).
> 그것을 오답으로 세면 시험 정확도는 RankNet 73.0% · 규칙 71.4% 이고 **상한이 78.8%**
> 다. 규칙 기반도 같은 한계를 지므로 비교는 성립하지만, 모델은 "정답이 여기 없다"를
> 신호로 내지 못한다 (확신도 판별력 AUC 0.281 — 역전돼 있다). 후보 밖 여부 자체는
> 예측 가능하므로(AUC 0.873) abstain 클래스를 두는 것이 다음 단계다.

`<모델>_best.pt`(검증 최고 epoch) · `<모델>_final.pt` · `train_report_<과제>.json` 을
저장하고, 바로 파이프라인에 꽂힌다.

```python
cfg.predictor = TorchPredictor("out/predict_model/ranknet_best.pt")
cfg.predictor = TorchPredictor("out/predict_model/waypointnet_best.pt",
                               mode="waypoints")
```

과적합·타운별 분해·후보 밖 한계·좌표 회귀의 대가는
[`docs/predict_model_io.md`](../docs/predict_model_io.md) §4.1 · §5.5–5.7 에 있다.

## LLM provider (Claude / OpenAI / Gemini)

기본은 Claude(Anthropic Messages API)다. `--provider` 로 형식을 바꿀 수 있고,
**내용은 provider 와 무관하게 같다** — 브리핑·구조화 JSON·질문·system 프롬프트·
응답 스키마가 동일하고 껍데기만 달라진다.

```bash
# 기본 (Claude)
python examples/run_deepaccident.py --root <루트> --carla-maps ./carla_map \
    --windows-dir out/windows

# Gemini / OpenAI — 모델 id 를 반드시 지정한다
python examples/run_deepaccident.py --root <루트> --carla-maps ./carla_map \
    --windows-dir out/win_gemini \
    --provider gemini --model gemini-2.5-pro
python examples/run_deepaccident.py --root <루트> --carla-maps ./carla_map \
    --windows-dir out/win_openai \
    --provider openai --model gpt-5
```

```python
cfg.serialize.provider = "gemini"        # 'claude'(기본) | 'openai' | 'gemini'
cfg.serialize.model = "gemini-2.5-pro"
cfg.serialize.provider_extra = {"generationConfig": {"temperature": 0.2}}
```

| | Claude | OpenAI | Gemini |
|---|---|---|---|
| 엔드포인트 | `/v1/messages` | `/v1/chat/completions` | `models/{model}:generateContent` |
| system | `system[]` 블록 (+`cache_control`) | `messages[0]` role=system | `systemInstruction.parts` |
| 사용자 입력 | `content[]` 블록 3개 유지 | 한 문자열로 합침 | `parts[]` 3개 유지 |
| 출력 제약 | `output_config.format.json_schema` | `response_format.json_schema` (strict) | `generationConfig.responseSchema` |
| 추론 | `thinking:{type:"adaptive"}` | `reasoning_effort` | (`provider_extra` 로 지정) |
| 토큰 상한 | `max_tokens` | `max_completion_tokens` | `generationConfig.maxOutputTokens` |
| 모델 위치 | 본문 `model` | 본문 `model` | **URL 경로** |

### 설계

- **payload 파일 = 요청 본문 그대로.** 그래야 `**payload` 로 바로 보낼 수 있고
  API 가 모르는 키가 섞이지 않는다. Gemini 는 모델이 URL 경로에 들어가므로 본문에
  `model` 을 넣지 않고, provider·모델·엔드포인트는 **manifest.json 에 기록**한다.
- **모델 id 를 짐작하지 않는다.** Claude 만 검증된 기본값(`claude-opus-5`)이 있고
  OpenAI·Gemini 는 `--model` 을 요구한다. 검증하지 않은 id 를 박아 두면 호출이
  알 수 없는 이유로 실패한다.
- **응답 스키마는 방언이 다르다.** 정본은 JSON Schema 이고, Gemini 용은
  OpenAPI 부분집합으로 옮긴다 — `type` 대문자화, `additionalProperties` 제거,
  필드 순서 고정을 위한 `propertyOrdering` 추가. OpenAI strict 모드가 요구하는
  "모든 object 에 `additionalProperties:false` + 전 속성 `required`"는 이 프로젝트
  스키마가 이미 만족한다.
- **버전에 따라 달라지는 값은 `provider_extra` 로.** 추론 예산 같은 파라미터를
  코드에 박지 않는다. 비추론 모델에는 `{"reasoning_effort": null}` 로 지운다.

> **검증 상태**: Claude 경로는 이 저장소에서 실제로 쓰는 형식이다. OpenAI·Gemini
> 형식은 각 API 문서 기준으로 작성했고 **실제 호출로 검증하지 않았다** (해당
> API 키가 없다). 형식·스키마 변환·응답 파싱은 단위 테스트로 확인했다.

### 호출과 채점

```bash
pip install anthropic                    # Claude 만 SDK 사용
$env:ANTHROPIC_API_KEY="sk-ant-..."      # 또는 OPENAI_API_KEY / GEMINI_API_KEY

python examples/ask_llm.py out/windows_example/DeepAccident_mini/accident --score
```

`ask_llm.py` 가 manifest 에서 provider·엔드포인트를 읽어 호출하고, 응답에서
구조화 출력을 꺼내 `ground_truth_*.json` 과 구간별로 대조한다. Claude 는 공식
SDK(스트리밍), OpenAI·Gemini 는 요청 본문을 그대로 HTTP POST 한다.

**이미 받은 응답을 지키는 두 옵션.** 호출에는 실제 비용이 들고, `scores.json` 은
호출마다 덮어써진다 (창을 하나씩 부르면 앞 창 결과가 사라진다).

| 옵션 | 무엇을 하는가 |
|---|---|
| `--rescore` | LLM 을 **부르지 않고** 저장된 `response_*.json` 만 다시 채점해 합산한다. `--early-credit` 을 바꿔 다시 채점할 때도 쓴다 |
| `--skip-existing` | 응답이 이미 있는 창은 호출하지 않는다. 중단된 배치를 이어 돌릴 때 쓴다 |
| `--max-tokens` | 잘린 응답(`stop_reason=max_tokens`)만 payload 재생성 없이 다시 받는다 |

```bash
# 전부 받은 뒤 합산 (LLM 호출 없음)
python examples/ask_llm.py <윈도우 디렉터리> --score --rescore
```

**조건별 비교.** 경로 예측기만 바꿔 만든 payload 의 결과를 나란히 본다 — 창 구성이
다르면 경고한다.

```bash
python examples/compare_llm_scores.py \
    rule=out/win_rule/... rank=out/win_rank/... waypoints=out/win_wp/...
```

## 출력 언어 (한국어 / 영어)

기본은 한국어다. `--language en` 을 주면 **모델에게 가는 모든 문자열**이 영어로
나온다 — 자연어 브리핑, 윈도우 텍스트, 질문, system 프롬프트, JSON 안의 라벨,
출력 스키마 설명, 그리고 정답 파일·manifest 의 주석까지.

```bash
python examples/run_deepaccident.py --root <루트> --scenario Town05 \
    --carla-maps ./carla_map \
    --windows-dir out/windows --language en
python examples/make_demo_data.py --language en
python -m traffic_llm.cli --map roads.geojson --vehicle ... --language en
```

```python
cfg = PipelineConfig()
cfg.serialize.language = "en"      # 'ko'(기본) | 'en'
```

| 항목 | 한국어 | 영어 |
|---|---|---|
| 브리핑 | `High St 북행 2차선 중 1차선 \| 43km/h \| 차선유지` | `High St northbound lane 1 of 2 \| 43km/h \| keeping lane` |
| 상호작용 | `교차로 J1, 도달시간차 0.5s \| 직교 진입 상충 예상` | `junction J1, arrival time gap 0.5s \| orthogonal entry conflict expected` |
| 질문 | `…미래 5초의 사고 발생 가능성을 1초 구간별로 판단하십시오` | `…judge the likelihood of an accident in each 1-second interval…` |
| 정답 주석 | `충돌은 마지막 관측 시점 0.2초 뒤에 일어난다.` | `The collision happens 0.2s after the last observed timestep.` |

**JSON 키·수치·actor id·도로명은 언어와 무관하게 동일하다.** 값만 번역되므로
채점 코드는 언어를 몰라도 된다. 도로명처럼 데이터에서 온 문자열은 번역하지
않는다.

### 설계

`i18n.py` 한 곳에 모았다.

- **데이터 모델에 저장하는 라벨의 정본은 한국어**이고(진행방향·기동·예측 경로),
  직렬화 경계에서 번역한다. 파이프라인 내부를 키로 바꾸면 로직과 테스트가 전부
  흔들리는데 얻는 것이 없다 — 라벨은 이 패키지만 만드는 닫힌 집합이고 번역이
  필요한 곳은 출력 한 지점뿐이다. 사전에 없는 값은 그대로 통과한다.
- **숫자가 섞인 문장은 문자열 치환이 아니라 형식 템플릿**으로 만든다. 상호작용
  설명처럼 값이 끼어드는 문장은 `Interaction` 의 구조화 필드
  (`junction_id`, `arrival_gap_s`, `conflict`, `turn_probs`, `target_lane`)에서
  언어별로 재구성한다.
- **지도 출처는 산문이 아니라 키**로 저장한다(`inferred_from_trajectories`).
  데이터 모델에 산문을 넣으면 언어 전환이 문자열 치환으로 퇴화한다 — 실제로
  처음엔 `"궤적 합성 (차선 수는 관측 하한)"` 을 저장해 두고 접두어만 번역했다가
  괄호 안이 한국어로 남았다.
- 코드 주석·콘솔 출력·이 문서는 계속 한국어를 쓴다. 번역 대상은 모델이 읽는
  문자열뿐이다. (BEV 이미지 안의 문자열은 폰트 문제로 항상 영어다 —
  [BEV 절](#bev-시계열-이미지-bev_renderpy) 참고.)

누락을 눈으로 잡기 어려우므로 **산출물 전체를 훑어 한글이 남았는지 검사**하는
테스트를 둔다: payload·정답·manifest 파일을 실제로 써 보고 전수 검사하며,
영어 템플릿 표에 빠진 키가 있는지도 대조한다.

## 입력 형식

### 텔레메트리 CSV (차량당 1개, 필수)

```csv
t,lat,lon,heading_deg,speed_mps,yaw_rate_dps
0.000,39.96116,-83.00070,0.00,12.000,0.0
0.100,39.96117,-83.00070,0.00,12.000,0.0
```

- `t`: **모든 차량이 공유하는 절대 시각**(unix epoch 또는 세션 상대시각). 차량 간
  동기화의 기준이므로 정확해야 한다. 오차가 있으면 `add_vehicle(t_offset=...)` 로 보정.
- `heading_deg`: 진북 기준 시계방향 (0=북, 90=동)
- `speed_mps` 대신 `speed_kph` 컬럼도 허용. `yaw_rate_dps`, `alt` 는 선택.

### 도로 지도 GeoJSON (필수)

`LineString` FeatureCollection. OSM 태그 관례를 따른다.

```json
{"type":"Feature",
 "properties":{"id":"high_n","name":"High St","lanes":4,
               "lanes:forward":2,"lanes:backward":2,
               "oneway":"no","maxspeed":"50","highway":"primary"},
 "geometry":{"type":"LineString","coordinates":[[-83.0007,39.9612],[-83.0007,39.9639]]}}
```

`maxspeed` 는 `"50"`(km/h) 과 `"40 mph"` 를 모두 파싱한다.
OSM 에서 받으려면:

```python
import osmnx as ox
g = ox.graph_from_point((39.9612, -83.0007), dist=800, network_type="drive")
ox.save_graph_geopackage(g, "roads.gpkg")  # → GeoJSON 으로 변환해 사용
```

OpenDRIVE / lanelet2 를 쓰는 경우 `RoadNetwork.from_geojson` 만 교체하면 된다
(동일한 `Road` 객체 리스트를 만들면 나머지는 그대로 동작).

### 검출 결과 JSON (선택 — 영상 대신 사용)

```json
{"fps": 30,
 "detections": [{"frame": 0, "track_id": 3, "cls": "car",
                 "conf": 0.91, "bbox": [820, 540, 980, 660]}]}
```

## 출력 형식

`to_text()` — LLM 추론 입력으로 가장 효과적:

```
# 교통 상황 스냅샷 (t = 4.0s, 지역: Columbus, OH (High St × Broad St))
관측차량 3대, 주변차량 3대 (전체 6대 중 상위 6대 수록)

※ 지도 규약: 우측통행 | 1차선은 중앙선쪽 | 차선폭 3.25m | 도로 4개·교차로 1개

## 관측차량 (탑재 카메라로 주변을 관측)
- EGO_V2: High St 북행 2차선 중 1차선 | 제한속도 50km/h | 다음 교차로까지 24m
  | 32km/h | 방위 12° | 차선유지 | 예상경로: 직진 56%, 좌회전 37%, 우회전 7%
...
## 주변차량
- V001(car): High St 북행 2차선 중 1차선 | 43km/h | 방위 8° | 차선변경(좌)
  | 관측: V1, V3 (존재확신 0.92, 위치정확도 0.85, 관측거리 25m)
  [EGO_V1 기준 전방 25m, 좌측 1.6m]

## 상호작용 및 주의 상황
- EGO_V1 → V001 | 선행차 추종 | 간격 25.0m | 헤드웨이 2.1s | TTC 8.3s
- EGO_V2 → V002 | 교차로 J1 동시 진입 예상 (도달시간차 0.6s) | TTC 2.1s

## BEV 개략도 (기준: EGO_V2) ...
```

`to_json()` — 정량 평가·툴 호출용 (payload 에는 **기본으로 넣지 않는다**,
[해당 절](#구조화-json-블록은-선택-기본-꺼짐) 참고). `build_messages()` —
provider 별 요청 본문 (Claude 기본: `claude-opus-5`, adaptive thinking,
system 프롬프트 캐싱).

절별 형식과 읽는 법은 [`docs/payload_text_structure.md`](../docs/payload_text_structure.md)
에 정리했다.

## 정확도 특성과 한계

단안 카메라 기반이므로 다음 한계가 있다. **이 파이프라인은 불확실성을 숨기지
않고 출력에 명시**하며(`observed_range_m`, `confidence`, `※위치·속도 불확실`),
판정이 불가능한 경우 없는 기동을 만들어내기보다 "판정 보류"를 택한다.

| 항목 | 특성 |
|---|---|
| 종방향 거리 | DeepAccident 실측: 15-70m 오차 중앙값 2.0~2.5m, 70m 초과에서 7.3m로 급증. 평면 노면 가정이 깨지는 경사로에서 추가 악화 |
| 횡방향 위치 | 실측 중앙값 1.02m. 거리 의존 편향이 존재하며 접근할수록 수렴 |
| 차선 판정 | 근거리에서 신뢰 가능. 원거리 차량은 인접 차선으로 오판될 수 있다 |
| 차선변경 검출 | 2.5초 관측창을 쓰므로 **약 1~2초 지연**된다. 관측거리가 급변하는 원거리 차량은 판정을 보류한다(실제 변경을 놓칠 수 있음) |
| 가감속 | 관측거리가 프레임 간 2m 이상 변하고 45m 초과인 구간에서는 보고하지 않는다 (편향 미분이 허위 급감속으로 나타남) |
| 방위각 | 주변차량은 궤적 미분값이므로 도로에 매칭되면 **도로축**을 쓴다(실측 중앙값 0.3°). 진행 방향(순/역)을 확신할 수 없으면 **미확정**으로 보고한다 — 확정률은 실측 지도 55%, 합성 도로망 15%. 확정된 것 중 약 6%가 여전히 180° 반대다. 정지·저속에서는 미확정. 관측차량은 GPS 값, 3D 센서가 있으면 측정값 사용 |
| 속도 | 최소자승 궤적 적합, 실측 오차 중앙값 0.45 m/s. 추정값이 적합 잔차의 2배에 못 미치거나 클래스별 **지속 주행 속도** 상한을 넘으면 **미확정**으로 보고한다. 표본 4개(2Hz 에서 1.5초)가 모이기 전에는 속도가 나오지 않는다. 첫 관측 시점에는 **미확정**으로 보고(정지로 단정하지 않음) |
| 근접(<5m) | 차량이 화면을 가득 채워 bbox 가 절단되고, 근접면 보정량이 거리의 절반에 달해 상대오차가 구조적으로 커진다 |

정확도를 높이려면: 카메라 내부·외부 파라미터를 체커보드로 실측(`CameraConfig`
기본값은 근사), 클래스별 실폭 사전값(`class_width_m`)을 대상 차종 분포에 맞게
조정, 그리고 가능하면 스테레오/LiDAR 로 측거를 대체.

## BEV 시계열 이미지 (`bev_render.py`)

지도 위에 차량을 **방위각대로 회전한 사각형**으로 그려 시나리오의 시간 변화를
`<시나리오이름>_<타임스탬프>.jpg` 이미지 집합으로 낸다
([DeepAccident `figs/First_video.gif`](https://github.com/tianqi-wang1996/DeepAccident/blob/main/figs/First_video.gif)
오른쪽 패널과 같은 형태).

```bash
python examples/run_deepaccident.py --root <루트> \
    --scenario Town05_type001_subtype0002 --type type1_subtype2_accident \
    --carla-maps C:/path/to/carla_map --bev-dir out/bev
```

```python
from traffic_llm.bev_render import BevConfig, BevRenderer, write_index_html

rend = BevRenderer(res.network, BevConfig(focus="ego"))
paths = rend.render_sequence(snaps, "out/bev", res.scenario.scenario)
write_index_html(paths, "out/bev/index.html")   # 브라우저로 훑어보기
```

그리는 것:

- 도로 노면(중심선을 차선 수만큼 좌우 옵셋한 다각형), 차선 파선, 중앙선
- 차량: 클래스 실측 치수 그대로의 회전 사각형 + 진행방향 삼각 표식
- 관측차량(ego)은 고유색 + 굵은 테두리, 노변 인프라는 마름모
- 최근 궤적 꼬리(실선), 위험 쌍(빨간 연결선 + TTC)
- 예상 경로(점선)는 **기본 꺼짐** — `--bev-predictions` 로 켠다
- HUD: 시나리오명·시각·프레임·차량 수, 축척바, 북 방향, 범례

| 옵션 | 뜻 |
|---|---|
| `--bev-focus ego\|all` | 시야 기준. `ego` 는 관측차량 궤적만으로 시야를 정한다 |
| `--bev-labels all\|ego_and_risky\|none` | 라벨 표시 범위 (혼잡할 때 줄인다) |
| `--bev-follow` | 프레임마다 시야 재계산 (기본은 시퀀스 고정 시야) |
| `--bev-predictions` | 예상 경로 점선을 겹쳐 그린다 (기본 꺼짐) |
| `--bev-width/--bev-height/--bev-format` | 해상도, `jpg`\|`png` |

**예상 경로 점선을 기본으로 끈 이유.** 액터 20~25대 전부의 점선이 겹쳐
읽기 어렵고, 범례에 없어 지도 표시인지 예측인지 구분되지 않으며, 무엇보다
**도로에 매칭되지 않은 액터**(인도를 걷는 보행자 등)의 경로는 방위를 그대로
외삽한 값이라 노면이 그려지지 않는 곳을 가로지르는 정체 불명의 대각선 조각으로
남는다. 방위 미확정 차량의 점선 테두리와도 헷갈린다. 켜면 범례에
`predicted path (dashed)` 항목이 붙고, 저속(1.5m/s 미만)·도로 미매칭 액터는
제외해 위 문제를 줄인다.

설계상 주의한 것:

- **이미지 안 문자열은 전부 영어.** 이미지가 어느 환경에서 열릴지, 어떤 폰트가
  깔려 있을지 알 수 없다 — 한글을 넣으면 폰트가 없는 환경에서 두부(□)가 된다.
  코드 주석·콘솔 출력·이 문서는 한글을 그대로 쓴다.
  테스트가 `ImageDraw.text` 를 가로채 **실제로 그려진 모든 문자열**을 검사한다
  (소스의 `dr.text(...)` 만 훑으면 미리 f-string 으로 조립한 문구를 놓친다).
- **기본 고정 시야.** 프레임마다 시야를 다시 맞추면 화면이 흔들려 시간 변화를
  읽을 수 없다. 단, 고정 시야로 담을 수 없을 만큼 멀리 이동하는 시나리오는
  자동으로 프레임별 시야로 전환한다 (아니면 대부분의 프레임이 빈 화면이 된다).
- **`focus='ego'` 기본.** 인프라는 100m 넘는 주차 차량까지 관측하므로 모든
  액터를 담으면 시야가 과도하게 넓어져 차량이 점이 된다.
- **방위 미확정 처리.** 진행 방향을 확신할 수 없으면 도로축에 맞춘 사각형을
  **점선 테두리 + 방향 화살표 생략**으로 그린다 (사각형은 180° 뒤집어도 같은
  모양이므로 축만 맞으면 표현 손실이 없다). 도로 매칭도 없으면 원으로 그린다 —
  방향을 아는 척하지 않는다.
- **궤적 꼬리 잡음 절단.** 함의 속도가
  `min(클래스 지속속도 상한, max(2 m/s, 2.5 × 추정속도))` 를 넘는 단계에서
  꼬리를 끊는다. 클래스 상한이 없으면 잡음으로 부풀려진 속도가 자기 자신의 잡음
  궤적을 정당화한다 — 보행자 추정속도 5m/s → 허용 12.5m/s → 인도 위를 가로지르는
  지그재그가 그대로 그려진다.
- **교차로를 한 장으로 덮는다.** 교차로 내부는 좁은 연결로 여러 개로 표현되어
  (Town05 는 256개 도로 중 206개가 교차로 내부 연결로) 그대로 이어 붙이면
  사이사이 빈틈이 남아 도로가 끊긴 것처럼 보인다. 교차로에 접하는 도로들의
  노면 단면 꼭짓점을 모아 볼록껍질로 한 장 덮는다. 단면을 도로 반폭만큼 교차로
  안쪽으로 연장해야 껍질이 마름모가 되지 않고 네 귀퉁이까지 덮인다.
- **교차로 안에는 차선 표시를 그리지 않는다.** 실제 도로와 같고, 좁은 연결로마다
  테두리를 그리면 교차로가 방사형으로 찢어져 보인다. 자르는 단위는 정점이 아니라
  **선분**이다(Cyrus–Beck) — 정점 단위로 자르면 거친 폴리라인(합성 도로망은
  300m 도로가 정점 2개)에서 선이 통째로 사라진다.
- **노면은 세그먼트별 사각형으로.** 좌·우 경계 폴리라인으로 큰 다각형 하나를
  만들면 곡률 반경보다 오프셋이 클 때 내측 경계가 스스로 교차해 노면이 쐐기로
  찌그러진다(교차로 진입부가 끊겨 보이는 원인). 세그먼트별 사각형은 곡선 내측에서
  조금 겹칠 뿐 빈틈이 없다.
- **HUD·범례에 가리는 라벨은 생략.** 반쯤 잘린 글자를 남기지 않는다.
- **긴 관측차량 이름은 축약.** `EGO_other_vehicle_behind` → `OTH-B` (전체 이름은
  범례에). 지도 위에 전체 이름을 쓰면 라벨이 서로 덮어 읽을 수 없다.

조감도에 **보행자(`P###`)와 주차 차량도 그대로 나온다** — DeepAccident 정답의
`pedestrian` 객체와 정차 차량이며 오류가 아니다. 이들은 인도·노상에 있어 그려진
차도 밖에 놓인다.

Pillow 가 없으면 **SVG 로 폴백**한다 — 추가 의존성 없이 벡터 품질이고 브라우저에서
바로 열린다.

## 좌표계 규약

| 대상 | 규약 |
|---|---|
| 픽셀 | (u, v), 좌상단 원점 |
| 차량 | (x_fwd, y_left, z_up) [m], 카메라 광학중심 원점, 오른손 좌표계 |
| 월드 | (e, n) [m], 지역 ENU 평면 (e=동, n=북) |
| 방위각 | [0,360), 진북 기준 시계방향 |
| 차선번호 | 기본 `from_median` (1=중앙선쪽, 한국 관례). 미국식은 `from_curb` |
| 통행방향 | 기본 `right` (우측통행) |

## 모듈

| 파일 | 역할 |
|---|---|
| `schemas.py` | 데이터 모델 및 좌표계 규약 |
| `config.py` | 카메라/차선/인지/융합/직렬화/DeepAccident/도로합성 설정 |
| `geometry.py` | 픽셀↔차량↔월드 변환, 카메라 모델(1대분), 폴리라인 정사영 |
| `roadmap.py` | 도로망 로딩, 차선 매칭, 하류 경로 열거 |
| `perception.py` | 검출·추적 백엔드 인터페이스 (`PerceptionBackend`) + YOLO / 사전계산 JSON 구현 |
| `fusion.py` | 다중 관측자 융합, 전역 트랙 ID 관리 |
| `kinematics.py` | 속도·가속도·기동 분류, 상호작용/위험 분석 |
| `prediction.py` | 지도제약 단기 경로 예측 |
| `serialize.py` | LLM 입력 직렬화 (텍스트/JSON/BEV/메시지/평가레코드) |
| `pipeline.py` | 오케스트레이션, 시각 동기화, **카메라별 역투영 파라미터 선택**, 인프라 관측자, 지도 맥락 |
| `cli.py` | 명령행 인터페이스 (일반 영상용) |
| **`deepaccident.py`** | DeepAccident 어댑터 (포맷 파싱, CARLA↔ENU, **카메라 열거·카메라별 가시성 필터**, 텔레메트리 합성) |
| **`da_runner.py`** | DeepAccident 시나리오 → 스냅샷 스트림 오케스트레이션 |
| **`da_eval.py`** | 정답 대비 정확도 평가 (신원 기반 매칭) |
| **`roadgen.py`** | 궤적 기반 도로망 합성 |
| **`carla_map.py`** | CARLA 타운 정보 + OpenDRIVE(.xodr) 임포트 |
| **`accident_qa.py`** | 슬라이딩 윈도우 사고 예측 질의 + 정답 파일 + 채점 + 응답 형식 검사 |
| **`bev_render.py`** | BEV 시계열 이미지 렌더링 (지도 + 회전 사각형 차량) |
| **`i18n.py`** | LLM 입력 문자열의 한국어/영어 전환 (라벨·템플릿·프롬프트·스키마) |
| **`providers.py`** | provider 별 요청 본문 (Claude/OpenAI/Gemini), 스키마 방언 변환, 응답 파싱 |
| **`predict_model.py`** | 경로 예측기 교체 지점 — 입력 계약(`PredictContext`), 특징 인코딩, PyTorch 예측기. **torch 없이 import 된다** |
| **`predict_nets.py`** | `RankNet`(후보 순위) · `WaypointNet`(좌표 회귀 + 기동 분류) · `HistoryEncoder`(과거 궤적 GRU/트랜스포머/MLP) 신경망. torch 를 import 시점에 요구하므로 분리했다 (저장한 모델이 어디서든 열리도록 패키지 안에 둔다). 구조는 [`docs/waypointnet.md`](../docs/waypointnet.md) |

## 테스트

```bash
python -m unittest discover -s tests -v

# DeepAccident 실데이터 테스트를 다른 경로에서 돌릴 때
set DEEPACCIDENT_ROOT=C:\path\to\DeepAccident_mini
```

408개 테스트.

| 파일 | 개수 | 무엇을 고정하는가 |
|---|---|---|
| `test_pipeline.py` | 116 | 기하 변환 왕복, 차선 배정, 정투영↔역투영 정확도, 융합 중복제거, 존재확신도/위치정확도 분리, 기동 오탐/미탐, 직렬화 절단 고지, 클래스별 신원·속도 관문, 최소자승 속도·방위 적합, 도로축 진행방향 판정, **전역 트랙 id 전제와 위반 탐지**, **다중 카메라**(후방 검출이 뒤에 놓이는지, 이름 중복 거부, 화각 합집합 커버리지), **예상 경로 기하**(국소 도로축 기준 방향, 차로 추종, 차량 위치에서 시작), **좌/우회전 라벨 규약**(방위 시계방향 = 우회전), 경로 중복 병합과 확률 반올림 |
| `test_accident_qa.py` | 109 | **누적 창 생성**(0-1, 0-2, …, 0-5, 1-6)과 고정 길이 모드, 창보다 짧은 데이터, 접근 쌍 추출과 **관측량 기반 상한**(임박 쌍은 상한을 넘겨도 포함), 구간 경계 규약, 정답 버킷 배정, **조기 예측 인정**(이른 것은 정답·늦은 것은 오답·사건 단위 계수), **데이터 범위 밖 구간 채점 제외**, 차량 단위 채점, **응답 형식 검사**(중복·누락·초과 k), **관측 관계 블록**, **JSON 블록 on/off 와 대체 정보 이전**, **관측자 이름 vs 액터 id 구별**, **재생성이 응답을 지우지 않는지**, **예상경로 자연어 표기**(같은 기동이 여러 진출로로 갈릴 때 도로명으로 구별), 파일 출력·manifest, 실데이터 검증 |
| `test_bev_render.py` | 68 | 회전 사각형 기하, 폴리라인 옵셋과 노면 폭, 세그먼트 사각형이 급한 꺾임에서 찌그러지지 않는지, 볼록껍질·내부 판정, 선분 클리핑(Cyrus–Beck)과 구간 차집합, 교차로 덮개가 네 귀퉁이까지 덮는지, 월드→픽셀 변환, 파일명 규약(사전순 = 시간순), 궤적 꼬리 잡음 절단, 방위 미확정 폴백, **예상 경로 점선의 기본 꺼짐과 필터**, **트루타입 폰트 탐색**(못 찾으면 비 ASCII 가 두부로 그려진다), **그려진 모든 문자열이 라틴 문자인지**, Pillow 없을 때 SVG 폴백 |
| `test_deepaccident.py` | 57 | 레이블/meta 파싱, CARLA↔ENU 변환, 카메라 파라미터 복원, 인프라 지상고, 도로망 합성, OpenDRIVE 임포트, **분할 상위 디렉터리 스캔**(mini/train/val 한꺼번에, 중복 없음, 필터 전파), **카메라 열거·해석**(오타 거부, 6대가 1대보다 더 봄), **객체 단위 통계 항등식**, 실데이터 검증(센서 간 월드좌표 일치 <5cm, `sensor3d` 정확성, 단안 편향 한계, 정답 누수 차단) |
| `test_i18n.py` | 34 | 언어 코드 정규화(`en-US` 허용, 미지원은 한국어), 라벨/템플릿 사전의 영어 표기 누락 전수 검사, 상호작용 설명의 구조화 재구성, JSON 키·수치가 언어와 무관하게 동일한지, **payload·정답·manifest 파일 전체에 한글이 남지 않았는지** |
| `test_providers.py` | 24 | provider 별 요청 본문 형태(엔드포인트, system 위치, 블록 유지/합침, 토큰 상한 키), 스키마 방언 변환(Gemini 대문자·`propertyOrdering`, OpenAI strict 요건), **내용이 provider 와 무관하게 같은지**, `provider_extra` 깊은 병합, 응답 파싱과 토큰 사용량 |

실데이터 테스트는 데이터셋이 없으면 자동 skip 된다.

## 선택 의존성이 없을 때

| 미설치 | 동작 |
|---|---|
| `scipy` | 융합 매칭이 최적 할당(Hungarian) → 탐욕 매칭. 밀집 상황에서 매칭 품질 저하 |
| `pyproj` | 정밀 좌표변환 → 등거리 근사. 원점 수 km 내 오차 1m 미만 |
| `ultralytics`/`opencv` | 영상 처리 불가. 사전계산 검출 JSON 필요 |
| `Pillow` | BEV 이미지가 JPEG/PNG → SVG 로 폴백 (기능 손실 없음) |
