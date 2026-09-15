1. 센서/레이블 정보를 중앙에서 aggregate하고 WaypointNet으로 미래 경로를 예측한 뒤,
   payload를 LLM에 보내 사고를 질의하는 방법
2. DeepAccident 논문의 아이디어를 현재 환경에서 재현한 rule based 및 BEV motion
   baseline

원 논문의 V2XFormer를 그대로 실행하는 것은 공개 weight와 현재 GPU/CUDA 환경의
호환성 문제로 불가능. 따라서 input을 동일하게 통제하고, 경로 예측·사고 판정·LLM 질의 방식을 분리해 비교할 수 있도록 구성.

## summary so far

- 최종 목표 scoring 기준은 fixed-length sliding window다. 다만 아래에 기록된 기존
  pair cache/전체 Gemini 수치는 expanding-prefix 실행에서 나온 값이며, `0-5`
  slice 결과도 별도로 표시한다.
- 기존 실행 설정은 최대 observation 5초, stride 1초, 미래 horizon 5초,
  history stride 1초였다.
- LLM 입력은 `compact` payload를 사용.
- Dynamic actor-pair candidate를 clearance 순으로(the smaller the higher risk of collision) 최대 50개 만든 뒤, V2 pair re-ranker가
  점수를 계산하고 상위 12개를 payload에 feeding
- pair re-ranker V2를 train split 전체에서 학습하고 validation split 전체에서 평가
- ~~Gemini flash 2.5 모델로 전체 validation 재실험 완료 (104개 시나리오 / 732개 window)~~
- 현재 테스트 결과: `679 passed, 25 skipped, 4 subtests`.

## Frozen evaluation protocol (Phase 0, v1)

다음 fixed-window 규칙을 주 실험의 기준으로 고정한다. 기존 expanding-prefix
결과는 삭제하지 않고 별도의 secondary/diagnostic 결과로만 보존한다.

- **Observation:** 5초 고정 길이 창, stride 1초 (`warmup=False`)
- **Prediction horizon:** 관측 종료 시점 이후 5초, 1초 bucket 5개
  (`(t_end,t_end+1]` … `(t_end+4,t_end+5]`)
- **Window construction:** 선두의 5초 미만 창과 별도 전체구간 창은 제외
  (`full_window=False`); 관측 중 이미 발생한 충돌 창은 제외
- **Primary metric:** `binary_scenario` — 한 영상/시나리오를 TP/FP/TN/FN 한 건으로
  집계한다. 사고 시나리오에서는 사고가 미래 horizon에 들어오는 유효 window 중
  하나라도 감지하면 TP다.
- **Secondary metrics:** `binary_window` (고정 window 하나당 한 건), `strict`,
  `early`, `weighted`, `binary_bucket`. 마지막 축들은 관측 시점과 미래 bucket의
  세부 timing을 진단한다.
- **Early-credit policy:** 예측 horizon 자체가 5초이므로 `early=5초`는 horizon 시작부터
  실제 충돌 bucket까지의 경보를 인정한다. 따라서 `binary_scenario`와 달리 사고 후(late) 경보만
  오답으로 남는 관대한 보조 기준이다.
- **Aggregation:** `binary_scenario`는 시나리오당 한 건만 주 수치로 집계한다.
  `binary_window`와 bucket-level 결과는 관측 시점/timing 진단용으로 별도 기록한다.
  Expanding-prefix는 prefix를 독립 샘플로 합산하지 않는다.
- **Threshold:** V2 threshold는 train scenario의 grouped out-of-fold 결과로만 선택하고,
  validation/test 실행 전에 고정한다. 기존 expanding threshold `0.927234`는 재사용하지 않는다.
- **Ablation:** 동일한 scenario/window/LLM 설정에서 `Raw compact`와
  `Compact + V2`를 비교하고, payload token 수와 latency도 함께 기록한다.
- **DeepAccident:** 논문과 입력 horizon/label 조건이 일치하는 별도 비교 track으로
  실행한다. 논문 수치와 실제 재현 수치는 구분해 보고한다.

## 1. DeepAccident 입력 및 window 평가 파이프라인

- `traffic_llm.da_runner`와 `traffic_llm.deepaccident`로 DeepAccident 레이블,
  센서 가시성, CARLA map을 읽는다.
- `sensor3d` 모드는 레이블의 3D 위치·방위·속도를 사용하고, `camera` 모드는
  카메라 bbox 역투영 경로를 사용한다.
- `run_deepaccident.py`는 한 시나리오의 윈도우를 생성하고 payload/manifest/eval
  결과를 저장한다.
- `run_validation_batch.py`는 validation scenario를 선택하고 시나리오별 디렉터리를
  만든 뒤, 생성·감사·LLM 호출을 재개 가능하게 수행한다.
- `--all-validation`을 추가해 validation split 전체를 사용할 수 있게 했다.
- 사고가 관측 window의 미래 5초 안에 있는지 묻는 `binary_scenario` 평가를 주 지표로
  두고, `binary_window`, `strict`, `early`, `weighted`, `binary_bucket`은 보조 지표로
  함께 계산한다.

#### Scoring criteria

한 관측 window에는 미래 1초 단위 bucket들이 있다. 예를 들어 관측이 `0~5초`까지
끝나면 미래는 `(5,6]`, `(6,7]`, `(7,8]`, `(8,9]`, `(9,10]`으로 나뉜다.

| 기준         | 판정 단위                    | 정답으로 인정하는 경우                                         | 목적                               |
| ------------ | ---------------------------- | -------------------------------------------------------------- | ---------------------------------- |
| `binary_scenario` | 시나리오 1건          | 유효한 사고 window 중 하나라도 감지하면 해당 영상은 TP       | 논문 비교용 주 지표                 |
| `binary_window`   | 고정 window 1건       | 해당 window의 미래 5초 안 사고 여부와 경보 유무 비교        | 관측 시점별 성능                    |
| `binary_bucket`   | bucket 1건            | 각 미래 bucket의 실제 사고 여부와 예측을 직접 비교          | 시간별 상세 진단                    |
| `strict`   | bucket 5건                  | 실제 충돌 bucket만 양성 (`early_credit_s=0`)                   | 시점까지 정확한지 평가             |
| `early`    | bucket 5건                  | 인정 범위의 대표 bucket 하나를 양성으로 인정                  | 선제적 경보 평가                   |
| `weighted` | bucket + 거리               | bucket confusion과 첫 경보 거리의 부분 점수                   | 늦은 예측을 부분 점수로 기록       |

예를 들어 실제 충돌이 관측 종료 3초 뒤라면, horizon 시작 시점의 경보도
`binary_window` 정답,
`strict` 오답, `early` 정답, `weighted=1.0`이다. 충돌 1초 후 경보는 `binary_window`는
정답이지만 `strict`와 `early`는 오답이며, `weighted=0.8`이다.

- `binary_bucket`은 bucket 위치를 직접 평가한다. 5개 bucket을 모두 사고라고 답하면
  실제 사고 bucket의 TP 1개와 나머지 FP 4개가 생긴다.
- `strict`는 early credit을 0초로 둔다. 정확한 bucket 밖의 사고 경보는 FP다.
- `early`는 정답의 `early_credit_s`(main 실험 기본 5초) 범위에서 대표 경보 bucket
  하나를 양성 target으로 삼고, 나머지 bucket도 계속 센다.
- `weighted`는 첫 사고 경보를 기준으로 하며, 충돌이 없는 window는 평균 점수 대신
  false alarm으로 따로 집계한다.

Binary 평가에서는 먼저 window의 정답을 확정할 수 있는지 판단하고, 유효한
window라면 요청한 미래 5초 **전체**의 경보를 사용한다. 예를 들어 관측 `0~5초`,
실제 사고와 녹화 종료가 `6초`, 모델의 사고 예측이 `8초`라면 `binary_window`는
TP이며, 이 window로 `binary_scenario`도 TP가 된다. 정확한 사고 시점을 맞췄다는
뜻은 아니다. Timing/bucket 평가는 녹화 이후의 미확인 bucket을 제외하므로 이
예시에서는 실제 사고 bucket을 놓친 FN이 남는다. 반대로 사고가 관측되지 않았고
미래 5초의 녹화도 부족한 window는 binary 정답을 확정할 수 없어 분모에서 제외한다.
요청 horizon 밖의 경보는 binary에도 반영하지 않는다.

### 2. Compact payload

Batch CLI (`examples/run_validation_batch.py`)는 기본으로 compact + V2 re-ranker를
사용한다. 기본 checkpoint는 프로젝트 기준
`out/pair_reranker_v2_full_model/pair_reranker_v2.json`이다.
`--no-pair-reranker`로 raw compact를, `--payload-profile standard`로 standard
payload를 선택한다(standard에서는 re-ranker도 비활성화).
다른 checkpoint는 `--pair-reranker-model PATH`로 지정한다.
새 payload 생성 시 checkpoint가 없으면 즉시 오류로 중단한다.
`--skip-generate`는 기존 manifest/payload를 재사용하며 새 기본값으로 변환하지 않는다.

previous payload was too verbose... LLM 입력이 너무 크고 불필요한 정보가 많아 compact profile을
추가. including:

- 현재 actor 상태와 짧은 관측 history
- WaypointNet이 예측한 미래 경로
- swept-path geometry로 계산한 동적 actor-pair 후보
- 후보 pair의 거리·접촉·상대 운동 관련 요약

상세한 TTC/상호작용 설명, ASCII BEV 등은 compact 실험에서 제외.

e.g., out/experiments/waypointnet_compact_val104/val/accident/Town01_type001_subtype0001_scenario00004/llm_payload_0-1.json

#### Raw compact vs. Compact + V2

두 조건은 관측 데이터, sensor mode, WaypointNet, window/horizon, LLM provider,
질문 형식이 모두 같다. 차이는 LLM에게 보여줄 actor-pair 12개를 고르는 방식뿐이다.

`Raw compact`는 학습 re-ranker 없이 swept-path의 minimum clearance만 사용한다.

```text
전체 actor pair
    -> minimum clearance가 작은 순서로 정렬
    -> 상위 12개를 compact payload에 포함
    -> LLM 호출
```

`Compact + V2 re-ranker`는 먼저 clearance 기준 후보 50개를 만든 뒤, 각 후보를
V2 MLP로 다시 점수화한다.

```text
전체 actor pair
    -> clearance 기준 후보 50개
    -> V2 score가 높은 순서로 재정렬
    -> 상위 12개를 compact payload에 포함
    -> LLM 호출
```

여기서 `raw`는 가공되지 않은 센서 데이터라는 뜻이 아니라, compact payload에서
V2 re-ranker를 사용하지 않았다는 뜻이다. V2는 clearance 외에도 상대 속도,
closing speed, 예상 접촉 여부, 접촉 시간, heading, 도로/차선 관계, ego 포함 여부,
WaypointNet 경로 통계, 차량 크기와 pair type을 사용한다. 따라서 clearance상 가장
가까운 pair와 실제 충돌 가능성이 가장 높은 pair의 순서가 달라질 수 있다.

| 항목            | Raw compact         | Compact + V2 re-ranker |
| --------------- | ------------------- | ---------------------- |
| 관측·예측 입력 | 동일                | 동일                   |
| 후보 pool       | clearance 상위 50개 | clearance 상위 50개    |
| 최종 payload    | clearance 상위 12개 | V2 score 상위 12개     |
| LLM             | 동일                | 동일                   |

그러므로 두 조건의 성능 차이는 LLM이나 WaypointNet의 차이가 아니라, LLM에 전달된
최종 12개 actor-pair의 차이로 해석해야 한다. V2는 false alarm과 actor 오지목을
줄이는 대신, 실제 사고 pair가 top-12에서 탈락하면 recall이 낮아질 수 있다.

### 3. Pair candidate pool 수정 및 V2 re-ranker

모든 actor pair를 그대로 LLM에 보내면 인접 차량·정차 차량·이미 겹친 라벨 때문에
false alarm이 포화되는 문제. 따라서 다음 두 단계로 수정:

```text
전체 동적 pair
    -> swept-path clearance 기준 후보 최대 50개
    -> V2 pair MLP score 기준 재정렬
    -> payload에는 상위 12개만 직렬화
```

V2 feature에는 pair clearance/contact/time 정보뿐 아니라 상대 속도, closing speed,
heading, road/lane 관계, ego 포함 여부, 두 actor의 속도·가속도·경로 통계 및 pair
type을 포함한다. 모델 파일은 JSON으로 저장되어 PyTorch 없이 inference할 수 있다.

학습 산출물:

- train cache: 483 scenarios, 3,643 windows, 181,993 pairs, 578 positive pairs
- validation cache: 104 scenarios, 732 windows, 36,578 pairs, 200 positive pairs
- model: [`out/pair_reranker_v2_full_model/pair_reranker_v2.json`](out/pair_reranker_v2_full_model/pair_reranker_v2.json)
- 선택 threshold: `0.927234`

관련 코드:

- `traffic_llm/pair_reranker_v2.py` — feature 계산 및 JSON inference
- `examples/collect_pair_reranker_v2.py` — pair cache 생성
- `examples/train_pair_reranker_v2.py` — scenario-grouped CV 및 학습
- `traffic_llm/accident_qa.py` — 실제 payload 생성 시 re-ranker 연결
- `examples/run_deepaccident.py` / `examples/run_validation_batch.py` — CLI 연결

### 4. 전체 validation LLM 실험

동일한 104개/732개 입력에 대해 raw compact와 V2 re-ranker를 각각 실행.
아래 취소선 표는 기존 expanding-prefix 전체 집계이며, 고정 window 성능으로
해석하면 안 된다. 비교에 사용할 `0-5` slice는 다음 절에 별도로 정리했다.

결과 파일:

- raw compact: [`out/experiments/waypointnet_compact_val104/scores_modes.json`](out/experiments/waypointnet_compact_val104/scores_modes.json)
- V2 re-ranker: [`out/experiments/waypointnet_compact_reranker_v2_val104/scores_modes.json`](out/experiments/waypointnet_compact_reranker_v2_val104/scores_modes.json)

> ⚠️ 아래 표의 수치는 expanding-prefix 및 이전 event-level scoring으로 생성된
> legacy 결과다. 현재 fixed-window + bucket-level 기준의 성능으로 해석하거나
> 비교하지 말고, 재실행 후 새 수치로 교체한다.

| 지표                    |     Raw compact | Compact + V2 re-ranker |
| ----------------------- | --------------: | ---------------------: |
| Binary accuracy         |           0.369 |        **0.477** |
| Binary precision        |           0.324 |        **0.345** |
| Binary recall           | **0.934** |                  0.747 |
| Binary F1               |           0.481 |                  0.472 |
| Strict accuracy         |           0.621 |        **0.799** |
| Strict actor precision  |           0.429 |        **0.699** |
| Early detection rate    | **0.834** |                  0.576 |
| Weighted detection rate | **0.847** |                  0.590 |
| False-alarm rate        |           0.720 |                  0.467 |

#### expanding 결과에서 `0-5` window만 extract한 legacy 결과 (재실행 필요)

- 대상: 86 windows (사고 33, 정상 53)
- 104개 시나리오 중 18개는 사고가 5초 이전에 발생해 `0-5` window가 없음

| 지표                | Raw compact | Compact + V2 re-ranker |
| ------------------- | ----------: | ---------------------: |
| Binary accuracy     |       0.453 |                  0.570 |
| Binary precision    |       0.410 |                  0.469 |
| Binary recall       |       0.970 |                  0.909 |
| Binary F1           |       0.577 |                  0.619 |
| Strict accuracy     |       0.627 |                  0.801 |
| Strict detection    |       0.667 |                  0.576 |
| Early accuracy      |       0.633 |                  0.817 |
| Early detection     |       0.818 |                  0.727 |
| Weighted mean score |       0.848 |                  0.758 |
| False-alarm rate    |       0.750 |                  0.462 |

원본 payload와 응답은 각각 다음 디렉터리에 있다.

- Raw: `out/experiments/waypointnet_compact_val104/val/*/*/llm_payload_0-5.json`
- V2: `out/experiments/waypointnet_compact_reranker_v2_val104/val/*/*/llm_payload_0-5.json`

## 재현 방법

### 환경

Python virtual environment는 `.venv/`에 있다. API key와 provider 설정은
`env.sh`에 둔다(비밀값은 저장소에 커밋하지 않는다).

```bash
source env.sh
.venv/bin/python -m pytest
```

### Pair re-ranker 재학습

```bash
.venv/bin/python examples/collect_pair_reranker_v2.py \
  --root /home/sryu/inclab-nas/DeepAccident \
  --carla-maps carla_map --split train \
  --predictor out/predict_model/waypointnet_best.pt \
  --out out/pair_reranker_v2_full/train_pairs_top50.jsonl

.venv/bin/python examples/collect_pair_reranker_v2.py \
  --root /home/sryu/inclab-nas/DeepAccident \
  --carla-maps carla_map --split val \
  --predictor out/predict_model/waypointnet_best.pt \
  --out out/pair_reranker_v2_full/validation_pairs_top50.jsonl

.venv/bin/python examples/train_pair_reranker_v2.py \
  --train out/pair_reranker_v2_full/train_pairs_top50.jsonl \
  --validation out/pair_reranker_v2_full/validation_pairs_top50.jsonl \
  --out out/pair_reranker_v2_full_model
```

### 전체 validation payload 생성 및 audit

```bash
.venv/bin/python examples/run_validation_batch.py \
  --root /home/sryu/inclab-nas/DeepAccident \
  --carla-maps carla_map \
  --out out/experiments/waypointnet_compact_reranker_v2_val104 \
  --all-validation --mode sensor3d \
  --window 5 --stride 1 --horizon 5 --history-stride 1 \
  --payload-profile compact \
  --predictor out/predict_model/waypointnet_best.pt \
  --pair-reranker-model out/pair_reranker_v2_full_model/pair_reranker_v2.json
```

### Gemini 호출

```bash
source env.sh
.venv/bin/python examples/run_validation_batch.py \
  --root /home/sryu/inclab-nas/DeepAccident \
  --carla-maps carla_map \
  --out out/experiments/waypointnet_compact_reranker_v2_val104 \
  --skip-generate --call-llm \
  --provider gemini --model gemini-2.5-flash \
  --language en --workers 4
```

`--skip-existing`를 추가하면 이미 생성된 응답을 재사용할 수 있다. API 호출은 비용이
발생하므로 먼저 10/20개 audit batch로 payload와 응답 형식을 확인하는 것이 안전하다.

## file tree

```text
.
├── README.md                         # 이 문서
├── meeting.txt                       # 실험 목표·평가 논의 원문
├── 2026-08-29-*.txt                  # Claude/VLM 작업 세션 기록
├── deepaccident-primer.md            # DeepAccident 데이터셋·논문 조사 메모
├── requirements.txt                  # Python 의존성
├── env.sh                            # LLM provider 환경 설정
├── carla_map/                        # Town01~Town10HD OpenDRIVE 지도
│
├── traffic_llm/                      # 우리 중앙 융합/예측/LLM pipeline
│   ├── deepaccident.py               # DeepAccident label adapter
│   ├── da_runner.py                  # 시나리오·window 실행기
│   ├── accident_qa.py                # window payload·scoring·LLM 질의
│   ├── pair_reranker.py              # 초기 pair ranking 유틸리티
│   ├── pair_reranker_v2.py           # 동적 pair feature + V2 MLP inference
│   ├── swept_path.py                 # 미래 경로 swept geometry/contact 계산
│   ├── predict_model.py              # 학습된 predictor wrapper
│   ├── predict_nets.py               # WaypointNet 구조
│   ├── prediction.py                 # 경로/기동 예측
│   ├── fusion.py                     # multi-observer actor fusion
│   ├── geometry.py                   # camera 역투영·좌표 변환
│   ├── roadmap.py / kinematics.py    # 차선 매칭·운동학
│   ├── serialize.py                  # 자연어/JSON 직렬화
│   ├── providers.py                  # Claude/OpenAI/Gemini provider
│   └── bev_render.py                 # BEV 시각화
│
├── deepaccident_replicate/           # 논문 비교 대상의 독립 재구현
│   ├── da_baseline.py                # 원본 사고 판정 규칙 이식
│   ├── da_motion.py                  # BEV motion-flow 사고 예측 모델
│   ├── train_da_motion.py             # motion cache 생성·학습
│   ├── eval_da_motion.py              # motion baseline 평가
│   └── apa.py / eval_apa.py           # APA 관련 baseline
│
├── examples/                         # 실행 가능한 CLI/학습 스크립트
│   ├── run_deepaccident.py           # 단일 시나리오 생성·평가
│   ├── run_validation_batch.py        # validation batch/audit/LLM 실행
│   ├── collect_pair_reranker_v2.py   # V2 pair cache 수집
│   ├── train_pair_reranker_v2.py     # V2 pair re-ranker 학습
│   ├── train_predict_model.py        # WaypointNet 학습
│   ├── make_predict_dataset.py       # predictor 학습 데이터 생성
│   ├── evaluate_swept_geometry.py    # geometry 후보 평가
│   └── analyze_llm_audit.py          # 응답·오류·metric 분석
│
├── docs/                             # 알고리즘·입출력·payload 문서
│   ├── waypointnet.md
│   ├── predict_model_io.md
│   ├── predicted_path_algorithm.md
│   ├── payload_modules.md
│   ├── payload_text_structure.md
│   └── rule_audit.md
│
├── tests/                            # unit/integration tests
├── reports/                          # 실험 보고서·prompt·Excel 결과
└── out/                              # 생성 데이터·모델·실험 결과
    ├── predict_dataset/              # WaypointNet 학습/검증 샘플
    ├── predict_model/                # WaypointNet checkpoint
    ├── pair_reranker_v2_full/        # train/validation pair cache
    ├── pair_reranker_v2_full_model/  # 학습된 V2 JSON model
    ├── experiments/                  # audit 및 전체 LLM 실험
    ├── vlm_scene/                    # VLM 입력 export
    ├── baselines/                    # rule/predictor baseline 결과
    └── llm_responses_archive/        # 이전 LLM 응답 보관
```
