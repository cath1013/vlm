# 실험 보고서

`traffic_llm`(iitp)과 BEV-VLM(`~/vlm`) 두 파이프라인을 같은 데이터·같은 정답으로
붙여 놓고 잰 결과. 2026-08-27 하루치다.

두 프로젝트는 DeepAccident val 104 시나리오를 공유하고, **좌표계(CARLA 월드)와
신원(CARLA actor id)이 오프셋 없이 같으며**, 충돌 라벨이 104/104 일치한다
([03 · 두 프로젝트는 이미 맞물린다](03-method-notes.md)). 그래서 한쪽의
구성요소를 다른 쪽에 끼워 넣고 **무엇이 바뀌는지 하나씩 격리**할 수 있었다.

---

## 한 문단 요약

V2X 사고 예측에서 **실질적으로 예측 가능한 범위는 2초 미만**이고, 그 안에서 성능을
올리는 것은 **연장한 궤적의 정확도 하나뿐**이다. 장면 설명을 더 주는 것은 — 그것이
지도에 맞춘 정확한 수치이더라도 — **정확도를 떨어뜨린다**. 표를 만드는 인지 출처를
바꾸는 것은 소수점까지 아무 효과가 없다.

기하 표를 받은 조건은 전부 **0.728** 에서 멈췄고, 그 값은 같은 표에 임계값 하나를 건
것과 같다 — LLM 이 산술 이상으로 뽑아낸 것이 없었다는 뜻이다. 그 표를 **WaypointNet**
으로 다시 계산하자 **0.786** 이 됐고 오탐이 절반 아래로 줄었다.

WaypointNet 은 DeepAccident 논문의 모델(V2XFormer)이 아니라 **이 프로젝트에서 직접
만들어 학습시킨 것**이다 — 차의 현재 상태와 과거 궤적을 받아 미래 좌표를 찍는 작은
신경망으로, 15만 표본으로 학습했다.

---

## 보고서

| | 내용 | 한 줄 |
|---|---|---|
| **[01](01-prediction-horizon.md)** | 몇 초 앞까지 예측할 수 있는가 | k=1 AUC 0.813 → k=5 0.423. 2초 너머는 운동학이 결정하지 않는다 |
| **[02](02-what-helps.md)** | 무엇이 성능을 올리는가 | 설명을 더 주기 ✗ · 위치 출처 바꾸기 ✗ · **미는 방식 바꾸기 ✓ (+0.058)** |
| **[03](03-method-notes.md)** | 재실험 전에 알아야 할 것 | LLM 잡음 = 예측기 차이 · Qwen은 산술을 못 한다 · 누설 감사 |
| **[04](04-prompts.md)** | 조건별 실제 프롬프트 | 코드 경로에서 뽑은 것 — 손으로 옮기지 않았다 |

**보고서마다 대상이 다르다.**

| | 무엇을 | 시나리오 | 판정 단위 |
|---|---|---|---|
| 01 | `traffic_llm` 자체 과제 | val **104** (충돌 51 / 무충돌 53) + train 120 | 창마다 1초 구간 5개 — 채점 가능 2,394개 |
| 02 | `~/vlm` 중앙 추론 과제 | val **103** (충돌 51 / 무충돌 52) | 시나리오당 판정 1개 |
| 03 | 함정·검증 | 항목마다 다름 (본문에 적었다) | |

**103 과 104 의 차이는 시나리오 하나다** — `Town10HD_…_scenario00034`(무충돌). vlm 의
sweep 이 균형 표집을 하면서 빠졌다. 그 하나 말고는 두 집합이 완전히 같다.

두 과제는 **묻는 것이 다르므로 숫자를 가로질러 비교하면 안 된다.** 01 의 AUC 0.813 과
02 의 정확도 0.728 은 같은 축이 아니다 — 전자는 1초 구간 하나를 가르는 능력이고
후자는 시나리오 전체에 대한 판정이다. 두 보고서를 잇는 것은 **같은 개입의 효과 크기**
(연장 방식 교체가 양쪽에서 +0.058)이지 절대값이 아니다.

## 용어

읽기 전에 이것만. 나머지는 본문에서 처음 나올 때 풀어 썼다.

**파이프라인의 세 단계**

| 말 | 영어 | 뜻 |
|---|---|---|
| **인지** | perception | 지금 뭐가 어디 있나 — 센서로 알아내는 단계 |
| **연장** | extrapolation | 앞으로 어디로 갈까 — 미래 위치를 찍는 단계 |
| **정책** | policy | 그래서 부딪히나 — 답을 정하는 규칙 |

**계산에 쓰는 것**

| 말 | 영어 | 뜻 |
|---|---|---|
| **기하** | geometry | 좌표와 거리 계산. 학습도 모델도 없다 |
| **기하 표** | gap table | 모든 차량 쌍의 "지금 간격"과 "1초 안에 가장 가까워지는 간격". 중심이 아니라 **차체끼리** 잰 거리 |
| **기준선** | baseline | 이것보다 못하면 그 방법은 쓸모없다는 선 |
| **등속 연장** | constant-velocity | 지금 속도로 계속 직진한다고 치고 미는 것 |
| **예측 범위** | horizon | 몇 초 앞까지 묻는가 (5초 또는 1초) |
| **구간 k** | bucket | 그 범위를 1초씩 쪼갠 조각. k=1 은 0~1초 |
| **액터** | actor | 여러 관측자가 본 것을 합쳐 확정한 차·사람 하나 |

**결과를 읽을 때**

| 말 | 영어 | 뜻 |
|---|---|---|
| **탐지** | detection | 충돌이 일어나는가 |
| **식별** | identification | 누가 부딪히는가 |
| **오탐** | false positive | 아닌데 사고라고 한 것 |
| **미탐** | false negative | 사고인데 놓친 것 |
| **정밀도** | precision | "사고"라 한 것 중 진짜 비율 |
| **재현율** | recall | 진짜 사고 중 잡아낸 비율 |
| **AUC** | — | 값 하나로 정답을 얼마나 가르는지. **0.5 면 정보 없음** |
| **균형정확도** | balanced accuracy | (사고를 맞힌 비율 + 무사고를 맞힌 비율) ÷ 2. 사고가 10% 뿐이라 그냥 정확도로 보면 "아무 일 없다"고만 해도 0.88 이 나온다 |

## 배경

데이터셋과 원 논문이 처음이면 [`../deepaccident-primer.md`](../deepaccident-primer.md)
부터 — 규모·구성 수치는 논문 인용이 아니라 데이터에서 재계산했고, 논문과 공개 코드가
어긋나는 지점도 모아 두었다.

파이프라인 자체는 [`../docs/`](../docs) — payload 가 어떤 절로 구성되는지
(`payload_text_structure.md`), 어떤 모듈이 무엇을 계산하는지(`payload_modules.md`),
경로 예측 모델의 입출력(`predict_model_io.md`)과 구조(`waypointnet.md`).

## 재현

기하 실험은 API 호출이 없고, LLM 조건은 `gemini-2.5-flash` 로 조건당 103 호출 ·
약 $0.5 · 2분.

```bash
# 기하 기준선 + 구간별 분해 (키 불필요)
.venv/bin/python examples/baselines_accident_qa.py \
    --root /home/sryu/inclab-nas/DeepAccident --carla-maps ./carla_map \
    --fit-split train --eval-split val --mode sensor3d --extrapolation cv \
    --records-out out/baselines/records_val_sensor3d.jsonl
.venv/bin/python examples/analyze_records.py out/baselines/records_val_*.jsonl

# vlm 조건에 iitp 기하 표를 끼워 넣기
.venv/bin/python examples/export_gaptable_for_vlm.py \
    --root /home/sryu/inclab-nas/DeepAccident --carla-maps ./carla_map \
    --split val --horizon 1.0 --out out/vlm_scene/gaps_val_h1.0.json
cd ~/vlm && source /home/sryu/inclab-nas/sryu/iitp/env.sh
.venv-vlm/bin/python scripts/add_condition.py \
    --condition K_iitp_geometry_only --label K_iitp_predicted \
    --iitp-gaps .../gaps_val_h1.0.json --gap-variant predicted
```

## 한계

- **LLM 조건은 1회 실행이고 반복이 없다.** 같은 입력을 두 번 부르면 구간 판정이
  20개 중 3개 뒤집힌다([03 · LLM 은 같은 입력에 다르게 답한다](03-method-notes.md)). 0.786 을 확정하려면
  재실행이 필요하다. 다만 오탐 11 → 6 은 그 잡음보다 큰 폭이다.
- 모델 하나(`gemini-2.5-flash`), 프롬프트 한 벌, 예측 범위 하나(1.0초)에서 잰 값이다.
- 인지는 **데이터셋 레이블의 3D 위치**를 쓴다(`observation_mode=sensor3d`). 단안
  카메라 위치 추정으로 바꾸면 유일하게 풀리는 구간의 신호가 무너진다
  ([01 · 위치를 직접 재느냐 짐작하느냐](01-prediction-horizon.md)).
