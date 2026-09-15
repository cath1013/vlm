# Gemini 3 Flash False Alert 분석

작성일: 2026-09-10

## 범위

기존 Gemini 3 Flash validation 결과에서 `val/normal` 52개 시나리오, 250개
window의 응답을 재분석했다. API를 다시 호출하지 않고 저장된 parsed response와
compact payload만 사용했다.

정상 folder의 52개 중 1개(`Town07_type001_subtype0002_scenario00028`)는 실제
collision ground truth도 있어 dataset label 예외로 별도 확인이 필요하다.

## 기본 현상

- Normal window: 250개
- 적어도 하나의 사고 alert가 있는 window: 154개 (61.6%)
- 사고 alert bucket: 364개
- Alert가 없는 window: 96개
- 정상 scenario 중 alert가 있는 scenario: 51/52
- Alert confidence: high 336개, medium 28개, low 0개
- Response schema/parse issue: 0개

Window별 alert bucket 수는 다음과 같다.

| alert bucket 수 | window 수 |
|---:|---:|
| 0 | 96 |
| 1 | 43 |
| 2 | 45 |
| 3 | 36 |
| 4 | 27 |
| 5 | 3 |

Alert가 있는 window에서는 평균 2.36개 bucket에서 연속 또는 반복적으로
사고라고 답했다. 한 번의 우연한 단발 응답만이 문제인 것은 아니다.

## Payload 후보와의 연결

각 alert의 `involved_actor_ids`를 payload의 `predicted_closest_pairs`와 비교했다.
Payload pair의 주요 필드는 다음과 같다.

```text
actor_a, actor_b, minimum_clearance_m,
time_after_observation_s, interval_index,
predicted_contact, joint_path_probability,
clearance_at_observation_m
```

결과:

| 분류 | alert 수 | 비율 | 의미 |
|---|---:|---:|---|
| 정확히 2개 actor ID가 payload pair와 일치 | 288 | 79.1% | Gemini가 payload pair를 그대로 지목 |
| 여러 actor를 지목했지만 그 안에 contact pair 존재 | 69 | 19.0% | 후보 pair를 과다하게 확장해서 지목 |
| payload에 있지만 `predicted_contact=false`인 pair와만 연결 | 4 | 1.1% | near miss를 collision으로 해석했을 가능성 |
| payload 후보와 actor ID overlap 없음 | 3 | 0.8% | payload에 근거하지 않은 actor pair 출력 |

즉 364 alert 중 357건(98.1%)은 적어도 하나의 payload 후보 pair와 연결된다.
이것은 Gemini가 완전히 임의의 사고를 만들어낸다기보다, 입력에 있는 위험 후보를
사고로 확정하는 패턴임을 보여준다.

정확히 일치한 288건은 모두 다음 payload 값을 가졌다.

```text
predicted_contact = true
minimum_clearance_m = 0.0
joint_path_probability = 1.0
```

Pair rank는 rank 1이 71건, rank 2가 50건, rank 3이 46건이었다. 따라서 낮은
순위의 pair만 문제라고 할 수 없다. 상위 3개 안의 contact 후보도 167건이었다.

## 실제 예시 1: upstream geometry 후보를 Gemini가 확정

시나리오:

```text
val/normal/Town01_type001_subtype0001_scenario00004
window: 2-7
```

Gemini 응답:

```text
k=5: accident_expected=true
actors: EGO_other_vehicle, V001
reason: stationary vehicle V001과 collision
confidence: high
```

해당 payload 후보:

```text
pair: EGO_other_vehicle, V001
minimum_clearance_m: 0.0
predicted_contact: true
time_after_observation_s: 4.4
interval_index: 5
```

Normal trace의 실제 collision label은 false다. 따라서 이 alert는 Gemini가
payload에 없는 pair를 hallucination한 사례가 아니라, WaypointNet의 미래 경로와
swept geometry가 만든 predicted contact를 Gemini가 실제 사고로 확정한 사례에
가깝다.

## 실제 예시 2: actor pair 과다 지목 및 후보 외 ID

시나리오:

```text
val/normal/Town07_type001_subtype0002_scenario00019
window: 2-7
```

Gemini는 다음과 같이 답했다.

```text
k=3,4,5: accident_expected=true
actors: V010, V011
confidence: high
```

그러나 payload의 `predicted_closest_pairs`에는 V010/V011 pair가 없었다. payload에
있는 pair들은 예를 들어 다음과 같았고 모두 `predicted_contact=false`였다.

```text
EGO_ego_vehicle, EGO_other_vehicle_behind: clearance 0.29m
EGO_other_vehicle, T002: clearance 0.66m
...
```

이 유형은 Gemini가 전체 actor 목록을 보고 payload의 12개 후보 pair에 없는 pair를
구성한 사례로, 별도의 schema/evidence gating이 필요하다.

## 실제 예시 3: non-contact near miss를 collision으로 해석

시나리오:

```text
val/normal/Town04_type001_subtype0002_scenario00014
window: 0-5
```

Gemini는 V002/V003 pair에 대해 k=3~5에서 collision을 예측했다. 그러나 payload의
관련 pair는:

```text
EGO_other_vehicle, V002: clearance 1.44m, predicted_contact=false
EGO_other_vehicle, V003: clearance 1.59m, predicted_contact=false
```

이 유형은 contact flag가 false인데도 trajectory overlap 또는 stopping failure를
근거로 사고를 선언한 사례다. 총 4 alert로 수는 작지만, prompt와 gating으로 직접
줄일 수 있는 LLM 판단 오류 후보다.

## Alert timing 분포

정상 alert bucket은 다음과 같다.

| Bucket | alert 수 |
|---:|---:|
| k=1 | 18 |
| k=2 | 101 |
| k=3 | 74 |
| k=4 | 64 |
| k=5 | 31 |

k=2~4에 집중되어 있어, 한 시점의 random output보다 WaypointNet predicted path의
지속적인 미래 geometry를 Gemini가 반복해서 읽고 있을 가능성이 높다.

## 원인별 우선순위

현재 증거를 기준으로 한 우선순위는 다음과 같다.

### 1. WaypointNet/swept geometry false contact

정확히 pair가 일치하는 288건 모두 `predicted_contact=true`이고 clearance가 0이다.
따라서 실제 normal trace에서 해당 pair의 ground-truth future clearance를 계산해,
predicted contact가 얼마나 자주 실제 near miss인지 확인하는 것이 최우선이다.

### 2. LLM의 multi-actor over-attribution

69건은 Gemini가 3~6개 actor를 한 번에 지목했다. Prompt와 출력 검증에서
“정확히 두 actor만 지목하고, 반드시 supplied predicted_closest_pairs 중 하나를
선택하라”는 제약을 강화해야 한다.

### 3. Candidate 외 actor pair 출력

3건은 payload 후보와 actor ID overlap이 전혀 없었다. 이 응답은 unsupported alert로
별도 표시하거나 재질문/무효 처리하는 evidence gate가 필요하다.

### 4. Non-contact near miss 오판

4건은 후보 pair가 존재하지만 `predicted_contact=false`였다. `predicted_contact`
조건을 hard gate로 적용하면 제거할 수 있지만, 이 경우 실제 사고 recall 손실을
반드시 별도 측정해야 한다.

### 5. Scenario-level any-alert aggregation

한 window에서 한 번의 alert만 있어도 scenario FP가 되는 현재 collapse 규칙은
false alert에 민감하다. 같은 pair의 연속 2개 bucket alert, 또는 contact pair의
반복을 요구하는 후처리 정책을 기존 응답에 offline 적용해 볼 수 있다.

## 다음 분석/실험 순서

API를 다시 호출하기 전에 다음을 기존 결과로 계산한다.

1. Normal alert pair의 predicted clearance/contact와 실제 ground-truth future
   clearance 비교
2. `predicted_contact=true` hard gate의 recall/false-alarm trade-off
3. 같은 pair 2개 연속 bucket 조건
4. 정확히 2개 actor이고 후보 pair에 포함된 경우만 허용하는 evidence gate
5. 현재 any-alert와 비교한 scenario/window/bucket confusion matrix

이 분석으로 upstream geometry 문제인지, Gemini interpretation 문제인지, temporal
aggregation 문제인지 분리한 뒤에야 V2 K/threshold나 prompt를 변경하는 것이 안전하다.

