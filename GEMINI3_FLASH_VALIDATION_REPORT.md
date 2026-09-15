## 1. setup

evaluated 104 scenarios in *DeepAccident validation split*

- model: `gemini-3-flash-preview`
- input: compact payload
- Actor-pair selection: V2 pair re-ranker
- observation window: fixed 5s
- window stride: 1s
- Prediction horizon: fixed 5s
- Warmup/expanding-prefix window: not used
- future bucket: 1초 단위 5개
- LLM workers: 4

output dir:

```text
out/experiments/waypointnet_compact_v2_fixed_val104_gemini3flash_20260909/
```

결과 파일:

- `audit.json`: payload 생성 및 dataset coverage audit 결과
- `scores_modes.json`: 전체 validation 집계 결과
- `records.jsonl`: window별 응답 및 채점 결과
- `val/*/*/responses/raw_*.json`: Gemini raw API 응답
- `val/*/*/responses/response_*.json`: 파싱된 Gemini 응답

## 2. results

| category                     |   count |
| ---------------------------- | ------: |
| total scenrios               |     104 |
| accident case                |      52 |
| normal case                  |      52 |
| 5s window 생성 가능 시나리오 |      86 |
| window가 없는 시나리오       |      18 |
| generated fixed window      |     324 |
| Gemini responses             | 324/324 |
| rejected                     |       0 |

window가 없었던 18개는 모두 accident case but warmup 없이 완전한 5초 observation을 만들 수 있는
pre-collision 구간 x (모두 사고가 5초 이내에 발생). 이 18개는 LLM에 전달할 payload 자체가 없으므로
모델 평가 대상에서 제외.

## 3. scoring 구조

single observation window가 `0~5s`라면 미래 5s는 다음과 같음:

| bucket | 미래 구간 |
| ------ | --------- |
| k=1    | (5, 6]    |
| k=2    | (6, 7]    |
| k=3    | (7, 8]    |
| k=4    | (8, 9]    |
| k=5    | (9, 10]   |

평가 순서:

```text
Gemini의 bucket별 응답
        ↓
binary_window / binary_bucket / strict / early 계산
        ↓
같은 시나리오의 binary_window를 합침
        ↓
binary_scenario 한 건 생성
```

| grading             | 판정 단위                                                | purpose                                                                              |
| ------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `binary_scenario` | 시나리오 1건                                             | 사고/정상 시나리오 판별                                                              |
| `binary_window`   | fixed observation window 1건<br />(e.g., 0-5s, 1-6s...) | 미래 5초 내 사고 존재 감지                                                           |
| `binary_bucket`   | 미래 1초 bucket 1건<br />(e.g., (5, 6])                   | 실제 사고 시점 직접 비교                                                             |
| `strict`          | bucket                                                   | 정확한 충돌 bucket과 actor id 평가<br />(확인 가능한 실제 사고 timing을 맞혔는지 ?) |
| `early`           | bucket                                                   | 인정 범위 내 early alert 평가                                                       |
| `weighted`        | 사고 window                                              | 첫 경보와 충돌 bucket 거리의 부분 점수                                               |



## 4. censoring rule (exclusion)

`binary_window`에서는 사고 window와 정상 window의 유효 조건이 다르게 정의됨:

- 미래 5s 안에서 실제 사고가 확인되면 positive window의 정답(positive label)이 확정된다. 모델이 경보하면 TP, 경보하지 않으면 FN이다.
- 사고가 없는 negative window는 미래 bucket 5개를 모두 관측해야 정답이 확정.
- 사고가 확인되지 않았고 미래 녹화가 중간에 끝나면 `UNSCORABLE`.
- Positive window의 정답이 확정되면 요청 horizon 안의 모든 모델 경보를 binary에
  반영한다. 정확한 timing bucket은 별도 지표에서 평가한다.
- Timing 지표 (strict, early)는 녹화 이후의 미확인 bucket을 제외하도록 설정.
  e.g., observation window = 0-5s, 사고 = 7.7s (k = 3), 시나리오의 평가용 snapshot coverage 종료시점 = 7.5s인 경우
  실제 사고 시점 자체는 collision ground truth로 확인되므로 positive window 정답은 확정.
  모델이 k=4나 k=5에서 alert했다면 binary에서는 TP로 인정된다. strict에서는 k=4, 5는 미확인 구간이라 제외되지만, 실제 사고 구간 k=3에서 경보하지 않았으므로 k=3은 FN이다.

`UNSCORABLE`은 모델 입력이 없거나 응답이 없었다는 뜻이 아니다. 모델이 해당 window를
관측하고 답했더라도, TP/FP/TN/FN 정답을 확정할 수 없어 최종 metric 분모에서 제외됐다는
뜻이다.

### --warmup flag ?

`no-warmup`인 경우(현재실행) observation = 5s.
정상 시나리오의 `0–5s` window는 미래 `5–10s`를 predict하고, snapshot이 9.5초에서 끝나면 k=5를 확인할 수 없기 때문에
negative `binary_window` 전체가 `UNSCORABLE`이 됨.

Warmup을 켜면 짧은 초반 observation도 추가된다.

```text
0–1초 observation → 미래 1–6초
0–2초 observation → 미래 2–7초
0–3초 observation → 미래 3–8초
0–4초 observation → 미래 4–9초
```

따라서 `data_end=9.5초`인 정상 시나리오에서는 `0–1` 같은 warmup window의 미래
5개 bucket이 모두 관측 가능해져, 모델이 전부 false라고 답하면 TN으로 계산할 수
있다. 이번 validation의 정상 52개 중 51개는 `data_end≥6.0초`였으므로 warmup으로
정상 window가 scorable해질 가능성이 높다. `data_end=5.5초`인 1개는 `1–6초`
horizon도 끝까지 관측할 수 없어 여전히 제외될 수 있다.

사고가 3.6초에 발생한 경우에도 warmup은 `0–1`, `0–2`, `0–3` observation을 만들어
사고 전 입력으로 사용할 수 있다. 다만 이 입력들은 각각 1초, 2초, 3초 history만
가지므로 fixed 5초 observation과 동일한 조건이 아니다.

따라서 warmup은 현재 censoring 문제를 상당 부분 줄일 수 있지만, fixed 5초 결과에
섞어서는 안 된다. `no-warmup` fixed condition을 main 실험으로 유지하고, warmup은
coverage를 높이는 별도 sensitivity condition으로 보고한다.

## 5. example case: negative window (no accident)

시나리오:

```text
val/normal/Town01_type001_subtype0001_scenario00004 
data_end_s = 9.5s
observation window: 0-5s
```

| Bucket | 구간    |      accident |      관측 가능 여부 |
| ------ | ------- | ------------: | ------------------: |
| k=1    | (5, 6]  |          없음 |                가능 |
| k=2    | (6, 7]  |          없음 |                가능 |
| k=3    | (7, 8]  |          없음 |                가능 |
| k=4    | (8, 9]  |          없음 |                가능 |
| k=5    | (9, 10] | 없음으로 기록 | 전체 구간 확인 불가 |

Gemini는 이 window의 모든 bucket을 정상(false)으로 응답했지만 k=5 전체를 확인할 수
없으므로 현재 `binary_window`의 negative-window 규칙에서는 window 전체가 TN이 아니라 `UNSCORABLE`.

반면 `strict`, `early`, `binary_bucket`은 k=1~4를 각각 TN으로 세고 k=5만 timing scoring에서 제외.

i applied this rule because the purpose of the binary window grading is to see 미래 5초 전체에 사고가 있는지 ?
and if we exclude k = 5 and label it as a negative window (with k = 1-4)... the question would be: 관측 가능한 일부 구간에 사고가 있는지?
which makes the prediction horizons vary among the windows
e.g., 0–5 window: future 4초 확인 가능, 1–6 window: future 3초 확인 가능, 2–7 window: future 2초 확인 가능...
이 window들을 모두 같은 TN/FP 로 count 하면 짧은 prediction horizon을 가진 window가 더 쉬운 문제가 되기 때문에... this needed to be sorted

```text
정답: negative로 보이지만 (gt = negative) 미래 5초 전체를 확인하지 못함
model prediction: negative
scoring: TN이 아니라 UNSCORABLE로 분류
```

이 시나리오의 나머지 window도 뒤로 갈수록 확인 가능한 미래 bucket이 줄어듦.

| observation window | 확인 가능한 미래 bucket 수 | `binary_window` |
| ------------------ | -------------------------: | ----------------- |
| 0-5                |                          4 | UNSCORABLE        |
| 1-6                |                          3 | UNSCORABLE        |
| 2-7                |                          2 | UNSCORABLE        |
| 3-8                |                          1 | UNSCORABLE        |
| 4-9                |                          0 | UNSCORABLE        |

따라서 scenario collapse 단계(여러 observation window 결과를 시나리오 한 건으로 aggregate하는 최종 단계)에서도
유효한 normal window가 0개이고, `binary_scenario=UNSCORABLE`이 됨. 모델이 정상으로 맞힌 `0-5` 결과도 TN으로 count 되지 않음.



warmup을 activate하면 초반에 short observation을 추가 가능:

```text
0–1초 observation → 미래 1–6초
0–2초 observation → 미래 2–7초
0–3초 observation → 미래 3–8초
0–4초 observation → 미래 4–9초
```

따라서 `data_end=9.5초`인 정상 시나리오에서는 `0–1s` 같은 warmup window의 미래 5개 bucket이 모두 관측 가능해져서
모델이 전부 false라고 답하면 TN으로 count할 수 있음.
다만 이 observation windows는 각각 1초, 2초, 3초 history만 가지므로 fixed 5초 observation과 동일한 조건 x
`no-warmup` fixed condition을 main으로, warmup은 additional sensitivity condition으로 report 하는 방법...?


## 6. example case: positive window (accident)

```text
val/accident/Town03_type001_subtype0001_scenario00006 
data_end_s=7.5
collision = 7.7s
```

실제 충돌 시점 (7.7s) 값은 LLM payload의 snapshot에서 읽은 값이 아닌
`estimate_collision()`이 원본 scenario의 full-resolution frame trajectory와
collision metadata를 사용해 offline ground truth로 계산.
따라서 이 케이스의 평가용 snapshot coverage는 7.5초까지이며, 개별 LLM 입력은 해당 observation window의 끝까지만 포함한다(0–5 window라면 5초까지). collision GT 7.7초는 모델에 제공되지 않은 평가용 oracle 정보이다.

### observation window 0-5s

- 실제 사고 bucket: k=3 `(7,8]`
- Gemini 사고 alert: k=4, k=5
- `binary_window`: 미래 5초 안에 alert 있으므로 TP
- `strict`: 정확한 k=3을 맞히지 못했으므로 FN

### observation window 1-6s

- 실제 사고 bucket: k=2 `(7,8]`
- Gemini가 k=2에 alert
- `binary_window`: TP
- `strict`: TP

### observation window 2-7s

- 실제 사고 bucket: k=1
- Gemini가 k=1에 경보
- `binary_window`: TP
- `strict`: TP

세 window를 scenario 단위로 합치면:

```text
0-5: TP
1-6: TP
2-7: TP
      ↓
유효한 사고 window 중 하나 이상 감지
      ↓
binary_scenario = TP 1건
```

window TP가 여러 개여도 scenario 결과는 한 개만 count.

## 7. `binary_scenario`  results

시나리오 단위 평가

| metrix            | count |
| ----------------- | ----: |
| TP                |    32 |
| FP                |     1 |
| TN                |     0 |
| FN                |     0 |
| Scorable scenario |    33 |
| Censored scenario |    53 |

- Accuracy: 97.0%
- Precision: 97.0%
- Recall: 100.0%
- F1: 98.5%

그러나 observation 가능한 5s window가 있는 86개 중 33개만 denomitor에 포함됨...

if we activate --warmup, accident = 3.6s와 같은 시나리오도 0–1, 0–2, 0–3 같은 short observation으로 scoreable 하게 만들 수 있음 (prediction horizon is still 5s)
but 다른 시나리오의 5s observation과 input 정보량이 달라지므로 별도의 condition으로 만들어야 할 듯...?

```text
전체 시나리오 104
├── 5초 window 없음: 18 (early exclusion)
└── 5초 window 있음: 86
    ├── binary_scenario 평가 가능: 33
    │   ├── TP: 32
    │   ├── FP: 1
    │   ├── TN: 0
    │   └── FN: 0
    └── binary_scenario UNSCORABLE: 53
        ├── 정상: 51 (again, 정상 window의 경우 미래 5초 bucket을 모두 observe할 수 있어야 하기 때문)
        └── 사고: 2
```

distribution을 보면 accuracy 97%는 `32/33`이며 전체 104개 또는 5초 window가 있는 86개에 대한 정확도 아님.
TN이 0개이기 때문에 정상 시나리오 구분 능력 반영되지 않음.

validation의 정상 52개 중 51개는 `data_end≥6.0초`였으므로 warmup으로 정상 window가 scorable해질 수 있을 듯?
`data_end=5.5초`인 1개는 `1–6초` horizon도 끝까지 관측할 수 없어 여전히 제외.


## 8. `binary_window` results

정확한 사고 시점과 무관하게 미래 5초 안에서 사고 정답과 alert가 각각 존재하는지만 비교.
observation window마다 1개로 count.

| metrix        | count |
| ------------- | ----: |
| TP            |    60 |
| FP            |     0 |
| TN            |     0 |
| FN            |     9 |
| 평가된 window |    69 |

- Accuracy: 87.0%
- Recall: 87.0%
- Precision: 100.0%
- F1: 93.0%

실제 사고가 확인된 69개 window 중 60개에서 미래 5초 내 alert 가능.
마찬가지로 negative window가 denomitor에 없으므로 precision 100%가 의미 없음...

## 9. `binary_bucket` results

각 미래 1초 bucket을 독립적인 binary sample로  count.
예를 들어 실제 사고가 k=3인데 모델이 k=2와 k=3에 경보했다면 k=2는 FP, k=3은 TP.

| 결과            |  수 |
| --------------- | --: |
| TP              |  45 |
| FP              |  98 |
| TN              | 457 |
| FN              |  24 |
| Scorable bucket | 624 |

- Accuracy: 80.4%
- Precision: 31.5%
- Recall: 65.2%
- F1: 42.5%

실제 사고 bucket은 `TP + FN = 69개`, 실제 정상 bucket은 `FP + TN = 555개`인데...
모든 bucket을 정상이라고만 답하는 all-negative baseline도 `555/624=88.9%`이므로 80.4% accuracy가 좋은 성능은 아닌듯...?
false alarm 98개를 줄여야...

## 10. `strict` results

실제 충돌 bucket만 positive로 인정. 따라서 confusion matrix는 `binary_bucket`과 동일.

| 결과 |  수 |
| ---- | --: |
| TP   |  45 |
| FP   |  98 |
| TN   | 457 |
| FN   |  24 |

- 실제 사고 window: 69개
- 정확한 bucket 감지: 45개
- 미검출: 24개
- **Exact detection rate: 65.2%**

`strict`는 confusion matrix 외에 대표 경보의 timing과 actor ID 정확도도 계산한다.

## 11. `early` results

`early`는 실제 사고보다 먼저 alert 한 케이스를 `early_credit_s` 범위 안에서 인정.
현재 early credit은 5s로 설정.
 credit 범위의 모든 bucket을 positive로 만드는 것이 아니라, 인정할 대표 조기 경보 bucket 하나로 positive target을 이동한다.

e.g., 실제 사고가 k=3이고 모델이 k=1에서 alert

| scoring | k=1 | k=3 |
| ------- | --- | --- |
| strict  | FP  | FN  |
| early   | TP  | TN  |

전체 결과:

| metrix | strict | early | 변화 |
| ------ | -----: | ----: | ---: |
| TP     |     45 |    51 |   +6 |
| FP     |     98 |    92 |   -6 |
| TN     |    457 |   463 |   +6 |
| FN     |     24 |    18 |   -6 |

- Exact detection: 45개
- Early credit (5s))으로 추가 인정: 6개
- 미검출(FN): 18개
- **Detection rate/recall: 73.9%**
- Precision: 35.7%


## 12. False alarm...

사고가 없는 scorable window 기준:

- no-event window: 202개
- 하나 이상의 false alert가 있던 window: 60개
- **window false-alarm rate: 29.7%**

Bucket 기준:

- 정상 bucket: 555개
- false-positive bucket: 98개
- **bucket false-alarm rate: 17.7%**

Censored bucket(unscorable)까지 포함하여 Gemini 응답 자체를 별도로 분석하면
normal folder의 250개 window 중 154개, 즉 61.6%에서 적어도 하나의 사고 alert가 있음. 

current report인 29.7%가 더 낮은 이유는 timing metric이 확인 가능한 bucket의 alert만 count 했기 때문...

## 13. Dataset label을 사용한 탐색적 scenario 분석

additionally... censoring을 무시하고 dataset folder의 accident/normal label을 scenario 정답으로 사용.
각 시나리오의 전체 생성 window 중 한 번이라도 사고 alert 있으면 positive로 count 했을 때:

| metrix | count |
| ------ | ----: |
| TP     |    34 |
| FP     |    51 |
| TN     |     1 |
| FN     |     0 |

- Accuracy: 40.7%
- Precision: 40.0%
- Recall: 100.0%
- Specificity: 1.9%
- 사고 시나리오 경보: 34/34
- 정상 시나리오 경보: 51/52

모델이 거의 모든 시나리오에서 사고 alert 발생시키는 positive bias 있는 듯.
`binary_scenario = 97%`에서는 정상 시나리오가 censor 되었으니까 드러나지 않음.

## 14. Actor ID 결과

Actor ID는 strict 또는 early에서 대표 사고 alert로 인정된 bucket에 대해서만 평가.

| scoring | vehicle recall | actor precision |
| ------- | -------------: | --------------: |
| strict  |          86.7% |           84.0% |
| early   |          84.3% |           82.0% |

실제 충돌 차량이 A와 B이고 모델이 A, B, C를 지목했다면 vehicle recall은 100%, actor precision은 66.7%.
timing상 인정된 alert에서는 실제 충돌 차량을 대체로 잘 포함하는듯 ?
Early 성능이 조금 낮은 것은 정확한 사고 순간보다 이른 시점에 alert 할 때차량 pair 특정이 더 어려웠기 때문...?

## 15. Dataset ??

normal folder에 있지만 실제 collision ground truth가 존재

```text
Town07_type001_subtype0002_scenario00028
```

현재 scenario count는 folder outcome을 normal label로 사용하므로 이 사례가 유일한 FP로 처리됨.
아마 label 수정해야 할 듯...

## 16. token and cost

| 항목                      |       count |
| ------------------------- | ----------: |
| Prompt tokens             |   2,824,870 |
| Candidate/output tokens   |     163,306 |
| Thinking tokens           |   2,781,047 |
| API reported total tokens |   5,769,223 |
| estimated cost            |  USD 10.25 |

Gemini 3 Flash preview의 입력 USD 0.50/1M tokens, 출력 및 thinking USD 3.00/1M tokens 기준으로 계산.

- average response time: 40.5s
- max: 112s


## 17. summary

- 사고가 있는 장면에서 사고 존재를 감지하는 sensitivty는 높은듯
- 정확한 사고 bucket 감지는 65.2%, early credit 포함 시 73.9% (which is good i guess..?)
- actor ID도 잘 식별하는 것 같음
- false alert !! 정상 시나리오에서도 사고를 지나치게 많이 예측...
- 현재 `binary_scenario`는 정상 시나리오 대부분을 censor하므로 이걸 메인으로 report 할 수 없을듯...
- `binary_bucket` accuracy 역시 normal bucket 수가 많은 class imbalance의 영향을 크게 받으므로
  precision, recall, false-alarm rate가 더 중요할 듯


## 18. False alert 원인 점검

2026-09-10 재감사 결과와 상세 생략 필드 비교는
[Compact/ranker FP 감사](docs/compact_ranker_fp_audit.md)에 정리했다.
이전 이 절에서 공통 observer를 상호 관측의 근거로 사용하고, 서로 다른 pair validation
집계 수치를 같은 평가처럼 비교한 부분을 정정한다. 기록 밖 normal alert도 확정 FP와 구분한다.

확인된 사실:

- 현재 compact에는 `observed_by`, signed `accel_mps2`, 속도 history가 있다.
  standard의 명시적 관측자 매핑·설명, 과거 도로/차선/기동, closing-pair 및 interaction 요약은 생략한다.
- 이전 compact와 V2 compact의 같은 732개 window에서 actor 정보와 system prompt는 전부 동일하다.
  달라진 것은 pair 목록과 선정 방식 필드뿐이다. V2가 actor 관측 정보를 삭제한 것은 아니다.
- V1과 V2 pair reranker 모두 관측 관계를 feature로 사용하지 않는다.
  V2는 현재 가속도의 절댓값을 사용하므로 다른 입력 고정 시 가속·감속 부호를 구별하지 못한다.
  다만 상대속도·미래 경로 속도 변화는 사용한다.
- V2는 clearance 기반 후보 최대 50개를 점수화해 최대 12개를 수록한다.
  threshold `0.927234`는 필터로 적용되지 않으며 실제 수치 score도 payload에 없다.

| 현재 scorable bucket 분석 | 결과 |
|---|---:|
| FP | 98 |
| FP involved IDs에 수록된 contact pair를 포함 | 96 |
| 위 contact pair의 대표 interval도 일치 | 79 |
| FP involved IDs에 상호 관측 EGO pair를 포함 | 51 |
| 정확히 두 actor를 지목한 상호 관측 FP | 36 |

TP45개 중에도 상호 관측 EGO subset이 32개 있다. 센서 상호 관측을 자동으로
`accident=false`로 바꿀 수는 없다. multi-actor subset은 모델이 특정 pair의 충돌을
정확히 지목했다는 의미가 아니다. normal 폴더 alert364개에는 censored 미래가 포함된다.

구체적인 scorable FP는 normal `Town03_type001_subtype0001_scenario00009`, window `1–6`,
예측 `(7,8]`이다. 서로 관측한 두 차량의 현재 가속도는 −1.43/−0.83 m/s²인데,
제공된 경로는 상대 차량이 계속 동진하여 +1.9초에 접촉한다고 예측하고 Gemini도 이를
high confidence 사고로 채택했다. 후속 `4–9` window history에서는 상대 차량이 t=7에
0.1 km/h, t=8에 0.0 km/h로 정지한다. 이 사례는 미래 정지를 놓친 경로 예측과
그 경로를 따른 사고 판정을 보여주지만, 정지의 원인이나 전체 FP의 원인 비중을 입증하지는 않는다.

따라서 upstream의 상호작용 정보 부족과 downstream의 경로 과신을 모두 조사해야 한다.
Gemini 자체의 bias를 배제하거나 V2만 원인이라고 단정할 근거는 아직 없다.
재현 스크립트: `examples/audit_payload_fp.py`. pipeline·prompt·모델은 변경하지 않았다.

## 19. next step ?

prediction horizon을 시나리오마다 flexible하게 만들면 서로 다른 difficulty의 문제를 같은 denomitor에서 계산하는 이슈...
prediction horizon = 5s로 유지하면서

1) warmup condition 추가 ?

2. false alert가 payload의 trajectory 후보, v2 re-ranker, or gemini verdict 중 어디에서 발생하는지 분석하고 해결 필요
