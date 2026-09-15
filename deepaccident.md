*DeepAccident: A Motion and Accident Prediction Benchmark for V2X Autonomous Driving*

Table 4와 Table 5의 `APA`는 일반적인 classification accuracy가 아니라. 논문이 제안한 `Accident Prediction Accuracy`이며, 충돌 발생 여부뿐 아니라 미래 collision 위치 오차까지 반영하는 TP/FP/FN 기반 지표.

입력, 출력, prediction horizon, 평가 단위, TN 처리 방식이 모두 다름...

## 1. dataset

- CARLA에서 생성한 교차로 주행/충돌 시나리오
- 총 691개 시나리오
  - train: 483
  - **validation: 104**
  - test: 104
- 원본 annotation frequency: 10Hz
- 총 57K annotated V2X frames, 285K annotated samples
- 각 시나리오에는 차량 4대와 infrastructure 1개가 존재
  - 충돌하도록 설계된 두 차량
  - 각 충돌 차량을 뒤따르는 차량 2대
  - 교차로를 관측하는 infrastructure
- 시나리오는 충돌 발생, ego 차량의 계획 경로 완료, 또는 10초 초과 시 종료

데이터셋은 camera와 LiDAR를 모두 포함하지만, 논문의 end-to-end motion/accident baseline은 camera-based 설정.
config에서도 `use_camera=True`, `use_lidar=False`.

## 2. model input/output

### model

- Single-vehicle baseline: BEVerse-tiny
- V2X 모델: V2XFormer
- 각 관측자의 multi-view image에서 BEV feature 추출
- 관측자별 BEV를 ego 좌표계로 spatial warping
- 차량/infrastructure feature를 fusion
- motion prediction과 3D object detection 수행

### 기본 temporal setting

```text
과거 관측: 1초
  = 2Hz 기준 3개 frame (current frame 포함)

미래 예측: 2초
  = 2Hz 기준 4개 future frame
```

`frame`은 camera 한 장이 아니라, 특정 시각에 동기화된 하나의 관측 snapshot을 의미하는듯...?
한 snapshot에는 해당 observer의 6개 camera view가 포함.

e.g., 현재 시점이 `t=5.0s`:

```text
과거 입력 frame:  t=4.0, 4.5, 5.0초
미래 예측 frame:  t=5.5, 6.0, 6.5, 7.0초
```

원본 annotation/sensor data는 10Hz(0.1초 간격)이지만, 모델의 temporal input/output은 2Hz(0.5초 간격)로 구성(**10Hz 중 0.5초 간격으로 선택한 temporal sample**).
따라서 future 4 frames는 **0.5초 간격의 네 미래 시점**이며, 네 개의 1초 bucket과 다름.

default config `receptive_field=3`, `future_frames=4`.

학습은 train split에서 20 epoch 수행, 논문 table 결과는 validation split에서 report.

## 3. accident candidate pair generation

모델이 직접 binary collision label만 출력하는 것은 아님

1. 각 actor의 미래 BEV motion/trajectory를 예측
2. 각 미래 시점에서 actor별 BEV instance polygon 만들기
3. 모든 actor pair의 polygon 간 최소 거리를 계산
4. 거리가 collision threshold보다 작으면 accident candidate로 판정
5. 여러 pair가 조건을 만족하면 가장 가까운 pair를 선택

GT motion에도 같은 post-processing을 적용해 GT accident를 만듦.
즉, GT가 단순히 metadata의 collision flag인 것이 아니라, GT BEV actor polygon의 근접성으로 다시 계산됨.

### threshold ??

the paper reports that they set the dangerous distance = `1.0m`
but the published code says:

```text
GT distance threshold:
0.001 pixel × 0.5 m/pixel = 0.0005m

Prediction distance threshold:
5 pixel × 0.5 m/pixel = 2.5m
```

so when we replicate this, we need to decide between  `1.0m` and `0.0005m/2.5m`

## 4. 6개 motion prediction과 accident 판정

모델은 motion Gaussian distribution을 학습.
평가 시:

- random BEV feature sample 5개
- Gaussian mean feature 1개
- 총 6개의 motion prediction 생성

Motion prediction metric은 mean prediction만 사용.
반면 accident prediction에서는 **6개 중 하나라도 collision으로 판정되면 accident prediction으로 처리.**

```text
6개 trajectory 중 하나라도 collision
→ accident prediction = positive
```

miss를 줄이기 위한 safety-first rule.

## 5. APA calculation ?

예측한 두 사고 차량의 위치가 GT 위치에서 얼마나 벗어나도 TP로 인정 ?
D: 예측 collision pair의 위치와 GT collision pair의 위치 사이의 총 위치 오차

```text
D = {5m, 10m, 15m}
```

각 `d`에 대해:

```text
APA_d = TP_d / (TP_d + 0.5 × FP_d + 0.5 × FN_d)
```

**최종 APA는 세 threshold의 평균.**

```text
APA = (APA_5 + APA_10 + APA_15) / 3
```

여기서 `TP_d`가 되려면:

1. prediction과 GT 모두 accident를 표시하고,
2. 두 collision actor의 위치 오차 합이 `d` 이하이어야 함.

위치 오차는 두 actor 각각의 Euclidean position error를 더한 값.
Actor 순서가 바뀐 경우까지 고려해 두 가지 대응 중 더 작은 합을 사용.

예를 들어 `d=10m`에서:

```text
GT collision = true
prediction collision = true
actor A 오차 = 4m
actor B 오차 = 5m
총 오차 = 9m
```

이면 `TP_10`.

오차가 10m를 초과하면 해당 threshold에서는 TP가 아님.
예측 collision은 있었지만 위치가 threshold 밖이면 코드에서는 FP로 처리, FN으로도 추가하지 않는 구현 있음.

more details...
**같은 예측을 세 가지 오차 허용 기준으로 regrading ???**

for example 두 collision actor의 위치 오차가:

```text
차량 A 오차 = 3m
차량 B 오차 = 4m
총 position error = 3 + 4 = 7m
```

this is evaluated by:

```text
D=5m  기준  → 실패 (7m > 5m)
D=10m 기준  → 성공 (7m ≤ 10m)
D=15m 기준  → 성공 (7m ≤ 15m)
```

각 기준에서 TP/FP/FN을 따로 누적한 뒤 APA를 계산, 세 APA의 평균을 최종 APA로 report.

again for example...

```text
D=5m:  TP=2, FP=1, FN=1  → APA_5  = 0.667
D=10m: TP=3, FP=1, FN=1  → APA_10 = 0.750
D=15m: TP=4, FP=1, FN=0  → APA_15 = 0.889

final APA = (0.667 + 0.750 + 0.889) / 3 = 0.769 = 76.9
```

### APA vs. our accuracy

- TN을 count 하지 않음
- 정상 장면에서 prediction이 없으면 TP/FP/FN 어느 것도 증가하지 않는다.
- GT 사고가 있는 장면에서 위치 오차가 threshold 밖인 경우 코드 기준 FP로 count (FN은 증가 x)
- GT가 없는 경우 코드에서는 평균 motion sample만 FP 검사에 사용...
- 

## 6. additional TP

position threshold를 `10m`로 고정해 TP를 정한 뒤, TP prediction만 대상으로 다음을 average:

- `ID error`
  - 예측 actor pair와 GT actor pair가 일치하면 0, otherwise 1
- `Position error`
  - 두 collision actor의 위치 오차 합 (단위: meter)
- `Time error`
  - predicted collision timestamp와 GT collision timestamp의 절대 차이 (단위: second)

Table 4의 `ID error=0.06`은 TP prediction들에 대한 평균 ID error.

Motion metric은 `mIoU`, `VPQ`, 3D detection metric은 center-distance threshold `{1, 2, 4}m`에서 평균한 `mAP`.

## 7. Table 4

| setting                        | mIoU |  VPQ |  APA | ID error | Position error |  mAP |
| ------------------------------ | ---: | ---: | ---: | -------: | -------------: | ---: |
| Single vehicle                 | 43.8 | 31.6 | 61.9 |     0.12 |          3.20m | 26.5 |
| Ego + behind vehicle           | 51.3 | 39.2 | 66.8 |     0.11 |          2.87m | 36.3 |
| Ego + other vehicle            | 52.1 | 39.9 | 67.4 |     0.10 |          2.85m | 36.6 |
| Ego + infrastructure           | 52.7 | 40.1 | 68.1 |     0.10 |          2.80m | 36.8 |
| Four vehicles                  | 55.5 | 42.5 | 68.9 |     0.07 |          2.91m | 39.0 |
| Four vehicles + infrastructure | 56.2 | 44.0 | 69.5 |     0.06 |          2.45m | 40.8 |

purpose: V2X observer 추가했을 때 성능이 어떻게 변하는지 ?

- the more observers, the better performance

## 8. Table 5: prediction horizon comparison

| Prediction horizon | All data | TTC 1s | TTC 2s | TTC 3s | TTC 4s |
| ------------------ | -------: | -----: | -----: | -----: | -----: |
| 2s                 |     61.9 |   74.7 |   28.7 |      - |      - |
| 3s                 |     50.5 |   71.5 |   25.7 |   21.2 |      - |
| 4s                 |     35.4 |   56.3 |   20.4 |   14.6 |   10.2 |

column: 모델이 예측하도록 학습된 prediction horizon
row: 실제 collision까지 남은 TTC(Time-To-Collision) 구간

for example:

- `2s / TTC 1s = 74.7`: 2초 예측 모델을 사고 1초 전에 평가한 APA
- `4s / TTC 4s = 10.2`: 4초 예측 모델을 사고 4초 전에 평가한 APA

trade-off 존재:

- 짧은 horizon 모델은 전체 APA가 높음
- 긴 horizon 모델은 더 일찍 사고를 예측할 수 있지만 위치 prediction 정확도 낮음
- 4초 horizon 모델만 TTC 4초 구간 평가 가능

## 9.  code ???

the paper says... 각 미래 timestamp에서 polygon 거리를 계산해 사고 시점을 찾는다...

그런데 claude says
`multi_gpu_test.py`의 `eval_accidents()`를 보면 `for t in range(sequence_length)` 안에서 GT/prediction을 계산한 뒤, 결과 처리 부분이 loop 바깥에 위치해 있음
현재 코드 그대로라면 마지막 future frame의 `t`만 결과 계산에 사용되는 것으로 보임

1. 논문이 의도한 frame-wise 평가
2. repository 코드 그대로의 평가
3. 이 구현 문제를 수정한 debugged 평가

need to decide

## 10. compare to ours ?

| category       | DeepAccident                 | ours                                            |
| -------------- | ---------------------------- | ----------------------------------------------- |
| input          | 6-camera raw image 기반 BEV  | 구조화된 payload, actor metadata, waypoint 정보 |
| output         | 모든 actor의 미래 trajectory | LLM의 사고 여부, timing, actor pair             |
| 기본 horizon   | 2초                          | 5초                                             |
| 관측 history   | 1초                          | 현재 fixed window 설정                          |
| 평가 단위      | temporal anchor/sample       | window/scenario                                 |
| 사고 판정      | BEV polygon distance         | waypoint/candidate pair와 LLM 응답              |
| 주 지표        | APA                          | binary_scenario                                 |
| TN 처리        | 없음                         | 있음                                            |
| 위치 threshold | 5/10/15m                     | binary metric에는 없음                          |

possible solutions would be...

1. waypoint trajectory에 DeepAccident's polygon collision post-processing 적용
2. 2Hz, 1초 history, 2초 prediction 조건으로 세팅...?
3. 위치 threshold `{5, 10, 15}m`에서 APA 계산
