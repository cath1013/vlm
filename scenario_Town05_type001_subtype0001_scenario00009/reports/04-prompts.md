# 조건별로 모델에게 실제로 보낸 것

[02 · 무엇이 성능을 올리는가](02-what-helps.md) 의 아홉 조건이 각각 무엇을 받았는지.

**손으로 옮겨 적은 것이 아니라** 실행에 쓰인 코드(`bev_prompts.central_prompt`)로 다시
뽑았다. 프롬프트를 바꾸면 이 문서도 다시 뽑아야 한다 — 손으로 고치면 실제로 보낸 것과
어긋나고, 그 어긋남은 나중에 결과를 다시 볼 때 드러나지 않는다.

전부 **같은 시나리오 하나**에서 뽑았으므로 조건끼리 나란히 비교할 수 있다.

```
시나리오  Town01_type001_subtype0001_scenario00004
정답      사고 있음 · 방향 same · 충격 부위 front · 승합차와 승용차
```

---

## 보내는 것의 구조

모델에 보내는 한 덩어리를 **payload** 라 한다. 두 부분이다.

```
system    누구로서 답하라는 지시                 425자, 모든 조건 공통
user      실제 자료 + 분류 정의 + 질문 + 답 형식   조건마다 다름
```

**바뀌는 것은 user 의 앞부분(자료)뿐이다.** 분류 정의·질문·답 형식은 아홉 조건이
전부 같다. 그래야 차이가 자료에서 왔다고 말할 수 있다.

| 모델이 자료로 받은 것 | user 길이 | 이미지 |
|---|---:|:--:|
| 기하 표만 (**등속**) | 3,018자 | |
| 기하 표 + 조감도 이미지 | 3,377자 | O |
| 관측자 리포트 2개 + 기하 표 | 8,249자 | |
| 기하 표 + iitp 장면 설명 전체 | 18,764자 | |
| 기하 표 + iitp 설명(판정 라벨 뺀 것) | 17,079자 | |
| 위 + 조감도 이미지 | 17,650자 | O |
| **iitp 가 만든 표 (등속)** | 3,102자 | |
| **iitp 가 만든 표 (WaypointNet)** | 3,102자 | |
| **위 + 조감도 이미지** | 3,461자 | O |

---

## 모든 조건이 똑같이 받는 것

### system — 누구로서 답하라

```text
You are the central reasoning module of a cooperative traffic-safety system at a
road junction. You receive written reports from one or more sensor platforms, each of which sees only part of the scene.

Weigh those reports. They are incomplete: some road users are missed entirely, and a platform's own risk ranking may be wrong. Reach your own conclusion rather than copying theirs.

Do not assume either outcome in advance.
```

### 충돌 분류 정의

답할 때 쓸 말을 정해 준다. 이것이 없으면 모델이 "옆에서 받힘" 같은 제멋대로의
표현을 내서 채점이 안 된다.

```text
COLLISION TAXONOMY
relative_direction - how the two vehicles' headings relate just before impact:
  "same"     : travelling in broadly the same direction (includes one turning
               across the other from the same side of the junction)
  "opposite" : approaching each other head-on or from opposing approaches

impact_zone - which part of the struck vehicle is hit:
  "front"      : struck squarely on the front
  "front_side" : struck on a front corner or forward flank
  "rear"       : struck squarely on the rear
  "rear_side"  : struck on a rear corner or rearward flank
```

### 질문과 답 형식

```text
TASK
Predict whether a collision occurs between any two road users within the next
1.0 seconds. It may involve a sensor platform itself, but very often it
is between two other vehicles.

Reply with a single JSON object and nothing else:
{
  "collision": boolean - will a collision occur between two road users?
  "confidence": number 0.0-1.0
  "relative_direction": one of ['same', 'opposite'], or null if collision is false
  "impact_zone": one of ['front', 'front_side', 'rear', 'rear_side'], or null if collision is false
  "involved": list of exactly 2 classes from ['car', 'van', 'truck', 'motorcycle', 'cyclist', 'pedestrian'], or [] if collision is false
  "description": 1-3 sentences describing what happens and why
  "avoidance": one concrete maneuver that would prevent it, or null if collision is false
  "used_platforms": ["which reports actually changed your answer"],
  "disagreement_with_observers": ["any observer claim you are overriding, and why"]
}
```

---

## 기하 표만 — 등속 (constant velocity) · 0.728

user 의 자료 부분이 **아래 표 하나가 전부**다. 이것이 여섯 조건이 넘지 못한 천장이고,
같은 표의 최솟값에 임계값 하나를 건 것과 정확도가 같다.

표를 읽는 법도 함께 준다 — "지금 0.00m 이면 그냥 붙어 서 있는 것이지 충돌 증거가
아니다" 같은 것. 이 안내가 없으면 모델이 주차된 차 줄을 충돌로 읽는다.

```text
COMPUTED PROXIMITY (constant-velocity extrapolation, vehicle footprints)
  Using each road user's actual rectangular footprint, not centre-to-centre
  distance. 0.00 m means the two bodies are touching or overlapping.

  'gap now'  = separation at this instant.
  'min gap'  = closest they get within the horizon if both hold course.

  A pair already at 0.00 m now is simply close together -- stopped in a
  queue or side by side -- and is not by itself evidence of a collision.
  A pair with a wide gap now that closes to 0.00 m is converging.

  pair          gap now (m)  min gap (m)  at t+ (s)
  2 & 4                5.74         0.00        0.8
  5 & 6                4.01         1.75        0.4
  1 & 4               19.75        11.23        1.0
  1 & 2               12.25        12.25        0.0
  4 & 5               12.41        12.41        0.0
  2 & 6               22.44        12.52        1.0
  2 & 5               14.10        14.10        0.0
  4 & 6               19.15        18.29        1.0
  ... 2 further pairs, all with larger gaps

Read the two gap columns together:
  wide now, 0.00 m later  -> converging. This is the dangerous one.
  0.00 m now and later    -> already side by side or queued, usually parked cars.
  narrow now, wider later -> diverging.
```

## iitp 가 만든 표 — 등속 (constant velocity) · 0.728

표를 만드는 재료만 바꿨다. vlm 은 LiDAR 점을 덩어리로 묶어 차를 세고, iitp 는 다섯
관측자가 본 것을 합쳐 차 목록을 확정한다. **미는 방식은 등속으로 똑같다.**

정확도가 소수점까지 같다 — 위치를 누가 알아냈느냐는 아무 차이가 없었다.

```text
COMPUTED PROXIMITY (constant-velocity extrapolation, vehicle footprints)
  Using each road user's actual rectangular footprint, not centre-to-centre
  distance. 0.00 m means the two bodies are touching or overlapping.

  'gap now'  = separation at this instant.
  'min gap'  = closest they get within the horizon if both hold course.

  A pair already at 0.00 m now is simply close together -- stopped in a
  queue or side by side -- and is not by itself evidence of a collision.
  A pair with a wide gap now that closes to 0.00 m is converging.

  pair          gap now (m)  min gap (m)  at t+ (s)
  EGO_ego_vehicle_behind & EGO_other_vehicle         7.17         0.18        0.8
  EGO_ego_vehicle & V001         4.22         2.07        0.8
  EGO_ego_vehicle & V002        13.65         3.95        1.0
  V002 & M003          7.01         4.39        1.0
  V007 & M008          8.74         8.74        0.0
  V006 & V007          8.79         8.79        0.0
  N004 & V010          9.68         9.54        0.3
  EGO_other_vehicle & EGO_other_vehicle_behind        11.33        11.33        0.0
  ... 142 further pairs, all with larger gaps

Read the two gap columns together:
  wide now, 0.00 m later  -> converging. This is the dangerous one.
  0.00 m now and later    -> already side by side or queued, usually parked cars.
  narrow now, wider later -> diverging.
```

## iitp 가 만든 표 — WaypointNet · 0.786

차량 목록·좌표·시점이 위 표와 **전부 같고, 앞으로 미는 방식만 다르다.**
등속은 "계속 직진한다"고 치는 것이고, WaypointNet 은 15만 표본으로 학습한 신경망이
"이 차는 1초 뒤 여기"를 찍어 준다.

위 표와 나란히 놓고 보면 같은 쌍의 `min gap`(가장 가까워지는 간격)과
`at t+`(그때가 몇 초 뒤인지)이 달라진 것이 보인다.

```text
COMPUTED PROXIMITY (constant-velocity extrapolation, vehicle footprints)
  Using each road user's actual rectangular footprint, not centre-to-centre
  distance. 0.00 m means the two bodies are touching or overlapping.

  'gap now'  = separation at this instant.
  'min gap'  = closest they get within the horizon if both hold course.

  A pair already at 0.00 m now is simply close together -- stopped in a
  queue or side by side -- and is not by itself evidence of a collision.
  A pair with a wide gap now that closes to 0.00 m is converging.

  pair          gap now (m)  min gap (m)  at t+ (s)
  EGO_ego_vehicle_behind & EGO_other_vehicle         7.17         0.66        0.9
  EGO_ego_vehicle & V001         4.22         2.06        0.8
  EGO_ego_vehicle & V002        13.65         3.28        1.0
  V002 & M003          7.01         4.84        1.0
  V007 & M008          8.74         8.74        0.0
  V006 & V007          8.79         8.79        0.0
  EGO_other_vehicle & EGO_other_vehicle_behind        11.33         9.43        0.9
  N004 & V010          9.68         9.61        1.0
  ... 142 further pairs, all with larger gaps

Read the two gap columns together:
  wide now, 0.00 m later  -> converging. This is the dangerous one.
  0.00 m now and later    -> already side by side or queued, usually parked cars.
  narrow now, wider later -> diverging.
```

---

## + iitp 장면 설명 전체 · 0.631

E 의 표는 그대로 두고 아래 블록을 이어 붙인다. 어느 도로 몇 차선, 교차로까지 몇 m,
제한속도, 가속도, 예상 경로, 그리고 차량 쌍마다 충돌까지 남은 시간(TTC)·차간
시간(headway) 같은 것이 들어 있다.

실제 길이는 **15,744자**이고 여기서는 앞부분만 싣는다.

```text
MAP-GROUNDED SCENE, FUSED ACROSS ALL PLATFORMS
The same scene as the table above, matched to the road network. Lane numbers,
junction distances, speeds, accelerations and the interaction list are computed,
not anyone's opinion. Ids here are global (EGO_* are the sensor platforms
themselves); they are NOT the numeric cluster ids used in the table above, so
link the two by position.

# Observation window 3-4 (t=3-4s, 3 snapshots)
Context: source DeepAccident, map Town01, weather=ClearNoon, road_type=three-way junction
At the last timestep: 4 observer vehicle(s), 15 other road user(s), 1 roadside sensor(s)

## Roadside sensors (V2X, not road users)
- infrastructure: mounted 3.3m, 11 detection(s) at this time

## Perception relations (who perceived whom)
- ego_vehicle (vehicle): perceives 11 | sole observer of V007 | own actor id EGO_ego_vehicle | 6 cameras, 360 deg coverage
- ego_vehicle_behind (vehicle): perceives 8 | own actor id EGO_ego_vehicle_behind | 6 cameras, 360 deg coverage
- other_vehicle (vehicle): perceives 12 | sole observer of V011, V016 | own actor id EGO_other_vehicle | 6 cameras, 360 deg coverage
- other_vehicle_behind (vehicle): perceives 10 | sole observer of M003, V006 | own actor id EGO_other_vehicle_behind | 6 cameras, 360 deg coverage
- infrastructure (infrastructure): perceives 11 | sole observer of V012, V013 | 6 cameras, 360 deg coverage

## Per-vehicle time course (3-4s, every 1s)
### EGO_ego_vehicle_behind (observer vehicle) — seen by: ego_vehicle_behind,ego_vehicle,other_vehicle,other_vehicle_behind,infrastructure
   t=    3  (  140.1,  -59.5)    31km/h  Road 9 eastbound lane 1/1 junction 5m  keeping lane
   t=    4  (  149.9,  -59.5)    34km/h  Road 354 eastbound lane 1/1 junction 17m  keeping lane
### EGO_other_vehicle (observer vehicle) — seen by: other_vehicle,ego_vehicle,ego_vehicle_behind,other_vehicle_behind,infrastructure
   t=    3  (  154.0,  -40.9)    29km/h  Road 25 southbound lane 1/1 junction 5m  keeping lane
   t=    4  (  154.6,  -49.2)    29km/h  Road 339 southbound lane 1/1 junction 16m  keeping lane
### EGO_ego_vehicle (observer vehicle) — seen by: ego_vehicle,ego_vehicle_behind,other_vehicle,other_vehicle_behind,infrastructure
   t=    3  (  159.1,  -59.5)    38km/h  Road 354 eastbound lane 1/1 junction 8m  keeping lane
   t=    4  (  169.4,  -59.5)    38km/h  Road 10 eastbound lane 1/1 junction 156m  keeping lane
### EGO_other_vehicle_behind (observer vehicle) — seen by: other_vehicle_behind,ego_vehicle,ego_vehicle_behind,other_vehicle,infrastructure
   t=    3  (  154.1,  -25.6)    28km/h  Road 25 southbound lane 1/1 junction 21m  keeping lane
   t=    4  (  154.0,  -33.1)    36km/h  Road 25 southbound lane 1/1 junction 13m  accelerating
### M008 (motorcycle) — seen by: ego_vehicle,other_vehicle_behind

  … (이하 12,944자 줄임)
```

## + iitp 설명에서 판정 라벨만 뺌 · 0.660

바로 위 조건과 **딱 한 가지만 다르다.** `keeping lane`(차선 유지) 처럼 **판정 성격의 말** 10종
7,057개를 지웠고, 숫자는 97.8% 남겼다.

차량 한 줄이 이렇게 바뀐다.

```text
G: - M008(motorcycle): Road 10 eastbound lane 1 of 1 | speed limit 40km/h | 108m to next junction | 8km/h | heading 90 deg | keeping lane | predicted path: off-candidate path | seen by ego
H: - M008(motorcycle): Road 10 eastbound lane 1 of 1 | speed limit 40km/h | 108m to next junction | 8km/h | heading 90 deg | seen by ego_vehicle, other_vehicle_behind (existence 1.00, posi
```

상호작용 한 줄은 이렇게 바뀐다.

```text
G: - EGO_ego_vehicle_behind → EGO_other_vehicle | junction J20, arrival time gap 0.1s | orthogonal entry conflict expected | TTC 1.8s
H: - EGO_ego_vehicle_behind → EGO_other_vehicle | junction J20, arrival time gap 0.1s | TTC 1.8s
```

지운 것과 개수:

```
2,753  predicted path (예상 경로)      1,495  keeping lane (차선 유지)
1,099  stopped (정지)                    914  NOTE (불확실하다는 훈수)
  518  car following (선행차 추종)        119  orthogonal entry conflict expected
  111  accelerating (가속 중)              33  decelerating (감속 중)
    9  turning left (좌회전)                6  turning right (우회전)
```

`own actor id EGO_*`(416개)는 **남겼다.** 숫자가 없어 라벨처럼 보이지만, 관측자 이름
(`ego_vehicle`)과 차량 번호(`EGO_ego_vehicle`)를 잇는 유일한 단서다. 지우면 모델이
둘을 같은 차로 볼 수 없다.

---

## 이미지가 붙는 조건

이미지는 글이 아니라 별도 첨부로 간다. 함께 보내는 안내문이 조건에 따라 다르다.

**원문 — 관측자 리포트가 있는 것을 전제로 쓰였다**

```text
You are also given one BEV image: the LiDAR returns of every platform fused into
a single top-down view, in the same format the platforms described. Colour is
time -- blue 1.0 s ago, amber 0.5 s ago, red now -- so a moving road user is a
blue-amber-red streak. Use it to check the reports. Where the image and a report
disagree, say which you trust and why.
```

**바꾼 것 — 리포트 없이 표와 이미지만 줄 때**

원문은 "리포트를 확인하는 데 쓰라"고 하는데 이 조건에는 리포트가 없다. 없는 것을
가리키게 되므로 문구를 바꿨다.

```text
You are also given one BEV image: the LiDAR returns of every platform fused into
a single top-down view, in the same format the platforms described. Colour is
time -- blue 1.0 s ago, amber 0.5 s ago, red now -- so a moving road user is a
blue-amber-red streak. Use it to check the reports. Where the image and a report
disagree, say which you trust and why.
```

---

## 답 형식

아홉 조건이 전부 같은 JSON 하나를 요구한다. `gemini-2.5-flash` 로 **927회 호출에
형식 오류 0건**이었다.

```json
{
  "collision": true,
  "confidence": 0.0,
  "relative_direction": "same | opposite | null",
  "impact_zone": "front | front_side | rear | rear_side | null",
  "involved": ["car", "van"],
  "description": "1-3 문장",
  "avoidance": "막을 수 있었던 기동 하나, 없으면 null",
  "used_platforms": [],
  "disagreement_with_observers": []
}
```

## 이 문서를 다시 뽑기

```bash
cd /home/sryu/inclab-nas/sryu/iitp
~/vlm/.venv-vlm/bin/python reports/_gen/dump_prompts.py > /tmp/prompts.json
.venv/bin/python reports/_gen/render_prompts.py /tmp/prompts.json
```

앞 단계는 **vlm 의 파이썬 환경**으로, 뒤 단계는 iitp 것으로 돈다 (필요한 라이브러리가
서로 다른 환경에 있다). 두 환경은 JSON 파일로만 주고받는다.
