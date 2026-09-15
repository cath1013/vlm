# Compact payload / pair reranker / FP 감사

분석일: 2026-09-10. API 재호출 없이 저장된 payload·응답·GT와 현재 구현을 비교했다.
현재 실험은 `out/experiments/waypointnet_compact_v2_fixed_val104_gemini3flash_20260909`
(Gemini 3 Flash preview, 영어, fixed 5초 관측, 5초 예측, 324개 응답)이다.

## 1. 결론과 확실성

- **V2 적용 때문에 actor 관측 정보가 payload에서 삭제됐다는 가설은 확인되지 않았다.**
  이전 compact와 V2 compact의 공통 732개 window에서 `actors`와 system prompt가 모두 동일하다.
  달라진 최상위 필드는 `predicted_closest_pairs`, `predicted_pair_method` 두 개뿐이다.
- **현재 pair reranker가 상호 관측을 활용하지 않는다는 가설은 맞다.** V1도 사용하지 않았다.
  V2는 가속도를 절댓값으로 변환하므로, 다른 입력을 고정하면 가속과 감속의 부호를 구별하지 못한다.
- **LLM에 현재 감속·관측 정보는 도달한다.** 다만 명시적인 관측자 매핑/설명, 역사적 도로·기동,
  접근 관계 요약 등은 standard보다 빈약하다. 현재 관측 목록이 있다고 반응 시점·감속 의도가 주어진 것은 아니다.
- **실제 FP에서 미래 정지를 놓친 경로와 그 경로를 인용한 Gemini 응답을 확인했다** (§5).
  전체 FP를 같은 원인으로 단정하거나, 이 결과만으로 Gemini의 편향을 배제할 수는 없다.

## 2. Compact가 무엇을 생략하는가

비교 대상은 실제 저장된 standard 자연어 입력(`waypointnet_audit20`), 이전 compact
(`waypointnet_compact_val104`), V2 compact(`waypointnet_compact_reranker_v2_val104`)이다.
standard의 선택적 JSON 블록과 실제 기본 자연어 입력을 혼동하지 않았다.

| 정보 | 이전 standard 자연어 | 현재 compact | 의미 |
|---|---|---|---|
| 현재 `observed_by` | actor별 관측자 이름 | `current_state`에 보존 | 현재 관측 관계 자체가 삭제된 것은 아님 |
| 관측자 설명 | 관측 수, sole-observer 대상, observer→본체 actor 매핑, 카메라 수/범위 | 별도 블록 없음 | `ego_vehicle`와 `EGO_ego_vehicle` 연결·해석을 모델에 맡김 |
| 시점별 관측자 목록 | 기본 history에는 없음; 선택적 `window_json`에서 지원 | 없음 | 기본 standard에 항상 있던 정보를 없앴다고 말하면 부정확 |
| history 위치·속도 | 있음 | 1초 간격 `[t,e,n,speed]` | 감속 추세를 계산할 기초 데이터는 남음 |
| history 도로·방향·차선·교차로 거리·기동 | `_actor_row`에 있음 | 없음 | 과거 차로/기동 변화를 구조적으로 읽기 어려워짐 |
| 현재 signed acceleration | 절댓값 0.5 초과 시 자연어 표시 | `accel_mps2` 보존, 결측은 null | LLM 단계에서 감속 부호가 사라진 것은 아님 |
| 현재 도로·방향·차선·횡오프셋 | 있음 | 보존 | 기본 공간 관계는 유지 |
| 다음 교차로 ID/거리 | 상세 도로 설명에 거리 등 | 별도 필드 없음 | 도로 상황 해석 정보 축소 |
| 관측 중 pair 접근 변화 | 시작/끝 거리, closing rate, 선형 접촉시간, 같은 도로 표시 | `closing_pairs` 없음 | 미래 경로와 실제 접근 추세를 비교하는 요약이 사라짐 |
| 마지막 시점 interactions | TTC/headway 등 파생 관계 요약 | 없음 | 위험 문구 과대해석을 막으려 제거했으나 관계 설명도 감소 |
| ASCII BEV | 설정에 따라 표시(기본 활성) | 없음 | 공간 표현 축소 |
| 맥락 | source/town/weather/road_type 등 가용 문맥 | 제한적인 map conventions | 신호 상태 등 원래 없던 데이터와 구분 필요 |
| 미래 경로 | 경로 설명/수치 | 수치 waypoint, maneuver, probability, 목적 도로 보존 | 예측 모델의 출력은 계속 강하게 노출 |
| 미래 pair 접촉 | standard의 접근/상호작용 설명 | swept footprint 기반 `predicted_contact`, clearance/time | 과거 사실이 아니라 예측 경로를 가공한 신호 |
| 사라진 actor, 현재 신뢰도/거리 | 있음 | 보존 | compact가 모든 품질 정보를 삭제한 것은 아님 |

`include_perception`, `trajectory_observation` 설정만 보고 compact에 해당 블록이 있다고
판단하면 안 된다. compact는 별도 renderer이며 history는 네 칼럼만 직렬화한다.
또한 standard의 perception 자연어 블록은 **모든 observed_now ID를 나열하는 JSON과 다르다**.

근거: `traffic_llm/accident_qa.py`의 `_actor_row`, `render_window_text`,
`compact_window_json`, `perception_summary`, `window_json` 및 `traffic_llm/serialize.py`.

## 3. 이전 ranker와 V2의 차이

여기서 세 종류의 ranking을 구분해야 한다.

| 단계/모델 | 입력·정렬 | 관측 관계·감속 반영 |
|---|---|---|
| 이전 compact의 pair 선정 | 미래 경로 footprint 최소 clearance 순 | 관측 관계 없음; 경로에 반영된 움직임만 간접 사용 |
| V1 pair reranker | 43차원 logistic: clearance/log, contact, 시간, ego 여부, pair type | 관측 관계·현재 속도·가속도 없음 |
| V2 pair reranker | 94차원 MLP: V1 계열+현재/미래 clearance, 접촉 지속, 상대속도/접근속도, 도로/차선, actor/path 통계 | 관측 관계 없음; 현재 가속도 **절댓값**; path 속도 변화 부호는 있음 |
| motion RankNet / WaypointNet | 개별 actor history·상태와 지도 후보로 경로 평가/예측 | signed acceleration 사용; 다른 actor 상태/관측 그래프를 명시적으로 받지 않음 |

실제 `waypointnet_compact_val104` payload에는 V2 선정 표시가 없으며,
그 이전 실험을 V1 학습 ranker 적용본이라고 부를 근거는 없다. V1 구현과 이전 운영 방식은 별개다.

현재 처리 순서:

`actor risk 기반 cap → 독립 미래 경로의 swept geometry → clearance 순 후보 최대 50개 → V2 score 순 최대 12개 → LLM`

- V2는 반올림 전 `ActorState`를 받아 feature를 계산한다. compact JSON에서 필드를 생략해서
  V2가 못 읽는 것이 아니라, **feature 함수가 관측 관계를 사용하지 않는 것**이다.
- 후보 50개 제한 이전에 빠진 pair를 V2가 복구할 수는 없다.
- actor cap은 pair reranker와 다른 단계이며, 같은 actor들에 대한 상위 pair 목록을 바꾼다.
- 체크포인트 threshold `0.927234`는 payload 메모에만 표시된다. 직렬화는 threshold 필터가 아니라
  score 순 top-12다. 따라서 **선정됐다는 사실은 threshold를 넘었다는 뜻이 아니다**.
- 수치 score와 탈락한 50개 후보 전체가 현재 payload에 없다. “그 상호 관측 pair의 실제 score가
  얼마였는지”는 저장 응답만으로 복원할 수 없다. 원본 상태 재생성이 필요하다.
- `observed_by`만 제거한 합성 입력, 가속도 부호만 뒤집은 합성 입력 모두 V2 feature/score가
  정확히 동일함을 실행 확인했다(다른 입력·미래 경로 고정). 이는 감속 추세를 전혀 못 본다는
  뜻은 아니다. 상대속도와 미래 path 속도 변화는 여전히 사용한다.
- symmetric min/max/absdiff actor 통계에는 명시적인 앞차/뒤차 역할별 감속, 관측 방향, 반응 지연이 없다.

근거: `traffic_llm/pair_reranker.py`, `pair_reranker_v2.py`, `predict_model.py`,
`accident_qa.py:compact_window_json`. V1→V2의 신경망 구조 변경 자체를 FP 원인으로 입증한 것은 아니다.

## 4. 관측 관계와 FP를 정확히 다시 세기

`B.observed_by`에 observer `a`가 있으면 관측차량 `EGO_a`가 B를 관측했다는 뜻이다.
EGO 본인의 observer ID는 자기 telemetry 등록으로도 들어가므로 자기 관측을 세면 안 된다.
두 EGO의 cross-membership을 확인해야 상호 관측이라고 부를 수 있다.
제3자가 둘을 봤다는 **공통 observer는 상호 관측과 다르다**.
일반 `V...` actor는 관측 장치가 보고하는 observer가 아니므로 그 차가 무엇을 봤는지는 unknown이다.

현재 응답 전체를 재집계한 bucket 결과:

| 항목 | FP | TP |
|---|---:|---:|
| scorable positive bucket | 98 | 45 |
| involved IDs에 수록된 contact pair를 포함 | 96 | 44 |
| 그 contact pair의 대표 interval도 일치 | 79 | 43 |
| involved IDs에 상호 관측 EGO pair를 포함 | 51 | 32 |
| 정확히 두 actor를 지목했고 그 둘이 상호 관측 | 36 | 29 |

전체 confusion은 TP45/FP98/TN457/FN24다. 위 contact 일치는 인과 검정이 아니라
공급된 경로 후보와 응답의 정합성 검사다. 대표 interval 불일치는 접촉 지속 등도 가능하므로
곧바로 hallucination으로 분류할 수 없다. 세 대 이상을 지목한 응답은 어떤 두 대 사이의
충돌을 뜻했는지 모호하여 subset 집계와 정확한 두 actor 집계를 분리했다.

normal 폴더의 positive bucket은 364개이지만, 기록 밖 미래를 포함하므로 **364개 모두 확정 FP가 아니다**.
그중 contact subset 357개, 상호 관측 subset 160개, 정확한 두 actor 상호 관측 98개다.
FP98도 독립 사건 98개가 아니며 accident 시나리오의 잘못된 시점 경보를 포함한다.

관측 가능/센서 검출은 운전자가 인지하고 적시에 제동한다는 보증이 아니다. 실제 TP에도 상호 관측이
있으므로 `mutual_observed ⇒ accident=false` 규칙은 부적절하다. 반대로 제동 여유·이미 나타난 감속·
상대 경로 변화가 있는데도 후보 접촉을 확정 사고로 읽는지 확인하는 것은 타당한 진단이다.

## 5. 실제 사례: 상호 관측·현재 감속이 있는데 미래 정지를 놓침

normal `Town03_type001_subtype0001_scenario00009`, 관측 `1–6`, 예측 `(7,8]`.
저장 GT는 `accident_expected=false, scorable=true`, `data_end_s=9.5`다.

| t=6 입력 | EGO_ego_vehicle | EGO_other_vehicle |
|---|---:|---:|
| 속도 (current_state 반올림값) | 18 km/h | 28 km/h |
| 가속도 | −1.43 m/s² | −0.83 m/s² |
| 상대 관측자 포함 | other_vehicle | ego_vehicle |
| 제공 미래 경로 | 남쪽으로 휘는 off-candidate path | 거의 등속 동진 |

선정된 pair는 clearance=0, 접촉 시점 +1.9초, `joint_path_probability=1.0`이다.
Gemini는 두 차량을 정확히 지목하고 7.9초 head-on/side-swipe를 high confidence로 예측했다.

같은 시나리오의 나중 `4–9` window history에서 미래 실제 관측을 확인하면:

| EGO_other_vehicle | t=7 위치 / 속도 | t=8 위치 / 속도 |
|---|---|---|
| t=6에 제공된 경로 | (−5.8, −134.3), 계속 동진 | (1.9, −134.3), 계속 동진 |
| 나중 관측 history | (−7.9, −134.3), 0.1 km/h | (−7.5, −134.3), 0.0 km/h |

이 사례에서 **정지를 놓친 upstream 경로 예측과, 그 경로의 충돌을 채택한 downstream 응답**을
확인할 수 있다. 후속 관측은 진단용이며 t=6의 LLM 입력에 주어진 정보가 아니다.
정지한 이유가 반드시 상호 관측 때문인지는 이 데이터로 확정할 수 없다.

또한 WaypointNet `_from_waypoints()`는 단일 경로에 `probability=1.0`을 넣는다.
이 숫자는 보정된 사고 확률도, 미래 행동이 100% 확실하다는 의미도 아니다.
compact prompt는 독립 경로의 contact를 자동 복사하지 말라고 이미 경고하지만,
동시에 physical contact geometry를 true 판정의 핵심 조건으로 둔다. 이 구성과 1.0 표기가
경로에 대한 과신을 유발하는지는 동일 입력 ablation으로 검증해야 한다.

원본:

- [t=6 실제 payload](../out/experiments/waypointnet_compact_v2_fixed_val104_gemini3flash_20260909/val/normal/Town03_type001_subtype0001_scenario00009/llm_payload_1-6.json)
- [Gemini 응답](../out/experiments/waypointnet_compact_v2_fixed_val104_gemini3flash_20260909/val/normal/Town03_type001_subtype0001_scenario00009/responses/response_1-6.json)
- [해당 GT](../out/experiments/waypointnet_compact_v2_fixed_val104_gemini3flash_20260909/val/normal/Town03_type001_subtype0001_scenario00009/ground_truth_1-6.json)
- [후속 실제 history를 포함한 payload](../out/experiments/waypointnet_compact_v2_fixed_val104_gemini3flash_20260909/val/normal/Town03_type001_subtype0001_scenario00009/llm_payload_4-9.json)

## 6. 우선순위: 원인 분리를 위한 다음 실험

1. 원본 상태에서 동일 window를 재생성해 후보 50개·V1/V2 score·rank·threshold 통과 여부·
   제외 이유를 진단 sidecar에 저장한다. 보지 못한 observer는 unknown으로 표현한다.
2. 동일 모델·window·GT를 고정하고 payload만 비교한다: 현재 입력 / 관측자 매핑·방향별
   가시성 history·실제 감속 추세 추가 / contact 요약 및 단일 경로의 1.0 표기 제거.
   상호 관측이면 사고 없음이라는 정답 규칙을 프롬프트에 넣지 않는다.
3. pair 선택을 고정한 정보 ablation과, payload 정보를 고정한 ranker ablation을 분리한다.
   V2에서 signed acceleration/역할별 제동·관측 feature를 추가하는 것은 재학습·검증 대상이다.
4. 해당 FP 사례들의 예측 경로와 후속 관측을 비교해 missed braking, wrong turn,
   timing-only error, 위치 오차 등을 분리한다. 독립 궤적 예측기의 이웃 차량 조건화도 검토한다.
5. scorable bucket FP와 censored alert, 사건 단위 중복을 분리해 보고한다. 예전 warmup/Gemini2.5와
   현재 fixed/Gemini3의 전체 FP 수를 바로 비교해 모델 편향의 증거로 쓰지 않는다.

현재 수행한 것은 진단과 문서 정정이며, pipeline·prompt·체크포인트 변경 또는 유료 재호출은 없다.
재현: `.venv/bin/python examples/audit_payload_fp.py` (통계, 정확한 상호 관측 FP 원본 참조,
이전 compact↔V2 전체 window 비교를 stdout JSON으로 출력).
