# FP 책임 분리: WaypointNet / V2 reranker / Gemini

2026-09-14. 대상은 Gemini 3 Flash fixed-5s 실험의 **scorable FP 98개 bucket**이다.
LLM을 다시 호출하지 않고, 각 FP 시나리오를 동일한 sensor3d fusion, OpenDRIVE map,
WaypointNet, 5초 window 설정으로 재생성했다. 저장 당시 V2 top-12와 재생성 top-12는
**98/98 완전히 동일**했다. 따라서 아래 V2 비교는 저장된 응답에서 추정한 것이 아니다.

## 책임을 판단하는 규칙

| 컴포넌트 | 직접 확인할 수 있는 것 | 이번 분석에서 책임으로 세지지 않는 것 |
|---|---|---|
| WaypointNet / fallback | 입력 종료 뒤의 제공 경로와 후속 관측의 정지·위치 불일치 | 후속 행동이 입력만으로 반드시 예측 가능했다는 주장 |
| V2 reranker | 동일 raw candidate 50개에서 raw top-12 밖의 contact를 V2가 top-12로 올렸는지 | 선택된 pair가 위험해 보인다는 사실 자체 |
| Gemini | V2 pair가 없는데 actor path에서 사고를 지목했는지, 선정된 접촉의 다른 bucket까지 사고로 확장했는지 | 경로가 거짓임을 모델이 당연히 알아야 했다는 주장 |

각 FP의 지목 actor들 안에서 `predicted_contact=true`인 pair를 찾아 raw clearance 순위와
V2 score 순위를 비교했다. actor future path 자체는 compact payload에 항상 있으므로,
V2 pair 표에서 빠졌다고 Gemini가 그 pair를 전혀 볼 수 없는 것은 아니다.

## V2 reranker: 이번 FP의 주 원인이 아님

| 결과 | FP bucket |
|---|---:|
| V2가 같은 bucket contact를 수록, 그러나 raw top-12에도 이미 존재 | 79 |
| V2가 다른 bucket contact를 수록, 그러나 raw top-12에도 이미 존재 | 17 |
| V2는 수록하지 않았지만 raw top-12 contact/actor path에서 Gemini가 사고를 지목 | 2 |
| **V2가 raw top-12 밖 contact를 새로 수록** | **0** |

즉 98/98에서 Gemini가 지목한 contact 후보는 **V2가 없어도 raw compact payload의 top-12에
노출됐을 후보**였다. V2가 후보를 새로 만들어낸 것도 아니고, 이 FP들의 contact 노출에
필요조건도 아니었다.

V2는 순서를 바꾸기는 했다. V2 top-12에 남은 지목 contact 114개(한 FP가 여러 contact를
가질 수 있음) 중 raw 대비 순위 상승 17개, 동일 47개, 하락 50개이며 중앙 변화는 0이다.
따라서 “V2가 위험 contact를 일반적으로 위로 끌어올려 FP를 만들었다”는 증거도 현재 없다.
payload list 순서의 salience 효과는 별도의 동일-내용 순서 A/B 없이는 판정할 수 없다.

이 결론은 `actor cap` 이전의 누락, raw candidate pool 50개 밖의 쌍, 또는 V2의 전체 recall/
precision 품질까지 무죄라는 뜻은 아니다. **현재 발생한 98 FP의 지목 contact 노출**에 한정된 결론이다.

## WaypointNet / fallback: 직접적인 upstream 오류 신호

후속 관측과의 대조에서는 다음이 나왔다.

| 경로-관측 불일치 screening | FP bucket | 시나리오 |
|---|---:|---:|
| 예측은 ≥10km/h 이동, 후속 관측속도는 ≤1km/h | 46 | 24 |
| 위를 EGO로 한정 | 31 | 17 |
| EGO의 같은 1초 위치 변화도 ≤2km/h여서 정지를 교차 확인 | 18 | 9 |
| 위 EGO가 입력 종료 시점에도 accel ≤−0.5m/s² | 14 | 7 |
| 지목 actor 중 하나의 미래 위치 오차 ≥5m | 61 | 34 |

이 수치는 원인 비율이 아니라 확실한 재검토 후보의 크기다. 예를 들어 FP051은 입력 속도
26.7→15.8→10.5km/h, 현재 accel −0.57m/s²인데 제공 경로는 계속 약 10km/h로 움직였고,
후속 관측은 정지했다. 여기서는 upstream 경로가 충돌 geometry를 만든 것이 직접 보인다.

단, 6개 FP에는 `no road match` 계열 경로가 있어 지도 후보 부재의 rule fallback일 수 있다.
따라서 “WaypointNet” 책임에는 신경망 경로와 fallback 경로를 다음 audit에서 반드시 분리해야 한다.

## Gemini: candidate를 사고로 확정하거나 bucket을 확장한 책임

- 79개는 V2가 수록한 **동일 bucket contact**를 Gemini가 사고라고 채택한 경우다.
  이 경우 Gemini는 contact를 만들지는 않았지만, independent future path의 overlap을 실제
  사고로 확정했다.
- 17개는 수록 contact의 대표 bucket과 Gemini 사고 bucket이 다르다. 이 중 15개가 normal
  시나리오이고 15개에는 미래 위치 오차 ≥5m가 있다. 일부는 첫 사고를 이후 bucket의
  “continued physical contact/loss of control”로 확대한다. 이는 Waypoint path 오류와 별도로
  Gemini의 시간 bucket 해석/지속 사고 추론을 점검해야 함을 뜻한다.
- 2개(FP062, FP063)는 V2 top-12에 해당 contact가 없었다. 그럼에도 actor future path와
  정지차량 상태로 rear-end 사고를 지목했다. 이 둘은 **V2 pair summary가 없어도 발생한
  Gemini verdict**다. 다만 raw top-12에는 해당 contact가 있었으며, compact가 actor paths를
  제공하므로 완전한 무근거 hallucination이라고 부르지 않는다.

Gemini가 각각의 경로 오류를 기각할 수 있었는지는 counterfactual prompt/payload 재호출 없이는
측정할 수 없다. 그러나 FP051처럼 입력 감속 증거가 있고 미래 path와도 충돌하는 사례는,
경로를 확정 사실로 채택한 판단을 완화할 여지가 있다.

## 현재의 책임표

| 우선순위 | 책임 구간 | 근거 | 권장 검증 |
|---:|---|---|---|
| 1 | future-path generator (WaypointNet 또는 fallback) | 정지/속도/위치 불일치가 반복되고 contact geometry의 원천 | neural vs fallback provenance를 저장하고, 고해상도 GT로 braking/turn miss 분류 |
| 2 | Gemini verdict / bucket semantics | 동일 contact를 실제 사고로 확정(79), 다른 bucket으로 확대(17), pair summary 없이도 경보(2) | 같은 actor paths에서 contact 요약 제거·감속 evidence 추가·사고 지속 금지 A/B |
| 3 | V2 reranker | 이번 FP에서 새 contact 노출 0; 순위 효과만 가능 | raw/V2 동일 pair list 순서 A/B, 별도 recall/precision 평가 |

**결론:** 현재 98개 FP에 대해 V2 reranker는 1차 원인으로 보이지 않는다. 우선
`경로 생성 → Gemini의 경로 채택/시간 해석` 경계를 개선·실험하고, V2는 그 뒤
ranking quality와 salience를 별도로 검증하는 순서가 타당하다.

재현:

```bash
.venv/bin/python examples/diagnose_fp_attribution.py \
  --root /home/sryu/inclab-nas/DeepAccident --carla-maps carla_map \
  --out out/analysis/fp_attribution_replay_20260914.json
```

원시 replay 결과는 [JSON](../out/analysis/fp_attribution_replay_20260914.json),
FP의 미래 관측 대조는 [이전 사례 분석](fp_case_analysis_20260914.md)에 있다.
