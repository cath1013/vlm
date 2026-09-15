# GPT-5.5 대 Gemini 3 Flash: FP 원인 분리 pilot

2026-09-14. 저장된 Gemini 3 Flash 실험의 장면 JSON, 시스템 프롬프트, 질문,
출력 스키마, GT는 바꾸지 않고 OpenAI Chat Completions wrapper와 `gpt-5.5`만
교체했다. 따라서 WaypointNet/fallback path, swept geometry, V2 reranker,
payload 내용은 고정이다.

## 표본

6개 window를 호출했다.

- 감속 후 정지를 놓친 normal FP 4개
- Gemini가 한 접촉을 여러 bucket으로 확장한 normal FP 1개
- 실제 사고 시점보다 이르게 경보한 accident 사례 1개

이는 전체 성능 추정 표본이 아니라, 기존 FP 원인 가설을 구분하기 위한 표적 pilot이다.

| window | GT | Gemini | GPT-5.5 | Gemini FP/TP | GPT-5.5 FP/TP |
|---|---|---|---|---:|---:|
| normal Town04 subtype1 scenario19, 0-5 | `.....` | `TTT..` | `TTT..` | 3/0 | 3/0 |
| normal Town04 subtype2 scenario24, 0-5 | `.....` | `..T.T` | `..T..` | 2/0 | 1/0 |
| normal Town05 subtype2 scenario33, 0-5 | `.....` | `..TTT` | `..TTT` | 3/0 | 3/0 |
| normal Town10HD subtype1 scenario34, 0-5 | `.....` | `...TT` | `...T.` | 2/0 | 1/0 |
| normal Town03 subtype2 scenario28, 0-5 | `.....` | `TTTTT` | `TT...` | 5/0 | 2/0 |
| accident Town04 subtype1 scenario38, 0-5 | `..T..` | `.T...` | `.TTTT` | 1/0 | 3/1 |

`T`는 해당 1초 bucket에서 accident_expected=true다.

## 관찰

1. **FP051 계열의 핵심 판단은 모델을 바꿔도 남았다.**
   Town04 scenario19에서 두 모델은 모두 `EGO_other_vehicle`의 감속·정지
   evidence보다 독립 경로의 0.5/1.7/2.8초 footprint overlap을 우선해 세 bucket을
   사고로 판정했다. GPT-5.5의 reason도 동일하게 "body overlap"을 근거로 든다.
2. **Gemini의 bucket 확장은 일부 줄었다.**
   Town03 scenario28은 Gemini가 5개 bucket 전부를 경보했지만 GPT-5.5는 2개만
   경보했다. 둘 다 첫 접촉은 사고로 확정했으므로, 경로/contact의 upstream 문제는
   그대로이고 지속 해석만 완화됐다.
3. **GPT-5.5는 사고 사례에서 더 많은 후속 bucket을 경보했다.**
   Town04 scenario38에서 실제 positive bucket(k=3)은 잡았지만 k=2,4,5도 true로
   하여 strict FP가 1개에서 3개로 늘었다. 모델 교체가 bucket-level FP를 일관되게
   낮춘다는 근거는 아니다.

정상 5개만 합치면 FP bucket은 Gemini 15개, GPT-5.5 10개다. 그러나 선택된 표본이므로
전체 FP rate 개선으로 일반화할 수 없다. 감속-정지 4개만 합치면 10개에서 8개로,
핵심 overlap 확정 문제가 대부분 유지된다.

## 결론과 다음 조치

이 결과는 Gemini만의 고유 bias가 아니라, **독립적으로 생성된 경로가 overlap을 보이면
사고로 확정하도록 유도하는 payload/판정 구조**가 FP의 공통 원인임을 뒷받침한다.
따라서 전체 GPT-5.5 재호출 전에 다음 A/B가 우선이다.

1. 동일 모델에서 `predicted_contact`를 후보로만 취급하게 하고, 감속/정지/상호관측 및
   path uncertainty가 있으면 overlap만으로 true를 내지 못하게 하는 prompt/schema A/B.
2. WaypointNet/fallback 쪽에는 상호작용 정보와 braking/stop 행동을 반영하거나, 최소한
   경로 확률/stop 대안을 제공해 단일 deterministic path를 사실처럼 제시하지 않기.
3. 위 조건이 정해진 뒤 full 324-window GPT-5.5 A/B를 실행해 전체 성능을 비교하기.

생성·호출 재현:

```bash
.venv/bin/python examples/repackage_gemini_payloads.py ...
source .env
.venv/bin/python examples/ask_llm.py <payload> --provider openai --score --modes all
```

원본과 GPT-5.5 응답은 각각
`out/experiments/waypointnet_compact_v2_fixed_val104_gemini3flash_20260909`와
`out/experiments/gpt55_fp_ab_pilot_20260914`에 있다.
