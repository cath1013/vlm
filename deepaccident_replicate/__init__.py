"""DeepAccident 사고 예측의 재구현 — `traffic_llm` 과 비교하기 위한 대상.

공개 저장소는 이 머신에서 돌지 않는다. 두 가지가 각각 독립적으로 막는다.

    가중치 없음   `README.md` 의 Model Zoo 절이 통째로 주석 처리돼 있고
                 (`[//]: # (## Model Zoo)`), 살아 있는 링크 둘은 지표가
                 mAP/NDS/Map IoU 인 **nuScenes 용 BEVDet** 가중치다.
                 DeepAccident 로 학습한 결과는 배포된 적이 없다.
    빌드 불가     원본은 Python 3.7 / PyTorch 1.10.2 / **CUDA 10.2** /
                 mmcv-full 1.3.14 를 요구한다. CUDA 10.2 가 지원하는 최대
                 아키텍처는 sm_75 인데 이 머신의 GPU 는 **sm_120**(Blackwell) 이다.

그래서 **방법을 다시 구현했다.** 이 패키지는 `traffic_llm` 에 의존하지만 그 반대는
아니다 — 비교 대상이 본체 파이프라인을 오염시키지 않게 분리해 둔다.

    da_baseline.py     사고 **판정 규칙** 이식 (거리 임계 하나). 학습 불필요
    da_motion.py       판정을 먹이는 **운동 예측 모델** (BEV 흐름 + 사고 헤드)
    train_da_motion.py 프레임 캐시 + 학습
    eval_da_motion.py  평가 — `traffic_llm.accident_qa.score_modes` 를 통과하므로
                       LLM 응답과 같은 표에 오른다

자세한 것은 `README.md`.
"""
