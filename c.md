sigLLM 논문에서 말하는 “SOTA보다 F1 약 30% 열세”의 SOTA는, **여러 전통·딥러닝 TSAD 모델들을 통칭한 평균적 기준**이다.[1][2]

- 비교 대상에 포함된 모델들  
  - 통계·고전 기법: ARIMA, Matrix Profile(MP), Moving Average(MAvg) 등.[1]
  - 딥러닝/하이브리드: LSTM‑DT(예측 기반), LSTM‑AE, VAE, TadGAN(재구성 기반), AER(예측+재구성 하이브리드).[2][1]
  - Transformer·서비스: Anomaly Transformer(AT), Microsoft Azure Anomaly Detector.[1]

- 평가 방식  
  - 11개 시계열 이상탐지 데이터셋(SMAP, MSL, Yahoo S5 등)에 대해, LLM 기반 sigLLM 파이프라인과 위 기법들을 모두 돌리고, **윈도우·세그먼트 단위 F1 점수(부분 탐지도 인정하는 TSAD용 F1)**를 사용.[2][1]
  - sigLLM의 best 파이프라인(Detector)이 평균 F1 ≈ 0.525 정도를 내는 반면, 위 SOTA 모델들의 평균은 이보다 약 30% 높은 수준이라고 정리한다.[3][1]

정리하면, 여기서의 SOTA는 특정 하나의 모델이 아니라 **AER, LSTM‑AE/VAE/TadGAN, Anomaly Transformer, Azure 등 “현 시점에서 잘 나가는 TSAD 모델들 전체의 평균 성능 레벨”**을 가리키는 표현으로 이해하는 게 정확하다.[4][1]

이 기준 정리가 네가 쓰려는 문맥(“LLM은 zero-shot으로는 쓸만하지만, 정석 TSAD SOTA 대비 아직 성능 갭이 있다”)을 뒷받침하는 근거로 충분해 보이는데, 이걸 논문에서는 어떤 식 문장 구조로 녹여 넣을 생각이야?

[1](https://arxiv.org/html/2405.14755v3)
[2](https://arxiv.org/html/2405.14755v1)
[3](https://www.themoonlight.io/en/review/large-language-models-can-be-zero-shot-anomaly-detectors-for-time-series)
[4](https://dai.lids.mit.edu/wp-content/uploads/2025/08/Salim_Anomaly_Detection.pdf)
[5](https://arxiv.org/abs/2405.14755)
[6](https://yoonji-ha.tistory.com/79)
[7](https://www.youtube.com/watch?v=JDOJDlUrjCY)
[8](https://www.linkedin.com/posts/eamonn-keogh-96ab25143_timeseries-datamining-anomaly-activity-7229673086571229184-XH7X)
[9](https://deeplearn.org/arxiv/517342/large-language-models-can-be-zero-shot-anomaly-detectors-for-time-series)
[10](https://velog.io/@hye0n_w00/Large-language-models-can-be-zero-shot-anomaly-detectors-for-time-series)
[11](https://www.semanticscholar.org/paper/Large-language-models-can-be-zero-shot-anomaly-for-Alnegheimish-Nguyen/8e736e11a7096ae33e75da52d4f057a5113e66e2)
[12](https://github.com/M-3LAB/awesome-industrial-anomaly-detection)
[13](https://www.promptlayer.com/research-papers/large-language-models-can-be-zero-shot-anomaly-detectors-for-time-series)
[14](https://openreview.net/forum?id=LGafQ1g2D2)
[15](https://www.ijcai.org/proceedings/2025/0080.pdf)

