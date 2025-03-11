# Bi-LSTM with Attention

## 개요
Bi-LSTM with Attention은 양방향 LSTM에 어텐션 메커니즘을 결합한 모델로, 시퀀스의 양방향 문맥을 포착하면서 중요한 부분에 집중할 수 있는 능력을 갖추고 있습니다. 특히 감성 분석과 같은 텍스트 분류 작업에서 우수한 성능을 보입니다.

## 주요 특징
- 양방향 LSTM을 통한 문맥 정보 포착
- 어텐션 메커니즘을 통한 중요 정보 강조
- 문장 내 중요 단어에 가중치 부여
- 분류 작업에 최적화된 구조

## 모델 구조 상세
Bi-LSTM with Attention은 다음과 같은 구조로 이루어져 있습니다:

1. **임베딩 레이어(Embedding Layer)**:
   - 각 단어를 고정 크기의 밀집 벡터로 변환
   - 사전 훈련된 임베딩(예: Word2Vec, GloVe) 사용 가능
   - 임베딩 차원: 일반적으로 100~300 사이

2. **양방향 LSTM 레이어(Bidirectional LSTM Layer)**:
   - 순방향 LSTM: 문장을 처음부터 끝까지 처리
   - 역방향 LSTM: 문장을 끝에서 처음까지 처리
   - 각 단어 위치에서 양방향 은닉 상태 결합: $h_i = [\overrightarrow{h_i}; \overleftarrow{h_i}]$
   - 모든 단어 위치의 은닉 상태 수집: $H = [h_1, h_2, ..., h_T]$

3. **어텐션 메커니즘(Attention Mechanism)**:
   - 셀프 어텐션(Self-Attention) 적용
   - 각 단어의 중요도 계산
   - 중요도에 따른 가중치 생성
   - 가중치를 사용한 문맥 벡터 생성

4. **출력 레이어(Output Layer)**:
   - 문맥 벡터를 입력으로 받는 완전 연결 레이어
   - 분류를 위한 소프트맥스 또는 시그모이드 활성화 함수

## 수식 및 수학적 설명
Bi-LSTM with Attention의 핵심 수식은 다음과 같습니다:

1. **양방향 LSTM 처리**:
   - 순방향 LSTM: $\overrightarrow{h_t} = \overrightarrow{LSTM}(x_t, \overrightarrow{h_{t-1}})$
   - 역방향 LSTM: $\overleftarrow{h_t} = \overleftarrow{LSTM}(x_t, \overleftarrow{h_{t+1}})$
   - 결합된 은닉 상태: $h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$

2. **어텐션 점수 계산**:
   - $u_t = \tanh(W_h h_t + b_h)$
   - $\alpha_t = \frac{\exp(u_t^T u_w)}{\sum_{j=1}^{T} \exp(u_j^T u_w)}$

3. **문맥 벡터 계산**:
   - $c = \sum_{t=1}^{T} \alpha_t h_t$

4. **출력 계산**:
   - 이진 분류: $\hat{y} = \sigma(W_c c + b_c)$
   - 다중 분류: $\hat{y} = \text{softmax}(W_c c + b_c)$

여기서:
- $x_t$: t번째 단어의 임베딩 벡터
- $h_t$: t번째 단어의 결합된 양방향 은닉 상태
- $u_t$: t번째 단어의 은닉 표현
- $\alpha_t$: t번째 단어의 어텐션 가중치
- $c$: 문맥 벡터
- $\hat{y}$: 예측 출력
- $W_h, W_c, u_w$: 학습 가능한 파라미터
- $b_h, b_c$: 편향 항

## 하이퍼파라미터
Bi-LSTM with Attention 모델의 주요 하이퍼파라미터는 다음과 같습니다:

- 임베딩 차원
- LSTM 은닉 상태 크기
- LSTM 층 수
- 드롭아웃 비율
- 학습률
- 배치 크기
- 어텐션 벡터 차원

## 계산 복잡도 및 리소스 요구사항
- **시간 복잡도**:
  - 양방향 LSTM: $O(T \times D \times H)$
  - 어텐션 계산: $O(T \times H)$
  - 전체: $O(T \times D \times H)$
  
- **공간 복잡도**:
  - 임베딩 행렬: $O(V \times D)$
  - LSTM 파라미터: $O(4 \times H \times (H + D))$
  - 어텐션 파라미터: $O(H)$
  - 전체 모델: $O(V \times D + H \times (H + D))$

- **메모리 요구사항**:
  - 배치 크기에 비례하여 메모리 사용량 증가
  - 문장 길이가 길수록 메모리 요구량 증가

## 성능 비교
- **일반 LSTM 대비**:
  - 분류 정확도: 약 2-5% 성능 향상
  - 특히 긴 문장에서 성능 차이 두드러짐
  
- **CNN 기반 모델 대비**:
  - 문맥 정보 포착 능력 우수
  - 지역적 패턴과 전역적 의존성 모두 포착 가능
  
- **트랜스포머 모델 대비**:
  - 계산 효율성 높음
  - 작은 데이터셋에서 경쟁력 있는 성능
  - 대규모 데이터셋에서는 트랜스포머에 비해 성능 열세

## 구현 내용
이 노트북에서는 PyTorch를 사용하여 Bi-LSTM with Attention 모델을 구현하고 있습니다:

1. 이진 감성 분류를 위한 데이터셋 준비
2. 모델 구현:
   - 임베딩 레이어
   - 양방향 LSTM 레이어
   - 셀프 어텐션(Self-Attention) 메커니즘
   - 어텐션 가중치를 통한 문맥 벡터 생성
   - 출력 레이어
3. 모델 학습 및 감성 분류 성능 평가

## 어텐션 메커니즘의 작동 방식
1. 양방향 LSTM을 통해 각 단어의 은닉 상태 생성
2. 은닉 상태에 대한 어텐션 점수 계산
3. 소프트맥스를 통해 어텐션 가중치 생성
4. 가중치를 사용하여 은닉 상태의 가중합 계산
5. 최종 문맥 벡터를 사용하여 분류 수행

## 장점
- 문장 내 중요 단어 식별 능력
- 긴 문장에서도 효과적인 정보 포착
- 모델의 결정에 대한 해석 가능성 제공
- 분류 성능 향상
- 양방향 문맥 정보 활용

## 단점
- 순차적 처리로 인한 병렬화 한계
- 매우 긴 문장에서 여전히 정보 손실 가능성
- 트랜스포머 모델에 비해 장거리 의존성 포착 능력 제한적
- 학습 시간이 상대적으로 긺

## 실용적 조언
- 사전 훈련된 임베딩(Word2Vec, GloVe, FastText)을 사용하면 성능 향상
- 드롭아웃을 적용하여 과적합 방지
- 그래디언트 클리핑(Gradient Clipping)을 통한 안정적인 학습
- 배치 정규화(Batch Normalization)로 학습 가속화
- 어텐션 가중치 시각화를 통한 모델 해석 및 디버깅

## 응용 분야
- 감성 분석
- 문서 분류
- 의도 분류
- 관계 추출
- 질의응답 시스템

## 시각화 및 해석
Bi-LSTM with Attention 모델의 주요 장점 중 하나는 어텐션 가중치를 시각화하여 모델이 어떤 단어에 집중하는지 확인할 수 있다는 점입니다. 이를 통해 모델의 결정 과정을 해석할 수 있으며, 특히 감성 분석에서는 어떤 단어가 긍정/부정 감성에 기여하는지 파악할 수 있습니다.

## 참고 자료
- Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016). Hierarchical attention networks for document classification. In Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies.
- Bi-LSTM with Attention은 현대 NLP와 LLM 시스템에서 중요한 구성 요소이며, 특히 분류 작업에서 널리 사용됩니다.
- 이 모델은 트랜스포머 모델이 등장하기 전까지 텍스트 분류의 최첨단 모델로 사용되었습니다.

## 향후 연구 방향
- Bi-LSTM with Attention 모델의 성능을 더욱 향상시키기 위한 연구
- 다른 어텐션 메커니즘의 적용
- 다양한 데이터셋에 대한 모델의 성능 평가
- 모델의 해석 가능성 향상을 위한 연구

## 결론
Bi-LSTM with Attention 모델은 텍스트 분류 작업에서 우수한 성능을 보이는 모델입니다. 이 모델은 양방향 LSTM과 어텐션 메커니즘을 결합하여 문장 내 중요 단어를 식별하고, 분류 성능을 향상시킵니다. 또한, 모델의 결정 과정을 해석할 수 있는 장점이 있습니다. 향후 연구에서는 이 모델의 성능을 더욱 향상시키기 위한 연구와 다른 어텐션 메커니즘의 적용이 필요합니다.
