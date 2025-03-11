# Transformer

## 개요
Transformer는 2017년에 발표된 "Attention Is All You Need" 논문에서 소개된 모델로, RNN이나 CNN을 사용하지 않고 오직 어텐션 메커니즘만으로 시퀀스 데이터를 처리하는 혁신적인 구조를 제안했습니다. 이 모델은 현대 NLP와 LLM의 기초가 되었으며, BERT, GPT 등 최신 언어 모델의 근간이 되었습니다.

## 주요 특징
- 셀프 어텐션(Self-Attention) 메커니즘 기반 구조
- 인코더-디코더 아키텍처
- 병렬 처리가 가능하여 학습 속도 향상
- 위치 인코딩(Positional Encoding)을 통한 순서 정보 반영
- 멀티 헤드 어텐션(Multi-Head Attention)을 통한 다양한 관점의 정보 포착

## 모델 구조 상세
Transformer는 인코더와 디코더로 구성된 아키텍처를 가지고 있습니다:

1. **인코더(Encoder)**:
   - N개의 동일한 레이어가 쌓인 구조
   - 각 레이어는 두 개의 서브 레이어로 구성:
     - 멀티 헤드 셀프 어텐션(Multi-Head Self-Attention)
     - 위치별 피드 포워드 네트워크(Position-wise Feed-Forward Network)
   - 각 서브 레이어는 잔차 연결(Residual Connection)과 레이어 정규화(Layer Normalization) 적용

2. **디코더(Decoder)**:
   - N개의 동일한 레이어가 쌓인 구조
   - 각 레이어는 세 개의 서브 레이어로 구성:
     - 마스크드 멀티 헤드 셀프 어텐션(Masked Multi-Head Self-Attention)
     - 인코더-디코더 멀티 헤드 어텐션(Encoder-Decoder Multi-Head Attention)
     - 위치별 피드 포워드 네트워크(Position-wise Feed-Forward Network)
   - 각 서브 레이어는 잔차 연결과 레이어 정규화 적용

3. **임베딩 및 위치 인코딩**:
   - 입력 토큰을 고정 크기의 임베딩 벡터로 변환
   - 위치 인코딩을 통해 순서 정보 주입

4. **출력 레이어**:
   - 선형 변환과 소프트맥스를 통한 다음 토큰 확률 계산

## 수식 및 수학적 설명
Transformer의 핵심 수식은 다음과 같습니다:

1. **스케일드 닷-프로덕트 어텐션(Scaled Dot-Product Attention)**:
   - $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
   - $Q$: 쿼리 행렬, $K$: 키 행렬, $V$: 값 행렬
   - $d_k$: 키 벡터의 차원

2. **멀티 헤드 어텐션(Multi-Head Attention)**:
   - $\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h)W^O$
   - $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
   - $W_i^Q, W_i^K, W_i^V, W^O$: 학습 가능한 파라미터 행렬

3. **위치별 피드 포워드 네트워크(Position-wise Feed-Forward Network)**:
   - $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$
   - $W_1, W_2, b_1, b_2$: 학습 가능한 파라미터

4. **위치 인코딩(Positional Encoding)**:
   - $\text{PE}_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$
   - $\text{PE}_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$
   - $pos$: 위치, $i$: 차원, $d_{model}$: 모델 차원

5. **잔차 연결 및 레이어 정규화**:
   - $\text{LayerNorm}(x + \text{Sublayer}(x))$
   - $\text{Sublayer}(x)$: 서브 레이어 함수(어텐션 또는 피드 포워드)

## 하이퍼파라미터
Transformer 모델의 주요 하이퍼파라미터는 다음과 같습니다:

- 모델 차원(d_model): 일반적으로 512
- 피드 포워드 네트워크 차원(d_ff): 일반적으로 2048
- 어텐션 헤드 수(h): 일반적으로 8
- 인코더/디코더 레이어 수(N): 일반적으로 6
- 드롭아웃 비율: 일반적으로 0.1
- 학습률 및 워밍업 스텝
- 레이블 스무딩(Label Smoothing) 계수: 일반적으로 0.1

## 계산 복잡도 및 리소스 요구사항
- **시간 복잡도**:
  - 셀프 어텐션: $O(n^2 \times d)$, $n$은 시퀀스 길이, $d$는 모델 차원
  - RNN 대비: $O(n \times d^2)$
  - 병렬 처리 가능하여 실제 학습 시간은 크게 단축
  
- **공간 복잡도**:
  - 어텐션 점수 행렬: $O(n^2)$
  - 모델 파라미터: $O(N \times d^2)$, $N$은 레이어 수
  
- **메모리 요구사항**:
  - 배치 크기와 시퀀스 길이에 비례하여 메모리 사용량 증가
  - 특히 어텐션 점수 행렬이 시퀀스 길이의 제곱에 비례하여 증가

## 성능 비교
- **RNN/LSTM 기반 모델 대비**:
  - 번역 작업에서 BLEU 점수 약 2-3점 향상
  - 학습 시간 크게 단축(병렬 처리 가능)
  - 긴 시퀀스에서 더 효과적인 정보 포착
  
- **CNN 기반 모델 대비**:
  - 전역적 의존성 포착 능력 우수
  - 더 적은 레이어로 유사한 성능 달성
  
- **이후 발전된 모델들(BERT, GPT 등)의 기초**:
  - 사전 학습-미세 조정 패러다임과 결합하여 성능 극대화
  - 다양한 NLP 작업에서 최첨단 성능 달성

## 구현 내용
이 노트북에서는 PyTorch를 사용하여 Transformer 모델을 구현하고 있습니다:

1. 번역 작업을 위한 데이터셋 준비
2. 인코더 구현:
   - 위치 인코딩
   - 멀티 헤드 셀프 어텐션
   - 피드 포워드 네트워크
   - 레이어 정규화 및 잔차 연결
3. 디코더 구현:
   - 위치 인코딩
   - 마스크드 멀티 헤드 셀프 어텐션
   - 인코더-디코더 어텐션
   - 피드 포워드 네트워크
   - 레이어 정규화 및 잔차 연결
4. 모델 학습 및 번역 성능 평가

## Transformer의 주요 구성 요소
1. **멀티 헤드 어텐션(Multi-Head Attention)**:
   - 쿼리(Q), 키(K), 값(V) 행렬을 사용한 스케일드 닷-프로덕트 어텐션
   - 여러 개의 어텐션 헤드를 병렬로 계산하여 다양한 관점의 정보 포착
   
2. **위치별 피드 포워드 네트워크(Position-wise Feed-Forward Network)**:
   - 두 개의 선형 변환과 ReLU 활성화 함수로 구성
   
3. **레이어 정규화(Layer Normalization) 및 잔차 연결(Residual Connection)**:
   - 학습 안정성 향상 및 기울기 소실 방지
   
4. **위치 인코딩(Positional Encoding)**:
   - 순서 정보를 모델에 주입하기 위한 사인/코사인 함수 기반 인코딩

## 장점
- 병렬 처리를 통한 학습 속도 향상
- 긴 시퀀스에서도 효과적인 정보 포착
- 문맥 이해 능력 향상
- 다양한 NLP 작업에 적용 가능한 범용성
- 인코더와 디코더를 독립적으로 사용 가능(BERT, GPT 등)
- 어텐션 가중치를 통한 해석 가능성 제공

## 단점
- 메모리 사용량이 시퀀스 길이의 제곱에 비례하여 증가
- 매우 긴 시퀀스(수천 토큰 이상)에서 계산 효율성 저하
- 위치 인코딩의 한계로 인한 초장거리 의존성 포착의 어려움
- 사전 학습 없이는 작은 데이터셋에서 과적합 위험

## 실용적 조언
- 충분한 데이터로 학습하여 과적합 방지
- 그래디언트 클리핑(Gradient Clipping)을 통한 안정적인 학습
- 학습률 스케줄링 및 워밍업 단계 적용
- 레이블 스무딩(Label Smoothing)을 통한 일반화 성능 향상
- 빔 서치(Beam Search)를 통한 디코딩 품질 향상
- 모델 크기와 데이터셋 크기의 균형 조정

## 응용 분야
- 기계 번역
- 텍스트 요약
- 텍스트 생성
- 질의응답 시스템
- 감성 분석
- 언어 모델링

## 시각화 및 해석
Transformer 모델의 주요 장점 중 하나는 어텐션 가중치를 시각화하여 모델이 어떤 단어에 집중하는지 확인할 수 있다는 점입니다. 이를 통해 모델의 결정 과정을 해석할 수 있으며, 특히 번역 작업에서는 소스 언어와 타겟 언어 간의 단어 정렬을 파악할 수 있습니다.

## 향후 연구 방향
- 더 효율적인 어텐션 메커니즘 개발(선형 복잡도)
- 더 효과적인 위치 인코딩 방법 연구
- 다양한 모달리티(이미지, 오디오 등)로의 확장
- 더 큰 모델과 더 많은 데이터를 통한 성능 향상
- 추론 속도 개선을 위한 모델 압축 및 양자화

## 참고 자료
- 원본 논문: [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems.
- Transformer는 현대 NLP와 LLM 시스템의 근간이 되는 모델로, BERT, GPT, T5 등 최신 언어 모델의 기초가 되었습니다.

## 결론
Transformer는 NLP 분야에 혁명적인 변화를 가져온 모델로, 어텐션 메커니즘만을 사용하여 RNN과 CNN의 한계를 극복했습니다. 이 모델은 병렬 처리가 가능하고 긴 시퀀스에서도 효과적인 정보 포착이 가능하여, 현대 대규모 언어 모델(LLM)의 기초가 되었습니다. Transformer의 등장 이후, NLP 분야는 BERT, GPT, T5 등 다양한 사전 학습 모델의 발전으로 이어졌으며, 이는 자연어 처리 작업에서 획기적인 성능 향상을 가져왔습니다.
