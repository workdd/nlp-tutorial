# TextLSTM (Long Short-Term Memory for Text)

## 개요
TextLSTM은 1997년에 발표된 "LONG SHORT-TERM MEMORY" 논문에서 소개된 LSTM(Long Short-Term Memory) 구조를 텍스트 처리에 적용한 모델입니다. 기존 RNN의 장기 의존성 문제를 해결하기 위해 설계되었습니다.

## 주요 특징
- 장기 의존성(Long-term dependency) 문제 해결을 위한 게이트 메커니즘
- 기울기 소실(Vanishing gradient) 문제 완화
- 정보의 선택적 기억과 망각 가능
- 복잡한 시퀀스 패턴 학습에 효과적

## 모델 구조 상세
TextLSTM은 다음과 같은 구조로 이루어져 있습니다:

1. **임베딩 레이어**:
   - 각 단어를 고정 크기 벡터로 변환
   - 크기: (배치 크기, 문장 길이, 임베딩 차원)
   - 사전 학습된 임베딩 사용 가능

2. **LSTM 레이어**:
   - 세 개의 게이트로 구성: 입력 게이트, 망각 게이트, 출력 게이트
   - 셀 상태(cell state)와 은닉 상태(hidden state)를 별도로 관리
   - 게이트를 통해 정보의 흐름을 제어
   - 다중 레이어 구성 가능

3. **출력 레이어**:
   - 마지막 은닉 상태를 입력으로 사용
   - 드롭아웃을 통한 정규화
   - 완전 연결 레이어를 통해 최종 분류

## LSTM 셀 구조
LSTM 셀은 다음 네 가지 주요 컴포넌트로 구성됩니다:

1. **망각 게이트(Forget Gate)**:
   - 이전 셀 상태에서 어떤 정보를 버릴지 결정
   - σ(W_f · [h_(t-1), x_t] + b_f)

2. **입력 게이트(Input Gate)**:
   - 새로운 정보 중 어떤 것을 저장할지 결정
   - i_t = σ(W_i · [h_(t-1), x_t] + b_i)
   - 새로운 후보 값 생성: C̃_t = tanh(W_C · [h_(t-1), x_t] + b_C)

3. **셀 상태 업데이트(Cell State Update)**:
   - 이전 셀 상태와 새 정보를 결합
   - C_t = f_t * C_(t-1) + i_t * C̃_t

4. **출력 게이트(Output Gate)**:
   - 어떤 정보를 출력할지 결정
   - o_t = σ(W_o · [h_(t-1), x_t] + b_o)
   - 은닉 상태 계산: h_t = o_t * tanh(C_t)

## 수식 표현
LSTM의 전체 수식은 다음과 같습니다:

1. 망각 게이트: 
   - f_t = σ(W_f · [h_(t-1), x_t] + b_f)

2. 입력 게이트:
   - i_t = σ(W_i · [h_(t-1), x_t] + b_i)
   - C̃_t = tanh(W_C · [h_(t-1), x_t] + b_C)

3. 셀 상태 업데이트:
   - C_t = f_t * C_(t-1) + i_t * C̃_t

4. 출력 게이트:
   - o_t = σ(W_o · [h_(t-1), x_t] + b_o)
   - h_t = o_t * tanh(C_t)

5. 최종 출력:
   - y_t = softmax(W_y · h_t + b_y)

## 하이퍼파라미터
- **임베딩 차원(embedding dimension)**: 100~300
- **은닉층 크기(hidden size)**: 128~1024
- **LSTM 레이어 수(num_layers)**: 1~4
- **드롭아웃 비율(dropout rate)**: 0.2~0.5
- **학습률(learning rate)**: 0.001~0.005
- **배치 크기(batch size)**: 16~128
- **에포크 수(epochs)**: 10~100
- **그래디언트 클리핑(gradient clipping)**: 1.0~5.0
- **LSTM 변형**: 기본 LSTM, Peephole LSTM, LSTM with projection 등

## 계산 복잡도 및 리소스 요구사항
- **시간 복잡도**: 
  - 순전파: O(4 × T × H × (H + D)), T는 문장 길이, H는 은닉층 크기, D는 임베딩 차원
  - 역전파: O(4 × T × H × (H + D))
- **공간 복잡도**: O(T × H + 4 × H × (H + D) + 4 × H)
- **GPU 요구사항**: 
  - 작은 모델(은닉층 크기 128): 2~4GB VRAM
  - 중간 크기 모델(은닉층 크기 512): 4~8GB VRAM
  - 대규모 모델(은닉층 크기 1024, 다중 레이어): 8~16GB VRAM
- **학습 시간**: 
  - 작은 데이터셋(수천 문서): GPU에서 30분~1시간
  - 중간 규모 데이터셋(수만 문서): GPU에서 2~6시간
  - 대규모 데이터셋(수십만 문서): GPU에서 하루~수일
  - 기본 RNN보다 2~3배 느린 학습 속도

## 성능 비교
| 모델 | 학습 속도 | 추론 속도 | 장기 의존성 처리 | 기울기 안정성 | 메모리 사용량 |
|------|---------|---------|--------------|------------|------------|
| TextRNN | 보통 | 보통 | 제한적 | 낮음 | 낮음 |
| TextLSTM | 느림 | 느림 | 우수 | 높음 | 높음 |
| GRU | 느림 | 느림 | 우수 | 높음 | 중간 |
| Transformer | 빠름 | 빠름 | 매우 우수 | 매우 높음 | 매우 높음 |

## 장점
- 장기 의존성 문제 효과적 해결
- 기울기 소실 문제 완화
- 정보의 선택적 기억과 망각 가능
- 복잡한 시퀀스 패턴 학습에 효과적
- 다양한 길이의 입력 처리 가능

## 단점
- 계산 복잡도가 높아 학습 속도가 느림
- 많은 파라미터로 인한 과적합 위험
- 병렬 처리의 어려움
- 매우 긴 시퀀스에서는 여전히 성능 제한
- 구현 및 튜닝이 상대적으로 복잡

## 실용적 조언
- 양방향(bidirectional) LSTM 사용 시 성능 향상
- 그래디언트 클리핑 적용으로 학습 안정화
- 드롭아웃을 LSTM 레이어 사이에 적용 (recurrent dropout)
- 레이어 정규화(layer normalization) 고려
- 시퀀스 길이가 매우 길 경우 계층적 접근 또는 어텐션 메커니즘 결합
- 배치 크기와 학습률 튜닝에 주의 (작은 배치, 낮은 학습률 시작)
- 사전 학습된 임베딩 사용 시 성능 향상
- 초기 에포크에서는 임베딩을 고정하고 후반부에 미세 조정

## 응용 분야
- 언어 모델링
- 텍스트 생성
- 자동 완성
- 기계 번역
- 감성 분석
- 질의응답 시스템

## 참고 자료
- 원본 논문: [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- TextLSTM은 현대 NLP와 LLM 시스템의 중요한 구성 요소이며, 트랜스포머 모델이 등장하기 전까지 시퀀스 모델링의 표준으로 사용되었습니다.
