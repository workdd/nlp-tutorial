# NNLM (Neural Network Language Model)

## 개요
NNLM(Neural Network Language Model)은 2003년에 발표된 논문 "A Neural Probabilistic Language Model"에서 소개된 언어 모델입니다. 이 모델은 자연어 처리에서 다음 단어를 예측하는 작업을 수행합니다.

## 주요 특징
- 단어 임베딩(Word Embedding)을 학습하는 초기 신경망 기반 언어 모델
- 문맥 정보를 활용하여 다음 단어를 예측
- 기존의 n-gram 모델보다 더 나은 일반화 성능 제공
- 고정된 윈도우 크기(n-1)의 문맥만 고려하는 제한적 구조

## 모델 구조 상세
NNLM 모델은 다음과 같은 계층 구조로 이루어져 있습니다:

1. **입력 레이어**: n-1개의 단어 인덱스를 입력으로 받음
2. **임베딩 레이어(C)**: 각 단어를 m차원의 분산 표현(distributed representation)으로 변환
3. **은닉층(H)**: 임베딩된 단어들을 연결(concatenate)하여 비선형 변환 적용
   - 활성화 함수로 tanh 사용
   - 수식: tanh(d + Hx), 여기서 d는 편향 벡터
4. **출력층**: 은닉층의 출력과 임베딩 레이어의 직접 연결(direct connection)을 결합
   - 수식: y = b + Wx + Utanh(d + Hx)
   - 소프트맥스 함수를 통해 다음 단어의 확률 분포 생성

## 하이퍼파라미터
- n_step (n-1): 문맥 윈도우 크기 (예제에서는 2)
- n_hidden (h): 은닉층의 뉴런 수 (예제에서는 2)
- m: 임베딩 차원 (예제에서는 2)
- n_class: 어휘 크기 (예제에서는 7)

## 계산 복잡도 및 리소스 요구사항
- **시간 복잡도**: O(n_step * m * n_hidden + n_hidden * n_class)
- **공간 복잡도**: O(n_class * m + n_hidden * n_class)
- **GPU 요구사항**: 매우 낮음 (예제 코드는 CPU에서도 빠르게 실행 가능)
- **학습 시간**: 
  - 작은 데이터셋(예제 코드)의 경우: 몇 초 이내
  - 실제 어휘 크기(10,000~50,000)와 대규모 코퍼스의 경우: 수 시간 ~ 하루
  - 임베딩 차원과 은닉층 크기에 따라 크게 달라짐

## 장점
- 구현이 간단하고 이해하기 쉬움
- 작은 데이터셋에서도 학습 가능
- n-gram 모델보다 더 나은 일반화 성능
- 단어의 의미적 유사성을 벡터 공간에 표현
- 현대 언어 모델의 기초가 되는 개념 제공

## 단점
- 고정된 윈도우 크기로 인한 장거리 의존성(long-range dependency) 포착 불가
- 윈도우 크기를 늘리면 파라미터 수가 급격히 증가
- 대규모 어휘에 대해 계산 비용이 높음 (출력층의 소프트맥스 계산)
- RNN, LSTM 등 현대적 모델에 비해 성능 제한적
- 문맥 윈도우 내 단어 순서 정보가 임베딩 연결 과정에서 일부 손실

## 실용적 조언
- 초기 학습률은 0.001~0.01 범위에서 설정하는 것이 좋음
- 임베딩 차원(m)은 보통 어휘 크기의 제곱근 정도로 설정 (실제 적용 시 50~300 범위)
- 은닉층 크기(n_hidden)는 임베딩 차원의 1~2배 정도로 설정
- 대규모 어휘에 대해서는 Hierarchical Softmax나 Negative Sampling 기법 고려
- 과적합 방지를 위해 드롭아웃(dropout)이나 가중치 감쇠(weight decay) 적용 권장

## 구현 내용
이 노트북에서는 PyTorch를 사용하여 간단한 NNLM 모델을 구현하고 있습니다:

1. 간단한 문장 데이터셋 생성 ("i like dog", "i love coffee", "i hate milk")
2. 단어 사전(vocabulary) 구축
3. NNLM 모델 구현:
   - 단어 임베딩 레이어
   - 은닉층(Hidden Layer)
   - 출력층(Output Layer)
4. 모델 학습 및 예측 수행

## 수식
NNLM 모델은 다음과 같은 수식으로 표현됩니다:
- 임베딩 레이어: C(X)
- 은닉층: tanh(d + H(X))
- 출력층: b + W(X) + U(tanh(d + H(X)))

## 응용 분야
- 언어 모델링
- 텍스트 생성
- 자연어 이해의 기초 모델

## 참고 자료
- 원본 논문: [A Neural Probabilistic Language Model(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- 이 모델은 현대 NLP와 LLM 발전의 중요한 기초가 되었습니다.
