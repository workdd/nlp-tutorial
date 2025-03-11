# Word2Vec (Skip-gram with Softmax)

## 개요
Word2Vec은 2013년에 발표된 "Distributed Representations of Words and Phrases and their Compositionality" 논문에서 소개된 단어 임베딩 모델입니다. 이 모델은 단어의 의미적 관계를 벡터 공간에 표현합니다.

## 주요 특징
- 단어를 고정된 차원의 실수 벡터로 표현
- Skip-gram 방식: 중심 단어로 주변 단어 예측
- 의미적으로 유사한 단어들은 벡터 공간에서 가까운 위치에 배치됨
- 단어 간의 관계(예: 'king' - 'man' + 'woman' = 'queen')를 벡터 연산으로 표현 가능
- 단어의 분산 표현(distributed representation)을 통해 차원 축소 효과

## 모델 구조 상세
Word2Vec의 Skip-gram 모델은 다음과 같은 구조로 이루어져 있습니다:

1. **입력 레이어**: 중심 단어의 원-핫 인코딩(one-hot encoding)
2. **임베딩 레이어**: 입력층과 은닉층 사이의 가중치 행렬(W)
   - 크기: |V| × N (V: 어휘 크기, N: 임베딩 차원)
   - 학습 후 이 행렬이 단어 임베딩이 됨
3. **출력 레이어**: 은닉층과 출력층 사이의 가중치 행렬(W')
   - 크기: N × |V|
   - 각 단어가 주변 단어일 확률 계산
4. **손실 함수**: 소프트맥스 크로스 엔트로피 (기본 구현)
   - 계산 효율성을 위해 Negative Sampling이나 Hierarchical Softmax로 대체 가능

## 학습 방법
1. **윈도우 기반 샘플링**: 텍스트에서 중심 단어와 문맥 단어 쌍 추출
   - 윈도우 크기(window size): 중심 단어 주변 몇 개의 단어를 고려할지 결정
   - 일반적으로 2~10 사이의 값 사용
2. **네거티브 샘플링(Negative Sampling)**: 모든 단어에 대한 소프트맥스 계산 대신 일부 부정 샘플만 사용
   - 계산 효율성 크게 향상
   - 일반적으로 5~20개의 부정 샘플 사용
3. **서브샘플링(Subsampling)**: 빈도가 높은 단어(the, a, in 등)를 확률적으로 제거
   - 학습 속도 향상 및 희소 단어에 대한 임베딩 품질 개선

## 하이퍼파라미터
- **임베딩 차원(embedding dimension)**: 50~300 (작업에 따라 다름)
- **윈도우 크기(window size)**: 2~10
- **네거티브 샘플 수(negative samples)**: 5~20
- **학습률(learning rate)**: 0.01~0.1 (점진적으로 감소)
- **반복 횟수(epochs)**: 5~15 (데이터셋 크기에 따라 다름)
- **최소 단어 빈도(min_count)**: 학습에 포함할 단어의 최소 등장 횟수 (보통 5~10)

## 계산 복잡도 및 리소스 요구사항
- **시간 복잡도**: 
  - 기본 소프트맥스: O(|V| × N)
  - 네거티브 샘플링 사용 시: O(k × N), k는 네거티브 샘플 수
- **공간 복잡도**: O(|V| × N)
- **GPU 요구사항**: 
  - 중간 규모 어휘(~100K)와 임베딩 차원(~300)의 경우: 2~4GB VRAM
  - 대규모 코퍼스(수십억 단어)의 경우: 8GB+ VRAM 권장
- **학습 시간**: 
  - 중간 규모 데이터셋(위키피디아 일부): GPU 사용 시 수 시간
  - 대규모 데이터셋(전체 위키피디아, 뉴스 코퍼스 등): GPU 사용 시도 1~3일
  - 분산 학습 프레임워크(Gensim 등)를 사용하면 시간 단축 가능

## 장점
- 단순한 구조로 효과적인 단어 표현 학습
- 의미적, 구문적 관계를 벡터 공간에 잘 포착
- 단어 유사도, 단어 유추 등 다양한 작업에 활용 가능
- 사전 학습된 임베딩을 다른 NLP 작업에 전이 학습(transfer learning)으로 활용 가능
- 학습 속도가 빠르고 메모리 효율적 (특히 네거티브 샘플링 사용 시)

## 단점
- 다의어(polysemy) 처리 불가 (한 단어에 하나의 벡터만 할당)
- 문맥에 따른 단어 의미 변화 포착 불가
- 희소한(rare) 단어에 대한 임베딩 품질 저하
- OOV(Out-of-Vocabulary) 단어 처리 방법 없음
- 단어 수준 임베딩으로 구문 구조 정보 제한적 포착

## 실용적 조언
- 대규모 코퍼스(최소 수백만 단어)에서 학습할 때 가장 효과적
- 전처리(토큰화, 불용어 제거, 소문자 변환 등)가 임베딩 품질에 큰 영향
- 사전 학습된 임베딩(Google의 Word2Vec, Facebook의 FastText 등) 활용 고려
- 임베딩 시각화 도구(TensorBoard, t-SNE 등)를 통해 품질 평가
- 특정 도메인 데이터에 맞게 사전 학습된 임베딩을 미세 조정(fine-tuning)하는 것이 효과적

## 응용 분야
- 텍스트 분류
- 감성 분석
- 기계 번역
- 정보 검색
- 추천 시스템

## Word2Vec의 발전
Word2Vec은 이후 FastText, GloVe 등 다양한 임베딩 모델의 기초가 되었으며, 현대 NLP와 LLM의 발전에 중요한 역할을 했습니다. 최근의 BERT, GPT 등의 모델들도 Word2Vec의 아이디어를 확장하여 문맥을 더 잘 반영하는 임베딩을 생성합니다.

## 참고 자료
- 원본 논문: [Distributed Representations of Words and Phrases and their Compositionality(2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- 이 모델은 단어 임베딩의 혁신적인 방법으로, 현대 NLP와 LLM의 기초가 되었습니다.
