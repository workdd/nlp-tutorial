# Seq2Seq (Sequence-to-Sequence)

## 개요
Seq2Seq는 2014년에 발표된 "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" 논문에서 소개된 모델로, 한 시퀀스를 다른 시퀀스로 변환하는 작업에 사용됩니다. 주로 기계 번역, 요약, 대화 생성 등에 활용됩니다.

## 주요 특징
- 인코더-디코더 구조를 통한 시퀀스 간 변환
- 가변 길이 입력을 가변 길이 출력으로 변환
- 인코더에서 입력 시퀀스의 정보를 압축
- 디코더에서 압축된 정보를 기반으로 출력 생성

## 모델 구조 상세
Seq2Seq은 다음과 같은 구조로 이루어져 있습니다:

1. **인코더(Encoder)**:
   - 입력 시퀀스를 처리하는 RNN/LSTM/GRU
   - 각 입력 토큰을 순차적으로 처리
   - 전체 입력 시퀀스의 정보를 컨텍스트 벡터(context vector)로 압축
   - 일반적으로 마지막 은닉 상태를 컨텍스트 벡터로 사용

2. **디코더(Decoder)**:
   - 출력 시퀀스를 생성하는 RNN/LSTM/GRU
   - 인코더의 컨텍스트 벡터를 초기 은닉 상태로 사용
   - 자기회귀적(autoregressive) 방식으로 토큰 생성
   - 이전 시점의 출력을 다음 시점의 입력으로 사용
   - 특수 토큰 

## 수식 및 수학적 설명
- 인코더의 RNN/LSTM/GRU 계산:
  - $h_t = f(W_{ih}x_t + W_{hh}h_{t-1} + b_h)$
  - $c_t = f(W_{ic}x_t + W_{hc}h_{t-1} + b_c)$
- 디코더의 RNN/LSTM/GRU 계산:
  - $h_t = f(W_{ih}y_t + W_{hh}h_{t-1} + b_h)$
  - $c_t = f(W_{ic}y_t + W_{hc}h_{t-1} + b_c)$
- 출력 계산:
  - $y_t = softmax(W_{ho}h_t + b_o)$

## 하이퍼파라미터
- 인코더와 디코더의 은닉 상태 크기
- 인코더와 디코더의 층 수
- 임베딩 크기
- 드롭아웃 비율

## 계산 복잡도
- 인코더: $O(T \times H \times D)$
- 디코더: $O(T \times H \times D)$
- 전체 모델: $O(2 \times T \times H \times D)$

## 구현 내용
이 노트북에서는 PyTorch를 사용하여 Seq2Seq 모델을 구현하고 있습니다:

1. 단어 변환 작업을 위한 데이터셋 준비
2. 인코더 구현:
   - 임베딩 레이어
   - RNN/LSTM 레이어
   - 최종 은닉 상태 출력
3. 디코더 구현:
   - 임베딩 레이어
   - RNN/LSTM 레이어
   - 출력 레이어
4. 모델 학습 및 단어 변환 성능 평가

## Seq2Seq의 작동 방식
1. 인코더: 입력 시퀀스를 처리하여 문맥 벡터(context vector) 생성
2. 디코더: 문맥 벡터를 초기 상태로 사용하여 출력 시퀀스 생성
3. 교사 강제(Teacher Forcing): 학습 시 디코더의 이전 출력 대신 실제 타겟 사용

## Seq2Seq의 장점
- 다양한 시퀀스 변환 작업에 적용 가능
- 가변 길이 입출력 처리 가능
- 복잡한 언어적 패턴 학습 가능

## Seq2Seq의 한계
- 긴 시퀀스에서 정보 손실 발생
- 문맥 벡터에 모든 정보를 압축하는 병목 현상
- 이러한 한계는 어텐션 메커니즘으로 개선됨

## 응용 분야
- 기계 번역
- 텍스트 요약
- 대화 시스템
- 질의응답
- 텍스트 생성

## 참고 자료
- 원본 논문: [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation(2014)](https://arxiv.org/pdf/1406.1078.pdf)
- Seq2Seq는 현대 NLP와 LLM 시스템의 중요한 구성 요소이며, 트랜스포머 모델의 기초가 되었습니다.
