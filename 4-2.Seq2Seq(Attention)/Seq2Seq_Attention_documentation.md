# Seq2Seq with Attention

## 개요
Seq2Seq with Attention은 2014년에 발표된 "Neural Machine Translation by Jointly Learning to Align and Translate" 논문에서 소개된 모델로, 기존 Seq2Seq 모델에 어텐션 메커니즘을 추가하여 성능을 크게 향상시켰습니다. 특히 기계 번역 작업에서 획기적인 성능 향상을 이루었습니다.

## 주요 특징
- 인코더-디코더 구조에 어텐션 메커니즘 추가
- 디코더가 출력을 생성할 때 입력 시퀀스의 관련 부분에 집중
- 병목 현상 해결 및 긴 시퀀스 처리 능력 향상
- 입력과 출력 간의 정렬(alignment) 학습

## 구현 내용
이 노트북에서는 PyTorch를 사용하여 Seq2Seq with Attention 모델을 구현하고 있습니다:

1. 번역 작업을 위한 데이터셋 준비
2. 인코더 구현:
   - 임베딩 레이어
   - 양방향 RNN/LSTM 레이어
   - 모든 은닉 상태 출력
3. 어텐션 디코더 구현:
   - 임베딩 레이어
   - 어텐션 계산 메커니즘
   - RNN/LSTM 레이어
   - 출력 레이어
4. 모델 학습 및 번역 성능 평가

## 어텐션 메커니즘의 작동 방식
1. 인코더: 입력 시퀀스의 모든 은닉 상태 출력
2. 디코더: 각 시점에서 다음을 수행
   - 현재 은닉 상태와 인코더 은닉 상태 간의 유사도 계산
   - 유사도를 기반으로 어텐션 가중치 생성
   - 가중치를 사용하여 인코더 은닉 상태의 가중합(context vector) 계산
   - 가중합과 현재 은닉 상태를 결합하여 출력 생성

## 어텐션의 장점
- 긴 시퀀스에서도 정보 손실 최소화
- 입력과 출력 간의 정렬 정보 제공
- 모델의 해석 가능성(interpretability) 향상
- 번역 품질 대폭 향상

## 응용 분야
- 기계 번역
- 텍스트 요약
- 이미지 캡셔닝
- 음성 인식
- 질의응답 시스템

## 참고 자료
- 원본 논문: [Neural Machine Translation by Jointly Learning to Align and Translate(2014)](https://arxiv.org/abs/1409.0473)
- 어텐션 메커니즘은 현대 NLP와 LLM 시스템의 핵심 구성 요소이며, 트랜스포머 모델의 기초가 되었습니다.
