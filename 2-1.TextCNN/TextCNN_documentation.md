# TextCNN (Convolutional Neural Network for Text)

## 개요
TextCNN은 2014년에 발표된 "Convolutional Neural Networks for Sentence Classification" 논문에서 소개된 모델로, 컴퓨터 비전에서 성공적으로 사용된 CNN(Convolutional Neural Network)을 텍스트 분류에 적용한 모델입니다.

## 주요 특징
- 1D 컨볼루션을 사용하여 텍스트의 지역적 패턴 포착
- 다양한 크기의 필터를 통해 n-gram 효과 구현
- 맥스 풀링(Max Pooling)을 통한 중요 특성 추출
- 이미지 처리에 사용되는 CNN의 아이디어를 텍스트에 적용

## 구현 내용
이 노트북에서는 PyTorch를 사용하여 TextCNN 모델을 구현하고 있습니다:

1. 이진 감성 분류를 위한 데이터셋 준비
2. 단어 임베딩 초기화
3. TextCNN 모델 구현:
   - 다양한 크기의 컨볼루션 필터
   - 맥스 풀링 레이어
   - 드롭아웃(Dropout)을 통한 정규화
   - 완전 연결 레이어(Fully Connected Layer)
4. 모델 학습 및 감성 분류 성능 평가

## TextCNN의 장점
- 병렬 처리가 가능하여 RNN보다 학습 속도가 빠름
- 지역적 특성(local feature)을 효과적으로 포착
- 비교적 간단한 구조로 좋은 성능 달성
- 긴 의존성(long-term dependency) 문제에서 자유로움

## 응용 분야
- 감성 분석
- 스팸 탐지
- 주제 분류
- 의도 분류
- 질문 분류

## 참고 자료
- 원본 논문: [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)
- TextCNN은 텍스트 분류 작업에서 기준(baseline) 모델로 자주 사용되며, 현대 NLP와 LLM 시스템의 중요한 구성 요소 중 하나입니다.
