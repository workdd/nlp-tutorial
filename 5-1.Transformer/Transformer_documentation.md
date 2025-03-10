# Transformer

## 개요
Transformer는 2017년에 발표된 "Attention Is All You Need" 논문에서 소개된 모델로, RNN이나 CNN을 사용하지 않고 오직 어텐션 메커니즘만으로 시퀀스 데이터를 처리하는 혁신적인 구조를 제안했습니다. 이 모델은 현대 NLP와 LLM의 기초가 되었으며, BERT, GPT 등 최신 언어 모델의 근간이 되었습니다.

## 주요 특징
- 셀프 어텐션(Self-Attention) 메커니즘 기반 구조
- 인코더-디코더 아키텍처
- 병렬 처리가 가능하여 학습 속도 향상
- 위치 인코딩(Positional Encoding)을 통한 순서 정보 반영
- 멀티 헤드 어텐션(Multi-Head Attention)을 통한 다양한 관점의 정보 포착

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

## Transformer의 장점
- 병렬 처리를 통한 학습 속도 향상
- 긴 시퀀스에서도 효과적인 정보 포착
- 문맥 이해 능력 향상
- 다양한 NLP 작업에 적용 가능한 범용성

## 응용 분야
- 기계 번역
- 텍스트 요약
- 텍스트 생성
- 질의응답 시스템
- 감성 분석
- 언어 모델링

## 참고 자료
- 원본 논문: [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)
- Transformer는 현대 NLP와 LLM 시스템의 근간이 되는 모델로, BERT, GPT, T5 등 최신 언어 모델의 기초가 되었습니다.
