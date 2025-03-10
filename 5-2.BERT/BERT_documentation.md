# BERT (Bidirectional Encoder Representations from Transformers)

## 개요
BERT는 2018년 Google에서 발표한 "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 논문에서 소개된 모델로, 트랜스포머 인코더를 기반으로 한 양방향 언어 모델입니다. 사전 학습과 미세 조정(fine-tuning)의 두 단계로 구성되며, 다양한 NLP 작업에서 획기적인 성능 향상을 이루었습니다.

## 주요 특징
- 트랜스포머 인코더 기반 구조
- 양방향 문맥 정보 활용
- 마스크드 언어 모델링(Masked Language Modeling)과 다음 문장 예측(Next Sentence Prediction) 작업으로 사전 학습
- 다양한 하위 작업에 쉽게 미세 조정 가능
- WordPiece 토큰화를 통한 미등록 단어 문제 해결

## 구현 내용
이 노트북에서는 PyTorch를 사용하여 간소화된 BERT 모델을 구현하고 있습니다:

1. 마스크드 언어 모델링과 다음 문장 예측을 위한 데이터셋 준비
2. BERT 모델 구현:
   - 임베딩 레이어(토큰, 세그먼트, 위치 임베딩)
   - 트랜스포머 인코더 레이어
   - 마스크드 언어 모델링을 위한 출력 레이어
   - 다음 문장 예측을 위한 출력 레이어
3. 모델 학습 및 성능 평가

## BERT의 사전 학습 작업
1. **마스크드 언어 모델링(MLM)**:
   - 입력 토큰의 15%를 무작위로 마스킹
   - 마스킹된 토큰을 예측하도록 학습
   - 이를 통해 양방향 문맥 정보 학습
   
2. **다음 문장 예측(NSP)**:
   - 두 문장이 연속적인지 여부를 예측
   - 문서 수준의 문맥 이해 능력 향상

## BERT의 장점
- 양방향 문맥 정보를 활용한 풍부한 언어 표현 학습
- 전이 학습(Transfer Learning)을 통한 효율적인 모델 활용
- 다양한 NLP 작업에서 우수한 성능
- 적은 양의 라벨링된 데이터로도 효과적인 미세 조정 가능

## BERT의 응용 분야
- 질의응답
- 감성 분석
- 개체명 인식
- 자연어 추론
- 문서 분류
- 텍스트 요약
- 기계 번역

## BERT의 발전과 영향
BERT는 NLP 분야에 혁명적인 변화를 가져왔으며, 이후 RoBERTa, ALBERT, DistilBERT, ELECTRA 등 다양한 변형 모델이 등장했습니다. 또한 GPT, T5 등 다른 트랜스포머 기반 모델의 발전에도 큰 영향을 미쳤으며, 현대 LLM의 기초가 되었습니다.

## 참고 자료
- 원본 논문: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805)
- BERT는 현대 NLP와 LLM 시스템의 중요한 이정표로, 사전 학습-미세 조정 패러다임을 확립하였습니다.
