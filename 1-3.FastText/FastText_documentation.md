# FastText

## 개요
FastText는 2016년 Facebook AI Research에서 발표한 "Bag of Tricks for Efficient Text Classification" 논문에서 소개된 텍스트 분류 모델입니다. Word2Vec의 확장 버전으로, 단어 수준을 넘어 하위 단어(subword) 정보를 활용합니다.

## 주요 특징
- 단어를 n-gram 문자 단위로 분해하여 표현
- 희소한(rare) 단어와 미등록 단어(OOV, Out-of-Vocabulary)에 대한 처리 개선
- 형태학적으로 풍부한 언어(한국어, 핀란드어 등)에 효과적
- 빠른 학습 속도와 효율적인 텍스트 분류 성능

## 구현 내용
이 노트북에서는 PyTorch를 사용하여 FastText 모델을 구현하고 있습니다:

1. 문장 분류를 위한 데이터셋 준비
2. 단어와 n-gram 문자 단위 처리
3. FastText 모델 구현:
   - 임베딩 레이어
   - 평균 풀링(Average Pooling)
   - 분류 레이어
4. 모델 학습 및 문장 분류 성능 평가

## FastText의 장점
- 단어의 내부 구조를 고려하여 더 풍부한 표현 학습
- 미등록 단어에 대해서도 의미 있는 벡터 생성 가능
- 작은 데이터셋에서도 비교적 좋은 성능
- 계산 효율성이 높아 대규모 데이터셋에 적합

## 응용 분야
- 텍스트 분류
- 감성 분석
- 언어 식별
- 스팸 필터링
- 문서 태깅

## 참고 자료
- 원본 논문: [Bag of Tricks for Efficient Text Classification(2016)](https://arxiv.org/pdf/1607.01759.pdf)
- FastText는 효율적인 텍스트 분류와 단어 표현 학습을 위한 중요한 모델로, 현대 NLP와 LLM 시스템에서 널리 사용됩니다.
