# Word2Vec (Skip-gram with Softmax)

## 개요
Word2Vec은 2013년에 발표된 "Distributed Representations of Words and Phrases and their Compositionality" 논문에서 소개된 단어 임베딩 모델입니다. 이 모델은 단어의 의미적 관계를 벡터 공간에 표현합니다.

## 주요 특징
- 단어를 고정된 차원의 실수 벡터로 표현
- Skip-gram 방식: 중심 단어로 주변 단어 예측
- 의미적으로 유사한 단어들은 벡터 공간에서 가까운 위치에 배치됨
- 단어 간의 관계(예: 'king' - 'man' + 'woman' = 'queen')를 벡터 연산으로 표현 가능

## 구현 내용
이 노트북에서는 PyTorch를 사용하여 Skip-gram 모델을 구현하고 있습니다:

1. 문장 데이터셋 생성 및 전처리
2. 중심 단어와 주변 단어 쌍 생성
3. Skip-gram 모델 구현:
   - 입력층과 출력층 임베딩 행렬
   - Softmax 함수를 통한 확률 계산
4. 모델 학습 및 단어 벡터 시각화

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
