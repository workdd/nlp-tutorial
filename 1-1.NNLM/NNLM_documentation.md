# NNLM (Neural Network Language Model)

## 개요
NNLM(Neural Network Language Model)은 2003년에 발표된 논문 "A Neural Probabilistic Language Model"에서 소개된 언어 모델입니다. 이 모델은 자연어 처리에서 다음 단어를 예측하는 작업을 수행합니다.

## 주요 특징
- 단어 임베딩(Word Embedding)을 학습하는 초기 신경망 기반 언어 모델
- 문맥 정보를 활용하여 다음 단어를 예측
- 기존의 n-gram 모델보다 더 나은 일반화 성능 제공

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
