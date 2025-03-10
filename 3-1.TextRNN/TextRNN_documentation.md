# TextRNN (Recurrent Neural Network for Text)

## 개요
TextRNN은 1990년에 발표된 "Finding Structure in Time" 논문에서 소개된 RNN(Recurrent Neural Network) 개념을 텍스트 처리에 적용한 모델입니다. 시퀀스 데이터의 시간적 의존성을 모델링하는 데 효과적입니다.

## 주요 특징
- 순차적 데이터 처리에 특화된 신경망 구조
- 이전 상태(hidden state)를 기억하여 문맥 정보 유지
- 가변 길이 입력 처리 가능
- 다음 단어/토큰 예측에 효과적

## 구현 내용
이 노트북에서는 PyTorch를 사용하여 TextRNN 모델을 구현하고 있습니다:

1. 텍스트 데이터셋 준비 및 전처리
2. 단어 임베딩 초기화
3. TextRNN 모델 구현:
   - 임베딩 레이어
   - RNN 셀(Cell)
   - 출력 레이어
4. 모델 학습 및 다음 단어 예측 성능 평가

## TextRNN의 장점
- 시퀀스 데이터의 시간적 패턴 포착 능력
- 가변 길이 입력 처리 가능
- 문맥 정보를 활용한 예측 성능

## TextRNN의 한계
- 긴 시퀀스에서 기울기 소실/폭발 문제
- 장기 의존성(long-term dependency) 포착의 어려움
- 병렬 처리의 제한

## 응용 분야
- 언어 모델링
- 텍스트 생성
- 감성 분석
- 기계 번역
- 음성 인식

## 참고 자료
- 원본 논문: [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
- TextRNN은 LSTM, GRU 등의 발전된 RNN 구조의 기초가 되었으며, 현대 NLP와 LLM 시스템에서 중요한 역할을 합니다.
