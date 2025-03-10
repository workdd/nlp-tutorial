# TextLSTM (Long Short-Term Memory for Text)

## 개요
TextLSTM은 1997년에 발표된 "LONG SHORT-TERM MEMORY" 논문에서 소개된 LSTM(Long Short-Term Memory) 구조를 텍스트 처리에 적용한 모델입니다. 기존 RNN의 장기 의존성 문제를 해결하기 위해 설계되었습니다.

## 주요 특징
- 게이트 메커니즘(입력, 망각, 출력 게이트)을 통한 정보 흐름 제어
- 장기 의존성(long-term dependency) 문제 해결
- 기울기 소실/폭발 문제 완화
- 문맥 정보를 더 효과적으로 기억

## 구현 내용
이 노트북에서는 PyTorch를 사용하여 TextLSTM 모델을 구현하고 있습니다:

1. 텍스트 데이터셋 준비 및 전처리
2. 단어 임베딩 초기화
3. TextLSTM 모델 구현:
   - 임베딩 레이어
   - LSTM 레이어
   - 출력 레이어
4. 모델 학습 및 자동 완성(autocomplete) 기능 구현

## LSTM의 구조
LSTM은 다음과 같은 주요 구성 요소를 가집니다:
- 셀 상태(Cell State): 장기 기억을 저장
- 입력 게이트(Input Gate): 새로운 정보의 저장 제어
- 망각 게이트(Forget Gate): 기존 정보의 삭제 제어
- 출력 게이트(Output Gate): 정보 출력 제어

## TextLSTM의 장점
- 긴 시퀀스에서도 정보를 효과적으로 기억
- 문맥 이해 능력 향상
- 다양한 NLP 작업에서 우수한 성능

## 응용 분야
- 언어 모델링
- 텍스트 생성
- 자동 완성
- 기계 번역
- 감성 분석
- 질의응답 시스템

## 참고 자료
- 원본 논문: [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- TextLSTM은 현대 NLP와 LLM 시스템의 중요한 구성 요소이며, 트랜스포머 모델이 등장하기 전까지 시퀀스 모델링의 표준으로 사용되었습니다.
