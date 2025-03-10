# NLP Tutorial 문서화

## 개요
이 저장소는 PyTorch를 사용한 자연어 처리(NLP) 및 언어 모델(LLM) 튜토리얼 모음입니다. 기본적인 임베딩 모델부터 트랜스포머 기반 모델까지 NLP의 주요 모델들이 100줄 이내의 코드로 구현되어 있습니다.

## 구조
이 저장소는 다음과 같은 커리큘럼 구조로 구성되어 있습니다:

### 1. 기본 임베딩 모델
- **NNLM (Neural Network Language Model)**: 다음 단어 예측
- **Word2Vec (Skip-gram)**: 단어 임베딩 및 시각화
- **FastText**: 문장 분류

### 2. CNN (Convolutional Neural Network)
- **TextCNN**: 이진 감성 분류

### 3. RNN (Recurrent Neural Network)
- **TextRNN**: 다음 단계 예측
- **TextLSTM**: 자동 완성
- **Bi-LSTM**: 긴 문장에서 다음 단어 예측

### 4. 어텐션 메커니즘
- **Seq2Seq**: 단어 변환
- **Seq2Seq with Attention**: 번역
- **Bi-LSTM with Attention**: 이진 감성 분류

### 5. 트랜스포머 기반 모델
- **Transformer**: 번역
- **BERT**: 다음 문장 분류 및 마스킹된 토큰 예측

## 각 모델별 문서화 파일
각 Jupyter 노트북에 대한 자세한 설명은 다음 문서에서 확인할 수 있습니다:

1. [NNLM 문서](/1-1.NNLM/NNLM_documentation.md)
2. [Word2Vec 문서](/1-2.Word2Vec/Word2Vec_documentation.md)
3. [FastText 문서](/1-3.FastText/FastText_documentation.md)
4. [TextCNN 문서](/2-1.TextCNN/TextCNN_documentation.md)
5. [TextRNN 문서](/3-1.TextRNN/TextRNN_documentation.md)
6. [TextLSTM 문서](/3-2.TextLSTM/TextLSTM_documentation.md)
7. [Bi-LSTM 문서](/3-3.Bi-LSTM/Bi-LSTM_documentation.md)
8. [Seq2Seq 문서](/4-1.Seq2Seq/Seq2Seq_documentation.md)
9. [Seq2Seq with Attention 문서](/4-2.Seq2Seq(Attention)/Seq2Seq_Attention_documentation.md)
10. [Bi-LSTM with Attention 문서](/4-3.Bi-LSTM(Attention)/Bi-LSTM_Attention_documentation.md)
11. [Transformer 문서](/5-1.Transformer/Transformer_documentation.md)
12. [BERT 문서](/5-2.BERT/BERT_documentation.md)

## NLP와 LLM의 발전
이 저장소의 모델들은 NLP와 LLM 발전의 주요 이정표를 보여줍니다:

1. **초기 신경망 기반 언어 모델**: NNLM, Word2Vec, FastText
2. **CNN 기반 텍스트 처리**: TextCNN
3. **RNN 기반 시퀀스 모델링**: TextRNN, TextLSTM, Bi-LSTM
4. **어텐션 메커니즘의 도입**: Seq2Seq with Attention, Bi-LSTM with Attention
5. **트랜스포머 아키텍처**: Transformer, BERT

이러한 발전은 현대 대규모 언어 모델(LLM)의 기초가 되었으며, GPT, T5, LLaMA 등 최신 모델들도 이러한 기본 개념들을 확장하고 있습니다.

## 의존성
- Python 3.5+
- PyTorch 1.0.0+

## 참고 자료
이 저장소의 각 모델은 해당 분야의 중요한 논문들을 기반으로 구현되었습니다. 각 모델별 문서에서 원본 논문 링크를 확인할 수 있습니다.
