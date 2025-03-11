# Seq2Seq (Sequence-to-Sequence)

## 개요
Seq2Seq는 2014년에 발표된 "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" 논문에서 소개된 모델로, 한 시퀀스를 다른 시퀀스로 변환하는 작업에 사용됩니다. 주로 기계 번역, 요약, 대화 생성 등에 활용됩니다.

## 주요 특징
- 인코더-디코더 구조를 통한 시퀀스 간 변환
- 가변 길이 입력을 가변 길이 출력으로 변환
- 인코더에서 입력 시퀀스의 정보를 압축
- 디코더에서 압축된 정보를 기반으로 출력 생성

## 모델 구조 상세
Seq2Seq은 다음과 같은 구조로 이루어져 있습니다:

1. **인코더(Encoder)**:
   - 입력 시퀀스를 처리하는 RNN/LSTM/GRU
   - 각 입력 토큰을 순차적으로 처리
   - 전체 입력 시퀀스의 정보를 컨텍스트 벡터(context vector)로 압축
   - 일반적으로 마지막 은닉 상태를 컨텍스트 벡터로 사용

2. **디코더(Decoder)**:
   - 출력 시퀀스를 생성하는 RNN/LSTM/GRU
   - 인코더의 컨텍스트 벡터를 초기 은닉 상태로 사용
   - 자기회귀적(autoregressive) 방식으로 토큰 생성
   - 이전 시점의 출력을 다음 시점의 입력으로 사용
   - 특수 토큰 

## 수식
인코더와 디코더의 수식은 다음과 같습니다:

- 인코더:
  - $h_t = f(x_t, h_{t-1})$
  - $c = h_T$

- 디코더:
  - $s_t = f(y_{t-1}, s_{t-1}, c)$
  - $y_t = g(s_t)$

## 하이퍼파라미터
Seq2Seq 모델의 하이퍼파라미터는 다음과 같습니다:

- 인코더와 디코더의 은닉 상태 크기
- 인코더와 디코더의 층 수
- 배치 크기
- 에포크 수
- 학습률

## 계산 복잡도
Seq2Seq 모델의 계산 복잡도는 다음과 같습니다:

- 인코더: $O(T \times H \times D)$
- 디코더: $O(T \times H \times D)$

## 참고문헌
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. Advances in neural information processing systems, 27.

## 예시 코드
Seq2Seq 모델을 구현하는 예시 코드는 다음과 같습니다:
```python
import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        c0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)

        out, _ = self.rnn(x, (h0, c0))
        return out[:, -1, :]

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, context):
        h0 = context.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
        c0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)

        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 모델 초기화
encoder = Encoder(input_dim=10, hidden_dim=20, num_layers=1)
decoder = Decoder(input_dim=10, hidden_dim=20, output_dim=10, num_layers=1)

# 입력 데이터
input_data = torch.randn(1, 10, 10)

# 인코더와 디코더를 통한 출력
context = encoder(input_data)
output = decoder(torch.randn(1, 10, 10), context)

print(output.shape)
