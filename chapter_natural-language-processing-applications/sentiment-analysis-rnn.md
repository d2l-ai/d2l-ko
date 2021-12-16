# 감성 분석: 순환 신경망 사용
:label:`sec_sentiment_rnn`

단어 유사성 및 유추 작업과 마찬가지로 사전 훈련된 단어 벡터를 감정 분석에 적용할 수도 있습니다.:numref:`sec_sentiment`의 IMDb 검토 데이터 세트는 그다지 크지 않기 때문에 대규모 상체에 대해 사전 학습된 텍스트 표현을 사용하면 모델의 과적합을 줄일 수 있습니다.:numref:`fig_nlp-map-sa-rnn`에 설명된 구체적인 예로서, 사전 훈련된 GLOVE 모델을 사용하여 각 토큰을 나타내고, 이러한 토큰 표현을 다층 양방향 RNN에 공급하여 텍스트 시퀀스 표현을 얻습니다. 텍스트 시퀀스 표현은 감정 분석 출력 :cite:`Maas.Daly.Pham.ea.2011`으로 변환됩니다.동일한 다운스트림 응용 프로그램의 경우 나중에 다른 아키텍처 선택을 고려할 것입니다. 

![This section feeds pretrained GloVe to an RNN-based architecture for sentiment analysis.](../img/nlp-map-sa-rnn.svg)
:label:`fig_nlp-map-sa-rnn`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

## RNN으로 단일 텍스트 표현하기

감성 분석과 같은 텍스트 분류 작업에서는 다양한 길이의 텍스트 시퀀스가 고정 길이 범주로 변환됩니다.다음 `BiRNN` 클래스에서, 텍스트 시퀀스의 각 토큰은 임베딩 계층 (`self.embedding`) 을 통해 개별 사전 훈련된 GloVE 표현을 가져오는 반면, 전체 시퀀스는 양방향 RNN (`self.encoder`) 에 의해 인코딩된다.보다 구체적으로, 초기 및 최종 시간 단계 모두에서 양방향 LSTM의 숨겨진 상태 (마지막 계층에서) 는 텍스트 시퀀스의 표현으로 연결됩니다.이 단일 텍스트 표현은 두 개의 출력 (“양수” 및 “음수”) 이 있는 완전 연결 계층 (`self.decoder`) 에 의해 출력 범주로 변환됩니다.

```{.python .input}
class BiRNN(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional RNN
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # The shape of `inputs` is (batch size, no. of time steps). Because
        # LSTM requires its input's first dimension to be the temporal
        # dimension, the input is transposed before obtaining token
        # representations. The output shape is (no. of time steps, batch size,
        # word vector dimension)
        embeddings = self.embedding(inputs.T)
        # Returns hidden states of the last hidden layer at different time
        # steps. The shape of `outputs` is (no. of time steps, batch size,
        # 2 * no. of hidden units)
        outputs = self.encoder(embeddings)
        # Concatenate the hidden states at the initial and final time steps as
        # the input of the fully-connected layer. Its shape is (batch size,
        # 4 * no. of hidden units)
        encoding = np.concatenate((outputs[0], outputs[-1]), axis=1)
        outs = self.decoder(encoding)
        return outs
```

```{.python .input}
#@tab pytorch
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional RNN
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # The shape of `inputs` is (batch size, no. of time steps). Because
        # LSTM requires its input's first dimension to be the temporal
        # dimension, the input is transposed before obtaining token
        # representations. The output shape is (no. of time steps, batch size,
        # word vector dimension)
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # Returns hidden states of the last hidden layer at different time
        # steps. The shape of `outputs` is (no. of time steps, batch size,
        # 2 * no. of hidden units)
        outputs, _ = self.encoder(embeddings)
        # Concatenate the hidden states of the initial time step and final
        # time step to use as the input of the fully connected layer. Its
        # shape is (batch size, 4 * no. of hidden units)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        # Concatenate the hidden states at the initial and final time steps as
        # the input of the fully-connected layer. Its shape is (batch size,
        # 4 * no. of hidden units)
        outs = self.decoder(encoding)
        return outs
```

감성 분석을 위해 단일 텍스트를 나타내는 두 개의 숨겨진 레이어로 양방향 RNN을 구성해 보겠습니다.

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, devices = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
```

```{.python .input}
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
net.apply(init_weights);
```

## 사전 훈련된 워드 벡터 불러오기

아래에서는 어휘의 토큰에 대한 사전 훈련된 100차원 (`embed_size`와 일치해야 함) GLOVE 임베딩을 로드합니다.

```{.python .input}
#@tab all
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
```

어휘의 모든 토큰에 대한 벡터의 모양을 인쇄합니다.

```{.python .input}
#@tab all
embeds = glove_embedding[vocab.idx_to_token]
embeds.shape
```

이러한 사전 훈련된 단어 벡터를 사용하여 리뷰에서 토큰을 나타내며 훈련 중에는 이러한 벡터를 업데이트하지 않습니다.

```{.python .input}
net.embedding.weight.set_data(embeds)
net.embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False
```

## 모델 훈련 및 평가

이제 감정 분석을 위해 양방향 RNN을 훈련시킬 수 있습니다.

```{.python .input}
lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

훈련된 모델 `net`를 사용하여 텍스트 시퀀스의 감정을 예측하기 위해 다음 함수를 정의합니다.

```{.python .input}
#@save
def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence."""
    sequence = np.array(vocab[sequence.split()], ctx=d2l.try_gpu())
    label = np.argmax(net(sequence.reshape(1, -1)), axis=1)
    return 'positive' if label == 1 else 'negative'
```

```{.python .input}
#@tab pytorch
#@save
def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence."""
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'
```

마지막으로 훈련된 모델을 사용하여 두 개의 간단한 문장에 대한 감정을 예측해 보겠습니다.

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so bad')
```

## 요약

* 사전 훈련된 워드 벡터는 텍스트 시퀀스의 개별 토큰을 나타낼 수 있습니다.
* 양방향 RNN은 초기 및 최종 시간 단계에서 숨겨진 상태의 연결을 통해 텍스트 시퀀스를 나타낼 수 있습니다.이 단일 텍스트 표현은 완전히 연결된 레이어를 사용하여 범주로 변환할 수 있습니다.

## 연습문제

1. 에포크 수를 늘립니다.교육 및 테스트 정확도를 개선할 수 있습니까?다른 하이퍼파라미터를 튜닝하는 것은 어떨까요?
1. 300차원 GLOVE 임베딩과 같이 더 큰 사전 훈련된 워드 벡터를 사용합니다.분류 정확도가 향상됩니까?
1. SpACy 토큰화를 사용하여 분류 정확도를 개선할 수 있습니까?스페이시 (`pip install spacy`) 를 설치하고 영어 패키지 (`python -m spacy download en`) 를 설치해야 합니다.코드에서 먼저 스페이시 (`import spacy`) 를 가져옵니다.그런 다음 스페이시 영어 패키지 (`spacy_en = spacy.load('en')`) 를 로드합니다.마지막으로 함수 `def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]`을 정의하고 원래 `tokenizer` 함수를 바꿉니다.GLOve와 Spacy에서 다양한 형태의 구문 토큰에 주목하십시오.예를 들어, 토큰 “뉴욕”이라는 문구는 글로브에서 “뉴욕”의 형태를 취하고 SpAcy 토큰화 후에는 “뉴욕”의 형태를 취합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/392)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1424)
:end_tab:
