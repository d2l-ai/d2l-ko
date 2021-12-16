# 감정 분석: 컨벌루션 신경망 사용 
:label:`sec_sentiment_cnn`

:numref:`chap_cnn`에서는 인접 픽셀과 같은 로컬 기능에 적용된 2차원 CNN으로 2차원 이미지 데이터를 처리하는 메커니즘을 조사했습니다.원래 컴퓨터 비전용으로 설계되었지만 CNN은 자연어 처리에도 널리 사용됩니다.간단히 말해, 텍스트 시퀀스를 일차원 이미지로 생각하기만 하면 됩니다.이러한 방식으로 1차원 CNN은 텍스트의 $n$그램과 같은 로컬 특징을 처리할 수 있습니다. 

이 단원에서는*TextCNN* 모델을 사용하여 단일 텍스트 :cite:`Kim.2014`를 나타내는 CNN 아키텍처를 설계하는 방법을 보여줍니다.감성 분석을 위해 GLOVE 사전 교육이 포함된 RNN 아키텍처를 사용하는 :numref:`fig_nlp-map-sa-rnn`과 비교할 때 :numref:`fig_nlp-map-sa-cnn`의 유일한 차이점은 아키텍처 선택에 있습니다. 

![This section feeds pretrained GloVe to a CNN-based architecture for sentiment analysis.](../img/nlp-map-sa-cnn.svg)
:label:`fig_nlp-map-sa-cnn`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
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

## 일차원 컨벌루션

모델을 소개하기 전에 1차원 컨벌루션이 어떻게 작동하는지 살펴보겠습니다.이는 교차 상관 연산을 기반으로 하는 2차원 컨벌루션의 특수한 경우일 뿐입니다. 

![One-dimensional cross-correlation operation. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $0\times1+1\times2=2$.](../img/conv1d.svg)
:label:`fig_conv1d`

:numref:`fig_conv1d`에서 볼 수 있듯이 1차원의 경우 컨볼루션 창이 입력 텐서에서 왼쪽에서 오른쪽으로 미끄러집니다.슬라이딩하는 동안, 특정 위치에서 컨볼루션 윈도우에 포함된 입력 서브텐서 (예를 들어, :numref:`fig_conv1d`의 $0$ 및 $1$) 와 커널 텐서 (예를 들어, :numref:`fig_conv1d`의 $1$ 및 $2$) 는 요소별로 곱해진다.이러한 곱셈의 합은 출력 텐서의 해당 위치에서 단일 스칼라 값 (예: :numref:`fig_conv1d`의 $0\times1+1\times2=2$) 을 제공합니다. 

다음 `corr1d` 함수에서 일차원 상호 상관을 구현합니다.입력 텐서 `X`와 커널 텐서 `K`이 주어지면 출력 텐서 `Y`를 반환합니다.

```{.python .input}
#@tab all
def corr1d(X, K):
    w = K.shape[0]
    Y = d2l.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y
```

위의 1 차원 상호 상관 구현의 출력을 검증하기 위해 :numref:`fig_conv1d`에서 입력 텐서 `X`와 커널 텐서 `K`을 구성 할 수 있습니다.

```{.python .input}
#@tab all
X, K = d2l.tensor([0, 1, 2, 3, 4, 5, 6]), d2l.tensor([1, 2])
corr1d(X, K)
```

채널이 여러 개인 1차원 입력의 경우 컨벌루션 커널의 입력 채널 수가 같아야 합니다.그런 다음 각 채널에 대해 입력의 1 차원 텐서와 컨볼 루션 커널의 1 차원 텐서에 대해 교차 상관 연산을 수행하고 모든 채널에 대한 결과를 합산하여 1 차원 출력 텐서를 생성합니다. :numref:`fig_conv1d_channel`는 1 차원 상호 상관 연산을 보여줍니다.3개의 입력 채널이 있습니다. 

![One-dimensional cross-correlation operation with 3 input channels. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $0\times1+1\times2+1\times3+2\times4+2\times(-1)+3\times(-3)=2$.](../img/conv1d-channel.svg)
:label:`fig_conv1d_channel`

여러 입력 채널에 대해 1차원 상호 상관 연산을 구현하고 :numref:`fig_conv1d_channel`에서 결과를 검증할 수 있습니다.

```{.python .input}
#@tab all
def corr1d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return sum(corr1d(x, k) for x, k in zip(X, K))

X = d2l.tensor([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])
K = d2l.tensor([[1, 2], [3, 4], [-1, -3]])
corr1d_multi_in(X, K)
```

다중 입력 채널 1차원 상호 상관은 단일 입력 채널 2차원 상호 상관과 동일합니다.예를 들어, :numref:`fig_conv1d_channel`에서 다중 입력 채널 1차원 상호 상관의 등가 형태는 :numref:`fig_conv1d_2d`의 단일 입력 채널 2차원 상호 상관이며, 여기서 컨볼루션 커널의 높이는 입력 텐서의 높이와 동일해야 합니다. 

![Two-dimensional cross-correlation operation with a single input channel. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $2\times(-1)+3\times(-3)+1\times3+2\times4+0\times1+1\times2=2$.](../img/conv1d-2d.svg)
:label:`fig_conv1d_2d`

:numref:`fig_conv1d`와 :numref:`fig_conv1d_channel`의 출력에는 모두 하나의 채널만 있습니다.:numref:`subsec_multi-output-channels`에 설명된 여러 출력 채널이 있는 2차원 컨벌루션과 마찬가지로 1차원 컨벌루션에 대해 여러 출력 채널을 지정할 수도 있습니다. 

## 최대 초과 시간 풀링

마찬가지로 풀링을 사용하여 시간 단계에서 가장 중요한 기능으로 시퀀스 표현에서 가장 높은 값을 추출 할 수 있습니다.TextCNN에 사용되는*시간별 최대값 풀링*은 1차원 전역 최대 풀링 :cite:`Collobert.Weston.Bottou.ea.2011`와 같이 작동합니다.각 채널이 서로 다른 시간 스텝의 값을 저장하는 다중 채널 입력의 경우 각 채널의 출력은 해당 채널의 최대값입니다.시간대별 최댓값 풀링은 채널마다 다른 수의 시간 스텝을 허용합니다. 

## 텍스트/CNN 모델

TextCNN 모델은 1차원 컨벌루션과 시간대별 최댓값 풀링을 사용하여 사전 훈련된 개별 토큰 표현을 입력값으로 취한 다음 다운스트림 응용 프로그램에 대한 시퀀스 표현을 가져오고 변환합니다. 

$d$차원 벡터로 표시되는 $n$ 토큰이 있는 단일 텍스트 시퀀스의 경우 입력 텐서의 너비, 높이 및 채널 수는 각각 $n$, $1$ 및 $d$입니다.TextCNN 모델은 다음과 같이 입력을 출력으로 변환합니다. 

1. 여러 1차원 컨벌루션 커널을 정의하고 입력값에 대해 개별적으로 컨벌루션 연산을 수행합니다.폭이 다른 컨볼루션 커널은 인접한 토큰의 개수가 서로 다른 로컬 특징을 캡처할 수 있습니다.
1. 모든 출력 채널에 대해 시간별로 최대로 풀링을 수행한 다음 모든 스칼라 풀링 출력값을 벡터로 결합합니다.
1. 완전 연결 계층을 사용하여 연결된 벡터를 출력 범주로 변환합니다.드롭아웃은 과적합을 줄이기 위해 사용할 수 있습니다.

![The model architecture of textCNN.](../img/textcnn.svg)
:label:`fig_conv1d_textcnn`

:numref:`fig_conv1d_textcnn`는 텍스트CNN의 모델 아키텍처를 구체적인 예와 함께 보여줍니다.입력값은 11개의 토큰이 있는 문장으로, 각 토큰은 6차원 벡터로 표시됩니다.따라서 너비가 11인 6채널 입력이 있습니다.너비가 2와 4인 두 개의 1차원 컨벌루션 커널을 각각 4와 5개의 출력 채널로 정의합니다.이 제품은 너비가 $11-2+1=10$인 4개의 출력 채널과 너비가 $11-4+1=8$인 5개의 출력 채널을 생성합니다.이러한 9개 채널의 너비는 다르지만, 시간대별 최댓값 풀링은 연결된 9차원 벡터를 제공하며, 이 벡터는 최종적으로 이진 센티멘트 예측을 위한 2차원 출력 벡터로 변환됩니다. 

### 모델 정의

다음 클래스에서 TextCNN 모델을 구현합니다.:numref:`sec_sentiment_rnn`의 양방향 RNN 모델과 비교할 때 순환 계층을 컨벌루션 계층으로 대체하는 것 외에도 두 개의 임베딩 계층을 사용합니다. 하나는 훈련 가능한 가중치가 있고 다른 하나는 고정 가중치를 사용합니다.

```{.python .input}
class TextCNN(nn.Block):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # The embedding layer not to be trained
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # The max-over-time pooling layer has no parameters, so this instance
        # can be shared
        self.pool = nn.GlobalMaxPool1D()
        # Create multiple one-dimensional convolutional layers
        self.convs = nn.Sequential()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # Concatenate two embedding layer outputs with shape (batch size, no.
        # of tokens, token vector dimension) along vectors
        embeddings = np.concatenate((
            self.embedding(inputs), self.constant_embedding(inputs)), axis=2)
        # Per the input format of one-dimensional convolutional layers,
        # rearrange the tensor so that the second dimension stores channels
        embeddings = embeddings.transpose(0, 2, 1)
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, a tensor of shape (batch size, no. of channels, 1) is
        # obtained. Remove the last dimension and concatenate along channels
        encoding = np.concatenate([
            np.squeeze(self.pool(conv(embeddings)), axis=-1)
            for conv in self.convs], axis=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

```{.python .input}
#@tab pytorch
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # The embedding layer not to be trained
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # The max-over-time pooling layer has no parameters, so this instance
        # can be shared
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # Create multiple one-dimensional convolutional layers
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # Concatenate two embedding layer outputs with shape (batch size, no.
        # of tokens, token vector dimension) along vectors
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # Per the input format of one-dimensional convolutional layers,
        # rearrange the tensor so that the second dimension stores channels
        embeddings = embeddings.permute(0, 2, 1)
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, a tensor of shape (batch size, no. of channels, 1) is
        # obtained. Remove the last dimension and concatenate along channels
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

우리가 TextCNN 인스턴스를 만들 수 있습니다.커널 너비가 3, 4, 5인 3개의 컨벌루션 계층이 있으며 모두 100개의 출력 채널이 있습니다.

```{.python .input}
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)

def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights);
```

### 사전 훈련된 워드 벡터 불러오기

:numref:`sec_sentiment_rnn`와 마찬가지로 사전 훈련된 100차원 GLOVE 임베딩을 초기화된 토큰 표현으로 로드합니다.이러한 토큰 표현 (가중치 포함) 은 `embedding`에서 교육되고 `constant_embedding`에서 수정됩니다.

```{.python .input}
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
net.constant_embedding.weight.set_data(embeds)
net.constant_embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad = False
```

### 모델 훈련 및 평가

이제 감성 분석을 위해 TextCNN 모델을 훈련시킬 수 있습니다.

```{.python .input}
lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

아래에서는 훈련된 모델을 사용하여 두 개의 간단한 문장에 대한 감정을 예측합니다.

```{.python .input}
#@tab all
d2l.predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
d2l.predict_sentiment(net, vocab, 'this movie is so bad')
```

## 요약

* 1차원 CNN은 텍스트에서 $n$그램과 같은 로컬 피쳐를 처리할 수 있습니다.
* 다중 입력 채널 1차원 상호 상관은 단일 입력 채널 2차원 상호 상관과 동일합니다.
* 시간대별 풀링을 사용하면 채널마다 다른 수의 시간 스텝을 사용할 수 있습니다.
* TextCNN 모델은 1차원 컨벌루션 계층과 시간대별 최대 풀링 계층을 사용하여 개별 토큰 표현을 다운스트림 응용 프로그램 출력으로 변환합니다.

## 연습문제

1. 분류 정확도 및 계산 효율성과 같은 :numref:`sec_sentiment_rnn`와 이 섹션의 감정 분석을 위해 하이퍼파라미터를 조정하고 두 아키텍처를 비교합니다.
1. :numref:`sec_sentiment_rnn`의 연습에 소개된 방법을 사용하여 모형의 분류 정확도를 더욱 향상시킬 수 있습니까?
1. 입력 표현에 위치 인코딩을 추가합니다.분류 정확도가 향상됩니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/393)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1425)
:end_tab:
