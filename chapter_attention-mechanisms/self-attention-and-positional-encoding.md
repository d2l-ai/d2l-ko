# 자기주의 및 위치 인코딩
:label:`sec_self-attention-and-positional-encoding`

딥러닝에서는 종종 CNN 또는 RNN을 사용하여 시퀀스를 인코딩합니다.이제 주의 메커니즘을 사용하여 동일한 토큰 집합이 쿼리, 키 및 값으로 작동하도록 일련의 토큰을 관심 풀링에 공급한다고 상상해보십시오.특히 각 쿼리는 모든 키-값 쌍을 처리하고 하나의 주의 출력을 생성합니다.쿼리, 키 및 값이 동일한 위치에서 제공되므로 이 작업이 수행됩니다.
*자기 주의력* :cite:`Lin.Feng.Santos.ea.2017,Vaswani.Shazeer.Parmar.ea.2017`는 *주의력* :cite:`Cheng.Dong.Lapata.2016,Parikh.Tackstrom.Das.ea.2016,Paulus.Xiong.Socher.2017`라고도 합니다.
이 섹션에서는 시퀀스 순서에 대한 추가 정보 사용을 포함하여 자체 주의를 사용한 시퀀스 인코딩에 대해 설명합니다.

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

## [**자기 주의**]

$\mathbf{x}_i \in \mathbb{R}^d$ ($1 \leq i \leq n$) 이있는 입력 토큰 $\mathbf{x}_1, \ldots, \mathbf{x}_n$의 시퀀스가 주어지면 자체 주의는 동일한 길이 $\mathbf{y}_1, \ldots, \mathbf{y}_n$의 시퀀스를 출력합니다. 

$$\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d$$

:eqref:`eq_attn-pooling`에서 주의력 풀링 $f$의 정의에 따르면.다음 코드 스 니펫은 다중 헤드 주의를 사용하여 모양이 있는 텐서의 자체 주의를 계산합니다 (배치 크기, 시간 단계 수 또는 토큰의 시퀀스 길이, $d$).출력 텐서의 모양은 같습니다.

```{.python .input}
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
#@tab pytorch
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
attention.eval()
```

```{.python .input}
#@tab tensorflow
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
```

```{.python .input}
#@tab mxnet, pytorch
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
attention(X, X, X, valid_lens).shape
```

```{.python .input}
#@tab tensorflow
batch_size, num_queries, valid_lens = 2, 4, tf.constant([3, 2])
X = tf.ones((batch_size, num_queries, num_hiddens))
attention(X, X, X, valid_lens, training=False).shape
```

## CNN, RNN 및 자기 주의 비교
:label:`subsec_cnn-rnn-self-attention`

$n$ 토큰의 시퀀스를 동일한 길이의 다른 시퀀스에 매핑하는 아키텍처를 비교해 보겠습니다. 여기서 각 입력 또는 출력 토큰은 $d$ 차원 벡터로 표시됩니다.구체적으로 CNN, RNN 및 자기 관심을 고려할 것입니다.계산 복잡성, 순차 연산 및 최대 경로 길이를 비교합니다.순차 연산은 병렬 계산을 방해하는 반면, 시퀀스 위치 조합 간의 경로가 짧으면 시퀀스 :cite:`Hochreiter.Bengio.Frasconi.ea.2001` 내에서 장거리 종속성을 더 쉽게 학습할 수 있습니다. 

![Comparing CNN (padding tokens are omitted), RNN, and self-attention architectures.](../img/cnn-rnn-self-attention.svg)
:label:`fig_cnn-rnn-self-attention`

커널 크기가 $k$인 컨벌루션 계층을 가정해 보겠습니다.CNN을 사용한 시퀀스 처리에 대한 자세한 내용은 다음 장에서 제공합니다.지금은 시퀀스 길이가 $n$이므로 입력 및 출력 채널의 수가 모두 $d$이고 컨벌루션 계층의 계산 복잡도는 $\mathcal{O}(knd^2)$이라는 것을 알아야합니다.:numref:`fig_cnn-rnn-self-attention`에서 볼 수 있듯이 CNN은 계층적이므로 $\mathcal{O}(1)$의 순차적 작업이 있으며 최대 경로 길이는 $\mathcal{O}(n/k)$입니다.예를 들어, $\mathbf{x}_1$ 및 $\mathbf{x}_5$는 :numref:`fig_cnn-rnn-self-attention`에서 커널 크기가 3인 2계층 CNN의 수용 필드 내에 있습니다. 

RNN의 은닉 상태를 업데이트할 때 $d \times d$ 가중치 행렬과 $d$차원 은닉 상태의 곱셈은 $\mathcal{O}(d^2)$의 계산 복잡도를 갖습니다.시퀀스 길이가 $n$이므로 순환 계층의 계산 복잡도는 $\mathcal{O}(nd^2)$입니다.:numref:`fig_cnn-rnn-self-attention`에 따르면 병렬화할 수 없는 $\mathcal{O}(n)$개의 순차 작업이 있으며 최대 경로 길이도 $\mathcal{O}(n)$입니다. 

자체 주의에서 쿼리, 키 및 값은 모두 $n \times d$ 행렬입니다.:eqref:`eq_softmax_QK_V`에서 스케일링된 내적 주의를 고려합니다. 여기서 $n \times d$ 행렬에 $d \times n$ 행렬을 곱한 다음 출력 $n \times n$ 행렬에 $n \times d$ 행렬을 곱합니다.결과적으로 자기주의는 $\mathcal{O}(n^2d)$의 계산 복잡성을 갖습니다.:numref:`fig_cnn-rnn-self-attention`에서 볼 수 있듯이 각 토큰은 자체 주의를 통해 다른 토큰에 직접 연결됩니다.따라서 계산은 $\mathcal{O}(1)$ 순차 연산과 병렬일 수 있으며 최대 경로 길이도 $\mathcal{O}(1)$입니다. 

대체로 CNN과 자주의는 모두 병렬 계산을 즐기고 자기주의는 최대 경로 길이가 가장 짧습니다.그러나 시퀀스 길이에 대한 2차 계산 복잡성으로 인해 매우 긴 시퀀스의 경우 자체 주의가 엄청나게 느려집니다. 

## [**위치 인코딩**]
:label:`subsec_positional-encoding`

시퀀스의 토큰을 하나씩 반복적으로 처리하는 RNN과 달리 자체 주의는 병렬 계산을 위해 순차 작업을 버립니다.시퀀스 순서 정보를 사용하기 위해 입력 표현에*위치 인코딩*을 추가하여 절대 또는 상대 위치 정보를 주입 할 수 있습니다.위치 인코딩은 학습하거나 고정할 수 있습니다.다음에서는 사인 및 코사인 함수 :cite:`Vaswani.Shazeer.Parmar.ea.2017`를 기반으로 한 고정 위치 인코딩에 대해 설명합니다. 

입력 표현 $\mathbf{X} \in \mathbb{R}^{n \times d}$에 시퀀스의 $n$ 토큰에 대한 $d$차원 임베딩이 포함되어 있다고 가정합니다.위치 인코딩은 동일한 형상의 위치 임베딩 행렬 ($\mathbf{P} \in \mathbb{R}^{n \times d}$) 을 사용하여 $\mathbf{X} + \mathbf{P}$를 출력하며, 이 행렬의 요소는 $i^\mathrm{th}$ 행 및 $(2j)^\mathrm{th}$ 또는 $(2j + 1)^\mathrm{th}$ 열에 있는 요소이다. 

$$\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right),\\p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right).\end{aligned}$$
:eqlabel:`eq_positional-encoding-def`

언뜻보기에는 이 삼각 함수 디자인이 이상해 보입니다.이 설계에 대해 설명하기 전에 먼저 다음 `PositionalEncoding` 클래스에서 구현하겠습니다.

```{.python .input}
#@save
class PositionalEncoding(nn.Block):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_ctx(X.ctx)
        return self.dropout(X)
```

```{.python .input}
#@tab pytorch
#@save
class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
```

```{.python .input}
#@tab tensorflow
#@save
class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        # Create a long enough `P`
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, num_hiddens, 2, dtype=np.float32) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)
        
    def call(self, X, **kwargs):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X, **kwargs)
```

위치 임베딩 행렬 $\mathbf{P}$에서 [**행은 시퀀스 내의 위치에 대응하고 열은 서로 다른 위치 인코딩 차원을 나타냅니다**].아래 예에서 위치 임베딩 행렬의 $6^{\mathrm{th}}$ 및 $7^{\mathrm{th}}$ 열이 $8^{\mathrm{th}}$ 및 $9^{\mathrm{th}}$ 열보다 주파수가 더 높다는 것을 알 수 있습니다.$6^{\mathrm{th}}$과 $7^{\mathrm{th}}$ ($8^{\mathrm{th}}$과 $9^{\mathrm{th}}$의 경우와 동일) 열 사이의 오프셋은 사인 함수와 코사인 함수의 교대에 기인합니다.

```{.python .input}
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.initialize()
X = pos_encoding(np.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
#@tab pytorch
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(d2l.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
#@tab tensorflow
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(tf.zeros((1, num_steps, encoding_dim)), training=False)
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(np.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in np.arange(6, 10)])
```

### 절대 위치 정보

인코딩 차원을 따라 단조롭게 감소된 주파수가 절대 위치 정보와 어떻게 관련되는지 확인하기 위해 $0, 1, \ldots, 7$의 [**이진 표현**] 을 인쇄해 보겠습니다.보시다시피 가장 낮은 비트, 두 번째로 낮은 비트 및 세 번째로 낮은 비트는 각각 모든 숫자, 두 숫자 및 네 개의 숫자에서 번갈아 나타납니다.

```{.python .input}
#@tab all
for i in range(8):
    print(f'{i} in binary is {i:>03b}')
```

이진 표현에서 비트가 높을수록 낮은 비트보다 주파수가 낮습니다.마찬가지로 아래 히트 맵에서 볼 수 있듯이 [**위치 인코딩은 삼각 함수를 사용하여 인코딩 차원을 따라 주파수를 감소**] 합니다.출력값이 부동 소수이기 때문에 이러한 연속 표현은 이진 표현보다 공간 효율적입니다.

```{.python .input}
P = np.expand_dims(np.expand_dims(P[0, :, :], 0), 0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
#@tab pytorch
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
#@tab tensorflow
P = tf.expand_dims(tf.expand_dims(P[0, :, :], axis=0), axis=0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

### 상대적 위치 정보

위의 위치 인코딩을 사용하면 절대 위치 정보를 캡처하는 것 외에도 모델이 상대 위치별로 참석하는 방법을 쉽게 배울 수 있습니다.이는 고정 위치 오프셋 $\delta$에 대해 위치 $i + \delta$에서의 위치 인코딩이 위치 $i$에서의 선형 투영으로 표현될 수 있기 때문입니다. 

이 투영은 수학적으로 설명할 수 있습니다.$\omega_j = 1/10000^{2j/d}$를 나타내는 :eqref:`eq_positional-encoding-def`의 $(p_{i, 2j}, p_{i, 2j+1})$의 모든 쌍은 고정 오프셋 $\delta$에 대해 $(p_{i+\delta, 2j}, p_{i+\delta, 2j+1})$로 선형으로 투영될 수 있습니다. 

$$\begin{aligned}
&\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}\\
=&\begin{bmatrix} \cos(\delta \omega_j) \sin(i \omega_j) + \sin(\delta \omega_j) \cos(i \omega_j) \\  -\sin(\delta \omega_j) \sin(i \omega_j) + \cos(\delta \omega_j) \cos(i \omega_j) \\ \end{bmatrix}\\
=&\begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix}\\
=& 
\begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix},
\end{aligned}$$

여기서 $2\times 2$ 투영 행렬은 위치 지수 $i$에 의존하지 않습니다. 

## 요약

* 자기 주의에서 쿼리, 키 및 값은 모두 같은 위치에서 가져옵니다.
* CNN과 자기 관심 모두 병렬 계산을 즐기고 자기주의는 최대 경로 길이가 가장 짧습니다.그러나 시퀀스 길이에 대한 2차 계산 복잡성으로 인해 매우 긴 시퀀스의 경우 자체 주의가 엄청나게 느려집니다.
* 시퀀스 순서 정보를 사용하기 위해 입력 표현에 위치 인코딩을 추가하여 절대 또는 상대 위치 정보를 삽입 할 수 있습니다.

## 연습문제

1. 위치 인코딩으로 자기주의 계층을 쌓아서 시퀀스를 나타내는 심층 아키텍처를 설계한다고 가정합니다.어떤 문제가 될 수 있을까요?
1. 학습 가능한 위치 인코딩 방법을 설계할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1651)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1652)
:end_tab:
