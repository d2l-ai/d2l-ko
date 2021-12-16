# 멀티 헤드 주의
:label:`sec_multihead-attention`

실제로 동일한 쿼리, 키 및 값 집합이 주어지면 모델이 시퀀스 내에서 다양한 범위 (예: 단거리 대 장거리) 의 종속성을 캡처하는 것과 같은 동일한 주의 메커니즘의 다양한 동작에서 얻은 지식을 결합하기를 원할 수 있습니다.따라서 주의 메커니즘이 쿼리, 키 및 값의 서로 다른 표현 하위 공간을 공동으로 사용할 수 있도록 하는 것이 유용할 수 있습니다. 

이를 위해 단일 주의 풀링을 수행하는 대신 $h$ 독립적으로 학습된 선형 투영을 사용하여 쿼리, 키 및 값을 변환할 수 있습니다.그런 다음 $h$개의 프로젝션된 쿼리, 키 및 값이 주의력 풀링에 병렬로 공급됩니다.결국 $h$ 주의 풀링 출력은 다른 학습된 선형 투영과 연결되고 변환되어 최종 출력을 생성합니다.이 디자인을 *멀티 헤드 어텐션*이라고 하며, 여기서 $h$ 어텐션 풀링 출력은 각각*헤드* :cite:`Vaswani.Shazeer.Parmar.ea.2017`입니다.완전 연결 계층을 사용하여 학습 가능한 선형 변환을 수행하는 :numref:`fig_multi-head-attention`는 다중 헤드 관심을 설명합니다. 

![Multi-head attention, where multiple heads are concatenated then linearly transformed.](../img/multi-head-attention.svg)
:label:`fig_multi-head-attention`

## 모델

멀티 헤드 관심의 구현을 제공하기 전에이 모델을 수학적으로 공식화하겠습니다.쿼리 $\mathbf{q} \in \mathbb{R}^{d_q}$, 키 $\mathbf{k} \in \mathbb{R}^{d_k}$ 및 값 $\mathbf{v} \in \mathbb{R}^{d_v}$이 주어지면 각 주의 헤드 $\mathbf{h}_i$ ($i = 1, \ldots, h$) 는 다음과 같이 계산됩니다. 

$$\mathbf{h}_i = f(\mathbf W_i^{(q)}\mathbf q, \mathbf W_i^{(k)}\mathbf k,\mathbf W_i^{(v)}\mathbf v) \in \mathbb R^{p_v},$$

여기서 학습 가능한 매개변수 $\mathbf W_i^{(q)}\in\mathbb R^{p_q\times d_q}$, $\mathbf W_i^{(k)}\in\mathbb R^{p_k\times d_k}$ 및 $\mathbf W_i^{(v)}\in\mathbb R^{p_v\times d_v}$ 및 $f$는 :numref:`sec_attention-scoring-functions`에서 가산 주의 및 스케일링된 점 곱 주의와 같은 주의력 풀링입니다.멀티 헤드 주의 출력은 $h$ 헤드 연결의 학습 가능한 파라미터 $\mathbf W_o\in\mathbb R^{p_o\times h p_v}$을 통한 또 다른 선형 변환입니다. 

$$\mathbf W_o \begin{bmatrix}\mathbf h_1\\\vdots\\\mathbf h_h\end{bmatrix} \in \mathbb{R}^{p_o}.$$

이 설계에 따라 각 헤드는 입력의 다른 부분에 참석할 수 있습니다.단순 가중 평균보다 더 정교한 함수를 표현할 수 있습니다.

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
import tensorflow as tf
```

## 구현

구현에서 우리는 멀티 헤드 주의의 [**각 머리에 대해 스케일링된 도트 제품 주의를 선택합니다**].계산 비용과 매개 변수화 비용의 상당한 증가를 피하기 위해 $p_q = p_k = p_v = p_o / h$을 설정했습니다.쿼리, 키 및 값에 대한 선형 변환의 출력 수를 $p_q h = p_k h = p_v h = p_o$로 설정하면 $h$ 헤드를 병렬로 계산할 수 있습니다.다음 구현에서 $p_o$은 인수 `num_hiddens`를 통해 지정됩니다.

```{.python .input}
#@save
class MultiHeadAttention(nn.Block):
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, dropout, use_bias=False,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_k = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_v = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_o = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = valid_lens.repeat(self.num_heads, axis=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens)
        
        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
```

```{.python .input}
#@tab pytorch
#@save
class MultiHeadAttention(nn.Module):
    """Multi-head attention."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
```

```{.python .input}
#@tab tensorflow
#@save
class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_o = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
    
    def call(self, queries, keys, values, valid_lens, **kwargs):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        
        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = tf.repeat(valid_lens, repeats=self.num_heads, axis=0)
            
        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries, `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens, **kwargs)
        
        # Shape of `output_concat`: (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
```

[**다중 헤드의 병렬 계산**] 을 허용하기 위해 위의 `MultiHeadAttention` 클래스는 아래에 정의된 두 개의 전치 함수를 사용합니다.특히 `transpose_output` 함수는 `transpose_qkv` 함수의 작동을 역전시킵니다.

```{.python .input}
#@save
def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads."""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.transpose(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


#@save
def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`."""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

```{.python .input}
#@tab pytorch
#@save
def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads."""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


#@save
def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`."""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

```{.python .input}
#@tab tensorflow
#@save
def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads."""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = tf.reshape(X, shape=(X.shape[0], X.shape[1], num_heads, -1))

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = tf.transpose(X, perm=(0, 2, 1, 3))

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return tf.reshape(X, shape=(-1, X.shape[2], X.shape[3]))


#@save
def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`."""
    X = tf.reshape(X, shape=(-1, num_heads, X.shape[1], X.shape[2]))
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    return tf.reshape(X, shape=(X.shape[0], X.shape[1], -1))
```

키와 값이 동일한 장난감 예제를 사용하여 [**구현된**] `MultiHeadAttention` 클래스를 테스트해 보겠습니다.결과적으로 멀티 헤드 어텐션 출력의 모양은 (`batch_size`, `num_queries`, `num_hiddens`) 입니다.

```{.python .input}
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
#@tab pytorch
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
attention.eval()
```

```{.python .input}
#@tab tensorflow
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
```

```{.python .input}
#@tab mxnet, pytorch
batch_size, num_queries, num_kvpairs, valid_lens = 2, 4, 6, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
Y = d2l.ones((batch_size, num_kvpairs, num_hiddens))
attention(X, Y, Y, valid_lens).shape
```

```{.python .input}
#@tab tensorflow
batch_size, num_queries, num_kvpairs, valid_lens = 2, 4, 6, d2l.tensor([3, 2])
X = tf.ones((batch_size, num_queries, num_hiddens))
Y = tf.ones((batch_size, num_kvpairs, num_hiddens))
attention(X, Y, Y, valid_lens, training=False).shape
```

## 요약

* 멀티 헤드 어텐션은 쿼리, 키 및 값의 서로 다른 표현 하위 공간을 통해 동일한 관심 풀링에 대한 지식을 결합합니다.
* 다중 헤드 주의의 여러 헤드를 병렬로 계산하려면 적절한 텐서 조작이 필요합니다.

## 연습문제

1. 이 실험에서 여러 머리의 주의력 가중치를 시각화합니다.
1. 다중 헤드 주의력을 기반으로 훈련된 모델이 있고 예측 속도를 높이기 위해 가장 중요하지 않은 관심 헤드를 정리하려고 한다고 가정합니다.주의력의 중요성을 측정하기 위한 실험을 어떻게 설계할 수 있을까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1634)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1635)
:end_tab:
