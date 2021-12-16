# 주의력 점수 함수
:label:`sec_attention-scoring-functions`

:numref:`sec_nadaraya-watson`에서는 쿼리와 키 간의 상호 작용을 모델링하기 위해 가우스 커널을 사용했습니다.:eqref:`eq_nadaraya-watson-gaussian`에서 가우스 커널의 지수를*주의 점수 함수* (또는 줄여서 *점수 함수*) 로 처리하면, 이 함수의 결과는 기본적으로 소프트맥스 연산에 입력되었습니다.그 결과 키와 쌍을 이루는 값에 대한 확률 분포 (주의 가중치) 를 얻었습니다.결국 주의력 풀링의 출력은 단순히 이러한 주의력 가중치를 기반으로 한 값의 가중 합계입니다. 

높은 수준에서 위의 알고리즘을 사용하여 :numref:`fig_qkv`에서 주의 메커니즘의 프레임 워크를 인스턴스화 할 수 있습니다.$a$으로 주의력 점수 함수를 나타내는 :numref:`fig_attention_output`는 주의력 풀링의 출력을 값의 가중 합계로 계산하는 방법을 보여줍니다.주의 가중치는 확률 분포이므로 가중 합계는 기본적으로 가중 평균입니다. 

![Computing the output of attention pooling as a weighted average of values.](../img/attention-output.svg)
:label:`fig_attention_output`

수학적으로, 쿼리 $\mathbf{q} \in \mathbb{R}^q$ 및 $m$ 키-값 쌍 $(\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)$이 있다고 가정합니다. 여기서 $\mathbf{k}_i \in \mathbb{R}^k$와 $\mathbf{v}_i \in \mathbb{R}^v$가 있습니다.주의 풀링 $f$은 값의 가중 합계로 인스턴스화됩니다. 

$$f(\mathbf{q}, (\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)) = \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i \in \mathbb{R}^v,$$
:eqlabel:`eq_attn-pooling`

여기서 쿼리 $\mathbf{q}$ 및 키 $\mathbf{k}_i$에 대한 주의 가중치 (스칼라) 는 두 벡터를 스칼라에 매핑하는 주의 점수 함수 $a$의 소프트맥스 연산에 의해 계산됩니다. 

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(a(\mathbf{q}, \mathbf{k}_i)) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_{j=1}^m \exp(a(\mathbf{q}, \mathbf{k}_j))} \in \mathbb{R}.$$
:eqlabel:`eq_attn-scoring-alpha`

보시다시피, 주의력 점수 함수 $a$의 다양한 선택은 주의력 풀링의 다른 행동으로 이어집니다.이 섹션에서는 나중에 보다 정교한 주의력 메커니즘을 개발하는 데 사용할 인기 있는 두 가지 채점 기능을 소개합니다.

```{.python .input}
import math
from d2l import mxnet as d2l
from mxnet import np, npx
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

## [**마스킹된 소프트맥스 작동**]

앞서 언급했듯이 소프트맥스 연산은 확률 분포를 주의 가중치로 출력하는 데 사용됩니다.경우에 따라 모든 값을 주의력 풀링에 적용해서는 안 됩니다.예를 들어 :numref:`sec_machine_translation`의 효율적인 미니배치 처리를 위해 일부 텍스트 시퀀스에는 의미가 없는 특수 토큰이 채워집니다.의미있는 토큰에 대해서만 값으로 관심을 끌기 위해 softmax를 계산할 때이 지정된 범위를 벗어난 시퀀스 길이를 필터링하기 위해 유효한 시퀀스 길이 (토큰 수) 를 지정할 수 있습니다.이런 식으로 다음 `masked_softmax` 함수에서 이러한*마스크 소프트 맥스 연산*을 구현할 수 있습니다. 여기서 유효한 길이를 초과하는 값은 0으로 마스킹됩니다.

```{.python .input}
#@save
def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return npx.softmax(X)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = valid_lens.repeat(shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = npx.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, True,
                              value=-1e6, axis=1)
        return npx.softmax(X).reshape(shape)
```

```{.python .input}
#@tab pytorch
#@save
def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
```

```{.python .input}
#@tab tensorflow
#@save
def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])
            
        else:
            valid_lens = tf.reshape(valid_lens, shape=-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0    
        X = d2l.sequence_mask(tf.reshape(X, shape=(-1, shape[-1])), valid_lens, value=-1e6)    
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)
```

[**이 함수의 작동 방식을 시연**] 하려면 두 개의 $2 \times 4$ 행렬 예제로 구성된 미니배치를 고려해 보십시오. 여기서 이 두 예제의 유효한 길이는 각각 2와 3입니다.마스킹된 소프트맥스 연산의 결과로 유효한 길이를 초과하는 값은 모두 0으로 마스킹됩니다.

```{.python .input}
masked_softmax(np.random.uniform(size=(2, 2, 4)), d2l.tensor([2, 3]))
```

```{.python .input}
#@tab pytorch
masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
```

```{.python .input}
#@tab tensorflow
masked_softmax(tf.random.uniform(shape=(2, 2, 4)), tf.constant([2, 3]))
```

마찬가지로 2 차원 텐서를 사용하여 각 행렬 예제의 모든 행에 유효한 길이를 지정할 수도 있습니다.

```{.python .input}
masked_softmax(np.random.uniform(size=(2, 2, 4)),
               d2l.tensor([[1, 3], [2, 4]]))
```

```{.python .input}
#@tab pytorch
masked_softmax(torch.rand(2, 2, 4), d2l.tensor([[1, 3], [2, 4]]))
```

```{.python .input}
#@tab tensorflow
masked_softmax(tf.random.uniform((2, 2, 4)), tf.constant([[1, 3], [2, 4]]))
```

## [**부가적인 주의**]
:label:`subsec_additive-attention`

일반적으로 쿼리와 키가 길이가 다른 벡터인 경우 가산적 주의를 스코어링 함수로 사용할 수 있습니다.쿼리 $\mathbf{q} \in \mathbb{R}^q$와 키 $\mathbf{k} \in \mathbb{R}^k$가 주어진*부가적인 주의* 점수 함수입니다. 

$$a(\mathbf q, \mathbf k) = \mathbf w_v^\top \text{tanh}(\mathbf W_q\mathbf q + \mathbf W_k \mathbf k) \in \mathbb{R},$$
:eqlabel:`eq_additive-attn`

여기서 학습 가능한 매개 변수는 $\mathbf W_q\in\mathbb R^{h\times q}$, $\mathbf W_k\in\mathbb R^{h\times k}$ 및 $\mathbf w_v\in\mathbb R^{h}$입니다.:eqref:`eq_additive-attn`와 동일하게 쿼리와 키는 연결되어 단일 은닉 계층이 있는 MLP에 공급되며, 숨겨진 단위 수는 하이퍼파라미터인 $h$입니다.$\tanh$을 활성화 함수로 사용하고 편향 항을 비활성화함으로써 다음과 같이 부가적인 주의를 구현합니다.

```{.python .input}
#@save
class AdditiveAttention(nn.Block):
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # Use `flatten=False` to only transform the last axis so that the
        # shapes for the other axes are kept the same
        self.W_k = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.W_q = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.w_v = nn.Dense(1, use_bias=False, flatten=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of `queries`: (`batch_size`, no. of
        # queries, 1, `num_hiddens`) and shape of `keys`: (`batch_size`, 1,
        # no. of key-value pairs, `num_hiddens`). Sum them up with
        # broadcasting
        features = np.expand_dims(queries, axis=2) + np.expand_dims(
            keys, axis=1)
        features = np.tanh(features)
        # There is only one output of `self.w_v`, so we remove the last
        # one-dimensional entry from the shape. Shape of `scores`:
        # (`batch_size`, no. of queries, no. of key-value pairs)
        scores = np.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of `values`: (`batch_size`, no. of key-value pairs, value
        # dimension)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```{.python .input}
#@tab pytorch
#@save
class AdditiveAttention(nn.Module):
    """Additive attention."""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of `queries`: (`batch_size`, no. of
        # queries, 1, `num_hiddens`) and shape of `keys`: (`batch_size`, 1,
        # no. of key-value pairs, `num_hiddens`). Sum them up with
        # broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of `self.w_v`, so we remove the last
        # one-dimensional entry from the shape. Shape of `scores`:
        # (`batch_size`, no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of `values`: (`batch_size`, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

```{.python .input}
#@tab tensorflow
#@save
class AdditiveAttention(tf.keras.layers.Layer):
    """Additive attention."""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.w_v = tf.keras.layers.Dense(1, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, queries, keys, values, valid_lens, **kwargs):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of `queries`: (`batch_size`, no. of
        # queries, 1, `num_hiddens`) and shape of `keys`: (`batch_size`, 1,
        # no. of key-value pairs, `num_hiddens`). Sum them up with
        # broadcasting
        features = tf.expand_dims(queries, axis=2) + tf.expand_dims(
            keys, axis=1)
        features = tf.nn.tanh(features)
        # There is only one output of `self.w_v`, so we remove the last
        # one-dimensional entry from the shape. Shape of `scores`:
        # (`batch_size`, no. of queries, no. of key-value pairs)
        scores = tf.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of `values`: (`batch_size`, no. of key-value pairs, value
        # dimension)
        return tf.matmul(self.dropout(
            self.attention_weights, **kwargs), values)
```

쿼리, 키 및 값의 모양 (배치 크기, 단계 수 또는 토큰의 시퀀스 길이, 피쳐 크기) 이 ($2$, $1$, $20$), ($20$), ($2$, $2$) 및 ($2$), $2$, $2$, $2$, $2$, $2$, $2$, $2$, $2$ 2293618, $4$) 을 각각 포함한다.주의 풀링 출력은 (배치 크기, 쿼리의 단계 수, 값의 특징 크기) 의 형태를 갖습니다.

```{.python .input}
queries, keys = d2l.normal(0, 1, (2, 1, 20)), d2l.ones((2, 10, 2))
# The two value matrices in the `values` minibatch are identical
values = np.arange(40).reshape(1, 10, 4).repeat(2, axis=0)
valid_lens = d2l.tensor([2, 6])

attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
attention.initialize()
attention(queries, keys, values, valid_lens)
```

```{.python .input}
#@tab pytorch
queries, keys = d2l.normal(0, 1, (2, 1, 20)), d2l.ones((2, 10, 2))
# The two value matrices in the `values` minibatch are identical
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
    2, 1, 1)
valid_lens = d2l.tensor([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)
attention.eval()
attention(queries, keys, values, valid_lens)
```

```{.python .input}
#@tab tensorflow
queries, keys = tf.random.normal(shape=(2, 1, 20)), tf.ones((2, 10, 2))
# The two value matrices in the `values` minibatch are identical
values = tf.repeat(tf.reshape(
    tf.range(40, dtype=tf.float32), shape=(1, 10, 4)), repeats=2, axis=0)
valid_lens = tf.constant([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)
attention(queries, keys, values, valid_lens, training=False)
```

가산적 주의에는 학습 가능한 파라미터가 포함되어 있지만 이 예에서는 모든 키가 동일하므로 [**주의 가중치**] 는 균일하며 지정된 유효한 길이에 의해 결정됩니다.

```{.python .input}
#@tab all
d2l.show_heatmaps(d2l.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## [**스케일 도트 제품 주의**]

스코어링 함수에 대한 계산적으로 보다 효율적인 설계는 단순히 내적이 될 수 있습니다.그러나 내적 연산을 수행하려면 쿼리와 키의 벡터 길이가 같아야 합니다 (예: $d$).쿼리와 키의 모든 요소가 평균이 0이고 단위 분산이 있는 독립적인 확률 변수라고 가정합니다.두 벡터의 내적은 평균이 0이고 분산이 $d$입니다.벡터 길이에 관계없이 내적의 분산이 여전히 1로 유지되도록 하기 위해*척도화된 내적 주의* 점수 함수를 사용합니다. 

$$a(\mathbf q, \mathbf k) = \mathbf{q}^\top \mathbf{k}  /\sqrt{d}$$

내적을 $\sqrt{d}$으로 나눕니다.실제로는 쿼리와 키의 길이가 $d$이고 값의 길이가 $v$인 $n$ 쿼리 및 $m$ 키-값 쌍에 대한 주의를 계산하는 것과 같이 효율성을 위해 미니 배치에서 생각하는 경우가 많습니다.쿼리 $\mathbf Q\in\mathbb R^{n\times d}$, 키 $\mathbf K\in\mathbb R^{m\times d}$ 및 값 $\mathbf V\in\mathbb R^{m\times v}$의 배율 조정된 내적 관심은 다음과 같습니다. 

$$ \mathrm{softmax}\left(\frac{\mathbf Q \mathbf K^\top }{\sqrt{d}}\right) \mathbf V \in \mathbb{R}^{n\times v}.$$
:eqlabel:`eq_softmax_QK_V`

스케일링된 내적 주의의 다음 구현에서는 모델 정규화에 드롭아웃을 사용합니다.

```{.python .input}
#@save
class DotProductAttention(nn.Block):
    """Scaled dot product attention."""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = npx.batch_dot(queries, keys, transpose_b=True) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```{.python .input}
#@tab pytorch
#@save
class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

```{.python .input}
#@tab tensorflow
#@save
class DotProductAttention(tf.keras.layers.Layer):
    """Scaled dot product attention."""
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def call(self, queries, keys, values, valid_lens, **kwargs):
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(
            tf.cast(d, dtype=tf.float32))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), values)
```

[**위의 `DotProductAttention` 클래스를 시연**] 하기 위해 부가적인 주의를 기울이기 위해 이전 장난감 예제와 동일한 키, 값 및 유효한 길이를 사용합니다.내적 연산의 경우 쿼리의 특징 크기를 키의 크기와 동일하게 만듭니다.

```{.python .input}
queries = d2l.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.initialize()
attention(queries, keys, values, valid_lens)
```

```{.python .input}
#@tab pytorch
queries = d2l.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
attention(queries, keys, values, valid_lens)
```

```{.python .input}
#@tab tensorflow
queries = tf.random.normal(shape=(2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention(queries, keys, values, valid_lens, training=False)
```

`keys`에는 쿼리로 구분할 수 없는 동일한 요소가 포함되어 있으므로 [**균일 주의 가중치**] 가 얻어집니다.

```{.python .input}
#@tab all
d2l.show_heatmaps(d2l.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## 요약

* 주의력 풀링의 출력을 값의 가중 평균으로 계산할 수 있습니다. 여기서 주의력 채점 함수의 다른 선택은 다른 주의력 풀링 동작으로 이어집니다.
* 쿼리와 키가 길이가 다른 벡터인 경우 가산 주의 점수 함수를 사용할 수 있습니다.동일한 경우 스케일링된 내적 주의력 점수 계산 함수가 계산적으로 더 효율적입니다.

## 연습문제

1. 장난감 예제에서 키를 수정하고 주의력 가중치를 시각화합니다.가산 주의와 스케일링된 점 제품 주의가 여전히 동일한 주의 가중치를 출력합니까?왜, 왜 안되니?
1. 행렬 곱셈만 사용하여 벡터 길이가 다른 쿼리와 키에 대해 새로운 스코어링 함수를 설계할 수 있습니까?
1. 쿼리와 키의 벡터 길이가 같은 경우 스코어링 함수에 대해 벡터 합계가 내적보다 더 나은 설계입니까?왜, 왜 안되니?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/346)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1064)
:end_tab:
