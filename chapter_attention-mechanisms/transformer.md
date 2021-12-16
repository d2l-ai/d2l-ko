# 트랜스포머
:label:`sec_transformer`

우리는 :numref:`subsec_cnn-rnn-self-attention`에서 CNN, RNN 및 자기 관심을 비교했습니다.특히 자기 주의는 병렬 계산과 가장 짧은 최대 경로 길이를 모두 즐깁니다.따라서 당연히 자기 주의를 사용하여 심층 아키텍처를 설계하는 것이 매력적입니다.입력 표현 :cite:`Cheng.Dong.Lapata.2016,Lin.Feng.Santos.ea.2017,Paulus.Xiong.Socher.2017`에 여전히 RNN에 의존하는 이전의 자기 주의 모델과 달리 트랜스포머 모델은 컨벌루션 또는 순환 계층 :cite:`Vaswani.Shazeer.Parmar.ea.2017`가 없는 주의 메커니즘만을 기반으로 합니다.원래 텍스트 데이터에 대한 시퀀스 대 시퀀스 학습을 위해 제안되었지만, 트랜스포머는 언어, 시각, 언어 및 강화 학습 분야와 같은 광범위한 최신 딥 러닝 애플리케이션에 널리 보급되어 왔습니다. 

## 모델

인코더-디코더 아키텍처의 예로서 변압기의 전체 아키텍처는 :numref:`fig_transformer`에 나와 있습니다.보시다시피 변압기는 인코더와 디코더로 구성됩니다.:numref:`fig_s2s_attention_details`에서 시퀀스 대 시퀀스 학습에 대한 Bahdanau의 관심과 달리 입력 (소스) 및 출력 (대상) 시퀀스 임베딩은 자기 주의에 따라 모듈을 스택하는 인코더 및 디코더에 공급되기 전에 위치 인코딩과 함께 추가됩니다. 

![The transformer architecture.](../img/transformer.svg)
:width:`500px`
:label:`fig_transformer`

이제 :numref:`fig_transformer`의 트랜스포머 아키텍처에 대한 개요를 제공합니다.높은 수준에서 트랜스포머 엔코더는 여러 개의 동일한 레이어로 구성된 스택이며 각 레이어에는 두 개의 하위 레이어가 있습니다 (둘 중 하나는 $\mathrm{sublayer}$로 표시됨).첫 번째는 다중 헤드 자기주의 풀링이고 두 번째는 위치 별 피드 포워드 네트워크입니다.특히 인코더 자체 주의에서 쿼리, 키 및 값은 모두 이전 인코더 레이어의 출력에서 가져옵니다.:numref:`sec_resnet`의 ResNet 설계에서 영감을 얻은 잔여 연결이 두 하위 레이어 주위에 사용됩니다.변압기에서 시퀀스의 임의 위치에있는 모든 입력 $\mathbf{x} \in \mathbb{R}^d$에 대해 잔류 연결 $\mathbf{x} + \mathrm{sublayer}(\mathbf{x}) \in \mathbb{R}^d$이 실현 가능하도록 $\mathrm{sublayer}(\mathbf{x}) \in \mathbb{R}^d$가 필요합니다.잔차 연결에서 이 추가 후 바로 계층 정규화 :cite:`Ba.Kiros.Hinton.2016`가 이어집니다.그 결과, 트랜스포머 인코더는 입력 시퀀스의 각 위치에 대해 $d$차원 벡터 표현을 출력한다. 

트랜스포머 디코더는 잔여 연결 및 계층 정규화가 있는 여러 개의 동일한 계층으로 구성된 스택이기도 합니다.인코더에 설명된 두 개의 하위 레이어 외에도 디코더는 이 두 레이어 사이에 인코더-디코더 주의라고 하는 세 번째 하위 레이어를 삽입합니다.인코더-디코더주의에서 쿼리는 이전 디코더 계층의 출력에서 비롯되며 키와 값은 트랜스포머 인코더 출력에서 가져온 것입니다.디코더 자체 주의에서 쿼리, 키 및 값은 모두 이전 디코더 계층의 출력에서 가져옵니다.그러나 디코더의 각 위치는 해당 위치까지 디코더의 모든 위치에만 참석할 수 있습니다.이러한*마스킹* 주의는 자동 회귀 속성을 보존하여 예측이 생성된 출력 토큰에만 의존하도록 합니다. 

우리는 이미 :numref:`sec_multihead-attention`의 스케일링된 도트 곱과 :numref:`subsec_positional-encoding`의 위치 인코딩을 기반으로 멀티 헤드 어텐션을 설명하고 구현했습니다.다음에서는 나머지 변압기 모델을 구현합니다.

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import autograd, np, npx
from mxnet.gluon import nn
import pandas as pd
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import pandas as pd
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import pandas as pd
import tensorflow as tf
```

## [**위치별 피드포워드 네트워크**]

위치별 피드포워드 네트워크는 동일한 MLP를 사용하여 모든 시퀀스 위치에서 표현을 변환합니다.이것이 바로 우리가 이것을 위치별*이라고 부르는 이유입니다.아래 구현에서 모양이 있는 입력 `X` (배치 크기, 시간 단계 수 또는 토큰의 시퀀스 길이, 은닉 유닛 수 또는 피처 차원) 는 2계층 MLP에 의해 형상의 출력 텐서 (배치 크기, 시간 스텝 수, `ffn_num_outputs`) 로 변환됩니다.

```{.python .input}
#@save
class PositionWiseFFN(nn.Block):
    """Positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Dense(ffn_num_hiddens, flatten=False,
                               activation='relu')
        self.dense2 = nn.Dense(ffn_num_outputs, flatten=False)

    def forward(self, X):
        return self.dense2(self.dense1(X))
```

```{.python .input}
#@tab pytorch
#@save
class PositionWiseFFN(nn.Module):
    """Positionwise feed-forward network."""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```

```{.python .input}
#@tab tensorflow
#@save
class PositionWiseFFN(tf.keras.layers.Layer):
    """Positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super().__init__(*kwargs)
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)

    def call(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```

다음 예는 [**텐서의 가장 안쪽 차원이 위치 방향 피드 포워드 네트워크의 출력 수로 변경됨**] 을 보여줍니다.모든 위치에서 동일한 MLP 변환이 이루어지기 때문에 이러한 모든 위치의 입력이 동일하면 출력도 동일합니다.

```{.python .input}
ffn = PositionWiseFFN(4, 8)
ffn.initialize()
ffn(np.ones((2, 3, 4)))[0]
```

```{.python .input}
#@tab pytorch
ffn = PositionWiseFFN(4, 4, 8)
ffn.eval()
ffn(d2l.ones((2, 3, 4)))[0]
```

```{.python .input}
#@tab tensorflow
ffn = PositionWiseFFN(4, 8)
ffn(tf.ones((2, 3, 4)))[0]
```

## 잔차 연결 및 계층 정규화

이제 :numref:`fig_transformer`의 “추가 및 표준” 구성 요소에 초점을 맞추겠습니다.이 섹션의 시작 부분에서 설명했듯이 이 연결은 바로 뒤에 계층 정규화가 이어지는 잔차 연결입니다.둘 다 효과적인 심층 아키텍처의 핵심입니다. 

:numref:`sec_batch_norm`에서는 배치 정규화가 미니배치 내의 예제에서 어떻게 재중심을 조정하고 다시 조정하는지 설명했습니다.계층 정규화는 전자가 특징 차원에서 정규화된다는 점을 제외하면 배치 정규화와 동일합니다.컴퓨터 비전에서 널리 응용되고 있음에도 불구하고 배치 정규화는 일반적으로 입력값이 가변 길이 시퀀스인 자연어 처리 작업의 계층 정규화보다 경험적으로 덜 효과적입니다. 

다음 코드 스니펫 [**계층 정규화 및 배치 정규화를 통해 서로 다른 차원의 정규화를 비교**].

```{.python .input}
ln = nn.LayerNorm()
ln.initialize()
bn = nn.BatchNorm()
bn.initialize()
X = d2l.tensor([[1, 2], [2, 3]])
# Compute mean and variance from `X` in the training mode
with autograd.record():
    print('layer norm:', ln(X), '\nbatch norm:', bn(X))
```

```{.python .input}
#@tab pytorch
ln = nn.LayerNorm(2)
bn = nn.BatchNorm1d(2)
X = d2l.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# Compute mean and variance from `X` in the training mode
print('layer norm:', ln(X), '\nbatch norm:', bn(X))
```

```{.python .input}
#@tab tensorflow
ln = tf.keras.layers.LayerNormalization()
bn = tf.keras.layers.BatchNormalization()
X = tf.constant([[1, 2], [2, 3]], dtype=tf.float32)
print('layer norm:', ln(X), '\nbatch norm:', bn(X, training=True))
```

이제 `AddNorm` 클래스 [**잔여 연결을 사용하고 계층 정규화**] 를 구현할 수 있습니다.드롭아웃은 정규화에도 적용됩니다.

```{.python .input}
#@save
class AddNorm(nn.Block):
    """Residual connection followed by layer normalization."""
    def __init__(self, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm()

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```

```{.python .input}
#@tab pytorch
#@save
class AddNorm(nn.Module):
    """Residual connection followed by layer normalization."""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```

```{.python .input}
#@tab tensorflow
#@save
class AddNorm(tf.keras.layers.Layer):
    """Residual connection followed by layer normalization."""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(normalized_shape)
        
    def call(self, X, Y, **kwargs):
        return self.ln(self.dropout(Y, **kwargs) + X)
```

잔차 연결은 두 입력의 모양이 같아야 [**가산 연산 후에도 출력 텐서도 동일한 모양을 갖게됩니다**].

```{.python .input}
add_norm = AddNorm(0.5)
add_norm.initialize()
add_norm(d2l.ones((2, 3, 4)), d2l.ones((2, 3, 4))).shape
```

```{.python .input}
#@tab pytorch
add_norm = AddNorm([3, 4], 0.5) # Normalized_shape is input.size()[1:]
add_norm.eval()
add_norm(d2l.ones((2, 3, 4)), d2l.ones((2, 3, 4))).shape
```

```{.python .input}
#@tab tensorflow
add_norm = AddNorm([1, 2], 0.5) # Normalized_shape is: [i for i in range(len(input.shape))][1:]
add_norm(tf.ones((2, 3, 4)), tf.ones((2, 3, 4)), training=False).shape
```

## 엔코더

트랜스포머 엔코더를 조립하는 데 필요한 모든 필수 구성 요소를 가지고 [**인코더 내의 단일 레이어**] 를 구현하는 것부터 시작하겠습니다.다음 `EncoderBlock` 클래스에는 두 개의 하위 계층이 포함되어 있습니다. 다중 헤드 자체 주의 및 위치별 피드포워드 신경망입니다. 여기서 잔차 연결과 계층 정규화가 두 하위 계층 주위에 사용됩니다.

```{.python .input}
#@save
class EncoderBlock(nn.Block):
    """Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
```

```{.python .input}
#@tab pytorch
#@save
class EncoderBlock(nn.Module):
    """Transformer encoder block."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
```

```{.python .input}
#@tab tensorflow
#@save
class EncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                                num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        
    def call(self, X, valid_lens, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, **kwargs), **kwargs)
        return self.addnorm2(Y, self.ffn(Y), **kwargs)
```

보시다시피 [**트랜스포머 엔코더의 모든 레이어는 입력 모양을 변경하지 않습니다.**]

```{.python .input}
X = d2l.ones((2, 100, 24))
valid_lens = d2l.tensor([3, 2])
encoder_blk = EncoderBlock(24, 48, 8, 0.5)
encoder_blk.initialize()
encoder_blk(X, valid_lens).shape
```

```{.python .input}
#@tab pytorch
X = d2l.ones((2, 100, 24))
valid_lens = d2l.tensor([3, 2])
encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
encoder_blk.eval()
encoder_blk(X, valid_lens).shape
```

```{.python .input}
#@tab tensorflow
X = tf.ones((2, 100, 24))
valid_lens = tf.constant([3, 2])
norm_shape = [i for i in range(len(X.shape))][1:]
encoder_blk = EncoderBlock(24, 24, 24, 24, norm_shape, 48, 8, 0.5)
encoder_blk(X, valid_lens, training=False).shape
```

다음 [**트랜스포머 인코더**] 구현에서는 위 `EncoderBlock` 클래스의 `num_layers` 인스턴스를 스택합니다.값이 항상 -1에서 1 사이인 고정 위치 인코딩을 사용하기 때문에 입력 임베딩과 위치 인코딩을 합산하기 전에 학습 가능한 입력 임베딩의 값에 임베딩 차원의 제곱근을 곱하여 다시 스케일링합니다.

```{.python .input}
#@save
class TransformerEncoder(d2l.Encoder):
    """Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(
                EncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout,
                             use_bias))

    def forward(self, X, valid_lens, *args):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

```{.python .input}
#@tab pytorch
#@save
class TransformerEncoder(d2l.Encoder):
    """Transformer encoder."""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

```{.python .input}
#@tab tensorflow
#@save
class TransformerEncoder(d2l.Encoder):
    """Transformer encoder."""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hiddens, num_heads,
                 num_layers, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = [EncoderBlock(
            key_size, query_size, value_size, num_hiddens, norm_shape,
            ffn_num_hiddens, num_heads, dropout, bias) for _ in range(
            num_layers)]
        
    def call(self, X, valid_lens, **kwargs):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * tf.math.sqrt(
            tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens, **kwargs)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

아래에서는 [**2계층 트랜스포머 인코더 만들기**] 에 하이퍼파라미터를 지정합니다.트랜스포머 엔코더 출력의 모양은 (배치 크기, 시간 단계 수, `num_hiddens`) 입니다.

```{.python .input}
encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
encoder.initialize()
encoder(np.ones((2, 100)), valid_lens).shape
```

```{.python .input}
#@tab pytorch
encoder = TransformerEncoder(
    200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
encoder.eval()
encoder(d2l.ones((2, 100), dtype=torch.long), valid_lens).shape
```

```{.python .input}
#@tab tensorflow
encoder = TransformerEncoder(200, 24, 24, 24, 24, [1, 2], 48, 8, 2, 0.5)
encoder(tf.ones((2, 100)), valid_lens, training=False).shape
```

## 디코더

:numref:`fig_transformer`에서 볼 수 있듯이 [**트랜스포머 디코더는 여러 개의 동일한 레이어로 구성됩니다**].각 계층은 디코더 자체 주의, 인코더-디코더 주의 및 위치 피드 포워드 네트워크의 세 가지 하위 계층을 포함하는 다음 `DecoderBlock` 클래스로 구현됩니다.이러한 하위 계층은 주위에 잔차 연결을 사용하고 계층 정규화를 사용합니다. 

이 섹션의 앞부분에서 설명한 것처럼 마스킹된 멀티 헤드 디코더 자체 주의 (첫 번째 하위 계층) 에서 쿼리, 키 및 값은 모두 이전 디코더 계층의 출력에서 가져옵니다.시퀀스-투-시퀀스 모델을 훈련시킬 때 출력 시퀀스의 모든 위치 (시간 단계) 에 있는 토큰을 알 수 있습니다.그러나 예측 중에 출력 시퀀스는 토큰별로 생성됩니다. 따라서 임의의 디코더 시간 단계에서 생성된 토큰만 디코더 자체 주의에 사용될 수 있습니다.디코더에서 자동 회귀를 유지하기 위해 마스킹된 자체 주의는 `dec_valid_lens`를 지정하여 모든 쿼리가 쿼리 위치까지 디코더의 모든 위치에만 참석하도록 합니다.

```{.python .input}
class DecoderBlock(nn.Block):
    # The `i`-th block in the decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm1 = AddNorm(dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm2 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so `state[2][self.i]` is `None` as initialized.
        # When decoding any output sequence token by token during prediction,
        # `state[2][self.i]` contains representations of the decoded output at
        # the `i`-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = np.concatenate((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        if autograd.is_training():
            batch_size, num_steps, _ = X.shape
            # Shape of `dec_valid_lens`: (`batch_size`, `num_steps`), where
            # every row is [1, 2, ..., `num_steps`]
            dec_valid_lens = np.tile(np.arange(1, num_steps + 1, ctx=X.ctx),
                                     (batch_size, 1))
        else:
            dec_valid_lens = None

        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention. Shape of `enc_outputs`:
        # (`batch_size`, `num_steps`, `num_hiddens`)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
```

```{.python .input}
#@tab pytorch
class DecoderBlock(nn.Module):
    # The `i`-th block in the decoder
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so `state[2][self.i]` is `None` as initialized.
        # When decoding any output sequence token by token during prediction,
        # `state[2][self.i]` contains representations of the decoded output at
        # the `i`-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # Shape of `dec_valid_lens`: (`batch_size`, `num_steps`), where
            # every row is [1, 2, ..., `num_steps`]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention. Shape of `enc_outputs`:
        # (`batch_size`, `num_steps`, `num_hiddens`)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
```

```{.python .input}
#@tab tensorflow
class DecoderBlock(tf.keras.layers.Layer):
    # The `i`-th block in the decoder
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super().__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)
        
    def call(self, X, state, **kwargs):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so `state[2][self.i]` is `None` as initialized.
        # When decoding any output sequence token by token during prediction,
        # `state[2][self.i]` contains representations of the decoded output at
        # the `i`-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = tf.concat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if kwargs["training"]:
            batch_size, num_steps, _ = X.shape
            # Shape of `dec_valid_lens`: (`batch_size`, `num_steps`), where
            # every row is [1, 2, ..., `num_steps`]
            dec_valid_lens = tf.repeat(tf.reshape(tf.range(1, num_steps + 1),
                                                 shape=(-1, num_steps)), repeats=batch_size, axis=0)

        else:
            dec_valid_lens = None
            
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens, **kwargs)
        Y = self.addnorm1(X, X2, **kwargs)
        # Encoder-decoder attention. Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hiddens`)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens, **kwargs)
        Z = self.addnorm2(Y, Y2, **kwargs)
        return self.addnorm3(Z, self.ffn(Z), **kwargs), state
```

잔여 연결에서 인코더-디코더 주의 및 추가 동작에서 스케일링된 내적 연산을 용이하게 하기 위해, [**디코더의 특징 치수 (`num_hiddens`) 는 인코더의 형상 치수 (`num_hiddens`) 와 동일합니다.**]

```{.python .input}
decoder_blk = DecoderBlock(24, 48, 8, 0.5, 0)
decoder_blk.initialize()
X = np.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
decoder_blk(X, state)[0].shape
```

```{.python .input}
#@tab pytorch
decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
decoder_blk.eval()
X = d2l.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
decoder_blk(X, state)[0].shape
```

```{.python .input}
#@tab tensorflow
decoder_blk = DecoderBlock(24, 24, 24, 24, [1, 2], 48, 8, 0.5, 0)
X = tf.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
decoder_blk(X, state, training=False)[0].shape
```

이제 `DecoderBlock`의 `num_layers` 인스턴스로 구성된 [**전체 트랜스포머 디코더를 구성**] 합니다.결국 완전히 연결된 계층은 가능한 모든 `vocab_size`의 출력 토큰에 대한 예측을 계산합니다.디코더 자기주의 가중치와 인코더-디코더 주의 가중치 모두 추후 시각화를 위해 저장됩니다.

```{.python .input}
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add(
                DecoderBlock(num_hiddens, ffn_num_hiddens, num_heads,
                             dropout, i))
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
#@tab pytorch
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
#@tab tensorflow
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hidens, num_heads, num_layers, dropout, **kwargs):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = [DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                                  ffn_num_hiddens, num_heads, dropout, i) for i in range(num_layers)]
        self.dense = tf.keras.layers.Dense(vocab_size)
        
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
    
    def call(self, X, state, **kwargs):
        X = self.pos_encoding(self.embedding(X) * tf.math.sqrt(tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]  # 2 Attention layers in decoder
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state, **kwargs)
            # Decoder self-attention weights
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state
    
    @property
    def attention_weights(self):
        return self._attention_weights
```

## [**교육**]

트랜스포머 아키텍처에 따라 인코더-디코더 모델을 인스턴스화 해 보겠습니다.여기서는 트랜스포머 엔코더와 트랜스포머 디코더 모두 4-head attention을 사용하여 2개의 레이어를 갖도록 지정합니다.:numref:`sec_seq2seq_training`와 마찬가지로 영어-프랑스어 기계 번역 데이터 세트에서 시퀀스 간 학습을 위해 변압기 모델을 학습합니다.

```{.python .input}
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_hiddens, num_heads = 64, 4

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(
    len(src_vocab), num_hiddens, ffn_num_hiddens, num_heads, num_layers,
    dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), num_hiddens, ffn_num_hiddens, num_heads, num_layers,
    dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

```{.python .input}
#@tab pytorch
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

```{.python .input}
#@tab tensorflow
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_hiddens, num_heads = 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [2]

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens, norm_shape,
    ffn_num_hiddens, num_heads, num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens, norm_shape,
    ffn_num_hiddens, num_heads, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

훈련 후에는 트랜스포머 모델을 사용하여 프랑스어로 [**영어 문장 몇 개를 번역**] 하고 BLEU 점수를 계산합니다.

```{.python .input}
#@tab mxnet, pytorch
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
```

```{.python .input}
#@tab tensorflow
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
```

마지막 영어 문장을 프랑스어로 번역할 때 [**변압기 주의 가중치를 시각화**] 해 보겠습니다.인코더 자체 주의 가중치의 모양은 (인코더 레이어 수, 주의 헤드 수, `num_steps` 또는 쿼리 수, `num_steps` 또는 키-값 쌍의 수) 입니다.

```{.python .input}
#@tab all
enc_attention_weights = d2l.reshape(
    d2l.concat(net.encoder.attention_weights, 0),
    (num_layers, num_heads, -1, num_steps))
enc_attention_weights.shape
```

인코더 자체 주의에서 쿼리와 키는 모두 동일한 입력 시퀀스에서 가져옵니다.패딩 토큰은 의미가 없으므로 입력 시퀀스의 유효한 길이를 지정하면 패딩 토큰의 위치에 쿼리가 전달되지 않습니다.다음에서는 두 개의 다중 헤드 주의력 가중치가 행별로 표시됩니다.각 헤드는 쿼리, 키 및 값의 개별 표현 하위 공간을 기반으로 독립적으로 참석합니다.

```{.python .input}
#@tab mxnet, tensorflow
d2l.show_heatmaps(
    enc_attention_weights, xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
```

```{.python .input}
#@tab pytorch
d2l.show_heatmaps(
    enc_attention_weights.cpu(), xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))
```

[**디코더 자체 주의 가중치와 인코더-디코더 주의 가중치를 모두 시각화하려면 더 많은 데이터 조작이 필요합니다.**] 예를 들어, 마스크된 주의 가중치를 0으로 채웁니다.디코더 자체 주의 가중치와 인코더-디코더 주의 가중치는 모두 동일한 쿼리를 갖습니다. 시퀀스 시작 토큰과 출력 토큰이 이어집니다.

```{.python .input}
dec_attention_weights_2d = [d2l.tensor(head[0]).tolist()
                            for step in dec_attention_weight_seq
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = d2l.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = d2l.reshape(dec_attention_weights_filled,
                                    (-1, 2, num_layers, num_heads, num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.transpose(1, 2, 3, 0, 4)
dec_self_attention_weights.shape, dec_inter_attention_weights.shape
```

```{.python .input}
#@tab pytorch
dec_attention_weights_2d = [head[0].tolist()
                            for step in dec_attention_weight_seq
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = d2l.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = d2l.reshape(dec_attention_weights_filled,
                                    (-1, 2, num_layers, num_heads, num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.permute(1, 2, 3, 0, 4)
dec_self_attention_weights.shape, dec_inter_attention_weights.shape
```

```{.python .input}
#@tab tensorflow
dec_attention_weights_2d = [head[0] for step in dec_attention_weight_seq
                            for attn in step 
                            for blk in attn for head in blk]
dec_attention_weights_filled = tf.convert_to_tensor(
    np.asarray(pd.DataFrame(dec_attention_weights_2d).fillna(
        0.0).values).astype(np.float32))
dec_attention_weights = tf.reshape(dec_attention_weights_filled, shape=(
    -1, 2, num_layers, num_heads, num_steps))
dec_self_attention_weights, dec_inter_attention_weights = tf.transpose(
    dec_attention_weights, perm=(1, 2, 3, 0, 4))
print(dec_self_attention_weights.shape, dec_inter_attention_weights.shape)
```

디코더 자체 주의의 자동 회귀 속성으로 인해 쿼리 위치 이후에는 키-값 쌍에 대한 쿼리가 수행되지 않습니다.

```{.python .input}
#@tab all
# Plus one to include the beginning-of-sequence token
d2l.show_heatmaps(
    dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
    xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
```

인코더 자체주의의 경우와 마찬가지로, 입력 시퀀스의 지정된 유효 길이를 통해 [**출력 시퀀스의 쿼리가 입력 시퀀스의 패딩 토큰에 참석하지 않습니다.**]

```{.python .input}
#@tab all
d2l.show_heatmaps(
    dec_inter_attention_weights, xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))
```

트랜스포머 아키텍처는 원래 시퀀스 대 시퀀스 학습을 위해 제안되었지만, 이 책의 뒷부분에서 알 수 있듯이 트랜스포머 인코더 또는 트랜스포머 디코더는 종종 서로 다른 딥 러닝 작업에 개별적으로 사용됩니다. 

## 요약

* 트랜스포머는 인코더-디코더 아키텍처의 인스턴스이지만 실제로는 인코더 또는 디코더를 개별적으로 사용할 수 있습니다.
* 트랜스포머에서는 입력 시퀀스와 출력 시퀀스를 나타내는 데 다중 헤드 자체 주의가 사용되지만 디코더는 마스크 버전을 통해 자동 회귀 속성을 보존해야 합니다.
* 변압기의 잔차 연결과 계층 정규화는 모두 매우 심층적인 모델을 훈련시키는 데 중요합니다.
* 트랜스포머 모델의 위치별 피드포워드 네트워크는 동일한 MLP를 사용하여 모든 시퀀스 위치에서 표현을 변환합니다.

## 연습문제

1. 실험에서 더 깊은 변압기를 훈련시킵니다.교육 속도와 번역 성능에 어떤 영향을 미칩니 까?
1. 변압기에서 스케일링된 도트 제품 주의를 부가적인 주의로 대체하는 것이 좋은 생각입니까?왜요?
1. 언어 모델링을 위해 트랜스포머 인코더, 디코더 또는 둘 다를 사용해야 합니까?이 방법을 설계하는 방법은 무엇입니까?
1. 입력 시퀀스가 매우 긴 경우 변압기에 어떤 문제가 될 수 있습니까?왜요?
1. 변압기의 계산 및 메모리 효율성을 향상시키는 방법은 무엇입니까?힌트: you may refer to the survey paper by Tay et al. :cite:`Tay.Dehghani.Bahri.ea.2020`.
1. CNN을 사용하지 않고 이미지 분류 작업을 위한 트랜스포머 기반 모델을 어떻게 설계할 수 있을까요?힌트: you may refer to the vision transformer :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/348)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1066)
:end_tab:
