#  시퀀스 대 시퀀스 학습
:label:`sec_seq2seq`

:numref:`sec_machine_translation`에서 보았 듯이 기계 번역에서는 입력과 출력이 모두 가변 길이 시퀀스입니다.이러한 유형의 문제를 해결하기 위해 :numref:`sec_encoder-decoder`에서 일반 인코더-디코더 아키텍처를 설계했습니다.이 섹션에서는 두 개의 RNN을 사용하여 이 아키텍처의 인코더와 디코더를 설계하고 기계 번역 :cite:`Sutskever.Vinyals.Le.2014,Cho.Van-Merrienboer.Gulcehre.ea.2014`를 위한*시퀀스 대 시퀀스* 학습에 적용합니다. 

인코더-디코더 아키텍처의 설계 원리에 따라 RNN 인코더는 가변 길이 시퀀스를 입력으로 받아 고정 모양의 은닉 상태로 변환 할 수 있습니다.즉, 입력 (소스) 시퀀스의 정보는 RNN 인코더의 은닉 상태에서* 인코딩* 됩니다.토큰에 의해 출력 시퀀스 토큰을 생성하기 위해, 별도의 RNN 디코더는 입력 시퀀스의 인코딩된 정보와 함께 (언어 모델링에서와 같이) 보여지거나 생성된 토큰에 기초하여 다음 토큰을 예측할 수 있다. :numref:`fig_seq2seq`는 시퀀스 대 시퀀스에 대해 두 개의 RNN을 사용하는 방법을 나타낸다.기계 번역 학습. 

![Sequence to sequence learning with an RNN encoder and an RNN decoder.](../img/seq2seq.svg)
:label:`fig_seq2seq`

:numref:`fig_seq2seq`에서 특수 <eos>"“토큰은 시퀀스의 끝을 표시합니다.이 토큰이 생성되면 모델에서 예측을 중단할 수 있습니다.RNN 디코더의 초기 시간 단계에는 두 가지 특별한 설계 결정이 있습니다.첫째, 특수 시퀀스 시작 "<bos>" 토큰이 입력입니다.둘째로, RNN 인코더의 최종 은닉 상태는 디코더의 은닉 상태를 개시하기 위해 사용된다.:cite:`Sutskever.Vinyals.Le.2014`와 같은 설계에서 이것은 인코딩된 입력 시퀀스 정보가 출력 (타겟) 시퀀스를 생성하기 위해 디코더로 공급되는 방식입니다.:cite:`Cho.Van-Merrienboer.Gulcehre.ea.2014`과 같은 일부 다른 설계에서, 인코더의 최종 은닉 상태는 :numref:`fig_seq2seq`에 도시된 바와 같이 매 시간 단계에서 입력의 일부로서 디코더로 공급된다.:numref:`sec_language_model`의 언어 모델 학습과 유사하게 레이블을 하나의 토큰으로 이동 한 원래 출력 시퀀스로 허용 할 수 있습니다. <bos>““, “Ils”, “관련”, “.” $\rightarrow$ “Ils”, “관련”,”.“,<eos>”. 

다음에서는 :numref:`fig_seq2seq`의 디자인에 대해 자세히 설명합니다.:numref:`sec_machine_translation`에 소개된 대로 영어-프랑스어 데이터 세트에서 기계 번역을 위해 이 모델을 교육할 것입니다.

```{.python .input}
import collections
from d2l import mxnet as d2l
import math
from mxnet import np, npx, init, gluon, autograd
from mxnet.gluon import nn, rnn
npx.set_np()
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
import collections
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

## 엔코더

기술적으로 말하면, 인코더는 가변 길이의 입력 시퀀스를 고정 형태*컨텍스트 변수* $\mathbf{c}$로 변환하고 입력 시퀀스 정보를 이 컨텍스트 변수로 인코딩합니다.:numref:`fig_seq2seq`에서 설명한 것처럼 RNN을 사용하여 인코더를 설계할 수 있습니다. 

시퀀스 예제 (배치 크기: 1) 를 고려해 보겠습니다.입력 시퀀스가 $x_1, \ldots, x_T$이고 $x_t$가 입력 텍스트 시퀀스의 $t^{\mathrm{th}}$ 토큰이라고 가정합니다.시간 스텝 ($t$) 에서, RNN은 $x_t$에 대한 입력 특징 벡터 ($\mathbf{x}_t$) 및 이전 시간 단계로부터의 은닉 상태 ($\mathbf{h} _{t-1}$) 를 현재의 은닉 상태 ($\mathbf{h}_t$) 로 변환한다.함수 $f$를 사용하여 RNN의 순환 계층의 변환을 표현할 수 있습니다. 

$$\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1}). $$

일반적으로 인코더는 사용자 정의 함수 $q$를 통해 모든 시간 단계에서 숨겨진 상태를 컨텍스트 변수로 변환합니다. 

$$\mathbf{c} =  q(\mathbf{h}_1, \ldots, \mathbf{h}_T).$$

예를 들어 :numref:`fig_seq2seq`에서와 같이 $q(\mathbf{h}_1, \ldots, \mathbf{h}_T) = \mathbf{h}_T$를 선택하는 경우 컨텍스트 변수는 마지막 시간 단계에서 입력 시퀀스의 숨겨진 상태 $\mathbf{h}_T$에 불과합니다. 

지금까지 단방향 RNN을 사용하여 인코더를 설계했습니다. 여기서 은닉 상태는 은닉 상태의 시간 단계 이전과 그 이전의 입력 서브 시퀀스에만 의존합니다.양방향 RNN을 사용하여 인코더를 구성할 수도 있습니다.이 경우 숨겨진 상태는 전체 시퀀스의 정보를 인코딩하는 시간 단계 전후의 하위 시퀀스 (현재 시간 스텝의 입력 포함) 에 따라 달라집니다. 

이제 [**RNN 인코더 구현**] 을 하겠습니다.*embedding 계층*을 사용하여 입력 시퀀스의 각 토큰에 대한 특징 벡터를 얻습니다.임베딩 레이어의 가중치는 행 수가 입력 어휘의 크기 (`vocab_size`) 와 같고 열 수가 특징 벡터의 차원 (`embed_size`) 과 동일한 행렬입니다.임의의 입력 토큰 인덱스 $i$에 대해 임베딩 계층은 가중치 행렬의 $i^{\mathrm{th}}$ 행 (0부터 시작) 을 가져와 특징 벡터를 반환합니다.또한 여기서는 엔코더를 구현하기 위해 다층 GRU를 선택합니다.

```{.python .input}
#@save
class Seq2SeqEncoder(d2l.Encoder):
    """The RNN encoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X = self.embedding(X)
        # In RNN models, the first axis corresponds to time steps
        X = X.swapaxes(0, 1)
        state = self.rnn.begin_state(batch_size=X.shape[1], ctx=X.ctx)
        output, state = self.rnn(X, state)
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state[0]` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
```

```{.python .input}
#@tab pytorch
#@save
class Seq2SeqEncoder(d2l.Encoder):
    """The RNN encoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X = self.embedding(X)
        # In RNN models, the first axis corresponds to time steps
        X = X.permute(1, 0, 2)
        # When state is not mentioned, it defaults to zeros
        output, state = self.rnn(X)
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
```

```{.python .input}
#@tab tensorflow
#@save
class Seq2SeqEncoder(d2l.Encoder):
    """The RNN encoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs): 
        super().__init__(*kwargs)
        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
             for _ in range(num_layers)]), return_sequences=True,
                                       return_state=True)
    
    def call(self, X, *args, **kwargs):
        # The input `X` shape: (`batch_size`, `num_steps`)
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X = self.embedding(X)
        output = self.rnn(X, **kwargs)
        state = output[1:]
        return output[0], state
```

순환 계층의 반환된 변수는 :numref:`sec_rnn-concise`에 설명되어 있습니다.구체적인 예를 사용하여 [**위의 인코더 구현을 설명하겠습니다.**] 아래에서는 은닉 유닛 수가 16인 2 레이어 GRU 인코더를 인스턴스화합니다.시퀀스 입력값 `X` (배치 크기: 4, 시간 스텝 수: 7) 의 미니 배치가 주어지면 모든 시간 스텝에서 마지막 레이어의 은닉 상태 (인코더의 반복 레이어에 의해 반환되는 `output`) 는 모양의 텐서 (시간 스텝 수, 배치 크기, 은닉 유닛 수) 입니다.

```{.python .input}
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.initialize()
X = d2l.zeros((4, 7))
output, state = encoder(X)
output.shape
```

```{.python .input}
#@tab pytorch
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.eval()
X = d2l.zeros((4, 7), dtype=torch.long)
output, state = encoder(X)
output.shape
```

```{.python .input}
#@tab tensorflow
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
X = tf.zeros((4, 7))
output, state = encoder(X, training=False)
output.shape
```

여기에 GRU가 사용되므로 마지막 시간 단계에서 다층 은닉 상태의 모양은 (은닉 레이어 수, 배치 크기, 은닉 유닛 수) 입니다.LSTM을 사용하는 경우 메모리 셀 정보도 `state`에 포함됩니다.

```{.python .input}
len(state), state[0].shape
```

```{.python .input}
#@tab pytorch
state.shape
```

```{.python .input}
#@tab tensorflow
len(state), [element.shape for element in state]
```

## [**디코더**]
:label:`sec_seq2seq_decoder`

앞에서 언급했듯이 엔코더 출력의 컨텍스트 변수 $\mathbf{c}$은 전체 입력 시퀀스 $x_1, \ldots, x_T$를 인코딩합니다.트레이닝 데이터셋으로부터의 출력 시퀀스 ($y_1, y_2, \ldots, y_{T'}$) 가 주어지면, 각 시간 스텝 ($t'$) (심볼은 입력 시퀀스 또는 인코더의 시간 스텝 $t$와 상이함) 에 대해, 디코더 출력 ($y_{t'}$) 의 확률은 이전 출력 서브시퀀스 ($y_1, \ldots, y_{t'-1}$) 및 컨텍스트 변수에 조건부이다.$\mathbf{c}$, 즉 $P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$. 

시퀀스에서 이 조건부 확률을 모델링하기 위해 다른 RNN을 디코더로 사용할 수 있습니다.출력 시퀀스상의 임의의 시간 스텝 ($t^\prime$) 에서, RNN은 이전 시간 스텝으로부터의 출력 $y_{t^\prime-1}$ 및 컨텍스트 변수 $\mathbf{c}$를 입력으로서 취한 다음, 이들 및 이전의 은닉 상태 ($\mathbf{s}_{t^\prime-1}$) 를 현재 시간 스텝에서 은닉 상태 ($\mathbf{s}_{t^\prime}$) 로 변환한다.결과적으로 함수 $g$을 사용하여 디코더의 숨겨진 계층의 변환을 표현할 수 있습니다. 

$$\mathbf{s}_{t^\prime} = g(y_{t^\prime-1}, \mathbf{c}, \mathbf{s}_{t^\prime-1}).$$
:eqlabel:`eq_seq2seq_s_t`

디코더의 숨겨진 상태를 얻은 후 출력 계층과 소프트맥스 연산을 사용하여 시간 단계 $t^\prime$에서 출력에 대한 조건부 확률 분포 $P(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \mathbf{c})$를 계산할 수 있습니다. 

:numref:`fig_seq2seq`에 이어 다음과 같이 디코더를 구현할 때 인코더의 마지막 시간 단계에서 숨겨진 상태를 직접 사용하여 디코더의 숨겨진 상태를 초기화합니다.이를 위해서는 RNN 인코더와 RNN 디코더가 동일한 수의 레이어와 은닉 유닛을 가져야 합니다.인코딩된 입력 시퀀스 정보를 더 통합하기 위해, 컨텍스트 변수는 모든 시간 스텝에서 디코더 입력과 연결된다.출력 토큰의 확률 분포를 예측하기 위해 완전히 연결된 계층을 사용하여 RNN 디코더의 최종 계층에서 은닉 상태를 변환합니다.

```{.python .input}
class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # The output `X` shape: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).swapaxes(0, 1)
        # `context` shape: (`batch_size`, `num_hiddens`)
        context = state[0][-1]
        # Broadcast `context` so it has the same `num_steps` as `X`
        context = np.broadcast_to(context, (
            X.shape[0], context.shape[0], context.shape[1]))
        X_and_context = d2l.concat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).swapaxes(0, 1)
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state[0]` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
```

```{.python .input}
#@tab pytorch
class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # The output `X` shape: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).permute(1, 0, 2)
        # Broadcast `context` so it has the same `num_steps` as `X`
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = d2l.concat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
```

```{.python .input}
#@tab tensorflow
class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
             for _ in range(num_layers)]), return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
        
    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]
    
    def call(self, X, state, **kwargs):
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X = self.embedding(X)
        # Broadcast `context` so it has the same `num_steps` as `X`
        context = tf.repeat(tf.expand_dims(state[-1], axis=1), repeats=X.shape[1], axis=1)
        X_and_context = tf.concat((X, context), axis=2)
        rnn_output = self.rnn(X_and_context, state, **kwargs)
        output = self.dense(rnn_output[0])
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state` is a list with `num_layers` entries. Each entry has shape:
        # (`batch_size`, `num_hiddens`)
        return output, rnn_output[1:]
```

[**구현된 디코더를 설명**] 하기 위해 아래에서 앞서 언급한 인코더의 동일한 하이퍼파라미터로 인스턴스화합니다.보시다시피 디코더의 출력 모양은 (배치 크기, 시간 단계 수, 어휘 크기) 가되어 텐서의 마지막 차원이 예측 된 토큰 분포를 저장합니다.

```{.python .input}
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.initialize()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
output.shape, len(state), state[0].shape
```

```{.python .input}
#@tab pytorch
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
output.shape, state.shape
```

```{.python .input}
#@tab tensorflow
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
state = decoder.init_state(encoder(X))
output, state = decoder(X, state, training=False)
output.shape, len(state), state[0].shape
```

요약하면, 위의 RNN 인코더-디코더 모델의 계층은 :numref:`fig_seq2seq_details`에 설명되어 있습니다. 

![Layers in an RNN encoder-decoder model.](../img/seq2seq-details.svg)
:label:`fig_seq2seq_details`

## 손실 함수

각 시간 스텝에서, 디코더는 출력 토큰에 대한 확률 분포를 예측한다.언어 모델링과 마찬가지로 softmax를 적용하여 분포를 얻고 최적화를 위해 교차 엔트로피 손실을 계산할 수 있습니다.:numref:`sec_machine_translation`는 시퀀스의 끝에 특수 패딩 토큰이 추가되어 다양한 길이의 시퀀스를 동일한 모양의 미니 배치로 효율적으로 로드할 수 있습니다.그러나 패딩 토큰의 예측은 손실 계산에서 제외해야 합니다. 

이를 위해 다음 `sequence_mask` 함수를 사용하여 [**0 값으로 관련없는 항목을 마스킹**] 할 수 있으므로 나중에 0인 관련없는 예측을 곱하면 0이 됩니다.예를 들어 패딩 토큰을 제외한 두 시퀀스의 유효한 길이가 각각 1과 2인 경우 첫 번째 항목과 처음 두 항목 뒤에 있는 나머지 항목은 0으로 지워집니다.

```{.python .input}
X = np.array([[1, 2, 3], [4, 5, 6]])
npx.sequence_mask(X, np.array([1, 2]), True, axis=1)
```

```{.python .input}
#@tab pytorch
#@save
def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
sequence_mask(X, torch.tensor([1, 2]))
```

```{.python .input}
#@tab tensorflow
#@save
def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.shape[1]
    mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[
        None, :] < tf.cast(valid_len[:, None], dtype=tf.float32)
    
    if len(X.shape) == 3:
        return tf.where(tf.expand_dims(mask, axis=-1), X, value)
    else:
        return tf.where(mask, X, value)
    
X = tf.constant([[1, 2, 3], [4, 5, 6]])
sequence_mask(X, tf.constant([1, 2]))
```

(**마지막 몇 개의 축에서 모든 항목을 마스킹할 수도 있습니다.**) 원하는 경우 이러한 항목을 0이 아닌 값으로 바꾸도록 지정할 수도 있습니다.

```{.python .input}
X = d2l.ones((2, 3, 4))
npx.sequence_mask(X, np.array([1, 2]), True, value=-1, axis=1)
```

```{.python .input}
#@tab pytorch
X = d2l.ones(2, 3, 4)
sequence_mask(X, torch.tensor([1, 2]), value=-1)
```

```{.python .input}
#@tab tensorflow
X = tf.ones((2,3,4))
sequence_mask(X, tf.constant([1, 2]), value=-1)
```

이제 관련없는 예측을 마스킹할 수 있도록 [**소프트맥스 교차 엔트로피 손실을 확장할 수 있습니다.**] 처음에는 예측된 모든 토큰에 대한 마스크가 1로 설정됩니다.유효한 길이가 주어지면 패딩 토큰에 해당하는 마스크가 0으로 지워집니다.결국 모든 토큰의 손실에 마스크가 곱해져서 손실에서 패딩 토큰의 관련없는 예측을 걸러냅니다.

```{.python .input}
#@save
class MaskedSoftmaxCELoss(gluon.loss.SoftmaxCELoss):
    """The softmax cross-entropy loss with masks."""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        # `weights` shape: (`batch_size`, `num_steps`, 1)
        weights = np.expand_dims(np.ones_like(label), axis=-1)
        weights = npx.sequence_mask(weights, valid_len, True, axis=1)
        return super(MaskedSoftmaxCELoss, self).forward(pred, label, weights)
```

```{.python .input}
#@tab pytorch
#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks."""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
```

```{.python .input}
#@tab tensorflow
#@save
class MaskedSoftmaxCELoss(tf.keras.losses.Loss):
    """The softmax cross-entropy loss with masks."""
    def __init__(self, valid_len):
        super().__init__(reduction='none')
        self.valid_len = valid_len
    
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def call(self, label, pred):
        weights = tf.ones_like(label, dtype=tf.float32)
        weights = sequence_mask(weights, self.valid_len)
        label_one_hot = tf.one_hot(label, depth=pred.shape[-1])
        unweighted_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction='none')(label_one_hot, pred)
        weighted_loss = tf.reduce_mean((unweighted_loss*weights), axis=1)
        return weighted_loss
```

[**a 온전성 검사**] 의 경우, 세 개의 동일한 시퀀스를 생성할 수 있습니다.그런 다음 이러한 시퀀스의 유효한 길이가 각각 4, 2, 0임을 지정할 수 있습니다.따라서 첫 번째 시퀀스의 손실은 두 번째 시퀀스의 손실보다 두 배 커야 하고 세 번째 시퀀스의 손실은 0이어야 합니다.

```{.python .input}
loss = MaskedSoftmaxCELoss()
loss(d2l.ones((3, 4, 10)), d2l.ones((3, 4)), np.array([4, 2, 0]))
```

```{.python .input}
#@tab pytorch
loss = MaskedSoftmaxCELoss()
loss(d2l.ones(3, 4, 10), d2l.ones((3, 4), dtype=torch.long),
     torch.tensor([4, 2, 0]))
```

```{.python .input}
#@tab tensorflow
loss = MaskedSoftmaxCELoss(tf.constant([4, 2, 0]))
loss(tf.ones((3,4), dtype = tf.int32), tf.ones((3, 4, 10))).numpy()
```

## [**교육**]
:label:`sec_seq2seq_training`

다음 훈련 루프에서는 :numref:`fig_seq2seq`와 같이 특수 시퀀스 시작 토큰과 최종 토큰을 제외한 원래 출력 시퀀스를 디코더의 입력으로 연결합니다.원래 출력 시퀀스 (토큰 레이블) 가 디코더에 공급되기 때문에*교사 강제*라고합니다.또는 이전 시간 스텝에서*predicted* 토큰을 디코더에 대한 현재 입력으로 공급할 수도 있습니다.

```{.python .input}
#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence."""
    net.initialize(init.Xavier(), force_reinit=True, ctx=device)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    loss = MaskedSoftmaxCELoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [
                x.as_in_ctx(device) for x in batch]
            bos = np.array(
                [tgt_vocab['<bos>']] * Y.shape[0], ctx=device).reshape(-1, 1)
            dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            with autograd.record():
                Y_hat, _ = net(X, dec_input, X_valid_len)
                l = loss(Y_hat, Y, Y_valid_len)
            l.backward()
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            trainer.step(num_tokens)
            metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence."""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
```

```{.python .input}
#@tab tensorflow
#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    animator = d2l.Animator(xlabel="epoch", ylabel="loss",
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x for x in batch]
            bos = tf.reshape(tf.constant([tgt_vocab['<bos>']] * Y.shape[0]),
                             shape=(-1, 1))
            dec_input = tf.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            with tf.GradientTape() as tape:
                Y_hat, _ = net(X, dec_input, X_valid_len, training=True)
                l = MaskedSoftmaxCELoss(Y_valid_len)(Y, Y_hat)
            gradients = tape.gradient(l, net.trainable_variables)
            gradients = d2l.grad_clipping(gradients, 1)
            optimizer.apply_gradients(zip(gradients, net.trainable_variables))
            num_tokens = tf.reduce_sum(Y_valid_len).numpy()
            metric.add(tf.reduce_sum(l), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
```

이제 기계 번역 데이터 세트에서 시퀀스 대 시퀀스 학습을 위해 [**RNN 인코더-디코더 모델을 만들고 훈련**] 할 수 있습니다.

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

## [**예측**]

토큰별로 출력 시퀀스 토큰을 예측하기 위해, 각 디코더 시간 스텝에서 이전 시간 스텝으로부터 예측된 토큰이 입력으로서 디코더에 공급된다.훈련과 마찬가지로 초기 시간 단계에서 시퀀스 시작 (” <bos>“) 토큰이 디코더에 공급됩니다.이 예측 과정은 :numref:`fig_seq2seq_predict`에 설명되어 있습니다.시퀀스 끝 (” <eos>“) 토큰이 예측되면 출력 시퀀스의 예측이 완료됩니다. 

![Predicting the output sequence token by token using an RNN encoder-decoder.](../img/seq2seq-predict.svg)
:label:`fig_seq2seq_predict`

:numref:`sec_beam-search`에서는 시퀀스 생성을 위한 다양한 전략을 소개합니다.

```{.python .input}
#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Predict for sequence to sequence."""
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = np.array([len(src_tokens)], ctx=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = np.expand_dims(np.array(src_tokens, ctx=device), axis=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = np.expand_dims(np.array([tgt_vocab['<bos>']], ctx=device), axis=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(axis=2)
        pred = dec_X.squeeze(axis=0).astype('int32').item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
```

```{.python .input}
#@tab pytorch
#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Predict for sequence to sequence."""
    # Set `net` to eval mode for inference
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
```

```{.python .input}
#@tab tensorflow
#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    save_attention_weights=False):
    """Predict for sequence to sequence."""
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = tf.constant([len(src_tokens)])
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = tf.expand_dims(src_tokens, axis=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len, training=False)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = tf.expand_dims(tf.constant([tgt_vocab['<bos>']]), axis=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state, training=False)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = tf.argmax(Y, axis=2)
        pred = tf.squeeze(dec_X, axis=0)
        # Save attention weights
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred.numpy())
    return ' '.join(tgt_vocab.to_tokens(tf.reshape(output_seq, shape = -1).numpy().tolist())), attention_weight_seq
```

## 예측된 시퀀스 평가

예측 시퀀스를 레이블 시퀀스 (실측) 와 비교하여 평가할 수 있습니다.BLEU (이중 언어 평가 대역 연구) 는 원래 기계 번역 결과 :cite:`Papineni.Roukos.Ward.ea.2002`를 평가하기 위해 제안되었지만 다양한 응용 분야의 출력 시퀀스 품질을 측정하는 데 광범위하게 사용되었습니다.원칙적으로 예측된 서열의 $n$그램에 대해 BLEU는 이 $n$그램이 라벨 시퀀스에 나타나는지 여부를 평가합니다. 

$n$그램의 정밀도를 $n$그램으로 나타냅니다. 이 정밀도는 예측 및 레이블 시퀀스에서 일치하는 $n$그램의 수와 예측된 시퀀스의 $n$그램 수의 비율입니다.설명하기 위해 레이블 시퀀스 $A$, $B$, $C$, $D$, $E$, $F$ 및 예측된 시퀀스 $A$, $B$, $B$, $B$, $C$, $D$가 있습니다. 20, 그리고 $p_4 = 0$입니다.또한 $\mathrm{len}_{\text{label}}$ 및 $\mathrm{len}_{\text{pred}}$을 각각 레이블 시퀀스와 예측 시퀀스의 토큰 수로 지정합니다.그런 다음 BLEU는 다음과 같이 정의됩니다. 

$$ \exp\left(\min\left(0, 1 - \frac{\mathrm{len}_{\text{label}}}{\mathrm{len}_{\text{pred}}}\right)\right) \prod_{n=1}^k p_n^{1/2^n},$$
:eqlabel:`eq_bleu`

여기서 $k$는 매칭을 위해 가장 긴 $n$그램입니다. 

:eqref:`eq_bleu`의 BLEU 정의에 따르면, 예측된 시퀀스가 레이블 시퀀스와 같을 때마다 BLEU는 1입니다.또한 더 긴 $n$그램을 매칭하는 것이 더 어렵 기 때문에 BLEU는 더 긴 $n$g의 정밀도에 더 큰 무게를 할당합니다.특히 $p_n$가 고정되면 $n$가 커짐에 따라 $p_n^{1/2^n}$이 증가합니다 (원본 용지는 $p_n^{1/n}$을 사용함).또한 더 짧은 시퀀스를 예측하면 더 높은 $p_n$ 값을 얻는 경향이 있으므로 :eqref:`eq_bleu`의 곱셈 항 앞의 계수는 더 짧은 예측 시퀀스에 불이익을 줍니다.예를 들어, $k=2$인 경우 레이블 시퀀스 $A$, $B$, $C$, $D$, $D$, $F$ 및 예측된 시퀀스 $A$, $B$이 주어진 경우 페널티 계수 $\exp(1-6/2) \approx 0.14$은 블루를 낮춥니다. 

다음과 같이 [**BLEU 측정값** 구현] 합니다.

```{.python .input}
#@tab all
def bleu(pred_seq, label_seq, k):  #@save
    """Compute the BLEU."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```

결국 훈련된 RNN 인코더-디코더를 사용하여 [**몇 개의 영어 문장을 프랑스어로 번역**] 하고 결과의 BLEU를 계산합니다.

```{.python .input}
#@tab mxnet, pytorch
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
```

```{.python .input}
#@tab tensorflow
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
```

## 요약

* 인코더-디코더 아키텍처의 설계에 따라 두 개의 RNN을 사용하여 시퀀스 대 시퀀스 학습을위한 모델을 설계 할 수 있습니다.
* 인코더와 디코더를 구현할 때 다층 RNN을 사용할 수 있습니다.
* 마스크를 사용하여 손실을 계산할 때와 같이 관련 없는 계산을 필터링할 수 있습니다.
* 인코더-디코더 교육에서 교사 강제 접근 방식은 원래 출력 시퀀스 (예측과 달리) 를 디코더에 공급합니다.
* BLEU는 예측된 시퀀스와 레이블 시퀀스 간에 $n$g을 일치시켜 출력 시퀀스를 평가하는 데 널리 사용되는 측정값입니다.

## 연습문제

1. 하이퍼파라미터를 조정하여 번역 결과를 개선할 수 있습니까?
1. 손실 계산에 마스크를 사용하지 않고 실험을 다시 실행합니다.어떤 결과를 보십니까?왜요?
1. 인코더와 디코더가 레이어 수 또는 은닉 유닛 수가 다른 경우 디코더의 숨겨진 상태를 어떻게 초기화 할 수 있습니까?
1. 교육에서 교사 강제를 이전 시간 단계의 예측을 디코더에 공급하는 것으로 대체하십시오.이것이 성능에 어떤 영향을 미칩니 까?
1. GRU를 LSTM으로 대체하여 실험을 다시 실행합니다.
1. 디코더의 출력 계층을 설계하는 다른 방법이 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/345)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1062)
:end_tab:
