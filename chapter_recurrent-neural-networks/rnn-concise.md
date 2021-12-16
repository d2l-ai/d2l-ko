# 순환 신경망의 간결한 구현
:label:`sec_rnn-concise`

:numref:`sec_rnn_scratch`는 RNN이 어떻게 구현되는지 확인하는 데 도움이 되었지만 편리하거나 빠르지는 않습니다.이 섹션에서는 딥러닝 프레임워크의 상위 수준 API에서 제공하는 함수를 사용하여 동일한 언어 모델을 보다 효율적으로 구현하는 방법을 보여줍니다.이전처럼 타임머신 데이터 세트를 읽는 것으로 시작합니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

## [**모델 정의**]

상위 수준 API는 순환 신경망의 구현을 제공합니다.단일 은닉 레이어와 256 개의 숨겨진 유닛으로 순환 신경망 계층 `rnn_layer`를 구성합니다.실제로 여러 레이어를 갖는 것이 무엇을 의미하는지 아직 논의하지 않았습니다. 이는 :numref:`sec_deep_rnn`에서 발생할 것입니다.지금은 여러 계층이 RNN의 다음 계층에 대한 입력으로 사용되는 RNN의 한 계층의 출력에 해당한다고 가정하면 충분합니다.

```{.python .input}
num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()
```

```{.python .input}
#@tab pytorch
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)
```

```{.python .input}
#@tab tensorflow
num_hiddens = 256
rnn_cell = tf.keras.layers.SimpleRNNCell(num_hiddens,
    kernel_initializer='glorot_uniform')
rnn_layer = tf.keras.layers.RNN(rnn_cell, time_major=True,
    return_sequences=True, return_state=True)
```

:begin_tab:`mxnet`
숨겨진 상태를 초기화하는 것은 간단합니다.멤버 함수 `begin_state`를 호출합니다.그러면 미니 배치의 각 예제에 대한 초기 은닉 상태가 포함된 목록 (`state`) 이 반환되며, 그 모양은 (은닉 레이어 수, 배치 크기, 은닉 유닛 수) 입니다.나중에 소개될 일부 모델 (예: 장단기 기억) 의 경우 이러한 목록에는 다른 정보도 포함되어 있습니다.
:end_tab:

:begin_tab:`pytorch`
우리는 (** 텐서를 사용하여 은닉 상태를 초기화**), 그 모양은 (은닉 레이어 수, 배치 크기, 은닉 유닛 수) 입니다.
:end_tab:

```{.python .input}
state = rnn_layer.begin_state(batch_size=batch_size)
len(state), state[0].shape
```

```{.python .input}
#@tab pytorch
state = torch.zeros((1, batch_size, num_hiddens))
state.shape
```

```{.python .input}
#@tab tensorflow
state = rnn_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
state.shape
```

[**은닉 상태와 입력을 사용하면 업데이트된 은닉 상태로 출력을 계산할 수 있습니다.**] `rnn_layer`의 “출력” (`Y`) 은 출력 레이어의 계산을 포함하지 않습니다*아닙니다*: *각* 시간 단계에서 은닉 상태를 참조하며후속 출력 계층입니다.

:begin_tab:`mxnet`
또한 `rnn_layer`에서 반환한 업데이트된 숨겨진 상태 (`state_new`) 는 미니배치의*마지막* 시간 단계에서 숨겨진 상태를 나타냅니다.순차적 파티셔닝에서 에포크 내에 다음 미니배치의 숨겨진 상태를 초기화하는 데 사용할 수 있습니다.숨겨진 레이어가 여러 개인 경우 각 레이어의 숨겨진 상태가 이 변수 (`state_new`) 에 저장됩니다.나중에 도입될 일부 모델 (예: 장단기 기억) 의 경우 이 변수에는 다른 정보도 포함됩니다.
:end_tab:

```{.python .input}
X = np.random.uniform(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, state_new.shape
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

:numref:`sec_rnn_scratch`와 유사하게, [**완전한 RNN 모델에 대해 `RNNModel` 클래스를 정의합니다.**] `rnn_layer`에는 숨겨진 순환 레이어만 포함되어 있으므로 별도의 출력 레이어를 만들어야 합니다.

```{.python .input}
#@save
class RNNModel(nn.Block):
    """The RNN model."""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = npx.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # The fully-connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `vocab_size`).
        output = self.dense(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

```{.python .input}
#@tab pytorch
#@save
class RNNModel(nn.Module):
    """The RNN model."""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # If the RNN is bidirectional (to be introduced later),
        # `num_directions` should be 2, else it should be 1.
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `vocab_size`).
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # `nn.LSTM` takes a tuple of hidden states
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))
```

```{.python .input}
#@tab tensorflow
#@save
class RNNModel(tf.keras.layers.Layer):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, state):
        X = tf.one_hot(tf.transpose(inputs), self.vocab_size)
        # Later RNN like `tf.keras.layers.LSTMCell` return more than two values
        Y, *state = self.rnn(X, state)
        output = self.dense(tf.reshape(Y, (-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.cell.get_initial_state(*args, **kwargs)
```

## 교육 및 예측

모델을 훈련시키기 전에 [**랜덤 가중치를 갖는 모델을 사용하여 예측해보십시오.**]

```{.python .input}
device = d2l.try_gpu()
net = RNNModel(rnn_layer, len(vocab))
net.initialize(force_reinit=True, ctx=device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
```

```{.python .input}
#@tab pytorch
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
```

```{.python .input}
#@tab tensorflow
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    net = RNNModel(rnn_layer, vocab_size=len(vocab))

d2l.predict_ch8('time traveller', 10, net, vocab)
```

명백하게, 이 모델은 전혀 작동하지 않습니다.다음으로 :numref:`sec_rnn_scratch`에 정의된 동일한 하이퍼파라미터를 사용하여 `train_ch8`를 호출하고 [**상위 수준 API로 모델 훈련**] 을 수행합니다.

```{.python .input}
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, strategy)
```

마지막 섹션과 비교할 때, 이 모델은 딥 러닝 프레임워크의 상위 수준 API에 의해 코드가 더 최적화되기 때문에 짧은 기간 내에 비슷한 난처함을 달성합니다. 

## 요약

* 딥러닝 프레임워크의 상위 수준 API는 RNN 계층의 구현을 제공합니다.
* 상위 수준 API의 RNN 계층은 출력 및 업데이트된 은닉 상태를 반환합니다. 여기서 출력에는 출력 계층 계산이 포함되지 않습니다.
* 높은 수준의 API를 사용하면 처음부터 구현을 사용하는 것보다 RNN 훈련이 더 빨라집니다.

## 연습문제

1. 상위 수준 API를 사용하여 RNN 모델을 과적합하게 만들 수 있습니까?
1. RNN 모델에서 은닉 레이어 수를 늘리면 어떻게 됩니까?모델을 작동시킬 수 있습니까?
1. RNN을 사용하여 :numref:`sec_sequence`의 자기 회귀 모델을 구현합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/335)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1053)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2211)
:end_tab:
