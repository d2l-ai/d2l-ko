# 장단기 기억 (LSTM)
:label:`sec_lstm`

잠재 변수 모델에서 장기적인 정보 보존 및 단기 입력 건너 뛰기를 해결해야 하는 과제는 오랫동안 존재해 왔습니다.이를 해결하기 위한 가장 초기의 접근법 중 하나는 장단기 기억 (LSTM) :cite:`Hochreiter.Schmidhuber.1997`였습니다.GRU의 많은 특성을 공유합니다.흥미롭게도 LSTM은 GRU보다 약간 더 복잡한 설계를 가지고 있지만 거의 20년 전에 GRU보다 우선합니다. 

## 게이트 메모리 셀

틀림없이 LSTM의 디자인은 컴퓨터의 논리 게이트에서 영감을 얻었습니다.LSTM은 추가 정보를 기록하도록 설계된 숨겨진 상태 (일부 문헌에서는 메모리 셀을 숨겨진 상태의 특수 유형으로 간주 함) 와 동일한 모양을 가진*메모리 셀* (또는 줄임으로*셀*) 를 소개합니다.메모리 셀을 제어하려면 여러 게이트가 필요합니다.셀에서 항목을 읽으려면 하나의 게이트가 필요합니다.우리는 이것을
*출력 게이트*.
데이터를 셀로 읽을 시기를 결정하려면 두 번째 게이트가 필요합니다.이를 *입력 게이트*라고 합니다.마지막으로*forget gate*에 의해 제어되는 셀의 내용을 재설정하는 메커니즘이 필요합니다.이러한 설계의 동기는 GRU의 동기와 동일합니다. 즉, 전용 메커니즘을 통해 은닉 상태의 입력을 언제 기억하고 무시할 것인지를 결정할 수 있습니다.이것이 실제로 어떻게 작동하는지 봅시다. 

### 입력 게이트, 포겟 게이트 및 출력 게이트

GRU와 마찬가지로 LSTM 게이트로 공급되는 데이터는 :numref:`lstm_0`에 나와 있는 것처럼 현재 시간 스텝의 입력과 이전 시간 스텝의 숨겨진 상태입니다.입력, forget. 및 출력 게이트의 값을 계산하기 위해 시그모이드 활성화 함수를 사용하여 완전히 연결된 세 개의 계층에 의해 처리됩니다.결과적으로 세 게이트의 값은 $(0, 1)$의 범위에 있습니다. 

![Computing the input gate, the forget gate, and the output gate in an LSTM model.](../img/lstm-0.svg)
:label:`lstm_0`

수학적으로 $h$개의 은닉 단위가 있고 배치 크기가 $n$이며 입력 개수가 $d$라고 가정합니다.따라서 입력은 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$이고 이전 시간 스텝의 숨겨진 상태는 $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$입니다.이에 대응하여, 시간 스텝 $t$에서의 게이트는 다음과 같이 정의된다: 입력 게이트는 $\mathbf{I}_t \in \mathbb{R}^{n \times h}$이고, 포겟 게이트는 $\mathbf{F}_t \in \mathbb{R}^{n \times h}$이며, 출력 게이트는 $\mathbf{O}_t \in \mathbb{R}^{n \times h}$이다.다음과 같이 계산됩니다. 

$$
\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xi} + \mathbf{H}_{t-1} \mathbf{W}_{hi} + \mathbf{b}_i),\\
\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xf} + \mathbf{H}_{t-1} \mathbf{W}_{hf} + \mathbf{b}_f),\\
\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xo} + \mathbf{H}_{t-1} \mathbf{W}_{ho} + \mathbf{b}_o),
\end{aligned}
$$

여기서 $\mathbf{W}_{xi}, \mathbf{W}_{xf}, \mathbf{W}_{xo} \in \mathbb{R}^{d \times h}$ 및 $\mathbf{W}_{hi}, \mathbf{W}_{hf}, \mathbf{W}_{ho} \in \mathbb{R}^{h \times h}$는 가중치 매개변수이고 $\mathbf{b}_i, \mathbf{b}_f, \mathbf{b}_o \in \mathbb{R}^{1 \times h}$는 편향 매개변수입니다. 

### 후보 메모리 셀

다음으로 메모리 셀을 설계합니다.아직 다양한 게이트의 동작을 지정하지 않았으므로 먼저*후보* 메모리 셀 $\tilde{\mathbf{C}}_t \in \mathbb{R}^{n \times h}$을 소개합니다.계산은 위에서 설명한 세 게이트의 계산과 유사하지만 $(-1, 1)$에 대한 값 범위를 가진 $\tanh$ 함수를 활성화 함수로 사용합니다.이렇게 하면 시간 단계 $t$에서 다음 방정식이 생성됩니다. 

$$\tilde{\mathbf{C}}_t = \text{tanh}(\mathbf{X}_t \mathbf{W}_{xc} + \mathbf{H}_{t-1} \mathbf{W}_{hc} + \mathbf{b}_c),$$

여기서 $\mathbf{W}_{xc} \in \mathbb{R}^{d \times h}$ 및 $\mathbf{W}_{hc} \in \mathbb{R}^{h \times h}$는 가중치 매개변수이고 $\mathbf{b}_c \in \mathbb{R}^{1 \times h}$은 편향 매개변수입니다. 

후보 메모리 셀의 빠른 그림은 :numref:`lstm_1`에 나와 있습니다. 

![Computing the candidate memory cell in an LSTM model.](../img/lstm-1.svg)
:label:`lstm_1`

### 메모리 셀

GRU에는 입력과 잊기 (또는 건너뛰기) 를 제어하는 메커니즘이 있습니다.마찬가지로 LSTM에는 이러한 목적을 위해 두 개의 전용 게이트가 있습니다. 입력 게이트 $\mathbf{I}_t$은 $\tilde{\mathbf{C}}_t$을 통해 새 데이터를 고려하는 양을 제어하고 잊어 버린 게이트 $\mathbf{F}_t$는 이전 메모리 셀 내용 $\mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}$의 양을 처리합니다.이전과 동일한 점별 곱셈 트릭을 사용하여 다음 업데이트 방정식에 도달합니다. 

$$\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t.$$

만약 포겟 게이트가 항상 대략 1이고 입력 게이트가 항상 대략 0이면, 과거의 메모리 셀 ($\mathbf{C}_{t-1}$) 은 시간에 따라 저장되고 현재 시간 스텝으로 전달될 것이다.이 설계는 사라지는 기울기 문제를 완화하고 시퀀스 내에서 장거리 종속성을 더 잘 캡처하기 위해 도입되었습니다. 

따라서 우리는 :numref:`lstm_2`의 흐름도에 도달합니다. 

![Computing the memory cell in an LSTM model.](../img/lstm-2.svg)

:label:`lstm_2` 

### 히든 스테이트

마지막으로 숨겨진 상태 $\mathbf{H}_t \in \mathbb{R}^{n \times h}$를 계산하는 방법을 정의해야 합니다.여기서 출력 게이트가 작동합니다.LSTM에서는 단순히 메모리 셀의 $\tanh$의 게이트 버전입니다.이렇게 하면 $\mathbf{H}_t$의 값이 항상 구간 $(-1, 1)$에 있게 됩니다. 

$$\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t).$$

출력 게이트가 1에 가까워질 때마다 모든 메모리 정보를 예측자로 효과적으로 전달하는 반면, 0에 가까운 출력 게이트의 경우 메모리 셀 내에서만 모든 정보를 유지하고 추가 처리를 수행하지 않습니다. 

:numref:`lstm_3`에는 데이터 흐름을 그래픽으로 보여 줍니다. 

![Computing the hidden state in an LSTM model.](../img/lstm-3.svg)
:label:`lstm_3`

## 처음부터 구현

이제 LSTM을 처음부터 구현해 보겠습니다.:numref:`sec_rnn_scratch`의 실험과 마찬가지로 먼저 타임머신 데이터세트를 로드합니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

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

### [**모델 매개변수 초기화**]

다음으로 모델 매개변수를 정의하고 초기화해야 합니다.이전과 마찬가지로 하이퍼파라미터 `num_hiddens`는 은닉 유닛의 수를 정의합니다.표준 편차가 0.01 인 가우스 분포에 따라 가중치를 초기화하고 편향을 0으로 설정합니다.

```{.python .input}
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                np.zeros(num_hiddens, ctx=device))

    W_xi, W_hi, b_i = three()  # Input gate parameters
    W_xf, W_hf, b_f = three()  # Forget gate parameters
    W_xo, W_ho, b_o = three()  # Output gate parameters
    W_xc, W_hc, b_c = three()  # Candidate memory cell parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = np.zeros(num_outputs, ctx=device)
    # Attach gradients
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.python .input}
#@tab pytorch
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                d2l.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # Input gate parameters
    W_xf, W_hf, b_f = three()  # Forget gate parameters
    W_xo, W_ho, b_o = three()  # Output gate parameters
    W_xc, W_hc, b_c = three()  # Candidate memory cell parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

```{.python .input}
#@tab tensorflow
def get_lstm_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return tf.Variable(tf.random.normal(shape=shape, stddev=0.01,
                                            mean=0, dtype=tf.float32))
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                tf.Variable(tf.zeros(num_hiddens), dtype=tf.float32))

    W_xi, W_hi, b_i = three()  # Input gate parameters
    W_xf, W_hf, b_f = three()  # Forget gate parameters
    W_xo, W_ho, b_o = three()  # Output gate parameters
    W_xc, W_hc, b_c = three()  # Candidate memory cell parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = tf.Variable(tf.zeros(num_outputs), dtype=tf.float32)
    # Attach gradients
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    return params
```

### 모델 정의

[**초기화 함수**] 에서 LSTM의 숨겨진 상태는 값이 0이고 모양이 (배치 크기, 은닉 유닛 수) 인*추가* 메모리 셀을 반환해야 합니다.따라서 다음과 같은 상태 초기화가 발생합니다.

```{.python .input}
def init_lstm_state(batch_size, num_hiddens, device):
    return (np.zeros((batch_size, num_hiddens), ctx=device),
            np.zeros((batch_size, num_hiddens), ctx=device))
```

```{.python .input}
#@tab pytorch
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))
```

```{.python .input}
#@tab tensorflow
def init_lstm_state(batch_size, num_hiddens):
    return (tf.zeros(shape=(batch_size, num_hiddens)),
            tf.zeros(shape=(batch_size, num_hiddens)))
```

[**실제 모델**] 은 앞에서 논의한 것과 같이 정의됩니다. 세 개의 게이트와 보조 메모리 셀을 제공합니다.숨겨진 상태만 출력 레이어로 전달됩니다.메모리 셀 ($\mathbf{C}_t$) 은 출력 계산에 직접 참여하지 않는다.

```{.python .input}
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = npx.sigmoid(np.dot(X, W_xi) + np.dot(H, W_hi) + b_i)
        F = npx.sigmoid(np.dot(X, W_xf) + np.dot(H, W_hf) + b_f)
        O = npx.sigmoid(np.dot(X, W_xo) + np.dot(H, W_ho) + b_o)
        C_tilda = np.tanh(np.dot(X, W_xc) + np.dot(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * np.tanh(C)
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H, C)
```

```{.python .input}
#@tab pytorch
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)
```

```{.python .input}
#@tab tensorflow
def lstm(inputs, state, params):
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    (H, C) = state
    outputs = []
    for X in inputs:
        X=tf.reshape(X,[-1,W_xi.shape[0]])
        I = tf.sigmoid(tf.matmul(X, W_xi) + tf.matmul(H, W_hi) + b_i)
        F = tf.sigmoid(tf.matmul(X, W_xf) + tf.matmul(H, W_hf) + b_f)
        O = tf.sigmoid(tf.matmul(X, W_xo) + tf.matmul(H, W_ho) + b_o)
        C_tilda = tf.tanh(tf.matmul(X, W_xc) + tf.matmul(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * tf.tanh(C)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return tf.concat(outputs, axis=0), (H,C)
```

### [**교육**] 및 예측

:numref:`sec_rnn_scratch`에 도입된 `RNNModelScratch` 클래스를 인스턴스화하여 :numref:`sec_gru`에서 수행한 것과 동일하게 LSTM을 훈련시켜 보겠습니다.

```{.python .input}
#@tab mxnet, pytorch
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                            init_lstm_state, lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab tensorflow
vocab_size, num_hiddens, device_name = len(vocab), 256, d2l.try_gpu()._device_name
num_epochs, lr = 500, 1
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    model = d2l.RNNModelScratch(len(vocab), num_hiddens, init_lstm_state, lstm, get_lstm_params)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, strategy)
```

## [**간결한 구현**]

상위 수준 API를 사용하여 `LSTM` 모델을 직접 인스턴스화할 수 있습니다.위에서 명시한 모든 구성 세부 사항을 캡슐화합니다.이전에 자세히 설명한 많은 세부 사항에 대해 Python이 아닌 컴파일된 연산자를 사용하므로 코드가 훨씬 빠릅니다.

```{.python .input}
lstm_layer = rnn.LSTM(num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab))
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab tensorflow
lstm_cell = tf.keras.layers.LSTMCell(num_hiddens,
    kernel_initializer='glorot_uniform')
lstm_layer = tf.keras.layers.RNN(lstm_cell, time_major=True,
    return_sequences=True, return_state=True)
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    model = d2l.RNNModel(lstm_layer, vocab_size=len(vocab))
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, strategy)
```

LSTM은 중요하지 않은 상태 제어를 갖는 원형 잠재 변수 자기 회귀 모델입니다.수년에 걸쳐 많은 변형, 예를 들어 다중 층, 잔류 연결, 다양한 유형의 정규화가 제안되었습니다.그러나 LSTM 및 기타 시퀀스 모델 (예: GRU) 을 훈련하면 시퀀스의 장거리 종속성 때문에 비용이 많이 듭니다.나중에 경우에 따라 사용할 수있는 변압기와 같은 대체 모델을 만나게 될 것입니다. 

## 요약

* LSTM에는 입력 게이트, 잊어 게이트 및 정보 흐름을 제어하는 출력 게이트의 세 가지 유형의 게이트가 있습니다.
* LSTM의 은닉 레이어 출력에는 히든 상태와 메모리 셀이 포함됩니다.숨겨진 상태만 출력 레이어로 전달됩니다.메모리 셀은 완전히 내부에 있습니다.
* LSTM은 소실 및 폭발하는 그라디언트를 완화할 수 있습니다.

## 연습문제

1. 하이퍼파라미터를 조정하고 하이퍼파라미터가 실행 시간, 난처함 및 출력 시퀀스에 미치는 영향을 분석합니다.
1. 일련의 문자가 아닌 적절한 단어를 생성하기 위해 모델을 어떻게 변경해야 합니까?
1. 주어진 은닉 차원에 대한 GRU, LSTM 및 일반 RNN의 계산 비용을 비교합니다.교육 및 추론 비용에 특히주의하십시오.
1. 후보 메모리 셀은 $\tanh$ 함수를 사용하여 값 범위가 $-1$와 $1$ 사이임을 보장하기 때문에, 출력 값 범위가 $-1$와 $1$ 사이임을 보장하기 위해 숨겨진 상태가 $\tanh$ 함수를 다시 사용해야 하는 이유는 무엇입니까?
1. 문자 시퀀스 예측이 아닌 시계열 예측을 위한 LSTM 모델을 구현합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/343)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1057)
:end_tab:
