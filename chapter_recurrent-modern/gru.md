# 게이트된 순환 장치 (GRU)
:label:`sec_gru`

:numref:`sec_bptt`에서는 RNN에서 그라디언트가 계산되는 방법에 대해 논의했습니다.특히 행렬의 긴 곱은 기울기가 사라지거나 폭발할 수 있다는 것을 발견했습니다.이러한 기울기 이상이 실제로 무엇을 의미하는지 간단히 생각해 보겠습니다. 

* 미래의 모든 관찰을 예측하는 데 조기 관찰이 매우 중요한 상황에 직면 할 수 있습니다.첫 번째 관측치에 체크섬이 포함되어 있고 시퀀스 끝에서 체크섬이 올바른지 여부를 식별하는 것이 목표인 다소 인위적인 경우를 생각해 보십시오.이 경우 첫 번째 토큰의 영향력이 매우 중요합니다.*메모리 셀*에 중요한 초기 정보를 저장하는 몇 가지 메커니즘을 갖고 싶습니다.이러한 메커니즘이 없으면 모든 후속 관측치에 영향을 미치기 때문에이 관측치에 매우 큰 기울기를 할당해야합니다.
* 일부 토큰이 적절한 관찰을 수행하지 않는 상황이 발생할 수 있습니다.예를 들어 웹 페이지를 구문 분석할 때 페이지에 전달된 센티멘트를 평가하기 위한 목적과 관련이 없는 보조 HTML 코드가 있을 수 있습니다.잠재 상태 표현에서 이러한 토큰을*건너 뛰기* 하는 몇 가지 메커니즘을 갖고 싶습니다.
* 시퀀스의 일부 간에 논리적 중단이 있는 상황이 발생할 수 있습니다.예를 들어, 책의 장 사이에 전환되거나 증권에 대한 약세장과 강세장 사이의 전환이있을 수 있습니다.이 경우 내부 상태 표현을*재설정* 할 수있는 수단을 갖는 것이 좋을 것입니다.

이 문제를 해결하기 위해 여러 가지 방법이 제안되었습니다.가장 초기의 것 중 하나는 장기 단기 기억 :cite:`Hochreiter.Schmidhuber.1997`이며, 이에 대해서는 :numref:`sec_lstm`에서 논의 할 것입니다.게이트 순환 장치 (GRU) :cite:`Cho.Van-Merrienboer.Bahdanau.ea.2014`은 종종 비슷한 성능을 제공하고 :cite:`Chung.Gulcehre.Cho.ea.2014`을 계산하는 데 훨씬 빠른 약간 더 간소화된 변형입니다.단순성 때문에 GRU부터 시작하겠습니다. 

## 게이티드 히든 스테이트

바닐라 RNN과 GRU의 주요 차이점은 후자가 숨겨진 상태의 게이팅을 지원한다는 것입니다.즉, 숨겨진 상태가*업데이트* 되어야 할 때와*재설정*해야 할 때를 위한 전용 메커니즘이 있습니다.이러한 메커니즘을 학습하고 위에 나열된 문제를 해결합니다.예를 들어, 첫 번째 토큰이 매우 중요하다면 첫 번째 관찰 후 숨겨진 상태를 업데이트하지 않는 법을 배웁니다.마찬가지로 관련없는 임시 관찰을 건너 뛰는 법을 배웁니다.마지막으로 필요할 때마다 잠복 상태를 재설정하는 방법을 배웁니다.이에 대해서는 아래에서 자세히 설명합니다. 

### 리셋 게이트 및 업데이트 게이트

가장 먼저 소개해야 할 것은*리셋 게이트*와*업데이트 게이트*입니다.볼록 조합을 수행할 수 있도록 $(0, 1)$의 항목을 가진 벡터로 설계합니다.예를 들어, 리셋 게이트를 사용하면 이전 상태 중 기억하고 싶은 상태를 제어할 수 있습니다.마찬가지로 업데이트 게이트를 사용하면 새 상태가 이전 상태의 복사본인 정도를 제어할 수 있습니다. 

먼저 이러한 게이트를 엔지니어링합니다. :numref:`fig_gru_1`는 현재 시간 스텝의 입력과 이전 시간 스텝의 숨겨진 상태를 고려할 때 GRU의 리셋 및 업데이트 게이트에 대한 입력을 보여줍니다.두 게이트의 출력값은 시그모이드 활성화 함수를 사용하여 완전히 연결된 두 계층에 의해 제공됩니다. 

![Computing the reset gate and the update gate in a GRU model.](../img/gru-1.svg)
:label:`fig_gru_1`

수학적으로 주어진 시간 스텝 $t$에 대해 입력이 미니배치 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (예 수: $n$, 입력 수: $d$) 이고 이전 시간 스텝의 숨겨진 상태가 $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$ (숨겨진 단위 수: $h$) 이라고 가정합니다.그 후, 리셋 게이트 $\mathbf{R}_t \in \mathbb{R}^{n \times h}$ 및 업데이트 게이트 $\mathbf{Z}_t \in \mathbb{R}^{n \times h}$은 다음과 같이 계산된다. 

$$
\begin{aligned}
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r),\\
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z),
\end{aligned}
$$

여기서 $\mathbf{W}_{xr}, \mathbf{W}_{xz} \in \mathbb{R}^{d \times h}$ 및 $\mathbf{W}_{hr}, \mathbf{W}_{hz} \in \mathbb{R}^{h \times h}$는 가중치 매개변수이고 $\mathbf{b}_r, \mathbf{b}_z \in \mathbb{R}^{1 \times h}$은 편향입니다.요약 중에 브로드캐스트 (:numref:`subsec_broadcasting` 참조) 가 트리거됩니다.시그모이드 함수 (:numref:`sec_mlp`에 소개됨) 를 사용하여 입력 값을 구간 $(0, 1)$로 변환합니다. 

### 후보 히든 스테이트

다음으로 :eqref:`rnn_h_with_state`의 리셋 게이트 $\mathbf{R}_t$를 일반 잠재 상태 업데이트 메커니즘과 통합해 보겠습니다.다음과 같은 결과가 발생합니다.
*후보 숨김 상태*
$\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$ 시간 단계 $t$에서: 

$$\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{hh} + \mathbf{b}_h),$$
:eqlabel:`gru_tilde_H`

여기서 $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$ 및 $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$는 가중치 모수이고, $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$은 치우침이며, 기호 $\odot$은 하다마르 (요소별) 곱 연산자입니다.여기서는 tanh 형식의 비선형성을 사용하여 후보 숨겨진 상태의 값이 간격 $(-1, 1)$에 유지되도록 합니다. 

업데이트 게이트의 동작을 통합해야 하기 때문에 결과는*candidate*입니다.:eqref:`rnn_h_with_state`와 비교할 때, 이제 :eqref:`gru_tilde_H`에서 $\mathbf{R}_t$과 $\mathbf{H}_{t-1}$의 요소별 곱셈으로 이전 상태의 영향을 줄일 수 있습니다.리셋 게이트 $\mathbf{R}_t$의 항목이 1에 가까울 때마다 :eqref:`rnn_h_with_state`와 같은 바닐라 RNN을 복구합니다.0에 가까운 리셋 게이트 $\mathbf{R}_t$의 모든 엔트리에 대해, 후보 은닉 상태는 $\mathbf{X}_t$을 입력으로 하는 MLP의 결과이다.따라서 기존의 숨겨진 상태는 기본값으로*reset*됩니다. 

:numref:`fig_gru_2`는 리셋 게이트를 적용한 후의 계산 흐름을 보여줍니다. 

![Computing the candidate hidden state in a GRU model.](../img/gru-2.svg)
:label:`fig_gru_2`

### 히든 스테이트

마지막으로 업데이트 게이트 $\mathbf{Z}_t$의 효과를 통합해야 합니다.이는 새로운 숨겨진 상태 $\mathbf{H}_t \in \mathbb{R}^{n \times h}$가 단지 이전 상태 $\mathbf{H}_{t-1}$인 정도와 새 후보 상태 $\tilde{\mathbf{H}}_t$가 사용되는 정도를 결정합니다.업데이트 게이트 $\mathbf{Z}_t$은 $\mathbf{H}_{t-1}$과 $\tilde{\mathbf{H}}_t$ 사이의 요소별 볼록 조합을 취함으로써 이러한 목적으로 사용될 수 있습니다.이렇게 하면 GRU에 대한 최종 업데이트 방정식이 생성됩니다. 

$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.$$

업데이트 게이트 $\mathbf{Z}_t$이 1에 가까울 때마다 이전 상태를 유지하기만 하면 됩니다.이 경우 $\mathbf{X}_t$의 정보는 기본적으로 무시되어 종속성 체인에서 시간 단계 $t$을 효과적으로 건너뜁니다.대조적으로, $\mathbf{Z}_t$이 0에 가까울 때마다, 새로운 잠재 상태 $\mathbf{H}_t$는 후보 잠재 상태 $\tilde{\mathbf{H}}_t$에 접근한다.이러한 설계는 RNN의 소실 기울기 문제에 대처하고 시간 스텝 거리가 큰 시퀀스에 대한 종속성을 더 잘 캡처하는 데 도움이 될 수 있습니다.예를 들어 업데이트 게이트가 전체 서브시퀀스의 모든 시간 스텝에 대해 1에 가까우면 서브시퀀스의 길이에 관계없이 시작 시간 스텝의 이전 은닉 상태가 쉽게 유지되고 끝까지 전달됩니다. 

:numref:`fig_gru_3`는 업데이트 게이트가 작동 중인 후의 계산 흐름을 보여 줍니다. 

![Computing the hidden state in a GRU model.](../img/gru-3.svg)
:label:`fig_gru_3`

요약하면 GRU에는 다음과 같은 두 가지 특징이 있습니다. 

* 리셋 게이트는 시퀀스의 단기 종속성을 캡처하는 데 도움이
* 업데이트 게이트는 시퀀스에서 장기적인 종속성을 캡처하는 데 도움이

## 처음부터 구현

GRU 모델을 더 잘 이해하기 위해 처음부터 구현해 보겠습니다.먼저 :numref:`sec_rnn_scratch`에서 사용한 타임머신 데이터 세트를 읽습니다.데이터 세트를 읽는 코드는 다음과 같습니다.

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

### (**모델 매개변수 초기화**)

다음 단계는 모델 매개변수를 초기화하는 것입니다.표준 편차가 0.01인 가우스 분포에서 가중치를 그리고 바이어스를 0으로 설정합니다.하이퍼파라미터 `num_hiddens`는 은닉 유닛의 개수를 정의합니다.업데이트 게이트, 리셋 게이트, 후보 은닉 상태 및 출력 레이어와 관련된 모든 가중치와 바이어스를 인스턴스화합니다.

```{.python .input}
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                np.zeros(num_hiddens, ctx=device))

    W_xz, W_hz, b_z = three()  # Update gate parameters
    W_xr, W_hr, b_r = three()  # Reset gate parameters
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = np.zeros(num_outputs, ctx=device)
    # Attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.python .input}
#@tab pytorch
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                d2l.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # Update gate parameters
    W_xr, W_hr, b_r = three()  # Reset gate parameters
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

```{.python .input}
#@tab tensorflow
def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return d2l.normal(shape=shape,stddev=0.01,mean=0,dtype=tf.float32)

    def three():
        return (tf.Variable(normal((num_inputs, num_hiddens)), dtype=tf.float32),
                tf.Variable(normal((num_hiddens, num_hiddens)), dtype=tf.float32),
                tf.Variable(d2l.zeros(num_hiddens), dtype=tf.float32))

    W_xz, W_hz, b_z = three()  # Update gate parameters
    W_xr, W_hr, b_r = three()  # Reset gate parameters
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameters
    # Output layer parameters
    W_hq = tf.Variable(normal((num_hiddens, num_outputs)), dtype=tf.float32)
    b_q = tf.Variable(d2l.zeros(num_outputs), dtype=tf.float32)
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    return params
```

### 모델 정의

이제 [**숨겨진 상태 초기화 함수**] `init_gru_state`를 정의하겠습니다.:numref:`sec_rnn_scratch`에 정의된 `init_rnn_state` 함수와 마찬가지로 이 함수는 값이 모두 0인 모양 (배치 크기, 은닉 단위 수) 을 가진 텐서를 반환합니다.

```{.python .input}
def init_gru_state(batch_size, num_hiddens, device):
    return (np.zeros(shape=(batch_size, num_hiddens), ctx=device), )
```

```{.python .input}
#@tab pytorch
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )
```

```{.python .input}
#@tab tensorflow
def init_gru_state(batch_size, num_hiddens):
    return (d2l.zeros((batch_size, num_hiddens)), )
```

이제 [**GRU 모델을 정의**] 할 준비가 되었습니다.업데이트 방정식이 더 복잡하다는 점을 제외하면 기본 RNN 셀의 구조와 동일합니다.

```{.python .input}
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = npx.sigmoid(np.dot(X, W_xz) + np.dot(H, W_hz) + b_z)
        R = npx.sigmoid(np.dot(X, W_xr) + np.dot(H, W_hr) + b_r)
        H_tilda = np.tanh(np.dot(X, W_xh) + np.dot(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)
```

```{.python .input}
#@tab pytorch
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

```{.python .input}
#@tab tensorflow
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        X = tf.reshape(X,[-1,W_xh.shape[0]])
        Z = tf.sigmoid(tf.matmul(X, W_xz) + tf.matmul(H, W_hz) + b_z)
        R = tf.sigmoid(tf.matmul(X, W_xr) + tf.matmul(H, W_hr) + b_r)
        H_tilda = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return tf.concat(outputs, axis=0), (H,)
```

### 교육 및 예측

[**교육**] 및 예측은 :numref:`sec_rnn_scratch`와 정확히 동일한 방식으로 작동합니다.훈련 후에는 제공된 접두사 “시간 여행자”와 “여행자”에 따라 훈련 세트와 예측 순서에 대한 당혹감을 인쇄합니다.

```{.python .input}
#@tab mxnet, pytorch
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab tensorflow
vocab_size, num_hiddens, device_name = len(vocab), 256, d2l.try_gpu()._device_name
# defining tensorflow training strategy
strategy = tf.distribute.OneDeviceStrategy(device_name)
num_epochs, lr = 500, 1
with strategy.scope():
    model = d2l.RNNModelScratch(len(vocab), num_hiddens, init_gru_state, gru, get_params)

d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, strategy)
```

## [**간결한 구현**]

상위 수준 API에서는 GPU 모델을 직접 인스턴스화할 수 있습니다.위에서 명시한 모든 구성 세부 사항을 캡슐화합니다.이전에 설명한 많은 세부 사항에 대해 Python이 아닌 컴파일된 연산자를 사용하므로 코드가 훨씬 빠릅니다.

```{.python .input}
gru_layer = rnn.GRU(num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab tensorflow
gru_cell = tf.keras.layers.GRUCell(num_hiddens,
    kernel_initializer='glorot_uniform')
gru_layer = tf.keras.layers.RNN(gru_cell, time_major=True,
    return_sequences=True, return_state=True)

device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    model = d2l.RNNModel(gru_layer, vocab_size=len(vocab))

d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, strategy)
```

## 요약

* 게이트 RNN은 시간 스텝 거리가 큰 시퀀스에 대한 종속성을 더 잘 캡처할 수 있습니다.
* 리셋 게이트는 시퀀스의 단기 종속성을 캡처하는 데 도움이
* 업데이트 게이트는 시퀀스에서 장기적인 종속성을 캡처하는 데 도움이
* GRU에는 리셋 게이트가 켜질 때마다 극단적 인 경우로 기본 RNN이 포함됩니다.업데이트 게이트를 켜서 후속 작업을 건너뛸 수도 있습니다.

## 연습문제

1. 시간 단계 $t > t'$에서 출력을 예측하기 위해 시간 단계 $t'$의 입력만 사용한다고 가정합니다.각 시간 단계의 재설정 및 업데이트 게이트에 가장 적합한 값은 무엇입니까?
1. 하이퍼파라미터를 조정하고 하이퍼파라미터가 실행 시간, 난처함 및 출력 시퀀스에 미치는 영향을 분석합니다.
1. `rnn.RNN` 및 `rnn.GRU` 구현에 대한 런타임, 당혹성 및 출력 문자열을 서로 비교합니다.
1. 예를 들어 리셋 게이트만 사용하거나 업데이트 게이트만 사용하여 GRU의 일부만 구현하면 어떻게 됩니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/342)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1056)
:end_tab:
