# 처음부터 순환 신경망 구현
:label:`sec_rnn_scratch`

이 섹션에서는 :numref:`sec_rnn`의 설명에 따라 문자 수준 언어 모델에 대해 처음부터 RNN을 구현합니다.이러한 모델은 H.G. Wells의*타임머신*에 대한 교육을 받게 됩니다.이전과 마찬가지로 :numref:`sec_language_model`에 소개된 데이터세트를 먼저 읽는 것으로 시작합니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input}
#@tab all
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab tensorflow
train_random_iter, vocab_random_iter = d2l.load_data_time_machine(
    batch_size, num_steps, use_random_iter=True)
```

## [**원-핫 인코딩**]

각 토큰은 `train_iter`에서 숫자 인덱스로 표시됩니다.이러한 지표를 신경망에 직접 공급하면 배우기가 어려울 수 있습니다.우리는 종종 각 토큰을 좀 더 표현적인 특징 벡터로 표현합니다.가장 쉬운 표현은 :numref:`subsec_classification-problem`에 도입된*원-핫 인코딩*입니다. 

간단히 말해서 각 인덱스를 다른 단위 벡터에 매핑합니다. 어휘의 서로 다른 토큰 수가 $N$ (`len(vocab)`) 이고 토큰 인덱스의 범위가 $0$에서 $N-1$까지라고 가정합니다.토큰의 인덱스가 정수 $i$이면 길이가 $N$인 모든 0으로 구성된 벡터를 만들고 $i$ 위치의 요소를 1로 설정합니다.이 벡터는 원래 토큰의 핫 벡터입니다.인덱스가 0과 2인 원핫 벡터는 다음과 같습니다.

```{.python .input}
npx.one_hot(np.array([0, 2]), len(vocab))
```

```{.python .input}
#@tab pytorch
F.one_hot(torch.tensor([0, 2]), len(vocab))
```

```{.python .input}
#@tab tensorflow
tf.one_hot(tf.constant([0, 2]), len(vocab))
```

매번 샘플링하는 (**미니배치의 모양**) (**는 (배치 크기, 시간 단계 수) 입니다.`one_hot` 함수는 이러한 미니 배치를 마지막 차원이 어휘 크기 (`len(vocab)`) 와 동일한 3 차원 텐서로 변환합니다.**) 우리는 종종 모양의 출력 (시간 단계 수, 배치 크기, 어휘 크기) 을 얻을 수 있도록 입력을 전치합니다.이렇게 하면 미니배치의 숨겨진 상태를 시간 단계별로 업데이트하기 위해 가장 바깥쪽 차원을 보다 편리하게 반복할 수 있습니다.

```{.python .input}
X = d2l.reshape(d2l.arange(10), (2, 5))
npx.one_hot(X.T, 28).shape
```

```{.python .input}
#@tab pytorch
X = d2l.reshape(d2l.arange(10), (2, 5))
F.one_hot(X.T, 28).shape
```

```{.python .input}
#@tab tensorflow
X = d2l.reshape(d2l.arange(10), (2, 5))
tf.one_hot(tf.transpose(X), 28).shape
```

## 모델 매개변수 초기화

다음으로 [**RNN 모델의 모델 매개 변수를 초기화**] 합니다.은닉 유닛 수 `num_hiddens`는 조정 가능한 하이퍼파라미터입니다.언어 모델을 학습할 때 입력과 출력은 동일한 어휘에서 나옵니다.따라서 어휘 크기와 동일한 차원이 있습니다.

```{.python .input}
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = d2l.zeros(num_hiddens, ctx=device)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, ctx=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.python .input}
#@tab pytorch
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = d2l.zeros(num_hiddens, device=device)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
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

    # Hidden layer parameters
    W_xh = tf.Variable(normal((num_inputs, num_hiddens)), dtype=tf.float32)
    W_hh = tf.Variable(normal((num_hiddens, num_hiddens)), dtype=tf.float32)
    b_h = tf.Variable(d2l.zeros(num_hiddens), dtype=tf.float32)
    # Output layer parameters
    W_hq = tf.Variable(normal((num_hiddens, num_outputs)), dtype=tf.float32)
    b_q = tf.Variable(d2l.zeros(num_outputs), dtype=tf.float32)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return params
```

## RNN 모델

RNN 모델을 정의하려면 먼저 초기화시 숨겨진 상태를 반환하는 [**`init_rnn_state` 함수가 필요합니다.**] 0으로 채워진 텐서와 모양 (배치 크기, 은닉 유닛 수) 을 반환합니다.튜플을 사용하면 숨겨진 상태가 여러 변수를 포함하는 상황을 더 쉽게 처리 할 수 있습니다. 이 문제는 이후 섹션에서 보게 될 것입니다.

```{.python .input}
def init_rnn_state(batch_size, num_hiddens, device):
    return (d2l.zeros((batch_size, num_hiddens), ctx=device), )
```

```{.python .input}
#@tab pytorch
def init_rnn_state(batch_size, num_hiddens, device):
    return (d2l.zeros((batch_size, num_hiddens), device=device), )
```

```{.python .input}
#@tab tensorflow
def init_rnn_state(batch_size, num_hiddens):
    return (d2l.zeros((batch_size, num_hiddens)), )
```

[**다음 `rnn` 함수는 시간 단계에서 은닉 상태와 출력값을 계산하는 방법을 정의합니다.**] RNN 모델은 가장 바깥쪽 차원 `inputs`을 반복하여 미니배치의 숨겨진 상태 `H`을 시간 단계별로 업데이트합니다.또한 여기서 활성화 함수는 $\tanh$ 함수를 사용합니다.:numref:`sec_mlp`에 설명된 대로 요소가 실수에 균일하게 분포되어 있는 경우 $\tanh$ 함수의 평균값은 0입니다.

```{.python .input}
def rnn(inputs, state, params):
    # Shape of `inputs`: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        H = np.tanh(np.dot(X, W_xh) + np.dot(H, W_hh) + b_h)
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)
```

```{.python .input}
#@tab pytorch
def rnn(inputs, state, params):
    # Here `inputs` shape: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

```{.python .input}
#@tab tensorflow
def rnn(inputs, state, params):
    # Here `inputs` shape: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        X = tf.reshape(X,[-1,W_xh.shape[0]])
        H = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(H, W_hh) + b_h)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return d2l.concat(outputs, axis=0), (H,)
```

필요한 모든 함수가 정의되면, 다음으로 처음부터 구현된 RNN 모델에 대해 [**이 함수를 래핑하고 매개 변수를 저장하는 클래스를 만들기**] 합니다.

```{.python .input}
class RNNModelScratch:  #@save
    """An RNN Model implemented from scratch."""
    def __init__(self, vocab_size, num_hiddens, device, get_params,
                 init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = npx.one_hot(X.T, self.vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, ctx):
        return self.init_state(batch_size, self.num_hiddens, ctx)
```

```{.python .input}
#@tab pytorch
class RNNModelScratch: #@save
    """A RNN Model implemented from scratch."""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
```

```{.python .input}
#@tab tensorflow
class RNNModelScratch: #@save
    """A RNN Model implemented from scratch."""
    def __init__(self, vocab_size, num_hiddens,
                 init_state, forward_fn, get_params):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.init_state, self.forward_fn = init_state, forward_fn
        self.trainable_variables = get_params(vocab_size, num_hiddens)

    def __call__(self, X, state):
        X = tf.one_hot(tf.transpose(X), self.vocab_size)
        X = tf.cast(X, tf.float32)
        return self.forward_fn(X, state, self.trainable_variables)

    def begin_state(self, batch_size, *args, **kwargs):
        return self.init_state(batch_size, self.num_hiddens)
```

예를 들어 숨겨진 상태의 차원이 변경되지 않도록 [**출력이 올바른 모양을 가지고 있는지 확인**] 하겠습니다.

```{.python .input}
#@tab mxnet
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.as_in_context(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
```

```{.python .input}
#@tab pytorch
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
```

```{.python .input}
#@tab tensorflow
# defining tensorflow training strategy
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)

num_hiddens = 512
with strategy.scope():
    net = RNNModelScratch(len(vocab), num_hiddens, init_rnn_state, rnn,
                          get_params)
state = net.begin_state(X.shape[0])
Y, new_state = net(X, state)
Y.shape, len(new_state), new_state[0].shape
```

출력 모양은 (시간 단계 수 $\times$ 배치 크기, 어휘 크기) 이고 숨겨진 상태 모양은 동일하게 유지됩니다 (배치 크기, 숨겨진 단위 수). 

## 예측

먼저 사용자가 제공 한 `prefix` 다음에 새 문자를 생성하는 예측 함수를 정의하겠습니다. 이 함수는 여러 문자를 포함하는 문자열입니다.`prefix`에서 이러한 시작 문자를 반복할 때 출력을 생성하지 않고 숨겨진 상태를 다음 시간 단계로 계속 전달합니다.이를*워밍업* 기간이라고 하며, 이 기간 동안 모델은 자체 업데이트 (예: 숨겨진 상태 업데이트) 를 수행하지만 예측하지는 않습니다.워밍업 기간이 지나면 숨겨진 상태가 일반적으로 처음에 초기화된 값보다 낫습니다.따라서 예측된 문자를 생성하고 방출합니다.

```{.python .input}
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1, ctx=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(
        d2l.tensor([outputs[-1]], ctx=device), (1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
#@tab pytorch
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor(
        [outputs[-1]], device=device), (1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
#@tab tensorflow
def predict_ch8(prefix, num_preds, net, vocab):  #@save
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1, dtype=tf.float32)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor([outputs[-1]]), (1, 1)).numpy()
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.numpy().argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

이제 `predict_ch8` 함수를 테스트할 수 있습니다.접두사를 `time traveller `로 지정하고 10자를 추가로 생성하도록 합니다.네트워크를 훈련시키지 않았으므로 무의미한 예측이 생성됩니다.

```{.python .input}
#@tab mxnet,pytorch
predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
predict_ch8('time traveller ', 10, net, vocab)
```

## [**그라데이션 클리핑**]

길이 $T$의 시퀀스에 대해 반복에서 이러한 $T$ 시간 단계에 대한 기울기를 계산합니다. 이로 인해 역전파 중에 길이가 $\mathcal{O}(T)$인 행렬-곱 체인이 생성됩니다.:numref:`sec_numerical_stability`에서 언급했듯이 수치적 불안정성이 발생할 수 있습니다. 예를 들어 $T$이 크면 그라디언트가 폭발하거나 사라질 수 있습니다.따라서 RNN 모델은 훈련을 안정화하기 위해 추가 도움이 필요한 경우가 많습니다. 

일반적으로 최적화 문제를 풀 때 모델 매개 변수에 대한 업데이트 단계 (예: 벡터 형식 $\mathbf{x}$) 를 미니 배치에서 음의 기울기 $\mathbf{g}$ 방향으로 수행합니다.예를 들어, $\eta > 0$를 학습률로 사용하여 한 번의 반복에서 $\mathbf{x}$을 $\mathbf{x} - \eta \mathbf{g}$로 업데이트합니다.또한 목적 함수 $f$가 상수 $L$를 사용하는*립쉬츠 연속*과 같이 잘 동작한다고 가정해 보겠습니다.즉, $\mathbf{x}$과 $\mathbf{y}$에 대해 우리는 

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

이 경우 매개 변수 벡터를 $\eta \mathbf{g}$로 업데이트하면 다음과 같이 안전하게 가정 할 수 있습니다. 

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|,$$

이는 우리가 $L \eta \|\mathbf{g}\|$ 이상의 변화를 관찰하지 않을 것임을 의미합니다.이것은 저주이자 축복입니다.저주 측면에서는 진전의 속도를 제한하는 반면, 축복 측면에서는 잘못된 방향으로 움직이면 상황이 잘못 될 수있는 정도를 제한합니다. 

때로는 그래디언트가 상당히 커서 최적화 알고리즘이 수렴하지 못할 수 있습니다.학습률 $\eta$를 줄임으로써 이 문제를 해결할 수 있습니다.하지만 그라디언트가 거의 발생하지 않는다면 어떻게 될까요?이 경우 이러한 접근 방식은 완전히 부당한 것으로 보일 수 있습니다.한 가지 인기있는 대안은 그라디언트 $\mathbf{g}$를 주어진 반경의 공 (예: $\theta$) 에 다시 투영하여 클리핑하는 것입니다. 

(**$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$달러**) 

이렇게 하면 기울기 노름이 $\theta$를 초과하지 않으며 업데이트된 그래디언트가 원래 방향인 $\mathbf{g}$와 완전히 정렬된다는 것을 알 수 있습니다.또한 주어진 미니배치 (및 그 안에서 주어진 샘플) 가 파라미터 벡터에 미칠 수 있는 영향을 제한하는 바람직한 부작용이 있습니다.이렇게 하면 모델에 어느 정도의 견고성이 부여됩니다.그라디언트 클리핑은 그라디언트 분해를 빠르게 수정할 수 있습니다.문제를 완전히 해결하지는 못하지만 문제를 완화하는 많은 기술 중 하나입니다. 

아래에서는 처음부터 구현되는 모델 또는 상위 수준 API로 구성된 모델의 그래디언트를 클립하는 함수를 정의합니다.또한 모든 모델 매개 변수에 대해 기울기 노름을 계산합니다.

```{.python .input}
def grad_clipping(net, theta):  #@save
    """Clip the gradient."""
    if isinstance(net, gluon.Block):
        params = [p.data() for p in net.collect_params().values()]
    else:
        params = net.params
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

```{.python .input}
#@tab pytorch
def grad_clipping(net, theta):  #@save
    """Clip the gradient."""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

```{.python .input}
#@tab tensorflow
def grad_clipping(grads, theta):  #@save
    """Clip the gradient."""
    theta = tf.constant(theta, dtype=tf.float32)
    new_grad = []
    for grad in grads:
        if isinstance(grad, tf.IndexedSlices):
            new_grad.append(tf.convert_to_tensor(grad))
        else:
            new_grad.append(grad)
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)).numpy()
                        for grad in new_grad))
    norm = tf.cast(norm, tf.float32)
    if tf.greater(norm, theta):
        for i, grad in enumerate(new_grad):
            new_grad[i] = grad * theta / norm
    else:
        new_grad = new_grad
    return new_grad
```

## 트레이닝

모델을 훈련시키기 전에 [**한 epoch에서 모델을 훈련시키는 함수를 정의**] 하겠습니다.:numref:`sec_softmax_scratch`의 모델을 세 곳에서 훈련시키는 방법과 다릅니다. 

1. 순차 데이터에 대한 샘플링 방법이 다르면 (랜덤 샘플링 및 순차 파티셔닝) 숨겨진 상태의 초기화에 차이가 발생합니다.
1. 모델 매개 변수를 업데이트하기 전에 그라디언트를 자릅니다.이렇게 하면 훈련 프로세스 중 특정 지점에서 기울기가 터지더라도 모델이 분기되지 않습니다.
1. 당혹감을 사용하여 모델을 평가합니다.:numref:`subsec_perplexity`에서 설명한 것처럼 길이가 다른 시퀀스를 비교할 수 있습니다.

특히 순차적 파티셔닝을 사용하는 경우 각 에포크의 시작 부분에서만 숨겨진 상태를 초기화합니다.다음 미니배치의 $i^\mathrm{th}$ 서브시퀀스 예제는 현재 $i^\mathrm{th}$ 서브시퀀스 예제와 인접하므로, 현재 미니배치의 끝에 있는 숨겨진 상태는 다음 미니배치의 시작 부분에서 숨겨진 상태를 초기화하는 데 사용됩니다.이러한 방식으로, 은닉 상태에 저장된 시퀀스의 히스토리 정보는 에포크 내의 인접한 서브시퀀스로 흐를 수 있다.그러나 어느 지점에서든 은닉 상태의 계산은 동일한 에포크의 모든 이전 미니 배치에 따라 달라지며, 이로 인해 기울기 계산이 복잡해집니다.계산 비용을 줄이기 위해 미니 배치를 처리하기 전에 기울기를 분리하여 숨겨진 상태의 기울기 계산이 항상 하나의 미니 배치의 시간 단계로 제한되도록합니다.  

랜덤 샘플링을 사용할 때는 각 예제가 임의의 위치로 샘플링되므로 각 반복마다 숨겨진 상태를 다시 초기화해야합니다.:numref:`sec_softmax_scratch`의 `train_epoch_ch3` 함수와 마찬가지로 `updater`은 모델 매개변수를 업데이트하는 일반적인 함수입니다.이는 처음부터 구현된 `d2l.sgd` 함수이거나 딥러닝 프레임워크의 내장 최적화 함수일 수 있습니다.

```{.python .input}
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train a model within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], ctx=device)
        else:
            for s in state:
                s.detach()
        y = Y.T.reshape(-1)
        X, y = X.as_in_ctx(device), y.as_in_ctx(device)
        with autograd.record():
            y_hat, state = net(X, state)
            l = loss(y_hat, y).mean()
        l.backward()
        grad_clipping(net, 1)
        updater(batch_size=1)  # Since the `mean` function has been invoked
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

```{.python .input}
#@tab pytorch
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train a net within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()
            else:
                # `state` is a tuple of tensors for `nn.LSTM` and
                # for our custom scratch implementation 
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

```{.python .input}
#@tab tensorflow
#@save
def train_epoch_ch8(net, train_iter, loss, updater, use_random_iter):
    """Train a model within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as g:
            y_hat, state = net(X, state)
            y = d2l.reshape(tf.transpose(Y), (-1))
            l = loss(y, y_hat)
        params = net.trainable_variables
        grads = g.gradient(l, params)
        grads = grad_clipping(grads, 1)
        updater.apply_gradients(zip(grads, params))

        # Keras loss by default returns the average loss in a batch
        # l_sum = l * float(d2l.size(y)) if isinstance(
        #     loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

[**훈련 함수는 처음부터 구현되거나 상위 수준 API를 사용하여 구현된 RNN 모델을 지원합니다.**]

```{.python .input}
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,  #@save
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, gluon.Block):
        net.initialize(ctx=device, force_reinit=True,
                         init=init.Normal(0.01))
        trainer = gluon.Trainer(net.collect_params(),
                                'sgd', {'learning_rate': lr})
        updater = lambda batch_size: trainer.step(batch_size)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

```{.python .input}
#@tab pytorch
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, strategy,
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    with strategy.scope():
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        updater = tf.keras.optimizers.SGD(lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater,
                                     use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    device = d2l.try_gpu()._device_name
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

[**이제 RNN 모델을 훈련시킬 수 있습니다.**] 데이터셋에서 10000개의 토큰만 사용하기 때문에 모델이 더 잘 수렴하려면 더 많은 Epoch가 필요합니다.

```{.python .input}
#@tab mxnet,pytorch
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, strategy)
```

[**마지막으로 랜덤 샘플링 방법을 사용한 결과를 확인해보겠습니다.**]

```{.python .input}
#@tab mxnet,pytorch
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)
```

```{.python .input}
#@tab tensorflow
with strategy.scope():
    net = RNNModelScratch(len(vocab), num_hiddens, init_rnn_state, rnn,
                          get_params)
train_ch8(net, train_iter, vocab_random_iter, lr, num_epochs, strategy,
          use_random_iter=True)
```

위의 RNN 모델을 처음부터 구현하는 것은 유익하지만 편리하지는 않습니다.다음 섹션에서는 RNN 모델을 더 쉽게 구현하고 더 빠르게 실행하는 방법과 같이 RNN 모델을 개선하는 방법을 살펴볼 것입니다. 

## 요약

* RNN 기반 문자 수준 언어 모델을 훈련시켜 사용자가 제공 한 텍스트 접두사 다음에 텍스트를 생성 할 수 있습니다.
* 간단한 RNN 언어 모델은 입력 인코딩, RNN 모델링 및 출력 생성으로 구성됩니다.
* RNN 모델은 훈련을 위해 상태 초기화가 필요하지만 랜덤 샘플링과 순차 분할은 다른 방식을 사용합니다.
* 순차 분할을 사용할 때는 계산 비용을 줄이기 위해 그래디언트를 분리해야 합니다.
* 워밍업 기간을 사용하면 예측하기 전에 모델이 자체적으로 업데이트 (예: 초기화된 값보다 더 나은 은닉 상태 획득) 할 수 있습니다.
* 그라디언트 클리핑은 그라디언트 폭발을 방지하지만 사라지는 그라디언트를 수정할 수는 없습니다.

## 연습문제

1. 원-핫 인코딩은 각 객체에 대해 다른 임베딩을 선택하는 것과 동일하다는 것을 보여줍니다.
1. 하이퍼파라미터 (예: Epoch 수, 은닉 유닛 수, 미니배치의 시간 스텝 수, 학습률) 를 조정하여 난처한 상황을 개선합니다.
    * 얼마나 낮게 갈 수 있니?
    * 원-핫 인코딩을 학습 가능한 임베딩으로 대체합니다.이로 인해 성능이 향상됩니까?
    * H.G. Wells의 다른 책 (예: [*The War of the Worlds*](http://www.gutenberg.org/ebooks/36)) 에서도 얼마나 잘 작동할까요?
1. 가장 가능성이 높은 다음 문자를 선택하는 대신 샘플링을 사용하도록 예측 함수를 수정합니다.
    * 어떻게 되나요?
    * 예를 들어 $\alpha > 1$에 대해 $q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$에서 샘플링하여 더 가능성이 높은 출력으로 모델을 편향시킵니다.
1. 그래디언트를 클리핑하지 않고 이 섹션의 코드를 실행합니다.어떻게 되나요?
1. 계산 그래프에서 은닉 상태를 분리하지 않도록 순차 분할을 변경합니다.실행 시간이 변경되나요?당혹감은 어때?
1. 이 섹션에 사용된 활성화 함수를 ReLU로 바꾸고 이 섹션의 실험을 반복합니다.그래디언트 클리핑이 여전히 필요한가요?왜요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/486)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1052)
:end_tab:
