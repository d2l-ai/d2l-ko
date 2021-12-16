# 이미지에 대한 컨볼루션
:label:`sec_conv_layer`

이제 컨벌루션 계층이 이론적으로 작동하는 방식을 이해했으므로 실제로 어떻게 작동하는지 확인할 준비가 되었습니다.이미지 데이터의 구조를 탐색하기 위한 효율적인 아키텍처로서 컨벌루션 신경망의 동기를 기반으로 이미지를 실행 예제로 사용합니다.

## 상호 상관 연산

엄밀히 말하면 컨벌루션 계층이 표현하는 연산이 상호 상관 관계로 더 정확하게 설명되기 때문에 컨벌루션 계층은 잘못된 이름입니다.:numref:`sec_why-conv`의 컨벌루션 계층에 대한 설명을 기반으로 이러한 계층에서 입력 텐서와 커널 텐서가 결합되어 (**상호 상관 연산**) 을 통해 출력 텐서를 생성합니다.

지금은 채널을 무시하고 2차원 데이터 및 숨겨진 표현에서 어떻게 작동하는지 살펴 보겠습니다.:numref:`fig_correlation`에서 입력은 높이가 3이고 너비가 3인 2차원 텐서입니다.텐서의 모양을 $3 \times 3$ 또는 ($3$, $3$) 으로 표시합니다.커널의 높이와 너비는 모두 2입니다.*커널 창* (또는*컨볼루션 창*) 의 모양은 커널의 높이와 너비로 지정됩니다 (여기서는 $2 \times 2$입니다).

![Two-dimensional cross-correlation operation. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
:label:`fig_correlation`

2 차원 상호 상관 연산에서는 입력 텐서의 왼쪽 상단 모서리에 위치한 컨볼 루션 창에서 시작하여 왼쪽에서 오른쪽으로, 위에서 아래로 입력 텐서를 가로 질러 슬라이드합니다.컨볼 루션 창이 특정 위치로 미끄러지면 해당 창에 포함 된 입력 서브 텐서와 커널 텐서가 요소별로 곱해지고 결과 텐서가 합산되어 단일 스칼라 값이 생성됩니다.이 결과는 해당 위치에서 출력 텐서의 값을 제공합니다.여기서 출력 텐서의 높이는 2이고 너비는 2이며 네 개의 요소는 2차원 상호 상관 연산에서 파생됩니다.

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.
$$

각 축을 따라 출력 크기는 입력 크기보다 약간 작습니다.커널의 너비와 높이가 1보다 크기 때문에 커널이 이미지 내에 완전히 맞는 위치에 대해서만 상호 상관을 올바르게 계산할 수 있습니다. 출력 크기는 입력 크기 $n_h \times n_w$에서 다음을 통해 컨볼루션 커널 $k_h \times k_w$의 크기를 뺀 값으로 계산됩니다.

$$(n_h-k_h+1) \times (n_w-k_w+1).$$

이미지 전체에서 컨볼루션 커널을 “이동”하기에 충분한 공간이 필요하기 때문입니다.나중에 커널을 이동할 충분한 공간이 있도록 경계 주위에 0으로 이미지를 채워 크기를 변경하지 않는 방법을 살펴 보겠습니다.다음으로 입력 텐서 `X`와 커널 텐서 `K`를 받아들이고 출력 텐서 `Y`을 반환하는 `corr2d` 함수에서이 프로세스를 구현합니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab mxnet, pytorch
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y
```

입력 텐서 `X`와 커널 텐서 `K`을 :numref:`fig_correlation`에서 구성하여 2 차원 상호 상관 연산의 [** 위 구현의 출력을 검증**] 할 수 있습니다.

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

## 컨벌루션 계층

컨벌루션 계층은 입력값과 커널을 교차 상관시키고 스칼라 바이어스를 추가하여 출력값을 생성합니다.컨벌루션 계층의 두 파라미터는 커널과 스칼라 편향입니다.컨벌루션 계층을 기반으로 모델을 훈련시킬 때는 일반적으로 완전 연결 계층에서와 마찬가지로 커널을 무작위로 초기화합니다.

이제 위에서 정의한 `corr2d` 함수를 기반으로 [**2차원 컨벌루션 계층을 구현**] 할 준비가 되었습니다.`__init__` 생성자 함수에서 `weight` 및 `bias`를 두 개의 모델 매개 변수로 선언합니다.순방향 전파 함수는 `corr2d` 함수를 호출하고 바이어스를 추가합니다.

```{.python .input}
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

```{.python .input}
#@tab pytorch
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

```{.python .input}
#@tab tensorflow
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias
```

$h \times w$ 컨볼루션 또는 $h \times w$ 컨볼루션 커널에서 컨볼루션 커널의 높이와 너비는 각각 $h$와 $w$입니다.또한 $h \times w$ 컨벌루션 커널이 있는 컨벌루션 계층을 간단히 $h \times w$ 컨벌루션 계층이라고 합니다.

## 이미지의 객체 가장자리 감지

잠시 시간을내어 픽셀 변화의 위치를 찾아 [**컨벌루션 계층의 간단한 적용: 이미지에서 객체의 가장자리 감지**] 를 구문 분석해 보겠습니다.먼저 $6\times 8$픽셀의 “이미지”를 구성합니다.가운데 4개의 열은 검은색 (0) 이고 나머지 열은 흰색 (1) 입니다.

```{.python .input}
#@tab mxnet, pytorch
X = d2l.ones((6, 8))
X[:, 2:6] = 0
X
```

```{.python .input}
#@tab tensorflow
X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X
```

다음으로 높이가 1이고 너비가 2인 커널 `K`를 구성합니다.입력값으로 상호 상관 연산을 수행 할 때 수평으로 인접한 요소가 같으면 출력은 0입니다.그렇지 않으면 출력은 0이 아닙니다.

```{.python .input}
#@tab all
K = d2l.tensor([[1.0, -1.0]])
```

인수 `X` (입력) 와 `K` (커널) 를 사용하여 상호 상관 연산을 수행할 준비가 되었습니다.보시다시피, [**흰색에서 검정으로 가장자리에 대해 1을 감지하고 검정에서 흰색으로 가장자리에 대해 -1을 감지합니다.**] 다른 모든 출력은 값 0을 갖습니다.

```{.python .input}
#@tab all
Y = corr2d(X, K)
Y
```

이제 전치된 이미지에 커널을 적용할 수 있습니다.예상대로 사라집니다.[**커널 `K`는 세로 가장자리만 감지합니다.**]

```{.python .input}
#@tab all
corr2d(d2l.transpose(X), K)
```

## 커널 학습

유한 차분 `[1, -1]`로 에지 검출기를 설계하는 것은 이것이 우리가 찾고있는 것임을 정확히 알고 있다면 깔끔합니다.그러나 더 큰 커널을 살펴보고 연속적인 컨벌루션 계층을 고려할 때 각 필터가 수동으로 수행해야 할 작업을 정확하게 지정하는 것은 불가능할 수 있습니다.

이제 입력-출력 쌍만 살펴봄으로써 [**`X`**]에서 `Y`를 생성한 커널을 학습할 수 있는지 살펴보겠습니다.먼저 컨볼 루션 계층을 만들고 커널을 랜덤 텐서로 초기화합니다.다음으로 각 반복에서 제곱 오차를 사용하여 `Y`를 컨벌루션 계층의 출력값과 비교합니다.그런 다음 기울기를 계산하여 커널을 업데이트할 수 있습니다.단순화를 위해 다음에서는 2차원 컨벌루션 계층에 내장 클래스를 사용하고 바이어스를 무시합니다.

```{.python .input}
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2D(1, kernel_size=(1, 2), use_bias=False)
conv2d.initialize()

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)
lr = 3e-2  # Learning rate

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # Update the kernel
    conv2d.weight.data()[:] -= lr * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {float(l.sum()):.3f}')
```

```{.python .input}
#@tab pytorch
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # Learning rate

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
```

```{.python .input}
#@tab tensorflow
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, height, width, channel), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))
lr = 3e-2  # Learning rate

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        # Update the kernel
        update = tf.multiply(lr, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'epoch {i + 1}, loss {tf.reduce_sum(l):.3f}')
```

10회 반복 후 오류가 작은 값으로 떨어졌습니다.이제 [**배운 커널 텐서를 살펴보겠습니다.**]

```{.python .input}
d2l.reshape(conv2d.weight.data(), (1, 2))
```

```{.python .input}
#@tab pytorch
d2l.reshape(conv2d.weight.data, (1, 2))
```

```{.python .input}
#@tab tensorflow
d2l.reshape(conv2d.get_weights()[0], (1, 2))
```

실제로 학습된 커널 텐서는 앞서 정의한 커널 텐서 `K`에 매우 가깝습니다.

## 교차 상관 및 컨볼루션

교차 상관 연산과 컨볼루션 연산 간의 대응에 대한 :numref:`sec_why-conv`의 관찰을 상기하십시오.여기서는 2 차원 컨볼 루션 계층을 계속 고려해 보겠습니다.이러한 계층이 교차 상관 대신 :eqref:`eq_2d-conv-discrete`에 정의된 대로 엄격한 컨벌루션 연산을 수행하면 어떻게 될까요?엄격한* 컨볼 루션* 연산의 출력을 얻으려면 2 차원 커널 텐서를 수평 및 수직으로 뒤집은 다음 입력 텐서로* 상호 상관 관계* 연산을 수행하면됩니다.

커널은 딥러닝의 데이터에서 학습되므로 컨벌루션 계층의 출력값은 이러한 계층이 엄격한 컨벌루션 연산이나 상호 상관 연산을 수행하더라도 영향을 받지 않습니다.

이를 설명하기 위해 컨벌루션 계층이*교차 상관 관계*를 수행하고 :numref:`fig_correlation`에서 커널을 학습한다고 가정합니다. 이 커널은 여기서 행렬 $\mathbf{K}$으로 표시됩니다.다른 조건이 변경되지 않는다고 가정할 때, 이 계층이 엄격한*컨벌루션*을 대신 수행하면 $\mathbf{K}'$가 가로 및 세로로 뒤집힌 후 학습된 커널 $\mathbf{K}'$는 $\mathbf{K}$과 같습니다.즉, 컨벌루션 계층이 :numref:`fig_correlation` 및 $\mathbf{K}'$의 입력에 대해 엄격한*컨벌루션*을 수행하면 :numref:`fig_correlation`에서 동일한 출력 (입력과 $\mathbf{K}$의 상호 상관) 이 얻어집니다.

딥 러닝 문헌의 표준 용어에 따라 엄격하게 말하면 약간 다르더라도 상호 상관 연산을 컨볼루션이라고 계속 지칭할 것입니다.게다가*요소*라는 용어를 사용하여 계층 표현 또는 컨볼 루션 커널을 나타내는 텐서의 항목 (또는 구성 요소) 을 나타냅니다.

## 특징 맵 및 수용 필드

:numref:`subsec_why-conv-channels`에서 설명한 대로 :numref:`fig_correlation`의 컨벌루션 계층 출력값은*특징 맵*이라고도 합니다. 이는 후속 계층에 대한 공간 차원 (예: 너비 및 높이) 에서 학습된 표현 (특징) 으로 간주될 수 있기 때문입니다.CNN에서 일부 층의 모든 요소 $x$에 대해*수용 필드*는 순방향 전파 중에 $x$의 계산에 영향을 미칠 수 있는 모든 요소 (이전 계층의 모든 요소) 를 나타냅니다.수신 필드는 입력의 실제 크기보다 클 수 있습니다.

수용 분야를 설명하기 위해 :numref:`fig_correlation`를 계속 사용하겠습니다.$2 \times 2$ 컨벌루션 커널이 주어지면 음영처리된 출력 요소 (값 $19$) 의 수용 필드는 입력의 음영 부분에 있는 4개의 요소입니다.이제 $2 \times 2$ 출력을 $\mathbf{Y}$으로 표시하고 $\mathbf{Y}$을 입력으로 사용하여 단일 요소 $z$을 출력하는 추가 $2 \times 2$ 컨벌루션 계층이 있는 더 깊은 CNN을 고려해 보겠습니다.이 경우 $\mathbf{Y}$의 $z$의 수용 필드에는 $\mathbf{Y}$의 네 가지 요소가 모두 포함되는 반면 입력의 수용 필드에는 9개의 입력 요소가 모두 포함됩니다.따라서 피처 맵의 요소가 더 넓은 영역에서 입력 피처를 감지하기 위해 더 큰 수용 필드가 필요한 경우 더 깊은 네트워크를 구축 할 수 있습니다.

## 요약

* 2차원 컨벌루션 계층의 핵심 계산은 2차원 상호 상관 연산입니다.가장 간단한 형태로 2차원 입력 데이터와 커널에 대해 상호 상관 연산을 수행한 다음 치우침을 추가합니다.
* 이미지의 가장자리를 감지하는 커널을 설계할 수 있습니다.
* 데이터로부터 커널의 파라미터를 배울 수 있습니다.
* 데이터에서 학습한 커널의 경우 컨벌루션 계층의 출력값은 해당 계층에서 수행한 연산 (엄격한 컨볼루션 또는 상호 상관) 에 관계없이 영향을 받지 않습니다.
* 피처 맵의 요소가 입력에서 더 넓은 피처를 탐지하기 위해 더 큰 수용 필드가 필요한 경우 더 깊은 네트워크를 고려할 수 있습니다.

## 연습문제

1. 대각선 가장자리가 있는 이미지 `X`를 구성합니다.
    1. 이 섹션의 커널 `K`를 이 커널에 적용하면 어떻게 되나요?
    1. `X`를 전치하면 어떻게 되나요?
    1. `K`를 전치하면 어떻게 되나요?
1. 우리가 만든 `Conv2D` 클래스의 그래디언트를 자동으로 찾으려고 할 때 어떤 종류의 오류 메시지가 표시됩니까?
1. 입력 텐서와 커널 텐서를 변경하여 교차 상관 연산을 행렬 곱셈으로 어떻게 표현합니까?
1. 일부 커널을 수동으로 설계합니다.
    1. 2차 미분에 대한 커널의 형태는 무엇입니까?
    1. 적분의 커널은 무엇입니까?
    1. $d$도의 도함수를 얻기 위한 커널의 최소 크기는 얼마입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/65)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/66)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/271)
:end_tab:
