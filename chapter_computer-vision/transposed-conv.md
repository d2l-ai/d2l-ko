# 전치된 컨볼루션
:label:`sec_transposed_conv`

컨벌루션 계층 (:numref:`sec_conv_layer`) 및 풀링 계층 (:numref:`sec_pooling`) 과 같이 지금까지 살펴본 CNN 계층은 일반적으로 입력의 공간 차원 (높이 및 너비) 을 축소 (다운샘플링) 하거나 변경하지 않고 유지합니다.픽셀 수준에서 분류되는 시맨틱 분할에서는 입력과 출력의 공간 차원이 동일하면 편리합니다.예를 들어, 한 출력 픽셀의 채널 차원은 동일한 공간 위치에 있는 입력 픽셀에 대한 분류 결과를 보유할 수 있습니다. 

이를 위해 특히 CNN 계층에 의해 공간 차원이 축소 된 후 중간 특징 맵의 공간 차원을 증가 (업 샘플링) 할 수있는 다른 유형의 CNN 레이어를 사용할 수 있습니다.이 섹션에서는 다음을 소개합니다. 
*전치된 컨벌루션*, *분수 보폭 컨벌루션* :cite:`Dumoulin.Visin.2016`라고도 합니다. 
컨벌루션에 의해 다운샘플링 연산을 반전하기 위한 것입니다.

```{.python .input}
from mxnet import np, npx, init
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from d2l import torch as d2l
```

## 기본 작동

지금은 채널을 무시하고 보폭이 1이고 패딩이 없는 기본 전치 컨볼루션 연산부터 시작하겠습니다.$n_h \times n_w$ 입력 텐서와 $k_h \times k_w$ 커널이 제공된다고 가정합니다.커널 창을 각 행에서 $n_w$번, 각 열에서 $n_h$번 보폭을 1로 슬라이딩하면 총 $n_h n_w$개의 중간 결과가 생성됩니다.각 중간 결과는 0으로 초기화되는 $(n_h + k_h - 1) \times (n_w + k_w - 1)$ 텐서입니다.각 중간 텐서를 계산하기 위해 입력 텐서의 각 요소에 커널을 곱하여 결과 $k_h \times k_w$ 텐서가 각 중간 텐서의 일부를 대체합니다.각 중간 텐서에서 대체된 부분의 위치는 계산에 사용되는 입력 텐서에서 요소의 위치에 해당합니다.결국 모든 중간 결과가 합산되어 출력을 생성합니다. 

예를 들어 :numref:`fig_trans_conv`는 $2\times 2$ 커널로 전치된 컨벌루션이 $2\times 2$ 입력 텐서에 대해 어떻게 계산되는지 보여줍니다. 

![Transposed convolution with a $2\times 2$ kernel. The shaded portions are a portion of an intermediate tensor as well as the input and kernel tensor elements used for the  computation.](../img/trans_conv.svg)
:label:`fig_trans_conv`

입력 행렬 `X` 및 커널 행렬 `K`에 대해 `trans_conv` (**이 기본 전치된 컨벌루션 연산을 구현**) 할 수 있습니다.

```{.python .input}
#@tab all
def trans_conv(X, K):
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y
```

커널을 통해 입력 요소를*감소시키는* 정규 컨벌루션 (:numref:`sec_conv_layer`) 과 달리 전치된 컨벌루션은
*브로드캐스트* 입력 요소 
커널을 통해 입력값보다 큰 출력을 생성합니다.기본 2 차원 전치 컨볼 루션 연산의 [** 위 구현의 출력을 검증**] 하기 위해 :numref:`fig_trans_conv`에서 입력 텐서 `X`과 커널 텐서 `K`를 구성 할 수 있습니다.

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
trans_conv(X, K)
```

또는 입력 `X`와 커널 `K`가 모두 4차원 텐서인 경우 [**상위 수준 API를 사용하여 동일한 결과를 얻습니다**] 할 수 있습니다.

```{.python .input}
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.Conv2DTranspose(1, kernel_size=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
tconv(X)
```

## [**패딩, 보폭, 다중 채널**]

입력에 패딩이 적용되는 일반 컨벌루션과는 달리, 전치된 컨벌루션의 출력값에 적용됩니다.예를 들어 높이와 너비의 양쪽에 있는 패딩 수를 1로 지정하면 전치된 컨벌루션 출력값에서 첫 번째 행과 마지막 행과 열이 제거됩니다.

```{.python .input}
tconv = nn.Conv2DTranspose(1, kernel_size=2, padding=1)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
tconv(X)
```

전치된 컨벌루션에서 스트라이드는 입력이 아닌 중간 결과 (따라서 출력값) 에 대해 지정됩니다.:numref:`fig_trans_conv`의 동일한 입력 및 커널 텐서를 사용하여 보폭을 1에서 2로 변경하면 중간 텐서의 높이와 무게가 모두 증가하므로 출력 텐서는 :numref:`fig_trans_conv_stride2`입니다. 

![Transposed convolution with a $2\times 2$ kernel with stride of 2. The shaded portions are a portion of an intermediate tensor as well as the input and kernel tensor elements used for the  computation.](../img/trans_conv_stride2.svg)
:label:`fig_trans_conv_stride2`

다음 코드 조각은 :numref:`fig_trans_conv_stride2`에서 보폭 2에 대해 전치된 컨벌루션 출력의 유효성을 검사할 수 있습니다.

```{.python .input}
tconv = nn.Conv2DTranspose(1, kernel_size=2, strides=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
tconv(X)
```

다중 입력 및 출력 채널의 경우 전치된 컨벌루션은 일반 컨벌루션과 동일한 방식으로 작동합니다.입력값에 $c_i$개의 채널이 있고 전치된 컨벌루션이 각 입력 채널에 $k_h\times k_w$ 커널 텐서를 할당한다고 가정합니다.여러 출력 채널을 지정하면 각 출력 채널에 대해 $c_i\times k_h\times k_w$ 커널이 생깁니다. 

모두와 마찬가지로 $\mathsf{X}$을 컨벌루션 계층 ($f$) 에 공급하여 $\mathsf{Y}=f(\mathsf{X})$을 출력하고 $f$과 동일한 하이퍼 파라미터를 가진 전치 된 컨벌루션 계층 ($g$) 을 만들면 출력 채널 수가 $\mathsf{X}$의 채널 수인 것을 제외하고는 $g(Y)$는 다음과 같은 모양을 갖습니다.$\mathsf{X}$.이 내용은 다음 예제에서 설명할 수 있습니다.

```{.python .input}
X = np.random.uniform(size=(1, 10, 16, 16))
conv = nn.Conv2D(20, kernel_size=5, padding=2, strides=3)
tconv = nn.Conv2DTranspose(10, kernel_size=5, padding=2, strides=3)
conv.initialize()
tconv.initialize()
tconv(conv(X)).shape == X.shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
tconv(conv(X)).shape == X.shape
```

## [**매트릭스 전치 연결**]
:label:`subsec-connection-to-mat-transposition`

전치된 컨벌루션은 행렬 전치 이름을 따서 명명됩니다.설명하기 위해 먼저 행렬 곱셈을 사용하여 컨볼 루션을 구현하는 방법을 살펴 보겠습니다.아래 예제에서는 $3\times 3$ 입력값 `X`과 $2\times 2$ 컨볼루션 커널 `K`를 정의한 다음 `corr2d` 함수를 사용하여 컨벌루션 출력값 `Y`를 계산합니다.

```{.python .input}
#@tab all
X = d2l.arange(9.0).reshape(3, 3)
K = d2l.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)
Y
```

다음으로 컨벌루션 커널 `K`를 많은 0을 포함하는 희소 가중치 행렬 `W`로 다시 작성합니다.가중치 행렬의 모양은 ($4$, $9$) 이며, 여기서 0이 아닌 요소는 컨볼루션 커널 `K`에서 가져옵니다.

```{.python .input}
#@tab all
def kernel2matrix(K):
    k, W = d2l.zeros(5), d2l.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
W
```

입력값 `X`을 행별로 연결하여 길이가 9인 벡터를 얻습니다.그런 다음 `W`의 행렬 곱셈과 벡터화된 `X`은 길이가 4인 벡터를 제공합니다.형태를 변경한 후 위의 원래 컨볼루션 연산에서 동일한 결과 `Y`를 얻을 수 있습니다. 행렬 곱셈을 사용하여 컨벌루션을 구현했습니다.

```{.python .input}
#@tab all
Y == d2l.matmul(W, d2l.reshape(X, -1)).reshape(2, 2)
```

마찬가지로 행렬 곱셈을 사용하여 전치 된 컨벌루션을 구현할 수 있습니다.다음 예제에서는 위의 정규 컨벌루션의 $2 \times 2$ 출력값 `Y`를 전치된 컨벌루션에 대한 입력으로 사용합니다.행렬을 곱하여 이 연산을 구현하려면 가중치 행렬 `W`를 새 셰이프 $(9, 4)$로 전치하기만 하면 됩니다.

```{.python .input}
#@tab all
Z = trans_conv(Y, K)
Z == d2l.matmul(W.T, d2l.reshape(Y, -1)).reshape(3, 3)
```

행렬을 곱하여 컨벌루션을 구현하는 것이 좋습니다.입력 벡터 $\mathbf{x}$ 및 가중치 행렬 $\mathbf{W}$이 주어지면, 컨벌루션의 순방향 전파 함수는 그 입력에 가중치 행렬을 곱하고 벡터 $\mathbf{y}=\mathbf{W}\mathbf{x}$을 출력함으로써 구현될 수 있다.역전파는 연쇄 규칙과 $\nabla_{\mathbf{x}}\mathbf{y}=\mathbf{W}^\top$를 따르기 때문에 컨볼루션의 역전파 함수는 입력에 전치된 가중치 행렬 $\mathbf{W}^\top$을 곱하여 구현할 수 있습니다.따라서 전치된 컨벌루션 계층은 컨벌루션 계층의 순방향 전파 함수와 역전파 함수를 교환할 수 있습니다. 순방향 전파 및 역전파 함수는 입력 벡터에 각각 $\mathbf{W}^\top$ 및 $\mathbf{W}$을 곱합니다. 

## 요약

* 커널을 통해 입력 요소를 줄이는 정규 컨벌루션과 달리, 전치된 컨벌루션은 커널을 통해 입력 요소를 브로드캐스트하여 입력보다 큰 출력을 생성합니다.
* $\mathsf{X}$을 컨벌루션 계층 $f$에 공급하여 $\mathsf{Y}=f(\mathsf{X})$을 출력하고 $\mathsf{X}$의 채널 수인 출력 채널 수를 제외하고 $f$과 동일한 하이퍼파라미터를 가진 전치된 컨벌루션 계층 $g$를 만들면 $g(Y)$는 $\mathsf{X}$과 동일한 모양을 갖습니다.
* 행렬 곱셈을 사용하여 컨벌루션을 구현할 수 있습니다.전치된 컨벌루션 계층은 컨벌루션 계층의 순방향 전파 함수와 역전파 함수를 교환할 수 있습니다.

## 연습문제

1. :numref:`subsec-connection-to-mat-transposition`에서 컨볼루션 입력 `X`과 전치된 컨벌루션 출력 `Z`의 모양은 동일합니다.동일한 가치를 지녔습니까?왜요?
1. 행렬 곱셈을 사용하여 컨벌루션을 구현하는 것이 효율적입니까?왜요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/376)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1450)
:end_tab:
