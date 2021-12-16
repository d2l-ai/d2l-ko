# 풀링
:label:`sec_pooling`

종종 이미지를 처리 할 때 숨겨진 표현의 공간 해상도를 점진적으로 줄이고 정보를 집계하여 네트워크에서 더 높은 곳으로 갈수록 각 숨겨진 노드가 민감한 수용 필드 (입력) 가 커지기를 원합니다. 

종종 우리의 궁극적 인 작업은 이미지에 대해 몇 가지 글로벌 질문을 던집니다. 예를 들어*고양이가 포함되어 있습니까?* 따라서 일반적으로 최종 레이어의 단위는 전체 입력에 민감해야 합니다.점차적으로 정보를 집계하고 더 거칠고 거친 맵을 생성함으로써 궁극적으로 전역 표현을 학습하는 동시에 컨벌루션 계층의 모든 이점을 처리의 중간 계층에서 유지한다는 목표를 달성합니다. 

또한 모서리 (:numref:`sec_conv_layer`에서 설명한 대로) 와 같은 하위 수준 피쳐를 탐지할 때 표현이 변환에 다소 불변하기를 원하는 경우가 많습니다.예를 들어, 흑백 사이의 선명한 묘사로 이미지 `X`을 가져 와서 전체 이미지를 오른쪽으로 한 픽셀 (예: `Z[i, j] = X[i, j + 1]`) 로 이동하면 새 이미지 `Z`의 출력이 크게 다를 수 있습니다.가장자리가 1픽셀씩 이동합니다.실제로 물체는 정확히 같은 장소에서 거의 발생하지 않습니다.실제로 삼각대와 고정 된 물체가 있더라도 셔터의 움직임으로 인한 카메라의 진동으로 인해 모든 것이 픽셀 단위로 이동할 수 있습니다 (고급 카메라에는이 문제를 해결하기위한 특수 기능이 탑재되어 있음). 

이 단원에서는*풀링 계층*을 소개합니다. 이 계층은 위치에 대한 컨벌루션 계층의 민감도를 완화하고 표현을 공간적으로 다운샘플링하는 두 가지 목적을 제공합니다. 

## 최대 풀링 및 평균 풀링

컨벌루션 계층과 마찬가지로, *풀링* 연산자는 보폭에 따라 입력값의 모든 영역에 걸쳐 슬라이딩되는 고정 형상 윈도우로 구성되며, 고정 형상 윈도우 (*풀링 윈도우라고도 함) 를 통과하는 각 위치에 대해 단일 출력값을 계산합니다.그러나 컨벌루션 계층의 입력값과 커널의 상호 상관 계산과 달리 풀링 계층에는 파라미터가 포함되지 않습니다 (*kernel*는 없음).대신 풀링 연산자는 결정론적이며 일반적으로 풀링 창에서 요소의 최대값 또는 평균값을 계산합니다.이러한 연산을 각각*최대 풀링* (줄여서 *최대 풀링*) 및*평균 풀링*이라고 합니다. 

두 경우 모두 상호 상관 연산자와 마찬가지로 풀링 창은 입력 텐서의 왼쪽 상단에서 시작하여 입력 텐서를 왼쪽에서 오른쪽으로, 위에서 아래로 미끄러지는 것으로 생각할 수 있습니다.풀링 윈도우가 도달하는 각 위치에서 최대 풀링을 사용하는지 평균 풀링을 사용하는지 여부에 따라 창에 있는 입력 서브텐서의 최대값 또는 평균값을 계산합니다. 

![Maximum pooling with a pooling window shape of $2\times 2$. The shaded portions are the first output element as well as the input tensor elements used for the output computation: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

:numref:`fig_pooling`의 출력 텐서는 높이가 2이고 너비가 2입니다.네 가지 요소는 각 풀링 창의 최대값에서 파생됩니다. 

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

풀링 창 모양이 $p \times q$인 풀링 계층을 $p \times q$ 풀링 계층이라고 합니다.풀링 작업을 $p \times q$ 풀링이라고 합니다. 

이 섹션의 시작 부분에 언급된 객체 가장자리 감지 예제로 돌아가 보겠습니다.이제 컨벌루션 계층의 출력값을 $2\times 2$ 최대 풀링의 입력값으로 사용하겠습니다.컨벌루션 계층 입력값을 `X`로 설정하고 풀링 계층 출력값을 `Y`으로 설정합니다.`X[i, j]`과 `X[i, j + 1]`의 값이 다른지 또는 `X[i, j + 1]`와 `X[i, j + 2]`의 값이 다른지 여부에 관계없이 풀링 계층은 항상 `Y[i, j] = 1`를 출력합니다.즉, $2\times 2$ 최대 풀링 계층을 사용하면 컨벌루션 계층이 인식하는 패턴이 높이 또는 너비에서 요소를 두 개 이상 이동하지 않는지 여전히 감지 할 수 있습니다. 

아래 코드에서는 `pool2d` 함수에서 (**풀링 계층의 순방향 전파를 구현**) 합니다.이 함수는 :numref:`sec_conv_layer`의 `corr2d` 함수와 유사합니다.그러나 여기에는 커널이 없으므로 출력을 입력의 각 영역의 최대 또는 평균으로 계산합니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
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
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = d2l.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode =='avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y
```

:numref:`fig_pooling`에서 입력 텐서 `X`를 구성하여 [**2차원 최대 풀링 계층의 출력을 검증**] 할 수 있습니다.

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

또한 (**평균 풀링 계층**) 을 사용하여 실험합니다.

```{.python .input}
#@tab all
pool2d(X, (2, 2), 'avg')
```

## [**패딩과 보폭**]

컨벌루션 계층과 마찬가지로 풀링 계층도 출력 형태를 변경할 수 있습니다.이전과 마찬가지로 입력을 채우고 보폭을 조정하여 원하는 출력 모양을 얻도록 작업을 변경할 수 있습니다.딥 러닝 프레임워크의 내장 2차원 최대 풀링 계층을 통해 풀링 계층에서 패딩과 스트라이드를 사용하는 방법을 시연할 수 있습니다.먼저 모양이 4 차원 인 입력 텐서 `X`를 구성합니다. 여기서 예제 수 (배치 크기) 와 채널 수는 모두 1입니다.

:begin_tab:`tensorflow`
tensorflow는*채널-마지막* 입력을 선호하고 최적화된다는 점에 유의하는 것이 중요합니다.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 1, 4, 4))
X
```

```{.python .input}
#@tab tensorflow
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 4, 4, 1))
X
```

기본적으로 (**프레임워크의 내장 클래스에 있는 인스턴스의 보폭과 풀링 창은 동일한 모양입니다.**) 아래에서는 셰이프 `(3, 3)`의 풀링 윈도우를 사용하므로 기본적으로 보폭 형상은 `(3, 3)`입니다.

```{.python .input}
pool2d = nn.MaxPool2D(3)
# Because there are no model parameters in the pooling layer, we do not need
# to call the parameter initialization function
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
pool2d(X)
```

[**보폭과 패딩은 수동으로 지정할 수 있습니다.**]

```{.python .input}
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)
```

:begin_tab:`mxnet`
물론 임의의 직사각형 풀링 윈도우를 지정하고 높이와 너비에 대한 패딩과 스트라이드를 각각 지정할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
물론 (**임의의 직사각형 풀링 윈도우를 지정하고 높이와 너비에 대한 패딩과 스트라이드**를 지정할 수 있습니다**).
:end_tab:

:begin_tab:`tensorflow`
물론 임의의 직사각형 풀링 윈도우를 지정하고 높이와 너비에 대한 패딩과 스트라이드를 각각 지정할 수 있습니다.
:end_tab:

```{.python .input}
pool2d = nn.MaxPool2D((2, 3), padding=(0, 1), strides=(2, 3))
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]])
X_padded = tf.pad(X, paddings, "CONSTANT")

pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='valid',
                                   strides=(2, 3))
pool2d(X_padded)
```

## 다중 채널

다채널 입력 데이터를 처리할 때 [**풀링 계층은 컨벌루션 계층에서와 같이 채널에 대한 입력을 합산하는 대신 각 입력 채널을 개별적으로 풀링합니다**].즉, 풀링 계층의 출력 채널 개수는 입력 채널의 개수와 같습니다.아래에서는 채널 차원에서 텐서 `X` 및 `X + 1`를 연결하여 2 채널로 입력을 구성합니다.

:begin_tab:`tensorflow`
채널-마지막 구문으로 인해 TensorFlow의 마지막 차원을 따라 연결해야 합니다.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
#@tab tensorflow
X = tf.concat([X, X + 1], 3)  # Concatenate along `dim=3` due to channels-last syntax
```

보시다시피 풀링 후에도 출력 채널 수는 여전히 2입니다.

```{.python .input}
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)
```

:begin_tab:`tensorflow`
텐서 플로우 풀링에 대한 출력은 언뜻보기에는 다르게 보이지만 수치적으로 동일한 결과가 MXNet 및 PyTorch와 같이 표시됩니다.차이점은 차원에 있으며 출력을 세로로 읽으면 다른 구현과 동일한 출력이 생성됩니다.
:end_tab:

## 요약

* 풀링 창의 입력 요소를 사용하여 최대값 풀링 연산은 최대값을 출력값으로 할당하고 평균 풀링 연산은 평균값을 출력값으로 할당합니다.
* 풀링 계층의 주요 이점 중 하나는 컨벌루션 계층의 위치에 대한 과도한 민감도를 완화하는 것입니다.
* 풀링 계층의 패딩과 스트라이드를 지정할 수 있습니다.
* 1보다 큰 보폭과 결합된 최대 풀링은 공간 차원 (예: 너비 및 높이) 을 줄이는 데 사용할 수 있습니다.
* 풀링 계층의 출력 채널 개수는 입력 채널의 개수와 같습니다.

## 연습문제

1. 컨벌루션 계층의 특수한 경우로 평균 풀링을 구현할 수 있습니까?그렇다면 그렇게 하십시오.
1. 컨벌루션 계층의 특수한 경우로 최댓값 풀링을 구현할 수 있습니까?그렇다면 그렇게 하십시오.
1. 풀링 계층의 계산 비용은 얼마입니까?풀링 계층에 대한 입력값의 크기가 $c\times h\times w$이고, 풀링 윈도우의 모양이 $p_h\times p_w$이고 패딩이 $(p_h, p_w)$이고 스트라이드가 $(s_h, s_w)$이라고 가정합니다.
1. 최대 풀링과 평균 풀링이 다르게 작동할 것으로 예상되는 이유는 무엇입니까?
1. 별도의 최소 풀링 계층이 필요합니까?다른 작업으로 교체할 수 있습니까?
1. 평균과 최대 풀링 사이에 고려할 수 있는 다른 연산이 있습니까 (힌트: 소프트맥스 리콜)?왜 그렇게 인기가 없을까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/71)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/72)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/274)
:end_tab:
