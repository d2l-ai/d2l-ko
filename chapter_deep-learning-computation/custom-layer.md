# 사용자 지정 레이어

딥 러닝의 성공을 뒷받침하는 한 가지 요인은 다양한 작업에 적합한 아키텍처를 설계하기 위해 창의적인 방식으로 구성할 수 있는 광범위한 계층의 가용성입니다.예를 들어, 연구자들은 이미지, 텍스트를 처리하고 순차 데이터를 반복하고 동적 프로그래밍을 수행하기 위해 특별히 계층을 발명했습니다.조만간 딥 러닝 프레임워크에 아직 존재하지 않는 계층을 만나거나 발명하게 될 것입니다.이러한 경우 사용자 지정 레이어를 만들어야 합니다.이 섹션에서는 그 방법을 보여줍니다. 

## (**매개 변수가 없는 레이어**)

먼저 자체 매개 변수가 없는 사용자 지정 레이어를 구성합니다.:numref:`sec_model_construction`에서 블록에 대한 소개를 기억한다면 친숙해 보일 것입니다.다음 `CenteredLayer` 클래스는 입력값에서 평균을 뺍니다.이를 구축하려면 기본 계층 클래스에서 상속하고 순방향 전파 함수를 구현하면됩니다.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)
```

레이어를 통해 일부 데이터를 공급하여 레이어가 의도한 대로 작동하는지 확인해 보겠습니다.

```{.python .input}
layer = CenteredLayer()
layer(np.array([1, 2, 3, 4, 5]))
```

```{.python .input}
#@tab pytorch
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
```

```{.python .input}
#@tab tensorflow
layer = CenteredLayer()
layer(tf.constant([1, 2, 3, 4, 5]))
```

이제 [**더 복잡한 모델을 구성할 때 레이어를 구성 요소로 통합할 수 있습니다.**]

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
```

```{.python .input}
#@tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

추가 온전성 검사로 네트워크를 통해 무작위 데이터를 보내고 평균이 실제로 0인지 확인할 수 있습니다.부동 소수점 숫자를 다루기 때문에 양자화로 인해 여전히 0이 아닌 매우 작은 숫자를 볼 수 있습니다.

```{.python .input}
Y = net(np.random.uniform(size=(4, 8)))
Y.mean()
```

```{.python .input}
#@tab pytorch
Y = net(torch.rand(4, 8))
Y.mean()
```

```{.python .input}
#@tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

## [**매개 변수가 있는 레이어**]

이제 간단한 계층을 정의하는 방법을 알았으므로 훈련을 통해 조정할 수 있는 매개 변수를 사용하여 계층을 정의해 보겠습니다.기본 제공 함수를 사용하여 몇 가지 기본적인 하우스 키핑 기능을 제공하는 매개 변수를 만들 수 있습니다.특히 모델 매개변수의 액세스, 초기화, 공유, 저장 및 로드를 제어합니다.이렇게 하면 다른 이점 중에서도 모든 사용자 지정 계층에 대해 사용자 지정 직렬화 루틴을 작성할 필요가 없습니다. 

이제 완전 연결 계층의 자체 버전을 구현해 보겠습니다.이 계층에는 가중치를 나타내는 매개 변수와 치우침을 나타내는 매개 변수 두 개가 필요합니다.이 구현에서는 ReLU 활성화를 기본값으로 베이크합니다.이 계층에는 각각 입력 및 출력 수를 나타내는 `in_units` 및 `units`와 같은 인수를 입력해야 합니다.

```{.python .input}
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

```{.python .input}
#@tab pytorch
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

```{.python .input}
#@tab tensorflow
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)
```

:begin_tab:`mxnet, tensorflow`
다음으로 `MyDense` 클래스를 인스턴스화하고 모델 매개 변수에 액세스합니다.
:end_tab:

:begin_tab:`pytorch`
다음으로 `MyLinear` 클래스를 인스턴스화하고 모델 매개 변수에 액세스합니다.
:end_tab:

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
#@tab pytorch
linear = MyLinear(5, 3)
linear.weight
```

```{.python .input}
#@tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

[**사용자 지정 레이어를 사용하여 순방향 전파 계산을 직접 수행할 수 있습니다.**]

```{.python .input}
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
#@tab pytorch
linear(torch.rand(2, 5))
```

```{.python .input}
#@tab tensorflow
dense(tf.random.uniform((2, 5)))
```

또한 (**사용자 지정 레이어를 사용하여 모델을 구성하십시오.**) 일단 내장 완전 연결 계층처럼 사용할 수 있습니다.

```{.python .input}
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

## 요약

* 기본 계층 클래스를 통해 사용자 지정 레이어를 디자인할 수 있습니다.이를 통해 라이브러리의 기존 레이어와 다르게 동작하는 유연한 새 레이어를 정의할 수 있습니다.
* 일단 정의되면, 사용자 지정 계층은 임의의 컨텍스트 및 아키텍처에서 호출될 수 있습니다.
* 레이어에는 내장 함수를 통해 생성할 수 있는 로컬 파라미터가 있을 수 있습니다.

## 연습문제

1. 입력을 받고 텐서 감소를 계산하는 계층을 설계합니다. 즉, $y_k = \sum_{i, j} W_{ijk} x_i x_j$를 반환합니다.
1. 데이터의 푸리에 계수의 선행 절반을 반환하는 계층을 설계합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/58)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/59)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/279)
:end_tab:
