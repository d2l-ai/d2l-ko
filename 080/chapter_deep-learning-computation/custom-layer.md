# Custom Layers
# 커스텀 층

0.15.0

One factor behind deep learning's success
is the availability of a wide range of layers
that can be composed in creative ways
to design architectures suitable
for a wide variety of tasks.
For instance, researchers have invented layers
specifically for handling images, text,
looping over sequential data,
and
performing dynamic programming.
Sooner or later, you will encounter or invent
a layer that does not exist yet in the deep learning framework.
In these cases, you must build a custom layer.
In this section, we show you how.

딥러닝을 성공으로 이끈 한 요소는 다양한 과제들에 적합한 아키텍처를 창의적인 방식으로 조합할 수 있도록 해준 다양한 층들의 있다는 것입니다. 예를 들어, 연구자들은 이미지와 텍스트를 처리하거나, 순차적인 데이터를 반복하거나, 동적 프로그래밍을 수행하기 위한 층들을 발명했습니다. 조만간 여러분은 딥러닝 프레임워크에 존재하지 않는 층을 접하게 되거나 발명할 것입니다.

## Layers without Parameters
## 파리미터가 없는 층들

To start, we construct a custom layer
that does not have any parameters of its own.
This should look familiar if you recall our
introduction to block in :numref:`sec_model_construction`.
The following `CenteredLayer` class simply
subtracts the mean from its input.
To build it, we simply need to inherit
from the base layer class and implement the forward propagation function.

우선 우리는 파라메터를 전혀 갖지 않는 커스텀 층을 만들어 보겠습니다. :numref:`sec_model_construction`에서 소개한 블록을 생각해보면, 여러분은 이 층에 대해서 익숙할 것입니다. 아래 `CenteredLayer` 클래스는 입력에서 평균을 뺍니다. 우리는 간단하게 기본 층 클래스를 상속하고, 정방향 전파 함수를 구현하면 됩니다.

```{.python .input}
from mxnet import gluon, np, npx
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
import torch.nn.functional as F

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

Let us verify that our layer works as intended by feeding some data through it.

자 그럼 데이터를 입력해서 우리의 층이 올바르게 동적하는지 보겠습니다.

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

We can now incorporate our layer as a component
in constructing more complex models.

이제 우리는 이 층을 더 복잡한 모델을 생성하는데 컴포넌트로 사용할 수 있습니다.

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

As an extra sanity check, we can send random data
through the network and check that the mean is in fact 0.
Because we are dealing with floating point numbers,
we may still see a very small nonzero number
due to quantization.

역시 잘 작동하는지 확인하기 위해서, 네트워크에 임의의 데이터를 입력한 결과의 평균이 실제로 0인 것을 확인해 봅니다. 부동 소수점을 다루고 있기 때문에 양자화(quantization)으로 인해서 결과가 아주 작은 0이 아닌 숫자일 수 있습니다.

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

## Layers with Parameters
## 파라미터를 갖는 층들

Now that we know how to define simple layers,
let us move on to defining layers with parameters
that can be adjusted through training.
We can use built-in functions to create parameters, which
provide some basic housekeeping functionality.
In particular, they govern access, initialization,
sharing, saving, and loading model parameters.
This way, among other benefits, we will not need to write
custom serialization routines for every custom layer.

간단한 층을 정의하는 법을 배웠으니, 학습을 통해서 조정될 수 있는 파라미터를 갖는 층을 정의해보겠습니다. 몇 가지 기본적인 관리 기능을 제공하는 빌트인 함수를 사용할 수 있습니다. 구체적으로는 이 함수들은 모델 파라미터에 대한 접근 관리, 초기화, 공유, 저장, 그리고 로딩 기능을 제공합니다. 다른 이점도 있지만, 이 기능 덕분에 우리는 모든 커스텀 층에 커스텀 직렬화를 일일이 작성할 필요가 없습니다.

Now let us implement our own version of the  fully-connected layer.
Recall that this layer requires two parameters,
one to represent the weight and the other for the bias.
In this implementation, we bake in the ReLU activation as a default.
This layer requires to input arguments: `in_units` and `units`, which
denote the number of inputs and outputs, respectively.

자 그럼 완전 연결층의 우리만의 버전을 만들어 봅시다. 이 층은 가중치와 편향을 대표하는 두 개의 파라미터를 갖는다는 것을 기억하세요. 이 구현에서 우리는 ReLU 활성화를 기본적으로 포함시키겠습니다. 이 층은 두 인자들 `in_units` 와 `units` 를 갖는데, 이는 각각 입력의 개수와 출력의 개수를 지정합니다.

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

Next, we instantiate the `MyDense` class
and access its model parameters.

다음으로  `MyDense` 클래스의 인스턴스를 만들어서 모델 파라미터들을 접근해 봅니다. 

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
#@tab pytorch
dense = MyLinear(5, 3)
dense.weight
```

```{.python .input}
#@tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

We can directly carry out forward propagation calculations using custom layers.

커스텀 층들을 이용해서 정방향 전파 연산을 직접 수행할 수도 있습니다.

```{.python .input}
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
#@tab pytorch
dense(torch.rand(2, 5))
```

```{.python .input}
#@tab tensorflow
dense(tf.random.uniform((2, 5)))
```

We can also construct models using custom layers.
Once we have that we can use it just like the built-in fully-connected layer.

또한, 커스텀 층을 사용한 모델을 만들수도 있고, 만든 후에는 빌트인 완전 연결층과 동일하게 사용할 수 있습니다.

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

## Summary
## 요약

* We can design custom layers via the basic layer class. This allows us to define flexible new layers that behave differently from any existing layers in the library.
* Once defined, custom layers can be invoked in arbitrary contexts and architectures.
* Layers can have local parameters, which can be created through built-in functions.

* 기본 층 클래스를 활용해서 커스텀 층을 설계할 수 있습니다. 이는 라이브러리에서 제공하는 층들과 다르게 동작하는 유연한 새로운 층들을 정의할 수 있도록 해줍니다.
* 커스텀 층들이 정의된 후에는 임의의 설정이나 아키텍처에서 사용될 수 있습니다.
* 층은 빌트인 함수들을 통해서 만들어진 로컬 파라미터를 갖을 수 있습니다.

## Exercises
## 연습문제

1. Design a layer that takes an input and computes a tensor reduction,
   i.e., it returns $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
1. Design a layer that returns the leading half of the Fourier coefficients of the data.


1. 입력을 받아서 텐서 축소를 계산하는 층을 설계하세요. 즉,  $y_k = \sum_{i, j} W_{ijk} x_i x_j$ 를 반환합니다.
1. 데이터의 퓨리에 계수의 앞의 반을 반환하는 층을 설계하세요.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/58)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/59)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/279)
:end_tab:
