# Parameter Management
# 파라미터 관리

0.15.0

Once we have chosen an architecture
and set our hyperparameters,
we proceed to the training loop,
where our goal is to find parameter values
that minimize our loss function.
After training, we will need these parameters
in order to make future predictions.
Additionally, we will sometimes wish
to extract the parameters
either to reuse them in some other context,
to save our model to disk so that
it may be executed in other software,
or for examination in the hope of
gaining scientific understanding.

아키텍처를 선정하고 하이퍼파라미터들을 설정한 후에 ,우리는 손실 함수를 최소화하는 파라이터 값들을 찾기 위한 학습 룹으로 진행합니다. 학습이 완료되면, 이 후에 예측을 수행하기 위해서 이 파라미터들이 필요할 것입니다. 더불어, 다른 용도로 재사용을 하거나, 다른 소프트웨어에서 사용할 수 있도록 모델을 디스크에 저장하거나, 또는 과학적인 이해를 얻기 위해서 이 파라미터들을 추출하기를 원할 수도 있습니다.

Most of the time, we will be able
to ignore the nitty-gritty details
of how parameters are declared
and manipulated, relying on deep learning frameworks
to do the heavy lifting.
However, when we move away from
stacked architectures with standard layers,
we will sometimes need to get into the weeds
of declaring and manipulating parameters.
In this section, we cover the following:

딥러닝 프래임워크가 모든 것을 해주기 때문에, 대분의 경우 우리는 파라미터들이 어떻게 선언되는지에 대한 자세한 내용은 무시할 수 있습니다. 하지만, 표준 층을 사용하는 기본으로 제공되는 아키텍처를 벗어나야 할 경우에는, 파라미터들이 선언되고 조작되는지를 알아야 합니다. 이 절에서 우리는 다음을 다룰 것입니다.

* Accessing parameters for debugging, diagnostics, and visualizations.
* Parameter initialization.
* Sharing parameters across different model components.



- 디버깅, 분석, 또는 시각화를 위해서 파라미터에 접근하기
- 파라미터 초기화
- 다른 모델 컴포넌트와 파라미터 공유하기

We start by focusing on an MLP with one hidden layer.

하나의 은닉층을 갖는 MLP를 살펴보는 것으로 시작해 봅니다.

```{.python .input}
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X)  # Forward computation
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
import numpy as np

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X)
```

## Parameter Access
## 파라미터 접근

Let us start with how to access parameters
from the models that you already know.
When a model is defined via the `Sequential` class,
we can first access any layer by indexing
into the model as though it were a list.
Each layer's parameters are conveniently
located in its attribute.
We can inspect the parameters of the second fully-connected layer as follows.

우리가 만든 모델의 파라미터들을 어떻게 접근하는지부터 살펴보겠습니다.  `Sequential` 클래스로 모델을 정의하면, 임의의 층을 리스트의 원소를 접근하듯이 인덱스를 사용해서 접근할 수 있습니다. 각 층의 파라미터들은 층의 속성으로 저장되어 있습니다. 우리는 다음과 같이 두번째 완전 연결층의 파라미터들을 조사할 수 있습니다.

```{.python .input}
print(net[1].params)
```

```{.python .input}
#@tab pytorch
print(net[2].state_dict())
```

```{.python .input}
#@tab tensorflow
print(net.layers[2].weights)
```

The output tells us a few important things.
First, this fully-connected layer
contains two parameters,
corresponding to that layer's
weights and biases, respectively.
Both are stored as single precision floats (float32).
Note that the names of the parameters
allow us to uniquely identify
each layer's parameters,
even in a network containing hundreds of layers.

이 결과는 우리에게 중요한 몇 가지 것들을 이야기해주고 있습니다. 첫번째, 이 완전 연결층은 두 개의 파라미터, 즉 이 층의 가중치와 편향을 갖습니다. 두 파라미터는 모두 단정밀도 실수(single precision float, float32) 형태로 저장되어 있습니다. 파라이터 이름을 사용하면 수백 개의 층을 갖고 있는 네트워크일지라도 각 층의 파라미터를 바로 접근할 수 있음을 기억하세요.


### Targeted Parameters
### 특정된 파라미터

Note that each parameter is represented
as an instance of the parameter class.
To do anything useful with the parameters,
we first need to access the underlying numerical values.
There are several ways to do this.
Some are simpler while others are more general.
The following code extracts the bias
from the second neural network layer, which returns a parameter class instance, and 
further accesses that parameter's value.

각 파라미터는 파라미터 클래스의 인스턴스로 표현됨을 기억하세요. 파라미터로 어떤 유용한 일을 하기 위해서는 실제 수치 값을 접근할 필요가 있습니다. 이를 위한 여러가지 방법들이 있습니다. 어떤 방법은 간단하고, 어떤 방법은 좀 더 일반적입니다. 다음 코드는 두번째 신경망 층의 편향을 얻어내는데, 이는 파라미터 클래스의 인스턴스이고, 이 인스턴스를 접근해서 파라미터 값을 얻습니다.

```{.python .input}
print(type(net[1].bias))
print(net[1].bias)
print(net[1].bias.data())
```

```{.python .input}
#@tab pytorch
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

```{.python .input}
#@tab tensorflow
print(type(net.layers[2].weights[1]))
print(net.layers[2].weights[1])
print(tf.convert_to_tensor(net.layers[2].weights[1]))
```

:begin_tab:`mxnet,pytorch`
Parameters are complex objects,
containing values, gradients,
and additional information.
That's why we need to request the value explicitly.

파라미터는 값, 미분값 그리고 부가적인 정보를 담고 있는 복잡한 객체입니다. 이것이 우리가 값을 명시적으로 요청해야하는 이유입니다. 

In addition to the value, each parameter also allows us to access the gradient. Because we have not invoked backpropagation for this network yet, it is in its initial state.

값과 더불어서 각 파라미터를 통해서 미분값에도 접근할 수 있습니다. 우리는 아직 이 네트워크의 역전파를 수행하지 않았기 때문에, 미분값은 초기 상태입니다.

:end_tab:

```{.python .input}
net[1].weight.grad()
```

```{.python .input}
#@tab pytorch
net[2].weight.grad == None
```

### All Parameters at Once
### 모든 파라미터를 한꺼번에 접근하기

When we need to perform operations on all parameters,
accessing them one-by-one can grow tedious.
The situation can grow especially unwieldy
when we work with more complex blocks (e.g., nested blocks),
since we would need to recurse
through the entire tree to extract
each sub-block's parameters. Below we demonstrate accessing the parameters of the first fully-connected layer vs. accessing all layers.

모든 파라미터들에 어떤 연산을 수행할 경우, 하나씩 접근하는 것은 아주 귀찮은 일입니다. 만약에 중첩된 블럭들을 갖는 더 복잡한 블럭을 다룰 경우에는, 각 서브-블럭의 파라미터를 추출하기 위해서 전체 트리를 접근해야하기 때문에, 상황은 다루기 어려워지게 될 것입니다. 다음은 첫번째 완전 연결층의 파라미터를 접근하는 방법과 전체 층들의 파라미터를 접근하는 방법을 보여줍니다.

```{.python .input}
print(net[0].collect_params())
print(net.collect_params())
```

```{.python .input}
#@tab pytorch
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

```{.python .input}
#@tab tensorflow
print(net.layers[1].weights)
print(net.get_weights())
```

This provides us with another way of accessing the parameters of the network as follows.

이것은 네트워크 파라미터들을 접근하는 다른 방법을 아래와 같이 제시합니다.

```{.python .input}
net.collect_params()['dense1_bias'].data()
```

```{.python .input}
#@tab pytorch
net.state_dict()['2.bias'].data
```

```{.python .input}
#@tab tensorflow
net.get_weights()[1]
```

### Collecting Parameters from Nested Blocks
## 중첩된 블록의 파라미터들 모으기

Let us see how the parameter naming conventions work
if we nest multiple blocks inside each other.
For that we first define a function that produces blocks
(a block factory, so to speak) and then
combine these inside yet larger blocks.

블록을 다른 블록에 중첩한 경우, 파라미터 이름 규칙이 어떻게 작동하는지 보겠습니다. 이를 위해서 우선 블럭을 생성하는 함수(일종의 블록 팩토리)를 정의한 후, 더 큰 블록에 포함시킵니다.

```{.python .input}
def block1():
    net = nn.Sequential()
    net.add(nn.Dense(32, activation='relu'))
    net.add(nn.Dense(16, activation='relu'))
    return net

def block2():
    net = nn.Sequential()
    for _ in range(4):
        # Nested here
        net.add(block1())
    return net

rgnet = nn.Sequential()
rgnet.add(block2())
rgnet.add(nn.Dense(10))
rgnet.initialize()
rgnet(X)
```

```{.python .input}
#@tab pytorch
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # Nested here
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```

```{.python .input}
#@tab tensorflow
def block1(name):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation=tf.nn.relu)],
        name=name)

def block2():
    net = tf.keras.Sequential()
    for i in range(4):
        # Nested here
        net.add(block1(name=f'block-{i}'))
    return net

rgnet = tf.keras.Sequential()
rgnet.add(block2())
rgnet.add(tf.keras.layers.Dense(1))
rgnet(X)
```

Now that we have designed the network,
let us see how it is organized.

이제 네트워크를 설계했으니, 어떻게 구성되어 있는지 살펴봅시다.

```{.python .input}
print(rgnet.collect_params)
print(rgnet.collect_params())
```

```{.python .input}
#@tab pytorch
print(rgnet)
```

```{.python .input}
#@tab tensorflow
print(rgnet.summary())
```

Since the layers are hierarchically nested,
we can also access them as though
indexing through nested lists.
For instance, we can access the first major block,
within it the second sub-block,
and within that the bias of the first layer,
with as follows.

층들이 계층적으로 중첩되어 있으니, 중첩 리스트들의 인덱스를 사용해서 접근할 수 있습니다. 예를 들어, 첫번째 주요 블록을 접근한 후, 그 안에 있는 두번째 서버-블록을 접근하고, 그 블록의 첫번째 층의 편향을 다음과 같이 접근할 수 있습니다.

```{.python .input}
rgnet[0][1][0].bias.data()
```

```{.python .input}
#@tab pytorch
rgnet[0][1][0].bias.data
```

```{.python .input}
#@tab tensorflow
rgnet.layers[0].layers[1].layers[1].weights[1]
```

## Parameter Initialization
## 파라미터 초기화

Now that we know how to access the parameters,
let us look at how to initialize them properly.
We discussed the need for proper initialization in :numref:`sec_numerical_stability`.
The deep learning framework provides default random initializations to its layers.
However, we often want to initialize our weights
according to various other protocols. The framework provides most commonly
used protocols, and also allows to create a custom initializer.

자 이제 파라미터를 어떻게 접근하는지 알았으니, 적절하게 파라미터들을 초기화하는 방법을 보겠습니다. :numref:`sec_numerical_stability`에서 적절한 초기화의 필요성에 대해서 알아봤습니다. 딥러닝 프레임워크는 각 층을 임의의 값으로 초기화하는 방법을 제공합니다. 하지만, 다양한 다른 방법들을 사용해서 가중치를 초기화하고 싶은 경우가 종종 있습니다. 프레임워크는 가장 일반적으로 사용되는 초기화 방법을 제공하는 동시에 커스텀 초기화를 만들 수 있도록 허용합니다.

:begin_tab:`mxnet`
By default, MXNet initializes weight parameters by randomly drawing from a uniform distribution $U(-0.07, 0.07)$,
clearing bias parameters to zero.
MXNet's `init` module provides a variety
of preset initialization methods.

기본적으로 MXNet은 가중치 파라미터를 $U(-0.07, 0.07)$ 를 따르는 균일 분포에서 무작위로 값을 추출해서 초기화하고, 편향은 모두 0으로 설정합니다. MXNet의 `init`  모듈은 미리 설정된 초기화 방법들을 다양하게 제공합니다. 

:end_tab:

:begin_tab:`pytorch`
By default, PyTorch initializes weight and bias matrices
uniformly by drawing from a range that is computed according to the input and output dimension.
PyTorch's `nn.init` module provides a variety
of preset initialization methods.

기본적으로 PyTorch는 가중치 파라미터를 입력와 출력의 차원에 따라 계산된 범위에서 값을 균일하게 추출해서 가중치와 편향 행렬들을 초기화합니다. PyTorch의  `nn.init`  모듈은 미리 설정된 초기화 방법들을 다양하게 제공합니다. 

:end_tab:

:begin_tab:`tensorflow`
By default, Keras initializes weight matrices uniformly by drawing from a range that is computed according to the input and output dimension, and the bias parameters are all set to zero.
TensorFlow provides a variety of initialization methods both in the root module and the `keras.initializers` module.

기본적으로 Keras는 가중치 파라미터를 입력와 출력의 차원에 따라 계산된 범위에서 값을 균일하게 추출해서 가중치와 편향 행렬들을 초기화합니다. PyTorch의  `keras.initializers`  모듈은 미리 설정된 초기화 방법들을 다양하게 제공합니다. :end_tab:

### Built-in Initialization
### 기본으로 제공되는 초기화

Let us begin by calling on built-in initializers.
The code below initializes all weight parameters
as Gaussian random variables
with standard deviation 0.01, while bias parameters cleared to zero.

기본으로 제공되는 초기화를 사용하는 것부터 시작하겠습니다. 아래 코드는 모든 가중치 파라미터들을 표준편차 0.01을 갖는 가우시안 랜덤 변수로 초기화하고, 편향 파라미터는 모두 0으로 설정합니다.

```{.python .input}
# Here `force_reinit` ensures that parameters are freshly initialized even if
# they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(X)
net.weights[0], net.weights[1]
```

We can also initialize all the parameters
to a given constant value (say, 1).

모든 파라미터들을 특정 상수 값으로도 초기화 할 수 있습니다. (예를 들어 1)

```{.python .input}
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
net.weights[0], net.weights[1]
```

We can also apply different initializers for certain blocks.
For example, below we initialize the first layer
with the Xavier initializer
and initialize the second layer
to a constant value of 42.

특정 블록에 대해서 다른 초기화를 적용하는 것도 가능합니다. 예를 들어, 첫번째 층은 Xavier 초기화를 적용하고, 두번째 층은 상수 값 42로 초기화를 다음 코드와 같이 할 수 있습니다.

```{.python .input}
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input}
#@tab pytorch
def xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        torch.nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(1)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

### Custom Initialization
### 커스텀 초기화

Sometimes, the initialization methods we need
are not provided by the deep learning framework.
In the example below, we define an initializer
for any weight parameter $w$ using the following strange distribution:

때로는 딥러닝 프레임워크가 우리가 필요한 초기화 방법을 제공하지 않기도 합니다. 다음 예제는 아래와 같은 이상한 분포를 따라서 어떤 가중치 파라미터  $w$ 를 초기화하는 것을 정의합니다.
$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ with probability } \frac{1}{4} \\
            0    & \text{ with probability } \frac{1}{2} \\
        U(-10, -5) & \text{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
Here we define a subclass of the `Initializer` class.
Usually, we only need to implement the `_init_weight` function
which takes a tensor argument (`data`)
and assigns to it the desired initialized values.

 `Initializer` 클래스의 서브 클래스를 정의합니다. 보통은 텐서 인자(`data`)를 받는 `_init_weight` 함수만 구현하고, 원하는 초기값으로 설정하기만 하면 됩니다. 

:end_tab:

:begin_tab:`pytorch`
Again, we implement a `my_init` function to apply to `net`.

다시 우리는 `my_init` 함수를 구현해서  `net` 에 적용합니다.

:end_tab:

:begin_tab:`tensorflow`
Here we define a subclass of `Initializer` and implement the `__call__`
function that return a desired tensor given the shape and data type.

여기서 우리는 `Initializer` 의 서브 클래스를 정의하고, 특정 모양와 데이터 타입을 갖는 원하는 텐서를 반환하는  `__call__` 함수를 구현합니다.

:end_tab:

```{.python .input}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input}
#@tab pytorch
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) 
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input}
#@tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        return tf.random.uniform(shape, dtype=dtype)

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])
```

Note that we always have the option
of setting parameters directly.

파라미터를 직접 설정하는 옵션도 있습니다.

```{.python .input}
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input}
#@tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

:begin_tab:`mxnet`
A note for advanced users:
if you want to adjust parameters within an `autograd` scope,
you need to use `set_data` to avoid confusing
the automatic differentiation mechanics.

고급 사용자를 위한 노트:  `autograd` 범위 안에서 파라미터를 조정하고 싶다면, 자동 미분 방식을 혼동시키지 않기 위해서 `set_data` 를 사용해야합니다.

:end_tab:

## Tied Parameters

## 묶인 파라미터들

Often, we want to share parameters across multiple layers.
Let us see how to do this elegantly.
In the following we allocate a dense layer
and then use its parameters specifically
to set those of another layer.

종종 우리는 여러 층들이 파라미터들을 공유하기를 원합니다. 이것을 멋지게하는 방법을 살펴보겠습니다. 아래 코드에서 우리는 덴스층(dense)을 할당하고, 이것의 파라미터를 다른 층의 파라미터를 설정하는데 사용합니다.

```{.python .input}
net = nn.Sequential()
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)

# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input}
#@tab pytorch
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
#@tab tensorflow
# tf.keras behaves a bit differently. It removes the duplicate layer
# automatically
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
# Check whether the parameters are different
print(len(net.layers) == 3)
```

:begin_tab:`mxnet,pytorch`
This example shows that the parameters
of the second and third layer are tied.
They are not just equal, they are
represented by the same exact tensor.
Thus, if we change one of the parameters,
the other one changes, too.
You might wonder,
when parameters are tied
what happens to the gradients?
Since the model parameters contain gradients,
the gradients of the second hidden layer
and the third hidden layer are added together
during backpropagation.

이 예제는 두번째 층과 세번째 층의 파라미터들이 서로 묶여있는 것을 보여줍니다. 이 파라미터들의 값이 서로 같은 것만 아니라, 정확하게 동일한 텐서를 사용합니다. 즉, 파라미터들 중 하나를 바꾸면, 다른 층의 파라미터도 함께 바뀝니다. 여러분은 파라미터가 묶여있으면, 미분에는 어떤일이 생길지 궁금해 할 것입니다. 모델 파라미터들은 미분을 갖고 있기 때문에, 두번재 은닉층의 미분과 세번째 은닉층의 미분은 역전파 과정에서 더해집니다.

:end_tab:

## Summary

* We have several ways to access, initialize, and tie model parameters.
* We can use custom initialization.



- 모델 파라미터를 접근하고, 초기화하고, 묶는 여러가지 방법이 있습니다.
- 커스텀 초기화를 사용할 수 있습니다.


## Exercises

1. Use the `FancyMLP` model defined in :numref:`sec_model_construction` and access the parameters of the various layers.
1. Look at the initialization module document to explore different initializers.
1. Construct an MLP containing a shared parameter layer and train it. During the training process, observe the model parameters and gradients of each layer.
1. Why is sharing parameters a good idea?



1. :numref:`sec_model_construction` 에서 정의한 `FancyMLP` 를 사용해서, 다양한 층들의 파라미터들을 접근해보세요.
2. 다양한 초기화 방법을 알아보기 위해서, 초기화 모듈 문서를 보세요.
3. 공유된 파라미터 층을 갖는 MLP을 만들고 학습을 수행하세요. 학습 과정에서 모델의 파라미터들과 각 층의 미분을 관찰해보세요.
4. 파라미터 공유는 왜 좋은 생각일까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/269)
:end_tab:
