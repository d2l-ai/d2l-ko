# 파라미터 관리

아키텍처를 선택하고 하이퍼 매개 변수를 설정 한 후에는 손실 함수를 최소화하는 매개 변수 값을 찾는 것이 목표 인 훈련 루프로 진행합니다.교육을 마친 후에는 향후 예측을 위해 이러한 매개 변수가 필요합니다.또한 매개 변수를 추출하여 다른 맥락에서 재사용하거나, 모델을 디스크에 저장하여 다른 소프트웨어에서 실행하거나, 과학적 이해를 얻기 위해 검사를 위해 매개 변수를 추출하려는 경우가 있습니다. 

대부분의 경우 매개 변수를 선언하고 조작하는 방법에 대한 핵심적인 세부 사항을 무시하고 딥 러닝 프레임 워크에 의존하여 무거운 작업을 수행 할 수 있습니다.그러나 표준 레이어가있는 스택 아키텍처에서 벗어날 때 매개 변수를 선언하고 조작하는 잡초에 들어가야 할 때가 있습니다.이 섹션에서는 다음 내용을 다룹니다. 

* 디버깅, 진단 및 시각화를 위한 매개 변수에 액세스합니다.
* 매개 변수 초기화.
* 여러 모델 구성 요소 간에 매개 변수를 공유합니다.

(**숨겨진 레이어 하나가 있는 MLP에 초점을 맞추는 것부터 시작합니다.**)

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

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X)
```

## [**파라미터 액세스**]

이미 알고 있는 모델에서 매개변수에 액세스하는 방법부터 시작하겠습니다.`Sequential` 클래스를 통해 모델을 정의하면 먼저 목록인 것처럼 모델을 인덱싱하여 모든 레이어에 액세스할 수 있습니다.각 레이어의 파라미터는 속성에 편리하게 위치합니다.두 번째 완전 연결 계층의 매개 변수를 다음과 같이 검사 할 수 있습니다.

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

출력은 몇 가지 중요한 사항을 알려줍니다.첫째, 이 완전 연결 계층에는 각각 해당 계층의 가중치와 편향에 해당하는 두 개의 파라미터가 포함됩니다.둘 다 단정밀도 부동 소수점 (float32) 으로 저장됩니다.파라미터 이름을 사용하면 수백 개의 계층이 포함된 네트워크에서도 각 계층의 파라미터를 고유하게 식별할 수 있습니다. 

### [**대상 매개변수**]

각 매개 변수는 매개 변수 클래스의 인스턴스로 표시됩니다.매개 변수를 사용하여 유용한 작업을 수행하려면 먼저 기본 숫자 값에 액세스해야 합니다.여러 가지 방법으로 이 작업을 수행할 수 있습니다.일부는 더 단순하지만 다른 일부는 더 일반적입니다.다음 코드는 파라미터 클래스 인스턴스를 반환하는 두 번째 신경망 계층에서 바이어스를 추출하고 추가로 해당 파라미터 값에 액세스합니다.

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
매개 변수는 값, 그라디언트 및 추가 정보를 포함하는 복잡한 개체입니다.따라서 값을 명시적으로 요청해야 합니다. 

값 외에도 각 매개 변수를 사용하여 그래디언트에 액세스 할 수도 있습니다.이 네트워크에 대해 역전파를 아직 호출하지 않았으므로 초기 상태입니다.
:end_tab:

```{.python .input}
net[1].weight.grad()
```

```{.python .input}
#@tab pytorch
net[2].weight.grad == None
```

### [**한 번에 모든 매개 변수**]

모든 매개 변수에 대해 작업을 수행해야 할 때 매개 변수에 하나씩 액세스하는 것이 지루할 수 있습니다.좀 더 복잡한 블록 (예: 중첩 블록) 으로 작업 할 때 상황이 특히 어려워 질 수 있습니다. 각 하위 블록의 매개 변수를 추출하려면 전체 트리를 재귀적으로 반복해야하기 때문입니다.아래에서는 첫 번째 완전 연결 계층의 파라미터에 액세스하는 것과 모든 레이어에 액세스하는 방법을 보여줍니다.

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

이를 통해 다음과 같이 네트워크의 매개 변수에 액세스하는 또 다른 방법이 제공됩니다.

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

### [**중첩된 블록에서 파라미터 수집**]

여러 블록을 서로 중첩하면 매개 변수 명명 규칙이 어떻게 작동하는지 살펴 보겠습니다.이를 위해 먼저 블록을 생성하는 함수 (말하자면 블록 팩토리) 를 정의한 다음 더 큰 블록 안에 결합합니다.

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

이제 [**네트워크를 설계했으니, 네트워크가 어떻게 구성되어 있는지 살펴보자.**]

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

계층이 계층적으로 중첩되어 있으므로 중첩 목록을 통해 인덱싱하는 것처럼 계층에 액세스할 수도 있습니다.예를 들어, 첫 번째 주요 블록, 두 번째 하위 블록 내에서, 그리고 첫 번째 레이어의 바이어스에 다음과 같이 액세스 할 수 있습니다.

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

## 매개 변수 초기화

이제 매개 변수에 액세스하는 방법을 알았으므로 매개 변수를 올바르게 초기화하는 방법을 살펴 보겠습니다.:numref:`sec_numerical_stability`에서 적절한 초기화의 필요성에 대해 논의했습니다.딥러닝 프레임워크는 해당 계층에 디폴트 랜덤 초기화를 제공합니다.하지만 다양한 다른 프로토콜에 따라 가중치를 초기화하고자 하는 경우가 많습니다.프레임워크는 가장 일반적으로 사용되는 프로토콜을 제공하며 사용자 지정 이니셜라이저를 만들 수도 있습니다.

:begin_tab:`mxnet`
기본적으로 MXNet은 균일 분포 $U(-0.07, 0.07)$에서 무작위로 추출하여 편향 모수를 0으로 지워 가중치 모수를 초기화합니다.MXNet의 `init` 모듈은 다양한 사전 설정 초기화 방법을 제공합니다.
:end_tab:

:begin_tab:`pytorch`
기본적으로 PyTorch는 입력 및 출력 차원에 따라 계산된 범위에서 추출하여 가중치 행렬과 편향 행렬을 균일하게 초기화합니다.파이토치의 `nn.init` 모듈은 다양한 사전 설정 초기화 방법을 제공합니다.
:end_tab:

:begin_tab:`tensorflow`
기본적으로 Keras는 입력 및 출력 차원에 따라 계산되는 범위에서 가져와 가중치 행렬을 균일하게 초기화하며 바이어스 매개 변수는 모두 0으로 설정됩니다.텐서플로우는 루트 모듈과 `keras.initializers` 모듈 모두에서 다양한 초기화 방법을 제공합니다.
:end_tab:

### [**내장 초기화**]

내장 이니셜라이저를 호출하여 시작하겠습니다.아래 코드는 모든 가중치 파라미터를 표준편차가 0.01인 가우스 랜덤 변수로 초기화하고 편향 파라미터는 0으로 지웁니다.

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

모든 매개 변수를 주어진 상수 값 (예: 1) 으로 초기화 할 수도 있습니다.

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

[**특정 블록에 다른 이니셜라이저를 적용할 수도 있습니다.**] 예를 들어 아래에서는 Xavier 이니셜라이저로 첫 번째 레이어를 초기화하고 두 번째 레이어를 상수 값 42로 초기화합니다.

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
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

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
        1, kernel_initializer=tf.keras.initializers.Constant(42)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

### [**사용자 지정 초기화**]

필요한 초기화 방법이 딥 러닝 프레임워크에서 제공되지 않는 경우가 있습니다.아래 예에서는 다음과 같은 이상한 분포를 사용하여 가중치 매개 변수 $w$에 대한 이니셜라이저를 정의합니다. 

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
여기서는 `Initializer` 클래스의 하위 클래스를 정의합니다.일반적으로 텐서 인수 (`data`) 를 취하고 원하는 초기화 값을 할당하는 `_init_weight` 함수 만 구현하면됩니다.
:end_tab:

:begin_tab:`pytorch`
다시 말하지만, `net`에 적용할 `my_init` 함수를 구현합니다.
:end_tab:

:begin_tab:`tensorflow`
여기서는 `Initializer`의 서브 클래스를 정의하고 모양과 데이터 유형이 주어진 원하는 텐서를 반환하는 `__call__` 함수를 구현합니다.
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
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor        

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

매개 변수를 직접 설정할 수 있는 옵션이 항상 있습니다.

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
고급 사용자를 위한 참고 사항: `autograd` 범위 내에서 파라미터를 조정하려면 자동 차별화 메커니즘을 혼동하지 않도록 `set_data`를 사용해야 합니다.
:end_tab:

## [**묶인 매개변수**]

여러 레이어에서 파라미터를 공유하고자 하는 경우가 많습니다.이 작업을 우아하게 수행하는 방법을 살펴 보겠습니다.다음에서는 고밀도 레이어를 할당한 다음 해당 매개 변수를 사용하여 다른 레이어의 매개 변수를 설정합니다.

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
이 예제에서는 두 번째 레이어와 세 번째 레이어의 매개 변수가 연결되어 있음을 보여줍니다.그것들은 단지 동일한 것이 아니라 동일한 정확한 텐서로 표현됩니다.따라서 매개 변수 중 하나를 변경하면 다른 매개 변수도 변경됩니다.매개 변수가 연결되면 그라디언트에 어떤 일이 발생하는지 궁금 할 것입니다.모델 파라미터에는 그래디언트가 포함되어 있으므로 역전파 중에 두 번째 은닉 레이어와 세 번째 은닉 레이어의 그래디언트가 함께 추가됩니다.
:end_tab:

## 요약

* 모델 매개변수에 액세스, 초기화 및 연결하는 방법에는 여러 가지가 있습니다.
* 사용자 지정 초기화를 사용할 수 있습니다.

## 연습문제

1. :numref:`sec_model_construction`에 정의된 `FancyMLP` 모델을 사용하고 다양한 레이어의 매개변수에 액세스합니다.
1. 초기화 모듈 문서를 보고 다양한 이니셜라이저를 살펴보세요.
1. 공유 파라미터 계층을 포함하는 MLP를 생성하고 훈련시킵니다.훈련 과정에서 각 계층의 모델 파라미터와 기울기를 관찰합니다.
1. 매개 변수를 공유하는 것이 좋은 이유는 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/269)
:end_tab:
