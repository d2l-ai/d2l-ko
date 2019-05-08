# 파라미터 관리

딥 네트워크 학습의 최종 목표는 주어진 아키텍처에 가장 잘 맞는 파라미터 값들을 찾는 것입니다. 일반적인 것 또는 표준에 준하는 것들을 다룰 때는 `nn.Sequential` 클래스가 이를 위한 완벽한 도구가 될 수 있습니다. 하지만, 소수의 모델이 완전히 표준이고, 대부분의 과학자들은 독창적인 것을 만들기를 원합니다. 이 절에서는 파라미터를 다루는 방법에 대해서 살펴보겠습니다. 좀 더 자세하게는 아래와 같은 것들을 포함합니다.

* 디버깅이나 분석을 위해서 파라미터를 접근하고, 그것들을 시각화하거나 저장하는 것을 통해서 커스텀 모델을 어떻게 만들어야 하는지 이해를 시작하겠습니다.
* 다음으로는 초기화 목적 등을 위해서 특별한 방법으로 파라미터들을 설정해야 하는데, 이를 위해서 파라미터 초기화 도구의 구조에 대해서 논의합니다.
* 마지막으로 일부 파라미터를 공유하는 네트워크를 만들면서 이 내용들이 어떻게 적용되는지 보겠습니다.

지금까지 그랬듯이 은닉층(hidden layer)을 갖는 다층 퍼셉트론(multilayer perceptron)으로부터 시작하겠습니다. 이를 이용해서 다양한 특징들을 살펴봅니다.

```{.python .input  n=1}
from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()  # Use the default initialization method

x = nd.random.uniform(shape=(2, 20))
net(x)  # Forward computation
```

## 파라미터 접근

Sequential 클래스의 경우, 네트워크의 각 층의 인덱스를 사용해서 파라미터를 쉽게 접근할 수 있습니다. params 변수가 필요한 데이터를 가지고 있습니다. 자 그럼 첫번째 층의 파라미터를 조사하는 것을 직접해 보겠습니다.

```{.python .input  n=2}
print(net[0].params)
print(net[1].params)
```

위 코드의 수행 결과는 많은 것을 우리에게 알려줍니다. 첫번째 정보는 예상대로 이 층은 파라미터들의 두 개의 세트, `dense0_weight` 와 `dense0_bias`,로 구성되어 있는 것을 확인할 수 있습니다. 이 값들은 모두 싱글 프리시전(single precision)이고, 입력 차원이 20이고 출력 차원이 256인 첫번째 층에 필요한 모양(shape)을 갖고 있습니다. 특히, 파라미터들의 이름이 주어지는데 이는 아주 유용합니다. 이름을 사용하면 간단하지 않은 구조를 갖는 수백개의 층들로 구성된 네트워크에서 파라미터를 쉽게 지정할 수 있기 때문입니다. 두 번째 층도 같은 방식으로 구성되어 있는 것을 확인할 수 있습니다.

### 지정된 파라미터

파라미터를 가지고 뭔가 유용한 일을 하기를 원한다면 이 값들을 접근할 수 있어야 합니다. 간단한 방법부터 일반적인 방법까지 다양한 방법이 있는데, 몇 가지를 살펴보겠습니다.

```{.python .input  n=3}
print(net[1].bias)
print(net[1].bias.data())
```

첫번째 코드는 두번째 층의 편향(bias)를 출력합니다. 이는 데이터, 그래디언트(gradient) 그리고 추가적인 정보를 가지고 있는 객체이기에, 우리는 데이터를 명시적으로 접근해야 합니다. 우리는 편향(bias)을 모두 0으로 초기화했기 때문에 편향(bias)이 모두 0임을 기억해두기 바랍니다. 이 값은 파라미터의 이름, `dense0_weight`, 을 이용해서 직접 접근할 수도 있습니다. 이렇게 할 수 있는 이유는 모든 레이어는 직접 접근할 수 있는 고유의 파라미터 사전(dictionary)를 갖고있기 때문입니다. 이 두 방법은 완전이 동일하나, 첫번째 방법이 조금 더 읽기 쉽습니다.

```{.python .input  n=4}
print(net[0].params['dense0_weight'])
print(net[0].params['dense0_weight'].data())
```

가중치들이 모두 0이 아닌 값으로 되어 있음을 주목하세요. 우리가 네트워크를 만들 때, 이 값들은 난수값으로 초기화했기 때문에 그렇습니다. `data`  함수만 있는 것이 아닙니다. 예를 들어 파라미터에 대해서 그래디언트(gradient)를 계산하고자 할 수도 있습니다. 이 결과는 가중치와 같은 모양(shape)을 갖게 됩니다. 하지만, 역전파(back propagation)을 아직 실행하지 않았기 때문에 이 값들은 모두 0으로 보여질 것입니다.

```{.python .input  n=5}
net[0].weight.grad()
```

### 한번에 모든 파라미터 지정

위 방법으로 파라미터를 접근하는 것은 다소 지루할 수 있습니다. 특히, 더 복잡한 블럭들을 갖거나, 블럭들로 구성된 블럭 (심지어는 블럭들을 블럭들의 블럭)으로 구성된 네트워크인 경우, 블럭들이 어떻게 생성되었는지 알기 위해서 전체 트리를 모두 뒤져봐야 하는 경우가 그런 예입니다. 이를 피하기 위해서, 블럭은 `collect_params` 라는 메소드를 제공하는데 이를 이용하면 네트워크의 모든 파라미터를 하나의 사전(dictionary)에 담아주고, 쉽게 조회할 수 있습니다. 이는 내부적으로 블럭의 모든 구성 요소들을 방문하면서 필요한 경우 서브블럭들에 `collect_params` 함수를 호출하는 식으로 동작합니다. 차이를 확인하기 위해서 아래 코드를 살펴 보겠습니다.

```{.python .input  n=6}
# parameters only for the first layer
print(net[0].collect_params())
# parameters of the entire network
print(net.collect_params())
```

이렇게 해서 네트워크의 파라미터를 접근하는 세번째 방법을 배웠습니다. 두번째 층의 편향(bias) 값을 확인하는 코드는 아래와 같이 간단하게 작성할 수 있습니다.

```{.python .input  n=7}
net.collect_params()['dense1_bias'].data()
```

이 책에서 설명을 계속하면서, 블럭들의 하위 블럭에 이름이 어떻게 부여되는지 보게될 것입니다. (그 중에, Sequential의 경우는 숫자를 할당합니다.) 이름 할당 규칙은 필요한 파라미터만 필터링하는 정규식을 사용할 수 있게해서 아주 편리합니다.

```{.python .input  n=8}
print(net.collect_params('.*weight'))
print(net.collect_params('dense0.*'))
```

### 루브 골드버그가 다시 공격하다.

블럭들이 중첩되어 있는 경우 파라미터의 이름이 어떤식으로 매겨지는지 보겠습니다. 이를 위해서 우리는 블럭들을 생성하는 함수(block factory 라고 불릴 수 있는) 를 정의하고, 이를 이용해서 더 큰 블럭들이 블럭을 포함시켜보겠습니다.

```{.python .input  n=20}
def block1():
    net = nn.Sequential()
    net.add(nn.Dense(32, activation='relu'))
    net.add(nn.Dense(16, activation='relu'))
    return net

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add(block1())
    return net

rgnet = nn.Sequential()
rgnet.add(block2())
rgnet.add(nn.Dense(10))
rgnet.initialize()
rgnet(x)
```

네트워크를 설계했으니, 어떻게 구성되는지 확인해봅니다. `collect_params` 를 이용하면 이름과 논리적인 구조에 대한 정보를 얻을 수 있습니다.

```{.python .input}
print(rgnet.collect_params)
print(rgnet.collect_params())
```

층들이 계층적으로 생성되어 있으니, 우리도 층들을 그렇게 접근할 수 있습니다. 예를 들어서, 첫번째 큰 블럭의 두번째 하위 블럭의 첫번째 층의 편향(bias) 값은 다음과 같이 접근이 가능합니다.

```{.python .input}
rgnet[0][1][0].bias.data()
```

## 파라미터 초기화

자 이제 파라미터를 어떻게 접근할 수 있는지 알게되었으니, 파라미터를 어떻게 적절하게 초기화할 수 있을지를 살펴볼 차례입니다. 이전 장에서 [초기화](../chapter_deep-learning-basics/numerical-stability-and-init.md) 가 왜 필요한지를 설명했습니다. 기본 설명으로는 MXNet은 가중치 행렬은  $U[-0.07, 0.07]$ 을 따르는 균일한 난수로, 편향(bias) 파라미터는 모두 0으로 설정합니다. 하지만, 때로는 가중치 값을 다르게 초기화 해야할 필요가 있습니다. MXNet의 `init` 모듈은 미리 설정된 다양한 초기화 방법들을 제공하는데, 만약 특별한 방법으로 초기화하는 것이 필요하다면 몇 가지 추가적인 일이 필요합니다.

### 제공되는 초기화

빌트인 초기화 방법들을 우선 살펴보겠습니다. 아래 코드는 모든 파라미터를 Gaussian 확률 변수로 초기화하는 예제입니다.

```{.python .input  n=9}
# force_reinit ensures that the variables are initialized again, regardless of
# whether they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

만약 파라미터들을 모두 1로 초기화하고 싶다면, 초기화 방법을 `Constant(1)` 로 바꾸기만 하면됩니다.

```{.python .input  n=10}
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

만약 특정 파라미터만 다른 방법으로 초기화를 하고 싶다면, 해당하는 서브블럭에 초기화 함수를 지정하는 것으로 간단히 구현할 수 있습니다. 예를 들어, 아래 코드는 두번째 층을 42라는 값으로 초기화하고, 첫번째 층의 가중치들은 `Xavier` 초기화 방법을 적용하고 있습니다.

```{.python .input  n=11}
net[1].initialize(init=init.Constant(42), force_reinit=True)
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
print(net[1].weight.data()[0,0])
print(net[0].weight.data()[0])
```

### 커스텀 초기화

때로는 우리가 필요한 초기화 방법이 `init` 모듈에 없을 수도 있습니다. 이 경우에는, `Initializer` 클래스의 하위 클래스를 정의해서 다른 초기화 메소드와 같은 방법으로 사용할 수 있습니다. 보통은, `_init_weight` 함수만 구현하면 됩니다. 이 함수는 입력 받은 NDArray를 원하는 초기값으로 바꿔줍니다. 아래 예제에서는 이를 잘 보여주기 위해서 다소 이상하고 특이한 분포를 사용해서 값을 초기화합니다.
$$
\begin{aligned}
    w \sim \begin{cases}
        U[5, 10] & \text{ with probability } \frac{1}{4} \\
            0    & \text{ with probability } \frac{1}{2} \\
        U[-10, -5] & \text{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

```{.python .input  n=12}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[0]
```

이 기능이 충분하지 않을 경우에는, 파라미터 값을 직접 설정할 수도 있습니다. `data()` 는 NDArray를 반환하기 때문에, 이를 이용하면 일반적인 행렬처럼 사용하면 됩니다. 고급 사용자들을 위해서 조금 더 설명하면, `autograd` 범위 안에서 파라미터를 조정하는 경우에는, 자동 미분 기능이 오작동하지 않도록 `set_data` 를 사용해야하는 것을 기억해두세요.

```{.python .input  n=13}
net[0].weight.data()[:] += 1
net[0].weight.data()[0,0] = 42
net[0].weight.data()[0]
```

## 묶인(Tied) 파라미터들

다른 어떤 경우에는, 여러 층들이 모델 파라미터를 공유하는 것이 필요하기도 합니다. 예를 들면, 좋은 단어 임베딩을 찾는 경우, 단어 인코딩과 디코딩에 같은 파라미터를 사용하도록 하는 결정할 수 있습니다. 이런 경우는 [Blocks](model-construction.md)에서도 소개되었습니다. 이것을 보다 깔끔하게 구현하는 방법을 알아보겠습니다. 아래 코드에서는 덴스층(dense layer)을 하나 정의하고, 다른 층에 파라미터값을 동일하게 설정하는 것을 보여주고 있습니다.

```{.python .input  n=14}
net = nn.Sequential()
# We need to give the shared layer a name such that we can reference its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

x = nd.random.uniform(shape=(2, 20))
net(x)

# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0,0] = 100
# Make sure that they're actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

위 예제는 두번째, 세번째 층의 파라미터가 묶여있는 것(tied)을 보여줍니다. 이 파라미터들은 값이 같은 수준이 아니라, 동일합니다. 즉, 하나의 파라미터를 바꾸면 다른 파라미터의 값도 함께 바뀝니다. 그래디언트(gradient)들에 일어나는 현상은 아주 독창적입니다. 모델은 파라미터는 그래디언트(gradient)를 갖고 있기 때문에, 두번째와 세번째 층의 그래디언트(gradient)들은 역전파(back propagation) 단계에서 `shared.params.grad()` 함수에 의해서 누적됩니다.

## 요약

* 모델 파라미터를 접근하고, 초기화하고, 서로 묶는 다양한 방법이 있습니다.
* 커스텀 초기화를 사용할 수 있습니다.
* Gluon은 독특하고 계층적인 방법으로 파라미터에 접근하는 정교한 방법을 제공합니다.


## 문제

1. [이전 절](model-construction.md) 의 FancyMLP 정의를 사용해서, 다양한 레이어의 파라미터에 접근해보세요.
1.  [MXNet documentation](http://beta.mxnet.io/api/gluon-related/mxnet.initializer.html) 의 다양한 초기화 방법들을 살펴보세요.
1. `net.initialize()` 수행 후와 `net(x)` 수행 전에 모델 파라미터를 확인해서, 모델 파라미터들의 모양(shape)를 관찰해보세요. 무엇 바뀌어 있고, 왜 그럴까요?
1. 파라미터를 공유하는 레이어를 갖는 다층 퍼셉트론(multilayer perceptron)을 만들어서 학습을 시켜보세요. 학습 과정을 수행하면서 모델 각 층의 파라미터들과 그래디언트(gradient) 값을 관찰해보세요.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2326)

![](../img/qr_parameters.svg)
