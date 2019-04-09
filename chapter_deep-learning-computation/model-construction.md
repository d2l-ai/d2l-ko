# 층(layer)과 블럭(Block)

딥러닝이 유명해질 수 있었던 중요 요소들 중에 하나는 바로 강력한 소프트웨어입니다. 반도체 설계를 하는데 엔지니어들이 논리 회로를 트랜지스터로 구현하던 것에서 코드를 작성하는 것으로 넘어간 것과 같은 일이 딥 네트워크 설계에도 비슷하게 일어나고 있습니다. 앞 장들은 단일 뉴런으로 부터 뉴런으로 구성된 전체 층들로 옮겨가는 것을 보여줬습니다. 하지만, 컴퓨터 비전 문제를 풀기 위해서 2016년에 [He et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) 에 의해서 제안된 ResNet-152의 경우처럼 152개의 층들을 갖는 네트워크 층들을 사용한 네트워크 설계 방법 조차도 지루할 수 있습니다.

이런 네트워크는 많은 정도로 반복되는 부분을 갖고, 반복되는 (또는 비슷하게 설계된) 층들의 *블럭*들로 구성됩니다. 이들 블럭들은 더 복잡한 네트워크 디자인을 구성하는 기본 요소가 됩니다. 간략하게 말하면, 블럭은 하나 또는 그 이상의 층의 조합니다. 마치 레고 공장이 만든 블럭을 이용해서 멋진 구조물을 만들 수 있는 것처럼, 이 디자인은 요청에 따라서 브럭을 생성하는 코드의 도움으로 만들어질 수 있습니다.

아주 간단한 블럭부터 살펴보겠습니다. 이 블럭은 [앞 장](../chapter_deep-learning-basics/mlp-gluon.md) 에서 본 다층 퍼셉트론(multilayer perception)을 위한 것입니다. 일반적인 방법으로 두개의 층을 갖는 네트워크를 다음과 같이 만들 수 있습니다.

```{.python .input  n=1}
from mxnet import nd
from mxnet.gluon import nn

x = nd.random.uniform(shape=(2, 20))

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(x)
```

이 코드는 256개의 유닛(unit)들을 갖는 은닉층(hidden layer) 한개를 포함한 네트워크를 생성합니다. 은닉층(hidden layer) 은 ReLU 활성화(activation)로 연결되어 있고, 결과 층의 10개 유닛(unit)들로 연결되어 있습니다. 여기서 우리는 `nn.Sequential` 생성자를 사용해서 빈 네트워크를 만들고, 그 다음에 층들을 추가했습니다. 아직은 `nn.Sequential` 내부에서 어떤 일이 벌어지는 지는 미스테리로 남아있습니다. 아래 내용을 통해서 이것은 실제로 블럭을 생성하고 있는 것을 확인할 것입니다. 이 블럭들은 더 큰 결과물로 합쳐지는데 때로는 재귀적으로 합쳐지기도 합니다. 아래 그림은 이 것이 어떻게 일어나는지 보여줍니다.

![Multiple layers are combined into blocks](../img/blocks.svg)

층(layer)을 정의하는 것부터 (하나 또는 그 이상이 층들을 갖는) 블럭을 정의하는 데 필요한 다양한 절차에 대해서 설명하겠습니다. 블럭은 멋진 층과 비슷하게 동작합니다. 즉, 블럭은 아래 기능을 제공합니다.

1. 데이터 (입력을) 받아야합니다.
1. 의미 있는 결과를 출력해야 합니다. 이는 `forward` 라고 불리는 함수에서 처리합니다. 원하는 결과을 얻기 위해서 `net(x)` 를 통해서 블럭을 수행할 수도 있는데, 실제로는 순전파(forward propagation)을 수행하는 `forward` 함수를 호출합니다.
1. `backward` 함수가 호출되면 입력에 대해서 그래티언트(gradient)를 생성해야 합니다. 일반적으로 이것은 자동으로 이뤄집니다.
1. 블럭에 속한 파라미터들을 저장해야 합니다. 예를 들면, 위 블럭은 두개의 은닉층(hidden layer)를 갖는데, 파라미터를 저장할 공간이 있어야 합니다.

## 커스텀 블럭

`nn.Block` 클래스는 우리가 필요로 하는 기능들을 제공합니다. `nn` 모듈에서 제공하는 모델 생성자로, 우리가 원하는 모델을 정의하기 위해서 상속하는 클래스입니다. 아래 코드는 이 절을 시작할 때 언급한 다층 퍼셉트론(multilayer perceptron)을 생성하기 위해서 Block 클래스를 상속하고 있습니다. 여기서  `MLP` 클래스는 Block 클래스의  `__init__` 과 `forward` 함수를 오버라이드하고 있습니다. 이 함수들은 각각 모델 파라미터들을 생성하고 forward 계산을 정의하는 함수입니다. Forward 연산은 역전파(forward propagation)을 의미합니다.

```{.python .input  n=1}
from mxnet import nd
from mxnet.gluon import nn

class MLP(nn.Block):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self, **kwargs):
        # Call the constructor of the MLP parent class Block to perform the
        # necessary initialization. In this way, other function parameters can
        # also be specified when constructing an instance, such as the model
        # parameter, params, described in the following sections
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # Hidden layer
        self.output = nn.Dense(10)  # Output layer

    # Define the forward computation of the model, that is, how to return the
    # required model output based on the input x
    def forward(self, x):
        return self.output(self.hidden(x))
```

조금 더 자세히 살펴보겠습니다. `forward` 메소드는 은닉층(hidden layer) `self.hidden(x)` 를 계산하고, 그 값을 이용해서 결과층 `self.output(…)` 을 계산합니다. 이것이 이 블럭의 forward 연산에서 해야하는 일입니다.

블럭이 어떤 값을 사용해서 계산을 수행해야하는지를 알기 위해서, 우리는 우선 층들을 정의 해야합니다. 이는 `__init__` 메소드가 하는 일입니다. 블럭과 관련된 모든 파라미터들을 초기화하고, 필요한 층을 생성합니다. 그리고, 관련 층들과 클래스에 필요한 파라미터들을 정의합니다. 시스템은 그래디언트(gradient)를 자동으로 계산해주는 `backward` 메소드를 자동으로 생성해줍니다.   `initialize` 메소드도 자동으로 생성됩니다. 한번 수행해보겠습니다.

```{.python .input  n=2}
net = MLP()
net.initialize()
net(x)
```

위에서 설명했듯이, 블럭 클래스는 무엇을 하는지에 따라서 아주 다르게 정의될 수 있습니다. 예를 들어, 그것의 하위 클래스가 (Gluon에서 제공하는 `Dense` 클래스와 같은) 층이 될 수도 있고, (우리가 막 정의한 `MLP` 클래스와 같은) 모델이 될 수도 있습니다. 또는 다른 모델의 일부가 될 수도 있습니다. 이는 아주 깊은 네트워크를 디자인할 때 사용되는 방법입니다. 이 장을 통해서 우리는 이것을 아주 유연하게 사용할 수 있는 방법에 대해서 알아보겠습니다.

## Sequential 블럭

Block 클래스는 데이터흐름을 기술하는 일반 컴포넌트입니다. 사실 Sequential 클래스는 Block 클래스로부터 정의됩니다. 모델을 forward 연산은 각 층에 대한 연산의 단순한 연결이기 때문에, 우리는 모델을 아주 간단한 방법으로 정의할 수 있습니다. Sequential 클래스의 목적은 유용한 편의 함수들을 제공하는 것에 있습니다. 특히, `add` 메소드는 연결된 Block 하위클래스의 인스턴스를 하나씩 더할 수 있게 해주고, 모델의 forward 연산은 이 인스턴스들을 더하기 순서대로 계산합니다.

아래 코드에서 `MySequential` 클래스를 정의했는데, 이는 Sequential 클래스와 같은 기능을 제공합니다. 이를 통해서 Sequential 클래스가 어떻게 동작하는 이해하는데 도움이 될 것입니다.

```{.python .input  n=3}
class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        # Here, block is an instance of a Block subclass, and we assume it has
        # a unique name. We save it in the member variable _children of the
        # Block class, and its type is OrderedDict. When the MySequential
        # instance calls the initialize function, the system automatically
        # initializes all members of _children
        self._children[block.name] = block

    def forward(self, x):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._children.values():
            x = block(x)
        return x
```

`add` 메소드가 핵심입니다. 이 메소드는 순서가 있는 사전(dictionary)에 블럭을 추가하는 일을 합니다. 순전파(forward propagation)가 호출되면 이 블럭들은 순서대로 수행됩니다. MLP가 어떻게 구현되는지 보겠습니다.

```{.python .input  n=4}
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(x)
```

실제로,  [“다층 페셉트론(multilayer perceptron)의 간결한 구현”](../chapter_deep-learning-basics/mlp-gluon.md) 에서 Sequential 클래스를 사용한 것과 `MySequential` 클래스를 사용한 것이 다르지 않다는 것을 볼 수 있습니다.


## 코드와 블록(Block)

Sequential 클래스가 모델 생성을 쉽게 해주고 `forward` 메소스를 별도로 구현할 필요 없게 해주지만, Block 클래스를 직접 상속하면 더 유연한 모델 생성을 할 수 있습니다. 특히, forward 메소스에서 Python의 제어 흐름을 이용하는 것을 예로 들어보겠습니다. 설명하기에 앞서서 *constant* 파라미터라는 개념에 대해서 알아보겠습니다. 이 파라미터들은 역전파(back propagation)이 호출되었을 때 사용되지는 않습니다. 추상적으로 들릴 수 있지만, 실제 일어나는 일이 그렇습니다. 어떤 함수가 있다고 가정합니다.

$$f(\mathbf{x},\mathbf{w}) = 3 \cdot \mathbf{w}^\top \mathbf{x}.$$

이 경우, 3이 상수(constant) 파라미터입니다. 우리는 3을 다른 값, 예를 들어  $c$ 로 바꿔서 다음과 같이 표현할 수 있습니다.

$$f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}.$$

 $c$ 의 값을 조절할 수 있게 된 것 이외에는 바뀐 것이 없습니다.  $\mathbf{w}$ 와 $\mathbf{x}$ 만을 생각해보면 여전히 상수입니다. 하지만, Gluon은 이것을 미리 알지 못하기 때문에, 도움을 주는 것이 필요합니다. 이렇게 하는 것은 Gluon이 변하지 않는 파라미터에 대해서는 신경 쓰지 않도록 할 수 있기 때문에 코드가 더 빠르게 수행되게 해줍니다. `get_constant` 메소드을 이용하면 됩니다. 실제 어떻게 구현되는지 살펴보겠습니다.

```{.python .input  n=5}
class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        # Random weight parameters created with the get_constant are not
        # iterated during training (i.e. constant parameters)
        self.rand_weight = self.params.get_constant(
            'rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        # Use the constant parameters created, as well as the relu and dot
        # functions of NDArray
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        x = self.dense(x)
        # Here in Control flow, we need to call asscalar to return the scalar
        # for comparison
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()
```

`FancyMLP` 모델에서 `rand_weight`라는 상수 가중치를 정의했습니다. (이 변수는 모델 파라에터는 아니다라는 것을 알아두세요). 그리고, 행렬 곱하기 연산 (`nd.dot()`)을 수행하고, 같은 `Dense` 층를 재사용합니다. 서로 다른 파라미터 세트를 사용한 두 개의 덴스층(dense layer)를 사용했던 것과 다른 형태로 구현되었음을 주목하세요. 우리는 대신, 같은 네트워크를 두 번 사용했습니다.  네트워크의 여러 부분이 같은 파라미터를 공유하는 경우 딥 네트워크에서 이 것을 파라미터가 서로 묶여 있다(tied)라고 말하기도 합니다. 이 클래스에 대한 인스턴스를 만들어서 데이터를 입력하면 어떤 일이 일어나는지 보겠습니다.

```{.python .input  n=6}
net = FancyMLP()
net.initialize()
net(x)
```

네트워크를 만들 때 이런 방법을 섞어서 사용하지 않을 이유가 없습니다. 아래 예제를 보면 어쩌면 키메라와 닮아 보일 수도 있고 조금 다르게 말하면, [Rube Goldberg Machine](https://en.wikipedia.org/wiki/Rube_Goldberg_machine)와 비슷하다고 할 수도 있습니다. 즉, 개별적인 블럭을 합쳐서 블럭을 만들고 이렇게 만들어진 블럭이 다시 블럭으로 사용될 수 있는 것을 예제를 다음과 같이 만들어 볼 수 있습니다. 더 나아가서는 같은 forward 함수 안에서 여러 전략을 합치는 것도 가능합니다. 아래 코드가 그런 예입니다.

```{.python .input  n=7}
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FancyMLP())

chimera.initialize()
chimera(x)
```

## 컴파일

여러분이 관심이 많다면 이런 접근 방법에 대한 효율에 대한 의심을 할 것입니다. 결국에는 많은 사전(dictionary) 참조, 코드 수행과 다른 Python 코드들 수행하면서 성능이 높은 딥러닝 라이브러리를 만들어야 합니다. Python의 [Global Interpreter Lock](https://wiki.python.org/moin/GlobalInterpreterLock) 은 아주 잘 알려진 문제로, 아주 성능이 좋은 GPU를 가지고 있을지라도 단일 CPU 코어에서 수행되는 Python 프로그램이 다음에 무엇을 해야할지를 알려주지 기다려야하기 때문에 딥러닝 환경에서 성능에 안좋은 영향을 미칩니다. 당연하게도 아주 나쁜 상황이지만, 이를 우회하는 여러 방법들이 존재합니다. Python 속도를 향상시키는 방법은 이 모든 것을 모두 제거하는 것이 최선입니다.

Gluon은  [Hybridization](../chapter_computational-performance/hybridize.md) 기능을 통해서 해결하고 있습니다. Python 코드 블럭이 처음 수행되면 Gluon 런타임은 무엇이 수행되었는지를 기록하고, **이후에 수행될 때는 Python을 호출하지 않고 빠른 코드를 수행합니다. ** 이 방법은 속도를 상당히 빠르게 해주지만, 제어 흐름을 다루는데 주의를 기울여야 합니다. 하이브리드화(Hybridization)와 컴파일(compilation)에 대해서 더 관심이 있다면 이 장을 마치고, 해당 내용이 있는 절을 읽어보세요.


## 요약

* 층들은 블럭입니다.
* 많은 층들이 하나의 블럭이 될 수 있습니다.
* 많은 블럭들이 하나의 블럭이 될 수 있습니다.
* 코드도 블럭이 될 수 있습니다.
* 블럭은 파라미터 초기화, 역전파(back propagation), 또는 관련된 일을 대신 처리해줍니다.
* 층들과 블럭들을 순차적으로 연결하는 것은 `Sequential` 블럭에 의해서 처리됩니다.

## 문제

1. What kind of error message will you get when calling an `__init__` method whose parent class not in the `__init__` function of the parent class?
1. `FancyMLP` 클래스에서 `asscalar` 함수를 삭제하면 어떤 문제가 발생하나요? 
1. `NestMLP` 클래스에서 Sequential 클래스의 인스턴스로 정의된 `self.net` 을 `self.net = [nn.Dense(64, activation='relu'), nn.Dense(32, activation='relu')]` 로 바꾸면 어떤 문제가 발생하나요?
1. 두 블럭 (`net1` 과 `net2`)를 인자로 받아서 forward pass의 두 네트워크의 결과를 연결해서 반환하는 블럭을 작성해보세요. (이는 parallel 블럭이라고 합니다)
1. 같은 네트워크의 여러 인스턴스를 연결하고자 가정합니다. 같은 블럭의 여러 인스턴스를 생성하는 factory 함수를 작성하고, 이를 사용해서 더 큰 네트워크를 만들어 보세요.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2325)

![](../img/qr_model-construction.svg)
