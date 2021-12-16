# 레이어 및 블록
:label:`sec_model_construction`

신경망을 처음 도입했을 때 단일 출력을 가진 선형 모델에 중점을 두었습니다.여기서 전체 모델은 단 하나의 뉴런으로 구성되어 있습니다.단일 뉴런 (i) 은 일부 입력 세트를 취하고, (ii) 대응하는 스칼라 출력값을 생성하며, (iii) 관심 있는 일부 목적 함수를 최적화하기 위해 업데이트할 수 있는 관련 파라미터 세트를 가지고 있습니다.그런 다음 출력값이 여러 개인 네트워크에 대해 생각하기 시작한 후 벡터화된 산술을 활용하여 뉴런의 전체 계층을 특성화했습니다.개별 뉴런과 마찬가지로, 계층은 (i) 입력 세트를 취하고, (ii) 대응하는 출력값을 생성하며, (iii) 조정 가능한 파라미터 세트로 설명됩니다.소프트맥스 회귀를 통해 작업했을 때 단일 계층 자체가 모델이었습니다.그러나 이후에 MLP를 도입하더라도 모델이 이와 동일한 기본 구조를 유지하는 것으로 생각할 수 있습니다. 

흥미롭게도 MLP의 경우 전체 모델과 해당 구성 계층이 모두 이 구조를 공유합니다.전체 모델은 원시 입력 (특징) 을 받고 출력 (예측) 을 생성하며 매개 변수 (모든 구성 계층에서 결합된 매개 변수) 를 보유합니다.마찬가지로 각 개별 계층은 입력 (이전 계층에서 제공) 을 수집하고 출력 (후속 계층에 대한 입력) 을 생성하고 후속 계층에서 역방향으로 흐르는 신호에 따라 업데이트되는 조정 가능한 파라미터 세트를 보유합니다. 

뉴런, 계층 및 모델이 비즈니스를 수행하기에 충분한 추상화를 제공한다고 생각할 수도 있지만 개별 계층보다 크지 만 전체 모델보다 작은 구성 요소에 대해 이야기하는 것이 편리하다는 것을 알 수 있습니다.예를 들어, 컴퓨터 비전에서 널리 사용되는 ResNet-152 아키텍처는 수백 개의 레이어를 가지고 있습니다.이러한 레이어는*레이어 그룹*의 반복 패턴으로 구성됩니다.이러한 네트워크를 한 번에 한 계층씩 구현하는 것은 지루할 수 있습니다.이러한 우려는 단순한 가설이 아닙니다. 이러한 설계 패턴은 실제로 일반적입니다.위에서 언급한 ResNet 아키텍처는 인식 및 감지 :cite:`He.Zhang.Ren.ea.2016`에서 2015년 ImageNet 및 COCO 컴퓨터 비전 대회에서 우승했으며 많은 비전 작업에서 여전히 인기 있는 아키텍처로 남아 있습니다.레이어가 다양한 반복 패턴으로 배열되는 유사한 아키텍처는 이제 자연어 처리 및 음성을 포함한 다른 영역에서 보편적입니다. 

이러한 복잡한 네트워크를 구현하기 위해 신경망*block*이라는 개념을 소개합니다.블록은 단일 레이어, 여러 레이어로 구성된 컴포넌트 또는 전체 모델 자체를 설명할 수 있습니다.블록 추상화 작업의 한 가지 이점은 블록 추상화를 종종 재귀적으로 더 큰 아티팩트로 결합할 수 있다는 것입니다.이것은 :numref:`fig_blocks`에 설명되어 있습니다.필요에 따라 임의의 복잡성의 블록을 생성하는 코드를 정의함으로써 놀랍도록 간결한 코드를 작성하면서도 복잡한 신경망을 구현할 수 있습니다. 

![Multiple layers are combined into blocks, forming repeating patterns of larger models.](../img/blocks.svg)
:label:`fig_blocks`

프로그래밍 관점에서 블록은*class*로 표현됩니다.모든 하위 클래스는 입력을 출력으로 변환하는 순방향 전파 함수를 정의해야 하며 필요한 매개 변수를 저장해야 합니다.일부 블록에는 매개변수가 전혀 필요하지 않습니다.마지막으로, 그래디언트를 계산하려면 블록에 역전파 함수가 있어야 합니다.다행히도 자체 블록을 정의할 때 자동 차별화 (:numref:`sec_autograd`에 도입) 가 제공하는 비하인드 스토리 마법으로 인해 매개 변수와 순방향 전파 함수에 대해서만 걱정할 필요가 있습니다. 

[**시작하기 위해 MLP를 구현하는 데 사용한 코드**](:numref:`sec_mlp_concise`) 를 다시 살펴보겠습니다.다음 코드는 256 유닛과 ReLU 활성화로 완전히 연결된 하나의 은닉 계층이 있는 네트워크를 생성하고, 그 뒤에 10개 유닛으로 구성된 완전히 연결된 출력 계층 (활성화 함수 없음) 이 이어집니다.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

X = tf.random.uniform((2, 20))
net(X)
```

:begin_tab:`mxnet`
이 예제에서는 `nn.Sequential`을 인스턴스화하고 반환된 객체를 `net` 변수에 할당하여 모델을 생성했습니다.다음으로 `add` 함수를 반복적으로 호출하여 실행해야 하는 순서대로 레이어를 추가합니다.요컨대, `nn.Sequential`은 글루온에 블록을 나타내는 클래스인 `Block`의 특별한 종류를 정의합니다.구성 요소 `Block`의 정렬 된 목록을 유지합니다.`add` 함수를 사용하면 연속된 각 `Block`을 목록에 쉽게 추가할 수 있습니다.각 계층은 그 자체가 `Block`의 하위 클래스인 `Dense` 클래스의 인스턴스입니다.순방향 전파 (`forward`) 함수도 매우 간단합니다. 목록의 각 `Block`을 함께 연결하여 각각의 출력을 다음 입력으로 전달합니다.지금까지 우리는 출력을 얻기 위해 `net(X)` 구성을 통해 모델을 호출했습니다.이것은 실제로 `Block` 클래스의 `__call__` 함수를 통해 달성된 매끄러운 파이썬 트릭인 `net.forward(X)`의 약칭입니다.
:end_tab:

:begin_tab:`pytorch`
이 예제에서는 실행 순서대로 레이어를 인수로 전달하여 `nn.Sequential`를 인스턴스화하여 모델을 구성했습니다.간단히 말해서, `nn.Sequential`는 특수한 종류의 `Module`를 정의합니다, 파이토치에서 블록을 나타내는 클래스입니다.구성 요소 `Module`s의 정렬 된 목록을 유지합니다.두 개의 완전히 연결된 계층은 각각 `Module`의 하위 클래스인 `Linear` 클래스의 인스턴스입니다.순방향 전파 (`forward`) 함수도 매우 간단합니다. 목록의 각 블록을 함께 연결하여 각 블록의 출력을 다음 입력으로 전달합니다.지금까지 우리는 출력을 얻기 위해 `net(X)` 구성을 통해 모델을 호출했습니다.이것은 실제로 `net.__call__(X)`의 속기에 불과합니다.
:end_tab:

:begin_tab:`tensorflow`
이 예제에서는 실행 순서대로 레이어를 인수로 전달하여 `keras.models.Sequential`를 인스턴스화하여 모델을 구성했습니다.요컨대, `Sequential`는 케라스에서 블록을 나타내는 클래스인 `keras.Model`의 특별한 종류를 정의합니다.구성 요소 `Model`의 정렬 된 목록을 유지합니다.두 개의 완전히 연결된 계층 각각은 `Dense` 클래스의 인스턴스이며, 그 자체가 `Model`의 하위 클래스입니다.순방향 전파 (`call`) 함수도 매우 간단합니다. 목록의 각 블록을 함께 연결하여 각 블록의 출력을 다음 입력으로 전달합니다.지금까지 우리는 출력을 얻기 위해 `net(X)` 구성을 통해 모델을 호출했습니다.이것은 실제로 블록 클래스의 `__call__` 함수를 통해 얻은 매끄러운 파이썬 트릭인 `net.call(X)`의 약칭입니다.
:end_tab:

## [**사용자 지정 블록**]

아마도 블록이 어떻게 작동하는지에 대한 직관을 개발하는 가장 쉬운 방법은 블록을 직접 구현하는 것입니다.사용자 지정 블록을 구현하기 전에 각 블록이 제공해야 하는 기본 기능을 간략하게 요약합니다.

:begin_tab:`mxnet, tensorflow`
1. 입력 데이터를 순방향 전파 함수의 인수로 수집합니다.
1. 순방향 전파 함수가 값을 반환하도록 하여 출력값을 생성합니다.출력의 모양이 입력과 다를 수 있습니다.예를 들어 위 모델의 첫 번째 완전 연결 계층은 임의 차원의 입력값을 수집하지만 차원 256이라는 출력값을 반환합니다.
1. 역전파 함수를 통해 액세스할 수 있는 입력에 대한 출력의 기울기를 계산합니다.일반적으로 이 작업은 자동으로 수행됩니다.
1. 순방향 전달 계산을 실행하는 데 필요한 매개변수를 저장하고 해당 매개변수에 대한 액세스를 제공합니다.
1. 필요에 따라 모델 매개변수를 초기화합니다.
:end_tab:

:begin_tab:`pytorch`
1. 입력 데이터를 순방향 전파 함수의 인수로 수집합니다.
1. 순방향 전파 함수가 값을 반환하도록 하여 출력값을 생성합니다.출력의 모양이 입력과 다를 수 있습니다.예를 들어 위 모델의 첫 번째 완전 연결 계층은 차원 20의 입력값을 수집하지만 차원 256의 출력값을 반환합니다.
1. 역전파 함수를 통해 액세스할 수 있는 입력에 대한 출력의 기울기를 계산합니다.일반적으로 이 작업은 자동으로 수행됩니다.
1. 순방향 전달 계산을 실행하는 데 필요한 매개변수를 저장하고 해당 매개변수에 대한 액세스를 제공합니다.
1. 필요에 따라 모델 매개변수를 초기화합니다.
:end_tab:

다음 스니펫에서는 256개의 은닉 유닛이 있는 하나의 은닉 레이어와 10차원 출력 레이어로 MLP에 해당하는 블록을 처음부터 코딩합니다.아래 `MLP` 클래스는 블록을 나타내는 클래스를 상속합니다.우리는 부모 클래스의 함수에 크게 의존하여 자체 생성자 (Python의 `__init__` 함수) 와 순방향 전파 함수만 제공합니다.

```{.python .input}
class MLP(nn.Block):
    # Declare a layer with model parameters. Here, we declare two
    # fully-connected layers
    def __init__(self, **kwargs):
        # Call the constructor of the `MLP` parent class `Block` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # Hidden layer
        self.out = nn.Dense(10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def forward(self, X):
        return self.out(self.hidden(X))
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the `MLP` parent class `Module` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # Hidden layer
        self.out = nn.Linear(256, 10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def forward(self, X):
        # Note here we use the funtional version of ReLU defined in the
        # nn.functional module.
        return self.out(F.relu(self.hidden(X)))
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the `MLP` parent class `Model` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__()
        # Hidden layer
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def call(self, X):
        return self.out(self.hidden((X)))
```

먼저 순방향 전파 함수에 초점을 맞추겠습니다.`X`을 입력값으로 사용하고, 활성화 함수가 적용된 은닉 표현을 계산하고, 로짓을 출력합니다.이 `MLP` 구현에서 두 계층은 모두 인스턴스 변수입니다.이것이 합리적인 이유를 확인하려면 두 MLP (`net1` 및 `net2`) 를 인스턴스화하고 서로 다른 데이터에 대해 훈련시키는 것을 상상해 보십시오.당연히 우리는 두 가지 다른 학습 모델을 나타낼 것으로 기대합니다. 

순방향 전파 함수를 호출할 때마다 생성자에서 [**MLP의 계층을 인스턴스화**] 합니다 (**그런 다음 이 레이어들을 호출합니다**).몇 가지 주요 세부 사항에 유의하십시오.첫째, 사용자 정의된 `__init__` 함수는 `super().__init__()`을 통해 상위 클래스의 `__init__` 함수를 호출하여 대부분의 블록에 적용할 수 있는 상용구 코드를 다시 작성하는 번거로움을 덜어줍니다.그런 다음 완전히 연결된 두 레이어를 인스턴스화하여 `self.hidden` 및 `self.out`에 할당합니다.new 연산자를 구현하지 않는 한 역전파 함수나 매개변수 초기화에 대해 걱정할 필요가 없습니다.시스템에서 이러한 함수를 자동으로 생성합니다.이것을 시도해 보겠습니다.

```{.python .input}
net = MLP()
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch
net = MLP()
net(X)
```

```{.python .input}
#@tab tensorflow
net = MLP()
net(X)
```

블록 추상화의 주요 장점은 다양성입니다.블록을 서브 클래스화하여 레이어 (예: 완전 연결 계층 클래스), 전체 모델 (예: 위의 `MLP` 클래스) 또는 중간 복잡성의 다양한 구성 요소를 만들 수 있습니다.컨벌루션 신경망을 다룰 때와 같이 다음 장에서 이러한 다양성을 활용합니다. 

## [**순차 블록**]

이제 `Sequential` 클래스의 작동 방식을 자세히 살펴볼 수 있습니다.`Sequential`는 다른 블록을 데이지 체인으로 연결하도록 설계되었습니다.단순화된 `MySequential`를 빌드하려면 다음 두 가지 주요 기능만 정의하면 됩니다.
1. 목록에 블록을 하나씩 추가하는 함수입니다.
2. 추가된 순서와 동일한 순서로 블록 체인을 통해 입력을 전달하는 순방향 전파 함수입니다.

다음 `MySequential` 클래스는 기본 `Sequential` 클래스와 동일한 기능을 제공합니다.

```{.python .input}
class MySequential(nn.Block):
    def add(self, block):
        # Here, `block` is an instance of a `Block` subclass, and we assume 
        # that it has a unique name. We save it in the member variable
        # `_children` of the `Block` class, and its type is OrderedDict. When
        # the `MySequential` instance calls the `initialize` function, the
        # system automatically initializes all members of `_children`
        self._children[block.name] = block

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._children.values():
            X = block(X)
        return X
```

```{.python .input}
#@tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # Here, `module` is an instance of a `Module` subclass. We save it
            # in the member variable `_modules` of the `Module` class, and its
            # type is OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._modules.values():
            X = block(X)
        return X
```

```{.python .input}
#@tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        for block in args:
            # Here, `block` is an instance of a `tf.keras.layers.Layer`
            # subclass
            self.modules.append(block)

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

:begin_tab:`mxnet`
`add` 함수는 순서가 지정된 사전 `_children`에 단일 블록을 추가합니다.모든 Gluon `Block`가 `_children` 속성을 가진 이유와 파이썬 목록을 직접 정의하는 것이 아니라 왜 사용했는지 궁금 할 것입니다.요컨대 `_children`의 가장 큰 장점은 블록의 매개 변수 초기화 중에 Gluon이 `_children` 딕셔너리를 살펴보고 매개 변수를 초기화해야하는 하위 블록을 찾는 것을 알고 있다는 것입니다.
:end_tab:

:begin_tab:`pytorch`
`__init__` 메서드에서는 순서가 지정된 사전 `_modules`에 모든 모듈을 하나씩 추가합니다.여러분은 왜 모든 `Module`가 `_modules` 어트리뷰트를 가지고 있는지, 그리고 파이썬 리스트를 직접 정의하는 것이 아니라 왜 우리가 그것을 사용했는지 궁금할 것입니다.요컨대 `_modules`의 가장 큰 장점은 모듈의 매개 변수 초기화 중에 시스템이 `_modules` 딕셔너리를 들여다보고 매개 변수를 초기화해야하는 하위 모듈을 찾는 것을 알고 있다는 것입니다.
:end_tab:

`MySequential`의 순방향 전파 함수가 호출되면 추가된 각 블록은 추가된 순서대로 실행됩니다.이제 `MySequential` 클래스를 사용하여 MLP를 다시 구현할 수 있습니다.

```{.python .input}
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```

```{.python .input}
#@tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(X)
```

이 `MySequential`의 사용은 이전에 `Sequential` 클래스에 대해 작성한 코드와 동일합니다 (:numref:`sec_mlp_concise`에 설명되어 있음). 

## [**순방향 전파 함수에서 코드 실행**]

`Sequential` 클래스는 모델 구성을 쉽게 만들어 자체 클래스를 정의하지 않고도 새로운 아키텍처를 조립할 수 있습니다.그러나 모든 아키텍처가 단순한 데이지 체인인 것은 아닙니다.더 큰 유연성이 필요한 경우 자체 블록을 정의해야 합니다.예를 들어, 순방향 전파 함수 내에서 파이썬의 제어 흐름을 실행하고 싶을 수 있습니다.또한 사전 정의된 신경망 계층에 의존하는 것이 아니라 임의의 수학적 연산을 수행할 수도 있습니다. 

지금까지 네트워크의 모든 작업이 네트워크의 활성화 및 매개 변수에 따라 작동했음을 눈치 챘을 것입니다.그러나 때로는 이전 레이어의 결과나 업데이트할 수 있는 매개변수가 아닌 용어를 통합하고자 할 수도 있습니다.이러한*상수 매개 변수*라고 부릅니다.예를 들어 함수 $f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$을 계산하는 계층을 원한다고 가정해 보겠습니다. 여기서 $\mathbf{x}$은 입력값이고 $\mathbf{w}$는 매개 변수이며 $c$은 최적화 중에 업데이트되지 않는 지정된 상수입니다.그래서 우리는 다음과 같이 `FixedHiddenMLP` 클래스를 구현합니다.

```{.python .input}
class FixedHiddenMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Random weight parameters created with the `get_constant` function
        # are not updated during training (i.e., constant parameters)
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, X):
        X = self.dense(X)
        # Use the created constant parameters, as well as the `relu` and `dot`
        # functions
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        # Reuse the fully-connected layer. This is equivalent to sharing
        # parameters with two fully-connected layers
        X = self.dense(X)
        # Control flow
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # Use the created constant parameters, as well as the `relu` and `mm`
        # functions
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # Reuse the fully-connected layer. This is equivalent to sharing
        # parameters with two fully-connected layers
        X = self.linear(X)
        # Control flow
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab tensorflow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # Random weight parameters created with `tf.constant` are not updated
        # during training (i.e., constant parameters)
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        X = self.flatten(inputs)
        # Use the created constant parameters, as well as the `relu` and
        # `matmul` functions
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # Reuse the fully-connected layer. This is equivalent to sharing
        # parameters with two fully-connected layers
        X = self.dense(X)
        # Control flow
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
```

이 `FixedHiddenMLP` 모델에서는 가중치 (`self.rand_weight`) 가 인스턴스화 시 무작위로 초기화되고 이후 일정한 은닉 레이어를 구현합니다.이 가중치는 모델 매개변수가 아니므로 역전파에 의해 업데이트되지 않습니다.그런 다음 네트워크는 이 “고정” 계층의 출력을 완전히 연결된 계층을 통해 전달합니다. 

출력을 반환하기 전에 모델이 이상한 작업을 수행했습니다.우리는 while 루프를 실행하여 $L_1$ 노름이 $1$보다 큰 조건을 테스트하고 조건을 만족할 때까지 출력 벡터를 $2$로 나누었습니다.마지막으로 `X`에서 항목의 합계를 반환했습니다.우리가 아는 한 표준 신경망은이 작업을 수행하지 않습니다.이 특정 작업은 실제 작업에서 유용하지 않을 수 있습니다.우리의 요점은 임의의 코드를 신경망 계산의 흐름에 통합하는 방법을 보여주는 것입니다.

```{.python .input}
net = FixedHiddenMLP()
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch, tensorflow
net = FixedHiddenMLP()
net(X)
```

[**블록을 함께 조립하는 다양한 방법을 혼합하고 일치시킬 수 있습니다.**] 다음 예에서는 몇 가지 창의적인 방법으로 블록을 중첩합니다.

```{.python .input}
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, X):
        return self.dense(self.net(X))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())
chimera.initialize()
chimera(X)
```

```{.python .input}
#@tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```

```{.python .input}
#@tab tensorflow
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(X)
```

## 효율성

:begin_tab:`mxnet`
열렬한 독자는 이러한 작업 중 일부의 효율성에 대해 걱정하기 시작할 수 있습니다.결국 우리는 고성능 딥 러닝 라이브러리라고 생각되는 많은 사전 조회, 코드 실행 및 기타 Pythonic 작업이 많이 발생합니다.파이썬의 [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock)의 문제점은 잘 알려져 있습니다.딥 러닝의 맥락에서, 우리는 매우 빠른 GPU가 다른 작업을 실행하기 전에 작은 CPU가 파이썬 코드를 실행할 때까지 기다려야 할 수도 있습니다.파이썬의 속도를 높이는 가장 좋은 방법은 파이썬을 완전히 피하는 것입니다. 

Gluon이 이 작업을 수행하는 한 가지 방법은 다음을 허용하는 것입니다.
*하이브리드화*, 이에 대해서는 나중에 설명하겠습니다.
여기서 파이썬 인터프리터는 블록을 처음 호출할 때 실행합니다.Gluon 런타임은 무슨 일이 일어나고 있는지 기록하고 다음에 Python에 대한 호출을 단락시킵니다.이것은 어떤 경우에는 일을 상당히 가속화 할 수 있지만 제어 흐름 (위와 같이) 이 그물을 통과하는 다른 패스에서 다른 가지로 이어질 때는주의를 기울여야합니다.관심있는 독자는 하이브리드화 섹션 (:numref:`sec_hybridize`) 을 확인하여 현재 장을 마친 후 컴파일에 대해 배우는 것이 좋습니다.
:end_tab:

:begin_tab:`pytorch`
열렬한 독자는 이러한 작업 중 일부의 효율성에 대해 걱정하기 시작할 수 있습니다.결국 우리는 고성능 딥 러닝 라이브러리라고 생각되는 많은 사전 조회, 코드 실행 및 기타 Pythonic 작업이 많이 발생합니다.파이썬의 [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock)의 문제점은 잘 알려져 있습니다.딥 러닝의 맥락에서, 우리는 매우 빠른 GPU가 다른 작업을 실행하기 전에 작은 CPU가 파이썬 코드를 실행할 때까지 기다려야 할 수도 있습니다.
:end_tab:

:begin_tab:`tensorflow`
열렬한 독자는 이러한 작업 중 일부의 효율성에 대해 걱정하기 시작할 수 있습니다.결국 우리는 고성능 딥 러닝 라이브러리라고 생각되는 많은 사전 조회, 코드 실행 및 기타 Pythonic 작업이 많이 발생합니다.파이썬의 [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock)의 문제점은 잘 알려져 있습니다.딥 러닝의 맥락에서, 우리는 매우 빠른 GPU가 다른 작업을 실행하기 전에 작은 CPU가 파이썬 코드를 실행할 때까지 기다려야 할 수도 있습니다.파이썬의 속도를 높이는 가장 좋은 방법은 파이썬을 완전히 피하는 것입니다.
:end_tab:

## 요약

* 레이어는 블록입니다.
* 많은 레이어가 블록을 구성할 수 있습니다.
* 많은 블록이 하나의 블록으로 구성될 수 있습니다.
* 블록에는 코드가 포함될 수 있습니다.
* 블록은 매개변수 초기화 및 역전파를 포함하여 많은 하우스키핑을 처리합니다.
* 레이어와 블록의 순차적 연결은 `Sequential` 블록에 의해 처리됩니다.

## 연습문제

1. 블록을 파이썬 목록에 저장하기 위해 `MySequential`를 변경하면 어떤 종류의 문제가 발생합니까?
1. 두 블록을 인수로 사용하는 블록을 구현합니다 (예: `net1` 및 `net2`). 순방향 전파에서 두 네트워크의 연결된 출력을 반환합니다.이를 병렬 블록이라고도 합니다.
1. 동일한 네트워크의 여러 인스턴스를 연결한다고 가정합니다.동일한 블록의 여러 인스턴스를 생성하는 팩토리 함수를 구현하고 이로부터 더 큰 네트워크를 구축합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/54)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/55)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/264)
:end_tab:
