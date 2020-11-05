# Layers and Blocks
# 층과 블럭
:label:`sec_model_construction`
:label:`sec_model_construction`

0.15.0

When we first introduced neural networks,
we focused on linear models with a single output.
Here, the entire model consists of just a single neuron.
Note that a single neuron
(i) takes some set of inputs;
(ii) generates a corresponding scalar output;
and (iii) has a set of associated parameters that can be updated
to optimize some objective function of interest.
Then, once we started thinking about networks with multiple outputs,
we leveraged vectorized arithmetic
to characterize an entire layer of neurons.
Just like individual neurons,
layers (i) take a set of inputs,
(ii) generate corresponding outputs,
and (iii) are described by a set of tunable parameters.
When we worked through softmax regression,
a single layer was itself the model.
However, even when we subsequently
introduced MLPs,
we could still think of the model as
retaining this same basic structure.

뉴럴 네트워크를 처음 소개할 때, 하나의 출력을 갖는 선형 모델들에 집중했습니다. 즉, 전체 모델은 단 한개의 뉴런으로 구성됩니다. 단일 뉴런은 (i) 입력들을 취해서, (ii) 관련된 스칼라 출력을 생성하며, (iii) 관심있는 어떤 목적 함수를 최적화를 위해서 업데이트되는 관련 파라미터들로 갖는다는 점을 기억하세요. 그리고, 여러 출력을 갖는 네트워크를 알아보기 시작했습니다. 이 때 우리는 뉴런의 전체 층을 표현하기 위해서 벡터 연산을 활용했습니다. 단일 뉴런과 마찬가지로, 층들을 (i) 입력들의 집합을 취하고, (ii) 관련된 출력들을 생성하고, (iii) 튜닝이 가능한 파라미터들의 집합으로 설명됩니다. 소프트맥스 회귀의 경우 단일 층이 모델 그 자체였습니다. 하지만, 이 후에 MLP를 소개하면서도 우리는 여전히 같은 기본 구조를 유지하는 모델을 생각할 수 있었습니다.

Interestingly, for MLPs,
both the entire model and its constituent layers
share this structure.
The entire model takes in raw inputs (the features),
generates outputs (the predictions),
and possesses parameters
(the combined parameters from all constituent layers).
Likewise, each individual layer ingests inputs
(supplied by the previous layer)
generates outputs (the inputs to the subsequent layer),
and possesses a set of tunable parameters that are updated
according to the signal that flows backwards
from the subsequent layer.

흥미롭게도 MLP의 경우 전체 모델과 이를 구성하는 층들 모두 이 구조를 공유합니다. 전체 모델은 입력(특성들)을 받아서, 출력들(예측들)을 생성하고, 파라미터들(모든 구성 층들의 파라미터들 전체)를 갖습니다. 마찬가지로 각 개별 층은 (이전 층이 제공하는) 입력을 받아서 출력을 생성(다음 층의 입력들이 됨)하고, 다음 층으로 부터 역으로 전달되오는 시그널에 따라서 업데이트되는 튜닝 가능한 파라미터들을 갖습니다.

While you might think that neurons, layers, and models
give us enough abstractions to go about our business,
it turns out that we often find it convenient
to speak about components that are
larger than an individual layer
but smaller than the entire model.
For example, the ResNet-152 architecture,
which is wildly popular in computer vision,
possesses hundreds of layers.
These layers consist of repeating patterns of *groups of layers*. Implementing such a network one layer at a time can grow tedious.
This concern is not just hypothetical---such
design patterns are common in practice.
The ResNet architecture mentioned above
won the 2015 ImageNet and COCO computer vision competitions
for both recognition and detection :cite:`He.Zhang.Ren.ea.2016`
and remains a go-to architecture for many vision tasks.
Similar architectures in which layers are arranged
in various repeating patterns
are now ubiquitous in other domains,
including natural language processing and speech.

여러분은 뉴런, 층, 그리고 모델이 우리의 문제들을 해결하기에 충분한 추상화를 제공하고 있다고 생각할지도 모르지만, 각 개별 층보다는 크지만 전체 모델보다는 작은 컴포넌트가 있다면 편리하다는 것을 알게될 것입니다. 예를 들어, 컴퓨터 비전에서 굉장히 유명한 ResNet-152 아키텍처는 수백 개의 층을 가지고 있습니다. 이 층들은 *층의 그룹들*이 반복되는 패턴으로 구성되어 있습니다. 이런 네트워크를 각 층을 일일이 구현하는 것은 지겨운 일입니다. 이러한 우려는 단지 가상의 것이 아니고, 이런 디자인 패턴은 실무에서 흔히 보입니다. 앞에서 언급한 ResNet 아키텍처는 2015년 ImageNet와 COCO 컴퓨터 비전 대회에서 이미지 인식 및 객체 탐지 부문 모두에서 우승한 모델이고 :cite:`He.Zhang.Ren.ea.2016`, 많은 비전 과제에서 많이 사용되는 아키텍처로 남아있습니다. 층들이 다양하게 반복되는 패턴으로 구성되는 비슷한 아키텍처가 지연어처리 및 음성을 포함한 다른 분야에서도 요즘에는 흔합니다.

To implement these complex networks,
we introduce the concept of a neural network *block*.
A block could describe a single layer,
a component consisting of multiple layers,
or the entire model itself!
One benefit of working with the block abstraction
is that they can be combined into larger artifacts,
often recursively. This is illustrated in :numref:`fig_blocks`. By defining code to generate blocks
of arbitrary complexity on demand,
we can write surprisingly compact code
and still implement complex neural networks.

이런 복잡한 네트워크를 구현하기 위해서 뉴럴 네트워크 *블록* 이라는 개념을 소개하겠습니다. 블록은 단일 층일 수도 있고, 여러 층으로 이뤄진 컴포넌트가 될 수도 또는 전체 모델 그 자체가 될 수도 있습니다.  블록 추상화를 사용했을 때 얻는 장점은 종종 재귀적으로 합쳐져서 더 큰 아피팩트를 구성할 수 있다는 것입니다. 이는 :numref:`fig_blocks` 에서 설명되어 있습니다. 요청에 따라 임의의 복잡도를 갖는 블럭을 생성하는 코드를 정의해서 우리는 놀라울 정도로 간결한 코드를 작성해서 복잡한 신경망을 구현할 수 있습니다.

![Multiple layers are combined into blocks, forming repeating patterns of larger models.](../img/blocks.svg)
![더 큰 모델의 반복되는 패턴을 구성하기 위해서, 여러 층이 블록들로 합쳐진다.](../img/blocks.svg)
:label:`fig_blocks`

From a programing standpoint, a block is represented by a *class*.
Any subclass of it must define a forward propagation function
that transforms its input into output
and must store any necessary parameters.
Note that some blocks do not require any parameters at all.
Finally a block must possess a backpropagation function,
for purposes of calculating gradients.
Fortunately, due to some behind-the-scenes magic
supplied by the auto differentiation
(introduced in :numref:`sec_autograd`)
when defining our own block,
we only need to worry about parameters
and the forward propagation function.

프로그래밍 관점에서 보면, 블록은 *클래스(class)*로 표현됩니다. 그것의 서브클래스는 입력을 출력으로 변환하는 정방향 전파 함수(forward propagation)를 구현하고 필요한 파라미터들을 저장해야만 합니다. 어떤 블록들은 파라미터가 전혀 필요없다는 것도 알아두세요. 마지막으로 블록은 미분 계산을 위해서 역전파 함수를 가져야합니다. 운이 좋게도 자동 미분이 제공하는 눈에 모이지 않는 마술 덕분에 (:numref:`sec_autograd`에서 소개된), 우리만의 블록을 정의할 때 파라미터들과 정방향 전파 함수만 걱정하면 됩니다.

To begin, we revisit the code
that we used to implement MLPs
(:numref:`sec_mlp_concise`).
The following code generates a network
with one fully-connected hidden layer
with 256 units and ReLU activation,
followed by a fully-connected output layer
with 10 units (no activation function).

시작하기 위해서 우리가 MLP를 구현하는데 사용한 코드(:numref:`sec_mlp_concise`)를 다시 보겠습니다. 아래 코드는 256 유닛과 ReLU 활성화를 갖는 한 개의 완전 연결 은닉층으로 구성된 네트워크를 생성합니다.

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
In this example, we constructed
our model by instantiating an `nn.Sequential`,
assigning the returned object to the `net` variable.
Next, we repeatedly call its `add` function,
appending layers in the order
that they should be executed.
In short, `nn.Sequential` defines a special kind of `Block`,
the class that presents a block in Gluon.
It maintains an ordered list of constituent `Block`s.
The `add` function simply facilitates
the addition of each successive `Block` to the list.
Note that each layer is an instance of the `Dense` class
which is itself a subclass of `Block`.
The forward propagation (`forward`) function is also remarkably simple:
it chains each `Block` in the list together,
passing the output of each as the input to the next.
Note that until now, we have been invoking our models
via the construction `net(X)` to obtain their outputs.
This is actually just shorthand for `net.forward(X)`,
a slick Python trick achieved via
the `Block` class's `__call__` function.

이 예제에서 `nn.Sequential`의 인스턴스를 만들어서 `net` 변수에 할당해서 우리의 모델을 생성했습니다. 그리고는 이 객체의 `add` 함수를 반복해서 호출하면서 실행될 순서대로 층들을 추가합니다. 간단하게 말하면, `nn.Sequential`는 Gluon에서 블록을 표한하는 클래스인 `Block`의 특별한 종류를 정의합니다. 이는 `Block`들의 순서가 있는 목록을 관리합니다.  `add` 함수는 단순히 연속되는 `Block` 을 목록에 추가하는 역할을 합니다. 각 층은  `Block`의 서브클래스인  `Dense` 클래스의 객체임을 주의하세요. 정방향 전파 함수(`forward`) 역시 굉장히 간단합니다: 목록에 있는  `Block`를 모두 순서대로 엵습니다. 각 블록의 출력은 다음 블록의 입력으로 전달합니다. 지금까지 출력을 얻기 위해서 생성자 `net(X)`를 통해서 모델을 호출했다는 것을 주의하세요. 이것은 사실 `net.forward(X)`에 대한 간략한 형태인데, `Block` 클래스의  `__call__` 함수를 통한 Python의 유용한 기능을 통해서 제공되는 것입니다.
:end_tab:

:begin_tab:`pytorch`
In this example, we constructed
our model by instantiating an `nn.Sequential`, with layers in the order
that they should be executed passed as arguments.
In short, `nn.Sequential` defines a special kind of `Module`,
the class that presents a block in PyTorch.
It maintains an ordered list of constituent `Module`s.
Note that each of the two fully-connected layers is an instance of the `Linear` class
which is itself a subclass of `Module`.
The forward propagation (`forward`) function is also remarkably simple:
it chains each block in the list together,
passing the output of each as the input to the next.
Note that until now, we have been invoking our models
via the construction `net(X)` to obtain their outputs.
This is actually just shorthand for `net.forward(X)`,
a slick Python trick achieved via
the Block class's `__call__` function.

이 예제에서 `nn.Sequential`의 인스턴스를 만들어서 `net` 변수에 할당해서 우리의 모델을 생성했습니다. 그리고는 이 객체의 `add` 함수를 반복해서 호출하면서 실행될 순서대로 층들을 추가합니다. 간단하게 말하면, `nn.Sequential`는 PyTorch에서 블록을 표한하는 클래스인 `Module`의 특별한 종류를 정의합니다. 이는 `Module`들의 순서가 있는 목록을 관리합니다.  `add` 함수는 단순히 연속되는 `Module` 을 목록에 추가하는 역할을 합니다. 각 층은  `Module`의 서브클래스인  `Linear` 클래스의 객체임을 주의하세요. 정방향 전파 함수(`forward`) 역시 굉장히 간단합니다: 목록에 있는  `Module`를 모두 순서대로 엵습니다. 각 블록의 출력은 다음 블록의 입력으로 전달합니다. 지금까지 출력을 얻기 위해서 생성자 `net(X)`를 통해서 모델을 호출했다는 것을 주의하세요. 이것은 사실 `net.forward(X)`에 대한 간략한 형태인데, 블록 클래스의  `__call__` 함수를 통한 Python의 유용한 기능을 통해서 제공되는 것입니다.
:end_tab:

:begin_tab:`tensorflow`
In this example, we constructed
our model by instantiating an `keras.models.Sequential`, with layers in the order
that they should be executed passed as arguments.
In short, `Sequential` defines a special kind of `keras.Model`,
the class that presents a block in Keras.
It maintains an ordered list of constituent `Model`s.
Note that each of the two fully-connected layers is an instance of the `Dense` class
which is itself a subclass of `Model`.
The forward propagation (`call`) function is also remarkably simple:
it chains each block in the list together,
passing the output of each as the input to the next.
Note that until now, we have been invoking our models
via the construction `net(X)` to obtain their outputs.
This is actually just shorthand for `net.call(X)`,
a slick Python trick achieved via
the Block class's `__call__` function.

이 예제에서 `keras.models.Sequential`의 인스턴스를 만들어서 `net` 변수에 할당해서 우리의 모델을 생성했습니다. 이 객체를 생성할 때, 실행될 순서로 구성된 층들이 인자로 전달됩니다. 간단하게 말하면, `Sequential`는 Keras에서 블록을 표한하는 클래스인 `keras.Model`의 특별한 종류를 정의합니다. 이는 `Model`들의 순서가 있는 목록을 관리합니다.  각 층은  `Model`의 서브클래스인  `Dense` 클래스의 객체임을 주의하세요. 정방향 전파 함수(`call`) 역시 굉장히 간단합니다: 목록에 있는  `Model`를 모두 순서대로 엵습니다. 각 블록의 출력은 다음 블록의 입력으로 전달합니다. 지금까지 출력을 얻기 위해서 생성자 `net(X)`를 통해서 모델을 호출했다는 것을 주의하세요. 이것은 사실 `net.call(X)`에 대한 간략한 형태인데, 블록 클래스의  `__call__` 함수를 통한 Python의 유용한 기능을 통해서 제공되는 것입니다.
:end_tab:

## A Custom Block
## 커스텀 블록

Perhaps the easiest way to develop intuition
about how a block works
is to implement one ourselves.
Before we implement our own custom block,
we briefly summarize the basic functionality
that each block must provide:

아마도 블록이 어떻게 작동하는지를 아는 가장 쉬운 방법은 직접 하나 만들어보는 것입니다. 커스텀 블록을 구현하기에 앞서, 블곡이 제공해야하는 기본적인 기능을을 간략하게 요약하면 다음과 같습니다.

1. Ingest input data as arguments to its forward propagation function.
1. Generate an output by having the forward propagation function return a value. Note that the output may have a different shape from the input. For example, the first fully-connected layer in our model above ingests an      input of arbitrary dimension but returns an output of dimension 256.
1. Calculate the gradient of its output with respect to its input, which can be accessed via its backpropagation function. Typically this happens automatically.
1. Store and provide access to those parameters necessary
   to execute the forward propagation computation.
1. Initialize model parameters as needed.

1. 입력 데이터를 정방향 전파 함수의 인자로 전달합니다.
1. 정방향 전차 함수가 값을 반환하게 해서 출력을 생성합니다. 출력은 입력과 모양이 다를 수 있음을 기억하세요. 예를 들어, 우리의 첫번째 완전 연결층은 임의의 차원의 입력을 받아서 256 차원의 출력을 반환합니다.
1. 입력에 대해서 출력의 미분을 계산하고, 이는 역전파 함수를 통해서 접근할 수 있습니다. 보통 이는 자동으로 일어납니다.
1. 파라미터를 저장하고, 정방향 전파 계산 수행에 사용될 수 있도록 파라미터에 접근 방법을 제공합니다.
1. 필요한 경우 모델 파라미터들을 초기화합니다.

In the following snippet,
we code up a block from scratch
corresponding to an MLP
with one hidden layer with 256 hidden units,
and a 10-dimensional output layer.
Note that the `MLP` class below inherits the class that represents a block.
We will heavily rely on the parent class's functions,
supplying only our own constructor (the `__init__` function in Python) and the forward propagation function.

다음 코드는 256 은닉 뉴닛 및 10-차원 출력층을 갖는 MLP를 위한 블록을 처음부터 구현한 것입니다. `MLP` 클래스는 블록을 표현하는 클래스를 상속하고 있습니다. 우리는 상위 클래스의 함수들에 많이 의존하면서, 우리의 생성자  ( Python에서 `__init__` 함수) 및 정방향 전파 함수만 제공합니다.

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
        # Call the constructor of the `MLP` parent class `Block` to perform
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
        # Call the constructor of the `MLP` parent class `Block` to perform
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

Let us first focus on the forward propagation function.
Note that it takes `X` as the input,
calculates the hidden representation
with the activation function applied,
and outputs its logits.
In this `MLP` implementation,
both layers are instance variables.
To see why this is reasonable, imagine
instantiating two MLPs, `net1` and `net2`,
and training them on different data.
Naturally, we would expect them
to represent two different learned models.

우선 정방향 전파 함수를 살표보겠습니다. `X`를 입력을 받아서, 활성화 함수를 적용한 은닉층을 계산한 후, 로짓(logit)들을 출력합니다. 이  `MLP` 구현에서 층들은 모두 인스턴스 변수입니다. 이것이 왜 합리적인지 알아보기 위해서 두 개의 MLP  `net1` 와 `net2`를 생성하고, 서로 다른 데이터를 사용해서 두 모델을 학습하는 경우를 생각해 보겠습니다. 자연스럽게도 우리는 두 MLP가 서로 다른 모델을 학습하기를 기대합니다.

We instantiate the MLP's layers
in the constructor
and subsequently invoke these layers
on each call to the forward propagation function.
Note a few key details.
First, our customized `__init__` function
invokes the parent class's `__init__` function
via `super().__init__()`
sparing us the pain of restating
boilerplate code applicable to most blocks.
We then instantiate our two fully-connected layers,
assigning them to `self.hidden` and `self.out`.
Note that unless we implement a new operator,
we need not worry about the backpropagation function
or parameter initialization.
The system will generate these functions automatically.
Let us try this out.

우리는 MLP 층들을 생성자에서 인스터스화 한 후, 정방향 전파 함수가 불리면 이 층들을 호출합니다. 몇 가지 중요한 세부 사항들을 알아보겠습니다. 첫 번째, 우리가 정의한  `__init__` 함수는 상위 클래스의  `__init__`  함수를 `super().__init__()`를 통해서 호출합니다. 이는 대부분 블록들에 적용되는 기본 코드들을 직접 작성하는 수고를 덜어줍니다. 그 다음 우리는 두 개의 완전 연결 층들을 초기화하고, 이 것들을  `self.hidden` 와 `self.out`에 각각 저장합니다. 새로운 연산을 정의하지 않는 한, 역전파 함수나 파라미터 초기화를 걱정하지 않아도 됩니다. 시스템이 이 함수들을 자동으로 생성해줍니다. 그럼 사용해보겠습니다. 

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

A key virtue of the block abstraction is its versatility.
We can subclass a block to create layers
(such as the fully-connected layer class),
entire models (such as the `MLP` class above),
or various components of intermediate complexity.
We exploit this versatility
throughout the following chapters,
such as when addressing
convolutional neural networks.

블록 추상화의 주요 이점은 다양성입니다. 우리는 블록의 서브클래스로 (완전 연결 층 클래스와 같은) 층을 만들 수 있고, ( 위에서 본 `MLP` 클래스와 같은) 전체 모델을 만들 수 있고, 또는 그 중간쯤의 복잡도의 다양한 컴포넌트를 만들 수 있습니다. 우리는 다음 장들에서 이 다양성을 이용할 것입니다. 예를 들어 컨볼루셔널 신경망을 만들 때사용할 것입니다.

## The Sequential Block
## 순차 블록(Sequential Block)

We can now take a closer look
at how the `Sequential` class works.
Recall that `Sequential` was designed
to daisy-chain other blocks together.
To build our own simplified `MySequential`,
we just need to define two key function:
1. A function to append blocks one by one to a list.
2. A forward propagation function to pass an input through the chain of blocks, in the same order as they were appended.

이제 우리는 `Sequential` 클래스가 어떻게 작동하는지 자세히 살펴볼 수 있습니다. `Sequential` 이 블록들을 데이지 체인 방식(daisy-chain)으로 연결하기 위해서 설계되었다는 것을 떠올려보세요. 간단한  `MySequential`을 만들기 위해서, 우리는 다음 두 개의 주요 함수를 정의하면 됩니다.
1. 블록을 하나씩 리스트에 추가하는 함수
2. 입력을 블록의 체인에 전달하는 정방향 전파 함수. 블록이 추가된 순서에 따라서 수행.

The following `MySequential` class delivers the same
functionality of the default `Sequential` class.

다음 `MySequential`  클래스는 기본  `Sequential` 클래스와 같은 기능을 제공합니다.

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
        for block in args:
            # Here, `block` is an instance of a `Module` subclass. We save it
            # in the member variable `_modules` of the `Module` class, and its
            # type is OrderedDict
            self._modules[block] = block

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
The `add` function adds a single block
to the ordered dictionary `_children`.
You might wonder why every Gluon `Block`
possesses a `_children` attribute
and why we used it rather than just
define a Python list ourselves.
In short the chief advantage of `_children`
is that during our block's parameter initialization,
Gluon knows to look inside the `_children`
dictionary to find sub-blocks whose
parameters also need to be initialized.
`add` 함수는 하나의 블록을 순서가 있는 사전 `_children`에 추가합니다. 여러분은 왜 모든 Gluon `Block`이 한 개의  `_children` 속성을 갖고 있는지, 그리고 Python 리스트를 정의하는 대신 이 속성을 사용하는지 의문을 갖을 것입니다. 간단하게 말하면,  `_children`의 주요 장점은 블록의 파라미터 초기화 과정에서, 초기화가 필요한 파라미터를 갖는 서브 블록을 찾기 위해서 Gluon가 `_children` 사전의 내부를 보면된다는 것을 안다는 것입니다.
:end_tab:

:begin_tab:`pytorch`
In the `__init__` method, we add every block
to the ordered dictionary `_modules` one by one.
You might wonder why every `Module`
possesses a `_modules` attribute
and why we used it rather than just
define a Python list ourselves.
In short the chief advantage of `_modules`
is that during our block's parameter initialization,
the system knows to look inside the `_modules`
dictionary to find sub-blocks whose
parameters also need to be initialized.
`__init__` 메소드에서 우리는 순서가 있는 사전 `_modules`에 모든 블록을 하나씩 추가합니다. 여러분은 왜 모든  `Module`이 한 개의  `_modules` 속성을 갖고 있는지, 그리고 Python 리스트를 정의하는 대신 이 속성을 사용하는지 의문을 갖을 것입니다. 간단하게 말하면,  `_modules`의 주요 장점은 블록의 파라미터 초기화 과정에서, 초기화가 필요한 파라미터를 갖는 서브 블록을 찾기 위해서 시스템이 `_modules` 사전의 내부를 보면된다는 것을 안다는 것입니다.
:end_tab:

When our `MySequential`'s forward propagation function is invoked,
each added block is executed
in the order in which they were added.
We can now reimplement an MLP
using our `MySequential` class.

`MySequential`의 정방향 전파 함수가 호출되면, 각 블록은 추가된 순서대로 실행됩니다. 이제 우리는 `MySequential` 클래스를 사용해서 MLP를 다시 구현할 수 있습니다.

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

Note that this use of `MySequential`
is identical to the code we previously wrote
for the `Sequential` class
(as described in :numref:`sec_mlp_concise`).

`MySequential`를 사용하는 것이 앞에서  `Sequential` 클래스를 사용한 코드와 같다는 것을 기억하세요. (:numref:`sec_mlp_concise` 참고) 


## Executing Code in the Forward Propagation Function
## 정방향 전파 함수에서 코드 수행

The `Sequential` class makes model construction easy,
allowing us to assemble new architectures
without having to define our own class.
However, not all architectures are simple daisy chains.
When greater flexibility is required,
we will want to define our own blocks.
For example, we might want to execute
Python's control flow within the forward propagation function.
Moreover, we might want to perform
arbitrary mathematical operations,
not simply relying on predefined neural network layers.

`Sequential` 클래스는 우리가 직접 클래스를 정의할 필요가 없이 새로운 아키텍처를 조합할 수 있게 해줌으로 모델의 생성을 쉽게 해줍니다.하지만, 모든 아키텍처가 간단한 데이지 체인은 아닙니다. 더 큰 유연성이 필요한 경우 우리는 블록을 직접 정의할 필요가 있습니다. 예를 들어, Python의 제어문을 정방향 전파 함수에서 수행하기를 원할 수도 있습니다. 더불어 사전에 정의된 뉴럴 네트워크 층의 의존하지 않고, 임의의 수학 연산을 수행하기를 원할 수도 있습니다.

You might have noticed that until now,
all of the operations in our networks
have acted upon our network's activations
and its parameters.
Sometimes, however, we might want to
incorporate terms
that are neither the result of previous layers
nor updatable parameters.
We call these *constant parameters*.
Say for example that we want a layer
that calculates the function
$f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$,
where $\mathbf{x}$ is the input, $\mathbf{w}$ is our parameter,
and $c$ is some specified constant
that is not updated during optimization.
So we implement a `FixedHiddenMLP` class as follows.

지금까지는 네트워크의 모든 연산이 네트워크의 활성화들과 파라미터들에 의해서만 작동했음을 알아차렸을 것입니다. 하지만 때로는 이전 층의 결과나 업데이트가 가능한 파라미터들이 아닌 항목들을 사용하고 싶은 때가 있습니다. 우리는 이것을 *상수 파라미터(constant parameter)*라고 부릅니다. 예를 들어, 우리는  $f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$ 함수를 계산하는 층을 구성하고 싶습니다. 여기서 $\mathbf{x}$는 입력이고, $\mathbf{w}$는 파라미터, 그리고 최적화 과정에서 업데이트되지 않는 $c$는 어떤 특정 상수 값입니다. `FixedHiddenMLP` 클래스를 다음과 같이 구현합니다.

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

In this `FixedHiddenMLP` model,
we implement a hidden layer whose weights
(`self.rand_weight`) are initialized randomly
at instantiation and are thereafter constant.
This weight is not a model parameter
and thus it is never updated by backpropagation.
The network then passes the output of this "fixed" layer
through a fully-connected layer.

`FixedHiddenMLP` 모델에서 우리는 가중치(`self.rand_weight`) 가 초기화시 임의의 값으로 초기화되고 그 후에는 상수로 남는 은닉층을 구현합니다. 이 가중치는 모델 파라미터가 아니고, 따라서 역전파에 의해서 절대로 업데이트되지 않습니다. 네트워크는 이 "고정된" 층의 결과를 완전 연결 층에 전달합니다.

Note that before returning the output,
our model did something unusual.
We ran a while-loop, testing
on the condition its $L_1$ norm is larger than $1$,
and dividing our output vector by $2$
until it satisfied the condition.
Finally, we returned the sum of the entries in `X`.
To our knowledge, no standard neural network
performs this operation.
Note that this particular operation may not be useful
in any real-world task.
Our point is only to show you how to integrate
arbitrary code into the flow of your
neural network computations.

결과를 반환하기 전에 우리의 모델은 일반적이지 않은 것을 하는 것을 주의하세요. 우리는 while-룹을 돌리면서 $L_1$ 놈이 $1$보다 큰 값인지를 테스트하고, 그 조건을 만족하면 결과 벡터를 $2$로 나눕니다. 마지막으로, `X`의 항목들의 함을 반환합니다. 우리가 알기로는 표준 신경망은 이런 연산을 수행하지 않습니다. 이 특별한 연산은 실제 상황에 유용하지 않을 수 있습니다. 이 예제는 임의의 코드를 신경망 연산의 흐름에 임의의 코드를 어떻게 넣을 수 있는지를 보여주기 위함입니다.

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

We can mix and match various
ways of assembling blocks together.
In the following example, we nest blocks
in some creative ways.

우리는 블록을 여러 방법으로 조합할 수 있습니다. 아래 예제는 블록을 다소 창의적인 방법으로 포함시키는 방법을 보여주고 있습니다.

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

## Efficiency

:begin_tab:`mxnet`
The avid reader might start to worry
about the efficiency of some of these operations.
After all, we have lots of dictionary lookups,
code execution, and lots of other Pythonic things
taking place in what is supposed to be
a high-performance deep learning library.
The problems of Python's [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock) are well known. 
In the context of deep learning,
we may worry that our extremely fast GPU(s)
might have to wait until a puny CPU
runs Python code before it gets another job to run.
The best way to speed up Python is by avoiding it altogether.

호기심이 많은 독자라면 연산들의 일부에 대한 효율설을 우려하기 시작할 수도 있습니다. 결국에는, 빠른 성능의 딥러닝 라이브러리가 되야 할 곳에, 많은 사전 조회, 코드 수행, 그리고 다른 Python 스타일의 것들을 갖을 것입니다. Python의  [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock) 문제는 아주 잘 알려져 있습니다. 딥러닝의 환경에서, 우리는 극도로 빠른 GPU(또는 GPU들)이 수행해야할 다른 작업을 얻기에 느린 CPU가 Python 코드를 수행하는 것을 기다려야하는 것에 대해서 우려할 수 있습니다.

One way that Gluon does this is by allowing for
*hybridization*, which will be described later.
Here, the Python interpreter executes a block
the first time it is invoked.
The Gluon runtime records what is happening
and the next time around it short-circuits calls to Python.
This can accelerate things considerably in some cases
but care needs to be taken when control flow (as above)
leads down different branches on different passes through the net.
We recommend that the interested reader checks out
the hybridization section (:numref:`sec_hybridize`)
to learn about compilation after finishing the current chapter.

Gluon이 이를 해결하는 한 가지 방법은 다음에 설명할 *하이브리드화(hybridization)*를 허용하는 것입니다. 처음으로 호출되는 블록은 Python 인퍼프린터가 수행합니다. Gluon 런터임은 무엇이 일어나고 있는지 기록하고, 다음 번에는 Python을 단락시킵니다. 이는 어떤 경우에 대해서는 상단한 가속을 가능하게 합니다. 하지만, (위의 예제와 같은) 흐름 제어가 네트워크 수행할 때 다른 브랜치를 수행하는 경우 주의가 필요합니다. 더 관심이 있다면 이 장일 마친 후 컴파일에 대해서 더 배우기 위해서 하이브리드화 절(:numref:`sec_hybridize`)을 읽는 것을 권장합니다.
:end_tab:

:begin_tab:`pytorch`
The avid reader might start to worry
about the efficiency of some of these operations.
After all, we have lots of dictionary lookups,
code execution, and lots of other Pythonic things
taking place in what is supposed to be
a high-performance deep learning library.
The problems of Python's [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock) are well known. 
In the context of deep learning,
we may worry that our extremely fast GPU(s)
might have to wait until a puny CPU
runs Python code before it gets another job to run.
호기심이 많은 독자라면 연산들의 일부에 대한 효율설을 우려하기 시작할 수도 있습니다. 결국에는, 빠른 성능의 딥러닝 라이브러리가 되야 할 곳에, 많은 사전 조회, 코드 수행, 그리고 다른 Python 스타일의 것들을 갖을 것입니다. Python의  [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock) 문제는 아주 잘 알려져 있습니다. 딥러닝의 환경에서, 우리는 극도로 빠른 GPU(또는 GPU들)이 수행해야할 다른 작업을 얻기에 느린 CPU가 Python 코드를 수행하는 것을 기다려야하는 것에 대해서 우려할 수 있습니다.
:end_tab:

:begin_tab:`tensorflow`
The avid reader might start to worry
about the efficiency of some of these operations.
After all, we have lots of dictionary lookups,
code execution, and lots of other Pythonic things
taking place in what is supposed to be
a high-performance deep learning library.
The problems of Python's [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock) are well known. 
In the context of deep learning,
we may worry that our extremely fast GPU(s)
might have to wait until a puny CPU
runs Python code before it gets another job to run.
The best way to speed up Python is by avoiding it altogether.
호기심이 많은 독자라면 연산들의 일부에 대한 효율설을 우려하기 시작할 수도 있습니다. 결국에는, 빠른 성능의 딥러닝 라이브러리가 되야 할 곳에, 많은 사전 조회, 코드 수행, 그리고 다른 Python 스타일의 것들을 갖을 것입니다. Python의  [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock) 문제는 아주 잘 알려져 있습니다. 딥러닝의 환경에서, 우리는 극도로 빠른 GPU(또는 GPU들)이 수행해야할 다른 작업을 얻기에 느린 CPU가 Python 코드를 수행하는 것을 기다려야하는 것에 대해서 우려할 수 있습니다. Python를 빠르게 하는 가장 좋은 방법은 이것을 모두 피하는 것입니다.
:end_tab:

## Summary
## 요약

* Layers are blocks.
* Many layers can comprise a block.
* Many blocks can comprise a block.
* A block can contain code.
* Blocks take care of lots of housekeeping, including parameter initialization and backpropagation.
* Sequential concatenations of layers and blocks are handled by the `Sequential` block.

* 층은 블록입니다.
* 많은 층들이 블록을 구성합니다.
* 많은 블록들이 블록을 구성합니다.
* 블록은 코드를 가질 수 있습니다.
* 블록은 파라미터 초기화와 역전파화 같은 일상적인 것들을 많이 제공합니다.
* 층과 블록의 순차적인 연결은 `Sequential` 블록으로 다뤄집니다.

## Exercises

1. What kinds of problems will occur if you change `MySequential` to store blocks in a Python list?
1. Implement a block that takes two blocks as an argument, say `net1` and `net2` and returns the concatenated output of both networks in the forward propagation. This is also called a parallel block.
1. Assume that you want to concatenate multiple instances of the same network. Implement a factory function that generates multiple instances of the same block and build a larger network from it.

1. 블록을 Python 리스트에 저장하기 위해서 `MySequential` 를 바꾸면 어떤 문제들이 일어날까요?
1. 두 개 블록, `net1` 과 `net2`을 인자로 받고, 정방향 전파에서 두 네트워크의 결과를 나란히 연결한 값을 반환하는 블록을 구현하세요. 이는 병령 블록이라고 합니다.
1. 같은 네트워크의 여러 인스턴스를 나란히 연결하고 싶습니다. 같은 블록의 인스턴스를 여러개 만들어서 더 큰 네트워크를 만드는 팩토리 함수를 구현하세요.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/54)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/55)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/264)
:end_tab:
