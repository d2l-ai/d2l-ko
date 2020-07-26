# Concise Implementation of Linear Regression
# 선형 회귀의 간결한 구현
:label:`sec_linear_concise`
Broad and intense interest in deep learning for the past several years
has inspired companies, academics, and hobbyists
to develop a variety of mature open source frameworks
for automating the repetitive work of implementing
gradient-based learning algorithms.
In :numref:`sec_linear_scratch`, we relied only on
(i) tensors for data storage and linear algebra;
and (ii) auto differentiation for calculating gradients.
In practice, because data iterators, loss functions, optimizers,
and neural network layers
are so common, modern libraries implement these components for us as well.

지난 몇 년간 딥러닝에 대한 폭 넓고 깊은 관심은 기업, 학계 및 애호가들이 그래디언트 기반의 학습 알고리즘을 구현 할 때의 반복적인 일을 자동화하기 위한 다양한 성숙한 오픈 소스 프래임워크를 개발하게 했습니다. :numref:`sec_linear_scratch`에서 우리는 (i) 데이터 저장을 위한 텐서와 선형 대수, 그리고 (ii) 경사값 계산을 위한 자동 미분에만 의존했습니다. 실제로는 데이터 반복자, 손실 함수, 최적화, 그리고 뉴럴 네트워크의 층들이 너무 일반적이기 때문에, 최근 라이브러리들은 이런 컴포넌트들이 모두 구현되어 있습니다.

In this section, we will show you how to implement
the linear regression model from :numref:`sec_linear_scratch`
concisely by using high-level APIs of deep learning frameworks.

이 절에서 우리는 :numref:`sec_linear_scratch`에서 구현했던 선형 회귀 모델을 딥러닝 프레임워크들의 상위 레벨 API를 사용해서 간결하게 구현하는 것을 살펴보겠습니다.

## Generating the Dataset
## 데이터셋 만들기

To start, we will generate the same dataset as in :numref:`sec_linear_scratch`.

시작을 위해서 :numref:`sec_linear_scratch` 에서와 같은 데이터셋을 만듭니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()

true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch.utils import data

true_w = torch.Tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
labels = labels.reshape(-1,1)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

## Reading the Dataset
## 데이터셋 읽기

Rather than rolling our own iterator,
we can call upon the existing API in a framework to read data.
We pass in `features` and `labels` as arguments and specify `batch_size`
when instantiating a data iterator object.
Besides, the boolean value `is_train`
indicates whether or not
we want the data iterator object to shuffle the data
on each epoch (pass through the dataset).

반복자를 직접 구현하는 대신에, 우리는 프래임워크에서 제공하는 API를 사용해서 데이터를 읽을 수 있습니다. 데이터 반복자 객체를 초기화 할 때,  `features` 와 `labels` 를 함수의 변수로 전달하고, `batch_size`를 정합니다. `is_train`에 대한 이진 값은 각 에폭(데이터 셋 전체를 사용하는) 마다 데이터 순서를 임의로 섞기를 원하는지 아닌지를 의미합니다.

```{.python .input}
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a Gluon data iterator."""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

```{.python .input}
#@tab pytorch
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

```{.python .input}
#@tab tensorflow
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a TensorFlow data iterator."""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset
```

```{.python .input}
#@tab all
batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

Now we can use `data_iter` in much the same way as we called
the `data_iter` function in :numref:`sec_linear_scratch`.
To verify that it is working, we can read and print
the first minibatch of examples.
Comparing with :numref:`sec_linear_scratch`,
here we use `iter` to construct a Python iterator and use `next` to obtain the first item from the iterator.

우리는 이제 :numref:`sec_linear_scratch`에서 `data_iter` 를 호출한 것과 같은 방법으로 `data_iter` 사용할 수 있다. 이것이 잘 동작하는지 확인하기 위해서, 예제들의 첫 번째 미니배치를 읽어서 출력해봅니다. :numref:`sec_linear_scratch`와 비교해보면, 여기서는 Python 반복자를 만들기 위해서 `iter`을 사용하고, 반복자에서 첫 번째 아이템을 얻기 위해서 `next`를 사용합니다.

```{.python .input}
#@tab all
next(iter(data_iter))
```

## Defining the Model
## 모델 정의하기

When we implemented linear regression from scratch
in :numref:`sec_linear_scratch`,
we defined our model parameters explicitly
and coded up the calculations to produce output
using basic linear algebra operations.
You *should* know how to do this.
But once your models get more complex,
and once you have to do this nearly every day,
you will be glad for the assistance.
The situation is similar to coding up your own blog from scratch.
Doing it once or twice is rewarding and instructive,
but you would be a lousy web developer
if every time you needed a blog you spent a month
reinventing the wheel.

:numref:`sec_linear_scratch`에서 선형 회귀를 직접 구현할 때, 모델 파라미터들을 명시적으로 정의하고, 기본적인 선형 대수 연산을 이용해서 결과를 생성하는 코드를 직접 구현했다. 이 것을 어떻게 하는지 *꼭 알야아합니다*. 하지만, 모델이 더 복잡해지고, 이 일을 거의 매일 해야한다면, 도움에 기뻐할 것입니다. 이는 마치 블로그를 직접 코딩하는 것과 비슷한 상황입니다. 한번 또는 두번 정도 직접 만드는 것은 보람되고 유익할 것입니다. 하지만, 블로그가 필요할 때마다 모든 것을 새로 구현하면서 한 달을 보낸다면, 여러분은 끔찍한 웹 개발자가 될 것입니다.

For standard operations, we can use a framework's predefined layers,
which allow us to focus especially
on the layers used to construct the model
rather than having to focus on the implementation.
We will first define a model variable `net`,
which will refer to an instance of the `Sequential` class.
The `Sequential` class defines a container
for several layers that will be chained together.
Given input data, a `Sequential` instance passes it through
the first layer, in turn passing the output
as the second layer's input and so forth.
In the following example, our model consists of only one layer,
so we do not really need `Sequential`.
But since nearly all of our future models
will involve multiple layers,
we will use it anyway just to familiarize you
with the most standard workflow.

표준 연산들에 대해서는 프레임워크에서 제공하는 층을 사용할 수 있습니다. 이는 구현에 집중하는 것하기 보다는 모델을 구성하는데 사용될 층들에 특별히 집중할 수 있게 해줍니다. 우선  `Sequential` 클래스의 인스턴스인 모델 변수 `net`을 정의합니다.  `Sequential` 클래스는 함께 이어질 될 몇 개의 층들을 위한 컨테이너를 정의합니다. 입력이 주어지면,  `Sequential` 인스턴스는 이 값을 첫 번째 층으로 전달하고, 그 결과를 다시 두번 째층의 입력으로 전달하는 것을 반복합니다. 아래 예제에서 우리의 모델은 단 한 개의 층만 가지고 있기에,  `Sequential`이 실제로는 필요하지 않습니다. 하지만, 앞으로 살펴볼 모든 모델들을 여러 층을 가지고 있기 때문에, 가장 표준 워크플로우에 친숙해지기 위해서 이를 사용할 것입니다.  

Recall the architecture of a single-layer network as shown in :numref:`fig_single_neuron``.
The layer is said to be *fully-connected*
because each of its inputs is connected to each of its outputs
by means of a matrix-vector multiplication.

:numref:`fig_single_neuron`에서 소개한 단 층 네트워크의 아키텍처를 기억해 보겠습니다. 이 층은 입력의 각각이 출력의 각각과 행렬-벡터의 곱 형태로 연결되기 때문에 *완전 연결(fully-connected)*라고 합니다.

:begin_tab:`mxnet`
In Gluon, the fully-connected layer is defined in the `Dense` class.
Since we only want to generate a single scalar output,
we set that number to 1.

Gluon에서 완전 연결 층은 `Dense` 클래스로 정의됩니다. 우리는 하나의 스칼라 출력을 만들기를 원하기 때문에, 1로 설정합니다.

It is worth noting that, for convenience,
Gluon does not require us to specify
the input shape for each layer.
So here, we do not need to tell Gluon
how many inputs go into this linear layer.
When we first try to pass data through our model,
e.g., when we execute `net(X)` later,
Gluon will automatically infer the number of inputs to each layer.
We will describe how this works in more detail later.

Gluon에서는 편의를 위해서 각 층의 입력 모양을 정의하지 않아도 됩니다. 그렇게에 여기서 우리는 이 선형 층에 몇 개의 입력이 전달되는지 Gluon에게 알려줄 필요가 없습니다. 모델에 처음으로 데이터가 입력될 때, 즉, 이 후에  `net(X)`가 수행될 때, Gluon은 각 층의 입력의 개수를 자동으로 추정합니다. 이 것이 어떻게 동작하는지는 나중에 자세히 다루겠습니다.
:end_tab:

:begin_tab:`pytorch`
In PyTorch, the fully-connected layer is defined in the `Linear` class. Note that we passed two arguments into `nn.Linear`. The first one specifies the input feature dimension, which is 2, and the second one is the output feature dimension, which is a single scalar and therefore 1.
PyTorch에서 완전 연결 층은 `Linear` 클래스로 정의됩니다.  `nn.Linear`에 두 개의 변수를 전달해야함을 염두하세요. 첫 번째는 입력 피처의 차원이고, 여기서는 2, 두 번째는 출력 치퍼의 차원인데, 여기서는 하나의 스칼라이기 때문에, 1입니다.
:end_tab:

:begin_tab:`tensorflow`
In Keras, the fully-connected layer is defined in the `Dense` class. Since we only want to generate a single scalar output, we set that number to 1.

Keras에서 완전 연결 측은 `Dense` 클래스로 정의됩니다. 우리는 하나의 스칼라 출력을 만들기를 원하기 때문에 1로 설정합니다.

It is worth noting that, for convenience,
Keras does not require us to specify
the input shape for each layer.
So here, we do not need to tell Keras
how many inputs go into this linear layer.
When we first try to pass data through our model,
e.g., when we execute `net(X)` later,
Keras will automatically infer the number of inputs to each layer.
We will describe how this works in more detail later.

Keras에서는 편의를 위해서 각 층의 입력 모양을 정의하지 않아도 됩니다. 그렇게에 여기서 우리는 이 선형 층에 몇 개의 입력이 전달되는지 Keras에게 알려줄 필요가 없습니다. 모델에 처음으로 데이터가 입력될 때, 즉, 이 후에  `net(X)`가 수행될 때, Keras은 각 층의 입력의 개수를 자동으로 추정합니다. 이 것이 어떻게 동작하는지는 나중에 자세히 다루겠습니다.
:end_tab:

```{.python .input}
# `nn` is an abbreviation for neural networks
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))
```

```{.python .input}
#@tab pytorch
# `nn` is an abbreviation for neural networks
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
```

```{.python .input}
#@tab tensorflow
# `keras` is the high-level API for TensorFlow
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1))
```

## Initializing Model Parameters
## 모델 파라미터 초기화하기 

Before using `net`, we need to initialize the model parameters,
such as the weights and bias in the linear regression model.
Deep learning frameworks often have a predefined way to initialize the parameters.
Here we specify that each weight parameter
should be randomly sampled from a normal distribution
with mean 0 and standard deviation 0.01.
The bias parameter will be initialized to zero.

`net`을 사용하기 전에 선형 회귀 모델의 가중치와 편향과 같은 모델 파라미터들을 초기화 해야합니다. 딥러닝 프래임워크들은 보통 파라미터들을 초기화하는 사전에 정의된 방법들을 가지고 있습니다. 여기서 우리는 각 가중치 파라미터가 평균이 0이고 표준 편차가 0.01인 정규 분포로 부터 임의로 추출되어야 한다고 정의합니다. 편향 파라미터는 모두 0으로 초기화합니다.

:begin_tab:`mxnet`
We will import the `initializer` module from MXNet.
This module provides various methods for model parameter initialization.
Gluon makes `init` available as a shortcut (abbreviation)
to access the `initializer` package.
We only specify how to initialize the weight by calling `init.Normal(sigma=0.01)`.
Bias parameters are initialized to zero by default.
MXNet에서 `initializer` 모듈을 임포트합니다. 이 모듈은 모델 파라미터 초기화하는 여러 방법들을 제공합니다. Gluon은  `initializer` 패키지를 사용하기 위해서 `init`를 축약어로 제공합니다.  `init.Normal(sigma=0.01)`를 호출하는 것 만으로 가중치를 초기화합니다. 편향은 기본적으로 0으로 초기화됩니다.
:end_tab:

:begin_tab:`pytorch`
As we have specified the input and output dimensions when constructing `nn.Linear`. Now we access the parameters directly to specify there initial values. We first locate the layer by `net[0]`, which is the first layer in the network, and then use the `weight.data` and `bias.data` methods to access the parameters. Next we use the replace methods `normal_` and `fill_` to overwrite parameter values.
`nn.Linear`를 생성할 때 입력과 출력의 차원을 명시했고, 초기값을 설정하기 위해서 파라미터들을 접근할 수 있습니다. 우선 네트워크의 첫 번째 층인 `net[0]`을 통해서 첫 번째 층을 지정하고, `weight.data`와 `bias.data` 메소드를 사용해서 파라미터들을 접근합니다. 다음으로 우리는 파라미터 값들을 덮어쓰기 위해서 값을 바꾸는 메소드인 `normal_` 과  `fill_` 를 호출합니다.
:end_tab:

:begin_tab:`tensorflow`
The `initializers` module in TensorFlow provides various methods for model parameter initialization. The easiest way to specify the initialization method in Keras is when creating the layer by specifying `kernel_initializer`. Here we recreate `net` again.
TensorFlow의  `initializers` 모듈은 모델 파라미터를 초기화하는 다양한 메소드들을 제공합니다. Keras에서 초기화 메소드를 지정하는 가장 쉬운 방법은 층을 생성할 때  `kernel_initializer`를 지정하는 것입니다. 여기서 우리는  `net` 을 다시 생성합니다.
:end_tab:

```{.python .input}
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```

```{.python .input}
#@tab tensorflow
initializer = tf.initializers.RandomNormal(stddev=0.01)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))
```

:begin_tab:`mxnet`
The code above may look straightforward but you should note
that something strange is happening here.
We are initializing parameters for a network
even though Gluon does not yet know
how many dimensions the input will have!
It might be 2 as in our example or it might be 2000.
Gluon lets us get away with this because behind the scene,
the initialization is actually *deferred*.
The real initialization will take place only
when we for the first time attempt to pass data through the network.
Just be careful to remember that since the parameters
have not been initialized yet,
we cannot access or manipulate them.
위 코드가 직관적으로 보일 수도 있지만, 이상한 일이 일어나고 있다는 것을 알아야합니다. Gluon이 입력값이 얼마나 많은 차원을 갖는지 알기도 전에 우리는 네트워크의 파라미터들을 초기화하고 있습니다! 우리 예제와 같이 2가 될 수도 있고, 2000이 될 수도 있습니다. Gluon은 초기화를 *지연*시키기 때문에 이를 우리가 신경쓰지 않아도 됩니다. 실제 초기화는 네트워크에 데이터 입력을 처음 시도할 때 일어납니다. 파라미터들이 아직 초기화되지 않았기 때문에 이 값들을 접근할 수도 변경할 수도 없다는 것을 주의하세요.
:end_tab:

:begin_tab:`pytorch`

:end_tab:

:begin_tab:`tensorflow`
The code above may look straightforward but you should note
that something strange is happening here.
We are initializing parameters for a network
even though Keras does not yet know
how many dimensions the input will have!
It might be 2 as in our example or it might be 2000.
Keras lets us get away with this because behind the scenes,
the initialization is actually *deferred*.
The real initialization will take place only
when we for the first time attempt to pass data through the network.
Just be careful to remember that since the parameters
have not been initialized yet,
we cannot access or manipulate them.
위 코드가 직관적으로 보일 수도 있지만, 이상한 일이 일어나고 있다는 것을 알아야 합니다. Keras가 입력값이 얼마나 많은 차원을 갖는지 알기도 전에 우리는 네트워크의 파라미터들을 초기화하고 있습다! 우리 예제와 같이 2가 될 수도 있고, 2000이 될 수도 있습다. Keras은 초기화를 *지연*시키기 때문에 이를 우리가 신경쓰지 않아도 됩니다. 실제 초기화는 네트워크에 데이터 입력을 처음 시도할 때 일어납니다. 파라미터들이 아직 초기화되지 않았기 때문에 이 값들을 접근할 수도 변경할 수도 없다는 것을 주의하세요.
:end_tab:

## Defining the Loss Function
## 손실 함수 정의하기

:begin_tab:`mxnet`
In Gluon, the `loss` module defines various loss functions.
In this example, we will use the Gluon
implementation of squared loss (`L2Loss`).
Gluon에서  `loss` 모듈은 여러 손실 함수를 정의합니다. 이 예제에서 우리는 제곱 손실의 Gluon 구현( (`L2Loss`)을 사용합니다.
:end_tab:

:begin_tab:`pytorch`
The `MSELoss` class computes the mean squared error, also known as squared $L_2$ norm.
By default it returns the average loss over examples.
`MSELoss` 클래스는 제곱 $L_2$ 놈이라고도 알려진 평균 제곱 오류를 계산합니다. 기본 설정으로는 예제들에 대한 평균 손실을 반환합니다.
:end_tab:

:begin_tab:`tensorflow`
The `MeanSquaredError` class computes the mean squared error, also known as squared $L_2$ norm.
By default it returns the average loss over examples.
`MeanSquaredError` 클래스는 제곱 $L_2$ 놈이라고도 알려진 평균 제곱 오류를 계산합니다. 기본 설정으로는 예제들에 대한 평균 손실을 반환합니다.
:end_tab:

```{.python .input}
loss = gluon.loss.L2Loss()
```

```{.python .input}
#@tab pytorch
loss = nn.MSELoss()
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.MeanSquaredError()
```

## Defining the Optimization Algorithm
## 최적화 알고리즘 정의하기

:begin_tab:`mxnet`
Minibatch stochastic gradient descent is a standard tool
for optimizing neural networks
and thus Gluon supports it alongside a number of
variations on this algorithm through its `Trainer` class.
When we instantiate `Trainer`,
we will specify the parameters to optimize over
(obtainable from our model `net` via `net.collect_params()`),
the optimization algorithm we wish to use (`sgd`),
and a dictionary of hyperparameters
required by our optimization algorithm.
Minibatch stochastic gradient descent just requires that
we set the value `learning_rate`, which is set to 0.03 here.
미니배치 확률적 경사 하강법은 뉴럴 네트워크를 최적화하는 표준 도구이며, Gluon은  `Trainer` 클래스를 통해서 이 알고리즘의 여러 가지 변종들을 포함해서 지원합니다.  `Trainer`를 인스턴스화 할 때, 최적화를 수행할 파라미터들( `net.collect_params()`을 호출해서  `net`로 부터 얻습니다), 사용하고 싶은 최적화 알고리즘(`sgd`), 그리고 최적화 알고리즘이 사용할 하이퍼파리미터 사전을 명시합니다. 미니배치 확률적 경사 하강법은  `learning_rate` 값만 설정하면 되는데, 여기서는 0.03으로 설정합니다.
:end_tab:

:begin_tab:`pytorch`
Minibatch stochastic gradient descent is a standard tool
for optimizing neural networks
and thus PyTorch supports it alongside a number of
variations on this algorithm in the `optim` module.
When we instantiate an `SGD` instance,
we will specify the parameters to optimize over
(obtainable from our net via `net.parameters()`), with a dictionary of hyperparameters
required by our optimization algorithm.
Minibatch stochastic gradient descent just requires that
we set the value `lr`, which is set to 0.03 here.
미니배치 확률적 경사 하강법은 뉴럴 네트워크를 최적화하는 표준 도구이며, PyTorch는  `optim` 모듈에서 이 알고리즘의 여러 가지 변종들을 포함해서 지원한다. `SGD`를 인스턴스화 할 때, 최적화를 수행할 파라미터들( `net.parameters()`을 호출해서 얻습니다)와 최적화 알고리즘이 사용할 하이퍼파리미터 사전을 명시합니다. 미니배치 확률적 경사 하강법은  `learning_rate` 값만 설정하면 되는데, 여기서는 0.03으로 설정합니다.
:end_tab:

:begin_tab:`tensorflow`
Minibatch stochastic gradient descent is a standard tool
for optimizing neural networks
and thus Keras supports it alongside a number of
variations on this algorithm in the `optimizers` module.
Minibatch stochastic gradient descent just requires that
we set the value `learning_rate`, which is set to 0.03 here.
미니배치 확률적 경사 하강법은 뉴럴 네트워크를 최적화하는 표준 도구이며, Keras는  `optimizers` 모듈에서 이 알고리즘의 여러 가지 변종들을 포함해서 지원합니다. 미니배치 확률적 경사 하강법은  `learning_rate` 값만 설정하면 되는데, 여기서는 0.03으로 설정합니다.
:end_tab:

```{.python .input}
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=0.03)
```

## Training
## 학습

You might have noticed that expressing our model through
high-level APIs of a deep learning framework
requires comparatively few lines of code.
We did not have to individually allocate parameters,
define our loss function, or implement minibatch stochastic gradient descent.
Once we start working with much more complex models,
advantages of high-level APIs will grow considerably.
However, once we have all the basic pieces in place,
the training loop itself is strikingly similar
to what we did when implementing everything from scratch.

딥러닝 프래임트워크 고차원 API를 이용해서 모델을 표현하는 것은 비교적 몇 줄만의 코드로 된다는 것을 눈치챘을 것입이다. 우리는 파라미터를 일일이 할당하거나, 손실 함수를 정의하거나 또는 미니배치 확률적 경사 하강법을 구현할 필요가 없었습니다. 아주 더 복잡한 모델을 다루기 시작하면, 고차원 API의 장점이 상당히 커질 것입니다. 하지만, 모든 기초적인 조각들이 마련된 후에, 학습 룹 자체는 우리가 모든 것을 직접 구현했을 때 했던 것과 놀랍도록 비슷합니다.

To refresh your memory: for some number of epochs,
we will make a complete pass over the dataset (`train_data`),
iteratively grabbing one minibatch of inputs
and the corresponding ground-truth labels.
For each minibatch, we go through the following ritual:

몇 에폭 동안 우리는 입력들의 미니배치를 임의로 추출하고, 그에 대응하는 그라운드-트루스(groud-truth) 레이블을 반복해서 사용하면서 데이터셋(`train_data`)의 전체를 학습에 사용합니다. 각 미니배치에 대해서 우리는 다음 절차들을 거칩니다.

* Generate predictions by calling `net(X)` and calculate the loss `l` (the forward propagation).
* Calculate gradients by running the backpropagation.
* Update the model parameters by invoking our optimizer.

* `net(X)`를 호툴해서 예측을 생성하고, (정방향 전파(forward propagation)) 손실 `l` 을 계산한다
* 역전파(backpropagation)를 수행해서 경사값을 계산한다
* 옵티마이저를 호출해서 모델 파라이터들을 업데이트한다

For good measure, we compute the loss after each epoch and print it to monitor progress.

측정을 위해서, 우리는 매 에폭 후 손실을 계산하고, 출력해서 진행 상태를 모니터링합니다.

```{.python .input}
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l.mean().asnumpy():f}')
```

```{.python .input}
#@tab pytorch
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

```{.python .input}
#@tab tensorflow
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with tf.GradientTape() as tape:
            l = loss(net(X, training=True), y)
        grads = tape.gradient(l, net.trainable_variables)
        trainer.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

Below, we compare the model parameters learned by training on finite data
and the actual parameters that generated our dataset.
To access parameters,
we first access the layer that we need from `net`
and then access that layer's weights and bias.
As in our from-scratch implementation,
note that our estimated parameters are
close to their ground-truth counterparts.

아래 코드는 유한한 데이터로 학습된 모델 파라미터와 데이터셋을 생성할 때 사용된 실제 파라미터르르 비교합니다. 파라미터를 접근하기 위해서 `net`로 부터 우리가 필요한 층을 접근한 후, 그 층의 가중치와 편향을 접근합니다. 처음부터 구현하기과 같이 예측된 파라미터들은 실제 파라미터들과 아주 근사합니다.

```{.python .input}
w = net[0].weight.data()
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
b = net[0].bias.data()
print(f'error in estimating b: {true_b - b}')
```

```{.python .input}
#@tab pytorch
w = net[0].weight.data
print('error in estimating w:', true_w - d2l.reshape(w, true_w.shape))
b = net[0].bias.data
print('error in estimating b:', true_b - b)
```

```{.python .input}
#@tab tensorflow
w = net.get_weights()[0]
print('error in estimating w', true_w - d2l.reshape(w, true_w.shape))
b = net.get_weights()[1]
print('error in estimating b', true_b - b)
```

## Summary
## 요약

:begin_tab:`mxnet`
* Using Gluon, we can implement models much more concisely.
* In Gluon, the `data` module provides tools for data processing, the `nn` module defines a large number of neural network layers, and the `loss` module defines many common loss functions.
* MXNet's module `initializer` provides various methods for model parameter initialization.
* Dimensionality and storage are automatically inferred, but be careful not to attempt to access parameters before they have been initialized.
* Gluon를 사용해서 우리는 모델들을 아주 더 간결하게 구현할 수 있습니다.
* Gluon에서  `data` 모듈은 데이터 처리를 위한 도구들을 제공하고, `nn` 모듈은 많은 종류의 뉴럴 네트워크 층을 정의하고 있고, `loss`모듈은 일반적인 손실 함수 여러개를 정의합니다.
* MXNet의 모듈 `initializer` 은 모델 파라미터 초기화를 위한 여러가지 메소드를 제공합니다.
* 차원과 스토리지는 자동으로 지연되지만, 초기화되기 전에 파라미터에 접근하지 않도록 주의를 기울이세요.
:end_tab:

:begin_tab:`pytorch`
* Using PyTorch's high-level APIs, we can implement models much more concisely.
* In PyTorch, the `data` module provides tools for data processing, the `nn` module defines a large number of neural network layers and common loss functions.
* We can initialize the parameters by replacing their values with methods ending with `_`.
* PyTorch의 고차원 API를 사용해서 우리는 모델들을 아주 더 간결하게 구현할 수 있습니다.
* PyTorch에서  `data` 모듈은 데이터 처리를 위한 도구들을 제공하고, `nn` 모듈은 많은 종류의 뉴럴 네트워크 층과 손실 함수들을 정의합니다.
* `_`로 끝나는 메소드를 사용해서 값을 바꾸는 것으로 파라미터들을 초기화할 수 있습니다.
:end_tab:

:begin_tab:`tensorflow`
* Using TensorFlow's high-level APIs, we can implement models much more concisely.
* In TensorFlow, the `data` module provides tools for data processing, the `keras` module defines a large number of neural network layers and common loss functions.
* TensorFlow's module `initializers` provides various methods for model parameter initialization.
* Dimensionality and storage are automatically inferred (but be careful not to attempt to access parameters before they have been initialized).
* TensorFlow의 고차원 API를 사용해서 우리는 모델들을 아주 더 간결하게 구현할 수 있습니다.
* TensorFlow에서  `data` 모듈은 데이터 처리를 위한 도구들을 제공하고, `keras` 모듈은 많은 종류의 뉴럴 네트워크 층과 손실 함수들을 정의합니다.
* TensorFlow 모듈 `initializers` 는 모델 파리미터 초기화를 위한 다양한 메소드를 제공합니다.
* 차원과 스토리지는 자동으로 지연되지만, 초기화되기 전에 파라미터에 접근하지 않도록 주의를 기울이세요.
:end_tab:

## Exercises
## 연습문제

:begin_tab:`mxnet`
1. If we replace `l = loss(output, y)` with `l = loss(output, y).mean()`, we need to change `trainer.step(batch_size)` to `trainer.step(1)` for the code to behave identically. Why?
1. Review the MXNet documentation to see what loss functions and initialization methods are provided in the modules `gluon.loss` and `init`. Replace the loss by Huber's loss.
1. How do you access the gradient of `dense.weight`?

1. `l = loss(output, y)`을 `l = loss(output, y).mean()`로 바꾸면, 코드가 동일하게 동작하려면 `trainer.step(batch_size)`를  `trainer.step(1)` 로 바꿔야합니다. 왜그럴까요?
1. `gluon.loss` 와 `init`에서 어떤 손실 함수와 초기화 메소드를 제공하는지 MXNet 문서를 리뷰하세요. 손실을 Huber 손실로 바꾸세요.
1. `dense.weight`의 경사값을 어떻게 접근할 수 있나요?
[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
1. If we replace `nn.MSELoss(reduction='sum')` with `nn.MSELoss()`, how can we change the learning rate for the code to behave identically. Why?
1. Review the PyTorch documentation to see what loss functions and initialization methods are provided. Replace the loss by Huber's loss.
1. How do you access the gradient of `net[0].weight`?

1. `nn.MSELoss(reduction='sum')` 를  `nn.MSELoss()`로 바꿀 경우, 코드가 동일하게 동작하게 만들려면 학습 속도를 어떻게 바꿔야하나? 왜 그런가요?
1. 어떤 손실 함수와 초기화 메소드를 제공하는지 PyTorch 문서를 리뷰하세요. 손실을 Huber 손실로 바꾸세요.
1. `net[0].weight`의 경사값을 어떻게 접근할 수 있나요?
[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
1. Review the TensorFlow documentation to see what loss functions and initialization methods are provided. Replace the loss by Huber's loss.
1. 어떤 손실 함수와 초기화 메소드를 제공하는지 TensorFlow 문서를 리뷰하세요. 손실을 Huber 손실로 바꾸세요.
[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:
