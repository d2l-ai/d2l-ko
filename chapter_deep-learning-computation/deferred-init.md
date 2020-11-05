# Deferred Initialization
# 지연된 초기화
:label:`sec_deferred_init`

0.15.0

So far, it might seem that we got away
with being sloppy in setting up our networks.
Specifically, we did the following unintuitive things,
which might not seem like they should work:

지금까지는 네트워크 설정을 엉성하게 한 것처럼 보일 수 있습니다. 구체적으로는 다음과 같은 직관적이지 않은 것들을 했는데, 이것들은 동작하지 못할 것처럼 보입니다.

* We defined the network architectures
  without specifying the input dimensionality.
* We added layers without specifying
  the output dimension of the previous layer.
* We even "initialized" these parameters
  before providing enough information to determine
  how many parameters our models should contain.

- 우리는 입력 차원을 명시하지 않은 채로 네트워크 아키텍처를 정의했습니다.
- 이전 층의 출력 차원을 명시하지 않고, 층들을 추가했습니다.
- 심지어 모델이 몇 개의 파라미터들을 갖아야하는지를 정하는 데 충분한 정보를 제공하지도 않고 파라미터들을 "초기화"했습니다.

You might be surprised that our code runs at all.
After all, there is no way the deep learning framework
could tell what the input dimensionality of a network would be.
The trick here is that the framework *defers initialization*,
waiting until the first time we pass data through the model,
to infer the sizes of each layer on the fly.

여러분은 우리의 코드가 실제로 동작한다는 것이 놀랄 것입니다. 결국, 딥러닝 프레임워크가 네트워크의 입력 차원이 무엇인지 알아낼 수 있는 방법이 없었습니다. 여기서 속임수는 프레임워크가 *초기화를 지연* 시킨다는 것입니다. 즉, 각 층의 크기를 즉석해서 알아내기 위해서 모델에 데이터가 처음으로 전달될 때까지 기다립니다.

Later on, when working with convolutional neural networks,
this technique will become even more convenient
since the input dimensionality
(i.e., the resolution of an image)
will affect the dimensionality
of each subsequent layer.
Hence, the ability to set parameters
without the need to know,
at the time of writing the code,
what the dimensionality is
can greatly simplify the task of specifying
and subsequently modifying our models.
Next, we go deeper into the mechanics of initialization.

이 후에 컨볼루셔널 신경망을 다룰 때 이 기법은 더 유용해집니다. 입력 차원(즉, 이미지의 해상도)가 각 층의 차원에 영향을 주기 때문입니다. 따라서, 코드를 작성하는 시점에 파라미터의 차원을 몰라도 되는 것은 모델을 정의하고 그 후에 변경하는 일을 굉장히 단순화시켜 줍니다. 다음으로, 우리는 초기화 작동법에 대해서 더 깊이 살펴보겠습니다.


## Instantiating a Network
## 네트워크 생성하기(instantiaing)

To begin, let us instantiate an MLP.

MLP를 생성하는 것으로 시작합니다.

```{.python .input}
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    return net

net = get_net()
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

At this point, the network cannot possibly know
the dimensions of the input layer's weights
because the input dimension remains unknown.
Consequently the framework has not yet initialized any parameters.
We confirm by attempting to access the parameters below.

이 때, 입력 차원이 알려지지 않았기 때문에, 네트워크는 입력 층의 가충치에 대한 차원을 알 수 있을 것입니다. 따라서, 프레임워크는 어떤 파라미터도 아직은 초기화 할 수 없습니다. 아래와 같이 파라미터 접근을 시도해서 확인해 보세요.

```{.python .input}
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
#@tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
Note that while the parameter objects exist,
the input dimension to each layer is listed as -1.
MXNet uses the special value -1 to indicate
that the parameter dimension remains unknown.
At this point, attempts to access `net[0].weight.data()`
would trigger a runtime error stating that the network
must be initialized before the parameters can be accessed.
Now let us see what happens when we attempt to initialize
parameters via the `initialize` function.

파라미터 객체가 존재하지만, 각 층의 입력 차원은 -1로 나오고 있음을 기억하세요. MXNet은 -1이라는 특별한 값으로 파라미터 차원이 알려지지 않은 상태인 것을 표현합니다. 이 시점에서  `net[0].weight.data()` 를 접근하려고 하면, 파라미터에 접근하기 전에 네트워크가 초기화되어야한다는 런터임 에러가 발생할 것입니다. 이제  `initialize` 함수를 통해서 파라미터를 초기화하면 어떤일이 일어나는지 보겠습니다.

:end_tab:

:begin_tab:`tensorflow`
Note that each layer objects exist but the weights are empty.
Using `net.get_weights()` would throw an error since the weights
have not been initialized yet.

각 층의 객체는 존재하지만, 가중치가 비워있음을 주목하세요.  `net.get_weights()` 를 호출하면 가중치가 아직 초기화되지 않았기 때문에, 에러가 발생할 것입니다.

:end_tab:

```{.python .input}
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
As we can see, nothing has changed.
When input dimensions are unknown,
calls to initialize do not truly initialize the parameters.
Instead, this call registers to MXNet that we wish
(and optionally, according to which distribution)
to initialize the parameters.

보이는 것처럼 아무것도 일어나지 않았습니다. 입력 차원을 모를 때는 초기화 호출이 파라미터들 실제로 초기화하지 않습니다. 대신, 이 호출은 MXNet에 파라미터 초기화를 원한다는 것을 등록해줍니다. (그리고 선택적으로 어떤 분포를 사용할지도)

:end_tab:

Next let us pass data through the network
to make the framework finally initialize parameters.

이제 네트워크에 데이터를 흘려서, 프레임워크가 최종적으로 파라미터를 초기화하도록 하겠습니다.

```{.python .input}
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

As soon as we know the input dimensionality,
20,
the framework can identify the shape of the first layer's weight matrix by plugging in the value of 20.
Having recognized the first layer's shape, the framework proceeds
to the second layer,
and so on through the computational graph
until all shapes are known.
Note that in this case,
only the first layer requires deferred initialization,
but the framework initializes sequentially.
Once all parameter shapes are known,
the framework can finally initialize the parameters.

입력 차원이 20임을 알게되자 마자, 20이라는 값을 넣어서 프레임워크는 첫번째 층의 가중치 행렬의 모양을 알아낼 수 있습니다. 첫번째 층의 모양을 알았기 때문에, 프레임워크는 두번째 층으로 진행하고, 모든 모양을 알아낼 때까지 연산 그래프를 진행합니다. 이경우 오직 첫번째 층만 지연된 초기화가 필요하고, 프레임워크가 순차적으로 초기화한다는 것을 주목하세요. 모든 파라미터들의 모양이 파악되면, 프레임워크는 파라미터들을 최기화를 최종적으로 수행합니다.

## Summary
## 요약

* Deferred initialization can be convenient, allowing the framework to infer parameter shapes automatically, making it easy to modify architectures and eliminating one common source of errors.
* We can pass data through the model to make the framework finally initialize parameters.

- 프레임워크가 파라미터 모양을 자동으로 추론하도록 하고, 아키텍처 수정하는 것을 쉽게 만들고, 일반적인 에러의 원천을 제거해주기 때문에, 지연된 초기화는 편리합니다. 
- 모델에 데이터를 전달하면, 프레임워크는 파라미터 초기화를 마침내 수행합니다.


## Exercises

1. What happens if you specify the input dimensions to the first layer but not to subsequent layers? Do you get immediate initialization?
1. What happens if you specify mismatching dimensions?
1. What would you need to do if you have input of varying dimensionality? Hint: look at the parameter tying.

1. 여러분이 첫번째 층의 입력 차원만 명시하고, 이후 층에 대해서는 하지 않을 경우 어떤 일이 발생할까요? 바로 초기화하 일어날까요?
2. 불일치하는 차원을 지정하면 어떤 일이 일어날까요?
3. 차원이 변하는 입력을 사용한다면 무엇을 해야할까요? 힌트: 파라미터 묶기를 보세요.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/281)
:end_tab:
