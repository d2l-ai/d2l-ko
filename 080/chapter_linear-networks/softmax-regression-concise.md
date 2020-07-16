# Concise Implementation of Softmax Regression
# 소트트맥스의 간결한 구현
:label:`sec_softmax_gluon`

Just as high-level APIs of deep learning frameworks
made it much easier
to implement linear regression in :numref:`sec_linear_gluon`,
we will find it similarly (or possibly more)
convenient for implementing classification models.

:numref:`sec_linear_gluon`에서 선형 회귀의 구현을 딥러닝 프래임워크의 고차원 API를 사용하면 간단하게 구현할 수 있었듯이, 분류 모델 구현도 비슷하게(또는 더) 편리하다는 것을 보여 줄 것입니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

Let us stick with the Fashion-MNIST dataset
and keep the batch size at 256 as in :numref:`sec_softmax_scratch`.

:numref:`sec_softmax_scratch`에서와 같이 Fashion-MNIST 데이터셋을 사용하고 배치 크기는 256으로 하겠습니다.

```{.python .input}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

```{.python .input}
#@tab pytorch
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

```{.python .input}
#@tab tensorflow
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Initializing Model Parameters
## 모델 파라미터 초기화하기

As mentioned in :numref:`sec_softmax`,
the output layer of softmax regression
is a fully-connected layer.
Therefore, to implement our model,
we just need to add one fully-connected layer
with 10 outputs to our `Sequential`.
Again, here, the `Sequential` is not really necessary,
but we might as well form the habit since it will be ubiquitous
when implementing deep models.
Again, we initialize the weights at random
with zero mean and standard deviation 0.01.

:numref:`sec_softmax`에서 언급했듯이 소프트맥스 회귀의 출력 층은 완전 연결층입니다. 따라서, 모델을 구현할 때, `Sequential`에 10개 출력을 갖는 완전 연결층을 추가하는 것이 필요합니다. 여기서도 `Sequential`은 실제로 필요하지 않습니다만, 딥러닝 모델을 구현할 때 어디에나 사용되기 때문에 우리도 사용하는 습관을 만들도록 합니다. 마찬가지로 가중치는 평균이 0이고 표준 편차가 0.01인 분포에 따른 난수로 초기화 합니다.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
# PyTorch doesn't implicitly reshape the inputs.
# Thus we define a layer to reshape the inputs in our network.
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1,784)

net = nn.Sequential(Reshape(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
net.add(tf.keras.layers.Dense(10, kernel_initializer=weight_initializer))
```

## Softmax Implementation Revisited
## 소프트맥스 구현 다시 보기

In the previous example of :numref:`sec_softmax_scratch`,
we calculated our model's output
and then ran this output through the cross-entropy loss.
Mathematically, that is a perfectly reasonable thing to do.
However, from a computational perspective,
exponentiation can be a source of numerical stability issues.

:numref:`sec_softmax_scratch`의 예제에서 모델의 출력을 계산하고, 이를 크로스-엔트로피 손실에 전달했습니다. 수학적으로 이렇게 하는 것은 완벽하게 논리적입니다. 하지만, 계산적인 관점에서 보면 지수 연산은 수치적 불안정 이슈의 근원이 될 수 있습니다.

Recall that the softmax function calculates
$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$,
where $\hat y_j$ is the $j^\text{th}$ element of
the predicted probability distribution $\hat{\mathbf{y}}$
and $o_j$ is the $j^\text{th}$ element of the logits
$\mathbf{o}$.
If some of the $o_k$ are very large (i.e., very positive),
then $\exp(o_k)$ might be larger than the largest number
we can have for certain data types (i.e., *overflow*).
This would make the denominator (and/or numerator) `inf` (infinity)
and we wind up encountering either 0, `inf`, or `nan` (not a number) for $\hat y_j$.
In these situations we do not get a well-defined
return value for cross entropy.

소프트맥스 함수는 $\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$ 를 계산합니다. 여기서, $\hat y_j$는 예측된 확률 분포 $\hat{\mathbf{y}}$의 $$번째 원소이고, $o_j$는 로짓 $\mathbf{o}$의 $j$번째 원소입니다. $o_k$ 중에 어떤 것들이 아주 크면 (즉, 매우 긍적적), $\exp(o_k)$도 특정 데이터 타입이 갖을 수 있는 값도다 더 커실 수 있습니다 (즉, 오버플로우). 이는 분모 (그리고/혹은 분자)를 `inf`(무한)으로 만들어버릴 수 있고, $\hat y_j$이 0,  `inf` 또는 `nan`(Not A Number)가 되버립니다. 이런 경우 우리는 잘 정의된 크로스 엔트로피 값을 얻지 못합니다.

One trick to get around this is to first subtract $\max(o_k)$
from all $o_k$ before proceeding with the softmax calculation.
You can verify that this shifting of each $o_k$ by constant factor
does not change the return value of softmax.
After the subtraction and normalization step,
it might be possible that some $o_j$ have large negative values
and thus that the corresponding $\exp(o_j)$ will take values close to zero.
These might be rounded to zero due to finite precision (i.e., *underflow*),
making $\hat y_j$ zero and giving us `-inf` for $\log(\hat y_j)$.
A few steps down the road in backpropagation,
we might find ourselves faced with a screenful
of the dreaded `nan` results.

이를 피하는 방법 중에 하나는 소프트맥스 계산을 하기 전에 모든 $o_k$에서 $\max(o_k)$를 뺍니다. 각 $o_k$를 상수만큼 이동시커도 소프트맥스의 결과값이 바뀌지 않는 것은 증명이 가능합니다. 빼기와 정규화를 적용하면, 어떤 $o_j$는 큰 음수 값을 갖을 가능성이 있고 따라서 이에 대한 $\exp(o_j)$ 값은 0에 가까운 수가 될 것입니다. 유한 소수점 (언더플로우)로 인해서 이 값들이 0이 될 수 있습니다. 이는 $\hat y_j$를 0으로 만들고,  $\log(\hat y_j)$는 `-inf`이 되도록 합니다. 역전파를 몇 단계 수행하면, 화면 가득 끔찍한 `nan` 결과가 화면 가득 나올 수 있습니다.

Fortunately, we are saved by the fact that
even though we are computing exponential functions,
we ultimately intend to take their log
(when calculating the cross-entropy loss).
By combining these two operators
softmax and cross entropy together,
we can escape the numerical stability issues
that might otherwise plague us during backpropagation.
As shown in the equation below, we avoid calculating $\exp(o_j)$
and can use instead $o_j$ directly due to the canceling in $\log(\exp(\cdot))$.

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j)}{\sum_k \exp(o_k)}\right) \\
& = \log{(\exp(o_j))}-\text{log}{\left( \sum_k \exp(o_k) \right)} \\
& = o_j -\log{\left( \sum_k \exp(o_k) \right)}.
\end{aligned}
$$

운이 좋게도 지수 함수를 계산하지만, (크로스-엔트로피 손실을 계산할 때) 로그를 취하려는 의도를 가지고 있다는 사실이 우리를 구해줍니다. 소프트맥스와 크로스 엔트로피 연산을 함께 합하면 연전파에서 격었을 수치적 불안정 이슈를 피할 수 있습니다. 아래 공식에서 보이듯이, $\exp(o_j)$ 계산을 피하고 $\log(\exp(\cdot))$에서 항을 삭제한 덕분에 $o_j$를 그대로 사용할 수 있습니다.

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j)}{\sum_k \exp(o_k)}\right) \\
& = \log{(\exp(o_j))}-\text{log}{\left( \sum_k \exp(o_k) \right)} \\
& = o_j -\log{\left( \sum_k \exp(o_k) \right)}.
\end{aligned}
$$

We will want to keep the conventional softmax function handy
in case we ever want to evaluate the output probabilities by our model.
But instead of passing softmax probabilities into our new loss function,
we will just pass the logits and compute the softmax and its log
all at once inside the cross entropy loss function,
which does smart things like the ["LogSumExp trick"](https://en.wikipedia.org/wiki/LogSumExp).

모델의 출력 확률을 평가하는 용도로 통상적인 소프트맥스 함수를 사용하기 쉽게 만들고 싶습니다. 하지만 소프트맥스 확률을 새로운 손실 함수에 전달하는 대신, ["LogSumExp trick"](https://en.wikipedia.org/wiki/LogSumExp)처럼 똑똑한 크로스 앤드로피 손실 함수에 로짓을 전달하고 소프트맥스와 로그를 한번에 계산합니다.

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss()
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

## Optimization Algorithm
## 최적화 알고리즘

Here, we use minibatch stochastic gradient descent
with a learning rate of 0.1 as the optimization algorithm.
Note that this is the same as we applied in the linear regression example
and it illustrates the general applicability of the optimizers.

여기서 우리는 최적화 알고리즘으로 학습 속도를  0.1로 설정한 미니배치 확률적 경사 하강법을 사용합니다. 이것은 선형 회귀 예제에서 적용한 것과 동일하고, 이 것은 옵티마이저의 일반적인 사용법임을 주의하세요.

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=.1)
```

## Training
## 학습

Next we call the training function defined in :numref:`sec_softmax_scratch` to train the model.

다음 단계로 모델을 학습하기 위해서 :numref:`sec_softmax_scratch`에서 정의한 학습 함수를 호출합니다.

```{.python .input}
#@tab all
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

As before, this algorithm converges to a solution
that achieves a decent accuracy,
albeit this time with fewer lines of code than before.
Note that in many cases, a deep learning framework takes additional precautions
beyond these most well-known tricks to ensure numerical stability,
saving us from even more pitfalls that we would encounter
if we tried to code all of our models from scratch in practice.

이전보다 적은 줄의 코드이지만 이 알고리즘은 이전과 같이 쓸만한 정확도를 달성하는 솔루션으로 수렴할 것입니다. 많은 경우에 딥러닝 프래임워크는 수치적 인정성을 보장하기 위한 잘 알려진 트릭들 이외의 추가적인 안전장치를 제공합니다. 이는 모델의 모든 코드를 직접 처음부터 구현할 경우 만날 수 있는 더 많은 오류를 피해갈 수 있도록  해줍니다.

## Exercises
## 연습문제

1. Try adjusting the hyperparameters, such as the batch size, number of epochs, and learning rate, to see what the results are.
1. Increase the numper of epochs for training. Why might the test accuracy decrease after a while? How could we fix this?

1. 배치 크기, 에폭수, 학습 속도와 같은 하이퍼파라미터를 조정해서 결과가 어떻게되는지 실험해 보세요.
1. 학습 에폭 수를 늘려보세요. 얼마 후, 테스트 정확도가 왜 줄어들까요? 어떻게 고칠 수 있나요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/260)
:end_tab:
