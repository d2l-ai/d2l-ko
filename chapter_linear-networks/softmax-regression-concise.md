# Softmax 회귀 분석의 간결한 구현
:label:`sec_softmax_concise`

딥 러닝 프레임워크의 상위 수준 API가 :numref:`sec_linear_concise`에서 선형 회귀를 구현하는 것이 훨씬 쉬워졌기 때문에 분류 모델을 구현하는 데 비슷하게 (또는 그 이상) 편리하다는 것을 알게 될 것입니다.우리가 패션 - MNIST 데이터 세트를 고수하고 :numref:`sec_softmax_scratch`에서와 같이 256에서 배치 크기를 유지하자.

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

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 모델 매개변수 초기화

:numref:`sec_softmax`에서 언급 한 바와 같이, 소프트 맥스 회귀의 출력 층은 완전히 연결된 층이다.따라서 모델을 구현하려면 `Sequential`에 10 개의 출력이있는 완전히 연결된 레이어 하나를 추가하기 만하면됩니다.다시 말하지만, 여기서 `Sequential`는 실제로 필요하지 않지만 깊은 모델을 구현할 때 유비쿼터스가 될 것이므로 습관을 형성 할 수도 있습니다.다시 말하지만, 우리는 제로 평균과 표준 편차 0.01로 무작위로 가중치를 초기화합니다.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
# PyTorch does not implicitly reshape the inputs. Thus we define a layer to
# reshape the inputs in our network
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

## Softmax 구현 재검토
:label:`subsec_softmax-implementation-revisited`

:numref:`sec_softmax_scratch`의 이전 예에서는 모델의 출력을 계산한 다음 교차 엔트로피 손실을 통해이 출력을 실행했습니다.수학적으로, 그것은 할 수있는 완벽하게 합리적인 일입니다.그러나 계산 관점에서 지수는 수치 안정성 문제의 원인이 될 수 있습니다.

소프트맥스 함수는 $\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$을 계산합니다. 여기서 $\hat y_j$는 예측 확률 분포 $\hat{\mathbf{y}}$의 원소이고 $o_j$은 로짓의 $j^\mathrm{th}$ 원소입니다.$o_k$ 중 일부가 매우 큰 경우 (즉, 매우 양수) $\exp(o_k)$은 특정 데이터 유형 (예: * 오버플로*) 에 대해 가질 수있는 가장 큰 수보다 클 수 있습니다.이것은 분모 (및/또는 분자) `inf` (무한대) 를 만들 것이고 우리는 $\hat y_j$에 대해 0, `inf` 또는 `nan` (숫자가 아님) 중 하나를 만나게됩니다.이러한 상황에서 우리는 크로스 엔트로피에 대해 잘 정의 된 반환 값을 얻지 못합니다.

이 문제를 해결할 수있는 한 가지 트릭은 소프트 맥스 계산을 진행하기 전에 모든 $o_k$에서 7323619를 먼저 뺍니다.상수 계수에 의한 각 $o_k$의 이동이 softmax의 반환 값을 변경하지 않는지 확인할 수 있습니다.빼기 및 정규화 단계 후에 일부 $o_j$은 큰 음수 값을 가질 수 있으므로 해당 $\exp(o_j)$이 0에 가까운 값을 취할 수 있습니다.유한 정밀도 (즉, * 언더 플로*) 로 인해 0으로 반올림 될 수 있으며 $\hat y_j$을 0으로 만들고 $\log(\hat y_j)$에 대해 `-inf`를 제공합니다.역 전파의 길 아래로 몇 걸음, 우리는 자신이 지칠대로 지친 `nan` 결과의 스크린에 직면 할 수 있습니다.

다행히도 우리는 지수 함수를 계산하고 있음에도 불구하고 궁극적으로 로그 (교차 엔트로피 손실을 계산할 때) 를 취하려고한다는 사실에 의해 저장됩니다.이 두 연산자 softmax와 교차 엔트로피를 결합하여 역 전파 중에 우리를 괴롭히는 수치 안정성 문제를 피할 수 있습니다.아래 방정식에서 볼 수 있듯이 $\exp(o_j)$를 계산하지 않고 $\log(\exp(\cdot))$의 취소로 인해 직접 $o_j$를 사용할 수 있습니다.

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j)}{\sum_k \exp(o_k)}\right) \\
& = \log{(\exp(o_j))}-\log{\left( \sum_k \exp(o_k) \right)} \\
& = o_j -\log{\left( \sum_k \exp(o_k) \right)}.
\end{aligned}
$$

우리는 우리의 모델에 의해 출력 확률을 평가하려는 경우에 대비하여 기존의 softmax 함수를 편리하게 유지하고자 할 것입니다.그러나 softmax 확률을 새로운 손실 함수에 전달하는 대신 로짓을 전달하고 크로스 엔트로피 손실 함수 내에서 softmax와 로그를 한꺼번에 계산합니다. 이 함수는 ["LogSumeXP 트릭"](https://en.wikipedia.org/wiki/LogSumExp).

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

## 최적화 알고리즘

여기서 우리는 최적화 알고리즘으로 0.1의 학습 속도를 가진 미니 배치 확률 적 그래디언트 강하를 사용합니다.이것은 선형 회귀 예제에서 적용한 것과 동일하며 옵티 마이저의 일반적인 적용 가능성을 보여줍니다.

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

## 교육

다음으로 :numref:`sec_softmax_scratch`에 정의된 교육 함수를 호출하여 모델을 학습합니다.

```{.python .input}
#@tab all
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

이전과 마찬가지로, 이 알고리즘은 이전보다 적은 코드 줄로 이번에는 적절한 정확도를 달성하는 솔루션으로 수렴합니다.

## 요약

* 높은 수준의 API를 사용하여, 우리는 훨씬 더 간결하게 softmax 회귀를 구현할 수 있습니다.
* 계산 관점에서 softmax 회귀 분석을 구현하는 것은 복잡합니다.대부분의 경우 딥 러닝 프레임워크는 수치 안정성을 보장하기 위해 이러한 가장 잘 알려진 트릭을 넘어서는 추가적인 예방 조치를 취하여 실제로 모든 모델을 처음부터 코딩하려고 시도하면 발생할 수 있는 더 많은 함정으로부터 우리를 보호합니다.

## 연습 문제

1. 배치 크기, 에포크 수 및 학습 속도와 같은 하이퍼매개 변수를 조정하여 결과가 무엇인지 확인합니다.
1. 훈련을위한 신기원의 너퍼를 늘리십시오.잠시 후 테스트 정확도가 감소하는 이유는 무엇입니까?어떻게 이 문제를 해결할 수 있을까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/260)
:end_tab:
