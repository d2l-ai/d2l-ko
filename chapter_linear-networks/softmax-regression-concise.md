# 소프트맥스 회귀의 간결한 구현
:label:`sec_softmax_concise`

:numref:`sec_linear_concise`의 딥 러닝 프레임워크 (**선형 회귀 구현이 훨씬 쉬워졌습니다**) 의 (**상위 수준 API**와 마찬가지로), 분류 모델을 구현하는 데 편리합니다 (**비슷하게 찾을 수 있습니다**) (~~here~~) (또는 그 이상).패션-MNIST 데이터세트를 고수하고 :numref:`sec_softmax_scratch`에서와 같이 배치 크기를 256으로 유지해 보겠습니다.

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

:numref:`sec_softmax`에서 언급했듯이 [**소프트맥스 회귀의 출력 계층은 완전 연결 계층입니다.**] 따라서 모델을 구현하려면 출력이 10개인 완전 연결 계층 하나를 `Sequential`에 추가하기만 하면 됩니다.다시 말하지만, `Sequential`는 실제로 필요하지 않지만 심층 모델을 구현할 때 어디에나 존재하기 때문에 습관을 형성 할 수도 있습니다.다시 말하지만 평균이 0이고 표준 편차가 0.01인 가중치를 임의로 초기화합니다.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
# PyTorch does not implicitly reshape the inputs. Thus we define the flatten
# layer to reshape the inputs before the linear layer in our network
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
net.add(tf.keras.layers.Dense(10, kernel_initializer=weight_initializer))
```

## 소프트맥스 구현 재검토
:label:`subsec_softmax-implementation-revisited`

:numref:`sec_softmax_scratch`의 이전 예에서는 모델의 출력을 계산한 다음 교차 엔트로피 손실을 통해 이 출력을 실행했습니다.수학적으로 이것은 완벽하게 합리적인 일입니다.그러나 계산적 관점에서 지수화는 수치 안정성 문제의 원인이 될 수 있습니다. 

소프트맥스 함수는 $\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$을 계산합니다. 여기서 $\hat y_j$는 예측된 확률 분포 $\hat{\mathbf{y}}$의 $j^\mathrm{th}$이고 $o_j$은 로짓 $\mathbf{o}$의 $j^\mathrm{th}$ 요소입니다.$o_k$ 중 일부가 매우 큰 경우 (즉, 매우 양수) $\exp(o_k)$이 특정 데이터 유형 (예: *오버플로*) 에 대해 가질 수 있는 가장 큰 수보다 클 수 있습니다.이것은 분모 (및/또는 분자) `inf` (무한대) 를 만들고 $\hat y_j$에 대해 0, `inf` 또는 `nan` (숫자가 아님) 를 만나게됩니다.이러한 상황에서는 교차 엔트로피에 대해 잘 정의된 반환 값을 얻지 못합니다. 

이 문제를 해결하기 위한 한 가지 비결은 소프트맥스 계산을 진행하기 전에 먼저 모든 $o_k$에서 $\max(o_k)$를 빼는 것입니다.상수 인자에 의해 각 $o_k$를 이동해도 소프트맥스의 반환 값은 변경되지 않는다는 것을 알 수 있습니다. 

$$
\begin{aligned}
\hat y_j & =  \frac{\exp(o_j - \max(o_k))\exp(\max(o_k))}{\sum_k \exp(o_k - \max(o_k))\exp(\max(o_k))} \\
& = \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}.
\end{aligned}
$$

빼기 및 정규화 단계 후에 일부 $o_j - \max(o_k)$는 큰 음수 값을 가지므로 해당 $\exp(o_j - \max(o_k))$이 0에 가까운 값을 가질 수 있습니다.유한 정밀도 (즉, *언더플로*) 로 인해 0으로 반올림되어 $\hat y_j$을 0으로 만들고 $\log(\hat y_j)$에 대해 `-inf`를 제공할 수 있습니다.역전파의 길을 몇 걸음 내려가면, 우리는 두려운 `nan` 결과를 스크린으로 보게 될 것입니다. 

다행히도 지수 함수를 계산하더라도 궁극적으로 (교차 엔트로피 손실을 계산할 때) 로그를 가져올 것이라는 사실에 의해 절약됩니다.이 두 연산자 softmax와 cross-entropy를 함께 결합하면 역전파 중에 우리를 괴롭힐 수 있는 수치 안정성 문제를 피할 수 있습니다.아래 방정식에서 볼 수 있듯이 $\exp(o_j - \max(o_k))$를 계산하는 것을 피하고 $\log(\exp(\cdot))$의 취소로 인해 $o_j - \max(o_k)$을 직접 사용할 수 있습니다. 

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}\right) \\
& = \log{(\exp(o_j - \max(o_k)))}-\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)} \\
& = o_j - \max(o_k) -\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)}.
\end{aligned}
$$

모델을 통해 출력 확률을 평가하려는 경우 기존 softmax 함수를 편리하게 사용할 수 있습니다.그러나 소프트맥스 확률을 새로운 손실 함수에 전달하는 대신 [**LogsumExp trick "](https://en.wikipedia.org/wiki/LogSumExp) 과 같은 스마트한 작업을 수행하는 교차 엔트로피 손실 함수 내에서 로짓을 전달하고 소프트맥스와 그 로그를 한꺼번에 계산합니다.

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

여기서는 최적화 알고리즘으로 학습률이 0.1인 (**미니배치 확률적 기울기 하강** 사용**) 합니다.이는 선형 회귀 예제에서 적용한 것과 동일하며 옵티마이저의 일반적인 적용 가능성을 보여줍니다.

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

## 트레이닝

다음으로 :numref:`sec_softmax_scratch`에서 [**정의된 훈련 함수를 호출**](~~이전~~) 하여 모델을 훈련시킵니다.

```{.python .input}
#@tab all
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

이전과 마찬가지로이 알고리즘은 이전보다 적은 코드 줄로 이번에는 적절한 정확도를 달성하는 솔루션으로 수렴됩니다. 

## 요약

* 상위 수준 API를 사용하여 소프트맥스 회귀를 훨씬 더 간결하게 구현할 수 있습니다.
* 계산적 관점에서 소프트맥스 회귀를 구현하는 데는 복잡한 작업이 있습니다.대부분의 경우 딥 러닝 프레임워크는 수치적 안정성을 보장하기 위해 가장 잘 알려진 트릭 외에 추가적인 예방 조치를 취하므로 실제로 모든 모델을 처음부터 코딩하려고 할 때 발생할 수 있는 더 많은 함정에서 벗어날 수 있습니다.

## 연습문제

1. 배치 크기, Epoch 수, 학습률과 같은 초모수를 조정하여 결과가 무엇인지 확인해 보십시오.
1. 훈련을 위한 Epoch 수를 늘립니다.잠시 후 테스트 정확도가 떨어지는 이유는 무엇입니까?이 문제를 어떻게 해결할 수 있을까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/260)
:end_tab:
