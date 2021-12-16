# 멀티레이어 퍼셉트론의 간결한 구현
:label:`sec_mlp_concise`

예상대로 (**상위 수준 API에 의존하면 MLP를 훨씬 더 간결하게 구현할 수 있습니다.**)

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

## 모델

소프트맥스 회귀 구현 (:numref:`sec_softmax_concise`) 의 간결한 구현과 비교할 때 유일한 차이점은
*두 개의* 완전 연결 레이어
(이전에는*one*을 추가했습니다).첫 번째는 [**우리의 히든 레이어**] 입니다. (**256개의 히든 유닛을 포함하고 ReLU 활성화 기능을 적용합니다**).두 번째는 출력 계층입니다.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)])
```

[**훈련 루프**] 는 소프트맥스 회귀를 구현했을 때와 정확히 동일합니다.이러한 모듈성을 통해 모델 아키텍처와 관련된 문제를 직교 고려 사항과 구분할 수 있습니다.

```{.python .input}
batch_size, lr, num_epochs = 256, 0.1, 10
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
```

```{.python .input}
#@tab pytorch
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
```

```{.python .input}
#@tab tensorflow
batch_size, lr, num_epochs = 256, 0.1, 10
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
```

```{.python .input}
#@tab all
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 요약

* 상위 수준 API를 사용하면 MLP를 훨씬 더 간결하게 구현할 수 있습니다.
* 동일한 분류 문제에서 MLP의 구현은 활성화 함수가 있는 추가 은닉 계층을 제외하면 소프트맥스 회귀의 구현과 동일합니다.

## 연습문제

1. 다른 개수의 은닉 레이어를 추가해 보십시오 (학습률을 수정할 수도 있음).어떤 설정이 가장 적합한가요?
1. 다양한 활성화 기능을 사용해 보십시오.어느 것이 가장 효과적일까요?
1. 가중치를 초기화하기 위해 다른 방법을 시도해 보십시오.어떤 방법이 가장 효과적입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/94)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/95)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/262)
:end_tab:
