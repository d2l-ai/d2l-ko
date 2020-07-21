# 선형 회귀의 간결한 구현
:label:`sec_linear_concise`

지난 몇 년간 딥 러닝에 대한 폭넓고 집중적인 관심으로 기업, 학계 및 취미자들은 그라데이션 기반 학습 알고리즘 구현의 반복 작업을 자동화하기 위한 다양한 성숙한 오픈 소스 프레임워크를 개발하게 되었습니다.:numref:`sec_linear_scratch`에서는 (i) 데이터 저장 및 선형 대수를위한 텐서; (ii) 그라디언트 계산을위한 자동 차별화에만 의존했습니다.실제로 데이터 반복자, 손실 함수, 옵티 마이저 및 신경망 계층이 매우 일반적이기 때문에 현대 라이브러리는 이러한 구성 요소를 구현합니다.

이 섹션에서는 딥 러닝 프레임워크의 고급 API를 사용하여 :numref:`sec_linear_scratch`의 선형 회귀 모델을 간결하게 구현하는 방법을 설명합니다.

## 데이터 세트 생성

시작하려면, 우리는 :numref:`sec_linear_scratch`에서와 동일한 데이터 세트를 생성합니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch.utils import data
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

## 데이터 세트 읽기

자체 반복자를 롤링하는 대신 프레임 워크의 기존 API를 호출하여 데이터를 읽을 수 있습니다.우리는 인수로 `features` 및 `labels`를 전달하고 데이터 반복자 객체를 인스턴스화 할 때 `batch_size`를 지정합니다.게다가 부울 값 `is_train`은 데이터 반복자 객체가 각 신기원 (데이터 집합 통과) 에서 데이터를 섞을 지 여부를 나타냅니다.

```{.python .input}
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a Gluon data iterator."""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab pytorch
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
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

이제 우리는 :numref:`sec_linear_scratch`에서 `data_iter` 함수를 호출 할 때 거의 같은 방식으로 `data_iter`를 사용할 수 있습니다.그것이 작동하는지 확인하기 위해, 우리는 읽고 예제의 첫 번째 미니 배치를 인쇄 할 수 있습니다.:numref:`sec_linear_scratch`와 비교하면, 여기에서 우리는 파이썬 반복자를 구성하고 `next`을 사용하여 반복자에서 첫 번째 항목을 얻을 수 있습니다.

```{.python .input}
#@tab all
next(iter(data_iter))
```

## 모델 정의하기

:numref:`sec_linear_scratch`에서 선형 회귀를 처음부터 구현했을 때 모델 매개 변수를 명시 적으로 정의하고 계산을 코딩하여 기본 선형 대수 연산을 사용하여 출력을 생성합니다.이 작업을 수행하는 방법을 알아야 합니다.그러나 일단 모델이 더 복잡해지면 거의 매일에이 작업을 수행해야한다면 도움을 받게되어 기쁩니다.상황은 처음부터 자신의 블로그를 코딩하는 것과 비슷합니다.한 두 번하는 것은 보람과 유익하지만 블로그가 필요할 때마다 바퀴를 재발명하는 데 한 달을 보냈다면 형편없는 웹 개발자가 될 것입니다.

표준 작업의 경우 프레임 워크의 미리 정의 된 레이어를 사용할 수 있습니다. 이 레이어를 사용하면 구현에 집중하지 않고 모델을 구성하는 데 사용되는 레이어에 특히 집중할 수 있습니다.우리는 먼저 `Sequential` 클래스의 인스턴스를 참조하는 모델 변수 `net`를 정의합니다.`Sequential` 클래스는 함께 연결될 여러 레이어에 대한 컨테이너를 정의합니다.입력 데이터가 주어지면 `Sequential` 인스턴스는 첫 번째 레이어를 통과하여 출력을 두 번째 레이어의 입력 등으로 전달합니다.다음 예에서 모델은 하나의 레이어로 구성되므로 `Sequential`가 실제로 필요하지 않습니다.그러나 향후 거의 모든 모델에 여러 레이어가 포함되므로 가장 표준적인 워크플로를 숙지하기 위해 이 모델을 사용할 것입니다.

:numref:`fig_single_neuron`와 같이 단일 레이어 네트워크의 아키텍처를 상기하십시오.레이어는 각 입력이 행렬 - 벡터 곱셈을 통해 각각의 출력에 연결되기 때문에*완전히 연결된*라고합니다.

:begin_tab:`mxnet`
글루온에서는 완전히 연결된 레이어가 `Dense` 클래스에 정의됩니다.단일 스칼라 출력 만 생성하려고하기 때문에 해당 숫자를 1로 설정합니다.

편의를 위해 Gluon은 각 레이어의 입력 모양을 지정할 필요가 없다는 점은 주목할 가치가 있습니다.그래서 여기, 우리는 얼마나 많은 입력이 선형 층에 들어가는 글루온에게 말할 필요가 없습니다.우리가 처음 모델을 통해 데이터를 전달하려고 할 때 (예: `net(X)`를 나중에 실행할 때), Gluon은 자동으로 각 레이어에 입력 수를 추론합니다.나중에 이것이 어떻게 작동하는지 자세히 설명 할 것입니다.
:end_tab:

:begin_tab:`pytorch`
파이 토치에서, 완전히 연결된 층은 `Linear` 클래스에 정의된다.우리는 `nn.Linear`에 두 개의 인수를 전달했습니다.첫 번째는 입력 피쳐 치수 (2) 를 지정하고 두 번째 치수는 단일 스칼라 (따라서 1) 인 출력 피쳐 치수입니다.
:end_tab:

:begin_tab:`tensorflow`
Keras에서 완전히 연결된 레이어는 `Dense` 클래스에서 정의됩니다.단일 스칼라 출력 만 생성하려고하기 때문에 해당 숫자를 1로 설정합니다.

편의상 Keras에서는 각 레이어의 입력 모양을 지정할 필요가 없습니다.그래서 여기, 우리는 Keras에게 얼마나 많은 입력이 선형 계층에 들어가는지 말할 필요가 없습니다.처음 모델을 통해 데이터를 전달하려고 할 때 (예: `net(X)`를 나중에 실행할 때) Keras는 각 레이어에 대한 입력 수를 자동으로 유추합니다.나중에 이것이 어떻게 작동하는지 자세히 설명 할 것입니다.
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

## 모델 매개변수 초기화

`net`를 사용하기 전에 선형 회귀 모델의 가중치 및 바이어스와 같은 모델 매개 변수를 초기화해야합니다.딥 러닝 프레임워크에는 매개 변수를 초기화하는 미리 정의된 방법이 있습니다.여기서는 평균 0과 표준 편차가 0.01인 정규 분포에서 각 가중치 모수를 랜덤하게 표본으로 추출해야 한다고 지정합니다.바이어스 매개 변수는 0으로 초기화됩니다.

:begin_tab:`mxnet`
우리는 MXNet에서 `initializer` 모듈을 가져옵니다.이 모듈은 모델 매개 변수 초기화를위한 다양한 방법을 제공합니다.글루온은 `init` 바로 가기 (약어) 로 사용할 수 있습니다 `initializer` 패키지에 액세스 할 수 있습니다.`init.Normal(sigma=0.01)`를 호출하여 가중치를 초기화하는 방법 만 지정합니다.바이어스 매개변수는 기본적으로 0으로 초기화됩니다.
:end_tab:

:begin_tab:`pytorch`
`nn.Linear`를 구성 할 때 우리는 입력 및 출력 치수를 지정했습니다.이제 매개 변수에 직접 액세스하여 초기 값을 지정합니다.먼저 네트워크의 첫 번째 계층 인 `net[0]`로 레이어를 찾은 다음 `weight.data` 및 `bias.data` 메서드를 사용하여 매개 변수에 액세스합니다.다음으로 우리는 대체 방법 `normal_` 및 `fill_`을 사용하여 매개 변수 값을 덮어 씁니다.
:end_tab:

:begin_tab:`tensorflow`
텐서플로우의 `initializers` 모듈은 모델 파라미터 초기화를 위한 다양한 방법을 제공합니다.Keras에서 초기화 방법을 지정하는 가장 쉬운 방법은 `kernel_initializer`를 지정하여 레이어를 만드는 것입니다.여기서 우리는 `net`를 다시 재현합니다.
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
위의 코드는 간단 해 보일 수 있지만 여기서 이상한 일이 일어나고 있음을 알아야합니다.우리는 Gluon이 입력에 얼마나 많은 치수를 가지고 있는지 아직 알지 못하더라도 네트워크에 대한 매개 변수를 초기화하고 있습니다!우리의 예에서와 같이 2 일 수도 있고 2000 일 수도 있습니다.Gluon은 장면 뒤에서 초기화가 실제로*연기되기 때문에 이것을 벗어나게합니다.실제 초기화는 네트워크를 통해 데이터를 전달하려고 처음 시도 할 때만 발생합니다.매개 변수가 아직 초기화되지 않았기 때문에 액세스하거나 조작 할 수 없다는 것을 기억하십시오.
:end_tab:

:begin_tab:`pytorch`

:end_tab:

:begin_tab:`tensorflow`
위의 코드는 간단 해 보일 수 있지만 여기서 이상한 일이 일어나고 있음을 알아야합니다.Keras가 입력에 얼마나 많은 차원을 갖는지 아직 알지 못하더라도 네트워크에 대한 매개 변수를 초기화하고 있습니다!우리의 예에서와 같이 2 일 수도 있고 2000 일 수도 있습니다.Keras는 백그라운드에서 초기화가 실제로*연기되기 때문에 이것을 벗어나게합니다.실제 초기화는 네트워크를 통해 데이터를 전달하려고 처음 시도 할 때만 발생합니다.매개 변수가 아직 초기화되지 않았기 때문에 액세스하거나 조작 할 수 없다는 것을 기억하십시오.
:end_tab:

## 손실 함수 정의

:begin_tab:`mxnet`
글루온 (Gluon) 에서 `loss` 모듈은 다양한 손실 기능을 정의합니다.이 예에서, 우리는 제곱 손실 (`L2Loss`) 의 글루온 구현을 사용합니다.
:end_tab:

:begin_tab:`pytorch`
`MSELoss` 클래스는 $L_2$ 표준의 제곱이라고도 하는 평균 제곱 오차를 계산합니다.기본적으로 예제를 통해 평균 손실을 반환합니다.
:end_tab:

:begin_tab:`tensorflow`
`MeanSquaredError` 클래스는 $L_2$ 표준의 제곱이라고도 하는 평균 제곱 오차를 계산합니다.기본적으로 예제를 통해 평균 손실을 반환합니다.
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

## 최적화 알고리즘 정의

:begin_tab:`mxnet`
Minibatch 확률 그래디언트 하강은 신경망을 최적화하기위한 표준 도구이므로 Gluon은 `Trainer` 클래스를 통해이 알고리즘의 여러 변형과 함께 지원합니다.`Trainer`을 인스턴스화 할 때 최적화 할 매개 변수 (`net.collect_params()`를 통해 모델 `net`에서 얻을 수 있음), 사용하려는 최적화 알고리즘 (`sgd`) 및 최적화 알고리즘에 필요한 하이퍼 매개 변수 사전을 지정합니다.미니 배치 확률 그래디언트 하강은 여기서 0.03으로 설정된 `learning_rate` 값을 설정해야합니다.
:end_tab:

:begin_tab:`pytorch`
미니 배치 확률 그라데이션 강하는 신경망을 최적화하기위한 표준 도구이므로 PyTorch는 `optim` 모듈에서이 알고리즘에 대한 다양한 변형과 함께 지원합니다.`SGD` 인스턴스를 인스턴스화 할 때 최적화 알고리즘에 필요한 하이퍼 매개 변수 사전을 사용하여 최적화 할 매개 변수를 지정합니다 (`net.parameters()`를 통해 인터넷에서 얻을 수 있음).미니 배치 확률 그래디언트 하강은 여기서 0.03으로 설정된 `lr` 값을 설정해야합니다.
:end_tab:

:begin_tab:`tensorflow`
Minibatch 확률 그래디언트 하강은 신경망을 최적화하기위한 표준 도구이므로 Keras는 `optimizers` 모듈에서이 알고리즘에 대한 다양한 변형과 함께 지원합니다.미니 배치 확률 그래디언트 하강은 여기서 0.03으로 설정된 값 `learning_rate`를 설정해야합니다.
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

## 교육

딥 러닝 프레임워크의 상위 수준 API를 통해 모델을 표현하려면 비교적 적은 수의 코드가 필요하다는 것을 눈치 챘을 것입니다.우리는 매개 변수를 개별적으로 할당하거나 손실 함수를 정의하거나 미니 배치 확률 적 그라데이션 강하를 구현할 필요가 없었습니다.훨씬 더 복잡한 모델로 작업하기 시작하면 고급 API의 이점이 상당히 커질 것입니다.그러나 일단 우리가 모든 기본 부분을 제자리에 갖게되면, 훈련 루프 자체는 처음부터 모든 것을 구현할 때 우리가 한 것과 현저하게 유사합니다.

메모리를 새로 고치려면: 일정 수의 신기원 동안 데이터 세트 (`train_data`) 를 완전히 통과하여 하나의 입력 및 해당 접지 진실 레이블을 반복적으로 가져옵니다.각 미니 배치에 대해 다음과 같은 의식을 거쳐야합니다.

* `net(X)`를 호출하여 예측을 생성하고 손실 `l` (전방 전파) 를 계산합니다.
* 역 전파를 실행하여 그라디언트를 계산합니다.
* 옵티 마이저를 호출하여 모델 매개 변수를 업데이트합니다.

좋은 측정을 위해 각 신기원 이후의 손실을 계산하고 진행률을 모니터링하기 위해 인쇄합니다.

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

아래에서는 유한 데이터에 대한 교육을 통해 얻은 모델 매개 변수와 데이터 집합을 생성한 실제 매개 변수를 비교합니다.매개 변수에 액세스하려면 먼저 `net`에서 필요한 레이어에 액세스 한 다음 해당 레이어의 가중치와 바이어스에 액세스합니다.처음부터 구현에서와 마찬가지로 추정 된 매개 변수는 기초 진실에 가깝습니다.

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

## 요약

:begin_tab:`mxnet`
* Gluon을 사용하면 모델을 훨씬 간결하게 구현할 수 있습니다.
* Gluon에서 `data` 모듈은 데이터 처리를위한 도구를 제공하고 `nn` 모듈은 많은 수의 신경망 계층을 정의하며 `loss` 모듈은 많은 일반적인 손실 기능을 정의합니다.
* MXNet의 모듈 `initializer`는 모델 매개변수 초기화를 위한 다양한 방법을 제공합니다.
* 치수 및 저장은 자동으로 추론되지만 초기화하기 전에 매개 변수에 액세스하지 않도록 주의하십시오.
:end_tab:

:begin_tab:`pytorch`
* PyTorch의 고급 API를 사용하여 모델을 훨씬 간결하게 구현할 수 있습니다.
* PyTorch에서 `data` 모듈은 데이터 처리를위한 도구를 제공하며 `nn` 모듈은 많은 수의 신경망 계층과 공통 손실 기능을 정의합니다.
* 우리는 `_`로 끝나는 방법으로 값을 대체하여 매개 변수를 초기화 할 수 있습니다.
:end_tab:

:begin_tab:`tensorflow`
* TensorFlow의 고급 API를 사용하여 모델을 훨씬 간결하게 구현할 수 있습니다.
* TensorFlow에서 `data` 모듈은 데이터 처리를위한 도구를 제공하며 `keras` 모듈은 많은 수의 신경망 계층과 공통 손실 기능을 정의합니다.
* 텐서플로우의 모듈 `initializers`는 모델 파라미터 초기화를 위한 다양한 방법을 제공합니다.
* 치수 및 저장은 자동으로 추론됩니다 (그러나 매개 변수가 초기화되기 전에 매개 변수에 액세스하지 않도록주의하십시오).
:end_tab:

## 연습 문제

:begin_tab:`mxnet`
1. 우리가 `l = loss(output, y).mean()`으로 `l = loss(output, y)`를 대체하는 경우, 우리는 코드가 동일하게 동작하기 위해 `trainer.step(1)`로 `trainer.step(batch_size)`를 변경해야합니다.왜?
1. MXNet 설명서를 검토하여 모듈 `gluon.loss` 및 `init`에서 제공되는 손실 함수 및 초기화 방법을 확인하십시오.후버의 손실로 손실을 대체하십시오.
1. `dense.weight`의 그라디언트에 어떻게 액세스합니까?

[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
1. `nn.MSELoss(reduction='sum')`를 `nn.MSELoss()`로 대체하면 코드의 학습 속도를 어떻게 변경할 수 있습니까?왜?
1. PyTorch 설명서를 검토하여 제공되는 손실 함수와 초기화 방법을 확인하십시오.후버의 손실로 손실을 대체하십시오.
1. `net[0].weight`의 그라디언트에 어떻게 액세스합니까?

[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
1. TensorFlow 설명서를 검토하여 제공되는 손실 함수 및 초기화 방법을 확인하십시오.후버의 손실로 손실을 대체하십시오.

[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:
