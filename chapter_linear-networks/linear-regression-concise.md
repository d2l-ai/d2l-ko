# 선형 회귀의 간결한 구현
:label:`sec_linear_concise`

지난 몇 년 동안 딥 러닝에 대한 광범위하고 강렬한 관심은 기업, 학계 및 애호가에게 그라디언트 기반 학습 알고리즘을 구현하는 반복적 인 작업을 자동화하기 위해 다양한 성숙한 오픈 소스 프레임 워크를 개발하도록 영감을주었습니다.:numref:`sec_linear_scratch`에서는 (i) 데이터 저장 및 선형 대수에 대한 텐서와 (ii) 기울기 계산을 위한 자동 미분에만 의존했습니다.실제로는 데이터 반복기, 손실 함수, 옵티마이저 및 신경망 계층이 매우 일반적이기 때문에 최신 라이브러리도 이러한 구성 요소를 구현합니다. 

이 섹션에서는 딥 러닝 프레임워크의 :numref:`sec_linear_scratch` (**상위 수준 API를 사용하여 간결하게**) 의 (**선형 회귀 모델을 구현하는 방법**) 을 보여줍니다. 

## 데이터세트 생성

시작하기 위해 :numref:`sec_linear_scratch`에서와 동일한 데이터 세트를 생성합니다.

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

## 데이터세트 읽기

자체 이터레이터를 롤링하는 대신 [**프레임워크의 기존 API를 호출하여 데이터를 읽을 수 있습니다.**] `features` 및 `labels`을 인수로 전달하고 데이터 이터레이터 객체를 인스턴스화할 때 `batch_size`를 지정합니다.또한 부울 값 `is_train`은 데이터 반복기 객체가 각 epoch의 데이터를 섞을 것인지 (데이터 세트를 통과) 할지 여부를 나타냅니다.

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

이제 `data_iter`를 :numref:`sec_linear_scratch`에서 `data_iter` 함수라고 부르는 것과 거의 같은 방식으로 사용할 수 있습니다.작동하는지 확인하기 위해 예제의 첫 번째 미니 배치를 읽고 인쇄 할 수 있습니다.:numref:`sec_linear_scratch`와 비교해 보면, 여기서는 `iter`을 사용하여 파이썬 이터레이터를 만들고 `next`을 사용하여 이터레이터에서 첫 번째 항목을 얻습니다.

```{.python .input}
#@tab all
next(iter(data_iter))
```

## 모델 정의

:numref:`sec_linear_scratch`에서 선형 회귀를 처음부터 구현했을 때 모델 매개 변수를 명시적으로 정의하고 계산을 코딩하여 기본 선형 대수 연산을 사용하여 출력을 생성했습니다.이 작업을 수행하는 방법을 알아야 합니다.그러나 모델이 더 복잡해지고 거의 매일 이 작업을 수행해야 하면 기꺼이 도움을 받을 수 있습니다.상황은 자신의 블로그를 처음부터 코딩하는 것과 비슷합니다.한두 번하는 것은 보람 있고 유익하지만 블로그가 필요할 때마다 한 달 동안 바퀴를 재발 명한다면 형편없는 웹 개발자가 될 것입니다. 

표준 작업의 경우 [**프레임워크의 사전 정의된 계층을 사용하여**] 를 사용하면 구현에 집중하지 않고 모델을 구성하는 데 사용되는 계층에 특히 집중할 수 있습니다.먼저 `Sequential` 클래스의 인스턴스를 참조하는 모델 변수 `net`를 정의합니다.`Sequential` 클래스는 함께 연결될 여러 레이어에 대한 컨테이너를 정의합니다.입력 데이터가 주어지면 `Sequential` 인스턴스가 첫 번째 계층을 통과한 다음 출력을 두 번째 계층의 입력으로 전달하는 식으로 계속됩니다.다음 예에서 모델은 하나의 레이어로만 구성되어 있으므로 `Sequential`는 실제로 필요하지 않습니다.그러나 향후 거의 모든 모델에는 여러 레이어가 포함되므로 가장 표준적인 워크 플로에 익숙해지기 위해 어쨌든 사용할 것입니다. 

:numref:`fig_single_neuron`에 나와 있는 것처럼 단일 계층 네트워크의 아키텍처를 생각해 보십시오.계층은 각 입력값이 행렬-벡터 곱셈을 통해 각 출력에 연결되기 때문에*완전히 연결된*이라고 합니다.

:begin_tab:`mxnet`
글루온에서는 완전 연결 계층이 `Dense` 클래스로 정의됩니다.단일 스칼라 출력값만 생성하려고 하기 때문에 이 숫자를 1로 설정합니다. 

편의를 위해 Gluon은 각 레이어의 입력 모양을 지정할 필요가 없습니다.따라서 여기서는 이 선형 계층에 얼마나 많은 입력값이 들어가는지 Gluon에게 말할 필요가 없습니다.모델을 통해 데이터를 처음 전달하려고 할 때 (예: 나중에 `net(X)`를 실행할 때) Gluon은 각 레이어에 대한 입력 수를 자동으로 추론합니다.나중에 어떻게 작동하는지 자세히 설명하겠습니다.
:end_tab:

:begin_tab:`pytorch`
파이토치에서 완전 연결 계층은 `Linear` 클래스로 정의됩니다.`nn.Linear`에 두 개의 인수를 전달했습니다.첫 번째 차원은 입력 피쳐 차원 (2) 을 지정하고, 두 번째 차원은 단일 스칼라이므로 1인 출력 피쳐 차원입니다.
:end_tab:

:begin_tab:`tensorflow`
케라스에서 완전 연결 계층은 `Dense` 클래스로 정의됩니다.단일 스칼라 출력값만 생성하려고 하기 때문에 이 숫자를 1로 설정합니다. 

편의상 Keras는 각 레이어의 입력 모양을 지정할 필요가 없습니다.따라서 여기서는 이 선형 계층에 얼마나 많은 입력이 입력되는지 Keras에게 말할 필요가 없습니다.모델을 통해 데이터를 처음 전달하려고 할 때 (예: 나중에 `net(X)`를 실행할 때) Keras는 각 레이어에 대한 입력 수를 자동으로 추론합니다.나중에 어떻게 작동하는지 자세히 설명하겠습니다.
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

`net`를 사용하기 전에 선형 회귀 모델의 가중치 및 치우침과 같은 (**모델 매개 변수를 초기화**) 해야 합니다.딥러닝 프레임워크에는 파라미터를 초기화하는 방법이 미리 정의되어 있는 경우가 많습니다.여기서 각 가중치 모수는 평균이 0이고 표준 편차가 0.01인 정규 분포에서 무작위로 샘플링되어야 함을 지정합니다.바이어스 파라미터는 0으로 초기화됩니다.

:begin_tab:`mxnet`
MXNet에서 `initializer` 모듈을 가져올 것입니다.이 모듈은 모델 매개 변수 초기화를 위한 다양한 방법을 제공합니다.글루온은 `init`를 `initializer` 패키지에 액세스하기 위한 바로 가기 (약어) 로 사용할 수 있도록 합니다.`init.Normal(sigma=0.01)`를 호출하여 가중치를 초기화하는 방법만 지정합니다.바이어스 매개변수는 기본적으로 0으로 초기화됩니다.
:end_tab:

:begin_tab:`pytorch`
`nn.Linear`를 구성할 때 입력 및 출력 차원을 지정했으므로 이제 매개 변수에 직접 액세스하여 초기 값을 지정할 수 있습니다.먼저 네트워크의 첫 번째 계층인 `net[0]`을 사용하여 계층을 찾은 다음 `weight.data` 및 `bias.data` 메서드를 사용하여 매개 변수에 액세스합니다.다음으로 대체 메서드 `normal_` 및 `fill_`을 사용하여 매개변수 값을 덮어씁니다.
:end_tab:

:begin_tab:`tensorflow`
텐서플로우의 `initializers` 모듈은 모델 파라미터 초기화를 위한 다양한 방법을 제공합니다.Keras에서 초기화 방법을 지정하는 가장 쉬운 방법은 `kernel_initializer`를 지정하여 레이어를 만드는 것입니다.여기서 `net`를 다시 다시 만듭니다.
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
위의 코드는 간단해 보일 수 있지만 여기서 이상한 일이 벌어지고 있다는 점에 유의해야 합니다.Gluon이 입력이 얼마나 많은 차원을 가질 지 아직 알지 못하더라도 네트워크에 대한 매개 변수를 초기화하고 있습니다!예와 같이 2이거나 2000일 수 있습니다.Gluon을 사용하면 장면 뒤에서 초기화가 실제로*지연*되기 때문에 이 문제를 해결할 수 있습니다.실제 초기화는 네트워크를 통해 데이터를 처음 전달하려고 할 때만 수행됩니다.매개 변수가 아직 초기화되지 않았으므로 액세스하거나 조작 할 수 없다는 점에 유의하십시오.
:end_tab:

:begin_tab:`pytorch`

:end_tab:

:begin_tab:`tensorflow`
위의 코드는 간단해 보일 수 있지만 여기서 이상한 일이 벌어지고 있다는 점에 유의해야 합니다.Keras가 입력의 차원 수를 아직 알지 못하더라도 네트워크에 대한 매개 변수를 초기화하고 있습니다!예와 같이 2이거나 2000일 수 있습니다.Keras를 사용하면 뒤에서 초기화가 실제로*지연*되기 때문에 이 문제를 해결할 수 있습니다.실제 초기화는 네트워크를 통해 데이터를 처음 전달하려고 할 때만 수행됩니다.매개 변수가 아직 초기화되지 않았으므로 액세스하거나 조작 할 수 없다는 점에 유의하십시오.
:end_tab:

## 손실 함수 정의하기

:begin_tab:`mxnet`
글루온에서는 `loss` 모듈이 다양한 손실 함수를 정의합니다.이 예에서는 제곱 손실 (`L2Loss`) 의 글루온 구현을 사용합니다.
:end_tab:

:begin_tab:`pytorch`
[**`MSELoss` 클래스는 평균 제곱 오차를 계산합니다 (:eqref:`eq_mse`에서는 $1/2$ 요인 제외) .**] 기본적으로 예에 대한 평균 손실을 반환합니다.
:end_tab:

:begin_tab:`tensorflow`
`MeanSquaredError` 클래스는 평균 제곱 오차를 계산합니다 (:eqref:`eq_mse`에서 $1/2$ 요인이 없는 경우).기본적으로 예제에 비해 평균 손실을 반환합니다.
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
미니배치 확률적 경사하강법은 신경망을 최적화하기 위한 표준 도구이므로 Gluon은 `Trainer` 클래스를 통해 이 알고리즘의 여러 변형과 함께 이를 지원합니다.`Trainer`을 인스턴스화할 때 최적화할 매개 변수 (`net.collect_params()`을 통해 모델 `net`에서 얻을 수 있음), 사용하려는 최적화 알고리즘 (`sgd`) 및 최적화 알고리즘에 필요한 하이퍼 매개 변수 사전을 지정합니다.미니배치 확률적 경사 하강법은 `learning_rate` 값을 설정하기만 하면 됩니다. 여기서는 0.03으로 설정됩니다.
:end_tab:

:begin_tab:`pytorch`
미니배치 확률적 경사하강법은 신경망을 최적화하기 위한 표준 도구이므로 PyTorch는 `optim` 모듈에서 이 알고리즘의 여러 변형과 함께 이를 지원합니다.(**`SGD` 인스턴스를 인스턴스화**) 하면 최적화 알고리즘에 필요한 하이퍼 파라미터 사전을 사용하여 최적화 할 매개 변수 (`net.parameters()`를 통해 인터넷에서 얻을 수 있음) 를 지정합니다.미니배치 확률적 경사 하강법은 `lr` 값을 설정하기만 하면 됩니다. 여기서는 0.03으로 설정됩니다.
:end_tab:

:begin_tab:`tensorflow`
Minibatch 확률적 경사하강법은 신경망을 최적화하기 위한 표준 도구이므로 Keras는 `optimizers` 모듈에서 이 알고리즘의 여러 변형과 함께 이를 지원합니다.미니배치 확률적 경사 하강법은 `learning_rate` 값을 설정하기만 하면 됩니다. 여기서는 0.03으로 설정됩니다.
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

## 트레이닝

딥 러닝 프레임워크의 상위 수준 API를 통해 모델을 표현하려면 비교적 몇 줄의 코드가 필요하다는 것을 눈치 챘을 것입니다.파라미터를 개별적으로 할당하거나, 손실 함수를 정의하거나, 미니배치 확률적 경사하강법을 구현할 필요가 없었습니다.훨씬 더 복잡한 모델로 작업을 시작하면 상위 수준 API의 이점이 상당히 커질 것입니다.그러나 일단 모든 기본 요소를 갖추면 [**훈련 루프 자체는 모든 것을 처음부터 구현할 때 했던 것과 매우 유사합니다.**] 

메모리를 새로 고치기 위해: 몇 가지 epoch에 대해 데이터 세트 (`train_data`) 를 완전히 통과하여 입력의 미니 배치 하나와 해당 지상 실측 레이블을 반복적으로 가져옵니다.각 미니 배치마다 다음 의식을 거칩니다. 

* `net(X)`를 호출하여 예측을 생성하고 손실 `l` (순방향 전파) 를 계산합니다.
* 역전파를 실행하여 그래디언트를 계산합니다.
* 옵티마이저를 호출하여 모델 파라미터를 업데이트합니다.

적절한 측정을 위해 각 에포크 이후의 손실을 계산하고 이를 인쇄하여 진행 상황을 모니터링합니다.

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

아래에서는 [**유한 데이터에 대한 교육을 통해 학습한 모델 파라미터와 데이터 세트를 생성한 실제 파라미터**] 를 비교합니다.파라미터에 액세스하려면 먼저 `net`에서 필요한 레이어에 액세스한 다음 해당 레이어의 가중치와 편향에 액세스합니다.처음부터 구현한 것처럼 추정된 매개 변수는 실제 매개 변수와 비슷하다는 점에 유의하십시오.

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
* Gluon을 사용하면 모델을 훨씬 더 간결하게 구현할 수 있습니다.
* Gluon에서 `data` 모듈은 데이터 처리를위한 도구를 제공하고 `nn` 모듈은 많은 수의 신경망 계층을 정의하며 `loss` 모듈은 많은 공통 손실 함수를 정의합니다.
* MXNet의 모듈 `initializer`는 모델 매개 변수 초기화를 위한 다양한 방법을 제공합니다.
* 차원 및 저장소는 자동으로 유추되지만 매개 변수가 초기화되기 전에 매개 변수에 액세스하려고 시도하지 않도록 주의하십시오.
:end_tab:

:begin_tab:`pytorch`
* PyTorch의 상위 수준 API를 사용하여 모델을 훨씬 더 간결하게 구현할 수 있습니다.
* 파이토치에서 `data` 모듈은 데이터 처리를 위한 도구를 제공하며, `nn` 모듈은 많은 수의 신경망 계층과 공통 손실 함수를 정의합니다.
* 매개 변수의 값을 `_`로 끝나는 메서드로 대체하여 매개 변수를 초기화 할 수 있습니다.
:end_tab:

:begin_tab:`tensorflow`
* TensorFlow의 고급 API를 사용하면 모델을 훨씬 더 간결하게 구현할 수 있습니다.
* TensorFlow에서 `data` 모듈은 데이터 처리를 위한 도구를 제공하며, `keras` 모듈은 많은 수의 신경망 계층과 공통 손실 함수를 정의합니다.
* 텐서플로우 모듈 `initializers`는 모델 파라미터 초기화를 위한 다양한 방법을 제공합니다.
* 차원 및 저장소는 자동으로 유추됩니다. 단, 매개변수가 초기화되기 전에 매개변수에 액세스하려고 시도하지 않도록 주의하십시오.
:end_tab:

## 연습문제

:begin_tab:`mxnet`
1. `l = loss(output, y)`를 `l = loss(output, y).mean()`로 대체하는 경우 코드가 동일하게 작동하도록 `trainer.step(batch_size)`를 `trainer.step(1)`으로 변경해야 합니다.왜요?
1. MXNet 설명서를 검토하여 모듈 `gluon.loss` 및 `init`에서 제공되는 손실 함수 및 초기화 방법을 확인하십시오.손실을 후버의 손실로 대체하십시오.
1. `dense.weight`의 그래디언트에 어떻게 액세스합니까?

[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
1. `nn.MSELoss(reduction='sum')`를 `nn.MSELoss()`로 대체하면 코드가 동일하게 작동하도록 학습률을 어떻게 변경할 수 있습니까?왜요?
1. PyTorch 문서를 검토하여 어떤 손실 함수와 초기화 메서드가 제공되는지 확인하십시오.손실을 후버의 손실로 대체하십시오.
1. `net[0].weight`의 그래디언트에 어떻게 액세스합니까?

[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
1. TensorFlow 문서를 검토하여 어떤 손실 함수와 초기화 메서드가 제공되는지 확인하세요.손실을 후버의 손실로 대체하십시오.

[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:
