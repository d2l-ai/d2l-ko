# 시퀀스 모델
:label:`sec_sequence`

Netflix에서 영화를 보고 있다고 가정해 보십시오.훌륭한 Netflix 사용자는 각 영화를 종교적으로 평가하기로 결정했습니다.결국 좋은 영화는 좋은 영화이고 더 많은 영화를보고 싶은데, 그렇죠?결과적으로 상황이 그렇게 간단하지는 않습니다.영화에 대한 사람들의 의견은 시간이 지남에 따라 상당히 바뀔 수 있습니다.실제로 심리학자들은 몇 가지 효과에 대한 이름도 가지고 있습니다. 

* 다른 사람의 의견에 따라*앵커링*이 있습니다.예를 들어, 오스카 상을 수상한 후에도 여전히 같은 영화이지만 해당 영화의 평점은 올라갑니다.이 효과는 상을 잊을 때까지 몇 달 동안 지속됩니다.효과가 등급을 반 포인트 이상 올리는 것으로 나타났습니다.
:cite:`Wu.Ahmed.Beutel.ea.2017`.
* * 쾌락 적응*이 있는데, 인간은 개선되거나 악화 된 상황을 새로운 정상으로 받아들이기 위해 빠르게 적응합니다.예를 들어, 좋은 영화를 많이 본 후 다음 영화가 똑같이 좋거나 더 좋다는 기대가 높습니다.따라서 많은 훌륭한 영화를 본 후에는 평균적인 영화조차도 나쁜 것으로 간주 될 수 있습니다.
* *계절성*이 있습니다.8 월에 산타 클로스 영화를 보는 것을 좋아하는 시청자는 거의 없습니다.
* 어떤 경우에는 제작에서 감독이나 배우의 잘못된 행동으로 인해 영화가 인기가 없어집니다.
* 일부 영화는 거의 우스꽝스럽게 나빴기 때문에 컬트 영화가됩니다.*우주 공간*의 계획 9와*트롤 2*는 이러한 이유로 높은 명성을 얻었습니다.

요컨대, 영화 평점은 고정되어 있지 않습니다.따라서 시간 역학을 사용하면 더 정확한 영화 권장 사항 :cite:`Koren.2009`를 얻을 수 있습니다.물론 시퀀스 데이터는 단순히 영화 등급에 관한 것이 아닙니다.다음은 더 많은 그림을 제공합니다. 

* 많은 사용자가 앱을 열 때 매우 특별한 행동을합니다.예를 들어, 소셜 미디어 앱은 방과 후 학생들에게 훨씬 더 인기가 있습니다.주식 시장 거래 앱은 시장이 열릴 때 더 일반적으로 사용됩니다.
* 어제 놓친 주가에 대한 빈칸을 채우는 것보다 내일 주가를 예측하는 것이 훨씬 어렵습니다. 둘 다 하나의 숫자를 추정하는 문제 일뿐입니다.결국 선견지명은 뒤늦은 시각보다 훨씬 어렵습니다.통계에서 전자 (알려진 관측치를 넘어서는 예측) 를*외삽*이라고하는 반면, 후자 (기존 관측치 간의 추정) 는*보간*이라고합니다.
* 음악, 연설, 텍스트 및 비디오는 모두 순차적입니다.우리가 그것들을 순회한다면 그들은 거의 말이되지 않을 것입니다.* 개가 남자를 물기*라는 제목은 단어가 동일하지만*남자가 개를 물는*보다 훨씬 덜 놀랍습니다.
* 지진은 밀접한 상관 관계가 있습니다. 즉, 대규모 지진 후에는 강한 지진이없는 것보다 훨씬 더 작은 여진이 여러 번 발생할 가능성이 큽니다.실제로 지진은 시공간적으로 상관 관계가 있습니다. 즉, 여진은 일반적으로 짧은 시간 내에 가까운 거리에서 발생합니다.
* 인간은 트위터 싸움, 춤 패턴 및 토론에서 볼 수 있듯이 순차적으로 서로 상호 작용합니다.

## 통계 도구

시퀀스 데이터를 처리하려면 통계 도구와 새로운 심층 신경망 아키텍처가 필요합니다.단순하게 유지하기 위해 :numref:`fig_ftse100`에 표시된 주가 (FTSE 100 지수) 를 예로 사용합니다. 

![FTSE 100 index over about 30 years.](../img/ftse100.png)
:width:`400px`
:label:`fig_ftse100`

가격을 $x_t$로 표시해 보겠습니다. 즉, *시간 단계* $t \in \mathbb{Z}^+$에서 가격 $x_t$를 준수합니다.이 텍스트의 시퀀스의 경우 $t$는 일반적으로 불연속형이며 정수 또는 해당 하위 집합에 따라 다릅니다.$t$일에 주식 시장에서 잘하고 싶은 거래자가 다음을 통해 $x_t$를 예측한다고 가정합니다. 

$$x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1).$$

### 자기 회귀 모델

이를 달성하기 위해 트레이더는 :numref:`sec_linear_concise`에서 훈련한 것과 같은 회귀 모델을 사용할 수 있습니다.한 가지 중요한 문제가 있습니다. 입력 수 $x_{t-1}, \ldots, x_1$은 $t$에 따라 다릅니다.즉, 우리가 접하는 데이터의 양에 따라 숫자가 증가하므로 계산적으로 다루기 쉽도록 근사치가 필요합니다.이 장의 다음 내용 중 대부분은 $P(x_t \mid x_{t-1}, \ldots, x_1)$을 효율적으로 추정하는 방법을 중심으로 진행됩니다.간단히 말해서 다음과 같은 두 가지 전략으로 요약됩니다. 

먼저, 잠재적으로 다소 긴 시퀀스 $x_{t-1}, \ldots, x_1$이 실제로 필요하지 않다고 가정합니다.이 경우 길이 $\tau$의 일부 시간 범위로 만족할 수 있으며 $x_{t-1}, \ldots, x_{t-\tau}$ 관측치만 사용할 수 있습니다.즉각적인 이점은 이제 인수 수가 적어도 $t > \tau$의 경우 항상 동일하다는 것입니다.이를 통해 위에서 설명한 대로 심층 네트워크를 훈련시킬 수 있습니다.이러한 모델은 말 그대로 스스로 회귀를 수행하기 때문에*자기 회귀 모델*이라고 합니다. 

:numref:`fig_sequence-model`에 표시된 두 번째 전략은 과거 관측치에 대한 요약 $h_t$을 유지하는 동시에 예측 $\hat{x}_t$ 외에도 $h_t$을 업데이트하는 것입니다.이로 인해 $\hat{x}_t = P(x_t \mid h_{t})$로 $x_t$를 추정하는 모델과 양식 $h_t = g(h_{t-1}, x_{t-1})$의 업데이트가 발생합니다.$h_t$은 관측되지 않기 때문에 이러한 모형을*잠재 자기회귀 모형*이라고도 합니다. 

![A latent autoregressive model.](../img/sequence-model.svg)
:label:`fig_sequence-model`

두 경우 모두 훈련 데이터를 생성하는 방법에 대한 분명한 의문을 제기합니다.일반적으로 과거 관측치를 사용하여 지금까지의 관측치가 주어진 다음 관측치를 예측합니다.분명히 우리는 시간이 가만히 서있을 것이라고 기대하지 않습니다.그러나 $x_t$의 특정 값은 변경될 수 있지만 적어도 시퀀스 자체의 동역학은 변경되지 않는다는 일반적인 가정입니다.새로운 역학은 참신하고 지금까지 가지고있는 데이터를 사용하여 예측할 수 없기 때문에 합리적입니다.통계학자는*고정적*을 변경하지 않는 역학을 호출합니다.우리가하는 일에 관계없이 다음을 통해 전체 시퀀스의 추정치를 얻을 수 있습니다. 

$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{t-1}, \ldots, x_1).$$

연속 숫자가 아닌 단어와 같은 개별 객체를 다루는 경우에도 위의 고려 사항이 여전히 유지됩니다.유일한 차이점은 이러한 상황에서는 회귀 모델이 아닌 분류기를 사용하여 $P(x_t \mid  x_{t-1}, \ldots, x_1)$를 추정해야 한다는 것입니다. 

### 마르코프 모델

자기 회귀 모형에서는 $x_t$를 추정하기 위해 $x_{t-1}, \ldots, x_1$ 대신 $x_{t-1}, \ldots, x_{t-\tau}$만 사용한다는 근사를 상기하십시오.이 근사치가 정확할 때마다 시퀀스가*마르코프 조건*을 충족한다고 말합니다.특히 $\tau = 1$인 경우*1차 마르코프 모델*이 있고 $P(x)$은 다음과 같이 지정됩니다. 

$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{t-1}) \text{ where } P(x_1 \mid x_0) = P(x_1).$$

이러한 모델은 $x_t$가 이산 값만 가정할 때마다 특히 유용합니다. 이 경우 동적 프로그래밍을 사용하여 체인을 따라 값을 정확하게 계산할 수 있기 때문입니다.예를 들어 $P(x_{t+1} \mid x_{t-1})$를 효율적으로 계산할 수 있습니다. 

$$\begin{aligned}
P(x_{t+1} \mid x_{t-1})
&= \frac{\sum_{x_t} P(x_{t+1}, x_t, x_{t-1})}{P(x_{t-1})}\\
&= \frac{\sum_{x_t} P(x_{t+1} \mid x_t, x_{t-1}) P(x_t, x_{t-1})}{P(x_{t-1})}\\
&= \sum_{x_t} P(x_{t+1} \mid x_t) P(x_t \mid x_{t-1})
\end{aligned}
$$

과거 관찰의 매우 짧은 역사 인 $P(x_{t+1} \mid x_t, x_{t-1}) = P(x_{t+1} \mid x_t)$ 만 고려하면된다는 사실을 사용함으로써.동적 프로그래밍에 대한 자세한 내용은 이 섹션의 범위를 벗어납니다.제어 및 강화 학습 알고리즘은 이러한 도구를 광범위하게 사용합니다. 

### 인과관계

원칙적으로 $P(x_1, \ldots, x_T)$를 역순으로 펼치는 것은 잘못된 것이 아닙니다.결국 컨디셔닝을 통해 언제든지 쓸 수 있습니다. 

$$P(x_1, \ldots, x_T) = \prod_{t=T}^1 P(x_t \mid x_{t+1}, \ldots, x_T).$$

실제로 마르코프 모델이 있다면 역 조건부 확률 분포도 얻을 수 있습니다.그러나 대부분의 경우 데이터에 대한 자연스러운 방향, 즉 시간이 지남에 따라 진행됩니다.미래의 사건이 과거에 영향을 미칠 수 없다는 것은 분명합니다.따라서 $x_t$을 변경하면 앞으로 $x_{t+1}$에 일어나는 일에 영향을 줄 수 있지만 그 반대의 경우에는 영향을 미치지 않을 수 있습니다.즉, $x_t$을 변경해도 과거 이벤트에 대한 분포는 변경되지 않습니다.따라서 $P(x_t \mid x_{t+1})$보다 $P(x_{t+1} \mid x_t)$을 설명하는 것이 더 쉬워야 합니다.예를 들어, 어떤 경우에는 일부 가산 잡음 $\epsilon$에 대해 $x_{t+1} = f(x_t) + \epsilon$을 찾을 수 있지만 그 반대는 사실 :cite:`Hoyer.Janzing.Mooij.ea.2009`가 아닌 것으로 나타났습니다.이것은 일반적으로 우리가 추정하는 데 관심이있는 전진 방향이기 때문에 좋은 소식입니다.Peters et al. 의 책은이 주제 :cite:`Peters.Janzing.Scholkopf.2017`에 대해 더 자세히 설명했습니다.우리는 그 표면을 간신히 긁고 있습니다. 

## 트레이닝

많은 통계 도구를 검토한 후 실제로 시도해 보겠습니다.먼저 몇 가지 데이터를 생성합니다.단순하게 유지하기 위해 (**시간 단계 $1, 2, \ldots, 1000$.에 약간의 가산 잡음이 있는 사인 함수를 사용하여 시퀀스 데이터를 생성합니다**)

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon, init
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab mxnet, pytorch
T = 1000  # Generate a total of 1000 points
time = d2l.arange(1, T + 1, dtype=d2l.float32)
x = d2l.sin(0.01 * time) + d2l.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

```{.python .input}
#@tab tensorflow
T = 1000  # Generate a total of 1000 points
time = d2l.arange(1, T + 1, dtype=d2l.float32)
x = d2l.sin(0.01 * time) + d2l.normal([T], 0, 0.2)
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

다음으로 이러한 시퀀스를 모델이 학습할 수 있는 기능과 레이블로 변환해야 합니다.임베딩 차원 $\tau$를 기반으로 [**데이터를 $y_t = x_t$ 및 $\mathbf{x}_t = [x_{t-\tau}, \ldots, x_{t-1}]$.** 쌍으로 매핑**] 기민한 독자는 첫 번째 $\tau$에 대한 충분한 기록이 없기 때문에 $\tau$ 더 적은 데이터 예제를 제공한다는 것을 알았을 것입니다.특히 시퀀스가 긴 경우 간단한 수정은 이러한 몇 가지 용어를 버리는 것입니다.또는 시퀀스를 0으로 채울 수도 있습니다.여기서는 처음 600개의 기능-레이블 쌍만 훈련에 사용합니다.

```{.python .input}
#@tab mxnet, pytorch
tau = 4
features = d2l.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = d2l.reshape(x[tau:], (-1, 1))
```

```{.python .input}
#@tab tensorflow
tau = 4
features = tf.Variable(d2l.zeros((T - tau, tau)))
for i in range(tau):
    features[:, i].assign(x[i: T - tau + i])
labels = d2l.reshape(x[tau:], (-1, 1))
```

```{.python .input}
#@tab all
batch_size, n_train = 16, 600
# Only the first `n_train` examples are used for training
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)
```

여기서는 완전히 연결된 두 개의 계층, ReLU 활성화 및 제곱 손실을 가진 [**아키텍처를 매우 단순하게 유지합니다: 단지 MLP**].

```{.python .input}
# A simple MLP
def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(10, activation='relu'),
            nn.Dense(1))
    net.initialize(init.Xavier())
    return net

# Square loss
loss = gluon.loss.L2Loss()
```

```{.python .input}
#@tab pytorch
# Function for initializing the weights of the network
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# A simple MLP
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# Note: `MSELoss` computes squared error without the 1/2 factor
loss = nn.MSELoss(reduction='none')
```

```{.python .input}
#@tab tensorflow
# Vanilla MLP architecture
def get_net():
    net = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu'),
                              tf.keras.layers.Dense(1)])
    return net

# Note: `MeanSquaredError` computes squared error without the 1/2 factor
loss = tf.keras.losses.MeanSquaredError()
```

이제 [**모델 훈련**] 을 할 준비가 되었습니다.아래 코드는 :numref:`sec_linear_concise`와 같은 이전 섹션의 훈련 루프와 본질적으로 동일합니다.따라서 우리는 자세히 설명하지 않을 것입니다.

```{.python .input}
def train(net, train_iter, loss, epochs, lr):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    for epoch in range(epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
```

```{.python .input}
#@tab tensorflow
def train(net, train_iter, loss, epochs, lr):
    trainer = tf.keras.optimizers.Adam()
    for epoch in range(epochs):
        for X, y in train_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            trainer.apply_gradients(zip(grads, params))
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
```

## 예측

훈련 손실이 적기 때문에 모델이 잘 작동 할 것으로 예상됩니다.이것이 실제로 무엇을 의미하는지 봅시다.가장 먼저 확인해야 할 것은 모델이 [**다음 단계에서 발생하는 상황을 예측**], 즉*한 단계 미리 예측*할 수 있는지입니다.

```{.python .input}
#@tab all
onestep_preds = net(features)
d2l.plot([time, time[tau:]], [d2l.numpy(x), d2l.numpy(onestep_preds)], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))
```

예상대로 한 단계 앞선 예측이 멋지게 보입니다.604 (`n_train + tau`) 관측치를 초과하더라도 예측은 여전히 신뢰할 수 있는 것으로 보입니다.그러나 여기에는 한 가지 작은 문제가 있습니다. 시간 단계 604까지만 시퀀스 데이터를 관찰하면 미래의 모든 한 단계 사전 예측에 대한 입력을 받기를 바랄 수 없습니다.대신 우리는 한 번에 한 단계씩 앞으로 나아가야 합니다. 

$$
\hat{x}_{605} = f(x_{601}, x_{602}, x_{603}, x_{604}), \\
\hat{x}_{606} = f(x_{602}, x_{603}, x_{604}, \hat{x}_{605}), \\
\hat{x}_{607} = f(x_{603}, x_{604}, \hat{x}_{605}, \hat{x}_{606}),\\
\hat{x}_{608} = f(x_{604}, \hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}),\\
\hat{x}_{609} = f(\hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}, \hat{x}_{608}),\\
\ldots
$$

일반적으로 최대 $x_t$까지의 관측된 시퀀스의 경우 시간 단계 $t+k$에서 예측된 출력 $\hat{x}_{t+k}$을 $k$*-단계별 예측*이라고 합니다.최대 $x_{604}$를 관찰했기 때문에 $k$단계 앞선 예측은 $\hat{x}_{604+k}$입니다.즉, [**자체 예측을 사용하여 다단계 예측을 수행**] 해야 합니다.이것이 얼마나 잘 진행되는지 봅시다.

```{.python .input}
#@tab mxnet, pytorch
multistep_preds = d2l.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        d2l.reshape(multistep_preds[i - tau: i], (1, -1)))
```

```{.python .input}
#@tab tensorflow
multistep_preds = tf.Variable(d2l.zeros(T))
multistep_preds[:n_train + tau].assign(x[:n_train + tau])
for i in range(n_train + tau, T):
    multistep_preds[i].assign(d2l.reshape(net(
        d2l.reshape(multistep_preds[i - tau: i], (1, -1))), ()))
```

```{.python .input}
#@tab all
d2l.plot([time, time[tau:], time[n_train + tau:]],
         [d2l.numpy(x), d2l.numpy(onestep_preds),
          d2l.numpy(multistep_preds[n_train + tau:])], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
```

위의 예에서 볼 수 있듯이 이것은 놀라운 실패입니다.예측은 몇 가지 예측 단계를 거친 후 매우 빠르게 상수로 감소합니다.알고리즘이 제대로 작동하지 않는 이유는 무엇입니까?이는 궁극적으로 오류가 쌓이기 때문입니다.1 단계 후에 $\epsilon_1 = \bar\epsilon$ 오류가 발생했다고 가정 해 보겠습니다.이제 2단계에 대한*입력*이 $\epsilon_1$에 의해 교란되므로 일부 상수 $c$에 대해 $\epsilon_2 = \bar\epsilon + c \epsilon_1$ 순서로 약간의 오류가 발생합니다.오차는 실제 관측치와 다소 빠르게 갈라질 수 있습니다.이것은 일반적인 현상입니다.예를 들어, 향후 24시간 동안의 일기 예보는 매우 정확한 경향이 있지만 그 이상으로 정확도는 급격히 떨어집니다.이 장 전체와 그 이후를 통해 이를 개선하는 방법에 대해 논의할 것입니다. 

$k = 1, 4, 16, 64$에 대한 전체 시퀀스에 대한 예측을 계산하여 [**$k$단계 사전 예측의 어려움을 자세히 살펴보십시오**].

```{.python .input}
#@tab all
max_steps = 64
```

```{.python .input}
#@tab mxnet, pytorch
features = d2l.zeros((T - tau - max_steps + 1, tau + max_steps))
# Column `i` (`i` < `tau`) are observations from `x` for time steps from
# `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# Column `i` (`i` >= `tau`) are the (`i - tau + 1`)-step-ahead predictions for
# time steps from `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau, tau + max_steps):
    features[:, i] = d2l.reshape(net(features[:, i - tau: i]), -1)
```

```{.python .input}
#@tab tensorflow
features = tf.Variable(d2l.zeros((T - tau - max_steps + 1, tau + max_steps)))
# Column `i` (`i` < `tau`) are observations from `x` for time steps from
# `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau):
    features[:, i].assign(x[i: i + T - tau - max_steps + 1].numpy())

# Column `i` (`i` >= `tau`) are the (`i - tau + 1`)-step-ahead predictions for
# time steps from `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau, tau + max_steps):
    features[:, i].assign(d2l.reshape(net((features[:, i - tau: i])), -1))
```

```{.python .input}
#@tab all
steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [d2l.numpy(features[:, tau + i - 1]) for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
```

이는 미래를 더 예측하려고 할 때 예측의 품질이 어떻게 변하는지를 명확하게 보여줍니다.4단계 사전 예측은 여전히 좋아 보이지만 그 이상의 것은 거의 쓸모가 없습니다. 

## 요약

* 보간과 외삽의 난이도에는 상당한 차이가 있습니다.따라서 시퀀스가 있는 경우 훈련할 때 항상 데이터의 시간적 순서를 존중하십시오. 즉, 향후 데이터에 대해 훈련하지 마십시오.
* 시퀀스 모델은 추정을 위한 특수 통계 도구가 필요합니다.널리 사용되는 두 가지 선택은 자기 회귀 모형과 잠재 변수 자기 회귀 모형입니다.
* 인과 관계 모델 (예: 앞으로의 시간) 의 경우 정방향을 추정하는 것이 일반적으로 반대 방향보다 훨씬 쉽습니다.
* 시간 스텝 $t$까지 관측된 시퀀스의 경우 시간 스텝 $t+k$에서 예측된 출력은 $k$*-스텝-어헤드 예측*입니다.$k$을 늘려 시간이 지남에 따라 더 예측하면 오류가 누적되고 예측 품질이 크게 저하되는 경우가 많습니다.

## 연습문제

1. 이 섹션의 실험에서 모델을 개선합니다.
    1. 지난 4개 이상의 관측치를 포함하시겠습니까?정말 몇 명이 필요한가요?
    1. 잡음이 없다면 과거 관측치가 얼마나 필요할까요?힌트: $\sin$ 및 $\cos$를 미분 방정식으로 쓸 수 있습니다.
    1. 전체 특징 수를 일정하게 유지하면서 이전 관측치를 통합할 수 있습니까?이렇게 하면 정확도가 향상됩니까?왜요?
    1. 신경망 아키텍처를 변경하고 성능을 평가합니다.
1. 투자자는 구매할 좋은 증권을 찾고 싶어합니다.그는 과거의 수익을 살펴보고 어느 것이 잘 될지 결정합니다.이 전략에서 무엇이 잘못 될 수 있을까요?
1. 인과관계가 텍스트에도 적용되나요?어느 정도까지?
1. 데이터의 동적을 캡처하기 위해 잠재 자기 회귀 모델이 필요할 수 있는 경우에 대한 예를 제공합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/113)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/114)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1048)
:end_tab:
