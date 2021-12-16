# 생성적 적대 신경망
:label:`sec_basic_gan`

이 책의 대부분에서 우리는 예측을 하는 방법에 대해 이야기했습니다.어떤 형태로든 우리는 심층 신경망을 사용하여 데이터 예제에서 레이블로 매핑했습니다.이런 종류의 학습을 차별적 학습이라고 부릅니다. 우리는 사진 고양이와 개 사진을 구별할 수 있기를 원합니다.분류자와 회귀자는 모두 차별적 학습의 예입니다.역전파로 훈련된 신경망은 크고 복잡한 데이터세트에 대한 차별적 학습에 대해 우리가 알고 있다고 생각했던 모든 것을 뒤엎었습니다.고해상도 이미지의 분류 정확도는 불과 5~6년 만에 쓸모없는 것에서 인간 수준 (몇 가지 주의 사항 포함) 으로 바뀌 었습니다.심층 신경망이 놀랍도록 잘 작동하는 다른 모든 차별적 작업에 대해 또 다른 과장되게 떠벌 리도록 하겠습니다. 

그러나 기계 학습에는 차별적인 작업을 해결하는 것 이상의 것이 있습니다.예를 들어 레이블이 없는 대규모 데이터셋이 주어진 경우 이 데이터의 특성을 간결하게 캡처하는 모델을 학습할 수 있습니다.이러한 모델이 주어지면 훈련 데이터의 분포와 유사한 합성 데이터 예제를 샘플링할 수 있습니다.예를 들어, 얼굴 사진의 큰 뭉치가 주어지면 동일한 데이터셋에서 나온 것처럼 보이는 새로운 사실적 이미지를 생성할 수 있기를 원할 수 있습니다.이런 종류의 학습을 제너레이티브 모델링이라고 합니다. 

최근까지 우리는 새로운 사실적 이미지를 합성할 수 있는 방법이 없었습니다.그러나 차별적 학습을 위한 심층 신경망의 성공은 새로운 가능성을 열었습니다.지난 3년 동안의 한 가지 큰 추세는 우리가 일반적으로 감독 학습 문제로 생각하지 않는 문제의 문제를 극복하기 위해 차별적 딥 네트를 적용하는 것입니다.순환 신경망 언어 모델은 일단 훈련되면 생성 모델로 작동할 수 있는 차별적 네트워크 (다음 특성을 예측하도록 훈련됨) 를 사용하는 한 예입니다. 

2014년 획기적인 논문에서는 차별적 모델의 힘을 활용하여 우수한 생성 모델을 얻을 수 있는 영리하고 새로운 방법인 생성적 적대 네트워크 (GAN) :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`를 소개했습니다.GAN은 가짜 데이터를 실제 데이터와 구별할 수 없다면 데이터 생성기가 좋다는 생각에 의존합니다.통계에서는 이를 2-표본 검정 (데이터셋 $X=\{x_1,\ldots, x_n\}$ 및 $X'=\{x'_1,\ldots, x'_n\}$) 이 동일한 분포에서 추출되었는지 여부에 대한 질문에 답하기 위한 검정이라고 합니다.대부분의 통계 논문과 GAN의 주요 차이점은 후자가이 아이디어를 건설적인 방식으로 사용한다는 것입니다.즉, “이 두 데이터 세트는 동일한 분포에서 나온 것처럼 보이지 않습니다”라고 말하는 모델을 훈련시키는 대신 [two-sample test](https://en.wikipedia.org/wiki/Two-sample_hypothesis_testing)을 사용하여 생성 모델에 훈련 신호를 제공합니다.이를 통해 실제 데이터와 유사한 것을 생성할 때까지 데이터 생성기를 개선할 수 있습니다.최소한 분류자를 속일 필요가 있습니다.분류기가 최첨단 심층 신경망인 경우에도 마찬가지입니다. 

![Generative Adversarial Networks](../img/gan.svg)
:label:`fig_gan`

GAN 아키텍처는 :numref:`fig_gan`에 설명되어 있습니다.보시다시피 GAN 아키텍처에는 두 가지 요소가 있습니다. 먼저 실제 네트워크처럼 보이는 데이터를 생성 할 수있는 장치 (예: 심층 네트워크이지만 실제로는 게임 렌더링 엔진과 같은 모든 것이 될 수 있음) 가 필요합니다.이미지를 다루는 경우 이미지를 생성해야 합니다.음성을 다루는 경우 오디오 시퀀스를 생성하는 등의 작업이 필요합니다.우리는 이것을 발전기 네트워크라고 부릅니다.두 번째 구성 요소는 판별자 네트워크입니다.가짜 데이터와 실제 데이터를 서로 구별하려고 시도합니다.두 네트워크 모두 서로 경쟁하고 있습니다.생성기 네트워크는 판별자 네트워크를 속이려고 시도합니다.이 시점에서 판별자 네트워크는 새로운 가짜 데이터에 적응합니다.이 정보는 발전기 네트워크를 개선하는 데 사용됩니다. 

판별기는 입력 $x$가 실제 (실제 데이터에서) 인지 가짜 (생성기에서) 인지 구별하는 이진 분류기입니다.전형적으로, 판별기는 은닉 크기 1을 갖는 조밀 계층을 사용하는 것과 같이 입력 $\mathbf x$에 대한 스칼라 예측 $o\in\mathbb R$을 출력한 다음, 시그모이드 함수를 적용하여 예측된 확률 $D(\mathbf x) = 1/(1+e^{-o})$를 구한다.실제 데이터에 대한 레이블 $y$이 가짜 데이터에 대해 $1$이고 $0$이라고 가정합니다.교차 엔트로피 손실*i.e.*를 최소화하도록 판별자를 훈련시킵니다. 

$$ \min_D \{ - y \log D(\mathbf x) - (1-y)\log(1-D(\mathbf x)) \},$$

생성기의 경우 먼저 임의성 소스 (*예:*), 정규 분포 $\mathbf z \sim \mathcal{N} (0, 1)$에서 일부 모수 $\mathbf z\in\mathbb R^d$을 그립니다.우리는 종종 $\mathbf z$를 잠재 변수로 부릅니다.그런 다음 함수를 적용하여 $\mathbf x'=G(\mathbf z)$을 생성합니다.생성기의 목표는 판별자를 속여 $\mathbf x'=G(\mathbf z)$을 실제 데이터 (*즉*) 로 분류하는 것입니다. 우리는 $D( G(\mathbf z)) \approx 1$을 원합니다.즉, 주어진 판별자 $D$에 대해 $y=0$, *i.e.* 일 때 교차 엔트로피 손실을 최대화하기 위해 생성기 $G$의 매개 변수를 업데이트합니다. 

$$ \max_G \{ - (1-y) \log(1-D(G(\mathbf z))) \} = \max_G \{ - \log(1-D(G(\mathbf z))) \}.$$

생성기가 완벽한 작업을 수행하면 $D(\mathbf x')\approx 1$가 위의 손실이 0에 가까워지므로 그래디언트가 너무 작아 판별자를 잘 진행할 수 없습니다.일반적으로 다음과 같은 손실을 최소화합니다. 

$$ \min_G \{ - y \log(D(G(\mathbf z))) \} = \min_G \{ - \log(D(G(\mathbf z))) \}, $$

그것은 단지 $\mathbf x'=G(\mathbf z)$를 판별자에게 공급하지만 라벨 $y=1$를 부여하는 것입니다. 

요약하면 $D$와 $G$는 포괄적인 목적 함수를 사용하여 “미니맥스” 게임을 플레이하고 있습니다. 

$$min_D max_G \{ -E_{x \sim \text{Data}} log D(\mathbf x) - E_{z \sim \text{Noise}} log(1 - D(G(\mathbf z))) \}.$$

대부분의 GaN 애플리케이션은 이미지 컨텍스트에 있습니다.데모 목적으로, 우리는 먼저 훨씬 더 간단한 배포판을 맞추는 것에 만족할 것입니다.GAN을 사용하여 가우스에 대한 세계에서 가장 비효율적인 파라미터 추정기를 구축하면 어떤 일이 발생하는지 설명하겠습니다.시작해 보겠습니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
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
from d2l import tensorflow as d2l
import tensorflow as tf
```

## 일부 “실제” 데이터 생성

이것은 세계에서 가장 절름발이 될 것이므로 가우스에서 가져온 데이터를 생성하기만 하면 됩니다.

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0.0, 1, (1000, 2))
A = d2l.tensor([[1, 2], [-0.1, 0.5]])
b = d2l.tensor([1, 2])
data = d2l.matmul(X, A) + b
```

```{.python .input}
#@tab tensorflow
X = d2l.normal((1000, 2), 0.0, 1)
A = d2l.tensor([[1, 2], [-0.1, 0.5]])
b = d2l.tensor([1, 2], tf.float32)
data = d2l.matmul(X, A) + b
```

우리가 무엇을 얻었는지 보자.이것은 평균 $b$와 공분산 행렬 $A^TA$를 사용하여 다소 임의적인 방식으로 이동된 가우스 행렬이어야 합니다.

```{.python .input}
#@tab mxnet, pytorch
d2l.set_figsize()
d2l.plt.scatter(d2l.numpy(data[:100, 0]), d2l.numpy(data[:100, 1]));
print(f'The covariance matrix is\n{d2l.matmul(A.T, A)}')
```

```{.python .input}
#@tab tensorflow
d2l.set_figsize()
d2l.plt.scatter(d2l.numpy(data[:100, 0]), d2l.numpy(data[:100, 1]));
print(f'The covariance matrix is\n{tf.matmul(A, A, transpose_a=True)}')
```

```{.python .input}
#@tab all
batch_size = 8
data_iter = d2l.load_array((data,), batch_size)
```

## 제너레이터

우리의 발전기 네트워크는 가능한 가장 단순한 네트워크, 즉 단일 계층 선형 모델이 될 것입니다.가우스 데이터 생성기로 선형 네트워크를 구동할 것이기 때문입니다.따라서 말 그대로 사물을 완벽하게 위조하는 매개 변수 만 배우면됩니다.

```{.python .input}
net_G = nn.Sequential()
net_G.add(nn.Dense(2))
```

```{.python .input}
#@tab pytorch
net_G = nn.Sequential(nn.Linear(2, 2))
```

```{.python .input}
#@tab tensorflow
net_G = tf.keras.layers.Dense(2)
```

## 판별자

판별자에게는 좀 더 차별적 일 것입니다. 우리는 3 개의 레이어가있는 MLP를 사용하여 좀 더 흥미롭게 만들 것입니다.

```{.python .input}
net_D = nn.Sequential()
net_D.add(nn.Dense(5, activation='tanh'),
          nn.Dense(3, activation='tanh'),
          nn.Dense(1))
```

```{.python .input}
#@tab pytorch
net_D = nn.Sequential(
    nn.Linear(2, 5), nn.Tanh(),
    nn.Linear(5, 3), nn.Tanh(),
    nn.Linear(3, 1))
```

```{.python .input}
#@tab tensorflow
net_D = tf.keras.models.Sequential([
    tf.keras.layers.Dense(5, activation="tanh", input_shape=(2,)),
    tf.keras.layers.Dense(3, activation="tanh"),
    tf.keras.layers.Dense(1)
])
```

## 트레이닝

먼저 판별자를 업데이트하는 함수를 정의합니다.

```{.python .input}
#@save
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = np.ones((batch_size,), ctx=X.ctx)
    zeros = np.zeros((batch_size,), ctx=X.ctx)
    with autograd.record():
        real_Y = net_D(X)
        fake_X = net_G(Z)
        # Do not need to compute gradient for `net_G`, detach it from
        # computing gradients.
        fake_Y = net_D(fake_X.detach())
        loss_D = (loss(real_Y, ones) + loss(fake_Y, zeros)) / 2
    loss_D.backward()
    trainer_D.step(batch_size)
    return float(loss_D.sum())
```

```{.python .input}
#@tab pytorch
#@save
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    # Do not need to compute gradient for `net_G`, detach it from
    # computing gradients.
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss(real_Y, ones.reshape(real_Y.shape)) + 
              loss(fake_Y, zeros.reshape(fake_Y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D
```

```{.python .input}
#@tab tensorflow
#@save
def update_D(X, Z, net_D, net_G, loss, optimizer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = tf.ones((batch_size,)) # Labels corresponding to real data
    zeros = tf.zeros((batch_size,)) # Labels corresponding to fake data
    # Do not need to compute gradient for `net_G`, so it's outside GradientTape
    fake_X = net_G(Z)
    with tf.GradientTape() as tape:
        real_Y = net_D(X)
        fake_Y = net_D(fake_X)
        # We multiply the loss by batch_size to match PyTorch's BCEWithLogitsLoss
        loss_D = (loss(ones, tf.squeeze(real_Y)) + loss(
            zeros, tf.squeeze(fake_Y))) * batch_size / 2
    grads_D = tape.gradient(loss_D, net_D.trainable_variables)
    optimizer_D.apply_gradients(zip(grads_D, net_D.trainable_variables))
    return loss_D
```

생성기도 비슷하게 업데이트됩니다.여기서는 교차 엔트로피 손실을 재사용하지만 가짜 데이터의 레이블을 $0$에서 $1$로 변경합니다.

```{.python .input}
#@save
def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = np.ones((batch_size,), ctx=Z.ctx)
    with autograd.record():
        # We could reuse `fake_X` from `update_D` to save computation
        fake_X = net_G(Z)
        # Recomputing `fake_Y` is needed since `net_D` is changed
        fake_Y = net_D(fake_X)
        loss_G = loss(fake_Y, ones)
    loss_G.backward()
    trainer_G.step(batch_size)
    return float(loss_G.sum())
```

```{.python .input}
#@tab pytorch
#@save
def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)
    trainer_G.zero_grad()
    # We could reuse `fake_X` from `update_D` to save computation
    fake_X = net_G(Z)
    # Recomputing `fake_Y` is needed since `net_D` is changed
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G
```

```{.python .input}
#@tab tensorflow
#@save
def update_G(Z, net_D, net_G, loss, optimizer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = tf.ones((batch_size,))
    with tf.GradientTape() as tape:
        # We could reuse `fake_X` from `update_D` to save computation
        fake_X = net_G(Z)
        # Recomputing `fake_Y` is needed since `net_D` is changed
        fake_Y = net_D(fake_X)
        # We multiply the loss by batch_size to match PyTorch's BCEWithLogits loss
        loss_G = loss(ones, tf.squeeze(fake_Y)) * batch_size
    grads_G = tape.gradient(loss_G, net_G.trainable_variables)
    optimizer_G.apply_gradients(zip(grads_G, net_G.trainable_variables))
    return loss_G
```

판별자와 생성기는 모두 교차 엔트로피 손실로 이항 로지스틱 회귀를 수행합니다.우리는 훈련 과정을 원활하게하기 위해 Adam을 사용합니다.각 반복에서 먼저 판별자를 업데이트한 다음 생성기를 업데이트합니다.손실과 생성된 예를 모두 시각화합니다.

```{.python .input}
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = gluon.loss.SigmoidBCELoss()
    net_D.initialize(init=init.Normal(0.02), force_reinit=True)
    net_G.initialize(init=init.Normal(0.02), force_reinit=True)
    trainer_D = gluon.Trainer(net_D.collect_params(),
                              'adam', {'learning_rate': lr_D})
    trainer_G = gluon.Trainer(net_G.collect_params(),
                              'adam', {'learning_rate': lr_G})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X in data_iter:
            batch_size = X.shape[0]
            Z = np.random.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Visualize generated examples
        Z = np.random.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).asnumpy()
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real', 'generated'])
        # Show the losses
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

```{.python .input}
#@tab pytorch
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Visualize generated examples
        Z = torch.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).detach().numpy()
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real', 'generated'])
        # Show the losses
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

```{.python .input}
#@tab tensorflow
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    for w in net_D.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))
    for w in net_G.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))
    optimizer_D = tf.keras.optimizers.Adam(learning_rate=lr_D)
    optimizer_G = tf.keras.optimizers.Adam(learning_rate=lr_G)
    animator = d2l.Animator(
        xlabel="epoch", ylabel="loss", xlim=[1, num_epochs], nrows=2,
        figsize=(5, 5), legend=["discriminator", "generator"])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = tf.random.normal(
                mean=0, stddev=1, shape=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, optimizer_D),
                       update_G(Z, net_D, net_G, loss, optimizer_G),
                       batch_size)
        # Visualize generated examples
        Z = tf.random.normal(mean=0, stddev=1, shape=(100, latent_dim))
        fake_X = net_G(Z)
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(["real", "generated"])
        
        # Show the losses
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
        
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

이제 가우스 분포에 맞는 초모수를 지정합니다.

```{.python .input}
#@tab all
lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 20
train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G,
      latent_dim, d2l.numpy(data[:100]))
```

## 요약

* 생성적 적대 네트워크 (GAN) 는 생성기와 판별자라는 두 개의 심층 네트워크로 구성됩니다.
* 생성기는 교차 엔트로피 손실 (*i.e.*, $\max \log(D(\mathbf{x'}))$) 을 최대화함으로써 판별자를 속이기 위해 가능한 한 실제 이미지에 훨씬 더 가까운 이미지를 생성합니다.
* 판별기는 교차 엔트로피 손실*i.e.*, $\min - y \log D(\mathbf{x}) - (1-y)\log(1-D(\mathbf{x}))$를 최소화하여 생성된 이미지와 실제 이미지를 구별하려고 시도합니다.

## 연습문제

* 생성기가 승리하는 곳에 평형이 존재합니까? 즉, 판별기가 유한 표본에서 두 분포를 구분할 수 없게 됩니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/408)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1082)
:end_tab:
