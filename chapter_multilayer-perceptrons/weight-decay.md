# 웨이트 디케이
:label:`sec_weight_decay`

이제 과적합 문제를 특성화했으므로 모델 정규화를 위한 몇 가지 표준 기법을 소개할 수 있습니다.외출하고 더 많은 훈련 데이터를 수집하여 항상 과적합을 완화할 수 있습니다.이는 비용이 많이 들거나 시간이 많이 걸리거나 완전히 통제 할 수 없기 때문에 단기적으로는 불가능할 수 있습니다.현재로서는 리소스가 허용하는 만큼의 고품질 데이터가 이미 있다고 가정하고 정규화 기술에 집중할 수 있습니다. 

다항식 회귀 예제 (:numref:`sec_model_selection`) 에서 피팅된 다항식의 차수를 조정하기만 하면 모델의 용량을 제한할 수 있습니다.실제로 기능 수를 제한하는 것은 과적합을 완화하는 데 널리 사용되는 기술입니다.그러나 단순히 피처를 버리는 것은 작업에 너무 무딘 도구가 될 수 있습니다.다항식 회귀 예제를 고수하면서 고차원 입력에서 발생할 수 있는 일을 고려합니다.다변량 데이터에 대한 다항식의 자연스러운 확장을*단항식*이라고 하며, 이는 단순히 변수의 거듭제곱의 곱입니다.단항식의 차수는 거듭제곱의 합입니다.예를 들어, $x_1^2 x_2$과 $x_3 x_5^2$는 모두 차수가 3인 단항식입니다. 

$d$이 커짐에 따라 차수가 $d$인 항의 수는 빠르게 폭발합니다.$k$개의 변수가 주어지면, $d$도 (즉, $k$ 다중 선택 $d$) 의 단항식의 수는 ${k - 1 + d} \choose {k - 1}$입니다.$2$에서 $3$로 정도의 작은 변화라도 모델의 복잡성이 크게 증가합니다.따라서 함수 복잡성을 조정하기 위해 보다 세분화된 도구가 필요한 경우가 많습니다. 

## 규범과 체중 감소

우리는 $L_2$ 규범과 $L_1$ 규범을 모두 설명했으며, 이는 :numref:`subsec_lin-algebra-norms`에서 더 일반적인 $L_p$ 표준의 특수한 경우입니다.(***체중 감소* (일반적으로 $L_2$ 정규화라고 함) 는 매개 변수 기계 학습 모델을 정규화하는 데 가장 널리 사용되는 기술일 수 있습니다.**) 이 기술은 모든 함수 $f$ 중에서 함수 $f = 0$ (모든 입력에 값 $0$ 할당) 이라는 기본 직관에 의해 동기가 부여됩니다.는 어떤 의미에서*가장 단순한*이며 함수의 복잡도를 0으로부터의 거리로 측정 할 수 있습니다.하지만 함수와 0 사이의 거리를 얼마나 정확하게 측정해야 할까요?정답은 하나도 없습니다.사실, 기능 분석의 일부와 Banach 공간 이론을 포함한 수학의 전체 분야가이 문제에 답하는 데 전념하고 있습니다. 

한 가지 간단한 해석은 선형 함수 $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$의 복잡도를 가중치 벡터의 일부 노름 (예: $\| \mathbf{w} \|^2$) 으로 측정하는 것입니다.작은 가중치 벡터를 보장하는 가장 일반적인 방법은 손실을 최소화하는 문제에 해당 노름을 페널티 항으로 추가하는 것입니다.따라서 우리는 원래 목표를 대체합니다.
*훈련 레이블*의 예측 손실을 최소화합니다.
새로운 목표를 가지고
*예측 손실과 페널티 항*의 합을 최소화합니다.
이제 가중치 벡터가 너무 커지면 학습 알고리즘은 가중치 노름 $\| \mathbf{w} \|^2$를 최소화하는 것과 훈련 오류를 최소화하는 데 초점을 맞출 수 있습니다.이것이 바로 우리가 원하는 것입니다.코드의 내용을 설명하기 위해 선형 회귀에 대한 :numref:`sec_linear_regression`의 이전 예제를 다시 부활시켜 보겠습니다.그곳에서 우리의 손실은 

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

$\mathbf{x}^{(i)}$이 특징이고, $y^{(i)}$는 모든 데이터에 대한 레이블입니다 (예: $i$), $(\mathbf{w}, b)$는 각각 가중치 및 편향 매개 변수입니다.가중치 벡터의 크기에 페널티를 적용하려면 어떻게 든 손실 함수에 $\| \mathbf{w} \|^2$을 추가해야 합니다. 하지만 모델이 이 새로운 가산 페널티에 대한 표준 손실을 어떻게 교환해야 할까요?실제로 검증 데이터를 사용하여 피팅하는 음이 아닌 하이퍼 파라미터인*정규화 상수* $\lambda$를 통해 이러한 절충안을 특성화합니다. 

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2,$$

$\lambda = 0$의 경우 원래 손실 함수를 복구합니다.$\lambda > 0$의 경우 $\| \mathbf{w} \|$의 크기를 제한합니다.규칙에 따라 $2$로 나눕니다. 2 차 함수의 도함수를 취하면 $2$ 및 $1/2$가 취소되어 업데이트에 대한 표현식이 멋지고 단순 해 보입니다.기민한 독자는 왜 우리가 표준 노름 (즉, 유클리드 거리) 이 아닌 제곱 노름으로 작업하는지 궁금해 할 것입니다.계산 편의를 위해 이 작업을 수행합니다.$L_2$ 노름을 제곱하여 제곱근을 제거하고 가중치 벡터의 각 구성 요소의 제곱합을 남깁니다.이렇게 하면 페널티의 미분을 쉽게 계산할 수 있습니다. 도함수의 합은 합의 도함수와 같습니다. 

또한 $L_1$ 표준이 아닌 $L_2$ 표준을 사용하는 이유를 물어볼 수 있습니다.실제로 통계 전반에 걸쳐 다른 선택이 유효하고 인기가 있습니다.$L_2$ 정규화 선형 모델은 고전적인*능선 회귀* 알고리즘을 구성하지만 $L_1$-정규화 선형 회귀는 통계에서 유사하게 기본 모델이며*올가미 회귀*로 널리 알려져 있습니다. 

$L_2$ 노름으로 작업하는 한 가지 이유는 가중치 벡터의 큰 성분에 큰 페널티를 부과하기 때문입니다.이로 인해 학습 알고리즘이 더 많은 기능에 가중치를 균등하게 분배하는 모델에 편향됩니다.실제로 이렇게 하면 단일 변수에서 측정 오차에 대해 더 강력해질 수 있습니다.반대로 $L_1$의 벌칙은 다른 가중치를 0으로 지워 작은 피처 집합에 가중치를 집중시키는 모델로 이어집니다.이것을*기능 선택*이라고 하며, 다른 이유로 바람직할 수 있습니다. 

:eqref:`eq_linreg_batch_update`에서 동일한 표기법을 사용하여 $L_2$-정규화된 회귀에 대한 미니배치 확률적 경사하강법 업데이트는 다음과 같습니다. 

$$
\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
$$

이전과 마찬가지로 추정치가 관측치와 다른 양을 기준으로 $\mathbf{w}$를 업데이트합니다.그러나 $\mathbf{w}$의 크기도 0으로 축소했습니다.그렇기 때문에 이 방법을 “가중치 감소”라고도 합니다. 페널티 항만 주어지면 최적화 알고리즘이 훈련의 각 단계에서 가중치를 감소*감소*합니다.특징 선택과 달리 가중치 감쇄는 함수의 복잡성을 조정하기 위한 지속적인 메커니즘을 제공합니다.$\lambda$의 값이 작을수록 제약이 적은 $\mathbf{w}$에 해당하는 반면, $\lambda$의 값이 클수록 $\mathbf{w}$가 더 상당히 제한됩니다. 

해당 바이어스 페널티 $b^2$를 포함하는지 여부는 구현에 따라 다를 수 있으며 신경망의 계층에 따라 다를 수 있습니다.네트워크 출력 계층의 편향 항을 정규화하지 않는 경우가 많습니다. 

## 고차원 선형 회귀

간단한 합성 예를 통해 체중 감량의 이점을 설명 할 수 있습니다.

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
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

먼저 [**이전처럼 일부 데이터를 생성합니다**] 

(**$$y = 0.05 +\ sum_ {i = 1} ^d 0.01 x_i +\ 엡실론\ 텍스트 {여기서}\ 엡실론\ 심\ 수학 {N} (0, 0.01^2) .$$**) 

레이블을 평균이 0이고 표준 편차가 0.01인 가우스 잡음에 의해 손상된 입력값의 선형 함수로 선택합니다.과적합의 효과를 두드러지게 하기 위해 문제의 차원을 $d = 200$로 늘리고 20개의 예만 포함된 작은 훈련 세트로 작업할 수 있습니다.

```{.python .input}
#@tab all
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
```

## 처음부터 구현

다음에서는 원래 대상 함수에 제곱 $L_2$ 페널티를 더하는 것만으로 가중치 감쇠를 처음부터 구현할 것입니다. 

### [**모델 매개변수 초기화**]

먼저 모델 매개 변수를 임의로 초기화하는 함수를 정의합니다.

```{.python .input}
def init_params():
    w = np.random.normal(scale=1, size=(num_inputs, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    return [w, b]
```

```{.python .input}
#@tab pytorch
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]
```

```{.python .input}
#@tab tensorflow
def init_params():
    w = tf.Variable(tf.random.normal(mean=1, shape=(num_inputs, 1)))
    b = tf.Variable(tf.zeros(shape=(1, )))
    return [w, b]
```

### (**$L_2$ 규범 페널티 정의**)

이 페널티를 이행하는 가장 편리한 방법은 모든 용어를 제곱하고 합산하는 것입니다.

```{.python .input}
def l2_penalty(w):
    return (w**2).sum() / 2
```

```{.python .input}
#@tab pytorch
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
```

```{.python .input}
#@tab tensorflow
def l2_penalty(w):
    return tf.reduce_sum(tf.pow(w, 2)) / 2
```

### [**훈련 루프 정의**]

다음 코드는 모델을 훈련 세트에 적합시키고 테스트 세트에서 평가합니다.선형 네트워크와 제곱 손실은 :numref:`chap_linear` 이후로 변경되지 않았으므로 `d2l.linreg` 및 `d2l.squared_loss`를 통해 가져올 것입니다.여기서 유일한 변화는 이제 우리의 손실에 페널티 기간이 포함된다는 것입니다.

```{.python .input}
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                # The L2 norm penalty term has been added, and broadcasting
                # makes `l2_penalty(w)` a vector whose length is `batch_size`
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', np.linalg.norm(w))
```

```{.python .input}
#@tab pytorch
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # The L2 norm penalty term has been added, and broadcasting
            # makes `l2_penalty(w)` a vector whose length is `batch_size`
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', torch.norm(w).item())
```

```{.python .input}
#@tab tensorflow
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # The L2 norm penalty term has been added, and broadcasting
                # makes `l2_penalty(w)` a vector whose length is `batch_size`
                l = loss(net(X), y) + lambd * l2_penalty(w)
            grads = tape.gradient(l, [w, b])
            d2l.sgd([w, b], grads, lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', tf.norm(w).numpy())
```

### [**정규화 없는 교육**]

이제 이 코드를 `lambd = 0`로 실행하여 가중치 감쇄를 비활성화합니다.과적합이 잘못되어 훈련 오차는 감소하지만 테스트 오차는 감소하지 않습니다. 이는 과적합의 교과서 사례입니다.

```{.python .input}
#@tab all
train(lambd=0)
```

### [**체중 감쇠 사용**]

아래에서는 상당한 체중 감소로 달립니다.훈련 오차는 증가하지만 테스트 오차는 감소합니다.이것이 바로 정규화에서 기대하는 효과입니다.

```{.python .input}
#@tab all
train(lambd=3)
```

## [**간결한 구현**]

가중치 감퇴는 신경망 최적화에서 보편적이기 때문에 딥 러닝 프레임워크는 체중 감쇠를 최적화 알고리즘 자체에 통합하여 손실 함수와 함께 쉽게 사용할 수 있도록 특히 편리합니다.또한 이러한 통합은 계산상의 이점을 제공하므로 추가 계산 오버헤드 없이 구현 트릭을 통해 알고리즘에 가중치 감소를 추가할 수 있습니다.업데이트의 가중치 감소 부분은 각 파라미터의 현재 값에만 의존하므로 옵티마이저는 각 파라미터를 한 번 터치해야 합니다.

:begin_tab:`mxnet`
다음 코드에서는 `Trainer`를 인스턴스화할 때 `wd`를 통해 직접 가중치 감쇠 하이퍼파라미터를 지정합니다.기본적으로 Gluon은 가중치와 편향을 동시에 감소시킵니다.모델 매개변수를 업데이트할 때 하이퍼파라미터 `wd`에 `wd_mult`이 곱해집니다.따라서 `wd_mult`을 0으로 설정하면 바이어스 매개 변수 $b$이 감쇠되지 않습니다.
:end_tab:

:begin_tab:`pytorch`
다음 코드에서는 옵티마이저를 인스턴스화할 때 `weight_decay`를 통해 직접 가중치 감쇠 하이퍼파라미터를 지정합니다.기본적으로 PyTorch는 가중치와 편향을 동시에 감소시킵니다.여기서는 가중치에 대해 `weight_decay`만 설정하므로 편향 매개변수 $b$는 감소하지 않습니다.
:end_tab:

:begin_tab:`tensorflow`
다음 코드에서는 가중치 감쇠 하이퍼파라미터 `wd`를 사용하여 $L_2$ 정규화기를 만들고 `kernel_regularizer` 인수를 통해 계층에 적용합니다.
:end_tab:

```{.python .input}
def train_concise(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    loss = gluon.loss.L2Loss()
    num_epochs, lr = 100, 0.003
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': wd})
    # The bias parameter has not decayed. Bias names generally end with "bias"
    net.collect_params('.*bias').setattr('wd_mult', 0)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', np.linalg.norm(net[0].weight.data()))
```

```{.python .input}
#@tab pytorch
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # The bias parameter has not decayed
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', net[0].weight.norm().item())
```

```{.python .input}
#@tab tensorflow
def train_concise(wd):
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(wd)))
    net.build(input_shape=(1, num_inputs))
    w, b = net.trainable_variables
    loss = tf.keras.losses.MeanSquaredError()
    num_epochs, lr = 100, 0.003
    trainer = tf.keras.optimizers.SGD(learning_rate=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # `tf.keras` requires retrieving and adding the losses from
                # layers manually for custom training loop.
                l = loss(net(X), y) + net.losses
            grads = tape.gradient(l, net.trainable_variables)
            trainer.apply_gradients(zip(grads, net.trainable_variables))
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', tf.norm(net.get_weights()[0]).numpy())
```

[**플롯은 처음부터 무게 감소를 구현했을 때의 플롯과 동일하게 보입니다**].그러나 훨씬 더 빠르게 실행되고 구현하기가 더 쉬우므로 더 큰 문제에서 더 두드러지게 나타납니다.

```{.python .input}
#@tab all
train_concise(0)
```

```{.python .input}
#@tab all
train_concise(3)
```

지금까지 우리는 단순한 선형 함수를 구성하는 것에 대한 한 가지 개념만 다루었습니다.게다가 단순한 비선형 함수를 구성하는 것은 훨씬 더 복잡한 질문이 될 수 있습니다.예를 들어, [커널 힐베르트 공간 (RKHS) 재현](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) 을 사용하면 비선형 컨텍스트에서 선형 함수에 도입 된 도구를 적용 할 수 있습니다.안타깝게도 RKH 기반 알고리즘은 대규모의 고차원 데이터로 확장되지 않는 경향이 있습니다.이 책에서는 심층 네트워크의 모든 계층에 가중치 감쇄를 적용하는 간단한 휴리스틱을 기본값으로 사용합니다. 

## 요약

* 정규화는 과적합을 처리하는 일반적인 방법입니다.학습 세트의 손실 함수에 페널티 항을 추가하여 학습된 모델의 복잡성을 줄입니다.
* 모형을 단순하게 유지하기 위한 한 가지 특별한 선택은 $L_2$ 페널티를 사용한 중량 감소입니다.이로 인해 학습 알고리즘의 업데이트 단계에서 가중치가 감소합니다.
* 가중치 감쇠 기능은 딥러닝 프레임워크의 옵티마이저에서 제공됩니다.
* 서로 다른 파라미터 세트는 동일한 훈련 루프 내에서 서로 다른 업데이트 동작을 가질 수 있습니다.

## 연습문제

1. 이 섹션의 추정 문제에서 $\lambda$의 값을 사용하여 실험합니다.훈련 및 테스트 정확도를 $\lambda$의 함수로 플로팅합니다.무엇을 관찰하시나요?
1. 검증 세트를 사용하여 최적값 $\lambda$를 구합니다.정말 최적의 가치일까요?이게 중요한가요?
1. $\|\mathbf{w}\|^2$ 대신 $\sum_i |w_i|$를 선택 페널티 ($L_1$ 정규화) 로 사용했다면 업데이트 방정식은 어떻게 생겼을까요?
1. 우리는 $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$라는 것을 알고 있습니다.행렬에 대해 비슷한 방정식을 찾을 수 있습니까 (:numref:`subsec_lin-algebra-norms`의 프로베니우스 노름 참조)?
1. 훈련 오류와 일반화 오류 간의 관계를 검토합니다.체중 감소, 훈련 증가 및 적절한 복잡성 모델 사용 외에도 과적합을 처리하는 다른 방법은 무엇입니까?
1. 베이지안 통계에서는 $P(w \mid x) \propto P(x \mid w) P(w)$를 통해 후방에 도달하기 위해 이전 및 가능성의 곱을 사용합니다.정규화를 통해 $P(w)$를 어떻게 식별할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/236)
:end_tab:
