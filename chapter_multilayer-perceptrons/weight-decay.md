# 무게 감쇠
:label:`sec_weight_decay`

이제 우리는 과대 적합의 문제를 특징으로했으므로 모델을 정규화하기위한 몇 가지 표준 기술을 도입 할 수 있습니다.우리는 외출하고 더 많은 교육 데이터를 수집함으로써 항상 과잉 피팅을 완화할 수 있음을 상기하십시오.이는 비용이 많이 들거나 시간이 많이 걸리거나 완전히 통제할 수 없기 때문에 단기적으로는 불가능할 수 있습니다.현재로서는 자원이 허용하는 만큼의 고품질 데이터를 이미 보유하고 있고 정규화 기술에 집중한다고 가정 할 수 있습니다.

다항식 회귀 분석 예제 (:numref:`sec_model_selection`) 에서는 적합된 다항식의 정도를 조정하여 모델의 용량을 제한 할 수 있습니다.실제로 기능의 수를 제한하는 것은 과잉 적합을 완화하기 위해 널리 사용되는 기술입니다.그러나 단순히 기능을 따로 던지기만 하면 작업에 너무 무딘 도구가 될 수 있습니다.다항식 회귀 예제를 고수하여 고차원 입력에서 발생할 수있는 일을 고려하십시오.다변량 데이터에 대한 다항식의 자연스러운 확장은*단일*이라고 하며, 이는 단순히 변수의 거듭 제곱의 산물이다.단일식의 정도는 힘의 합입니다.예를 들어, $x_1^2 x_2$과 $x_3 x_5^2$는 모두 차수의 단일식입니다.

$d$가 73214가 커짐에 따라 7323614의 항이 급속하게 날아갑니다.$k$ 변수를 감안할 때, 학위 $d$ (즉, $k$ 다중 선택) 의 모노미어의 수는 ${k - 1 + d} \choose {k - 1}$입니다.$2$에서 $3$으로 말하는 정도의 작은 변화조차도 우리 모델의 복잡성을 크게 증가시킵니다.따라서 우리는 종종 함수 복잡성을 조정하기위한보다 세밀한 도구가 필요합니다.

## 규범 및 체중 감쇠

우리는 $L_2$ 표준과 $L_1$ 규범을 모두 설명했으며, 이는 :numref:`subsec_lin-algebra-norms`에서보다 일반적인 $L_p$ 표준의 특수한 경우입니다.
*체중 감량* (일반적으로 $L_2$ 정규화라고 함)
는 파라메트릭 기계 학습 모델을 정규화하는 데 가장 널리 사용되는 기술일 수 있습니다.이 기술은 모든 함수 중에서 $f$ 함수 $f = 0$ (모든 입력에 값 $0$ 할당) 가 어떤 의미에서는*가장 단순한*이고 0으로부터의 거리로 함수의 복잡성을 측정 할 수 있다는 기본 직감에 의해 동기 부여됩니다.그러나 함수와 0 사이의 거리를 얼마나 정확하게 측정해야합니까?정답은 하나도 없습니다.실제로 기능 분석의 일부와 Banach 공간 이론을 포함한 전체 수학 분야는이 문제에 대한 답변에 전념합니다.

한 가지 간단한 해석은 선형 함수 $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$의 가중치 벡터의 일부 표준 (예: $\| \mathbf{w} \|^2$) 으로 복잡성을 측정하는 것입니다.작은 무게 벡터를 보장하는 가장 일반적인 방법은 손실을 최소화하는 문제에 페널티 용어로 표준을 추가하는 것입니다.따라서 우리는 원래의 목적,
*트레이닝 라벨에 대한 예측 손실 최소화*,
새로운 목표,
*예측 손실과 페널티 용어의 합을 최소화합니다.
이제 가중치 벡터가 너무 커지면 학습 알고리즘은 훈련 오류를 최소화하는 것과 비교하여 체중 기준 $\| \mathbf{w} \|^2$를 최소화하는 데 집중할 수 있습니다.그것이 바로 우리가 원하는 것입니다.코드에서 일을 설명하기 위해, 우리가 선형 회귀 :numref:`sec_linear_regression`에서 우리의 이전 예제를 부활 할 수 있습니다.저기, 우리의 손실은

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

$\mathbf{x}^{(i)}$은 기능이고 $y^{(i)}$는 모든 데이터 포인트 $i$의 레이블이며 $(\mathbf{w}, b)$은 각각 가중치 및 바이어스 매개 변수입니다.가중치 벡터의 크기를 처벌하려면 어떻게 든 $\| \mathbf{w} \|^2$을 손실 함수에 추가해야하지만 모델은이 새로운 첨가제 페널티의 표준 손실을 어떻게 상쇄해야합니까?실제로 유효성 검사 데이터를 사용하여 적합한 음수가 아닌 하이퍼 매개 변수 인 정규화 상수* $\lambda$를 통해이 절충점을 특성화합니다.

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2,$$

$\lambda = 0$의 경우 원래의 손실 기능을 복구합니다.$\lambda > 0$의 경우 $\| \mathbf{w} \|$의 크기를 제한합니다.우리는 규칙에 따라 $2$로 나눕니다. 2 차 함수의 파생물을 취하면 $2$ 및 $1/2$가 취소되어 업데이트에 대한 표현식이 멋지고 단순 해 보입니다.기만한 독자는 왜 우리가 표준 표준 (즉, 유클리드 거리) 이 아닌 제곱 된 표준으로 작업하는지 궁금해 할 것입니다.우리는 계산상의 편의를 위해이 작업을 수행합니다.$L_2$ 규범을 제곱하여 가중치 벡터의 각 구성 요소의 제곱합을 남기고 제곱근을 제거합니다.이로 인해 페널티의 미분을 쉽게 계산할 수 있습니다. 파생 상품의 합은 합계의 미분과 같습니다.

또한 $L_1$ 표준이 아닌 $L_2$ 표준으로 작업하는 이유를 물어볼 수 있습니다.사실, 다른 선택은 통계 전반에 걸쳐 유효하고 인기가 있습니다.$L_2$-정규화된 선형 모델이 클래식*릿지 회귀* 알고리즘을 구성하는 반면, $L_1$-정규화된 선형 회귀 분석은 통계에서 유사하게 기본 모델이며, 일반적은*올가미 회귀*라고 합니다.

$L_2$ 표준으로 작업하는 한 가지 이유는 무게 벡터의 대형 구성 요소에 대형 페널티를 부과하기 때문입니다.이것은 우리의 학습 알고리즘을 더 많은 수의 기능에 걸쳐 균등하게 가중치를 분배하는 모델에 편향시킵니다.실제로 이것은 단일 변수에서 오류를 측정하는 데 더 견고하게 만들 수 있습니다.반대로 $L_1$ 페널티는 다른 가중치를 0으로 제거하여 작은 피쳐 세트에 가중치를 집중시키는 모델을 초래합니다.이를 *기능 선택*이라고 하며, 다른 이유로 바람직 할 수 있습니다.

:eqref:`eq_linreg_batch_update`에서 동일한 표기법을 사용하여 $L_2$-정규화 회귀 분석에 대한 미니 배치 확률 그라데이션 하강 업데이트는 다음과 같습니다.

$$
\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
$$

이전과 마찬가지로 추정치가 관측치와 다른 양에 따라 $\mathbf{w}$를 업데이트합니다.그러나 $\mathbf{w}$의 크기를 0으로 축소합니다.이것이 바로 이 방법을 “체중 감량”이라고도 하는 이유입니다. 페널티 용어만으로, 우리의 최적화 알고리즘은 훈련의 각 단계에서 가중치를 *감쇄합니다.기능 선택과는 달리, 무게 감쇠는 우리에게 함수의 복잡성을 조정하기위한 지속적인 메커니즘을 제공합니다.$\lambda$의 값이 작을수록 제약된 $\mathbf{w}$에 해당하는 반면, $\lambda$의 값이 클수록 $\mathbf{w}$가 더 많이 제한됩니다.

해당 바이어스 페널티 $b^2$를 포함하는지 여부는 구현마다 다를 수 있으며 신경망 계층마다 다를 수 있습니다.종종 네트워크 출력 계층의 바이어스 용어를 정규화하지 않습니다.

## 고차원 선형 회귀 분석

우리는 간단한 합성 예를 통해 체중 감량의 이점을 설명 할 수 있습니다.

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
import torch.nn as nn
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

첫째, 이전과 같이 일부 데이터를 생성합니다.

$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$

우리는 제로 평균 및 표준 편차 0.01로 가우스 노이즈에 의해 손상된 입력의 선형 함수로 우리의 레이블을 선택합니다.과잉 적합의 효과를 발음하기 위해 문제의 차원을 $d = 200$로 높이고 20 가지 예제 만 포함하는 작은 교육 세트로 작업 할 수 있습니다.

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

다음에서는 원래 대상 함수에 제곱 된 $L_2$ 페널티를 추가하여 처음부터 체중 감량을 구현할 것입니다.

### 모델 매개변수 초기화

첫째, 우리는 무작위로 우리의 모델 매개 변수를 초기화하는 함수를 정의합니다.

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

### $L_2$ 표준 페널티 정의

아마도이 벌금을 구현하는 가장 편리한 방법은 모든 용어를 제자리에 제곱하고 요약하는 것입니다.

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

### 교육 루프 정의

다음 코드는 학습 집합의 모델을 적합시키고 테스트 집합에서 평가합니다.선형 네트워크와 제곱 손실은 :numref:`chap_linear` 이후로 변경되지 않았으므로 `d2l.linreg` 및 `d2l.squared_loss`를 통해 가져올 것입니다.여기서 유일한 변화는 이제 우리의 손실에 페널티 기간이 포함된다는 것입니다.

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
            with torch.enable_grad():
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

### 정규화 없는 교육

이제 `lambd = 0`로이 코드를 실행하여 체중 감량을 비활성화합니다.우리는 심하게 과도하게 적합하며, 훈련 오류는 감소하지만 테스트 오류는 감소하지 않습니다.

```{.python .input}
#@tab all
train(lambd=0)
```

### 가중치 감소 사용

아래, 우리는 상당한 체중 감량으로 실행합니다.훈련 오류는 증가하지만 테스트 오류는 감소합니다.이것은 정확하게 우리가 정규화에서 기대하는 효과입니다.

```{.python .input}
#@tab all
train(lambd=3)
```

## 간결한 구현

체중 감량은 신경망 최적화에서 유비쿼터스이기 때문에 딥 러닝 프레임워크는 체중 감량을 최적화 알고리즘 자체에 통합하여 손실 함수와 함께 쉽게 사용할 수 있습니다.더욱이 이러한 통합은 계산 상의 이점을 제공하므로 구현 트릭이 추가 계산 오버 헤드없이 알고리즘에 가중치를 추가 할 수 있습니다.업데이트의 가중치 감소 부분은 각 매개 변수의 현재 값에만 의존하므로 옵티마이 저는 각 매개 변수를 한 번 터치해야합니다.

:begin_tab:`mxnet`
우리의 `Trainer`를 인스턴스화 할 때 다음 코드에서, 우리는 `wd`을 통해 직접 무게 감쇠 하이퍼 매개 변수를 지정합니다.기본적으로 글루온은 가중치와 바이어스를 동시에 감소시킵니다.모델 매개변수를 업데이트할 때 하이퍼파라미터 `wd`에 7323615를 곱합니다.따라서 `wd_mult`를 0으로 설정하면 바이어스 매개 변수 $b$이 붕괴되지 않습니다.
:end_tab:

:begin_tab:`pytorch`
다음 코드에서는 옵티 마이저를 인스턴스화 할 때 `weight_decay`를 통해 직접 체중 감량 하이퍼 매개 변수를 지정합니다.기본적으로 PyTorch는 가중치와 바이어스를 동시에 감소시킵니다.여기서 우리는 무게에 대해 `weight_decay` 만 설정하므로 바이어스 매개 변수 $b$는 부패되지 않습니다.
:end_tab:

:begin_tab:`tensorflow`
다음 코드에서는 체중 감량 하이퍼 매개 변수 `wd`를 사용하여 $L_2$ 정규기를 만들고 `kernel_regularizer` 인수를 통해 레이어에 적용합니다.
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
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    # The bias parameter has not decayed
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                trainer.zero_grad()
                l = loss(net(X), y)
            l.backward()
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

플롯은 처음부터 체중 감량을 구현할 때의 플롯과 동일하게 보입니다.그러나, 그들은 상당히 빠르게 실행되고 구현하기 쉽기 때문에 더 큰 문제에 대해 더 두드러지게 될 것입니다.

```{.python .input}
#@tab all
train_concise(0)
```

```{.python .input}
#@tab all
train_concise(3)
```

지금까지 우리는 단순한 선형 함수를 구성하는 것에 대한 한 가지 개념만을 만졌습니다.또한 단순한 비선형 함수를 구성하는 것은 훨씬 더 복잡한 질문 일 수 있습니다.예를 들어, [커널 Hilbert 공간 (RKHS) 재현](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) 을 사용하면 선형 함수에 대해 도입 된 도구를 비선형 컨텍스트에서 적용 할 수 있습니다.불행히도 RKHS 기반 알고리즘은 순수하게 대규모 고차원 데이터로 확장되는 경향이 있습니다.이 책에서 우리는 깊은 네트워크의 모든 계층에 체중 감량을 적용하는 간단한 경험적 방법을 기본값으로 할 것입니다.

## 요약

* 정규화는 과잉 적합을 처리하는 일반적인 방법입니다.학습 집합의 손실 함수에 페널티 용어를 추가하여 학습된 모델의 복잡성을 줄입니다.
* 모델을 단순하게 유지하기 위한 한 가지 특별한 선택은 $L_2$ 페널티를 사용한 무게 감쇠입니다.이로 인해 학습 알고리즘의 업데이트 단계에서 체중 감량이 발생합니다.
* 무게 감쇠 기능은 딥 러닝 프레임워크의 옵티마이저에서 제공됩니다.
* 매개변수 세트마다 동일한 학습 루프 내에서 서로 다른 업데이트 동작이 있을 수 있습니다.

## 연습 문제

1. 이 섹션의 추정 문제에서 $\lambda$의 값으로 실험하십시오.$\lambda$의 함수로 트레이닝 및 테스트 정확도를 플롯합니다.당신은 무엇을 관찰합니까?
1. 유효성 검사 집합을 사용하여 최적의 값 $\lambda$를 찾습니다.정말 최적의 가치입니까?이게 중요한가?
1. $\|\mathbf{w}\|^2$ 대신 $\sum_i |w_i|$을 선택의 페널티 ($L_1$ 정규화) 로 사용하면 업데이트 방정식은 어떻게 생겼습니까?
1. 우리는 그 $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$를 알고 있습니다.행렬에 대해 비슷한 방정식을 찾을 수 있습니까 (:numref:`subsec_lin-algebra-norms`의 프로베니우스 표준 참조)?
1. 교육 오류와 일반화 오류 간의 관계를 검토합니다.체중 감퇴, 증가 된 훈련 및 적절한 복잡성의 모델 사용 외에도 과잉 피팅을 다루기 위해 다른 어떤 방법을 생각할 수 있습니까?
1. 베이지안 통계에서는 $P(w \mid x) \propto P(x \mid w) P(w)$를 통해 후부까지 도달할 수 있는 이전 및 가능성의 제품을 사용합니다.정규화로 $P(w)$를 어떻게 식별 할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/236)
:end_tab:
