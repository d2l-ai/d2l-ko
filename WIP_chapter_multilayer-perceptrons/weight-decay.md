# 가중치 감쇠 (weight decay)

오버피팅 문제가 어떤 특징을 가지고 있는지 확인했고 용량 제어가 왜 필요한지를 살펴봤으니, 이제 우리는 이를 해결하기 위해서 사용되는 유명한 기법들을 알아보겠습니다. 더 많은 학습 데이터를 사용하면 오버피팅 문제는 늘 해결 할 수 있지만, 데이터를 더 수집하는 것은 비용이 들고 시간이 걸리기 때문에 일반적으로 짧은 기간에 이뤄질 수 없습니다. 지금은 우리의 리소스를 최대한 사용해서 높은 품질의 데이터를 이미 얻었다고 가정하고, 사용할 함수 클래스들의 용량을 제한하는 기법들에 집중하겠습니다.

앞에서 살펴본 간단한 예제에서 다항식의 복잡도는 차원은 차수을 조정하는 것으로 제어할 수 있는 것을 확인했습니다. 하지만, 대부분의 머신러닝은 다항 곡선으로 구성되어 있지 않습니다. 그리고 더구나 다항식 회귀을 다루는 경우에도, 고차원 데이터를 다룰 때는 차수 $d$ 를 조정하면서 모델의 용량을 바꾸는 것은 문제가 있습니다. 그 이유 설명하기 위해서, 다중 변수 데이터에서는 단순히 변수들의 제곱들의 곱인 *모노미얼(monomial)* 들을 포함하는 다항식의 개념을 일반화를 해야한다는 것을 주목하세요. 예를 들어, $x_1^2 x_2$ 와 $x_3 x_5^2$ 는 차수가 3인 모노미얼입니다. 어떤 차수 $d$ 의 항들의 개수는 차수 $d$ 의 함수에 따라서 급격히 증가합니다.

구체적으로 차원이 $D$ 인 벡터에 대해서 어떤 차수 $d$ 의 모노미얼의 개수는 ${D -1 + d} \choose {D-1}$ 입니다. 따라서, 1에서 2 또는 2에서 3과 같은 차수의 작은 변화는 모델의 복잡도에 큰 증가를 가져옵니다. 즉, 차수를 조정하는 것은 너무 무딘 망치에 불과합니다. 대신 우리는 함수의 복잡도를 조정하는 보다 세밀한 도구가 필요합니다.

## 제곱 놈 정칙화(squared norm regularization)

일반적으로 L2 정칙화라고 불리는 가중치 감쇠(weight decay)는 파라메터 기반의 머신러닝 모델을 정칙화 하는데 가장 널리 사용되는 기술입니다. 가중치 감쇠에 대한 기본적인 직관은 모든 함수 $f$ 중에서 함수 $f = 0$ 가 가장 단순하다는 생각에 착안하고 있습니다. 직관적으로 설명하면 우리는 0과 가까운 정도로 함수를 측정할 수 있습니다. 하지만 어떻게 함수와 0 사이의 거리를 정확하게 측정해야 할까요? 정답은 없습니다. 사실, 이를 측정하는 방법은 다양한데 별도의 수학 분야가 존재하기까지 합니다. 예를 들면, 이 문제에 대한 답을 찾는 것에 목적을 두고 있는 함수 해석학과 바나흐 공간 이론 (the theory of Banach spaces)이 있습니다.

우리의 현재 목적을 위해서는 아주 간단한 형태면 충분합니다. 우리는 가중치 벡터가 작다면 선형 함수 $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$ 가 간단하다고 간주하겠습니다. 우리는 이것을 $||\mathbf{w}||^2$ 로 측정할 수 있습니다. 가중치 벡터를 작게 유지하는 방법 중에 하나는 손실(loss)를 최소화하는 문제에 놈을 패널티 항목으로 더하는 것입니다. 즉, 우리는 원리 목적인 *학습 레이블에 대한 예측 오류를 최소화* 하는 것에서 새로운 목적 *예측 오류와 패널티 항목의 함을 최소화* 하는 문제로 바꿉니다. 이제 가중치 벡터가 너무 커지면 우리의 학습 알고리즘은  학습 오류를 최소화하는 것 보다는 놈 $|| \mathbf{w} ||^2$ 을 최소화하는 방법을 찾을 것입니다. 이것이 바로 우리가 원하는 것입니다. 코드로 설명하기 위해서  [선형 회귀](linear-regression.md) 장에서 다룬 예제를 다시 보겠습니다. 그 예제에서 손실은 $$l(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2$$  로 정의되었습니다.

 $\mathbf{x}^{(i)}$ 는 관측값, $y^{(i)}$ 는 레이블, 그리고  $(\mathbf{w}, b)$ 는 각각 가중치와 편향임을 기억하세요. 가중치 벡터의 크기에 대한 퍼널티를 주는 새로운 손실 함수를 갖기 위해서 우리는 $|| \mathbf{w} ||^2$ 를 더해야 했습니다. 하지만, 얼마나 더하는게 맞을까요? 이를 해결하는 방법으로 우리는 새로운 하이퍼파라미터를 추가합니다. 이는 *정칙화 상수(regularization constant)* 라고 부르고 $\lambda$ 라고 표기합니다:

$$l(\mathbf{w}, b) + \frac{\lambda}{2} \|\boldsymbol{w}\|^2$$

음수가 아닌 이 파라미터 $\lambda \geq 0$  가 정칙화의 정도를 조절합니다. $\lambda = 0$ 인 경우, 원래의 손실 함수가 되고, $\lambda > 0$ 인 경우에는, $\mathbf{w}$ 가 너무 커지지 않게 강제합니다. 통찰력이 있는 여러분은  가중치 벡터의 놈을 제곱하는 이유에 대해서 궁금해할 것입니다. 이렇게 하는데는 두 가지 이유가 있습니다. 첫번째 이유는, 연산 편의성을 위함입니다. L2 놈을 제곱하면, 우리는 제곱근을 제거할 수 있고, 그 결과 가중치 벡터의 각 컴포넌트의 제곱의 합이 됩니다. 항들의 합에 대한 미분을 구하는 것이 쉽기 때문에 계산을 편리하게 해줍니다. (미분들의 합은 합의 미분과 같이 때문입니다.)

여러분은 또한 왜 L1 놈이나 다른 거리 함술르 사용하지 않고 L2 놈을 사용하는지 궁금할 것입니다. 사실 다른 여러 방법들도 유효하고 통계학에서 유명합니다. L2 정칙화 선형 모델을 전통적인 *리지 회귀 (ridge regression)* 알고리즘을 구성하는 반면, L1 정칙화 선형 모델은 통계학에서 *라소 회귀 (lasso regression)*으로 유명하게 알려져 있는 비슷한 기초적인 모델을 구성합니다.

다른 놈을 사용하지 않고 L2 놈을 사용하는 수학적인 이유 중에 하나는 가중치 벡터의 큰 컴포넌트에 작은 것들 보다 더 많은 패널티를 주기 때문입니다. 이렇게 하는 것은 우리의 학습 알로리즘이 부주의하게 한 개의 특성에 의존하는 대신에 많은 특성들에 걸쳐서 가충지를 분산할 수 있게하기 때문입니다. 확률적 경사 하강법은 L2 정칙화된 회귀를 다름과 같이 업데이트 합니다.
$$
\begin{aligned}
w & \leftarrow \left(1- \frac{\eta\lambda}{|\mathcal{B}|} \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),
\end{aligned}
$$

이전과 같이, 관찰된 값과 예측된 값의 차이에 따라서  $\mathbf{w}$ 를 업데이트합니다. 하지만,  $\mathbf{w}$ 의 크기를  $0$ 과 가까워지게 줄이고 있습니다. 즉, 가중치를 감쇠하게(decay) 만듭니다. 이전과 동일하게 우리는 관찰된 값과 추정된 값의 차이의 정도를 기반으로  $\mathbf{w}$ 를 업데이트 합니다. 하지만, 우리는  $\mathbf{w}$ 의 크기를 0에 가까워지도록 줄일 수도 있습니다. 이것이 우리가 이 방법을 "가중침 감소(weight decay)" 라고 부르는 이유인데, 즉, 패널티 항은 우리의 최적화 알고리즘이 매 학습 단계에서 가중치의 정도를 *감쇠* 시키고 있기 때문입니다. 이것은 다항식에 파라미터 개수를 선택하는 것보다 더 편한 방법입니다. 특히, $f$ 의 복잡도를 조절하는 연속성이 있는 방법을 갖게 되었습니다. 작은 $\lambda$ 값은  $\mathbf{w}$ 를 적게 제약하는 반면, 큰 값은  $\mathbf{w}$ 를 많이 제약합니다. 편향(bias) 항 역시 큰 값을 갖기를 원하지 않기 때문에,  $b^2$ 를 패널티로 더하기도 합니다.

## 고차원 선형 회귀

고차원 회귀(regression)에서 생략할 정확한 차원을 선택하기 어려운데, 가중치 감쇠 정칙화(weight-decay regularization)는 아주 간편한 대안이 됩니다. 왜 그런지를 지금부터 설명하겠습니다. 우선, 입력은 평균이 0이고 분산이 0.01인 가우시안 노이즈를 따르며, 레이블은 입력들에 대한 선형 함수로 표현되는 아래 공식을 사용해서 데이터를 생성합니다.

$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.01)$$

오버피팅 효과를 더 쉽게 관찰하기 위해서, 데이터 차원 $d = 200$ 로 설정하고 학습 샘플을 상대적으로 적게해서 (여기서는 샘플 크기를 20으로 설정합니다) 고차원(high-dimensional)로 만들겠습니다. 

```{.python .input  n=2}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = nd.ones((num_inputs, 1)) * 0.01, 0.05

features = nd.random.normal(shape=(n_train + n_test, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]
```

## 처음부터 구현하기

자 이제, 가중치 감쇠를 직접 구현해 보겠습니다. 우리가 해야할 일은 원래의 타켓 함수에 추가 손실 항으로 제곱한 $\ell_2$ 패널티를 더해주면 됩니다. 제곱 놈 패널티라고 부르는 이유는 제곱값 $\sum_i w_i^2$ 를 더하기 때문입니다. $\ell_2$ 는 p-놈의 무한한 종류 중에 하나이고, 앞으로 여러 놈의 형태를 접할지도 모릅니다. 일반적으로, 어떤 수 $p$ 에 대해서  $\ell_p$ 놈은 다음과 같이 정의합니다.

$\|\mathbf{w}\|_p^p := \sum_{i=1}^d |w_i|^p$

### 파라미터 초기화하기

먼저 우리는 모델 파라미터들을 임의의 수로 초기화하는 함수를 구현하고, 앞으로 계산할 미분값에 대한 메모리를 할당하기 위해서 `attach_grad` 를 각 파라미터에 대해서 수행합니다.

```{.python .input  n=5}
def init_params():
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    return [w, b]
```

### $\ell_2$ 놈 페널티(Norm Penalty) 정의하기

아마도 이 패널티를 구현하는 가장 편리한 방법은 모든 항의 제곱을 구하고, 이들의 합을 계산하는 것입니다. 이 값을 편의를 위해서 2로 나눕니다. (2차 함수를 미분하면 2가 나오는데 $1/2$ 는 이를 제거해줍니다. 이는 업데이트할 수식을 보기 좋고 간단하게 만들어 줍니다. )

```{.python .input  n=6}
def l2_penalty(w):
    return (w**2).sum() / 2
```

### 학습 및 테스트 정의하기

다음 코드는 학습 데이터셋과 테스트 데이터셋을 이용해서 모델 학습과 모델 테스트를 어떻게 수행할지를 정의합니다. 이전 절들과는 다르게 여기서는 $\ell_2$ 놈 패널티 항은 최종 손실 함수를 계산할 때 더합니다. 선형 네트워크와 제곱 손실에 대한 구현은 이전 장의 것과 동일하기 때문에,  `d2l.linreg` 와 `d2l.squared_loss` 로 import 하겠습니다.

```{.python .input  n=7}
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss
train_iter = gdata.DataLoader(gdata.ArrayDataset(
    train_features, train_labels), batch_size, shuffle=True)

def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                # The L2 norm penalty term has been added
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features, w, b),
                            test_labels).mean().asscalar())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('l2 norm of w:', w.norm().asscalar())
```

### 정칙화(regularization) 없이 학습하기

자 이제 고차원의 선형 회귀(linear regression) 모델을 학습시키고 테스트해봅니다.  `lambd = 0`  인 경우에는 가중치 감쇠(weight decay)를 사용하지 않습니다. 그 결과로, 학습 오류가 줄어드는 반면, 테스트 오류는 줄어들지 않게 됩니다. 즉, 오버피팅(overfitting)의 완벽한 예제가 만들어졌습니다.

```{.python .input  n=8}
fit_and_plot(lambd=0)
```

### 가중치 감쇠(weight decay) 사용하기

아래 예는 학습 오류는 증가하는 반면, 테스트 오류는 감소하는 것을 보여줍니다. 이것은 가중치 감쇠(weight decay)를 사용하면서 예상한 개선된 결과입니다. 완벽하지는 않지만, 오버피팅(overfitting) 문제가 어느정도 해결되었습니다. 추가로, 가중치  $\mathbf{w}$ 에 대한  $\ell_2$ 놈(norm)도 가중치 감쇠(weight decay)를 사용하지 않을 때보다 작아졌습니다.

```{.python .input  n=9}
fit_and_plot(lambd=3)
```

## 간결한 구현

가중치 감쇠는 뉴럴 네트워크 최적화에서 많이 쓰기는 것이기 때문에, Gluon은 어떤 손실 함수와 함께 쉽게 사용될 수 있도록 가중치 감쇠를 최적화 알고리즘에 통합해서 편리하게 만들어놨습니다. 또한, 이런 통합은 추가적인 연산 부하 없이 알고리즘에 가중치 감쇠를 추가하는 것을 가능하게 했기 때문에, 연산상의 이점도 제공합니다. 그 이유는 가중치 감쇠의 업데이트는 각 파라미터의 현재 값에만 의존하고, 즉, 옵티마이저는 각 파라미터를 한번만 사용하면 되기 때문입니다.

아래 코드는 `Trainer` 를 초기화할 때, `wd` 파라미터를 통해서 가중치 감쇠 하이퍼파라미터를 설정합니다. 기본적으로 Gluon은 가중치와 편향을 동시에 감쇠시킵니다. 우리는 다른 파라미터들의 세트에 *다른* 옵티마이저를 사용할 수 있다는 것을 주의해주세요. 예를 들면, 가중치 $\mathbf{w}$ 에 대해서 가중치 감소를 갖는 `Trainer` 한 개를 사용하는 반면에, 편향 $b$ 를 위해서 가중치 감소가 없는 다른 옵티마이저를 사용할 수 있습니다.

```{.python .input}
def fit_and_plot_gluon(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    # The weight parameter has been decayed. Weight names generally end with
    # "weight".
    trainer_w = gluon.Trainer(net.collect_params('.*weight'), 'sgd',
                              {'learning_rate': lr, 'wd': wd})
    # The bias parameter has not decayed. Bias names generally end with "bias"
    trainer_b = gluon.Trainer(net.collect_params('.*bias'), 'sgd',
                              {'learning_rate': lr})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            # Call the step function on each of the two Trainer instances to
            # update the weight and bias separately
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        train_ls.append(loss(net(train_features),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features),
                            test_labels).mean().asscalar())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net[0].weight.data().norm().asscalar())
```

그래프는 가중치 감쇠(weight decay)를 직접 구현해서 얻었던 것과 아주 비슷하게 생겼습니다. 하지만, 더 빠르고 더 구현하기 쉬웠고, 이 이점은 큰 문제의 경우에는 더욱 드러나게 됩니다.

```{.python .input}
fit_and_plot_gluon(0)
```

```{.python .input}
fit_and_plot_gluon(3)
```

지금까지 우리는 간단한 *선형* 함수를 구성하는 것들에 대해서만 다뤘습니다. 비선형 함수의 경우 무엇인 *간결함* 을 구성하는지는 아무 복잡한 질문일 것입니다. 예를 들어 선형 함수들에 적용되는 다양한 도구들을 비선형 상황에서 사용할 수 있도록 하는 [Reproducing Kernel Hilbert Spaces (RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) 이 있습니다. 하지만 불행하게도 RKSH 기반의 알고리즘은 많은 양의 데이터에 대해서 항상 확장성을 보이지는 않습니다. 이 책의 목적에 부합하기 위해서 우리는 다른 층의 가중치들을 단순히 합하는 것  $\sum_l \|\mathbf{w}_l\|^2$ 을 사용하겠습니다. 이는 모든 층에 적용된 가중치 감쇠와 동일합니다.

## 요약

* 정칙화(regularization)는 오버피팅(overfitting)을 다루는 일반적인 방법입니다. 학습된 모델의 복잡도를 줄이기 위해서 학습 데이터에 대한 손실 함수(loss function)의 값에 패널티 항목을 더합니다.
* 모델을 간단하게 유지하는 방법으로  $\ell_2$ 놈 패널티(norm penalty)를 사용하는 가중치 감쇠(weight decay)를 선택했습니다. 이를 통해서, 학습 알고리즘의 업데이트 단계에서 가중치 감쇠(weight decay)가 적용됩니다.
* Gluon은 옵티마이저(optimizer)에 하이퍼파라미터(hyperparameter) `wd` 를 설정하는 것으로 가중치 감쇠(weight decay) 기능을 자동으로 추가할 수 있습니다.
* 같은 학습에서 파라메미마다 다른 옵티마이저(optimizer)를 적용할 수 있습니다.

## 연습문제

1. 이 장의 예측 문제에서  $\lambda$ 값을 실험해보세요.  $\lambda$ 에 대한 함수의 형태로 학습 정확도와 테스트 정확도를 도식화해보세요. 어떤 것이 관찰되나요?
1. 검증 데이터셋을 이용해서 최적의 $\lambda$ 값을 찾아보세요. 찾은 값이 진짜 최적 값인가요? 진짜 값을 찾는 것이 중요한가요?
1. 패널티 항목으로 $\|\mathbf{w}\|^2$ 대신  $\sum_i |w_i|$ 를 사용하면 업데이트 공식이 어떻게 될까요? (이는  $\ell_1$ 정칙화(regularzation)라고 합니다.)
1. $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$ 입니다. 행렬에서 비슷한 공식을 찾아볼 수 있나요? (수학자들은 이를 [Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm) 이라고 합니다)
1. 학습 오류와 일반화 오류의 관계를 복습해보세요. 가중치 감쇠(weight decay), 학습 데이터셋 늘리기, 적당한 복잡도를 갖는 모델 사용하기 외에, 오버피팅(overfitting)을 다를 수 있는 방법이 어떤 것들이 있을까요?
1. 베이시안 통계에서,  prior 와  likelihood 곱을 이용해서 posterior를 구할 수 있습니다.  $p(w|x) \propto p(x|w) p(w)$.  $p(w)$ 가 정칙화(regularization)와 어떻게 동일할까요?

## QR 코드를 스캔해서 [논의하기](https://discuss.mxnet.io/t/2342)

![](../img/qr_weight-decay.svg)
