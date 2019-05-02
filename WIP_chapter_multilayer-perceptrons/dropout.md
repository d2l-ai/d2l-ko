# 드롭아웃(dropout)

방금 우리는 가중치의 $\ell_2$ 놈을 패널티로 주는 방법으로 통계적인 모델을 정규화하는 고전적인 방법을 소개했습니다. 확률적인 용어로는 가중치들이 평균이 0인 가우시안 분포를 따른다는 과거 믿음(prior belife)를 가정하는 것으로 이 방법을 정당화할 수 있습니다. 더 직관적으로 설명하면, 가중치 값들이 많은 특성들 사이에 더 확산되도록 하고, 잠재적인 가짜 연관들에 더 적게 의존되도록 모델을 학습시킨다고 할 수 있습니다.

## 오버피팅 다시 살펴보기

샘플들 보다 더 많은 특성(feature)들이 주어지면, 선형 모델은 오버핏(overfit) 될 수 있습니다. 반면에 특성(feature) 수 보다 샘플이 더 많은 경우에는 일반적으로 선형 모델은 오버핏이 되지 않습니다. 아쉽게도, 선형 모델이 일반화를 안정적으로 하기 위해서는 그에 따른 비용이 들어갑니다. 

선형 모델들은 특성들인의 상호작용을 고려할 수 없습니다. 매 특성(feature)에 대해서, 선형 모델은 양수 또는 음수의 가중치를 할당 해야만 합니다. 선형 모델은 상황을 고려하는 유연성이 없습니다.

좀 더 공식적인 용어로 이야기하면, *편향-분산 트레이드오프(bias-variance tradeoff)*로 논의되는 일반화와 유연성의 기본적인 텐션을 보게될 것입니다. 선형 모델은 높은 편향(bias) (표현할 수 있는 함수의 개수가 적습니다)를 보이나, 분산(variance)은 낮습니다 (다른 랜덤 샘플 데이터에 대해서 비슷한 결과를 줍니다)

딥 뉴럴 네트워크는 우리를 편향-분산 스팩트램의 반대쪽으로 보냅니다. 뉴럴 네트워크는 각 특성을 독립적으로 보는 제약이 없기 때문에 유연합니다. 대신, 특성(feature)들의 그룹들에서 복잡한 상관관계를 학습할 수 있습니다. 예를 들면, 뉴럴 네트워크는 "Nigeria"와 "Western Union"이 이메일에 함께 나오면 그 이메일을 스팸으로 판단하고, "Western Union"이라는 단어가 없이 "Nigeria"가 등장하는 이메일은 스팸이 아니라고 판단할 수 있습니다.

적은 개수의 특성들을 가지고 있을 경우에도 딥 뉴럴 네트워크는 오버피팅될 수 있습니다. 2017년에 한 연구 그룹이 지금은 잘 알려진 뉴럴 네트워크의 굉장한 유연성에 대한 데모를 시연했습니다. 그들은 임의로 레이블링된 이미지들을 뉴럴 네트워크에 입력했고 (입력과 출력을 연결하는 실제 패턴이 없는 데이터들), SGD로 최적화된 뉴럴 네트워크가 학습 셋의 모든 이미지들을 완벽하게 레이블을 예측할 수 있었습니다.

이것이 무엇을 뜻하는지 생각해봅시다. 레이블들이 균일하게 할당되어 있고, 10개의 클래스가 있을 때, 어떤 분류기도 10% 이상의 정확도를 얻을 수 없습니다. 그리고 학습할 진짜 패턴이 없는 이런 경우에도, 뉴럴 네트워크는 학습 레이블들에 완벽하게 맞쳐질 수 있습니다

## 변화를 통한 견고함

좋은 통계적인 모델로 부터 무엇을 기대할 수 있는지에 대해서 간단히 알아보겠습니다. 우리는 이 모델이 보지 않은 테스트 데이터에 대해서 잘 작동하기를 기대합니다. 이를 달성하는 방법 중에 하나로는 어떤 것이 "간단한" 모델을 만드는지를 묻는 것입니다. 단항 함수 (monomial basis function)을 사용해서 모델을 학습시키면서 언급했던 것처럼 차원의 수가 적은 것이 간단함이 될 수 있습니다. 또한 간단함은 기본이 되는 함수의 작은 놈(norm)의 형태로 만들어질 수도 있습니다. 가중치 감쇠(weight decay)와  $\ell_2$ 정규화(regularization)이 그런 예입니다. 간단함을 만드는 세번째 요소는 입력의 작은 변화에도 큰 영향을 받지 않는 함수입니다. 예를 들면, 이미지를 분류할 때, 약간의 랜덤 노이즈들을 픽셀에 추가해도 결과에 영향을 미치지 않기를 기대하는 것입니다.

1995년에 Christopher Bishop이 [*training with input noise is equivalent to Tikhonov regularization*](https://www.mitpressjournals.org/doi/10.1162/neco.1995.7.1.108) 를 증명하면서 이 아이디어의 형태를 만들었습니다. 즉, 그는 가중치 감쇠에 대한 절에서 설명한 함수가 부드러워야 한다는 (즉, 간단한) 요건과 입력의 변화에 탄력적이어야 한다는 요구 사항의 관계에 대한 명확한 수학적인 연결을 만들었습니다.

그 후 2014년에 [Srivastava et al., 2014](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf) 은 Bishop의 아이디어를 네트워크 *내부* 층에 적용하는 독창적인 아이디어를 만들었습니다. 즉, 학습 과정에서 다음 층을 계산하기 전에 네트워크의 각 층에 노이즈를 삽입하는 것을 제안했습니다. 많은 층을 갖는 깊은 네트워크를 학습시킬 때, 입력-출력 매핑에만 부드러움(smoothness)를 강제하는 것은 네트워크에서 내부적으로 무엇이 일어나는 지를 놓치기 쉽다는 것을 깨달았습니다. 그들은 *드롭아웃(dropout)* 이라는 아이디어를 제안했고, 이것은 현재 뉴럴 네트워크를 학습시키는데 널리 사용되는 표준 기법입니다. 학습 전반의 각 반복에서 드롭아웃 정규화는 다음 층을 계산하기 전에 각 층의 노드들 중 일부를 (일반적으로는 50%) 단순히 0으로 만드는 것입니다.

이 때 가장 어려운 점은 지나친 통계적인 편향을 가져오지 않으면서도 노이즈를 어떻게 추가하는가 입니다. 즉, 학습을 수행하는 동안 노이즈를 전혀 추가하지 않은 경우의 출력된 값과 유사한 결과가 나올 수는 방법으로 각 층의 입력에 혼동을 주기를 원합니다.

Bishop의 경우에는 선형 모델에 가우시안 노이즈를 추가하는 것은 간단합니다. 매번 학습 반복마다 평균이 0인 분산으로 부터 노이즈를 샘플링, $\epsilon \sim \mathcal{N}(0,\sigma^2)$ , 한 값을 입력 $\mathbf{x}$ 에 더해서 변경된 점  $\mathbf{x}' = \mathbf{x} + \epsilon$ 를 얻습니다. 기대값은 $\mathbf{E} [\mathbf{x}'] = \mathbf{x}$ 이 됩니다.

드롭아웃 정규화의 경우에는 제거되지 않은 노드들을 표준하는 것으로 각 층을 편향을 제거할 수 있습니다. 즉, 드롭 확률 $p$ 만큼 드롭아웃을 적용하는 것은 다음과 같습니다.
$$
\begin{aligned}
h' =
\begin{cases}
    0 & \text{ 확률 } p \text{ 인 경우}\\
    \frac{h}{1-p} & \text{ 그 외의 경우}
\end{cases}
\end{aligned}
$$

설계상으로는 기대값이 변하지 않습니다. 즉, $\mathbf{E}[h'] = h$ 입니다. 중간 레이어들에 적용되는 활성화(activation) $h$ 를 같은 기대값을 갖는 랜덤 변수  $h'$ 로 바꾸는 것이 드롭아웃(dropout)의 핵심 아이디어 입니다. '드롭아웃' 이라는 이름은 마지막 결과를 계산하기 위해서 사용되는 연산의 몇몇 뉴런들을 누락(drop out) 시킨다는 개념에서 왔습니다. 학습 과정에서, 중간의 활성화(activation)들을 활률 변수로 바꿉니다.

## 드롭아웃 실제 적용하기 <— 여기부터 다시하기

5개의 은닉 유닛(hidden unit)을 갖는 한개의 은닉층을 사용하는 [다층 퍼셉트론(multilayer perceptron)](mlp.md) 의 예를 다시 들어보겠습니다. 이 네트워크의 아키텍처는 다음과 같이 표현됩니다.
$$
\begin{aligned}
    h & = \sigma(W_1 x + b_1) \\
    o & = W_2 h + b_2 \\
    \hat{y} & = \mathrm{softmax}(o)
\end{aligned}
$$

은닉층에 드롭아웃을 적용하면, 확률 $p$ 로 각 은닉 유닛들을 제거합니다. (예를 들면, 그 유닛들의 출력을 0으로 설정함). 그 결과는 원래 뉴런들의 서브셋 만을 포함한 네트워크가 됩니다. 아래 그림을 보면, $h_2$ 과  $h_5$ 가 제거된 것을 확인 할 수 있습니다. 결론적으로 $y$ 값을 계산할 때 $h_2$ 과 $h_5$ 는 사용되지 않으며, 역전파에서도 이 유닛들의 그래디언트도 사라집니다. 이런 방법을 적용해서 결과층의 계산이 $h_1, \ldots, h_5$ 중 어느 하나에 많이 의존되는 것을 방지할 수 있습니다. 딥러닝 연구자들이 종종 직관을 설명하는 것처럼, 직관적으로 보면, 이는 네트워크의 결과가 특정 활성화 경로에 너무 불안정하게 의존하지 않게하는 만드는 것입니다. 드롭아웃 기법의 원저자들은 특징 탐지기들이 *함께 적용되는 것(co-adaptation)*을 방지하려는 노력이라고 그들의 직관을 설명했습니다.

![드롭아웃적용 전, 후의 MLP](../img/dropout2.svg)

테스트를 수행할 때는 일반적으로 드롭아웃을 사용하지 않습니다. 하지만, 몇 가지 예외가 있습니다. 어떤 연구자들은 뉴럴 네트워크 예측의 *확신도* 를 추정하기 위한 경헙적 접근법으로 테스트 수행에 드롭아웃을 사용합니다: 예측이 다양한 드롭아웃 마스크들에 걸쳐서 동일하다면, 그 네트워크는 더 신뢰도가 높다고 말할 수도 있습니다. 여기서는 불확실성 예측에 대한 고급 주제는 다음 장과 책들에서 다루기로 하고 넘어가겠습니다.

## 직접 구현하기

단일층에 대한 드롭아웃 함수를 구현하기 위해서 우리는 베르누이(Bernoulli)(이진) 확률 변수로 부터 샘플을 그 층이 갖는 차원(dimension) 개수 만큼 뽑아야 합니다. 이 때, 확률 변수는 $1-p$ 확률로 1을 갖거나, $p$ 확률로 0을 갖습니다. 1은 유닛을 그대로 사용하는 것이고, 0은 드롭하는 것을 의미합니다. 이것을 구현하는 쉬운 방법 중 하나는 균등한 분포 $U[0,1]$ 로 부터 샘플들을 추출하고, 샘플값이 $p$ 보다 큰 경우는 해당 노드를 그대로 사용하고, 작은 경우에는 드롭하는 것입니다.

아래 코드에서 우리는 `dropout` 함수를 구현합니다. 이 함수는 NDArray 입력 `x` 에 대해서 확률 `drop_prob` 을 사용해서 원소들을 드롭시키고, 위에서 설명한 것처럼 남은 노드들의 값을 재조정(recale) 합니다. (남은 값들을 `1.0-drop_prob` 으로 나눕니다.)

```{.python .input}
import sys
sys.path.insert(0, '..')

import d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn

def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    # In this case, all elements are dropped out
    if drop_prob == 1:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape) > drop_prob
    return mask * X / (1.0-drop_prob)
```

몇 개의 샘플을 사용해서 `dropout` 함수를 테스트할 수 있습니다. 아래 몇 줄의 코드에서 우리는 입력 `X` 을 확률 0, 0.5 그리고 1을 사용해서 드롭아웃 연산에 적용합니다.

```{.python .input}
X = nd.arange(16).reshape((2, 8))
print(dropout(X, 0))
print(dropout(X, 0.5))
print(dropout(X, 1))
```

## 모델 파라미터 정의하기

이전과 마찬가지로 ["Softmax 회귀 처음부터 구현하기"](softmax-regression-scratch.md) 에서 소개한 Fashion-MNIST 데이터셋을 사용하겠습니다. 두 개의 은닉층을 갖는 다층 퍼셉트론을 정의할 것이고, 두 은닉층은 256개의 출력을 갖습니다. 

```{.python .input}
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens1))
b1 = nd.zeros(num_hiddens1)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens1, num_hiddens2))
b2 = nd.zeros(num_hiddens2)
W3 = nd.random.normal(scale=0.01, shape=(num_hiddens2, num_outputs))
b3 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()
```

## 모델 정의하기

정의하는 모델은 각 활성화 함수의 결과에 드롭아웃을 적용하면서 완전 연결층과 활성화 함수 ReLU를 연결하도록 되어 있습니다. 각 층에 서로 다른 드롭아웃 확률을 설정할 수 있습니다. 일반적으로는 입력층에 가까울 수록 낮은 드롭아웃 확률값을 사용하는 것을 권장합니다. 아래 모델에서는 첫번째 층에는 0.2를 두번째 층에는 0.5를 적용하고 있습니다. ["Autograd"](../chapter_prerequisite/autograd.md) 절에서 정의한 `is_training` 을 사용하면, 학습할 때만 드롭아웃이 적용될 수 있게 할 수 있습니다.

```{.python .input}
drop_prob1, drop_prob2 = 0.2, 0.5

def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = (nd.dot(X, W1) + b1).relu()
    # Use dropout only when training the model
    if autograd.is_training():
        # Add a dropout layer after the first fully connected layer
        H1 = dropout(H1, drop_prob1)
    H2 = (nd.dot(H1, W2) + b2).relu()
    if autograd.is_training():
        # Add a dropout layer after the second fully connected layer
        H2 = dropout(H2, drop_prob2)
    return nd.dot(H2, W3) + b3
```

## 학습 및 테스트

다층 퍼셉트론(multilayer perceptron)의 학습과 테스트는 이전에 설명한 것과 비슷합니다.

```{.python .input}
num_epochs, lr, batch_size = 10, 0.5, 256
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params, lr)
```

## 간결한 구현

Gluon을 사용하면, 각 완전 연결층 다음에 드롭아웃 확률값을 생성자의 유일한 변수로 지정하면서 `Dropout` 층을 추가하기만 하면 됩니다. (`Dropout` 은 `nn` 패키지에 포함되어 있습니다.) 학습하는 동안 `Dropout` 층은 이전 층의 결과(또는 동일하게 다음 층의 입력)를 정의된 드롭아웃 확률에 따라서 임의로 드롭시킵니다. MXNet이 학습 모드가 아닌 경우에는 `Dropout` 층은 테스트 수행시 데이터를 그냥 통과시킵니다.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation="relu"),
        # Add a dropout layer after the first fully connected layer
        nn.Dropout(drop_prob1),
        nn.Dense(256, activation="relu"),
        # Add a dropout layer after the second fully connected layer
        nn.Dropout(drop_prob2),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

다음으로 모델을 학습시키고 테스트를 수행합니다.

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)
```

## 요약

* 차원 수를 조절하고 가중치 벡터의 크기를 제어하는 것 이외에, 드롭아웃은 오버피팅을 해결하는 또 다른 방법입니다.  이 세가지는 종종 함께 사용됩니다.
* 드롭아웃은  $h$ 를 같은 기대값  $h$ 를 갖는 확률 변수 $h'$ 로 드롭아웃 확률 $p$ 만큼 바꾸는 것입니다.
* 드롭아웃은 학습에만 적용합니다.

## 연습문제

1. 층 1과 2에서 드롭아웃 확률값을 바꾸면서 그 결과를 관찰해보세요. 특히, 두 층에 대한 드롭아웃 확률을 동시에 바꾸면 어떻게될까요?
1. 에포크 수를 늘리면서 드롭아웃을 적용할 때와 적용하지 않을 때의 결과를 비교해보세요.
1. 드롭아웃을 적용한 후, 활성화(activation) 확률 변수의 편차를 계산해보세요.
1. 왜 일반적으로 드롭아웃을 사용하지 않아야 하나요?
1. 은닉층 유닛(hideen layer unit)을 추가하는 것처럼 모델의 복잡도를 높이는 변경을 할때, 드롭아웃을 사용하는 효과가 오버피팅(overfitting) 문제를 해결하는 더 확실한가요?
1. 위 예제를 이용해서 드롭아웃과 가중치 감쇠(weight decay) 효과를 비교해보세요.
1. 활성화 결과가 아니라 가중치 행렬(weight matrix)의 각 가중치에 적용하면 어떻게 될까요?
1. $[0, \gamma/2, \gamma]$ 에서 추출한 값을 갖도록 드롭아웃을 바꿔보세요. 이진 드롭아웃(binary dropout) 함수보다 더 좋은 것을 만들어볼 수 있나요? 왜 그런 방법을 사용할 것인가요? 왜 아닌가요?

## 참고자료

[1] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).  JMLR

## QR 코드를 스캔해서 [논의하기](https://discuss.mxnet.io/t/2343)

![](../img/qr_dropout.svg)
