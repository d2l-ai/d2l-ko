# Dropout

앞에서 우리는 통계적인 모델을 정규화(regularize)하는 전통적인 방법을 알아봤습니다. weight의 크기 ($\ell_2$ norm)을 패널티로 사용해서, weight 값이 강제로 작아지도록 했습니다. 확률적인 용어로 말하자면, Gaussian prior를 weight 값에 적용한다고 할 수 있습니다. 하지만, 더 직관적인 함수의 용어를 사용하면, weight 값들이 다른 feature들 사이에 더 확산되도록 하고, 잠재적으로 가짜 연관들에 더 적게 의존되도록 모델을 학습시킨다고 할 수 있습니다.

## 오버피팅 다시 살펴보기

뛰어난 유연성은 overfitting에 대한 책임이 따릅니다. 

샘플들 보다 더 많은 feature들이 주어지면, 선형 모델은 overfit 될 수 있습니다. 반면에 feature 수 보다 샘플이 더 많은 경우에는 선형 모델은 일반적으로 overfit 되지 않습니다. 아쉽게도, 일반화를 잘하기 위해서는 그에 따른 비용이 들어갑니다. 매 feature에 대해서, 선형 모델은 양수 또는 음수의 weight을 할당 해야합니다. 선형 모델은 feature들 사이의 미묘한 상호작용을 설명하지 못 합니다. 좀 더 공식적인 용어로 이야기하면, bias-variance tradeoff 로 논의되는 현상을 볼 것입니다. Linear 모델은 높은 bias (표현할 수 있는 함수의 개수가 적습니다)를 보이나, variance 는 낮습니다 (다른 랜덤 샘플 데이터에 대해서 비슷한 결과를 줍니다)

반면에 딥 뉴럴 네트워크는 bias-variance 스팩트럼에서 반대의 현상을 보입니다. 뉴럴 네트워크는 각 feature를 독립적으로 보는 제약이 없기 때문에 유연합니다. 대신, feature들의 그룹들에서 복잡한 상관관계를 학습할 수 있습니다. 예를 들면, 뉴럴 네트워크는 "Nigeria"와 "Western Union"이 이메일에 함께 나오면 그 이메일을 스팸으로 판단하고, "Western Union"이라는 단어가 없이 "Nigeria"가 등장하는 이메일은 스팸이 아니라고 판단할 수 있습니다.

Feature들의 개수가 적은 경우에도, 딥 뉴럴 네트워크는 overfitting 될 수 있습니다. 뉴럴 네트워크의 뛰어난 유연성을 보여주는 예로, 연구자들은 임의로 label이 할당된 데이터를 완벽하게 분류하는 것을 입증했습니다. 이것이 무엇을 뜻하는지 생각해봅시다. 10개의 분류로 된 Label들이 균일하게 임의로 부여되어 있는 경우, 어떤 분류기도 10% 이상의 정확도를 얻을 수 없습니다. 이렇게 패턴을 학습할 수 없는 경우에도, 뉴럴 네트워크는 학습 label들에 완벽하게 맞춰질 수 있습니다.

## 변화를 통한 견고함

좋은 통계적인 모델로 부터 무엇을 기대할 수 있는지에 대해서 간단히 알아보겠습니다. 당연하게 우리는 이 모델이 보지 않은 테스트 데이터에 대해서 잘 작동하기를 기대합니다. 이를 달성하는 방법 중에 하나로 어떤 것이 "간단한" 모델을 만드는지를 묻는 것입니다. 단항 함수 (monomial basis function)을 사용해서 모델을 학습시키면서 언급했던 것처럼 차원의 수가 적은 것으로부터 간단함이 유도될 수 있습니다. 또한 간단함은 기본이 되는 함수의 작은 norm의 형태로 만들어질 수도 있습니다. 즉, weight decay와  $\ell_2$ regularization이 그런 예입니다. 간단함을 만드는 또 다른 요소는 입력의 완만한 변화에도 큰 영향을 받지 않는 함수를 들 수 있습니다. 예를 들어 이미지를 분류할 때, 몇개의 픽셀들의 변경으로 인해서 결과에 영향을 미치지 않기를 기대하는 것입니다.

사실 이 개념은 1995년 Bishop이  [Training with Input Noise is Equivalent to Tikhonov Regularization](https://www.mitpressjournals.org/doi/10.1162/neco.1995.7.1.108) 를 증명하면서  공식화 되었습니다. 즉, 그는 부드러운 (따라서 간단한) 함수의 개념을 입력의 변화에 탄력적인 것과 연관을 시켰습니다. 2014년으로 흘러가서, 여러 레이어를 갖는 딥 네트워크의 복잡도가 주어졌을 때, 입력에 부드러움을 강제하는 것은 꼭 다음 레이어들에서도 보장되지는 않습니다.  [Srivastava et al., 2014](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf) 에서 발표된 독창적인 아이디어는 Bishop의 아이디어를 네트워크의 내부 레이어들에 적용했습니다. 이는, 학습 과정에 네트워크 연산 경로에 노이즈를 집어넣는 것입니다.

여기서 주요 과제는 지나친 bias를 추가하지 않으면서 어떻게 노이즈를 추가하는지 입니다. 입력  $\mathbf{x}$ 에 대해서는 노이즈를 추가하는 것은 상대적으로 간단합니다. 즉,  $\epsilon \sim \mathcal{N}(0,\sigma^2)$ 노이즈를 입력에 더한 후  $\mathbf{x}' = \mathbf{x} + \epsilon$  이 것을 학습 데이터로 사용하면 됩니다. 이렇게 했을 때 주요 특징은  $\mathbf{E}[\mathbf{x}'] = \mathbf{x}$ 을 갖는 것입니다. 하지만, 중간 레이어들에서는 이 노이즈의 스캐일이 적절하지 않을 수 있기 때문에 이 특징을 기대하기 어렵습니다. 대안은 다음과 같이 좌표를 뒤틀어 놓는 것입니다.
$$
\begin{aligned}
h' =
\begin{cases}
    0 & \text{ with probability } p \\
    \frac{h}{1-p} & \text{ otherwise}
\end{cases}
\end{aligned}
$$

설계상으로는 기대값이 변화지 않습니다. 즉, $\mathbf{E}[h'] = h$ 입니다. 중간 레이어들에 적용되는 activation $h$ 를 같은 기대값을 갖는 랜덤 변수  $h'$ 로 바꾸는 것이 dropout의 핵심 아이디어 입니다. 'dropout' 이라는 이름은 마지막 결과를 계산하기 위해서 사용되는 연산의 몇몇 뉴런들을 누락 (drop out) 시킨다는 개념에서 왔습니다. 학습 과정에서, 중간의 activation들을 랜덤 변수로 바꿉니다.

## Dropout 실제 적용하기

5개의 hidden unit을 갖는 한개의 hidden 레이어를 사용하는 [multilayer perceptron](mlp.md) 의 예를 다시 들어보겠습니다. 이 네트워크의 아키텍처는 다음과 같이 표현됩니다.
$$
\begin{aligned}
    h & = \sigma(W_1 x + b_1) \\
    o & = W_2 h + b_2 \\
    \hat{y} & = \mathrm{softmax}(o)
\end{aligned}
$$

Hidden 레이어에 dropout을 확률 $p$ 로 적용하는 경우, hidden unit들을 $p$ 확률로 제거하는 것이 됩니다. 이유는, 그 확률을 이용해서 output을 0으로 설정하기 때문입니다. 이를 적용한 네트워크는 아래 그림과 같습니다. 여기서  $h_2$ 와 $h_5$ 가 제거되었습니다. 결과적으로 $y$ 를 계산할 때, $h_2$ 와 $h_5$ 는 사용되지 않게 되고, backprop을 수행할 때 이 것들에 대한 gradient들도 적용되지 않습니다. 이렇게 해서 output 레이어를 계산할 때 $h_1, \ldots, h_5$ 중 어느 하나에 전적으로 의존되지 않게 합니다. 이것이 overfitting 문제를 해결하는 정규화(regularization) 목적을 위해서 필요한 것입니다. 테스트 시에는, 더 확실한 결과를 얻기 위해서 dropout을 사용하지 않는 것이 일반적입니다.

![MLP before and after dropout](../img/dropout2.svg)

## 직접 구현하기

Dropout를 구현하기 위해서는 입력 개수 만큼의 랜덤 변수를 균일한 분포 $U[0,1]$ 에서 추출 해야합니다. Dropout의 정의에 따르면, 이를 간단하게 구현할 수 있습니다. 다음 `dropout` 함수는 NDArray 입력 `x` 의 원소들을 `drop_prob` 확률로 누락시킵니다. 

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

몇가지 예제에 적용해서 어떻게 동작하는지 살펴보겠습니다. dropout 확률을 각각 0, 0.5, 그리고 1로 설정해봅니다.

```{.python .input}
X = nd.arange(16).reshape((2, 8))
print(dropout(X, 0))
print(dropout(X, 0.5))
print(dropout(X, 1))
```

## 모델 파라메터 정의하기

 ["Softmax Regression - Starting From Scratch"](softmax-regression-scratch.md) 절에서 사용한 Fashion-MNIST 데이터셋을 다시 사용합니다. 두개의 hidden 레이어들을 갖는 multilayer perceptron을 정의하는데, 각 hidden 레이어는 256개의 output을 출력합니다.

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

정의하는 모델은 각 activation 함수의 결과에 droput을 적용하면서 fully connected 레이어와 activation 함수ReLU를 연결하도록 되어 있습니다. 각 레이어에 서로 다른 dropout 확률을 설정할 수 있습니다. 일반적으로는 입력 레이어에 가까울 수록 낮은 dropout 확률값을 사용하는 것을 권장합니다. 아래 모델에서는 첫번째 레이어에는 0.2를 두번째 레이어에는 0.5를 적용하고 있습니다. ["Autograd"](../chapter_prerequisite/autograd.md) 절에서 정의한 `is_training` 을 사용하면, 학습할 때만 dropout 이 적용될 수 있게 할 수 있습니다.

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

Multilayer perceptron의 학습과 테스트는 이전에 설명한 것과 비슷합니다.

```{.python .input}
num_epochs, lr, batch_size = 10, 0.5, 256
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params, lr)
```

## 간결한 구현

Gluon을 이용하면, fully connected 레이어 다음에 dropout 확률값을 주면서 `Dropout` 레이어를 추가하기만 하면 됩니다. 모델을 학습시킬 때 `Dropout` 레이어는 명시된 dropout 확률에 따라서 결과 원소들을 임의로 누락시켜주고, 테스트를 수행할 때는 데이터를 그냥 통과 시킵니다.

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

* 차원 수를 조절하고 weight vector의 크기를 제어하는 것 이외에, dropout은 overfitting을 해결하는 또 다른 방법입니다.  이 세가지는 종종 함께 사용됩니다.
* Dropout은  $h$ 를 같은 기대값  $h$ 를 갖는 랜덤 변수 $h'$ 로 dropout 확률 $p$ 만큼 바꾸는 것입니다.
* Dropout은 학습에만 적용합니다.

## 문제

1. Layer 1과 2에서 dropout 확률값을 바꾸면서 그 결과를 관찰해보세요. 특히, 두 layer에 대한 dropout 확률을 동시에 바꾸면 어떻게될까요?
1. epoch 수를 늘리면서 dropout을 적용할 때와 적용하지 않을 때의 결과를 비교해보세요.
1. Dropout을 적용한 후, activation 랜덤 변수의 편차를 계산해보세요.
1. 왜 일반적으로 dropout을 사용하지 않아야 하나요?
1. Hideen layer unit을 추가하는 것처럼 모델의 복잡도를 높이는 변경을 할때, dropout을 사용하는 효과가 overfitting 문제를 해결하는 더 확실한가요?
1. 위 예제를 이용해서 dropout과 weight decay 효과를 비교해보세요.
1. Activation 결과가 아니라 weight matrix의 각 weight에 적용하면 어떻게 될까요?
1. $[0, \gamma/2, \gamma]$ 에서 추출한 값을 갖도록 dropout을 바꿔보세요. binary dropout 함수보다 더 좋은 것을 만들어볼 수 있나요? 왜 그런 방법을 사용할 것인가요? 왜 아닌가요?

## References

[1] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).  JMLR

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2343)

![](../img/qr_dropout.svg)
