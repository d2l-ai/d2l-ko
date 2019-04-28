# 드롭아웃(dropout)

방금 우리는 가중치의 $\ell_2$ 놈을 패널티로 주는 방법으로 통계적인 모델을 정규화하는 고전적인 방법을 소개했습니다. 확률적인 용어로는 가중치들이 평균이 0인 가우시안 분포를 따른다는 과거 믿음(prior belife)를 가정하는 것으로 이 방법을 정당화할 수 있습니다. 더 직관적으로 설명하면, 가중치 값들이 많은 특성들 사이에 더 확산되도록 하고, 잠재적인 가짜 연관들에 더 적게 의존되도록 모델을 학습시킨다고 할 수 있습니다.

## 오버피팅 다시 살펴보기

샘플들 보다 더 많은 특성(feature)들이 주어지면, 선형 모델은 오버핏(overfit) 될 수 있습니다. 반면에 특성(feature) 수 보다 샘플이 더 많은 경우에는 일반적으로 선형 모델은 오버핏이 되지 않습니다. 아쉽게도, 선형 모델이 일반화를 안정적으로 하기 위해서는 그에 따른 비용이 들어갑니다. 

선형 모델들은 특성들인의 상호작용을 고려할 수 없습니다. 매 특성(feature)에 대해서, 선형 모델은 양수 또는 음수의 가중치를 할당 해야만 합니다. 선형 모델은 상황을 고려하는 유연성이 없습니다.

좀 더 공식적인 용어로 이야기하면, *편향-분산 트레이드오프(bias-variance tradeoff)*로 논의되는 일반화와 유연성의 기본적인 텐션을 보게될 것입니다. 선형 모델은 높은 편향(bias) (표현할 수 있는 함수의 개수가 적습니다)를 보이나, 분산(variance)은 낮습니다 (다른 랜덤 샘플 데이터에 대해서 비슷한 결과를 줍니다)

딥 뉴럴 네트워크는 우리를 편향-분산 스팩트램의 반대쪽으로 보냅니다. 뉴럴 네트워크는 각 특성을 독립적으로 보는 제약이 없기 때문에 유연합니다. 대신, 특성(feature)들의 그룹들에서 복잡한 상관관계를 학습할 수 있습니다. 예를 들면, 뉴럴 네트워크는 "Nigeria"와 "Western Union"이 이메일에 함께 나오면 그 이메일을 스팸으로 판단하고, "Western Union"이라는 단어가 없이 "Nigeria"가 등장하는 이메일은 스팸이 아니라고 판단할 수 있습니다.

적은 개수의 특성들을 가지고 있을 경우에도 딥 뉴럴 네트워크는 오버피팅될 수 있습니다. 2017년에 한 연구 그룹이 지금은 잘 알려진 뉴럴 네트워크의 굉장한 유연성에 대한 데모를 시연했습니다. 그들은 임의로 레이블링된 이미지들을 뉴럴 네트워크에 입력했고 (입력과 출력을 연결하는 실제 패턴이 없는 데이터들), SGD로 최적화된 뉴럴 네트워크가 학습 셋의 모든 이미지들을 완벽하게 레이블을 예측할 수 있었습니다.

이것이 무엇을 뜻하는지 생각해봅시다. 레이블들이 균일하게 할당되어 있고, 10개의 클래스가 있을 때, 어떤 분류기도 10% 이상의 정확도를 얻을 수 없습니다. 그리고 학습할 진짜 패턴이 없는 이런 경우인 경우에도, 뉴럴 네트워크는 학습 레이블들에 완벽하게 맞쳐질 수 있습니다

## 변화를 통한 견고함 <— 여기서 부터 다시

좋은 통계적인 모델로 부터 무엇을 기대할 수 있는지에 대해서 간단히 알아보겠습니다. 우리는 이 모델이 보지 않은 테스트 데이터에 대해서 잘 작동하기를 기대합니다. 이를 달성하는 방법 중에 하나로는 어떤 것이 "간단한" 모델을 만드는지를 묻는 것입니다. 단항 함수 (monomial basis function)을 사용해서 모델을 학습시키면서 언급했던 것처럼 차원의 수가 적은 것이 간단함이 될 수 있습니다. 또한 간단함은 기본이 되는 함수의 작은 놈(norm)의 형태로 만들어질 수도 있습니다. 가중치 감쇠(weight decay)와  $\ell_2$ 정규화(regularization)이 그런 예입니다. 간단함을 만드는 세번째 요소는 입력의 작은 변화에도 큰 영향을 받지 않는 함수입니다. 예를 들면, 이미지를 분류할 때, 약간의 랜덤 노이즈들을 픽셀에 추가해도 결과에 영향을 미치지 않기를 기대하는 것입니다.

*사실 이 개념은 1995년 Bishop이  [Training with Input Noise is Equivalent to Tikhonov Regularization](https://www.mitpressjournals.org/doi/10.1162/neco.1995.7.1.108) 를 증명하면서  공식화 되었습니다. 즉, 그는 부드러운 (따라서 간단한) 함수의 개념을 입력의 변화에 탄력적인 것과 연관을 시켰습니다. 2014년으로 흘러가서, 여러층을 갖는 딥 네트워크의 복잡도가 주어졌을 때, 입력에 부드러움을 강제하는 것은 꼭 다음 층들에서도 보장되지는 않습니다.  [Srivastava et al., 2014](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf) 에서 발표된 독창적인 아이디어는 Bishop의 아이디어를 네트워크의 내부층들에 적용했습니다. 이는, 학습 과정에 네트워크 연산 경로에 노이즈를 집어넣는 것입니다.*

*여기서 주요 과제는 지나친 편향(bias)을 추가하지 않으면서 어떻게 노이즈를 추가하는지 입니다. 입력  $\mathbf{x}$ 에 대해서는 노이즈를 추가하는 것은 상대적으로 간단합니다. 즉,  $\epsilon \sim \mathcal{N}(0,\sigma^2)$ 노이즈를 입력에 더한 후  $\mathbf{x}' = \mathbf{x} + \epsilon$  이 것을 학습 데이터로 사용하면 됩니다. 이렇게 했을 때 주요 특징은  $\mathbf{E}[\mathbf{x}'] = \mathbf{x}$ 을 갖는 것입니다. 하지만, 중간층들에서는 이 노이즈의 스캐일이 적절하지 않을 수 있기 때문에 이 특징을 기대하기 어렵습니다. 대안은 다음과 같이 좌표를 뒤틀어 놓는 것입니다.*

In 1995, Christopher Bishop formalized 
a form of this idea when he proved 
that [*training with input noise is equivalent to Tikhonov regularization*](https://www.mitpressjournals.org/doi/10.1162/neco.1995.7.1.108). 
In other words, he drew a clear mathematical connection 
between the requirement that a function be smooth (and thus simple),
as we discussed in the section on weight decay,
with and the requirement that it be resilient to perturbations in the input. 

Then in 2014, [Srivastava et al., 2014](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf),
developed a clever idea for how to apply Bishop's idea 
to the *internal* layers of the network, too.
Namely they proposed to inject noise into each layer of the network
before calculating the subsequent layer during training. 
They realized that when training deep network with many layers, 
enforcing smoothness just on the input-output mapping 
misses out on what is happening internally in the network.
Their proposed idea is called *dropout*, 
and it is now a standard technique
that is widely used for training neural networks. 
Throughout trainin, on each iteration,
dropout regularization consists simply of zeroing out 
some fraction (typically 50%) of the nodes in each layer
before calculating the subsequent layer.

The key challenge then is how to inject this noise 
without introducing undue statistical *bias*. 
In other words, we want to perturb the inputs 
to each layer during training
in such a way that the expected value of the layer 
is equal to the value it would have taken 
had we not introduced any noise at all.

In Bishop's case, when we are adding 
Gaussian noise to a linear model, 
this is simple:
At each training iteration, just add noise 
sampled from a distribution with mean zero
$\epsilon \sim \mathcal{N}(0,\sigma^2)$ to the input $\mathbf{x}$ , 
yielding a perturbed point $\mathbf{x}' = \mathbf{x} + \epsilon$.
In expectation, $\mathbf{E}[\mathbf{x}'] = \mathbf{x}$. 

In the case of dropout regularization,
one can debias each layer
by normalizing by the fraction of nodes that were not dropped out.
In other words, dropout with drop probability $p$ is applied as follows:
$$
\begin{aligned}
h' =
\begin{cases}
    0 & \text{ 확률 } p \text{ 인 경우}\\
    \frac{h}{1-p} & \text{ 그 외의 경우}
\end{cases}
\end{aligned}
$$

설계상으로는 기대값이 변하지 않습니다. 즉, $\mathbf{E}[h'] = h$ 입니다. 중간 레이어들에 적용되는 활성화(activation) $h$ 를 같은 기대값을 갖는 랜덤 변수  $h'$ 로 바꾸는 것이 드롭아웃(dropout)의 핵심 아이디어 입니다. '드롭아웃(dropout)' 이라는 이름은 마지막 결과를 계산하기 위해서 사용되는 연산의 몇몇 뉴런들을 누락(drop out) 시킨다는 개념에서 왔습니다. 학습 과정에서, 중간의 활성화(activation)들을 활률 변수로 바꿉니다.

## 드롭아웃(dropout) 실제 적용하기

5개의 은닉 유닛(hidden unit)을 갖는 한개의 은닉층을 사용하는 [다층 퍼셉트론(multilayer perceptron)](mlp.md) 의 예를 다시 들어보겠습니다. 이 네트워크의 아키텍처는 다음과 같이 표현됩니다.
$$
\begin{aligned}
    h & = \sigma(W_1 x + b_1) \\
    o & = W_2 h + b_2 \\
    \hat{y} & = \mathrm{softmax}(o)
\end{aligned}
$$

*은닉층에 드롭아웃(dropout)을 확률 $p$ 로 적용하는 경우, 은닉 유닛들을 $p$ 확률로 제거하는 것이 됩니다. 이유는, 그 확률을 이용해서 출력을 0으로 설정하기 때문입니다. 이를 적용한 네트워크는 아래 그림과 같습니다. 여기서  $h_2$ 와 $h_5$ 가 제거되었습니다. 결과적으로 $y$ 를 계산할 때, $h_2$ 와 $h_5$ 는 사용되지 않게 되고, 역전파(backprop)을 수행할 때 이 것들에 대한 그래디언트(gradient)들도 적용되지 않습니다. 이렇게 해서 출력층으ㄹ 계산할 때 $h_1, \ldots, h_5$ 중 어느 하나에 전적으로 의존되지 않게 합니다. 이것이 오버피팅(overfitting) 문제를 해결하는 정규화(regularization) 목적을 위해서 필요한 것입니다. 테스트 시에는, 더 확실한 결과를 얻기 위해서 드롭아웃(dropout)을 사용하지 않는 것이 일반적입니다.*

When we apply dropout to the hidden layer, 
we are essentially removing each hidden unit with probability $p$,
(i.e., setting their output to $0$). 
We can view the result as a network containing 
only a subset of the original neurons. 
In the image below, $h_2$ and $h_5$ are removed. 
Consequently, the calculation of $y$ no longer depends on $h_2$ and $h_5$ 
and their respective gradient also vanishes when performing backprop. 
In this way, the calculation of the output layer 
cannot be overly dependent on any one element of $h_1, \ldots, h_5$. 
Intuitively, deep learning researchers often explain the inutition thusly:
we do not want the network's output to depend 
too precariously on the exact activation pathway through the network. 
The original authors of the dropout technique 
described their intuition as an effort 
to prevent the *co-adaptation* of feature detectors.

![MLP before and after 드롭아웃(dropout)](../img/dropout2.svg)

At test time, we typically do not use dropout.
However, we note that there are some exceptions:
some researchers use dropout at test time as a heuristic appraoch
for estimating the *confidence* of neural network predictions:
if the predictions agree across many different dropout masks,
then we might say that the network is more confident. 
For now we will put off the advanced topic of uncertainty estimation
for subsequent chapters and volumes.

## 직접 구현하기

*드롭아웃(dropout)를 구현하기 위해서는 입력 개수 만큼의 확률 변수를 균일한 분포 $U[0,1]$ 에서 추출해야합니다. 드롭아웃(dropout)의 정의에 따르면, 이를 간단하게 구현할 수 있습니다. 다음 `드롭아웃(dropout)` 함수는 NDArray 입력 `x` 의 원소들을 `drop_prob` 확률로 누락시킵니다.* 

To implement the dropout function for a single layer, 
we must draw as many samples from a Bernoulli (binary) random variable 
as our layer has dimensions, where the random variable takes value $1$ (keep) with probability $1-p$ and $0$ (drop) with probability $p$.
One easy way to implement this is to first draw samples
from the uniform distribution $U[0,1]$.
then we can keep those nodes for which the corresponding 
sample is greater than $p$, dropping the rest. 

In the following code, we implement a `dropout` function
that drops out the elements in the NDArray input `X` 
with probability `drop_prob`, 
rescaling the remainder as described above 
(dividing the survivors by `1.0-drop_prob`).

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

*몇가지 예제에 적용해서 어떻게 동작하는지 살펴보겠습니다. 드롭아웃(dropout) 확률을 각각 0, 0.5, 그리고 1로 설정해봅니다.*

We can test out the `dropout` function on a few examples. 
In the following lines of code, we pass our input `X` 
through the dropout operation, with probabilities 0, 0.5, and 1, respectively.

```{.python .input}
X = nd.arange(16).reshape((2, 8))
print(dropout(X, 0))
print(dropout(X, 0.5))
print(dropout(X, 1))
```

## 모델 파라미터 정의하기

여기서도 다시  ["Softmax 회귀(regression)를 처음부터 구현하기"](softmax-regression-scratch.md) 절에서 사용한 Fashion-MNIST 데이터셋을 다시 사용합니다. 두개의 은닉층들을 갖는 다층 퍼셉트론(multilayer perceptron)을 정의하는데, 각 은닉층은 256개의 결과를 출력합니다.

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

정의하는 모델은 각 활성화 함수(activation function)의 결과에 드롭아웃(dropout)을 적용하면서 완전 연결층(fully connected layer)와 활성화 함수(activation function) ReLU를 연결하도록 되어 있습니다. 각 층에 서로 다른 드롭아웃(dropout) 확률을 설정할 수 있습니다. 일반적으로는 입력층에 가까울 수록 낮은 드롭아웃(dropout) 확률값을 사용하는 것을 권장합니다. 아래 모델에서는 첫번째 층에는 0.2를 두번째 층에는 0.5를 적용하고 있습니다. ["Autograd"](../chapter_prerequisite/autograd.md) 절에서 정의한 `is_training` 을 사용하면, 학습할 때만 드롭아웃(dropout) 이 적용될 수 있게 할 수 있습니다.

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

*Gluon을 이용하면, 완전 연결층(fully connected layer) 다음에 드롭아웃(dropout) 확률값을 주면서 `드롭아웃(dropout)` 층을 추가하기만 하면 됩니다. 모델을 학습시킬 때 `드롭아웃(dropout)` 층은 명시된 드롭아웃(dropout) 확률에 따라서 결과 원소들을 임의로 누락시켜주고, 테스트를 수행할 때는 데이터를 그냥 통과 시킵니다.*

Using Gluon, all we need to do is add a `Dropout` layer 
(also in the `nn` package)
after each fully-connected layer, passing in the dropout probability
as the only argument to its constructor. 
During training, the `Dropout` layer will randomly 
drop out outputs of the previous layer
(or equivalently, the inputs to the subequent layer)
according to the specified dropout probability. 
When MXNet is not in training mode, 
the `Dropout` layer simply passes the data through during testing.

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

* 차원 수를 조절하고 가중치 벡터(weight vector)의 크기를 제어하는 것 이외에, 드롭아웃(dropout)은 오버피팅(overfitting)을 해결하는 또 다른 방법입니다.  이 세가지는 종종 함께 사용됩니다.
* 드롭아웃(dropout)은  $h$ 를 같은 기대값  $h$ 를 갖는 확률 변수 $h'$ 로 드롭아웃(dropout) 확률 $p$ 만큼 바꾸는 것입니다.
* 드롭아웃(dropout)은 학습에만 적용합니다.

## 연습문제

1. 층 1과 2에서 드롭아웃(dropout) 확률값을 바꾸면서 그 결과를 관찰해보세요. 특히, 두 층에 대한 드롭아웃(dropout) 확률을 동시에 바꾸면 어떻게될까요?
1. 에포크(epoch) 수를 늘리면서 드롭아웃(dropout)을 적용할 때와 적용하지 않을 때의 결과를 비교해보세요.
1. 드롭아웃(dropout)을 적용한 후, 활성화(activation) 확률 변수의 편차를 계산해보세요.
1. 왜 일반적으로 드롭아웃(dropout)을 사용하지 않아야 하나요?
1. 은닉층 유닛(hideen layer unit)을 추가하는 것처럼 모델의 복잡도를 높이는 변경을 할때, 드롭아웃(dropout)을 사용하는 효과가 오버피팅(overfitting) 문제를 해결하는 더 확실한가요?
1. 위 예제를 이용해서 드롭아웃(dropout)과 가중치 감쇠(weight decay) 효과를 비교해보세요.
1. 활성화 결과가 아니라 가중치 행렬(weight matrix)의 각 가중치에 적용하면 어떻게 될까요?
1. $[0, \gamma/2, \gamma]$ 에서 추출한 값을 갖도록 드롭아웃(dropout)을 바꿔보세요. 이진 드롭아웃(binary dropout) 함수보다 더 좋은 것을 만들어볼 수 있나요? 왜 그런 방법을 사용할 것인가요? 왜 아닌가요?

## 참고자료

[1] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).  JMLR

## QR 코드를 스캔해서 [논의하기](https://discuss.mxnet.io/t/2343)

![](../img/qr_dropout.svg)
