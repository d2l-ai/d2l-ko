# 수치 안정성(numerical stability) 및 초기화

*지금까지 우리는 다층 퍼셉트론(multilayer perception)을 구현하는데 필요한 도구, 회귀와 분류의 문제를 어떻게 풀 수 있는지, 그리고 모델의 용량을 어떻게 제어해야하는지에 대해서 다뤘습니다. 하지만, 파라미터의 초기화는 당연한 것으로 간주하면서, 특별히 중요하지 않은 것으로 단순하게 가정했습니다. 이 절에서는 이것들에 대해서 자세히 살펴보고, 유용한 경험적 방법론에 대해서 논의하겠습니다.

두번째로 우리는 활성화 함수(activation function) 선택에 큰 관심을 두지 않았습니다. 실제로 얕은 네트워크에서는 크게 중요하지 않지만, 딥 네트워크 (deep network)에서는 비선형성과 초기화의 선택이 최적화 알고리즘을 빠르게 수렴시키는데 중요한 역할을 합니다. 이 이슈들을 중요하게 생각하지 않으면 그래디언트 소실(vanishing) 또는 폭발(exploding)이 발생할 수 있습니다.*

In the past few sections, each model that we implemented
required initializing our parameters according to some specified distribution.
However, until now, we glossed over the details,
taking the initialization hyperparameters for granted.
You might even have gotten the impression that these choices 
are not especially important.
However, the choice of initialization scheme
plays a significant role in neural network learning,
and can prove essentially to maintaining numerical stability.
Moreover, these choices can be tied up in interesting ways 
with the choice of the activation function.
Which nonlinear activation function we choose,
and how we decide to initialize our parameters
can play a crucial role in making 
the optimization algorithm converge rapidly. 
Failure to be mindful of these issues 
can lead to either exploding or vanishing gradients.
In this section, we delve into these topics with greater detail 
and discuss some useful heuristics that you may use 
frequently throughout your career in deep learning.

앞 절들에서 우리가 구현을 한 각 모델은 어떤 특정 분포에 따라서 파라미터들을 초기화하는 것을 요구했었습니다. 하지만, 지금까지 우리는 하이퍼파라미터 초기화를 당연한 것으로 여기며 자세한 설명을 하지 않았습니다. 여러분은 이런 선택들이 특별히 중요하지 않다는 인상을 받았을 수도 있습니다. 하지만, 초기화 방법의 선택은 뉴럴 네트워크 학습에 중요한 역할을 하고, 기본적으로 수치 안정성을 유지하는 것으로 증명될 수 있습니다. 더욱이, 이런 선택들은 활성화 함수의 선택과 흥미있는 방법으로 조합될 수 있습니다. 우리가 선택하는 비선형 활성화 함수와 파라미터 초기화를 어떻게 할 것인지를 결정하는 것이 최적화 알고리즘이 빠르게 수렴할 수 있도록 하는데 중요한 역할을 할 수 있습니다. 이 이슈들을 염두하지 않는다면 그래디언트 폭발이나 소실이 일어날 것입니다. 이 절에서 우리는 이 주제들에 대해서 심도있게 살펴볼 것이고, 딥러닝 경력을 쌓으면서 자주 사용하게 될 몇 몇 유용한 경험들에 대해서도 논의하겠습니다.

## 그래디언트 소실(vanishing)과 폭발(exploding)

입력이  $\mathbf{x}$ , 출력이  $\mathbf{o}$ 이고 $d$ 층을 갖는 딥 네트워크를 예로 들겠습니다. 각 층은 다음을 만족합니다.

$$\mathbf{h}^{t+1} = f_t (\mathbf{h}^t) \text{ 이고, 따라서 } \mathbf{o} = f_d \circ \ldots \circ f_1(\mathbf{x})$$

모든 활성화(activation)들과 입력들이 벡터인 경우, $t$ 번째 층의 함수 $f_t$ 와 관련된 파라미터 $\mathbf{W}_t$ 의 임의의 세트에 대한  $\mathbf{o}$ 의 그래디언트(gradient)는 다음과 같이 표현됩니다.

$$\partial_{\mathbf{W}_t} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{d-1}} \mathbf{h}^d}_{:= \mathbf{M}_d} \cdot \ldots \cdot \underbrace{\partial_{\mathbf{h}^{t}} \mathbf{h}^{t+1}}_{:= \mathbf{M}_t} \underbrace{\partial_{\mathbf{W}_t} \mathbf{h}^t}_{:= \mathbf{v}_t}.$$

다르게 말하면, 위 공식은 $d-t$ 개의 행렬  $\mathbf{M}_d \cdot \ldots \cdot \mathbf{M}_t$ 과 그래디언트(gradient) 벡터  $\mathbf{v}_t$ 의 곱입니다. 너무 많은 확률을 곱할 때 산술적인 언더플로우(underflow)를 경험할 때와 비슷한 상황이 발생합니다. 이 문제를 로그 공간으로 전환시켜서, 즉 문제를 가수(mantissa)에서 수치 표현의 지수로 이동시켜서 완화할 수 있었습니다. 처음에 행렬들 $M_t$ 은 다양한 고유값(eigenvalue)들을 갖을 것입니다. 어떤 것들은 작을 수도, 어떤 것은 클 수도 있습니다. 특히 그것들의 곱이 아주 크거나 아주 작을 수도 있습니다. 이것은 수치적인 표현의 문제일 뿐만 아니라 최적화 알고리즘이 수렴되지 않을 수 있다는 것을 의미합니다. 아주 크거나 아주 작은 그래디언트를 얻게됩니다. 그 결과로는 (i) 아모델이 못 쓰게될 정도로 파라미터들아 아주 커져버릴 만큼의 아주 큰게 되거나 (*폭발(exploding)* 그래디언트 문제), 또는 (ii) (그래디언트 *소실(vanishing)* 문제) 파라미터들이 거의 변하지 않을 정도록 아주 작은 값이 될 것입니다. 따라서, 학습 프로세스는 진척되지 않게 됩니다.

### 그래이언트 소실

그래디언트 소실의 주요 원인의 하나로는 각 층의 선형 연산들에 끼어있는 활성화 함수 $\sigma$ 에 대한 선택입니다. 역사적으로는 ([다층 퍼셉트론 절](../chapter_deep-learning-basics/mlp.md)에서 소개된) 시그모이드 함수 $(1 + \exp(-x))$ 는 한계값 함수와 비슷한 이유로 인기 있는 선택입니다.  생물학적인 뉴럴 네트워크으로 부터 영감을 얻은 초기의 인공 뉴럴 네트워크는 생물학적인 뉴럴 네트워크로 부터 영감을 얻었기 때문에, 뉴런이 활성화되거나 활성화되지 않는 뉴런에 대한 아이디어는 매력적인 것처럼 보였습니다. (사실 생물학적인 뉴런은 부분적으로 활성화되지 않습니다) 그럼 함수를 자세히 살펴보면서, 왜 이것이 그래디언트 소실에 대해서 문제가 될 수 있는지 살펴보겠습니다.

```{.python .input}
%matplotlib inline
import mxnet as mx
from mxnet import nd, autograd
from matplotlib import pyplot as plt

from mxnet import nd, autograd
x = nd.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = x.sigmoid()
y.backward()

plt.figure(figsize=(8, 4))
plt.plot(x.asnumpy(), y.asnumpy())
plt.plot(x.asnumpy(), x.grad.asnumpy())
plt.legend(['sigmoid', 'gradient'])
plt.show()
```

As we can see, the gradient of the sigmoid vanishes 
both when its inputs are large and when they are small.
Moreover, when we excecute backward propagation, due to the chain rule, 
this means that unless we are in the Goldilocks zone,
where the inputs to most of the sigmoids are in the range of, say $[-4, 4]$, 
the gradients of the overall product may vanish. 
When we have many layers, unless we are especially careful,
we are likely to find that our gradient is cut off at *some* layer. 
Before ReLUs ($\max(0,x)$) were proposed 
as an alternative to squashing functions, 
this problem used to plague deep network training. 
As a consequence, ReLUs have become 
the default choice when designing activation functions in deep networks.

보이는 것처럼 시그모이드의 그래이언트는 입력이 크거나 또는 작을 때 모두 소실됩니다. 더구나 역전파를 수행할 때 체인룰로 인해서, 시그모이드의 대부분의 입력이 $[-4, 4]$ 범위에 있지 않을 경우에는 대부분 곱에 대한 그래디언트들도 소실될 것 입니다. 여러 층을 사용할 경우 주의를 기울이지 않는다면 그래디언트가 *어떤* 층에서는 없어지는 것을 쉽게 볼 수 있습니다. ReLU ($\max(0,x)$) 가 스쿼시 함수의 대안으로 소개되기 전에는 이 문제가 딥 네트워크 학습을 괴롭혔습니다. 그 결과로 ReLU가 딥 네트워크에서 활성화 함수를 설계할 때 기본 선택이 되었습니다.

### Exploding Gradients

The opposite problem, when gradients explode, 
can be similarly vexing.
To illustrate this a bit better, 
we draw $100$ Gaussian random matrices 
and multiply them with some initial matrix. 
For the scale that we picked 
(the choice of the variance $\sigma^2=1$), 
the matrix product explodes. 
If this were to happen to us with a deep network,
we would have no realistic chance of getting 
a gradient descent optimizer to converge.

반대 문제로 그래디언트가 폭발하는 것 역시 비슷하게 성가신 문제입니다. 이를 조금 더 설명하기 위해서, 우리는 $100$ 개의 가우시안 난수 행렬을 뽑고, 이것들을 어떤 초기 행렬과 곱합니다. 우리가 선택한 스캐일 (분산을  $\sigma^2=1$ 로 선택) 때문에, 행렬 곱은 폭발합니다. 만약 이것이 딥 네트워크에서 발생한다면, 그래이언트 하강 최적화가 수렴할 현실적인 기회가 없을 것입니다.

```{.python .input  n=5}
M = nd.random.normal(shape=(4,4))
print('A single matrix', M)
for i in range(100):
    M = nd.dot(M, nd.random.normal(shape=(4,4)))

print('After multiplying 100 matrices', M)
```

### 대칭성

Another problem in deep network design 
is the symmetry inherent in their parametrization. 
Assume that we have a deep network 
with one hidden layer with two units, say $h_1$ and $h_2$. 
In this case, we could permute the weights $\mathbf{W}_1$ 
of the first layer and likewise permute the weights of the output layer
to obtain the same function function.
There is nothing special differentiating 
the first hidden unit vs the second hidden unit. 
In other words, we have permutation symmetry 
among the hidden units of each layer. 

This is more than just a theoretical nuisance. 
Imagine what would happen if we initialized 
all of the parameters of some layer as $\mathbf{W}_l = c$
for some constant $c$.  
In this case, the gradients for all dimensions are identical:
thus not only would each unit take the same value,
but it would receive the same update.
Stochastic gradient descent would never break the symmetry on its own
and we might never be able to realize the networks expressive power. 
The hidden layer would behave as if it had only a single unit.
As an aside, note that while SGD would not break this symmetry,
dropout regularization would!

—> 여기 부터 

## Parameter Initialization

One way of addressing, or at least mitigating the issues raised above 
is through careful initialization of the weight vectors. 
This way we can ensure that (at least initially) the gradients do not vanish a
and that they maintain a reasonable scale 
where the network weights do not diverge. 
Additional care during optimization and suitable regularization 
ensures that things never get too bad. 





### 기본 초기화

[“Concise Implementation of Linear Regression”](linear-regression-gluon.md) 절에서 우리는 `net.initialize(init.Normal(sigma=0.01))` 을 이용해서 가중치 값을 초기화했습니다. 만약 초기화 방법을 명시하지 않은 경우, 즉, `net.initialize()` 를 호출하는 경우에 MXNet은 기본 랜덤 초기화 방법을 적용합니다. 이는, 가중치의 각 원소는  $U[-0.07, 0.07]$ 범위의 균일 분포에서 선택된 값을 갖고, 편향(bias) 파라미터는 모두 0으로 설정됩니다. 일반적인 문제의 크기에서 이 두 방법은 상당히 잘 작동합니다.

### Xavier 초기화

어떤 층의 은닉 유닛(hidden unit) $h_{i}$에 적용된 활성화(activation)의 범위 분포를 살펴보겠습니다. 이 값들은 다음과 같이 계산됩니다.

$$h_{i} = \sum_{j=1}^{n_\mathrm{in}} W_{ij} x_j$$

가중치 $W_{ij}$ 들은 같은 분포에서 서로 독립적으로 선택됩니다. 더 나아가 이 분포는 평균이 0이고 분산이  $\sigma^2$ 라고 가정하겠습니다. (하지만, 이 분포가 가우시안(Gaussian) 이어야 한다는 것은 아니고, 단지 평균과 분산이 필요할 뿐입니다.)  층의 입력  $x_j$ 를 제어할 수 있는 방법이 없지만, 그 값들의 평균이 0이고 분산이 $\gamma^2$ 이고,  $\mathbf{W}$ 과는 독립적이라는 다소 비현실적인 가정을 하겠습니다. 이 경우,  $h_i$ 의 평균과 분산을 다음과 같이 계산할 수 있습니다.
$$
\begin{aligned}
    \mathbf{E}[h_i] & = \sum_{j=1}^{n_\mathrm{in}} \mathbf{E}[W_{ij} x_j] = 0 \\
    \mathbf{E}[h_i^2] & = \sum_{j=1}^{n_\mathrm{in}} \mathbf{E}[W^2_{ij} x^2_j] \\
        & = \sum_{j=1}^{n_\mathrm{in}} \mathbf{E}[W^2_{ij}] \mathbf{E}[x^2_j] \\
        & = n_\mathrm{in} \sigma^2 \gamma^2
\end{aligned}
$$

$n_\mathrm{in} \sigma^2 = 1$ 을 적용하면 분산을 고정시킬 수 있습니다. 이제 backpropagation을 고려해봅니다. 가장 상위층들로부터 전달되는 그래디언트(gradient) 와 함께 비슷한 문제를 만나게 됩니다. 즉,  $\mathbf{W} \mathbf{w}$ 대신에  $\mathbf{W}^\top \mathbf{g}$ 를 다뤄야합니다. 여기서 $\mathbf{g}$ 는 상위층으로부터 전달되는 그래디언트(gradient)를 의미합니다. 포워드 프로퍼게이션(forward propagation)에서와 같은 논리로,  $n_\mathrm{out} \sigma^2 = 1$ 이 아닐 경우에는 그래디언트(gradient)의 분산이 너무 커질 수 있습니다. 이 상황이 우리를 다음과 같은 딜리마에 빠지게 합니다. 즉, 우리는 두 조건을 동시에 만족시킬 수 없습니다. 대신, 다음은 조건은 쉽게 만족시킬 수 있습니다.
$$
\begin{aligned}
\frac{1}{2} (n_\mathrm{in} + n_\mathrm{out}) \sigma^2 = 1 \text{ 또는 동일하게 }
\sigma = \sqrt{\frac{2}{n_\mathrm{in} + n_\mathrm{out}}}
\end{aligned}
$$

이것이 2010년에  [Xavier Glorot and Yoshua Bengio](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) 이 제안한 Xavier 초기화의 기본이 되는 논리입니다. 이 방법은 실제로 충분이 잘 작동합니다. 가우시안(Gaussian) 확률 변수에서 Xavier 초기화는 평균이 0이고 분산이  $\sigma^2 = 2/(n_\mathrm{in} + n_\mathrm{out})$ 인 정규 분포에서 값을 선택합니다.  $U[-a, a]$ 에 균등하게 분포한 확률 변수의 경우, 분산이  $a^2/3$ 이 됩니다.  $a^2/3$ 을  $\sigma^2$ 에 대한 조건에 대입하면 다음과 같은 분포의 초기화를 할 수 있습니다.

$U\left[-\sqrt{6/(n_\mathrm{in} + n_\mathrm{out})}, \sqrt{6/(n_\mathrm{in} + n_\mathrm{out})}\right]$.

### 그 외의 것들

The reasoning above barely scratches the surface 
of modern approaches to parameter initialization. 
In fact, MXNet has an entire `mxnet.initializer` module 
implementing over a dozen different heuristics. 
Moreover, intialization continues to be a hot area of inquiry
within research into the fundamental theory of neural network optimization.
Some of these heuristics are especially suited 
for when parameters are tied 
(i.e., when parameters of in different parts the network are shared), 
for superresolution, sequence models, and related problems. 
We recommend that the interested reader take a closer look 
at what is offered as part of this module,
and investigate the recent research on parameter initialization.
Perhaps you may come across a recent clever idea 
and contribute its implementation to MXNet, 
or you may even invent your own scheme!     

## 연습문제

* 그래디언트 소멸(Vanishing gradient)와 그래디언트 폭발(exploding gradient)은 아주 깊은 네트워크에서 발생하는 흔한 문제입니다. 이를 위해서 그래디언트(gradient)와 파라미터가 잘 통제되도록 하는 것이 중요합니다.
* 초기화 방법은 최소한 초기의 그래디언트(gradient)들이 너무 커지거나 너무 작아지지 않도록 하는데 필요합니다.
* ReLU는 그래디언트 소멸(vanishing gradient) 문제 중에 하나를 해결합니다. 즉, 매우 큰 입력에 대해서 그래디언트(gradient) 가 사라지는 것을 해결합니다. 이는 수렴을 아주 빠르게 가속화해줍니다.
* 랜덤 초기화는 최적화를 수행하기 전 대칭을 깨 주는데 중요합니다.

## 문제

1. 치환 대칭성(permutation symmetry) 이외에 대칭성(symmetry)를 깨는 다른 사례를 디자인할 수 있나요?
1. 선형 회귀나 softmax 회귀에서 모든 가중치 파라미터를 같은 값으로 초기화할 수 있나요?
1. 두 행렬의 곱에서 고유값(eigenvalue)들의 해석적 범위(analytic bound)를 찾아보세요. 그래디언트(gradient)들을 잘 제어되도록 하는 것에 대해서 어떤 것을 알 수 있나요?
1. 어떤 항들의 값이 커지고 있는 것을 알게 된 경우, 이 후에 이를 고칠 수 있나요?  [You, Gitman and Ginsburg, 2017](https://arxiv.org/pdf/1708.03888.pdf) 의 LARS 논문을 참고해보세요.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2345)

![](../img/qr_numerical-stability-and-init.svg)
