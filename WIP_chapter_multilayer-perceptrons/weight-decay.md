# 가중치 감쇠 (weight decay)

*앞 절에서 우리는 오버피팅(overfitting)에 대해서 알아봤고, 이를 해결하기 위해서 용량 제어(capacity control)의 필요성에 대해서도 이야기했습니다. 학습 데이터셋의 양을 늘리는 것은 오버피팅(overfitting) 문제를 해결할 수도 있지만, 학습 데이터를 추가로 확보하는 것은 일반적으로 어려운 일입니다. 그렇기 때문에, 사용하는 함수의 복잡도를 조정하는 것을 더 선호합니다. 구체적으로는 차수를 조정해서 다항식의 복잡도를 조절할 수 있는 것을 확인했습니다. 이 방법은 일차원 데이터를 다루는 문제에 대해서는 좋은 전략이 될 수 있지만, 이 방법은 쉽게 복잡해지기 때문에 관리가 어려워질 수 있고, 너무 투박한 방법입니다. 예를 들면, $D$ 차원 벡터의 경우, $d$  차수에 대한 단항의 개수는  ${D -1 + d} \choose {D-1}$ 가 됩니다. 따라서, 여러 함수에 대한 제어를 하는 것보다는 함수의 복잡도를 조절할 수 있는 보다 정교한 툴이 필요합니다.*

Now that we have characterized the problem of overfitting 
and motivated the need for capacity control,
we can begin discussing some of the popular techniques
used to these ends in practice.
Recall that we can always mitigate overfitting 
by going out and collecting more training data,
that can be costly and time consuming,
typically making it impossible in the short run.
For now, let's assume that we have already obtained 
as much high-quality data as our resources permit
and focus on techniques aimed at limiting the capacity 
of the fuction classes under consideration. 

오버피팅 문제가 어떤 특징을 가지고 있는지 확인했고 용량 제어가 왜 필요한지를 살펴봤으니, 이제 우리는 이를 해결하기 위해서 사용되는 유명한 기법들을 알아보겠습니다. 더 많은 학습 데이터를 사용하면 오버피팅 문제는 늘 해결 할 수 있지만, 데이터를 더 수집하는 것은 비용이 들고 시간이 걸리기 때문에 일반적으로 짧은 기간에 이뤄질 수 없습니다. 지금은 우리의 리소스를 최대한 사용해서 높은 품질의 데이터를 이미 얻었다고 가정하고, 사용할 함수 클래스들의 용량을 제한하는 기법들에 집중하겠습니다.

In our toy example, 
we saw that we could control the complexity of a polynomial 
by adjusting its degree. 
However, most of machine learning 
does not consist of polynomial curve fitting.
And moreover, even when we focus on polynomial regression,
when we deal with high-dimensional data,
manipulating model capacity by tweaking the degree $d$ is problematic.
To see why, note that for multivariate data
we must generalize the concept of polynomials 
to include *monomials*, which are simply
products of powers of variables.
For example, $x_1^2 x_2$, and $x_3 x_5^2$ are both monomials of degree $3$.
The number of such terms with a given degree $d$
blows up as a function of the degree $d$.

앞에서 살펴본 간단한 예제에서 다항식의 복잡도는 차원은 차수을 조정하는 것으로 제어할 수 있는 것을 확인했습니다. 하지만, 대부분의 머신러닝은 다항 곡선으로 구성되어 있지 않습니다. 그리고 더구나 다항식 회귀을 다루는 경우에도, 고차원 데이터를 다룰 때는 차수 $d$ 를 조정하면서 모델의 용량을 바꾸는 것은 문제가 있습니다. 그 이유 설명하기 위해서, 다중 변수 데이터에서는 단순히 변수들의 제곱들의 곱인 *모노미얼(monomial)* 들을 포함하는 다항식의 개념을 일반화를 해야한다는 것을 주목하세요. 예를 들어, $x_1^2 x_2$ 와 $x_3 x_5^2$ 는 차수가 3인 모노미얼입니다. 어떤 차수 $d$ 의 항들의 개수는 차수 $d$ 의 함수에 따라서 급격히 증가합니다.

Concretely, for vectors of dimensionality $D$,
the number of monomials of a given degree $d$ is ${D -1 + d} \choose {D-1}$.
Hence, a small change in degree, even from say $1$ to $2$ or $2$ to $3$ 
would entail a massive blowup in the complexity of our model.
Thus, tweaking the degree is too blunt a hammer.
Instead, we need a more fine-grained tool 
for adjusting function complexity.

구체적으로 차원이 $D$ 인 벡터에 대해서 어떤 차수 $d$ 의 모노미얼의 개수는 ${D -1 + d} \choose {D-1}$ 입니다. 따라서, 1에서 2 또는 2에서 3과 같은 차수의 작은 변화는 모델의 복잡도에 큰 증가를 가져옵니다. 즉, 차수를 조정하는 것은 너무 무딘 망치에 불과합니다. 대신 우리는 함수의 복잡도를 조정하는 보다 세밀한 도구가 필요합니다.

## 제곱 놈 정규화(squared norm regularization)

*가장 많이 사용하는 기법 중에 하나로 가중치 감쇠(weight decay)가 있습니다. 이 방법은 모든 함수 $f$ 들 중에서 $f=0$ 이 가장 간단한 형태라는 것에 착안하고 있습니다. 따라서, 0와 얼마나 가까운 가를 이용해서 함수에 대한 측정을 할 수 있습니다. 이를 측정하는 방법은 다양한데 별도의 수학 분야가 존재하기까지 합니다. 예를 들면, 이 문제에 대한 답을 찾는 것에 목적을 두고 있는 함수 분석과 Banach 공간 이론 (the theory of Banach spaces)를 들 수 있습니다.*

*우리의 목적을 위해서는 아주 간단한 것을 사용해도 충분합니다:*

*선형 함수 $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$  에서 가중치 벡터(weight vector)가 작을 경우 ''이 함수는 간단하다''라고 간주합니다. 이것은  $\|\mathbf{w}\|^2$ 로 측정될 수 있습니다. 가중치 벡터(weight vector)를 작게 유지하는 방법 중에 하나는 손실(loss)을 최소화하는 문제에 이 값을 패널티(penalty)로 더하는 것입니다. 이렇게 하면, 가중치 벡터(weight vector)가 너무 커지면, 학습 알고리즘은 학습 오류를 최소화하는 것보다 $\mathbf{w}$ 를 최소화하는데 우선 순위를 둘 것입니다. 이것이 바로 우리가 원하는 것입니다. 코드에서 이를 설명하기 위해서,  앞 절의 [“Linear Regression”](linear-regression.md) 를 고려해보면, 손실(loss)은 다음과 같이 주어집니다.*

*Weight decay* (commonly called *L2* regularization), 
might be the most widely-used technique 
for regularizing parametric machine learning models.
The basic intuition behind weight decay is 
the notion that among all functions $f$, 
the function $f = 0$ is the simplest. 
Intuitively, we can then measure functions by their proximity to zero. 
But how precisely should we measure
the distance between a function and zero? 
There is no single right answer.
In fact, entire branches of mathematics, 
e.g. in functional analysis and the theory of Banach spaces
are devoted to answering this issue.

일반적으로 L2 정규화라고 불리는 가중치 감쇠(weight decay)는 파라메터 기반의 머신러닝 모델을 정규화 하는데 가장 널리 사용되는 기술입니다. 가중치 감쇠에 대한 기본적인 직관은 모든 함수 $f$ 중에서 함수 $f = 0$ 가 가장 단순하다는 생각에 착안하고 있습니다. 직관적으로 설명하면 우리는 0과 가까운 정도로 함수를 측정할 수 있습니다. 하지만 어떻게 함수와 0 사이의 거리를 정확하게 측정해야 할까요? 정답은 없습니다. 사실, 이를 측정하는 방법은 다양한데 별도의 수학 분야가 존재하기까지 합니다. 예를 들면, 이 문제에 대한 답을 찾는 것에 목적을 두고 있는 함수 해석학과 바나흐 공간 이론 (the theory of Banach spaces)이 있습니다.

For our present purposes, a very simple interpretation will suffice:
We will consider a linear function 
$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$ 
to be simple if its weight vector is small. 
We can measure this via $||\mathbf{w}||^2$. 
One way of keeping the weight vector small 
is to add its norm as a penalty term 
to the problem of minimizing the loss. 
Thus we replace our original objective, 
*minimize the prediction error on the training labels*,
with new objective,
*minimize the sum of the prediction error and the penalty term*.
Now, if the weight vector becomes too large, 
our learning algorithm will find more profit in
minimizing the norm $|| \mathbf{w} ||^2$ 
versus minimizing the training error. 
That's exactly what we want. 
To illustrate things in code, let's revive our previous example
from our chapter on [Linear Regression](linear-regression.md). 
There, our loss was given by

$$l(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

우리의 현재 목적을 위해서는 아주 간단한 형태면 충분합니다. 우리는 가중치 벡터가 작다면 선형 함수 $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$ 가 간단하다고 간주하겠습니다. 우리는 이것을 $||\mathbf{w}||^2$ 로 측정할 수 있습니다. 가중치 벡터를 작게 유지하는 방법 중에 하나는 손실(loss)를 최소화하는 문제에 놈을 패널티 항목으로 더하는 것입니다. 즉, 우리는 원리 목적인 *학습 레이블에 대한 예측 오류를 최소화* 하는 것에서 새로운 목적 *예측 오류와 패널티 항목의 함을 최소화* 하는 문제로 바꿉니다. 이제 가중치 벡터가 너무 커지면 우리의 학습 알고리즘은  학습 오류를 최소화하는 것 보다는 놈 $|| \mathbf{w} ||^2$ 을 최소화하는 방법을 찾을 것입니다. 이것이 바로 우리가 원하는 것입니다. 코드로 설명하기 위해서  [선형 회귀](linear-regression.md) 장에서 다룬 예제를 다시 보겠습니다. 그 예제에서 손실은 $$l(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2$$  로 정의되었습니다.

*위 수식에서 $\mathbf{x}^{(i)}$ 는 관찰들이고,  $y^{(i)}$ 는 label, $(\mathbf{w}, b)$ 는 가중치와 편향(bias) 파라미터들입니다. 가중치 벡터(weight vector)의 크기에 대한 패널티를 주는 새로운 손실 함수(loss function)를 만들기 위해서,  $\|\mathbf{w}\|^2$ 를 더합니다. 하지만, 얼마나 더해야 할까요? 이를 조절하는 정규화 상수(regularization constant)인  $\lambda$  하이퍼파라미터(hyperparameter)가 그 역할을 합니다.*

Recall that $\mathbf{x}^{(i)}$ are the observations, 
$y^{(i)}$ are labels, and $(\mathbf{w}, b)$ 
are the weight and bias parameters respectively. 
To arrive at a new loss function 
that penalizes the size of the weight vector, 
we need to add $||mathbf{w}||^2$, but how much should we add? 
To address this, we need to add a new hyperparameter,
that we will call the *regularization constant* and denote by $\lambda$:

$$l(\mathbf{w}, b) + \frac{\lambda}{2} \|\boldsymbol{w}\|^2$$

—> 여기부터 다시 !!

*$\lambda \geq 0$  는 정규화(regularzation)의 정도를 조절합니다.  $\lambda = 0$ 인 경우, 원래의 손실 함수(loss function)가 되고,  $\lambda > 0$ 이면,  $\mathbf{w}$ 가 너무 커지지 않도록 강제합니다. 통찰력이 있는 분은 가중치 벡터(weight vector)를 왜 제곱을 하는지 의아해할 것입니다. 이는 두가지 이유 때문인데, 하나는 미분 계산이 쉬워지기 때문에 연산의 편의성을 위함이고, 다른 하나는 작은 가중치 벡터(weight vector)들 보다 큰 가중치 벡터(weight vector)에 더 많은 패널티를 부여하는 것으로 통계적인 성능 향상을 얻기 위하는 것입니다. 확률적 경사 하강법(Stochastic gradient descent) 업데이트는 다음과 같이 이뤄집니다.*

This non-negatice parameter $\lambda \geq 0$ 
governs the amount of regularization. 
For $\lambda = 0$, we recover our original loss function, 
whereas for $\lambda > 0$ we ensure that $\mathbf{w}$ cannot grow too large. The astute reader might wonder why we are squaring 
the norm of the weight vector. 
We do this for two reasons.
First, we do it for computational convenience.
By squaring the L2 norm, we remove the square root,
leaving the sum of squares of each component of the weight vector.
This is convenient because it is easy to compute derivatives of a sum of terms (the sum of derivatives equals the derivative of the sum). 

Moreover, you might ask, why the L2 norm in the first place and not the L1 norm, or some other distance function.
In fact, several other choices are valid 
and are popular throughout statistics.
While L2-regularized linear models constitute 
the classic *ridge regression* algorithm
L1-regularizaed linear regression 
is a similarly fundamental model in statistics 
popularly known as *lasso regression*.

One mathematical reason for working with the L2 norm and not some other norm,
is that it penalizes large components of the weight vector
much more than it penalizes small ones. 
This encourages our learning algorithm to discover models 
which distribute their weight across a larger number of features,
which might make them more robust in practice 
since they do not depend precariously on a single feature.
The stochastic gradient descent updates for L2-regularied regression
are as follows:
$$
\begin{aligned}
w & \leftarrow \left(1- \frac{\eta\lambda}{|\mathcal{B}|} \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),
\end{aligned}
$$

*이전과 같이, 관찰된 값과 예측된 값의 차이에 따라서  $\mathbf{w}$ 를 업데이트합니다. 하지만,  $\mathbf{w}$ 의 크기를  $0$ 과 가까워지게 줄이고 있습니다. 즉, 가중치를 감쇠하게(decay) 만듭니다.* As before, we update $\mathbf{w}$ based on the amount 
by which our estimate differs from the observation. 
However, we also shrink the size of $\mathbf{w}$ towards $0$.
That's why the method is sometimes called "weight decay":
because the penalty term literally causes our optimization algorthm 
to *decay* the magnitude of the weight at each step of training. 이것은 다항식에 파라미터 개수를 선택하는 것보다 더 편한 방법입니다. 특히, $f$ 의 복잡도를 조절하는 연속성이 있는 방법을 갖게 되었습니다. 작은 $\lambda$ 값은  $\mathbf{w}$ 를 적게 제약하는 반면, 큰 값은  $\mathbf{w}$ 를 많이 제약합니다. 편향(bias) 항 역시 큰 값을 갖기를 원하지 않기 때문에,  $b^2$ 를 패널티로 더하기도 합니다.

## 고차원 선형 회귀

고차원 회귀(regression)에서 생략할 정확한 차원을 선택하기 어려운데, 가중치 감쇠 정규화(weight-decay regularization)는 아주 간편한 대안이 됩니다. 왜 그런지를 지금부터 설명하겠습니다.. 우선, 아래 공식을 사용해서 데이터를 생성합니다.

$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.01)$$

*즉, 이 식에서는 평균이 0이고 표준편차가 0.01인 가우시안(Gaussian) 노이즈를 추가했습니다. 오버피팅(overfitting)을 더 잘 재현하기 위해서, 차원 $d$ 가 200인 고차원 문제를 선택하고, 적은 양의 학습 데이터 (20개)를 사용하겠습니다. 이전과 같이 필요한 패키지를 import 합니다.*

representing our label as a linear function of our inputs,
corrupted by Gaussian noise with zero mean and variance 0.01. 
To observe the effects of overfitting more easily,
we can make our problem high-dimensional,
setting the data dimension to $d = 200$ 
and working with a relatively small number of training examples—here we'll set the sample size to 20:

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

*다음으로는 가중치 감쇠(weight decay)를 직접 구현해보겠습니다. 이를 위해서, 간단하게 타켓(target) 함수 다음에  $\ell_2$ 패널티를 추가 손실 항목으로 더합니다. 제곱 놈(squared norm) 패널티라는 이름은 제곱수를 더하는 것,  $\sum_i x_i^2$, 으로 부터 왔습니다. 이 외에도 여러가지 패널티들이 있습니다.  $\ell_p$ 놈(norm) 은 다음과 같이 정의됩니다.*

Next, we will show how to implement weight decay from scratch. 
All we have to do here is to add the squared $\ell_2$ penalty 
as an additional loss term added to the original target function. 
The squared norm penalty derives its name from the fact 
that we are adding the second power $\sum_i w_i^2$. 
The $\ell_2$ is just one among an infinite class of norms call p-norms,
many of which you might encounter in the future.
In general, for some number $p$, the $\ell_p$ norm is defined as

$\|\mathbf{w}\|_p^p := \sum_{i=1}^d |w_i|^p$

### 파라미터 초기화하기

*우선 모델 파라미터를 임의로 초기화하는 함수를 정의합니다. 이 함수는 각 파라미터에 그래디언트(gradient)를 붙입니다.*

First, we'll define a function to randomly initialize our model parameters and run `attach_grad` on each to allocate memory for the gradients we will calculate.

```{.python .input  n=5}
def init_params():
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    return [w, b]
```

### $\ell_2$ 놈 페널티(Norm Penalty) 정의하기

*이 페널티를 정의하는 간단한 방법은 각 항을 모두 제곱하고 이를 더하는 것입니다. 수식이 멋지고 간단하게 보이기 위해서 2로 나눕니다.*

Perhaps the most convenient way to implement this penalty 
is to square all terms in place and summ them up. 
We divide by $2$ by convention
(when we take the derivative of a quadratic function,
the $2$ and $1/2$ cancel out, ensuring that the expression 
for the update looks nice and simple).

```{.python .input  n=6}
def l2_penalty(w):
    return (w**2).sum() / 2
```

### 학습 및 테스트 정의하기

*아래 코드는 학습 데이터셋과 테스트 데이터셋을 이용해서 모델을 학습시키고 테스트하는 함수를 정의합니다. 이전 절의 예와는 다르게, 여기서는  $\ell_2$ 놈 패널티(norm penalty)를 최종 손실 함수(loss function)를 계산할 때 더합니다. 선형 네트워크와 제곱 손실(squared loss)은 이전과 같기 때문에, `d2l.linreg` 와 `d2l.squared_loss` 를 import 해서 사용하겠습니다.*

The following code defines how to train and test the model 
separately on the training data set and the test data set. 
Unlike the previous sections, here, the $\ell_2$ norm penalty term 
is added when calculating the final loss function. 
The linear network and the squared loss 
haven't changed since the previous chapter, 
so we'll just import them via `d2l.linreg` and `d2l.squared_loss` 

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

### 정규화(regularization) 없이 학습하기

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

*Gluon에는 최적화 알고리즘에 가중치 감쇠(weight decay)가 통합되어 있어 더 편하게 적용할 수 있습니다. 그 이유는 옵티마이져(optimizer)가 모든 파라미터를 직접 다루기 때문에, 옵티마이져(optimizer)가 가중치 감쇠(weight decay)를 직접 관리하고, 관련된 것을 최적화 알고리즘에서 다루는 것이 실행 속도면에서 더 빠르기 때문입니다.*

*아래 예제에서는 `Trainer`  인스턴스를 생성할 때, `wd` 파타메터를 통해서 가중치 감쇠(weight decay) 하이퍼파라미터(hyperparameter)를 직접 지정합니다. Gluon의 기본 설정은 가중치와 편향(bias)를 모두 감쇠(decay) 시킵니다. 다른 종류의 파라미터에 대해서 다른 옵티마이져(optimizer)를 사용할 수 있습니다. 예를 들면,  $\mathbf{w}$ 에는 가중치 감쇠(weight decay) 적용하는 `Trainer` 를 하나 만들고,  $b$  에는 가중치 감쇠(weight decay)를 적용하지 않은 다른 `Trainer` 를 각각 만들 수 있습니다.*

Because weight decay is ubiquitous in neural network optimization,
Gluon makes it especially convenient,
integrating weight decay into the optimization algorithm itself
for easy use in combination with any loss function. 
Moreover, this integration serves a computational benefit,
allowing implementation tricks to add weight decay to the algorithm,
without any additional computational overhead.
Since the weight decay portion of the update 
depdends only on the current value of each parameter,
and the optimizer must to touch each parameter once anyway.

In the following code, we specify 
the weight decay hyper-parameter directly 
through the `wd` parameter when instantiating our `Trainer`. 
By default, Gluon decays both weights and biases simultaneously. 
Note that we can have *different* optimizers 
for different sets of parameters. 
For instance, we can have one `Trainer` 
with weight decay for the weights $\mathbf{w}$ 
and another without weight decay to take care of the bias $b$.

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

*지금까지 우리는 간단한 선형 함수를 구성하는 것들만을 다뤘습니다. 비선형 함수에 대해서 이것들을 다루는 것은 훨씬 더 복잡합니다. 예를 들어,  [Reproducing Kernel Hilbert Spaces](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) 라는 것이 있는데, 이를 이용하면 선형 함수에서 사용한 많은 도구들을 비선형에서 사용할 수 있게 해줍니다. 하지만 안타깝게도, 사용되는 알고리즘들이 데이터가 매우 많은 경우 잘 동작하지 않는 확장성 문제가 있습니다. 따라서, 이 책의 목적을 위해서 우리는 각 층의 가중치들을 단순히 더하는 방법, $\sum_l \|\mathbf{w}_l\|^2$ 을 사용하겠습니다. 이렇게 하는 것은 전체 층들에 가중치 감쇠(weight decay)를 적용하는 것과 같습니다.*

So far, we only touched upon one notion of 
what constitutes a simple *linear* function. 
For nonlinear functions, what constitutes *simplicity* 
can be a far more complex question. 
For instance, there exist [Reproducing Kernel Hilbert Spaces (RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space)
which allow one to use many of the tools 
introduced for linear functions in a nonlinear context. 
Unfortunately, RKHS-based algorithms 
do not always scale well to massive amounts of data. 
For the purposes of this book, we limit ourselves 
to simply summing over the weights for different layers, 
e.g. via $\sum_l \|\mathbf{w}_l\|^2$, 
which is equivalent to weight decay applied to all layers.

## 요약

* 정규화(regularization)은 오버피팅(overfitting)을 다루는 일반적인 방법입니다. 학습된 모델의 복잡도를 줄이기 위해서 학습 데이터에 대한 손실 함수(loss function)의 값에 패널티 항목을 더합니다.
* 모델을 간단하게 유지하는 방법으로  $\ell_2$ 놈 패널티(norm penalty)를 사용하는 가중치 감쇠(weight decay)를 선택했습니다. 이를 통해서, 학습 알고리즘의 업데이트 단계에서 가중치 감쇠(weight decay)가 적용됩니다.
* Gluon은 옵티마이저(optimizer)에 하이퍼파라미터(hyperparameter) `wd` 를 설정하는 것으로 가중치 감쇠(weight decay) 기능을 자동으로 추가할 수 있습니다.
* 같은 학습에서 파라메미마다 다른 옵티마이저(optimizer)를 적용할 수 있습니다.

## 연습문제

1. 이 장의 예측 문제에서  $\lambda$ 값을 실험해보세요.  $\lambda$ 에 대한 함수의 형태로 학습 정확도와 테스트 정확도를 도식화해보세요. 어떤 것이 관찰되나요?
1. 검증 데이터셋을 이용해서 최적의 $\lambda$ 값을 찾아보세요. 찾은 값이 진짜 최적 값인가요? 진짜 값을 찾는 것이 중요한가요?
1. 패널티 항목으로 $\|\mathbf{w}\|^2$ 대신  $\sum_i |w_i|$ 를 사용하면 업데이트 공식이 어떻게 될까요? (이는  $\ell_1$ 정규화(regularzation)라고 합니다.)
1. $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$ 입니다. 행렬에서 비슷한 공식을 찾아볼 수 있나요? (수학자들은 이를 [Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm) 이라고 합니다)
1. 학습 오류와 일반화 오류의 관계를 복습해보세요. 가중치 감쇠(weight decay), 학습 데이터셋 늘리기, 적당한 복잡도를 갖는 모델 사용하기 외에, 오버피팅(overfitting)을 다를 수 있는 방법이 어떤 것들이 있을까요?
1. 베이시안 통계에서,  prior 와  likelihood 곱을 이용해서 posterior를 구할 수 있습니다.  $p(w|x) \propto p(x|w) p(w)$.  $p(w)$ 가 정규화(regularization)와 어떻게 동일할까요?

## QR 코드를 스캔해서 [논의하기](https://discuss.mxnet.io/t/2342)

![](../img/qr_weight-decay.svg)
