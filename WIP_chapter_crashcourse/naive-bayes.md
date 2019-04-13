# 나이브 베이즈 분류(Naive Nayes Classification)

조건에 독립적인 것은 데이터를 다루는데 있어서 많은 공식을 간단하게 해주기 때문에 유용합니다. 간단하고 유명한 알고리즘으로 나이브 베이즈 분류(Naive Bayes Classifier)가 있습니다. 이 알고리즘의 주요 가정은 주어진 레이블들에 대해서 모든 속성들이 서로 영향을 주지 않는다는 것입니다. 즉, 다음과 같은 수식을 만족시킵니다.

$$p(\mathbf{x} | y) = \prod_i p(x_i | y)$$

베이즈 이론을 적용해보면,  $p(y | \mathbf{x}) = \frac{\prod_i p(x_i | y) p(y)}{p(\mathbf{x})}$  분류를 얻을 수 있습니다. 하지만 아쉽게도, $p(x)$ 를 모르기 때문에 다루기 어렵습니다. 다행인 것은,  $\sum_y p(y | \mathbf{x}) = 1$ 인 것을 알고 있기 때문에, $p(x)$가 필요가 없습니다. 따라서, 우리는 항상 표준화(normalization)를 구할 수 있습니다.

$$p(y | \mathbf{x}) \propto \prod_i p(x_i | y) p(y).​$$

스팸 메일과 일반 메일을 분류하는 것을 예로 들어서 설명해보겠습니다. `Nigeria`, `prince`, `money`, `rich` 와 같은 단어들이 이메일에 있다는 것은 그 이메일이 스팸일 가능성이 있다는 것을 암시한다고 할 수 있습니다. 반면에, `theorem`, `network`, `Bayes`, `statistics` 같은 단어는 메시지가 실질적인 내용을 담고 있는 것을 암시한다고 할 수 있습니다. 따라서, 이런 단어들이 속한 클래스가 주어졌을 때 각각에 대한 확률에 대한 모델을 만들고, 문장이 스팸일 가능성에 대한 점수를 매기는데 사용하는 것이 가능합니다. 실제로 이 방법은 많은 오랫동안 [Bayesian spam filters](https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering)가 사용하는 방법입니다.

## 광학 문자 인지 (optical character recognition)

이미지가 다루기 더 쉽기 때문에, MNIST 데이터셋의 숫자를 구분하는 문제에 나이브 베이즈 분류를 적용해보겠습니다. 여기서 문제는 $p(y)$ 와 $p(x_i | y)$ 를 모른다는 것입니다. 그렇게 때문에 우리가 해야할 일은 주어진 학습 데이터를 사용해서 이 확률들을 *추정*해야합니다. 즉, 모델을 학습시키는 것이 필요합니다. $p(y)$ 를 추정하는 것은 어려운 일이 아닙니다. 우리가 다루는 클래스의 개수가 10개이기 때문에, 아주 쉽게 추정할 수 있습니다 - 즉, 각 숫자가 나오는 것을 카운팅 한 수 $n_y$ 전체 데이터 개수 $n$ 으로 나누면 됩니다. 예를 들어 숫자 8이 나온 횟수가 $n_8 = 5,800$ 이고 전체 데이터 개수가 of $n = 60,000$ 개이면, 추정 확률은 $p(y=8) = 0.0967$ 입니다.

이제 조금 더 어려운 $p(x_i | y)$에 대해서 이야기해보겠습니다. 이미지가 흑백으로 되어 있기 때문에, $p(x_i | y)$ 는 픽셀 $i$ 가 클래스 $y$ 에 속할 확률을 나타냅니다. 앞에서 적용한 방법을 이용해서 어떤 이벤트가 발생한 횟수 $n_{iy}$ 를 카운트하고, y가 일어난 전체 횟수  $n_y$ 로 나눌 수 있습니다. 그러나 약간 까다로운 문제가 있습니다 - 어떤 픽셀은 절대로 검정색이지 않을 수 있습니다. (만약, 이미지를 잘 잘라내면, 코너 픽셀들은 항상 흰색일 것이기 때문입니다.) 통계학자들이 이런 문제를 다루는 편리한 방법으로 의사(pseudo) 카운트를 모든 것에 더하는 것입니다. 즉, $n_{iy}$ 를 사용하는 대신 $n_{iy}+1$ 를, $n_y$  대신 $n_{y} + 1$ 를 사용하는 것입니다. 이 방법은  [Laplace Smoothing](https://en.wikipedia.org/wiki/Additive_smoothing)이라고 불리는 것입니다.

```{.python .input  n=1}
%matplotlib inline
from matplotlib import pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')
import mxnet as mx
from mxnet import nd
import numpy as np

# We go over one observation at a time (speed doesn't matter here)
def transform(data, label):
    return (nd.floor(data/128)).astype(np.float32), label.astype(np.float32)
mnist_train = mx.gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test  = mx.gluon.data.vision.MNIST(train=False, transform=transform)

# Initialize the counters
xcount = nd.ones((784,10))
ycount = nd.ones((10))

for data, label in mnist_train:
    y = int(label)
    ycount[y] += 1
    xcount[:,y] += data.reshape((784))

# using broadcast again for division
py = ycount / ycount.sum()
px = (xcount / ycount.reshape(1,10))
```

모든 픽셀들에 대해서 픽셀 별로 등장하는 횟수를 계산했으니, 우리의 모델이 어떻게 동작하는지 그림을 그려서 보겠습니다. 이미지를 이용하면 아주 편한 점은 시각화가 가능하다는 것입니다. 28x28x10 확률을 시각화하는 것은 무의미한 일이니, 이미지 형태로 그려서 빠르게 살펴보겠습니다. 눈치가 빠른 분은 숫자를 닯은 어떤 평균임을 알아차렸을지도 모르겠습니다.

```{.python .input  n=2}
import matplotlib.pyplot as plt
fig, figarr = plt.subplots(1, 10, figsize=(10, 10))
for i in range(10):
    figarr[i].imshow(xcount[:, i].reshape((28, 28)).asnumpy(), cmap='hot')
    figarr[i].axes.get_xaxis().set_visible(False)
    figarr[i].axes.get_yaxis().set_visible(False)

plt.show()
print('Class probabilities', py)
```

우리의 모델에 근거해서 이미지의 가능성들에 대한 계산을 할 수 있게 되었습니다. 통계 용어로 말하면,  $p(x | y)​$ 을 계산할 수 있습니다. 즉, 이는 주어진 이미지가 특정 레이블에 속할 확률이 됩니다. ''모든 픽셀이 독립적이다''라고 가정하는 나이브 베이즈 모델에 따르면, 다음 공식을 얻습니다.

$$p(\mathbf{x} | y) = \prod_{i} p(x_i | y)​$$

따라서, 베이즈 법칙을 사용하면,  $p(y | \mathbf{x})$ 의 값은 다음 식으로 구할 수 있습니다.

$$p(y | \mathbf{x}) = \frac{p(\mathbf{x} | y) p(y)}{\sum_{y'} p(\mathbf{x} | y')}​$$

그럼 코드를 실행해 보겠습니다.

```{.python .input  n=3}
# Get the first test item
data, label = mnist_test[0]
data = data.reshape((784,1))

# Compute the per pixel conditional probabilities
xprob = (px * data + (1-px) * (1-data))
# Take the product
xprob = xprob.prod(0) * py
print('Unnormalized Probabilities', xprob)
# Normalize
xprob = xprob / xprob.sum()
print('Normalized Probabilities', xprob)
```

완전히 틀린 결과를 얻었습니다. 이유를 찾기 위해 각 픽셀의 확률을 살펴보겠습니다. 일반적으로 확률값은 $0.001$ 와 $1$ 사이의 값입니다. 우리는 $784$ 개의 값을 곱했습니다. 바로 이 부분에서 문제가 발생합니다. 즉, 고정소수점 연산을 수행하면서 수치적인 언더플로우($numerial underflow$)가 발생합니다. 작은 수들을 계속 곱하면 결국 0이되는 되고, 결국 0을 나누는 일이 발생하기 때문에 결과가 `nan` 이 되는 것입니다.

이 문제를 해결하기 위해서  $\log a b = \log a + \log b$ 이라는 것을 사용하겠습니다. 즉, 로그(logarithm) 합으로 바꾸기 입니다. 이를 적용하면 log-공간에서 비표준화 된 확률을 얻게 되니, 표준화하기 위해서 다음과 같은 공식을 활용합니다.

$$\frac{\exp(a)}{\exp(a) + \exp(b)} = \frac{\exp(a + c)}{\exp(a + c) + \exp(b + c)}​$$

분모 항의 하나는 1이 되도록 $c = -\max(a,b)​$ 로 선택합니다.

```{.python .input  n=4}
logpx = nd.log(px)
logpxneg = nd.log(1-px)
logpy = nd.log(py)

def bayespost(data):
    # We need to incorporate the prior probability p(y) since p(y|x) is
    # proportional to p(x|y) p(y)
    logpost = logpy.copy()
    logpost += (logpx * data + logpxneg * (1-data)).sum(0)
    # Normalize to prevent overflow or underflow by subtracting the largest
    # value
    logpost -= nd.max(logpost)
    # Compute the softmax using logpx
    post = nd.exp(logpost).asnumpy()
    post /= np.sum(post)
    return post

fig, figarr = plt.subplots(2, 10, figsize=(10, 3))

# Show 10 images
ctr = 0
for data, label in mnist_test:
    x = data.reshape((784,1))
    y = int(label)

    post = bayespost(x)

    # Bar chart and image of digit
    figarr[1, ctr].bar(range(10), post)
    figarr[1, ctr].axes.get_yaxis().set_visible(False)
    figarr[0, ctr].imshow(x.reshape((28, 28)).asnumpy(), cmap='hot')
    figarr[0, ctr].axes.get_xaxis().set_visible(False)
    figarr[0, ctr].axes.get_yaxis().set_visible(False)
    ctr += 1

    if ctr == 10:
        break

plt.show()
```

보이는 것처럼, 이 분류기가 많은 경우 잘 동작하고 있습니다. 하지만, 뒤에서 두번째 숫자는 예측이 틀리기도 하고 그 잘못된 예측에 대해서 너무 높은 확신값을 주고 있습니다. 즉, 완전히 틀린 추측일 경우에도 확률을 0 또는 1에 아주 가깝게 출력하고 있습니다. 이런 모델은 사용할 수 있는 수준이 아닙니다. 이 분류기의 전반적인 정확도를 계산해서 이 모델이 얼마나 좋은지 확인합니다.

```{.python .input  n=5}
# Initialize counter
ctr = 0
err = 0

for data, label in mnist_test:
    ctr += 1
    x = data.reshape((784,1))
    y = int(label)

    post = bayespost(x)
    if (post[y] < post.max()):
        err += 1

print('Naive Bayes has an error rate of', err/ctr)
```

현대 딥 네트워크는 0.01 보다 낮은 에러율을 달성합니다. 나이브 베이즈 분류기는 80년대나 90년대에 스팸 필터 등을 만드는데 많이 사용되었지만, 이제는 더이상 사용되지 않습니다. 성능이 나쁜 이유는 우리가 모델을 만들때 했던 통계적인 가정이 틀렸기 때문입니다 - 즉, 모든 픽셀은 서로 연관이 없고, 오직 레이블에만 관련이 있다고 가정한 것이 틀렸기 때문입니다. 사람들이 숫자를 적는 방법이 다양하다는 것을 반영하지 못하는 틀린 가정이 잘 작동하지 않는 분류기를 만들어낸 것입니다. 자 이제부터 딥 네트워크를 만드는 것을 시작해보겠습니다.

## 요약

* 나이브 베이즈는 $p(\mathbf{x} | y) = \prod_i p(x_i | y)$ 를 가정하는 분류기를 만들기 쉽습니다.
* 분류기는 학습시키기 쉽지만, 예측이 많이 틀리기 쉽습니다.
* 전반적인 신뢰수준과 틀린 예측을 해결하기 위해서,  $p(x_i|y)$ 확률을 Laplace smoothing 과 같은 방법을 적용할 수 있습니다. 즉, 모든 카운트에 상수를 더하는 방법을 적용할 수 있습니다.
* 나이브 베이즈 분류기는 관찰(observation)들 사이의 관계를 고려하지 않습니다. 

## 문제

1. $p(x_i | y)$ 가 표준 분포를 따를 때, 나이브 베이즈 회귀 모델(Naive Bayes regression estimator)을 만들어보세요.
1. 어떤 경우 나이브 베이즈가 잘 동작할까요?
1. 어떤 목격자가 가해자를 다시 봤을 경우, 90% 정확도로 그 사람을 인식할 수 있다고 확신합니다.
   * 5명의 용의자가 있을 경우, 이 사실이 유용할까요?
   * 50명일 경우에도 유용할까요?


## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2320)

![](../img/qr_naive-bayes.svg)
