# 배포판
:label:`sec_distributions`

이산 및 연속 설정 모두에서 확률로 작업하는 방법을 배웠으므로 이제 발생하는 일반적인 분포 중 일부를 알아 보겠습니다.머신 러닝의 영역에 따라 이러한 기능에 대해 훨씬 더 잘 알고 있어야 할 수도 있고, 딥 러닝의 일부 영역에서는 전혀 익숙하지 않을 수도 있습니다.그러나 이 목록은 잘 알고 있어야 할 기본 목록입니다.먼저 몇 가지 공통 라이브러리를 가져 오겠습니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from math import erf, factorial
import numpy as np
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from math import erf, factorial
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  # Define pi in torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from math import erf, factorial
import tensorflow as tf
import tensorflow_probability as tfp

tf.pi = tf.acos(tf.zeros(1)) * 2  # Define pi in TensorFlow
```

## 베르누이

이것은 일반적으로 접하는 가장 간단한 확률 변수입니다.이 랜덤 변수는 확률이 $p$인 $1$과 확률이 $1-p$인 $0$로 나오는 동전 뒤집기를 인코딩합니다.이 분포를 가진 확률 변수 $X$가 있으면 다음과 같이 작성합니다. 

$$
X \sim \mathrm{Bernoulli}(p).
$$

누적 분포 함수는 다음과 같습니다.  

$$F(x) = \begin{cases} 0 & x < 0, \\ 1-p & 0 \le x < 1, \\ 1 & x >= 1 . \end{cases}$$
:eqlabel:`eq_bernoulli_cdf`

확률 질량 함수는 아래에 그려져 있습니다.

```{.python .input}
#@tab all
p = 0.3

d2l.set_figsize()
d2l.plt.stem([0, 1], [1 - p, p], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

이제 누적 분포 함수 :eqref:`eq_bernoulli_cdf`를 플로팅해 보겠습니다.

```{.python .input}
x = np.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, tf.constant([F(y) for y in x]), 'x', 'c.d.f.')
```

$X \sim \mathrm{Bernoulli}(p)$인 경우 다음을 수행합니다. 

* $\mu_X = p$,
* $\sigma_X^2 = p(1-p)$.

다음과 같이 Bernoulli 확률 변수에서 임의의 모양의 배열을 샘플링 할 수 있습니다.

```{.python .input}
1*(np.random.rand(10, 10) < p)
```

```{.python .input}
#@tab pytorch
1*(torch.rand(10, 10) < p)
```

```{.python .input}
#@tab tensorflow
tf.cast(tf.random.uniform((10, 10)) < p, dtype=tf.float32)
```

## 이산 유니폼

다음으로 일반적으로 발생하는 랜덤 변수는 이산 균일입니다.여기서 논의 할 때 정수 $\{1, 2, \ldots, n\}$에서 지원된다고 가정하지만 다른 값 집합은 자유롭게 선택할 수 있습니다.이 문맥에서*uniform*이라는 단어의 의미는 가능한 모든 값이 똑같이 될 가능성이 높다는 것입니다.각 값 $i \in \{1, 2, 3, \ldots, n\}$에 대한 확률은 $p_i = \frac{1}{n}$입니다.이 분포를 다음과 같이 사용하여 랜덤 변수 $X$를 나타냅니다. 

$$
X \sim U(n).
$$

누적 분포 함수는 다음과 같습니다.  

$$F(x) = \begin{cases} 0 & x < 1, \\ \frac{k}{n} & k \le x < k+1 \text{ with } 1 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_discrete_uniform_cdf`

먼저 확률질량함수를 플로팅해 보겠습니다.

```{.python .input}
#@tab all
n = 5

d2l.plt.stem([i+1 for i in range(n)], n*[1 / n], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

이제 누적 분포 함수 :eqref:`eq_discrete_uniform_cdf`를 플로팅해 보겠습니다.

```{.python .input}
x = np.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else np.floor(x) / n

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else torch.floor(x) / n

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else tf.floor(x) / n

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

$X \sim U(n)$인 경우 다음을 수행합니다. 

* $\mu_X = \frac{1+n}{2}$,
* $\sigma_X^2 = \frac{n^2-1}{12}$.

다음과 같이 이산 균일 확률 변수에서 임의의 모양의 배열을 샘플링 할 수 있습니다.

```{.python .input}
np.random.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.uniform((10, 10), 1, n, dtype=tf.int32)
```

## 연속 유니폼

다음으로 연속 균등 분포에 대해 설명하겠습니다.이 랜덤 변수의 기본 개념은 이산 균일 분포에서 $n$를 늘린 다음 구간 $[a, b]$ 내에 맞도록 스케일링하면 $[a, b]$에서 임의의 값을 모두 동일한 확률로 선택하는 연속 확률 변수에 접근한다는 것입니다.이 분포를 다음과 같이 나타냅니다. 

$$
X \sim U(a, b).
$$

확률 밀도 함수는 다음과 같습니다.  

$$p(x) = \begin{cases} \frac{1}{b-a} & x \in [a, b], \\ 0 & x \not\in [a, b].\end{cases}$$
:eqlabel:`eq_cont_uniform_pdf`

누적 분포 함수는 다음과 같습니다.  

$$F(x) = \begin{cases} 0 & x < a, \\ \frac{x-a}{b-a} & x \in [a, b], \\ 1 & x >= b . \end{cases}$$
:eqlabel:`eq_cont_uniform_cdf`

먼저 확률 밀도 함수 :eqref:`eq_cont_uniform_pdf`를 플로팅해 보겠습니다.

```{.python .input}
a, b = 1, 3

x = np.arange(0, 4, 0.01)
p = (x > a)*(x < b)/(b - a)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
a, b = 1, 3

x = torch.arange(0, 4, 0.01)
p = (x > a).type(torch.float32)*(x < b).type(torch.float32)/(b-a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
a, b = 1, 3

x = tf.range(0, 4, 0.01)
p = tf.cast(x > a, tf.float32) * tf.cast(x < b, tf.float32) / (b - a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

이제 누적 분포 함수 :eqref:`eq_cont_uniform_cdf`를 플로팅해 보겠습니다.

```{.python .input}
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

$X \sim U(a, b)$인 경우 다음을 수행합니다. 

* $\mu_X = \frac{a+b}{2}$,
* $\sigma_X^2 = \frac{(b-a)^2}{12}$.

다음과 같이 균일 확률 변수에서 임의의 모양의 배열을 샘플링 할 수 있습니다.기본적으로 $U(0,1)$에서 샘플링되므로 다른 범위를 원할 경우 크기를 조정해야 합니다.

```{.python .input}
(b - a) * np.random.rand(10, 10) + a
```

```{.python .input}
#@tab pytorch
(b - a) * torch.rand(10, 10) + a
```

```{.python .input}
#@tab tensorflow
(b - a) * tf.random.uniform((10, 10)) + a
```

## 이항

좀 더 복잡하게 만들고*이항식* 확률 변수를 살펴 보겠습니다.이 랜덤 변수는 $n$ 독립 실험의 시퀀스를 수행하는 데서 시작되며, 각 실험은 성공할 확률이 $p$이고 얼마나 많은 성공을 기대하는지 묻습니다. 

이것을 수학적으로 표현해 봅시다.각 실험은 독립 랜덤 변수 $X_i$이며, 여기서 $1$를 사용하여 성공을 인코딩하고 $0$을 사용하여 실패를 인코딩합니다.각각은 확률 $p$로 성공한 독립적 인 동전 뒤집기이기 때문에 $X_i \sim \mathrm{Bernoulli}(p)$이라고 말할 수 있습니다.그런 다음 이항 확률 변수는 다음과 같습니다. 

$$
X = \sum_{i=1}^n X_i.
$$

이 경우 다음과 같이 작성합니다. 

$$
X \sim \mathrm{Binomial}(n, p).
$$

누적 분포 함수를 얻으려면 $\ binom {n} {k} =\ frac {n!} 에서 정확히 $k$의 성공을 얻을 수 있음을 알아야합니다.{k!(n-k)!}$ ways each of which has a probability of $p^k (1-p) ^ {n-k} $가 발생합니다.따라서 누적 분포 함수는 다음과 같습니다. 

$$F(x) = \begin{cases} 0 & x < 0, \\ \sum_{m \le k} \binom{n}{m} p^m(1-p)^{n-m}  & k \le x < k+1 \text{ with } 0 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_binomial_cdf`

먼저 확률질량함수를 플로팅해 보겠습니다.

```{.python .input}
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = np.array([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = d2l.tensor([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = tf.constant([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

이제 누적 분포 함수 :eqref:`eq_binomial_cdf`를 플로팅해 보겠습니다.

```{.python .input}
x = np.arange(-1, 11, 0.01)
cmf = np.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 11, 0.01)
cmf = torch.cumsum(pmf, dim=0)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 11, 0.01)
cmf = tf.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

$X \sim \mathrm{Binomial}(n, p)$인 경우 다음을 수행합니다. 

* $\mu_X = np$,
* $\sigma_X^2 = np(1-p)$.

이는 $n$ Bernoulli 랜덤 변수의 합에 대한 기대값의 선형성과 독립 랜덤 변수의 합의 분산이 분산의 합이라는 사실에서 비롯됩니다.다음과 같이 샘플링할 수 있습니다.

```{.python .input}
np.random.binomial(n, p, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.binomial.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

## 푸아송 이제 사고 실험을 해봅시다.우리는 버스 정류장에 서 있으며 다음 순간에 얼마나 많은 버스가 도착할지 알고 싶습니다.먼저 버스가 1분 창에 도착할 확률인 $X^{(1)} \sim \mathrm{Bernoulli}(p)$를 고려해 보겠습니다.도심에서 멀리 떨어진 버스 정류장의 경우 이는 꽤 좋은 근사치일 수 있습니다.한 분 안에 버스가 두 대 이상 보이지 않을 수도 있습니다. 

그러나 우리가 바쁜 지역에 있다면 두 대의 버스가 도착할 수도 있고 심지어 가능할 수도 있습니다.랜덤 변수를 처음 30초 또는 두 번째 30초 동안 두 부분으로 분할하여 이를 모델링할 수 있습니다.이 경우 다음과 같이 쓸 수 있습니다. 

$$
X^{(2)} \sim X^{(2)}_1 + X^{(2)}_2,
$$

여기서 $X^{(2)}$는 총 합계이고 $X^{(2)}_i \sim \mathrm{Bernoulli}(p/2)$입니다.그런 다음 총 분포는 $X^{(2)} \sim \mathrm{Binomial}(2, p/2)$입니다. 

왜 여기서 멈춰요?그 분을 $n$ 부분으로 계속 나누겠습니다.위와 같은 추론으로 우리는 

$$X^{(n)} \sim \mathrm{Binomial}(n, p/n).$$
:eqlabel:`eq_eq_poisson_approx`

이러한 랜덤 변수를 고려해 보십시오.이전 섹션에서는 :eqref:`eq_eq_poisson_approx`의 평균이 $\mu_{X^{(n)}} = n(p/n) = p$이고 분산 $\sigma_{X^{(n)}}^2 = n(p/n)(1-(p/n)) = p(1-p/n)$가 있음을 알 수 있습니다.$n \rightarrow \infty$을 취하면, 이 숫자들이 $\mu_{X^{(\infty)}} = p$로 안정화되고 분산이 $\sigma_{X^{(\infty)}}^2 = p$으로 안정화된다는 것을 알 수 있습니다.이것은 이 무한 분할 한계에서 정의할 수 있는 랜덤 변수가*될 수 있음을 나타냅니다.   

현실 세계에서는 버스 도착 횟수를 계산할 수 있기 때문에 이것은 놀랄 일이 아닙니다. 그러나 수학적 모델이 잘 정의되어 있다는 것을 알면 좋습니다.이 토론은*희귀 사건의 법칙*으로 공식적으로 이루어질 수 있습니다. 

이 추론을 신중하게 따라 다음 모델에 도달 할 수 있습니다.확률 변수 인 경우 $X \sim \mathrm{Poisson}(\lambda)$라고 말할 것입니다. 확률 변수 인 경우 $\{0,1,2, \ldots\}$ 값을 확률로 취합니다. 

$$p_k = \frac{\lambda^ke^{-\lambda}}{k!}.$$
:eqlabel:`eq_poisson_mass`

값 $\lambda > 0$는*rate* (또는*shape* 매개 변수) 로 알려져 있으며, 한 시간 단위로 예상되는 평균 도착 횟수를 나타냅니다.   

이 확률 질량 함수를 합산하여 누적 분포 함수를 얻을 수 있습니다. 

$$F(x) = \begin{cases} 0 & x < 0, \\ e^{-\lambda}\sum_{m = 0}^k \frac{\lambda^m}{m!} & k \le x < k+1 \text{ with } 0 \le k. \end{cases}$$
:eqlabel:`eq_poisson_cdf`

먼저 확률질량함수 :eqref:`eq_poisson_mass`를 플로팅해 보겠습니다.

```{.python .input}
lam = 5.0

xs = [i for i in range(20)]
pmf = np.array([np.exp(-lam) * lam**k / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
lam = 5.0

xs = [i for i in range(20)]
pmf = torch.tensor([torch.exp(torch.tensor(-lam)) * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
lam = 5.0

xs = [i for i in range(20)]
pmf = tf.constant([tf.exp(tf.constant(-lam)).numpy() * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

이제 누적 분포 함수 :eqref:`eq_poisson_cdf`를 플로팅해 보겠습니다.

```{.python .input}
x = np.arange(-1, 21, 0.01)
cmf = np.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 21, 0.01)
cmf = torch.cumsum(pmf, dim=0)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 21, 0.01)
cmf = tf.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

위에서 보았 듯이 평균과 분산은 특히 간결합니다.$X \sim \mathrm{Poisson}(\lambda)$인 경우 다음을 수행합니다. 

* $\mu_X = \lambda$,
* $\sigma_X^2 = \lambda$.

다음과 같이 샘플링할 수 있습니다.

```{.python .input}
np.random.poisson(lam, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.poisson.Poisson(lam)
m.sample((10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Poisson(lam)
m.sample((10, 10))
```

## Gaussian Now 다른 관련 실험을 해보겠습니다.$n$ 독립적인 $\mathrm{Bernoulli}(p)$ 측정 $X_i$을 다시 수행하고 있다고 가정해 보겠습니다.이들의 합계의 분포는 $X^{(n)} \sim \mathrm{Binomial}(n, p)$입니다.$n$이 증가하고 $p$가 감소함에 따라 한도를 취하는 대신 $p$를 수정한 다음 $n \rightarrow \infty$을 보내 보겠습니다.이 경우 $\mu_{X^{(n)}} = np \rightarrow \infty$ 및 $\sigma_{X^{(n)}}^2 = np(1-p) \rightarrow \infty$이므로 이 제한을 잘 정의해야 한다고 생각할 이유가 없습니다. 

그러나 모든 희망이 사라지는 것은 아닙니다!정의를 통해 평균과 분산이 잘 작동하도록 만들어 보겠습니다. 

$$
Y^{(n)} = \frac{X^{(n)} - \mu_{X^{(n)}}}{\sigma_{X^{(n)}}}.
$$

이것은 평균 0과 분산 1을 갖는 것으로 볼 수 있으므로 일부 제한 분포에 수렴 할 것이라고 믿는 것이 그럴듯합니다.이러한 분포가 어떻게 생겼는지 플로팅하면 효과가 있다는 확신을 갖게 될 것입니다.

```{.python .input}
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = np.array([p**i * (1-p)**(n-i) * binom(n, i) for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/np.sqrt(n*p*(1 - p)) for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = torch.tensor([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/torch.sqrt(torch.tensor(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = tf.constant([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/tf.sqrt(tf.constant(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

한 가지 주목할 점은 푸아송 사례와 비교할 때 표준 편차로 나누고 있습니다. 즉, 가능한 결과를 더 작고 작은 영역으로 압박하고 있음을 의미합니다.이는 한도가 더 이상 이산적이지 않고 오히려 연속적이라는 표시입니다. 

발생하는 일의 파생은 이 문서의 범위를 벗어나지만, *중심 한계 정리*는 $n \rightarrow \infty$로서 가우스 분포 (또는 때로는 정규 분포) 를 산출한다고 명시하고 있습니다.더 명시적으로, $a, b$의 경우: 

$$
\lim_{n \rightarrow \infty} P(Y^{(n)} \in [a, b]) = P(\mathcal{N}(0,1) \in [a, b]),
$$

여기서 랜덤 변수는 주어진 평균 $\mu$과 분산 $\sigma^2$으로 정규 분포되어 있다고 말합니다. $X$가 밀도를 갖는 경우 $X \sim \mathcal{N}(\mu, \sigma^2)$로 작성됩니다. 

$$p_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}.$$
:eqlabel:`eq_gaussian_pdf`

먼저 확률 밀도 함수 :eqref:`eq_gaussian_pdf`를 플로팅해 보겠습니다.

```{.python .input}
mu, sigma = 0, 1

x = np.arange(-3, 3, 0.01)
p = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
mu, sigma = 0, 1

x = torch.arange(-3, 3, 0.01)
p = 1 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
mu, sigma = 0, 1

x = tf.range(-3, 3, 0.01)
p = 1 / tf.sqrt(2 * tf.pi * sigma**2) * tf.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

이제 누적 분포 함수를 플로팅해 보겠습니다.이 부록의 범위를 벗어나지만 가우스 c.d.f. 에는 더 많은 기본 함수 측면에서 폐쇄 형식 공식이 없습니다.이 적분을 수치적으로 계산하는 방법을 제공하는 `erf`를 사용할 것입니다.

```{.python .input}
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * np.sqrt(2)))) / 2.0

d2l.plot(x, np.array([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * torch.sqrt(d2l.tensor(2.))))) / 2.0

d2l.plot(x, torch.tensor([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * tf.sqrt(tf.constant(2.))))) / 2.0

d2l.plot(x, [phi(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

예리한 독자는 이러한 용어 중 일부를 인정할 것입니다.실제로 우리는 :numref:`sec_integral_calculus`에서 이 적분을 만났습니다.실제로 우리는 이 $p_X(x)$가 총 면적 1을 가지므로 유효한 밀도임을 알기 위해서는 정확히 그 계산이 필요합니다. 

코인 플립으로 작업하기로 한 우리의 선택은 계산을 더 짧게 만들었지 만, 그 선택에 대한 근본적인 것은 없었습니다.실제로 독립적으로 동일하게 분포 된 확률 변수 $X_i$의 모음을 취하면 다음과 같은 형식을 취합니다. 

$$
X^{(N)} = \sum_{i=1}^N X_i.
$$

그럼 

$$
\frac{X^{(N)} - \mu_{X^{(N)}}}{\sigma_{X^{(N)}}}
$$

대략 가우시안 일 것입니다.작동하려면 가장 일반적으로 $E[X^4] < \infty$라는 추가 요구 사항이 필요하지만 철학은 분명합니다. 

중심 극한 정리는 가우스가 확률, 통계 및 기계 학습의 기본이되는 이유입니다.우리가 측정 한 것이 많은 작은 독립적 기여의 합계라고 말할 수있을 때마다 측정되는 것이 가우시안에 가까울 것이라고 가정 할 수 있습니다.   

가우시안 (Gaussians) 에는 더 많은 매력적인 특성이 있습니다. 여기서 한 가지 더 논의하고 싶습니다.가우시안 값은*최대 엔트로피 분포*로 알려져 있습니다.우리는 :numref:`sec_information_theory`에서 엔트로피에 더 깊이 들어갈 것입니다. 그러나 이 시점에서 우리가 알아야 할 것은 그것이 무작위성의 척도라는 것입니다.엄격한 수학적 의미에서 가우스는 평균과 분산이 고정 된 랜덤 변수의*가장* 무작위 선택이라고 생각할 수 있습니다.따라서 랜덤 변수에 평균과 분산이 있다는 것을 알면 가우스는 어떤 의미에서 우리가 할 수있는 가장 보수적 인 분포 선택입니다. 

섹션을 닫으려면 $X \sim \mathcal{N}(\mu, \sigma^2)$이면 다음을 기억해 보겠습니다. 

* $\mu_X = \mu$,
* $\sigma_X^2 = \sigma^2$.

아래와 같이 가우스 (또는 표준 정규) 분포에서 표본을 추출할 수 있습니다.

```{.python .input}
np.random.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.normal((10, 10), mu, sigma)
```

## 지수 패밀리
:label:`subsec_exponential_family`

위에 나열된 모든 분포에 대한 공유 속성 중 하나는 모두*지수 군*으로 알려진 분포가 모두 속한다는 것입니다.지수 군은 밀도를 다음과 같은 형식으로 표현할 수 있는 분포의 집합입니다. 

$$p(\mathbf{x} | \boldsymbol{\eta}) = h(\mathbf{x}) \cdot \mathrm{exp} \left( \boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) - A(\boldsymbol{\eta}) \right)$$
:eqlabel:`eq_exp_pdf`

이 정의는 약간 미묘할 수 있으므로 자세히 살펴 보겠습니다.   

첫째, $h(\mathbf{x})$는*기본 측정값* 또는 
*기본 측정*.이것은 원래의 측정 선택으로 볼 수 있습니다. 
지수 가중치로 수정합니다.   

둘째, 우리는 벡터 $\ 굵은 기호 {\ eta} = (\ eta_1,\ eta_2,...,\ eta_l)\ in\ mathbb {R} ^l$는*자연 매개 변수* 또는*표준 매개 변수*라고합니다.기본 측정값을 수정하는 방법을 정의합니다.자연 매개 변수는 $\ mathbf {x} = (x_1, x_2,..., x_n)\ in\ mathbb {R} ^N$ and exponentiated. The vector $T (\ mathbf {x}) = (T_1 (\ mathbf {x}), T_2 (\ mathbf {x}) 에 대해 이러한 매개 변수의 내적을 취하여 새 측정값으로 들어갑니다.}),..., t_l (\ mathbf {x})) $는 $\boldsymbol{\eta}$에 대한*충분한 통계*라고 합니다..이 이름은 $T(\mathbf{x})$으로 표시된 정보가 확률 밀도를 계산하기에 충분하고 표본 $\mathbf{x}$의 다른 정보가 필요하지 않기 때문에 사용됩니다. 

셋째, 우리는*누적 함수*라고 하는 $A(\boldsymbol{\eta})$를 가지고 있으며, 이는 위의 분포 :eqref:`eq_exp_pdf`가 하나에 통합되도록 보장합니다. 

$$A(\boldsymbol{\eta})  = \log \left[\int h(\mathbf{x}) \cdot \mathrm{exp}
\left(\boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) \right) d\mathbf{x} \right].$$

구체적으로 가우시안 (Gaussian) 을 고려해 보겠습니다.$\mathbf{x}$가 일변량 변수라고 가정하면 밀도가 

$$
\begin{aligned}
p(x | \mu, \sigma) &= \frac{1}{\sqrt{2 \pi \sigma^2}} \cdot \mathrm{exp} 
\left\{ \frac{-(x-\mu)^2}{2 \sigma^2} \right\} \\
&= \frac{1}{\sqrt{2 \pi}} \cdot \mathrm{exp} \left\{ \frac{\mu}{\sigma^2}x
-\frac{1}{2 \sigma^2} x^2 - \left( \frac{1}{2 \sigma^2} \mu^2
+\log(\sigma) \right) \right\}.
\end{aligned}
$$

지수 패밀리의 정의는 다음과 일치합니다. 

* *기본 측정값*: $h(x) = \frac{1}{\sqrt{2 \pi}}$,
* *자연 매개 변수*: $\ 굵은 기호 {\ eta} =\ 시작 {b 매트릭스}\ eta_1\\ eta_2
\ 끝 {매트릭스} =\ 시작 {매트릭스}\ FRAC {\ 뮤} {\ 시그마^2}\\ FRAC {1} {2\ 시그마^2}\ 끝 {매트릭스} $,
* *충분한 통계*: $T(x) = \begin{bmatrix}x\\-x^2\end{bmatrix}$, 그리고
* *누적 함수*: $A ({\ 굵은 기호\ eta}) =\ frac {1} {2\ 시그마^2}\ mu^2 +\ 로그 (\ 시그마)
=\ frac {\ eta_1^2} {4\ 에타_2} -\ 프락 {1} {2}\ 로그 (2\ 에타_2) $. 

위의 각 용어의 정확한 선택은 다소 임의적이라는 점은 주목할 가치가 있습니다.실제로 중요한 특징은 분포를 정확한 형식 자체가 아니라 이 형식으로 표현할 수 있다는 것입니다. 

:numref:`subsec_softmax_and_derivatives`에서 언급했듯이 널리 사용되는 기술은 최종 출력 $\mathbf{y}$가 지수 가족 분포를 따른다고 가정하는 것입니다.지수 군은 기계 학습에서 자주 접하는 일반적이고 강력한 분포 군입니다. 

## 요약* Bernoulli 확률 변수를 사용하여 예/아니오 결과가 있는 이벤트를 모델링할 수 있습니다.* 이산 균일 분포 모델은 유한한 가능성 집합에서 선택합니다.* 연속적인 균일 분포는 구간에서 선택합니다.* 이항 분포는 일련의 베르누이 확률 변수를 모형화하고 개수를 계산합니다.성공 횟수.* 푸아송 랜덤 변수는 희귀 사건의 도래를 모형화합니다. * 가우스 랜덤 변수는 많은 수의 독립 확률 변수를 함께 더한 결과를 모형화합니다.* 위의 모든 분포는 지수 군에 속합니다. 

## 연습문제

1. 두 독립 이항 랜덤 변수 $X, Y \sim \mathrm{Binomial}(16, 1/2)$의 차이 $X-Y$인 랜덤 변수의 표준 편차는 얼마입니까?
2. 푸아송 확률 변수 $X \sim \mathrm{Poisson}(\lambda)$를 취하고 $(X - \lambda)/\sqrt{\lambda}$를 $\lambda \rightarrow \infty$으로 간주하면 이것이 대략 가우스가 된다는 것을 보여줄 수 있습니다.이것이 왜 이치에 맞을까요?
3. $n$ 원소에서 두 개의 이산 균일 확률 변수의 합에 대한 확률 질량 함수는 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/417)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1098)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1099)
:end_tab:
