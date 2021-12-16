# 랜덤 변수
:label:`sec_random_variables`

:numref:`sec_prob`에서 우리는 이산 확률 변수로 작업하는 방법에 대한 기본 사항을 보았습니다. 이 경우 가능한 값의 유한 집합 또는 정수를 취하는 확률 변수를 나타냅니다.이 섹션에서는 실제 값을 취할 수 있는 랜덤 변수인*연속 랜덤 변수*이론을 개발합니다. 

## 연속 랜덤 변수

연속 랜덤 변수는 이산 랜덤 변수보다 훨씬 더 미묘한 주제입니다.공정한 비유는 기술적 점프가 숫자 목록을 추가하고 함수를 통합하는 것 사이의 점프와 비슷하다는 것입니다.따라서 이론을 개발하는 데 시간이 좀 걸릴 것입니다. 

### 불연속형에서 연속형으로

연속 랜덤 변수로 작업 할 때 발생하는 추가 기술적 문제를 이해하기 위해 사고 실험을 수행해 보겠습니다.다트 보드에 다트를 던지고 보드 중앙에서 정확히 $2 \text{cm}$에 닿을 확률을 알고 싶다고 가정합니다. 

우선, 한 자릿수의 정확도, 즉 $0 \text{cm}$, $1 \text{cm}$, $2 \text{cm}$ 등에 대한 빈을 측정하는 것을 상상합니다.우리는 다트 보드에 $100$ 개의 다트를 던지고, 그 중 $20$ 개가 $2\text{cm}$의 빈에 떨어지면 중앙에서 $20\%$ of the darts we throw hit the board $2\ 텍스트 {cm} $ 떨어져 있다고 결론을 내립니다. 

그러나 자세히 살펴보면 이것이 우리의 질문과 일치하지 않습니다!우리는 정확한 평등을 원했지만, 이 쓰레기통은 $1.5\text{cm}$와 $2.5\text{cm}$ 사이에 있는 모든 것을 담고 있습니다. 

우리는 계속해서 더 나아갑니다.우리는 $1.9\text{cm}$, $2.0\text{cm}$, $2.1\text{cm}$와 같이 훨씬 더 정확하게 측정하고, 이제 $100$ 다트 중 $3$이 $2.0\text{cm}$ 버킷에서 보드에 부딪히는 것을 볼 수 있습니다.따라서 확률은 $3\ %$라고 결론을 내립니다. 

그러나 이것으로 아무 것도 해결되지 않습니다!방금 문제를 한 자릿수 더 낮추었습니다.조금 추상화해 보겠습니다.첫 번째 $k$ 숫자가 $2.00000\ldots$과 일치할 확률을 알고 있으며 처음 $k+1$자리와 일치할 확률을 알고 싶다고 가정해 보겠습니다.${k+1}^{\mathrm{th}}$ 숫자는 기본적으로 집합 $\{0, 1, 2, \ldots, 9\}$에서 임의의 선택이라고 가정하는 것이 합리적입니다.적어도, 우리는 중앙에서 멀리 떨어진 마이크로 미터 수가 $7$ 대 $3$로 끝나는 것을 선호하도록 강요하는 물리적으로 의미있는 과정을 생각할 수 없습니다. 

이것이 의미하는 바는 본질적으로 필요한 각 추가 정확도 자릿수가 일치 확률을 $10$ 배만큼 줄여야한다는 것입니다.아니면 다른 말로하면, 우리는 

$$
P(\text{distance is}\; 2.00\ldots, \;\text{to}\; k \;\text{digits} ) \approx p\cdot10^{-k}.
$$

값 $p$는 기본적으로 처음 몇 자리 숫자로 발생하는 작업을 인코딩하고 $10^{-k}$는 나머지를 처리합니다. 

소수점 이하 $k=4$ 자리까지 정확한 위치를 알고 있다면. 즉, 값이 길이 $2.00005-1.99995 = 10^{-4}$의 간격 인 $[(1.99995,2.00005]$과 같은 간격 내에 속한다는 것을 알 수 있습니다.따라서이 구간의 길이를 $\epsilon$라고 부르면 다음과 같이 말할 수 있습니다. 

$$
P(\text{distance is in an}\; \epsilon\text{-sized interval around}\; 2 ) \approx \epsilon \cdot p.
$$

이 마지막 단계를 더 진행해 보겠습니다.우리는 전체적으로 $2$ 요점에 대해 생각해 왔지만 다른 점에 대해서는 생각하지 않았습니다.근본적으로 다른 것은 없지만 값 $p$가 다를 수 있습니다.우리는 적어도 다트 던지기가 $20\text{cm}$이 아닌 $2\text{cm}$과 같이 중앙 근처의 지점에 도달 할 가능성이 더 높기를 바랍니다.따라서 값 $p$는 고정되어 있지 않지만 점 $x$에 따라 달라야합니다.이것은 우리가 기대해야 한다는 것을 말해줍니다. 

$$P(\text{distance is in an}\; \epsilon \text{-sized interval around}\; x ) \approx \epsilon \cdot p(x).$$
:eqlabel:`eq_pdf_deriv`

실제로 :eqref:`eq_pdf_deriv`는*확률 밀도 함수*를 정확하게 정의합니다.한 지점 대 다른 지점 근처에서 타격 할 상대적 확률을 인코딩하는 함수 $p(x)$입니다.이러한 함수가 어떻게 생겼는지 시각화해 보겠습니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Plot the probability density function for some random variable
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2)/np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2)/np.sqrt(2 * np.pi)

d2l.plot(x, p, 'x', 'Density')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Define pi in torch

# Plot the probability density function for some random variable
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi)) + \
    0.8*torch.exp(-(x + 1)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi))

d2l.plot(x, p, 'x', 'Density')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # Define pi in TensorFlow

# Plot the probability density function for some random variable
x = tf.range(-5, 5, 0.01)
p = 0.2*tf.exp(-(x - 3)**2 / 2)/tf.sqrt(2 * tf.constant(tf.pi)) + \
    0.8*tf.exp(-(x + 1)**2 / 2)/tf.sqrt(2 * tf.constant(tf.pi))

d2l.plot(x, p, 'x', 'Density')
```

함수 값이 큰 위치는 난수 값을 찾을 가능성이 더 높은 영역을 나타냅니다.낮은 부분은 난수 값을 찾을 가능성이 낮은 영역입니다. 

### 확률 밀도 함수

이제 이에 대해 더 자세히 조사해 보겠습니다.우리는 이미 확률 변수 $X$에 대해 직관적으로 확률 밀도 함수가 무엇인지 보았습니다. 즉, 밀도 함수는 함수 $p(x)$이므로 

$$P(X \; \text{is in an}\; \epsilon \text{-sized interval around}\; x ) \approx \epsilon \cdot p(x).$$
:eqlabel:`eq_pdf_def`

그러나 이것이 $p(x)$의 속성에 대해 무엇을 의미합니까? 

첫째, 확률은 결코 음수가 아니므로 $p(x) \ge 0$도 예상해야 합니다. 

둘째, $\mathbb{R}$를 $\epsilon$ 너비의 무한한 수의 슬라이스로 슬라이스한다고 상상해 봅시다. 예를 들어 슬라이스 $(\epsilon\cdot i, \epsilon \cdot (i+1)]$을 사용합니다.이들 각각에 대해 :eqref:`eq_pdf_def`에서 확률은 대략 다음과 같다는 것을 알 수 있습니다. 

$$
P(X \; \text{is in an}\; \epsilon\text{-sized interval around}\; x ) \approx \epsilon \cdot p(\epsilon \cdot i),
$$

그래서 그들 모두를 합산하면 

$$
P(X\in\mathbb{R}) \approx \sum_i \epsilon \cdot p(\epsilon\cdot i).
$$

이것은 :numref:`sec_integral_calculus`에서 논의 된 적분의 근사치에 지나지 않으므로 다음과 같이 말할 수 있습니다. 

$$
P(X\in\mathbb{R}) = \int_{-\infty}^{\infty} p(x) \; dx.
$$

우리는 $P(X\in\mathbb{R}) = 1$가 랜덤 변수가*일부* 숫자를 취해야하기 때문에 모든 밀도에 대해 결론을 내릴 수 있다는 것을 알고 있습니다. 

$$
\int_{-\infty}^{\infty} p(x) \; dx = 1.
$$

실제로, 이것을 더 자세히 살펴보면 $a$ 및 $b$에 대해 우리는 

$$
P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.
$$

이전과 동일한 이산 근사 방법을 사용하여 코드에서 이를 근사화할 수 있습니다.이 경우 파란색 영역에서 떨어질 확률을 근사화 할 수 있습니다.

```{.python .input}
# Approximate probability using numerical integration
epsilon = 0.01
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2) / np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2) / np.sqrt(2 * np.pi)

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {np.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab pytorch
# Approximate probability using numerical integration
epsilon = 0.01
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi)) +\
    0.8*torch.exp(-(x + 1)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {torch.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab tensorflow
# Approximate probability using numerical integration
epsilon = 0.01
x = tf.range(-5, 5, 0.01)
p = 0.2*tf.exp(-(x - 3)**2 / 2) / tf.sqrt(2 * tf.constant(tf.pi)) +\
    0.8*tf.exp(-(x + 1)**2 / 2) / tf.sqrt(2 * tf.constant(tf.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.numpy().tolist()[300:800], p.numpy().tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {tf.reduce_sum(epsilon*p[300:800])}'
```

이 두 속성은 가능한 확률 밀도 함수 (또는 일반적으로 발생하는 약어의 경우*p.d.f.*) 의 공간을 정확하게 설명합니다.그것들은 음수가 아닌 함수 $p(x) \ge 0$입니다. 

$$\int_{-\infty}^{\infty} p(x) \; dx = 1.$$
:eqlabel:`eq_pdf_int_one`

적분을 사용하여 랜덤 변수가 특정 구간에있을 확률을 얻음으로써 이 함수를 해석합니다. 

$$P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.$$
:eqlabel:`eq_pdf_int_int`

:numref:`sec_distributions`에서는 여러 가지 일반적인 배포판을 볼 수 있지만 초록에서 계속 작업해 보겠습니다. 

### 누적 분포 함수

이전 섹션에서 우리는 p.d.f의 개념을 보았습니다. 실제로 이것은 연속 확률 변수를 논의하기 위해 일반적으로 접하는 방법이지만 한 가지 중요한 함정이 있습니다. p.d.f의 값 자체가 확률이 아니라 산출하기 위해 통합해야하는 함수라는 것입니다.확률.밀도가 $10$보다 큰 경우 길이가 $1/10$보다 크지 않은 한 밀도가 $10$보다 크면 아무런 문제가 없습니다.이것은 직관적이지 않을 수 있으므로 사람들은 종종*누적 분포 함수* 또는 확률인*인 c.d.f. 의 관점에서 생각합니다. 

특히 :eqref:`eq_pdf_int_int`를 사용하여 밀도가 $p(x)$인 랜덤 변수 $X$에 대한 c.d.f. 를 다음과 같이 정의합니다. 

$$
F(x) = \int _ {-\infty}^{x} p(x) \; dx = P(X \le x).
$$

몇 가지 속성을 살펴 보겠습니다. 

* $F(x) \rightarrow 0$는 $x\rightarrow -\infty$입니다.
* $F(x) \rightarrow 1$는 $x\rightarrow \infty$로 알려져 있습니다.
* $F(x)$는 감소하지 않습니다 ($y > x \implies F(y) \ge F(x)$).
* $X$가 계량형 랜덤 변수인 경우 $F(x)$는 연속형 (점프 없음) 입니다.

네 번째 글 머리 기호에서 $X$이 이산형이라면 사실이 아닐 것입니다. 예를 들어 $0$ 및 $1$ 값을 모두 확률 $1/2$으로 취합니다.이 경우 

$$
F(x) = \begin{cases}
0 & x < 0, \\
\frac{1}{2} & x < 1, \\
1 & x \ge 1.
\end{cases}
$$

이 예에서는 cd.f. 로 작업 할 때의 이점 중 하나, 동일한 프레임 워크에서 연속 또는 이산 확률 변수를 처리하는 기능 또는 실제로 두 변수의 혼합물 (동전 뒤집기: 머리가 주사위의 롤을 반환하는 경우, 꼬리가 다트 중심에서 다트 던지기 거리를 반환하는 경우) 을 볼 수 있습니다.보드). 

### 수단

랜덤 변수 $X$를 다루고 있다고 가정합니다.분포 자체는 해석하기 어려울 수 있습니다.랜덤 변수의 동작을 간결하게 요약할 수 있는 것이 유용한 경우가 많습니다.랜덤 변수의 동작을 포착하는 데 도움이 되는 숫자를*요약 통계*라고 합니다.가장 일반적으로 발생하는 것은*평균*, *분산* 및*표준 편차*입니다. 

*mean*은 랜덤 변수의 평균값을 인코딩합니다.확률 $p_i$의 값 $x_i$를 취하는 이산 확률 변수 $X$이있는 경우 평균은 가중 평균으로 주어집니다. 값에 랜덤 변수가 해당 값에 걸릴 확률을 곱한 값을 합산합니다. 

$$\mu_X = E[X] = \sum_i x_i p_i.$$
:eqlabel:`eq_exp_def`

평균을 해석하는 방법은 (주의하지만) 기본적으로 랜덤 변수가 어디에 위치하는지 알려주는 것입니다. 

이 섹션 전체에서 살펴볼 최소한의 예로서 $X$을 확률 $p$, 확률이 $p$, 확률이 $p$인 $a+2$ 및 확률이 $1-2p$인 $a$을 취하는 확률 변수로 사용하겠습니다.:eqref:`eq_exp_def`를 사용하여 $a$ 및 $p$ 중 가능한 모든 선택에 대해 평균은 다음과 같은 계산을 할 수 있습니다. 

$$
\mu_X = E[X] = \sum_i x_i p_i = (a-2)p + a(1-2p) + (a+2)p = a.
$$

따라서 평균이 $a$임을 알 수 있습니다.$a$는 랜덤 변수를 중심으로 한 위치이기 때문에 이것은 직관과 일치합니다. 

유용하므로 몇 가지 속성을 요약해 보겠습니다. 

* 임의의 랜덤 변수 $X$과 숫자 $a$와 $b$에 대해, 우리는 $\mu_{aX+b} = a\mu_X + b$을 갖습니다.
* 두 개의 랜덤 변수 $X$과 $Y$가 있는 경우 $\mu_{X+Y} = \mu_X+\mu_Y$가 있습니다.

평균은 랜덤 변수의 평균 동작을 이해하는 데 유용하지만 평균만으로는 완전히 직관적으로 이해할 수 없습니다.판매 당 $\$10\ pm\ $1$의 수익을 창출하는 것은 동일한 평균값을 가지고 있음에도 불구하고 판매 당 $\$10\ pm\ $15$를 만드는 것과는 매우 다릅니다.두 번째 변동이 훨씬 더 크므로 훨씬 더 큰 위험을 나타냅니다.따라서 랜덤 변수의 동작을 이해하려면 최소한 하나 이상의 측정이 필요합니다. 랜덤 변수가 얼마나 크게 변동하는지에 대한 척도입니다. 

### 분산

이로 인해 랜덤 변수의*분산*을 고려할 수 있습니다.이것은 랜덤 변수가 평균에서 얼마나 멀리 벗어나는지에 대한 정량적 측도입니다.$X - \mu_X$라는 표현을 생각해 보십시오.랜덤 변수와 평균의 편차입니다.이 값은 양수 또는 음수일 수 있으므로 편차의 크기를 측정하기 위해 양수로 만들어야 합니다. 

시도해야 할 합리적인 것은 $\left|X-\mu_X\right|$를 보는 것입니다. 실제로는*평균 절대 편차*라는 유용한 양으로 이어지지만 다른 수학 및 통계 영역과의 연결로 인해 사람들은 종종 다른 솔루션을 사용합니다. 

특히 $(X-\mu_X)^2.$를 봅니다. 평균을 취하여 이 수량의 일반적인 크기를 보면 분산에 도달합니다. 

$$\sigma_X^2 = \mathrm{Var}(X) = E\left[(X-\mu_X)^2\right] = E[X^2] - \mu_X^2.$$
:eqlabel:`eq_var_def`

:eqref:`eq_var_def`의 마지막 평등은 중간에 정의를 확장하고 기대의 속성을 적용함으로써 유지됩니다. 

$X$이 확률 $a-2$을 가진 값 $a-2$, 확률이 $p$인 $a+2$ 및 확률이 $1-2p$인 $a$을 취하는 확률 변수입니다.이 경우 $\mu_X = a$이므로 계산해야 할 것은 $E\left[X^2\right]$뿐입니다.이 작업은 쉽게 수행할 수 있습니다. 

$$
E\left[X^2\right] = (a-2)^2p + a^2(1-2p) + (a+2)^2p = a^2 + 8p.
$$

따라서 :eqref:`eq_var_def`에 의해 우리의 분산은 다음과 같음을 알 수 있습니다. 

$$
\sigma_X^2 = \mathrm{Var}(X) = E[X^2] - \mu_X^2 = a^2 + 8p - a^2 = 8p.
$$

이 결과는 다시 의미가 있습니다.가장 큰 $p$는 $1/2$이 될 수 있으며, 이는 동전 플립으로 $a-2$ 또는 $a+2$를 선택하는 것에 해당합니다.이 분산은 $4$이라는 사실에 해당합니다. $a-2$과 $a+2$가 모두 평균에서 $2$단위 떨어져 있고 $2^2 = 4$이라는 사실에 해당합니다.스펙트럼의 다른 쪽 끝에서 $p=0$인 경우 이 랜덤 변수는 항상 값 $0$를 취하므로 분산이 전혀 없습니다. 

아래에 몇 가지 분산 속성을 나열합니다. 

* 임의의 랜덤 변수 $X$, $\mathrm{Var}(X) \ge 0$에 대해, $X$이 상수인 경우에만 $\mathrm{Var}(X) = 0$를 사용합니다.
* 임의의 랜덤 변수 $X$과 숫자 $a$와 $b$에 대해, 우리는 $\mathrm{Var}(aX+b) = a^2\mathrm{Var}(X)$를 갖습니다.
* 두 개의*독립* 랜덤 변수 $X$과 $Y$가 있는 경우 $\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$가 있습니다.

이 값을 해석할 때 약간의 딸꾹질이 발생할 수 있습니다.특히 이 계산을 통해 단위를 추적하면 어떤 일이 발생하는지 상상해 봅시다.웹 페이지에서 제품에 할당된 별 등급을 사용하고 있다고 가정해 보겠습니다.그런 다음 $a$, $a-2$ 및 $a+2$이 모두 별 단위로 측정됩니다.마찬가지로 평균 $\mu_X$도 별에서 측정됩니다 (가중 평균).그러나 분산에 도달하면 즉시 문제가 발생합니다. 즉, *제곱 별*의 단위인 $(X-\mu_X)^2$를 살펴보겠습니다.이는 분산 자체가 원래 측정값과 비교할 수 없음을 의미합니다.해석이 가능하도록 하려면 원래 단위로 돌아가야 합니다. 

### 표준 편차

이 요약 통계는 제곱근을 취하여 분산에서 항상 추론할 수 있습니다!따라서*표준 편차*를 다음과 같이 정의합니다. 

$$
\sigma_X = \sqrt{\mathrm{Var}(X)}.
$$

이 예에서는 이제 표준 편차가 $\sigma_X = 2\sqrt{2p}$임을 의미합니다.검토 예에서 별 단위를 다루는 경우 $\sigma_X$는 다시 별 단위입니다. 

분산에 대한 특성을 표준 편차에 대해 다시 계산할 수 있습니다. 

* 임의의 랜덤 변수 $X$, $\sigma_{X} \ge 0$에 대해 설명합니다.
* 임의의 랜덤 변수 $X$과 숫자 $a$ 및 $b$에 대해 우리는 $\sigma_{aX+b} = |a|\sigma_{X}$를 갖습니다.
* 두 개의*독립* 랜덤 변수 $X$과 $Y$가 있는 경우 $\sigma_{X+Y} = \sqrt{\sigma_{X}^2 + \sigma_{Y}^2}$가 있습니다.

이 순간 “표준 편차가 원래 확률 변수의 단위라면 그 랜덤 변수와 관련하여 그릴 수있는 것을 나타내는가?” 라고 묻는 것은 당연합니다.대답은 '예'입니다!실제로 평균이 랜덤 변수의 일반적인 위치를 알려주는 것처럼 표준 편차는 해당 랜덤 변수의 일반적인 변동 범위를 제공합니다.우리는 체비셰프의 불평등으로 알려진 것을 엄격하게 만들 수 있습니다. 

$$P\left(X \not\in [\mu_X - \alpha\sigma_X, \mu_X + \alpha\sigma_X]\right) \le \frac{1}{\alpha^2}.$$
:eqlabel:`eq_chebyshev`

또는 평균의 $\alpha=10$달러, 99달러\ %$ of the samples from any random variable fall within $10$ 표준 편차의 경우 구두로 명시합니다.이를 통해 표준 요약 통계를 즉시 해석할 수 있습니다. 

이 문장이 어떻게 미묘한지 알아보기 위해 실행 예제를 다시 살펴 보겠습니다. 여기서 $X$는 확률 $p$, 확률 $p$의 $a+2$ 및 확률이 $a$인 $a$의 값을 취하는 확률 변수입니다.평균이 $a$이고 표준 편차가 $2\sqrt{2p}$라는 것을 확인했습니다.즉, 체비셰프의 불평등 :eqref:`eq_chebyshev`를 $\alpha = 2$과 함께 취하면 다음과 같은 표현이 

$$
P\left(X \not\in [a - 4\sqrt{2p}, a + 4\sqrt{2p}]\right) \le \frac{1}{4}.
$$

즉, 시간의 75\ %$, 이 랜덤 변수는 $p$의 모든 값에 대해 이 간격 내에 포함됩니다.이제 $p\ 오른쪽 화살표 0$, this interval also converges to the single point $a$.  But we know that our random variable takes the values $a-2로, a$, and $a+2$ only so eventually we can be certain $a-2$ and $a+2$가 간격을 벗어날 것임을 알 수 있습니다!문제는 $p$가 어떤 일이 일어나는가입니다.그래서 우리는 해결하고자 합니다: $p$가 $a+4\sqrt{2p} = a+2$을 수행하는 작업에 대해, 이는 $p=1/8$, *정확히* 첫 번째 $p$일 때 해결됩니다. 분포에서 나온 표본의 $1/4$ 이하가 왼쪽의 구간 ($1/8$) 을 벗어날 것이라는 주장을 위반하지 않고 발생할 수 있습니다.$1/8$ (오른쪽으로) 을 참조하십시오. 

이것을 시각화해 보겠습니다.세 값을 확률에 비례하는 높이를 가진 세 개의 수직 막대로 얻을 확률을 보여줍니다.간격은 중간에 수평선으로 그려집니다.첫 번째 그림은 구간에 모든 점이 안전하게 포함되는 $p > 1/8$에 대해 어떤 일이 발생하는지 보여줍니다.

```{.python .input}
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * np.sqrt(2 * p),
                   a + 4 * np.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * np.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * np.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, 0.2)
```

```{.python .input}
#@tab pytorch
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * torch.sqrt(2 * p),
                   a + 4 * torch.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * torch.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * torch.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, torch.tensor(0.2))
```

```{.python .input}
#@tab tensorflow
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * tf.sqrt(2 * p),
                   a + 4 * tf.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * tf.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * tf.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, tf.constant(0.2))
```

두 번째는 $p = 1/8$에서 간격이 두 점에 정확히 닿는다는 것을 보여줍니다.이것은 부등식을 true로 유지하면서 더 작은 간격을 취할 수 없기 때문에 부등식이*sharp*임을 보여줍니다.

```{.python .input}
# Plot interval when p = 1/8
plot_chebyshev(0.0, 0.125)
```

```{.python .input}
#@tab pytorch
# Plot interval when p = 1/8
plot_chebyshev(0.0, torch.tensor(0.125))
```

```{.python .input}
#@tab tensorflow
# Plot interval when p = 1/8
plot_chebyshev(0.0, tf.constant(0.125))
```

세 번째는 $p < 1/8$의 경우 구간에 중심만 포함된다는 것을 보여줍니다.확률의 $1/4$를 넘지 않도록 하기 위해서만 필요했기 때문에 부등식을 무효화하지는 않습니다. 즉, $p < 1/8$이 되면 $a-2$와 $a+2$의 두 점을 버릴 수 있습니다.

```{.python .input}
# Plot interval when p < 1/8
plot_chebyshev(0.0, 0.05)
```

```{.python .input}
#@tab pytorch
# Plot interval when p < 1/8
plot_chebyshev(0.0, torch.tensor(0.05))
```

```{.python .input}
#@tab tensorflow
# Plot interval when p < 1/8
plot_chebyshev(0.0, tf.constant(0.05))
```

### 연속체의 평균과 분산

이것은 모두 이산 확률 변수의 관점에서 이루어졌지만 연속 확률 변수의 경우는 비슷합니다.이것이 어떻게 작동하는지 직관적으로 이해하기 위해 실수 선을 $(\epsilon i, \epsilon (i+1)]$에 의해 주어진 길이 $\epsilon$의 간격으로 분할한다고 상상해보십시오.이렇게 하면 연속 확률 변수가 이산형으로 만들어졌고 :eqref:`eq_exp_def`를 사용할 수 있습니다. 

$$
\begin{aligned}
\mu_X & \approx \sum_{i} (\epsilon i)P(X \in (\epsilon i, \epsilon (i+1)]) \\
& \approx \sum_{i} (\epsilon i)p_X(\epsilon i)\epsilon, \\
\end{aligned}
$$

여기서 $p_X$는 $X$의 밀도입니다.이것은 $xp_X(x)$의 적분에 대한 근사치이므로 다음과 같은 결론을 내릴 수 있습니다. 

$$
\mu_X = \int_{-\infty}^\infty xp_X(x) \; dx.
$$

마찬가지로 :eqref:`eq_var_def`를 사용하면 분산을 다음과 같이 작성할 수 있습니다. 

$$
\sigma^2_X = E[X^2] - \mu_X^2 = \int_{-\infty}^\infty x^2p_X(x) \; dx - \left(\int_{-\infty}^\infty xp_X(x) \; dx\right)^2.
$$

이 경우에도 평균, 분산 및 표준 편차에 대해 위에서 언급한 모든 내용이 계속 적용됩니다.예를 들어, 밀도를 갖는 랜덤 변수를 고려한다면 

$$
p(x) = \begin{cases}
1 & x \in [0,1], \\
0 & \text{otherwise}.
\end{cases}
$$

우리가 할 수 있는 

$$
\mu_X = \int_{-\infty}^\infty xp(x) \; dx = \int_0^1 x \; dx = \frac{1}{2}.
$$

과 

$$
\sigma_X^2 = \int_{-\infty}^\infty x^2p(x) \; dx - \left(\frac{1}{2}\right)^2 = \frac{1}{3} - \frac{1}{4} = \frac{1}{12}.
$$

경고로, *Cauchy 분포*라고 하는 예를 하나 더 살펴보겠습니다.이것은 다음과 같이 주어진 p.d.f. 를 사용한 분포입니다. 

$$
p(x) = \frac{1}{1+x^2}.
$$

```{.python .input}
# Plot the Cauchy distribution p.d.f.
x = np.arange(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
# Plot the Cauchy distribution p.d.f.
x = torch.arange(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
# Plot the Cauchy distribution p.d.f.
x = tf.range(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

이 함수는 결백 해 보이며 실제로 적분 표를 참조하면 그 아래에 영역 1이 있음을 보여 주므로 연속 확률 변수를 정의합니다. 

무엇이 잘못되었는지 알아보기 위해, 이것의 분산을 계산해 봅시다.여기에는 :eqref:`eq_var_def` 컴퓨팅을 사용하는 것이 포함됩니다. 

$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx.
$$

내부 함수는 다음과 같습니다.

```{.python .input}
# Plot the integrand needed to compute the variance
x = np.arange(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

```{.python .input}
#@tab pytorch
# Plot the integrand needed to compute the variance
x = torch.arange(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

```{.python .input}
#@tab tensorflow
# Plot the integrand needed to compute the variance
x = tf.range(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

이 함수는 본질적으로 0에 가까운 작은 딥을 가진 상수 함수이기 때문에 분명히 그 아래에 무한한 영역을 가지고 있습니다. 

$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx = \infty.
$$

즉, 잘 정의된 유한 분산이 없습니다. 

그러나 더 깊이 들여다 보면 훨씬 더 혼란스러운 결과를 알 수 있습니다.:eqref:`eq_exp_def`를 사용하여 평균을 계산해 보겠습니다.변수 변경 공식을 사용하면 

$$
\mu_X = \int_{-\infty}^{\infty} \frac{x}{1+x^2} \; dx = \frac{1}{2}\int_1^\infty \frac{1}{u} \; du.
$$

내부의 적분은 로그의 정의이므로 본질적으로 $\log(\infty) = \infty$이므로 잘 정의 된 평균값도 없습니다! 

기계 학습 과학자들은 이러한 문제를 대부분 처리 할 필요가 없도록 모델을 정의하며, 대부분의 경우 잘 정의 된 평균과 분산을 가진 랜덤 변수를 처리합니다.그러나*무거운 꼬리*를 가진 랜덤 변수 (즉, 큰 값을 얻을 확률이 평균 또는 분산과 같은 것을 정의하지 않을 정도로 큰 확률 변수) 는 물리적 시스템을 모델링하는 데 유용하므로 존재한다는 것을 아는 것이 좋습니다. 

### 관절 밀도 함수

위의 작업은 모두 단일 실수 값 확률 변수로 작업하고 있다고 가정합니다.하지만 잠재적으로 높은 상관 관계가 있는 확률 변수를 두 개 이상 다루면 어떨까요?이러한 상황은 기계 학습의 표준입니다. 이미지의 $(i, j)$ 좌표에서 픽셀의 빨간색 값을 인코딩하는 $R_{i, j}$ 또는 시간 $t$의 주가에 의해 주어진 랜덤 변수인 $P_t$와 같은 랜덤 변수를 상상해보십시오.주변 픽셀은 색상이 비슷하고 주변 시간대는 가격이 비슷한 경향이 있습니다.우리는 그것들을 별도의 랜덤 변수로 취급 할 수 없으며 성공적인 모델을 만들 것으로 기대합니다 (:numref:`sec_naive_bayes`에서는 이러한 가정으로 인해 실적이 저조한 모델을 보게 될 것입니다).이러한 상관 관계가 있는 연속 확률 변수를 처리하기 위해 수학적 언어를 개발해야 합니다. 

고맙게도 :numref:`sec_integral_calculus`의 여러 적분을 사용하면 이러한 언어를 개발할 수 있습니다.간단히 말해서 상관 관계를 가질 수 있는 두 개의 랜덤 변수 $X, Y$가 있다고 가정합니다.그런 다음 단일 변수의 경우와 마찬가지로 다음과 같은 질문을 할 수 있습니다. 

$$
P(X \;\text{is in an}\; \epsilon \text{-sized interval around}\; x \; \text{and} \;Y \;\text{is in an}\; \epsilon \text{-sized interval around}\; y ).
$$

단일 변수 사례와 유사한 추론은 이것이 대략적이어야 함을 보여줍니다. 

$$
P(X \;\text{is in an}\; \epsilon \text{-sized interval around}\; x \; \text{and} \;Y \;\text{is in an}\; \epsilon \text{-sized interval around}\; y ) \approx \epsilon^{2}p(x, y),
$$

일부 함수 $p(x, y)$에 대해 설명합니다.이를 접합 밀도는 $X$ 및 $Y$라고 합니다.단일 변수 사례에서 보았 듯이 이와 유사한 속성이 적용됩니다.즉: 

* $p(x, y) \ge 0$;
* $\int _ {\mathbb{R}^2} p(x, y) \;dx \;dy = 1$;
* $P((X, Y) \in \mathcal{D}) = \int _ {\mathcal{D}} p(x, y) \;dx \;dy$.

이런 식으로 잠재적으로 상관 관계가 있는 여러 확률 변수를 처리할 수 있습니다.두 개 이상의 랜덤 변수로 작업하려는 경우 $p(\mathbf{x}) = p(x_1, \ldots, x_n)$를 고려하여 다변량 밀도를 원하는 만큼 많은 좌표로 확장할 수 있습니다.음수가 아니며 하나의 총 적분을 갖는 것과 동일한 특성이 여전히 유지됩니다. 

### 한계 분포 여러 변수를 다룰 때 우리는 종종 관계를 무시하고 “이 변수가 어떻게 분포되어 있는가?”이러한 분포를*주변 분포*라고 합니다. 

구체적으로 말하자면, 접합 밀도가 $p _ {X, Y}(x, y)$로 주어진 두 개의 랜덤 변수 $X, Y$가 있다고 가정해 보겠습니다.아래 첨자를 사용하여 밀도가 어떤 랜덤 변수인지 나타냅니다.주변 분포를 찾는 문제는 이 함수를 사용하여 $p _ X(x)$을 찾는 것입니다. 

대부분의 경우와 마찬가지로 직관적 인 그림으로 돌아가서 무엇이 사실인지 파악하는 것이 가장 좋습니다.밀도는 함수 $p _ X$이므로 

$$
P(X \in [x, x+\epsilon]) \approx \epsilon \cdot p _ X(x).
$$

$Y$에 대한 언급은 없지만, 우리가 주어진 모든 것이 $p _{X, Y}$라면, 우리는 어떻게 든 $Y$를 포함시켜야 합니다.먼저 이것이 다음과 같다는 것을 관찰 할 수 있습니다. 

$$
P(X \in [x, x+\epsilon] \text{, and } Y \in \mathbb{R}) \approx \epsilon \cdot p _ X(x).
$$

우리의 밀도는 이 경우에 어떤 일이 일어나는지 직접적으로 알려주지 않습니다. 우리는 $y$에서도 작은 간격으로 나누어야 합니다. 

$$
\begin{aligned}
\epsilon \cdot p _ X(x) & \approx \sum _ {i} P(X \in [x, x+\epsilon] \text{, and } Y \in [\epsilon \cdot i, \epsilon \cdot (i+1)]) \\
& \approx \sum _ {i} \epsilon^{2} p _ {X, Y}(x, \epsilon\cdot i).
\end{aligned}
$$

![By summing along the columns of our array of probabilities, we are able to obtain the marginal distribution for just the random variable represented along the $x$-axis.](../img/marginal.svg)
:label:`fig_marginal`

이것은 :numref:`fig_marginal`에 표시된 것처럼 일련의 사각형을 따라 밀도 값을 합산하도록 알려줍니다.실제로 양쪽에서 엡실론의 한 요소를 취소하고 오른쪽의 합이 $y$에 대한 적분임을 인식 한 후 다음과 같이 결론을 내릴 수 있습니다. 

$$
\begin{aligned}
 p _ X(x) &  \approx \sum _ {i} \epsilon p _ {X, Y}(x, \epsilon\cdot i) \\
 & \approx \int_{-\infty}^\infty p_{X, Y}(x, y) \; dy.
\end{aligned}
$$

따라서 우리는 

$$
p _ X(x) = \int_{-\infty}^\infty p_{X, Y}(x, y) \; dy.
$$

이것은 한계 분포를 얻기 위해 우리가 신경 쓰지 않는 변수에 대해 적분한다는 것을 말해줍니다.이 과정을 종종 불필요한 변수를*적분* 또는*소외된*이라고 합니다. 

### 공분산

여러 랜덤 변수를 다룰 때 알아두면 도움이 되는 추가 요약 통계량은*공분산*입니다.이 값은 두 랜덤 변수가 함께 변동하는 정도를 측정합니다. 

두 개의 확률 변수 $X$과 $Y$가 있다고 가정합니다. 먼저 확률 $p_{ij}$의 값 $(x_i, y_j)$을 사용하여 이산형이라고 가정 해 보겠습니다.이 경우 공분산은 다음과 같이 정의됩니다. 

$$\sigma_{XY} = \mathrm{Cov}(X, Y) = \sum_{i, j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij}. = E[XY] - E[X]E[Y].$$
:eqlabel:`eq_cov_def`

이것을 직관적으로 생각하려면 다음과 같은 확률 변수 쌍을 고려하십시오.$X$이 $1$ 및 $3$의 값을 취하고 $Y$가 $-1$ 및 $3$의 값을 취한다고 가정합니다.다음과 같은 확률이 있다고 가정합니다. 

$$
\begin{aligned}
P(X = 1 \; \text{and} \; Y = -1) & = \frac{p}{2}, \\
P(X = 1 \; \text{and} \; Y = 3) & = \frac{1-p}{2}, \\
P(X = 3 \; \text{and} \; Y = -1) & = \frac{1-p}{2}, \\
P(X = 3 \; \text{and} \; Y = 3) & = \frac{p}{2},
\end{aligned}
$$

여기서 $p$는 우리가 선택할 수 있는 $[0,1]$의 매개 변수입니다.$p=1$인 경우 둘 다 항상 동시에 최소값 또는 최대값이고 $p=0$인 경우 뒤집힌 값을 동시에 사용할 수 있습니다 (하나는 작으면 크고 그 반대도 마찬가지입니다).$p=1/2$인 경우 네 가지 가능성이 모두 동일하며 둘 다 관련되어서는 안 됩니다.공분산을 계산해 보겠습니다.먼저 $\mu_X = 2$과 $\mu_Y = 1$을 기록해 두십시오. 따라서 :eqref:`eq_cov_def`를 사용하여 계산할 수 있습니다. 

$$
\begin{aligned}
\mathrm{Cov}(X, Y) & = \sum_{i, j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij} \\
& = (1-2)(-1-1)\frac{p}{2} + (1-2)(3-1)\frac{1-p}{2} + (3-2)(-1-1)\frac{1-p}{2} + (3-2)(3-1)\frac{p}{2} \\
& = 4p-2.
\end{aligned}
$$

$p=1$ (둘 다 동시에 최대로 양수 또는 음성인 경우) 의 공분산이 $2$인 경우$p=0$ (반전된 경우) 인 경우 공분산은 $-2$입니다.마지막으로 $p=1/2$ (관련이 없는 경우) 인 경우 공분산은 $0$입니다.따라서 공분산이이 두 확률 변수가 어떻게 관련되어 있는지 측정합니다. 

공분산에 대한 간단한 참고 사항은 이러한 선형 관계만 측정한다는 것입니다.$X = Y^2$와 같은 더 복잡한 관계는 $Y$이 $\{-2, -1, 0, 1, 2\}$에서 무작위로 선택되고 동일한 확률로 누락될 수 있습니다.실제로 빠른 계산은 이러한 확률 변수가 다른 변수의 결정론적 함수임에도 불구하고 공분산이 0임을 보여줍니다. 

연속 랜덤 변수의 경우 거의 동일한 이야기가 유지됩니다.이 시점에서 우리는 이산형과 연속 사이의 전환을 수행하는 데 매우 편하므로 유도없이 :eqref:`eq_cov_def`의 연속 아날로그를 제공 할 것입니다. 

$$
\sigma_{XY} = \int_{\mathbb{R}^2} (x-\mu_X)(y-\mu_Y)p(x, y) \;dx \;dy.
$$

시각화를 위해 조정 가능한 공분산을 갖는 랜덤 변수 모음을 살펴 보겠습니다.

```{.python .input}
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = np.random.normal(0, 1, 500)
    Y = covs[i]*X + np.random.normal(0, 1, (500))

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.asnumpy(), Y.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = torch.randn(500)
    Y = covs[i]*X + torch.randn(500)

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = tf.random.normal((500, ))
    Y = covs[i]*X + tf.random.normal((500, ))

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

공분산의 몇 가지 속성을 살펴 보겠습니다. 

* 임의의 랜덤 변수 $X$, $\mathrm{Cov}(X, X) = \mathrm{Var}(X)$에 대해 설명합니다.
* 임의의 랜덤 변수 $X, Y$ 및 숫자 $a$ 및 $b$, $\mathrm{Cov}(aX+b, Y) = \mathrm{Cov}(X, aY+b) = a\mathrm{Cov}(X, Y)$에 대해 설명합니다.
* $X$과 $Y$가 독립적이라면 $\mathrm{Cov}(X, Y) = 0$입니다.

또한 공분산을 사용하여 이전에 본 관계를 확장할 수 있습니다.즉, $X$와 $Y$는 두 개의 독립적 랜덤 변수입니다. 

$$
\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y).
$$

공분산에 대한 지식이 있으면 이 관계를 확장할 수 있습니다.사실, 어떤 대수학은 일반적으로 

$$
\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\mathrm{Cov}(X, Y).
$$

이를 통해 상관 랜덤 변수에 대한 분산 합계 규칙을 일반화할 수 있습니다. 

### 상관 관계

평균과 분산의 경우처럼 이제 단위를 고려해 보겠습니다.$X$이 한 단위 (예: 인치) 로 측정되고 $Y$가 다른 단위 (예: 달러) 로 측정되는 경우 공분산은 이 두 단위 $\text{inches} \times \text{dollars}$의 곱으로 측정됩니다.이러한 단위는 해석하기 어려울 수 있습니다.이 경우 우리가 종종 원하는 것은 관련성에 대한 단위가 없는 측정입니다.실제로 우리는 종종 정확한 양적 상관 관계에 신경 쓰지 않고 상관 관계가 같은 방향인지, 관계가 얼마나 강한지를 묻습니다. 

무엇이 의미가 있는지 알아보기 위해 사고 실험을 수행해 보겠습니다.인치와 달러 단위의 랜덤 변수를 인치와 센트로 변환한다고 가정합니다.이 경우 랜덤 변수 $Y$에 $100$을 곱합니다.정의를 통해 작업하면 $\mathrm{Cov}(X, Y)$에 $100$이 곱해질 것임을 의미합니다.따라서 이 경우 단위를 변경하면 공분산이 $100$의 요인만큼 변경된다는 것을 알 수 있습니다.따라서 단위 불변 상관 관계 측정값을 찾으려면 $100$으로 확장되는 다른 것으로 나누어야 합니다.실제로 표준 편차라는 명확한 후보가 있습니다!실제로*상관 계수*를 다음과 같이 정의하면 

$$\rho(X, Y) = \frac{\mathrm{Cov}(X, Y)}{\sigma_{X}\sigma_{Y}},$$
:eqlabel:`eq_cor_def`

우리는 이것이 단위가 없는 값이라는 것을 알 수 있습니다.약간의 수학은이 숫자가 $-1$에서 $1$ 사이이며 $1$는 최대로 양의 상관 관계가 있음을 보여줄 수 있지만 $-1$는 최대로 음의 상관 관계가 있음을 의미합니다. 

위의 명시 적 이산 예제로 돌아가서 $\sigma_X = 1$와 $\sigma_Y = 2$을 볼 수 있으므로 :eqref:`eq_cor_def`를 사용하여 두 확률 변수 간의 상관 관계를 계산하여 

$$
\rho(X, Y) = \frac{4p-2}{1\cdot 2} = 2p-1.
$$

이제 이 범위는 $-1$에서 $1$ 사이이며 예상되는 동작은 $1$는 가장 상관 관계가 있음을 의미하고 $-1$는 최소한의 상관 관계를 의미합니다. 

또 다른 예로 $X$를 랜덤 변수로, $Y=aX+b$를 $X$의 선형 결정론적 함수로 간주하십시오.그러면 이를 계산할 수 있습니다. 

$$\sigma_{Y} = \sigma_{aX+b} = |a|\sigma_{X},$$

$$\mathrm{Cov}(X, Y) = \mathrm{Cov}(X, aX+b) = a\mathrm{Cov}(X, X) = a\mathrm{Var}(X),$$

따라서 :eqref:`eq_cor_def`에 의해 

$$
\rho(X, Y) = \frac{a\mathrm{Var}(X)}{|a|\sigma_{X}^2} = \frac{a}{|a|} = \mathrm{sign}(a).
$$

따라서 우리는 상관 관계가 $a > 0$에 대해 $+1$이고 $a < 0$에 대해 $-1$임을 알 수 있습니다. 이는 상관 관계가 변동이 취하는 척도가 아니라 두 확률 변수가 관련된 정도와 방향성을 측정한다는 것을 보여줍니다. 

조정 가능한 상관 관계를 갖는 확률 변수 모음을 다시 플로팅하겠습니다.

```{.python .input}
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = np.random.normal(0, 1, 500)
    Y = cors[i] * X + np.sqrt(1 - cors[i]**2) * np.random.normal(0, 1, 500)

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.asnumpy(), Y.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = torch.randn(500)
    Y = cors[i] * X + torch.sqrt(torch.tensor(1) -
                                 cors[i]**2) * torch.randn(500)

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = tf.random.normal((500, ))
    Y = cors[i] * X + tf.sqrt(tf.constant(1.) -
                                 cors[i]**2) * tf.random.normal((500, ))

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

아래에 상관 관계의 몇 가지 속성을 나열 해 보겠습니다. 

* 임의의 랜덤 변수 $X$, $\rho(X, X) = 1$에 대해 설명합니다.
* 임의의 랜덤 변수 $X, Y$ 및 숫자 $a$ 및 $b$, $\rho(aX+b, Y) = \rho(X, aY+b) = \rho(X, Y)$에 대해 설명합니다.
* $X$과 $Y$가 분산이 0이 아닌 독립적이면 $\rho(X, Y) = 0$입니다.

마지막으로, 이러한 공식 중 일부는 친숙하다고 느낄 수 있습니다.실제로 $\mu_X = \mu_Y = 0$라고 가정하여 모든 것을 확장하면 이것이 

$$
\rho(X, Y) = \frac{\sum_{i, j} x_iy_ip_{ij}}{\sqrt{\sum_{i, j}x_i^2 p_{ij}}\sqrt{\sum_{i, j}y_j^2 p_{ij}}}.
$$

이것은 항의 곱을 항의 합의 제곱근으로 나눈 것과 같습니다.이것은 서로 다른 좌표가 $p_{ij}$로 가중치를 부여한 두 벡터 $\mathbf{v}, \mathbf{w}$ 사이의 각도의 코사인에 대한 공식입니다. 

$$
\cos(\theta) = \frac{\mathbf{v}\cdot \mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|} = \frac{\sum_{i} v_iw_i}{\sqrt{\sum_{i}v_i^2}\sqrt{\sum_{i}w_i^2}}.
$$

실제로 규범이 표준 편차와 관련이 있고 상관 관계가 각도의 코사인이라고 생각하면 기하학에서 얻은 직관의 대부분은 랜덤 변수에 대한 생각에 적용될 수 있습니다. 

## 요약* 연속형 랜덤 변수는 값의 연속체를 취할 수 있는 랜덤 변수입니다.이산 확률 변수에 비해 작업하기가 더 어려워지는 몇 가지 기술적 어려움이 있습니다.* 확률 밀도 함수를 사용하면 곡선 아래 영역이 특정 구간에서해당 구간의 샘플 점. * 누적 분포 함수는 랜덤 변수가 주어진 분계점보다 작다는 것을 관측할 확률입니다.이산 변수와 계량형 변수를 통합하는 유용한 대체 관점을 제공할 수 있습니다.* 평균은 랜덤 변수의 평균값입니다. * 분산은 랜덤 변수와 그 평균 간의 차이에 대한 예상 제곱입니다.* 표준 편차는 분산의 제곱근입니다.랜덤 변수가 취할 수 있는 값의 범위를 측정하는 것으로 생각할 수 있습니다.* 체비쇼프의 부등식은 대부분의 경우 랜덤 변수를 포함하는 명시적인 구간을 제공함으로써 이러한 직관을 엄격하게 만들 수 있습니다.* 관절 밀도를 사용하면 상관 관계가 있는 확률 변수로 작업할 수 있습니다.원하는 랜덤 변수의 분포를 얻기 위해 원치 않는 랜덤 변수를 적분하여 관절 밀도를 소외 할 수 있습니다.* 공분산 및 상관 계수는 상관 관계가 있는 두 확률 변수 간의 선형 관계를 측정하는 방법을 제공합니다. 

## 연습 문제 

1. Suppose that we have the random variable with density given by $p(x) = \frac{1}{x^2}$ for $x \ge 1$ and $p(x) = 0$ otherwise.  What is $P(X > 2)$?
2. The Laplace distribution is a random variable whose density is given by $p(x = \frac{1}{2}e^{-|x|}$.  What is the mean and the standard deviation of this function?  As a hint, $\int_0^\infty xe^{-x} \; dx = 1$ and $\int_0^\infty x^2e^{-x} \; dx = 2$.
3. I walk up to you on the street and say "I have a random variable with mean $1$, standard deviation $2$, and I observed $25\%$ of my samples taking a value larger than $9$."  Do you believe me?  Why or why not?
4. Suppose that you have two random variables $X, Y$, with joint density given by $p_{XY}(x, y) = 4xy$ for $x, y \in [0,1]$ and $p_{XY}(x, y) = 0$ otherwise.  What is the covariance of $X$ and $Y$?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/415)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1094)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1095)
:end_tab:
