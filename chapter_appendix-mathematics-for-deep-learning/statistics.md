# 통계
:label:`sec_statistics`

의심 할 여지없이 최고의 딥 러닝 전문가가 되려면 최첨단 고정밀 모델을 학습하는 능력이 중요합니다.그러나 개선이 중요한 시기가 언제인지 또는 훈련 과정의 무작위 변동으로 인한 결과만 명확하지 않은 경우가 많습니다.추정값의 불확실성을 논의하려면 몇 가지 통계를 배워야 합니다. 

*통계*에 대한 가장 초기의 참조는 $9^{\mathrm{th}}$세기의 아랍 학자 Al-Kindi로 거슬러 올라갈 수 있습니다. Al-Kindi는 통계 및 빈도 분석을 사용하여 암호화된 메시지를 해독하는 방법에 대한 자세한 설명을 제공했습니다.800 년 후, 연구자들은 인구 통계 및 경제 데이터 수집 및 분석에 중점을 둔 1700 년대 독일에서 현대 통계가 나왔습니다.오늘날 통계는 데이터의 수집, 처리, 분석, 해석 및 시각화와 관련된 과학 주제입니다.또한 통계의 핵심 이론은 학계, 산업 및 정부 내 연구에 널리 사용되었습니다. 

보다 구체적으로, 통계량은*기술 통계량*과*통계적 추론*으로 나눌 수 있습니다.전자는 관찰된 데이터 모음의 특징을 요약하고 설명하는 데 중점을 둡니다. 이를 *샘플*이라고 합니다.표본은*모집단*에서 추출되며 실험 관심 있는 유사한 개인, 항목 또는 사건의 총 집합을 나타냅니다.기술 통계량과 달리, *통계적 추론*은 표본 분포가 모집단 분포를 어느 정도 반복할 수 있다는 가정을 기반으로 주어진*표본*에서 모집단의 특성을 추가로 추론합니다. 

“기계 학습과 통계의 근본적인 차이점은 무엇입니까?”기본적으로 통계는 추론 문제에 중점을 둡니다.이러한 유형의 문제에는 인과 추론과 같은 변수 간의 관계를 모델링하고 A/B 테스트와 같은 모델 매개 변수의 통계적 유의성을 테스트하는 것이 포함됩니다.반면 기계 학습은 각 매개 변수의 기능을 명시적으로 프로그래밍하고 이해하지 않고 정확한 예측을 수행하는 데 중점을 둡니다. 

이 섹션에서는 추정기 평가 및 비교, 가설 검정 수행 및 신뢰 구간 구성의 세 가지 유형의 통계 추론 방법을 소개합니다.이러한 방법은 주어진 모집단의 특성, 즉 실제 매개 변수 $\theta$를 추론하는 데 도움이 될 수 있습니다.간단히 설명하려면 주어진 모집단의 실제 모수 $\theta$가 스칼라 값이라고 가정합니다.$\theta$가 벡터 또는 텐서 인 경우로 확장하는 것은 간단하므로 토론에서 생략합니다. 

## 추정기 평가 및 비교

통계에서*추정기*는 실제 모수 $\theta$를 추정하는 데 사용되는 주어진 표본의 함수입니다.샘플 {$x_1, x_2, \ldots, x_n$} 을 관찰한 후 $\theta$의 추정치에 대해 $\hat{\theta}_n = \hat{f}(x_1, \ldots, x_n)$를 작성할 것입니다. 

이전에 :numref:`sec_maximum_likelihood` 절에서 추정기의 간단한 예를 보았습니다.Bernoulli 랜덤 변수의 표본 수가 많은 경우 관측된 표본 수를 세고 총 표본 수로 나누어 랜덤 변수가 1일 확률에 대한 최대우도 추정치를 얻을 수 있습니다.마찬가지로 한 연습에서는 표본 수가 주어진 가우스 평균의 최대우도 추정치를 모든 표본의 평균값으로 지정한다는 것을 보여달라고 요청했습니다.이러한 추정기는 모수의 실제 값을 거의 제공하지 않지만, 많은 수의 표본에 대해 추정치가 근접하는 것이 이상적입니다. 

예를 들어, 평균이 0이고 분산이 1인 가우스 랜덤 변수의 실제 밀도와 해당 가우스의 수집 샘플을 아래에 보여줍니다.모든 점을 볼 수 있고 원래 밀도와의 관계가 더 명확하도록 $y$ 좌표를 구성했습니다.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()

# Sample datapoints and create y coordinate
epsilon = 0.1
random.seed(8675309)
xs = np.random.normal(loc=0, scale=1, size=(300,))

ys = [np.sum(np.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))
             / np.sqrt(2*np.pi*epsilon**2)) / len(xs) for i in range(len(xs))]

# Compute true density
xd = np.arange(np.min(xs), np.max(xs), 0.01)
yd = np.exp(-xd**2/2) / np.sqrt(2 * np.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=np.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(np.mean(xs)):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  #define pi in torch

# Sample datapoints and create y coordinate
epsilon = 0.1
torch.manual_seed(8675309)
xs = torch.randn(size=(300,))

ys = torch.tensor(
    [torch.sum(torch.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))\
               / torch.sqrt(2*torch.pi*epsilon**2)) / len(xs)\
     for i in range(len(xs))])

# Compute true density
xd = torch.arange(torch.min(xs), torch.max(xs), 0.01)
yd = torch.exp(-xd**2/2) / torch.sqrt(2 * torch.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=torch.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(torch.mean(xs).item()):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

tf.pi = tf.acos(tf.zeros(1)) * 2  # define pi in TensorFlow

# Sample datapoints and create y coordinate
epsilon = 0.1
xs = tf.random.normal((300,))

ys = tf.constant(
    [(tf.reduce_sum(tf.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2)) \
               / tf.sqrt(2*tf.pi*epsilon**2)) / tf.cast(
        tf.size(xs), dtype=tf.float32)).numpy() \
     for i in range(tf.size(xs))])

# Compute true density
xd = tf.range(tf.reduce_min(xs), tf.reduce_max(xs), 0.01)
yd = tf.exp(-xd**2/2) / tf.sqrt(2 * tf.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=tf.reduce_mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(tf.reduce_mean(xs).numpy()):.2f}')
d2l.plt.show()
```

파라미터 $\hat{\theta}_n$의 추정기를 계산하는 방법에는 여러 가지가 있습니다.이 섹션에서는 추정기를 평가하고 비교하는 세 가지 일반적인 방법, 즉 평균 제곱 오차, 표준 편차 및 통계적 편향을 소개합니다. 

### 평균 제곱 오차

추정기를 평가하는 데 사용되는 가장 간단한 척도는 추정기의*평균 제곱 오차 (MSE) * (또는 $l_2$ 손실) 는 다음과 같이 정의할 수 있습니다. 

$$\mathrm{MSE} (\hat{\theta}_n, \theta) = E[(\hat{\theta}_n - \theta)^2].$$
:eqlabel:`eq_mse_est`

이를 통해 실제 값에서 평균 제곱 편차를 정량화할 수 있습니다.MSE는 항상 음수가 아닙니다.:numref:`sec_linear_regression`를 읽은 경우 가장 일반적으로 사용되는 회귀 손실 함수로 인식할 수 있습니다.추정기를 평가하기 위한 척도로서 값이 0에 가까울수록 추정기는 실제 모수 $\theta$에 더 가깝습니다. 

### 통계 편향

MSE는 자연스러운 메트릭을 제공하지만, 이를 크게 만들 수 있는 여러 가지 현상을 쉽게 상상할 수 있습니다.근본적으로 중요한 두 가지는 데이터 세트의 임의성으로 인한 추정기의 변동과 추정 절차로 인한 추정기의 체계적인 오류입니다. 

먼저 체계적인 오차를 측정해 보겠습니다.추정기 $\hat{\theta}_n$의 경우*통계적 편향*의 수학적 그림은 다음과 같이 정의할 수 있습니다. 

$$\mathrm{bias}(\hat{\theta}_n) = E(\hat{\theta}_n - \theta) = E(\hat{\theta}_n) - \theta.$$
:eqlabel:`eq_bias`

$\mathrm{bias}(\hat{\theta}_n) = 0$인 경우 추정기 $\hat{\theta}_n$의 기대치는 모수의 실제 값과 같습니다.이 경우 $\hat{\theta}_n$가 편향되지 않은 추정치라고 말합니다.일반적으로 비편향 추정기는 기대치가 실제 모수와 같기 때문에 편향 추정기보다 낫습니다. 

그러나 실제로 편향된 추정기가 자주 사용된다는 사실을 알고 있어야 합니다.편향되지 않은 추정기가 추가 가정 없이 존재하지 않거나 계산하기 어려운 경우가 있습니다.이것은 추정기에서 중요한 결함처럼 보일 수 있지만 실제로 발생하는 대부분의 추정기는 사용 가능한 샘플의 수가 무한대가되는 경향이 있기 때문에 치우침이 0이되는 경향이 있다는 의미에서 적어도 점근 적으로 편향되지 않습니다 ($\lim_{n \rightarrow \infty} \mathrm{bias}(\hat{\theta}_n) = 0$). 

### 분산 및 표준 편차

둘째, 추정기에서 임의성을 측정해 보겠습니다.:numref:`sec_random_variables`에서*표준 편차* (또는*표준 오차*) 는 분산의 제곱근으로 정의됩니다.해당 추정기의 표준 편차 또는 분산을 측정하여 추정기의 변동 정도를 측정 할 수 있습니다. 

$$\sigma_{\hat{\theta}_n} = \sqrt{\mathrm{Var} (\hat{\theta}_n )} = \sqrt{E[(\hat{\theta}_n - E(\hat{\theta}_n))^2]}.$$
:eqlabel:`eq_var_est`

:eqref:`eq_var_est`와 :eqref:`eq_mse_est`를 비교하는 것이 중요합니다.이 방정식에서는 실제 모집단 값 $\theta$과 비교하지 않고 대신 예상 표본 평균인 $E(\hat{\theta}_n)$과 비교합니다.따라서 추정기가 실제 값에서 얼마나 멀리 떨어져 있는지 측정하는 것이 아니라 추정기 자체의 변동을 측정합니다. 

### 편향-분산 트레이드 오프

이 두 가지 주요 성분이 평균 제곱 오차에 기여한다는 것은 직관적으로 분명합니다.다소 충격적인 것은 이것이 실제로 평균 제곱 오차를이 두 기여와 세 번째 기여로 나눈 값의*분해*임을 보여줄 수 있다는 것입니다.즉, 평균 제곱 오차를 편향의 제곱, 분산 및 환원 할 수없는 오차의 합으로 쓸 수 있습니다. 

$$
\begin{aligned}
\mathrm{MSE} (\hat{\theta}_n, \theta) &= E[(\hat{\theta}_n - \theta)^2] \\
 &= E[(\hat{\theta}_n)^2] + E[\theta^2] - 2E[\hat{\theta}_n\theta] \\
 &= \mathrm{Var} [\hat{\theta}_n] + E[\hat{\theta}_n]^2 + \mathrm{Var} [\theta] + E[\theta]^2 - 2E[\hat{\theta}_n]E[\theta] \\
 &= (E[\hat{\theta}_n] - E[\theta])^2 + \mathrm{Var} [\hat{\theta}_n] + \mathrm{Var} [\theta] \\
 &= (E[\hat{\theta}_n - \theta])^2 + \mathrm{Var} [\hat{\theta}_n] + \mathrm{Var} [\theta] \\
 &= (\mathrm{bias} [\hat{\theta}_n])^2 + \mathrm{Var} (\hat{\theta}_n) + \mathrm{Var} [\theta].\\
\end{aligned}
$$

위 공식은*편향 분산 트레이드 오프*라고 합니다.평균 제곱 오차는 세 가지 오류 원인 (: the error from high bias, the error from high variance and the irreducible error. The bias error is commonly seen in a simple model (such as a linear regression model), which cannot extract high dimensional relations between the features and the outputs. If a model suffers from high bias error, we often say it is *underfitting* or lack of *flexibilty* as introduced in (:numref:`sec_model_selection`) 으로 나눌 수 있습니다.높은 분산은 일반적으로 훈련 데이터를 과적합하는 너무 복잡한 모델에서 발생합니다.따라서, *과적합* 모형은 데이터의 작은 변동에 민감합니다.모형이 높은 분산을 겪는 경우 (:numref:`sec_model_selection`) 에 소개된 것처럼*과적합* 및*일반화*가 부족하다고 말하는 경우가 많습니다.감소시킬 수 없는 오차는 $\theta$ 자체의 노이즈로 인한 결과입니다. 

### 코드에서 추정기 평가

추정기의 표준 편차는 단순히 텐서 `a`에 대해 `a.std()`를 호출하여 구현되었으므로 건너 뛰고 통계적 편향과 평균 제곱 오차를 구현합니다.

```{.python .input}
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(np.mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(np.mean(np.square(data - true_theta)))
```

```{.python .input}
#@tab pytorch
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(torch.mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(torch.mean(torch.square(data - true_theta)))
```

```{.python .input}
#@tab tensorflow
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(tf.reduce_mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(tf.reduce_mean(tf.square(data - true_theta)))
```

편향 분산 트레이드 오프의 방정식을 설명하기 위해 $10,000$ 표본을 사용하여 정규 분포 $\mathcal{N}(\theta, \sigma^2)$를 시뮬레이션해 보겠습니다.여기서는 $\theta = 1$과 $\sigma = 4$을 사용합니다.추정기는 주어진 표본의 함수이므로 여기서는 표본의 평균을 이 정규 분포 $\mathcal{N}(\theta, \sigma^2)$에서 참 $\theta$에 대한 추정기로 사용합니다.

```{.python .input}
theta_true = 1
sigma = 4
sample_len = 10000
samples = np.random.normal(theta_true, sigma, sample_len)
theta_est = np.mean(samples)
theta_est
```

```{.python .input}
#@tab pytorch
theta_true = 1
sigma = 4
sample_len = 10000
samples = torch.normal(theta_true, sigma, size=(sample_len, 1))
theta_est = torch.mean(samples)
theta_est
```

```{.python .input}
#@tab tensorflow
theta_true = 1
sigma = 4
sample_len = 10000
samples = tf.random.normal((sample_len, 1), theta_true, sigma)
theta_est = tf.reduce_mean(samples)
theta_est
```

제곱 편향의 합과 추정기의 분산을 계산하여 트레이드 오프 방정식을 검증해 보겠습니다.먼저 추정기의 MSE를 계산합니다.

```{.python .input}
#@tab all
mse(samples, theta_true)
```

다음으로 아래와 같이 $\mathrm{Var} (\hat{\theta}_n) + [\mathrm{bias} (\hat{\theta}_n)]^2$를 계산합니다.보시다시피 두 값은 수치 정밀도와 일치합니다.

```{.python .input}
bias = stat_bias(theta_true, theta_est)
np.square(samples.std()) + np.square(bias)
```

```{.python .input}
#@tab pytorch
bias = stat_bias(theta_true, theta_est)
torch.square(samples.std(unbiased=False)) + torch.square(bias)
```

```{.python .input}
#@tab tensorflow
bias = stat_bias(theta_true, theta_est)
tf.square(tf.math.reduce_std(samples)) + tf.square(bias)
```

## 가설 테스트 수행

통계적 추론에서 가장 일반적으로 접하는 주제는 가설 검정입니다.가설 검정은 20세기 초에 대중화되었지만, 첫 번째 용도는 1700년대에 John Arbuthnot으로 거슬러 올라갈 수 있습니다.John은 런던에서 80년 출생 기록을 추적하여 매년 여성보다 남성이 더 많이 태어난다고 결론지었습니다.그 후 현대의 유의성 테스트는 $p$ 값과 피어슨의 카이 제곱 검정을 발명 한 칼 피어슨, 학생의 t 분포의 아버지인 윌리엄 고셋, 귀무 가설과 유의 검정을 시작한 로널드 피셔의 지능 유산입니다. 

*가설 검정*은 모집단에 대한 기본 진술에 반하여 일부 증거를 평가하는 방법입니다.디폴트 명령문을*null 가설* $H_0$이라고 부르며, 관찰 된 데이터를 사용하여 거부하려고합니다.여기서는 $H_0$을 통계적 유의성 검정의 시작점으로 사용합니다.*대립 가설* $H_A$ (또는 $H_1$) 는 귀무 가설에 위배되는 진술입니다.귀무 가설은 종종 변수 간의 관계를 나타내는 선언적 형식으로 명시됩니다.브리핑을 가능한 한 명시적으로 반영하고 통계 이론으로 테스트할 수 있어야 합니다. 

당신이 화학자라고 상상해 보세요.실험실에서 수천 시간을 보낸 후 수학 이해 능력을 크게 향상시킬 수있는 새로운 약을 개발합니다.마법의 힘을 보여주기 위해서는 그것을 시험해봐야 합니다.당연히 약을 복용하고 수학을 더 잘 배우는 데 도움이 될 수 있는지 확인하기 위해 일부 자원 봉사자가 필요할 수 있습니다.어떻게 시작할 수 있을까요? 

첫째, 신중하게 무작위로 선택된 두 그룹의 자원 봉사자가 필요하므로 일부 메트릭으로 측정 한 수학 이해 능력간에 차이가 없습니다.두 그룹을 일반적으로 테스트 그룹 및 대조군이라고 합니다.*테스트 그룹* (또는*치료 그룹*) 은 약을 경험하게 될 개인 그룹이고, *대조군*은 벤치마크로 따로 설정된 사용자 그룹, 즉 이 약을 복용하지 않는 동일한 환경 설정을 나타냅니다.이러한 방식으로 독립 변수가 치료에 미치는 영향을 제외하고 모든 변수의 영향이 최소화됩니다. 

둘째, 약을 복용 한 후 자원 봉사자가 새로운 수학 공식을 배운 후 동일한 테스트를 수행하도록하는 것과 같은 지표로 두 그룹의 수학 이해도를 측정해야합니다.그런 다음 성능을 수집하고 결과를 비교할 수 있습니다.이 경우 귀무 가설은 두 그룹 사이에 차이가 없으며 대안은 차이가 있다는 것입니다. 

이것은 아직 완전히 형식적이지 않습니다.신중하게 생각해야 할 세부 사항이 많이 있습니다.예를 들어 수학 이해 능력을 테스트하는 데 적합한 메트릭은 무엇입니까?약의 효과를 확신할 수 있도록 검사 지원자는 몇 명입니까?테스트를 얼마나 오래 실행해야 합니까?두 그룹 간에 차이가 있는지 어떻게 결정합니까?평균 성과에만 관심이 있습니까, 아니면 점수의 변동 범위에도 관심이 있습니까?등등. 

이러한 방식으로 가설 검정은 관찰된 결과의 확실성에 대한 실험 설계 및 추론을 위한 프레임워크를 제공합니다.이제 귀무 가설이 참일 가능성이 매우 낮다는 것을 보여줄 수 있다면 확신을 가지고 기각 할 수 있습니다. 

가설 검정을 다루는 방법에 대한 이야기를 완성하기 위해 이제 몇 가지 추가 용어를 도입하고 일부 개념을 공식보다 높게 만들어야 합니다. 

### 통계적 중요성

*통계적 유의성*은 귀무 가설 $H_0$를 기각해서는 안 될 때, 즉 귀무 가설을 잘못 기각할 확률을 측정합니다. 

$$ \text{statistical significance }= 1 - \alpha = 1 - P(\text{reject } H_0 \mid H_0 \text{ is true} ).$$

*유형 I 오류* 또는*가양성 (false positive) *이라고도 합니다.$\alpha$는*중요도 수준*으로 불리며 일반적으로 사용되는 값은 $5\%$, i.e., $1-\ 알파 = 95\%$입니다.유의 수준은 실제 귀무 가설을 기각할 때 기꺼이 감수할 위험 수준으로 설명할 수 있습니다. 

:numref:`fig_statistical_significance`는 2-표본 가설 검정에서 주어진 정규 분포의 관측치 값과 확률을 보여줍니다.관측치 데이터 예제가 $95\%$ 임계값을 벗어나는 경우 귀무 가설 가정 하에서는 관측치가 될 가능성이 매우 낮습니다.따라서 귀무 가설에 문제가 있을 수 있으며 이를 기각할 것입니다. 

![Statistical significance.](../img/statistical-significance.svg)
:label:`fig_statistical_significance`

### 통계적 파워

*통계적 검정력* (또는*민감도*) 은 귀무 가설을 기각해야 할 때 귀무 가설 $H_0$를 기각할 확률을 측정합니다. 

$$ \text{statistical power }= 1 - \beta = 1 - P(\text{ fail to reject } H_0  \mid H_0 \text{ is false} ).$$

*유형 I 오류*는 귀무 가설이 참일 때 귀무 가설을 기각하여 발생하는 오류인 반면, *유형 II 오류*는 귀무 가설이 거짓일 때 귀무 가설을 기각하지 못한 경우에 발생합니다.유형 II 오류는 일반적으로 $\beta$로 표시되므로 해당 통계적 검정력은 $1-\beta$입니다. 

직관적으로 통계적 검정력은 테스트가 원하는 통계적 유의성 수준에서 최소 크기의 실제 불일치를 감지 할 가능성으로 해석 할 수 있습니다. $80\%$는 일반적으로 사용되는 통계적 검정력 임계 값입니다.통계적 검정력이 높을수록 실제 차이를 탐지할 가능성이 높아집니다. 

통계적 검정력의 가장 일반적인 용도 중 하나는 필요한 표본 수를 결정하는 것입니다.귀무 가설이 거짓일 때 귀무 가설을 기각할 확률은 귀무 가설이 거짓인 정도 (*효과 크기*라고 함) 와 보유한 표본 수에 따라 다릅니다.예상대로 효과 크기가 작으면 높은 확률로 탐지할 수 있는 매우 많은 수의 표본이 필요합니다.예를 들어, 이 간단한 부록의 범위를 벗어나 샘플이 평균 0 분산 1 가우스에서 나왔다는 귀무 가설을 기각하고 샘플의 평균이 실제로 1에 가깝다고 생각하지만 표본 크기가 다음과 같은 허용 가능한 오류율로 그렇게 할 수 있습니다.$8$에 불과합니다.그러나 표본 모집단 실제 평균이 $0.01$에 가깝다고 생각하면 차이를 탐지하기 위해 거의 $80000$의 표본 크기가 필요합니다. 

우리는 그 힘을 정수 필터라고 상상할 수 있습니다.이 비유에서 고출력 가설 검정은 가능한 한 물 속의 유해 물질을 줄이는 고품질 정수 시스템과 같습니다.반면에 불일치가 작을수록 품질이 낮은 정수 필터와 같아서 상대적으로 작은 물질이 틈에서 쉽게 빠져 나올 수 있습니다.마찬가지로 통계적 검정력이 충분히 높은 검정력이 아니면 검정에서 더 작은 불일치를 포착하지 못할 수도 있습니다. 

### 테스트 통계

*검정 통계량* $T(x)$는 표본 데이터의 일부 특성을 요약하는 스칼라입니다.이러한 통계량을 정의하는 목적은 서로 다른 분포를 구별하고 가설 검정을 수행할 수 있도록 하는 것입니다.화학자의 예를 다시 생각해 보면, 한 모집단이 다른 모집단보다 더 나은 성과를 보인다는 것을 보여주고 싶다면 평균을 검정 통계량으로 사용하는 것이 합리적일 수 있습니다.검정 통계량을 다르게 선택하면 통계적 검정력이 크게 다른 통계 검정으로 이어질 수 있습니다. 

귀무 가설 하에서 고려할 때 $T(X)$ (귀무 가설에 따른 검정 통계량의 분포) 는 정규 분포와 같은 일반적인 확률 분포를 적어도 대략적으로 따릅니다.이러한 분포를 명시적으로 도출한 다음 데이터셋에서 검정 통계량을 측정할 수 있다면 통계량이 예상 범위를 훨씬 벗어나면 귀무 가설을 안전하게 기각할 수 있습니다.이를 정량적으로 만들면 $p$ 값의 개념으로 이어집니다. 

### $p$개의 값

$p$ 값 (또는*확률 값*) 은 귀무 가설이*참*이라고 가정할 때 $T(X)$가 관측된 검정 통계량 $T(x)$만큼 극단적일 확률입니다. 즉, 

$$ p\text{-value} = P_{H_0}(T(X) \geq T(x)).$$

$p$ 값이 미리 정의되고 고정된 통계적 유의성 수준 $\alpha$보다 작거나 같으면 귀무 가설을 기각할 수 있습니다.그렇지 않으면 귀무 가설을 기각할 증거가 부족하다는 결론을 내릴 것입니다.지정된 모집단 분포에 대해*기각 영역*은 통계적 유의성 수준 $\alpha$보다 작은 $p$ 값을 갖는 모든 점에 포함된 구간이 됩니다. 

### 단면 테스트 및 양면 테스트

일반적으로 유의성 검정에는 단측 검정과 양측 검정의 두 종류가 있습니다.*단측 검정* (또는*단측 검정*) 은 귀무 가설과 대립 가설의 방향이 한 방향인 경우에만 적용할 수 있습니다.예를 들어, 귀무 가설은 실제 모수 $\theta$가 값 $c$보다 작거나 같다는 것을 나타낼 수 있습니다.대립 가설은 $\theta$가 $c$보다 크다는 것입니다.즉, 거부 영역은 표본 분포의 한 쪽에만 있습니다.단측 검정과 달리*양측 검정* (또는*양측 검정*) 은 기각 영역이 표본 추출 분포의 양쪽에 있는 경우에 적용할 수 있습니다.이 경우의 예는 참 모수 $\theta$가 값 $c$와 같다는 귀무 가설이 있을 수 있습니다.대립 가설은 $\theta$가 $c$와 같지 않다는 것입니다. 

### 가설 검정의 일반적인 단계

위의 개념에 익숙해지면 가설 검정의 일반적인 단계를 살펴 보겠습니다. 

1. 질문을 명시하고 귀무 가설 $H_0$를 설정합니다.
2. 통계적 유의성 수준 $\alpha$와 통계적 검정력 ($1 - \beta$) 을 설정합니다.
3. 실험을 통해 샘플을 얻습니다.필요한 표본 수는 통계적 검정력과 예상되는 효과 크기에 따라 달라집니다.
4. 검정 통계량과 $p$ 값을 계산합니다.
5. $p$ 값과 통계적 유의성 수준 $\alpha$를 기반으로 귀무 가설을 유지하거나 기각하기로 결정합니다.

가설 검정을 수행하기 위해 먼저 귀무 가설과 기꺼이 감수할 위험 수준을 정의합니다.그런 다음 검정 통계량의 극단값을 귀무 가설에 반하는 증거로 사용하여 표본의 검정 통계량을 계산합니다.검정 통계량이 기각 영역 내에 있으면 대안에 유리하게 귀무 가설을 기각할 수 있습니다. 

가설 검정은 임상 추적 및 A/B 테스트와 같은 다양한 시나리오에 적용 할 수 있습니다. 

## 신뢰 구간 구성

모수 $\theta$의 값을 추정할 때 $\hat \theta$과 같은 점 추정기는 불확실성에 대한 개념을 포함하지 않기 때문에 유용성이 제한적입니다.오히려 실제 모수 $\theta$를 높은 확률로 포함하는 구간을 생성할 수 있다면 훨씬 더 좋을 것입니다.한 세기 전에 그러한 아이디어에 관심이 있었다면 1937 년에 신뢰 구간의 개념을 처음 도입 한 Jerzy Neyman :cite:`Neyman.1937`의 “고전 확률 이론에 근거한 통계 추정 이론의 개요”를 읽게되어 기뻤을 것입니다. 

유용하려면 주어진 확실성에 대해 신뢰 구간이 가능한 한 작아야 합니다.그것을 도출하는 방법을 보자. 

### 정의

수학적으로 실제 모수 $\theta$에 대한*신뢰 구간*은 다음과 같은 표본 데이터에서 계산된 구간 $C_n$입니다. 

$$P_{\theta} (C_n \ni \theta) \geq 1 - \alpha, \forall \theta.$$
:eqlabel:`eq_confidence`

여기서 $\alpha \in (0, 1)$ 및 $1 - \alpha$를 구간의*신뢰 수준* 또는*적용 범위*라고 합니다.이는 위에서 논의한 유의 수준과 $\alpha$와 동일합니다. 

:eqref:`eq_confidence`는 고정된 $\theta$가 아니라 변수 $C_n$에 관한 것입니다.이를 강조하기 위해 $P_{\theta} (\theta \in C_n)$이 아닌 $P_{\theta} (C_n \ni \theta)$을 작성합니다. 

### 통역

생성된 간격의  달러$95\%$ confidence interval as an interval where you can be $95\%$ sure the true parameter lies, however this is sadly not true.  The true parameter is fixed, and it is the interval that is random.  Thus a better interpretation would be to say that if you generated a large number of confidence intervals by this procedure, $95\%$에 실제 매개 변수가 포함된다고 해석하는 것은 매우 유혹적입니다. 

이것은 현학적 인 것처럼 보일 수 있지만 결과 해석에 실질적인 영향을 미칠 수 있습니다.특히, 거의 확실한* 실제 값을 포함하지 않는 구간을 구성하여 :eqref:`eq_confidence`를 만족시킬 수 있습니다.유혹적이지만 잘못된 세 가지 진술을 제공하여 이 섹션을 마무리합니다.이러한 점에 대한 심층적 인 논의는 :cite:`Morey.Hoekstra.Rouder.ea.2016`에서 찾을 수 있습니다. 

* **오류 1**.신뢰 구간이 좁으면 모수를 정확하게 추정할 수 있습니다.
* **오류 2**.신뢰 구간 내의 값은 구간을 벗어난 값보다 실제 값일 가능성이 높습니다.
* **오류 3**.특정 사람이 $95\%$ confidence interval contains the true value is $95\%$를 관찰했을 확률입니다.

신뢰 구간은 미묘한 객체라고 말할 수 있습니다.그러나 해석을 명확하게 유지하면 강력한 도구가 될 수 있습니다. 

### 가우스 예제

가장 고전적인 예인 평균과 분산이 알려지지 않은 가우스 평균에 대한 신뢰 구간에 대해 설명하겠습니다.가우스 $\mathcal{N}(\mu, \sigma^2)$에서 $n$개의 샘플 $\{x_i\}_{i=1}^n$를 수집한다고 가정합니다.다음을 수행하여 평균과 표준 편차에 대한 추정기를 계산할 수 있습니다. 

$$\hat\mu_n = \frac{1}{n}\sum_{i=1}^n x_i \;\text{and}\; \hat\sigma^2_n = \frac{1}{n-1}\sum_{i=1}^n (x_i - \hat\mu)^2.$$

이제 랜덤 변수를 고려한다면 

$$
T = \frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}},
$$

* $n-1$*자유도*에서*학생의 t 분포라고 불리는 잘 알려진 분포를 따라 랜덤 변수를 얻습니다. 

이 분포는 매우 잘 연구되어 있으며, 예를 들어 $n\rightarrow \infty$으로 대략 표준 가우스이므로 표에서 가우스 c.d.f. 의 값을 조회하면 $T$의 값이 $[-1.96, 1.96]$ 최소 $95\%$ of the time.  For finite values of $n$에 있다는 결론을 내릴 수 있습니다.다소 크지만 테이블에서 잘 알려져 있고 미리 계산되어 있습니다. 

따라서 우리는 $n$에 대해 다음과 같은 결론을 내릴 수 있습니다. 

$$
P\left(\frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}} \in [-1.96, 1.96]\right) \ge 0.95.
$$

양면에 $\hat\sigma_n/\sqrt{n}$를 곱한 다음 $\hat\mu_n$를 더하여 이것을 다시 정렬하면 

$$
P\left(\mu \in \left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right]\right) \ge 0.95.
$$

따라서 우리는 달러 $95\%$ 신뢰 구간 ($\left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right].$
:eqlabel:`eq_gauss_confidence`달러) 을 찾았다는 것을 알고 있습니다. 

:eqref:`eq_gauss_confidence`는 통계에서 가장 많이 사용되는 공식 중 하나라고 말할 수 있습니다.통계를 구현하여 통계에 대한 논의를 마무리하겠습니다.단순화를 위해 우리는 점근 체제에 있다고 가정합니다.$N$의 작은 값에는 프로그래밍 방식으로 또는 $t$-테이블에서 얻은 올바른 값 `t_star`가 포함되어야 합니다.

```{.python .input}
# Number of samples
N = 1000

# Sample dataset
samples = np.random.normal(loc=0, scale=1, size=(N,))

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = np.mean(samples)
sigma_hat = samples.std(ddof=1)
(mu_hat - t_star*sigma_hat/np.sqrt(N), mu_hat + t_star*sigma_hat/np.sqrt(N))
```

```{.python .input}
#@tab pytorch
# PyTorch uses Bessel's correction by default, which means the use of ddof=1
# instead of default ddof=0 in numpy. We can use unbiased=False to imitate
# ddof=0.

# Number of samples
N = 1000

# Sample dataset
samples = torch.normal(0, 1, size=(N,))

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = torch.mean(samples)
sigma_hat = samples.std(unbiased=True)
(mu_hat - t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)),\
 mu_hat + t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)))
```

```{.python .input}
#@tab tensorflow
# Number of samples
N = 1000

# Sample dataset
samples = tf.random.normal((N,), 0, 1)

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = tf.reduce_mean(samples)
sigma_hat = tf.math.reduce_std(samples)
(mu_hat - t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)), \
 mu_hat + t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)))
```

## 요약

* 통계는 추론 문제에 초점을 맞추는 반면, 딥 러닝은 명시적으로 프로그래밍하고 이해하지 않고 정확한 예측을 수행하는 데 중점을 둡니다.
* 세 가지 일반적인 통계 추론 방법이 있습니다. 추정기 평가 및 비교, 가설 검정 수행, 신뢰 구간 구성입니다.
* 통계적 편향, 표준 편차 및 평균 제곱 오차라는 세 가지 가장 일반적인 추정치가 있습니다.
* 신뢰 구간은 표본이 주어지면 구성할 수 있는 실제 모집단 모수의 추정 범위입니다.
* 가설 검정은 모집단에 대한 기본 진술과 비교하여 일부 증거를 평가하는 방법입니다.

## 연습문제

1. $X_1, X_2, \ldots, X_n \overset{\text{iid}}{\sim} \mathrm{Unif}(0, \theta)$를 지정합니다. 여기서 “iid”는*독립적이고 동일하게 분포된*을 나타냅니다.$\theta$의 다음 추정치를 고려해 보십시오.
$\hat{\theta} = \max \{X_1, X_2, \ldots, X_n \};$달러 $\tilde{\theta} = 2 \bar{X_n} = \frac{2}{n} \sum_{i=1}^n X_i.$$
    * $\hat{\theta}.$의 통계적 치우침, 표준 편차 및 평균 제곱 오차 구하기
    * $\tilde{\theta}.$의 통계적 치우침, 표준 편차 및 평균 제곱 오차 구하기
    * 어떤 추정기가 더 나은가요?
1. 소개에 있는 화학자 예제에서 양측 가설 검정을 수행하는 5단계를 도출할 수 있습니까?통계적 유의성 수준 $\alpha = 0.05$와 통계적 검정력 $1 - \beta = 0.8$를 감안할 때.
1. 독립적으로 생성된 $100$에 대해 $N=2$ 및 $\alpha = 0.5$을 사용하여 신뢰 구간 코드를 실행하고 결과 구간 (이 경우 `t_star = 1.0`) 을 플로팅합니다.실제 평균 $0$를 포함하지 않는 매우 짧은 구간을 여러 개 볼 수 있습니다.이것이 신뢰 구간의 해석과 모순됩니까?고정밀 추정치를 나타내기 위해 짧은 구간을 사용하는 것이 편하다고 느끼십니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/419)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1102)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1103)
:end_tab:
