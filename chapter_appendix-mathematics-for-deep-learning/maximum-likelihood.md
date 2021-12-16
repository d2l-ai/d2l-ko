# 최대화 가능성
:label:`sec_maximum_likelihood`

기계 학습에서 가장 일반적으로 접하는 사고 방식 중 하나는 최대 가능성의 관점입니다.이것은 알 수 없는 모수를 가진 확률적 모델로 작업할 때 데이터의 확률을 가장 높게 만드는 모수가 가장 가능성이 높은 모수라는 개념입니다. 

## 최대화 가능성 원칙

이것은 생각하는 데 도움이 될 수있는 베이지안 해석이 있습니다.매개 변수가 $\boldsymbol{\theta}$인 모델과 데이터 예제 $X$의 모음이 있다고 가정합니다.구체적으로 $\boldsymbol{\theta}$는 뒤집힐 때 동전이 앞쪽으로 올 확률을 나타내는 단일 값이고 $X$는 일련의 독립적 인 동전 뒤집기라고 상상할 수 있습니다.나중에 이 예제를 자세히 살펴보겠습니다. 

모델의 매개 변수에 대해 가장 가능성이 높은 값을 찾으려면 다음을 찾고 싶다는 의미입니다. 

$$\mathop{\mathrm{argmax}} P(\boldsymbol{\theta}\mid X).$$
:eqlabel:`eq_max_like`

베이즈의 규칙에 따르면 이것은 다음과 같습니다. 

$$
\mathop{\mathrm{argmax}} \frac{P(X \mid \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(X)}.
$$

데이터를 생성할 매개변수에 구애받지 않는 확률인 $P(X)$ 표현식은 $\boldsymbol{\theta}$에 전혀 의존하지 않으므로 $\boldsymbol{\theta}$의 최상의 선택을 변경하지 않고 삭제할 수 있습니다.마찬가지로 이제 어떤 매개 변수 집합이 다른 매개 변수 집합보다 나은지에 대한 사전 가정이 없다고 가정 할 수 있으므로 $P(\boldsymbol{\theta})$도 theta에 의존하지 않는다고 선언 할 수 있습니다!예를 들어, 이것은 동전 뒤집기 예에서 공정하거나 그렇지 않은 사전 믿음없이 $[0,1]$의 모든 값이 될 수있는 동전 뒤집기 예에서 의미가 있습니다 (종종*정보가없는 사전*이라고 함).따라서 베이즈 규칙을 적용하면 $\boldsymbol{\theta}$의 최선의 선택이 $\boldsymbol{\theta}$에 대한 최대 가능성 추정치임을 알 수 있습니다. 

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}} _ {\boldsymbol{\theta}} P(X \mid \boldsymbol{\theta}).
$$

일반적인 용어로 모수 ($P(X \mid \boldsymbol{\theta})$) 가 주어진 데이터의 확률을*가능성*이라고합니다. 

### 구체적인 예

구체적인 예를 들어 이것이 어떻게 작동하는지 살펴 보겠습니다.동전 뒤집기가 앞일 확률을 나타내는 단일 모수 $\theta$가 있다고 가정합니다.그런 다음 꼬리를 얻을 확률은 $1-\theta$입니다. 따라서 관측된 데이터 $X$가 $n_H$개의 머리와 $n_T$의 꼬리를 가진 시퀀스라면 독립 확률이 곱해진다는 사실을 사용하여  

$$
P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}.
$$

우리가 $13$개의 동전을 뒤집고 $n_H = 9$와 $n_T = 4$을 가진 시퀀스 “HHHTHTTHHHHHT”를 얻는다면, 우리는 이것이 

$$
P(X \mid \theta) = \theta^9(1-\theta)^4.
$$

이 예제의 한 가지 좋은 점은 우리가 답을 알고 있다는 것입니다.실제로 우리가 구두로 “13 개의 동전을 뒤집었고 9 개가 머리를 올렸습니다. 동전이 우리에게 올 확률에 대한 가장 좋은 추측은 무엇입니까?“모두가 $9/13$를 정확하게 추측 할 것입니다.이 최대 가능성 방법이 우리에게 제공하는 것은 훨씬 더 복잡한 상황으로 일반화되는 방식으로 첫 번째 교장으로부터 그 숫자를 얻는 방법입니다. 

이 예에서 $P(X \mid \theta)$의 플롯은 다음과 같습니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

theta = np.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

theta = torch.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

theta = tf.range(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

이 최대 값은 예상되는 $9/13 \approx 0.7\ldots$ 근방에 있습니다.정확히 거기에 있는지 확인하기 위해 미적분학을 사용할 수 있습니다.최대에서 함수의 기울기는 평평합니다.따라서 도함수가 0인 $\theta$의 값을 찾고 가장 높은 확률을 제공하는 값을 찾아 최대우도 추정치 :eqref:`eq_max_like`를 찾을 수 있습니다.우리는 다음을 계산합니다. 

$$
\begin{aligned}
0 & = \frac{d}{d\theta} P(X \mid \theta) \\
& = \frac{d}{d\theta} \theta^9(1-\theta)^4 \\
& = 9\theta^8(1-\theta)^4 - 4\theta^9(1-\theta)^3 \\
& = \theta^8(1-\theta)^3(9-13\theta).
\end{aligned}
$$

여기에는 $0$, $1$ 및 $9/13$의 세 가지 솔루션이 있습니다.처음 두 개는 시퀀스에 확률 $0$를 할당하기 때문에 최대 값이 아니라 분명히 최소값입니다.최종 값은 시퀀스에 확률을 0으로 할당하지 않습니다*. 따라서 최대우도 추정치 $\hat \theta = 9/13$이어야 합니다. 

## 수치 최적화 및 음의 로그 우도

이전 예제는 훌륭하지만 수십억 개의 매개 변수와 데이터 예제가 있다면 어떨까요? 

먼저, 모든 데이터 예제가 독립적이라고 가정하면 많은 확률의 곱이므로 더 이상 가능성 자체를 실제로 고려할 수 없습니다.실제로 각 확률은 $[0,1]$이며, 일반적으로 약 $1/2$의 값이며 $(1/2)^{1000000000}$의 곱은 기계 정밀도보다 훨씬 낮습니다.우리는 직접 작업할 수 없습니다.   

그러나 로그가 곱을 합계로 바꾼다는 것을 상기하십시오.  

$$
\log((1/2)^{1000000000}) = 1000000000\cdot\log(1/2) \approx -301029995.6\ldots
$$

이 숫자는 단정밀도 $32$비트 플로트에도 완벽하게 맞습니다.따라서*로그 우도*를 고려해야 합니다. 

$$
\log(P(X \mid \boldsymbol{\theta})).
$$

함수 $x \mapsto \log(x)$가 증가하고 있으므로 우도를 최대화하는 것은 로그 우도를 최대화하는 것과 같습니다.실제로 :numref:`sec_naive_bayes`에서는 나이브 베이즈 분류기의 구체적인 예를 사용하여 작업 할 때 이러한 추론이 적용되는 것을 볼 수 있습니다. 

우리는 종종 손실을 최소화하고자 하는 손실 함수로 작업합니다.*음의 로그 가능도*인 $-\log(P(X \mid \boldsymbol{\theta}))$를 사용하여 최대 가능성을 손실 최소화로 바꿀 수 있습니다. 

이를 설명하기 위해 이전의 동전 뒤집기 문제를 고려하고 폐쇄 형 솔루션을 모르는 척하십시오.우리는 그것을 계산할 수 있습니다. 

$$
-\log(P(X \mid \boldsymbol{\theta})) = -\log(\theta^{n_H}(1-\theta)^{n_T}) = -(n_H\log(\theta) + n_T\log(1-\theta)).
$$

이것은 코드로 작성할 수 있으며 수십억 개의 코인 플립에도 자유롭게 최적화 할 수 있습니다.

```{.python .input}
# Set up our data
n_H = 8675309
n_T = 25624

# Initialize our paramteres
theta = np.array(0.5)
theta.attach_grad()

# Perform gradient descent
lr = 0.00000000001
for iter in range(10):
    with autograd.record():
        loss = -(n_H * np.log(theta) + n_T * np.log(1 - theta))
    loss.backward()
    theta -= lr * theta.grad

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab pytorch
# Set up our data
n_H = 8675309
n_T = 25624

# Initialize our paramteres
theta = torch.tensor(0.5, requires_grad=True)

# Perform gradient descent
lr = 0.00000000001
for iter in range(10):
    loss = -(n_H * torch.log(theta) + n_T * torch.log(1 - theta))
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad
    theta.grad.zero_()

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab tensorflow
# Set up our data
n_H = 8675309
n_T = 25624

# Initialize our paramteres
theta = tf.Variable(tf.constant(0.5))

# Perform gradient descent
lr = 0.00000000001
for iter in range(10):
    with tf.GradientTape() as t:
        loss = -(n_H * tf.math.log(theta) + n_T * tf.math.log(1 - theta))
    theta.assign_sub(lr * t.gradient(loss, theta))

# Check output
theta, n_H / (n_H + n_T)
```

사람들이 음수 로그 가능성을 사용하는 것을 좋아하는 유일한 이유는 수치적 편의성이 아닙니다.선호되는 몇 가지 다른 이유가 있습니다. 

로그 우도를 고려하는 두 번째 이유는 미적분 규칙을 간단하게 적용하기 때문입니다.위에서 설명한 것처럼 독립성 가정으로 인해 기계 학습에서 발생하는 대부분의 확률은 개별 확률의 곱입니다. 

$$
P(X\mid\boldsymbol{\theta}) = p(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta})\cdots p(x_n\mid\boldsymbol{\theta}).
$$

즉, 곱 규칙을 직접 적용하여 미분을 계산하면 

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{\theta}} P(X\mid\boldsymbol{\theta}) & = \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right)\cdot P(x_2\mid\boldsymbol{\theta})\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_2\mid\boldsymbol{\theta})\right)\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot P(x_2\mid\boldsymbol{\theta}) \cdots \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
\end{aligned}
$$

이를 위해서는 $n(n-1)$의 곱셈과 $(n-1)$의 덧셈이 필요하므로 입력의 2차 시간에 비례합니다!그룹화 용어의 충분한 영리함은 이것을 선형 시간으로 줄이지 만 약간의 생각이 필요합니다.음의 로그 우도를 위해 대신 

$$
-\log\left(P(X\mid\boldsymbol{\theta})\right) = -\log(P(x_1\mid\boldsymbol{\theta})) - \log(P(x_2\mid\boldsymbol{\theta})) \cdots - \log(P(x_n\mid\boldsymbol{\theta})),
$$

그런 다음 

$$
- \frac{\partial}{\partial \boldsymbol{\theta}} \log\left(P(X\mid\boldsymbol{\theta})\right) = \frac{1}{P(x_1\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right) + \cdots + \frac{1}{P(x_n\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
$$

여기에는 $n$의 분할과 $n-1$의 합만 필요하므로 입력에서 선형 시간입니다. 

음의 로그 우도를 고려하는 세 번째이자 마지막 이유는 정보 이론과의 관계이며, 이에 대해서는 :numref:`sec_information_theory`에서 자세히 설명하겠습니다.이것은 랜덤 변수에서 정보의 정도 또는 임의성을 측정하는 방법을 제공하는 엄격한 수학적 이론입니다.그 분야의 주요 연구 대상은 엔트로피입니다.  

$$
H(p) = -\sum_{i} p_i \log_2(p_i),
$$

소스의 임의성을 측정합니다.이것은 평균 $-\log$ 확률에 지나지 않으므로 음의 로그 우도를 취하여 데이터 예제 수로 나누면 교차 엔트로피라고 알려진 엔트로피의 친척을 얻습니다.이러한 이론적 해석만으로도 모델 성능을 측정하는 방법으로 데이터 세트에 대한 평균 음의 로그 우도를 보고하도록 동기를 부여하기에 충분히 설득력이 있습니다. 

## 계량형 변수의 최대우도

지금까지 수행한 모든 작업은 이산 확률 변수로 작업하고 있다고 가정하지만 연속형 랜덤 변수로 작업하려면 어떻게 해야 할까요? 

간단히 요약하면 확률의 모든 인스턴스를 확률 밀도로 대체하는 것을 제외하고는 전혀 변하지 않는다는 것입니다.밀도를 소문자 $p$로 작성한다는 것을 상기하면, 예를 들어 이제 

$$
-\log\left(p(X\mid\boldsymbol{\theta})\right) = -\log(p(x_1\mid\boldsymbol{\theta})) - \log(p(x_2\mid\boldsymbol{\theta})) \cdots - \log(p(x_n\mid\boldsymbol{\theta})) = -\sum_i \log(p(x_i \mid \theta)).
$$

질문은 “왜 괜찮은가?”결국 밀도를 도입 한 이유는 특정 결과를 얻을 확률이 0이기 때문에 매개 변수 집합에 대해 데이터를 생성 할 확률이 0이 아니기 때문입니다. 

실제로 이것이 사실이며 밀도로 이동할 수있는 이유를 이해하는 것은 엡실론에 어떤 일이 발생하는지 추적하는 연습입니다. 

먼저 목표를 다시 정의해 보겠습니다.연속 확률 변수의 경우 더 이상 정확한 값을 얻을 확률을 계산하지 않고 대신 특정 범위 $\epsilon$ 내에서 일치시킬 확률을 계산한다고 가정합니다.단순화하기 위해 데이터가 동일하게 분포된 랜덤 변수 $X_1, \ldots, X_N$의 반복 관측치 $x_1, \ldots, x_N$라고 가정합니다.이전에 살펴본 것처럼 다음과 같이 작성할 수 있습니다. 

$$
\begin{aligned}
&P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta}) \\
\approx &\epsilon^Np(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta}) \cdots p(x_n\mid\boldsymbol{\theta}).
\end{aligned}
$$

따라서 음의 로그를 취하면 

$$
\begin{aligned}
&-\log(P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta})) \\
\approx & -N\log(\epsilon) - \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
\end{aligned}
$$

이 식을 살펴보면 $\epsilon$가 발생하는 유일한 위치는 가산 상수 $-N\log(\epsilon)$입니다.이것은 매개 변수 $\boldsymbol{\theta}$에 전혀 의존하지 않으므로 $\boldsymbol{\theta}$의 최적 선택은 $\epsilon$의 선택에 의존하지 않습니다!네 자리 또는 4 백 자리를 요구하는 경우 $\boldsymbol{\theta}$의 최상의 선택은 동일하게 유지되므로 엡실론을 자유롭게 떨어 뜨려 최적화하려는 것이 

$$
- \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
$$

따라서 최대 가능도 관점은 확률을 확률 밀도로 대체하여 이산 확률 변수처럼 쉽게 연속 확률 변수로 작동 할 수 있음을 알 수 있습니다. 

## 요약* 최대우도 원칙은 주어진 데이터셋에 가장 적합한 모형이 가장 높은 확률로 데이터를 생성하는 모형이라는 것을 알려줍니다.* 종종 사람들은 수치적 안정성, 곱을 합계로 변환하는 등 다양한 이유로 음의 로그 우도로 작업합니다.결과 기울기 계산의 단순화) 및 정보 이론과의 이론적 유대 관계* 이산 설정에서 동기를 부여하는 것이 가장 간단하지만 데이터 포인트에 할당 된 확률 밀도를 최대화하여 연속 설정으로 자유롭게 일반화 할 수 있습니다. 

## 연습 문제 1.랜덤 변수가 일부 값 $\alpha$에 대해 밀도가 $\frac{1}{\alpha}e^{-\alpha x}$라는 것을 알고 있다고 가정합니다.숫자 $3$인 랜덤 변수에서 단일 관측치를 얻습니다.$\alpha$에 대한 최대우도 추정치는 얼마입니까?2.평균은 알 수 없지만 분산이 $1$인 가우스에서 추출한 표본 $\{x_i\}_{i=1}^N$의 데이터셋이 있다고 가정합니다.평균에 대한 최대우도 추정치는 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/416)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1096)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1097)
:end_tab:
