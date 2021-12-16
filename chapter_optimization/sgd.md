# 확률적 경사하강법
:label:`sec_sgd`

그러나 이전 장에서는 왜 작동하는지 설명하지 않고 훈련 절차에서 확률적 경사 하강을 계속 사용했습니다.이를 밝히기 위해 :numref:`sec_gd`에서 기울기 하강의 기본 원리를 설명했습니다.이 섹션에서는 계속해서 논의합니다.
*확률적 그래디언트 하강*을 더 자세히 설명합니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

## 스토캐스틱 그라데이션 업데이트

딥러닝에서 목적 함수는 일반적으로 훈련 데이터셋의 각 예제에 대한 손실 함수의 평균입니다.$n$ 예제로 구성된 훈련 데이터셋이 주어지면 $f_i(\mathbf{x})$이 인덱스 $i$의 훈련 예제와 관련된 손실 함수라고 가정합니다. 여기서 $\mathbf{x}$는 파라미터 벡터입니다.그런 다음 목적 함수에 도달합니다. 

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$

$\mathbf{x}$에서 목적 함수의 기울기는 다음과 같이 계산됩니다. 

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

기울기 하강법이 사용되는 경우 각 독립 변수 반복에 대한 계산 비용은 $\mathcal{O}(n)$이며, 이는 $n$와 함께 선형적으로 증가합니다.따라서 훈련 데이터셋이 더 크면 각 반복에 대한 경사하강법 비용이 더 많이 듭니다. 

확률적 경사하강법 (SGD) 은 각 반복에서 계산 비용을 절감합니다.확률적 경사하강법의 각 반복에서 무작위로 데이터 예제에 대한 인덱스 $i\in\{1,\ldots, n\}$를 균일하게 샘플링하고 기울기 $\nabla f_i(\mathbf{x})$을 계산하여 $\mathbf{x}$를 업데이트합니다. 

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$

여기서 $\eta$은 학습률입니다.각 반복에 대한 계산 비용이 기울기 하강의 $\mathcal{O}(n)$에서 상수 $\mathcal{O}(1)$로 떨어지는 것을 볼 수 있습니다.또한 확률 적 기울기 $\nabla f_i(\mathbf{x})$은 다음과 같은 이유로 전체 기울기 $\nabla f(\mathbf{x})$의 편향되지 않은 추정치라는 점을 강조하고 싶습니다. 

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

즉, 평균적으로 확률적 기울기는 기울기에 대한 좋은 추정치입니다. 

이제 확률적 경사 하강을 시뮬레이션하기 위해 평균이 0이고 분산이 1인 랜덤 노이즈를 기울기에 추가하여 기울기 하강과 비교합니다.

```{.python .input}
#@tab all
def f(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # Gradient of the objective function
    return 2 * x1, 4 * x2
```

```{.python .input}
#@tab mxnet, pytorch
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab tensorflow
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal([1], 0.0, 1)
    g2 += d2l.normal([1], 0.0, 1)
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab all
def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # Constant learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

보시다시피, 확률 적 기울기 하강에서 변수의 궤적은 :numref:`sec_gd`의 기울기 하강에서 관찰 한 것보다 훨씬 더 노이즈가 많습니다.이는 그래디언트의 확률적 특성 때문입니다.즉, 최소값에 도달하더라도 $\eta \nabla f_i(\mathbf{x})$를 통해 순간 기울기에 의해 주입되는 불확실성에 여전히 영향을받습니다.50 단계 후에도 품질은 여전히 좋지 않습니다.더 나쁜 것은 추가 단계를 수행해도 개선되지 않는다는 것입니다 (이를 확인하기 위해 더 많은 단계를 실험해 보는 것이 좋습니다).이것은 우리에게 유일한 대안을 남깁니다. 학습률 $\eta$을 변경하십시오.그러나 이것을 너무 작게 선택하면 처음에는 의미있는 진전을 이루지 못할 것입니다.반면에 너무 크게 선택하면 위에서 볼 수 있듯이 좋은 해결책을 얻지 못할 것입니다.이러한 상충되는 목표를 해결하는 유일한 방법은 최적화가 진행됨에 따라 학습률을*동적으로* 낮추는 것입니다. 

이것이 `sgd` 스텝 함수에 학습률 함수 `lr`를 추가하는 이유이기도 합니다.위의 예에서 학습률 스케줄링 기능은 관련 `lr` 함수를 상수로 설정했기 때문에 휴면 상태입니다. 

## 동적 학습률

$\eta$를 시간 종속 학습률 $\eta(t)$로 대체하면 최적화 알고리즘의 수렴 제어가 복잡해집니다.특히 $\eta$가 얼마나 빨리 부패해야 하는지 알아내야 합니다.속도가 너무 빠르면 조기에 최적화를 중단합니다.너무 느리게 줄이면 최적화에 너무 많은 시간을 낭비하게 됩니다.다음은 시간이 지남에 따라 $\eta$를 조정하는 데 사용되는 몇 가지 기본 전략입니다 (나중에 고급 전략에 대해 설명하겠습니다). 

$$
\begin{aligned}
    \eta(t) & = \eta_i \text{ if } t_i \leq t \leq t_{i+1}  && \text{piecewise constant} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \text{exponential decay} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \text{polynomial decay}
\end{aligned}
$$

첫 번째*조각별 상수* 시나리오에서는 예를 들어 최적화 진행이 멈출 때마다 학습률을 줄입니다.이는 심층 네트워크를 훈련시키기 위한 일반적인 전략입니다.또는*지수 붕괴*를 통해 훨씬 더 공격적으로 줄일 수 있습니다.안타깝게도 이로 인해 알고리즘이 수렴되기 전에 조기에 중지되는 경우가 많습니다.인기 있는 선택은 $\alpha = 0.5$와 함께*다항식 붕괴*입니다.볼록 최적화의 경우 이 비율이 잘 동작한다는 것을 보여주는 여러 가지 증거가 있습니다. 

실제로 지수 붕괴가 어떻게 생겼는지 봅시다.

```{.python .input}
#@tab all
def exponential_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
```

예상대로 모수의 분산이 크게 감소합니다.그러나 이는 최적 솔루션 $\mathbf{x} = (0, 0)$로 수렴하지 못하는 비용이 발생합니다.1000번의 반복 단계가 지난 후에도 우리는 여전히 최적의 솔루션에서 멀리 떨어져 있습니다.실제로 알고리즘은 전혀 수렴하지 못합니다.반면에 단계 수의 역 제곱근으로 학습률이 감소하는 다항식 붕괴를 사용하면 50 단계만으로 수렴이 향상됩니다.

```{.python .input}
#@tab all
def polynomial_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

학습률을 설정하는 방법에는 더 많은 선택 사항이 있습니다.예를 들어, 작은 요율로 시작한 다음 빠르게 상승한 다음 더 느리더라도 다시 낮출 수 있습니다.더 작은 학습률과 더 큰 학습률을 번갈아 사용할 수도 있습니다.이러한 일정에는 매우 다양한 일정이 있습니다.지금은 포괄적 인 이론적 분석이 가능한 학습 속도 일정, 즉 볼록한 설정의 학습률에 중점을 두겠습니다.일반적인 비볼록 문제의 경우 일반적으로 비선형 비볼록 문제를 최소화하는 것이 NP가 어렵 기 때문에 의미 있는 수렴 보장을 얻기가 매우 어렵습니다.설문 조사는 예를 들어 티브시라니 2015의 우수한 [강의 노트](https://www.stat.cmu.edu/~ryantibs/convexopt-F15/lectures/26-nonconvex.pdf) 를 참조하십시오. 

## 볼록 대물렌즈의 수렴 분석

볼록 목적 함수에 대한 확률적 경사하강법에 대한 다음 수렴 분석은 선택 사항이며 주로 문제에 대한 더 많은 직관을 전달하는 데 사용됩니다.우리는 :cite:`Nesterov.Vial.2000`의 가장 단순한 증거 중 하나로 제한합니다.예를 들어 목적 함수가 특히 잘 동작할 때마다 훨씬 더 발전된 증명 기법이 존재합니다. 

목적 함수 $f(\boldsymbol{\xi}, \mathbf{x})$가 모든 $\boldsymbol{\xi}$에 대해 $\mathbf{x}$에서 볼록하다고 가정합니다.좀 더 구체적으로 말하자면, 확률적 경사하강법 업데이트를 고려한다. 

$$\mathbf{x}_{t+1} = \mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}),$$

여기서 $f(\boldsymbol{\xi}_t, \mathbf{x})$은 $t$단계의 일부 분포에서 도출된 훈련 예제 $\boldsymbol{\xi}_t$에 대한 목적 함수이고 $\mathbf{x}$는 모델 모수입니다.표시 기준 

$$R(\mathbf{x}) = E_{\boldsymbol{\xi}}[f(\boldsymbol{\xi}, \mathbf{x})]$$

예상 위험과 $R^*$까지 $\mathbf{x}$과 관련하여 최소값입니다.마지막으로 $\mathbf{x}^*$를 미니마이저로 지정합니다 ($\mathbf{x}$이 정의된 도메인 내에 존재한다고 가정합니다).이 경우 시간 $t$의 현재 매개변수 $\mathbf{x}_t$과 위험 최소화 장치 $\mathbf{x}^*$ 사이의 거리를 추적하고 시간이 지남에 따라 개선되는지 확인할 수 있습니다. 

$$\begin{aligned}    &\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \\ =& \|\mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}) - \mathbf{x}^*\|^2 \\    =& \|\mathbf{x}_{t} - \mathbf{x}^*\|^2 + \eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 - 2 \eta_t    \left\langle \mathbf{x}_t - \mathbf{x}^*, \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\right\rangle.   \end{aligned}$$
:eqlabel:`eq_sgd-xt+1-xstar`

우리는 확률 적 기울기 $\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})$의 $L_2$ 규범이 어떤 상수 $L$로 묶여 있다고 가정하므로 

$$\eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 \leq \eta_t^2 L^2.$$
:eqlabel:`eq_sgd-L`

우리는 주로 $\mathbf{x}_t$과 $\mathbf{x}^*$ 사이의 거리가 예상보다 어떻게 변하는지에 관심이 있습니다.실제로 특정 단계 시퀀스에 대해 $\boldsymbol{\xi}_t$에 따라 거리가 늘어날 수 있습니다.따라서 내적을 바인딩해야 합니다.모든 볼록 함수 $f$에 대해 모든 $\mathbf{x}$와 $\mathbf{y}$에 대해 $f(\mathbf{y}) \geq f(\mathbf{x}) + \langle f'(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$을 보유하고 있기 때문에 볼록성에 의해 우리는 

$$f(\boldsymbol{\xi}_t, \mathbf{x}^*) \geq f(\boldsymbol{\xi}_t, \mathbf{x}_t) + \left\langle \mathbf{x}^* - \mathbf{x}_t, \partial_{\mathbf{x}} f(\boldsymbol{\xi}_t, \mathbf{x}_t) \right\rangle.$$
:eqlabel:`eq_sgd-f-xi-xstar`

두 부등식 :eqref:`eq_sgd-L`와 :eqref:`eq_sgd-f-xi-xstar`를 :eqref:`eq_sgd-xt+1-xstar`에 연결하면 다음과 같이 시간 $t+1$에서 매개 변수 간의 거리에 대한 경계를 얻습니다. 

$$\|\mathbf{x}_{t} - \mathbf{x}^*\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \geq 2 \eta_t (f(\boldsymbol{\xi}_t, \mathbf{x}_t) - f(\boldsymbol{\xi}_t, \mathbf{x}^*)) - \eta_t^2 L^2.$$
:eqlabel:`eqref_sgd-xt-diff`

이는 현재 손실과 최적 손실의 차이가 $\eta_t L^2/2$를 초과하는 한 진전을 이루고 있음을 의미합니다.이 차이는 0으로 수렴할 수밖에 없기 때문에 학습률 $\eta_t$도 *사라져야합니다*. 

다음으로 우리는 :eqref:`eqref_sgd-xt-diff` 이상의 기대치를 가지고 있습니다.이 수익률 

$$E\left[\|\mathbf{x}_{t} - \mathbf{x}^*\|^2\right] - E\left[\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2\right] \geq 2 \eta_t [E[R(\mathbf{x}_t)] - R^*] -  \eta_t^2 L^2.$$

마지막 단계는 $t \in \{1, \ldots, T\}$의 불평등을 합산하는 것입니다.합계가 망원경이기 때문에 우리는 더 낮은 용어를 떨어 뜨림으로써 

$$\|\mathbf{x}_1 - \mathbf{x}^*\|^2 \geq 2 \left (\sum_{t=1}^T   \eta_t \right) [E[R(\mathbf{x}_t)] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$
:eqlabel:`eq_sgd-x1-xstar`

우리는 $\mathbf{x}_1$가 주어져서 기대치를 떨어 뜨릴 수 있다는 것을 악용했습니다.마지막 정의 

$$\bar{\mathbf{x}} \stackrel{\mathrm{def}}{=} \frac{\sum_{t=1}^T \eta_t \mathbf{x}_t}{\sum_{t=1}^T \eta_t}.$$

이후 

$$E\left(\frac{\sum_{t=1}^T \eta_t R(\mathbf{x}_t)}{\sum_{t=1}^T \eta_t}\right) = \frac{\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)]}{\sum_{t=1}^T \eta_t} = E[R(\mathbf{x}_t)],$$

젠슨의 불평등 ($i=t$에서 $i=t$, $\alpha_i = \eta_t/\sum_{t=1}^T \eta_t$을 설정) 과 $R$의 볼록성에 따라 $E[R(\mathbf{x}_t)] \geq E[R(\bar{\mathbf{x}})]$을 따릅니다. 

$$\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)] \geq \sum_{t=1}^T \eta_t  E\left[R(\bar{\mathbf{x}})\right].$$

이것을 부등식 :eqref:`eq_sgd-x1-xstar`에 꽂으면 한계가 생깁니다. 

$$
\left[E[\bar{\mathbf{x}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t},
$$

여기서 $r^2 \stackrel{\mathrm{def}}{=} \|\mathbf{x}_1 - \mathbf{x}^*\|^2$는 모수의 초기 선택과 최종 결과 사이의 거리에 대한 경계입니다.요컨대, 수렴 속도는 확률적 기울기의 노름이 어떻게 제한되는지 ($L$) 와 초기 모수 값이 최적성에서 얼마나 멀리 떨어져 있는지 ($r$) 에 달려 있습니다.한도는 $\mathbf{x}_T$이 아닌 $\bar{\mathbf{x}}$로 간주된다는 점에 유의하십시오.$\bar{\mathbf{x}}$는 최적화 경로의 평활화된 버전이기 때문에 이러한 경우입니다.$r, L$ 및 $T$이 알려질 때마다 학습률 $\eta = r/(L \sqrt{T})$를 선택할 수 있습니다.이는 상한값 $rL/\sqrt{T}$로 산출됩니다.즉, 우리는 비율 $\mathcal{O}(1/\sqrt{T})$과 함께 최적의 솔루션으로 수렴합니다. 

## 스토캐스틱 그래디언트 및 유한 샘플

지금까지 확률 적 경사 하강에 대해 이야기 할 때 약간 빠르고 느슨하게 연주했습니다.일반적으로 일부 배포판 $p(x, y)$의 레이블 $y_i$을 사용하여 인스턴스 $x_i$를 그리고 이를 사용하여 어떤 방식으로 모델 매개변수를 업데이트한다고 가정했습니다.특히 유한 표본 크기의 경우 일부 함수 $\delta_{x_i}$ 및 $\delta_{y_i}$에 대한 이산 분포 $p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$을 사용하면 확률 적 기울기 하강을 수행 할 수 있다고 주장했습니다. 

그러나 이것은 실제로 우리가 한 일이 아닙니다.현재 섹션의 장난감 예제에서 우리는 단순히 확률적이지 않은 그래디언트에 노이즈를 추가했습니다. 즉, $(x_i, y_i)$ 쌍을 가진 척했습니다.이것이 여기에 정당하다는 것이 밝혀졌습니다 (자세한 논의는 연습 문제 참조).더 문제가되는 것은 이전의 모든 토론에서 분명히 그렇게하지 않았다는 것입니다.대신 모든 인스턴스를*정확히 한 번* 반복했습니다.이것이 바람직한 이유를 확인하려면 그 반대를 고려하십시오. 즉, 이산 분포*를 대체*에서 $n$개의 관측치를 표본으로 추출하고 있습니다.요소 $i$를 무작위로 선택할 확률은 $1/n$입니다.따라서* 적어도* 한 번 선택하면됩니다. 

$$P(\mathrm{choose~} i) = 1 - P(\mathrm{omit~} i) = 1 - (1-1/n)^n \approx 1-e^{-1} \approx 0.63.$$

비슷한 추론에 따르면 일부 샘플 (예: 훈련 예) *정확히 한 번*을 선택할 확률은 다음과 같습니다. 

$${n \choose 1} \frac{1}{n} \left(1-\frac{1}{n}\right)^{n-1} = \frac{n}{n-1} \left(1-\frac{1}{n}\right)^{n} \approx e^{-1} \approx 0.37.$$

이로 인해 표본 추출*대체 없이*에 비해 분산이 증가하고 데이터 효율성이 감소합니다.따라서 실제로 후자를 수행합니다 (이 책 전체에서 기본 선택입니다).마지막으로, 훈련 데이터세트를 반복하여 통과하면*다른* 무작위 순서로 통과합니다. 

## 요약

* 볼록 문제의 경우 다양한 학습 속도 선택에 대해 확률 적 기울기 하강이 최적의 해로 수렴한다는 것을 증명할 수 있습니다.
* 딥 러닝의 경우 일반적으로 그렇지 않습니다.그러나 볼록 문제를 분석하면 최적화에 접근하는 방법, 즉 학습 속도를 너무 빠르지는 않지만 점진적으로 줄이는 방법에 대한 유용한 통찰력을 얻을 수 있습니다.
* 학습률이 너무 작거나 너무 클 때 문제가 발생합니다.실제로 적절한 학습률은 여러 번의 실험 후에 만 발견되는 경우가 많습니다.
* 훈련 데이터셋에 더 많은 예제가 있는 경우 경사하강법에 대한 각 반복을 계산하는 데 더 많은 비용이 들기 때문에 이러한 경우에는 확률적 경사하강법을 사용하는 것이 좋습니다.
* 검사가 필요한 국소 최솟값 수가 지수일 수 있으므로 확률적 경사하강법에 대한 최적성 보장은 일반적으로 볼록하지 않은 경우에는 사용할 수 없습니다.

## 연습문제

1. 확률적 경사하강법 및 다양한 반복 횟수에 대해 서로 다른 학습률 일정을 실험해 보십시오.특히 최적해 $(0, 0)$로부터의 거리를 반복 횟수의 함수로 플로팅합니다.
1. 함수 $f(x_1, x_2) = x_1^2 + 2 x_2^2$에 대해 기울기에 정규 잡음을 추가하는 것이 손실 함수 $f(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$를 최소화하는 것과 동일하다는 것을 증명하십시오. 여기서 $\mathbf{x}$는 정규 분포에서 추출됩니다.
1. $\{(x_1, y_1), \ldots, (x_n, y_n)\}$에서 대체를 사용하여 샘플링할 때와 대체 없이 샘플링하는 경우 확률적 경사하강법의 수렴을 비교합니다.
1. 일부 기울기 (또는 이와 관련된 일부 좌표) 가 다른 모든 기울기보다 일관되게 큰 경우 확률적 경사 하강 솔버를 어떻게 변경하시겠습니까?
1. $f(x) = x^2 (1 + \sin x)$라고 가정합니다.$f$에는 몇 개의 로컬 최소값이 있습니까?$f$를 최소화하기 위해 모든 국소 최솟값을 평가해야 하는 방식으로 변경할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/352)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/497)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1067)
:end_tab:
