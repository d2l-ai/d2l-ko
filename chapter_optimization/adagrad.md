# 아다그라드
:label:`sec_adagrad`

자주 발생하지 않는 특징을 가진 학습 문제를 고려하는 것부터 시작하겠습니다. 

## 희소 특징 및 학습률

우리가 언어 모델을 훈련하고 있다고 상상해 보세요.정확성을 높이기 위해 일반적으로 훈련을 계속할 때 일반적으로 $\mathcal{O}(t^{-\frac{1}{2}})$ 이하의 속도로 학습률을 낮추는 것이 좋습니다.이제 희소 특징, 즉 드물게 발생하는 특징에 대한 모델 학습을 고려해 보십시오.이것은 자연어에서 흔히 볼 수 있습니다. 예를 들어, *학습*보다 *사전 조건*이라는 단어를 볼 가능성이 훨씬 적습니다.그러나 전산 광고 및 개인화 된 협업 필터링과 같은 다른 영역에서도 일반적입니다.결국 소수의 사람들에게만 흥미로운 것들이 많이 있습니다. 

자주 사용되지 않는 기능과 관련된 매개 변수는 이러한 기능이 발생할 때마다 의미 있는 업데이트만 받습니다.학습 속도가 감소하면 공통 기능에 대한 매개 변수가 최적 값으로 다소 빠르게 수렴되는 상황에 처할 수 있지만 드문 기능의 경우 최적 값을 결정하기 전에 충분히 자주 관찰하지 못하는 상황이 발생할 수 있습니다.즉, 학습 속도가 빈번한 특징의 경우 너무 느리게 감소하거나 드문 특징의 경우 너무 빨리 감소합니다. 

이 문제를 해결하기 위한 가능한 해킹은 특정 기능을 보는 횟수를 세고 이를 학습률을 조정하기 위한 시계로 사용하는 것입니다.즉, $\eta = \frac{\eta_0}{\sqrt{t + c}}$ 형식의 학습률을 선택하는 대신 $\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$을 사용할 수 있습니다.여기서 $s(i, t)$은 $t$ 시간까지 관찰한 피쳐 $i$에 대한 0이 아닌 수의 수를 계산합니다.이는 의미 있는 오버헤드 없이 실제로 구현하기가 매우 쉽습니다.그러나 희소성이 없을 때마다 실패하지만 그래디언트가 매우 작고 거의 크지 않은 데이터만 있으면 실패합니다.결국 관찰 된 특징으로 인정되는 것 사이에 선을 그릴 곳이 어디인지는 확실하지 않습니다. 

:cite:`Duchi.Hazan.Singer.2011`의 아다그라드는 다소 조잡한 카운터 $s(i, t)$를 이전에 관찰된 기울기의 제곱의 집합으로 대체하여 이 문제를 해결합니다.특히 학습률을 조정하는 수단으로 $s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$을 사용합니다.여기에는 두 가지 이점이 있습니다. 첫째, 그래디언트가 충분히 큰 시점을 더 이상 결정할 필요가 없습니다.둘째, 그라디언트의 크기에 따라 자동으로 크기가 조정됩니다.일상적으로 큰 그라디언트에 해당하는 좌표는 크게 축소되는 반면, 그라디언트가 작은 좌표는 훨씬 더 부드럽게 처리됩니다.실제로 이것은 계산 광고 및 관련 문제에 대한 매우 효과적인 최적화 절차로 이어집니다.그러나 이것은 사전 컨디셔닝의 맥락에서 가장 잘 이해되는 아다그라드에 내재된 몇 가지 추가 이점을 숨 깁니다. 

## 사전 컨디셔닝

볼록 최적화 문제는 알고리즘의 특성을 분석하는 데 유용합니다.결국 대부분의 비볼록 문제의 경우 의미 있는 이론적 보장을 도출하기는 어렵지만, *직감*과*통찰력*은 종종 이어집니다.$f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$를 최소화하는 문제를 살펴 보겠습니다. 

:numref:`sec_momentum`에서 보았 듯이이 문제를 고유 구성 $\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$로 다시 작성하여 각 좌표를 개별적으로 해결할 수있는 훨씬 단순화 된 문제에 도달 할 수 있습니다. 

$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$

여기서 우리는 $\mathbf{x} = \mathbf{U} \mathbf{x}$을 사용했고 결과적으로 $\mathbf{c} = \mathbf{U} \mathbf{c}$를 사용했습니다.수정된 문제는 최소값 $\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$와 최소값 $-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$을 갖습니다.$\boldsymbol{\Lambda}$는 고유값 $\mathbf{Q}$을 포함하는 대각 행렬이므로 계산하기가 훨씬 쉽습니다. 

$\mathbf{c}$를 약간 교란하면 $f$의 미니마이저에서 약간의 변화만 발견하기를 바랍니다.안타깝게도 그렇지 않습니다.$\mathbf{c}$의 약간의 변화는 $\bar{\mathbf{c}}$에서도 똑같이 약간의 변화로 이어지지만, 미니마이저 $f$ (및 각각 $\bar{f}$) 의 경우에는 해당되지 않습니다.고유값 $\boldsymbol{\Lambda}_i$이 클 때마다 $\bar{x}_i$과 최소값 $\bar{f}$의 작은 변화만 볼 수 있습니다.반대로, $\boldsymbol{\Lambda}_i$의 작은 경우 $\bar{x}_i$의 변화는 극적일 수 있습니다.가장 큰 고유값과 가장 작은 고유값 간의 비율을 최적화 문제의 조건수라고 합니다. 

$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$

조건 번호 $\kappa$이 크면 최적화 문제를 정확하게 풀기가 어렵습니다.우리는 큰 동적 범위의 값을 올바르게 얻기 위해 주의를 기울여야 합니다.우리의 분석은 다소 순진한 질문이지만 명백한 질문으로 이어집니다. 모든 고유값이 $1$가되도록 공간을 왜곡하여 문제를 단순히 “해결”할 수는 없습니다.이론적으로 이것은 매우 쉽습니다. 문제를 $\mathbf{x}$에서 $\mathbf{z} := \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$의 1로 다시 스케일링하려면 $\mathbf{Q}$의 고유값과 고유 벡터만 있으면됩니다.새 좌표계에서는 $\mathbf{x}^\top \mathbf{Q} \mathbf{x}$을 $\|\mathbf{z}\|^2$로 단순화할 수 있습니다.아아, 이것은 다소 비실용적 인 제안입니다.고유값과 고유벡터를 계산하는 것은 일반적으로 실제 문제를 푸는 것보다 훨씬 더 비쌉니다. 

고유값을 정확히 계산하는 것은 비용이 많이 들 수 있지만, 고유값을 추측하고 어느 정도 대략적으로 계산하는 것이 아무것도 하지 않는 것보다 훨씬 나을 수 있습니다.특히 $\mathbf{Q}$의 대각선 항목을 사용하고 그에 따라 크기를 다시 조정할 수 있습니다.이 방법은 고유값을 계산하는 것보다 훨씬 저렴합니다. 

$$\tilde{\mathbf{Q}} = \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$

이 경우 우리는 $\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$, 특히 $i$에 대해 $\tilde{\mathbf{Q}}_{ii} = 1$를 가지고 있습니다.대부분의 경우 이렇게 하면 조건 번호가 상당히 단순화됩니다.예를 들어 앞서 논의한 사례에서는 문제가 축 정렬되므로 당면한 문제를 완전히 없앨 수 있습니다. 

안타깝게도 우리는 또 다른 문제에 직면합니다. 딥 러닝에서는 일반적으로 목적 함수의 2 차 도함수에 액세스 할 수 없습니다. $\mathbf{x} \in \mathbb{R}^d$의 경우 미니 배치에서도 두 번째 도함수가 $\mathcal{O}(d^2)$ 공간이 필요하고 계산하는 데 작동하므로 실제로 실현 불가능합니다.Adagrad의 독창적인 아이디어는 헤세 행렬의 애매한 대각선에 프록시를 사용하는 것입니다. 이 프록시는 계산하기에 상대적으로 저렴하고 효과적인 기울기 자체의 크기입니다. 

이것이 왜 작동하는지 알아보기 위해 $\bar{f}(\bar{\mathbf{x}})$를 살펴보겠습니다.우리는 그것을 가지고 있습니다 

$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$

여기서 $\bar{\mathbf{x}}_0$은 $\bar{f}$의 최소화기입니다.따라서 기울기의 크기는 $\boldsymbol{\Lambda}$와 최적성으로부터의 거리에 따라 달라집니다.$\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$이 변경되지 않았다면 이것이 필요한 전부일 것입니다.결국, 이 경우 기울기 $\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$의 크기로 충분합니다.AdaGrad는 확률적 경사 하강 알고리즘이므로 최적성에서도 분산이 0이 아닌 기울기를 볼 수 있습니다.따라서 기울기의 분산을 헤세 행렬의 척도에 대한 값싼 프록시로 안전하게 사용할 수 있습니다.철저한 분석은 이 섹션의 범위를 벗어납니다 (여러 페이지로 구성됨).자세한 내용은 독자를 :cite:`Duchi.Hazan.Singer.2011`로 안내합니다. 

## 알고리즘

위에서 토론을 공식화하겠습니다.변수 $\mathbf{s}_t$를 사용하여 다음과 같이 과거 기울기 분산을 누적합니다. 

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

여기서 연산은 좌표로 적용됩니다.즉, $\mathbf{v}^2$에는 항목 $v_i^2$이 있습니다.마찬가지로 $\frac{1}{\sqrt{v}}$에는 항목 $\frac{1}{\sqrt{v_i}}$가 있고 $\mathbf{u} \cdot \mathbf{v}$에는 항목 $u_i v_i$이 있습니다.이전과 마찬가지로 $\eta$는 학습률이고 $\epsilon$은 $0$으로 나누지 않도록 보장하는 가산 상수입니다.마지막으로 $\mathbf{s}_0 = \mathbf{0}$를 초기화합니다. 

모멘텀의 경우와 마찬가지로 보조 변수를 추적해야 합니다. 이 경우 좌표당 개별 학습률을 허용합니다.주요 비용은 일반적으로 $l(y_t, f(\mathbf{x}_t, \mathbf{w}))$와 그 파생물을 계산하는 것이기 때문에 SGD에 비해 Adagrad의 비용이 크게 증가하지는 않습니다. 

$\mathbf{s}_t$에서 제곱 그래디언트를 누적한다는 것은 $\mathbf{s}_t$가 기본적으로 선형 속도로 증가한다는 것을 의미합니다 (기울기가 초기에 감소하기 때문에 실제로는 선형적으로보다 다소 느림).이로 인해 좌표별로 조정되지만 $\mathcal{O}(t^{-\frac{1}{2}})$의 학습 속도가 발생합니다.볼록 문제의 경우 이는 완벽하게 적절합니다.하지만 딥 러닝에서는 학습 속도를 좀 더 느리게 낮추고 싶을 수도 있습니다.이로 인해 여러 가지 Adagrad 변형이 생겨 다음 장에서 논의 할 것입니다.지금은 이차 볼록 문제에서 어떻게 작동하는지 살펴 보겠습니다.이전과 동일한 문제를 사용합니다. 

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

우리는 이전과 동일한 학습률, 즉 $\eta = 0.4$를 사용하여 아다그라드를 구현할 것입니다.보시다시피 독립 변수의 반복 궤적은 더 부드럽습니다.그러나 $\boldsymbol{s}_t$의 누적 효과로 인해 학습률이 지속적으로 감소하므로 독립 변수가 반복의 이후 단계에서 많이 움직이지 않습니다.

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

```{.python .input}
#@tab all
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

학습률을 $2$로 높이면 훨씬 더 나은 행동을 볼 수 있습니다.이는 이미 잡음이없는 경우에도 학습률 감소가 다소 공격적 일 수 있음을 나타내며 매개 변수가 적절하게 수렴되도록해야합니다.

```{.python .input}
#@tab all
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

## 처음부터 구현

모멘텀 방법과 마찬가지로 아다그라드는 매개변수와 동일한 모양의 상태 변수를 유지해야 합니다.

```{.python .input}
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def init_adagrad_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)

def adagrad(params, grads, states, hyperparams):
    eps = 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(s + tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

:numref:`sec_minibatch_sgd`의 실험과 비교하여 모델을 훈련시키기 위해 더 큰 학습률을 사용합니다.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

## 간결한 구현

알고리즘 `adagrad`의 `Trainer` 인스턴스를 사용하여 글루온에서 아다그라드 알고리즘을 호출할 수 있습니다.

```{.python .input}
d2l.train_concise_ch11('adagrad', {'learning_rate': 0.1}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adagrad
d2l.train_concise_ch11(trainer, {'learning_rate' : 0.1}, data_iter)
```

## 요약

* Adagrad는 좌표별로 학습률을 동적으로 줄입니다.
* 기울기의 크기를 진행 속도를 조정하는 수단으로 사용합니다. 기울기가 큰 좌표는 더 작은 학습률로 보정됩니다.
* 딥러닝 문제에서는 메모리와 계산상의 제약으로 인해 일반적으로 정확한 2차 도함수를 계산할 수 없습니다.그라데이션은 유용한 프록시가 될 수 있습니다.
* 최적화 문제의 구조가 다소 고르지 않은 경우 Adagrad는 왜곡을 완화하는 데 도움이 될 수 있습니다.
* Adagrad는 자주 발생하지 않는 용어에 대해 학습률이 더 느리게 감소해야 하는 희소 특징에 특히 효과적입니다.
* 딥 러닝 문제에서 Adagrad는 때때로 학습률을 낮추는 데 너무 공격적일 수 있습니다.:numref:`sec_adam`의 맥락에서 이를 완화하기 위한 전략에 대해 논의할 것입니다.

## 연습문제

1. 직교 행렬 $\mathbf{U}$ 및 벡터 $\mathbf{c}$에 대해 $\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$가 유지된다는 것을 증명하십시오.변수의 직교 변화 후에도 섭동의 크기가 변하지 않는다는 것을 의미하는 이유는 무엇입니까?
1. $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$에 대해 아다그라드를 시험해보고 목적 함수가 45도 회전한 경우, 즉 $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$에 대해 시도해보십시오.다르게 동작하나요?
1. 행렬 $\mathbf{M}$의 고유값 $\lambda_i$가 $j$ 중 적어도 하나의 선택 항목에 대해 $|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$을 충족한다는 것을 나타내는 [게르슈고린의 원 정리](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem) 를 증명합니다.
1. 게르슈고린의 정리는 대각선으로 사전 조건화된 행렬 $\mathrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{M})$의 고유값에 대해 무엇을 알려줍니까?
1. 패션 MNIST에 적용할 때 :numref:`sec_lenet`와 같은 적절한 심층 네트워크를 위해 아다그라드를 사용해 보십시오.
1. 학습률이 덜 공격적으로 감소하기 위해 Adagrad를 어떻게 수정해야 할까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/355)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1072)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1073)
:end_tab:
