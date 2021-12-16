# 모멘텀
:label:`sec_momentum`

:numref:`sec_sgd`에서는 확률적 경사 하강을 수행할 때, 즉 그래디언트의 잡음이 있는 변형만 사용할 수 있는 최적화를 수행할 때 어떤 일이 발생하는지 검토했습니다.특히, 잡음이 있는 그래디언트의 경우 잡음에 직면한 학습 속도를 선택할 때 더욱 주의해야 한다는 것을 알게 되었습니다.너무 빨리 줄이면 수렴이 멈춥니다.너무 관대하면 소음이 계속해서 최적에서 멀어지기 때문에 충분한 솔루션으로 수렴하지 못합니다. 

## 기초

이 섹션에서는 특히 일반적으로 사용되는 특정 유형의 최적화 문제에 대해 보다 효과적인 최적화 알고리즘을 살펴보겠습니다. 

### 누설 평균

이전 섹션에서는 계산 속도를 높이기 위한 수단으로 미니배치 SGD에 대해 논의했습니다.또한 그라디언트를 평균화하면 분산의 양이 줄어드는 좋은 부작용도 있었습니다.미니배치 확률적 경사하강법은 다음과 같이 계산할 수 있습니다. 

$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

표기법을 단순하게 유지하기 위해 여기서는 시간 $t-1$에 업데이트된 가중치를 사용하여 샘플 $i$에 대한 확률적 기울기 하강법으로 $\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$를 사용했습니다.미니 배치에서 평균 그래디언트를 넘어서는 분산 감소 효과를 활용할 수 있다면 좋을 것입니다.이 작업을 수행하는 한 가지 옵션은 기울기 계산을 “누설 평균”으로 바꾸는 것입니다. 

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$

일부 $\beta \in (0, 1)$를 위해.이것은 순시 그래디언트를 여러*과거* 그라디언트에 대해 평균화된 것으로 효과적으로 대체합니다. $\mathbf{v}$를*모멘텀*이라고 합니다.목적 함수 풍경을 굴리는 무거운 공이 과거의 힘에 어떻게 통합되는지와 유사한 과거 그라디언트를 축적합니다.무슨 일이 일어나고 있는지 자세히 알아보기 위해 $\mathbf{v}_t$을 재귀적으로 확장해 보겠습니다. 

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$

큰 $\beta$는 장거리 평균에 해당하는 반면, 작은 $\beta$는 기울기 방법에 비해 약간의 수정에 불과합니다.새로운 그래디언트 대체는 더 이상 특정 인스턴스에서 가장 가파른 하강 방향을 가리키지 않고 과거 그라디언트의 가중 평균 방향을 가리킵니다.이를 통해 실제로 그래디언트를 계산하는 비용 없이 배치에 대한 평균화의 이점을 대부분 실현할 수 있습니다.이 평균화 절차에 대해서는 나중에 더 자세히 살펴보겠습니다. 

위의 추론은 모멘텀이 있는 그래디언트와 같이 현재*가속된* 기울기 방법으로 알려진 기초를 형성했습니다.최적화 문제가 조건이 좋지 않은 경우 (즉, 좁은 협곡과 유사한 다른 방향보다 진행이 훨씬 느린 방향이있는 경우) 훨씬 더 효과적이라는 추가적인 이점을 누리고 있습니다.또한 더 안정적인 하강 방향을 얻기 위해 후속 기울기에 대해 평균을 낼 수 있습니다.실제로 잡음이없는 볼록 문제에 대해서도 가속의 측면은 모멘텀이 작동하는 주요 이유 중 하나이며 모멘텀이 잘 작동하는 이유 중 하나입니다. 

예상대로, 효능으로 인해 모멘텀은 딥 러닝 및 그 이상을 위한 최적화에서 잘 연구된 주제입니다.예를 들어, 심층 분석 및 대화형 애니메이션은 아름다운 [해설 기사](https://distill.pub/2017/momentum/) by :cite:`Goh.2017`) 를 참조하십시오.:cite:`Polyak.1964`에 의해 제안되었습니다. :cite:`Nesterov.2018`는 볼록 최적화의 맥락에서 상세한 이론적 토론을 가지고 있습니다.딥 러닝의 모멘텀은 오랫동안 유익한 것으로 알려져 왔습니다.예를 들어, 자세한 내용은 :cite:`Sutskever.Martens.Dahl.ea.2013`의 토론을 참조하십시오. 

### 조건이 좋지 않은 문제

운동량 방법의 기하학적 속성을 더 잘 이해하기 위해 상당히 덜 유쾌한 목적 함수를 사용하더라도 기울기 하강을 다시 살펴봅니다.:numref:`sec_gd`에서 우리는 $f(\mathbf{x}) = x_1^2 + 2 x_2^2$, 즉 적당히 왜곡 된 타원체 대물 렌즈를 사용했음을 상기하십시오.다음을 통해 $x_1$ 방향으로 확장하여이 기능을 더 왜곡합니다. 

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

이전과 마찬가지로 $f$의 최소값은 $(0, 0)$입니다.이 기능은 $x_1$ 방향으로 매우 평평합니다.이 새로운 함수에서 이전과 같이 기울기 하강을 수행 할 때 어떤 일이 발생하는지 살펴 보겠습니다.우리는 $0.4$의 학습 속도를 선택합니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

구조적으로 $x_2$ 방향의 기울기는*훨씬 더 높으며 수평 $x_1$ 방향보다 훨씬 빠르게 변합니다.따라서 우리는 두 가지 바람직하지 않은 선택 사이에 갇혀 있습니다. 작은 학습 속도를 선택하면 해가 $x_2$ 방향으로 갈라지지 않지만 $x_1$ 방향으로 느린 수렴으로 안장됩니다.반대로 학습률이 높으면 $x_1$ 방향으로 빠르게 진행되지만 $x_2$에서는 갈라집니다.아래 예는 학습률이 $0.4$에서 $0.6$로 약간 증가한 후에도 어떤 일이 발생하는지 보여줍니다.$x_1$ 방향의 수렴은 향상되지만 전체 솔루션 품질은 훨씬 나쁩니다.

```{.python .input}
#@tab all
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

### 모멘텀 방법

운동량 방법을 사용하면 위에서 설명한 기울기 하강 문제를 해결할 수 있습니다.위의 최적화 추적을 살펴보면 과거의 평균 그래디언트가 잘 작동한다는 것을 알 수 있습니다.결국 $x_1$ 방향에서 이것은 잘 정렬 된 그라디언트를 집계하여 모든 단계에서 커버하는 거리를 증가시킵니다.반대로, 그라디언트가 진동하는 $x_2$ 방향에서 집계 그래디언트는 서로 상쇄되는 진동으로 인해 스텝 크기를 줄입니다.기울기 $\mathbf{g}_t$ 대신 $\mathbf{v}_t$을 사용하면 다음과 같은 업데이트 방정식이 생성됩니다. 

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

$\beta = 0$의 경우 규칙적인 기울기 하강을 복구합니다.수학적 속성을 자세히 살펴보기 전에 알고리즘이 실제로 어떻게 작동하는지 간단히 살펴 보겠습니다.

```{.python .input}
#@tab all
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

보시다시피, 이전에 사용한 것과 동일한 학습률에도 불구하고 모멘텀은 여전히 잘 수렴됩니다.모멘텀 파라미터를 낮추면 어떤 일이 발생하는지 살펴보겠습니다.$\beta = 0.25$로 절반으로 줄이면 거의 수렴하지 않는 궤도가 생깁니다.그럼에도 불구하고 모멘텀이 없는 것보다 (해답이 갈라질 때) 훨씬 낫습니다.

```{.python .input}
#@tab all
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

모멘텀을 확률적 경사하강법, 특히 미니배치 확률적 경사하강법과 결합할 수 있습니다.유일한 변경 사항은 이 경우 그라디언트 $\mathbf{g}_{t, t-1}$를 $\mathbf{g}_t$로 대체한다는 것입니다.마지막으로 편의를 위해 $\mathbf{v}_0 = 0$을 시간 $t=0$에 초기화합니다.누수 평균화가 업데이트에 실제로 어떤 영향을 미치는지 살펴 보겠습니다. 

### 유효 샘플 중량

$\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$을 상기하십시오.한도에서 용어의 합계는 최대 $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$입니다.즉, 기울기 하강 또는 확률적 경사 하강에서 크기 $\eta$의 단계를 취하는 대신 크기 $\frac{\eta}{1-\beta}$의 단계를 취하는 동시에 잠재적으로 훨씬 더 나은 동작하는 하강 방향을 처리합니다.이는 하나로 두 가지 이점이 있습니다.$\beta$의 다양한 선택 항목에 대해 가중치가 어떻게 작용하는지 설명하기 위해 아래 다이어그램을 살펴보십시오.

```{.python .input}
#@tab all
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

## 실용적인 실험

모멘텀이 실제로 어떻게 작동하는지, 즉 적절한 옵티마이저의 맥락에서 사용될 때 살펴 보겠습니다.이를 위해서는 좀 더 확장 가능한 구현이 필요합니다. 

### 처음부터 구현

(미니배치) 확률적 구배 하강법과 비교할 때 운동량 방법은 일련의 보조 변수, 즉 속도를 유지해야 합니다.기울기 (및 최적화 문제의 변수) 와 모양이 같습니다.아래 구현에서는 이러한 변수를 `states`라고 부릅니다.

```{.python .input}
#@tab mxnet,pytorch
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)
```

```{.python .input}
#@tab tensorflow
def init_momentum_states(features_dim):
    v_w = tf.Variable(d2l.zeros((features_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    return (v_w, v_b)
```

```{.python .input}
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v
```

```{.python .input}
#@tab pytorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
            v[:].assign(hyperparams['momentum'] * v + g)
            p[:].assign(p - hyperparams['lr'] * v)
```

이것이 실제로 어떻게 작동하는지 봅시다.

```{.python .input}
#@tab all
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

모멘텀 하이퍼파라미터 `momentum`를 0.9로 늘리면 유효 표본 크기인 $\frac{1}{1 - 0.9} = 10$가 훨씬 더 커집니다.문제를 통제하기 위해 학습률을 $0.01$으로 약간 줄입니다.

```{.python .input}
#@tab all
train_momentum(0.01, 0.9)
```

학습률을 줄이면 원활하지 않은 최적화 문제의 모든 문제를 추가로 해결할 수 있습니다.이 값을 $0.005$로 설정하면 수렴 특성이 양호합니다.

```{.python .input}
#@tab all
train_momentum(0.005, 0.9)
```

### 간결한 구현

표준 `sgd` 솔버에 이미 모멘텀이 내장되어 있기 때문에 글루온에서는 할 일이 거의 없습니다.일치하는 매개변수를 설정하면 매우 유사한 궤적이 생성됩니다.

```{.python .input}
d2l.train_concise_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

## 이론적 분석

지금까지 $f(x) = 0.1 x_1^2 + 2 x_2^2$의 2D 예는 다소 인위적인 것처럼 보였습니다.이제 우리는 이것이 적어도 볼록 2차 목적 함수를 최소화하는 경우에 발생할 수 있는 문제의 유형을 실제로 상당히 대표한다는 것을 알게 될 것입니다. 

### 2차 볼록 함수

다음 함수를 고려하십시오. 

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$

이것은 일반적인 2차 함수입니다.양의 정부호 행렬 $\mathbf{Q} \succ 0$의 경우, 즉 양의 고유값을 갖는 행렬의 경우 최소값 $b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$을 갖는 최솟값이 $\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$입니다.따라서 $h$를 다음과 같이 다시 작성할 수 있습니다. 

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$

기울기는 $\partial_{\mathbf{x}} f(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$로 지정됩니다.즉, $\mathbf{x}$과 미니마이저 사이의 거리에 $\mathbf{Q}$를 곱한 값으로 주어집니다.결과적으로 모멘텀은 $\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$ 항의 선형 조합입니다. 

$\mathbf{Q}$은 양의 정부호이므로 직교 (회전) 행렬 $\mathbf{O}$와 양의 고유값으로 구성된 대각 행렬 $\boldsymbol{\Lambda}$에 대해 $\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$을 통해 고유시스템으로 분해될 수 있습니다.이를 통해 변수를 $\mathbf{x}$에서 $\mathbf{z} := \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$로 변경하여 훨씬 간단한 표현식을 얻을 수 있습니다. 

$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$

여기 $b' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$입니다.$\mathbf{O}$는 직교 행렬일 뿐이므로 기울기를 의미 있는 방식으로 교란시키지 않습니다.$\mathbf{z}$로 표현되는 기울기 하강은 다음과 같습니다. 

$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$

이 표현식에서 중요한 사실은 기울기 하강법이 서로 다른 고유공간 사이에*혼합되지 않는다는 것입니다.즉, $\mathbf{Q}$의 고유시스템으로 표현될 때 최적화 문제는 좌표 방식으로 진행됩니다.이는 또한 모멘텀을 유지합니다. 

$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$

이를 통해 우리는 다음과 같은 정리를 증명했습니다. 볼록 2 차 함수에 대한 모멘텀이 있거나없는 기울기 하강은 2 차 행렬의 고유 벡터 방향으로 좌표 별 최적화로 분해됩니다. 

### 스칼라 함수

위의 결과를 감안할 때 $f(x) = \frac{\lambda}{2} x^2$ 함수를 최소화 할 때 어떤 일이 발생하는지 살펴 보겠습니다.기울기 하강의 경우 

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$

$|1 - \eta \lambda| < 1$이 될 때마다 이 최적화는 기하 급수적 인 속도로 수렴합니다. $t$ 단계 후에 $x_t = (1 - \eta \lambda)^t x_0$가 있기 때문입니다.이것은 학습률 $\eta$를 $\eta \lambda = 1$까지 증가시키면서 초기에 수렴률이 어떻게 향상되는지 보여줍니다.그 외에도 상황이 갈라지고 $\eta \lambda > 2$의 경우 최적화 문제가 갈라집니다.

```{.python .input}
#@tab all
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

모멘텀의 경우 수렴을 분석하기 위해 먼저 업데이트 방정식을 두 개의 스칼라로 다시 작성합니다. 하나는 $x$이고 다른 하나는 모멘텀 $v$입니다.이 결과는 다음과 같습니다 

$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$

우리는 $\mathbf{R}$을 사용하여 수렴 동작을 지배하는 $2 \times 2$를 나타 냈습니다.$t$ 단계 후에 초기 선택 항목 $[v_0, x_0]$은 $\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$이 됩니다.따라서 수렴 속도를 결정하는 것은 $\mathbf{R}$의 고유값에 달려 있습니다.훌륭한 애니메이션은 [증류 게시물](https://://distill.pub/2017/momentum/) of :cite:`Goh.2017`) 을 참조하고 자세한 분석은 :cite:`Flammarion.Bach.2015`를 참조하십시오.$0 < \eta \lambda < 2 + 2 \beta$의 모멘텀이 수렴한다는 것을 보여줄 수 있습니다.기울기 하강법의 경우 $0 < \eta \lambda < 2$과 비교할 때 실현 가능한 모수의 범위가 더 넓습니다.또한 일반적으로 $\beta$의 큰 값이 바람직하다는 것을 암시합니다.자세한 내용은 상당한 양의 기술적 세부 사항이 필요하며 관심있는 독자는 원본 출판물을 참조하는 것이 좋습니다. 

## 요약

* 모멘텀은 그래디언트를 과거 그래디언트에 비해 누수 평균으로 대체합니다이로 인해 수렴이 크게 가속화됩니다.
* 잡음이 없는 경사하강법과 (잡음이 있는) 확률적 경사하강법 모두에 적합합니다.
* 모멘텀은 확률적 경사하강법 시 발생할 가능성이 훨씬 높은 최적화 프로세스의 실속을 방지합니다.
* 과거 데이터의 지수 하향 가중치로 인해 유효 기울기 수는 $\frac{1}{1-\beta}$에 의해 제공됩니다.
* 볼록 2차 문제의 경우 이를 명시적으로 자세히 분석할 수 있습니다.
* 구현은 매우 간단하지만 추가 상태 벡터 (모멘텀 $\mathbf{v}$) 를 저장해야 합니다.

## 연습문제

1. 모멘텀 하이퍼파라미터와 학습률의 다른 조합을 사용하여 다양한 실험 결과를 관찰하고 분석합니다.
1. 고유값이 여러 개인 2차 문제 (예: $f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$, 예: $\lambda_i = 2^{-i}$) 에 대해 GD와 모멘텀을 사용해 보십시오.초기화 $x_i = 1$에서 $x$의 값이 어떻게 감소하는지 플로팅합니다.
1. $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$에 대한 최소값과 미니마이저를 파생합니다.
1. 모멘텀으로 확률적 경사 하강을 수행할 때 어떤 변화가 있을까요?모멘텀과 함께 미니배치 확률적 경사하강법을 사용하면 어떻게 될까요?매개 변수로 실험해 보시겠습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/354)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1070)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1071)
:end_tab:
