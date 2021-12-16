# 다변수 미적분학
:label:`sec_multivariable_calculus`

이제 단일 변수의 함수의 도함수를 상당히 잘 이해했으므로 잠재적으로 수십억 개의 가중치의 손실 함수를 고려하고 있던 원래 질문으로 돌아가 보겠습니다. 

## 고차원 차별화 :numref:`sec_single_variable_calculus`가 말하는 것은 수십억 개의 가중치 중 하나를 변경하여 다른 모든 가중치를 고정하면 어떤 일이 일어날 지 알 수 있다는 것입니다!이것은 단일 변수의 함수에 지나지 않으므로 다음과 같이 작성할 수 있습니다. 

$$L(w_1+\epsilon_1, w_2, \ldots, w_N) \approx L(w_1, w_2, \ldots, w_N) + \epsilon_1 \frac{d}{dw_1} L(w_1, w_2, \ldots, w_N).$$
:eqlabel:`eq_part_der`

한 변수에서 도함수를 호출하고 다른 변수는*부분 미분*으로 고정하고 :eqref:`eq_part_der`에서 도함수에 $\frac{\partial}{\partial w_1}$ 표기법을 사용합니다. 

이제 이것을 가지고 $w_2$를 $w_2 + \epsilon_2$로 약간 변경해 보겠습니다. 

$$
\begin{aligned}
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N) & \approx L(w_1, w_2+\epsilon_2, \ldots, w_N) + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1\epsilon_2\frac{\partial}{\partial w_2}\frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N).
\end{aligned}
$$

우리는 $\epsilon_1\epsilon_2$가 :eqref:`eq_part_der`에서 본 것과 함께 이전 섹션에서 $\epsilon^{2}$을 버릴 수있는 것과 같은 방식으로 버릴 수있는 고차 용어라는 생각을 다시 사용했습니다.이런 식으로 계속하면 다음과 같이 쓸 수 있습니다. 

$$
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \approx L(w_1, w_2, \ldots, w_N) + \sum_i \epsilon_i \frac{\partial}{\partial w_i} L(w_1, w_2, \ldots, w_N).
$$

이것은 엉망처럼 보일지 모르지만, 오른쪽의 합이 정확히 내적처럼 보인다는 것을 주목함으로써 이것을 더 친숙하게 만들 수 있습니다. 

$$
\boldsymbol{\epsilon} = [\epsilon_1, \ldots, \epsilon_N]^\top \; \text{and} \;
\nabla_{\mathbf{x}} L = \left[\frac{\partial L}{\partial x_1}, \ldots, \frac{\partial L}{\partial x_N}\right]^\top,
$$

그때 

$$L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).$$
:eqlabel:`eq_nabla_use`

벡터 $\nabla_{\mathbf{w}} L$를 $L$의*그라디언트*라고 부릅니다. 

방정식 :eqref:`eq_nabla_use`는 잠시 숙고해 볼 가치가 있습니다.한 차원에서 접한 형식과 정확히 일치합니다. 모든 것을 벡터와 내적으로 변환했습니다.이를 통해 입력에 대한 섭동이 주어지면 함수 $L$가 어떻게 변할지 대략적으로 알 수 있습니다.다음 섹션에서 볼 수 있듯이 그래디언트에 포함 된 정보를 사용하여 학습하는 방법을 기하학적으로 이해하는 데 중요한 도구를 제공합니다. 

하지만 먼저 예제를 통해 이 근사치를 살펴보겠습니다.함수를 사용하여 작업하고 있다고 가정합니다. 

$$
f(x, y) = \log(e^x + e^y) \text{ with gradient } \nabla f (x, y) = \left[\frac{e^x}{e^x+e^y}, \frac{e^y}{e^x+e^y}\right].
$$

$(0, \log(2))$와 같은 지점을 보면 

$$
f(x, y) = \log(3) \text{ with gradient } \nabla f (x, y) = \left[\frac{1}{3}, \frac{2}{3}\right].
$$

따라서 $(\epsilon_1, \log(2) + \epsilon_2)$에서 $f$를 근사화하려면 :eqref:`eq_nabla_use`의 특정 인스턴스가 있어야 함을 알 수 있습니다. 

$$
f(\epsilon_1, \log(2) + \epsilon_2) \approx \log(3) + \frac{1}{3}\epsilon_1 + \frac{2}{3}\epsilon_2.
$$

이를 코드로 테스트하여 근사치가 얼마나 좋은지 확인할 수 있습니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mpl_toolkits import mplot3d
from mxnet import autograd, np, npx
npx.set_np()

def f(x, y):
    return np.log(np.exp(x) + np.exp(y))
def grad_f(x, y):
    return np.array([np.exp(x) / (np.exp(x) + np.exp(y)),
                     np.exp(y) / (np.exp(x) + np.exp(y))])

epsilon = np.array([0.01, -0.03])
grad_approx = f(0, np.log(2)) + epsilon.dot(grad_f(0, np.log(2)))
true_value = f(0 + epsilon[0], np.log(2) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from mpl_toolkits import mplot3d
import torch
import numpy as np

def f(x, y):
    return torch.log(torch.exp(x) + torch.exp(y))
def grad_f(x, y):
    return torch.tensor([torch.exp(x) / (torch.exp(x) + torch.exp(y)),
                     torch.exp(y) / (torch.exp(x) + torch.exp(y))])

epsilon = torch.tensor([0.01, -0.03])
grad_approx = f(torch.tensor([0.]), torch.log(
    torch.tensor([2.]))) + epsilon.dot(
    grad_f(torch.tensor([0.]), torch.log(torch.tensor(2.))))
true_value = f(torch.tensor([0.]) + epsilon[0], torch.log(
    torch.tensor([2.])) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from mpl_toolkits import mplot3d
import tensorflow as tf
import numpy as np

def f(x, y):
    return tf.math.log(tf.exp(x) + tf.exp(y))
def grad_f(x, y):
    return tf.constant([(tf.exp(x) / (tf.exp(x) + tf.exp(y))).numpy(),
                        (tf.exp(y) / (tf.exp(x) + tf.exp(y))).numpy()])

epsilon = tf.constant([0.01, -0.03])
grad_approx = f(tf.constant([0.]), tf.math.log(
    tf.constant([2.]))) + tf.tensordot(
    epsilon, grad_f(tf.constant([0.]), tf.math.log(tf.constant(2.))), axes=1)
true_value = f(tf.constant([0.]) + epsilon[0], tf.math.log(
    tf.constant([2.])) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

## 그래디언트 지오메트리 및 경사하강법 :eqref:`eq_nabla_use`의 표현식을 다시 생각해 보십시오. 

$$
L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).
$$

손실을 최소화하기 위해 이것을 사용하고 싶다고 가정 해 보겠습니다. $L$.:numref:`sec_autograd`에서 처음 설명한 기울기 하강 알고리즘을 기하학적으로 이해합시다.우리가 할 일은 다음과 같습니다. 

1. 초기 매개변수 $\mathbf{w}$를 무작위로 선택하여 시작합니다.
2. $\mathbf{w}$에서 $L$가 가장 빠르게 감소하게 만드는 방향 $\mathbf{v}$을 찾습니다.
3. 그 방향으로 작은 발걸음을 내딛으십시오: $\mathbf{w} \rightarrow \mathbf{w} + \epsilon\mathbf{v}$.
4. 반복.

정확히 어떻게 해야할지 모르는 유일한 방법은 두 번째 단계에서 벡터 $\mathbf{v}$을 계산하는 것입니다.이러한 방향을*가장 가파른 하강 방향*이라고 부릅니다.:numref:`sec_geometry-linear-algebraic-ops`의 내적의 기하학적 이해를 사용하여 :eqref:`eq_nabla_use`를 다음과 같이 다시 작성할 수 있음을 알 수 있습니다. 

$$
L(\mathbf{w} + \mathbf{v}) \approx L(\mathbf{w}) + \mathbf{v}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}) = L(\mathbf{w}) + \|\nabla_{\mathbf{w}} L(\mathbf{w})\|\cos(\theta).
$$

편의를 위해 길이 1을 갖도록 방향을 잡았고 $\mathbf{v}$과 $\nabla_{\mathbf{w}} L(\mathbf{w})$ 사이의 각도에 $\theta$를 사용했습니다.$L$가 가능한 한 빨리 감소하는 방향을 찾으려면 이 표현식을 가능한 한 음수로 만들고 싶습니다.우리가 선택한 방향이 이 방정식에 들어가는 유일한 방법은 $\cos(\theta)$을 통해서입니다. 따라서 이 코사인을 가능한 한 음수로 만들고자 합니다.이제 코사인의 모양을 상기하면 $\cos(\theta) = -1$을 만들거나 기울기와 선택한 방향 사이의 각도를 $\pi$ 라디안 또는 이에 상응하는 $180$도로 만들어 가능한 한 음수로 만들 수 있습니다.이를 달성하는 유일한 방법은 정확히 반대 방향으로 향하는 것입니다. $\mathbf{v}$을 선택하여 $\nabla_{\mathbf{w}} L(\mathbf{w})$와 정확히 반대 방향을 가리킵니다! 

이를 통해 기계 학습에서 가장 중요한 수학적 개념 중 하나 인 $-\nabla_{\mathbf{w}}L(\mathbf{w})$ 방향으로 가장 가파른 점수의 방향을 알 수 있습니다.따라서 비공식 알고리즘은 다음과 같이 다시 작성할 수 있습니다. 

1. 초기 매개변수 $\mathbf{w}$를 무작위로 선택하여 시작합니다.
2. $\nabla_{\mathbf{w}} L(\mathbf{w})$를 계산합니다.
3. 그 방향과 반대되는 작은 발걸음을 내딛으십시오: $\mathbf{w} \rightarrow \mathbf{w} - \epsilon\nabla_{\mathbf{w}} L(\mathbf{w})$.
4. 반복.

이 기본 알고리즘은 많은 연구자들에 의해 여러 가지 방법으로 수정되고 적용되었지만 핵심 개념은 모두 동일하게 유지됩니다.그래디언트를 사용하여 손실을 최대한 빠르게 줄이는 방향을 찾고 매개변수를 업데이트하여 해당 방향으로 한 걸음 내딛습니다. 

## 수학적 최적화에 대한 참고 사항 이 책에서는 딥러닝 설정에서 접하는 모든 함수가 너무 복잡해서 명시적으로 최소화할 수 없는 실제적인 이유 때문에 수치 최적화 기법에 중점을 두고 있습니다. 

그러나 위에서 얻은 기하학적 이해가 함수를 직접 최적화하는 것에 대해 무엇을 의미하는지 생각해 보는 것은 유용한 연습입니다. 

일부 함수 $L(\mathbf{x})$을 최소화하는 $\mathbf{x}_0$의 값을 찾고 싶다고 가정합니다.또한 누군가가 우리에게 가치를 부여하고 $L$를 최소화하는 가치라고 말한다고 가정 해 봅시다.그들의 대답이 그럴듯한지 확인하기 위해 우리가 확인할 수 있는 것이 있을까요? 

다시 :eqref:`eq_nabla_use`를 고려하십시오: $$ L (\ mathbf {x} _0 +\ 굵은 기호 {\ 엡실론})\ 약 L (\ mathbf {x} _0) +\ 굵은 기호 {\ 엡실론}\ cdot\ nabla_ {\ mathbf {x}} L (\ mathbf {x} _0).$$ 

기울기가 0이 아닌 경우 $-\epsilon \nabla_{\mathbf{x}} L(\mathbf{x}_0)$ 방향으로 한 걸음 더 작은 $L$의 값을 찾을 수 있다는 것을 알고 있습니다.따라서 우리가 진정으로 최소한이라면 그럴 수 없습니다!$\mathbf{x}_0$가 최소값이면 $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$이라는 결론을 내릴 수 있습니다.우리는 $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$*임계점*으로 포인트를 호출합니다. 

드문 설정에서는 그래디언트가 0 인 모든 점을 명시 적으로 찾고 가장 작은 값을 가진 점을 찾을 수 있기 때문에 좋습니다. 

구체적인 예를 들어, 함수 $$ f (x) = 3x^4 - 4x^3 -12x^2를 고려해 보십시오.$$ 

이 함수는 미분 $$\ frac {df} {dx} = 12x^3 - 12x^2 -24x = 12x (x-2) (x+1) 를 갖습니다..$$ 

최소값의 유일한 가능한 위치는 $x = -1, 0, 2$이며, 여기서 함수는 각각 $-5,0, -32$의 값을 취하므로 $x = 2$일 때 함수를 최소화한다는 결론을 내릴 수 있습니다.빠른 플롯으로 확인할 수 있습니다.

```{.python .input}
x = np.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

이것은 이론적으로나 수치적으로 작업 할 때 알아야 할 중요한 사실을 강조합니다. 함수를 최소화 (또는 최대화) 할 수있는 유일한 가능한 점은 기울기가 0이지만 기울기 0이있는 모든 점이 실제*전역* 최소값 (또는 최대값) 인 것은 아닙니다. 

## 다변량 연쇄 규칙 많은 항을 구성하여 만들 수 있는 네 가지 변수 ($w, x, y$ 및 $z$) 의 함수가 있다고 가정해 보겠습니다. 

$$\begin{aligned}f(u, v) & = (u+v)^{2} \\u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.\end{aligned}$$
:eqlabel:`eq_multi_func_def`

이러한 방정식 체인은 신경망으로 작업할 때 일반적이므로 이러한 함수의 기울기를 계산하는 방법을 이해하는 것이 중요합니다.어떤 변수가 서로 직접적으로 관련되어 있는지 살펴보면 :numref:`fig_chain-1`에서 이 연결에 대한 시각적 힌트를 볼 수 있습니다. 

![The function relations above where nodes represent values and edges show functional dependence.](../img/chain-net1.svg)
:label:`fig_chain-1`

:eqref:`eq_multi_func_def`의 모든 것을 작곡하고 그것을 쓰는 것을 막는 것은 없습니다. 

$$
f(w, x, y, z) = \left(\left((w+x+y+z)^2+(w+x-y-z)^2\right)^2+\left((w+x+y+z)^2-(w+x-y-z)^2\right)^2\right)^2.
$$

그런 다음 단일 변수 도함수를 사용하여 미분을 취할 수 있지만, 그렇게하면 항으로 가득 차게 될 것입니다. 그 중 많은 부분이 반복됩니다!실제로, 예를 들어 다음과 같은 것을 볼 수 있습니다. 

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = 2 \left(2 \left(2 (w + x + y + z) - 2 (w + x - y - z)\right) \left((w + x + y + z)^{2}- (w + x - y - z)^{2}\right) + \right.\\
& \left. \quad 2 \left(2 (w + x - y - z) + 2 (w + x + y + z)\right) \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)\right) \times \\
& \quad \left(\left((w + x + y + z)^{2}- (w + x - y - z)^2\right)^{2}+ \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)^{2}\right).
\end{aligned}
$$

그런 다음 $\frac{\partial f}{\partial x}$도 계산하려면 반복되는 항이 많고 두 도함수 사이에 많은*공유* 반복 항으로 다시 유사한 방정식이 생깁니다.이것은 엄청난 양의 낭비되는 작업을 나타내며, 이러한 방식으로 도함수를 계산해야 한다면 딥 러닝 혁명 전체가 시작되기 전에 멈췄을 것입니다! 

문제를 해결해 보겠습니다.우리는 $a$을 변경할 때 $f$이 어떻게 변하는지 이해하려고 노력하는 것으로 시작하겠습니다. 본질적으로 $w, x, y$와 $z$가 모두 존재하지 않는다고 가정합니다.그래디언트로 처음 작업했을 때와 마찬가지로 추론할 것입니다.$a$을 가져 와서 소량의 $\epsilon$을 추가합시다. 

$$
\begin{aligned}
& f(u(a+\epsilon, b), v(a+\epsilon, b)) \\
\approx & f\left(u(a, b) + \epsilon\frac{\partial u}{\partial a}(a, b), v(a, b) + \epsilon\frac{\partial v}{\partial a}(a, b)\right) \\
\approx & f(u(a, b), v(a, b)) + \epsilon\left[\frac{\partial f}{\partial u}(u(a, b), v(a, b))\frac{\partial u}{\partial a}(a, b) + \frac{\partial f}{\partial v}(u(a, b), v(a, b))\frac{\partial v}{\partial a}(a, b)\right].
\end{aligned}
$$

첫 번째 줄은 편미분의 정의에서 따르고 두 번째 줄은 기울기의 정의에서 따릅니다.$\frac{\partial f}{\partial u}(u(a, b), v(a, b))$ 표현에서와 같이 모든 파생물을 평가하는 위치를 정확히 추적하는 것은 표기적으로 부담스럽기 때문에 종종 이것을 훨씬 더 기억에 남는 것으로 축약합니다. 

$$
\frac{\partial f}{\partial a} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}.
$$

프로세스의 의미를 생각해 보는 것이 유용합니다.우리는 $f(u(a, b), v(a, b))$ 양식의 함수가 $a$의 변경에 따라 그 값을 어떻게 변경하는지 이해하려고 노력하고 있습니다.이러한 경로가 발생할 수 있는 두 가지 경로가 있습니다. 즉, $a \rightarrow u \rightarrow f$과 $a \rightarrow v \rightarrow f$의 경로가 있습니다.체인 규칙 (각각 $\frac{\partial w}{\partial u} \cdot \frac{\partial u}{\partial x}$과 $\frac{\partial w}{\partial v} \cdot \frac{\partial v}{\partial x}$) 을 통해 이러한 기여도를 모두 계산할 수 있으며 합산됩니다. 

:numref:`fig_chain-2`에서 볼 수 있듯이 오른쪽의 기능이 왼쪽에 연결된 기능에 따라 달라지는 다른 기능 네트워크가 있다고 상상해보십시오. 

![Another more subtle example of the chain rule.](../img/chain-net2.svg)
:label:`fig_chain-2`

$\frac{\partial f}{\partial y}$과 같은 것을 계산하려면 $y$에서 $f$까지의 모든 경로 (이 경우 $3$) 를 합산해야합니다. 

$$
\frac{\partial f}{\partial y} = \frac{\partial f}{\partial a} \frac{\partial a}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial b} \frac{\partial b}{\partial v} \frac{\partial v}{\partial y}.
$$

이러한 방식으로 연쇄 규칙을 이해하면 그래디언트가 네트워크를 통해 흐르는 방식과 LSTM (:numref:`sec_lstm`) 또는 잔차 계층 (:numref:`sec_resnet`) 과 같은 다양한 아키텍처 선택이 기울기 흐름을 제어하여 학습 과정을 형성하는 데 도움이 되는 이유를 이해하려고 할 때 큰 이점을 얻을 수 있습니다. 

## 역전파 알고리즘

이전 섹션의 :eqref:`eq_multi_func_def`의 예제로 돌아가 보겠습니다. 

$$
\begin{aligned}
f(u, v) & = (u+v)^{2} \\
u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\
a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.
\end{aligned}
$$

$\frac{\partial f}{\partial w}$를 계산하려면 다변량 체인 규칙을 적용하여 다음을 확인할 수 있습니다. 

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial w} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial w}, \\
\frac{\partial u}{\partial w} & = \frac{\partial u}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial u}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial v}{\partial w} & = \frac{\partial v}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial v}{\partial b}\frac{\partial b}{\partial w}.
\end{aligned}
$$

이 분해를 사용하여 $\frac{\partial f}{\partial w}$를 계산해 보겠습니다.여기에 필요한 것은 다양한 단일 단계 부분뿐입니다. 

$$
\begin{aligned}
\frac{\partial f}{\partial u} = 2(u+v), & \quad\frac{\partial f}{\partial v} = 2(u+v), \\
\frac{\partial u}{\partial a} = 2(a+b), & \quad\frac{\partial u}{\partial b} = 2(a+b), \\
\frac{\partial v}{\partial a} = 2(a-b), & \quad\frac{\partial v}{\partial b} = -2(a-b), \\
\frac{\partial a}{\partial w} = 2(w+x+y+z), & \quad\frac{\partial b}{\partial w} = 2(w+x-y-z).
\end{aligned}
$$

이것을 코드에 작성하면 상당히 관리하기 쉬운 표현식이 됩니다.

```{.python .input}
#@tab all
# Compute the value of the function from inputs to outputs
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'    f at {w}, {x}, {y}, {z} is {f}')

# Compute the single step partials
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)

# Compute the final result from inputs to outputs
du_dw, dv_dw = du_da*da_dw + du_db*db_dw, dv_da*da_dw + dv_db*db_dw
df_dw = df_du*du_dw + df_dv*dv_dw
print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
```

그러나 여전히 $\frac{\partial f}{\partial x}$와 같은 것을 쉽게 계산할 수는 없습니다.그 이유는 연쇄 규칙을 적용하기로 선택한*방법* 때문입니다.위에서 한 일을 살펴보면 가능할 때 항상 분모에 $\partial w$를 유지했습니다.이런 식으로 $w$이 다른 모든 변수를 어떻게 변경했는지 확인하는 연쇄 규칙을 적용하기로 결정했습니다.그게 우리가 원했던 것이라면 좋은 생각이 될 것입니다.그러나 딥 러닝의 동기를 다시 생각해 보십시오. 모든 매개 변수가*손실*을 어떻게 변화시키는지 확인하고자 합니다.본질적으로 우리는 가능할 때마다 분자에 $\partial f$을 유지하는 연쇄 규칙을 적용하고 싶습니다! 

좀 더 명확하게 말하면 다음과 같이 작성할 수 있습니다. 

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial w} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial f}{\partial a} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}, \\
\frac{\partial f}{\partial b} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial b}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial b}.
\end{aligned}
$$

이 체인 규칙을 적용하면 $\frac{\partial f}{\partial u}, \frac{\partial f}{\partial v}, \frac{\partial f}{\partial a}, \frac{\partial f}{\partial b}, \; \text{and} \; \frac{\partial f}{\partial w}$를 명시적으로 계산하게 됩니다.방정식을 포함시키는 것을 막을 수있는 것은 없습니다. 

$$
\begin{aligned}
\frac{\partial f}{\partial x} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial x} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial x}, \\
\frac{\partial f}{\partial y} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial y}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial y}, \\
\frac{\partial f}{\partial z} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial z}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial z}.
\end{aligned}
$$

그런 다음 전체 네트워크에서*any* 노드를 변경할 때 $f$가 어떻게 변경되는지 추적합니다.구현해 보겠습니다.

```{.python .input}
#@tab all
# Compute the value of the function from inputs to outputs
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'f at {w}, {x}, {y}, {z} is {f}')

# Compute the derivative using the decomposition above
# First compute the single step partials
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)
da_dx, db_dx = 2*(w + x + y + z), 2*(w + x - y - z)
da_dy, db_dy = 2*(w + x + y + z), -2*(w + x - y - z)
da_dz, db_dz = 2*(w + x + y + z), -2*(w + x - y - z)

# Now compute how f changes when we change any value from output to input
df_da, df_db = df_du*du_da + df_dv*dv_da, df_du*du_db + df_dv*dv_db
df_dw, df_dx = df_da*da_dw + df_db*db_dw, df_da*da_dx + df_db*db_dx
df_dy, df_dz = df_da*da_dy + df_db*db_dy, df_da*da_dz + df_db*db_dz

print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
print(f'df/dx at {w}, {x}, {y}, {z} is {df_dx}')
print(f'df/dy at {w}, {x}, {y}, {z} is {df_dy}')
print(f'df/dz at {w}, {x}, {y}, {z} is {df_dz}')
```

(위의 첫 번째 코드 스 니펫에서했던 것처럼) 입력에서 출력으로 전달되는 것이 아니라 입력을 향해 $f$에서 도함수를 다시 계산한다는 사실은이 알고리즘의 이름을*backpropagation*으로 부여합니다.다음 두 단계가 있습니다.
1. 함수의 값과 앞에서 뒤로 한 단계 부분을 계산합니다.위에서 수행하지는 않았지만, 이를 단일*순방향 패스*로 결합할 수 있습니다.
2. 뒤쪽에서 앞쪽으로 $f$의 기울기를 계산합니다.우리는*역방향 패스*라고 부릅니다.

이것이 바로 모든 딥러닝 알고리즘이 구현하여 네트워크의 모든 가중치에 대한 손실의 기울기를 한 번에 계산할 수 있도록 하는 것입니다.우리가 그런 분해를 겪고 있다는 것은 놀라운 사실입니다. 

이를 캡슐화하는 방법을 보려면 이 예제를 간단히 살펴보겠습니다.

```{.python .input}
# Initialize as ndarrays, then attach gradients
w, x, y, z = np.array(-1), np.array(0), np.array(-2), np.array(1)

w.attach_grad()
x.attach_grad()
y.attach_grad()
z.attach_grad()

# Do the computation like usual, tracking gradients
with autograd.record():
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Execute backward pass
f.backward()

print(f'df/dw at {w}, {x}, {y}, {z} is {w.grad}')
print(f'df/dx at {w}, {x}, {y}, {z} is {x.grad}')
print(f'df/dy at {w}, {x}, {y}, {z} is {y.grad}')
print(f'df/dz at {w}, {x}, {y}, {z} is {z.grad}')
```

```{.python .input}
#@tab pytorch
# Initialize as ndarrays, then attach gradients
w = torch.tensor([-1.], requires_grad=True)
x = torch.tensor([0.], requires_grad=True)
y = torch.tensor([-2.], requires_grad=True)
z = torch.tensor([1.], requires_grad=True)
# Do the computation like usual, tracking gradients
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2

# Execute backward pass
f.backward()

print(f'df/dw at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {w.grad.data.item()}')
print(f'df/dx at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {x.grad.data.item()}')
print(f'df/dy at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {y.grad.data.item()}')
print(f'df/dz at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {z.grad.data.item()}')
```

```{.python .input}
#@tab tensorflow
# Initialize as ndarrays, then attach gradients
w = tf.Variable(tf.constant([-1.]))
x = tf.Variable(tf.constant([0.]))
y = tf.Variable(tf.constant([-2.]))
z = tf.Variable(tf.constant([1.]))
# Do the computation like usual, tracking gradients
with tf.GradientTape(persistent=True) as t:
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Execute backward pass
w_grad = t.gradient(f, w).numpy()
x_grad = t.gradient(f, x).numpy()
y_grad = t.gradient(f, y).numpy()
z_grad = t.gradient(f, z).numpy()

print(f'df/dw at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {w_grad}')
print(f'df/dx at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {x_grad}')
print(f'df/dy at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {y_grad}')
print(f'df/dz at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {z_grad}')
```

위에서 수행한 모든 작업은 `f.backwards()`로 전화하여 자동으로 수행할 수 있습니다. 

## 헤세 단일 변수 미적분학과 마찬가지로 기울기를 단독으로 사용하는 것보다 함수에 대한 더 나은 근사를 얻는 방법에 대한 핸들을 얻으려면 고차 도함수를 고려하는 것이 유용합니다. 

여러 변수의 함수의 고차 도함수를 사용할 때 발생하는 즉각적인 문제가 하나 있는데, 그 중 많은 수가 있다는 것입니다.$n$ 변수의 함수 $f(x_1, \ldots, x_n)$가 있다면 $n^{2}$의 많은 두 번째 도함수를 사용할 수 있습니다. 즉, $i$ 및 $j$ 중 하나를 선택할 수 있습니다. 

$$
\frac{d^2f}{dx_idx_j} = \frac{d}{dx_i}\left(\frac{d}{dx_j}f\right).
$$

이것은 전통적으로 *Hessian*이라는 행렬로 조립됩니다. 

$$\mathbf{H}_f = \begin{bmatrix} \frac{d^2f}{dx_1dx_1} & \cdots & \frac{d^2f}{dx_1dx_n} \\ \vdots & \ddots & \vdots \\ \frac{d^2f}{dx_ndx_1} & \cdots & \frac{d^2f}{dx_ndx_n} \\ \end{bmatrix}.$$
:eqlabel:`eq_hess_def`

이 행렬의 모든 항목이 독립적이지는 않습니다.실제로*혼합 부분* (둘 이상의 변수에 대한 편미분) 이 모두 존재하고 연속적이라면 $i$ 및 $j$에 대해 그렇게 말할 수 있습니다. 

$$
\frac{d^2f}{dx_idx_j} = \frac{d^2f}{dx_jdx_i}.
$$

이것은 먼저 $x_i$ 방향으로 함수를 교란시킨 다음 $x_j$에서 교란시킨 다음 그 결과를 우리가 처음 $x_j$와 $x_i$를 교란하면 일어나는 일과 비교하는 것입니다. 이 두 주문이 모두 동일한 최종 변화로 이어진다는 지식과 비교합니다.$f$의 출력이 출력됩니다. 

단일 변수와 마찬가지로 이러한 도함수를 사용하여 함수가 한 지점 근처에서 어떻게 동작하는지 훨씬 더 잘 알 수 있습니다.특히 단일 변수에서 보았 듯이 점 $\mathbf{x}_0$ 근처에서 가장 적합한 2 차를 찾는 데 사용할 수 있습니다. 

예를 들어 보겠습니다.$f(x_1, x_2) = a + b_1x_1 + b_2x_2 + c_{11}x_1^{2} + c_{12}x_1x_2 + c_{22}x_2^{2}$라고 가정해 보겠습니다.이것은 두 변수의 2차 변수에 대한 일반적인 형식입니다.함수의 값, 기울기 및 헤세 행렬 :eqref:`eq_hess_def`를 모두 점 0에서 보면 다음과 같습니다. 

$$
\begin{aligned}
f(0,0) & = a, \\
\nabla f (0,0) & = \begin{bmatrix}b_1 \\ b_2\end{bmatrix}, \\
\mathbf{H} f (0,0) & = \begin{bmatrix}2 c_{11} & c_{12} \\ c_{12} & 2c_{22}\end{bmatrix},
\end{aligned}
$$

우리는 원래 다항식을 다시 얻을 수 있습니다. 

$$
f(\mathbf{x}) = f(0) + \nabla f (0) \cdot \mathbf{x} + \frac{1}{2}\mathbf{x}^\top \mathbf{H} f (0) \mathbf{x}.
$$

일반적으로 이 확장을 $\mathbf{x}_0$ 지점으로 계산하면 

$$
f(\mathbf{x}) = f(\mathbf{x}_0) + \nabla f (\mathbf{x}_0) \cdot (\mathbf{x}-\mathbf{x}_0) + \frac{1}{2}(\mathbf{x}-\mathbf{x}_0)^\top \mathbf{H} f (\mathbf{x}_0) (\mathbf{x}-\mathbf{x}_0).
$$

이는 모든 차원 입력에 대해 작동하며 한 점의 모든 함수에 가장 적합한 2차 근삿값을 제공합니다.예제를 제공하기 위해 함수를 플로팅해 보겠습니다. 

$$
f(x, y) = xe^{-x^2-y^2}.
$$

기울기와 헤세 행렬은 $$\ nabla f (x, y) = e^ {-x^2-y^2}\ 시작 {pmatrix} 1-2x^2\\ -2xy\ 끝 {pmatrix}\;\ 텍스트 {및}\;\ mathbf {H} f (x, y) = e^ {-x^2-y^2}\ 시작 {pmatrix} 4x^3 - 6x & 4x^2y - 2년\\ 4x^2y-2y &4xy^2-2x\ 끝 {p매트릭스}.$$ 

따라서 약간의 대수학을 사용하면 $[-1,0]^\top$에서 근사한 2차 방정식이 다음과 같음을 알 수 있습니다. 

$$
f(x, y) \approx e^{-1}\left(-1 - (x+1) +(x+1)^2+y^2\right).
$$

```{.python .input}
# Construct grid and compute function
x, y = np.meshgrid(np.linspace(-2, 2, 101),
                   np.linspace(-2, 2, 101), indexing='ij')
z = x*np.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = np.exp(-1)*(-1 - (x + 1) + (x + 1)**2 + y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x, y, w, **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab pytorch
# Construct grid and compute function
x, y = torch.meshgrid(torch.linspace(-2, 2, 101),
                   torch.linspace(-2, 2, 101))

z = x*torch.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = torch.exp(torch.tensor([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab tensorflow
# Construct grid and compute function
x, y = tf.meshgrid(tf.linspace(-2., 2., 101),
                   tf.linspace(-2., 2., 101))

z = x*tf.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = tf.exp(tf.constant([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

이것은 :numref:`sec_gd`에서 논의한 Newton 알고리즘의 기초를 형성합니다. 여기서 수치 최적화를 수행하여 반복적으로 가장 적합한 2 차를 찾은 다음 그 2 차를 정확히 최소화합니다. 

## 행렬을 포함하는 함수의 미적분 미적분 도함수가 특히 좋습니다.이 섹션은 표기법으로 무거워 질 수 있으므로 첫 번째 읽기에서 건너 뛸 수 있지만, 특히 중심 행렬 연산이 딥 러닝 응용 프로그램에 얼마나 중요한지 고려할 때 일반적인 행렬 연산과 관련된 함수의 도함수가 처음에 예상했던 것보다 훨씬 깨끗하다는 것을 아는 것이 유용합니다. 

예를 들어 시작하겠습니다.고정 열 벡터 $\boldsymbol{\beta}$가 있고 곱 함수 $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$을 사용하여 $\mathbf{x}$를 변경할 때 내적이 어떻게 변하는지 이해하려고 한다고 가정합니다. 

ML에서 행렬 도함수를 사용하여 작업 할 때 유용한 약간의 표기법을*분모 레이아웃 행렬 도함수*라고합니다. 여기서 부분 도함수를 미분의 분모에있는 벡터, 행렬 또는 텐서의 모양으로 조립합니다.이 경우 다음과 같이 작성합니다. 

$$
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix},
$$

여기서 열 벡터 $\mathbf{x}$의 모양을 일치시켰습니다. 

함수를 컴포넌트에 작성하면 

$$
f(\mathbf{x}) = \sum_{i = 1}^{n} \beta_ix_i = \beta_1x_1 + \cdots + \beta_nx_n.
$$

이제 $\beta_1$와 관련하여 편미분을 취하면 모든 것이 0이지만 첫 번째 항 ($x_1$에 $\beta_1$를 곱한 값) 에 유의하십시오. 

$$
\frac{df}{dx_1} = \beta_1,
$$

또는 더 일반적으로 

$$
\frac{df}{dx_i} = \beta_i.
$$

이제 이것을 행렬로 재조합해서 

$$
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix} = \begin{bmatrix}
\beta_1 \\
\vdots \\
\beta_n
\end{bmatrix} = \boldsymbol{\beta}.
$$

이것은 행렬 미적분에 대한 몇 가지 요인을 보여 주며, 이 섹션 전체에서 자주 다루게 될 것입니다. 

* 첫째, 계산이 다소 복잡해질 것입니다.
* 둘째, 최종 결과는 중간 프로세스보다 훨씬 깨끗하며 항상 단일 변수 사례와 유사하게 보입니다.이 경우 $\frac{d}{dx}(bx) = b$와 $\frac{d}{d\mathbf{x}} (\boldsymbol{\beta}^\top\mathbf{x}) = \boldsymbol{\beta}$가 모두 유사하다는 점에 유의하십시오.
* 셋째, 조옮김은 종종 아무데도 나타나지 않을 수 있습니다.이것의 핵심 이유는 분모의 모양과 일치하는 규칙입니다. 따라서 행렬을 곱할 때 원래 항의 모양과 다시 일치하도록 전치를 가져와야합니다.

직관을 계속 구축하기 위해 조금 더 어려운 계산을 시도해 보겠습니다.열 벡터 $\mathbf{x}$와 정사각 행렬 $A$가 있고 다음과 같이 계산하려고 한다고 가정합니다. 

$$\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}).$$
:eqlabel:`eq_mat_goal_1`

표기법을 더 쉽게 조작하기 위해 아인슈타인 표기법을 사용하여이 문제를 고려해 보겠습니다.이 경우 함수를 다음과 같이 작성할 수 있습니다. 

$$
\mathbf{x}^\top A \mathbf{x} = x_ia_{ij}x_j.
$$

미분을 계산하려면 $k$마다 값이 무엇인지 이해해야 합니다. 

$$
\frac{d}{dx_k}(\mathbf{x}^\top A \mathbf{x}) = \frac{d}{dx_k}x_ia_{ij}x_j.
$$

제품 규칙에 따라 다음과 같습니다. 

$$
\frac{d}{dx_k}x_ia_{ij}x_j = \frac{dx_i}{dx_k}a_{ij}x_j + x_ia_{ij}\frac{dx_j}{dx_k}.
$$

$\frac{dx_i}{dx_k}$과 같은 용어의 경우 $i=k$ 일 때 이것이 하나이고 그렇지 않으면 0이라는 것을 알기가 어렵지 않습니다.즉, $i$와 $k$가 다른 모든 항이 이 합계에서 사라지므로 첫 번째 합계에 남아 있는 유일한 항은 $i=k$입니다.$j=k$이 필요한 두 번째 용어에도 동일한 추론이 적용됩니다.이것은 준다 

$$
\frac{d}{dx_k}x_ia_{ij}x_j = a_{kj}x_j + x_ia_{ik}.
$$

이제 아인슈타인 표기법에서 인덱스의 이름은 임의적입니다. $i$와 $j$가 다르다는 사실은 이 시점에서 이 계산에 중요하지 않습니다. 그래서 우리는 재색인을 만들어서 둘 다 $i$를 사용하여 

$$
\frac{d}{dx_k}x_ia_{ij}x_j = a_{ki}x_i + x_ia_{ik} = (a_{ki} + a_{ik})x_i.
$$

이제 더 나아가기 위해 몇 가지 연습이 필요한 곳이 있습니다.행렬 연산의 관점에서 이 결과를 확인해 보겠습니다. $a_{ki} + a_{ik}$는 $\mathbf{A} + \mathbf{A}^\top$의 $k, i$번째 구성 요소입니다.이것은 준다 

$$
\frac{d}{dx_k}x_ia_{ij}x_j = [\mathbf{A} + \mathbf{A}^\top]_{ki}x_i.
$$

마찬가지로, 이 항은 이제 벡터 $\mathbf{x}$에 의한 행렬 $\mathbf{A} + \mathbf{A}^\top$의 곱이므로 

$$
\left[\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x})\right]_k = \frac{d}{dx_k}x_ia_{ij}x_j = [(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}]_k.
$$

따라서 :eqref:`eq_mat_goal_1`에서 원하는 도함수의 $k$ 번째 항목은 오른쪽 벡터의 $k$ 번째 항목에 불과하므로 두 항목이 동일하다는 것을 알 수 있습니다.따라서 수익률 

$$
\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}.
$$

이를 위해서는 마지막 작업보다 훨씬 많은 작업이 필요했지만 최종 결과는 작습니다.그 외에도 기존의 단일 변수 도함수에 대해 다음과 같은 계산을 고려하십시오. 

$$
\frac{d}{dx}(xax) = \frac{dx}{dx}ax + xa\frac{dx}{dx} = (a+a)x.
$$

동등하게 $\frac{d}{dx}(ax^2) = 2ax = (a+a)x$입니다.다시 말하지만, 단일 변수 결과처럼 보이지만 조옮김이 던져진 결과를 얻습니다. 

이 시점에서 패턴은 다소 의심스러워 보일 것이므로 그 이유를 알아 보겠습니다.이와 같은 행렬 도함수를 취할 때, 먼저 우리가 얻는 표현식이 또 다른 행렬 표현식이라고 가정해 봅시다. 행렬의 곱과 합과 전치 (transposes) 로 쓸 수 있는 표현식입니다.이러한 표현식이 존재하면 모든 행렬에 대해 true여야 합니다.특히 $1 \times 1$ 행렬에 해당해야 합니다. 이 경우 행렬 곱은 숫자의 곱이고 행렬 합은 합에 불과하며 전치는 전혀 수행하지 않습니다!즉, 어떤 표현식이든*must*는 단일 변수 표현식과 일치해야 합니다.즉, 어떤 연습에서는 연관된 단일 변수 표현식이 어떻게 생겼는지 아는 것만으로도 행렬 도함수를 추측할 수 있습니다! 

이것을 시도해 보겠습니다.$\mathbf{X}$가 $n \times m$ 행렬이고, $\mathbf{U}$이 $n \times r$이고, $\mathbf{V}$이 $r \times m$이라고 가정합니다.우리가 계산을 해보자. 

$$\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2} = \;?$$
:eqlabel:`eq_mat_goal_2`

이 계산은 행렬 분해라는 영역에서 중요합니다.그러나 우리에게는 계산해야 할 파생물에 불과합니다.$1\times1$ 행렬에 대해 이것이 무엇인지 이미징해 보겠습니다.이 경우 다음과 같은 표현식을 얻습니다. 

$$
\frac{d}{dv} (x-uv)^{2}= -2(x-uv)u,
$$

여기서 미분은 다소 표준입니다.이것을 다시 행렬 표현식으로 변환하려고 하면 

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2(\mathbf{X} - \mathbf{U}\mathbf{V})\mathbf{U}.
$$

그러나 이것을 보면 제대로 작동하지 않습니다.$\mathbf{X}$는 $n \times m$이고 $\mathbf{U}\mathbf{V}$도 마찬가지입니다. 따라서 행렬 $2(\mathbf{X} - \mathbf{U}\mathbf{V})$은 $n \times m$입니다.반면에 $\mathbf{U}$은 $n \times r$이며, 치수가 일치하지 않기 때문에 $n \times m$와 $n \times r$ 행렬을 곱할 수 없습니다! 

우리는 $\frac{d}{d\mathbf{V}}$를 얻고 싶습니다. $\frac{d}{d\mathbf{V}}$는 $\mathbf{V}$과 같은 모양입니다.그래서 어떻게 든 우리는 $n \times m$ 행렬과 $n \times r$ 행렬을 가져와 $r \times m$을 얻기 위해 함께 곱해야합니다 (아마도 일부 전치와 함께).$U^\top$에 $(\mathbf{X} - \mathbf{U}\mathbf{V})$을 곱하여 이 작업을 수행할 수 있습니다.따라서 :eqref:`eq_mat_goal_2`에 대한 해결책은 다음과 같다고 추측 할 수 있습니다. 

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$

이것이 효과가 있음을 보여주기 위해 자세한 계산을 제공하지 않는 것이 좋습니다.이 경험 법칙이 효과가 있다고 이미 믿는다면, 이 파생을 건너 뛰어도 좋습니다.계산하려면 

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^2,
$$

우리는 모든 $a$와 $b$에 대해 찾아야 합니다. 

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \frac{d}{dv_{ab}} \sum_{i, j}\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)^2.
$$

$\mathbf{X}$ 및 $\mathbf{U}$의 모든 항목은 $\frac{d}{dv_{ab}}$에 관한 한 상수라는 것을 상기하면 도함수를 합 안에 밀어 넣고 연쇄 규칙을 제곱에 적용하여 

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \sum_{i, j}2\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)\left(-\sum_k u_{ik}\frac{dv_{kj}}{dv_{ab}} \right).
$$

이전 파생에서와 마찬가지로 $\frac{dv_{kj}}{dv_{ab}}$은 $k=a$ 및 $j=b$의 경우 0이 아니라는 것을 알 수 있습니다.이러한 조건 중 하나가 유지되지 않으면 합계의 항이 0이므로 자유롭게 버릴 수 있습니다.우리는 그것을 봅니다 

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}\left(x_{ib} - \sum_k u_{ik}v_{kb}\right)u_{ia}.
$$

여기서 중요한 미묘함은 $k=a$가 내부 합 내에서 발생하지 않는다는 요구 사항이 $k$가 내부 항 내부에서 합산하는 더미 변수이기 때문입니다.표기법으로 더 명확한 예를 보려면 이유를 고려하십시오. 

$$
\frac{d}{dx_1} \left(\sum_i x_i \right)^{2}= 2\left(\sum_i x_i \right).
$$

이 시점부터 합계의 구성 요소를 식별하기 시작할 수 있습니다.먼저, 

$$
\sum_k u_{ik}v_{kb} = [\mathbf{U}\mathbf{V}]_{ib}.
$$

따라서 합계 안에 있는 전체 표현식은 다음과 같습니다. 

$$
x_{ib} - \sum_k u_{ik}v_{kb} = [\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$

즉, 이제 파생물을 다음과 같이 쓸 수 있습니다. 

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}u_{ia}.
$$

행렬의 $a, b$ 요소처럼 보이기를 원하므로 이전 예제와 같은 기법을 사용하여 행렬 표현식에 도달할 수 있습니다. 즉, $u_{ia}$에서 인덱스의 순서를 교환해야 합니다.$u_{ia} = [\mathbf{U}^\top]_{ai}$를 발견하면 다음과 같이 쓸 수 있습니다. 

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i} [\mathbf{U}^\top]_{ai}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$

이것은 행렬 곱이므로 다음과 같은 결론을 내릴 수 있습니다. 

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2[\mathbf{U}^\top(\mathbf{X}-\mathbf{U}\mathbf{V})]_{ab}.
$$

따라서 우리는 :eqref:`eq_mat_goal_2`에 대한 해결책을 쓸 수 있습니다. 

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$

이것은 위에서 추측한 해결책과 일치합니다! 

이 시점에서 “내가 배운 모든 미적분 규칙의 행렬 버전만 적어 둘 수 없는 이유는 무엇입니까?이것이 여전히 기계적이라는 것은 분명합니다.왜 우리는 그걸 극복하지 않을까요!”실제로 그러한 규칙이 있으며 :cite:`Petersen.Pedersen.ea.2008`는 훌륭한 요약을 제공합니다.그러나 단일 값에 비해 행렬 연산을 결합할 수 있는 방법이 너무 많기 때문에 단일 변수 규칙보다 행렬 도함수 규칙이 더 많습니다.인덱스로 작업하거나 적절한 경우 자동 미분 상태로 두는 것이 가장 좋은 경우가 많습니다. 

## 요약

* 더 높은 차원에서는 한 차원의 도함수와 동일한 목적을 제공하는 그라디언트를 정의할 수 있습니다.이를 통해 입력값을 임의로 약간 변경할 때 다중 변수 함수가 어떻게 변하는지 확인할 수 있습니다.
* 역전파 알고리즘은 많은 부분 도함수를 효율적으로 계산할 수 있도록 다중 변수 체인 규칙을 구성하는 방법으로 볼 수 있습니다.
* 행렬 미적분을 사용하면 행렬 표현식의 도함수를 간결하게 작성할 수 있습니다.

## 연습 문제 1.열 벡터 $\boldsymbol{\beta}$이 주어지면 $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$과 $g(\mathbf{x}) = \mathbf{x}^\top\boldsymbol{\beta}$의 도함수를 모두 계산합니다.왜 같은 답을 얻나요?2.$\mathbf{v}$를 $n$ 차원 벡터로 지정합니다.$\frac{\partial}{\partial\mathbf{v}}\|\mathbf{v}\|_2$이란 무엇입니까?3.$L(x, y) = \log(e^x + e^y)$를 보자.그래디언트를 계산합니다.그래디언트 구성 요소의 합은 얼마입니까?4.$f(x, y) = x^2y + xy^2$을 보자.유일한 임계점이 $(0,0)$임을 보여줍니다.$f(x, x)$을 고려하여 $(0,0)$가 최대값인지, 최소값인지 또는 둘 다 아닌지 확인합니다.함수 $f(\mathbf{x}) = g(\mathbf{x}) + h(\mathbf{x})$을 최소화한다고 가정합니다.$g$과 $h$의 관점에서 $\nabla f = 0$의 조건을 어떻게 기하학적으로 해석할 수 있을까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/413)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1090)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1091)
:end_tab:
