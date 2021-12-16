# 볼록
:label:`sec_convexity`

볼록성은 최적화 알고리즘의 설계에서 중요한 역할을 합니다.이는 주로 이러한 맥락에서 알고리즘을 분석하고 테스트하는 것이 훨씬 쉽다는 사실 때문입니다.즉, 알고리즘이 볼록 설정에서도 성능이 좋지 않은 경우 일반적으로 그렇지 않으면 좋은 결과를 볼 수 없기를 바랍니다.또한 딥러닝의 최적화 문제는 일반적으로 볼록하지 않지만 국소 최솟값 근처에서 볼록 문제의 일부 특성을 나타내는 경우가 많습니다.이로 인해 :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`와 같은 흥미로운 새 최적화 변형이 생길 수 있습니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

## 정의

볼록 분석을 하기 전에*볼록 집합*과*볼록 함수*를 정의해야 합니다.기계 학습에 일반적으로 적용되는 수학적 도구로 이어집니다. 

### 볼록 세트

집합은 볼록성의 기초입니다.간단히 말해서, 벡터 공간의 세트 $\mathcal{X}$은 $a, b \in \mathcal{X}$에 대해 $a$과 $b$을 연결하는 선 세그먼트도 $\mathcal{X}$에 있는 경우*볼록*입니다.수학적 용어로 이것은 모든 $\lambda \in [0, 1]$에 대해 

$$\lambda  a + (1-\lambda)  b \in \mathcal{X} \text{ whenever } a, b \in \mathcal{X}.$$

약간 추상적으로 들립니다.:numref:`fig_pacman`를 고려해 보십시오.첫 번째 세트는 포함되지 않은 선 세그먼트가 있으므로 볼록하지 않습니다.다른 두 세트는 그런 문제가 없습니다. 

![The first set is nonconvex and the other two are convex.](../img/pacman.svg)
:label:`fig_pacman`

정의 자체로는 무언가를 할 수 없다면 특별히 유용하지 않습니다.이 경우 :numref:`fig_convex_intersect`와 같이 교차로를 볼 수 있습니다.$\mathcal{X}$ 및 $\mathcal{Y}$이 볼록 세트라고 가정합니다.그런 다음 $\mathcal{X} \cap \mathcal{Y}$도 볼록합니다.이를 확인하려면 $a, b \in \mathcal{X} \cap \mathcal{Y}$를 고려하십시오.$\mathcal{X}$와 $\mathcal{Y}$은 볼록하기 때문에 $a$과 $b$을 연결하는 선 세그먼트는 $\mathcal{X}$와 $\mathcal{Y}$에 모두 포함되어 있습니다.그 점을 감안할 때, 그것들은 또한 $\mathcal{X} \cap \mathcal{Y}$에 포함되어야하므로 우리의 정리를 증명해야합니다. 

![The intersection between two convex sets is convex.](../img/convex-intersect.svg)
:label:`fig_convex_intersect`

적은 노력으로이 결과를 강화할 수 있습니다. 볼록 세트 $\mathcal{X}_i$이 주어지면 교차점 $\cap_{i} \mathcal{X}_i$은 볼록합니다.그 반대가 사실이 아님을 확인하려면 두 개의 분리된 집합 $\mathcal{X} \cap \mathcal{Y} = \emptyset$을 고려하십시오.이제 $a \in \mathcal{X}$와 $b \in \mathcal{Y}$를 선택합니다.$a$과 $b$를 연결하는 :numref:`fig_nonconvex`의 선 세그먼트에는 $\mathcal{X}$이나 $\mathcal{Y}$에 없는 일부가 포함되어야 합니다. 왜냐하면 우리는 $\mathcal{X} \cap \mathcal{Y} = \emptyset$이라고 가정했기 때문입니다.따라서 선 세그먼트는 $\mathcal{X} \cup \mathcal{Y}$에도 없으므로 일반적으로 볼록 집합의 결합에서 볼록 할 필요가 없음을 증명합니다. 

![The union of two convex sets need not be convex.](../img/nonconvex.svg)
:label:`fig_nonconvex`

일반적으로 딥러닝의 문제는 볼록 집합으로 정의됩니다.예를 들어, 실수로 구성된 $d$차원 벡터의 집합인 $\mathbb{R}^d$은 볼록 집합입니다 (결국 $\mathbb{R}^d$의 두 점 사이의 선은 $\mathbb{R}^d$에 남아 있음).경우에 따라 $\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\| \leq r\}$에 정의된 반지름 $r$의 볼과 같이 제한된 길이의 변수를 사용하여 작업합니다. 

### 볼록 함수

볼록 집합이 생겼으면*볼록 함수* $f$를 도입할 수 있습니다.볼록 집합 $\mathcal{X}$이 주어지면 모든 $x, x' \in \mathcal{X}$과 모든 $\lambda \in [0, 1]$에 대해 함수 $f: \mathcal{X} \to \mathbb{R}$은*볼록*입니다. 

$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

이를 설명하기 위해 몇 가지 함수를 플로팅하고 요구 사항을 충족하는 함수를 확인하겠습니다.아래에서는 볼록 함수와 비볼록 함수를 정의합니다.

```{.python .input}
#@tab all
f = lambda x: 0.5 * x**2  # Convex
g = lambda x: d2l.cos(np.pi * x)  # Nonconvex
h = lambda x: d2l.exp(0.5 * x)  # Convex

x, segment = d2l.arange(-2, 2, 0.01), d2l.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
```

예상대로 코사인 함수는*nonconvex*인 반면 포물선과 지수 함수는 입니다.조건을 이해하려면 $\mathcal{X}$가 볼록 집합이라는 요구 사항이 필요합니다.그렇지 않으면 $f(\lambda x + (1-\lambda) x')$의 결과가 잘 정의되지 않을 수 있습니다. 

### 젠슨의 불평등

볼록 함수 $f$가 주어지면 가장 유용한 수학 도구 중 하나는*젠슨의 불평등*입니다.볼록 성의 정의를 일반화하는 것과 같습니다. 

$$\sum_i \alpha_i f(x_i)  \geq f\left(\sum_i \alpha_i x_i\right)    \text{ and }    E_X[f(X)]  \geq f\left(E_X[X]\right),$$
:eqlabel:`eq_jensens-inequality`

여기서 $\alpha_i$은 $\sum_i \alpha_i = 1$ 및 $X$가 랜덤 변수인 음수가 아닌 실수입니다.즉, 볼록 함수의 기대치는 기대값의 볼록 함수보다 작지 않으며, 여기서 볼록 함수는 일반적으로 더 간단한 표현식입니다.첫 번째 불평등을 증명하기 위해 볼록성의 정의를 한 번에 합계의 한 항에 반복적으로 적용합니다. 

Jensen의 불평등의 일반적인 응용 분야 중 하나는 더 복잡한 표현을 더 간단한 표현으로 묶는 것입니다.예를 들어, 부분적으로 관측된 랜덤 변수의 로그 우도와 관련해서 적용할 수 있습니다.즉, 우리는 

$$E_{Y \sim P(Y)}[-\log P(X \mid Y)] \geq -\log P(X),$$

$\int P(Y) P(X \mid Y) dY = P(X)$년부터.변동 방법에서 사용할 수 있습니다.여기서 $Y$는 일반적으로 관찰되지 않은 확률 변수이고, $P(Y)$은 분포 방법을 가장 잘 추측하고, $P(X)$은 $Y$가 통합된 분포입니다.예를 들어 클러스터링에서 $Y$는 클러스터 레이블이고 $P(X \mid Y)$는 클러스터 레이블을 적용할 때 생성 모델일 수 있습니다. 

## 등록 정보

볼록 함수에는 많은 유용한 속성이 있습니다.아래에서 일반적으로 사용되는 몇 가지 항목에 대해 설명합니다. 

### 로컬 미니마 아레 글로벌 미니마

무엇보다도 볼록 함수의 국소 최솟값도 전역 최솟값입니다.다음과 같이 모순을 통해 증명할 수 있습니다. 

볼록 집합 $\mathcal{X}$에 정의된 볼록 함수 $f$를 가정해 보겠습니다.$x^{\ast} \in \mathcal{X}$가 국소 최솟값이라고 가정합니다. 작은 양수 값 $p$이 존재하므로 $0 < |x - x^{\ast}| \leq p$을 충족하는 $x \in \mathcal{X}$에 대해 $f(x^{\ast}) < f(x)$이 있습니다. 

로컬 최소값 $x^{\ast}$이 글로벌 최소값인 $f$가 아니라고 가정합니다. $x' \in \mathcal{X}$이 존재하며, 이 경우 $f(x') < f(x^{\ast})$이 있습니다.또한 $\lambda = 1 - \frac{p}{|x^{\ast} - x'|}$와 같은 $\lambda \in [0, 1)$이 존재하므로 $0 < |\lambda x^{\ast} + (1-\lambda) x' - x^{\ast}| \leq p$가 있습니다.  

그러나 볼록 함수의 정의에 따르면 

$$\begin{aligned}
    f(\lambda x^{\ast} + (1-\lambda) x') &\leq \lambda f(x^{\ast}) + (1-\lambda) f(x') \\
    &< \lambda f(x^{\ast}) + (1-\lambda) f(x^{\ast}) \\
    &= f(x^{\ast}),
\end{aligned}$$

이는 $x^{\ast}$가 현지 최소값이라는 우리의 진술과 모순됩니다.따라서 $x' \in \mathcal{X}$에는 $x' \in \mathcal{X}$이 존재하지 않습니다.로컬 최소값 $x^{\ast}$도 글로벌 최소값입니다. 

예를 들어, 볼록 함수 $f(x) = (x-1)^2$의 국소 최솟값은 $x=1$이며, 이는 전역 최솟값이기도 합니다.

```{.python .input}
#@tab all
f = lambda x: (x - 1) ** 2
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```

볼록 함수의 국소 최솟값도 전역 최솟값이라는 사실은 매우 편리합니다.즉, 함수를 최소화하면 “막힐”수 없습니다.그러나 이것이 글로벌 최솟값이 두 개 이상 있을 수 없거나 존재할 수도 있다는 의미는 아닙니다.예를 들어, 함수 $f(x) = \mathrm{max}(|x|-1, 0)$은 구간 $[-1, 1]$에 걸쳐 최소값을 얻습니다.반대로, 함수 $f(x) = \exp(x)$는 $\mathbb{R}$에서 최소값을 얻지 못합니다. $x \to -\infty$의 경우 $0$으로 점근선하지만 $x$에는 $x$가 없습니다. 

### 아래 볼록 함수 세트는 볼록 함수입니다

볼록 함수의*아래 집합*을 통해 볼록 집합을 편리하게 정의할 수 있습니다.구체적으로 볼록 세트 $\mathcal{X}$에 정의된 볼록 함수 $f$가 주어지면 아래 집합이 설정됩니다. 

$$\mathcal{S}_b := \{x | x \in \mathcal{X} \text{ and } f(x) \leq b\}$$

볼록합니다.  

이것을 빨리 증명해 보겠습니다.$x, x' \in \mathcal{S}_b$에 대해 우리는 $\lambda x + (1-\lambda) x' \in \mathcal{S}_b$을 $\lambda \in [0, 1]$만큼 오래 보여줄 필요가 있다는 것을 상기하십시오.$f(x) \leq b$과 $f(x') \leq b$ 이후, 볼록성의 정의에 따라 우리는  

$$f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b.$$

### 볼록 및 2차 도함수

함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}$의 2차 도함수가 존재할 때마다 $f$가 볼록한지 확인하는 것은 매우 쉽습니다.우리가 해야 할 일은 $f$의 헤세 행렬이 양의 반한정인지 확인하는 것뿐입니다. 즉, 모든 $\mathbf{x} \in \mathbb{R}^n$에 대해 헤세 행렬 $\nabla^2f$을 $\mathbf{H}$, $\mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0$로 나타냅니다.예를 들어, 함수 $f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2$는 $\nabla^2 f = \mathbf{1}$부터 볼록합니다. 즉, 헤세 행렬은 단위 행렬입니다. 

공식적으로 두 번 미분 가능한 1차원 함수 $f: \mathbb{R} \rightarrow \mathbb{R}$은 두 번째 도함수 $f'' \geq 0$인 경우에만 볼록합니다.두 번 미분 가능한 다차원 함수 $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$의 경우 헤세 행렬 $\nabla^2f \succeq 0$인 경우에만 볼록합니다. 

먼저, 우리는 일차원적 사례를 증명해야 합니다.$f$의 볼록성이 $f'' \geq 0$를 의미한다는 것을 알기 위해 우리는 다음과 같은 사실을 사용합니다. 

$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$

두 번째 미분은 유한 차분에 대한 한계에 의해 주어지기 때문에 다음과 같습니다. 

$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$

$f'' \geq 0$가 $f$가 볼록하다는 것을 의미하는지 확인하기 위해 $f'' \geq 0$가 $f'$가 단조롭게 감소하지 않는 함수라는 것을 의미한다는 사실을 사용합니다.$a < x < b$을 $\mathbb{R}$에서 세 점으로 삼으십시오. 여기서 $x = (1-\lambda)a + \lambda b$과 $\lambda \in (0, 1)$입니다.평균값 정리에 따르면 $\alpha \in [a, x]$ 및 $\beta \in [x, b]$이 존재합니다. 

$$f'(\alpha) = \frac{f(x) - f(a)}{x-a} \text{ and } f'(\beta) = \frac{f(b) - f(x)}{b-x}.$$

따라서 단조성 $f'(\beta) \geq f'(\alpha)$에 의해 

$$\frac{x-a}{b-a}f(b) + \frac{b-x}{b-a}f(a) \geq f(x).$$

$x = (1-\lambda)a + \lambda b$년 이래로, 우리는 

$$\lambda f(b) + (1-\lambda)f(a) \geq f((1-\lambda)a + \lambda b),$$

따라서 볼록함을 증명합니다. 

둘째, 다차원 사례를 증명하기 전에 보조 정리가 필요합니다. $f: \mathbb{R}^n \rightarrow \mathbb{R}$는 모든 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$의 경우에만 볼록합니다. 

$$g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y}) \text{ where } z \in [0,1]$$ 

볼록합니다. 

$f$의 볼록성이 $g$가 볼록하다는 것을 의미한다는 것을 증명하기 위해, 우리는 모든 $a, b, \lambda \in [0, 1]$ (따라서 $0 \leq \lambda a + (1-\lambda) b \leq 1$) 에 대해 그것을 보여줄 수 있습니다. 

$$\begin{aligned} &g(\lambda a + (1-\lambda) b)\\
=&f\left(\left(\lambda a + (1-\lambda) b\right)\mathbf{x} + \left(1-\lambda a - (1-\lambda) b\right)\mathbf{y} \right)\\
=&f\left(\lambda \left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) \left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \right)\\
\leq& \lambda f\left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) f\left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \\
=& \lambda g(a) + (1-\lambda) g(b).
\end{aligned}$$

그 반대를 증명하기 위해 우리는 $\lambda \in [0, 1]$ 모두에 대해 그것을 보여줄 수 있습니다.  

$$\begin{aligned} &f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y})\\
=&g(\lambda \cdot 1 + (1-\lambda) \cdot 0)\\
\leq& \lambda g(1)  + (1-\lambda) g(0) \\
=& \lambda f(\mathbf{x}) + (1-\lambda) g(\mathbf{y}).
\end{aligned}$$

마지막으로 위의 보조 정리와 1 차원 사례의 결과를 사용하여 다차원 사례를 다음과 같이 입증 할 수 있습니다.다차원 함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}$은 모든 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ $g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y})$ (여기서 $z \in [0,1]$) 에 대해 볼록한 경우에만 볼록합니다.일차원 사례에 따르면, 이는 모든 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$에 대해 $g'' = (\mathbf{x} - \mathbf{y})^\top \mathbf{H}(\mathbf{x} - \mathbf{y}) \geq 0$ ($\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2f$) 인 경우에만 보유되며, 이는 양의 반유한 행렬의 정의에 따라 $\mathbf{H} \succeq 0$에 해당합니다. 

## 제약 조건

볼록 최적화의 좋은 특성 중 하나는 제약 조건을 효율적으로 처리할 수 있다는 것입니다.즉, 다음과 같은 형태의*제약 조건이 있는 최적화* 문제를 풀 수 있습니다. 

$$\begin{aligned} \mathop{\mathrm{minimize~}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \text{ subject to } & c_i(\mathbf{x}) \leq 0 \text{ for all } i \in \{1, \ldots, n\},
\end{aligned}$$

여기서 $f$는 목적 함수이고 함수 $c_i$는 제약 조건 함수입니다.이것이 무엇인지 확인하려면 $c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$의 경우를 고려하십시오.이 경우 매개변수 $\mathbf{x}$은 단위 볼로 제한됩니다.두 번째 제약 조건이 $c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$인 경우 이는 반쪽 공간에 있는 모든 $\mathbf{x}$에 해당합니다.두 제약 조건을 동시에 충족하는 것은 공 조각을 선택하는 것과 같습니다. 

### 라그랑주

일반적으로 제약 조건이 있는 최적화 문제를 푸는 것은 어렵습니다.이를 해결하는 한 가지 방법은 다소 단순한 직관을 가진 물리학에서 비롯됩니다.상자 안에 공이 있다고 상상해 보세요.공은 가장 낮은 곳으로 굴러 가고 중력은 상자의 측면이 공에 가할 수있는 힘과 균형을 이룹니다.요컨대, 목적 함수의 기울기 (즉, 중력) 는 제약 함수의 기울기에 의해 오프셋됩니다 (공은 벽이 “뒤로 밀기”때문에 상자 안에 남아 있어야 함).일부 구속조건은 활성화되지 않을 수 있습니다. 즉, 공에 닿지 않은 벽은 공에 힘을 가할 수 없습니다. 

*Lagrangian* $L$의 파생을 건너 뛰면 위의 추론은 다음 안장 지점 최적화 문제를 통해 표현할 수 있습니다. 

$$L(\mathbf{x}, \alpha_1, \ldots, \alpha_n) = f(\mathbf{x}) + \sum_{i=1}^n \alpha_i c_i(\mathbf{x}) \text{ where } \alpha_i \geq 0.$$

여기서 변수 $\alpha_i$ ($i=1,\ldots,n$) 는 제약 조건이 제대로 적용되도록 하는 이른바*라그랑주 승수*입니다.그것들은 모든 $i$에 대해 $c_i(\mathbf{x}) \leq 0$을 보장 할 수있을만큼 충분히 크게 선택됩니다.예를 들어, $c_i(\mathbf{x}) < 0$가 자연적으로 있는 $\mathbf{x}$의 경우, 우리는 결국 $\alpha_i = 0$을 선택하게 될 것입니다.또한 이것은 모든 $\alpha_i$에 대해 $L$를*최대화* 하고 $\mathbf{x}$에 대해 동시에*최소화*하려는 안장 지점 최적화 문제입니다.$L(\mathbf{x}, \alpha_1, \ldots, \alpha_n)$ 함수에 도달하는 방법을 설명하는 풍부한 문헌이 있습니다.우리의 목적을 위해 $L$의 안장 지점이 원래의 제약 최적화 문제가 최적으로 해결되는 곳이라는 것을 아는 것으로 충분합니다. 

### 패널티

제약 조건이 있는 최적화 문제를 최소*대략적으로* 충족하는 한 가지 방법은 라그랑주 $L$를 적용하는 것입니다.$c_i(\mathbf{x}) \leq 0$을 만족시키는 대신 목적 함수 $f(x)$에 $\alpha_i c_i(\mathbf{x})$을 추가하기만 하면 됩니다.이렇게 하면 제약 조건이 너무 심하게 위반되지 않습니다. 

사실, 우리는 이 트릭을 모두 사용해 왔습니다.:numref:`sec_weight_decay`에서 체중 감소를 고려하십시오.여기서는 목적 함수에 $\frac{\lambda}{2} \|\mathbf{w}\|^2$를 추가하여 $\mathbf{w}$가 너무 커지지 않도록 합니다.제한된 최적화 관점에서 볼 때 일부 반경 $r$에 대해 $\|\mathbf{w}\|^2 - r^2 \leq 0$이 보장된다는 것을 알 수 있습니다.$\lambda$의 값을 조정하면 $\mathbf{w}$의 크기를 변경할 수 있습니다. 

일반적으로 페널티를 추가하는 것은 대략적인 제약 조건 충족을 보장하는 좋은 방법입니다.실제로 이것은 정확한 만족보다 훨씬 더 강력합니다.또한 비볼록 문제의 경우 볼록 사례에서 정확한 접근법을 매우 매력적으로 만드는 많은 속성 (예: 최적성) 이 더 이상 유지되지 않습니다. 

### 투영법

제약 조건을 충족하기 위한 대체 전략은 투영입니다.예를 들어 :numref:`sec_rnn_scratch`에서 그래디언트 클리핑을 다룰 때 이전에 다시 만났습니다.거기에서 그라디언트의 길이가 $\theta$를 통해 제한되도록했습니다. 

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, \theta/\|\mathbf{g}\|).$$

이것은 반경 $\theta$의 볼에 $\mathbf{g}$의*투영*으로 판명되었습니다.보다 일반적으로 볼록 세트 $\mathcal{X}$의 투영은 다음과 같이 정의됩니다. 

$$\mathrm{Proj}_\mathcal{X}(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in \mathcal{X}} \|\mathbf{x} - \mathbf{x}'\|,$$

이는 $\mathcal{X}$에서 $\mathbf{x}$에서 가장 가까운 지점입니다.  

![Convex Projections.](../img/projections.svg)
:label:`fig_projections`

투영의 수학적 정의는 약간 추상적으로 들릴 수 있습니다. :numref:`fig_projections`는 좀 더 명확하게 설명합니다.여기에는 원과 다이아몬드의 두 개의 볼록 세트가 있습니다.두 세트 안의 점 (노란색) 은 투영 중에 변경되지 않은 상태로 유지됩니다.두 세트 외부의 점 (검은색) 은 원래 점에 가장 가까운 세트 내부 점 (빨간색) 에 투영됩니다 (검은색).$L_2$ 볼의 경우 방향이 변경되지 않지만 다이아몬드의 경우처럼 일반적으로 그럴 필요는 없습니다. 

볼록 투영의 용도 중 하나는 희소 가중치 벡터를 계산하는 것입니다.이 경우 :numref:`fig_projections`의 다이아몬드 케이스의 일반화된 버전인 $L_1$ 볼에 웨이트 벡터를 투영합니다. 

## 요약

딥 러닝의 맥락에서 볼록 함수의 주요 목적은 최적화 알고리즘에 동기를 부여하고 이를 자세히 이해하도록 돕는 것입니다.다음에서는 그래디언트 하강법과 확률적 경사하강법을 그에 따라 어떻게 도출할 수 있는지 살펴보겠습니다. 

* 볼록 집합의 교차점은 볼록합니다.노동 조합은 그렇지 않습니다.
* 볼록 함수의 기대치는 기대값의 볼록 함수 (Jensen의 부등식) 보다 작지 않습니다.
* 두 번 미분 가능한 함수는 헤세 행렬 (2차 도함수로 구성된 행렬) 이 양의 반유한 함수인 경우에만 볼록합니다.
* 라그랑주 를 통해 볼록 제약 조건을 추가할 수 있습니다.실제로 목적 함수에 페널티를 추가하기만 하면 됩니다.
* 투영은 원래 점에 가장 가까운 볼록 세트의 점에 매핑됩니다.

## 연습문제

1. 세트 내의 점 사이에 모든 선을 그리고 선이 포함되어 있는지 확인하여 세트의 볼록성을 확인한다고 가정합니다.
    1. 경계의 점만 확인하면 충분하다는 것을 입증하십시오.
    1. 이 세트의 꼭짓점만을 확인하는 것으로 충분하다는 것을 입증하십시오.
1. $p$ 규범을 사용하여 반경 $r$의 공을 $\mathcal{B}_p[r] \stackrel{\mathrm{def}}{=} \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\|_p \leq r\}$로 나타냅니다.$\mathcal{B}_p[r]$이 모든 $p \geq 1$에 대해 볼록하다는 것을 증명하십시오.
1. 볼록 함수 $f$ 및 $g$가 주어지면 $\mathrm{max}(f, g)$도 볼록하다는 것을 알 수 있습니다.$\mathrm{min}(f, g)$이 볼록하지 않다는 것을 증명하십시오.
1. softmax 함수의 정규화가 볼록하다는 것을 증명하십시오.보다 구체적으로 $f(x) = \log \sum_i \exp(x_i)$의 볼록성을 입증하십시오.
1. 선형 부분공간, 즉 $\mathcal{X} = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$가 볼록 집합임을 입증합니다.
1. $\mathbf{b} = \mathbf{0}$이 있는 선형 부분공간의 경우 투영 $\mathrm{Proj}_\mathcal{X}$를 일부 행렬 $\mathbf{M}$에 대해 $\mathbf{M} \mathbf{x}$으로 쓸 수 있음을 입증하십시오.
1. 두 번 미분 가능한 볼록 함수 $f$의 경우 일부 $\xi \in [0, \epsilon]$에 대해 $f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$를 작성할 수 있음을 보여줍니다.
1. $\|\mathbf{w}\|_1 > 1$가 있는 벡터 $\mathbf{w} \in \mathbb{R}^d$가 주어지면 $L_1$ 단위 볼에 대한 투영을 계산합니다.
    1. 중간 단계로 페널티된 목표 $\|\mathbf{w} - \mathbf{w}'\|^2 + \lambda \|\mathbf{w}'\|_1$를 작성하고 주어진 $\lambda > 0$에 대한 해를 계산합니다.
    1. 많은 시행 착오 없이 $\lambda$의 “올바른” 값을 찾을 수 있습니까?
1. 볼록 집합 $\mathcal{X}$과 두 개의 벡터 $\mathbf{x}$ 및 $\mathbf{y}$가 주어지면 투영이 거리를 증가시키지 않는다는 것을 증명합니다. 즉, $\|\mathbf{x} - \mathbf{y}\| \geq \|\mathrm{Proj}_\mathcal{X}(\mathbf{x}) - \mathrm{Proj}_\mathcal{X}(\mathbf{y})\|$입니다.

[Discussions](https://discuss.d2l.ai/t/350)
