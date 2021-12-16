# 그라데이션 하강
:label:`sec_gd`

이 섹션에서는*기울기 하강*의 기본 개념을 소개합니다.딥러닝에서 직접 사용되는 경우는 거의 없지만 확률적 경사하강법 알고리즘을 이해하기 위해서는 기울기 하강법을 이해하는 것이 중요합니다.예를 들어, 학습 속도가 너무 커서 최적화 문제가 서로 다를 수 있습니다.이 현상은 기울기 하강에서도 이미 볼 수 있습니다.마찬가지로, 사전 조건은 기울기 하강법의 일반적인 기법이며 고급 알고리즘으로 이어집니다.간단한 특수 사례부터 시작하겠습니다. 

## 1차원 경사하강법

한 차원의 기울기 하강법은 기울기 하강 알고리즘이 목적 함수의 값을 줄일 수 있는 이유를 설명하는 훌륭한 예입니다.연속적으로 미분 가능한 실수 값 함수 $f: \mathbb{R} \rightarrow \mathbb{R}$를 고려해 보십시오.Taylor 확장을 사용하여 

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

즉, 1차 근사에서 $f(x+\epsilon)$는 함수 값 $f(x)$와 $x$에서 1차 도함수 $f'(x)$에 의해 주어집니다.작은 $\epsilon$의 경우 음의 기울기 방향으로 이동하면 $f$이 감소한다고 가정하는 것은 무리가 아닙니다.작업을 단순하게 유지하기 위해 고정 스텝 크기 $\eta > 0$를 선택하고 $\epsilon = -\eta f'(x)$을 선택합니다.이것을 위의 Taylor 확장팩에 연결하면 

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$
:eqlabel:`gd-taylor-2`

미분 $f'(x) \neq 0$가 사라지지 않으면 $\eta f'^2(x)>0$ 이후 진전을 이룹니다.또한 고차 항이 관련되지 않도록 항상 충분히 작게 $\eta$를 선택할 수 있습니다.따라서 우리는 

$$f(x - \eta f'(x)) \lessapprox f(x).$$

즉, 

$$x \leftarrow x - \eta f'(x)$$

$x$을 반복하기 위해 함수 $f(x)$의 값이 감소할 수 있습니다.따라서 기울기 하강에서는 먼저 초기 값 $x$과 상수 $\eta > 0$를 선택한 다음 정지 조건에 도달 할 때까지 (예: 기울기 $|f'(x)|$) 의 크기가 충분히 작거나 반복 횟수가 특정 값에 도달 할 때까지 $x$을 계속 반복하는 데 사용합니다.값. 

간단하게 하기 위해 기울기 하강을 구현하는 방법을 설명하기 위해 목적 함수 $f(x)=x^2$을 선택합니다.$x=0$가 $f(x)$를 최소화하는 해결책이라는 것을 알고 있지만, 여전히 이 간단한 함수를 사용하여 $x$이 어떻게 변하는지 관찰합니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
def f(x):  # Objective function
    return x ** 2

def f_grad(x):  # Gradient (derivative) of the objective function
    return 2 * x
```

다음으로 $x=10$을 초기값으로 사용하고 $\eta=0.2$를 가정합니다.기울기 하강을 사용하여 $x$를 10번 반복하면 결국 $x$ 값이 최적 해에 근접한다는 것을 알 수 있습니다.

```{.python .input}
#@tab all
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
```

$x$를 통한 최적화 진행 상황은 다음과 같이 그려질 수 있습니다.

```{.python .input}
#@tab all
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```

### 학습 속도
:label:`subsec_gd-learningrate`

학습률 $\eta$는 알고리즘 설계자가 설정할 수 있습니다.너무 작은 학습률을 사용하면 $x$가 매우 느리게 업데이트되어 더 나은 솔루션을 얻기 위해 더 많은 반복이 필요합니다.이러한 경우에 어떤 일이 발생하는지 확인하려면 $\eta = 0.05$에 대해 동일한 최적화 문제의 진행 상황을 고려해 보십시오.보시다시피, 10 단계 후에도 여전히 최적의 솔루션과는 거리가 멀습니다.

```{.python .input}
#@tab all
show_trace(gd(0.05, f_grad), f)
```

반대로 지나치게 높은 학습률을 사용하면 $\left|\eta f'(x)\right|$이 1차 Taylor 확장 공식에 비해 너무 클 수 있습니다.즉, :eqref:`gd-taylor-2`에서 $\mathcal{O}(\eta^2 f'^2(x))$이라는 용어가 중요해질 수 있습니다.이 경우 $x$의 반복이 $f(x)$의 값을 낮출 수 있다고 보장할 수 없습니다.예를 들어 학습률을 $\eta=1.1$로 설정하면 $x$이 최적 해 $x=0$을 초과하여 점차 분기합니다.

```{.python .input}
#@tab all
show_trace(gd(1.1, f_grad), f)
```

### 로컬 미니마

비볼록 함수에서 어떤 일이 발생하는지 설명하기 위해 일부 상수 $c$에 대한 $f(x) = x \cdot \cos(cx)$의 경우를 생각해 보십시오.이 함수는 무한히 많은 국소 최솟값을 갖습니다.학습률을 선택하고 문제가 얼마나 잘 조절되었는지에 따라 많은 해결책 중 하나가 될 수 있습니다.아래 예는 (비현실적으로) 높은 학습률이 지역 최솟값을 낮추는 방법을 보여줍니다.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

show_trace(gd(2, f_grad), f)
```

## 다변량 경사하강법

이제 일변량 사례에 대한 더 나은 직관을 얻었으므로 $\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$의 상황을 고려해 보겠습니다.즉, 목적 함수 $f: \mathbb{R}^d \to \mathbb{R}$은 벡터를 스칼라로 매핑합니다.이에 따라 기울기도 다변량입니다.이 벡터는 $d$ 편도함수로 구성된 벡터입니다. 

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

기울기의 각 부분 미분 요소 ($\partial f(\mathbf{x})/\partial x_i$) 는 입력 $x_i$에 대한 $\mathbf{x}$에서 $f$의 변화율을 나타낸다.일변량의 경우와 마찬가지로 다변량 함수에 해당하는 Taylor 근사를 사용하여 수행해야 할 작업에 대한 아이디어를 얻을 수 있습니다.특히, 우리는 

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \mathbf{\boldsymbol{\epsilon}}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

즉, $\boldsymbol{\epsilon}$에서 최대 2차 항까지 가장 가파른 하강 방향은 음의 기울기 $-\nabla f(\mathbf{x})$으로 지정됩니다.적절한 학습률 $\eta > 0$를 선택하면 원형 기울기 하강 알고리즘이 생성됩니다. 

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

알고리즘이 실제로 어떻게 작동하는지 확인하기 위해 2차원 벡터 $\mathbf{x} = [x_1, x_2]^\top$을 입력으로, 스칼라를 출력으로 사용하여 목적 함수 $f(\mathbf{x})=x_1^2+2x_2^2$을 생성해 보겠습니다.기울기는 $\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$로 지정됩니다.초기 위치 $[-5, -2]$에서 기울기 하강에 의해 $\mathbf{x}$의 궤적을 관찰합니다.  

먼저 도우미 함수가 두 개 더 필요합니다.첫 번째 함수는 업데이트 함수를 사용하여 초기 값에 20번 적용합니다.두 번째 도우미는 $\mathbf{x}$의 궤적을 시각화합니다.

```{.python .input}
#@tab all
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """Optimize a 2D objective function with a customized trainer."""
    # `s1` and `s2` are internal state variables that will be used later
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

다음으로 학습률 $\eta = 0.1$에 대한 최적화 변수 $\mathbf{x}$의 궤적을 관찰합니다.20단계 후에 $\mathbf{x}$의 값이 최소값인 $[0, 0]$에 근접한다는 것을 알 수 있습니다.진행은 다소 느리지 만 상당히 잘 행동합니다.

```{.python .input}
#@tab all
def f_2d(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # Gradient of the objective function
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

## 적응형 방법

:numref:`subsec_gd-learningrate`에서 볼 수 있듯이 학습률 $\eta$를 “올바르게”얻는 것은 까다 롭습니다.너무 작게 선택하면 진전이 거의 없습니다.너무 크게 선택하면 해가 진동하고 최악의 경우 발산할 수도 있습니다.$\eta$를 자동으로 결정하거나 학습률을 전혀 선택하지 않아도 된다면 어떨까요?이 경우 목적 함수의 값과 기울기뿐 아니라*curvature*도 살펴보는 2차 방법이 도움이 될 수 있습니다.이러한 방법은 계산 비용 때문에 딥러닝에 직접 적용할 수는 없지만, 아래에 설명된 알고리즘의 바람직한 여러 특성을 모방하는 고급 최적화 알고리즘을 설계하는 방법에 대한 유용한 직관을 제공합니다. 

### 뉴턴의 방법

일부 함수 $f: \mathbb{R}^d \rightarrow \mathbb{R}$의 테일러 확장을 검토하면 첫 번째 학기 이후에 중단 할 필요가 없습니다.실제로 다음과 같이 쓸 수 있습니다. 

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`

번거로운 표기법을 피하기 위해 $\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2 f(\mathbf{x})$를 $f$의 헤세 행렬 ($d \times d$ 행렬) 으로 정의합니다.작은 $d$과 간단한 문제의 경우 $\mathbf{H}$는 계산하기 쉽습니다.반면 심층 신경망의 경우 $\mathbf{H}$는 $\mathcal{O}(d^2)$ 항목을 저장하는 데 드는 비용으로 인해 엄청나게 클 수 있습니다.또한 역전파를 통해 계산하기에는 너무 많은 비용이 들 수 있습니다.지금은 이러한 고려 사항을 무시하고 어떤 알고리즘을 얻을 수 있는지 살펴 보겠습니다. 

결국 최소값 $f$은 $\nabla f = 0$을 충족합니다.:numref:`subsec_calculus-grad`의 미적분 규칙을 따르고 $\boldsymbol{\epsilon}$과 관련하여 :eqref:`gd-hot-taylor`의 도함수를 취하고 우리가 도달하는 고차 항을 무시함으로써 

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \text{ and hence }
\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

즉, 최적화 문제의 일부로 헤세 행렬 $\mathbf{H}$를 반전해야 합니다. 

간단한 예로, $f(x) = \frac{1}{2} x^2$의 경우 $\nabla f(x) = x$와 $\mathbf{H} = 1$가 있습니다.따라서 $x$에 대해 우리는 $\epsilon = -x$를 얻습니다.즉, *단일* 단계만으로도 조정할 필요 없이 완벽하게 수렴할 수 있습니다!아아, 여기서 조금 운이 좋았습니다. 테일러 확장팩은 $f(x+\epsilon)= \frac{1}{2} x^2 + \epsilon x + \frac{1}{2} \epsilon^2$ 이후 정확했습니다.  

다른 문제에서 어떤 일이 일어나는지 봅시다.일부 상수 $c$에 대해 볼록 쌍곡선 코사인 함수 $f(x) = \cosh(cx)$가 주어지면 몇 번의 반복 후에 $x=0$의 전역 최소값에 도달한다는 것을 알 수 있습니다.

```{.python .input}
#@tab all
c = d2l.tensor(0.5)

def f(x):  # Objective function
    return d2l.cosh(c * x)

def f_grad(x):  # Gradient of the objective function
    return c * d2l.sinh(c * x)

def f_hess(x):  # Hessian of the objective function
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)
```

이제 일부 상수 $c$에 대해 $f(x) = x \cos(c x)$과 같은*볼록하지 않은* 함수를 고려해 보겠습니다.결국 뉴턴의 방법에서는 결국 헤세 행렬로 나눕니다.즉, 두 번째 도함수가*음수*이면 $f$의 값을*증가* 하는 방향으로 걸어갈 수 있습니다.이것이 알고리즘의 치명적인 결함입니다.실제로 어떤 일이 일어나는지 봅시다.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # Hessian of the objective function
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

이것은 엄청나게 잘못되었습니다.어떻게 고칠 수 있을까요?한 가지 방법은 헤세 행렬의 절대값을 대신 취하여 헤세 행렬을 “고정”하는 것입니다.또 다른 전략은 학습률을 회복하는 것입니다.이것은 목적을 무너 뜨리는 것처럼 보이지만 그렇지 않습니다.2차 정보가 있으면 곡률이 클 때마다 주의를 기울이고 목적 함수가 더 평평할 때마다 더 긴 스텝을 취할 수 있습니다.$\eta = 0.5$와 같이 약간 더 작은 학습 속도로 이것이 어떻게 작동하는지 살펴 보겠습니다.보시다시피 상당히 효율적인 알고리즘이 있습니다.

```{.python .input}
#@tab all
show_trace(newton(0.5), f)
```

### 컨버전스 분석

우리는 일부 볼록 및 세 배 미분 가능한 목적 함수 $f$에 대한 뉴턴 방법의 수렴률만을 분석합니다. 여기서 두 번째 미분은 0이 아닙니다, 즉 $f'' > 0$입니다.다변량 증거는 아래의 1 차원 인수를 간단하게 확장 한 것이며 직관 측면에서 그다지 도움이되지 않기 때문에 생략되었습니다. 

$k^\mathrm{th}$ 반복에서 $x$의 값을 $x^{(k)}$로 나타내고 $e^{(k)} \stackrel{\mathrm{def}}{=} x^{(k)} - x^*$를 $k^\mathrm{th}$ 반복에서 최적성과의 거리가 되도록 합니다.테일러 확장을 통해 우리는 조건 $f'(x^*) = 0$을 다음과 같이 쓸 수 있습니다. 

$$0 = f'(x^{(k)} - e^{(k)}) = f'(x^{(k)}) - e^{(k)} f''(x^{(k)}) + \frac{1}{2} (e^{(k)})^2 f'''(\xi^{(k)}),$$

이는 일부 $\xi^{(k)} \in [x^{(k)} - e^{(k)}, x^{(k)}]$를 보유하고 있습니다.위의 확장을 $f''(x^{(k)})$ 수익률로 나누면 

$$e^{(k)} - \frac{f'(x^{(k)})}{f''(x^{(k)})} = \frac{1}{2} (e^{(k)})^2 \frac{f'''(\xi^{(k)})}{f''(x^{(k)})}.$$

$x^{(k+1)} = x^{(k)} - f'(x^{(k)}) / f''(x^{(k)})$ 업데이트가 있음을 상기하십시오.이 업데이트 방정식을 연결하고 양쪽의 절대 값을 취하면 

$$\left|e^{(k+1)}\right| = \frac{1}{2}(e^{(k)})^2 \frac{\left|f'''(\xi^{(k)})\right|}{f''(x^{(k)})}.$$

결과적으로 경계 $\left|f'''(\xi^{(k)})\right| / (2f''(x^{(k)})) \leq c$의 영역에 있을 때마다 2차 감소 오차가 발생합니다.  

$$\left|e^{(k+1)}\right| \leq c (e^{(k)})^2.$$

제쳐두고 최적화 연구자들은 이것을*선형* 수렴이라고 부르는 반면 $\left|e^{(k+1)}\right| \leq \alpha \left|e^{(k)}\right|$와 같은 조건은*일정* 수렴률이라고 합니다.이 분석에는 여러 가지 주의 사항이 있습니다.첫째, 빠른 수렴 영역에 도달 할 때 실제로 보장이 많지 않습니다.대신 일단 도달하면 수렴이 매우 빠르다는 것을 알 수 있습니다.둘째, 이 분석을 위해서는 $f$가 고차 파생 상품까지 잘 동작해야 합니다.$f$가 값을 어떻게 변경할 수 있는지에 대한 “놀라운” 속성이 없는지 확인하는 것입니다. 

### 사전 컨디셔닝

당연히 전체 헤세 행렬을 계산하고 저장하는 것은 매우 비쌉니다.따라서 대안을 찾는 것이 바람직합니다.문제를 개선하는 한 가지 방법은*사전 조건*입니다.헤세 행렬을 완전히 계산하지 않고*대각선* 항목만 계산합니다.이로 인해 양식의 알고리즘이 업데이트됩니다. 

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \mathrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}).$$

이것이 전체 Newton의 방법만큼 좋지는 않지만 사용하지 않는 것보다 훨씬 낫습니다.왜 이것이 좋은 생각인지 확인하려면 한 변수가 높이를 밀리미터 단위로 나타내고 다른 변수는 높이를 킬로미터 단위로 나타내는 상황을 생각해 보십시오.두 자연 척도가 미터 단위라고 가정하면 매개 변수화에 끔찍한 불일치가 있습니다.다행히도 사전 조건을 사용하면 이 문제가 제거됩니다.기울기 하강을 사용하여 효과적으로 사전 컨디셔닝하면 각 변수 (벡터 $\mathbf{x}$의 좌표) 에 대해 다른 학습률을 선택하는 데 해당합니다.나중에 살펴보겠지만, 사전 조정은 확률적 경사하강법 최적화 알고리즘의 일부 혁신을 주도합니다.  

### 기울기 하강법 및 선 검색

기울기 하강법의 주요 문제 중 하나는 목표를 초과하거나 불충분 한 진전을 이룰 수 있다는 것입니다.이 문제에 대한 간단한 수정은 기울기 하강법과 함께 선 검색을 사용하는 것입니다.즉, $\nabla f(\mathbf{x})$에 의해 주어진 방향을 사용한 다음 학습률 $\eta$가 $f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$를 최소화하는 이진 검색을 수행합니다. 

이 알고리즘은 빠르게 수렴됩니다 (분석 및 증명은 예: :cite:`Boyd.Vandenberghe.2004` 참조).그러나 딥러닝의 목적상, 직선 탐색의 각 단계에서 전체 데이터셋에 대한 목적 함수를 평가해야 하기 때문에 이렇게 하는 것은 그리 현실적이지 않습니다.이를 달성하기에는 너무 많은 비용이 듭니다. 

## 요약

* 학습률이 중요합니다.너무 크고 우리는 갈라지고 너무 작아서 진전을 이루지 못합니다.
* 경사하강법은 국소 최솟값에 갇힐 수 있습니다.
* 높은 차원에서 학습률을 조정하는 것은 복잡합니다.
* 사전 조정은 스케일 조정에 도움이 될 수 있습니다.
* Newton의 방법은 볼록 문제에서 제대로 작동하기 시작하면 훨씬 빠릅니다.
* 비볼록 문제를 조정하지 않고 Newton의 방법을 사용하지 않도록 주의하십시오.

## 연습문제

1. 경사하강법을 위한 다양한 학습률과 목적 함수를 실험해 봅니다.
1. 구간 $[a, b]$에서 볼록 함수를 최소화하기 위해 직선 탐색을 구현합니다.
    1. 이진 검색, 즉 $[a, (a+b)/2]$를 선택할지 $[(a+b)/2, b]$를 선택할지 여부를 결정하기 위해 파생물이 필요합니까?
    1. 알고리즘의 수렴 속도는 얼마나 빠릅니까?
    1. 알고리즘을 구현하고 $\log (\exp(x) + \exp(-2x -3))$를 최소화하는 데 적용합니다.
1. 기울기 하강이 매우 느린 $\mathbb{R}^2$에 정의된 목적 함수를 설계합니다.힌트: 좌표의 크기를 다르게 조정합니다.
1. 사전 조건을 사용하여 Newton 방법의 경량 버전을 구현합니다.
    1. 대각 헤세 행렬을 선조건자로 사용합니다.
    1. 실제 (부호가 있을 수 있는) 값 대신 절대값을 사용합니다.
    1. 위의 문제에 이를 적용합니다.
1. 위의 알고리즘을 여러 목적 함수 (볼록 또는 그렇지 않음) 에 적용합니다.좌표를 $45$도 회전하면 어떻게 되나요?

[Discussions](https://discuss.d2l.ai/t/351)
