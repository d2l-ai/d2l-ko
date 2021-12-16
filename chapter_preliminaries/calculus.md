# 미적분
:label:`sec_calculus`

다각형의 면적을 찾는 것은 적어도 2,500년 전까지 고대 그리스인들이 다각형을 삼각형으로 나누고 면적을 합산할 때까지 신비로 남아있었습니다.원과 같은 곡선 모양의 영역을 찾기 위해 고대 그리스인들은 이러한 모양의 다각형을 새겼습니다.:numref:`fig_circle_area`에서 볼 수 있듯이 길이가 같은 변이 더 많은 내접 다각형은 원과 더 비슷합니다.이 과정은*소진 방법*으로도 알려져 있습니다. 

![Find the area of a circle with the method of exhaustion.](../img/polygon-circle.svg)
:label:`fig_circle_area`

실제로 고갈 방법은*적분 미적분* (:numref:`sec_integral_calculus`에 설명 됨) 이 시작된 곳입니다.2,000년이 지난 후, 미적분학의 다른 지점인 *미적분학*이 발명되었습니다.미분 미적분학의 가장 중요한 응용 분야 중 최적화 문제는*최고*를 수행하는 방법을 고려합니다.:numref:`subsec_norms_and_objectives`에서 설명한 것처럼 이러한 문제는 딥 러닝에서 어디에나 존재합니다. 

딥 러닝에서는 점점 더 많은 데이터를 볼 때 더 좋아질 수 있도록 모델을 연속적으로 업데이트하여*훈련* 합니다.일반적으로 더 나아진다는 것은 “모델이 얼마나 나쁜지*”라는 질문에 답하는 점수인*손실 함수*를 최소화하는 것을 의미합니다.이 질문은 보이는 것보다 더 미묘합니다.궁극적으로 우리가 정말로 관심을 갖는 것은 이전에 보지 못했던 데이터에서 잘 작동하는 모델을 만드는 것입니다.하지만 실제로 볼 수 있는 데이터에만 모델을 맞출 수 있습니다.따라서 모델을 피팅하는 작업을 두 가지 주요 관심사로 분해 할 수 있습니다. (i) *최적화*: 관찰 된 데이터에 모델을 피팅하는 프로세스; (ii) *일반화*: 유효성이 정확한 데이터 세트를 넘어서는 모델을 생성하는 방법을 안내하는 수학적 원리와 실무자의 지혜예제를 훈련시키는 데 사용됩니다. 

이후 장에서 최적화 문제와 방법을 이해하는 데 도움이 되도록 딥 러닝에서 일반적으로 사용되는 미분 미적분에 대한 간략한 입문서를 제공합니다. 

## 파생 상품 및 차별화

먼저 거의 모든 딥러닝 최적화 알고리즘에서 중요한 단계인 도함수 계산을 다룹니다.딥러닝에서는 일반적으로 모델의 파라미터와 관련하여 구분할 수 있는 손실 함수를 선택합니다.간단히 말해서, 이것은 각 매개 변수에 대해 손실이 얼마나 빨리 증가 또는 감소하는지 결정할 수 있음을 의미합니다. 즉, 해당 매개 변수를 무한히 적은 양으로*증가* 또는*감소*할 수 있습니다. 

입력과 출력이 모두 스칼라 인 함수 $f: \mathbb{R} \rightarrow \mathbb{R}$가 있다고 가정합니다.[**$f$의*미분*은 다음과 같이 정의됩니다.] 

(**$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h},$달러**) :eqlabel:`eq_derivative` 

이 제한이 존재하는 경우$f'(a)$가 존재하는 경우 $f$은 $a$에서*차별화 가능한*이라고 합니다.$f$이 구간의 모든 수에서 미분 가능한 경우 이 함수는 이 구간에서 미분 가능합니다.:eqref:`eq_derivative`의 도함수 $f'(x)$을 $x$에 대한 $f(x)$의*순간적* 변화율로 해석할 수 있습니다.소위 순간 변화율은 $x$의 변동 $h$을 기반으로하며, 이는 $0$에 근접합니다. 

도함수를 설명하기 위해 예제를 사용해 보겠습니다.(**$u = f(x) = 3x^2-4x$.** 정의)

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

[**$x=1$를 설정하고 $h$이 $0$에 접근하게함으로써 $\frac{f(x+h) - f(x)}{h}$의 수치 결과**] :eqref:`eq_derivative`에서 (**$2$에 접근합니다**) 이 실험은 수학적 증거는 아니지만 나중에 $u'$이 $x=1$일 때 $u'$이 $2$라는 것을 알게 될 것입니다.

```{.python .input}
#@tab all
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1
```

파생 상품에 대한 몇 가지 동등한 표기법을 익히겠습니다.$y = f(x)$을 감안할 때, 여기서 $x$와 $y$는 각각 함수 $f$의 독립 변수이자 종속 변수입니다.다음 표현식은 동일합니다. 

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

여기서 기호 $\frac{d}{dx}$ 및 $D$는*미분*의 연산을 나타내는*미분 연산자*입니다.다음 규칙을 사용하여 공통 함수를 구분할 수 있습니다. 

* $DC = 0$ ($C$는 상수입니다),
* $Dx^n = nx^{n-1}$ (*거듭제곱 규칙*, $n$는 임의의 실수입니다),
* $De^x = e^x$,
* $D\ln(x) = 1/x.$

위의 공통 함수와 같은 몇 가지 간단한 함수로 구성된 함수를 구별하기 위해 다음 규칙이 유용 할 수 있습니다.함수 $f$과 $g$가 모두 미분 가능하고 $C$가 상수라고 가정하면*상수 다중 규칙*이 있습니다. 

$$\frac{d}{dx} [Cf(x)] = C \frac{d}{dx} f(x),$$

*합계 규칙* 

$$\frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx} f(x) + \frac{d}{dx} g(x),$$

*제품 규칙* 

$$\frac{d}{dx} [f(x)g(x)] = f(x) \frac{d}{dx} [g(x)] + g(x) \frac{d}{dx} [f(x)],$$

그리고*몫 규칙* 

$$\frac{d}{dx} \left[\frac{f(x)}{g(x)}\right] = \frac{g(x) \frac{d}{dx} [f(x)] - f(x) \frac{d}{dx} [g(x)]}{[g(x)]^2}.$$

이제 위의 규칙 중 몇 가지를 적용하여 $u' = f'(x) = 3 \frac{d}{dx} x^2-4\frac{d}{dx}x = 6x-4$을 찾을 수 있습니다.따라서 $x = 1$를 설정하면 $u' = 2$가 생깁니다. 이는 수치 결과가 $2$에 접근하는 이 섹션의 이전 실험에서 뒷받침됩니다.이 도함수는 $x = 1$일 때 곡선 $u = f(x)$에 대한 접선의 기울기이기도 합니다. 

[**이러한 도함수 해석을 시각화하기 위해 파이썬에서 널리 사용되는 플로팅 라이브러리인 `matplotlib`, **] 를 사용할 것입니다.`matplotlib`에서 생성된 그림의 속성을 구성하려면 몇 가지 함수를 정의해야 합니다.다음에서 `use_svg_display` 함수는 더 선명한 이미지를 위해 svg 그림을 출력할 `matplotlib` 패키지를 지정합니다.`# @save `주석은 다음 함수, 클래스 또는 명령문이 `d2l` 패키지에 저장되어 나중에 다시 정의하지 않고 직접 호출 (예: `d2l.use_svg_display()`) 할 수 있는 특별한 표시입니다.

```{.python .input}
#@tab all
def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')
```

그림 크기를 지정하기 위해 `set_figsize` 함수를 정의합니다.여기서는 가져오기 문 `from matplotlib import pyplot as plt`이 서문의 `d2l` 패키지에 저장되도록 표시되었으므로 `d2l.plt`를 직접 사용합니다.

```{.python .input}
#@tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

다음 `set_axes` 함수는 `matplotlib`에 의해 생성된 그림 축의 속성을 설정합니다.

```{.python .input}
#@tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

그림 구성을 위한 이 세 가지 함수를 사용하여 책 전체에서 많은 곡선을 시각화해야 하므로 여러 곡선을 간결하게 플로팅하는 `plot` 함수를 정의합니다.

```{.python .input}
#@tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

이제 [**함수 $u = f(x)$와 해당 접선 $y = 2x - 3$을 $x=1$**에 플로팅할 수 있습니다. 여기서 계수 $2$는 접선의 기울기입니다.

```{.python .input}
#@tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## 부분 도함수

지금까지 우리는 단 하나의 변수의 함수의 차별화를 다루었습니다.딥러닝에서 함수는 종종*많은* 변수에 종속됩니다.따라서 차별화 아이디어를 이러한*다변량* 함수로 확장해야 합니다. 

$y = f(x_1, x_2, \ldots, x_n)$을 변수가 $n$인 함수로 설정합니다.$i^\mathrm{th}$ 매개변수 $x_i$에 대한 $y$의*편미함*은 다음과 같습니다. 

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$

$\frac{\partial y}{\partial x_i}$을 계산하기 위해 $x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$를 상수로 취급하고 $x_i$에 대해 $y$의 미분을 계산할 수 있습니다.편도함수 표기법의 경우 다음과 같습니다. 

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

## 그라디언트
:label:`subsec_calculus-grad`

모든 변수에 대해 다변량 함수의 편도함수를 결합하여 함수의*gradient* 벡터를 얻을 수 있습니다.함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}$의 입력값이 $n$차원 벡터 $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$이고 출력값이 스칼라라고 가정합니다.$\mathbf{x}$에 대한 함수 $f(\mathbf{x})$의 기울기는 $n$ 편도함수로 구성된 벡터입니다. 

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,$$

여기서 $\nabla_{\mathbf{x}} f(\mathbf{x})$는 모호성이 없는 경우 $\nabla f(\mathbf{x})$로 대체되는 경우가 많습니다. 

$\mathbf{x}$를 $n$차원 벡터로 설정합니다. 다변량 함수를 구분할 때 다음 규칙이 자주 사용됩니다. 

* 모든 $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$에 대해,
* 모든 $\mathbf{A} \in \mathbb{R}^{n \times m}$, $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$에 대해,
* 모든 $\mathbf{A} \in \mathbb{R}^{n \times n}$, $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$에 대해,
* $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$.

마찬가지로 모든 행렬 $\mathbf{X}$에 대해 $\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$가 있습니다.나중에 살펴보겠지만 그래디언트는 딥러닝에서 최적화 알고리즘을 설계하는 데 유용합니다. 

## 체인 룰

그러나 이러한 그라디언트는 찾기가 어려울 수 있습니다.이는 딥러닝의 다변량 함수가 종종*복합적*이기 때문에 이러한 함수를 구분하기 위해 앞서 언급한 규칙을 적용하지 않을 수 있기 때문입니다.다행히도*체인 규칙*을 사용하면 복합 함수를 구분할 수 있습니다. 

먼저 단일 변수의 함수를 고려해 보겠습니다.함수 $y=f(u)$와 $u=g(x)$가 모두 미분 가능하다고 가정하면 연쇄 규칙은 다음과 같이 명시합니다. 

$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$

이제 함수에 임의의 수의 변수가 있는 좀 더 일반적인 시나리오를 살펴보겠습니다.미분 가능 함수 $y$에 변수 $u_1, u_2, \ldots, u_m$가 있으며, 여기서 각 미분 가능 함수 $u_i$에는 변수 $x_1, x_2, \ldots, x_n$이 있다고 가정합니다.$y$은 $x_1, x_2, \ldots, x_n$의 함수라는 점에 유의하십시오.그런 다음 체인 규칙이 

$$\frac{dy}{dx_i} = \frac{dy}{du_1} \frac{du_1}{dx_i} + \frac{dy}{du_2} \frac{du_2}{dx_i} + \cdots + \frac{dy}{du_m} \frac{du_m}{dx_i}$$

모든 $i = 1, 2, \ldots, n$를 위해. 

## 요약

* 미분 미적분과 적분 미적분은 미적분의 두 가지로, 전자는 딥 러닝의 유비쿼터스 최적화 문제에 적용될 수 있습니다.
* 미분은 변수에 대한 함수의 순간 변화율로 해석할 수 있습니다.또한 함수의 곡선에 대한 접선의 기울기이기도 합니다.
* 기울기는 구성 요소가 모든 변수에 대한 다변량 함수의 편도함수인 벡터입니다.
* 연쇄 규칙을 통해 복합 함수를 구분할 수 있습니다.

## 연습문제

1. $x = 1$일 때 함수 $y = f(x) = x^3 - \frac{1}{x}$와 해당 접선을 플로팅합니다.
1. 함수 $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$의 기울기를 구합니다.
1. 함수 $f(\mathbf{x}) = \|\mathbf{x}\|_2$의 기울기는 무엇입니까?
1. $u = f(x, y, z)$ 및 $x = x(a, b)$, $y = y(a, b)$ 및 $z = z(a, b)$인 경우에 대한 체인 규칙을 작성할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/32)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/33)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/197)
:end_tab:
