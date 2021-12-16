# 적분 계산법
:label:`sec_integral_calculus`

차별화는 전통적인 미적분 교육 내용의 절반에 불과합니다.다른 기둥인 통합은 “이 곡선 아래 영역은 무엇입니까?” 라는 다소 단절된 질문으로 보이기 시작합니다.겉으로는 관련이 없지만 통합은*미적분학의 기본 정리*로 알려진 것을 통한 차별화와 밀접하게 얽혀 있습니다. 

이 책에서 논의하는 기계 학습 수준에서는 통합에 대해 깊이 이해할 필요가 없습니다.그러나 나중에 접하게 될 추가 응용 프로그램의 토대를 마련하기 위해 간략한 소개를 제공 할 것입니다. 

## 기하학적 해석 $f(x)$ 함수가 있다고 가정합니다.단순화를 위해 $f(x)$가 음수가 아니라고 가정해 보겠습니다 (0보다 작은 값을 취하지 않음).우리가 시도하고 이해하고자 하는 것은 $f(x)$와 $x$축 사이에 포함된 면적은 얼마입니까?

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()

x = np.arange(-2, 2, 0.01)
f = np.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist(), f.tolist())
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from mpl_toolkits import mplot3d
import torch

x = torch.arange(-2, 2, 0.01)
f = torch.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist(), f.tolist())
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from mpl_toolkits import mplot3d
import tensorflow as tf

x = tf.range(-2, 2, 0.01)
f = tf.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.numpy(), f.numpy())
d2l.plt.show()
```

대부분의 경우이 영역은 무한하거나 정의되지 않으므로 ($f(x) = x^{2}$ 미만의 영역 고려) 사람들은 종종 한 쌍의 끝 사이의 영역 (예: $a$ 및 $b$) 에 대해 이야기합니다.

```{.python .input}
x = np.arange(-2, 2, 0.01)
f = np.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist()[50:250], f.tolist()[50:250])
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
x = torch.arange(-2, 2, 0.01)
f = torch.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist()[50:250], f.tolist()[50:250])
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
x = tf.range(-2, 2, 0.01)
f = tf.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.numpy()[50:250], f.numpy()[50:250])
d2l.plt.show()
```

이 영역을 아래의 적분 기호로 나타냅니다. 

$$
\mathrm{Area}(\mathcal{A}) = \int_a^b f(x) \;dx.
$$

내부 변수는 $\sum$의 합 인덱스와 매우 유사한 더미 변수이므로 원하는 내부 값으로 동등하게 작성할 수 있습니다. 

$$
\int_a^b f(x) \;dx = \int_a^b f(z) \;dz.
$$

이러한 적분을 근사화하는 방법을 시도하고 이해하는 전통적인 방법이 있습니다. $a$과 $b$ 사이의 영역을 가져와 $N$ 수직 슬라이스로 자르는 것을 상상할 수 있습니다.$N$가 크면 사각형으로 각 슬라이스의 면적을 근사한 다음 면적을 더하여 곡선 아래의 전체 면적을 얻을 수 있습니다.코드에서 이 작업을 수행하는 예제를 살펴보겠습니다.실제 값을 얻는 방법은 이후 섹션에서 살펴볼 것입니다.

```{.python .input}
epsilon = 0.05
a = 0
b = 2

x = np.arange(a, b, epsilon)
f = x / (1 + x**2)

approx = np.sum(epsilon*f)
true = np.log(2) / 2

d2l.set_figsize()
d2l.plt.bar(x.asnumpy(), f.asnumpy(), width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

```{.python .input}
#@tab pytorch
epsilon = 0.05
a = 0
b = 2

x = torch.arange(a, b, epsilon)
f = x / (1 + x**2)

approx = torch.sum(epsilon*f)
true = torch.log(torch.tensor([5.])) / 2

d2l.set_figsize()
d2l.plt.bar(x, f, width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

```{.python .input}
#@tab tensorflow
epsilon = 0.05
a = 0
b = 2

x = tf.range(a, b, epsilon)
f = x / (1 + x**2)

approx = tf.reduce_sum(epsilon*f)
true = tf.math.log(tf.constant([5.])) / 2

d2l.set_figsize()
d2l.plt.bar(x, f, width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

문제는 수치적으로 수행 할 수 있지만 다음과 같은 가장 간단한 함수에 대해서만 분석적으로 이러한 접근 방식을 수행 할 수 있다는 것입니다. 

$$
\int_a^b x \;dx.
$$

위 코드의 예제와 같이 좀 더 복잡한 것은 

$$
\int_a^b \frac{x}{1+x^{2}} \;dx.
$$

이러한 직접적인 방법으로 해결할 수 있는 것 이상입니다. 

대신 다른 접근 방식을 취할 것입니다.우리는 영역의 개념을 직관적으로 작업하고 적분을 찾는 데 사용되는 주요 계산 도구 인 미적분학의 기본 정리*를 배웁니다.이것이 통합 연구의 기초가 될 것입니다. 

## 미적분학의 기본 정리

통합 이론에 대해 자세히 알아보기 위해 함수를 소개하겠습니다. 

$$
F(x) = \int_0^x f(y) dy.
$$

이 함수는 $x$를 변경하는 방법에 따라 $0$와 $x$ 사이의 면적을 측정합니다.이것이 우리가 필요한 전부라는 것을 알 수 있습니다. 

$$
\int_a^b f(x) \;dx = F(b) - F(a).
$$

이것은 :numref:`fig_area-subtract`에 표시된 것처럼 먼 끝점까지 면적을 측정 한 다음 영역을 가까운 끝점까지 뺄 수 있다는 사실을 수학적으로 인코딩한 것입니다. 

![Visualizing why we may reduce the problem of computing the area under a curve between two points to computing the area to the left of a point.](../img/sub-area.svg)
:label:`fig_area-subtract`

따라서 $F(x)$가 무엇인지 파악하여 모든 구간에 걸쳐 적분이 무엇인지 파악할 수 있습니다. 

이를 위해 실험을 고려해 보겠습니다.미적분학에서 자주 하는 것처럼, 값을 조금만 움직이면 어떤 일이 일어나는지 상상해 봅시다.위의 의견에서 우리는 

$$
F(x+\epsilon) - F(x) = \int_x^{x+\epsilon} f(y) \; dy.
$$

이것은 함수의 작은 조각 아래 영역에 따라 함수가 변경된다는 것을 알려줍니다. 

이것이 우리가 근사치를 만드는 지점입니다.이와 같은 작은 영역을 보면 이 영역이 높이 값이 $f(x)$이고 밑면 너비가 $\epsilon$인 직사각형 영역에 가까운 것처럼 보입니다.실제로 $\epsilon \rightarrow 0$로서이 근사치가 점점 더 좋아진다는 것을 보여줄 수 있습니다.따라서 결론을 내릴 수 있습니다. 

$$
F(x+\epsilon) - F(x) \approx \epsilon f(x).
$$

그러나 이제 알 수 있습니다. 이것이 $F$의 도함수를 계산할 때 예상되는 패턴입니다!따라서 다음과 같은 다소 놀라운 사실을 알 수 있습니다. 

$$
\frac{dF}{dx}(x) = f(x).
$$

이것이*미적분학의 기본 정리*입니다.확장된 형태로 $\frac{d}{dx}\int_{-\infty}^x f(y) \; dy = f(x).$달러 :eqlabel:`eq_ftc`달러로 쓸 수 있습니다. 

영역을 찾는 개념 (*a priori* 다소 어려움) 을 취하고 문장 도함수 (훨씬 더 완전히 이해되는 것) 로 축소합니다.우리가해야 할 마지막 의견은 이것이 $F(x)$가 무엇인지 정확히 알려주지 않는다는 것입니다.실제로 $C$에 대한 $F(x) + C$은 동일한 파생물을 갖습니다.이것은 통합 이론에서 삶의 사실입니다.고맙게도 정적분을 사용하여 작업할 때 상수가 누락되므로 결과와 관련이 없습니다. 

$$
\int_a^b f(x) \; dx = (F(b) + C) - (F(a) + C) = F(b) - F(a).
$$

이것은 추상적이지 않은 것처럼 보일 수 있지만 잠시 시간을내어 컴퓨팅 적분에 대한 완전히 새로운 관점을 제공했음을 이해하겠습니다.우리의 목표는 더 이상 영역을 복구하기 위해 일종의 찹 앤 섬 프로세스를 수행하는 것이 아니라 파생물이 우리가 가진 함수인 함수만 찾으면 됩니다!:numref:`sec_derivative_table`의 표를 뒤집어서 다소 어려운 적분을 많이 나열 할 수 있기 때문에 이것은 놀라운 일입니다.예를 들어, 우리는 $x^{n}$의 파생물이 $nx^{n-1}$이라는 것을 알고 있습니다.따라서 기본 정리 :eqref:`eq_ftc`를 사용하여 다음과 같이 말할 수 있습니다. 

$$
\int_0^{x} ny^{n-1} \; dy = x^n - 0^n = x^n.
$$

마찬가지로 우리는 $e^{x}$의 도함수가 그 자체라는 것을 알고 있습니다. 

$$
\int_0^{x} e^{x} \; dx = e^{x} - e^{0} = e^x - 1.
$$

이런 식으로 우리는 미분 미적분학의 아이디어를 자유롭게 활용하여 전체 통합 이론을 개발할 수 있습니다.모든 통합 규칙은 이 한 가지 사실에서 파생됩니다. 

## 변수 변경
:label:`integral_example`

미분과 마찬가지로 적분 계산을 더 다루기 쉽게 만드는 여러 규칙이 있습니다.실제로, 미분 미적분학의 모든 규칙 (예: 제품 규칙, 합계 규칙 및 연쇄 규칙) 에는 적분 미적분 (부분별 적분, 적분의 선형성 및 변수 공식의 변경) 에 대한 해당 규칙이 있습니다.이 섹션에서는 목록에서 가장 중요한 요소 인 변수 수식의 변경에 대해 자세히 설명합니다. 

먼저, 그 자체가 적분인 함수가 있다고 가정하겠습니다. 

$$
F(x) = \int_0^x f(y) \; dy.
$$

$F(u(x))$를 얻기 위해 다른 함수로 함수를 작성할 때이 함수가 어떻게 보이는지 알고 싶다고 가정 해 보겠습니다.연쇄 법칙에 따르면, 우리는 

$$
\frac{d}{dx}F(u(x)) = \frac{dF}{du}(u(x))\cdot \frac{du}{dx}.
$$

위와 같이 기본 정리 :eqref:`eq_ftc`를 사용하여 이것을 통합에 대한 진술로 바꿀 수 있습니다.이것은 준다 

$$
F(u(x)) - F(u(0)) = \int_0^x \frac{dF}{du}(u(y))\cdot \frac{du}{dy} \;dy.
$$

$F$ 자체가 적분이라는 것을 상기하면 왼쪽이 다시 쓰여질 수 있습니다. 

$$
\int_{u(0)}^{u(x)} f(y) \; dy = \int_0^x \frac{dF}{du}(u(y))\cdot \frac{du}{dy} \;dy.
$$

마찬가지로 $F$가 적분이라는 것을 상기하면 기본 정리 :eqref:`eq_ftc`를 사용하여 $\frac{dF}{dx} = f$을 인식 할 수 있으므로 결론을 내릴 수 있습니다. 

$$\int_{u(0)}^{u(x)} f(y) \; dy = \int_0^x f(u(y))\cdot \frac{du}{dy} \;dy.$$
:eqlabel:`eq_change_var`

이것이*변수의 변화* 공식입니다. 

보다 직관적인 파생을 위해 $x$과 $x+\epsilon$ 사이에서 $f(u(x))$의 적분을 취할 때 어떤 일이 발생하는지 고려하십시오.작은 $\epsilon$의 경우 이 적분은 연관된 직사각형의 면적인 약 $\epsilon f(u(x))$입니다.이제 이것을 $u(x)$에서 $u(x+\epsilon)$까지 $f(y)$의 적분과 비교해 보겠습니다.우리는 $u(x+\epsilon) \approx u(x) + \epsilon \frac{du}{dx}(x)$라는 것을 알고 있습니다. 그래서 이 직사각형의 면적은 약 $\epsilon \frac{du}{dx}(x)f(u(x))$입니다.따라서이 두 직사각형의 면적이 일치하도록하려면 :numref:`fig_rect-transform`에 표시된 것처럼 첫 번째 직사각형에 $\frac{du}{dx}(x)$을 곱해야합니다. 

![Visualizing the transformation of a single thin rectangle under the change of variables.](../img/rect-trans.svg)
:label:`fig_rect-transform`

이것은 우리에게 

$$
\int_x^{x+\epsilon} f(u(y))\frac{du}{dy}(y)\;dy = \int_{u(x)}^{u(x+\epsilon)} f(y) \; dy.
$$

이것은 하나의 작은 사각형에 대해 표현된 변수 공식의 변화입니다. 

$u(x)$ 및 $f(x)$를 올바르게 선택하면 매우 복잡한 적분을 계산할 수 있습니다.예를 들어, $f(y) = 1$과 $u(x) = e^{-x^{2}}$ ($\frac{du}{dx}(x) = -2xe^{-x^{2}}$를 의미) 을 선택했다면, 예를 들어 다음과 같은 것을 보여줄 수 있습니다. 

$$
e^{-1} - 1 = \int_{e^{-0}}^{e^{-1}} 1 \; dy = -2\int_0^{1} ye^{-y^2}\;dy,
$$

따라서 그것을 재배열함으로써 

$$
\int_0^{1} ye^{-y^2}\; dy = \frac{1-e^{-1}}{2}.
$$

## 사인 규칙에 대한 의견

예리한 눈을 가진 독자는 위의 계산에 대해 이상한 것을 관찰 할 것입니다.즉, 다음과 같은 계산 

$$
\int_{e^{-0}}^{e^{-1}} 1 \; dy = e^{-1} -1 < 0,
$$

음수를 생성할 수 있습니다.영역을 생각할 때 음수 값을 보는 것이 이상 할 수 있으므로 컨벤션이 무엇인지 파헤쳐 볼 가치가 있습니다. 

수학자들은 서명 영역의 개념을 취합니다.이것은 두 가지 방식으로 나타납니다.먼저, 때때로 0보다 작은 함수 $f(x)$를 고려하면 영역도 음수가 될 것입니다.예를 들어 

$$
\int_0^{1} (-1)\;dx = -1.
$$

마찬가지로 왼쪽에서 오른쪽으로 진행되지 않고 오른쪽에서 왼쪽으로 진행되는 적분도 음수 영역으로 간주됩니다. 

$$
\int_0^{-1} 1\; dx = -1.
$$

표준 영역 (양수 함수의 왼쪽에서 오른쪽으로) 은 항상 양수입니다.뒤집어서 얻은 모든 것 (예: $x$ 축을 뒤집어 음수의 적분을 얻거나 $y$ 축을 뒤집어 잘못된 순서로 적분을 얻는 것) 은 음수 영역을 생성합니다.그리고 실제로 두 번 뒤집으면 긍정적 인 영역을 갖기 위해 상쇄되는 한 쌍의 부정적인 신호가 생깁니다. 

$$
\int_0^{-1} (-1)\;dx =  1.
$$

이 토론이 친숙하게 들리면 그렇습니다!:numref:`sec_geometry-linear-algebraic-ops`에서 행렬식이 서명된 영역을 거의 같은 방식으로 나타내는 방법에 대해 논의했습니다. 

## 다중 적분 경우에 따라 더 높은 차원에서 작업해야 할 수도 있습니다.예를 들어, $f(x, y)$과 같은 두 변수의 함수가 있고 $x$의 범위가 $[a, b]$을 초과하고 $y$의 범위가 $[c, d]$를 초과하는 경우 $f$ 미만의 부피를 알고 싶다고 가정합니다.

```{.python .input}
# Construct grid and compute function
x, y = np.meshgrid(np.linspace(-2, 2, 101), np.linspace(-2, 2, 101),
                   indexing='ij')
z = np.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

```{.python .input}
#@tab pytorch
# Construct grid and compute function
x, y = torch.meshgrid(torch.linspace(-2, 2, 101), torch.linspace(-2, 2, 101))
z = torch.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

```{.python .input}
#@tab tensorflow
# Construct grid and compute function
x, y = tf.meshgrid(tf.linspace(-2., 2., 101), tf.linspace(-2., 2., 101))
z = tf.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

우리는 이것을 다음과 같이 씁니다. 

$$
\int_{[a, b]\times[c, d]} f(x, y)\;dx\;dy.
$$

이 적분을 계산하려고 한다고 가정합니다.제 주장은 $x$에서 먼저 적분을 반복적으로 계산 한 다음 $y$의 적분으로 이동함으로써 이것을 할 수 있다는 것입니다. 

$$
\int_{[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int_c^{d} \left(\int_a^{b} f(x, y) \;dx\right) \; dy.
$$

이것이 왜 그런지 봅시다. 

함수를 $\epsilon \times \epsilon$ 사각형으로 분할 한 위의 그림을 고려하면 정수 좌표 $i, j$로 인덱싱합니다.이 경우 적분은 대략 

$$
\sum_{i, j} \epsilon^{2} f(\epsilon i, \epsilon j).
$$

문제를 불연속화하면 원하는 순서로 이러한 제곱에 값을 더할 수 있으며 값 변경에 대해 걱정할 필요가 없습니다.이것은 :numref:`fig_sum-order`에 설명되어 있습니다.특히 다음과 같이 말할 수 있습니다. 

$$
 \sum _ {j} \epsilon \left(\sum_{i} \epsilon f(\epsilon i, \epsilon j)\right).
$$

![Illustrating how to decompose a sum over many squares as a sum over first the columns (1), then adding the column sums together (2).](../img/sum-order.svg)
:label:`fig_sum-order`

내부의 합은 정확히 적분의 이산화입니다. 

$$
G(\epsilon j) = \int _a^{b} f(x, \epsilon j) \; dx.
$$

마지막으로, 이 두 표현식을 결합하면 

$$
\sum _ {j} \epsilon G(\epsilon j) \approx \int _ {c}^{d} G(y) \; dy = \int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy.
$$

따라서 모든 것을 합치면 

$$
\int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int _ c^{d} \left(\int _ a^{b} f(x, y) \;dx\right) \; dy.
$$

일단 이산화된 후에는 숫자 목록을 추가한 순서를 재정렬하는 것뿐이라는 것을 알 수 있습니다.이것은 아무것도 아닌 것처럼 보일 수 있지만, 이 결과 (*Fubini의 정리*라고 함) 는 항상 사실이 아닙니다!기계 학습 (연속 함수) 을 수행 할 때 발생하는 수학 유형에 대해서는 걱정할 필요가 없지만 실패한 예제를 만들 수 있습니다 (예: 사각형 $[0,2]\times[0,1]$ 위의 함수 $f(x, y) = xy(x^2-y^2)/(x^2+y^2)^3$). 

$x$에서 먼저 적분을 수행 한 다음 $y$에서 적분을 수행하는 선택은 임의적이었습니다.우리는 $y$를 먼저 한 다음 $x$를 보도록 똑같이 잘 선택할 수 있었을 것입니다. 

$$
\int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int _ a^{b} \left(\int _ c^{d} f(x, y) \;dy\right) \; dx.
$$

종종 벡터 표기법으로 요약하고 $U = [a, b]\times [c, d]$의 경우 다음과 같이 말합니다. 

$$
\int _ U f(\mathbf{x})\;d\mathbf{x}.
$$

## 다중 적분의 변수 변경 :eqref:`eq_change_var`의 단일 변수와 마찬가지로 더 높은 차원의 적분 내에서 변수를 변경하는 기능이 핵심 도구입니다.결과를 도출하지 않고 요약해 보겠습니다. 

통합 영역을 다시 매개변수화하는 함수가 필요합니다.우리는 이것을 $\phi : \mathbb{R}^n \rightarrow \mathbb{R}^n$로 취할 수 있습니다. 즉, $n$의 실수 변수를 취하고 또 다른 $n$을 반환하는 함수입니다.표현식을 깨끗하게 유지하기 위해 $\phi$는*주입적*이라고 가정합니다. 즉, 절대 접히지 않습니다 ($\phi(\mathbf{x}) = \phi(\mathbf{y}) \implies \mathbf{x} = \mathbf{y}$). 

이 경우 다음과 같이 말할 수 있습니다. 

$$
\int _ {\phi(U)} f(\mathbf{x})\;d\mathbf{x} = \int _ {U} f(\phi(\mathbf{x})) \left|\det(D\phi(\mathbf{x}))\right|\;d\mathbf{x}.
$$

여기서 $D\phi$는 $\phi$의*야코비안*이며, 이는 $\boldsymbol{\phi} = (\phi_1(x_1, \ldots, x_n), \ldots, \phi_n(x_1, \ldots, x_n))$의 편도함수 행렬입니다. 

$$
D\boldsymbol{\phi} = \begin{bmatrix}
\frac{\partial \phi _ 1}{\partial x _ 1} & \cdots & \frac{\partial \phi _ 1}{\partial x _ n} \\
\vdots & \ddots & \vdots \\
\frac{\partial \phi _ n}{\partial x _ 1} & \cdots & \frac{\partial \phi _ n}{\partial x _ n}
\end{bmatrix}.
$$

자세히 살펴보면 $\frac{du}{dx}(x)$이라는 용어를 $\left|\det(D\phi(\mathbf{x}))\right|$로 대체했다는 점을 제외하면 단일 가변 체인 규칙 :eqref:`eq_change_var`와 유사하다는 것을 알 수 있습니다.이 용어를 어떻게 해석할 수 있는지 봅시다.$\frac{du}{dx}(x)$ 용어는 $u$를 적용하여 $x$ 축을 얼마나 늘렸는지 말하기 위해 존재했음을 상기하십시오.더 높은 차원에서 동일한 프로세스는 $\boldsymbol{\phi}$를 적용하여 작은 사각형 (또는 작은*하이퍼 큐브*) 의 영역 (또는 볼륨 또는 하이퍼 볼륨) 을 얼마나 늘릴지 결정하는 것입니다.$\boldsymbol{\phi}$가 행렬에 의한 곱셈이라면 행렬식이 이미 어떻게 답을 제공하는지 알 수 있습니다. 

일부 작업을 통해*Jacobian*이 도함수 및 기울기를 사용하여 선이나 평면으로 근사 할 수있는 것과 같은 방식으로 행렬에 의한 점에서 다변수 함수 $\boldsymbol{\phi}$에 가장 적합한 근사를 제공한다는 것을 보여줄 수 있습니다.따라서 야코비 행렬의 행렬식은 한 차원에서 식별한 스케일링 인자를 정확하게 반영합니다. 

여기에 세부 사항을 채우려면 약간의 작업이 필요하므로 지금은 명확하지 않더라도 걱정하지 마십시오.나중에 사용할 예제를 하나 이상 살펴 보겠습니다.적분을 고려하십시오. 

$$
\int _ {-\infty}^{\infty} \int _ {-\infty}^{\infty} e^{-x^{2}-y^{2}} \;dx\;dy.
$$

이 적분을 직접 가지고 노는 것은 어디에도 갈 수 없지만 변수를 변경하면 상당한 진전을 이룰 수 있습니다.$\boldsymbol{\phi}(r, \theta) = (r \cos(\theta),  r\sin(\theta))$ (즉, $x = r \cos(\theta)$, $y = r \sin(\theta)$) 를 허용하면 변수 공식의 변경을 적용하여 이것이 다음과 동일하다는 것을 알 수 있습니다. 

$$
\int _ 0^\infty \int_0 ^ {2\pi} e^{-r^{2}} \left|\det(D\mathbf{\phi}(\mathbf{x}))\right|\;d\theta\;dr,
$$

여기서 

$$
\left|\det(D\mathbf{\phi}(\mathbf{x}))\right| = \left|\det\begin{bmatrix}
\cos(\theta) & -r\sin(\theta) \\
\sin(\theta) & r\cos(\theta)
\end{bmatrix}\right| = r(\cos^{2}(\theta) + \sin^{2}(\theta)) = r.
$$

따라서 적분은 다음과 같습니다. 

$$
\int _ 0^\infty \int _ 0 ^ {2\pi} re^{-r^{2}} \;d\theta\;dr = 2\pi\int _ 0^\infty re^{-r^{2}} \;dr = \pi,
$$

여기서 최종 등식은 섹션 :numref:`integral_example`에서 사용한 것과 동일한 계산이 이어집니다. 

:numref:`sec_random_variables`에서 연속 확률 변수를 연구 할 때이 적분을 다시 만날 것입니다. 

## 요약

* 통합 이론을 통해 영역이나 볼륨에 대한 질문에 답할 수 있습니다.
* 미적분의 기본 정리를 통해 도함수에 대한 지식을 활용하여 특정 지점까지의 영역의 미분이 통합되는 함수의 값에 의해 주어진다는 관찰을 통해 영역을 계산할 수 있습니다.
* 더 높은 차원의 적분은 단일 변수 적분을 반복하여 계산할 수 있습니다.

## 연습 문제 1.$\int_1^2 \frac{1}{x} \;dx$이란 무엇입니까?2.변수 변경 공식을 사용하여 $\int_0^{\sqrt{\pi}}x\sin(x^2)\;dx$을 적분합니다.$\int_{[0,1]^2} xy \;dx\;dy$란 무엇입니까?4.변수 변경 공식을 사용하여 $\int_0^2\int_0^1xy(x^2-y^2)/(x^2+y^2)^3\;dy\;dx$ 및 $\int_0^1\int_0^2f(x, y) = xy(x^2-y^2)/(x^2+y^2)^3\;dx\;dy$을 계산하여 변수가 다른지 확인합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/414)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1092)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1093)
:end_tab:
