# 단일 변수 미적분
:label:`sec_single_variable_calculus`

:numref:`sec_calculus`에서 우리는 미분 미적분학의 기본 요소를 보았습니다.이 섹션에서는 미적분학의 기초와 이를 기계 학습의 맥락에서 이해하고 적용하는 방법에 대해 자세히 설명합니다. 

## 미분 미적분학은 근본적으로 작은 변화에서 함수가 어떻게 행동하는지에 대한 연구입니다.이것이 딥 러닝의 핵심인 이유를 알아보기 위해 예를 들어 보겠습니다. 

편의를 위해 가중치가 단일 벡터 $\mathbf{w} = (w_1, \ldots, w_n)$로 연결된 심층 신경망이 있다고 가정합니다.훈련 데이터 세트가 주어지면 이 데이터 세트에서 신경망의 손실을 고려합니다. 이 데이터 세트는 $\mathcal{L}(\mathbf{w})$로 작성합니다.   

이 함수는 이 데이터셋에서 주어진 아키텍처의 가능한 모든 모델의 성능을 인코딩하는 매우 복잡하므로 어떤 가중치 세트 $\mathbf{w}$가 손실을 최소화할 수 있는지 확인하는 것은 거의 불가능합니다.따라서 실제로는 가중치를*무작위로* 초기화하는 것으로 시작한 다음 손실을 최대한 빨리 감소시키는 방향으로 작은 단계를 반복적으로 수행합니다. 

그런 다음 문제는 표면에서 더 쉽지 않은 것이 됩니다. 가중치를 최대한 빨리 줄이는 방향을 어떻게 찾을 수 있을까요?이를 파헤 치기 위해 먼저 단일 가중치로 사례를 살펴 보겠습니다. 단일 실수 값 $x$에 대해 $L(\mathbf{w}) = L(x)$입니다.  

$x$를 취하여 $x + \epsilon$로 소량 변경하면 어떤 일이 발생하는지 이해하려고 노력합시다.구체적으로 말하고 싶다면 $\epsilon = 0.0000001$과 같은 숫자를 생각해보십시오.어떤 일이 발생하는지 시각화하는 데 도움이 되도록 $[0, 3]$에 대한 예제 함수 $f(x) = \sin(x^x)$을 그래프로 표시해 보겠습니다.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Plot a function in a normal range
x_big = np.arange(0.01, 3.01, 0.01)
ys = np.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Define pi in torch

# Plot a function in a normal range
x_big = torch.arange(0.01, 3.01, 0.01)
ys = torch.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # Define pi in TensorFlow

# Plot a function in a normal range
x_big = tf.range(0.01, 3.01, 0.01)
ys = tf.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

이 대규모에서는 함수의 동작이 간단하지 않습니다.그러나 범위를 $[1.75,2.25]$와 같이 더 작게 줄이면 그래프가 훨씬 단순해지는 것을 알 수 있습니다.

```{.python .input}
# Plot a the same function in a tiny range
x_med = np.arange(1.75, 2.25, 0.001)
ys = np.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Plot a the same function in a tiny range
x_med = torch.arange(1.75, 2.25, 0.001)
ys = torch.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# Plot a the same function in a tiny range
x_med = tf.range(1.75, 2.25, 0.001)
ys = tf.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

극단적으로 보면, 작은 세그먼트로 확대하면 동작이 훨씬 간단해집니다. 단지 직선일 뿐입니다.

```{.python .input}
# Plot a the same function in a tiny range
x_small = np.arange(2.0, 2.01, 0.0001)
ys = np.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Plot a the same function in a tiny range
x_small = torch.arange(2.0, 2.01, 0.0001)
ys = torch.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# Plot a the same function in a tiny range
x_small = tf.range(2.0, 2.01, 0.0001)
ys = tf.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

이것은 단일 변수 미적분학의 주요 관찰입니다. 익숙한 함수의 동작은 충분히 작은 범위의 선으로 모델링 할 수 있습니다.즉, 대부분의 함수에서 함수의 $x$ 값을 약간 이동하면 출력 $f(x)$도 약간 이동할 것으로 예상하는 것이 합리적입니다.우리가 대답해야 할 유일한 질문은 “입력의 변화에 비해 출력의 변화가 얼마나 큰가요?절반 정도 크나요?두 배 더 커요?” 

따라서 함수 입력의 작은 변화에 대한 함수 출력의 변화 비율을 고려할 수 있습니다.이것을 공식적으로 다음과 같이 쓸 수 있습니다. 

$$
\frac{L(x+\epsilon) - L(x)}{(x+\epsilon) - x} = \frac{L(x+\epsilon) - L(x)}{\epsilon}.
$$

이것은 이미 코드를 가지고 놀기 시작하기에 충분합니다.예를 들어, $L(x) = x^{2} + 1701(x-4)^3$를 알고 있다고 가정하면 다음과 같이 $x = 4$ 지점에서 이 값이 얼마나 큰지 알 수 있습니다.

```{.python .input}
#@tab all
# Define our function
def L(x):
    return x**2 + 1701*(x-4)**3

# Print the difference divided by epsilon for several epsilon
for epsilon in [0.1, 0.001, 0.0001, 0.00001]:
    print(f'epsilon = {epsilon:.5f} -> {(L(4+epsilon) - L(4)) / epsilon:.5f}')
```

이제 관찰력이 있다면이 숫자의 출력이 의심스럽게 $8$에 가깝다는 것을 알 수 있습니다.실제로 $\epsilon$를 줄이면 가치가 $8$에 점차 가까워지는 것을 볼 수 있습니다.따라서 우리가 추구하는 값 (입력의 변화가 출력을 변경하는 정도) 이 $x=4$ 지점에서 $8$이어야한다는 결론을 내릴 수 있습니다.수학자가 이 사실을 부호화하는 방법은 

$$
\lim_{\epsilon \rightarrow 0}\frac{L(4+\epsilon) - L(4)}{\epsilon} = 8.
$$

약간의 역사적 여담으로: 신경망 연구의 처음 수십 년 동안 과학자들은이 알고리즘 (* 유한 차이의 방법*) 을 사용하여 작은 섭동 하에서 손실 함수가 어떻게 변했는지 평가했습니다. 가중치를 변경하고 손실이 어떻게 변했는지 확인하십시오.이는 계산적으로 비효율적이며, 한 변수의 단일 변경이 손실에 어떤 영향을 미치는지 확인하기 위해 손실 함수를 두 번 평가해야 합니다.몇 천 개의 매개 변수로도 이 작업을 수행하려고 시도했다면 전체 데이터 세트에 대해 수천 건의 네트워크 평가가 필요합니다!1986년까지 :cite:`Rumelhart.Hinton.Williams.ea.1988`에 도입된*역전파 알고리즘*이 가중치의*모든* 변화가 데이터 세트를 통한 네트워크의 단일 예측과 동일한 계산 시간에 손실을 어떻게 변화시키는지 계산하는 방법을 제공했다는 것은 해결되지 않았습니다. 

이 예에서 이 값 $8$는 $x$의 값에 따라 다르므로 $x$의 함수로 정의하는 것이 좋습니다.보다 공식적으로, 이 값 종속 변화율은*미분*이라고 하며, 이는 다음과 같이 작성됩니다. 

$$\frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon}.$$
:eqlabel:`eq_der_def`

텍스트마다 파생물에 대해 서로 다른 표기법을 사용합니다.예를 들어, 아래 표기법은 모두 같은 것을 나타냅니다. 

$$
\frac{df}{dx} = \frac{d}{dx}f = f' = \nabla_xf = D_xf = f_x.
$$

대부분의 저자는 단일 표기법을 선택하여 고수하지만 보장되지는 않습니다.이 모든 것에 익숙해지는 것이 가장 좋습니다.복잡한 표현식의 도함수를 사용하지 않는 한이 텍스트 전체에서 $\frac{df}{dx}$ 표기법을 사용합니다. 이 경우 $\frac{d}{dx}f$를 사용하여 $$\ frac {d} {dx}\ left [x^4+\ cos\ left (\ frac {x^2+1} {2x-1}\ 오른쪽)\ 오른쪽].$$ 

종종 도함수 :eqref:`eq_der_def`의 정의를 다시 풀어서 $x$를 약간 변경할 때 함수가 어떻게 변하는지 확인하는 것이 직관적으로 유용합니다. 

$$\begin{aligned} \frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon} & \implies \frac{df}{dx}(x) \approx \frac{f(x+\epsilon) - f(x)}{\epsilon} \\ & \implies \epsilon \frac{df}{dx}(x) \approx f(x+\epsilon) - f(x) \\ & \implies f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x). \end{aligned}$$
:eqlabel:`eq_small_change`

마지막 방정식은 명시적으로 호출할 가치가 있습니다.함수를 취하여 입력값을 조금만 변경하면 파생물에 의해 스케일링된 작은 양만큼 출력이 변한다는 것을 알 수 있습니다. 

이런 식으로 미분을 입력 변경으로 인해 출력에서 얼마나 큰 변화가 발생하는지 알려주는 스케일링 인자로 이해할 수 있습니다. 

## 미적분학 규칙
:label:`sec_derivative_table`

이제 명시적 함수의 도함수를 계산하는 방법을 이해하는 작업으로 넘어갑니다.미적분학의 완전한 공식적인 처리는 첫 번째 원칙에서 모든 것을 도출합니다.우리는 여기서 이 유혹에 빠지지 않고 오히려 직면하게 되는 일반적인 규칙에 대한 이해를 제공할 것입니다. 

### 공통 도함수 :numref:`sec_calculus`에서 볼 수 있듯이, 도함수를 계산할 때 종종 일련의 규칙을 사용하여 계산을 몇 가지 핵심 함수로 줄일 수 있습니다.쉽게 참조 할 수 있도록 여기에서 반복합니다. 

* **상수의 도함.** $\frac{d}{dx}c = 0$.
* **선형 함수의 도함수** $\frac{d}{dx}(ax) = a$.
* **전원 규칙.** $\frac{d}{dx}x^n = nx^{n-1}$.
* **지수의 도함수.** $\frac{d}{dx}e^x = e^x$.
* **로그의 도함수.** $\frac{d}{dx}\log(x) = \frac{1}{x}$.

### 미분 규칙 모든 파생물을 별도로 계산하여 테이블에 저장해야 하는 경우 미분 미적분은 거의 불가능합니다.위의 도함수를 일반화하고 $f(x) = \log\left(1+(x-1)^{10}\right)$의 도함수를 찾는 것과 같은 더 복잡한 도함수를 계산할 수 있다는 것은 수학의 선물입니다.:numref:`sec_calculus`에서 언급했듯이, 그렇게하는 열쇠는 기능을 수행하고 다양한 방법, 가장 중요한 것은 합계, 제품 및 구성으로 결합 할 때 발생하는 일을 성문화하는 것입니다. 

* **합계 규칙.** $\frac{d}{dx}\left(g(x) + h(x)\right) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$.
* **제품 규칙.** $\frac{d}{dx}\left(g(x)\cdot h(x)\right) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$.
* **체인 규칙.** $\frac{d}{dx}g(h(x)) = \frac{dg}{dh}(h(x))\cdot \frac{dh}{dx}(x)$.

이러한 규칙을 이해하기 위해 :eqref:`eq_small_change`를 어떻게 사용할 수 있는지 살펴 보겠습니다.합계 규칙의 경우 다음과 같은 추론 체인을 고려하십시오. 

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon) + h(x+\epsilon) \\
& \approx g(x) + \epsilon \frac{dg}{dx}(x) + h(x) + \epsilon \frac{dh}{dx}(x) \\
& = g(x) + h(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right) \\
& = f(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right).
\end{aligned}
$$

이 결과를 $f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x)$이라는 사실과 비교하면 $\frac{df}{dx}(x) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$가 원하는대로 알 수 있습니다.여기서 직관은 다음과 같습니다. 입력을 변경할 때 $x$, $g$ 및 $h$이 공동으로 $\frac{dg}{dx}(x)$ 및 $\frac{dh}{dx}(x)$에 의한 출력 변경에 기여합니다. 

제품은 더 미묘하며 이러한 표현식으로 작업하는 방법에 대한 새로운 관찰이 필요합니다.:eqref:`eq_small_change`를 사용하기 전과 같이 시작할 것입니다. 

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon)\cdot h(x+\epsilon) \\
& \approx \left(g(x) + \epsilon \frac{dg}{dx}(x)\right)\cdot\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& = g(x)\cdot h(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x) \\
& = f(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x). \\
\end{aligned}
$$

이것은 위에서 수행 한 계산과 유사하며 실제로 우리의 대답 ($\frac{df}{dx}(x) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$) 이 $\epsilon$ 옆에 있지만 크기 $\epsilon^{2}$의 용어에 대한 문제가 있습니다.$\epsilon^2$의 거듭제곱이 $\epsilon^1$의 거듭제곱보다 높기 때문에*고차 항*이라고 합니다.이후 섹션에서 때때로 이러한 사항을 추적하고 싶을 것입니다. 그러나 지금은 $\epsilon = 0.0000001$이면 $\epsilon^{2}= 0.0000000000001$가 훨씬 작다는 것을 관찰하십시오.$\epsilon \rightarrow 0$을 보낼 때 더 높은 주문 조건을 무시할 수 있습니다.이 부록의 일반적인 규칙으로 “$\approx$"을 사용하여 두 항이 고차 항과 동일하다는 것을 나타냅니다.그러나 좀 더 공식적이기를 원한다면 차이 지수를 살펴볼 수 있습니다. 

$$
\frac{f(x+\epsilon) - f(x)}{\epsilon} = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x) + \epsilon \frac{dg}{dx}(x)\frac{dh}{dx}(x),
$$

그리고 우리가 $\epsilon \rightarrow 0$를 보낼 때 오른쪽 항도 0이 되는 것을 볼 수 있습니다. 

마지막으로 연쇄 규칙을 사용하면 :eqref:`eq_small_change`를 사용하기 전과 같이 다시 진행할 수 있습니다. 

$$
\begin{aligned}
f(x+\epsilon) & = g(h(x+\epsilon)) \\
& \approx g\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& \approx g(h(x)) + \epsilon \frac{dh}{dx}(x) \frac{dg}{dh}(h(x))\\
& = f(x) + \epsilon \frac{dg}{dh}(h(x))\frac{dh}{dx}(x),
\end{aligned}
$$

여기서 두 번째 줄에서는 함수 $g$가 입력 ($h(x)$) 이 소량 $\epsilon \frac{dh}{dx}(x)$만큼 이동된 것으로 봅니다. 

이러한 규칙은 기본적으로 원하는 표현식을 계산할 수 있는 유연한 도구 세트를 제공합니다.예를 들어, 

$$
\begin{aligned}
\frac{d}{dx}\left[\log\left(1+(x-1)^{10}\right)\right] & = \left(1+(x-1)^{10}\right)^{-1}\frac{d}{dx}\left[1+(x-1)^{10}\right]\\
& = \left(1+(x-1)^{10}\right)^{-1}\left(\frac{d}{dx}[1] + \frac{d}{dx}[(x-1)^{10}]\right) \\
& = \left(1+(x-1)^{10}\right)^{-1}\left(0 + 10(x-1)^9\frac{d}{dx}[x-1]\right) \\
& = 10\left(1+(x-1)^{10}\right)^{-1}(x-1)^9 \\
& = \frac{10(x-1)^9}{1+(x-1)^{10}}.
\end{aligned}
$$

여기서 각 행은 다음 규칙을 사용했습니다. 

1. 로그의 연쇄 규칙과 도함수.
2. 합계 규칙입니다.
3. 상수, 연쇄 규칙 및 검정력 규칙의 도함수입니다.
4. 합 규칙, 선형 함수의 도함수, 상수의 도함수.

이 예제를 수행한 후에는 다음 두 가지 사항이 명확해야 합니다. 

1. 합, 곱, 상수, 거듭제곱, 지수 및 로그를 사용하여 기록할 수 있는 모든 함수는 다음 규칙에 따라 파생물을 기계적으로 계산할 수 있습니다.
2. 인간이 이러한 규칙을 따르는 것은 지루하고 오류가 발생하기 쉽습니다!

고맙게도, 이 두 가지 사실은 함께 앞으로 나아갈 길을 암시합니다. 이것은 기계화를 위한 완벽한 후보입니다!실제로 이 섹션의 뒷부분에서 다시 살펴볼 역전파가 바로 그것입니다. 

### 선형 근사 도함수를 사용할 때 위에서 사용한 근사값을 기하학적으로 해석하는 것이 유용한 경우가 많습니다.특히 다음 방정식에 유의하십시오.  

$$
f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x),
$$

는 점 $(x, f(x))$을 통과하고 경사가 $\frac{df}{dx}(x)$인 선으로 $f$의 값을 근사화합니다.이러한 방식으로 미분은 아래 그림과 같이 함수 $f$에 선형 근사를 제공한다고 말합니다.

```{.python .input}
# Compute sin
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Compute some linear approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Compute sin
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Compute some linear approximations. Use d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) * 
                 torch.cos(torch.tensor(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# Compute sin
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]

# Compute some linear approximations. Use d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) * 
                 tf.cos(tf.constant(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

### 고차 파생상품

이제 표면에 이상하게 보일 수 있는 일을 해봅시다.함수 $f$를 취하고 도함수 $\frac{df}{dx}$를 계산합니다.이것은 어느 시점에서든 $f$의 변화율을 제공합니다. 

그러나 도함수 $\frac{df}{dx}$는 함수 자체로 볼 수 있으므로 $\frac{df}{dx}$의 도함수를 계산하여 $\frac{d^2f}{dx^2} = \frac{df}{dx}\left(\frac{df}{dx}\right)$을 얻는 것을 막을 수있는 것은 없습니다.우리는 이것을 $f$의 두 번째 파생물이라고 부를 것입니다.이 함수는 변화율 $f$의 변화율, 즉 변화율이 어떻게 변하는 지에 대한 변화율입니다.당사는 $n$번째 파생물을 얻기 위해 파생물을 여러 번 적용할 수 있습니다.표기법을 깨끗하게 유지하기 위해 $n$번째 도함수를 다음과 같이 표시합니다.  

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$

*왜*이것이 유용한 개념인지 이해하려고 노력합시다.아래에서는 $f^{(2)}(x)$, $f^{(1)}(x)$ 및 $f(x)$을 시각화합니다.   

먼저, 두 번째 도함수 $f^{(2)}(x)$가 양의 상수인 경우를 고려하십시오.이는 1차 도함수의 기울기가 양수임을 의미합니다.결과적으로, 1차 도함수 $f^{(1)}(x)$은 음수에서 시작하여 한 지점에서 0이 된 다음 결국 양수가 될 수 있습니다.이것은 원래 함수 $f$의 기울기를 알려주므로 함수 $f$ 자체가 감소하고 평평 해졌다가 증가합니다.즉, 함수 $f$은 위로 곡선을 이루며 :numref:`fig_positive-second`에 표시된 것처럼 단일 최솟값을 갖습니다. 

![If we assume the second derivative is a positive constant, then the fist derivative in increasing, which implies the function itself has a minimum.](../img/posSecDer.svg)
:label:`fig_positive-second`

둘째, 두 번째 도함수가 음의 상수이면 첫 번째 도함수가 감소하고 있음을 의미합니다.이는 1차 도함수가 양수에서 시작하여 한 지점에서 0이 된 다음 음수가 될 수 있음을 의미합니다.따라서 함수 $f$ 자체가 증가하고 평평 해졌다가 감소합니다.즉, 함수 $f$는 아래로 구부러지며 :numref:`fig_negative-second`에 표시된 것처럼 단일 최대값을 갖습니다. 

![If we assume the second derivative is a negative constant, then the fist derivative in decreasing, which implies the function itself has a maximum.](../img/negSecDer.svg)
:label:`fig_negative-second`

셋째, 두 번째 도함수가 항상 0이면 첫 번째 도함수가 변경되지 않고 일정합니다!즉, $f$는 고정 금리로 증가 (또는 감소) 하고 $f$는 :numref:`fig_zero-second`에 표시된 것처럼 그 자체가 직선임을 의미합니다. 

![If we assume the second derivative is zero, then the fist derivative is constant, which implies the function itself is a straight line.](../img/zeroSecDer.svg)
:label:`fig_zero-second`

요약하면, 2차 도함수는 함수 $f$가 구부러지는 방식을 설명하는 것으로 해석할 수 있습니다.양의 제 2 미분은 상향 곡선으로 이어지는 반면, 음의 제 2 미분은 $f$가 아래쪽으로 휘어지는 것을 의미하고, 제로 제 2 도함수는 $f$가 전혀 곡선을 이루지 않음을 의미합니다. 

한 걸음 더 나아갑시다.함수 $g(x) = ax^{2}+ bx + c$를 고려해 보십시오.그런 다음 계산할 수 있습니다. 

$$
\begin{aligned}
\frac{dg}{dx}(x) & = 2ax + b \\
\frac{d^2g}{dx^2}(x) & = 2a.
\end{aligned}
$$

원래 함수 $f(x)$을 염두에 두고 있다면 처음 두 도함수를 계산하고 $a, b$ 및 $c$에 대한 값을 구할 수 있습니다.첫 번째 도함수가 직선으로 최상의 근사를 제공한다는 것을 본 이전 섹션과 마찬가지로이 구성은 2 차원으로 최상의 근사치를 제공합니다.$f(x) = \sin(x)$에 대해 이것을 시각화해 보겠습니다.

```{.python .input}
# Compute sin
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0) -
                              (xs - x0)**2 * np.sin(x0) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Compute sin
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) * 
                 torch.cos(torch.tensor(x0)) - (xs - x0)**2 *
                 torch.sin(torch.tensor(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# Compute sin
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) * 
                 tf.cos(tf.constant(x0)) - (xs - x0)**2 *
                 tf.sin(tf.constant(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

이 아이디어를 다음 섹션에서*Taylor 시리즈*의 아이디어로 확장하겠습니다.  

### 테일러 시리즈

*테일러 시리즈*는 점 $x_0$, 즉 $\left\{ f(x_0), f^{(1)}(x_0), f^{(2)}(x_0), \ldots, f^{(n)}(x_0) \right\}$에서 첫 번째 $n$ 도함수에 대한 값이 주어지면 함수 $f(x)$를 근사화하는 방법을 제공합니다.아이디어는 $x_0$에서 주어진 모든 도함수와 일치하는 차수 $n$을 찾는 것입니다. 

이전 섹션에서 $n=2$의 경우를 보았고 약간의 대수학은 이것이 

$$
f(x) \approx \frac{1}{2}\frac{d^2f}{dx^2}(x_0)(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

위에서 볼 수 있듯이 $2$의 분모는 $x^2$의 두 도함수를 취할 때 얻는 $2$를 취소하고 다른 항은 모두 0입니다.1차 도함수와 값 자체에도 동일한 논리가 적용됩니다. 

논리를 $n=3$로 더 밀어 넣으면 다음과 같은 결론을 내릴 것입니다. 

$$
f(x) \approx \frac{\frac{d^3f}{dx^3}(x_0)}{6}(x-x_0)^3 + \frac{\frac{d^2f}{dx^2}(x_0)}{2}(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

여기서 6 달러 = 3\ 곱하기 2 = 3!$ comes from the constant we get in front if we take three derivatives of $x^3$. 

또한 다음과 같이 $n$도 다항식을 얻을 수 있습니다.  

$$
P_n(x) = \sum_{i = 0}^{n} \frac{f^{(i)}(x_0)}{i!}(x-x_0)^{i}.
$$

여기서 표기법은  

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$

실제로 $P_n(x)$는 함수 $f(x)$에 대한 최상의 $n$차 다항식 근사로 볼 수 있습니다. 

위의 근사치의 오차에 대해 자세히 설명하지는 않겠지 만 무한 한계를 언급 할 가치가 있습니다.이 경우 $\cos(x)$ 또는 $e^{x}$와 같이 잘 작동하는 함수 (실제 분석 함수라고 함) 의 경우 무한한 수의 항을 작성하고 정확히 동일한 함수를 근사할 수 있습니다. 

$$
f(x) = \sum_{n = 0}^\infty \frac{f^{(n)}(x_0)}{n!}(x-x_0)^{n}.
$$

$f(x) = e^{x}$를 예로 들어 보겠습니다.$e^{x}$은 자체 파생 상품이기 때문에 $f^{(n)}(x) = e^{x}$라는 것을 알고 있습니다.따라서 $e^{x}$은 테일러 시리즈를 $x_0 = 0$에서 취함으로써 재구성 될 수 있습니다. 

$$
e^{x} = \sum_{n = 0}^\infty \frac{x^{n}}{n!} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots.
$$

이것이 코드에서 어떻게 작동하는지 살펴보고 Taylor 근사의 정도를 높이면 원하는 함수 $e^x$에 더 가까워지는 방법을 관찰해 보겠습니다.

```{.python .input}
# Compute the exponential function
xs = np.arange(0, 3, 0.01)
ys = np.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab pytorch
# Compute the exponential function
xs = torch.arange(0, 3, 0.01)
ys = torch.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab tensorflow
# Compute the exponential function
xs = tf.range(0, 3, 0.01)
ys = tf.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

Taylor 시리즈에는 두 가지 주요 응용 프로그램이 

1. *이론적 응용*: 종종 너무 복잡한 함수를 이해하려고 할 때 Taylor 시리즈를 사용하면 직접 작업 할 수있는 다항식으로 변환 할 수 있습니다.

2. *수치 적용*: $e^{x}$ 또는 $\cos(x)$와 같은 일부 함수는 기계가 계산하기가 어렵습니다.값 테이블을 고정 정밀도로 저장할 수 있지만 (이 작업은 종종 수행됨) “$\cos(1)$의 1000번째 숫자는 무엇입니까?” 와 같은 열린 질문은 여전히 남습니다.Taylor 시리즈는 종종 이러한 질문에 답하는 데 도움이됩니다.  

## 요약

* 미분은 입력을 소량 변경할 때 함수가 어떻게 변하는지를 표현하는 데 사용할 수 있습니다.
* 미분 규칙을 사용하여 기본 도함수를 결합하여 임의로 복잡한 도함수를 만들 수 있습니다.
* 2차 또는 고차 도함수를 얻기 위해 도함수를 반복할 수 있습니다.순서가 증가할 때마다 함수의 동작에 대한 보다 세밀한 정보가 제공됩니다.
* 단일 데이터 예제의 도함수에서 정보를 사용하여 Taylor 시리즈에서 얻은 다항식으로 잘 동작하는 함수를 근사 할 수 있습니다.

## 연습문제

1. $x^3-4x+1$의 파생물은 무엇입니까?
2. $\log(\frac{1}{x})$의 파생물은 무엇입니까?
3. 참 또는 거짓: $f'(x) = 0$인 경우 $f$의 최대값 또는 최소값이 $x$입니까?
4. $x\ge0$에 대한 최소값 $f(x) = x\log(x)$은 어디에 있습니까 (여기서 $f$가 $f(0)$에서 $0$의 한계 값을 취한다고 가정)?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/412)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1088)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1089)
:end_tab:
