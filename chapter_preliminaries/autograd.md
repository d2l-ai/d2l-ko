# 자동 차별화
:label:`sec_autograd`

:numref:`sec_calculus`에서 설명했듯이 차별화는 거의 모든 딥 러닝 최적화 알고리즘에서 중요한 단계입니다.이러한 도함수를 취하기위한 계산은 간단하지만 몇 가지 기본 미적분학 만 필요하지만 복잡한 모델의 경우 수동으로 업데이트를 수행하는 것은 고통 스러울 수 있습니다 (종종 오류가 발생하기 쉽습니다).

딥 러닝 프레임워크는 도함수 (예: *자동 미분*) 를 자동으로 계산하여 이 작업을 신속하게 처리합니다.실제로 설계된 모델을 기반으로 시스템은*계산 그래프*를 구축하여 어떤 데이터를 결합하여 어떤 작업을 통해 출력을 생성하는지 추적합니다.자동 미분을 통해 시스템은 그라디언트를 역전파할 수 있습니다.여기서*backpropagate*는 단순히 계산 그래프를 통해 추적하여 각 매개 변수에 대한 편도함수를 채우는 것을 의미합니다.

## 간단한 예제

장난감 예로, (**열 벡터 $\mathbf{x}$.에 대해 함수 $y = 2\mathbf{x}^{\top}\mathbf{x}$을 구별하기**) 에 관심이 있다고 가정해 보겠습니다. 시작하려면 변수 `x`를 만들고 초기 값을 할당해 보겠습니다.

```{.python .input}
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(4.0)
x
```

```{.python .input}
#@tab pytorch
import torch

x = torch.arange(4.0)
x
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.range(4, dtype=tf.float32)
x
```

[**$\mathbf{x}$와 관련하여 $y$의 기울기를 계산하기 전에 저장할 장소가 필요합니다.**] 동일한 매개 변수를 수천 또는 수백만 번 업데이트하는 경우가 많기 때문에 매개 변수와 관련하여 파생물을 취할 때마다 새 메모리를 할당하지 않는 것이 중요합니다.메모리가 빨리 부족할 수 있습니다.벡터 $\mathbf{x}$에 대한 스칼라 값 함수의 기울기는 그 자체로 벡터 값이며 $\mathbf{x}$와 모양이 같습니다.

```{.python .input}
# We allocate memory for a tensor's gradient by invoking `attach_grad`
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
x.grad
```

```{.python .input}
#@tab pytorch
x.requires_grad_(True)  # Same as `x = torch.arange(4.0, requires_grad=True)`
x.grad  # The default value is None
```

```{.python .input}
#@tab tensorflow
x = tf.Variable(x)
```

(**이제 $y$.를 계산해 보겠습니다. **)

```{.python .input}
# Place our code inside an `autograd.record` scope to build the computational
# graph
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input}
#@tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
#@tab tensorflow
# Record all computations onto a tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

`x`는 길이가 4인 벡터이므로 `x` 및 `x`의 내적이 수행되어 `y`에 할당하는 스칼라 출력값이 생성됩니다.다음으로 역전파 함수를 호출하고 그래디언트를 인쇄하여 `x`의 각 구성 요소에 대한 `y`의 기울기를 자동으로 계산할 수 있습니다.

```{.python .input}
y.backward()
x.grad
```

```{.python .input}
#@tab pytorch
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

(**$\mathbf{x}$에 대한 함수 $y = 2\mathbf{x}^{\top}\mathbf{x}$의 기울기는 $4\mathbf{x}$.이어야 합니다.**) 원하는 기울기가 올바르게 계산되었는지 신속하게 확인하겠습니다.

```{.python .input}
x.grad == 4 * x
```

```{.python .input}
#@tab pytorch
x.grad == 4 * x
```

```{.python .input}
#@tab tensorflow
x_grad == 4 * x
```

[**이제 `x`.의 다른 함수를 계산해 보겠습니다.**]

```{.python .input}
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Overwritten by the newly calculated gradient
```

```{.python .input}
#@tab pytorch
# PyTorch accumulates the gradient in default, we need to clear the previous
# values
x.grad.zero_()
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
```

## 스칼라 변수가 아닌 경우 역방향

기술적으로 `y`가 스칼라가 아닌 경우 벡터 `x`에 대한 벡터 `y`의 미분을 가장 자연스럽게 해석하는 것은 행렬입니다.고차 및 고차원 `y` 및 `x`의 경우 차별화 결과는 고차 텐서가 될 수 있습니다.

그러나 이러한 이국적인 객체는 고급 기계 학습 ([**딥 러닝에서**] 포함) 에 나타나지만, 더 자주 (**벡터를 역방향으로 호출 할 때**) 훈련 예제의*배치*의 각 구성 요소에 대한 손실 함수의 도함수를 계산하려고합니다.여기서 (**우리의 의도는**) 미분 행렬을 계산하는 것이 아니라 배치에서 (**각 예에 대해 개별적으로 계산된 편도함수의 합**) 을 계산하는 것입니다.

```{.python .input}
# When we invoke `backward` on a vector-valued variable `y` (function of `x`),
# a new scalar variable is created by summing the elements in `y`. Then the
# gradient of that scalar variable with respect to `x` is computed
with autograd.record():
    y = x * x  # `y` is a vector
y.backward()
x.grad  # Equals to y = sum(x * x)
```

```{.python .input}
#@tab pytorch
# Invoking `backward` on a non-scalar requires passing in a `gradient` argument
# which specifies the gradient of the differentiated function w.r.t `self`.
# In our case, we simply want to sum the partial derivatives, so passing
# in a gradient of ones is appropriate
x.grad.zero_()
y = x * x
# y.backward(torch.ones(len(x))) equivalent to the below
y.sum().backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as `y = tf.reduce_sum(x * x)`
```

## 분리 계산

경우에 따라 [**기록된 계산 그래프 외부로 일부 계산을 이동**] 예를 들어 `y`이 `x`의 함수로 계산되었고 이후 `z`가 `y`과 `x`의 함수로 계산되었다고 가정해 보겠습니다.이제 `x`와 관련하여 `z`의 기울기를 계산하고 싶었지만 어떤 이유로 `y`을 상수로 취급하고 `y`이 계산 된 후 `x`가 수행 한 역할만 고려하기를 원한다고 상상해보십시오.

여기서 `y`를 분리하여 `y`와 동일한 값을 갖지만 계산 그래프에서 `y`가 어떻게 계산되었는지에 대한 정보는 버리는 새 변수 `u`을 반환할 수 있습니다.즉, 그라데이션은 `u`을 통해 `x`까지 뒤로 흐르지 않습니다.따라서, 다음의 역전파 함수는 `x`에 대한 `z = x * x * x`의 편미분 대신 `u`을 상수로 처리하면서 `x`에 대한 `z = u * x`의 편도함수를 계산한다.

```{.python .input}
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
#@tab tensorflow
# Set `persistent=True` to run `t.gradient` more than once
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

`y`의 계산이 기록되었으므로 이후 `y`에서 역전파를 호출하여 `x`에 대한 `y = x * x`의 도함수를 얻을 수 있습니다.

```{.python .input}
y.backward()
x.grad == 2 * x
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
#@tab tensorflow
t.gradient(y, x) == 2 * x
```

## 파이썬 제어 흐름의 기울기 계산하기

자동 미분을 사용하는 한 가지 이점은 (**파이썬 제어 흐름의 미로를 통과해야하는 함수**) (예: 조건문, 루프 및 임의의 함수 호출) 의 계산 그래프를 작성하는 [**심지어**], (**우리는 여전히 결과 변수의 기울기를 계산할 수 있습니다.**)다음 스니펫에서 `while` 루프의 반복 횟수와 `if` 문의 평가는 모두 입력 `a`의 값에 따라 달라집니다.

```{.python .input}
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

그라디언트를 계산해 보겠습니다.

```{.python .input}
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
#@tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
#@tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

이제 위에서 정의한 `f` 함수를 분석할 수 있습니다.입력 `a`에서 조각별 선형이라는 점에 유의하십시오.즉, `a`에 대해 `f(a) = k * a`와 같은 일정한 스칼라 `k`가 존재하며, 여기서 `k`의 값은 입력 `a`에 따라 달라집니다.결과적으로 `d / a`을 사용하면 그래디언트가 올바른지 확인할 수 있습니다.

```{.python .input}
a.grad == d / a
```

```{.python .input}
#@tab pytorch
a.grad == d / a
```

```{.python .input}
#@tab tensorflow
d_grad == d / a
```

## 요약

* 딥러닝 프레임워크는 도함수 계산을 자동화할 수 있습니다.이를 사용하기 위해 먼저 편도함수를 원하는 변수에 기울기를 붙입니다.그런 다음 목표값의 계산을 기록하고 역전파를 위한 함수를 실행한 다음 결과 기울기에 액세스합니다.

## 연습문제

1. 2차 도함수가 1차 도함수보다 계산 비용이 훨씬 더 비싼 이유는 무엇입니까?
1. 역전파를 위해 함수를 실행한 후 즉시 다시 실행하여 어떤 일이 발생하는지 확인합니다.
1. `a`에 대해 `d`의 도함수를 계산하는 제어 흐름 예제에서 변수 `a`를 랜덤 벡터 또는 행렬로 변경하면 어떻게 될까요?이 시점에서 `f(a)` 계산의 결과는 더 이상 스칼라가 아닙니다.결과는 어떻게 되나요?어떻게 분석할까요?
1. 제어 흐름의 기울기를 찾는 예제를 다시 디자인합니다.결과를 실행하고 분석합니다.
1. $f(x) = \sin(x)$을 보자.$f(x)$ 및 $\frac{df(x)}{dx}$를 플로팅합니다. 여기서 후자는 $f'(x) = \cos(x)$을 악용하지 않고 계산됩니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/200)
:end_tab:
