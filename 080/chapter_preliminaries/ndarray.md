# Data Manipulation

# 데이터 처리

:label:`sec_ndarray`

In order to get anything done, we need some way to store and manipulate data. Generally, there are two important things we need to do with data: (i) acquire them; and (ii) process them once they are inside the computer.  There is no point in acquiring data without some way to store it, so let us get our hands dirty first by playing with synthetic data.  To start, we introduce the $n$-dimensional array, which is also called the *tensor*.

어떤 작업에서나 데이터를 저장하고 처리할 방법이 필요합니다. 일반적으로, 두 가지 데이터를 사용하는 중요한 작업이 있습니다. (i) 데이터를 수집하고 (ii) 컴퓨터 내부에서 처리하는 것입니다. 데이터를 저장할 방법을 모르고 데이터를 수집하는 것은 아무 의미가 없을 것입니다. 먼저 합성 데이터를 다루면서 익숙해지도록 하겠습니다. 시작하면서, (*텐서*라고도 부르는) $n$-차원 배열을 소개합니다.

If you have worked with NumPy, the most widely-used scientific computing package in Python, then you will find this section familiar. No matter which framework you use, its *tensor class* (`ndarray` in MXNet, `Tensor` in both PyTorch and TensorFlow) is similar to NumPy's `ndarray` with a few killer features. First, GPU is well-supported to accelerate the computation whereas NumPy only supports CPU computation. Second, the tensor class supports automatic differentiation. These properties make the tensor class suitable for deep learning. Throughout the book, when we say tensors, we are referring to instances of the tensor class unless otherwise stated.

파이썬에서 가장 널리 사용되는 과학 연산 패키지 인 NumPy를 써본 적이 있다면, 이번 섹션이 익숙할 것입니다. 어떤 프레임 워크를 사용하든 상관 없이, *텐서 클래스* (MXNet의 `ndarray`, PyTorch와 텐서 플로우의 `Tensor`)는 NumPy의 `ndarray` 에 몇 가지 킬러 기능을 덧붙인 것과 비슷합니다. 첫째, NumPy가 CPU 연산만을 지원하는 반면, 텐서 클래스는 빠른 연산을 위해 GPU를 잘 지원합니다. 둘째, 텐서 클래스는 자동 미분을 지원합니다. 이러한 특성으로 인해 텐서 클래스는 딥러닝에 적합하다고 할 수 있습니다. 이 책 전체에서, 텐서라고 말할 때 달리 명시하지 않는 한 텐서 클래스의 인스턴스를 언급하는 것입니다.



## Getting Started

## 시작하며

In this section, we aim to get you up and running, equipping you with the basic math and numerical computing tools that you will build on as you progress through the book. Do not worry if you struggle to grok some of the mathematical concepts or library functions. The following sections will revisit this material in the context of practical examples and it will sink. On the other hand, if you already have some background and want to go deeper into the mathematical content, just skip this section.

이번 섹션에서는, 이 책을 진행하면서 사용할 기본 수학 및 수치연산 도구를 시작해 보는 것을 목표로 합니다. 이해가 어려운 수학 개념이나 라이브러리 함수가 있더라도 걱정하지 마십시오. 뒤에 배울 섹션에서 실제 예제를 통해 다시 살펴보면 이해될 것입니다. 한편, 이미 배경 지식이 있어서 수학적 내용을 더 깊이 알고 싶다면, 이 섹션을 건너 뛰십시오.



:begin_tab:`mxnet`
To start, we import the `np` (`numpy`) and `npx` (`numpy_extension`) modules from MXNet. Here, the `np` module includes functions supported by NumPy, while the `npx` module contains a set of extensions developed to empower deep learning within a NumPy-like environment. When using tensors, we almost always invoke the `set_np` function: this is for compatibility of tensor processing by other components of MXNet.

시작하기 위해 MXNet에서 `np` (`numpy`) 와 `npx` (`numpy_extension`) 모듈을 임포트합니다. `np` 모듈에는 NumPy가 지원하는 함수가 포함되고, `npx` 모듈에는 NumPy 환경에서 딥러닝을 지원하기 위한 확장 기능이 포함되어 있습니다. 텐서를 사용할 때, 우리는 거의 항상 `set_np` 함수를 호출합니다. 이것은 다른 MXNet 구성 요소들의 텐서 처리 호환성을 위한 것입니다.
:end_tab:

:begin_tab:`pytorch`
To start, we import `torch`. Note that though it's called PyTorch, we should
import `torch` instead of `pytorch`.

시작하려면 `torch` 를 임포트합니다. PyTorch라고 부르지만, `pytorch` 가 아니라 `torch` 를 임포트하는 것에 주의하세요.
:end_tab:

:begin_tab:`tensorflow`
To start, we import `tensorflow`. As the name is a little long, we often import
it with a short alias `tf`. 
시작하려면  `tensorflow` 를 임포트합니다. 이름이 조금 길기 때문에 짧은 별칭 `tf` 로 임포트하는 경우가 많습니다.
:end_tab:

```{.python .input}
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
```

A tensor represents a (possibly multi-dimensional) array of numerical values. With one axis, a tensor corresponds (in math) to a *vector*. With two axes, a tensor corresponds to a *matrix*. Tensors with more than two axes do not have special mathematical names.

텐서는 숫자 값의 (다차원) 배열을 나타냅니다. 하나의 축에서 텐서는 (수학의) *벡터*에 해당합니다. 축이 두 개인 경우 텐서는 *행렬*에 해당합니다. 축이 두 개 이상인 텐서에는 특별한 수학적 이름이 없습니다.



To start, we can use `arange` to create a row vector `x` containing the first 12 integers starting with 0, though they are created as floats by default. Each of the values in a tensor is called an *element* of the tensor. For instance, there are 12 elements in the tensor `x`. Unless otherwise specified, a new tensor will be stored in main memory and designated for CPU-based computation.

시작하기 위해, `arange` 를 사용해 0으로 시작하는 정수(실수형으로 만들어지지만) 12개를 포함하는 행 벡터 `x` 를 만들 수 있습니다. 텐서의 각 값을 텐서의 *요소*라고 합니다. 예를 들어, 텐서 `x` 에는 12 개의 요소가 있습니다. 달리 지정하지 않으면 새 텐서는 주 메모리에 저장되고 CPU로 계산됩니다.

```{.python .input}
x = np.arange(12)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(12)
x
```

```{.python .input}
#@tab tensorflow
x = tf.constant(range(12))
x
```

We can access a tensor's *shape* (the length along each axis) by inspecting its `shape` property.

`shape` 속성을 확인하면 텐서의 *shape* (각 축 방향의 길이)에 액세스 할 수 있습니다.

```{.python .input}
#@tab all
x.shape
```

If we just want to know the total number of elements in a tensor, i.e., the product of all of the shape elements, we can inspect its size. Because we are dealing with a vector here, the single element of its `shape` is identical to its size.

텐서의 전체 요소 개수(즉, 모든 shape 요소의 곱)를 알고 싶다면 크기를 확인하면 됩니다. 벡터를 다룬다면 `shape`가 크기와 동일합니다.

```{.python .input}
x.size
```

```{.python .input}
#@tab pytorch
x.numel()
```

```{.python .input}
#@tab tensorflow
tf.size(x)
```

To change the shape of a tensor without altering either the number of elements or their values, we can invoke the `reshape` function. For example, we can transform our tensor, `x`, from a row vector with shape (12,) to a matrix with shape (3, 4). This new tensor contains the exact same values, but views them as a matrix organized as 3 rows and 4 columns. To reiterate, although the shape has changed, the elements in `x` have not. Note that the size is unaltered by reshaping.

요소의 수나 값을 변경하지 않고 텐서의 모양을 변경하려면 `reshape` 함수를 호출하면 됩니다. 예를 들어, (12,) shape의 행 벡터에서 (3, 4) shape의 행렬로 텐서 `x` 를 변환 할 수 있습니다. 이 새로운 텐서는 완전히 같은 값들을 가지고 있지만, 3개의 행과 4개의 열을 가진 행렬로 간주됩니다. 반복하자면, 모양이 바뀌었어도 `x` 의 요소는 바뀌지 않았습니다. reshape 해도 크기는 변경되지 않습니다.

```{.python .input}
#@tab mxnet, pytorch
x = x.reshape(3, 4)
x
```

```{.python .input}
#@tab tensorflow
x = tf.reshape(x, (3, 4))
x
```

Reshaping by manually specifying every dimension is unnecessary. If our target shape is a matrix with shape (height, width), then after we know the width, the height is given implicitly. Why should we have to perform the division ourselves? In the example above, to get a matrix with 3 rows, we specified both that it should have 3 rows and 4 columns. Fortunately, tensors can automatically work out one dimension given the rest. We invoke this capability by placing `-1` for the dimension that we would like tensors to automatically infer. In our case, instead of calling `x.reshape(3, 4)`, we could have equivalently called `x.reshape(-1, 4)` or `x.reshape(3, -1)`.

Reshape할 때 꼭 수동으로 모든 차원을 지정할 필요는 없습니다. 타겟 shape이 (height, width)인 행렬이라면, width만 알게되면 height는 암시적으로 정해집니다. 굳이 나눗셈을 할 필요가 없습니다. 위의 예에서, 3개의 행이 있는 행렬을 얻기 위해 3개의 행과 4개의 열이라고 지정했었습니다. 다행히도 텐서는 (나머지 차원들을 고려해서) 자동으로 마지막 한 차원을 결정할 수 있습니다. 이 기능을 사용하려면 텐서가 자동으로 알아내길 원하는 차원에 `-1` 이라고 지정합니다. 위의 예에서는, `x.reshape(3, 4)` 를 호출하는 대신 `x.reshape(-1, 4)` 또는 `x.reshape(3, -1)` 라고 호출할 수 있습니다.

Typically, we will want our matrices initialized either with zeros, ones, some other constants, or numbers randomly sampled from a specific distribution. We can create a tensor representing a tensor with all elements set to 0 and a shape of (2, 3, 4) as follows:

일반적으로 행렬은 0, 1, 다른 어떤 상수 또는 특정 분포에서 무작위로 샘플링 된  숫자로 초기화됩니다. 우리는 다음과 같이 모든 요소가 0이고 (2, 3, 4) shape을 가진 텐서를 만들 수 있습니다:

```{.python .input}
np.zeros((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.zeros(2, 3, 4)
```

```{.python .input}
#@tab tensorflow
tf.zeros((2, 3, 4))
```

Similarly, we can create tensors with each element set to 1 as follows:

마찬가지로 각 요소가 1로 설정된 텐서를 다음과 같이 만들 수 있습니다.

```{.python .input}
np.ones((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.ones((2, 3, 4))
```

Often, we want to randomly sample the values for each element in a tensor from some probability distribution. For example, when we construct arrays to serve as parameters in a neural network, we will typically initialize their values randomly. The following snippet creates a tensor with shape (3, 4). Each of its elements is randomly sampled from a standard Gaussian (normal) distribution with a mean of 0 and a standard deviation of 1.

종종 텐서의 각 요소들의 값을 특정 확률 분포에서 무작위로 샘플링할 필요가 있습니다. 예를 들어 신경망의 파라미터로 사용할 배열을 만들 때, 랜덤한 값으로 초기화하는 것이 일반적입니다. 다음 코드는 (3, 4) shape의 텐서를 만듭니다. 각 요소는 평균이 0이고 표준편차가 1인 표준정규분포에서 무작위로 샘플링됩니다.

```{.python .input}
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
#@tab pytorch
torch.randn(3, 4)
```

```{.python .input}
#@tab tensorflow
tf.random.normal(shape=[3, 4])
```

We can also specify the exact values for each element in the desired tensor by supplying a Python list (or list of lists) containing the numerical values. Here, the outermost list corresponds to axis 0, and the inner list to axis 1.

Python의 list (또는 list의 list)를 사용해서 텐서의 각 요소에 정확한 값을 지정할 수도 있습니다. 이 때, 가장 바깥쪽 list가 axis 0에 해당하고, 안쪽 list가 axis 1에 해당합니다.

```{.python .input}
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```



## Operations

## 연산

This book is not about software engineering.
Our interests are not limited to simply reading and writing data from/to arrays.
We want to perform mathematical operations on those arrays.
Some of the simplest and most useful operations are the *elementwise* operations.
These apply a standard scalar operation to each element of an array.
For functions that take two arrays as inputs, elementwise operations apply some standard binary operator on each pair of corresponding elements from the two arrays.
We can create an elementwise function from any function that maps from a scalar to a scalar.

이 책은 소프트웨어 공학에 관한 것이 아닙니다. 우리의 관심사는 단순히 배열에서 데이터를 읽고 쓰는 것만이 아니죠. 이 배열들에 수학 연산을 수행할 수 있어야합니다. 가장 단순하고 유용한 것으로 *elementwise (요소별)* 연산이 있습니다. 이들은 배열의 각 요소에 표준 스칼라 연산을 적용합니다. 두 개의 배열을 입력으로 받는 함수에 대해서 생각해 보면, 요소별 연산은 두 배열의 매칭하는 각 요소 쌍에 대해 표준 이항 연산을 적용합니다. 스칼라에서 스칼라로 매핑되는 모든 함수로 요소별 함수를 만들 수 있습니다.



In mathematical notation, we would denote such
a *unary* scalar operator (taking one input)
by the signature $f: \mathbb{R} \rightarrow \mathbb{R}$.
This just means that the function is mapping
from any real number ($\mathbb{R}$) onto another.
Likewise, we denote a *binary* scalar operator
(taking two real inputs, and yielding one output)
by the signature $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$.
Given any two vectors $\mathbf{u}$ and $\mathbf{v}$ *of the same shape*,
and a binary operator $f$, we can produce a vector
$\mathbf{c} = F(\mathbf{u},\mathbf{v})$
by setting $c_i \gets f(u_i, v_i)$ for all $i$,
where $c_i, u_i$, and $v_i$ are the $i^\mathrm{th}$ elements
of vectors $\mathbf{c}, \mathbf{u}$, and $\mathbf{v}$.
Here, we produced the vector-valued
$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$
by *lifting* the scalar function to an elementwise vector operation.

수학적 표기법에서 *단항* (하나의 입력을 받는) 스칼라 연산자는 다음과 같이 표시됩니다:
$f: \mathbb{R} \rightarrow \mathbb{R}$.
이것은 함수가 어떤 실수($\mathbb{R}$)가 다른 실수로 매핑된다는 것을 의미합니다.
마찬가지로 (두 개의 실수 입력을 받아 하나의 출력을 생성하는) *이항* 스칼라 연산자는 다음과 같이 표시됩니다:
$f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$.
*동일한 shape을 가지는* 두 벡터 $\mathbf{u}$, $\mathbf{v}$ 와 이항 연산자 $f$ 로부터, 벡터 $\mathbf{c} = F(\mathbf{u},\mathbf{v})$는 다음과 같이 계산할 수 있습니다:
벡터 $\mathbf{c}, \mathbf{u}$, $\mathbf{v}$의 $i^\mathrm{th}$ 요소 $c_i, u_i$, $v_i$에 대하여, 모든 $i$에 대해 $c_i \gets f(u_i, v_i)$ 를 계산.
여기서 벡터 값을 가지는 $F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$를 계산하기 위해, 스칼라 함수를 요소별 벡터 연산에 적용했습니다.

The common standard arithmetic operators (`+`, `-`, `*`, `/`, and `**`) have all been *lifted* to elementwise operations for any identically-shaped tensors of arbitrary shape. We can call elementwise operations on any two tensors of the same shape. In the following example, we use commas to formulate a 5-element tuple, where each element is the result of an elementwise operation.

일반적인 표준 산술 연산자(`+`, `-`, `*`, `/`, `**`)는 임의의 동일한 shape의 텐서에 대해 요소별 연산으로 적용됩니다. 동일한 shape을 가지는 임의의 두 텐서에 대해 요소별 연산을 호출할 수 있습니다. 다음은 쉼표를 사용해 요소 5개의 튜플을 수식화하는 예제입니다. 여기서 각 요소는 요소별 연산의 결과입니다.

```{.python .input}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

Many more operations can be applied elementwise, including unary operators like exponentiation.

지수와 같은 단항 연산자를 포함해서, 그 밖의 많은 연산을 요소 단위로 적용할 수 있습니다.

```{.python .input}
np.exp(x)
```

```{.python .input}
#@tab pytorch
torch.exp(x)
```

```{.python .input}
#@tab tensorflow
tf.exp(x)
```

In addition to elementwise computations, we can also perform linear algebra operations, including vector dot products and matrix multiplication. We will explain the crucial bits of linear algebra (with no assumed prior knowledge) in :numref:`sec_linear-algebra`.

요소별 계산 외에도 벡터 내적, 행렬 곱셈을 비롯한 선형 대수 연산을 수행할 수 있습니다. :numref:`sec_linear-algebra`에서 선형 대수의 핵심 부분을 설명하겠습니다(사전 지식 없음 가정).

We can also *concatenate* multiple tensors together, stacking them end-to-end to form a larger tensor. We just need to provide a list of tensors and tell the system along which axis to concatenate. The example below shows what happens when we concatenate two matrices along rows (axis 0, the first element of the shape) vs. columns (axis 1, the second element of the shape). We can see that the first output tensor's axis-0 length ($6$) is the sum of the two input tensors' axis-0 lengths ($3 + 3$); while the second output tensor's axis-1 length ($8$) is the sum of the two input tensors' axis-1 lengths ($4 + 4$).

또한 여러 텐서를 함께 연결하여 더 큰 텐서를 형성 할 수 있습니다. 텐서 목록을 제공하고 연결할 축을 시스템에 알려 주면 됩니다. 아래 예제는 행(axis 0, shape의 첫 번째 요소)과 열 (axis 1, shape의 두 번째 요소)을 따라 두 개의 행렬을 연결하는 과정을 보여줍니다. 첫 번째 출력 텐서의 axis-0 길이($6$)가 두 입력 텐서의 axis-0 길이($3 + 3$)의 합임을 알 수 있습니다. 두 번째 출력 텐서의 axis-1 길이($8$)는 두 입력 텐서의 axis-1 길이($4 + 4$)의 합입니다.

```{.python .input}
x = np.arange(12).reshape(3, 4)
y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([x, y], axis=0), np.concatenate([x, y], axis=1)
```

```{.python .input}
#@tab pytorch
x = torch.arange(12, dtype=torch.float32).reshape((3,4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((x, y), dim=0), torch.cat((x, y), dim=1)
```

```{.python .input}
#@tab tensorflow
x = tf.constant(range(12), dtype=tf.float32, shape=(3, 4))
y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([x, y], axis=0), tf.concat([x, y], axis=1)
```

Sometimes, we want to construct a binary tensor via *logical statements*. Take `x == y` as an example. For each position, if `x` and `y` are equal at that position, the corresponding entry in the new tensor takes a value of 1, meaning that the logical statement `x == y` is true at that position; otherwise that position takes 0.

*논리문*을 사용해 이진 텐서를 구성할 수도 있습니다. `x == y`를 예로 들어 보겠습니다. 각 위치마다 `x`와 `y`가 해당 위치에서 같으면 새 텐서의 해당 항목 값은 1이 됩니다. 즉, 논리문 `x == y'는 해당 위치에서 true 입니다. 그렇지 않으면 그 위치의 값은 0이 됩니다.

```{.python .input}
#@tab all
x == y
```

Summing all the elements in the tensor yields a tensor with only one element.

텐서의 모든 요소를 합하면 요소가 하나 뿐인 텐서가 만들어집니다.

```{.python .input}
#@tab mxnet, pytorch
x.sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(x)
```



## Broadcasting Mechanism

## 브로드캐스팅 메커니즘

:label:`subsec_broadcasting`

In the above section, we saw how to perform elementwise operations on two tensors of the same shape. Under certain conditions, even when shapes differ, we can still perform elementwise operations by invoking the *broadcasting mechanism*. This mechanism works in the following way: First, expand one or both arrays by copying elements appropriately so that after this transformation, the two tensors have the same shape. Second, carry out the elementwise operations on the resulting arrays.

위 섹션에서, 같은 shape의 두 텐서에서 요소별 연산을 수행하는 방법을 보았습니다. 특정 조건에서 shape이 다른 경우에도 *브로드캐스팅 메커니즘*을 사용해 요소별 연산을 수행할 수 있습니다. 이 메커니즘은 다음과 같은 방식으로 작동합니다. 먼저, 두 텐서의 모양이 동일해지도록 요소를 적절히 복사하여 (한 개 또는 두 개의) 배열을 확장합니다. 둘째, 결과 배열에서 요소별 연산을 수행합니다.

In most cases, we broadcast along an axis where an array initially only has length 1, such as in the following example:

대부분의 경우, 다음 예제와 같이 배열의 길이가 1인 축을 따라 브로드캐스트합니다.

```{.python .input}
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
#@tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
#@tab tensorflow
a = tf.constant(range(3), shape=(3, 1))
b = tf.constant(range(2), shape=(1, 2))
a, b
```

Since `a` and `b` are $3\times1$ and $1\times2$ matrices respectively, their shapes do not match up if we want to add them. We *broadcast* the entries of both matrices into a larger $3\times2$ matrix as follows: for matrix `a` it replicates the columns and for matrix `b` it replicates the rows before adding up both elementwise.

`a`와 `b`는 각각 $3\times1$ 과 $1\times2$ 행렬이므로, 서로 더하려면 그 shape이 일치하지 않습니다. 두 행렬의 요소들을 아래와 같이 더 큰 $3\times2$ 행렬로 *브로드캐스트* 합니다. 요소별 덧셈을 하기 전에, 행렬 `a` 는 열을 복제하고 행렬 `b` 는 행을 복제합니다.

```{.python .input}
#@tab all
a + b
```



## Indexing and Slicing

## 인덱싱과 슬라이싱

Just as in any other Python array, elements in a tensor can be accessed by index. As in any Python array, the first element has index 0 and ranges are specified to include the first but *before* the last element. As in standard Python lists, we can access elements according to their relative position to the end of the list by using negative indices.

다른 파이썬 배열과 마찬가지로 텐서의 요소는 인덱스로 액세스 할 수 있습니다. 첫 번째 요소는 인덱스 값이 0이며, 범위를 지정할 때 첫 번째 요소는 포함되지만 마지막 요소는 *포함되지 않습니다*. 파이썬 리스트와 마찬가지로, 음수 인덱스를 사용해서 리스트 끝 부분부터의 상대적 위치를 액세스할 수 있습니다.

Thus, `[-1]` selects the last element and `[1:3]` selects the second and the third elements as follows:

다음과 같이 `[-1] ` 은 마지막 요소를 선택하고 `[1:3]` 은 두 번째와 세 번째 요소를 선택합니다.

```{.python .input}
#@tab all
x[-1], x[1:3]
```

Beyond reading, we can also write elements of a matrix by specifying indices.

읽는 것 뿐만 아니라, 인덱스를 지정해서 행렬의 요소에 값을 쓸 수도 있습니다.

```{.python .input}
#@tab mxnet, pytorch
x[1, 2] = 9
x
```

```{.python .input}
#@tab tensorflow
x = tf.convert_to_tensor(tf.Variable(x)[1, 2].assign(9))
x
```

If we want to assign multiple elements the same value, we simply index all of them and then assign them the value. For instance, `[0:2, :]` accesses the first and second rows, where `:` takes all the elements along axis 1 (column). While we discussed indexing for matrices, this obviously also works for vectors and for tensors of more than 2 dimensions.

같은 값을 여러 개의 요소에 한꺼번에 쓰려면, 인덱스를 만들어서 값을 지정하면 됩니다. 예를 들어, `[0:2, :]` 는 첫 번째와 두 번째 행을 액세스합니다. 여기서 `:`는 axis-1 (열)을 따라 모든 요소를 가져옵니다. 지금까지는 행렬의 인덱싱을 설명했지만, 이 내용은 벡터와 2차원 이상의 텐서에도 적용됩니다.

```{.python .input}
#@tab mxnet, pytorch
x[0:2, :] = 12
x
```

```{.python .input}
#@tab tensorflow
x_var = tf.Variable(x)
x_var[1:2,:].assign(tf.ones(x_var[1:2,:].shape, dtype = tf.float32)*12)
x = tf.convert_to_tensor(x_var)
x
```



## Saving Memory

## 메모리 절약

Running operations can cause new memory to be allocated to host results. For example, if we write `y = x + y`, we will dereference the tensor that `y` used to point to and instead point `y` at the newly allocated memory. In the following example, we demonstrate this with Python's `id()` function, which gives us the exact address of the referenced object in memory. After running `y = y + x`, we will find that `id(y)` points to a different location. That is because Python first evaluates `y + x`, allocating new memory for the result and then makes `y` point to this new location in memory.

연산을 실행하면 결과를 저장하기 위해 새로운 메모리가 할당될 수 있습니다. 예를 들어, `y = x + y` 라는 코드는 텐서 `y` 가 가리키던 주소에서 값을 읽고 난 다음, `y` 로 새로 할당된 메모리 주소를 가리키게 됩니다. 다음 예에서, 파이썬의 `id()` 함수(참조된 객체의 정확한 메모리 주소를 제공하는)를 통해 이를 확인할 수 있습니다. `y = y + x`를 실행하면 `id(y)`가 다른 위치를 가리킵니다. 파이썬이 먼저 `y + x` 를 계산하고, 결과 값에 새로운 메모리를 할당 한 다음, `y` 가 새로운 메모리의 위치를 가리키도록 하기 때문입니다.

```{.python .input}
#@tab all
before = id(y)
y = y + x
id(y) == before
```

This might be undesirable for two reasons. First, we do not want to run around allocating memory unnecessarily all the time. In machine learning, we might have hundreds of megabytes of parameters and update all of them multiple times per second. Typically, we will want to perform these updates *in place*. Second, we might point at the same parameters from multiple variables. If we do not update in place, other references will still point to the old memory location, making it possible for parts of our code to inadvertently reference stale parameters.

이것은 두 가지 이유로 바람직하지 않을 수 있습니다. 첫째, 우리는 불필요하게 메모리를 항상 할당하고 싶지 않습니다. 머신러닝에서는 수백 메가바이트의 매개 변수 전체를 초당 여러 번 업데이트할 수도 있습니다. 일반적으로, 이러한 업데이트는 (새로운 메모리를 할당하지 않고) *같은 위치에서* 일어나길 원할 것입니다. 둘째, 여러 변수에서 동일한 매개 변수를 가리킬 수 있습니다. 같은 위치에서 업데이트하지 않으면, 여전히 이전 메모리 위치를 가리키는 참조들이 실수로 잘못된 매개 변수를 참조하게 할 수 있습니다.

Fortunately, performing in-place operations in MXNet is easy. We can assign the result of an operation to a previously allocated array with slice notation, e.g., `y[:] = <expression>`. To illustrate this concept, we first create a new matrix `z` with the same shape as another `y`, using `zeros_like` to allocate a block of $0$ entries.

다행히도 같은 위치에서 업데이트하는 연산을 하는 것은 쉽습니다. 예를 들면, `y[:] = <expression>` 과 같이 슬라이스 표기법을 사용해 이전에 할당된 배열에 연산 결과를 저장할 수 있습니다. 이 개념을 설명하기 위해, 먼저 행렬 `y` 와 같은 모양을 가진 새로운 행렬 `z` 를 만듭니다. 이 때 0 값으로 초기화된 블록을 할당하기 위해 `zeros_like` 을 사용했습니다.

```{.python .input}
z = np.zeros_like(y)
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

```{.python .input}
#@tab pytorch
z = torch.zeros_like(y)
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

```{.python .input}
#@tab tensorflow
z = tf.Variable(tf.zeros_like(y))
print('id(z):', id(z))
z[:].assign(x + y)
print('id(z):', id(z))
```

If the value of `x` is not reused in subsequent computations, we can also use `x[:] = x + y` or `x += y` to reduce the memory overhead of the operation.

`x` 값이 뒤의 계산에서 재사용되지 않는다면, `x[:] = x + y` 또는 `x += y` 를 사용해 연산의 메모리 오버 헤드를 줄일 수 있습니다.

```{.python .input}
#@tab mxnet, pytorch
before = id(x)
x += y
id(x) == before
```

```{.python .input}
#@tab tensorflow
before = id(x)
tf.Variable(x).assign(x + y)
id(x) == before
```



## Conversion to Other Python Objects

## 다른 파이썬 객체로 변환

Converting to a NumPy tensor, or vice versa, is easy. The converted result does not share memory. This minor inconvenience is actually quite important: when you perform operations on the CPU or on GPUs, you do not want to halt computation, waiting to see whether the NumPy package of Python might want to be doing something else with the same chunk of memory.

NumPy 텐서로 또는 그 반대로 쉽게 변환할 수 있습니다. 변환된 결과는 메모리를 공유하지 않습니다. 이 사소한 불편함은 실제로 매우 중요합니다. CPU나 GPU에서 작업을 수행할 때, 파이썬의 NumPy 패키지가 같은 메모리 블럭에 다른 작업을 수행할지 확인하느라 계산을 중단하고 싶지 않기 때문입니다.

```{.python .input}
a = x.asnumpy()
b = np.array(a)
type(a), type(b)
```

```{.python .input}
#@tab pytorch
a = x.numpy()
b = torch.tensor(a)
type(a), type(b)
```

```{.python .input}
#@tab tensorflow
a = x.numpy()
b = tf.constant(a)
type(a), type(b)
```

To convert a size-1 tensor to a Python scalar, we can invoke the `item` function or Python's built-in functions.

크기가 1인 텐서를 파이썬 스칼라로 변환하기 위해, `item` 함수 또는 파이썬의 내장 함수를 호출할 수 있습니다.

```{.python .input}
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```



## The `d2l` Package

## `d2l` 패키지

Throughout the online version of this book, we will provide implementations of multiple frameworks. However, different frameworks may be different in their API names or usage. To better reuse the same code block across multiple frameworks, we unify a few commonly-used functions in the `d2l` package. The comment `#@save` is a special mark where the following function, class, or statements are saved in the `d2l` package. For instance, later we can directly invoke `d2l.numpy(a)` to convert a tensor `a`, which can be defined in any supported framework, into a NumPy tensor.

이 책의 온라인 버전에서는 다양한 프레임워크의 구현을 제공할 것입니다. 하지만 개별 프레임워크마다 API 이름이나 사용법이 다를 수 있습니다. 여러 프레임 워크에서 동일한 코드 블록을 더 잘 재사용하기 위해, 몇 가지 일반적인 기능을 `d2l` 패키지에 통합했습니다. 주석 `#@save` 은 다음의 함수, 클래스, 명령문이 `d2l` 패키지에 포함된다는 특별한 표시입니다. 예를 들어, 나중에 (지원되는 모든 프레임 워크에서 정의된) `d2l.numpy(a)` 를 직접 호출하여 텐서 `a` 를 NumPy 텐서로 변환 할 수 있습니다.

```{.python .input}
#@save
numpy = lambda a: a.asnumpy()
size = lambda a: a.size
reshape = lambda a, *args: a.reshape(*args)
ones = np.ones
zeros = np.zeros
```

```{.python .input}
#@tab pytorch
#@save
numpy = lambda a: a.detach().numpy()
size = lambda a: a.numel()
reshape = lambda a, *args: a.reshape(*args)
ones = torch.ones
zeros = torch.zeros
```

```{.python .input}
#@tab tensorflow
#@save
numpy = lambda a: a.numpy()
size = lambda a: tf.size(a).numpy()
reshape = tf.reshape
ones = tf.ones
zeros = tf.zeros
```

In the rest of the book, we often define more complicated functions or classes. For those that can be used later, we will also save them in the `d2l` package so later they can be directly invoked without being redefined.

이 책의 나머지 부분에서는 종종 더 복잡한 함수나 클래스를 정의합니다. 그중 나중에 사용할 수 있는 것들은 `d2l` 패키지에 저장되므로, 다시 정의하지 않고 직접 호출할 수 있습니다.



## Summary


## 요약

* The main interface to store and manipulate data for deep learning is the tensor ($n$-dimensional array). It provides a variety of functionalities including basic mathematics operations, broadcasting, indexing, slicing, memory saving, and conversion to other Python objects.
  딥러닝에서 데이터를 저장하고 조작하는 기본 인터페이스는 텐서($n$ 차원 배열)입니다. 텐서는 기본 수학 연산, 브로드캐스트, 인덱싱, 슬라이싱, 메모리 절약 및 다른 Python 객체로의 변환을 포함한 다양한 기능을 제공합니다.



## Exercises


## 연습 문제

1. Run the code in this section. Change the conditional statement `x == y` in this section to `x < y` or `x > y`, and then see what kind of tensor you can get. 이 섹션의 코드를 실행합니다. 
   이 섹션의 조건문 `x == y` 를 `x < y` 또는 `x > y` 로 변경한 다음, 어떤 종류의 텐서를 얻을 수 있는지 확인하십시오.
1. Replace the two tensors that operate by element in the broadcasting mechanism with other shapes, e.g., 3-dimensional tensors. Is the result the same as expected? 
   브로드캐스트 메커니즘의 요소별 연산에서, 두 개의 텐서를 다른 모양(예: 3차원 텐서)으로 바꿔보십시오. 결과가 예상했던 것과 같습니까?



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/187)
:end_tab:
